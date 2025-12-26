"""
Orchestrator - main control loop for the Copilot-Gemini feedback cycle.

M6: Production-ready with M5 hardening integration:
- Atomic checkpointing for crash recovery
- Circuit breakers and retry policies for resilience
- Metrics collection for observability
- UI desync detection and recovery
- Enhanced kill switch with preemptive checks
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, List, Any
from dataclasses import dataclass

from copilot_agent.config import AgentConfig
from copilot_agent.state import StateManager, SessionPhase, GeminiVerdict
from copilot_agent.safety.killswitch import (
    KillSwitch, KillSwitchTriggered,
    async_wait_with_killswitch,
)
from copilot_agent.reviewer.gemini import GeminiReviewer, ReviewResult, ReviewVerdict
from copilot_agent.perception.pipeline import PerceptionPipeline, CaptureResult, CaptureMethod
from copilot_agent.actuator.actions import ActionExecutor
from copilot_agent.logging import get_logger

# M5 modules
from copilot_agent.checkpoint import (
    AtomicCheckpointer, StepType, CheckpointState,
)
from copilot_agent.resilience import (
    CircuitBreaker, RetryPolicy, CircuitOpenError, RetryExhaustedError,
    retry_with_backoff,
    create_reviewer_circuit, create_vision_circuit, create_ui_circuit,
    REVIEWER_RETRY_POLICY, VISION_RETRY_POLICY, UI_ACTION_RETRY_POLICY,
)
from copilot_agent.metrics import (
    MetricsCollector, SessionMetrics, MetricType,
)
from copilot_agent.desync import (
    DesyncDetector, RecoveryManager, ParseFailureTracker,
    DesyncEvent, RecoveryAction,
)

logger = get_logger(__name__)


@dataclass
class IterationResult:
    """Result of a single iteration."""
    
    success: bool
    iteration: int
    verdict: Optional[ReviewVerdict] = None
    feedback: Optional[str] = None
    captured_response: Optional[str] = None
    should_continue: bool = True
    stop_reason: Optional[str] = None
    duration_ms: int = 0


@dataclass
class OrchestratorBudget:
    """Budget limits for orchestrator operations.
    
    Enforces resource limits to prevent runaway sessions and quota exhaustion.
    """
    
    max_reviewer_calls: int = 50
    max_vision_calls: int = 100
    max_ui_actions: int = 200
    max_runtime_seconds: int = 1800  # 30 minutes
    
    # Warning thresholds (percentage)
    warning_threshold: float = 0.8
    
    # Current usage
    reviewer_calls: int = 0
    vision_calls: int = 0
    ui_actions: int = 0
    start_time: Optional[float] = None
    
    @classmethod
    def from_config(cls, config: AgentConfig) -> "OrchestratorBudget":
        """Create budget from config."""
        return cls(
            max_reviewer_calls=config.reviewer.max_reviews_per_session,
            max_vision_calls=config.perception.max_vision_per_session,
            max_ui_actions=config.automation.max_iterations * 20,  # ~20 actions per iteration
            max_runtime_seconds=config.automation.max_runtime_minutes * 60,
        )
    
    def start(self) -> None:
        """Start budget tracking."""
        import time
        self.start_time = time.time()
    
    def check_reviewer(self) -> bool:
        """Check if reviewer budget is available."""
        return self.reviewer_calls < self.max_reviewer_calls
    
    def check_vision(self) -> bool:
        """Check if vision budget is available."""
        return self.vision_calls < self.max_vision_calls
    
    def check_ui(self) -> bool:
        """Check if UI action budget is available."""
        return self.ui_actions < self.max_ui_actions
    
    def check_runtime(self) -> bool:
        """Check if runtime budget is available."""
        if self.start_time is None:
            return True
        import time
        elapsed = time.time() - self.start_time
        return elapsed < self.max_runtime_seconds
    
    def check_all(self) -> tuple[bool, Optional[str]]:
        """Check all budgets and return (ok, exhausted_resource)."""
        if not self.check_runtime():
            return False, "runtime"
        if not self.check_reviewer():
            return False, "reviewer_calls"
        if not self.check_vision():
            return False, "vision_calls"
        if not self.check_ui():
            return False, "ui_actions"
        return True, None
    
    def get_warnings(self) -> List[tuple[str, int, int]]:
        """Get list of resources approaching limit: (name, used, max)."""
        warnings = []
        
        if self.reviewer_calls >= self.max_reviewer_calls * self.warning_threshold:
            warnings.append(("reviewer_calls", self.reviewer_calls, self.max_reviewer_calls))
        
        if self.vision_calls >= self.max_vision_calls * self.warning_threshold:
            warnings.append(("vision_calls", self.vision_calls, self.max_vision_calls))
        
        if self.ui_actions >= self.max_ui_actions * self.warning_threshold:
            warnings.append(("ui_actions", self.ui_actions, self.max_ui_actions))
        
        return warnings
    
    def use_reviewer(self) -> None:
        """Consume one reviewer call."""
        self.reviewer_calls += 1
    
    def use_vision(self) -> None:
        """Consume one vision call."""
        self.vision_calls += 1
    
    def use_ui(self) -> None:
        """Consume one UI action."""
        self.ui_actions += 1
    
    def get_elapsed_seconds(self) -> float:
        """Get elapsed runtime in seconds."""
        if self.start_time is None:
            return 0.0
        import time
        return time.time() - self.start_time
    
    def get_usage_report(self) -> dict[str, Any]:
        """Get usage report."""
        elapsed = self.get_elapsed_seconds()
        return {
            "reviewer": f"{self.reviewer_calls}/{self.max_reviewer_calls}",
            "vision": f"{self.vision_calls}/{self.max_vision_calls}",
            "ui_actions": f"{self.ui_actions}/{self.max_ui_actions}",
            "runtime": f"{int(elapsed)}/{self.max_runtime_seconds}s",
        }
    
    def get_usage_percentage(self) -> dict[str, float]:
        """Get usage as percentages."""
        elapsed = self.get_elapsed_seconds()
        return {
            "reviewer": self.reviewer_calls / self.max_reviewer_calls * 100 if self.max_reviewer_calls else 0,
            "vision": self.vision_calls / self.max_vision_calls * 100 if self.max_vision_calls else 0,
            "ui_actions": self.ui_actions / self.max_ui_actions * 100 if self.max_ui_actions else 0,
            "runtime": elapsed / self.max_runtime_seconds * 100 if self.max_runtime_seconds else 0,
        }


class Orchestrator:
    """
    Main orchestrator that runs the Copilot-Gemini feedback loop.
    
    M6 Production Features:
    - Atomic checkpointing at each step for crash recovery
    - Circuit breakers for reviewer/vision/UI with automatic recovery
    - Retry policies with exponential backoff
    - Metrics collection for observability
    - UI desync detection and recovery
    - Budget enforcement with graceful pause
    - Kill switch integration with preemptive checks
    
    Loop flow:
    1. Send prompt to Copilot (initial or follow-up)
    2. Wait for Copilot response
    3. Capture response via perception pipeline
    4. Send to Gemini for review
    5. If ACCEPT → stop with success
    6. If CRITIQUE → send feedback to Copilot → repeat
    7. If CLARIFY → pause for user input
    8. On error or max iterations → stop
    """
    
    def __init__(
        self,
        config: AgentConfig,
        state_manager: StateManager,
        kill_switch: KillSwitch,
        reviewer: Optional[GeminiReviewer] = None,
        perception: Optional[PerceptionPipeline] = None,
        actions: Optional[ActionExecutor] = None,
        dry_run: bool = False,
        on_pause_callback: Optional[Callable[[], bool]] = None,
        # M5 components (created if not provided)
        checkpointer: Optional[AtomicCheckpointer] = None,
        metrics: Optional[MetricsCollector] = None,
        budget: Optional[OrchestratorBudget] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Agent configuration
            state_manager: State manager instance
            kill_switch: Kill switch instance
            reviewer: Gemini reviewer (created if not provided)
            perception: Perception pipeline (created if not provided)
            actions: Action executor (created if not provided)
            dry_run: If True, simulate actions without executing
            on_pause_callback: Called when pausing for user input, returns True to continue
            checkpointer: Atomic checkpointer for crash recovery
            metrics: Metrics collector for observability
            budget: Budget limits for operations
        """
        self.config = config
        self.state_manager = state_manager
        self.kill_switch = kill_switch
        self.dry_run = dry_run
        self._on_pause_callback = on_pause_callback
        
        # Initialize reviewer
        self.reviewer = reviewer or GeminiReviewer(
            model=config.gemini.model,
            timeout_seconds=config.gemini.timeout_seconds,
            max_retries=config.gemini.max_retries,
        )
        
        # Initialize perception (lazy - only when needed)
        self._perception = perception
        
        # Initialize actions (lazy - only when needed)
        self._actions = actions
        
        # M5: Checkpointing
        self._checkpointer = checkpointer
        
        # M5: Metrics
        self._metrics_collector = metrics or MetricsCollector()
        self._session_metrics = SessionMetrics(self._metrics_collector)
        
        # M5: Circuit breakers
        self._reviewer_circuit = create_reviewer_circuit()
        self._vision_circuit = create_vision_circuit()
        self._ui_circuit = create_ui_circuit()
        
        # M5: Desync detection
        self._desync_detector = DesyncDetector(
            on_desync=self._handle_desync,
        )
        self._recovery_manager = RecoveryManager(
            refocus_fn=self._refocus_window,
            recapture_fn=self._do_recapture,
        )
        self._parse_tracker = ParseFailureTracker(threshold=3)
        
        # M6: Budget from config
        self._budget = budget or OrchestratorBudget.from_config(config)
        
        # Iteration control
        self._repeated_critiques: List[str] = []
        self._consecutive_errors = 0
        self._paused = False
        self._should_stop = False
        
        logger.info(
            "Orchestrator initialized (M6 production)",
            dry_run=dry_run,
            max_iterations=config.reviewer.max_iterations,
            pause_before_send=config.reviewer.pause_before_send,
            checkpointing=checkpointer is not None,
            budget_limits=self._budget.get_usage_report(),
        )
    
    @property
    def perception(self) -> PerceptionPipeline:
        """Get or create perception pipeline."""
        if self._perception is None:
            session_path = self.state_manager.session_path
            output_dir = session_path / "captures" if session_path else None
            self._perception = PerceptionPipeline(output_dir=output_dir)
        return self._perception
    
    @property
    def actions(self) -> ActionExecutor:
        """Get or create action executor."""
        if self._actions is None:
            self._actions = ActionExecutor(
                dry_run=self.dry_run,
                kill_switch_check=lambda: self.kill_switch.triggered,
            )
        return self._actions
    
    @property
    def checkpointer(self) -> Optional[AtomicCheckpointer]:
        """Get checkpointer instance."""
        return self._checkpointer
    
    @property
    def metrics(self) -> SessionMetrics:
        """Get session metrics."""
        return self._session_metrics
    
    @property
    def budget(self) -> OrchestratorBudget:
        """Get budget tracker."""
        return self._budget
    
    def get_stats(self) -> str:
        """Get formatted statistics for display."""
        return self._metrics_collector.get_stats_display()
    
    def get_circuit_status(self) -> dict[str, str]:
        """Get status of all circuit breakers."""
        return {
            "reviewer": self._reviewer_circuit.state.value,
            "vision": self._vision_circuit.state.value,
            "ui": self._ui_circuit.state.value,
        }
    
    # M5: Desync handling
    def _handle_desync(self, event: DesyncEvent) -> None:
        """Handle desync detection event."""
        logger.warning(
            "UI desync detected",
            reason=event.reason.value,
            action=event.action,
            iteration=event.iteration,
        )
        self._session_metrics.ui_desync(event.reason.value)
    
    def _refocus_window(self) -> bool:
        """Attempt to refocus the VS Code window."""
        try:
            # Use action executor to focus window
            # This will be implemented by actuator
            return True
        except Exception as e:
            logger.error("Failed to refocus window", error=str(e))
            return False
    
    def _do_recapture(self) -> Optional[CaptureResult]:
        """Perform a fresh capture."""
        try:
            return self.perception.capture_copilot_response()
        except Exception as e:
            logger.error("Recapture failed", error=str(e))
            return None
    
    async def run(self) -> None:
        """
        Run the main orchestration loop.
        
        M6 Production features:
        - Checkpoints at each step for resume
        - Kill switch checks before each operation
        - Circuit breaker protection for external calls
        - Metrics emission throughout
        - Budget enforcement with warnings
        """
        session = self.state_manager.session
        if not session:
            raise RuntimeError("No active session")
        
        max_iterations = self.config.reviewer.max_iterations
        
        # M6: Start budget tracking
        self._budget.start()
        
        # M5: Initialize metrics for this session
        self._metrics_collector.set_session(
            session.session_id,
            self.state_manager.session_path,
        )
        self._session_metrics.session_start(session.task_description)
        
        # M5: Initialize checkpointing
        if self._checkpointer:
            self._checkpointer.initialize(
                session_id=session.session_id,
                task=session.task_description,
                max_iterations=max_iterations,
            )
        
        logger.info(
            "Starting orchestration (M6 production)",
            session_id=session.session_id,
            task=session.task_description[:50],
            max_iterations=max_iterations,
            budget=self._budget.get_usage_report(),
        )
        
        try:
            # M5: Check kill switch before starting
            self.kill_switch.check()
            
            # Transition to prompting phase
            self.state_manager.transition_to(SessionPhase.PROMPTING)
            
            while not self.kill_switch.triggered and not self._should_stop:
                # M5: Preemptive kill switch check
                try:
                    self.kill_switch.check()
                except KillSwitchTriggered:
                    logger.info("Kill switch triggered during loop")
                    self._session_metrics.kill_switch("loop_check")
                    break
                
                # Check iteration limit
                if session.iteration_count >= max_iterations:
                    logger.info("Max iterations reached", max=max_iterations)
                    session.completion_reason = "max_iterations"
                    self.state_manager.transition_to(SessionPhase.COMPLETE)
                    break
                
                # M6: Enhanced budget check with warnings
                budget_ok, exhausted = self._check_budget_enhanced()
                if not budget_ok:
                    logger.warning("Budget exhausted", resource=exhausted)
                    self._session_metrics.budget_exhausted(
                        exhausted or "unknown",
                        getattr(self._budget, exhausted.replace("_calls", "_calls") if exhausted else "reviewer_calls", 0),
                        getattr(self._budget, f"max_{exhausted}" if exhausted else "max_reviewer_calls", 0),
                    )
                    session.completion_reason = f"budget_exhausted_{exhausted}"
                    self.state_manager.transition_to(SessionPhase.PAUSED)
                    
                    if self._checkpointer:
                        self._checkpointer.mark_paused(f"budget:{exhausted}")
                    break
                
                # Run one iteration
                result = await self._run_iteration()
                
                if not result.should_continue:
                    logger.info(
                        "Stopping orchestration",
                        reason=result.stop_reason,
                        iteration=result.iteration,
                    )
                    break
                
                # Brief pause between iterations with kill switch check
                await async_wait_with_killswitch(self.kill_switch, 0.5)
        
        except KillSwitchTriggered:
            logger.warning("Kill switch triggered, stopping orchestration")
            self._session_metrics.kill_switch("exception")
            session.completion_reason = "kill_switch"
            self.state_manager.transition_to(SessionPhase.PAUSED)
            
            if self._checkpointer:
                self._checkpointer.mark_paused("kill_switch")
        
        except CircuitOpenError as e:
            logger.error("Circuit breaker open", circuit=e.circuit_name)
            self._metrics_collector.record(
                MetricType.CIRCUIT_OPEN,
                circuit=e.circuit_name,
                success=False,
            )
            session.completion_reason = f"circuit_open_{e.circuit_name}"
            self.state_manager.transition_to(SessionPhase.PAUSED)
            
            if self._checkpointer:
                self._checkpointer.mark_paused(f"circuit_open:{e.circuit_name}")
        
        except Exception as e:
            logger.error("Orchestration error", error=str(e))
            self.state_manager.record_error("orchestration_error", str(e), recoverable=False)
            self.state_manager.transition_to(SessionPhase.FAILED)
            
            self._metrics_collector.record_error(
                "orchestration_error",
                str(e),
                phase="orchestration",
                recoverable=False,
            )
            
            if self._checkpointer:
                self._checkpointer.mark_aborted(str(e))
        
        finally:
            # Final checkpoint
            self.state_manager.checkpoint()
            
            # M5: Record session end
            self._session_metrics.session_end(
                reason=session.completion_reason or "unknown",
                success=session.phase == SessionPhase.COMPLETE,
            )
            
            logger.info(
                "Orchestration ended",
                session_id=session.session_id,
                phase=session.phase.value,
                iterations=session.iteration_count,
                verdict=session.current_verdict.value if session.current_verdict else None,
                budget=self._budget.get_usage_report(),
            )
    
    def _check_budget(self) -> bool:
        """Check if budget is available for next iteration (legacy)."""
        ok, _ = self._check_budget_enhanced()
        return ok
    
    def _check_budget_enhanced(self) -> tuple[bool, Optional[str]]:
        """
        Check all budgets and emit warnings.
        
        Returns:
            (ok, exhausted_resource) tuple
        """
        # Check all budgets
        ok, exhausted = self._budget.check_all()
        
        if not ok:
            return False, exhausted
        
        # Emit warnings for resources approaching limit
        for resource, used, limit in self._budget.get_warnings():
            self._session_metrics.budget_warning(resource, used, limit)
            logger.warning(
                "Budget warning",
                resource=resource,
                used=used,
                limit=limit,
                percent=int(used / limit * 100),
            )
        
        return True, None
    
    async def _run_iteration(self) -> IterationResult:
        """
        Run a single iteration of the review loop.
        
        M6 Production: Checkpointing, retries, metrics at each step.
        
        Returns:
            IterationResult with verdict and continuation flag
        """
        session = self.state_manager.session
        if not session:
            return IterationResult(success=False, iteration=0, stop_reason="no_session")
        
        start = time.time()
        iteration = session.iteration_count + 1
        
        # M5: Update iteration in desync detector
        self._desync_detector.set_iteration(iteration)
        self._session_metrics.set_iteration(iteration)
        
        # Determine the prompt to send
        prompt = session.next_prompt or session.current_prompt or session.task_description
        source = "initial" if iteration == 1 else "gemini_followup"
        
        # Start iteration in state
        self.state_manager.start_iteration(prompt, source)
        
        # M5: Record iteration start
        self._session_metrics.iteration_start(iteration, source)
        
        # M5: Checkpoint iteration start
        if self._checkpointer:
            self._checkpointer.start_iteration(prompt)
        
        logger.info(
            "Starting iteration",
            iteration=iteration,
            prompt_length=len(prompt),
            source=source,
        )
        
        try:
            # M5: Kill switch check
            self.kill_switch.check()
            
            # Step 1: Send prompt to Copilot (via GUI)
            if not self.dry_run:
                self.state_manager.transition_to(SessionPhase.PROMPTING)
                self._session_metrics.set_phase("prompting")
                
                # M5: Use retry policy for UI actions
                await self._send_prompt_with_retry(prompt)
            
            # M5: Kill switch check before wait
            self.kill_switch.check()
            
            # Step 2: Wait for Copilot response (interruptible)
            self.state_manager.transition_to(SessionPhase.WAITING)
            self._session_metrics.set_phase("waiting")
            
            if self._checkpointer:
                self._checkpointer.record_step(StepType.WAITING_RESPONSE)
            
            await async_wait_with_killswitch(
                self.kill_switch,
                self.config.reviewer.response_wait_seconds,
            )
            
            # M5: Kill switch check before capture
            self.kill_switch.check()
            
            # Step 3: Capture response with retry and desync detection
            self.state_manager.transition_to(SessionPhase.CAPTURING)
            self._session_metrics.set_phase("capturing")
            
            if self._checkpointer:
                self._checkpointer.record_step(StepType.CAPTURE_STARTED)
            
            capture_result = await self._capture_with_retry()
            
            if not capture_result.success:
                return self._handle_capture_failure(capture_result, iteration, start)
            
            self._consecutive_errors = 0
            copilot_response = capture_result.text
            self.state_manager.record_response(copilot_response, capture_result.method.value)
            
            # M5: Record capture metrics and checkpoint
            self._session_metrics.capture(
                success=True,
                method=capture_result.method.value,
            )
            
            if self._checkpointer:
                self._checkpointer.record_capture(
                    copilot_response,
                    capture_result.method.value,
                )
            
            logger.info(
                "Response captured",
                iteration=iteration,
                length=len(copilot_response),
                method=capture_result.method.value,
                confidence=capture_result.confidence,
            )
            
            # M5: Kill switch check before review
            self.kill_switch.check()
            
            # Step 4: Send to Gemini for review with circuit breaker
            self.state_manager.transition_to(SessionPhase.REVIEWING)
            self._session_metrics.set_phase("reviewing")
            
            if self._checkpointer:
                self._checkpointer.record_step(StepType.REVIEW_STARTED)
            
            # M5: Consume budget
            self._budget.use_reviewer()
            
            # Build history summary for context
            history_summary = self._build_history_summary()
            
            # M5: Review with retry and circuit breaker
            review_result = await self._review_with_retry(
                task=session.task_description,
                copilot_response=copilot_response,
                iteration=iteration,
                history_summary=history_summary,
            )
            
            # M5: Track parse failures
            if review_result.verdict == ReviewVerdict.ERROR:
                if self._parse_tracker.record_failure(review_result.reasoning or "Unknown error"):
                    logger.error("Parse failure threshold reached, pausing")
                    self.state_manager.transition_to(SessionPhase.PAUSED)
                    return IterationResult(
                        success=False,
                        iteration=iteration,
                        stop_reason="parse_failures",
                        duration_ms=int((time.time() - start) * 1000),
                    )
            else:
                self._parse_tracker.record_success()
            
            # Map ReviewVerdict to GeminiVerdict for state
            verdict_map = {
                ReviewVerdict.ACCEPT: GeminiVerdict.ACCEPT,
                ReviewVerdict.CRITIQUE: GeminiVerdict.CRITIQUE,
                ReviewVerdict.CLARIFY: GeminiVerdict.CLARIFY,
                ReviewVerdict.ERROR: GeminiVerdict.ERROR,
            }
            state_verdict = verdict_map.get(review_result.verdict, GeminiVerdict.ERROR)
            
            # Record verdict in state
            self.state_manager.record_verdict(
                verdict=state_verdict,
                feedback=review_result.reasoning,
                confidence=review_result.confidence,
                next_prompt=review_result.follow_up_prompt,
            )
            
            # M5: Checkpoint review result
            if self._checkpointer:
                self._checkpointer.record_review(
                    verdict=review_result.verdict.value,
                    feedback=review_result.reasoning or "",
                    confidence=review_result.confidence,
                    next_prompt=review_result.follow_up_prompt,
                )
            
            logger.info(
                "Review complete",
                iteration=iteration,
                verdict=review_result.verdict.value,
                confidence=review_result.confidence,
                has_followup=bool(review_result.follow_up_prompt),
            )
            
            # Complete iteration and checkpoint
            self.state_manager.complete_iteration()
            
            if self._checkpointer:
                self._checkpointer.complete_iteration()
            
            # Step 5: Handle verdict
            duration_ms = int((time.time() - start) * 1000)
            
            # M5: Record iteration end
            self._session_metrics.iteration_end(iteration, review_result.verdict.value)
            
            return await self._handle_verdict(
                review_result=review_result,
                copilot_response=copilot_response,
                iteration=iteration,
                duration_ms=duration_ms,
            )
        
        except KillSwitchTriggered:
            # Let it propagate to run() for proper handling
            raise
        
        except CircuitOpenError:
            # Let it propagate to run() for proper handling
            raise
        
        except Exception as e:
            logger.error("Iteration failed", iteration=iteration, error=str(e))
            self.state_manager.record_error("iteration_error", str(e))
            
            self._metrics_collector.record_error(
                "iteration_error",
                str(e),
                phase="iteration",
                iteration=iteration,
            )
            
            return IterationResult(
                success=False,
                iteration=iteration,
                stop_reason=f"exception: {str(e)}",
                duration_ms=int((time.time() - start) * 1000),
            )
    
    def _handle_capture_failure(
        self,
        capture_result: CaptureResult,
        iteration: int,
        start: float,
    ) -> IterationResult:
        """Handle capture failure with recovery attempt."""
        self._consecutive_errors += 1
        self.state_manager.record_error(
            "capture_failed",
            capture_result.error or "Unknown capture error",
        )
        
        self._session_metrics.capture(
            success=False,
            method="failed",
        )
        
        if self._consecutive_errors >= 3:
            return IterationResult(
                success=False,
                iteration=iteration,
                stop_reason="consecutive_capture_failures",
                duration_ms=int((time.time() - start) * 1000),
            )
        
        # Retry this iteration
        session = self.state_manager.session
        if session:
            session.iteration_count -= 1  # Don't count failed attempts
        
        return IterationResult(
            success=False,
            iteration=iteration,
            should_continue=True,
            duration_ms=int((time.time() - start) * 1000),
        )
    
    async def _handle_verdict(
        self,
        review_result: ReviewResult,
        copilot_response: str,
        iteration: int,
        duration_ms: int,
    ) -> IterationResult:
        """Handle review verdict and determine next action."""
        session = self.state_manager.session
        if not session:
            return IterationResult(
                success=False,
                iteration=iteration,
                stop_reason="no_session",
            )
        
        if review_result.verdict == ReviewVerdict.ACCEPT:
            session.completion_reason = "gemini_accepted"
            session.final_result = copilot_response
            self.state_manager.transition_to(SessionPhase.COMPLETE)
            
            if self._checkpointer:
                self._checkpointer.mark_complete("accepted")
            
            return IterationResult(
                success=True,
                iteration=iteration,
                verdict=review_result.verdict,
                captured_response=copilot_response,
                should_continue=False,
                stop_reason="accepted",
                duration_ms=duration_ms,
            )
        
        elif review_result.verdict == ReviewVerdict.CRITIQUE:
            # Check for repeated critiques
            if self._check_repeated_critiques(review_result.follow_up_prompt):
                session.completion_reason = "repeated_critiques"
                self.state_manager.transition_to(SessionPhase.COMPLETE)
                
                if self._checkpointer:
                    self._checkpointer.mark_complete("repeated_critiques")
                
                return IterationResult(
                    success=False,
                    iteration=iteration,
                    verdict=review_result.verdict,
                    feedback=review_result.follow_up_prompt,
                    should_continue=False,
                    stop_reason="repeated_critiques",
                    duration_ms=duration_ms,
                )
            
            # Pause before sending feedback if configured
            if self.config.reviewer.pause_before_send:
                self.state_manager.transition_to(SessionPhase.PAUSED)
                
                if self._checkpointer:
                    self._checkpointer.mark_paused("pause_before_send")
                
                should_continue = await self._pause_for_approval(review_result)
                
                if not should_continue:
                    session.completion_reason = "user_stopped"
                    return IterationResult(
                        success=False,
                        iteration=iteration,
                        verdict=review_result.verdict,
                        should_continue=False,
                        stop_reason="user_stopped",
                        duration_ms=duration_ms,
                    )
            
            # Set up next prompt
            session.next_prompt = review_result.follow_up_prompt
            
            return IterationResult(
                success=True,
                iteration=iteration,
                verdict=review_result.verdict,
                feedback=review_result.follow_up_prompt,
                captured_response=copilot_response,
                should_continue=True,
                duration_ms=duration_ms,
            )
        
        elif review_result.verdict == ReviewVerdict.CLARIFY:
            # Pause for human input
            self.state_manager.transition_to(SessionPhase.PAUSED)
            
            if self._checkpointer:
                self._checkpointer.mark_paused("clarify_needed")
            
            should_continue = await self._pause_for_clarification(review_result)
            
            if not should_continue:
                session.completion_reason = "clarification_stopped"
                return IterationResult(
                    success=False,
                    iteration=iteration,
                    verdict=review_result.verdict,
                    should_continue=False,
                    stop_reason="clarification_stopped",
                    duration_ms=duration_ms,
                )
            
            return IterationResult(
                success=True,
                iteration=iteration,
                verdict=review_result.verdict,
                should_continue=True,
                duration_ms=duration_ms,
            )
        
        else:  # ERROR
            self._consecutive_errors += 1
            
            if self._consecutive_errors >= 3:
                session.completion_reason = "review_errors"
                self.state_manager.transition_to(SessionPhase.FAILED)
                
                if self._checkpointer:
                    self._checkpointer.mark_aborted("review_errors")
                
                return IterationResult(
                    success=False,
                    iteration=iteration,
                    verdict=review_result.verdict,
                    should_continue=False,
                    stop_reason="review_errors",
                    duration_ms=duration_ms,
                )
            
            return IterationResult(
                success=False,
                iteration=iteration,
                verdict=review_result.verdict,
                should_continue=True,
                duration_ms=duration_ms,
            )
    
    async def _send_prompt_with_retry(self, prompt: str) -> bool:
        """
        Send prompt to Copilot with retry and circuit breaker.
        
        Args:
            prompt: Text to send to Copilot
            
        Returns:
            True if successful
        """
        logger.info("Sending prompt to Copilot", length=len(prompt))
        
        async def _do_send() -> bool:
            # M5: Check circuit breaker
            if not await self._ui_circuit.can_execute():
                raise CircuitOpenError(self._ui_circuit.name)
            
            # M5: Track UI action
            self._budget.use_ui()
            self._metrics_collector.start_timer("ui_type")
            
            try:
                # Type the prompt using paste for speed
                result = self.actions.type_text(prompt, use_clipboard=True)
                if not result.success:
                    raise RuntimeError(result.error or "Failed to type prompt")
                
                # Press Enter to submit
                await asyncio.sleep(0.1)
                result = self.actions.press_key("enter")
                if not result.success:
                    raise RuntimeError(result.error or "Failed to press Enter")
                
                duration = self._metrics_collector.stop_timer("ui_type")
                self._session_metrics.ui_action("type_prompt", True, duration)
                await self._ui_circuit.record_success()
                
                return True
                
            except Exception as e:
                duration = self._metrics_collector.stop_timer("ui_type")
                self._session_metrics.ui_action("type_prompt", False, duration)
                await self._ui_circuit.record_failure(e)
                raise
        
        # M5: Retry with policy
        return await retry_with_backoff(
            _do_send,
            policy=UI_ACTION_RETRY_POLICY,
            circuit=self._ui_circuit,
            on_retry=lambda attempt, error: self._metrics_collector.record_retry(
                "send_prompt", attempt, UI_ACTION_RETRY_POLICY.max_retries, 
                UI_ACTION_RETRY_POLICY.get_delay(attempt), str(error)
            ),
        )
    
    async def _capture_with_retry(self) -> CaptureResult:
        """
        Capture Copilot's response with retry and desync detection.
        
        Returns:
            CaptureResult with extracted text
        """
        logger.info("Capturing response")
        
        # Wait for response to stabilize
        stability_ms = self.config.reviewer.response_stability_ms
        await async_wait_with_killswitch(
            self.kill_switch,
            stability_ms / 1000,
        )
        
        async def _do_capture() -> CaptureResult:
            # M5: Check circuit breaker
            if not await self._vision_circuit.can_execute():
                raise CircuitOpenError(self._vision_circuit.name)
            
            self._budget.use_vision()
            self._metrics_collector.start_timer("capture")
            
            try:
                result = self.perception.capture_copilot_response(
                    use_preprocessing=True,
                    use_vision_fallback=True,
                )
                
                duration = self._metrics_collector.stop_timer("capture")
                
                if result.success:
                    await self._vision_circuit.record_success()
                else:
                    await self._vision_circuit.record_failure(
                        RuntimeError(result.error or "Capture failed")
                    )
                
                return result
                
            except Exception as e:
                self._metrics_collector.stop_timer("capture")
                await self._vision_circuit.record_failure(e)
                raise
        
        try:
            return await retry_with_backoff(
                _do_capture,
                policy=VISION_RETRY_POLICY,
                circuit=self._vision_circuit,
                on_retry=lambda attempt, error: self._metrics_collector.record_retry(
                    "capture", attempt, VISION_RETRY_POLICY.max_retries,
                    VISION_RETRY_POLICY.get_delay(attempt), str(error)
                ),
            )
        except (CircuitOpenError, RetryExhaustedError) as e:
            # Return failed capture result
            return CaptureResult(
                success=False,
                text="",
                method=CaptureMethod.OCR,
                error=str(e),
            )
    
    async def _review_with_retry(
        self,
        task: str,
        copilot_response: str,
        iteration: int,
        history_summary: Optional[str],
    ) -> ReviewResult:
        """
        Send to Gemini for review with retry and circuit breaker.
        
        Args:
            task: Task description
            copilot_response: Response to review
            iteration: Current iteration
            history_summary: Previous iteration summary
            
        Returns:
            ReviewResult from Gemini
        """
        async def _do_review() -> ReviewResult:
            # M5: Check circuit breaker
            if not await self._reviewer_circuit.can_execute():
                raise CircuitOpenError(self._reviewer_circuit.name)
            
            self._metrics_collector.start_timer("review")
            
            try:
                result = await self.reviewer.review(
                    task=task,
                    copilot_response=copilot_response,
                    iteration=iteration,
                    max_iterations=self.config.reviewer.max_iterations,
                    history_summary=history_summary,
                )
                
                duration = self._metrics_collector.stop_timer("review")
                
                self._session_metrics.reviewer_call(
                    success=result.verdict != ReviewVerdict.ERROR,
                    verdict=result.verdict.value,
                    duration_ms=duration,
                    model=self.config.gemini.model,
                )
                
                if result.verdict != ReviewVerdict.ERROR:
                    await self._reviewer_circuit.record_success()
                else:
                    await self._reviewer_circuit.record_failure(
                        RuntimeError(result.reasoning or "Review error")
                    )
                
                return result
                
            except Exception as e:
                duration = self._metrics_collector.stop_timer("review")
                self._session_metrics.reviewer_call(
                    success=False,
                    duration_ms=duration,
                    model=self.config.gemini.model,
                )
                await self._reviewer_circuit.record_failure(e)
                raise
        
        try:
            return await retry_with_backoff(
                _do_review,
                policy=REVIEWER_RETRY_POLICY,
                circuit=self._reviewer_circuit,
                on_retry=lambda attempt, error: self._metrics_collector.record_retry(
                    "review", attempt, REVIEWER_RETRY_POLICY.max_retries,
                    REVIEWER_RETRY_POLICY.get_delay(attempt), str(error)
                ),
            )
        except (CircuitOpenError, RetryExhaustedError) as e:
            # Return error result
            return ReviewResult(
                verdict=ReviewVerdict.ERROR,
                reasoning=str(e),
                confidence="low",
            )

    async def _send_prompt_to_copilot(self, prompt: str) -> bool:
        """
        Send prompt to Copilot via GUI automation.
        
        Deprecated: Use _send_prompt_with_retry instead.
        
        Args:
            prompt: Text to send to Copilot
            
        Returns:
            True if successful
        """
        return await self._send_prompt_with_retry(prompt)
    
    async def _capture_response(self) -> CaptureResult:
        """
        Capture Copilot's response using perception pipeline.
        
        Deprecated: Use _capture_with_retry instead.
        
        Returns:
            CaptureResult with extracted text
        """
        return await self._capture_with_retry()
    
    def _build_history_summary(self) -> Optional[str]:
        """
        Build a summary of previous iterations for context.
        
        Returns:
            Summary string or None
        """
        session = self.state_manager.session
        if not session or not session.iteration_history:
            return None
        
        # Summarize last 3 iterations
        recent = session.iteration_history[-3:]
        
        lines = []
        for record in recent:
            verdict = record.gemini_verdict or "unknown"
            feedback = record.gemini_feedback or ""
            if len(feedback) > 100:
                feedback = feedback[:100] + "..."
            lines.append(f"Iteration {record.iteration_number}: {verdict.upper()} - {feedback}")
        
        return "\n".join(lines)
    
    def _check_repeated_critiques(self, feedback: Optional[str]) -> bool:
        """
        Check if we're getting repeated identical critiques.
        
        Args:
            feedback: Current feedback to check
            
        Returns:
            True if we should stop due to repeated critiques
        """
        if not feedback:
            return False
        
        # Normalize feedback for comparison
        normalized = feedback.strip().lower()
        
        self._repeated_critiques.append(normalized)
        
        # Keep only recent critiques
        max_history = self.config.reviewer.stop_on_repeated_critiques + 1
        if len(self._repeated_critiques) > max_history:
            self._repeated_critiques = self._repeated_critiques[-max_history:]
        
        # Check for repeats
        if len(self._repeated_critiques) >= self.config.reviewer.stop_on_repeated_critiques:
            # Check if all recent critiques are identical
            recent = self._repeated_critiques[-self.config.reviewer.stop_on_repeated_critiques:]
            if len(set(recent)) == 1:
                logger.warning(
                    "Detected repeated critiques",
                    count=self.config.reviewer.stop_on_repeated_critiques,
                )
                return True
        
        return False
    
    async def _pause_for_approval(self, review_result: ReviewResult) -> bool:
        """
        Pause and wait for user approval before sending feedback.
        
        Args:
            review_result: The review result to show user
            
        Returns:
            True if user approves, False to stop
        """
        logger.info(
            "Pausing for approval",
            feedback=review_result.follow_up_prompt[:100] if review_result.follow_up_prompt else "",
        )
        
        self._paused = True
        
        if self._on_pause_callback:
            # Call the callback and wait for response
            try:
                result = self._on_pause_callback()
                return result
            except Exception as e:
                logger.error("Pause callback error", error=str(e))
                return False
        
        # Default: wait indefinitely for resume with kill switch checks
        try:
            while self._paused and not self.kill_switch.triggered:
                await async_wait_with_killswitch(self.kill_switch, 0.5)
        except KillSwitchTriggered:
            return False
        
        return not self.kill_switch.triggered
    
    async def _pause_for_clarification(self, review_result: ReviewResult) -> bool:
        """
        Pause for user to provide clarification.
        
        Args:
            review_result: The review result requesting clarification
            
        Returns:
            True if user provides clarification, False to stop
        """
        logger.info(
            "Pausing for clarification",
            reason=review_result.reasoning,
        )
        
        self._paused = True
        
        if self._on_pause_callback:
            try:
                return self._on_pause_callback()
            except Exception as e:
                logger.error("Clarification callback error", error=str(e))
                return False
        
        # Wait for resume with kill switch checks
        try:
            while self._paused and not self.kill_switch.triggered:
                await async_wait_with_killswitch(self.kill_switch, 0.5)
        except KillSwitchTriggered:
            return False
        
        return not self.kill_switch.triggered
    
    def resume(self) -> None:
        """Resume from paused state."""
        self._paused = False
        logger.info("Resumed from pause")
    
    def stop(self) -> None:
        """Signal the orchestrator to stop."""
        self._should_stop = True
        self._paused = False
        logger.info("Stop requested")
    
    def set_next_prompt(self, prompt: str) -> None:
        """
        Set a custom prompt for the next iteration (human override).
        
        Args:
            prompt: Custom prompt to use
        """
        session = self.state_manager.session
        if session:
            session.next_prompt = prompt
            session.current_prompt_source = "human_override"
            logger.info("Custom prompt set", length=len(prompt))
    
    async def resume_from_checkpoint(self) -> bool:
        """
        Resume session from checkpoint.
        
        Returns:
            True if resume successful, False otherwise
        """
        if not self._checkpointer:
            logger.warning("No checkpointer available for resume")
            return False
        
        state = self._checkpointer.load()
        if not state:
            logger.warning("No checkpoint found for resume")
            return False
        
        if not state.resumable:
            logger.warning("Session is not resumable", session_id=state.session_id)
            return False
        
        resume_info = self._checkpointer.get_resume_point()
        if not resume_info:
            logger.warning("Could not determine resume point")
            return False
        
        logger.info(
            "Resuming from checkpoint",
            session_id=state.session_id,
            iteration=state.current_iteration,
            last_step=state.last_step_type.value if state.last_step_type else None,
            resume_action=resume_info.get("resume_action"),
        )
        
        # Restore state
        session = self.state_manager.session
        if session:
            session.iteration_count = state.current_iteration
            session.next_prompt = state.next_prompt or state.current_prompt
        
        # Record resume in metrics
        self._metrics_collector.record(
            MetricType.SESSION_RESUME,
            iteration=state.current_iteration,
            resume_action=resume_info.get("resume_action"),
        )
        
        return True
    
    def get_resume_info(self) -> Optional[dict]:
        """
        Get information about available resume point.
        
        Returns:
            Dict with resume info or None
        """
        if not self._checkpointer:
            return None
        
        state = self._checkpointer.load()
        if not state or not state.resumable:
            return None
        
        return self._checkpointer.get_resume_point()
