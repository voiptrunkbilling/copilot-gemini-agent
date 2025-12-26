"""
Orchestrator - main control loop for the Copilot-Gemini feedback cycle.

M4: Full implementation with reviewer integration, iteration control,
and pause-before-send safety mode.
"""

import asyncio
import time
from typing import Optional, Callable, List
from dataclasses import dataclass

from copilot_agent.config import AgentConfig
from copilot_agent.state import StateManager, SessionPhase, GeminiVerdict
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.reviewer.gemini import GeminiReviewer, ReviewResult, ReviewVerdict
from copilot_agent.perception.pipeline import PerceptionPipeline, CaptureResult
from copilot_agent.actuator.actions import ActionExecutor
from copilot_agent.logging import get_logger

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


class Orchestrator:
    """
    Main orchestrator that runs the Copilot-Gemini feedback loop.
    
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
        
        # Iteration control
        self._repeated_critiques: List[str] = []
        self._consecutive_errors = 0
        self._paused = False
        self._should_stop = False
        
        logger.info(
            "Orchestrator initialized",
            dry_run=dry_run,
            max_iterations=config.reviewer.max_iterations,
            pause_before_send=config.reviewer.pause_before_send,
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
    
    async def run(self) -> None:
        """
        Run the main orchestration loop.
        
        This is the primary entry point for running the agent.
        """
        session = self.state_manager.session
        if not session:
            raise RuntimeError("No active session")
        
        max_iterations = self.config.reviewer.max_iterations
        
        logger.info(
            "Starting orchestration",
            session_id=session.session_id,
            task=session.task_description[:50],
            max_iterations=max_iterations,
        )
        
        try:
            # Transition to prompting phase
            self.state_manager.transition_to(SessionPhase.PROMPTING)
            
            while not self.kill_switch.triggered and not self._should_stop:
                # Check iteration limit
                if session.iteration_count >= max_iterations:
                    logger.info("Max iterations reached", max=max_iterations)
                    session.completion_reason = "max_iterations"
                    self.state_manager.transition_to(SessionPhase.COMPLETE)
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
                
                # Brief pause between iterations
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error("Orchestration error", error=str(e))
            self.state_manager.record_error("orchestration_error", str(e), recoverable=False)
            self.state_manager.transition_to(SessionPhase.FAILED)
        
        finally:
            # Final checkpoint
            self.state_manager.checkpoint()
            
            logger.info(
                "Orchestration ended",
                session_id=session.session_id,
                phase=session.phase.value,
                iterations=session.iteration_count,
                verdict=session.current_verdict.value if session.current_verdict else None,
            )
    
    async def _run_iteration(self) -> IterationResult:
        """
        Run a single iteration of the review loop.
        
        Returns:
            IterationResult with verdict and continuation flag
        """
        session = self.state_manager.session
        if not session:
            return IterationResult(success=False, iteration=0, stop_reason="no_session")
        
        start = time.time()
        iteration = session.iteration_count + 1
        
        # Determine the prompt to send
        prompt = session.next_prompt or session.current_prompt or session.task_description
        source = "initial" if iteration == 1 else "gemini_followup"
        
        # Start iteration in state
        self.state_manager.start_iteration(prompt, source)
        
        logger.info(
            "Starting iteration",
            iteration=iteration,
            prompt_length=len(prompt),
            source=source,
        )
        
        try:
            # Step 1: Send prompt to Copilot (via GUI)
            if not self.dry_run:
                self.state_manager.transition_to(SessionPhase.PROMPTING)
                await self._send_prompt_to_copilot(prompt)
            
            # Step 2: Wait for Copilot response
            self.state_manager.transition_to(SessionPhase.WAITING)
            await asyncio.sleep(self.config.reviewer.response_wait_seconds)
            
            # Step 3: Capture response
            self.state_manager.transition_to(SessionPhase.CAPTURING)
            capture_result = await self._capture_response()
            
            if not capture_result.success:
                self._consecutive_errors += 1
                self.state_manager.record_error(
                    "capture_failed",
                    capture_result.error or "Unknown capture error",
                )
                
                if self._consecutive_errors >= 3:
                    return IterationResult(
                        success=False,
                        iteration=iteration,
                        stop_reason="consecutive_capture_failures",
                        duration_ms=int((time.time() - start) * 1000),
                    )
                
                # Retry this iteration
                session.iteration_count -= 1  # Don't count failed attempts
                return IterationResult(
                    success=False,
                    iteration=iteration,
                    should_continue=True,
                    duration_ms=int((time.time() - start) * 1000),
                )
            
            self._consecutive_errors = 0
            copilot_response = capture_result.text
            self.state_manager.record_response(copilot_response, capture_result.method.value)
            
            logger.info(
                "Response captured",
                iteration=iteration,
                length=len(copilot_response),
                method=capture_result.method.value,
                confidence=capture_result.confidence,
            )
            
            # Step 4: Send to Gemini for review
            self.state_manager.transition_to(SessionPhase.REVIEWING)
            
            # Build history summary for context
            history_summary = self._build_history_summary()
            
            review_result = await self.reviewer.review(
                task=session.task_description,
                copilot_response=copilot_response,
                iteration=iteration,
                max_iterations=self.config.reviewer.max_iterations,
                history_summary=history_summary,
            )
            
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
            
            logger.info(
                "Review complete",
                iteration=iteration,
                verdict=review_result.verdict.value,
                confidence=review_result.confidence,
                has_followup=bool(review_result.follow_up_prompt),
            )
            
            # Complete iteration and checkpoint
            self.state_manager.complete_iteration()
            
            # Step 5: Handle verdict
            duration_ms = int((time.time() - start) * 1000)
            
            if review_result.verdict == ReviewVerdict.ACCEPT:
                session.completion_reason = "gemini_accepted"
                session.final_result = copilot_response
                self.state_manager.transition_to(SessionPhase.COMPLETE)
                
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
        
        except Exception as e:
            logger.error("Iteration failed", iteration=iteration, error=str(e))
            self.state_manager.record_error("iteration_error", str(e))
            
            return IterationResult(
                success=False,
                iteration=iteration,
                stop_reason=f"exception: {str(e)}",
                duration_ms=int((time.time() - start) * 1000),
            )
    
    async def _send_prompt_to_copilot(self, prompt: str) -> bool:
        """
        Send prompt to Copilot via GUI automation.
        
        Args:
            prompt: Text to send to Copilot
            
        Returns:
            True if successful
        """
        logger.info("Sending prompt to Copilot", length=len(prompt))
        
        # Type the prompt using paste for speed
        result = self.actions.type_text(prompt, use_clipboard=True)
        if not result.success:
            logger.error("Failed to type prompt", error=result.error)
            return False
        
        # Press Enter to submit
        await asyncio.sleep(0.1)
        result = self.actions.press_key("enter")
        if not result.success:
            logger.error("Failed to press Enter", error=result.error)
            return False
        
        return True
    
    async def _capture_response(self) -> CaptureResult:
        """
        Capture Copilot's response using perception pipeline.
        
        Returns:
            CaptureResult with extracted text
        """
        logger.info("Capturing response")
        
        # Wait for response to stabilize
        stability_ms = self.config.reviewer.response_stability_ms
        await asyncio.sleep(stability_ms / 1000)
        
        # Capture using perception pipeline
        result = self.perception.capture_copilot_response(
            use_preprocessing=True,
            use_vision_fallback=True,
        )
        
        return result
    
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
        
        # Default: wait indefinitely for resume
        while self._paused and not self.kill_switch.triggered:
            await asyncio.sleep(0.5)
        
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
        
        # Wait for resume
        while self._paused and not self.kill_switch.triggered:
            await asyncio.sleep(0.5)
        
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
