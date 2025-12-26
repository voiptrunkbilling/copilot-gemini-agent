"""
Unit tests for M4/M6: Orchestrator.

Updated for M6 with M5 integration (checkpointing, metrics, resilience).
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from copilot_agent.orchestrator import Orchestrator, IterationResult, OrchestratorBudget
from copilot_agent.config import AgentConfig
from copilot_agent.state import StateManager, SessionPhase, GeminiVerdict
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.reviewer.gemini import ReviewVerdict, ReviewResult
from copilot_agent.perception.pipeline import CaptureMethod


def run_async(coro):
    """Helper to run async tests without pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestOrchestratorBudget:
    """Tests for OrchestratorBudget."""
    
    def test_default_budget(self):
        """Test default budget values."""
        budget = OrchestratorBudget()
        
        assert budget.max_reviewer_calls == 50
        assert budget.max_vision_calls == 100
        assert budget.max_ui_actions == 200
        assert budget.reviewer_calls == 0
    
    def test_from_config(self):
        """Test creating budget from config."""
        from copilot_agent.config import AgentConfig
        config = AgentConfig()
        budget = OrchestratorBudget.from_config(config)
        
        assert budget.max_reviewer_calls == config.reviewer.max_reviews_per_session
        assert budget.max_vision_calls == config.perception.max_vision_per_session
    
    def test_check_and_use(self):
        """Test budget check and use."""
        budget = OrchestratorBudget(max_reviewer_calls=3)
        
        assert budget.check_reviewer() is True
        budget.use_reviewer()
        assert budget.reviewer_calls == 1
        
        budget.use_reviewer()
        budget.use_reviewer()
        assert budget.check_reviewer() is False
    
    def test_check_all(self):
        """Test checking all budgets at once."""
        budget = OrchestratorBudget(max_reviewer_calls=1)
        budget.start()
        
        ok, exhausted = budget.check_all()
        assert ok is True
        assert exhausted is None
        
        budget.use_reviewer()
        ok, exhausted = budget.check_all()
        assert ok is False
        assert exhausted == "reviewer_calls"
    
    def test_get_warnings(self):
        """Test warning threshold detection."""
        budget = OrchestratorBudget(max_reviewer_calls=10, warning_threshold=0.8)
        
        # No warning yet
        assert len(budget.get_warnings()) == 0
        
        # Use 8 calls (80% threshold)
        for _ in range(8):
            budget.use_reviewer()
        
        warnings = budget.get_warnings()
        assert len(warnings) == 1
        assert warnings[0][0] == "reviewer_calls"
    
    def test_get_usage_report(self):
        """Test usage report generation."""
        budget = OrchestratorBudget()
        budget.start()
        budget.use_reviewer()
        budget.use_vision()
        
        report = budget.get_usage_report()
        assert "1/50" in report["reviewer"]
        assert "1/100" in report["vision"]
        assert "runtime" in report
    
    def test_get_usage_percentage(self):
        """Test usage percentage calculation."""
        budget = OrchestratorBudget(max_reviewer_calls=100)
        
        for _ in range(50):
            budget.use_reviewer()
        
        percentages = budget.get_usage_percentage()
        assert percentages["reviewer"] == 50.0


class TestIterationResult:
    """Tests for IterationResult dataclass."""
    
    def test_create_success_result(self):
        """Test creating a successful result."""
        result = IterationResult(
            success=True,
            iteration=1,
            verdict=ReviewVerdict.ACCEPT,
            captured_response="Hello World",
            should_continue=False,
            stop_reason="accepted",
        )
        
        assert result.success is True
        assert result.iteration == 1
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.should_continue is False
    
    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = IterationResult(
            success=False,
            iteration=3,
            stop_reason="max_iterations",
        )
        
        assert result.success is False
        assert result.iteration == 3
        assert result.stop_reason == "max_iterations"


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""
    
    @pytest.fixture
    def config(self):
        return AgentConfig()
    
    @pytest.fixture
    def state_manager(self, config):
        return StateManager(config)
    
    @pytest.fixture
    def kill_switch(self):
        return KillSwitch()
    
    def test_init_default(self, config, state_manager, kill_switch):
        """Test default initialization."""
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
        )
        
        assert orchestrator.config == config
        assert orchestrator.state_manager == state_manager
        assert orchestrator.kill_switch == kill_switch
        assert orchestrator.dry_run is False
    
    def test_init_dry_run(self, config, state_manager, kill_switch):
        """Test dry-run initialization."""
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
        
        assert orchestrator.dry_run is True
    
    def test_reviewer_created(self, config, state_manager, kill_switch):
        """Test reviewer is created if not provided."""
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
        )
        
        assert orchestrator.reviewer is not None
        assert orchestrator.reviewer.model == config.gemini.model
    
    def test_custom_reviewer(self, config, state_manager, kill_switch):
        """Test using custom reviewer."""
        from copilot_agent.reviewer.gemini import GeminiReviewer
        
        custom_reviewer = GeminiReviewer(model="custom-model")
        
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            reviewer=custom_reviewer,
        )
        
        assert orchestrator.reviewer == custom_reviewer


class TestOrchestratorControl:
    """Tests for orchestrator control methods."""
    
    @pytest.fixture
    def orchestrator(self):
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        return Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
    
    def test_stop(self, orchestrator):
        """Test stop method."""
        orchestrator.stop()
        
        assert orchestrator._should_stop is True
        assert orchestrator._paused is False
    
    def test_resume(self, orchestrator):
        """Test resume method."""
        orchestrator._paused = True
        orchestrator.resume()
        
        assert orchestrator._paused is False
    
    def test_set_next_prompt(self, orchestrator):
        """Test setting custom next prompt."""
        orchestrator.state_manager.create_session("Test task")
        
        orchestrator.set_next_prompt("Custom prompt")
        
        session = orchestrator.state_manager.session
        assert session.next_prompt == "Custom prompt"
        assert session.current_prompt_source == "human_override"


class TestOrchestratorHelpers:
    """Tests for orchestrator helper methods."""
    
    @pytest.fixture
    def orchestrator(self):
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        return Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
    
    def test_build_history_summary_no_session(self, orchestrator):
        """Test building history summary without session."""
        result = orchestrator._build_history_summary()
        assert result is None
    
    def test_build_history_summary_no_history(self, orchestrator):
        """Test building history summary with no iterations."""
        orchestrator.state_manager.create_session("Test")
        
        result = orchestrator._build_history_summary()
        assert result is None
    
    def test_build_history_summary_with_history(self, orchestrator):
        """Test building history summary with iterations."""
        orchestrator.state_manager.create_session("Test")
        session = orchestrator.state_manager.session
        
        # Add mock iterations
        from copilot_agent.state import IterationRecord
        session.iteration_history = [
            IterationRecord(
                iteration_number=1,
                started_at="2024-01-01T00:00:00Z",
                prompt="Initial",
                prompt_source="initial",
                gemini_verdict="critique",
                gemini_feedback="Add tests",
            ),
            IterationRecord(
                iteration_number=2,
                started_at="2024-01-01T00:01:00Z",
                prompt="Add tests",
                prompt_source="gemini_followup",
                gemini_verdict="accept",
                gemini_feedback="Looks good",
            ),
        ]
        
        result = orchestrator._build_history_summary()
        
        assert "Iteration 1" in result
        assert "CRITIQUE" in result
        assert "Iteration 2" in result
        assert "ACCEPT" in result
    
    def test_check_repeated_critiques_first(self, orchestrator):
        """Test repeated critique check on first critique."""
        result = orchestrator._check_repeated_critiques("Add error handling")
        assert result is False
    
    def test_check_repeated_critiques_different(self, orchestrator):
        """Test repeated critique check with different critiques."""
        orchestrator._check_repeated_critiques("Add error handling")
        orchestrator._check_repeated_critiques("Add tests")
        result = orchestrator._check_repeated_critiques("Add docs")
        
        assert result is False
    
    def test_check_repeated_critiques_same(self, orchestrator):
        """Test repeated critique check with same critiques."""
        orchestrator.config.reviewer.stop_on_repeated_critiques = 3
        
        orchestrator._check_repeated_critiques("Fix the bug")
        orchestrator._check_repeated_critiques("Fix the bug")
        result = orchestrator._check_repeated_critiques("Fix the bug")
        
        assert result is True


class TestOrchestratorPerceptionActions:
    """Tests for perception and action properties."""
    
    @pytest.fixture
    def orchestrator(self):
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        return Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
    
    def test_perception_lazy_init(self, orchestrator):
        """Test perception is lazily initialized."""
        assert orchestrator._perception is None
        
        # Access property
        perception = orchestrator.perception
        
        assert perception is not None
        assert orchestrator._perception is not None
    
    def test_actions_lazy_init(self, orchestrator):
        """Test actions is lazily initialized."""
        assert orchestrator._actions is None
        
        # Access property
        actions = orchestrator.actions
        
        assert actions is not None
        assert orchestrator._actions is not None
        assert actions.dry_run is True  # Should inherit dry_run


class TestOrchestratorRun:
    """Tests for main run method."""
    
    @pytest.fixture
    def orchestrator(self):
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        return Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
    
    def test_run_no_session(self, orchestrator):
        """Test run fails without session."""
        async def _test():
            with pytest.raises(RuntimeError, match="No active session"):
                await orchestrator.run()
        run_async(_test())
    
    def test_run_kill_switch(self, orchestrator):
        """Test run stops on kill switch."""
        async def _test():
            orchestrator.state_manager.create_session("Test task")
            orchestrator.kill_switch.trigger()
            
            await orchestrator.run()
            
            # Should have stopped immediately
            session = orchestrator.state_manager.session
            assert session.phase in [SessionPhase.COMPLETE, SessionPhase.PROMPTING, SessionPhase.PAUSED]
        run_async(_test())
    
    def test_run_max_iterations(self, orchestrator):
        """Test run stops at max iterations."""
        async def _test():
            orchestrator.state_manager.create_session("Test task")
            orchestrator.config.reviewer.max_iterations = 0  # Immediate stop
            
            await orchestrator.run()
            
            session = orchestrator.state_manager.session
            assert session.completion_reason == "max_iterations"
        run_async(_test())


class TestOrchestratorIteration:
    """Tests for single iteration execution."""
    
    @pytest.fixture
    def orchestrator(self):
        config = AgentConfig()
        config.reviewer.response_wait_seconds = 0  # No wait for tests
        config.reviewer.response_stability_ms = 0
        
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        return Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
    
    def test_iteration_no_session(self, orchestrator):
        """Test iteration fails without session."""
        async def _test():
            result = await orchestrator._run_iteration()
            
            assert result.success is False
            assert result.stop_reason == "no_session"
        run_async(_test())
    
    def test_iteration_with_accept(self, orchestrator):
        """Test iteration with ACCEPT verdict."""
        async def _test():
            orchestrator.state_manager.create_session("Test task")
            
            # Mock perception capture (need to mock the internal capture method)
            mock_capture = Mock()
            mock_capture.success = True
            mock_capture.text = "print('hello')"
            mock_capture.method = CaptureMethod.OCR
            mock_capture.confidence = 0.9
            
            # Patch the internal capture to bypass retry wrapper
            async def mock_capture_with_retry():
                return mock_capture
            orchestrator._capture_with_retry = mock_capture_with_retry
            
            # Mock reviewer
            mock_review = ReviewResult(
                verdict=ReviewVerdict.ACCEPT,
                confidence="high",
                reasoning="Looks good",
                issues=[],
                follow_up_prompt=None,
            )
            
            async def mock_review_with_retry(**kwargs):
                return mock_review
            orchestrator._review_with_retry = mock_review_with_retry
            
            result = await orchestrator._run_iteration()
            
            assert result.success is True
            assert result.verdict == ReviewVerdict.ACCEPT
            assert result.should_continue is False
            assert result.stop_reason == "accepted"
        run_async(_test())
    
    def test_iteration_with_critique(self, orchestrator):
        """Test iteration with CRITIQUE verdict."""
        async def _test():
            orchestrator.state_manager.create_session("Test task")
            orchestrator.config.reviewer.pause_before_send = False  # Skip pause
            
            # Mock perception capture
            mock_capture = Mock()
            mock_capture.success = True
            mock_capture.text = "def foo(): pass"
            mock_capture.method = CaptureMethod.OCR
            mock_capture.confidence = 0.85
            
            async def mock_capture_with_retry():
                return mock_capture
            orchestrator._capture_with_retry = mock_capture_with_retry
            
            # Mock reviewer
            mock_review = ReviewResult(
                verdict=ReviewVerdict.CRITIQUE,
                confidence="medium",
                reasoning="Missing docstring",
                issues=["No docstring"],
                follow_up_prompt="Add a docstring",
            )
            
            async def mock_review_with_retry(**kwargs):
                return mock_review
            orchestrator._review_with_retry = mock_review_with_retry
            
            result = await orchestrator._run_iteration()
            
            assert result.success is True
            assert result.verdict == ReviewVerdict.CRITIQUE
            assert result.should_continue is True
            assert result.feedback == "Add a docstring"
        run_async(_test())
    
    def test_iteration_capture_failure(self, orchestrator):
        """Test iteration with capture failure."""
        async def _test():
            orchestrator.state_manager.create_session("Test task")
            
            # Mock failed capture
            mock_capture = Mock()
            mock_capture.success = False
            mock_capture.error = "Screenshot failed"
            mock_capture.method = CaptureMethod.OCR
            
            async def mock_capture_with_retry():
                return mock_capture
            orchestrator._capture_with_retry = mock_capture_with_retry
            
            result = await orchestrator._run_iteration()
            
            assert result.success is False
            assert result.should_continue is True  # Will retry
        run_async(_test())


class TestOrchestratorM6Features:
    """Tests for M6 production features."""
    
    @pytest.fixture
    def orchestrator(self):
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        return Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
    
    def test_get_circuit_status(self, orchestrator):
        """Test circuit breaker status."""
        status = orchestrator.get_circuit_status()
        
        assert "reviewer" in status
        assert "vision" in status
        assert "ui" in status
        assert status["reviewer"] == "closed"
    
    def test_metrics_accessor(self, orchestrator):
        """Test metrics accessor."""
        metrics = orchestrator.metrics
        assert metrics is not None
    
    def test_budget_accessor(self, orchestrator):
        """Test budget accessor."""
        budget = orchestrator.budget
        assert budget is not None
        assert budget.max_reviewer_calls > 0
    
    def test_get_stats(self, orchestrator):
        """Test stats display."""
        stats = orchestrator.get_stats()
        assert isinstance(stats, str)
        assert "Session" in stats
    
    def test_checkpointer_accessor(self, orchestrator):
        """Test checkpointer accessor returns None when not configured."""
        assert orchestrator.checkpointer is None
    
    def test_get_resume_info_no_checkpointer(self, orchestrator):
        """Test resume info when no checkpointer."""
        info = orchestrator.get_resume_info()
        assert info is None
