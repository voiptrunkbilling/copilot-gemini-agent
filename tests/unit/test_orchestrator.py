"""
Unit tests for M4: Orchestrator.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from copilot_agent.orchestrator import Orchestrator, IterationResult
from copilot_agent.config import AgentConfig
from copilot_agent.state import StateManager, SessionPhase, GeminiVerdict
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.reviewer.gemini import ReviewVerdict, ReviewResult


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
    
    @pytest.mark.asyncio
    async def test_run_no_session(self, orchestrator):
        """Test run fails without session."""
        with pytest.raises(RuntimeError, match="No active session"):
            await orchestrator.run()
    
    @pytest.mark.asyncio
    async def test_run_kill_switch(self, orchestrator):
        """Test run stops on kill switch."""
        orchestrator.state_manager.create_session("Test task")
        orchestrator.kill_switch.trigger()
        
        await orchestrator.run()
        
        # Should have stopped immediately
        session = orchestrator.state_manager.session
        assert session.phase in [SessionPhase.COMPLETE, SessionPhase.PROMPTING]
    
    @pytest.mark.asyncio
    async def test_run_max_iterations(self, orchestrator):
        """Test run stops at max iterations."""
        orchestrator.state_manager.create_session("Test task")
        orchestrator.config.reviewer.max_iterations = 0  # Immediate stop
        
        await orchestrator.run()
        
        session = orchestrator.state_manager.session
        assert session.completion_reason == "max_iterations"


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
    
    @pytest.mark.asyncio
    async def test_iteration_no_session(self, orchestrator):
        """Test iteration fails without session."""
        result = await orchestrator._run_iteration()
        
        assert result.success is False
        assert result.stop_reason == "no_session"
    
    @pytest.mark.asyncio
    async def test_iteration_with_accept(self, orchestrator):
        """Test iteration with ACCEPT verdict."""
        orchestrator.state_manager.create_session("Test task")
        
        # Mock perception
        mock_capture = Mock()
        mock_capture.success = True
        mock_capture.text = "print('hello')"
        mock_capture.method = Mock()
        mock_capture.method.value = "ocr"
        mock_capture.confidence = 0.9
        
        mock_perception = Mock()
        mock_perception.capture_copilot_response = Mock(return_value=mock_capture)
        orchestrator._perception = mock_perception
        
        # Mock reviewer
        mock_review = ReviewResult(
            verdict=ReviewVerdict.ACCEPT,
            confidence="high",
            reasoning="Looks good",
            issues=[],
            follow_up_prompt=None,
        )
        orchestrator.reviewer.review = AsyncMock(return_value=mock_review)
        
        result = await orchestrator._run_iteration()
        
        assert result.success is True
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.should_continue is False
        assert result.stop_reason == "accepted"
    
    @pytest.mark.asyncio
    async def test_iteration_with_critique(self, orchestrator):
        """Test iteration with CRITIQUE verdict."""
        orchestrator.state_manager.create_session("Test task")
        orchestrator.config.reviewer.pause_before_send = False  # Skip pause
        
        # Mock perception
        mock_capture = Mock()
        mock_capture.success = True
        mock_capture.text = "def foo(): pass"
        mock_capture.method = Mock()
        mock_capture.method.value = "ocr"
        mock_capture.confidence = 0.85
        
        mock_perception = Mock()
        mock_perception.capture_copilot_response = Mock(return_value=mock_capture)
        orchestrator._perception = mock_perception
        
        # Mock reviewer
        mock_review = ReviewResult(
            verdict=ReviewVerdict.CRITIQUE,
            confidence="medium",
            reasoning="Missing docstring",
            issues=["No docstring"],
            follow_up_prompt="Add a docstring",
        )
        orchestrator.reviewer.review = AsyncMock(return_value=mock_review)
        
        result = await orchestrator._run_iteration()
        
        assert result.success is True
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert result.should_continue is True
        assert result.feedback == "Add a docstring"
    
    @pytest.mark.asyncio
    async def test_iteration_capture_failure(self, orchestrator):
        """Test iteration with capture failure."""
        orchestrator.state_manager.create_session("Test task")
        
        # Mock failed capture
        mock_capture = Mock()
        mock_capture.success = False
        mock_capture.error = "Screenshot failed"
        
        mock_perception = Mock()
        mock_perception.capture_copilot_response = Mock(return_value=mock_capture)
        orchestrator._perception = mock_perception
        
        result = await orchestrator._run_iteration()
        
        assert result.success is False
        assert result.should_continue is True  # Will retry
