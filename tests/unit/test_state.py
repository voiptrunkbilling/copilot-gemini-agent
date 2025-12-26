"""
Unit tests for state management module.
"""

import pytest
import tempfile
import json
from pathlib import Path

from copilot_agent.config import AgentConfig, StorageConfig
from copilot_agent.state import (
    StateManager,
    Session,
    SessionPhase,
    GeminiVerdict,
    IterationRecord,
    CalibrationData,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config(temp_storage):
    """Create config with temporary storage."""
    return AgentConfig(
        storage=StorageConfig(base_path=temp_storage)
    )


@pytest.fixture
def state_manager(config):
    """Create state manager with temp storage."""
    return StateManager(config)


class TestSession:
    """Tests for Session dataclass."""
    
    def test_to_dict(self):
        session = Session(
            session_id="test123",
            created_at="2024-01-01T00:00:00Z",
            task_description="Test task",
            phase=SessionPhase.IDLE,
        )
        
        data = session.to_dict()
        
        assert data["session_id"] == "test123"
        assert data["phase"] == "idle"
        assert data["task_description"] == "Test task"
    
    def test_from_dict(self):
        data = {
            "session_id": "test456",
            "created_at": "2024-01-01T00:00:00Z",
            "task_description": "Test task",
            "phase": "reviewing",
            "iteration_count": 3,
            "max_iterations": 20,
            "current_prompt": None,
            "current_prompt_source": "initial",
            "current_response": None,
            "current_verdict": "critique",
            "current_feedback": "Needs work",
            "next_prompt": None,
            "iteration_history": [],
            "calibration": {},
            "consecutive_errors": 0,
            "total_errors": 0,
            "last_error": None,
            "final_result": None,
            "completion_reason": None,
            "started_at": None,
            "last_activity_at": None,
            "total_duration_ms": 0,
            "vision_calls_count": 0,
            "capture_attempts": 0,
            "capture_successes": 0,
        }
        
        session = Session.from_dict(data)
        
        assert session.session_id == "test456"
        assert session.phase == SessionPhase.REVIEWING
        assert session.current_verdict == GeminiVerdict.CRITIQUE


class TestStateManager:
    """Tests for StateManager."""
    
    def test_create_session(self, state_manager):
        session = state_manager.create_session(task="Test task")
        
        assert session.session_id is not None
        assert len(session.session_id) == 8
        assert session.task_description == "Test task"
        assert session.phase == SessionPhase.IDLE
    
    def test_session_directory_created(self, state_manager, config):
        session = state_manager.create_session(task="Test task")
        
        session_dir = config.sessions_path / session.session_id
        assert session_dir.exists()
        assert (session_dir / "screenshots").exists()
        assert (session_dir / "captures").exists()
    
    def test_checkpoint_created(self, state_manager, config):
        session = state_manager.create_session(task="Test task")
        
        checkpoint_path = config.sessions_path / session.session_id / "checkpoint.json"
        assert checkpoint_path.exists()
        
        with open(checkpoint_path) as f:
            data = json.load(f)
        assert data["session_id"] == session.session_id
    
    def test_transition_to(self, state_manager):
        state_manager.create_session(task="Test task")
        
        state_manager.transition_to(SessionPhase.PROMPTING)
        assert state_manager.session.phase == SessionPhase.PROMPTING
        
        state_manager.transition_to(SessionPhase.WAITING)
        assert state_manager.session.phase == SessionPhase.WAITING
    
    def test_record_error(self, state_manager):
        state_manager.create_session(task="Test task")
        
        state_manager.record_error("test_error", "Test message", recoverable=True)
        
        assert state_manager.session.total_errors == 1
        assert state_manager.session.consecutive_errors == 1
        assert state_manager.session.last_error.error_type == "test_error"
    
    def test_clear_consecutive_errors(self, state_manager):
        state_manager.create_session(task="Test task")
        
        state_manager.record_error("error1", "Error 1")
        state_manager.record_error("error2", "Error 2")
        assert state_manager.session.consecutive_errors == 2
        
        state_manager.clear_consecutive_errors()
        assert state_manager.session.consecutive_errors == 0
        assert state_manager.session.total_errors == 2  # Total unchanged
    
    def test_start_iteration(self, state_manager):
        state_manager.create_session(task="Test task")
        
        state_manager.start_iteration(prompt="Fix the bug", source="initial")
        
        assert state_manager.session.iteration_count == 1
        assert state_manager.session.current_prompt == "Fix the bug"
        assert state_manager.session.current_prompt_source == "initial"
    
    def test_record_response(self, state_manager):
        state_manager.create_session(task="Test task")
        state_manager.start_iteration(prompt="Test")
        
        state_manager.record_response("Here is the fix...", "copy_button")
        
        assert state_manager.session.current_response == "Here is the fix..."
        assert state_manager.session.capture_successes == 1
    
    def test_record_verdict(self, state_manager):
        state_manager.create_session(task="Test task")
        state_manager.start_iteration(prompt="Test")
        
        state_manager.record_verdict(
            verdict=GeminiVerdict.ACCEPT,
            feedback="Looks good!",
            confidence="high",
        )
        
        assert state_manager.session.current_verdict == GeminiVerdict.ACCEPT
        assert state_manager.session.current_feedback == "Looks good!"
    
    def test_complete_iteration(self, state_manager):
        state_manager.create_session(task="Test task")
        state_manager.start_iteration(prompt="Test prompt")
        state_manager.record_response("Test response", "ocr")
        state_manager.record_verdict(
            GeminiVerdict.CRITIQUE,
            "Needs improvement",
            "medium",
            "Please fix X",
        )
        
        state_manager.complete_iteration()
        
        assert len(state_manager.session.iteration_history) == 1
        record = state_manager.session.iteration_history[0]
        assert record.prompt == "Test prompt"
        assert record.copilot_response == "Test response"
        assert record.gemini_verdict == "critique"
    
    def test_load_session(self, state_manager, config):
        # Create and save session
        session = state_manager.create_session(task="Test task")
        session_id = session.session_id
        state_manager.start_iteration("Test")
        state_manager.checkpoint()
        
        # Create new state manager and load
        new_manager = StateManager(config)
        loaded = new_manager.load_session(session_id)
        
        assert loaded.session_id == session_id
        assert loaded.task_description == "Test task"
        assert loaded.iteration_count == 1
    
    def test_audit_log(self, state_manager, config):
        session = state_manager.create_session(task="Test task")
        
        state_manager.append_audit_log("test_event", {"key": "value"})
        
        audit_path = config.sessions_path / session.session_id / "audit.jsonl"
        assert audit_path.exists()
        
        with open(audit_path) as f:
            line = f.readline()
            entry = json.loads(line)
        
        assert entry["event_type"] == "test_event"
        assert entry["details"]["key"] == "value"
