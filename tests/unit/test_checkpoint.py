"""
Tests for checkpoint module.
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from copilot_agent.checkpoint import (
    AtomicCheckpointer,
    CheckpointState,
    StepType,
    StepStatus,
    StepRecord,
    utc_now_iso,
)


class TestCheckpointState:
    """Tests for CheckpointState."""
    
    def test_create_state(self):
        """Test creating checkpoint state."""
        state = CheckpointState(
            session_id="test-123",
            task="Test task",
            created_at=utc_now_iso(),
        )
        
        assert state.session_id == "test-123"
        assert state.task == "Test task"
        assert state.current_iteration == 0
        assert state.total_steps == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = CheckpointState(
            session_id="test-123",
            task="Test task",
            created_at="2024-01-01T00:00:00Z",
            current_iteration=5,
        )
        
        d = state.to_dict()
        
        assert d["session_id"] == "test-123"
        assert d["task"] == "Test task"
        assert d["current_iteration"] == 5
    
    def test_from_dict(self):
        """Test loading from dictionary."""
        data = {
            "session_id": "test-123",
            "task": "Loaded task",
            "created_at": "2024-01-01T00:00:00Z",
            "current_iteration": 10,
            "max_iterations": 20,
            "total_steps": 5,
        }
        
        state = CheckpointState.from_dict(data)
        
        assert state.session_id == "test-123"
        assert state.task == "Loaded task"
        assert state.current_iteration == 10


class TestStepRecord:
    """Tests for StepRecord."""
    
    def test_create_step(self):
        """Test creating a step record."""
        step = StepRecord(
            step_id=1,
            step_type=StepType.SESSION_START,
            status=StepStatus.COMPLETE,
            started_at=utc_now_iso(),
            iteration=0,
        )
        
        assert step.step_type == StepType.SESSION_START
        assert step.status == StepStatus.COMPLETE
        assert step.step_id == 1
    
    def test_step_to_dict(self):
        """Test step serialization."""
        step = StepRecord(
            step_id=3,
            step_type=StepType.REVIEW_COMPLETE,
            status=StepStatus.COMPLETE,
            started_at="2024-01-01T00:00:00Z",
            iteration=3,
            data={"verdict": "ACCEPT"},
        )
        
        d = step.to_dict()
        
        assert d["step_type"] == "review_complete"
        assert d["status"] == "complete"
        assert d["iteration"] == 3
        assert d["data"]["verdict"] == "ACCEPT"


class TestAtomicCheckpointer:
    """Tests for AtomicCheckpointer."""
    
    @pytest.fixture
    def checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_init_creates_dir(self, checkpoint_dir):
        """Test initializing creates session directory."""
        session_path = checkpoint_dir / "session-1"
        cp = AtomicCheckpointer(session_path=session_path)
        
        assert session_path.exists()
    
    def test_initialize_state(self, checkpoint_dir):
        """Test initializing new state."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        state = cp.initialize(
            session_id="test-session",
            task="Test task",
        )
        
        assert state.session_id == "test-session"
        assert state.task == "Test task"
        assert state.current_iteration == 0
        
        # Checkpoint file should exist
        assert (checkpoint_dir / "checkpoint.json").exists()
    
    def test_load_state(self, checkpoint_dir):
        """Test loading checkpoint state."""
        # Create and save
        cp1 = AtomicCheckpointer(session_path=checkpoint_dir)
        cp1.initialize(session_id="test-session", task="Test task")
        cp1.update_data(current_iteration=5)
        
        # Load in new instance
        cp2 = AtomicCheckpointer(session_path=checkpoint_dir)
        state = cp2.load()
        
        assert state.task == "Test task"
        assert state.current_iteration == 5
    
    def test_record_step(self, checkpoint_dir):
        """Test recording a step."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        
        step = cp.record_step(StepType.SESSION_START)
        
        assert step.step_type == StepType.SESSION_START
        assert step.status == StepStatus.COMPLETE
        assert cp.state.total_steps == 1
        
        # Check step log file
        step_log = checkpoint_dir / "steps.jsonl"
        assert step_log.exists()
    
    def test_start_and_complete_step(self, checkpoint_dir):
        """Test step lifecycle."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        
        # Start step
        step = cp.start_step(StepType.CAPTURE_STARTED, data={"method": "mss"})
        assert step.status == StepStatus.IN_PROGRESS
        
        # Complete step
        cp.complete_step(step.step_id, data={"size": 1024})
        
        # Check the step was updated
        updated = next(s for s in cp.state.steps if s.step_id == step.step_id)
        assert updated.status == StepStatus.COMPLETE
        assert updated.data["size"] == 1024
    
    def test_complete_step_with_error(self, checkpoint_dir):
        """Test completing a step with error."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        
        step = cp.start_step(StepType.REVIEW_STARTED)
        cp.complete_step(step.step_id, error="API error")
        
        updated = next(s for s in cp.state.steps if s.step_id == step.step_id)
        assert updated.status == StepStatus.FAILED
        assert "API error" in updated.error
    
    def test_start_iteration(self, checkpoint_dir):
        """Test starting an iteration."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        
        cp.start_iteration(prompt="Test prompt")
        
        assert cp.state.current_iteration == 1
        assert cp.state.current_prompt == "Test prompt"
    
    def test_record_capture(self, checkpoint_dir):
        """Test recording capture."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        cp.start_iteration(prompt="Test")
        
        cp.record_capture(response="Response text", method="clipboard")
        
        assert cp.state.current_response == "Response text"
        assert cp.state.capture_attempts == 1
    
    def test_record_review(self, checkpoint_dir):
        """Test recording review."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        cp.start_iteration(prompt="Test")
        
        cp.record_review(
            verdict="ACCEPT",
            feedback="Looks good",
            confidence="high",
        )
        
        assert cp.state.current_verdict == "ACCEPT"
        assert cp.state.reviewer_calls == 1
    
    def test_mark_paused(self, checkpoint_dir):
        """Test marking session as paused."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        
        cp.mark_paused(reason="User requested pause")
        
        assert cp.state.resumable is True
        assert cp.state.resume_reason == "User requested pause"
    
    def test_get_resume_point_not_resumable(self, checkpoint_dir):
        """Test resume point when marked complete (not resumable)."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        cp.mark_complete(reason="Task finished")
        
        resume = cp.get_resume_point()
        
        assert resume is None
    
    def test_get_resume_point_resumable(self, checkpoint_dir):
        """Test resume point when resumable."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        cp.start_iteration(prompt="Test")
        cp.record_capture(response="Response", method="mss")
        cp.mark_paused(reason="Kill switch")
        
        resume = cp.get_resume_point()
        
        assert resume is not None
        assert resume["session_id"] == "test-session"
        assert resume["iteration"] == 1
    
    def test_update_data(self, checkpoint_dir):
        """Test updating checkpoint data."""
        cp = AtomicCheckpointer(session_path=checkpoint_dir)
        cp.initialize(session_id="test-session", task="Test task")
        
        cp.update_data(current_iteration=5, max_iterations=20)
        
        assert cp.state.current_iteration == 5
        assert cp.state.max_iterations == 20
