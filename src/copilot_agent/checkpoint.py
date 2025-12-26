"""
Atomic checkpointing and resume support.

Provides step-level persistence for crash recovery and session resume.
Each step (prompt sent, capture done, review received) is atomically saved.
"""

import json
import os
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Any, List
from filelock import FileLock, Timeout

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


def utc_now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StepType(str, Enum):
    """Types of atomic steps."""
    
    SESSION_START = "session_start"
    PROMPT_SENT = "prompt_sent"
    WAITING_RESPONSE = "waiting_response"
    CAPTURE_STARTED = "capture_started"
    CAPTURE_COMPLETE = "capture_complete"
    REVIEW_STARTED = "review_started"
    REVIEW_COMPLETE = "review_complete"
    FEEDBACK_READY = "feedback_ready"
    ITERATION_COMPLETE = "iteration_complete"
    SESSION_COMPLETE = "session_complete"
    SESSION_PAUSED = "session_paused"
    SESSION_ABORTED = "session_aborted"


class StepStatus(str, Enum):
    """Status of a step."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepRecord:
    """Record of a single atomic step."""
    
    step_id: int
    step_type: StepType
    status: StepStatus
    started_at: str
    iteration: int
    data: dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["step_type"] = self.step_type.value
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        """Create from dictionary."""
        data["step_type"] = StepType(data["step_type"])
        data["status"] = StepStatus(data["status"])
        return cls(**data)


@dataclass
class CheckpointState:
    """Complete checkpoint state for resume."""
    
    # Session identity
    session_id: str
    task: str
    created_at: str
    
    # Progress
    current_iteration: int = 0
    max_iterations: int = 10
    total_steps: int = 0
    
    # Current step
    last_completed_step: Optional[int] = None
    last_step_type: Optional[StepType] = None
    
    # Step history
    steps: List[StepRecord] = field(default_factory=list)
    
    # Data cache (for resume)
    current_prompt: Optional[str] = None
    current_response: Optional[str] = None
    current_verdict: Optional[str] = None
    current_feedback: Optional[str] = None
    next_prompt: Optional[str] = None
    
    # Metrics
    reviewer_calls: int = 0
    vision_calls: int = 0
    capture_attempts: int = 0
    errors: int = 0
    
    # Resume info
    resumable: bool = True
    resume_from_step: Optional[int] = None
    resume_reason: Optional[str] = None
    
    # Timestamps
    last_checkpoint_at: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        if self.last_step_type:
            d["last_step_type"] = self.last_step_type.value
        d["steps"] = [s.to_dict() for s in self.steps]
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        if data.get("last_step_type"):
            data["last_step_type"] = StepType(data["last_step_type"])
        if data.get("steps"):
            data["steps"] = [StepRecord.from_dict(s) for s in data["steps"]]
        return cls(**data)


class AtomicCheckpointer:
    """
    Atomic checkpointing system.
    
    Writes checkpoint files atomically using write-to-temp-then-rename pattern.
    Uses file locks for concurrent access safety.
    """
    
    CHECKPOINT_FILE = "checkpoint.json"
    LOCK_FILE = "checkpoint.lock"
    STEPS_FILE = "steps.jsonl"
    LOCK_TIMEOUT = 5.0  # seconds
    
    def __init__(self, session_path: Path):
        """
        Initialize checkpointer for a session.
        
        Args:
            session_path: Path to session directory
        """
        self.session_path = Path(session_path)
        self.checkpoint_path = self.session_path / self.CHECKPOINT_FILE
        self.lock_path = self.session_path / self.LOCK_FILE
        self.steps_path = self.session_path / self.STEPS_FILE
        
        # Ensure session directory exists
        self.session_path.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self._state: Optional[CheckpointState] = None
        self._step_counter = 0
    
    @property
    def state(self) -> Optional[CheckpointState]:
        """Get current checkpoint state."""
        return self._state
    
    def initialize(
        self,
        session_id: str,
        task: str,
        max_iterations: int = 10,
    ) -> CheckpointState:
        """
        Initialize a new checkpoint state.
        
        Args:
            session_id: Unique session identifier
            task: Task description
            max_iterations: Maximum iterations allowed
            
        Returns:
            New checkpoint state
        """
        self._state = CheckpointState(
            session_id=session_id,
            task=task,
            created_at=utc_now_iso(),
            max_iterations=max_iterations,
        )
        self._step_counter = 0
        
        # Write initial checkpoint
        self._atomic_write()
        
        logger.info(
            "Checkpoint initialized",
            session_id=session_id,
            path=str(self.checkpoint_path),
        )
        
        return self._state
    
    def load(self) -> Optional[CheckpointState]:
        """
        Load checkpoint state from disk.
        
        Returns:
            Loaded state or None if not found
        """
        if not self.checkpoint_path.exists():
            logger.debug("No checkpoint file found", path=str(self.checkpoint_path))
            return None
        
        try:
            with self._acquire_lock():
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                
                self._state = CheckpointState.from_dict(data)
                self._step_counter = self._state.total_steps
                
                logger.info(
                    "Checkpoint loaded",
                    session_id=self._state.session_id,
                    iteration=self._state.current_iteration,
                    last_step=self._state.last_step_type.value if self._state.last_step_type else None,
                )
                
                return self._state
                
        except Exception as e:
            logger.error("Failed to load checkpoint", error=str(e))
            return None
    
    def record_step(
        self,
        step_type: StepType,
        data: Optional[dict[str, Any]] = None,
        status: StepStatus = StepStatus.COMPLETE,
        error: Optional[str] = None,
    ) -> StepRecord:
        """
        Record an atomic step.
        
        Args:
            step_type: Type of step
            data: Optional data associated with step
            status: Step completion status
            error: Error message if failed
            
        Returns:
            The recorded step
        """
        if not self._state:
            raise RuntimeError("Checkpoint not initialized")
        
        self._step_counter += 1
        now = utc_now_iso()
        
        step = StepRecord(
            step_id=self._step_counter,
            step_type=step_type,
            status=status,
            started_at=now,
            completed_at=now if status == StepStatus.COMPLETE else None,
            iteration=self._state.current_iteration,
            data=data or {},
            error=error,
        )
        
        # Update state
        self._state.steps.append(step)
        self._state.total_steps = self._step_counter
        self._state.last_step_type = step_type
        
        if status == StepStatus.COMPLETE:
            self._state.last_completed_step = step.step_id
        
        if status == StepStatus.FAILED:
            self._state.errors += 1
        
        # Update resume info based on step type
        self._update_resume_info(step)
        
        # Write atomically
        self._atomic_write()
        self._append_step_log(step)
        
        logger.debug(
            "Step recorded",
            step_id=step.step_id,
            step_type=step_type.value,
            status=status.value,
        )
        
        return step
    
    def start_step(
        self,
        step_type: StepType,
        data: Optional[dict[str, Any]] = None,
    ) -> StepRecord:
        """
        Start a step (mark as in-progress).
        
        Args:
            step_type: Type of step
            data: Optional data associated with step
            
        Returns:
            The started step
        """
        return self.record_step(
            step_type=step_type,
            data=data,
            status=StepStatus.IN_PROGRESS,
        )
    
    def complete_step(
        self,
        step_id: int,
        data: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Complete a previously started step.
        
        Args:
            step_id: ID of step to complete
            data: Additional data to merge
            error: Error message if failed
        """
        if not self._state:
            raise RuntimeError("Checkpoint not initialized")
        
        # Find the step
        step = next((s for s in self._state.steps if s.step_id == step_id), None)
        if not step:
            logger.warning("Step not found for completion", step_id=step_id)
            return
        
        # Update step
        now = utc_now_iso()
        step.completed_at = now
        step.status = StepStatus.FAILED if error else StepStatus.COMPLETE
        step.error = error
        
        if data:
            step.data.update(data)
        
        # Calculate duration
        if step.started_at:
            try:
                start = datetime.fromisoformat(step.started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(now.replace("Z", "+00:00"))
                step.duration_ms = int((end - start).total_seconds() * 1000)
            except Exception:
                pass
        
        # Update state
        if step.status == StepStatus.COMPLETE:
            self._state.last_completed_step = step_id
        if step.status == StepStatus.FAILED:
            self._state.errors += 1
        
        # Write atomically
        self._atomic_write()
        self._append_step_log(step)
    
    def update_data(self, **kwargs: Any) -> None:
        """
        Update checkpoint data fields.
        
        Args:
            **kwargs: Fields to update
        """
        if not self._state:
            raise RuntimeError("Checkpoint not initialized")
        
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        
        self._atomic_write()
    
    def start_iteration(self, prompt: str) -> None:
        """
        Start a new iteration.
        
        Args:
            prompt: The prompt for this iteration
        """
        if not self._state:
            raise RuntimeError("Checkpoint not initialized")
        
        self._state.current_iteration += 1
        self._state.current_prompt = prompt
        self._state.current_response = None
        self._state.current_verdict = None
        self._state.current_feedback = None
        self._state.next_prompt = None
        
        self.record_step(
            StepType.PROMPT_SENT,
            data={"prompt": prompt[:500], "iteration": self._state.current_iteration},
        )
    
    def record_capture(self, response: str, method: str) -> None:
        """
        Record a captured response.
        
        Args:
            response: Captured response text
            method: Capture method used
        """
        if not self._state:
            raise RuntimeError("Checkpoint not initialized")
        
        self._state.current_response = response
        self._state.capture_attempts += 1
        
        self.record_step(
            StepType.CAPTURE_COMPLETE,
            data={
                "method": method,
                "length": len(response),
                "preview": response[:200],
            },
        )
    
    def record_review(
        self,
        verdict: str,
        feedback: str,
        confidence: str,
        next_prompt: Optional[str] = None,
    ) -> None:
        """
        Record a review result.
        
        Args:
            verdict: Review verdict
            feedback: Review feedback
            confidence: Confidence level
            next_prompt: Follow-up prompt if any
        """
        if not self._state:
            raise RuntimeError("Checkpoint not initialized")
        
        self._state.current_verdict = verdict
        self._state.current_feedback = feedback
        self._state.next_prompt = next_prompt
        self._state.reviewer_calls += 1
        
        self.record_step(
            StepType.REVIEW_COMPLETE,
            data={
                "verdict": verdict,
                "confidence": confidence,
                "has_followup": next_prompt is not None,
            },
        )
    
    def complete_iteration(self) -> None:
        """Mark current iteration as complete."""
        self.record_step(
            StepType.ITERATION_COMPLETE,
            data={"iteration": self._state.current_iteration if self._state else 0},
        )
    
    def mark_complete(self, reason: str) -> None:
        """
        Mark session as complete.
        
        Args:
            reason: Completion reason
        """
        if not self._state:
            return
        
        self._state.resumable = False
        self.record_step(
            StepType.SESSION_COMPLETE,
            data={"reason": reason},
        )
    
    def mark_paused(self, reason: str) -> None:
        """
        Mark session as paused (resumable).
        
        Args:
            reason: Pause reason
        """
        if not self._state:
            return
        
        self._state.resumable = True
        self._state.resume_reason = reason
        self.record_step(
            StepType.SESSION_PAUSED,
            data={"reason": reason},
        )
    
    def mark_aborted(self, reason: str) -> None:
        """
        Mark session as aborted.
        
        Args:
            reason: Abort reason
        """
        if not self._state:
            return
        
        self._state.resumable = False
        self.record_step(
            StepType.SESSION_ABORTED,
            data={"reason": reason},
        )
    
    def get_resume_point(self) -> Optional[dict[str, Any]]:
        """
        Get resume point information.
        
        Returns:
            Dict with resume info or None if not resumable
        """
        if not self._state or not self._state.resumable:
            return None
        
        # Find the last completed step
        last_step = self._state.last_step_type
        
        resume_info = {
            "session_id": self._state.session_id,
            "iteration": self._state.current_iteration,
            "last_step": last_step.value if last_step else None,
            "current_prompt": self._state.current_prompt,
            "current_response": self._state.current_response,
            "current_verdict": self._state.current_verdict,
            "next_prompt": self._state.next_prompt,
        }
        
        # Determine what to resume
        if last_step == StepType.PROMPT_SENT:
            resume_info["resume_action"] = "wait_for_response"
        elif last_step == StepType.WAITING_RESPONSE:
            resume_info["resume_action"] = "capture_response"
        elif last_step == StepType.CAPTURE_COMPLETE:
            resume_info["resume_action"] = "review_response"
        elif last_step == StepType.REVIEW_COMPLETE:
            if self._state.next_prompt:
                resume_info["resume_action"] = "send_followup"
            else:
                resume_info["resume_action"] = "complete"
        elif last_step == StepType.FEEDBACK_READY:
            resume_info["resume_action"] = "send_followup"
        elif last_step == StepType.ITERATION_COMPLETE:
            if self._state.current_iteration < self._state.max_iterations:
                resume_info["resume_action"] = "start_next_iteration"
            else:
                resume_info["resume_action"] = "complete"
        else:
            resume_info["resume_action"] = "start_fresh"
        
        return resume_info
    
    def _update_resume_info(self, step: StepRecord) -> None:
        """Update resume info based on step."""
        if not self._state:
            return
        
        # Most steps are resumable
        self._state.resumable = step.status != StepStatus.FAILED
        
        # Some step types mark specific resume points
        if step.step_type in (StepType.SESSION_COMPLETE, StepType.SESSION_ABORTED):
            self._state.resumable = False
    
    def _atomic_write(self) -> None:
        """Atomically write checkpoint to disk."""
        if not self._state:
            return
        
        self._state.last_checkpoint_at = utc_now_iso()
        
        try:
            with self._acquire_lock():
                # Write to temp file first
                fd, temp_path = tempfile.mkstemp(
                    suffix=".json",
                    prefix="checkpoint_",
                    dir=str(self.session_path),
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(self._state.to_dict(), f, indent=2)
                    
                    # Atomic rename
                    shutil.move(temp_path, self.checkpoint_path)
                    
                except Exception:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
                    
        except Timeout:
            logger.warning("Lock timeout during checkpoint write")
        except Exception as e:
            logger.error("Failed to write checkpoint", error=str(e))
    
    def _append_step_log(self, step: StepRecord) -> None:
        """Append step to JSONL log."""
        try:
            with open(self.steps_path, "a") as f:
                f.write(json.dumps(step.to_dict()) + "\n")
        except Exception as e:
            logger.warning("Failed to append step log", error=str(e))
    
    def _acquire_lock(self) -> FileLock:
        """Acquire file lock."""
        lock = FileLock(self.lock_path, timeout=self.LOCK_TIMEOUT)
        lock.acquire()
        return lock


def list_resumable_sessions(sessions_path: Path) -> List[dict[str, Any]]:
    """
    List all resumable sessions.
    
    Args:
        sessions_path: Path to sessions directory
        
    Returns:
        List of resumable session info dicts
    """
    resumable = []
    
    if not sessions_path.exists():
        return resumable
    
    for session_dir in sessions_path.iterdir():
        if not session_dir.is_dir():
            continue
        
        checkpoint_path = session_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            
            if data.get("resumable", False):
                resumable.append({
                    "session_id": data.get("session_id"),
                    "task": data.get("task", "")[:50],
                    "iteration": data.get("current_iteration", 0),
                    "last_step": data.get("last_step_type"),
                    "created_at": data.get("created_at"),
                    "last_checkpoint_at": data.get("last_checkpoint_at"),
                    "path": str(session_dir),
                })
        except Exception:
            continue
    
    return resumable
