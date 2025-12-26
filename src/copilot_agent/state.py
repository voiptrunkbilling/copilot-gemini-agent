"""
State management and persistence.
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import dataclass, field, asdict

from copilot_agent.config import AgentConfig
from copilot_agent.logging import get_logger

logger = get_logger(__name__)


def utc_now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SessionPhase(str, Enum):
    """Session phase states."""
    
    IDLE = "idle"
    PROMPTING = "prompting"
    WAITING = "waiting"
    CAPTURING = "capturing"
    REVIEWING = "reviewing"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"


class GeminiVerdict(str, Enum):
    """Gemini reviewer verdicts."""
    
    ACCEPT = "accept"
    CRITIQUE = "critique"
    CLARIFY = "clarify"
    ERROR = "error"


@dataclass
class IterationRecord:
    """Record of a single iteration."""
    
    iteration_number: int
    started_at: str
    prompt: str
    prompt_source: str  # "initial", "gemini_followup", "human_override"
    copilot_response: Optional[str] = None
    capture_method: Optional[str] = None  # "copy_button", "select_copy", "ocr"
    gemini_verdict: Optional[str] = None
    gemini_feedback: Optional[str] = None
    gemini_confidence: Optional[str] = None
    duration_ms: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    completed_at: Optional[str] = None


@dataclass
class CalibrationData:
    """UI element calibration data."""
    
    input_field: Optional[tuple[int, int]] = None
    response_region_top_left: Optional[tuple[int, int]] = None
    response_region_bottom_right: Optional[tuple[int, int]] = None
    window_geometry: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
    calibrated_at: Optional[str] = None


@dataclass
class SessionError:
    """Error record."""
    
    error_type: str
    message: str
    timestamp: str
    recoverable: bool = True


@dataclass
class Session:
    """Complete session state."""
    
    # Identity
    session_id: str
    created_at: str
    task_description: str
    
    # Progress
    phase: SessionPhase = SessionPhase.IDLE
    iteration_count: int = 0
    max_iterations: int = 20
    
    # Current iteration
    current_prompt: Optional[str] = None
    current_prompt_source: str = "initial"
    current_response: Optional[str] = None
    current_verdict: Optional[GeminiVerdict] = None
    current_feedback: Optional[str] = None
    next_prompt: Optional[str] = None
    
    # History
    iteration_history: List[IterationRecord] = field(default_factory=list)
    
    # Calibration
    calibration: CalibrationData = field(default_factory=CalibrationData)
    
    # Error tracking
    consecutive_errors: int = 0
    total_errors: int = 0
    last_error: Optional[SessionError] = None
    
    # Outputs
    final_result: Optional[str] = None
    completion_reason: Optional[str] = None  # "gemini_accepted", "max_iterations", etc.
    
    # Timing
    started_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    total_duration_ms: int = 0
    
    # Metrics
    vision_calls_count: int = 0
    capture_attempts: int = 0
    capture_successes: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["phase"] = self.phase.value
        if self.current_verdict:
            data["current_verdict"] = self.current_verdict.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        # Convert strings back to enums
        data["phase"] = SessionPhase(data["phase"])
        if data.get("current_verdict"):
            data["current_verdict"] = GeminiVerdict(data["current_verdict"])
        
        # Reconstruct nested dataclasses
        if data.get("calibration"):
            data["calibration"] = CalibrationData(**data["calibration"])
        if data.get("last_error"):
            data["last_error"] = SessionError(**data["last_error"])
        if data.get("iteration_history"):
            data["iteration_history"] = [
                IterationRecord(**rec) for rec in data["iteration_history"]
            ]
        
        return cls(**data)


class StateManager:
    """Manages session state and persistence."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session: Optional[Session] = None
        self._session_path: Optional[Path] = None
        
        # Ensure storage directories exist
        self.config.sessions_path.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, task: str) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]
        now = utc_now_iso()
        
        self.session = Session(
            session_id=session_id,
            created_at=now,
            task_description=task,
            max_iterations=self.config.automation.max_iterations,
            started_at=now,
            last_activity_at=now,
        )
        
        # Create session directory
        self._session_path = self.config.sessions_path / session_id
        self._session_path.mkdir(parents=True, exist_ok=True)
        (self._session_path / "screenshots").mkdir(exist_ok=True)
        (self._session_path / "captures").mkdir(exist_ok=True)
        
        # Initial checkpoint
        self.checkpoint()
        
        logger.info(
            "Session created",
            session_id=session_id,
            task=task[:50] + "..." if len(task) > 50 else task,
        )
        
        return self.session
    
    def load_session(self, session_id: str) -> Session:
        """Load session from checkpoint."""
        session_path = self.config.sessions_path / session_id
        checkpoint_path = session_path / "checkpoint.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found for session {session_id}")
        
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        
        self.session = Session.from_dict(data)
        self._session_path = session_path
        
        logger.info("Session loaded", session_id=session_id, phase=self.session.phase.value)
        
        return self.session
    
    def transition_to(self, phase: SessionPhase) -> None:
        """Transition to a new phase."""
        if not self.session:
            raise RuntimeError("No active session")
        
        old_phase = self.session.phase
        self.session.phase = phase
        self.session.last_activity_at = utc_now_iso()
        
        logger.info("Phase transition", old=old_phase.value, new=phase.value)
        
        # Auto-checkpoint on significant transitions
        if phase in (SessionPhase.COMPLETE, SessionPhase.FAILED, SessionPhase.ABORTED):
            self.checkpoint()
    
    def record_error(
        self, error_type: str, message: str, recoverable: bool = True
    ) -> None:
        """Record an error."""
        if not self.session:
            raise RuntimeError("No active session")
        
        error = SessionError(
            error_type=error_type,
            message=message,
            timestamp=utc_now_iso(),
            recoverable=recoverable,
        )
        
        self.session.last_error = error
        self.session.total_errors += 1
        self.session.consecutive_errors += 1
        
        logger.error(
            "Error recorded",
            error_type=error_type,
            message=message,
            consecutive=self.session.consecutive_errors,
        )
    
    def clear_consecutive_errors(self) -> None:
        """Clear consecutive error count (on success)."""
        if self.session:
            self.session.consecutive_errors = 0
    
    def start_iteration(self, prompt: str, source: str = "initial") -> None:
        """Start a new iteration."""
        if not self.session:
            raise RuntimeError("No active session")
        
        self.session.iteration_count += 1
        self.session.current_prompt = prompt
        self.session.current_prompt_source = source
        self.session.current_response = None
        self.session.current_verdict = None
        self.session.current_feedback = None
        
        logger.info(
            "Iteration started",
            iteration=self.session.iteration_count,
            source=source,
        )
    
    def record_response(self, response: str, capture_method: str) -> None:
        """Record captured Copilot response."""
        if not self.session:
            raise RuntimeError("No active session")
        
        self.session.current_response = response
        self.session.capture_attempts += 1
        self.session.capture_successes += 1
        
        logger.info(
            "Response captured",
            method=capture_method,
            length=len(response),
        )
    
    def record_verdict(
        self,
        verdict: GeminiVerdict,
        feedback: str,
        confidence: str,
        next_prompt: Optional[str] = None,
    ) -> None:
        """Record Gemini verdict."""
        if not self.session:
            raise RuntimeError("No active session")
        
        self.session.current_verdict = verdict
        self.session.current_feedback = feedback
        self.session.next_prompt = next_prompt
        
        logger.info(
            "Verdict recorded",
            verdict=verdict.value,
            confidence=confidence,
        )
    
    def complete_iteration(self) -> None:
        """Complete current iteration and add to history."""
        if not self.session:
            raise RuntimeError("No active session")
        
        record = IterationRecord(
            iteration_number=self.session.iteration_count,
            started_at=self.session.last_activity_at or "",
            prompt=self.session.current_prompt or "",
            prompt_source=self.session.current_prompt_source,
            copilot_response=self.session.current_response,
            gemini_verdict=self.session.current_verdict.value if self.session.current_verdict else None,
            gemini_feedback=self.session.current_feedback,
            completed_at=utc_now_iso(),
        )
        
        self.session.iteration_history.append(record)
        self.checkpoint()
        
        logger.info("Iteration complete", iteration=self.session.iteration_count)
    
    def checkpoint(self) -> None:
        """Save current state to checkpoint file."""
        if not self.session or not self._session_path:
            return
        
        checkpoint_path = self._session_path / "checkpoint.json"
        
        with open(checkpoint_path, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2)
        
        logger.debug("Checkpoint saved", path=str(checkpoint_path))
    
    def append_audit_log(self, event_type: str, details: dict[str, Any]) -> None:
        """Append entry to audit log."""
        if not self._session_path:
            return
        
        audit_path = self._session_path / "audit.jsonl"
        
        entry = {
            "timestamp": utc_now_iso(),
            "session_id": self.session.session_id if self.session else None,
            "iteration": self.session.iteration_count if self.session else 0,
            "event_type": event_type,
            "details": details,
        }
        
        with open(audit_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    @property
    def session_path(self) -> Optional[Path]:
        """Get current session path."""
        return self._session_path
