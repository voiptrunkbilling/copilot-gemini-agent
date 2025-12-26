"""
Observability and metrics collection.

Provides structured metrics logging in JSONL format and
aggregation for the `agent stats` command.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Any, List, Dict
from collections import defaultdict

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


def utc_now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MetricType(str, Enum):
    """Types of metrics."""
    
    # Session metrics
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_PAUSE = "session_pause"
    SESSION_RESUME = "session_resume"
    
    # Iteration metrics
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    
    # API metrics
    REVIEWER_CALL = "reviewer_call"
    VISION_CALL = "vision_call"
    
    # Capture metrics
    CAPTURE_ATTEMPT = "capture_attempt"
    CAPTURE_SUCCESS = "capture_success"
    CAPTURE_FAILURE = "capture_failure"
    
    # UI metrics
    UI_ACTION = "ui_action"
    UI_FOCUS = "ui_focus"
    UI_DESYNC = "ui_desync"
    
    # Error metrics
    ERROR = "error"
    RETRY = "retry"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"
    
    # Safety metrics
    KILL_SWITCH = "kill_switch"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class Metric:
    """Single metric record."""
    
    ts: str  # ISO timestamp
    session_id: Optional[str]
    metric_type: MetricType
    phase: Optional[str] = None
    operation: Optional[str] = None
    success: bool = True
    duration_ms: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON."""
        d = asdict(self)
        d["metric_type"] = self.metric_type.value
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class MetricsCollector:
    """
    Collects and persists metrics.
    
    Writes metrics to JSONL file for later analysis.
    """
    
    METRICS_FILE = "metrics.jsonl"
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        output_path: Optional[Path] = None,
    ):
        """
        Initialize metrics collector.
        
        Args:
            session_id: Current session ID
            output_path: Path for metrics file (default: session directory)
        """
        self.session_id = session_id
        self.output_path = output_path
        self._metrics: List[Metric] = []
        self._start_times: Dict[str, float] = {}
        
        # Aggregates
        self._counts: Dict[str, int] = defaultdict(int)
        self._durations: Dict[str, List[int]] = defaultdict(list)
        self._errors: Dict[str, int] = defaultdict(int)
    
    def set_session(self, session_id: str, session_path: Optional[Path] = None) -> None:
        """
        Set current session.
        
        Args:
            session_id: Session identifier
            session_path: Optional path to session directory
        """
        self.session_id = session_id
        if session_path:
            self.output_path = session_path / self.METRICS_FILE
    
    def record(
        self,
        metric_type: MetricType,
        phase: Optional[str] = None,
        operation: Optional[str] = None,
        success: bool = True,
        duration_ms: Optional[int] = None,
        **data: Any,
    ) -> Metric:
        """
        Record a metric.
        
        Args:
            metric_type: Type of metric
            phase: Current session phase
            operation: Specific operation
            success: Whether operation succeeded
            duration_ms: Duration in milliseconds
            **data: Additional metric data
            
        Returns:
            The recorded metric
        """
        metric = Metric(
            ts=utc_now_iso(),
            session_id=self.session_id,
            metric_type=metric_type,
            phase=phase,
            operation=operation,
            success=success,
            duration_ms=duration_ms,
            data=data,
        )
        
        self._metrics.append(metric)
        self._update_aggregates(metric)
        self._persist(metric)
        
        return metric
    
    def start_timer(self, key: str) -> None:
        """
        Start a timer for duration tracking.
        
        Args:
            key: Timer identifier
        """
        self._start_times[key] = time.time()
    
    def stop_timer(self, key: str) -> Optional[int]:
        """
        Stop a timer and return duration.
        
        Args:
            key: Timer identifier
            
        Returns:
            Duration in milliseconds or None if timer not found
        """
        start = self._start_times.pop(key, None)
        if start is None:
            return None
        
        return int((time.time() - start) * 1000)
    
    def record_timed(
        self,
        key: str,
        metric_type: MetricType,
        success: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """
        Record a metric with timing from a started timer.
        
        Args:
            key: Timer key
            metric_type: Type of metric
            success: Whether operation succeeded
            **kwargs: Additional metric data
            
        Returns:
            The recorded metric
        """
        duration_ms = self.stop_timer(key)
        return self.record(
            metric_type=metric_type,
            success=success,
            duration_ms=duration_ms,
            **kwargs,
        )
    
    def record_error(
        self,
        error_type: str,
        message: str,
        phase: Optional[str] = None,
        recoverable: bool = True,
        **data: Any,
    ) -> Metric:
        """
        Record an error metric.
        
        Args:
            error_type: Type/category of error
            message: Error message
            phase: Current phase
            recoverable: Whether error is recoverable
            **data: Additional data
            
        Returns:
            The recorded metric
        """
        return self.record(
            metric_type=MetricType.ERROR,
            phase=phase,
            success=False,
            error_type=error_type,
            message=message,
            recoverable=recoverable,
            **data,
        )
    
    def record_retry(
        self,
        operation: str,
        attempt: int,
        max_attempts: int,
        delay: float,
        error: str,
    ) -> Metric:
        """
        Record a retry metric.
        
        Args:
            operation: Operation being retried
            attempt: Current attempt number
            max_attempts: Maximum attempts allowed
            delay: Delay before next retry
            error: Error that caused retry
            
        Returns:
            The recorded metric
        """
        return self.record(
            metric_type=MetricType.RETRY,
            operation=operation,
            success=False,
            attempt=attempt,
            max_attempts=max_attempts,
            delay=delay,
            error=error,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Summary statistics
        """
        return {
            "session_id": self.session_id,
            "total_metrics": len(self._metrics),
            "counts": dict(self._counts),
            "errors": dict(self._errors),
            "durations": {
                k: {
                    "count": len(v),
                    "total_ms": sum(v),
                    "avg_ms": sum(v) / len(v) if v else 0,
                    "min_ms": min(v) if v else 0,
                    "max_ms": max(v) if v else 0,
                }
                for k, v in self._durations.items()
            },
        }
    
    def get_stats_display(self) -> str:
        """
        Get formatted stats for display.
        
        Returns:
            Human-readable stats string
        """
        summary = self.get_summary()
        lines = [
            "=" * 50,
            f"  Session: {summary['session_id'] or 'N/A'}",
            "=" * 50,
            "",
            "Counts:",
        ]
        
        for key, count in sorted(summary["counts"].items()):
            lines.append(f"  {key}: {count}")
        
        lines.append("")
        lines.append("Durations:")
        
        for key, stats in sorted(summary["durations"].items()):
            lines.append(
                f"  {key}: {stats['count']} calls, "
                f"avg={stats['avg_ms']:.0f}ms, "
                f"total={stats['total_ms']}ms"
            )
        
        if summary["errors"]:
            lines.append("")
            lines.append("Errors:")
            for key, count in sorted(summary["errors"].items()):
                lines.append(f"  {key}: {count}")
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def _update_aggregates(self, metric: Metric) -> None:
        """Update aggregate statistics."""
        key = metric.metric_type.value
        
        self._counts[key] += 1
        
        if metric.duration_ms is not None:
            self._durations[key].append(metric.duration_ms)
        
        if not metric.success:
            self._errors[key] += 1
    
    def _persist(self, metric: Metric) -> None:
        """Write metric to file."""
        if not self.output_path:
            return
        
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a") as f:
                f.write(metric.to_json() + "\n")
        except Exception as e:
            logger.warning("Failed to persist metric", error=str(e))


class SessionMetrics:
    """
    High-level session metrics tracking.
    
    Provides convenient methods for common metric patterns.
    """
    
    def __init__(self, collector: MetricsCollector):
        """
        Initialize session metrics.
        
        Args:
            collector: Underlying metrics collector
        """
        self.collector = collector
        self._phase: Optional[str] = None
        self._iteration: int = 0
    
    def set_phase(self, phase: str) -> None:
        """Set current phase."""
        self._phase = phase
    
    def set_iteration(self, iteration: int) -> None:
        """Set current iteration."""
        self._iteration = iteration
    
    def session_start(self, task: str) -> None:
        """Record session start."""
        self.collector.record(
            MetricType.SESSION_START,
            phase="idle",
            task=task[:100],
        )
    
    def session_end(self, reason: str, success: bool = True) -> None:
        """Record session end."""
        self.collector.record(
            MetricType.SESSION_END,
            phase=self._phase,
            success=success,
            reason=reason,
            iteration=self._iteration,
        )
    
    def iteration_start(self, iteration: int, prompt_source: str) -> None:
        """Record iteration start."""
        self._iteration = iteration
        self.collector.start_timer(f"iteration_{iteration}")
        self.collector.record(
            MetricType.ITERATION_START,
            phase=self._phase,
            iteration=iteration,
            prompt_source=prompt_source,
        )
    
    def iteration_end(self, iteration: int, verdict: str) -> None:
        """Record iteration end."""
        self.collector.record_timed(
            f"iteration_{iteration}",
            MetricType.ITERATION_END,
            phase=self._phase,
            iteration=iteration,
            verdict=verdict,
        )
    
    def reviewer_call(
        self,
        success: bool,
        verdict: Optional[str] = None,
        duration_ms: Optional[int] = None,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
    ) -> None:
        """Record reviewer API call."""
        self.collector.record(
            MetricType.REVIEWER_CALL,
            phase=self._phase,
            operation="review",
            success=success,
            duration_ms=duration_ms,
            verdict=verdict,
            model=model,
            tokens=tokens,
            iteration=self._iteration,
        )
    
    def vision_call(
        self,
        success: bool,
        duration_ms: Optional[int] = None,
        method: Optional[str] = None,
    ) -> None:
        """Record vision API call."""
        self.collector.record(
            MetricType.VISION_CALL,
            phase=self._phase,
            operation="vision",
            success=success,
            duration_ms=duration_ms,
            method=method,
            iteration=self._iteration,
        )
    
    def capture(self, success: bool, method: str, duration_ms: Optional[int] = None) -> None:
        """Record capture attempt."""
        metric_type = MetricType.CAPTURE_SUCCESS if success else MetricType.CAPTURE_FAILURE
        self.collector.record(
            metric_type,
            phase=self._phase,
            operation=method,
            success=success,
            duration_ms=duration_ms,
            iteration=self._iteration,
        )
    
    def ui_action(
        self,
        action: str,
        success: bool,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Record UI action."""
        self.collector.record(
            MetricType.UI_ACTION,
            phase=self._phase,
            operation=action,
            success=success,
            duration_ms=duration_ms,
            iteration=self._iteration,
        )
    
    def ui_desync(self, reason: str, recovered: bool = False) -> None:
        """Record UI desync event."""
        self.collector.record(
            MetricType.UI_DESYNC,
            phase=self._phase,
            success=recovered,
            reason=reason,
            iteration=self._iteration,
        )
    
    def kill_switch(self, triggered_by: str) -> None:
        """Record kill switch activation."""
        self.collector.record(
            MetricType.KILL_SWITCH,
            phase=self._phase,
            success=True,
            triggered_by=triggered_by,
            iteration=self._iteration,
        )
    
    def budget_warning(self, resource: str, used: int, limit: int) -> None:
        """Record budget warning."""
        self.collector.record(
            MetricType.BUDGET_WARNING,
            phase=self._phase,
            resource=resource,
            used=used,
            limit=limit,
            percent=int(used / limit * 100) if limit > 0 else 100,
        )
    
    def budget_exhausted(self, resource: str, used: int, limit: int) -> None:
        """Record budget exhaustion."""
        self.collector.record(
            MetricType.BUDGET_EXHAUSTED,
            phase=self._phase,
            success=False,
            resource=resource,
            used=used,
            limit=limit,
        )


def load_session_metrics(session_path: Path) -> List[Metric]:
    """
    Load metrics from a session.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        List of metrics
    """
    metrics_file = session_path / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    
    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["metric_type"] = MetricType(data["metric_type"])
                metrics.append(Metric(**data))
            except Exception:
                continue
    
    return metrics


def aggregate_session_stats(session_path: Path) -> Dict[str, Any]:
    """
    Aggregate stats from a session's metrics.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        Aggregated statistics
    """
    metrics = load_session_metrics(session_path)
    
    if not metrics:
        return {}
    
    stats: Dict[str, Any] = {
        "total_metrics": len(metrics),
        "session_id": metrics[0].session_id if metrics else None,
        "counts": defaultdict(int),
        "durations": defaultdict(list),
        "errors": 0,
        "retries": 0,
    }
    
    for m in metrics:
        stats["counts"][m.metric_type.value] += 1
        
        if m.duration_ms is not None:
            stats["durations"][m.metric_type.value].append(m.duration_ms)
        
        if not m.success:
            stats["errors"] += 1
        
        if m.metric_type == MetricType.RETRY:
            stats["retries"] += 1
    
    # Calculate duration stats
    duration_stats = {}
    for key, values in stats["durations"].items():
        if values:
            duration_stats[key] = {
                "count": len(values),
                "avg_ms": sum(values) / len(values),
                "total_ms": sum(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
    stats["durations"] = duration_stats
    stats["counts"] = dict(stats["counts"])
    
    return stats
