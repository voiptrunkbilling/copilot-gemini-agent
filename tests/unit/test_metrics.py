"""
Tests for metrics module.
"""

import json
import pytest
import tempfile
from pathlib import Path

from copilot_agent.metrics import (
    MetricType,
    Metric,
    MetricsCollector,
    SessionMetrics,
    load_session_metrics,
    aggregate_session_stats,
    utc_now_iso,
)


class TestMetricType:
    """Tests for MetricType enum."""
    
    def test_metric_types_exist(self):
        """Test key metric types are defined."""
        assert MetricType.SESSION_START.value == "session_start"
        assert MetricType.ITERATION_START.value == "iteration_start"
        assert MetricType.REVIEWER_CALL.value == "reviewer_call"
        assert MetricType.ERROR.value == "error"
        assert MetricType.KILL_SWITCH.value == "kill_switch"


class TestMetric:
    """Tests for Metric dataclass."""
    
    def test_create_metric(self):
        """Test creating a metric."""
        metric = Metric(
            ts="2024-01-01T00:00:00Z",
            session_id="test-session",
            metric_type=MetricType.SESSION_START,
        )
        
        assert metric.ts == "2024-01-01T00:00:00Z"
        assert metric.session_id == "test-session"
        assert metric.metric_type == MetricType.SESSION_START
        assert metric.success is True
    
    def test_metric_to_dict(self):
        """Test converting metric to dictionary."""
        metric = Metric(
            ts="2024-01-01T00:00:00Z",
            session_id="test-session",
            metric_type=MetricType.REVIEWER_CALL,
            phase="running",
            duration_ms=500,
            data={"model": "gemma"},
        )
        
        d = metric.to_dict()
        
        assert d["ts"] == "2024-01-01T00:00:00Z"
        assert d["metric_type"] == "reviewer_call"
        assert d["duration_ms"] == 500
        assert d["data"]["model"] == "gemma"
    
    def test_metric_to_json(self):
        """Test JSON serialization."""
        metric = Metric(
            ts="2024-01-01T00:00:00Z",
            session_id="test",
            metric_type=MetricType.ERROR,
            success=False,
        )
        
        json_str = metric.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["success"] is False
        assert parsed["metric_type"] == "error"


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector(session_id="test")
        
        metric = collector.record(
            metric_type=MetricType.SESSION_START,
            phase="idle",
        )
        
        assert metric.session_id == "test"
        assert metric.metric_type == MetricType.SESSION_START
    
    def test_record_with_data(self):
        """Test recording metric with extra data."""
        collector = MetricsCollector(session_id="test")
        
        metric = collector.record(
            metric_type=MetricType.REVIEWER_CALL,
            model="gemma",
            tokens=100,
        )
        
        assert metric.data["model"] == "gemma"
        assert metric.data["tokens"] == 100
    
    def test_timer_operations(self):
        """Test start/stop timer."""
        collector = MetricsCollector(session_id="test")
        
        collector.start_timer("test_op")
        import time
        time.sleep(0.05)
        duration = collector.stop_timer("test_op")
        
        assert duration is not None
        assert duration >= 40  # At least 40ms
    
    def test_record_timed(self):
        """Test recording timed metric."""
        collector = MetricsCollector(session_id="test")
        
        collector.start_timer("review")
        metric = collector.record_timed(
            "review",
            MetricType.REVIEWER_CALL,
        )
        
        assert metric.duration_ms is not None
    
    def test_record_error(self):
        """Test recording error metric."""
        collector = MetricsCollector(session_id="test")
        
        metric = collector.record_error(
            error_type="APIError",
            message="Rate limit exceeded",
        )
        
        assert metric.metric_type == MetricType.ERROR
        assert metric.success is False
        assert metric.data["error_type"] == "APIError"
    
    def test_record_retry(self):
        """Test recording retry metric."""
        collector = MetricsCollector(session_id="test")
        
        metric = collector.record_retry(
            operation="reviewer_call",
            attempt=2,
            max_attempts=5,
            delay=4.0,
            error="Timeout",
        )
        
        assert metric.metric_type == MetricType.RETRY
        assert metric.data["attempt"] == 2
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector(session_id="test")
        
        collector.record(MetricType.SESSION_START)
        collector.record(MetricType.ITERATION_START, duration_ms=100)
        collector.record(MetricType.ITERATION_END, duration_ms=200)
        collector.record(MetricType.ITERATION_START, duration_ms=150)
        
        summary = collector.get_summary()
        
        assert summary["total_metrics"] == 4
        assert summary["counts"]["session_start"] == 1
        assert summary["counts"]["iteration_start"] == 2
    
    def test_persistence(self, temp_dir):
        """Test metrics are written to file."""
        metrics_file = temp_dir / "metrics.jsonl"
        
        collector = MetricsCollector(
            session_id="test",
            output_path=metrics_file,
        )
        
        collector.record(MetricType.SESSION_START)
        collector.record(MetricType.ITERATION_START)
        
        assert metrics_file.exists()
        
        lines = metrics_file.read_text().strip().split("\n")
        assert len(lines) == 2
    
    def test_get_stats_display(self):
        """Test formatted stats display."""
        collector = MetricsCollector(session_id="test")
        collector.record(MetricType.SESSION_START)
        
        display = collector.get_stats_display()
        
        assert "test" in display
        assert "session_start" in display


class TestSessionMetrics:
    """Tests for SessionMetrics."""
    
    def test_session_lifecycle(self):
        """Test session start/end metrics."""
        collector = MetricsCollector(session_id="test")
        session = SessionMetrics(collector)
        
        session.session_start("Test task")
        session.session_end("completed", success=True)
        
        summary = collector.get_summary()
        assert summary["counts"]["session_start"] == 1
        assert summary["counts"]["session_end"] == 1
    
    def test_iteration_lifecycle(self):
        """Test iteration metrics."""
        collector = MetricsCollector(session_id="test")
        session = SessionMetrics(collector)
        
        session.iteration_start(1, "clipboard")
        session.iteration_end(1, "ACCEPT")
        
        summary = collector.get_summary()
        assert summary["counts"]["iteration_start"] == 1
        assert summary["counts"]["iteration_end"] == 1
    
    def test_reviewer_call(self):
        """Test reviewer call metric."""
        collector = MetricsCollector(session_id="test")
        session = SessionMetrics(collector)
        
        session.reviewer_call(
            success=True,
            verdict="ACCEPT",
            duration_ms=1500,
            model="gemma-3-27b-it",
        )
        
        summary = collector.get_summary()
        assert summary["counts"]["reviewer_call"] == 1
        assert summary["durations"]["reviewer_call"]["avg_ms"] == 1500
    
    def test_ui_desync(self):
        """Test UI desync metric."""
        collector = MetricsCollector(session_id="test")
        session = SessionMetrics(collector)
        
        session.ui_desync("no_change", recovered=False)
        
        summary = collector.get_summary()
        assert summary["counts"]["ui_desync"] == 1
        # Check errors dict for ui_desync errors
        assert summary["errors"]["ui_desync"] == 1
    
    def test_budget_warning(self):
        """Test budget warning metric."""
        collector = MetricsCollector(session_id="test")
        session = SessionMetrics(collector)
        
        session.budget_warning("reviews", used=40, limit=50)
        
        # Check metric recorded
        assert len(collector._metrics) == 1
        metric = collector._metrics[0]
        assert metric.data["percent"] == 80


class TestLoadAndAggregate:
    """Tests for loading and aggregating metrics."""
    
    @pytest.fixture
    def session_with_metrics(self):
        """Create session directory with metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir)
            metrics_file = session_path / "metrics.jsonl"
            
            # Write some metrics
            metrics = [
                Metric(
                    ts="2024-01-01T00:00:00Z",
                    session_id="test",
                    metric_type=MetricType.SESSION_START,
                ),
                Metric(
                    ts="2024-01-01T00:00:01Z",
                    session_id="test",
                    metric_type=MetricType.REVIEWER_CALL,
                    duration_ms=1000,
                ),
                Metric(
                    ts="2024-01-01T00:00:02Z",
                    session_id="test",
                    metric_type=MetricType.REVIEWER_CALL,
                    duration_ms=2000,
                ),
            ]
            
            with open(metrics_file, "w") as f:
                for m in metrics:
                    f.write(m.to_json() + "\n")
            
            yield session_path
    
    def test_load_session_metrics(self, session_with_metrics):
        """Test loading metrics from file."""
        metrics = load_session_metrics(session_with_metrics)
        
        assert len(metrics) == 3
        assert metrics[0].metric_type == MetricType.SESSION_START
    
    def test_load_empty_session(self, tmp_path):
        """Test loading from empty session."""
        metrics = load_session_metrics(tmp_path)
        assert metrics == []
    
    def test_aggregate_stats(self, session_with_metrics):
        """Test aggregating session stats."""
        stats = aggregate_session_stats(session_with_metrics)
        
        assert stats["total_metrics"] == 3
        assert stats["counts"]["session_start"] == 1
        assert stats["counts"]["reviewer_call"] == 2
        assert stats["durations"]["reviewer_call"]["avg_ms"] == 1500
        assert stats["durations"]["reviewer_call"]["total_ms"] == 3000
