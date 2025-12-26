"""
Tests for desync detection module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from copilot_agent.desync import (
    DesyncReason,
    RecoveryAction,
    DesyncEvent,
    ScreenshotComparer,
    DesyncDetector,
    RecoveryManager,
    ParseFailureTracker,
)


class TestScreenshotComparer:
    """Tests for ScreenshotComparer."""
    
    @pytest.fixture
    def temp_images(self):
        """Create temporary image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            
            # Create two identical files
            img1 = dir_path / "img1.png"
            img2 = dir_path / "img2.png"
            img1.write_bytes(b"identical content")
            img2.write_bytes(b"identical content")
            
            # Create different file
            img3 = dir_path / "img3.png"
            img3.write_bytes(b"different content")
            
            yield {"same1": img1, "same2": img2, "different": img3}
    
    def test_compute_hash(self, temp_images):
        """Test hash computation."""
        comparer = ScreenshotComparer()
        
        hash1 = comparer.compute_hash(temp_images["same1"])
        hash2 = comparer.compute_hash(temp_images["same2"])
        hash3 = comparer.compute_hash(temp_images["different"])
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_set_baseline(self, temp_images):
        """Test setting baseline."""
        comparer = ScreenshotComparer()
        
        comparer.set_baseline(temp_images["same1"])
        
        assert comparer._last_hash is not None
        assert comparer._last_path == temp_images["same1"]
    
    def test_has_changed_identical(self, temp_images):
        """Test detection of identical images."""
        comparer = ScreenshotComparer()
        comparer.set_baseline(temp_images["same1"])
        
        assert not comparer.has_changed(temp_images["same2"])
    
    def test_has_changed_different(self, temp_images):
        """Test detection of different images."""
        comparer = ScreenshotComparer()
        comparer.set_baseline(temp_images["same1"])
        
        assert comparer.has_changed(temp_images["different"])
    
    def test_no_baseline_returns_changed(self, temp_images):
        """Test no baseline returns changed."""
        comparer = ScreenshotComparer()
        
        assert comparer.has_changed(temp_images["same1"])


class TestDesyncDetector:
    """Tests for DesyncDetector."""
    
    @pytest.fixture
    def temp_images(self):
        """Create temporary image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            
            img1 = dir_path / "img1.png"
            img1.write_bytes(b"content1")
            
            img2 = dir_path / "img2.png"
            img2.write_bytes(b"content1")  # Same as img1
            
            img3 = dir_path / "img3.png"
            img3.write_bytes(b"content3")  # Different
            
            yield {"base": img1, "same": img2, "different": img3}
    
    def test_no_desync_on_change(self, temp_images):
        """Test no desync when UI changes."""
        detector = DesyncDetector()
        
        detector.before_action("click", temp_images["base"])
        event = detector.after_action(temp_images["different"])
        
        assert event is None
    
    def test_desync_on_no_change_threshold(self, temp_images):
        """Test desync detected after threshold."""
        detector = DesyncDetector()
        detector.NO_CHANGE_THRESHOLD = 2
        
        # First no-change
        detector.before_action("click1", temp_images["base"])
        event1 = detector.after_action(temp_images["same"])
        assert event1 is None
        
        # Second no-change - triggers desync
        detector.before_action("click2", temp_images["base"])
        event2 = detector.after_action(temp_images["same"])
        
        assert event2 is not None
        assert event2.reason == DesyncReason.NO_CHANGE
    
    def test_desync_callback(self, temp_images):
        """Test desync callback is called."""
        callback = MagicMock()
        detector = DesyncDetector(on_desync=callback)
        detector.NO_CHANGE_THRESHOLD = 1
        
        detector.before_action("click", temp_images["base"])
        detector.after_action(temp_images["same"])
        
        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, DesyncEvent)
    
    def test_check_focus_lost(self):
        """Test focus loss detection."""
        detector = DesyncDetector()
        
        event = detector.check_focus(
            expected_window="VS Code",
            current_window="Explorer",
        )
        
        assert event is not None
        assert event.reason == DesyncReason.FOCUS_LOST
        assert event.suggested_recovery == RecoveryAction.REFOCUS
    
    def test_check_focus_ok(self):
        """Test focus check passes."""
        detector = DesyncDetector()
        
        event = detector.check_focus(
            expected_window="VS Code",
            current_window="VS Code - project",
        )
        
        assert event is None
    
    def test_reset(self, temp_images):
        """Test detector reset."""
        detector = DesyncDetector()
        detector.NO_CHANGE_THRESHOLD = 1
        detector._consecutive_no_change = 5
        
        detector.reset()
        
        assert detector._consecutive_no_change == 0


class TestRecoveryManager:
    """Tests for RecoveryManager."""
    
    def test_retry_recovery(self):
        """Test retry recovery action."""
        manager = RecoveryManager()
        
        event = DesyncEvent(
            reason=DesyncReason.NO_CHANGE,
            action="click",
            iteration=1,
            details="test",
            suggested_recovery=RecoveryAction.RETRY,
        )
        
        success = manager.attempt_recovery(event)
        assert success is True
    
    def test_refocus_recovery_with_function(self):
        """Test refocus recovery with callback."""
        refocus_fn = MagicMock(return_value=True)
        manager = RecoveryManager(refocus_fn=refocus_fn)
        
        event = DesyncEvent(
            reason=DesyncReason.FOCUS_LOST,
            action="type",
            iteration=1,
            details="test",
            suggested_recovery=RecoveryAction.REFOCUS,
        )
        
        success = manager.attempt_recovery(event)
        
        assert success is True
        refocus_fn.assert_called_once()
    
    def test_refocus_recovery_fails(self):
        """Test refocus recovery failure."""
        refocus_fn = MagicMock(return_value=False)
        manager = RecoveryManager(refocus_fn=refocus_fn)
        
        event = DesyncEvent(
            reason=DesyncReason.FOCUS_LOST,
            action="type",
            iteration=1,
            details="test",
            suggested_recovery=RecoveryAction.REFOCUS,
        )
        
        success = manager.attempt_recovery(event)
        assert success is False
    
    def test_max_recovery_attempts(self):
        """Test max recovery attempts limit."""
        manager = RecoveryManager()
        manager.MAX_RECOVERY_ATTEMPTS = 2
        
        event = DesyncEvent(
            reason=DesyncReason.NO_CHANGE,
            action="click",
            iteration=1,
            details="test",
            suggested_recovery=RecoveryAction.RETRY,
        )
        
        # First two attempts succeed
        assert manager.attempt_recovery(event) is True
        assert manager.attempt_recovery(event) is True
        
        # Third attempt fails (max reached)
        assert manager.attempt_recovery(event) is False
    
    def test_reset(self):
        """Test recovery manager reset."""
        manager = RecoveryManager()
        manager._recovery_attempts = 5
        
        manager.reset()
        
        assert manager._recovery_attempts == 0


class TestParseFailureTracker:
    """Tests for ParseFailureTracker."""
    
    def test_initial_state(self):
        """Test initial state."""
        tracker = ParseFailureTracker()
        
        assert tracker.consecutive_failures == 0
        assert tracker.total_failures == 0
        assert tracker.should_pause is False
    
    def test_record_success(self):
        """Test success resets consecutive count."""
        tracker = ParseFailureTracker()
        
        tracker.record_failure("error1")
        tracker.record_failure("error2")
        assert tracker.consecutive_failures == 2
        
        tracker.record_success()
        
        assert tracker.consecutive_failures == 0
        assert tracker.total_failures == 2  # Total unchanged
    
    def test_threshold_reached(self):
        """Test threshold triggers pause."""
        tracker = ParseFailureTracker(threshold=3)
        
        assert tracker.record_failure("error1") is False
        assert tracker.record_failure("error2") is False
        assert tracker.record_failure("error3") is True  # Threshold reached
        
        assert tracker.should_pause is True
    
    def test_reset(self):
        """Test tracker reset."""
        tracker = ParseFailureTracker()
        
        tracker.record_failure("error")
        tracker.record_failure("error")
        
        tracker.reset()
        
        assert tracker.consecutive_failures == 0
        # Total failures not reset
        assert tracker.total_failures == 2


class TestDesyncEvent:
    """Tests for DesyncEvent dataclass."""
    
    def test_create_event(self):
        """Test creating desync event."""
        event = DesyncEvent(
            reason=DesyncReason.TIMEOUT,
            action="api_call",
            iteration=5,
            details="Request timed out after 30s",
        )
        
        assert event.reason == DesyncReason.TIMEOUT
        assert event.action == "api_call"
        assert event.iteration == 5
        assert event.suggested_recovery == RecoveryAction.RETRY
    
    def test_event_with_screenshots(self):
        """Test event with screenshot info."""
        event = DesyncEvent(
            reason=DesyncReason.NO_CHANGE,
            action="click",
            iteration=1,
            details="No change",
            screenshot_before="/path/before.png",
            screenshot_after="/path/after.png",
            suggested_recovery=RecoveryAction.RECAPTURE,
        )
        
        assert event.screenshot_before == "/path/before.png"
        assert event.screenshot_after == "/path/after.png"
