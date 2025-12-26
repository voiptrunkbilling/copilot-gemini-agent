"""
UI desync detection and recovery.

Detects when UI actions don't produce expected changes and
provides recovery strategies.
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import hashlib

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


class DesyncReason(str, Enum):
    """Reasons for UI desync detection."""
    
    NO_CHANGE = "no_change"  # Screenshot didn't change after action
    UNEXPECTED_STATE = "unexpected_state"  # UI in unexpected state
    ELEMENT_MISSING = "element_missing"  # Expected element not found
    FOCUS_LOST = "focus_lost"  # Window focus was lost
    TIMEOUT = "timeout"  # Action timed out
    EXCEPTION = "exception"  # Action raised exception


class RecoveryAction(str, Enum):
    """Actions that can be taken for recovery."""
    
    RETRY = "retry"  # Retry the action
    REFOCUS = "refocus"  # Refocus the window and retry
    WAIT = "wait"  # Wait and check again
    RECAPTURE = "recapture"  # Take new screenshot and re-analyze
    PAUSE = "pause"  # Pause for human intervention
    ABORT = "abort"  # Abort the session


@dataclass
class DesyncEvent:
    """Detected desync event."""
    
    reason: DesyncReason
    action: str  # Action that caused desync
    iteration: int
    details: str
    screenshot_before: Optional[str] = None  # Path or hash
    screenshot_after: Optional[str] = None
    suggested_recovery: RecoveryAction = RecoveryAction.RETRY


class ScreenshotComparer:
    """
    Compares screenshots to detect UI changes.
    
    Uses image hashing for fast comparison.
    """
    
    # Threshold for considering images different (0-100)
    # Lower = more sensitive to changes
    SIMILARITY_THRESHOLD = 98
    
    def __init__(self):
        """Initialize comparer."""
        self._last_hash: Optional[str] = None
        self._last_path: Optional[Path] = None
    
    def compute_hash(self, image_path: Path) -> str:
        """
        Compute hash of image.
        
        Uses MD5 for speed - not cryptographic use.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Hash string
        """
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def set_baseline(self, image_path: Path) -> None:
        """
        Set baseline screenshot for comparison.
        
        Args:
            image_path: Path to baseline image
        """
        self._last_path = image_path
        self._last_hash = self.compute_hash(image_path)
    
    def has_changed(self, image_path: Path) -> bool:
        """
        Check if image has changed from baseline.
        
        Args:
            image_path: Path to new image
            
        Returns:
            True if images are different
        """
        if self._last_hash is None:
            # No baseline - consider it changed
            return True
        
        new_hash = self.compute_hash(image_path)
        return new_hash != self._last_hash
    
    def update_baseline(self, image_path: Path) -> None:
        """Update baseline to new image."""
        self.set_baseline(image_path)


class DesyncDetector:
    """
    Detects UI desync conditions.
    
    Monitors for situations where UI doesn't respond as expected.
    """
    
    # Consecutive no-change threshold before flagging desync
    NO_CHANGE_THRESHOLD = 2
    
    # Time to wait between action and screenshot (seconds)
    ACTION_SETTLE_TIME = 0.5
    
    def __init__(
        self,
        comparer: Optional[ScreenshotComparer] = None,
        on_desync: Optional[Callable[[DesyncEvent], None]] = None,
    ):
        """
        Initialize detector.
        
        Args:
            comparer: Screenshot comparer instance
            on_desync: Callback for desync events
        """
        self.comparer = comparer or ScreenshotComparer()
        self.on_desync = on_desync
        
        self._consecutive_no_change = 0
        self._last_action: Optional[str] = None
        self._iteration: int = 0
    
    def set_iteration(self, iteration: int) -> None:
        """Set current iteration number."""
        self._iteration = iteration
    
    def before_action(
        self,
        action: str,
        screenshot_path: Optional[Path] = None,
    ) -> None:
        """
        Call before performing an action.
        
        Args:
            action: Description of action
            screenshot_path: Current screenshot for baseline
        """
        self._last_action = action
        
        if screenshot_path:
            self.comparer.set_baseline(screenshot_path)
    
    def after_action(
        self,
        screenshot_path: Path,
        expected_change: bool = True,
    ) -> Optional[DesyncEvent]:
        """
        Call after performing an action.
        
        Args:
            screenshot_path: New screenshot after action
            expected_change: Whether we expected the UI to change
            
        Returns:
            DesyncEvent if desync detected, None otherwise
        """
        has_changed = self.comparer.has_changed(screenshot_path)
        
        if expected_change and not has_changed:
            self._consecutive_no_change += 1
            
            if self._consecutive_no_change >= self.NO_CHANGE_THRESHOLD:
                event = DesyncEvent(
                    reason=DesyncReason.NO_CHANGE,
                    action=self._last_action or "unknown",
                    iteration=self._iteration,
                    details=f"UI unchanged after {self._consecutive_no_change} consecutive actions",
                    suggested_recovery=RecoveryAction.RECAPTURE,
                )
                
                if self.on_desync:
                    self.on_desync(event)
                
                return event
        else:
            # Reset counter on change
            self._consecutive_no_change = 0
        
        # Update baseline for next comparison
        self.comparer.update_baseline(screenshot_path)
        
        return None
    
    def check_focus(self, expected_window: str, current_window: str) -> Optional[DesyncEvent]:
        """
        Check if focus is on expected window.
        
        Args:
            expected_window: Expected window title
            current_window: Current foreground window title
            
        Returns:
            DesyncEvent if focus lost, None otherwise
        """
        if expected_window.lower() not in current_window.lower():
            event = DesyncEvent(
                reason=DesyncReason.FOCUS_LOST,
                action=self._last_action or "unknown",
                iteration=self._iteration,
                details=f"Expected '{expected_window}', got '{current_window}'",
                suggested_recovery=RecoveryAction.REFOCUS,
            )
            
            if self.on_desync:
                self.on_desync(event)
            
            return event
        
        return None
    
    def reset(self) -> None:
        """Reset detector state."""
        self._consecutive_no_change = 0
        self._last_action = None


class RecoveryManager:
    """
    Manages recovery from desync conditions.
    
    Provides strategies for recovering from various desync types.
    """
    
    # Maximum recovery attempts before pausing
    MAX_RECOVERY_ATTEMPTS = 3
    
    def __init__(
        self,
        refocus_fn: Optional[Callable[[], bool]] = None,
        recapture_fn: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize recovery manager.
        
        Args:
            refocus_fn: Function to refocus window (returns success)
            recapture_fn: Function to recapture screenshot
        """
        self.refocus_fn = refocus_fn
        self.recapture_fn = recapture_fn
        
        self._recovery_attempts: int = 0
    
    def attempt_recovery(self, event: DesyncEvent) -> bool:
        """
        Attempt to recover from desync.
        
        Args:
            event: Desync event to recover from
            
        Returns:
            True if recovery successful, False if should pause
        """
        self._recovery_attempts += 1
        
        if self._recovery_attempts > self.MAX_RECOVERY_ATTEMPTS:
            logger.warning(
                "Max recovery attempts reached",
                attempts=self._recovery_attempts,
                reason=event.reason.value,
            )
            return False
        
        recovery_action = event.suggested_recovery
        
        if recovery_action == RecoveryAction.RETRY:
            return self._handle_retry(event)
        elif recovery_action == RecoveryAction.REFOCUS:
            return self._handle_refocus(event)
        elif recovery_action == RecoveryAction.WAIT:
            return self._handle_wait(event)
        elif recovery_action == RecoveryAction.RECAPTURE:
            return self._handle_recapture(event)
        elif recovery_action == RecoveryAction.PAUSE:
            return False
        elif recovery_action == RecoveryAction.ABORT:
            return False
        
        return False
    
    def _handle_retry(self, event: DesyncEvent) -> bool:
        """Handle retry recovery."""
        logger.info(
            "Retry recovery",
            action=event.action,
            attempt=self._recovery_attempts,
        )
        # Caller should retry the action
        return True
    
    def _handle_refocus(self, event: DesyncEvent) -> bool:
        """Handle refocus recovery."""
        if self.refocus_fn:
            logger.info("Attempting to refocus window")
            success = self.refocus_fn()
            if success:
                logger.info("Window refocus successful")
                return True
            else:
                logger.warning("Window refocus failed")
        return False
    
    def _handle_wait(self, event: DesyncEvent) -> bool:
        """Handle wait recovery."""
        wait_time = 1.0 * self._recovery_attempts
        logger.info("Waiting for recovery", wait_seconds=wait_time)
        time.sleep(wait_time)
        return True
    
    def _handle_recapture(self, event: DesyncEvent) -> bool:
        """Handle recapture recovery."""
        if self.recapture_fn:
            logger.info("Recapturing screenshot")
            self.recapture_fn()
            return True
        return False
    
    def reset(self) -> None:
        """Reset recovery state."""
        self._recovery_attempts = 0


class ParseFailureTracker:
    """
    Tracks JSON parse failures for stop-on-unexpected-output.
    
    Pauses after N consecutive parse failures.
    """
    
    DEFAULT_THRESHOLD = 3
    
    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        """
        Initialize tracker.
        
        Args:
            threshold: Number of consecutive failures before pause
        """
        self.threshold = threshold
        self._consecutive_failures = 0
        self._total_failures = 0
    
    def record_success(self) -> None:
        """Record successful parse."""
        self._consecutive_failures = 0
    
    def record_failure(self, error: str) -> bool:
        """
        Record parse failure.
        
        Args:
            error: Error message
            
        Returns:
            True if should pause, False otherwise
        """
        self._consecutive_failures += 1
        self._total_failures += 1
        
        logger.warning(
            "Parse failure",
            consecutive=self._consecutive_failures,
            total=self._total_failures,
            error=error[:100],
        )
        
        if self._consecutive_failures >= self.threshold:
            logger.error(
                "Parse failure threshold reached",
                threshold=self.threshold,
                consecutive=self._consecutive_failures,
            )
            return True
        
        return False
    
    @property
    def should_pause(self) -> bool:
        """Check if should pause."""
        return self._consecutive_failures >= self.threshold
    
    @property
    def consecutive_failures(self) -> int:
        """Get consecutive failure count."""
        return self._consecutive_failures
    
    @property
    def total_failures(self) -> int:
        """Get total failure count."""
        return self._total_failures
    
    def reset(self) -> None:
        """Reset tracker."""
        self._consecutive_failures = 0
