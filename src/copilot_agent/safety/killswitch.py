"""
Kill switch implementation using pynput for global hotkeys.

Enhanced with:
- Preemptive check support for blocking operations
- Interrupt callbacks for cancelling async operations
- Thread-safe trigger with guaranteed callback execution
"""

import threading
import asyncio
from typing import Callable, Optional, Set, Any, List, TYPE_CHECKING
from dataclasses import dataclass
from contextlib import contextmanager
import time

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

# Import pynput - handle import error gracefully for testing
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    keyboard = None  # type: ignore

# Type alias for key events when pynput may not be available
KeyType = Any  # Will be keyboard.Key | keyboard.KeyCode when pynput is available


class KillSwitchTriggered(Exception):
    """Exception raised when kill switch is triggered during blocking operation."""
    pass


@dataclass
class HotkeyConfig:
    """Parsed hotkey configuration."""
    
    modifiers: Set[str]
    key: str
    
    @classmethod
    def parse(cls, hotkey_str: str) -> "HotkeyConfig":
        """
        Parse hotkey string like 'ctrl+shift+k' into components.
        """
        parts = hotkey_str.lower().split("+")
        key = parts[-1]
        modifiers = set(parts[:-1])
        return cls(modifiers=modifiers, key=key)


class KillSwitch:
    """
    Global kill switch using pynput.
    
    Runs in a separate thread and monitors for:
    - Configured hotkey (default: Ctrl+Shift+K)
    - Rapid Escape key presses (3x within 1 second)
    
    Enhanced features:
    - check() method for preemptive checks in loops
    - Interrupt callbacks for async operation cancellation
    - Context manager for protecting blocking operations
    """
    
    def __init__(
        self,
        hotkey: str = "ctrl+shift+k",
        on_trigger: Optional[Callable[[], None]] = None,
    ):
        self.hotkey_config = HotkeyConfig.parse(hotkey)
        self.on_trigger = on_trigger
        
        self._triggered = threading.Event()
        self._listener: Optional[keyboard.Listener] = None  # type: ignore
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Interrupt callbacks for cancelling operations
        self._interrupt_callbacks: List[Callable[[], None]] = []
        
        # Track pressed modifiers
        self._pressed_modifiers: Set[str] = set()
        
        # Track escape presses for rapid-fire detection
        self._escape_times: list[float] = []
        self._escape_threshold = 3  # Number of presses
        self._escape_window = 1.0  # Time window in seconds
        
        # Trigger metadata
        self._trigger_time: Optional[float] = None
        self._trigger_source: Optional[str] = None
        
        logger.info(
            "Kill switch initialized",
            hotkey=hotkey,
            modifiers=list(self.hotkey_config.modifiers),
            key=self.hotkey_config.key,
        )
    
    @property
    def triggered(self) -> bool:
        """Check if kill switch has been triggered."""
        return self._triggered.is_set()
    
    @property
    def trigger_time(self) -> Optional[float]:
        """Get time when kill switch was triggered."""
        return self._trigger_time
    
    @property
    def trigger_source(self) -> Optional[str]:
        """Get source that triggered the kill switch."""
        return self._trigger_source
    
    def check(self) -> None:
        """
        Check if kill switch is triggered and raise if so.
        
        Call this at the start of loops and before long operations.
        
        Raises:
            KillSwitchTriggered: If kill switch has been triggered
        """
        if self._triggered.is_set():
            raise KillSwitchTriggered(
                f"Kill switch triggered by {self._trigger_source or 'unknown'}"
            )
    
    def register_interrupt(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when kill switch triggers.
        
        Use for cancelling async operations, closing connections, etc.
        
        Args:
            callback: Function to call on trigger
        """
        with self._lock:
            self._interrupt_callbacks.append(callback)
    
    def unregister_interrupt(self, callback: Callable[[], None]) -> None:
        """
        Unregister an interrupt callback.
        
        Args:
            callback: Previously registered callback
        """
        with self._lock:
            try:
                self._interrupt_callbacks.remove(callback)
            except ValueError:
                pass
    
    @contextmanager
    def interruptible(self, cleanup: Optional[Callable[[], None]] = None):
        """
        Context manager for interruptible operations.
        
        Usage:
            with kill_switch.interruptible(cancel_request):
                response = await long_api_call()
        
        Args:
            cleanup: Optional cleanup function to call on interrupt
            
        Raises:
            KillSwitchTriggered: If kill switch triggers during operation
        """
        if cleanup:
            self.register_interrupt(cleanup)
        
        try:
            self.check()  # Check before starting
            yield
            self.check()  # Check after completing
        finally:
            if cleanup:
                self.unregister_interrupt(cleanup)
    
    def start(self) -> None:
        """Start the kill switch listener."""
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not available, kill switch disabled")
            return
        
        if self._listener is not None:
            return
        
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        
        logger.info("Kill switch listener started")
    
    def stop(self) -> None:
        """Stop the kill switch listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Kill switch listener stopped")
    
    def trigger(self) -> None:
        """Manually trigger the kill switch."""
        self._do_trigger("manual")
    
    def _do_trigger(self, source: str) -> None:
        """
        Internal trigger with source tracking.
        
        Args:
            source: What triggered the kill switch
        """
        with self._lock:
            if self._triggered.is_set():
                return
            
            self._trigger_time = time.time()
            self._trigger_source = source
            
            logger.warning(
                "Kill switch TRIGGERED!",
                source=source,
            )
            
            self._triggered.set()
            
            # Call interrupt callbacks
            for callback in self._interrupt_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(
                        "Error in interrupt callback",
                        error=str(e),
                    )
            
            # Call main trigger callback
            if self.on_trigger:
                try:
                    self.on_trigger()
                except Exception as e:
                    logger.error("Error in kill switch callback", error=str(e))
    
    def reset(self) -> None:
        """Reset the kill switch (for testing)."""
        with self._lock:
            self._triggered.clear()
            self._pressed_modifiers.clear()
            self._escape_times.clear()
            self._trigger_time = None
            self._trigger_source = None
            self._interrupt_callbacks.clear()
    
    def _on_press(self, key: KeyType) -> None:
        """Handle key press event."""
        if self._triggered.is_set():
            return
        
        key_name = self._get_key_name(key)
        
        # Track modifiers
        if key_name in ("ctrl", "shift", "alt", "cmd"):
            self._pressed_modifiers.add(key_name)
        
        # Check for configured hotkey
        if self._check_hotkey(key_name):
            self._do_trigger(f"hotkey:{self.hotkey_config.modifiers}+{self.hotkey_config.key}")
            return
        
        # Check for rapid Escape
        if key_name == "escape":
            self._check_escape_sequence()
    
    def _on_release(self, key: KeyType) -> None:
        """Handle key release event."""
        key_name = self._get_key_name(key)
        
        if key_name in self._pressed_modifiers:
            self._pressed_modifiers.discard(key_name)
    
    def _get_key_name(self, key: KeyType) -> str:
        """Get normalized key name."""
        if not PYNPUT_AVAILABLE:
            return ""
        
        try:
            # Special keys (Ctrl, Shift, etc.)
            if hasattr(key, "name"):
                name = key.name.lower()
                # Normalize modifier names
                if name in ("ctrl_l", "ctrl_r"):
                    return "ctrl"
                if name in ("shift_l", "shift_r"):
                    return "shift"
                if name in ("alt_l", "alt_r", "alt_gr"):
                    return "alt"
                if name in ("cmd_l", "cmd_r", "cmd"):
                    return "cmd"
                return name
            # Regular characters
            if hasattr(key, "char") and key.char:
                return key.char.lower()
        except Exception:
            pass
        
        return ""
    
    def _check_hotkey(self, key_name: str) -> bool:
        """Check if the configured hotkey is pressed."""
        # Check if all modifiers are pressed
        for modifier in self.hotkey_config.modifiers:
            if modifier not in self._pressed_modifiers:
                return False
        
        # Check if the key matches
        return key_name == self.hotkey_config.key
    
    def _check_escape_sequence(self) -> None:
        """Check for rapid Escape key presses."""
        now = time.time()
        
        # Add current press
        self._escape_times.append(now)
        
        # Remove old presses outside the window
        self._escape_times = [
            t for t in self._escape_times
            if now - t < self._escape_window
        ]
        
        # Check if threshold reached
        if len(self._escape_times) >= self._escape_threshold:
            logger.warning("Rapid Escape sequence detected!")
            self._do_trigger("rapid_escape")


def wait_with_killswitch(
    kill_switch: KillSwitch,
    duration: float,
    check_interval: float = 0.1,
) -> None:
    """
    Wait for duration while checking kill switch.
    
    Use instead of time.sleep() for interruptible waits.
    
    Args:
        kill_switch: Kill switch instance
        duration: Total wait time in seconds
        check_interval: How often to check kill switch
        
    Raises:
        KillSwitchTriggered: If kill switch is triggered during wait
    """
    end_time = time.time() + duration
    while time.time() < end_time:
        kill_switch.check()
        remaining = end_time - time.time()
        time.sleep(min(check_interval, max(0, remaining)))


async def async_wait_with_killswitch(
    kill_switch: KillSwitch,
    duration: float,
    check_interval: float = 0.1,
) -> None:
    """
    Async wait for duration while checking kill switch.
    
    Use instead of asyncio.sleep() for interruptible waits.
    
    Args:
        kill_switch: Kill switch instance
        duration: Total wait time in seconds
        check_interval: How often to check kill switch
        
    Raises:
        KillSwitchTriggered: If kill switch is triggered during wait
    """
    end_time = time.time() + duration
    while time.time() < end_time:
        kill_switch.check()
        remaining = end_time - time.time()
        await asyncio.sleep(min(check_interval, max(0, remaining)))


class MockKillSwitch(KillSwitch):
    """Mock kill switch for testing (no pynput dependency)."""
    
    def start(self) -> None:
        """No-op start."""
        logger.info("Mock kill switch started")
    
    def stop(self) -> None:
        """No-op stop."""
        logger.info("Mock kill switch stopped")
