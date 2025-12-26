"""
Kill switch implementation using pynput for global hotkeys.
"""

import threading
from typing import Callable, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
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
        
        # Track pressed modifiers
        self._pressed_modifiers: Set[str] = set()
        
        # Track escape presses for rapid-fire detection
        self._escape_times: list[float] = []
        self._escape_threshold = 3  # Number of presses
        self._escape_window = 1.0  # Time window in seconds
        
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
        if self._triggered.is_set():
            return
        
        logger.warning("Kill switch TRIGGERED!")
        self._triggered.set()
        
        if self.on_trigger:
            try:
                self.on_trigger()
            except Exception as e:
                logger.error("Error in kill switch callback", error=str(e))
    
    def reset(self) -> None:
        """Reset the kill switch (for testing)."""
        self._triggered.clear()
        self._pressed_modifiers.clear()
        self._escape_times.clear()
    
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
            self.trigger()
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
            self.trigger()


class MockKillSwitch(KillSwitch):
    """Mock kill switch for testing (no pynput dependency)."""
    
    def start(self) -> None:
        """No-op start."""
        logger.info("Mock kill switch started")
    
    def stop(self) -> None:
        """No-op stop."""
        logger.info("Mock kill switch stopped")
