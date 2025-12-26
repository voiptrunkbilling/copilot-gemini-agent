"""
GUI action executor.

Handles mouse, keyboard, and clipboard operations with kill switch checks.
Windows-first implementation using pyautogui.
"""

import time
from typing import Optional, Tuple, Callable, Any, List
from dataclasses import dataclass, field
from enum import Enum

from copilot_agent.logging import get_logger
from copilot_agent.actuator.platform import (
    IS_WINDOWS, 
    get_dpi_info, 
    get_primary_screen,
    set_dpi_awareness,
)

logger = get_logger(__name__)

# Try to import pyautogui
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move to corner to abort
    pyautogui.PAUSE = 0.05  # Default pause between actions
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    logger.warning("pyautogui not available")


class ActionType(str, Enum):
    """Types of GUI actions."""
    
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    HOTKEY = "hotkey"
    PRESS_KEY = "press_key"
    COPY = "copy"
    PASTE = "paste"
    MOVE_MOUSE = "move_mouse"
    SCROLL = "scroll"
    WAIT = "wait"
    FOCUS_WINDOW = "focus_window"
    SCREENSHOT = "screenshot"
    READ_CLIPBOARD = "read_clipboard"


@dataclass
class ActionResult:
    """Result of an action execution."""
    
    success: bool
    action_type: ActionType
    message: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    data: Optional[Any] = None  # For returning clipboard contents, etc.


@dataclass
class ActionConfig:
    """Configuration for action timing."""
    
    action_delay_ms: int = 50
    typing_delay_ms: int = 10
    hotkey_delay_ms: int = 50
    click_delay_ms: int = 50
    screenshot_delay_ms: int = 100
    max_text_length: int = 10000
    max_wait_ms: int = 60000
    
    # Forbidden hotkey combinations
    forbidden_hotkeys: List[str] = field(default_factory=lambda: [
        "alt+f4",  # Close window
        "ctrl+w",  # Close tab
        "ctrl+q",  # Quit application (Linux)
        "ctrl+shift+w",  # Close all tabs
        "cmd+q",  # Quit (macOS)
        "super+d",  # Show desktop
    ])


class ActionExecutor:
    """
    Executes GUI actions with validation, safety checks, and kill switch support.
    
    All actions check kill switch before execution.
    Supports dry-run mode for testing without actual GUI interaction.
    """
    
    def __init__(
        self,
        config: Optional[ActionConfig] = None,
        dry_run: bool = False,
        kill_switch_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize ActionExecutor.
        
        Args:
            config: Action timing configuration
            dry_run: If True, only log actions without executing
            kill_switch_check: Function that returns True if kill switch triggered
        """
        self.config = config or ActionConfig()
        self.dry_run = dry_run
        self._kill_switch_check = kill_switch_check
        
        # Set DPI awareness early
        set_dpi_awareness()
        
        # Get screen info
        self._dpi_info = get_dpi_info()
        self._screen = get_primary_screen()
        self._screen_width = self._screen.width
        self._screen_height = self._screen.height
        self._scale_factor = self._screen.scale_factor
        
        # Configure pyautogui
        if HAS_PYAUTOGUI:
            pyautogui.PAUSE = self.config.action_delay_ms / 1000.0
        
        logger.info(
            "ActionExecutor initialized",
            dry_run=dry_run,
            has_pyautogui=HAS_PYAUTOGUI,
            screen_size=(self._screen_width, self._screen_height),
            dpi_scale=self._scale_factor,
        )
    
    def _check_kill_switch(self) -> bool:
        """Check if kill switch is triggered."""
        if self._kill_switch_check is not None:
            return self._kill_switch_check()
        return False
    
    def _pre_action_check(self, action_type: ActionType) -> Optional[ActionResult]:
        """
        Perform pre-action safety checks.
        
        Returns ActionResult with error if check fails, None if OK.
        """
        if self._check_kill_switch():
            return ActionResult(
                success=False,
                action_type=action_type,
                error="Kill switch triggered - action aborted",
            )
        return None
    
    def _apply_dpi_scaling(self, x: int, y: int) -> Tuple[int, int]:
        """Apply DPI scaling to coordinates if needed."""
        if self._dpi_info.is_aware:
            # Already DPI aware, coordinates are in physical pixels
            return (x, y)
        else:
            # Need to scale coordinates
            return (
                int(x / self._scale_factor),
                int(y / self._scale_factor),
            )
    
    def click(
        self,
        x: int,
        y: int,
        clicks: int = 1,
        button: str = "left",
        interval: float = 0.1,
    ) -> ActionResult:
        """
        Click at coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            clicks: Number of clicks (1, 2, or 3)
            button: Mouse button ('left', 'right', or 'middle')
            interval: Interval between multi-clicks
        """
        action_type = {
            1: ActionType.CLICK,
            2: ActionType.DOUBLE_CLICK,
            3: ActionType.TRIPLE_CLICK,
        }.get(clicks, ActionType.CLICK)
        
        if button == "right":
            action_type = ActionType.RIGHT_CLICK
        
        # Pre-action check
        check = self._pre_action_check(action_type)
        if check:
            return check
        
        # Validate coordinates
        if not self._validate_coordinates(x, y):
            return ActionResult(
                success=False,
                action_type=action_type,
                error=f"Coordinates out of bounds: ({x}, {y}) not in ({self._screen_width}, {self._screen_height})",
            )
        
        # Apply DPI scaling
        scaled_x, scaled_y = self._apply_dpi_scaling(x, y)
        
        if self.dry_run:
            logger.info("DRY-RUN: click", x=scaled_x, y=scaled_y, clicks=clicks, button=button)
            return ActionResult(
                success=True,
                action_type=action_type,
                message=f"Would click at ({scaled_x}, {scaled_y}) {clicks}x with {button}",
            )
        
        if not HAS_PYAUTOGUI:
            return ActionResult(
                success=False,
                action_type=action_type,
                error="pyautogui not available",
            )
        
        start = time.time()
        try:
            pyautogui.click(
                x=scaled_x,
                y=scaled_y,
                clicks=clicks,
                interval=interval,
                button=button,
            )
            duration_ms = int((time.time() - start) * 1000)
            
            logger.debug("Click executed", x=scaled_x, y=scaled_y, clicks=clicks, duration_ms=duration_ms)
            
            return ActionResult(
                success=True,
                action_type=action_type,
                message=f"Clicked at ({scaled_x}, {scaled_y})",
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=action_type,
                error=f"Click failed: {str(e)}",
            )
    
    def move_mouse(self, x: int, y: int, duration: float = 0.25) -> ActionResult:
        """
        Move mouse to coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate  
            duration: Movement duration in seconds
        """
        check = self._pre_action_check(ActionType.MOVE_MOUSE)
        if check:
            return check
        
        if not self._validate_coordinates(x, y):
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE_MOUSE,
                error=f"Coordinates out of bounds: ({x}, {y})",
            )
        
        scaled_x, scaled_y = self._apply_dpi_scaling(x, y)
        
        if self.dry_run:
            logger.info("DRY-RUN: move_mouse", x=scaled_x, y=scaled_y)
            return ActionResult(
                success=True,
                action_type=ActionType.MOVE_MOUSE,
                message=f"Would move to ({scaled_x}, {scaled_y})",
            )
        
        if not HAS_PYAUTOGUI:
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE_MOUSE,
                error="pyautogui not available",
            )
        
        try:
            pyautogui.moveTo(scaled_x, scaled_y, duration=duration)
            return ActionResult(
                success=True,
                action_type=ActionType.MOVE_MOUSE,
                message=f"Moved to ({scaled_x}, {scaled_y})",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE_MOUSE,
                error=f"Move failed: {str(e)}",
            )
    
    def type_text(
        self, 
        text: str, 
        interval: Optional[float] = None,
        use_clipboard: bool = False,
    ) -> ActionResult:
        """
        Type text.
        
        Args:
            text: Text to type
            interval: Interval between keystrokes (None = use config)
            use_clipboard: If True, use clipboard paste for faster input
        """
        check = self._pre_action_check(ActionType.TYPE_TEXT)
        if check:
            return check
        
        # Validate text length
        if len(text) > self.config.max_text_length:
            return ActionResult(
                success=False,
                action_type=ActionType.TYPE_TEXT,
                error=f"Text too long: {len(text)} > {self.config.max_text_length}",
            )
        
        if self.dry_run:
            preview = text[:50] + "..." if len(text) > 50 else text
            logger.info("DRY-RUN: type_text", length=len(text), preview=preview)
            return ActionResult(
                success=True,
                action_type=ActionType.TYPE_TEXT,
                message=f"Would type {len(text)} characters",
            )
        
        if not HAS_PYAUTOGUI:
            return ActionResult(
                success=False,
                action_type=ActionType.TYPE_TEXT,
                error="pyautogui not available",
            )
        
        start = time.time()
        try:
            if use_clipboard:
                # Use clipboard for faster input (avoids special char issues)
                self._set_clipboard(text)
                time.sleep(0.05)
                pyautogui.hotkey("ctrl", "v")
            else:
                # Type character by character
                type_interval = interval if interval is not None else (self.config.typing_delay_ms / 1000.0)
                
                # Check kill switch periodically during typing
                chunk_size = 50
                for i in range(0, len(text), chunk_size):
                    if self._check_kill_switch():
                        return ActionResult(
                            success=False,
                            action_type=ActionType.TYPE_TEXT,
                            error=f"Kill switch triggered after {i} characters",
                        )
                    chunk = text[i:i + chunk_size]
                    pyautogui.write(chunk, interval=type_interval)
            
            duration_ms = int((time.time() - start) * 1000)
            
            return ActionResult(
                success=True,
                action_type=ActionType.TYPE_TEXT,
                message=f"Typed {len(text)} characters",
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.TYPE_TEXT,
                error=f"Type failed: {str(e)}",
            )
    
    def press_key(self, key: str, presses: int = 1, interval: float = 0.1) -> ActionResult:
        """
        Press a single key.
        
        Args:
            key: Key name (e.g., 'enter', 'tab', 'escape')
            presses: Number of times to press
            interval: Interval between presses
        """
        check = self._pre_action_check(ActionType.PRESS_KEY)
        if check:
            return check
        
        if self.dry_run:
            logger.info("DRY-RUN: press_key", key=key, presses=presses)
            return ActionResult(
                success=True,
                action_type=ActionType.PRESS_KEY,
                message=f"Would press {key} {presses}x",
            )
        
        if not HAS_PYAUTOGUI:
            return ActionResult(
                success=False,
                action_type=ActionType.PRESS_KEY,
                error="pyautogui not available",
            )
        
        try:
            pyautogui.press(key, presses=presses, interval=interval)
            return ActionResult(
                success=True,
                action_type=ActionType.PRESS_KEY,
                message=f"Pressed {key}",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.PRESS_KEY,
                error=f"Press failed: {str(e)}",
            )
    
    def hotkey(self, *keys: str) -> ActionResult:
        """
        Execute a hotkey combination.
        
        Args:
            keys: Key names (e.g., "ctrl", "c")
        """
        check = self._pre_action_check(ActionType.HOTKEY)
        if check:
            return check
        
        # Validate hotkey is not forbidden
        combo = "+".join(k.lower() for k in keys)
        if combo in self.config.forbidden_hotkeys:
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error=f"Forbidden hotkey: {combo}",
            )
        
        if self.dry_run:
            logger.info("DRY-RUN: hotkey", keys=keys)
            return ActionResult(
                success=True,
                action_type=ActionType.HOTKEY,
                message=f"Would press {'+'.join(keys)}",
            )
        
        if not HAS_PYAUTOGUI:
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error="pyautogui not available",
            )
        
        try:
            pyautogui.hotkey(*keys, interval=self.config.hotkey_delay_ms / 1000.0)
            return ActionResult(
                success=True,
                action_type=ActionType.HOTKEY,
                message=f"Pressed {'+'.join(keys)}",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error=f"Hotkey failed: {str(e)}",
            )
    
    def copy_selection(self) -> ActionResult:
        """
        Copy current selection to clipboard (Ctrl+C).
        
        Returns:
            ActionResult with clipboard content in data field
        """
        check = self._pre_action_check(ActionType.COPY)
        if check:
            return check
        
        if self.dry_run:
            logger.info("DRY-RUN: copy_selection")
            return ActionResult(
                success=True,
                action_type=ActionType.COPY,
                message="Would copy selection",
            )
        
        try:
            # Execute Ctrl+C
            result = self.hotkey("ctrl", "c")
            if not result.success:
                return result
            
            # Small delay for clipboard to update
            time.sleep(0.1)
            
            # Read clipboard
            content = self._get_clipboard()
            
            return ActionResult(
                success=True,
                action_type=ActionType.COPY,
                message=f"Copied {len(content) if content else 0} characters",
                data=content,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.COPY,
                error=f"Copy failed: {str(e)}",
            )
    
    def paste_text(self, text: Optional[str] = None) -> ActionResult:
        """
        Paste from clipboard or paste specific text.
        
        Args:
            text: If provided, set clipboard to this text first
        """
        check = self._pre_action_check(ActionType.PASTE)
        if check:
            return check
        
        if self.dry_run:
            logger.info("DRY-RUN: paste_text", has_text=text is not None)
            return ActionResult(
                success=True,
                action_type=ActionType.PASTE,
                message="Would paste",
            )
        
        try:
            if text is not None:
                self._set_clipboard(text)
                time.sleep(0.05)
            
            result = self.hotkey("ctrl", "v")
            if not result.success:
                return result
            
            return ActionResult(
                success=True,
                action_type=ActionType.PASTE,
                message=f"Pasted{' text' if text else ' from clipboard'}",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.PASTE,
                error=f"Paste failed: {str(e)}",
            )
    
    def read_clipboard(self) -> ActionResult:
        """
        Read current clipboard contents.
        
        Returns:
            ActionResult with clipboard content in data field
        """
        check = self._pre_action_check(ActionType.READ_CLIPBOARD)
        if check:
            return check
        
        if self.dry_run:
            logger.info("DRY-RUN: read_clipboard")
            return ActionResult(
                success=True,
                action_type=ActionType.READ_CLIPBOARD,
                message="Would read clipboard",
                data="[DRY-RUN: clipboard content]",
            )
        
        try:
            content = self._get_clipboard()
            return ActionResult(
                success=True,
                action_type=ActionType.READ_CLIPBOARD,
                message=f"Read {len(content) if content else 0} characters",
                data=content,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.READ_CLIPBOARD,
                error=f"Read clipboard failed: {str(e)}",
            )
    
    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> ActionResult:
        """
        Scroll the mouse wheel.
        
        Args:
            clicks: Number of "clicks" to scroll (positive = up, negative = down)
            x: X coordinate to scroll at (None = current position)
            y: Y coordinate to scroll at
        """
        check = self._pre_action_check(ActionType.SCROLL)
        if check:
            return check
        
        if self.dry_run:
            logger.info("DRY-RUN: scroll", clicks=clicks, x=x, y=y)
            return ActionResult(
                success=True,
                action_type=ActionType.SCROLL,
                message=f"Would scroll {clicks} clicks",
            )
        
        if not HAS_PYAUTOGUI:
            return ActionResult(
                success=False,
                action_type=ActionType.SCROLL,
                error="pyautogui not available",
            )
        
        try:
            pyautogui.scroll(clicks, x=x, y=y)
            return ActionResult(
                success=True,
                action_type=ActionType.SCROLL,
                message=f"Scrolled {clicks} clicks",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.SCROLL,
                error=f"Scroll failed: {str(e)}",
            )
    
    def wait(self, duration_ms: int) -> ActionResult:
        """
        Wait for specified duration.
        
        Args:
            duration_ms: Duration in milliseconds
        """
        if duration_ms > self.config.max_wait_ms:
            return ActionResult(
                success=False,
                action_type=ActionType.WAIT,
                error=f"Wait too long: {duration_ms}ms > {self.config.max_wait_ms}ms",
            )
        
        if duration_ms <= 0:
            return ActionResult(
                success=True,
                action_type=ActionType.WAIT,
                message="No wait (0ms)",
                duration_ms=0,
            )
        
        if self.dry_run:
            logger.info("DRY-RUN: wait", duration_ms=duration_ms)
            return ActionResult(
                success=True,
                action_type=ActionType.WAIT,
                message=f"Would wait {duration_ms}ms",
                duration_ms=0,
            )
        
        # Check kill switch periodically during long waits
        check_interval = 100  # ms
        elapsed = 0
        
        while elapsed < duration_ms:
            if self._check_kill_switch():
                return ActionResult(
                    success=False,
                    action_type=ActionType.WAIT,
                    error=f"Kill switch triggered during wait at {elapsed}ms",
                )
            
            sleep_time = min(check_interval, duration_ms - elapsed)
            time.sleep(sleep_time / 1000.0)
            elapsed += sleep_time
        
        return ActionResult(
            success=True,
            action_type=ActionType.WAIT,
            message=f"Waited {duration_ms}ms",
            duration_ms=duration_ms,
        )
    
    def _validate_coordinates(self, x: int, y: int) -> bool:
        """Validate that coordinates are within screen bounds."""
        # Allow some margin for DPI edge cases
        margin = 10
        return (
            -margin <= x <= self._screen_width + margin and
            -margin <= y <= self._screen_height + margin
        )
    
    def _get_clipboard(self) -> Optional[str]:
        """Get clipboard content."""
        if IS_WINDOWS:
            try:
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                    return data
                finally:
                    win32clipboard.CloseClipboard()
            except ImportError:
                pass
            except Exception as e:
                logger.debug("Win32 clipboard read failed", error=str(e))
        
        # Fallback to pyperclip if available
        try:
            import pyperclip
            return pyperclip.paste()
        except ImportError:
            pass
        
        # Last resort: pyautogui (may not work on all platforms)
        if HAS_PYAUTOGUI:
            try:
                return pyautogui.paste() if hasattr(pyautogui, 'paste') else None
            except Exception:
                pass
        
        return None
    
    def _set_clipboard(self, text: str) -> bool:
        """Set clipboard content."""
        if IS_WINDOWS:
            try:
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
                    return True
                finally:
                    win32clipboard.CloseClipboard()
            except ImportError:
                pass
            except Exception as e:
                logger.debug("Win32 clipboard write failed", error=str(e))
        
        # Fallback to pyperclip
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except ImportError:
            pass
        
        return False
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        if HAS_PYAUTOGUI:
            return pyautogui.position()
        return (0, 0)
