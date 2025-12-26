"""
Window management with pywin32.

Windows-first implementation for focus, enumeration, and verification.
"""

import re
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass

from copilot_agent.logging import get_logger
from copilot_agent.actuator.platform import IS_WINDOWS, get_dpi_info

logger = get_logger(__name__)

# Windows-specific imports
if IS_WINDOWS:
    try:
        import win32gui
        import win32con
        import win32process
        import win32api
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        logger.warning("pywin32 not available, window operations will be limited")
else:
    HAS_WIN32 = False


@dataclass
class WindowInfo:
    """Information about a window."""
    
    hwnd: int
    title: str
    class_name: str
    x: int
    y: int
    width: int
    height: int
    is_visible: bool
    is_minimized: bool
    is_foreground: bool
    pid: Optional[int] = None
    
    # Alias for backwards compatibility
    @property
    def handle(self) -> int:
        return self.hwnd


class WindowManager:
    """
    Manages window focus and verification.
    
    Primary implementation uses pywin32 on Windows.
    Falls back to basic operations on other platforms.
    """
    
    def __init__(
        self, 
        dry_run: bool = False,
        allowed_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize WindowManager.
        
        Args:
            dry_run: If True, only log actions without executing
            allowed_patterns: Regex patterns for allowed windows
        """
        self.dry_run = dry_run
        self._dpi_info = get_dpi_info()
        
        self.allowed_patterns = allowed_patterns or [
            r"^Visual Studio Code$",
            r"^Code -",
            r".+ - Visual Studio Code$",
        ]
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.allowed_patterns
        ]
        
        if not IS_WINDOWS:
            logger.info("WindowManager: Running on non-Windows platform, limited functionality")
        elif not HAS_WIN32:
            logger.warning("WindowManager: pywin32 not installed")
    
    def find_window_by_title(
        self, 
        title_contains: str,
        exact: bool = False,
    ) -> Optional[WindowInfo]:
        """
        Find a window by title.
        
        Args:
            title_contains: Substring to search for in window title
            exact: If True, require exact match
            
        Returns:
            WindowInfo if found, None otherwise
        """
        if not IS_WINDOWS or not HAS_WIN32:
            logger.warning("find_window: pywin32 required for window search")
            return None
        
        found = []
        
        def enum_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            
            if exact:
                matches = title == title_contains
            else:
                matches = title_contains.lower() in title.lower()
            
            if matches:
                found.append(hwnd)
            
            return True
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            logger.error("Error enumerating windows", error=str(e))
            return None
        
        if not found:
            return None
        
        # Return the first match
        return self._get_window_info(found[0])
    
    def find_windows_by_title(
        self, 
        title_contains: str,
        exact: bool = False,
    ) -> List[WindowInfo]:
        """
        Find all windows matching title.
        
        Args:
            title_contains: Substring to search for
            exact: If True, require exact match
            
        Returns:
            List of WindowInfo for matching windows
        """
        if not IS_WINDOWS or not HAS_WIN32:
            return []
        
        results = []
        
        def enum_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            
            if exact:
                matches = title == title_contains
            else:
                matches = title_contains.lower() in title.lower()
            
            if matches:
                info = self._get_window_info(hwnd)
                if info:
                    results.append(info)
            
            return True
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            logger.error("Error enumerating windows", error=str(e))
        
        return results
    
    def find_vscode_window(self) -> Optional[WindowInfo]:
        """
        Find the VS Code window.
        
        Returns:
            WindowInfo if found, None otherwise
        """
        # Try common VS Code title patterns
        patterns = [
            "Visual Studio Code",
            "Code -",
            "- Visual Studio Code",
        ]
        
        for pattern in patterns:
            window = self.find_window_by_title(pattern)
            if window and self.is_allowed_window(window.title):
                return window
        
        return None
    
    def _get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """Get WindowInfo for a window handle."""
        if not HAS_WIN32:
            return None
        
        try:
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            width = x2 - x
            height = y2 - y
            
            is_visible = win32gui.IsWindowVisible(hwnd)
            is_minimized = win32gui.IsIconic(hwnd)
            is_foreground = win32gui.GetForegroundWindow() == hwnd
            
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            return WindowInfo(
                hwnd=hwnd,
                title=title,
                class_name=class_name,
                x=x,
                y=y,
                width=width,
                height=height,
                is_visible=is_visible,
                is_minimized=is_minimized,
                is_foreground=is_foreground,
                pid=pid,
            )
        except Exception as e:
            logger.error("Error getting window info", hwnd=hwnd, error=str(e))
            return None
    
    def focus_window(
        self, 
        window: Optional[WindowInfo] = None,
        hwnd: Optional[int] = None,
        title_contains: Optional[str] = None,
        restore_if_minimized: bool = True,
        verify: bool = True,
        retries: int = 3,
        retry_delay: float = 0.1,
    ) -> Tuple[bool, Optional[str]]:
        """
        Focus a window.
        
        Args:
            window: WindowInfo object (if known)
            hwnd: Window handle (if known)
            title_contains: Find window by title if hwnd not provided
            restore_if_minimized: Restore window if minimized
            verify: Verify window is focused after attempt
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tuple of (success, error_message)
        """
        # Extract hwnd from window if provided
        if window is not None:
            hwnd = window.hwnd
        
        # Find window if needed
        if hwnd is None and title_contains:
            info = self.find_window_by_title(title_contains)
            if info:
                hwnd = info.hwnd
            else:
                return (False, f"Window not found: {title_contains}")
        
        if hwnd is None:
            return (False, "No window specified")
        
        if self.dry_run:
            logger.info("DRY-RUN: focus_window", hwnd=hwnd)
            return (True, None)
        
        if not IS_WINDOWS or not HAS_WIN32:
            return (False, "pywin32 required for window focus")
        
        for attempt in range(retries):
            try:
                # Check if minimized and restore
                if restore_if_minimized and win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    time.sleep(0.05)
                
                # Try multiple focus methods
                # Method 1: SetForegroundWindow
                try:
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass
                
                # Method 2: Bring to top
                try:
                    win32gui.BringWindowToTop(hwnd)
                except Exception:
                    pass
                
                # Method 3: ShowWindow + SetFocus combo
                try:
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                except Exception:
                    pass
                
                # Short delay for focus to take effect
                time.sleep(0.05)
                
                if verify:
                    if win32gui.GetForegroundWindow() == hwnd:
                        logger.debug("Window focused successfully", hwnd=hwnd, attempt=attempt+1)
                        return (True, None)
                else:
                    return (True, None)
                
            except Exception as e:
                logger.warning("Focus attempt failed", hwnd=hwnd, attempt=attempt+1, error=str(e))
            
            if attempt < retries - 1:
                time.sleep(retry_delay)
        
        return (False, "Failed to focus window after retries")
    
    def verify_active_window(
        self,
        hwnd: Optional[int] = None,
        title_contains: Optional[str] = None,
    ) -> Tuple[bool, Optional[WindowInfo]]:
        """
        Verify a specific window is in foreground.
        
        Args:
            hwnd: Expected window handle
            title_contains: Expected title substring (if hwnd not provided)
            
        Returns:
            Tuple of (is_active, current_foreground_window_info)
        """
        if not IS_WINDOWS or not HAS_WIN32:
            # Non-Windows: always return True for dry-run compatibility
            return (True, None)
        
        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            fg_info = self._get_window_info(fg_hwnd)
            
            if hwnd is not None:
                return (fg_hwnd == hwnd, fg_info)
            
            if title_contains and fg_info:
                matches = title_contains.lower() in fg_info.title.lower()
                return (matches, fg_info)
            
            return (True, fg_info)
            
        except Exception as e:
            logger.error("Error verifying active window", error=str(e))
            return (False, None)
    
    def get_foreground_window(self) -> Optional[WindowInfo]:
        """Get the current foreground window."""
        if not IS_WINDOWS or not HAS_WIN32:
            return None
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            return self._get_window_info(hwnd)
        except Exception as e:
            logger.error("Error getting foreground window", error=str(e))
            return None
    
    def get_window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get window rectangle (x, y, width, height).
        
        Accounts for DPI scaling if needed.
        """
        if not IS_WINDOWS or not HAS_WIN32:
            return None
        
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            return (x, y, x2 - x, y2 - y)
        except Exception as e:
            logger.error("Error getting window rect", hwnd=hwnd, error=str(e))
            return None
    
    def get_client_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get window client area rectangle.
        
        Client rect is the area inside window borders.
        Returns (x, y, width, height) in screen coordinates.
        """
        if not IS_WINDOWS or not HAS_WIN32:
            return None
        
        try:
            import ctypes
            
            # Get client rect (relative to window)
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            
            # Convert to screen coordinates
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            
            pt = POINT(0, 0)
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
            
            return (pt.x, pt.y, right - left, bottom - top)
            
        except Exception as e:
            logger.error("Error getting client rect", hwnd=hwnd, error=str(e))
            return None
    
    def is_allowed_window(self, title: str) -> bool:
        """
        Check if window title matches allowed patterns.
        
        Args:
            title: Window title to check
            
        Returns:
            True if window is allowed
        """
        for pattern in self._compiled_patterns:
            if pattern.search(title):
                return True
        return False
    
    def verify_focus(self, expected_pattern: str) -> bool:
        """
        Verify that the foreground window matches expected pattern.
        
        Args:
            expected_pattern: Regex pattern to match
            
        Returns:
            True if foreground window matches
        """
        window = self.get_foreground_window()
        if not window:
            return False
        
        pattern = re.compile(expected_pattern, re.IGNORECASE)
        return bool(pattern.search(window.title))
    
    def list_visible_windows(self) -> List[WindowInfo]:
        """List all visible windows."""
        if not IS_WINDOWS or not HAS_WIN32:
            return []
        
        windows = []
        
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:  # Only windows with titles
                    info = self._get_window_info(hwnd)
                    if info:
                        windows.append(info)
            return True
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            logger.error("Error listing windows", error=str(e))
        
        return windows
