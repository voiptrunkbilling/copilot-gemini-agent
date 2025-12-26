"""
Screenshot capture using mss.

Fast cross-platform screenshot capture with region support.
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from copilot_agent.logging import get_logger
from copilot_agent.actuator.platform import IS_WINDOWS, get_primary_screen

logger = get_logger(__name__)

# Try to import mss
try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    logger.warning("mss not available, screenshots will be limited")

# Fallback to PIL/pyautogui
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


@dataclass
class Region:
    """Screen region definition."""
    
    x: int
    y: int
    width: int
    height: int
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        return self.y + self.height
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def to_mss_dict(self) -> dict:
        """Return as mss monitor dict format."""
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        """Create from dictionary."""
        return cls(
            x=d.get("x", d.get("left", 0)),
            y=d.get("y", d.get("top", 0)),
            width=d["width"],
            height=d["height"],
        )


@dataclass
class ScreenshotResult:
    """Result of screenshot capture."""
    
    success: bool
    path: Optional[Path] = None
    region: Optional[Region] = None
    width: int = 0
    height: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    # Raw image data (for in-memory processing)
    raw_data: Optional[bytes] = None


class ScreenshotCapture:
    """
    Captures screenshots using mss for performance.
    
    Supports full screen and region capture, with fallbacks.
    """
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize screenshot capture.
        
        Args:
            dry_run: If True, don't actually capture screenshots
        """
        self.dry_run = dry_run
        self._screen = get_primary_screen()
        
        # Determine available capture method
        if HAS_MSS:
            self._method = "mss"
        elif HAS_PYAUTOGUI:
            self._method = "pyautogui"
        else:
            self._method = None
        
        logger.info(
            "ScreenshotCapture initialized",
            method=self._method,
            dry_run=dry_run,
        )
    
    def capture_full_screen(
        self,
        save_path: Optional[Union[str, Path]] = None,
        monitor_index: int = 0,
    ) -> ScreenshotResult:
        """
        Capture full screen.
        
        Args:
            save_path: Path to save screenshot (optional)
            monitor_index: Index of monitor to capture (0 = primary)
            
        Returns:
            ScreenshotResult with path and/or raw data
        """
        if self.dry_run:
            logger.info("DRY-RUN: capture_full_screen", monitor=monitor_index)
            return ScreenshotResult(
                success=True,
                region=Region(0, 0, self._screen.width, self._screen.height),
                width=self._screen.width,
                height=self._screen.height,
            )
        
        if self._method is None:
            return ScreenshotResult(
                success=False,
                error="No screenshot library available",
            )
        
        start = time.time()
        
        try:
            if self._method == "mss":
                return self._capture_mss(None, save_path, monitor_index)
            else:
                return self._capture_pyautogui(None, save_path)
                
        except Exception as e:
            logger.error("Screenshot failed", error=str(e))
            return ScreenshotResult(
                success=False,
                error=f"Screenshot failed: {str(e)}",
            )
    
    def capture_region(
        self,
        region: Union[Region, Tuple[int, int, int, int], dict],
        save_path: Optional[Union[str, Path]] = None,
    ) -> ScreenshotResult:
        """
        Capture a specific screen region.
        
        Args:
            region: Region to capture (Region, tuple, or dict)
            save_path: Path to save screenshot (optional)
            
        Returns:
            ScreenshotResult with path and/or raw data
        """
        # Normalize region
        if isinstance(region, tuple):
            region = Region(*region)
        elif isinstance(region, dict):
            region = Region.from_dict(region)
        
        # Validate region
        if region.width <= 0 or region.height <= 0:
            return ScreenshotResult(
                success=False,
                region=region,
                error=f"Invalid region size: {region.width}x{region.height}",
            )
        
        if self.dry_run:
            logger.info("DRY-RUN: capture_region", region=region.to_tuple())
            return ScreenshotResult(
                success=True,
                region=region,
                width=region.width,
                height=region.height,
            )
        
        if self._method is None:
            return ScreenshotResult(
                success=False,
                region=region,
                error="No screenshot library available",
            )
        
        try:
            if self._method == "mss":
                return self._capture_mss(region, save_path)
            else:
                return self._capture_pyautogui(region, save_path)
                
        except Exception as e:
            logger.error("Region capture failed", region=region.to_tuple(), error=str(e))
            return ScreenshotResult(
                success=False,
                region=region,
                error=f"Capture failed: {str(e)}",
            )
    
    def _capture_mss(
        self,
        region: Optional[Region],
        save_path: Optional[Union[str, Path]],
        monitor_index: int = 0,
    ) -> ScreenshotResult:
        """Capture using mss."""
        start = time.time()
        
        with mss.mss() as sct:
            if region:
                # Capture specific region
                monitor = region.to_mss_dict()
            else:
                # Capture full monitor
                # mss monitors: 0 = all monitors, 1+ = individual
                monitor = sct.monitors[monitor_index + 1] if monitor_index < len(sct.monitors) - 1 else sct.monitors[1]
            
            img = sct.grab(monitor)
            
            # Save if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                mss.tools.to_png(img.rgb, img.size, output=str(save_path))
            
            duration_ms = int((time.time() - start) * 1000)
            
            result_region = region or Region(
                x=monitor.get("left", 0),
                y=monitor.get("top", 0),
                width=monitor["width"],
                height=monitor["height"],
            )
            
            return ScreenshotResult(
                success=True,
                path=save_path,
                region=result_region,
                width=img.width,
                height=img.height,
                duration_ms=duration_ms,
                raw_data=img.rgb,
            )
    
    def _capture_pyautogui(
        self,
        region: Optional[Region],
        save_path: Optional[Union[str, Path]],
    ) -> ScreenshotResult:
        """Capture using pyautogui."""
        start = time.time()
        
        if region:
            img = pyautogui.screenshot(region=region.to_tuple())
        else:
            img = pyautogui.screenshot()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(save_path))
        
        duration_ms = int((time.time() - start) * 1000)
        
        result_region = region or Region(0, 0, img.width, img.height)
        
        return ScreenshotResult(
            success=True,
            path=save_path,
            region=result_region,
            width=img.width,
            height=img.height,
            duration_ms=duration_ms,
        )
    
    def capture_window(
        self,
        hwnd: int,
        save_path: Optional[Union[str, Path]] = None,
        client_area_only: bool = True,
    ) -> ScreenshotResult:
        """
        Capture a specific window.
        
        Args:
            hwnd: Window handle
            save_path: Path to save screenshot
            client_area_only: If True, capture only client area (no title bar)
            
        Returns:
            ScreenshotResult
        """
        if not IS_WINDOWS:
            return ScreenshotResult(
                success=False,
                error="Window capture only supported on Windows",
            )
        
        if self.dry_run:
            logger.info("DRY-RUN: capture_window", hwnd=hwnd)
            return ScreenshotResult(success=True, width=800, height=600)
        
        try:
            from copilot_agent.actuator.window import WindowManager
            
            wm = WindowManager(dry_run=False)
            
            if client_area_only:
                rect = wm.get_client_rect(hwnd)
            else:
                rect = wm.get_window_rect(hwnd)
            
            if not rect:
                return ScreenshotResult(
                    success=False,
                    error=f"Could not get window rect for hwnd={hwnd}",
                )
            
            x, y, w, h = rect
            return self.capture_region(Region(x, y, w, h), save_path)
            
        except Exception as e:
            return ScreenshotResult(
                success=False,
                error=f"Window capture failed: {str(e)}",
            )
