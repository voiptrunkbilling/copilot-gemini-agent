"""
Platform detection and DPI awareness utilities.

Windows-first implementation for M2.
"""

import sys
import ctypes
from typing import Tuple, Optional
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

# Platform detection
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"


@dataclass
class ScreenInfo:
    """Information about a screen/monitor."""
    
    index: int
    x: int
    y: int
    width: int
    height: int
    scale_factor: float  # DPI scale (1.0 = 100%, 1.25 = 125%, etc.)
    is_primary: bool = False


@dataclass 
class DPIInfo:
    """DPI awareness information."""
    
    is_aware: bool
    scale_factor: float
    system_dpi: int
    raw_dpi: Optional[Tuple[int, int]] = None  # (dpi_x, dpi_y)


def set_dpi_awareness() -> bool:
    """
    Set process DPI awareness on Windows.
    
    Must be called early in application startup.
    Returns True if successful.
    """
    if not IS_WINDOWS:
        logger.debug("DPI awareness: not Windows, skipping")
        return True
    
    try:
        # Try Windows 10+ per-monitor DPI awareness v2
        awareness = ctypes.c_int()
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE_V2
            logger.info("DPI awareness set: per-monitor v2")
            return True
        except (AttributeError, OSError):
            pass
        
        # Try Windows 8.1+ per-monitor DPI awareness
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            logger.info("DPI awareness set: per-monitor")
            return True
        except (AttributeError, OSError):
            pass
        
        # Fallback to system DPI aware (Windows Vista+)
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            logger.info("DPI awareness set: system")
            return True
        except (AttributeError, OSError):
            pass
        
        logger.warning("Could not set DPI awareness")
        return False
        
    except Exception as e:
        logger.error("Error setting DPI awareness", error=str(e))
        return False


def get_dpi_info() -> DPIInfo:
    """
    Get current DPI information.
    
    Returns DPIInfo with scale factor and awareness status.
    """
    if not IS_WINDOWS:
        return DPIInfo(is_aware=True, scale_factor=1.0, system_dpi=96)
    
    try:
        # Get system DPI
        hdc = ctypes.windll.user32.GetDC(0)
        dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        dpi_y = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
        ctypes.windll.user32.ReleaseDC(0, hdc)
        
        scale_factor = dpi_x / 96.0  # 96 DPI is 100%
        
        # Check DPI awareness
        try:
            awareness = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
            is_aware = awareness.value > 0
        except (AttributeError, OSError):
            is_aware = False
        
        return DPIInfo(
            is_aware=is_aware,
            scale_factor=scale_factor,
            system_dpi=dpi_x,
            raw_dpi=(dpi_x, dpi_y),
        )
        
    except Exception as e:
        logger.error("Error getting DPI info", error=str(e))
        return DPIInfo(is_aware=False, scale_factor=1.0, system_dpi=96)


def get_screen_info() -> list[ScreenInfo]:
    """
    Get information about all screens/monitors.
    
    Returns list of ScreenInfo for each monitor.
    """
    screens = []
    
    if IS_WINDOWS:
        try:
            import ctypes.wintypes
            
            # EnumDisplayMonitors callback
            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool,
                ctypes.c_void_p,  # hMonitor
                ctypes.c_void_p,  # hdcMonitor  
                ctypes.POINTER(ctypes.wintypes.RECT),  # lprcMonitor
                ctypes.c_void_p,  # dwData
            )
            
            monitors = []
            
            def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                monitors.append((hMonitor, lprcMonitor[0]))
                return True
            
            ctypes.windll.user32.EnumDisplayMonitors(
                None, None, MONITORENUMPROC(callback), 0
            )
            
            for i, (hMonitor, rect) in enumerate(monitors):
                # Get monitor info
                class MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", ctypes.wintypes.DWORD),
                        ("rcMonitor", ctypes.wintypes.RECT),
                        ("rcWork", ctypes.wintypes.RECT),
                        ("dwFlags", ctypes.wintypes.DWORD),
                    ]
                
                mi = MONITORINFO()
                mi.cbSize = ctypes.sizeof(MONITORINFO)
                ctypes.windll.user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
                
                is_primary = bool(mi.dwFlags & 1)  # MONITORINFOF_PRIMARY
                
                # Get DPI for this monitor
                try:
                    dpi_x = ctypes.c_uint()
                    dpi_y = ctypes.c_uint()
                    ctypes.windll.shcore.GetDpiForMonitor(
                        hMonitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)
                    )
                    scale = dpi_x.value / 96.0
                except (AttributeError, OSError):
                    scale = 1.0
                
                screens.append(ScreenInfo(
                    index=i,
                    x=rect.left,
                    y=rect.top,
                    width=rect.right - rect.left,
                    height=rect.bottom - rect.top,
                    scale_factor=scale,
                    is_primary=is_primary,
                ))
            
        except Exception as e:
            logger.error("Error enumerating monitors", error=str(e))
    
    # Fallback: use pyautogui for basic info
    if not screens:
        try:
            import pyautogui
            size = pyautogui.size()
            screens.append(ScreenInfo(
                index=0,
                x=0,
                y=0,
                width=size[0],
                height=size[1],
                scale_factor=1.0,
                is_primary=True,
            ))
        except ImportError:
            # Last resort fallback
            screens.append(ScreenInfo(
                index=0, x=0, y=0, width=1920, height=1080,
                scale_factor=1.0, is_primary=True,
            ))
    
    return screens


def get_primary_screen() -> ScreenInfo:
    """Get the primary screen info."""
    screens = get_screen_info()
    for screen in screens:
        if screen.is_primary:
            return screen
    return screens[0] if screens else ScreenInfo(
        index=0, x=0, y=0, width=1920, height=1080, scale_factor=1.0, is_primary=True
    )


def scale_coordinates(x: int, y: int, scale_factor: float) -> Tuple[int, int]:
    """Scale coordinates by DPI factor."""
    return (int(x / scale_factor), int(y / scale_factor))


def unscale_coordinates(x: int, y: int, scale_factor: float) -> Tuple[int, int]:
    """Unscale coordinates (physical to logical)."""
    return (int(x * scale_factor), int(y * scale_factor))
