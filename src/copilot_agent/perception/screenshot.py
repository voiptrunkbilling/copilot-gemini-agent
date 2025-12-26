"""
Screenshot capture using mss.
"""

from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Screenshot:
    """Captured screenshot data."""
    
    width: int
    height: int
    data: bytes
    path: Optional[Path] = None


class ScreenCapture:
    """
    Screenshot capture using mss library.
    
    Note: Full implementation in M2/M3.
    """
    
    def __init__(self):
        logger.info("ScreenCapture initialized")
    
    def capture_full_screen(self) -> Optional[Screenshot]:
        """
        Capture the entire screen.
        
        Returns:
            Screenshot object
        """
        # TODO: Implement in M2
        logger.warning("Full screen capture not implemented yet (M2)")
        return None
    
    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Optional[Screenshot]:
        """
        Capture a specific region.
        
        Args:
            x: Top-left X coordinate
            y: Top-left Y coordinate
            width: Region width
            height: Region height
            
        Returns:
            Screenshot object
        """
        # TODO: Implement in M2
        logger.warning("Region capture not implemented yet (M2)")
        return None
    
    def capture_window(self, window_handle: int) -> Optional[Screenshot]:
        """
        Capture a specific window.
        
        Args:
            window_handle: Window handle
            
        Returns:
            Screenshot object
        """
        # TODO: Implement in M2
        logger.warning("Window capture not implemented yet (M2)")
        return None
    
    def save_screenshot(
        self,
        screenshot: Screenshot,
        path: Path,
    ) -> bool:
        """
        Save screenshot to file.
        
        Args:
            screenshot: Screenshot to save
            path: Output path
            
        Returns:
            True if successful
        """
        # TODO: Implement in M2
        logger.warning("Save screenshot not implemented yet (M2)")
        return False
