"""
Vision fallback using Gemini Vision API.
"""

from typing import Optional, Tuple
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VisionResult:
    """Result from vision API."""
    
    element_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    description: str


class VisionFallback:
    """
    Vision fallback using Gemini Vision API.
    
    Note: Full implementation in M3.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        max_calls_per_iteration: int = 3,
        max_calls_per_session: int = 20,
    ):
        self.api_key = api_key
        self.model = model
        self.max_calls_per_iteration = max_calls_per_iteration
        self.max_calls_per_session = max_calls_per_session
        
        self._session_calls = 0
        self._iteration_calls = 0
        
        logger.info(
            "VisionFallback initialized",
            model=model,
            max_per_iteration=max_calls_per_iteration,
            max_per_session=max_calls_per_session,
        )
    
    def reset_iteration(self) -> None:
        """Reset iteration call counter."""
        self._iteration_calls = 0
    
    def can_call(self) -> bool:
        """Check if vision API can be called."""
        return (
            self._iteration_calls < self.max_calls_per_iteration
            and self._session_calls < self.max_calls_per_session
        )
    
    def find_element(
        self,
        image_path: str,
        element_description: str,
    ) -> Optional[VisionResult]:
        """
        Find an element in an image using vision API.
        
        Args:
            image_path: Path to screenshot
            element_description: What to look for (e.g., "Copilot Chat input field")
            
        Returns:
            VisionResult if found
        """
        if not self.can_call():
            logger.warning(
                "Vision API call limit reached",
                iteration_calls=self._iteration_calls,
                session_calls=self._session_calls,
            )
            return None
        
        # TODO: Implement in M3
        logger.warning("Vision find_element not implemented yet (M3)")
        return None
    
    def read_text(
        self,
        image_path: str,
        region_description: str,
    ) -> Optional[str]:
        """
        Read text from a region using vision API.
        
        Args:
            image_path: Path to screenshot
            region_description: Where to read from
            
        Returns:
            Extracted text
        """
        if not self.can_call():
            logger.warning("Vision API call limit reached")
            return None
        
        # TODO: Implement in M3
        logger.warning("Vision read_text not implemented yet (M3)")
        return None
