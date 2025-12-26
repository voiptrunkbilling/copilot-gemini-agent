"""
OCR engine using Tesseract.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextRegion:
    """Detected text region."""
    
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float


class OCREngine:
    """
    OCR using Tesseract via pytesseract.
    
    Note: Full implementation in M3.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        self.tesseract_path = tesseract_path
        logger.info("OCREngine initialized", tesseract_path=tesseract_path)
    
    def detect_text(
        self,
        image_path: str,
        confidence_threshold: float = 0.6,
    ) -> List[TextRegion]:
        """
        Detect all text in an image.
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for results
            
        Returns:
            List of detected text regions
        """
        # TODO: Implement in M3
        logger.warning("OCR detect_text not implemented yet (M3)")
        return []
    
    def find_text(
        self,
        image_path: str,
        target_text: str,
        fuzzy: bool = False,
    ) -> Optional[TextRegion]:
        """
        Find specific text in an image.
        
        Args:
            image_path: Path to image file
            target_text: Text to search for
            fuzzy: Allow fuzzy matching
            
        Returns:
            TextRegion if found
        """
        # TODO: Implement in M3
        logger.warning("OCR find_text not implemented yet (M3)")
        return None
    
    def read_region(
        self,
        image_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> str:
        """
        Read text from a specific region.
        
        Args:
            image_path: Path to image file
            x: Region X
            y: Region Y
            width: Region width
            height: Region height
            
        Returns:
            Extracted text
        """
        # TODO: Implement in M3
        logger.warning("OCR read_region not implemented yet (M3)")
        return ""
