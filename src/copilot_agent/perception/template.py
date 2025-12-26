"""
Template matching using OpenCV.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Template match result."""
    
    template_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    center: Tuple[int, int]


class TemplateMatcher:
    """
    Template matching using OpenCV.
    
    Note: Full implementation in M3.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir
        self._templates: dict[str, bytes] = {}
        logger.info("TemplateMatcher initialized", templates_dir=str(templates_dir))
    
    def load_templates(self) -> int:
        """
        Load template images from templates directory.
        
        Returns:
            Number of templates loaded
        """
        # TODO: Implement in M3
        logger.warning("Template loading not implemented yet (M3)")
        return 0
    
    def find_template(
        self,
        image_path: str,
        template_name: str,
        confidence_threshold: float = 0.7,
    ) -> Optional[MatchResult]:
        """
        Find a template in an image.
        
        Args:
            image_path: Path to image to search
            template_name: Name of template to find
            confidence_threshold: Minimum match confidence
            
        Returns:
            MatchResult if found
        """
        # TODO: Implement in M3
        logger.warning("Template find not implemented yet (M3)")
        return None
    
    def find_all_templates(
        self,
        image_path: str,
        template_name: str,
        confidence_threshold: float = 0.7,
    ) -> List[MatchResult]:
        """
        Find all instances of a template in an image.
        
        Args:
            image_path: Path to image to search
            template_name: Name of template to find
            confidence_threshold: Minimum match confidence
            
        Returns:
            List of all matches
        """
        # TODO: Implement in M3
        logger.warning("Template find_all not implemented yet (M3)")
        return []
