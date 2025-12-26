"""
Template matching for UI element detection.

Uses OpenCV template matching to find UI anchors reliably.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

# Try to import OpenCV
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None
    np = None


@dataclass
class MatchResult:
    """Result from template matching."""
    
    found: bool
    template_name: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0
    method: str = "template"
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of match."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Get region as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "found": self.found,
            "template_name": self.template_name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "method": self.method,
        }


class TemplateMatcher:
    """
    Template matcher for finding UI elements.
    
    Uses OpenCV's template matching to locate UI anchors
    like Copilot icons, chat panels, and input fields.
    """
    
    # OpenCV matching methods
    METHOD_CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED if HAS_CV2 else 0
    METHOD_CCORR_NORMED = cv2.TM_CCORR_NORMED if HAS_CV2 else 0
    METHOD_SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED if HAS_CV2 else 0
    
    # Default confidence thresholds
    DEFAULT_THRESHOLD = 0.8
    HIGH_THRESHOLD = 0.9
    LOW_THRESHOLD = 0.6
    
    def __init__(
        self,
        template_dir: Optional[Union[str, Path]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        scale_factors: Optional[List[float]] = None,
    ):
        """
        Initialize template matcher.
        
        Args:
            template_dir: Directory containing template images
            threshold: Minimum confidence threshold (0-1)
            scale_factors: Scale factors for multi-scale matching
        """
        self.threshold = threshold
        self.scale_factors = scale_factors or [0.75, 0.875, 1.0, 1.125, 1.25]
        
        # Template cache: name -> (image, grayscale)
        self._templates: dict = {}
        
        # Load templates from directory
        if template_dir:
            self.template_dir = Path(template_dir)
            self._load_templates()
        else:
            self.template_dir = None
        
        logger.info(
            "TemplateMatcher initialized",
            available=HAS_CV2,
            template_dir=str(template_dir) if template_dir else None,
            threshold=threshold,
            templates=len(self._templates),
        )
    
    @property
    def available(self) -> bool:
        """Check if template matching is available."""
        return HAS_CV2
    
    def _load_templates(self) -> None:
        """Load all templates from directory."""
        if not self.template_dir or not self.template_dir.exists():
            return
        
        for pattern in ["*.png", "*.jpg", "*.jpeg"]:
            for path in self.template_dir.glob(pattern):
                self.load_template(path.stem, path)
    
    def load_template(
        self,
        name: str,
        path: Union[str, Path],
    ) -> bool:
        """
        Load a template image.
        
        Args:
            name: Template name for reference
            path: Path to template image
            
        Returns:
            True if loaded successfully
        """
        if not HAS_CV2:
            logger.warning("OpenCV not available for template loading")
            return False
        
        path = Path(path)
        if not path.exists():
            logger.warning("Template not found", name=name, path=str(path))
            return False
        
        try:
            # Load in color and grayscale
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Failed to read template", name=name)
                return False
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            self._templates[name] = (img, gray)
            logger.debug("Template loaded", name=name, size=img.shape[:2])
            return True
            
        except Exception as e:
            logger.error("Failed to load template", name=name, error=str(e))
            return False
    
    def add_template_from_image(
        self,
        name: str,
        image: "np.ndarray",
    ) -> bool:
        """
        Add a template from an in-memory image.
        
        Args:
            name: Template name
            image: OpenCV image array
            
        Returns:
            True if added successfully
        """
        if not HAS_CV2:
            return False
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            self._templates[name] = (image, gray)
            return True
        except Exception as e:
            logger.error("Failed to add template", name=name, error=str(e))
            return False
    
    def match(
        self,
        screenshot_path: Union[str, Path],
        template_name: str,
        threshold: Optional[float] = None,
        use_multiscale: bool = True,
    ) -> MatchResult:
        """
        Find a template in a screenshot.
        
        Args:
            screenshot_path: Path to screenshot image
            template_name: Name of loaded template
            threshold: Override confidence threshold
            use_multiscale: Try multiple scales
            
        Returns:
            MatchResult with location if found
        """
        if not HAS_CV2:
            return MatchResult(
                found=False,
                template_name=template_name,
            )
        
        if template_name not in self._templates:
            logger.warning("Template not loaded", name=template_name)
            return MatchResult(found=False, template_name=template_name)
        
        threshold = threshold if threshold is not None else self.threshold
        _, template_gray = self._templates[template_name]
        
        # Load screenshot
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            return MatchResult(found=False, template_name=template_name)
        
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            return MatchResult(found=False, template_name=template_name)
        
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        if use_multiscale:
            return self._multiscale_match(
                screenshot_gray,
                template_gray,
                template_name,
                threshold,
            )
        else:
            return self._single_match(
                screenshot_gray,
                template_gray,
                template_name,
                threshold,
            )
    
    def _single_match(
        self,
        screenshot_gray: "np.ndarray",
        template_gray: "np.ndarray",
        template_name: str,
        threshold: float,
    ) -> MatchResult:
        """Perform single-scale template matching."""
        th, tw = template_gray.shape[:2]
        
        try:
            result = cv2.matchTemplate(
                screenshot_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                return MatchResult(
                    found=True,
                    template_name=template_name,
                    x=max_loc[0],
                    y=max_loc[1],
                    width=tw,
                    height=th,
                    confidence=float(max_val),
                )
        except Exception as e:
            logger.error("Template match failed", error=str(e))
        
        return MatchResult(found=False, template_name=template_name)
    
    def _multiscale_match(
        self,
        screenshot_gray: "np.ndarray",
        template_gray: "np.ndarray",
        template_name: str,
        threshold: float,
    ) -> MatchResult:
        """Perform multi-scale template matching."""
        best_match = MatchResult(found=False, template_name=template_name)
        best_val = 0.0
        
        th, tw = template_gray.shape[:2]
        sh, sw = screenshot_gray.shape[:2]
        
        for scale in self.scale_factors:
            # Resize template
            new_w = int(tw * scale)
            new_h = int(th * scale)
            
            # Skip if template larger than screenshot
            if new_w > sw or new_h > sh:
                continue
            
            # Skip if template too small
            if new_w < 10 or new_h < 10:
                continue
            
            try:
                scaled = cv2.resize(template_gray, (new_w, new_h))
                
                result = cv2.matchTemplate(
                    screenshot_gray,
                    scaled,
                    cv2.TM_CCOEFF_NORMED,
                )
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_val:
                    best_val = max_val
                    if max_val >= threshold:
                        best_match = MatchResult(
                            found=True,
                            template_name=template_name,
                            x=max_loc[0],
                            y=max_loc[1],
                            width=new_w,
                            height=new_h,
                            confidence=float(max_val),
                        )
            except Exception as e:
                logger.debug("Scale match failed", scale=scale, error=str(e))
                continue
        
        if best_match.found:
            logger.debug(
                "Template matched",
                name=template_name,
                confidence=best_match.confidence,
                location=(best_match.x, best_match.y),
            )
        
        return best_match
    
    def match_any(
        self,
        screenshot_path: Union[str, Path],
        template_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """
        Find any matching template.
        
        Args:
            screenshot_path: Path to screenshot
            template_names: Templates to try (default: all)
            threshold: Confidence threshold
            
        Returns:
            Best MatchResult if any found
        """
        names = template_names or list(self._templates.keys())
        best_result: Optional[MatchResult] = None
        
        for name in names:
            result = self.match(screenshot_path, name, threshold)
            if result.found:
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
        
        return best_result
    
    def match_all(
        self,
        screenshot_path: Union[str, Path],
        template_name: str,
        threshold: Optional[float] = None,
        max_results: int = 10,
    ) -> List[MatchResult]:
        """
        Find all instances of a template.
        
        Args:
            screenshot_path: Path to screenshot
            template_name: Template to find
            threshold: Confidence threshold
            max_results: Maximum matches to return
            
        Returns:
            List of MatchResults
        """
        if not HAS_CV2:
            return []
        
        if template_name not in self._templates:
            return []
        
        threshold = threshold if threshold is not None else self.threshold
        _, template_gray = self._templates[template_name]
        
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            return []
        
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        th, tw = template_gray.shape[:2]
        
        try:
            result = cv2.matchTemplate(
                screenshot_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )
            
            # Find all matches above threshold
            locations = np.where(result >= threshold)
            matches = []
            
            for pt in zip(*locations[::-1]):
                matches.append(MatchResult(
                    found=True,
                    template_name=template_name,
                    x=pt[0],
                    y=pt[1],
                    width=tw,
                    height=th,
                    confidence=float(result[pt[1], pt[0]]),
                ))
            
            # Sort by confidence and apply non-max suppression
            matches.sort(key=lambda m: m.confidence, reverse=True)
            
            # Simple NMS: filter overlapping
            filtered = []
            for match in matches:
                overlap = False
                for existing in filtered:
                    dx = abs(match.x - existing.x)
                    dy = abs(match.y - existing.y)
                    if dx < tw // 2 and dy < th // 2:
                        overlap = True
                        break
                
                if not overlap:
                    filtered.append(match)
                    if len(filtered) >= max_results:
                        break
            
            return filtered
            
        except Exception as e:
            logger.error("Match all failed", error=str(e))
            return []
    
    def get_template_names(self) -> List[str]:
        """Get list of loaded template names."""
        return list(self._templates.keys())
