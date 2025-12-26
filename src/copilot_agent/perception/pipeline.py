"""
Perception pipeline orchestrator.

Coordinates screenshot capture, preprocessing, OCR, template matching,
and vision fallback to reliably extract Copilot responses.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from copilot_agent.logging import get_logger
from copilot_agent.actuator.screenshot import ScreenshotCapture, Region
from copilot_agent.perception.preprocessing import ImagePreprocessor, PreprocessMode
from copilot_agent.perception.ocr import OCREngine, OCRResult
from copilot_agent.perception.template import TemplateMatcher, MatchResult
from copilot_agent.perception.vision import VisionFallback, VisionResult

logger = get_logger(__name__)


class CaptureMethod(Enum):
    """Method used to capture text."""
    OCR = "ocr"
    VISION = "vision"
    TEMPLATE = "template"
    HYBRID = "hybrid"


@dataclass
class CaptureResult:
    """Result from the perception pipeline."""
    
    success: bool
    text: str = ""
    method: CaptureMethod = CaptureMethod.OCR
    confidence: float = 0.0
    duration_ms: int = 0
    screenshot_path: Optional[Path] = None
    ocr_result: Optional[OCRResult] = None
    vision_result: Optional[VisionResult] = None
    template_result: Optional[MatchResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "text": self.text,
            "method": self.method.value,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "screenshot_path": str(self.screenshot_path) if self.screenshot_path else None,
            "metadata": self.metadata,
            "error": self.error,
        }


class PerceptionPipeline:
    """
    Orchestrates the perception pipeline for capturing Copilot responses.
    
    Pipeline stages:
    1. Screenshot capture (mss)
    2. Optional crop to Copilot panel region
    3. Preprocessing for OCR
    4. OCR with confidence check
    5. Vision fallback if OCR fails
    6. Text normalization and validation
    """
    
    # Confidence thresholds
    OCR_CONFIDENCE_THRESHOLD = 0.65
    VISION_CONFIDENCE_THRESHOLD = 0.75
    
    # Text validation patterns
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
    COPILOT_MARKERS = ["copilot", "github", "suggestion", "fix", "refactor"]
    
    def __init__(
        self,
        screenshot: Optional[ScreenshotCapture] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        ocr: Optional[OCREngine] = None,
        template_matcher: Optional[TemplateMatcher] = None,
        vision: Optional[VisionFallback] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_captures: bool = True,
    ):
        """
        Initialize the perception pipeline.
        
        Args:
            screenshot: Screenshot capture instance
            preprocessor: Image preprocessor instance
            ocr: OCR engine instance
            template_matcher: Template matcher instance
            vision: Vision fallback instance
            output_dir: Directory to save captures
            save_captures: Whether to save screenshots and results
        """
        # Initialize components
        self.screenshot = screenshot or ScreenshotCapture()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.ocr = ocr or OCREngine()
        self.template_matcher = template_matcher or TemplateMatcher()
        self.vision = vision
        
        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_captures = save_captures
        
        if self.output_dir and self.save_captures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self._capture_count = 0
        self._ocr_success_count = 0
        self._vision_fallback_count = 0
        
        logger.info(
            "PerceptionPipeline initialized",
            ocr_available=self.ocr.available,
            vision_available=self.vision.available if self.vision else False,
            template_available=self.template_matcher.available,
            output_dir=str(self.output_dir) if self.output_dir else None,
        )
    
    def capture_copilot_response(
        self,
        region: Optional[Region] = None,
        use_preprocessing: bool = True,
        use_vision_fallback: bool = True,
        timeout_ms: int = 5000,
    ) -> CaptureResult:
        """
        Capture and extract Copilot response text.
        
        This is the main entry point for capturing Copilot responses.
        
        Args:
            region: Optional region to capture (defaults to full screen)
            use_preprocessing: Apply preprocessing before OCR
            use_vision_fallback: Use vision API if OCR fails
            timeout_ms: Maximum time to spend
            
        Returns:
            CaptureResult with extracted text
        """
        start = time.time()
        self._capture_count += 1
        capture_id = f"capture_{self._capture_count:04d}"
        
        logger.info("Starting capture", capture_id=capture_id)
        
        try:
            # Stage 1: Screenshot
            screenshot_result = self.screenshot.capture(region=region)
            
            if not screenshot_result.success:
                return CaptureResult(
                    success=False,
                    error=f"Screenshot failed: {screenshot_result.error}",
                    duration_ms=int((time.time() - start) * 1000),
                )
            
            screenshot_path = screenshot_result.path
            
            # Save screenshot if configured
            if self.save_captures and self.output_dir:
                saved_path = self.output_dir / f"{capture_id}_screenshot.png"
                if screenshot_path != saved_path:
                    import shutil
                    shutil.copy(screenshot_path, saved_path)
                    screenshot_path = saved_path
            
            # Stage 2: Preprocessing
            if use_preprocessing:
                preprocess_result = self.preprocessor.full_pipeline(screenshot_path)
                if preprocess_result.success:
                    processed_path = preprocess_result.output_path
                else:
                    processed_path = screenshot_path
                    logger.warning("Preprocessing failed, using original")
            else:
                processed_path = screenshot_path
            
            # Stage 3: OCR
            ocr_result = self.ocr.extract_text(processed_path)
            
            # Check OCR confidence
            if ocr_result.success and ocr_result.confidence >= self.OCR_CONFIDENCE_THRESHOLD:
                self._ocr_success_count += 1
                
                # Normalize text
                normalized_text = self._normalize_text(ocr_result.text)
                
                logger.info(
                    "OCR capture successful",
                    capture_id=capture_id,
                    confidence=ocr_result.confidence,
                    chars=len(normalized_text),
                )
                
                return CaptureResult(
                    success=True,
                    text=normalized_text,
                    method=CaptureMethod.OCR,
                    confidence=ocr_result.confidence,
                    duration_ms=int((time.time() - start) * 1000),
                    screenshot_path=screenshot_path,
                    ocr_result=ocr_result,
                    metadata={
                        "capture_id": capture_id,
                        "preprocessed": use_preprocessing,
                    },
                )
            
            # Stage 4: Vision fallback
            if use_vision_fallback and self.vision and self.vision.available:
                elapsed = (time.time() - start) * 1000
                if elapsed < timeout_ms:
                    self._vision_fallback_count += 1
                    vision_result = self.vision.extract_text(screenshot_path)
                    
                    if vision_result.success:
                        normalized_text = self._normalize_text(vision_result.text)
                        
                        logger.info(
                            "Vision fallback successful",
                            capture_id=capture_id,
                            confidence=vision_result.confidence,
                            chars=len(normalized_text),
                        )
                        
                        return CaptureResult(
                            success=True,
                            text=normalized_text,
                            method=CaptureMethod.VISION,
                            confidence=vision_result.confidence,
                            duration_ms=int((time.time() - start) * 1000),
                            screenshot_path=screenshot_path,
                            ocr_result=ocr_result,
                            vision_result=vision_result,
                            metadata={
                                "capture_id": capture_id,
                                "ocr_confidence": ocr_result.confidence if ocr_result else 0,
                                "fallback_reason": "low_ocr_confidence",
                            },
                        )
            
            # Return OCR result even with low confidence
            if ocr_result.success:
                normalized_text = self._normalize_text(ocr_result.text)
                return CaptureResult(
                    success=True,
                    text=normalized_text,
                    method=CaptureMethod.OCR,
                    confidence=ocr_result.confidence,
                    duration_ms=int((time.time() - start) * 1000),
                    screenshot_path=screenshot_path,
                    ocr_result=ocr_result,
                    metadata={
                        "capture_id": capture_id,
                        "low_confidence": True,
                    },
                )
            
            return CaptureResult(
                success=False,
                error="All extraction methods failed",
                duration_ms=int((time.time() - start) * 1000),
                screenshot_path=screenshot_path,
                ocr_result=ocr_result,
            )
            
        except Exception as e:
            logger.error("Capture failed", error=str(e), capture_id=capture_id)
            return CaptureResult(
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )
    
    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> CaptureResult:
        """
        Capture text from a specific screen region.
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Region width
            height: Region height
            
        Returns:
            CaptureResult with extracted text
        """
        region = Region(x=x, y=y, width=width, height=height)
        return self.capture_copilot_response(region=region)
    
    def find_and_capture(
        self,
        template_name: str,
        offset: Tuple[int, int] = (0, 0),
        capture_size: Tuple[int, int] = (500, 300),
    ) -> CaptureResult:
        """
        Find a template and capture text near it.
        
        Args:
            template_name: Name of template to find
            offset: Offset from template location (x, y)
            capture_size: Size of capture region (width, height)
            
        Returns:
            CaptureResult with extracted text
        """
        start = time.time()
        
        # Take full screenshot first
        screenshot_result = self.screenshot.capture()
        if not screenshot_result.success:
            return CaptureResult(
                success=False,
                error="Screenshot failed for template matching",
            )
        
        # Find template
        match = self.template_matcher.match(
            screenshot_result.path,
            template_name,
        )
        
        if not match.found:
            return CaptureResult(
                success=False,
                error=f"Template '{template_name}' not found",
                screenshot_path=screenshot_result.path,
            )
        
        # Calculate capture region relative to template
        x = match.x + offset[0]
        y = match.y + offset[1]
        width = capture_size[0]
        height = capture_size[1]
        
        # Capture the region
        result = self.capture_region(x, y, width, height)
        result.template_result = match
        result.metadata["template_matched"] = template_name
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize extracted text.
        
        - Fixes common OCR errors
        - Normalizes whitespace
        - Preserves code blocks
        """
        if not text:
            return ""
        
        # Common OCR substitutions
        substitutions = {
            "0": "O",  # Be careful with these
            "l": "I",
            "|": "l",
        }
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Normalize multiple blank lines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        
        # Strip trailing whitespace from lines while preserving indentation
        lines = []
        for line in text.split("\n"):
            lines.append(line.rstrip())
        
        return "\n".join(lines).strip()
    
    def validate_copilot_response(self, text: str) -> Tuple[bool, float]:
        """
        Validate that text looks like a Copilot response.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, confidence)
        """
        if not text or len(text) < 10:
            return False, 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check for code blocks
        if self.CODE_BLOCK_PATTERN.search(text):
            confidence += 0.2
        
        # Check for Copilot markers
        text_lower = text.lower()
        for marker in self.COPILOT_MARKERS:
            if marker in text_lower:
                confidence += 0.1
                break
        
        # Check for reasonable length
        if len(text) > 50:
            confidence += 0.1
        
        # Check for code-like content
        code_indicators = ["def ", "function", "class ", "const ", "let ", "var "]
        for indicator in code_indicators:
            if indicator in text:
                confidence += 0.1
                break
        
        return confidence >= 0.6, min(confidence, 1.0)
    
    def save_capture_result(
        self,
        result: CaptureResult,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Save capture result to disk.
        
        Args:
            result: CaptureResult to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not self.output_dir:
            return None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = filename or f"capture_{self._capture_count:04d}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return filepath
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_captures": self._capture_count,
            "ocr_successes": self._ocr_success_count,
            "vision_fallbacks": self._vision_fallback_count,
            "ocr_success_rate": (
                self._ocr_success_count / self._capture_count
                if self._capture_count > 0 else 0
            ),
            "vision_rate": (
                self._vision_fallback_count / self._capture_count
                if self._capture_count > 0 else 0
            ),
        }
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._capture_count = 0
        self._ocr_success_count = 0
        self._vision_fallback_count = 0
