"""
OCR engine using Tesseract.

Provides text extraction from screenshots with confidence scoring.
"""

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

from copilot_agent.logging import get_logger
from copilot_agent.perception.preprocessing import (
    ImagePreprocessor,
    PreprocessMode,
)

logger = get_logger(__name__)

# Try to import pytesseract
try:
    import pytesseract
    from pytesseract import Output
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None
    Output = None

# Try to import PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


@dataclass
class TextRegion:
    """Detected text region with bounding box."""
    
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of region."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


@dataclass
class OCRResult:
    """Complete OCR result with full text and regions."""
    
    success: bool
    text: str = ""
    confidence: float = 0.0
    regions: List[TextRegion] = field(default_factory=list)
    word_count: int = 0
    line_count: int = 0
    duration_ms: int = 0
    preprocessed: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "text": self.text,
            "confidence": self.confidence,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "duration_ms": self.duration_ms,
            "preprocessed": self.preprocessed,
            "error": self.error,
        }


class OCREngine:
    """
    OCR engine using Tesseract.
    
    Extracts text from images with confidence scoring and
    automatic preprocessing for better accuracy.
    """
    
    # Default Tesseract config for screen text
    DEFAULT_CONFIG = "--oem 3 --psm 6"  # LSTM + assume block of text
    
    # Minimum confidence threshold
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        lang: str = "eng",
        preprocess: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize OCR engine.
        
        Args:
            tesseract_path: Path to tesseract executable (optional)
            lang: OCR language(s), e.g., "eng" or "eng+chi_sim"
            preprocess: Whether to preprocess images before OCR
            confidence_threshold: Minimum confidence for word inclusion
        """
        self.lang = lang
        self.preprocess_enabled = preprocess
        self.confidence_threshold = confidence_threshold
        
        # Set tesseract path if provided
        if tesseract_path and HAS_TESSERACT:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Try to find tesseract on Windows
        if HAS_TESSERACT and not tesseract_path:
            self._find_tesseract()
        
        # Initialize preprocessor
        self._preprocessor = ImagePreprocessor() if preprocess else None
        
        # Verify tesseract is available
        self._available = self._check_tesseract()
        
        logger.info(
            "OCREngine initialized",
            available=self._available,
            lang=lang,
            preprocess=preprocess,
        )
    
    @property
    def available(self) -> bool:
        """Check if OCR is available."""
        return self._available
    
    def _find_tesseract(self) -> None:
        """Find tesseract installation on Windows."""
        import sys
        if sys.platform != "win32":
            return
        
        # Common Windows paths
        paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
                Path.home().name
            ),
        ]
        
        for path in paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                logger.debug("Found Tesseract", path=path)
                return
    
    def _check_tesseract(self) -> bool:
        """Check if tesseract is available."""
        if not HAS_TESSERACT:
            logger.warning("pytesseract not installed")
            return False
        
        if not HAS_PIL:
            logger.warning("PIL not installed")
            return False
        
        try:
            version = pytesseract.get_tesseract_version()
            logger.debug("Tesseract version", version=str(version))
            return True
        except Exception as e:
            logger.warning("Tesseract not available", error=str(e))
            return False
    
    def extract_text(
        self,
        image_path: Union[str, Path],
        config: Optional[str] = None,
        preprocess_mode: Optional[PreprocessMode] = None,
    ) -> OCRResult:
        """
        Extract all text from an image.
        
        Args:
            image_path: Path to image file
            config: Tesseract config string (optional)
            preprocess_mode: Preprocessing mode (optional)
            
        Returns:
            OCRResult with extracted text and confidence
        """
        if not self._available:
            return OCRResult(
                success=False,
                error="Tesseract not available",
            )
        
        image_path = Path(image_path)
        if not image_path.exists():
            return OCRResult(
                success=False,
                error=f"Image not found: {image_path}",
            )
        
        start = time.time()
        config = config or self.DEFAULT_CONFIG
        preprocessed = False
        
        try:
            # Preprocess if enabled
            process_path = image_path
            if self.preprocess_enabled and self._preprocessor:
                mode = preprocess_mode or PreprocessMode.FULL
                result = self._preprocessor.preprocess(image_path, mode=mode)
                if result.success and result.output_path:
                    process_path = result.output_path
                    preprocessed = True
            
            # Open image
            img = Image.open(process_path)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=config,
                output_type=Output.DICT,
            )
            
            # Parse results
            regions = []
            confidences = []
            texts = []
            
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                
                if text and conf >= self.confidence_threshold * 100:
                    region = TextRegion(
                        text=text,
                        x=data["left"][i],
                        y=data["top"][i],
                        width=data["width"][i],
                        height=data["height"][i],
                        confidence=conf / 100.0,
                    )
                    regions.append(region)
                    texts.append(text)
                    confidences.append(conf)
            
            # Build full text preserving line structure
            full_text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=config,
            ).strip()
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Count lines and words
            lines = [l for l in full_text.split("\n") if l.strip()]
            words = full_text.split()
            
            duration_ms = int((time.time() - start) * 1000)
            
            # Clean up temp preprocessed file
            if preprocessed and process_path != image_path:
                try:
                    process_path.unlink()
                except:
                    pass
            
            return OCRResult(
                success=True,
                text=full_text,
                confidence=avg_confidence / 100.0,
                regions=regions,
                word_count=len(words),
                line_count=len(lines),
                duration_ms=duration_ms,
                preprocessed=preprocessed,
            )
            
        except Exception as e:
            logger.error("OCR extraction failed", error=str(e))
            return OCRResult(
                success=False,
                error=f"OCR failed: {str(e)}",
                duration_ms=int((time.time() - start) * 1000),
            )
    
    def detect_text(
        self,
        image_path: Union[str, Path],
        confidence_threshold: Optional[float] = None,
    ) -> List[TextRegion]:
        """
        Detect all text regions in an image.
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for results
            
        Returns:
            List of detected text regions
        """
        result = self.extract_text(image_path)
        if not result.success:
            return []
        
        threshold = confidence_threshold or self.confidence_threshold
        return [r for r in result.regions if r.confidence >= threshold]
    
    def find_text(
        self,
        image_path: Union[str, Path],
        target_text: str,
        fuzzy: bool = False,
        case_sensitive: bool = False,
    ) -> Optional[TextRegion]:
        """
        Find specific text in an image.
        
        Args:
            image_path: Path to image file
            target_text: Text to search for
            fuzzy: Allow partial/fuzzy matching
            case_sensitive: Case-sensitive search
            
        Returns:
            TextRegion if found, None otherwise
        """
        result = self.extract_text(image_path)
        if not result.success:
            return None
        
        search_text = target_text if case_sensitive else target_text.lower()
        
        for region in result.regions:
            region_text = region.text if case_sensitive else region.text.lower()
            
            if fuzzy:
                if search_text in region_text or region_text in search_text:
                    return region
            else:
                if region_text == search_text:
                    return region
        
        return None
    
    def read_region(
        self,
        image_path: Union[str, Path],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> OCRResult:
        """
        Read text from a specific region of an image.
        
        Args:
            image_path: Path to image file
            x: Region X coordinate
            y: Region Y coordinate
            width: Region width
            height: Region height
            
        Returns:
            OCRResult with extracted text
        """
        if not self._available:
            return OCRResult(
                success=False,
                error="Tesseract not available",
            )
        
        image_path = Path(image_path)
        if not image_path.exists():
            return OCRResult(
                success=False,
                error=f"Image not found: {image_path}",
            )
        
        try:
            # Crop region first
            if self._preprocessor:
                crop_result = self._preprocessor.crop_region(
                    image_path, x, y, width, height
                )
                if not crop_result.success:
                    return OCRResult(
                        success=False,
                        error=crop_result.error,
                    )
                
                # OCR the cropped region
                result = self.extract_text(crop_result.output_path)
                
                # Clean up
                try:
                    crop_result.output_path.unlink()
                except:
                    pass
                
                return result
            else:
                # Fallback: crop with PIL
                img = Image.open(image_path)
                cropped = img.crop((x, y, x + width, y + height))
                
                # Save to temp file
                temp_path = image_path.with_stem(f"{image_path.stem}_region")
                cropped.save(temp_path)
                
                result = self.extract_text(temp_path)
                
                try:
                    temp_path.unlink()
                except:
                    pass
                
                return result
                
        except Exception as e:
            return OCRResult(
                success=False,
                error=f"Region OCR failed: {str(e)}",
            )
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get human-readable confidence level.
        
        Args:
            confidence: Confidence score (0.0 - 1.0)
            
        Returns:
            Level string: "high", "medium", "low", or "very_low"
        """
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"
