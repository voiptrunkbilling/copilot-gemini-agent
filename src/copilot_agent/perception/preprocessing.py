"""
Image preprocessing for OCR.

Applies various image transformations to improve OCR accuracy.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

# Try to import image processing libraries
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None
    np = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


class PreprocessMode(Enum):
    """Image preprocessing modes."""
    
    NONE = auto()           # No preprocessing
    GRAYSCALE = auto()      # Convert to grayscale
    BINARY = auto()         # Binary threshold
    ADAPTIVE = auto()       # Adaptive thresholding
    DENOISE = auto()        # Denoising
    SHARPEN = auto()        # Sharpening
    FULL = auto()           # Full preprocessing pipeline


@dataclass
class PreprocessResult:
    """Result of image preprocessing."""
    
    success: bool
    output_path: Optional[Path] = None
    original_size: Optional[Tuple[int, int]] = None
    processed_size: Optional[Tuple[int, int]] = None
    mode: PreprocessMode = PreprocessMode.NONE
    error: Optional[str] = None


class ImagePreprocessor:
    """
    Preprocesses images for OCR.
    
    Applies grayscale, thresholding, denoising, and other
    transformations to improve Tesseract accuracy.
    """
    
    def __init__(
        self,
        upscale_factor: float = 2.0,
        denoise_strength: int = 10,
        sharpen_amount: float = 1.5,
    ):
        """
        Initialize preprocessor.
        
        Args:
            upscale_factor: Factor to upscale image (improves small text)
            denoise_strength: Denoising strength (0-30)
            sharpen_amount: Sharpening factor (1.0 = none)
        """
        self.upscale_factor = upscale_factor
        self.denoise_strength = denoise_strength
        self.sharpen_amount = sharpen_amount
        
        if not HAS_CV2:
            logger.warning("OpenCV not available, preprocessing limited")
        
        logger.debug(
            "ImagePreprocessor initialized",
            upscale=upscale_factor,
            denoise=denoise_strength,
        )
    
    def preprocess(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        mode: PreprocessMode = PreprocessMode.FULL,
    ) -> PreprocessResult:
        """
        Preprocess an image for OCR.
        
        Args:
            image_path: Path to input image
            output_path: Path to save processed image (optional)
            mode: Preprocessing mode
            
        Returns:
            PreprocessResult with output path
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return PreprocessResult(
                success=False,
                error=f"Image not found: {image_path}",
            )
        
        if not HAS_CV2:
            # Fallback to PIL-only preprocessing
            return self._preprocess_pil(image_path, output_path, mode)
        
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return PreprocessResult(
                    success=False,
                    error=f"Failed to read image: {image_path}",
                )
            
            original_size = (img.shape[1], img.shape[0])
            
            # Apply preprocessing based on mode
            if mode == PreprocessMode.NONE:
                processed = img
            elif mode == PreprocessMode.GRAYSCALE:
                processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif mode == PreprocessMode.BINARY:
                processed = self._apply_binary_threshold(img)
            elif mode == PreprocessMode.ADAPTIVE:
                processed = self._apply_adaptive_threshold(img)
            elif mode == PreprocessMode.DENOISE:
                processed = self._apply_denoise(img)
            elif mode == PreprocessMode.SHARPEN:
                processed = self._apply_sharpen(img)
            elif mode == PreprocessMode.FULL:
                processed = self._apply_full_pipeline(img)
            else:
                processed = img
            
            processed_size = (
                processed.shape[1] if len(processed.shape) > 1 else processed.shape[0],
                processed.shape[0],
            )
            
            # Save processed image
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), processed)
            else:
                # Save to temp path
                output_path = image_path.with_stem(f"{image_path.stem}_processed")
                cv2.imwrite(str(output_path), processed)
            
            logger.debug(
                "Image preprocessed",
                mode=mode.name,
                original=original_size,
                processed=processed_size,
            )
            
            return PreprocessResult(
                success=True,
                output_path=output_path,
                original_size=original_size,
                processed_size=processed_size,
                mode=mode,
            )
            
        except Exception as e:
            logger.error("Preprocessing failed", error=str(e))
            return PreprocessResult(
                success=False,
                error=f"Preprocessing failed: {str(e)}",
            )
    
    def _apply_binary_threshold(self, img: "np.ndarray") -> "np.ndarray":
        """Apply simple binary threshold."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def _apply_adaptive_threshold(self, img: "np.ndarray") -> "np.ndarray":
        """Apply adaptive thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2,
        )
        return adaptive
    
    def _apply_denoise(self, img: "np.ndarray") -> "np.ndarray":
        """Apply denoising."""
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, self.denoise_strength)
        return cv2.fastNlMeansDenoisingColored(img, None, self.denoise_strength)
    
    def _apply_sharpen(self, img: "np.ndarray") -> "np.ndarray":
        """Apply sharpening."""
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1],
        ]) * self.sharpen_amount
        return cv2.filter2D(img, -1, kernel)
    
    def _apply_full_pipeline(self, img: "np.ndarray") -> "np.ndarray":
        """
        Apply full preprocessing pipeline optimized for screen text OCR.
        
        Steps:
        1. Upscale (if factor > 1)
        2. Convert to grayscale
        3. Denoise
        4. Increase contrast
        5. Apply adaptive threshold for clean text
        """
        # 1. Upscale for small text
        if self.upscale_factor > 1.0:
            width = int(img.shape[1] * self.upscale_factor)
            height = int(img.shape[0] * self.upscale_factor)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, self.denoise_strength, 7, 21)
        
        # 4. Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # 5. Light sharpening
        kernel = np.array([
            [0, -1, 0],
            [-1,  5, -1],
            [0, -1, 0],
        ])
        sharpened = cv2.filter2D(contrast, -1, kernel)
        
        return sharpened
    
    def _preprocess_pil(
        self,
        image_path: Path,
        output_path: Optional[Path],
        mode: PreprocessMode,
    ) -> PreprocessResult:
        """Fallback preprocessing using PIL."""
        if not HAS_PIL:
            return PreprocessResult(
                success=False,
                error="Neither OpenCV nor PIL available",
            )
        
        try:
            img = Image.open(image_path)
            original_size = img.size
            
            if mode == PreprocessMode.GRAYSCALE or mode == PreprocessMode.FULL:
                img = img.convert("L")
            
            if mode == PreprocessMode.SHARPEN or mode == PreprocessMode.FULL:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(self.sharpen_amount)
            
            if mode == PreprocessMode.FULL:
                # Increase contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
            
            # Save
            if output_path:
                output_path = Path(output_path)
            else:
                output_path = image_path.with_stem(f"{image_path.stem}_processed")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            
            return PreprocessResult(
                success=True,
                output_path=output_path,
                original_size=original_size,
                processed_size=img.size,
                mode=mode,
            )
            
        except Exception as e:
            return PreprocessResult(
                success=False,
                error=f"PIL preprocessing failed: {str(e)}",
            )
    
    def crop_region(
        self,
        image_path: Union[str, Path],
        x: int,
        y: int,
        width: int,
        height: int,
        output_path: Optional[Union[str, Path]] = None,
    ) -> PreprocessResult:
        """
        Crop a region from an image.
        
        Args:
            image_path: Path to input image
            x: Left coordinate
            y: Top coordinate
            width: Crop width
            height: Crop height
            output_path: Path to save cropped image
            
        Returns:
            PreprocessResult
        """
        image_path = Path(image_path)
        
        if not HAS_CV2 and not HAS_PIL:
            return PreprocessResult(
                success=False,
                error="No image library available",
            )
        
        try:
            if HAS_CV2:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"Could not read image: {image_path}")
                
                # Validate bounds
                h, w = img.shape[:2]
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                x2 = min(x + width, w)
                y2 = min(y + height, h)
                
                cropped = img[y:y2, x:x2]
                
                if output_path:
                    output_path = Path(output_path)
                else:
                    output_path = image_path.with_stem(f"{image_path.stem}_cropped")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), cropped)
                
            else:
                # PIL fallback
                img = Image.open(image_path)
                w, h = img.size
                
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                x2 = min(x + width, w)
                y2 = min(y + height, h)
                
                cropped = img.crop((x, y, x2, y2))
                
                if output_path:
                    output_path = Path(output_path)
                else:
                    output_path = image_path.with_stem(f"{image_path.stem}_cropped")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(output_path)
            
            return PreprocessResult(
                success=True,
                output_path=output_path,
                original_size=(w, h),
                processed_size=(x2 - x, y2 - y),
            )
            
        except Exception as e:
            return PreprocessResult(
                success=False,
                error=f"Crop failed: {str(e)}",
            )
