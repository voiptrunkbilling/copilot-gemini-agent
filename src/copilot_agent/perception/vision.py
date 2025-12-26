"""
Vision fallback using Gemini Vision API.

Provides reliable text extraction when OCR fails or has low confidence.
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

# Try to import Google AI SDK
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None


@dataclass
class VisionResult:
    """Result from vision API."""
    
    success: bool
    text: str = ""
    confidence: float = 0.0
    method: str = "vision"
    duration_ms: int = 0
    tokens_used: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "text": self.text,
            "confidence": self.confidence,
            "method": self.method,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


class VisionFallback:
    """
    Vision fallback using Gemini Vision API.
    
    Used when OCR fails or has low confidence. Includes
    rate limiting and cost controls.
    """
    
    # System prompt for text extraction
    EXTRACT_PROMPT = """You are a precise OCR assistant. Extract ALL text visible in this screenshot of a code editor or chat interface.

Rules:
1. Extract text EXACTLY as shown - preserve formatting, indentation, newlines
2. Include code blocks, markdown, and any visible text
3. Do NOT add explanations or commentary
4. Do NOT skip any visible text
5. If text is partially visible or unclear, include your best interpretation in [brackets]
6. Return ONLY the extracted text, nothing else

Extract all text from this image:"""
    
    # Default rate limits
    DEFAULT_MAX_PER_ITERATION = 3
    DEFAULT_MAX_PER_SESSION = 20
    DEFAULT_COOLDOWN_SECONDS = 5
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        max_calls_per_iteration: int = DEFAULT_MAX_PER_ITERATION,
        max_calls_per_session: int = DEFAULT_MAX_PER_SESSION,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    ):
        """
        Initialize vision fallback.
        
        Args:
            api_key: Gemini API key (or use GEMINI_API_KEY env var)
            model: Model to use for vision
            max_calls_per_iteration: Max vision calls per iteration
            max_calls_per_session: Max vision calls per session
            cooldown_seconds: Minimum time between calls
        """
        self.model_name = model
        self.max_calls_per_iteration = max_calls_per_iteration
        self.max_calls_per_session = max_calls_per_session
        self.cooldown_seconds = cooldown_seconds
        
        # Get API key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Rate limiting state
        self._session_calls = 0
        self._iteration_calls = 0
        self._last_call_time: Optional[float] = None
        
        # Initialize client
        self._model = None
        self._available = self._initialize()
        
        logger.info(
            "VisionFallback initialized",
            available=self._available,
            model=model,
            max_per_iteration=max_calls_per_iteration,
            max_per_session=max_calls_per_session,
        )
    
    @property
    def available(self) -> bool:
        """Check if vision API is available."""
        return self._available
    
    @property
    def calls_remaining_iteration(self) -> int:
        """Get remaining calls for current iteration."""
        return max(0, self.max_calls_per_iteration - self._iteration_calls)
    
    @property
    def calls_remaining_session(self) -> int:
        """Get remaining calls for session."""
        return max(0, self.max_calls_per_session - self._session_calls)
    
    def _initialize(self) -> bool:
        """Initialize the Gemini client."""
        if not HAS_GENAI:
            logger.warning("google-generativeai not installed")
            return False
        
        if not self.api_key:
            logger.warning("No Gemini API key provided")
            return False
        
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.debug("Gemini Vision client initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize Gemini", error=str(e))
            return False
    
    def reset_iteration(self) -> None:
        """Reset iteration call counter."""
        self._iteration_calls = 0
        logger.debug("Vision iteration counter reset")
    
    def reset_session(self) -> None:
        """Reset session call counter."""
        self._session_calls = 0
        self._iteration_calls = 0
        logger.debug("Vision session counter reset")
    
    def can_call(self) -> bool:
        """Check if vision API can be called."""
        if not self._available:
            return False
        
        if self._iteration_calls >= self.max_calls_per_iteration:
            logger.debug("Vision iteration limit reached")
            return False
        
        if self._session_calls >= self.max_calls_per_session:
            logger.debug("Vision session limit reached")
            return False
        
        # Check cooldown
        if self._last_call_time:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.cooldown_seconds:
                logger.debug(
                    "Vision cooldown active",
                    remaining=self.cooldown_seconds - elapsed,
                )
                return False
        
        return True
    
    def extract_text(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> VisionResult:
        """
        Extract text from an image using Gemini Vision.
        
        Args:
            image_path: Path to image file
            prompt: Custom prompt (optional, uses default)
            
        Returns:
            VisionResult with extracted text
        """
        if not self._available:
            return VisionResult(
                success=False,
                error="Gemini Vision not available",
            )
        
        if not self.can_call():
            return VisionResult(
                success=False,
                error=f"Rate limit: {self._iteration_calls}/{self.max_calls_per_iteration} iter, {self._session_calls}/{self.max_calls_per_session} session",
            )
        
        image_path = Path(image_path)
        if not image_path.exists():
            return VisionResult(
                success=False,
                error=f"Image not found: {image_path}",
            )
        
        start = time.time()
        prompt = prompt or self.EXTRACT_PROMPT
        
        try:
            # Read and encode image
            image_data = image_path.read_bytes()
            
            # Determine MIME type
            suffix = image_path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/png")
            
            # Create image part
            image_part = {
                "mime_type": mime_type,
                "data": image_data,
            }
            
            # Make API call
            response = self._model.generate_content([prompt, image_part])
            
            # Update counters
            self._iteration_calls += 1
            self._session_calls += 1
            self._last_call_time = time.time()
            
            duration_ms = int((time.time() - start) * 1000)
            
            # Extract text from response
            if response.text:
                text = response.text.strip()
                
                # Estimate tokens (rough approximation)
                tokens = len(text.split()) + 100  # Include image tokens
                
                logger.info(
                    "Vision extraction successful",
                    chars=len(text),
                    duration_ms=duration_ms,
                    iter_calls=self._iteration_calls,
                    session_calls=self._session_calls,
                )
                
                return VisionResult(
                    success=True,
                    text=text,
                    confidence=0.85,  # Vision typically high confidence
                    duration_ms=duration_ms,
                    tokens_used=tokens,
                )
            else:
                return VisionResult(
                    success=False,
                    error="Empty response from Gemini",
                    duration_ms=duration_ms,
                )
                
        except Exception as e:
            logger.error("Vision extraction failed", error=str(e))
            return VisionResult(
                success=False,
                error=f"Vision API error: {str(e)}",
                duration_ms=int((time.time() - start) * 1000),
            )
    
    def find_element(
        self,
        image_path: Union[str, Path],
        element_description: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a UI element in an image.
        
        Args:
            image_path: Path to screenshot
            element_description: What to look for
            
        Returns:
            Tuple of (x, y, width, height) if found
        """
        if not self.can_call():
            return None
        
        prompt = f"""Find the UI element described below in this screenshot and return its bounding box coordinates.

Element: {element_description}

Return ONLY the coordinates in this exact format:
x: <left>
y: <top>
width: <width>
height: <height>

If the element is not found, return:
NOT_FOUND"""
        
        result = self.extract_text(Path(image_path), prompt)
        
        if not result.success:
            return None
        
        # Parse coordinates
        try:
            lines = result.text.strip().split("\n")
            if "NOT_FOUND" in result.text:
                return None
            
            coords = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    coords[key.strip().lower()] = int(value.strip())
            
            if all(k in coords for k in ["x", "y", "width", "height"]):
                return (
                    coords["x"],
                    coords["y"],
                    coords["width"],
                    coords["height"],
                )
        except Exception as e:
            logger.warning("Failed to parse element coordinates", error=str(e))
        
        return None
    
    def get_usage_stats(self) -> dict:
        """Get current usage statistics."""
        return {
            "iteration_calls": self._iteration_calls,
            "session_calls": self._session_calls,
            "max_per_iteration": self.max_calls_per_iteration,
            "max_per_session": self.max_calls_per_session,
            "can_call": self.can_call(),
        }
