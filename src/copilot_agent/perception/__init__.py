"""
Perception module - screenshot, OCR, template matching, vision, and pipeline.
"""

from copilot_agent.perception.screenshot import ScreenCapture
from copilot_agent.perception.ocr import OCREngine, OCRResult, TextRegion
from copilot_agent.perception.preprocessing import (
    ImagePreprocessor,
    PreprocessMode,
    PreprocessResult,
)
from copilot_agent.perception.template import TemplateMatcher, MatchResult
from copilot_agent.perception.vision import VisionFallback, VisionResult
from copilot_agent.perception.pipeline import (
    PerceptionPipeline,
    CaptureResult,
    CaptureMethod,
)

__all__ = [
    # Screenshot
    "ScreenCapture",
    # OCR
    "OCREngine",
    "OCRResult",
    "TextRegion",
    # Preprocessing
    "ImagePreprocessor",
    "PreprocessMode",
    "PreprocessResult",
    # Template Matching
    "TemplateMatcher",
    "MatchResult",
    # Vision Fallback
    "VisionFallback",
    "VisionResult",
    # Pipeline
    "PerceptionPipeline",
    "CaptureResult",
    "CaptureMethod",
]
