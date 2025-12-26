"""
Perception module - screenshot, OCR, template matching, vision.
"""

from copilot_agent.perception.screenshot import ScreenCapture
from copilot_agent.perception.ocr import OCREngine
from copilot_agent.perception.template import TemplateMatcher
from copilot_agent.perception.vision import VisionFallback

__all__ = ["ScreenCapture", "OCREngine", "TemplateMatcher", "VisionFallback"]
