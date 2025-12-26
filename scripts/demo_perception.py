#!/usr/bin/env python
"""
M3 Perception Pipeline Demo Script.

Tests the full perception pipeline on Windows:
1. Takes a screenshot
2. Applies preprocessing
3. Runs OCR with Tesseract
4. Checks confidence
5. Falls back to Gemini Vision if needed
6. Saves captures and logs results

Run with:
    python scripts/demo_perception.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from copilot_agent.logging import get_logger, setup_logging
from copilot_agent.actuator.screenshot import ScreenshotCapture, Region
from copilot_agent.perception.preprocessing import ImagePreprocessor, PreprocessMode
from copilot_agent.perception.ocr import OCREngine
from copilot_agent.perception.template import TemplateMatcher
from copilot_agent.perception.vision import VisionFallback
from copilot_agent.perception.pipeline import PerceptionPipeline, CaptureMethod

# Configure logging
setup_logging(level="DEBUG")
logger = get_logger(__name__)


def create_output_dirs() -> tuple[Path, Path, Path]:
    """Create output directories for demo."""
    base = Path(__file__).parent.parent
    
    screenshots_dir = base / "screenshots"
    captures_dir = base / "captures"
    logs_dir = base / "logs"
    
    screenshots_dir.mkdir(exist_ok=True)
    captures_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    return screenshots_dir, captures_dir, logs_dir


def test_screenshot():
    """Test basic screenshot capture."""
    print("\n" + "="*60)
    print("TEST 1: Screenshot Capture")
    print("="*60)
    
    screenshots_dir, _, _ = create_output_dirs()
    
    capture = ScreenshotCapture()
    
    # Full screen capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = screenshots_dir / f"screenshot_{timestamp}.png"
    
    print("\n[1.1] Full screen capture...")
    result = capture.capture_full_screen(save_path=save_path)
    
    if result.success:
        print(f"  ✅ Screenshot saved: {result.path}")
        print(f"  📐 Size: {result.width}x{result.height}")
        print(f"  ⏱️  Duration: {result.duration_ms}ms")
        return result.path
    else:
        print(f"  ❌ Failed: {result.error}")
        return None


def test_preprocessing(screenshot_path: Path):
    """Test image preprocessing."""
    print("\n" + "="*60)
    print("TEST 2: Image Preprocessing")
    print("="*60)
    
    preprocessor = ImagePreprocessor()
    
    print(f"\n[2.1] Running full preprocessing pipeline on: {screenshot_path.name}")
    
    result = preprocessor.preprocess(screenshot_path, mode=PreprocessMode.FULL)
    
    if result.success:
        print(f"  ✅ Preprocessed: {result.output_path}")
        print(f"  📊 Method: full pipeline (grayscale → binary → denoise → sharpen)")
        return result.output_path
    else:
        print(f"  ⚠️  Preprocessing failed: {result.error}")
        print(f"  Using original screenshot instead")
        return screenshot_path


def test_ocr(image_path: Path):
    """Test OCR extraction."""
    print("\n" + "="*60)
    print("TEST 3: OCR Extraction (Tesseract)")
    print("="*60)
    
    ocr = OCREngine()
    
    print(f"\n[3.1] OCR Engine Status:")
    print(f"  Available: {ocr.available}")
    
    if not ocr.available:
        print("  ❌ Tesseract not available!")
        return None
    
    print(f"\n[3.2] Extracting text from: {image_path.name}")
    
    result = ocr.extract_text(image_path)
    
    print(f"\n[3.3] OCR Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Duration: {result.duration_ms}ms")
    print(f"  Word count: {result.word_count}")
    print(f"  Line count: {result.line_count}")
    
    if result.text:
        preview = result.text[:500] + "..." if len(result.text) > 500 else result.text
        print(f"\n[3.4] Text Preview (first 500 chars):")
        print("-" * 40)
        print(preview)
        print("-" * 40)
    
    return result


def test_vision_fallback():
    """Test Gemini Vision fallback availability."""
    print("\n" + "="*60)
    print("TEST 4: Gemini Vision Fallback")
    print("="*60)
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("\n  ⚠️  GEMINI_API_KEY not set - Vision fallback unavailable")
        print("  Set it with: $env:GEMINI_API_KEY = 'your-key'")
        return None
    
    vision = VisionFallback(api_key=api_key)
    
    print(f"\n[4.1] Vision Fallback Status:")
    print(f"  Available: {vision.available}")
    print(f"  Rate Limits: {vision.max_calls_per_iteration}/iter, {vision.max_calls_per_session}/session")
    print(f"  Remaining: {vision.calls_remaining_iteration}/iter, {vision.calls_remaining_session}/session")
    
    return vision


def test_full_pipeline():
    """Test the complete perception pipeline."""
    print("\n" + "="*60)
    print("TEST 5: Full Perception Pipeline")
    print("="*60)
    
    screenshots_dir, captures_dir, _ = create_output_dirs()
    
    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    vision = VisionFallback(api_key=api_key) if api_key else None
    
    pipeline = PerceptionPipeline(
        output_dir=captures_dir,
        save_captures=True,
        vision=vision,
    )
    
    print(f"\n[5.1] Pipeline Status:")
    print(f"  OCR Available: {pipeline.ocr.available}")
    print(f"  Vision Available: {vision.available if vision else False}")
    print(f"  Template Matching: {pipeline.template_matcher.available}")
    print(f"  Output Dir: {pipeline.output_dir}")
    
    print(f"\n[5.2] Running capture...")
    start = time.time()
    
    result = pipeline.capture_copilot_response(
        use_preprocessing=True,
        use_vision_fallback=True,
    )
    
    print(f"\n[5.3] Capture Result:")
    print(f"  Success: {result.success}")
    print(f"  Method: {result.method.value}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Duration: {result.duration_ms}ms")
    print(f"  Screenshot: {result.screenshot_path}")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    if result.text:
        preview = result.text[:300] + "..." if len(result.text) > 300 else result.text
        print(f"\n[5.4] Captured Text Preview (first 300 chars):")
        print("-" * 40)
        print(preview)
        print("-" * 40)
    
    # Save result to captures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_file = captures_dir / f"capture_{timestamp}.json"
    
    with open(capture_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\n[5.5] Saved capture to: {capture_file}")
    
    # Save full text
    if result.text:
        text_file = captures_dir / f"capture_{timestamp}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(result.text)
        print(f"  Saved text to: {text_file}")
    
    # Pipeline stats
    stats = pipeline.get_stats()
    print(f"\n[5.6] Pipeline Stats:")
    print(f"  Total Captures: {stats['total_captures']}")
    print(f"  OCR Successes: {stats['ocr_successes']}")
    print(f"  Vision Fallbacks: {stats['vision_fallbacks']}")
    
    return result


def run_demo():
    """Run the full perception demo."""
    print("\n" + "="*60)
    print("  M3 PERCEPTION PIPELINE DEMO")
    print("  Windows Validation - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    # Create output directories
    screenshots_dir, captures_dir, logs_dir = create_output_dirs()
    print(f"\n📁 Output Directories:")
    print(f"  Screenshots: {screenshots_dir}")
    print(f"  Captures: {captures_dir}")
    print(f"  Logs: {logs_dir}")
    
    # Test 1: Screenshot
    screenshot_path = test_screenshot()
    if not screenshot_path:
        print("\n❌ Demo aborted: Screenshot failed")
        return False
    
    # Test 2: Preprocessing
    processed_path = test_preprocessing(screenshot_path)
    
    # Test 3: OCR
    ocr_result = test_ocr(processed_path)
    
    # Test 4: Vision Fallback
    test_vision_fallback()
    
    # Test 5: Full Pipeline
    pipeline_result = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("  DEMO SUMMARY")
    print("="*60)
    
    ocr_conf_str = f"{ocr_result.confidence:.2%}" if ocr_result else "N/A"
    ocr_status = "✅" if ocr_result and ocr_result.success else "❌"
    vision_status = "✅" if os.environ.get("GEMINI_API_KEY") else "⚠️ (no API key)"
    pipeline_status = "✅" if pipeline_result and pipeline_result.success else "❌"
    
    print(f"""
  Screenshot:     ✅
  Preprocessing:  ✅
  OCR (Tesseract): {ocr_status} (confidence: {ocr_conf_str})
  Vision Fallback: {vision_status}
  Full Pipeline:  {pipeline_status}
  
  📂 Outputs saved to:
     - screenshots/
     - captures/
  
  {"✅ M3 PERCEPTION VALIDATED" if pipeline_status == "✅" else "⚠️  Review results above"}
""")
    
    return pipeline_result and pipeline_result.success


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
