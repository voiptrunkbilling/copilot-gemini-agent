# M3: Perception Pipeline (Eyes)

## Summary
Implements the complete perception pipeline for capturing and extracting text from VS Code / Copilot Chat UI.

## Changes

### Core Modules
- **preprocessing.py** - Image preprocessing with OpenCV/PIL (grayscale, binary, denoise, sharpen, 2x upscale)
- **ocr.py** - Full Tesseract OCR engine with confidence scoring, text regions, word detection
- **template.py** - OpenCV template matching with multi-scale support
- **vision.py** - Gemini Vision API fallback with rate limiting (3/iter, 20/session)
- **pipeline.py** - Pipeline orchestrator: screenshot → preprocess → OCR → validate → vision fallback → normalize

### Demo & Validation
- **scripts/demo_perception.py** - Windows validation script with full pipeline test
- **.gitignore** - Added output directories (screenshots/, captures/, logs/)

### Dependencies
- `numpy>=1.24.0`
- `google-generativeai>=0.3.0`

## Windows Validation Results

| Component | Status | Details |
|-----------|--------|---------|
| Tesseract | ✅ | v5.4.0 installed via winget |
| Screenshot | ✅ | 1920x1080, ~130ms (mss) |
| Preprocessing | ✅ | 2x upscale + full pipeline |
| OCR Extraction | ✅ | 88-90% confidence |
| Template Matching | ✅ | OpenCV available |
| Vision Fallback | ⚠️ | Requires GEMINI_API_KEY |
| Full Pipeline | ✅ | End-to-end working |

## Test Results
```
============================= 128 passed in 1.34s =============================
```

## Captures Output
Pipeline saves captures to:
- `captures/capture_XXXX_screenshot.png` - Raw screenshot
- `captures/capture_YYYYMMDD_HHMMSS.json` - Structured result
- `captures/capture_YYYYMMDD_HHMMSS.txt` - Extracted text

## Demo Output
```
  M3 PERCEPTION PIPELINE DEMO
  Windows Validation - 2025-12-26

  Screenshot:     ✅
  Preprocessing:  ✅
  OCR (Tesseract): ✅ (confidence: 88.73%)
  Vision Fallback: ⚠️ (no API key)
  Full Pipeline:  ✅

  ✅ M3 PERCEPTION VALIDATED
```

## How to Test
```powershell
# Install Tesseract (if not installed)
winget install --id UB-Mannheim.TesseractOCR -e

# Run demo
python scripts/demo_perception.py

# Run tests
pytest tests/unit/test_perception.py -v
```

## Next Steps
M4: Reviewer Loop (Brain) - Gemini text reviewer integration
