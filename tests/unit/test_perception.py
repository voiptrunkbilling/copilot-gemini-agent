"""
Unit tests for the perception pipeline modules.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest


class TestOCREngine:
    """Tests for OCR engine."""
    
    def test_ocr_engine_init(self):
        """Test OCR engine initialization."""
        from copilot_agent.perception.ocr import OCREngine
        
        engine = OCREngine()
        # Engine should initialize (may not be available if tesseract not installed)
        assert engine is not None
    
    def test_ocr_result_dataclass(self):
        """Test OCRResult dataclass."""
        from copilot_agent.perception.ocr import OCRResult
        
        result = OCRResult(
            success=True,
            text="Hello World",
            confidence=0.95,
            duration_ms=100,
        )
        
        assert result.success is True
        assert result.text == "Hello World"
        assert result.confidence == 0.95
        assert result.duration_ms == 100
        
        # Test to_dict
        d = result.to_dict()
        assert d["success"] is True
        assert d["text"] == "Hello World"
    
    def test_text_region_dataclass(self):
        """Test TextRegion dataclass."""
        from copilot_agent.perception.ocr import TextRegion
        
        region = TextRegion(
            text="test",
            x=10,
            y=20,
            width=100,
            height=50,
            confidence=0.8,
        )
        
        assert region.text == "test"
        assert region.center == (60, 45)
        # Verify basic coordinates
        assert region.x == 10
        assert region.y == 20
        assert region.width == 100
        assert region.height == 50
    
    def test_ocr_available_property(self):
        """Test available property."""
        from copilot_agent.perception.ocr import OCREngine
        
        engine = OCREngine()
        # available should be a boolean
        assert isinstance(engine.available, bool)


class TestImagePreprocessor:
    """Tests for image preprocessor."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        from copilot_agent.perception.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None
        # Preprocessor uses HAS_CV2 or HAS_PIL internally
    
    def test_preprocess_mode_enum(self):
        """Test PreprocessMode enum values."""
        from copilot_agent.perception.preprocessing import PreprocessMode
        
        # PreprocessMode uses auto() so check they are distinct
        assert PreprocessMode.GRAYSCALE != PreprocessMode.BINARY
        assert PreprocessMode.DENOISE != PreprocessMode.SHARPEN
        assert len(PreprocessMode) >= 4
    
    def test_preprocess_result_dataclass(self):
        """Test PreprocessResult dataclass."""
        from copilot_agent.perception.preprocessing import PreprocessResult
        
        result = PreprocessResult(
            success=True,
            output_path=Path("/tmp/test.png"),
        )
        
        assert result.success is True
        assert result.output_path == Path("/tmp/test.png")


class TestTemplateMatcher:
    """Tests for template matcher."""
    
    def test_matcher_init(self):
        """Test matcher initialization."""
        from copilot_agent.perception.template import TemplateMatcher
        
        matcher = TemplateMatcher()
        assert matcher is not None
        assert matcher.threshold == 0.8
    
    def test_matcher_with_custom_threshold(self):
        """Test matcher with custom threshold."""
        from copilot_agent.perception.template import TemplateMatcher
        
        matcher = TemplateMatcher(threshold=0.9)
        assert matcher.threshold == 0.9
    
    def test_match_result_dataclass(self):
        """Test MatchResult dataclass."""
        from copilot_agent.perception.template import MatchResult
        
        result = MatchResult(
            found=True,
            template_name="test_template",
            x=100,
            y=200,
            width=50,
            height=30,
            confidence=0.95,
        )
        
        assert result.found is True
        assert result.template_name == "test_template"
        assert result.center == (125, 215)
        assert result.region == (100, 200, 50, 30)
    
    def test_match_not_found(self):
        """Test match result when template not found."""
        from copilot_agent.perception.template import MatchResult
        
        result = MatchResult(
            found=False,
            template_name="missing",
        )
        
        assert result.found is False
        assert result.confidence == 0.0
    
    def test_get_template_names_empty(self):
        """Test get_template_names with no templates."""
        from copilot_agent.perception.template import TemplateMatcher
        
        matcher = TemplateMatcher()
        assert matcher.get_template_names() == []


class TestVisionFallback:
    """Tests for vision fallback."""
    
    def test_vision_init_no_api_key(self):
        """Test vision fallback initialization without API key."""
        from copilot_agent.perception.vision import VisionFallback
        
        # Clear any env var
        with patch.dict("os.environ", {}, clear=True):
            vision = VisionFallback()
            # Should not be available without API key
            # (unless google-generativeai is not installed)
    
    def test_vision_result_dataclass(self):
        """Test VisionResult dataclass."""
        from copilot_agent.perception.vision import VisionResult
        
        result = VisionResult(
            success=True,
            text="Extracted text",
            confidence=0.9,
            duration_ms=500,
            tokens_used=150,
        )
        
        assert result.success is True
        assert result.text == "Extracted text"
        assert result.confidence == 0.9
        assert result.tokens_used == 150
    
    def test_vision_rate_limits(self):
        """Test vision rate limit properties."""
        from copilot_agent.perception.vision import VisionFallback
        
        vision = VisionFallback(
            max_calls_per_iteration=5,
            max_calls_per_session=50,
        )
        
        assert vision.max_calls_per_iteration == 5
        assert vision.max_calls_per_session == 50
        assert vision.calls_remaining_iteration == 5
        assert vision.calls_remaining_session == 50
    
    def test_vision_reset_iteration(self):
        """Test reset_iteration."""
        from copilot_agent.perception.vision import VisionFallback
        
        vision = VisionFallback(max_calls_per_iteration=3)
        vision._iteration_calls = 3
        
        assert vision.calls_remaining_iteration == 0
        
        vision.reset_iteration()
        assert vision.calls_remaining_iteration == 3
    
    def test_vision_reset_session(self):
        """Test reset_session."""
        from copilot_agent.perception.vision import VisionFallback
        
        vision = VisionFallback(max_calls_per_iteration=3, max_calls_per_session=10)
        vision._iteration_calls = 3
        vision._session_calls = 10
        
        vision.reset_session()
        assert vision.calls_remaining_iteration == 3
        assert vision.calls_remaining_session == 10


class TestPerceptionPipeline:
    """Tests for perception pipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        from copilot_agent.perception.pipeline import PerceptionPipeline
        
        pipeline = PerceptionPipeline()
        assert pipeline is not None
    
    def test_pipeline_with_output_dir(self):
        """Test pipeline with output directory."""
        from copilot_agent.perception.pipeline import PerceptionPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = PerceptionPipeline(
                output_dir=tmpdir,
                save_captures=True,
            )
            assert pipeline.output_dir == Path(tmpdir)
            assert pipeline.save_captures is True
    
    def test_capture_method_enum(self):
        """Test CaptureMethod enum."""
        from copilot_agent.perception.pipeline import CaptureMethod
        
        assert CaptureMethod.OCR.value == "ocr"
        assert CaptureMethod.VISION.value == "vision"
        assert CaptureMethod.TEMPLATE.value == "template"
        assert CaptureMethod.HYBRID.value == "hybrid"
    
    def test_capture_result_dataclass(self):
        """Test CaptureResult dataclass."""
        from copilot_agent.perception.pipeline import CaptureResult, CaptureMethod
        
        result = CaptureResult(
            success=True,
            text="Hello World",
            method=CaptureMethod.OCR,
            confidence=0.9,
            duration_ms=200,
        )
        
        assert result.success is True
        assert result.text == "Hello World"
        assert result.method == CaptureMethod.OCR
        
        # Test to_dict
        d = result.to_dict()
        assert d["success"] is True
        assert d["method"] == "ocr"
    
    def test_normalize_text(self):
        """Test text normalization."""
        from copilot_agent.perception.pipeline import PerceptionPipeline
        
        pipeline = PerceptionPipeline()
        
        # Test basic normalization
        text = "Hello World  \n\n\n\nTest"
        normalized = pipeline._normalize_text(text)
        assert "\n\n\n" not in normalized
        
        # Test empty string
        assert pipeline._normalize_text("") == ""
        assert pipeline._normalize_text(None) == ""
    
    def test_validate_copilot_response(self):
        """Test response validation."""
        from copilot_agent.perception.pipeline import PerceptionPipeline
        
        pipeline = PerceptionPipeline()
        
        # Empty text should be invalid
        is_valid, confidence = pipeline.validate_copilot_response("")
        assert is_valid is False
        
        # Short text should be invalid
        is_valid, confidence = pipeline.validate_copilot_response("Hi")
        assert is_valid is False
        
        # Text with code block should have higher confidence
        text_with_code = """
        Here's the fix:
        ```python
        def hello():
            print("Hello")
        ```
        """
        is_valid, confidence = pipeline.validate_copilot_response(text_with_code)
        assert confidence > 0.5
    
    def test_get_stats(self):
        """Test get_stats."""
        from copilot_agent.perception.pipeline import PerceptionPipeline
        
        pipeline = PerceptionPipeline()
        stats = pipeline.get_stats()
        
        assert "total_captures" in stats
        assert "ocr_successes" in stats
        assert "vision_fallbacks" in stats
        assert stats["total_captures"] == 0
    
    def test_reset_stats(self):
        """Test reset_stats."""
        from copilot_agent.perception.pipeline import PerceptionPipeline
        
        pipeline = PerceptionPipeline()
        pipeline._capture_count = 10
        pipeline._ocr_success_count = 8
        
        pipeline.reset_stats()
        
        assert pipeline._capture_count == 0
        assert pipeline._ocr_success_count == 0


class TestIntegration:
    """Integration tests for perception modules."""
    
    def test_ocr_engine_extract_text_missing_file(self):
        """Test OCR with missing file."""
        from copilot_agent.perception.ocr import OCREngine
        
        engine = OCREngine()
        result = engine.extract_text(Path("/nonexistent/file.png"))
        
        assert result.success is False
        # Error message could be about tesseract not available or file not found
        assert result.error is not None
    
    def test_template_match_missing_template(self):
        """Test template match with unloaded template."""
        from copilot_agent.perception.template import TemplateMatcher
        
        matcher = TemplateMatcher()
        result = matcher.match(Path("/tmp/test.png"), "nonexistent")
        
        assert result.found is False
    
    def test_vision_extract_missing_file(self):
        """Test vision with missing file."""
        from copilot_agent.perception.vision import VisionFallback
        
        vision = VisionFallback(api_key="test_key")
        
        # This should handle missing file gracefully
        result = vision.extract_text(Path("/nonexistent/file.png"))
        assert result.success is False
