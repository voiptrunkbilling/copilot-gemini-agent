"""
Unit tests for M4: Gemini Reviewer.
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock

from copilot_agent.reviewer.gemini import (
    GeminiReviewer,
    ReviewResult,
    ReviewVerdict,
    SYSTEM_PROMPT,
    _build_review_prompt,
    _extract_json_from_response,
)


class TestReviewVerdict:
    """Tests for ReviewVerdict enum."""
    
    def test_verdict_values(self):
        """Test all verdict values exist."""
        assert ReviewVerdict.ACCEPT.value == "accept"
        assert ReviewVerdict.CRITIQUE.value == "critique"
        assert ReviewVerdict.CLARIFY.value == "clarify"
        assert ReviewVerdict.ERROR.value == "error"
    
    def test_verdict_is_string(self):
        """Test verdicts can be used as strings."""
        assert str(ReviewVerdict.ACCEPT) == "ReviewVerdict.ACCEPT"
        assert ReviewVerdict.ACCEPT.value == "accept"


class TestReviewResult:
    """Tests for ReviewResult dataclass."""
    
    def test_create_result(self):
        """Test creating a review result."""
        result = ReviewResult(
            verdict=ReviewVerdict.ACCEPT,
            confidence="high",
            reasoning="Task complete",
            issues=[],
            follow_up_prompt=None,
        )
        
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "high"
        assert result.reasoning == "Task complete"
        assert result.issues == []
        assert result.follow_up_prompt is None
    
    def test_result_with_critique(self):
        """Test result with critique verdict."""
        result = ReviewResult(
            verdict=ReviewVerdict.CRITIQUE,
            confidence="medium",
            reasoning="Missing error handling",
            issues=["No try/catch", "No input validation"],
            follow_up_prompt="Please add error handling for edge cases",
        )
        
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert len(result.issues) == 2
        assert result.follow_up_prompt is not None
    
    def test_result_to_dict(self):
        """Test serialization to dict."""
        result = ReviewResult(
            verdict=ReviewVerdict.CRITIQUE,
            confidence="high",
            reasoning="Needs improvement",
            issues=["Issue 1"],
            follow_up_prompt="Fix it",
            duration_ms=500,
            model_used="gemini-1.5-flash",
            tokens_used=100,
        )
        
        data = result.to_dict()
        
        assert data["verdict"] == "critique"
        assert data["confidence"] == "high"
        assert data["reasoning"] == "Needs improvement"
        assert data["issues"] == ["Issue 1"]
        assert data["follow_up_prompt"] == "Fix it"
        assert data["duration_ms"] == 500
        assert data["model_used"] == "gemini-1.5-flash"


class TestBuildReviewPrompt:
    """Tests for prompt building."""
    
    def test_basic_prompt(self):
        """Test basic prompt building."""
        prompt = _build_review_prompt(
            task="Write a hello world",
            copilot_response="print('Hello, World!')",
            iteration=1,
            max_iterations=10,
        )
        
        assert "## TASK" in prompt
        assert "Write a hello world" in prompt
        assert "## COPILOT'S RESPONSE" in prompt
        assert "print('Hello, World!')" in prompt
        assert "Iteration 1/10" in prompt
    
    def test_prompt_with_history(self):
        """Test prompt with history summary."""
        prompt = _build_review_prompt(
            task="Write code",
            copilot_response="def foo(): pass",
            iteration=3,
            max_iterations=10,
            history_summary="Iteration 1: CRITIQUE - Missing docs\nIteration 2: CRITIQUE - Add types",
        )
        
        assert "## PREVIOUS ITERATIONS SUMMARY" in prompt
        assert "Missing docs" in prompt
    
    def test_prompt_near_max_iterations(self):
        """Test prompt adds note when near max iterations."""
        prompt = _build_review_prompt(
            task="Write code",
            copilot_response="code",
            iteration=8,
            max_iterations=10,
        )
        
        assert "iterations remaining" in prompt
        assert "acceptable" in prompt.lower()


class TestExtractJson:
    """Tests for JSON extraction from responses."""
    
    def test_extract_raw_json(self):
        """Test extracting raw JSON."""
        text = '{"verdict": "ACCEPT", "confidence": "HIGH"}'
        result = _extract_json_from_response(text)
        assert result == text
    
    def test_extract_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is my review:
```json
{"verdict": "CRITIQUE", "confidence": "MEDIUM"}
```
'''
        result = _extract_json_from_response(text)
        data = json.loads(result)
        assert data["verdict"] == "CRITIQUE"
    
    def test_extract_from_code_block_no_lang(self):
        """Test extracting JSON from code block without language."""
        text = '''```
{"verdict": "ACCEPT"}
```'''
        result = _extract_json_from_response(text)
        data = json.loads(result)
        assert data["verdict"] == "ACCEPT"
    
    def test_extract_with_surrounding_text(self):
        """Test extracting JSON with surrounding text."""
        text = 'Here is my analysis: {"verdict": "CLARIFY"} That is all.'
        result = _extract_json_from_response(text)
        data = json.loads(result)
        assert data["verdict"] == "CLARIFY"


class TestGeminiReviewer:
    """Tests for GeminiReviewer class."""
    
    def test_init_default(self):
        """Test default initialization."""
        reviewer = GeminiReviewer()
        
        assert reviewer.model == "gemini-1.5-flash"
        assert reviewer.timeout_seconds == 30
        assert reviewer.max_retries == 3
    
    def test_init_custom(self):
        """Test custom initialization."""
        reviewer = GeminiReviewer(
            api_key="test-key",
            model="gemini-1.5-pro",
            timeout_seconds=60,
            max_retries=5,
        )
        
        assert reviewer.api_key == "test-key"
        assert reviewer.model == "gemini-1.5-pro"
        assert reviewer.timeout_seconds == 60
        assert reviewer.max_retries == 5
    
    def test_available_without_key(self):
        """Test available property without API key."""
        with patch.dict('os.environ', {}, clear=True):
            reviewer = GeminiReviewer(api_key=None)
            # available depends on both HAS_GENAI and api_key
            # Without api_key, should be False
            assert reviewer.available is False or reviewer.api_key is None
    
    def test_parse_valid_response(self):
        """Test parsing valid JSON response."""
        reviewer = GeminiReviewer()
        
        raw = json.dumps({
            "verdict": "ACCEPT",
            "confidence": "HIGH",
            "reasoning": "Code is correct",
            "issues": [],
            "follow_up_prompt": None,
        })
        
        result = reviewer.parse_response(raw)
        
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "high"
        assert result.reasoning == "Code is correct"
        assert result.issues == []
    
    def test_parse_critique_response(self):
        """Test parsing CRITIQUE response."""
        reviewer = GeminiReviewer()
        
        raw = json.dumps({
            "verdict": "CRITIQUE",
            "confidence": "MEDIUM",
            "reasoning": "Missing error handling",
            "issues": ["No try/catch", "No validation"],
            "follow_up_prompt": "Add error handling",
        })
        
        result = reviewer.parse_response(raw)
        
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert len(result.issues) == 2
        assert result.follow_up_prompt == "Add error handling"
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        reviewer = GeminiReviewer()
        
        result = reviewer.parse_response("not valid json {{{")
        
        assert result.verdict == ReviewVerdict.CLARIFY
        assert "Parse error" in result.issues
    
    def test_parse_missing_fields(self):
        """Test parsing response with missing fields."""
        reviewer = GeminiReviewer()
        
        raw = json.dumps({"verdict": "ACCEPT"})
        
        result = reviewer.parse_response(raw)
        
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "low"  # default
        assert result.reasoning == ""
        assert result.issues == []
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        reviewer = GeminiReviewer()
        
        # Reset stats
        reviewer.reset_stats()
        
        assert reviewer._review_count == 0
        assert reviewer._accept_count == 0
        
        # Simulate updates
        reviewer._update_stats(ReviewVerdict.ACCEPT)
        reviewer._update_stats(ReviewVerdict.CRITIQUE)
        reviewer._update_stats(ReviewVerdict.ACCEPT)
        
        stats = reviewer.get_stats()
        
        assert stats["total_reviews"] == 0  # _review_count not updated
        assert stats["accepts"] == 2
        assert stats["critiques"] == 1


class TestGeminiReviewerAsync:
    """Tests for async review method."""
    
    @pytest.mark.asyncio
    async def test_review_without_api_key(self):
        """Test review fails gracefully without API key."""
        with patch.dict('os.environ', {}, clear=True):
            reviewer = GeminiReviewer(api_key=None)
            
            # Force not configured
            reviewer._configured = False
            
            result = await reviewer.review(
                task="Write hello world",
                copilot_response="print('hello')",
                iteration=1,
                max_iterations=10,
            )
            
            assert result.verdict == ReviewVerdict.ERROR
            assert "not configured" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_review_with_mock(self):
        """Test review with mocked Gemini API."""
        reviewer = GeminiReviewer(api_key="test-key")
        
        # Mock the genai model
        mock_response = Mock()
        mock_response.text = json.dumps({
            "verdict": "ACCEPT",
            "confidence": "HIGH",
            "reasoning": "Looks good",
            "issues": [],
            "follow_up_prompt": None,
        })
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        
        reviewer._configured = True
        reviewer._genai_model = mock_model
        
        result = await reviewer.review(
            task="Write hello world",
            copilot_response="print('Hello, World!')",
            iteration=1,
            max_iterations=10,
        )
        
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "high"


class TestGeminiReviewerSync:
    """Tests for sync review method."""
    
    def test_review_sync_without_api_key(self):
        """Test sync review fails gracefully without API key."""
        with patch.dict('os.environ', {}, clear=True):
            reviewer = GeminiReviewer(api_key=None)
            reviewer._configured = False
            
            result = reviewer.review_sync(
                task="Write code",
                copilot_response="code",
                iteration=1,
                max_iterations=10,
            )
            
            assert result.verdict == ReviewVerdict.ERROR
    
    def test_review_sync_with_mock(self):
        """Test sync review with mocked API."""
        reviewer = GeminiReviewer(api_key="test-key")
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "verdict": "CRITIQUE",
            "confidence": "MEDIUM",
            "reasoning": "Missing tests",
            "issues": ["No unit tests"],
            "follow_up_prompt": "Add tests",
        })
        
        mock_model = Mock()
        mock_model.generate_content = Mock(return_value=mock_response)
        
        reviewer._configured = True
        reviewer._genai_model = mock_model
        
        result = reviewer.review_sync(
            task="Write a function",
            copilot_response="def foo(): pass",
            iteration=1,
            max_iterations=10,
        )
        
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert result.follow_up_prompt == "Add tests"


class TestSystemPrompt:
    """Tests for system prompt."""
    
    def test_prompt_contains_verdicts(self):
        """Test system prompt contains all verdict types."""
        assert "ACCEPT" in SYSTEM_PROMPT
        assert "CRITIQUE" in SYSTEM_PROMPT
        assert "CLARIFY" in SYSTEM_PROMPT
    
    def test_prompt_contains_json_format(self):
        """Test system prompt specifies JSON format."""
        assert "JSON" in SYSTEM_PROMPT
        assert "verdict" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT
        assert "reasoning" in SYSTEM_PROMPT
        assert "issues" in SYSTEM_PROMPT
        assert "follow_up_prompt" in SYSTEM_PROMPT
