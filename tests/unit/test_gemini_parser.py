"""
Unit tests for Gemini response parsing.
"""

import pytest

from copilot_agent.reviewer.gemini import (
    GeminiReviewer,
    ReviewResult,
    ReviewVerdict,
)


class TestGeminiResponseParsing:
    """Tests for Gemini response parsing."""
    
    @pytest.fixture
    def reviewer(self):
        return GeminiReviewer(api_key="test")
    
    def test_parse_accept(self, reviewer):
        response = '''
        {
            "verdict": "ACCEPT",
            "confidence": "HIGH",
            "reasoning": "The code looks good and handles all edge cases.",
            "issues": [],
            "follow_up_prompt": null
        }
        '''
        
        result = reviewer.parse_response(response)
        
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "high"
        assert result.reasoning == "The code looks good and handles all edge cases."
        assert result.issues == []
        assert result.follow_up_prompt is None
    
    def test_parse_critique(self, reviewer):
        response = '''
        {
            "verdict": "CRITIQUE",
            "confidence": "MEDIUM",
            "reasoning": "The code has a bug in error handling.",
            "issues": ["Missing try-catch", "No input validation"],
            "follow_up_prompt": "Please add error handling for invalid inputs."
        }
        '''
        
        result = reviewer.parse_response(response)
        
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert result.confidence == "medium"
        assert len(result.issues) == 2
        assert result.follow_up_prompt == "Please add error handling for invalid inputs."
    
    def test_parse_clarify(self, reviewer):
        response = '''
        {
            "verdict": "CLARIFY",
            "confidence": "LOW",
            "reasoning": "I need more information about the expected behavior.",
            "issues": ["Unclear requirements"],
            "follow_up_prompt": null
        }
        '''
        
        result = reviewer.parse_response(response)
        
        assert result.verdict == ReviewVerdict.CLARIFY
        assert result.confidence == "low"
    
    def test_parse_invalid_json(self, reviewer):
        response = "This is not JSON"
        
        result = reviewer.parse_response(response)
        
        assert result.verdict == ReviewVerdict.CLARIFY
        assert "parse" in result.reasoning.lower()
    
    def test_parse_missing_fields(self, reviewer):
        response = '{"verdict": "ACCEPT"}'
        
        result = reviewer.parse_response(response)
        
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "low"  # Default
        assert result.reasoning == ""  # Default
    
    def test_parse_lowercase_verdict(self, reviewer):
        response = '''
        {
            "verdict": "accept",
            "confidence": "high",
            "reasoning": "OK",
            "issues": [],
            "follow_up_prompt": null
        }
        '''
        
        result = reviewer.parse_response(response)
        assert result.verdict == ReviewVerdict.ACCEPT
