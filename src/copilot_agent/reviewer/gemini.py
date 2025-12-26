"""
Gemini reviewer integration.
"""

import json
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from copilot_agent.logging import get_logger

logger = get_logger(__name__)


class ReviewVerdict(str, Enum):
    """Review verdict types."""
    
    ACCEPT = "accept"
    CRITIQUE = "critique"
    CLARIFY = "clarify"


@dataclass
class ReviewResult:
    """Result from Gemini review."""
    
    verdict: ReviewVerdict
    confidence: str  # "high", "medium", "low"
    reasoning: str
    issues: list[str]
    follow_up_prompt: Optional[str]
    raw_response: str


SYSTEM_PROMPT = """You are a senior code reviewer acting as a quality gate in an automated coding loop. 
A coding assistant (Copilot) has produced a response. Evaluate it and decide the next step.

RULES:
1. ACCEPT only if the task is fully and correctly complete
2. CRITIQUE if specific, actionable improvements are needed
3. CLARIFY if requirements are ambiguous or you need human input
4. Be concise and specific — vague feedback wastes iterations
5. Your FOLLOW_UP_PROMPT will be sent directly to Copilot — write it as a clear instruction

RESPOND IN THIS EXACT JSON FORMAT:
{
  "verdict": "ACCEPT" | "CRITIQUE" | "CLARIFY",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "2-3 sentences explaining your verdict",
  "issues": ["issue 1", "issue 2"],
  "follow_up_prompt": "The exact prompt to send to Copilot (only if CRITIQUE)"
}

Do not include any text outside the JSON object."""


class GeminiReviewer:
    """
    Gemini API client for code review.
    
    Note: Full implementation in M4.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        logger.info(
            "GeminiReviewer initialized",
            model=model,
            timeout=timeout_seconds,
        )
    
    async def review(
        self,
        task: str,
        copilot_response: str,
        iteration: int,
        max_iterations: int,
        history_summary: Optional[str] = None,
    ) -> ReviewResult:
        """
        Review Copilot's response.
        
        Args:
            task: Original task description
            copilot_response: Copilot's response to review
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            history_summary: Summary of previous iterations
            
        Returns:
            ReviewResult with verdict and feedback
        """
        # TODO: Implement in M4
        logger.warning("Gemini review not implemented yet (M4)")
        
        # Return mock result for testing
        return ReviewResult(
            verdict=ReviewVerdict.CRITIQUE,
            confidence="low",
            reasoning="Review not implemented yet (M4)",
            issues=["Implementation pending"],
            follow_up_prompt=None,
            raw_response="{}",
        )
    
    def parse_response(self, raw_response: str) -> ReviewResult:
        """
        Parse Gemini's JSON response.
        
        Args:
            raw_response: Raw response text from Gemini
            
        Returns:
            Parsed ReviewResult
        """
        try:
            data = json.loads(raw_response)
            
            verdict_str = data.get("verdict", "CLARIFY").upper()
            verdict = ReviewVerdict(verdict_str.lower())
            
            return ReviewResult(
                verdict=verdict,
                confidence=data.get("confidence", "low").lower(),
                reasoning=data.get("reasoning", ""),
                issues=data.get("issues", []),
                follow_up_prompt=data.get("follow_up_prompt"),
                raw_response=raw_response,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse Gemini response", error=str(e))
            return ReviewResult(
                verdict=ReviewVerdict.CLARIFY,
                confidence="low",
                reasoning=f"Failed to parse response: {e}",
                issues=["Parse error"],
                follow_up_prompt=None,
                raw_response=raw_response,
            )
