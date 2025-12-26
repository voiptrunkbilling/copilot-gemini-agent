"""
Gemini reviewer integration.

M4: Full implementation of text reviewer with structured JSON output.
"""

import asyncio
import json
import os
import re
import time
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

# Import google-generativeai
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-generativeai not installed")


class ReviewVerdict(str, Enum):
    """Review verdict types."""
    
    ACCEPT = "accept"
    CRITIQUE = "critique"
    CLARIFY = "clarify"
    ERROR = "error"


@dataclass
class ReviewResult:
    """Result from Gemini review."""
    
    verdict: ReviewVerdict
    confidence: str  # "high", "medium", "low"
    reasoning: str
    issues: List[str] = field(default_factory=list)
    follow_up_prompt: Optional[str] = None
    raw_response: str = ""
    duration_ms: int = 0
    model_used: str = ""
    tokens_used: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
            "follow_up_prompt": self.follow_up_prompt,
            "duration_ms": self.duration_ms,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
        }


# Strict reviewer prompt designed for consistent JSON output
SYSTEM_PROMPT = """You are a senior code reviewer acting as a quality gate in an automated Copilot-Gemini feedback loop.

CONTEXT:
- A human user gave Copilot a coding task
- Copilot has produced a response
- Your job: evaluate the response and decide the next step

VERDICTS:
• ACCEPT — The response fully and correctly completes the task. No further iteration needed.
• CRITIQUE — The response has specific, actionable issues. You will provide feedback to be sent back to Copilot.
• CLARIFY — Requirements are ambiguous or you need human input. The loop will pause for user intervention.

RULES:
1. ACCEPT only when the task is demonstrably complete and correct
2. CRITIQUE with specific, actionable feedback — vague feedback wastes iterations
3. CLARIFY only when you cannot proceed without human guidance
4. Your `follow_up_prompt` will be sent directly to Copilot — write it as a clear instruction
5. Be concise but thorough in your reasoning
6. Consider edge cases, error handling, and best practices
7. If near max iterations, lean toward ACCEPT if the solution is acceptable

OUTPUT FORMAT (strict JSON, no markdown, no extra text):
{
  "verdict": "ACCEPT" | "CRITIQUE" | "CLARIFY",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "2-3 sentences explaining your verdict",
  "issues": ["issue 1", "issue 2"],
  "follow_up_prompt": "The exact prompt to send to Copilot (required if CRITIQUE, null otherwise)"
}"""


def _build_review_prompt(
    task: str,
    copilot_response: str,
    iteration: int,
    max_iterations: int,
    history_summary: Optional[str] = None,
) -> str:
    """Build the review prompt with context."""
    prompt_parts = [
        f"## TASK (Given to Copilot)\n{task}",
        f"\n## COPILOT'S RESPONSE (Iteration {iteration}/{max_iterations})\n{copilot_response}",
    ]
    
    if history_summary:
        prompt_parts.insert(1, f"\n## PREVIOUS ITERATIONS SUMMARY\n{history_summary}")
    
    if iteration >= max_iterations - 2:
        prompt_parts.append(
            f"\n## NOTE: Only {max_iterations - iteration} iterations remaining. "
            "Consider accepting if the solution is acceptable."
        )
    
    prompt_parts.append("\n## YOUR REVIEW (JSON only):")
    
    return "\n".join(prompt_parts)


def _extract_json_from_response(text: str) -> str:
    """Extract JSON from response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        return code_block_match.group(1)
    
    # Try to find raw JSON
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json_match.group(0)
    
    return text


class GeminiReviewer:
    """
    Gemini API client for code review.
    
    Provides structured review of Copilot responses with ACCEPT/CRITIQUE/CLARIFY verdicts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the Gemini reviewer.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model ID to use
            timeout_seconds: Request timeout
            max_retries: Max retries on failure
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._configured = False
        self._genai_model = None
        
        # Statistics
        self._review_count = 0
        self._accept_count = 0
        self._critique_count = 0
        self._clarify_count = 0
        self._error_count = 0
        
        logger.info(
            "GeminiReviewer initialized",
            model=model,
            timeout=timeout_seconds,
            has_api_key=bool(self.api_key),
        )
    
    @property
    def available(self) -> bool:
        """Check if reviewer is available."""
        return HAS_GENAI and bool(self.api_key)
    
    def _ensure_configured(self) -> bool:
        """Ensure Gemini is configured."""
        if self._configured:
            return True
        
        if not HAS_GENAI:
            logger.error("google-generativeai not installed")
            return False
        
        if not self.api_key:
            logger.error("GEMINI_API_KEY not set")
            return False
        
        try:
            genai.configure(api_key=self.api_key)
            self._genai_model = genai.GenerativeModel(
                self.model,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent JSON
                    top_p=0.95,
                    max_output_tokens=1024,
                ),
            )
            self._configured = True
            logger.info("Gemini configured", model=self.model)
            return True
        except Exception as e:
            logger.error("Failed to configure Gemini", error=str(e))
            return False
    
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
        start = time.time()
        self._review_count += 1
        
        if not self._ensure_configured():
            self._error_count += 1
            return ReviewResult(
                verdict=ReviewVerdict.ERROR,
                confidence="low",
                reasoning="Gemini API not configured",
                issues=["API not available"],
                raw_response="",
            )
        
        # Build prompt
        prompt = _build_review_prompt(
            task=task,
            copilot_response=copilot_response,
            iteration=iteration,
            max_iterations=max_iterations,
            history_summary=history_summary,
        )
        
        # Call Gemini API with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._genai_model.generate_content(prompt),
                    ),
                    timeout=self.timeout_seconds,
                )
                
                raw_text = response.text
                duration_ms = int((time.time() - start) * 1000)
                
                # Parse response
                result = self.parse_response(raw_text)
                result.duration_ms = duration_ms
                result.model_used = self.model
                
                # Try to get token count
                if hasattr(response, 'usage_metadata'):
                    result.tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
                
                # Update statistics
                self._update_stats(result.verdict)
                
                logger.info(
                    "Review complete",
                    verdict=result.verdict.value,
                    confidence=result.confidence,
                    duration_ms=duration_ms,
                    iteration=iteration,
                )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout_seconds}s"
                logger.warning(f"Review timeout, attempt {attempt + 1}/{self.max_retries}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Review failed, attempt {attempt + 1}/{self.max_retries}", error=last_error)
            
            # Brief delay before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(1.0)
        
        # All retries failed
        self._error_count += 1
        duration_ms = int((time.time() - start) * 1000)
        
        return ReviewResult(
            verdict=ReviewVerdict.ERROR,
            confidence="low",
            reasoning=f"Review failed after {self.max_retries} attempts: {last_error}",
            issues=["API error"],
            duration_ms=duration_ms,
            model_used=self.model,
        )
    
    def review_sync(
        self,
        task: str,
        copilot_response: str,
        iteration: int,
        max_iterations: int,
        history_summary: Optional[str] = None,
    ) -> ReviewResult:
        """
        Synchronous version of review for non-async contexts.
        
        Args:
            task: Original task description
            copilot_response: Copilot's response to review
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            history_summary: Summary of previous iterations
            
        Returns:
            ReviewResult with verdict and feedback
        """
        start = time.time()
        self._review_count += 1
        
        if not self._ensure_configured():
            self._error_count += 1
            return ReviewResult(
                verdict=ReviewVerdict.ERROR,
                confidence="low",
                reasoning="Gemini API not configured",
                issues=["API not available"],
                raw_response="",
            )
        
        # Build prompt
        prompt = _build_review_prompt(
            task=task,
            copilot_response=copilot_response,
            iteration=iteration,
            max_iterations=max_iterations,
            history_summary=history_summary,
        )
        
        # Call Gemini API with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._genai_model.generate_content(prompt)
                
                raw_text = response.text
                duration_ms = int((time.time() - start) * 1000)
                
                # Parse response
                result = self.parse_response(raw_text)
                result.duration_ms = duration_ms
                result.model_used = self.model
                
                # Try to get token count
                if hasattr(response, 'usage_metadata'):
                    result.tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
                
                # Update statistics
                self._update_stats(result.verdict)
                
                logger.info(
                    "Review complete",
                    verdict=result.verdict.value,
                    confidence=result.confidence,
                    duration_ms=duration_ms,
                    iteration=iteration,
                )
                
                return result
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Review failed, attempt {attempt + 1}/{self.max_retries}", error=last_error)
            
            # Brief delay before retry
            if attempt < self.max_retries - 1:
                time.sleep(1.0)
        
        # All retries failed
        self._error_count += 1
        duration_ms = int((time.time() - start) * 1000)
        
        return ReviewResult(
            verdict=ReviewVerdict.ERROR,
            confidence="low",
            reasoning=f"Review failed after {self.max_retries} attempts: {last_error}",
            issues=["API error"],
            duration_ms=duration_ms,
            model_used=self.model,
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
            # Extract JSON from response
            json_str = _extract_json_from_response(raw_response)
            data = json.loads(json_str)
            
            verdict_str = data.get("verdict", "CLARIFY").upper()
            
            # Map verdict string to enum
            verdict_map = {
                "ACCEPT": ReviewVerdict.ACCEPT,
                "CRITIQUE": ReviewVerdict.CRITIQUE,
                "CLARIFY": ReviewVerdict.CLARIFY,
            }
            verdict = verdict_map.get(verdict_str, ReviewVerdict.CLARIFY)
            
            return ReviewResult(
                verdict=verdict,
                confidence=data.get("confidence", "low").lower(),
                reasoning=data.get("reasoning", ""),
                issues=data.get("issues", []),
                follow_up_prompt=data.get("follow_up_prompt"),
                raw_response=raw_response,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse Gemini response", error=str(e), response=raw_response[:500])
            return ReviewResult(
                verdict=ReviewVerdict.CLARIFY,
                confidence="low",
                reasoning=f"Failed to parse response: {e}",
                issues=["Parse error"],
                follow_up_prompt=None,
                raw_response=raw_response,
            )
    
    def _update_stats(self, verdict: ReviewVerdict) -> None:
        """Update verdict statistics."""
        if verdict == ReviewVerdict.ACCEPT:
            self._accept_count += 1
        elif verdict == ReviewVerdict.CRITIQUE:
            self._critique_count += 1
        elif verdict == ReviewVerdict.CLARIFY:
            self._clarify_count += 1
        else:
            self._error_count += 1
    
    def get_stats(self) -> dict:
        """Get reviewer statistics."""
        return {
            "total_reviews": self._review_count,
            "accepts": self._accept_count,
            "critiques": self._critique_count,
            "clarifies": self._clarify_count,
            "errors": self._error_count,
            "accept_rate": self._accept_count / self._review_count if self._review_count > 0 else 0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._review_count = 0
        self._accept_count = 0
        self._critique_count = 0
        self._clarify_count = 0
        self._error_count = 0
