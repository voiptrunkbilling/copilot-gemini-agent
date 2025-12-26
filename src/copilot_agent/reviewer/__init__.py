"""
Reviewer module - Gemini API integration for code review.

M4: Full implementation with structured JSON verdicts.
"""

from copilot_agent.reviewer.gemini import (
    GeminiReviewer,
    ReviewResult,
    ReviewVerdict,
    SYSTEM_PROMPT,
)

__all__ = [
    "GeminiReviewer",
    "ReviewResult",
    "ReviewVerdict",
    "SYSTEM_PROMPT",
]
