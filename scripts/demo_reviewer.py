#!/usr/bin/env python3
"""
M4 Reviewer Loop Demo Script

Validates the complete review loop on Windows:
1. GeminiReviewer initialization and API connection
2. Review prompt generation and parsing
3. Orchestrator iteration control
4. TUI verdict display
5. End-to-end loop simulation

Usage:
    python scripts/demo_reviewer.py              # Run all tests
    python scripts/demo_reviewer.py --api-test   # Test real API (requires GEMINI_API_KEY)
    python scripts/demo_reviewer.py --mock       # Mock-only tests
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from copilot_agent.logging import setup_logging
from copilot_agent.config import AgentConfig
from copilot_agent.state import StateManager, SessionPhase, GeminiVerdict
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.reviewer.gemini import (
    GeminiReviewer,
    ReviewResult,
    ReviewVerdict,
    _build_review_prompt,
    _extract_json_from_response,
)
from copilot_agent.orchestrator import Orchestrator, IterationResult

# Setup logging
setup_logging(level="INFO")


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, message: str = "") -> None:
    """Print a test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status}: {name}")
    if message:
        print(f"         {message}")


def test_prompt_building() -> bool:
    """Test review prompt construction."""
    print_header("Testing Prompt Building")
    
    try:
        # Basic prompt
        prompt = _build_review_prompt(
            task="Write a Python function that calculates factorial",
            copilot_response="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            iteration=1,
            max_iterations=10,
        )
        
        assert "## TASK" in prompt
        assert "factorial" in prompt
        assert "Iteration 1/10" in prompt
        print_result("Basic prompt construction", True)
        
        # Prompt with history
        prompt_with_history = _build_review_prompt(
            task="Write code",
            copilot_response="code",
            iteration=3,
            max_iterations=10,
            history_summary="Iteration 1: CRITIQUE\nIteration 2: CRITIQUE",
        )
        
        assert "PREVIOUS ITERATIONS" in prompt_with_history
        print_result("Prompt with history", True)
        
        # Prompt near max iterations
        prompt_near_end = _build_review_prompt(
            task="Task",
            copilot_response="Response",
            iteration=9,
            max_iterations=10,
        )
        
        assert "remaining" in prompt_near_end.lower()
        print_result("Prompt near max iterations warning", True)
        
        return True
        
    except Exception as e:
        print_result("Prompt building", False, str(e))
        return False


def test_json_extraction() -> bool:
    """Test JSON extraction from various response formats."""
    print_header("Testing JSON Extraction")
    
    try:
        # Raw JSON
        raw = '{"verdict": "ACCEPT", "confidence": "HIGH"}'
        extracted = _extract_json_from_response(raw)
        data = json.loads(extracted)
        assert data["verdict"] == "ACCEPT"
        print_result("Raw JSON extraction", True)
        
        # JSON in code block
        code_block = '''Here is my review:
```json
{"verdict": "CRITIQUE", "reasoning": "Needs tests"}
```
'''
        extracted = _extract_json_from_response(code_block)
        data = json.loads(extracted)
        assert data["verdict"] == "CRITIQUE"
        print_result("Code block extraction", True)
        
        # JSON with surrounding text
        mixed = 'Analysis complete. {"verdict": "CLARIFY"} End.'
        extracted = _extract_json_from_response(mixed)
        data = json.loads(extracted)
        assert data["verdict"] == "CLARIFY"
        print_result("Mixed text extraction", True)
        
        return True
        
    except Exception as e:
        print_result("JSON extraction", False, str(e))
        return False


def test_reviewer_init() -> bool:
    """Test GeminiReviewer initialization."""
    print_header("Testing Reviewer Initialization")
    
    try:
        # Default init
        reviewer = GeminiReviewer()
        assert reviewer.model == "gemma-3-27b-it"
        assert reviewer.timeout_seconds == 30
        print_result("Default initialization", True)
        
        # Custom init
        reviewer = GeminiReviewer(
            model="gemini-2.5-pro",
            timeout_seconds=60,
            max_retries=5,
        )
        assert reviewer.model == "gemini-2.5-pro"
        assert reviewer.max_retries == 5
        print_result("Custom initialization", True)
        
        # Check availability
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        print_result(
            "API key check",
            True,
            f"Available: {has_key}"
        )
        
        return True
        
    except Exception as e:
        print_result("Reviewer init", False, str(e))
        return False


def test_response_parsing() -> bool:
    """Test parsing of Gemini responses."""
    print_header("Testing Response Parsing")
    
    reviewer = GeminiReviewer()
    
    try:
        # Valid ACCEPT response
        accept_json = json.dumps({
            "verdict": "ACCEPT",
            "confidence": "HIGH",
            "reasoning": "Code is correct and complete",
            "issues": [],
            "follow_up_prompt": None,
        })
        
        result = reviewer.parse_response(accept_json)
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "high"
        print_result("Parse ACCEPT response", True)
        
        # Valid CRITIQUE response
        critique_json = json.dumps({
            "verdict": "CRITIQUE",
            "confidence": "MEDIUM",
            "reasoning": "Missing error handling",
            "issues": ["No try/catch", "No input validation"],
            "follow_up_prompt": "Add error handling for edge cases",
        })
        
        result = reviewer.parse_response(critique_json)
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert len(result.issues) == 2
        assert result.follow_up_prompt is not None
        print_result("Parse CRITIQUE response", True)
        
        # Invalid JSON
        result = reviewer.parse_response("not valid json {{{")
        assert result.verdict == ReviewVerdict.CLARIFY
        assert "Parse error" in result.issues
        print_result("Handle invalid JSON", True)
        
        # Missing fields
        result = reviewer.parse_response('{"verdict": "ACCEPT"}')
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.confidence == "low"  # default
        print_result("Handle missing fields", True)
        
        return True
        
    except Exception as e:
        print_result("Response parsing", False, str(e))
        return False


def test_reviewer_stats() -> bool:
    """Test reviewer statistics tracking."""
    print_header("Testing Statistics Tracking")
    
    try:
        reviewer = GeminiReviewer()
        reviewer.reset_stats()
        
        # Simulate verdicts
        reviewer._update_stats(ReviewVerdict.ACCEPT)
        reviewer._update_stats(ReviewVerdict.CRITIQUE)
        reviewer._update_stats(ReviewVerdict.CRITIQUE)
        reviewer._update_stats(ReviewVerdict.ACCEPT)
        
        stats = reviewer.get_stats()
        
        assert stats["accepts"] == 2
        assert stats["critiques"] == 2
        print_result("Stats tracking", True, f"Accepts: {stats['accepts']}, Critiques: {stats['critiques']}")
        
        return True
        
    except Exception as e:
        print_result("Stats tracking", False, str(e))
        return False


def test_orchestrator_setup() -> bool:
    """Test orchestrator initialization."""
    print_header("Testing Orchestrator Setup")
    
    try:
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        # Create orchestrator
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
        
        assert orchestrator.dry_run is True
        assert orchestrator.reviewer is not None
        print_result("Orchestrator creation", True)
        
        # Create session
        state_manager.create_session("Test task: write hello world")
        assert state_manager.session is not None
        print_result("Session creation", True)
        
        # Test control methods
        orchestrator.stop()
        assert orchestrator._should_stop is True
        print_result("Stop control", True)
        
        orchestrator.set_next_prompt("Custom prompt")
        assert state_manager.session.next_prompt == "Custom prompt"
        print_result("Custom prompt override", True)
        
        return True
        
    except Exception as e:
        print_result("Orchestrator setup", False, str(e))
        return False


def test_repeated_critique_detection() -> bool:
    """Test detection of repeated critiques."""
    print_header("Testing Repeated Critique Detection")
    
    try:
        config = AgentConfig()
        config.reviewer.stop_on_repeated_critiques = 3
        
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
        
        # Different critiques should not trigger
        result = orchestrator._check_repeated_critiques("Add tests")
        assert result is False
        result = orchestrator._check_repeated_critiques("Add docs")
        assert result is False
        print_result("Different critiques allowed", True)
        
        # Reset and test repeated
        orchestrator._repeated_critiques = []
        
        result = orchestrator._check_repeated_critiques("Fix the bug")
        assert result is False
        result = orchestrator._check_repeated_critiques("Fix the bug")
        assert result is False
        result = orchestrator._check_repeated_critiques("Fix the bug")
        assert result is True
        print_result("Repeated critiques detected", True)
        
        return True
        
    except Exception as e:
        print_result("Repeated critique detection", False, str(e))
        return False


async def test_mock_iteration() -> bool:
    """Test a mock iteration of the review loop."""
    print_header("Testing Mock Iteration")
    
    try:
        from unittest.mock import Mock, AsyncMock
        
        config = AgentConfig()
        config.reviewer.response_wait_seconds = 0
        config.reviewer.response_stability_ms = 0
        config.reviewer.pause_before_send = False
        
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        state_manager.create_session("Write a hello world function")
        
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
        
        # Mock perception
        mock_capture = Mock()
        mock_capture.success = True
        mock_capture.text = "def hello():\n    print('Hello, World!')"
        mock_capture.method = Mock()
        mock_capture.method.value = "ocr"
        mock_capture.confidence = 0.9
        
        mock_perception = Mock()
        mock_perception.capture_copilot_response = Mock(return_value=mock_capture)
        orchestrator._perception = mock_perception
        
        # Mock reviewer with ACCEPT
        mock_review = ReviewResult(
            verdict=ReviewVerdict.ACCEPT,
            confidence="high",
            reasoning="Code is correct and complete",
            issues=[],
            follow_up_prompt=None,
        )
        orchestrator.reviewer.review = AsyncMock(return_value=mock_review)
        
        # Run iteration
        result = await orchestrator._run_iteration()
        
        assert result.success is True
        assert result.verdict == ReviewVerdict.ACCEPT
        assert result.should_continue is False
        print_result("Mock ACCEPT iteration", True)
        
        # Test CRITIQUE iteration
        state_manager.create_session("Write better code")
        
        mock_review_critique = ReviewResult(
            verdict=ReviewVerdict.CRITIQUE,
            confidence="medium",
            reasoning="Missing docstring",
            issues=["No docstring"],
            follow_up_prompt="Add a docstring to the function",
        )
        orchestrator.reviewer.review = AsyncMock(return_value=mock_review_critique)
        
        result = await orchestrator._run_iteration()
        
        assert result.success is True
        assert result.verdict == ReviewVerdict.CRITIQUE
        assert result.should_continue is True
        assert result.feedback == "Add a docstring to the function"
        print_result("Mock CRITIQUE iteration", True)
        
        return True
        
    except Exception as e:
        print_result("Mock iteration", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_api_review(skip: bool = False) -> bool:
    """Test real API review (requires GEMINI_API_KEY)."""
    print_header("Testing Real API Review")
    
    if skip:
        print("  ⏭️  Skipped (use --api-test to enable)")
        return True
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set - skipping API test")
        return True
    
    try:
        reviewer = GeminiReviewer(api_key=api_key)
        
        print("  Sending review request to Gemini...")
        start = time.time()
        
        result = await reviewer.review(
            task="Write a Python function that returns the factorial of a number",
            copilot_response='''def factorial(n):
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
            iteration=1,
            max_iterations=10,
        )
        
        elapsed = time.time() - start
        
        print(f"\n  Response received in {elapsed:.2f}s")
        print(f"  Verdict: {result.verdict.value.upper()}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning[:100]}...")
        
        if result.issues:
            print(f"  Issues: {result.issues}")
        
        if result.follow_up_prompt:
            print(f"  Follow-up: {result.follow_up_prompt[:100]}...")
        
        assert result.verdict in [ReviewVerdict.ACCEPT, ReviewVerdict.CRITIQUE, ReviewVerdict.CLARIFY]
        print_result("Real API review", True, f"Verdict: {result.verdict.value}")
        
        return True
        
    except Exception as e:
        print_result("Real API review", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_tui_display() -> bool:
    """Test TUI display components."""
    print_header("Testing TUI Display")
    
    try:
        from copilot_agent.tui import AgentTUI
        
        config = AgentConfig()
        state_manager = StateManager(config)
        kill_switch = KillSwitch()
        
        tui = AgentTUI(
            state_manager=state_manager,
            kill_switch=kill_switch,
            dry_run=True,
        )
        
        # Test without session
        layout = tui.make_layout()
        assert layout is not None
        print_result("TUI layout without session", True)
        
        # Create session and test with verdict
        state_manager.create_session("Test task")
        session = state_manager.session
        session.current_prompt = "Write hello world"
        session.current_response = "print('Hello')"
        session.current_verdict = GeminiVerdict.CRITIQUE
        session.current_feedback = "Add error handling"
        
        layout = tui.make_layout()
        assert layout is not None
        
        # Test verdict formatting
        verdict_str = tui._format_verdict(GeminiVerdict.ACCEPT)
        assert "ACCEPT" in verdict_str
        assert "green" in verdict_str
        print_result("TUI verdict display", True)
        
        verdict_str = tui._format_verdict(GeminiVerdict.CRITIQUE)
        assert "CRITIQUE" in verdict_str
        assert "yellow" in verdict_str
        print_result("TUI critique display", True)
        
        return True
        
    except Exception as e:
        print_result("TUI display", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def run_demo(api_test: bool = False) -> bool:
    """Run all demo tests."""
    print("\n" + "=" * 60)
    print("  M4 REVIEWER LOOP DEMO - Windows Validation")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Prompt Building", test_prompt_building()))
    results.append(("JSON Extraction", test_json_extraction()))
    results.append(("Reviewer Init", test_reviewer_init()))
    results.append(("Response Parsing", test_response_parsing()))
    results.append(("Stats Tracking", test_reviewer_stats()))
    results.append(("Orchestrator Setup", test_orchestrator_setup()))
    results.append(("Repeated Critique Detection", test_repeated_critique_detection()))
    results.append(("Mock Iteration", await test_mock_iteration()))
    results.append(("TUI Display", test_tui_display()))
    results.append(("Real API Review", await test_api_review(skip=not api_test)))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 All M4 tests passed!")
        return True
    else:
        print("\n  ⚠️  Some tests failed")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="M4 Reviewer Loop Demo")
    parser.add_argument("--api-test", action="store_true", help="Run real API tests")
    parser.add_argument("--mock", action="store_true", help="Run mock-only tests")
    args = parser.parse_args()
    
    success = asyncio.run(run_demo(api_test=args.api_test))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
