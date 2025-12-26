# Copilot-Gemini Agent

Local desktop automation agent that orchestrates iterative coding loops between GitHub Copilot Chat (VS Code) and Gemini (reviewer).

## Overview

This agent automates the human role in a Copilot ↔ LLM review loop:

1. Types prompts into Copilot Chat (GUI automation)
2. Captures Copilot's responses
3. Sends them to Gemini for review
4. Feeds Gemini's critique back into Copilot
5. Iterates until task is complete or limits reached

## Requirements

- **OS:** Windows 10/11 (primary), Linux (secondary)
- **Python:** 3.10+
- **VS Code:** With GitHub Copilot extension installed and logged in

## Installation

```bash
# Clone repository
git clone https://github.com/voiptrunkbilling/copilot-gemini-agent.git
cd copilot-gemini-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Configuration

Create a `config.yaml` file or use environment variables:

```yaml
# Reviewer settings
reviewer:
  max_iterations: 10        # Maximum review cycles (default: 10)
  pause_before_send: true   # Pause for confirmation before API calls
  stop_on_repeated_critiques: 3  # Stop after N identical critiques
  response_wait_seconds: 30 # Timeout for Gemini responses

# Actuator settings
actuator:
  typing_speed: 0.05       # Seconds between keystrokes
  target_window_pattern: "Visual Studio Code"
```

### Environment Variables

```bash
export GEMINI_API_KEY="your-api-key"  # Required for live review
```

## Usage

```bash
# Start with a task
python -m copilot_agent run --task "Fix the type error in handleSubmit"

# Run calibration
python -m copilot_agent calibrate

# Dry-run mode (no Gemini API calls)
python -m copilot_agent run --task "..." --dry-run
```

### Demo Scripts

Validate the reviewer integration:

```bash
# Run mock tests (no API key needed)
python scripts/demo_reviewer.py --mock

# Test live Gemini API (requires GEMINI_API_KEY)
python scripts/demo_reviewer.py --api-test

# Full demo with all tests
python scripts/demo_reviewer.py
```

Expected output for `--mock`:
```
[PASS] Mock initialization
[PASS] Mock review (good response)
...
✓ 10/10 tests passed
```

## Review Loop

The agent runs an iterative review loop:

```
┌─────────────────────────────────────────────────────────┐
│  1. Prompt → Copilot                                    │
│  2. Wait for response                                   │
│  3. Capture screenshot + OCR                            │
│  4. Send to Gemini reviewer                             │
│  5. Parse verdict:                                      │
│     • ACCEPT → Done!                                    │
│     • CRITIQUE → Feed back to Copilot, goto 1          │
│     • CLARIFY → Request more info, goto 1              │
│  6. Repeat until max_iterations or acceptance          │
└─────────────────────────────────────────────────────────┘
```

## Safety

- **Kill switch:** `Ctrl+Shift+K` stops automation immediately
- **Window allowlist:** Only interacts with VS Code windows
- **Pause mode:** `pause_before_send: true` requires confirmation before API calls
- **Iteration limits:** Hard cap on review cycles prevents runaway loops
- **Secret redaction:** API keys are scrubbed from all logs

## License

MIT
