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

## Usage

```bash
# Start with a task
python -m copilot_agent run --task "Fix the type error in handleSubmit"

# Run calibration
python -m copilot_agent calibrate

# Dry-run mode
python -m copilot_agent run --task "..." --dry-run
```

## Safety

- **Kill switch:** `Ctrl+Shift+K` stops automation immediately
- **Window allowlist:** Only interacts with VS Code windows

## License

MIT
