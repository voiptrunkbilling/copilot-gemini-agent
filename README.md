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
- **Python:** 3.11+
- **VS Code:** With GitHub Copilot extension installed and logged in
- **Tesseract OCR:** For text detection

### Install Tesseract (Windows)

```powershell
# Option 1: Chocolatey
choco install tesseract

# Option 2: Download installer
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Install Tesseract (Linux)

```bash
sudo apt install tesseract-ocr
```

## Installation

```bash
# Clone repository
git clone https://github.com/fntelecomllc/copilot-gemini-agent.git
cd copilot-gemini-agent

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Configuration

### Environment Variables

```bash
# Required: Gemini API key
set GEMINI_API_KEY=your-api-key-here   # Windows
export GEMINI_API_KEY=your-api-key-here # Linux
```

### Config File (Optional)

Create `~/.copilot-agent/config.yaml`:

```yaml
gemini:
  model: gemini-1.5-flash
  max_retries: 3
  timeout_seconds: 30

perception:
  ocr_attempts: 2
  template_attempts: 2
  vision_enabled: true
  max_vision_per_iteration: 3
  max_vision_per_session: 20

automation:
  default_mode: approve  # approve | step | auto
  max_iterations: 20
  max_runtime_minutes: 30
  action_delay_ms: 50

storage:
  base_path: ~/.copilot-agent
  max_total_mb: 500
  retention_days: 7

safety:
  kill_switch_hotkey: ctrl+shift+k
  allowed_window_patterns:
    - "^Visual Studio Code$"
    - "^Code -"
    - ".+ - Visual Studio Code$"
```

## Usage

### Basic Run

```bash
# Start with a task (default: APPROVE_ITERATIONS mode)
python -m copilot_agent run --task "Fix the type error in handleSubmit"

# With explicit mode
python -m copilot_agent run --task "..." --mode approve  # Pause between iterations
python -m copilot_agent run --task "..." --mode step     # Pause before each action
python -m copilot_agent run --task "..." --mode auto     # No pauses (use carefully)
```

### Calibration

```bash
# Run manual calibration (recommended for first use)
python -m copilot_agent calibrate
```

### Resume Session

```bash
# Resume from checkpoint
python -m copilot_agent resume --session <session-id>
```

### Other Commands

```bash
# Show version
python -m copilot_agent --version

# View session stats
python -m copilot_agent stats --session <session-id>

# Dry run (log actions without executing)
python -m copilot_agent run --task "..." --dry-run
```

## TUI Controls

| Key | Action |
|-----|--------|
| `P` | Pause automation |
| `R` | Resume automation |
| `A` | Abort session |
| `S` | Toggle step mode |
| `V` | Force vision detection |
| `C` | Continue (in approve mode) |
| `E` | Edit next prompt |
| `D` | Show debug details |
| `Q` | Quit |

### Global Hotkeys (Work Anytime)

| Hotkey | Action |
|--------|--------|
| `Ctrl+Shift+K` | Kill switch (immediate stop) |
| `Ctrl+Shift+P` | Pause |
| `Esc` × 3 | Emergency stop |

## Project Structure

```
copilot-gemini-agent/
├── src/
│   └── copilot_agent/
│       ├── __init__.py
│       ├── __main__.py          # Entry point
│       ├── cli.py               # Click CLI
│       ├── config.py            # Pydantic config
│       ├── state.py             # State manager
│       ├── orchestrator.py      # Main control loop
│       ├── tui.py               # Rich TUI
│       ├── actuator/            # GUI actions
│       │   ├── __init__.py
│       │   ├── actions.py
│       │   └── window.py
│       ├── perception/          # Screenshot, OCR, vision
│       │   ├── __init__.py
│       │   ├── screenshot.py
│       │   ├── ocr.py
│       │   ├── template.py
│       │   └── vision.py
│       ├── reviewer/            # Gemini integration
│       │   ├── __init__.py
│       │   └── gemini.py
│       └── safety/              # Kill switch, guardrails
│           ├── __init__.py
│           └── killswitch.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── templates/                   # UI element templates
├── scripts/                     # Helper scripts
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── README.md
└── DEMO.md                      # Milestone validation guide
```

## Development

```bash
# Run tests
pytest tests/unit -v

# Run linter
ruff check src/

# Run type checker
mypy src/

# Format code
ruff format src/
```

## Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Skeleton + TUI + Kill switch | 🚧 In Progress |
| M2 | GUI Actions | ⏳ Planned |
| M3 | Perception (OCR + Templates) | ⏳ Planned |
| M4 | Full Loop (End-to-End) | ⏳ Planned |
| M5 | Production MVP | ⏳ Planned |

## Safety

- **Kill switch:** `Ctrl+Shift+K` stops automation immediately
- **Window allowlist:** Only interacts with VS Code windows
- **Action validation:** All clicks/keystrokes validated before execution
- **Iteration limits:** Default max 20 iterations per session
- **Audit logging:** All actions logged to `audit.jsonl`

## License

MIT
