# Copilot-Gemini Agent

> Local desktop automation agent that orchestrates iterative coding loops between GitHub Copilot Chat (VS Code) and Gemini (reviewer).

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This agent automates the human role in a Copilot ↔ LLM review loop:

1. **Types prompts** into Copilot Chat (GUI automation)
2. **Captures responses** via screenshot + OCR
3. **Sends to Gemini** for expert code review
4. **Feeds critique back** into Copilot for iteration
5. **Repeats** until acceptance or limits reached

```
┌─────────────────────────────────────────────────────────┐
│  User Task: "Fix the type error in handleSubmit"       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│  │  Copilot │────►│  Agent   │────►│  Gemini  │        │
│  │   Chat   │◄────│(Actuator)│◄────│(Reviewer)│        │
│  └──────────┘     └──────────┘     └──────────┘        │
│                          │                              │
│                    ┌─────┴─────┐                       │
│                    │ ACCEPT ✓  │                       │
│                    │ CRITIQUE  │──► Iterate            │
│                    │ CLARIFY   │──► Request info       │
│                    └───────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Windows Installation (Recommended)

```powershell
# Clone the repository
git clone https://github.com/voiptrunkbilling/copilot-gemini-agent.git
cd copilot-gemini-agent

# Run the installer (creates venv, installs deps, Tesseract)
.\scripts\install.bat
```

### Manual Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .

# Install Tesseract OCR (Windows)
winget install --id UB-Mannheim.TesseractOCR
```

### Configure API Key

```bash
# Copy the example .env file
copy .env.example .env

# Edit .env and add your Gemini API key
# Get one from: https://aistudio.google.com/apikey
```

### First Run

```bash
# 1. Calibrate screen coordinates (one-time setup)
agent calibrate

# 2. Run with a task
agent run -t "Add error handling to the fetchData function"
```

## Requirements

| Component | Requirement |
|-----------|------------|
| **OS** | Windows 10/11 (primary), Linux (experimental) |
| **Python** | 3.10 or later |
| **VS Code** | With GitHub Copilot Chat extension |
| **Tesseract** | OCR engine for text capture |
| **API Key** | Gemini API key (free tier available) |

## Usage

### Start a Session

```bash
# Basic run with task
agent run --task "Implement the login form validation"

# With options
agent run \
  --task "Fix TypeScript errors in auth.ts" \
  --mode approve \
  --max-iterations 10

# Dry-run (no actual GUI actions)
agent run --task "Test task" --dry-run
```

### Commands Reference

| Command | Description |
|---------|-------------|
| `agent run` | Start a new automation session |
| `agent calibrate` | Calibrate screen coordinates |
| `agent resume` | Resume from checkpoint |
| `agent stats` | View session statistics |
| `agent config` | Manage configuration |
| `agent test-actions` | Test GUI primitives |

### Options for `run`

| Option | Description | Default |
|--------|-------------|---------|
| `--task, -t` | Task description | Required |
| `--mode` | Mode: `approve`, `step`, `auto` | `approve` |
| `--max-iterations, -n` | Max iterations | 20 |
| `--dry-run` | Log actions without executing | Off |
| `--no-vision` | Disable Gemini Vision | Off |
| `--config, -c` | Path to config file | Default |

## Configuration

### Config File

Create `~/.copilot-agent/config.yaml`:

```yaml
# Reviewer settings
reviewer:
  model: "gemma-3-27b-it"
  max_iterations: 10
  pause_before_send: true
  stop_on_repeated_critiques: 3

# Automation settings
automation:
  default_mode: "approve"
  max_iterations: 20
  max_runtime_minutes: 30

# Safety settings
safety:
  kill_switch_hotkey: "ctrl+shift+k"
  pause_hotkey: "ctrl+shift+p"
```

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your-api-key-here

# Optional overrides
COPILOT_MODEL=gemma-3-27b-it
COPILOT_MAX_ITERATIONS=20
COPILOT_STORAGE_PATH=~/.copilot-agent
COPILOT_KILLSWITCH_KEY=ctrl+shift+k
```

### Validate Configuration

```bash
# Check if config is valid
agent config --validate

# View current settings
agent config --show

# Check secrets status
agent config --secrets
```

## Safety Features

### Kill Switch

Press **`Ctrl+Shift+K`** at any time to immediately stop all automation.

### Budget Controls

The agent enforces hard limits:
- **Reviewer calls:** 50 per session (configurable)
- **Vision calls:** 20 per session
- **UI actions:** 400 per session
- **Runtime:** 30 minutes (configurable)

### Circuit Breakers

Automatic protection against cascading failures:
- Reviewer circuit opens after 3 consecutive failures
- Vision circuit opens after 2 consecutive failures
- UI circuit opens after 2 consecutive failures

### Checkpoint & Resume

Sessions are automatically checkpointed every iteration:

```bash
# List resumable sessions
agent resume --list

# Resume most recent session
agent resume

# Resume specific session
agent resume --session abc123
```

### Window Safety

Only interacts with allowed windows (VS Code by default).

## Development

### Run Tests

```bash
# All unit tests
pytest tests/unit/ -q

# Specific test file
pytest tests/unit/test_orchestrator.py -v
```

### Validation Scripts

```bash
# Full M6 validation (Windows)
.\scripts\validate_m6.ps1

# Quick validation
.\scripts\validate_m6.ps1 -QuickMode
```

### Project Structure

```
copilot-gemini-agent/
├── src/copilot_agent/
│   ├── orchestrator.py    # Main control loop
│   ├── cli.py             # CLI interface
│   ├── config.py          # Configuration
│   ├── actuator/          # GUI automation
│   ├── perception/        # OCR, vision, templates
│   ├── reviewer/          # Gemini integration
│   └── safety/            # Kill switch, checkpoints
├── scripts/               # Installation & validation
└── tests/                 # Unit & integration tests
```

## Troubleshooting

### "Kill switch triggered unexpectedly"

Check if your keyboard shortcut (Ctrl+Shift+K) conflicts with other software.

### "Calibration coordinates wrong"

Re-run calibration at your current DPI/resolution:
```bash
agent calibrate --recalibrate
```

### "Tesseract not found"

Install Tesseract and ensure it's in PATH:
```powershell
winget install --id UB-Mannheim.TesseractOCR
```

### "API key not configured"

Set your Gemini API key:
```bash
# In .env file
GEMINI_API_KEY=your-key-here

# Or as environment variable
set GEMINI_API_KEY=your-key-here
```

### View Session Stats

```bash
# Latest session
agent stats

# All sessions
agent stats --all
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/ -q`
5. Commit: `git commit -m "feat: add my feature"`
6. Push: `git push origin feat/my-feature`
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
