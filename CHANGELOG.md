# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2025-01-XX (M6: Production Readiness)

### Added
- **Orchestrator M5 integration**:
  - Full checkpoint/restore with `AtomicCheckpointer`
  - Circuit breakers for reviewer, vision, and UI operations
  - Retry policies with exponential backoff
  - Metrics emission for all operations
  - Desync detection and recovery
  - Kill switch preemptive checks

- **Enhanced budget system** (`OrchestratorBudget`):
  - Config-driven limits via `from_config(AgentConfig)`
  - Runtime tracking with `start()`/`get_runtime_seconds()`
  - Usage warnings at configurable thresholds
  - `check_all()` for comprehensive validation
  - `get_usage_percentage()` for budget visibility

- **CLI improvements**:
  - `agent stats` - Session statistics with Rich tables
  - `agent stats --all` - All sessions summary
  - `agent resume` - Resume from checkpoint
  - `agent resume --list` - List resumable sessions
  - `agent config` - Configuration management
  - `agent config --validate` - Pre-run validation
  - `agent config --secrets` - Secrets status

- **Windows installer** (`scripts/install_windows.ps1`):
  - Automatic venv creation
  - Pip dependency installation
  - Tesseract installation via winget
  - Sample calibration setup
  - Environment file creation
  - Installation validation

- **Secrets management** (`SecretsManager`):
  - Env-only secrets (never in config files)
  - `get_gemini_api_key()` with helpful errors
  - `validate_all_required()` for pre-run checks
  - `get_status()` with masked values

- **Configuration validation**:
  - `ConfigurationError` with suggestions
  - Environment variable overrides
  - Cross-field consistency checks
  - `validate_for_run()` pre-flight checks

- **M6 validation script** (`scripts/validate_m6.ps1`):
  - Installation verification
  - Calibration test
  - Full run loop test
  - Kill switch test
  - Crash/resume test
  - Stats command test

### Changed
- README completely rewritten with:
  - Quick start guide
  - Command reference tables
  - Troubleshooting section
  - Safety features documentation
  
- `load_config()` now raises `ConfigurationError` for missing explicit paths
- Configuration supports env var overrides (COPILOT_*, GEMINI_*)

## [0.5.0] - 2025-01-XX (M5: Hardening & Recovery)

### Added
- **Checkpoint system** (`safety/checkpoint.py`):
  - `AtomicCheckpointer` with temp file + rename
  - Auto-save every N iterations
  - Session listing with metadata
  - Resume point calculation

- **Metrics collection** (`safety/metrics.py`):
  - `MetricsCollector` with event recording
  - Duration tracking for operations
  - Session aggregation for stats
  - JSON persistence

- **Resilience primitives** (`safety/resilience.py`):
  - `CircuitBreaker` with configurable thresholds
  - `RetryPolicy` with exponential backoff
  - Factory functions for reviewer/vision/UI

- **Desync detection** (`safety/desync.py`):
  - `DesyncDetector` for loop anomalies
  - Iteration timing analysis
  - Staleness detection
  - Repeated prompt detection

### Changed
- Kill switch now supports ScrollLock as alternative key
- All safety modules have comprehensive unit tests

## [0.2.0] - 2024-12-26 (M2: GUI Actions)

### Added
- **Platform utilities** (`actuator/platform.py`):
  - OS detection (Windows/Linux/macOS)
  - DPI awareness support for Windows 10/11
  - Screen info and multi-monitor detection
  - Coordinate scaling utilities

- **Window management** (`actuator/window.py`):
  - pywin32-based window focus and enumeration
  - `find_window_by_title()`, `find_vscode_window()`
  - `focus_window()` with retry logic
  - `verify_active_window()` for focus verification
  - Window rect and client area queries

- **Action executor** (`actuator/actions.py`):
  - Full pyautogui integration
  - `click()`, `type_text()`, `hotkey()`, `press_key()`
  - `copy_selection()`, `paste_text()`, `read_clipboard()`
  - `scroll()`, `move_mouse()`, `wait()`
  - Kill switch pre-action checks
  - DPI-aware coordinate handling
  - Forbidden hotkey protection

- **Screenshot capture** (`actuator/screenshot.py`):
  - mss-based fast screenshot capture
  - Full screen and region capture
  - Window capture support
  - Region dataclass with conversions

- **Calibration system** (`actuator/calibration.py`):
  - Tkinter overlay for manual UI element calibration
  - Calibration points: VS Code window, Copilot input, response area
  - JSON persistence in `~/.copilot-agent/calibration/`
  - CLI fallback for headless environments

- **Action pipeline** (`actuator/pipeline.py`):
  - Unified interface for all GUI actions
  - Calibration-aware coordinates
  - Kill switch integration throughout
  - High-level actions: `focus_vscode()`, `click_copilot_input()`, `send_prompt()`
  - Pipeline step execution with verification

- **CLI enhancements**:
  - `agent calibrate` - Full calibration workflow
  - `agent calibrate --show` - View current calibration
  - `agent calibrate --recalibrate` - Force recalibration
  - `agent test-actions` - M2 validation command

- **Unit tests**: 50+ new tests for actuator components

### Changed
- CLI now initializes ActionPipeline in `run` command
- ActionExecutor uses configurable ActionConfig

### Fixed
- Type annotations for optional pynput import
- datetime.utcnow() deprecation warnings

## [0.1.0] - 2024-12-26 (M1: Skeleton)

### Added
- Initial project skeleton
- Configuration schema with Pydantic
- State manager with JSON checkpoints
- Kill switch with global hotkey support
- Basic TUI shell using Rich
- CLI commands: `run`, `calibrate`, `resume`, `stats`
- Logging infrastructure with structlog
- Unit tests for core modules (58 tests)
