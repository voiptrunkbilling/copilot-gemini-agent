# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
