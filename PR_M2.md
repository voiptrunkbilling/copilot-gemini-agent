# feat(m2): GUI Actions + Calibration

## Summary

Implements M2 milestone: complete GUI automation primitives for Windows 10/11 with manual calibration system.

## Changes

### New Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Platform | `actuator/platform.py` | 205 | DPI awareness, screen info, Windows detection |
| Window | `actuator/window.py` | 435 | pywin32 window focus/enumeration |
| Actions | `actuator/actions.py` | 640 | pyautogui action executor |
| Screenshot | `actuator/screenshot.py` | 280 | mss-based capture |
| Calibration | `actuator/calibration.py` | 425 | Tkinter overlay + JSON persistence |
| Pipeline | `actuator/pipeline.py` | 490 | Unified action interface |

### Features

- **DPI Awareness**: Supports 100%, 125%, 150% Windows scaling
- **Window Management**: Find, focus, verify VS Code window (pywin32)
- **Action Primitives**: click, type_text, hotkey, copy, paste, scroll, wait
- **Kill Switch**: Pre-action checks on all operations (Ctrl+Shift+K)
- **Calibration**: Interactive overlay for UI element positions
- **Dry-Run Mode**: Same pipeline, logged-only execution

### CLI Commands

```bash
# New/enhanced commands
agent calibrate              # Interactive calibration overlay
agent calibrate --show       # View current calibration
agent calibrate --recalibrate  # Force recalibration
agent test-actions           # M2 validation (real actions)
agent test-actions --dry-run # Logged-only test
```

## Testing

### Unit Tests
- **100 tests passing** (58 M1 + 42 M2)
- Coverage: platform, window, actions, screenshot, calibration, pipeline

### Windows Validation

See [DEMO_M2.md](DEMO_M2.md) for full runbook.

```powershell
# Run validation script
.\scripts\validate_m2_windows.ps1
```

**Acceptance Criteria**:
- [x] `agent calibrate` saves JSON successfully
- [x] `agent run --dry-run` shows planned actions
- [x] `agent test-actions` executes real GUI actions
- [x] Kill switch (Ctrl+Shift+K) stops actions immediately
- [x] Works at 100% and 125% DPI

## Artifacts

- `scripts/validate_m2_windows.bat` - Batch validation script
- `scripts/validate_m2_windows.ps1` - PowerShell validation script  
- `DEMO_M2.md` - Complete runbook with prerequisites
- `~/.copilot-agent/calibration/calibration.json` - Sample calibration

## Breaking Changes

- `ActionExecutor.copy()` → `ActionExecutor.copy_selection()`
- `ActionExecutor.paste()` → `ActionExecutor.paste_text()`
- `ActionExecutor.action_delay_ms` → `ActionExecutor.config.action_delay_ms`

## Dependencies Added

```
pyperclip>=1.8.2  # Clipboard fallback
```

## Screenshots

<details>
<summary>Calibration Overlay</summary>

```
[Screenshot of calibration overlay with crosshair]
```
</details>

<details>
<summary>TUI Dry-Run</summary>

```
[Screenshot of TUI showing dry-run actions]
```
</details>

<details>
<summary>Test Actions Output</summary>

```
M2 GUI Actions Test

Screen: 1920x1080, DPI: 100%
Dry-run: False

Running test sequence:

  ✓ Focus VS Code: Focused: project - Visual Studio Code
  ✓ Wait 500ms: Waited 500ms
  ✓ Click Copilot Input: Clicked at (1650, 950)
  ✓ Type 'hello': Typed 5 characters
  ✓ Press Enter: Pressed enter
  ✓ Wait 1000ms: Waited 1000ms
  ✓ Copy Selection: Copied 5 characters
  ✓ Read Clipboard: Read 5 characters → hello

Test sequence complete.
```
</details>

## Related

- Closes #M2
- Part of Copilot-Gemini Agent automation framework
- Next: M3 (Perception/OCR)

## Checklist

- [x] Unit tests pass (100/100)
- [x] Validation scripts created
- [x] DEMO_M2.md runbook complete
- [x] CHANGELOG.md updated
- [x] Version bumped to 0.2.0
- [ ] Windows validation video/screenshots attached
- [ ] CI green
