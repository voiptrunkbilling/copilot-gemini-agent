# M2 Demo Runbook: GUI Actions + Calibration

**Version**: 0.2.0  
**Target**: Windows 10/11  
**Date**: December 26, 2024

---

## Prerequisites

### 1. System Requirements

- **OS**: Windows 10 (1903+) or Windows 11
- **Python**: 3.11 or higher
- **VS Code**: Latest version with GitHub Copilot Chat extension
- **Display**: Any resolution, tested at 100% and 125% DPI scaling

### 2. Install Dependencies

```powershell
# Clone and enter project
cd copilot-gemini-agent

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installation
python -m copilot_agent --version
# Expected: copilot-gemini-agent v0.2.0
```

### 3. Dependency Checklist

| Package | Version | Purpose |
|---------|---------|---------|
| pyautogui | ≥0.9.54 | Mouse/keyboard automation |
| pywin32 | ≥306 | Windows window management |
| mss | ≥9.0.1 | Fast screenshot capture |
| pynput | ≥1.7.6 | Global hotkey listener |
| rich | ≥13.7.0 | TUI rendering |

Verify with:
```powershell
pip list | findstr "pyautogui pywin32 mss pynput rich"
```

---

## Validation Steps

### Step 1: System Info Check

```powershell
python -c "from copilot_agent.actuator import get_dpi_info, get_primary_screen, IS_WINDOWS; dpi=get_dpi_info(); screen=get_primary_screen(); print(f'Windows: {IS_WINDOWS}'); print(f'Screen: {screen.width}x{screen.height}'); print(f'DPI: {dpi.scale_factor:.0%}')"
```

**Expected output**:
```
Windows: True
Screen: 1920x1080
DPI: 100%
```

### Step 2: Calibration

**Preparation**:
1. Open VS Code
2. Open Copilot Chat panel (Ctrl+Shift+I or click Copilot icon)
3. Position VS Code so Copilot Chat is visible

**Run calibration**:
```powershell
python -m copilot_agent calibrate
```

**During calibration**:
1. Semi-transparent overlay appears
2. Click **VS Code title bar** when prompted
3. Click **Copilot Chat input box**
4. Click **Response area top-left corner**
5. Click **Response area bottom-right corner**

**Verify calibration saved**:
```powershell
python -m copilot_agent calibrate --show
```

**Expected output**:
```
System Information
┌─────────────┬───────────────┐
│ Platform    │ Windows       │
│ Screen Size │ 1920 x 1080   │
│ DPI Scale   │ 100% (96 DPI) │
│ DPI Aware   │ Yes           │
└─────────────┴───────────────┘

Current Calibration
┌──────────────────────────────────┬──────┬──────┬────────┐
│ Element                          │ X    │ Y    │ Status │
├──────────────────────────────────┼──────┼──────┼────────┤
│ VS Code window (click title bar) │ 960  │ 15   │ ✓      │
│ Copilot Chat input box           │ 1650 │ 950  │ ✓      │
│ Copilot response area (top-left) │ 1400 │ 200  │ ✓      │
│ Copilot response area (bottom-…) │ 1900 │ 900  │ ✓      │
└──────────────────────────────────┴──────┴──────┴────────┘
```

**Calibration file location**:
```
%USERPROFILE%\.copilot-agent\calibration\calibration.json
```

### Step 3: Dry-Run Test

```powershell
python -m copilot_agent run --dry-run --task "Test prompt for M2 validation"
```

**Expected behavior**:
- TUI renders with header, status panel, task description
- Actions logged with `DRY-RUN:` prefix
- No actual GUI interaction
- Press `q` to exit TUI

### Step 4: Real Actions Test

**WARNING**: This performs real GUI actions!

```powershell
python -m copilot_agent test-actions
```

**Expected sequence**:
1. ✓ Focus VS Code window
2. ✓ Click Copilot input box
3. ✓ Type "hello"
4. ✓ Press Enter
5. ✓ Select all (Ctrl+A)
6. ✓ Copy (Ctrl+C)
7. ✓ Read clipboard content

**Expected output**:
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
  ✓ Copy Selection (Ctrl+A, Ctrl+C): Copied 5 characters
  ✓ Read Clipboard: Read 5 characters → hello

Test sequence complete.
```

### Step 5: Kill Switch Test

**Test procedure**:
1. Start a long typing operation
2. Press **Ctrl+Shift+K** during typing
3. Verify typing stops immediately

```powershell
# Run test with kill switch active
python -c "
from copilot_agent.actuator import ActionExecutor
from copilot_agent.safety.killswitch import KillSwitch
import time

ks = KillSwitch()
ks.start()
print('Kill switch active. Press Ctrl+Shift+K to test.')
print('Starting long typing in 3 seconds...')
time.sleep(3)

ex = ActionExecutor(dry_run=False, kill_switch_check=ks.is_triggered)
result = ex.type_text('Testing kill switch... ' * 20)
print(f'Result: success={result.success}, error={result.error}')
ks.stop()
"
```

**If kill switch works**: Typing stops mid-sentence, `error=Kill switch triggered`

### Step 6: DPI Scaling Test

**Change DPI**:
1. Right-click Desktop → Display Settings
2. Under "Scale and layout", change to **125%**
3. Sign out/in or restart (some changes require this)

**Re-run validation**:
```powershell
# Check new DPI is detected
python -c "from copilot_agent.actuator import get_dpi_info; print(get_dpi_info())"

# Re-calibrate for new DPI
python -m copilot_agent calibrate --recalibrate

# Test actions
python -m copilot_agent test-actions --dry-run
```

**Expected**: Coordinates should still map correctly after recalibration.

---

## Automated Validation Script

Run the full validation suite:

```powershell
scripts\validate_m2_windows.bat
```

This script runs all tests interactively with prompts.

---

## Troubleshooting

### Window Focus Fails

**Symptom**: `focus_window` returns error

**Solutions**:
1. Run as Administrator (some apps block focus from non-admin)
2. Disable "Prevent focus stealing" in Windows settings
3. Ensure VS Code is not minimized

### Calibration Overlay Doesn't Appear

**Symptom**: Calibration hangs or no overlay

**Solutions**:
1. Install tkinter: `pip install tk` (usually bundled with Python)
2. Use CLI fallback: Enter coordinates manually when prompted

### Clipboard Races

**Symptom**: Wrong text copied

**Solutions**:
1. Increase delays in `ActionConfig`:
   ```python
   config = ActionConfig(action_delay_ms=100, hotkey_delay_ms=100)
   ```
2. Check no other app is accessing clipboard

### DPI Coordinates Off

**Symptom**: Clicks miss targets at 125% DPI

**Solutions**:
1. Verify DPI awareness: `get_dpi_info().is_aware` should be `True`
2. Recalibrate after DPI change
3. Restart Python process after DPI change

---

## Session Artifacts

After running tests, check these directories:

```
%USERPROFILE%\.copilot-agent\
├── calibration\
│   └── calibration.json      # Saved calibration points
└── sessions\
    └── <session-id>\
        ├── checkpoint.json   # Session state
        ├── screenshots\      # Captured screenshots
        └── captures\         # Raw capture data
```

---

## Demo Video Checklist

When recording the M2 demo video, capture:

1. [ ] Terminal showing `python -m copilot_agent --version` → v0.2.0
2. [ ] System info output (screen size, DPI)
3. [ ] Calibration overlay appearing
4. [ ] Clicking each calibration point
5. [ ] Calibration success message
6. [ ] `calibrate --show` output
7. [ ] TUI rendering during `run --dry-run`
8. [ ] Real actions typing "hello" in VS Code
9. [ ] Kill switch stopping typing
10. [ ] Final summary

---

## Sign-Off Criteria

M2 is validated when:

- [x] All unit tests pass (100/100)
- [ ] `validate_m2_windows.bat` completes on Windows 10/11
- [ ] Calibration saves JSON correctly
- [ ] Dry-run shows planned actions
- [ ] Real actions work at 100% DPI
- [ ] Real actions work at 125% DPI
- [ ] Kill switch stops actions immediately
- [ ] Session checkpoint created

---

## Next Steps (M3)

After M2 sign-off:

1. **M3: Perception** (Target: Jan 4)
   - Tesseract OCR integration
   - Template matching with OpenCV
   - Gemini Vision fallback
   - Auto-detection of Copilot Chat elements

2. **M3 Prerequisites**:
   - Install Tesseract OCR for Windows
   - Gemini API key configured
