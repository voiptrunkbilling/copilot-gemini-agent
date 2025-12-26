# Milestone Validation Guide

This document describes how to validate each milestone.

---

## M1: Core Skeleton

**Target Date:** Dec 28, 2024

### What's Delivered

- [x] Project structure with all module stubs
- [x] Pydantic config schema + YAML loading
- [x] State manager with JSON checkpoint
- [x] Basic TUI shell (status display)
- [x] Kill switch listener (global hotkey)
- [x] CLI commands (`--version`, `run`, `calibrate`)
- [x] Logging infrastructure

### Acceptance Test

**Test 1: Version Check**
```bash
python -m copilot_agent --version
# Expected: copilot-gemini-agent v0.1.0
```

**Test 2: TUI Launch**
```bash
python -m copilot_agent run --task "Test task" --dry-run
# Expected: TUI appears showing:
#   - Session ID
#   - Task description
#   - Phase: IDLE
#   - "Press Q to quit"
```

**Test 3: Kill Switch**
1. Start the agent: `python -m copilot_agent run --task "Test" --dry-run`
2. Press `Ctrl+Shift+K`
3. Expected: TUI shows "Kill switch triggered", clean shutdown within 1 second

**Test 4: Config Loading**
1. Create `~/.copilot-agent/config.yaml` with custom values
2. Run: `python -m copilot_agent run --task "Test" --dry-run`
3. Expected: Agent uses config values (visible in debug output with `-v` flag)

**Test 5: State Checkpoint**
1. Run agent briefly, then quit
2. Check `~/.copilot-agent/sessions/<id>/checkpoint.json` exists
3. Expected: Valid JSON with session_id, task, phase

### Manual Verification Checklist

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `python -m copilot_agent --help` shows all commands
- [ ] TUI renders correctly in terminal (Windows Terminal, PowerShell, CMD)
- [ ] TUI renders correctly over SSH
- [ ] Kill switch works even when TUI not focused
- [ ] Logs written to session folder
- [ ] No secrets appear in logs

---

## M2: GUI Actions

**Target Date:** Dec 31, 2024

### What's Delivered

- [ ] Actuator controller with all core actions
- [ ] Action validation (bounds, allowlist)
- [ ] Manual calibration overlay
- [ ] Window focus management (pywin32)
- [ ] Click, type, hotkey, copy/paste
- [ ] Screenshot capture
- [ ] DPI awareness

### Acceptance Test

**Test 1: Calibration**
```bash
python -m copilot_agent calibrate
# Expected:
#   1. Semi-transparent overlay appears
#   2. Instructions shown: "Click on Copilot Chat input field"
#   3. After 3 clicks, calibration saved
#   4. Coordinates stored in session
```

**Test 2: Dry Run Actions**
```bash
# With VS Code open
python -m copilot_agent run --task "Test" --dry-run -v
# Expected: Logs show planned click coordinates for VS Code
```

**Test 3: Focus Window**
```bash
# With VS Code open but not focused
python -m copilot_agent test-focus
# Expected: VS Code window comes to foreground
```

**Test 4: Type Text (Controlled)**
```bash
# With VS Code open, cursor in a text file
python -m copilot_agent test-type --text "Hello World"
# Expected: "Hello World" typed into active input
```

### Manual Verification Checklist

- [ ] Calibration overlay visible on all monitors
- [ ] Clicks registered at correct positions
- [ ] Type speed is reasonable (not too fast)
- [ ] Copy/paste works reliably
- [ ] Works at 100%, 125%, 150% scaling
- [ ] Window focus works when VS Code minimized (brings back)
- [ ] Actions blocked on non-allowed windows

---

## M3: Perception MVP

**Target Date:** Jan 4, 2025

### What's Delivered

- [ ] Screenshot → OCR pipeline
- [ ] Text anchor detection ("Copilot", "Ask Copilot")
- [ ] Template matching for icons
- [ ] Auto-detection of Copilot Chat elements
- [ ] Response area boundary detection
- [ ] Response completion detection
- [ ] Gemini Vision fallback

### Acceptance Test

**Test 1: OCR Detection**
```bash
# With VS Code open, Copilot Chat visible
python -m copilot_agent test-perception
# Expected:
#   - "Copilot Chat detected at (x, y)"
#   - "Input field detected at (x, y)"
#   - "Detection method: OCR"
```

**Test 2: Multi-Theme**
1. Change VS Code to Light theme
2. Run: `python -m copilot_agent test-perception`
3. Expected: Detection succeeds

**Test 3: Scaling**
1. Set Windows display scaling to 150%
2. Run: `python -m copilot_agent test-perception`
3. Expected: Detection succeeds with correct coordinates

**Test 4: Vision Fallback**
```bash
python -m copilot_agent test-perception --force-vision
# Expected: Uses Gemini Vision, logs API call, returns coordinates
```

### Screenshot Evidence Required

Attach screenshots showing successful detection on:
- [ ] Dark theme @ 100%
- [ ] Dark theme @ 150%
- [ ] Light theme @ 100%

---

## M4: Full Loop

**Target Date:** Jan 8, 2025

### What's Delivered

- [ ] Gemini API client with structured prompting
- [ ] Response parsing (verdict extraction)
- [ ] Complete orchestrator loop (one iteration)
- [ ] TUI real-time updates
- [ ] APPROVE_ITERATIONS mode
- [ ] Error handling for API failures

### Acceptance Test

**Primary Test: One Full Iteration**

1. Setup:
   - VS Code open with Copilot Chat visible
   - `GEMINI_API_KEY` set
   
2. Run:
   ```bash
   python -m copilot_agent run --task "Write a Python function that reverses a string"
   ```

3. Expected sequence:
   - [ ] Agent focuses VS Code
   - [ ] Agent clicks Copilot Chat input
   - [ ] Agent types the prompt
   - [ ] Agent presses Enter
   - [ ] Agent waits for response
   - [ ] Agent captures Copilot's response
   - [ ] TUI shows captured response
   - [ ] Agent sends to Gemini
   - [ ] TUI shows Gemini verdict (ACCEPT/CRITIQUE/CLARIFY)
   - [ ] Agent pauses, waits for human input (C to continue, A to abort)

4. Verify:
   - [ ] `checkpoint.json` contains captured response
   - [ ] `audit.jsonl` contains all actions
   - [ ] Gemini response properly parsed

### Demo Recording

30-minute session covering:
- [ ] Clean install
- [ ] Configuration
- [ ] Run with simple task
- [ ] Walk through one iteration
- [ ] Show TUI updates
- [ ] Demonstrate abort
- [ ] Review logs

---

## M5: Production MVP

**Target Date:** Jan 14, 2025

### What's Delivered

- [ ] Multi-iteration loops
- [ ] Automatic follow-up prompt injection
- [ ] Stopping conditions (ACCEPT, max iterations)
- [ ] Full error handling and retry logic
- [ ] Checkpoint save/resume
- [ ] All operation modes (STEP, APPROVE, AUTO)
- [ ] Metrics collection and display

### Acceptance Test

**Test 1: Multi-Iteration**
```bash
python -m copilot_agent run \
  --task "Write a function to validate email addresses" \
  --max-iterations 5 \
  --mode auto
```
Expected: Agent completes 2-5 iterations until Gemini accepts

**Test 2: Resume from Checkpoint**
1. Start a session
2. Kill process mid-iteration (Ctrl+C)
3. Run: `python -m copilot_agent resume --session <id>`
4. Expected: Resumes from last checkpoint

**Test 3: Metrics**
```bash
python -m copilot_agent stats --session <latest>
```
Expected: Shows capture rate, OCR success, Gemini parse rate, timing

**Test 4: Error Recovery**
1. Disconnect network mid-session (simulate Gemini failure)
2. Expected: Agent retries 3 times, then pauses for human intervention

### Final Checklist

- [ ] All modes work (step, approve, auto)
- [ ] Kill switch works in all modes
- [ ] Checkpoint/resume reliable
- [ ] Metrics accurate
- [ ] Audit trail complete
- [ ] No memory leaks in long sessions
- [ ] Documentation complete

---

## Sign-Off Criteria

### M4 Gate (Required for M5)

- [ ] Demo completed (live or recorded)
- [ ] All M4 acceptance tests pass
- [ ] At least one successful end-to-end iteration on user's machine
- [ ] No blocking bugs

### MVP Release

- [ ] All M5 acceptance tests pass
- [ ] Documentation complete
- [ ] README installation instructions verified
- [ ] At least 3 successful multi-iteration sessions
- [ ] Metrics show >80% capture success rate
