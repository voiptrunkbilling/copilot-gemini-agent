"""
Unified action pipeline.

Coordinates all GUI actions through a single interface with:
- Kill switch integration
- Dry-run support  
- Action logging and verification
- Calibration-aware coordinates
"""

import time
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict, Union
from dataclasses import dataclass, field
from enum import Enum

from copilot_agent.logging import get_logger
from copilot_agent.actuator.platform import IS_WINDOWS, get_dpi_info, set_dpi_awareness
from copilot_agent.actuator.window import WindowManager, WindowInfo
from copilot_agent.actuator.actions import ActionExecutor, ActionResult, ActionType, ActionConfig
from copilot_agent.actuator.screenshot import ScreenshotCapture, Region, ScreenshotResult
from copilot_agent.actuator.calibration import (
    CalibrationManager, 
    CalibrationData, 
    run_calibration,
)

logger = get_logger(__name__)


class PipelineAction(str, Enum):
    """High-level pipeline actions."""
    
    FOCUS_VSCODE = "focus_vscode"
    CLICK_COPILOT_INPUT = "click_copilot_input"
    TYPE_PROMPT = "type_prompt"
    SEND_PROMPT = "send_prompt"
    WAIT_FOR_RESPONSE = "wait_for_response"
    CAPTURE_RESPONSE = "capture_response"
    COPY_SELECTION = "copy_selection"
    READ_CLIPBOARD = "read_clipboard"
    SCREENSHOT = "screenshot"
    CUSTOM_CLICK = "custom_click"
    CUSTOM_TYPE = "custom_type"
    WAIT = "wait"


@dataclass
class PipelineStep:
    """A step in the action pipeline."""
    
    action: PipelineAction
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    # Verification
    verify_window: Optional[str] = None  # Title pattern to verify
    verify_timeout_ms: int = 5000
    
    # Timing
    pre_delay_ms: int = 0
    post_delay_ms: int = 0


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    
    success: bool
    steps_completed: int
    total_steps: int
    results: List[ActionResult] = field(default_factory=list)
    error: Optional[str] = None
    aborted_by_killswitch: bool = False
    data: Dict[str, Any] = field(default_factory=dict)  # For captured text, etc.


class ActionPipeline:
    """
    Unified action pipeline for GUI automation.
    
    Provides a high-level interface that:
    - Integrates all action executors
    - Uses calibration data for coordinates
    - Checks kill switch before each action
    - Supports dry-run mode
    - Logs all actions
    """
    
    def __init__(
        self,
        dry_run: bool = False,
        kill_switch_check: Optional[Callable[[], bool]] = None,
        calibration_path: Optional[Path] = None,
        action_config: Optional[ActionConfig] = None,
    ):
        """
        Initialize action pipeline.
        
        Args:
            dry_run: If True, don't execute actual actions
            kill_switch_check: Function returning True if kill switch triggered
            calibration_path: Path to calibration data
            action_config: Configuration for action timing
        """
        self.dry_run = dry_run
        self._kill_switch_check = kill_switch_check
        
        # Set DPI awareness early
        set_dpi_awareness()
        
        # Load calibration
        self._calibration_manager = CalibrationManager(calibration_path)
        self._calibration: Optional[CalibrationData] = None
        self._load_calibration()
        
        # Initialize executors
        self._window_manager = WindowManager(dry_run=dry_run)
        self._action_executor = ActionExecutor(
            config=action_config,
            dry_run=dry_run,
            kill_switch_check=kill_switch_check,
        )
        self._screenshot = ScreenshotCapture(dry_run=dry_run)
        
        # Track VS Code window
        self._vscode_hwnd: Optional[int] = None
        
        logger.info(
            "ActionPipeline initialized",
            dry_run=dry_run,
            has_calibration=self._calibration is not None and self._calibration.is_complete(),
        )
    
    def _load_calibration(self):
        """Load calibration data."""
        self._calibration = self._calibration_manager.load()
        if self._calibration and self._calibration.is_complete():
            logger.info("Calibration loaded", dpi=self._calibration.dpi_scale)
        else:
            logger.warning("No complete calibration found")
    
    def reload_calibration(self):
        """Reload calibration data from disk."""
        self._load_calibration()
    
    @property
    def has_calibration(self) -> bool:
        """Check if calibration data is available."""
        return self._calibration is not None and self._calibration.is_complete()
    
    def _check_kill_switch(self) -> bool:
        """Check if kill switch is triggered."""
        if self._kill_switch_check:
            return self._kill_switch_check()
        return False
    
    # High-level actions
    
    def focus_vscode(self) -> ActionResult:
        """Focus VS Code window."""
        if self._check_kill_switch():
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error="Kill switch triggered",
            )
        
        # Try by hwnd first if we have it
        if self._vscode_hwnd:
            success, error = self._window_manager.focus_window(hwnd=self._vscode_hwnd)
            if success:
                return ActionResult(
                    success=True,
                    action_type=ActionType.FOCUS_WINDOW,
                    message="Focused VS Code (cached hwnd)",
                )
        
        # Try to find VS Code window
        window = self._window_manager.find_vscode_window()
        if not window:
            # Try broader search
            window = self._window_manager.find_window_by_title("Visual Studio Code")
            if not window:
                window = self._window_manager.find_window_by_title("Code -")
        
        if not window:
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error="VS Code window not found",
            )
        
        # Focus it
        self._vscode_hwnd = window.hwnd
        success, error = self._window_manager.focus_window(window=window)
        
        return ActionResult(
            success=success,
            action_type=ActionType.FOCUS_WINDOW,
            message=f"Focused: {window.title}" if success else "",
            error=error,
        )
    
    def verify_vscode_focused(self) -> bool:
        """Verify VS Code is currently focused."""
        is_active, info = self._window_manager.verify_active_window(
            title_contains="Visual Studio Code"
        )
        if not is_active and info:
            # Also check for "Code -" pattern
            is_active = "code" in info.title.lower()
        return is_active
    
    def click_copilot_input(self) -> ActionResult:
        """Click in the Copilot Chat input box."""
        if not self.has_calibration:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error="No calibration data - run 'agent calibrate' first",
            )
        
        point = self._calibration.copilot_input
        if not point.calibrated or point.x is None or point.y is None:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error="Copilot input not calibrated",
            )
        
        return self._action_executor.click(point.x, point.y)
    
    def type_text(self, text: str, use_clipboard: bool = False) -> ActionResult:
        """Type text at current cursor position."""
        return self._action_executor.type_text(text, use_clipboard=use_clipboard)
    
    def send_prompt(self, prompt: str) -> ActionResult:
        """
        Send a prompt to Copilot Chat.
        
        This is a composite action:
        1. Focus VS Code
        2. Click Copilot input
        3. Type prompt
        4. Press Enter
        """
        # Focus VS Code first
        result = self.focus_vscode()
        if not result.success:
            return result
        
        # Click input
        result = self.click_copilot_input()
        if not result.success:
            return result
        
        # Small delay for focus
        self._action_executor.wait(100)
        
        # Type prompt (use clipboard for longer prompts)
        use_clipboard = len(prompt) > 200
        result = self.type_text(prompt, use_clipboard=use_clipboard)
        if not result.success:
            return result
        
        # Press Enter to send
        result = self._action_executor.press_key("enter")
        
        return result
    
    def capture_response_screenshot(
        self, 
        save_path: Optional[Path] = None,
    ) -> ScreenshotResult:
        """Capture screenshot of Copilot response area."""
        if not self.has_calibration:
            return ScreenshotResult(
                success=False,
                error="No calibration data",
            )
        
        region = self._calibration.get_response_region()
        if not region:
            return ScreenshotResult(
                success=False,
                error="Response region not calibrated",
            )
        
        return self._screenshot.capture_region(region, save_path)
    
    def copy_selection(self) -> ActionResult:
        """Copy current selection to clipboard."""
        return self._action_executor.copy_selection()
    
    def read_clipboard(self) -> ActionResult:
        """Read clipboard contents."""
        return self._action_executor.read_clipboard()
    
    def hotkey(self, *keys: str) -> ActionResult:
        """Execute a hotkey combination."""
        return self._action_executor.hotkey(*keys)
    
    def click(self, x: int, y: int, clicks: int = 1) -> ActionResult:
        """Click at specific coordinates."""
        return self._action_executor.click(x, y, clicks=clicks)
    
    def wait(self, duration_ms: int) -> ActionResult:
        """Wait for specified duration."""
        return self._action_executor.wait(duration_ms)
    
    def screenshot(
        self, 
        region: Optional[Union[Region, tuple]] = None,
        save_path: Optional[Path] = None,
    ) -> ScreenshotResult:
        """Capture screenshot."""
        if region:
            return self._screenshot.capture_region(region, save_path)
        return self._screenshot.capture_full_screen(save_path)
    
    # Pipeline execution
    
    def execute_step(self, step: PipelineStep) -> ActionResult:
        """
        Execute a single pipeline step.
        
        Args:
            step: Step to execute
            
        Returns:
            ActionResult
        """
        # Pre-delay
        if step.pre_delay_ms > 0:
            self._action_executor.wait(step.pre_delay_ms)
        
        # Check kill switch
        if self._check_kill_switch():
            return ActionResult(
                success=False,
                action_type=ActionType.WAIT,
                error="Kill switch triggered",
            )
        
        # Execute action
        result: ActionResult
        
        if step.action == PipelineAction.FOCUS_VSCODE:
            result = self.focus_vscode()
            
        elif step.action == PipelineAction.CLICK_COPILOT_INPUT:
            result = self.click_copilot_input()
            
        elif step.action == PipelineAction.TYPE_PROMPT:
            text = step.params.get("text", "")
            use_clipboard = step.params.get("use_clipboard", False)
            result = self.type_text(text, use_clipboard=use_clipboard)
            
        elif step.action == PipelineAction.SEND_PROMPT:
            prompt = step.params.get("prompt", "")
            result = self.send_prompt(prompt)
            
        elif step.action == PipelineAction.COPY_SELECTION:
            result = self.copy_selection()
            
        elif step.action == PipelineAction.READ_CLIPBOARD:
            result = self.read_clipboard()
            
        elif step.action == PipelineAction.CUSTOM_CLICK:
            x = step.params.get("x", 0)
            y = step.params.get("y", 0)
            clicks = step.params.get("clicks", 1)
            result = self.click(x, y, clicks)
            
        elif step.action == PipelineAction.CUSTOM_TYPE:
            text = step.params.get("text", "")
            result = self.type_text(text)
            
        elif step.action == PipelineAction.WAIT:
            duration = step.params.get("duration_ms", 1000)
            result = self.wait(duration)
            
        elif step.action == PipelineAction.SCREENSHOT:
            region = step.params.get("region")
            path = step.params.get("save_path")
            screenshot_result = self.screenshot(region, path)
            result = ActionResult(
                success=screenshot_result.success,
                action_type=ActionType.SCREENSHOT,
                error=screenshot_result.error,
            )
            
        else:
            result = ActionResult(
                success=False,
                action_type=ActionType.WAIT,
                error=f"Unknown action: {step.action}",
            )
        
        # Post-delay
        if step.post_delay_ms > 0 and result.success:
            self._action_executor.wait(step.post_delay_ms)
        
        # Verify window if requested
        if step.verify_window and result.success:
            is_active, _ = self._window_manager.verify_active_window(
                title_contains=step.verify_window
            )
            if not is_active:
                logger.warning(
                    "Window verification failed",
                    expected=step.verify_window,
                )
        
        return result
    
    def execute_pipeline(self, steps: List[PipelineStep]) -> PipelineResult:
        """
        Execute a sequence of pipeline steps.
        
        Stops on first failure or kill switch.
        
        Args:
            steps: List of steps to execute
            
        Returns:
            PipelineResult with all results
        """
        results: List[ActionResult] = []
        data: Dict[str, Any] = {}
        
        for i, step in enumerate(steps):
            # Check kill switch
            if self._check_kill_switch():
                return PipelineResult(
                    success=False,
                    steps_completed=i,
                    total_steps=len(steps),
                    results=results,
                    error="Kill switch triggered",
                    aborted_by_killswitch=True,
                )
            
            logger.info(
                "Executing step",
                step=i+1,
                total=len(steps),
                action=step.action.value,
                description=step.description,
            )
            
            result = self.execute_step(step)
            results.append(result)
            
            # Collect data from certain actions
            if result.success and result.data:
                if step.action == PipelineAction.READ_CLIPBOARD:
                    data["clipboard"] = result.data
                elif step.action == PipelineAction.COPY_SELECTION:
                    data["copied"] = result.data
            
            if not result.success:
                return PipelineResult(
                    success=False,
                    steps_completed=i,
                    total_steps=len(steps),
                    results=results,
                    error=result.error,
                    data=data,
                )
        
        return PipelineResult(
            success=True,
            steps_completed=len(steps),
            total_steps=len(steps),
            results=results,
            data=data,
        )
    
    # Utility methods
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of current calibration."""
        if not self._calibration:
            return {"status": "not_loaded"}
        
        return {
            "status": "complete" if self._calibration.is_complete() else "incomplete",
            "dpi_scale": self._calibration.dpi_scale,
            "screen_size": (self._calibration.screen_width, self._calibration.screen_height),
            "vscode_window": {
                "calibrated": self._calibration.vscode_window.calibrated,
                "position": (self._calibration.vscode_window.x, self._calibration.vscode_window.y),
            },
            "copilot_input": {
                "calibrated": self._calibration.copilot_input.calibrated,
                "position": (self._calibration.copilot_input.x, self._calibration.copilot_input.y),
            },
            "copilot_response": {
                "calibrated": self._calibration.copilot_response.calibrated,
                "position": (self._calibration.copilot_response.x, self._calibration.copilot_response.y),
            },
        }
