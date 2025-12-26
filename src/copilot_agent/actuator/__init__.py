"""
Actuator module - GUI actions (mouse, keyboard, clipboard).

M2 Components:
- platform: DPI awareness, screen info
- window: Window focus and management (pywin32)
- actions: GUI action executor (pyautogui)
- screenshot: Fast screenshot capture (mss)
- calibration: Manual UI element calibration
- pipeline: Unified action pipeline
"""

from copilot_agent.actuator.platform import (
    IS_WINDOWS,
    IS_LINUX,
    IS_MACOS,
    ScreenInfo,
    DPIInfo,
    set_dpi_awareness,
    get_dpi_info,
    get_screen_info,
    get_primary_screen,
)
from copilot_agent.actuator.window import WindowManager, WindowInfo
from copilot_agent.actuator.actions import (
    ActionExecutor,
    ActionResult,
    ActionType,
    ActionConfig,
)
from copilot_agent.actuator.screenshot import (
    ScreenshotCapture,
    ScreenshotResult,
    Region,
)
from copilot_agent.actuator.calibration import (
    CalibrationManager,
    CalibrationData,
    CalibrationPoint,
    CalibrationOverlay,
    run_calibration,
    run_calibration_cli,
)
from copilot_agent.actuator.pipeline import (
    ActionPipeline,
    PipelineAction,
    PipelineStep,
    PipelineResult,
)

__all__ = [
    # Platform
    "IS_WINDOWS",
    "IS_LINUX", 
    "IS_MACOS",
    "ScreenInfo",
    "DPIInfo",
    "set_dpi_awareness",
    "get_dpi_info",
    "get_screen_info",
    "get_primary_screen",
    # Window
    "WindowManager",
    "WindowInfo",
    # Actions
    "ActionExecutor",
    "ActionResult",
    "ActionType",
    "ActionConfig",
    # Screenshot
    "ScreenshotCapture",
    "ScreenshotResult",
    "Region",
    # Calibration
    "CalibrationManager",
    "CalibrationData",
    "CalibrationPoint",
    "CalibrationOverlay",
    "run_calibration",
    "run_calibration_cli",
    # Pipeline
    "ActionPipeline",
    "PipelineAction",
    "PipelineStep",
    "PipelineResult",
]
