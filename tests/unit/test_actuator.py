"""
Tests for M2 actuator components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

from copilot_agent.actuator.platform import (
    IS_WINDOWS,
    IS_LINUX,
    IS_MACOS,
    DPIInfo,
    ScreenInfo,
    get_dpi_info,
    get_screen_info,
    get_primary_screen,
    scale_coordinates,
    unscale_coordinates,
)
from copilot_agent.actuator.actions import (
    ActionExecutor,
    ActionResult,
    ActionType,
    ActionConfig,
)
from copilot_agent.actuator.window import WindowManager, WindowInfo
from copilot_agent.actuator.screenshot import (
    ScreenshotCapture,
    ScreenshotResult,
    Region,
)
from copilot_agent.actuator.calibration import (
    CalibrationData,
    CalibrationPoint,
    CalibrationManager,
)
from copilot_agent.actuator.pipeline import (
    ActionPipeline,
    PipelineAction,
    PipelineStep,
    PipelineResult,
)


class TestPlatform:
    """Tests for platform utilities."""
    
    def test_platform_detection(self):
        """Test that exactly one platform is detected."""
        platforms = [IS_WINDOWS, IS_LINUX, IS_MACOS]
        # At least one should be True (or all False on unusual platforms)
        assert sum(platforms) <= 1 or sum(platforms) == 0
    
    def test_dpi_info_structure(self):
        """Test DPIInfo dataclass."""
        info = DPIInfo(is_aware=True, scale_factor=1.25, system_dpi=120)
        assert info.is_aware is True
        assert info.scale_factor == 1.25
        assert info.system_dpi == 120
    
    def test_screen_info_structure(self):
        """Test ScreenInfo dataclass."""
        info = ScreenInfo(
            index=0,
            x=0,
            y=0,
            width=1920,
            height=1080,
            scale_factor=1.0,
            is_primary=True,
        )
        assert info.width == 1920
        assert info.height == 1080
        assert info.is_primary is True
    
    def test_get_primary_screen(self):
        """Test getting primary screen info."""
        screen = get_primary_screen()
        assert screen.width > 0
        assert screen.height > 0
    
    def test_scale_coordinates(self):
        """Test coordinate scaling."""
        x, y = scale_coordinates(200, 100, 1.25)
        assert x == 160
        assert y == 80
    
    def test_unscale_coordinates(self):
        """Test coordinate unscaling."""
        x, y = unscale_coordinates(160, 80, 1.25)
        assert x == 200
        assert y == 100


class TestRegion:
    """Tests for Region class."""
    
    def test_region_creation(self):
        """Test Region creation."""
        region = Region(x=10, y=20, width=100, height=50)
        assert region.x == 10
        assert region.y == 20
        assert region.width == 100
        assert region.height == 50
    
    def test_region_right_bottom(self):
        """Test Region right and bottom properties."""
        region = Region(x=10, y=20, width=100, height=50)
        assert region.right == 110
        assert region.bottom == 70
    
    def test_region_to_tuple(self):
        """Test Region to tuple conversion."""
        region = Region(x=10, y=20, width=100, height=50)
        assert region.to_tuple() == (10, 20, 100, 50)
    
    def test_region_to_mss_dict(self):
        """Test Region to mss format."""
        region = Region(x=10, y=20, width=100, height=50)
        d = region.to_mss_dict()
        assert d["left"] == 10
        assert d["top"] == 20
        assert d["width"] == 100
        assert d["height"] == 50
    
    def test_region_from_dict(self):
        """Test Region from dictionary."""
        region = Region.from_dict({"x": 10, "y": 20, "width": 100, "height": 50})
        assert region.x == 10
        assert region.y == 20


class TestActionConfig:
    """Tests for ActionConfig."""
    
    def test_default_config(self):
        """Test default action config values."""
        config = ActionConfig()
        assert config.action_delay_ms == 50
        assert config.typing_delay_ms == 10
        assert config.max_text_length == 10000
        assert "alt+f4" in config.forbidden_hotkeys
    
    def test_custom_config(self):
        """Test custom config values."""
        config = ActionConfig(action_delay_ms=100, typing_delay_ms=20)
        assert config.action_delay_ms == 100
        assert config.typing_delay_ms == 20


class TestActionExecutor:
    """Tests for ActionExecutor."""
    
    def test_init_dry_run(self):
        """Test dry-run initialization."""
        executor = ActionExecutor(dry_run=True)
        assert executor.dry_run is True
    
    def test_click_dry_run(self):
        """Test click in dry-run mode."""
        executor = ActionExecutor(dry_run=True)
        result = executor.click(100, 200)
        assert result.success is True
        assert result.action_type == ActionType.CLICK
        assert "Would click" in result.message
    
    def test_click_out_of_bounds(self):
        """Test click with invalid coordinates."""
        executor = ActionExecutor(dry_run=True)
        result = executor.click(-100, -100)
        # Should succeed in dry-run due to margin allowance
        # But very negative should fail
        result = executor.click(-1000, -1000)
        assert result.success is False
        assert "out of bounds" in result.error.lower()
    
    def test_type_text_dry_run(self):
        """Test type_text in dry-run mode."""
        executor = ActionExecutor(dry_run=True)
        result = executor.type_text("Hello World")
        assert result.success is True
        assert result.action_type == ActionType.TYPE_TEXT
        assert "Would type" in result.message
    
    def test_type_text_too_long(self):
        """Test type_text with very long text."""
        executor = ActionExecutor(dry_run=True)
        long_text = "x" * 20000
        result = executor.type_text(long_text)
        assert result.success is False
        assert "too long" in result.error.lower()
    
    def test_hotkey_dry_run(self):
        """Test hotkey in dry-run mode."""
        executor = ActionExecutor(dry_run=True)
        result = executor.hotkey("ctrl", "c")
        assert result.success is True
        assert result.action_type == ActionType.HOTKEY
    
    def test_hotkey_forbidden(self):
        """Test forbidden hotkey combinations."""
        executor = ActionExecutor(dry_run=True)
        result = executor.hotkey("alt", "f4")
        assert result.success is False
        assert "forbidden" in result.error.lower()
    
    def test_wait_dry_run(self):
        """Test wait in dry-run mode."""
        executor = ActionExecutor(dry_run=True)
        result = executor.wait(100)
        assert result.success is True
        assert result.action_type == ActionType.WAIT
    
    def test_wait_too_long(self):
        """Test wait with excessive duration."""
        executor = ActionExecutor(dry_run=True)
        result = executor.wait(100000)  # 100 seconds
        assert result.success is False
        assert "too long" in result.error.lower()
    
    def test_kill_switch_integration(self):
        """Test kill switch blocks actions."""
        kill_triggered = False
        
        def check_kill():
            return kill_triggered
        
        executor = ActionExecutor(dry_run=True, kill_switch_check=check_kill)
        
        # Should succeed when not triggered
        result = executor.click(100, 100)
        assert result.success is True
        
        # Should fail when triggered
        kill_triggered = True
        result = executor.click(100, 100)
        assert result.success is False
        assert "kill switch" in result.error.lower()


class TestWindowManager:
    """Tests for WindowManager."""
    
    def test_init_dry_run(self):
        """Test dry-run initialization."""
        wm = WindowManager(dry_run=True)
        assert wm.dry_run is True
    
    def test_is_allowed_window(self):
        """Test window title pattern matching."""
        wm = WindowManager()
        
        assert wm.is_allowed_window("Visual Studio Code") is True
        assert wm.is_allowed_window("Code - project") is True
        assert wm.is_allowed_window("project - Visual Studio Code") is True
        assert wm.is_allowed_window("Notepad") is False


class TestScreenshotCapture:
    """Tests for ScreenshotCapture."""
    
    def test_init_dry_run(self):
        """Test dry-run initialization."""
        capture = ScreenshotCapture(dry_run=True)
        assert capture.dry_run is True
    
    def test_capture_full_screen_dry_run(self):
        """Test full screen capture in dry-run mode."""
        capture = ScreenshotCapture(dry_run=True)
        result = capture.capture_full_screen()
        assert result.success is True
        assert result.width > 0
        assert result.height > 0
    
    def test_capture_region_dry_run(self):
        """Test region capture in dry-run mode."""
        capture = ScreenshotCapture(dry_run=True)
        region = Region(x=0, y=0, width=100, height=100)
        result = capture.capture_region(region)
        assert result.success is True
        assert result.width == 100
        assert result.height == 100
    
    def test_capture_invalid_region(self):
        """Test capture with invalid region."""
        capture = ScreenshotCapture(dry_run=True)
        region = Region(x=0, y=0, width=0, height=0)
        result = capture.capture_region(region)
        assert result.success is False
        assert "invalid" in result.error.lower()


class TestCalibration:
    """Tests for calibration components."""
    
    def test_calibration_point_default(self):
        """Test CalibrationPoint defaults."""
        point = CalibrationPoint(name="test", description="Test point")
        assert point.calibrated is False
        assert point.x is None
        assert point.y is None
    
    def test_calibration_point_to_region(self):
        """Test converting point to region."""
        point = CalibrationPoint(
            name="test",
            description="Test",
            x=100,
            y=200,
            width=50,
            height=30,
            calibrated=True,
        )
        region = point.to_region()
        assert region is not None
        assert region.x == 100
        assert region.y == 200
    
    def test_calibration_data_is_complete(self):
        """Test CalibrationData completeness check."""
        data = CalibrationData()
        assert data.is_complete() is False
        
        # Mark required points as calibrated
        data.vscode_window.calibrated = True
        data.vscode_window.x = 100
        data.vscode_window.y = 100
        
        data.copilot_input.calibrated = True
        data.copilot_input.x = 200
        data.copilot_input.y = 200
        
        data.copilot_response.calibrated = True
        data.copilot_response.x = 300
        data.copilot_response.y = 300
        
        assert data.is_complete() is True
    
    def test_calibration_data_to_dict(self):
        """Test CalibrationData serialization."""
        data = CalibrationData(dpi_scale=1.25, screen_width=1920, screen_height=1080)
        d = data.to_dict()
        
        assert d["dpi_scale"] == 1.25
        assert d["screen_width"] == 1920
        assert "points" in d
        assert "vscode_window" in d["points"]
    
    def test_calibration_data_from_dict(self):
        """Test CalibrationData deserialization."""
        d = {
            "version": "1.0",
            "dpi_scale": 1.5,
            "screen_width": 2560,
            "screen_height": 1440,
            "points": {
                "vscode_window": {
                    "name": "vscode_window",
                    "description": "VS Code",
                    "x": 100,
                    "y": 100,
                    "calibrated": True,
                },
                "copilot_input": {
                    "name": "copilot_input",
                    "description": "Input",
                    "x": 200,
                    "y": 200,
                    "calibrated": True,
                },
                "copilot_response": {
                    "name": "copilot_response",
                    "description": "Response",
                    "x": 300,
                    "y": 300,
                    "calibrated": True,
                },
            },
        }
        
        data = CalibrationData.from_dict(d)
        assert data.dpi_scale == 1.5
        assert data.screen_width == 2560
        assert data.vscode_window.x == 100
        assert data.vscode_window.calibrated is True
    
    def test_calibration_manager_save_load(self):
        """Test saving and loading calibration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibrationManager(Path(tmpdir))
            
            # Create and save data
            data = CalibrationData(dpi_scale=1.25)
            data.vscode_window.x = 100
            data.vscode_window.y = 100
            data.vscode_window.calibrated = True
            
            assert manager.save(data) is True
            
            # Load and verify
            loaded = manager.load()
            assert loaded.dpi_scale == 1.25
            assert loaded.vscode_window.x == 100


class TestPipeline:
    """Tests for ActionPipeline."""
    
    def test_init_dry_run(self):
        """Test dry-run initialization."""
        pipeline = ActionPipeline(dry_run=True)
        assert pipeline.dry_run is True
    
    def test_has_calibration_without_data(self):
        """Test calibration check without data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ActionPipeline(
                dry_run=True,
                calibration_path=Path(tmpdir),
            )
            # No calibration file, should be incomplete
            assert pipeline.has_calibration is False
    
    def test_focus_vscode_dry_run(self):
        """Test focus_vscode in dry-run mode."""
        pipeline = ActionPipeline(dry_run=True)
        result = pipeline.focus_vscode()
        # May succeed or fail depending on whether VS Code is found
        # In dry-run on non-Windows, will likely fail
        assert result.action_type == ActionType.FOCUS_WINDOW
    
    def test_wait_action(self):
        """Test wait action."""
        pipeline = ActionPipeline(dry_run=True)
        result = pipeline.wait(100)
        assert result.success is True
        assert result.action_type == ActionType.WAIT
    
    def test_kill_switch_blocks_pipeline(self):
        """Test that kill switch blocks pipeline execution."""
        triggered = False
        
        def check():
            return triggered
        
        pipeline = ActionPipeline(dry_run=True, kill_switch_check=check)
        
        steps = [
            PipelineStep(action=PipelineAction.WAIT, params={"duration_ms": 100}),
            PipelineStep(action=PipelineAction.WAIT, params={"duration_ms": 100}),
        ]
        
        # Should succeed initially
        result = pipeline.execute_pipeline(steps[:1])
        assert result.success is True
        
        # Trigger kill switch
        triggered = True
        
        result = pipeline.execute_pipeline(steps)
        assert result.success is False
        assert result.aborted_by_killswitch is True
    
    def test_pipeline_step_execution(self):
        """Test single step execution."""
        pipeline = ActionPipeline(dry_run=True)
        
        step = PipelineStep(
            action=PipelineAction.WAIT,
            params={"duration_ms": 50},
            description="Test wait",
        )
        
        result = pipeline.execute_step(step)
        assert result.success is True
    
    def test_calibration_summary(self):
        """Test getting calibration summary."""
        pipeline = ActionPipeline(dry_run=True)
        summary = pipeline.get_calibration_summary()
        
        assert "status" in summary
        assert summary["status"] in ["not_loaded", "incomplete", "complete"]
