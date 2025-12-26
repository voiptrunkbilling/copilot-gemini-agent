"""
Unit tests for actions module (M1 legacy tests, updated for M2 API).
"""

import pytest

from copilot_agent.actuator.actions import (
    ActionExecutor,
    ActionType,
    ActionResult,
    ActionConfig,
)


class TestActionExecutor:
    """Tests for ActionExecutor."""
    
    @pytest.fixture
    def executor(self):
        return ActionExecutor(dry_run=True)
    
    def test_init(self, executor):
        assert executor.dry_run is True
        # M2: action_delay_ms is now in config
        assert executor.config.action_delay_ms == 50
    
    def test_click_dry_run(self, executor):
        result = executor.click(100, 200)
        
        assert result.success is True
        assert result.action_type == ActionType.CLICK
        assert "100" in result.message
        assert "200" in result.message
    
    def test_click_out_of_bounds(self, executor):
        # M2: Small negative values are allowed (margin of 10)
        # Very negative values should fail
        result = executor.click(-100, 200)
        
        assert result.success is False
        assert "out of bounds" in result.error.lower()
    
    def test_double_click(self, executor):
        result = executor.click(100, 200, clicks=2)
        
        assert result.success is True
        assert result.action_type == ActionType.DOUBLE_CLICK
    
    def test_triple_click(self, executor):
        result = executor.click(100, 200, clicks=3)
        
        assert result.success is True
        assert result.action_type == ActionType.TRIPLE_CLICK
    
    def test_type_text_dry_run(self, executor):
        result = executor.type_text("Hello, World!")
        
        assert result.success is True
        assert result.action_type == ActionType.TYPE_TEXT
    
    def test_type_text_too_long(self, executor):
        long_text = "x" * 20000
        result = executor.type_text(long_text)
        
        assert result.success is False
        assert "too long" in result.error.lower()
    
    def test_hotkey_dry_run(self, executor):
        result = executor.hotkey("ctrl", "c")
        
        assert result.success is True
        assert result.action_type == ActionType.HOTKEY
    
    def test_hotkey_forbidden(self, executor):
        result = executor.hotkey("alt", "f4")
        
        assert result.success is False
        assert "forbidden" in result.error.lower()
    
    def test_hotkey_forbidden_ctrl_w(self, executor):
        result = executor.hotkey("ctrl", "w")
        
        assert result.success is False
    
    def test_copy_dry_run(self, executor):
        # M2: copy() renamed to copy_selection()
        result = executor.copy_selection()
        
        assert result.success is True
        assert result.action_type == ActionType.COPY
    
    def test_paste_dry_run(self, executor):
        # M2: paste() renamed to paste_text()
        result = executor.paste_text()
        
        assert result.success is True
        assert result.action_type == ActionType.PASTE
    
    def test_wait_dry_run(self, executor):
        result = executor.wait(100)
        
        assert result.success is True
        assert result.action_type == ActionType.WAIT
        # M2: dry-run returns 0 duration since no actual wait
        # duration_ms is 0 in dry run mode
    
    def test_wait_too_long(self, executor):
        result = executor.wait(120000)  # 2 minutes
        
        assert result.success is False
        assert "too long" in result.error.lower()
