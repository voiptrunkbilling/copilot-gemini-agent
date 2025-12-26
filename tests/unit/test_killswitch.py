"""
Unit tests for kill switch module.
"""

import pytest
import threading
import time

from copilot_agent.safety.killswitch import (
    KillSwitch,
    HotkeyConfig,
    MockKillSwitch,
)


class TestHotkeyConfig:
    """Tests for HotkeyConfig parsing."""
    
    def test_parse_simple(self):
        config = HotkeyConfig.parse("ctrl+k")
        assert config.modifiers == {"ctrl"}
        assert config.key == "k"
    
    def test_parse_multiple_modifiers(self):
        config = HotkeyConfig.parse("ctrl+shift+k")
        assert config.modifiers == {"ctrl", "shift"}
        assert config.key == "k"
    
    def test_parse_case_insensitive(self):
        config = HotkeyConfig.parse("CTRL+SHIFT+K")
        assert config.modifiers == {"ctrl", "shift"}
        assert config.key == "k"
    
    def test_parse_three_modifiers(self):
        config = HotkeyConfig.parse("ctrl+alt+shift+k")
        assert config.modifiers == {"ctrl", "alt", "shift"}
        assert config.key == "k"


class TestKillSwitch:
    """Tests for KillSwitch."""
    
    def test_initial_state(self):
        ks = KillSwitch()
        assert not ks.triggered
    
    def test_manual_trigger(self):
        triggered = []
        
        def callback():
            triggered.append(True)
        
        ks = KillSwitch(on_trigger=callback)
        ks.trigger()
        
        assert ks.triggered
        assert len(triggered) == 1
    
    def test_trigger_only_once(self):
        triggered = []
        
        def callback():
            triggered.append(True)
        
        ks = KillSwitch(on_trigger=callback)
        ks.trigger()
        ks.trigger()
        ks.trigger()
        
        assert len(triggered) == 1  # Only triggered once
    
    def test_reset(self):
        ks = KillSwitch()
        ks.trigger()
        assert ks.triggered
        
        ks.reset()
        assert not ks.triggered
    
    def test_hotkey_parsing(self):
        ks = KillSwitch(hotkey="ctrl+alt+x")
        assert ks.hotkey_config.modifiers == {"ctrl", "alt"}
        assert ks.hotkey_config.key == "x"


class TestMockKillSwitch:
    """Tests for MockKillSwitch."""
    
    def test_start_stop_no_op(self):
        ks = MockKillSwitch()
        ks.start()  # Should not raise
        ks.stop()   # Should not raise
    
    def test_trigger_works(self):
        ks = MockKillSwitch()
        assert not ks.triggered
        ks.trigger()
        assert ks.triggered
