"""
Unit tests for configuration module.
"""

import pytest
import tempfile
from pathlib import Path

from copilot_agent.config import (
    AgentConfig,
    GeminiConfig,
    ReviewerConfig,
    PerceptionConfig,
    AutomationConfig,
    StorageConfig,
    SafetyConfig,
    load_config,
    save_config,
)


class TestGeminiConfig:
    """Tests for GeminiConfig."""
    
    def test_defaults(self):
        config = GeminiConfig()
        assert config.model == "gemini-2.0-flash"
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
    
    def test_custom_values(self):
        config = GeminiConfig(model="gemini-2.5-pro", max_retries=5)
        assert config.model == "gemini-2.5-pro"
        assert config.max_retries == 5


class TestReviewerConfig:
    """Tests for ReviewerConfig."""
    
    def test_defaults(self):
        config = ReviewerConfig()
        assert config.max_iterations == 10
        assert config.timeout_per_review_seconds == 30
        assert config.stop_on_repeated_critiques == 3
        assert config.pause_before_send is True
    
    def test_custom_values(self):
        config = ReviewerConfig(
            max_iterations=5,
            pause_before_send=False,
            stop_on_repeated_critiques=5,
        )
        assert config.max_iterations == 5
        assert config.pause_before_send is False
        assert config.stop_on_repeated_critiques == 5
    
    def test_auto_accept_disabled(self):
        config = ReviewerConfig()
        assert config.auto_accept_after == 0
    
    def test_auto_accept_enabled(self):
        config = ReviewerConfig(auto_accept_after=5)
        assert config.auto_accept_after == 5


class TestPerceptionConfig:
    """Tests for PerceptionConfig."""
    
    def test_defaults(self):
        config = PerceptionConfig()
        assert config.ocr_attempts == 2
        assert config.vision_enabled is True
        assert config.max_vision_per_iteration == 3
        assert config.max_vision_per_session == 20
    
    def test_vision_disabled(self):
        config = PerceptionConfig(vision_enabled=False)
        assert config.vision_enabled is False


class TestAutomationConfig:
    """Tests for AutomationConfig."""
    
    def test_defaults(self):
        config = AutomationConfig()
        assert config.default_mode == "approve"
        assert config.max_iterations == 20
        assert config.max_runtime_minutes == 30
    
    def test_mode_validation(self):
        # Valid modes
        for mode in ["approve", "step", "auto"]:
            config = AutomationConfig(default_mode=mode)
            assert config.default_mode == mode
        
        # Invalid mode should raise
        with pytest.raises(ValueError):
            AutomationConfig(default_mode="invalid")


class TestSafetyConfig:
    """Tests for SafetyConfig."""
    
    def test_defaults(self):
        config = SafetyConfig()
        assert config.kill_switch_hotkey == "ctrl+shift+k"
        assert len(config.allowed_window_patterns) > 0
    
    def test_custom_patterns(self):
        config = SafetyConfig(allowed_window_patterns=["^Test$"])
        assert config.allowed_window_patterns == ["^Test$"]


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_defaults(self):
        config = AgentConfig()
        assert config.gemini.model == "gemini-2.0-flash"
        assert config.automation.default_mode == "approve"
        assert config.reviewer.pause_before_send is True
    
    def test_storage_path(self):
        config = AgentConfig()
        assert config.storage_path.name == ".copilot-agent"
    
    def test_sessions_path(self):
        config = AgentConfig()
        assert config.sessions_path.name == "sessions"
    
    def test_reviewer_config_present(self):
        config = AgentConfig()
        assert hasattr(config, 'reviewer')
        assert config.reviewer.max_iterations == 10


class TestConfigIO:
    """Tests for config loading and saving."""
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            # Create custom config
            config = AgentConfig(
                gemini=GeminiConfig(model="test-model"),
                automation=AutomationConfig(max_iterations=10),
                reviewer=ReviewerConfig(pause_before_send=False),
            )
            
            # Save
            save_config(config, str(config_path))
            assert config_path.exists()
            
            # Load
            loaded = load_config(str(config_path))
            assert loaded.gemini.model == "test-model"
            assert loaded.automation.max_iterations == 10
            assert loaded.reviewer.pause_before_send is False
    
    def test_load_nonexistent_returns_defaults(self):
        config = load_config("/nonexistent/path/config.yaml")
        assert config.gemini.model == "gemini-2.0-flash"
        assert config.reviewer.max_iterations == 10
