"""
Configuration schema using Pydantic.
"""

import os
from pathlib import Path
from typing import Optional, List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class GeminiConfig(BaseModel):
    """Gemini API configuration."""
    
    model: str = Field(default="gemini-2.0-flash", description="Gemini model ID")
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    api_key_env: str = Field(default="GEMINI_API_KEY", description="Environment variable for API key")


class ReviewerConfig(BaseModel):
    """Reviewer loop configuration."""
    
    # Iteration control
    max_iterations: int = Field(default=10, ge=1, le=50, description="Max review iterations")
    timeout_per_review_seconds: int = Field(default=30, ge=5, le=120)
    
    # Stop conditions
    stop_on_repeated_critiques: int = Field(default=3, ge=2, le=10, description="Stop after N identical critiques")
    stop_on_low_confidence: bool = Field(default=False, description="Stop if reviewer has low confidence")
    
    # Safety mode
    pause_before_send: bool = Field(default=True, description="Pause for user approval before sending feedback")
    auto_accept_after: int = Field(default=0, ge=0, le=50, description="Auto-accept after N successful iterations (0=disabled)")
    
    # Response detection
    response_wait_seconds: int = Field(default=30, ge=5, le=120, description="Wait for Copilot response")
    response_stability_ms: int = Field(default=2000, ge=500, le=10000, description="Wait for response to stabilize")


class PerceptionConfig(BaseModel):
    """Perception engine configuration."""
    
    ocr_attempts: int = Field(default=2, ge=1, le=5)
    template_attempts: int = Field(default=2, ge=1, le=5)
    vision_enabled: bool = Field(default=True)
    max_vision_per_iteration: int = Field(default=3, ge=0, le=10)
    max_vision_per_session: int = Field(default=20, ge=0, le=100)
    vision_cooldown_seconds: int = Field(default=5, ge=1, le=60)


class AutomationConfig(BaseModel):
    """Automation behavior configuration."""
    
    default_mode: str = Field(default="approve", pattern="^(approve|step|auto)$")
    max_iterations: int = Field(default=20, ge=1, le=100)
    max_runtime_minutes: int = Field(default=30, ge=1, le=180)
    action_delay_ms: int = Field(default=50, ge=0, le=1000)
    typing_delay_ms: int = Field(default=20, ge=0, le=200)
    response_timeout_seconds: int = Field(default=120, ge=30, le=300)
    response_stability_ms: int = Field(default=2000, ge=500, le=10000)


class StorageConfig(BaseModel):
    """Storage and persistence configuration."""
    
    base_path: str = Field(default="~/.copilot-agent")
    max_total_mb: int = Field(default=500, ge=50, le=5000)
    max_screenshots_per_session: int = Field(default=100, ge=10, le=1000)
    retention_days: int = Field(default=7, ge=1, le=90)


class SafetyConfig(BaseModel):
    """Safety guardrails configuration."""
    
    kill_switch_hotkey: str = Field(default="ctrl+shift+k")
    pause_hotkey: str = Field(default="ctrl+shift+p")
    allowed_window_patterns: List[str] = Field(
        default=[
            "^Visual Studio Code$",
            "^Code -",
            ".+ - Visual Studio Code$",
        ]
    )
    max_actions_per_second: int = Field(default=10, ge=1, le=50)
    require_focus_verification: bool = Field(default=True)


class AgentConfig(BaseModel):
    """Root configuration for Copilot-Gemini Agent."""
    
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    reviewer: ReviewerConfig = Field(default_factory=ReviewerConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    
    @property
    def storage_path(self) -> Path:
        """Get resolved storage path."""
        return Path(self.storage.base_path).expanduser()
    
    @property
    def sessions_path(self) -> Path:
        """Get sessions directory path."""
        return self.storage_path / "sessions"


def get_default_config_path() -> Path:
    """Get default config file path."""
    return Path.home() / ".copilot-agent" / "config.yaml"


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    """
    Load configuration from YAML file.
    
    Falls back to defaults if file doesn't exist.
    """
    if config_path:
        path = Path(config_path)
    else:
        path = get_default_config_path()
    
    if path.exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return AgentConfig(**data)
    
    return AgentConfig()


def save_config(config: AgentConfig, config_path: Optional[str] = None) -> None:
    """Save configuration to YAML file."""
    if config_path:
        path = Path(config_path)
    else:
        path = get_default_config_path()
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
