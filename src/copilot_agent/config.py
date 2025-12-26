"""
Configuration schema using Pydantic.

Secrets are loaded exclusively from environment variables for security.
Configuration can be loaded from YAML files or environment variables.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    
    def __init__(self, message: str, field: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.message = message
        self.field = field
        self.suggestions = suggestions or []
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        lines = [self.message]
        if self.field:
            lines.append(f"  Field: {self.field}")
        if self.suggestions:
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")
        return "\n".join(lines)


class SecretsManager:
    """
    Manages secrets loaded from environment variables only.
    
    Secrets are NEVER stored in config files for security.
    """
    
    # Required secrets (must be set for agent to function)
    REQUIRED_SECRETS = {
        "GEMINI_API_KEY": "Gemini API key for vision and review",
    }
    
    # Optional secrets
    OPTIONAL_SECRETS = {
        "ANTHROPIC_API_KEY": "Anthropic API key (optional fallback)",
        "OPENAI_API_KEY": "OpenAI API key (optional fallback)",
    }
    
    @classmethod
    def get_secret(cls, key: str, required: bool = False) -> Optional[str]:
        """
        Get a secret from environment variables.
        
        Args:
            key: Environment variable name
            required: If True, raise error when missing
            
        Returns:
            Secret value or None
            
        Raises:
            ConfigurationError: If required secret is missing
        """
        value = os.environ.get(key)
        
        if required and not value:
            suggestions = []
            
            # Check if .env file exists
            env_file = Path.cwd() / ".env"
            env_example = Path.cwd() / ".env.example"
            
            if env_example.exists() and not env_file.exists():
                suggestions.append("Copy .env.example to .env and fill in your values")
            elif not env_file.exists():
                suggestions.append(f"Create a .env file with: {key}=your-key-here")
            else:
                suggestions.append(f"Add {key}=your-key-here to your .env file")
            
            if key == "GEMINI_API_KEY":
                suggestions.append("Get your API key from: https://aistudio.google.com/apikey")
            
            suggestions.append(f"Or set environment variable: set {key}=your-key-here")
            
            raise ConfigurationError(
                f"Required secret '{key}' is not set",
                field=key,
                suggestions=suggestions
            )
        
        return value
    
    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get Gemini API key (required)."""
        return cls.get_secret("GEMINI_API_KEY", required=True)
    
    @classmethod
    def validate_all_required(cls) -> Tuple[bool, List[str]]:
        """
        Validate all required secrets are present.
        
        Returns:
            Tuple of (all_valid, list of missing keys)
        """
        missing = []
        for key in cls.REQUIRED_SECRETS:
            if not os.environ.get(key):
                missing.append(key)
        return len(missing) == 0, missing
    
    @classmethod
    def get_status(cls) -> Dict[str, str]:
        """Get status of all secrets (masked)."""
        status = {}
        for key in cls.REQUIRED_SECRETS:
            value = os.environ.get(key)
            if value:
                # Mask the secret
                status[key] = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
            else:
                status[key] = "NOT SET (required)"
        
        for key in cls.OPTIONAL_SECRETS:
            value = os.environ.get(key)
            if value:
                status[key] = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
            else:
                status[key] = "not set"
        
        return status


class GeminiConfig(BaseModel):
    """Gemini API configuration (used for vision/perception)."""
    
    model: str = Field(default="gemini-2.5-flash", description="Gemini model ID for vision")
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    api_key_env: str = Field(default="GEMINI_API_KEY", description="Environment variable for API key")


class ReviewerConfig(BaseModel):
    """Reviewer loop configuration."""
    
    # Model selection
    model: str = Field(default="gemma-3-27b-it", description="Reviewer model ID")
    fallback_model: str = Field(default="gemini-2.5-flash", description="Fallback model if primary unavailable")
    
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
    
    # Quota guardrails
    max_reviews_per_session: int = Field(default=50, ge=5, le=200, description="Max API calls per session")
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=5.0, description="Exponential backoff multiplier on 429")
    initial_backoff_seconds: float = Field(default=1.0, ge=0.5, le=10.0, description="Initial retry delay")
    max_backoff_seconds: float = Field(default=60.0, ge=10.0, le=300.0, description="Maximum retry delay")
    pause_on_quota_exhausted: bool = Field(default=True, description="Pause session when quota exceeded")


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
    
    @model_validator(mode='after')
    def validate_consistency(self) -> 'AgentConfig':
        """Validate cross-field consistency."""
        # Ensure perception vision limit doesn't exceed review limit
        if self.perception.max_vision_per_session > self.reviewer.max_reviews_per_session:
            # Auto-correct rather than error
            pass
        
        # Ensure automation timeout is reasonable
        if self.automation.response_timeout_seconds < self.reviewer.timeout_per_review_seconds:
            raise ValueError(
                f"automation.response_timeout_seconds ({self.automation.response_timeout_seconds}) "
                f"must be >= reviewer.timeout_per_review_seconds ({self.reviewer.timeout_per_review_seconds})"
            )
        
        return self
    
    def validate_for_run(self) -> List[str]:
        """
        Validate configuration is ready for a run.
        
        Returns list of warning messages (empty if all good).
        """
        warnings = []
        
        # Check secrets
        valid, missing = SecretsManager.validate_all_required()
        if not valid:
            for key in missing:
                warnings.append(f"Required secret not set: {key}")
        
        # Check storage path is writable
        try:
            storage = self.storage_path
            storage.mkdir(parents=True, exist_ok=True)
            test_file = storage / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            warnings.append(f"Storage path not writable: {storage} ({e})")
        
        # Check calibration exists
        cal_file = self.storage_path / "calibration" / "calibration.json"
        if not cal_file.exists():
            # Also check legacy location
            legacy_cal = self.storage_path / "calibration.json"
            if not legacy_cal.exists():
                warnings.append("Calibration not found. Run 'agent calibrate' first.")
        
        return warnings


def get_default_config_path() -> Path:
    """Get default config file path."""
    return Path.home() / ".copilot-agent" / "config.yaml"


def load_config(config_path: Optional[str] = None, validate: bool = True) -> AgentConfig:
    """
    Load configuration from YAML file.
    
    Falls back to defaults if file doesn't exist.
    Environment variables can override config file values.
    
    Args:
        config_path: Path to config file (optional)
        validate: If True, validate config after loading
        
    Returns:
        Loaded configuration
        
    Raises:
        ConfigurationError: If config file is invalid
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(
                f"Config file not found: {path}",
                suggestions=[
                    f"Create the config file at {path}",
                    "Use 'agent init' to create a default config",
                    "Or run without --config to use defaults"
                ]
            )
    else:
        path = get_default_config_path()
    
    data = {}
    
    if path.exists():
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in config file: {path}",
                suggestions=[
                    f"Check syntax at line {e.problem_mark.line if hasattr(e, 'problem_mark') else 'unknown'}",
                    "Use a YAML validator to check your config",
                    "See config.example.yaml for reference"
                ]
            )
    
    # Apply environment variable overrides
    env_overrides = _get_env_overrides()
    data = _deep_merge(data, env_overrides)
    
    try:
        config = AgentConfig(**data)
    except Exception as e:
        raise ConfigurationError(
            f"Invalid configuration: {e}",
            suggestions=[
                "Check field names and values in your config",
                "See config.example.yaml for reference",
                "Run 'agent config --validate' to check"
            ]
        )
    
    return config


def _get_env_overrides() -> Dict[str, Any]:
    """Get configuration overrides from environment variables."""
    overrides: Dict[str, Any] = {}
    
    # Map env vars to config paths
    env_mappings = {
        "COPILOT_MODEL": ("reviewer", "model"),
        "GEMINI_MODEL": ("gemini", "model"),
        "COPILOT_MAX_ITERATIONS": ("automation", "max_iterations"),
        "COPILOT_MAX_RUNTIME": ("automation", "max_runtime_minutes"),
        "COPILOT_STORAGE_PATH": ("storage", "base_path"),
        "COPILOT_KILLSWITCH_KEY": ("safety", "kill_switch_hotkey"),
    }
    
    for env_key, (section, field) in env_mappings.items():
        value = os.environ.get(env_key)
        if value:
            if section not in overrides:
                overrides[section] = {}
            # Try to convert to int if it looks like a number
            if value.isdigit():
                value = int(value)
            overrides[section][field] = value
    
    return overrides


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: AgentConfig, config_path: Optional[str] = None) -> None:
    """Save configuration to YAML file."""
    if config_path:
        path = Path(config_path)
    else:
        path = get_default_config_path()
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
