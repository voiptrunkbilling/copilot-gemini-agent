"""
Safety module - kill switch and guardrails.
"""
from copilot_agent.safety.killswitch import (
    KillSwitch,
    KillSwitchTriggered,
    MockKillSwitch,
    HotkeyConfig,
    wait_with_killswitch,
    async_wait_with_killswitch,
)

__all__ = [
    "KillSwitch",
    "KillSwitchTriggered",
    "MockKillSwitch",
    "HotkeyConfig",
    "wait_with_killswitch",
    "async_wait_with_killswitch",
]