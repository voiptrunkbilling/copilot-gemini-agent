"""
Orchestrator - main control loop.

Note: Full implementation in M4.
"""

import asyncio
from typing import Optional

from copilot_agent.config import AgentConfig
from copilot_agent.state import StateManager, SessionPhase
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """
    Main orchestrator that runs the Copilot-Gemini loop.
    
    Note: Full implementation in M4.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        state_manager: StateManager,
        kill_switch: KillSwitch,
        dry_run: bool = False,
    ):
        self.config = config
        self.state_manager = state_manager
        self.kill_switch = kill_switch
        self.dry_run = dry_run
        
        logger.info("Orchestrator initialized", dry_run=dry_run)
    
    async def run(self) -> None:
        """
        Run the main orchestration loop.
        """
        session = self.state_manager.session
        if not session:
            raise RuntimeError("No active session")
        
        logger.info("Starting orchestration", session_id=session.session_id)
        
        try:
            while not self.kill_switch.triggered:
                # Check iteration limit
                if session.iteration_count >= session.max_iterations:
                    logger.info("Max iterations reached")
                    session.completion_reason = "max_iterations"
                    self.state_manager.transition_to(SessionPhase.COMPLETE)
                    break
                
                # Run one iteration
                should_continue = await self._run_iteration()
                
                if not should_continue:
                    break
                
                # Brief pause between iterations
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error("Orchestration error", error=str(e))
            self.state_manager.record_error("orchestration_error", str(e), recoverable=False)
            self.state_manager.transition_to(SessionPhase.FAILED)
        
        finally:
            self.state_manager.checkpoint()
    
    async def _run_iteration(self) -> bool:
        """
        Run a single iteration.
        
        Returns:
            True if should continue, False to stop
        """
        # TODO: Implement full iteration in M4
        logger.warning("Full iteration not implemented yet (M4)")
        
        # For M1, just log and return False to stop
        self.state_manager.transition_to(SessionPhase.COMPLETE)
        return False
