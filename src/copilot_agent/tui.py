"""
Terminal User Interface using Rich.
"""

import asyncio
from datetime import datetime
from typing import Optional, Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from copilot_agent.state import StateManager, SessionPhase
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.logging import get_logger

logger = get_logger(__name__)


class AgentTUI:
    """Terminal User Interface for the agent."""
    
    def __init__(
        self,
        state_manager: StateManager,
        kill_switch: KillSwitch,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.state_manager = state_manager
        self.kill_switch = kill_switch
        self.dry_run = dry_run
        self.verbose = verbose
        self.console = Console()
        self._running = False
        self._paused = False
    
    def make_header(self) -> Panel:
        """Create header panel."""
        session = self.state_manager.session
        
        if self.kill_switch.triggered:
            status = "[bold red]KILLED[/bold red] ⛔"
        elif self._paused:
            status = "[bold yellow]PAUSED[/bold yellow] ⏸️"
        elif session and session.phase == SessionPhase.COMPLETE:
            status = "[bold green]COMPLETE[/bold green] ✅"
        elif session and session.phase == SessionPhase.FAILED:
            status = "[bold red]FAILED[/bold red] ❌"
        elif session and session.phase == SessionPhase.ABORTED:
            status = "[bold red]ABORTED[/bold red] 🛑"
        elif self._running:
            status = "[bold green]RUNNING[/bold green] 🟢"
        else:
            status = "[dim]IDLE[/dim] ⚪"
        
        mode = "DRY RUN" if self.dry_run else "LIVE"
        mode_color = "yellow" if self.dry_run else "green"
        
        title = Text()
        title.append("COPILOT-GEMINI AGENT", style="bold white")
        title.append(f"  [{mode_color}]{mode}[/{mode_color}]")
        title.append(f"  {status}")
        
        return Panel(title, style="blue")
    
    def make_status_table(self) -> Table:
        """Create status information table."""
        session = self.state_manager.session
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value")
        
        if session:
            # Calculate runtime
            if session.started_at:
                start = datetime.fromisoformat(session.started_at.replace("Z", "+00:00"))
                runtime = datetime.now(start.tzinfo) - start
                runtime_str = str(runtime).split(".")[0]  # Remove microseconds
            else:
                runtime_str = "00:00:00"
            
            table.add_row("Session", session.session_id)
            table.add_row("Iteration", f"{session.iteration_count}/{session.max_iterations}")
            table.add_row("Runtime", runtime_str)
            table.add_row("Phase", session.phase.value.upper())
            table.add_row("Errors", f"{session.total_errors} (consecutive: {session.consecutive_errors})")
        else:
            table.add_row("Session", "[dim]None[/dim]")
        
        return table
    
    def make_task_panel(self) -> Panel:
        """Create task description panel."""
        session = self.state_manager.session
        
        if session:
            task = session.task_description
            if len(task) > 200:
                task = task[:200] + "..."
            content = Text(task)
        else:
            content = Text("[No task]", style="dim")
        
        return Panel(content, title="Task", border_style="blue")
    
    def make_current_panel(self) -> Panel:
        """Create current iteration panel."""
        session = self.state_manager.session
        
        if not session:
            return Panel("[dim]No active session[/dim]", title="Current", border_style="dim")
        
        lines = []
        
        # Current prompt
        if session.current_prompt:
            prompt = session.current_prompt
            if len(prompt) > 100:
                prompt = prompt[:100] + "..."
            lines.append(f"[bold]Prompt:[/bold] {prompt}")
        
        # Current response
        if session.current_response:
            response = session.current_response
            if len(response) > 200:
                response = response[:200] + "..."
            lines.append(f"\n[bold]Response:[/bold]\n{response}")
        elif session.phase == SessionPhase.WAITING:
            lines.append("\n[yellow]⏳ Waiting for Copilot response...[/yellow]")
        elif session.phase == SessionPhase.CAPTURING:
            lines.append("\n[yellow]📋 Capturing response...[/yellow]")
        
        # Verdict
        if session.current_verdict:
            verdict = session.current_verdict.value.upper()
            if verdict == "ACCEPT":
                verdict_style = "bold green"
            elif verdict == "CRITIQUE":
                verdict_style = "bold yellow"
            else:
                verdict_style = "bold red"
            lines.append(f"\n[bold]Verdict:[/bold] [{verdict_style}]{verdict}[/{verdict_style}]")
            
            if session.current_feedback:
                feedback = session.current_feedback
                if len(feedback) > 150:
                    feedback = feedback[:150] + "..."
                lines.append(f"[dim]{feedback}[/dim]")
        
        content = "\n".join(lines) if lines else "[dim]Iteration not started[/dim]"
        
        return Panel(content, title="Current Iteration", border_style="green")
    
    def make_controls_panel(self) -> Panel:
        """Create keyboard controls panel."""
        controls = [
            "[P]ause",
            "[R]esume",
            "[A]bort",
            "[S]tep",
            "[V]ision",
            "[C]ontinue",
            "[D]etails",
            "[Q]uit",
        ]
        
        return Panel(
            "  ".join(controls),
            title="Controls",
            border_style="dim",
        )
    
    def make_layout(self) -> Layout:
        """Create the full TUI layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="controls", size=3),
        )
        
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )
        
        layout["left"].split_column(
            Layout(name="status"),
            Layout(name="task"),
        )
        
        # Populate layout
        layout["header"].update(self.make_header())
        layout["status"].update(Panel(self.make_status_table(), title="Status", border_style="blue"))
        layout["task"].update(self.make_task_panel())
        layout["right"].update(self.make_current_panel())
        layout["controls"].update(self.make_controls_panel())
        
        return layout
    
    def run(self) -> None:
        """Run the TUI (blocking)."""
        self._running = True
        
        logger.info("TUI started")
        
        try:
            with Live(
                self.make_layout(),
                console=self.console,
                refresh_per_second=2,
                screen=True,
            ) as live:
                while self._running:
                    # Check for kill switch
                    if self.kill_switch.triggered:
                        self._running = False
                        break
                    
                    # Update display
                    live.update(self.make_layout())
                    
                    # Simple key handling (non-blocking would require more complexity)
                    # For M1, we'll use a simple loop with timeout
                    import time
                    time.sleep(0.5)
                    
                    # Check if session is in terminal state
                    session = self.state_manager.session
                    if session and session.phase in (
                        SessionPhase.COMPLETE,
                        SessionPhase.FAILED,
                        SessionPhase.ABORTED,
                    ):
                        # Show final state for a moment, then exit
                        time.sleep(2)
                        self._running = False
        
        except KeyboardInterrupt:
            self._running = False
            logger.info("TUI interrupted")
        
        finally:
            self._running = False
            logger.info("TUI stopped")
    
    def pause(self) -> None:
        """Pause automation."""
        self._paused = True
        if self.state_manager.session:
            self.state_manager.transition_to(SessionPhase.PAUSED)
        logger.info("Paused")
    
    def resume(self) -> None:
        """Resume automation."""
        self._paused = False
        logger.info("Resumed")
    
    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
