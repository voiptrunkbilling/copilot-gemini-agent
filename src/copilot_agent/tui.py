"""
Terminal User Interface using Rich.

M4: Enhanced with verdict display, iteration history, and pause controls.
"""

import asyncio
from datetime import datetime
from typing import Optional, Callable, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from copilot_agent.state import StateManager, SessionPhase, GeminiVerdict
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
        on_user_continue: Optional[Callable[[], None]] = None,
        on_user_stop: Optional[Callable[[], None]] = None,
        on_user_override: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the TUI.
        
        Args:
            state_manager: State manager instance
            kill_switch: Kill switch instance
            dry_run: Whether running in dry-run mode
            verbose: Show verbose output
            on_user_continue: Callback when user approves continuation
            on_user_stop: Callback when user stops the loop
            on_user_override: Callback when user provides custom prompt
        """
        self.state_manager = state_manager
        self.kill_switch = kill_switch
        self.dry_run = dry_run
        self.verbose = verbose
        self.console = Console()
        
        # Callbacks
        self._on_continue = on_user_continue
        self._on_stop = on_user_stop
        self._on_override = on_user_override
        
        # State
        self._running = False
        self._paused = False
        self._awaiting_input = False
        self._user_message: Optional[str] = None
    
    def make_header(self) -> Panel:
        """Create header panel."""
        session = self.state_manager.session
        
        if self.kill_switch.triggered:
            status = "[bold red]KILLED[/bold red] ⛔"
        elif self._paused or (session and session.phase == SessionPhase.PAUSED):
            status = "[bold yellow]PAUSED[/bold yellow] ⏸️"
        elif session and session.phase == SessionPhase.COMPLETE:
            status = "[bold green]COMPLETE[/bold green] ✅"
        elif session and session.phase == SessionPhase.FAILED:
            status = "[bold red]FAILED[/bold red] ❌"
        elif session and session.phase == SessionPhase.ABORTED:
            status = "[bold red]ABORTED[/bold red] 🛑"
        elif self._running:
            phase_emoji = {
                SessionPhase.PROMPTING: "📝",
                SessionPhase.WAITING: "⏳",
                SessionPhase.CAPTURING: "📸",
                SessionPhase.REVIEWING: "🔍",
            }.get(session.phase if session else None, "🟢")
            status = f"[bold green]RUNNING[/bold green] {phase_emoji}"
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
            
            # Get max iterations from config
            max_iter = session.max_iterations
            
            table.add_row("Session", session.session_id)
            table.add_row("Iteration", f"[bold]{session.iteration_count}[/bold]/{max_iter}")
            table.add_row("Runtime", runtime_str)
            table.add_row("Phase", self._format_phase(session.phase))
            table.add_row("Errors", f"{session.total_errors} (consecutive: {session.consecutive_errors})")
            
            # Show completion reason if finished
            if session.completion_reason:
                table.add_row("Result", f"[cyan]{session.completion_reason}[/cyan]")
        else:
            table.add_row("Session", "[dim]None[/dim]")
        
        return table
    
    def _format_phase(self, phase: SessionPhase) -> str:
        """Format phase for display."""
        phase_colors = {
            SessionPhase.IDLE: "dim",
            SessionPhase.PROMPTING: "cyan",
            SessionPhase.WAITING: "yellow",
            SessionPhase.CAPTURING: "magenta",
            SessionPhase.REVIEWING: "blue",
            SessionPhase.PAUSED: "yellow",
            SessionPhase.COMPLETE: "green",
            SessionPhase.FAILED: "red",
            SessionPhase.ABORTED: "red",
        }
        color = phase_colors.get(phase, "white")
        return f"[{color}]{phase.value.upper()}[/{color}]"
    
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
        """Create current iteration panel with verdict display."""
        session = self.state_manager.session
        
        if not session:
            return Panel("[dim]No active session[/dim]", title="Current", border_style="dim")
        
        lines = []
        
        # Current prompt
        if session.current_prompt:
            prompt = session.current_prompt
            if len(prompt) > 100:
                prompt = prompt[:100] + "..."
            source = session.current_prompt_source
            source_badge = {
                "initial": "[cyan]initial[/cyan]",
                "gemini_followup": "[magenta]gemini[/magenta]",
                "human_override": "[green]human[/green]",
            }.get(source, source)
            lines.append(f"[bold]Prompt[/bold] ({source_badge}):\n{prompt}")
        
        # Current response
        if session.current_response:
            response = session.current_response
            if len(response) > 200:
                response = response[:200] + "..."
            lines.append(f"\n[bold]Copilot Response:[/bold]\n{response}")
        elif session.phase == SessionPhase.WAITING:
            lines.append("\n[yellow]⏳ Waiting for Copilot response...[/yellow]")
        elif session.phase == SessionPhase.CAPTURING:
            lines.append("\n[yellow]📸 Capturing response...[/yellow]")
        elif session.phase == SessionPhase.REVIEWING:
            lines.append("\n[blue]🔍 Sending to Gemini for review...[/blue]")
        
        # Verdict with prominent display
        if session.current_verdict:
            verdict = session.current_verdict.value.upper()
            verdict_display = self._format_verdict(session.current_verdict)
            lines.append(f"\n[bold]Gemini Verdict:[/bold] {verdict_display}")
            
            if session.current_feedback:
                feedback = session.current_feedback
                if len(feedback) > 150:
                    feedback = feedback[:150] + "..."
                lines.append(f"[dim]{feedback}[/dim]")
            
            # Show next action for CRITIQUE
            if session.current_verdict == GeminiVerdict.CRITIQUE and session.next_prompt:
                next_prompt = session.next_prompt
                if len(next_prompt) > 100:
                    next_prompt = next_prompt[:100] + "..."
                lines.append(f"\n[bold]Next Prompt:[/bold]\n[italic]{next_prompt}[/italic]")
        
        # Pause message
        if session.phase == SessionPhase.PAUSED:
            lines.append("\n[yellow bold]⏸️ PAUSED - Press [C] to continue or [Q] to quit[/yellow bold]")
        
        content = "\n".join(lines) if lines else "[dim]Iteration not started[/dim]"
        
        return Panel(content, title=f"Current Iteration ({session.iteration_count})", border_style="green")
    
    def _format_verdict(self, verdict: GeminiVerdict) -> str:
        """Format verdict with color and emoji."""
        formats = {
            GeminiVerdict.ACCEPT: "[bold green]✅ ACCEPT[/bold green]",
            GeminiVerdict.CRITIQUE: "[bold yellow]🔄 CRITIQUE[/bold yellow]",
            GeminiVerdict.CLARIFY: "[bold magenta]❓ CLARIFY[/bold magenta]",
            GeminiVerdict.ERROR: "[bold red]❌ ERROR[/bold red]",
        }
        return formats.get(verdict, verdict.value.upper())
    
    def make_history_panel(self) -> Panel:
        """Create iteration history panel."""
        session = self.state_manager.session
        
        if not session or not session.iteration_history:
            return Panel("[dim]No history yet[/dim]", title="History", border_style="dim")
        
        # Show last 5 iterations
        recent = session.iteration_history[-5:]
        
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Verdict", width=12)
        table.add_column("Summary", ratio=1)
        
        for record in recent:
            num = str(record.iteration_number)
            
            verdict = record.gemini_verdict or "?"
            verdict_style = {
                "accept": "green",
                "critique": "yellow",
                "clarify": "magenta",
                "error": "red",
            }.get(verdict.lower(), "white")
            
            summary = record.gemini_feedback or ""
            if len(summary) > 40:
                summary = summary[:40] + "..."
            
            table.add_row(
                num,
                f"[{verdict_style}]{verdict.upper()}[/{verdict_style}]",
                summary,
            )
        
        return Panel(table, title="Recent History", border_style="blue")
    
    def make_controls_panel(self) -> Panel:
        """Create keyboard controls panel."""
        session = self.state_manager.session
        
        if session and session.phase == SessionPhase.PAUSED:
            controls = [
                "[bold][C]ontinue[/bold]",
                "[bold][O]verride prompt[/bold]",
                "[bold][S]kip[/bold]",
                "[bold red][Q]uit[/bold red]",
            ]
        else:
            controls = [
                "[P]ause",
                "[K]ill switch",
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
            Layout(name="history"),
        )
        
        # Populate layout
        layout["header"].update(self.make_header())
        layout["status"].update(Panel(self.make_status_table(), title="Status", border_style="blue"))
        layout["task"].update(self.make_task_panel())
        layout["history"].update(self.make_history_panel())
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
    
    async def run_async(self) -> None:
        """Run the TUI asynchronously."""
        self._running = True
        
        logger.info("TUI started (async)")
        
        try:
            with Live(
                self.make_layout(),
                console=self.console,
                refresh_per_second=2,
                screen=True,
            ) as live:
                while self._running:
                    if self.kill_switch.triggered:
                        self._running = False
                        break
                    
                    live.update(self.make_layout())
                    await asyncio.sleep(0.5)
                    
                    session = self.state_manager.session
                    if session and session.phase in (
                        SessionPhase.COMPLETE,
                        SessionPhase.FAILED,
                        SessionPhase.ABORTED,
                    ):
                        await asyncio.sleep(2)
                        self._running = False
        
        except asyncio.CancelledError:
            self._running = False
            logger.info("TUI cancelled")
        
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
        if self._on_continue:
            self._on_continue()
        logger.info("Resumed")
    
    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
        if self._on_stop:
            self._on_stop()
    
    def set_user_message(self, message: str) -> None:
        """Set a message from user input."""
        self._user_message = message
        self._awaiting_input = False
    
    def print_summary(self) -> None:
        """Print a final summary after TUI closes."""
        session = self.state_manager.session
        if not session:
            return
        
        self.console.print()
        self.console.print("[bold]Session Summary[/bold]")
        self.console.print(f"  Session ID: {session.session_id}")
        self.console.print(f"  Iterations: {session.iteration_count}")
        self.console.print(f"  Final Phase: {session.phase.value}")
        
        if session.current_verdict:
            verdict_str = self._format_verdict(session.current_verdict)
            self.console.print(f"  Final Verdict: {verdict_str}")
        
        if session.completion_reason:
            self.console.print(f"  Completion: {session.completion_reason}")
        
        if session.final_result:
            result_preview = session.final_result[:200]
            if len(session.final_result) > 200:
                result_preview += "..."
            self.console.print(f"\n[bold]Final Result:[/bold]\n{result_preview}")
        
        self.console.print()
