"""
CLI interface using Click.

M6: Production-ready with stats, resume, and enhanced UX.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from copilot_agent import __version__
from copilot_agent.config import load_config, AgentConfig
from copilot_agent.state import StateManager, SessionPhase
from copilot_agent.tui import AgentTUI
from copilot_agent.safety.killswitch import KillSwitch
from copilot_agent.logging import setup_logging, get_logger
from copilot_agent.actuator import (
    ActionPipeline,
    PipelineAction,
    PipelineStep,
    run_calibration,
    run_calibration_cli,
    CalibrationManager,
    IS_WINDOWS,
    get_dpi_info,
    get_primary_screen,
)
from copilot_agent.checkpoint import list_resumable_sessions, AtomicCheckpointer
from copilot_agent.metrics import (
    MetricsCollector, aggregate_session_stats, load_session_metrics,
    MetricType,
)

console = Console()
logger = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, version: bool, verbose: bool) -> None:
    """Copilot-Gemini Agent - Automated coding loop orchestrator."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    if verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")
    
    if version:
        console.print(f"copilot-gemini-agent v{__version__}")
        sys.exit(0)
    
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@main.command()
@click.option("--task", "-t", required=True, help="Task description for Copilot")
@click.option(
    "--mode",
    type=click.Choice(["approve", "step", "auto"]),
    default="approve",
    help="Operation mode (default: approve)",
)
@click.option("--max-iterations", "-n", default=20, help="Maximum iterations (default: 20)")
@click.option("--dry-run", is_flag=True, help="Log actions without executing")
@click.option("--no-vision", is_flag=True, help="Disable Gemini Vision fallback")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.pass_context
def run(
    ctx: click.Context,
    task: str,
    mode: str,
    max_iterations: int,
    dry_run: bool,
    no_vision: bool,
    config: Optional[str],
) -> None:
    """Start a new automation session."""
    logger.info("Starting new session", task=task, mode=mode)
    
    # Load configuration
    agent_config = load_config(config)
    
    # Override config with CLI options
    if mode:
        agent_config.automation.default_mode = mode
    if max_iterations:
        agent_config.automation.max_iterations = max_iterations
    if no_vision:
        agent_config.perception.vision_enabled = False
    
    # Initialize state manager
    state_manager = StateManager(config=agent_config)
    session = state_manager.create_session(task=task)
    
    logger.info("Session created", session_id=session.session_id)
    
    # Initialize kill switch
    kill_switch = KillSwitch(
        hotkey=agent_config.safety.kill_switch_hotkey,
        on_trigger=lambda: _handle_kill_switch(state_manager),
    )
    
    # Initialize action pipeline
    pipeline = ActionPipeline(
        dry_run=dry_run,
        kill_switch_check=lambda: kill_switch.triggered,
    )
    
    # Initialize TUI
    tui = AgentTUI(
        state_manager=state_manager,
        kill_switch=kill_switch,
        dry_run=dry_run,
        verbose=ctx.obj.get("verbose", False),
    )
    
    try:
        # Start kill switch listener
        kill_switch.start()
        logger.info("Kill switch active", hotkey=agent_config.safety.kill_switch_hotkey)
        
        # Run TUI (blocks until exit)
        tui.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        kill_switch.stop()
        state_manager.checkpoint()
        logger.info("Session ended", session_id=session.session_id)


@main.command()
@click.option("--recalibrate", "-r", is_flag=True, help="Force recalibration even if existing")
@click.option("--show", "-s", is_flag=True, help="Show current calibration without changing")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
def calibrate(recalibrate: bool, show: bool, config: Optional[str]) -> None:
    """Run manual calibration for UI element detection."""
    
    # Show system info
    console.print("\n[bold]System Information[/bold]")
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    
    screen = get_primary_screen()
    dpi = get_dpi_info()
    
    table.add_row("Platform", "Windows" if IS_WINDOWS else "Linux/Mac")
    table.add_row("Screen Size", f"{screen.width} x {screen.height}")
    table.add_row("DPI Scale", f"{dpi.scale_factor:.0%} ({dpi.system_dpi} DPI)")
    table.add_row("DPI Aware", "Yes" if dpi.is_aware else "No")
    console.print(table)
    
    # Load existing calibration
    manager = CalibrationManager()
    existing = manager.load()
    
    if show:
        # Just show current calibration
        console.print("\n[bold]Current Calibration[/bold]")
        if existing.is_complete():
            cal_table = Table()
            cal_table.add_column("Element")
            cal_table.add_column("X")
            cal_table.add_column("Y")
            cal_table.add_column("Status")
            
            for point in [existing.vscode_window, existing.copilot_input, 
                         existing.copilot_response, existing.copilot_response_end]:
                status = "[green]✓[/green]" if point.calibrated else "[red]✗[/red]"
                x = str(point.x) if point.x else "-"
                y = str(point.y) if point.y else "-"
                cal_table.add_row(point.description, x, y, status)
            
            console.print(cal_table)
            console.print(f"\nLast updated: {existing.updated_at}")
        else:
            console.print("[yellow]No calibration data found.[/yellow]")
            console.print("Run 'agent calibrate' to start calibration.")
        return
    
    # Check if calibration exists
    if existing.is_complete() and not recalibrate:
        console.print("\n[green]✓ Calibration already exists![/green]")
        console.print("Use --recalibrate to overwrite.")
        return
    
    # Check platform
    if not IS_WINDOWS:
        console.print("\n[yellow]Warning: Calibration is designed for Windows.[/yellow]")
        console.print("GUI overlay may not work correctly on this platform.\n")
    
    # Instructions
    console.print("\n[bold]Calibration Instructions[/bold]")
    console.print("1. Make sure VS Code is open with Copilot Chat visible")
    console.print("2. A semi-transparent overlay will appear")
    console.print("3. Click on each element when prompted:")
    console.print("   - VS Code title bar")
    console.print("   - Copilot Chat input box")
    console.print("   - Response area (top-left corner)")
    console.print("   - Response area (bottom-right corner)")
    console.print("4. Press ESC to cancel at any time\n")
    
    if not click.confirm("Ready to start calibration?"):
        console.print("Calibration cancelled.")
        return
    
    # Run calibration
    console.print("\n[bold]Starting calibration overlay...[/bold]")
    
    try:
        result = run_calibration(recalibrate=recalibrate)
        
        if result and result.is_complete():
            console.print("\n[green]✓ Calibration completed successfully![/green]")
            console.print(f"VS Code window: ({result.vscode_window.x}, {result.vscode_window.y})")
            console.print(f"Copilot input: ({result.copilot_input.x}, {result.copilot_input.y})")
            console.print(f"Response area: ({result.copilot_response.x}, {result.copilot_response.y})")
        else:
            console.print("\n[yellow]Calibration was cancelled or incomplete.[/yellow]")
            
            # Offer CLI fallback
            if click.confirm("Would you like to enter coordinates manually?"):
                run_calibration_cli()
                
    except Exception as e:
        console.print(f"\n[red]Calibration error: {e}[/red]")
        console.print("Try running 'agent calibrate' again or enter coordinates manually.")


@main.command()
@click.option("--session", "-s", help="Session ID to resume (default: most recent)")
@click.option("--list", "-l", "list_sessions", is_flag=True, help="List resumable sessions")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.pass_context
def resume(ctx: click.Context, session: Optional[str], list_sessions: bool, config: Optional[str]) -> None:
    """Resume a session from checkpoint."""
    
    # Load configuration
    agent_config = load_config(config)
    sessions_path = agent_config.sessions_path
    
    if list_sessions:
        # List all resumable sessions
        resumable = list_resumable_sessions(sessions_path)
        
        if not resumable:
            console.print("[yellow]No resumable sessions found.[/yellow]")
            return
        
        console.print(f"\n[bold]Resumable Sessions ({len(resumable)})[/bold]\n")
        
        table = Table()
        table.add_column("Session ID", style="cyan")
        table.add_column("Task")
        table.add_column("Iteration")
        table.add_column("Last Step")
        table.add_column("Last Updated")
        
        for s in resumable:
            task_preview = s["task"][:40] + "..." if len(s["task"]) > 40 else s["task"]
            table.add_row(
                s["session_id"][:12] + "...",
                task_preview,
                str(s["iteration"]),
                s["last_step"] or "-",
                s["last_checkpoint_at"][:19] if s.get("last_checkpoint_at") else "-",
            )
        
        console.print(table)
        console.print("\n[dim]Use 'agent resume -s <session_id>' to resume a session[/dim]")
        return
    
    # Find session to resume
    if not session:
        # Get most recent resumable session
        resumable = list_resumable_sessions(sessions_path)
        if not resumable:
            console.print("[yellow]No resumable sessions found.[/yellow]")
            console.print("Use 'agent run' to start a new session.")
            return
        
        # Sort by last checkpoint time and get most recent
        resumable.sort(key=lambda x: x.get("last_checkpoint_at", ""), reverse=True)
        session_info = resumable[0]
        session = session_info["session_id"]
        
        console.print(f"[dim]Resuming most recent session: {session[:12]}...[/dim]")
    else:
        # Find specified session
        resumable = list_resumable_sessions(sessions_path)
        session_info = next((s for s in resumable if s["session_id"].startswith(session)), None)
        
        if not session_info:
            console.print(f"[red]Session not found or not resumable: {session}[/red]")
            console.print("Use 'agent resume --list' to see available sessions.")
            return
    
    session_path = Path(session_info["path"])
    
    # Load checkpoint
    checkpointer = AtomicCheckpointer(session_path)
    state = checkpointer.load()
    
    if not state:
        console.print("[red]Failed to load checkpoint.[/red]")
        return
    
    resume_point = checkpointer.get_resume_point()
    
    # Show resume info
    console.print(f"\n[bold]Resuming Session[/bold]")
    console.print(f"  Session ID: {state.session_id[:12]}...")
    console.print(f"  Task: {state.task[:60]}...")
    console.print(f"  Iteration: {state.current_iteration}/{state.max_iterations}")
    console.print(f"  Last Step: {state.last_step_type.value if state.last_step_type else 'N/A'}")
    console.print(f"  Resume Action: {resume_point.get('resume_action', 'N/A')}")
    console.print()
    
    if not click.confirm("Resume this session?"):
        console.print("Resume cancelled.")
        return
    
    # Initialize state manager with existing session
    state_manager = StateManager(config=agent_config)
    # Load the session from checkpoint
    state_manager._session_path = session_path
    
    # Re-create session from checkpoint state
    from copilot_agent.state import Session
    restored_session = Session(
        session_id=state.session_id,
        task_description=state.task,
        iteration_count=state.current_iteration,
        next_prompt=state.next_prompt,
    )
    state_manager._session = restored_session
    
    logger.info("Session loaded for resume", session_id=state.session_id)
    
    # Initialize kill switch
    kill_switch = KillSwitch(
        hotkey=agent_config.safety.kill_switch_hotkey,
        on_trigger=lambda: _handle_kill_switch(state_manager),
    )
    
    # Initialize TUI with checkpointer for resume
    tui = AgentTUI(
        state_manager=state_manager,
        kill_switch=kill_switch,
        dry_run=False,
        verbose=ctx.obj.get("verbose", False),
    )
    
    try:
        kill_switch.start()
        console.print("[green]Resuming session...[/green]\n")
        tui.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        kill_switch.stop()
        state_manager.checkpoint()
        logger.info("Session ended", session_id=state.session_id)


@main.command()
@click.option("--session", "-s", help="Session ID (default: latest)")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show stats for all sessions")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
def stats(session: Optional[str], show_all: bool, config: Optional[str]) -> None:
    """Show session statistics and metrics."""
    
    # Load configuration
    agent_config = load_config(config)
    sessions_path = agent_config.sessions_path
    
    if not sessions_path.exists():
        console.print("[yellow]No sessions found.[/yellow]")
        return
    
    # Find sessions
    session_dirs = [d for d in sessions_path.iterdir() if d.is_dir()]
    
    if not session_dirs:
        console.print("[yellow]No sessions found.[/yellow]")
        return
    
    if show_all:
        # Show summary for all sessions
        console.print(f"\n[bold]Session Statistics ({len(session_dirs)} sessions)[/bold]\n")
        
        table = Table()
        table.add_column("Session", style="cyan")
        table.add_column("Iterations")
        table.add_column("Reviewer")
        table.add_column("Vision")
        table.add_column("Errors")
        table.add_column("Status")
        
        for session_dir in sorted(session_dirs, reverse=True)[:10]:  # Last 10
            stats_data = aggregate_session_stats(session_dir)
            
            if not stats_data:
                continue
            
            counts = stats_data.get("counts", {})
            iterations = counts.get("iteration_end", 0)
            reviewer = counts.get("reviewer_call", 0)
            vision = counts.get("vision_call", 0)
            errors = stats_data.get("errors", 0)
            
            # Determine status from metrics
            has_accept = counts.get("session_end", 0) > 0
            status = "[green]Complete[/green]" if has_accept else "[yellow]Paused[/yellow]"
            
            table.add_row(
                session_dir.name[:12] + "...",
                str(iterations),
                str(reviewer),
                str(vision),
                str(errors),
                status,
            )
        
        console.print(table)
        console.print("\n[dim]Use 'agent stats -s <session_id>' for details[/dim]")
        return
    
    # Find specific session or latest
    if session:
        session_dir = next(
            (d for d in session_dirs if d.name.startswith(session)),
            None
        )
        if not session_dir:
            console.print(f"[red]Session not found: {session}[/red]")
            return
    else:
        # Get latest session
        session_dir = max(session_dirs, key=lambda d: d.stat().st_mtime)
    
    # Load stats
    stats_data = aggregate_session_stats(session_dir)
    
    if not stats_data:
        console.print(f"[yellow]No metrics found for session: {session_dir.name}[/yellow]")
        return
    
    # Display detailed stats
    console.print(f"\n[bold]Session Statistics[/bold]")
    console.print(f"  Session: {session_dir.name}")
    console.print()
    
    # Counts table
    counts = stats_data.get("counts", {})
    console.print("[bold]Operation Counts[/bold]")
    count_table = Table(show_header=False)
    count_table.add_column("Metric", style="cyan")
    count_table.add_column("Count", justify="right")
    
    for key in ["iteration_start", "iteration_end", "reviewer_call", "vision_call",
                "capture_success", "capture_failure", "ui_action"]:
        if key in counts:
            display_key = key.replace("_", " ").title()
            count_table.add_row(display_key, str(counts[key]))
    
    console.print(count_table)
    
    # Duration stats
    durations = stats_data.get("durations", {})
    if durations:
        console.print("\n[bold]Timing Statistics[/bold]")
        dur_table = Table()
        dur_table.add_column("Operation", style="cyan")
        dur_table.add_column("Calls", justify="right")
        dur_table.add_column("Avg (ms)", justify="right")
        dur_table.add_column("Total (s)", justify="right")
        
        for key, d in durations.items():
            display_key = key.replace("_", " ").title()
            dur_table.add_row(
                display_key,
                str(d["count"]),
                f"{d['avg_ms']:.0f}",
                f"{d['total_ms'] / 1000:.1f}",
            )
        
        console.print(dur_table)
    
    # Errors
    errors = stats_data.get("errors", 0)
    retries = stats_data.get("retries", 0)
    
    if errors or retries:
        console.print(f"\n[bold]Errors & Retries[/bold]")
        console.print(f"  Total errors: {errors}")
        console.print(f"  Total retries: {retries}")
    
    console.print()


@main.command()
@click.option("--dry-run", is_flag=True, help="Only log actions without executing")
@click.option("--skip-focus", is_flag=True, help="Skip window focus step")
def test_actions(dry_run: bool, skip_focus: bool) -> None:
    """Test GUI action primitives (M2 validation)."""
    
    console.print("\n[bold]M2 GUI Actions Test[/bold]\n")
    
    # Show system info
    screen = get_primary_screen()
    dpi = get_dpi_info()
    console.print(f"Screen: {screen.width}x{screen.height}, DPI: {dpi.scale_factor:.0%}")
    console.print(f"Dry-run: {dry_run}\n")
    
    # Create pipeline
    pipeline = ActionPipeline(dry_run=dry_run)
    
    # Check calibration
    if not pipeline.has_calibration:
        console.print("[yellow]⚠ No calibration found.[/yellow]")
        console.print("Some tests will be skipped. Run 'agent calibrate' first.\n")
    
    # Test sequence
    tests = [
        ("Focus VS Code", lambda: pipeline.focus_vscode() if not skip_focus else None),
        ("Wait 500ms", lambda: pipeline.wait(500)),
        ("Click Copilot Input", lambda: pipeline.click_copilot_input() if pipeline.has_calibration else None),
        ("Type 'hello'", lambda: pipeline.type_text("hello")),
        ("Press Enter", lambda: pipeline._action_executor.press_key("enter")),
        ("Wait 1000ms", lambda: pipeline.wait(1000)),
        ("Copy Selection (Ctrl+A, Ctrl+C)", lambda: (
            pipeline.hotkey("ctrl", "a"),
            pipeline.wait(100),
            pipeline.copy_selection(),
        )),
        ("Read Clipboard", lambda: pipeline.read_clipboard()),
    ]
    
    console.print("[bold]Running test sequence:[/bold]\n")
    
    for name, test_fn in tests:
        if test_fn is None:
            console.print(f"  [dim]⊘ {name} (skipped)[/dim]")
            continue
        
        try:
            result = test_fn()
            
            # Handle tuple results (from compound tests)
            if isinstance(result, tuple):
                result = result[-1]  # Take last result
            
            if result is None:
                console.print(f"  [dim]⊘ {name} (skipped)[/dim]")
            elif result.success:
                msg = result.message or "OK"
                if result.data:
                    # Show clipboard content preview
                    data_preview = str(result.data)[:50]
                    if len(str(result.data)) > 50:
                        data_preview += "..."
                    msg = f"{msg} → {data_preview}"
                console.print(f"  [green]✓ {name}[/green]: {msg}")
            else:
                console.print(f"  [red]✗ {name}[/red]: {result.error}")
                
        except Exception as e:
            console.print(f"  [red]✗ {name}[/red]: Exception: {e}")
    
    console.print("\n[bold]Test sequence complete.[/bold]")
    
    if dry_run:
        console.print("\n[dim]Note: This was a dry-run. No actual GUI actions were performed.[/dim]")


def _handle_kill_switch(state_manager: StateManager) -> None:
    """Handle kill switch activation."""
    logger.warning("Kill switch triggered!")
    state_manager.transition_to(SessionPhase.ABORTED)
    state_manager.checkpoint()


if __name__ == "__main__":
    main()
