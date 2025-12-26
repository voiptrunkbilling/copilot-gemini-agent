"""
Manual calibration system for UI element regions.

Provides an overlay where user clicks to identify:
- VS Code window
- Copilot input box
- Response area
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from copilot_agent.logging import get_logger
from copilot_agent.actuator.platform import IS_WINDOWS, get_primary_screen, get_dpi_info
from copilot_agent.actuator.screenshot import Region

logger = get_logger(__name__)

# Try to import tkinter for overlay
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False
    logger.warning("tkinter not available, calibration overlay unavailable")


@dataclass
class CalibrationPoint:
    """A single calibration point."""
    
    name: str
    description: str
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    calibrated: bool = False
    timestamp: Optional[str] = None
    
    def to_region(self) -> Optional[Region]:
        """Convert to Region if fully calibrated."""
        if self.x is not None and self.y is not None:
            return Region(
                x=self.x,
                y=self.y,
                width=self.width or 100,
                height=self.height or 50,
            )
        return None


@dataclass
class CalibrationData:
    """Complete calibration data for a session."""
    
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    dpi_scale: float = 1.0
    screen_width: int = 1920
    screen_height: int = 1080
    
    # Calibration points
    vscode_window: CalibrationPoint = field(default_factory=lambda: CalibrationPoint(
        name="vscode_window",
        description="VS Code window (click title bar)",
    ))
    copilot_input: CalibrationPoint = field(default_factory=lambda: CalibrationPoint(
        name="copilot_input",
        description="Copilot Chat input box",
    ))
    copilot_response: CalibrationPoint = field(default_factory=lambda: CalibrationPoint(
        name="copilot_response",
        description="Copilot response area (top-left corner)",
    ))
    copilot_response_end: CalibrationPoint = field(default_factory=lambda: CalibrationPoint(
        name="copilot_response_end",
        description="Copilot response area (bottom-right corner)",
    ))
    
    def is_complete(self) -> bool:
        """Check if all required points are calibrated."""
        return all([
            self.vscode_window.calibrated,
            self.copilot_input.calibrated,
            self.copilot_response.calibrated,
        ])
    
    def get_response_region(self) -> Optional[Region]:
        """Get the response area as a Region."""
        if not self.copilot_response.calibrated:
            return None
        
        x = self.copilot_response.x or 0
        y = self.copilot_response.y or 0
        
        # Use end point if calibrated, otherwise default size
        if self.copilot_response_end.calibrated:
            width = (self.copilot_response_end.x or x + 400) - x
            height = (self.copilot_response_end.y or y + 300) - y
        else:
            width = self.copilot_response.width or 400
            height = self.copilot_response.height or 300
        
        return Region(x=x, y=y, width=width, height=height)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dpi_scale": self.dpi_scale,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "points": {
                "vscode_window": asdict(self.vscode_window),
                "copilot_input": asdict(self.copilot_input),
                "copilot_response": asdict(self.copilot_response),
                "copilot_response_end": asdict(self.copilot_response_end),
            },
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationData":
        """Create from dictionary."""
        data = cls(
            version=d.get("version", "1.0"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            dpi_scale=d.get("dpi_scale", 1.0),
            screen_width=d.get("screen_width", 1920),
            screen_height=d.get("screen_height", 1080),
        )
        
        points = d.get("points", {})
        
        if "vscode_window" in points:
            data.vscode_window = CalibrationPoint(**points["vscode_window"])
        if "copilot_input" in points:
            data.copilot_input = CalibrationPoint(**points["copilot_input"])
        if "copilot_response" in points:
            data.copilot_response = CalibrationPoint(**points["copilot_response"])
        if "copilot_response_end" in points:
            data.copilot_response_end = CalibrationPoint(**points["copilot_response_end"])
        
        return data


class CalibrationManager:
    """
    Manages calibration data persistence.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize calibration manager.
        
        Args:
            storage_path: Path to calibration data directory
        """
        self.storage_path = storage_path or Path.home() / ".copilot-agent" / "calibration"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._calibration_file = self.storage_path / "calibration.json"
    
    def load(self) -> CalibrationData:
        """Load calibration data from disk."""
        if not self._calibration_file.exists():
            return CalibrationData()
        
        try:
            with open(self._calibration_file, "r") as f:
                data = json.load(f)
            return CalibrationData.from_dict(data)
        except Exception as e:
            logger.error("Failed to load calibration", error=str(e))
            return CalibrationData()
    
    def save(self, data: CalibrationData) -> bool:
        """Save calibration data to disk."""
        try:
            data.updated_at = datetime.now(timezone.utc).isoformat()
            if not data.created_at:
                data.created_at = data.updated_at
            
            with open(self._calibration_file, "w") as f:
                json.dump(data.to_dict(), f, indent=2)
            
            logger.info("Calibration saved", path=str(self._calibration_file))
            return True
            
        except Exception as e:
            logger.error("Failed to save calibration", error=str(e))
            return False
    
    def clear(self) -> bool:
        """Clear calibration data."""
        try:
            if self._calibration_file.exists():
                self._calibration_file.unlink()
            return True
        except Exception as e:
            logger.error("Failed to clear calibration", error=str(e))
            return False


class CalibrationOverlay:
    """
    Tkinter-based overlay for manual calibration.
    
    Shows a transparent overlay with instructions and captures clicks.
    """
    
    def __init__(
        self,
        data: Optional[CalibrationData] = None,
        on_complete: Optional[Callable[[CalibrationData], None]] = None,
    ):
        """
        Initialize calibration overlay.
        
        Args:
            data: Existing calibration data to edit
            on_complete: Callback when calibration completes
        """
        if not HAS_TK:
            raise RuntimeError("tkinter not available")
        
        self.data = data or CalibrationData()
        self.on_complete = on_complete
        
        # Get screen info
        screen = get_primary_screen()
        dpi = get_dpi_info()
        
        self.data.screen_width = screen.width
        self.data.screen_height = screen.height
        self.data.dpi_scale = dpi.scale_factor
        
        # Calibration sequence
        self._sequence = [
            self.data.vscode_window,
            self.data.copilot_input,
            self.data.copilot_response,
            self.data.copilot_response_end,
        ]
        self._current_index = 0
        
        # UI components
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._label: Optional[tk.Label] = None
        self._crosshair_id = None
    
    def run(self) -> Optional[CalibrationData]:
        """
        Run the calibration overlay.
        
        Returns:
            CalibrationData if completed, None if cancelled
        """
        if not HAS_TK:
            logger.error("tkinter not available for calibration")
            return None
        
        self._root = tk.Tk()
        self._root.title("Copilot Agent Calibration")
        
        # Make window fullscreen and semi-transparent
        self._root.attributes("-fullscreen", True)
        self._root.attributes("-alpha", 0.3)
        
        # Windows-specific: allow click-through except for our canvas
        if IS_WINDOWS:
            try:
                self._root.attributes("-topmost", True)
            except Exception:
                pass
        
        # Bind escape to cancel
        self._root.bind("<Escape>", self._on_escape)
        self._root.bind("<Button-1>", self._on_click)
        self._root.bind("<Motion>", self._on_motion)
        
        # Create canvas
        self._canvas = tk.Canvas(
            self._root,
            highlightthickness=0,
            bg="gray",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create instruction label
        self._label = tk.Label(
            self._root,
            font=("Arial", 24, "bold"),
            fg="white",
            bg="black",
            padx=20,
            pady=10,
        )
        self._label.place(relx=0.5, rely=0.1, anchor="center")
        
        # Create progress label
        self._progress_label = tk.Label(
            self._root,
            font=("Arial", 14),
            fg="white",
            bg="black",
            padx=10,
            pady=5,
        )
        self._progress_label.place(relx=0.5, rely=0.15, anchor="center")
        
        # Create help text
        help_text = "Press ESC to cancel | Click to calibrate"
        self._help_label = tk.Label(
            self._root,
            text=help_text,
            font=("Arial", 12),
            fg="yellow",
            bg="black",
            padx=10,
            pady=5,
        )
        self._help_label.place(relx=0.5, rely=0.95, anchor="center")
        
        # Update display for first point
        self._update_display()
        
        # Run main loop
        self._root.mainloop()
        
        # Check if completed
        if self.data.is_complete():
            return self.data
        return None
    
    def _update_display(self):
        """Update the instruction display."""
        if self._current_index >= len(self._sequence):
            # All done
            self._on_complete_internal()
            return
        
        point = self._sequence[self._current_index]
        
        self._label.config(text=f"Click: {point.description}")
        self._progress_label.config(
            text=f"Step {self._current_index + 1} of {len(self._sequence)}"
        )
    
    def _on_click(self, event):
        """Handle click event."""
        if self._current_index >= len(self._sequence):
            return
        
        point = self._sequence[self._current_index]
        
        # Record click position
        point.x = event.x_root
        point.y = event.y_root
        point.calibrated = True
        point.timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info(
            "Calibration point recorded",
            name=point.name,
            x=point.x,
            y=point.y,
        )
        
        # Visual feedback
        self._canvas.create_oval(
            event.x - 5, event.y - 5,
            event.x + 5, event.y + 5,
            fill="green",
            outline="white",
        )
        
        # Move to next point
        self._current_index += 1
        self._root.after(200, self._update_display)
    
    def _on_motion(self, event):
        """Update crosshair on mouse motion."""
        if self._crosshair_id:
            self._canvas.delete(self._crosshair_id)
        
        # Draw crosshair
        self._crosshair_id = self._canvas.create_line(
            event.x, 0, event.x, self._root.winfo_height(),
            fill="red", width=1,
        )
        self._canvas.create_line(
            0, event.y, self._root.winfo_width(), event.y,
            fill="red", width=1,
            tags="crosshair",
        )
    
    def _on_escape(self, event):
        """Handle escape key."""
        if self._root:
            self._root.destroy()
    
    def _on_complete_internal(self):
        """Handle calibration completion."""
        if self.on_complete:
            self.on_complete(self.data)
        
        if self._root:
            # Show completion message
            messagebox.showinfo(
                "Calibration Complete",
                "All points have been calibrated.\n\n"
                f"VS Code: ({self.data.vscode_window.x}, {self.data.vscode_window.y})\n"
                f"Copilot Input: ({self.data.copilot_input.x}, {self.data.copilot_input.y})\n"
                f"Response Area: ({self.data.copilot_response.x}, {self.data.copilot_response.y})",
            )
            self._root.destroy()


def run_calibration(
    storage_path: Optional[Path] = None,
    recalibrate: bool = False,
) -> Optional[CalibrationData]:
    """
    Run the calibration process.
    
    Args:
        storage_path: Path to store calibration data
        recalibrate: If True, ignore existing calibration
        
    Returns:
        CalibrationData if successful, None otherwise
    """
    manager = CalibrationManager(storage_path)
    
    # Load existing calibration
    existing = None if recalibrate else manager.load()
    
    if existing and existing.is_complete():
        logger.info("Using existing calibration")
        return existing
    
    if not HAS_TK:
        logger.error("Cannot run calibration: tkinter not available")
        return None
    
    # Run overlay
    overlay = CalibrationOverlay(data=existing)
    result = overlay.run()
    
    if result:
        manager.save(result)
        logger.info("Calibration completed and saved")
    else:
        logger.warning("Calibration cancelled")
    
    return result


def run_calibration_cli() -> bool:
    """
    Run calibration from CLI (non-GUI fallback).
    
    Returns:
        True if successful
    """
    if not IS_WINDOWS:
        print("Manual calibration requires Windows with GUI access.")
        print("Please run on a Windows desktop with VS Code visible.")
        return False
    
    if HAS_TK:
        # Use GUI calibration
        result = run_calibration(recalibrate=True)
        return result is not None
    
    # Fallback: manual coordinate entry
    print("\n=== Manual Calibration ===")
    print("tkinter not available. Enter coordinates manually.\n")
    
    manager = CalibrationManager()
    data = CalibrationData()
    
    screen = get_primary_screen()
    data.screen_width = screen.width
    data.screen_height = screen.height
    data.dpi_scale = get_dpi_info().scale_factor
    
    points = [
        ("VS Code window title bar", data.vscode_window),
        ("Copilot Chat input box", data.copilot_input),
        ("Copilot response area (top-left)", data.copilot_response),
    ]
    
    for desc, point in points:
        print(f"\nEnter coordinates for: {desc}")
        try:
            x = int(input("  X: ").strip())
            y = int(input("  Y: ").strip())
            
            point.x = x
            point.y = y
            point.calibrated = True
            point.timestamp = datetime.now(timezone.utc).isoformat()
            
            print(f"  ✓ Recorded ({x}, {y})")
            
        except (ValueError, KeyboardInterrupt):
            print("\nCalibration cancelled.")
            return False
    
    # Save
    if manager.save(data):
        print(f"\n✓ Calibration saved to {manager._calibration_file}")
        return True
    
    return False
