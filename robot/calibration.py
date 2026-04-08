"""
Board calibration: maps chess squares to physical (x, y) coordinates.

Calibration is defined by two corner points measured by the robot arm:
  - A1 corner  (file a, rank 1  →  board col 0, row 7 in 0-indexed grid)
  - H1 corner  (file h, rank 1  →  board col 7, row 7)

From these two points the full 8×8 grid is derived via bilinear interpolation,
assuming the board is axis-aligned along the A1→H1 edge and that ranks go
perpendicular to that edge.

Coordinate system: whatever the robot arm uses (e.g. mm in its base frame).
Only x and y are stored; z (height) is robot-specific and handled in the
concrete RobotInterface implementation.

Usage:
    cal = Calibration.load()          # load robot/calibration.json
    xy  = cal.square_to_xy("e4")      # → (x, y) in robot coordinates
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

# Chess helpers
FILES = "abcdefgh"   # a=0 … h=7
RANKS = "12345678"   # 1=0 … 8=7


def _square_to_col_row(square: str) -> tuple[int, int]:
    """Return (col, row) indices for a square name.
    col: 0=a … 7=h
    row: 0=rank1 … 7=rank8
    """
    if len(square) != 2:
        raise ValueError(f"Invalid square name: {square!r}")
    file_char = square[0].lower()
    rank_char = square[1]
    if file_char not in FILES or rank_char not in RANKS:
        raise ValueError(f"Invalid square name: {square!r}")
    col = FILES.index(file_char)   # 0–7
    row = int(rank_char) - 1       # 0–7  (rank 1 → row 0)
    return col, row


class Calibration:
    """Holds A1 and H1 physical coordinates and derives all 64 squares."""

    def __init__(self, a1: tuple[float, float], h1: tuple[float, float]):
        """
        Parameters
        ----------
        a1 : (x, y) physical position of the centre of square A1
        h1 : (x, y) physical position of the centre of square H1
        """
        self.a1 = a1
        self.h1 = h1

        # Unit vector along the file direction (A→H, i.e. col direction)
        dx = h1[0] - a1[0]
        dy = h1[1] - a1[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            raise ValueError("A1 and H1 cannot be at the same position")
        self._step_file = (dx / 7.0, dy / 7.0)   # one square in file direction

        # Unit vector along the rank direction (rank 1→8, perpendicular to file)
        # Rotate file vector 90° CCW: (dx, dy) → (-dy, dx)
        self._step_rank = (-dy / 7.0, dx / 7.0)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def square_to_xy(self, square: str) -> tuple[float, float]:
        """Return physical (x, y) centre of the given square (e.g. 'e4')."""
        col, row = _square_to_col_row(square)
        x = (self.a1[0]
             + col * self._step_file[0]
             + row * self._step_rank[0])
        y = (self.a1[1]
             + col * self._step_file[1]
             + row * self._step_rank[1])
        return (x, y)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        data: dict[str, Any] = {
            "a1": list(self.a1),
            "h1": list(self.h1),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        print(f"Calibration saved to {path}")

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "Calibration":
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {path}\n"
                "Run  python -m robot.calibrate  before the match."
            )
        data = json.loads(path.read_text())
        a1 = tuple(data["a1"])
        h1 = tuple(data["h1"])
        return cls(a1=a1, h1=h1)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"Calibration(a1={self.a1}, h1={self.h1})"
