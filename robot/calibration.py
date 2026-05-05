"""
Board calibration: maps chess squares to physical (x, y) coordinates.

Calibration is defined by two corner points measured by the robot arm:
  - A1 corner  (file a, rank 1  —  white queen-side rook)
  - H8 corner  (file h, rank 8  —  black king-side rook)

From these two diagonal corners the full 8×8 grid is derived.
Assuming the board is square and ranks run perpendicular (90° CCW) to files,
the file and rank unit vectors are solved from the diagonal:

    H8 = A1 + 7·file_step + 7·rank_step
    rank_step = rotate90CCW(file_step) = (−file_y, file_x)

    → file_x = (dx + dy) / 14
      file_y = (dy − dx) / 14
    where (dx, dy) = H8 − A1

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
    """Holds A1 and H8 physical coordinates and derives all 64 squares."""

    def __init__(self, a1: tuple[float, float], h8: tuple[float, float]):
        """
        Parameters
        ----------
        a1 : (x, y) physical position of the centre of square A1
        h8 : (x, y) physical position of the centre of square H8

        The rank direction is assumed to be 90° CCW from the file direction.
        This means: when standing at A1 and looking towards H1, rank 8 is to
        the left.  If the board is oriented differently relative to the robot
        base frame, flip the sign of _step_rank after construction.
        """
        self.a1 = a1
        self.h8 = h8

        dx = h8[0] - a1[0]
        dy = h8[1] - a1[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            raise ValueError("A1 and H8 cannot be at the same position")

        # Solve for file_step from the A1→H8 diagonal.
        # H8 = A1 + 7·file + 7·rank  where rank = rotate90CCW(file)
        # dx = 7·fx − 7·fy
        # dy = 7·fy + 7·fx
        fx = (dx + dy) / 14.0
        fy = (dy - dx) / 14.0
        self._step_file = (fx, fy)

        # rank_step = rotate90CCW(file_step): (fx,fy) → (−fy, fx)
        self._step_rank = (-fy, fx)

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
            "h8": list(self.h8),
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
        h8 = tuple(data["h8"])
        return cls(a1=a1, h8=h8)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"Calibration(a1={self.a1}, h8={self.h8})"
