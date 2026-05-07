"""
Graveyard manager: tracks where to place captured pieces next to the board.

The graveyard zone is defined by two points in physical space:
  - graveyard_start : (x, y) of the first slot
  - graveyard_step  : (dx, dy) offset between consecutive slots

Both are stored inside calibration.json alongside the board corners.
Pieces are placed sequentially; call reset() at the start of each game.

Usage:
    graveyard = Graveyard.from_calibration_file()
    xy = graveyard.next_slot()   # call after each capture
    graveyard.save_state()       # optional: persist slot counter mid-game
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot.calibration import CALIBRATION_FILE

# Defaults used when no graveyard config is found in calibration.json.
# Override by running the calibration script with graveyard teaching.
_DEFAULT_START = (400.0, 0.0)   # mm, example — replace via calibration
_DEFAULT_STEP  = (30.0, 0.0)    # 30 mm between slots along X axis

MAX_GRAVEYARD_SLOTS = 15  # robot plays Black; only White pieces are captured (king excluded)


class Graveyard:
    def __init__(
        self,
        start: tuple[float, float] = _DEFAULT_START,
        step:  tuple[float, float] = _DEFAULT_STEP,
    ):
        self.start = start
        self.step  = step
        self._next_index: int = 0

    # ------------------------------------------------------------------
    # Slot allocation
    # ------------------------------------------------------------------

    def next_slot(self) -> tuple[float, float]:
        """Return the (x, y) position for the next captured piece and
        advance the counter.  Raises RuntimeError when the zone is full."""
        if self._next_index >= MAX_GRAVEYARD_SLOTS:
            raise RuntimeError(
                f"Graveyard full: all {MAX_GRAVEYARD_SLOTS} slots used. "
                "This should never happen in a legal game."
            )
        x = self.start[0] + self._next_index * self.step[0]
        y = self.start[1] + self._next_index * self.step[1]
        self._next_index += 1
        return (x, y)

    def reset(self) -> None:
        """Reset to the first slot (call at the start of a new game)."""
        self._next_index = 0

    @property
    def slots_used(self) -> int:
        return self._next_index

    # ------------------------------------------------------------------
    # Persistence (piggybacked onto calibration.json)
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration_file(cls, path: Path = CALIBRATION_FILE) -> "Graveyard":
        """Load graveyard config from calibration.json.
        Falls back to defaults if the keys are absent."""
        if path.exists():
            data = json.loads(path.read_text())
            start = tuple(data.get("graveyard_start", list(_DEFAULT_START)))
            step  = tuple(data.get("graveyard_step",  list(_DEFAULT_STEP)))
        else:
            start, step = _DEFAULT_START, _DEFAULT_STEP
        return cls(start=start, step=step)  # type: ignore[arg-type]

    def save_to_calibration_file(self, path: Path = CALIBRATION_FILE) -> None:
        """Merge graveyard config into the existing calibration.json."""
        data: dict[str, Any] = {}
        if path.exists():
            data = json.loads(path.read_text())
        data["graveyard_start"] = list(self.start)
        data["graveyard_step"]  = list(self.step)
        path.write_text(json.dumps(data, indent=2))
