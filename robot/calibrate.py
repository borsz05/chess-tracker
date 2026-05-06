"""
robot/calibrate.py — pre-match calibration script.

Guides the operator through teaching the robot two board corners:
  1. A1 corner  (white queen-side rook square)
  2. H8 corner  (black king-side rook square)

After both points are recorded the script saves robot/calibration.json.
Optionally it also records the graveyard zone (first and second slot,
from which the step vector is derived).

Run once before each match (and whenever the board or robot base moves):
    python -m robot.calibrate

Requires a concrete RobotInterface implementation.
Edit the ROBOT_IMPL import below to point at your specific driver.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Configure your robot implementation here ────────────────────────────────
# Replace the import below with the actual implementation once it exists.
# Example:
#   from robot.impl.ur5 import UR5Robot as RobotImpl
#
# During development a simple stub is used so the script can be tested
# without a real robot.

try:
    from robot.impl import FrankaRobot as RobotImpl  # type: ignore[import]
except ImportError:
    # Fallback stub — prompts the user to enter coordinates manually.
    from robot.interface import RobotInterface

    class RobotImpl(RobotInterface):  # type: ignore[no-redef]
        """Stub: asks the operator to type the current position."""

        def get_position(self) -> tuple[float, float, float]:
            raw = input("  Enter current position as  x y z  (mm, space-separated): ")
            parts = raw.strip().split()
            if len(parts) != 3:
                raise ValueError("Expected exactly three numbers")
            return (float(parts[0]), float(parts[1]), float(parts[2]))

        def execute_move(self, descriptor: dict) -> None:
            raise NotImplementedError("Stub cannot execute moves")

# ── calibration helpers ─────────────────────────────────────────────────────

from robot.calibration import Calibration, CALIBRATION_FILE
from robot.graveyard import Graveyard


def _teach_corner(robot: RobotImpl, label: str) -> tuple[float, float]:
    print(f"\n{'─' * 60}")
    print(f"  Step: position the robot end-effector over the {label} square.")
    input("  Press ENTER when the robot is in position… ")
    x, y, z = robot.get_position()
    print(f"  Recorded {label}: x={x:.2f}  y={y:.2f}  z={z:.2f}")
    return (x, y)


def _teach_graveyard(robot: RobotImpl) -> tuple[tuple[float, float], tuple[float, float]]:
    print(f"\n{'─' * 60}")
    print("  Graveyard calibration (optional).")
    ans = input("  Calibrate graveyard zone? [y/N]: ").strip().lower()
    if ans != "y":
        print("  Skipping graveyard — defaults will be used.")
        return None, None  # type: ignore[return-value]

    print("  Move robot to the FIRST graveyard slot.")
    input("  Press ENTER when ready… ")
    x0, y0, _ = robot.get_position()
    print(f"  Slot 0: x={x0:.2f}  y={y0:.2f}")

    print("  Move robot to the SECOND graveyard slot.")
    input("  Press ENTER when ready… ")
    x1, y1, _ = robot.get_position()
    print(f"  Slot 1: x={x1:.2f}  y={y1:.2f}")

    start = (x0, y0)
    step  = (x1 - x0, y1 - y0)
    print(f"  Graveyard start={start}  step={step}")
    return start, step


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Chess Robot — Board Calibration")
    print("=" * 60)
    print("  This script records the physical positions of the A1 and H8")
    print("  board corners so the robot knows where every square is.")
    print()
    print("  Camera orientation: A1 must be in the TOP-LEFT of the camera")
    print("  frame, H8 in the BOTTOM-RIGHT.")
    print()

    robot = RobotImpl()

    a1 = _teach_corner(robot, "A1")
    h8 = _teach_corner(robot, "H8")

    cal = Calibration(a1=a1, h8=h8)
    cal.save(CALIBRATION_FILE)

    # Optional graveyard
    g_start, g_step = _teach_graveyard(robot)
    if g_start is not None:
        graveyard = Graveyard(start=g_start, step=g_step)
        graveyard.save_to_calibration_file(CALIBRATION_FILE)

    print(f"\n{'=' * 60}")
    print(f"  Calibration complete.  File: {CALIBRATION_FILE}")
    print(f"  A1 = {a1}")
    print(f"  H8 = {h8}")

    # Quick sanity check
    print("\n  Square spot-check:")
    for sq in ["a1", "h8", "e4", "a8", "h1"]:
        xy = cal.square_to_xy(sq)
        print(f"    {sq.upper()} → x={xy[0]:.2f}  y={xy[1]:.2f}")

    print("\n  Done.  You can now start the vision pipeline.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
        sys.exit(1)
