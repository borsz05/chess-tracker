"""
Robot execution service: builds a move descriptor from a UCI string
and dispatches it to the robot arm.

Loaded once at backend startup. If calibration.json is missing or
the executor is unreachable, the service stays None and the backend
runs in vision-only mode.
"""
from __future__ import annotations

import chess

from robot.calibration import Calibration
from robot.graveyard import Graveyard
from robot.move_descriptor import build_move_descriptor
from robot.impl import FrankaRobot


class RobotExecutionService:
    def __init__(self) -> None:
        self.calibration = Calibration.load()
        self.graveyard = Graveyard.from_calibration_file()
        self.robot = FrankaRobot()

    def execute(self, uci: str, board_before: chess.Board) -> None:
        """Build the physical descriptor and send it to the robot.

        board_before must be the board state BEFORE the move is applied —
        that is what build_move_descriptor needs to detect captures,
        castling, en passant, etc.
        """
        descriptor = build_move_descriptor(
            uci, board_before, self.calibration, self.graveyard
        )
        self.robot.execute_move(descriptor)

    def reset_graveyard(self) -> None:
        self.graveyard.reset()
