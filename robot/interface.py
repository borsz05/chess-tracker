"""
Abstract base class for robot arm implementations.
Implement this class for your specific robot SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RobotInterface(ABC):
    """Robot-agnostic interface. Implement execute_move and get_position
    for your specific robot arm SDK."""

    @abstractmethod
    def execute_move(self, descriptor: dict[str, Any]) -> None:
        """Execute a chess move described by a move descriptor dict.

        The descriptor is produced by robot.move_descriptor.build_move_descriptor
        and contains all physical coordinates the robot needs.
        """

    @abstractmethod
    def get_position(self) -> tuple[float, float, float]:
        """Return the robot's current end-effector position as (x, y, z)
        in the same coordinate system used by calibration (millimetres)."""
