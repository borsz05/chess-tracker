"""
Franka Research 3 — HTTP kliens a chess_executor.py szerverhez.

A chess_executor.py a Docker containeren belül fut (ROS2 Humble + pymoveit2),
és egy HTTP API-n keresztül fogadja a robot parancsokat.
Így elkerüljük a Humble/Jazzy moveit_msgs CDR inkompatibilitást.

A run.sh automatikusan elindítja a szervert a containerben.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from robot.interface import RobotInterface

_EXECUTOR_URL = "http://localhost:8002"
_TIMEOUT = 60   # másodperc — egy teljes sakklépés végrehajtási ideje


class RobotImpl(RobotInterface):
    """HTTP kliens a Docker containeren belül futó chess_executor.py-hoz."""

    def __init__(self, url: str = _EXECUTOR_URL) -> None:
        self._url = url
        self._wait_for_server()

    def _wait_for_server(self, retries: int = 20, delay: float = 1.5) -> None:
        print(f"Robot executor elérése: {self._url} ...")
        for i in range(retries):
            try:
                r = requests.get(f"{self._url}/health", timeout=2)
                if r.json().get("ok"):
                    print("Robot executor kész.")
                    return
            except Exception:
                pass
            if i < retries - 1:
                print(f"  Várakozás... ({i+1}/{retries})")
                time.sleep(delay)
        raise RuntimeError(
            f"Robot executor nem elérhető: {self._url}\n"
            "Ellenőrizd hogy a run.sh elindította-e a chess_executor.py-t."
        )

    # ── RobotInterface ───────────────────────────────────────────────────────

    def get_position(self) -> tuple[float, float, float]:
        r = requests.get(f"{self._url}/position", timeout=5)
        r.raise_for_status()
        d = r.json()
        return (d["x"], d["y"], d["z"])

    def execute_move(self, descriptor: dict[str, Any]) -> None:
        r = requests.post(
            f"{self._url}/execute",
            json=descriptor,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        result = r.json()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Ismeretlen hiba"))

    # ── Extra metódusok a tesztscripthez ────────────────────────────────────

    def open_gripper(self) -> None:
        requests.post(f"{self._url}/gripper/open", timeout=10).raise_for_status()

    def close_gripper(self) -> None:
        requests.post(f"{self._url}/gripper/close", timeout=10).raise_for_status()

    def move_to(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        r = requests.post(
            f"{self._url}/move",
            json={"x": x_mm, "y": y_mm, "z": z_mm},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
