#!/usr/bin/env python3
"""
Chess Robot HTTP Executor — fut a Docker containeren belül (ROS2 Humble).

Indítás a containerben (a run.sh csinálja automatikusan):
    source /ros2_ws/install/setup.bash
    python3 /ros2_ws/src/chess_executor.py

HTTP API (port 8001):
    GET  /health          → {"ok": true}
    GET  /position        → {"x": mm, "y": mm, "z": mm}
    POST /execute         → {"descriptor": {...}}  →  {"ok": true} vagy {"error": "..."}
    POST /gripper/open    → {"ok": true}
    POST /gripper/close   → {"ok": true}
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2, MoveIt2Gripper

# ── Robot konstansok ─────────────────────────────────────────────────────────
_BASE_FRAME  = "fr3_link0"
_EEF_FRAME   = "fr3_hand_tcp"
_GROUP_NAME  = "fr3_arm"
_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
_GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
_DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]

# ── Fizikai paraméterek (méter) ──────────────────────────────────────────────
GRIPPER_OPEN  = [0.030, 0.030]
GRIPPER_GRIP  = [0.012, 0.012]
Z_TRAVEL      = 0.25
Z_PICK        = 0.04
Z_PLACE       = 0.03
VELOCITY      = 0.3
ACCELERATION  = 0.3

PORT = 8002

# ── Globális robot objektumok ────────────────────────────────────────────────
_node:     Node | None     = None
_moveit2:  MoveIt2 | None  = None
_gripper:  MoveIt2Gripper | None = None
_tf_buf:   Buffer | None   = None
_lock = threading.Lock()


def _init_robot() -> None:
    global _node, _moveit2, _gripper, _tf_buf

    rclpy.init()
    _node = Node("chess_executor")

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    _moveit2 = MoveIt2(
        node=_node,
        joint_names=_JOINT_NAMES,
        base_link_name=_BASE_FRAME,
        end_effector_name=_EEF_FRAME,
        group_name=_GROUP_NAME,
    )
    _moveit2.max_velocity     = VELOCITY
    _moveit2.max_acceleration = ACCELERATION

    _gripper = MoveIt2Gripper(
        node=_node,
        gripper_joint_names=_GRIPPER_JOINTS,
        open_gripper_joint_positions=GRIPPER_OPEN,
        closed_gripper_joint_positions=GRIPPER_GRIP,
    )

    _tf_buf = Buffer()
    TransformListener(_tf_buf, _node)

    time.sleep(1.5)
    print(f"[chess_executor] Robot kész. HTTP szerver: port {PORT}", flush=True)


def _get_position() -> tuple[float, float, float]:
    t = _tf_buf.lookup_transform(_BASE_FRAME, _EEF_FRAME, rclpy.time.Time())
    tr = t.transform.translation
    return (tr.x * 1000.0, tr.y * 1000.0, tr.z * 1000.0)


def _move_to(x_mm: float, y_mm: float, z_m: float) -> None:
    _moveit2.move_to_pose(
        position=[x_mm / 1000.0, y_mm / 1000.0, z_m],
        quat_xyzw=_DOWN_QUAT,
    )
    _moveit2.wait_until_executed()


def _pick_and_place(from_xy: tuple, to_xy: tuple) -> None:
    fx, fy = from_xy
    tx, ty = to_xy
    _move_to(fx, fy, Z_TRAVEL)
    _move_to(fx, fy, Z_PICK)
    _gripper.close(); _gripper.wait_until_executed()
    _move_to(fx, fy, Z_TRAVEL)
    _move_to(tx, ty, Z_TRAVEL)
    _move_to(tx, ty, Z_PLACE)
    _gripper.open(); _gripper.wait_until_executed()
    _move_to(tx, ty, Z_TRAVEL)


def _execute_move(descriptor: dict[str, Any]) -> None:
    t = descriptor["type"]
    if t == "simple":
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"])
    elif t in ("capture", "en_passant"):
        _pick_and_place(descriptor["captured_xy"], descriptor["graveyard_xy"])
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"])
    elif t == "castling":
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"])
        rook = descriptor["castling_rook"]
        _pick_and_place(rook["rook_from_xy"], rook["rook_to_xy"])
    else:
        raise ValueError(f"Ismeretlen lépéstípus: {t!r}")


# ── HTTP szerver ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[http] {fmt % args}", flush=True)

    def _send(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        elif self.path == "/position":
            try:
                x, y, z = _get_position()
                self._send(200, {"x": x, "y": y, "z": z})
            except Exception as e:
                self._send(500, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        with _lock:
            try:
                if self.path == "/execute":
                    desc = self._read_json()
                    _execute_move(desc)
                    self._send(200, {"ok": True})
                elif self.path == "/gripper/open":
                    _gripper.open(); _gripper.wait_until_executed()
                    self._send(200, {"ok": True})
                elif self.path == "/gripper/close":
                    _gripper.close(); _gripper.wait_until_executed()
                    self._send(200, {"ok": True})
                elif self.path == "/move":
                    body = self._read_json()
                    _move_to(body["x"], body["y"], body["z"] / 1000.0)
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, {"error": str(e)})


if __name__ == "__main__":
    _init_robot()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[chess_executor] Listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
