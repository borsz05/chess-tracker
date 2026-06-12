#!/usr/bin/env python3
# Requires: franka_msgs (franka_ros2 package)
"""
Chess Robot HTTP Executor — fut a Docker containeren belül (ROS2 Humble).

Indítás a containerben (a run.sh csinálja automatikusan):
    source /ros2_ws/install/setup.bash
    python3 /ros2_ws/src/chess_executor.py

HTTP API (port 8002):
    GET  /health          → {"ok": true}
    GET  /position        → {"x": mm, "y": mm, "z": mm}
    POST /execute         → {"descriptor": {...}}  →  {"ok": true} vagy {"error": "..."}
    POST /gripper/open    → {"ok": true}
    POST /gripper/close   → {"ok": true}
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2
from franka_msgs.action import Move as GripperMove, Grasp

# ── Robot konstansok ─────────────────────────────────────────────────────────
_BASE_FRAME  = "fr3_link0"
_EEF_FRAME   = "fr3_hand_tcp"
_GROUP_NAME  = "fr3_arm"
_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
_DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]

# ── Fizikai paraméterek (méter) ──────────────────────────────────────────────
GRIPPER_OPEN_WIDTH  = 0.06    # metres — clears any chess piece without hitting neighbours
GRIPPER_GRASP_FORCE = 20.0    # Newtons
GRIPPER_GRASP_SPEED = 0.05    # m/s
GRIPPER_GRASP_EPS   = 0.015   # metres inner/outer epsilon

Z_LIFT    = 0.12   # metres — safe clearance above tallest piece (king ~7cm + piece in gripper ~3cm margin)
Z_TRAVEL  = 0.20   # metres — arc peak height
Z_PICK    = 0.04   # metres — approach height for picking up a piece
Z_PLACE   = 0.03   # metres — approach height for placing a piece
ARC_WAYPOINTS = 8  # number of intermediate points along the arc

# Piece-specific grasp heights (metres above table).
# Values can be overridden via "piece_grasp_heights" in calibration.json.
PIECE_GRASP_HEIGHT: dict[str, float] = {
    "P": 0.020,   # gyalog ~33-40mm → nyak ~20mm
    "R": 0.025,   # bástya ~45mm → nyak ~25mm
    "N": 0.030,   # huszár ~55mm → nyak ~30mm
    "B": 0.035,   # futó ~65mm → nyak ~35mm
    "Q": 0.040,   # vezér ~80mm → nyak ~40mm
    "K": 0.045,   # király ~90mm → nyak ~45mm
}

VELOCITY     = 0.3
ACCELERATION = 0.3

PORT = 8002

# Calibration file path (same directory as this script inside the container).
_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

def _load_piece_grasp_heights() -> None:
    """Override PIECE_GRASP_HEIGHT from calibration.json if 'piece_grasp_heights' key present."""
    if not os.path.exists(_CAL_PATH):
        return
    try:
        data = json.loads(open(_CAL_PATH).read())
        overrides = data.get("piece_grasp_heights")
        if overrides:
            PIECE_GRASP_HEIGHT.update({k.upper(): float(v) for k, v in overrides.items()})
            print(f"[chess_executor] piece_grasp_heights betöltve: {overrides}", flush=True)
    except Exception as e:
        print(f"[chess_executor] piece_grasp_heights betöltési hiba (ignorálva): {e}", flush=True)


# ── Globális robot objektumok ────────────────────────────────────────────────
_node:    Node | None    = None
_moveit2: MoveIt2 | None = None
_tf_buf:  Buffer | None  = None
_gripper_move_client:  ActionClient | None = None
_gripper_grasp_client: ActionClient | None = None
_lock = threading.Lock()


def _init_robot() -> None:
    global _node, _moveit2, _tf_buf, _gripper_move_client, _gripper_grasp_client

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

    _tf_buf = Buffer()
    TransformListener(_tf_buf, _node)

    _gripper_move_client  = ActionClient(_node, GripperMove, "/fr3_gripper/move")
    _gripper_grasp_client = ActionClient(_node, Grasp,       "/fr3_gripper/grasp")

    _load_piece_grasp_heights()

    time.sleep(1.5)
    print(f"[chess_executor] Robot kész. HTTP szerver: port {PORT}", flush=True)


def _get_position() -> tuple[float, float, float]:
    t = _tf_buf.lookup_transform(_BASE_FRAME, _EEF_FRAME, rclpy.time.Time())
    tr = t.transform.translation
    return (tr.x * 1000.0, tr.y * 1000.0, tr.z * 1000.0)


# ── Gripper ──────────────────────────────────────────────────────────────────

def _gripper_open() -> None:
    goal = GripperMove.Goal(width=GRIPPER_OPEN_WIDTH, speed=0.1)
    future = _gripper_move_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(_node, future)
    goal_handle = future.result()
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(_node, result_future)


def _gripper_grasp() -> None:
    goal = Grasp.Goal(
        width=0.0,
        speed=GRIPPER_GRASP_SPEED,
        force=GRIPPER_GRASP_FORCE,
        epsilon=Grasp.Goal.GraspEpsilon(inner=GRIPPER_GRASP_EPS, outer=GRIPPER_GRASP_EPS),
    )
    future = _gripper_grasp_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(_node, future)
    goal_handle = future.result()
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(_node, result_future)


# ── Mozgástervezés ────────────────────────────────────────────────────────────

def _move_to(x_mm: float, y_mm: float, z_mm: float) -> None:
    """Egyszerű pose-alapú mozgás — kalibráló endpointhoz."""
    _moveit2.move_to_pose(
        position=[x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0],
        quat_xyzw=_DOWN_QUAT,
    )
    _moveit2.wait_until_executed()


def _assert_reached(expected_m: list[float], tolerance_mm: float = 5.0) -> None:
    """Raise RuntimeError if the robot is further than tolerance_mm from expected_m (metres)."""
    ax, ay, az = _get_position()
    ex = expected_m[0] * 1000.0
    ey = expected_m[1] * 1000.0
    ez = expected_m[2] * 1000.0
    dist = math.sqrt((ax - ex) ** 2 + (ay - ey) ** 2 + (az - ez) ** 2)
    if dist > tolerance_mm:
        raise RuntimeError(
            f"Robot nem érte el a várt pozíciót: "
            f"várt ({ex:.1f}, {ey:.1f}, {ez:.1f}) mm, "
            f"tényleges ({ax:.1f}, {ay:.1f}, {az:.1f}) mm, "
            f"eltérés {dist:.1f} mm > {tolerance_mm:.0f} mm"
        )


def _cartesian_move(waypoints: list[list[float]]) -> None:
    """Cartesian pályán halad végig a waypontokon (méterben, orientáció: _DOWN_QUAT).
    Ha a compute_cartesian_path nem elérhető, szekvenciális move_to_pose hívásokra esik vissza.
    """
    try:
        _moveit2.compute_cartesian_path(
            waypoints,
            quat_xyzw=_DOWN_QUAT,
            max_step=0.01,
        )
        _moveit2.wait_until_executed()
    except (AttributeError, TypeError):
        for wp in waypoints:
            _moveit2.move_to_pose(position=wp, quat_xyzw=_DOWN_QUAT)
            _moveit2.wait_until_executed()
    _assert_reached(waypoints[-1])


def _vertical_path(x_mm: float, y_mm: float, z_start: float, z_end: float) -> None:
    """Egyenes fel/le mozgás rögzített XY pozícióban."""
    waypoints = [
        [x_mm / 1000.0, y_mm / 1000.0, z_start],
        [x_mm / 1000.0, y_mm / 1000.0, z_end],
    ]
    _cartesian_move(waypoints)


def _arc_path(from_xy: tuple, to_xy: tuple) -> None:
    """
    Sima parabolikus ív from_xy-tól to_xy-ig, mindkét végpont Z_LIFT magasságon.
    Az ív csúcsa Z_TRAVEL. A parabola paraméteres formája biztosítja, hogy
    t=0 és t=1-nél dz/dt=0, azaz a robot vízszintesen indul és érkezik Z_LIFT-en.
    """
    fx, fy = from_xy[0] / 1000.0, from_xy[1] / 1000.0
    tx, ty = to_xy[0] / 1000.0,   to_xy[1] / 1000.0
    waypoints = []
    for i in range(ARC_WAYPOINTS + 1):
        t = i / ARC_WAYPOINTS
        x = fx + t * (tx - fx)
        y = fy + t * (ty - fy)
        # parabola: z(t) = Z_LIFT + 4*(Z_TRAVEL - Z_LIFT)*t*(1-t)
        # t=0 és t=1-nél: z = Z_LIFT;  t=0.5-nél: z = Z_TRAVEL
        z = Z_LIFT + 4.0 * (Z_TRAVEL - Z_LIFT) * t * (1.0 - t)
        waypoints.append([x, y, z])
    _cartesian_move(waypoints)


def _pick_and_place(from_xy: tuple, to_xy: tuple, piece: str = "P") -> None:
    fx, fy = from_xy
    tx, ty = to_xy
    z_pick = Z_PICK + PIECE_GRASP_HEIGHT.get(piece.upper(), 0.0)
    _vertical_path(fx, fy, Z_LIFT, z_pick)       # 1. fázis: leereszkedés a bábura
    _gripper_grasp()                              # 2. fázis: megfogás z_pick magasságon
    _vertical_path(fx, fy, z_pick, Z_LIFT)       # 3. fázis: felemelés
    _arc_path(from_xy, to_xy)                    # 4. fázis: sima ív
    _vertical_path(tx, ty, Z_LIFT, Z_PLACE)      # 5. fázis: egyenes leengedés
    _gripper_open()                               # 6. fázis: elengedés
    _vertical_path(tx, ty, Z_PLACE, Z_LIFT)      # 7. fázis: visszaemelés


# ── Lépésvégrehajtás ─────────────────────────────────────────────────────────

def _execute_move(descriptor: dict[str, Any]) -> None:
    t = descriptor["type"]
    piece = descriptor.get("piece", "P")
    if t == "simple":
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"], piece)
    elif t == "capture":
        captured_piece = descriptor.get("captured_piece", "P")
        _pick_and_place(descriptor["captured_xy"], descriptor["graveyard_xy"], captured_piece)
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"], piece)
    elif t == "en_passant":
        _pick_and_place(descriptor["captured_xy"], descriptor["graveyard_xy"], "P")
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"], piece)
    elif t == "castling":
        _pick_and_place(descriptor["piece_from_xy"], descriptor["piece_to_xy"], "K")
        rook = descriptor["castling_rook"]
        _pick_and_place(rook["rook_from_xy"], rook["rook_to_xy"], "R")
    elif t == "promotion":
        if descriptor.get("captured_xy") is not None:
            captured_piece = descriptor.get("captured_piece", "P")
            _pick_and_place(descriptor["captured_xy"], descriptor["graveyard_xy"], captured_piece)
        _pick_and_place(descriptor["piece_from_xy"], descriptor["pawn_graveyard_xy"], "P")
        _pick_and_place(descriptor["piece_to_xy"], descriptor["promotion_target_xy"], "Q")
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
                    _gripper_open()
                    self._send(200, {"ok": True})
                elif self.path == "/gripper/close":
                    _gripper_grasp()
                    self._send(200, {"ok": True})
                elif self.path == "/move":
                    body = self._read_json()
                    _move_to(body["x"], body["y"], body["z"])
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
