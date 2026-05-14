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
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2
from franka_msgs.action import Move as GripperMove, Grasp

# ── Kamera konstansok ────────────────────────────────────────────────────────
_CAMERA_TOPIC = "/camera/color/image_raw"

# A kalibrációs fájl elérési útja a containerben (docker-compose mountolja)
_CALIBRATION_PATH = Path("/robot/calibration.json")

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

VELOCITY     = 0.3
ACCELERATION = 0.3

PORT = 8002

# ── Globális robot objektumok ────────────────────────────────────────────────
_node:    Node | None    = None
_moveit2: MoveIt2 | None = None
_tf_buf:  Buffer | None  = None
_lock = threading.Lock()

_camera_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_at_home: bool = False
_home_pose_xyz: list[float] | None = None  # [x, y, z] méterben, kalibráció alapján


def _camera_callback(msg: RosImage) -> None:
    global _latest_jpeg
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        arr = arr[:, :, ::-1]  # RGB → BGR
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with _camera_lock:
            _latest_jpeg = buf.tobytes()


def _compute_home_from_calibration() -> list[float]:
    """Kiszámítja a home pozíciót a kalibrációból.

    XY: a sakktábla közepe, 50mm-rel a robot alapja felé eltolva
    (hogy a csuklón lévő kamera pontosan a tábla közepére nézzen).
    Z:  a lehetséges legnagyobb elérhető magasság 700mm-től 500mm-ig,
        IK-ellenőrzéssel meghatározva.
    """
    if not _CALIBRATION_PATH.exists():
        raise RuntimeError(
            f"Kalibrációs fájl nem található: {_CALIBRATION_PATH}\n"
            "Futtasd előbb: python -m robot.calibrate"
        )

    data = json.loads(_CALIBRATION_PATH.read_text())
    a1_x, a1_y = data["a1"]
    h8_x, h8_y = data["h8"]

    # Tábla közepe mm-ben
    cx = (a1_x + h8_x) / 2.0
    cy = (a1_y + h8_y) / 2.0

    # 50mm eltolás a robot alapja (0,0) irányába
    dx, dy = -cx, -cy
    dist = math.hypot(dx, dy)
    if dist > 0:
        cx += dx / dist * 50.0
        cy += dy / dist * 50.0

    x_m = cx / 1000.0
    y_m = cy / 1000.0

    # Legmagasabb elérhető Z keresése IK-ellenőrzéssel
    for z_mm in range(700, 449, -50):
        try:
            result = _moveit2.compute_ik(
                position=[x_m, y_m, z_mm / 1000.0],
                quat_xyzw=_DOWN_QUAT,
            )
            if result is not None:
                print(
                    f"[chess_executor] Home pose: x={cx:.0f}mm  y={cy:.0f}mm  z={z_mm}mm",
                    flush=True,
                )
                return [x_m, y_m, z_mm / 1000.0]
        except Exception:
            pass

    print(f"[chess_executor] Home pose (fallback 500mm): x={cx:.0f}mm  y={cy:.0f}mm", flush=True)
    return [x_m, y_m, 0.5]


def _go_home() -> None:
    global _at_home, _home_pose_xyz
    _at_home = False
    if _home_pose_xyz is None:
        try:
            _home_pose_xyz = _compute_home_from_calibration()
        except RuntimeError as e:
            print(f"[chess_executor] Home pose kihagyva (még nincs kalibráció): {e}", flush=True)
            return
    _moveit2.move_to_pose(position=_home_pose_xyz, quat_xyzw=_DOWN_QUAT)
    _moveit2.wait_until_executed()
    _at_home = True


def _init_robot() -> None:
    global _node, _moveit2, _tf_buf

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

    _node.create_subscription(RosImage, _CAMERA_TOPIC, _camera_callback, 10)

    time.sleep(1.5)
    _go_home()
    print(f"[chess_executor] Robot kész. HTTP szerver: port {PORT}", flush=True)


def _get_position() -> tuple[float, float, float]:
    t = _tf_buf.lookup_transform(_BASE_FRAME, _EEF_FRAME, rclpy.time.Time())
    tr = t.transform.translation
    return (tr.x * 1000.0, tr.y * 1000.0, tr.z * 1000.0)


# ── Gripper ──────────────────────────────────────────────────────────────────

def _gripper_open() -> None:
    client = ActionClient(_node, GripperMove, "/fr3_gripper/move")
    goal = GripperMove.Goal(width=GRIPPER_OPEN_WIDTH, speed=0.1)
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(_node, future)
    future.result().get_result_async()


def _gripper_grasp() -> None:
    client = ActionClient(_node, Grasp, "/fr3_gripper/grasp")
    goal = Grasp.Goal(
        width=0.0,
        speed=GRIPPER_GRASP_SPEED,
        force=GRIPPER_GRASP_FORCE,
        epsilon=Grasp.Goal.GraspEpsilon(inner=GRIPPER_GRASP_EPS, outer=GRIPPER_GRASP_EPS),
    )
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(_node, future)
    future.result().get_result_async()


# ── Mozgástervezés ────────────────────────────────────────────────────────────

def _move_to(x_mm: float, y_mm: float, z_m: float) -> None:
    """Egyszerű pose-alapú mozgás — kalibráló endpointhoz."""
    _moveit2.move_to_pose(
        position=[x_mm / 1000.0, y_mm / 1000.0, z_m],
        quat_xyzw=_DOWN_QUAT,
    )
    _moveit2.wait_until_executed()


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


def _pick_and_place(from_xy: tuple, to_xy: tuple) -> None:
    fx, fy = from_xy
    tx, ty = to_xy
    _vertical_path(fx, fy, Z_PICK, Z_LIFT)      # 1. fázis: egyenes felemelés
    _gripper_grasp()                              # megfogás felemelés után
    _arc_path(from_xy, to_xy)                    # 2. fázis: sima ív
    _vertical_path(tx, ty, Z_LIFT, Z_PLACE)      # 3. fázis: egyenes leengedés
    _gripper_open()                               # elengedés
    _vertical_path(tx, ty, Z_PLACE, Z_LIFT)      # 4. fázis: visszaemelés a következő lépés előtt


# ── Lépésvégrehajtás ─────────────────────────────────────────────────────────

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
    elif t == "promotion":
        _pick_and_place(descriptor["piece_from_xy"], descriptor["pawn_graveyard_xy"])
        _pick_and_place(descriptor["piece_to_xy"], descriptor["promotion_target_xy"])
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
        elif self.path == "/camera/frame":
            with _camera_lock:
                frame = _latest_jpeg
            if frame is None:
                self._send(503, {"error": "no frame yet"})
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
        elif self.path == "/robot/at_home":
            self._send(200, {"at_home": _at_home})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        with _lock:
            try:
                if self.path == "/execute":
                    desc = self._read_json()
                    global _at_home
                    _at_home = False
                    _execute_move(desc)
                    _go_home()
                    self._send(200, {"ok": True})
                elif self.path == "/gripper/open":
                    _gripper_open()
                    self._send(200, {"ok": True})
                elif self.path == "/gripper/close":
                    _gripper_grasp()
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
