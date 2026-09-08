#!/usr/bin/env python3
# Requires: franka_msgs (franka_ros2 package)
"""
Chess Robot HTTP Executor — fut a Docker containeren belül (ROS2 Humble).

Indítás a containerben (a run.sh csinálja automatikusan):
    source /ros2_ws/install/setup.bash
    python3 /ros2_ws/src/chess_executor.py

HTTP API (port 8002):
    GET  /health          → {"ok": true, "errors_active": bool}
    GET  /status          → pozíció, Franka hibajelzők, felderített képességek,
                            utolsó strukturált hiba — ez a hibakeresés belépője
    GET  /position        → {"x": mm, "y": mm, "z": mm}
    POST /execute         → {"descriptor": {...}}  →  {"ok": true} vagy részletes hiba
    POST /move            → {"x": mm, "y": mm, "z": mm}
    POST /gripper/open    → {"ok": true}
    POST /gripper/close   → {"ok": true}
    POST /recover         → Franka hibafeloldás reflex után
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
from rclpy.callback_groups import ReentrantCallbackGroup

try:  # a graph-lekérdezés helye rclpy-verziónként eltér
    from rclpy.action import get_action_names_and_types
except ImportError:  # pragma: no cover
    get_action_names_and_types = None
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2
from franka_msgs.action import Move as GripperMove, Grasp

try:  # a típusfeloldás verziófüggetlen módja
    from rosidl_runtime_py.utilities import get_interface, get_message
except ImportError:  # pragma: no cover - régebbi ROS2
    get_interface = get_message = None

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
GRIPPER_GRASP_FORCE = 10.0    # Newtons
GRIPPER_GRASP_SPEED = 0.05    # m/s
GRIPPER_GRASP_EPS   = 0.015   # metres inner/outer epsilon
# A cél szorítási szélesség. 0.0 + 15 mm epszilon azt jelenti: 0 és 15 mm közötti
# tárgyat fogad el sikeresnek — egy bábu NYAKA belefér, a TALPA (25-30 mm) nem.
GRIPPER_GRASP_WIDTH = 0.0     # metres

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

VELOCITY     = 0.05
ACCELERATION = 0.05

PORT = 8002

# A kalibrációs fájl helye a containerben. A run.sh a szkript mellé másolja,
# de korábban CSAK a chess_executor.py-t másolta be, így ez a fájl sosem volt
# ott, és a piece_grasp_heights felülírás néma halott kód volt.
_CAL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"),
    "/ros2_ws/calibration.json",
    "/ros2_ws/robot/calibration.json",
]
_CAL_PATH = next((p for p in _CAL_CANDIDATES if os.path.exists(p)), _CAL_CANDIDATES[0])

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
_recovery_client:      ActionClient | None = None
_lock = threading.Lock()
_cb_group = None

# Mit talált meg a rendszer indulásnál. A /status ezt adja vissza, hogy az
# első valódi robotpróbánál ne kelljen találgatni, mi van és mi nincs.
_caps: dict[str, Any] = {}

# Az utolsó hiba strukturáltan — a nyers str(e) kevés a Franka hibakódokhoz.
_last_error: dict[str, Any] | None = None

# A legutóbbi Franka állapotüzenet (ha van ilyen topic).
_franka_state: Any = None
_franka_state_t: float = 0.0

ACTION_TIMEOUT_S = 30.0   # egy gripper/recovery action felső határa


def _record_error(where: str, exc: BaseException | None = None, **extra: Any) -> dict[str, Any]:
    """Strukturált hibarögzítés. Ez kerül a HTTP válaszba és a /status-ba."""
    global _last_error
    info: dict[str, Any] = {"where": where, "t": time.time()}
    if exc is not None:
        info["error"] = f"{type(exc).__name__}: {exc}"
    info.update(extra)
    errs = _franka_errors()
    if errs:
        info["franka_errors"] = errs
    _last_error = info
    print(f"[chess_executor] HIBA {where}: {info}", flush=True)
    return info


def _wait_future(future, timeout_s: float, what: str):
    """Megvárja a future-t a HÁTTÉRBEN futó executorral.

    Nem hívunk spin_until_future_complete-et: a node-ot már pörgeti egy másik
    szál, és ugyanazt a node-ot két helyről pörgetni rclpy-ban nem megengedett
    (futásidejű hiba vagy holtpont). Itt csak várunk az eredményre.
    """
    deadline = time.monotonic() + timeout_s
    while not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{what}: nem érkezett válasz {timeout_s:.0f} s alatt")
        time.sleep(0.01)
    return future.result()


def _find_action(patterns: list[str]) -> tuple[str | None, str | None]:
    """Megkeresi az első action nevet, ami minden mintát tartalmaz.

    A franka_ros2 verziók között az action-nevek eltérnek (pl. a hibafeloldás
    hol /service_server/error_recovery, hol /franka_control/error_recovery),
    ezért nem nevet írunk be fixen, hanem futásidőben megkeressük.
    """
    if get_action_names_and_types is None:
        return None, None
    try:
        for name, types in get_action_names_and_types(node=_node):
            low = name.lower()
            if all(p in low for p in patterns):
                return name, (types[0] if types else None)
    except Exception as e:
        print(f"[chess_executor] action-lista nem olvasható: {e}", flush=True)
    return None, None


def _find_topic(patterns: list[str]) -> tuple[str | None, str | None]:
    try:
        for name, types in _node.get_topic_names_and_types():
            low = name.lower()
            if all(p in low for p in patterns):
                return name, (types[0] if types else None)
    except Exception as e:
        print(f"[chess_executor] topic-lista nem olvasható: {e}", flush=True)
    return None, None


def _setup_recovery() -> None:
    """Hibafeloldó action bekötése, ha a stack kínál ilyet.

    Reflex után a Franka hibaállapotban marad és MINDEN további parancsot
    elutasít. Enélkül egyetlen rossz mozgás után az egész rendszer halott.
    """
    global _recovery_client
    name, type_str = _find_action(["recover"])
    _caps["recovery_action"] = name
    _caps["recovery_type"] = type_str
    if name is None or type_str is None or get_interface is None:
        return
    try:
        _recovery_client = ActionClient(_node, get_interface(type_str), name,
                                        callback_group=_cb_group)
    except Exception as e:
        _caps["recovery_error"] = f"{type(e).__name__}: {e}"


def _setup_franka_state() -> None:
    """Feliratkozás a Franka állapot-topicra, ha van ilyen."""
    name, type_str = _find_topic(["franka", "state"])
    _caps["franka_state_topic"] = name
    _caps["franka_state_type"] = type_str
    if name is None or type_str is None or get_message is None:
        return

    def _on_state(msg):
        global _franka_state, _franka_state_t
        _franka_state = msg
        _franka_state_t = time.time()

    try:
        _node.create_subscription(get_message(type_str), name, _on_state, 1,
                                  callback_group=_cb_group)
    except Exception as e:
        _caps["franka_state_error"] = f"{type(e).__name__}: {e}"


def _franka_errors() -> list[str]:
    """A Franka üzenetből kiszedi az aktív hibajelzőket.

    A mezőnevek verziónként változnak, ezért nem nevesítjük őket: minden
    "error" nevű almezőben megkeressük az igazra állított logikai jelzőket.
    """
    msg = _franka_state
    if msg is None:
        return []
    out: list[str] = []
    try:
        for field in getattr(msg, "get_fields_and_field_types", dict)():
            if "error" not in field.lower():
                continue
            sub = getattr(msg, field, None)
            if isinstance(sub, bool):
                if sub:
                    out.append(field)
            elif hasattr(sub, "get_fields_and_field_types"):
                for name in sub.get_fields_and_field_types():
                    if getattr(sub, name, False) is True:
                        out.append(f"{field}.{name}")
    except Exception:
        pass
    return out


def _wait_ready(timeout_s: float = 30.0) -> dict[str, Any]:
    """Megvárja, hogy a robot tényleg vezérelhető legyen.

    A korábbi fix `time.sleep(1.5)` nem garantált semmit: ha a /joint_states
    még nem érkezett meg, a tervezés elavult kiinduló állapotból indul — ami
    pontosan az a hibaosztály, amit üldözünk.
    """
    deadline = time.monotonic() + timeout_s
    ready = {"joint_state": False, "tf": False, "gripper_move": False,
             "gripper_grasp": False, "recovery": _recovery_client is None}

    while time.monotonic() < deadline and not all(ready.values()):
        if not ready["joint_state"]:
            js = getattr(_moveit2, "joint_state", None)
            ready["joint_state"] = js is not None
        if not ready["tf"]:
            try:
                _tf_buf.lookup_transform(_BASE_FRAME, _EEF_FRAME, rclpy.time.Time())
                ready["tf"] = True
            except Exception:
                pass
        for key, client in (("gripper_move", _gripper_move_client),
                            ("gripper_grasp", _gripper_grasp_client),
                            ("recovery", _recovery_client)):
            if not ready[key] and client is not None:
                ready[key] = client.server_is_ready()
        if all(ready.values()):
            break
        time.sleep(0.1)

    _caps["ready"] = ready
    missing = [k for k, v in ready.items() if not v]
    if missing:
        print(f"[chess_executor] FIGYELEM: nem áll készen: {', '.join(missing)} "
              f"({timeout_s:.0f} s alatt) — a /status mutatja a részleteket", flush=True)
    return ready


def _init_robot() -> None:
    global _node, _moveit2, _tf_buf, _gripper_move_client, _gripper_grasp_client, _cb_group

    rclpy.init()
    _node = Node("chess_executor")
    _cb_group = ReentrantCallbackGroup()

    # 2 szál kevés volt a MoveIt2 + TF + action kliensek mellé: ha minden
    # callback ugyanazon a két szálon versenyez, a gripper válasza megvárathatja
    # a joint_state frissítést.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        _moveit2 = MoveIt2(
            node=_node,
            joint_names=_JOINT_NAMES,
            base_link_name=_BASE_FRAME,
            end_effector_name=_EEF_FRAME,
            group_name=_GROUP_NAME,
            callback_group=_cb_group,
        )
    except TypeError:
        # régebbi pymoveit2: nincs callback_group paraméter
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

    _gripper_move_client  = ActionClient(_node, GripperMove, "/fr3_gripper/move",
                                         callback_group=_cb_group)
    _gripper_grasp_client = ActionClient(_node, Grasp, "/fr3_gripper/grasp",
                                         callback_group=_cb_group)

    _load_piece_grasp_heights()
    _setup_recovery()
    _setup_franka_state()

    # Mit tud a pymoveit2 ebben a verzióban? Ezt nem tudjuk előre, ezért
    # megnézzük és kiírjuk — az első valódi próbánál ez sok kört megspórol.
    _caps["moveit2_attrs"] = sorted(
        a for a in ("motion_suceeded", "motion_succeeded", "compute_cartesian_path",
                    "move_to_pose", "wait_until_executed", "joint_state", "execute")
        if hasattr(_moveit2, a)
    )
    _caps["calibration_file"] = _CAL_PATH if os.path.exists(_CAL_PATH) else None

    _wait_ready()
    print(f"[chess_executor] Robot kész. HTTP szerver: port {PORT}", flush=True)
    print(f"[chess_executor] Felderítés: {json.dumps(_caps, default=str)}", flush=True)


def _get_position(timeout_s: float = 3.0) -> tuple[float, float, float]:
    """Aktuális TCP pozíció mm-ben.

    A TF buffer indulás után nem azonnal telik fel, és most már a legelső
    mozgás ELŐTT is szükség van a pozícióra (_approach), ezért rövid ideig
    várunk rá ahelyett, hogy egy hideg buffer megbuktassa az első parancsot.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while True:
        try:
            t = _tf_buf.lookup_transform(_BASE_FRAME, _EEF_FRAME, rclpy.time.Time())
            tr = t.transform.translation
            return (tr.x * 1000.0, tr.y * 1000.0, tr.z * 1000.0)
        except Exception as e:
            last_err = e
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Nem olvasható a robot pozíciója ({_BASE_FRAME} → {_EEF_FRAME}) "
                    f"{timeout_s:.0f} s alatt: {last_err}"
                ) from last_err
            time.sleep(0.05)


# ── Gripper ──────────────────────────────────────────────────────────────────

def _send_action(client: ActionClient, goal, what: str, timeout_s: float = ACTION_TIMEOUT_S):
    """Action küldése és a VALÓDI eredmény kiolvasása.

    Eddig a válasz eldobódott, így egy elutasított vagy elbukott gripper-parancs
    észrevétlen maradt, és a szekvencia bábu nélkül ment tovább.
    """
    if client is None:
        raise RuntimeError(f"{what}: nincs kliens (a felderítés nem találta meg)")
    if not client.server_is_ready() and not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError(f"{what}: az action szerver nem érhető el")

    handle = _wait_future(client.send_goal_async(goal), timeout_s, f"{what} (goal)")
    if handle is None or not handle.accepted:
        raise RuntimeError(f"{what}: az action szerver elutasította a parancsot")

    result = _wait_future(handle.get_result_async(), timeout_s, f"{what} (result)")
    status = getattr(result, "status", None)
    payload = getattr(result, "result", None)
    # GoalStatus.STATUS_SUCCEEDED == 4
    if status is not None and status != 4:
        raise RuntimeError(f"{what}: sikertelen (status={status}, "
                           f"error={getattr(payload, 'error', None)})")
    if payload is not None and getattr(payload, "success", True) is False:
        raise RuntimeError(f"{what}: sikertelen ({getattr(payload, 'error', 'ismeretlen ok')})")
    return payload


def _gripper_open(width: float = GRIPPER_OPEN_WIDTH) -> None:
    _send_action(_gripper_move_client, GripperMove.Goal(width=width, speed=0.1),
                 "gripper nyitás")


def _gripper_grasp(width: float = GRIPPER_GRASP_WIDTH) -> None:
    goal = Grasp.Goal(
        width=width,
        speed=GRIPPER_GRASP_SPEED,
        force=GRIPPER_GRASP_FORCE,
        epsilon=Grasp.Goal.GraspEpsilon(inner=GRIPPER_GRASP_EPS, outer=GRIPPER_GRASP_EPS),
    )
    _send_action(_gripper_grasp_client, goal, "gripper fogás")


def _recover() -> dict[str, Any]:
    """Franka hibafeloldás. Reflex után enélkül semmi nem működik tovább."""
    if _recovery_client is None:
        return {"ok": False, "reason": "nincs hibafeloldó action ezen a stacken",
                "searched": _caps.get("recovery_action")}
    try:
        goal_cls = get_interface(_caps["recovery_type"]).Goal
        _send_action(_recovery_client, goal_cls(), "hibafeloldás")
        return {"ok": True, "action": _caps.get("recovery_action")}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


# ── Mozgástervezés ────────────────────────────────────────────────────────────
#
# ALAPSZABÁLY: a MoveIt a Cartesian waypointokat CÉLPONTKÉNT értelmezi, az
# aktuális pózból indulva. A waypoint-listába ezért SOHA nem szabad beletenni
# azt a pontot, ahol a kar éppen áll — abból a trajektória első szegmense
# ugrássá válik, amit a Franka nem folytonos indulási sebességként utasít el
# (reflex, a kar bemerevedik), vagy a MoveIt már a start-toleranciánál eldob.

_APPROACH_TOL_MM = 3.0   # ennél közelebb már ott vagyunk, nincs ráállás


def _moveit_outcome() -> dict[str, Any]:
    """Amit a pymoveit2 elárul a végrehajtásról.

    A verziók között eltér, mit tesz elérhetővé (és a `motion_suceeded` névben
    elgépelés van), ezért nem egy névre építünk: összeszedjük, ami van.
    """
    out: dict[str, Any] = {}
    for attr in ("motion_suceeded", "motion_succeeded"):
        if hasattr(_moveit2, attr):
            out["motion_succeeded"] = getattr(_moveit2, attr)
            break
    return out


def _execute_planned(target_m: list[float], what: str) -> None:
    """Tervezett (szabad-tér) mozgás egyetlen pózba, kimenetel-ellenőrzéssel."""
    _moveit2.move_to_pose(position=list(target_m), quat_xyzw=_DOWN_QUAT)
    executed = _moveit2.wait_until_executed()
    outcome = _moveit_outcome()
    if executed is False or outcome.get("motion_succeeded") is False:
        raise RuntimeError(f"{what}: a MoveIt nem hajtotta végre a mozgást "
                           f"(wait_until_executed={executed}, {outcome})")


def _move_to(x_mm: float, y_mm: float, z_mm: float) -> None:
    """Egyszerű pose-alapú mozgás — kalibráló endpointhoz."""
    target = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    _execute_planned(target, "pose-mozgás")
    _assert_reached(target, tolerance_mm=10.0)


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


def _run_cartesian(waypoints: list[list[float]]) -> None:
    """Egyenes vonalú Cartesian pálya a MEGADOTT waypointokon (méterben).

    A lista az aktuális pózból induló CÉLPONTOKAT tartalmazza. Az alapszabályt
    itt, egyetlen helyen kényszerítjük ki: kiszűrjük azokat a waypointokat,
    amik gyakorlatilag ott vannak, ahol a kar már áll. Ezek nem mozgások,
    hanem nulla hosszú szegmensek — és épp ezekből lesz az indulási ugrás.
    """
    if not waypoints:
        return

    prev_mm = _get_position()
    targets: list[list[float]] = []
    for wp in waypoints:
        wp_mm = (wp[0] * 1000.0, wp[1] * 1000.0, wp[2] * 1000.0)
        if math.dist(prev_mm, wp_mm) <= _APPROACH_TOL_MM:
            continue
        targets.append(list(wp))
        prev_mm = wp_mm

    if targets:
        try:
            _moveit2.compute_cartesian_path(
                targets,
                quat_xyzw=_DOWN_QUAT,
                max_step=0.01,
            )
            executed = _moveit2.wait_until_executed()
            outcome = _moveit_outcome()
            if executed is False or outcome.get("motion_succeeded") is False:
                raise RuntimeError(
                    f"Cartesian pálya nem hajtódott végre "
                    f"({len(targets)} waypoint, wait_until_executed={executed}, {outcome})"
                )
        except (AttributeError, TypeError):
            # ez a pymoveit2 verzió nem ismeri a compute_cartesian_path-t
            for wp in targets:
                _execute_planned(wp, "Cartesian tartalék (pose-onként)")

    _assert_reached(waypoints[-1])


def _approach(x_mm: float, y_mm: float, z_m: float) -> None:
    """Biztonságos ráállás egy pont fölé ISMERETLEN kiindulási pózból.

    Ez az egyetlen hely, ahol a kar nem tudja, honnan indul. Ezért nem
    szabad szabad-tér tervezésre bízni (az OMPL a bábuk közé is lemerülhet):
    mindig felemelkedik Z_TRAVEL magasságra, ott halad vízszintesen, és csak
    a cél XY fölött ereszkedik le. Ha már ott van, nem csinál semmit.
    """
    cx, cy, cz = _get_position()
    z_mm = z_m * 1000.0
    if (abs(cx - x_mm) <= _APPROACH_TOL_MM
            and abs(cy - y_mm) <= _APPROACH_TOL_MM
            and abs(cz - z_mm) <= _APPROACH_TOL_MM):
        return

    tx, ty = x_mm / 1000.0, y_mm / 1000.0
    waypoints: list[list[float]] = []
    if cz < Z_TRAVEL * 1000.0 - _APPROACH_TOL_MM:
        # 1. függőleges emelkedés a JELENLEGI XY fölött — vízszintesen még
        #    nem mozdulunk, amíg nem vagyunk a bábuk fölött
        waypoints.append([cx / 1000.0, cy / 1000.0, Z_TRAVEL])
    # 2. vízszintes áthelyezés a cél XY fölé, végig Z_TRAVEL-en vagy fölötte
    waypoints.append([tx, ty, Z_TRAVEL])
    # 3. leereszkedés a kért magasságra
    waypoints.append([tx, ty, z_m])
    _run_cartesian(waypoints)


def _vertical_path(x_mm: float, y_mm: float, z_start: float, z_end: float) -> None:
    """Egyenes fel/le mozgás rögzített XY pozícióban.

    A z_start nem célpont, hanem elvárás: a hívó szerint itt áll a kar.
    Ha mégsem — például a szekvencia legelején —, előbb biztonságosan
    ráállunk, és csak utána indul az egyenes szakasz.
    """
    _approach(x_mm, y_mm, z_start)
    _run_cartesian([[x_mm / 1000.0, y_mm / 1000.0, z_end]])


def _arc_path(from_xy: tuple, to_xy: tuple) -> None:
    """
    Sima parabolikus ív from_xy-tól to_xy-ig, mindkét végpont Z_LIFT magasságon.
    Az ív csúcsa Z_TRAVEL. A parabola paraméteres formája biztosítja, hogy
    t=0 és t=1-nél dz/dt=0, azaz a robot vízszintesen indul és érkezik Z_LIFT-en.

    A t=0 pont maga a kiindulás, ezért NEM kerül be a waypointok közé.
    """
    _approach(from_xy[0], from_xy[1], Z_LIFT)

    fx, fy = from_xy[0] / 1000.0, from_xy[1] / 1000.0
    tx, ty = to_xy[0] / 1000.0,   to_xy[1] / 1000.0
    waypoints = []
    for i in range(1, ARC_WAYPOINTS + 1):
        t = i / ARC_WAYPOINTS
        x = fx + t * (tx - fx)
        y = fy + t * (ty - fy)
        # parabola: z(t) = Z_LIFT + 4*(Z_TRAVEL - Z_LIFT)*t*(1-t)
        # t=0 és t=1-nél: z = Z_LIFT;  t=0.5-nél: z = Z_TRAVEL
        z = Z_LIFT + 4.0 * (Z_TRAVEL - Z_LIFT) * t * (1.0 - t)
        waypoints.append([x, y, z])
    _run_cartesian(waypoints)


def _pick_and_place(from_xy: tuple, to_xy: tuple, piece: str = "P") -> None:
    fx, fy = from_xy
    tx, ty = to_xy
    z_pick = Z_PICK + PIECE_GRASP_HEIGHT.get(piece.upper(), 0.0)
    _approach(fx, fy, Z_LIFT)                    # 0. fázis: biztonságos ráállás
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

    def _status(self) -> dict:
        """Minden, amiből egy hibás futás utólag megérthető."""
        pos = None
        pos_error = None
        try:
            x, y, z = _get_position(timeout_s=0.5)
            pos = {"x": x, "y": y, "z": z}
        except Exception as e:
            pos_error = f"{type(e).__name__}: {e}"

        js = getattr(_moveit2, "joint_state", None) if _moveit2 is not None else None
        return {
            "ok": True,
            "position": pos,
            "position_error": pos_error,
            "joint_state_seen": js is not None,
            "franka_errors": _franka_errors(),
            "franka_state_age_s": (time.time() - _franka_state_t) if _franka_state_t else None,
            "capabilities": _caps,
            "last_error": _last_error,
        }

    def do_GET(self):
        if self.path == "/health":
            # A /health szándékosan olcsó és mindig 200 — a részletek a /status-ban.
            self._send(200, {"ok": True, "errors_active": bool(_franka_errors())})
        elif self.path == "/status":
            try:
                self._send(200, self._status())
            except Exception as e:
                self._send(500, _record_error("/status", e))
        elif self.path == "/position":
            try:
                x, y, z = _get_position()
                self._send(200, {"x": x, "y": y, "z": z})
            except Exception as e:
                self._send(500, _record_error("/position", e))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        with _lock:
            try:
                if self.path == "/execute":
                    desc = self._read_json()
                    try:
                        _execute_move(desc)
                    except Exception as e:
                        # A hibafeloldás lefut, hogy a robot használható maradjon,
                        # DE a lépést nem ismételjük meg magunktól: félbeszakadt
                        # szekvenciában a megfogóban lehet egy bábu, és a vak
                        # újrapróbálás onnan többet ront, mint javít.
                        info = _record_error("/execute", e, descriptor_type=desc.get("type"))
                        info["recovery"] = _recover()
                        self._send(500, info)
                        return
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
                elif self.path == "/recover":
                    result = _recover()
                    self._send(200 if result.get("ok") else 500, result)
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, _record_error(self.path, e))


if __name__ == "__main__":
    _init_robot()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[chess_executor] Listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
