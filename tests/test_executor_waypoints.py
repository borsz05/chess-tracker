"""A chess_executor mozgástervezésének offline ellenőrzése — robot nélkül.

A lényegi invariáns, amit véd:

    Egyetlen Cartesian szakasz első waypointja sem lehet az a pont, ahol a
    kar éppen áll.

A MoveIt a waypointokat célpontként értelmezi az aktuális pózból indulva, így
egy ilyen waypoint ugrássá teszi a trajektória első szegmensét — ezt a Franka
nem folytonos indulási sebességként utasítja el (reflex), vagy a MoveIt már a
start-toleranciánál eldobja. Emiatt nem mozdult meg a kar egyetlen próbánál sem.

A modul rclpy/pymoveit2/franka_msgs importokat tartalmaz, amik a hoston nincsenek
telepítve (a Docker containerben futnak), ezért ezeket kicseréljük.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

_EXECUTOR_PATH = Path(__file__).resolve().parents[1] / "franka" / "chess_executor.py"


def _install_stubs() -> None:
    """Minimális álmodulok, hogy a chess_executor importálható legyen."""
    def mod(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    class _Time:
        def __init__(self, *a, **k): pass

    rclpy = mod("rclpy", init=lambda *a, **k: None)
    rclpy.time = mod("rclpy.time", Time=_Time)
    mod("rclpy.action", ActionClient=object)
    mod("rclpy.executors", MultiThreadedExecutor=object)
    mod("rclpy.node", Node=object)
    mod("tf2_ros", Buffer=object, TransformListener=object)
    mod("pymoveit2", MoveIt2=object)

    franka_msgs = mod("franka_msgs")
    action = mod("franka_msgs.action", Move=object, Grasp=object)
    franka_msgs.action = action


def _load_executor():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("chess_executor_under_test", _EXECUTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMoveIt2:
    """Rögzíti a kiadott szakaszokat, és követi, hol áll a kar."""

    def __init__(self, start_m: tuple[float, float, float]) -> None:
        self.position = list(start_m)
        self.segments: list[list[list[float]]] = []   # Cartesian szakaszok
        self.pose_moves: list[list[float]] = []       # move_to_pose hívások
        self.max_velocity = 0.0
        self.max_acceleration = 0.0

    def compute_cartesian_path(self, waypoints, quat_xyzw=None, max_step=None):
        assert waypoints, "üres Cartesian szakaszt nem szabad kiadni"
        self.segments.append([list(w) for w in waypoints])
        self.position = list(waypoints[-1])

    def move_to_pose(self, position=None, quat_xyzw=None):
        self.pose_moves.append(list(position))
        self.position = list(position)

    def wait_until_executed(self):
        return True


@pytest.fixture
def ex():
    return _load_executor()


def _wire(ex, start_m):
    """Fake MoveIt2 és a hozzá kötött pozíció-lekérdezés beszerelése."""
    fake = FakeMoveIt2(start_m)
    ex._moveit2 = fake
    ex._get_position = lambda *a, **k: tuple(v * 1000.0 for v in fake.position)
    ex._gripper_grasp = lambda: None
    ex._gripper_open = lambda: None
    return fake


def _run_and_trace(ex, fake, fn):
    """Lefuttat egy mozgást úgy, hogy közben minden szakasz elejéhez
    eltárolja, hol állt a kar a szakasz indulásakor."""
    starts: list[list[float]] = []
    real = fake.compute_cartesian_path

    def spy(waypoints, **kw):
        starts.append(list(fake.position))
        return real(waypoints, **kw)

    fake.compute_cartesian_path = spy
    fn()
    return list(zip(starts, fake.segments))


def _dist_mm(a, b):
    return math.dist([v * 1000.0 for v in a], [v * 1000.0 for v in b])


# ── A központi invariáns ─────────────────────────────────────────────────────

START_POSES = [
    (0.30, 0.10, 0.35),    # tipikus nyugalmi póz, magasan
    (0.45, -0.175, 0.12),  # egy előző lépés vége, Z_LIFT-en
    (0.10, 0.175, 0.03),   # lent maradt a kar
    (0.45, -0.175, 0.20),  # pont Z_TRAVEL-en
]


@pytest.mark.parametrize("start", START_POSES)
def test_no_segment_starts_where_the_arm_already_is(ex, start):
    fake = _wire(ex, start)
    trace = _run_and_trace(
        ex, fake,
        lambda: ex._pick_and_place((450.0, -175.0), (100.0, 175.0), "N"),
    )
    assert trace, "nem futott egyetlen Cartesian szakasz sem"

    for i, (pose_at_start, waypoints) in enumerate(trace):
        d = _dist_mm(pose_at_start, waypoints[0])
        assert d > ex._APPROACH_TOL_MM, (
            f"{i}. szakasz első waypointja gyakorlatilag az aktuális póz "
            f"({d:.2f} mm) — ebből indulási ugrás lesz"
        )


@pytest.mark.parametrize("start", START_POSES)
def test_horizontal_travel_never_dips_below_travel_height(ex, start):
    """Ismeretlen pózból a ráállás soha nem söpör végig a bábuk között."""
    fake = _wire(ex, start)
    trace = _run_and_trace(
        ex, fake,
        lambda: ex._pick_and_place((450.0, -175.0), (100.0, 175.0), "P"),
    )

    # csak a legelső szakasz a "ráállás ismeretlen pózból"; utána a szekvencia
    # minden pontja ismert és szándékosan halad Z_LIFT környékén
    pose_at_start, waypoints = trace[0]
    prev = list(pose_at_start)
    for wp in waypoints:
        moved_horizontally = math.dist(prev[:2], wp[:2]) > 1e-6
        if moved_horizontally:
            assert min(prev[2], wp[2]) >= ex.Z_TRAVEL - 1e-9, (
                f"vízszintes mozgás {min(prev[2], wp[2]):.3f} m magasságban, "
                f"Z_TRAVEL={ex.Z_TRAVEL} m alatt"
            )
        prev = list(wp)


def test_approach_is_skipped_when_already_there(ex):
    """A szekvencián belül — ahol a kar tudja, hol áll — nincs felesleges ráállás."""
    fake = _wire(ex, (0.450, -0.175, ex.Z_LIFT))
    ex._approach(450.0, -175.0, ex.Z_LIFT)
    assert fake.segments == [], "felesleges ráállás ott, ahol a kar már áll"


def test_arc_omits_its_own_starting_point(ex):
    fake = _wire(ex, (0.450, -0.175, ex.Z_LIFT))
    ex._arc_path((450.0, -175.0), (100.0, 175.0))

    assert len(fake.segments) == 1, "a ráállásnak ki kellett volna maradnia"
    waypoints = fake.segments[0]
    assert len(waypoints) == ex.ARC_WAYPOINTS, "a t=0 pont nem lehet waypoint"
    assert _dist_mm(waypoints[0], (0.450, -0.175, ex.Z_LIFT)) > ex._APPROACH_TOL_MM
    # az ív a cél fölött, Z_LIFT-en ér véget
    assert waypoints[-1] == pytest.approx([0.100, 0.175, ex.Z_LIFT])


def test_vertical_path_emits_only_the_end_point(ex):
    fake = _wire(ex, (0.450, -0.175, ex.Z_LIFT))
    ex._vertical_path(450.0, -175.0, ex.Z_LIFT, ex.Z_PICK)

    assert len(fake.segments) == 1
    assert fake.segments[0] == [[0.450, -0.175, ex.Z_PICK]]


def test_full_sequence_ends_above_the_destination(ex):
    fake = _wire(ex, (0.30, 0.10, 0.35))
    ex._pick_and_place((450.0, -175.0), (100.0, 175.0), "Q")
    assert fake.position == pytest.approx([0.100, 0.175, ex.Z_LIFT])


def test_grasp_height_matches_the_piece(ex):
    """A megfogás magassága a bábu szerint változik — a fázissorrend ne csússzon el."""
    for piece, extra in ex.PIECE_GRASP_HEIGHT.items():
        fake = _wire(ex, (0.30, 0.10, 0.35))
        ex._pick_and_place((450.0, -175.0), (100.0, 175.0), piece)
        lowest = min(wp[2] for seg in fake.segments for wp in seg)
        assert lowest == pytest.approx(min(ex.Z_PICK + extra, ex.Z_PLACE)), piece
