"""Háttér-őrszem: mikor kiált kamera-elmozdulást és mikor hallgat.

A küszöbök mögötti mérés (ezen a gépen, data_fen/frames):
  - álló táblán egymást követő detektálások eltérése p50 0.37 px, p95 1.01 px
    (a mezőköz 114 px, azaz ~0.9%)
  - egy valóban elmozdult tábla 37.6 px-et ugrott (33%)
  - kicsinyített (672 px széles) detektálás 184/184 képen sikeres volt,
    43-95 ms, és a remegése nem romlott
"""
from __future__ import annotations

import numpy as np
import pytest

import vision.pipeline.board_watchdog as wd_mod
from vision.pipeline.board_detector import DetectionResult
from vision.pipeline.board_watchdog import BoardWatchdog, WatchdogParams, mean_nearest_shift

CELL = 114.0
SCALE = WatchdogParams().check_width_px / 1920.0   # az ellenőrzés kicsinyítése


def _grid(dx: float = 0.0, dy: float = 0.0, scale: float = 1.0):
    """8x8 mezőközép-rács. dx/dy TELJES felbontású pixelben értendő; a scale
    azt modellezi, hogy a kicsinyített ellenőrzés kisebb koordinátákat ad vissza."""
    return [[((c * CELL + dx) * scale, (r * CELL + dy) * scale) for c in range(8)]
            for r in range(8)]


def _det(dx: float = 0.0, dy: float = 0.0, ok: bool = True,
         scale: float = 1.0) -> DetectionResult:
    g = _grid(dx, dy, scale)
    return DetectionResult(ok=ok, M=np.eye(3), bbox_warp=None,
                           centers_warp=g, centers_img=g)


def _small(dx: float = 0.0, dy: float = 0.0, ok: bool = True) -> DetectionResult:
    """Amit a kicsinyített ellenőrzés adna vissza."""
    return _det(dx, dy, ok, scale=SCALE)


class FakeStabilizer:
    def __init__(self, last_motion_t=None):
        self.last_motion_t = last_motion_t


class FakeTracker:
    def __init__(self, initialized=True, centers=None, last_motion_t=None):
        self.initialized = initialized
        self.stabilizer = FakeStabilizer(last_motion_t)
        self._centers = centers if centers is not None else _grid()
        self.offered: list[tuple] = []
        self.align_calls = 0

    def detection_snapshot(self):
        return self._centers, np.zeros((8, 8), np.int32)

    def align_offered_detection(self, det, frame_bgr):
        self.align_calls += 1
        return "rot0"

    def offer_detection(self, det, alignment):
        self.offered.append((det, alignment))


@pytest.fixture
def watchdog(monkeypatch):
    """Az őrszem valódi döntési logikája, ál-detektorral és ál-trackerrel."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    state = {"tracker": FakeTracker(), "small": _small(), "full": _det(), "now": 1000.0}

    def _detect(gray, *, cell, inner_pad_ratio):
        # a kicsinyített ellenőrzés kisebb cellával fut
        return state["small"] if cell < 90 else state["full"]

    monkeypatch.setattr(wd_mod, "detect_board_on_frame", _detect)

    cfg = type("Cfg", (), {"cell": 96, "inner_pad_ratio": 0.06})()
    w = BoardWatchdog(
        tracker_getter=lambda: state["tracker"],
        frame_getter=lambda: frame,
        cfg=cfg,
        params=WatchdogParams(confirmations=2, quiet_after_motion_s=1.0),
        clock=lambda: state["now"],
    )
    return w, state


# ── a mérőszám ───────────────────────────────────────────────────────────────

def test_shift_is_independent_of_grid_order():
    """A detektor rács-orientációja nem garantált, ezért indexre hasonlítani
    félrevezető lenne — a mérőszám a legközelebbi szomszédhoz mér."""
    a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rotated = a[[3, 2, 1, 0]]
    assert mean_nearest_shift(rotated, a) == pytest.approx(0.0)


def test_shift_measures_a_real_translation():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert mean_nearest_shift(a + np.array([5.0, 0.0]), a) == pytest.approx(5.0)


# ── mikor hallgat ────────────────────────────────────────────────────────────

def test_no_check_before_initialisation(watchdog):
    """Init közben úgyis minden képkockán fut a detektálás."""
    w, state = watchdog
    state["tracker"].initialized = False
    assert "init" in w.check_once()
    assert w.checks == 0


def test_no_check_while_something_moves_over_the_board(watchdog):
    """Egy kéz eltakarhatja a sarkokat és hamis elmozdulást mutatna."""
    w, state = watchdog
    state["tracker"].stabilizer.last_motion_t = state["now"] - 0.2
    assert "mozgás" in w.check_once()
    assert w.checks == 0


def test_check_resumes_once_the_board_is_quiet_again(watchdog):
    w, state = watchdog
    state["tracker"].stabilizer.last_motion_t = state["now"] - 5.0
    assert w.check_once().startswith("változatlan")


def test_a_still_board_is_reported_unchanged(watchdog):
    """A mért 1 px-es remegés bőven a küszöb alatt van."""
    w, state = watchdog
    state["small"] = _small(dx=1.0, dy=0.5)
    assert w.check_once().startswith("változatlan")
    assert state["tracker"].offered == []


def test_a_failed_check_is_not_an_event(watchdog):
    """Takarás vagy rossz fény: nem tudunk semmit, tehát nem állítunk semmit."""
    w, state = watchdog
    state["small"] = _small(ok=False)
    assert w.check_once() == "nem találta a táblát"
    assert state["tracker"].offered == []


# ── mikor kiált ──────────────────────────────────────────────────────────────

def test_one_outlier_alone_does_not_trigger_a_redetect(watchdog):
    """Egyetlen kiugró detektálás nem elég — a valódi elmozdulás megmarad."""
    w, state = watchdog
    state["small"] = _small(dx=40.0)
    assert "gyanú 1/2" in w.check_once()
    assert state["tracker"].offered == []


def test_a_persistent_shift_triggers_the_redetect(watchdog):
    w, state = watchdog
    state["small"] = _small(dx=40.0)
    w.check_once()
    msg = w.check_once()

    assert msg.startswith("újradetektálva")
    assert len(state["tracker"].offered) == 1
    det, alignment = state["tracker"].offered[0]
    assert alignment == "rot0" and det is state["full"]
    assert state["tracker"].align_calls == 1
    assert w.swaps == 1


def test_a_shift_that_goes_away_resets_the_streak(watchdog):
    w, state = watchdog
    state["small"] = _small(dx=40.0)
    w.check_once()
    state["small"] = _small(dx=0.5)
    w.check_once()
    state["small"] = _small(dx=40.0)

    assert "gyanú 1/2" in w.check_once()
    assert state["tracker"].offered == []


def test_a_failed_check_does_not_cancel_a_pending_suspicion(watchdog):
    """A sikertelen ellenőrzés nem cáfolja az elmozdulást, csak nem tud róla."""
    w, state = watchdog
    state["small"] = _small(dx=40.0)
    w.check_once()
    state["small"] = _small(ok=False)
    w.check_once()
    state["small"] = _small(dx=40.0)

    assert w.check_once().startswith("újradetektálva")


def test_a_failed_full_redetect_is_reported_without_swapping(watchdog):
    """Ha a teljes felbontású detektálás sem sikerül, a régi rács marad —
    jobb egy elavult tábla, mint semmilyen."""
    w, state = watchdog
    state["small"] = _small(dx=40.0)
    state["full"] = _det(ok=False)
    w.check_once()
    msg = w.check_once()

    assert "nem sikerült" in msg
    assert state["tracker"].offered == []
    assert w.swaps == 0


def test_the_loop_never_dies_on_an_exception(watchdog):
    """Az őrszem sosem viheti magával a rendszert."""
    w, state = watchdog
    w._tracker_getter = lambda: (_ for _ in ()).throw(RuntimeError("bumm"))
    w._stop.set()
    w._loop()          # nem dob
    w._stop.clear()
    w._tracker_getter = lambda: state["tracker"]


# ── a lépés megerősítését nem szabad megzavarni ──────────────────────────────
#
# Minden csere note_motion-t vált ki, ami újraindítja a stabilizátor statikus
# ablakát. Ha ez egy lépés megerősítése közben történik, az elfogadás
# késleltetődik — ez az egyetlen mód, ahogy az őrszem lassíthatna.


class MoveInFlightStabilizer:
    def __init__(self, candidate, reference):
        self.last_motion_t = None
        self.candidate = candidate
        self.reference = reference


def test_no_swap_while_a_move_is_being_confirmed(watchdog):
    w, state = watchdog
    ref = [[0] * 8 for _ in range(8)]
    cand = [row[:] for row in ref]
    cand[3][4] = 1
    state["tracker"].stabilizer = MoveInFlightStabilizer(cand, ref)
    state["small"] = _small(dx=40.0)

    assert "lépés megerősítése" in w.check_once()
    assert state["tracker"].offered == []


def test_swap_resumes_once_the_candidate_matches_the_reference(watchdog):
    w, state = watchdog
    ref = [[0] * 8 for _ in range(8)]
    state["tracker"].stabilizer = MoveInFlightStabilizer([row[:] for row in ref], ref)
    state["small"] = _small(dx=40.0)

    w.check_once()
    assert w.check_once().startswith("újradetektálva")


# ── kézi kérés ('d') ─────────────────────────────────────────────────────────

def test_a_manual_request_redetects_without_any_shift(watchdog):
    """A 'd' feltétel nélkül újradetektál — de a munka így is a háttérszálon."""
    w, state = watchdog
    state["small"] = _small()                     # semmi nem mozdult
    msg = w.check_once(forced=True)

    assert msg.startswith("kézi kérés")
    assert len(state["tracker"].offered) == 1
    assert w.swaps == 1


def test_a_manual_request_ignores_the_motion_guard(watchdog):
    w, state = watchdog
    state["tracker"].stabilizer.last_motion_t = state["now"]
    assert w.check_once(forced=True).startswith("kézi kérés")
    assert len(state["tracker"].offered) == 1


def test_a_manual_request_still_needs_an_initialised_tracker(watchdog):
    w, state = watchdog
    state["tracker"].initialized = False
    assert "init" in w.check_once(forced=True)
    assert state["tracker"].offered == []


# ── statisztika az időzítési exporthoz ───────────────────────────────────────

def test_stats_report_what_the_watchdog_actually_did(watchdog):
    w, state = watchdog
    state["small"] = _small(dx=1.0)
    w.check_once()
    state["tracker"].initialized = False
    w.check_once()
    state["tracker"].initialized = True
    state["small"] = _small(dx=40.0)
    w.check_once()
    w.check_once()

    s = w.stats()
    assert s["checks"] == 3 and s["skips"] == 1 and s["swaps"] == 1
    assert s["shift_max_px"] > 30 and s["shift_p50_px"] > 0
    assert "status" not in s, "a szöveges állapot nem való CSV-be"
