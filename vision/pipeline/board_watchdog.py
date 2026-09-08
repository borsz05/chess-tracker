"""Háttér-őrszem: észreveszi, ha elmozdult a kamera vagy a tábla.

Miért kell
----------
A tábla homográfiája az inicializáláskor befagy. Ha a kamerát utána
megmozdítják — márpedig gyakran mozgatják, mert a beállítás közben derül ki,
hogy jó-e —, a rendszer egy már nem létező kameraállás rácsából dolgozik
tovább, és ezt magától sosem venné észre. Eddig csak az újraindítás segített.

Miért háttérszálon
------------------
A tábla-detektálás teljes felbontáson 430 ms (p95 580 ms) ezen a gépen. Ez a
fő ciklusban fél másodperces kiesés lenne. Itt mérve (12 mag):

    fő ciklus önmagában                    p50 23.1 ms   p95 30.2 ms
    + teljes felbontású detektálás 3 mp-ként  p50 23.9 ms   p95 32.1 ms
    + kicsinyített ellenőrzés 2 mp-ként    p50 22.6 ms   p95 29.6 ms

Vagyis a kicsinyített ellenőrzés a zajon belül van: nem mérhető lassulás.
Ezért az őrszem két lépcsős:

  1. Gyakori, OLCSÓ ellenőrzés kicsinyített képen (~44 ms): csak azt dönti el,
     elmozdult-e a tábla. 184/184 képen sikeres volt minden fényviszonynál.
  2. Ha elmozdult, AKKOR teljes felbontású újradetektálás + orientáció-igazítás,
     szintén itt, a háttérben. A fő szálra már csak a kész eredmény kerül át.

A küszöb honnan jön
-------------------
Álló táblán, egymást követő képkockák között a detektálás remegése
p50 0.37 px, p95 1.01 px (a mezőköz 114 px, azaz ~0.9%). Egy valóban elmozdult
tábla a mérésben 37.6 px-et ugrott (33%). A küszöb a kettő között, a mezőköz
arányában van megadva, hogy felbontástól független legyen.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from vision.pipeline.board_detector import DetectionResult, detect_board_on_frame
from vision.pipeline.orientation import IDENTITY


@dataclass(frozen=True)
class WatchdogParams:
    interval_s: float = 2.0
    """Két ellenőrzés között eltelt idő."""

    check_width_px: int = 672
    """Ide kicsinyítjük az ellenőrzéshez. 1920 -> 672 mellett a detektálás
    430 ms helyett 44 ms, és a pontossága nem romlott."""

    shift_threshold_ratio: float = 0.06
    """Elmozdulás-küszöb a mezőköz arányában (1080p-n ~6.8 px). A mért
    remegés 0.9%, egy valódi elmozdulás 33% volt."""

    confirmations: int = 2
    """Ennyi egymást követő ellenőrzésnek kell elmozdulást mutatnia. Egyetlen
    kiugró detektálás így nem indít felesleges újradetektálást."""

    quiet_after_motion_s: float = 1.0
    """Ennyi ideig nem ellenőrzünk azután, hogy a tábla fölött mozgás volt —
    egy kéz eltakarhatja a sarkokat és hamis elmozdulást mutatna."""


def mean_nearest_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Két mezőközép-halmaz átlagos eltérése, a rács SORRENDJÉTŐL függetlenül.

    A detektor rács-orientációja nem garantált (184 képből 18-nál jött vissza
    elforgatva), ezért indexre hasonlítani félrevezető lenne: minden új pontot
    a hozzá legközelebbi régihez mérünk.
    """
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def _flat_centers(centers) -> np.ndarray:
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)


def _square_spacing(pts: np.ndarray) -> float:
    """Két szomszédos mezőközép távolsága — ehhez arányosítjuk a küszöböt."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


class BoardWatchdog:
    """Külön szálon figyeli, elmozdult-e a tábla a befagyasztott rácshoz képest."""

    def __init__(
        self,
        tracker_getter: Callable[[], object],
        frame_getter: Callable[[], np.ndarray | None],
        cfg,
        params: WatchdogParams | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._tracker_getter = tracker_getter
        self._frame_getter = frame_getter
        self._cfg = cfg
        self.params = params or WatchdogParams()
        self._clock = clock

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # állapot az overlay-hez
        self.checks = 0
        self.over_threshold_streak = 0
        self.swaps = 0
        self.last_shift_px: float | None = None
        self.last_status: str = "-"

    # ── életciklus ───────────────────────────────────────────────────────────

    def start(self) -> "BoardWatchdog":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.params.interval_s):
            try:
                self.last_status = self.check_once()
            except Exception as e:  # egy őrszem sosem viheti magával a rendszert
                self.last_status = f"hiba: {type(e).__name__}: {e}"

    # ── egy ellenőrzés ───────────────────────────────────────────────────────

    def _should_skip(self, tracker) -> str | None:
        if tracker is None or not getattr(tracker, "initialized", False):
            # Init közben úgyis minden képkockán fut a detektálás.
            return "init"
        last_motion = getattr(tracker.stabilizer, "last_motion_t", None)
        if last_motion is not None and (self._clock() - last_motion) < self.params.quiet_after_motion_s:
            return "mozgás"
        return None

    def _detect_small(self, frame_bgr: np.ndarray) -> tuple[DetectionResult, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.params.check_width_px / gray.shape[1])
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cell = max(16, int(round(self._cfg.cell * scale)))
        return detect_board_on_frame(gray, cell=cell, inner_pad_ratio=self._cfg.inner_pad_ratio), scale

    def check_once(self) -> str:
        tracker = self._tracker_getter()
        skip = self._should_skip(tracker)
        if skip:
            return f"kihagyva ({skip})"

        frame = self._frame_getter()
        if frame is None:
            return "kihagyva (nincs képkocka)"

        current_centers, _ = tracker.detection_snapshot()
        if current_centers is None:
            return "kihagyva (nincs rács)"

        det, scale = self._detect_small(frame)
        self.checks += 1
        if not det.ok:
            # Nem esemény: takarás, rossz fény. A streaket nem nullázzuk, mert
            # egy sikertelen ellenőrzés nem cáfolja az elmozdulást.
            return "nem találta a táblát"

        old = _flat_centers(current_centers)
        new = _flat_centers(det.centers_img) / scale
        shift = mean_nearest_shift(new, old)
        self.last_shift_px = shift
        threshold = _square_spacing(old) * self.params.shift_threshold_ratio

        if shift < threshold:
            self.over_threshold_streak = 0
            return f"változatlan ({shift:.1f} < {threshold:.1f} px)"

        self.over_threshold_streak += 1
        if self.over_threshold_streak < self.params.confirmations:
            return (f"elmozdulás gyanú {self.over_threshold_streak}/{self.params.confirmations} "
                    f"({shift:.1f} > {threshold:.1f} px)")

        self.over_threshold_streak = 0
        return self._full_redetect(tracker, frame, shift)

    def _full_redetect(self, tracker, frame_bgr: np.ndarray, shift: float) -> str:
        """Teljes felbontású újradetektálás + orientáció-igazítás — a háttérben."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        det = detect_board_on_frame(gray, cell=self._cfg.cell,
                                    inner_pad_ratio=self._cfg.inner_pad_ratio)
        if not det.ok:
            return f"elmozdult ({shift:.1f} px), de a teljes detektálás nem sikerült"

        alignment = tracker.align_offered_detection(det, frame_bgr)
        tracker.offer_detection(det, alignment)
        self.swaps += 1
        return f"újradetektálva ({shift:.1f} px, align={alignment})"

    # ── overlay ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "checks": self.checks,
            "swaps": self.swaps,
            "last_shift_px": self.last_shift_px,
            "status": self.last_status,
        }
