"""
StateStabilizer — vision -> chess handoff (10. szakasz, átszervezett változat).

Nem sakk-szabályokat kezel, hanem azt dönti el, hogy a klasszifikátor
képkockánkénti 8x8 foglaltság-rácsa MIKOR tekinthető elég stabilnak ahhoz,
hogy a chess logic (resolver) lépésként értelmezze.

Miből állt a régi (öt rétegű) zajszűrés, és mi lett vele — a döntés alapja a
tools/measure_pipeline_noise.py mérése az új MobileNetV3 modellel
(docs/pipeline_tuning.md, 2026-09-06):

  1. frame throttle (minden 3. frame)      -> ELHAGYVA (a tracker minden frame-et
                                              feldolgoz; a költség 15-35 ms < 33 ms)
  2. többségi szavazás 5 frame-es pufferen -> ELHAGYVA. Statikus táblán 333
                                              frame / 21 312 mezőosztályozás alatt
                                              NULLA címkeváltás volt; a puffer csak
                                              +2-3 frame késleltetést adott.
  3. perzisztencia-számláló (4 frame)      -> MEGTARTVA, de FALI IDŐBEN mérve
                                              (min_stable_s) + minimális frame-szám.
                                              Ez az egyetlen réteg, amely valódi
                                              információt hordoz: "a jelenet ennyi
                                              ideje változatlan".
  4. konfidencia-kapuk (tábla-átlag, stb.)  -> A tábla-átlag kapu ELHAGYVA (egy
                                              bizonytalan mező a tábla túlvégén
                                              blokkolt egy biztos lépést). Maradt egy
                                              ALACSONY padló a VÁLTOZOTT mezőkre
                                              (min_changed_conf=0,40) és egy "szemét frame"
                                              szűrő (frame_reject_mean_conf). Mért
                                              tény: a hibás predikciók konfidenciája
                                              ugyanolyan magas (p50 0,976), mint a
                                              helyeseké, a kapu tehát nem hibaszűrő.
  5. resolver zajtűrése                     -> a chess_logic-ban maradt, de szigorúbb
                                              (min_changed_cells=2, a fuzzy célmezőt
                                              foglaltnak kell látni).

ÚJ réteg — mozgás-kapu (motion gate): a tracker a warpolt tábla négyzetenkénti
|diff|-jét adja át (ugyanaz a jel, amit a részleges újraklasszifikálás is
használ); amíg a táblán mozgás van (kéz, kar, árnyék), NEM adunk ki állapotot,
és a mozgás megszűnése után min_static_s-ig várunk. Ez a réteg hordozza azt
az információt, amit a régi hosszú (600 ms-os) várakozás csak közvetve: "a
lépés véget ért". Ezért lehet a stabilitási ablak rövidebb a régi ~600 ms-nál
anélkül, hogy a hamis elfogadás valószínűsége nőne.

SÉRTHETETLEN INVARIÁNS: minden legális lépés >= 2 mezőt változtat. Az 1 mezős
eltérés SOSEM kerül kiadásra (régen kiadtuk, és a resolver fuzzy ága
lépést tippelt hozzá — ez volt a legveszélyesebb hamis-elfogadási út).

A flicker-tolerancia (1 mezős eltérés nem nullázza a perzisztenciát; 3 egymás
utáni "flicker" után mégis átvesszük a rácsot) VÁLTOZATLAN — ez korábbi,
bizonyítottan működő javítás.

A küszöbök EGY helyen: StabilizerParams (itt a definíció + a mért indoklás
mezőnként), az élesben használt értékek vision/app/config.py make_stabilizer().
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

OccGrid = list[list[int]]
ConfGrid = list[list[float]]
Cell = tuple[int, int]


def _grid_copy(g: OccGrid) -> OccGrid:
    return copy.deepcopy(g)


def _changed_cells(a: OccGrid, b: OccGrid) -> list[Cell]:
    return [(r, c) for r in range(8) for c in range(8) if a[r][c] != b[r][c]]


def _hamming_occ(a: OccGrid, b: OccGrid) -> int:
    return len(_changed_cells(a, b))


def _mean_conf(confs: ConfGrid) -> float:
    total = 0.0
    for r in range(8):
        for c in range(8):
            total += float(confs[r][c])
    return total / 64.0


@dataclass(frozen=True)
class StabilizerParams:
    """A stabilizer minden numerikus küszöbe — egy helyen, mérési indoklással.

    Az értékek újrahangolása: python -m tools.measure_pipeline_noise
    (statikus felvétel -> villódzás, konfidencia, mozgás-alapvonal) és
    python -m tools.replay_frames (latencia / hamis elfogadás visszajátszáson).
    """

    # Fali idő, ameddig a jelölt rácsnak (<=1 mezős flickerrel) változatlannak
    # kell lennie, mielőtt kiadható. A régi rendszer effektív ablaka ~600 ms
    # volt (5-ös szavazópuffer + 4 frame perzisztencia 10 fps-en); most a
    # mozgás-kapu viszi a "lépés véget ért" információt, ez az ablak a
    # klasszifikátor-tranzienseket és a nagyon rövid ideig fennálló félkész
    # állapotokat szűri. Mért: statikus táblán 0 villódzás -> az ablakot nem
    # a zaj, hanem a biztonsági tartalék indokolja.
    min_stable_s: float = 0.25
    # Legalább ennyi egymást követő, a jelölttel egyező frame is kell (alacsony
    # feldolgozási fps esetén — pl. 10 fps-nél ez 300 ms — ez a kötőbb feltétel).
    min_stable_frames: int = 3

    # Mozgás-kapu: a négyzetenkénti mean|diff| (0-255) ezen küszöb felett = a
    # táblán mozgás van. Mért statikus alapvonal (MJPG felvétel, félhomály):
    # p99,9 = 6,9, max = 7,1; a részleges újraklasszifikálás küszöbe 18.
    motion_threshold: float = 12.0
    # A mozgás megszűnése után ennyi ideig még nem adunk ki állapotot.
    min_static_s: float = 0.20

    # Flicker-tolerancia (korábbi javítás, változatlan): ennyi mezős eltérés a
    # jelölttől nem nullázza a perzisztenciát, de max_candidate_misses egymás
    # utáni ilyen frame után a jelöltet lecseréljük.
    flicker_tolerance_cells: int = 1
    max_candidate_misses: int = 3

    # Sakk-invariáns: egy lépés 2..4 mezőt változtat (sánc 4). 6 = tartalék.
    min_changed_for_move: int = 2
    max_changed_for_move: int = 6

    # A VÁLTOZOTT mezők konfidenciájának minimuma (nem a tábla átlaga!). Mért:
    # a helyes predikciók minimuma 0,47 (184 kép) / 0,595 (statikus felvétel);
    # a hibás predikciók 98%-a 0,6 FELETT — a kapu tehát nem hibaszűrő, csak
    # padló a közel-egyenletes (értékelhetetlen) kimenet ellen. 3 osztálynál
    # 0,40 alatt a modell gyakorlatilag nem preferál osztályt. (0,60-nal a
    # visszajátszásban 2/142 lépés beragadt egy 0,48-as, HELYES célmezőn.)
    min_changed_conf: float = 0.40
    # A teljes frame elvetése, ha a tábla átlagkonfidenciája ez alatt van
    # (fény-összeomlás, kéz az egész táblán). Statikus táblán az átlag ~0,97.
    frame_reject_mean_conf: float = 0.60


@dataclass(slots=True)
class StabilizerDecision:
    emit_occ: OccGrid | None
    reason: str
    mode: str
    changed_cells: list[Cell] = field(default_factory=list)
    stable_s: float = 0.0          # mióta változatlan a jelölt (fali idő)
    stable_frames: int = 0         # hány egymást követő egyező frame
    static_s: float = math.inf     # mióta nincs mozgás a táblán
    candidate_since: float | None = None   # a jelölt első frame-jének ideje
    last_motion_t: float | None = None     # az utolsó mozgás ideje


# Módok:
#   WARMUP    nincs referencia (init előtt)
#   STABLE    a megfigyelt rács == referencia (rendben, nincs teendő)
#   CANDIDATE egy >=2 mezős eltérés érlelődik (időablak / konfidencia)
#   NOISE     tartós 1 mezős eltérés — sosem lépés; a tracker újraklasszifikál
#   BLOCKED   mozgás / elvetett frame — nem döntünk
SETTLED_REASONS = frozenset({"no_change"})


class StateStabilizer:
    """
    update(occ, confs, now_s=..., motion=...) -> StabilizerDecision

    A referencia (amihez az eltérést mérjük) a tracker által ELFOGADOTT állás
    (set_reference) — így "no_change" tényleg azt jelenti, hogy a kamera azt
    látja, amit a sakklogika hisz. A stabilizer maga sosem írja át a
    referenciát kiadáskor: a tracker dönt (elfogadás -> set_reference az
    elvárt rácsra; elutasítás -> marad, és a tracker nem oldja fel újra
    ugyanazt a rácsot).
    """

    def __init__(self, params: StabilizerParams | None = None, **overrides):
        base = params or StabilizerParams()
        self.params = StabilizerParams(**{**base.__dict__, **overrides}) if overrides else base

        self._reference: OccGrid | None = None
        self._candidate: OccGrid | None = None
        self._cand_confs: ConfGrid | None = None
        self._candidate_since: float | None = None
        self._candidate_frames = 0
        self._candidate_miss = 0
        self._last_motion_t: float | None = None
        self._mode: str = "WARMUP"
        self._frames = 0

    # -- állapot --------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def reference(self) -> OccGrid | None:
        return self._reference

    @property
    def last_emitted(self) -> OccGrid | None:
        """Kompatibilitás: a referencia (a régi API 'utoljára kiadott' rácsa)."""
        return self._reference

    @property
    def candidate(self) -> OccGrid | None:
        return self._candidate

    @property
    def candidate_since(self) -> float | None:
        return self._candidate_since

    @property
    def last_motion_t(self) -> float | None:
        return self._last_motion_t

    def reset(self) -> None:
        self._reference = None
        self._candidate = None
        self._cand_confs = None
        self._candidate_since = None
        self._candidate_frames = 0
        self._candidate_miss = 0
        self._last_motion_t = None
        self._mode = "WARMUP"
        self._frames = 0

    def set_reference(self, occ: OccGrid) -> None:
        """Az elfogadott állás (a tracker hívja init után és minden elfogadott lépés után)."""
        self._reference = _grid_copy(occ)
        if self._mode == "WARMUP":
            self._mode = "STABLE"

    def note_motion(self, now_s: float) -> None:
        """Külső zavar jelzése (pl. robot mozog, újradetektálás): mozgásként számít."""
        self._last_motion_t = float(now_s)

    # -- jelölt-követés (flicker-toleráns, változatlan logika) ----------------

    def _classify_candidate_change(self, occ: OccGrid) -> str:
        """SAME / FLICKER / CHANGED — lásd a modul docstringjét."""
        if self._candidate is None:
            return "CHANGED"
        dist = _hamming_occ(occ, self._candidate)
        if dist == 0:
            return "SAME"
        if dist <= self.params.flicker_tolerance_cells and self._candidate_miss < self.params.max_candidate_misses:
            return "FLICKER"
        return "CHANGED"

    def _advance_candidate(self, occ: OccGrid, confs: ConfGrid, now: float) -> str:
        verdict = self._classify_candidate_change(occ)
        if verdict == "SAME":
            self._candidate_frames += 1
            self._candidate_miss = 0
            self._cand_confs = copy.deepcopy(confs)
        elif verdict == "FLICKER":
            self._candidate_miss += 1        # a jelölt és az ablak marad
        else:
            self._candidate = _grid_copy(occ)
            self._cand_confs = copy.deepcopy(confs)
            self._candidate_since = now
            self._candidate_frames = 1
            self._candidate_miss = 0
        return verdict

    # -- fő belépési pont -----------------------------------------------------

    def update(
        self,
        occ: OccGrid | None,
        confs: ConfGrid | None,
        *,
        now_s: float | None = None,
        motion: float | None = None,
    ) -> StabilizerDecision:
        """
        occ/confs: a frame 8x8 címkéi és konfidenciái (standard orientáció).
        motion: a tábla négyzetenkénti |diff|-jének maximuma az előző (és/vagy
            ~100 ms-mal korábbi) feldolgozott frame-hez képest; None = nincs
            információ (első frame, újradetektálás után) -> a kapu nem blokkol,
            a stabilitási ablak önmagában véd.
        """
        now = time.time() if now_s is None else float(now_s)
        p = self.params

        if motion is not None and motion > p.motion_threshold:
            self._last_motion_t = now

        if occ is None or confs is None:
            return self._decision(None, "no_data", self._mode, now)

        self._frames += 1
        mean_conf = _mean_conf(confs)
        if mean_conf < p.frame_reject_mean_conf:
            # Szemét frame (fény-összeomlás, kéz az egész táblán): nem frissíti a
            # jelöltet, és zavarásnak számít (a mozgás-kapu újraindul).
            self._last_motion_t = now
            self._mode = "BLOCKED"
            return self._decision(None, f"frame_rejected(mean_conf={mean_conf:.2f})", self._mode, now)

        self._advance_candidate(occ, confs, now)
        cand = self._candidate
        assert cand is not None and self._candidate_since is not None

        stable_s = now - self._candidate_since
        stable_frames = self._candidate_frames
        time_ok = stable_s >= p.min_stable_s and stable_frames >= p.min_stable_frames
        static_s = self._static_s(now)
        static_ok = static_s >= p.min_static_s

        if self._reference is None:
            # Önálló használat (tesztek): az első stabil, statikus rács a kiindulás.
            if time_ok and static_ok:
                self._reference = _grid_copy(cand)
                self._mode = "STABLE"
                return self._decision(_grid_copy(cand), "emit_initial_baseline", self._mode, now)
            self._mode = "WARMUP"
            return self._decision(None, "building_initial_baseline", self._mode, now)

        changed = _changed_cells(self._reference, cand)
        n = len(changed)

        if n == 0:
            self._mode = "STABLE"
            return self._decision(None, "no_change", self._mode, now)

        if n < p.min_changed_for_move:
            # Invariáns: 1 mezős eltérés nem lehet lépés. Nem adjuk ki — a
            # tracker célzott újraklasszifikálással próbálja tisztázni.
            self._mode = "NOISE"
            return self._decision(None, f"single_cell_diff({stable_s * 1000:.0f}ms)", self._mode, now, changed)

        if n > p.max_changed_for_move:
            self._mode = "CANDIDATE"
            return self._decision(None, f"too_many_changed({n})", self._mode, now, changed)

        if not time_ok:
            self._mode = "CANDIDATE"
            return self._decision(None, f"candidate_not_stable({stable_s * 1000:.0f}ms/{stable_frames}f)", self._mode, now, changed)

        if not static_ok:
            self._mode = "BLOCKED"
            return self._decision(None, f"board_in_motion({static_s * 1000:.0f}ms)", self._mode, now, changed)

        cc = self._cand_confs if self._cand_confs is not None else confs
        min_conf = min(float(cc[r][c]) for r, c in changed)
        if min_conf < p.min_changed_conf:
            self._mode = "CANDIDATE"
            return self._decision(None, f"changed_conf_low({min_conf:.2f})", self._mode, now, changed)

        self._mode = "STABLE"
        return self._decision(_grid_copy(cand), f"emit_change(changed={n})", self._mode, now, changed)

    # -- segédek --------------------------------------------------------------

    def _static_s(self, now: float) -> float:
        return math.inf if self._last_motion_t is None else max(0.0, now - self._last_motion_t)

    def _decision(self, emit: OccGrid | None, reason: str, mode: str, now: float,
                  changed: list[Cell] | None = None) -> StabilizerDecision:
        return StabilizerDecision(
            emit_occ=emit,
            reason=reason,
            mode=mode,
            changed_cells=list(changed or []),
            stable_s=0.0 if self._candidate_since is None else now - self._candidate_since,
            stable_frames=self._candidate_frames,
            static_s=self._static_s(now),
            candidate_since=self._candidate_since,
            last_motion_t=self._last_motion_t,
        )
