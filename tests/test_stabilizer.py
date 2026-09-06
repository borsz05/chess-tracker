"""
StateStabilizer (vision/pipeline/stabilizer.py) — a 10. szakasz zajszűrő-rétegeinek
viselkedése és a sérthetetlen invariánsok:

  - 1 mezős eltérés SOHA nem kerül kiadásra (bármeddig stabil is);
  - 2 mezős eltérés csak min_stable_s fali idő ÉS min_stable_frames után;
  - a flicker-tolerancia nem nullázza az ablakot; tartós flicker átveszi a rácsot;
  - a mozgás-kapu blokkol, amíg a tábla mozog, és min_static_s-ig utána is;
  - a változott mezők alacsony konfidenciája blokkol, a tábla más részéé NEM;
  - túl sok változott mező -> nincs kiadás;
  - elvetett (szemét) frame nem zavar, de mozgásnak számít;
  - regresszió: villódzó statikus táblán nincs "candidate_not_persistent" beragadás,
    a valódi lépés az ablak + kis ráhagyás alatt kiadásra kerül.
"""
from __future__ import annotations

import random

import pytest

from vision.pipeline.stabilizer import StabilizerParams, StateStabilizer

P = StabilizerParams(
    min_stable_s=0.25, min_stable_frames=3, motion_threshold=12.0, min_static_s=0.20,
    flicker_tolerance_cells=1, max_candidate_misses=3, min_changed_for_move=2,
    max_changed_for_move=6, min_changed_conf=0.60, frame_reject_mean_conf=0.60,
)
DT = 1 / 30.0


def _start_occ():
    occ = [[0] * 8 for _ in range(8)]
    for c in range(8):
        occ[0][c] = occ[1][c] = 2
        occ[6][c] = occ[7][c] = 1
    return occ


def _confs(v=0.97):
    return [[v] * 8 for _ in range(8)]


def _moved(occ, frm, to, color=1):
    o = [row[:] for row in occ]
    o[frm[0]][frm[1]] = 0
    o[to[0]][to[1]] = color
    return o


def _stab(**kw):
    s = StateStabilizer(P, **kw)
    s.set_reference(_start_occ())
    return s


def _feed(s, occ, t0, n, *, confs=None, motion=0.0):
    """n frame ugyanabból a rácsból DT-nként; visszaadja az utolsó döntést és az időt."""
    d = None
    t = t0
    for i in range(n):
        t = t0 + i * DT
        d = s.update(occ, confs or _confs(), now_s=t, motion=motion)
    return d, t


def test_no_change_is_settled():
    s = _stab()
    d, _ = _feed(s, _start_occ(), 0.0, 5)
    assert d.emit_occ is None and d.reason == "no_change" and d.mode == "STABLE"


def test_two_cell_change_emits_after_time_and_frames():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))       # e2e4
    t0 = 1.0
    emitted_at = None
    for i in range(30):
        t = t0 + i * DT
        d = s.update(new, _confs(), now_s=t, motion=0.0)
        if d.emit_occ is not None:
            emitted_at = t
            assert d.emit_occ == new and sorted(d.changed_cells) == [(4, 4), (6, 4)]
            assert d.reason.startswith("emit_change")
            break
        assert d.reason.startswith("candidate_not_stable")
    assert emitted_at is not None
    # az ablak: >= min_stable_s a jelölt első frame-jétől, legfeljebb +1 frame
    assert P.min_stable_s <= emitted_at - t0 <= P.min_stable_s + DT + 1e-9


def test_min_stable_frames_binds_at_low_fps():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    # 1 fps: az idő rég letelt, de csak a 3. egyező frame-nél adható ki
    d1 = s.update(new, _confs(), now_s=10.0, motion=0.0)
    d2 = s.update(new, _confs(), now_s=11.0, motion=0.0)
    d3 = s.update(new, _confs(), now_s=12.0, motion=0.0)
    assert d1.emit_occ is None and d2.emit_occ is None and d3.emit_occ is not None


def test_single_cell_difference_is_never_emitted():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    one = [row[:] for row in _start_occ()]
    one[6][4] = 0                                   # "eltűnt" az e2 gyalog
    d, _ = _feed(s, one, 1.0, 300)                  # 10 másodpercig stabil
    assert d.emit_occ is None and d.mode == "NOISE" and d.reason.startswith("single_cell_diff")
    assert d.changed_cells == [(6, 4)]


def test_lift_then_place_completes_within_window():
    """Csendes lépés két lépcsőben: előbb a from ürül (1 mező), aztán a to
    foglalt lesz. Az 1 mezős fázis nem adható ki; a 2 mezős állapot az
    ablak (+ a flicker-tolerancia legfeljebb 3 frame-je) alatt kiadásra kerül."""
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    lifted = [row[:] for row in _start_occ()]
    lifted[6][4] = 0
    d, t = _feed(s, lifted, 1.0, 15, motion=0.0)
    assert d.emit_occ is None and d.mode == "NOISE"
    placed = _moved(_start_occ(), (6, 4), (4, 4))
    t0 = t + DT
    for i in range(40):
        tt = t0 + i * DT
        d = s.update(placed, _confs(), now_s=tt, motion=0.0)
        if d.emit_occ is not None:
            break
    assert d.emit_occ == placed
    assert tt - t0 <= P.min_stable_s + (P.max_candidate_misses + 1) * DT + 1e-9


def test_flicker_does_not_reset_window_but_persistent_flicker_is_adopted():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    flick = [row[:] for row in new]
    flick[0][0] = 0                                  # egy távoli mező villan
    t0 = 1.0
    s.update(new, _confs(), now_s=t0, motion=0.0)
    s.update(flick, _confs(), now_s=t0 + DT, motion=0.0)          # FLICKER
    d = s.update(new, _confs(), now_s=t0 + 2 * DT, motion=0.0)
    assert d.candidate_since == t0                                 # az ablak nem indult újra
    # tartós flicker: max_candidate_misses után a jelölt lecserélődik
    for i in range(P.max_candidate_misses + 1):
        d = s.update(flick, _confs(), now_s=t0 + (3 + i) * DT, motion=0.0)
    assert s.candidate == flick and d.candidate_since > t0


def test_motion_gate_blocks_until_static():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    t0 = 1.0
    # az állapot rég stabil, de mozgás van a táblán
    d, t = _feed(s, new, t0, 20, motion=40.0)
    assert d.emit_occ is None and d.reason.startswith("board_in_motion") and d.mode == "BLOCKED"
    # a mozgás megszűnik: min_static_s múlva jön az emit
    t_stop = t
    emitted = None
    for i in range(1, 40):
        tt = t_stop + i * DT
        d = s.update(new, _confs(), now_s=tt, motion=0.0)
        if d.emit_occ is not None:
            emitted = tt
            break
    assert emitted is not None
    assert P.min_static_s <= emitted - t_stop <= P.min_static_s + DT + 1e-9


def test_motion_none_means_no_information():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5, motion=None)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    d, _ = _feed(s, new, 1.0, 12, motion=None)
    assert d.emit_occ is not None


def test_changed_cell_low_conf_blocks_but_remote_low_conf_does_not():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    low_remote = _confs()
    low_remote[0][0] = 0.30                          # a tábla túlvége bizonytalan
    d, _ = _feed(s, new, 1.0, 12, confs=low_remote)
    assert d.emit_occ is not None                    # NEM blokkol (régen a tábla-átlag kapu itt is szólhatott)

    s2 = _stab()
    _feed(s2, _start_occ(), 0.0, 5)
    low_changed = _confs()
    low_changed[4][4] = 0.40                         # a célmező bizonytalan
    d2, _ = _feed(s2, new, 1.0, 12, confs=low_changed)
    assert d2.emit_occ is None and d2.reason.startswith("changed_conf_low")


def test_too_many_changed_blocks():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    many = [row[:] for row in _start_occ()]
    for c in range(7):
        many[3][c] = 1
    d, _ = _feed(s, many, 1.0, 12)
    assert d.emit_occ is None and d.reason.startswith("too_many_changed")


def test_rejected_frame_counts_as_disturbance():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    d, t = _feed(s, new, 1.0, 12)
    assert d.emit_occ is not None
    # egy szemét frame (átlag-konfidencia alacsony)
    d = s.update(new, _confs(0.3), now_s=t + DT, motion=0.0)
    assert d.emit_occ is None and d.reason.startswith("frame_rejected") and d.last_motion_t == t + DT
    # utána min_static_s-ig blokkol, de a jelölt ablaka nem indult újra
    d = s.update(new, _confs(), now_s=t + 2 * DT, motion=0.0)
    assert d.reason.startswith("board_in_motion") and d.candidate_since == 1.0


def test_reference_update_settles():
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    new = _moved(_start_occ(), (6, 4), (4, 4))
    d, t = _feed(s, new, 1.0, 12)
    assert d.emit_occ is not None
    # a tracker elfogadta: referencia = az elvárt rács -> no_change
    s.set_reference(new)
    d = s.update(new, _confs(), now_s=t + DT, motion=0.0)
    assert d.reason == "no_change" and d.mode == "STABLE"


def test_emit_repeats_while_state_persists_and_reference_unchanged():
    """Ha a tracker elutasította (nincs legális lépés), a referencia marad, és a
    stabil eltérés minden frame-ben újra kiadásra kerül — a tracker deduplikál."""
    s = _stab()
    _feed(s, _start_occ(), 0.0, 5)
    weird = [row[:] for row in _start_occ()]
    weird[3][3] = weird[4][4] = 1
    d, t = _feed(s, weird, 1.0, 12)
    assert d.emit_occ is not None
    d = s.update(weird, _confs(), now_s=t + DT, motion=0.0)
    assert d.emit_occ is not None


def test_regression_no_stall_on_flickering_static_board():
    """Statikus tábla, véletlen 1 mezős flickerekkel minden 4. frame-ben; majd
    egy valódi lépés. Régen a flickerek nullázták a perzisztenciát
    (candidate_not_persistent beragadás). Most: a flicker alatt nincs kiadás,
    a lépés az ablak alatt kiadásra kerül."""
    rng = random.Random(3)
    s = _stab()
    base = _start_occ()
    t = 0.0
    for i in range(300):
        occ = [row[:] for row in base]
        if i % 4 == 0:
            r, c = rng.randrange(8), rng.randrange(8)
            occ[r][c] = (occ[r][c] + 1) % 3
        d = s.update(occ, _confs(), now_s=t, motion=0.0)
        assert d.emit_occ is None
        t += DT
    new = _moved(base, (6, 4), (4, 4))
    t0 = t
    emitted = None
    for i in range(40):
        occ = [row[:] for row in new]
        if i % 4 == 2:
            occ[0][0] = 0                            # flicker a lépés alatt is
        d = s.update(occ, _confs(), now_s=t, motion=0.0)
        if d.emit_occ is not None:
            emitted = t
            assert d.emit_occ == new
            break
        t += DT
    assert emitted is not None and emitted - t0 <= P.min_stable_s + 2 * DT + 1e-9


def test_standalone_initial_baseline():
    s = StateStabilizer(P)
    reasons = [s.update(_start_occ(), _confs(), now_s=i * DT, motion=0.0).reason for i in range(12)]
    assert "emit_initial_baseline" in reasons and s.reference == _start_occ()
    assert reasons[-1] == "no_change"


def test_params_overrides_and_reset():
    s = StateStabilizer(P, min_stable_s=0.5)
    assert s.params.min_stable_s == 0.5 and s.params.motion_threshold == P.motion_threshold
    s.set_reference(_start_occ())
    s.update(_start_occ(), _confs(), now_s=0.0, motion=50.0)
    s.reset()
    assert s.mode == "WARMUP" and s.reference is None and s.last_motion_t is None


def test_config_make_stabilizer_uses_documented_params():
    from vision.app.config import STABILIZER_PARAMS, make_stabilizer
    s = make_stabilizer()
    assert s.params == STABILIZER_PARAMS
    assert s.params.min_changed_for_move == 2      # az invariáns nem hangolható ki alatta
    assert s.params.motion_threshold > 7.2         # a mért statikus alapvonal maximuma felett
    assert s.params.min_stable_s <= 0.6            # rövidebb a régi effektív ~600 ms-nál
