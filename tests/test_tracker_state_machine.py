"""
ChessVisionTracker állapotgép — szintetikus kamerakép + ál-modell + ál-detektor.

A "kamerakép" maga a warpolt tábla (M = egység), a mezők színe kódolja a
foglaltságot és a bábutípust, a kéz egy piros folt. Az ál-modell a ROI közepét
olvassa. Így a teljes tracker-út (warp -> diff/mozgás -> részleges/gördülő
klasszifikáció -> raw_to_standard -> stabilizer -> resolver -> elfogadás) fut,
determinisztikus órával.

Amit rögzítünk:
  - init az alapállásból; elforgatott detektor-rács igazítása init-kor;
  - csendes lépés két lépcsőben (felemel, lebegtet mozgással, letesz):
    a lebegés alatt NINCS elfogadás, letétel után az ablakon belül van;
  - tartós 1 mezős eltérés: nincs lépés, NOISE mód, ébresztő átosztályozás;
  - 2 mezős, de nem legális állapot: no-legal-fit, nincs lépés;
  - bástyával kezdett sánc: várakozás, a király érkezésével O-O; ha csak a
    bástya mozgott és úgy marad, a hosszabb ablak után Rf1;
  - promóció: a típus-fej gyalogot lát -> várakozás; csere után a látott bábu;
    csere nélkül a várakozás lejárta után vezér;
  - beragadás -> újradetektálás elforgatott ráccsal -> orientáció-igazítás;
  - elfogadási latencia-számvitel (accept_info).
"""
from __future__ import annotations

import numpy as np
import pytest

import vision.pipeline.tracker as tracker_mod
from chess_logic import board_to_occupancy
from chess_logic.resolver import TYPE_INDEX, chess_square_to_coords
from vision.app.config import AppConfig, STABILIZER_PARAMS
from vision.models.occupancy_color_model import Prediction
from vision.pipeline.board_detector import DetectionResult
from vision.pipeline.orientation import apply_symmetry
from vision.pipeline.tracker import ChessVisionTracker, standard_to_raw

import chess

CELL = 96
PAD = 6
OFF = 4 * CELL
SIZE = 17 * CELL
DT = 1 / 30.0
OCC_CODE = {0: 100, 1: 200, 2: 50}
P = STABILIZER_PARAMS


# ---------------------------------------------------------------------------
# ál-detektor / ál-modell / festés
# ---------------------------------------------------------------------------

def _bbox_grid():
    return [[(OFF + c * CELL + PAD, OFF + r * CELL + PAD, OFF + (c + 1) * CELL - PAD, OFF + (r + 1) * CELL - PAD)
             for c in range(8)] for r in range(8)]


def _centers(bbox):
    return [[((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in row] for row in bbox]


def _fake_det(symmetry: str = "rot0") -> DetectionResult:
    bbox = apply_symmetry(_bbox_grid(), symmetry)
    return DetectionResult(ok=True, M=np.eye(3, dtype=np.float64), bbox_warp=bbox,
                           centers_warp=_centers(bbox), centers_img=_centers(bbox))


class FakeModel:
    """A ROI közepének színéből olvas: B = foglaltság, G = típus, R = kéz."""

    def __init__(self):
        self.calls: list[int] = []

    def predict_rois(self, rois):
        self.calls.append(len(rois))
        n = len(rois)
        labels = np.zeros(n, np.int32)
        confs = np.zeros(n, np.float32)
        types = np.zeros((n, len(TYPE_INDEX)), np.float32)
        for i, roi in enumerate(rois):
            h, w = roi.shape[:2]
            b, g, r = [int(v) for v in roi[h // 2, w // 2]]
            if r > 128:                                   # kéz
                labels[i], confs[i] = 1, 0.55
                types[i, 0] = 1.0
                continue
            labels[i] = min(OCC_CODE, key=lambda k: abs(OCC_CODE[k] - b))
            confs[i] = 0.99
            types[i, min(6, max(0, g // 10))] = 0.92
        return Prediction(labels=labels, confs=confs, color_probs=np.zeros((n, 3), np.float32), type_probs=types)


def _std_types_from_board(board: chess.Board) -> np.ndarray:
    t = np.zeros((8, 8), np.int32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None:
            r, c = chess_square_to_coords(sq)
            t[r, c] = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}[p.piece_type]
    return t


class Scene:
    """Egy jelenet: standard foglaltság + típus rács, opcionális kéz-mezők; frame-et fest."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.bbox = _bbox_grid()

    def frame(self, occ_std, types_std=None, hand_std=(), noise=True):
        occ_raw = standard_to_raw(np.asarray(occ_std))
        typ_raw = standard_to_raw(np.asarray(types_std)) if types_std is not None else np.zeros((8, 8), np.int32)
        hand_raw = set()
        if hand_std:
            mask = np.zeros((8, 8), np.int32)
            for r, c in hand_std:
                mask[r, c] = 1
            mr = standard_to_raw(mask)
            hand_raw = {(r, c) for r in range(8) for c in range(8) if mr[r, c]}
        img = np.full((SIZE, SIZE, 3), 120, np.uint8)
        for r in range(8):
            for c in range(8):
                x0, y0, x1, y1 = self.bbox[r][c]
                img[y0:y1, x0:x1] = (OCC_CODE[int(occ_raw[r, c])], int(typ_raw[r, c]) * 10 + 5, 0)
                if (r, c) in hand_raw:
                    img[y0:y1, x0:x1, 2] = 255
        if noise:
            n = self.rng.integers(-2, 3, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        return img


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt


def _cfg(**kw) -> AppConfig:
    base = dict(enable_pipeline_profiler=False, weights_path=None)
    base.update(kw)
    return AppConfig(**base)


@pytest.fixture
def fake_detect(monkeypatch):
    """Cserélhető ál-detektor: a lista elemei sorban jönnek vissza, az utolsó ismétlődik."""
    seq: list[DetectionResult] = [_fake_det()]

    def _detect(gray, *, cell, inner_pad_ratio):
        det = seq.pop(0) if len(seq) > 1 else seq[0]
        # friss példány, hogy a tracker helyben módosíthassa
        return DetectionResult(ok=det.ok, M=det.M.copy(), bbox_warp=[row[:] for row in det.bbox_warp],
                               centers_warp=[row[:] for row in det.centers_warp], centers_img=[row[:] for row in det.centers_img])

    monkeypatch.setattr(tracker_mod, "detect_board_on_frame", _detect)
    return seq


def _run(tracker, clock, scene, occ, types=None, hand=(), n=1, until_accept=False, max_frames=400):
    """n frame (vagy elfogadásig); visszaadja az eredmények listáját."""
    out = []
    for i in range(max_frames if until_accept else n):
        res = tracker.process_frame(scene.frame(occ, types, hand))
        out.append(res)
        clock.tick()
        if until_accept and res.board_changed:
            break
    return out


def _init_tracker(fake_detect, fen=chess.STARTING_FEN, **cfg_kw):
    cfg = _cfg(start_fen=fen, **cfg_kw)
    clock = Clock()
    model = FakeModel()
    tr = ChessVisionTracker(cfg, model=model, clock=clock)
    scene = Scene()
    board = chess.Board(fen)
    occ = np.asarray(board_to_occupancy(board), np.int32)
    types = _std_types_from_board(board)
    res = _run(tr, clock, scene, occ, types, n=cfg.init_buffer_frames)
    assert tr.initialized, [r.mode for r in res]
    assert np.array_equal(tr.accepted_occ, occ)
    return tr, clock, scene, model, board


def _after(board: chess.Board, uci: str):
    b = board.copy()
    b.push_uci(uci)
    return np.asarray(board_to_occupancy(b), np.int32), _std_types_from_board(b), b


# ---------------------------------------------------------------------------
# tesztek
# ---------------------------------------------------------------------------

def test_init_from_start_position(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    assert tr.init_stats["alignment"] == "rot0"
    # statikus tábla: no_change, nincs lépés, nincs mozgás
    res = _run(tr, clock, scene, tr.accepted_occ, _std_types_from_board(board), n=10)
    assert all(not r.board_changed for r in res)
    assert res[-1].reason == "no_change" and res[-1].motion is not None and res[-1].motion < P.motion_threshold


@pytest.mark.parametrize("sym", ["rot90", "rot180", "rot270"])
def test_init_aligns_rotated_detector_grid(fake_detect, sym):
    fake_detect[:] = [_fake_det(sym)]
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    assert tr.init_stats["alignment"] != "rot0"
    assert tr.det.bbox_warp == _bbox_grid()          # a rács vissza lett rendezve
    occ2, typ2, _ = _after(board, "e2e4")
    res = _run(tr, clock, scene, occ2, typ2, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "e2e4"


def test_quiet_move_lift_hover_place(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    occ0 = tr.accepted_occ.copy()
    typ0 = _std_types_from_board(board)
    lifted = occ0.copy()
    r_e2, c_e2 = chess_square_to_coords(chess.E2)
    lifted[r_e2, c_e2] = 0
    # 1) felemelve, kéz az e2 felett (mozgás)
    res = _run(tr, clock, scene, lifted, typ0, hand=[(r_e2, c_e2)], n=10)
    assert all(not r.board_changed for r in res)
    # 2) lebegtetés e3/e4 fölött ~1 s, minden frame-ben más mezőn (mozgás) — a
    #    letett állapotot MUTATJA a kép (a bábu a kézben a célmező felett), de mozog
    occ2, typ2, board2 = _after(board, "e2e4")
    r_e4, c_e4 = chess_square_to_coords(chess.E4)
    r_e3, c_e3 = chess_square_to_coords(chess.E3)
    for i in range(30):
        hand = [(r_e3, c_e3)] if i % 2 else [(r_e3, c_e3), (r_e4 - 1, c_e4)]
        res = _run(tr, clock, scene, occ2, typ2, hand=hand, n=1)
        assert not res[-1].board_changed, "elfogadás lebegő kéz alatt"
    t_hand_gone = clock.t
    # 3) kéz elment, tábla statikus
    res = _run(tr, clock, scene, occ2, typ2, until_accept=True)
    acc = res[-1]
    assert acc.board_changed and acc.uci == "e2e4" and acc.san == "e4"
    latency = acc.accept_info["t_accept"] - t_hand_gone
    assert latency <= max(P.min_static_s, P.min_stable_s) + 3 * DT + 1e-6
    assert latency >= P.min_static_s - 1e-6
    # a stabilizer referenciája az elfogadott állás -> no_change
    res = _run(tr, clock, scene, occ2, typ2, n=3)
    assert res[-1].reason == "no_change"
    assert np.array_equal(tr.accepted_occ, occ2)


def test_persistent_single_cell_error_never_moves_and_wakes_full_reclassify(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    bad = tr.accepted_occ.copy()
    r, c = chess_square_to_coords(chess.A1)
    bad[r, c] = 0                                   # az a1 bástya "eltűnik"
    model.calls.clear()
    res = _run(tr, clock, scene, bad, _std_types_from_board(board), n=90)   # 3 s
    assert all(not x.board_changed for x in res)
    assert res[-1].mode.startswith("NOISE:single_cell_diff")
    # ébresztő teljes átosztályozás történt (64 ROI-s hívás), rate-limitelve
    full_calls = sum(1 for n in model.calls if n == 64)
    assert 2 <= full_calls <= 90 * DT / tr.cfg.wakeup_full_reclassify_s + 3


def test_illegal_two_cell_state_is_not_a_move(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    weird = tr.accepted_occ.copy()
    r1, c1 = chess_square_to_coords(chess.A1)
    r4, c4 = chess_square_to_coords(chess.A4)
    weird[r1, c1] = 0
    weird[r4, c4] = 1                               # bástya "átugrotta" az a2 gyalogot
    res = _run(tr, clock, scene, weird, _std_types_from_board(board), n=60)
    assert all(not x.board_changed for x in res)
    assert any(x.mode == "no-legal-fit" for x in res)
    assert len(tr.game.move_history) == 0


def test_rook_first_castling_waits_then_accepts_castle(fake_detect):
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    tr, clock, scene, model, board = _init_tracker(fake_detect, fen=fen)
    rook_only, typ_r, _ = _after(board, "h1f1")
    res = _run(tr, clock, scene, rook_only, typ_r, n=int((P.min_stable_s + 0.5) / DT))
    assert all(not x.board_changed for x in res)
    assert any(x.mode == "ambiguous-wait" for x in res)
    # jön a király is -> O-O
    castled, typ_c, _ = _after(board, "e1g1")
    res = _run(tr, clock, scene, castled, typ_c, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "e1g1"


def test_rook_move_alone_accepted_after_extra_window(fake_detect):
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    tr, clock, scene, model, board = _init_tracker(fake_detect, fen=fen)
    rook_only, typ_r, _ = _after(board, "h1f1")
    t0 = clock.t
    res = _run(tr, clock, scene, rook_only, typ_r, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "h1f1"
    waited = res[-1].accept_info["t_accept"] - t0
    need = P.min_stable_s + tr.cfg.prefix_ambiguity_extra_s
    assert need - 1e-6 <= waited <= need + 2 * DT + 1e-6


def test_promotion_waits_for_piece_swap_and_uses_type_head(fake_detect):
    fen = "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1"
    tr, clock, scene, model, board = _init_tracker(fake_detect, fen=fen)
    occ2, typ2, _ = _after(board, "a7a8q")
    r8, c8 = chess_square_to_coords(chess.A8)
    typ_pawn = typ2.copy()
    typ_pawn[r8, c8] = TYPE_INDEX["pawn"]          # a gyalog még a 8. soron áll
    res = _run(tr, clock, scene, occ2, typ_pawn, n=30)
    assert all(not x.board_changed for x in res)
    assert any(x.mode == "promotion-wait" for x in res)
    typ_knight = typ2.copy()
    typ_knight[r8, c8] = TYPE_INDEX["knight"]      # lecserélte huszárra
    res = _run(tr, clock, scene, occ2, typ_knight, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "a7a8n"


def test_promotion_timeout_falls_back_to_queen(fake_detect):
    fen = "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1"
    tr, clock, scene, model, board = _init_tracker(fake_detect, fen=fen, promotion_wait_s=1.0)
    occ2, typ2, _ = _after(board, "a7a8q")
    r8, c8 = chess_square_to_coords(chess.A8)
    typ_pawn = typ2.copy()
    typ_pawn[r8, c8] = TYPE_INDEX["pawn"]
    res = _run(tr, clock, scene, occ2, typ_pawn, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "a7a8q"


def test_stuck_triggers_redetect_and_realigns_rotated_grid(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect, redetect_after_stuck_s=0.5, redetect_min_interval_s=0.5)
    typ0 = _std_types_from_board(board)
    # a következő detektálás elforgatott rácsot ad
    fake_detect[:] = [_fake_det("rot180"), _fake_det("rot180")]
    # "elmozdult a tábla": a befagyott rács egy sorral elcsúszik a jelenethez képest
    # -> a mezők másik mezőt látnak -> tartós, feloldhatatlan eltérés -> beragadás
    good = _bbox_grid()
    tr.det.bbox_warp = [good[r + 1] if r < 7 else [(x0, y0 + CELL, x1, y1 + CELL) for (x0, y0, x1, y1) in good[7]]
                        for r in range(8)]
    tr.prev_warp = None                                 # a cache a régi rácshoz tartozott
    res = _run(tr, clock, scene, tr.accepted_occ, typ0, n=40)
    assert all(not x.board_changed for x in res)
    assert any(x.mode == "redetect-ok" for x in res)
    assert tr.redetect_count >= 1 and tr.last_redetect_alignment == "rot180"
    assert tr.det.bbox_warp == _bbox_grid()
    # a jelenet visszaáll az elfogadott állásra -> no_change, majd egy lépés elfogadható
    res = _run(tr, clock, scene, tr.accepted_occ, typ0, n=10)
    assert res[-1].reason == "no_change"
    occ2, typ2, _ = _after(board, "g1f3")
    res = _run(tr, clock, scene, occ2, typ2, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "g1f3"


def test_partial_and_rolling_refresh_paths_are_used(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    model.calls.clear()
    _run(tr, clock, scene, tr.accepted_occ, _std_types_from_board(board), n=20)
    # statikus táblán csak a gördülő frissítés fut (rolling_refresh_squares ROI / frame)
    assert model.calls and max(model.calls) <= tr.cfg.rolling_refresh_squares + tr.cfg.partial_max_squares
    assert all(n == tr.cfg.rolling_refresh_squares for n in model.calls)


def test_accept_info_has_latency_fields(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    occ2, typ2, _ = _after(board, "d2d4")
    res = _run(tr, clock, scene, occ2, typ2, until_accept=True)
    info = res[-1].accept_info
    assert info and info["candidate_since"] is not None and info["t_accept"] >= info["candidate_since"]
    assert info["stable_s"] >= P.min_stable_s and info["changed_cells"] == 2


# ---------------------------------------------------------------------------
# háttér-őrszem által átadott detektálás
# ---------------------------------------------------------------------------
#
# A kamerát a rendszer indulása UTÁN is meg szokás mozgatni; a befagyasztott
# homográfia onnantól egy nem létező kameraállásra vonatkozik. A detektálás
# 340-580 ms, ezért a háttérszál végzi el, és ide már csak a kész eredmény
# érkezik — a fő ciklusban ez referenciacsere.


def test_offered_detection_is_swapped_in_without_touching_the_game(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    occ2, typ2, board2 = _after(board, "e2e4")
    assert _run(tr, clock, scene, occ2, typ2, until_accept=True)[-1].board_changed
    moves_before = list(tr.game.move_history)
    accepted_before = tr.accepted_occ.copy()

    tr.offer_detection(_fake_det(), "rot0")
    res = _run(tr, clock, scene, occ2, typ2, n=3)

    assert tr.watchdog_swaps == 1 and tr.last_watchdog_swap == "rot0"
    # a lényeg: a parti nem indult újra
    assert list(tr.game.move_history) == moves_before
    assert np.array_equal(tr.accepted_occ, accepted_before)
    assert all(not x.board_changed for x in res)

    # és utána is működik a követés
    occ3, typ3, _ = _after(board2, "e7e5")
    res = _run(tr, clock, scene, occ3, typ3, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "e7e5"


def test_offered_detection_forces_a_full_reclassify(fake_detect):
    """A warp elmozdult: a gyorsítótárazott előző képkocka már nem ehhez a
    rácshoz tartozik, ezért el kell dobni."""
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    typ0 = _std_types_from_board(board)
    _run(tr, clock, scene, tr.accepted_occ, typ0, n=5)

    model.calls.clear()
    tr.offer_detection(_fake_det(), "rot0")
    _run(tr, clock, scene, tr.accepted_occ, typ0, n=1)
    assert model.calls[0] == 64, "a csere után teljes átosztályozásnak kell jönnie"


def test_align_offered_detection_rearranges_a_rotated_grid(fake_detect):
    """A detektor rács-orientációja nem garantált — a háttérszál igazítja."""
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    occ2, typ2, board2 = _after(board, "d2d4")
    _run(tr, clock, scene, occ2, typ2, until_accept=True)

    rotated = _fake_det("rot90")
    frame = scene.frame(occ2, typ2)
    alignment = tr.align_offered_detection(rotated, frame)

    assert alignment == "rot90"
    assert rotated.bbox_warp == _bbox_grid(), "az igazítás helyben rendezi át a rácsot"

    tr.offer_detection(rotated, alignment)
    _run(tr, clock, scene, occ2, typ2, n=2)
    occ3, typ3, _ = _after(board2, "d7d5")
    res = _run(tr, clock, scene, occ3, typ3, until_accept=True)
    assert res[-1].board_changed and res[-1].uci == "d7d5"


def test_detection_snapshot_is_readable_for_the_watchdog(fake_detect):
    tr, clock, scene, model, board = _init_tracker(fake_detect)
    centers, accepted = tr.detection_snapshot()
    assert centers == tr.det.centers_img
    assert np.array_equal(accepted, tr.accepted_occ)


def test_detection_snapshot_is_empty_before_any_detection():
    tr = ChessVisionTracker(_cfg(), model=FakeModel(), clock=Clock())
    assert tr.detection_snapshot() == (None, None)
