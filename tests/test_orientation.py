"""vision/pipeline/orientation.py — szimmetriák, inverz, biztonságos igazítás-választás."""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_logic import board_to_occupancy
from vision.pipeline.orientation import (
    IDENTITY,
    SYMMETRY_NAMES,
    apply_symmetry,
    choose_alignment,
    inverse_symmetry,
    rank_symmetries,
    symmetries,
)
from vision.pipeline.tracker import ROTATION_SYMMETRIES, align_detection_to_expected, standard_to_raw
from vision.pipeline.board_detector import DetectionResult


@pytest.mark.parametrize("name", SYMMETRY_NAMES)
def test_apply_matches_symmetries_and_inverse(name):
    g = np.arange(64).reshape(8, 8)
    assert np.array_equal(apply_symmetry(g, name), dict(symmetries(g))[name])
    inv = inverse_symmetry(name)
    assert np.array_equal(apply_symmetry(apply_symmetry(g, name), inv), g)
    # lista bemenet (pl. bbox-tuple-ök) ugyanúgy
    lst = [[(r, c) for c in range(8)] for r in range(8)]
    out = apply_symmetry(lst, name)
    gs = apply_symmetry(g, name)
    for r in range(8):
        for c in range(8):
            assert out[r][c] == (int(gs[r, c]) // 8, int(gs[r, c]) % 8)


def test_rank_orders_identity_first_on_ties():
    occ = np.asarray(board_to_occupancy(chess.Board()))
    ranking = rank_symmetries(occ, occ)
    assert ranking[0] == (IDENTITY, 0)
    assert ranking[1][1] == 0 and ranking[1][0] == "rot0+flip"      # az alapállás bal-jobb szimmetrikus


def test_choose_alignment_keeps_identity_when_good_enough():
    occ = np.asarray(board_to_occupancy(chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")))
    noisy = occ.copy()
    noisy[3, 3] = 2
    name, _ = choose_alignment(noisy, occ, max_mismatch=2, min_margin=8)
    assert name == IDENTITY


@pytest.mark.parametrize("sym", ["rot90", "rot180", "rot270"])
def test_choose_alignment_finds_unambiguous_rotation(sym):
    occ = np.asarray(board_to_occupancy(chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")))
    observed = apply_symmetry(occ, sym)
    observed[0, 0] = (observed[0, 0] + 1) % 3                       # 1 zajos mező
    name, ranking = choose_alignment(observed, occ, max_mismatch=2, min_margin=8, allowed=ROTATION_SYMMETRIES)
    assert name == sym and ranking[0][1] == 1


def test_choose_alignment_refuses_ambiguous_start_position_with_flips():
    """Alapállás: rot90 és rot90+flip azonos eltérésű -> a 8 szimmetriával nem dönt,
    csak forgatásokkal igen."""
    occ = np.asarray(board_to_occupancy(chess.Board()))
    observed = apply_symmetry(occ, "rot90")
    name_all, _ = choose_alignment(observed, occ, max_mismatch=2, min_margin=4)
    name_rot, _ = choose_alignment(observed, occ, max_mismatch=2, min_margin=4, allowed=ROTATION_SYMMETRIES)
    assert name_all == IDENTITY and name_rot == "rot90"


def test_choose_alignment_refuses_when_board_is_far_from_expected():
    occ = np.asarray(board_to_occupancy(chess.Board()))
    observed = apply_symmetry(occ, "rot90")
    for i in range(3):
        observed[3, i] = 1                                            # 3 idegen bábu
    name, _ = choose_alignment(observed, occ, max_mismatch=2, min_margin=4, allowed=ROTATION_SYMMETRIES)
    assert name == IDENTITY


def test_align_detection_reorders_bbox_grid_so_labels_match():
    """Elforgatott detektor-rács: az igazítás után a rács mezőnként a helyes bbox-ot adja."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    expected_std = np.asarray(board_to_occupancy(board))
    good_bbox = [[(c, r, c + 1, r + 1) for c in range(8)] for r in range(8)]
    for sym in ("rot90", "rot180", "rot270"):
        det = DetectionResult(ok=True, M=np.eye(3), bbox_warp=apply_symmetry(good_bbox, sym),
                              centers_warp=apply_symmetry([[(c, r) for c in range(8)] for r in range(8)], sym), centers_img=None)
        # amit a klasszifikátor az elforgatott rácson lát: a valós nyers rács a rács szimmetriája szerint
        observed_raw = apply_symmetry(standard_to_raw(expected_std), sym)
        name, _ = align_detection_to_expected(det, observed_raw, expected_std, max_mismatch=2, min_margin=4)
        assert name == sym
        assert det.bbox_warp == good_bbox
        assert det.centers_warp == [[(c, r) for c in range(8)] for r in range(8)]
