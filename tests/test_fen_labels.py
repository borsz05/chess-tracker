"""FEN -> nyers rács címkézés: az orientáció-fordítás a pipeline-éval konzisztens."""
import chess
import numpy as np
import pytest

from chess_logic.resolver import board_to_occupancy
from tools.fen_labels import fen_to_raw_labels, fen_to_raw_occupancy, raw_rc_to_square_name, standard_to_raw
from vision.pipeline.tracker import raw_to_standard

START = chess.STARTING_FEN
MIDGAME = "r1bqk2r/pp2bppp/2n2n2/3pp3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 0 7"
PROMO = "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1"


@pytest.mark.parametrize("fen", [START, MIDGAME, PROMO])
def test_standard_to_raw_is_inverse_of_raw_to_standard(fen):
    occ_std = np.asarray(board_to_occupancy(fen))
    occ_raw = standard_to_raw(occ_std)
    assert np.array_equal(raw_to_standard(occ_raw), occ_std)
    assert np.array_equal(fen_to_raw_occupancy(fen), occ_raw)


def test_roundtrip_on_arbitrary_grid():
    g = np.arange(64).reshape(8, 8)
    assert np.array_equal(raw_to_standard(standard_to_raw(g)), g)
    assert np.array_equal(standard_to_raw(raw_to_standard(g)), g)


def test_square_names_cover_board_once():
    names = {raw_rc_to_square_name(r, c) for r in range(8) for c in range(8)}
    assert names == {chess.square_name(sq) for sq in chess.SQUARES}


def test_labels_match_python_chess_per_square():
    board = chess.Board(MIDGAME)
    labels = fen_to_raw_labels(MIDGAME)
    for r in range(8):
        for c in range(8):
            lab = labels[r][c]
            piece = board.piece_at(chess.parse_square(lab.square))
            if piece is None:
                assert (lab.occ, lab.symbol, lab.type_idx, lab.color_name) == (0, "x", 0, "empty")
            else:
                assert lab.symbol == piece.symbol()
                assert lab.occ == (1 if piece.color == chess.WHITE else 2)
                assert lab.color_name == ("white" if piece.color == chess.WHITE else "black")
                assert lab.type_idx == {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}[piece.symbol().lower()]


def test_start_position_counts():
    labels = fen_to_raw_labels(START)
    flat = [lab for row in labels for lab in row]
    assert sum(l.occ == 1 for l in flat) == 16
    assert sum(l.occ == 2 for l in flat) == 16
    assert sum(l.occ == 0 for l in flat) == 32
    assert sum(l.type_name == "pawn" for l in flat) == 16
    assert sum(l.type_name == "king" for l in flat) == 2


def test_current_orientation_documented():
    # A kód szerinti orientáció: raw (r, c) -> standard (7 - c, 7 - r).
    # Ha ez a teszt elromlik, a raw_to_standard változott — akkor a
    # fen_labels dokumentációját (és a gyűjtő overlay-ét) frissíteni kell.
    assert raw_rc_to_square_name(0, 0) == "h1"
    assert raw_rc_to_square_name(7, 0) == "a1"
    assert raw_rc_to_square_name(0, 7) == "h8"
    assert raw_rc_to_square_name(7, 7) == "a8"
