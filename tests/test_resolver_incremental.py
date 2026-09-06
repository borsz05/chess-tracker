"""
Resolver (chess_logic/resolver.py) — a 10. szakasz 2. pontja:

  - expected_occupancy_after_move (inkrementális) == board.copy()+push()+
    board_to_occupancy (referencia) MINDEN legális lépésre, sok álláson
    (sánc, en passant, promóció, ütés-promóció is);
  - a resolver eredménye változatlan a referencia-implementációhoz képest;
  - az 1 mezős eltérésre a fuzzy ág SEM tippel lépést (min_changed_cells);
  - promóciónál a típus-fej tippje választ bábut, tipp nélkül vezér;
  - a use_type_hint_for_moves kapcsoló kikapcsolva nem változtat semmin.
"""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chess_logic.resolver import (
    DEFAULT_PROMOTION_MIN_CONF,
    prefix_ambiguities,
    PROMOTION_TYPE_INDEX,
    TYPE_INDEX,
    board_to_occupancy,
    chess_square_to_coords,
    expected_occupancy_after_move,
    promotion_preference,
    resolve_move_from_occupancy,
)


def _reference_expected(board: chess.Board, move: chess.Move) -> list[list[int]]:
    tmp = board.copy()
    tmp.push(move)
    return board_to_occupancy(tmp)


def _random_positions(n_games: int, plies: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            # promóciót/sáncot/ep-t előnyben, hogy biztosan legyen ilyen minta
            special = [m for m in moves if m.promotion or b.is_castling(m) or b.is_en_passant(m)]
            b.push(rng.choice(special) if special and rng.random() < 0.7 else rng.choice(moves))
            out.append(b.copy())
    return out


SPECIAL_FENS = [
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",        # mindkét sánc
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant
    "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",                          # promóció
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",                            # ütés-promóció
    "4k3/8/8/8/8/8/1p6/R3K3 b Q - 0 1",                            # fekete ütés-promóció
]


@pytest.mark.parametrize("fen", SPECIAL_FENS)
def test_incremental_matches_reference_special(fen):
    b = chess.Board(fen)
    occ0 = board_to_occupancy(b)
    seen = {"castle": 0, "ep": 0, "promo": 0}
    for mv in b.legal_moves:
        assert expected_occupancy_after_move(b, mv, occ0) == _reference_expected(b, mv), mv
        seen["castle"] += b.is_castling(mv)
        seen["ep"] += b.is_en_passant(mv)
        seen["promo"] += mv.promotion is not None
    assert sum(seen.values()) > 0


def test_incremental_matches_reference_random_games():
    n_moves = 0
    for b in _random_positions(n_games=25, plies=60, seed=7):
        occ0 = board_to_occupancy(b)
        for mv in b.legal_moves:
            assert expected_occupancy_after_move(b, mv, occ0) == _reference_expected(b, mv), (b.fen(), mv)
            n_moves += 1
    assert n_moves > 5000


def test_every_legal_move_changes_at_least_two_cells():
    """A rendszer sérthetetlen invariánsa — itt is rögzítve."""
    for b in _random_positions(n_games=10, plies=80, seed=3) + [chess.Board(f) for f in SPECIAL_FENS]:
        occ0 = board_to_occupancy(b)
        for mv in b.legal_moves:
            occ1 = expected_occupancy_after_move(b, mv, occ0)
            n = sum(occ0[r][c] != occ1[r][c] for r in range(8) for c in range(8))
            assert n >= 2, (b.fen(), mv, n)


def test_resolver_exact_and_fuzzy_unchanged_vs_reference():
    """Ugyanazt a lépést adja, mint a régi (másolós) megvalósítás."""
    rng = random.Random(11)
    for b in _random_positions(n_games=8, plies=40, seed=5):
        moves = list(b.legal_moves)
        if not moves:
            continue
        mv = rng.choice(moves)
        expected = _reference_expected(b, mv)
        conf = [[0.97] * 8 for _ in range(8)]
        guess, occ, mode = resolve_move_from_occupancy(b, expected, conf, max_noise_cells=1, max_weighted_cost=0.9)
        assert guess is not None and mode == "exact" and occ == expected
        # promóciónál tipp nélkül a vezér az alapértelmezés (régi viselkedés)
        exp_uci = mv.uci() if mv.promotion is None else mv.uci()[:4] + "q"
        assert guess.to_uci() == exp_uci

        # fuzzy: a célmező színe rossz (fehér <-> fekete), a geometria jó.
        # Ütésnél a "rossz szín" az ütés előtti tartalom -> 1 mezős eltérés,
        # azt az invariáns miatt (helyesen) nem oldjuk fel, ezért csak csendes lépésre.
        if not b.is_capture(mv):
            tr, tc = chess_square_to_coords(mv.to_square)
            noisy = [row[:] for row in expected]
            noisy[tr][tc] = 2 if noisy[tr][tc] == 1 else 1
            conf[tr][tc] = 0.55
            g2, occ2, mode2 = resolve_move_from_occupancy(b, noisy, conf, max_noise_cells=1, max_weighted_cost=0.9)
            assert g2 is not None and mode2.startswith("fuzzy") and occ2 == expected
            assert g2.to_uci() == exp_uci


def test_single_cell_difference_is_never_resolved():
    """1 mezős eltérés = zaj: sem exact, sem fuzzy lépés (régen a fuzzy ág a
    legkisebb konfidenciájú célmezőt tippelte hozzá)."""
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    occ = board_to_occupancy(b)
    # a f3 huszár "eltűnik"
    r, c = chess_square_to_coords(chess.F3)
    occ[r][c] = 0
    conf = [[0.99] * 8 for _ in range(8)]
    # minden lehetséges célmező bizonytalan -> a régi fuzzy ág itt lépést adott volna
    for sq in (chess.G5, chess.E5, chess.D4, chess.H4, chess.G1):
        rr, cc = chess_square_to_coords(sq)
        conf[rr][cc] = 0.3
    assert resolve_move_from_occupancy(b, occ, conf, max_noise_cells=1, max_weighted_cost=0.9) == (None, None, None)
    # a régi viselkedés explicit kéréssel elérhető (min_changed_cells=1) — de nem alapértelmezett
    g, _, mode = resolve_move_from_occupancy(b, occ, conf, max_noise_cells=1, max_weighted_cost=0.9, min_changed_cells=1)
    assert g is not None and mode.startswith("fuzzy")


def _type_grid(fill_idx: int | None = None, at: tuple[int, int] | None = None, vec=None):
    g = np.zeros((8, 8, 7), dtype=np.float32)
    g[..., TYPE_INDEX["none"]] = 1.0
    if at is not None and vec is not None:
        g[at[0], at[1]] = vec
    return g


def test_promotion_uses_type_head_hint():
    b = chess.Board("4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1")
    mv = chess.Move.from_uci("a7a8n")
    expected = _reference_expected(b, mv)
    conf = [[0.97] * 8 for _ in range(8)]
    tr, tc = chess_square_to_coords(mv.to_square)

    # tipp nélkül: vezér
    g, _, _ = resolve_move_from_occupancy(b, expected, conf)
    assert g.to_uci() == "a7a8q"

    # magabiztos huszár-tipp -> huszár
    vec = np.zeros(7, np.float32)
    vec[PROMOTION_TYPE_INDEX["n"]] = 0.85
    vec[PROMOTION_TYPE_INDEX["q"]] = 0.10
    g, _, mode = resolve_move_from_occupancy(b, expected, conf, type_probs=_type_grid(at=(tr, tc), vec=vec))
    assert g.to_uci() == "a7a8n" and mode == "exact"

    # bizonytalan tipp (min_conf alatt) -> marad a vezér
    weak = np.zeros(7, np.float32)
    weak[PROMOTION_TYPE_INDEX["r"]] = DEFAULT_PROMOTION_MIN_CONF - 0.05
    weak[PROMOTION_TYPE_INDEX["q"]] = 0.30
    g, _, _ = resolve_move_from_occupancy(b, expected, conf, type_probs=_type_grid(at=(tr, tc), vec=weak))
    assert g.to_uci() == "a7a8q"

    # a tipp csak a promóciós bábut érinti: nem-promóciós lépésnél nincs hatása
    mv2 = chess.Move.from_uci("e1d1")
    g2, _, _ = resolve_move_from_occupancy(b, _reference_expected(b, mv2), conf, type_probs=_type_grid(at=(tr, tc), vec=vec))
    assert g2.to_uci() == "e1d1"


def test_promotion_preference_ordering():
    vec = np.zeros(7)
    vec[PROMOTION_TYPE_INDEX["b"]] = 0.6
    vec[PROMOTION_TYPE_INDEX["r"]] = 0.3
    g = _type_grid(at=(0, 0), vec=vec)
    assert promotion_preference(g, 0, 0) == ["b", "r", "q", "n"]
    assert promotion_preference(None, 0, 0) == ["q", "r", "b", "n"]
    assert promotion_preference(g, 3, 3) == ["q", "r", "b", "n"]     # ott nincs tipp (none=1.0)


def test_type_hint_switch_off_is_identical():
    """A kapcsoló alapból ki: a típus-rács jelenléte nem változtat a nem-promóciós döntésen."""
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    mv = chess.Move.from_uci("f1c4")
    expected = _reference_expected(b, mv)
    conf = [[0.97] * 8 for _ in range(8)]
    rng = np.random.default_rng(0)
    tp = rng.random((8, 8, 7)).astype(np.float32)
    a = resolve_move_from_occupancy(b, expected, conf, type_probs=tp)
    c = resolve_move_from_occupancy(b, expected, conf)
    assert a[0].to_uci() == c[0].to_uci() == "f1c4" and a[2] == c[2]
    # bekapcsolva is csak egyenlő költségű fuzzy jelöltek között dönt -> exact találatnál ugyanaz
    d = resolve_move_from_occupancy(b, expected, conf, type_probs=tp, use_type_hint_for_moves=True)
    assert d[0].to_uci() == "f1c4"


def test_type_index_contract_matches_square_net():
    """A chess_logic saját típus-sorrendje == vision/models/square_net.PIECE_TYPE_CLASSES."""
    from vision.models.square_net import PIECE_TYPE_CLASSES, TYPE_IDX_TO_PROMOTION_CHAR
    assert tuple(TYPE_INDEX) == PIECE_TYPE_CLASSES
    for idx, ch in TYPE_IDX_TO_PROMOTION_CHAR.items():
        assert PROMOTION_TYPE_INDEX[ch] == idx


def test_resolver_is_fast():
    import time
    b = chess.Board("r1bqk2r/pp2bppp/2n2n2/3pp3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 7")
    occ = board_to_occupancy(b)
    conf = [[0.97] * 8 for _ in range(8)]
    mv = list(b.legal_moves)[-1]
    exp = _reference_expected(b, mv)
    for _ in range(5):
        resolve_move_from_occupancy(b, exp, conf, max_noise_cells=1)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        resolve_move_from_occupancy(b, occ, conf, max_noise_cells=1, min_changed_cells=0)   # teljes fuzzy scan
    ms = (time.perf_counter() - t0) * 1000 / n
    assert ms < 3.0, f"resolver fuzzy scan {ms:.2f} ms"


def test_fuzzy_requires_destination_seen_occupied():
    """Félkész ütés (leütött bábu levéve, a lépő még kézben): from üres, to üres
    -> 2 mezős eltérés, de a célmező üres -> NEM fogadjuk el."""
    b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    occ = board_to_occupancy(b)
    fr, fc = chess_square_to_coords(chess.E4)
    tr, tc = chess_square_to_coords(chess.D5)
    occ[fr][fc] = 0
    occ[tr][tc] = 0
    conf = [[0.97] * 8 for _ in range(8)]
    conf[tr][tc] = 0.6
    assert resolve_move_from_occupancy(b, occ, conf, max_noise_cells=1, max_weighted_cost=0.9) == (None, None, None)
    # ütésnél a "téves szín" a célmezőn az ütés előtti tartalom -> 1 mezős eltérés -> szintén nincs lépés
    occ[tr][tc] = 2
    assert resolve_move_from_occupancy(b, occ, conf, max_noise_cells=1, max_weighted_cost=0.9) == (None, None, None)
    # csendes lépésnél (e4e5) a téves színű, de FOGLALT célmező fuzzy-val feloldható
    occ = board_to_occupancy(b)
    occ[fr][fc] = 0
    r5, c5 = chess_square_to_coords(chess.E5)
    occ[r5][c5] = 2
    conf[r5][c5] = 0.6
    g, _, mode = resolve_move_from_occupancy(b, occ, conf, max_noise_cells=1, max_weighted_cost=0.9)
    assert g is not None and g.to_uci() == "e4e5" and mode.startswith("fuzzy")


def test_prefix_ambiguity_castling():
    b = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    assert prefix_ambiguities(b, "h1f1") == ["e1g1"]
    assert prefix_ambiguities(b, "a1d1") == ["e1c1"]
    assert prefix_ambiguities(b, "e1g1") == []
    # a reláció foglaltság-szintű: Rg1 is a sánc részhalmaza (h1 ürül, g1 fehér lesz) -> konzervatívan ez is várakozik
    assert prefix_ambiguities(b, "h1g1") == ["e1g1"]
    assert prefix_ambiguities(b, "a1b1") == []
    assert prefix_ambiguities(b, "a2a4") == []
    # sánc joga nélkül a bástyalépés nem kétértelmű
    assert prefix_ambiguities(chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"), "h1f1") == []


def test_game_result_carries_ambiguity():
    from chess_logic import Game
    game = Game("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    occ = board_to_occupancy(game.board)
    r0, c0 = chess_square_to_coords(chess.H1)
    r1, c1 = chess_square_to_coords(chess.F1)
    occ[r0][c0] = 0
    occ[r1][c1] = 1
    res = game.resolve_from_occupancy(occ, [[0.97] * 8 for _ in range(8)], max_noise_cells=1, max_weighted_cost=0.9)
    assert res.move is not None and res.move.to_uci() == "h1f1" and res.ambiguous_with == ["e1g1"]
