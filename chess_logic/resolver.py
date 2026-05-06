from __future__ import annotations

from typing import Iterable

import chess

from .move_types import MoveGuess


def coords_to_chess_square(row: int, col: int) -> int:
    rank = 7 - row
    file_ = col
    return chess.square(file_, rank)


def chess_square_to_coords(square: int) -> tuple[int, int]:
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    return 7 - rank, file_


def _promotion_char(piece_type: int) -> str:
    mapping = {
        chess.QUEEN: "q",
        chess.ROOK: "r",
        chess.BISHOP: "b",
        chess.KNIGHT: "n",
    }
    return mapping.get(piece_type, "q")


def _fen_key4(board: chess.Board) -> str:
    parts = board.fen().split()
    return " ".join(parts[:4])


def _as_chess_board(obj: chess.Board | str) -> chess.Board:
    if isinstance(obj, chess.Board):
        return obj.copy()

    if isinstance(obj, str):
        return chess.Board(obj)

    raise TypeError(f"Nem támogatott board típus: {type(obj)!r}")


def board_to_occupancy(board: chess.Board | str) -> list[list[int]]:
    ch_board = _as_chess_board(board)
    occ = [[0] * 8 for _ in range(8)]

    for sq in chess.SQUARES:
        piece = ch_board.piece_at(sq)
        if piece is None:
            continue

        file_ = chess.square_file(sq)
        rank = chess.square_rank(sq)
        row = 7 - rank
        col = file_

        occ[row][col] = 1 if piece.color == chess.WHITE else 2

    return occ


def same_occupancy(a: Iterable[Iterable[int]], b: Iterable[Iterable[int]]) -> bool:
    a_rows = list(a)
    b_rows = list(b)

    if len(a_rows) != 8 or len(b_rows) != 8:
        return False

    for r in range(8):
        ra = list(a_rows[r])
        rb = list(b_rows[r])
        if len(ra) != 8 or len(rb) != 8:
            return False
        for c in range(8):
            if int(ra[c]) != int(rb[c]):
                return False
    return True


def occupancy_distance(a: Iterable[Iterable[int]], b: Iterable[Iterable[int]]) -> int:
    a_rows = list(a)
    b_rows = list(b)

    diff = 0
    for r in range(8):
        ra = list(a_rows[r])
        rb = list(b_rows[r])
        for c in range(8):
            if int(ra[c]) != int(rb[c]):
                diff += 1
    return diff


def weighted_diff(
    observed_occ: Iterable[Iterable[int]],
    expected_occ: Iterable[Iterable[int]],
    observed_conf: Iterable[Iterable[float]],
) -> float:
    obs_rows = list(observed_occ)
    exp_rows = list(expected_occ)
    conf_rows = list(observed_conf)

    total = 0.0
    for r in range(8):
        o = list(obs_rows[r])
        e = list(exp_rows[r])
        cf = list(conf_rows[r])
        for c in range(8):
            if int(o[c]) != int(e[c]):
                total += float(cf[c])
    return total


def _promotion_priority(move: chess.Move) -> int:
    if move.promotion is None:
        return 0
    if move.promotion == chess.QUEEN:
        return 1
    return 2


def _ordered_legal_moves(ch_board: chess.Board) -> list[chess.Move]:
    return sorted(ch_board.legal_moves, key=_promotion_priority)


def chess_move_to_moveguess(board, move: chess.Move) -> MoveGuess:
    ch_board = _as_chess_board(board)

    moving_piece = ch_board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError("Nincs bábu a from_square-on.")

    from_row, from_col = chess_square_to_coords(move.from_square)
    to_row, to_col = chess_square_to_coords(move.to_square)

    piece_symbol = moving_piece.symbol()
    is_castling = ch_board.is_castling(move)
    is_en_passant = ch_board.is_en_passant(move)
    captured = ch_board.is_capture(move)

    promotion_piece = None
    if move.promotion is not None:
        promotion_piece = _promotion_char(move.promotion)

    castling_color = None
    castling_side = None
    if is_castling:
        castling_color = "w" if moving_piece.color == chess.WHITE else "b"
        castling_side = "K" if to_col > from_col else "Q"

    return MoveGuess(
        from_row=from_row,
        from_col=from_col,
        to_row=to_row,
        to_col=to_col,
        piece=piece_symbol,
        captured=captured,
        is_castling=is_castling,
        castling_color=castling_color,
        castling_side=castling_side,
        is_en_passant=is_en_passant,
        promotion_piece=promotion_piece,
    )




def resolve_move_from_occupancy(
    current_board,
    observed_occ: list[list[int]],
    observed_conf: list[list[float]] | None = None,
    *,
    max_noise_cells: int = 2,
    max_weighted_cost: float = 1.2,
) -> tuple[MoveGuess | None, list[list[int]] | None, str | None]:
    """
    Exact, majd opcionálisan fuzzy legal move feloldás.
    A sakklogika teljes egészében itt marad, nem a vision trackerben.
    """
    ch_board = _as_chess_board(current_board)

    best_exact_move: MoveGuess | None = None
    best_exact_occ: list[list[int]] | None = None

    for mv in _ordered_legal_moves(ch_board):
        tmp = ch_board.copy()
        tmp.push(mv)
        expected_occ = board_to_occupancy(tmp)

        if same_occupancy(expected_occ, observed_occ):
            best_exact_move = chess_move_to_moveguess(ch_board, mv)
            best_exact_occ = expected_occ
            break

    if best_exact_move is not None:
        return best_exact_move, best_exact_occ, "exact"

    if observed_conf is None:
        return None, None, None

    best = None

    for mv in _ordered_legal_moves(ch_board):
        tmp = ch_board.copy()
        tmp.push(mv)
        expected_occ = board_to_occupancy(tmp)

        noise = occupancy_distance(expected_occ, observed_occ)
        if noise > max_noise_cells:
            continue

        cost = weighted_diff(observed_occ, expected_occ, observed_conf)

        if best is None or (cost < best[0]) or (cost == best[0] and noise < best[1]):
            best = (cost, noise, mv, expected_occ)

    if best is None:
        return None, None, None

    cost, noise, mv, expected_occ = best
    if cost > max_weighted_cost:
        return None, None, None

    return (
        chess_move_to_moveguess(ch_board, mv),
        expected_occ,
        f"fuzzy c={cost:.2f} n={noise}",
    )
