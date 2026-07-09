"""
Move descriptor: converts a UCI string + python-chess Board into a
structured dict that tells the robot exactly what physical actions to take.

Supported move types
--------------------
  "simple"    — move a piece from one square to another
  "capture"   — move a piece and remove the captured piece to the graveyard
  "castling"  — two separate piece moves (king then rook)
  "en_passant"  — like capture but the captured pawn is on a different square
                  than the destination
  "promotion" — pawn reaches back rank: pawn goes to graveyard, spare queen
                is picked up from beside the board and placed on the target square

The descriptor always contains:
  type          : str                     one of the types above
  uci           : str                     original UCI string
  piece_from_xy : (x, y)                  pick-up position
  piece_to_xy   : (x, y)                  place-down position
  captured_xy   : (x, y) | None           where to pick up the captured piece
  graveyard_xy  : (x, y) | None           where to drop the captured piece
  castling_rook : dict | None             rook sub-move for castling
    └─ rook_from_xy / rook_to_xy : (x, y)

Usage:
    import chess
    from robot.calibration import Calibration
    from robot.graveyard import Graveyard
    from robot.move_descriptor import build_move_descriptor

    cal      = Calibration.load()
    graveyard = Graveyard.from_calibration_file()
    board    = chess.Board()

    descriptor = build_move_descriptor("e2e4", board, cal, graveyard)
    # → {"type": "simple", "uci": "e2e4", "piece_from_xy": (...), ...}
"""
from __future__ import annotations

from typing import Any

import chess

from robot.calibration import Calibration
from robot.graveyard import Graveyard


def build_move_descriptor(
    uci: str,
    board: chess.Board,
    calibration: Calibration,
    graveyard: Graveyard,
) -> dict[str, Any]:
    """Return a move descriptor dict for the robot.

    Parameters
    ----------
    uci         : UCI move string, e.g. "e2e4" or "e1g1" (castling)
    board       : python-chess Board *before* the move is applied
    calibration : loaded Calibration object
    graveyard   : Graveyard tracker (will call next_slot() on captures)
    """
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {uci!r} on current board")

    from_sq_name = chess.square_name(move.from_square)
    to_sq_name   = chess.square_name(move.to_square)

    piece_from_xy = calibration.square_to_xy(from_sq_name)
    piece_to_xy   = calibration.square_to_xy(to_sq_name)

    moving_piece = board.piece_at(move.from_square)
    piece_symbol = moving_piece.symbol().upper() if moving_piece else "P"

    descriptor: dict[str, Any] = {
        "uci":                 uci,
        "piece":               piece_symbol,
        "piece_from_xy":       piece_from_xy,
        "piece_to_xy":         piece_to_xy,
        "captured_xy":         None,
        "captured_piece":      None,
        "graveyard_xy":        None,
        "castling_rook":       None,
        "promotion_target_xy": None,
        "pawn_graveyard_xy":   None,
        "promotion":           chess.piece_name(move.promotion) if move.promotion else None,
    }

    # ── Promotion ─────────────────────────────────────────────────────
    if move.promotion is not None:
        descriptor["type"] = "promotion"
        descriptor["piece_to_xy"] = calibration.promotion_queen_xy
        descriptor["promotion_target_xy"] = calibration.square_to_xy(to_sq_name)
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            descriptor["captured_piece"] = captured.symbol().upper() if captured else "P"
            descriptor["captured_xy"]    = piece_to_xy
            descriptor["graveyard_xy"]   = graveyard.next_slot()
        descriptor["pawn_graveyard_xy"] = graveyard.next_slot()
        return descriptor

    # ── Castling ──────────────────────────────────────────────────────
    if board.is_castling(move):
        descriptor["type"] = "castling"
        rook_from_sq, rook_to_sq = _castling_rook_squares(board, move)
        descriptor["castling_rook"] = {
            "rook_from_xy": calibration.square_to_xy(chess.square_name(rook_from_sq)),
            "rook_to_xy":   calibration.square_to_xy(chess.square_name(rook_to_sq)),
        }
        return descriptor

    # ── En passant ────────────────────────────────────────────────────
    if board.is_en_passant(move):
        descriptor["type"] = "en_passant"
        # The captured pawn sits on the same file as destination but on the
        # rank of the moving pawn's origin
        ep_capture_sq = _en_passant_capture_square(board, move)
        descriptor["captured_piece"] = "P"
        descriptor["captured_xy"]    = calibration.square_to_xy(chess.square_name(ep_capture_sq))
        descriptor["graveyard_xy"]   = graveyard.next_slot()
        return descriptor

    # ── Capture ───────────────────────────────────────────────────────
    if board.is_capture(move):
        descriptor["type"] = "capture"
        captured = board.piece_at(move.to_square)
        descriptor["captured_piece"] = captured.symbol().upper() if captured else "P"
        descriptor["captured_xy"]    = piece_to_xy   # piece already on destination
        descriptor["graveyard_xy"]   = graveyard.next_slot()
        return descriptor

    # ── Simple move ───────────────────────────────────────────────────
    descriptor["type"] = "simple"
    return descriptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _castling_rook_squares(
    board: chess.Board, move: chess.Move
) -> tuple[chess.Square, chess.Square]:
    """Return (rook_from, rook_to) squares for a castling move."""
    king_to = move.to_square
    is_kingside = chess.square_file(king_to) == 6  # g-file

    if board.turn == chess.WHITE:
        rook_from = chess.H1 if is_kingside else chess.A1
        rook_to   = chess.F1 if is_kingside else chess.D1
    else:
        rook_from = chess.H8 if is_kingside else chess.A8
        rook_to   = chess.F8 if is_kingside else chess.D8

    return rook_from, rook_to


def _en_passant_capture_square(
    board: chess.Board, move: chess.Move
) -> chess.Square:
    """Return the square of the pawn captured by en passant."""
    # The captured pawn is on the same file as the destination but
    # on the same rank as the moving piece's origin.
    dest_file = chess.square_file(move.to_square)
    src_rank  = chess.square_rank(move.from_square)
    return chess.square(dest_file, src_rank)
