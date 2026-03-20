from __future__ import annotations

import chess
import chess.pgn

from .resolver import (
    chess_move_to_moveguess,
    resolve_move_from_occupancy,
)
from .types import MoveGuess, MoveRecord, OccupancyResolveResult


class Board(chess.Board):
    """
    Minimális kompatibilitási réteg a python-chess fölött.
    """

    def __init__(self, fen: str | None = None):
        super().__init__(fen or chess.STARTING_FEN)

    def to_fen(self) -> str:
        return self.fen()

    @property
    def side_to_move(self) -> str:
        return "w" if self.turn == chess.WHITE else "b"

    def copy(self, *, stack: bool = True) -> "Board":
        copied = super().copy(stack=stack)
        return Board(copied.fen())


class Game:
    """
    A játékállapot egyetlen igazságforrása a python-chess.
    """

    def __init__(self, start_fen: str):
        self.board = Board(start_fen)
        self.start_fen = self.board.to_fen()

        self.move_history: list[MoveRecord] = []
        self.result: str = "*"

        self.last_top_lines: list[dict] | None = None
        self.analysis_pending: bool = False

    def _build_record(
        self,
        ch_before: chess.Board,
        ch_move: chess.Move,
        move_guess: MoveGuess,
    ) -> MoveRecord:
        fen_before = ch_before.fen()
        san = ch_before.san(ch_move)

        ch_after = ch_before.copy()
        ch_after.push(ch_move)

        fen_after = ch_after.fen()
        outcome = ch_after.outcome(claim_draw=True)
        result_after = outcome.result() if outcome is not None else None

        is_check = ch_after.is_check()
        is_checkmate = ch_after.is_checkmate()
        is_stalemate = ch_after.is_stalemate()

        draw_by_fifty = ch_after.halfmove_clock >= 100
        draw_by_repetition = ch_after.is_repetition(3)
        draw_by_insufficient = ch_after.is_insufficient_material()

        return MoveRecord(
            ply_index=len(self.move_history) + 1,
            move=move_guess,
            fen_before=fen_before,
            fen_after=fen_after,
            san=san,
            is_check=is_check,
            is_checkmate=is_checkmate,
            is_stalemate=is_stalemate,
            result_after_move=result_after,
            is_fifty_move_draw=draw_by_fifty,
            is_threefold_repetition=draw_by_repetition,
            is_insufficient_material=draw_by_insufficient,
        )

    def _apply_chess_move(self, ch_move: chess.Move) -> MoveGuess | None:
        if self.result != "*":
            return None

        ch_before = self.board.copy()

        if ch_move not in ch_before.legal_moves:
            return None

        canonical_guess = chess_move_to_moveguess(ch_before, ch_move)
        record = self._build_record(ch_before, ch_move, canonical_guess)

        self.board.push(ch_move)
        self.move_history.append(record)
        self.result = record.result_after_move or "*"

        return canonical_guess

    def apply_uci(self, uci: str) -> MoveGuess | None:
        try:
            ch_move = chess.Move.from_uci(uci.strip())
        except Exception:
            return None

        return self._apply_chess_move(ch_move)

    def apply_move(self, move: MoveGuess) -> bool:
        try:
            ch_move = chess.Move.from_uci(move.to_uci())
        except Exception:
            return False

        return self._apply_chess_move(ch_move) is not None

    def to_pgn(
        self,
        result: str | None = None,
        white: str = "White",
        black: str = "Black",
        event: str = "?",
        site: str = "?",
        date: str = "????.??.??",
    ) -> str:
        if result is None:
            result = self.result
        if result not in ("1-0", "0-1", "1/2-1/2"):
            result = "*"

        game = chess.pgn.Game()
        game.headers["Event"] = event
        game.headers["Site"] = site
        game.headers["Date"] = date
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Result"] = result

        if self.start_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.start_fen

        node = game
        replay = Board(self.start_fen)

        for rec in self.move_history:
            mv = chess.Move.from_uci(rec.move.to_uci())
            if mv not in replay.legal_moves:
                break
            node = node.add_variation(mv)
            replay.push(mv)

        return str(game).strip()

    def resolve_from_occupancy(
        self,
        new_occ: list[list[int]],
        confs: list[list[float]] | None = None,
        *,
        max_noise_cells: int = 2,
        max_weighted_cost: float = 1.2,
    ) -> OccupancyResolveResult:
        """
        Csak feloldja a lépést, de NEM módosítja a belső Game/Board állapotot.
        A tényleges apply a tracker vote-ja után történjen.
        """
        move, expected_occ, mode = resolve_move_from_occupancy(
            self.board,
            new_occ,
            confs,
            max_noise_cells=max_noise_cells,
            max_weighted_cost=max_weighted_cost,
        )

        if move is None:
            return OccupancyResolveResult(
                applied=False,
                move=None,
                san=None,
                mode="no-legal-fit",
                expected_occ=None,
            )

        return OccupancyResolveResult(
            applied=False,
            move=move,
            san=None,
            mode=mode or "exact",
            expected_occ=expected_occ,
        )

    def to_state_dict(self) -> dict:
        fen = self.board.to_fen()
        side_to_move = self.board.side_to_move
        fullmove_number = self.board.fullmove_number
        halfmove_clock = self.board.halfmove_clock

        last_rec = self.move_history[-1] if self.move_history else None

        status = {
            "result": self.result,
            "is_check": last_rec.is_check if last_rec else False,
            "is_checkmate": last_rec.is_checkmate if last_rec else False,
            "is_stalemate": last_rec.is_stalemate if last_rec else False,
            "is_fifty_move_draw": last_rec.is_fifty_move_draw if last_rec else False,
            "is_threefold_repetition": last_rec.is_threefold_repetition if last_rec else False,
            "is_insufficient_material": last_rec.is_insufficient_material if last_rec else False,
        }

        last_move = None
        if last_rec is not None:
            m = last_rec.move
            last_move = {
                "ply_index": last_rec.ply_index,
                "san": last_rec.san,
                "uci": m.to_uci(),
                "from": {"row": m.from_row, "col": m.from_col},
                "to": {"row": m.to_row, "col": m.to_col},
            }

        moves: list[dict] = []
        for rec in self.move_history:
            move_no = (rec.ply_index + 1) // 2
            color = "w" if rec.ply_index % 2 == 1 else "b"

            moves.append({
                "ply_index": rec.ply_index,
                "move_no": move_no,
                "color": color,
                "san": rec.san,
                "uci": rec.move.to_uci(),
                "fen_after": rec.fen_after,
                "engine_eval": rec.engine_eval,
                "engine_bestmove": rec.engine_bestmove,
            })

        return {
            "fen": fen,
            "side_to_move": side_to_move,
            "fullmove_number": fullmove_number,
            "halfmove_clock": halfmove_clock,
            "status": status,
            "result": self.result,
            "last_move": last_move,
            "moves": moves,
            "pgn": self.to_pgn(),
            "top_lines": self.last_top_lines,
            "analysis_pending": self.analysis_pending,
        }

    def reset(self, fen: str) -> None:
        self.board = Board(fen)
        self.start_fen = self.board.to_fen()
        self.move_history.clear()
        self.result = "*"
        self.last_top_lines = None
        self.analysis_pending = False
