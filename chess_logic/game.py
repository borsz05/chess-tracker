from __future__ import annotations

import chess
import chess.pgn

from .resolver import chess_move_to_moveguess, resolve_move_from_occupancy
from .types import MoveGuess, MoveRecord, OccupancyResolveResult


class Board(chess.Board):
    """Vékony wrapper a python-chess fölött."""

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
    """A belső sakkállapot egyetlen igaz forrása."""

    def __init__(self, start_fen: str):
        self.board = Board(start_fen)
        self.start_fen = self.board.to_fen()

        self.move_history: list[MoveRecord] = []
        self.result: str = "*"

        self.last_top_lines: list[dict] | None = None
        self.analysis_pending: bool = False

    def _current_ply_index(self) -> int:
        return len(self.move_history) + 1

    def _build_record(self, ch_before: chess.Board, ch_move: chess.Move, move_guess: MoveGuess) -> MoveRecord:
        ch_after = ch_before.copy()
        ch_after.push(ch_move)

        outcome = ch_after.outcome(claim_draw=True)
        result_after = outcome.result() if outcome is not None else None

        return MoveRecord(
            ply_index=self._current_ply_index(),
            move=move_guess,
            fen_before=ch_before.fen(),
            fen_after=ch_after.fen(),
            san=ch_before.san(ch_move),
            is_check=ch_after.is_check(),
            is_checkmate=ch_after.is_checkmate(),
            is_stalemate=ch_after.is_stalemate(),
            result_after_move=result_after,
            is_fifty_move_draw=ch_after.halfmove_clock >= 100,
            is_threefold_repetition=ch_after.is_repetition(3),
            is_insufficient_material=ch_after.is_insufficient_material(),
        )

    def _can_apply_move(self, ch_move: chess.Move, ch_before: chess.Board) -> bool:
        if self.result != "*":
            return False
        return ch_move in ch_before.legal_moves

    def _store_applied_move(self, record: MoveRecord):
        self.board.push(chess.Move.from_uci(record.move.to_uci()))
        self.move_history.append(record)
        self.result = record.result_after_move or "*"

    def _apply_chess_move(self, ch_move: chess.Move) -> MoveGuess | None:
        ch_before = self.board.copy()
        if not self._can_apply_move(ch_move, ch_before):
            return None

        move_guess = chess_move_to_moveguess(ch_before, ch_move)
        record = self._build_record(ch_before, ch_move, move_guess)
        self._store_applied_move(record)
        return move_guess

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

    def _normalized_result(self, result: str | None) -> str:
        if result is None:
            result = self.result
        if result in ("1-0", "0-1", "1/2-1/2"):
            return result
        return "*"

    def _build_pgn_headers(
        self,
        game: chess.pgn.Game,
        *,
        result: str,
        white: str,
        black: str,
        event: str,
        site: str,
        date: str,
    ) -> None:
        game.headers["Event"] = event
        game.headers["Site"] = site
        game.headers["Date"] = date
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Result"] = result

        if self.start_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.start_fen

    def to_pgn(
        self,
        result: str | None = None,
        white: str = "White",
        black: str = "Black",
        event: str = "?",
        site: str = "?",
        date: str = "????.??.??",
    ) -> str:
        game = chess.pgn.Game()
        self._build_pgn_headers(
            game,
            result=self._normalized_result(result),
            white=white,
            black=black,
            event=event,
            site=site,
            date=date,
        )

        node = game
        replay = Board(self.start_fen)

        for record in self.move_history:
            move = chess.Move.from_uci(record.move.to_uci())
            if move not in replay.legal_moves:
                break
            node = node.add_variation(move)
            replay.push(move)

        return str(game).strip()

    def resolve_from_occupancy(
        self,
        new_occ: list[list[int]],
        confs: list[list[float]] | None = None,
        *,
        max_noise_cells: int = 2,
        max_weighted_cost: float = 1.2,
    ) -> OccupancyResolveResult:
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

    def _build_status_dict(self, last_record: MoveRecord | None) -> dict:
        return {
            "result": self.result,
            "is_check": last_record.is_check if last_record else False,
            "is_checkmate": last_record.is_checkmate if last_record else False,
            "is_stalemate": last_record.is_stalemate if last_record else False,
            "is_fifty_move_draw": last_record.is_fifty_move_draw if last_record else False,
            "is_threefold_repetition": last_record.is_threefold_repetition if last_record else False,
            "is_insufficient_material": last_record.is_insufficient_material if last_record else False,
        }

    def _build_last_move_dict(self, last_record: MoveRecord | None) -> dict | None:
        if last_record is None:
            return None

        move = last_record.move
        return {
            "ply_index": last_record.ply_index,
            "san": last_record.san,
            "uci": move.to_uci(),
            "from": {"row": move.from_row, "col": move.from_col},
            "to": {"row": move.to_row, "col": move.to_col},
        }

    def _build_moves_list(self) -> list[dict]:
        moves: list[dict] = []

        for record in self.move_history:
            moves.append({
                "ply_index": record.ply_index,
                "move_no": (record.ply_index + 1) // 2,
                "color": "w" if record.ply_index % 2 == 1 else "b",
                "san": record.san,
                "uci": record.move.to_uci(),
                "fen_after": record.fen_after,
                "engine_eval": record.engine_eval,
                "engine_bestmove": record.engine_bestmove,
            })

        return moves

    def to_state_dict(self) -> dict:
        last_record = self.move_history[-1] if self.move_history else None

        return {
            "fen": self.board.to_fen(),
            "side_to_move": self.board.side_to_move,
            "fullmove_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "status": self._build_status_dict(last_record),
            "result": self.result,
            "last_move": self._build_last_move_dict(last_record),
            "moves": self._build_moves_list(),
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
