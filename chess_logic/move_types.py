from __future__ import annotations

from dataclasses import dataclass, field

FILES = "abcdefgh"


def coords_to_uci(
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    promotion: str | None = None,
) -> str:
    from_file = FILES[from_col]
    from_rank = 8 - from_row
    to_file = FILES[to_col]
    to_rank = 8 - to_row

    uci = f"{from_file}{from_rank}{to_file}{to_rank}"
    if promotion:
        uci += promotion.lower()
    return uci


@dataclass(slots=True)
class MoveGuess:
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    piece: str

    captured: bool = False

    is_castling: bool = False
    castling_color: str | None = None
    castling_side: str | None = None

    is_en_passant: bool = False
    promotion_piece: str | None = None

    def to_uci(self) -> str:
        return coords_to_uci(
            self.from_row,
            self.from_col,
            self.to_row,
            self.to_col,
            self.promotion_piece,
        )


@dataclass(slots=True)
class MoveRecord:
    ply_index: int
    move: MoveGuess
    fen_before: str
    fen_after: str
    san: str | None = None

    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    result_after_move: str | None = None

    is_fifty_move_draw: bool = False
    is_threefold_repetition: bool = False
    is_insufficient_material: bool = False


@dataclass(slots=True)
class OccupancyResolveResult:
    applied: bool
    move: MoveGuess | None = None
    san: str | None = None
    mode: str | None = None
    expected_occ: list[list[int]] | None = None
    # Legális lépések (UCI), amelyeknek a talált lépés a foglaltság szintjén
    # előtagja (pl. Rf1 <- O-O bástyával kezdve). Ha nem üres, a tracker
    # hosszabb megerősítést vár. Lásd resolver.prefix_ambiguities.
    ambiguous_with: list[str] = field(default_factory=list)