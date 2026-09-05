"""
FEN -> mezőnkénti címkék a NYERS (kamera/warp) rácsban.

A pipeline a klasszifikátor nyers 8x8 rácsát (bbox_warp[r][c] sorrend: r = a
warpolt kép sora fentről, c = oszlopa balról) a `raw_to_standard` (rot90 +
fliplr) függvénnyel fordítja "standard" rácsra, ahol row 0 = 8. sor, col 0 =
a-vonal (lásd chess_logic.resolver.board_to_occupancy). A tanítóadat-gyűjtő
ezt az utat FORDÍTVA járja: FEN -> standard rács -> nyers rács, hogy a
kivágott ROI-k (nyers r,c) helyes címkét kapjanak.

Az inverz leképezést NEM kézzel írt képlettel, hanem a tényleges
`raw_to_standard` numerikus invertálásával számoljuk — így ha a pipeline
orientációja valaha változik, ez a modul automatikusan követi.
(Az aktuális kód szerint: raw (r, c) -> standard (7 - c, 7 - r), tehát a
warpolt kép bal-felső mezője H1, bal-alsó A1, jobb-felső H8.)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chess
import numpy as np

from chess_logic.resolver import board_to_occupancy
from vision.models.square_net import PIECE_SYMBOL_TO_TYPE_IDX, PIECE_TYPE_CLASSES
from vision.pipeline.tracker import raw_to_standard

OCC_NAME = {0: "empty", 1: "white", 2: "black"}
EMPTY_SYMBOL = "x"   # fájlnév-tag üres mezőre


@lru_cache(maxsize=1)
def _std_to_raw_lut() -> np.ndarray:
    """LUT: [std_row, std_col] -> (raw_r, raw_c), a raw_to_standard invertálásával."""
    raw_idx = np.arange(64, dtype=np.int64).reshape(8, 8)      # raw (r,c) -> r*8+c
    std = raw_to_standard(raw_idx)                              # std[row,col] = raw index
    lut = np.zeros((8, 8, 2), dtype=np.int64)
    for row in range(8):
        for col in range(8):
            k = int(std[row, col])
            lut[row, col] = (k // 8, k % 8)
    return lut


def standard_to_raw(grid_std: np.ndarray) -> np.ndarray:
    """Standard 8x8 rács -> nyers 8x8 rács (a raw_to_standard inverze)."""
    grid_std = np.asarray(grid_std)
    lut = _std_to_raw_lut()
    out = np.empty_like(grid_std)
    for row in range(8):
        for col in range(8):
            r, c = lut[row, col]
            out[r, c] = grid_std[row, col]
    return out


def raw_rc_to_square_name(r: int, c: int) -> str:
    """Nyers (r, c) -> mezőnév ('a1'...'h8')."""
    lut = _std_to_raw_lut()
    for row in range(8):
        for col in range(8):
            if lut[row, col, 0] == r and lut[row, col, 1] == c:
                return chess.square_name(chess.square(col, 7 - row))
    raise ValueError((r, c))


@dataclass(frozen=True)
class SquareLabel:
    raw_r: int
    raw_c: int
    square: str          # 'e4'
    occ: int             # 0 empty / 1 white / 2 black  (pipeline-kódolás)
    color_name: str      # 'empty' / 'white' / 'black'  (mappa neve)
    symbol: str          # python-chess szimbólum ('P', 'n', ...) vagy 'x' ha üres
    type_idx: int        # PIECE_TYPE_CLASSES index (0 = none)
    type_name: str       # 'none' / 'pawn' / ...


def fen_to_raw_labels(fen: str) -> list[list[SquareLabel]]:
    """
    FEN -> 8x8 lista SquareLabel-ekből a NYERS rácsban (labels[r][c] a
    bbox_warp[r][c]-hez tartozó mező címkéje). A foglaltság a
    chess_logic.resolver.board_to_occupancy-ból jön (a rendszer egyetlen
    igaz forrása), a bábutípus a python-chess-ből.
    """
    board = chess.Board(fen)
    occ_std = np.asarray(board_to_occupancy(board), dtype=np.int64)

    sym_std = np.full((8, 8), EMPTY_SYMBOL, dtype="<U1")
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            sym_std[7 - chess.square_rank(sq), chess.square_file(sq)] = piece.symbol()

    occ_raw = standard_to_raw(occ_std)
    sym_raw = standard_to_raw(sym_std)

    out: list[list[SquareLabel]] = []
    for r in range(8):
        row: list[SquareLabel] = []
        for c in range(8):
            sym = str(sym_raw[r, c])
            occ = int(occ_raw[r, c])
            t_idx = 0 if sym == EMPTY_SYMBOL else PIECE_SYMBOL_TO_TYPE_IDX[sym.lower()]
            row.append(SquareLabel(
                raw_r=r, raw_c=c, square=raw_rc_to_square_name(r, c), occ=occ,
                color_name=OCC_NAME[occ], symbol=sym, type_idx=t_idx, type_name=PIECE_TYPE_CLASSES[t_idx],
            ))
        out.append(row)
    return out


def fen_to_raw_occupancy(fen: str) -> np.ndarray:
    """Csak a foglaltság (0/1/2) nyers rácsban — a gyűjtő orientáció-ellenőrzéséhez."""
    return np.asarray([[lab.occ for lab in row] for row in fen_to_raw_labels(fen)], dtype=np.int32)
