from __future__ import annotations

from typing import Any, Iterable, Sequence

import chess

from .move_types import MoveGuess


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


# ---------------------------------------------------------------------------
# Bábutípus-tipp (a vision típus-fejéből) — EGYELŐRE csak promóciónál használjuk
# ---------------------------------------------------------------------------

# A típus-fej kimeneti sorrendje: (none, pawn, knight, bishop, rook, queen, king)
# — ugyanaz, mint vision/models/square_net.PIECE_TYPE_CLASSES (teszt köti össze
# a kettőt; a chess_logic szándékosan nem importál vision/torch modult).
TYPE_INDEX = {"none": 0, "pawn": 1, "knight": 2, "bishop": 3, "rook": 4, "queen": 5, "king": 6}
PROMOTION_TYPE_INDEX = {"q": TYPE_INDEX["queen"], "r": TYPE_INDEX["rook"],
                        "b": TYPE_INDEX["bishop"], "n": TYPE_INDEX["knight"]}
PIECE_TYPE_TO_TYPE_INDEX = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}
# Ennél kisebb valószínűségű típus-tippet nem hiszünk el promóciónál: marad a vezér.
# A küszöb a NÉGY LEHETSÉGES promóciós típusra újranormált valószínűségre
# vonatkozik (lásd promotion_preference).
DEFAULT_PROMOTION_MIN_CONF = 0.50

TypeProbGrid = Any   # 8x8 rács, elemenként 7 hosszú valószínűség-vektor (lista vagy numpy), standard orientáció


def _type_probs_at(type_probs: TypeProbGrid | None, row: int, col: int):
    if type_probs is None:
        return None
    try:
        v = type_probs[row][col]
    except (IndexError, KeyError, TypeError):
        return None
    if v is None:
        return None
    v = [float(x) for x in v]
    return v if len(v) == len(TYPE_INDEX) else None


def promotion_preference(
    type_probs: TypeProbGrid | None,
    to_row: int,
    to_col: int,
    *,
    min_conf: float = DEFAULT_PROMOTION_MIN_CONF,
) -> list[str]:
    """
    A promóciós bábu sorrendje ('q','r','b','n' permutációja) a típus-fej
    célmezőn mért kimenete alapján. A visszaadott sorrend első eleme nyer, mert
    a négy promóció foglaltság szerint megkülönböztethetetlen.

    GYALOG ÉS KIRÁLY KIZÁRVA: promóciónál ez a két típus lehetetlen (a gyalog
    definíció szerint átváltozik, királlyá pedig nem lehet), ezért a rájuk eső
    valószínűséget ELDOBJUK, és a maradék négyet ÚJRANORMÁLJUK. Így ha a modell
    elsőként gyalogot vagy királyt mond, automatikusan a következő legbiztosabb
    (már lehetséges) válasz dönt, nem esünk vissza az alapértelmezett vezérre.

    A min_conf küszöb az újranormált értékre vonatkozik: ha a legjobb lehetséges
    típus így sem éri el, marad az alapsorrend (vezér elöl) — ez a korábbi
    viselkedés arra az esetre, amikor a négy jelölt között sincs érdemi
    különbség (pl. mind ~0.25).
    """
    default = ["q", "r", "b", "n"]
    v = _type_probs_at(type_probs, to_row, to_col)
    if v is None:
        return default

    # Csak a négy lehetséges promóciós típus marad; a gyalog/király/üres tömeget
    # eldobjuk, es a maradekot ujranormaljuk.
    total = sum(v[PROMOTION_TYPE_INDEX[ch]] for ch in default)
    if total <= 0.0:
        return default
    norm = {ch: v[PROMOTION_TYPE_INDEX[ch]] / total for ch in default}

    scored = sorted(default, key=lambda ch: (-norm[ch], default.index(ch)))
    if norm[scored[0]] < min_conf:
        return default
    return scored


def _promotion_priority(move: chess.Move) -> int:
    if move.promotion is None:
        return 0
    if move.promotion == chess.QUEEN:
        return 1
    return 2


def _ordered_legal_moves(
    ch_board: chess.Board,
    type_probs: TypeProbGrid | None = None,
    promotion_min_conf: float = DEFAULT_PROMOTION_MIN_CONF,
) -> list[chess.Move]:
    """Legális lépések: előbb a nem-promóciók, aztán a promóciók a típus-tipp
    (vagy alapból a vezér) szerinti sorrendben. Promóció nélkül a sorrend a
    régi (_promotion_priority)."""
    moves = list(ch_board.legal_moves)
    if type_probs is None or not any(m.promotion for m in moves):
        return sorted(moves, key=_promotion_priority)

    def key(m: chess.Move):
        if m.promotion is None:
            return (0, 0)
        to_row, to_col = chess_square_to_coords(m.to_square)
        pref = promotion_preference(type_probs, to_row, to_col, min_conf=promotion_min_conf)
        return (1, pref.index(_promotion_char(m.promotion)))

    return sorted(moves, key=key)


# ---------------------------------------------------------------------------
# Inkrementális foglaltság: a lépés utáni rács a lépés előttiből, másolás nélkül
# ---------------------------------------------------------------------------

def expected_occupancy_after_move(
    ch_board: chess.Board,
    move: chess.Move,
    occ_before: Sequence[Sequence[int]],
) -> list[list[int]]:
    """
    A `move` utáni foglaltság az `occ_before`-ból (ami ch_board foglaltsága).
    Ugyanazt adja, mint board.copy() + push() + board_to_occupancy(), csak
    ~100x olcsóbban: egy lépés legfeljebb 4 mezőt érint (honnan, hova, en
    passant-nál a leütött gyalog, sáncnál a bástya). A szabálylogika marad a
    python-chess-é (legal_moves, is_en_passant, is_castling) — ez a függvény
    csak a foglaltság-diffet írja fel. tests/test_resolver_incremental.py a
    referencia-úttal veti össze minden lépésre.
    """
    occ = [list(row) for row in occ_before]
    color = 1 if ch_board.turn == chess.WHITE else 2

    fr, fc = chess_square_to_coords(move.from_square)
    tr, tc = chess_square_to_coords(move.to_square)
    occ[fr][fc] = 0
    occ[tr][tc] = color

    if ch_board.is_en_passant(move):
        # a leütött gyalog a célmező oszlopában, a kiinduló mező sorában áll
        occ[fr][tc] = 0
    elif ch_board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)   # h -> f
        else:
            rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)   # a -> d
        rf_r, rf_c = chess_square_to_coords(rook_from)
        rt_r, rt_c = chess_square_to_coords(rook_to)
        occ[rf_r][rf_c] = 0
        occ[rt_r][rt_c] = color

    return occ


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


def _moving_type_index(ch_board: chess.Board, move: chess.Move) -> int:
    """A célmezőre kerülő bábu típus-indexe (promóciónál a promotált bábué)."""
    if move.promotion is not None:
        return PIECE_TYPE_TO_TYPE_INDEX[move.promotion]
    piece = ch_board.piece_at(move.from_square)
    return PIECE_TYPE_TO_TYPE_INDEX[piece.piece_type] if piece is not None else 0


def resolve_move_from_occupancy(
    current_board,
    observed_occ: list[list[int]],
    observed_conf: list[list[float]] | None = None,
    *,
    max_noise_cells: int = 2,
    max_weighted_cost: float = 1.2,
    min_changed_cells: int = 2,
    type_probs: TypeProbGrid | None = None,
    promotion_min_conf: float = DEFAULT_PROMOTION_MIN_CONF,
    use_type_hint_for_moves: bool = False,
) -> tuple[MoveGuess | None, list[list[int]] | None, str | None]:
    """
    Exact, majd opcionálisan fuzzy legal move feloldás.
    A sakklogika teljes egészében itt marad, nem a vision trackerben.

    min_changed_cells: a megfigyelt rács legalább ennyi mezőben térjen el a
        jelenlegi állástól — minden legális lépés >= 2 mezőt változtat, az
        1 mezős eltérés zaj, arra a fuzzy ág SEM tippelhet lépést (ez volt a
        korábbi hamis-elfogadási út: "eltűnt" egy bábu, a resolver pedig a
        legkisebb konfidenciájú célmezőt választotta hozzá).
    type_probs: a vision típus-fejének 8x8x7 kimenete (standard orientáció).
        EGYELŐRE csak a promóciós bábu kiválasztásához használjuk.
    use_type_hint_for_moves: kapcsoló a jövőbeli kiterjesztéshez — True esetén
        az AZONOS költségű fuzzy jelöltek között a típus-fej dönt (a célmezőre
        kerülő bábu típusának valószínűsége). Alapból False: a 3-osztályos út
        viselkedése változatlan.
    """
    ch_board = _as_chess_board(current_board)
    occ_before = board_to_occupancy(ch_board)

    if occupancy_distance(observed_occ, occ_before) < min_changed_cells:
        return None, None, None

    ordered = _ordered_legal_moves(ch_board, type_probs, promotion_min_conf)

    for mv in ordered:
        expected_occ = expected_occupancy_after_move(ch_board, mv, occ_before)
        if same_occupancy(expected_occ, observed_occ):
            return chess_move_to_moveguess(ch_board, mv), expected_occ, "exact"

    if observed_conf is None:
        return None, None, None

    best = None

    for mv in ordered:
        expected_occ = expected_occupancy_after_move(ch_board, mv, occ_before)

        noise = occupancy_distance(expected_occ, observed_occ)
        if noise > max_noise_cells:
            continue

        # Geometriai feltétel: a célmezőt FOGLALTNAK kell látni. A zaj csak
        # szín-tévesztés lehet (vagy sánc/en passant mellékmezője), nem az,
        # hogy "a bábu még nem érkezett meg" — az utóbbi félkész lépés
        # (leütött bábu levéve, a lépő még kézben), arra nem tippelünk.
        to_row, to_col = chess_square_to_coords(mv.to_square)
        if int(observed_occ[to_row][to_col]) == 0:
            continue

        cost = weighted_diff(observed_occ, expected_occ, observed_conf)

        # rangsor: költség, zaj; opcionálisan (kapcsolóval) a típus-fej egyezése
        type_score = 0.0
        if use_type_hint_for_moves and type_probs is not None:
            v = _type_probs_at(type_probs, to_row, to_col)
            if v is not None:
                type_score = -v[_moving_type_index(ch_board, mv)]
        key = (cost, noise, type_score)

        if best is None or key < best[0]:
            best = (key, mv, expected_occ)

    if best is None:
        return None, None, None

    (cost, noise, _), mv, expected_occ = best
    if cost > max_weighted_cost:
        return None, None, None

    return (
        chess_move_to_moveguess(ch_board, mv),
        expected_occ,
        f"fuzzy c={cost:.2f} n={noise}",
    )


def _changed_cells_of(occ_before: Sequence[Sequence[int]], occ_after: Sequence[Sequence[int]]) -> dict[tuple[int, int], int]:
    return {(r, c): int(occ_after[r][c]) for r in range(8) for c in range(8) if int(occ_before[r][c]) != int(occ_after[r][c])}


def prefix_ambiguities(current_board, move: chess.Move | str) -> list[str]:
    """
    Azok a legális lépések (UCI), amelyeknek a `move` foglaltság-változása
    VALÓDI RÉSZHALMAZA — vagyis a `move` megfigyelt állapota egy másik, még
    folyamatban lévő lépés köztes állapota is lehet. A gyakorlatban: a
    bástyával KEZDETT sánc (h1->f1 megfigyelve, a király még e1-en) pontosan a
    Rf1 lépésnek látszik, miközben O-O készül. A hívó (tracker) ilyenkor
    hosszabb megerősítési időt vár, mielőtt elfogadja a rövidebb lépést.
    Nem-sánc állásokban üres lista (a többi lépéstípusnak nincs legális
    "előtagja" a foglaltság szintjén).
    """
    ch_board = _as_chess_board(current_board)
    mv = chess.Move.from_uci(move) if isinstance(move, str) else move
    occ0 = board_to_occupancy(ch_board)
    mine = _changed_cells_of(occ0, expected_occupancy_after_move(ch_board, mv, occ0))
    out: list[str] = []
    for other in ch_board.legal_moves:
        if other == mv:
            continue
        theirs = _changed_cells_of(occ0, expected_occupancy_after_move(ch_board, other, occ0))
        if len(theirs) > len(mine) and all(theirs.get(cell) == val for cell, val in mine.items()):
            out.append(other.uci())
    return out
