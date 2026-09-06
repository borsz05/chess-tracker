#!/usr/bin/env python3
"""
FEN-lista generálása a tanítóadat-gyűjtéshez (tools/collect_fen_dataset.py).

MIÉRT NEM RANDOM ADATBÁZIS:
    Egy puzzle-adatbázisból vett állásokat egyesével, a nulláról kellene
    felrakni a fizikai táblán (~25 bábu/állás). Ehelyett a lista JÁTSZMÁK
    egymás utáni pozícióiból áll: két szomszédos FEN között pontosan EGY
    lépés a különbség, tehát a gyűjtés közben csak egy bábut kell mozgatni.
    Teljes újrarakás csak játszmahatáron kell (alapállás).

MIRE OPTIMALIZÁL:
    - sok bábu marad a táblán (a lépésválasztás bünteti az ütéseket),
    - mezőnkénti lefedettség: minden mező lásson black / white / empty
      címkét is a lista során,
    - kiemelten a FEKETE bábu SÖTÉT mezőn eset, mert élesben ez a hibás.

    A generátor több jelölt-játszmát készít, majd mohón azokat választja ki,
    amelyek a legtöbb ÚJ (mező, osztály) kombinációt hozzák.

Használat:
    python -m tools.make_fen_positions                       # 150 állás -> positions.txt
    python -m tools.make_fen_positions -n 200 -o sajat.txt
    python -m tools.make_fen_positions --moves-per-game 25 --seed 7

Utána:
    python -m tools.collect_fen_dataset --session delelott_ablak --fen-file positions.txt

A gyűjtőben `n` = következő FEN a listából.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess

# Sötét mező: a chess.SQUARES indexelésénél (file + rank) páros -> sötét.
DARK = {sq for sq in chess.SQUARES if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0}


def square_class(board: chess.Board, sq: int) -> str:
    piece = board.piece_at(sq)
    if piece is None:
        return "empty"
    return "white" if piece.color == chess.WHITE else "black"


def play_game(rng: random.Random, n_moves: int, capture_penalty: float) -> list[str]:
    """Egy játszma pozíciói. A lépésválasztás bünteti az ütéseket, hogy sok
    bábu maradjon a táblán (több ROI/fotó és több fekete bábu)."""
    board = chess.Board()
    # A KIINDULO allas is bekerul: enelkul a lista elso FEN-je mar egy lepes
    # utan van, a felhasznalo viszont az alapallast rakja fel -> elteres a
    # cimke es a valosag kozott (rossz cimkeju mentes).
    fens: list[str] = [board.fen()]

    for _ in range(n_moves - 1):
        moves = list(board.legal_moves)
        if not moves:
            break
        weights = [
            (capture_penalty if board.is_capture(m) else 1.0)
            for m in moves
        ]
        move = rng.choices(moves, weights=weights, k=1)[0]
        board.push(move)
        fens.append(board.fen())
        if board.is_game_over():
            break

    return fens


def rotate_fen_180(fen: str) -> str:
    """A tábla 180°-os elforgatása.

    Fizikailag: a felhasználó megfordítja a táblát, és a kamera így a fekete
    sereget látja az alsó sorokban. Enélkül a fekete bábuk SOHA nem kerülnek
    a kamera 1-3. sorába (a fekete a saját térfelén marad), és pont ez a
    lefedettségi hiány a fekete-bábu problémánál.

    A sáncjogot töröljük: elforgatás után értelmezhetetlen. Az állás nem
    feltétlenül legális sakkállás, de a gyűjtőnek csak a bábuk elhelyezkedése
    kell (FEN -> board_to_occupancy), nem generál belőle lépéseket.
    """
    board = chess.Board(fen)
    rotated = board.transform(chess.flip_vertical).transform(chess.flip_horizontal)
    rotated.set_castling_fen("-")
    rotated.ep_square = None
    return rotated.fen()


def coverage_keys(fen: str) -> set[tuple[int, str]]:
    """(mező, osztály) párok, amiket ez az állás lefed."""
    board = chess.Board(fen)
    return {(sq, square_class(board, sq)) for sq in chess.SQUARES}


def dark_black_count(fen: str) -> int:
    """Hány fekete bábu áll sötét mezőn — ez a kritikus eset."""
    board = chess.Board(fen)
    n = 0
    for sq in DARK:
        p = board.piece_at(sq)
        if p is not None and p.color == chess.BLACK:
            n += 1
    return n


def piece_count(fen: str) -> int:
    return sum(1 for sq in chess.SQUARES if chess.Board(fen).piece_at(sq) is not None)


def select_games(games: list[list[str]], n_target: int, per_game: int) -> list[list[str]]:
    """Mohó kiválasztás: mindig az a játszma jön, amelyik a legtöbb ÚJ
    (mező, osztály) kombinációt adja hozzá a már kiválasztottakhoz."""
    covered: set[tuple[int, str]] = set()
    chosen: list[list[str]] = []
    pool = list(games)

    while pool and sum(len(g) for g in chosen) < n_target:
        best_i, best_gain, best_dark = 0, -1, -1
        for i, g in enumerate(pool):
            keys: set[tuple[int, str]] = set()
            for fen in g:
                keys |= coverage_keys(fen)
            gain = len(keys - covered)
            dark = sum(dark_black_count(f) for f in g)
            if gain > best_gain or (gain == best_gain and dark > best_dark):
                best_i, best_gain, best_dark = i, gain, dark

        g = pool.pop(best_i)
        for fen in g:
            covered |= coverage_keys(fen)
        chosen.append(g[:per_game])

    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--count", type=int, default=150, help="hány FEN kerüljön a listába (alap: 150)")
    ap.add_argument("-o", "--out", type=Path, default=Path("positions.txt"))
    ap.add_argument("--moves-per-game", type=int, default=30,
                    help="hány pozíció egy játszmából (ennyi lépést kell egymás után megtenned)")
    ap.add_argument("--capture-penalty", type=float, default=0.15,
                    help="ütő lépések relatív súlya (kisebb = több bábu marad a táblán)")
    ap.add_argument("--pool", type=int, default=60, help="ennyi jelölt játszmából válogat")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pool = [
        play_game(rng, args.moves_per_game, args.capture_penalty)
        for _ in range(args.pool)
    ]
    pool = [g for g in pool if len(g) >= args.moves_per_game // 2]

    games = select_games(pool, args.count, args.moves_per_game)

    lines: list[str] = [
        "# FEN-lista a tanitoadat-gyujteshez (tools/make_fen_positions.py)",
        "# Ket egymast koveto FEN kozott EGY lepes a kulonbseg -> csak egy babut mozgass.",
        "# Uj jatszma kezdetenel allitsd vissza az ALAPALLAST.",
        "",
    ]

    total = 0
    all_covered: set[tuple[int, str]] = set()
    dark_black_total = 0
    piece_counts: list[int] = []

    for gi, g in enumerate(games, 1):
        if total >= args.count:
            break
        # Minden MASODIK jatszma 180°-kal forgatott: igy a fekete babuk a
        # kamera also soraiban is megjelennek. Valtakozva (nem a lista masodik
        # feleben), hogy egy rovidebb reszlet is lefedje mindket allast — a
        # gyakorlatban ritkan fotozza le valaki mind a 150 allast egy
        # fenyviszony mellett.
        rotate = gi % 2 == 0
        if rotate:
            g = [rotate_fen_180(f) for f in g]
            lines.append(f"# ---- {gi}. jatszma — FORDITSD MEG A TABLAT 180 FOKKAL, "
                         f"rakd fel az alapallast a megforditott tablan ----")
        else:
            lines.append(f"# ---- {gi}. jatszma — ALLITSD VISSZA AZ ALAPALLAST, majd lepesenkent kovesd ----")
        for fen in g:
            if total >= args.count:
                break
            lines.append(fen)
            all_covered |= coverage_keys(fen)
            dark_black_total += dark_black_count(fen)
            piece_counts.append(piece_count(fen))
            total += 1
        lines.append("")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- statisztika ------------------------------------------------------
    missing = [
        (chess.square_name(sq), cls)
        for sq in chess.SQUARES
        for cls in ("empty", "white", "black")
        if (sq, cls) not in all_covered
    ]

    print(f"Kiirva: {args.out.resolve()}")
    print(f"  allasok         : {total}  ({len(games)} jatszma, jatszmankent max {args.moves_per_game})")
    print(f"  babu/allas       : atlag {sum(piece_counts)/max(1,len(piece_counts)):.1f}  "
          f"min {min(piece_counts)}  max {max(piece_counts)}")
    print(f"  ROI osszesen     : {total * 64} (mezo-kivagas, ha minden allast lefotozol)")
    print(f"  fekete babu sotet mezon: osszesen {dark_black_total}  "
          f"(atlag {dark_black_total/max(1,total):.1f} allasonkent)")
    print(f"  (mezo, osztaly) lefedettseg: {len(all_covered)}/192")
    if missing:
        print(f"  HIANYZO kombinaciok ({len(missing)}): "
              + ", ".join(f"{s}:{c}" for s, c in missing[:20])
              + (" ..." if len(missing) > 20 else ""))
    else:
        print("  minden mezo latott black/white/empty cimket is.")


if __name__ == "__main__":
    main()
