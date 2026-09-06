#!/usr/bin/env python3
"""
FEN-lista generálása a tanítóadat-gyűjtéshez (tools/collect_fen_dataset.py).

FORRÁS: Stockfish játszik önmaga ellen (véletlenítve a top-N lépés közül).
Ez valódi, kifejlődött KÖZÉPJÁTÉK-állásokat ad — a tisztek nem a hátsó sorban
ragadnak, a király sáncolt, a gyalogszerkezet változatos. A korábbi,
véletlen-lépéses generátor pont ezért volt rossz: 25 félépés alatt a tisztek
alig mozdultak ki.

FELÉPÍTÉS (egy "adag"):
    1. rész: --per-side állás EGY játszma középjátékából, normál tábla-állásban
    2. rész: --per-side állás egy MÁSIK játszmából, 180 fokkal forgatva

    Egy részen belül a szomszédos állások között EGY lépés a különbség, tehát
    csak a legelső állást kell a nulláról felraknod (ehhez rajzol a gyűjtő
    tábla-diagramot), utána mezőnként egy bábut mozgatsz.

    A 180 fokos forgatás azért kell, mert enélkül a fekete bábuk soha nem
    kerülnek a kamera alsó soraiba — élesben viszont oda is kerülnek.

MIRE OPTIMALIZÁL:
    - sok bábu a táblán (--min-pieces), hogy egy fotó sok ROI-t adjon
    - nagy variancia: a kiinduló mezőjükről elmozdult bábuk száma maximális
    - mezőnkénti lefedettség (minden mező lásson black/white/empty címkét is)

Használat:
    python -m tools.make_fen_positions                     # 15+15 -> positions.txt
    python -m tools.make_fen_positions --per-side 20 --min-pieces 28
    python -m tools.make_fen_positions --seed 7 -o adag2.txt

Utána:
    python -m tools.collect_fen_dataset --session <fenyviszony> --fen-file positions.txt
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import chess
import chess.engine

DARK = {sq for sq in chess.SQUARES if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0}
START = chess.Board()


def square_class(board: chess.Board, sq: int) -> str:
    p = board.piece_at(sq)
    return "empty" if p is None else ("white" if p.color == chess.WHITE else "black")


def coverage_keys(board: chess.Board) -> set[tuple[int, str]]:
    return {(sq, square_class(board, sq)) for sq in chess.SQUARES}


def n_pieces(board: chess.Board) -> int:
    return len(board.piece_map())


def n_developed(board: chess.Board) -> int:
    """Hány bábu áll MÁS mezőn, mint az alapállásban — ez a 'variancia' mérőszáma."""
    n = 0
    for sq, piece in board.piece_map().items():
        if START.piece_at(sq) != piece:
            n += 1
    return n


def dark_black(board: chess.Board) -> int:
    return sum(1 for sq in DARK if (p := board.piece_at(sq)) is not None and p.color == chess.BLACK)


def load_covered(labels_csv: Path) -> set[tuple[int, str]]:
    """A mar osszegyujtott (mezo, osztaly) kombinaciok a gyujto labels.csv-jebol."""
    import csv as _csv

    covered: set[tuple[int, str]] = set()
    if not labels_csv.exists():
        print(f"  (nincs meg ilyen fajl: {labels_csv} — a teljes lefedettsegre optimalizalok)")
        return covered
    with open(labels_csv, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            sq, color = row.get("square"), row.get("color")
            if sq and color:
                try:
                    covered.add((chess.parse_square(sq), color))
                except ValueError:
                    continue
    return covered


def rotate_180(board: chess.Board) -> chess.Board:
    """A tábla 180 fokos elforgatása (a felhasználó fizikailag megfordítja).

    A sáncjog és az en passant értelmezhetetlen utána, ezért töröljük. Az állás
    nem feltétlenül legális sakkállás, de a gyűjtőnek csak a bábuk elhelyezése
    kell (FEN -> board_to_occupancy), lépéseket nem generál belőle.
    """
    r = board.transform(chess.flip_vertical).transform(chess.flip_horizontal)
    r.set_castling_fen("-")
    r.ep_square = None
    return r


def play_game(engine: chess.engine.SimpleEngine, rng: random.Random,
              plies: int, depth: int, multipv: int) -> list[chess.Board]:
    """Stockfish önmaga ellen, a top-N lépés közül véletlenszerűen választva.

    A véletlenítés adja a varianciát: enélkül minden játszma ugyanaz lenne.
    """
    board = chess.Board()
    out: list[chess.Board] = []
    limit = chess.engine.Limit(depth=depth)

    for _ in range(plies):
        if board.is_game_over():
            break
        try:
            info = engine.analyse(board, limit, multipv=multipv)
        except chess.engine.EngineError:
            break
        moves = [i["pv"][0] for i in info if i.get("pv")]
        if not moves:
            break
        board.push(rng.choice(moves))
        out.append(board.copy())

    return out


def best_window(positions: list[chess.Board], length: int, min_pieces: int) -> list[chess.Board] | None:
    """A legjobb `length` hosszú, EGYMÁST KÖVETŐ szakasz a játszmából.

    Egymást követő állásokat választunk, mert így két fotó között csak egy
    bábut kell mozgatni. A szakasz pontszáma: a kifejlődöttség (variancia)
    és a fekete-sötét mezőn arány, azzal a feltétellel, hogy végig legyen
    elég bábu a táblán.
    """
    best, best_score = None, -1.0
    for i in range(len(positions) - length + 1):
        win = positions[i:i + length]
        if min(n_pieces(b) for b in win) < min_pieces:
            continue
        score = sum(n_developed(b) for b in win) + 0.5 * sum(dark_black(b) for b in win)
        if score > best_score:
            best, best_score = win, score
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-side", type=int, default=15, help="állás orientációnként (alap: 15 + 15 forgatva)")
    ap.add_argument("--min-pieces", type=int, default=26, help="ennyi bábu legyen legalább a táblán végig")
    ap.add_argument("-o", "--out", type=Path, default=Path("positions.txt"))
    ap.add_argument("--engine", default=shutil.which("stockfish") or "/usr/games/stockfish")
    ap.add_argument("--depth", type=int, default=8, help="Stockfish keresési mélység (kicsi is elég ide)")
    ap.add_argument("--multipv", type=int, default=4, help="ennyi legjobb lépés közül választ véletlenül (variancia)")
    ap.add_argument("--plies", type=int, default=44, help="ennyi félépést játszik le egy játszmában")
    ap.add_argument("--games", type=int, default=6, help="ennyi jelölt játszmát generál, a legjobb kettőt használja")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cover-gaps", type=Path, default=None,
                    help="a mar osszegyujtott adat labels.csv-je: az uj adag a MEG HIANYZO "
                         "(mezo, osztaly) kombinaciokra optimalizal (pl. data_fen/labels.csv)")
    args = ap.parse_args()

    if not Path(args.engine).exists():
        raise SystemExit(f"Stockfish nem talalhato: {args.engine}\n  telepites: sudo apt install stockfish")

    rng = random.Random(args.seed)
    print(f"Stockfish: {args.engine}  (depth={args.depth}, multipv={args.multipv})")

    games: list[list[chess.Board]] = []
    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        for gi in range(args.games):
            g = play_game(engine, rng, args.plies, args.depth, args.multipv)
            win = best_window(g, args.per_side, args.min_pieces)
            if win:
                games.append(win)
                print(f"  {gi + 1}. jatszma: {len(g)} allas -> szakasz kivalasztva "
                      f"(babu {min(n_pieces(b) for b in win)}-{max(n_pieces(b) for b in win)}, "
                      f"elmozdult atlag {sum(n_developed(b) for b in win) / len(win):.1f})")
            else:
                print(f"  {gi + 1}. jatszma: nincs megfelelo szakasz (tul keves babu maradt)")

    if len(games) < 2:
        raise SystemExit("Nem sikerult ket hasznalhato szakaszt generalni — probald kisebb --min-pieces ertekkel.")

    # A ket szakasz kivalasztasa PARBAN, a kozos lefedettsegre optimalizalva.
    # Kulon-kulon a legfejlettebb kettot valasztani rossz: ugyanazokat a
    # mezoket fedhetik le. Mivel ugyanezt a listat fotozod le minden
    # fenyviszonyban, a lefedettseg egyszer dol el — itt.
    cov_cache = [set().union(*(coverage_keys(b) for b in w)) for w in games]
    rot_cache = [[rotate_180(b) for b in w] for w in games]
    rot_cov = [set().union(*(coverage_keys(b) for b in w)) for w in rot_cache]

    already = load_covered(args.cover_gaps) if args.cover_gaps else set()
    if args.cover_gaps:
        print(f"\n  mar osszegyujtve ({args.cover_gaps}): {len(already)}/192 mezo-osztaly "
              f"-> az uj adag a hianyzo {192 - len(already)}-re optimalizal")

    best_pair, best_cov = (0, 1), -1
    for i in range(len(games)):
        for j in range(len(games)):
            if i == j:
                continue
            # Csak az UJ lefedettseg szamit: amit mar begyujtottel, azt nem
            # eri meg megegyszer lefotozni.
            score = len((cov_cache[i] | rot_cov[j]) - already)
            if score > best_cov:
                best_pair, best_cov = (i, j), score

    i, j = best_pair
    normal, rotated = games[i], rot_cache[j]
    print(f"\n  kivalasztott par: {i + 1}. (normal) + {j + 1}. (forgatva) — egyutt {best_cov}/192 mezo-osztaly")

    lines = [
        "# FEN-lista a tanitoadat-gyujteshez (tools/make_fen_positions.py)",
        "# Forras: Stockfish onmaga ellen, kozepjatek-allasok (kifejlodott tisztek).",
        "# Egy reszen belul ket szomszedos allas kozott EGY lepes a kulonbseg.",
        "# A gyujto minden allashoz kirajzolja a tablat (o = diagram ki/be).",
        "",
        f"# ===== 1. resz ({len(normal)} allas) — NORMAL tabla-allas =====",
        "# Az ELSO allast a nullarol rakd fel a diagram alapjan, utana lepesenkent.",
    ]
    lines += [b.fen() for b in normal]
    lines += [
        "",
        f"# ===== 2. resz ({len(rotated)} allas) — FORDITSD MEG A TABLAT 180 FOKKAL =====",
        "# Ismet a nullarol rakd fel az elso allast a diagram alapjan.",
    ]
    lines += [b.fen() for b in rotated]

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- statisztika ------------------------------------------------------
    allb = normal + rotated
    cov: set[tuple[int, str]] = set()
    for b in allb:
        cov |= coverage_keys(b)
    missing = [(chess.square_name(sq), c) for sq in chess.SQUARES
               for c in ("empty", "white", "black") if (sq, c) not in cov]
    pc = [n_pieces(b) for b in allb]
    dev = [n_developed(b) for b in allb]

    print()
    print(f"Kiirva: {args.out.resolve()}")
    print(f"  allasok            : {len(allb)}  ({len(normal)} normal + {len(rotated)} forgatott)")
    print(f"  babu/allas         : atlag {sum(pc) / len(pc):.1f}  (min {min(pc)}, max {max(pc)})")
    print(f"  elmozdult babu/allas: atlag {sum(dev) / len(dev):.1f}  (min {min(dev)}, max {max(dev)})"
          f"   <- variancia")
    print(f"  fekete babu sotet mezon: atlag {sum(dark_black(b) for b in allb) / len(allb):.1f}")
    print(f"  ROI osszesen       : {len(allb) * 64}")
    print(f"  (mezo, osztaly) lefedettseg: {len(cov)}/192")
    if missing:
        print(f"  hianyzo ({len(missing)}): " + ", ".join(f"{s}:{c}" for s, c in missing[:16])
              + (" ..." if len(missing) > 16 else ""))


if __name__ == "__main__":
    main()
