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

MÁSODIK MÓD — TISZT-DRILL a bábutípus-fejhez (Stockfish nélkül):
    python -m tools.make_fen_positions --mode tiszt --shots 16 -o positions_tiszt.txt
A játszma-alapú állásokban a király/vezér mindig ugyanazon a pár mezőn áll
(mérve: king 9/64, queen 17/64 különböző mező), ezért a típus-fej a MEZŐT
tanulja meg a bábu helyett. A tiszt-drill csúszó kétsoros mintával viszi végig
a 16 tisztet a tábla minden során és oszlopán, világos és sötét mezőn is.

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


# ---------------------------------------------------------------------------
# TISZT-DRILL mod (--mode tiszt): a BABUTIPUS-FEJ celzott adata, Stockfish nelkul
# ---------------------------------------------------------------------------
#
# Miert kell: a jatszma-alapu allasokban a kiraly es a vezer szinte mindig
# ugyanazon a par mezon all (a mert adatban: king 9/64, queen 17/64, rook 17/64,
# bishop 20/64, knight 19/64 kulonbozo mezo), a gyalog viszont 45/64-en. A
# tipus-fej igy a MEZOT tanulja meg a babu helyett. Ez a mod nem jatszmakbol
# mintavetelez, hanem SZANDEKOSAN vegigviszi a 16 tisztet a tabla minden soran
# es oszlopan.
#
# Fizikai elrendezes (ezert ilyen egyszeru a minta):
#   - ket sor: 8 feher tiszt az egyik soron, 8 fekete tiszt egy masikon,
#     a ket sor mindig 4 sorra van egymastol (nem takarjak egymast a kameranak)
#   - fotonkent MINDKET sor eggyel feljebb csuszik, es a babuk sorrendje
#     ciklikusan eltolodik -> minden tiszt vegigjarja mind a 8 oszlopot es
#     mind a 8 sort, vilagos es sotet mezon egyarant
#   - a felrakas egy sorra ~8 babu athelyezese, a gyujto diagramja mutatja
#
INVENTORY = "KQRRBBNN"   # egy szinbol fizikailag ennyi tiszt van a keszletben


def officer_positions(shots: int, order_shift: int = 3, row_shift: int = 1,
                      black_gap: int = 4, mirror_black: bool = True) -> list[chess.Board]:
    """`shots` darab tiszt-drill allas, csuszo ket-soros mintaval.

    A ciklikus eltolasok (order_shift, row_shift) egymashoz kepest primek, igy
    egy tiszt nem ragad ugyanabba az oszlopba: 8 foto alatt 8 kulonbozo
    oszlopot es 8 kulonbozo sort jar be.
    """
    out: list[chess.Board] = []
    n = len(INVENTORY)
    for i in range(shots):
        b = chess.Board(None)                      # ures tabla, nincs babu
        w_rank = (i * row_shift) % 8
        b_rank = (w_rank + black_gap) % 8
        # A sor- es a sorrend-eltolasnak is paratlannak kell lennie a teljes
        # sor-/oszlop-lefedettseghez, ez viszont azt jelenti, hogy egy adott
        # tiszt mezoszine (f + rank paritasa) minden fotoban ugyanaz maradna
        # (a kiraly vegig sotet, a vezer vegig vilagos mezon). A 8 fotonkenti
        # +1 oszlop-offset toriti ezt: a masodik korben minden tiszt az
        # ELLENTETES szinu mezokre kerul.
        lap = i // 8
        for f in range(8):
            w_sym = INVENTORY[(f + i * order_shift + lap) % n]
            # a fekete sor ellentetes iranyban forog, kulonben a K mindig a K
            # ala kerulne (es ugyanazok a tipus-parok allnanak egymas mellett)
            b_idx = (-(f + i * order_shift + lap) if mirror_black else (f + i * order_shift + lap + 4)) % n
            b.set_piece_at(chess.square(f, w_rank), chess.Piece.from_symbol(w_sym))
            b.set_piece_at(chess.square(f, b_rank), chess.Piece.from_symbol(INVENTORY[b_idx].lower()))
        out.append(b)
    return out


def type_coverage(boards: list[chess.Board]) -> dict[str, set[int]]:
    """tipusnev -> mely mezokon lattuk (a szin nem szamit, a tipus-fej szinvak)."""
    cov: dict[str, set[int]] = {}
    for b in boards:
        for sq, piece in b.piece_map().items():
            cov.setdefault(chess.piece_name(piece.piece_type), set()).add(sq)
    return cov


def write_officer_list(boards: list[chess.Board], out: Path) -> None:
    lines = [
        "# TISZT-DRILL FEN-lista (tools/make_fen_positions.py --mode tiszt)",
        "# Cel: a babutipus-fej. Minden tiszt sok KULONBOZO mezon (vilagoson es",
        "# soteten is) szerepeljen, ne a mezot tanulja meg a modell a tipus helyett.",
        "#",
        "# FELRAKAS: ket sor, soronkent 8 tiszt; a gyujto diagramja (jobb oldali",
        "# ablak) mutatja pontosan. Fotonkent mindket sor eggyel feljebb csuszik es",
        "# a babuk sorrendje eltolodik -> ~8-16 babut kell athelyezni allasonkent.",
        "#",
        "# FORGATAS: minden allas felrakasa utan forgasd el a babukat a sajat",
        "# tengelyuk korul (kulonosen a HUSZART: nezzen elore/hatra/oldalra), es",
        "# ket-harom allasonkent a TABLAT is forditsd meg 180 fokkal.",
        "",
    ]
    lines += [b.fen() for b in boards]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")



# ---------------------------------------------------------------------------
# ARNYEK-DRILL mod (--mode arnyek): URES mezok, amikre BABU vet arnyekot
# ---------------------------------------------------------------------------
#
# A hiba, amire gyujtunk: oldalrol jovo fenynel egy nagy arnyek az ures mezon
# "feher babunak" latszik. Az arnyekot a szomszedos BABUK vetik, tehat ures
# tablaval nem lehet gyujteni ra. Ezert sakktabla-mintaban allunk fel: minden
# masodik mezon all babu, es a KOZTUK levo ures mezok mindegyikenek 4 foglalt
# ortogonalis szomszedja van -> barmelyik iranybol is jon a feny, esik ra arnyek.
#
# A gyujtot ehhez `--only-classes empty`-vel kell inditani: a babuk mezoi nem
# kerulnek a datasetbe (nem akarjuk a babu-osztalyokat sakktabla-mintaval
# hizlalni), csak az arnyekos ures mezok.
#
# Egy mintazat felrakasa 32 babu -> lassu, ezert KEVES allas van a listaban, es
# egy allason belul sok fotot kell csinalni: a BABUK maradnak, a LAMPAT mozgatod.
#
# 1-4. allas: 32 babu az egyik szinu mezokon (a 32 ures mezo a masik szinu) —
#             igy vilagos ES sotet ures mezorol is lesz arnyekos kepunk.
# 5-6. allas: ritkabb, 4x4-es racs 16 tiszttel: kevesebb arnyekforras, viszont
#             48 ures mezo fotonkent, es hosszu, tobb mezon atnyulo arnyekok.

FULL_INVENTORY = "KQRRBBNNPPPPPPPP"      # egy szin 16 babuja (a valodi keszlet)
OFFICERS = "KQRRBBNN"


def _fill(squares: list[int], symbols: str, white: bool) -> list[tuple[int, chess.Piece]]:
    return [(sq, chess.Piece.from_symbol(sym if white else sym.lower()))
            for sq, sym in zip(squares, symbols)]


def _checker_board(parity: int, offset: int, white_low: bool) -> chess.Board:
    """32 babu az adott szinu (parity) mezokon; a masik 32 mezo URES.

    `offset`: a keszlet ciklikus eltolasa -> mas babu all mas mezon, tehat mas
    magassagu/formaju arnyek esik ugyanarra az ures mezore.
    `white_low`: a feher babuk az also 4 soron (True) vagy a felsokon (False).
    """
    sqs = sorted(sq for sq in chess.SQUARES
                 if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == parity)
    low, high = sqs[:16], sqs[16:]
    inv = FULL_INVENTORY[offset:] + FULL_INVENTORY[:offset]
    b = chess.Board(None)
    for sq, piece in _fill(low, inv, white=white_low) + _fill(high, inv[::-1], white=not white_low):
        b.set_piece_at(sq, piece)
    return b


def _lattice_board(file_off: int, rank_off: int) -> chess.Board:
    """16 TISZT egy 4x4-es racson (minden masodik vonal es sor) — 48 ures mezo."""
    sqs = [chess.square(f, r) for r in range(rank_off, 8, 2) for f in range(file_off, 8, 2)]
    # A 16 mezore PONTOSAN a keszlet 8+8 tisztje kerul (nem ismetlodhet a
    # kiraly/vezer, mert fizikailag egy van beloluk): also fel feher, felso fekete.
    b = chess.Board(None)
    for sq, piece in _fill(sqs[:8], OFFICERS, white=True) + _fill(sqs[8:], OFFICERS[::-1], white=False):
        b.set_piece_at(sq, piece)
    return b


def shadow_positions() -> list[tuple[chess.Board, str]]:
    """(allas, magyarazat) parok az arnyek-gyujteshez."""
    return [
        (_checker_board(0, 0, True),  "32 babu a SOTET mezokon (feher lent) -> a 32 VILAGOS mezo ures"),
        (_checker_board(0, 5, False), "ugyanaz, mas babukiosztas es forditott szinek -> mas arnyekformak"),
        (_checker_board(1, 0, True),  "32 babu a VILAGOS mezokon -> a 32 SOTET mezo ures"),
        (_checker_board(1, 5, False), "ugyanaz, mas babukiosztas es forditott szinek"),
        (_lattice_board(0, 0),        "16 tiszt 4x4-es racson (a1-rol indulva) -> 48 ures mezo, hosszu arnyekok"),
        (_lattice_board(1, 1),        "ugyanaz egy mezovel eltolva (b2-rol indulva)"),
    ]


def write_shadow_list(items: list[tuple[chess.Board, str]], out: Path) -> None:
    lines = [
        "# ARNYEK-DRILL FEN-lista (tools/make_fen_positions.py --mode arnyek)",
        "# Cel: URES mezok, amikre a szomszedos BABUK vetnek arnyekot.",
        "#",
        "# A GYUJTOT IGY INDITSD (csak az ures mezok kepei kellenek):",
        "#   python -m tools.collect_fen_dataset --session <nev>",
        "#       --fen-file positions_arnyek.txt --only-classes empty --max-mismatch 30",
        "#",
        "# MUNKAMENET: a babuk EGY allason belul NEM mozdulnak — a LAMPAT mozgasd,",
        "# es allasonkent 6-10 fotot csinalj kulonbozo fenyiranybol/magassagbol:",
        "#   - alacsony, oldalrol jovo feny (leghosszabb arnyek)",
        "#   - a feny 4 fo iranybol; kozte olyan allas is, ahol az arnyek ELE epp",
        "#     atvag egy ures mezot (ez a legnehezebb eset)",
        "#   - 1-2 foto a robotkar / a kezed arnyekaval (a kar a kepen kivul legyen)",
        "# Csak ezutan lepj a kovetkezo allasra (n) es rakd at a babukat.",
        "",
    ]
    for b, why in items:
        lines.append(f"{b.fen()}   # {why}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("jatszma", "tiszt", "arnyek"), default="jatszma",
                    help="jatszma: Stockfish-kozepjatek (alap, foglaltsag-fej); "
                         "tiszt: tiszt-drill a babutipus-fejhez; "
                         "arnyek: sakktabla-mintazat, hogy a babuk arnyekot vessenek az URES mezokre "
                         "(a gyujtot --only-classes empty-vel inditsd). Az utobbi ketto Stockfish nelkul fut.")
    ap.add_argument("--shots", type=int, default=16, help="[--mode tiszt] ennyi allast general")
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

    if args.mode == "arnyek":
        items = shadow_positions()
        write_shadow_list(items, args.out)
        n_empty = [sum(1 for sq in chess.SQUARES if b.piece_at(sq) is None) for b, _ in items]
        print(f"Kiirva: {args.out.resolve()}")
        print(f"  allasok        : {len(items)}")
        print(f"  ures mezo/allas: {n_empty}  (= ennyi ROI mentodik fotonkent --only-classes empty mellett)")
        print(f"  8 foto/allas eseten: {8 * sum(n_empty)} arnyekos ures ROI")
        for b, why in items:
            print(f"    - {why}")
        return

    if args.mode == "tiszt":
        boards = officer_positions(args.shots)
        write_officer_list(boards, args.out)
        cov = type_coverage(boards)
        print(f"Kiirva: {args.out.resolve()}")
        print(f"  allasok       : {len(boards)}  (allasonkent 16 tiszt + 48 ures mezo)")
        print(f"  ROI osszesen  : {len(boards) * 64}  ebbol tipus-cimkezett: {len(boards) * 16}")
        print("  tipus-lefedettseg (hany kulonbozo mezon):")
        for name in ("king", "queen", "rook", "bishop", "knight"):
            sqs = cov.get(name, set())
            dark = sum(1 for sq in sqs if sq in DARK)
            print(f"    {name:7s} {len(sqs):2d}/64   (sotet mezon {dark}, vilagoson {len(sqs) - dark})")
        return

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
