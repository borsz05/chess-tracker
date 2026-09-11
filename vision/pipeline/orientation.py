"""
A 8x8 rács nyolc szimmetriája (4 forgatás x tükrözés) — tisztán numerikus segédek.

A `board_detector` rácsának orientációja NEM garantált: ugyanaz a fizikai tábla
két detektálás között más rácsállást adhat (elforgatott / tükrözött ideális
rács a homográfia-illesztésben). Az éles pipeline `raw_to_standard` leképezése
FIX, arra épül a python-chess irányítás — ezért az orientációt nem ott, hanem
a detektálás KIMENETÉN (a bbox-rács átrendezésével) igazítjuk, ha ismert
foglaltsághoz képest egyértelműen más szimmetria illeszkedik
(ChessVisionTracker: újradetektálás után, az elfogadott álláshoz képest).

Ugyanezeket a függvényeket használja a FEN-gyűjtő (tools/collect_fen_dataset)
a címkék automatikus igazítására.

Szimmetria-nevek: "rot0", "rot0+flip", "rot90", "rot90+flip", "rot180",
"rot180+flip", "rot270", "rot270+flip".  `rotK` = np.rot90(grid, K/90),
`+flip` = utána np.fliplr.  apply_symmetry(grid, name) == dict(symmetries(grid))[name].
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

SYMMETRY_NAMES: tuple[str, ...] = (
    "rot0", "rot0+flip", "rot90", "rot90+flip", "rot180", "rot180+flip", "rot270", "rot270+flip",
)
IDENTITY = "rot0"


def _as_object_array(grid) -> np.ndarray:
    a = np.empty((8, 8), dtype=object)
    for r in range(8):
        for c in range(8):
            a[r, c] = grid[r][c]
    return a


def apply_symmetry(grid, name: str):
    """Egy 8x8 rács (numpy tömb VAGY listák listája, bármilyen elemtípussal)
    átrendezése a megadott szimmetriával. numpy bemenetre numpy-t, lista
    bemenetre listák listáját ad vissza."""
    if name not in SYMMETRY_NAMES:
        raise ValueError(f"ismeretlen szimmetria: {name!r}")
    is_np = isinstance(grid, np.ndarray)
    a = grid if is_np else _as_object_array(grid)
    k = int(name.replace("+flip", "").replace("rot", "")) // 90
    out = np.rot90(a, k)
    if name.endswith("+flip"):
        out = np.fliplr(out)
    if is_np:
        return np.ascontiguousarray(out)
    return [[out[r, c] for c in range(8)] for r in range(8)]


def symmetries(grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Mind a nyolc (név, átrendezett rács) pár, SYMMETRY_NAMES sorrendben."""
    out = []
    for k in range(4):
        g = np.rot90(grid, k)
        out.append((f"rot{90 * k}", g))
        out.append((f"rot{90 * k}+flip", np.fliplr(g)))
    return out


# A nyolcelemű szimmetriacsoport inverzei. A forgatások egymás inverzei
# (rot90 <-> rot270, rot0 és rot180 önmaguké), a tükrözéssel kombinált tagok
# pedig mind ÖNINVERZEK: a "rotK+flip" kétszer alkalmazva visszaadja az
# eredetit. Korábban ezt minden híváskor nyolc próbaillesztéssel kerestük meg.
# A táblát a tests/test_orientation.py mind a nyolc névre ellenőrzi:
# apply_symmetry(apply_symmetry(g, n), inverse_symmetry(n)) == g.
_INVERSE_SYMMETRY = {
    "rot0": "rot0",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "rot0+flip": "rot0+flip",
    "rot90+flip": "rot90+flip",
    "rot180+flip": "rot180+flip",
    "rot270+flip": "rot270+flip",
}


def inverse_symmetry(name: str) -> str:
    """Az a szimmetria, amely apply_symmetry(apply_symmetry(g, name), inv) == g-t ad."""
    if name not in _INVERSE_SYMMETRY:
        raise ValueError(f"ismeretlen szimmetria: {name!r}")
    return _INVERSE_SYMMETRY[name]


def rank_symmetries(
    observed: np.ndarray,
    expected: np.ndarray,
    allowed: Sequence[str] | None = None,
) -> list[tuple[str, int]]:
    """Minden (engedélyezett) szimmetriára: hány mezőben tér el
    apply_symmetry(expected, s) az observed-tól. Növekvő eltérés szerint
    rendezve (azonos eltérésnél a SYMMETRY_NAMES sorrend, tehát rot0 elöl)."""
    observed = np.asarray(observed)
    allow = set(SYMMETRY_NAMES if allowed is None else allowed)
    scored = [(name, int(np.count_nonzero(observed != g))) for name, g in symmetries(np.asarray(expected)) if name in allow]
    return sorted(scored, key=lambda t: (t[1], SYMMETRY_NAMES.index(t[0])))


def choose_alignment(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    max_mismatch: int,
    min_margin: int,
    allowed: Sequence[str] | None = None,
) -> tuple[str, list[tuple[str, int]]]:
    """
    Melyik szimmetriával kell az `expected` rácsot átrendezni, hogy az
    `observed`-hoz illeszkedjen — BIZTONSÁGOS döntéssel:

      - ha az identitás (rot0) legfeljebb `max_mismatch` mezőben tér el, az
        identitást tartjuk (nem igazítunk feleslegesen);
      - különben a legjobb szimmetria csak akkor nyer, ha legfeljebb
        `max_mismatch` az eltérése ÉS legalább `min_margin` mezővel jobb a
        második legjobbnál (egyértelműség: szimmetrikus állásoknál — pl. az
        alapállás bal-jobb tükrözése — nem döntünk);
      - minden más esetben identitás (a hívó dönt, mit tesz a rossz illeszkedéssel).

    `allowed`: a szóba jövő szimmetriák (alapból mind a 8; a tracker csak a
    forgatásokat engedi — lásd tracker.ROTATION_SYMMETRIES). Az identitásnak
    mindig szerepelnie kell.

    Visszaad: (választott szimmetria, a rangsor).
    """
    ranking = rank_symmetries(observed, expected, allowed)
    by_name = dict(ranking)
    if by_name[IDENTITY] <= max_mismatch:
        return IDENTITY, ranking
    best_name, best_n = ranking[0]
    second_n = ranking[1][1] if len(ranking) > 1 else 64
    if best_name != IDENTITY and best_n <= max_mismatch and (second_n - best_n) >= min_margin:
        return best_name, ranking
    return IDENTITY, ranking


