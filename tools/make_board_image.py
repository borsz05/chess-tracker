"""Sakktábla-háttérkép generálása a frontendhez.

A tábla mezőit a böngészőben nem a CSS festi, hanem egy háttérkép adja
(frontend/style.css, .board-b72b1). Ez a szkript állítja elő azt a képet.

Miért generáljuk, és nem letöltjük? Mert nincs mit letölteni: egy sakktábla
két tömör színből álló, szabályos minta — nulla textúra, nulla átmenet.
A korábban használt letöltött PNG és az itt generált kép PIXELRE AZONOS volt
(2 560 000 pixelből nulla eltérés), tehát a kép semmilyen alkotást nem
hordoz, viszont generálva paraméterezhető és egyértelműen a projekt saját
eszköze.

A méret azért osztható nyolccal, mert a CSS `background-size: 100% 100%`-kal
feszíti a táblára, és a mezőhatárok csak így esnek pixelre pontosan a rácsra.

Használat:
    python -m tools.make_board_image
    python -m tools.make_board_image --light EDD6B0 --dark B88762 --square 200
    python -m tools.make_board_image --out frontend/.../boards/green.png \
        --light EBECD0 --dark 739552
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# A jelenlegi barna tábla színei. Ugyanez a két érték szerepel a
# style.css --board-light / --board-dark változóiban is: azok a koordináta-
# feliratoknak kellenek, mert tudniuk kell, milyen színű mezőre írnak.
DEFAULT_LIGHT = "EDD6B0"
DEFAULT_DARK = "B88762"
DEFAULT_SQUARE = 200

DEFAULT_OUT = Path("frontend/assets/chessboardjs-1.0.0/img/boards/brown.png")


def parse_hex(value: str) -> tuple[int, int, int]:
    """'EDD6B0' vagy '#EDD6B0' -> (237, 214, 176)."""
    text = value.lstrip("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError(f"Hat hexa jegy kell, ez jött: {value!r}")
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Nem hexa szám: {value!r}") from e


def build_board(light: tuple[int, int, int],
                dark: tuple[int, int, int],
                square: int) -> Image.Image:
    """8x8-as sakktábla. Az a8 (bal felső) VILÁGOS, ahogy a valódi táblán."""
    # Mezőnkénti minta, aztán mezőméretre nagyítva — így nincs pixelenkénti
    # ciklus, és a mezőhatárok pontosan a rácsra esnek.
    pattern = np.indices((8, 8)).sum(axis=0) % 2          # 0 = világos, 1 = sötét
    idx = np.repeat(np.repeat(pattern.astype(np.uint8), square, axis=0), square, axis=1)

    # Palettás kép: mindössze KÉT szín van benne. Mentésnél 1 bites
    # színmélységgel írjuk ki (lásd save(..., bits=1)) — a 8 bites paletta
    # ennél a képnél nagyobb fájlt adna, mint a sima RGB.
    img = Image.fromarray(idx, mode="P")
    img.putpalette(list(light) + list(dark) + [0] * (256 * 3 - 6))
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--light", type=parse_hex, default=parse_hex(DEFAULT_LIGHT),
                   help=f"világos mező hexa színe (alap: {DEFAULT_LIGHT})")
    p.add_argument("--dark", type=parse_hex, default=parse_hex(DEFAULT_DARK),
                   help=f"sötét mező hexa színe (alap: {DEFAULT_DARK})")
    p.add_argument("--square", type=int, default=DEFAULT_SQUARE,
                   help=f"egy mező oldala pixelben (alap: {DEFAULT_SQUARE})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"kimeneti fájl (alap: {DEFAULT_OUT})")
    args = p.parse_args()

    if args.square < 1:
        p.error("a mezőméretnek pozitívnak kell lennie")

    board = build_board(args.light, args.dark, args.square)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.out, optimize=True, bits=1)

    size = args.square * 8
    print(f"{args.out}  —  {size}x{size} px, {args.square} px/mező")
    print(f"  világos #{'%02X%02X%02X' % args.light}   sötét #{'%02X%02X%02X' % args.dark}")


if __name__ == "__main__":
    main()
