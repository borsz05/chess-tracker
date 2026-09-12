"""A leütött bábuk sziluett-útvonalainak előállítása a Cburnett SVG-kből.

MIÉRT: a leütött bábuk sávjában nem a teljes bábukép kell, hanem lapos
sziluett, amiben a BELSŐ vonalak kivágásként látszanak. Enélkül több azonos
bábu egymásra csúsztatva egyetlen olvashatatlan folttá olvad — a futó rése,
a bástya oromzata és a szomszédos alakok elválása mind elveszik.

A forrás a Cburnett-féle sakkbábu-készlet (Colin M.L. Burnett, CC BY-SA 3.0),
ugyanaz, amit a Wikipédia és a lichess is használ — és amit a tábla is
rajzol (frontend/assets/.../chesspieces/wikipedia).

Miért a VILÁGOS változatokból dolgozunk? Mert azoknál minden vonal fekete
`stroke`-ként szerepel, tehát egy fájlban benne van a teljes részletesség.
A sötét változatokban ugyanez fehér stroke-okra van szétbontva. A sziluett
formája mindkét oldalon azonos — csak a szín tér el, azt pedig a CSS adja.

Minden path három szerep egyikét kapja:
  body — a bábu alapszínével kitöltött test  -> a sziluett színe lesz
  cut  — kontrasztszínnel kitöltött részlet  -> a háttér színe (kivágás)
  line — fill:none, csak vonal               -> a háttér színe (elválasztás)

Használat:
    python -m tools.make_piece_paths            # letölt és generál
    python -m tools.make_piece_paths --offline  # csak a helyi cache-ből
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# A Cburnett fájlok a Wikimedia Commonson. Az elérési út a fájlnév MD5-éből
# adódik: /commons/<md5[0]>/<md5[:2]>/<fájlnév>.
COMMONS = "https://upload.wikimedia.org/wikipedia/commons"
PIECES = ["p", "b", "n", "r", "q"]

CACHE = Path("tools/.cburnett-cache")
OUT = Path("frontend/js/ui/piece-paths.js")


def commons_url(filename: str) -> str:
    h = hashlib.md5(filename.encode()).hexdigest()
    return f"{COMMONS}/{h[0]}/{h[:2]}/{filename}"


def fetch(piece: str, offline: bool) -> str:
    """A VILÁGOS változat SVG-je (Chess_<p>lt45.svg)."""
    filename = f"Chess_{piece}lt45.svg"
    cached = CACHE / filename

    if cached.exists():
        return cached.read_text(encoding="utf-8")
    if offline:
        raise SystemExit(f"Hiányzik a cache-ből: {cached} (futtasd --offline nélkül)")

    req = urllib.request.Request(commons_url(filename),
                                 headers={"User-Agent": "chess-tracker-dev/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8")

    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def style_value(style: str, key: str) -> str | None:
    m = re.search(rf"{re.escape(key)}\s*:\s*([^;\"]+)", style or "")
    return m.group(1).strip() if m else None


def parse_piece(svg: str) -> list[dict]:
    """path-ok szerepekkel, a fájlbeli sorrendben (a rétegzés számít).

    A stílus ÖRÖKLŐDIK a <g> elemekről, és ezek egymásba is ágyazódnak — a
    futónál például a külső <g> fill:none, a belső viszont fill:#ffffff.
    Regexszel ez nem megfogható, ezért rendes XML-bejárás kell.
    """
    # A DOCTYPE külső DTD-re hivatkozik; az elemzéshez nem kell.
    cleaned = re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.S)
    root = ET.fromstring(cleaned)
    ns = "{http://www.w3.org/2000/svg}"

    out: list[dict] = []

    def walk(node, inherited: dict) -> None:
        style = dict(inherited)
        for key in ("fill", "stroke-linecap", "stroke-linejoin"):
            value = style_value(node.get("style", ""), key) or node.get(key)
            if value:
                style[key] = value

        if node.tag == f"{ns}path":
            d = node.get("d")
            if not d:
                return

            fill = style.get("fill", "#ffffff")
            if fill == "none":
                role = "line"
            elif fill.lower() in ("#ffffff", "#fff"):
                role = "body"      # a világos bábu alapszíne
            else:
                role = "cut"       # kontrasztos részlet (pl. a huszár szeme)

            entry = {"d": " ".join(d.split()), "role": role}
            # A vonalvégek/sarkok érdemben változtatnak a formán (a bástya
            # sávjainál pl. butt a végződés), ezért megtartjuk őket.
            for key in ("stroke-linecap", "stroke-linejoin"):
                if key in style:
                    entry[key] = style[key]
            out.append(entry)
            return

        for child in node:
            walk(child, style)

    walk(root, {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="ne töltsön le, csak a tools/.cburnett-cache-ből dolgozzon")
    args = ap.parse_args()

    data = {p: parse_piece(fetch(p, args.offline)) for p in PIECES}

    for p, paths in data.items():
        roles = ", ".join(f"{r}:{sum(1 for x in paths if x['role'] == r)}"
                          for r in ("body", "cut", "line"))
        print(f"  {p}: {len(paths)} path  ({roles})")

    body = json.dumps(data, ensure_ascii=False, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "/* GENERÁLT FÁJL — ne szerkeszd kézzel.\n"
        "   Előállítja: tools/make_piece_paths.py\n\n"
        "   A leütött bábuk sziluettjeinek útvonalai, a Cburnett-féle sakkbábu-\n"
        "   készletből (Colin M.L. Burnett, CC BY-SA 3.0) — ugyanabból, amit a\n"
        "   tábla is rajzol. A viewBox 45x45, ahogy az eredetiben.\n\n"
        "   Szerepek:\n"
        "     body — a test: a sziluett színével kitöltve\n"
        "     cut  — kontrasztos részlet: a háttér színével (kivágás)\n"
        "     line — csak vonal: a háttér színével (elválasztás)\n"
        "   A színeket a style.css adja, itt csak a geometria van. */\n\n"
        f"export const PIECE_PATHS = {body};\n",
        encoding="utf-8",
    )
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
