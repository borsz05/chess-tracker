#!/usr/bin/env python3
"""Frontend kiszolgálása fejlesztéshez — gyorsítótár NÉLKÜL.

Miért nem a `python -m http.server`: az `Last-Modified` alapján enged
gyorsítótárazni, az ES-modulokat pedig a böngésző agresszíven eltárolja. Ha
egy modul régi marad, a hiba úgy néz ki, mintha a javítás meg sem történt
volna — és a Ctrl+Shift+R sem mindig segít, mert a modul-gráf almoduljai
külön kérésekben jönnek.

Használat:
    python -m tools.serve_frontend            # 8000-es port
    python -m tools.serve_frontend --port 8080
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # A no-store csak az INNENTŐL letöltött fájlokra hat. Ha a böngészőben
        # még egy korábbi (pl. `python -m http.server`) kiszolgálás Last-Modified
        # alapú bejegyzései ülnek, azok maradnak — és a hiba úgy néz ki, mintha
        # a javítás meg sem történt volna (friss HTML + régi JS/CSS). Az oldal
        # betöltésekor ezért kitakaríttatjuk az origó teljes gyorsítótárát; a
        # már megkapott válaszokat ez nem érinti, a következő kérés úgyis a
        # szerverhez megy.
        if self._is_navigation:
            self.send_header("Clear-Site-Data", '"cache"')
        super().end_headers()

    @property
    def _is_navigation(self) -> bool:
        path = self.path.split("?", 1)[0]
        return path.endswith("/") or path.endswith(".html")

    def send_header(self, keyword, value):
        # a szülő Last-Modified-ot is küldene, ami feltételes kérést enged
        if keyword.lower() == "last-modified":
            return
        super().send_header(keyword, value)

    def log_message(self, fmt, *args):
        print(f"[frontend] {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dir", default=str(FRONTEND_DIR))
    args = ap.parse_args()

    handler = partial(NoCacheHandler, directory=args.dir)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Frontend: http://localhost:{args.port}   (gyorsítótár kikapcsolva)")
    print(f"Könyvtár: {args.dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLeállítva.")


if __name__ == "__main__":
    main()
