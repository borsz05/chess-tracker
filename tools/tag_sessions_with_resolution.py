#!/usr/bin/env python3
"""
A session-nevekhez hozzáfűzi a kamerafelbontást (`_720` / `_1080`).

MIÉRT: egy session átnyúlhat egy kamerabeállítás-váltáson (nálunk a
`maxra_allitott_lampaval_es_kislampa` fele 720p, fele 1080p), és így a
tanításnál nem lehet felbontás szerint szűrni. A felbontás a fájlnévbe kerülve
a dataset önmagát dokumentálja: `--data-root` szintű szűrés nélkül is
kiválasztható, melyik képek készültek melyik beállítással.

A felbontás forrása a `frames/<base>.jpg` teljes kamerakép mérete.

Átírja:
  - a ROI-fájlok nevét            train_new/<szin>/<base>_r<r>c<c>_<mezo>_<babu>.jpg
  - a teljes frame-ek nevét       frames/<base>.jpg
  - labels.csv                    `file` es `session` oszlop
  - shots.jsonl                   `session`, `base`, `shot_id`

Használat (repo gyökérből):
    python -m tools.tag_sessions_with_resolution              # DRY RUN: csak kiírja
    python -m tools.tag_sessions_with_resolution --apply      # tényleges átnevezés

Az --apply előtt biztonsági másolatot készít a labels.csv-ről és a
shots.jsonl-ről (.bak kiterjesztéssel).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolution_tag(frame_path: Path) -> str | None:
    if not frame_path.exists():
        return None
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    return "1080" if img.shape[1] >= 1920 else "720"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data_fen")
    ap.add_argument("--apply", action="store_true", help="tényleges átnevezés (enélkül csak kiírja, mit tenne)")
    args = ap.parse_args()

    d = args.data_dir
    shots_path, csv_path = d / "shots.jsonl", d / "labels.csv"
    for p in (shots_path, csv_path):
        if not p.exists():
            raise SystemExit(f"Nem talalhato: {p}")

    shots = [json.loads(line) for line in shots_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # shot_id -> (regi session, uj session, regi base, uj base)
    plan: dict[str, tuple[str, str, str, str]] = {}
    skipped = []
    for s in shots:
        tag = resolution_tag(d / "frames" / f"{s['base']}.jpg")
        if tag is None:
            skipped.append(s["shot_id"])
            continue
        old_sess = s["session"]
        if old_sess.endswith(("_720", "_1080")):        # mar meg van jelolve
            continue
        new_sess = f"{old_sess}_{tag}"
        new_base = s["base"].replace(old_sess, new_sess, 1)
        plan[s["shot_id"]] = (old_sess, new_sess, s["base"], new_base)

    if skipped:
        print(f"FIGYELEM: {len(skipped)} fotohoz nincs frame kep, ezeket kihagyom "
              f"(a --no-save-frames kapcsoloval keszultek?)")

    print(f"{len(plan)} foto atnevezese\n")
    per = Counter((o, n) for o, n, _, _ in plan.values())
    print(f"{'regi session':40s} -> {'uj session':44s} {'foto':>5s}")
    print("-" * 95)
    for (o, n), cnt in sorted(per.items()):
        print(f"{o:40s} -> {n:44s} {cnt:5d}")
    print()

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    renames: list[tuple[Path, Path]] = []
    new_rows = []
    for r in rows:
        p = plan.get(r["shot_id"])
        if p is None:
            new_rows.append(r)
            continue
        old_sess, new_sess, old_base, new_base = p
        rel = Path(r["file"])
        new_name = rel.name.replace(old_base, new_base, 1)
        renames.append((d / rel, d / rel.parent / new_name))
        r = dict(r)
        r["file"] = str(rel.parent / new_name)
        r["session"] = new_sess
        new_rows.append(r)

    # frame-ek is
    for old_sess, new_sess, old_base, new_base in plan.values():
        src = d / "frames" / f"{old_base}.jpg"
        if src.exists():
            renames.append((src, d / "frames" / f"{new_base}.jpg"))

    print(f"atnevezendo fajl: {len(renames)}")
    missing = [s for s, _ in renames if not s.exists()]
    if missing:
        print(f"  ebbol mar nem letezik: {len(missing)} (torolt kepek — kihagyom)")
    collisions = [t for _, t in renames if t.exists()]
    if collisions:
        raise SystemExit(f"HIBA: {len(collisions)} celfajl mar letezik, megszakitom (pl. {collisions[0]})")

    if not args.apply:
        print("\nDRY RUN — semmit nem irtam at. Vegrehajtas: --apply")
        return

    shutil.copy2(csv_path, csv_path.with_suffix(".csv.bak2"))
    shutil.copy2(shots_path, shots_path.with_suffix(".jsonl.bak"))

    done = 0
    for src, dst in renames:
        if src.exists():
            src.rename(dst)
            done += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(new_rows)

    with open(shots_path, "w", encoding="utf-8") as f:
        for s in shots:
            p = plan.get(s["shot_id"])
            if p is not None:
                old_sess, new_sess, old_base, new_base = p
                s["session"] = new_sess
                s["base"] = new_base
                s["shot_id"] = s["shot_id"].replace(old_sess, new_sess, 1)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nKESZ: {done} fajl atnevezve, labels.csv es shots.jsonl frissitve")
    print(f"  biztonsagi masolat: {csv_path.name}.bak2, {shots_path.name}.bak")


if __name__ == "__main__":
    main()
