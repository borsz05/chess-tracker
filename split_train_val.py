"""
Train/val split script.
Helyezd a _SAKKPROJEKT_FINAL/ gyökérmappájába és futtasd onnan.

Beolvassa a data/empty, data/white, data/black mappákat,
és 80/20 arányban véletlenszerűen szétosztja őket:
    train_new/empty, train_new/white, train_new/black
    val_new/empty,   val_new/white,   val_new/black
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfig
# ---------------------------------------------------------------------------
SEED         = 42
TRAIN_RATIO  = 0.80
CLASSES      = ("empty", "white", "black")

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN    = ROOT / "train_new"
VAL      = ROOT / "val_new"

# ---------------------------------------------------------------------------
# Mappák létrehozása
# ---------------------------------------------------------------------------
for split in (TRAIN, VAL):
    for cls in CLASSES:
        (split / cls).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Split és másolás
# ---------------------------------------------------------------------------
random.seed(SEED)

total_train = 0
total_val   = 0

for cls in CLASSES:
    src = DATA_DIR / cls
    if not src.exists():
        print(f"  ✗ Nem található: {src} – kihagyva.")
        continue

    files = sorted(src.glob("*"))
    files = [f for f in files if f.is_file()]

    if not files:
        print(f"  ✗ Üres mappa: {src} – kihagyva.")
        continue

    random.shuffle(files)
    split_idx = max(1, int(len(files) * TRAIN_RATIO))

    train_files = files[:split_idx]
    val_files   = files[split_idx:]

    for f in train_files:
        shutil.copy2(f, TRAIN / cls / f.name)

    for f in val_files:
        shutil.copy2(f, VAL / cls / f.name)

    print(f"  {cls:6s}: {len(train_files):4d} train | {len(val_files):4d} val  (összesen: {len(files)})")

    total_train += len(train_files)
    total_val   += len(val_files)

print(f"\nKész. Train: {total_train} kép | Val: {total_val} kép")
print(f"Mappák: {TRAIN.name}/ és {VAL.name}/")