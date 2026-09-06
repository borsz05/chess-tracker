#!/usr/bin/env python3
"""
Sakkmező-osztályozó tanítás (Google Colab-ready) + CPU / ONNX Runtime benchmark.

Ez a tools/train_colab.py utódja. Három dolgot csinál:

  1. --benchmark-only : a jelöltarchitektúrákat CPU-n, ONNX Runtime alatt méri
                        (batch=64 és batch=16, 100 és 128 px), és táblázatot ír.
  2. tanítás          : hibrid multi-task modell (3-osztályos szín-fej +
                        7-osztályos bábutípus-fej), fekete-bábu-célzott
                        augmentációval, macro-recall alapú modellkiválasztással.
  3. checkpoint       : az OccupancyColorModel szerződése szerinti .pt fájl
                        (variant / model_state / class_names / img_size / normalize),
                        + mentés utáni önteszt az osztálysorrend-fordításra.

A modellarchitektúra EGYETLEN forrása: vision/models/square_net.py (a tanítás
és az inferencia ugyanazt importálja, így nem tud szétcsúszni).

===========================================================================
ARCHITEKTÚRA-DÖNTÉS — MÉRÉS ALAPJÁN (ne találgatás)
===========================================================================

Mérés helye: a CÉLGÉP (Intel Alder Lake-UP3, 12 mag, Iris Xe, NINCS CUDA),
ONNX Runtime 1.29 CPUExecutionProvider, intra_op_threads=10, fp32,
véletlen súlyokkal (a latencia nem függ a súlyoktól), 30 futás p50.
Futtatás: `python tools/train_square_classifier.py --benchmark-only`
(a számokat a szkript a --bench-out JSON-ba is kiírja).

    variant             img  params   ORT b64 p50   ORT b16 p50   torch b64 p50
    ------------------  ---  ------   -----------   -----------   -------------
    tiny_cnn            100   0.48M      25.4 ms       6.7 ms       108.5 ms
    tiny_cnn            128   0.48M      39.8 ms      11.4 ms       174.2 ms
    mobilenet_v3_small  100   0.93M      23.4 ms       6.5 ms        53.4 ms   <- VÁLASZTOTT
    mobilenet_v3_small  128   0.93M      32.7 ms       7.5 ms        81.6 ms
    shufflenet_v2_x1_0  100   1.26M      38.4 ms      11.7 ms        83.0 ms
    shufflenet_v2_x1_0  128   1.26M      54.5 ms      13.3 ms       134.8 ms
    efficientnet_b0     100   4.02M     137.7 ms      30.9 ms       353.8 ms   (depthwise+SE: CPU-n lassú)
    efficientnet_b0     128   4.02M     208.6 ms      44.0 ms       576.6 ms
    resnet18            100  11.18M     193.6 ms      48.6 ms       352.3 ms   (jelenlegi referencia)
    resnet18            128  11.18M     254.3 ms      65.2 ms       501.3 ms
    (p95 a 100 px-es sorokban: tiny 26.1 | mnv3 25.5 | shuffle 46.1 | effb0 159.9 | resnet18 201.1 ms)

Referencia a jelenlegi rendszerből (PipelineProfiler, classifier_full,
ResNet18 @100px, torch CPU): mean 300 ms, p50 310 ms, p95 345 ms.
Ugyanez ezen a gépen, torch, batch 64 @100px, 10 szál, újramérve: p50 ~354 ms.

Pontosság-referencia (jelenlegi resnet18_best_topdown.pt, 3 osztály):
    val_new (820 kép)         : empty 100 % | white 100 % | black 100 %   -> TELÍTETT,
                                 ez a halmaz NEM méri a fekete-bábu problémát
    régi val (1999 kép, PNG)  : empty 97.2 % | white 99.3 % | black 97.9 %, macro 98.1 %
A fekete bábu élesben mért hibája (sötét mezőn "empty"/"white") tehát csak a
FEN-ből címkézett új, éles gyűjtésen (tools/collect_fen_dataset.py) mérhető:
ezért támogatja a szkript a --eval-dir kapcsolót, ahol a `black` recall-t
külön riportáljuk.

DÖNTÉS: MobileNetV3-Small, 100 px, ImageNet-előtanított backbone.

  Latencia (a domináns szempont, célgépen mérve): ORT batch 64 @100 px p50 23.4 ms,
  p95 25.5 ms -> a 300 ms-os classifier_full ~13x gyorsul, a 60 ms-os cél alatt
  2.5x tartalékkal; a partial út (16 mező) 6.5 ms. Csak a tiny_cnn van hasonló
  sávban (25.4 ms), a ShuffleNet 1.6x lassabb, az EfficientNet-B0 és a ResNet18
  a célt sem hozza.

  Pontosság (rövid, 10 epochos helyi kontroll-tanítás CPU-n, csak a train_new-n,
  azonos augmentációval; `--data-root <dataset> --eval-dir <dataset>/val`):
                          val_new (in-domain)     régi val (MÁS kamera, 1999 kép)
    mobilenet_v3_small    100 % / black 100 %     macro 98.5 % | black 100 % | empty 95.6 % | white 99.8 %
    tiny_cnn (scratch)    100 % / black 100 %     macro 77.2 % | black 94.5 % | empty 46.3 % | white 90.9 %
    resnet18 (jelenlegi)  100 % / black 100 %     macro 98.1 % | black 97.9 % | empty 97.2 % | white 99.3 %
  A val_new mindhárom modellnek telített, ott nincs különbség. A NEM látott
  kameraeloszláson az előtanított MobileNetV3 tartja a pontosságot (és a black
  recall-ban a jelenlegi ResNet18-at is veri), a nulláról tanított tiny_cnn
  összeomlik (empty -> black). Az architektúra tehát: ugyanaz a sebesség, de
  az ImageNet-feature-ök adják a megvilágítás/kamera-robusztusságot, ami a
  fekete-bábu problémánál a lényeg. Kontroll: ha a tiny_cnn a régi train-t IS
  látja (11 269 kép), a régi val-on ő is 99.9 % macro / 100 % black -> a
  különbség adat-lefedettség, nem kapacitás; a 3-osztályos feladathoz mindkét
  háló elég. Ezért a döntő szempont a NEM látott körülményekre való
  robusztusság (ezt adja az előtanítás), és a tiny_cnn csak tartalék
  (--arch tiny_cnn), ha valaha még 2 ms-ot kellene faragni.

  Bemeneti méret: 128 px 40 %-kal lassabb (32.7 vs 23.4 ms); pontosságnyereségét
  NEM mértük, és a pipeline 100 px-re méretez -> marad a 100 px (13. pont: a
  felbontást csak mért nyereségért emeljük — ha kell, --img-size 128 + --eval-dir).

  Érvényességi korlát: a fenti pontosság-számok rövid CPU-s kontroll-futásból
  jönnek; a végleges modellt Colabon, teljes epochszámmal, a régi + új
  (FEN-ből címkézett) adaton kell tanítani, és a black recall-t az éles
  gyűjtés val_new-ján (--eval-dir) riportálni.

===========================================================================
COLAB GYORSINDÍTÁS (cellánként)
===========================================================================

    # A szkriptet NEM bemásolni kell, hanem a repót klónozni: importálja a
    # vision/models/square_net.py-t (az architektúra egyetlen forrása).
    !git clone https://github.com/borsz05/chess-tracker
    !git clone https://github.com/borsz05/sakk_modelltanitas          # regi dataset
    !git clone https://github.com/borsz05/modelltanitas_kepek         # uj, FEN-cimkezett gyujtes
    %cd chess-tracker
    !pip install -q onnx onnxruntime

    from google.colab import drive; drive.mount('/content/drive')     # a checkpointnak

    # (opcionális) a benchmark Colab-CPU-n — csak a SORREND informatív,
    # az abszolút számok a célgépen mérendők:
    !python tools/train_square_classifier.py --benchmark-only

    # tanítás — a régi GitHub dataset + az új FEN-címkézett gyűjtés együtt.
    # A --eval-dir az UJ gyujtes val_new-ja: a `black` recall fo szama ott
    # mérendő, mert a régi val telített (minden modell 100%-ot ad rajta).
    !python tools/train_square_classifier.py \
        --data-root ../sakk_modelltanitas \
        --data-root ../modelltanitas_kepek \
        --arch mobilenet_v3_small --img-size 128 --epochs 60 \
        --eval-dir ../modelltanitas_kepek/val_new \
        --out /content/drive/MyDrive/mnv3_squares.pt

Adatszerkezet (minden --data-root alatt), az ImageFolder-konvenció szerint:
    <root>/train_new/{black,empty,white}/*.jpg
    <root>/val_new/{black,empty,white}/*.jpg
Bábutípus-címke a fájlnévből: `..._<mező>_<bábu>.jpg`, pl. `..._e4_P.jpg`,
`..._h8_x.jpg` (x = üres). Ha a fájlnévben nincs ilyen (régi adat), a típus
ismeretlen -> a típus-loss kihagyja (ignore_index), a szín-loss használja.
Az `empty/` mappában lévő képek típusa akkor is "none", ha nincs címke.

===========================================================================
KÖTELEZŐ CHECKPOINT-SZERZŐDÉS (OccupancyColorModel)
===========================================================================
    variant, model_state, class_names, img_size, normalize{mean,std}
    class_names PONTOSAN a color-fej kimeneti indexeinek sorrendjében —
    ImageFolder-nél ábécésorrend: ['black', 'empty', 'white'].
    A pipeline-kódolás (empty=0, white=1, black=2) fordítását az
    idx_to_label végzi; a szkript mentés után visszatölti a fájlt és
    ellenőrzi, hogy a fordítás a val-halmazon ugyanazt adja.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ---------------------------------------------------------------------------
# Projekt import — a repo gyökere a sys.path-ra, akár `python tools/x.py`,
# akár `python -m tools.x` a hívás (Colabban a klónozott repo gyökeréből).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from vision.models.square_net import (  # noqa: E402
        PIECE_TYPE_CLASSES,
        PIPELINE_LABEL,
        TYPE_IGNORE_INDEX,
        VARIANTS,
        MultiTaskSquareNet,
        build_from_checkpoint,
        class_names_to_idx_to_label,
        make_checkpoint,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Nem található vision/models/square_net.py — a szkriptet a chess-tracker repo "
        "gyökeréből futtasd (Colabban: !git clone https://github.com/borsz05/chess-tracker). "
        f"({exc})"
    )

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
COLOR_CLASS_NAMES = ["black", "empty", "white"]   # ImageFolder ábécésorrend — a szerződés

# fájlnév-végződés: _<mező>_<bábu>.<ext>   (a collect_fen_dataset.py írja)
_TYPE_TAG_RE = re.compile(r"_([a-h][1-8])_([PNBRQKpnbrqkx])\.[A-Za-z]+$")
_SYMBOL_TO_TYPE_IDX = {"x": 0, "p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}


# ===========================================================================
# 1. BENCHMARK — CPU, ONNX Runtime, batch 64 / 16
# ===========================================================================

def export_onnx(model: nn.Module, img_size: int, path: Path) -> None:
    """Dinamikus batch-dimenzió: a partial útvonal változó számú mezőt küld."""
    model = model.eval()
    dummy = torch.randn(2, 3, img_size, img_size)
    kwargs: dict[str, Any] = dict(
        input_names=["input"],
        output_names=["color_logits", "type_logits"],
        dynamic_axes={"input": {0: "batch"}, "color_logits": {0: "batch"}, "type_logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:  # régebbi torch: nincs `dynamo` kulcsszó
        torch.onnx.export(model, (dummy,), str(path), **kwargs)


def _time_fn(fn, n_warmup: int, n_runs: int) -> dict[str, float]:
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "min": float(a.min())}


def bench_onnx(path: Path, img_size: int, batch: int, threads: int, n_runs: int) -> dict[str, float]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, 3, img_size, img_size).astype(np.float32)
    return _time_fn(lambda: sess.run(None, {"input": x}), n_warmup=5, n_runs=n_runs)


def bench_torch(model: nn.Module, img_size: int, batch: int, threads: int, n_runs: int) -> dict[str, float]:
    torch.set_num_threads(int(threads))
    model = model.eval()
    x = torch.randn(batch, 3, img_size, img_size)
    with torch.inference_mode():
        return _time_fn(lambda: model(x), n_warmup=3, n_runs=n_runs)


def run_benchmark(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    archs = args.bench_archs or list(VARIANTS)
    sizes = args.bench_img_sizes or [100, 128]
    out_dir = Path(args.bench_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  BENCHMARK  onnxruntime {ort.__version__} | torch {torch.__version__} | "
          f"threads={args.threads} | cpu_count={os.cpu_count()} | providers={ort.get_available_providers()}")
    print(f"  batch={args.bench_batch} (teljes tábla) és {args.bench_partial_batch} (partial út), fp32, véletlen súlyok")
    print("=" * 78)

    rows: list[dict[str, Any]] = []
    for arch in archs:
        for size in sizes:
            model = MultiTaskSquareNet(arch, pretrained=False, dropout=0.0).eval()
            n_params = sum(p.numel() for p in model.parameters())
            onnx_path = out_dir / f"bench_{arch}_{size}.onnx"
            export_onnx(model, size, onnx_path)

            r_full = bench_onnx(onnx_path, size, args.bench_batch, args.threads, args.bench_runs)
            r_part = bench_onnx(onnx_path, size, args.bench_partial_batch, args.threads, args.bench_runs)
            r_torch = bench_torch(model, size, args.bench_batch, args.threads, max(10, args.bench_runs // 3))

            row = {
                "variant": arch, "img_size": size, "params": n_params,
                "onnx_size_mb": onnx_path.stat().st_size / 1e6,
                "ort_full": r_full, "ort_partial": r_part, "torch_full": r_torch,
            }
            rows.append(row)
            print(f"  {arch:20s} {size:4d}px  params={n_params/1e6:5.2f}M  "
                  f"ORT b{args.bench_batch}: p50={r_full['p50']:7.1f} p95={r_full['p95']:7.1f} ms | "
                  f"ORT b{args.bench_partial_batch}: p50={r_part['p50']:6.1f} ms | "
                  f"torch b{args.bench_batch}: p50={r_torch['p50']:7.1f} ms", flush=True)

    print("\n  Összefoglaló (p50, ms) — cél: classifier_full (64 mező) < 60 ms")
    print(f"  {'variant':20s} {'img':>4s} {'params':>8s} {'ORT b64':>9s} {'ORT b16':>9s} {'torch b64':>10s}  cél")
    for r in rows:
        ok = "OK " if r["ort_full"]["p50"] < 60 else "---"
        print(f"  {r['variant']:20s} {r['img_size']:4d} {r['params']/1e6:7.2f}M "
              f"{r['ort_full']['p50']:9.1f} {r['ort_partial']['p50']:9.1f} {r['torch_full']['p50']:10.1f}  {ok}")

    if args.bench_out:
        Path(args.bench_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.bench_out, "w", encoding="utf-8") as f:
            json.dump({"onnxruntime": ort.__version__, "torch": torch.__version__, "threads": args.threads,
                       "cpu_count": os.cpu_count(), "rows": rows}, f, indent=2)
        print(f"\n  Benchmark JSON mentve: {args.bench_out}")


# ===========================================================================
# 2. ADAT
# ===========================================================================

def type_idx_from_filename(name: str, color_dir: str) -> int:
    """`..._e4_P.jpg` -> 1 (pawn); `..._h8_x.jpg` -> 0; címke nélkül: empty->0, egyébként ignore."""
    m = _TYPE_TAG_RE.search(name)
    if m:
        return _SYMBOL_TO_TYPE_IDX[m.group(2).lower()]
    return 0 if color_dir == "empty" else TYPE_IGNORE_INDEX


def gather_samples(roots: list[Path], split: str) -> list[tuple[Path, int, int]]:
    """
    Minden root alatt <root>/<split>/{black,empty,white}/ — ha nincs <split>
    almappa, de a root közvetlenül tartalmazza az osztálymappákat, azt vesszük
    (így egy egyszerű `data/{black,empty,white}` mappa is megadható --eval-dir-nek).
    Visszaad: [(path, color_idx, type_idx)], color_idx a COLOR_CLASS_NAMES sorrendben.
    """
    out: list[tuple[Path, int, int]] = []
    for root in roots:
        base = root / split
        if not base.is_dir() and all((root / c).is_dir() for c in COLOR_CLASS_NAMES):
            base = root
        if not base.is_dir():
            continue
        for ci, cname in enumerate(COLOR_CLASS_NAMES):
            d = base / cname
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    out.append((p, ci, type_idx_from_filename(p.name, cname)))
    return out


class SquareDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, color, ptype = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(color), int(ptype)


# --- átméretezés: PONTOSAN az éles út szerint ------------------------------

class CvResize:
    """126x126 ROI -> img_size, ugyanazzal a módszerrel, mint élesben.

    Az éles út (vision/models/occupancy_color_model.preprocess_rois) cv2-t
    használ: INTER_AREA kicsinyítéskor, INTER_CUBIC nagyításkor. A torchvision
    `transforms.Resize` bilineáris útja ettől mérhetően eltér — 200 valódi
    fekete-bábu ROI-n átlag 0,5/255 szint, a pixelek 1,4%-ánál >5 szint.
    Kicsi, de ingyen megszüntethető train/inference eltérés.

    A resize csatornánként független, ezért mindegy, hogy RGB (PIL) vagy BGR
    (éles út) sorrendben fut — az eredmény azonos.
    """

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img)
        h, w = a.shape[:2]
        if (h, w) == (self.size, self.size):
            return img
        interp = cv2.INTER_CUBIC if (w < self.size or h < self.size) else cv2.INTER_AREA
        return Image.fromarray(cv2.resize(a, (self.size, self.size), interpolation=interp))


# --- egyedi augmentációk (PIL -> PIL) --------------------------------------

class RandomGamma:
    """Log-egyenletes gamma [lo, hi]: gamma>1 sötétít (fekete bábu sötét mezőn még sötétebb)."""

    def __init__(self, lo: float = 0.55, hi: float = 1.8, p: float = 0.6):
        self.lo, self.hi, self.p = lo, hi, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        g = math.exp(random.uniform(math.log(self.lo), math.log(self.hi)))
        lut = [int(255.0 * ((i / 255.0) ** g) + 0.5) for i in range(256)]
        return img.point(lut * 3)


class RandomShadow:
    """
    Árnyék-szimuláció: a robotkar (és a játékos keze) lágy szélű árnyékot vet a
    táblára. Egy véletlen egyenes egyik oldalát 0.35–0.85-szörösre sötétítjük,
    lágy (sigmoid) átmenettel; néha egy elliptikus foltot használunk.
    """

    def __init__(self, p: float = 0.35, strength: tuple[float, float] = (0.35, 0.85)):
        self.p, self.strength = p, strength

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        a = np.asarray(img, dtype=np.float32)
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        k = random.uniform(*self.strength)
        if random.random() < 0.7:
            theta = random.uniform(0, 2 * math.pi)
            nx, ny = math.cos(theta), math.sin(theta)
            cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
            d = (xx - cx) * nx + (yy - cy) * ny
            soft = random.uniform(2.0, 12.0)
            mask = 1.0 / (1.0 + np.exp(-d / soft))         # 0..1, egyik oldal árnyékos
        else:
            cx, cy = random.uniform(0.1, 0.9) * w, random.uniform(0.1, 0.9) * h
            rx, ry = random.uniform(0.25, 0.7) * w, random.uniform(0.25, 0.7) * h
            d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            mask = np.clip(1.0 - d, 0.0, 1.0) ** 0.5
        factor = 1.0 - (1.0 - k) * mask[..., None]
        return Image.fromarray(np.clip(a * factor, 0, 255).astype(np.uint8))


class RandomGaussianNoise:
    """Szenzorzaj a normalizált tenzoron (gyenge fényben a kamera zajosabb)."""

    def __init__(self, p: float = 0.3, sigma: tuple[float, float] = (0.01, 0.05)):
        self.p, self.sigma = p, sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        return x + torch.randn_like(x) * random.uniform(*self.sigma)


def build_train_transform(img_size: int, strong: bool = True):
    """
    A fekete-bábu problémára célzott augmentáció (feladatleírás 7.3):
      - erős brightness/contrast + random gamma (megvilágítás)
      - RandomAutocontrast / RandomEqualize (fekete bábu kontrasztja sötét mezőn)
      - árnyék-szimuláció (robotkar), RandomErasing (részleges takarás)
    Geometria: a kamera EGY oldalról nézi a táblát, a bábuk a warpon egy
    irányba "dőlnek", ezért függőleges tükrözés NINCS (irreális képet adna);
    vízszintes tükrözés és kis affin torzítás igen.
    """
    aug: list[Any] = [
        CvResize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=6, translate=(0.04, 0.04), scale=(0.94, 1.06), shear=3),
    ]
    if strong:
        aug += [
            RandomShadow(p=0.35),
            transforms.ColorJitter(brightness=0.55, contrast=0.55, saturation=0.30, hue=0.03),
            RandomGamma(0.55, 1.8, p=0.6),
            transforms.RandomAutocontrast(p=0.30),
            transforms.RandomEqualize(p=0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.30),
        ]
    aug += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        RandomGaussianNoise(p=0.30),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ]
    return transforms.Compose(aug)


def build_eval_transform(img_size: int):
    return transforms.Compose([
        CvResize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ===========================================================================
# 3. LOSS — focal (szín) + maszkolt CE (típus)
# ===========================================================================

class FocalLoss(nn.Module):
    """FL = (1-p_t)^gamma * CE ; gamma=0 -> sima (súlyozott, label-smoothed) CE."""

    def __init__(self, gamma: float, weight: torch.Tensor | None, label_smoothing: float):
        super().__init__()
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        if self.gamma <= 0:
            return ce.mean()
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class MultiTaskLoss(nn.Module):
    """
    total = color_loss + type_weight * type_loss
    A típus-loss csak a címkézett (type != TYPE_IGNORE_INDEX) mintákon számít;
    ha egy batchben nincs ilyen, a típus-tag 0. A típus-fej gradiensét a
    backbone felé a modell GradScale-je skálázza (--type-grad-scale), így a
    3-osztályos fejet egy rossz típus-jel nem tudja elrontani.
    """

    def __init__(self, color_loss: nn.Module, type_weight: float, type_label_smoothing: float = 0.05):
        super().__init__()
        self.color_loss = color_loss
        self.type_weight = float(type_weight)
        self.type_ce = nn.CrossEntropyLoss(ignore_index=TYPE_IGNORE_INDEX, label_smoothing=type_label_smoothing)

    def forward(self, color_logits, type_logits, color_t, type_t):
        lc = self.color_loss(color_logits, color_t)
        if self.type_weight > 0 and bool((type_t != TYPE_IGNORE_INDEX).any()):
            lt = self.type_ce(type_logits, type_t)
        else:
            lt = torch.zeros((), device=color_logits.device)
        return lc + self.type_weight * lt, lc.detach(), lt.detach()


# ===========================================================================
# 4. KIÉRTÉKELÉS
# ===========================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: MultiTaskLoss | None = None) -> dict[str, Any]:
    model.eval()
    nC, nT = len(COLOR_CLASS_NAMES), len(PIECE_TYPE_CLASSES)
    cm_color = np.zeros((nC, nC), dtype=np.int64)
    cm_type = np.zeros((nT, nT), dtype=np.int64)
    total_loss, n = 0.0, 0
    for imgs, ct, tt in loader:
        imgs, ct, tt = imgs.to(device), ct.to(device), tt.to(device)
        cl, tl = model(imgs)
        if criterion is not None:
            loss, _, _ = criterion(cl, tl, ct, tt)
            total_loss += float(loss) * imgs.size(0)
        n += imgs.size(0)
        pc = cl.argmax(1)
        for t, p in zip(ct.cpu().numpy(), pc.cpu().numpy()):
            cm_color[t, p] += 1
        m = tt != TYPE_IGNORE_INDEX
        if bool(m.any()):
            pt = tl[m].argmax(1)
            for t, p in zip(tt[m].cpu().numpy(), pt.cpu().numpy()):
                cm_type[t, p] += 1

    support = cm_color.sum(1)
    recall = np.divide(np.diag(cm_color), np.maximum(support, 1), dtype=np.float64)
    precision = np.divide(np.diag(cm_color), np.maximum(cm_color.sum(0), 1), dtype=np.float64)
    present = support > 0
    macro_recall = float(recall[present].mean()) if present.any() else 0.0
    acc = float(np.trace(cm_color) / max(1, cm_color.sum()))

    t_support = cm_type.sum(1)
    t_present = t_support > 0
    t_recall = np.divide(np.diag(cm_type), np.maximum(t_support, 1), dtype=np.float64)
    type_acc = float(np.trace(cm_type) / cm_type.sum()) if cm_type.sum() > 0 else float("nan")
    type_macro = float(t_recall[t_present].mean()) if t_present.any() else float("nan")

    bi = COLOR_CLASS_NAMES.index("black")
    return {
        "n": n, "loss": total_loss / max(1, n) if criterion is not None else float("nan"),
        "acc": acc, "macro_recall": macro_recall,
        "recall": {c: float(recall[i]) for i, c in enumerate(COLOR_CLASS_NAMES)},
        "precision": {c: float(precision[i]) for i, c in enumerate(COLOR_CLASS_NAMES)},
        "support": {c: int(support[i]) for i, c in enumerate(COLOR_CLASS_NAMES)},
        "black_recall": float(recall[bi]),
        "black_precision": float(precision[bi]),
        "confusion_color": cm_color.tolist(),
        "type_n": int(cm_type.sum()), "type_acc": type_acc, "type_macro_recall": type_macro,
        "type_recall": {c: float(t_recall[i]) for i, c in enumerate(PIECE_TYPE_CLASSES) if t_present[i]},
        "confusion_type": cm_type.tolist(),
    }


def format_confusion(cm: list[list[int]], names: list[str] | tuple[str, ...]) -> str:
    w = max(8, max(len(n) for n in names))
    head = " " * (w + 3) + "".join(f"{n:>{w}s}" for n in names) + "   (pred ->)"
    lines = [head]
    for i, n in enumerate(names):
        lines.append(f"{n:>{w}s} | " + "".join(f"{cm[i][j]:>{w}d}" for j in range(len(names))))
    lines.append("(sorok = valódi, oszlopok = predikált)")
    return "\n".join(lines)


def format_report(name: str, ev: dict[str, Any]) -> str:
    out = [f"--- {name}: n={ev['n']} acc={ev['acc']*100:.2f}% macro_recall={ev['macro_recall']*100:.2f}%"]
    for c in COLOR_CLASS_NAMES:
        out.append(f"    {c:6s} recall={ev['recall'][c]*100:6.2f}%  precision={ev['precision'][c]*100:6.2f}%  n={ev['support'][c]}")
    out.append(f"    >>> BLACK RECALL = {ev['black_recall']*100:.2f}%  (a projekt kulcsmetrikája)")
    out.append(format_confusion(ev["confusion_color"], COLOR_CLASS_NAMES))
    if ev["type_n"] > 0:
        out.append(f"    típus-fej: n={ev['type_n']} acc={ev['type_acc']*100:.2f}% macro_recall={ev['type_macro_recall']*100:.2f}%")
        out.append("    " + "  ".join(f"{k}={v*100:.1f}%" for k, v in ev["type_recall"].items()))
        out.append(format_confusion(ev["confusion_type"], PIECE_TYPE_CLASSES))
    else:
        out.append("    típus-fej: nincs típus-címkézett minta ebben a halmazban")
    return "\n".join(out)


# ===========================================================================
# 5. TANÍTÁS
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds: Dataset, batch_size: int, workers: int, sampler=None, shuffle: bool = False, device=None) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle and sampler is None,
        num_workers=workers, pin_memory=(device is not None and device.type == "cuda"),
        persistent_workers=workers > 0, drop_last=False,
    )


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    roots = [Path(r) for r in args.data_root]
    train_samples = gather_samples(roots, "train_new")
    val_samples = gather_samples(roots, "val_new")
    if args.quick:
        random.shuffle(train_samples); random.shuffle(val_samples)
        train_samples, val_samples = train_samples[: args.quick], val_samples[: max(64, args.quick // 4)]
    if not train_samples or not val_samples:
        raise SystemExit(f"Nincs adat: train={len(train_samples)} val={len(val_samples)} a {roots} alatt "
                         "(várt: <root>/train_new/{black,empty,white}, <root>/val_new/...)")

    def _describe(name: str, s: list[tuple[Path, int, int]]) -> None:
        cc = np.bincount([c for _, c, _ in s], minlength=3)
        typed = sum(1 for _, _, t in s if t != TYPE_IGNORE_INDEX)
        print(f"  {name}: {len(s)} kép | " + " ".join(f"{n}={cc[i]}" for i, n in enumerate(COLOR_CLASS_NAMES))
              + f" | típus-címkézett: {typed}")
    print("Adat:"); _describe("train", train_samples); _describe("val", val_samples)

    train_ds = SquareDataset(train_samples, build_train_transform(args.img_size, strong=not args.weak_aug))
    val_ds = SquareDataset(val_samples, build_eval_transform(args.img_size))

    # WeightedRandomSampler — osztály-egyensúly a batchekben (empty ~2x annyi mint black)
    labels = np.array([c for _, c, _ in train_samples])
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    sampler = None
    if not args.no_balanced_sampler:
        w = (1.0 / np.maximum(counts, 1))[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)

    train_loader = make_loader(train_ds, args.batch_size, args.workers, sampler=sampler, shuffle=True, device=device)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, device=device)

    # Modell
    model = MultiTaskSquareNet(
        args.arch, n_color=3, n_type=len(PIECE_TYPE_CLASSES),
        pretrained=not args.no_pretrained and args.arch != "tiny_cnn",
        type_grad_scale=args.type_grad_scale, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {args.arch} | params={n_params/1e6:.2f}M | img={args.img_size} | "
          f"type_loss_weight={args.type_loss_weight} type_grad_scale={args.type_grad_scale}")

    # Loss: a sampler már kiegyensúlyoz, ezért a class-weight alapból 1.0 —
    # a --black-weight csak akkor emelendő, ha az éles eval black recall-ja
    # a többi osztály mögött marad (a kettő együtt túlsúlyozhat).
    cw = torch.ones(3, device=device)
    cw[COLOR_CLASS_NAMES.index("black")] = args.black_weight
    criterion = MultiTaskLoss(
        FocalLoss(args.focal_gamma, weight=cw, label_smoothing=args.label_smoothing),
        type_weight=args.type_loss_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = out_path.parent / (out_path.stem + "_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    best_metric, best_epoch, best_state, best_eval = -1.0, 0, None, None
    patience_left = args.patience
    history: list[dict[str, Any]] = []
    stall_epochs = 0

    print("\n" + "=" * 78)
    print(f"  Tanítás: {args.epochs} epoch | batch={args.batch_size} | lr={args.lr} | focal_gamma={args.focal_gamma} | amp={use_amp}")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sum_loss = sum_lc = sum_lt = 0.0
        seen = 0
        for imgs, ct, tt in train_loader:
            imgs, ct, tt = imgs.to(device, non_blocking=True), ct.to(device), tt.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                cl, tl = model(imgs)
                loss, lc, lt = criterion(cl.float(), tl.float(), ct, tt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            b = imgs.size(0)
            sum_loss += loss.item() * b; sum_lc += lc.item() * b; sum_lt += lt.item() * b; seen += b

        ev = evaluate(model, val_loader, device, criterion)
        metric = ev["macro_recall"]          # <- modellkiválasztás: osztályonkénti recall átlaga
        elapsed = time.time() - t0
        rec = {"epoch": epoch, "train_loss": sum_loss / seen, "train_color_loss": sum_lc / seen,
               "train_type_loss": sum_lt / seen, "val": {k: v for k, v in ev.items() if not k.startswith("confusion")},
               "lr": scheduler.get_last_lr()[0], "type_grad_scale": model.type_grad.scale, "sec": elapsed}
        history.append(rec)
        print(f"Ep {epoch:3d}/{args.epochs} | loss={rec['train_loss']:.4f} (c={rec['train_color_loss']:.4f} t={rec['train_type_loss']:.4f}) | "
              f"val_loss={ev['loss']:.4f} acc={ev['acc']*100:.2f}% macro={metric*100:.2f}% | "
              + " ".join(f"{c[:1]}={ev['recall'][c]*100:.1f}" for c in COLOR_CLASS_NAMES)
              + (f" | type_acc={ev['type_acc']*100:.1f}%" if ev["type_n"] else "")
              + f" | {elapsed:.0f}s", flush=True)

        if metric > best_metric + 1e-6:
            best_metric, best_epoch, best_eval = metric, epoch, ev
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
            stall_epochs = 0
            print(f"   -> új legjobb (macro_recall={metric*100:.2f}%, black_recall={ev['black_recall']*100:.2f}%)")
        else:
            patience_left -= 1
            stall_epochs += 1
            # Negatív transzfer-őr: ha a szín-fej tartósan nem javul, a
            # típus-fej gradiensét megfelezzük a backbone felé (naplózva).
            if (args.auto_type_grad and args.type_loss_weight > 0 and model.type_grad.scale > 0.05
                    and stall_epochs >= max(2, args.patience // 3)):
                model.set_type_grad_scale(model.type_grad.scale * 0.5)
                stall_epochs = 0
                print(f"   ! szín-fej stagnál -> type_grad_scale={model.type_grad.scale:.3f}")
            if patience_left <= 0:
                print(f"   early stopping ({args.patience} epoch óta nincs macro_recall javulás)")
                break

    assert best_state is not None and best_eval is not None
    model.load_state_dict(best_state)
    model.eval()
    print(f"\nLegjobb epoch: {best_epoch} | macro_recall={best_metric*100:.2f}% | black_recall={best_eval['black_recall']*100:.2f}%")
    print(format_report("val_new (legjobb checkpoint)", best_eval))

    # Külön eval-halmazok (pl. az új, FEN-ből címkézett éles gyűjtés)
    extra_reports: dict[str, Any] = {}
    for d in args.eval_dir or []:
        s = gather_samples([Path(d)], "val_new")
        if not s:
            print(f"  (eval-dir üres vagy hiányzó szerkezet: {d})"); continue
        ev_x = evaluate(model, make_loader(SquareDataset(s, build_eval_transform(args.img_size)), args.batch_size, args.workers, device=device), device)
        extra_reports[str(d)] = ev_x
        print(format_report(f"eval-dir {d}", ev_x))

    # ---- Mentés — a szerződés szerint ---------------------------------------
    ckpt = make_checkpoint(
        model, class_names=COLOR_CLASS_NAMES, img_size=args.img_size,
        normalize={"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        extra={
            "epoch": best_epoch, "macro_recall": best_metric, "val_acc": best_eval["acc"],
            "black_recall": best_eval["black_recall"], "train_args": vars(args),
            "type_grad_scale_final": model.type_grad.scale,
        },
    )
    torch.save(ckpt, out_path)
    print(f"\nCheckpoint mentve: {out_path.resolve()}  (variant={ckpt['variant']}, class_names={ckpt['class_names']}, idx_to_label={ckpt['idx_to_label']})")

    with open(report_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(report_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "val_new": best_eval, "eval_dirs": extra_reports, "args": vars(args)}, f, indent=2, ensure_ascii=False)
    with open(report_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(format_report("val_new", best_eval) + "\n\n")
        for k, v in extra_reports.items():
            f.write(format_report(k, v) + "\n\n")
    np.save(report_dir / "confusion_color.npy", np.asarray(best_eval["confusion_color"]))
    print(f"Riportok: {report_dir.resolve()}")

    self_test_checkpoint(out_path, val_ds, device)
    return out_path


# ===========================================================================
# 6. ÖNTESZT — a mentett fájl visszatöltése + osztálysorrend-fordítás
# ===========================================================================

@torch.no_grad()
def self_test_checkpoint(path: Path, val_ds: Dataset, device: torch.device, n: int = 128) -> None:
    """
    1) A checkpoint visszatölthető a közös build_from_checkpoint-tal.
    2) A `class_names` -> idx_to_label fordítás a pipeline-kódolást adja:
       a modell argmax indexének NEVE (class_names[idx]) ugyanaz az osztály,
       amit idx_to_label[idx] pipeline-kódja jelent. Ha ez elcsúszik, a fehér
       és a fekete némán felcserélődik — pont ezt fogja meg ez a teszt.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("variant", "model_state", "class_names", "img_size", "normalize"):
        assert k in ckpt, f"checkpoint-szerződés sérül: hiányzik '{k}'"
    model = build_from_checkpoint(ckpt).to(device)
    idx_to_label = class_names_to_idx_to_label(ckpt["class_names"])
    assert idx_to_label.tolist() == ckpt["idx_to_label"]

    idxs = list(range(min(n, len(val_ds))))
    xs, cts = [], []
    for i in idxs:
        x, c, _ = val_ds[i]
        xs.append(x); cts.append(c)
    color_logits, _ = model(torch.stack(xs).to(device))
    pred_idx = color_logits.argmax(1).cpu().numpy()
    for pi, ci in zip(pred_idx, cts):
        name = ckpt["class_names"][pi]
        assert PIPELINE_LABEL[name] == int(idx_to_label[pi]), (name, pi, idx_to_label)
    # a val-címke neve is ugyanezen a sorrenden alapul
    truth_names = [ckpt["class_names"][c] for c in cts]
    agree = float(np.mean([ckpt["class_names"][p] == t for p, t in zip(pred_idx, truth_names)]))
    print(f"Önteszt OK: visszatöltés + class_names->idx_to_label fordítás konzisztens "
          f"({len(idxs)} val minta, egyezés a címkékkel: {agree*100:.1f}%)")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # adat
    p.add_argument("--data-root", action="append", default=None,
                   help="dataset gyökér (<root>/train_new, <root>/val_new); többször is megadható, összefűzi")
    p.add_argument("--eval-dir", action="append", default=None,
                   help="külön riportált eval-halmaz(ok) (pl. az új éles gyűjtés val_new-ja); itt a black recall a fő szám")
    p.add_argument("--out", default="mnv3_squares.pt", help="checkpoint kimenet (.pt)")
    # modell
    p.add_argument("--arch", default="mobilenet_v3_small", choices=list(VARIANTS),
                   help="mért döntés a fejlécben: mobilenet_v3_small (ORT 23 ms / 64 mező, legjobb cross-domain recall)")
    p.add_argument("--img-size", type=int, default=100, help="a pipeline 100 px-t használ; 128 csak mért latencia-árral")
    p.add_argument("--no-pretrained", action="store_true", help="torchvision backbone-ok ImageNet súly nélkül")
    p.add_argument("--dropout", type=float, default=0.10)
    # multi-task
    p.add_argument("--type-loss-weight", type=float, default=0.3, help="0 = típus-fej kikapcsolva a loss-ból")
    p.add_argument("--type-grad-scale", type=float, default=1.0, help="típus-fej gradiense a backbone felé (0 = leválasztva)")
    p.add_argument("--auto-type-grad", action="store_true", default=True)
    p.add_argument("--no-auto-type-grad", dest="auto_type_grad", action="store_false")
    # loss / sampler
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--black-weight", type=float, default=1.0, help="class-weight a black osztályra (a sampler mellett alapból 1.0)")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-balanced-sampler", action="store_true")
    p.add_argument("--weak-aug", action="store_true", help="megvilágítás/árnyék augmentáció KI (ablációhoz)")
    # optimalizálás
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", type=int, default=0, help="csak N train mintával (smoke test)")
    # benchmark
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--bench-archs", nargs="*", default=None, choices=list(VARIANTS))
    p.add_argument("--bench-img-sizes", nargs="*", type=int, default=None)
    p.add_argument("--bench-batch", type=int, default=64)
    p.add_argument("--bench-partial-batch", type=int, default=16)
    p.add_argument("--bench-runs", type=int, default=30)
    p.add_argument("--threads", type=int, default=10, help="ORT intra-op / torch szálak (a célgépen torch 10-et használ)")
    p.add_argument("--bench-dir", default="/tmp/square_bench", help="ide kerülnek az ideiglenes .onnx fájlok")
    p.add_argument("--bench-out", default=None, help="benchmark eredmény JSON")
    args = p.parse_args(argv)
    if args.data_root is None:
        args.data_root = ["."]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.benchmark_only:
        run_benchmark(args)
        return
    train(args)


if __name__ == "__main__":
    main()
