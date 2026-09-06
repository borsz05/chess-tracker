#!/usr/bin/env python3
"""
A klasszifikátor ZAJÁNAK mérése az ÉLES pipeline-úton — a stabilizer küszöbeinek
számszerű alapja (docs/refaktor_prompt.md 10. szakasz: "mérj, ne becsülj").

Két mérés:

1. `frames` — a FEN-gyűjtő teljes kameraképei (data_fen/frames + shots.jsonl):
   detect_board_on_frame -> warpPerspective -> crop_with_context -> modell, a
   címke a FEN-ből (tools/fen_labels, a pipeline raw_to_standard inverzével).
   Kimenet: pontosság / osztályonkénti recall / konfúzió sessionönként,
   a HIBÁS és a HELYES predikciók konfidencia-eloszlása (ebből jön a
   `min_changed_conf`), és a képenkénti hibaszám-eloszlás (ebből jön, hogy
   kell-e még többségi szavazás).

   FIGYELEM: ezek a képek a tanítóhalmaz részei (train_new) vagy a val_new
   split állásai (near-duplicate szivárgással), tehát a számok FELSŐ becslések.

2. `video` — statikus tábláról készült felvétel (tools/record_camera.py vagy
   bármely AVI/MP4): képkockánként teljes klasszifikáció, ebből a
   mezőnkénti VILLÓDZÁS (a módusz-címkétől eltérő frame-ek aránya), a
   képkockánkénti eltérésszám (0 / 1 / >=2 mező), és a mozgás-alapvonal
   (négyzetenkénti |diff| két egymás utáni frame között — ebből jön a
   `motion_diff_threshold`). --fen megadásával pontosság is.

Használat (repo gyökérből):

    python -m tools.measure_pipeline_noise frames
    python -m tools.measure_pipeline_noise frames --sessions sotetben_1080 jo_fenyviszony_1080
    python -m tools.measure_pipeline_noise video felvetel.avi --fen "<FEN>"
    python -m tools.measure_pipeline_noise video felvetel.avi --max-frames 300

A tábladetektálás lassú (~1-2 s/kép), ezért a `frames` mód a detektálási
eredményt cache-eli (--cache, alapból data_fen/.det_cache.pkl); a cache-t
töröld, ha a detektor vagy a cell/inner_pad_ratio változik.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fen_labels import fen_to_raw_labels  # noqa: E402
from vision.app.config import AppConfig  # noqa: E402
from vision.models.occupancy_color_model import OccupancyColorModel  # noqa: E402
from vision.pipeline.batch_classifier import crop_with_context  # noqa: E402
from vision.pipeline.board_detector import DetectionResult, detect_board_on_frame  # noqa: E402
from vision.pipeline.orientation import apply_symmetry, rank_symmetries  # noqa: E402
from vision.pipeline.tracker import compute_square_diffs  # noqa: E402

OCC_NAMES = ("empty", "white", "black")
CONF_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)


# ---------------------------------------------------------------------------
# Segédek
# ---------------------------------------------------------------------------

def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _print_conf_table(title: str, correct: np.ndarray, wrong: np.ndarray) -> None:
    print(f"\n  {title}")
    print(f"    helyes predikció konfidenciája:  n={len(correct):6d}  p1={_pct(correct, 1):.3f}  p5={_pct(correct, 5):.3f}  "
          f"p50={_pct(correct, 50):.3f}  min={correct.min() if len(correct) else float('nan'):.3f}")
    if len(wrong):
        print(f"    HIBÁS predikció konfidenciája:   n={len(wrong):6d}  p50={_pct(wrong, 50):.3f}  p95={_pct(wrong, 95):.3f}  "
              f"max={wrong.max():.3f}")
    else:
        print("    HIBÁS predikció: nincs")
    print("    küszöb  | hibás predikciók a küszöb FELETT | helyes predikciók a küszöb ALATT (elutasítva)")
    for t in CONF_THRESHOLDS:
        w_above = int(np.count_nonzero(wrong >= t)) if len(wrong) else 0
        c_below = int(np.count_nonzero(correct < t))
        print(f"    {t:.2f}    | {w_above:5d} / {len(wrong):5d}                     | {c_below:6d} / {len(correct):6d}  ({100.0 * c_below / max(1, len(correct)):.2f}%)")


def _confusion(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(truth.ravel(), pred.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def _print_confusion(cm: np.ndarray) -> None:
    print("    konfúzió (sor = valós, oszlop = predikció):   " + "  ".join(f"{n:>6s}" for n in OCC_NAMES))
    for i, n in enumerate(OCC_NAMES):
        rec = cm[i].sum()
        print(f"      {n:>6s}: " + "  ".join(f"{cm[i, j]:6d}" for j in range(3)) + f"   recall={cm[i, i] / rec if rec else float('nan'):.4f}")


def _load_model(weights: str | None) -> OccupancyColorModel:
    cfg = AppConfig()
    m = OccupancyColorModel(weights_path=weights or cfg.weights_path, backend=cfg.inference_backend,
                            onnx_path=cfg.onnx_path, num_threads=cfg.inference_threads, allow_int8=cfg.allow_int8)
    print(f"[modell] {m!r}")
    return m


def _classify_all(model: OccupancyColorModel, img_warp: np.ndarray, det: DetectionResult, context: float):
    rois = [crop_with_context(img_warp, det.bbox_warp[r][c], context=context) for r in range(8) for c in range(8)]
    pred = model.predict_rois(rois)
    labels = pred.labels.reshape(8, 8)
    confs = pred.confs.reshape(8, 8)
    types = None if pred.type_probs is None else pred.type_probs.reshape(8, 8, -1)
    return labels, confs, types


def _truth_grids(fen: str, orient: str):
    labs = apply_symmetry(fen_to_raw_labels(fen), orient)
    occ = np.array([[labs[r][c].occ for c in range(8)] for r in range(8)], dtype=np.int32)
    typ = np.array([[labs[r][c].type_idx for c in range(8)] for r in range(8)], dtype=np.int32)
    return occ, typ


# ---------------------------------------------------------------------------
# 1. mód: mentett képek FEN-címkével
# ---------------------------------------------------------------------------

def run_frames(args: argparse.Namespace) -> None:
    cfg = AppConfig()
    model = _load_model(args.weights)
    shots = [json.loads(l) for l in open(args.shots, encoding="utf-8") if l.strip()]
    if args.sessions:
        shots = [s for s in shots if s["session"] in set(args.sessions)]
    print(f"[frames] {len(shots)} kép, {len({s['session'] for s in shots})} session")

    cache: dict[str, DetectionResult] = {}
    if args.cache.exists():
        with open(args.cache, "rb") as f:
            cache = pickle.load(f)
        print(f"[frames] detektálás-cache: {len(cache)} bejegyzés ({args.cache})")

    per_session: dict[str, dict] = defaultdict(lambda: {"cm": np.zeros((3, 3), np.int64), "err_per_frame": [],
                                                        "conf_ok": [], "conf_bad": [], "type_ok": 0, "type_n": 0,
                                                        "det_fail": 0, "n": 0, "cell_err": Counter()})
    all_ok: list[float] = []
    all_bad: list[float] = []
    err_hist = Counter()
    orient_hist = Counter()
    detect_ms: list[float] = []
    t_start = time.perf_counter()
    dirty = False

    for i, s in enumerate(shots):
        path = args.frames_dir / f"{s['base']}.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ! hiányzó kép: {path}")
            continue
        det = cache.get(s["base"])
        if det is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()
            det = detect_board_on_frame(gray, cell=cfg.cell, inner_pad_ratio=cfg.inner_pad_ratio)
            detect_ms.append((time.perf_counter() - t0) * 1000.0)
            cache[s["base"]] = det
            dirty = True
        ps = per_session[s["session"]]
        ps["n"] += 1
        if not det.ok:
            ps["det_fail"] += 1
            continue

        img_warp = cv2.warpPerspective(frame, det.M, cfg.warp_size, flags=cv2.WARP_INVERSE_MAP)
        labels, confs, types = _classify_all(model, img_warp, det, cfg.context)
        occ_t, typ_t = _truth_grids(s["fen"], s.get("orient") or "rot0")

        # A detektor rácsának orientációja NEM garantált: ugyanarra a jelenetre
        # (itt: a mentett JPEG az élő frame helyett) más szimmetriát adhat.
        # Megkeressük, melyik szimmetria illeszkedik, és ahhoz igazítjuk a
        # címkét — a választott szimmetriák eloszlása maga is mérés (mennyire
        # instabil az orientáció újradetektáláskor).
        ranking = rank_symmetries(labels, occ_t)
        best_name, best_n = ranking[0]
        ident_n = dict(ranking)["rot0"]
        if best_name != "rot0" and ident_n - best_n >= args.orient_margin:
            occ_t = apply_symmetry(occ_t, best_name)
            typ_t = apply_symmetry(typ_t, best_name)
            orient_hist[best_name] += 1
            if args.verbose:
                print(f"  {s['base']}: a detektor rácsa {best_name} (eltérés {best_n} vs rot0 {ident_n})")
        else:
            orient_hist["rot0"] += 1

        wrong = labels != occ_t
        for r, c in zip(*np.nonzero(wrong)):
            ps["cell_err"][(int(r), int(c), OCC_NAMES[occ_t[r, c]], OCC_NAMES[labels[r, c]])] += 1
        n_err = int(wrong.sum())
        err_hist[n_err] += 1
        ps["err_per_frame"].append(n_err)
        ps["cm"] += _confusion(occ_t, labels)
        ps["conf_ok"].extend(confs[~wrong].tolist())
        ps["conf_bad"].extend(confs[wrong].tolist())
        all_ok.extend(confs[~wrong].tolist())
        all_bad.extend(confs[wrong].tolist())
        if types is not None:
            occupied = occ_t != 0
            ps["type_n"] += int(occupied.sum())
            ps["type_ok"] += int((types.argmax(axis=2)[occupied] == typ_t[occupied]).sum())

        if args.verbose and n_err:
            cells = ", ".join(f"r{r}c{c}:{OCC_NAMES[occ_t[r, c]]}->{OCC_NAMES[labels[r, c]]}({confs[r, c]:.2f})"
                              for r, c in zip(*np.nonzero(wrong)))
            print(f"  {s['base']}: {n_err} hiba  {cells}")
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(shots)} ({time.perf_counter() - t_start:.0f} s)")

    if dirty:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump(cache, f)

    # ---- riport
    print("\n" + "=" * 96)
    print("  KLASSZIFIKÁTOR-ZAJ A MENTETT KÉPEKEN (éles kivágási út, FEN-címke)")
    print("=" * 96)
    if detect_ms:
        print(f"  board_detect: n={len(detect_ms)}  p50={_pct(np.array(detect_ms), 50):.0f} ms  max={max(detect_ms):.0f} ms")
    total_cm = np.zeros((3, 3), np.int64)
    for name in sorted(per_session):
        ps = per_session[name]
        cm = ps["cm"]
        total_cm += cm
        n_sq = cm.sum()
        acc = np.trace(cm) / n_sq if n_sq else float("nan")
        errs = np.array(ps["err_per_frame"])
        print(f"\n  session {name}: {ps['n']} kép (detektálás sikertelen: {ps['det_fail']}), mező-pontosság {acc:.4f}, "
              f"black recall {cm[2, 2] / cm[2].sum() if cm[2].sum() else float('nan'):.4f}, "
              f"képenként >=1 hiba: {int((errs >= 1).sum())}, >=2 hiba: {int((errs >= 2).sum())}"
              + (f", típus-fej pontosság (foglalt mezőkön): {ps['type_ok'] / ps['type_n']:.4f}" if ps["type_n"] else ""))
        _print_confusion(cm)
        persistent = [(k, v) for k, v in ps["cell_err"].items() if v >= 3]
        if persistent:
            print("    ISMÉTLŐDŐ hibás mezők (>=3 képen ugyanaz — valószínűleg a FEN/felrakás hibája, nem a modellé):")
            for (r, c, t, pr), v in sorted(persistent, key=lambda kv: -kv[1]):
                print(f"      r{r}c{c}: valós {t} -> predikció {pr}  ({v} képen)")

    print("\n  ÖSSZESEN:")
    _print_confusion(total_cm)
    n_frames = sum(err_hist.values())
    print(f"\n  a detektor rácsának orientációja a mentett képen (a gyűjtéskori rot0-hoz képest): "
          + "  ".join(f"{k}:{v}" for k, v in sorted(orient_hist.items(), key=lambda kv: -kv[1])))
    print(f"\n  képenkénti hibaszám ({n_frames} kép): " + "  ".join(f"{k}:{v}" for k, v in sorted(err_hist.items())))
    p1 = sum(v for k, v in err_hist.items() if k >= 1) / max(1, n_frames)
    p2 = sum(v for k, v in err_hist.items() if k >= 2) / max(1, n_frames)
    print(f"    P(>=1 hibás mező / kép) = {p1:.4f}    P(>=2 hibás mező / kép) = {p2:.4f}")
    _print_conf_table("konfidencia (győztes osztály valószínűsége), minden session:", np.array(all_ok), np.array(all_bad))


# ---------------------------------------------------------------------------
# 2. mód: statikus felvétel -> villódzás + mozgás-alapvonal
# ---------------------------------------------------------------------------

def run_video(args: argparse.Namespace) -> None:
    cfg = AppConfig()
    model = _load_model(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"nem nyitható: {args.video}")

    det: DetectionResult | None = None
    prev_warp = None
    label_seq: list[np.ndarray] = []
    conf_seq: list[np.ndarray] = []
    diff_seq: list[np.ndarray] = []     # (64,) négyzetenkénti mean |diff| az előző frame-hez
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        if det is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()
            det = detect_board_on_frame(gray, cell=cfg.cell, inner_pad_ratio=cfg.inner_pad_ratio)
            print(f"[video] board_detect: {(time.perf_counter() - t0) * 1000:.0f} ms, ok={det.ok}")
            if not det.ok:
                det = None
                n += 1
                continue
        img_warp = cv2.warpPerspective(frame, det.M, cfg.warp_size, flags=cv2.WARP_INVERSE_MAP)
        labels, confs, _ = _classify_all(model, img_warp, det, cfg.context)
        label_seq.append(labels)
        conf_seq.append(confs)
        if prev_warp is not None:
            d = np.zeros(64, np.float32)
            for diff, r, c in compute_square_diffs(prev_warp, img_warp, det.bbox_warp):
                d[r * 8 + c] = diff
            diff_seq.append(d)
        prev_warp = img_warp
        n += 1
    cap.release()
    if not label_seq:
        raise SystemExit("nincs feldolgozott frame (detektálás sikertelen?)")

    L = np.stack(label_seq)              # (T, 8, 8)
    C = np.stack(conf_seq)               # (T, 8, 8)
    T = L.shape[0]
    mode = np.zeros((8, 8), np.int32)
    for r in range(8):
        for c in range(8):
            mode[r, c] = np.bincount(L[:, r, c], minlength=3).argmax()
    dev = L != mode[None]                # módusztól eltérő frame-ek
    flicker_per_sq = dev.mean(axis=0)    # (8,8)
    errs_per_frame = dev.reshape(T, -1).sum(axis=1)
    transitions = (L[1:] != L[:-1]).reshape(T - 1, -1).sum(axis=1) if T > 1 else np.zeros(0, int)

    print("\n" + "=" * 96)
    print(f"  STATIKUS FELVÉTEL: {T} frame, {args.video}")
    print("=" * 96)
    print(f"  módusz-rács foglalt mezők: {int((mode != 0).sum())} (white {int((mode == 1).sum())}, black {int((mode == 2).sum())})")
    print(f"  villódzó mezők (módusztól >0 frame-ben eltér): {int((flicker_per_sq > 0).sum())} / 64; "
          f"legrosszabb mező eltérési aránya: {flicker_per_sq.max():.4f}")
    worst = sorted(((flicker_per_sq[r, c], r, c) for r in range(8) for c in range(8)), reverse=True)[:5]
    print("  legrosszabb mezők (arány, r, c, módusz):  " + "  ".join(f"{f:.3f}@r{r}c{c}:{OCC_NAMES[mode[r, c]]}" for f, r, c in worst))
    print(f"  képkockánként a módusztól eltérő mezők: 0: {int((errs_per_frame == 0).sum())}  1: {int((errs_per_frame == 1).sum())}  "
          f">=2: {int((errs_per_frame >= 2).sum())}   -> P(>=1)={float((errs_per_frame >= 1).mean()):.4f}  "
          f"P(>=2)={float((errs_per_frame >= 2).mean()):.4f}")
    if len(transitions):
        print(f"  két egymás utáni frame között címkét váltó mezők: 0: {int((transitions == 0).sum())}  1: {int((transitions == 1).sum())}  "
              f">=2: {int((transitions >= 2).sum())}   (P(>=2) = {float((transitions >= 2).mean()):.4f})")
    # leghosszabb eltérés-sorozat mezőnként (hány egymás utáni frame-ben tér el a módusztól)
    longest = 0
    for r in range(8):
        for c in range(8):
            run = best = 0
            for v in dev[:, r, c]:
                run = run + 1 if v else 0
                best = max(best, run)
            longest = max(longest, best)
    print(f"  leghosszabb egybefüggő eltérés egy mezőn: {longest} frame")

    if args.fen:
        occ_t, _ = _truth_grids(args.fen, args.orient)
        wrong = L != occ_t[None]
        cm = _confusion(np.repeat(occ_t[None], T, axis=0), L)
        print(f"\n  FEN-hez képest: mező-pontosság {np.trace(cm) / cm.sum():.4f}, módusz-rács hibái: {int((mode != occ_t).sum())}")
        _print_confusion(cm)
        _print_conf_table("konfidencia (FEN a valóság):", C[~wrong], C[wrong])
    else:
        _print_conf_table("konfidencia (a módusz-rács a 'valóság'):", C[~dev], C[dev])

    if diff_seq:
        D = np.stack(diff_seq)           # (T-1, 64)
        print("\n  MOZGÁS-ALAPVONAL statikus táblán (négyzetenkénti mean|diff| két frame között, 0-255):")
        print(f"    p50={_pct(D, 50):.2f}  p95={_pct(D, 95):.2f}  p99={_pct(D, 99):.2f}  p99.9={_pct(D, 99.9):.2f}  max={D.max():.2f}")
        mx = D.max(axis=1)
        print(f"    képkockánkénti MAX a 64 mezőn: p50={_pct(mx, 50):.2f}  p95={_pct(mx, 95):.2f}  max={mx.max():.2f}")
        print(f"    (AppConfig.partial_diff_threshold = {cfg.partial_diff_threshold}; a mozgás-kapu küszöbe legyen a p99.9 felett)")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    pf = sub.add_parser("frames", help="mentett kameraképek FEN-címkével")
    pf.add_argument("--frames-dir", type=Path, default=ROOT / "data_fen" / "frames")
    pf.add_argument("--shots", type=Path, default=ROOT / "data_fen" / "shots.jsonl")
    pf.add_argument("--cache", type=Path, default=ROOT / "data_fen" / ".det_cache.pkl")
    pf.add_argument("--sessions", nargs="*", default=None)
    pf.add_argument("--weights", default=None)
    pf.add_argument("--verbose", action="store_true", help="minden hibás mező kiírása")
    pf.add_argument("--orient-margin", type=int, default=12,
                    help="ennyivel kevesebb eltérés kell egy nem-identitás szimmetriához, hogy ahhoz igazítsuk a címkét")

    pv = sub.add_parser("video", help="statikus tábláról készült felvétel")
    pv.add_argument("video", type=Path)
    pv.add_argument("--fen", default=None)
    pv.add_argument("--orient", default="rot0")
    pv.add_argument("--max-frames", type=int, default=0)
    pv.add_argument("--weights", default=None)

    args = p.parse_args()
    if args.mode == "frames":
        run_frames(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()
