#!/usr/bin/env python3
"""
PyTorch checkpoint -> ONNX export (dinamikus batch) + paritás-ellenőrzés + INT8 kiértékelés.

Működik a legacy egyfejes checkpointtal (resnet18_best_topdown.pt) ÉS a
vision/models/square_net.MultiTaskSquareNet checkpointjával (két kimenet:
color_logits, type_logits). A kimeneti .onnx mellé `<név>.onnx.json` sidecar
kerül a metaadatokkal (class_names, img_size, normalize, kimenetek), ezt olvassa
az OccupancyColorModel ONNX módban.

Használat (repo gyökérből):

    python -m tools.export_onnx --weights vision/models/weights/resnet18_best_topdown.pt
        -> vision/models/weights/resnet18_best_topdown.onnx (+ .onnx.json)

    # INT8 kiértékelés (dinamikus és statikus kvantálás), black recall + latencia:
    python -m tools.export_onnx --weights <ckpt.pt> --int8 --eval-dir <dataset>/val --calib-dir <dataset>/train_new

    # paritás valós ROI-kon (alapból tools/live_dump/*/*.jpg, ha van):
    python -m tools.export_onnx --weights <ckpt.pt> --roi-dir tools/live_dump

A szkript minden lépés után számokat ír: max |Δlogit|, címke-egyezés, p50
latencia batch 64 / 16 (torch vs ORT fp32 vs ORT INT8), és eval-dir esetén
osztályonkénti recall fp32 vs INT8. Az INT8 döntés a `black` recall-on áll:
ha az INT8 black recall a fp32 alá esik (> --int8-max-drop pontnál többel),
a szkript NEM ajánlja az INT8 fájlt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.models.occupancy_color_model import (  # noqa: E402
    OccupancyColorModel,
    build_torch_model_from_checkpoint,
    checkpoint_metadata,
    preprocess_rois,
    sidecar_path,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
COLOR_DIRS = ("black", "empty", "white")


class _ColorOnly(torch.nn.Module):
    """Legacy modell: egy kimenet, 'color_logits' néven."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(weights: Path, out: Path, opset: int = 17) -> tuple[Path, dict]:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    meta = checkpoint_metadata(ckpt)
    model, has_type = build_torch_model_from_checkpoint(ckpt)
    model = model.eval().cpu()
    size = meta["img_size"]
    dummy = torch.randn(2, 3, size, size)
    outputs = ["color_logits", "type_logits"] if has_type else ["color_logits"]
    dyn = {"input": {0: "batch"}, **{o: {0: "batch"} for o in outputs}}
    kwargs = dict(input_names=["input"], output_names=outputs, dynamic_axes=dyn, opset_version=opset)
    export_model = model if has_type else _ColorOnly(model)
    try:
        torch.onnx.export(export_model, (dummy,), str(out), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(export_model, (dummy,), str(out), **kwargs)

    import onnx
    onnx.checker.check_model(str(out))

    meta.update({
        "outputs": outputs, "has_type_head": has_type, "opset": opset,
        "source_checkpoint": str(weights), "quantized": None,
        "onnx_size_mb": round(out.stat().st_size / 1e6, 2),
    })
    sidecar_path(out).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ONNX export OK: {out} ({meta['onnx_size_mb']} MB), kimenetek={outputs}, img_size={size}, class_names={meta['class_names']}")
    return out, meta


def write_sidecar_copy(src_onnx: Path, dst_onnx: Path, quantized: str) -> None:
    meta = json.loads(sidecar_path(src_onnx).read_text(encoding="utf-8"))
    meta["quantized"] = quantized
    meta["onnx_size_mb"] = round(dst_onnx.stat().st_size / 1e6, 2)
    sidecar_path(dst_onnx).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Adat
# ---------------------------------------------------------------------------

def load_rois(roi_dir: Path | None, limit: int) -> list[np.ndarray]:
    if roi_dir is None or not roi_dir.exists():
        return []
    files = sorted(p for p in roi_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)[:limit]
    return [cv2.imread(str(f)) for f in files]


def load_labelled(eval_dir: Path, per_class_limit: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """<dir>/{black,empty,white}/* (vagy <dir>/val_new/...) -> (rois, pipeline labels)."""
    base = eval_dir / "val_new" if (eval_dir / "val_new").is_dir() else eval_dir
    rois, labels = [], []
    for name, lab in (("empty", 0), ("white", 1), ("black", 2)):
        d = base / name
        if not d.is_dir():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if per_class_limit:
            files = files[:per_class_limit]
        for f in files:
            im = cv2.imread(str(f))
            if im is not None:
                rois.append(im); labels.append(lab)
    return rois, np.asarray(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Mérések
# ---------------------------------------------------------------------------

def p50(fn, n_warm=3, n=20) -> float:
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def parity(a: OccupancyColorModel, b: OccupancyColorModel, x: np.ndarray, name: str) -> dict:
    la, ta = a.forward_logits(x)
    lb, tb = b.forward_logits(x)
    pa, pb = a.predict_batch(x), b.predict_batch(x)
    d = float(np.abs(la - lb).max())
    agree = float((pa.labels == pb.labels).mean())
    conf_d = float(np.abs(pa.confs - pb.confs).max())
    t_d = float(np.abs(ta - tb).max()) if ta is not None and tb is not None else None
    print(f"  paritás [{name}] n={len(x)}: max|Δcolor_logit|={d:.2e}  max|Δconf|={conf_d:.2e}  címke-egyezés={agree*100:.2f}%"
          + (f"  max|Δtype_logit|={t_d:.2e}" if t_d is not None else ""))
    return {"n": int(len(x)), "max_abs_logit_diff": d, "max_abs_conf_diff": conf_d, "label_agreement": agree, "max_abs_type_diff": t_d}


def recall_report(model: OccupancyColorModel, x: np.ndarray, y: np.ndarray, name: str) -> dict:
    pred = model.predict_batch(x).labels
    out = {}
    for lab, cname in ((0, "empty"), (1, "white"), (2, "black")):
        m = y == lab
        out[cname] = float((pred[m] == lab).mean()) if m.any() else float("nan")
    out["acc"] = float((pred == y).mean())
    out["macro"] = float(np.nanmean([out["empty"], out["white"], out["black"]]))
    print(f"  recall [{name}] n={len(y)}: empty={out['empty']*100:.2f}% white={out['white']*100:.2f}% "
          f"BLACK={out['black']*100:.2f}% | acc={out['acc']*100:.2f}% macro={out['macro']*100:.2f}%")
    return out


# ---------------------------------------------------------------------------
# INT8
# ---------------------------------------------------------------------------

class _CalibReader:
    def __init__(self, x: np.ndarray, batch: int = 32):
        self.x, self.batch, self.i = x, batch, 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        b = self.x[self.i:self.i + self.batch]
        self.i += self.batch
        return {"input": np.ascontiguousarray(b, dtype=np.float32)}


def quantize(fp32: Path, mode: str, calib_x: np.ndarray | None) -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32.with_name(fp32.stem + ".preproc.onnx")
    quant_pre_process(str(fp32), str(pre))
    out = fp32.with_name(f"{fp32.stem}.int8_{mode}.onnx")
    if mode == "dynamic":
        quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8)
    else:
        if calib_x is None or len(calib_x) == 0:
            raise SystemExit("statikus kvantáláshoz --calib-dir kell")
        quantize_static(str(pre), str(out), _CalibReader(calib_x), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink(missing_ok=True)
    write_sidecar_copy(fp32, out, quantized=mode)
    print(f"INT8 ({mode}) mentve: {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="alap: <weights>.onnx (ugyanott)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--roi-dir", type=Path, default=ROOT / "tools" / "live_dump", help="valós ROI-k a paritáshoz")
    p.add_argument("--roi-limit", type=int, default=256)
    p.add_argument("--eval-dir", type=Path, default=None, help="címkézett halmaz ({black,empty,white}/ vagy val_new/) a recall-hoz")
    p.add_argument("--int8", action="store_true", help="dinamikus + statikus INT8 változat készítése és kiértékelése")
    p.add_argument("--calib-dir", type=Path, default=None, help="kalibrációs képek a statikus INT8-hoz (train_new/)")
    p.add_argument("--calib-per-class", type=int, default=100)
    p.add_argument("--int8-max-drop", type=float, default=0.5, help="megengedett black-recall esés százalékPONTBAN")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--report", type=Path, default=None, help="JSON riport útvonal (alap: <out>.report.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.weights.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"weights": str(args.weights), "onnx": str(out)}

    onnx_path, meta = export(args.weights, out, args.opset)
    size = meta["img_size"]

    torch_m = OccupancyColorModel(args.weights, device="cpu", backend="torch", num_threads=args.threads)
    # allow_spinning=True: itt a modell CSUCS-atbocsatasat merjuk (fp32 vs INT8),
    # ahhoz a porgo szalpool a helyes beallitas. Elesben forditva van, lasd
    # vision/models/occupancy_color_model.py _init_onnx.
    onnx_m = OccupancyColorModel(args.weights, backend="onnx", onnx_path=onnx_path, num_threads=args.threads, allow_int8=False, allow_spinning=True)
    print(f"torch: {torch_m!r}\nonnx : {onnx_m!r}")

    # --- paritás -------------------------------------------------------------
    print("\n== Paritás (.pt vs .onnx) ==")
    rng = np.random.default_rng(0)
    x_rand = rng.standard_normal((64, 3, size, size), dtype=np.float32)
    report["parity_random"] = parity(torch_m, onnx_m, x_rand, "random 64")
    rois = load_rois(args.roi_dir, args.roi_limit)
    x_real = None
    if rois:
        x_real = onnx_m.preprocess(rois)
        report["parity_real_rois"] = parity(torch_m, onnx_m, x_real, f"valós ROI {args.roi_dir}")
        # a régi mezőnkénti torch-preprocess és a vektorizált ugyanazt adja?
        ref = torch.stack([_legacy_preprocess(r, size, torch_m.norm_mean_cpu, torch_m.norm_std_cpu) for r in rois]).numpy()
        pd = float(np.abs(ref - x_real).max())
        print(f"  preprocess: vektorizált vs régi mezőnkénti út max|Δ|={pd:.2e}")
        report["preprocess_max_abs_diff"] = pd

    # --- latencia -------------------------------------------------------------
    if not args.no_bench:
        print(f"\n== Latencia (p50 ms, {args.threads} szál) ==")
        lat = {}
        for b in (64, 16):
            xb = x_rand[:b]
            lat[f"torch_b{b}"] = p50(lambda: torch_m.forward_logits(xb))
            lat[f"onnx_b{b}"] = p50(lambda: onnx_m.forward_logits(xb))
            print(f"  batch {b:2d}: torch={lat[f'torch_b{b}']:7.1f}  onnx fp32={lat[f'onnx_b{b}']:7.1f}")
        if rois:
            r64 = (rois * (64 // len(rois) + 1))[:64]
            lat["preprocess_64_vectorized"] = p50(lambda: onnx_m.preprocess(r64))
            lat["preprocess_64_legacy_loop"] = p50(lambda: torch.stack([_legacy_preprocess(r, size, torch_m.norm_mean_cpu, torch_m.norm_std_cpu) for r in r64]))
            print(f"  preprocess 64 ROI: vektorizált={lat['preprocess_64_vectorized']:.2f}  régi ciklus={lat['preprocess_64_legacy_loop']:.2f}")
        report["latency_ms"] = lat

    # --- eval + INT8 --------------------------------------------------------------
    x_eval = y_eval = None
    if args.eval_dir:
        print(f"\n== Recall ({args.eval_dir}) ==")
        rois_e, y_eval = load_labelled(args.eval_dir)
        x_eval = onnx_m.preprocess(rois_e)
        report["recall_torch"] = recall_report(torch_m, x_eval, y_eval, "torch fp32")
        report["recall_onnx_fp32"] = recall_report(onnx_m, x_eval, y_eval, "onnx fp32")

    if args.int8:
        print("\n== INT8 kvantálás ==")
        calib_x = None
        if args.calib_dir:
            rois_c, _ = load_labelled(args.calib_dir, per_class_limit=args.calib_per_class)
            calib_x = onnx_m.preprocess(rois_c)
        report["int8"] = {}
        for mode in ("dynamic", "static"):
            if mode == "static" and calib_x is None:
                print("  statikus: kihagyva (nincs --calib-dir)"); continue
            try:
                q_path = quantize(onnx_path, mode, calib_x)
            except Exception as e:
                print(f"  {mode}: HIBA {type(e).__name__}: {e}"); continue
            q_m = OccupancyColorModel(args.weights, backend="onnx", onnx_path=q_path, num_threads=args.threads, allow_int8=False, allow_spinning=True)
            entry = {"path": str(q_path)}
            entry["parity_random"] = parity(onnx_m, q_m, x_rand, f"fp32 vs int8-{mode}, random")
            if x_real is not None:
                entry["parity_real"] = parity(onnx_m, q_m, x_real, f"fp32 vs int8-{mode}, valós ROI")
            if not args.no_bench:
                entry["latency_ms"] = {f"b{b}": p50(lambda xb=x_rand[:b]: q_m.forward_logits(xb)) for b in (64, 16)}
                print(f"  latencia int8-{mode}: b64={entry['latency_ms']['b64']:.1f} ms  b16={entry['latency_ms']['b16']:.1f} ms")
            if x_eval is not None:
                entry["recall"] = recall_report(q_m, x_eval, y_eval, f"onnx int8-{mode}")
                drop = (report["recall_onnx_fp32"]["black"] - entry["recall"]["black"]) * 100
                entry["black_recall_drop_pts"] = drop
                entry["recommended"] = drop <= args.int8_max_drop and entry["recall"]["macro"] >= report["recall_onnx_fp32"]["macro"] - args.int8_max_drop / 100
                print(f"  -> black recall változás: {-drop:+.2f} pont; {'AJÁNLOTT' if entry['recommended'] else 'NEM ajánlott'} "
                      f"(küszöb {args.int8_max_drop} pont)")
            report["int8"][mode] = entry

    # Az ajánlott INT8 fájl bejegyzése a fp32 sidecar-ba -> OccupancyColorModel
    # auto módban ezt tölti (AppConfig.allow_int8=False kikapcsolja).
    sc = sidecar_path(onnx_path)
    meta_fp32 = json.loads(sc.read_text(encoding="utf-8"))
    rec = [e for e in report.get("int8", {}).values() if e.get("recommended")]
    if rec:
        best = min(rec, key=lambda e: e.get("latency_ms", {}).get("b64", 1e9))
        meta_fp32["recommended_int8"] = Path(best["path"]).name
        print(f"Ajánlott INT8: {best['path']} (a sidecar 'recommended_int8' mezője; auto backend ezt tölti)")
    else:
        meta_fp32.pop("recommended_int8", None)
    sc.write_text(json.dumps(meta_fp32, indent=2, ensure_ascii=False), encoding="utf-8")

    rp = args.report or Path(str(out) + ".report.json")
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nRiport: {rp}")
    print("Használat a pipeline-ban: AppConfig.inference_backend='auto'/'onnx' automatikusan a "
          f"{onnx_path.name} fájlt tölti; INT8-hoz AppConfig.onnx_path a kvantált fájlra mutasson.")


def _legacy_preprocess(roi: np.ndarray, size: int, mean_cpu, std_cpu):
    """A régi batch_classifier.preprocess_roi_for_batch másolata — referenciaként."""
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    h, w = roi.shape[:2]
    interp = cv2.INTER_CUBIC if (w < size or h < size) else cv2.INTER_AREA
    roi = cv2.resize(roi, (size, size), interpolation=interp)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(roi)).permute(2, 0, 1).float().div_(255.0)
    return x.sub_(mean_cpu).div_(std_cpu)


if __name__ == "__main__":
    main()
