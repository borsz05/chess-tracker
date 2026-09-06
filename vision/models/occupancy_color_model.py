"""
Sakkmező-osztályozó inferencia-wrapper — ONNX Runtime (alapértelmezett) vagy PyTorch (fallback).

    empty -> 0 | white -> 1 | black -> 2      (pipeline foglaltság-kódolás)

Két checkpoint-generációt támogat:
  - legacy egyfejes torchvision háló (resnet18 / efficientnet_b0), `variant` kulcs
  - vision/models/square_net.MultiTaskSquareNet (`model_class` kulcs): 3-osztályos
    szín-fej + 7-osztályos bábutípus-fej. A típus-kimenet a `predict_rois()`
    eredményében elérhető (`type_probs`), de a pipeline EGYELŐRE csak
    promóciónál használhatja — a 3-osztályos út érintetlen.

Backend-választás (`backend` paraméter / AppConfig.inference_backend):
  "auto"  : ha a .pt mellett van .onnx (vagy onnx_path meg van adva) és az
            onnxruntime importálható -> ONNX; különben PyTorch, figyelmeztetéssel
  "onnx"  : kötelezően ONNX (hiba, ha nincs fájl)
  "torch" : kötelezően PyTorch

ONNX fájl előállítása: `python -m tools.export_onnx --weights <ckpt.pt>` — ez a
.onnx mellé egy `<név>.onnx.json` sidecar-t ír a metaadatokkal (class_names,
img_size, normalize, kimenetek), így ONNX módban a 134 MB-os .pt-t be sem kell
olvasni.

A preprocess (N crop -> resize -> BGR->RGB -> normalizálás) VEKTORIZÁLT: a
resize mezőnként fut (eltérő ROI-méretek), minden más egy (N,S,S,3) tömbön.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PIPELINE_LABEL = {"empty": 0, "white": 1, "black": 2}
DEFAULT_CLASS_NAMES = ["empty", "white", "black"]


# ---------------------------------------------------------------------------
# Torch modell felépítése checkpointból (legacy + multi-task)
# ---------------------------------------------------------------------------

def _build_legacy_model(variant: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    if variant == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_torch_model_from_checkpoint(ckpt: dict[str, Any]):
    """
    Checkpoint dict -> eval módú torch modul.
    Visszaad: (model, has_type_head). A multi-task modell forwardja
    (color_logits, type_logits) tuple-t ad, a legacy egy tenzort.
    """
    if ckpt.get("model_class") == "MultiTaskSquareNet":
        from vision.models.square_net import build_from_checkpoint
        return build_from_checkpoint(ckpt), True
    model = _build_legacy_model(ckpt.get("variant", "resnet18"), len(ckpt.get("class_names", DEFAULT_CLASS_NAMES)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, False


def checkpoint_metadata(ckpt: dict[str, Any], img_size_default: int = 100) -> dict[str, Any]:
    """A wrapper által használt metaadatok kiemelése (ugyanez kerül az ONNX sidecar-ba)."""
    norm = ckpt.get("normalize")
    if isinstance(norm, dict) and "mean" in norm and "std" in norm:
        mean, std = list(norm["mean"]), list(norm["std"])
    else:
        mean = list(ckpt.get("normalize_mean", [0.5, 0.5, 0.5]))
        std = list(ckpt.get("normalize_std", [0.25, 0.25, 0.25]))
    return {
        "variant": ckpt.get("variant", "resnet18"),
        "model_class": ckpt.get("model_class", "legacy"),
        "class_names": list(ckpt.get("class_names", DEFAULT_CLASS_NAMES)),
        "type_class_names": list(ckpt.get("type_class_names", [])),
        "img_size": int(ckpt.get("img_size", img_size_default)),
        "normalize": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
    }


def sidecar_path(onnx_path: str | Path) -> Path:
    return Path(str(onnx_path) + ".json")


# ---------------------------------------------------------------------------
# Vektorizált preprocess
# ---------------------------------------------------------------------------

def preprocess_rois(rois: list[np.ndarray], img_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    N darab BGR ROI -> (N, 3, S, S) float32, RGB, normalizált.
    Ugyanaz az eredmény, mint a régi mezőnkénti torch-út (resize INTER_AREA ha
    kicsinyítünk, INTER_CUBIC ha nagyítunk; /255; (x-mean)/std).

    Mezőnként csak az elkerülhetetlen resize + BGR->RGB + CHW-be írás fut
    (uint8, cv2); a normalizálás egyetlen fuzionált affin lépés a teljes
    tömbön: x*(1/(255*std)) + (-mean/std). Mért (64 valós ROI, p50, célgép):
    7.3 ms vs. a régi mezőnkénti torch-ciklus 14.4 ms (élő ORT session mellett
    6.9 vs 11.0 ms). A naiv numpy [..., ::-1] + transpose másolás 14 ms volt.
    """
    n = len(rois)
    out = np.empty((n, 3, img_size, img_size), dtype=np.float32)
    if n == 0:
        return out
    for i, roi in enumerate(rois):
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        h, w = roi.shape[:2]
        if h != img_size or w != img_size:
            interp = cv2.INTER_CUBIC if (w < img_size or h < img_size) else cv2.INTER_AREA
            roi = cv2.resize(roi, (img_size, img_size), interpolation=interp)
        out[i] = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    scale = (1.0 / (255.0 * std.astype(np.float32))).reshape(1, 3, 1, 1)
    shift = (-mean.astype(np.float32) / std.astype(np.float32)).reshape(1, 3, 1, 1)
    out *= scale
    out += shift
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class Prediction:
    labels: np.ndarray                 # (N,) int32, pipeline-kódolás (0/1/2)
    confs: np.ndarray                  # (N,) float32, a győztes szín-osztály valószínűsége
    color_probs: np.ndarray            # (N, n_color) float32, class_names sorrendben
    type_probs: np.ndarray | None      # (N, n_type) float32 vagy None (legacy modell)


# ---------------------------------------------------------------------------
# A wrapper
# ---------------------------------------------------------------------------

class OccupancyColorModel:
    """
    3-class square classifier: empty -> 0 | white -> 1 | black -> 2

    backend: "auto" | "onnx" | "torch"
    """

    def __init__(
        self,
        weights_path: str | Path | None,
        device=None,
        img_size: int = 100,
        *,
        backend: str = "auto",
        onnx_path: str | Path | None = None,
        num_threads: int | None = None,
        allow_int8: bool = True,
    ):
        self.weights_path = None if weights_path is None else Path(weights_path)
        self.onnx_path = Path(onnx_path) if onnx_path else (
            self.weights_path.with_suffix(".onnx") if self.weights_path is not None else None
        )
        # Ha a fp32 .onnx sidecar-ja INT8 változatot ajánl (tools/export_onnx.py
        # --int8: a black recall nem romlott), auto módban azt töltjük.
        if onnx_path is None and allow_int8 and self.onnx_path is not None and self.onnx_path.exists():
            self.onnx_path = self._follow_int8_recommendation(self.onnx_path)
        self.num_threads = num_threads
        self.model = None            # torch modul (csak torch backendnél)
        self.session = None          # onnxruntime.InferenceSession (csak onnx backendnél)
        self.device = None

        if backend not in ("auto", "onnx", "torch"):
            raise ValueError(f"ismeretlen backend: {backend!r}")

        use_onnx = False
        if backend in ("auto", "onnx"):
            ok_file = self.onnx_path is not None and self.onnx_path.exists()
            try:
                import onnxruntime  # noqa: F401
                ok_ort = True
            except ImportError:
                ok_ort = False
            use_onnx = ok_file and ok_ort
            if backend == "onnx" and not use_onnx:
                raise FileNotFoundError(
                    f"ONNX backend kérve, de {'nincs onnxruntime' if not ok_ort else f'nincs ONNX fájl: {self.onnx_path}'} "
                    "(készítsd el: python -m tools.export_onnx --weights <ckpt.pt>)"
                )
            if backend == "auto" and not use_onnx:
                why = "nincs onnxruntime" if not ok_ort else f"nincs ONNX fájl: {self.onnx_path}"
                print(f"[OccupancyColorModel] {why} -> PyTorch fallback "
                      "(gyorsítás: python -m tools.export_onnx --weights <ckpt.pt>)", file=sys.stderr)

        if use_onnx:
            self._init_onnx(img_size)
        else:
            self._init_torch(device, img_size)

        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.idx_to_label = np.zeros(len(self.class_names), dtype=np.int32)
        for idx, name in self.idx_to_class.items():
            self.idx_to_label[idx] = PIPELINE_LABEL.get(name, 0)
        self.norm_mean = np.asarray(self.normalize["mean"], dtype=np.float32)
        self.norm_std = np.asarray(self.normalize["std"], dtype=np.float32)

    # -- init -----------------------------------------------------------------

    @staticmethod
    def _follow_int8_recommendation(fp32_onnx: Path) -> Path:
        sc = sidecar_path(fp32_onnx)
        if not sc.exists():
            return fp32_onnx
        try:
            rec = json.loads(sc.read_text(encoding="utf-8")).get("recommended_int8")
        except (OSError, ValueError):
            return fp32_onnx
        if rec:
            cand = Path(rec)
            if not cand.is_absolute():
                cand = fp32_onnx.parent / cand
            if cand.exists() and sidecar_path(cand).exists():
                return cand
        return fp32_onnx

    def _apply_meta(self, meta: dict[str, Any]) -> None:
        self.variant = meta["variant"]
        self.model_class = meta.get("model_class", "legacy")
        self.class_names = list(meta["class_names"])
        self.type_class_names = list(meta.get("type_class_names", []))
        self.img_size = int(meta["img_size"])
        self.normalize = meta["normalize"]

    def _init_onnx(self, img_size_default: int) -> None:
        import onnxruntime as ort

        self.backend = "onnx"
        sc = sidecar_path(self.onnx_path)
        if sc.exists():
            meta = json.loads(sc.read_text(encoding="utf-8"))
        else:
            if self.weights_path is None or not self.weights_path.exists():
                raise FileNotFoundError(f"nincs sidecar ({sc}) és nincs .pt a metaadatokhoz")
            import torch
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            meta = checkpoint_metadata(ckpt, img_size_default)
        self._apply_meta(meta)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.inter_op_num_threads = 1
        if self.num_threads:
            so.intra_op_num_threads = int(self.num_threads)
        self.session = ort.InferenceSession(str(self.onnx_path), so, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        self.has_type_head = "type_logits" in self._output_names
        self.quantized = meta.get("quantized")
        # kompatibilitás: a régi kód torch-tenzoros normalizálót várt
        self.norm_mean_cpu = self.norm_std_cpu = None

    def _init_torch(self, device, img_size_default: int) -> None:
        import torch

        self.backend = "torch"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self._apply_meta(checkpoint_metadata(ckpt, img_size_default))
        self.model, self.has_type_head = build_torch_model_from_checkpoint(ckpt)
        self.quantized = None
        self.model.eval().to(self.device)
        if self.num_threads and self.device.type == "cpu":
            torch.set_num_threads(int(self.num_threads))
        mean, std = self.normalize["mean"], self.normalize["std"]
        self.norm_mean_cpu = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.norm_std_cpu = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    # -- inferencia -----------------------------------------------------------

    def preprocess(self, rois: list[np.ndarray]) -> np.ndarray:
        return preprocess_rois(rois, self.img_size, self.norm_mean, self.norm_std)

    def forward_logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(N,3,S,S) float32 -> (color_logits, type_logits | None), numpy."""
        if x.shape[0] == 0:
            return np.zeros((0, len(self.class_names)), np.float32), None
        if self.backend == "onnx":
            outs = self.session.run(None, {self._input_name: np.ascontiguousarray(x, dtype=np.float32)})
            by_name = dict(zip(self._output_names, outs))
            color = by_name.get("color_logits", outs[0])
            return np.asarray(color, np.float32), (np.asarray(by_name["type_logits"], np.float32) if self.has_type_head else None)
        import torch
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x).to(self.device, non_blocking=True))
        if isinstance(out, (tuple, list)):
            return out[0].float().cpu().numpy(), out[1].float().cpu().numpy()
        return out.float().cpu().numpy(), None

    def predict_batch(self, x: np.ndarray) -> Prediction:
        color_logits, type_logits = self.forward_logits(x)
        probs = _softmax(color_logits) if len(color_logits) else color_logits
        idx = probs.argmax(axis=1) if len(probs) else np.zeros((0,), np.int64)
        labels = self.idx_to_label[idx].astype(np.int32)
        confs = (probs[np.arange(len(idx)), idx] if len(idx) else np.zeros((0,), np.float32)).astype(np.float32)
        type_probs = _softmax(type_logits).astype(np.float32) if type_logits is not None and len(type_logits) else None
        return Prediction(labels=labels, confs=confs, color_probs=probs.astype(np.float32), type_probs=type_probs)

    def predict_rois(self, rois: list[np.ndarray]) -> Prediction:
        """BGR ROI lista -> Prediction (a pipeline fő belépési pontja)."""
        return self.predict_batch(self.preprocess(rois))

    def predict_square(self, roi):
        if roi is None or roi.size == 0:
            return "empty", 0.0
        p = self.predict_rois([roi])
        idx = int(p.color_probs[0].argmax())
        return self.idx_to_class[idx], float(p.color_probs[0, idx])

    def __repr__(self) -> str:
        src = self.onnx_path if self.backend == "onnx" else self.weights_path
        q = f", int8={self.quantized}" if self.quantized else ""
        return (f"OccupancyColorModel(backend={self.backend}{q}, variant={self.variant}, img_size={self.img_size}, "
                f"class_names={self.class_names}, type_head={self.has_type_head}, file={src})")
