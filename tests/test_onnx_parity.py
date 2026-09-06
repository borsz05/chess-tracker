"""
.pt és .onnx numerikus egyezése (feladatleírás 9. és 11.6), backend-választás,
vektorizált preprocess == régi mezőnkénti út.
"""
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from vision.models.occupancy_color_model import OccupancyColorModel, preprocess_rois, sidecar_path
from vision.models.square_net import MultiTaskSquareNet, PIECE_TYPE_CLASSES, make_checkpoint

ROOT = Path(__file__).resolve().parents[1]
REAL_CKPT = ROOT / "vision" / "models" / "weights" / "resnet18_best_topdown.pt"
LIVE_ROIS = sorted(glob.glob(str(ROOT / "tools" / "live_dump" / "*" / "*.jpg")))
NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _rois(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    if LIVE_ROIS:
        return [cv2.imread(f) for f in LIVE_ROIS[:n]]
    return [rng.integers(0, 255, (126, 126, 3), dtype=np.uint8) for _ in range(n)]


def _legacy_preprocess(roi, size, mean, std):
    """A régi batch_classifier.preprocess_roi_for_batch (mezőnkénti torch-út)."""
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    h, w = roi.shape[:2]
    interp = cv2.INTER_CUBIC if (w < size or h < size) else cv2.INTER_AREA
    roi = cv2.resize(roi, (size, size), interpolation=interp)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(roi)).permute(2, 0, 1).float().div_(255.0)
    return x.sub_(torch.tensor(mean).view(3, 1, 1)).div_(torch.tensor(std).view(3, 1, 1))


@pytest.fixture(scope="module")
def multitask_ckpt(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    model = MultiTaskSquareNet("mobilenet_v3_small", pretrained=False, dropout=0.0).eval()
    ckpt = make_checkpoint(model, class_names=["black", "empty", "white"], img_size=100, normalize=NORM)
    p = tmp_path_factory.mktemp("mt") / "mnv3.pt"
    torch.save(ckpt, p)
    return p


def _export(weights: Path) -> Path:
    from tools.export_onnx import export
    out, _ = export(weights, weights.with_suffix(".onnx"))
    return out


def test_vectorized_preprocess_matches_legacy_loop():
    rng = np.random.default_rng(1)
    rois = _rois(32, rng) + [rng.integers(0, 255, (60, 70, 3), dtype=np.uint8),   # nagyítás ág
                             rng.integers(0, 255, (100, 100), dtype=np.uint8)]     # szürke ág
    mean, std = np.asarray(NORM["mean"], np.float32), np.asarray(NORM["std"], np.float32)
    x = preprocess_rois(rois, 100, mean, std)
    ref = torch.stack([_legacy_preprocess(r, 100, NORM["mean"], NORM["std"]) for r in rois]).numpy()
    assert x.shape == ref.shape == (len(rois), 3, 100, 100)
    np.testing.assert_allclose(x, ref, atol=2e-6)


def test_multitask_pt_vs_onnx_parity_and_type_head(multitask_ckpt):
    onnx_path = _export(multitask_ckpt)
    assert sidecar_path(onnx_path).exists()
    meta = json.loads(sidecar_path(onnx_path).read_text())
    assert meta["outputs"] == ["color_logits", "type_logits"] and meta["class_names"] == ["black", "empty", "white"]

    t = OccupancyColorModel(multitask_ckpt, device="cpu", backend="torch")
    o = OccupancyColorModel(multitask_ckpt, backend="onnx")
    assert t.backend == "torch" and o.backend == "onnx" and o.has_type_head and t.has_type_head
    assert o.idx_to_label.tolist() == t.idx_to_label.tolist() == [2, 0, 1]

    rng = np.random.default_rng(2)
    for n in (1, 13, 64):                       # dinamikus batch: partial és teljes út
        rois = _rois(n, rng) if n <= len(LIVE_ROIS) or not LIVE_ROIS else [rng.integers(0, 255, (126, 126, 3), dtype=np.uint8) for _ in range(n)]
        pt, po = t.predict_rois(rois), o.predict_rois(rois)
        assert pt.labels.shape == (n,) and po.type_probs.shape == (n, len(PIECE_TYPE_CLASSES))
        np.testing.assert_array_equal(pt.labels, po.labels)
        np.testing.assert_allclose(pt.color_probs, po.color_probs, atol=1e-5)
        np.testing.assert_allclose(pt.type_probs, po.type_probs, atol=1e-5)
        np.testing.assert_allclose(pt.confs, po.confs, atol=1e-5)


def test_auto_backend_prefers_onnx_when_present_else_torch(multitask_ckpt, tmp_path):
    onnx_path = multitask_ckpt.with_suffix(".onnx")
    if not onnx_path.exists():
        _export(multitask_ckpt)
    assert OccupancyColorModel(multitask_ckpt, backend="auto").backend == "onnx"
    # .pt másolat .onnx nélkül -> torch fallback
    lone = tmp_path / "lone.pt"
    lone.write_bytes(multitask_ckpt.read_bytes())
    assert OccupancyColorModel(lone, backend="auto").backend == "torch"
    with pytest.raises(FileNotFoundError):
        OccupancyColorModel(lone, backend="onnx")


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="nincs resnet18 checkpoint")
def test_real_resnet18_pt_vs_onnx_parity(tmp_path):
    ck = tmp_path / "resnet18.pt"
    ck.symlink_to(REAL_CKPT)
    onnx_path = _export(ck)
    t = OccupancyColorModel(ck, device="cpu", backend="torch")
    o = OccupancyColorModel(ck, backend="onnx", onnx_path=onnx_path)
    assert not o.has_type_head and o.class_names == t.class_names
    rois = _rois(64, np.random.default_rng(3))
    pt, po = t.predict_rois(rois), o.predict_rois(rois)
    np.testing.assert_array_equal(pt.labels, po.labels)
    np.testing.assert_allclose(pt.color_probs, po.color_probs, atol=1e-5)
    assert po.type_probs is None
    # a legacy API is él
    name, conf = o.predict_square(rois[0])
    assert name in t.class_names and 0.0 <= conf <= 1.0
