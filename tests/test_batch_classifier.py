"""A pipeline klasszifikációs belépési pontjai az új wrapperrel (ONNX és torch backend)."""
import numpy as np
import pytest
import torch

from vision.models.occupancy_color_model import OccupancyColorModel
from vision.models.square_net import MultiTaskSquareNet, PIECE_TYPE_CLASSES, make_checkpoint
from vision.pipeline.batch_classifier import classify_squares_batch, classify_warp_squares_batch

NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    from tools.export_onnx import export
    torch.manual_seed(0)
    net = MultiTaskSquareNet("mobilenet_v3_small", pretrained=False, dropout=0.0).eval()
    p = tmp_path_factory.mktemp("bc") / "m.pt"
    torch.save(make_checkpoint(net, class_names=["black", "empty", "white"], img_size=100, normalize=NORM), p)
    export(p, p.with_suffix(".onnx"))
    return OccupancyColorModel(p, backend="onnx"), OccupancyColorModel(p, device="cpu", backend="torch")


def _grid(cell=96, pad=6):
    # 17x17 cellás warp, a tábla a közepén: 8x8 bbox
    off = 4 * cell
    return [[(off + c * cell + pad, off + r * cell + pad, off + (c + 1) * cell - pad, off + (r + 1) * cell - pad)
             for c in range(8)] for r in range(8)]


def test_full_and_partial_paths_agree_across_backends(models):
    onnx_m, torch_m = models
    rng = np.random.default_rng(0)
    warp = rng.integers(0, 255, (17 * 96, 17 * 96, 3), dtype=np.uint8)
    grid = _grid()

    a = classify_warp_squares_batch(warp, grid, onnx_m, context=0.5)
    b = classify_warp_squares_batch(warp, grid, torch_m, context=0.5)
    assert a.labels.shape == (8, 8) and a.confs.shape == (8, 8)
    assert a.type_probs is not None and a.type_probs.shape == (8, 8, len(PIECE_TYPE_CLASSES))
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.confs, b.confs, atol=1e-5)
    np.testing.assert_allclose(a.type_probs, b.type_probs, atol=1e-5)
    assert set(np.unique(a.labels)).issubset({0, 1, 2})

    squares = [(0, 0), (3, 4), (7, 7), (5, 1)]
    part = classify_squares_batch(warp, grid, squares, onnx_m, context=0.5)
    assert part.positions == squares
    for i, (r, c) in enumerate(part.positions):
        assert int(part.labels[i]) == a.labels[r, c]
        assert abs(float(part.confs[i]) - a.confs[r, c]) < 1e-5
    assert classify_squares_batch(warp, grid, [], onnx_m).positions == []
