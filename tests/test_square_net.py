"""MultiTaskSquareNet: kimeneti alakok, típus-fej leválasztása, dinamikus batch ONNX-ben."""
import numpy as np
import pytest
import torch

from vision.models.square_net import VARIANTS, GradScale, MultiTaskSquareNet, PIECE_TYPE_CLASSES


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_shapes(variant):
    m = MultiTaskSquareNet(variant, pretrained=False).eval()
    with torch.no_grad():
        c, t = m(torch.randn(2, 3, 100, 100))
    assert c.shape == (2, 3)
    assert t.shape == (2, len(PIECE_TYPE_CLASSES))
    assert torch.allclose(m.forward_color(torch.randn(1, 3, 100, 100)) * 0, torch.zeros(1, 3))


def test_type_head_detached_when_grad_scale_zero():
    m = MultiTaskSquareNet("tiny_cnn", type_grad_scale=0.0)
    x = torch.randn(4, 3, 100, 100)
    _, t = m(x)
    t.sum().backward()
    # a backbone nem kap gradienst a típus-fejből
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in m.backbone.parameters())
    assert m.type_head.weight.grad is not None and torch.count_nonzero(m.type_head.weight.grad) > 0


def test_grad_scale_scales_backward():
    x = torch.randn(3, 5, requires_grad=True)
    GradScale(0.25)(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 0.25))
    x2 = torch.randn(3, 5, requires_grad=True)
    GradScale(1.0)(x2).sum().backward()
    assert torch.allclose(x2.grad, torch.ones_like(x2))


def test_type_head_full_grad_reaches_backbone():
    m = MultiTaskSquareNet("tiny_cnn", type_grad_scale=1.0)
    _, t = m(torch.randn(4, 3, 100, 100))
    t.sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in m.backbone.parameters())


def test_onnx_export_dynamic_batch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from tools.train_square_classifier import export_onnx

    m = MultiTaskSquareNet("tiny_cnn", dropout=0.0).eval()
    p = tmp_path / "m.onnx"
    export_onnx(m, 100, p)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    for b in (1, 13, 64):
        x = np.random.randn(b, 3, 100, 100).astype(np.float32)
        c, t = sess.run(None, {"input": x})
        assert c.shape == (b, 3) and t.shape == (b, len(PIECE_TYPE_CLASSES))
        with torch.no_grad():
            c_t, t_t = m(torch.from_numpy(x))
        np.testing.assert_allclose(c, c_t.numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(t, t_t.numpy(), rtol=1e-4, atol=1e-4)
