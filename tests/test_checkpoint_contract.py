"""
Osztálysorrend-szerződés: a checkpoint `class_names` -> OccupancyColorModel.idx_to_label
fordítása a pipeline kódolását (empty=0, white=1, black=2) adja, MINDEN sorrendnél.
Ha ez elromlik, a fehér és a fekete némán felcserélődik.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

from vision.models.occupancy_color_model import OccupancyColorModel
from vision.models.square_net import (
    PIECE_TYPE_CLASSES,
    PIPELINE_LABEL,
    MultiTaskSquareNet,
    build_from_checkpoint,
    class_names_to_idx_to_label,
    make_checkpoint,
)

ORDERS = [
    ["black", "empty", "white"],   # ImageFolder ábécésorrend — ezt menti a tanítószkript
    ["empty", "white", "black"],   # a pipeline belső sorrendje
    ["white", "black", "empty"],
]


@pytest.mark.parametrize("names", ORDERS)
def test_helper_maps_every_order_to_pipeline_codes(names):
    m = class_names_to_idx_to_label(names)
    for idx, name in enumerate(names):
        assert int(m[idx]) == PIPELINE_LABEL[name]


def test_imagefolder_order_gives_2_0_1():
    assert class_names_to_idx_to_label(["black", "empty", "white"]).tolist() == [2, 0, 1]


@pytest.mark.parametrize("names", ORDERS)
def test_occupancy_color_model_idx_to_label_matches_helper(tmp_path, names):
    # valódi (kis) resnet18 checkpoint a jelenlegi betöltőn keresztül
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 3)
    ckpt = {
        "variant": "resnet18",
        "model_state": net.state_dict(),
        "class_names": names,
        "img_size": 100,
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    p = tmp_path / "ck.pt"
    torch.save(ckpt, p)
    m = OccupancyColorModel(weights_path=str(p), device="cpu")
    assert m.class_names == names
    assert m.idx_to_label.tolist() == class_names_to_idx_to_label(names).tolist()
    # a modell argmax indexének neve és a pipeline-kód ugyanazt az osztályt jelenti
    for idx, name in enumerate(names):
        assert PIPELINE_LABEL[name] == int(m.idx_to_label[idx])


def test_make_checkpoint_has_contract_keys_and_roundtrips():
    model = MultiTaskSquareNet("tiny_cnn", dropout=0.0).eval()
    ckpt = make_checkpoint(model, class_names=["black", "empty", "white"], img_size=100,
                           normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    for k in ("variant", "model_state", "class_names", "img_size", "normalize"):
        assert k in ckpt
    assert ckpt["idx_to_label"] == [2, 0, 1]
    assert ckpt["heads"] == {"color": 3, "type": len(PIECE_TYPE_CLASSES)}
    assert ckpt["type_class_names"][0] == "none"

    rebuilt = build_from_checkpoint(ckpt)
    x = torch.randn(4, 3, 100, 100)
    with torch.no_grad():
        c1, t1 = model(x)
        c2, t2 = rebuilt(x)
    assert torch.allclose(c1, c2) and torch.allclose(t1, t2)


def test_make_checkpoint_rejects_wrong_class_names():
    model = MultiTaskSquareNet("tiny_cnn")
    with pytest.raises(AssertionError):
        make_checkpoint(model, class_names=["black", "empty", "unsure"], img_size=100,
                        normalize={"mean": [0, 0, 0], "std": [1, 1, 1]})
