"""
Közös modell-architektúra a sakkmező-osztályozóhoz (tanítás ÉS inferencia).

Ez a modul a tanítószkript (tools/train_square_classifier.py) és a későbbi
inferencia-betöltő (OccupancyColorModel utódja, ONNX export) EGYETLEN közös
forrása az architektúrára. Szándékosan csak torch/torchvision-t importál
(nincs cv2, nincs chess), hogy Colabban és a pipeline-ban is ugyanaz fusson.

Multi-task felépítés
--------------------
    backbone(x)  -> pooled feature vector  (N, feat_dim)
    color_head   -> 3 logit   (elsődleges: empty / white / black — a sorrend a
                                checkpoint `class_names` mezőjéből jön!)
    type_head    -> 7 logit   (másodlagos: none / pawn / knight / bishop / rook /
                                queen / king)

A típus-fej a backbone-nal a GradScale-en keresztül van összekötve: a
`type_grad_scale` 1.0-nál teljes multi-task tanítás, 0.0-nál a típus-fej
csak a "befagyasztott" feature-ön tanul (nem tud negatív transzfert okozni a
3-osztályos fejre). Inferenciakor a forward mindkét fejet visszaadja; a
pipeline-nak a típus-kimenetet egyelőre csak promóciónál szabad használnia.

Miért 7 típusosztály és nem 13 (szín x típus)?
    A színt már az elsődleges fej adja, így a típus-fejnek nem kell újra
    megtanulnia. 7 osztállyal minden típus-osztály KÉTSZER annyi mintát kap
    (fehér és fekete gyalog ugyanaz az osztály), ami a kevés típus-címkézett
    adatnál számít. Promóciónál a színt a lépés oldala már megadja, tehát a
    7-osztályos kimenet bőven elég a (q/r/b/n) döntéshez.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------------------------
# Osztály-konvenciók
# ---------------------------------------------------------------------------

# A pipeline belső foglaltság-kódolása (chess_logic.resolver.board_to_occupancy):
PIPELINE_LABEL = {"empty": 0, "white": 1, "black": 2}

# A típus-fej kimeneti sorrendje — ez FIX, nem ImageFolder-ből jön.
PIECE_TYPE_CLASSES: tuple[str, ...] = ("none", "pawn", "knight", "bishop", "rook", "queen", "king")

# python-chess `piece.symbol().lower()` -> típus index
PIECE_SYMBOL_TO_TYPE_IDX = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}
TYPE_IDX_TO_PROMOTION_CHAR = {2: "n", 3: "b", 4: "r", 5: "q"}

# Ismeretlen típus (régi, csak színnel címkézett adat) -> a típus-loss kihagyja
TYPE_IGNORE_INDEX = -1

# A jelöltmodellek nevei; a checkpoint `variant` mezője ezek egyike.
VARIANTS: tuple[str, ...] = (
    "tiny_cnn",
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
    "resnet18",
)


def class_names_to_idx_to_label(class_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """
    A modell kimeneti indexét (class_names sorrend) fordítja a pipeline
    kódolására (empty=0, white=1, black=2).

    UGYANAZ a logika, mint az OccupancyColorModel.__init__-ben: a checkpoint
    `class_names` mezője abban a sorrendben kell legyen, ahogy a modell
    kimeneti indexei jelentik. ImageFolder esetén ez ábécésorrend:
    ['black', 'empty', 'white'] -> idx_to_label = [2, 0, 1].
    """
    out = np.zeros(len(class_names), dtype=np.int32)
    for idx, name in enumerate(class_names):
        out[idx] = PIPELINE_LABEL.get(str(name), 0)
    return out


# ---------------------------------------------------------------------------
# Gradiens-skálázás a típus-fej és a backbone között
# ---------------------------------------------------------------------------

class _GradScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:  # type: ignore[override]
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out * ctx.scale, None


class GradScale(nn.Module):
    """Identitás előre, `scale`-szeres gradiens hátra. scale=0 -> leválasztás."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() or not x.requires_grad:
            return x
        if self.scale == 1.0:
            return x
        if self.scale == 0.0:
            return x.detach()
        return _GradScaleFn.apply(x, self.scale)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


# ---------------------------------------------------------------------------
# Saját kis CNN — 100 px-es, 3 osztályos feladathoz
# ---------------------------------------------------------------------------

def _conv_bn_act(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyCNN(nn.Module):
    """
    5 konvolúciós blokk, sűrű (nem depthwise) 3x3 konvolúciók — CPU-n ezek a
    leghatékonyabbak (jó cache-viselkedés, oneDNN-optimalizált). Kb. 0.5 M
    paraméter, ~100 px bemeneten ~25 MFLOP/kép.

    Blokkonként stride-2 downsampling:  100 -> 50 -> 25 -> 13 -> 7 -> 4,
    majd global average pool -> feat_dim.
    """

    def __init__(self, widths: tuple[int, ...] = (24, 32, 64, 96, 128), in_ch: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        cin = in_ch
        for i, w in enumerate(widths):
            # az első blokk stride 1 + stride 2, hogy a finom él-információ
            # (bábu kontúr sötét mezőn) ne vesszen el már az első lépésben
            layers.append(_conv_bn_act(cin, w, stride=1 if i == 0 else 2))
            layers.append(_conv_bn_act(w, w, stride=2 if i == 0 else 1))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(self.pool(x), 1)


# ---------------------------------------------------------------------------
# Backbone gyár
# ---------------------------------------------------------------------------

def build_backbone(variant: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    """
    Visszaad: (backbone, feat_dim). A backbone kimenete (N, feat_dim) pooled
    feature; a torchvision fejét levágjuk.
    """
    if variant == "tiny_cnn":
        m = TinyCNN()
        return m, m.feat_dim

    if variant == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat

    if variant == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feat = m.classifier[0].in_features  # 576
        m.classifier = nn.Identity()
        return m, feat

    if variant == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feat = m.fc.in_features  # 1024
        m.fc = nn.Identity()
        return m, feat

    if variant == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feat = m.classifier[1].in_features  # 1280
        m.classifier = nn.Identity()
        return m, feat

    raise ValueError(f"Ismeretlen variant: {variant!r} (várt: {VARIANTS})")


# ---------------------------------------------------------------------------
# Multi-task modell
# ---------------------------------------------------------------------------

class MultiTaskSquareNet(nn.Module):
    """
    forward(x) -> (color_logits [N, n_color], type_logits [N, n_type])

    A két fej ugyanazon a pooled feature-ön dolgozik; a számítás ~99%-a a
    backbone-ban van, így a második fej az inferenciát mérhetően nem lassítja.
    """

    MODEL_CLASS = "MultiTaskSquareNet"

    def __init__(
        self,
        variant: str,
        n_color: int = 3,
        n_type: int = len(PIECE_TYPE_CLASSES),
        *,
        pretrained: bool = False,
        type_grad_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = variant
        self.n_color = int(n_color)
        self.n_type = int(n_type)
        self.backbone, self.feat_dim = build_backbone(variant, pretrained=pretrained)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.color_head = nn.Linear(self.feat_dim, self.n_color)
        self.type_grad = GradScale(type_grad_scale)
        self.type_head = nn.Linear(self.feat_dim, self.n_type)

    def set_type_grad_scale(self, scale: float) -> None:
        self.type_grad.scale = float(scale)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.backbone(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        color = self.color_head(f)
        ptype = self.type_head(self.type_grad(f))
        return color, ptype

    def forward_color(self, x: torch.Tensor) -> torch.Tensor:
        """Csak a 3-osztályos fej (a fő lépésdetektálási útvonal ezt használja)."""
        return self.color_head(self.features(x))


# ---------------------------------------------------------------------------
# Checkpoint-szerződés
# ---------------------------------------------------------------------------

def make_checkpoint(
    model: MultiTaskSquareNet,
    *,
    class_names: list[str],
    img_size: int,
    normalize: dict[str, list[float]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Az OccupancyColorModel által olvasott KÖTELEZŐ kulcsok:
        variant, model_state, class_names, img_size, normalize
    Plusz a multi-task betöltéshez: model_class, heads, type_class_names.

    `class_names` PONTOSAN a color_head kimeneti indexeinek sorrendjében!
    """
    assert len(class_names) == model.n_color, "class_names hossza != color fej kimenete"
    assert set(class_names) == set(PIPELINE_LABEL), f"váratlan osztálynevek: {class_names}"
    ckpt: dict[str, Any] = {
        "variant": model.variant,
        "model_class": MultiTaskSquareNet.MODEL_CLASS,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": list(class_names),
        "img_size": int(img_size),
        "normalize": {"mean": list(map(float, normalize["mean"])), "std": list(map(float, normalize["std"]))},
        "heads": {"color": model.n_color, "type": model.n_type},
        "type_class_names": list(PIECE_TYPE_CLASSES),
        "idx_to_label": class_names_to_idx_to_label(class_names).tolist(),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def build_from_checkpoint(ckpt: dict[str, Any]) -> MultiTaskSquareNet:
    """Checkpoint dict -> betöltött, eval módú MultiTaskSquareNet."""
    heads = ckpt.get("heads", {})
    model = MultiTaskSquareNet(
        ckpt["variant"],
        n_color=int(heads.get("color", len(ckpt["class_names"]))),
        n_type=int(heads.get("type", len(PIECE_TYPE_CLASSES))),
        pretrained=False,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
