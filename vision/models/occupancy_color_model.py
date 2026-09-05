import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def _build_model(variant: str, num_classes: int) -> nn.Module:
    """Builds a classifier head for the given backbone variant."""
    if variant == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    # Default / legacy: resnet18
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class OccupancyColorModel:
    """
    3-class square classifier: empty → 0 | white → 1 | black → 2

    Supports both ResNet18 (legacy) and EfficientNet-B0 checkpoints.
    The checkpoint's 'variant' key selects the architecture automatically.
    """

    def __init__(self, weights_path, device=None, img_size=100):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(weights_path, map_location=self.device)

        self.class_names = ckpt.get("class_names", ["empty", "white", "black"])
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

        self.img_size = int(ckpt.get("img_size", img_size))

        norm = ckpt.get("normalize", None)
        if isinstance(norm, dict) and "mean" in norm and "std" in norm:
            mean = norm["mean"]
            std = norm["std"]
        else:
            mean = ckpt.get("normalize_mean", [0.5, 0.5, 0.5])
            std = ckpt.get("normalize_std", [0.25, 0.25, 0.25])

        self.norm_mean_cpu = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.norm_std_cpu = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        variant = ckpt.get("variant", "resnet18")
        self.model = _build_model(variant, len(self.class_names))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(self.device)

        self.idx_to_label = np.zeros(len(self.class_names), dtype=np.int32)
        for idx, name in self.idx_to_class.items():
            if name == "white":
                self.idx_to_label[idx] = 1
            elif name == "black":
                self.idx_to_label[idx] = 2
            else:
                self.idx_to_label[idx] = 0

    def _preprocess(self, roi):
        if roi is None or roi.size == 0:
            return None

        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        h, w = roi.shape[:2]
        size = self.img_size
        interp = cv2.INTER_CUBIC if (w < size or h < size) else cv2.INTER_AREA
        roi = cv2.resize(roi, (size, size), interpolation=interp)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(np.ascontiguousarray(roi))
        x = x.permute(2, 0, 1).float().div_(255.0)
        x.sub_(self.norm_mean_cpu).div_(self.norm_std_cpu)

        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_square(self, roi):
        x = self._preprocess(roi)
        if x is None:
            return "empty", 0.0

        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(prob))
        return self.idx_to_class[idx], float(prob[idx])
