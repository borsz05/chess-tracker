"""
Chess square classifier training script — Google Colab ready.

Tanít egy EfficientNet-B0 modellt (ResNet18-nál jobb, kisebb, gyorsabb)
a 3-osztályos feladatra: empty / white / black.

Különös hangsúly a fekete bábu helyes felismerésére:
- erős brightness/contrast augmentáció (sötét mezőn fekete bábu)
- class-weighted loss (black osztály hangsúlyosabb)
- focal loss opció

Mappastruktúra (ugyanaz mint collect_training_data.py gyűjti):
    data/
        empty/   *.jpg
        white/   *.jpg
        black/   *.jpg

Futtatás Colab-on:
    1. Töltsd fel a data/ mappát Google Drive-ra
    2. Másold ezt a scriptet is Drive-ra
    3. Futtasd cellánként az alábbiakat

Kimenet: best_efficientnet_b0.pt  (betölthető az OccupancyColorModel-be)
"""

# ===========================================================================
# Colab setup — ha helyi gépen futtatod, kommenteld ki ezt a blokkot
# ===========================================================================
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.chdir('/content/drive/MyDrive/chess-tracker')  # vagy ahova tetted

# ===========================================================================
# Importok
# ===========================================================================
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ===========================================================================
# Konfig — itt állítsd be a sajátodat
# ===========================================================================
DATA_DIR       = Path("data")          # ahol az empty/ white/ black/ mappák vannak
OUTPUT_PATH    = Path("best_efficientnet_b0.pt")
IMG_SIZE       = 128                   # 100 -> 128: több részlet a fekete bábu / sötét mező elkülönítéséhez
CONTEXT        = 0.50                  # crop context, ugyanaz mint AppConfig.context

EPOCHS         = 80
BATCH_SIZE     = 64
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS    = 2                     # Colab-on 2-4 ajánlott
EARLY_STOP_PATIENCE = 10               # macro_acc alapján, ha ennyi epoch alatt nincs javulás

# Focal loss gamma — 0 = sima CE, 2 = erős focus a nehéz példákra (fekete bábu)
FOCAL_GAMMA    = 2.0

SEED = 42
TRAIN_RATIO = 0.85

# ===========================================================================
# Seed
# ===========================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===========================================================================
# Augmentáció
# ===========================================================================
# ImageNet normalizáció — mert EfficientNet ImageNet pre-trained súlyokat kap
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Erős train augmentáció — külön hangsúly a megvilágítás-variációra
# mert a fekete bábu sötét mezőn szinte láthatatlan gyenge augmentációval
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Tükrözések — sakkmezők szimmetrikusak, így ez biztonságos
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    # Kis elforgatás — kamera nem mindig tökéletesen felülről néz
    transforms.RandomRotation(degrees=8),

    # Perspektíva torzítás — kamera szög eltérések
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),

    # === MEGVILÁGÍTÁS AUGMENTÁCIÓ — ez a legkritikusabb ===
    # Nagy brightness/contrast range: fekete bábu sötét mezőn jól látható
    # legyen rossz megvilágításban is
    transforms.ColorJitter(
        brightness=0.5,   # ±50% fényerő változás
        contrast=0.5,     # ±50% kontraszt változás
        saturation=0.3,   # ±30% telítettség
        hue=0.08,         # kis színárnyalat eltolás
    ),

    # Véletlenszerű szürkeárnyalat — megvilágítás-független jellemzők tanulása
    transforms.RandomGrayscale(p=0.10),

    # Gauss blur — kamera fókuszhiba szimulálása
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

    # Automatikus kontraszt — segít a fekete bábun sötét mezőn
    transforms.RandomAutocontrast(p=0.3),

    # RandomEqualize — hisztogram kiegyenlítés, kontrasztnövelés
    transforms.RandomEqualize(p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

    # Random erasing — részleges eltakarás szimulálása (kéz, árnyék)
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ===========================================================================
# Dataset betöltés + train/val split
# ===========================================================================
print(f"\nAdatbetöltés: {DATA_DIR.resolve()}")

full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=None)
class_names = full_dataset.classes
print(f"Osztályok: {class_names}")

# Darabszámok per osztály
class_counts = [0] * len(class_names)
for _, label in full_dataset.samples:
    class_counts[label] += 1
for name, cnt in zip(class_names, class_counts):
    print(f"  {name}: {cnt} kép")

total = len(full_dataset)
print(f"  Összesen: {total} kép")

if total == 0:
    raise RuntimeError(f"Nincs adat a {DATA_DIR} mappában!")

# Train/val split — stratified
all_indices = list(range(total))
random.shuffle(all_indices)

by_class: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
for idx in all_indices:
    _, label = full_dataset.samples[idx]
    by_class[label].append(idx)

train_idx, val_idx = [], []
for label, idxs in by_class.items():
    n_train = max(1, int(len(idxs) * TRAIN_RATIO))
    train_idx.extend(idxs[:n_train])
    val_idx.extend(idxs[n_train:])

random.shuffle(train_idx)
random.shuffle(val_idx)
print(f"\nTrain: {len(train_idx)} | Val: {len(val_idx)}")

# Külön transform a két splitnek
from torch.utils.data import Subset

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_subset = Subset(full_dataset, train_idx)
val_subset   = Subset(full_dataset, val_idx)

# ImageFolder PIL képeket ad vissza, transform kell
from torchvision.datasets.folder import default_loader
import PIL

class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.ImageFolder, indices: list[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = SubsetWithTransform(full_dataset, train_idx, train_transform)
val_ds   = SubsetWithTransform(full_dataset, val_idx,   val_transform)

# ===========================================================================
# WeightedRandomSampler — kiegyensúlyozza az osztályokat
# (különösen fontos ha kevés black minta van)
# ===========================================================================
train_labels = [full_dataset.samples[i][1] for i in train_idx]
class_sample_count = np.bincount(train_labels, minlength=len(class_names))
class_weights = 1.0 / np.maximum(class_sample_count, 1)
sample_weights = [class_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(train_idx),
    replacement=True,
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

# ===========================================================================
# Modell — EfficientNet-B0
# ===========================================================================
print("\nModell betöltése (EfficientNet-B0, ImageNet pre-trained)...")
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Utolsó réteg cseréje 3 osztályra
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))

model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Paraméterek: {n_params:,}")

# ===========================================================================
# Focal Loss — segít a nehéz eseteken (pl. fekete bábu sötét mezőn)
# ===========================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p) = -alpha * (1 - p)^gamma * log(p)
    gamma=0 → sima cross entropy
    gamma=2 → erős focus a nehéz, alacsony konfidenciájú példákra
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# Class weights a loss-ban — black osztály nehezebb, súlyozzuk fel
loss_class_weights = torch.ones(len(class_names), device=device)
black_idx = class_names.index("black") if "black" in class_names else -1
if black_idx >= 0:
    loss_class_weights[black_idx] = 2.0  # black 2x súlyú a loss-ban
    print(f"Black osztály loss weight: 2.0 (index: {black_idx})")

criterion = FocalLoss(
    gamma=FOCAL_GAMMA,
    label_smoothing=LABEL_SMOOTHING,
    weight=loss_class_weights,
)

# ===========================================================================
# Optimizer + Scheduler
# ===========================================================================
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

# ===========================================================================
# Tanítás
# ===========================================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = [0] * len(class_names)
    per_class_total   = [0] * len(class_names)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[l] += 1
                if p == l:
                    per_class_correct[l] += 1

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    per_class_acc = [
        per_class_correct[i] / max(1, per_class_total[i])
        for i in range(len(class_names))
    ]
    return avg_loss, acc, per_class_acc


print("\n" + "=" * 65)
print(f"  Tanítás indul: {EPOCHS} epoch, batch={BATCH_SIZE}, lr={LR}")
print("=" * 65)

best_macro_acc = 0.0
best_val_acc_at_best = 0.0
best_epoch = 0
best_state = None
history = []
patience = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    t0 = time.time()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)

    scheduler.step()

    train_loss /= len(train_idx)
    val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)
    # Osztályonkénti pontosságok átlaga — ez a modellkiválasztás alapja,
    # nem a nyers val_acc, mert a nyers acc-ot a domináns (empty/white)
    # osztályok elnyomhatják, és pont a "black" recall-ja a lényeg.
    macro_acc = float(np.mean(per_class_acc))
    elapsed = time.time() - t0

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "macro_acc": macro_acc,
        "per_class_acc": per_class_acc,
        "lr": scheduler.get_last_lr()[0],
    })

    acc_str = " | ".join(
        f"{class_names[i]}: {per_class_acc[i]*100:.1f}%"
        for i in range(len(class_names))
    )

    print(
        f"Epoch {epoch:3d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"val_acc={val_acc*100:.2f}% | macro_acc={macro_acc*100:.2f}% | {acc_str} | "
        f"{elapsed:.1f}s"
    )

    if macro_acc > best_macro_acc:
        best_macro_acc = macro_acc
        best_val_acc_at_best = val_acc
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
        print(f"  → Legjobb modell elmentve (macro_acc={best_macro_acc*100:.2f}%, val_acc={val_acc*100:.2f}%)")
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print(f"  ⏹ Early stopping — {EARLY_STOP_PATIENCE} epoch óta nincs javulás macro_acc-ban.")
            break

print(f"\nLegjobb macro_acc: {best_macro_acc*100:.2f}% (val_acc={best_val_acc_at_best*100:.2f}%) @ epoch {best_epoch}")

# ===========================================================================
# Mentés — UGYANOLYAN FORMÁTUM mint a jelenlegi pipeline vár
# ===========================================================================
checkpoint = {
    "variant": "efficientnet_b0",
    "epoch": best_epoch,
    "model_state": best_state,
    "optimizer_state": optimizer.state_dict(),
    "val_acc": best_val_acc_at_best,
    "macro_acc": best_macro_acc,
    "class_names": class_names,
    "img_size": IMG_SIZE,
    "normalize": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "history": history,
}

torch.save(checkpoint, OUTPUT_PATH)
print(f"\nModell mentve: {OUTPUT_PATH.resolve()}")
print("A pipeline-ban cseréld le a resnet18_best_topdown.pt fájlt erre.")
print("Az OccupancyColorModel automatikusan betölti a helyes architectúrát")
print("ha frissíted azt is (lásd lentebb).")

# ===========================================================================
# Per-class confusion matrix
# ===========================================================================
print("\n" + "=" * 45)
print("  KONFÚZIÓS MÁTRIX (val set)")
print("=" * 45)

model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        for p, l in zip(preds, labels.numpy()):
            confusion[l, p] += 1

header = "         " + "  ".join(f"{n:>8s}" for n in class_names)
print(f"{header}   (pred →)")
for i, name in enumerate(class_names):
    row = "  ".join(f"{confusion[i, j]:>8d}" for j in range(len(class_names)))
    print(f"{name:>8s} | {row}")

print("\n(sorok = valódi osztály, oszlopok = predikált osztály)")

# ===========================================================================
# History + konfúziós mátrix mentése (elemzéshez, a checkpoint mellé)
# ===========================================================================
cm_path = OUTPUT_PATH.with_suffix(".confusion_matrix.npy")
hist_path = OUTPUT_PATH.with_suffix(".history.json")
np.save(cm_path, confusion)
with open(hist_path, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)
print(f"\nKonfúziós mátrix mentve: {cm_path.resolve()}")
print(f"Történet (history) mentve: {hist_path.resolve()}")
