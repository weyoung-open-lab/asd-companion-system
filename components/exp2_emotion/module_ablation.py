"""
EfficientNet-B0 模块消融实验（基于AffectNet 8类预训练）
========================================================
在最佳预训练基础上叠加轻量模块:
  1. SE Attention (Channel Attention, ~200K params)
  2. Focal Loss (no extra params)
  3. SE + Focal

Usage:
  python module_ablation.py --phase all
"""

import os
import argparse
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib
matplotlib.use('Agg')

# ============================================================
# CONFIG
# ============================================================
FERAC_ROOT = r"./data/FERAC Dataset"
MODEL_DIR = r"./output/transfer_exp/models"
OUTPUT_DIR = r"./output/module_ablation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
N_FOLDS = 5
SEED = 42


# ============================================================
# SE ATTENTION MODULE
# ============================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns channel-wise attention weights.
    Input: (B, C) feature vector
    Output: (B, C) reweighted feature vector
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)  # (B, C)
        return x * w     # channel-wise reweighting


# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            smooth = torch.zeros_like(inputs)
            smooth.fill_(self.label_smoothing / (n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            focal_weight = (1 - probs) ** self.gamma
            loss = (-focal_weight * log_probs * smooth).sum(dim=1)
        else:
            focal_weight = (1 - probs.gather(1, targets.unsqueeze(1))) ** self.gamma
            loss = (-focal_weight * log_probs.gather(1, targets.unsqueeze(1))).squeeze(1)

        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss

        return loss.mean()


# ============================================================
# MODEL
# ============================================================
class EfficientNetSE(nn.Module):
    """EfficientNet-B0 + optional SE Attention."""
    def __init__(self, n_classes=4, use_se=False):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = effnet.features
        self.avgpool = effnet.avgpool
        self.use_se = use_se

        if use_se:
            self.se = SEBlock(1280, reduction=16)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)  # (B, 1280)
        if self.use_se:
            x = self.se(x)
        return self.classifier(x)


# ============================================================
# DATASET
# ============================================================
class FERACDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.samples = []
        class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        for split in ["train", "test", "训练", "测试"]:
            split_dir = os.path.join(root, split)
            if not os.path.exists(split_dir):
                continue
            for cls_folder in os.listdir(split_dir):
                cls_name = CN_TO_EN.get(cls_folder, cls_folder)
                if cls_name not in class_to_idx:
                    continue
                cls_dir = os.path.join(split_dir, cls_folder)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append((os.path.join(cls_dir, fname),
                                            class_to_idx[cls_name]))
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        return Image.open(path).convert('RGB'), label


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# TRANSFORMS & UTILS
# ============================================================
def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_sampler(targets):
    counts = Counter(targets)
    total = sum(counts.values())
    weights = {c: total / cnt for c, cnt in counts.items()}
    return WeightedRandomSampler([weights[t] for t in targets], len(targets))


def compute_class_weights(targets):
    counts = Counter(targets)
    total = sum(counts.values())
    n = len(counts)
    w = torch.ones(NUM_CLASSES).to(DEVICE)
    for c, cnt in counts.items():
        w[c] = total / (n * cnt)
    return w


# ============================================================
# TRAINING
# ============================================================
def train_one_fold(train_idx, val_idx, dataset, fold_idx, use_se=False, use_focal=False):
    train_ds = TransformSubset(dataset, train_idx, get_transforms("train"))
    val_ds = TransformSubset(dataset, val_idx, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=16,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Build model with SE option
    model = EfficientNetSE(n_classes=NUM_CLASSES, use_se=use_se)

    # Load AffectNet 8cls pretrained
    pretrained_path = os.path.join(MODEL_DIR, "effb0_affectnet_8cls.pth")
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=DEVICE)
        model_dict = model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)

    model = model.to(DEVICE)

    # Freeze early layers
    n_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i < int(n_blocks * 0.6):
            for p in block.parameters():
                p.requires_grad = False

    # Loss
    class_weights = compute_class_weights(train_ds.targets)
    if use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-3)

    epochs = 80
    warmup = 5
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_f1 = 0
    patience_counter = 0
    best_per_class = None

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                all_preds.extend(model(images).argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
        scheduler.step()

        if epoch % 15 == 0 or val_f1 > best_f1:
            print(f"    Fold {fold_idx} Ep {epoch:>3d}: val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_per_class = per_class
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"    Fold {fold_idx}: early stop ep {epoch}")
                break

    return best_f1, best_acc, best_per_class


def run_cv(name, use_se=False, use_focal=False):
    print(f"\n{'='*60}")
    print(f"  {name} (SE={use_se}, Focal={use_focal})")
    print(f"{'='*60}")

    set_seed(SEED)
    dataset = FERACDataset(FERAC_ROOT)
    targets = np.array(dataset.targets)
    print(f"  Total: {len(dataset)} images")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s, fold_accs, fold_per_class = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), targets), 1):
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} ---")
        set_seed(SEED + fold_idx)
        f1, acc, pc = train_one_fold(
            train_idx.tolist(), val_idx.tolist(), dataset, fold_idx,
            use_se=use_se, use_focal=use_focal)
        fold_f1s.append(f1)
        fold_accs.append(acc)
        fold_per_class.append(pc.tolist())

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    mean_per_class = np.mean(fold_per_class, axis=0)
    std_per_class = np.std(fold_per_class, axis=0)

    print(f"\n  {name}: F1 = {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  Folds: {[f'{f:.3f}' for f in fold_f1s]}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:<12}: {mean_per_class[i]:.3f} ± {std_per_class[i]:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{name}_cv.json"), "w") as f:
        json.dump({
            "name": name, "use_se": use_se, "use_focal": use_focal,
            "mean_f1": float(mean_f1), "std_f1": float(std_f1),
            "fold_f1s": [float(x) for x in fold_f1s],
            "mean_per_class": mean_per_class.tolist(),
            "std_per_class": std_per_class.tolist(),
            "class_names": CLASSES,
        }, f, indent=2)

    return mean_f1, std_f1


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["se_only", "focal_only", "se_focal", "all"])
    args = parser.parse_args()

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if args.phase in ["se_only", "all"]:
        f1, std = run_cv("PT_SE", use_se=True, use_focal=False)
        results["PT+SE"] = f"{f1:.3f}±{std:.3f}"

    if args.phase in ["focal_only", "all"]:
        f1, std = run_cv("PT_Focal", use_se=False, use_focal=True)
        results["PT+Focal"] = f"{f1:.3f}±{std:.3f}"

    if args.phase in ["se_focal", "all"]:
        f1, std = run_cv("PT_SE_Focal", use_se=True, use_focal=True)
        results["PT+SE+Focal"] = f"{f1:.3f}±{std:.3f}"

    if args.phase == "all":
        print(f"\n{'='*60}")
        print(f"  MODULE ABLATION SUMMARY")
        print(f"  (All with AffectNet 8-cls pretrain)")
        print(f"{'='*60}")
        print(f"  Baseline (PT only):  0.701 ± 0.058")
        for name, val in results.items():
            print(f"  {name:<20}: {val}")


if __name__ == "__main__":
    main()
