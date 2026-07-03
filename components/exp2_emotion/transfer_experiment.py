"""
EfficientNet-B0 迁移学习实验（4类 vs 8类预训练对比）
======================================================
对比三种设置:
  1. 无预训练 (ImageNet only)
  2. AffectNet 8类预训练
  3. AffectNet 4类预训练 (只用Neutral/Anger/Fear/Happy)

Usage:
  python transfer_experiment.py --phase all
  python transfer_experiment.py --phase pretrain_4cls
  python transfer_experiment.py --phase cv_compare
"""

import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
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
AFFECTNET_ROOT = r"./data/AffectNet_Processed"
OUTPUT_DIR = r"./output/transfer_exp"
MODEL_DIR = r"./output/transfer_exp/models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
N_FOLDS = 5
SEED = 42

# AffectNet 4类映射: 0_Neutral->0, 6_Anger->1, 4_Fear->2, 1_Happy->3
AFFECTNET_4CLS_KEEP = {"0_Neutral": 0, "6_Anger": 1, "4_Fear": 2, "1_Happy": 3}


# ============================================================
# DATASETS
# ============================================================
class FERACDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
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
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class AffectNet4ClsDataset(torch.utils.data.Dataset):
    """AffectNet filtered to 4 classes matching FERAC."""
    def __init__(self, root, split, transform):
        self.transform = transform
        full_ds = datasets.ImageFolder(os.path.join(root, split))

        self.orig_to_new = {}
        for cls_name, new_idx in AFFECTNET_4CLS_KEEP.items():
            if cls_name in full_ds.class_to_idx:
                self.orig_to_new[full_ds.class_to_idx[cls_name]] = new_idx

        self.samples = []
        self.targets = []
        for path, orig_label in full_ds.samples:
            if orig_label in self.orig_to_new:
                new_label = self.orig_to_new[orig_label]
                self.samples.append((path, new_label))
                self.targets.append(new_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# TRANSFORMS
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


# ============================================================
# UTILITIES
# ============================================================
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
    sample_w = [weights[t] for t in targets]
    return WeightedRandomSampler(sample_w, len(sample_w))

def compute_class_weights(targets, n_classes):
    counts = Counter(targets)
    total = sum(counts.values())
    n = len(counts)
    w = torch.ones(n_classes).to(DEVICE)
    for c, cnt in counts.items():
        w[c] = total / (n * cnt)
    return w


# ============================================================
# PRETRAIN
# ============================================================
def pretrain_affectnet(n_pretrain_classes=8, epochs=50, batch_size=32, lr=1e-4):
    tag = f"{n_pretrain_classes}cls"
    print(f"\n{'='*60}")
    print(f"  AffectNet Pretrain: EfficientNet-B0 ({tag})")
    print(f"{'='*60}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"effb0_affectnet_{tag}.pth")

    if os.path.exists(save_path):
        print(f"  Already exists: {save_path}")
        return

    if n_pretrain_classes == 4:
        train_ds = AffectNet4ClsDataset(AFFECTNET_ROOT, "train", get_transforms("train"))
        val_ds = AffectNet4ClsDataset(AFFECTNET_ROOT, "val", get_transforms("val"))
        n_classes = 4
    else:
        train_ds = datasets.ImageFolder(
            os.path.join(AFFECTNET_ROOT, "train"), get_transforms("train"))
        val_ds = datasets.ImageFolder(
            os.path.join(AFFECTNET_ROOT, "val"), get_transforms("val"))
        n_classes = len(train_ds.classes)

    print(f"  Data: {len(train_ds)} train, {len(val_ds)} val, {n_classes} classes")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, n_classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_ds.targets, n_classes), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                all_preds.extend(model(images).argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch}: val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("classifier")}
            torch.save(state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Done. Best val F1: {best_f1:.3f}")


# ============================================================
# BUILD MODEL
# ============================================================
def build_effb0(pretrain_tag=None):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))

    if pretrain_tag:
        path = os.path.join(MODEL_DIR, f"effb0_affectnet_{pretrain_tag}.pth")
        if os.path.exists(path):
            state = torch.load(path, map_location=DEVICE)
            model_dict = model.state_dict()
            matched = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(matched)
            model.load_state_dict(model_dict, strict=False)
            print(f"    Loaded {pretrain_tag} pretrained ({len(matched)} params)")

    n_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i < int(n_blocks * 0.6):
            for p in block.parameters():
                p.requires_grad = False

    return model.to(DEVICE)


# ============================================================
# 5-FOLD CV
# ============================================================
def train_one_fold(train_idx, val_idx, dataset, fold_idx, pretrain_tag=None):
    train_ds = SubsetWithTransform(dataset, train_idx, get_transforms("train"))
    val_ds = SubsetWithTransform(dataset, val_idx, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=16,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = build_effb0(pretrain_tag=pretrain_tag)

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_ds.targets, NUM_CLASSES), label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-3)

    epochs = 80
    warmup = 5
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
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


def run_cv(pretrain_tag=None):
    label = pretrain_tag if pretrain_tag else "no_pretrain"
    print(f"\n{'='*60}")
    print(f"  EfficientNet-B0 5-Fold CV ({label})")
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
        f1, acc, pc = train_one_fold(train_idx.tolist(), val_idx.tolist(),
                                      dataset, fold_idx, pretrain_tag)
        fold_f1s.append(f1)
        fold_accs.append(acc)
        fold_per_class.append(pc.tolist())

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    mean_per_class = np.mean(fold_per_class, axis=0)
    std_per_class = np.std(fold_per_class, axis=0)

    print(f"\n  {label}: F1 = {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  Folds: {[f'{f:.3f}' for f in fold_f1s]}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:<12}: {mean_per_class[i]:.3f} ± {std_per_class[i]:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"effb0_{label}_cv.json"), "w") as f:
        json.dump({
            "label": label, "mean_f1": float(mean_f1), "std_f1": float(std_f1),
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
                        choices=["pretrain_8cls", "pretrain_4cls", "cv_compare", "all"])
    args = parser.parse_args()

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.phase == "pretrain_8cls":
        pretrain_affectnet(n_pretrain_classes=8)

    elif args.phase == "pretrain_4cls":
        pretrain_affectnet(n_pretrain_classes=4)

    elif args.phase == "cv_compare":
        f1_no, std_no = run_cv(pretrain_tag=None)
        f1_8, std_8 = run_cv(pretrain_tag="8cls")
        f1_4, std_4 = run_cv(pretrain_tag="4cls")

        print(f"\n{'='*60}")
        print(f"  TRANSFER LEARNING COMPARISON")
        print(f"{'='*60}")
        print(f"  No pretrain:      {f1_no:.3f} ± {std_no:.3f}")
        print(f"  AffectNet 8-cls:  {f1_8:.3f} ± {std_8:.3f}  ({f1_8-f1_no:+.3f})")
        print(f"  AffectNet 4-cls:  {f1_4:.3f} ± {std_4:.3f}  ({f1_4-f1_no:+.3f})")

    elif args.phase == "all":
        pretrain_affectnet(n_pretrain_classes=8)
        pretrain_affectnet(n_pretrain_classes=4)

        f1_no, std_no = run_cv(pretrain_tag=None)
        f1_8, std_8 = run_cv(pretrain_tag="8cls")
        f1_4, std_4 = run_cv(pretrain_tag="4cls")

        print(f"\n{'='*60}")
        print(f"  TRANSFER LEARNING COMPARISON")
        print(f"{'='*60}")
        print(f"  No pretrain:      {f1_no:.3f} ± {std_no:.3f}")
        print(f"  AffectNet 8-cls:  {f1_8:.3f} ± {std_8:.3f}  ({f1_8-f1_no:+.3f})")
        print(f"  AffectNet 4-cls:  {f1_4:.3f} ± {std_4:.3f}  ({f1_4-f1_no:+.3f})")


if __name__ == "__main__":
    main()