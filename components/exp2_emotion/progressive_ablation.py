"""
渐进式组件叠加实验 + 不确定性估计
====================================
基于 EfficientNet-B0 + AffectNet 8cls pretrain + Focal Loss

逐步叠加:
  1. + Mixup
  2. + Label Smoothing 0.2
  3. + TTA (Test-Time Augmentation)
  4. + MC Dropout 不确定性估计

Usage:
  python progressive_ablation.py --phase all
  python progressive_ablation.py --phase uncertainty
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
FERAC_ROOT = r"./data/FERAC Dataset"
MODEL_DIR = r"./output/transfer_exp/models"
OUTPUT_DIR = r"./output/progressive_ablation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
N_FOLDS = 5
SEED = 42


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
        n_cls = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            smooth = torch.zeros_like(inputs)
            smooth.fill_(self.label_smoothing / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            focal_w = (1 - probs) ** self.gamma
            loss = (-focal_w * log_probs * smooth).sum(dim=1)
        else:
            focal_w = (1 - probs.gather(1, targets.unsqueeze(1))) ** self.gamma
            loss = (-focal_w * log_probs.gather(1, targets.unsqueeze(1))).squeeze(1)

        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss
        return loss.mean()


# ============================================================
# MIXUP
# ============================================================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return img, label


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


def get_tta_transforms():
    """3 transforms for Test-Time Augmentation."""
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]


# ============================================================
# UTILS
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
# MODEL
# ============================================================
def build_model():
    """EfficientNet-B0 + AffectNet 8cls pretrain."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))

    pretrained_path = os.path.join(MODEL_DIR, "effb0_affectnet_8cls.pth")
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=DEVICE)
        model_dict = model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)

    # Freeze early layers
    n_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i < int(n_blocks * 0.6):
            for p in block.parameters():
                p.requires_grad = False

    return model.to(DEVICE)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
@torch.no_grad()
def evaluate_standard(model, loader):
    """Standard evaluation."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        all_preds.extend(model(images).argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    return f1, acc, per_class, all_preds, all_labels


def evaluate_tta(model, dataset, indices):
    """Test-Time Augmentation: average predictions from 3 transforms."""
    model.eval()
    tta_transforms = get_tta_transforms()
    all_preds, all_labels = [], []

    for idx in indices:
        img, label = dataset[idx]  # PIL image
        all_labels.append(label)

        avg_probs = None
        for t in tta_transforms:
            img_tensor = t(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(img_tensor), dim=1)
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs += probs

        avg_probs /= len(tta_transforms)
        all_preds.append(avg_probs.argmax(1).item())

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    return f1, acc, per_class


def evaluate_mc_dropout(model, loader, n_forward=10):
    """
    Monte Carlo Dropout: run multiple forward passes with dropout enabled.
    Returns predictions, uncertainty scores, and per-sample confidence.
    """
    model.train()  # Keep dropout active
    all_probs_list = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        batch_probs = []
        for _ in range(n_forward):
            with torch.no_grad():
                probs = torch.softmax(model(images), dim=1)
            batch_probs.append(probs.cpu().numpy())
        all_probs_list.append(np.stack(batch_probs, axis=0))  # (n_forward, B, C)
        all_labels.extend(labels.numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs_list, axis=1)  # (n_forward, N, C)

    # Mean prediction
    mean_probs = all_probs.mean(axis=0)  # (N, C)
    preds = mean_probs.argmax(axis=1)

    # Uncertainty: predictive entropy
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)  # (N,)

    # Confidence: max probability
    confidence = mean_probs.max(axis=1)  # (N,)

    # Epistemic uncertainty: variance of predictions across forward passes
    epistemic = all_probs.var(axis=0).mean(axis=1)  # (N,)

    f1 = f1_score(all_labels, preds, average='macro')
    acc = accuracy_score(all_labels, preds)
    per_class = f1_score(all_labels, preds, average=None, labels=list(range(NUM_CLASSES)))

    return f1, acc, per_class, {
        "preds": preds.tolist(),
        "labels": all_labels,
        "confidence": confidence.tolist(),
        "entropy": entropy.tolist(),
        "epistemic": epistemic.tolist(),
    }


# ============================================================
# TRAINING ONE FOLD
# ============================================================
def train_one_fold(train_idx, val_idx, dataset, fold_idx,
                   use_mixup=False, label_smoothing=0.1, use_tta=False,
                   use_mc_dropout=False):

    train_ds = TransformSubset(dataset, train_idx, get_transforms("train"))
    val_ds = TransformSubset(dataset, val_idx, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=16,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()

    class_weights = compute_class_weights(train_ds.targets)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0,
                          label_smoothing=label_smoothing)

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
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if use_mixup and random.random() < 0.5:
                mixed, y_a, y_b, lam = mixup_data(images, labels)
                loss = mixup_criterion(criterion, model(mixed), y_a, y_b, lam)
            else:
                loss = criterion(model(images), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Standard eval for early stopping
        val_f1, val_acc, _, _, _ = evaluate_standard(model, val_loader)
        scheduler.step()

        if epoch % 15 == 0 or val_f1 > best_f1:
            print(f"    Fold {fold_idx} Ep {epoch:>3d}: val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"    Fold {fold_idx}: early stop ep {epoch}")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation with optional TTA and MC Dropout
    std_f1, std_acc, std_pc, _, _ = evaluate_standard(model, val_loader)

    result = {
        "standard_f1": std_f1,
        "standard_acc": std_acc,
        "standard_per_class": std_pc.tolist(),
    }

    if use_tta:
        tta_f1, tta_acc, tta_pc = evaluate_tta(model, dataset, val_idx)
        result["tta_f1"] = tta_f1
        result["tta_acc"] = tta_acc
        result["tta_per_class"] = tta_pc.tolist()
        print(f"    Fold {fold_idx}: standard_f1={std_f1:.3f}, tta_f1={tta_f1:.3f}")

    if use_mc_dropout:
        mc_f1, mc_acc, mc_pc, mc_data = evaluate_mc_dropout(model, val_loader)
        result["mc_f1"] = mc_f1
        result["mc_acc"] = mc_acc
        result["mc_per_class"] = mc_pc.tolist()
        result["mc_data"] = mc_data
        avg_conf = np.mean(mc_data["confidence"])
        avg_entropy = np.mean(mc_data["entropy"])
        print(f"    Fold {fold_idx}: mc_f1={mc_f1:.3f}, "
              f"avg_confidence={avg_conf:.3f}, avg_entropy={avg_entropy:.3f}")

    return result


# ============================================================
# 5-FOLD CV
# ============================================================
def run_cv(name, use_mixup=False, label_smoothing=0.1,
           use_tta=False, use_mc_dropout=False):

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  mixup={use_mixup}, ls={label_smoothing}, "
          f"tta={use_tta}, mc_dropout={use_mc_dropout}")
    print(f"{'='*60}")

    set_seed(SEED)
    dataset = FERACDataset(FERAC_ROOT)
    targets = np.array(dataset.targets)
    print(f"  Total: {len(dataset)} images")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), targets), 1):
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} ---")
        set_seed(SEED + fold_idx)
        result = train_one_fold(
            train_idx.tolist(), val_idx.tolist(), dataset, fold_idx,
            use_mixup=use_mixup, label_smoothing=label_smoothing,
            use_tta=use_tta, use_mc_dropout=use_mc_dropout)
        fold_results.append(result)

    # Aggregate
    std_f1s = [r["standard_f1"] for r in fold_results]
    std_pcs = [r["standard_per_class"] for r in fold_results]

    mean_f1 = np.mean(std_f1s)
    std_f1 = np.std(std_f1s)
    mean_pc = np.mean(std_pcs, axis=0)
    std_pc = np.std(std_pcs, axis=0)

    print(f"\n  {name} Standard: F1 = {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  Folds: {[f'{f:.3f}' for f in std_f1s]}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:<12}: {mean_pc[i]:.3f} ± {std_pc[i]:.3f}")

    summary = {
        "name": name,
        "standard_f1_mean": float(mean_f1),
        "standard_f1_std": float(std_f1),
        "standard_f1_folds": [float(f) for f in std_f1s],
        "standard_per_class_mean": mean_pc.tolist(),
        "standard_per_class_std": std_pc.tolist(),
    }

    if use_tta:
        tta_f1s = [r["tta_f1"] for r in fold_results]
        tta_mean = np.mean(tta_f1s)
        tta_std = np.std(tta_f1s)
        print(f"  {name} TTA:      F1 = {tta_mean:.3f} ± {tta_std:.3f}")
        summary["tta_f1_mean"] = float(tta_mean)
        summary["tta_f1_std"] = float(tta_std)
        summary["tta_f1_folds"] = [float(f) for f in tta_f1s]

    if use_mc_dropout:
        mc_f1s = [r["mc_f1"] for r in fold_results]
        mc_mean = np.mean(mc_f1s)
        mc_std = np.std(mc_f1s)
        all_conf = []
        all_entropy = []
        for r in fold_results:
            all_conf.extend(r["mc_data"]["confidence"])
            all_entropy.extend(r["mc_data"]["entropy"])
        print(f"  {name} MC:       F1 = {mc_mean:.3f} ± {mc_std:.3f}")
        print(f"  Avg confidence: {np.mean(all_conf):.3f}")
        print(f"  Avg entropy:    {np.mean(all_entropy):.3f}")
        summary["mc_f1_mean"] = float(mc_mean)
        summary["mc_f1_std"] = float(mc_std)
        summary["avg_confidence"] = float(np.mean(all_conf))
        summary["avg_entropy"] = float(np.mean(all_entropy))

        # Confidence threshold analysis
        print(f"\n  Confidence threshold analysis:")
        print(f"  {'Threshold':<12} {'Coverage':>10} {'F1 (filtered)':>15}")
        print(f"  {'-'*40}")
        all_labels = []
        all_preds = []
        for r in fold_results:
            all_labels.extend(r["mc_data"]["labels"])
            all_preds.extend(r["mc_data"]["preds"])

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_results = []
        for thresh in thresholds:
            mask = [c >= thresh for c in all_conf]
            n_kept = sum(mask)
            coverage = n_kept / len(mask)
            if n_kept > 0:
                filtered_preds = [p for p, m in zip(all_preds, mask) if m]
                filtered_labels = [l for l, m in zip(all_labels, mask) if m]
                filtered_f1 = f1_score(filtered_labels, filtered_preds, average='macro')
            else:
                filtered_f1 = 0
            print(f"  {thresh:<12.1f} {coverage:>9.1%} {filtered_f1:>15.3f}")
            threshold_results.append({
                "threshold": thresh, "coverage": float(coverage),
                "f1": float(filtered_f1)
            })
        summary["threshold_analysis"] = threshold_results

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{name}_cv.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["step1", "step2", "step3", "step4", "all"])
    args = parser.parse_args()

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    all_summaries = {}

    if args.phase in ["step1", "all"]:
        s = run_cv("S1_PT_Focal_Mixup",
                    use_mixup=True, label_smoothing=0.1)
        all_summaries["S1"] = s

    if args.phase in ["step2", "all"]:
        s = run_cv("S2_PT_Focal_Mixup_LS02",
                    use_mixup=True, label_smoothing=0.2)
        all_summaries["S2"] = s

    if args.phase in ["step3", "all"]:
        s = run_cv("S3_PT_Focal_Mixup_LS02_TTA",
                    use_mixup=True, label_smoothing=0.2, use_tta=True)
        all_summaries["S3"] = s

    if args.phase in ["step4", "all"]:
        s = run_cv("S4_Full_with_Uncertainty",
                    use_mixup=True, label_smoothing=0.2,
                    use_tta=True, use_mc_dropout=True)
        all_summaries["S4"] = s

    if args.phase == "all":
        print(f"\n{'='*60}")
        print(f"  PROGRESSIVE ABLATION SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Setting':<35} {'F1':>12}")
        print(f"  {'-'*50}")
        print(f"  {'Baseline (PT+Focal)':<35} {'0.703±0.049':>12}")
        for key, s in all_summaries.items():
            f1_str = f"{s['standard_f1_mean']:.3f}±{s['standard_f1_std']:.3f}"
            print(f"  {s['name']:<35} {f1_str:>12}")
            if "tta_f1_mean" in s:
                tta_str = f"{s['tta_f1_mean']:.3f}±{s['tta_f1_std']:.3f}"
                print(f"  {'  └─ with TTA':<35} {tta_str:>12}")
            if "mc_f1_mean" in s:
                mc_str = f"{s['mc_f1_mean']:.3f}±{s['mc_f1_std']:.3f}"
                print(f"  {'  └─ with MC Dropout':<35} {mc_str:>12}")


if __name__ == "__main__":
    main()
