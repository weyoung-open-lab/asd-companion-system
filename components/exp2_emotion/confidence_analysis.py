"""
标准Softmax置信度分析
======================
用训练好的模型（eval模式）做正常推理，
分析softmax输出的置信度分布和阈值效果。

Usage:
  python confidence_analysis.py
"""

import os
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
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
FERAC_ROOT = r"./data/FERAC Dataset"
MODEL_DIR = r"./output/transfer_exp/models"
OUTPUT_DIR = r"./output/confidence_analysis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
N_FOLDS = 5
SEED = 42


# ============================================================
# DATASET & TRANSFORMS (same as before)
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


def get_train_transform():
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

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


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
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_cls = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        smooth = torch.zeros_like(inputs)
        smooth.fill_(self.label_smoothing / (n_cls - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        focal_w = (1 - probs) ** self.gamma
        loss = (-focal_w * log_probs * smooth).sum(dim=1)
        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss
        return loss.mean()


# ============================================================
# MIXUP
# ============================================================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# MODEL
# ============================================================
def build_model():
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

    n_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i < int(n_blocks * 0.6):
            for p in block.parameters():
                p.requires_grad = False

    return model.to(DEVICE)


# ============================================================
# MAIN: Train 5 folds, collect confidence data
# ============================================================
def main():
    print("=" * 60)
    print("  Softmax Confidence Analysis (5-Fold CV)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    dataset = FERACDataset(FERAC_ROOT)
    targets = np.array(dataset.targets)
    print(f"  Total: {len(dataset)} images")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_preds = []
    all_labels = []
    all_confidences = []
    all_entropies = []
    all_correct = []
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), targets), 1):
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} ---")
        set_seed(SEED + fold_idx)

        train_ds = TransformSubset(dataset, train_idx.tolist(), get_train_transform())
        val_ds = TransformSubset(dataset, val_idx.tolist(), get_val_transform())

        train_loader = DataLoader(train_ds, batch_size=16,
                                  sampler=create_sampler(train_ds.targets),
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16,
                                shuffle=False, num_workers=4, pin_memory=True)

        model = build_model()
        class_weights = compute_class_weights(train_ds.targets)
        criterion = FocalLoss(alpha=class_weights)
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
        patience = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                if random.random() < 0.5:
                    mixed, y_a, y_b, lam = mixup_data(images, labels)
                    loss = mixup_criterion(criterion, model(mixed), y_a, y_b, lam)
                else:
                    loss = criterion(model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            preds, lbls = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    preds.extend(model(images).argmax(1).cpu().numpy())
                    lbls.extend(labels.numpy())
            f1 = f1_score(lbls, preds, average='macro')
            scheduler.step()

            if f1 > best_f1:
                best_f1 = f1
                patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 15:
                    break

        # Evaluate with best model - collect confidence
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds_batch = probs.argmax(axis=1)
                conf_batch = probs.max(axis=1)
                entropy_batch = -np.sum(probs * np.log(probs + 1e-10), axis=1)

                for i in range(len(labels)):
                    all_preds.append(int(preds_batch[i]))
                    all_labels.append(int(labels[i]))
                    all_confidences.append(float(conf_batch[i]))
                    all_entropies.append(float(entropy_batch[i]))
                    all_correct.append(int(preds_batch[i]) == int(labels[i]))

        fold_f1s.append(best_f1)
        print(f"  Fold {fold_idx}: F1={best_f1:.3f}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  CONFIDENCE ANALYSIS RESULTS")
    print(f"{'='*60}")

    mean_f1 = np.mean(fold_f1s)
    print(f"\n  Overall F1: {mean_f1:.3f} ± {np.std(fold_f1s):.3f}")
    print(f"  Avg confidence: {np.mean(all_confidences):.3f}")
    print(f"  Avg entropy: {np.mean(all_entropies):.3f}")

    # Correct vs Incorrect confidence
    correct_conf = [c for c, ok in zip(all_confidences, all_correct) if ok]
    wrong_conf = [c for c, ok in zip(all_confidences, all_correct) if not ok]
    print(f"\n  Correct predictions:   avg confidence = {np.mean(correct_conf):.3f}")
    print(f"  Incorrect predictions: avg confidence = {np.mean(wrong_conf):.3f}")
    print(f"  → Confidence gap: {np.mean(correct_conf) - np.mean(wrong_conf):.3f}")

    # Threshold analysis
    print(f"\n  Confidence Threshold Analysis:")
    print(f"  {'Threshold':<12} {'Coverage':>10} {'Accuracy':>10} {'F1':>10} {'Rejected':>10}")
    print(f"  {'-'*55}")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = []
    for thresh in thresholds:
        mask = [c >= thresh for c in all_confidences]
        n_kept = sum(mask)
        coverage = n_kept / len(mask)
        if n_kept > 0:
            f_preds = [p for p, m in zip(all_preds, mask) if m]
            f_labels = [l for l, m in zip(all_labels, mask) if m]
            f_f1 = f1_score(f_labels, f_preds, average='macro')
            f_acc = accuracy_score(f_labels, f_preds)
        else:
            f_f1, f_acc = 0, 0
        rejected = 1 - coverage
        print(f"  {thresh:<12.1f} {coverage:>9.1%} {f_acc:>10.3f} {f_f1:>10.3f} {rejected:>9.1%}")
        threshold_results.append({
            "threshold": thresh, "coverage": float(coverage),
            "accuracy": float(f_acc), "f1": float(f_f1),
            "rejected": float(rejected)
        })

    # Per-class confidence
    print(f"\n  Per-class Avg Confidence:")
    for i, cls in enumerate(CLASSES):
        cls_conf = [c for c, l in zip(all_confidences, all_labels) if l == i]
        cls_correct = [ok for ok, l in zip(all_correct, all_labels) if l == i]
        cls_acc = np.mean(cls_correct) if cls_correct else 0
        print(f"    {cls:<12}: conf={np.mean(cls_conf):.3f}, acc={cls_acc:.3f}")

    # Save
    report = {
        "fold_f1s": [float(f) for f in fold_f1s],
        "mean_f1": float(mean_f1),
        "avg_confidence": float(np.mean(all_confidences)),
        "avg_entropy": float(np.mean(all_entropies)),
        "correct_avg_conf": float(np.mean(correct_conf)),
        "wrong_avg_conf": float(np.mean(wrong_conf)),
        "threshold_analysis": threshold_results,
        "class_names": CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, "confidence_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
