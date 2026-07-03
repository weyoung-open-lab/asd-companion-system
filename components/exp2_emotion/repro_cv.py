# -*- coding: utf-8 -*-
"""
Exp2 第0步：纯复现（faithful reproduction of confidence_analysis.py）
=====================================================================
忠实沿用原协议（一字不改训练数学）：
  - 数据：干净 FERAC 770（4类），train+test 合并成池 → 5折 StratifiedKFold
  - backbone：effb0_affectnet_8cls.pth（strict=False）+ Linear(1280,4)，冻结前60% block
  - loss：FocalLoss(alpha=class_weights, gamma=2, label_smoothing=0.2)
  - Mixup(alpha=0.2, p=0.5)、WeightedRandomSampler
  - AdamW(lr=1e-4, wd=1e-3)、warmup5 + cosine、80 epoch、patience=15
  - 单种子 42（与原始 0.707±0.051 一致；±0.051 是跨折 std）

相比原脚本只改两点（不动训练逻辑）：
  1) 路径走 common/paths.py（不硬编码）
  2) 额外保存 OOF logits（供第1步 bootstrap / 第2步温度缩放）

输出：
  artifacts/exp2/repro_cv_results.json   每折/聚合 Macro-F1、acc、mean±std、config
  artifacts/exp2/oof/oof_logits.npz      OOF 原始 logits(770,4)+labels+fold
  artifacts/exp2/oof/oof_paths.json      每个 OOF 样本的图片路径（与 npz 行对齐）

用法：
  python repro_cv.py                 # 完整 5 折，种子 42
  python repro_cv.py --smoke         # 1 折 × 2 epoch 自检（验证管线/GPU）
  python repro_cv.py --seed 123      # 换种子（稳健性扩展用）
"""
import os, sys, json, math, random, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.paths import FERAC_ROOT, MODELS_DIR, EXP2_OUTPUT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
N_FOLDS = 5
PRETRAIN_CKPT = MODELS_DIR / "effb0_affectnet_8cls.pth"


# ============ 数据（照搬 confidence_analysis.py） ============
class FERACDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.samples = []
        c2i = {c: i for i, c in enumerate(CLASSES)}
        for split in ["train", "test", "训练", "测试"]:
            sd = os.path.join(root, split)
            if not os.path.exists(sd):
                continue
            for cf in sorted(os.listdir(sd)):
                cn = CN_TO_EN.get(cf, cf)
                if cn not in c2i:
                    continue
                cd = os.path.join(sd, cf)
                if not os.path.isdir(cd):
                    continue
                for fn in sorted(os.listdir(cd)):
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append((os.path.join(cd, fn), c2i[cn]))
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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


# ============ utils（照搬） ============
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))
    if os.path.exists(PRETRAIN_CKPT):
        state = torch.load(PRETRAIN_CKPT, map_location=DEVICE)
        md = model.state_dict()
        matched = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
        md.update(matched)
        model.load_state_dict(md, strict=False)
    n_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i < int(n_blocks * 0.6):
            for p in block.parameters():
                p.requires_grad = False
    return model.to(DEVICE)


# ============ 单折训练（照搬训练数学） ============
def train_one_fold(dataset, train_idx, val_idx, fold_idx, epochs, patience, mixup_alpha):
    train_ds = TransformSubset(dataset, train_idx, get_train_transform())
    val_ds = TransformSubset(dataset, val_idx, get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=16, sampler=create_sampler(train_ds.targets),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model()
    class_weights = compute_class_weights(train_ds.targets)
    criterion = FocalLoss(alpha=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-3)
    warmup = 5

    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_f1, patience_ctr, best_state, best_acc = 0.0, 0, None, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if random.random() < 0.5:
                mixed, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
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
                preds.extend(model(images.to(DEVICE)).argmax(1).cpu().numpy())
                lbls.extend(labels.numpy())
        f1 = f1_score(lbls, preds, average='macro')
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1
            best_acc = accuracy_score(lbls, preds)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    print(f"    Fold {fold_idx}: best Macro-F1={best_f1:.4f} acc={best_acc:.4f}")

    # 用 best 模型采 OOF logits（pre-softmax，供温度缩放）
    model.load_state_dict(best_state)
    model.eval()
    oof_logits, oof_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(DEVICE)).cpu().numpy()
            oof_logits.append(logits)
            oof_labels.append(labels.numpy())
    oof_logits = np.concatenate(oof_logits)
    oof_labels = np.concatenate(oof_labels)
    return best_f1, best_acc, oof_logits, oof_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--smoke", action="store_true", help="1 fold x 2 epochs sanity check")
    args = ap.parse_args()

    epochs = 2 if args.smoke else args.epochs
    max_folds = 1 if args.smoke else N_FOLDS

    # determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  FERAC_ROOT: {FERAC_ROOT}")
    print(f"  pretrain backbone: {PRETRAIN_CKPT.name} (exists={os.path.exists(PRETRAIN_CKPT)})")

    dataset = FERACDataset(FERAC_ROOT)
    targets = np.array(dataset.targets)
    print(f"  pooled images: {len(dataset)}  per-class: "
          + ", ".join(f"{c}={int((targets==i).sum())}" for i, c in enumerate(CLASSES)))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold_f1s, fold_accs = [], []
    # OOF 容器：用 sample index 对齐到原始 dataset 顺序
    oof_logits_all = np.zeros((len(dataset), NUM_CLASSES), dtype=np.float32)
    oof_label_all = np.full(len(dataset), -1, dtype=np.int64)
    oof_fold_all = np.full(len(dataset), -1, dtype=np.int64)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(range(len(dataset)), targets), 1):
        if fold_idx > max_folds:
            break
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} (train={len(tr_idx)}, val={len(va_idx)}) ---")
        set_seed(args.seed + fold_idx)
        f1, acc, logits, labels = train_one_fold(
            dataset, tr_idx.tolist(), va_idx.tolist(), fold_idx,
            epochs, args.patience, args.mixup_alpha)
        fold_f1s.append(float(f1))
        fold_accs.append(float(acc))
        oof_logits_all[va_idx] = logits
        oof_label_all[va_idx] = labels
        oof_fold_all[va_idx] = fold_idx

    mean_f1, std_f1 = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    print(f"\n  ===== Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}  (folds: {[round(x,4) for x in fold_f1s]}) =====")
    print(f"  ===== Accuracy = {mean_acc:.4f} ± {std_acc:.4f} =====")

    EXP2_OUTPUT.mkdir(parents=True, exist_ok=True)
    out = {
        "step": "0_reproduction",
        "faithful_source": "confidence_analysis.py",
        "seed": args.seed,
        "n_folds": N_FOLDS if not args.smoke else max_folds,
        "smoke": args.smoke,
        "config": {
            "backbone": "effb0_affectnet_8cls.pth (strict=False) + Linear(1280,4), freeze first 60% blocks",
            "loss": "FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.2)",
            "mixup_alpha": args.mixup_alpha, "mixup_p": 0.5,
            "optimizer": "AdamW(lr=1e-4, weight_decay=1e-3)",
            "scheduler": "warmup5 + cosine", "epochs": epochs, "patience": args.patience,
            "batch_size": 16, "sampler": "WeightedRandomSampler",
            "img": "train RandomCrop224<-Resize256; val Resize224; ImageNet norm",
            "cudnn_deterministic": True,
        },
        "per_fold_macro_f1": fold_f1s,
        "per_fold_accuracy": fold_accs,
        "mean_macro_f1": mean_f1, "std_macro_f1": std_f1,
        "mean_accuracy": mean_acc, "std_accuracy": std_acc,
        "target_repro": "0.707 ± 0.051 (band 0.65-0.76)",
        "classes": CLASSES,
        "pooled_total": int(len(dataset)),
        "per_class_count": {c: int((targets == i).sum()) for i, c in enumerate(CLASSES)},
    }
    res_path = EXP2_OUTPUT / ("repro_cv_results_smoke.json" if args.smoke else "repro_cv_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  saved: {res_path}")

    if not args.smoke:
        oof_dir = EXP2_OUTPUT / "oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        np.savez(oof_dir / "oof_logits.npz",
                 logits=oof_logits_all, labels=oof_label_all, fold=oof_fold_all,
                 classes=np.array(CLASSES))
        with open(oof_dir / "oof_paths.json", "w", encoding="utf-8") as f:
            json.dump([p for p, _ in dataset.samples], f, ensure_ascii=False, indent=1)
        covered = int((oof_fold_all >= 0).sum())
        print(f"  saved OOF logits: {oof_dir/'oof_logits.npz'}  (covered {covered}/{len(dataset)} samples)")


if __name__ == "__main__":
    main()
