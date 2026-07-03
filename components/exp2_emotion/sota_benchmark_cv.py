"""
SOTA Benchmark: 5-Fold Stratified CV on FERAC
================================================
9 models × 5 folds, report mean±std for each model.
No offline augmentation needed - online augmentation during training.

Usage:
  python sota_benchmark_cv.py --phase run_all
  python sota_benchmark_cv.py --phase run_one --model densenet121
  python sota_benchmark_cv.py --phase summary
"""

import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
FERAC_ROOT = r"./data/FERAC Dataset"  # 改成你的路径
OUTPUT_DIR = r"./output/sota_cv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
N_FOLDS = 5
SEED = 42


# ============================================================
# DATASET: Load all FERAC images into one pool
# ============================================================
class FERACDataset(torch.utils.data.Dataset):
    """Load all FERAC images (train+test combined) into one dataset."""
    def __init__(self, root, transform=None):
        self.samples = []  # [(path, label), ...]
        self.transform = transform
        self.classes = CLASSES
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
                        self.samples.append((
                            os.path.join(cls_dir, fname),
                            class_to_idx[cls_name]
                        ))

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class SubsetWithTransform(torch.utils.data.Dataset):
    """Subset that applies a different transform (train vs val)."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# TRANSFORMS
# ============================================================
def get_transforms(split="train", img_size=224):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
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
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ============================================================
# MODELS
# ============================================================
MODEL_CONFIGS = {
    "vgg16": {
        "lr": 5e-5, "epochs": 60, "weight_decay": 5e-4,
        "img_size": 224, "type": "CNN Classic",
    },
    "resnet18": {
        "lr": 1e-4, "epochs": 80, "weight_decay": 1e-3,
        "img_size": 224, "type": "CNN Classic",
    },
    "resnet50": {
        "lr": 5e-5, "epochs": 60, "weight_decay": 1e-3,
        "img_size": 224, "type": "CNN Classic",
    },
    "densenet121": {
        "lr": 1e-4, "epochs": 80, "weight_decay": 1e-3,
        "img_size": 224, "type": "CNN Classic",
    },
    "efficientnet_b0": {
        "lr": 1e-4, "epochs": 80, "weight_decay": 1e-3,
        "img_size": 224, "type": "CNN Light",
    },
    "mobilenet_v2": {
        "lr": 1e-4, "epochs": 80, "weight_decay": 1e-3,
        "img_size": 224, "type": "CNN Light",
    },
    "xception": {
        "lr": 5e-5, "epochs": 60, "weight_decay": 1e-3,
        "img_size": 299, "type": "CNN Other",
    },
    "swin_t": {
        "lr": 3e-5, "epochs": 60, "weight_decay": 5e-4,
        "img_size": 224, "type": "Transformer",
    },
    "vit_small": {
        "lr": 3e-5, "epochs": 60, "weight_decay": 5e-4,
        "img_size": 224, "type": "Transformer",
    },
}


def build_model(model_name):
    config = MODEL_CONFIGS[model_name]
    img_size = config.get("img_size", 224)

    if model_name == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        base.classifier[6] = nn.Linear(4096, NUM_CLASSES)
        for i, layer in enumerate(base.features):
            if i < 24:
                for p in layer.parameters():
                    p.requires_grad = False

    elif model_name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
        for name, p in base.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                p.requires_grad = False

    elif model_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, NUM_CLASSES))
        for name, p in base.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                p.requires_grad = False

    elif model_name == "densenet121":
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        base.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, NUM_CLASSES))
        for name, p in base.features.named_parameters():
            if "denseblock3" not in name and "denseblock4" not in name \
               and "transition3" not in name:
                p.requires_grad = False

    elif model_name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        base.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))
        n_blocks = len(base.features)
        for i, block in enumerate(base.features):
            if i < int(n_blocks * 0.6):
                for p in block.parameters():
                    p.requires_grad = False

    elif model_name == "mobilenet_v2":
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        base.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))
        n_blocks = len(base.features)
        for i, block in enumerate(base.features):
            if i < int(n_blocks * 0.6):
                for p in block.parameters():
                    p.requires_grad = False

    elif model_name == "xception":
        try:
            import timm
            base = timm.create_model('xception', pretrained=True, num_classes=NUM_CLASSES)
            frozen = 0
            total = len(list(base.parameters()))
            for p in base.parameters():
                if frozen < total * 0.6:
                    p.requires_grad = False
                    frozen += 1
        except ImportError:
            base = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
            base.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, NUM_CLASSES))
            base.aux_logits = False
            for name, p in base.named_parameters():
                if "Mixed_7" not in name and "fc" not in name:
                    p.requires_grad = False
            img_size = 299

    elif model_name == "swin_t":
        base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        base.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, NUM_CLASSES))
        for name, p in base.named_parameters():
            if "features.0" in name or "features.1" in name or \
               "features.2" in name or "features.3" in name:
                p.requires_grad = False

    elif model_name == "vit_small":
        try:
            import timm
            base = timm.create_model('vit_small_patch16_224', pretrained=True,
                                      num_classes=NUM_CLASSES)
            for name, p in base.named_parameters():
                if "blocks." in name:
                    block_idx = int(name.split("blocks.")[1].split(".")[0])
                    if block_idx < 8:
                        p.requires_grad = False
                elif "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
                    p.requires_grad = False
        except ImportError:
            base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            base.heads = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, NUM_CLASSES))
            for name, p in base.named_parameters():
                if "encoder.layers.encoder_layer_" in name:
                    layer_idx = int(name.split("encoder_layer_")[1].split(".")[0])
                    if layer_idx < 8:
                        p.requires_grad = False
                elif "conv_proj" in name or "class_token" in name:
                    p.requires_grad = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return base.to(DEVICE), img_size


# ============================================================
# TRAINING UTILITIES
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


def compute_class_weights(targets):
    counts = Counter(targets)
    total = sum(counts.values())
    n = len(counts)
    w = torch.ones(NUM_CLASSES).to(DEVICE)
    for c, cnt in counts.items():
        w[c] = total / (n * cnt)
    return w


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro')


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    return f1, acc, all_preds, all_labels, per_class_f1


# ============================================================
# SINGLE FOLD TRAINING
# ============================================================
def train_one_fold(model_name, train_indices, val_indices, dataset, fold_idx):
    """Train model on one fold, return val F1 and per-class F1."""
    config = MODEL_CONFIGS[model_name]
    img_size = config.get("img_size", 224)
    lr = config["lr"]
    epochs = config["epochs"]
    wd = config["weight_decay"]
    batch_size = 16

    # Create fold datasets with appropriate transforms
    train_ds = SubsetWithTransform(dataset, train_indices, get_transforms("train", img_size))
    val_ds = SubsetWithTransform(dataset, val_indices, get_transforms("val", img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Build fresh model for each fold
    model, _ = build_model(model_name)

    class_weights = compute_class_weights(train_ds.targets)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)

    warmup = 5
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_f1 = 0
    patience_counter = 0
    patience = 15
    best_per_class = None

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
        val_f1, val_acc, preds, labels, per_class_f1 = evaluate(model, val_loader)
        scheduler.step()

        if epoch % 15 == 0 or epoch <= 2 or val_f1 > best_f1:
            print(f"    Fold {fold_idx} Ep {epoch:>3d}: "
                  f"train_f1={train_f1:.3f}, val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_per_class = per_class_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Fold {fold_idx}: early stop at epoch {epoch}")
                break

    print(f"    Fold {fold_idx}: Best F1={best_f1:.3f}, Acc={best_acc:.3f}")
    return best_f1, best_acc, best_per_class


# ============================================================
# 5-FOLD CV FOR ONE MODEL
# ============================================================
def run_model_cv(model_name):
    """Run 5-fold stratified CV for one model."""
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} - 5-Fold Stratified CV")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    # Load all data
    dataset = FERACDataset(FERAC_ROOT)
    targets = np.array(dataset.targets)
    print(f"  Total images: {len(dataset)}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {(targets == i).sum()}")

    # 5-fold stratified split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s = []
    fold_accs = []
    fold_per_class = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), targets), 1):
        print(f"\n  --- Fold {fold_idx}/{N_FOLDS} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Show val class distribution
        val_targets = targets[val_idx]
        for i, cls in enumerate(CLASSES):
            print(f"    Val {cls}: {(val_targets == i).sum()}")

        set_seed(SEED + fold_idx)  # Different seed per fold for model init

        f1, acc, per_class = train_one_fold(
            model_name, train_idx.tolist(), val_idx.tolist(), dataset, fold_idx)
        fold_f1s.append(f1)
        fold_accs.append(acc)
        fold_per_class.append(per_class.tolist())

    # Summary
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_per_class = np.mean(fold_per_class, axis=0)
    std_per_class = np.std(fold_per_class, axis=0)

    print(f"\n{'='*60}")
    print(f"  {model_name}: {N_FOLDS}-Fold Results")
    print(f"{'='*60}")
    print(f"  Macro F1: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  Fold F1s: {[f'{f:.3f}' for f in fold_f1s]}")
    print(f"\n  Per-class F1 (mean ± std):")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:<12}: {mean_per_class[i]:.3f} ± {std_per_class[i]:.3f}")

    # Save
    results = {
        "model": model_name,
        "type": MODEL_CONFIGS[model_name]["type"],
        "n_folds": N_FOLDS,
        "fold_f1s": [float(f) for f in fold_f1s],
        "fold_accs": [float(a) for a in fold_accs],
        "fold_per_class": fold_per_class,
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "mean_per_class_f1": mean_per_class.tolist(),
        "std_per_class_f1": std_per_class.tolist(),
        "class_names": CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_cv.json"), "w") as f:
        json.dump(results, f, indent=2)

    return mean_f1, std_f1


# ============================================================
# SUMMARY
# ============================================================
def print_summary():
    print(f"\n{'='*60}")
    print(f"  FERAC 4-CLASS: 5-FOLD CV SUMMARY")
    print(f"{'='*60}")

    results = {}
    for name in MODEL_CONFIGS:
        path = os.path.join(OUTPUT_DIR, f"{name}_cv.json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)

    if not results:
        print("  No results found.")
        return

    sorted_models = sorted(results.items(), key=lambda x: x[1]["mean_f1"], reverse=True)

    print(f"\n  {'Rank':<5} {'Model':<20} {'Type':<15} {'F1':>12} {'Acc':>12}")
    print(f"  {'-'*68}")
    for rank, (name, data) in enumerate(sorted_models, 1):
        f1_str = f"{data['mean_f1']:.3f}±{data['std_f1']:.3f}"
        acc_str = f"{data['mean_acc']:.3f}±{data['std_acc']:.3f}"
        print(f"  {rank:<5} {name:<20} {data['type']:<15} {f1_str:>12} {acc_str:>12}")

    # Per-class breakdown for top 3
    print(f"\n  Per-class F1 (top 3 models):")
    print(f"  {'Model':<20} ", end="")
    for cls in CLASSES:
        print(f"  {cls:<14}", end="")
    print()
    print(f"  {'-'*80}")

    for name, data in sorted_models[:3]:
        print(f"  {name:<20} ", end="")
        for i in range(NUM_CLASSES):
            m = data['mean_per_class_f1'][i]
            s = data['std_per_class_f1'][i]
            print(f"  {m:.3f}±{s:.3f}   ", end="")
        print()

    with open(os.path.join(OUTPUT_DIR, "summary_cv.json"), "w") as f:
        json.dump({name: data for name, data in sorted_models}, f, indent=2)

    print(f"\n  Results saved to {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["run_all", "run_one", "summary"])
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.phase == "run_all":
        for name in MODEL_CONFIGS:
            run_model_cv(name)
        print_summary()

    elif args.phase == "run_one":
        if args.model is None:
            print("  Specify --model")
            return
        run_model_cv(args.model)

    elif args.phase == "summary":
        print_summary()


if __name__ == "__main__":
    main()
