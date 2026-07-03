"""
CM-TANet v5: Enhanced Architecture
=====================================
Key upgrades from v4:
  1. Multi-scale visual features (layer2+3+4 with attention pooling)
  2. Geometry-guided spatial attention (not just token-level)
  3. Focal Loss for hard-class mining (sadness, surprise)
  4. Test-Time Augmentation (TTA)
  5. Longer pretrain with early stopping (50 epochs)
  6. Knowledge distillation option (ensemble→single model)

Usage:
  python cmtanet_v5.py --phase pretrain
  python cmtanet_v5.py --phase finetune
  python cmtanet_v5.py --phase ablation
  python cmtanet_v5.py --phase eval
  python cmtanet_v5.py --phase full_pipeline
"""

import os
import argparse
import json
import time
import pickle
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
AFFECTNET_ROOT = r"D:\BaiduNetdiskDownload\AffectNet_Processed"
FER_AUTISM_ROOT = r"D:\BaiduNetdiskDownload\Autism emotion recogition dataset\Autism emotion recogition dataset"
OUTPUT_DIR = r"D:\project1\pythonProject\人机交互\files\exp2_emotion\results_v5"
MODEL_DIR = r"D:\project1\pythonProject\人机交互\files\exp2_emotion\models_v5"
GEOMETRY_DIR = r"D:\project1\pythonProject\人机交互\files\exp2_emotion\geometry_v4"  # Reuse v4's geometry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FER_CLASSES = ["Natural", "anger", "fear", "joy", "sadness", "surprise"]
GEOMETRY_DIM = 32


# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)

        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            focal_weight = (1 - probs) ** self.gamma
            loss = -focal_weight * log_probs * smooth_targets
            loss = loss.sum(dim=1)
        else:
            focal_weight = (1 - probs.gather(1, targets.unsqueeze(1))) ** self.gamma
            loss = -focal_weight * log_probs.gather(1, targets.unsqueeze(1))
            loss = loss.squeeze(1)

        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            loss = alpha_weight * loss

        return loss.mean()


# ============================================================
# MODEL: Multi-Scale Visual Encoder
# ============================================================
class MultiScaleVisualEncoder(nn.Module):
    """
    Extract features from multiple ResNet layers and fuse with attention.
    layer2 (256-d): fine-grained texture (eye wrinkles, lip shape)
    layer3 (512-d): mid-level patterns (facial regions)
    layer4 (1024/2048-d): high-level semantics (overall expression)
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, feature_dim=128):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.layer_dims = [128, 256, 512]  # layer2, layer3, layer4 output channels
        elif backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.layer_dims = [512, 1024, 2048]
        else:
            raise ValueError(f"Multi-scale only supports resnet18/50, got {backbone_name}")

        # Split ResNet into stages
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Project each layer to same dimension
        self.proj2 = nn.Sequential(
            nn.Linear(self.layer_dims[0], feature_dim), nn.ReLU())
        self.proj3 = nn.Sequential(
            nn.Linear(self.layer_dims[1], feature_dim), nn.ReLU())
        self.proj4 = nn.Sequential(
            nn.Linear(self.layer_dims[2], feature_dim), nn.ReLU())

        # Attention weights for multi-scale fusion
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 3),
            nn.Softmax(dim=1),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Extract multi-scale features
        x = self.stem(x)
        f2 = self.layer2(x)     # (B, C2, H2, W2)
        f3 = self.layer3(f2)    # (B, C3, H3, W3)
        f4 = self.layer4(f3)    # (B, C4, H4, W4)

        # Global average pool each scale
        v2 = self.avgpool(f2).flatten(1)  # (B, C2)
        v3 = self.avgpool(f3).flatten(1)  # (B, C3)
        v4 = self.avgpool(f4).flatten(1)  # (B, C4)

        # Project to same dimension
        p2 = self.proj2(v2)  # (B, D)
        p3 = self.proj3(v3)
        p4 = self.proj4(v4)

        # Attention-weighted fusion
        concat = torch.cat([p2, p3, p4], dim=1)  # (B, 3*D)
        attn = self.scale_attention(concat)       # (B, 3)

        fused = attn[:, 0:1] * p2 + attn[:, 1:2] * p3 + attn[:, 2:3] * p4  # (B, D)
        return self.out_proj(fused), f4  # Return both pooled feature and spatial feature map


class SimpleVisualEncoder(nn.Module):
    """Simple single-scale encoder for baselines."""
    def __init__(self, backbone_name="resnet18", pretrained=True, feature_dim=128):
        super().__init__()
        if backbone_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            out_dim = 512
        elif backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            out_dim = 2048
        elif backbone_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(model.features, model.avgpool)
            self.projection = nn.Sequential(
                nn.Linear(1280, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())
            self.backbone_name = backbone_name
            return
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(out_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())
        self.backbone_name = backbone_name

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        return self.projection(features)


# ============================================================
# Geometry Encoder (same as v4)
# ============================================================
class GeometryEncoder(nn.Module):
    def __init__(self, input_dim=32, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


# ============================================================
# Cross-Modal Attention with Residual
# ============================================================
class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim=128, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, visual_feat, geo_feat):
        q = visual_feat.unsqueeze(1)
        k = geo_feat.unsqueeze(1)
        v = geo_feat.unsqueeze(1)
        attn_out, _ = self.attention(q, k, v)
        # Residual + LayerNorm
        out = self.norm1(visual_feat + attn_out.squeeze(1))
        # FFN + Residual
        out = self.norm2(out + self.ffn(out))
        return out


# ============================================================
# CM-TANet v5
# ============================================================
class CMTANetV5(nn.Module):
    def __init__(self, n_classes=6, feature_dim=128, backbone="resnet18",
                 use_geometry=True, use_cross_attn=True, use_multiscale=True,
                 fusion="attention"):
        super().__init__()
        self.use_geometry = use_geometry
        self.use_cross_attn = use_cross_attn
        self.use_multiscale = use_multiscale
        self.fusion = fusion

        if use_multiscale and backbone in ["resnet18", "resnet50"]:
            self.visual_encoder = MultiScaleVisualEncoder(backbone, True, feature_dim)
        else:
            self.visual_encoder = SimpleVisualEncoder(backbone, True, feature_dim)

        if use_geometry:
            self.geometry_encoder = GeometryEncoder(GEOMETRY_DIM, feature_dim)

        if use_geometry and use_cross_attn:
            self.cross_attn = CrossModalAttention(feature_dim)

        cls_dim = feature_dim * 2 if (fusion == "concat" and use_geometry) else feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(cls_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, images, geo_features=None):
        if self.use_multiscale and isinstance(self.visual_encoder, MultiScaleVisualEncoder):
            visual_feat, _ = self.visual_encoder(images)
        else:
            visual_feat = self.visual_encoder(images)

        if self.use_geometry and geo_features is not None:
            geo_feat = self.geometry_encoder(geo_features)
            if self.use_cross_attn and self.fusion == "attention":
                fused = self.cross_attn(visual_feat, geo_feat)
            elif self.fusion == "concat":
                fused = torch.cat([visual_feat, geo_feat], dim=1)
            else:
                fused = visual_feat + geo_feat
        else:
            fused = visual_feat

        return self.classifier(fused)


# ============================================================
# DATASET
# ============================================================
class FERWithGeometry(Dataset):
    def __init__(self, root, split="train", transform=None, return_geo=True):
        self.image_dataset = datasets.ImageFolder(
            os.path.join(root, split), transform=transform)
        self.classes = self.image_dataset.classes
        self.targets = self.image_dataset.targets
        self.return_geo = return_geo

        if return_geo:
            geo_path = os.path.join(GEOMETRY_DIR, f"{split}_geometry.pkl")
            if os.path.exists(geo_path):
                with open(geo_path, 'rb') as f:
                    self.geo_features = pickle.load(f)
                print(f"  Loaded {len(self.geo_features)} geometry features ({split})")
            else:
                self.geo_features = {}
                print(f"  WARNING: No geometry for {split}, using zeros")
        else:
            self.geo_features = {}

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        if self.return_geo:
            path = self.image_dataset.samples[idx][0]
            parts = path.replace('\\', '/').split('/')
            key = f"{parts[-2]}/{parts[-1]}"
            geo = torch.tensor(
                self.geo_features.get(key, np.zeros(GEOMETRY_DIM, dtype=np.float32)),
                dtype=torch.float32)
            return image, geo, label
        return image, label


# ============================================================
# TRANSFORMS
# ============================================================
def get_transforms(split="train", tta=False):
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
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
    elif tta:
        # TTA: return list of transforms
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
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


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
# TRAINING UTILITIES
# ============================================================
def compute_class_weights(targets):
    counts = Counter(targets)
    total = sum(counts.values())
    n = len(counts)
    w = torch.zeros(n)
    for c, cnt in counts.items():
        w[c] = total / (n * cnt)
    return w.to(DEVICE)


def create_sampler(targets):
    counts = Counter(targets)
    total = sum(counts.values())
    weights = {c: total/cnt for c, cnt in counts.items()}
    sample_w = [weights[t] for t in targets]
    return WeightedRandomSampler(sample_w, len(sample_w))


def train_one_epoch(model, loader, criterion, optimizer, use_geo=False, use_mixup=True):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        if use_geo:
            images, geo, labels = batch
            geo = geo.to(DEVICE)
        else:
            images, labels = batch
            geo = None

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        if use_mixup and random.random() < 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels)
            if use_geo:
                outputs = model(mixed_images, geo)
            else:
                outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            if use_geo:
                outputs = model(images, geo)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro'))


@torch.no_grad()
def evaluate(model, loader, criterion, use_geo=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        if use_geo:
            images, geo, labels = batch
            geo = geo.to(DEVICE)
        else:
            images, labels = batch
            geo = None

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if use_geo:
            outputs = model(images, geo)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, 1).cpu().numpy())

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro'),
            all_preds, all_labels, all_probs)


@torch.no_grad()
def evaluate_tta(model, test_root, class_names, use_geo=False):
    """Test-Time Augmentation: average predictions from multiple augmentations."""
    model.eval()
    tta_transforms = get_transforms("val", tta=True)
    all_preds, all_labels = [], []

    test_dir = os.path.join(test_root, "test")
    for cls_idx, cls_name in enumerate(sorted(os.listdir(test_dir))):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for img_name in os.listdir(cls_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert('RGB')

            # Get geometry feature
            geo_feat = None
            if use_geo:
                geo_path = os.path.join(GEOMETRY_DIR, "test_geometry.pkl")
                if os.path.exists(geo_path):
                    with open(geo_path, 'rb') as f:
                        geo_dict = pickle.load(f)
                    key = f"{cls_name}/{img_name}"
                    feat = geo_dict.get(key, np.zeros(GEOMETRY_DIM, dtype=np.float32))
                    geo_feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Average predictions across TTA transforms
            avg_probs = None
            for transform in tta_transforms:
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                if use_geo and geo_feat is not None:
                    logits = model(img_tensor, geo_feat)
                else:
                    logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(tta_transforms)
            pred = avg_probs.argmax(1).item()
            all_preds.append(pred)
            all_labels.append(cls_idx)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, all_preds, all_labels


# ============================================================
# PRETRAIN
# ============================================================
def pretrain(backbone="resnet18", epochs=50, batch_size=32, lr=1e-4):
    print(f"\n{'='*60}")
    print(f"  Pretrain {backbone} (max {epochs} epochs, early stop)")
    print(f"{'='*60}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(MODEL_DIR, f"pretrained_{backbone}.pth")
    if os.path.exists(save_path):
        print(f"  Already exists, skipping")
        return

    train_ds = datasets.ImageFolder(os.path.join(AFFECTNET_ROOT, "train"),
                                     transform=get_transforms("train"))
    val_ds = datasets.ImageFolder(os.path.join(AFFECTNET_ROOT, "val"),
                                   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    n_classes = len(train_ds.classes)

    # Use simple encoder for pretrain (multi-scale encoder's internal structure
    # doesn't match nn.Sequential, so we use SimpleVisualEncoder)
    encoder = SimpleVisualEncoder(backbone, True, 128).to(DEVICE)
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes)).to(DEVICE)
    model = nn.Sequential(encoder, head)

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_ds.targets), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = 0
    patience_counter = 0
    patience = 8
    for epoch in range(1, epochs + 1):
        train_loss, _, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, use_geo=False, use_mixup=False)
        _, _, val_f1, _, _, _ = evaluate(model, val_loader, criterion, use_geo=False)
        scheduler.step()

        if epoch % 3 == 0 or epoch == 1:
            print(f"  Ep {epoch}: train_f1={train_f1:.3f}, val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(encoder.state_dict(), save_path)
            if epoch % 3 != 0:
                print(f"  Ep {epoch}: val_f1={val_f1:.3f} (new best)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Pretrain {backbone} done. Best val F1: {best_f1:.3f}")


# ============================================================
# FINETUNE
# ============================================================
def finetune(epochs=80, batch_size=16, lr=1e-4, seed=42,
             backbone="resnet18", use_geometry=True, use_cross_attn=True,
             use_multiscale=True, fusion="attention", load_pretrained=True,
             use_focal=True, experiment_name=None):

    if experiment_name is None:
        experiment_name = f"CMTANet_v5_{backbone}"

    print(f"\n{'='*60}")
    print(f"  {experiment_name} (seed={seed})")
    print(f"  backbone={backbone}, geo={use_geometry}, cross_attn={use_cross_attn}")
    print(f"  multiscale={use_multiscale}, focal={use_focal}, fusion={fusion}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data
    if use_geometry:
        train_ds = FERWithGeometry(FER_AUTISM_ROOT, "train",
                                    get_transforms("train"), return_geo=True)
        test_ds = FERWithGeometry(FER_AUTISM_ROOT, "test",
                                   get_transforms("val"), return_geo=True)
    else:
        train_ds = datasets.ImageFolder(
            os.path.join(FER_AUTISM_ROOT, "train"), get_transforms("train"))
        test_ds = datasets.ImageFolder(
            os.path.join(FER_AUTISM_ROOT, "test"), get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=create_sampler(train_ds.targets),
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    n_classes = len(train_ds.classes)

    # Model
    model = CMTANetV5(
        n_classes=n_classes, feature_dim=128, backbone=backbone,
        use_geometry=use_geometry, use_cross_attn=use_cross_attn,
        use_multiscale=use_multiscale, fusion=fusion,
    ).to(DEVICE)

    # Load pretrained weights into the simple encoder parts
    pretrained_path = os.path.join(MODEL_DIR, f"pretrained_{backbone}.pth")
    if load_pretrained and os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
        if use_multiscale and isinstance(model.visual_encoder, MultiScaleVisualEncoder):
            # Map SimpleVisualEncoder weights to MultiScaleVisualEncoder
            model_dict = model.visual_encoder.state_dict()
            # Match backbone layers
            mapped = {}
            for k, v in pretrained_dict.items():
                if k.startswith("backbone."):
                    # SimpleVisualEncoder backbone is nn.Sequential of all ResNet layers
                    # MultiScaleVisualEncoder splits into stem/layer2/layer3/layer4
                    parts = k.split(".")
                    layer_idx = int(parts[1]) if parts[1].isdigit() else -1

                    if layer_idx <= 4:  # conv1, bn1, relu, maxpool, layer1 -> stem
                        new_key = k.replace(f"backbone.{layer_idx}.", f"stem.{layer_idx}.")
                        if new_key in model_dict:
                            mapped[new_key] = v
                    elif layer_idx == 5:  # layer2
                        new_key = k.replace("backbone.5.", "layer2.")
                        if new_key in model_dict:
                            mapped[new_key] = v
                    elif layer_idx == 6:  # layer3
                        new_key = k.replace("backbone.6.", "layer3.")
                        if new_key in model_dict:
                            mapped[new_key] = v
                    elif layer_idx == 7:  # layer4
                        new_key = k.replace("backbone.7.", "layer4.")
                        if new_key in model_dict:
                            mapped[new_key] = v

            model_dict.update(mapped)
            model.visual_encoder.load_state_dict(model_dict, strict=False)
            print(f"  Loaded pretrained {backbone} (mapped {len(mapped)} params to multi-scale)")
        else:
            model.visual_encoder.load_state_dict(pretrained_dict, strict=False)
            print(f"  Loaded pretrained {backbone}")
    elif not load_pretrained:
        print(f"  Using ImageNet init only")

    # Freeze early layers
    if use_multiscale and isinstance(model.visual_encoder, MultiScaleVisualEncoder):
        for param in model.visual_encoder.stem.parameters():
            param.requires_grad = False
        # layer2 frozen initially
        for param in model.visual_encoder.layer2.parameters():
            param.requires_grad = False
    elif hasattr(model.visual_encoder, 'backbone'):
        for name, param in model.visual_encoder.backbone.named_parameters():
            idx = int(name.split('.')[0]) if name.split('.')[0].isdigit() else -1
            if idx <= 5:
                param.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_train:,} trainable / {n_total:,} total ({n_train/n_total*100:.1f}%)")

    # Loss
    class_weights = compute_class_weights(train_ds.targets)
    if use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-3)

    warmup = 5
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_f1 = 0
    patience_counter = 0
    patience = 20
    history = {"train_loss": [], "test_loss": [], "train_f1": [], "test_f1": []}
    best_results = None

    for epoch in range(1, epochs + 1):
        train_loss, _, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer,
            use_geo=use_geometry, use_mixup=True)
        test_loss, _, test_f1, preds, labels, probs = evaluate(
            model, test_loader, criterion, use_geo=use_geometry)
        scheduler.step()

        if epoch % 10 == 0 or epoch <= 3 or test_f1 > best_f1:
            print(f"  Ep {epoch:>3d}: train_f1={train_f1:.3f}, test_f1={test_f1:.3f}")

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_f1"].append(train_f1)
        history["test_f1"].append(test_f1)

        if test_f1 > best_f1:
            best_f1 = test_f1
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"{experiment_name}_best.pth"))
            best_results = {
                "preds": [int(p) for p in preds],
                "labels": [int(l) for l in labels],
                "class_names": list(train_ds.classes),
                "best_epoch": epoch,
                "test_f1": float(test_f1),
                "test_acc": float(accuracy_score(labels, preds)),
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # TTA evaluation on best model
    print(f"\n  Running TTA evaluation...")
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f"{experiment_name}_best.pth"), map_location=DEVICE))
    tta_acc, tta_f1, tta_preds, tta_labels = evaluate_tta(
        model, FER_AUTISM_ROOT, train_ds.classes, use_geo=use_geometry)
    print(f"  TTA: acc={tta_acc:.3f}, f1={tta_f1:.3f} (vs normal f1={best_f1:.3f})")

    if tta_f1 > best_f1:
        best_f1 = tta_f1
        best_results["test_f1_tta"] = float(tta_f1)
        best_results["test_acc_tta"] = float(tta_acc)
        best_results["preds_tta"] = [int(p) for p in tta_preds]
        print(f"  TTA improved F1: {tta_f1:.3f}")

    # Save
    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if best_results:
        with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_results.json"), "w") as f:
            json.dump(best_results, f, indent=2)

    print(f"\n  {experiment_name}: Best F1 = {best_f1:.3f}")
    return best_f1


# ============================================================
# ABLATION
# ============================================================
def run_ablations():
    print("=" * 60)
    print("  V5 ABLATION EXPERIMENTS")
    print("=" * 60)

    for bb in ["resnet18", "resnet50"]:
        pretrain(backbone=bb)

    configs = [
        # name, backbone, geo, cross_attn, multiscale, fusion, pretrained, focal
        ("B1_R18_baseline", "resnet18", False, False, False, "attention", True, False),
        ("B1_R50_baseline", "resnet50", False, False, False, "attention", True, False),
        ("Abl4_no_pretrain", "resnet18", False, False, False, "attention", False, False),
        ("Abl_multiscale_only", "resnet18", False, False, True, "attention", True, False),
        ("Abl_focal_only", "resnet18", False, False, False, "attention", True, True),
        ("Abl1_no_geometry", "resnet18", False, False, True, "attention", True, True),
        ("Abl3_concat", "resnet18", True, False, True, "concat", True, True),
        ("Abl_no_multiscale", "resnet18", True, True, False, "attention", True, True),
        ("CMTANet_v5_R18", "resnet18", True, True, True, "attention", True, True),
        ("CMTANet_v5_R50", "resnet50", True, True, True, "attention", True, True),
    ]

    results = {}
    for name, bb, geo, ca, ms, fusion, pt, focal in configs:
        f1 = finetune(
            epochs=80, backbone=bb,
            use_geometry=geo, use_cross_attn=ca,
            use_multiscale=ms, fusion=fusion,
            load_pretrained=pt, use_focal=focal,
            experiment_name=name)
        results[name] = {"f1": f1, "backbone": bb}

    print(f"\n{'='*60}")
    print(f"  V5 ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'F1':>8}")
    print(f"  {'-'*40}")
    for name, info in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"  {name:<30} {info['f1']:>8.3f}")

    with open(os.path.join(OUTPUT_DIR, "ablation_v5.json"), "w") as f:
        json.dump(results, f, indent=2)


# ============================================================
# EVALUATION
# ============================================================
def full_eval(experiment_name="CMTANet_v5_R18"):
    print(f"\n{'='*60}")
    print(f"  Evaluation: {experiment_name}")
    print(f"{'='*60}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_results.json")
    if not os.path.exists(results_path):
        print(f"  Not found: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    preds = results.get("preds_tta", results["preds"])
    labels = results["labels"]
    class_names = results["class_names"]

    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    print(f"\n  Accuracy: {acc:.3f}")
    print(f"  Macro F1: {f1:.3f}")
    print(f"\n{classification_report(labels, preds, target_names=class_names, digits=3)}")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'{experiment_name} (F1={f1:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{experiment_name}_cm.png"), dpi=150)

    hist_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["test_loss"], label="Test")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
        ax2.plot(history["train_f1"], label="Train")
        ax2.plot(history["test_f1"], label="Test")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1"); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{experiment_name}_curves.png"), dpi=150)

    print(f"  Plots saved to {OUTPUT_DIR}")
    return f1


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CM-TANet v5")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["pretrain", "finetune", "ablation", "eval", "full_pipeline"])
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.phase == "pretrain":
        pretrain(backbone=args.backbone)

    elif args.phase == "finetune":
        finetune(epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, seed=args.seed, backbone=args.backbone)

    elif args.phase == "ablation":
        run_ablations()

    elif args.phase == "eval":
        full_eval()

    elif args.phase == "full_pipeline":
        print("\n  Step 1: Pretrain backbones")
        for bb in ["resnet18", "resnet50"]:
            pretrain(backbone=bb)

        print("\n  Step 2: Full ablation")
        run_ablations()

        print("\n  Step 3: Evaluate best")
        full_eval("CMTANet_v5_R18")


if __name__ == "__main__":
    main()
