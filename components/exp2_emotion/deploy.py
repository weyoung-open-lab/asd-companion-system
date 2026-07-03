# -*- coding: utf-8 -*-
"""
Exp2 第3步：训练并保存可部署 checkpoint
============================================
- 用原始 train(615) 训练最终模型（同忠实协议）。为拟合温度T+早停+选阈值，从615中分层
  carve 一个内部验证集 val_deploy（不碰 test155）。
- test(155) 只在最后报一次：混淆矩阵 + Macro-F1 + 校准后门控工作点。
- 保存 models/ferac_efficientnetb0_deploy.pth（state_dict + 温度T + 阈值 + 类别 + 预处理元信息）。

⚠️ 纪律：本部署分数是"成品在155测试集的具体表现"，不等于第0步CV的0.707（方法无偏估计）。

用法：python deploy.py [--seed 42] [--target_acc 0.90] [--val_frac 0.15]
"""
import os, sys, json, math, random, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.paths import FERAC_ROOT, MODELS_DIR, EXP2_OUTPUT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Natural", "anger", "fear", "joy"]
CN_TO_EN = {"自然": "Natural", "愤怒": "anger", "恐惧": "fear", "喜悦": "joy"}
NUM_CLASSES = 4
LABELS = list(range(4))
PRETRAIN_CKPT = MODELS_DIR / "effb0_affectnet_8cls.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]


def scan_split(root, split):
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = []
    sd = os.path.join(root, split)
    for cf in sorted(os.listdir(sd)):
        cn = CN_TO_EN.get(cf, cf)
        if cn not in c2i:
            continue
        cd = os.path.join(sd, cf)
        if not os.path.isdir(cd):
            continue
        for fn in sorted(os.listdir(cd)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                items.append((os.path.join(cd, fn), c2i[cn]))
    return items


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items; self.transform = transform
        self.targets = [l for _, l in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def train_tf():
    return transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1), transforms.RandomGrayscale(0.05),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15)])


def val_tf():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def sampler(targets):
    cnt = Counter(targets); tot = sum(cnt.values())
    w = {c: tot / n for c, n in cnt.items()}
    return WeightedRandomSampler([w[t] for t in targets], len(targets))


def class_weights(targets):
    cnt = Counter(targets); tot = sum(cnt.values()); n = len(cnt)
    w = torch.ones(NUM_CLASSES).to(DEVICE)
    for c, k in cnt.items(): w[c] = tot / (n * k)
    return w


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ls=0.2):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.ls = ls

    def forward(self, x, y):
        ncls = x.size(1); logp = F.log_softmax(x, 1); p = torch.exp(logp)
        sm = torch.zeros_like(x); sm.fill_(self.ls / (ncls - 1))
        sm.scatter_(1, y.unsqueeze(1), 1 - self.ls)
        loss = (-(1 - p) ** self.gamma * logp * sm).sum(1)
        if self.alpha is not None: loss = self.alpha.gather(0, y) * loss
        return loss.mean()


def mixup(x, y, a=0.2):
    lam = np.random.beta(a, a); idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))
    if os.path.exists(PRETRAIN_CKPT):
        st = torch.load(PRETRAIN_CKPT, map_location=DEVICE)
        md = m.state_dict(); matched = {k: v for k, v in st.items() if k in md and v.shape == md[k].shape}
        md.update(matched); m.load_state_dict(md, strict=False)
    nb = len(m.features)
    for i, blk in enumerate(m.features):
        if i < int(nb * 0.6):
            for p in blk.parameters(): p.requires_grad = False
    return m.to(DEVICE)


@torch.no_grad()
def collect_logits(model, loader):
    model.eval(); L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(DEVICE)).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(L), np.concatenate(Y)


def fit_T(logits, labels):
    lg = torch.tensor(logits, dtype=torch.float32); ly = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=200)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lg / logT.exp(), ly); loss.backward(); return loss
    opt.step(closure); return float(logT.exp().item())


def softmax_T(logits, T):
    z = logits / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_acc", type=float, default=0.90)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    set_seed(args.seed)

    train_items = scan_split(FERAC_ROOT, "train")
    test_items = scan_split(FERAC_ROOT, "test")
    print(f"  train={len(train_items)}  test={len(test_items)}  device={DEVICE}")

    # 分层 carve val_deploy（用于早停+温度+选阈值；绝不碰 test）
    tr_paths = [p for p, _ in train_items]; tr_y = [l for _, l in train_items]
    idx = np.arange(len(train_items))
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_frac, stratify=tr_y, random_state=args.seed)
    tr_sub = [train_items[i] for i in tr_idx]; va_sub = [train_items[i] for i in va_idx]
    print(f"  train_deploy={len(tr_sub)}  val_deploy={len(va_sub)}  (val per-class: "
          + ", ".join(f"{CLASSES[c]}={sum(1 for _,l in va_sub if l==c)}" for c in range(4)) + ")")

    tr_loader = DataLoader(ListDataset(tr_sub, train_tf()), batch_size=16,
                           sampler=sampler([l for _, l in tr_sub]), num_workers=0, pin_memory=True)
    va_loader = DataLoader(ListDataset(va_sub, val_tf()), batch_size=16, shuffle=False, num_workers=0)
    te_loader = DataLoader(ListDataset(test_items, val_tf()), batch_size=16, shuffle=False, num_workers=0)

    model = build_model()
    crit = FocalLoss(alpha=class_weights([l for _, l in tr_sub]))
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
    warmup = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warmup if ep < warmup
                                        else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, args.epochs - warmup))))
    best_f1, best_state, pc = 0.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            if random.random() < 0.5:
                mx, ya, yb, lam = mixup(x, y); loss = lam * crit(model(mx), ya) + (1 - lam) * crit(model(mx), yb)
            else:
                loss = crit(model(x), y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vl, vy = collect_logits(model, va_loader)
        f1 = f1_score(vy, vl.argmax(1), average="macro", labels=LABELS, zero_division=0)
        sched.step()
        if f1 > best_f1:
            best_f1 = f1; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pc = 0
        else:
            pc += 1
            if pc >= args.patience: break
    print(f"  best val_deploy Macro-F1 = {best_f1:.4f}")
    model.load_state_dict(best_state)

    # 温度T + 阈值：在 val_deploy 上拟合/选择
    vl, vy = collect_logits(model, va_loader)
    T = fit_T(vl, vy)
    vp = softmax_T(vl, T)
    conf = vp.max(1); pred = vp.argmax(1)
    chosen_t = None
    for t in np.round(np.arange(0.30, 0.951, 0.01), 2):
        keep = conf >= t
        if keep.sum() < 10: break
        if accuracy_score(vy[keep], pred[keep]) >= args.target_acc:
            chosen_t = float(t); break
    if chosen_t is None: chosen_t = 0.70
    print(f"  Temperature T={T:.4f}  chosen threshold={chosen_t} (selected on val_deploy)")

    # ===== 测试集 155：只此一次 =====
    tl, ty = collect_logits(model, te_loader)
    tp = softmax_T(tl, T)
    test_pred = tp.argmax(1); test_conf = tp.max(1)
    test_macro = f1_score(ty, test_pred, average="macro", labels=LABELS, zero_division=0)
    test_acc = accuracy_score(ty, test_pred)
    cm = confusion_matrix(ty, test_pred, labels=LABELS)
    perclass = f1_score(ty, test_pred, average=None, labels=LABELS, zero_division=0)
    # 门控工作点
    keep = test_conf >= chosen_t
    gated = {"threshold": chosen_t, "coverage": float(keep.mean()),
             "n_kept": int(keep.sum()), "rejection": float(1 - keep.mean()),
             "accuracy_on_kept": float(accuracy_score(ty[keep], test_pred[keep])) if keep.sum() else 0.0,
             "macro_f1_on_kept": float(f1_score(ty[keep], test_pred[keep], average="macro", labels=LABELS, zero_division=0)) if keep.sum() else 0.0}
    print(f"\n  ===== TEST(155) [deploy checkpoint, NOT the 0.707 CV number] =====")
    print(f"  Macro-F1={test_macro:.4f}  Acc={test_acc:.4f}")
    for c in range(4): print(f"    {CLASSES[c]:<9} F1={perclass[c]:.3f}")
    print(f"  gated@{chosen_t}: coverage={gated['coverage']:.3f} acc_kept={gated['accuracy_on_kept']:.3f} reject={gated['rejection']:.3f}")

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(5, 4.4)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Deploy on TEST155 (Macro-F1={test_macro:.3f})")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    EXP2_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP2_OUTPUT / "deploy_confusion_test155.png", dpi=150); plt.close(fig)

    # ===== 保存部署 checkpoint =====
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "arch": "efficientnet_b0 + Dropout(0.5)+Linear(1280,4)",
        "classes": CLASSES, "num_classes": NUM_CLASSES,
        "temperature": T, "gate_threshold": chosen_t, "gate_measure": "max_softmax",
        "preprocess": {"resize": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "seed": args.seed, "val_deploy_macro_f1": float(best_f1),
        "note": "部署成品；test155 Macro-F1 见 deploy_report.json，不等于第0步CV的0.707",
    }
    ckpt_path = MODELS_DIR / "ferac_efficientnetb0_deploy.pth"
    torch.save(ckpt, ckpt_path)

    report = {
        "step": "3_deploy",
        "disclaimer": "部署checkpoint在test155的一次性表现；CV的0.707±0.051才是方法无偏估计，二者不可混淆。",
        "train_size": len(tr_sub), "val_deploy_size": len(va_sub), "test_size": len(test_items),
        "temperature_T": T, "gate_threshold": chosen_t, "gate_measure": "max_softmax",
        "val_deploy_macro_f1": float(best_f1),
        "test155": {"macro_f1": float(test_macro), "accuracy": float(test_acc),
                    "per_class_f1": {CLASSES[c]: float(perclass[c]) for c in range(4)},
                    "confusion_matrix": cm.tolist()},
        "test155_gated_operating_point": gated,
        "checkpoint": str(ckpt_path.name), "seed": args.seed,
    }
    with open(EXP2_OUTPUT / "deploy_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {ckpt_path}")
    print(f"  saved: {EXP2_OUTPUT/'deploy_report.json'} , deploy_confusion_test155.png")


if __name__ == "__main__":
    main()
