# -*- coding: utf-8 -*-
"""
Exp2 第1步：诚实的小样本评估（叠加，不重训）
=================================================
基于第0步的 OOF 预测（artifacts/exp2/oof/oof_logits.npz），补充：
  1) 4×4 混淆矩阵（原始计数 + 行归一化），保存 PNG
  2) 每类样本数 + 每类 Precision/Recall/F1（标注 Fear 仅 49 张）
  3) Macro-F1 与每类 F1 的 bootstrap 95% CI（对 OOF 预测重采样 N≥2000）

不重新训练。输出：artifacts/exp2/honest_eval/

用法：python honest_eval.py [--n_boot 2000] [--seed 42]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.paths import EXP2_OUTPUT

CLASSES = ["Natural", "anger", "fear", "joy"]
LABELS = list(range(4))


def load_oof():
    d = np.load(EXP2_OUTPUT / "oof" / "oof_logits.npz", allow_pickle=True)
    logits, labels, fold = d["logits"], d["labels"], d["fold"]
    assert (fold >= 0).all(), "OOF 未全覆盖"
    preds = logits.argmax(1)
    return logits, labels, preds


def plot_confusion(cm, out_png, normalize=False, title=""):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    data = cm.astype(float)
    if normalize:
        row = data.sum(1, keepdims=True)
        data = np.divide(data, row, out=np.zeros_like(data), where=row > 0)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(1.0 if normalize else data.max()))
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(4):
        for j in range(4):
            txt = f"{data[i,j]:.2f}" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if data[i, j] > (0.5 if normalize else data.max()/2) else "black",
                    fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def bootstrap_ci(labels, preds, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    macro, per_class = [], [[], [], [], []]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # 有放回重采样
        yl, yp = labels[idx], preds[idx]
        macro.append(f1_score(yl, yp, average="macro", labels=LABELS, zero_division=0))
        pc = f1_score(yl, yp, average=None, labels=LABELS, zero_division=0)
        for c in range(4):
            per_class[c].append(pc[c])

    def ci(arr):
        a = np.array(arr)
        return {"mean": float(a.mean()),
                "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5)),
                "width": float(np.percentile(a, 97.5) - np.percentile(a, 2.5))}
    return ci(macro), {CLASSES[c]: ci(per_class[c]) for c in range(4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = EXP2_OUTPUT / "honest_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    logits, labels, preds = load_oof()
    n = len(labels)
    print(f"  OOF samples: {n}")

    # 1) 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=LABELS)
    plot_confusion(cm, out_dir / "confusion_matrix_counts.png", False, "FERAC OOF — counts")
    plot_confusion(cm, out_dir / "confusion_matrix_rownorm.png", True, "FERAC OOF — row-normalized (recall)")
    print("  saved confusion matrices (counts + rownorm)")

    # 2) 每类 P/R/F1 + 样本数
    P, R, F, S = precision_recall_fscore_support(labels, preds, labels=LABELS, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", labels=LABELS, zero_division=0)
    per_class = {}
    print(f"\n  {'class':<9}{'n':>5}{'P':>8}{'R':>8}{'F1':>8}")
    for c in range(4):
        per_class[CLASSES[c]] = {"n": int(S[c]), "precision": float(P[c]),
                                 "recall": float(R[c]), "f1": float(F[c])}
        print(f"  {CLASSES[c]:<9}{int(S[c]):>5}{P[c]:>8.3f}{R[c]:>8.3f}{F[c]:>8.3f}")
    print(f"  {'MACRO':<9}{n:>5}{'':>8}{'':>8}{macro_f1:>8.3f}")

    with open(out_dir / "per_class_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"n_total": n, "oof_macro_f1": float(macro_f1),
                   "per_class": per_class,
                   "confusion_matrix_counts": cm.tolist(),
                   "classes": CLASSES,
                   "note": "OOF = 5-fold out-of-fold predictions, single-pass (no resampling)."},
                  f, indent=2, ensure_ascii=False)

    # 3) bootstrap CI
    print(f"\n  bootstrap (N={args.n_boot}, seed={args.seed}) ...")
    macro_ci, per_class_ci = bootstrap_ci(labels, preds, args.n_boot, args.seed)
    print(f"  Macro-F1: {macro_ci['mean']:.3f}  95%CI [{macro_ci['lo95']:.3f}, {macro_ci['hi95']:.3f}] (width {macro_ci['width']:.3f})")
    for c in CLASSES:
        ci = per_class_ci[c]
        print(f"    {c:<9} F1: {ci['mean']:.3f}  95%CI [{ci['lo95']:.3f}, {ci['hi95']:.3f}] (width {ci['width']:.3f})")

    with open(out_dir / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump({"n_boot": args.n_boot, "seed": args.seed,
                   "macro_f1_ci": macro_ci, "per_class_f1_ci": per_class_ci,
                   "note": "Resample OOF predictions with replacement; Fear (n=49) CI is widest."},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {out_dir}/ (confusion PNGs, per_class_metrics.json, bootstrap_ci.json)")


if __name__ == "__main__":
    main()
