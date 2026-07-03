# -*- coding: utf-8 -*-
"""
Exp2 第2步：强化置信度门控（诚实弃权）
==========================================
全部在 OOF（5折折外预测，artifacts/exp2/oof/oof_logits.npz）上做 —— 这是天然留出的
验证面。所有"选择"(温度T/不确定度/工作点阈值)都在此完成；真正的测试集155报告留到第3步。

产出（artifacts/exp2/confidence_gating/）：
  1) 温度缩放：在OOF上拟合标量T(最小化NLL)，报校准前后 ECE + reliability diagram
  2) 三种不确定度(max-softmax / entropy / margin)的 risk-coverage 曲线 + AURC
  3) 阈值表(Table4扩展)：阈值0.3..0.9 的 coverage/accuracy/macroF1/rejection，pre vs post-T
  4) 工作点：在OOF上选"高置信子集accuracy≥目标"的最低阈值 -> gating_operating_points.json

用法：python confidence_gating.py [--target_acc 0.90] [--seed 42]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.paths import EXP2_OUTPUT

CLASSES = ["Natural", "anger", "fear", "joy"]
LABELS = list(range(4))


def softmax_np(logits, T=1.0):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (confidence = max prob)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            rows.append((0.5*(bins[i]+bins[i+1]), 0, 0, 0)); continue
        acc = correct[m].mean(); avgc = conf[m].mean(); w = m.mean()
        e += w * abs(acc - avgc)
        rows.append((0.5*(bins[i]+bins[i+1]), acc, avgc, int(m.sum())))
    return float(e), rows


def fit_temperature(logits, labels, max_iter=200):
    """LBFGS 在 OOF logits 上最小化 NLL 拟合标量 T。"""
    lg = torch.tensor(logits, dtype=torch.float32)
    ly = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)  # T=exp(logT)>0
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp(), ly)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().item())


def uncertainty_scores(probs):
    """返回三种 '越大越自信' 的分数。"""
    p_sorted = np.sort(probs, axis=1)
    max_soft = probs.max(1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)
    margin = p_sorted[:, -1] - p_sorted[:, -2]
    return {"max_softmax": max_soft,
            "neg_entropy": -entropy,    # 越大越自信
            "margin": margin}


def risk_coverage(conf_score, labels, preds, n_points=50):
    """按置信度降序，扫覆盖率，算保留子集的 error(risk) 与 accuracy。AURC=risk对coverage积分。"""
    order = np.argsort(-conf_score)
    correct = (preds == labels).astype(float)[order]
    n = len(correct)
    cov, risk, acc = [], [], []
    for k in range(1, n + 1):
        c = correct[:k]
        cov.append(k / n); risk.append(1 - c.mean()); acc.append(c.mean())
    cov, risk, acc = np.array(cov), np.array(risk), np.array(acc)
    aurc = float(np.trapezoid(risk, cov))
    # 抽稀到 n_points 便于存储/画图
    idx = np.linspace(0, n - 1, n_points).astype(int)
    return cov[idx].tolist(), acc[idx].tolist(), risk[idx].tolist(), aurc


def threshold_table(probs, labels, n_total):
    pred = probs.argmax(1); conf = probs.max(1)
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = conf >= t
        n_keep = int(keep.sum())
        cov = n_keep / n_total
        if n_keep > 0:
            acc = accuracy_score(labels[keep], pred[keep])
            mf1 = f1_score(labels[keep], pred[keep], average="macro", labels=LABELS, zero_division=0)
        else:
            acc = mf1 = 0.0
        rows.append({"threshold": t, "coverage": float(cov), "accuracy": float(acc),
                     "macro_f1": float(mf1), "rejection": float(1 - cov), "n_kept": n_keep})
    return rows


def plot_reliability(rows_before, rows_after, ece_b, ece_a, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, rows, ec, title in [(axes[0], rows_before, ece_b, "Before T (raw)"),
                                 (axes[1], rows_after, ec if False else ece_a, "After T")]:
        centers = [r[0] for r in rows]; accs = [r[1] for r in rows]; confs = [r[2] for r in rows]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.bar(centers, accs, width=1/len(rows)*0.9, alpha=0.7, edgecolor="k", label="accuracy")
        ax.plot(centers, confs, "ro-", ms=3, lw=1, label="avg confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"{title}  ECE={ec:.3f}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_risk_coverage(rc, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, d in rc.items():
        axes[0].plot(d["coverage"], d["accuracy"], "-o", ms=2, label=f"{name} (AURC={d['aurc']:.3f})")
        axes[1].plot(d["coverage"], d["risk"], "-o", ms=2, label=name)
    axes[0].set_xlabel("coverage"); axes[0].set_ylabel("accuracy (kept subset)")
    axes[0].set_title("Accuracy-Coverage"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].set_xlabel("coverage"); axes[1].set_ylabel("risk = error rate")
    axes[1].set_title("Risk-Coverage (lower=better)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_acc", type=float, default=0.90,
                    help="高置信子集目标accuracy(用于选工作点)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    out = EXP2_OUTPUT / "confidence_gating"; out.mkdir(parents=True, exist_ok=True)
    d = np.load(EXP2_OUTPUT / "oof" / "oof_logits.npz", allow_pickle=True)
    logits, labels = d["logits"].astype(np.float64), d["labels"]
    n = len(labels); preds = logits.argmax(1)
    print(f"  OOF samples: {n}")

    # ---- 1) 温度缩放 ----
    probs_raw = softmax_np(logits, 1.0)
    T = fit_temperature(logits, labels)
    probs_cal = softmax_np(logits, T)
    ece_b, rel_b = ece(probs_raw, labels)
    ece_a, rel_a = ece(probs_cal, labels)
    print(f"  Temperature T = {T:.4f}")
    print(f"  ECE before = {ece_b:.4f}  ->  after = {ece_a:.4f}")
    plot_reliability(rel_b, rel_a, ece_b, ece_a, out / "reliability_diagram.png")

    # ---- 2) 三种不确定度 risk-coverage（用校准后概率）----
    scores = uncertainty_scores(probs_cal)
    rc = {}
    for name, s in scores.items():
        cov, acc, risk, aurc = risk_coverage(s, labels, preds)
        rc[name] = {"coverage": cov, "accuracy": acc, "risk": risk, "aurc": aurc}
        print(f"  risk-coverage [{name:<12}] AURC={aurc:.4f} (lower=better)")
    plot_risk_coverage(rc, out / "risk_coverage.png")
    best_measure = min(rc, key=lambda k: rc[k]["aurc"])
    print(f"  -> best uncertainty measure (min AURC): {best_measure}")

    # ---- 3) 阈值表 pre vs post-T ----
    tbl_raw = threshold_table(probs_raw, labels, n)
    tbl_cal = threshold_table(probs_cal, labels, n)
    print(f"\n  {'thr':>4} | {'cov_raw':>7} {'acc_raw':>7} | {'cov_cal':>7} {'acc_cal':>7}")
    for r0, r1 in zip(tbl_raw, tbl_cal):
        print(f"  {r0['threshold']:>4} | {r0['coverage']:>7.3f} {r0['accuracy']:>7.3f} | {r1['coverage']:>7.3f} {r1['accuracy']:>7.3f}")

    # ---- 4) 工作点选择（OOF上选：达到目标accuracy的最低max-softmax阈值，校准后概率）----
    conf = probs_cal.max(1); pred = probs_cal.argmax(1)
    chosen = None
    for t in np.round(np.arange(0.30, 0.951, 0.01), 2):
        keep = conf >= t
        if keep.sum() < 20:  # 子集太小不可信
            break
        acc = accuracy_score(labels[keep], pred[keep])
        if acc >= args.target_acc:
            chosen = {"threshold": float(t), "coverage": float(keep.mean()),
                      "accuracy_on_kept": float(acc), "n_kept": int(keep.sum())}
            break
    print(f"\n  operating point (target acc>={args.target_acc}, selected on OOF): {chosen}")

    op = {"temperature_T": T, "ece_before": ece_b, "ece_after": ece_a,
          "uncertainty_measure_for_gating": "max_softmax",
          "best_measure_by_aurc": best_measure,
          "aurc": {k: rc[k]["aurc"] for k in rc},
          "target_high_conf_accuracy": args.target_acc,
          "operating_point": chosen,
          "selected_on": "OOF (5-fold out-of-fold of the 770 pool); applied to test155 in Step 3",
          "seed": args.seed}
    with open(out / "gating_operating_points.json", "w", encoding="utf-8") as f:
        json.dump(op, f, indent=2, ensure_ascii=False)
    with open(out / "ece_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"temperature_T": T, "ece_before": ece_b, "ece_after": ece_a,
                   "reliability_before": rel_b, "reliability_after": rel_a,
                   "threshold_table_raw": tbl_raw, "threshold_table_calibrated": tbl_cal},
                  f, indent=2, ensure_ascii=False)
    with open(out / "risk_coverage.json", "w", encoding="utf-8") as f:
        json.dump(rc, f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {out}/ (reliability_diagram.png, risk_coverage.png, *.json)")


if __name__ == "__main__":
    main()
