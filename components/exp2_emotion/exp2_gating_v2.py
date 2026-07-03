# -*- coding: utf-8 -*-
"""
实验2 感知门控改善（不动数据/模型，仅改校准 + 门控）：
  (1) per-class 温度缩放 (每类独立 T)  vs  全局温度缩放 —— ECE + per-class(classwise) ECE。
  (2) per-class 阈值 + 三级门控 (accept / cautious-accept / abstain)，每类阈值独立。

★诚实纪律：
  - 校准参数 + 阈值只在【训练折(验证)】上 fit；【留出折】只报告(好坏都报)，绝不调到留出折好看。
  - 用 5 折交叉验证(复用 OOF 的 fold 列)避免过拟合。
  - 改善的是"门控精细度 / 对不同可靠性类的区别对待"，**不是感知准确度**；fear 仍仅 49 张、仍不可靠，
    不声称"现在 fear 可靠了"。
  - cautious 档语义：中等置信 → 采纳但下游应触发更保守策略。

数据：artifacts/exp2/oof/oof_logits.npz (770×4 OOF 原始 logits + labels + fold + classes)。
用法：python exp2_gating_v2.py
"""
import json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "artifacts/exp2/oof/oof_logits.npz"
OUT = ROOT / "artifacts/exp2/gating_v2"; OUT.mkdir(parents=True, exist_ok=True)
FIGD = OUT / "figures"; FIGD.mkdir(exist_ok=True)
TARGET_ACCEPT = 0.90      # accept 档目标精度(与部署"高置信≥0.9"口径一致)
TARGET_CAUTIOUS = 0.70    # cautious 档目标精度(中等)
NBINS = 15

def softmax_T(logits, T):
    """T: scalar or (K,) per-class temperature. calibrated logit_c = z_c / T_c."""
    z = logits / np.asarray(T, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def ece(probs, labels, n_bins=NBINS):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0: continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

def classwise_ece(probs, labels, n_bins=NBINS):
    K = probs.shape[1]; eces = []
    bins = np.linspace(0, 1, n_bins + 1)
    for c in range(K):
        p = probs[:, c]; y = (labels == c).astype(float); e = 0.0
        for i in range(n_bins):
            m = (p > bins[i]) & (p <= bins[i + 1])
            if m.sum() == 0: continue
            e += m.mean() * abs(y[m].mean() - p[m].mean())
        eces.append(float(e))
    return eces, float(np.mean(eces))

def fit_global_T(logits, labels, max_iter=200):
    lg = torch.tensor(logits, dtype=torch.float32); ly = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=max_iter)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lg / logT.exp(), ly); loss.backward(); return loss
    opt.step(closure); return float(logT.exp().item())

def fit_perclass_T(logits, labels, max_iter=300):
    """每类独立温度：calibrated logit_c = z_c / T_c。LBFGS 拟合 K 个 logT。"""
    lg = torch.tensor(logits, dtype=torch.float32); ly = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(logits.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=max_iter)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lg / logT.exp(), ly); loss.backward(); return loss
    opt.step(closure); return logT.exp().detach().numpy().astype(float)

def threshold_for_precision(conf, correct, target):
    """在 pred==c 的子集上：求最低阈值 t，使 {conf>=t} 的精度 >= target（top-down 累计精度）。
    达不到 -> 1.01 (永不 accept)。"""
    if len(conf) == 0: return 1.01
    order = np.argsort(-conf); c = conf[order]; ok = correct[order].astype(float)
    cum = np.cumsum(ok); n = np.arange(1, len(c) + 1); prec = cum / n
    valid = np.where(prec >= target)[0]
    return float(c[valid[-1]]) if len(valid) else 1.01

def main():
    d = np.load(OOF, allow_pickle=True)
    logits = d["logits"].astype(np.float64); labels = d["labels"].astype(int)
    fold = d["fold"].astype(int); classes = [str(x) for x in d["classes"]]
    K = len(classes); folds = sorted(np.unique(fold))
    N = len(labels)
    print(f"OOF: N={N} classes={classes} folds={folds} label_dist={np.bincount(labels).tolist()}")

    # ===== Part 1: 校准对比 (CV) =====
    raw = softmax_T(logits, 1.0)
    cal_g = np.zeros_like(raw); cal_pc = np.zeros_like(raw)
    Tg_folds = []; Tpc_folds = []
    for f in folds:
        tr = fold != f; te = fold == f
        Tg = fit_global_T(logits[tr], labels[tr]); Tpc = fit_perclass_T(logits[tr], labels[tr])
        Tg_folds.append(Tg); Tpc_folds.append(Tpc.tolist())
        cal_g[te] = softmax_T(logits[te], Tg); cal_pc[te] = softmax_T(logits[te], Tpc)
    ece_raw = ece(raw, labels); ece_g = ece(cal_g, labels); ece_pc = ece(cal_pc, labels)
    cwe_raw, cwm_raw = classwise_ece(raw, labels); cwe_g, cwm_g = classwise_ece(cal_g, labels)
    cwe_pc, cwm_pc = classwise_ece(cal_pc, labels)
    # 部署用温度(全 OOF fit)
    Tg_full = fit_global_T(logits, labels); Tpc_full = fit_perclass_T(logits, labels)
    print("\n== Part1 校准 (5-fold CV, 留出折评估) ==")
    print(f"  ECE(max-prob):  raw {ece_raw:.4f} -> global-T {ece_g:.4f} -> per-class-T {ece_pc:.4f}")
    print(f"  classwise-ECE:  raw {cwm_raw:.4f} -> global-T {cwm_g:.4f} -> per-class-T {cwm_pc:.4f}")
    print(f"  per-class ECE (per-class-T): " + ", ".join(f"{classes[c]}={cwe_pc[c]:.3f}" for c in range(K)))
    print(f"  部署温度: global T={Tg_full:.3f} | per-class T=" + ", ".join(f"{classes[c]}={Tpc_full[c]:.3f}" for c in range(K)))

    # ===== Part 2: per-class 三级门控 (CV) + 对比全局阈值 =====
    # 门控用【全局温度】校准概率(global-T, max-prob ECE 最佳)；per-class 阈值吸收各类校准差异。
    # per-class 温度因 max-prob ECE 变差(过度锐化多数类 joy)，仅作 Part1 对比、不用于门控(遵循"不稳不强用")。
    rows = []  # (pred, tier, correct, conf, label)
    accth_folds = {c: [] for c in range(K)}; cauth_folds = {c: [] for c in range(K)}
    for f in folds:
        tr = fold != f; te = fold == f
        Tg_f = fit_global_T(logits[tr], labels[tr])
        p_tr = softmax_T(logits[tr], Tg_f); p_te = softmax_T(logits[te], Tg_f)
        pred_tr = p_tr.argmax(1); conf_tr = p_tr.max(1); corr_tr = (pred_tr == labels[tr])
        acc_th = {}; cau_th = {}
        for c in range(K):
            m = pred_tr == c
            acc_th[c] = threshold_for_precision(conf_tr[m], corr_tr[m], TARGET_ACCEPT)
            cau_th[c] = threshold_for_precision(conf_tr[m], corr_tr[m], TARGET_CAUTIOUS)
            accth_folds[c].append(acc_th[c]); cauth_folds[c].append(cau_th[c])
        pred_te = p_te.argmax(1); conf_te = p_te.max(1); corr_te = (pred_te == labels[te])
        lab_te = labels[te]
        for i in range(te.sum()):
            c = int(pred_te[i]); cf = float(conf_te[i])
            tier = "accept" if cf >= acc_th[c] else ("cautious" if cf >= cau_th[c] else "abstain")
            rows.append((c, tier, bool(corr_te[i]), cf, int(lab_te[i])))

    def tier_stats(sel):
        n = len(sel); cov = n / N
        prec = np.mean([r[2] for r in sel]) if n else None
        return {"n": n, "coverage": round(cov, 4), "precision": (round(prec, 4) if prec is not None else None)}
    # 整体三级
    overall = {t: tier_stats([r for r in rows if r[1] == t]) for t in ["accept", "cautious", "abstain"]}
    kept = [r for r in rows if r[1] in ("accept", "cautious")]
    overall["accept+cautious_kept"] = tier_stats(kept)
    # 每类三级
    per_class = {}
    for c in range(K):
        cr = [r for r in rows if r[0] == c]
        per_class[classes[c]] = {
            "predicted_n": len(cr),
            "accept_threshold(CV mean)": round(float(np.mean([t for t in accth_folds[c] if t <= 1.0])) if any(t <= 1.0 for t in accth_folds[c]) else 1.01, 3),
            "cautious_threshold(CV mean)": round(float(np.mean([t for t in cauth_folds[c] if t <= 1.0])) if any(t <= 1.0 for t in cauth_folds[c]) else 1.01, 3),
            "accept": tier_stats([r for r in cr if r[1] == "accept"]),
            "cautious": tier_stats([r for r in cr if r[1] == "cautious"]),
            "abstain": tier_stats([r for r in cr if r[1] == "abstain"])}

    # 部署阈值(全 OOF fit, global-T 校准概率)
    p_full = softmax_T(logits, Tg_full); pred_full = p_full.argmax(1); conf_full = p_full.max(1); corr_full = (pred_full == labels)
    deploy_th = {}
    for c in range(K):
        m = pred_full == c
        deploy_th[classes[c]] = {"accept": round(threshold_for_precision(conf_full[m], corr_full[m], TARGET_ACCEPT), 3),
                                 "cautious": round(threshold_for_precision(conf_full[m], corr_full[m], TARGET_CAUTIOUS), 3)}

    # ===== 对比旧全局阈值 0.86 / 0.73 (二元, global-T 校准概率, CV) =====
    g_pred = cal_g.argmax(1); g_conf = cal_g.max(1); g_corr = (g_pred == labels)
    global_base = {}
    for th in [0.86, 0.73]:
        keep = g_conf >= th
        global_base[str(th)] = {"coverage": round(keep.mean(), 4),
                                "precision_on_kept": round(float(g_corr[keep].mean()) if keep.sum() else 0.0, 4),
                                "n_kept": int(keep.sum())}

    report = {
        "n_total": N, "classes": classes, "cv": "5-fold (OOF fold column); calib+thresholds fit on train folds only",
        "calibration": {
            "ece_max_prob": {"raw_T1": round(ece_raw, 4), "global_T": round(ece_g, 4), "per_class_T": round(ece_pc, 4)},
            "classwise_ece": {"raw_T1": round(cwm_raw, 4), "global_T": round(cwm_g, 4), "per_class_T": round(cwm_pc, 4)},
            "per_class_ece_per_class_T": {classes[c]: round(cwe_pc[c], 4) for c in range(K)},
            "deploy_global_T": round(Tg_full, 4),
            "deploy_per_class_T": {classes[c]: round(float(Tpc_full[c]), 4) for c in range(K)}},
        "three_level_gating": {"target_accept_precision": TARGET_ACCEPT, "target_cautious_precision": TARGET_CAUTIOUS,
                               "overall": overall, "per_class": per_class, "deploy_thresholds": deploy_th},
        "old_global_thresholds_binary": global_base}
    json.dump(report, open(OUT / "gating_v2.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n== Part2 三级门控 (CV 留出折汇总) ==")
    for t in ["accept", "cautious", "abstain", "accept+cautious_kept"]:
        s = overall[t]; print(f"  {t:<22} cov={s['coverage']} prec={s['precision']} (n={s['n']})")
    print("\n  per-class (CV 阈值均值 + 留出折分层):")
    for c in range(K):
        pc = per_class[classes[c]]
        print(f"   {classes[c]:<8} acc_th={pc['accept_threshold(CV mean)']} cau_th={pc['cautious_threshold(CV mean)']} | "
              f"accept(cov={pc['accept']['coverage']},prec={pc['accept']['precision']}) "
              f"cautious(cov={pc['cautious']['coverage']},prec={pc['cautious']['precision']}) "
              f"abstain(cov={pc['abstain']['coverage']})")
    print("\n  对比旧全局阈值(二元, global-T):")
    for th, s in global_base.items():
        print(f"   th={th}: coverage={s['coverage']} precision_on_kept={s['precision_on_kept']} n_kept={s['n_kept']}")
    plot(report, classes); print("\nsaved: gating_v2.json + figures/")

def plot(report, classes):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cal = report["calibration"]; tl = report["three_level_gating"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # ECE 对比
    ax = axes[0]; x = np.arange(3); w = 0.35
    em = [cal["ece_max_prob"][k] for k in ["raw_T1", "global_T", "per_class_T"]]
    cw = [cal["classwise_ece"][k] for k in ["raw_T1", "global_T", "per_class_T"]]
    ax.bar(x - w/2, em, w, label="ECE (max-prob)", color="#4C72B0")
    ax.bar(x + w/2, cw, w, label="classwise-ECE", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(["raw (T=1)", "global-T", "per-class-T"]); ax.set_ylabel("ECE (lower=better)")
    ax.set_title("Calibration: per-class vs global temperature (5-fold CV)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    # 每类三级 coverage 堆叠
    ax = axes[1]; pc = tl["per_class"]; xs = np.arange(len(classes))
    acc = [pc[c]["accept"]["coverage"] / max(pc[c]["accept"]["coverage"]+pc[c]["cautious"]["coverage"]+pc[c]["abstain"]["coverage"], 1e-9) for c in classes]
    cau = [pc[c]["cautious"]["coverage"] / max(pc[c]["accept"]["coverage"]+pc[c]["cautious"]["coverage"]+pc[c]["abstain"]["coverage"], 1e-9) for c in classes]
    aba = [pc[c]["abstain"]["coverage"] / max(pc[c]["accept"]["coverage"]+pc[c]["cautious"]["coverage"]+pc[c]["abstain"]["coverage"], 1e-9) for c in classes]
    ax.bar(xs, acc, label="accept", color="#55A868")
    ax.bar(xs, cau, bottom=acc, label="cautious", color="#DD8452")
    ax.bar(xs, aba, bottom=np.array(acc)+np.array(cau), label="abstain", color="#C44E52")
    ax.set_xticks(xs); ax.set_xticklabels(classes); ax.set_ylabel("fraction of predicted-as-class")
    ax.set_title("Per-class 3-level gating mix (reliable→more accept, unreliable→conservative)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGD / "gating_v2.png", dpi=300); fig.savefig(FIGD / "gating_v2.pdf"); plt.close(fig)

if __name__ == "__main__":
    main()
