"""
Engagnition Calibration Extractor v6 (Final)
==============================================
Fixes applied from peer review:

Fix 1: Persona names -> P3_low/P2_mid/P1_high (not "Fluctuator")
Fix 2: Transition matrix computed per-participant then aggregated
       (no cross-participant splicing artifact)
Fix 3: Phase = equal-duration temporal phases (documented)
Fix 4: Change rate labeled as "engagement state change frequency"
Fix 5: Segment duration = coarse persistence descriptor (documented)
Fix 6: Burden association = discussion-only, not core calibration

Output: calibration_final.json

Usage: python calibration_v6_final.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json

DATASET_ROOT = r"D:\project1\pythonProject\人机交互\files\Engagnition Dataset"


# ============================================================
# Core utilities
# ============================================================
def load_all_engagement_data(root):
    all_data = []
    for condition in ["LPE condition", "HPE condition"]:
        condition_path = os.path.join(root, condition)
        if not os.path.exists(condition_path):
            continue
        participants = sorted([
            d for d in os.listdir(condition_path)
            if os.path.isdir(os.path.join(condition_path, d)) and d.startswith("P")
        ])
        for pid in participants:
            eng_file = os.path.join(condition_path, pid, "EngagementData.csv")
            if not os.path.exists(eng_file):
                continue
            try:
                df = pd.read_csv(eng_file)
                if "Engagement" not in df.columns:
                    continue
                all_data.append({
                    "participant": pid,
                    "condition": condition.replace(" condition", ""),
                    "engagement_seq": df["Engagement"].values.astype(int),
                    "time_seq": df["SGTime"].values,
                })
            except Exception as e:
                print(f"  Error: {eng_file}: {e}")
    return all_data


def resample_to_windows(time_seq, eng_seq, window_sec=30):
    time_in_sec = time_seq * 60
    max_time = time_in_sec[-1]
    windows = []
    t = 0
    while t < max_time:
        mask = (time_in_sec >= t) & (time_in_sec < t + window_sec)
        if mask.sum() > 0:
            counts = np.bincount(eng_seq[mask], minlength=3)
            windows.append(np.argmax(counts))
        t += window_sec
    return np.array(windows)


def compute_transition_counts(seq, n_states=3):
    """Return raw counts matrix (NOT normalized). For single participant."""
    counts = np.zeros((n_states, n_states))
    for t in range(len(seq) - 1):
        c, n = seq[t], seq[t + 1]
        if 0 <= c < n_states and 0 <= n < n_states:
            counts[c][n] += 1
    return counts


def counts_to_probability(counts):
    """Normalize a counts matrix into probability matrix."""
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def aggregate_transition_matrix(data_list, window_sec=30):
    """
    FIX 2: Compute transition matrix by aggregating per-participant counts.
    Each participant's windowed sequence is counted independently,
    then counts are summed across participants before normalizing.
    This avoids the cross-participant splicing artifact.
    """
    total_counts = np.zeros((3, 3))
    for d in data_list:
        w = resample_to_windows(d["time_seq"], d["engagement_seq"], window_sec)
        if len(w) > 1:
            total_counts += compute_transition_counts(w)
    return counts_to_probability(total_counts), total_counts


def extract_segments(time_seq, eng_seq):
    if len(eng_seq) == 0:
        return []
    time_sec = time_seq * 60
    segments = []
    current_state = eng_seq[0]
    start_time = time_sec[0]
    for i in range(1, len(eng_seq)):
        if eng_seq[i] != current_state:
            segments.append({"state": int(current_state), "duration_sec": float(time_sec[i] - start_time)})
            current_state = eng_seq[i]
            start_time = time_sec[i]
    segments.append({"state": int(current_state), "duration_sec": float(time_sec[-1] - start_time)})
    return segments


def extract_features(data):
    seq = data["engagement_seq"]
    time_seq = data["time_seq"]
    total_time = (time_seq[-1] - time_seq[0]) * 60 if len(time_seq) > 1 else 1
    segments = extract_segments(time_seq, seq)
    state_time = {0: 0, 1: 0, 2: 0}
    for s in segments:
        state_time[s["state"]] += s["duration_sec"]
    for k in state_time:
        state_time[k] /= max(total_time, 1)
    n_changes = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
    change_rate = n_changes / max(total_time / 60, 0.01)
    return {
        "participant": data["participant"],
        "condition": data["condition"],
        "mean_eng": float(np.mean(seq)),
        "std_eng": float(np.std(seq)),
        "prop_0": state_time[0],
        "prop_1": state_time[1],
        "prop_2": state_time[2],
        "change_rate": change_rate,
    }


def do_clustering(features_list, n_clusters=3):
    X = np.array([[f["mean_eng"], f["std_eng"], f["prop_0"], f["prop_1"],
                    f["prop_2"], f["change_rate"]] for f in features_list])
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    cluster_means = {c: np.mean(X[clusters == c, 0]) for c in range(n_clusters)}
    sorted_ids = sorted(cluster_means, key=lambda x: cluster_means[x])
    mapping = {orig: rank for rank, orig in enumerate(sorted_ids)}
    return np.array([mapping[c] for c in clusters])


# ============================================================
# Phase analysis (FIX 2 applied: per-participant counting)
# ============================================================
def compute_phase_analysis(time_seq, eng_seq, n_phases=3, window_sec=30):
    t_min, t_max = time_seq[0], time_seq[-1]
    block_size = (t_max - t_min) / n_phases
    phases = []

    for p in range(n_phases):
        t_start = t_min + p * block_size
        t_end = t_start + block_size
        if p == n_phases - 1:
            mask = (time_seq >= t_start) & (time_seq <= t_end)
        else:
            mask = (time_seq >= t_start) & (time_seq < t_end)

        block_eng = eng_seq[mask]
        block_time = time_seq[mask]

        if len(block_eng) < 2:
            phases.append(None)
            continue

        dist = {s: float(np.mean(block_eng == s)) for s in [0, 1, 2]}
        windowed = resample_to_windows(block_time, block_eng, window_sec)

        # Per-participant counts for this phase (single participant, no splicing issue)
        if len(windowed) > 1:
            phase_counts = compute_transition_counts(windowed)
            phase_matrix = counts_to_probability(phase_counts)
        else:
            phase_matrix = np.eye(3)

        n_changes = sum(1 for i in range(len(block_eng) - 1) if block_eng[i] != block_eng[i + 1])
        phase_dur_min = (block_time[-1] - block_time[0]) if len(block_time) > 1 else 0.01

        phases.append({
            "mean": round(float(np.mean(block_eng)), 3),
            "std": round(float(np.std(block_eng)), 3),
            "distribution": {str(k): round(v, 4) for k, v in dist.items()},
            "transition_matrix_30s": [[round(float(v), 4) for v in row] for row in phase_matrix],
            "state_change_freq_per_min": round(float(n_changes / max(phase_dur_min, 0.01)), 3),
            "n_windows": len(windowed),
        })

    return phases


# ============================================================
# Session burden (Layer 2)
# ============================================================
def load_session_burden(root):
    xlsx_path = os.path.join(root, "Session Elapsed Time.xlsx")
    if not os.path.exists(xlsx_path):
        return {}
    df = pd.read_excel(xlsx_path, header=None)
    burden = {}
    for i in range(2, len(df)):
        row = df.iloc[i]
        cond, pid = row[0], row[1]
        if pd.isna(cond) or pd.isna(pid):
            continue
        durations = []
        for s in range(20):
            try:
                durations.append(float(row[2 + s]))
            except (ValueError, TypeError):
                durations.append(np.nan)
        valid = np.array([d for d in durations if not np.isnan(d)])
        if len(valid) < 4:
            continue
        x = np.arange(len(valid))
        slope, _, _, p_val, _ = stats.linregress(x, valid)
        burden[str(pid)] = {
            "mean_sec": round(float(np.mean(valid)), 2),
            "std_sec": round(float(np.std(valid)), 2),
            "max_sec": round(float(np.max(valid)), 2),
            "total_sec": round(float(np.sum(valid)), 2),
            "early_5_mean_sec": round(float(np.mean(valid[:5])), 2),
            "late_5_mean_sec": round(float(np.mean(valid[-5:])), 2),
            "slope": round(float(slope), 3),
            "slope_p": round(float(p_val), 4),
            "n_sessions": len(valid),
        }
    return burden


# ============================================================
# Associations (Layer 3 - discussion only)
# ============================================================
def compute_associations(all_data, features_list, burden_lookup):
    records = []
    for i, d in enumerate(all_data):
        pid = d["participant"]
        f = features_list[i]
        if pid not in burden_lookup:
            continue
        b = burden_lookup[pid]
        phases = compute_phase_analysis(d["time_seq"], d["engagement_seq"], 3)
        e0 = phases[0]["mean"] if phases[0] else 0
        e2 = phases[2]["mean"] if phases[2] else 0
        records.append({
            "pid": pid, "cond": d["condition"],
            "mean_eng": f["mean_eng"], "std_eng": f["std_eng"],
            "change_rate": f["change_rate"], "decline": e0 - e2,
            "mean_ses": b["mean_sec"], "std_ses": b["std_sec"], "slope": b["slope"],
        })
    if len(records) < 5:
        return None, None
    df = pd.DataFrame(records)

    assoc = {}
    pairs = [
        ("burden_vs_engagement", "mean_ses", "mean_eng",
         "Task burden (mean session sec) vs overall engagement"),
        ("burden_vs_decline", "mean_ses", "decline",
         "Task burden vs engagement decline (early - late)"),
        ("session_var_vs_eng_var", "std_ses", "std_eng",
         "Session duration variability vs engagement variability"),
        ("duration_trend_vs_decline", "slope", "decline",
         "Session duration trend vs engagement decline"),
    ]
    for name, x_col, y_col, desc in pairs:
        r, p = stats.pearsonr(df[x_col], df[y_col])
        assoc[name] = {"description": desc, "r": round(float(r), 3), "p": round(float(p), 4)}

    # Burden terciles
    try:
        df["level"] = pd.qcut(df["mean_ses"], q=3,
                               labels=["low_burden", "mid_burden", "high_burden"],
                               duplicates="drop")
    except ValueError:
        df["level"] = pd.qcut(df["mean_ses"], q=2,
                               labels=["low_burden", "high_burden"], duplicates="drop")

    bmap = {}
    for lv in sorted(df["level"].unique()):
        sub = df[df["level"] == lv]
        bmap[str(lv)] = {
            "n": len(sub), "mean_eng": round(float(sub.mean_eng.mean()), 3),
            "std_eng": round(float(sub.mean_eng.std()), 3),
            "mean_ses_sec": round(float(sub.mean_ses.mean()), 1),
        }
    return assoc, bmap


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  CALIBRATION v6 (All Peer Review Fixes Applied)")
    print("=" * 70)
    print()

    all_data = load_all_engagement_data(DATASET_ROOT)
    features_list = [extract_features(d) for d in all_data]
    burden_lookup = load_session_burden(DATASET_ROOT)
    print(f"Loaded: {len(all_data)} engagement records, {len(burden_lookup)} session records\n")

    clusters = do_clustering(features_list)

    # FIX 1: Neutral persona names based on engagement level
    persona_keys = ["P3_low_engagement", "P2_mid_engagement", "P1_high_engagement"]
    persona_labels = {
        "P3_low_engagement": "P3 (Low engagement)",
        "P2_mid_engagement": "P2 (Mid engagement)",
        "P1_high_engagement": "P1 (High engagement)",
    }

    calibration = {
        "metadata": {
            "dataset": "Engagnition (Nature Scientific Data, 2024)",
            "n_records": len(all_data),
            "window_size_sec": 30,
            "n_temporal_phases": 3,
            # FIX 3: explicit description of phase method
            "phase_method": "equal-duration temporal phases (total session time / 3), "
                            "NOT session-based or task-defined phases",
            "core_calibration_dimensions": [
                "1. Transition matrix (30s windows) -> KL divergence < 0.15",
                "2. State distribution -> Wasserstein distance < 0.20",
                "3. Temporal dynamics (early/mid/late) -> phase mean error < 20%",
                # FIX 4: careful wording for change rate
                "4. Engagement state change frequency -> within 50% of real",
                # FIX 5: segment duration as coarse descriptor
                "5. Segment duration (coarse persistence descriptor) -> median within 50%",
            ],
            # FIX 6: burden is discussion-only
            "supplementary_analysis": "Task burden association (Layer 3) is for Discussion "
                                      "section only, not a core calibration dimension.",
            "session_burden_note":
                "Session elapsed time is an indirect proxy for task burden. "
                "Longer sessions may reflect slower execution, attention drift, "
                "educator intervention, or individual ability differences - "
                "not solely task difficulty.",
            # FIX 2: documented
            "transition_matrix_method": "Per-participant windowed counts aggregated then "
                                        "normalized. No cross-participant sequence splicing.",
        },
        "personas": {},
        "global": {},
        "discussion_only": {},
    }

    # --- Global (FIX 2: aggregate per-participant) ---
    all_eng = np.concatenate([d["engagement_seq"] for d in all_data])
    global_matrix, global_counts = aggregate_transition_matrix(all_data, 30)
    calibration["global"] = {
        "n_datapoints": int(len(all_eng)),
        "engagement_distribution": {str(s): round(float(np.mean(all_eng == s)), 4) for s in [0, 1, 2]},
        "transition_matrix_30s": [[round(float(v), 4) for v in row] for row in global_matrix],
    }

    # ============================================================
    # Per-persona
    # ============================================================
    print("=" * 70)
    print("  LAYER 1: Per-Persona Calibration")
    print("=" * 70)
    print()

    for rank in range(3):
        pkey = persona_keys[rank]
        plabel = persona_labels[pkey]
        mask = clusters == rank
        indices = [i for i in range(len(features_list)) if mask[i]]
        if not indices:
            continue

        member_data = [all_data[i] for i in indices]
        member_features = [features_list[i] for i in indices]
        member_pids = sorted(set(f["participant"] for f in member_features))
        member_conds = sorted(set(f["condition"] for f in member_features))

        avg_mean = np.mean([f["mean_eng"] for f in member_features])
        avg_rate = np.mean([f["change_rate"] for f in member_features])
        avg_prop = {k: np.mean([f[f"prop_{k}"] for f in member_features]) for k in [0, 1, 2]}

        # FIX 2: aggregate per-participant transition counts
        matrix, counts = aggregate_transition_matrix(member_data, 30)

        # Segments
        seg_dur = {0: [], 1: [], 2: []}
        for d in member_data:
            for s in extract_segments(d["time_seq"], d["engagement_seq"]):
                seg_dur[s["state"]].append(s["duration_sec"])

        # Phase analysis (per-participant, then average)
        all_phases = [compute_phase_analysis(d["time_seq"], d["engagement_seq"], 3) for d in member_data]
        phase_summary = {}
        for p_idx, label in enumerate(["early", "mid", "late"]):
            valid = [ph[p_idx] for ph in all_phases if ph[p_idx] is not None]
            if not valid:
                phase_summary[label] = None
                continue

            # FIX 2: aggregate phase transition matrices per-participant
            phase_total_counts = np.zeros((3, 3))
            for ph in valid:
                phase_total_counts += np.array(ph["transition_matrix_30s"]) * max(ph["n_windows"] - 1, 1)
            # Re-normalize (approximate: weighted by window count)
            phase_matrix = counts_to_probability(phase_total_counts)

            phase_summary[label] = {
                "mean_engagement": round(float(np.mean([p["mean"] for p in valid])), 3),
                "std_engagement": round(float(np.mean([p["std"] for p in valid])), 3),
                "distribution": {
                    str(s): round(float(np.mean([p["distribution"][str(s)] for p in valid])), 4)
                    for s in [0, 1, 2]
                },
                "transition_matrix_30s": [[round(float(v), 4) for v in row] for row in phase_matrix],
                "state_change_freq_per_min": round(float(np.mean([p["state_change_freq_per_min"] for p in valid])), 3),
            }

        e_early = phase_summary["early"]["mean_engagement"] if phase_summary.get("early") else 0
        e_late = phase_summary["late"]["mean_engagement"] if phase_summary.get("late") else 0
        decline = e_early - e_late
        trend = "declining" if decline > 0.05 else "increasing" if decline < -0.05 else "stable"

        # Print
        print(f"  {plabel}:")
        print(f"    N={len(indices)}, PIDs: {', '.join(member_pids)}")
        print(f"    Conditions: {', '.join(member_conds)}")
        print(f"    Mean eng: {avg_mean:.2f}, State change freq: {avg_rate:.3f}/min")
        print(f"    Distribution: 0={avg_prop[0]*100:.1f}%, 1={avg_prop[1]*100:.1f}%, 2={avg_prop[2]*100:.1f}%")
        print(f"    Transition (30s, per-participant aggregated):")
        print(f"           To 0    To 1    To 2")
        for i in range(3):
            print(f"      From {i}: {matrix[i][0]:.4f}  {matrix[i][1]:.4f}  {matrix[i][2]:.4f}")

        off_diag = [(i, j, matrix[i][j]) for i in range(3) for j in range(3) if i != j and matrix[i][j] > 0.001]
        if off_diag:
            print(f"    Key transitions: {', '.join(f'{i}->{j}: {v:.3f}' for i,j,v in off_diag)}")

        print(f"    Temporal phases (equal-duration):")
        for label in ["early", "mid", "late"]:
            ps = phase_summary.get(label)
            if ps:
                print(f"      {label:5s}: mean={ps['mean_engagement']:.2f}, "
                      f"std={ps['std_engagement']:.2f}, "
                      f"freq={ps['state_change_freq_per_min']:.3f}/min")
        print(f"    Trend: {trend} (early={e_early:.2f} -> late={e_late:.2f}, delta={decline:+.3f})")
        print(f"    Segment persistence (median):", end="")
        for st in [0, 1, 2]:
            if seg_dur[st]:
                print(f"  S{st}={np.median(seg_dur[st]):.0f}s", end="")
        print("\n")

        calibration["personas"][pkey] = {
            "participants": member_pids,
            "conditions": member_conds,
            "n_records": len(indices),
            "mean_engagement": round(float(avg_mean), 3),
            "state_change_freq_per_min": round(float(avg_rate), 3),
            "engagement_distribution": {str(k): round(float(v), 4) for k, v in avg_prop.items()},
            "transition_matrix_30s": [[round(float(v), 4) for v in row] for row in matrix],
            "temporal_phases": phase_summary,
            "temporal_trend": trend,
            "temporal_decline": round(float(decline), 3),
            "segment_persistence_median_sec": {
                str(k): round(float(np.median(v)), 1) if v else 0
                for k, v in seg_dur.items()
            },
            "segment_persistence_mean_sec": {
                str(k): round(float(np.mean(v)), 1) if v else 0
                for k, v in seg_dur.items()
            },
        }

    # ============================================================
    # Layer 2 & 3: Discussion-only
    # ============================================================
    print("=" * 70)
    print("  LAYER 2-3: Discussion-Only Analysis")
    print("=" * 70)
    print()

    assoc, bmap = compute_associations(all_data, features_list, burden_lookup)
    if assoc:
        print("  Correlations (participant-level):")
        for name, a in assoc.items():
            sig = " *" if a["p"] < 0.05 else ""
            print(f"    {a['description']}: r={a['r']:+.3f}, p={a['p']:.4f}{sig}")
        print()

        print("  Burden -> Engagement mapping:")
        for level, info in sorted(bmap.items()):
            print(f"    {level}: n={info['n']}, eng={info['mean_eng']:.2f}+/-{info['std_eng']:.2f}")
        print()

        calibration["discussion_only"] = {
            "associations": assoc,
            "burden_mapping": bmap,
            "session_burden": burden_lookup,
            "note": "These results are for the Discussion section only. "
                    "They are NOT core calibration targets.",
        }

    # --- Save ---
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "calibration_final.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {output_file}\n")
    print("=" * 70)
    print("  CORE CALIBRATION DIMENSIONS (LLM must match)")
    print("=" * 70)
    print("  1. Transition matrix     KL(P_real||P_llm) < 0.15")
    print("  2. State distribution    Wasserstein < 0.20")
    print("  3. Temporal dynamics     Phase mean error < 20%")
    print("  4. State change freq     Within 50% of real")
    print("  5. Segment persistence   Median within 50% of real")
    print()
    print("  Discussion-only: burden association, LPE/HPE comparison")
    print("=" * 70)
