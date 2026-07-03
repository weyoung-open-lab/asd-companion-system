"""
Engagnition Data Explorer v3 - Condition-Aware Analysis
=========================================================
Compares 3 approaches:
  A) Mixed (current): LPE+HPE combined, ignore condition
  B) Split: Separate clustering & matrices for LPE and HPE
  C) Condition-as-feature: Cluster with condition as a feature

Usage: python explore_engagnition_v3.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

DATASET_ROOT = r"D:\project1\pythonProject\人机交互\files\Engagnition Dataset"


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


def compute_transition_matrix(seq, n_states=3):
    counts = np.zeros((n_states, n_states))
    for t in range(len(seq) - 1):
        c, n = seq[t], seq[t + 1]
        if 0 <= c < n_states and 0 <= n < n_states:
            counts[c][n] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums, counts


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
        "n_changes": n_changes,
        "total_sec": total_time,
    }


def do_clustering(features_list, n_clusters=3, extra_features=None):
    """Cluster and return sorted by mean engagement (low->mid->high)"""
    X = []
    for i, f in enumerate(features_list):
        row = [f["mean_eng"], f["std_eng"], f["prop_0"], f["prop_1"], f["prop_2"], f["change_rate"]]
        if extra_features is not None:
            row.extend(extra_features[i])
        X.append(row)
    X = np.array(X)

    # Handle edge case: fewer samples than clusters
    if len(X) < n_clusters:
        return list(range(len(X))), X

    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Sort clusters by mean engagement
    cluster_means = {}
    for c in range(n_clusters):
        mask = clusters == c
        if mask.sum() > 0:
            cluster_means[c] = np.mean(X[mask, 0])
        else:
            cluster_means[c] = 0

    sorted_ids = sorted(cluster_means, key=lambda x: cluster_means[x])
    mapping = {orig: rank for rank, orig in enumerate(sorted_ids)}
    remapped = np.array([mapping[c] for c in clusters])

    return remapped, X


def print_cluster_results(all_data, features_list, clusters, label, persona_names=None):
    if persona_names is None:
        persona_names = ["P3 (Low Engager)", "P2 (Fluctuator)", "P1 (High Engager)"]

    n_clusters = len(set(clusters))
    results = {}

    for rank in range(n_clusters):
        if rank >= len(persona_names):
            pname = f"Cluster {rank}"
        else:
            pname = persona_names[rank]

        mask = clusters == rank
        indices = [i for i in range(len(features_list)) if mask[i]]

        if not indices:
            print(f"  {pname}: (empty cluster)")
            continue

        member_features = [features_list[i] for i in indices]
        member_pids = sorted(set(f["participant"] for f in member_features))
        member_conds = sorted(set(f["condition"] for f in member_features))

        # Windowed transition matrix
        cluster_windowed = []
        for i in indices:
            d = all_data[i]
            w = resample_to_windows(d["time_seq"], d["engagement_seq"], 30)
            cluster_windowed.extend(w)
        cluster_windowed = np.array(cluster_windowed)
        matrix, counts = compute_transition_matrix(cluster_windowed)

        # Segment durations
        seg_dur = {0: [], 1: [], 2: []}
        for i in indices:
            d = all_data[i]
            for s in extract_segments(d["time_seq"], d["engagement_seq"]):
                seg_dur[s["state"]].append(s["duration_sec"])

        avg_mean = np.mean([f["mean_eng"] for f in member_features])
        avg_rate = np.mean([f["change_rate"] for f in member_features])
        avg_prop = {k: np.mean([f[f"prop_{k}"] for f in member_features]) for k in [0, 1, 2]}

        print(f"  {pname}:")
        print(f"    N={len(indices)}, PIDs: {', '.join(member_pids)}")
        print(f"    Conditions: {', '.join(member_conds)}")
        print(f"    Mean eng: {avg_mean:.2f}, Change rate: {avg_rate:.2f}/min")
        print(f"    Time: 0={avg_prop[0]*100:.1f}%, 1={avg_prop[1]*100:.1f}%, 2={avg_prop[2]*100:.1f}%")
        print(f"    Transition matrix (30s windows):")
        print(f"           To 0    To 1    To 2")
        for i in range(3):
            print(f"      From {i}: {matrix[i][0]:.3f}  {matrix[i][1]:.3f}  {matrix[i][2]:.3f}")

        # Non-zero transition highlights
        off_diag = []
        for i in range(3):
            for j in range(3):
                if i != j and matrix[i][j] > 0.001:
                    off_diag.append(f"{i}->{j}: {matrix[i][j]:.3f}")
        if off_diag:
            print(f"    Key transitions: {', '.join(off_diag)}")

        print(f"    Segment durations (median):", end="")
        for st in [0, 1, 2]:
            if seg_dur[st]:
                print(f"  S{st}={np.median(seg_dur[st]):.0f}s", end="")
        print()
        print()

        results[pname] = {
            "transition_matrix_30s": matrix.tolist(),
            "engagement_distribution": {str(k): round(v, 4) for k, v in avg_prop.items()},
            "mean_engagement": round(float(avg_mean), 3),
            "change_rate_per_min": round(float(avg_rate), 2),
            "n_records": len(indices),
            "participants": member_pids,
            "conditions": member_conds,
            "segment_duration_median_sec": {
                str(k): round(float(np.median(v)), 1) if v else 0
                for k, v in seg_dur.items()
            },
        }

    return results


# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Engagnition v3: Condition-Aware Comparison")
    print("=" * 70)
    print()

    all_data = load_all_engagement_data(DATASET_ROOT)
    features_list = [extract_features(d) for d in all_data]
    print(f"Loaded {len(all_data)} records "
          f"(LPE: {sum(1 for d in all_data if d['condition']=='LPE')}, "
          f"HPE: {sum(1 for d in all_data if d['condition']=='HPE')})\n")

    all_results = {}

    # ============================================================
    # Approach A: Mixed (LPE + HPE combined)
    # ============================================================
    print("=" * 70)
    print("  APPROACH A: Mixed (ignore condition)")
    print("=" * 70)
    clusters_a, _ = do_clustering(features_list)
    results_a = print_cluster_results(all_data, features_list, clusters_a, "A")
    all_results["A_mixed"] = results_a

    # ============================================================
    # Approach B: Split by condition
    # ============================================================
    print("=" * 70)
    print("  APPROACH B: Split by Condition")
    print("=" * 70)

    results_b = {}
    for cond in ["LPE", "HPE"]:
        print(f"\n  --- {cond} ---")
        cond_indices = [i for i, d in enumerate(all_data) if d["condition"] == cond]
        cond_data = [all_data[i] for i in cond_indices]
        cond_features = [features_list[i] for i in cond_indices]

        if len(cond_features) < 3:
            print(f"  Only {len(cond_features)} records, skipping clustering.")
            continue

        # Try 3 clusters, but fall back to 2 if a cluster would be empty
        n_clust = min(3, len(cond_features))
        clusters_b, _ = do_clustering(cond_features, n_clusters=n_clust)
        r = print_cluster_results(cond_data, cond_features, clusters_b, f"B_{cond}")
        results_b[cond] = r

    all_results["B_split"] = results_b

    # ============================================================
    # Approach C: Condition as feature
    # ============================================================
    print("=" * 70)
    print("  APPROACH C: Condition as Feature")
    print("=" * 70)
    # Encode condition: LPE=0, HPE=1
    extra = [[1.0 if f["condition"] == "HPE" else 0.0] for f in features_list]
    clusters_c, _ = do_clustering(features_list, extra_features=extra)
    results_c = print_cluster_results(all_data, features_list, clusters_c, "C")
    all_results["C_condition_feature"] = results_c

    # ============================================================
    # Summary comparison
    # ============================================================
    print("=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    print()

    print("  Approach A (Mixed):")
    for name, r in results_a.items():
        print(f"    {name}: n={r['n_records']}, mean_eng={r['mean_engagement']:.2f}, "
              f"conds={r['conditions']}")

    print()
    print("  Approach B (Split):")
    for cond, cond_results in results_b.items():
        for name, r in cond_results.items():
            print(f"    [{cond}] {name}: n={r['n_records']}, mean_eng={r['mean_engagement']:.2f}")

    print()
    print("  Approach C (Condition as Feature):")
    for name, r in results_c.items():
        print(f"    {name}: n={r['n_records']}, mean_eng={r['mean_engagement']:.2f}, "
              f"conds={r['conditions']}")

    print()
    print("  DECISION GUIDE:")
    print("  - If B's per-condition clusters have meaningful transitions")
    print("    (off-diagonal > 0.01), prefer B for richer calibration.")
    print("  - If B's clusters are too small or have empty rows,")
    print("    prefer A or C as fallback.")
    print("  - C is best if condition matters but sample size is limited.")
    print()

    # Save all results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "calibration_comparison.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  All results saved to: {output_file}")
    print()
