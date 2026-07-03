"""Quick exploration of Performance data to find session boundaries"""
import pandas as pd
import numpy as np

ROOT = r"D:\project1\pythonProject\人机交互\files\Engagnition Dataset"

for cond, pid in [("LPE condition", "P20"), ("LPE condition", "P36"), ("HPE condition", "P39")]:
    print(f"{'='*60}")
    print(f"  {pid} ({cond.replace(' condition','')})")
    print(f"{'='*60}")

    eng = pd.read_csv(f"{ROOT}\\{cond}\\{pid}\\EngagementData.csv")
    perf = pd.read_csv(f"{ROOT}\\{cond}\\{pid}\\PerformanceData.csv")

    print(f"Engagement: {eng.SGTime.min():.2f} - {eng.SGTime.max():.2f} min, {len(eng)} rows")
    print(f"Performance: {perf.SGTime.min():.2f} - {perf.SGTime.max():.2f} min, {len(perf)} rows")
    print(f"Performance values: {sorted(perf.Performance.unique())}")
    print()

    # Find gaps in Performance time series (potential session boundaries)
    gaps = np.diff(perf.SGTime.values) * 60  # convert to seconds
    big_gaps = np.where(gaps > 5)[0]  # gaps > 5 seconds

    print(f"Time gaps > 5 seconds in Performance: {len(big_gaps)} found")
    for i in big_gaps[:25]:
        t1 = perf.SGTime.iloc[i]
        t2 = perf.SGTime.iloc[i + 1]
        gap_sec = gaps[i]

        # What was engagement during this gap?
        eng_during = eng[(eng.SGTime >= t1) & (eng.SGTime <= t2)]
        if len(eng_during) > 0:
            eng_mean = eng_during.Engagement.mean()
            eng_vals = eng_during.Engagement.value_counts().to_dict()
        else:
            eng_mean = -1
            eng_vals = {}

        print(f"  Gap {i+1}: {t1:.2f}-{t2:.2f} min (gap={gap_sec:.1f}s), "
              f"eng_during: mean={eng_mean:.2f}, {eng_vals}")

    # Also: try to match session count
    print(f"\nNumber of segments (gaps+1): {len(big_gaps)+1}")
    print(f"Expected sessions: 20")
    print()

    # Look at engagement changes over time (split into 5 equal time blocks)
    print("Engagement over time (5 equal blocks):")
    time_min = eng.SGTime.min()
    time_max = eng.SGTime.max()
    block_size = (time_max - time_min) / 5
    for b in range(5):
        t_start = time_min + b * block_size
        t_end = t_start + block_size
        block = eng[(eng.SGTime >= t_start) & (eng.SGTime < t_end)]
        if len(block) > 0:
            dist = block.Engagement.value_counts().sort_index()
            total = len(block)
            parts = [f"{v}={dist.get(v,0)/total*100:.0f}%" for v in [0,1,2]]
            print(f"  Block {b+1} ({t_start:.1f}-{t_end:.1f} min): "
                  f"mean={block.Engagement.mean():.2f}, {', '.join(parts)}")
    print()
