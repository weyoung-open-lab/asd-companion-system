# -*- coding: utf-8 -*-
"""等 pro 30-ep 轨迹落盘 -> 自动重算 N=30 std + 重渲图 + 报告。只读轨迹文件，不重生成。"""
import json, os, time, sys, subprocess
from collections import Counter
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
F = ROOT/"artifacts/exp1/traj_deepseek-v4-pro_P3.json"
DEADLINE = time.time() + 45*60   # 最多等 45 分钟

def n_eps():
    try: return len(json.load(open(F))["eps"])
    except Exception: return 0

while n_eps() < 30 and time.time() < DEADLINE:
    time.sleep(30)

n = n_eps()
if n < 30:
    print(f"TIMEOUT: pro 轨迹仍为 {n} 条(<30)，pro regen 可能变慢/卡死。std 维持 N=10 的 0.031 标注。")
    sys.exit(0)

eps = json.load(open(F))["eps"]
late = np.array([x[20:] for x in eps if len(x) == 30])
std = round(float(late.std(0).mean()), 4)
rows = [tuple(np.round(r, 6)) for r in late]; c = Counter(rows)
uniq = len(c); mode = c.most_common(1)[0][1]
print(f"PRO_N30_DONE std={std} unique={uniq}/30 mode={mode}/30")
# 重渲图（make_figures 会从 30-ep 文件读 pro 的 N=30 std）
r = subprocess.run([sys.executable, str(ROOT/"components/exp1_simulator/make_figures.py")],
                   capture_output=True, text=True)
print("figures re-rendered:", "OK" if r.returncode == 0 else r.stderr[-300:])
# 落一个状态文件便于读取
json.dump({"pro_n30_std": std, "unique": uniq, "mode": mode},
          open(ROOT/"artifacts/exp1/pro_n30_std.json", "w"))
