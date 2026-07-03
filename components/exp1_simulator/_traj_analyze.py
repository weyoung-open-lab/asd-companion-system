# -*- coding: utf-8 -*-
import json,sys
from pathlib import Path
import numpy as np
sys.path.append("components/exp1_simulator")
from exp1_ablation import eng_to_state
ROOT=Path(".")

def load_4o():
    d=json.load(open("artifacts/exp1/v2_official_N30_steps.json",encoding="utf-8"))
    EP={}
    for r in d["P3"]: EP.setdefault(r["ep"],[]).append((r["step"],r["fin"]))
    eps=[[e for _,e in sorted(v)] for k,v in sorted(EP.items())]
    return [e for e in eps if len(e)==30]

def load_ds(model):
    f=f"artifacts/exp1/traj_{model}_P3.json"
    if not Path(f).exists(): return None
    return json.load(open(f))["eps"]

def analyze(name,eps):
    A=np.array([e for e in eps if len(e)==30])
    late=A[:,20:30]
    st=np.vectorize(eng_to_state)(late)
    def upticks(seq): return int(np.sum(np.diff(seq)>0.02))
    ups=[upticks(late[i]) for i in range(late.shape[0])]
    back=0
    for i in range(late.shape[0]):
        s=np.vectorize(eng_to_state)(late[i])
        for j in range(1,len(s)):
            if s[j]==1 and s[j-1]==0: back+=1
    print(f"\n=== {name} (P3 late, n_ep={A.shape[0]}) ===")
    print(f"  多样性: 跨ep每step std均值={late.std(axis=0).mean():.4f} | ep均值的std={late.mean(axis=1).std():.4f} | 值域[{late.min():.2f},{late.max():.2f}]")
    print(f"  状态分布: S0={(st==0).mean()*100:.1f}% S1={(st==1).mean()*100:.1f}% S2={(st==2).mean()*100:.1f}%  (真实P3 late S1~10.9%)")
    print(f"  波动: 含上抬(Δ>.02)的ep={sum(1 for u in ups if u>0)}/{A.shape[0]} | 平均上抬/ep={np.mean(ups):.2f} | 纯单调ep={sum(1 for u in ups if u==0)}/{A.shape[0]}")
    print(f"  S0->S1回升事件={back}")

analyze("gpt-4o (N=30全量)",load_4o())
for m in ["deepseek-v4-flash","deepseek-v4-pro"]:
    e=load_ds(m)
    if e: analyze(m+" (前10ep复现)",e)
    else: print(f"\n{m}: 轨迹未就绪")
