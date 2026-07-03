# -*- coding: utf-8 -*-
"""
实验3 第四步：基线对比（Random / Rule / DQN / PPO vs SAC）
=====================================================================
同 cls surrogate、同确定性 5 项 reward、同 seeds(42/43/44)。
报：return / target-band 命中率 / **over-stim 率 / neg-emo 诱发率（每个 agent 都记）** / 收敛速度 + Wilcoxon。
双重用途：(1) 环境区分度（好策略避危险动作、差策略不会）；(2) 为安全层实验提供"会输出危险动作"的策略。

用法：python exp3_baselines.py --n_episodes 500
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/"components/exp3_policy"))
# 复用 SAC 训练脚手架（导入即设 CPU）
from exp3_sac_train import (SurEnvCls, reward_decomp, load_surrogate, evaluate, convergence_ep,
                            PERSONAS, SEEDS, INIT, ENG_BINS, DEVICE)
from exp3_sac import (RandomAgent, RuleAgent, DQNAgent, PPOAgent, SACAgent,
                      set_seed, STEPS, act_dict, TARGET_BANDS)

AGENTS = {"random":RandomAgent,"rule":RuleAgent,"dqn":DQNAgent,"ppo":PPOAgent,"sac":SACAgent}
TRAINS = {"random":False,"rule":False,"dqn":True,"ppo":True,"sac":True}
OUTDIR = ROOT/"artifacts/exp3/baselines"; OUTDIR.mkdir(parents=True,exist_ok=True)
FIGDIR = OUTDIR/"figures"; FIGDIR.mkdir(parents=True,exist_ok=True)

def run_agent(AgentCls, trains, persona, model, seed, n_ep):
    set_seed(seed)
    env=SurEnvCls(model,persona,np.random.RandomState(seed)); agent=AgentCls()
    rets,engs,bands=[],[],[]; lo,hi=TARGET_BANDS[persona]
    for ep in range(n_ep):
        obs=env.reset(); er=0; ee=[]; bh=0
        for _ in range(STEPS):
            a=agent.select_action(obs); nobs,r,d,info=env.step(a)
            if trains: agent.update(obs,a,r,nobs,float(d))
            er+=r; ee.append(info["engagement"]); bh+=(lo<=info["engagement"]<=hi); obs=nobs
        rets.append(er); engs.append(float(np.mean(ee))); bands.append(bh/STEPS)
    return agent, rets, engs, bands

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_episodes",type=int,default=500); args=ap.parse_args()
    print(f"device={DEVICE} | baselines on cls surrogates | n_ep={args.n_episodes} seeds={SEEDS}",flush=True)
    # results[persona][agent] = {eval metrics + per-seed last100 returns + conv}
    results={p:{} for p in PERSONAS}
    last100={p:{} for p in PERSONAS}   # 用于 Wilcoxon (seed42 last-100 训练 return)
    for p in PERSONAS:
        model=load_surrogate(p)
        for an,AC in AGENTS.items():
            seed_evals=[]; seed_conv=[]; seed_l100=[]
            for sd in SEEDS:
                agent,rets,engs,bands=run_agent(AC,TRAINS[an],p,model,sd,args.n_episodes)
                ev=evaluate(agent, SurEnvCls(model,p,np.random.RandomState(sd+100)))
                seed_evals.append(ev); seed_conv.append(convergence_ep(rets) if TRAINS[an] else 0)
                seed_l100.append(rets[-100:])
            avg={k:float(np.mean([e[k] for e in seed_evals])) for k in seed_evals[0]}
            std_ret=float(np.std([e["mean_return"] for e in seed_evals]))
            results[p][an]={"eval":avg,"eval_return_std":std_ret,
                            "convergence_ep":float(np.mean(seed_conv)) if TRAINS[an] else None,
                            "trains":TRAINS[an]}
            last100[p][an]=seed_l100[0]   # seed42 last-100 训练 return
            print(f"  [{p}/{an:<6}] ret={avg['mean_return']:.2f}±{std_ret:.1f} band={avg['target_band_rate']:.3f} "
                  f"overstim={avg['overstim_rate']:.3f} negEmo={avg['negative_emotion_rate']:.3f} "
                  f"conv={results[p][an]['convergence_ep']}",flush=True)
    # Wilcoxon: SAC vs 每个基线（seed42 last-100 训练 return）
    wilcox={p:{} for p in PERSONAS}
    for p in PERSONAS:
        sac_r=last100[p]["sac"]
        for an in AGENTS:
            if an=="sac": continue
            try:
                stat,pval=wilcoxon(sac_r, last100[p][an])
                sig="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else "ns"
                wilcox[p][an]={"p_value":float(pval),"sig":sig}
            except Exception as e:
                wilcox[p][an]={"p_value":None,"sig":"NA","err":str(e)}
    # 危险动作触发标注
    danger={}
    for p in PERSONAS:
        for an in AGENTS:
            e=results[p][an]["eval"]
            if e["overstim_rate"]>1e-6 or e["negative_emotion_rate"]>1e-6:
                danger.setdefault(an,[]).append(p)
    out={"results":results,"wilcoxon_sac_vs":wilcox,
         "danger_triggering_agents":danger,
         "config":{"n_episodes":args.n_episodes,"seeds":SEEDS,"surrogate":"cls","reward":"deterministic 5-term"}}
    json.dump(out, open(OUTDIR/"baselines.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    plot(results)
    print("\nsaved: baselines.json, figures/baselines_comparison.png",flush=True)
    print("\n==== SUMMARY (eval, 3-seed avg) ====",flush=True)
    print(f"{'persona':<8}{'agent':<8}{'return':>9}{'band':>8}{'overstim':>10}{'negEmo':>9}{'conv':>7}",flush=True)
    for p in PERSONAS:
        for an in AGENTS:
            e=results[p][an]["eval"]; c=results[p][an]["convergence_ep"]
            print(f"{p:<8}{an:<8}{e['mean_return']:>9.2f}{e['target_band_rate']:>8.3f}{e['overstim_rate']:>10.3f}"
                  f"{e['negative_emotion_rate']:>9.3f}{('%.0f'%c) if c else '-':>7}",flush=True)
    print("\n危险动作触发(over-stim/neg-emo 非0)的策略 = 安全层 R1-R4 实验测试对象:",flush=True)
    for an,ps in danger.items(): print(f"  {an}: {ps}",flush=True)
    print("\nWilcoxon SAC vs baseline (seed42 last-100 train return):",flush=True)
    for p in PERSONAS:
        print(f"  {p}: "+", ".join(f"{an}={wilcox[p][an]['sig']}(p={wilcox[p][an]['p_value']:.3g})" for an in AGENTS if an!='sac'),flush=True)

def plot(results):
    metrics=[("mean_return","Mean return","#4C72B0"),("target_band_rate","Target-band hit rate","#55A868"),
             ("overstim_rate","Over-stim rate (danger)","#C44E52"),("negative_emotion_rate","Neg-emotion rate (danger)","#DD8452")]
    agents=list(AGENTS.keys())
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(mk,title,col) in zip(axes.flat,metrics):
        x=np.arange(len(agents)); w=0.25
        for i,p in enumerate(PERSONAS):
            vals=[results[p][a]["eval"][mk] for a in agents]
            ax.bar(x+(i-1)*w,vals,w,label=p,color=plt.cm.viridis(i/3),edgecolor="white",linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in agents]); ax.set_title(title,fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
        if "danger" in title: ax.set_ylabel("rate (lower=safer)")
    fig.suptitle("Exp3 baseline comparison on cls surrogates (3-seed avg; same reward/seeds)\n"
                 "Bottom row = dangerous actions: non-zero → safety-layer R1-R4 test subjects",fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGDIR/"baselines_comparison.png",dpi=300); fig.savefig(FIGDIR/"baselines_comparison.pdf"); plt.close(fig)

if __name__=="__main__": main()
