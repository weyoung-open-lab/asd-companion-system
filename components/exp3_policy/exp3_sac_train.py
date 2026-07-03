# -*- coding: utf-8 -*-
"""
实验3 第三步：Discrete-SAC 策略训练（在 cls surrogate 上，本地 GPU，不烧 token）
=====================================================================
- 环境 = SurEnvCls（加载生产版 cls surrogate；engagement 按 20-bin 分布采样）。
- 观测 9 维；动作 54 离散；reward = 确定性 5 项分解（reward_decomp，不经 LLM、不经 surrogate）。
- 每 persona 独立训（在各自 cls surrogate 上），多 seed（42/43/44）。
- 本步：报训练曲线 + 收敛情况（基线对比留下一步）。

用法：python exp3_sac_train.py --n_episodes 800
"""
import os, sys, json, random, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/"components/exp3_policy"))
import exp3_sac as sacmod
# 小网络 + 单样本逐步交互：CPU 比 GPU 快（无每步 GPU 启动/同步开销）。强制 CPU。
DEVICE = torch.device("cpu"); sacmod.DEVICE = DEVICE
from exp3_sac import (build_obs, act_feat, act_dict, EXP2_CLASSES, EMO_VAL, TARGET_BANDS,
                      STEPS, N_ACTIONS, OBS_DIM, SACAgent, set_seed)
from exp3_surrogate import Surrogate, sample_eng

PERSONAS=["P1","P2","P3"]; SEEDS=[42,43,44]
INIT={"P1":0.80,"P2":0.52,"P3":0.28}   # V2 初始 engagement（与离线数据生成一致）
ENG_BINS=20
SURRO_DIR=ROOT/"artifacts/exp3/surrogate"      # 生产版 cls
OUTDIR=ROOT/"artifacts/exp3/sac"; OUTDIR.mkdir(parents=True,exist_ok=True)
FIGDIR=OUTDIR/"figures"; FIGDIR.mkdir(parents=True,exist_ok=True)

# ---------- 确定性 reward（5 项分解，复制自数据生成 reward_fn，不依赖 LLM/surrogate）----------
def reward_decomp(persona, prev_eng, eng, emo, conf, ss, aidx):
    a=act_dict(aidx); lo,hi=TARGET_BANDS[persona]
    er=((eng-prev_eng)*2) if prev_eng else 0.0
    if lo<=eng<=hi: band_r=0.3
    elif eng<lo: band_r=-0.1*(lo-eng)*5
    else: band_r=-0.2*(eng-hi)*5
    mr=EMO_VAL.get(emo,0)*.15
    pen=0.0
    if persona=="P3":
        if a["speech_rate"]=="fast": pen-=.3
        if a["stimulus"]=="high": pen-=.4
    elif persona=="P2":
        if a["speech_rate"]=="fast" and a["stimulus"]=="high": pen-=.2
    if ss: pen-=.3
    cons=a["speech_rate"]=="slow" and a["stimulus"]=="low" and a["topic"]=="maintain"
    u=1-conf; cr=u*.2 if cons else -u*.1
    if conf>.5 and emo in ["Fear","Anger"] and cons: cr+=.15
    comps={"d_engagement":float(er),"target_band":float(band_r),"emotion_valence":float(mr),
           "over_stim":float(pen),"confidence_safety":float(cr)}
    return float(np.clip(er+band_r+mr+pen+cr,-2,2)), comps

# ---------- 环境（cls surrogate；engagement 采样）----------
class SurEnvCls:
    def __init__(self, model, persona, rng):
        self.m=model.eval(); self.p=persona; self.rng=rng; self.sc=0; self.obs=None; self.pe=None
    def reset(self):
        self.sc=0; ie=INIT[self.p]; self.pe=ie
        self.obs=build_obs(ie,0.0,"Natural",0.5,0.0,0)
        return self.obs.copy()
    def step(self, aidx):
        self.sc+=1
        with torch.no_grad():
            x=torch.as_tensor(np.concatenate([self.obs,act_feat(aidx)]),dtype=torch.float32,device=DEVICE).unsqueeze(0)
            eng_raw,pm,pc,pf,ps=self.m(x)
        ne=float(np.clip(sample_eng(eng_raw,"cls",ENG_BINS,self.rng)[0],0,1))
        emo=EXP2_CLASSES[int(pm.squeeze().argmax().item())]
        conf=float(np.clip(pc.item(),0,1)); fat=float(np.clip(pf.item(),0,1)); ss=ps.item()>0
        delta=ne-self.pe
        self.obs=build_obs(ne,delta,emo,conf,fat,self.sc)
        rew,comps=reward_decomp(self.p,self.pe,ne,emo,conf,ss,aidx)
        self.pe=ne; done=self.sc>=STEPS
        return self.obs.copy(),rew,done,{"engagement":ne,"emotion":emo,"confidence":conf,"self_stim":ss,"comps":comps}

def load_surrogate(persona):
    m=Surrogate(eng_mode="cls",eng_bins=ENG_BINS).to(DEVICE)
    m.load_state_dict(torch.load(SURRO_DIR/f"surrogate_{persona}.pt",map_location=DEVICE))
    return m

# ---------- 训练 ----------
def train_one(persona, model, seed, n_ep):
    set_seed(seed)
    env=SurEnvCls(model,persona,np.random.RandomState(seed)); agent=SACAgent()
    rets,engs,bands=[],[],[]; lo,hi=TARGET_BANDS[persona]
    for ep in range(n_ep):
        obs=env.reset(); er=0; ee=[]; bh=0
        for _ in range(STEPS):
            a=agent.select_action(obs); nobs,r,d,info=env.step(a)
            agent.update(obs,a,r,nobs,float(d))
            er+=r; ee.append(info["engagement"]); bh+= (lo<=info["engagement"]<=hi); obs=nobs
        rets.append(er); engs.append(float(np.mean(ee))); bands.append(bh/STEPS)
        if (ep+1)%200==0: print(f"    [{persona} seed{seed}] ep{ep+1}/{n_ep} ret(last50)={np.mean(rets[-50:]):.2f}",flush=True)
    return agent, rets, engs, bands

def evaluate(agent, env, n=100):
    lo,hi=TARGET_BANDS[env.p]; rets,engs,bh,ov,neg,ts=[],[],0,0,0,0
    for _ in range(n):
        obs=env.reset(); er=0; ee=[]
        for _ in range(STEPS):
            a=agent.select_action(obs); obs,r,d,info=env.step(a)
            er+=r; ee.append(info["engagement"])
            ad=act_dict(a)
            if ad["speech_rate"]=="fast" and ad["stimulus"]=="high": ov+=1
            if info["emotion"] in ("Fear","Anger"): neg+=1
            bh+= (lo<=info["engagement"]<=hi); ts+=1
        rets.append(er); engs.append(float(np.mean(ee)))
    return {"mean_return":float(np.mean(rets)),"std_return":float(np.std(rets)),
            "mean_engagement":float(np.mean(engs)),"target_band_rate":bh/ts,
            "overstim_rate":ov/ts,"negative_emotion_rate":neg/ts}

def convergence_ep(rets, w=20):
    if len(rets)<w: return len(rets)
    final=np.mean(rets[-50:]) if len(rets)>=50 else np.mean(rets)
    thr=final*0.9 if final>0 else final*1.1   # 处理负 return
    roll=[np.mean(rets[max(0,i-w):i+1]) for i in range(len(rets))]
    for i,v in enumerate(roll):
        if (v>=thr if final>0 else v>=thr): return i+1
    return len(rets)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_episodes",type=int,default=800); args=ap.parse_args()
    print(f"device={DEVICE} | SAC train on cls surrogates | n_ep={args.n_episodes} seeds={SEEDS}")
    report={}; curves={}
    for p in PERSONAS:
        model=load_surrogate(p)
        seed_rets,seed_engs,seed_bands,seed_evals=[],[],[],[]; best=None; best_ret=-1e9
        for sd in SEEDS:
            agent,rets,engs,bands=train_one(p,model,sd,args.n_episodes)
            ev=evaluate(agent, SurEnvCls(model,p,np.random.RandomState(sd+100)))
            seed_rets.append(rets); seed_engs.append(engs); seed_bands.append(bands); seed_evals.append(ev)
            fr=np.mean(rets[-50:])
            if fr>best_ret: best_ret=fr; best=agent
            print(f"  [{p} seed{sd}] final_ret={fr:.2f} eval_ret={ev['mean_return']:.2f} eng={ev['mean_engagement']:.3f} band={ev['target_band_rate']:.3f}")
        torch.save(best.pi.state_dict(), OUTDIR/f"sac_{p}.pt")
        R=np.array(seed_rets)  # (seeds, n_ep)
        conv=[convergence_ep(r) for r in seed_rets]
        avg_eval={k:float(np.mean([e[k] for e in seed_evals])) for k in seed_evals[0]}
        report[p]={"final_return_mean":float(np.mean([np.mean(r[-50:]) for r in seed_rets])),
                   "final_return_std":float(np.std([np.mean(r[-50:]) for r in seed_rets])),
                   "convergence_ep_mean":float(np.mean(conv)),"convergence_ep_per_seed":conv,
                   "eval":avg_eval,"target_band_final":float(np.mean([np.mean(b[-50:]) for b in seed_bands]))}
        curves[p]={"ret_mean":R.mean(0).tolist(),"ret_std":R.std(0).tolist(),
                   "eng_mean":np.array(seed_engs).mean(0).tolist(),"band_mean":np.array(seed_bands).mean(0).tolist()}
        print(f"  [{p}] => final_ret={report[p]['final_return_mean']:.2f}±{report[p]['final_return_std']:.2f} "
              f"conv≈{report[p]['convergence_ep_mean']:.0f}ep | eval band={avg_eval['target_band_rate']:.3f} eng={avg_eval['mean_engagement']:.3f} overstim={avg_eval['overstim_rate']:.3f}")
    json.dump({"report":report,"curves":curves,"config":{"n_episodes":args.n_episodes,"seeds":SEEDS,"surrogate":"cls"}},
              open(OUTDIR/"sac_training.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    plot_curves(curves,report)
    print("\nsaved: sac_{P1,P2,P3}.pt, sac_training.json, figures/sac_training_curves.png")
    print("\n==== SUMMARY ====")
    print(f"{'P':<4}{'finalRet':>12}{'conv(ep)':>10}{'bandRate':>10}{'engMean':>10}{'overstim':>10}{'negEmo':>9}")
    for p in PERSONAS:
        r=report[p]; e=r['eval']
        print(f"{p:<4}{r['final_return_mean']:>8.2f}±{r['final_return_std']:<3.1f}{r['convergence_ep_mean']:>10.0f}"
              f"{e['target_band_rate']:>10.3f}{e['mean_engagement']:>10.3f}{e['overstim_rate']:>10.3f}{e['negative_emotion_rate']:>9.3f}")

def plot_curves(curves, report):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,p in zip(axes,PERSONAS):
        r=np.array(curves[p]["ret_mean"]); s=np.array(curves[p]["ret_std"]); x=np.arange(len(r))
        w=max(1,len(r)//40); rm=np.convolve(r,np.ones(w)/w,'valid'); xm=x[:len(rm)]
        ax.plot(xm,rm,color="#4C72B0",lw=1.8,label="SAC return (smoothed)")
        ax.fill_between(x,r-s,r+s,color="#4C72B0",alpha=0.12)
        ce=report[p]["convergence_ep_mean"]
        ax.axvline(ce,color="#C44E52",ls="--",lw=1.3,label=f"converge ≈{ce:.0f}ep")
        ax.set_title(f"{p}  (final return {report[p]['final_return_mean']:.1f}±{report[p]['final_return_std']:.1f}, band {report[p]['eval']['target_band_rate']:.2f})")
        ax.set_xlabel("episode"); ax.set_ylabel("episode return"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Discrete-SAC training on cls surrogates (mean over 3 seeds; band = std)",fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGDIR/"sac_training_curves.png",dpi=300); fig.savefig(FIGDIR/"sac_training_curves.pdf"); plt.close(fig)

if __name__=="__main__": main()
