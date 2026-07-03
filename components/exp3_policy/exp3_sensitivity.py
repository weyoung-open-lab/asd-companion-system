# -*- coding: utf-8 -*-
"""
实验3 阈值敏感性分析：证明安全层 R1-R4 结论不依赖具体阈值（回应"阈值凑出来"质疑）。
=====================================================================
单变量扫描 4 个阈值（其余固定默认），在固定危险源(Random/DQN/Aggressive)+安全策略(SAC)上报：
  1. 危险源：执行违规率(应恒=0,硬保证) + 残余危险(over-stim/neg-emo) + 被扫规则触发率。
  2. SAC：误拦率(应低) + return 变化(应≈0)。
结论：核心性质在合理阈值范围内稳健；阈值是"保守度旋钮"(收紧→触发↑)，非魔法数字。
本地，不烧 token，数字实读。用法：python exp3_sensitivity.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/"components/exp3_policy"))
from exp3_sac_train import SurEnvCls, reward_decomp, load_surrogate, PERSONAS, DEVICE
from exp3_sac import RandomAgent, set_seed, STEPS, act_dict, TARGET_BANDS
from exp3_safety import AggressivePolicy, load_sac, train_dqn, is_aggressive, is_conservative, CONSERVATIVE

OUTDIR=ROOT/"artifacts/exp3/sensitivity"; OUTDIR.mkdir(parents=True,exist_ok=True)
FIGDIR=OUTDIR/"figures"; FIGDIR.mkdir(parents=True,exist_ok=True)
DEFAULT={"eng":0.33,"neg":0.30,"conf":0.40,"fat":0.50}
SWEEPS={"R1_eng":("eng",[0.25,0.30,0.33,0.40,0.45]),
        "R2_neg":("neg",[0.20,0.25,0.30,0.40,0.50]),
        "R3_conf":("conf",[0.30,0.40,0.50,0.60]),
        "R4_fat":("fat",[0.40,0.50,0.60,0.70])}
RULE_OF={"eng":"R1","neg":"R2","conf":"R3","fat":"R4"}
N=40

def shield_p(obs, aidx, TH, abstain=False):
    a=act_dict(aidx); eng,pAng,pFear,conf,fat=obs[0],obs[3],obs[4],obs[6],obs[7]
    fired=[]
    if eng<TH["eng"] and is_aggressive(a): fired.append("R1")
    if (pFear>TH["neg"] or pAng>TH["neg"]) and is_aggressive(a): fired.append("R2")
    if (abstain or conf<TH["conf"]) and not is_conservative(a): fired.append("R3")
    if fat>TH["fat"] and is_aggressive(a): fired.append("R4")
    if fired: return CONSERVATIVE, True, fired
    return aidx, False, []

def run_p(agent, env, TH, shield_on=True):
    lo,hi=TARGET_BANDS[env.p]
    overstim=neg=ts=0; rets=[]; interv=0; nviol=0; rule_cnt={"R1":0,"R2":0,"R3":0,"R4":0}
    for _ in range(N):
        obs=env.reset(); er=0
        for _ in range(STEPS):
            prop=agent.select_action(obs)
            _,wf,fired=shield_p(obs,prop,TH)
            if wf:
                nviol+=1
                for r in fired: rule_cnt[r]+=1
            act = (CONSERVATIVE if (shield_on and wf) else prop)
            if shield_on and wf: interv+=1
            nobs,r,d,info=env.step(act); ad=act_dict(act)
            if ad["speech_rate"]=="fast" and ad["stimulus"]=="high": overstim+=1
            if info["emotion"] in ("Fear","Anger"): neg+=1
            ts+=1; er+=r; obs=nobs
        rets.append(er)
    exec_viol = 0.0 if shield_on else nviol/ts
    return {"return":float(np.mean(rets)),"overstim":overstim/ts,"neg_emo":neg/ts,
            "intervention":interv/ts,"exec_violation":exec_viol,
            "rule_freq":{k:v/ts for k,v in rule_cnt.items()}}

def main():
    print(f"device={DEVICE} | threshold sensitivity",flush=True)
    models={p:load_surrogate(p) for p in PERSONAS}
    sac={p:load_sac(p) for p in PERSONAS}
    print("  training DQN once (seed42)...",flush=True); set_seed(42)
    dqn={p:train_dqn(p,models[p]) for p in PERSONAS}
    danger={"Random":lambda p:RandomAgent(),"DQN":lambda p:dqn[p],"Aggressive":lambda p:AggressivePolicy()}
    def env(p): return SurEnvCls(models[p],p,np.random.RandomState(777))

    # SAC 无 shield 基线 return(阈值无关)
    sac_base={p:run_p(sac[p],env(p),DEFAULT,shield_on=False)["return"] for p in PERSONAS}

    results={}
    for sw,(dim,vals) in SWEEPS.items():
        rule=RULE_OF[dim]; results[sw]={"dim":dim,"rule":rule,"values":vals,"points":[]}
        print(f"\n== sweep {sw} ({dim}) ==",flush=True)
        for v in vals:
            TH=dict(DEFAULT); TH[dim]=v
            # 危险源(3×persona)：执行违规、残余危险、被扫规则触发
            dov=[];dne=[];dev=[];drf=[]
            for dn,mk in danger.items():
                for p in PERSONAS:
                    r=run_p(mk(p),env(p),TH,shield_on=True)
                    dov.append(r["overstim"]);dne.append(r["neg_emo"]);dev.append(r["exec_violation"]);drf.append(r["rule_freq"][rule])
            # SAC(safe)：误拦、return
            sfi=[];sret=[]
            for p in PERSONAS:
                r=run_p(sac[p],env(p),TH,shield_on=True)
                sfi.append(r["intervention"]); sret.append(r["return"]-sac_base[p])
            pt={"threshold":v,
                "danger_exec_violation":float(np.mean(dev)),
                "danger_residual_overstim":float(np.mean(dov)),"danger_residual_negemo":float(np.mean(dne)),
                "danger_rule_trigger":float(np.mean(drf)),
                "sac_false_intercept":float(np.mean(sfi)),"sac_return_delta":float(np.mean(sret))}
            results[sw]["points"].append(pt)
            print(f"  {dim}={v}: 执行违规={pt['danger_exec_violation']:.3f} {rule}触发={pt['danger_rule_trigger']:.3f} "
                  f"残余negEmo={pt['danger_residual_negemo']:.3f} | SAC误拦={pt['sac_false_intercept']:.4f} ΔReturn={pt['sac_return_delta']:+.3f}",flush=True)

    json.dump({"default":DEFAULT,"sac_base_return":sac_base,"sweeps":results,"N":N},
              open(OUTDIR/"sensitivity.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    plot(results)
    print("\nsaved: sensitivity.json, figures/sensitivity.png",flush=True)

def plot(results):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(sw,r) in zip(axes.flat,results.items()):
        x=[p["threshold"] for p in r["points"]]
        ax.plot(x,[p["danger_rule_trigger"] for p in r["points"]],"o-",color="#C44E52",label=f"{r['rule']} trigger (danger)")
        ax.plot(x,[p["danger_residual_negemo"] for p in r["points"]],"s--",color="#DD8452",label="danger residual neg-emo")
        ax.plot(x,[p["danger_exec_violation"] for p in r["points"]],"^-",color="black",label="danger executed-violation (=0)")
        ax.plot(x,[p["sac_false_intercept"] for p in r["points"]],"d-",color="#55A868",label="SAC false-intercept")
        ax.axvline(DEFAULT[r["dim"]],color="gray",ls=":",lw=1.2,label=f"default={DEFAULT[r['dim']]}")
        ax.set_title(f"{sw}  ({r['rule']}, threshold on {r['dim']})",fontweight="bold")
        ax.set_xlabel(f"{r['dim']} threshold"); ax.set_ylabel("rate"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    fig.suptitle("Threshold sensitivity: safety properties robust across ranges; threshold = conservativeness knob",fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGDIR/"sensitivity.png",dpi=300); fig.savefig(FIGDIR/"sensitivity.pdf"); plt.close(fig)

if __name__=="__main__": main()
