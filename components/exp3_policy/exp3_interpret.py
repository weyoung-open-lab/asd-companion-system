# -*- coding: utf-8 -*-
"""
实验3 第六步：可解释 + 临床框架实验（本地，不烧 token，数据实读）
=====================================================================
论点：本系统每个决策可拆解为可理解、且临床相关的贡献项——reward 的确定性 5 项分解
直接对应临床意图（target_band=最优唤醒；over_stim/emotion/confidence=安全），策略可审计、可解释。

Part A  Reward Decomposition 逐步归因（代表性 episode）+ 各 persona reward 项占比
Part B  Counterfactual：扰动 confidence / engagement / emotion → 动作 + 安全层响应（临床合理性）
Part C  安全层拦截的规则×维度归因案例

诚实口径：surrogate 内指标，不声称真实疗效；临床锚定 = 受最优唤醒/SOR-焦虑/诚实弃权启发的工程化形式化。
用法：python exp3_interpret.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/"components/exp3_policy"))
from exp3_sac_train import SurEnvCls, reward_decomp, load_surrogate, PERSONAS, INIT, DEVICE
from exp3_sac import SACAgent, set_seed, STEPS, act_dict, build_obs, EXP2_CLASSES, TARGET_BANDS
from exp3_safety import shield, load_sac, is_aggressive, is_conservative, CONSERVATIVE

OUTDIR=ROOT/"artifacts/exp3/interpret"; OUTDIR.mkdir(parents=True,exist_ok=True)
FIGDIR=OUTDIR/"figures"; FIGDIR.mkdir(parents=True,exist_ok=True)
TERMS=["d_engagement","target_band","emotion_valence","over_stim","confidence_safety"]
TERM_LABEL={"d_engagement":"Δengagement","target_band":"target-band (★arousal)","emotion_valence":"emotion-valence",
            "over_stim":"over-stim (★safety)","confidence_safety":"confidence-safety (★safety)"}
COL={"d_engagement":"#4C72B0","target_band":"#55A868","emotion_valence":"#8172B3","over_stim":"#C44E52","confidence_safety":"#DD8452"}

def sac_argmax(agent, obs):
    with torch.no_grad():
        p=agent.pi(torch.as_tensor(np.asarray(obs,dtype=np.float32),device=DEVICE))
    return int(p.argmax().item())

# ---------------- Part A ----------------
def part_A(models, sac):
    print("== Part A: reward 分解逐步归因 + 占比 ==",flush=True)
    rep_ep={}; prop={}; meanc={}
    for p in PERSONAS:
        set_seed(42); env=SurEnvCls(models[p],p,np.random.RandomState(42)); ag=sac[p]
        eps=[]
        for _ in range(100):
            obs=env.reset(); steps=[]; tot=0
            for _ in range(STEPS):
                a=ag.select_action(obs); nobs,r,d,info=env.step(a)
                steps.append({"comps":info["comps"],"reward":r,"engagement":info["engagement"],"action":a}); tot+=r; obs=nobs
            eps.append({"steps":steps,"return":tot})
        rets=[e["return"] for e in eps]; mr=np.mean(rets)
        rep=min(eps,key=lambda e:abs(e["return"]-mr))   # 代表性 episode = return 最接近均值
        rep_ep[p]=rep
        # 占比：所有 step 所有 episode，每项 mean 与 mean|.|
        allc={t:[] for t in TERMS}
        for e in eps:
            for s in e["steps"]:
                for t in TERMS: allc[t].append(s["comps"][t])
        mc={t:float(np.mean(allc[t])) for t in TERMS}
        mabs={t:float(np.mean(np.abs(allc[t]))) for t in TERMS}; tot_abs=sum(mabs.values())
        prop[p]={t:mabs[t]/tot_abs for t in TERMS}; meanc[p]=mc
        print(f"  [{p}] 代表episode return={rep['return']:.2f} | 占比(|贡献|): "
              +", ".join(f"{t.split('_')[0]}={prop[p][t]*100:.0f}%" for t in TERMS),flush=True)
    return rep_ep, prop, meanc

def plot_A(rep_ep, prop):
    # Fig A1: 代表 episode 逐步 5 项贡献(线) + 总 reward
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),sharey=True)
    for ax,p in zip(axes,PERSONAS):
        steps=np.arange(1,STEPS+1); S=rep_ep[p]["steps"]
        for t in TERMS:
            ax.plot(steps,[s["comps"][t] for s in S],color=COL[t],lw=1.6,label=TERM_LABEL[t] if p=="P1" else None)
        ax.plot(steps,[s["reward"] for s in S],color="black",lw=2.0,ls="--",label="total reward" if p=="P1" else None)
        ax.axhline(0,color="gray",lw=0.6); ax.set_title(f"{p} (representative episode, return={rep_ep[p]['return']:.1f})")
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("reward contribution")
    fig.legend(loc="upper center",ncol=6,bbox_to_anchor=(0.5,1.06),frameon=False,fontsize=9)
    fig.suptitle("Part A — per-step reward decomposition (deterministic 5-term, hand-reproducible)",y=1.12,fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGDIR/"A1_stepwise_decomposition.png",dpi=300,bbox_inches="tight"); fig.savefig(FIGDIR/"A1_stepwise_decomposition.pdf",bbox_inches="tight"); plt.close(fig)
    # Fig A2: 各 persona 占比
    fig,ax=plt.subplots(figsize=(9,4.6)); x=np.arange(len(PERSONAS)); w=0.16
    for i,t in enumerate(TERMS):
        ax.bar(x+(i-2)*w,[prop[p][t] for p in PERSONAS],w,color=COL[t],label=TERM_LABEL[t],edgecolor="white",linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(PERSONAS); ax.set_ylabel("mean |contribution| proportion")
    ax.set_title("Part A — reward-term influence per persona (what the policy weighs)",fontweight="bold")
    ax.legend(fontsize=8,ncol=2); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGDIR/"A2_term_proportion.png",dpi=300); fig.savefig(FIGDIR/"A2_term_proportion.pdf"); plt.close(fig)

# ---------------- Part B ----------------
def decode(a): d=act_dict(a); return f"{d['speech_rate']}/{d['stimulus']}/{d['topic']}/{d['encouragement']}"

def _rew(persona, obs, aidx):
    eng=float(obs[0]); emo=EXP2_CLASSES[int(np.argmax(obs[2:6]))]; conf=float(obs[6])
    return reward_decomp(persona, eng, eng, emo, conf, False, aidx)

def cf_case(name, persona, sac, base_kw, pert_kw, perturb_dim, key_term, clinical, abstain_pert=False):
    """反事实：扰动一个临床维度，看 (1) reward 结构如何偏向保守 (2) 安全层对危险动作的确定性判决 (3) 学到的 SAC 动作。"""
    base=build_obs(**base_kw); pert=build_obs(**pert_kw)
    # 关键临床项：扰动后状态下 aggressive vs conservative 的 reward(关键项 + 总)
    rb_a,cb_a=_rew(persona,base,53); rb_c,cb_c=_rew(persona,base,CONSERVATIVE)
    rp_a,cp_a=_rew(persona,pert,53); rp_c,cp_c=_rew(persona,pert,CONSERVATIVE)
    # 安全层对"激进动作"的判决（前后）
    _,ib,fb=shield(base,53); _,ip,fp=shield(pert,53,abstain=abstain_pert)
    a_sac=sac_argmax(sac[persona],pert)
    def fmt(kw): return {k:(v if isinstance(v,str) else round(float(v),2)) for k,v in kw.items() if k in("eng","conf","fatigue","emo_4cls")}
    return {"name":name,"persona":persona,"perturb":perturb_dim,"key_term":key_term,
            "base_state":fmt(base_kw),"pert_state":fmt(pert_kw),
            "key_term_base_agg":round(cb_a[key_term],3) if key_term else None,"key_term_pert_agg":round(cp_a[key_term],3) if key_term else None,
            "key_term_base_con":round(cb_c[key_term],3) if key_term else None,"key_term_pert_con":round(cp_c[key_term],3) if key_term else None,
            "reward_pert_agg":round(rp_a,3),"reward_pert_con":round(rp_c,3),
            "shield_aggr_base":("intercept["+",".join(fb)+"]" if ib else "allow"),
            "shield_aggr_pert":("intercept["+",".join(fp)+"]" if ip else "allow"),
            "sac_argmax":decode(a_sac),"clinical":clinical}

def part_B(sac):
    print("\n== Part B: counterfactual (reward 结构 + 安全层确定性判决) ==",flush=True)
    cases=[]
    cases.append(cf_case("置信度↓ (0.90→0.20)","P2",sac,
        dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.90,fatigue=0.0,step=10),
        dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.20,fatigue=0.0,step=10),
        "confidence 0.90→0.20","confidence_safety",
        "低置信→confidence_safety 项使保守动作奖励↑、激进↓(谨慎原则)；安全层 R3 进一步确定性强制保守"))
    cases.append(cf_case("感知弃权 (abstain)","P2",sac,
        dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.90,fatigue=0.0,step=10),
        dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.90,fatigue=0.0,step=10),
        "perception abstain=True",None,
        "感知弃权(不可靠信号)→安全层 R3 确定性强制保守(reward 看不到 abstain，故由硬规则兜底)",
        abstain_pert=True))
    cases.append(cf_case("参与度↓ (0.50→0.15)","P3",sac,
        dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.70,fatigue=0.0,step=10),
        dict(eng=0.15,delta=-0.1,emo_4cls="Natural",conf=0.70,fatigue=0.0,step=10),
        "engagement 0.50→0.15","over_stim",
        "低参与+高刺激→over_stim 惩罚(P3 对 fast/high 重罚)；安全层 R1 确定性拦截(最优唤醒：不把低参与儿童推向过度唤醒)"))
    cases.append(cf_case("注入恐惧 (Natural→Fear)","P2",sac,
        dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.70,fatigue=0.0,step=10),
        dict(eng=0.50,delta=0.0,emo_4cls="Fear",conf=0.70,fatigue=0.0,step=10),
        "emotion Natural→Fear","emotion_valence",
        "负面情绪→emotion_valence 惩罚负面效价；安全层 R2 拦截升刺激(SOR-焦虑：强刺激与焦虑互相加剧)"))
    cases.append(cf_case("疲劳↑ (0.0→0.70)","P1",sac,
        dict(eng=0.80,delta=0.0,emo_4cls="Joy",conf=0.80,fatigue=0.0,step=20),
        dict(eng=0.80,delta=0.0,emo_4cls="Joy",conf=0.80,fatigue=0.70,step=20),
        "fatigue 0.0→0.70",None,
        "高疲劳→安全层 R4 拦截高强度(感觉餐：疲劳下持续高强度适得其反；reward 不含 fatigue 项，由硬规则兜底)"))
    for c in cases:
        kt=f"{c['key_term']}: agg {c['key_term_base_agg']}→{c['key_term_pert_agg']}, con {c['key_term_base_con']}→{c['key_term_pert_con']}" if c['key_term'] else "(reward 无对应项, 仅硬规则)"
        print(f"  [{c['name']}|{c['persona']}] {kt} | shield(aggr) {c['shield_aggr_base']}→{c['shield_aggr_pert']} | SAC argmax={c['sac_argmax']}",flush=True)
    return cases

# ---------------- Part C ----------------
def part_C(models, sac):
    """安全层拦截的规则×维度归因案例：构造危险动作在不同风险态下，展示哪条规则因哪个维度触发。"""
    print("\n== Part C: 安全层拦截规则×维度归因 ==",flush=True)
    AGG=53  # fast/high/switch/frequent
    cases=[]
    scen=[
        ("低参与态","P3",dict(eng=0.15,delta=0.0,emo_4cls="Natural",conf=0.7,fatigue=0.0,step=10),False,"engagement=0.15<0.33"),
        ("负面情绪态","P2",dict(eng=0.50,delta=0.0,emo_4cls="Fear",conf=0.7,fatigue=0.0,step=10),False,"pFear>0.30"),
        ("疲劳态","P1",dict(eng=0.80,delta=0.0,emo_4cls="Joy",conf=0.8,fatigue=0.70,step=20),False,"fatigue=0.70>0.50"),
        ("低置信态","P2",dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.20,fatigue=0.0,step=10),False,"confidence=0.20<0.40"),
        ("感知弃权","P2",dict(eng=0.50,delta=0.0,emo_4cls="Natural",conf=0.9,fatigue=0.0,step=10),True,"abstain=True"),
    ]
    for nm,p,kw,ab,dimdesc in scen:
        obs=build_obs(**kw); safe,intc,fired=shield(obs,AGG,abstain=ab)
        cases.append({"scenario":nm,"persona":p,"proposed_action":decode(AGG),"triggered_rules":fired,
                      "trigger_dim":dimdesc,"replaced_with":decode(safe),"intercepted":intc})
        print(f"  [{nm}] 危险动作 {decode(AGG)} -> 触发 {fired} (因 {dimdesc}) -> 替换为 {decode(safe)}",flush=True)
    return cases

def main():
    print(f"device={DEVICE} | interpretability experiment",flush=True)
    models={p:load_surrogate(p) for p in PERSONAS}
    sac={p:load_sac(p) for p in PERSONAS}
    rep_ep,prop,meanc=part_A(models,sac); plot_A(rep_ep,prop)
    cfB=part_B(sac); cfC=part_C(models,sac)
    out={"partA_proportion":prop,"partA_mean_contribution":meanc,
         "partA_representative_return":{p:rep_ep[p]["return"] for p in PERSONAS},
         "partB_counterfactual":cfB,"partC_safety_attribution":cfC}
    json.dump(out,open(OUTDIR/"interpret.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print("\nsaved: interpret.json, figures/A1_*, A2_*",flush=True)

if __name__=="__main__": main()
