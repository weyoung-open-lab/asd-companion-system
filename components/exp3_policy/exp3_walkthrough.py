# -*- coding: utf-8 -*-
"""
实验3 端到端集成 — Simulated Walkthrough（纯仿真走查，证明模块集成，不证明疗效）
=====================================================================
闭环：cls surrogate(儿童)→真实情绪→FERAC检索→真实Exp2感知(含弃权)→9维观测(标注真实/仿真)
      →SAC决策→安全层R1-R4→可解释(5项分解+规则归因)→执行回环。

★诚实边界：纯仿真，证明"模块能串起来协同"，绝不声称对真实ASD儿童有效；零真人；
  "交还人类"=仿真触发事件+暂停+呈现决策点，非真人接手。
本地（Exp2感知本地推理，不烧token；儿童用surrogate不烧token）。
用法：python exp3_walkthrough.py
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT)); sys.path.append(str(ROOT/"components/exp3_policy"))
from exp3_sac_train import load_surrogate, reward_decomp, INIT, DEVICE
from exp3_sac import set_seed, STEPS, act_dict, act_feat, build_obs, EXP2_CLASSES, TARGET_BANDS
from exp3_surrogate import sample_eng, Surrogate
from exp3_safety import shield, shield_tiered, gate_tier, load_sac, CONSERVATIVE
from system.inference.perception import Perception
from system.inference.contract import PerceptionConfig

OUTDIR=ROOT/"artifacts/exp3/walkthrough"; OUTDIR.mkdir(parents=True,exist_ok=True)
FIGDIR=OUTDIR/"figures"; FIGDIR.mkdir(parents=True,exist_ok=True)
ENG_BINS=20
EMO2DIR={"Natural":"Natural","Anger":"anger","Fear":"fear","Joy":"joy"}
FER_OUT=["natural","anger","fear","joy"]   # perception.classes_out 顺序 = obs dims 2-5

# 全局感知器（真实 Exp2：EfficientNet-B0 + 温度缩放 + 弃权门控）
PERC=Perception(checkpoint_path=str(ROOT/"models/ferac_efficientnetb0_deploy.pth"))
PCFG=PerceptionConfig(face_required=False, blur_threshold=0.0)  # FERAC 已是裁好的脸,跳过Haar(已知局限F2-6)

def retrieve_image(true_emo, rng):
    d=ROOT/f"data/ferac/test/{EMO2DIR[true_emo]}"
    imgs=sorted(glob.glob(str(d/"*")))
    return imgs[rng.randint(len(imgs))]

def perceive(true_emo, rng):
    """真实 Exp2 感知一张该情绪类的 FERAC 图 -> (4维概率, 置信度, abstain, decision, reason, 文件名, 预测类)"""
    img=retrieve_image(true_emo,rng); r=PERC.predict(img,PCFG)
    probs=np.array([r.emotion_probs.get(k,0.25) for k in FER_OUT],dtype=np.float32) if r.emotion_probs else np.full(4,0.25,np.float32)
    conf=float(r.confidence) if r.confidence is not None else float(probs.max())
    pred_class=FER_OUT[int(np.argmax(probs))]   # 预测类(用于 per-class 三级门控阈值)
    return probs, conf, (r.decision=="abstain"), r.decision, r.abstain_reason, os.path.basename(img), pred_class

class Child:
    """儿童 agent = cls surrogate，维护真实隐状态(engagement, 真实情绪, fatigue)，驱动动力学。"""
    def __init__(self, persona, model, rng):
        self.p=persona; self.m=model.eval(); self.rng=rng
    def reset(self):
        self.eng=INIT[self.p]; self.emo="Natural"; self.fat=0.0; self.prev=self.eng; self.sc=0
        return self.eng, self.emo, self.fat
    def step(self, aidx):
        self.sc+=1
        internal=build_obs(self.eng, self.eng-self.prev, self.emo, 0.7, self.fat, self.sc-1)
        with torch.no_grad():
            x=torch.as_tensor(np.concatenate([internal,act_feat(aidx)]),dtype=torch.float32,device=DEVICE).unsqueeze(0)
            eng_raw,pm,pc,pf,ps=self.m(x)
        ne=float(np.clip(sample_eng(eng_raw,"cls",ENG_BINS,self.rng)[0],0,1))
        self.emo=EXP2_CLASSES[int(pm.squeeze().argmax().item())]; self.fat=float(np.clip(pf.item(),0,1))
        self.prev=self.eng; self.eng=ne
        return self.eng, self.emo, self.fat

def run_walkthrough(persona, seed=7, distress_window=None):
    """distress_window=(a,b)：构造的应激场景——steps a..b 强制儿童真实情绪=Fear(模拟逐步崩溃)，
    用于演示'交还人类'触发(持续弃权+负面情绪+参与度不改善)。其余为自然走查。"""
    set_seed(seed); rng=np.random.RandomState(seed)
    child=Child(persona, load_surrogate(persona), np.random.RandomState(seed+1))
    sac=load_sac(persona); lo,hi=TARGET_BANDS[persona]; mid=(lo+hi)/2
    eng,true_emo,fat=child.reset(); prev_eng=eng
    perc=perceive(true_emo,rng)   # (probs,conf,abstain,decision,reason,img)
    obs=np.array([eng,0.0,*perc[0],perc[1],fat,0.0],dtype=np.float32)
    rec=[]; concern_win=[]; handback_step=None
    for step in range(1,STEPS+1):
        pprobs,pconf,pab,pdec,preason,pimg,ppred=perc
        # ★三级门控：per-class 阈值对校准置信度分档 (accept/cautious/abstain)
        tier=gate_tier(ppred,pconf)
        # SAC 决策(argmax 确定性) — 作用于当前观测(由当前感知构建)
        with torch.no_grad():
            a_raw=int(sac.pi(torch.as_tensor(obs,dtype=torch.float32,device=DEVICE)).argmax().item())
        # 安全层(三级适配)：accept→待命 / cautious→soft_cap / abstain→全保守
        safe_a, intc, rules, level = shield_tiered(obs, a_raw, perc_tier=tier)
        # 旧二元门控对照(反事实, 不执行)：感知自带 0.86 gate 的 abstain + 二元 shield
        old_safe, old_intc, old_rules = shield(obs, a_raw, abstain=pab)
        # 可解释：reward 5项分解(用当前感知情绪/置信)
        perc_emo=EXP2_CLASSES[int(np.argmax(obs[2:6]))]
        _, comps = reward_decomp(persona, prev_eng, float(obs[0]), perc_emo, float(obs[6]), False, safe_a)
        # R3 待命"在线"反事实：若此刻提议激进动作，三级安全层会怎样处理
        _, cf_intc, cf_rules, cf_level = shield_tiered(obs, 53, perc_tier=tier)
        # 交还人类监测：窗口内三级 abstain 档(感知持续不可靠) + 负面感知情绪(≥3/5)
        concern_win.append((tier=="abstain", perc_emo in ("Fear","Anger")))
        w=concern_win[-5:]
        if handback_step is None and step>=5 and sum(a for a,_ in w)>=3 and sum(b for _,b in w)>=3:
            handback_step=step
        rec.append({"step":step,
            "child_true":{"engagement":round(eng,3),"true_emotion":true_emo,"fatigue":round(fat,3)},  # 仿真信号
            "perception":{"retrieved_img":pimg,"decision":pdec,"abstain_reason":preason,             # 真实感知(Exp2)
                          "perceived_probs":{k:round(float(v),3) for k,v in zip(FER_OUT,pprobs)},"confidence":round(pconf,3),
                          "predicted_class":ppred,"gating_tier":tier},                                # ★三级档
            "obs_9d":{"engagement[SIM]":round(float(obs[0]),3),"delta[SIM]":round(float(obs[1]),3),
                      "emotion_probs[REAL]":[round(float(x),3) for x in obs[2:6]],"confidence[REAL]":round(float(obs[6]),3),
                      "fatigue[SIM]":round(float(obs[7]),3),"tau[SIM]":round(float(obs[8]),3)},
            "sac_action":decode(a_raw),
            "safety":{"intercepted":intc,"rules":rules,"executed":decode(safe_a),"level":level,        # ★ level: none/soft_cap/full_conservative
                      "R3_standby_counterfactual":{"aggressive_would_intercept":cf_intc,"rules":cf_rules,"level":cf_level}},
            "old_binary_gate":{"decision":pdec,"intercepted":bool(old_intc),"executed":decode(old_safe)},  # 旧二元对照
            "reward_decomposition":{k:round(v,3) for k,v in comps.items()},
            "handback_triggered":(handback_step==step)})
        # 执行 -> 儿童转移
        prev_eng=child.eng; eng,true_emo,fat=child.step(safe_a)
        if distress_window and distress_window[0]<=step+1<=distress_window[1]:
            true_emo="Fear"   # 构造应激：模拟儿童逐步崩溃
        perc=perceive(true_emo,rng)
        obs=np.array([eng,eng-prev_eng,*perc[0],perc[1],fat,step/STEPS],dtype=np.float32)
    return rec, handback_step, (lo,hi)

def decode(a): d=act_dict(a); return f"{d['speech_rate']}/{d['stimulus']}/{d['topic']}/{d['encouragement']}"

def find_moments(rec):
    """定位关键时刻(三级)：正常协同(accept+放行)/cautious 部分保守(soft_cap)/弃权兜底(abstain→R3全保守)/硬安全拦截(R1/R2/R4)/交还人类。"""
    m={}
    for r in rec:
        t=r["perception"]["gating_tier"]; lvl=r["safety"]["level"]; rules=r["safety"]["rules"]
        if "normal" not in m and t=="accept" and not r["safety"]["intercepted"]: m["normal"]=r["step"]
        if "cautious_softcap" not in m and t=="cautious" and lvl=="soft_cap": m["cautious_softcap"]=r["step"]
        if "abstain_R3_full" not in m and t=="abstain" and "R3" in rules: m["abstain_R3_full"]=r["step"]
        if "hard_intercept" not in m and any(x in rules for x in ("R1","R2","R4")): m["hard_intercept"]=r["step"]
        if "handback" not in m and r["handback_triggered"]: m["handback"]=r["step"]
    return m

def plot_timeline(persona, rec, band, handback):
    steps=[r["step"] for r in rec]; eng=[r["child_true"]["engagement"] for r in rec]
    lo,hi=band
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(13,6),height_ratios=[2.2,1],sharex=True)
    ax.axhspan(lo,hi,alpha=0.10,color="green",label=f"target band [{lo},{hi}]")
    ax.plot(steps,eng,"o-",color="#4C72B0",lw=1.8,label="child engagement (SIM)")
    for r in rec:
        lvl=r["safety"]["level"]
        if lvl=="full_conservative": ax.axvline(r["step"],color="#C44E52",alpha=0.22,lw=6)
        elif lvl=="soft_cap": ax.axvline(r["step"],color="#DD8452",alpha=0.18,lw=6)
    if handback: ax.axvline(handback,color="purple",lw=2.0,ls="--",label=f"hand-back trigger @step{handback}")
    ax.set_ylabel("engagement"); ax.legend(loc="upper right",fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"End-to-end walkthrough — {persona} (3-level gating; SIMULATED, integration demo only)",fontweight="bold")
    # 感知三级 track + R3 level
    for r in rec:
        t=r["perception"]["gating_tier"]
        c={"accept":"#55A868","cautious":"#DD8452","abstain":"#C44E52"}[t]
        ax2.scatter(r["step"],1,color=c,s=40)
        lvl=r["safety"]["level"]
        if lvl=="soft_cap": ax2.scatter(r["step"],0.5,color="#DD8452",marker="^",s=46)
        elif lvl=="full_conservative": ax2.scatter(r["step"],0.5,color="#C44E52",marker="x",s=50)
    for lab,col in [("accept","#55A868"),("cautious","#DD8452"),("abstain","#C44E52")]: ax2.scatter([],[],color=col,label=lab)
    ax2.scatter([],[],color="#DD8452",marker="^",label="R3 soft_cap"); ax2.scatter([],[],color="#C44E52",marker="x",label="R3 full")
    ax2.set_yticks([]); ax2.set_ylim(0,1.5); ax2.set_xlabel("step"); ax2.legend(ncol=5,fontsize=7.5,loc="upper center")
    fig.tight_layout(); fig.savefig(FIGDIR/f"timeline_{persona}.png",dpi=300); fig.savefig(FIGDIR/f"timeline_{persona}.pdf"); plt.close(fig)

def main():
    print(f"device={DEVICE} | end-to-end walkthrough (SIMULATED)",flush=True)
    out={}
    runs=[("P3",None),("P1",None),("P3_distress",(8,16))]  # P3自然 / P1正常协同 / P3构造应激(演示交还人类)
    for tag,dw in runs:
        persona=tag.split("_")[0]
        rec,handback,band=run_walkthrough(persona, distress_window=dw)
        moments=find_moments(rec)
        from collections import Counter
        tier_cnt=Counter(r["perception"]["gating_tier"] for r in rec)
        level_cnt=Counter(r["safety"]["level"] for r in rec)
        old_abstain=sum(1 for r in rec if r["old_binary_gate"]["decision"]=="abstain")
        old_intc=sum(1 for r in rec if r["old_binary_gate"]["intercepted"])
        new_intc=sum(1 for r in rec if r["safety"]["intercepted"])
        out[tag]={"persona":persona,"constructed_distress_window":dw,"trajectory":rec,"key_moments":moments,
                  "handback_step":handback,"band":band,
                  "tier_counts":{k:tier_cnt.get(k,0) for k in ["accept","cautious","abstain"]},
                  "level_counts":{k:level_cnt.get(k,0) for k in ["none","soft_cap","full_conservative"]},
                  "old_binary":{"abstain":old_abstain,"intercepts":old_intc},
                  "new_three_level_intercepts":new_intc}
        plot_timeline(tag,rec,band,handback)
        print(f"  [{tag}] tiers(acc/cau/aba)={[tier_cnt.get(k,0) for k in ['accept','cautious','abstain']]} "
              f"levels(none/soft/full)={[level_cnt.get(k,0) for k in ['none','soft_cap','full_conservative']]} | "
              f"OLD: abstain={old_abstain} intc={old_intc} -> NEW intc={new_intc} | moments={moments} hb={handback}",flush=True)
    json.dump(out,open(OUTDIR/"walkthrough.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print("\nsaved: walkthrough.json, figures/timeline_*.png",flush=True)

if __name__=="__main__": main()
