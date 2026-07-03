# -*- coding: utf-8 -*-
"""
实验3 第二步：训练 per-Persona Surrogate 世界模型（本地 GPU，不烧 token）
=====================================================================
数据：artifacts/exp3/offline_data_v2.jsonl（4o V2 模拟器生成的 4500 条 (s,a,s',r)）。
任务：给定 (s_t, a_t) -> 预测 s_{t+1}，作为 LLM 模拟器的快速廉价替身，供 SAC 训练。

engagement 两种模式（--eng_mode）：
  reg : sigmoid 回归 + MSE（基线；P2 回归向均值、压窄方差）
  cls : K-bin 分类(softmax) + 采样（方案1：恢复两端状态、破回归向均值）

其余头：emotion(4 softmax) / confidence / fatigue(sigmoid 回归) / self_stim(logit 辅助)。
Δeng、τ 规则推导不学；reward 由 exp3_sac._reward 确定性计算，surrogate 不学 reward。
划分按 episode 80/20 无泄漏；rollout 用 build_obs 重构留在流形上。

用法：
  python exp3_surrogate.py --eng_mode reg            # 基线
  python exp3_surrogate.py --eng_mode cls --eng_bins 20 --out_suffix _cls
"""
import os, sys, json, random, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/"components/exp3_policy"))
from exp3_sac import act_feat, build_obs, EXP2_CLASSES, EXP2_IDX, STEPS

DATA = ROOT/"artifacts/exp3/offline_data_v2.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERSONAS = ["P1","P2","P3"]; SEED = 42; LN2 = np.log(2)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def eng_to_state(e): return 0 if e<=0.33 else (1 if e<=0.66 else 2)
def jsd_log2(p,q):
    p=np.clip(p,1e-12,1); q=np.clip(q,1e-12,1); p=p/p.sum(); q=q/q.sum(); m=0.5*(p+q)
    return float((0.5*np.sum(rel_entr(p,m))+0.5*np.sum(rel_entr(q,m)))/LN2)
def cosine(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float)
    return float(p@q/(np.linalg.norm(p)*np.linalg.norm(q)+1e-12))

# ---------------- 数据 ----------------
def load_grouped():
    by={}
    for ln in open(DATA, encoding="utf-8"):
        if not ln.strip(): continue
        t=json.loads(ln); by.setdefault(t["persona"],{}).setdefault(t["episode_idx"],[]).append(t)
    for p in by:
        for ep in by[p]: by[p][ep].sort(key=lambda x:x["step"])
    return by

def build_xy(transitions):
    X,Yeng,Yprobs,Yconf,Yfat,Ystim=[],[],[],[],[],[]
    for t in transitions:
        s=np.asarray(t["state"],dtype=np.float32); ns=np.asarray(t["next_state"],dtype=np.float32)
        X.append(np.concatenate([s, act_feat(t["action"])]))
        Yeng.append(ns[0]); Yprobs.append(ns[2:6]); Yconf.append(ns[6]); Yfat.append(ns[7])
        Ystim.append(1.0 if t["_aux"].get("self_stim",False) else 0.0)
    return (np.array(X,dtype=np.float32),np.array(Yeng,dtype=np.float32),np.array(Yprobs,dtype=np.float32),
            np.array(Yconf,dtype=np.float32),np.array(Yfat,dtype=np.float32),np.array(Ystim,dtype=np.float32))

# ---------------- 模型 ----------------
class Surrogate(nn.Module):
    def __init__(self, eng_mode="reg", eng_bins=20, in_dim=13, h=256, p_drop=0.1):
        super().__init__()
        self.eng_mode=eng_mode; self.K=eng_bins
        self.body=nn.Sequential(nn.Linear(in_dim,h),nn.ReLU(),nn.Dropout(p_drop),
                                nn.Linear(h,h),nn.ReLU(),nn.Dropout(p_drop))
        self.eng=nn.Linear(h, eng_bins if eng_mode=="cls" else 1)
        self.emo=nn.Linear(h,4); self.conf=nn.Linear(h,1); self.fat=nn.Linear(h,1); self.stim=nn.Linear(h,1)
    def forward(self,x):
        hh=self.body(x)
        eng_raw=self.eng(hh)   # cls: (...,K) logits ; reg: (...,1)
        return (eng_raw, torch.softmax(self.emo(hh),dim=-1),
                torch.sigmoid(self.conf(hh)).squeeze(-1), torch.sigmoid(self.fat(hh)).squeeze(-1),
                self.stim(hh).squeeze(-1))

def bin_centers(K): return (np.arange(K)+0.5)/K
def eng_to_bin(e,K): return int(min(int(e*K),K-1))

def expected_eng(eng_raw, mode, K):
    if mode=="reg": return torch.sigmoid(eng_raw).squeeze(-1)
    probs=torch.softmax(eng_raw,dim=-1)
    centers=torch.as_tensor(bin_centers(K),dtype=torch.float32,device=eng_raw.device)
    return (probs*centers).sum(-1)

def sample_eng(eng_raw, mode, K, rng):
    """返回采样 engagement (numpy 1d)。"""
    if mode=="reg": return torch.sigmoid(eng_raw).squeeze(-1).cpu().numpy()
    probs=torch.softmax(eng_raw,dim=-1).cpu().numpy(); centers=bin_centers(K); w=1.0/K; out=[]
    for pr in probs:
        k=rng.choice(K,p=pr/pr.sum()); out.append(np.clip(centers[k]+rng.uniform(-w/2,w/2),0,1))
    return np.array(out,dtype=np.float32)

def loss_fn(pred, tgt, mode, K):
    eng_raw,pm,pc,pf,ps=pred; ye,ym,yc,yf,ys=tgt
    mse=nn.functional.mse_loss; bce=nn.functional.binary_cross_entropy_with_logits
    if mode=="reg":
        eloss=mse(torch.sigmoid(eng_raw).squeeze(-1),ye)
    else:
        tb=torch.clamp((ye*K).long(),0,K-1)
        eloss=nn.functional.cross_entropy(eng_raw,tb)*0.15   # 缩放使与 MSE 头量级可比
    return eloss + 0.5*mse(pm,ym) + 0.5*mse(pc,yc) + 0.3*mse(pf,yf) + 0.3*bce(ps,ys)

def to_dev(arrs): return [torch.as_tensor(a,device=DEVICE) for a in arrs]

# ---------------- 训练 ----------------
def split_ids(episodes):
    ids=sorted(episodes.keys()); rng=np.random.RandomState(SEED); rng.shuffle(ids)
    nv=max(1,int(round(len(ids)*0.2)))
    return [e for e in ids if e not in set(ids[:nv])], list(ids[:nv])

def train_persona(persona, episodes, mode, K, epochs=300, patience=30, lr=1e-3, wd=1e-4):
    set_seed(SEED)
    tr_ids,val_ids=split_ids(episodes)
    tr=[t for e in tr_ids for t in episodes[e]]; va=[t for e in val_ids for t in episodes[e]]
    Xtr=build_xy(tr); Xva=build_xy(va)
    print(f"  [{persona}] ep tr/val={len(tr_ids)}/{len(val_ids)} | trans {len(tr)}/{len(va)} | mode={mode}")
    Xt=to_dev(Xtr); Xv=to_dev(Xva)
    model=Surrogate(eng_mode=mode,eng_bins=K).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*Xt),batch_size=256,shuffle=True)
    hist={"train":[],"val":[]}; best=1e9; best_state=None; bad=0
    for ep in range(epochs):
        model.train(); tl=0
        for b in dl:
            x=b[0]; tgt=b[1:]; opt.zero_grad(); l=loss_fn(model(x),tgt,mode,K); l.backward(); opt.step(); tl+=l.item()*len(x)
        tl/=len(Xt[0]); model.eval()
        with torch.no_grad(): vl=loss_fn(model(Xv[0]),Xv[1:],mode,K).item()
        hist["train"].append(tl); hist["val"].append(vl)
        if vl<best-1e-5: best=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: print(f"  [{persona}] early stop @ep{ep+1} best_val={best:.5f}"); break
    model.load_state_dict(best_state); model.eval()
    return model, hist, (Xva, tr_ids, val_ids, episodes)

# ---------------- 评估 ----------------
def eval_singlestep(model, Xva, mode, K):
    X,Yeng,Yprobs,Yconf,Yfat,Ystim=Xva
    with torch.no_grad():
        eng_raw,pm,pc,pf,ps=model(torch.as_tensor(X,device=DEVICE))
    pe=expected_eng(eng_raw,mode,K).cpu().numpy()   # 点估计(期望)
    pm=pm.cpu().numpy(); pc=pc.cpu().numpy(); pf=pf.cpu().numpy(); ps=(ps.cpu().numpy()>0).astype(float)
    mae=lambda a,b: float(np.mean(np.abs(a-b))); rmse=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    return {"engagement_mae":mae(pe,Yeng),"engagement_rmse":rmse(pe,Yeng),
            "confidence_mae":mae(pc,Yconf),"fatigue_mae":mae(pf,Yfat),
            "emotion_dist_mse":float(np.mean((pm-Yprobs)**2)),
            "emotion_argmax_acc":float((pm.argmax(1)==Yprobs.argmax(1)).mean()),
            "self_stim_acc":float((ps==Ystim).mean()),"n_val":int(len(Yeng))}

def eval_distribution(model, Xva, mode, K, seed=123):
    """engagement 三态分布用【采样】(还原两端)；emotion 用 argmax。"""
    X,Yeng,Yprobs,Yconf,Yfat,Ystim=Xva
    with torch.no_grad():
        eng_raw,pm,_,_,_=model(torch.as_tensor(X,device=DEVICE))
    samp=sample_eng(eng_raw,mode,K,np.random.RandomState(seed)); pm=pm.cpu().numpy()
    real=np.bincount([eng_to_state(e) for e in Yeng],minlength=3)/len(Yeng)
    sim =np.bincount([eng_to_state(e) for e in samp],minlength=3)/len(samp)
    rem=np.bincount(Yprobs.argmax(1),minlength=4)/len(Yprobs); sem=np.bincount(pm.argmax(1),minlength=4)/len(pm)
    return {"engagement_state_JSD_log2":round(jsd_log2(real,sim),4),"engagement_state_cosine":round(cosine(real,sim),4),
            "emotion_JSD_log2":round(jsd_log2(rem,sem),4),"emotion_cosine":round(cosine(rem,sem),4),
            "real_eng_dist":[round(x,3) for x in real],"sim_eng_dist":[round(x,3) for x in sim]}

def rollout(model, mode, K, val_ids, episodes, n_sample=3, seed=7):
    """期望-rollout 测 MAE(对照真实轨迹) + 采样-rollout 测稳定性(发散/塌缩)。state 用 build_obs 重构。"""
    rng=np.random.RandomState(seed)
    roll_errs=[]; pred_traj=[]; real_traj=[]; diverged=0; total_roll=0
    for e in val_ids:
        ep=episodes[e]; s0=np.asarray(ep[0]["state"],dtype=np.float32)
        # ---- 期望 rollout (确定性, 测 MAE) ----
        obs=s0.copy(); prev=float(s0[0]); pr=[]
        for i,t in enumerate(ep):
            with torch.no_grad():
                x=torch.as_tensor(np.concatenate([obs,act_feat(t["action"])]),dtype=torch.float32,device=DEVICE).unsqueeze(0)
                eng_raw,pm,pc,pf,ps=model(x)
            ne=float(np.clip(expected_eng(eng_raw,mode,K).item(),0,1)); conf=float(np.clip(pc.item(),0,1)); fat=float(np.clip(pf.item(),0,1))
            cls=EXP2_CLASSES[int(pm.squeeze().argmax().item())]
            obs=build_obs(ne,ne-prev,cls,conf,fat,i+1); prev=ne; pr.append(ne)
        pr=np.array(pr); rl=np.array([float(t["next_state"][0]) for t in ep])
        roll_errs.append(float(np.mean(np.abs(pr-rl)))); pred_traj.append(pr); real_traj.append(rl)
        # ---- 采样 rollout (随机, 测稳定性) ----
        for _ in range(n_sample):
            obs=s0.copy(); prev=float(s0[0]); tr=[]
            for i,t in enumerate(ep):
                with torch.no_grad():
                    x=torch.as_tensor(np.concatenate([obs,act_feat(t["action"])]),dtype=torch.float32,device=DEVICE).unsqueeze(0)
                    eng_raw,pm,pc,pf,ps=model(x)
                ne=float(sample_eng(eng_raw,mode,K,rng)[0]); conf=float(np.clip(pc.item(),0,1)); fat=float(np.clip(pf.item(),0,1))
                cls=EXP2_CLASSES[int(pm.squeeze().argmax().item())]
                obs=build_obs(ne,ne-prev,cls,conf,fat,i+1); prev=ne; tr.append(ne)
            tr=np.array(tr); total_roll+=1
            if tr.max()>1.01 or tr.min()<-0.01 or np.std(tr)<1e-4: diverged+=1
    return {"rollout_eng_mae_mean":round(float(np.mean(roll_errs)),4),"rollout_eng_mae_std":round(float(np.std(roll_errs)),4),
            "n_rollout_episodes":len(val_ids),"sampled_rollouts":total_roll,"diverged_or_collapsed":diverged,
            "_pred":pred_traj,"_real":real_traj}

# ---------------- 画图 ----------------
def plot_loss(hist_all, tag, FIGDIR):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,p in zip(axes,PERSONAS):
        h=hist_all[p]; ax.plot(h["train"],label="train",lw=1.6); ax.plot(h["val"],label="val",lw=1.6)
        ax.set_title(f"{p} (best val={min(h['val']):.4f})"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"Surrogate loss ({tag})",fontweight="bold"); fig.tight_layout()
    fig.savefig(FIGDIR/f"surrogate_loss_curves.png",dpi=300); fig.savefig(FIGDIR/f"surrogate_loss_curves.pdf"); plt.close(fig)

def plot_rollout(roll_all, tag, FIGDIR):
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,p in zip(axes,PERSONAS):
        r=roll_all[p]; steps=np.arange(1,STEPS+1)
        for pr in r["_pred"][:8]: ax.plot(steps,pr,color="#C44E52",alpha=0.45,lw=1.0)
        for re in r["_real"][:8]: ax.plot(steps,re,color="#4C72B0",alpha=0.45,lw=1.0)
        ax.plot([],[],color="#C44E52",label="surrogate"); ax.plot([],[],color="#4C72B0",label="real")
        ax.set_title(f"{p} (MAE={r['rollout_eng_mae_mean']}, diverged={r['diverged_or_collapsed']}/{r['sampled_rollouts']})")
        ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("engagement")
    fig.suptitle(f"Multi-step rollout ({tag}): expected vs real; stability from sampled rollouts",fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGDIR/f"surrogate_rollout.png",dpi=300); fig.savefig(FIGDIR/f"surrogate_rollout.pdf"); plt.close(fig)

# ---------------- 主 ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eng_mode",default="reg",choices=["reg","cls"])
    ap.add_argument("--eng_bins",type=int,default=20)
    ap.add_argument("--out_suffix",default="")
    args=ap.parse_args()
    OUTDIR=ROOT/f"artifacts/exp3/surrogate{args.out_suffix}"; OUTDIR.mkdir(parents=True,exist_ok=True)
    FIGDIR=OUTDIR/"figures"; FIGDIR.mkdir(parents=True,exist_ok=True)
    print(f"device={DEVICE} | mode={args.eng_mode} bins={args.eng_bins} | out={OUTDIR.name}")
    by=load_grouped(); report={}; hist_all={}; roll_all={}
    for p in PERSONAS:
        model,hist,(Xva,tr_ids,val_ids,episodes)=train_persona(p,by[p],args.eng_mode,args.eng_bins)
        ss=eval_singlestep(model,Xva,args.eng_mode,args.eng_bins)
        dd=eval_distribution(model,Xva,args.eng_mode,args.eng_bins)
        rr=rollout(model,args.eng_mode,args.eng_bins,val_ids,episodes)
        torch.save(model.state_dict(), OUTDIR/f"surrogate_{p}.pt")
        hist_all[p]=hist; roll_all[p]=rr
        report[p]={"single_step":ss,"distribution":dd,"rollout":{k:v for k,v in rr.items() if not k.startswith("_")},
                   "config":{"eng_mode":args.eng_mode,"eng_bins":args.eng_bins},
                   "train_episodes":len(tr_ids),"val_episodes":len(val_ids)}
        print(f"  [{p}] engMAE={ss['engagement_mae']:.4f} emoAcc={ss['emotion_argmax_acc']:.3f} engJSD={dd['engagement_state_JSD_log2']} "
              f"sim_dist={dd['sim_eng_dist']} rollMAE={rr['rollout_eng_mae_mean']} diverged={rr['diverged_or_collapsed']}/{rr['sampled_rollouts']}")
    plot_loss(hist_all,args.eng_mode,FIGDIR); plot_rollout(roll_all,args.eng_mode,FIGDIR)
    json.dump(report,open(OUTDIR/"surrogate_eval.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print("\n==== SUMMARY ("+args.eng_mode+") ====")
    print(f"{'P':<4}{'engMAE':>9}{'emoAcc':>8}{'engJSD':>9}{'real_dist':>20}{'sim_dist':>20}{'rollMAE':>9}{'div':>6}")
    for p in PERSONAS:
        s=report[p]; ss=s['single_step']; dd=s['distribution']; rr=s['rollout']
        print(f"{p:<4}{ss['engagement_mae']:>9.4f}{ss['emotion_argmax_acc']:>8.3f}{dd['engagement_state_JSD_log2']:>9.4f}"
              f"{str(dd['real_eng_dist']):>20}{str(dd['sim_eng_dist']):>20}{rr['rollout_eng_mae_mean']:>9.4f}{rr['diverged_or_collapsed']:>6}")

if __name__=="__main__": main()
