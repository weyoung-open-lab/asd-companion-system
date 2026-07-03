"""
实验三：SAC策略训练（修正版）
修复：1.target leakage 2.self_stim loss 3.target-band reward 4.多seed

Usage:
  python exp3_sac.py --phase train --data_path output/offline_data.json
  python exp3_sac.py --phase train --agent sac --n_episodes 500
  python exp3_sac.py --phase plot
"""

import os,json,random,argparse
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.optim as optim
from scipy.stats import wilcoxon

OUTPUT_DIR=r"D:\project1\pythonProject\人机交互\files\exp3_sac\output"
SEED=42; PERSONAS=["P1","P2","P3"]; STEPS=30
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS=54; OBS_DIM=9
SEEDS=[42,43,44]  # 3 seeds for reproducibility

EXP2_CLASSES=["Natural","Anger","Fear","Joy"]
EXP2_IDX={c:i for i,c in enumerate(EXP2_CLASSES)}
EMO_VAL={"Joy":1.0,"Natural":0.2,"Fear":-0.8,"Anger":-0.5}

# Persona-specific target engagement bands (ASD-appropriate)
TARGET_BANDS={"P1":(0.70,0.95),"P2":(0.35,0.65),"P3":(0.08,0.35)}

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed(s)

def act_feat(idx):
    s=idx//(3*2*3);r=idx%(3*2*3);st=r//(2*3);r=r%(2*3);t=r//3;e=r%3
    return np.array([s/2.,st/2.,float(t),e/2.],dtype=np.float32)

def act_dict(idx):
    s=idx//(3*2*3);r=idx%(3*2*3);st=r//(2*3);r=r%(2*3);t=r//3;e=r%3
    return {"speech_rate":["slow","normal","fast"][s],"stimulus":["low","medium","high"][st],
            "topic":["maintain","switch"][t],"encouragement":["none","moderate","frequent"][e]}

def build_obs(eng, delta, emo_4cls, conf, fatigue, step):
    """Build 9-dim observation from state components."""
    probs=np.zeros(4)
    idx=EXP2_IDX.get(emo_4cls,0)
    probs[idx]=conf
    for i in range(4):
        if i!=idx: probs[i]=(1-conf)/3
    return np.array([eng,delta,*probs,conf,fatigue,step/STEPS],dtype=np.float32)

# ============================================================
# SURROGATE MODEL（修复：加self_stim loss）
# ============================================================
class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(OBS_DIM+4,128),nn.ReLU(),
                               nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.eng=nn.Linear(128,1)
        self.emo=nn.Linear(128,4)
        self.fat=nn.Linear(128,1)
        self.stim=nn.Linear(128,1)  # self_stim prediction (logit, no sigmoid)

    def forward(self,obs,act):
        h=self.net(torch.cat([obs,act],-1))
        return torch.sigmoid(self.eng(h)),self.emo(h),torch.sigmoid(self.fat(h)),self.stim(h)

def train_surrogate(data, persona, epochs=100):
    """修复：obs_t + action_t → target_{t+1}，避免target leakage"""
    print(f"  Training surrogate for {persona}...")
    episodes=data[persona]
    X_o,X_a,Y_e,Y_m,Y_f,Y_s=[],[],[],[],[],[]
    ie={"P1":.8,"P2":.45,"P3":.15}[persona]

    for ep in episodes:
        # 初始obs由init state构造
        prev_eng=ie
        prev_obs=build_obs(ie, 0.0, "Natural", 0.5, 0.0, 0)

        for si,t in enumerate(ep):
            # 输入：上一步的obs + 当前动作
            X_o.append(prev_obs)
            X_a.append(act_feat(t["action_idx"]))

            # 目标：当前步的输出（这才是state transition: s_t + a_t → s_{t+1}）
            out=t["output"]
            Y_e.append(float(out["engagement"]))
            Y_m.append(EXP2_IDX.get(out.get("emotion_4cls","Natural"),0))
            Y_f.append(float(out.get("fatigue",0)))
            Y_s.append(float(out.get("self_stim",False)))

            # 更新obs为当前步输出，供下一步使用
            eng=float(out["engagement"])
            delta=eng-prev_eng
            emo4=out.get("emotion_4cls","Natural")
            conf=float(out.get("confidence",0.5))
            fat=float(out.get("fatigue",0))
            prev_obs=build_obs(eng, delta, emo4, conf, fat, si+1)
            prev_eng=eng

    n=len(X_o)
    print(f"    Samples: {n}")

    ds=torch.utils.data.TensorDataset(
        torch.FloatTensor(np.array(X_o)),torch.FloatTensor(np.array(X_a)),
        torch.FloatTensor(Y_e),torch.LongTensor(Y_m),
        torch.FloatTensor(Y_f),torch.FloatTensor(Y_s))
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True)

    model=Surrogate().to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=1e-3)
    mse=nn.MSELoss(); ce=nn.CrossEntropyLoss(); bce=nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total=0
        for o,a,e,m,f,s in dl:
            o,a,e,m,f,s=[x.to(DEVICE) for x in [o,a,e,m,f,s]]
            pe,pm,pf,ps=model(o,a)
            loss=mse(pe.squeeze(),e)+.5*ce(pm,m)+.3*mse(pf.squeeze(),f)+.3*bce(ps.squeeze(),s)
            opt.zero_grad();loss.backward();opt.step()
            total+=loss.item()
        if (epoch+1)%20==0: print(f"    Epoch {epoch+1}: loss={total/len(dl):.4f}")

    # Surrogate fidelity validation
    model.eval()
    with torch.no_grad():
        all_o=torch.FloatTensor(np.array(X_o)).to(DEVICE)
        all_a=torch.FloatTensor(np.array(X_a)).to(DEVICE)
        pe,pm,pf,ps=model(all_o,all_a)
        # Engagement MAE
        eng_mae=float(torch.abs(pe.squeeze().cpu()-torch.FloatTensor(Y_e)).mean())
        # Emotion accuracy
        emo_pred=pm.cpu().argmax(dim=1).numpy()
        emo_true=np.array(Y_m)
        emo_acc=float((emo_pred==emo_true).mean())
        # Self_stim accuracy (binary)
        stim_pred=(ps.squeeze().cpu()>0).float().numpy()
        stim_true=np.array(Y_s)
        stim_acc=float((stim_pred==stim_true).mean())
        # Fatigue MAE
        fat_mae=float(torch.abs(pf.squeeze().cpu()-torch.FloatTensor(Y_f)).mean())
    model.train()
    print(f"    Surrogate Fidelity:")
    print(f"      Engagement MAE: {eng_mae:.4f}")
    print(f"      Emotion Accuracy: {emo_acc:.3f}")
    print(f"      Self_stim Accuracy: {stim_acc:.3f}")
    print(f"      Fatigue MAE: {fat_mae:.4f}")
    return model, {"engagement_mae":eng_mae,"emotion_accuracy":emo_acc,"self_stim_accuracy":stim_acc,"fatigue_mae":fat_mae}

# ============================================================
# SURROGATE ENVIRONMENT（修复：target-band reward）
# ============================================================
class SurEnv:
    def __init__(self,model,persona):
        self.m=model;self.p=persona;self.sc=0;self.obs=None;self.pe=None

    def reset(self):
        self.sc=0
        ie={"P1":.8,"P2":.45,"P3":.15}[self.p]
        self.pe=ie
        self.obs=build_obs(ie,0,"Natural",.5,0,0)
        return self.obs.copy()

    def step(self,aidx):
        self.sc+=1
        af=act_feat(aidx)
        with torch.no_grad():
            ot=torch.FloatTensor(self.obs).unsqueeze(0).to(DEVICE)
            at=torch.FloatTensor(af).unsqueeze(0).to(DEVICE)
            pe,pm,pf,ps=self.m(ot,at)
        ne=float(np.clip(pe.squeeze().cpu(),0,1))
        el=pm.squeeze().cpu().numpy()
        ep=np.exp(el)/np.exp(el).sum()
        ei=np.argmax(ep);emo=EXP2_CLASSES[ei]
        conf=float(ep[ei])
        fat=float(np.clip(pf.squeeze().cpu(),0,1))
        ss=float(ps.squeeze().cpu())>0  # logit > 0 means self_stim=True
        delta=ne-self.pe
        self.obs=build_obs(ne,delta,emo,conf,fat,self.sc)
        rew=self._reward(ne,emo,conf,ss,aidx)
        self.pe=ne
        done=self.sc>=STEPS
        return self.obs.copy(),rew,done,{"engagement":ne,"emotion":emo,"confidence":conf,"self_stim":ss}

    def _reward(self,eng,emo,conf,ss,aidx):
        a=act_dict(aidx)
        lo,hi=TARGET_BANDS[self.p]

        # 1. Engagement change (0.3)
        er=((eng-self.pe)*2) if self.pe else 0

        # 2. Target-band reward (0.3) — 核心：不是越高越好，而是在舒适区间
        if lo<=eng<=hi:
            band_r=0.3  # 在目标区间内
        elif eng<lo:
            band_r=-0.1*(lo-eng)*5  # 低于目标
        else:
            band_r=-0.2*(eng-hi)*5  # 高于目标（对P3尤其重要）

        # 3. Emotion valence (0.15)
        mr=EMO_VAL.get(emo,0)*.15

        # 4. Overstimulation penalty (0.15)
        pen=0
        if self.p=="P3":
            if a["speech_rate"]=="fast": pen-=.3
            if a["stimulus"]=="high": pen-=.4
        elif self.p=="P2":
            if a["speech_rate"]=="fast" and a["stimulus"]=="high": pen-=.2
        if ss: pen-=.3

        # 5. Confidence-aware (0.1)
        cons=a["speech_rate"]=="slow" and a["stimulus"]=="low" and a["topic"]=="maintain"
        u=1-conf
        cr=u*.2 if cons else -u*.1
        if conf>.5 and emo in ["Fear","Anger"] and cons: cr+=.15

        return float(np.clip(er+band_r+mr+pen+cr,-2,2))

# ============================================================
# RL AGENTS
# ============================================================
class RandomAgent:
    name="Random"
    def select_action(self,obs): return random.randint(0,N_ACTIONS-1)
    def update(self,*a): pass

class RuleAgent:
    name="Rule-Based"
    def select_action(self,obs):
        eng,ep,conf,fat=obs[0],obs[2:6],obs[6],obs[7]
        ei=np.argmax(ep);emo=EXP2_CLASSES[ei]
        if conf<.4: return self._e("slow","low","maintain","moderate")
        if eng<.3: return self._e("slow","low","maintain","moderate")
        if emo in ["Fear","Anger"]: return self._e("slow","low","maintain","none")
        if eng>.6 and emo=="Joy":
            return self._e("normal","medium","switch" if random.random()<.3 else "maintain","moderate")
        if fat>.5: return self._e("slow","low","maintain","none")
        return self._e("normal","low","maintain","moderate")
    def _e(self,s,st,t,e):
        return {"slow":0,"normal":1,"fast":2}[s]*(3*2*3)+{"low":0,"medium":1,"high":2}[st]*(2*3)+{"maintain":0,"switch":1}[t]*3+{"none":0,"moderate":1,"frequent":2}[e]
    def update(self,*a): pass

class DQNAgent:
    name="DQN"
    def __init__(self):
        self.q=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS)).to(DEVICE)
        self.qt=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS)).to(DEVICE)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt=optim.Adam(self.q.parameters(),lr=1e-3)
        self.buf=[];self.eps=1.0;self.steps=0
    def select_action(self,obs):
        if random.random()<self.eps: return random.randint(0,N_ACTIONS-1)
        with torch.no_grad(): return self.q(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    def update(self,o,a,r,n,d):
        self.buf.append((o,a,r,n,d))
        if len(self.buf)>10000: self.buf.pop(0)
        if len(self.buf)<64: return
        b=random.sample(self.buf,64)
        ob=torch.FloatTensor([x[0] for x in b]).to(DEVICE)
        ab=torch.LongTensor([x[1] for x in b]).to(DEVICE)
        rb=torch.FloatTensor([x[2] for x in b]).to(DEVICE)
        nb=torch.FloatTensor([x[3] for x in b]).to(DEVICE)
        db=torch.FloatTensor([x[4] for x in b]).to(DEVICE)
        qv=self.q(ob).gather(1,ab.unsqueeze(1)).squeeze()
        with torch.no_grad(): tgt=rb+(1-db)*.99*self.qt(nb).max(1)[0]
        self.opt.zero_grad();nn.MSELoss()(qv,tgt).backward();self.opt.step()
        self.eps=max(.05,self.eps*.995);self.steps+=1
        if self.steps%100==0: self.qt.load_state_dict(self.q.state_dict())

class PPOAgent:
    name="PPO"
    def __init__(self):
        self.actor=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS),nn.Softmax(dim=-1)).to(DEVICE)
        self.critic=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1)).to(DEVICE)
        self.opt=optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()),lr=3e-4)
        self.traj=[]
    def select_action(self,obs):
        with torch.no_grad():
            p=self.actor(torch.FloatTensor(obs).to(DEVICE))
            return torch.distributions.Categorical(p).sample().item()
    def update(self,o,a,r,n,d):
        self.traj.append((o,a,r))
        if d: self._train();self.traj=[]
    def _train(self):
        if not self.traj: return
        o=torch.FloatTensor([t[0] for t in self.traj]).to(DEVICE)
        a=torch.LongTensor([t[1] for t in self.traj]).to(DEVICE)
        R=0;rets=[]
        for r in reversed([t[2] for t in self.traj]): R=r+.99*R;rets.insert(0,R)
        rets=torch.FloatTensor(rets).to(DEVICE)
        rets=(rets-rets.mean())/(rets.std()+1e-8)
        with torch.no_grad(): old=torch.log(self.actor(o).gather(1,a.unsqueeze(1))+1e-8).squeeze()
        for _ in range(4):
            p=self.actor(o);lp=torch.log(p.gather(1,a.unsqueeze(1))+1e-8).squeeze()
            v=self.critic(o).squeeze();ratio=torch.exp(lp-old);adv=rets-v.detach()
            loss=-torch.min(ratio*adv,torch.clamp(ratio,.8,1.2)*adv).mean()+.5*(rets-v).pow(2).mean()
            self.opt.zero_grad();loss.backward();self.opt.step()

class SACAgent:
    name="SAC"
    def __init__(self):
        def mq(): return nn.Sequential(nn.Linear(OBS_DIM,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,N_ACTIONS)).to(DEVICE)
        self.q1,self.q2,self.q1t,self.q2t=mq(),mq(),mq(),mq()
        self.q1t.load_state_dict(self.q1.state_dict());self.q2t.load_state_dict(self.q2.state_dict())
        self.pi=nn.Sequential(nn.Linear(OBS_DIM,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,N_ACTIONS),nn.Softmax(dim=-1)).to(DEVICE)
        self.qopt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=3e-4)
        self.popt=optim.Adam(self.pi.parameters(),lr=3e-4)
        self.te=-np.log(1./N_ACTIONS)*.98
        self.la=torch.zeros(1,requires_grad=True,device=DEVICE)
        self.aopt=optim.Adam([self.la],lr=3e-4)
        self.buf=[];self.tau=.005
    def select_action(self,obs):
        with torch.no_grad():
            p=self.pi(torch.FloatTensor(obs).to(DEVICE))
            return torch.distributions.Categorical(p).sample().item()
    def update(self,o,a,r,n,d):
        self.buf.append((o,a,r,n,d))
        if len(self.buf)>50000: self.buf.pop(0)
        if len(self.buf)<128: return
        b=random.sample(self.buf,64)
        ob=torch.FloatTensor([x[0] for x in b]).to(DEVICE)
        ab=torch.LongTensor([x[1] for x in b]).to(DEVICE)
        rb=torch.FloatTensor([x[2] for x in b]).to(DEVICE)
        nb=torch.FloatTensor([x[3] for x in b]).to(DEVICE)
        db=torch.FloatTensor([x[4] for x in b]).to(DEVICE)
        alpha=self.la.exp().item()
        with torch.no_grad():
            np_=self.pi(nb);nlp=torch.log(np_+1e-8)
            nq=torch.min(self.q1t(nb),self.q2t(nb))
            tgt=rb+(1-db)*.99*(np_*(nq-alpha*nlp)).sum(1)
        q1v=self.q1(ob).gather(1,ab.unsqueeze(1)).squeeze()
        q2v=self.q2(ob).gather(1,ab.unsqueeze(1)).squeeze()
        ql=((q1v-tgt).pow(2)+(q2v-tgt).pow(2)).mean()
        self.qopt.zero_grad();ql.backward();self.qopt.step()
        p=self.pi(ob);lp=torch.log(p+1e-8)
        mq=torch.min(self.q1(ob),self.q2(ob))
        pl=(p*(alpha*lp-mq)).sum(1).mean()
        self.popt.zero_grad();pl.backward();self.popt.step()
        ent=-(p*lp).sum(1).mean()
        al=-(self.la*(ent.detach()-self.te))
        self.aopt.zero_grad();al.backward();self.aopt.step()
        for p1,t1 in zip(self.q1.parameters(),self.q1t.parameters()): t1.data.copy_(self.tau*p1.data+(1-self.tau)*t1.data)
        for p2,t2 in zip(self.q2.parameters(),self.q2t.parameters()): t2.data.copy_(self.tau*p2.data+(1-self.tau)*t2.data)

# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_agent(agent,env,n_ep):
    rets,engs=[],[]
    for ep in range(n_ep):
        obs=env.reset();er=0;ee=[]
        for _ in range(STEPS):
            a=agent.select_action(obs)
            nobs,r,done,info=env.step(a)
            agent.update(obs,a,r,nobs,float(done))
            er+=r;ee.append(info["engagement"]);obs=nobs
        rets.append(er);engs.append(np.mean(ee))
        if (ep+1)%100==0: print(f"    Ep {ep+1}: avg_ret={np.mean(rets[-50:]):.2f}")
    return rets,engs

def evaluate(agent,env,n=100):
    rets,engs,ov,ts,emos,band_hits=[],[], 0,0,[],0
    lo,hi=TARGET_BANDS[env.p]
    for _ in range(n):
        obs=env.reset();er=0;ee=[]
        for __ in range(STEPS):
            a=agent.select_action(obs);nobs,r,done,info=env.step(a)
            er+=r;ee.append(info["engagement"]);emos.append(info["emotion"])
            ad=act_dict(a)
            if ad["speech_rate"]=="fast" and ad["stimulus"]=="high": ov+=1
            if lo<=info["engagement"]<=hi: band_hits+=1
            ts+=1;obs=nobs
        rets.append(er);engs.append(np.mean(ee))
    ed={e:emos.count(e)/len(emos) for e in EXP2_CLASSES}
    neg_rate=(emos.count("Fear")+emos.count("Anger"))/len(emos)
    return {"mean_return":float(np.mean(rets)),"std_return":float(np.std(rets)),
            "mean_engagement":float(np.mean(engs)),"std_engagement":float(np.std(engs)),
            "overstim_rate":ov/ts,"target_band_rate":band_hits/ts,
            "negative_emotion_rate":neg_rate,"emotion_dist":ed}

# ============================================================
# MAIN
# ============================================================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=["train","plot"])
    parser.add_argument("--data_path",default=None)
    parser.add_argument("--n_episodes",type=int,default=500)
    parser.add_argument("--surrogate_epochs",type=int,default=100)
    parser.add_argument("--agent",default="all",choices=["all","random","rule","dqn","ppo","sac"])
    args=parser.parse_args()
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    if args.phase=="train":
        dp=args.data_path or os.path.join(OUTPUT_DIR,"offline_data.json")
        print(f"  Loading: {dp}")
        with open(dp,encoding="utf-8") as f: data=json.load(f)

        # Auto-detect available personas from data
        available_personas = [p for p in PERSONAS if p in data]
        print(f"  Available personas in data: {available_personas}")

        # Train surrogates
        print(f"\n  Phase 1: Training surrogates")
        surrogates={}
        surrogate_fidelity={}
        for p in available_personas:
            set_seed(SEED)
            model, fidelity = train_surrogate(data,p,args.surrogate_epochs)
            surrogates[p] = model
            surrogate_fidelity[p] = fidelity
            torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,f"surrogate_{p}.pt"))

        # Print surrogate fidelity summary table
        print(f"\n  Surrogate Fidelity Summary:")
        print(f"  {'Persona':<8} {'Eng MAE':>10} {'Emo Acc':>10} {'Stim Acc':>10} {'Fat MAE':>10}")
        for p in available_personas:
            f = surrogate_fidelity[p]
            print(f"  {p:<8} {f['engagement_mae']:>10.4f} {f['emotion_accuracy']:>10.3f} {f['self_stim_accuracy']:>10.3f} {f['fatigue_mae']:>10.4f}")

        # Save fidelity
        with open(os.path.join(OUTPUT_DIR,"surrogate_fidelity.json"),"w") as ff:
            json.dump(surrogate_fidelity,ff,indent=2)

        # Train agents with multiple seeds
        amap={"random":RandomAgent,"rule":RuleAgent,"dqn":DQNAgent,"ppo":PPOAgent,"sac":SACAgent}
        if args.agent!="all": amap={args.agent:amap[args.agent]}

        all_results={}
        for an,AC in amap.items():
            print(f"\n{'='*60}\n  {an.upper()}\n{'='*60}")
            agent_res={}
            for p in available_personas:
                print(f"\n  --- {p} ---")
                seed_rets,seed_engs,seed_evals=[],[],[]
                for seed in SEEDS:
                    set_seed(seed)
                    agent=AC();env=SurEnv(surrogates[p],p)
                    if an in ["random","rule"]:
                        rets,engs=[],[]
                        for _ in range(args.n_episodes):
                            obs=env.reset();er=0;ee=[]
                            for __ in range(STEPS):
                                a=agent.select_action(obs);obs,r,d,info=env.step(a)
                                er+=r;ee.append(info["engagement"])
                            rets.append(er);engs.append(np.mean(ee))
                    else:
                        rets,engs=train_agent(agent,env,args.n_episodes)
                    ev=evaluate(agent,env)
                    seed_rets.append(rets);seed_engs.append(engs);seed_evals.append(ev)

                # Average across seeds
                avg_eval={k:np.mean([e[k] for e in seed_evals]) for k in seed_evals[0] if isinstance(seed_evals[0][k],(int,float))}
                avg_eval["emotion_dist"]=seed_evals[0]["emotion_dist"]
                print(f"    Ret={avg_eval['mean_return']:.2f} Eng={avg_eval['mean_engagement']:.3f} Band={avg_eval['target_band_rate']:.3f} Overstim={avg_eval['overstim_rate']:.3f}")

                agent_res[p]={"train_returns":[float(r) for r in seed_rets[0]],
                              "train_engagements":[float(e) for e in seed_engs[0]],
                              "eval":avg_eval,"seed_evals":[{k:v for k,v in e.items() if isinstance(v,(int,float))} for e in seed_evals]}
            all_results[an]=agent_res

        # Wilcoxon test: SAC vs each baseline
        if "sac" in all_results:
            print(f"\n  Wilcoxon significance tests (SAC vs others):")
            for an in all_results:
                if an=="sac": continue
                for p in available_personas:
                    sac_rets=all_results["sac"][p]["train_returns"][-100:]
                    other_rets=all_results[an][p]["train_returns"][-100:]
                    if len(sac_rets)==len(other_rets):
                        try:
                            stat,pval=wilcoxon(sac_rets,other_rets)
                            sig="***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else "ns"
                            print(f"    SAC vs {an} ({p}): p={pval:.4f} {sig}")
                        except: print(f"    SAC vs {an} ({p}): test failed")

        with open(os.path.join(OUTPUT_DIR,"training_results.json"),"w") as f:
            json.dump(all_results,f,indent=2,default=lambda x:float(x) if hasattr(x,'item') else str(x))

        # Compute additional metrics
        print(f"\n{'='*60}\n  FULL RESULTS TABLE\n{'='*60}")
        print(f"  {'Agent':<10} {'Persona':<6} {'Return':>8} {'Eng':>7} {'Band%':>7} {'Ovstim%':>8} {'NegEmo%':>8} {'Conv':>6}")
        print(f"  {'-'*63}")
        for an,ar in all_results.items():
            for p in available_personas:
                ev=ar[p]['eval']
                # Convergence speed: first episode where rolling avg reaches 90% of final return
                rets=ar[p]['train_returns']
                final_ret=np.mean(rets[-50:]) if len(rets)>=50 else np.mean(rets)
                threshold=final_ret*0.9
                conv_ep=len(rets)  # default: never converged
                if len(rets)>=20:
                    rolling=[np.mean(rets[max(0,i-20):i+1]) for i in range(len(rets))]
                    for i,v in enumerate(rolling):
                        if v>=threshold:
                            conv_ep=i+1; break
                neg=ev.get('negative_emotion_rate',0)
                print(f"  {an:<10} {p:<6} {ev['mean_return']:>8.2f} {ev['mean_engagement']:>7.3f} {ev['target_band_rate']*100:>6.1f}% {ev['overstim_rate']*100:>7.1f}% {neg*100:>7.1f}% {conv_ep:>5}ep")

        # Cross-persona variance (lower = better generalization)
        print(f"\n  Cross-persona Return Variance (lower = better generalization):")
        for an,ar in all_results.items():
            rets_per_p=[ar[p]['eval']['mean_return'] for p in available_personas]
            var=np.var(rets_per_p)
            print(f"    {an:<12}: variance={var:.2f}, std={np.std(rets_per_p):.2f}")

        # Summary compact
        print(f"\n{'='*60}\n  COMPACT SUMMARY\n{'='*60}")
        print(f"  {'Agent':<12}"+' '.join(f'{p+" Ret":>10}' for p in available_personas)+' '.join(f'{p+" Band":>10}' for p in available_personas))
        for an,ar in all_results.items():
            row=f"  {an:<12}"
            for p in available_personas: row+=f"{ar[p]['eval']['mean_return']:>10.2f}"
            for p in available_personas: row+=f"{ar[p]['eval']['target_band_rate']:>10.3f}"
            print(row)

    elif args.phase=="plot":
        dp=args.data_path or os.path.join(OUTPUT_DIR,"training_results.json")
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        with open(dp) as f: results=json.load(f)
        agents=[a for a in ["random","rule","dqn","ppo","sac"] if a in results]
        available_personas=[p for p in PERSONAS if p in results[agents[0]]]

        # Fig1: Learning curves
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for i,p in enumerate(available_personas):
            for an in agents:
                r=results[an][p]["train_returns"]
                w=max(1,len(r)//20)
                axes[i].plot(np.convolve(r,np.ones(w)/w,'valid'),label=an.upper(),linewidth=2)
            axes[i].set_title(p);axes[i].set_xlabel("Episode");axes[i].set_ylabel("Return")
            axes[i].legend();axes[i].grid(True,alpha=.3)
        plt.suptitle("Learning Curves",fontsize=16);plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,"learning_curves.png"),dpi=150)
        print("  Saved: learning_curves.png")

        # Fig2: Performance bars (Return + Band Rate + Overstim)
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        x=np.arange(len(agents));w=.25
        metrics=[("mean_return","Mean Return"),("target_band_rate","Target Band Rate"),("overstim_rate","Overstim Rate")]
        for j,(mk,title) in enumerate(metrics):
            for i,p in enumerate(available_personas):
                axes[j].bar(x+i*w,[results[a][p]["eval"][mk] for a in agents],w,label=p)
            axes[j].set_xticks(x+w);axes[j].set_xticklabels([a.upper() for a in agents])
            axes[j].set_title(title);axes[j].legend();axes[j].grid(True,alpha=.3,axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,"performance_comparison.png"),dpi=150)
        print("  Saved: performance_comparison.png")

        # Fig3: SAC action distribution heatmap
        if "sac" in results:
            print("  Action heatmap requires re-running evaluation (skipped in plot phase)")

        print(f"  All plots saved to {OUTPUT_DIR}")

if __name__=="__main__": main()