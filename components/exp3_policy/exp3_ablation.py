"""
实验三消融实验
==============
在已训练的surrogate环境上，对比SAC的不同配置：
  SA-Full: 完整版本（9维obs + 完整reward）
  SA-1: w/o emotion（去掉4维情绪概率，obs=5维）
  SA-2: w/o confidence（去掉confidence，obs=8维）
  SA-3: w/o safety penalty（reward去掉过激惩罚）

Usage:
  python exp3_ablation.py --n_episodes 500
"""

import os, json, random, argparse
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.optim as optim

OUTPUT_DIR = r"D:\project1\pythonProject\人机交互\files\exp3_sac\output"
PERSONAS = ["P1","P2","P3"]
STEPS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 54
OBS_DIM = 9
SEEDS = [42, 43, 44]

EXP2_CLASSES = ["Natural","Anger","Fear","Joy"]
EXP2_IDX = {c:i for i,c in enumerate(EXP2_CLASSES)}
EMO_VAL = {"Joy":1.0,"Natural":0.2,"Fear":-0.8,"Anger":-0.5}
TARGET_BANDS = {"P1":(0.70,0.95),"P2":(0.35,0.65),"P3":(0.08,0.35)}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed(s)

def act_feat(idx):
    s=idx//(3*2*3);r=idx%(3*2*3);st=r//(2*3);r=r%(2*3);t=r//3;e=r%3
    return np.array([s/2.,st/2.,float(t),e/2.],dtype=np.float32)

def act_dict(idx):
    s=idx//(3*2*3);r=idx%(3*2*3);st=r//(2*3);r=r%(2*3);t=r//3;e=r%3
    return {"speech_rate":["slow","normal","fast"][s],"stimulus":["low","medium","high"][st],
            "topic":["maintain","switch"][t],"encouragement":["none","moderate","frequent"][e]}

def build_obs(eng, delta, emo_4cls, conf, fatigue, step):
    probs = np.zeros(4)
    idx = EXP2_IDX.get(emo_4cls, 0)
    probs[idx] = conf
    for i in range(4):
        if i != idx: probs[i] = (1-conf)/3
    return np.array([eng, delta, *probs, conf, fatigue, step/STEPS], dtype=np.float32)

# ============================================================
# SURROGATE MODEL (same architecture, load weights)
# ============================================================
class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM+4,128),nn.ReLU(),
                                 nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.eng = nn.Linear(128,1)
        self.emo = nn.Linear(128,4)
        self.fat = nn.Linear(128,1)
        self.stim = nn.Linear(128,1)

    def forward(self, obs, act):
        h = self.net(torch.cat([obs, act], -1))
        return torch.sigmoid(self.eng(h)), self.emo(h), torch.sigmoid(self.fat(h)), self.stim(h)

# ============================================================
# ABLATION ENVIRONMENT
# ============================================================
class AblationEnv:
    def __init__(self, surrogate, persona, ablation="full"):
        self.m = surrogate
        self.p = persona
        self.ablation = ablation  # "full", "no_emotion", "no_confidence", "no_safety"
        self.sc = 0
        self.obs_full = None  # always maintain full 9-dim for surrogate
        self.pe = None

    def _get_agent_obs(self):
        """Return observation according to ablation config."""
        o = self.obs_full.copy()
        if self.ablation == "no_emotion":
            # Remove emotion probs (indices 2-5), keep [eng, delta, conf, fatigue, step]
            return np.array([o[0], o[1], o[6], o[7], o[8]], dtype=np.float32)
        elif self.ablation == "no_confidence":
            # Remove confidence (index 6), keep [eng, delta, 4 emo probs, fatigue, step]
            return np.array([o[0], o[1], o[2], o[3], o[4], o[5], o[7], o[8]], dtype=np.float32)
        else:
            return o

    @property
    def obs_dim(self):
        if self.ablation == "no_emotion": return 5
        elif self.ablation == "no_confidence": return 8
        else: return 9

    def reset(self):
        self.sc = 0
        ie = {"P1":.8,"P2":.45,"P3":.15}[self.p]
        self.pe = ie
        self.obs_full = build_obs(ie, 0, "Natural", .5, 0, 0)
        return self._get_agent_obs()

    def step(self, aidx):
        self.sc += 1
        af = act_feat(aidx)
        with torch.no_grad():
            # Surrogate always uses full 9-dim obs
            ot = torch.FloatTensor(self.obs_full).unsqueeze(0).to(DEVICE)
            at = torch.FloatTensor(af).unsqueeze(0).to(DEVICE)
            pe, pm, pf, ps = self.m(ot, at)

        ne = float(np.clip(pe.squeeze().cpu(), 0, 1))
        el = pm.squeeze().cpu().numpy()
        ep = np.exp(el)/np.exp(el).sum()
        ei = np.argmax(ep); emo = EXP2_CLASSES[ei]
        conf = float(ep[ei])
        fat = float(np.clip(pf.squeeze().cpu(), 0, 1))
        ss = float(ps.squeeze().cpu()) > 0

        delta = ne - self.pe
        self.obs_full = build_obs(ne, delta, emo, conf, fat, self.sc)
        rew = self._reward(ne, emo, conf, ss, aidx)
        self.pe = ne
        done = self.sc >= STEPS
        return self._get_agent_obs(), rew, done, {"engagement":ne,"emotion":emo,"confidence":conf,"self_stim":ss}

    def _reward(self, eng, emo, conf, ss, aidx):
        a = act_dict(aidx)
        lo, hi = TARGET_BANDS[self.p]

        # 1. Engagement change (0.3)
        er = ((eng - self.pe) * 2) if self.pe else 0

        # 2. Target-band (0.3)
        if lo <= eng <= hi:
            band_r = 0.3
        elif eng < lo:
            band_r = -0.1 * (lo - eng) * 5
        else:
            band_r = -0.2 * (eng - hi) * 5

        # 3. Emotion valence (0.15)
        mr = EMO_VAL.get(emo, 0) * .15

        # 4. Overstimulation penalty (0.15) — REMOVED in no_safety ablation
        pen = 0
        if self.ablation != "no_safety":
            if self.p == "P3":
                if a["speech_rate"] == "fast": pen -= .3
                if a["stimulus"] == "high": pen -= .4
            elif self.p == "P2":
                if a["speech_rate"] == "fast" and a["stimulus"] == "high": pen -= .2
            if ss: pen -= .3

        # 5. Confidence-aware (0.1)
        cons = a["speech_rate"]=="slow" and a["stimulus"]=="low" and a["topic"]=="maintain"
        u = 1 - conf
        cr = u * .2 if cons else -u * .1
        if conf > .5 and emo in ["Fear","Anger"] and cons: cr += .15

        return float(np.clip(er + band_r + mr + pen + cr, -2, 2))

# ============================================================
# SAC AGENT (supports variable obs_dim)
# ============================================================
class SACAgent:
    def __init__(self, obs_dim):
        self.name = "SAC"
        self.obs_dim = obs_dim
        def mq():
            return nn.Sequential(nn.Linear(obs_dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,N_ACTIONS)).to(DEVICE)
        self.q1,self.q2,self.q1t,self.q2t = mq(),mq(),mq(),mq()
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.pi = nn.Sequential(nn.Linear(obs_dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,N_ACTIONS),nn.Softmax(dim=-1)).to(DEVICE)
        self.qopt = optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=3e-4)
        self.popt = optim.Adam(self.pi.parameters(), lr=3e-4)
        self.te = -np.log(1./N_ACTIONS) * .98
        self.la = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.aopt = optim.Adam([self.la], lr=3e-4)
        self.buf = []; self.tau = .005

    def select_action(self, obs):
        with torch.no_grad():
            p = self.pi(torch.FloatTensor(obs).to(DEVICE))
            return torch.distributions.Categorical(p).sample().item()

    def update(self, o, a, r, n, d):
        self.buf.append((o, a, r, n, d))
        if len(self.buf) > 50000: self.buf.pop(0)
        if len(self.buf) < 128: return
        b = random.sample(self.buf, 64)
        ob = torch.FloatTensor([x[0] for x in b]).to(DEVICE)
        ab = torch.LongTensor([x[1] for x in b]).to(DEVICE)
        rb = torch.FloatTensor([x[2] for x in b]).to(DEVICE)
        nb = torch.FloatTensor([x[3] for x in b]).to(DEVICE)
        db = torch.FloatTensor([x[4] for x in b]).to(DEVICE)
        alpha = self.la.exp().item()
        with torch.no_grad():
            np_ = self.pi(nb); nlp = torch.log(np_+1e-8)
            nq = torch.min(self.q1t(nb), self.q2t(nb))
            tgt = rb + (1-db) * .99 * (np_*(nq-alpha*nlp)).sum(1)
        q1v = self.q1(ob).gather(1,ab.unsqueeze(1)).squeeze()
        q2v = self.q2(ob).gather(1,ab.unsqueeze(1)).squeeze()
        ql = ((q1v-tgt).pow(2)+(q2v-tgt).pow(2)).mean()
        self.qopt.zero_grad(); ql.backward(); self.qopt.step()
        p = self.pi(ob); lp = torch.log(p+1e-8)
        mq = torch.min(self.q1(ob), self.q2(ob))
        pl = (p*(alpha*lp-mq)).sum(1).mean()
        self.popt.zero_grad(); pl.backward(); self.popt.step()
        ent = -(p*lp).sum(1).mean()
        al = -(self.la*(ent.detach()-self.te))
        self.aopt.zero_grad(); al.backward(); self.aopt.step()
        for p1,t1 in zip(self.q1.parameters(),self.q1t.parameters()): t1.data.copy_(self.tau*p1.data+(1-self.tau)*t1.data)
        for p2,t2 in zip(self.q2.parameters(),self.q2t.parameters()): t2.data.copy_(self.tau*p2.data+(1-self.tau)*t2.data)

# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_agent(agent, env, n_ep):
    rets, engs = [], []
    for ep in range(n_ep):
        obs = env.reset(); er = 0; ee = []
        for _ in range(STEPS):
            a = agent.select_action(obs)
            nobs, r, done, info = env.step(a)
            agent.update(obs, a, r, nobs, float(done))
            er += r; ee.append(info["engagement"]); obs = nobs
        rets.append(er); engs.append(np.mean(ee))
        if (ep+1) % 100 == 0:
            print(f"      Ep {ep+1}: avg_ret={np.mean(rets[-50:]):.2f}")
    return rets, engs

def evaluate(agent, env, n=100):
    rets, engs, ov, ts, emos, band_hits = [], [], 0, 0, [], 0
    lo, hi = TARGET_BANDS[env.p]
    for _ in range(n):
        obs = env.reset(); er = 0; ee = []
        for __ in range(STEPS):
            a = agent.select_action(obs)
            nobs, r, done, info = env.step(a)
            er += r; ee.append(info["engagement"]); emos.append(info["emotion"])
            ad = act_dict(a)
            if ad["speech_rate"]=="fast" and ad["stimulus"]=="high": ov += 1
            if lo <= info["engagement"] <= hi: band_hits += 1
            ts += 1; obs = nobs
        rets.append(er); engs.append(np.mean(ee))
    neg = (emos.count("Fear")+emos.count("Anger"))/len(emos)
    return {"mean_return":float(np.mean(rets)), "std_return":float(np.std(rets)),
            "mean_engagement":float(np.mean(engs)), "target_band_rate":band_hits/ts,
            "overstim_rate":ov/ts, "negative_emotion_rate":neg}

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=500)
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load surrogates
    surrogates = {}
    for p in PERSONAS:
        path = os.path.join(OUTPUT_DIR, f"surrogate_{p}.pt")
        model = Surrogate().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        surrogates[p] = model
    print("  Surrogates loaded.")

    # Ablation configs
    ablations = {
        "SA-Full":         {"ablation": "full",          "desc": "Complete (9-dim obs + full reward)"},
        "SA-1 w/o emotion":    {"ablation": "no_emotion",    "desc": "Remove 4-dim emotion probs (obs=5-dim)"},
        "SA-2 w/o confidence": {"ablation": "no_confidence", "desc": "Remove confidence (obs=8-dim)"},
        "SA-3 w/o safety":     {"ablation": "no_safety",     "desc": "Remove overstimulation penalty"},
    }

    all_results = {}
    for ab_name, ab_cfg in ablations.items():
        print(f"\n{'='*60}")
        print(f"  {ab_name}: {ab_cfg['desc']}")
        print(f"{'='*60}")

        ab_results = {}
        for p in PERSONAS:
            print(f"\n    --- {p} ---")
            seed_evals = []
            for seed in SEEDS:
                set_seed(seed)
                env = AblationEnv(surrogates[p], p, ab_cfg["ablation"])
                agent = SACAgent(env.obs_dim)
                rets, engs = train_agent(agent, env, args.n_episodes)
                ev = evaluate(agent, env)
                seed_evals.append(ev)

            # Average across seeds
            avg = {k: np.mean([e[k] for e in seed_evals]) for k in seed_evals[0]}
            std_ret = np.std([e["mean_return"] for e in seed_evals])
            print(f"      Ret={avg['mean_return']:.2f}±{std_ret:.2f} Band={avg['target_band_rate']*100:.1f}% Ovstim={avg['overstim_rate']*100:.1f}% NegEmo={avg['negative_emotion_rate']*100:.1f}%")
            ab_results[p] = {"eval": avg, "std_return": float(std_ret)}

        all_results[ab_name] = ab_results

    # Save
    save_path = os.path.join(OUTPUT_DIR, "ablation_results_exp3.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))
    print(f"\n  Saved: {save_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Config':<22} {'Persona':<6} {'Return':>8} {'Band%':>7} {'Ovstim%':>8} {'NegEmo%':>8}")
    print(f"  {'-'*65}")
    for ab_name in ablations:
        for p in PERSONAS:
            ev = all_results[ab_name][p]["eval"]
            print(f"  {ab_name:<22} {p:<6} {ev['mean_return']:>8.2f} {ev['target_band_rate']*100:>6.1f}% {ev['overstim_rate']*100:>7.1f}% {ev['negative_emotion_rate']*100:>7.1f}%")
        print()

    # Degradation vs full
    print(f"\n  DEGRADATION vs SA-Full:")
    print(f"  {'Config':<22} {'P1 Ret Δ':>10} {'P2 Ret Δ':>10} {'P3 Ret Δ':>10} {'P3 Band Δ':>10} {'P3 Ovstim Δ':>12}")
    full = all_results["SA-Full"]
    for ab_name in list(ablations.keys())[1:]:
        ab = all_results[ab_name]
        row = f"  {ab_name:<22}"
        for p in PERSONAS:
            delta = ab[p]["eval"]["mean_return"] - full[p]["eval"]["mean_return"]
            row += f"{delta:>+10.2f}"
        p3_band = (ab["P3"]["eval"]["target_band_rate"] - full["P3"]["eval"]["target_band_rate"]) * 100
        p3_ov = (ab["P3"]["eval"]["overstim_rate"] - full["P3"]["eval"]["overstim_rate"]) * 100
        row += f"{p3_band:>+10.1f}%{p3_ov:>+11.1f}%"
        print(row)


if __name__ == "__main__":
    main()
