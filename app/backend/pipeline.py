# -*- coding: utf-8 -*-
"""
Live end-to-end pipeline for the runnable demo.

One step of the closed loop:
  LLM child (YOUR OpenAI key) -> displayed emotion -> FERAC image
   -> REAL Exp2 perception (with abstention) -> 9-D observation (real/sim tagged)
   -> SAC decision -> deterministic R1-R4 safety shield -> reward decomposition
   -> action sent back to the LLM child.

Honest scope: SIMULATED demonstration of module integration. The "child" is an LLM,
not a real child; no efficacy claims; "hand-back-to-human" is a simulated pause.
"""
import os, sys, json, glob, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "components/exp1_simulator"))
sys.path.append(str(ROOT / "components/exp3_policy"))

from openai import OpenAI
from exp1_ablation_v2 import build_prompt_v2
from exp3_generate import ALL_ACTIONS, action_to_text, action_to_idx, map_emotion
from exp3_sac import SACAgent, act_dict, build_obs, EXP2_CLASSES, TARGET_BANDS
from exp3_sac_train import reward_decomp, DEVICE
from exp3_safety import shield, shield_tiered, gate_tier, CONSERVATIVE
from system.inference.perception import Perception
from system.inference.contract import PerceptionConfig

INIT = {"P1": 0.80, "P2": 0.52, "P3": 0.28}
EMO2DIR = {"Natural": "Natural", "Anger": "anger", "Fear": "fear", "Joy": "joy"}
FER_OUT = ["natural", "anger", "fear", "joy"]
STEPS = 30
EMOTION_DIVERSITY = """
## Emotion diversity (required for this study)
Set `emotion` to EXACTLY one of: happy, neutral, anxious, frustrated, sad, excited.
Do NOT output "neutral" every turn — choose the emotion that fits your current engagement and the robot's action.
Vary emotion across turns in line with your engagement trajectory.

## Output format (STRICT — keep it short)
Respond with ONLY a compact JSON object on a single line. Required keys:
"engagement" (float 0-1), "emotion" (one word from the list), "self_stim" (bool), "fatigue" (float 0-1), "interest" (float 0-1).
Do NOT add long free-text. If you include any reasoning/verbal field, keep it UNDER 8 words.
The entire response MUST be short (well under 100 tokens).
"""

# ---- shared singletons (loaded once) ----
_PERC = Perception(checkpoint_path=str(ROOT / "models/efficientnet_ferac.pt"))
_PCFG = PerceptionConfig(face_required=False, blur_threshold=0.0)  # FERAC = pre-cropped faces; Haar limitation F2-6

def _load_sac(persona):
    ag = SACAgent()
    sd = ROOT / f"models/sac_{persona}.pt"
    if not sd.exists():
        sd = ROOT / f"artifacts/exp3/sac/sac_{persona}.pt"
    ag.pi.load_state_dict(torch.load(sd, map_location=DEVICE))
    return ag

def _perceive(emo4, rng):
    d = ROOT / f"data/ferac/test/{EMO2DIR.get(emo4, 'Natural')}"
    imgs = sorted(glob.glob(str(d / "*")))
    if not imgs:  # data/ferac not present -> degrade to abstain (honest)
        return np.full(4, 0.25, np.float32), 0.25, True, "abstain", "no_data", None, "natural"
    img = imgs[rng.randint(len(imgs))]
    r = _PERC.predict(img, _PCFG)
    probs = (np.array([r.emotion_probs.get(k, 0.25) for k in FER_OUT], np.float32)
             if r.emotion_probs else np.full(4, 0.25, np.float32))
    conf = float(r.confidence) if r.confidence is not None else float(probs.max())
    pred_class = FER_OUT[int(np.argmax(probs))]  # for per-class 3-level gating
    return probs, conf, (r.decision == "abstain"), r.decision, r.abstain_reason, os.path.basename(img), pred_class


class Session:
    def __init__(self, persona, api_key, model="gpt-4o", seed=7):
        self.persona = persona
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.rng = np.random.RandomState(seed)
        self.sac = _load_sac(persona)
        self.lo, self.hi = TARGET_BANDS[persona]
        self.mid = (self.lo + self.hi) / 2
        self.sysp = build_prompt_v2(persona) + EMOTION_DIVERSITY
        ie = INIT[persona]
        self.eng = ie; self.prev_eng = ie; self.fat = 0.0
        # the emotion the child is CURRENTLY displaying (the one perceived to build the current obs)
        self.cur_emo_raw = "neutral"; self.cur_emo4 = "Natural"
        self.history = [{"role": "assistant", "content": json.dumps(
            {"engagement": ie, "emotion": "neutral", "self_stim": False, "fatigue": 0.0, "interest": 0.5})}]
        self.step_i = 0
        self.concern = []
        self.handback_done = False
        self.log = []
        # initial perception (of the initially-displayed emotion) — matches the initial obs
        self.probs, self.conf, self.abstain, self.decision, self.reason, self.img, self.pred_class = _perceive(self.cur_emo4, self.rng)
        self.tier = gate_tier(self.pred_class, self.conf)   # 3-level: accept/cautious/abstain
        self.obs = np.array([ie, 0.0, *self.probs, self.conf, self.fat, 0.0], np.float32)

    def _child_respond(self, action_idx):
        """LLM child responds to the executed action (uses the user's key).
        Robust JSON handling (same as the Exp3 data generation): json_object response format
        + fence-strip + retry-to-valid; only falls back to a safe default if all retries fail."""
        action = ALL_ACTIONS[action_idx]
        self.history.append({"role": "user",
                             "content": f"[Turn {self.step_i}/30] {action_to_text(action)}\nRespond as JSON."})
        last_err = None
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "system", "content": self.sysp}, *self.history],
                    temperature=0, max_tokens=600, response_format={"type": "json_object"})
                fin = r.choices[0].finish_reason
                txt = (r.choices[0].message.content or "").strip()
                if fin == "length":  # truncated -> would be invalid JSON; surface clearly
                    raise ValueError("response truncated (finish_reason=length)")
                if txt.startswith("```"):
                    txt = txt.split("\n", 1)[1] if "\n" in txt else txt[3:]
                if txt.endswith("```"):
                    txt = txt[:-3]
                res = json.loads(txt.strip())
                if "engagement" not in res:
                    raise ValueError("valid JSON but missing 'engagement'")
                self.history.append({"role": "assistant", "content": json.dumps(res)})
                eng = float(res.get("engagement", self.eng))
                emo_raw = str(res.get("emotion", "neutral"))
                fat = float(res.get("fatigue", self.fat))
                return eng, emo_raw, fat, None
            except Exception as e:
                last_err = e; time.sleep(0.4)
        # all retries failed -> safe fallback, keep conversation consistent
        self.history.append({"role": "assistant",
                             "content": json.dumps({"engagement": self.eng, "emotion": "neutral"})})
        return self.eng, "neutral", self.fat, f"LLM error ({type(last_err).__name__}) — using safe fallback"

    def step(self):
        if self.step_i >= STEPS:
            return {"done": True}
        self.step_i += 1
        obs = self.obs
        # ---- snapshot the CURRENT observed moment (perception that built `obs`) for a self-consistent record ----
        # confidence/probs/decision/reason/img ALL come from the same perception event that produced `obs`.
        cur_tier = self.tier
        cur = {"decision": self.decision, "confidence": round(float(obs[6]), 3),
               "abstain_reason": self.reason, "retrieved_img": self.img,
               "probs": {k: round(float(v), 3) for k, v in zip(FER_OUT, obs[2:6])},
               "predicted_class": self.pred_class, "gating_tier": cur_tier}   # 3-level
        cur_emo_raw, cur_emo4 = self.cur_emo_raw, self.cur_emo4
        # 1. SAC decision (argmax)
        with torch.no_grad():
            a_raw = int(self.sac.pi(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)).argmax().item())
        # 2. safety shield (3-level: accept->standby / cautious->soft_cap / abstain->full conservative)
        safe_a, intc, rules, level = shield_tiered(obs, a_raw, perc_tier=cur_tier)
        # 3. interpretability
        perc_emo = EXP2_CLASSES[int(np.argmax(obs[2:6]))]
        _, comps = reward_decomp(self.persona, self.prev_eng, float(obs[0]), perc_emo, float(obs[6]), False, safe_a)
        # R3 standby counterfactual (aggressive action under current tier)
        _, cf_intc, cf_rules, cf_level = shield_tiered(obs, 53, perc_tier=cur_tier)
        # 4. execute -> LLM child responds (this produces the NEXT displayed state)
        self.prev_eng = self.eng
        eng, emo_raw, fat, llm_err = self._child_respond(safe_a)
        self.eng, self.fat = eng, fat
        emo4, _ = map_emotion(emo_raw)
        # 5. real Exp2 perception of the NEW displayed emotion -> becomes the next observed moment
        self.probs, self.conf, self.abstain, self.decision, self.reason, self.img, self.pred_class = _perceive(emo4, self.rng)
        self.tier = gate_tier(self.pred_class, self.conf)
        self.cur_emo_raw, self.cur_emo4 = emo_raw, emo4
        # 6. hand-back monitor (3-level abstain tier + perceived distress)
        self.concern.append((self.tier == "abstain", EXP2_CLASSES[int(np.argmax(self.probs))] in ("Fear", "Anger")))
        w = self.concern[-5:]
        handback = (not self.handback_done and self.step_i >= 5
                    and sum(a for a, _ in w) >= 3 and sum(b for _, b in w) >= 3)
        if handback:
            self.handback_done = True
        # 7. build next obs
        self.obs = np.array([eng, eng - self.prev_eng, *self.probs, self.conf, fat, self.step_i / STEPS], np.float32)

        rec = {
            "done": self.step_i >= STEPS, "step": self.step_i, "llm_note": llm_err,
            "child": {"engagement": round(float(obs[0]), 3), "true_emotion_raw": cur_emo_raw,
                      "displayed_emotion": cur_emo4, "fatigue": round(float(obs[7]), 3)},
            "perception": cur,
            "obs_9d": {"engagement[SIM]": round(float(obs[0]), 3), "delta[SIM]": round(float(obs[1]), 3),
                       "emotion_probs[REAL]": [round(float(x), 3) for x in obs[2:6]],
                       "confidence[REAL]": round(float(obs[6]), 3),
                       "fatigue[SIM]": round(float(obs[7]), 3), "tau[SIM]": round(float(obs[8]), 3)},
            "sac_action": _dec(a_raw),
            "safety": {"intercepted": intc, "rules": rules, "executed": _dec(safe_a), "level": level,
                       "r3_standby": {"aggressive_would_intercept": cf_intc, "rules": cf_rules, "level": cf_level}},
            "reward_decomposition": {k: round(v, 3) for k, v in comps.items()},
            "handback": handback,
        }
        self.log.append(rec)
        return rec

    def summary(self):
        n = len(self.log)
        tc = lambda t: sum(1 for r in self.log if r["perception"]["gating_tier"] == t)
        lc = lambda l: sum(1 for r in self.log if r["safety"]["level"] == l)
        return {"persona": self.persona, "steps": n,
                "tier_accept": tc("accept"), "tier_cautious": tc("cautious"), "tier_abstain": tc("abstain"),
                "R3_soft_cap": lc("soft_cap"), "R3_full_conservative": lc("full_conservative"),
                "safety_interceptions": sum(1 for r in self.log if r["safety"]["intercepted"]),
                "handback_step": next((r["step"] for r in self.log if r["handback"]), None),
                "target_band": [self.lo, self.hi],
                "mean_engagement": round(float(np.mean([r["child"]["engagement"] for r in self.log])), 3) if n else None}


def _dec(a):
    d = act_dict(a)
    return f"{d['speech_rate']}/{d['stimulus']}/{d['topic']}/{d['encouragement']}"
