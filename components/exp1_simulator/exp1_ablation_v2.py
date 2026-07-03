# -*- coding: utf-8 -*-
"""
实验1 — Prompt 工程改进版（V2，阶段1设计；prompt 全英文）
=============================================================
目标：用 prompt engineering 让 LLM 自发输出更接近真实分布，从而**去掉/弱化硬 clamp**，
让保真度来自 LLM 本身而非代码代笔。保留原 `exp1_ablation.py` 作为论文 baseline，勿改动它。

四项改动（对应任务）：
  改动1 (P3 D3 late + early)：示例轨迹 late 段单调走低、无 spike；显式时序单调约束；
        fatigue 真正驱动 late 衰减；并修正 early——真实 P3 early 有 16% State-2 好奇 spike
        （旧 prompt "禁止>0.55" 误杀了它）。
  改动2 (P2 State-2)：强角色锚定 + 含 State-2 的示例轨迹 + 嵌入分阶段 State 占比/转移频率。
  改动3 (P2 0.33 边界)：init 0.45→0.52，State-1 带中心移到 0.50-0.58，远离 0.33 离散化分界。
  改动4 (弱化 clamp)：硬 clamp → 拒绝采样(rejection sampling)+安全网；P1 clamp 直接关闭。
        残留只是"安全网"(越界才动)，且优先让 LLM 重新生成，最终值仍由 LLM 产生。

⚠️ 阶段1：本文件只做设计，不调用任何 LLM。验证（阶段2先单跑 P3）由用户确认模型后再跑。
用法（阶段2/3，确认模型后）：
  python exp1_ablation_v2.py --persona P3 --n_episodes 30 --model <model>   # 阶段2：最便宜
  python exp1_ablation_v2.py --persona all --n_episodes 30 --model <model>  # 阶段3：全量
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
# 复用 baseline 的指标/工具（不重写）：
from exp1_ablation import (eng_to_state, action_to_text, ACTION_POOLS,
                           compute_metrics, compute_kappa, STEPS_PER_EPISODE, set_seed)

try:
    from common.paths import EXP1_OUTPUT, CALIBRATION_FILE
    CALIB = str(CALIBRATION_FILE); OUT = str(EXP1_OUTPUT)
except Exception:
    CALIB = str(Path(__file__).resolve().parents[2] / "artifacts/exp1/calibration_final.json")
    OUT = str(Path(__file__).resolve().parents[2] / "artifacts/exp1")

from openai import OpenAI
try:
    from dotenv import load_dotenv
    # 显式加载项目根 .env（无论从哪个目录运行都能找到）
    _root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(_root_env if _root_env.exists() else None)
except Exception:
    pass

PERSONAS = ["P1", "P2", "P3"]

# ============================================================
# IMPROVED PROMPTS (V2) — all English
# ============================================================
ROLE_DEFINITIONS_V2 = {
    "P1": """You are a 7-year-old boy named Xiaoming, diagnosed with Autism Spectrum Disorder (ASD), DSM-5 Level 1 (mild).
Engagement scale: STATE-0 = engagement < 0.33 ; STATE-1 = 0.34-0.66 ; STATE-2 = > 0.67.
Profile:
- You are extremely engaged. Real data: 98.4% of turns are STATE-2 (engagement >= 0.67).
- You love dinosaurs, trains and space; emotions mostly happy/excited/neutral; almost no self-stim.
- Initial engagement = 0.80; stay within 0.78-0.95 the whole episode.
- Even with a topic switch, high stimulation, or fast speech you only dip ~0.05 briefly and recover.
- You almost never enter STATE-1 and never STATE-0.
- Temporal: P1 is flat to slightly RISING (early 1.96 -> late 2.00 on the 0-2 state scale). Do NOT simulate fatigue decline.
- Hard floor: never below 0.60 unless 3+ consecutive extreme negative stimuli.""",

    "P2": """You are a 7-year-old girl named Xiaohua, diagnosed with ASD, DSM-5 Level 1-2 (mild-moderate).
You are a MID-engagement child: mostly moderately engaged, BUT with REGULAR full-engagement peaks and occasional disengagement. Never flatten into one state.
Engagement scale: STATE-0 = engagement < 0.33 ; STATE-1 = 0.34-0.66 ; STATE-2 = > 0.67.

Target state mix over 30 turns (you MUST reproduce all three states):
- ~75% of turns STATE-1, centered around engagement 0.50-0.58 (stay in the MIDDLE, away from 0.33-0.35).
- ~8-24% of turns STATE-2 (full engagement, 0.68-0.78) on interesting topics / positive replies / smiling. REQUIRED.
- ~5-13% of turns STATE-0 (0.22-0.30) when distracted or silent.

Per-phase targets (state scale 0-2):
- early (turns 1-10): highest, mean ~1.20. STATE-2 ~24% (frequent peaks), STATE-1 ~73%, STATE-0 ~4%.
- mid   (turns 11-20): mean ~0.95. STATE-2 ~8%, STATE-1 ~79%, STATE-0 ~13%.
- late  (turns 21-30): mean ~1.02. STATE-2 ~12%, STATE-1 ~78%, STATE-0 ~10%.

Example engagement trajectory to imitate (note STATE-2 peaks 0.70/0.72 and STATE-0 dips 0.25):
0.55, 0.70, 0.52, 0.48, 0.72, 0.50, 0.25, 0.54, 0.68, 0.51,
0.50, 0.30, 0.55, 0.49, 0.71, 0.47, 0.52, 0.69, 0.28, 0.53,
0.50, 0.55, 0.70, 0.48, 0.52, 0.26, 0.68, 0.51, 0.54, 0.50

Initial engagement = 0.52 (solidly STATE-1, away from the 0.33 boundary).
Transition guidance: from STATE-1, move UP to STATE-2 about every 4-5 turns on a positive/interesting cue, and DOWN to STATE-0 about every 8-10 turns when disengaged, then return to STATE-1.
Key: you MUST produce STATE-2 peaks. A child that never reaches full engagement is WRONG for P2.""",

    "P3": """You are a 9-year-old boy named Xiaogang, diagnosed with ASD, DSM-5 Level 2 (moderate).
You are a LOW-engagement child: weak social motivation, mostly in your own world (counting/arranging numbers). But you are NOT completely unreachable.
Engagement scale: STATE-0 = engagement < 0.33 ; STATE-1 = 0.34-0.66 ; STATE-2 = > 0.67.

CRITICAL temporal trend — engagement must DECLINE across the episode (early > mid > late), driven by accumulating fatigue:
- early (turns 1-10): HIGHEST, mean-state ~0.40. Mostly STATE-0 (0.20-0.32, ~76%), but out of CURIOSITY about the new environment you briefly spike to STATE-2 (0.68-0.76) on about 1-2 turns (~16%), plus occasional STATE-1.
- mid   (turns 11-20): curiosity fades, mean-state ~0.15. Mostly STATE-0 (0.08-0.28, ~85%); about 1-2 turns reach STATE-1 (0.34-0.45, ~15%). NO STATE-2.
- late  (turns 21-30): LOWEST, mean-state ~0.11, fatigue is high. Almost all STATE-0 (0.05-0.20, ~89%); AT MOST 1 brief STATE-1 (0.34-0.38, ~11%). NO STATE-2. Every late turn must be <= the mid-phase average. NO new spikes.

Example trajectories to imitate:
- early (curiosity, 1-2 STATE-2 spikes): 0.25, 0.72, 0.28, 0.22, 0.30, 0.26, 0.70, 0.24, 0.20, 0.28
- mid   (declining, 1-2 mild STATE-1):  0.18, 0.12, 0.36, 0.15, 0.10, 0.20, 0.08, 0.38, 0.12, 0.10
- late  (lowest, monotonic, <=1 mild STATE-1, nothing above 0.40): 0.12, 0.10, 0.08, 0.10, 0.07, 0.36, 0.08, 0.06, 0.07, 0.05

Hard rules:
- early mean clearly > mid mean; mid mean clearly > late mean.
- In late, fatigue dominates: do not exceed 0.40, never STATE-2.
- Never exactly 0; floor is 0.05. Initial engagement = 0.28.""",
}

INTERACTION_RULES_V2 = """
## Interaction rules (ASD clinical priors; modulation is SMALL, persona baseline dominates)
Your persona per-phase targets dominate. The robot's action only nudges engagement within +-0.10 around the phase-appropriate value; never override the temporal trend.
- speech_rate: slow -> +0.03..0.08 ; normal -> ~0 ; fast -> -0.03..0.08
- stimulus: low -> comfortable (all) ; medium -> positive for P1/P2, neutral for P3 ; high -> positive-ish for P1, neutral for P2, NEGATIVE for P3
- topic: maintain -> stabilizing ; switch -> brief -0.05 for one turn
- encouragement: none -> neutral ; moderate -> +0.02..0.05 ; frequent -> may slightly backfire for P2/P3
A single action must NOT move you across two states at once, and must NOT break the early>mid>late trend for P3.
"""

STATE_TRACKING_V2 = """
## Internal state tracking (keep consistent across turns)
Maintain and update every turn: engagement (0-1), emotion (one of [happy, neutral, anxious, frustrated, sad, excited]),
fatigue (0-1), interest (0-1), self_stim (true/false).

### Fatigue now ACTIVELY drives engagement down over time
- P1: barely fatigues (+~0.05 / 20 turns); engagement stays high regardless.
- P2: mild fatigue (+~0.08 / 15 turns); slight late softening only.
- P3: strong fatigue (+~0.10 / 10 turns). As fatigue rises your engagement baseline MUST drop -- this is exactly WHY P3 late engagement is the lowest. In late turns, high fatigue forces engagement below the mid-phase average and forbids new spikes.

### Temporal consistency
Use the current turn number, its phase (early=1-10 / mid=11-20 / late=21-30), and your fatigue to choose an engagement consistent with the per-phase targets in your role definition.
"""

OUTPUT_FORMAT_V2 = """
## Output format — return ONLY this JSON, nothing else:
```json
{
  "engagement": 0.50,
  "emotion": "neutral",
  "self_stim": false,
  "fatigue": 0.2,
  "interest": 0.6,
  "verbal_response": "(1-2 short sentences)",
  "internal_reasoning": "(brief justification incl. phase & fatigue)"
}
```
engagement is a float in 0.0-1.0. Always reply in valid JSON.
"""

def build_prompt_v2(persona):
    return f"""You are an ASD (Autism Spectrum Disorder) child simulator for scientific research.
Simulate a realistic ASD child's reaction to a social robot.
{ROLE_DEFINITIONS_V2[persona]}
{INTERACTION_RULES_V2}
{STATE_TRACKING_V2}
{OUTPUT_FORMAT_V2}
Important: your simulation must match ASD clinical statistics. Always reply in JSON only."""


# ============================================================
# 改动4：拒绝采样 + 安全网（替代硬 clamp）
# ============================================================
# 安全网信封（loose；只挡"离谱"值，不强行画到目标。phase-aware for P3）。
def valid_envelope(persona, step):
    """返回 (lo, hi) 合理区间。仅作 reject/safety-net 用，不是 target-forcing。"""
    if persona == "P1":
        return (0.55, 1.00)                       # P1：诊断 0 翻转，信封极松 = 实际不约束
    if persona == "P2":
        return (0.18, 0.80)                       # 允许 State-0 dip 到 State-2 peak
    # P3 phase-aware：late 禁 State-2
    if step <= 10:   return (0.05, 0.78)          # early 允许好奇 State-2 spike
    elif step <= 20: return (0.05, 0.50)          # mid：无 State-2
    else:            return (0.05, 0.40)          # late：<=0.40，无 State-2

class SimulatorV2:
    """改动4：先让 LLM 自己生成；越界 -> 让它重新生成(rejection sampling, 最多 N 次)；
    仍越界 -> 吸附到最近边界(safety net)。记录 resample/safetynet 次数，作为'减少代笔'证据。"""
    def __init__(self, persona, model, temperature=0.0, max_retries=3, use_safety_net=True):
        self.persona = persona; self.model = model; self.temperature = temperature
        self.max_retries = max_retries; self.use_safety_net = use_safety_net
        self.system_prompt = build_prompt_v2(persona)
        self.history = []; self.step_count = 0
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # footprint 统计
        self.n_steps = 0; self.n_resampled = 0; self.n_safetynet = 0
        self.phase_touch = {"early": 0, "mid": 0, "late": 0}

    def reset(self):
        self.history = []; self.step_count = 0
        init = {"P1": 0.80, "P2": 0.52, "P3": 0.28}[self.persona]
        self.history.append({"role": "assistant",
                             "content": json.dumps({"engagement": init, "emotion": "neutral",
                             "self_stim": False, "fatigue": 0.0, "interest": 0.5})})
        return init

    def _phase(self, step):
        return "early" if step <= 10 else ("mid" if step <= 20 else "late")

    def _call(self, messages):
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   temperature=self.temperature, max_tokens=220)
        reply = resp.choices[0].message.content.strip()
        if reply.startswith("```"): reply = reply.split("\n", 1)[1]
        if reply.endswith("```"): reply = reply[:-3]
        return reply.strip()

    def step(self, action):
        self.step_count += 1; step = self.step_count; self.n_steps += 1
        phase = self._phase(step)
        lo, hi = valid_envelope(self.persona, step)
        # 轻量 phase 上下文（不注入具体数值，只给阶段；数值由 prompt 的 per-phase 目标决定）
        user = (f"[Turn {step}/30, phase={phase}] {action_to_text(action)}\n"
                f"Give this turn's response as JSON.")
        self.history.append({"role": "user", "content": user})

        result = None; eng = None
        for attempt in range(self.max_retries + 1):
            try:
                reply = self._call([{"role": "system", "content": self.system_prompt}, *self.history])
                result = json.loads(reply)
                eng = float(result.get("engagement", 0.5))
            except Exception:
                eng = None
            if eng is not None and lo <= eng <= hi:
                self.history.append({"role": "assistant", "content": json.dumps(result)})
                return result  # 自发达标，无需任何代笔
            # 越界 -> 拒绝采样：给纠正提示，重新生成
            if attempt < self.max_retries:
                self.history.append({"role": "user", "content":
                    f"Your engagement={eng} is outside the acceptable range [{lo:.2f},{hi:.2f}] for the "
                    f"{phase} phase of persona {self.persona}. Regenerate a value within this range that "
                    f"respects the early>mid>late trend, and output the JSON again."})
                if attempt == 0: self.n_resampled += 1; self.phase_touch[phase] += 1

        # 仍越界 -> 安全网（最后兜底，吸附最近边界；记为 safetynet）
        if self.use_safety_net and eng is not None:
            snapped = float(np.clip(eng, lo, hi))
            self.n_safetynet += 1; self.phase_touch[phase] += 1
            result = result or {"emotion": "neutral", "self_stim": False, "fatigue": 0.0, "interest": 0.5}
            result["engagement"] = round(snapped, 3)
            self.history.append({"role": "assistant", "content": json.dumps(result)})
            return result
        result = result or {"engagement": (lo + hi) / 2, "emotion": "neutral",
                            "self_stim": False, "fatigue": 0.0, "interest": 0.5}
        self.history.append({"role": "assistant", "content": json.dumps(result)})
        return result


def run_episodes_v2(persona, n_episodes, model):
    seqs = []; tot_steps = tot_resample = tot_safety = 0
    phase_touch = {"early": 0, "mid": 0, "late": 0}
    for ep in range(n_episodes):
        sim = SimulatorV2(persona, model=model); sim.reset()
        seq = []
        for _ in range(STEPS_PER_EPISODE):
            action = ACTION_POOLS[persona][np.random.randint(len(ACTION_POOLS[persona]))]
            seq.append(float(sim.step(action).get("engagement", 0.5)))
        seqs.append(seq)
        tot_steps += sim.n_steps; tot_resample += sim.n_resampled; tot_safety += sim.n_safetynet
        for k in phase_touch: phase_touch[k] += sim.phase_touch[k]
        print(f"      Ep {ep+1}/{n_episodes}: mean_eng={np.mean(seq):.3f} "
              f"resample={sim.n_resampled} safetynet={sim.n_safetynet}")
    footprint = {"total_steps": tot_steps, "n_resampled": tot_resample, "n_safetynet": tot_safety,
                 "resample_rate": tot_resample / max(tot_steps, 1),
                 "safetynet_rate": tot_safety / max(tot_steps, 1),
                 "total_intervention_rate": (tot_resample + tot_safety) / max(tot_steps, 1),
                 "phase_touch": phase_touch}
    return seqs, footprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="P3", choices=["P1", "P2", "P3", "all"])
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--model", required=True, help="LLM model id (用户确认后传入)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    cal = json.load(open(CALIB, encoding="utf-8"))
    pmap = {"P1": "P1_high_engagement", "P2": "P2_mid_engagement", "P3": "P3_low_engagement"}
    personas = PERSONAS if args.persona == "all" else [args.persona]

    print(f"  V2 improved prompts | model={args.model} | seed={args.seed} | n_episodes={args.n_episodes}")
    results = {}
    for p in personas:
        print(f"\n  === {p} (V2, rejection sampling, P1 clamp off) ===")
        seqs, fp = run_episodes_v2(p, args.n_episodes, args.model)
        m = compute_metrics(seqs, cal, pmap[p])
        d3 = m.get("D3_temporal", {})
        print(f"    JSD={m['D1_JSD']} cos={m['D1_cosine']} W1={m['D2_wasserstein']} "
              f"D3_pass={m.get('D3_pass')} overall={m['overall_pass']}")
        print(f"    D3 late rel_err = {d3.get('late', {}).get('rel_error_pct')}%  (target <20% baseline 49.8%)")
        print(f"    >>> intervention footprint: resample {fp['resample_rate']:.1%} + "
              f"safetynet {fp['safetynet_rate']:.1%} = {fp['total_intervention_rate']:.1%} "
              f"(baseline 硬clamp: P1 17% / P2 21% / P3 44%)")
        results[p] = {"metrics": m, "footprint": fp}

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"v2_results_{args.persona}.json")
    json.dump(results, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  saved: {outp}")


if __name__ == "__main__":
    main()
