# System overview

The ASD Companion System chains three research modules into one closed loop. Each
module was developed and evaluated separately (Exp1/2/3); the system demonstrates
they **integrate** into a single safe, interpretable decision loop.

> Honest scope: a **simulation research framework**, not a medical device. No real
> children; no efficacy claims. See the root `README.md`.

## The three modules

### Exp1 — LLM-ASD child simulator
A prompt-engineered LLM (GPT-4o, V2 four-module prompt) simulating an ASD child's
engagement/emotion dynamics, **calibrated against real behavioural data** (Engagnition;
K-means → personas P1 high / P2 mid / P3 low engagement). Provides the "child" the
system interacts with (in the live app) or a learned **surrogate** of it (offline / static viewer).

### Exp2 — Confidence-aware perception
EfficientNet-B0 facial-emotion recognizer (4 classes: Natural/Anger/Fear/Joy) with
**temperature scaling** and an **honest-abstention gate**: when calibrated confidence is
below the operating threshold (0.86), it returns `abstain` as a **first-class output**
instead of a possibly-wrong label. This is the system's "unreliable signal" detector.

### Exp3 — Personalized + safe + interpretable policy
- **World model (surrogate)**: a per-persona MLP learned from offline LLM-generated data,
  predicting state transitions so the policy can train cheaply (no token cost). Engagement
  is modelled as a **classification + sampling** head to preserve behavioural diversity.
- **Policy (Discrete-SAC)**: a per-persona policy that keeps engagement in the persona's
  **target band** ("optimal arousal") while avoiding over-stimulation and negative affect.
- **Safety shield (R1–R4)**: a **deterministic, always-on hard guarantee** that intercepts
  rule-violating actions (low-engagement + high-stim, negative-emotion + escalation,
  low-confidence/abstain, fatigue + high-intensity) and replaces them with a conservative action.
- **Interpretability**: every step's reward is a deterministic **5-term decomposition**
  mapped to clinical intent (target-band ★arousal; over-stim / confidence-safety ★safety).

## The closed loop (one step)

```
 child state (engagement + emotion)            ── Exp1 LLM / surrogate (SIMULATED)
   │ displayed emotion
   ▼ retrieve a FERAC face image of that emotion
 perception                                     ── Exp2 (REAL): emotion probs + confidence (+ abstain)
   │
   ▼ build 9-D observation  [engagement, Δ, 4×emotion probs, confidence, fatigue, τ]
     REAL = emotion probs + confidence (Exp2) ; SIM = engagement / Δ / fatigue / τ
   │
   ▼ SAC decision  → proposed action
   │
   ▼ safety shield R1–R4  → executed (safe) action  (+ reward decomposition for interpretability)
   │
   └─▶ action fed back to the child → next step
```

When perception is **persistently unreliable** and the child shows **perceived distress**,
the system raises a **simulated "hand-back-to-human"** event: it pauses automatic
decision-making and surfaces a decision point. (No real human takes over.)

## Key system-level properties (all surrogate-internal, no efficacy claim)

| Property | Evidence (see `docs/paper_materials/`) |
|---|---|
| Personalized policy that maintains optimal arousal | `exp3_sac_training.md` (target-band hit rate, convergence) |
| Best vs. baselines | `exp3_baselines.md` (SAC ≈ PPO > DQN > Rule > Random; DQN = empirical evidence that a trained policy can still be unsafe) |
| Always-on hard safety guarantee | `exp3_safety.md` (100% interception, 0 false intercepts, R3 unreliable-signal fallback) + `exp3_sensitivity.md` (robust to thresholds) |
| Transparent, clinically-anchored decisions | `exp3_interpretability.md` (5-term decomposition + counterfactuals) |
| End-to-end integration | `exp3_walkthrough.md` + the static viewer in `results/` |

> The per-experiment reports under `docs/paper_materials/` are the detailed, honest write-ups
> (including limitations). They are research notes (currently in Chinese).
