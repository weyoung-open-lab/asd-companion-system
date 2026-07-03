# The end-to-end walkthrough, explained

The static viewer (`results/index.html`) replays a **recorded** end-to-end episode
(`results/walkthrough.json`). This page explains what each panel means and what the
key moments demonstrate.

> Honest scope: a **simulated** module-integration demo. The child is an LLM / learned
> surrogate; **no real children**; **no efficacy claims**. "Hand-back-to-human" is a
> simulated pause, not a real handoff.

## What each panel shows (per step)

| Panel | Meaning |
|---|---|
| **Child agent** `SIMULATED` | The child's hidden true state: engagement, true emotion, fatigue (driven by the LLM / surrogate). |
| **Perception (Exp2)** `REAL` | The **real** Exp2 recognizer's output on a FERAC face image of the child's displayed emotion: decision (accept/abstain), confidence, 4-class probabilities, abstain reason. Imperfect — may misclassify or abstain. |
| **9-D observation** | The state vector the policy sees. Each dimension is tagged `REAL` (from Exp2: emotion probs + confidence) or `SIM` (simulated: engagement / Δ / fatigue / τ). |
| **Robot decision & safety** | The SAC policy's proposed action, the deterministic R1–R4 shield verdict (allow / intercept + which rule), and the executed action. Plus an R3 "stand-by" counterfactual: would an aggressive action *here* be intercepted? |
| **Interpretability** | The deterministic 5-term reward decomposition for this step (hand-reproducible, mapped to clinical intent). |

The timeline at the top shows engagement over the episode, the target band, and markers
for perception accept/abstain, safety interceptions, and the hand-back trigger.

## The four key moments

1. **Normal cooperation** — perception accepts, the policy acts, the shield allows; engagement is maintained in band. (Clearest in the P1 episode.)
2. **Abstain → R3 fallback** — Exp2 returns `abstain` (unreliable signal). If the policy proposes a non-conservative action, **R3 deterministically forces a conservative action**. This is the "safe under unreliable signals" behaviour, happening live in the loop.
3. **Safety interception** — a proposed action triggers a rule (R1–R4); the shield replaces it with a conservative action, and the interpretability module attributes *which rule* fired and *why*.
4. **Hand-back-to-human (simulated)** — when perception is persistently unreliable **and** the child shows perceived distress, the system **pauses** and surfaces a decision point. (Shown in the constructed-distress episode `P3_distress`.) This demonstrates that the system **knows when to exit automatic decision-making** — it is *not* a real human taking over.

## Three recorded episodes

- **P1** (high engagement) — smooth normal cooperation, no interceptions, no hand-back.
- **P3** (low engagement, natural) — frequent abstentions, an abstain→R3 fallback; engagement stays in P3's (low) target band, so no hand-back is warranted.
- **P3_distress** (constructed) — escalating distress (forced fear) + persistent abstention → the hand-back trigger fires. Explicitly a constructed scenario to demonstrate the trigger.

## Honest caveats (visible in the data)

- Exp2 perception **misclassifies** sometimes (e.g. a Natural face read as anger) and abstains
  often under the strict 0.86 gate — shown, not hidden.
- The SAC policy was trained on *synthetic* perception; the walkthrough feeds it *real* Exp2
  outputs (different distribution). The demo shows the **wiring**, not policy optimality under
  real perception (in-the-loop retraining is future work).

For the full methodology and metrics, see `docs/paper_materials/exp3_walkthrough.md`.
