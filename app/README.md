# Runnable live demo (`app/`)

A small **frontend + backend** that runs the full closed loop in real time:

```
LLM child (YOUR OpenAI key)  ──displayed emotion──▶  FERAC face image
        ▲                                                    │
        │ executed action                                    ▼
   SAC ◀── safety shield R1–R4 ◀── 9-D observation ◀── REAL Exp2 perception (with abstention)
```

> ⚠️ **Simulated demonstration — not a clinical trial.** The child is an LLM; **no real
> children**; **no efficacy claims**. “Hand-back-to-human” is a **simulated pause**, not a real handoff.

## What you need

- Python 3.10+
- An **OpenAI API key** (the demo uses *your* key for the LLM child agent — you pay only for your own usage; the key stays on your machine).
- The **FERAC test images** under `data/ferac/test/{Natural,anger,fear,joy}/` (not redistributed — see [`../data/README.md`](../data/README.md)). Without them, perception gracefully degrades to `abstain`.
- The release model checkpoints in `../models/` (shipped in this repo).

## Setup

```bash
# from the repository root
cp app/.env.example app/.env        # then edit app/.env and paste your key
pip install -r requirements.txt
python app/backend/server.py        # starts FastAPI on http://127.0.0.1:8000
```

Then open **http://127.0.0.1:8000** in your browser (the backend serves the frontend).

## Using it

1. Pick a **persona** (P1 high / P2 mid / P3 low engagement) and click **Start session**.
2. Click **Step** (or **Auto**) to advance the loop. Each step shows:
   - **Child (LLM)** — engagement, displayed/raw emotion, fatigue *(simulated)*.
   - **Perception (Exp2, real)** — accept/abstain, confidence, 4-class probabilities.
   - **9-D observation** — with `REAL`/`SIM` tags per dimension.
   - **Robot decision & safety** — SAC's action, the R1–R4 shield verdict, executed action.
   - **Interpretability** — the deterministic 5-term reward decomposition.
3. If the system hits **persistently unreliable perception + perceived distress**, it triggers a
   **simulated hand-back-to-human**: a modal pauses the loop and surfaces a decision point. You
   role-play the human expert — apply oversight and continue, or end the session. *(No real human takes over.)*
4. At the end, a **session summary** reports accepts/abstains, interceptions, R3 triggers, hand-back, mean engagement.

## Key handling (security)

- Only `app/.env.example` (a placeholder) is committed. Your real `app/.env` is **git-ignored**.
- `server.py` reads `OPENAI_API_KEY` from `app/.env` (or repo-root `.env`). The key is **never** sent to the frontend or logged.

## Notes / honest limitations

- The SAC policy was trained on *synthetic* perception signals; here it receives *real* Exp2
  outputs (different distribution). The demo shows the **wiring works**, not that the policy is
  optimal under the real-perception distribution (closing that loop with in-the-loop retraining is future work).
- Exp2 perception is imperfect (misclassifies / abstains under the strict 0.86 gate) — shown honestly.
- Face detection is bypassed (`face_required=False`) because FERAC images are already cropped faces (the OpenCV Haar detector is a documented limitation).
