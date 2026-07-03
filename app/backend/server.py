# -*- coding: utf-8 -*-
"""
Backend for the runnable demo (FastAPI).

Run:
    cp .env.example .env          # then put your OpenAI key in .env
    pip install -r requirements.txt
    python app/backend/server.py
    # then open http://127.0.0.1:8000  (frontend is served from here)

Honest scope: SIMULATED module-integration demo. The child is an LLM (your key);
no real children; no efficacy claims; hand-back is a simulated pause.
"""
import os, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "app/backend"))

from dotenv import load_dotenv
# load .env from app/ first, then repo root
load_dotenv(ROOT / "app/.env")
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import Session

def _resolve_key():
    """Prefer the standard OPENAI_API_KEY; fall back to OPENAI_API_KEY_1/_2/... if present
    (convenience for repos whose .env uses numbered keys). Only sk-... OpenAI keys are used."""
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k.startswith("sk-"):
        return k
    cands = [(n, v) for n, v in os.environ.items()
             if n.upper().startswith("OPENAI_API_KEY") and str(v).strip().startswith("sk-")]
    cands.sort(key=lambda nv: nv[0])
    return cands[0][1].strip() if cands else k

API_KEY = _resolve_key()
MODEL = os.environ.get("OPENAI_CHILD_MODEL", "gpt-4o").strip() or "gpt-4o"
FRONTEND = ROOT / "app/frontend"

app = FastAPI(title="ASD Companion System — live demo")
SESSIONS = {}

class StartReq(BaseModel):
    persona: str = "P3"

class StepReq(BaseModel):
    session_id: str

@app.get("/api/health")
def health():
    return {"ok": True, "key_configured": bool(API_KEY), "model": MODEL}

@app.post("/api/start")
def start(req: StartReq):
    if not API_KEY:
        raise HTTPException(400, "No OPENAI_API_KEY configured. Copy .env.example to .env and add your key.")
    if req.persona not in ("P1", "P2", "P3"):
        raise HTTPException(400, "persona must be P1, P2 or P3")
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = Session(req.persona, API_KEY, model=MODEL)
    return {"session_id": sid, "persona": req.persona,
            "disclaimer": "SIMULATED demo — child is an LLM, no real children, no efficacy claims."}

@app.post("/api/step")
def step(req: StepReq):
    s = SESSIONS.get(req.session_id)
    if s is None:
        raise HTTPException(404, "unknown session_id (start a new session)")
    try:
        return s.step()
    except Exception as e:
        raise HTTPException(500, f"step failed: {type(e).__name__}: {e}")

@app.post("/api/summary")
def summary(req: StepReq):
    s = SESSIONS.get(req.session_id)
    if s is None:
        raise HTTPException(404, "unknown session_id")
    return s.summary()

# ---- static frontend ----
@app.get("/")
def index():
    return FileResponse(str(FRONTEND / "index.html"))

app.mount("/", StaticFiles(directory=str(FRONTEND)), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"  key configured: {bool(API_KEY)} | child model: {MODEL}")
    print("  open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
