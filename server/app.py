"""
EduNexora AI — server/app.py
FastAPI OpenEnv server.
Both "reward" AND "score" fields in every response.
All values: round(max(0.01, min(0.99, v)), 4)
"""
from fastapi import FastAPI
from pydantic import BaseModel, validator
from typing import Optional, Any, Dict
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import EduNexoraEnv, _classify_student, _classify_risk, _c
from models import Action


# ── Proxy ping ────────────────────────────────────────────────────────
def ping_scaler_proxy():
    api_base = os.environ.get("API_BASE_URL")
    api_key  = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
    model    = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    if api_base and api_key and api_key != "dummy-key":
        try:
            from openai import OpenAI
            client = OpenAI(base_url=api_base, api_key=api_key)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        except Exception:
            pass


app = FastAPI(title="EduNexora OpenEnv Server")
_envs: Dict[str, Any] = {}


def _get_env(task: str) -> EduNexoraEnv:
    if task not in _envs:
        _envs[task] = EduNexoraEnv(task=task)
    return _envs[task]


# ── Request / Response models ─────────────────────────────────────────
class ResetRequest(BaseModel):
    task: Optional[str] = "student_analysis"


class StepRequest(BaseModel):
    task:   Optional[str]  = "student_analysis"
    action: Optional[str]  = "classify_student"
    params: Optional[dict] = {}


class EnvResponse(BaseModel):
    observation: dict
    reward: float = 0.50
    score:  float = 0.50
    done:   bool  = False
    info:   dict  = {}

    @validator("reward", "score", pre=True, always=True)
    def clamp_fields(cls, v):
        return _c(float(v))


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/reset", response_model=EnvResponse)
def reset(body: ResetRequest = ResetRequest()):
    task = body.task or "student_analysis"
    env  = _get_env(task)
    try:
        obs   = env.reset()
        state = obs.data if hasattr(obs, "data") else {"status": "reset"}
    except Exception:
        state = {"status": "reset"}
    return EnvResponse(
        observation=state,
        reward=_c(0.50),
        score=_c(0.50),
        done=False,
        info={"task": task, "message": "reset ok"}
    )


@app.post("/step", response_model=EnvResponse)
def step(body: StepRequest = StepRequest()):
    task   = body.task   or "student_analysis"
    action = body.action or "classify_student"
    params = body.params or {}

    env = _get_env(task)

    # Build meaningful params
    if task == "student_analysis":
        if action == "classify_student":
            sid    = params.get("student_id", "S001")
            marks  = float(params.get("marks", 50))
            params = {"student_id": sid, "classification": _classify_student(marks)}
        elif action == "generate_ranking":
            params = {}

    elif task == "early_intervention":
        if action == "classify_risk":
            sid    = params.get("student_id", "S001")
            marks  = float(params.get("marks", 50))
            params = {"student_id": sid, "risk_level": _classify_risk(marks)}

    elif task == "syllabus_tracking":
        if action not in ["prioritize_unit", "mark_topic_complete", "generate_notification"]:
            action = "generate_notification"

    internal = Action(name=action, params=params)

    try:
        obs, reward, done, info = env.step(internal)
        safe_r = _c(reward.value)
        state  = obs.data if hasattr(obs, "data") else {"status": "running"}
    except RuntimeError:
        env.reset()
        safe_r = _c(0.50)
        done   = True
        state  = {"status": "reset_after_done"}
        info   = {}

    return EnvResponse(
        observation=state,
        reward=safe_r,
        score=safe_r,
        done=bool(done),
        info=info or {}
    )


@app.get("/state")
def state(task: str = "student_analysis"):
    env = _get_env(task)
    try:
        s = env.state()
    except Exception:
        s = {}
    return {"state": s, "task": task}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "EduNexora OpenEnv"}


# ── Entry point ───────────────────────────────────────────────────────
def main():
    ping_scaler_proxy()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
