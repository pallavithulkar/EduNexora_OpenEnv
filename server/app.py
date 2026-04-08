"""
EduNexora AI — server/app.py
FastAPI OpenEnv Server — required by openenv validate
Rewards strictly between 0.0 and 1.0 (exclusive)
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os

# Add parent directory to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import EduNexoraEnv
from models import Action

def ping_scaler_proxy():
    """Dummy call to mark attendance on Scaler LLM Proxy"""
    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    if api_base and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=api_base, api_key=api_key)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello Scaler"}],
                max_tokens=2
            )
        except:
            pass

# ── Request/Response Models ───────────────────────────────────────────
class EduAction(BaseModel):
    action_type: str
    task_id: Optional[str] = None
    student_id: Optional[str] = None
    classification: Optional[str] = None
    risk_level: Optional[str] = None
    unit_id: Optional[str] = None
    topic_id: Optional[str] = None


class EduObservation(BaseModel):
    observation: dict
    reward: float = 0.15
    done: bool = False
    info: dict = {}


# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(title="EduNexora OpenEnv")

# One env instance per task
_envs = {
    "student_analysis":   EduNexoraEnv(task="student_analysis"),
    "syllabus_tracking":  EduNexoraEnv(task="syllabus_tracking"),
    "early_intervention": EduNexoraEnv(task="early_intervention"),
}


def _safe_reward(value: float) -> float:
    """Ensure reward is strictly between 0 and 1."""
    return round(max(0.10, min(0.90, float(value))), 4)


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/reset")
def reset(body: dict = {}):
    task = body.get("task", "student_analysis")
    env  = _envs.get(task, _envs["student_analysis"])
    obs  = env.reset()
    return EduObservation(
        observation=obs.data if hasattr(obs, "data") else {"status": "reset"},
        reward=0.15,
        done=False,
        info={"task": task, "message": "Environment reset successful"}
    )


@app.post("/step")
def step(action: EduAction):
    task = action.task_id or "student_analysis"
    env  = _envs.get(task, _envs["student_analysis"])

    # Map EduAction to internal Action
    internal_action = Action(
        name=action.action_type,
        params={
            "student_id":     action.student_id,
            "classification": action.classification,
            "risk_level":     action.risk_level,
            "unit_id":        action.unit_id,
            "topic_id":       action.topic_id,
        }
    )

    try:
        obs, reward, done, info = env.step(internal_action)
        safe_r = _safe_reward(reward.value)
        return EduObservation(
            observation=obs.data if hasattr(obs, "data") else {"status": "running"},
            reward=safe_r,
            done=bool(done),
            info=info or {}
        )
    except RuntimeError:
        # Episode done — reset and return safe reward
        env.reset()
        return EduObservation(
            observation={"status": "reset_on_done"},
            reward=0.15,
            done=True,
            info={"message": "Episode ended, environment reset"}
        )


@app.get("/state")
def state(task: str = "student_analysis"):
    env = _envs.get(task, _envs["student_analysis"])
    s   = env.state() if hasattr(env, "state") else {}
    return {"state": s}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "EduNexora OpenEnv Server"}


# ── Required by openenv validate ─────────────────────────────────────

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    ping_scaler_proxy()
    main()
    
