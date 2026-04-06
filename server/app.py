from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os

# Add parent directory to path so we can import your existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import EduNexoraEnv

# --- Models ---

class EduAction(BaseModel):
    action_type: str          # "classify_student", "generate_ranking",
                              # "prioritize_unit", "mark_topic_complete",
                              # "classify_risk", "assign_intervention"
    task_id: Optional[str] = None
    student_id: Optional[str] = None
    classification: Optional[str] = None   # "pass" | "fail" | "backlog"
    risk_level: Optional[str] = None       # "high" | "medium" | "low"
    unit_id: Optional[str] = None
    topic_id: Optional[str] = None

class EduObservation(BaseModel):
    observation: dict
    reward: float = 0.0
    done: bool = False
    info: dict = {}

# --- App ---

app = FastAPI(title="EduNexora OpenEnv")
env = EduNexoraEnv()

@app.post("/reset")
def reset(body: dict = {}):
    obs = env.reset()
    return EduObservation(observation=obs if isinstance(obs, dict) else {"state": str(obs)})

@app.post("/step")
def step(action: EduAction):
    result = env.step(action.dict())
    if isinstance(result, tuple):
        obs, reward, done, info = result
    else:
        obs, reward, done, info = result, 0.0, False, {}
    return EduObservation(
        observation=obs if isinstance(obs, dict) else {"state": str(obs)},
        reward=float(reward),
        done=bool(done),
        info=info or {}
    )

@app.get("/state")
def state():
    s = env.state() if hasattr(env, "state") else {}
    return {"state": s}

@app.get("/health")
def health():
    return {"status": "healthy"}

# --- Required by openenv validate ---

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
