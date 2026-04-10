"""
EduNexora AI — models.py
All rewards clamped: round(max(0.01, min(0.99, v)), 4)
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


def clamp(v: float) -> float:
    return round(max(0.01, min(0.99, float(v))), 4)


class Observation(BaseModel):
    task: str
    data: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        arbitrary_types_allowed = True


class Action(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        arbitrary_types_allowed = True


class Reward(BaseModel):
    value: float
    task: str
    step: int

    @validator("value", pre=True, always=True)
    def clamp_reward(cls, v):
        return clamp(float(v))


class TaskResult(BaseModel):
    task: str
    success: bool
    total_steps: int
    total_reward: float
    rewards: List[float]
    details: Dict[str, Any] = Field(default_factory=dict)

    @validator("total_reward", pre=True, always=True)
    def clamp_total(cls, v):
        return clamp(float(v))


class StudentData(BaseModel):
    id: str
    name: str
    marks: float = Field(..., ge=0.0, le=100.0)
    subjects: Dict[str, float] = Field(default_factory=dict)
    classification: Optional[str] = None
    risk_level: Optional[str] = None
    rank: Optional[int] = None

    @validator("classification")
    def valid_cls(cls, v):
        if v and v not in ("pass", "fail", "backlog"):
            raise ValueError("must be pass/fail/backlog")
        return v

    @validator("risk_level")
    def valid_risk(cls, v):
        if v and v not in ("high", "medium", "low"):
            raise ValueError("must be high/medium/low")
        return v


class TopicData(BaseModel):
    id: str
    name: str
    completed: bool = False


class UnitData(BaseModel):
    id: str
    name: str
    topics: List[TopicData] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=4)


class TaskResultSimple(BaseModel):
    task: str
    score: float
    reward: float
    done: bool
    info: dict = {}

    @validator("score", "reward", pre=True, always=True)
    def clamp_fields(cls, v):
        return clamp(float(v))
