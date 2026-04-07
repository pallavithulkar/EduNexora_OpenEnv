"""
EduNexora AI — Pydantic Models
Observation, Action, Reward and domain-specific data models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


# ──────────────────────────────────────────────────────────────────────────── #
#  Core RL Models                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class Observation(BaseModel):
    """Represents the environment state returned to the agent."""
    task: str
    data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class Action(BaseModel):
    """Represents an action the agent can take."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class Reward(BaseModel):
    """Represents the reward signal from the environment."""
   value: float = Field(..., gt=0.0, lt=1.0)
    task: str
    step: int

  @validator("value")
  def clamp_reward(cls, v: float) -> float:
    return max(0.01, min(0.99, round(v, 4)))


# ──────────────────────────────────────────────────────────────────────────── #
#  Domain Models                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

class SubjectScore(BaseModel):
    subject: str
    score: float = Field(..., ge=0.0, le=100.0)


class StudentData(BaseModel):
    """A single student's academic record."""
    id: str
    name: str
    marks: float = Field(..., ge=0.0, le=100.0)
    subjects: Dict[str, float] = Field(default_factory=dict)
    classification: Optional[str] = None   # pass / fail / backlog
    risk_level: Optional[str] = None       # high / medium / low
    rank: Optional[int] = None

    @validator("classification")
    def valid_classification(cls, v):
        if v is not None and v not in ("pass", "fail", "backlog"):
            raise ValueError("classification must be pass, fail, or backlog")
        return v

    @validator("risk_level")
    def valid_risk(cls, v):
        if v is not None and v not in ("high", "medium", "low"):
            raise ValueError("risk_level must be high, medium, or low")
        return v


class TopicData(BaseModel):
    """A single topic inside a syllabus unit."""
    id: str
    name: str
    completed: bool = False


class UnitData(BaseModel):
    """A syllabus unit containing topics."""
    id: str
    name: str
    topics: List[TopicData] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=4)

    @property
    def progress_pct(self) -> float:
        if not self.topics:
            return 0.0
        done = sum(1 for t in self.topics if t.completed)
        return round((done / len(self.topics)) * 100, 1)


class SyllabusData(BaseModel):
    """Full syllabus structure."""
    units: List[UnitData] = Field(default_factory=list)

    @property
    def overall_progress(self) -> float:
        total = sum(len(u.topics) for u in self.units)
        done  = sum(1 for u in self.units for t in u.topics if t.completed)
        return round((done / total) * 100, 1) if total else 0.0


class InterventionRecord(BaseModel):
    """Intervention assignment for a student."""
    student_id: str
    risk_level: str
    intervention: str


class RiskData(BaseModel):
    """Risk assessment results."""
    student_id: str
    marks: float
    risk_level: str
    intervention: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────── #
#  API / Frontend Response Models                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class TaskResult(BaseModel):
    task: str
    success: bool
    total_steps: int
    total_reward: float
    rewards: List[float]
    details: Dict[str, Any] = Field(default_factory=dict)


class DashboardData(BaseModel):
    task1: Optional[TaskResult] = None
    task2: Optional[TaskResult] = None
    task3: Optional[TaskResult] = None
    source: str = "dummy"  # "dummy" or "uploaded"
