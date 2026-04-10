"""
EduNexora AI — env.py
All rewards: round(max(0.01, min(0.99, v)), 4)
"""
import os
import random
import copy
from typing import Any, Dict, Tuple
from models import Observation, Action, Reward


# ── Proxy ping ────────────────────────────────────────────────────────
def ping_scaler_proxy():
    api_base = os.environ.get("API_BASE_URL")
    api_key  = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy-key")
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


# ── Clamp helper ──────────────────────────────────────────────────────
def _c(v: float) -> float:
    return round(max(0.01, min(0.99, float(v))), 4)


# ── Dummy data ────────────────────────────────────────────────────────
DUMMY_STUDENTS = [
    {
        "id":   f"S{str(i).zfill(3)}",
        "name": f"Student {i}",
        "marks": random.randint(20, 95),
        "subjects": {"Math": 80, "Science": 70, "English": 60, "History": 90}
    }
    for i in range(1, 101)
]

DUMMY_SYLLABUS = {
    "unit_1": {
        "name": "Algebra",
        "topics": {
            "t1_1": {"name": "Variables",  "completed": True},
            "t1_2": {"name": "Equations",  "completed": False},
        },
        "priority": 2,
    },
    "unit_2": {
        "name": "Geometry",
        "topics": {
            "t2_1": {"name": "Triangles",  "completed": True},
            "t2_2": {"name": "Circles",    "completed": False},
        },
        "priority": 4,
    },
    "unit_3": {
        "name": "Statistics",
        "topics": {
            "t3_1": {"name": "Mean",        "completed": True},
            "t3_2": {"name": "Probability", "completed": False},
        },
        "priority": 1,
    },
    "unit_4": {
        "name": "Calculus",
        "topics": {
            "t4_1": {"name": "Limits",      "completed": True},
            "t4_2": {"name": "Derivatives", "completed": False},
        },
        "priority": 3,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────
def _compute_progress(syllabus: Dict) -> float:
    total = sum(len(u["topics"]) for u in syllabus.values())
    done  = sum(1 for u in syllabus.values() for t in u["topics"].values() if t["completed"])
    return round((done / total) * 100, 1) if total else 0.0


def _classify_student(marks: float) -> str:
    if marks >= 40:   return "pass"
    elif marks >= 35: return "fail"
    return "backlog"


def _classify_risk(marks: float) -> str:
    if marks < 40:    return "high"
    elif marks <= 60: return "medium"
    return "low"


def _assign_intervention(risk: str) -> str:
    return {
        "high":   "Immediate counseling + remedial classes",
        "medium": "Weekly mentoring sessions",
        "low":    "Regular monitoring",
    }.get(risk, "Standard support")


# ── Environment ───────────────────────────────────────────────────────
class EduNexoraEnv:
    TASKS = ["student_analysis", "syllabus_tracking", "early_intervention"]

    def __init__(self, task: str = "student_analysis"):
        assert task in self.TASKS
        self.task        = task
        self._step_count = 0
        self._done       = False
        self._state_data: Dict[str, Any] = {}
        self._rewards: list = []
        ping_scaler_proxy()
        self.reset()

    # ── reset ─────────────────────────────────────────────────────────
    def reset(self) -> Observation:
        self._step_count = 0
        self._done       = False
        self._rewards    = []

        if self.task == "student_analysis":
            self._state_data = {
                "students":       copy.deepcopy(DUMMY_STUDENTS),
                "classifications": {},
                "ranking":        [],
            }
        elif self.task == "syllabus_tracking":
            self._state_data = {
                "syllabus":        copy.deepcopy(DUMMY_SYLLABUS),
                "prioritized":     None,
                "notifications":   [],
                "completed_steps": 0,
            }
        elif self.task == "early_intervention":
            self._state_data = {
                "students":      copy.deepcopy(DUMMY_STUDENTS),
                "risk_levels":   {},
                "interventions": {},
            }

        return Observation(task=self.task, data={"status": "reset"})

    # ── step ──────────────────────────────────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        self._step_count += 1
        rv   = 0.50
        info: Dict[str, Any] = {}

        # ── Task 1 ────────────────────────────────────────────────────
        if self.task == "student_analysis":
            if action.name == "classify_student":
                sid = action.params.get("student_id")
                st  = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if st:
                    expected = _classify_student(st["marks"])
                    rv = 0.85 if action.params.get("classification") == expected else 0.15
                    self._state_data["classifications"][sid] = action.params.get("classification")
                else:
                    rv = 0.15
            elif action.name == "generate_ranking":
                self._state_data["ranking"] = [
                    s["id"] for s in sorted(self._state_data["students"],
                                            key=lambda s: s["marks"], reverse=True)
                ]
                rv = 0.80
                info["ranking"] = self._state_data["ranking"]
                if len(self._state_data["classifications"]) >= len(self._state_data["students"]):
                    self._done = True

        # ── Task 2 ────────────────────────────────────────────────────
        elif self.task == "syllabus_tracking":
            if action.name == "prioritize_unit":
                rv = 0.80
            elif action.name == "mark_topic_complete":
                tid   = action.params.get("topic_id")
                found = False
                for ud in self._state_data["syllabus"].values():
                    if tid in ud["topics"]:
                        ud["topics"][tid]["completed"] = True
                        found = True
                progress = _compute_progress(self._state_data["syllabus"])
                rv = _c(progress / 100.0 * 0.85) if found else 0.15
                info["progress"] = progress
            elif action.name == "generate_notification":
                progress = _compute_progress(self._state_data["syllabus"])
                rv = _c(0.40 + progress / 200.0)
                self._done = True
            if self._step_count >= 20:
                self._done = True

        # ── Task 3 ────────────────────────────────────────────────────
        elif self.task == "early_intervention":
            if action.name == "classify_risk":
                sid = action.params.get("student_id")
                st  = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if st:
                    expected = _classify_risk(st["marks"])
                    rv = 0.85 if action.params.get("risk_level") == expected else 0.15
                    self._state_data["risk_levels"][sid] = action.params.get("risk_level")
                else:
                    rv = 0.15
            elif action.name == "assign_intervention":
                sid = action.params.get("student_id")
                if sid:
                    self._state_data["interventions"][sid] = _assign_intervention(
                        _classify_risk(
                            next((s["marks"] for s in self._state_data["students"] if s["id"] == sid), 50)
                        )
                    )
                    rv = 0.80
            c = len(self._state_data["risk_levels"])
            t = len(self._state_data["students"])
            if c >= t:
                self._done = True

        rv     = _c(rv)
        reward = Reward(value=rv, task=self.task, step=self._step_count)
        self._rewards.append(reward.value)
        return Observation(task=self.task, data=self.state()), reward, self._done, info

    # ── state ─────────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        base = {
            "task":    self.task,
            "step":    self._step_count,
            "done":    self._done,
            "rewards": list(self._rewards),
        }
        base.update(self._state_data)
        return base

    @property
    def cumulative_reward(self) -> float:
        if not self._rewards:
            return _c(0.50)
        return _c(sum(self._rewards) / len(self._rewards))  # AVERAGE not sum

    @property
    def step_count(self) -> int:
        return self._step_count

    def is_done(self) -> bool:
        return self._done
