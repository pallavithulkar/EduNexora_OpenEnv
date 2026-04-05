"""
EduNexora AI OpenEnv RL Environment
Implements reset(), step(action), state() for 3 educational tasks.
"""

import random
from typing import Any, Dict, Optional, Tuple
from models import Observation, Action, Reward, StudentData, SyllabusData, RiskData


import random

DUMMY_STUDENTS = [
    {
        "id": f"S{str(i).zfill(3)}",
        "name": f"Student {i}",
        "marks": random.randint(20, 95),
        "subjects": {
            "Math": random.randint(20, 95),
            "Science": random.randint(20, 95),
            "English": random.randint(20, 95),
            "History": random.randint(20, 95),
        }
    }
    for i in range(1, 101)
]

DUMMY_SYLLABUS = {
    "unit_1": {
        "name": "Fundamentals of Algebra",
        "topics": {
            "t1_1": {"name": "Variables and Expressions",  "completed": True},
            "t1_2": {"name": "Linear Equations",           "completed": True},
            "t1_3": {"name": "Quadratic Equations",        "completed": False},
            "t1_4": {"name": "Polynomial Functions",       "completed": False},
        },
        "priority": 2,
    },
    "unit_2": {
        "name": "Geometry and Trigonometry",
        "topics": {
            "t2_1": {"name": "Triangles and Congruence",   "completed": True},
            "t2_2": {"name": "Circles and Arcs",           "completed": True},
            "t2_3": {"name": "Trigonometric Ratios",       "completed": True},
            "t2_4": {"name": "3D Geometry",                "completed": True},
        },
        "priority": 4,
    },
    "unit_3": {
        "name": "Statistics and Probability",
        "topics": {
            "t3_1": {"name": "Descriptive Statistics",     "completed": True},
            "t3_2": {"name": "Probability Basics",         "completed": False},
            "t3_3": {"name": "Distributions",              "completed": False},
            "t3_4": {"name": "Hypothesis Testing",         "completed": False},
        },
        "priority": 1,
    },
    "unit_4": {
        "name": "Calculus Foundations",
        "topics": {
            "t4_1": {"name": "Limits and Continuity",      "completed": True},
            "t4_2": {"name": "Derivatives",                "completed": False},
            "t4_3": {"name": "Integrals",                  "completed": False},
            "t4_4": {"name": "Applications of Calculus",   "completed": False},
        },
        "priority": 3,
    },
}


def _compute_progress(syllabus: Dict) -> float:
    total = sum(len(u["topics"]) for u in syllabus.values())
    done  = sum(1 for u in syllabus.values() for t in u["topics"].values() if t["completed"])
    return round((done / total) * 100, 1) if total else 0.0


def _classify_student(marks: float) -> str:
    if marks >= 40:
        return "pass"
    elif marks >= 35:
        return "fail"
    else:
        return "backlog"


def _classify_risk(marks: float) -> str:
    if marks < 40:
        return "high"
    elif marks <= 60:
        return "medium"
    else:
        return "low"


def _assign_intervention(risk: str) -> str:
    mapping = {
        "high":   "Immediate counseling + remedial classes + parent meeting",
        "medium": "Weekly mentoring sessions + additional practice material",
        "low":    "Regular monitoring + optional enrichment classes",
    }
    return mapping.get(risk, "Standard support")


class EduNexoraEnv:
    """
    OpenEnv-compliant RL environment for EduNexora AI.
    Handles three distinct educational tasks.
    """

    TASKS = ["student_analysis", "syllabus_tracking", "early_intervention"]

    def __init__(self, task: str = "student_analysis"):
        assert task in self.TASKS, f"Unknown task: {task}. Choose from {self.TASKS}"
        self.task         = task
        self._step_count  = 0
        self._done        = False
        self._state_data: Dict[str, Any] = {}
        self._rewards: list  = []
        self.reset()

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        self._step_count = 0
        self._done       = False
        self._rewards    = []

        if self.task == "student_analysis":
            self._state_data = {
                "students": [dict(s) for s in DUMMY_STUDENTS],
                "classifications": {},
                "ranking": [],
            }
            obs_payload = {
                "students": self._state_data["students"],
                "task": self.task,
            }

        elif self.task == "syllabus_tracking":
            import copy
            self._state_data = {
                "syllabus":       copy.deepcopy(DUMMY_SYLLABUS),
                "prioritized":    None,
                "notifications":  [],
                "completed_steps": 0,
            }
            obs_payload = {
                "syllabus":  self._state_data["syllabus"],
                "progress":  _compute_progress(self._state_data["syllabus"]),
                "task":      self.task,
            }

        elif self.task == "early_intervention":
            self._state_data = {
                "students":      [dict(s) for s in DUMMY_STUDENTS],
                "risk_levels":   {},
                "interventions": {},
            }
            obs_payload = {
                "students": self._state_data["students"],
                "task":     self.task,
            }

        return Observation(task=self.task, data=obs_payload)

    # ------------------------------------------------------------------ #
    #  step(action)                                                        #
    # ------------------------------------------------------------------ #
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1
        reward_value = 0.0
        info: Dict[str, Any] = {}

        # ── TASK 1: Student Analysis ──────────────────────────────────── #
        if self.task == "student_analysis":
            if action.name == "classify_student":
                sid         = action.params.get("student_id")
                predicted   = action.params.get("classification")
                student     = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    expected = _classify_student(student["marks"])
                    if predicted == expected:
                        reward_value = 1.0
                        info["result"] = f"Correct: {sid} → {expected}"
                    else:
                        reward_value = 0.0
                        info["result"] = f"Wrong: {sid} predicted {predicted}, expected {expected}"
                    self._state_data["classifications"][sid] = predicted
                else:
                    reward_value = 0.0
                    info["error"] = f"Student {sid} not found"

            elif action.name == "generate_ranking":
                sorted_students = sorted(
                    self._state_data["students"], key=lambda s: s["marks"], reverse=True
                )
                self._state_data["ranking"] = [s["id"] for s in sorted_students]
                reward_value = 0.5
                info["ranking"] = self._state_data["ranking"]

            # Done only when ranking has also been generated
            if (action.name == "generate_ranking" and
                    len(self._state_data["classifications"]) >= len(self._state_data["students"])):
                self._done = True

        # ── TASK 2: Syllabus Tracking ─────────────────────────────────── #
        elif self.task == "syllabus_tracking":
            if action.name == "prioritize_unit":
                unit_id      = action.params.get("unit_id")
                syllabus     = self._state_data["syllabus"]
                # Correct priority = unit with fewest completed topics (most lagging)
                lag_scores   = {}
                for uid, udata in syllabus.items():
                    topics   = udata["topics"]
                    pending  = sum(1 for t in topics.values() if not t["completed"])
                    lag_scores[uid] = pending
                correct_unit = max(lag_scores, key=lag_scores.get)

                if unit_id == correct_unit:
                    reward_value = 0.5
                    info["result"] = f"Correct prioritization: {unit_id}"
                else:
                    reward_value = 0.2
                    info["result"] = f"Sub-optimal: {unit_id}, best was {correct_unit}"
                self._state_data["prioritized"] = unit_id

            elif action.name == "mark_topic_complete":
                topic_id = action.params.get("topic_id")
                found    = False
                for udata in self._state_data["syllabus"].values():
                    if topic_id in udata["topics"]:
                        udata["topics"][topic_id]["completed"] = True
                        found = True
                        self._state_data["completed_steps"] += 1
                progress = _compute_progress(self._state_data["syllabus"])
                if found:
                    reward_value = 1.0 if progress == 100.0 else round(progress / 100, 2)
                    info["progress"] = progress
                else:
                    reward_value = 0.0
                    info["error"] = f"Topic {topic_id} not found"

            elif action.name == "generate_notification":
                progress = _compute_progress(self._state_data["syllabus"])
                notes = []
                for uid, udata in self._state_data["syllabus"].items():
                    pending = sum(1 for t in udata["topics"].values() if not t["completed"])
                    total   = len(udata["topics"])
                    pct     = ((total - pending) / total) * 100 if total else 100
                    if pct < 50:
                        notes.append(f"{udata['name']} is lagging ({pct:.0f}% complete)")
                if progress < 50:
                    notes.append(f"Overall progress {progress:.0f}% — Extra class required")
                else:
                    notes.append(f"Progress {progress:.0f}%")
                self._state_data["notifications"].extend(notes)
                reward_value = 0.5
                info["notifications"] = notes

            progress = _compute_progress(self._state_data["syllabus"])
            if (progress >= 100.0 and action.name == "generate_notification") or self._step_count >= 20:
                self._done = True

        # ── TASK 3: Early Intervention ────────────────────────────────── #
        elif self.task == "early_intervention":
            if action.name == "classify_risk":
                sid       = action.params.get("student_id")
                predicted = action.params.get("risk_level")
                student   = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    expected = _classify_risk(student["marks"])
                    if predicted == expected:
                        reward_value = 1.0
                        info["result"] = f"Correct risk: {sid} → {expected}"
                    else:
                        reward_value = 0.0
                        info["result"] = f"Wrong risk: {sid} predicted {predicted}, expected {expected}"
                    self._state_data["risk_levels"][sid] = predicted
                else:
                    reward_value = 0.0
                    info["error"] = f"Student {sid} not found"

            elif action.name == "assign_intervention":
                sid      = action.params.get("student_id")
                student  = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    risk     = _classify_risk(student["marks"])
                    assigned = _assign_intervention(risk)
                    self._state_data["interventions"][sid] = assigned
                    reward_value = 0.5
                    info["intervention"] = assigned
                else:
                    reward_value = 0.0
                    info["error"] = f"Student {sid} not found"

            classified  = len(self._state_data["risk_levels"])
            intervened  = len(self._state_data["interventions"])
            total       = len(self._state_data["students"])
            if classified >= total and intervened >= total:
                self._done = True

        # ── Shared ────────────────────────────────────────────────────── #
        reward = Reward(
            value=round(reward_value, 4),
            task=self.task,
            step=self._step_count,
        )
        self._rewards.append(reward.value)

        next_obs = Observation(task=self.task, data=self.state())
        return next_obs, reward, self._done, info

    # ------------------------------------------------------------------ #
    #  state()                                                             #
    # ------------------------------------------------------------------ #
    def state(self) -> Dict[str, Any]:
        base = {
            "task":       self.task,
            "step":       self._step_count,
            "done":       self._done,
            "rewards":    list(self._rewards),
        }
        base.update(self._state_data)
        return base

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #
    @property
    def cumulative_reward(self) -> float:
        return round(sum(self._rewards), 4)

    @property
    def step_count(self) -> int:
        return self._step_count

    def is_done(self) -> bool:
        return self._done