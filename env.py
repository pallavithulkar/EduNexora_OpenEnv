"""
EduNexora AI OpenEnv RL Environment
All rewards strictly between 0.0 and 1.0 (exclusive)
"""
import random
from typing import Any, Dict, Optional, Tuple
from models import Observation, Action, Reward

DUMMY_STUDENTS = [
    {
        "id": f"S{str(i).zfill(3)}",
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
            "t1_1": {"name": "Variables", "completed": True},
            "t1_2": {"name": "Equations", "completed": False}
        },
        "priority": 2
    },
    "unit_2": {
        "name": "Geometry",
        "topics": {
            "t2_1": {"name": "Triangles", "completed": True},
            "t2_2": {"name": "Circles", "completed": False}
        },
        "priority": 4
    },
    "unit_3": {
        "name": "Statistics",
        "topics": {
            "t3_1": {"name": "Mean", "completed": True},
            "t3_2": {"name": "Probability", "completed": False}
        },
        "priority": 1
    },
    "unit_4": {
        "name": "Calculus",
        "topics": {
            "t4_1": {"name": "Limits", "completed": True},
            "t4_2": {"name": "Derivatives", "completed": False}
        },
        "priority": 3
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
    return "backlog"


def _classify_risk(marks: float) -> str:
    if marks < 40:
        return "high"
    elif marks <= 60:
        return "medium"
    return "low"


def _assign_intervention(risk: str) -> str:
    mapping = {
        "high":   "Immediate counseling + remedial classes",
        "medium": "Weekly mentoring sessions",
        "low":    "Regular monitoring"
    }
    return mapping.get(risk, "Standard support")


def _safe_reward(value: float) -> float:
    """Ensure reward is strictly between 0 and 1."""
    return round(max(0.10, min(0.90, value)), 4)


class EduNexoraEnv:
    TASKS = ["student_analysis", "syllabus_tracking", "early_intervention"]

    def __init__(self, task: str = "student_analysis"):
        self.task = task
        self._step_count = 0
        self._done = False
        self._state_data: Dict[str, Any] = {}
        self._rewards: list = []
        self.reset()

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        self._rewards = []

        import copy
        if self.task == "student_analysis":
            self._state_data = {
                "students": copy.deepcopy(DUMMY_STUDENTS),
                "classifications": {},
                "ranking": []
            }
        elif self.task == "syllabus_tracking":
            self._state_data = {
                "syllabus": copy.deepcopy(DUMMY_SYLLABUS),
                "prioritized": None,
                "notifications": [],
                "completed_steps": 0
            }
        elif self.task == "early_intervention":
            self._state_data = {
                "students": copy.deepcopy(DUMMY_STUDENTS),
                "risk_levels": {},
                "interventions": {}
            }

        return Observation(task=self.task, data={"status": "reset"})

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1
        reward_value = 0.50
        info: Dict[str, Any] = {}

        if self.task == "student_analysis":
            if action.name == "classify_student":
                sid = action.params.get("student_id")
                student = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    expected = _classify_student(student["marks"])
                    reward_value = 0.90 if action.params.get("classification") == expected else 0.10
                    self._state_data["classifications"][sid] = action.params.get("classification")
                else:
                    reward_value = 0.10

            elif action.name == "generate_ranking":
                self._state_data["ranking"] = [
                    s["id"] for s in sorted(self._state_data["students"], key=lambda s: s["marks"], reverse=True)
                ]
                reward_value = 0.80
                info["ranking"] = self._state_data["ranking"]

            if (action.name == "generate_ranking" and
                    len(self._state_data["classifications"]) >= len(self._state_data["students"])):
                self._done = True

        elif self.task == "syllabus_tracking":
            if action.name == "prioritize_unit":
                reward_value = 0.80
            elif action.name == "mark_topic_complete":
                topic_id = action.params.get("topic_id")
                found = False
                for udata in self._state_data["syllabus"].values():
                    if topic_id in udata["topics"]:
                        udata["topics"][topic_id]["completed"] = True
                        found = True
                progress = _compute_progress(self._state_data["syllabus"])
                reward_value = _safe_reward(progress / 100.0 * 0.90) if found else 0.10
                info["progress"] = progress
            elif action.name == "generate_notification":
                progress = _compute_progress(self._state_data["syllabus"])
                reward_value = _safe_reward(0.50 + progress / 200.0)
                self._done = True

            if self._step_count >= 20:
                self._done = True

        elif self.task == "early_intervention":
            if action.name == "classify_risk":
                sid = action.params.get("student_id")
                student = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    expected = _classify_risk(student["marks"])
                    reward_value = 0.90 if action.params.get("risk_level") == expected else 0.10
                    self._state_data["risk_levels"][sid] = action.params.get("risk_level")
                else:
                    reward_value = 0.10

            elif action.name == "assign_intervention":
                sid = action.params.get("student_id")
                if sid:
                    self._state_data["interventions"][sid] = "Assigned"
                    reward_value = 0.80

            classified = len(self._state_data["risk_levels"])
            intervened = len(self._state_data["interventions"])
            total = len(self._state_data["students"])
            if classified >= total and intervened >= total:
                self._done = True

        reward_value = _safe_reward(reward_value)
        reward = Reward(value=reward_value, task=self.task, step=self._step_count)
        self._rewards.append(reward.value)
        return Observation(task=self.task, data=self.state()), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        base = {
            "task": self.task,
            "step": self._step_count,
            "done": self._done,
            "rewards": list(self._rewards)
        }
        base.update(self._state_data)
        return base

    @property
    def cumulative_reward(self) -> float:
        return round(max(0.10, min(0.90, sum(self._rewards))), 4)

    @property
    def step_count(self) -> int:
        return self._step_count

    def is_done(self) -> bool:
        return self._done
