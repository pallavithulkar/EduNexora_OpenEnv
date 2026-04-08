import random
import copy
from typing import Any, Dict, Optional, Tuple
from models import Observation, Action, Reward
from graders import (grade_classification, grade_ranking, grade_prioritization,
                     grade_topic_completion, grade_notification, 
                     grade_risk_classification, grade_intervention, _clip)

# --- Yeh functions tasks.py ke liye zaroori hain ---
def _classify_student(marks: float) -> str:
    if marks >= 40: return "pass"
    elif marks >= 35: return "fail"
    else: return "backlog"

def _classify_risk(marks: float) -> str:
    if marks < 40: return "high"
    elif marks <= 60: return "medium"
    else: return "low"

def _assign_intervention(risk: str) -> str:
    mapping = {
        "high": "Immediate counseling + remedial classes",
        "medium": "Weekly mentoring sessions",
        "low": "Regular monitoring",
    }
    return mapping.get(risk, "Standard support")

def _compute_progress(syllabus: Dict) -> float:
    if not syllabus: return 0.0
    total = sum(len(u["topics"]) for u in syllabus.values())
    done  = sum(1 for u in syllabus.values() for t in u["topics"].values() if t["completed"])
    return round((done / total) * 100, 1) if total else 0.0

# --- Dummy Data ---
DUMMY_STUDENTS = [{"id": f"S{str(i).zfill(3)}", "marks": random.randint(20, 95)} for i in range(1, 101)]
DUMMY_SYLLABUS = {
    "unit_1": {
        "name": "Fundamentals",
        "topics": {"t1": {"name": "Topic 1", "completed": False}}
    }
}

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
        self._state_data = {"students": DUMMY_STUDENTS, "syllabus": copy.deepcopy(DUMMY_SYLLABUS)}
        return Observation(task=self.task, data={"status": "reset"})

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        self._step_count += 1
        reward_value = 0.01 
        info: Dict[str, Any] = {}

        if self.task == "student_analysis":
            if action.name == "classify_student":
                sid = action.params.get("student_id")
                predicted = action.params.get("classification")
                student = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    reward_value = grade_classification(predicted, student["marks"])
                else:
                    reward_value = 0.01

            elif action.name == "generate_ranking":
                predicted_order = action.params.get("predicted_order", [])
                actual_marks = {s["id"]: s["marks"] for s in self._state_data["students"]}
                reward_value = grade_ranking(predicted_order, actual_marks)
                self._done = True

        elif self.task == "syllabus_tracking":
            if action.name == "prioritize_unit":
                reward_value = grade_prioritization(action.params.get("unit_id"), {"unit_1": 1})
            elif action.name == "mark_topic_complete":
                reward_value = grade_topic_completion(50.0)
            elif action.name == "generate_notification":
                reward_value = grade_notification(action.params.get("notifications", []))
                self._done = True
            if self._step_count >= 20: self._done = True

        elif self.task == "early_intervention":
            if action.name == "classify_risk":
                sid = action.params.get("student_id")
                predicted = action.params.get("risk_level")
                student = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    reward_value = grade_risk_classification(predicted, student["marks"])
                else:
                    reward_value = 0.01
            elif action.name == "assign_intervention": 
                reward_value = grade_intervention("high", action.params.get("intervention", ""))
            
            if self._step_count >= 100: self._done = True

        reward = Reward(value=_clip(reward_value), task=self.task, step=self._step_count)
        self._rewards.append(reward.value)
        return Observation(task=self.task, data={"status": "running"}), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        return {"task": self.task, "step": self._step_count, "done": self._done}

    @property
    def cumulative_reward(self) -> float:
        total = sum(self._rewards) / len(self._rewards) if self._rewards else 0.01
        return _clip(total)
