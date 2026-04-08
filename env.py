"""
EduNexora AI OpenEnv RL Environment
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
    "unit_1": {"name": "Algebra", "topics": {"t1_1": {"name": "Variables", "completed": True}, "t1_2": {"name": "Equations", "completed": False}}, "priority": 2},
    "unit_2": {"name": "Geometry", "topics": {"t2_1": {"name": "Triangles", "completed": True}}, "priority": 4},
}

def _compute_progress(syllabus: Dict) -> float:
    total = sum(len(u["topics"]) for u in syllabus.values())
    done  = sum(1 for u in syllabus.values() for t in u["topics"].values() if t["completed"])
    return round((done / total) * 100, 1) if total else 0.0

def _classify_student(marks: float) -> str:
    if marks >= 40: return "pass"
    elif marks >= 35: return "fail"
    return "backlog"

def _classify_risk(marks: float) -> str:
    if marks < 40: return "high"
    elif marks <= 60: return "medium"
    return "low"

def _assign_intervention(risk: str) -> str:
    mapping = {"high": "Immediate counseling", "medium": "Weekly mentoring", "low": "Regular monitoring"}
    return mapping.get(risk, "Standard support")

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
            self._state_data = {"students": copy.deepcopy(DUMMY_STUDENTS), "classifications": {}, "ranking": []}
        elif self.task == "syllabus_tracking":
            self._state_data = {"syllabus": copy.deepcopy(DUMMY_SYLLABUS), "prioritized": None, "notifications": [], "completed_steps": 0}
        elif self.task == "early_intervention":
            self._state_data = {"students": copy.deepcopy(DUMMY_STUDENTS), "risk_levels": {}, "interventions": {}}
            
        return Observation(task=self.task, data={"status": "reset"})

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done: raise RuntimeError("Episode is done. Call reset() first.")
        self._step_count += 1
        reward_value = 0.0005 
        info: Dict[str, Any] = {}

        if self.task == "student_analysis":
            if action.name == "classify_student":
                sid = action.params.get("student_id")
                student = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    expected = _classify_student(student["marks"])
                    reward_value = 0.008 if action.params.get("classification") == expected else 0.0005
                    self._state_data["classifications"][sid] = action.params.get("classification")
            elif action.name == "generate_ranking":
                self._state_data["ranking"] = [s["id"] for s in sorted(self._state_data["students"], key=lambda s: s["marks"], reverse=True)]
                reward_value = 0.05
            
            if action.name == "generate_ranking" and len(self._state_data["classifications"]) >= len(self._state_data["students"]):
                self._done = True

        elif self.task == "syllabus_tracking":
            if action.name == "prioritize_unit": reward_value = 0.04
            elif action.name == "mark_topic_complete": reward_value = 0.04
            elif action.name == "generate_notification": 
                reward_value = 0.04
                self._done = True
            if self._step_count >= 20: self._done = True

        elif self.task == "early_intervention":
            if action.name == "classify_risk":
                sid = action.params.get("student_id")
                student = next((s for s in self._state_data["students"] if s["id"] == sid), None)
                if student:
                    expected = _classify_risk(student["marks"])
                    reward_value = 0.004 if action.params.get("risk_level") == expected else 0.0005
                    self._state_data["risk_levels"][sid] = action.params.get("risk_level")
            elif action.name == "assign_intervention":
                sid = action.params.get("student_id")
                if sid:
                    self._state_data["interventions"][sid] = "Assigned"
                    reward_value = 0.004
            
            if len(self._state_data["risk_levels"]) >= len(self._state_data["students"]) and len(self._state_data["interventions"]) >= len(self._state_data["students"]):
                self._done = True

        reward = Reward(value=round(reward_value, 4), task=self.task, step=self._step_count)
        self._rewards.append(reward.value)
        return Observation(task=self.task, data=self.state()), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        base = {"task": self.task, "step": self._step_count, "done": self._done, "rewards": list(self._rewards)}
        base.update(self._state_data)
        return base

    @property
    def cumulative_reward(self) -> float:
        return round(max(0.001, min(0.999, sum(self._rewards))), 4)

    @property
    def step_count(self) -> int: return self._step_count
    def is_done(self) -> bool: return self._done
