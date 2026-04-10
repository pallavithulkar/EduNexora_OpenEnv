"""
EduNexora AI — tasks.py
total_reward = AVERAGE of step rewards (always between 0.01 and 0.99)
SUM of printed step rewards also < 1.0 (divided equally)
"""
from typing import Any, Dict, List, Optional
from env import (
    EduNexoraEnv, _classify_student, _classify_risk,
    _assign_intervention, _compute_progress,
    DUMMY_STUDENTS, DUMMY_SYLLABUS, _c
)
from models import Action, TaskResult


def _avg(rewards: list) -> float:
    if not rewards:
        return _c(0.50)
    return _c(sum(rewards) / len(rewards))


# ── Task 1 ────────────────────────────────────────────────────────────
def run_task1(students: Optional[List[Dict]] = None) -> TaskResult:
    env = EduNexoraEnv(task="student_analysis")
    src = students if students else DUMMY_STUDENTS
    steps, rewards = 0, []

    details: Dict[str, Any] = {
        "classifications": {},
        "ranking": [],
        "top_5": [],
        "backlog_students": [],
        "summary": {}
    }

    for s in src:
        label = _classify_student(s["marks"])
        _, rw, _, _ = env.step(Action(
            name="classify_student",
            params={"student_id": s["id"], "classification": label}
        ))
        rewards.append(rw.value)
        steps += 1
        details["classifications"][s["id"]] = {
            "name": s.get("name", s["id"]),
            "marks": s["marks"],
            "classification": label
        }
        if label == "backlog":
            details["backlog_students"].append({"id": s["id"], "name": s.get("name",""), "marks": s["marks"]})

    _, rw, _, info = env.step(Action(name="generate_ranking", params={}))
    rewards.append(rw.value)
    steps += 1

    sorted_s = sorted(src, key=lambda s: s["marks"], reverse=True)
    details["ranking"] = [{"id": s["id"], "marks": s["marks"]} for s in sorted_s]
    details["top_5"]   = details["ranking"][:5]
    details["summary"] = {
        "total_students": len(src),
        "pass":    sum(1 for v in details["classifications"].values() if v["classification"] == "pass"),
        "fail":    sum(1 for v in details["classifications"].values() if v["classification"] == "fail"),
        "backlog": len(details["backlog_students"])
    }

    return TaskResult(
        task="student_analysis", success=True,
        total_steps=steps, total_reward=_avg(rewards),
        rewards=rewards, details=details
    )


# ── Task 2 ────────────────────────────────────────────────────────────
def run_task2(syllabus: Optional[Dict] = None) -> TaskResult:
    import copy
    env = EduNexoraEnv(task="syllabus_tracking")
    src = copy.deepcopy(syllabus) if syllabus else copy.deepcopy(DUMMY_SYLLABUS)

    details: Dict[str, Any] = {"unit_status": {}, "summary": {}, "notifications": []}

    for uid, ud in src.items():
        total = len(ud["topics"])
        done  = sum(1 for t in ud["topics"].values() if t.get("completed", False))
        details["unit_status"][uid] = {"progress": round((done/total)*100, 2) if total else 0.0}

    progress = _compute_progress(src)
    details["summary"]["progress_percent"] = progress

    _, rw, _, _ = env.step(Action(name="generate_notification", params={}))

    return TaskResult(
        task="syllabus_tracking", success=True,
        total_steps=1, total_reward=_c(rw.value),
        rewards=[_c(rw.value)], details=details
    )


# ── Task 3 ────────────────────────────────────────────────────────────
def run_task3(students: Optional[List[Dict]] = None) -> TaskResult:
    env = EduNexoraEnv(task="early_intervention")
    src = students if students else DUMMY_STUDENTS
    steps, rewards = 0, []

    details = {"high": 0, "medium": 0, "low": 0}

    for s in src:
        risk = _classify_risk(s["marks"])
        details[risk] += 1
        _, rw, _, _ = env.step(Action(
            name="classify_risk",
            params={"student_id": s["id"], "risk_level": risk}
        ))
        rewards.append(rw.value)
        steps += 1

    return TaskResult(
        task="early_intervention", success=True,
        total_steps=steps, total_reward=_avg(rewards),
        rewards=rewards, details=details
    )


def run_all_tasks(students=None, syllabus=None):
    return {
        "task1": run_task1(students),
        "task2": run_task2(syllabus),
        "task3": run_task3(students),
    }
