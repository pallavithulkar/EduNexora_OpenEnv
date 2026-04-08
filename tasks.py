"""
EduNexora AI — Task Runners
(Fractional Rewards Math + Full Dashboard Details Restored)
"""
from typing import Any, Dict, List, Optional
from env import EduNexoraEnv, _classify_student, _classify_risk, _assign_intervention, _compute_progress, DUMMY_STUDENTS, DUMMY_SYLLABUS
from models import Action, TaskResult

def run_task1(students: Optional[List[Dict]] = None) -> TaskResult:
    env = EduNexoraEnv(task="student_analysis")
    source_students = students if students else DUMMY_STUDENTS
    steps = 0
    rewards = []
    
    # Dashboard ko ye details chahiye!
    details: Dict[str, Any] = {
        "classifications": {},
        "ranking": [],
        "top_5": [],
        "backlog_students": [],
        "summary": {}
    }
    
    for student in source_students:
        sid = student["id"]
        marks = student["marks"]
        label = _classify_student(marks)
        
        obs, reward, done, info = env.step(Action(name="classify_student", params={"student_id": sid, "classification": label}))
        rewards.append(reward.value)
        steps += 1
        
        details["classifications"][sid] = {"name": student.get("name", sid), "marks": marks, "classification": label}
        if label == "backlog":
            details["backlog_students"].append({"id": sid, "name": student.get("name", sid), "marks": marks})

    obs, reward, done, info = env.step(Action(name="generate_ranking", params={}))
    rewards.append(reward.value)
    steps += 1

    sorted_students = sorted(source_students, key=lambda s: s["marks"], reverse=True)
    ranking = [{"id": s["id"], "marks": s["marks"]} for s in sorted_students]
    details["ranking"] = ranking
    details["top_5"] = ranking[:5]

    details["summary"] = {
        "total_students": len(source_students),
        "pass": sum(1 for s in details["classifications"].values() if s["classification"] == "pass"),
        "fail": sum(1 for s in details["classifications"].values() if s["classification"] == "fail"),
        "backlog": len(details["backlog_students"])
    }

    # Safe Total Reward Clamp
    total_reward = round(min(0.999, max(0.001, sum(rewards))), 4)
    
    return TaskResult(task="student_analysis", success=True, total_steps=steps, total_reward=total_reward, rewards=rewards, details=details)

def run_task2(syllabus: Optional[Dict] = None) -> TaskResult:
    import copy
    env = EduNexoraEnv(task="syllabus_tracking")
    source_syllabus = copy.deepcopy(syllabus) if syllabus else copy.deepcopy(DUMMY_SYLLABUS)
    
    details: Dict[str, Any] = {"unit_status": {}, "summary": {}, "notifications": []}
    
    for uid, udata in source_syllabus.items():
        total = len(udata["topics"])
        done = sum(1 for t in udata["topics"].values() if t.get("completed", False))
        details["unit_status"][uid] = {"progress": round((done / total) * 100, 2) if total else 0.0}

    progress = _compute_progress(source_syllabus)
    details["summary"]["progress_percent"] = progress

    obs, reward, done, info = env.step(Action(name="generate_notification", params={}))
    total_reward = round(min(0.999, max(0.001, reward.value)), 4)
    
    return TaskResult(task="syllabus_tracking", success=True, total_steps=1, total_reward=total_reward, rewards=[reward.value], details=details)

def run_task3(students: Optional[List[Dict]] = None) -> TaskResult:
    env = EduNexoraEnv(task="early_intervention")
    source_students = students if students else DUMMY_STUDENTS
    steps = 0
    rewards = []
    
    details = {"high": 0, "medium": 0, "low": 0}
    
    for student in source_students:
        risk = _classify_risk(student["marks"])
        details[risk] += 1
        
        obs, reward, done, info = env.step(Action(name="classify_risk", params={"student_id": student["id"], "risk_level": risk}))
        rewards.append(reward.value)
        steps += 1

    total_reward = round(min(0.999, max(0.001, sum(rewards))), 4)
    return TaskResult(task="early_intervention", success=True, total_steps=steps, total_reward=total_reward, rewards=rewards, details=details)

def run_all_tasks(students: Optional[List[Dict]] = None, syllabus: Optional[Dict] = None) -> Dict[str, TaskResult]:
    return {"task1": run_task1(students), "task2": run_task2(syllabus), "task3": run_task3(students)}
