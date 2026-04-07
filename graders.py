"""
EduNexora AI — Graders
Computes dynamic rewards for each task based on action correctness.
All rewards are strictly in (0.0, 1.0) — i.e. clamped to [0.01, 0.99].
"""

from typing import Any, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 1 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_classification(predicted: str, marks: float) -> float:
    if marks >= 40:
        expected = "pass"
    elif marks >= 35:
        expected = "fail"
    else:
        expected = "backlog"

    return 0.99 if predicted == expected else 0.01


def grade_ranking(predicted_order: list, actual_marks: Dict[str, float]) -> float:
    if not predicted_order:
        return 0.01  
    sorted_ids = sorted(actual_marks.keys(), key=lambda sid: actual_marks[sid], reverse=True)
    if predicted_order == sorted_ids:
        return 0.5
    top_n   = min(3, len(sorted_ids))
    correct = sum(1 for i in range(top_n) if i < len(predicted_order) and predicted_order[i] == sorted_ids[i])
    return round(max(0.01, (correct / top_n) * 0.5), 4) 


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 2 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_prioritization(selected_unit: str, lag_scores: Dict[str, int]) -> float:
    if not lag_scores:
        return 0.01 
    best = max(lag_scores, key=lag_scores.get)
    if selected_unit == best:
        return 0.5
    selected_lag = lag_scores.get(selected_unit, 0)
    best_lag     = lag_scores[best]
    if best_lag == 0:
        return 0.01 
    return round(max(0.2, (selected_lag / best_lag) * 0.5), 4)


def grade_topic_completion(progress_pct: float) -> float:
    return round(min(0.99, max(0.01, progress_pct / 100.0)), 4) 


def grade_notification(notifications: list) -> float:
    if not notifications:
        return 0.01 
    keywords = ["lagging", "progress", "class", "complete", "required", "%"]
    hits     = sum(1 for n in notifications for kw in keywords if kw.lower() in n.lower())
    score    = min(0.99, hits / max(len(notifications) * 2, 1)) 
    return round(max(0.3, score), 4)


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 3 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_risk_classification(predicted_risk: str, marks: float) -> float:
    if marks < 40:
        expected = "high"
    elif marks <= 60:
        expected = "medium"
    else:
        expected = "low"

    return 0.99 if predicted_risk == expected else 0.01 


def grade_intervention(risk_level: str, intervention: str) -> float:
    if not intervention:
        return 0.01 
    severity_keywords = {
        "high":   ["counseling", "remedial", "parent", "immediate"],
        "medium": ["mentoring", "weekly", "practice"],
        "low":    ["monitoring", "enrichment", "optional"],
    }
    expected_kws = severity_keywords.get(risk_level, [])
    hits = sum(1 for kw in expected_kws if kw.lower() in intervention.lower())
    if hits >= 2:
        return 0.5
    elif hits == 1:
        return 0.3
    return 0.1


# ──────────────────────────────────────────────────────────────────────────── #
#  Aggregate Grader (FIXED TOTAL SCORE BUG)                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_episode_score(rewards: list) -> Dict[str, Any]:
    if not rewards:
        return {"mean": 0.01, "max": 0.01, "min": 0.01, "total": 0.01, "count": 0} 
    
    avg = sum(rewards) / len(rewards)
    return {
        "mean":  round(max(0.01, min(0.99, avg)), 4),
        "max":   round(max(0.01, min(0.99, max(rewards))), 4),
        "min":   round(max(0.01, min(0.99, min(rewards))), 4),
        # Total ko sum ki jagah clamped average diya hai, taaki 1.0 se upar na jaye
        "total": round(max(0.01, min(0.99, avg)), 4), 
        "count": len(rewards),
    }
    
