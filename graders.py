"""
EduNexora AI — Graders
FRACTIONAL DIVERSE REWARDS (Score safe for 100 steps)
"""
from typing import Any, Dict, Optional

def grade_classification(predicted: str, marks: float) -> float:
    if marks >= 40: expected = "pass"
    elif marks >= 35: expected = "fail"
    else: expected = "backlog"
    
    # 0.008 * 100 students = 0.80 Total (Safe!)
    if predicted == expected: return 0.008  
    if expected == "fail" and predicted == "backlog": return 0.004
    if expected == "backlog" and predicted == "fail": return 0.003
    if expected == "pass" and predicted == "fail": return 0.002
    return 0.0005

def grade_ranking(predicted_order: list, actual_marks: Dict[str, float]) -> float:
    if not predicted_order: return 0.0005
    sorted_ids = sorted(actual_marks.keys(), key=lambda sid: actual_marks[sid], reverse=True)
    if predicted_order == sorted_ids: return 0.05
    top_n = min(3, len(sorted_ids))
    correct = sum(1 for i in range(top_n) if i < len(predicted_order) and predicted_order[i] == sorted_ids[i])
    return round(max(0.0005, (correct / top_n) * 0.05), 4)

def grade_prioritization(selected_unit: str, lag_scores: Dict[str, int]) -> float:
    if not lag_scores: return 0.001
    best = max(lag_scores, key=lag_scores.get)
    if selected_unit == best: return 0.04
    selected_lag = lag_scores.get(selected_unit, 0)
    best_lag = lag_scores[best]
    if best_lag == 0: return 0.001
    return round(max(0.01, (selected_lag / best_lag) * 0.04), 4)

def grade_topic_completion(progress_pct: float) -> float:
    return round(max(0.001, (progress_pct / 100.0) * 0.04), 4)

def grade_notification(notifications: list) -> float:
    if not notifications: return 0.001
    keywords = ["lagging", "progress", "class", "complete", "required", "%"]
    hits = sum(1 for n in notifications for kw in keywords if kw.lower() in n.lower())
    score = hits / max(len(notifications) * 2, 1)
    return round(max(0.01, score * 0.04), 4)

def grade_risk_classification(predicted_risk: str, marks: float) -> float:
    if marks < 40: expected = "high"
    elif marks <= 60: expected = "medium"
    else: expected = "low"
    
    if predicted_risk == expected: return 0.004 
    if expected == "high" and predicted_risk == "medium": return 0.002
    if expected == "low" and predicted_risk == "medium": return 0.002
    if expected == "medium" and predicted_risk in ["high", "low"]: return 0.001
    return 0.0005

def grade_intervention(risk_level: str, intervention: str) -> float:
    if not intervention: return 0.0005
    severity_keywords = {
        "high": ["counseling", "remedial", "parent", "immediate"],
        "medium": ["mentoring", "weekly", "practice"],
        "low": ["monitoring", "enrichment", "optional"],
    }
    expected_kws = severity_keywords.get(risk_level, [])
    hits = sum(1 for kw in expected_kws if kw.lower() in intervention.lower())
    if hits >= 2: return 0.004
    elif hits == 1: return 0.002
    return 0.0005

def compute_episode_score(rewards: list) -> Dict[str, Any]:
    if not rewards: 
        return {"mean": 0.001, "max": 0.001, "min": 0.001, "total": 0.001, "count": 0} 
    total_sum = sum(rewards)
    return {
        "mean":  round(total_sum / len(rewards), 4),
        "max":   round(max(rewards), 4),
        "min":   round(min(rewards), 4),
        "total": round(max(0.001, min(0.999, total_sum)), 4), 
        "count": len(rewards),
    }
