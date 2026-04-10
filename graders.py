"""
EduNexora AI — graders.py
All returns: round(max(0.01, min(0.99, v)), 4)
"""
from typing import Any, Dict


def _c(v: float) -> float:
    return round(max(0.01, min(0.99, float(v))), 4)


def grade_classification(predicted: str, marks: float) -> float:
    if marks >= 40:   expected = "pass"
    elif marks >= 35: expected = "fail"
    else:             expected = "backlog"
    if predicted == expected:                              return _c(0.85)
    if expected == "fail"    and predicted == "backlog":   return _c(0.50)
    if expected == "backlog" and predicted == "fail":      return _c(0.40)
    if expected == "pass"    and predicted == "fail":      return _c(0.20)
    return _c(0.10)


def grade_ranking(predicted_order: list, actual_marks: Dict[str, float]) -> float:
    if not predicted_order or not actual_marks:
        return _c(0.10)
    sorted_ids = sorted(actual_marks.keys(), key=lambda s: actual_marks[s], reverse=True)
    if predicted_order == sorted_ids:
        return _c(0.80)
    top_n   = min(3, len(sorted_ids))
    correct = sum(1 for i in range(top_n) if i < len(predicted_order) and predicted_order[i] == sorted_ids[i])
    return _c((correct / top_n) * 0.80)


def grade_prioritization(selected_unit: str, lag_scores: Dict[str, int]) -> float:
    if not lag_scores: return _c(0.10)
    best = max(lag_scores, key=lag_scores.get)
    if selected_unit == best: return _c(0.80)
    selected_lag = lag_scores.get(selected_unit, 0)
    best_lag     = lag_scores.get(best, 1)
    return _c((selected_lag / best_lag) * 0.80) if best_lag else _c(0.10)


def grade_topic_completion(progress_pct: float) -> float:
    return _c(progress_pct / 100.0 * 0.85)


def grade_notification(notifications: list) -> float:
    if not notifications: return _c(0.10)
    keywords = ["lagging", "progress", "complete", "required", "%", "behind"]
    hits     = sum(1 for n in notifications for kw in keywords if kw.lower() in n.lower())
    score    = hits / max(len(notifications) * 2, 1)
    return _c(score * 0.80)


def grade_risk_classification(predicted_risk: str, marks: float) -> float:
    if marks < 40:    expected = "high"
    elif marks <= 60: expected = "medium"
    else:             expected = "low"
    if predicted_risk == expected:                             return _c(0.85)
    if expected in ("high","low") and predicted_risk == "medium": return _c(0.50)
    return _c(0.10)


def grade_intervention(risk_level: str, intervention: str) -> float:
    if not intervention: return _c(0.10)
    kws = {
        "high":   ["counseling", "remedial", "parent", "immediate"],
        "medium": ["mentoring", "weekly", "practice"],
        "low":    ["monitoring", "enrichment", "optional"],
    }
    hits = sum(1 for kw in kws.get(risk_level, []) if kw.lower() in intervention.lower())
    if hits >= 2: return _c(0.80)
    if hits == 1: return _c(0.50)
    return _c(0.10)


def compute_episode_score(rewards: list) -> Dict[str, Any]:
    if not rewards:
        return {"mean": _c(0.10), "max": _c(0.10), "min": _c(0.10), "total": _c(0.10), "count": 0}
    avg = sum(rewards) / len(rewards)
    return {
        "mean":  _c(avg),
        "max":   _c(max(rewards)),
        "min":   _c(min(rewards)),
        "total": _c(avg),       # Use AVERAGE not sum — sum could exceed 1.0
        "count": len(rewards),
    }
