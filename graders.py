"""
EduNexora AI — Graders
DIVERSE AND MEANINGFUL REWARDS: Strictly bounded [0.01, 0.99].
Perfect actions get ~0.99, partial get ~0.5, wrong get ~0.01.
"""
from typing import Any, Dict

def _clip(val: float) -> float:
    """Universal safety clamp to strictly enforce (0, 1) range."""
    return round(max(0.01, min(0.99, float(val))), 4)

# ──────────────────────────────────────────────────────────────────────────── #

def grade_classification(predicted: str, marks: float) -> float:
    if marks >= 40: expected = "pass"
    elif marks >= 35: expected = "fail"
    else: expected = "backlog"

    if predicted == expected: return _clip(0.99)
    if expected == "fail" and predicted == "backlog": return _clip(0.60)
    if expected == "backlog" and predicted == "fail": return _clip(0.50)
    if expected == "pass" and predicted == "fail": return _clip(0.30)
    return _clip(0.01)

def grade_ranking(predicted_order: list, actual_marks: Dict[str, float]) -> float:
    if not predicted_order: return _clip(0.01)
    sorted_ids = sorted(actual_marks.keys(), key=lambda sid: actual_marks[sid], reverse=True)
    if predicted_order == sorted_ids: return _clip(0.99)
    top_n = min(3, len(sorted_ids))
    correct = sum(1 for i in range(top_n) if i < len(predicted_order) and predicted_order[i] == sorted_ids[i])
    return _clip((correct / top_n) * 0.99)

# ──────────────────────────────────────────────────────────────────────────── #

def grade_prioritization(selected_unit: str, lag_scores: Dict[str, int]) -> float:
    if not lag_scores: return _clip(0.01)
    best = max(lag_scores, key=lag_scores.get)
    if selected_unit == best: return _clip(0.99)
    selected_lag = lag_scores.get(selected_unit, 0)
    best_lag = lag_scores[best]
    if best_lag == 0: return _clip(0.01)
    return _clip(max(0.2, (selected_lag / best_lag) * 0.8))

def grade_topic_completion(progress_pct: float) -> float:
    return _clip(progress_pct / 100.0)

def grade_notification(notifications: list) -> float:
    if not notifications: return _clip(0.01)
    keywords = ["lagging", "progress", "class", "complete", "required", "%"]
    hits = sum(1 for n in notifications for kw in keywords if kw.lower() in n.lower())
    score = hits / max(len(notifications) * 2, 1)
    return _clip(max(0.3, score))

# ──────────────────────────────────────────────────────────────────────────── #

def grade_risk_classification(predicted_risk: str, marks: float) -> float:
    if marks < 40: expected = "high"
    elif marks <= 60: expected = "medium"
    else: expected = "low"

    if predicted_risk == expected: return _clip(0.99)
    if expected == "high" and predicted_risk == "medium": return _clip(0.55)
    if expected == "low" and predicted_risk == "medium": return _clip(0.55)
    if expected == "medium" and predicted_risk in ["high", "low"]: return _clip(0.35)
    return _clip(0.01)

def grade_intervention(risk_level: str, intervention: str) -> float:
    if not intervention: return _clip(0.01)
    severity_keywords = {
        "high": ["counseling", "remedial", "parent", "immediate"],
        "medium": ["mentoring", "weekly", "practice"],
        "low": ["monitoring", "enrichment", "optional"],
    }
    expected_kws = severity_keywords.get(risk_level, [])
    hits = sum(1 for kw in expected_kws if kw.lower() in intervention.lower())
    if hits >= 2: return _clip(0.99)
    if hits == 1: return _clip(0.60)
    return _clip(0.20)

# ──────────────────────────────────────────────────────────────────────────── #

def compute_episode_score(rewards: list) -> Dict[str, Any]:
    if not rewards: 
        return {"score": 0.01, "total": 0.01, "mean": 0.01, "max": 0.01, "min": 0.01, "count": 0} 
    
    avg_reward = sum(rewards) / len(rewards)
    clamped_avg = _clip(avg_reward)
    
    return {
        "score": clamped_avg,    # Added specifically for Validator compatibility
        "total": clamped_avg,    # Total is represented as average so it NEVER exceeds 1.0
        "mean":  clamped_avg,
        "max":   _clip(max(rewards)),
        "min":   _clip(min(rewards)),
        "count": len(rewards),
    }
