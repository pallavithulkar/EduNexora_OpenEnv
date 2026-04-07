"""
EduNexora AI — Graders
Computes dynamic rewards for each task based on action correctness.
All rewards are in [0.0, 1.0].
"""

from typing import Any, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 1 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_classification(predicted: str, marks: float) -> float:
    """
    Grades student classification accuracy.
    marks >= 40  → pass   → 1.0 if correct
    35 <= marks < 40 → fail  → 1.0 if correct
    marks < 35   → backlog → 1.0 if correct
    Wrong: 0.0
    """
    if marks >= 40:
        expected = "pass"
    elif marks >= 35:
        expected = "fail"
    else:
        expected = "backlog"

    return 1.0 if predicted == expected else 0.0


def grade_ranking(predicted_order: list, actual_marks: Dict[str, float]) -> float:
    """
    Grades ranking quality using Spearman-like partial reward.
    Returns 0.5 for any valid sorted list (binary grading).
    """
    if not predicted_order:
        return 0.0
    sorted_ids = sorted(actual_marks.keys(), key=lambda sid: actual_marks[sid], reverse=True)
    if predicted_order == sorted_ids:
        return 0.5
    # Partial credit: proportion of correctly-placed top-3
    top_n      = min(3, len(sorted_ids))
    correct    = sum(1 for i in range(top_n) if i < len(predicted_order) and predicted_order[i] == sorted_ids[i])
    return round((correct / top_n) * 0.5, 4)


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 2 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_prioritization(selected_unit: str, lag_scores: Dict[str, int]) -> float:
    """
    Grades unit prioritization.
    Best unit (most lagging) → 0.5
    Sub-optimal → 0.2
    """
    if not lag_scores:
        return 0.0
    best = max(lag_scores, key=lag_scores.get)
    if selected_unit == best:
        return 0.5
    # Partial: proportional to lag of selected vs best
    selected_lag = lag_scores.get(selected_unit, 0)
    best_lag     = lag_scores[best]
    if best_lag == 0:
        return 0.0
    return round(max(0.2, (selected_lag / best_lag) * 0.5), 4)


def grade_topic_completion(progress_pct: float) -> float:
    """
    Grades topic completion by current progress percentage.
    100% → 1.0, otherwise proportional.
    """
    return round(min(1.0, progress_pct / 100.0), 4)


def grade_notification(notifications: list) -> float:
    """
    Grades quality of generated notifications.
    Returns 0.5 if notifications are non-empty and informative.
    """
    if not notifications:
        return 0.0
    # Check for key information keywords
    keywords = ["lagging", "progress", "class", "complete", "required", "%"]
    hits      = sum(1 for n in notifications for kw in keywords if kw.lower() in n.lower())
    score     = min(1.0, hits / max(len(notifications) * 2, 1))
    return round(max(0.3, score), 4)


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 3 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_risk_classification(predicted_risk: str, marks: float) -> float:
    """
    Grades risk classification.
    < 40   → high   → 1.0 if correct, 0.0 otherwise
    40-60  → medium → 1.0 if correct, 0.0 otherwise
    > 60   → low    → 1.0 if correct, 0.0 otherwise
    """
    if marks < 40:
        expected = "high"
    elif marks <= 60:
        expected = "medium"
    else:
        expected = "low"

    return 1.0 if predicted_risk == expected else 0.0


def grade_intervention(risk_level: str, intervention: str) -> float:
    """
    Grades intervention appropriateness.
    If intervention is non-empty and matches expected severity → 0.5
    """
    if not intervention:
        return 0.0
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
#  Aggregate Grader                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_episode_score(rewards: list) -> Dict[str, Any]:
    """
    Computes aggregate statistics for an episode's rewards.
    """
    if not rewards:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "total": 0.0, "count": 0}
    return {
        "mean":  round(sum(rewards) / len(rewards), 4),
        "max":   round(max(rewards), 4),
        "min":   round(min(rewards), 4),
        "total": round(sum(rewards), 4),
        "count": len(rewards),
    }
    
