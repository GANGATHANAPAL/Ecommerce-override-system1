"""Recommendation override helpers."""

from app.inference.recommendation_override import evaluate_override

__all__ = ["evaluate_override", "check_override"]


def check_override(churn_probability: float, drift_severity: float, linucb_action: str):
    return evaluate_override(churn_probability, drift_severity, linucb_action)
