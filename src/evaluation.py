"""Evaluation helpers for retention scoring."""

from app.inference.linucb import ACTION_LABELS, choose_linucb_action
from app.inference.pipeline import classify_prediction

__all__ = ["ACTION_LABELS", "choose_linucb_action", "classify_prediction", "evaluate_action_choice"]


def evaluate_action_choice(churn_probability: float, crpi_score: float, drift_severity: float, clv_score: float, risk_profile: str = "Medium"):
    return choose_linucb_action(churn_probability, crpi_score, drift_severity, clv_score, risk_profile)
