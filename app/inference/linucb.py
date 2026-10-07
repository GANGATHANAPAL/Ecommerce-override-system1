from __future__ import annotations

import math


ACTION_LABELS = [
    "Personalized Recommendation",
    "Loyalty Incentive",
    "Proactive Offer",
    "No Intervention",
]


def choose_linucb_action(churn_probability: float, crpi_score: float, drift_severity: float, clv_score: float, risk_profile: str = "Medium") -> tuple[str, int]:
    """Simple offline/synthetic LinUCB-style selector for demo UI purposes."""
    normalized_risk = float(churn_probability)
    normalized_crpi = float(crpi_score)
    drift_weight = min(float(drift_severity) / 5.0, 1.0)
    clv_weight = min(float(clv_score), 1.0)

    risk_bias = {"Low": -0.10, "Medium": 0.0, "High": 0.12}.get(risk_profile.title(), 0.0)

    action_scores = {
        "Personalized Recommendation": 0.38 * normalized_risk + 0.35 * normalized_crpi + 0.15 * drift_weight + 0.12 * clv_weight + risk_bias,
        "Loyalty Incentive": 0.42 * normalized_risk + 0.30 * normalized_crpi + 0.18 * drift_weight + 0.10 * clv_weight + 0.05,
        "Proactive Offer": 0.34 * normalized_risk + 0.23 * normalized_crpi + 0.28 * drift_weight + 0.15 * clv_weight,
        "No Intervention": 0.12 * normalized_risk + 0.10 * normalized_crpi + 0.05 * drift_weight + 0.05 * clv_weight - 0.10,
    }

    chosen_action = max(action_scores, key=action_scores.get)
    chosen_index = ACTION_LABELS.index(chosen_action)
    return chosen_action, chosen_index


def safe_float(value, default=0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default
