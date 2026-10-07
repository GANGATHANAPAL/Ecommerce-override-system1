from __future__ import annotations


def evaluate_override(churn_probability: float, drift_severity: float, linucb_action: str) -> tuple[bool, str]:
    """Demo-only override rule for the UI. The project workspace does not contain a saved override model."""
    trigger = bool(churn_probability >= 0.72 and drift_severity >= 2.2)
    if trigger:
        if linucb_action == "No Intervention":
            return True, "Loyalty Incentive"
        return True, "Proactive Offer" if linucb_action != "Loyalty Incentive" else "Loyalty Incentive"
    return False, linucb_action
