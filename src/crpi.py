"""CRPI scoring helpers.

These helpers are intentionally lightweight wrappers around the runtime pipeline used
by the demo app so both the app package and the source module tree expose the same
logic.
"""

from app.inference.pipeline import compute_crpi, classify_priority

__all__ = ["compute_crpi", "classify_priority", "compute_crpi_score", "classify_retention_priority"]


def compute_crpi_score(churn_probability: float, clv_proxy: float, drift_severity: float) -> float:
    return compute_crpi(churn_probability, clv_proxy, drift_severity)


def classify_retention_priority(crpi_score: float) -> str:
    return classify_priority(crpi_score)
