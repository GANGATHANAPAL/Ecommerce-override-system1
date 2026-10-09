"""Hybrid classifier interfaces for the demo retention project."""

from app.inference.pipeline import (
    _rf_probability,
    _xgb_probability,
    meta_learner_probability,
)

__all__ = [
    "rf_probability",
    "xgb_probability",
    "meta_learner_probability",
    "ensemble_probability",
]


def rf_probability(static_features, risk_profile):
    return _rf_probability(static_features, risk_profile)


def xgb_probability(static_features, risk_profile):
    return _xgb_probability(static_features, risk_profile)


def ensemble_probability(static_features, risk_profile):
    rf_value = rf_probability(static_features, risk_profile)
    xgb_value = xgb_probability(static_features, risk_profile)
    return (rf_value + xgb_value) / 2.0
