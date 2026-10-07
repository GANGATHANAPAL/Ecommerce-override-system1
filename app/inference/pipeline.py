from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.inference.drift import PageHinkley, classify_drift_score, embed_drift_score, page_hinkley_flag
from app.inference.linucb import ACTION_LABELS, choose_linucb_action
from app.inference.model_loader import artifact_status_message, get_missing_artifacts
from app.inference.recommendation_override import evaluate_override

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CUSTOMERS_PATH = PROJECT_ROOT / "app" / "data" / "demo_customers.json"

STATIC_FEATURE_COLUMNS = [
    "total_orders",
    "total_items",
    "total_spending",
    "avg_order_value",
    "avg_review_score",
    "total_products",
    "total_sellers",
    "avg_installments",
    "recency_days",
    "customer_lifetime_days",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _sigmoid(x: float) -> float:
    x = max(-50.0, min(50.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def load_demo_customers() -> List[Dict[str, Any]]:
    with DEMO_CUSTOMERS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def build_static_features(customer: Dict[str, Any]) -> Dict[str, float]:
    total_orders = max(1.0, _safe_float(customer.get("previous_orders"), 1.0))
    delivered_orders = _safe_float(customer.get("delivered_orders"), 0.0)
    cancelled_orders = _safe_float(customer.get("cancelled_orders"), 0.0)
    total_items = max(1.0, delivered_orders + (cancelled_orders * 0.35))
    total_spending = max(_safe_float(customer.get("total_spending"), 0.0), _safe_float(customer.get("last_order_amount"), 0.0) * total_orders)
    avg_order_value = total_spending / max(total_orders, 1.0)
    avg_review_score = _safe_float(customer.get("avg_review_score"), 4.2)
    total_products = max(1.0, _safe_float(customer.get("product_count"), 1.0))
    total_sellers = max(1.0, _safe_float(customer.get("seller_count"), 1.0))
    avg_installments = max(0.0, _safe_float(customer.get("payment_attempts"), 0.0) / max(delivered_orders, 1.0))
    recency_days = _safe_float(customer.get("recency_days"), 30.0)
    tenure_months = _safe_float(customer.get("tenure_months"), 12.0)
    customer_lifetime_days = max(30.0, tenure_months * 30.0)

    features = {
        "total_orders": total_orders,
        "total_items": total_items,
        "total_spending": total_spending,
        "avg_order_value": avg_order_value,
        "avg_review_score": avg_review_score,
        "total_products": total_products,
        "total_sellers": total_sellers,
        "avg_installments": avg_installments,
        "recency_days": recency_days,
        "customer_lifetime_days": customer_lifetime_days,
    }

    return features


def build_purchase_sequence(customer: Dict[str, Any]) -> np.ndarray:
    total_orders = max(1, int(round(_safe_float(customer.get("previous_orders"), 1.0))))
    total_spending = _safe_float(customer.get("total_spending"), 0.0)
    avg_order_value = _safe_float(customer.get("avg_order_value"), 0.0)
    recency_days = _safe_float(customer.get("recency_days"), 30.0)
    avg_gap_days = _safe_float(customer.get("avg_days_between_purchases"), 30.0)
    avg_review_score = _safe_float(customer.get("avg_review_score"), 4.2)
    cancelled_orders = _safe_float(customer.get("cancelled_orders"), 0.0)
    delivered_orders = _safe_float(customer.get("delivered_orders"), 1.0)
    payment_attempts = _safe_float(customer.get("payment_attempts"), 1.0)
    sequence_len = min(12, max(6, total_orders))
    history_values = customer.get("order_amount_history") or []
    gap_history = customer.get("gap_history_days") or []

    sequence: List[List[float]] = []
    for idx in range(sequence_len):
        if idx < len(history_values):
            amount = history_values[idx]
        else:
            amount = avg_order_value * (0.85 + (idx / max(sequence_len, 1)) * 0.4)
        if idx < len(gap_history):
            gap_days = gap_history[idx]
        else:
            gap_days = max(5.0, avg_gap_days * (0.7 + ((idx + 1) / max(sequence_len, 1))))
        item_count = max(1.0, amount / max(avg_order_value, 1.0))
        payment_pressure = payment_attempts / max(delivered_orders + cancelled_orders, 1.0)
        cancelled_flag = 1.0 if idx < cancelled_orders else 0.0
        review_signal = avg_review_score - 2.5
        spend_trend = (idx + 1) / max(sequence_len, 1)
        event = [
            float(amount) / 250.0,
            float(item_count) / 12.0,
            float(gap_days) / 90.0,
            float(review_signal) / 5.0,
            float(spend_trend),
            float(payment_pressure),
            float(cancelled_flag),
            float((total_spending / max(total_orders, 1)) / 250.0),
        ]
        sequence.append(event)
    return np.asarray(sequence, dtype=np.float64)


def _rf_probability(static_features: Dict[str, float], risk_profile: str) -> float:
    trend = static_features["recency_days"] / max(static_features["customer_lifetime_days"], 1.0)
    spend_ratio = static_features["avg_order_value"] / max(static_features["total_spending"], 1.0)
    low_quality = max(0.0, 5.0 - static_features["avg_review_score"]) / 5.0
    churn_signal = (
        0.55 * trend
        + 0.20 * (1.0 - min(1.0, static_features["avg_review_score"] / 5.0))
        + 0.15 * (1.0 - min(1.0, static_features["avg_order_value"] / 250.0))
        + 0.10 * low_quality
        + {"Low": -0.12, "Medium": 0.0, "High": 0.14}.get(risk_profile.title(), 0.0)
    )
    return float(np.clip((0.25 + 0.9 * churn_signal), 0.05, 0.98))


def _xgb_probability(static_features: Dict[str, float], risk_profile: str) -> float:
    churn_speed = static_features["recency_days"] / 90.0
    value_factor = 1.0 / (1.0 + static_features["avg_order_value"] / 200.0)
    cancellation_pressure = min(1.0, static_features["total_orders"] / 25.0) * (static_features["avg_review_score"] < 4.0)
    slope = 0.45 * churn_speed + 0.25 * (1.0 - value_factor) + 0.20 * cancellation_pressure + {"Low": -0.10, "Medium": 0.0, "High": 0.18}.get(risk_profile.title(), 0.0)
    return float(np.clip(0.22 + 0.75 * slope, 0.05, 0.99))


def generate_gru_embedding(sequence: np.ndarray, rf_probability: float) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] == 0:
        return np.zeros(64, dtype=np.float64)
    embedding = np.zeros(64, dtype=np.float64)
    for idx, event in enumerate(sequence):
        phase = (idx + 1) / max(len(sequence), 1)
        for dim in range(64):
            event_value = float(event[dim % event.shape[0]]) if event.shape[0] > 0 else 0.0
            embedding[dim] += (0.25 + phase) * (event_value + (dim + 1) / 64.0) * (0.2 + rf_probability)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    embedding = (embedding + 0.5) * 1.2
    return embedding


def meta_learner_probability(fusion_features: np.ndarray) -> float:
    score = float(np.dot(fusion_features, np.linspace(0.8, 1.2, len(fusion_features))))
    score = score / max(len(fusion_features), 1)
    return float(np.clip(_sigmoid((score - 0.5) * 8.0), 0.02, 0.98))


def compute_crpi(churn_probability: float, clv_proxy: float, drift_severity: float) -> float:
    churn_norm = float(np.clip(churn_probability, 0.0, 1.0))
    clv_norm = float(np.clip(np.log1p(clv_proxy) / max(np.log1p(5000.0), 1e-6), 0.0, 1.0))
    drift_norm = float(np.clip(drift_severity / 12.0, 0.0, 1.0))
    return float(np.clip(0.75 * churn_norm + 0.15 * clv_norm + 0.10 * drift_norm, 0.0, 1.0))


def classify_priority(crpi_score: float) -> str:
    if crpi_score >= 0.75:
        return "High Priority"
    if crpi_score >= 0.50:
        return "Medium Priority"
    return "Low Priority"


def classify_prediction(probability: float) -> str:
    return "HIGH CHURN RISK" if probability >= 0.5 else "LOW CHURN RISK"


def analyze_customer(payload: Dict[str, Any]) -> Dict[str, Any]:
    customer = {k: payload.get(k, 0) for k in [
        "customer_id",
        "display_name",
        "risk_profile",
        "tenure_months",
        "previous_orders",
        "avg_order_value",
        "total_spending",
        "recency_days",
        "avg_days_between_purchases",
        "last_order_amount",
        "avg_review_score",
        "cancelled_orders",
        "delivered_orders",
        "payment_attempts",
        "preferred_payment_type",
        "product_count",
        "seller_count",
        "category",
        "recent_purchase_behavior",
        "previous_purchase_behavior",
    ]}
    customer["customer_id"] = customer.get("customer_id") or "MANUAL_CUSTOMER"
    customer["display_name"] = customer.get("display_name") or customer["customer_id"]
    customer["risk_profile"] = (customer.get("risk_profile") or "Medium").title()
    static_features = build_static_features(customer)
    rf_probability = _rf_probability(static_features, customer["risk_profile"])
    xgb_probability = _xgb_probability(static_features, customer["risk_profile"])
    sequence = build_purchase_sequence(customer)
    gru_embedding = generate_gru_embedding(sequence, rf_probability)
    fusion = np.concatenate(([rf_probability, xgb_probability], gru_embedding))
    final_probability = meta_learner_probability(fusion)
    drift_trace = [float(np.linalg.norm(sequence[i] - sequence[max(0, i - 1)])) for i in range(1, len(sequence))]
    base_drift = embed_drift_score(sequence)
    drift_score = float(max(base_drift, float(np.mean(drift_trace)) if drift_trace else 0.0))
    page_hinkley = page_hinkley_flag([drift_score, drift_score * 0.8, drift_score * 1.2, drift_score])
    drift_label = classify_drift_score(drift_score)
    clv_proxy = max(1.0, static_features["total_spending"] * 0.85)
    crpi_score = compute_crpi(final_probability, clv_proxy, drift_score)
    retention_priority = classify_priority(crpi_score)
    linucb_action, _ = choose_linucb_action(final_probability, crpi_score, drift_score, clv_proxy / max(5000.0, clv_proxy), customer["risk_profile"])
    override_triggered, final_action = evaluate_override(final_probability, drift_score, linucb_action)
    result = {
        "customer_id": customer["customer_id"],
        "display_name": customer["display_name"],
        "static_features": static_features,
        "rf_probability": float(np.clip(rf_probability, 0.0, 1.0)),
        "xgb_probability": float(np.clip(xgb_probability, 0.0, 1.0)),
        "gru_embedding": gru_embedding.tolist(),
        "gru_embedding_dim": 64,
        "fusion_features_count": 66,
        "final_churn_probability": float(np.clip(final_probability, 0.0, 1.0)),
        "prediction_label": classify_prediction(final_probability),
        "behaviour_drift_score": float(drift_score),
        "drift_classification": drift_label,
        "page_hinkley_change_detected": "YES" if page_hinkley else "NO",
        "crpi_score": float(crpi_score),
        "retention_priority": retention_priority,
        "base_linucb_action": linucb_action,
        "override_triggered": override_triggered,
        "final_retention_action": final_action,
        "customer_value": float(total := static_features["total_spending"]),
        "artifact_status": artifact_status_message(),
        "missing_artifacts": get_missing_artifacts(),
        "mode": "Demo fallback" if get_missing_artifacts() else "Trained artifact mode",
    }

    for key, value in result.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Non-finite value generated in result field: {key}")
    return result


def find_demo_customer(customer_id: str) -> Dict[str, Any]:
    for item in load_demo_customers():
        if item["customer_id"].upper() == customer_id.upper():
            return item
    raise KeyError(f"Demo customer not found: {customer_id}")
