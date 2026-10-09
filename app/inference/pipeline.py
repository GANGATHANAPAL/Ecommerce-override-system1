from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from app.inference.drift import PageHinkley, classify_drift_score
from app.inference.model_loader import (
    artifact_status_message,
    get_missing_artifacts,
    load_trained_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CUSTOMERS_PATH = PROJECT_ROOT / "app" / "data" / "demo_customers.json"

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
    delivered_orders = max(1.0, _safe_float(customer.get("delivered_orders"), 1.0))
    avg_gap_days = _safe_float(customer.get("avg_days_between_purchases"), 30.0)
    avg_review_score = _safe_float(customer.get("avg_review_score"), 4.2)
    payment_attempts = _safe_float(customer.get("payment_attempts"), 1.0)
    sequence_len = min(12, max(6, total_orders))
    history_values = customer.get("order_amount_history") or []
    gap_history = customer.get("gap_history_days") or []
    average_items = max(
        1.0,
        (_safe_float(customer.get("delivered_orders"), 0.0)
         + 0.35 * _safe_float(customer.get("cancelled_orders"), 0.0))
        / total_orders,
    )
    average_products = max(
        1.0,
        _safe_float(customer.get("product_count"), 1.0) / total_orders,
    )
    average_sellers = max(
        1.0,
        _safe_float(customer.get("seller_count"), 1.0) / total_orders,
    )
    average_installments = max(0.0, payment_attempts / delivered_orders)

    sequence: List[List[float]] = []
    for idx in range(sequence_len):
        if idx < len(history_values):
            amount = _safe_float(history_values[idx])
        else:
            amount = _safe_float(customer.get("avg_order_value"), 0.0)
        if idx < len(gap_history):
            gap_days = _safe_float(gap_history[idx])
        else:
            gap_days = max(5.0, avg_gap_days * (0.7 + ((idx + 1) / max(sequence_len, 1))))
        event = [
            average_items,
            amount,
            average_products,
            average_sellers,
            amount * 1.1,
            average_installments,
            avg_review_score,
            gap_days,
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


@lru_cache(maxsize=1)
def _load_trained_pipeline() -> Dict[str, Any]:
    return load_trained_artifacts()


@lru_cache(maxsize=1)
def _load_observation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    features_path = PROJECT_ROOT / "data" / "processed" / "customer_features_observation.csv"
    orders_path = PROJECT_ROOT / "data" / "processed" / "observation_orders.csv"
    features = pd.read_csv(features_path, dtype={"customer_unique_id": "string"})
    orders = pd.read_csv(orders_path, dtype={"customer_unique_id": "string"})
    return features, orders


def analyze_customer(payload: Dict[str, Any]) -> Dict[str, Any]:
    customer_id = str(
        payload.get("customer_unique_id") or payload.get("customer_id") or ""
    ).strip()
    if not customer_id:
        raise ValueError("A customer ID is required for trained-model inference.")

    artifacts = _load_trained_pipeline()
    feature_list = artifacts["feature_list"]
    static_columns = feature_list["static_features"]
    sequence_columns = feature_list["sequence_features"]
    is_demo_profile = customer_id.upper().startswith("DEMO_")
    if is_demo_profile:
        demo_customer = {**find_demo_customer(customer_id), **payload}
        static_features = build_static_features(demo_customer)
        matching_features = pd.DataFrame(
            [[static_features[column] for column in static_columns]],
            columns=static_columns,
        )
        raw_sequence = pd.DataFrame(
            build_purchase_sequence(demo_customer),
            columns=sequence_columns,
        )
    else:
        features, orders = _load_observation_data()
        matching_features = features.loc[
            features["customer_unique_id"] == customer_id,
            static_columns,
        ]
        if len(matching_features) != 1:
            raise KeyError(f"Customer not found in the observation-period feature data: {customer_id}")

        raw_events = orders.loc[
            orders["customer_unique_id"] == customer_id,
            [
                "order_purchase_timestamp",
                *[
                    column
                    for column in sequence_columns
                    if column != "time_since_previous_order_days"
                ],
            ],
        ].copy()
        if raw_events.empty:
            raise KeyError(f"No observation-period order sequence found for customer: {customer_id}")
        raw_events["order_purchase_timestamp"] = pd.to_datetime(
            raw_events["order_purchase_timestamp"],
            errors="raise",
        )
        raw_events = raw_events.sort_values("order_purchase_timestamp")
        if len(raw_events) < 2:
            raise ValueError("The trained pipeline requires at least two observed orders.")

        gap_days = (
            raw_events["order_purchase_timestamp"].diff().dt.total_seconds()
            / (60 * 60 * 24)
        )
        raw_events["time_since_previous_order_days"] = gap_days.fillna(0.0)
        raw_sequence = raw_events[sequence_columns]

    raw_sequence = raw_sequence.astype(np.float32)
    imputed_sequence = artifacts["sequence_imputer"].transform(raw_sequence)
    sequence = artifacts["sequence_scaler"].transform(
        pd.DataFrame(imputed_sequence, columns=sequence_columns)
    ).astype(np.float32, copy=False)
    static_matrix = artifacts["static_imputer"].transform(matching_features)
    static_frame = pd.DataFrame(static_matrix, columns=static_columns)

    rf_model = artifacts["rf_model"]
    xgb_model = artifacts["xgb_model"]
    meta_model = artifacts["meta_model"]
    rf_probability = float(
        rf_model.predict_proba(static_frame)[0, list(rf_model.classes_).index(1)]
    )
    xgb_probability = float(
        xgb_model.predict_proba(static_frame)[0, list(xgb_model.classes_).index(1)]
    )

    gru_model = artifacts["gru_model"]
    sequence_tensor = torch.as_tensor(sequence, dtype=torch.float32).unsqueeze(0)
    sequence_lengths = torch.tensor([len(sequence)], dtype=torch.long)
    with torch.no_grad():
        hidden_states, final_embedding, _ = gru_model(sequence_tensor, sequence_lengths)
    gru_embedding = final_embedding[0].cpu().numpy()
    fusion = np.concatenate(
        (
            np.asarray([rf_probability, xgb_probability], dtype=np.float64),
            gru_embedding.astype(np.float64, copy=False),
        )
    ).reshape(1, -1)
    final_probability = float(
        meta_model.predict_proba(fusion)[0, list(meta_model.classes_).index(1)]
    )

    valid_states = hidden_states[0, : len(sequence)].cpu().numpy()
    split_point = max(1, len(valid_states) // 2)
    early_states = valid_states[:split_point]
    recent_states = valid_states[split_point:]
    if len(recent_states) == 0:
        recent_states = valid_states[-1:]
    drift_score = float(
        np.linalg.norm(recent_states.mean(axis=0) - early_states.mean(axis=0))
    )

    drift_config = artifacts["drift_config"]
    drift_threshold = drift_config["drift_threshold"]
    drift_flag = int(drift_score >= drift_threshold)
    page_hinkley_config = drift_config["page_hinkley"]
    page_hinkley_detector = PageHinkley(
        delta=page_hinkley_config["delta"],
        threshold=page_hinkley_config["threshold"],
        alpha=page_hinkley_config["alpha"],
    )
    page_hinkley_detector.mean = page_hinkley_config["mean"]
    page_hinkley_detector.sum = page_hinkley_config["sum"]
    page_hinkley_detector.min_sum = page_hinkley_config["min_sum"]
    page_hinkley_detector.count = page_hinkley_config["count"]
    page_hinkley = page_hinkley_detector.update(drift_score)

    static_values = static_frame.iloc[0].to_dict()
    total_orders = float(static_values["total_orders"])
    total_spending = float(static_values["total_spending"])
    customer_lifetime_days = float(static_values["customer_lifetime_days"])
    avg_order_value_clv = total_spending / total_orders
    order_frequency = total_orders / max(customer_lifetime_days, 1.0)
    clv_proxy = float(avg_order_value_clv * order_frequency)

    crpi_scalers = artifacts["crpi_scalers"]
    churn_scaler = crpi_scalers["churn_scaler"]
    clv_scaler = crpi_scalers["clv_scaler"]
    drift_scaler = crpi_scalers["drift_scaler"]
    churn_normalized = float(
        churn_scaler.transform(np.asarray([[final_probability]]))[0, 0]
    )
    clv_normalized = float(
        clv_scaler.transform(
            pd.DataFrame([[np.log1p(clv_proxy)]], columns=["clv_log"])
        )[0, 0]
    )
    drift_normalized = float(
        drift_scaler.transform(
            pd.DataFrame([[drift_score]], columns=["drift_severity"])
        )[0, 0]
    )
    crpi_score = float((churn_normalized + clv_normalized + drift_normalized) / 3.0)
    if crpi_score >= crpi_scalers["crpi_high_threshold"]:
        retention_priority = "High"
    elif crpi_score >= crpi_scalers["crpi_medium_threshold"]:
        retention_priority = "Medium"
    else:
        retention_priority = "Low"

    context_features = feature_list["bandit_context_features"]
    context = pd.DataFrame(
        [[final_probability, crpi_score, drift_score, clv_normalized]],
        columns=context_features,
    )
    policy = artifacts["linucb_policy"]
    normalized_context = policy["context_scaler"].transform(context)[0]
    action_scores = []
    for action in range(policy["n_actions"]):
        inverse = np.linalg.inv(policy["A"][action])
        theta = inverse @ policy["b"][action]
        exploitation = theta @ normalized_context
        exploration = policy["alpha"] * np.sqrt(
            normalized_context @ inverse @ normalized_context
        )
        action_scores.append(exploitation + exploration)
    action_index = int(np.argmax(action_scores))
    linucb_action = feature_list["retention_actions"][action_index].title()
    override_triggered = False
    final_action = linucb_action
    static_features = {
        key: float(value)
        for key, value in static_values.items()
    }
    priority_label = f"{retention_priority} Priority"
    drift_label = classify_drift_score(drift_score)
    result = {
        "customer_id": customer_id,
        "display_name": payload.get("display_name") or customer_id,
        "static_features": static_features,
        "rf_probability": rf_probability,
        "xgb_probability": xgb_probability,
        "gru_embedding": gru_embedding.tolist(),
        "gru_embedding_dim": 64,
        "fusion_features_count": 66,
        "final_churn_probability": final_probability,
        "prediction_label": classify_prediction(final_probability),
        "behaviour_drift_score": float(drift_score),
        "high_drift": drift_flag,
        "drift_classification": drift_label,
        "page_hinkley_change_detected": "YES" if page_hinkley else "NO",
        "crpi_score": float(crpi_score),
        "clv_proxy": clv_proxy,
        "clv_normalized": clv_normalized,
        "retention_priority": priority_label,
        "base_linucb_action": linucb_action,
        "override_triggered": override_triggered,
        "final_retention_action": final_action,
        "override_status": "No trained override rule is defined; LinUCB action retained.",
        "customer_value": total_spending,
        "artifact_status": artifact_status_message(),
        "missing_artifacts": [],
        "mode": (
            "Trained artifact mode (synthetic demo profile)"
            if is_demo_profile
            else "Trained artifact mode"
        ),
    }

    for key, value in result.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Non-finite value generated in result field: {key}")
    return result


def infer_trained_customer(customer_id: str) -> Dict[str, Any]:
    customer_features, observation_orders = _load_observation_data()
    manifest_path = PROJECT_ROOT / "data" / "processed" / "customer_split_manifest.csv"
    split_manifest = pd.read_csv(manifest_path, dtype={"customer_unique_id": "string"})
    test_ids = set(
        split_manifest.loc[split_manifest["split"] == "test", "customer_unique_id"].astype(str)
    )
    if customer_id not in test_ids:
        raise KeyError(f"Customer is not present in the held-out test observation data: {customer_id}")
    if not (customer_features["customer_unique_id"].astype(str) == str(customer_id)).any():
        raise KeyError(f"Customer is not present in the observation-period feature data: {customer_id}")
    if not (observation_orders["customer_unique_id"].astype(str) == str(customer_id)).any():
        raise KeyError(f"No observation-period orders found for customer: {customer_id}")
    return analyze_customer({"customer_unique_id": str(customer_id)})


def find_demo_customer(customer_id: str) -> Dict[str, Any]:
    for item in load_demo_customers():
        if item["customer_id"].upper() == customer_id.upper():
            return item
    raise KeyError(f"Demo customer not found: {customer_id}")
