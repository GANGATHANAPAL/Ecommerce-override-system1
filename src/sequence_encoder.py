"""Sequence encoder compatibility layer."""

from app.inference.pipeline import (
    _rf_probability,
    build_purchase_sequence,
    build_static_features,
    generate_gru_embedding,
)

__all__ = ["build_purchase_sequence", "generate_gru_embedding", "encode_customer_sequence"]


def encode_customer_sequence(customer):
    sequence = build_purchase_sequence(customer)
    static_features = build_static_features(customer)
    rf_probability = _rf_probability(static_features, (customer.get("risk_profile") or "Medium").title())
    return generate_gru_embedding(sequence, rf_probability)
