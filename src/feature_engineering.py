"""Feature engineering helpers compatible with the demo pipeline."""

from app.inference.pipeline import build_purchase_sequence, build_static_features

__all__ = [
    "build_static_features",
    "build_purchase_sequence",
    "build_customer_features",
]


def build_customer_features(customer):
    static = build_static_features(customer)
    purchase_sequence = build_purchase_sequence(customer)
    return {"static_features": static, "purchase_sequence": purchase_sequence}
