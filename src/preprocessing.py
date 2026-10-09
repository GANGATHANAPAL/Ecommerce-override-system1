"""High-level preprocessing helpers mirrored from the app pipeline."""

from app.inference.pipeline import build_static_features, load_demo_customers

__all__ = [
    "build_static_features",
    "load_demo_customers",
    "prepare_customer_payload",
]


def prepare_customer_payload(raw_customer):
    customer = {k: raw_customer.get(k, 0) for k in [
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
    return customer
