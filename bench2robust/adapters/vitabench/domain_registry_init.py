# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Register VitaBench domains (delivery + instore) with the DomainRegistry.

Call register_vitabench_domains() during framework initialization to enable
domain-aware noise injection for VitaBench tasks.
"""

from bench2robust.uncertainty.domain_registry import DomainRegistry


def register_vitabench_domains():
    """Register both VitaBench domains with critical fields and stale rules."""

    # =========================================================================
    # DELIVERY DOMAIN
    # =========================================================================
    DomainRegistry.register_critical_fields("vitabench_delivery", {
        "get_delivery_store_info": ["score", "tags", "products"],
        "get_delivery_product_info": ["price", "score", "tags"],
        "get_delivery_order_detail": ["status", "products", "total_price", "address"],
        "get_delivery_order_status": ["status"],
        "search_delivery_orders": ["status", "total_price"],
        "delivery_store_search_recommend": ["score", "tags"],
        "delivery_product_search_recommend": ["price", "score"],
    })

    DomainRegistry.register_stale_rules("vitabench_delivery", {
        "get_delivery_order_detail": {
            "status": lambda rng, v: "unpaid" if v in ("paid", "processed", "delivered") else v,
        },
        "get_delivery_order_status": {
            "status": lambda rng, v: "unpaid" if v in ("paid", "processed", "delivered") else v,
        },
    })

    DomainRegistry.register_status_transitions("vitabench_delivery", {
        "delivered": "processed",
        "processed": "paid",
        "paid": "unpaid",
        "cancelled": "unpaid",
    })

    DomainRegistry.register_empty_sentinels("vitabench_delivery", {
        "products": [],
        "tags": [],
        "address": "",
        "total_price": 0.0,
    })

    DomainRegistry.register_factual_error_rules("vitabench_delivery", {
        "price": lambda rng, v: round(v * rng.uniform(0.75, 1.25), 2) if isinstance(v, (int, float)) else v,
        "score": lambda rng, v: round(max(1.0, min(5.0, v + rng.uniform(-0.5, 0.5))), 1) if isinstance(v, (int, float)) else v,
        "total_price": lambda rng, v: round(v * rng.uniform(0.8, 1.2), 2) if isinstance(v, (int, float)) else v,
    })

    # =========================================================================
    # INSTORE DOMAIN
    # =========================================================================
    DomainRegistry.register_critical_fields("vitabench_instore", {
        "get_instore_books": ["status", "time", "shop_id"],
        "search_instore_book": ["status", "time", "shop_id"],
        "get_instore_reservations": ["status", "time", "customer_count"],
        "search_instore_reservation": ["status", "time", "customer_count"],
        "get_instore_orders": ["status", "total_price", "products"],
        "instore_shop_search_recommend": ["score", "tags"],
        "instore_product_search_recommend": ["price", "score"],
    })

    DomainRegistry.register_stale_rules("vitabench_instore", {
        "get_instore_books": {
            "status": lambda rng, v: "unpaid" if v in ("paid", "confirmed") else v,
        },
        "get_instore_reservations": {
            "status": lambda rng, v: "pending" if v in ("confirmed", "completed") else v,
        },
    })

    DomainRegistry.register_status_transitions("vitabench_instore", {
        "confirmed": "pending",
        "completed": "confirmed",
        "paid": "unpaid",
        "cancelled": "pending",
    })

    DomainRegistry.register_empty_sentinels("vitabench_instore", {
        "products": [],
        "tags": [],
        "time": "",
        "total_price": 0.0,
    })

    DomainRegistry.register_factual_error_rules("vitabench_instore", {
        "price": lambda rng, v: round(v * rng.uniform(0.75, 1.25), 2) if isinstance(v, (int, float)) else v,
        "score": lambda rng, v: round(max(1.0, min(5.0, v + rng.uniform(-0.5, 0.5))), 1) if isinstance(v, (int, float)) else v,
        "customer_count": lambda rng, v: max(1, v + rng.choice([-1, 1])) if isinstance(v, int) else v,
    })
