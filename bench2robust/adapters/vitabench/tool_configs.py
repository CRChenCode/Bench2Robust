# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tool configurations for VitaBench domains (delivery + instore).

Defines tool classifications, alternative relationships, and uncertainty priors
for VitaBench integration with Bench2Robust.

VitaBench tools natively include both ID-based lookup and keyword-based search paths,
providing natural equivalence classes without requiring new tool additions.
"""

from bench2robust.uncertainty.config import ToolConfig
from typing import Dict

# Default uncertainty priors for VitaBench read tools (same marginal as Retail)
VITA_DEFAULT_PRIORS = {
    "clean": 0.60,
    "timeout": 0.06,
    "rate_limit": 0.05,
    "server_error": 0.05,
    "auth_error": 0.03,
    "malformed_response": 0.04,
    "schema_drift": 0.03,
    "partial": 0.05,
    "stale": 0.04,
    "factual_error": 0.05,
}

# Meta/utility tools never receive noise
NO_NOISE_PRIORS = {"clean": 1.0}

# Write action priors: no noise (VitaBench evaluator expects deterministic writes)
WRITE_ACTION_PRIORS = NO_NOISE_PRIORS


# ============================================================================
# DELIVERY DOMAIN TOOL CONFIGURATIONS (12 tools)
# ============================================================================

DELIVERY_TOOL_CONFIGS: Dict[str, ToolConfig] = {
    # Meta tools
    "transfer_to_human_agents": ToolConfig(
        name="transfer_to_human_agents",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    # Utility/infrastructure tools (no noise — like calculate in Retail)
    "delivery_distance_to_time": ToolConfig(
        name="delivery_distance_to_time",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "address_to_longitude_latitude": ToolConfig(
        name="address_to_longitude_latitude",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "longitude_latitude_to_distance": ToolConfig(
        name="longitude_latitude_to_distance",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_nearby": ToolConfig(
        name="get_nearby",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "weather": ToolConfig(
        name="weather",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_user_historical_behaviors": ToolConfig(
        name="get_user_historical_behaviors",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_user_all_orders": ToolConfig(
        name="get_user_all_orders",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_now": ToolConfig(
        name="get_now",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_date_holiday_info": ToolConfig(
        name="get_date_holiday_info",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_holiday_date": ToolConfig(
        name="get_holiday_date",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),

    # Read tools with alternatives (equivalence pairs)
    "get_delivery_store_info": ToolConfig(
        name="get_delivery_store_info",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["delivery_store_search_recommend"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "delivery_store_search_recommend": ToolConfig(
        name="delivery_store_search_recommend",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_delivery_store_info"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "get_delivery_product_info": ToolConfig(
        name="get_delivery_product_info",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["delivery_product_search_recommend"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "delivery_product_search_recommend": ToolConfig(
        name="delivery_product_search_recommend",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_delivery_product_info"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "get_delivery_order_detail": ToolConfig(
        name="get_delivery_order_detail",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["search_delivery_orders"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "search_delivery_orders": ToolConfig(
        name="search_delivery_orders",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_delivery_order_detail"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    # Read tools without alternatives
    "get_delivery_order_status": ToolConfig(
        name="get_delivery_order_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    # Write tools (no noise)
    "create_delivery_order": ToolConfig(
        name="create_delivery_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "pay_delivery_order": ToolConfig(
        name="pay_delivery_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "cancel_delivery_order": ToolConfig(
        name="cancel_delivery_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "modify_delivery_order": ToolConfig(
        name="modify_delivery_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
}


# ============================================================================
# INSTORE DOMAIN TOOL CONFIGURATIONS (14 tools)
# ============================================================================

INSTORE_TOOL_CONFIGS: Dict[str, ToolConfig] = {
    # Meta tools
    "transfer_to_human_agents": ToolConfig(
        name="transfer_to_human_agents",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    # Utility/infrastructure tools (no noise)
    "address_to_longitude_latitude": ToolConfig(
        name="address_to_longitude_latitude",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "longitude_latitude_to_distance": ToolConfig(
        name="longitude_latitude_to_distance",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_nearby": ToolConfig(
        name="get_nearby",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "weather": ToolConfig(
        name="weather",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_user_historical_behaviors": ToolConfig(
        name="get_user_historical_behaviors",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_user_all_orders": ToolConfig(
        name="get_user_all_orders",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_now": ToolConfig(
        name="get_now",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_date_holiday_info": ToolConfig(
        name="get_date_holiday_info",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    "get_holiday_date": ToolConfig(
        name="get_holiday_date",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS,
    ),
    # Read tools with alternatives (equivalence pairs)
    "get_instore_books": ToolConfig(
        name="get_instore_books",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["search_instore_book"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "search_instore_book": ToolConfig(
        name="search_instore_book",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_instore_books"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "get_instore_reservations": ToolConfig(
        name="get_instore_reservations",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["search_instore_reservation"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "search_instore_reservation": ToolConfig(
        name="search_instore_reservation",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_instore_reservations"],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    # Read tools without alternatives
    "instore_shop_search_recommend": ToolConfig(
        name="instore_shop_search_recommend",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "instore_product_search_recommend": ToolConfig(
        name="instore_product_search_recommend",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    "get_instore_orders": ToolConfig(
        name="get_instore_orders",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=VITA_DEFAULT_PRIORS,
    ),
    # Write tools (no noise)
    "create_instore_product_order": ToolConfig(
        name="create_instore_product_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "pay_instore_order": ToolConfig(
        name="pay_instore_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "instore_cancel_order": ToolConfig(
        name="instore_cancel_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "instore_book": ToolConfig(
        name="instore_book",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "pay_instore_book": ToolConfig(
        name="pay_instore_book",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "instore_cancel_book": ToolConfig(
        name="instore_cancel_book",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "instore_reservation": ToolConfig(
        name="instore_reservation",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "instore_modify_reservation": ToolConfig(
        name="instore_modify_reservation",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
    "instore_cancel_reservation": ToolConfig(
        name="instore_cancel_reservation",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS,
    ),
}


# ============================================================================
# EQUIVALENCE PAIRS (for S2 scenario blocking)
# ============================================================================

DELIVERY_TOOL_PAIRS = [
    ({"get_delivery_store_info"}, {"delivery_store_search_recommend"}),
    ({"get_delivery_product_info"}, {"delivery_product_search_recommend"}),
    ({"get_delivery_order_detail"}, {"search_delivery_orders"}),
]

INSTORE_TOOL_PAIRS = [
    ({"get_instore_books"}, {"search_instore_book"}),
    ({"get_instore_reservations"}, {"search_instore_reservation"}),
]
