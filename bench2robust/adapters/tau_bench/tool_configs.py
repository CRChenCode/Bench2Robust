# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tool configurations for Tau-Bench domains.

This module defines complete tool classifications for both retail and airline
domains, including write action flags, alternative relationships, and uncertainty priors.

Write actions are explicitly classified and never receive injected noise.
This preserves evaluator state consistency and prevents a failed-looking write
from being replayed as a duplicate state mutation.

Meta tools (think, calculate, transfer_to_human_agents) also remain clean=1.0.
"""

from bench2robust.uncertainty.config import ToolConfig
from typing import Dict

# Default uncertainty priors for read tools
DEFAULT_PRIORS = {
    "clean": 0.7,
    "timeout": 0.05,
    "rate_limit": 0.04,
    "server_error": 0.04,
    "auth_error": 0.02,
    "malformed_response": 0.03,
    "schema_drift": 0.02,
    "partial": 0.04,
    "stale": 0.03,
    "factual_error": 0.03
}

# Meta tools never receive noise
NO_NOISE_PRIORS = {"clean": 1.0}

# Write action priors: no noise (tau2-bench evaluator replays tool calls
# and expects deterministic results; injecting timeout on writes causes
# state mismatch during evaluation)
WRITE_ACTION_PRIORS = NO_NOISE_PRIORS

# ============================================================================
# RETAIL DOMAIN TOOL CONFIGURATIONS (18 tools)
# ============================================================================

RETAIL_TOOL_CONFIGS: Dict[str, ToolConfig] = {
    # Meta tools (3)
    "calculate": ToolConfig(
        name="calculate",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "think": ToolConfig(
        name="think",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "transfer_to_human_agents": ToolConfig(
        name="transfer_to_human_agents",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    
    # Read tools with alternatives (2)
    "find_user_id_by_name_zip": ToolConfig(
        name="find_user_id_by_name_zip",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["find_user_id_by_email"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "find_user_id_by_email": ToolConfig(
        name="find_user_id_by_email",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["find_user_id_by_name_zip"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    
    # Read tools without alternatives (4)
    "get_order_details": ToolConfig(
        name="get_order_details",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_orders_for_user"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_product_details": ToolConfig(
        name="get_product_details",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["search_product_by_name"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_user_details": ToolConfig(
        name="get_user_details",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "list_all_product_types": ToolConfig(
        name="list_all_product_types",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_item_details": ToolConfig(
        name="get_item_details",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_item_by_product_and_options"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    
    # Write actions (8) - no noise
    "cancel_pending_order": ToolConfig(
        name="cancel_pending_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "exchange_delivered_order_items": ToolConfig(
        name="exchange_delivered_order_items",
        is_write_action=True,
        has_alternatives=True,
        alternatives=["place_new_order"],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "modify_pending_order_address": ToolConfig(
        name="modify_pending_order_address",
        is_write_action=True,
        has_alternatives=True,
        alternatives=["place_new_order"],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "modify_pending_order_items": ToolConfig(
        name="modify_pending_order_items",
        is_write_action=True,
        has_alternatives=True,
        alternatives=["place_new_order"],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "modify_pending_order_payment": ToolConfig(
        name="modify_pending_order_payment",
        is_write_action=True,
        has_alternatives=True,
        alternatives=["place_new_order"],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "modify_user_address": ToolConfig(
        name="modify_user_address",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "return_delivered_order_items": ToolConfig(
        name="return_delivered_order_items",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "place_new_order": ToolConfig(
        name="place_new_order",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),

    # New alternative read tools (injection controlled dynamically by UncertaintyEnvironment)
    "get_orders_for_user": ToolConfig(
        name="get_orders_for_user",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_order_details"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "search_product_by_name": ToolConfig(
        name="search_product_by_name",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_product_details"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_item_by_product_and_options": ToolConfig(
        name="get_item_by_product_and_options",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_item_details"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_user_by_order": ToolConfig(
        name="get_user_by_order",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["find_user_id_by_name_zip", "find_user_id_by_email"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_order_item_availability": ToolConfig(
        name="check_order_item_availability",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_product_details", "get_order_details"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
}

# ============================================================================
# AIRLINE DOMAIN TOOL CONFIGURATIONS (14 tools)
# ============================================================================

AIRLINE_TOOL_CONFIGS: Dict[str, ToolConfig] = {
    # Meta tools (3)
    "calculate": ToolConfig(
        name="calculate",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "think": ToolConfig(
        name="think",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "transfer_to_human_agents": ToolConfig(
        name="transfer_to_human_agents",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    
    # Read tools with alternatives (4)
    "search_direct_flight": ToolConfig(
        name="search_direct_flight",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["search_onestop_flight"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "search_onestop_flight": ToolConfig(
        name="search_onestop_flight",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["search_direct_flight"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    
    # Read tools without alternatives (4)
    "get_reservation_details": ToolConfig(
        name="get_reservation_details",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_all_reservations_for_user"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_all_reservations_for_user": ToolConfig(
        name="get_all_reservations_for_user",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_reservation_details"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_user_details": ToolConfig(
        name="get_user_details",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["find_user_by_email"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "list_all_airports": ToolConfig(
        name="list_all_airports",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    
    # Write actions (6) - no noise
    "book_reservation": ToolConfig(
        name="book_reservation",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "cancel_reservation": ToolConfig(
        name="cancel_reservation",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "send_certificate": ToolConfig(
        name="send_certificate",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "update_reservation_baggages": ToolConfig(
        name="update_reservation_baggages",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "update_reservation_flights": ToolConfig(
        name="update_reservation_flights",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "update_reservation_passengers": ToolConfig(
        name="update_reservation_passengers",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "get_flight_status": ToolConfig(
        name="get_flight_status",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_flight_details"],
        uncertainty_priors=DEFAULT_PRIORS
    ),

    # New alternative tools
    "find_user_by_email": ToolConfig(
        name="find_user_by_email",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_user_details"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "get_flight_details": ToolConfig(
        name="get_flight_details",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_flight_status"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
}

# ============================================================================
# TELECOM DOMAIN TOOL CONFIGURATIONS (13 tools)
# ============================================================================

TELECOM_TOOL_CONFIGS: Dict[str, ToolConfig] = {
    # Meta tools (1)
    "transfer_to_human_agents": ToolConfig(
        name="transfer_to_human_agents",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=NO_NOISE_PRIORS
    ),

    # Read tools with alternatives (3 + 1 new)
    "get_customer_by_phone": ToolConfig(
        name="get_customer_by_phone",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_customer_by_id", "get_customer_by_name", "get_customer_by_email"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_customer_by_id": ToolConfig(
        name="get_customer_by_id",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_customer_by_phone", "get_customer_by_name", "get_customer_by_email"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_customer_by_name": ToolConfig(
        name="get_customer_by_name",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_customer_by_phone", "get_customer_by_id", "get_customer_by_email"],
        uncertainty_priors=DEFAULT_PRIORS
    ),

    # Read tools with new alternatives
    "get_details_by_id": ToolConfig(
        name="get_details_by_id",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_all_lines_for_customer", "check_line_status", "list_all_plans"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_bills_for_customer": ToolConfig(
        name="get_bills_for_customer",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_bill_by_period"],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "get_data_usage": ToolConfig(
        name="get_data_usage",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_data_usage_summary"],
        uncertainty_priors=DEFAULT_PRIORS
    ),

    # Write actions (6) - no noise
    "suspend_line": ToolConfig(
        name="suspend_line",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "resume_line": ToolConfig(
        name="resume_line",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "send_payment_request": ToolConfig(
        name="send_payment_request",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "enable_roaming": ToolConfig(
        name="enable_roaming",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "disable_roaming": ToolConfig(
        name="disable_roaming",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "refuel_data": ToolConfig(
        name="refuel_data",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),

    # ---- User-side (device) tools ----

    # Read tools (user device diagnostics)
    "check_status_bar": ToolConfig(
        name="check_status_bar",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_network_status": ToolConfig(
        name="check_network_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_network_mode_preference": ToolConfig(
        name="check_network_mode_preference",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "run_speed_test": ToolConfig(
        name="run_speed_test",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_sim_status": ToolConfig(
        name="check_sim_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_data_restriction_status": ToolConfig(
        name="check_data_restriction_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_apn_settings": ToolConfig(
        name="check_apn_settings",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_wifi_status": ToolConfig(
        name="check_wifi_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_wifi_calling_status": ToolConfig(
        name="check_wifi_calling_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_vpn_status": ToolConfig(
        name="check_vpn_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_installed_apps": ToolConfig(
        name="check_installed_apps",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_app_status": ToolConfig(
        name="check_app_status",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_app_permissions": ToolConfig(
        name="check_app_permissions",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "can_send_mms": ToolConfig(
        name="can_send_mms",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),
    "check_payment_request": ToolConfig(
        name="check_payment_request",
        is_write_action=False,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=DEFAULT_PRIORS
    ),

    # Write tools (user device actions) - no noise
    "set_network_mode_preference": ToolConfig(
        name="set_network_mode_preference",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "toggle_airplane_mode": ToolConfig(
        name="toggle_airplane_mode",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "reseat_sim_card": ToolConfig(
        name="reseat_sim_card",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "toggle_data": ToolConfig(
        name="toggle_data",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "toggle_roaming": ToolConfig(
        name="toggle_roaming",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "toggle_data_saver_mode": ToolConfig(
        name="toggle_data_saver_mode",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "set_apn_settings": ToolConfig(
        name="set_apn_settings",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "reset_apn_settings": ToolConfig(
        name="reset_apn_settings",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "toggle_wifi": ToolConfig(
        name="toggle_wifi",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "toggle_wifi_calling": ToolConfig(
        name="toggle_wifi_calling",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "connect_vpn": ToolConfig(
        name="connect_vpn",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "disconnect_vpn": ToolConfig(
        name="disconnect_vpn",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "grant_app_permission": ToolConfig(
        name="grant_app_permission",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "reboot_device": ToolConfig(
        name="reboot_device",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),
    "make_payment": ToolConfig(
        name="make_payment",
        is_write_action=True,
        has_alternatives=False,
        alternatives=[],
        uncertainty_priors=WRITE_ACTION_PRIORS
    ),

    # New alternative tools (always clean)
    "get_customer_by_email": ToolConfig(
        name="get_customer_by_email",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "get_all_lines_for_customer": ToolConfig(
        name="get_all_lines_for_customer",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_details_by_id"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "get_bill_by_period": ToolConfig(
        name="get_bill_by_period",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_bills_for_customer"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "get_data_usage_summary": ToolConfig(
        name="get_data_usage_summary",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_data_usage"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "check_line_status": ToolConfig(
        name="check_line_status",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_details_by_id"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
    "list_all_plans": ToolConfig(
        name="list_all_plans",
        is_write_action=False,
        has_alternatives=True,
        alternatives=["get_details_by_id"],
        uncertainty_priors=NO_NOISE_PRIORS
    ),
}
