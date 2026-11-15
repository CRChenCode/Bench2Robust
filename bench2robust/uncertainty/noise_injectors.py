# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Noise injection implementations for uncertainty simulation.

This module implements nine types of uncertainty injection with deterministic,
seeded randomness.

Design principle for partial / stale / factual_error:
  These three injectors are *contextual* — they corrupt the fields that the agent
  will actually use to make its next decision, rather than random structural noise
  that the agent can safely ignore.  The tool_name and tool_kwargs passed to
  inject() are used to identify decision-critical fields for each call.
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import random
from copy import deepcopy
from bench2robust.uncertainty.config import NoiseType
from bench2robust.uncertainty.domain_registry import DomainRegistry


# ---------------------------------------------------------------------------
# Decision-critical field registry
# ---------------------------------------------------------------------------

# Maps tool_name → list of field paths (dot-separated) that the agent reads to
# decide what to do next.  These are the fields that *must* be corrupted for the
# injection to actually influence the agent's behaviour.
#
# A field path like "items.product_id" means response["items"][*]["product_id"].
# A plain key like "status" means response["status"].

_RETAIL_CRITICAL_FIELDS: Dict[str, List[str]] = {
    # The agent uses order status to decide whether to cancel/exchange/return,
    # items to know what products are in the order, and fulfillments to infer
    # delivery state.
    "get_order_details": ["status", "items", "fulfillments", "payment_history"],
    # The agent uses variant availability and price to decide whether to
    # recommend an exchange item.
    "get_product_details": ["variants"],
    # The agent uses user payment methods and orders to propose modifications.
    "get_user_details": ["payment_methods", "orders"],
    # The product type list is the whole payload; truncating it is the injection.
    "list_all_product_types": [],
    # User-lookup tools: the returned id is the only thing that matters.
    "find_user_id_by_name_zip": ["user_id"],
    "find_user_id_by_email": ["user_id"],
}

_AIRLINE_CRITICAL_FIELDS: Dict[str, List[str]] = {
    "get_reservation_details": ["status", "flights", "passengers", "payment_methods"],
    "get_user_details": ["payment_methods", "addresses", "reservations"],
    "search_direct_flight": ["price", "available_seats", "departure_time", "arrival_time"],
    "search_onestop_flight": ["price", "available_seats", "stops"],
    "list_all_airports": [],
}

_CRITICAL_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "retail": _RETAIL_CRITICAL_FIELDS,
    "airline": _AIRLINE_CRITICAL_FIELDS,
}


def _get_critical_fields(tool_name: str, domain: str) -> List[str]:
    """Return decision-critical fields from the extensible domain registry."""
    registered = DomainRegistry.get_critical_fields(domain, tool_name)
    if registered:
        return registered
    return _CRITICAL_FIELDS.get(domain, {}).get(tool_name, [])


class NoiseInjector(ABC):
    """Abstract base class for noise injectors."""
    
    def __init__(self, seed: int):
        """
        Initialize injector with seeded random number generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
    
    @abstractmethod
    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> Any:
        """
        Inject noise into a clean response.

        Args:
            clean_response: Original response from tool
            tool_name: Name of the tool (for context)
            **kwargs:
                tool_kwargs (dict): The arguments the agent passed to the tool.
                    Used by contextual injectors to identify decision-critical
                    fields.
                response_cache: Required by StaleResponseInjector.
        
        Returns:
            Noisy response
        """
        pass


class TimeoutInjector(NoiseInjector):
    """Injects timeout errors."""
    
    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> str:
        """
        Return timeout error message.
        
        Args:
            clean_response: Ignored (timeout prevents execution)
            tool_name: Name of the tool
        
        Returns:
            Timeout error message in Tau-Bench format
        """
        return "Error: request timed out"


class RateLimitInjector(NoiseInjector):
    """
    Injects HTTP 429 rate-limit errors.

    The correct agent response is to retry after a short delay.  The error is
    transient and identical to timeout in that no data is returned, but the
    semantic signal is different: the system is overloaded, not unresponsive.
    """

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> str:
        return (
            "Error: HTTP 429 Too Many Requests. The service rejected the call "
            "because the per-minute rate limit has been exceeded. Please retry "
            "after a brief wait."
        )


class ServerErrorInjector(NoiseInjector):
    """
    Injects HTTP 5xx server errors.

    The correct agent response is to retry (transient backend failure).
    Unlike timeout, the server explicitly acknowledged receipt but failed
    to process the request.
    """

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> str:
        # Randomly select a specific 5xx code for variety
        code = self.rng.choice([500, 502, 503])
        messages = {
            500: "Internal Server Error. The remote endpoint failed to handle the request.",
            502: "Bad Gateway. The upstream service returned an invalid response.",
            503: "Service Unavailable. The service is temporarily overloaded or under maintenance.",
        }
        return f"Error: HTTP {code} {messages[code]}"


class AuthErrorInjector(NoiseInjector):
    """
    Injects HTTP 401/403 authentication/authorization errors.

    Unlike timeout/rate-limit/5xx, retrying with the same parameters is
    unlikely to help — the credentials are invalid or expired.  The correct
    agent response is to report the issue to the user or escalate, NOT to
    blindly retry.
    """

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> str:
        code = self.rng.choice([401, 403])
        if code == 401:
            return (
                "Error: HTTP 401 Unauthorized. The supplied credentials are "
                "invalid or have expired. Unable to authenticate the request."
            )
        else:
            return (
                "Error: HTTP 403 Forbidden. The current credentials do not have "
                "permission to access this resource."
            )


class MalformedResponseInjector(NoiseInjector):
    """
    Injects malformed/unparseable response errors.

    The tool executed but returned garbage that could not be parsed as valid
    JSON.  The correct agent response is to retry (transient serialization
    failure on the server side).
    """

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> str:
        return (
            "Error: Malformed response from tool execution. The response body "
            "could not be parsed as valid JSON. Raw content: '{\"status\": \"ok\", "
            "\"data\": {truncated...'"
        )


class SchemaDriftInjector(NoiseInjector):
    """
    Injects schema validation failures.

    The tool returned a structurally valid response, but it does not match the
    declared output schema — fields are missing, renamed, or have unexpected
    types.  This simulates API version drift where the backend was updated but
    the tool definition was not.

    The correct agent response depends on context:
    - If a single field is invalid, retry may work (transient).
    - If the schema fundamentally changed, the agent should report the issue.
    """

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> str:
        # Pick a plausible field name based on the tool being called
        domain = kwargs.get("domain", "retail")
        critical = _get_critical_fields(tool_name, domain)
        if critical:
            field = self.rng.choice(critical)
        else:
            field = "result"

        variant = self.rng.choice(["missing", "type_mismatch", "renamed"])
        if variant == "missing":
            return (
                f"Error: Schema validation failed. Required field '{field}' is "
                f"missing from the response. The tool's output schema may have "
                f"changed."
            )
        elif variant == "type_mismatch":
            return (
                f"Error: Schema validation failed. Field '{field}' expected type "
                f"'object' but received 'string'. The response did not match the "
                f"tool's declared output schema."
            )
        else:
            return (
                f"Error: Schema validation failed. Unknown field '{field}_v2' "
                f"present in response; expected '{field}'. The tool's output "
                f"schema may have been updated."
            )


class PartialResponseInjector(NoiseInjector):
    """
    Injects partial/truncated responses.

    Harder mode: instead of dropping keys (which the agent recognises as missing
    data and retries), critical fields are replaced with empty containers
    ([], {}, "") so the response looks structurally complete but semantically
    empty.  The agent is more likely to interpret this as "no data exists" and
    proceed with a wrong decision rather than retry.
    """

    # Maps field name → the empty sentinel value to substitute
    _EMPTY_SENTINELS: Dict[str, Any] = {
        # Lists → empty list  (agent sees "no items", "no orders", etc.)
        "items": [],
        "fulfillments": [],
        "orders": [],
        "payment_history": [],
        "flights": [],
        "passengers": [],
        "reservations": [],
        "addresses": [],
        "stops": [],
        # Dicts → empty dict  (agent sees "no variants", "no payment methods")
        "variants": {},
        "payment_methods": {},
        # Strings → empty string  (agent sees blank status)
        "status": "",
        # Numbers → zero  (agent sees 0 seats / 0 price)
        "available_seats": 0,
        "price": 0.0,
        "departure_time": "",
        "arrival_time": "",
    }

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> Any:
        """
        Replace decision-critical fields with empty containers.

        Critical fields get an empty sentinel ([], {}, 0, "").
        Non-critical fields are left intact so the response looks plausible.
        """
        if clean_response is None:
            return None

        domain = kwargs.get("domain", "retail")
        critical = set(_get_critical_fields(tool_name, domain))

        response = deepcopy(clean_response)

        if isinstance(response, dict):
            return self._empty_critical(response, critical, domain)
        elif isinstance(response, list):
            # For list responses (e.g. list_all_product_types returns a dict,
            # but guard anyway): truncate to ~30% so most entries are missing
            if not response:
                return response
            cutoff = max(1, int(len(response) * self.rng.uniform(0.2, 0.4)))
            return response[:cutoff]
        return response

    def _empty_critical(self, obj: Dict, critical: set, domain: str) -> Dict:
        """
        Replace critical fields with empty sentinels; leave others intact.

        For critical fields not in the sentinel map, fall back to None so the
        agent still sees the key but with a clearly invalid value.
        """
        result = {}
        critical_present = [k for k in obj if k in critical]

        # Always empty ALL critical fields present in this response
        fields_to_empty = set(critical_present)

        # Also lightly corrupt 0-1 non-critical fields for cosmetic noise
        non_critical = [k for k in obj if k not in critical]
        if non_critical and self.rng.random() < 0.3:
            fields_to_empty.add(self.rng.choice(non_critical))

        for key, value in obj.items():
            if key in fields_to_empty:
                sentinel = DomainRegistry.get_empty_sentinel(domain, key)
                if sentinel is None:
                    sentinel = self._EMPTY_SENTINELS.get(key)
                if sentinel is not None:
                    result[key] = deepcopy(sentinel)
                else:
                    # Unknown critical field: use type-matched empty or None
                    if isinstance(value, list):
                        result[key] = []
                    elif isinstance(value, dict):
                        result[key] = {}
                    elif isinstance(value, str):
                        result[key] = ""
                    elif isinstance(value, (int, float)):
                        result[key] = 0
                    else:
                        result[key] = None
            else:
                result[key] = value

        return result


class StaleResponseInjector(NoiseInjector):
    """
    Injects stale (cached) responses.

    Harder mode: the stale response keeps all structural identifiers intact
    (order_id, reservation_id, user_id) so the agent cannot detect the staleness
    by cross-checking IDs.  Only the decision-critical *state* fields are
    back-dated to a plausible prior value that will cause the agent to choose
    the wrong action type.

    Examples:
      - get_order_details: status "delivered" → "pending", items/fulfillments
        cleared so the agent thinks the order hasn't shipped and tries to cancel.
      - get_product_details: all variants marked unavailable so the agent aborts
        an exchange it should complete.
      - get_user_details: payment_methods emptied so the agent can't confirm
        which card to charge; orders list cleared so it can't find the order.
    """

    # Maps domain → tool_name → field → callable(rng, current_value) → stale_value
    # Structural identity fields (order_id, user_id, etc.) are intentionally
    # NOT listed here so they pass through unchanged.
    _STALE_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
        "retail": {
            "get_order_details": {
                # Back-date status: delivered/shipped → pending/processing
                # so the agent tries to cancel instead of return/exchange.
                "status": lambda rng, v: (
                    rng.choice(["pending", "processing"])
                    if isinstance(v, str) and v in ("delivered", "shipped", "return_processed")
                    else ("pending" if isinstance(v, str) else v)
                ),
                # Clear fulfillments so the agent can't infer delivery from tracking.
                "fulfillments": lambda rng, v: [],
                # Clear items so the agent can't verify what's in the order.
                "items": lambda rng, v: [],
                # Wipe payment history so refund eligibility looks unclear.
                "payment_history": lambda rng, v: [],
            },
            "get_product_details": {
                # Mark ALL variants unavailable → agent aborts exchange.
                "variants": lambda rng, v: {
                    k: dict(variant, available=False)
                    for k, variant in v.items()
                } if isinstance(v, dict) else v,
            },
            "get_user_details": {
                # Empty payment methods → agent can't confirm refund target.
                "payment_methods": lambda rng, v: {},
                # Empty orders list → agent can't locate the order.
                "orders": lambda rng, v: [],
            },
            "find_user_id_by_name_zip": {
                # Return a plausible-looking but wrong user id.
                # Keep the name prefix so it doesn't look obviously fake.
                "user_id": lambda rng, v: (
                    "_".join(v.split("_")[:-1]) + f"_{rng.randint(1000, 9999)}"
                    if isinstance(v, str) and "_" in v
                    else f"stale_user_{rng.randint(1000, 9999)}"
                ),
            },
            "find_user_id_by_email": {
                "user_id": lambda rng, v: (
                    "_".join(v.split("_")[:-1]) + f"_{rng.randint(1000, 9999)}"
                    if isinstance(v, str) and "_" in v
                    else f"stale_user_{rng.randint(1000, 9999)}"
                ),
            },
        },
        "airline": {
            "get_reservation_details": {
                # Airline reservations have NO status field.
                # Strategy: replace reservation_id with another reservation from
                # the same user (from response_cache) so the agent operates on
                # the wrong booking.  Also inject a fake "status": "cancelled"
                # field so the agent thinks the target reservation is already
                # closed and skips the required action.
                # These rules are applied via _inject_airline_reservation() below
                # rather than the generic field-rule loop.
            },
            "get_user_details": {
                # Empty payment methods → agent can't confirm payment target.
                "payment_methods": lambda rng, v: {},
                # Empty reservations list → agent can't find the booking.
                "reservations": lambda rng, v: [],
            },
            "search_direct_flight": {
                # Show 0 seats → agent aborts a booking that's actually available.
                "available_seats": lambda rng, v: 0
                    if isinstance(v, int) else v,
            },
            "search_onestop_flight": {
                "available_seats": lambda rng, v: 0
                    if isinstance(v, int) else v,
            },
        },
    }

    def _inject_airline_reservation(
        self, clean: Dict, response_cache: Any
    ) -> Dict:
        """
        Stale injection for get_reservation_details.

        Two simultaneous corruptions:
        1. Replace reservation_id with a different reservation cached from an
           earlier get_reservation_details call (if available), so the agent
           thinks it's looking at the right booking but is actually reading a
           different one.
        2. Add a synthetic "status": "cancelled" field so the agent believes
           the reservation is already closed and skips the required action
           (cancel/modify/update).
        """
        stale = deepcopy(clean)

        # Corruption 1: swap reservation_id with a cached one
        if response_cache is not None:
            cached = response_cache.get_cached_response("get_reservation_details")
            if cached and isinstance(cached, dict):
                cached_id = cached.get("reservation_id")
                current_id = stale.get("reservation_id")
                if cached_id and cached_id != current_id:
                    # Swap: agent sees a different reservation's ID but the
                    # current reservation's other fields (flights, passengers)
                    # so the data looks internally consistent but points to
                    # the wrong booking.
                    stale["reservation_id"] = cached_id

        # Corruption 2: inject fake "cancelled" status
        # (real reservations don't have this field, so it looks like a
        #  system-added status that the agent will trust)
        stale["status"] = "cancelled"

        # Clear flights and passengers so the agent can't verify the itinerary
        stale["flights"] = []
        stale["passengers"] = []

        return stale

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> Any:
        """
        Return a stale version of the response.

        For airline get_reservation_details: uses the dedicated
        _inject_airline_reservation() method which swaps reservation_id and
        injects a fake "cancelled" status.

        For all other tools: applies domain-specific field-level back-dating
        rules, preserving structural identity fields.

        Falls back to the first cached response if no rule exists.
        """
        domain = kwargs.get("domain", "retail")
        response_cache = kwargs.get("response_cache")

        # Special path: airline reservation details
        if domain == "airline" and tool_name == "get_reservation_details":
            if isinstance(clean_response, dict):
                return self._inject_airline_reservation(clean_response, response_cache)

        tool_rules = DomainRegistry.get_stale_rules(domain, tool_name)
        if tool_rules is None:
            tool_rules = self._STALE_RULES.get(domain, {}).get(tool_name)

        if tool_rules and isinstance(clean_response, dict):
            stale = deepcopy(clean_response)
            for field, rule_fn in tool_rules.items():
                if field in stale:
                    stale[field] = rule_fn(self.rng, stale[field])
            return stale

        # Fallback: return first cached response
        if response_cache is not None:
            cached = response_cache.get_cached_response(tool_name)
            if cached is not None:
                return cached

        return clean_response


class FactualErrorInjector(NoiseInjector):
    """
    Injects factual errors by corrupting decision-critical field values.

    Harder mode: errors are designed to be *plausible* — small enough that the
    agent won't immediately dismiss them as obviously wrong, but large enough to
    cause an incorrect decision.

    Key changes vs. the original:
    - status: adjacent-state corruption only (pending→processing, delivered→shipped)
      so the agent can't detect the error by checking whether the status makes sense.
    - price: ±10-25% shift (not ±30-100%) so the agent doesn't notice the
      discrepancy when comparing with other data sources.
    - available: flip only when the variant is currently available=True, so the
      agent believes a valid exchange target is out of stock.
    - items/variants: corrupt item_ids and option values (not just prices) so the
      agent selects the wrong product for an exchange/return.
    """

    # Adjacent-state transition maps: each status maps to its most plausible
    # "prior" state so the corruption looks like a realistic race condition.
    _RETAIL_STATUS_TRANSITIONS = {
        "delivered":         "shipped",
        "shipped":           "processing",
        "processing":        "pending",
        "pending":           "processing",   # forward confusion
        "cancelled":         "pending",
        "return_processed":  "delivered",
    }

    _AIRLINE_STATUS_TRANSITIONS = {
        "cancelled":  "confirmed",
        "completed":  "confirmed",
        "confirmed":  "pending",
        "pending":    "confirmed",
        "delayed":    "on_time",
        "on_time":    "delayed",
    }

    RETAIL_CORRUPTION_RULES = {
        # Price: small plausible shift (±10-25%) — hard to detect by cross-check
        'price': lambda rng, v: round(
            v * rng.choice([rng.uniform(0.75, 0.90), rng.uniform(1.10, 1.25)]), 2
        ) if isinstance(v, (int, float)) else v,
        'cost': lambda rng, v: round(
            v * rng.choice([rng.uniform(0.75, 0.90), rng.uniform(1.10, 1.25)]), 2
        ) if isinstance(v, (int, float)) else v,
        'total': lambda rng, v: round(
            v * rng.choice([rng.uniform(0.75, 0.90), rng.uniform(1.10, 1.25)]), 2
        ) if isinstance(v, (int, float)) else v,
        # available: flip True→False only (agent thinks item is out of stock)
        'available': lambda rng, v: False if v is True else v,
        'in_stock':  lambda rng, v: False if v is True else v,
        # status: adjacent-state transition (defined in __init__ via closure)
        'status': None,   # replaced in __init__
        # items: corrupt item_id and one option value per item
        'items': lambda rng, v: [
            dict(
                item,
                item_id=item.get('item_id', 'X') + "_ERR" if rng.random() < 0.6 else item.get('item_id'),
                options={
                    k: (rng.choice([x for x in ['S', 'M', 'L', 'XL', 'blue', 'black', 'red', 'white'] if x != vv])
                        if isinstance(vv, str) and rng.random() < 0.5 else vv)
                    for k, vv in item.get('options', {}).items()
                } if item.get('options') else item.get('options'),
            )
            for item in v
        ] if isinstance(v, list) and v else v,
        # variants: flip available=True→False and apply small price shift
        'variants': lambda rng, v: {
            k: dict(
                variant,
                available=False if variant.get('available') is True else variant.get('available'),
                price=round(
                    variant['price'] * rng.choice([rng.uniform(0.80, 0.92), rng.uniform(1.08, 1.22)]), 2
                ) if 'price' in variant else variant.get('price'),
            )
            for k, variant in v.items()
        } if isinstance(v, dict) else v,
        # payment_methods: corrupt last_four digits (looks like a different card)
        'payment_methods': lambda rng, v: {
            k: dict(pm, last_four=str(rng.randint(1000, 9999)))
            if isinstance(pm, dict) and 'last_four' in pm else pm
            for k, pm in v.items()
        } if isinstance(v, dict) else v,
    }

    AIRLINE_CORRUPTION_RULES = {
        'price': lambda rng, v: round(
            v * rng.choice([rng.uniform(0.80, 0.92), rng.uniform(1.08, 1.22)]), 2
        ) if isinstance(v, (int, float)) else v,
        'fare': lambda rng, v: round(
            v * rng.choice([rng.uniform(0.80, 0.92), rng.uniform(1.08, 1.22)]), 2
        ) if isinstance(v, (int, float)) else v,
        # available_seats: small shift (±1-5) so agent may think it's nearly full
        'available_seats': lambda rng, v: max(0, v + rng.randint(-5, -1))
            if isinstance(v, int) and v > 0 else v,
        'status': None,   # replaced in __init__
        # cabin_class: adjacent class (economy↔business, not economy→first)
        'cabin_class': lambda rng, v: (
            'business' if v == 'economy' else
            'economy'  if v == 'business' else
            'business' if v == 'first' else v
        ) if isinstance(v, str) else v,
    }

    def __init__(self, seed: int, domain: str = "retail"):
        super().__init__(seed)
        self.domain = domain

        # Wire up the status corruption rule with the correct transition map
        if domain == "retail":
            transitions = self._RETAIL_STATUS_TRANSITIONS
            rules = dict(self.RETAIL_CORRUPTION_RULES)
        elif domain == "airline":
            transitions = self._AIRLINE_STATUS_TRANSITIONS
            rules = dict(self.AIRLINE_CORRUPTION_RULES)
        else:
            transitions = DomainRegistry.get_status_transitions(domain)
            rules = dict(DomainRegistry.get_factual_error_rules(domain))

        if transitions:
            rules['status'] = lambda rng, v, _t=transitions: (
                _t.get(v, v) if isinstance(v, str) else v
            )
        self.corruption_rules = rules

    def inject(self, clean_response: Any, tool_name: str, **kwargs) -> Any:
        """
        Corrupt decision-critical fields with plausible adjacent-state errors.
        """
        if clean_response is None:
            return None

        domain = kwargs.get("domain", self.domain)
        critical = set(_get_critical_fields(tool_name, domain))

        response = deepcopy(clean_response)

        if isinstance(response, dict):
            return self._corrupt_dict(response, critical)
        elif isinstance(response, list) and response:
            num_to_corrupt = min(len(response), self.rng.randint(1, 3))
            indices = self.rng.sample(range(len(response)), num_to_corrupt)
            for idx in indices:
                if isinstance(response[idx], dict):
                    response[idx] = self._corrupt_dict(response[idx], critical)

        return response

    def _corrupt_dict(self, obj: Dict, critical: set) -> Dict:
        """
        Corrupt ALL critical fields that have a rule (guaranteed decision impact),
        then optionally corrupt one non-critical field for cosmetic noise.
        """
        result = obj.copy()

        # Priority: corrupt every critical field that has a rule
        critical_present = [k for k in obj if k in critical and k in self.corruption_rules]
        if critical_present:
            for field in critical_present:
                result[field] = self.corruption_rules[field](self.rng, obj[field])
            return result

        # Fallback: generic field-name matching
        corruptible = [k for k in obj if k in self.corruption_rules]
        if not corruptible:
            return obj

        num_to_corrupt = min(len(corruptible), self.rng.randint(1, 2))
        for field in self.rng.sample(corruptible, num_to_corrupt):
            result[field] = self.corruption_rules[field](self.rng, obj[field])

        return result


class NoiseInjectorFactory:
    """Factory for creating noise injectors."""
    
    @staticmethod
    def create_injector(
        noise_type: NoiseType,
        seed: int,
        domain: str = "retail"
    ) -> NoiseInjector:
        """
        Create appropriate injector for noise type.
        
        Args:
            noise_type: Type of noise to inject
            seed: Random seed
            domain: Domain for domain-specific injectors
        
        Returns:
            NoiseInjector instance
        """
        if noise_type == NoiseType.TIMEOUT:
            return TimeoutInjector(seed)
        elif noise_type == NoiseType.RATE_LIMIT:
            return RateLimitInjector(seed)
        elif noise_type == NoiseType.SERVER_ERROR:
            return ServerErrorInjector(seed)
        elif noise_type == NoiseType.AUTH_ERROR:
            return AuthErrorInjector(seed)
        elif noise_type == NoiseType.MALFORMED_RESPONSE:
            return MalformedResponseInjector(seed)
        elif noise_type == NoiseType.SCHEMA_DRIFT:
            return SchemaDriftInjector(seed)
        elif noise_type == NoiseType.PARTIAL:
            return PartialResponseInjector(seed)
        elif noise_type == NoiseType.STALE:
            return StaleResponseInjector(seed)
        elif noise_type == NoiseType.FACTUAL_ERROR:
            return FactualErrorInjector(seed, domain)
        else:
            raise ValueError(f"No injector for noise type: {noise_type}")
