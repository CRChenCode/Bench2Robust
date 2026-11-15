# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Domain Registry: Externalized domain-specific knowledge for noise injection.

This module decouples domain-specific corruption rules from the injection engine,
enabling new benchmarks to register their own critical fields, stale rules, and
factual error corruption patterns without modifying the core injectors.

Usage:
    from bench2robust.uncertainty.domain_registry import DomainRegistry

    # Register a new domain
    DomainRegistry.register_critical_fields("vitabench_delivery", {
        "get_delivery_order_detail": ["status", "products", "total_price"],
        "get_delivery_store_info": ["score", "tags", "products"],
    })

    DomainRegistry.register_stale_rules("vitabench_delivery", {
        "get_delivery_order_detail": {
            "status": lambda rng, v: "unpaid" if v == "processed" else v,
            "products": lambda rng, v: [],
        },
    })
"""

from typing import Any, Callable, Dict, List, Optional
import random


# Type aliases for clarity
FieldList = List[str]
StaleRule = Callable[[random.Random, Any], Any]
CorruptionRule = Callable[[random.Random, Any], Any]


class DomainRegistry:
    """
    Centralized registry for domain-specific injection knowledge.
    
    Benchmarks register their domain knowledge here; the core injectors
    query this registry at runtime rather than using hardcoded dicts.
    """
    
    # Domain → tool_name → list of decision-critical field names
    _critical_fields: Dict[str, Dict[str, FieldList]] = {}
    
    # Domain → tool_name → field → stale_rule_fn(rng, current_value) → stale_value
    _stale_rules: Dict[str, Dict[str, Dict[str, StaleRule]]] = {}
    
    # Domain → field_name → corruption_rule_fn(rng, current_value) → corrupted_value
    _factual_error_rules: Dict[str, Dict[str, CorruptionRule]] = {}
    
    # Domain → field_name → empty_sentinel_value
    _empty_sentinels: Dict[str, Dict[str, Any]] = {}
    
    # Domain → status_transitions (for factual error adjacent-state corruption)
    _status_transitions: Dict[str, Dict[str, str]] = {}
    
    @classmethod
    def register_critical_fields(cls, domain: str, fields: Dict[str, FieldList]) -> None:
        """
        Register decision-critical fields for a domain.
        
        Args:
            domain: Domain identifier (e.g., "retail", "vitabench_delivery")
            fields: Mapping of tool_name → list of critical field names
        """
        if domain not in cls._critical_fields:
            cls._critical_fields[domain] = {}
        cls._critical_fields[domain].update(fields)
    
    @classmethod
    def register_stale_rules(cls, domain: str, rules: Dict[str, Dict[str, StaleRule]]) -> None:
        """
        Register stale response corruption rules for a domain.
        
        Args:
            domain: Domain identifier
            rules: Mapping of tool_name → field → stale_rule_fn
        """
        if domain not in cls._stale_rules:
            cls._stale_rules[domain] = {}
        cls._stale_rules[domain].update(rules)
    
    @classmethod
    def register_factual_error_rules(cls, domain: str, rules: Dict[str, CorruptionRule]) -> None:
        """
        Register factual error corruption rules for a domain.
        
        Args:
            domain: Domain identifier
            rules: Mapping of field_name → corruption_rule_fn
        """
        if domain not in cls._factual_error_rules:
            cls._factual_error_rules[domain] = {}
        cls._factual_error_rules[domain].update(rules)
    
    @classmethod
    def register_empty_sentinels(cls, domain: str, sentinels: Dict[str, Any]) -> None:
        """
        Register empty sentinel values for partial response injection.
        
        Args:
            domain: Domain identifier
            sentinels: Mapping of field_name → empty sentinel value
        """
        if domain not in cls._empty_sentinels:
            cls._empty_sentinels[domain] = {}
        cls._empty_sentinels[domain].update(sentinels)
    
    @classmethod
    def register_status_transitions(cls, domain: str, transitions: Dict[str, str]) -> None:
        """
        Register status transition map for factual error injection.
        
        Args:
            domain: Domain identifier
            transitions: Mapping of current_status → adjacent_stale_status
        """
        cls._status_transitions[domain] = transitions
    
    @classmethod
    def get_critical_fields(cls, domain: str, tool_name: str) -> FieldList:
        """Get critical fields for a (domain, tool) pair."""
        return cls._critical_fields.get(domain, {}).get(tool_name, [])
    
    @classmethod
    def get_stale_rules(cls, domain: str, tool_name: str) -> Optional[Dict[str, StaleRule]]:
        """Get stale rules for a (domain, tool) pair."""
        return cls._stale_rules.get(domain, {}).get(tool_name)
    
    @classmethod
    def get_factual_error_rules(cls, domain: str) -> Dict[str, CorruptionRule]:
        """Get factual error rules for a domain."""
        return cls._factual_error_rules.get(domain, {})
    
    @classmethod
    def get_empty_sentinel(cls, domain: str, field_name: str) -> Optional[Any]:
        """Get empty sentinel for a field in a domain."""
        return cls._empty_sentinels.get(domain, {}).get(field_name)
    
    @classmethod
    def get_status_transitions(cls, domain: str) -> Dict[str, str]:
        """Get status transitions for a domain."""
        return cls._status_transitions.get(domain, {})
    
    @classmethod
    def has_domain(cls, domain: str) -> bool:
        """Check if a domain is registered."""
        return domain in cls._critical_fields
    
    @classmethod
    def list_domains(cls) -> List[str]:
        """List all registered domains."""
        all_domains = set()
        all_domains.update(cls._critical_fields.keys())
        all_domains.update(cls._stale_rules.keys())
        all_domains.update(cls._factual_error_rules.keys())
        return sorted(all_domains)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._critical_fields.clear()
        cls._stale_rules.clear()
        cls._factual_error_rules.clear()
        cls._empty_sentinels.clear()
        cls._status_transitions.clear()
