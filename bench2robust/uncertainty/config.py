# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Configuration system for the API Uncertainty Simulation Framework.

This module provides data structures for defining uncertainty behavior at both
the tool and episode levels.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Literal, Optional

from enum import Enum


class NoiseType(Enum):
    """Enumeration of supported uncertainty types."""
    CLEAN = "clean"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    AUTH_ERROR = "auth_error"
    MALFORMED_RESPONSE = "malformed_response"
    SCHEMA_DRIFT = "schema_drift"
    PARTIAL = "partial"
    STALE = "stale"
    FACTUAL_ERROR = "factual_error"
    NATURAL_ERROR = "natural_error"  # Not injected, but logged


class TriggerTiming(Enum):
    """When to inject uncertainty during episode."""
    EARLY = "early"      # First 33% of steps
    MIDDLE = "middle"    # Middle 33% of steps
    LATE = "late"        # Last 33% of steps
    UNIFORM = "uniform"  # Any time


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


@dataclass
class ToolConfig:
    """Configuration for a single tool."""
    name: str
    is_write_action: bool
    has_alternatives: bool
    alternatives: list = field(default_factory=list)
    uncertainty_priors: Dict[str, float] = field(default_factory=lambda: {
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
    })
    # Per-tool override for max consecutive failures; None means use episode-level default
    max_consecutive_failures: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate probabilities sum to 1.0
        total = sum(self.uncertainty_priors.values())
        if abs(total - 1.0) > 0.001:
            raise ConfigurationError(
                f"Tool {self.name}: probabilities sum to {total}, must be 1.0"
            )
        
        # Validate all probabilities are non-negative
        for noise_type, prob in self.uncertainty_priors.items():
            if prob < 0:
                raise ConfigurationError(
                    f"Tool {self.name}: {noise_type} probability {prob} < 0"
                )
        
        # Validate "clean" is present
        if "clean" not in self.uncertainty_priors:
            raise ConfigurationError(
                f"Tool {self.name}: must include 'clean' probability"
            )


@dataclass
class EpisodeUncertaintyConfig:
    """Episode-level uncertainty configuration."""
    tool_configs: Dict[str, ToolConfig]
    trigger_timing: TriggerTiming = TriggerTiming.UNIFORM
    max_consecutive_failures: int = 2
    max_total_injections: int = 5
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_consecutive_failures <= 0:
            raise ConfigurationError(
                f"max_consecutive_failures must be > 0, got {self.max_consecutive_failures}"
            )
        
        if self.max_total_injections < 0:
            raise ConfigurationError(
                f"max_total_injections must be >= 0, got {self.max_total_injections}"
            )
        
        if self.seed < 0:
            raise ConfigurationError(
                f"seed must be non-negative, got {self.seed}"
            )
        
        # Validate trigger timing
        if not isinstance(self.trigger_timing, TriggerTiming):
            raise ConfigurationError(
                f"trigger_timing must be TriggerTiming enum, got {type(self.trigger_timing)}"
            )
    
    def get_tool_config(self, tool_name: str) -> ToolConfig:
        """Get configuration for a specific tool."""
        if tool_name not in self.tool_configs:
            raise ConfigurationError(f"No configuration for tool: {tool_name}")
        return self.tool_configs[tool_name]
    
    def is_write_action(self, tool_name: str) -> bool:
        """Check if tool is a write action."""
        return self.get_tool_config(tool_name).is_write_action
    
    def has_alternatives(self, tool_name: str) -> bool:
        """Check if tool has alternatives."""
        return self.get_tool_config(tool_name).has_alternatives


class ConfigFactory:
    """Factory for creating domain-specific configurations."""
    
    @staticmethod
    def create_tau_bench_config(
        domain: Literal["retail", "airline", "telecom"],
        trigger_timing: TriggerTiming = TriggerTiming.UNIFORM,
        max_consecutive_failures: int = 2,
        max_total_injections: int = 5,
        seed: int = 42,
        custom_priors: Optional[Dict[str, Dict[str, float]]] = None
    ) -> EpisodeUncertaintyConfig:
        """
        Create configuration for a Tau-Bench domain.

        Args:
            domain: Supported Tau-Bench domain name
            trigger_timing: When to inject uncertainty
            max_consecutive_failures: Max consecutive failures per tool
            max_total_injections: Max total injections per episode
            seed: Random seed for reproducibility
            custom_priors: Optional per-tool probability overrides

        Returns:
            EpisodeUncertaintyConfig with independent tool configurations
        """
        from bench2robust.adapters.tau_bench.tool_configs import (
            RETAIL_TOOL_CONFIGS,
            AIRLINE_TOOL_CONFIGS,
            TELECOM_TOOL_CONFIGS,
        )

        configs_by_domain = {
            "retail": RETAIL_TOOL_CONFIGS,
            "airline": AIRLINE_TOOL_CONFIGS,
            "telecom": TELECOM_TOOL_CONFIGS,
        }
        if domain not in configs_by_domain:
            raise ConfigurationError(f"Unknown domain: {domain}")

        # Clone mutable fields so one experiment cannot modify the defaults used
        # by another experiment in the same process.
        tool_configs = {
            name: replace(
                config,
                alternatives=list(config.alternatives),
                uncertainty_priors=dict(config.uncertainty_priors),
            )
            for name, config in configs_by_domain[domain].items()
        }

        if custom_priors:
            unknown_tools = set(custom_priors) - set(tool_configs)
            if unknown_tools:
                names = ", ".join(sorted(unknown_tools))
                raise ConfigurationError(f"Unknown tools in custom_priors: {names}")
            for tool_name, priors in custom_priors.items():
                tool_configs[tool_name] = replace(
                    tool_configs[tool_name],
                    uncertainty_priors=dict(priors),
                )

        return EpisodeUncertaintyConfig(
            tool_configs=tool_configs,
            trigger_timing=trigger_timing,
            max_consecutive_failures=max_consecutive_failures,
            max_total_injections=max_total_injections,
            seed=seed
        )
