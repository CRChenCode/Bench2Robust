# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Bench2Robust: A controllable robustness harness for tool-use benchmarks.

Converts clean tool-use benchmarks into controlled stochastic environments
with configurable injection budgets, bounded consecutive failures, and
alternative tool paths.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Benchmark Adapters (tau_bench, vitabench, bfcl)    │
    │  - Wrap environment / intercept tool execution      │
    │  - Map benchmark-specific types to generic format   │
    └────────────────────────┬────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────┐
    │  Generic Injection Engine                            │
    │  - Policy enforcement (budget, write protection)    │
    │  - Noise type selection                             │
    │  - Event logging                                    │
    └────────────────────────┬────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────┐
    │  Noise Injectors (9 types)                          │
    │  - Explicit-signal: timeout, rate_limit, server_error, │
    │    auth_error, malformed, schema_drift              │
    │  - Silent: partial, stale, factual_error            │
    └────────────────────────┬────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────┐
    │  Domain Registry (extensible)                       │
    │  - Critical fields per (domain, tool)               │
    │  - Stale rules, corruption rules, status transitions│
    └─────────────────────────────────────────────────────┘

Quick start:
    from bench2robust.adapters import get_adapter
    from bench2robust.uncertainty.config import ConfigFactory

    # Create adapter for your benchmark
    adapter = get_adapter("tau_bench", domain="retail")

    # Create injection config
    config = ConfigFactory.create_tau_bench_config(domain="retail", seed=42)

    # Wrap environment
    wrapped_env = adapter.wrap_environment(original_env, config)
"""

__version__ = "0.2.0"

# Core exports
from bench2robust.runner import BenchmarkRunner, BenchmarkTask, DomainConfig, EpisodeResult
from bench2robust.uncertainty.config import (
    EpisodeUncertaintyConfig,
    ToolConfig,
    NoiseType,
    ConfigFactory,
)
from bench2robust.uncertainty.base_adapter import BaseAdapter
from bench2robust.uncertainty.generic_wrapper import GenericInjectionEngine
from bench2robust.uncertainty.domain_registry import DomainRegistry
from bench2robust.adapters import get_adapter, register_adapter, list_adapters

__all__ = [
    # Runner interface
    "BenchmarkRunner",
    "BenchmarkTask",
    "DomainConfig",
    "EpisodeResult",
    # Config
    "EpisodeUncertaintyConfig",
    "ToolConfig",
    "NoiseType",
    "ConfigFactory",
    # Adapters
    "BaseAdapter",
    "get_adapter",
    "register_adapter",
    "list_adapters",
    # Engine
    "GenericInjectionEngine",
    "DomainRegistry",
]
