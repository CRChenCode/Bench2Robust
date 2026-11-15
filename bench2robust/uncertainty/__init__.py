# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
API Uncertainty Simulation Framework - Core Module

This module provides the core framework for injecting controlled uncertainty
into tool-based agent benchmarks.
"""

from bench2robust.uncertainty.config import (
    NoiseType,
    TriggerTiming,
    ToolConfig,
    EpisodeUncertaintyConfig,
    ConfigurationError,
    ConfigFactory,
)
from bench2robust.uncertainty.response_cache import ResponseCache
from bench2robust.uncertainty.noise_injectors import (
    NoiseInjector,
    TimeoutInjector,
    RateLimitInjector,
    ServerErrorInjector,
    AuthErrorInjector,
    MalformedResponseInjector,
    SchemaDriftInjector,
    PartialResponseInjector,
    StaleResponseInjector,
    FactualErrorInjector,
    NoiseInjectorFactory,
)
from bench2robust.uncertainty.injection_policy import InjectionPolicy, InjectionState
from bench2robust.uncertainty.logger import (
    ToolExecutionLog,
    EpisodeLog,
    EpisodeLogger,
    LogExporter,
)
from bench2robust.uncertainty.base_adapter import BaseAdapter
from bench2robust.uncertainty.scenario import (
    DEFAULT_FALLBACK_NOISE,
    FALLBACK_NOISE_MODES,
    Decision,
    DecisionKind,
    Scenario,
    ScenarioController,
    ScenarioSpec,
)

__version__ = "0.2.0"

__all__ = [
    # Config
    "NoiseType",
    "TriggerTiming",
    "ToolConfig",
    "EpisodeUncertaintyConfig",
    "ConfigurationError",
    "ConfigFactory",
    # Cache
    "ResponseCache",
    # Injectors
    "NoiseInjector",
    "TimeoutInjector",
    "RateLimitInjector",
    "ServerErrorInjector",
    "AuthErrorInjector",
    "MalformedResponseInjector",
    "SchemaDriftInjector",
    "PartialResponseInjector",
    "StaleResponseInjector",
    "FactualErrorInjector",
    "NoiseInjectorFactory",
    # Policy
    "InjectionPolicy",
    "InjectionState",
    # Logger
    "ToolExecutionLog",
    "EpisodeLog",
    "EpisodeLogger",
    "LogExporter",
    # Adapter
    "BaseAdapter",
    # Scenario control (S1/S2/S3), separate from stochastic injection
    "Decision",
    "DecisionKind",
    "Scenario",
    "ScenarioController",
    "ScenarioSpec",
    "FALLBACK_NOISE_MODES",
    "DEFAULT_FALLBACK_NOISE",
]
