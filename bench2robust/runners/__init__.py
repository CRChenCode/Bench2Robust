# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Benchmark runners for the unified evaluation pipeline.

Each runner implements the BenchmarkRunner interface for a specific benchmark.
"""

from bench2robust.runner import BenchmarkRunner, BenchmarkTask, DomainConfig, EpisodeResult
from bench2robust.runners.tau2_bench_runner import Tau2BenchRunner

# Registry of available runners (populated by register_runner)
_RUNNERS: dict = {}


def register_runner(runner_class):
    """Register a benchmark runner class."""
    instance = runner_class()
    _RUNNERS[instance.name] = instance
    return runner_class


def get_runner(name: str) -> BenchmarkRunner:
    """Get a registered runner by name."""
    if name not in _RUNNERS:
        available = list(_RUNNERS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return _RUNNERS[name]


def list_runners() -> list:
    """List all registered runner names."""
    return list(_RUNNERS.keys())


register_runner(Tau2BenchRunner)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTask",
    "DomainConfig",
    "EpisodeResult",
    "Tau2BenchRunner",
    "get_runner",
    "list_runners",
    "register_runner",
]
