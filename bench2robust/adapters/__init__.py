# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Bench2Robust Adapters Registry.

Each adapter provides benchmark-specific integration with the core injection
framework. Adapters are responsible for:
  - Wrapping the benchmark's environment to intercept tool calls
  - Mapping benchmark-specific message types to/from the injection engine
  - Providing domain-specific tool classifications

Supported benchmarks:
  - tau_bench: τ-bench / τ²-bench (multi-turn customer service)
  - vitabench: VitaBench (multi-turn life services)
  - bfcl: BFCL v3/v4 (multi-turn function calling)
"""

from typing import Dict, Type
from bench2robust.uncertainty.base_adapter import BaseAdapter

# Lazy imports to avoid requiring all benchmark dependencies
_ADAPTER_REGISTRY: Dict[str, str] = {
    "tau_bench": "bench2robust.adapters.tau_bench.adapter.TauBenchAdapter",
    "vitabench": "bench2robust.adapters.vitabench.adapter.VitaBenchAdapter",
    "bfcl": "bench2robust.adapters.bfcl.adapter.BFCLAdapter",
}


def get_adapter(benchmark: str, **kwargs) -> BaseAdapter:
    """
    Factory function to get an adapter by benchmark name.
    
    Args:
        benchmark: One of "tau_bench", "vitabench", "bfcl"
        **kwargs: Passed to adapter constructor (e.g., domain="retail")
        
    Returns:
        Initialized adapter instance
        
    Raises:
        ValueError: If benchmark is not registered
    """
    if benchmark not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown benchmark: '{benchmark}'. Available: {available}"
        )
    
    module_path, class_name = _ADAPTER_REGISTRY[benchmark].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    return adapter_class(**kwargs)


def register_adapter(name: str, adapter_path: str) -> None:
    """
    Register a new adapter for dynamic discovery.
    
    Args:
        name: Short name for the benchmark
        adapter_path: Full dotted path to the adapter class
            (e.g., "bench2robust.adapters.my_bench.adapter.MyAdapter")
    """
    _ADAPTER_REGISTRY[name] = adapter_path


def list_adapters() -> list:
    """List all registered adapter names."""
    return sorted(_ADAPTER_REGISTRY.keys())
