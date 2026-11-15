# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tau2-Bench adapter metadata.

The executable Tau2 integration lives in :class:`Tau2BenchRunner`. This adapter
exposes domain tool classifications for discovery without maintaining a second,
divergent environment wrapper.
"""

from dataclasses import replace
from typing import Any, Dict, List

from bench2robust.uncertainty.base_adapter import BaseAdapter
from bench2robust.uncertainty.config import EpisodeUncertaintyConfig, ToolConfig
from bench2robust.adapters.tau_bench.tool_configs import (
    RETAIL_TOOL_CONFIGS,
    AIRLINE_TOOL_CONFIGS,
    TELECOM_TOOL_CONFIGS,
)


class TauBenchAdapter(BaseAdapter):
    """Tool-classification adapter for Tau2-Bench domains."""

    def __init__(self, domain: str):
        """Initialize metadata for ``retail``, ``airline``, or ``telecom``."""
        configs_by_domain: Dict[str, Dict[str, ToolConfig]] = {
            "retail": RETAIL_TOOL_CONFIGS,
            "airline": AIRLINE_TOOL_CONFIGS,
            "telecom": TELECOM_TOOL_CONFIGS,
        }
        if domain not in configs_by_domain:
            supported = ", ".join(sorted(configs_by_domain))
            raise ValueError(f"Unknown domain: {domain}. Must be one of: {supported}")

        self.domain = domain
        self.tool_configs = {
            name: replace(
                config,
                alternatives=list(config.alternatives),
                uncertainty_priors=dict(config.uncertainty_priors),
            )
            for name, config in configs_by_domain[domain].items()
        }

    def get_tool_list(self) -> List[str]:
        """Return tool names configured for this domain."""
        return list(self.tool_configs)

    def wrap_tool(self, tool: Any, config: ToolConfig) -> Any:
        """Return the tool unchanged; Tau2 interception occurs in its runner."""
        return tool

    def wrap_environment(
        self,
        env: Any,
        config: EpisodeUncertaintyConfig,
    ) -> Any:
        """Reject the removed legacy wrapper and direct callers to the runner."""
        raise NotImplementedError(
            "Tau2 environment wrapping is provided by Tau2BenchRunner.run_episode(); "
            "the legacy adapter wrapper was removed to avoid divergent behavior."
        )

    def extract_tool_response(self, result: Any) -> Any:
        """Extract content from a Tau2 ToolMessage-like result."""
        return result.content if hasattr(result, "content") else result

    def format_error_message(self, error_type: str) -> str:
        """Format an error in Tau2's textual tool-response form."""
        if error_type == "timeout":
            return "Error: request timed out"
        return f"Error: {error_type}"
