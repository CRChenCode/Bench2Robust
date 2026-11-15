# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Injection policy for enforcing solvability constraints.

This module implements the decision logic that determines when and what type of
uncertainty to inject while maintaining task solvability.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from bench2robust.uncertainty.config import (
    EpisodeUncertaintyConfig,
    NoiseType,
    TriggerTiming,
    ToolConfig
)
import random


@dataclass
class InjectionState:
    """Tracks injection state for solvability constraints."""
    
    # Per-tool consecutive failure count
    consecutive_failures: Dict[str, int] = field(default_factory=dict)
    
    # Per-tool, per-error-type injection count
    # Format: {"tool_name": {"timeout": 1, "partial": 0, ...}}
    per_tool_error_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Total injections across all tools
    total_injections: int = 0
    
    # Current step number
    current_step: int = 0
    
    # Total steps in episode (for timing windows)
    total_steps: Optional[int] = None
    
    def reset(self):
        """Reset state for new episode."""
        self.consecutive_failures.clear()
        self.per_tool_error_counts.clear()
        self.total_injections = 0
        self.current_step = 0
        self.total_steps = None
    
    def record_injection(self, tool_name: str, noise_type: NoiseType):
        """Record an injection."""
        # Update consecutive failures
        if tool_name not in self.consecutive_failures:
            self.consecutive_failures[tool_name] = 0
        self.consecutive_failures[tool_name] += 1
        
        # Update per-tool, per-error-type counts
        if tool_name not in self.per_tool_error_counts:
            self.per_tool_error_counts[tool_name] = {}
        
        noise_type_str = noise_type.value
        if noise_type_str not in self.per_tool_error_counts[tool_name]:
            self.per_tool_error_counts[tool_name][noise_type_str] = 0
        self.per_tool_error_counts[tool_name][noise_type_str] += 1
        
        # Update total
        self.total_injections += 1
    
    def record_clean(self, tool_name: str):
        """Record a clean execution (resets consecutive failures)."""
        self.consecutive_failures[tool_name] = 0
    
    def get_consecutive_failures(self, tool_name: str) -> int:
        """Get consecutive failure count for tool."""
        return self.consecutive_failures.get(tool_name, 0)
    
    def get_error_type_count(self, tool_name: str, noise_type: NoiseType) -> int:
        """Get injection count for specific tool and error type."""
        if tool_name not in self.per_tool_error_counts:
            return 0
        return self.per_tool_error_counts[tool_name].get(noise_type.value, 0)


class InjectionPolicy:
    """
    Decides when and what type of uncertainty to inject.
    Enforces solvability constraints.
    """
    
    def __init__(self, config: EpisodeUncertaintyConfig):
        """
        Initialize policy with configuration.
        
        Args:
            config: Episode uncertainty configuration
        """
        self.config = config
        self.state = InjectionState()
        self.rng = random.Random(config.seed)
    
    def reset(self, total_steps: Optional[int] = None):
        """
        Reset policy for new episode.
        
        Args:
            total_steps: Total steps in episode (for timing windows)
        """
        self.state.reset()
        self.state.total_steps = total_steps
        self.rng = random.Random(self.config.seed)
    
    def can_inject(self, tool_name: str, step: int) -> bool:
        """Return whether policy constraints permit an injection at this step."""
        self.state.current_step = step

        # Unknown tools fail closed: executing them cleanly is safer than
        # corrupting a response without an explicit tool classification.
        if tool_name not in self.config.tool_configs:
            return False

        tool_config = self.config.get_tool_config(tool_name)
        if tool_config.is_write_action or self._is_meta_tool(tool_name):
            return False
        if self.state.total_injections >= self.config.max_total_injections:
            return False

        consecutive = self.state.get_consecutive_failures(tool_name)
        effective_limit = (
            tool_config.max_consecutive_failures
            if tool_config.max_consecutive_failures is not None
            else self.config.max_consecutive_failures
        )
        if consecutive >= effective_limit:
            return False
        if not self._in_timing_window(step):
            return False
        return True

    def should_inject(
        self,
        tool_name: str,
        step: int
    ) -> Tuple[bool, Optional[NoiseType]]:
        """Decide whether to inject uncertainty and select its type."""
        if not self.can_inject(tool_name, step):
            return (False, None)

        tool_config = self.config.get_tool_config(tool_name)
        noise_type = self._sample_noise_type(tool_config)
        if noise_type == NoiseType.CLEAN:
            return (False, None)

        # Tools without alternatives receive each error type at most once.
        if (
            not tool_config.has_alternatives
            and self.state.get_error_type_count(tool_name, noise_type) > 0
        ):
            return (False, None)

        return (True, noise_type)
    
    # ------------------------------------------------------------------
    # Explicit decision API (ICLR revision).
    # Additive: should_inject() / can_inject() above are unchanged. Scenario-blocked
    # tools are handled by ScenarioController and never reach this policy.
    # ------------------------------------------------------------------
    def decide(self, tool_name: str, step: int):
        """Stochastic decision for an UNBLOCKED tool call.

        Returns a bench2robust.uncertainty.scenario.Decision of kind CLEAN,
        STOCHASTIC_EXEC_FAILURE (do not execute) or STOCHASTIC_SILENT_CORRUPTION
        (execute, then corrupt). Scenario blocking is handled by ScenarioController
        and must be checked BEFORE calling this method; this method never returns
        SCENARIO_BLOCK and consumes exactly the same RNG draws as should_inject().
        """
        from bench2robust.uncertainty.scenario import decision_from_noise
        should, noise_type = self.should_inject(tool_name, step)
        return decision_from_noise(noise_type if should else None, reason="stochastic policy")

    def record_decision(self, tool_name: str, decision) -> None:
        """Update stochastic counters for a decision returned by decide().

        SCENARIO_BLOCK decisions are ignored on purpose: blocked calls never consume
        the budget B, the K_max consecutive-failure counter or per-error counts.
        """
        from bench2robust.uncertainty.scenario import DecisionKind
        if decision.kind == DecisionKind.SCENARIO_BLOCK:
            return
        if decision.is_stochastic and decision.noise_type is not None:
            self.state.record_injection(tool_name, decision.noise_type)
        else:
            self.state.record_clean(tool_name)

    def record_execution(
        self,
        tool_name: str,
        injected: bool,
        noise_type: Optional[NoiseType]
    ):
        """
        Record tool execution result.
        
        Args:
            tool_name: Name of the tool
            injected: Whether noise was injected
            noise_type: Type of noise injected (or None if clean)
        """
        if injected and noise_type is not None:
            self.state.record_injection(tool_name, noise_type)
        else:
            self.state.record_clean(tool_name)
    
    def _is_meta_tool(self, tool_name: str) -> bool:
        """Check if tool is a meta tool."""
        meta_tools = {'calculate', 'think', 'transfer_to_human_agents'}
        return tool_name in meta_tools
    
    def _in_timing_window(self, step: int) -> bool:
        """Check if current step is in configured timing window."""
        if self.config.trigger_timing == TriggerTiming.UNIFORM:
            return True
        
        if self.state.total_steps is None:
            # Unknown total steps, allow injection
            return True
        
        third = self.state.total_steps // 3
        
        if self.config.trigger_timing == TriggerTiming.EARLY:
            return step < third
        elif self.config.trigger_timing == TriggerTiming.MIDDLE:
            return third <= step < 2 * third
        elif self.config.trigger_timing == TriggerTiming.LATE:
            return step >= 2 * third
        
        return True
    
    def _sample_noise_type(self, tool_config: ToolConfig) -> NoiseType:
        """
        Sample noise type from tool's uncertainty priors.
        
        Args:
            tool_config: Tool configuration
        
        Returns:
            Sampled noise type
        """
        noise_types = list(tool_config.uncertainty_priors.keys())
        probabilities = list(tool_config.uncertainty_priors.values())
        
        sampled = self.rng.choices(noise_types, weights=probabilities, k=1)[0]
        
        # Convert string to NoiseType enum
        if sampled == "clean":
            return NoiseType.CLEAN
        elif sampled == "timeout":
            return NoiseType.TIMEOUT
        elif sampled == "rate_limit":
            return NoiseType.RATE_LIMIT
        elif sampled == "server_error":
            return NoiseType.SERVER_ERROR
        elif sampled == "auth_error":
            return NoiseType.AUTH_ERROR
        elif sampled == "malformed_response":
            return NoiseType.MALFORMED_RESPONSE
        elif sampled == "schema_drift":
            return NoiseType.SCHEMA_DRIFT
        elif sampled == "partial":
            return NoiseType.PARTIAL
        elif sampled == "stale":
            return NoiseType.STALE
        elif sampled == "factual_error":
            return NoiseType.FACTUAL_ERROR
        else:
            raise ValueError(f"Unknown noise type: {sampled}")
    
    def get_state_summary(self) -> Dict:
        """Get summary of current injection state for logging."""
        return {
            "total_injections": self.state.total_injections,
            "current_step": self.state.current_step,
            "consecutive_failures": dict(self.state.consecutive_failures),
            "per_tool_error_counts": dict(self.state.per_tool_error_counts)
        }
