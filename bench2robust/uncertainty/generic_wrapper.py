# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Benchmark-agnostic uncertainty injection engine.
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, Optional, Tuple

from bench2robust.uncertainty.config import EpisodeUncertaintyConfig, NoiseType
from bench2robust.uncertainty.injection_policy import InjectionPolicy
from bench2robust.uncertainty.response_cache import ResponseCache
from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
from bench2robust.uncertainty.scenario import (
    BLOCKED_RESPONSE_NOISE,
    DOMAIN_EQUIVALENCE_CLASSES,
    EXEC_FAILURE_NOISE,
    Decision,
    DecisionKind,
    ScenarioController,
    ScenarioSpec,
)


@dataclass
class InjectionEvent:
    """Record of a single injection event."""

    step: int
    tool: str
    noise_type: str
    clean_obs: str
    injected_obs: str
    permanently_blocked: bool = False
    decision: str = ""


class GenericInjectionEngine:
    """
    Benchmark-agnostic injection engine.

    Pipeline (ICLR revision; Algorithm 1):
        scenario controller (episode-persistent blocked paths)
          -> stochastic injection policy (B, K_max, timing, per-error limits)
          -> explicit Decision: SCENARIO_BLOCK | STOCHASTIC_EXEC_FAILURE |
                                STOCHASTIC_SILENT_CORRUPTION | CLEAN

    Preferred usage (execution-suppressing):
        engine = GenericInjectionEngine(config, domain="retail",
                                        permanently_blocked_tools={"get_order_details"})
        decision = engine.decide(tool_name)
        if not decision.execute:                       # SCENARIO_BLOCK or EXEC_FAILURE
            response = engine.respond_without_execution(decision, tool_name, tool_kwargs)
        else:
            clean = run_the_tool(...)
            response = engine.finish_executed(decision, tool_name, clean, tool_kwargs)

    Legacy usage (should_inject / inject / record_clean) is still supported. For a
    scenario-blocked tool, inject() returns the blocked response and ignores the clean
    response, never samples a silent corruption and never touches policy counters. The
    caller is still responsible for not executing the tool; use decide() for that.

    Scenario-blocked calls:
      * never consult the stochastic policy;
      * never consume B, K_max consecutive-failure or per-error counters
        (counted separately in scenario.blocked_calls);
      * never receive silent corruption;
      * stay blocked until reset() installs the next episode's scenario.

    Unblocked fallback calls (scenario_spec.fallback_paths), with fallback_noise="exempt"
    (default) or "exempt_s1":
      * get a CLEAN decision without consulting the stochastic policy, so they never
        receive an explicit failure or a silent corruption;
      * never consume B or per-error counters (counted in scenario.fallback_exempt_calls);
      * with fallback_noise="noisy" they follow the normal stochastic policy instead.

    Summary: scenario-blocked tools are never executed and stay blocked until episode
    reset, regardless of B and K_max; unblocked fallback tools are exempt from stochastic
    noise (configurable via fallback_noise); all other tools follow the stochastic policy.
    """

    def __init__(
        self,
        config: EpisodeUncertaintyConfig,
        domain: str = "retail",
        permanently_blocked_tools: Optional[set] = None,
        scenario_spec: Optional[ScenarioSpec] = None,
        fallback_noise: Optional[str] = None,
    ):
        """
        Initialize injection engine.

        Args:
            config: Episode uncertainty configuration
            domain: Domain identifier (for domain-specific corruption rules)
            permanently_blocked_tools: Scenario-blocked tools (S2/S3 path set P). The
                fallback side is derived from the domain's equivalence classes.
            scenario_spec: Full scenario spec (label + blocked + fallback paths);
                overrides permanently_blocked_tools when given.
            fallback_noise: "exempt" (default), "exempt_s1" or "noisy"; see
                bench2robust.uncertainty.scenario.FALLBACK_NOISE_MODES.
        """
        self.config = config
        self.domain = domain
        self._classes = DOMAIN_EQUIVALENCE_CLASSES.get(domain, ())
        if scenario_spec is None:
            scenario_spec = ScenarioSpec.from_blocked(permanently_blocked_tools, classes=self._classes)
        self.scenario = ScenarioController(scenario_spec, seed=config.seed, domain=domain,
                                           fallback_noise=fallback_noise)

        # Core components
        self.injection_policy = InjectionPolicy(config)
        self.response_cache = ResponseCache()
        self.injectors = self._create_injectors()

        # State
        self.step_count = 0
        self.events: list[InjectionEvent] = []
        self.total_injections = 0          # stochastic injections only
        self.last_decision: Optional[Decision] = None

    def _create_injectors(self) -> Dict[NoiseType, Any]:
        """Create a fresh deterministic injector set for one episode."""
        return {
            noise_type: NoiseInjectorFactory.create_injector(
                noise_type,
                self.config.seed,
                self.domain,
            )
            for noise_type in NoiseType
            if noise_type not in (NoiseType.CLEAN, NoiseType.NATURAL_ERROR)
        }

    # -- scenario state -------------------------------------------------
    @property
    def permanently_blocked_tools(self) -> set:
        """Mutable view of the current episode's blocked path set."""
        return self.scenario.blocked

    @permanently_blocked_tools.setter
    def permanently_blocked_tools(self, tools) -> None:
        self.scenario.reset(ScenarioSpec.from_blocked(tools, classes=self._classes))

    @property
    def scenario_blocked_calls(self) -> int:
        return self.scenario.blocked_calls

    @property
    def fallback_noise(self) -> str:
        return self.scenario.fallback_noise

    @property
    def fallback_exempt_calls(self) -> int:
        return self.scenario.fallback_exempt_calls

    def reset(self, permanently_blocked_tools: Optional[set] = None,
              scenario_spec: Optional[ScenarioSpec] = None,
              fallback_noise: Optional[str] = None) -> None:
        """Reset engine for a new episode.

        Clears policy counters, cache, events AND the scenario state. The next
        episode's scenario is given here (default: no blocking). fallback_noise=None
        keeps the current mode.
        """
        self.injection_policy.reset()
        self.response_cache.clear()
        self.injectors = self._create_injectors()
        self.step_count = 0
        self.events.clear()
        self.total_injections = 0
        self.last_decision = None
        if scenario_spec is None:
            scenario_spec = ScenarioSpec.from_blocked(permanently_blocked_tools, classes=self._classes)
        self.scenario.reset(scenario_spec, fallback_noise=fallback_noise)

    # -- decision API ---------------------------------------------------
    def decide(self, tool_name: str) -> Decision:
        """Return the interception decision for the next tool call (call once per call)."""
        self.step_count += 1
        pre = self.scenario.decide(tool_name)
        if pre is not None:
            # Scenario blocking (SCENARIO_BLOCK) and noise-exempt fallbacks (CLEAN) take
            # precedence and never consult the stochastic policy.
            self.last_decision = pre
            return pre
        decision = self.injection_policy.decide(tool_name, self.step_count)
        self.last_decision = decision
        return decision

    def respond_without_execution(self, decision: Decision, tool_name: str,
                                  tool_kwargs: Optional[Dict] = None) -> Any:
        """Response for SCENARIO_BLOCK / STOCHASTIC_EXEC_FAILURE (tool not executed)."""
        if decision.kind == DecisionKind.SCENARIO_BLOCK:
            return self.blocked_response(tool_name)
        if decision.kind == DecisionKind.STOCHASTIC_EXEC_FAILURE:
            return self.inject(decision.noise_type, tool_name, None, tool_kwargs)
        raise ValueError(f"decision {decision.kind} requires execution")

    def finish_executed(self, decision: Decision, tool_name: str, clean_response: Any,
                        tool_kwargs: Optional[Dict] = None) -> Any:
        """Response for CLEAN / STOCHASTIC_SILENT_CORRUPTION after the tool executed."""
        if decision.kind == DecisionKind.STOCHASTIC_SILENT_CORRUPTION:
            return self.inject(decision.noise_type, tool_name, clean_response, tool_kwargs)
        if decision.kind == DecisionKind.CLEAN:
            self.record_clean(tool_name, clean_response)
            return clean_response
        raise ValueError(f"decision {decision.kind} must not execute the tool")

    def blocked_response(self, tool_name: str) -> str:
        """Episode-persistent blocked response. Does not touch policy counters."""
        noise_type, text = self.scenario.blocked_response(tool_name, self.step_count)
        self.events.append(InjectionEvent(
            step=self.step_count,
            tool=tool_name,
            noise_type=noise_type.value,
            clean_obs="(permanently blocked)",
            injected_obs=self._truncate(text),
            permanently_blocked=True,
            decision=DecisionKind.SCENARIO_BLOCK.value,
        ))
        return text

    # -- legacy API -----------------------------------------------------
    def should_inject(self, tool_name: str) -> Tuple[bool, Optional[NoiseType]]:
        """
        Legacy API: determine whether to inject noise on this tool call.

        For a scenario-blocked tool this returns (True, <explicit failure type>); the
        subsequent inject() call returns the blocked response without consulting or
        updating the stochastic policy. Prefer decide(), which also tells the caller
        whether the tool may be executed.
        """
        decision = self.decide(tool_name)
        if decision.kind == DecisionKind.SCENARIO_BLOCK:
            return True, BLOCKED_RESPONSE_NOISE[0]
        if decision.is_stochastic:
            return True, decision.noise_type
        return False, None

    def inject(
        self,
        noise_type: NoiseType,
        tool_name: str,
        clean_response: Any,
        tool_kwargs: Optional[Dict] = None,
    ) -> Any:
        """
        Inject noise into a clean response.

        Args:
            noise_type: Type of noise to inject
            tool_name: Name of the tool
            clean_response: Original clean response (str or dict); ignored for
                scenario-blocked tools and explicit failures
            tool_kwargs: Arguments passed to the tool (for contextual injectors)

        Returns:
            Injected (noisy) response
        """
        if self.scenario.is_blocked(tool_name):
            # Scenario blocks are never silent and never consume stochastic counters.
            return self.blocked_response(tool_name)
        if self.scenario.is_fallback_exempt(tool_name):
            # Noise-exempt fallback: never corrupted; recorded as a clean execution.
            if clean_response is None:
                raise ValueError(f"{tool_name} is a noise-exempt fallback and must be executed")
            self.record_clean(tool_name, clean_response)
            return clean_response

        injector = self.injectors.get(noise_type)
        if injector is None:
            return clean_response

        kwargs = {
            "domain": self.domain,
            "tool_kwargs": tool_kwargs or {},
            "response_cache": self.response_cache,
        }

        injected = injector.inject(clean_response, tool_name, **kwargs)

        self.events.append(InjectionEvent(
            step=self.step_count,
            tool=tool_name,
            noise_type=noise_type.value,
            clean_obs=("(not executed)" if clean_response is None
                       else self._truncate(clean_response)),
            injected_obs=self._truncate(injected),
            permanently_blocked=False,
            decision=(DecisionKind.STOCHASTIC_EXEC_FAILURE.value if noise_type in EXEC_FAILURE_NOISE
                      else DecisionKind.STOCHASTIC_SILENT_CORRUPTION.value),
        ))
        self.total_injections += 1

        # Update policy state (stochastic injections only)
        self.injection_policy.state.record_injection(tool_name, noise_type)

        return injected

    def record_clean(self, tool_name: str, clean_response: Any) -> None:
        """
        Record a clean (non-injected) tool execution.
        Caches the response for potential future stale injection.
        """
        self.response_cache.cache_response(tool_name, clean_response)
        self.injection_policy.state.record_clean(tool_name)

    def get_events(self) -> list[dict]:
        """Get injection events as serializable dicts."""
        return [
            {
                "step": e.step,
                "tool": e.tool,
                "noise_type": e.noise_type,
                "clean_obs": e.clean_obs,
                "injected_obs": e.injected_obs,
                "permanently_blocked": e.permanently_blocked,
                "decision": e.decision,
            }
            for e in self.events
        ]

    def get_counters(self) -> dict:
        """Stochastic and scenario counters, kept separate."""
        st = self.injection_policy.state
        return {
            "stochastic_total_injections": st.total_injections,
            "stochastic_consecutive_failures": dict(st.consecutive_failures),
            "stochastic_per_tool_error_counts": {k: dict(v) for k, v in st.per_tool_error_counts.items()},
            "engine_total_injections": self.total_injections,
            "scenario_blocked_calls": self.scenario.blocked_calls,
            "fallback_exempt_calls": self.scenario.fallback_exempt_calls,
            "scenario": self.scenario.summary(),
        }

    @staticmethod
    def _truncate(response: Any, max_len: int = 200) -> str:
        """Convert a response to a bounded log representation."""
        if response is None:
            return "null"
        if isinstance(response, str):
            return response[:max_len]
        try:
            return json.dumps(response, default=str)[:max_len]
        except (TypeError, ValueError):
            return str(response)[:max_len]
