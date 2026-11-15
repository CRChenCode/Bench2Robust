# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
VitaBench adapter for Bench2Robust.

VitaBench uses Environment.get_response(ToolCall) → ToolMessage,
which is nearly identical to τ-bench. The injection point is the same:
intercept get_response() and optionally corrupt the response.

VitaBench domains: delivery, instore, ota, cross_domain
"""

from typing import Any, List, Dict, Optional
from bench2robust.uncertainty.base_adapter import BaseAdapter
from bench2robust.uncertainty.config import EpisodeUncertaintyConfig, ToolConfig
from bench2robust.uncertainty.generic_wrapper import GenericInjectionEngine
from bench2robust.adapters.vitabench.domain_registry_init import register_vitabench_domains


class InstrumentedVitaBenchEnv:
    """
    Wraps VitaBench Environment to inject uncertainty.
    
    Drop-in replacement that intercepts get_response() calls.
    VitaBench's Environment.get_response(ToolCall) returns ToolMessage with:
      - id: str
      - name: str
      - content: str (JSON)
      - requestor: str
      - role: "tool"
      - error: bool
    """
    
    def __init__(
        self,
        base_env: Any,
        config: EpisodeUncertaintyConfig,
        domain: str,
        permanently_blocked_tools: Optional[set] = None,
        scenario_spec: Optional[Any] = None,
        fallback_noise: Optional[str] = None,
    ):
        self.base_env = base_env
        self.config = config
        self.domain = domain
        self.engine = GenericInjectionEngine(
            config=config,
            domain=domain,
            permanently_blocked_tools=permanently_blocked_tools,
            scenario_spec=scenario_spec,
            fallback_noise=fallback_noise,
        )
    
    def get_response(self, message: Any) -> Any:
        """
        Intercept VitaBench tool call and optionally inject noise.

        ICLR revision: the decision is taken BEFORE execution.
        base_env.get_response is NOT called for scenario blocks or explicit
        (execution-suppressing) stochastic failures; it is called only for clean
        calls (including noise-exempt fallbacks) and silent corruptions.

        Args:
            message: VitaBench ToolCall object with .name, .arguments, .id, .requestor

        Returns:
            VitaBench ToolMessage (possibly with injected noise)
        """
        import json
        from bench2robust.uncertainty.scenario import DecisionKind

        tool_name = message.name
        tool_kwargs = message.arguments if hasattr(message, 'arguments') else {}

        decision = self.engine.decide(tool_name)

        if not decision.execute:
            # SCENARIO_BLOCK or STOCHASTIC_EXEC_FAILURE: the tool is not executed.
            content = self.engine.respond_without_execution(decision, tool_name, tool_kwargs)
            if not isinstance(content, str):
                content = json.dumps(content, default=str, ensure_ascii=False)
            return self._tool_message(message, content, error=True)

        # Execute (CLEAN or STOCHASTIC_SILENT_CORRUPTION)
        clean_msg = self.base_env.get_response(message)
        try:
            clean_data = json.loads(clean_msg.content) if isinstance(clean_msg.content, str) else clean_msg.content
        except (json.JSONDecodeError, TypeError):
            clean_data = clean_msg.content

        if decision.kind == DecisionKind.STOCHASTIC_SILENT_CORRUPTION:
            injected = self.engine.finish_executed(decision, tool_name, clean_data, tool_kwargs)
            if isinstance(injected, str):
                injected_content = injected
            else:
                injected_content = json.dumps(injected, default=str, ensure_ascii=False)
            return self._tool_message(message, injected_content,
                                      error="Error" in injected_content)

        self.engine.finish_executed(decision, tool_name, clean_data, tool_kwargs)
        return clean_msg

    @staticmethod
    def _tool_message(message: Any, content: str, error: bool) -> Any:
        # Construct ToolMessage in VitaBench format; import lazily to avoid hard dependency
        from vita.data_model.message import ToolMessage
        return ToolMessage(
            id=message.id,
            name=message.name,
            content=content,
            requestor=message.requestor,
            role="tool",
            error=error,
        )

    def reset(self, permanently_blocked_tools: Optional[set] = None, scenario_spec=None,
              fallback_noise: Optional[str] = None) -> None:
        """Episode boundary: clears counters and installs the next scenario."""
        self.engine.reset(permanently_blocked_tools=permanently_blocked_tools,
                          scenario_spec=scenario_spec, fallback_noise=fallback_noise)

    def get_events(self) -> list:
        """Get injection events for logging."""
        return self.engine.get_events()
    
    def __getattr__(self, name):
        """Delegate all other attributes to base environment."""
        return getattr(self.base_env, name)


class VitaBenchAdapter(BaseAdapter):
    """Adapter for VitaBench environments."""
    
    def __init__(self, domain: str):
        """
        Initialize adapter for a VitaBench domain.
        
        Args:
            domain: "delivery", "instore", "ota", or "cross_domain"
        """
        register_vitabench_domains()
        self.domain = f"vitabench_{domain}"
        self.vita_domain = domain
    
    def get_tool_list(self) -> List[str]:
        """Get list of tools in the VitaBench domain."""
        # Loaded lazily from VitaBench's toolkit
        # Each domain has its own set of tools
        return []  # Populated at runtime from environment
    
    def wrap_tool(self, tool: Any, config: ToolConfig) -> Any:
        """Not used - we wrap at environment level."""
        return tool
    
    def wrap_environment(
        self,
        env: Any,
        config: EpisodeUncertaintyConfig,
        permanently_blocked_tools: Optional[set] = None,
        scenario_spec: Optional[Any] = None,
        fallback_noise: Optional[str] = None,
    ) -> Any:
        """Wrap VitaBench environment with uncertainty injection.

        scenario_spec / fallback_noise: see GenericInjectionEngine (fallback tools are
        noise-exempt by default).
        """
        return InstrumentedVitaBenchEnv(
            base_env=env,
            config=config,
            domain=self.domain,
            permanently_blocked_tools=permanently_blocked_tools,
            scenario_spec=scenario_spec,
            fallback_noise=fallback_noise,
        )
    
    def extract_tool_response(self, result: Any) -> Any:
        """Extract response content from VitaBench ToolMessage."""
        if hasattr(result, 'content'):
            return result.content
        return result
    
    def format_error_message(self, error_type: str) -> str:
        """Format error in VitaBench-compatible format."""
        if error_type == "timeout":
            return "Error: request timed out"
        return f"Error: {error_type}"
