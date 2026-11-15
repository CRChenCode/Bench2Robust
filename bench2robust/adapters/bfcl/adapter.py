# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
BFCL (Berkeley Function Calling Leaderboard) adapter for Bench2Robust.

BFCL v3/v4 multi-turn evaluation executes function calls as Python code
via execute_multi_turn_func_call(). The injection point is different from
τ-bench/VitaBench: instead of intercepting a get_response() method, we
need to intercept the function execution results.

BFCL's multi-turn evaluation:
  1. Model generates function calls (parsed as Python AST)
  2. Calls are executed against stateful class instances
  3. Execution results (stdout strings) are compared with ground truth

Integration approach:
  - Preferred (ICLR revision): execute_with_interception()
    decides BEFORE execution. Scenario-blocked calls and explicit (execution-
    suppressing) failures are removed from the call list and never executed; only
    clean calls and silent corruptions run against the stateful BFCL instances.
  - Legacy: maybe_inject_results() corrupts results AFTER execution. This is
    observation corruption only and must not be described as execution-suppressing
    blocking.
"""

from typing import Any, List, Dict, Optional, Callable
from bench2robust.uncertainty.base_adapter import BaseAdapter
from bench2robust.uncertainty.config import EpisodeUncertaintyConfig, ToolConfig
from bench2robust.uncertainty.generic_wrapper import GenericInjectionEngine


class BFCLExecutionInterceptor:
    """
    Intercepts BFCL's multi-turn function execution results.
    
    BFCL executes function calls and returns result strings. This interceptor
    wraps the execution to optionally inject noise into the returned strings.
    
    Usage:
        interceptor = BFCLExecutionInterceptor(config, domain="bfcl_multi_turn")
        
        # Wrap the execution function
        original_execute = execute_multi_turn_func_call
        def patched_execute(func_call_list, **kwargs):
            results, instances = original_execute(func_call_list, **kwargs)
            return interceptor.maybe_inject_results(results, func_call_list), instances
    """
    
    def __init__(
        self,
        config: EpisodeUncertaintyConfig,
        domain: str = "bfcl_multi_turn",
        permanently_blocked_tools: Optional[set] = None,
        scenario_spec: Optional[Any] = None,
        fallback_noise: Optional[str] = None,
    ):
        self.config = config
        self.domain = domain
        self.engine = GenericInjectionEngine(
            config=config,
            domain=domain,
            permanently_blocked_tools=permanently_blocked_tools,
            scenario_spec=scenario_spec,
            fallback_noise=fallback_noise,
        )
    
    def maybe_inject_results(
        self,
        execution_results: List[str],
        func_call_list: List[str],
    ) -> List[str]:
        """
        Potentially inject noise into execution results.
        
        Args:
            execution_results: List of result strings from function execution
            func_call_list: List of function call strings (for tool name extraction)
            
        Returns:
            Possibly-corrupted execution results
        """
        import re
        
        injected_results = []
        for i, (result, call_str) in enumerate(zip(execution_results, func_call_list)):
            # Extract tool name from function call string
            # BFCL calls look like: "ClassName.method_name(args...)"
            tool_name = self._extract_tool_name(call_str)
            
            should_inject, noise_type = self.engine.should_inject(tool_name)
            
            if should_inject and noise_type is not None:
                injected = self.engine.inject(noise_type, tool_name, result, {})
                if isinstance(injected, str):
                    injected_results.append(injected)
                else:
                    import json
                    injected_results.append(json.dumps(injected, default=str))
            else:
                self.engine.record_clean(tool_name, result)
                injected_results.append(result)
        
        return injected_results
    
    def execute_with_interception(
        self,
        execute_fn: Callable,
        func_call_list: List[str],
        **execute_kwargs,
    ):
        """Pre-execution interception for one BFCL turn.

        execute_fn must have the signature of BFCL's execute_multi_turn_func_call
        (func_call_list=..., **kwargs) -> (results, instances). Calls are executed in
        their original order; suppressed calls are skipped, so they cannot change the
        BFCL instance state.

        Returns (results, instances, decisions).
        """
        import json
        from bench2robust.uncertainty.scenario import DecisionKind

        decisions = [self.engine.decide(self._extract_tool_name(c)) for c in func_call_list]
        results: List[Any] = [None] * len(func_call_list)
        instances = None
        # Execute the allowed calls one at a time, preserving order relative to each other.
        for i, (call, d) in enumerate(zip(func_call_list, decisions)):
            tool_name = self._extract_tool_name(call)
            if not d.execute:
                results[i] = self.engine.respond_without_execution(d, tool_name, {})
                continue
            res, instances = execute_fn(func_call_list=[call], **execute_kwargs)
            clean = res[0] if res else ""
            out = self.engine.finish_executed(d, tool_name, clean, {})
            results[i] = out if isinstance(out, str) else json.dumps(out, default=str)
        return results, instances, decisions

    def _extract_tool_name(self, call_str: str) -> str:
        """Extract tool/method name from BFCL function call string."""
        import re
        # Pattern: "ClassName.method_name(...)" or "method_name(...)"
        match = re.match(r'(?:(\w+)\.)?(\w+)\s*\(', call_str.strip())
        if match:
            class_name = match.group(1)
            method_name = match.group(2)
            if class_name:
                return f"{class_name}.{method_name}"
            return method_name
        return call_str.split("(")[0].strip() if "(" in call_str else call_str
    
    def get_events(self) -> list:
        """Get injection events for logging."""
        return self.engine.get_events()
    
    def reset(self, permanently_blocked_tools: Optional[set] = None, scenario_spec=None,
              fallback_noise: Optional[str] = None) -> None:
        """Reset for a new episode (clears counters and installs the next scenario)."""
        self.engine.reset(permanently_blocked_tools=permanently_blocked_tools,
                          scenario_spec=scenario_spec, fallback_noise=fallback_noise)


class BFCLAdapter(BaseAdapter):
    """Adapter for BFCL multi-turn evaluation."""
    
    def __init__(self, test_category: str = "multi_turn_base"):
        """
        Initialize adapter for BFCL.
        
        Args:
            test_category: BFCL test category (e.g., "multi_turn_base",
                          "multi_turn_miss_func", "multi_turn_miss_param")
        """
        self.domain = "bfcl_multi_turn"
        self.test_category = test_category
    
    def get_tool_list(self) -> List[str]:
        """Get list of tools (extracted from test data at runtime)."""
        return []
    
    def wrap_tool(self, tool: Any, config: ToolConfig) -> Any:
        """Not used - we intercept at execution level."""
        return tool
    
    def wrap_environment(self, env: Any, config: EpisodeUncertaintyConfig,
                         permanently_blocked_tools: Optional[set] = None,
                         scenario_spec: Optional[Any] = None,
                         fallback_noise: Optional[str] = None) -> Any:
        """
        Create an execution interceptor for BFCL.
        
        Note: For BFCL, 'env' is not a traditional environment object.
        Instead, we return a BFCLExecutionInterceptor that patches the
        execution function.
        """
        return BFCLExecutionInterceptor(config=config, domain=self.domain,
                                        permanently_blocked_tools=permanently_blocked_tools,
                                        scenario_spec=scenario_spec, fallback_noise=fallback_noise)
    
    def extract_tool_response(self, result: Any) -> Any:
        """Extract response from BFCL execution result."""
        return str(result)
    
    def format_error_message(self, error_type: str) -> str:
        """Format error in BFCL-compatible format."""
        if error_type == "timeout":
            return "Error: Function execution timed out"
        return f"Error: {error_type}"
