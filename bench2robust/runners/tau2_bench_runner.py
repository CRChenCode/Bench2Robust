# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tau2-Bench runner implementation.

Encapsulates all tau2-bench-specific logic:
  - Task loading via tau2.runner.get_tasks
  - Orchestrator building and simulation execution
  - Environment wrapping for uncertainty injection
  - tau2-specific monkey-patches and fixes
"""

import importlib
import importlib.util
import json
import logging
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Optional tau2 import (deferred to setup() so core imports stay dependency-free)
# ---------------------------------------------------------------------------

_TAU2_PATH_ENV = "BENCH2ROBUST_TAU2_PATH"
_NL_EVAL_MODEL_ENV = "BENCH2ROBUST_NL_EVAL_MODEL"
_tau2_imported = False


def _ensure_imports() -> None:
    """Make an installed or explicitly configured tau2 package importable."""
    global _tau2_imported
    if _tau2_imported:
        return

    if importlib.util.find_spec("tau2") is None:
        tau2_path = os.environ.get(_TAU2_PATH_ENV)
        if tau2_path:
            sys.path.insert(0, tau2_path)
            importlib.invalidate_caches()

    if importlib.util.find_spec("tau2") is None:
        raise ImportError(
            "tau2 is required for Tau2BenchRunner. Install it with "
            "python -m pip install \"bench2robust[tau2]\" or set "
            "BENCH2ROBUST_TAU2_PATH to the checkout's src directory."
        )

    _tau2_imported = True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

from bench2robust.runner import (
    BenchmarkRunner, BenchmarkTask, DomainConfig, EpisodeResult,
)


# Domain constants
DOMAIN_TASK_SET = {
    "retail": "retail",
    "airline": "airline",
    "telecom": "telecom",
}

MAX_STEPS = 200

WRITE_TOOLS: Dict[str, Set[str]] = {
    "retail": {
        "cancel_pending_order", "exchange_delivered_order_items",
        "modify_pending_order_address", "modify_pending_order_items",
        "modify_pending_order_payment", "modify_user_address",
        "return_delivered_order_items", "place_new_order",
    },
    "airline": {
        "book_reservation", "cancel_reservation", "send_certificate",
        "update_reservation_baggages", "update_reservation_flights",
        "update_reservation_passengers",
    },
    "telecom": {
        "suspend_line", "resume_line", "send_payment_request",
        "enable_roaming", "disable_roaming", "refuel_data",
        "set_network_mode_preference", "toggle_airplane_mode",
        "reseat_sim_card", "toggle_data", "toggle_roaming",
        "toggle_data_saver_mode", "set_apn_settings", "reset_apn_settings",
        "toggle_wifi", "toggle_wifi_calling", "connect_vpn", "disconnect_vpn",
        "grant_app_permission", "reboot_device", "make_payment",
    },
}

ALTERNATIVE_TOOLS: Dict[str, Set[str]] = {
    "retail": {
        "find_user_id_by_email", "place_new_order",
        "get_orders_for_user", "search_product_by_name", "get_item_by_product_and_options",
    },
    "airline": {
        "search_onestop_flight", "get_all_reservations_for_user",
        "find_user_by_email", "get_flight_details",
    },
    "telecom": {
        "get_customer_by_email", "get_all_lines_for_customer",
        "get_bill_by_period", "get_data_usage_summary",
        "check_line_status", "list_all_plans",
    },
}


class Tau2BenchRunner(BenchmarkRunner):
    """Runner for tau2-bench customer service benchmark."""

    @property
    def name(self) -> str:
        return "tau2-bench"

    def get_domains(self) -> List[str]:
        return list(DOMAIN_TASK_SET.keys())

    def get_domain_config(self, domain: str) -> DomainConfig:
        from bench2robust.adapters.tau_bench.tool_configs import (
            RETAIL_TOOL_CONFIGS, AIRLINE_TOOL_CONFIGS, TELECOM_TOOL_CONFIGS,
        )
        tool_configs_map = {
            "retail": RETAIL_TOOL_CONFIGS,
            "airline": AIRLINE_TOOL_CONFIGS,
            "telecom": TELECOM_TOOL_CONFIGS,
        }
        return DomainConfig(
            name=domain,
            write_tools=WRITE_TOOLS.get(domain, set()),
            alternative_tools=ALTERNATIVE_TOOLS.get(domain, set()),
            tool_configs=tool_configs_map.get(domain, {}),
        )

    def get_tasks(
        self,
        domain: str,
        split: str = "base",
        task_ids: Optional[List[str]] = None,
        start: int = 0,
        end: int = -1,
    ) -> List[BenchmarkTask]:
        _ensure_imports()
        from tau2.runner import get_tasks as tau2_get_tasks

        task_set = DOMAIN_TASK_SET[domain]
        if task_ids:
            raw_tasks = tau2_get_tasks(task_set, task_split_name=split, task_ids=task_ids)
        else:
            raw_tasks = tau2_get_tasks(task_set, task_split_name=split)
            end_idx = len(raw_tasks) if end == -1 else min(end, len(raw_tasks))
            raw_tasks = raw_tasks[start:end_idx]

        return [
            BenchmarkTask(
                id=str(t.id),
                domain=domain,
                instruction=str(t.user_scenario),
                raw=t,
            )
            for t in raw_tasks
        ]

    def setup(self) -> None:
        """Apply tau2-specific patches."""
        _ensure_imports()

        # Silence noisy loggers
        try:
            from loguru import logger as _loguru_logger
            _loguru_logger.remove()
        except ImportError:
            pass
        for name in ("LiteLLM", "httpx", "httpcore", "openai", "litellm"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

        # Fix: tau2's llm_utils sets litellm.client_session to httpx.Client
        import litellm as _litellm
        _litellm.client_session = None
        _litellm.aclient_session = None

        # Preserve tau2's evaluator default unless the caller explicitly opts
        # into a different model through a portable environment variable.
        import tau2.config as _tau2_config
        import tau2.evaluator.evaluator_nl_assertions as _tau2_nl_eval
        from tau2.utils.llm_utils import extract_json_from_llm_response as _extract_json

        nl_eval_model = os.environ.get(_NL_EVAL_MODEL_ENV)
        if nl_eval_model:
            _tau2_config.DEFAULT_LLM_NL_ASSERTIONS = nl_eval_model
            _tau2_nl_eval.DEFAULT_LLM_NL_ASSERTIONS = nl_eval_model

        _orig_nl_generate = _tau2_nl_eval.generate

        def _nl_generate_patched(model, messages, **kwargs):
            msg = _orig_nl_generate(model, messages, **kwargs)
            if kwargs.get("call_name") == "nl_assertions_eval" and msg.content:
                msg.content = _extract_json(msg.content)
            return msg

        _tau2_nl_eval.generate = _nl_generate_patched

    def run_episode(
        self,
        task: BenchmarkTask,
        agent_model_str: str,
        agent_model_kwargs: Dict,
        user_model_str: str,
        agent_name: str,
        save_trajectory: bool = True,
        uncertainty_config: Optional[Any] = None,
        with_alternatives: bool = False,
        scenario_spec: Optional[Any] = None,
        protect_alternatives: Optional[bool] = None,
        fallback_noise: Optional[str] = None,
    ) -> EpisodeResult:
        """Run one tau2 episode.

        ICLR revision: ``scenario_spec`` (bench2robust.uncertainty.scenario.ScenarioSpec)
        installs an episode-persistent S1/S2/S3 blocked path set that is applied BEFORE the
        base environment executes the tool. Scenario-blocked tools are never executed and
        stay blocked until episode reset, regardless of B and K_max; unblocked fallback
        tools (``scenario_spec.fallback_paths``) are exempt from stochastic noise
        (configurable via ``fallback_noise``: "exempt" (default), "exempt_s1" or "noisy");
        all other tools follow the stochastic policy. ``protect_alternatives`` keeps the
        legacy behaviour of never injecting noise on any alternative tool; it defaults to
        that legacy behaviour only when no scenario_spec is given.
        """
        _ensure_imports()
        from tau2.runner import build_text_orchestrator, run_simulation
        from tau2.data_model.simulation import TextRunConfig
        from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage
        from tau2.environment.environment import Environment

        domain = task.domain
        raw_task = task.raw
        inject_mode = uncertainty_config is not None

        try:
            config = TextRunConfig(
                domain=domain,
                agent=agent_name,
                llm_agent=agent_model_str,
                llm_args_agent=agent_model_kwargs,
                llm_user=user_model_str,
                llm_args_user={},
                max_steps=MAX_STEPS,
                task_split_name=None,
            )

            orchestrator = build_text_orchestrator(config, raw_task, seed=42)

            # Wrap environment with uncertainty injection if requested
            unc_env = None
            if inject_mode:
                from bench2robust.uncertainty.config import NoiseType
                from bench2robust.uncertainty.injection_policy import InjectionPolicy
                from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
                from bench2robust.uncertainty.response_cache import ResponseCache

                unc_env = _Tau2UncertaintyEnv(
                    base_env=orchestrator.environment,
                    domain=domain,
                    uncertainty_config=uncertainty_config,
                    with_alternatives=with_alternatives,
                    scenario_spec=scenario_spec,
                    protect_alternatives=protect_alternatives,
                    fallback_noise=fallback_noise,
                )
                orchestrator.environment = unc_env

            t0 = time.time()
            result = run_simulation(orchestrator)
            duration = time.time() - t0

            reward = result.reward_info.reward if result.reward_info else 0.0
            reward_info = (
                result.reward_info.model_dump(exclude={"info"})
                if result.reward_info else {}
            )

            # Count tokens
            agent_prompt = agent_completion = 0
            user_prompt = user_completion = 0
            steps = 0
            for m in result.messages:
                if isinstance(m, AssistantMessage) and m.usage:
                    agent_prompt += m.usage.get("prompt_tokens", 0)
                    agent_completion += m.usage.get("completion_tokens", 0)
                    steps += 1
                elif isinstance(m, UserMessage) and m.usage:
                    user_prompt += m.usage.get("prompt_tokens", 0)
                    user_completion += m.usage.get("completion_tokens", 0)

            # Serialize messages
            chat_history = None
            if save_trajectory:
                chat_history = _serialize_tau2_messages(result.messages)

            ep = EpisodeResult(
                task_id=task.id,
                domain=domain,
                benchmark=self.name,
                model_id=agent_model_str,
                instruction=task.instruction[:500],
                reward=reward,
                reward_info=reward_info,
                inject_mode=inject_mode,
                uncertainty_seed=uncertainty_config.seed if inject_mode else None,
                max_injections_configured=(
                    uncertainty_config.max_total_injections if inject_mode else None
                ),
                total_injections_actual=unc_env.total_injections if unc_env else 0,
                uncertainty_events=unc_env.uncertainty_events if unc_env else None,
                agent_prompt_tokens=agent_prompt,
                agent_completion_tokens=agent_completion,
                user_prompt_tokens=user_prompt,
                user_completion_tokens=user_completion,
                total_steps=steps,
                duration=duration,
                chat_history=chat_history,
            )

            # Init-time scenario label (for scenario-conditioned analysis)
            if unc_env is not None:
                ep.extra["scenario"] = unc_env.scenario.summary()
            # Store raw messages for judge access
            ep.extra["_raw_messages"] = result.messages
            return ep

        except Exception as e:
            tb = traceback.format_exc()
            sys.__stderr__.write(f"\n[task={task.id}] EXCEPTION:\n{tb}\n")
            sys.__stderr__.flush()
            return EpisodeResult(
                task_id=task.id,
                domain=domain,
                benchmark=self.name,
                model_id=agent_model_str,
                instruction=task.instruction[:500],
                reward=0.0,
                reward_info={"error": str(e), "traceback": tb},
                inject_mode=inject_mode,
                error=str(e),
            )

    def get_messages_for_judge(self, result: EpisodeResult) -> Optional[List]:
        """Return raw tau2 message objects for the judge."""
        return result.extra.get("_raw_messages")


# ---------------------------------------------------------------------------
# Tau2-specific uncertainty environment wrapper
# ---------------------------------------------------------------------------

class _Tau2UncertaintyEnv:
    """
    Wraps a tau2 Environment and injects noise into tool responses.
    Kept internal to this module — the runner is the public interface.

    ICLR revision: every call is decided BEFORE execution.
      1. scenario controller: blocked path -> blocked response, tool NOT executed,
         stochastic counters untouched; noise-exempt fallback (fallback_noise "exempt" /
         "exempt_s1") -> executed clean, stochastic policy not consulted;
      2. stochastic policy: explicit failure -> tool NOT executed; silent corruption ->
         execute then corrupt; clean -> execute.
    """

    def __init__(self, base_env, domain: str, uncertainty_config, with_alternatives: bool = False,
                 scenario_spec=None, protect_alternatives: Optional[bool] = None,
                 permanently_blocked_tools: Optional[Set[str]] = None,
                 fallback_noise: Optional[str] = None):
        from bench2robust.uncertainty.config import NoiseType
        from bench2robust.uncertainty.injection_policy import InjectionPolicy
        from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
        from bench2robust.uncertainty.response_cache import ResponseCache
        from bench2robust.uncertainty.scenario import (
            DOMAIN_EQUIVALENCE_CLASSES, ScenarioController, ScenarioSpec)
        from tau2.environment.environment import Environment

        # Copy base env attributes for tau2 compatibility
        self._base_env = base_env
        self.domain_name = base_env.domain_name
        self.policy = base_env.policy
        self.tools = base_env.tools
        self.user_tools = base_env.user_tools
        self.solo_mode = base_env.solo_mode

        self.domain = domain
        self.injection_policy = InjectionPolicy(uncertainty_config)
        self.injection_policy.reset()
        self.response_cache = ResponseCache()

        all_noise = [
            NoiseType.TIMEOUT, NoiseType.RATE_LIMIT, NoiseType.SERVER_ERROR,
            NoiseType.AUTH_ERROR, NoiseType.MALFORMED_RESPONSE, NoiseType.SCHEMA_DRIFT,
            NoiseType.PARTIAL, NoiseType.STALE, NoiseType.FACTUAL_ERROR,
        ]
        self.noise_injectors = {
            nt: NoiseInjectorFactory.create_injector(nt, uncertainty_config.seed, domain)
            for nt in all_noise
        }
        self.with_alternatives = with_alternatives
        scenario_requested = scenario_spec is not None or permanently_blocked_tools is not None
        # Equivalence classes are only in play when alternative tools are offered.
        classes = DOMAIN_EQUIVALENCE_CLASSES.get(domain, ()) if with_alternatives else ()
        if scenario_spec is None:
            scenario_spec = ScenarioSpec.from_blocked(permanently_blocked_tools, classes=classes)
        self.scenario = ScenarioController(scenario_spec, seed=uncertainty_config.seed, domain=domain,
                                           fallback_noise=fallback_noise, classes=classes)
        if protect_alternatives is None:
            # Legacy release behaviour (every alternative tool always clean) only when no
            # scenario control is requested. With a scenario, the episode's fallback tools
            # are noise-exempt per fallback_noise and all other tools are noisy.
            protect_alternatives = with_alternatives and not scenario_requested
        self.protect_alternatives = bool(protect_alternatives)
        self.uncertainty_events: List[Dict] = []
        self.total_injections = 0          # stochastic injections only
        self._step = 0

    @property
    def scenario_blocked_calls(self) -> int:
        return self.scenario.blocked_calls

    @property
    def fallback_noise(self) -> str:
        return self.scenario.fallback_noise

    @property
    def fallback_exempt_calls(self) -> int:
        return self.scenario.fallback_exempt_calls

    def get_response(self, message):
        """Intercept tool calls: scenario block -> stochastic decision -> execute or not."""
        from bench2robust.uncertainty.config import NoiseType
        from bench2robust.uncertainty.scenario import DecisionKind

        self._step += 1
        tool_name = message.name

        # 1) Scenario blocking takes precedence; the tool is never executed and the
        #    stochastic policy is never consulted.
        if self.scenario.is_blocked(tool_name):
            noise_type, text = self.scenario.blocked_response(tool_name, self._step)
            self.uncertainty_events.append({
                "step": self._step, "tool": tool_name,
                "noise_type": noise_type.value,
                "clean_obs": "(permanently blocked)",
                "injected_obs": text[:300],
                "permanently_blocked": True,
                "decision": DecisionKind.SCENARIO_BLOCK.value,
            })
            return self._make_tool_message(message, text, error=True)

        # 1b) Noise-exempt fallback (fallback_noise "exempt" / "exempt_s1"): executed clean,
        #     the stochastic policy is not consulted, so B / K_max are not consumed.
        pre = self.scenario.decide(tool_name)
        exempt = pre is not None and pre.fallback_exempt

        write_tools = WRITE_TOOLS.get(self.domain, set())
        alt_tools = ALTERNATIVE_TOOLS.get(self.domain, set())
        is_write = tool_name in write_tools
        is_alt = self.protect_alternatives and tool_name in alt_tools
        known = tool_name in self.injection_policy.config.tool_configs

        # 2) Stochastic decision BEFORE execution.
        if is_alt or exempt or not known:
            decision = None
        else:
            decision = self.injection_policy.decide(tool_name, self._step)

        if decision is not None and decision.kind == DecisionKind.STOCHASTIC_EXEC_FAILURE:
            injector = self.noise_injectors[decision.noise_type]
            injected_obs = injector.inject(None, tool_name, domain=self.domain)
            self.injection_policy.record_decision(tool_name, decision)
            self.total_injections += 1
            self.uncertainty_events.append({
                "step": self._step, "tool": tool_name,
                "noise_type": decision.noise_type.value,
                "clean_obs": "(not executed)",
                "injected_obs": str(injected_obs)[:300],
                "permanently_blocked": False,
                "decision": decision.kind.value,
            })
            return self._make_tool_message(message, injected_obs, error=True)

        # 3) Execute (clean or silent corruption)
        result = self._base_env.get_response(message)
        clean_obs = result.content

        if not is_write:
            try:
                parsed = json.loads(clean_obs) if isinstance(clean_obs, str) else clean_obs
            except (json.JSONDecodeError, TypeError):
                parsed = clean_obs
            self.response_cache.cache_response(tool_name, parsed)

        if (decision is not None and decision.kind == DecisionKind.STOCHASTIC_SILENT_CORRUPTION
                and not self._is_error_response(clean_obs)):
            injected_obs = self._apply_injection(decision.noise_type, clean_obs, tool_name, message.arguments)
            self.injection_policy.record_decision(tool_name, decision)
            self.total_injections += 1
            self.uncertainty_events.append({
                "step": self._step, "tool": tool_name,
                "noise_type": decision.noise_type.value,
                "clean_obs": str(clean_obs)[:300],
                "injected_obs": str(injected_obs)[:300],
                "permanently_blocked": False,
                "decision": decision.kind.value,
            })
            return self._make_tool_message(message, str(injected_obs), error=False)

        if known:
            self.injection_policy.record_execution(tool_name, False, None)
        return result

    def _apply_injection(self, noise_type, clean_obs, tool_name, tool_kwargs):
        from bench2robust.uncertainty.config import NoiseType

        injector = self.noise_injectors[noise_type]
        extra = {"domain": self.domain, "tool_kwargs": tool_kwargs}

        # All transient error types: return error string, ignore clean response
        if noise_type in (
            NoiseType.TIMEOUT, NoiseType.RATE_LIMIT, NoiseType.SERVER_ERROR,
            NoiseType.AUTH_ERROR, NoiseType.MALFORMED_RESPONSE, NoiseType.SCHEMA_DRIFT,
        ):
            return injector.inject(clean_obs, tool_name, **extra)

        parsed = clean_obs
        was_string = isinstance(clean_obs, str)
        if was_string:
            try:
                parsed = json.loads(clean_obs)
            except (json.JSONDecodeError, TypeError):
                pass

        if noise_type == NoiseType.STALE:
            result = injector.inject(parsed, tool_name, response_cache=self.response_cache, **extra)
        else:
            result = injector.inject(parsed, tool_name, **extra)

        if was_string and not isinstance(result, str):
            return json.dumps(result)
        return result

    def _is_error_response(self, response) -> bool:
        """
        Check if a tool response is already an error (not a successful result).
        
        Injecting noise on top of an existing error has no behavioral impact —
        the agent would retry regardless of the specific error message.
        Only inject on successful responses to create meaningful signal.
        """
        if response is None:
            return True
        s = str(response)
        # Explicit error patterns from tau2-bench tool implementations
        error_indicators = [
            "Error:", "error:", "not found", "Not found",
            "No reservation", "No user", "No flight",
            "Invalid", "invalid",
        ]
        for indicator in error_indicators:
            if indicator in s:
                return True
        # Empty list/dict responses (e.g., no flights found) — these are
        # valid "no results" responses, not errors per se, but injecting on
        # them also has minimal impact (agent already knows tool returned nothing)
        if s.strip() in ("[]", "{}", "null", "None", ""):
            return True
        return False

    def _make_tool_message(self, original_msg, content, error=False):
        from tau2.data_model.message import ToolMessage
        return ToolMessage(
            id=original_msg.id,
            content=content,
            requestor=original_msg.requestor,
            role="tool",
            error=error,
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to base environment."""
        return getattr(self._base_env, name)


# ---------------------------------------------------------------------------
# Message serialization (tau2-specific)
# ---------------------------------------------------------------------------

def _serialize_tau2_messages(messages) -> List[Dict]:
    """Convert tau2 message objects into JSON-serializable dicts."""
    from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage

    history = []
    for m in messages:
        entry: Dict[str, Any] = {}

        if isinstance(m, AssistantMessage):
            entry["role"] = "assistant"
            if m.content:
                entry["content"] = m.content
            if m.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in m.tool_calls
                ]
            thinking = _extract_thinking(m)
            if thinking:
                entry["thinking"] = thinking
            if m.usage:
                entry["usage"] = m.usage
        elif isinstance(m, UserMessage):
            entry["role"] = "user"
            if m.content:
                entry["content"] = m.content
            if m.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in m.tool_calls
                ]
        elif isinstance(m, ToolMessage):
            entry["role"] = "tool"
            entry["tool_call_id"] = m.id
            entry["content"] = m.content
            if m.error:
                entry["error"] = True
        else:
            entry["role"] = str(getattr(m, "role", "unknown"))
            entry["content"] = str(m)

        history.append(entry)
    return history


def _extract_thinking(msg) -> Optional[str]:
    """Extract thinking/reasoning content from an AssistantMessage."""
    # Bedrock-style: raw_data contains thinking blocks
    if msg.raw_data:
        choices = msg.raw_data.get("choices", [])
        if choices:
            message_data = choices[0].get("message", {})
            reasoning = message_data.get("reasoning_content")
            if reasoning:
                return reasoning

        content_blocks = msg.raw_data.get("content", [])
        if isinstance(content_blocks, list):
            thinking_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
            if thinking_parts:
                return "\n".join(thinking_parts)

    # Qwen3-style: <think>...</think> tags in content
    if msg.content and "<think>" in msg.content:
        match = re.search(r"<think>(.*?)</think>", msg.content, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None
