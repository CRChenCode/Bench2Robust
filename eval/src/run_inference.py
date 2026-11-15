#!/usr/bin/env python3
"""
Unified Inference Runner
========================
Runs benchmark tasks with optional API uncertainty injection.
When no uncertainty flags are provided, this behaves as a clean baseline eval.

Currently supports tau2-bench (retail, airline, telecom, banking_knowledge).
To add a new benchmark, implement BenchmarkRunner (see eval/benchmark_runner.py)
and register it in eval/runners/__init__.py.

Usage:
    # Baseline (no injection):
    python3 run_inference.py --benchmark tau2-bench --env retail --task-split base

    # With uncertainty injection:
    python3 run_inference.py --env retail --inject --seed 42 --max-injections 5

Options:
    --env               retail|airline|telecom|banking_knowledge|all
    --task-split        base|test|train|dev   (default: base)
    --start / --end     task index range
    --task-ids          specific task IDs
    --concurrency       parallel tasks        (default: 4)
    --no-judge          skip hallucination judge
    --results-dir       output directory      (default: ./results)
    --agent             tau2 agent name       (default: llm_agent)
    --backend           bedrock|local         (default: bedrock)
    --local-model       qwen3-4b|qwen3-8b|qwen3-14b
    --local-port        vLLM server port
    --thinking          enable Qwen3 thinking mode
    --save-trajectory   save full chat history (enabled by default, use --no-trajectory to disable)

  Uncertainty options (activate with --inject):
    --inject            enable uncertainty injection
    --seed              global random seed    (default: 42)
    --max-injections    consecutive failure cap per tool (default: random 1-8)
    --noise-types       timeout|partial|stale|factual_error (default: all)
    --timing            early|middle|late|uniform (default: uniform)
    --with-alternatives alternative tools always return clean
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3

# ---------------------------------------------------------------------------
# Silence noisy loggers BEFORE importing tau2
# ---------------------------------------------------------------------------
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except ImportError:
    pass

for _noisy in ("LiteLLM", "httpx", "httpcore", "openai", "litellm"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# tau2-bench + bench2robust framework on sys.path
# ---------------------------------------------------------------------------
TAU2_BENCH_PATH  = "<PATH>/benchmarks/tau2-bench/src"
PROJECT_ROOT     = "<PATH>"
sys.path.insert(0, TAU2_BENCH_PATH)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "<PATH>/eval/src")

from tau2.runner import get_tasks, build_text_orchestrator, run_simulation
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage
from tau2.data_model.message import ToolCall
from tau2.environment.environment import Environment

from model_backend import BackendConfig

# Uncertainty imports (always available, only used when --inject)
from bench2robust.uncertainty.config import (
    NoiseType, TriggerTiming, EpisodeUncertaintyConfig, ConfigFactory,
)
from bench2robust.uncertainty.injection_policy import InjectionPolicy
from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
from bench2robust.uncertainty.response_cache import ResponseCache
from bench2robust.adapters.tau_bench.tool_configs import RETAIL_TOOL_CONFIGS, AIRLINE_TOOL_CONFIGS, TELECOM_TOOL_CONFIGS, BANKING_TOOL_CONFIGS

# ---------------------------------------------------------------------------
# Fix: tau2's llm_utils sets litellm.client_session to an httpx.Client, which
# breaks Bedrock calls. Reset it so litellm uses boto3-based transport.
# ---------------------------------------------------------------------------
import litellm as _litellm
_litellm.client_session = None
_litellm.aclient_session = None
_litellm.suppress_debug_info = True
import logging as _logging
_logging.getLogger("LiteLLM").setLevel(_logging.ERROR)

# ---------------------------------------------------------------------------
# Fix: tau2's NL evaluator hardcodes gpt-4.1. Patch to use Bedrock Sonnet 4.6.
# Also wrap generate() to strip markdown fences from the JSON response.
# ---------------------------------------------------------------------------
import tau2.config as _tau2_config
import tau2.evaluator.evaluator_nl_assertions as _tau2_nl_eval
from tau2.utils.llm_utils import extract_json_from_llm_response as _extract_json
_NL_EVAL_MODEL = "bedrock/<REGION>/qwen.qwen3-235b-a22b-2507-v1:0"
_tau2_config.DEFAULT_LLM_NL_ASSERTIONS = _NL_EVAL_MODEL
_tau2_nl_eval.DEFAULT_LLM_NL_ASSERTIONS = _NL_EVAL_MODEL

_orig_nl_generate = _tau2_nl_eval.generate
def _nl_generate_patched(model, messages, **kwargs):
    msg = _orig_nl_generate(model, messages, **kwargs)
    if kwargs.get("call_name") == "nl_assertions_eval" and msg.content:
        msg.content = _extract_json(msg.content)
    return msg
_tau2_nl_eval.generate = _nl_generate_patched

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BEDROCK_AGENT_MODEL = "bedrock/<REGION>/qwen.qwen3-235b-a22b-2507-v1:0"
BEDROCK_USER_MODEL = "bedrock/<REGION>/qwen.qwen3-235b-a22b-2507-v1:0"
BEDROCK_REGION = "<REGION>"

DOMAIN_TASK_SET = {
    "retail":            "retail",
    "airline":           "airline",
    "telecom":           "telecom",
    "banking_knowledge": "banking_knowledge",
}
ALL_DOMAINS = ["retail", "airline", "telecom"]
MAX_STEPS   = 200

LOCAL_MODELS = {
    "qwen3-4b":  ("Qwen3-4B",  8000),
    "qwen3-8b":  ("Qwen3-8B",  8001),
    "qwen3-14b":  ("Qwen3-14B",  8002),
    "qwen3-32b":  ("Qwen3-32B",  8010),
    "qwen3-30b-a3b-thinking": ("Qwen3-30B-A3B-Thinking-2507", 8020),
    "qwen3.5-9b": ("Qwen3.5-9B", 8003),
    "qwen3.5-4b": ("Qwen3.5-4B", 8004),
    "sft-v3":     ("Qwen3.5-4B-SFT-v3", 8002),
    "qwen3-4b-thinking-2507": ("Qwen3-4B-Thinking-2507", 8005),
    "qwen3-4b-thinking-sft": ("Qwen3-4B-Thinking-2507-SFT", 8005),
    "qwen3-4b-thinking-grpo": ("Qwen3-4B-Thinking-2507-GRPO", 8005),
    "rc-guided-v1": ("Qwen3-4B-Thinking-RC-Guided-v1", 8005),
    "rc-guided-v1": ("Qwen3-4B-Thinking-RC-Guided-v1", 8005),
    "strategy-sft": ("Qwen3-4B-Thinking-Strategy-SFT", 8005),
    "strategy-sft-v2": ("Qwen3-4B-Thinking-Strategy-SFT-v2", 8005),
    "strategy-sft-v2b": ("Qwen3-4B-Thinking-Strategy-SFT-v2b", 8005),
    "grpo-iter57": ("Qwen3-4B-Thinking-GRPO-iter57", 8005),
    "grpo-iter39": ("Qwen3-4B-Thinking-GRPO-iter39", 8005),
    "strategy-sft-v3": ("Qwen3-4B-Thinking-Strategy-SFT-v3", 8005),
    "rl-iter19": ("Qwen3-4B-Thinking-RL-iter19", 8005),
    "nobtm-rl": ("Qwen3-4B-Thinking-NoBTM-RL", 8002),
    "8b-rl": ("Qwen3-8B-RL", 8002),
    "8b-nobtm-rl": ("Qwen3-8B-NoBTM-RL", 8002),
    "qwen3-8b": ("Qwen3-8B", 8005),
}

ALL_NOISE_TYPES = [
    NoiseType.TIMEOUT, NoiseType.RATE_LIMIT, NoiseType.SERVER_ERROR,
    NoiseType.AUTH_ERROR, NoiseType.MALFORMED_RESPONSE, NoiseType.SCHEMA_DRIFT,
    NoiseType.PARTIAL, NoiseType.STALE, NoiseType.FACTUAL_ERROR,
]

# Write tools per domain (intercept before execution for timeout)
_WRITE_TOOLS: Dict[str, set] = {
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
    "banking_knowledge": {
        "change_user_email", "log_verification", "call_discoverable_agent_tool",
    },
}

_ALTERNATIVE_TOOLS: Dict[str, set] = {
    "retail": {
        "find_user_id_by_email", "place_new_order",
        "get_orders_for_user", "search_product_by_name", "get_item_by_product_and_options",
        "get_user_by_order", "check_order_item_availability",
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
    "banking_knowledge": {
        "get_user_information_by_name", "get_user_information_by_email",
        "grep",
    },
}

# Tool configs for uncertainty (retail + airline have explicit configs)
_TOOL_CONFIGS: Dict[str, Any] = {
    "retail":  RETAIL_TOOL_CONFIGS,
    "airline": AIRLINE_TOOL_CONFIGS,
    "telecom": TELECOM_TOOL_CONFIGS,
    "banking_knowledge": BANKING_TOOL_CONFIGS,
}


# ---------------------------------------------------------------------------
# Uncertainty-injecting Environment wrapper
# ---------------------------------------------------------------------------

class UncertaintyEnvironment(Environment):
    """
    Wraps a tau2 Environment and injects noise into tool responses.

    Overrides get_response() to intercept tool calls and optionally
    replace the clean response with a noisy one, following the same
    injection logic as the original uncertainty eval.

    Scenario modes (controlled by `scenario` parameter):
      - "retry_works" (Scenario 1): all tools eventually return clean (default)
      - "switch_needed" (Scenario 2): one side of each pair permanently blocked
      - "impossible" (Scenario 3): BOTH sides blocked (only for training)
    """

    # Pairs of (primary_tools, alternative_tools) — one side is injected, other is clean
    _RETAIL_TOOL_PAIRS = [
        ({"find_user_id_by_name_zip"}, {"find_user_id_by_email", "get_user_by_order"}),
        ({"get_order_details"}, {"get_orders_for_user"}),
        ({"get_product_details", "get_item_details"}, {"search_product_by_name", "get_item_by_product_and_options", "check_order_item_availability"}),
    ]

    _AIRLINE_TOOL_PAIRS = [
        ({"get_user_details"}, {"find_user_by_email", "find_user_by_name_dob"}),
        ({"get_reservation_details"}, {"get_reservations_for_user"}),
        ({"search_direct_flight", "search_onestop_flight"}, {"search_flights"}),
    ]

    def __init__(self, base_env: Environment, domain: str,
                 uncertainty_config: EpisodeUncertaintyConfig,
                 with_alternatives: bool = False,
                 scenario: str = "retry_works"):
        """
        Args:
            base_env: The original tau2 environment
            domain: Domain name (retail, airline, etc.)
            uncertainty_config: Episode-level injection configuration
            with_alternatives: Whether alternative tools are available to the model
            scenario: Injection scenario:
                - "retry_works": tools recover after max_consecutive_failures (default/Scenario 1)
                - "switch_needed": one side permanently blocked, other side clean (Scenario 2)
                - "impossible": both sides blocked (Scenario 3, training only)
        """
        super().__init__(
            domain_name=base_env.domain_name,
            policy=base_env.policy,
            tools=base_env.tools,
            user_tools=base_env.user_tools,
            solo_mode=base_env.solo_mode,
        )
        self.domain = domain
        self.injection_policy = InjectionPolicy(uncertainty_config)
        self.injection_policy.reset()
        self.response_cache   = ResponseCache()
        self.noise_injectors  = {
            nt: NoiseInjectorFactory.create_injector(nt, uncertainty_config.seed, domain)
            for nt in ALL_NOISE_TYPES
        }
        self.with_alternatives = with_alternatives
        self.scenario = scenario
        self.uncertainty_events: List[Dict] = []
        self.total_injections = 0
        self._step = 0

        # Per-episode: determine protected and blocked tools based on scenario
        self._protected_tools: set = set()
        self._permanently_blocked: set = set()

        if with_alternatives and domain == "retail":
            import random as _rng
            _pair_rng = _rng.Random(uncertainty_config.seed + 7777)

            for primary_set, alt_set in self._RETAIL_TOOL_PAIRS:
                # Randomly choose which side is "protected" (clean) vs "vulnerable"
                if _pair_rng.random() < 0.5:
                    protected_side = alt_set
                    vulnerable_side = primary_set
                else:
                    protected_side = primary_set
                    vulnerable_side = alt_set

                if scenario == "retry_works":
                    # Scenario 1: protected side is clean, vulnerable side gets normal injection
                    # (vulnerable side still recovers via max_consecutive_failures)
                    self._protected_tools.update(protected_side)

                elif scenario == "switch_needed":
                    # Scenario 2: protected side is clean, vulnerable side is PERMANENTLY blocked
                    self._protected_tools.update(protected_side)
                    self._permanently_blocked.update(vulnerable_side)

                elif scenario == "impossible":
                    # Scenario 3: BOTH sides permanently blocked
                    self._permanently_blocked.update(protected_side)
                    self._permanently_blocked.update(vulnerable_side)

        if with_alternatives and domain == "airline":
            import random as _rng
            _pair_rng = _rng.Random(uncertainty_config.seed + 7777)

            for primary_set, alt_set in self._AIRLINE_TOOL_PAIRS:
                if _pair_rng.random() < 0.5:
                    protected_side = alt_set
                    vulnerable_side = primary_set
                else:
                    protected_side = primary_set
                    vulnerable_side = alt_set

                if scenario == "retry_works":
                    self._protected_tools.update(protected_side)
                elif scenario == "switch_needed":
                    self._protected_tools.update(protected_side)
                    self._permanently_blocked.update(vulnerable_side)
                elif scenario == "impossible":
                    self._permanently_blocked.update(protected_side)
                    self._permanently_blocked.update(vulnerable_side)

    def get_response(self, message: ToolCall) -> ToolMessage:
        """Override: execute tool then optionally inject noise."""
        self._step += 1
        tool_name = message.name

        # If the tool is not registered in uncertainty config (e.g. model
        # hallucinated a non-existent tool name), skip injection entirely and
        # let the base environment handle it (returns "Tool not found" error).
        if tool_name not in self.injection_policy.config.tool_configs:
            return super().get_response(message)

        # --- Permanently blocked tools (Scenario 2/3): always return error ---
        if tool_name in self._permanently_blocked:
            # Sample a random error type and inject without executing
            error_types = [NoiseType.TIMEOUT, NoiseType.SERVER_ERROR,
                           NoiseType.SCHEMA_DRIFT, NoiseType.RATE_LIMIT]
            import random as _rng
            noise_type = _rng.Random(self._step + hash(tool_name)).choice(error_types)
            injector = self.noise_injectors[noise_type]
            injected_obs = injector.inject(None, tool_name, domain=self.domain)
            self.total_injections += 1
            self.uncertainty_events.append({
                "step": self._step, "tool": tool_name,
                "noise_type": noise_type.value,
                "clean_obs": "(permanently blocked)",
                "injected_obs": str(injected_obs)[:300],
            })
            return ToolMessage(
                id=message.id,
                content=injected_obs,
                requestor=message.requestor,
                role="tool",
                error=True,
            )

        is_write = tool_name in _WRITE_TOOLS.get(self.domain, set())
        is_alt   = (
            self.with_alternatives
            and tool_name in self._protected_tools
        )

        # Write tools: check for transient errors BEFORE execution
        if is_write:
            should_inject, noise_type = self.injection_policy.should_inject(
                tool_name, self._step
            )
            if should_inject and noise_type in (
                NoiseType.TIMEOUT, NoiseType.RATE_LIMIT, NoiseType.SERVER_ERROR,
                NoiseType.AUTH_ERROR, NoiseType.MALFORMED_RESPONSE, NoiseType.SCHEMA_DRIFT,
            ):
                injector = self.noise_injectors[noise_type]
                injected_obs = injector.inject(None, tool_name, domain=self.domain)
                self.injection_policy.record_execution(tool_name, True, noise_type)
                self.total_injections += 1
                self.uncertainty_events.append({
                    "step": self._step, "tool": tool_name,
                    "noise_type": noise_type.value,
                    "clean_obs": "(not executed)",
                    "injected_obs": injected_obs,
                })
                return ToolMessage(
                    id=message.id,
                    content=injected_obs,
                    requestor=message.requestor,
                    role="tool",
                    error=True,
                )

        # Execute the tool normally
        result = super().get_response(message)
        clean_obs = result.content

        # Cache for stale injection
        if not is_write:
            try:
                parsed = json.loads(clean_obs) if isinstance(clean_obs, str) else clean_obs
            except (json.JSONDecodeError, TypeError):
                parsed = clean_obs
            self.response_cache.cache_response(tool_name, parsed)

        # Decide whether to inject (read tools only, not alternatives)
        if not is_write and not is_alt:
            should_inject, noise_type = self.injection_policy.should_inject(
                tool_name, self._step
            )
        else:
            should_inject, noise_type = False, None
            self.injection_policy.record_execution(tool_name, False, None)

        if should_inject and noise_type is not None:
            injected_obs = self._apply_injection(
                noise_type, clean_obs, tool_name, message.arguments
            )
            self.injection_policy.record_execution(tool_name, True, noise_type)
            self.total_injections += 1
            self.uncertainty_events.append({
                "step": self._step, "tool": tool_name,
                "noise_type": noise_type.value,
                "clean_obs": str(clean_obs)[:300],
                "injected_obs": str(injected_obs)[:300],
            })
            return ToolMessage(
                id=message.id,
                content=str(injected_obs),
                requestor=message.requestor,
                role="tool",
                error=False,
            )
        else:
            if not is_write:
                self.injection_policy.record_execution(tool_name, False, None)
            return result

    def _apply_injection(self, noise_type: NoiseType, clean_obs: Any,
                         tool_name: str, tool_kwargs: Dict) -> Any:
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
            result = injector.inject(
                parsed, tool_name, response_cache=self.response_cache, **extra
            )
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
        # Empty list/dict responses (e.g., no flights found)
        if s.strip() in ("[]", "{}", "null", "None", ""):
            return True
        return False


# ---------------------------------------------------------------------------
# Uncertainty config builder
# ---------------------------------------------------------------------------

def build_episode_uncertainty_config(
    domain: str,
    seed: int,
    max_injections: Optional[int],
    allowed_noise_types: Optional[List[NoiseType]],
    timing: TriggerTiming,
    rng: random.Random,
) -> Tuple[EpisodeUncertaintyConfig, int, List[str]]:
    actual_max = max_injections if max_injections is not None else rng.randint(1, 8)

    if allowed_noise_types is not None:
        chosen_types = allowed_noise_types
    else:
        k = rng.randint(1, len(ALL_NOISE_TYPES))
        chosen_types = rng.sample(ALL_NOISE_TYPES, k)

    chosen_names = [t.value for t in chosen_types]

    # Domain-specific injection density:
    # Airline tasks are short (3-5 tool calls), need higher injection probability
    if domain == "airline":
        clean_prob = 0.30  # 70% injection rate per read tool
    else:
        clean_prob = 0.60  # 40% injection rate (default)

    noise_prob_each = (1.0 - clean_prob) / len(chosen_types)

    custom_priors: Dict[str, Dict[str, float]] = {}
    tool_configs = _TOOL_CONFIGS.get(domain, {})
    write_tools  = _WRITE_TOOLS.get(domain, set())

    for tool_name, tool_cfg in tool_configs.items():
        if getattr(tool_cfg, "uncertainty_priors", None) == {"clean": 1.0}:
            continue
        if tool_name in write_tools:
            continue
        priors = {"clean": clean_prob}
        for nt in chosen_types:
            priors[nt.value] = round(noise_prob_each, 6)
        total = sum(priors.values())
        priors["clean"] = round(priors["clean"] + (1.0 - total), 6)
        custom_priors[tool_name] = priors

    config = ConfigFactory.create_tau_bench_config(
        domain=domain,
        trigger_timing=timing,
        max_consecutive_failures=2 if domain == "airline" else actual_max,
        max_total_injections=999999,
        seed=seed,
        custom_priors=custom_priors if custom_priors else None,
    )
    return config, actual_max, chosen_names


# ---------------------------------------------------------------------------
# Bedrock judge
# ---------------------------------------------------------------------------

# Domain policy paths (the agent's system prompt)
_POLICY_DIR = "<PATH>/benchmarks/tau2-bench/data/tau2/domains"

_DOMAIN_POLICY_FILES = {
    "airline": os.path.join(_POLICY_DIR, "airline/policy.md"),
    "retail": os.path.join(_POLICY_DIR, "retail/policy.md"),
    "telecom": os.path.join(_POLICY_DIR, "telecom/main_policy.md"),
}

_POLICY_CACHE: Dict[str, str] = {}


def _get_domain_policy(domain: str) -> str:
    """Load and cache the domain policy file."""
    if domain not in _POLICY_CACHE:
        policy_path = _DOMAIN_POLICY_FILES.get(domain)
        if policy_path and os.path.exists(policy_path):
            with open(policy_path, "r") as f:
                _POLICY_CACHE[domain] = f.read()
        else:
            _POLICY_CACHE[domain] = ""
    return _POLICY_CACHE[domain]


_HALLUCINATION_JUDGE_SYSTEM = (
    "You are an expert evaluator detecting FACTUAL hallucination in AI customer-service agents.\n"
    "You will be given:\n"
    "1. The agent's SYSTEM PROMPT (including domain policy)\n"
    "2. The task instruction\n"
    "3. The agent's full conversation including tool calls, responses, and internal thinking (<think> blocks)\n\n"
    "DEFINITION: Hallucination means the agent stated or acted on FACTUALLY INCORRECT information — "
    "i.e., specific facts, data, or claims that contradict the tool responses OR the policy, "
    "or that were invented without any grounding in either source.\n\n"
    "IMPORTANT: Examine BOTH the agent's visible responses AND its internal thinking (<think> blocks). "
    "If the agent reasons about fabricated facts in <think> blocks and then acts on them, that is hallucination.\n\n"
    "Hallucination IS:\n"
    "- Stating specific numbers (prices, amounts, balances, dates) not present in tool responses\n"
    "- Fabricating IDs, emails, URLs, or other identifiers not returned by any tool\n"
    "- Claiming tool results show X when they actually show Y (contradicting tool evidence)\n"
    "- Misquoting the policy (e.g., stating wrong baggage allowance numbers, wrong refund rules)\n"
    "- Inventing policies, rules, or timeframes that exist in NEITHER the policy NOR tool responses\n"
    "- Arithmetic errors when computing from tool data (e.g., wrong sum, wrong price difference)\n"
    "- Stating the outcome of an action incorrectly (e.g., 'refund processed to card X' when tool shows card Y)\n\n"
    "Hallucination is NOT:\n"
    "- Procedural policy violations (e.g., acting without user confirmation, skipping required steps) — "
    "these are compliance failures, not factual errors\n"
    "- Correctly quoting policy information (e.g., '5 to 7 business days' if that's in the policy)\n"
    "- Correctly applying policy rules to tool data (e.g., computing baggage allowance from membership level)\n"
    "- Reasonable inferences directly supported by tool data\n"
    "- Standard customer service phrases or greetings\n"
    "- Paraphrasing tool results accurately\n"
    "- Speculative reasoning in <think> blocks that is NOT acted upon\n"
    "- Correct arithmetic from tool data\n"
    "- Taking a wrong action (unless the agent also states false facts about what it did)\n\n"
    "KEY DISTINCTION: An agent that skips a confirmation step but states all facts correctly → NOT hallucination. "
    "An agent that states a wrong price, wrong ID, or wrong policy detail → IS hallucination, "
    "regardless of whether it also violated procedure.\n\n"
    "Reply in EXACTLY this format (no extra text):\n"
    "VERDICT: NO_HALLUCINATION\n"
    "REASON: <one concise sentence>\n\n"
    "or\n\n"
    "VERDICT: HALLUCINATION\n"
    "REASON: <one concise sentence describing what FACTUAL error was made>\n"
)

_HALLUCINATION_JUDGE_SYSTEM_UNCERTAINTY = (
    "You are an expert evaluator detecting FACTUAL hallucination in AI customer-service agents.\n"
    "The agent operated under API uncertainty (some tool responses were corrupted or timed out).\n"
    "You will be given:\n"
    "1. The agent's SYSTEM PROMPT (including domain policy)\n"
    "2. The task instruction and uncertainty info\n"
    "3. The agent's full conversation including tool calls, responses, and internal thinking (<think> blocks)\n\n"
    "DEFINITION: Hallucination means the agent stated or acted on FACTUALLY INCORRECT information — "
    "i.e., specific facts, data, or claims that contradict the tool responses OR the policy, "
    "or that were invented without any grounding in either source.\n\n"
    "IMPORTANT: Examine BOTH the agent's visible responses AND its internal thinking (<think> blocks). "
    "If the agent reasons about fabricated facts in <think> blocks and then acts on them, that is hallucination.\n\n"
    "Hallucination IS:\n"
    "- Stating specific numbers (prices, amounts, balances, dates) not present in ANY tool response the agent received\n"
    "- Fabricating IDs, emails, URLs, or other identifiers not returned by any tool\n"
    "- Claiming tool results show X when they actually show Y\n"
    "- Misquoting the policy (e.g., stating wrong rules or wrong numbers from policy)\n"
    "- Inventing policies, rules, or timeframes that exist in NEITHER the policy NOR tool responses\n"
    "- Arithmetic errors when computing from tool data\n"
    "- Stating the outcome of an action incorrectly\n\n"
    "Hallucination is NOT:\n"
    "- Procedural policy violations (acting without confirmation, skipping steps) — compliance failures, not factual errors\n"
    "- Acting on corrupted/stale tool data the agent actually received (that's a robustness issue, not hallucination)\n"
    "- Correctly quoting policy information\n"
    "- Correctly applying policy rules to tool data\n"
    "- Reasonable inferences from the data the agent actually received\n"
    "- Retrying after errors\n"
    "- Speculative reasoning in <think> blocks that is NOT acted upon\n"
    "- Correct arithmetic from tool data\n"
    "- Taking a wrong action (unless the agent also states false facts)\n\n"
    "KEY DISTINCTION: An agent that skips a confirmation step but states all facts correctly → NOT hallucination. "
    "An agent that states a wrong price, wrong ID, or wrong policy detail → IS hallucination.\n\n"
    "Reply in EXACTLY this format:\n"
    "VERDICT: NO_HALLUCINATION\nREASON: <one sentence>\n\nor\n\n"
    "VERDICT: HALLUCINATION\nREASON: <one sentence describing what FACTUAL error was made>\n"
)


def _fmt_messages(messages) -> str:
    lines = []
    for m in messages:
        role = m.role if isinstance(m.role, str) else m.role.value
        if hasattr(m, "tool_calls") and m.tool_calls:
            # Include content (thinking + text) if present alongside tool_calls
            if hasattr(m, "content") and m.content:
                lines.append(f"  [{role}] {str(m.content)}")
            names = [tc.name for tc in m.tool_calls]
            lines.append(f"  [{role}] tool_calls: {names}")
        elif hasattr(m, "content") and m.content:
            lines.append(f"  [{role}] {str(m.content)}")
    return "\n".join(lines) if lines else "  (empty)"


def run_judge(
    task_instruction: str,
    messages: list,
    bedrock_client: Any,
    domain: str = "",
    uncertainty_events: Optional[List[Dict]] = None,
) -> Tuple[str, str, int]:
    """Run hallucination judge via Bedrock. Returns (verdict, reason, tokens)."""
    inject_mode = uncertainty_events is not None and len(uncertainty_events) > 0

    # Build the agent system prompt section (domain policy)
    domain_policy = _get_domain_policy(domain)
    agent_system_prompt_section = ""
    if domain_policy:
        agent_system_prompt_section = (
            f"AGENT'S SYSTEM PROMPT (domain policy the agent was given):\n"
            f"<agent_policy>\n{domain_policy}\n</agent_policy>\n\n"
        )

    if inject_mode:
        system_prompt = _HALLUCINATION_JUDGE_SYSTEM_UNCERTAINTY
        n_inj = len(uncertainty_events)
        noise_types = list({e["noise_type"] for e in uncertainty_events})
        user_msg = (
            f"{agent_system_prompt_section}"
            f"TASK:\n{task_instruction}\n\n"
            f"UNCERTAINTY: {n_inj} injection(s) ({', '.join(noise_types)})\n\n"
            f"CONVERSATION:\n{_fmt_messages(messages)}"
        )
    else:
        system_prompt = _HALLUCINATION_JUDGE_SYSTEM
        user_msg = (
            f"{agent_system_prompt_section}"
            f"TASK:\n{task_instruction}\n\n"
            f"CONVERSATION:\n{_fmt_messages(messages)}"
        )

    def _call_judge(system: str, user: str, max_tok: int) -> Tuple[str, int]:
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tok,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        resp = bedrock_client.invoke_model(
            modelId="qwen.qwen3-235b-a22b-2507-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        # Qwen3 on Bedrock returns {"choices": [{"message": {"content": "..."}}]}
        choices = result.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = result.get("usage", {})
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return text, tokens

    try:
        text, tokens = _call_judge(system_prompt, user_msg, max_tok=1024)
        # Use the LAST verdict match — the judge sometimes self-corrects mid-reasoning
        vm_all = list(re.finditer(r"VERDICT\s*:\s*(\w+)", text, re.IGNORECASE))
        rm_all = list(re.finditer(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL))
        verdict = vm_all[-1].group(1).upper() if vm_all else "UNKNOWN"
        reason  = rm_all[-1].group(1).strip() if rm_all else text.strip()

        if verdict == "UNKNOWN":
            retry_system = (
                system_prompt
                + "\n\nIMPORTANT: Your response MUST start with exactly "
                "'VERDICT: NO_HALLUCINATION' or 'VERDICT: HALLUCINATION'. "
                "Do not include any preamble or analysis before the verdict line."
            )
            retry_user = user_msg + "\n\nRemember: start your reply with VERDICT: NO_HALLUCINATION or VERDICT: HALLUCINATION."
            text2, tokens2 = _call_judge(retry_system, retry_user, max_tok=512)
            tokens += tokens2
            vm2_all = list(re.finditer(r"VERDICT\s*:\s*(\w+)", text2, re.IGNORECASE))
            rm2_all = list(re.finditer(r"REASON\s*:\s*(.+)", text2, re.IGNORECASE | re.DOTALL))
            verdict = vm2_all[-1].group(1).upper() if vm2_all else "UNKNOWN"
            reason  = rm2_all[-1].group(1).strip() if rm2_all else text2.strip()

        return verdict, reason, tokens
    except Exception as e:
        return "ERROR", str(e), 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Unified result for both baseline and uncertainty episodes."""
    task_id: str
    domain: str
    model_id: str
    instruction: str
    tau_reward: float
    tau_info: Dict
    # Uncertainty fields (None/empty for baseline)
    inject_mode: bool = False
    uncertainty_seed: Optional[int] = None
    max_injections_configured: Optional[int] = None
    noise_types_configured: Optional[List[str]] = None
    total_injections_actual: int = 0
    uncertainty_events: Optional[List[Dict]] = None
    # Hallucination judge (Bedrock)
    hallucination_verdict: Optional[str] = None
    hallucination_reason: Optional[str] = None
    hallucination_judge_tokens: int = 0
    # Token usage
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    agent_total_tokens: int = 0
    user_prompt_tokens: int = 0
    user_completion_tokens: int = 0
    total_steps: int = 0
    tool_call_count: int = 0
    duration: float = 0.0
    error: Optional[str] = None
    chat_history: Optional[List[Dict]] = None
    # Recovery controller (None if not enabled)
    recovery_controller_events: Optional[List[Dict]] = None


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------

def serialize_messages(messages) -> List[Dict]:
    """Convert tau2 message objects into JSON-serializable dicts.
    
    Captures:
      - User messages: role, content, tool_calls
      - Assistant messages: role, content, tool_calls (with arguments),
        thinking/reasoning content (from raw_data or <think> tags), usage
      - Tool messages: role, tool_call_id, content, error flag
    """
    history = []
    for m in messages:
        entry: Dict[str, Any] = {}

        if isinstance(m, AssistantMessage):
            entry["role"] = "assistant"
            if m.content:
                entry["content"] = m.content
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in m.tool_calls
                ]
            # Extract thinking/reasoning content
            thinking = _extract_thinking(m)
            if thinking:
                entry["thinking"] = thinking
            # Include token usage per message
            if m.usage:
                entry["usage"] = m.usage
        elif isinstance(m, UserMessage):
            entry["role"] = "user"
            if m.content:
                entry["content"] = m.content
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
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


def _extract_thinking(msg: AssistantMessage) -> Optional[str]:
    """Extract thinking/reasoning content from an AssistantMessage.
    
    Supports two formats:
      1. Bedrock Claude extended thinking: raw_data contains content blocks
         with type="thinking".
      2. Qwen3 / vLLM: thinking is inline in content as <think>...</think> tags.
    """
    # Method 1: Check raw_data for Bedrock-style thinking blocks
    if msg.raw_data:
        # litellm response dict: choices[0].message has thinking field
        # or content blocks with type="thinking"
        choices = msg.raw_data.get("choices", [])
        if choices:
            message_data = choices[0].get("message", {})
            # Some litellm versions put reasoning_content directly
            reasoning = message_data.get("reasoning_content")
            if reasoning:
                return reasoning
        
        # Anthropic raw format: content is a list of blocks
        content_blocks = msg.raw_data.get("content", [])
        if isinstance(content_blocks, list):
            thinking_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
            if thinking_parts:
                return "\n".join(thinking_parts)
    
    # Method 2: Check for <think>...</think> tags in content (Qwen3 style)
    if msg.content and "<think>" in msg.content:
        match = re.search(r"<think>(.*?)</think>", msg.content, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    domain: str,
    task,
    agent_model_str: str,
    agent_model_kwargs: Dict,
    user_model_str: str,
    tau2_agent: str,
    run_judge_flag: bool,
    bedrock_client: Any,
    save_trajectory: bool = False,
    # Uncertainty params (all None for baseline)
    uncertainty_config: Optional[EpisodeUncertaintyConfig] = None,
    with_alternatives: bool = False,
    chosen_names: Optional[List[str]] = None,
    system_prompt_extra: Optional[str] = None,
    scenario: str = "retry_works",
    # Recovery controller (None = disabled)
    recovery_controller: Optional[Any] = None,
    # Belief provider (None = disabled)
    belief_provider: Optional[Any] = None,
) -> EpisodeResult:
    """Run one tau2 task episode. If uncertainty_config is provided, inject noise."""

    instruction = str(task.user_scenario)
    inject_mode = uncertainty_config is not None
    episode_controller = None  # Will be set if recovery_controller is enabled

    try:
        config = TextRunConfig(
            domain=domain,
            agent=tau2_agent,
            llm_agent=agent_model_str,
            llm_args_agent=agent_model_kwargs,
            llm_user=user_model_str,
            llm_args_user={},
            max_steps=MAX_STEPS,
            task_split_name=None,
            **({"retrieval_config": "bm25_grep"} if domain == "banking_knowledge" else {}),
        )

        orchestrator = build_text_orchestrator(config, task, seed=42)

        # If with_alternatives: add discoverable (alternative) tools to agent's tool list
        if with_alternatives:
            from tau2.environment.tool import as_tool as _as_tool
            disc_methods = orchestrator.environment.tools.get_discoverable_tools()
            if disc_methods:
                alt_tool_objs = [_as_tool(m) for m in disc_methods.values()]
                orchestrator.agent.tools.extend(alt_tool_objs)

        # Append extra system prompt content if provided
        if system_prompt_extra:
            agent = orchestrator.agent
            # Patch the agent's system_prompt property to include extra content
            original_prompt = agent.system_prompt
            patched_prompt = original_prompt + "\n\n" + system_prompt_extra
            # Override the property via a dynamic subclass
            agent._patched_system_prompt = patched_prompt
            agent.__class__ = type(
                agent.__class__.__name__ + "Patched",
                (agent.__class__,),
                {"system_prompt": property(lambda self: self._patched_system_prompt)},
            )

        # Wrap environment with uncertainty injection if requested
        unc_env = None
        if inject_mode:
            base_env = orchestrator.environment
            unc_env  = UncertaintyEnvironment(
                base_env=base_env,
                domain=domain,
                uncertainty_config=uncertainty_config,
                with_alternatives=with_alternatives,
                scenario=scenario,
            )
            # Wrap with recovery controller if enabled
            if recovery_controller is not None:
                from recovery_controller import (
                    ControlledUncertaintyEnvironment, RecoveryController
                )
                # Create a per-episode controller instance (thread-safe)
                # Share beliefs from the parent if persist_beliefs is enabled
                episode_controller = RecoveryController(
                    config=recovery_controller.config,
                    domain=domain,
                    with_alternatives=with_alternatives,
                    persist_beliefs=False,
                )
                if recovery_controller.persist_beliefs:
                    episode_controller.beliefs = recovery_controller.beliefs
                episode_controller.reset_episode()
                controlled_env = ControlledUncertaintyEnvironment(
                    unc_env, episode_controller
                )
                orchestrator.environment = controlled_env
            else:
                orchestrator.environment = unc_env

        # Wrap with belief provider if enabled (outermost wrapper)
        # (beliefs are now in system prompt, no environment wrapping needed)

        # Run simulation. If evaluation replay fails (e.g. model passed empty
        # args to a write tool causing state mismatch), treat as reward=0 but
        # not a fatal error — the agent did run, just failed evaluation.
        eval_replay_error = None
        simulation = orchestrator.run()
        simulation.policy = orchestrator.environment.get_policy()

        try:
            from tau2.runner.simulation import evaluate_simulation
            from tau2.evaluator.evaluator import EvaluationType
            from tau2.orchestrator.modes import CommunicationMode

            # Banking domain needs env_kwargs for retrieval config
            eval_env_kwargs = {}
            if domain == "banking_knowledge":
                eval_env_kwargs = {"retrieval_variant": "bm25_grep", "task": task}

            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
                mode=CommunicationMode.HALF_DUPLEX,
                env_kwargs=eval_env_kwargs,
            )
            simulation.reward_info = reward_info
        except ValueError as ve:
            ve_str = str(ve)
            if "Tool call:" in ve_str and ("Returned:" in ve_str or "Expected:" in ve_str):
                # Evaluation replay mismatch — agent ran but evaluator can't
                # replay state due to non-deterministic tool output (e.g. empty
                # args causing different env state). Mark as reward=0.
                eval_replay_error = ve_str
                simulation.reward_info = None
            else:
                raise

        result = simulation

        if eval_replay_error:
            tau_reward = 0.0
            tau_info = {
                "reward": 0.0,
                "eval_replay_error": eval_replay_error[:500],
            }
        else:
            tau_reward = result.reward_info.reward if result.reward_info else 0.0
            tau_info   = (
                result.reward_info.model_dump(exclude={"info"})
                if result.reward_info else {}
            )

        # Count tokens and tool calls from messages
        agent_prompt = agent_completion = 0
        user_prompt = user_completion = 0
        steps = 0
        tool_calls = 0
        for m in result.messages:
            if isinstance(m, AssistantMessage):
                if m.usage:
                    agent_prompt     += m.usage.get("prompt_tokens", 0)
                    agent_completion += m.usage.get("completion_tokens", 0)
                steps += 1
                # Count tool calls made by the agent
                if m.tool_calls:
                    tool_calls += len(m.tool_calls)
            elif isinstance(m, UserMessage) and m.usage:
                user_prompt     += m.usage.get("prompt_tokens", 0)
                user_completion += m.usage.get("completion_tokens", 0)

        ep = EpisodeResult(
            task_id=str(task.id),
            domain=domain,
            model_id=agent_model_str,
            instruction=instruction,
            tau_reward=tau_reward,
            tau_info=tau_info,
            inject_mode=inject_mode,
            uncertainty_seed=uncertainty_config.seed if inject_mode else None,
            max_injections_configured=(
                uncertainty_config.max_total_injections if inject_mode else None
            ),
            noise_types_configured=chosen_names,
            total_injections_actual=unc_env.total_injections if unc_env else 0,
            uncertainty_events=unc_env.uncertainty_events if unc_env else None,
            agent_prompt_tokens=agent_prompt,
            agent_completion_tokens=agent_completion,
            agent_total_tokens=agent_prompt + agent_completion,
            user_prompt_tokens=user_prompt,
            user_completion_tokens=user_completion,
            total_steps=steps,
            tool_call_count=tool_calls,
            duration=result.duration or 0.0,
            chat_history=serialize_messages(result.messages) if save_trajectory else None,
            recovery_controller_events=(
                episode_controller.get_episode_summary()
                if episode_controller else None
            ),
        )

        if run_judge_flag:
            verdict, reason, jtokens = run_judge(
                instruction, result.messages, bedrock_client,
                domain=domain,
                uncertainty_events=unc_env.uncertainty_events if unc_env else None,
            )
            ep.hallucination_verdict = verdict
            ep.hallucination_reason  = reason
            ep.hallucination_judge_tokens = jtokens

        # Update recovery controller beliefs post-episode (for learning mode)
        if episode_controller is not None:
            episode_controller.update_beliefs_post_episode(
                tau_reward=tau_reward,
                hallucination_verdict=ep.hallucination_verdict,
                uncertainty_events=unc_env.uncertainty_events if unc_env else None,
            )

        return ep

    except Exception as e:
        tb = traceback.format_exc()
        sys.__stderr__.write(f"\n[task={task.id}] EXCEPTION:\n{tb}\n")
        sys.__stderr__.flush()
        return EpisodeResult(
            task_id=str(task.id),
            domain=domain,
            model_id=agent_model_str,
            instruction=instruction,
            tau_reward=0.0,
            tau_info={"error": str(e), "traceback": tb},
            inject_mode=inject_mode,
            noise_types_configured=chosen_names,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[EpisodeResult]) -> Dict:
    n = len(results)
    if n == 0:
        return {}

    tau_rewards = [r.tau_reward for r in results]
    tau_pass    = sum(1 for r in tau_rewards if r >= 1.0 - 1e-6)

    hallucination_evaluated = sum(
        1 for r in results if r.hallucination_verdict not in (None, "ERROR")
    )
    hallucination_count = sum(
        1 for r in results
        if r.hallucination_verdict and r.hallucination_verdict.startswith("HALLUCINATION")
    )

    avg_steps = sum(r.total_steps for r in results) / n
    avg_tool_calls = sum(r.tool_call_count for r in results) / n
    avg_agent_tokens = sum(r.agent_total_tokens for r in results) / n
    avg_duration = sum(r.duration for r in results) / n

    metrics = {
        "n_tasks": n,
        "tau_bench": {
            "pass_count": tau_pass,
            "pass_rate":  round(tau_pass / n, 4),
            "avg_reward": round(sum(tau_rewards) / n, 4),
        },
        "hallucination_judge": {
            "evaluated":           hallucination_evaluated,
            "hallucination_count": hallucination_count,
            "hallucination_rate":  round(hallucination_count / hallucination_evaluated, 4)
                                   if hallucination_evaluated else None,
        },
        "efficiency": {
            "avg_steps":        round(avg_steps, 2),
            "avg_tool_calls":   round(avg_tool_calls, 2),
            "avg_agent_tokens": round(avg_agent_tokens, 1),
            "avg_duration_s":   round(avg_duration, 2),
        },
    }

    # Add uncertainty metrics if applicable
    inject_mode = any(r.inject_mode for r in results)
    if inject_mode:
        avg_inj = sum(r.total_injections_actual for r in results) / n
        metrics["uncertainty"] = {
            "avg_injections_per_episode": round(avg_inj, 2),
        }

    return metrics


def print_metrics(label: str, metrics: Dict) -> None:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    tb = metrics.get("tau_bench", {})
    hj = metrics.get("hallucination_judge", {})
    uc = metrics.get("uncertainty", {})
    ef = metrics.get("efficiency", {})
    n  = metrics.get("n_tasks", 0)
    print(f"  Tasks evaluated  : {n}")
    print(f"  Tau-bench pass   : {tb.get('pass_count',0)}/{n}  "
          f"({tb.get('pass_rate',0)*100:.1f}%)  "
          f"avg_reward={tb.get('avg_reward',0):.3f}")
    if hj.get("hallucination_rate") is not None:
        print(f"  Hallucination    : {hj.get('hallucination_count',0)}"
              f"/{hj.get('evaluated',0)}  "
              f"({hj.get('hallucination_rate',0)*100:.1f}%)")
    if uc:
        print(f"  Avg injections   : {uc.get('avg_injections_per_episode','?')}")
    print(f"  Avg steps        : {ef.get('avg_steps','?')}")
    print(f"  Avg tool calls   : {ef.get('avg_tool_calls','?')}")
    print(f"  Avg agent tokens : {ef.get('avg_agent_tokens','?')}")
    print(f"  Avg duration (s) : {ef.get('avg_duration_s','?')}")


# ---------------------------------------------------------------------------
# Domain runner
# ---------------------------------------------------------------------------

def run_domain(
    domain: str,
    task_split: str,
    task_ids: Optional[List[str]],
    start: int,
    end: int,
    agent_model_str: str,
    agent_model_kwargs: Dict,
    user_model_str: str,
    tau2_agent: str,
    concurrency: int,
    run_judge_flag: bool,
    results_dir: str,
    save_trajectory: bool = False,
    # Uncertainty params (all None/defaults for baseline)
    inject_mode: bool = False,
    global_seed: int = 42,
    fixed_max_injections: Optional[int] = None,
    allowed_noise_types: Optional[List[NoiseType]] = None,
    timing: TriggerTiming = TriggerTiming.UNIFORM,
    with_alternatives: bool = False,
    pass_idx: int = 0,
    resume_file: Optional[str] = None,
    system_prompt_extra: Optional[str] = None,
    allow_impossible: bool = False,
    # Recovery controller
    recovery_controller: Optional[Any] = None,
    # Belief provider
    belief_provider: Optional[Any] = None,
) -> List[EpisodeResult]:

    task_set = DOMAIN_TASK_SET[domain]

    if task_ids:
        all_tasks = get_tasks(task_set, task_split_name=task_split, task_ids=task_ids)
    else:
        all_tasks = get_tasks(task_set, task_split_name=task_split)
        end_idx   = len(all_tasks) if end == -1 else min(end, len(all_tasks))
        all_tasks = all_tasks[start:end_idx]

    mode_tag = "baseline"
    if inject_mode:
        mode_tag = "with_alt" if with_alternatives else "no_alt"

    # --- Resume logic: load prior results and skip completed tasks ---
    prior_results: List[EpisodeResult] = []
    if resume_file and os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            prior_data = json.load(f)
        completed_ids = {str(r["task_id"]) for r in prior_data}
        # Reconstruct EpisodeResult objects from prior data for metrics
        for r in prior_data:
            # Minimal reconstruction — just what's needed for metrics/saving
            prior_results.append(EpisodeResult(**{
                k: v for k, v in r.items()
                if k in EpisodeResult.__dataclass_fields__
            }))
        original_count = len(all_tasks)
        all_tasks = [t for t in all_tasks if str(t.id) not in completed_ids]
        print(f"[RESUME] Loaded {len(completed_ids)} completed tasks from {resume_file}")
        print(f"[RESUME] Remaining: {len(all_tasks)}/{original_count} tasks to run")
    # -----------------------------------------------------------------

    print(f"\n[{domain.upper()}] model={agent_model_str}  "
          f"tasks={len(all_tasks)}  split={task_split}  "
          f"concurrency={concurrency}  mode={mode_tag}  pass={pass_idx}")
    if inject_mode:
        print(f"  uncertainty: seed={global_seed}  "
              f"max_inj={'random 1-8' if fixed_max_injections is None else fixed_max_injections}  "
              f"noise={'random' if allowed_noise_types is None else [t.value for t in allowed_noise_types]}  "
              f"timing={timing.value}")

    bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    results: List[EpisodeResult] = list(prior_results)  # Start with prior results

    # Prepare output path for incremental saving
    # Strip litellm provider prefix (openai/, bedrock/, etc.) for cleaner folder names
    model_name_clean = re.sub(r"^[a-zA-Z]+/", "", agent_model_str)
    model_slug = re.sub(r"[/:.]", "_", model_name_clean)
    prefix     = "uncertainty" if inject_mode else "baseline"
    model_dir  = os.path.join(results_dir, model_slug)
    # If resuming, write back to the same file; otherwise create a new one
    if resume_file and os.path.exists(resume_file):
        out_path = resume_file
    else:
        ts = datetime.now().strftime("%m%d_%H%M%S")
        out_path = os.path.join(
            model_dir, f"{prefix}_{domain}_{mode_tag}_pass{pass_idx}_{ts}.json"
        )
    os.makedirs(model_dir, exist_ok=True)

    # Use a lock for thread-safe incremental writes
    _write_lock = threading.Lock()

    def _save_incremental(result_list: List[EpisodeResult]) -> None:
        """Atomically rewrite the results file with all completed results so far."""
        sorted_results = sorted(result_list, key=lambda r: r.task_id)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in sorted_results], f, indent=2)

    def _run_one(task) -> EpisodeResult:
        # Multi-instance round-robin: pick next api_base for this task
        task_model_kwargs = agent_model_kwargs
        if agent_model_kwargs.get("_multi_instance"):
            task_model_kwargs = {
                k: v for k, v in agent_model_kwargs.items()
                if not k.startswith("_")
            }
            task_model_kwargs["api_base"] = agent_model_kwargs["_get_next_api_base"]()

        # Build uncertainty config per-task if in inject mode
        uc_config = None
        chosen_names: Optional[List[str]] = None
        task_scenario = "retry_works"
        if inject_mode:
            task_seed = global_seed + hash(str(task.id)) % 100000
            task_rng  = random.Random(task_seed)
            uc_config, _, chosen_names = build_episode_uncertainty_config(
                domain=domain,
                seed=task_seed,
                max_injections=fixed_max_injections,
                allowed_noise_types=allowed_noise_types,
                timing=timing,
                rng=task_rng,
            )
            # Sample scenario for this episode (only with_alternatives enables Scenario 2/3)
            if with_alternatives:
                scenario_roll = task_rng.random()
                if scenario_roll < 0.50:
                    task_scenario = "retry_works"      # Scenario 1: 50%
                elif scenario_roll < 0.85:
                    task_scenario = "switch_needed"    # Scenario 2: 35%
                elif allow_impossible:
                    task_scenario = "impossible"       # Scenario 3: 15% (training only)
                else:
                    task_scenario = "switch_needed"    # Eval: Scenario 3 → fallback to Scenario 2

        t0  = time.time()
        res = run_episode(
            domain=domain,
            task=task,
            agent_model_str=agent_model_str,
            agent_model_kwargs=task_model_kwargs,
            user_model_str=user_model_str,
            tau2_agent=tau2_agent,
            run_judge_flag=run_judge_flag,
            bedrock_client=bedrock_client,
            save_trajectory=save_trajectory,
            uncertainty_config=uc_config,
            with_alternatives=with_alternatives,
            chosen_names=chosen_names,
            system_prompt_extra=system_prompt_extra,
            scenario=task_scenario,
            recovery_controller=recovery_controller,
            belief_provider=belief_provider,
        )
        elapsed = time.time() - t0
        icon = "✅" if res.tau_reward >= 1.0 - 1e-6 else "❌"
        hall_str = f"  hall={res.hallucination_verdict}" if res.hallucination_verdict else ""
        err_str   = f"  ERR={res.error[:60]}" if res.error else ""
        inj_str   = f"  inj={res.total_injections_actual}" if inject_mode else ""
        noise_str = (
            f"  noise={','.join(res.noise_types_configured)}"
            if inject_mode and res.noise_types_configured else ""
        )
        print(
            f"  {icon} task={str(res.task_id):>6}  tau={res.tau_reward:.1f}"
            f"{inj_str}{noise_str}{hall_str}  steps={res.total_steps}  "
            f"tools={res.tool_call_count}  tokens={res.agent_total_tokens}  "
            f"{elapsed:.1f}s{err_str}",
            flush=True,
        )
        return res

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_run_one, task): task for task in all_tasks}
        for fut in as_completed(futures):
            ep = fut.result()
            with _write_lock:
                results.append(ep)
                _save_incremental(results)

    results.sort(key=lambda r: r.task_id)
    # Final save (sorted, complete)
    _save_incremental(results)
    print(f"  💾 saved → {out_path}  ({len(results)} tasks)")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified inference runner (baseline + uncertainty). "
                    "Supports multiple benchmarks via --benchmark flag."
    )
    parser.add_argument("--benchmark", default="tau2-bench",
                        help="Benchmark to run (default: tau2-bench). "
                             "See eval/runners/ for available benchmarks.")
    parser.add_argument("--env", default="all")
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=-1)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--agent", default="llm_agent")
    # Backend
    parser.add_argument("--backend", default="bedrock", choices=["bedrock", "local"])
    parser.add_argument("--model", default=None,
                        help="Override agent model ID (e.g. 'bedrock/zai.glm-5'). "
                             "Defaults to DEFAULT_BEDROCK_AGENT_MODEL for bedrock backend.")
    parser.add_argument("--local-model", default="qwen3-14b",
                        choices=list(LOCAL_MODELS.keys()))
    parser.add_argument("--local-port", type=int, default=None)
    parser.add_argument("--num-instances", type=int, default=1,
                        help="Number of vLLM instances (ports base_port to base_port+N-1)")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Agent sampling temperature (default: 0.0 for local, 1.0 for bedrock thinking)")
    parser.add_argument("--system-prompt-extra", type=str, default=None,
                        help="Path to a text file whose content is appended to the agent system prompt")
    parser.add_argument("--no-trajectory", action="store_true",
                        help="Disable saving full chat history to results (saved by default)")
    parser.add_argument("--num-passes", type=int, default=1,
                        help="Number of passes to run per domain (default: 1)")
    # Uncertainty options
    parser.add_argument("--inject", action="store_true",
                        help="Enable uncertainty injection (without this flag, runs as baseline)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-injections", type=int, default=None)
    parser.add_argument(
        "--noise-types", nargs="+",
        choices=["timeout", "rate_limit", "server_error", "auth_error",
                 "malformed_response", "schema_drift", "partial", "stale", "factual_error"],
        default=None,
    )
    parser.add_argument(
        "--timing", default="uniform",
        choices=["early", "middle", "late", "uniform"],
    )
    parser.add_argument("--with-alternatives", action="store_true")
    parser.add_argument("--allow-impossible", action="store_true",
                        help="Allow Scenario 3 (impossible tasks). Use for training only, not eval.")
    parser.add_argument("--recovery-controller", action="store_true",
                        help="Enable attribution-aware Bayesian recovery controller. "
                             "Injects structured recovery guidance into tool error messages.")
    parser.add_argument("--recovery-controller-verbose", action="store_true",
                        help="Include full belief details in recovery controller prompts.")
    parser.add_argument("--recovery-controller-persist", action="store_true",
                        help="Persist controller beliefs across episodes (learning mode).")
    parser.add_argument("--belief-provider", default=None,
                        help="Path to precomputed beliefs JSON. Appends tool reliability "
                             "context to every read-tool response.")
    parser.add_argument("--resume", default=None,
                        help="Path to a partial results JSON file to resume from. "
                             "Already-completed task IDs will be skipped.")
    args = parser.parse_args()

    domains        = ALL_DOMAINS if args.env == "all" else [args.env]
    # Validate domain against benchmark (tau2-bench for now, extensible via runners)
    if args.benchmark != "tau2-bench":
        print(f"[WARN] --benchmark={args.benchmark} not yet integrated into this script. "
              f"See eval/runners/ for the new runner-based interface.")
        print(f"[INFO] Falling back to tau2-bench.")
    run_judge_flag = not args.no_judge
    inject_mode    = args.inject
    timing         = TriggerTiming(args.timing)
    with_alternatives = args.with_alternatives
    allowed_noise  = (
        [NoiseType(t) for t in args.noise_types]
        if args.noise_types else None
    )

    # Build agent model string
    if args.backend == "local":
        served_name, default_port = LOCAL_MODELS[args.local_model]
        port = args.local_port or default_port
        num_instances = args.num_instances

        agent_cfg = BackendConfig.for_local(
            args.local_model, port=port, enable_thinking=args.thinking
        )
        # Override temperature if specified
        if args.temperature is not None:
            agent_cfg.temperature = args.temperature
        agent_model_str    = f"openai/{agent_cfg.model_id}"
        agent_model_kwargs = {
            "api_base": agent_cfg.base_url,
            "api_key":  agent_cfg.api_key,
        }
        if not args.thinking:
            agent_model_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        # Multi-instance round-robin: store list of base URLs
        if num_instances > 1:
            _instance_urls = [
                f"http://localhost:{port + i}/v1" for i in range(num_instances)
            ]
            _instance_counter = [0]  # mutable counter for round-robin

            def _get_next_api_base():
                idx = _instance_counter[0] % len(_instance_urls)
                _instance_counter[0] += 1
                return _instance_urls[idx]

            agent_model_kwargs["_multi_instance"] = True
            agent_model_kwargs["_get_next_api_base"] = _get_next_api_base
            agent_model_kwargs["_instance_urls"] = _instance_urls
            print(f"[INFO] Backend: local vLLM  model={agent_cfg.model_id}  "
                  f"instances={num_instances}  ports={port}-{port+num_instances-1}  "
                  f"thinking={args.thinking}")
        else:
            print(f"[INFO] Backend: local vLLM  model={agent_cfg.model_id}  "
                  f"url={agent_cfg.base_url}  thinking={args.thinking}")
    else:
        agent_model_str    = args.model if args.model else DEFAULT_BEDROCK_AGENT_MODEL
        agent_model_kwargs = {}
        if args.thinking:
            agent_model_kwargs = {
                "thinking": {"type": "enabled", "budget_tokens": 8000},
                "temperature": 1.0,
                "max_tokens": 16000,
            }
        print(f"[INFO] Backend: AWS Bedrock  model={agent_model_str}  thinking={args.thinking}")

    user_model_str = BEDROCK_USER_MODEL
    mode_label = "uncertainty" if inject_mode else "baseline"
    print(f"[INFO] Mode: {mode_label}")
    print(f"[INFO] User simulator: {user_model_str}")

    # Load extra system prompt if specified
    system_prompt_extra = None
    if args.system_prompt_extra:
        prompt_path = args.system_prompt_extra
        # Auto-select variant based on with_alternatives flag
        if with_alternatives and "uncertainty_guidance" in prompt_path:
            alt_prompt_path = prompt_path.replace("uncertainty_guidance.txt", "uncertainty_guidance_with_alt.txt")
            if os.path.exists(alt_prompt_path):
                prompt_path = alt_prompt_path
                print(f"[INFO] Auto-selected with-alt prompt variant: {alt_prompt_path}")
        elif not with_alternatives and "uncertainty_guidance" in prompt_path:
            no_alt_prompt_path = prompt_path.replace("uncertainty_guidance.txt", "uncertainty_guidance_no_alt.txt")
            if os.path.exists(no_alt_prompt_path):
                prompt_path = no_alt_prompt_path
                print(f"[INFO] Auto-selected no-alt prompt variant: {no_alt_prompt_path}")
        with open(prompt_path, "r") as f:
            system_prompt_extra = f.read().strip()
        print(f"[INFO] Extra system prompt loaded from: {prompt_path} ({len(system_prompt_extra)} chars)")

    # --- Recovery Controller setup ---
    rc_instance = None
    if args.recovery_controller and inject_mode:
        from recovery_controller import create_recovery_controller
        rc_instance = create_recovery_controller(
            domain=domains[0] if len(domains) == 1 else "retail",
            with_alternatives=with_alternatives,
            persist_beliefs=args.recovery_controller_persist,
            seed=args.seed,
            verbose=args.recovery_controller_verbose,
        )
        print(f"[INFO] Recovery controller: ENABLED  "
              f"persist={args.recovery_controller_persist}  "
              f"verbose={args.recovery_controller_verbose}")
    elif args.recovery_controller and not inject_mode:
        print("[WARN] --recovery-controller requires --inject. Ignoring.")

    # --- Belief Provider setup ---
    bp_instance = None
    _belief_provider_path = args.belief_provider if hasattr(args, 'belief_provider') and args.belief_provider else None

    all_summary: Dict[str, Dict] = {}

    for domain in domains:
        # Re-instantiate belief provider per domain (domain-specific fallback maps)
        if _belief_provider_path:
            from belief_provider import BeliefProvider
            bp_instance = BeliefProvider.from_precomputed(
                _belief_provider_path,
                with_alternatives=with_alternatives,
                domain=domain,
            )
            belief_block = bp_instance.get_system_prompt_block()
            if args.system_prompt_extra:
                _sp_extra = args.system_prompt_extra + "\n\n" + belief_block
            else:
                _sp_extra = belief_block
            if domain == domains[0]:
                print(f"[INFO] Belief provider: ENABLED  "
                      f"with_alt={with_alternatives}  domain={domain}  "
                      f"belief_block={len(belief_block)} chars appended to system prompt")
        else:
            bp_instance = None
            _sp_extra = system_prompt_extra
        # Update recovery controller domain if running multiple domains
        if rc_instance is not None:
            from recovery_controller import DOMAIN_ALTERNATIVES
            rc_instance.domain = domain
            rc_instance.alternatives = DOMAIN_ALTERNATIVES.get(domain, {})

        for pass_idx in range(args.num_passes):
            mode_tag = mode_label
            if inject_mode:
                mode_tag = "with_alt" if with_alternatives else "no_alt"
            label = f"{domain} / {agent_model_str} ({mode_tag}) pass={pass_idx}"

            results = run_domain(
                domain=domain,
                task_split=args.task_split,
                task_ids=args.task_ids,
                start=args.start,
                end=args.end,
                agent_model_str=agent_model_str,
                agent_model_kwargs=agent_model_kwargs,
                user_model_str=user_model_str,
                tau2_agent=args.agent,
                concurrency=args.concurrency,
                run_judge_flag=run_judge_flag,
                results_dir=args.results_dir,
                save_trajectory=not args.no_trajectory,
                inject_mode=inject_mode,
                global_seed=args.seed,
                fixed_max_injections=args.max_injections,
                allowed_noise_types=allowed_noise,
                timing=timing,
                with_alternatives=with_alternatives,
                pass_idx=pass_idx,
                resume_file=args.resume,
                system_prompt_extra=_sp_extra,
                allow_impossible=args.allow_impossible,
                recovery_controller=rc_instance,
                belief_provider=bp_instance,
            )
            metrics = compute_metrics(results)
            all_summary[label] = metrics
            print_metrics(label, metrics)

    # Summary table
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    has_inj = inject_mode
    if has_inj:
        hdr = f"{'Domain':<44}  {'Tau%':>6}  {'Hall%':>7}  {'Inj/ep':>7}  {'Tools':>6}  {'Tokens':>8}"
    else:
        hdr = f"{'Domain / Model':<52}  {'Tau%':>6}  {'Hall%':>7}  {'Tools':>6}  {'Tokens':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, m in all_summary.items():
        tb = m.get("tau_bench", {})
        hj = m.get("hallucination_judge", {})
        uc = m.get("uncertainty", {})
        ef = m.get("efficiency", {})
        tau_pct   = f"{tb.get('pass_rate', 0)*100:.1f}%"
        hall_pct = (
            f"{hj.get('hallucination_rate', 0)*100:.1f}%"
            if hj.get("hallucination_rate") is not None else "  n/a"
        )
        tools  = ef.get("avg_tool_calls", "?")
        tokens = ef.get("avg_agent_tokens", "?")
        if has_inj:
            inj = uc.get("avg_injections_per_episode", "?")
            print(f"{label:<44}  {tau_pct:>6}  {hall_pct:>7}  {inj:>7}  {tools:>6}  {tokens:>8}")
        else:
            print(f"{label:<52}  {tau_pct:>6}  {hall_pct:>7}  {tools:>6}  {tokens:>8}")


if __name__ == "__main__":
    main()
