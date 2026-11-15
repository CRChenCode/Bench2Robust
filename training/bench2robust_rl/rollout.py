"""
bench2robust_rl rollout.py
-----------------------
generate / reward_func entry points for slime framework.

Runs full tau2-bench multi-turn simulation with bench2robust uncertainty injection.
The SGLang engine generates agent responses, while the tau2 environment handles
tool execution and the user simulator (via Bedrock).

All tau2 logic runs IN-PROCESS (no subprocess) to avoid dependency issues in docker.

Hooks:
    --custom-generate-function-path   bench2robust_rl.rollout.generate
    --custom-rm-path                  bench2robust_rl.rollout.reward_func
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from agentflow.core.llm_engine import SGLangEngine, GenerationOutput
from bench2robust_rl.reward import reward_func  # noqa: F401

logger = logging.getLogger(__name__)

# Ensure tau2 and bench2robust are importable
_TAU2_PATH = "<PATH>/benchmarks/tau2-bench/src"
_VITA_PATH = "<PATH>/benchmarks/vitabench/src"
_PROJECT_ROOT = "<PATH>"
if _TAU2_PATH not in sys.path:
    sys.path.insert(0, _TAU2_PATH)
if _VITA_PATH not in sys.path:
    sys.path.insert(0, _VITA_PATH)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("TAU2_DATA_DIR", "<PATH>/benchmarks/tau2-bench/data")

ROLLOUT_SAMPLE_TIMEOUT = 900  # 10 min per episode (30 steps × ~20s/step)

# VitaBench generate patch flag
_vita_generate_patched = False


def _patch_vita_generate():
    """
    Replace VitaBench's custom HTTP-based generate() with litellm-based one
    that supports Bedrock and other providers.
    """
    global _vita_generate_patched
    if _vita_generate_patched:
        return
    _vita_generate_patched = True

    import litellm
    import vita.utils.llm_utils as vita_llm
    from vita.data_model.message import AssistantMessage as _VAssist, ToolCall as _VTC

    def litellm_generate(*args, **kwargs):
        model = kwargs.pop("model", args[0] if args else None)
        messages = kwargs.pop("messages", args[1] if len(args) > 1 else [])
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        enable_think = kwargs.pop("enable_think", False)
        kwargs.pop("api_base", None)
        kwargs.pop("num_retries", None)
        kwargs.pop("seed", None)
        max_tokens = kwargs.pop("max_tokens", 8192)
        temperature = kwargs.pop("temperature", 0.0)

        messages_formatted = vita_llm.format_messages(messages)
        tools_formatted = [tool.openai_schema for tool in tools] if tools else None

        call_kwargs = {
            "model": model,
            "messages": messages_formatted,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_retries": 3,
        }
        if tools_formatted:
            call_kwargs["tools"] = tools_formatted
            call_kwargs["tool_choice"] = tool_choice or "auto"

        if not enable_think and "qwen" in model.lower():
            for msg in call_kwargs["messages"]:
                if msg["role"] == "user":
                    if "/no_think" not in msg["content"]:
                        msg["content"] = msg["content"] + "\n/no_think"
                    break

        response = litellm.completion(**call_kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls_out = None
        if msg.tool_calls:
            tool_calls_out = []
            for tc in msg.tool_calls:
                tc_args = tc.function.arguments
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {}
                tool_calls_out.append(_VTC(
                    id=tc.id or f"call_{random.randint(0,9999)}",
                    name=tc.function.name,
                    arguments=tc_args,
                    requestor="assistant",
                ))

        content = msg.content or ""
        result = _VAssist(
            role="assistant",
            content=content,
            tool_calls=tool_calls_out,
        )
        # Add cost/usage info if available
        if hasattr(response, 'usage') and response.usage:
            result.usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return result

    # Monkey-patch
    vita_llm.generate = litellm_generate
    # Also patch all modules that import generate directly
    import vita.user.user_simulator as _vus
    _vus.generate = litellm_generate
    try:
        import vita.evaluator.evaluator_traj as _vtraj
        _vtraj.generate = litellm_generate
    except (ImportError, AttributeError):
        pass
    try:
        import vita.evaluator.evaluator as _veval
        if hasattr(_veval, 'generate'):
            _veval.generate = litellm_generate
    except (ImportError, AttributeError):
        pass
    logger.info("[vita] Patched VitaBench generate() with litellm backend")

# Limit concurrent engine.generate() calls to avoid starving SGLang engines
# 8 SGLang engines × 4 concurrent requests each = 32 max concurrent generations
_GENERATE_SEMAPHORE = None

def _get_generate_semaphore():
    global _GENERATE_SEMAPHORE
    if _GENERATE_SEMAPHORE is None:
        _GENERATE_SEMAPHORE = asyncio.Semaphore(24)  # 24 concurrent episodes
    return _GENERATE_SEMAPHORE


async def _direct_generate(url: str, tokenizer, messages: list, sampling_params: dict, enable_thinking: bool = True, tools: list = None):
    """
    Direct HTTP call to SGLang /generate endpoint.
    Bypasses slime's http_utils.post (which has abort/retry logic that conflicts with multi-turn).
    Returns a GenerationOutput.
    """
    import aiohttp

    # Apply chat template with tools (Qwen3 injects tool definitions into prompt)
    template_kwargs = {"enable_thinking": enable_thinking}
    if tools:
        template_kwargs["tools"] = tools

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    payload = {
        "text": prompt_text,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    timeout = aiohttp.ClientTimeout(total=90)
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"SGLang returned {resp.status}: {await resp.text()}")
            out = await resp.json()

    meta = out["meta_info"]
    finish_reason = meta["finish_reason"]["type"]
    response = out["text"]
    token_ids = [item[1] for item in meta["output_token_logprobs"]]
    log_probs = [item[0] for item in meta["output_token_logprobs"]]

    return GenerationOutput(
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        response=response,
        token_ids=token_ids,
        log_probs=log_probs,
        finish_reason=finish_reason,
    )

# Uncertainty guidance prompts
UNCERTAINTY_GUIDANCE_NO_ALT = """<uncertainty_handling>
## Error Recovery Strategy
When a tool call returns an error, you must decide between three strategies:
1. **Retry**: Call the same tool again (useful if the error seems transient)
2. **Workaround**: Use a different combination of available tools to obtain the same information
3. **Abstain**: Transfer to a human agent (only when no path forward exists)

Choose your strategy by considering:
- Has this tool already failed before in this conversation? If yes, retrying is less likely to help.
- Is there another tool or combination of tools that can give you equivalent information?
- Efficiency matters: resolve the issue in as few steps as possible.

## Available Workarounds
- find_user_id_by_name_zip ↔ find_user_id_by_email: mutual alternatives for authentication.
- get_product_details failing: list_all_product_types can confirm product existence (but not variants).
- get_item_details failing: get_product_details returns all variants including this item.
- get_order_details failing: get_user_details shows which orders belong to the user.

## Factual Accuracy Rules (CRITICAL)
- NEVER state a price, amount, or balance unless it comes directly from a tool response.
- NEVER claim a product has specific attributes (color, material, size) unless you see them in tool output.
- NEVER say an item is "unavailable" or "not in stock" unless a tool explicitly shows available=false.
- For ANY arithmetic (price differences, refunds, totals): verify your computation before stating it to the user.
- If you are unsure about a fact: call the relevant tool to verify. Do NOT guess.

## When to Transfer
Transfer to a human agent only when you have exhausted available approaches and cannot obtain the critical information needed to proceed.
</uncertainty_handling>"""

UNCERTAINTY_GUIDANCE_WITH_ALT = """<uncertainty_handling>
## Error Recovery Strategy
When a tool call returns an error, you must decide between three strategies:
1. **Retry**: Call the same tool again (useful if the error seems transient and this is the first failure)
2. **Try Alternative**: Switch to a different tool that provides equivalent information (preferred when a tool has already failed once)
3. **Abstain**: Transfer to a human agent (only when both primary and alternative paths have been exhausted)

Choose your strategy by considering:
- Has this tool already failed before in this conversation? If yes, switch to an alternative immediately.
- Is there an alternative tool listed below that provides the same information? If yes, prefer switching over retrying.
- Efficiency matters: resolve the issue in as few steps as possible. Do not repeatedly retry a failing tool.

## Alternative Tool Paths
- find_user_id_by_name_zip ↔ find_user_id_by_email ↔ get_user_by_order: three ways to authenticate.
- get_order_details ↔ get_orders_for_user: both provide order information.
- get_product_details ↔ search_product_by_name: both return full product variants (different lookup key).
- get_item_details ↔ get_item_by_product_and_options: both find specific item variants.
- check_order_item_availability: combines order + product lookup for exchange/modify scenarios.

## Factual Accuracy Rules (CRITICAL)
- NEVER state a price, amount, or balance unless it comes directly from a tool response.
- NEVER claim a product has specific attributes (color, material, size) unless you see them in tool output.
- NEVER say an item is "unavailable" or "not in stock" unless a tool explicitly shows available=false.
- For ANY arithmetic (price differences, refunds, totals): verify your computation before stating it to the user.
- If you are unsure about a fact: call the relevant tool to verify. Do NOT guess.

## When to Transfer
Transfer to a human agent only when you have exhausted both primary and alternative tool paths and cannot obtain the critical information needed to proceed.
</uncertainty_handling>"""

# Default (backwards compatible)
UNCERTAINTY_GUIDANCE = UNCERTAINTY_GUIDANCE_NO_ALT

# Lazy-loaded tau2 modules
_tau2_loaded = False

# ---------------------------------------------------------------------------
# Belief Provider (replaces static UNCERTAINTY_GUIDANCE in system prompt)
# ---------------------------------------------------------------------------
_belief_provider_with_alt = None
_belief_provider_no_alt = None


def _get_belief_block(with_alternatives: bool, domain: str = "retail") -> str:
    """
    Get belief-based system prompt block.
    Uses precomputed beliefs (static in Phase 1-3, dynamic in Phase 4+).
    Falls back to UNCERTAINTY_GUIDANCE if belief file not found.
    
    Ablation D: Set BENCH2ROBUST_NO_BTM_TRAINING=1 to disable beliefs during training.
    """
    # Ablation D: disable BTM during training
    if os.environ.get("BENCH2ROBUST_NO_BTM_TRAINING", "0") == "1":
        return ""  # No belief block injected

    global _belief_provider_with_alt, _belief_provider_no_alt

    beliefs_path = os.environ.get("BENCH2ROBUST_BELIEFS_PATH", "<PATH>/eval/src/precomputed_beliefs.json")

    try:
        if with_alternatives:
            if _belief_provider_with_alt is None:
                sys.path.insert(0, "<PATH>/eval/src")
                from belief_provider import BeliefProvider
                _belief_provider_with_alt = BeliefProvider.from_precomputed(
                    beliefs_path, with_alternatives=True, domain=domain
                )
            return _belief_provider_with_alt.get_system_prompt_block()
        else:
            if _belief_provider_no_alt is None:
                sys.path.insert(0, "<PATH>/eval/src")
                from belief_provider import BeliefProvider
                _belief_provider_no_alt = BeliefProvider.from_precomputed(
                    beliefs_path, with_alternatives=False, domain=domain
                )
            return _belief_provider_no_alt.get_system_prompt_block()
    except Exception as e:
        logger.warning("[beliefs] Failed to load belief provider: %s. Falling back to static prompt.", e)
        return UNCERTAINTY_GUIDANCE_WITH_ALT if with_alternatives else UNCERTAINTY_GUIDANCE_NO_ALT


def _ensure_tau2():
    """Lazy-load tau2 modules (heavy imports)."""
    global _tau2_loaded
    if _tau2_loaded:
        return
    # Silence noisy loggers
    for name in ("LiteLLM", "httpx", "httpcore", "openai", "litellm"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        from loguru import logger as _loguru
        _loguru.remove()
    except ImportError:
        pass
    _tau2_loaded = True


_last_cred_refresh = 0


def _refresh_aws_credentials():
    """Refresh AWS credentials from host-written file. Called every episode."""
    global _last_cred_refresh
    import time as _time

    # Only refresh every 5 minutes (avoid file I/O on every episode)
    now = _time.time()
    if now - _last_cred_refresh < 300:
        return
    _last_cred_refresh = now

    # Try reading from host-written creds file (updated by cron every 30 min)
    _creds_file = "<PATH>/<CREDS_FILE>"
    try:
        if os.path.exists(_creds_file):
            import json as _json
            with open(_creds_file) as _f:
                _cdata = _json.load(_f)
            if _cdata.get("AWS_ACCESS_KEY_ID"):
                os.environ["AWS_ACCESS_KEY_ID"] = _cdata["AWS_ACCESS_KEY_ID"]
                os.environ["AWS_SECRET_ACCESS_KEY"] = _cdata["AWS_SECRET_ACCESS_KEY"]
                os.environ["AWS_SESSION_TOKEN"] = _cdata.get("AWS_SESSION_TOKEN", "")
                os.environ["AWS_DEFAULT_REGION"] = _cdata.get("AWS_DEFAULT_REGION", "<REGION>")
                logger.info("[aws] Credentials refreshed from %s: %s...",
                           _creds_file, _cdata["AWS_ACCESS_KEY_ID"][:12])
                return
    except Exception as _e:
        logger.warning("[aws] Failed to read %s: %s", _creds_file, _e)

    # Fallback: try IMDS directly
    try:
        import requests as _req
        _token = _req.put("<IMDS_ENDPOINT>/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, timeout=2).text
        _role = _req.get("<IMDS_ENDPOINT>/latest/meta-data/iam/security-credentials/",
            headers={"X-aws-ec2-metadata-token": _token}, timeout=2).text
        _creds = _req.get(f"<IMDS_ENDPOINT>/latest/meta-data/iam/security-credentials/{_role}",
            headers={"X-aws-ec2-metadata-token": _token}, timeout=2).json()
        os.environ["AWS_ACCESS_KEY_ID"] = _creds["AccessKeyId"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = _creds["SecretAccessKey"]
        os.environ["AWS_SESSION_TOKEN"] = _creds["Token"]
        os.environ["AWS_DEFAULT_REGION"] = "<REGION>"
        logger.info("[aws] Credentials from IMDS: %s...", _creds["AccessKeyId"][:12])
    except Exception as _e:
        logger.warning("[aws] IMDS fallback failed: %s", _e)


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """
    slime framework entry point.

    Runs a full tau2-bench episode:
    1. Load task from retail_combined dataset
    2. Build tau2 orchestrator (user simulator + environment)
    3. Agent generates responses via SGLang engine (multi-turn)
    4. Optionally inject uncertainty via bench2robust
    5. Return full trajectory with per-turn tokens/log_probs
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    engine = SGLangEngine(
        url=url,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        enable_thinking=True,
    )

    meta = sample.metadata if isinstance(sample.metadata, dict) else {}

    # Per-rollout seed variation: each rollout of the same prompt gets a different injection pattern
    # This creates reward variance for GRPO even within the same prompt group
    rollout_seed_offset = getattr(sample, 'index', 0) or 0
    meta["_rollout_seed_offset"] = rollout_seed_offset

    try:
        _ensure_tau2()
        _refresh_aws_credentials()
        async with _get_generate_semaphore():
            domain = meta.get("domain", "retail")
            if domain.startswith("vitabench_"):
                out = await _run_vita_episode(engine, meta)
            else:
                out = await _run_tau2_episode(engine, meta)

        if out is None:
            sample.status = Sample.Status.FAILED
            sample.reward = {"score": 0.0}
            sample.rollout_log_probs = []
            return sample

        sample.prompt = out.prompt_text
        sample.response = out.response
        sample.tokens = out.prompt_token_ids + out.token_ids
        sample.response_length = len(out.token_ids)
        sample.loss_mask = out.loss_mask if out.loss_mask else [1] * len(out.token_ids)
        sample.rollout_log_probs = out.log_probs
        sample.status = (
            Sample.Status.TRUNCATED if out.finish_reason == "length"
            else Sample.Status.COMPLETED
        )

        # Store turns and metadata for reward computation
        tau2_reward = 0.0
        uncertainty_events = []
        if out.turns:
            sample.train_metadata = {"turns": out.turns}
            for t in out.turns:
                if "tau2_reward_info" in t:
                    ri = t["tau2_reward_info"]
                    tau2_reward = float(ri.get("reward", 0.0)) if isinstance(ri, dict) else float(ri)
                if "uncertainty_events" in t:
                    uncertainty_events = t["uncertainty_events"]

        sample.reward = {"score": tau2_reward}
        meta["tau2_reward"] = tau2_reward
        meta["uncertainty_events"] = uncertainty_events
        meta["num_turns"] = len(out.turns) if out.turns else 0
        sample.metadata = meta

        # Runtime checks
        logger.info(
            "[sample] task=%s reward=%.3f tokens=%d loss_mask=%d log_probs=%d response_length=%d",
            meta.get("task_id"), tau2_reward,
            len(sample.tokens), len(sample.loss_mask) if sample.loss_mask else 0,
            len(sample.rollout_log_probs) if sample.rollout_log_probs else 0,
            sample.response_length,
        )
        # Shape assertions
        assert sample.response_length == len(sample.loss_mask) if sample.loss_mask else True, \
            f"response_length={sample.response_length} loss_mask={len(sample.loss_mask)}"
        assert sample.response_length == len(sample.rollout_log_probs) if sample.rollout_log_probs else True, \
            f"response_length={sample.response_length} log_probs={len(sample.rollout_log_probs)}"

    except TimeoutError:
        logger.warning("[bench2robust_rl] episode timed out")
        sample.response = ""
        sample.rollout_log_probs = []
        sample.status = Sample.Status.FAILED
        sample.reward = {"score": 0.0}
    except Exception as e:
        logger.error("[bench2robust_rl] episode failed: %s", e)
        traceback.print_exc()
        sample.response = ""
        sample.rollout_log_probs = []
        sample.status = Sample.Status.FAILED
        sample.reward = {"score": 0.0}

    # Ensure reward is always set
    if sample.reward is None:
        sample.reward = {"score": 0.0}

    return sample


async def _run_tau2_episode(
    engine: SGLangEngine,
    meta: dict,
) -> Optional[GenerationOutput]:
    """
    Run a complete tau2-bench episode in-process.

    Flow:
    1. Load task → build environment + user simulator
    2. Get initial user message
    3. Loop: agent generates → parse tool calls → execute tools (with uncertainty) → get user response
    4. Evaluate with tau2 evaluator
    5. Return GenerationOutput with all turns
    """
    from tau2.runner import get_tasks, build_text_orchestrator
    from tau2.data_model.simulation import TextRunConfig
    from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage, ToolCall

    domain = meta.get("domain", "retail")
    task_id = str(meta.get("task_id", "0"))
    task_split = meta.get("task_split", "all")
    inject = bool(meta.get("inject", False))
    if os.environ.get("BENCH2ROBUST_DISABLE_INJECTION", "0") == "1":
        inject = False
    seed = meta.get("seed", 42)
    # Per-rollout seed variation for different injection patterns
    rollout_offset = meta.get("_rollout_seed_offset", 0)
    seed = seed + rollout_offset * 1000  # Each rollout gets a distinct seed

    # Hybrid rollout: first half of rollouts run CLEAN, second half run NOISY
    # This ensures GRPO always has reward variance within each prompt group
    n_samples = int(os.environ.get("BENCH2ROBUST_N_SAMPLES", "32"))
    _phase = int(os.environ.get("BENCH2ROBUST_CURRICULUM_PHASE", "0"))

    if _phase == 0:
        # Phase 0 removed — treat as Phase 1
        _phase = 1

    if inject and rollout_offset < n_samples // 2:
        # Phase 1+: first half clean, second half inject
        inject = False
        meta["_hybrid_clean"] = True
    else:
        meta["_hybrid_clean"] = not inject

    # Phase 1: without alternatives (16 tools only)
    # Phase 2+: with alternatives (21 tools)
    with_alternatives = meta.get("with_alternatives", False)
    if _phase <= 1:
        with_alternatives = False  # Force no alternatives in Phase 0-1
    elif _phase >= 2:
        with_alternatives = True   # Force alternatives on in Phase 2+

    max_injections = meta.get("max_injections", 5)
    # Curriculum: control injection intensity via env var (can be updated during training)
    curriculum_max = int(os.environ.get("BENCH2ROBUST_MAX_INJECTIONS", str(max_injections)))
    max_injections = min(max_injections, curriculum_max)

    # Load task
    tasks = get_tasks(domain, task_split_name=task_split, task_ids=[task_id])
    if not tasks:
        logger.warning("[tau2] Task %s not found in %s/%s", task_id, domain, task_split)
        return None
    task = tasks[0]

    # Build orchestrator (handles environment + user simulator)
    user_model = os.environ.get(
        "BENCH2ROBUST_USER_MODEL",
        "bedrock/<REGION>/qwen.qwen3-235b-a22b-2507-v1:0"
    )
    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent="placeholder",  # We'll generate externally via SGLang
        llm_args_agent={},
        llm_user=user_model,
        llm_args_user={},
        max_steps=20,
        task_split_name=None,
    )
    orchestrator = build_text_orchestrator(config, task, seed=seed)

    # Setup uncertainty injection
    injection_policy = None
    noise_factory = None
    uncertainty_events = []

    # Recovery Controller (v3c) — DISABLED
    # We now rely on beliefs v5 system prompt + RL reward signal
    # instead of prescriptive per-tool-call recommendations.
    _rc_controller = None
    _rc_enabled = False

    if inject:
        try:
            from bench2robust.uncertainty.config import (
                NoiseType, TriggerTiming, EpisodeUncertaintyConfig,
            )
            from bench2robust.uncertainty.injection_policy import InjectionPolicy
            from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
            from bench2robust.adapters.tau_bench.tool_configs import RETAIL_TOOL_CONFIGS, AIRLINE_TOOL_CONFIGS

            # Select tool configs based on domain
            _domain_tool_configs = {
                "retail": RETAIL_TOOL_CONFIGS,
                "airline": AIRLINE_TOOL_CONFIGS,
            }
            _tool_configs = _domain_tool_configs.get(domain, RETAIL_TOOL_CONFIGS)

            import hashlib
            task_hash = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
            task_seed = seed + task_hash % 100000
            rng = random.Random(task_seed)
            max_inj = max_injections if max_injections else rng.randint(1, 8)

            uc_config = EpisodeUncertaintyConfig(
                seed=task_seed,
                max_consecutive_failures=2,
                max_total_injections=max_inj,
                trigger_timing=TriggerTiming.UNIFORM,
                tool_configs=_tool_configs,
            )
            injection_policy = InjectionPolicy(uc_config)
            injection_policy.reset()
            noise_factory = NoiseInjectorFactory()
        except Exception as e:
            logger.warning("[tau2] Failed to setup uncertainty injection: %s", e)

    # Initialize Recovery Controller (v3c) if enabled and phase allows
    _current_phase = int(os.environ.get("BENCH2ROBUST_CURRICULUM_PHASE", "1"))
    if _rc_enabled and inject and _current_phase >= _rc_phase_min:
        try:
            sys.path.insert(0, "<PATH>/eval/src")
            from recovery_controller import RecoveryController, RecoveryControllerConfig
            _rc_config = RecoveryControllerConfig(seed=seed, verbose_prompt=True)
            _rc_controller = RecoveryController(
                config=_rc_config,
                domain=domain,
                with_alternatives=with_alternatives,
                persist_beliefs=False,  # Per-episode (no cross-episode learning in RL)
            )
            _rc_controller.reset_episode()
            logger.info("[RC] Recovery controller v3c enabled for task=%s phase=%d",
                        task_id, _current_phase)
        except Exception as e:
            logger.warning("[RC] Failed to init recovery controller: %s", e)
            _rc_controller = None

    # Scenario-based tool blocking (for with_alternatives mode)
    _permanently_blocked: set = set()
    _protected_tools: set = set()

    # Default scenario for inject episodes without alternatives (Phase 1)
    if inject and not with_alternatives:
        meta["_scenario"] = "retry_works"

    if with_alternatives and inject and domain == "retail":
        _TOOL_PAIRS = [
            ({"find_user_id_by_name_zip"}, {"find_user_id_by_email", "get_user_by_order"}),
            ({"get_order_details"}, {"get_orders_for_user"}),
            ({"get_product_details", "get_item_details"}, {"search_product_by_name", "get_item_by_product_and_options", "check_order_item_availability"}),
        ]
        import hashlib
        _scenario_seed = seed + int(hashlib.md5(task_id.encode()).hexdigest()[:6], 16)
        _scenario_rng = random.Random(_scenario_seed + 7777)

        # Ablation: disable scenario control (random-noise baseline)
        # When BENCH2ROBUST_NO_SCENARIO_CONTROL=1, all episodes use "retry_works"
        # (no permanent blocking), making noise purely random without solvability guarantees.
        _no_scenario_control = os.environ.get("BENCH2ROBUST_NO_SCENARIO_CONTROL", "0") == "1"
        if _no_scenario_control:
            _scenario = "retry_works"
            meta["_scenario"] = "retry_works (random-noise ablation)"
        else:
            # Sample scenario based on curriculum phase (controllable via env var)
            # Phase 1: pure retry (consolidate basic error recovery)
            # Phase 2: introduce switch (20% S2)
            # Phase 3: more switch (40% S2)
            # Phase 4: add abstain (S2 dominant + some S3)
            # Phase 5: full difficulty
            _phase = int(os.environ.get("BENCH2ROBUST_CURRICULUM_PHASE", "1"))
            _roll = _scenario_rng.random()
            if _phase == 1:
                # 100% S1 — pure retry learning
                _scenario = "retry_works"
            elif _phase == 2:
                # 80% S1, 20% S2
                if _roll < 0.80:
                    _scenario = "retry_works"
                else:
                    _scenario = "switch_needed"
            elif _phase == 3:
                # 60% S1, 40% S2
                if _roll < 0.60:
                    _scenario = "retry_works"
                else:
                    _scenario = "switch_needed"
            elif _phase == 4:
                # 40% S1, 50% S2, 10% S3
                if _roll < 0.40:
                    _scenario = "retry_works"
                elif _roll < 0.90:
                    _scenario = "switch_needed"
                else:
                    _scenario = "impossible"
            else:
                # Phase 5+: 30% S1, 45% S2, 25% S3
                if _roll < 0.30:
                    _scenario = "retry_works"
                elif _roll < 0.75:
                    _scenario = "switch_needed"
                else:
                    _scenario = "impossible"

        for primary_set, alt_set in _TOOL_PAIRS:
            if _scenario_rng.random() < 0.5:
                protected_side, vulnerable_side = alt_set, primary_set
            else:
                protected_side, vulnerable_side = primary_set, alt_set

            if _scenario == "retry_works":
                _protected_tools.update(protected_side)
            elif _scenario == "switch_needed":
                _protected_tools.update(protected_side)
                _permanently_blocked.update(vulnerable_side)
            elif _scenario == "impossible":
                _permanently_blocked.update(protected_side)
                _permanently_blocked.update(vulnerable_side)

        logger.info("[tau2] task=%s scenario=%s protected=%s blocked=%s",
                    task_id, _scenario, sorted(_protected_tools), sorted(_permanently_blocked))
        meta["_scenario"] = _scenario  # Store for curriculum tracking in custom_convert

    if with_alternatives and inject and domain == "airline":
        _TOOL_PAIRS = [
            ({"get_user_details"}, {"find_user_by_email", "find_user_by_name_dob"}),
            ({"get_reservation_details"}, {"get_reservations_for_user"}),
            ({"search_direct_flight", "search_onestop_flight"}, {"search_flights"}),
        ]
        import hashlib
        _scenario_seed = seed + int(hashlib.md5(task_id.encode()).hexdigest()[:6], 16)
        _scenario_rng = random.Random(_scenario_seed + 7777)

        _no_scenario_control = os.environ.get("BENCH2ROBUST_NO_SCENARIO_CONTROL", "0") == "1"
        if _no_scenario_control:
            _scenario = "retry_works"
            meta["_scenario"] = "retry_works (random-noise ablation)"
        else:
            _phase = int(os.environ.get("BENCH2ROBUST_CURRICULUM_PHASE", "1"))
            _roll = _scenario_rng.random()
            if _phase == 1:
                _scenario = "retry_works"
            elif _phase == 2:
                if _roll < 0.80:
                    _scenario = "retry_works"
                else:
                    _scenario = "switch_needed"
            elif _phase == 3:
                if _roll < 0.60:
                    _scenario = "retry_works"
                else:
                    _scenario = "switch_needed"
            elif _phase == 4:
                if _roll < 0.40:
                    _scenario = "retry_works"
                elif _roll < 0.90:
                    _scenario = "switch_needed"
                else:
                    _scenario = "impossible"
            else:
                if _roll < 0.30:
                    _scenario = "retry_works"
                elif _roll < 0.75:
                    _scenario = "switch_needed"
                else:
                    _scenario = "impossible"

        for primary_set, alt_set in _TOOL_PAIRS:
            if _scenario_rng.random() < 0.5:
                protected_side, vulnerable_side = alt_set, primary_set
            else:
                protected_side, vulnerable_side = primary_set, alt_set

            if _scenario == "retry_works":
                _protected_tools.update(protected_side)
            elif _scenario == "switch_needed":
                _protected_tools.update(protected_side)
                _permanently_blocked.update(vulnerable_side)
            elif _scenario == "impossible":
                _permanently_blocked.update(protected_side)
                _permanently_blocked.update(vulnerable_side)

        logger.info("[tau2] task=%s scenario=%s protected=%s blocked=%s",
                    task_id, _scenario, sorted(_protected_tools), sorted(_permanently_blocked))
        meta["_scenario"] = _scenario

    # Get environment and policy
    env = orchestrator.environment
    policy_text = env.get_policy()

    # Build tools schema from environment
    tools_schema = []
    for tool in env.get_tools():
        tools_schema.append(tool.openai_schema)

    # Add alternative (discoverable) tools if with_alternatives is enabled
    if with_alternatives:
        from tau2.environment.tool import as_tool as _as_tool
        disc_methods = env.tools.get_discoverable_tools()
        for name, method in disc_methods.items():
            tool_obj = _as_tool(method)
            tools_schema.append(tool_obj.openai_schema)

    # Add tool definitions to system prompt so model knows available tools
    tools_desc = "\n<tools>\nYou have access to the following tools. To call a tool, output a JSON object with \"name\" and \"arguments\" keys.\n\n"
    for ts in tools_schema:
        fn = ts.get("function", ts)
        tools_desc += f"- {fn['name']}: {fn.get('description', '')}\n"
        if fn.get('parameters', {}).get('properties'):
            params = fn['parameters']['properties']
            param_names = list(params.keys())
            tools_desc += f"  Parameters: {', '.join(param_names)}\n"
    tools_desc += "\nTo send a message to the user, output: {\"message\": \"your text\"}\n\nIMPORTANT: Your final output for each turn must be exactly one JSON object and nothing else. Do not include markdown, explanations, or text outside the JSON object.\n</tools>\n"

    # Build system prompt with belief-based uncertainty guidance
    _belief_block = _get_belief_block(with_alternatives, domain=domain)

    system_content = (
        f"<instructions>\nYou are a customer service agent that helps the user "
        f"according to the <policy> provided below.\nIn each turn you can either:\n"
        f"- Send a message to the user (output JSON: {{\"message\": \"your text\"}})\n"
        f"- Make a tool call (output JSON: {{\"name\": \"tool_name\", \"arguments\": {{...}}}})\n"
        f"You cannot do both at the same time.\n\n"
        f"Try to be helpful and always follow the policy.\n</instructions>\n"
        f"<policy>\n{policy_text}\n</policy>\n"
        f"{tools_desc}\n"
        f"{_belief_block}"
    )

    # Get first user message via user simulator
    user_state = orchestrator.user.get_init_state()
    first_agent_msg = AssistantMessage(role="assistant", content="Hi! How can I help you today?")
    first_user_msg, user_state = await asyncio.to_thread(
        orchestrator.user.generate_next_message, first_agent_msg, user_state
    )

    # Initialize conversation
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": str(first_user_msg.content)},
    ]

    turns: List[dict] = []
    first_out: Optional[GenerationOutput] = None
    step_idx = 0
    max_steps = 25
    all_agent_messages = [first_agent_msg]  # For evaluation
    all_messages = [first_user_msg]

    import time as _time
    _episode_start = _time.time()

    while step_idx < max_steps:
        # Check episode time limit (only counts actual execution, not queue wait)
        if _time.time() - _episode_start > ROLLOUT_SAMPLE_TIMEOUT:
            logger.warning("[tau2] task=%s episode time limit reached at step %d", task_id, step_idx)
            break

        # Generate agent response via direct HTTP to SGLang
        try:
            logger.info("[tau2] task=%s step=%d starting generate", task_id, step_idx)
            gen_out = await _direct_generate(
                url=engine.url,
                tokenizer=engine.tokenizer,
                messages=messages,
                sampling_params=engine.sampling_params,
                enable_thinking=True,
                tools=tools_schema,
            )
            logger.info("[tau2] task=%s step=%d generate done, %d tokens", task_id, step_idx, len(gen_out.token_ids))
        except Exception as e:
            logger.warning("[tau2] Generate failed at step %d: %s", step_idx, e)
            break

        if first_out is None:
            first_out = gen_out

        # Parse response for tool calls (Hermes format)
        # NOTE: SGLang with enable_thinking returns thinking + response without <think> tags
        # The thinking is separated from response by double newline or end-of-thinking marker
        content, tool_calls = _parse_hermes_response(gen_out.response)
        logger.info("[tau2] task=%s step=%d resp_len=%d raw_end=%s parsed: content=%s tool_calls=%s",
                    task_id, step_idx, len(gen_out.response),
                    repr(gen_out.response[-200:]),
                    repr(content[:60]) if content else "None",
                    tool_calls)

        # Record turn
        turn_data = {
            "tokens": list(gen_out.prompt_token_ids) + list(gen_out.token_ids),
            "response_length": len(gen_out.token_ids),
            "loss_mask": [1] * len(gen_out.token_ids),
            "rollout_log_probs": list(gen_out.log_probs),
            "prompt_text": gen_out.prompt_text,
            "finish_reason": gen_out.finish_reason,
            "step_idx": step_idx,
            "has_tool_call": bool(tool_calls),
            "_response": gen_out.response,
            "_content": content,
            "_tool_calls": tool_calls,
        }
        turns.append(turn_data)

        if tool_calls:
            # Only allow one tool call per turn (tau2 retail standard)
            if len(tool_calls) > 1:
                logger.warning("[tau2] multiple tool calls parsed; keeping first only")
                tool_calls = tool_calls[:1]

            # Add assistant message ONCE for all tool calls
            assistant_content = content if content else ""
            if not assistant_content and tool_calls:
                assistant_content = json.dumps(
                    {"name": tool_calls[0]["name"], "arguments": tool_calls[0].get("arguments", {})},
                    ensure_ascii=False
                )
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call
            should_stop = False
            eval_tool_messages = []
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Check for transfer (terminal action)
                if tool_name == "transfer_to_human_agents":
                    if step_idx >= 3:
                        should_stop = True
                        break
                    else:
                        # Don't allow early transfer — force agent to keep helping
                        tool_response = json.dumps({"error": "Transfer is not available yet. Continue helping the user."})
                        messages.append({"role": "tool", "content": tool_response})
                        eval_tool_messages.append(ToolMessage(role="tool", id=f"call_{step_idx}_{tool_name}", content=tool_response))
                        continue

                # Execute tool
                try:
                    tool_fn = getattr(env.tools, tool_name, None)
                    if tool_fn and callable(tool_fn):
                        result = tool_fn(**tool_args)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    # Serialize result — handle Pydantic models and non-serializable objects
                    if isinstance(result, str):
                        tool_response = result
                    elif hasattr(result, 'model_dump'):
                        tool_response = json.dumps(result.model_dump(), default=str)
                    else:
                        try:
                            tool_response = json.dumps(result, default=str)
                        except (TypeError, ValueError):
                            tool_response = str(result)
                except Exception as e:
                    tool_response = json.dumps({"error": str(e)})

                # Apply uncertainty injection
                if injection_policy and noise_factory:
                    # Check if tool is permanently blocked (Scenario 2/3)
                    if tool_name in _permanently_blocked:
                        # Always return error, bypass max_consecutive_failures
                        from bench2robust.uncertainty.config import NoiseType as _NT
                        _err_types = [_NT.TIMEOUT, _NT.SERVER_ERROR, _NT.SCHEMA_DRIFT, _NT.RATE_LIMIT]
                        _noise = random.choice(_err_types)
                        _inj = noise_factory.create_injector(_noise, seed=seed + step_idx)
                        tool_response = _inj.inject(None, tool_name) or json.dumps({"error": f"Error: {_noise.value}"})
                        if not isinstance(tool_response, str):
                            tool_response = json.dumps(tool_response)
                        uncertainty_events.append({
                            "step_idx": step_idx, "tool_name": tool_name,
                            "noise_type": _noise.value, "blocked": True,
                        })
                    elif tool_name in _protected_tools:
                        # Protected: never inject, always clean
                        injection_policy.state.record_clean(tool_name)
                    else:
                        # Normal injection logic
                        should_inject, noise_type = injection_policy.should_inject(tool_name, step_idx)
                        if should_inject and noise_type:
                            injector = noise_factory.create_injector(noise_type, seed=seed + step_idx)
                            noisy = injector.inject(tool_response, tool_name)
                            tool_response = noisy if isinstance(noisy, str) else json.dumps(noisy)
                            injection_policy.state.record_injection(tool_name, noise_type)
                            uncertainty_events.append({
                                "step_idx": step_idx,
                                "tool_name": tool_name,
                                "noise_type": noise_type.value,
                            })
                        else:
                            injection_policy.state.record_clean(tool_name)

                # --- RC v3c: Recovery Controller guidance injection ---
                # If RC is enabled and an injection just happened, append guidance
                if _rc_controller is not None:
                    _last_event_count = len(uncertainty_events)
                    _was_injected = (
                        _last_event_count > 0
                        and uncertainty_events[-1].get("step_idx") == step_idx
                        and uncertainty_events[-1].get("tool_name") == tool_name
                    )
                    if _was_injected:
                        _noise_type = uncertainty_events[-1].get("noise_type")
                        _rc_guidance = _rc_controller.observe_tool_result(
                            tool_name=tool_name,
                            was_error=True,
                            error_type=_noise_type,
                            tool_content=tool_response,
                            is_injected=True,
                        )
                        if _rc_guidance:
                            tool_response = tool_response + _rc_guidance
                    else:
                        # Clean result — track for belief updates + post-switch verify
                        _rc_controller.observe_tool_result(
                            tool_name=tool_name,
                            was_error=False,
                            error_type=None,
                            tool_content=tool_response,
                            is_injected=False,
                        )
                        # Post-switch verify: append grounding check
                        if _rc_controller.session.pending_post_switch_verify:
                            _rc_controller.session.pending_post_switch_verify = False
                            _rc_controller.session.verify_after_switch_count += 1
                            _vas_cap = _rc_controller.tool_capabilities.get(tool_name)
                            _vas_lines = [
                                "\n\n--- VERIFY_AFTER_SWITCH ---",
                                f"You obtained this data from an alternative tool after the primary tool failed.",
                            ]
                            if _vas_cap and _vas_cap.is_summary:
                                _vas_lines.append(f"\n⚠ GRANULARITY WARNING: {_vas_cap.usage_note}")
                                if _vas_cap.does_not_provide:
                                    _vas_lines.append(f"  Fields NOT available: {', '.join(sorted(_vas_cap.does_not_provide))}")
                                _vas_lines.append("Do NOT infer item-level details from summary data.")
                            _vas_lines.extend([
                                "\nVerify before acting:",
                                "1. Are ALL required fields present?",
                                "2. Do IDs match the user's request?",
                                "3. Any conflict with earlier observations?",
                                "4. Does your next action require guessing?",
                                "\nDecide: VERIFIED_PROCEED / NEED_MORE_EVIDENCE / GRANULARITY_MISMATCH / SAFE_FALLBACK",
                                "--- End VERIFY_AFTER_SWITCH ---",
                            ])
                            tool_response = tool_response + "\n".join(_vas_lines)

                messages.append({"role": "tool", "content": tool_response})
                eval_tool_messages.append(ToolMessage(role="tool", id=f"call_{step_idx}_{tool_name}", content=tool_response))

            # Track for evaluation
            agent_msg = AssistantMessage(
                role="assistant",
                content=content or "",
                tool_calls=[ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in tool_calls],
            )
            all_agent_messages.append(agent_msg)
            all_messages.append(agent_msg)
            all_messages.extend(eval_tool_messages)

            if should_stop:
                break

        else:
            # Agent sent a text message to user — use parsed content (not raw thinking)
            clean_response = content if content else ""
            if not clean_response:
                # Fallback: try stripping thinking from raw response
                if "</think>" in gen_out.response:
                    think_end = gen_out.response.rfind("</think>")
                    clean_response = gen_out.response[think_end + len("</think>"):].strip()
                else:
                    clean_response = re.sub(r"<think>.*?</think>\s*", "", gen_out.response, flags=re.DOTALL).strip()
                clean_response = re.sub(r"<\|im_end\|>$", "", clean_response).strip()
            messages.append({"role": "assistant", "content": clean_response})
            agent_msg = AssistantMessage(role="assistant", content=clean_response)
            all_agent_messages.append(agent_msg)
            all_messages.append(agent_msg)

            # Check for transfer/termination
            if _is_terminal(content, step_idx):
                break

            # Get next user message
            try:
                # Use parsed content (clean, no thinking) for user simulator
                if not clean_response:
                    clean_response = "I'll help you with that."
                agent_msg_for_user = AssistantMessage(role="assistant", content=clean_response)
                # Run user simulator in a thread to avoid blocking the event loop
                logger.info("[tau2] task=%s step=%d calling user sim", task_id, step_idx)
                next_user_msg, user_state = await asyncio.to_thread(
                    orchestrator.user.generate_next_message,
                    agent_msg_for_user, user_state
                )
                logger.info("[tau2] task=%s step=%d user sim done", task_id, step_idx)
                user_content = str(next_user_msg.content)
                messages.append({"role": "user", "content": user_content})
                all_messages.append(next_user_msg)

                # Check if user simulator signals end of conversation
                if "###TRANSFER###" in user_content or "###END###" in user_content or "###STOP###" in user_content:
                    break
            except Exception as e:
                logger.warning("[tau2] User simulator failed at step %d: %s", step_idx, e)
                break

        step_idx += 1

    # Log episode completion
    logger.info("[tau2] Episode done: task=%s steps=%d turns=%d", task_id, step_idx, len(turns))

    # Evaluate the episode using tau2 evaluator
    reward_info = _evaluate_episode(orchestrator, task, all_messages, all_agent_messages)

    # Compute reward: simple binary (pass/fail from tau2 evaluator)
    episode_reward = reward_info.get("reward", 0.0)
    partial_credit = reward_info.get("partial_credit", 0.0)

    # Combined reward: full pass (1.0) + partial credit for incomplete episodes
    # This ensures GRPO has variance even when no episode fully passes.
    # Full pass: 1.0, Partial (e.g. 4/6 actions): 0.3 * 0.67 = 0.20
    #
    # Ablation F: Set BENCH2ROBUST_NO_PARTIAL_CREDIT=1 to use binary reward only
    # (failed episodes get 0, passed episodes get 1.0 * efficiency).
    PARTIAL_WEIGHT = 0.3
    _no_partial = os.environ.get("BENCH2ROBUST_NO_PARTIAL_CREDIT", "0") == "1"
    if episode_reward >= 1.0:
        combined_reward = 1.0
    elif _no_partial:
        combined_reward = 0.0
    else:
        combined_reward = PARTIAL_WEIGHT * partial_credit

    reward_info["episode_reward"] = episode_reward
    reward_info["partial_credit"] = partial_credit
    reward_info["total_steps"] = step_idx
    reward_info["reward"] = combined_reward

    logger.info("[tau2] task=%s reward=%.2f (pass=%d partial=%.2f steps=%d)",
                task_id, combined_reward, int(episode_reward >= 1.0), partial_credit, step_idx)

    # Save rollout for inspection
    _save_rollout_log(task_id, step_idx, len(turns), messages, reward_info, uncertainty_events)

    # Store reward and uncertainty events in the last turn
    if turns:
        turns[-1]["tau2_reward_info"] = reward_info
        turns[-1]["uncertainty_events"] = uncertainty_events

    # Build output
    return _build_generation_output(turns, first_out, reward_info, seed=seed)


async def _run_vita_episode(
    engine: SGLangEngine,
    meta: dict,
) -> Optional[GenerationOutput]:
    """
    Run a complete VitaBench episode in-process.

    Same flow as _run_tau2_episode but uses VitaBench's environment,
    user simulator, and evaluator.

    Flow:
    1. Load task via vita.run.get_tasks()
    2. Build VitaBench environment + user simulator
    3. Loop: agent generates → parse tool calls → execute tools (with uncertainty) → user response
    4. Evaluate with VitaBench evaluator
    5. Return GenerationOutput with all turns
    """
    from vita.run import get_tasks as vita_get_tasks, _clear_global_state
    from vita.registry import registry
    from vita.user.user_simulator import UserSimulator
    from vita.evaluator.evaluator import evaluate_simulation
    from vita.data_model.message import ToolMessage as VitaToolMessage, AssistantMessage as VitaAssistantMessage
    from tau2.data_model.message import AssistantMessage, ToolCall

    # Patch VitaBench's generate to use litellm (supports Bedrock model IDs)
    _patch_vita_generate()

    domain_full = meta.get("domain", "vitabench_delivery")  # e.g. "vitabench_delivery"
    vita_domain = domain_full.replace("vitabench_", "")  # e.g. "delivery"
    task_id = str(meta.get("task_id", "0"))
    task_split = meta.get("task_split", "delivery_train").replace("_train", "").replace("_test", "")
    inject = bool(meta.get("inject", False))
    if os.environ.get("BENCH2ROBUST_DISABLE_INJECTION", "0") == "1":
        inject = False
    seed = meta.get("seed", 42)
    rollout_offset = meta.get("_rollout_seed_offset", 0)
    seed = seed + rollout_offset * 1000

    # Hybrid rollout: first half clean, second half noisy
    n_samples = int(os.environ.get("BENCH2ROBUST_N_SAMPLES", "16"))
    _phase = int(os.environ.get("BENCH2ROBUST_CURRICULUM_PHASE", "1"))
    if _phase == 0:
        _phase = 1

    if inject and rollout_offset < n_samples // 2:
        inject = False
        meta["_hybrid_clean"] = True
    else:
        meta["_hybrid_clean"] = not inject

    with_alternatives = meta.get("with_alternatives", False)
    if _phase <= 1:
        with_alternatives = False
    elif _phase >= 2:
        with_alternatives = True

    max_injections = meta.get("max_injections", 5)
    curriculum_max = int(os.environ.get("BENCH2ROBUST_MAX_INJECTIONS", str(max_injections)))
    max_injections = min(max_injections, curriculum_max)

    # Load VitaBench task
    _clear_global_state()
    try:
        tasks = vita_get_tasks(vita_domain, task_ids=[task_id], language="english")
    except Exception as e:
        logger.warning("[vita] Task %s not found in %s: %s", task_id, vita_domain, e)
        return None
    if not tasks:
        logger.warning("[vita] Task %s not found in %s", task_id, vita_domain)
        return None
    task = tasks[0]

    # Build environment
    environment_constructor = registry.get_env_constructor(vita_domain)
    environment = environment_constructor(task.environment, "english")

    # Setup uncertainty injection
    injection_policy = None
    noise_factory = None
    uncertainty_events = []
    _permanently_blocked: set = set()
    _protected_tools: set = set()

    if inject:
        try:
            from bench2robust.uncertainty.config import (
                NoiseType, TriggerTiming, EpisodeUncertaintyConfig,
            )
            from bench2robust.uncertainty.injection_policy import InjectionPolicy
            from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
            from bench2robust.adapters.vitabench.tool_configs import (
                DELIVERY_TOOL_CONFIGS, INSTORE_TOOL_CONFIGS,
                DELIVERY_TOOL_PAIRS, INSTORE_TOOL_PAIRS,
            )

            tool_configs = DELIVERY_TOOL_CONFIGS if vita_domain == "delivery" else INSTORE_TOOL_CONFIGS
            tool_pairs = DELIVERY_TOOL_PAIRS if vita_domain == "delivery" else INSTORE_TOOL_PAIRS

            import hashlib
            task_hash = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
            task_seed = seed + task_hash % 100000
            rng = random.Random(task_seed)

            uc_config = EpisodeUncertaintyConfig(
                seed=task_seed,
                max_consecutive_failures=2,
                max_total_injections=max_injections,
                trigger_timing=TriggerTiming.UNIFORM,
                tool_configs=tool_configs,
            )
            injection_policy = InjectionPolicy(uc_config)
            injection_policy.reset()
            noise_factory = NoiseInjectorFactory()

            # Scenario-based blocking for with_alternatives
            if with_alternatives:
                _scenario_seed = seed + int(hashlib.md5(task_id.encode()).hexdigest()[:6], 16)
                _scenario_rng = random.Random(_scenario_seed + 7777)

                _roll = _scenario_rng.random()
                if _phase == 1:
                    _scenario = "retry_works"
                elif _phase == 2:
                    _scenario = "retry_works" if _roll < 0.80 else "switch_needed"
                elif _phase == 3:
                    _scenario = "retry_works" if _roll < 0.60 else "switch_needed"
                elif _phase == 4:
                    if _roll < 0.40:
                        _scenario = "retry_works"
                    elif _roll < 0.90:
                        _scenario = "switch_needed"
                    else:
                        _scenario = "impossible"
                else:
                    if _roll < 0.30:
                        _scenario = "retry_works"
                    elif _roll < 0.75:
                        _scenario = "switch_needed"
                    else:
                        _scenario = "impossible"

                for primary_set, alt_set in tool_pairs:
                    if _scenario_rng.random() < 0.5:
                        protected_side, vulnerable_side = alt_set, primary_set
                    else:
                        protected_side, vulnerable_side = primary_set, alt_set

                    if _scenario == "retry_works":
                        _protected_tools.update(protected_side)
                    elif _scenario == "switch_needed":
                        _protected_tools.update(protected_side)
                        _permanently_blocked.update(vulnerable_side)
                    elif _scenario == "impossible":
                        _permanently_blocked.update(protected_side)
                        _permanently_blocked.update(vulnerable_side)

                meta["_scenario"] = _scenario
            else:
                meta["_scenario"] = "retry_works"

        except Exception as e:
            logger.warning("[vita] Failed to setup uncertainty injection: %s", e)

    # Build tools schema from environment
    tools_schema = []
    for tool in environment.get_tools():
        tools_schema.append(tool.openai_schema)

    # Build system prompt
    policy_text = environment.get_policy()

    tools_desc = "\n<tools>\nYou have access to the following tools. To call a tool, output a JSON object with \"name\" and \"arguments\" keys.\n\n"
    for ts in tools_schema:
        fn = ts.get("function", ts)
        tools_desc += f"- {fn['name']}: {fn.get('description', '')}\n"
        if fn.get('parameters', {}).get('properties'):
            params = fn['parameters']['properties']
            param_names = list(params.keys())
            tools_desc += f"  Parameters: {', '.join(param_names)}\n"
    tools_desc += "\nTo send a message to the user, output: {\"message\": \"your text\"}\n\nIMPORTANT: Your final output for each turn must be exactly one JSON object and nothing else. Do not include markdown, explanations, or text outside the JSON object.\n</tools>\n"

    # Get belief block for VitaBench domain
    _vita_belief_block = _get_vita_belief_block(vita_domain, with_alternatives)

    system_content = (
        f"<instructions>\nYou are a customer service agent for a life-services platform. "
        f"Help the user according to the <policy> provided below.\n"
        f"In each turn you can either:\n"
        f"- Send a message to the user (output JSON: {{\"message\": \"your text\"}})\n"
        f"- Make a tool call (output JSON: {{\"name\": \"tool_name\", \"arguments\": {{...}}}})\n"
        f"You cannot do both at the same time.\n\n"
        f"Try to be helpful and always follow the policy.\n</instructions>\n"
        f"<policy>\n{policy_text}\n</policy>\n"
        f"{tools_desc}\n"
        f"{_vita_belief_block}"
    )

    # Build user simulator
    user_model = os.environ.get(
        "BENCH2ROBUST_USER_MODEL",
        "bedrock/<REGION>/qwen.qwen3-235b-a22b-2507-v1:0"
    )
    user = UserSimulator(
        persona=str(task.user_scenario.user_profile),
        instructions=str(task.instructions),
        llm=user_model,
        llm_args={"temperature": 0},
        language="english",
    )

    # Get initial user message (VitaBench uses stateful user simulator)
    user_state = user.get_init_state()
    first_agent_msg = VitaAssistantMessage(role="assistant", content="Hi! How can I help you today?")
    first_user_msg, user_state = await asyncio.to_thread(user.generate_next_message, first_agent_msg, user_state)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": str(first_user_msg.content)},
    ]

    turns: List[dict] = []
    first_out: Optional[GenerationOutput] = None
    step_idx = 0
    max_steps = int(os.environ.get("BENCH2ROBUST_MAX_STEPS", "25"))
    simulation_messages = []  # For VitaBench evaluation

    import time as _time
    _episode_start = _time.time()

    while step_idx < max_steps:
        if _time.time() - _episode_start > ROLLOUT_SAMPLE_TIMEOUT:
            logger.warning("[vita] task=%s episode time limit at step %d", task_id, step_idx)
            break

        # Generate agent response
        try:
            gen_out = await _direct_generate(
                url=engine.url,
                tokenizer=engine.tokenizer,
                messages=messages,
                sampling_params=engine.sampling_params,
                enable_thinking=True,
                tools=tools_schema,
            )
        except Exception as e:
            logger.warning("[vita] Generate failed at step %d: %s", step_idx, e)
            break

        if first_out is None:
            first_out = gen_out

        content, tool_calls = _parse_hermes_response(gen_out.response)

        turn_data = {
            "tokens": list(gen_out.prompt_token_ids) + list(gen_out.token_ids),
            "response_length": len(gen_out.token_ids),
            "loss_mask": [1] * len(gen_out.token_ids),
            "rollout_log_probs": list(gen_out.log_probs),
            "prompt_text": gen_out.prompt_text,
            "finish_reason": gen_out.finish_reason,
            "step_idx": step_idx,
            "has_tool_call": bool(tool_calls),
            "_response": gen_out.response,
            "_content": content,
            "_tool_calls": tool_calls,
        }
        turns.append(turn_data)

        if tool_calls:
            # Check if "tool call" is actually a text message (model outputs {"message": "..."})
            if len(tool_calls) == 1 and tool_calls[0].get("name") == "message":
                # Treat as text message to user
                msg_content = tool_calls[0].get("arguments", {})
                if isinstance(msg_content, dict):
                    clean_response = msg_content.get("message", msg_content.get("content", str(msg_content)))
                else:
                    clean_response = str(msg_content)
                messages.append({"role": "assistant", "content": clean_response})
                simulation_messages.append({"role": "assistant", "content": clean_response})

                if _is_terminal(clean_response, step_idx):
                    step_idx += 1
                    break

                # Get next user message
                try:
                    agent_msg_for_user = VitaAssistantMessage(role="assistant", content=clean_response or "I'll help you with that.")
                    next_user_msg, user_state = await asyncio.to_thread(user.generate_next_message, agent_msg_for_user, user_state)
                    user_content = str(next_user_msg.content)
                    messages.append({"role": "user", "content": user_content})
                    simulation_messages.append({"role": "user", "content": user_content})
                    if "###TRANSFER###" in user_content or "###END###" in user_content or "###STOP###" in user_content:
                        step_idx += 1
                        break
                except Exception as e:
                    logger.warning("[vita] User sim failed at step %d: %s", step_idx, e)
                    step_idx += 1
                    break

                step_idx += 1
                continue

            if len(tool_calls) > 1:
                tool_calls = tool_calls[:1]

            assistant_content = content if content else ""
            if not assistant_content and tool_calls:
                assistant_content = json.dumps(
                    {"name": tool_calls[0]["name"], "arguments": tool_calls[0].get("arguments", {})},
                    ensure_ascii=False
                )
            messages.append({"role": "assistant", "content": assistant_content})

            should_stop = False
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                if tool_name == "transfer_to_human_agents":
                    if step_idx >= 3:
                        should_stop = True
                        break
                    else:
                        tool_response = json.dumps({"error": "Transfer is not available yet. Continue helping the user."})
                        messages.append({"role": "tool", "content": tool_response})
                        continue

                # Execute tool via VitaBench environment
                try:
                    tool_fn = getattr(environment.tools, tool_name, None)
                    if tool_fn and callable(tool_fn):
                        result = tool_fn(**tool_args)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    if isinstance(result, str):
                        tool_response = result
                    elif hasattr(result, 'model_dump'):
                        tool_response = json.dumps(result.model_dump(), default=str)
                    else:
                        try:
                            tool_response = json.dumps(result, default=str)
                        except (TypeError, ValueError):
                            tool_response = str(result)
                except Exception as e:
                    tool_response = json.dumps({"error": str(e)})

                # Apply uncertainty injection
                if injection_policy and noise_factory:
                    if tool_name in _permanently_blocked:
                        from bench2robust.uncertainty.config import NoiseType as _NT
                        _err_types = [_NT.TIMEOUT, _NT.SERVER_ERROR, _NT.SCHEMA_DRIFT, _NT.RATE_LIMIT]
                        _noise = random.choice(_err_types)
                        _inj = noise_factory.create_injector(_noise, seed=seed + step_idx)
                        tool_response = _inj.inject(None, tool_name) or json.dumps({"error": f"Error: {_noise.value}"})
                        if not isinstance(tool_response, str):
                            tool_response = json.dumps(tool_response)
                        uncertainty_events.append({
                            "step_idx": step_idx, "tool_name": tool_name,
                            "noise_type": _noise.value, "blocked": True,
                        })
                    elif tool_name in _protected_tools:
                        injection_policy.state.record_clean(tool_name)
                    else:
                        should_inject_flag, noise_type = injection_policy.should_inject(tool_name, step_idx)
                        if should_inject_flag and noise_type:
                            injector = noise_factory.create_injector(noise_type, seed=seed + step_idx)
                            noisy = injector.inject(tool_response, tool_name)
                            tool_response = noisy if isinstance(noisy, str) else json.dumps(noisy)
                            injection_policy.state.record_injection(tool_name, noise_type)
                            uncertainty_events.append({
                                "step_idx": step_idx,
                                "tool_name": tool_name,
                                "noise_type": noise_type.value,
                            })
                        else:
                            injection_policy.state.record_clean(tool_name)

                messages.append({"role": "tool", "content": tool_response})
                simulation_messages.append({"role": "tool", "name": tool_name, "content": tool_response})

            simulation_messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})

            if should_stop:
                break

        else:
            # Agent sent text message to user
            clean_response = content if content else ""
            if not clean_response:
                if "</think>" in gen_out.response:
                    think_end = gen_out.response.rfind("</think>")
                    clean_response = gen_out.response[think_end + len("</think>"):].strip()
                else:
                    clean_response = re.sub(r"<think>.*?</think>\s*", "", gen_out.response, flags=re.DOTALL).strip()
                clean_response = re.sub(r"<\|im_end\|>$", "", clean_response).strip()
            messages.append({"role": "assistant", "content": clean_response})
            simulation_messages.append({"role": "assistant", "content": clean_response})

            if _is_terminal(content, step_idx):
                break

            # Get next user message
            try:
                agent_msg_for_user = VitaAssistantMessage(role="assistant", content=clean_response or "I'll help you with that.")
                next_user_msg, user_state = await asyncio.to_thread(user.generate_next_message, agent_msg_for_user, user_state)
                user_content = str(next_user_msg.content)
                messages.append({"role": "user", "content": user_content})
                simulation_messages.append({"role": "user", "content": user_content})

                if "###TRANSFER###" in user_content or "###END###" in user_content or "###STOP###" in user_content:
                    break
            except Exception as e:
                logger.warning("[vita] User sim failed at step %d: %s", step_idx, e)
                break

        step_idx += 1

    logger.info("[vita] Episode done: task=%s steps=%d turns=%d", task_id, step_idx, len(turns))

    # ========================================================================
    # Deterministic reward (no LLM judge) — fast + dense signal
    # Rewards based on observable actions in the episode:
    #   - Successful tool calls (non-error responses)
    #   - Write actions (create_order, pay, book, reservation) = task progress
    #   - User satisfaction (###STOP### = user voluntarily ended)
    #   - Penalty for repetition and early failure
    # ========================================================================
    _use_deterministic_reward = os.environ.get("BENCH2ROBUST_DETERMINISTIC_REWARD", "1") == "1"

    if _use_deterministic_reward:
        # Count tool calls and their outcomes from messages
        _write_tools = {
            "create_delivery_order", "pay_delivery_order", "cancel_delivery_order", "modify_delivery_order",
            "create_instore_product_order", "pay_instore_order", "instore_cancel_order",
            "instore_book", "pay_instore_book", "instore_cancel_book",
            "instore_reservation", "instore_modify_reservation", "instore_cancel_reservation",
        }
        _search_tools = {
            "instore_shop_search_recommend", "instore_product_search_recommend",
            "delivery_store_search_recommend", "delivery_product_search_recommend",
            "get_nearby", "get_delivery_store_info", "get_delivery_product_info",
        }

        successful_writes = 0
        successful_searches = 0
        total_tool_calls = 0
        error_tool_calls = 0
        user_stopped = False
        got_user_id = False

        for i, m in enumerate(messages):
            if m.get("role") == "tool":
                total_tool_calls += 1
                content = m.get("content", "")
                is_error = "error" in content.lower()[:50] or "Error" in content[:50]
                if is_error:
                    error_tool_calls += 1
                else:
                    # Check what kind of tool this was (look at previous assistant msg)
                    prev_assistant = None
                    for j in range(i-1, -1, -1):
                        if messages[j].get("role") == "assistant":
                            prev_assistant = messages[j].get("content", "")
                            break
                    if prev_assistant:
                        try:
                            tc_data = json.loads(prev_assistant) if prev_assistant.startswith("{") else {}
                            tc_name = tc_data.get("name", "")
                        except (json.JSONDecodeError, TypeError):
                            tc_name = ""
                        if tc_name in _write_tools:
                            successful_writes += 1
                        elif tc_name in _search_tools:
                            successful_searches += 1
            elif m.get("role") == "user":
                content = m.get("content", "")
                if "###STOP###" in content or "###END###" in content:
                    user_stopped = True

        # Compute deterministic reward components
        # 1. Search progress: found relevant info (0-0.3)
        search_score = min(successful_searches * 0.1, 0.3)

        # 2. Write progress: completed actions (0-0.5)
        write_score = min(successful_writes * 0.25, 0.5)

        # 3. Completion bonus: user voluntarily stopped (0.2)
        completion_score = 0.2 if user_stopped else 0.0

        # 4. Penalty: too many errors or no progress
        error_penalty = 0.0
        if total_tool_calls > 0 and error_tool_calls / total_tool_calls > 0.5:
            error_penalty = -0.2
        if total_tool_calls == 0 and step_idx > 5:
            error_penalty = -0.3  # Model is just chatting without using tools

        # Raw deterministic reward
        det_reward = max(0.0, search_score + write_score + completion_score + error_penalty)

        # Apply efficiency and repetition penalties
        eff = max(0.3, 1.0 - 0.02 * max(0, step_idx - 12))
        det_reward *= eff

        # Repetition penalty
        rep_penalty = 1.0
        _consecutive = 0
        _last_tc = None
        for m in messages:
            if m.get("role") == "assistant":
                if m.get("content") == _last_tc and _last_tc:
                    _consecutive += 1
                else:
                    _consecutive = 0
                _last_tc = m.get("content")
        if _consecutive >= 4:
            rep_penalty = 0.0
        elif _consecutive >= 3:
            rep_penalty = 0.5
        det_reward *= rep_penalty

        combined_reward = det_reward
        episode_reward = 1.0 if (successful_writes >= 2 and user_stopped) else 0.0
        partial_credit = det_reward

        logger.info("[vita] task=%s det_reward=%.3f (search=%.2f write=%.2f complete=%.2f err=%.2f eff=%.2f rep=%.2f) writes=%d searches=%d steps=%d",
                    task_id, combined_reward, search_score, write_score, completion_score, error_penalty, eff, rep_penalty,
                    successful_writes, successful_searches, step_idx)

    else:
        # Original LLM judge evaluation (slow but accurate)
        try:
            from vita.data_model.simulation import SimulationRun, TerminationReason, RewardInfo
            from vita.data_model.message import UserMessage as VitaUserMessage
            import uuid
            from datetime import datetime

            vita_messages = []
            for sm in simulation_messages:
                role = sm.get("role", "")
                content = sm.get("content", "")
                if role == "user":
                    vita_messages.append(VitaUserMessage(role="user", content=content))
                elif role == "assistant":
                    vita_messages.append(VitaAssistantMessage(role="assistant", content=content))
                elif role == "tool":
                    vita_messages.append(VitaToolMessage(
                        id=sm.get("id", f"call_{len(vita_messages)}"),
                        name=sm.get("name", "unknown"),
                        content=content,
                        requestor="assistant",
                        role="tool",
                    ))

            if step_idx >= max_steps:
                term_reason = TerminationReason.MAX_STEPS
            else:
                term_reason = TerminationReason.AGENT_STOP

            now_str = datetime.now().isoformat()
            sim_run = SimulationRun(
                id=str(uuid.uuid4()),
                task_id=task_id,
                start_time=now_str,
                end_time=now_str,
                duration=_time.time() - _episode_start,
                termination_reason=term_reason,
                messages=vita_messages,
                states={"old_states": [], "new_states": []},
                seed=seed,
            )

            reward_info_vita = await asyncio.to_thread(
                evaluate_simulation,
                simulation=sim_run,
                task=task,
                evaluation_type="trajectory",
                domain=vita_domain,
                llm_evaluator=user_model,
                llm_args_evaluator={"temperature": 0, "max_tokens": 4096},
                language="english",
            )
            episode_reward = float(reward_info_vita.reward) if reward_info_vita else 0.0
            partial_credit = 0.0
            if reward_info_vita and hasattr(reward_info_vita, 'reward_breakdown') and reward_info_vita.reward_breakdown:
                from vita.data_model.tasks import RewardType as VitaRewardType
                rubric_score = reward_info_vita.reward_breakdown.get(VitaRewardType.NL_ASSERTION, 0.0)
                if isinstance(rubric_score, (int, float)):
                    partial_credit = float(rubric_score)
            if partial_credit == 0.0 and reward_info_vita and hasattr(reward_info_vita, 'nl_rubrics') and reward_info_vita.nl_rubrics:
                met_count = sum(1 for r in reward_info_vita.nl_rubrics if r.met)
                total_count = len(reward_info_vita.nl_rubrics)
                if total_count > 0:
                    partial_credit = met_count / total_count
            logger.info("[vita] Evaluation result: episode_reward=%.2f partial_credit=%.2f", episode_reward, partial_credit)
        except Exception as e:
            logger.warning("[vita] Evaluation failed for task %s: %s", task_id, e)
            episode_reward = 0.0
            partial_credit = 0.0

        # Compute combined reward — higher partial weight for VitaBench (sparse signal)
        PARTIAL_WEIGHT = 0.5
        _no_partial = os.environ.get("BENCH2ROBUST_NO_PARTIAL_CREDIT", "0") == "1"
        if episode_reward >= 1.0:
            combined_reward = 1.0
        elif _no_partial:
            combined_reward = 0.0
        else:
            combined_reward = PARTIAL_WEIGHT * partial_credit

        # Efficiency discount
        eff = max(0.3, 1.0 - 0.02 * max(0, step_idx - 12))
        if episode_reward >= 1.0:
            combined_reward *= eff

        # Repetition penalty
        rep_penalty = 1.0
        _consecutive = 0
        _last_tc = None
        for m in messages:
            if m.get("role") == "assistant":
                if m.get("content") == _last_tc and _last_tc:
                    _consecutive += 1
                else:
                    _consecutive = 0
                _last_tc = m.get("content")
        if _consecutive >= 4:
            rep_penalty = 0.0
        elif _consecutive >= 3:
            rep_penalty = 0.5
        combined_reward *= rep_penalty

    reward_info = {
        "reward": combined_reward,
        "episode_reward": episode_reward,
        "partial_credit": partial_credit,
        "total_steps": step_idx,
        "efficiency": eff,
        "rep_penalty": rep_penalty,
    }

    logger.info("[vita] task=%s reward=%.2f (pass=%d partial=%.2f steps=%d)",
                task_id, combined_reward, int(episode_reward >= 1.0), partial_credit, step_idx)

    _save_rollout_log(task_id, step_idx, len(turns), messages, reward_info, uncertainty_events)

    if turns:
        turns[-1]["tau2_reward_info"] = reward_info
        turns[-1]["uncertainty_events"] = uncertainty_events

    return _build_generation_output(turns, first_out, reward_info, seed=seed)


# VitaBench belief provider cache
_vita_belief_provider_cache = {}


def _get_vita_belief_block(vita_domain: str, with_alternatives: bool) -> str:
    """Get belief block for VitaBench domain."""
    if os.environ.get("BENCH2ROBUST_NO_BTM_TRAINING", "0") == "1":
        return ""

    global _vita_belief_provider_cache
    cache_key = f"{vita_domain}_{with_alternatives}"
    if cache_key in _vita_belief_provider_cache:
        return _vita_belief_provider_cache[cache_key]

    beliefs_path = os.environ.get(
        "BENCH2ROBUST_VITA_BELIEFS_PATH",
        "<PATH>/eval/src/precomputed_beliefs_vita.json"
    )
    try:
        sys.path.insert(0, "<PATH>/eval/src")
        from belief_provider import BeliefProvider
        bp = BeliefProvider.from_precomputed(beliefs_path, with_alternatives=with_alternatives)
        block = bp.get_system_prompt_block()
        _vita_belief_provider_cache[cache_key] = block
        return block
    except Exception as e:
        logger.warning("[vita beliefs] Failed to load beliefs for %s: %s", vita_domain, e)
        _vita_belief_provider_cache[cache_key] = ""
        return ""


def _parse_hermes_response(text: str) -> Tuple[Optional[str], Optional[List[dict]]]:
    """Parse tool calls from model output. Supports multiple formats:

    1. Hermes format: <tool_call>{"name":..., "arguments":...}</tool_call>
    2. SGLang thinking mode: thinking text followed by JSON at the end
       - Tool call: {"name": "tool_name", "arguments": {...}}
       - Message: {"message": "text"} or {"content": "text"}
    3. Raw text response (no JSON)
    """
    # First try: strip thinking content (SGLang thinking mode outputs: thinking...</think>\n\nresponse)
    # Note: SGLang may not include opening <think> tag, only closing </think>
    if "</think>" in text:
        think_end = text.rfind("</think>")
        cleaned = text[think_end + len("</think>"):].strip()
    else:
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    # Strip <|im_end|> if present
    cleaned = re.sub(r"<\|im_end\|>$", "", cleaned).strip()

    # Format 1: Hermes-style <tool_call> tags
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = tool_call_pattern.findall(cleaned)
    if matches:
        tool_calls = []
        for m in matches:
            try:
                tc = json.loads(m)
                if isinstance(tc, dict) and "name" in tc:
                    tool_calls.append({"name": tc["name"], "arguments": tc.get("arguments", {})})
            except json.JSONDecodeError:
                continue
        if tool_calls:
            content = tool_call_pattern.sub("", cleaned).strip() or None
            return content, tool_calls

    # Format 2: Find the LAST JSON object in the text (SGLang thinking mode)
    # The model outputs: <thinking text> + <JSON response>
    # Find the last { that starts a valid JSON object
    last_json = None
    last_json_start = -1
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] == '}':
            # Find matching opening brace
            depth = 0
            for j in range(i, -1, -1):
                if cleaned[j] == '}':
                    depth += 1
                elif cleaned[j] == '{':
                    depth -= 1
                if depth == 0:
                    candidate = cleaned[j:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            last_json = obj
                            last_json_start = j
                    except json.JSONDecodeError:
                        pass
                    break
            if last_json is not None:
                break

    if last_json:
        # It's a tool call
        if "name" in last_json:
            return None, [{"name": last_json["name"], "arguments": last_json.get("arguments", {})}]
        # It's a message response
        if "message" in last_json:
            return last_json["message"], None
        if "content" in last_json:
            return last_json["content"], None

    # Format 3: No JSON found — entire text is the response (rare)
    # Try to extract just the last paragraph as the actual response
    return cleaned if cleaned else None, None




def _check_hallucination(messages, policy, task_instruction="", uncertainty_events=None):
    """Hallucination judge via Qwen3-235B (see hallucination_judge.py)."""
    from bench2robust_rl.hallucination_judge import check_hallucination
    return check_hallucination(messages, policy, task_instruction, uncertainty_events)


_ROLLOUT_LOG_DIR = "<PATH>/rollout_logs"
_rollout_counter = [0]

# Hallucination judge prompt (same as eval/src/run_inference.py)

_ROLLOUT_LOG_DIR = "<PATH>/rollout_logs"
_rollout_counter = [0]

# Hallucination judge prompt (same as eval/src/run_inference.py)

_ROLLOUT_LOG_DIR = "<PATH>/rollout_logs"
_rollout_counter = [0]

# Hallucination judge prompt (same as eval/src/run_inference.py)
def _save_rollout_log(task_id, steps, turns, messages, reward_info, uncertainty_events):
    """Save rollout conversation to a JSONL file for inspection."""
    import time
    import os

    log_dir = "<PATH>/rollout_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rollouts.jsonl")

    # Build concise conversation log from messages
    conversation = []
    for msg in (messages or []):
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "?")
            content = str(getattr(msg, "content", ""))
        # Truncate very long messages for readability
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"
        entry = {"role": role, "content": content}
        # Include tool_calls if present
        if isinstance(msg, dict) and "tool_calls" in msg:
            entry["tool_calls"] = msg["tool_calls"]
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        conversation.append(entry)

    record = {
        "task_id": task_id,
        "timestamp": time.time(),
        "num_turns": turns,
        "steps": steps,
        "reward": reward_info.get("reward", 0.0),
        "episode_reward": reward_info.get("episode_reward", 0.0),
        "avg_turn_bonus": reward_info.get("avg_turn_bonus", 0.0),
        "hallucination_penalty": reward_info.get("hallucination_penalty", 0.0),
        "efficiency_bonus": reward_info.get("efficiency_bonus", 0.0),
        "uncertainty_events": uncertainty_events,
        "conversation": conversation,
    }

    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.warning("[rollout_log] Failed to write: %s", str(e)[:100])

def _is_terminal(content: Optional[str], step_idx: int = 0) -> bool:
    """Check if agent response signals end of conversation.
    Only triggers after at least 3 steps (to avoid premature termination).
    """
    if not content:
        return False
    # Don't terminate too early — need at least a few turns for meaningful interaction
    if step_idx < 3:
        return False
    lower = content.lower()
    return any(p in lower for p in [
        "transfer_to_human",
        "transfer to a specialist",
        "transfer you to",
        "you are being transferred",
    ])


def _evaluate_episode(orchestrator, task, all_messages, all_agent_messages) -> dict:
    """Evaluate episode reward using tau2's evaluator.
    
    Returns partial action credit: matched_actions / total_required_actions.
    This gives dense signal even for partially-correct episodes.
    """
    try:
        from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
        from tau2.orchestrator.modes import CommunicationMode
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        import uuid
        from datetime import datetime

        tool_call_count = sum(1 for m in all_messages if hasattr(m, 'tool_calls') and m.tool_calls)
        logger.info("[tau2] Building SimulationRun: %d messages, %d with tool_calls", len(all_messages), tool_call_count)

        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=str(task.id),
            timestamp=datetime.now().isoformat(),
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=all_messages,
            policy=orchestrator.environment.get_policy(),
            mode="half_duplex",
        )

        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ACTION,
            solo_mode=False,
            domain="retail",
            mode=CommunicationMode.HALF_DUPLEX,
        )

        reward = reward_info.reward if reward_info else 0.0

        # Extract partial action credit from action_checks
        partial_credit = 0.0
        actions_matched = 0
        actions_total = 0
        if reward_info and hasattr(reward_info, 'action_checks') and reward_info.action_checks:
            checks = reward_info.action_checks
            actions_total = len(checks)
            actions_matched = sum(1 for c in checks if c.action_match)
            if actions_total > 0:
                partial_credit = actions_matched / actions_total

        info = reward_info.info if (reward_info and hasattr(reward_info, 'info')) else {}
        logger.info("[tau2] Evaluation: reward=%.2f partial=%.2f (%d/%d actions) info=%s",
                    reward, partial_credit, actions_matched, actions_total, str(info)[:200])

        return {
            "reward": float(reward),
            "partial_credit": partial_credit,
            "actions_matched": actions_matched,
            "actions_total": actions_total,
        }

    except Exception as e:
        logger.warning("[tau2] Evaluation failed: %s", str(e)[:200])
        return {"reward": 0.0, "eval_error": str(e)[:500]}


def _build_generation_output(
    turns: List[dict],
    first_out: Optional[GenerationOutput],
    reward_info: dict,
    seed: int = 42,
) -> Optional[GenerationOutput]:
    """Assemble generation output for training.
    
    Selects a single turn to train on (preferring tool-call turns 70% of the time).
    This avoids semantic issues with multi-turn token concatenation.
    """
    if not turns or first_out is None:
        return None

    # Prefer tool-call turns (70%) — they carry the most action-learning signal
    tool_turns = [t for t in turns if t.get("has_tool_call")]
    message_turns = [t for t in turns if not t.get("has_tool_call")]

    rng = random.Random(seed)
    if tool_turns and rng.random() < 0.7:
        selected = rng.choice(tool_turns)
    else:
        selected = rng.choice(message_turns or turns)

    prompt_len = len(selected["tokens"]) - selected["response_length"]

    prompt_token_ids = selected["tokens"][:prompt_len]
    token_ids = selected["tokens"][prompt_len:]
    loss_mask = selected["loss_mask"]
    log_probs = selected["rollout_log_probs"]

    # Truncate overly long responses to prevent OOM during training
    MAX_RESPONSE_TOKENS = 4096
    if len(token_ids) > MAX_RESPONSE_TOKENS:
        token_ids = token_ids[:MAX_RESPONSE_TOKENS]
        loss_mask = loss_mask[:MAX_RESPONSE_TOKENS]
        log_probs = log_probs[:MAX_RESPONSE_TOKENS]

    reward = reward_info.get("reward", 0.0) if isinstance(reward_info, dict) else 0.0

    return GenerationOutput(
        prompt_text=selected.get("prompt_text", first_out.prompt_text),
        prompt_token_ids=prompt_token_ids,
        response=selected.get("_response", ""),
        token_ids=token_ids,
        log_probs=log_probs,
        finish_reason=selected.get("finish_reason", "stop"),
        loss_mask=loss_mask,
        final_output=str(reward),
        turns=turns,
    )
