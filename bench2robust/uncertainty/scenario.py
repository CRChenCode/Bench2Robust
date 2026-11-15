# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Scenario control (S1/S2/S3), kept separate from stochastic injection.

Added for the ICLR revision. Algorithm 1 of the paper has two
independent mechanisms:

    tool call
      -> scenario controller
           if the tool is on a blocked path: return an episode-persistent blocked
           response WITHOUT executing the tool
           elif the tool is an unblocked fallback (fallback_noise="exempt"): execute it
           clean, without consulting the stochastic policy
           else: continue
      -> stochastic injection policy (budget B, K_max, timing, per-error limits)
      -> execute (clean / silent corruption) or not (explicit failure)

Semantics: scenario-blocked tools are never executed and stay blocked until episode
reset, regardless of B and K_max; unblocked fallback tools are exempt from stochastic
noise (configurable via ``fallback_noise``); all other tools follow the stochastic policy.

``fallback_noise`` (FALLBACK_NOISE_MODES):
  * "exempt" (default): tools in ScenarioSpec.fallback_paths (the unblocked side of an
    S2 equivalence class) return a CLEAN decision. They never receive a stochastic
    execution failure or a silent corruption and never consume B or K_max. In S1 and S3
    nothing is exempt (S1 has no blocked class, S3 has no unblocked side).
  * "exempt_s1": as "exempt", and in S1 one randomly chosen side of each equivalence
    class is also clean (drawn with random.Random(seed + 7777), one draw per class).
  * "noisy": fallback tools follow the normal stochastic policy.

This module provides the explicit decision type and the scenario controller. It does
not import any benchmark code, so every adapter (tau2, VitaBench, BFCL) and the
experimental eval/training paths can share one implementation.

Invariants enforced here (tests: tests/test_scenario_control.py):
  * a scenario-blocked call never consults the stochastic policy (no can_inject /
    should_inject call), never executes, never updates B / K_max / per-error counters,
    and never receives a silent corruption (partial / stale / factual_error);
  * the blocked set is fixed for the whole episode and only changes on reset();
  * under fallback_noise="exempt", a fallback call never consults the stochastic policy
    and never updates B / K_max / per-error counters;
  * blocked-response texts are drawn from a private RNG so they do not perturb the
    stochastic injectors' RNG streams.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from bench2robust.uncertainty.config import NoiseType


# ---------------------------------------------------------------------------
# Decision type
# ---------------------------------------------------------------------------

class DecisionKind(str, Enum):
    SCENARIO_BLOCK = "scenario_block"
    STOCHASTIC_EXEC_FAILURE = "stochastic_exec_failure"
    STOCHASTIC_SILENT_CORRUPTION = "stochastic_silent_corruption"
    CLEAN = "clean"


# Explicit-signal failures: the call is NOT executed (paper Sec. 3.1, Alg. 1 line 105).
EXEC_FAILURE_NOISE: FrozenSet[NoiseType] = frozenset({
    NoiseType.TIMEOUT, NoiseType.RATE_LIMIT, NoiseType.SERVER_ERROR,
    NoiseType.AUTH_ERROR, NoiseType.MALFORMED_RESPONSE, NoiseType.SCHEMA_DRIFT,
})
# Silent corruptions: the call executes and the observation is corrupted (Alg. 1 line 107).
SILENT_NOISE: FrozenSet[NoiseType] = frozenset({
    NoiseType.PARTIAL, NoiseType.STALE, NoiseType.FACTUAL_ERROR,
})
# Texts used for scenario-blocked responses. Same four explicit types as the experimental
# eval (run_inference.py) and training (rollout.py) code, so the blocked response looks
# like an explicit failure and never carries data.
BLOCKED_RESPONSE_NOISE: Tuple[NoiseType, ...] = (
    NoiseType.TIMEOUT, NoiseType.SERVER_ERROR, NoiseType.SCHEMA_DRIFT, NoiseType.RATE_LIMIT,
)


@dataclass(frozen=True)
class Decision:
    """Outcome of the interception step for one tool call."""

    kind: DecisionKind
    execute: bool
    noise_type: Optional[NoiseType] = None
    reason: str = ""
    fallback_exempt: bool = False   # CLEAN because the tool is a noise-exempt fallback

    @classmethod
    def clean(cls, reason: str = "") -> "Decision":
        return cls(DecisionKind.CLEAN, execute=True, noise_type=None, reason=reason)

    @classmethod
    def exempt_fallback(cls, reason: str = "fallback exempt from stochastic noise") -> "Decision":
        """CLEAN decision for an unblocked fallback tool (fallback_noise exempt / exempt_s1)."""
        return cls(DecisionKind.CLEAN, execute=True, noise_type=None, reason=reason,
                   fallback_exempt=True)

    @classmethod
    def scenario_block(cls, reason: str = "scenario-blocked path") -> "Decision":
        return cls(DecisionKind.SCENARIO_BLOCK, execute=False, noise_type=None, reason=reason)

    @property
    def is_stochastic(self) -> bool:
        return self.kind in (DecisionKind.STOCHASTIC_EXEC_FAILURE,
                             DecisionKind.STOCHASTIC_SILENT_CORRUPTION)


def decision_from_noise(noise_type: Optional[NoiseType], reason: str = "") -> Decision:
    """Map a sampled stochastic noise type to a Decision."""
    if noise_type is None or noise_type == NoiseType.CLEAN:
        return Decision.clean(reason)
    if noise_type in EXEC_FAILURE_NOISE:
        return Decision(DecisionKind.STOCHASTIC_EXEC_FAILURE, execute=False,
                        noise_type=noise_type, reason=reason)
    if noise_type in SILENT_NOISE:
        return Decision(DecisionKind.STOCHASTIC_SILENT_CORRUPTION, execute=True,
                        noise_type=noise_type, reason=reason)
    raise ValueError(f"noise type {noise_type!r} cannot be injected")


# ---------------------------------------------------------------------------
# Scenarios and equivalence classes
# ---------------------------------------------------------------------------

class Scenario(str, Enum):
    S1 = "S1"   # original path viable (no blocking)
    S2 = "S2"   # switch needed (one side of an equivalence class blocked)
    S3 = "S3"   # impossible (all viable task paths blocked)


_LEGACY_TO_SCENARIO = {
    "retry_works": Scenario.S1, "switch_needed": Scenario.S2, "impossible": Scenario.S3,
    "s1": Scenario.S1, "s2": Scenario.S2, "s3": Scenario.S3,
}
_SCENARIO_TO_LEGACY = {Scenario.S1: "retry_works", Scenario.S2: "switch_needed",
                       Scenario.S3: "impossible"}


def normalize_scenario(value) -> Scenario:
    """Accept 'S1'/'S2'/'S3' or the legacy names used by the experimental code."""
    if isinstance(value, Scenario):
        return value
    key = str(value).strip()
    if key.upper() in ("S1", "S2", "S3"):
        return Scenario(key.upper())
    if key.lower() in _LEGACY_TO_SCENARIO:
        return _LEGACY_TO_SCENARIO[key.lower()]
    raise ValueError(f"unknown scenario {value!r}")


def legacy_name(scenario) -> str:
    return _SCENARIO_TO_LEGACY[normalize_scenario(scenario)]


@dataclass(frozen=True)
class EquivalenceClass:
    """Two fallback-equivalent sides of one task step (primary tools vs alternatives)."""

    name: str
    primary: FrozenSet[str]
    alternatives: FrozenSet[str]

    @property
    def tools(self) -> FrozenSet[str]:
        return self.primary | self.alternatives


def _ec(name, primary, alternatives) -> EquivalenceClass:
    return EquivalenceClass(name, frozenset(primary), frozenset(alternatives))


# Same three pairs as eval/src/run_inference.py (_RETAIL_TOOL_PAIRS) and
# slime/agentic/bench2robust_rl/rollout.py (_TOOL_PAIRS).
RETAIL_EQUIVALENCE_CLASSES: Tuple[EquivalenceClass, ...] = (
    _ec("user_lookup", {"find_user_id_by_name_zip"}, {"find_user_id_by_email", "get_user_by_order"}),
    _ec("order_lookup", {"get_order_details"}, {"get_orders_for_user"}),
    _ec("product_lookup", {"get_product_details", "get_item_details"},
        {"search_product_by_name", "get_item_by_product_and_options", "check_order_item_availability"}),
)
AIRLINE_EQUIVALENCE_CLASSES: Tuple[EquivalenceClass, ...] = (
    _ec("user_lookup", {"get_user_details"}, {"find_user_by_email", "find_user_by_name_dob"}),
    _ec("reservation_lookup", {"get_reservation_details"}, {"get_reservations_for_user"}),
    _ec("flight_search", {"search_direct_flight", "search_onestop_flight"}, {"search_flights"}),
)
DOMAIN_EQUIVALENCE_CLASSES: Dict[str, Tuple[EquivalenceClass, ...]] = {
    "retail": RETAIL_EQUIVALENCE_CLASSES,
    "airline": AIRLINE_EQUIVALENCE_CLASSES,
}


def derive_fallback_paths(blocked: Iterable[str],
                          classes: Optional[Sequence[EquivalenceClass]]) -> FrozenSet[str]:
    """Unblocked side of every partly blocked class (empty if no classes are given)."""
    b = frozenset(blocked or ())
    out: Set[str] = set()
    for c in classes or ():
        if (c.tools & b) and not c.tools <= b:
            out |= c.tools - b
    return frozenset(out)


# ---------------------------------------------------------------------------
# Fallback noise
# ---------------------------------------------------------------------------

# exempt    = unblocked fallback tools (ScenarioSpec.fallback_paths) are executed clean;
#             no stochastic draw, B / K_max untouched (default; the benchmark's design).
# exempt_s1 = exempt + in S1 one random side of each equivalence class is clean
#             (random.Random(seed + 7777), one draw per class, in class order).
# noisy     = fallback tools follow the normal stochastic policy.
FALLBACK_NOISE_MODES: Tuple[str, ...] = ("exempt", "noisy", "exempt_s1")
DEFAULT_FALLBACK_NOISE = "exempt"


def normalize_fallback_noise(value: Optional[str]) -> str:
    """None / "" -> DEFAULT_FALLBACK_NOISE; anything outside FALLBACK_NOISE_MODES raises."""
    mode = (str(value).strip() if value is not None else "") or DEFAULT_FALLBACK_NOISE
    if mode not in FALLBACK_NOISE_MODES:
        raise ValueError(f"unknown fallback_noise {value!r}; expected one of {FALLBACK_NOISE_MODES}")
    return mode


@dataclass(frozen=True)
class ScenarioSpec:
    """Episode-level scenario: label plus the blocked path set P of Algorithm 1."""

    scenario: Scenario = Scenario.S1
    blocked_paths: FrozenSet[str] = frozenset()
    fallback_paths: FrozenSet[str] = frozenset()
    selected_classes: Tuple[str, ...] = ()
    source: str = "explicit"

    @classmethod
    def none(cls) -> "ScenarioSpec":
        return cls(Scenario.S1, frozenset(), frozenset(), (), "none")

    @classmethod
    def from_blocked(cls, blocked: Optional[Iterable[str]], scenario=None,
                     source: str = "explicit",
                     classes: Optional[Sequence[EquivalenceClass]] = None) -> "ScenarioSpec":
        """Spec from a bare blocked set.

        If ``classes`` is given and the scenario is S2, fallback_paths is the unblocked
        side of every partly blocked class (what build_scenario_spec returns for the same
        blocked set). Without ``classes`` no fallback is known (fallback_paths is empty).
        """
        b = frozenset(blocked or ())
        sc = normalize_scenario(scenario) if scenario is not None else (Scenario.S2 if b else Scenario.S1)
        return cls(sc, b, derive_fallback_paths(b, classes) if sc == Scenario.S2 else frozenset(),
                   (), source)

    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario.value,
            "scenario_legacy_name": legacy_name(self.scenario),
            "blocked_paths": sorted(self.blocked_paths),
            "fallback_paths": sorted(self.fallback_paths),
            "selected_classes": list(self.selected_classes),
            "source": self.source,
        }


def task_relevant_classes(classes: Sequence[EquivalenceClass],
                          task_tools: Optional[Iterable[str]]) -> List[EquivalenceClass]:
    """Classes touched by the task's gold actions; all classes if none (or no info)."""
    if not task_tools:
        return list(classes)
    tt = set(task_tools)
    rel = [c for c in classes if c.tools & tt]
    return rel or list(classes)


def build_scenario_spec(
    scenario,
    classes: Sequence[EquivalenceClass],
    rng: random.Random,
    *,
    s2_mode: str = "one_class",
    s2_side: str = "random",
    s3_scope: str = "all_classes",
    task_tools: Optional[Iterable[str]] = None,
    source: str = "sampled",
) -> ScenarioSpec:
    """Construct the blocked path set P for one episode.

    s2_mode:  "one_class"   -> Algorithm 1: one task-relevant class E, one side blocked;
              "all_classes" -> legacy experimental behaviour: one side of every class.
    s2_side:  "random"  -> the blocked side is chosen uniformly (Algorithm 1 wording);
              "primary" -> always block the primary side (forces a switch for tasks whose
                           gold path uses the primary tool).
    s3_scope: "all_classes" -> both sides of every class (legacy, superset of V_tau);
              "relevant"    -> both sides of the task-relevant classes only.
    """
    sc = normalize_scenario(scenario)
    if sc == Scenario.S1 or not classes:
        return ScenarioSpec(Scenario.S1 if sc == Scenario.S1 else sc, frozenset(), frozenset(), (), source)
    if sc == Scenario.S2:
        if s2_mode == "all_classes":
            chosen = list(classes)
        elif s2_mode == "one_class":
            rel = task_relevant_classes(classes, task_tools)
            chosen = [rel[rng.randrange(len(rel))]]
        else:
            raise ValueError(f"unknown s2_mode {s2_mode!r}")
        blocked: Set[str] = set()
        fallback: Set[str] = set()
        for c in chosen:
            if s2_side == "primary":
                block_primary = True
            elif s2_side == "random":
                block_primary = rng.random() < 0.5
            else:
                raise ValueError(f"unknown s2_side {s2_side!r}")
            blocked |= c.primary if block_primary else c.alternatives
            fallback |= c.alternatives if block_primary else c.primary
        return ScenarioSpec(Scenario.S2, frozenset(blocked), frozenset(fallback),
                            tuple(c.name for c in chosen), source)
    # S3
    if s3_scope == "all_classes":
        chosen = list(classes)
    elif s3_scope == "relevant":
        chosen = task_relevant_classes(classes, task_tools)
    else:
        raise ValueError(f"unknown s3_scope {s3_scope!r}")
    blocked = set()
    for c in chosen:
        blocked |= c.tools
    return ScenarioSpec(Scenario.S3, frozenset(blocked), frozenset(),
                        tuple(c.name for c in chosen), source)


def parse_scenario_mix(text: str) -> Dict[Scenario, float]:
    """Parse 'S1=0.5,S2=0.35,S3=0.15' (also accepts ':' separators)."""
    mix: Dict[Scenario, float] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.replace(":", "=").split("=", 1)
        mix[normalize_scenario(k)] = float(v)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError(f"empty scenario mix {text!r}")
    return {k: v / total for k, v in mix.items()}


def sample_scenario(rng: random.Random, mix: Mapping[Scenario, float]) -> Scenario:
    """One rng.random() draw, cumulative over S1, S2, S3 in that order."""
    roll = rng.random()
    acc = 0.0
    last = Scenario.S1
    for sc in (Scenario.S1, Scenario.S2, Scenario.S3):
        p = mix.get(sc, 0.0)
        if p <= 0:
            continue
        acc += p
        last = sc
        if roll < acc:
            return sc
    return last


def stable_hash_int(*parts, digits: int = 8) -> int:
    """Process-independent hash (md5); replaces Python's salted hash() for seeding."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:digits], 16)


def stable_task_seed(global_seed: int, task_id) -> int:
    return global_seed + stable_hash_int(task_id) % 100000


# ---------------------------------------------------------------------------
# Scenario controller
# ---------------------------------------------------------------------------

class ScenarioController:
    """Episode-persistent blocked-path set and fallback noise exemption.

    Independent of InjectionPolicy. ``fallback_noise`` (see FALLBACK_NOISE_MODES) decides
    whether the unblocked fallback tools of the episode (``protected_tools``) bypass the
    stochastic policy. ``classes`` (default: the domain's equivalence classes) is only
    used by "exempt_s1" in S1.
    """

    def __init__(self, spec: Optional[ScenarioSpec] = None, seed: int = 0, domain: str = "retail",
                 fallback_noise: Optional[str] = None,
                 classes: Optional[Sequence[EquivalenceClass]] = None):
        self.seed = seed
        self.domain = domain
        self.fallback_noise = normalize_fallback_noise(fallback_noise)
        self.classes: Tuple[EquivalenceClass, ...] = (
            tuple(DOMAIN_EQUIVALENCE_CLASSES.get(domain, ())) if classes is None else tuple(classes))
        self._set_spec(spec or ScenarioSpec.none())

    def _set_spec(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        # Mutable view kept for backwards compatibility (callers used to mutate
        # engine.permanently_blocked_tools directly).
        self.blocked: Set[str] = set(spec.blocked_paths)
        self.blocked_calls = 0
        self.blocked_calls_per_tool: Dict[str, int] = {}
        # Noise-exempt (clean) tools of this episode.
        self.protected_tools: Set[str] = set()
        if self.fallback_noise != "noisy":
            self.protected_tools |= set(spec.fallback_paths)
            if self.fallback_noise == "exempt_s1" and spec.scenario == Scenario.S1:
                s1_rng = random.Random((self.seed or 0) + 7777)
                for c in self.classes:
                    self.protected_tools |= c.alternatives if s1_rng.random() < 0.5 else c.primary
            self.protected_tools -= self.blocked
        self.fallback_exempt_calls = 0
        self.fallback_exempt_calls_per_tool: Dict[str, int] = {}

    @property
    def scenario(self) -> Scenario:
        return self.spec.scenario

    def is_blocked(self, tool_name: str) -> bool:
        return tool_name in self.blocked

    def is_fallback_exempt(self, tool_name: str) -> bool:
        """True if the tool is an unblocked, noise-exempt fallback in this episode."""
        return tool_name in self.protected_tools and tool_name not in self.blocked

    def decide(self, tool_name: str) -> Optional[Decision]:
        """Scenario-level decision for one tool call (call once per call).

        SCENARIO_BLOCK for blocked tools; a CLEAN ``Decision.exempt_fallback`` for noise-exempt
        fallback tools (counted in fallback_exempt_calls); None otherwise (defer to the
        stochastic policy).
        """
        if tool_name in self.blocked:
            return Decision.scenario_block(f"{tool_name} is on a blocked path ({self.spec.scenario.value})")
        if self.is_fallback_exempt(tool_name):
            self.fallback_exempt_calls += 1
            self.fallback_exempt_calls_per_tool[tool_name] = \
                self.fallback_exempt_calls_per_tool.get(tool_name, 0) + 1
            return Decision.exempt_fallback(
                f"{tool_name} is an unblocked fallback (fallback_noise={self.fallback_noise})")
        return None

    def blocked_response(self, tool_name: str, step: int) -> Tuple[NoiseType, str]:
        """Deterministic explicit-failure text; updates only the scenario counters."""
        # Private, per-call seeded injector: same wording as the stochastic explicit
        # failures, but the stochastic injectors' RNG streams are not consumed.
        from bench2robust.uncertainty.noise_injectors import NoiseInjectorFactory
        h = stable_hash_int(self.seed, tool_name, step)
        nt = BLOCKED_RESPONSE_NOISE[h % len(BLOCKED_RESPONSE_NOISE)]
        text = NoiseInjectorFactory.create_injector(nt, h, self.domain).inject(
            None, tool_name, domain=self.domain)
        self.blocked_calls += 1
        self.blocked_calls_per_tool[tool_name] = self.blocked_calls_per_tool.get(tool_name, 0) + 1
        return nt, str(text)

    def reset(self, spec: Optional[ScenarioSpec] = None, fallback_noise: Optional[str] = None) -> None:
        """Episode boundary: install the next episode's spec (default: no blocking).

        ``fallback_noise`` changes the mode for the next episode; None keeps the current mode.
        """
        if fallback_noise is not None:
            self.fallback_noise = normalize_fallback_noise(fallback_noise)
        self._set_spec(spec or ScenarioSpec.none())

    def summary(self) -> Dict:
        d = self.spec.to_dict()
        d["blocked_paths"] = sorted(self.blocked)
        d["scenario_blocked_calls"] = self.blocked_calls
        d["scenario_blocked_calls_per_tool"] = dict(self.blocked_calls_per_tool)
        d["fallback_noise"] = self.fallback_noise
        d["protected_tools"] = sorted(self.protected_tools - self.blocked)
        d["fallback_exempt_calls"] = self.fallback_exempt_calls
        d["fallback_exempt_calls_per_tool"] = dict(self.fallback_exempt_calls_per_tool)
        return d


__all__ = [
    "DecisionKind", "Decision", "decision_from_noise",
    "EXEC_FAILURE_NOISE", "SILENT_NOISE", "BLOCKED_RESPONSE_NOISE",
    "Scenario", "normalize_scenario", "legacy_name",
    "EquivalenceClass", "RETAIL_EQUIVALENCE_CLASSES", "AIRLINE_EQUIVALENCE_CLASSES",
    "DOMAIN_EQUIVALENCE_CLASSES", "ScenarioSpec", "task_relevant_classes", "derive_fallback_paths",
    "FALLBACK_NOISE_MODES", "DEFAULT_FALLBACK_NOISE", "normalize_fallback_noise",
    "build_scenario_spec", "parse_scenario_mix", "sample_scenario",
    "stable_hash_int", "stable_task_seed", "ScenarioController",
]
