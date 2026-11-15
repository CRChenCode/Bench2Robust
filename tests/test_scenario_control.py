# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Scenario-control regression tests for the released library (tests 1-12).

The paths are exercised over a small in-memory retail-like environment, so the tests
need no benchmark installation. The tau2 runner path additionally needs ``tau2`` to be
importable (it builds tau2 ToolMessage objects) and is skipped otherwise.

Paths:
  engine        GenericInjectionEngine decision API (decide / respond_without_execution /
                finish_executed), executed by a minimal driver
  vita_adapter  adapters.vitabench.adapter.InstrumentedVitaBenchEnv
  tau2_runner   runners.tau2_bench_runner._Tau2UncertaintyEnv
  bfcl          adapters.bfcl.adapter.BFCLExecutionInterceptor.execute_with_interception

Semantics under test: scenario-blocked tools are never executed and stay blocked until episode
reset, regardless of B and K_max; unblocked fallback tools are exempt from stochastic noise
(configurable via ``fallback_noise``: "exempt" (default), "exempt_s1" or "noisy"); all other
tools follow the stochastic policy.
"""

import copy
import hashlib
import importlib.util
import json
import random
import sys
import types
import unittest

from bench2robust.uncertainty.config import EpisodeUncertaintyConfig, NoiseType, ToolConfig, TriggerTiming
from bench2robust.uncertainty.generic_wrapper import GenericInjectionEngine
from bench2robust.uncertainty.scenario import (
    BLOCKED_RESPONSE_NOISE, DecisionKind, RETAIL_EQUIVALENCE_CLASSES, Scenario, ScenarioController,
    ScenarioSpec, build_scenario_spec, parse_scenario_mix, sample_scenario, stable_task_seed,
)

HIGH_NOISE = {"clean": 0.0, "timeout": 0.5, "server_error": 0.5}
NO_NOISE = {"clean": 1.0}
READS = ("find_user_id_by_name_zip", "find_user_id_by_email", "get_user_by_order",
         "get_order_details", "get_orders_for_user", "get_user_details")
WRITES = ("cancel_pending_order",)
PRIMARY = {"find_user_id_by_name_zip", "get_order_details"}
FALLBACK = {"find_user_id_by_email", "get_user_by_order", "get_orders_for_user"}
HAS_TAU2 = importlib.util.find_spec("tau2") is not None


def _ensure_vita_toolmessage():
    try:
        from vita.data_model.message import ToolMessage  # noqa: F401
        return
    except Exception:
        pass

    class ToolMessage(types.SimpleNamespace):
        pass
    m1, m2, m3 = (types.ModuleType(n) for n in ("vita", "vita.data_model", "vita.data_model.message"))
    m3.ToolMessage = ToolMessage
    sys.modules.update({"vita": m1, "vita.data_model": m2, "vita.data_model.message": m3})


class FakeRetailEnv:
    """Tiny stateful environment; get_response(message) executes one tool."""

    domain_name = "retail"
    policy = ""
    tools = None
    user_tools = None
    solo_mode = False

    def __init__(self):
        self.db = {"users": {"u1": {"email": "a@x", "zip": "1", "orders": ["#1"]}},
                   "orders": {"#1": {"user_id": "u1", "status": "pending", "items": ["i1"]}}}
        self.exec_log = []

    def state_hash(self):
        return hashlib.sha256(json.dumps(self.db, sort_keys=True).encode()).hexdigest()

    def _run(self, name, args):
        self.exec_log.append(name)
        if name in ("find_user_id_by_name_zip", "find_user_id_by_email", "get_user_by_order"):
            return "u1"
        if name == "get_user_details":
            return self.db["users"]["u1"]
        if name == "get_order_details":
            return {"order_id": "#1", **self.db["orders"]["#1"]}
        if name == "get_orders_for_user":
            return [{"order_id": k, **v} for k, v in self.db["orders"].items()]
        if name == "cancel_pending_order":
            if self.db["orders"]["#1"]["status"] != "pending":
                return "Error: Non-pending order cannot be cancelled"
            self.db["orders"]["#1"]["status"] = "cancelled"
            return {"order_id": "#1", "status": "cancelled"}
        return f"Error: Tool '{name}' not found."

    def get_response(self, message):
        out = self._run(message.name, message.arguments)
        content = out if isinstance(out, str) else json.dumps(out)
        try:
            from tau2.data_model.message import ToolMessage
            return ToolMessage(id=message.id, content=content, requestor=message.requestor, role="tool",
                               error=content.startswith("Error"))
        except Exception:
            return types.SimpleNamespace(id=message.id, content=content, error=content.startswith("Error"))


def make_config(B, K, priors, seed=11):
    tc = {}
    for n in READS:
        tc[n] = ToolConfig(name=n, is_write_action=False, has_alternatives=True, alternatives=[],
                           uncertainty_priors=dict(priors))
    for n in WRITES:
        tc[n] = ToolConfig(name=n, is_write_action=True, has_alternatives=False, alternatives=[],
                           uncertainty_priors={"clean": 1.0})
    return EpisodeUncertaintyConfig(tool_configs=tc, trigger_timing=TriggerTiming.UNIFORM,
                                    max_consecutive_failures=K, max_total_injections=B, seed=seed)


def msg(name, i, args=None):
    return types.SimpleNamespace(id=f"c{i}", name=name, arguments=args or {}, requestor="assistant")


class Path:
    """Uniform driver: call(tool) -> record dict."""

    def __init__(self, kind, blocked, B, K, priors, seed=11, scenario="S2", fallback_noise=None):
        self.kind = kind
        self.env = FakeRetailEnv()
        cfg = make_config(B, K, priors, seed)
        # Explicit spec (engine, tau2 runner) and bare blocked set (VitaBench, BFCL) give the same
        # fallback side: the unblocked side of every partly blocked retail class.
        spec = ScenarioSpec.from_blocked(blocked, scenario=scenario, classes=RETAIL_EQUIVALENCE_CLASSES)
        self.i = 0
        fb = {"fallback_noise": fallback_noise}
        if kind == "engine":
            self.engine = GenericInjectionEngine(cfg, domain="retail", scenario_spec=spec, **fb)
        elif kind == "vita_adapter":
            _ensure_vita_toolmessage()
            from bench2robust.adapters.vitabench.adapter import InstrumentedVitaBenchEnv
            self.w = InstrumentedVitaBenchEnv(self.env, cfg, "retail", permanently_blocked_tools=set(blocked), **fb)
            self.engine = self.w.engine
        elif kind == "tau2_runner":
            from bench2robust.runners.tau2_bench_runner import _Tau2UncertaintyEnv
            self.w = _Tau2UncertaintyEnv(self.env, "retail", cfg, with_alternatives=True, scenario_spec=spec, **fb)
            self.engine = None
        elif kind == "bfcl":
            from bench2robust.adapters.bfcl.adapter import BFCLExecutionInterceptor
            self.w = BFCLExecutionInterceptor(cfg, domain="retail", permanently_blocked_tools=set(blocked), **fb)
            self.engine = self.w.engine
        else:
            raise ValueError(kind)

    @property
    def policy_state(self):
        return (self.w.injection_policy if self.kind == "tau2_runner" else self.engine.injection_policy).state

    @property
    def blocked(self):
        return self.w.scenario.blocked if self.kind == "tau2_runner" else self.engine.permanently_blocked_tools

    @property
    def controller(self):
        return self.w.scenario if self.kind == "tau2_runner" else self.engine.scenario

    def counters(self):
        st = self.policy_state
        return (st.total_injections, dict(st.consecutive_failures),
                {k: dict(v) for k, v in st.per_tool_error_counts.items()})

    def call(self, tool):
        self.i += 1
        h0, n0, c0, blk = self.env.state_hash(), len(self.env.exec_log), self.counters(), tool in self.blocked
        if self.kind == "engine":
            d = self.engine.decide(tool)
            if d.execute:
                clean = self.env._run(tool, {})
                out = self.engine.finish_executed(d, tool, clean)
            else:
                out = self.engine.respond_without_execution(d, tool)
            kind = d.kind.value
        elif self.kind == "bfcl":
            def execute(func_call_list, **kw):
                return [json.dumps(self.env._run(c.split("(")[0], {})) for c in func_call_list], None
            res, _, dec = self.w.execute_with_interception(execute, [f"{tool}()"])
            out, kind = res[0], dec[0].kind.value
        else:
            m = self.w.get_response(msg(tool, self.i))
            out = m.content
            ev = (self.engine.get_events() if self.engine else self.w.uncertainty_events)
            kind = None
            if ev and ev[-1]["step"] == (self.engine.step_count if self.engine else self.w._step):
                kind = ev[-1].get("decision")
        out = out if isinstance(out, str) else json.dumps(out)
        return {"tool": tool, "blocked": blk, "executed": len(self.env.exec_log) > n0, "out": out,
                "kind": kind, "c0": c0, "c1": self.counters(), "h0": h0, "h1": self.env.state_hash()}

    def reset(self, blocked=()):
        if self.kind == "tau2_runner":
            raise unittest.SkipTest("tau2 runner builds a new wrapper per episode (run_episode)")
        self.env = FakeRetailEnv()
        if self.kind == "vita_adapter":
            self.w.base_env = self.env
            self.w.reset(permanently_blocked_tools=set(blocked))
        elif self.kind == "bfcl":
            self.w.reset(permanently_blocked_tools=set(blocked))
        else:
            self.engine.reset(permanently_blocked_tools=set(blocked))


KINDS = ["engine", "vita_adapter", "bfcl"] + (["tau2_runner"] if HAS_TAU2 else [])


def is_err(s):
    return "Error" in s or "error" in s or "timed out" in s


class ScenarioControlTests(unittest.TestCase):
    def each(self):
        for k in KINDS:
            with self.subTest(path=k):
                yield k

    def test_01_blocked_after_budget_B_exhausted(self):
        for k in self.each():
            p = Path(k, PRIMARY, B=1, K=1, priors=HIGH_NOISE)
            r_u = p.call("get_user_details")                      # consumes B=1
            self.assertTrue(r_u["kind"].startswith("stochastic"))
            for _ in range(4):
                r = p.call("get_order_details")
                self.assertEqual(r["kind"], DecisionKind.SCENARIO_BLOCK.value)
                self.assertTrue(is_err(r["out"]))

    def test_02_blocked_after_K_max_consecutive(self):
        for k in self.each():
            p = Path(k, PRIMARY, B=10, K=1, priors=HIGH_NOISE)
            outs = [p.call("find_user_id_by_name_zip") for _ in range(5)]
            self.assertTrue(all(r["kind"] == "scenario_block" and "u1" not in r["out"] for r in outs))

    def test_03_blocked_calls_do_not_change_stochastic_counters(self):
        for k in self.each():
            p = Path(k, PRIMARY, B=1, K=1, priors=HIGH_NOISE)
            for t in ("get_order_details", "get_user_details", "get_order_details", "find_user_id_by_name_zip"):
                r = p.call(t)
                if r["blocked"]:
                    self.assertEqual(r["c0"], r["c1"])

    def test_04_blocked_calls_do_not_execute(self):
        for k in self.each():
            p = Path(k, PRIMARY, B=5, K=2, priors=NO_NOISE)
            for _ in range(3):
                self.assertFalse(p.call("get_order_details")["executed"])
            self.assertTrue(p.call("get_user_details")["executed"])

    def test_05_blocked_write_leaves_state_unchanged(self):
        for k in self.each():
            p = Path(k, PRIMARY | {"cancel_pending_order"}, B=5, K=2, priors=NO_NOISE)
            for _ in range(2):
                r = p.call("cancel_pending_order")
                self.assertFalse(r["executed"])
                self.assertEqual(r["h0"], r["h1"])
            p.blocked.discard("cancel_pending_order")             # positive control
            r = p.call("cancel_pending_order")
            self.assertTrue(r["executed"])
            self.assertNotEqual(r["h0"], r["h1"])

    def test_06_s2_not_solvable_by_retrying_blocked_primary(self):
        for k in self.each():
            p = Path(k, PRIMARY, B=5, K=2, priors=NO_NOISE)
            outs = [p.call(t)["out"] for t in ["find_user_id_by_name_zip", "get_order_details"] * 5]
            self.assertFalse(any("u1" in o or "#1" in o for o in outs))

    def test_07_s2_solvable_through_fallback(self):
        for k in self.each():
            p = Path(k, PRIMARY, B=5, K=2, priors=NO_NOISE)
            self.assertIn("u1", p.call("find_user_id_by_email")["out"])
            self.assertIn("#1", p.call("get_orders_for_user")["out"])
            r = p.call("cancel_pending_order")
            self.assertTrue(r["executed"] and r["h0"] != r["h1"])

    def test_08_s3_all_viable_paths_blocked_for_whole_episode(self):
        s3 = set(READS) - {"get_user_details"}
        for k in self.each():
            p = Path(k, s3, B=1, K=1, priors=HIGH_NOISE, scenario="S3")
            for _ in range(4):
                for t in sorted(s3):
                    r = p.call(t)
                    self.assertEqual(r["kind"], "scenario_block")
                    self.assertFalse(r["executed"])

    def test_09_fallback_noise_noisy_mode_follows_stochastic_policy(self):
        # fallback_noise="noisy": unblocked fallbacks follow the normal stochastic policy (the
        # previous default of this release); see test_09b for the default ("exempt").
        for k in self.each():
            p = Path(k, PRIMARY, B=5, K=2, priors=HIGH_NOISE, fallback_noise="noisy")
            kinds = [p.call(t)["kind"] for t in ("find_user_id_by_email", "get_orders_for_user", "get_user_by_order")]
            self.assertTrue(any(x and x.startswith("stochastic") for x in kinds), kinds)

    def test_09b_fallback_exempt_by_default_never_noised(self):
        # Default fallback_noise="exempt": with P(noise)=1 every fallback call is clean, executed,
        # and leaves B / K_max untouched; other unblocked tools are still noised; blocked tools
        # stay blocked after B and K_max are used up.
        for k in self.each():
            p = Path(k, PRIMARY, B=1, K=1, priors=HIGH_NOISE)
            self.assertEqual(self.controller_mode(p), "exempt")
            self.assertEqual(set(p.controller.protected_tools), FALLBACK)
            for _ in range(4):
                for t in sorted(FALLBACK):
                    r = p.call(t)
                    self.assertIn(r["kind"], (None, DecisionKind.CLEAN.value), r)
                    self.assertTrue(r["executed"] and not is_err(r["out"]), r)
                    self.assertEqual(r["c0"][0], r["c1"][0])
                    self.assertEqual(r["c1"][1].get(t, 0), 0)
            r = p.call("get_user_details")                        # B=1 still available
            self.assertTrue(r["kind"].startswith("stochastic"), r)
            for _ in range(4):
                for t in sorted(PRIMARY):
                    r = p.call(t)
                    self.assertEqual(r["kind"], "scenario_block")
                    self.assertFalse(r["executed"])
                r = p.call("find_user_id_by_email")
                self.assertTrue(r["executed"] and "u1" in r["out"] and not is_err(r["out"]), r)
            self.assertEqual(p.counters()[0], 1)
            self.assertEqual(p.controller.summary()["fallback_exempt_calls"], 16)

    def controller_mode(self, p):
        return p.controller.fallback_noise

    def test_09c_exempt_s1_one_side_per_pair(self):
        # fallback_noise="exempt_s1": in S1 one random side of each equivalence class is clean
        # (random.Random(seed + 7777), one draw per class); "exempt" exempts nothing in S1.
        pair_tools = sorted(set(READS) & set().union(*[c.tools for c in RETAIL_EQUIVALENCE_CLASSES]))
        for k in self.each():
            for seed in (11, 12, 13):
                r_ = random.Random(seed + 7777)
                want = set()
                for c in RETAIL_EQUIVALENCE_CLASSES:
                    want |= c.alternatives if r_.random() < 0.5 else c.primary
                p = Path(k, (), B=50, K=2, priors=HIGH_NOISE, seed=seed, scenario="S1", fallback_noise="exempt_s1")
                self.assertEqual(set(p.controller.protected_tools), want)
                for t in pair_tools:
                    r = p.call(t)
                    if t in want:
                        self.assertIn(r["kind"], (None, "clean"), r)
                        self.assertTrue(r["executed"])
                    else:
                        self.assertTrue(r["kind"] and r["kind"].startswith("stochastic"), (t, r))
                q = Path(k, (), B=50, K=2, priors=HIGH_NOISE, seed=seed, scenario="S1")
                self.assertEqual(set(q.controller.protected_tools), set())

    def test_09d_fallback_noise_validation_and_summary(self):
        with self.assertRaises(ValueError):
            GenericInjectionEngine(make_config(1, 1, HIGH_NOISE), domain="retail", fallback_noise="bogus")
        eng = GenericInjectionEngine(make_config(1, 1, HIGH_NOISE), domain="retail",
                                     permanently_blocked_tools=set(PRIMARY))
        d = eng.decide("get_orders_for_user")
        self.assertTrue(d.kind == DecisionKind.CLEAN and d.fallback_exempt and d.execute)
        self.assertEqual(eng.should_inject("get_user_by_order"), (False, None))
        self.assertEqual(eng.inject(NoiseType.PARTIAL, "get_user_by_order", {"u": 1}), {"u": 1})
        sc = eng.get_counters()["scenario"]
        self.assertEqual((sc["fallback_noise"], sc["fallback_exempt_calls"]), ("exempt", 2))
        self.assertEqual(set(sc["protected_tools"]), FALLBACK)
        eng.reset(permanently_blocked_tools=set(PRIMARY), fallback_noise="noisy")
        self.assertTrue(eng.decide("get_orders_for_user").is_stochastic)
        s3 = ScenarioSpec.from_blocked(set().union(*[c.tools for c in RETAIL_EQUIVALENCE_CLASSES]),
                                       scenario="S3", classes=RETAIL_EQUIVALENCE_CLASSES)
        self.assertEqual(s3.fallback_paths, frozenset())

    def test_10_reset_clears_scenario_state(self):
        for k in self.each():
            if k == "tau2_runner":
                continue
            p = Path(k, PRIMARY, B=1, K=1, priors=HIGH_NOISE)
            p.call("get_order_details")
            p.call("get_user_details")
            p.reset(blocked=())
            self.assertEqual(p.counters(), (0, {}, {}))
            self.assertEqual(set(p.blocked), set())
            r = p.call("get_order_details")
            self.assertNotEqual(r["kind"], "scenario_block")

    def test_11_fixed_seed_reproduces_scenario_and_noise(self):
        seq = ["get_order_details", "get_user_details", "find_user_id_by_email", "get_orders_for_user"] * 4
        pri = {"clean": 0.4, "timeout": 0.3, "server_error": 0.3}
        for k in self.each():
            a, b = Path(k, PRIMARY, 3, 2, pri, seed=5), Path(k, PRIMARY, 3, 2, pri, seed=5)
            self.assertEqual([(r["kind"], r["out"]) for r in (a.call(t) for t in seq)],
                             [(r["kind"], r["out"]) for r in (b.call(t) for t in seq)])
        mix = parse_scenario_mix("S1=0.5,S2=0.35,S3=0.15")
        self.assertEqual([sample_scenario(random.Random(s), mix) for s in range(100)],
                         [sample_scenario(random.Random(s), mix) for s in range(100)])
        self.assertEqual(stable_task_seed(42, "66"), stable_task_seed(42, "66"))
        s_a = build_scenario_spec("S2", RETAIL_EQUIVALENCE_CLASSES, random.Random(9), s2_mode="one_class")
        s_b = build_scenario_spec("S2", RETAIL_EQUIVALENCE_CLASSES, random.Random(9), s2_mode="one_class")
        self.assertEqual(s_a, s_b)

    def test_12_blocked_response_never_silent_and_never_consults_policy(self):
        from unittest.mock import patch
        for k in self.each():
            p = Path(k, PRIMARY, B=5, K=5, priors={"clean": 0.0, "partial": 0.5, "factual_error": 0.5})
            pol = p.w.injection_policy if k == "tau2_runner" else p.engine.injection_policy
            guards = [patch.object(pol, n, side_effect=AssertionError(n))
                      for n in ("should_inject", "decide") + (("can_inject",) if hasattr(pol, "can_inject") else ())]
            for g in guards:
                g.start()
            try:
                for _ in range(3):
                    r = p.call("get_order_details")
                    self.assertTrue(is_err(r["out"]) and "#1" not in r["out"])
            finally:
                for g in guards:
                    g.stop()

    def test_12_legacy_api_on_blocked_tool(self):
        eng = GenericInjectionEngine(make_config(1, 1, {"clean": 0.0, "partial": 1.0}),
                                     permanently_blocked_tools={"get_order_details"})
        ok, nt = eng.should_inject("get_order_details")
        self.assertTrue(ok)
        self.assertIn(nt, BLOCKED_RESPONSE_NOISE)
        out = eng.inject(NoiseType.PARTIAL, "get_order_details", {"order_id": "#1"})
        self.assertTrue(is_err(out) and "#1" not in out)
        self.assertEqual((eng.total_injections, eng.injection_policy.state.total_injections), (0, 0))

    def test_scenario_controller_reset(self):
        ctl = ScenarioController(ScenarioSpec.from_blocked({"x"}, "S2"), seed=1)
        self.assertTrue(ctl.is_blocked("x"))
        ctl.blocked_response("x", 1)
        ctl.reset()
        self.assertEqual((ctl.is_blocked("x"), ctl.blocked_calls, ctl.scenario), (False, 0, Scenario.S1))


if __name__ == "__main__":
    unittest.main()
