# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

import copy
import unittest
from unittest.mock import patch

import bench2robust
from bench2robust.adapters import list_adapters
from bench2robust.adapters.tau_bench import TauBenchAdapter
from bench2robust.adapters.tau_bench.tool_configs import (
    AIRLINE_TOOL_CONFIGS,
    RETAIL_TOOL_CONFIGS,
    TELECOM_TOOL_CONFIGS,
)
from bench2robust.adapters.vitabench import VitaBenchAdapter
from bench2robust.runners import Tau2BenchRunner
from bench2robust.runners import tau2_bench_runner
from bench2robust.uncertainty.config import (
    ConfigFactory,
    ConfigurationError,
    EpisodeUncertaintyConfig,
    NoiseType,
    ToolConfig,
)
from bench2robust.uncertainty.domain_registry import DomainRegistry
from bench2robust.uncertainty.generic_wrapper import GenericInjectionEngine
from bench2robust.uncertainty.injection_policy import InjectionPolicy
from bench2robust.uncertainty.noise_injectors import (
    FactualErrorInjector,
    PartialResponseInjector,
    StaleResponseInjector,
)


def _tool(
    name="lookup",
    *,
    write=False,
    alternatives=False,
    priors=None,
    max_consecutive_failures=None,
):
    return ToolConfig(
        name=name,
        is_write_action=write,
        has_alternatives=alternatives,
        uncertainty_priors=priors or {"clean": 0.0, "timeout": 1.0},
        max_consecutive_failures=max_consecutive_failures,
    )


class ConfigurationTests(unittest.TestCase):
    def test_rejects_invalid_probability_sum(self):
        with self.assertRaises(ConfigurationError):
            _tool(priors={"clean": 0.4, "timeout": 0.4})

    def test_factory_overrides_do_not_mutate_shared_defaults(self):
        baseline = ConfigFactory.create_tau_bench_config("retail")
        tool_name = next(iter(baseline.tool_configs))
        baseline_priors = dict(baseline.tool_configs[tool_name].uncertainty_priors)

        overridden = ConfigFactory.create_tau_bench_config(
            "retail",
            custom_priors={tool_name: {"clean": 0.0, "timeout": 1.0}},
        )
        fresh = ConfigFactory.create_tau_bench_config("retail")

        self.assertEqual(
            overridden.tool_configs[tool_name].uncertainty_priors,
            {"clean": 0.0, "timeout": 1.0},
        )
        self.assertEqual(fresh.tool_configs[tool_name].uncertainty_priors, baseline_priors)
        self.assertIsNot(fresh.tool_configs[tool_name], baseline.tool_configs[tool_name])

    def test_factory_rejects_unknown_override_tool(self):
        with self.assertRaises(ConfigurationError):
            ConfigFactory.create_tau_bench_config(
                "retail",
                custom_priors={"not_a_tool": {"clean": 1.0}},
            )


class InjectionPolicyTests(unittest.TestCase):
    def test_never_injects_write_actions(self):
        config = EpisodeUncertaintyConfig(tool_configs={"write": _tool("write", write=True)})
        policy = InjectionPolicy(config)
        self.assertEqual(policy.should_inject("write", step=1), (False, None))

    def test_unknown_tool_fails_closed(self):
        config = EpisodeUncertaintyConfig(tool_configs={"lookup": _tool()})
        policy = InjectionPolicy(config)
        self.assertEqual(policy.should_inject("unknown", step=1), (False, None))

    def test_enforces_budget(self):
        config = EpisodeUncertaintyConfig(
            tool_configs={"lookup": _tool(alternatives=True)},
            max_consecutive_failures=10,
            max_total_injections=1,
        )
        policy = InjectionPolicy(config)
        first = policy.should_inject("lookup", step=1)
        self.assertEqual(first, (True, NoiseType.TIMEOUT))
        policy.record_execution("lookup", *first)
        self.assertEqual(policy.should_inject("lookup", step=2), (False, None))

    def test_limits_repeated_error_without_alternative(self):
        config = EpisodeUncertaintyConfig(
            tool_configs={"lookup": _tool(alternatives=False)},
            max_consecutive_failures=10,
            max_total_injections=5,
        )
        policy = InjectionPolicy(config)
        first = policy.should_inject("lookup", step=1)
        policy.record_execution("lookup", *first)
        self.assertEqual(policy.should_inject("lookup", step=2), (False, None))

    def test_reset_replays_seeded_decision(self):
        config = EpisodeUncertaintyConfig(
            tool_configs={
                "lookup": _tool(
                    priors={"clean": 0.5, "timeout": 0.5},
                    alternatives=True,
                )
            },
            seed=7,
        )
        policy = InjectionPolicy(config)
        first = policy.should_inject("lookup", step=1)
        policy.reset()
        self.assertEqual(policy.should_inject("lookup", step=1), first)


class NoiseInjectorTests(unittest.TestCase):
    def test_partial_response_is_deterministic_and_non_mutating(self):
        response = {
            "order_id": "synthetic-order-1",
            "status": "delivered",
            "items": [{"item_id": "synthetic-item-1"}],
            "fulfillments": [{"state": "delivered"}],
            "payment_history": [{"amount": 10}],
        }
        original = copy.deepcopy(response)

        first = PartialResponseInjector(seed=11).inject(
            response, "get_order_details", domain="retail"
        )
        second = PartialResponseInjector(seed=11).inject(
            response, "get_order_details", domain="retail"
        )

        self.assertEqual(first, second)
        self.assertEqual(response, original)
        self.assertEqual(first["status"], "")
        self.assertEqual(first["items"], [])
        self.assertEqual(first["fulfillments"], [])
        self.assertEqual(first["payment_history"], [])
        self.assertEqual(first["order_id"], "synthetic-order-1")


class DomainRegistryTests(unittest.TestCase):
    def setUp(self):
        DomainRegistry.clear()
        VitaBenchAdapter("delivery")

    def test_vitabench_partial_uses_registered_critical_fields_and_sentinels(self):
        response = {
            "status": "delivered",
            "products": [{"id": "synthetic-product-1"}],
            "total_price": 25.0,
        }
        observed = PartialResponseInjector(seed=3).inject(
            response,
            "get_delivery_order_detail",
            domain="vitabench_delivery",
        )

        self.assertEqual(observed["status"], "")
        self.assertEqual(observed["products"], [])
        self.assertEqual(observed["total_price"], 0.0)
        self.assertEqual(response["status"], "delivered")

    def test_vitabench_stale_and_factual_rules_are_registered(self):
        stale = StaleResponseInjector(seed=3).inject(
            {"status": "delivered"},
            "get_delivery_order_status",
            domain="vitabench_delivery",
        )
        first = FactualErrorInjector(seed=7, domain="vitabench_delivery").inject(
            {"price": 100.0, "score": 4.5},
            "get_delivery_product_info",
            domain="vitabench_delivery",
        )
        second = FactualErrorInjector(seed=7, domain="vitabench_delivery").inject(
            {"price": 100.0, "score": 4.5},
            "get_delivery_product_info",
            domain="vitabench_delivery",
        )

        self.assertEqual(stale["status"], "unpaid")
        self.assertEqual(first, second)
        self.assertNotEqual(first, {"price": 100.0, "score": 4.5})

    def test_tau_write_tools_are_protected_by_policy(self):
        expected_write_tools = {
            "retail": {
                "cancel_pending_order",
                "exchange_delivered_order_items",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "modify_user_address",
                "return_delivered_order_items",
                "place_new_order",
            },
            "airline": {
                "book_reservation",
                "cancel_reservation",
                "send_certificate",
                "update_reservation_baggages",
                "update_reservation_flights",
                "update_reservation_passengers",
            },
            "telecom": {
                "suspend_line",
                "resume_line",
                "send_payment_request",
                "enable_roaming",
                "disable_roaming",
                "refuel_data",
                "set_network_mode_preference",
                "toggle_airplane_mode",
                "reseat_sim_card",
                "toggle_data",
                "toggle_roaming",
                "toggle_data_saver_mode",
                "set_apn_settings",
                "reset_apn_settings",
                "toggle_wifi",
                "toggle_wifi_calling",
                "connect_vpn",
                "disconnect_vpn",
                "grant_app_permission",
                "reboot_device",
                "make_payment",
            },
        }
        configs = {
            "retail": RETAIL_TOOL_CONFIGS,
            "airline": AIRLINE_TOOL_CONFIGS,
            "telecom": TELECOM_TOOL_CONFIGS,
        }

        for domain, domain_configs in configs.items():
            actual_write_tools = {
                name for name, config in domain_configs.items()
                if config.is_write_action
            }
            self.assertEqual(actual_write_tools, expected_write_tools[domain])
            for tool_name in actual_write_tools:
                with self.subTest(domain=domain, tool_name=tool_name):
                    self.assertEqual(
                        domain_configs[tool_name].uncertainty_priors,
                        {"clean": 1.0},
                    )


class EngineAndPackagingTests(unittest.TestCase):
    def test_permanently_blocked_tool_records_event(self):
        config = EpisodeUncertaintyConfig(
            tool_configs={"lookup": _tool(alternatives=True)},
            seed=5,
        )
        engine = GenericInjectionEngine(config, permanently_blocked_tools={"lookup"})

        should_inject, noise_type = engine.should_inject("lookup")
        observed = engine.inject(noise_type, "lookup", {"status": "ok"})

        self.assertTrue(should_inject)
        self.assertNotEqual(observed, {"status": "ok"})
        # Revision: scenario blocks are recorded as events but are NOT stochastic
        # injections (they never consume B / K_max / per-error counters).
        self.assertEqual(engine.total_injections, 0)
        self.assertEqual(engine.scenario_blocked_calls, 1)
        self.assertEqual(engine.injection_policy.state.total_injections, 0)
        self.assertEqual(engine.get_events()[0]["tool"], "lookup")
        self.assertTrue(engine.get_events()[0]["permanently_blocked"])

    def test_permanent_block_persists_beyond_write_protection_and_budget(self):
        """Revision: scenario blocking is episode-persistent.

        The previous release asserted that a blocked tool becomes CLEAN once the
        stochastic budget is used (blocking lapsed). Algorithm 1 requires the
        opposite: blocked calls (including blocked write actions) are never executed,
        never consult can_inject(), and stay blocked until reset().
        """
        config = EpisodeUncertaintyConfig(
            tool_configs={
                "write": _tool("write", write=True),
                "lookup": _tool("lookup", alternatives=True),
            },
            max_total_injections=1,
        )
        engine = GenericInjectionEngine(
            config,
            permanently_blocked_tools={"write", "lookup"},
        )

        from bench2robust.uncertainty.scenario import DecisionKind

        with patch.object(engine.injection_policy, "can_inject",
                          side_effect=AssertionError("blocked tools must not consult can_inject")):
            for tool in ("write", "lookup", "lookup", "lookup"):
                decision = engine.decide(tool)
                self.assertEqual(decision.kind, DecisionKind.SCENARIO_BLOCK)
                self.assertFalse(decision.execute)
                engine.respond_without_execution(decision, tool)
        self.assertEqual(engine.injection_policy.state.total_injections, 0)
        self.assertEqual(engine.injection_policy.state.consecutive_failures, {})
        self.assertEqual(engine.scenario_blocked_calls, 4)
        engine.reset()
        self.assertNotEqual(engine.decide("lookup").kind, DecisionKind.SCENARIO_BLOCK)

    def test_engine_reset_recreates_seeded_injectors(self):
        config = EpisodeUncertaintyConfig(tool_configs={"lookup": _tool()})
        engine = GenericInjectionEngine(config)
        first_injector = engine.injectors[NoiseType.PARTIAL]

        engine.reset()

        self.assertIsNot(engine.injectors[NoiseType.PARTIAL], first_injector)
        self.assertEqual(engine.should_inject("lookup"), (True, NoiseType.TIMEOUT))

    def test_tau_adapter_metadata_is_isolated_and_legacy_wrapper_is_removed(self):
        first = TauBenchAdapter("retail")
        second = TauBenchAdapter("retail")
        tool_name = "get_order_details"
        first.tool_configs[tool_name].uncertainty_priors["clean"] = 0.0

        self.assertNotEqual(
            first.tool_configs[tool_name].uncertainty_priors,
            second.tool_configs[tool_name].uncertainty_priors,
        )
        with self.assertRaisesRegex(NotImplementedError, "Tau2BenchRunner"):
            first.wrap_environment(object(), ConfigFactory.create_tau_bench_config("retail"))

    def test_tau_domain_metadata_does_not_require_tau2(self):
        with patch.object(tau2_bench_runner, "_tau2_imported", False):
            config = Tau2BenchRunner().get_domain_config("telecom")
        self.assertEqual(config.write_tools, {
            name for name, tool in TELECOM_TOOL_CONFIGS.items() if tool.is_write_action
        })

    def test_public_imports_and_lazy_adapter_discovery(self):
        self.assertEqual(bench2robust.__version__, "0.2.0")
        from bench2robust import uncertainty
        self.assertEqual(uncertainty.__version__, bench2robust.__version__)
        self.assertEqual(list_adapters(), ["bfcl", "tau_bench", "vitabench"])
        self.assertEqual(Tau2BenchRunner().name, "tau2-bench")

    def test_missing_tau2_dependency_has_actionable_error(self):
        with (
            patch.object(tau2_bench_runner, "_tau2_imported", False),
            patch.object(tau2_bench_runner.importlib.util, "find_spec", return_value=None),
            patch.dict(tau2_bench_runner.os.environ, {}, clear=True),
            self.assertRaisesRegex(ImportError, "BENCH2ROBUST_TAU2_PATH"),
        ):
            tau2_bench_runner._ensure_imports()


if __name__ == "__main__":
    unittest.main()
