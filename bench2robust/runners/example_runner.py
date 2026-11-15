# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Example benchmark runner template.

Copy this file and implement the abstract methods to add support for a new
benchmark. Then register it in eval/runners/__init__.py.

Example usage after implementation:
    python3 run_inference.py --benchmark my-bench --env my_domain
"""

from typing import Any, Dict, List, Optional, Set

from bench2robust.runner import (
    BenchmarkRunner, BenchmarkTask, DomainConfig, EpisodeResult,
)


class ExampleBenchRunner(BenchmarkRunner):
    """
    Template runner for a hypothetical benchmark.

    Replace this with your actual benchmark integration.
    """

    @property
    def name(self) -> str:
        return "example-bench"

    def get_domains(self) -> List[str]:
        """Return available domains/environments in your benchmark."""
        return ["domain_a", "domain_b"]

    def get_domain_config(self, domain: str) -> DomainConfig:
        """Define tool classifications for uncertainty injection."""
        return DomainConfig(
            name=domain,
            write_tools={"create_item", "delete_item", "update_item"},
            alternative_tools={"search_by_name"},
            tool_configs={},  # Populate with ToolConfig objects for fine-grained control
        )

    def get_tasks(
        self,
        domain: str,
        split: str = "base",
        task_ids: Optional[List[str]] = None,
        start: int = 0,
        end: int = -1,
    ) -> List[BenchmarkTask]:
        """Load tasks from your benchmark's data format."""
        # Example: load from a JSON file
        # tasks_path = f"/path/to/benchmark/data/{domain}/tasks_{split}.json"
        # with open(tasks_path) as f:
        #     raw_tasks = json.load(f)
        #
        # return [
        #     BenchmarkTask(
        #         id=str(t["id"]),
        #         domain=domain,
        #         instruction=t["instruction"],
        #         raw=t,  # Keep original for run_episode
        #     )
        #     for t in raw_tasks[start:end]
        # ]
        raise NotImplementedError("Implement task loading for your benchmark")

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
    ) -> EpisodeResult:
        """
        Run one episode of your benchmark.

        This is where you:
        1. Build your agent/orchestrator
        2. Optionally wrap the environment for uncertainty injection
        3. Run the simulation
        4. Collect and return results
        """
        # Example skeleton:
        #
        # orchestrator = build_my_orchestrator(task.raw, agent_model_str, ...)
        #
        # if uncertainty_config:
        #     orchestrator.env = MyUncertaintyWrapper(
        #         orchestrator.env, uncertainty_config, ...
        #     )
        #
        # result = orchestrator.run()
        #
        # return EpisodeResult(
        #     task_id=task.id,
        #     domain=task.domain,
        #     benchmark=self.name,
        #     model_id=agent_model_str,
        #     instruction=task.instruction,
        #     reward=result.score,
        #     ...
        # )
        raise NotImplementedError("Implement episode execution for your benchmark")

    def setup(self) -> None:
        """One-time initialization (import libs, apply patches, etc.)."""
        pass
