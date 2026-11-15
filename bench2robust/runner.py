# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
benchmark_runner.py
===================
Abstract interface for benchmark runners.

Any benchmark (tau2-bench, SWE-bench, custom, etc.) must implement this
interface to plug into the unified evaluation pipeline.

The shared runner interface defines:
  - Benchmark-specific task loading
  - Episode execution and standardized result serialization
  - Optional uncertainty injection through runner-owned integration hooks
  - Domain-specific tool classifications

Model clients, evaluator backends, concurrency, and progress orchestration are
owned by each benchmark integration or by downstream applications; they are not
provided by this core package.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Standardized task representation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """A single task from any benchmark, normalized to a common format."""
    id: str
    domain: str
    instruction: str  # Human-readable task description
    raw: Any = None   # Original benchmark-specific task object (for the runner)
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Domain configuration (tool classifications for uncertainty)
# ---------------------------------------------------------------------------

@dataclass
class DomainConfig:
    """Per-domain tool classification for uncertainty injection."""
    name: str
    write_tools: Set[str] = field(default_factory=set)
    alternative_tools: Set[str] = field(default_factory=set)
    tool_configs: Dict = field(default_factory=dict)  # tool_name → ToolConfig


# ---------------------------------------------------------------------------
# Episode result (benchmark-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Unified result for any benchmark episode."""
    task_id: str
    domain: str
    benchmark: str  # e.g. "tau2-bench", "swe-bench", "custom"
    model_id: str
    instruction: str
    reward: float  # Primary success metric (0.0 - 1.0)
    reward_info: Dict = field(default_factory=dict)
    # Uncertainty fields
    inject_mode: bool = False
    uncertainty_seed: Optional[int] = None
    max_injections_configured: Optional[int] = None
    noise_types_configured: Optional[List[str]] = None
    total_injections_actual: int = 0
    uncertainty_events: Optional[List[Dict]] = None
    # Judge
    judge_verdict: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_tokens: int = 0
    # Token usage
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    user_prompt_tokens: int = 0
    user_completion_tokens: int = 0
    total_steps: int = 0
    duration: float = 0.0
    error: Optional[str] = None
    chat_history: Optional[List[Dict]] = None
    # Benchmark-specific extra data
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners.

    Subclass this to add support for a new benchmark. The eval pipeline
    calls these methods in order:

        1. get_domains()          → list available domains
        2. get_domain_config(d)   → tool classifications for uncertainty
        3. get_tasks(...)         → load tasks for a domain
        4. run_episode(...)       → execute one task, return EpisodeResult
        5. get_messages(result)   → extract messages for LLM judge (optional)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this benchmark (e.g. 'tau2-bench')."""
        ...

    @abstractmethod
    def get_domains(self) -> List[str]:
        """Return list of available domains/environments."""
        ...

    @abstractmethod
    def get_domain_config(self, domain: str) -> DomainConfig:
        """
        Return tool classification config for a domain.

        Used by the uncertainty injection framework to know which tools
        are write actions, which have alternatives, and their noise priors.
        """
        ...

    @abstractmethod
    def get_tasks(
        self,
        domain: str,
        split: str = "base",
        task_ids: Optional[List[str]] = None,
        start: int = 0,
        end: int = -1,
    ) -> List[BenchmarkTask]:
        """
        Load tasks for a given domain and split.

        Args:
            domain: Domain name (from get_domains())
            split: Data split (e.g. "base", "test", "train")
            task_ids: If provided, load only these specific task IDs
            start: Start index for slicing
            end: End index for slicing (-1 = all)

        Returns:
            List of BenchmarkTask objects
        """
        ...

    @abstractmethod
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
        Run a single episode for one task.

        This is where the benchmark-specific orchestration happens:
        building the agent loop, running the simulation, collecting results.

        Args:
            task: The task to run
            agent_model_str: Provider-qualified model identifier (e.g. "<provider>/<model-name>")
            agent_model_kwargs: Extra kwargs for the model (api_base, thinking, etc.)
            user_model_str: User simulator model
            agent_name: Agent type identifier
            save_trajectory: Whether to record full chat history
            uncertainty_config: If provided, inject noise (EpisodeUncertaintyConfig)
            with_alternatives: Whether alternative tools stay clean

        Returns:
            EpisodeResult with all fields populated
        """
        ...

    def get_messages_for_judge(self, result: EpisodeResult) -> Optional[List]:
        """
        Extract message objects for the LLM judge.

        Override this if your benchmark stores messages in a format that
        the judge can use directly. Default implementation returns None,
        which means the judge will use result.chat_history instead.
        """
        return None

    def setup(self) -> None:
        """
        Optional one-time setup (import libraries, patch configs, etc.).
        Called once before any tasks are run.
        """
        pass

    def teardown(self) -> None:
        """Optional cleanup after all tasks are done."""
        pass
