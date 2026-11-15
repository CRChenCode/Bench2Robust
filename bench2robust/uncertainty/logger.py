# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Episode logging system for comprehensive execution tracking.

This module provides logging infrastructure for recording all tool executions
with both clean and observed responses.
"""

from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Optional
import json
import time
from pathlib import Path


@dataclass
class ToolExecutionLog:
    """Log entry for a single tool execution."""
    episode_id: str
    step: int
    tool_name: str
    tool_input: Dict
    observed_response: Any  # What agent saw
    clean_response: Any     # What it should have been
    noise_type: str         # NoiseType value or "natural_error"
    timestamp: float
    
    # Metadata
    is_write_action: bool
    has_alternatives: bool
    consecutive_failures: int  # At time of this call
    total_injections_so_far: int


@dataclass
class EpisodeLog:
    """Complete log for an episode."""
    episode_id: str
    task_index: int
    domain: str  # "retail" | "airline"
    config: Dict  # Serialized EpisodeUncertaintyConfig
    tool_executions: List[ToolExecutionLog]
    final_reward: float
    task_solved: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "task_index": self.task_index,
            "domain": self.domain,
            "config": self.config,
            "tool_executions": [asdict(log) for log in self.tool_executions],
            "final_reward": self.final_reward,
            "task_solved": self.task_solved
        }


class EpisodeLogger:
    """Logs tool executions and episode outcomes."""
    
    def __init__(self, episode_id: str, task_index: int, domain: str, config: Dict):
        """
        Initialize logger for an episode.
        
        Args:
            episode_id: Unique episode identifier
            task_index: Task index in benchmark
            domain: Domain name (e.g., "retail", "airline")
            config: Serialized configuration
        """
        self.episode_id = episode_id
        self.task_index = task_index
        self.domain = domain
        self.config = config
        self.tool_executions: List[ToolExecutionLog] = []
        self.final_reward: Optional[float] = None
        self.task_solved: Optional[bool] = None
    
    def log_tool_execution(
        self,
        step: int,
        tool_name: str,
        tool_input: Dict,
        observed_response: Any,
        clean_response: Any,
        noise_type: str,
        is_write_action: bool,
        has_alternatives: bool,
        consecutive_failures: int,
        total_injections_so_far: int
    ):
        """
        Log a tool execution.
        
        Args:
            step: Step number
            tool_name: Name of the tool
            tool_input: Input to the tool
            observed_response: Response seen by agent
            clean_response: Clean response (what should have been)
            noise_type: Type of noise injected
            is_write_action: Whether tool is a write action
            has_alternatives: Whether tool has alternatives
            consecutive_failures: Consecutive failures at time of call
            total_injections_so_far: Total injections so far
        """
        log_entry = ToolExecutionLog(
            episode_id=self.episode_id,
            step=step,
            tool_name=tool_name,
            tool_input=tool_input,
            observed_response=observed_response,
            clean_response=clean_response,
            noise_type=noise_type,
            timestamp=time.time(),
            is_write_action=is_write_action,
            has_alternatives=has_alternatives,
            consecutive_failures=consecutive_failures,
            total_injections_so_far=total_injections_so_far
        )
        self.tool_executions.append(log_entry)
    
    def finalize(self, final_reward: float, task_solved: bool) -> EpisodeLog:
        """
        Finalize episode log.
        
        Args:
            final_reward: Final reward received
            task_solved: Whether task was solved
        
        Returns:
            Complete EpisodeLog
        """
        self.final_reward = final_reward
        self.task_solved = task_solved
        
        return EpisodeLog(
            episode_id=self.episode_id,
            task_index=self.task_index,
            domain=self.domain,
            config=self.config,
            tool_executions=self.tool_executions,
            final_reward=final_reward,
            task_solved=task_solved
        )
    
    def get_current_log(self) -> EpisodeLog:
        """Get current state of episode log (before finalization)."""
        return EpisodeLog(
            episode_id=self.episode_id,
            task_index=self.task_index,
            domain=self.domain,
            config=self.config,
            tool_executions=self.tool_executions,
            final_reward=self.final_reward or 0.0,
            task_solved=self.task_solved or False
        )


class LogExporter:
    """Exports episode logs to files."""
    
    @staticmethod
    def export_to_jsonl(logs: List[EpisodeLog], output_path: Path):
        """
        Export logs to JSON Lines format.
        
        Args:
            logs: List of episode logs
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for log in logs:
                json_line = json.dumps(log.to_dict(), default=str)
                f.write(json_line + '\n')
    
    @staticmethod
    def export_to_json(logs: List[EpisodeLog], output_path: Path):
        """
        Export logs to JSON format.
        
        Args:
            logs: List of episode logs
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([log.to_dict() for log in logs], f, indent=2, default=str)
