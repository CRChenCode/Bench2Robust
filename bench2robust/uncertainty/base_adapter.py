# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Base adapter interface for benchmark integration.

This module defines the abstract interface that benchmark-specific adapters
must implement to integrate with the uncertainty simulation framework.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from bench2robust.uncertainty.config import EpisodeUncertaintyConfig, ToolConfig


class BaseAdapter(ABC):
    """
    Abstract base class for benchmark adapters.
    
    Adapters connect the core framework to specific benchmarks by implementing
    benchmark-specific tool wrapping, response extraction, and noise injection.
    """
    
    @abstractmethod
    def get_tool_list(self) -> List[str]:
        """
        Get list of available tools in the benchmark.
        
        Returns:
            List of tool names
        """
        pass
    
    @abstractmethod
    def wrap_tool(self, tool: Any, config: ToolConfig) -> Any:
        """
        Wrap a benchmark tool with uncertainty injection.
        
        Args:
            tool: Original tool object
            config: Tool configuration
        
        Returns:
            Wrapped tool
        """
        pass
    
    @abstractmethod
    def wrap_environment(self, env: Any, config: EpisodeUncertaintyConfig) -> Any:
        """
        Wrap a benchmark environment with uncertainty injection.
        
        Args:
            env: Original environment
            config: Episode configuration
        
        Returns:
            Wrapped environment
        """
        pass
    
    @abstractmethod
    def extract_tool_response(self, result: Any) -> Any:
        """
        Extract tool response from benchmark-specific result format.
        
        Args:
            result: Tool execution result
        
        Returns:
            Extracted response
        """
        pass
    
    @abstractmethod
    def format_error_message(self, error_type: str) -> str:
        """
        Format error message in benchmark-specific format.
        
        Args:
            error_type: Type of error (e.g., "timeout")
        
        Returns:
            Formatted error message
        """
        pass
