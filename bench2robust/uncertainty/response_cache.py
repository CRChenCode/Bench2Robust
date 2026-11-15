# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Response caching system for stale response injection.

This module provides a simple cache for storing the first response from each tool
to enable realistic stale data injection.
"""

from typing import Any, Optional, Dict
from copy import deepcopy


class ResponseCache:
    """
    Simple cache for storing first response from each tool.
    Used for stale response injection.
    """
    
    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, Any] = {}
    
    def cache_response(self, tool_name: str, response: Any) -> None:
        """
        Cache the first response from a tool.
        
        Args:
            tool_name: Name of the tool
            response: Response to cache (will be deep copied)
        """
        if tool_name not in self._cache:
            # Deep copy to prevent mutations
            self._cache[tool_name] = deepcopy(response)
    
    def get_cached_response(self, tool_name: str) -> Optional[Any]:
        """
        Retrieve cached response for a tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Cached response (deep copy) or None if not cached
        """
        if tool_name in self._cache:
            return deepcopy(self._cache[tool_name])
        return None
    
    def has_cached_response(self, tool_name: str) -> bool:
        """Check if tool has a cached response."""
        return tool_name in self._cache
    
    def clear(self) -> None:
        """Clear all cached responses (called on episode reset)."""
        self._cache.clear()
    
    def __len__(self) -> int:
        """Return number of cached responses."""
        return len(self._cache)
