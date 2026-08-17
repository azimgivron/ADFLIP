from __future__ import annotations

import bisect
import time
from typing import Any, Callable

from sortedcontainers import SortedList


class TimeSortedCache:
    """
    Cache that sorts elements by the time it took to process them.
    If it is out of space, it removes the cheapest element.
    """

    def __init__(self, capacity: int, parse_fn: Callable[[Any], Any]) -> None:
        """Initialize the TimeSortedCache.

        Args:
            capacity: Capacity value.
            parse_fn: Parse fn value.
        """
        self.capacity = capacity
        self.parse_fn = parse_fn
        self.times = []  # List to store processing times
        self.cache = {}  # Dictionary to store key-value pairs

    def get(self, key: Any) -> Any:
        """Execute the get operation.

        Args:
            key: Key value.

        Returns:
            Result of the get operation.
        """
        if key not in self.cache:
            return self._process_and_cache(key)

        return self.cache[key][1]

    def _process_and_cache(self, key: Any) -> Any:
        """Process and cache.

        Args:
            key: Key value.

        Returns:
            Result of the process and cache operation.
        """
        start_time = time.perf_counter()

        value = self.parse_fn(key)

        process_time = time.perf_counter() - start_time

        if self.capacity > -1 and (len(self.cache) >= self.capacity):
            if self.times[0] < process_time:
                # Remove the cheapest item
                cheapest_time = self.times.pop(0)
                cheapest_key = next(
                    k for k, v in self.cache.items() if v[0] == cheapest_time
                )
                del self.cache[cheapest_key]
                self._insert_item(key, value, process_time)

        else:
            self._insert_item(key, value, process_time)

        return value

    def __len__(self) -> int:
        """Return the number of contained items.

        Returns:
            Computed integer value.
        """
        return len(self.cache)

    def _insert_item(self, key: Any, value: Any, process_time: Any) -> None:
        """Execute the insert item operation.

        Args:
            key: Key value.
            value: Value value.
            process_time: Process time value.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a human-readable representation.

        Returns:
            Result of the str operation.
        """
        return f"TimeSortedCache(capacity={self.capacity}, items={len(self.cache)})"


class TimeSortedCacheBisect(TimeSortedCache):
    """Implement the time sorted cache bisect component."""

    def _insert_item(self, key: Any, value: Any, process_time: Any) -> None:
        """Execute the insert item operation.

        Args:
            key: Key value.
            value: Value value.
            process_time: Process time value.
        """
        bisect.insort(self.times, process_time)
        self.cache[key] = (process_time, value)


class TimeSortedCacheRBT(TimeSortedCache):
    """Implement the time sorted cache rbt component."""

    def __init__(self, capacity: int, parse_fn: Callable[[Any], Any]) -> None:
        """Initialize the TimeSortedCacheRBT.

        Args:
            capacity: Capacity value.
            parse_fn: Parse fn value.
        """
        super().__init__(capacity, parse_fn)
        self.times = SortedList()

    def _insert_item(self, key: Any, value: Any, process_time: Any) -> None:
        """Execute the insert item operation.

        Args:
            key: Key value.
            value: Value value.
            process_time: Process time value.
        """
        self.times.add(process_time)
        self.cache[key] = (process_time, value)
