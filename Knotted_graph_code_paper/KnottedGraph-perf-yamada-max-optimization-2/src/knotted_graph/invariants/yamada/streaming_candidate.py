"""Experimental bounded-memory wrapper for native Yamada state batches.

This module is deliberately not selected by production dispatch. It lets tests
and benchmarks evaluate the exact same native recurrence/memo while consuming
resolved crossing states in bounded chunks instead of materializing the entire
3^c stream as dense Python matrices at once.
"""

from __future__ import annotations

import itertools

from .fast import add, shift


class ChunkedEvaluatorProxy:
    """Preserve one evaluator/memo while bounding ``compute_many`` materialization."""

    def __init__(self, evaluator, *, chunk_size: int = 256):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.evaluator = evaluator
        self.chunk_size = int(chunk_size)

    @property
    def memo(self):
        return self.evaluator.memo

    @property
    def memo_size(self):
        return getattr(self.evaluator, "memo_size", len(self.evaluator.memo))

    def compute_laurent(self, graph):
        return self.evaluator.compute_laurent(graph)

    def compute_many_laurent(self, states):
        iterator = iter(states)
        total = ()
        while True:
            chunk = list(itertools.islice(iterator, self.chunk_size))
            if not chunk:
                return total
            if hasattr(self.evaluator, "compute_many_laurent"):
                subtotal = self.evaluator.compute_many_laurent(chunk)
            else:
                subtotal = ()
                for graph, exponent in chunk:
                    subtotal = add(
                        subtotal,
                        shift(
                            self.evaluator.compute_laurent(graph),
                            int(exponent),
                        ),
                    )
            total = add(total, subtotal)
