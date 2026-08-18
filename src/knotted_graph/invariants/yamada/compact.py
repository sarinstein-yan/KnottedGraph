"""Compact exact multigraph kernels for Yamada evaluation.

The Python recurrence in this module never mutates the caller's NetworkX graph
and does not use NetworkX or SymPy inside the hot loop. Production constructors
select the compiled C++ recurrence when available and transparently retain this
arbitrary-precision Python implementation as the exact fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import sympy as sp

from .fast import (
    Laurent,
    ONE,
    SIGMA,
    ZERO,
    add,
    constant,
    multiply,
    multiply_sigma,
    scale,
    to_sympy,
)


@dataclass(frozen=True, slots=True)
class CompactGraph:
    rows: tuple[tuple[int, ...], ...]

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def edge_count(self) -> int:
        n = self.n
        return sum(self.rows[i][i] for i in range(n)) + sum(
            self.rows[i][j] for i in range(n) for j in range(i + 1, n)
        )

    @classmethod
    def from_networkx(cls, graph: nx.MultiGraph) -> "CompactGraph":
        nodes = sorted(graph.nodes(), key=repr)
        index = {node: i for i, node in enumerate(nodes)}
        matrix = [[0] * len(nodes) for _ in nodes]
        for u, v in graph.edges():
            i = index[u]
            j = index[v]
            matrix[i][j] += 1
            if i != j:
                matrix[j][i] += 1
        return cls(tuple(tuple(row) for row in matrix))

    def degree(self, i: int) -> int:
        row = self.rows[i]
        return 2 * row[i] + sum(row[j] for j in range(self.n) if j != i)

    def scan(self) -> tuple[int, tuple[int, ...], int | None, tuple[int, int] | None]:
        """Collect hot-loop scalar graph data in one multiplicity-matrix pass."""
        n = self.n
        edge_count = 0
        degrees = [0] * n
        first_loop = None
        first_edge = None
        for i in range(n):
            loop_count = self.rows[i][i]
            if loop_count:
                edge_count += loop_count
                degrees[i] += 2 * loop_count
                if first_loop is None:
                    first_loop = i
            for j in range(i + 1, n):
                count = self.rows[i][j]
                if not count:
                    continue
                edge_count += count
                degrees[i] += count
                degrees[j] += count
                if first_edge is None:
                    first_edge = (i, j)
        return edge_count, tuple(degrees), first_loop, first_edge

    def components(self) -> list[tuple[int, ...]]:
        n = self.n
        seen = bytearray(n)
        out: list[tuple[int, ...]] = []
        for start in range(n):
            if seen[start]:
                continue
            seen[start] = 1
            stack = [start]
            component = []
            while stack:
                u = stack.pop()
                component.append(u)
                row = self.rows[u]
                for v in range(n):
                    if v != u and row[v] and not seen[v]:
                        seen[v] = 1
                        stack.append(v)
            out.append(tuple(sorted(component)))
        return out

    def induced(
        self,
        nodes: tuple[int, ...],
        *,
        keep_loop_at: int | None = None,
    ) -> "CompactGraph":
        nodes = tuple(nodes)
        rows = [[self.rows[i][j] for j in nodes] for i in nodes]
        if keep_loop_at is not None and keep_loop_at in nodes:
            pass
        return CompactGraph(tuple(tuple(row) for row in rows))

    def bridge_and_articulation(self) -> tuple[bool, int | None]:
        """Find bridges and one articulation point in one Tarjan traversal."""
        n = self.n
        if n <= 1:
            return False, None
        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        tick = 0
        found_bridge = False
        articulation = None

        def dfs(u: int) -> None:
            nonlocal tick, found_bridge, articulation
            disc[u] = low[u] = tick
            tick += 1
            children = 0
            for v, count in enumerate(self.rows[u]):
                if v == u or not count:
                    continue
                if disc[v] == -1:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u] and count == 1:
                        found_bridge = True
                    if parent[u] == -1:
                        if children > 1 and articulation is None:
                            articulation = u
                    elif low[v] >= disc[u] and articulation is None:
                        articulation = u
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for root in range(n):
            if disc[root] == -1:
                dfs(root)
        return found_bridge, articulation

    def has_bridge(self) -> bool:
        """Tarjan bridges on the underlying simple graph, respecting multiplicity."""
        return self.bridge_and_articulation()[0]

    def is_cycle(self) -> bool:
        edge_count, degrees, _, _ = self.scan()
        if self.n == 0 or edge_count == 0:
            return False
        if len(self.components()) != 1:
            return False
        return all(degree == 2 for degree in degrees)

    def theta_count(self) -> int | None:
        if self.n != 2:
            return None
        edge_count, _, _, _ = self.scan()
        if edge_count == 0 or self.rows[0][0] or self.rows[1][1]:
            return None
        count = self.rows[0][1]
        return count if count == edge_count else None

    def first_loop(self) -> int | None:
        for i in range(self.n):
            if self.rows[i][i]:
                return i
        return None

    def first_nonloop(self) -> tuple[int, int] | None:
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.rows[i][j]:
                    return i, j
        return None

    def delete_loop(self, i: int) -> "CompactGraph":
        matrix = [list(row) for row in self.rows]
        matrix[i][i] -= 1
        return CompactGraph(tuple(tuple(row) for row in matrix))

    def delete_edge(self, i: int, j: int) -> "CompactGraph":
        matrix = [list(row) for row in self.rows]
        matrix[i][j] -= 1
        matrix[j][i] -= 1
        return CompactGraph(tuple(tuple(row) for row in matrix))

    def contract_edge(self, i: int, j: int) -> "CompactGraph":
        """Contract one occurrence of non-loop edge (i,j), preserving multiplicity."""
        if i == j:
            raise ValueError("cannot contract a loop")
        if i > j:
            i, j = j, i
        if self.rows[i][j] <= 0:
            raise ValueError("edge does not exist")

        matrix = [list(row) for row in self.rows]
        matrix[i][j] -= 1
        matrix[j][i] -= 1
        matrix[i][i] += matrix[j][j] + matrix[i][j]

        for k in range(self.n):
            if k in (i, j):
                continue
            matrix[i][k] += matrix[j][k]
            matrix[k][i] = matrix[i][k]

        matrix.pop(j)
        for row in matrix:
            row.pop(j)
        return CompactGraph(tuple(tuple(row) for row in matrix))

    def articulation_parts_at(self, cut: int) -> list["CompactGraph"] | None:
        """Return one-point-union factors for a known articulation point."""
        n = self.n
        if n < 3 or not 0 <= cut < n:
            return None
        remaining = [i for i in range(n) if i != cut]
        seen = bytearray(n)
        seen[cut] = 1
        components: list[tuple[int, ...]] = []
        for start in remaining:
            if seen[start]:
                continue
            seen[start] = 1
            stack = [start]
            component = []
            while stack:
                u = stack.pop()
                component.append(u)
                for v in remaining:
                    if not seen[v] and self.rows[u][v]:
                        seen[v] = 1
                        stack.append(v)
            components.append(tuple(sorted(component)))
        if len(components) < 2:
            return None

        parts = []
        for part_index, component in enumerate(components):
            nodes = tuple(sorted((*component, cut)))
            rows = [[self.rows[a][b] for b in nodes] for a in nodes]
            if part_index > 0:
                local_cut = nodes.index(cut)
                rows[local_cut][local_cut] = 0
            parts.append(CompactGraph(tuple(tuple(row) for row in rows)))
        return parts

    def articulation_parts(self) -> list["CompactGraph"] | None:
        """Return one-point-union factors using exact articulation detection."""
        _, cut = self.bridge_and_articulation()
        if cut is None:
            return None
        return self.articulation_parts_at(cut)


def _theta_value(theta: int) -> Laurent:
    """Exact Yamada value of a crossing-free theta multigraph."""
    value = ZERO
    power = ONE
    for p in range(1, theta):
        power = multiply_sigma(power)
        value = add(value, scale(power, -1 if p % 2 == 0 else 1))
    return value


class _CompactBase:
    def __init__(self):
        self.memo: dict[CompactGraph, Laurent] = {}

    def compute_laurent(self, graph: nx.MultiGraph | CompactGraph) -> Laurent:
        compact = graph if isinstance(graph, CompactGraph) else CompactGraph.from_networkx(graph)
        return self._rec(compact)

    def compute(self, graph: nx.MultiGraph | CompactGraph, variable: sp.Symbol) -> sp.Expr:
        return to_sympy(self.compute_laurent(graph), variable)

    def _rec(self, graph: CompactGraph) -> Laurent:
        cached = self.memo.get(graph)
        if cached is not None:
            return cached

        edge_count, degrees, loop, edge = graph.scan()
        if edge_count == 0:
            value = constant((-1) ** graph.n)
            self.memo[graph] = value
            return value

        components = graph.components()
        if len(components) > 1:
            value = ONE
            for component in components:
                value = multiply(value, self._rec(graph.induced(component)))
            self.memo[graph] = value
            return value

        if graph.n == 2 and not graph.rows[0][0] and not graph.rows[1][1]:
            theta = graph.rows[0][1]
            if theta == edge_count:
                value = _theta_value(theta)
                self.memo[graph] = value
                return value

        if graph.n and all(degree == 2 for degree in degrees):
            self.memo[graph] = SIGMA
            return SIGMA

        has_bridge, articulation = graph.bridge_and_articulation()
        if has_bridge:
            self.memo[graph] = ZERO
            return ZERO

        if loop is not None:
            value = multiply_sigma(self._rec(graph.delete_loop(loop)), sign=-1)
            self.memo[graph] = value
            return value

        if articulation is not None:
            parts = graph.articulation_parts_at(articulation)
            if parts is not None:
                value = ONE
                for part in parts:
                    value = multiply(value, self._rec(part))
                if (len(parts) - 1) % 2:
                    value = scale(value, -1)
                self.memo[graph] = value
                return value

        if edge is None:
            value = constant((-1) ** graph.n)
        else:
            i, j = edge
            value = add(
                self._rec(graph.delete_edge(i, j)),
                self._rec(graph.contract_edge(i, j)),
            )
        self.memo[graph] = value
        return value


class PythonCompactYamadaEvaluator(_CompactBase):
    """Explicit pure-Python exact evaluator, retained for testing/fallback."""


class PythonCompactNegamiSpecializedEvaluator(_CompactBase):
    """Explicit pure-Python specialized Negami evaluator for testing/fallback."""


class CompactYamadaEvaluator:
    """Fastest available exact direct-Yamada evaluator."""

    def __new__(cls):
        from .native import make_native_or_python_evaluator

        return make_native_or_python_evaluator(PythonCompactYamadaEvaluator)


class CompactNegamiSpecializedEvaluator:
    """Fastest available exact specialized-Negami evaluator."""

    def __new__(cls):
        from .native import make_native_or_python_evaluator

        return make_native_or_python_evaluator(PythonCompactNegamiSpecializedEvaluator)
