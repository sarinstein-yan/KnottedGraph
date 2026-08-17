"""Compact exact multigraph kernels for Yamada evaluation.

The recursion in this module never mutates the caller's NetworkX graph and does
not use NetworkX or SymPy inside the hot loop.  A multigraph is represented by
a symmetric integer multiplicity matrix.  Loops live on the diagonal.
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

    def components(self) -> list[tuple[int, ...]]:
        n = self.n
        unseen = set(range(n))
        out: list[tuple[int, ...]] = []
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            stack = [start]
            component = [start]
            while stack:
                u = stack.pop()
                for v, count in enumerate(self.rows[u]):
                    if v != u and count and v in unseen:
                        unseen.remove(v)
                        stack.append(v)
                        component.append(v)
            out.append(tuple(sorted(component)))
        return out

    def induced(self, nodes: tuple[int, ...], *, keep_loop_at: int | None = None) -> "CompactGraph":
        nodes = tuple(nodes)
        rows = [[self.rows[i][j] for j in nodes] for i in nodes]
        if keep_loop_at is not None and keep_loop_at in nodes:
            # caller handles any desired loop removal after construction
            pass
        return CompactGraph(tuple(tuple(row) for row in rows))

    def has_bridge(self) -> bool:
        """Tarjan bridges on the underlying simple graph, respecting multiplicity."""
        n = self.n
        if n <= 1:
            return False
        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        tick = 0
        found = False

        def dfs(u: int):
            nonlocal tick, found
            disc[u] = low[u] = tick
            tick += 1
            for v in range(n):
                if v == u or self.rows[u][v] == 0:
                    continue
                if disc[v] == -1:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u] and self.rows[u][v] == 1:
                        found = True
                        return
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for root in range(n):
            if disc[root] == -1:
                dfs(root)
                if found:
                    return True
        return False

    def is_cycle(self) -> bool:
        if self.n == 0 or self.edge_count == 0:
            return False
        if len(self.components()) != 1:
            return False
        return all(self.degree(i) == 2 for i in range(self.n))

    def theta_count(self) -> int | None:
        if self.n != 2 or self.edge_count == 0:
            return None
        if self.rows[0][0] or self.rows[1][1]:
            return None
        count = self.rows[0][1]
        return count if count == self.edge_count else None

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
        m = [list(row) for row in self.rows]
        m[i][i] -= 1
        return CompactGraph(tuple(tuple(row) for row in m))

    def delete_edge(self, i: int, j: int) -> "CompactGraph":
        m = [list(row) for row in self.rows]
        m[i][j] -= 1
        m[j][i] -= 1
        return CompactGraph(tuple(tuple(row) for row in m))

    def contract_edge(self, i: int, j: int) -> "CompactGraph":
        """Contract one occurrence of non-loop edge (i,j), preserving multiplicity."""
        if i == j:
            raise ValueError("cannot contract a loop")
        if i > j:
            i, j = j, i
        if self.rows[i][j] <= 0:
            raise ValueError("edge does not exist")

        m = [list(row) for row in self.rows]
        m[i][j] -= 1
        m[j][i] -= 1

        # Remaining i-j parallel edges and loops based at j become loops at i.
        m[i][i] += m[j][j] + m[i][j]

        for k in range(self.n):
            if k in (i, j):
                continue
            m[i][k] += m[j][k]
            m[k][i] = m[i][k]

        # Remove vertex j.  The i-j cell is removed with that row/column.
        m.pop(j)
        for row in m:
            row.pop(j)
        return CompactGraph(tuple(tuple(row) for row in m))

    def articulation_parts(self) -> list["CompactGraph"] | None:
        """Return one-point-union factors using a small-graph exact search."""
        n = self.n
        if n < 3:
            return None

        for cut in range(n):
            remaining = tuple(i for i in range(n) if i != cut)
            if not remaining:
                continue

            unseen = set(remaining)
            components: list[tuple[int, ...]] = []
            while unseen:
                start = min(unseen)
                unseen.remove(start)
                stack = [start]
                comp = [start]
                while stack:
                    u = stack.pop()
                    for v in tuple(unseen):
                        if self.rows[u][v]:
                            unseen.remove(v)
                            stack.append(v)
                            comp.append(v)
                components.append(tuple(sorted(comp)))

            if len(components) < 2:
                continue

            parts = []
            for part_index, component in enumerate(components):
                nodes = tuple(sorted((*component, cut)))
                rows = [[self.rows[a][b] for b in nodes] for a in nodes]
                if part_index > 0:
                    local_cut = nodes.index(cut)
                    rows[local_cut][local_cut] = 0
                parts.append(CompactGraph(tuple(tuple(row) for row in rows)))
            return parts

        return None


class _CompactBase:
    def __init__(self):
        self.memo: dict[CompactGraph, Laurent] = {}

    def compute_laurent(self, graph: nx.MultiGraph | CompactGraph) -> Laurent:
        compact = graph if isinstance(graph, CompactGraph) else CompactGraph.from_networkx(graph)
        return self._rec(compact)

    def compute(self, graph: nx.MultiGraph | CompactGraph, variable: sp.Symbol) -> sp.Expr:
        return to_sympy(self.compute_laurent(graph), variable)


class CompactYamadaEvaluator(_CompactBase):
    def _rec(self, graph: CompactGraph) -> Laurent:
        cached = self.memo.get(graph)
        if cached is not None:
            return cached

        if graph.edge_count == 0:
            value = constant((-1) ** graph.n)
        else:
            components = graph.components()
            if len(components) > 1:
                value = ONE
                for component in components:
                    value = multiply(value, self._rec(graph.induced(component)))
            elif graph.has_bridge():
                value = ZERO
            elif graph.is_cycle():
                value = SIGMA
            else:
                theta = graph.theta_count()
                if theta is not None:
                    value = ZERO
                    power = ONE
                    for p in range(1, theta):
                        power = multiply_sigma(power)
                        value = add(value, scale(power, -1 if p % 2 == 0 else 1))
                else:
                    loop = graph.first_loop()
                    if loop is not None:
                        value = multiply_sigma(self._rec(graph.delete_loop(loop)), sign=-1)
                    else:
                        parts = graph.articulation_parts()
                        if parts is not None:
                            value = ONE
                            for part in parts:
                                value = multiply(value, self._rec(part))
                            if (len(parts) - 1) % 2:
                                value = scale(value, -1)
                        else:
                            edge = graph.first_nonloop()
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


class CompactNegamiSpecializedEvaluator(_CompactBase):
    def _rec(self, graph: CompactGraph) -> Laurent:
        cached = self.memo.get(graph)
        if cached is not None:
            return cached

        if graph.edge_count == 0:
            value = constant((-1) ** graph.n)
        else:
            components = graph.components()
            if len(components) > 1:
                value = ONE
                for component in components:
                    value = multiply(value, self._rec(graph.induced(component)))
            elif graph.has_bridge():
                value = ZERO
            else:
                loop = graph.first_loop()
                if loop is not None:
                    value = multiply_sigma(self._rec(graph.delete_loop(loop)), sign=-1)
                else:
                    parts = graph.articulation_parts()
                    if parts is not None:
                        value = ONE
                        for part in parts:
                            value = multiply(value, self._rec(part))
                        if (len(parts) - 1) % 2:
                            value = scale(value, -1)
                    else:
                        edge = graph.first_nonloop()
                        if edge is None:
                            value = constant((-1) ** graph.n)
                        else:
                            i, j = edge
                            value = add(
                                self._rec(graph.contract_edge(i, j)),
                                self._rec(graph.delete_edge(i, j)),
                            )

        self.memo[graph] = value
        return value
