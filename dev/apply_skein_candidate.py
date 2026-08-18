"""Apply the experimental reduced-diagram skein solver on the analysis branch."""

from pathlib import Path


BUILDER_METHODS = r'''    def _with_crossing_removed_by_smoothing(self, crossing_index, pairs):
        crossing_ports = self.ordered_ports[crossing_index]
        removed_ports = set(crossing_ports)
        partner = list(self.arc_partner)

        for a, b in pairs:
            left_port = crossing_ports[a]
            right_port = crossing_ports[b]
            remote_left = partner[left_port]
            remote_right = partner[right_port]
            if remote_left in removed_ports or remote_right in removed_ports:
                raise ValueError("Degenerate self-adjacent crossing is not supported by fast smoothing.")
            if remote_left == remote_right:
                raise ValueError("Degenerate smoothing would identify one arc endpoint with itself.")
            partner[remote_left] = remote_right
            partner[remote_right] = remote_left

        active_ports = [
            port for port in range(len(self.arc_partner)) if port not in removed_ports
        ]
        old_to_new = {old: new for new, old in enumerate(active_ports)}
        surviving_crossings = [
            index
            for index in range(len(self.crossing_ids))
            if index != crossing_index
        ]
        crossing_remap = {
            old: new for new, old in enumerate(surviving_crossings)
        }

        new_arc_partner = tuple(old_to_new[partner[old]] for old in active_ports)
        new_fixed_terminal = tuple(
            self.fixed_terminal_index[old] for old in active_ports
        )
        new_crossing_for_port = []
        for old in active_ports:
            old_crossing = self.crossing_for_port[old]
            if old_crossing < 0:
                new_crossing_for_port.append(-1)
            elif old_crossing == crossing_index:
                raise RuntimeError("Smoothing retained a removed crossing port.")
            else:
                new_crossing_for_port.append(crossing_remap[old_crossing])

        new_ordered_ports = tuple(
            tuple(old_to_new[port] for port in self.ordered_ports[index])
            for index in surviving_crossings
        )
        plus_partner, minus_partner = self._resolution_tables(
            new_ordered_ports,
            len(active_ports),
        )
        return PreparedCompactStateBuilder(
            vertex_ids=self.vertex_ids,
            crossing_ids=tuple(self.crossing_ids[index] for index in surviving_crossings),
            ordered_ports=new_ordered_ports,
            arc_partner=new_arc_partner,
            fixed_terminal_index=new_fixed_terminal,
            crossing_for_port=tuple(new_crossing_for_port),
            plus_partner=plus_partner,
            minus_partner=minus_partner,
        )

    @staticmethod
    def _resolution_tables(ordered_ports, port_count):
        plus_partner = [-1] * port_count
        minus_partner = [-1] * port_count
        for ports in ordered_ports:
            for a, b in _PLUS_PAIRS:
                pa, pb = ports[a], ports[b]
                plus_partner[pa] = pb
                plus_partner[pb] = pa
            for a, b in _MINUS_PAIRS:
                pa, pb = ports[a], ports[b]
                minus_partner[pa] = pb
                minus_partner[pb] = pa
        return tuple(plus_partner), tuple(minus_partner)

    def resolve_crossing(self, crossing_index: int, spin: int):
        """Resolve one crossing while retaining all other crossings unresolved."""
        if crossing_index < 0 or crossing_index >= len(self.crossing_ids):
            raise IndexError(crossing_index)
        if spin == 0:
            return self._with_crossing_removed_by_smoothing(
                crossing_index,
                _PLUS_PAIRS,
            )
        if spin == 1:
            return self._with_crossing_removed_by_smoothing(
                crossing_index,
                _MINUS_PAIRS,
            )
        if spin != 2:
            raise ValueError("Invalid spin configuration.")

        crossing_ports = set(self.ordered_ports[crossing_index])
        surviving_crossings = [
            index
            for index in range(len(self.crossing_ids))
            if index != crossing_index
        ]
        crossing_remap = {
            old: new for new, old in enumerate(surviving_crossings)
        }

        new_vertex_index = len(self.vertex_ids)
        synthetic_id = max(
            (*self.vertex_ids, *self.crossing_ids),
            default=-1,
        ) + 1
        fixed_terminal = list(self.fixed_terminal_index)
        crossing_for_port = list(self.crossing_for_port)
        for port in crossing_ports:
            fixed_terminal[port] = new_vertex_index
            crossing_for_port[port] = -1
        for port, old_crossing in enumerate(self.crossing_for_port):
            if old_crossing < 0 or old_crossing == crossing_index:
                continue
            crossing_for_port[port] = crossing_remap[old_crossing]

        new_ordered_ports = tuple(
            self.ordered_ports[index] for index in surviving_crossings
        )
        plus_partner, minus_partner = self._resolution_tables(
            new_ordered_ports,
            len(self.arc_partner),
        )
        return PreparedCompactStateBuilder(
            vertex_ids=self.vertex_ids + (synthetic_id,),
            crossing_ids=tuple(self.crossing_ids[index] for index in surviving_crossings),
            ordered_ports=new_ordered_ports,
            arc_partner=self.arc_partner,
            fixed_terminal_index=tuple(fixed_terminal),
            crossing_for_port=tuple(crossing_for_port),
            plus_partner=plus_partner,
            minus_partner=minus_partner,
        )

    def invert_crossing(self, crossing_index: int):
        """Swap over/under information at one crossing without changing its arcs."""
        if crossing_index < 0 or crossing_index >= len(self.crossing_ids):
            raise IndexError(crossing_index)
        ordered_ports = list(self.ordered_ports)
        ports = ordered_ports[crossing_index]
        ordered_ports[crossing_index] = ports[1:] + ports[:1]
        ordered_ports_tuple = tuple(ordered_ports)
        plus_partner, minus_partner = self._resolution_tables(
            ordered_ports_tuple,
            len(self.arc_partner),
        )
        return PreparedCompactStateBuilder(
            vertex_ids=self.vertex_ids,
            crossing_ids=self.crossing_ids,
            ordered_ports=ordered_ports_tuple,
            arc_partner=self.arc_partner,
            fixed_terminal_index=self.fixed_terminal_index,
            crossing_for_port=self.crossing_for_port,
            plus_partner=plus_partner,
            minus_partner=minus_partner,
        )

    def exact_diagram_key(self):
        """Exact labeled key for memoizing internal partially resolved diagrams."""
        return (
            self.vertex_ids,
            self.crossing_ids,
            self.ordered_ports,
            self.arc_partner,
            self.fixed_terminal_index,
            self.crossing_for_port,
        )

'''


POLY_HELPERS = r'''
def _iter_prepared_compact_states(prepared):
    crossing_count = len(prepared.crossing_ids)
    for config in itertools.product([0, 1, 2], repeat=crossing_count):
        yield prepared.build(config), config.count(0) - config.count(1)


def _compute_prepared_bulk(prepared, evaluator):
    states = _iter_prepared_compact_states(prepared)
    if hasattr(evaluator, "compute_many_laurent"):
        return evaluator.compute_many_laurent(states)
    return _sum_laurent_states_raw(
        (
            _evaluate_fast_state(evaluator, graph, exponent)
            for graph, exponent in states
        )
    )


def _skein_delta(positive, negative):
    """Return (A-A^-1)*(positive-negative) as an exact Laurent tuple."""
    positive_part = laurent_add(
        laurent_shift(positive, 1),
        laurent_scale(laurent_shift(positive, -1), -1),
    )
    negative_part = laurent_add(
        laurent_scale(laurent_shift(negative, 1), -1),
        laurent_shift(negative, -1),
    )
    return laurent_add(positive_part, negative_part)


def _compute_prepared_with_skein_lookahead(prepared, evaluator, memo):
    """Evaluate a prepared diagram after exact RII reduction.

    The ordinary optimized bulk state sum remains the fallback. A recursive
    skein step is used only when inverting a crossing exposes at least one RII
    cancellation, guaranteeing that the inverted branch has strictly fewer
    crossings than the current diagram.
    """
    prepared, _ = prepared.reduce_reidemeister_ii()
    key = prepared.exact_diagram_key()
    cached = memo.get(key)
    if cached is not None:
        return cached

    crossing_count = len(prepared.crossing_ids)
    if crossing_count == 0:
        value = evaluator.compute_laurent(prepared.build(()))
        memo[key] = value
        return value

    for crossing_index in range(crossing_count):
        inverted = prepared.invert_crossing(crossing_index)
        inverted_reduced, moves = inverted.reduce_reidemeister_ii()
        if moves == 0:
            continue

        try:
            positive = prepared.resolve_crossing(crossing_index, 0)
            negative = prepared.resolve_crossing(crossing_index, 1)
        except ValueError:
            continue

        positive_value = _compute_prepared_with_skein_lookahead(
            positive,
            evaluator,
            memo,
        )
        negative_value = _compute_prepared_with_skein_lookahead(
            negative,
            evaluator,
            memo,
        )
        inverted_value = _compute_prepared_with_skein_lookahead(
            inverted_reduced,
            evaluator,
            memo,
        )
        value = laurent_add(
            _skein_delta(positive_value, negative_value),
            inverted_value,
        )
        memo[key] = value
        return value

    value = _compute_prepared_bulk(prepared, evaluator)
    memo[key] = value
    return value

'''


def patch_builder() -> None:
    path = Path("src/knotted_graph/invariants/yamada/state_compact.py")
    text = path.read_text()
    if "def exact_diagram_key(self):" in text:
        return
    marker = "    def build(self, state: tuple[int, ...]) -> CompactGraph:\n"
    assert text.count(marker) == 1
    path.write_text(text.replace(marker, BUILDER_METHODS + marker))


def patch_polynomial() -> None:
    path = Path("src/knotted_graph/invariants/yamada/polynomial.py")
    text = path.read_text()
    if "def _compute_prepared_with_skein_lookahead" in text:
        return

    old_import = "    shift as laurent_shift,\n    to_sympy as laurent_to_sympy,\n"
    new_import = (
        "    shift as laurent_shift,\n"
        "    scale as laurent_scale,\n"
        "    to_sympy as laurent_to_sympy,\n"
    )
    assert text.count(old_import) == 1
    text = text.replace(old_import, new_import)

    helper_marker = "\ndef compute_yamada_from_states(\n"
    assert text.count(helper_marker) == 1
    text = text.replace(helper_marker, "\n" + POLY_HELPERS + helper_marker)

    old_method = '''    def _compute_laurent_block(self, evaluator):\n        states = self._iter_compact_states()\n        if hasattr(evaluator, "compute_many_laurent"):\n            return evaluator.compute_many_laurent(states)\n        evaluated_states = (\n            _evaluate_fast_state(evaluator, graph, exponent)\n            for graph, exponent in states\n        )\n        return _sum_laurent_states_raw(evaluated_states)\n'''
    new_method = '''    def _compute_laurent_block(self, evaluator):\n        prepared = PreparedCompactStateBuilder.prepare(\n            self.vertices,\n            self.crossings,\n            self.arcs,\n            _ordered_crossing_ports,\n        )\n        return _compute_prepared_with_skein_lookahead(prepared, evaluator, {})\n'''
    assert text.count(old_method) == 1
    text = text.replace(old_method, new_method)
    path.write_text(text)


if __name__ == "__main__":
    patch_builder()
    patch_polynomial()
