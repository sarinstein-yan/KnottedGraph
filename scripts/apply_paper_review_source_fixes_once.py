#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} block, found {count}. "
            "The source may have changed since validation."
        )
    return text.replace(old, new, 1)


def patch_skeleton() -> None:
    path = ROOT / "src/knotted_graph/applications/nodal/skeleton.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """        G = skeleton_image_to_graph(self._skeleton_image) \\
            if skeleton_image is None else skeleton_image
""",
        """        image = (
            self._skeleton_image
            if skeleton_image is None
            else np.asarray(skeleton_image, dtype=bool)
        )
        if image.ndim != 3:
            raise ValueError(
                "skeleton_image must be a three-dimensional array."
            )
        G = skeleton_image_to_graph(image)
""",
        "external skeleton-image conversion",
    )

    text = replace_once(
        text,
        """        berry = self.berry_curvature.copy()
        berry = berry.reshape(-1, 3, order='F')
        berry_norm = np.linalg.norm(berry, axis=-1)
        vol.point_data['berry'] = berry
        vol.point_data['|berry|'] = berry_norm
        vol.point_data['log10(|berry|+1)'] = np.log10(berry_norm + 1)
""",
        """        if self._berry_prerequisites['valid']:
            berry = self.berry_curvature.copy()
            berry = berry.reshape(-1, 3, order='F')
            berry_norm = np.linalg.norm(berry, axis=-1)
            vol.point_data['berry'] = berry
            vol.point_data['|berry|'] = berry_norm
            vol.point_data['log10(|berry|+1)'] = np.log10(berry_norm + 1)
""",
        "optional Berry-field block",
    )

    path.write_text(text, encoding="utf-8")


def patch_pd_code() -> None:
    path = ROOT / "src/knotted_graph/projection/pd_code.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """            for idx in tree.query(seg):
                other_seg = tree.geometries.take(idx)
                # Skip segments that are visited before or connect at endpoints
                if segments.index(other_seg) <= i or seg.touches(other_seg):
                    continue
""",
        """            for idx in tree.query(seg):
                idx = int(idx)
                other_seg = tree.geometries.take(idx)
                # STRtree already returns the segment index. Reusing it avoids
                # a linear ``segments.index(...)`` search for every candidate.
                if idx <= i or seg.touches(other_seg):
                    continue
""",
        "STRtree crossing-search block",
    )

    path.write_text(text, encoding="utf-8")


def patch_polynomial() -> None:
    path = ROOT / "src/knotted_graph/invariants/yamada/polynomial.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """    Y = sp.expand(sp.cancel(total_poly))

    if normalize:
        terms = Y.as_ordered_terms()
        lowest_exp = min(t.as_coeff_exponent(A)[1] for t in terms)
        Y = Y * (-A) ** (-lowest_exp)
        Y = sp.expand(sp.cancel(Y))

    return Y


def _angle_delta(a: float, b: float) -> float:
""",
        """    return _finalize_yamada_total(
        total_poly,
        A,
        normalize=normalize,
    )


def _finalize_yamada_total(
    total_poly: sp.Expr,
    A: sp.Symbol,
    *,
    normalize: bool,
) -> sp.Expr:
    Y = sp.expand(sp.cancel(total_poly))
    if normalize:
        terms = Y.as_ordered_terms()
        lowest_exp = min(term.as_coeff_exponent(A)[1] for term in terms)
        Y = Y * (-A) ** (-lowest_exp)
        Y = sp.expand(sp.cancel(Y))
    return Y


def _evaluate_state_with_exponent(evaluator, graph: nx.MultiGraph, exponent: int):
    return exponent, evaluator.compute(graph)


def _angle_delta(a: float, b: float) -> float:
""",
        "Yamada finalization helper block",
    )

    text = replace_once(
        text,
        '''    def _build_state_graphs(self):
        num_x = len(self.crossings)
        configurations = list(itertools.product([0, 1, 2], repeat=num_x))
        exponents = [s.count(0) - s.count(1) for s in configurations]
        state_graphs = [
            _build_state_graph_from_ports(
                self.vertices,
                self.crossings,
                self.arcs,
                config,
            )
            for config in configurations
        ]
        return state_graphs, exponents

    def compute(
        self,
        variable: sp.Symbol,
        normalize: bool = True,
        n_jobs: int = -1,
        method: str = "negami",
    ) -> sp.Expr:
        """Compute the Yamada polynomial for the diagram."""

        state_graphs, exponents = self._build_state_graphs()
        return compute_yamada_from_states(
            state_graphs,
            exponents,
            variable,
            normalize=normalize,
            n_jobs=n_jobs,
            method=method,
        )
''',
        '''    def _iter_state_graphs(self):
        for config in itertools.product([0, 1, 2], repeat=len(self.crossings)):
            yield (
                _build_state_graph_from_ports(
                    self.vertices,
                    self.crossings,
                    self.arcs,
                    config,
                ),
                config.count(0) - config.count(1),
            )

    def _build_state_graphs(self):
        """Materialize all states for diagnostic/backward-compatible callers."""
        states = list(self._iter_state_graphs())
        return (
            [graph for graph, _ in states],
            [exponent for _, exponent in states],
        )

    def compute(
        self,
        variable: sp.Symbol,
        normalize: bool = True,
        n_jobs: int = -1,
        method: str = "negami",
    ) -> sp.Expr:
        """Compute the Yamada polynomial without materializing all state graphs."""
        if method not in {"negami", "recursive"}:
            raise ValueError("method must be either 'negami' or 'recursive'.")

        if method == "negami":
            x, y = sp.symbols("x y")
            evaluator = NegamiRecursiveEvaluator(x, y)
        else:
            evaluator = YamadaRecursiveEvaluator(variable)

        evaluated_states = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
        )(
            delayed(_evaluate_state_with_exponent)(
                evaluator,
                graph,
                exponent,
            )
            for graph, exponent in self._iter_state_graphs()
        )

        total_poly = sp.Integer(0)
        for exponent, state_value in evaluated_states:
            if method == "negami":
                state_value = sp.expand(
                    state_value.xreplace(
                        {
                            x: -1,
                            y: -variable - 2 - variable ** (-1),
                        }
                    )
                )
            total_poly += (variable**exponent) * state_value

        return _finalize_yamada_total(
            total_poly,
            variable,
            normalize=normalize,
        )
''',
        "Yamada eager-state block",
    )

    path.write_text(text, encoding="utf-8")


def patch_mathematics_notebook_and_move_csv() -> None:
    old_csv = ROOT / "User_guide/applications/structured_graph_yamada_dataset.csv"
    new_csv = ROOT / "doc/assets/data/structured_graph_yamada_dataset.csv"
    new_csv.parent.mkdir(parents=True, exist_ok=True)

    if old_csv.exists():
        if new_csv.exists():
            if old_csv.read_bytes() != new_csv.read_bytes():
                raise RuntimeError(
                    "Old and new structured Yamada datasets both exist "
                    "with different contents."
                )
            old_csv.unlink()
        else:
            old_csv.replace(new_csv)
    elif not new_csv.exists():
        raise RuntimeError("Structured Yamada dataset was not found.")

    path = ROOT / "User_guide/applications/02_mathematics_applications.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    dataset_cells = 0

    locator_pattern = re.compile(
        r"def find_local_structured_dataset\(\):\n"
        r".*?"
        r"dataset_path = \(\n"
        r"    find_local_structured_dataset\(\)\n"
        r"\)",
        flags=re.DOTALL,
    )

    locator_replacement = '''dataset_path = (
    PROJECT_ROOT
    / "doc"
    / "assets"
    / "data"
    / "structured_graph_yamada_dataset.csv"
).resolve()

if not dataset_path.exists():
    raise FileNotFoundError(
        "Could not find the supplied structured Yamada dataset at "
        f"{dataset_path}."
    )'''

    for cell in notebook.get("cells", []):
        source_obj = cell.get("source", [])
        source = "".join(source_obj) if isinstance(source_obj, list) else str(source_obj)
        original = source

        if cell.get("cell_type") == "markdown":
            source = source.replace(
                "`structured_graph_yamada_dataset.csv` in the **same folder as this notebook**",
                "`doc/assets/data/structured_graph_yamada_dataset.csv` at the repository root",
            )
            source = source.replace(
                "The CSV is distributed in the **same `applications/` folder** as this notebook.",
                "The CSV is distributed under **`doc/assets/data/`** at the repository root.",
            )
            source = source.replace(
                "not overwrite `structured_graph_yamada_dataset.csv`.",
                "not overwrite `doc/assets/data/structured_graph_yamada_dataset.csv`.",
            )

        if cell.get("cell_type") == "code" and "def find_local_structured_dataset" in source:
            source, count = locator_pattern.subn(locator_replacement, source, count=1)
            if count != 1:
                raise RuntimeError(
                    "Dataset locator cell no longer matches the validated source."
                )
            dataset_cells += 1
            cell["outputs"] = []
            cell["execution_count"] = None

        if source != original:
            cell["source"] = (
                source.splitlines(keepends=True)
                if isinstance(source_obj, list)
                else source
            )

    if dataset_cells != 1:
        raise RuntimeError(
            f"Expected exactly one dataset locator cell, found {dataset_cells}."
        )

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_no_trivalent_change() -> None:
    path = ROOT / "src/knotted_graph/core/graphs.py"
    text = path.read_text(encoding="utf-8")
    if "return all(degree <= 3 for node, degree in degs)" not in text:
        raise RuntimeError("is_trivalent() convention changed unexpectedly.")


def validate_python_syntax() -> None:
    for rel in (
        "src/knotted_graph/invariants/yamada/polynomial.py",
        "src/knotted_graph/projection/pd_code.py",
        "src/knotted_graph/applications/nodal/skeleton.py",
    ):
        ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def main() -> None:
    validate_no_trivalent_change()
    patch_skeleton()
    patch_pd_code()
    patch_polynomial()
    patch_mathematics_notebook_and_move_csv()
    validate_no_trivalent_change()
    validate_python_syntax()
    print("Validated paper-review source fixes applied.")


if __name__ == "__main__":
    main()
