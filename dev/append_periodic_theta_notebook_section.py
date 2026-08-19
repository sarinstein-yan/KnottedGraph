from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("User_guide/applications/02_mathematics_applications.ipynb")
MARKER_ID = "periodic-theta-closed-form-md"
INSERT_BEFORE_ID = "c8ea66b2"


def md(cell_id: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }


def code(cell_id: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


new_cells = [
    md(
        MARKER_ID,
        r"""## 12. Periodic theta family: corrected Yamada table and closed form

The periodic theta family is a useful example of how exact computation can expose and then verify a closed algebraic structure.  Here `build_periodic_theta_graph(s)` constructs the same family denoted by $\theta_{s,P}$: two pole vertices are joined through midpoint vertices $m_0,\ldots,m_{s-1}$, and the midpoint vertices are connected periodically,

$$
V(\theta_{s,P})=\{u,v,m_0,\ldots,m_{s-1}\},
$$

$$
E(\theta_{s,P})=
\{u m_i,\; v m_i,\; m_i m_{i+1}\}_{i\;\mathrm{mod}\;s}.
$$

Thus $|V|=s+2$ and $|E|=3s$.  For $s\ge 3$ this is the $s$-gonal bipyramid; the $s=2$ catalog case is the corresponding multigraph, with the two periodic midpoint edges parallel.

Write

$$
\sigma=Y+1+Y^{-1}.
$$

For this family the closed form is

$$
\boxed{
\Upsilon(\theta_{s,P};Y)
=
P_s(\sigma)
=
\frac{
(\sigma^2-\sigma+1)^s
+\sigma(2-\sigma)^s
+\sigma(-\sigma)^s
+\sigma^2-\sigma-1
}{\sigma+1}
},
\qquad s\ge 2.
$$

The denominator is removable: the numerator vanishes at $\sigma=-1$, so $P_s(\sigma)$ is an ordinary polynomial.  The cells below do **not** copy the earlier table. They regenerate the values from this formula, compare small cases directly with `KnottedGraph`, independently check the prism-dual chromatic formula with NetworkX/SymPy, and finally print a corrected table for $2\le s\le18$ together with copy-ready LaTeX rows.
""",
    ),
    code(
        "periodic-theta-formula-code",
        r'''from knotted_graph.applications.mathematical import build_periodic_theta_graph
from knotted_graph.invariants.yamada import (
    compute_yamada_polynomial_recursive,
    laurent_y_to_sigma_polynomial,
)

sigma_periodic = sp.Symbol("sigma")
q_periodic = sp.Symbol("q")


def periodic_theta_closed_form_sigma(s, sigma=sigma_periodic):
    """Exact P_s(sigma) for the periodic theta family."""
    if s < 2:
        raise ValueError("This table uses s >= 2.")

    numerator = sp.expand(
        (sigma**2 - sigma + 1) ** s
        + sigma * (2 - sigma) ** s
        + sigma * (-sigma) ** s
        + sigma**2
        - sigma
        - 1
    )

    quotient, remainder = sp.div(
        sp.Poly(numerator, sigma),
        sp.Poly(sigma + 1, sigma),
    )
    assert remainder.is_zero
    return quotient


for s in range(2, 19):
    poly = periodic_theta_closed_form_sigma(s)
    assert poly.degree() == 2 * s - 1
    assert poly.LC() == 1

print("closed form is polynomial, monic, and degree 2s-1 for s=2,...,18")
''',
    ),
    md(
        "periodic-theta-verification-md",
        r"""### 12.1 Verify the closed form independently

There are two complementary checks here.

First, we compute the crossing-free Yamada polynomial directly with the library for several nontrivial members of the family and convert the Laurent polynomial in $Y$ to a polynomial in $\sigma$. SymPy must reduce the difference from the proposed closed form to exactly zero.

Second, for $s\ge3$ the plane dual of the periodic theta graph is the prism graph $C_s\square K_2$.  With $q=\sigma+1$, the relevant prism chromatic polynomial is

$$
\chi_{C_s\square K_2}(q)
=
(q^2-3q+3)^s
+(q-1)(3-q)^s
+(q-1)(1-q)^s
+q^2-3q+1.
$$

For this family $|E|-|V|=2s-2$ is even, so the Yamada/Negami specialization agrees with the flow polynomial at $q=\sigma+1$; planar duality gives

$$
P_s(\sigma)
=
\frac{\chi_{C_s\square K_2}(\sigma+1)}{\sigma+1}.
$$

The code checks the chromatic expression independently against `networkx.chromatic_polynomial` for $s=3,4,5$, and then asks SymPy to simplify the dual expression to the boxed closed form.
""",
    ),
    code(
        "periodic-theta-verification-code",
        r'''# Direct KnottedGraph checks.  These are deliberately modest so this
# user-guide notebook remains practical to run from start to finish.
direct_periodic_checks = {}
for s in range(2, 7):
    graph = build_periodic_theta_graph(s)
    yamada_y = sp.expand(
        compute_yamada_polynomial_recursive(graph, Y)
    )
    yamada_sigma = sp.Poly(
        laurent_y_to_sigma_polynomial(
            yamada_y,
            Y,
            sigma_periodic,
        ).as_expr(),
        sigma_periodic,
    )
    closed_sigma = periodic_theta_closed_form_sigma(s)

    difference = sp.expand(
        yamada_sigma.as_expr() - closed_sigma.as_expr()
    )
    assert difference == 0, (s, difference)

    direct_periodic_checks[s] = yamada_sigma
    print(f"s={s:2d}: KnottedGraph == closed form  PASS")


def prism_chromatic_closed_form(s, q=q_periodic):
    return sp.expand(
        (q**2 - 3*q + 3) ** s
        + (q - 1) * (3 - q) ** s
        + (q - 1) * (1 - q) ** s
        + q**2
        - 3 * q
        + 1
    )


# Independent finite checks of the prism chromatic expression.
for s in (3, 4, 5):
    prism = nx.circular_ladder_graph(s)
    chromatic = nx.chromatic_polynomial(prism)
    chromatic_symbol = next(iter(chromatic.free_symbols))
    chromatic_q = sp.expand(
        chromatic.subs(chromatic_symbol, q_periodic)
    )
    expected_q = prism_chromatic_closed_form(s)
    assert sp.expand(chromatic_q - expected_q) == 0
    print(f"s={s:2d}: prism chromatic formula       PASS")


# Symbolically verify that planar-dual substitution gives the stated P_s.
for s in range(3, 19):
    from_dual = sp.cancel(
        prism_chromatic_closed_form(s, q_periodic).subs(
            q_periodic,
            sigma_periodic + 1,
        )
        / (sigma_periodic + 1)
    )
    difference = sp.expand(
        from_dual
        - periodic_theta_closed_form_sigma(s).as_expr()
    )
    assert difference == 0, (s, difference)

print("SymPy dual-form check passed for s=3,...,18")
''',
    ),
    md(
        "periodic-theta-table-md",
        r"""### 12.2 Corrected table for the manuscript

The table below is regenerated from the verified closed form.  In particular, the correct polynomial is monic and has degree $2s-1$.  This immediately diagnoses the previously generated high-degree entries as erroneous.
""",
    ),
    code(
        "periodic-theta-table-code",
        r'''from IPython.display import Markdown

periodic_theta_table = []
for s in range(2, 19):
    poly = periodic_theta_closed_form_sigma(s)
    periodic_theta_table.append(
        {
            "s": s,
            "degree": poly.degree(),
            "polynomial": sp.expand(poly.as_expr()),
        }
    )

periodic_table_lines = [
    "| $s$ | degree | corrected $\\Upsilon(\\theta_{s,P})=P_s(\\sigma)$ |",
    "| ---: | ---: | --- |",
]
for row in periodic_theta_table:
    periodic_table_lines.append(
        f"| {row['s']} | {row['degree']} | $"
        f"{sp.latex(row['polynomial'])}$ |"
    )

display(Markdown("\n".join(periodic_table_lines)))
''',
    ),
    code(
        "periodic-theta-latex-code",
        r'''print("Copy-ready LaTeX longtable rows:\n")
for row in periodic_theta_table:
    latex_poly = sp.latex(row["polynomial"])
    print(
        f"${row['s']}$ & "
        f"$\\displaystyle {latex_poly}$ \\\\" 
    )
''',
    ),
]


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    if any(cell.get("id") == MARKER_ID for cell in cells):
        print("periodic-theta section already present")
        return

    insert_at = next(
        index
        for index, cell in enumerate(cells)
        if cell.get("id") == INSERT_BEFORE_ID
    )
    notebook["cells"] = cells[:insert_at] + new_cells + cells[insert_at:]
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"inserted {len(new_cells)} cells before {INSERT_BEFORE_ID}")


if __name__ == "__main__":
    main()
