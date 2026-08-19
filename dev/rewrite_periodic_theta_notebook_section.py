from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("User_guide/applications/02_mathematics_applications.ipynb")


def lines(text: str) -> list[str]:
    out = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        out[-1] += "\n"
    return out


REPLACEMENTS = {
    "periodic-theta-closed-form-md": r"""## 12. Periodic theta graphs: computing and verifying a Yamada closed form

This example shows how `KnottedGraph` can be used on a parameterized graph family to move from **exact computation** to a **symbolic formula** and then verify that formula independently.

`build_periodic_theta_graph(s)` constructs the periodic theta graph $\theta_{s,P}$ from two pole vertices and $s$ midpoint vertices $m_0,\ldots,m_{s-1}$. Each midpoint is connected to both poles, and the midpoint vertices are connected cyclically:

$$
V(\theta_{s,P})=\{u,v,m_0,\ldots,m_{s-1}\},
$$

$$
E(\theta_{s,P})=
\{u m_i,\;v m_i,\;m_i m_{i+1}\}_{i\;\mathrm{mod}\;s}.
$$

Hence

$$
|V|=s+2,
\qquad
|E|=3s.
$$

Introduce the standard Yamada combination

$$
\sigma=Y+1+Y^{-1}.
$$

The predicted Yamada polynomial for the family is

$$
\boxed{
\Upsilon(\theta_{s,P};Y)
=
\frac{
(\sigma^2-\sigma+1)^s
+\sigma(2-\sigma)^s
+\sigma(-\sigma)^s
+\sigma^2-\sigma-1
}{\sigma+1}
},
\qquad s\ge2.
$$

Although this expression is written as a quotient, the numerator is divisible by $\sigma+1$, so the result is an ordinary polynomial in $\sigma$. The cells below verify the formula directly with `KnottedGraph`, check an independent planar-dual route with NetworkX/SymPy, and then generate a table of exact Yamada polynomials for $2\le s\le18$.
""",
    "periodic-theta-formula-code": r'''from knotted_graph.applications.mathematical import build_periodic_theta_graph
from knotted_graph.invariants.yamada import (
    compute_yamada_polynomial_recursive,
    laurent_y_to_sigma_polynomial,
)

sigma_periodic = sp.Symbol("sigma")
q_periodic = sp.Symbol("q")


def predicted_periodic_theta_yamada_sigma(s, sigma=sigma_periodic):
    """Predicted Yamada polynomial Upsilon(theta_{s,P}; Y), written in sigma."""
    if s < 2:
        raise ValueError("This family formula is used for s >= 2.")

    # Explicit prediction:
    # Upsilon(theta_{s,P};Y) =
    # [
    #   (sigma^2-sigma+1)^s
    #   + sigma(2-sigma)^s
    #   + sigma(-sigma)^s
    #   + sigma^2-sigma-1
    # ] / (sigma+1),
    # with sigma = Y + 1 + Y^(-1).
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
    predicted = predicted_periodic_theta_yamada_sigma(s)
    assert predicted.degree() == 2 * s - 1
    assert predicted.LC() == 1

print("Predicted formula is polynomial, monic, and degree 2s-1 for s=2,...,18")
''',
    "periodic-theta-verification-md": r"""### 12.1 Verify the prediction

We use two checks that start from different descriptions of the same family.

**Direct library check.** For several values of $s$, `KnottedGraph` computes $\Upsilon(\theta_{s,P};Y)$ directly from the graph. We then rewrite the result in $\sigma=Y+1+Y^{-1}$ and ask SymPy to verify

$$
\Upsilon_{\mathrm{KnottedGraph}}(\theta_{s,P};Y)
-
\frac{
(\sigma^2-\sigma+1)^s
+\sigma(2-\sigma)^s
+\sigma(-\sigma)^s
+\sigma^2-\sigma-1
}{\sigma+1}
=0.
$$

**Independent planar-dual check.** For $s\ge3$, the planar dual of $\theta_{s,P}$ is the ordinary $s$-prism graph. Its chromatic polynomial can be written explicitly as

$$
\chi_s(q)
=
(q^2-3q+3)^s
+(q-1)(3-q)^s
+(q-1)(1-q)^s
+q^2-3q+1.
$$

NetworkX is used to check this chromatic-polynomial expression for small $s$. Substituting $q=\sigma+1$ into the planar duality relation then gives exactly the same explicit Yamada prediction above. This provides a verification route that is separate from the direct Yamada recursion.
""",
    "periodic-theta-verification-code": r'''# 1) Direct KnottedGraph computation versus the explicit prediction.
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

    predicted_sigma = predicted_periodic_theta_yamada_sigma(s)
    difference = sp.expand(
        yamada_sigma.as_expr() - predicted_sigma.as_expr()
    )
    assert difference == 0, (s, difference)

    direct_periodic_checks[s] = yamada_sigma
    print(f"s={s:2d}: KnottedGraph == explicit Yamada prediction  PASS")


# 2) Independent chromatic-polynomial check for the planar dual prism.
def prism_chromatic_polynomial_formula(s, q=q_periodic):
    return sp.expand(
        (q**2 - 3*q + 3) ** s
        + (q - 1) * (3 - q) ** s
        + (q - 1) * (1 - q) ** s
        + q**2
        - 3 * q
        + 1
    )


for s in (3, 4, 5):
    prism = nx.circular_ladder_graph(s)
    chromatic = nx.chromatic_polynomial(prism)
    chromatic_symbol = next(iter(chromatic.free_symbols))
    chromatic_q = sp.expand(
        chromatic.subs(chromatic_symbol, q_periodic)
    )
    expected_q = prism_chromatic_polynomial_formula(s)
    assert sp.expand(chromatic_q - expected_q) == 0
    print(f"s={s:2d}: independent prism chromatic formula          PASS")


# 3) SymPy verifies that the planar-dual expression reduces to the same
#    explicit predicted Yamada polynomial for the full displayed range.
for s in range(3, 19):
    dual_yamada_sigma = sp.cancel(
        prism_chromatic_polynomial_formula(s, q_periodic).subs(
            q_periodic,
            sigma_periodic + 1,
        )
        / (sigma_periodic + 1)
    )

    explicit_prediction = predicted_periodic_theta_yamada_sigma(s).as_expr()
    difference = sp.expand(dual_yamada_sigma - explicit_prediction)
    assert difference == 0, (s, difference)

print("SymPy independent-form check passed for s=3,...,18")
''',
    "periodic-theta-table-md": r"""### 12.2 Exact Yamada values for the family

Once the formula has been verified, it can be used to generate exact values efficiently for larger $s$. The table below lists

$$
\Upsilon(\theta_{s,P};Y)
=
\frac{
(\sigma^2-\sigma+1)^s
+\sigma(2-\sigma)^s
+\sigma(-\sigma)^s
+\sigma^2-\sigma-1
}{\sigma+1}
$$

as an expanded polynomial in $\sigma$ for $2\le s\le18$.
""",
    "periodic-theta-table-code": r'''from IPython.display import Markdown

periodic_theta_table = []
for s in range(2, 19):
    yamada_sigma = predicted_periodic_theta_yamada_sigma(s)
    periodic_theta_table.append(
        {
            "s": s,
            "degree": yamada_sigma.degree(),
            "polynomial": sp.expand(yamada_sigma.as_expr()),
        }
    )

periodic_table_lines = [
    "| $s$ | degree | $\\Upsilon(\\theta_{s,P};Y)$ as a polynomial in $\\sigma$ |",
    "| ---: | ---: | --- |",
]
for row in periodic_theta_table:
    periodic_table_lines.append(
        f"| {row['s']} | {row['degree']} | $"
        f"{sp.latex(row['polynomial'])}$ |"
    )

display(Markdown("\n".join(periodic_table_lines)))
''',
    "periodic-theta-latex-code": r'''print("Optional copy-ready LaTeX table rows:\n")
for row in periodic_theta_table:
    latex_poly = sp.latex(row["polynomial"])
    print(
        f"${row['s']}$ & "
        f"$\\displaystyle {latex_poly}$ \\\\" 
    )
''',
}


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    by_id = {cell.get("id"): cell for cell in cells}

    missing = sorted(set(REPLACEMENTS) - set(by_id))
    if missing:
        raise RuntimeError(f"Missing target cells: {missing}")

    for cell_id, text in REPLACEMENTS.items():
        by_id[cell_id]["source"] = lines(text)
        if by_id[cell_id]["cell_type"] == "code":
            by_id[cell_id]["execution_count"] = None
            by_id[cell_id]["outputs"] = []

    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"rewrote {len(REPLACEMENTS)} periodic-theta cells")


if __name__ == "__main__":
    main()
