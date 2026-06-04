# Yamada Debug Notes

This directory contains local diagnostic scripts for the current Yamada
polynomial implementation in `src/knotted_graph/yamada`. The goal is to
separate two questions:

1. whether the polynomial differences in the repulsive-layout figure are caused
   by our figure-generation code, and
2. whether the current PD-code-to-Yamada implementation is stable under
   isotopic changes of projection.

The current conclusion is that the issue is not caused by the figure rendering
side. The source implementation can produce projection-dependent Yamada
polynomials for the same embedded spatial graph, so the values shown in figures
should be treated as provisional until this is reviewed and fixed.

## Observed Problem

The unmodified source implementation shows two red flags:

- A crossing-free theta graph gives the expected nonzero baseline polynomial,
  but a theta graph with a removable Reidemeister-II bigon evaluates to `0`
  instead of a monomial-equivalent value.
- For the tested protein examples (`1aoc`, `3ulk`, `5osq`), two different
  projection angles of the same relaxed spatial graph can produce different
  normalized Yamada polynomials.

This suggests that the bug is deeper than formatting or camera choice. It is
likely in the crossing/state-graph construction used before the recursive
Yamada evaluation.

## Files

- `reproduce_yamada_issues.py`

  Runs the unmodified source implementation. It constructs small synthetic theta
  graph examples, checks a removable Reidemeister-II bigon, checks an in-plane
  rotation sanity case, and compares two projection angles for the three protein
  examples when their layout JSON files are available.

- `yamada_fixed_copy.py`

  A prototype implementation that is intentionally kept outside
  `src/knotted_graph/yamada`. It rebuilds the resolved state graphs using
  explicit crossing half-edge ports instead of mutating a NetworkX graph while
  resolving crossings. This avoids losing local crossing information when one
  crossing resolution removes or re-adds nodes needed by another crossing.

- `run_fixed_copy_checks.py`

  Runs the same small examples and protein projection checks using
  `yamada_fixed_copy.py`. In the tests run during debugging, this copy made the
  removable bigon equivalent to the crossing-free theta baseline and made the
  tested protein projections agree after normalization.

## How To Run

From the repository root:

```powershell
python build\yamada_debug\reproduce_yamada_issues.py
python build\yamada_debug\run_fixed_copy_checks.py
```

The protein checks expect layout files under:

```text
build/repulsive_layout_route_c_stronger_final_relax/<sample>/04_route_c_stronger_smooth_relax/layout.json
```

If those build artifacts are absent, use the minimal theta-graph checks first.

## Status

These scripts are diagnostic only. They do not modify the source package and
should not be treated as the approved Yamada implementation. The intended next
step is to discuss the diagnosis with the Yamada-code owner, then port the
minimal corrected state-graph construction into `src/knotted_graph/yamada` with
proper tests if the approach is accepted.
