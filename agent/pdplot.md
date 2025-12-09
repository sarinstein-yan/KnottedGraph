# Planar Diagram Layout & Plotting Design

## 1. Motivation & Current State

The `NodalSkeleton` class currently exposes a `plot_planar_diagram` method that:

* Takes a `PDCode` (planar diagram) associated to a spatial multigraph.
* Uses the existing 2D projection stored in the `PDCode` (as `shapely.LineString` arcs plus vertex/crossing positions).
* Trims undercrossing arcs using a small offset (`undercrossing_offset`) and plots straight line segments with Matplotlib.

This produces a **topologically correct** planar diagram (no spurious intersections), but the layout is often visually suboptimal:

* Vertices and crossings may be crowded.
* Edges can have unnecessary kinks or be very close to each other.
* Multi-component diagrams are not separated or packed nicely.

We want a **geometry-only** improvement step that “massages” the planar diagram into a cleaner layout while preserving its ambient isotopy class.

---

## 2. High-Level Goal

Given a `PDCode` object (which already encodes a valid planar diagram):

1. **Do not modify** the original `PDCode`.
2. Construct an internal **PlanarDiagram layout object** that:

   * Copies the planar embedding (vertices, crossings, arcs, over/under data).
   * Represents all geometric degrees of freedom as a set of 2D control points.
3. Run an **iterative force-based optimization** on these control points:

   * Use Van der Waals-like attraction/repulsion between points.
   * Enforce small step sizes relative to the current minimal segment–segment distance.
   * Preserve the planar embedding (ambient isotopy) at every step.
4. After convergence, **reconstruct arc geometries** (optionally as splines) and:

   * Re-apply undercrossing trimming in world units.
   * Plot the result with Matplotlib.
5. For **multi-component diagrams**, optimize each component separately and then pack components.

---

## 3. Terminology & Data Model

We assume the existing `PDCode` provides:

* **Vertices**: non-crossing nodes (endpoints, degree ≥1).
* **Crossings** (type `'x'`): special nodes representing over/under crossings, with:

  * An ID (`x.id`),
  * A 2D point (`x.point` as `shapely.Point`),
  * A CCW-ordered list of incident arc IDs (`x.ccw_ordered_arcs`),
  * Over/under information encoded via arcs’ `start_type`/`end_type` and positions.
* **Arcs**: edges of the planar multigraph, each with:

  * `id`: arc identifier.
  * `start_type`, `start_id`, `end_type`, `end_id` (vertex or crossing).
  * A `shapely.LineString` (`arc.line`) representing the piecewise-linear geometry from start to end.

We treat the planar diagram as a **planar spatial multigraph**: a planar embedding + geometry.

In the layout algorithm we distinguish:

* **Topological nodes**: vertices and crossings (degree ≠ 2, and all crossings).
* **Geometric control points**:

  * All topological nodes (their positions are variables).
  * Degree-2 nodes that are currently only there for geometric refinement.
  * Interior points on arcs (if we decide to refine polylines).

We **never change** the combinatorial embedding:

* Incidence of arcs to nodes.
* Over/under information at crossings.
* CCW order of arcs around each node.

Only coordinates of control points change.

---

## 4. Proposed API

### 4.1 New Layout Class

Create a new module, e.g.:

```python
# src/knotted_graph/layout/planar_diagram.py
class PlanarDiagramLayout:
    def __init__(
        self,
        pd: PDCode,
        scale_box: tuple[float, float, float, float] = (-10, 10, -10, 10),
        vertex_radius: float = 0.75,
        crossing_radius: float = 0.75,
        edge_point_radius: float = 0.25,
        **hyperparams,
    ):
        ...
```

Key responsibilities:

* Hold a **deep copy** of the PD topology and geometry (internal “layout graph”).
* Maintain bidirectional mappings:

  * `node_id → control point indices`,
  * `arc_id → list of control point indices (ordered along the arc)`.
* Implement:

  * `run_force_layout(...)` – runs the iterative layout.
  * `to_pdcode()` – returns a new `PDCode` with updated geometry (optional).
  * `plot(...)` – plots the optimized diagram (with splines / straight edges).

### 4.2 Integration from `NodalSkeleton`

Add a convenience method that uses the layout:

```python
def plot_pretty_planar_diagram(
    self,
    rotation_angles: Optional[tuple[float]] = None,
    rotation_order: str = "ZYX",
    use_splines: bool = False,
    layout_kwargs: Optional[dict] = None,
    plot_kwargs: Optional[dict] = None,
) -> plt.Axes:
    """
    1. Compute PDCode from NodalSkeleton (existing pipeline).
    2. Build PlanarDiagramLayout(pd, **layout_kwargs).
    3. Run layout.run_force_layout().
    4. Call layout.plot(use_splines=use_splines, **plot_kwargs).
    """
```

The existing `plot_planar_diagram` (straight lines, no layout) can stay as a “raw” option.

---

## 5. Geometry Normalization

Before running the force-based layout:

1. **Work on a copy**
   Build an internal representation of PD:

   * Copy all nodes, crossings, arcs.
   * Keep IDs and over/under annotations.

2. **Scale to a canonical box**
   Compute bounding box `(xmin, ymin, xmax, ymax)` of all node and control point coordinates.
   Apply an affine transform to map this to a square, default:

   * Target box: `(-10, -10) → (10, 10)`.

   All subsequent layout parameters (`vertex_radius`, `edge_point_radius`, `undercrossing_offset`, etc.) are defined in this normalized coordinate system.

3. **Control point extraction**

   For each arc:

   * Start with its `LineString` coordinates.
   * Identify which points correspond to topological nodes (start/end and any degree-2 nodes).
   * Mark:

     * **Topological node control points** (must exist; connected to node IDs).
     * **Edge control points** (interior polyline vertices; no topological identity).

   Degree-2 nodes that exist only for geometric refinement should be treated as **edge control points** in the force simulation, but we must still be able to reconstruct the original topological structure.

---

## 6. Force-Field Layout Algorithm

We use a **Van der Waals-like force model** on control points:

### 6.1 Particles & Radii

Each control point `i` is a particle at position `p_i = (x_i, y_i)` with a “hard radius”:

* Vertex control point: `r_i = vertex_radius` (≈ 0.75 units).
* Crossing control point: `r_i = crossing_radius` (≈ 0.75 units).
* Edge control point: `r_i = edge_point_radius` (≈ 0.25 units).

Radii and strengths are hyperparameters.

### 6.2 Forces

Conceptually, for any pair of points `(i, j)` we define:

* **Repulsive force** when `‖p_i − p_j‖` is smaller than some cutoff (say `r_repulsive`):

  * E.g. `F_rep ~ k_rep / d^α` (α ≥ 2), with stronger effect at short distances.
* **Weak attractive force** beyond a comfortable range to avoid components collapsing into a tiny region:

  * E.g. `F_att ~ -k_att * (d - d0)` for `d > d0`, where `d0` is a “comfortable” spacing.

Additionally, we should have **elastic forces along arcs**:

* Treat each arc as a chain of control points.
* Add “springs” between consecutive control points:

  * Keep edge length roughly constant (to avoid collapsing).
  * Add a small curvature penalty to prefer smoother edges.

We don’t need a closed-form “layout loss” explicitly in code; these forces are gradients of an implicit potential.

### 6.3 Time Integration

We use discrete steps:

1. For each iteration:

   * Compute net force `F_i` on each control point (sum of all pairwise and internal forces).
   * Compute tentative displacement `Δp_i = dt * F_i` (dt is a time step).
2. Apply a global displacement cap based on geometry (see below).
3. Update positions: `p_i ← p_i + Δp_i`.
4. Rebuild shapely geometries and validate topology.
5. If topology invalid, reduce `dt` and retry step.

Use standard tricks:

* Global damping factor to avoid oscillations.
* Iterative schedule: `dt` can shrink as we approach equilibrium.

Stop when:

* Max `‖Δp_i‖` across all points < `epsilon_disp` (e.g. 1e-3),
* Or max absolute force < `epsilon_force`,
* Or a max number of iterations is reached.

---

## 7. Step Size Constraint via STRtree

To avoid accidentally introducing segment intersections:

1. From the current geometry:

   * Build a list of **segments** representing all arcs, but with degree-2 vertices collapsed as ordinary interior points.
   * Insert these segments into a `shapely.STRtree` for efficient spatial queries.

2. Compute the **approximate minimal segment–segment distance** `d_min`:

   * For each segment, query nearby segments via STRtree.
   * Only consider segments from **different arcs** or non-adjacent segments on the same arc.
   * Ignore segments sharing a common endpoint (those are “legitimate” connections).

3. Set a global max step size:

   ```text
   max_step = α * d_min
   ```

   where α ∈ (0, 0.5] (e.g., 0.25).

4. For each control point, after computing `Δp_i`, clip:

   ```python
   if ||Δp_i|| > max_step:
       Δp_i *= (max_step / ||Δp_i||)
   ```

This ensures that in a single iteration we do not move any point by more than a fraction of the current minimum separation between segments, greatly reducing the risk of creating new intersections.

---

## 8. Ambient Isotopy Preservation

We preserve the ambient isotopy class by enforcing two conditions:

1. **No new intersections** beyond existing crossings.
2. **Preserve CCW order of arcs at every vertex and crossing.**

### 8.1 Intersection Check

After each tentative position update:

* Rebuild segment list for all arcs (using updated control point positions).
* For each pair of segments:

  * If they share an endpoint, skip.
  * If their intersection is non-empty:

    * Check if this corresponds to an existing crossing.

      * i.e., the intersection point coincides (within tolerance) with a crossing node.
      * And the two segments belong to the two arcs that define that crossing.
    * If not, we’ve introduced an illegal crossing → **reject step**.

On rejection:

* Reduce `dt` (e.g., `dt ← dt / 2`) and recompute the forces / displacements for this iteration.
* If repeated failures occur, abort layout and return best-so-far layout.

### 8.2 CCW Order Preservation

For each vertex/crossing node:

1. At initialization, compute the CCW order of incident arcs:

   * For each incident arc, compute the outgoing direction vector near the node (e.g., using the first segment of that arc from the node).
   * Compute its angle `θ` via `atan2`.
   * Sort arcs by `θ` to obtain the reference ordering (this should match `x.ccw_ordered_arcs` for crossings).

2. After each step:

   * Recompute the angles and their ordering.
   * Verify that the new ordering is the same as the reference up to rotation (cyclic shift).

     * A cyclic shift is allowed (just a different starting point on the same circle).
   * If any vertex/crossing has a different cyclic ordering, the local embedding changed → **reject step** (reduce `dt` and retry).

This ensures that we do not perform unintended “Reidemeister moves” that change the knot/link type.

---

## 9. Undercrossing Trimming (Final Stage)

We reuse the logic of the existing `plot_planar_diagram`, but:

* Apply **after** the layout has converged.
* Work in world units of the normalized coordinates (e.g. undercrossing offset = `0.1`).

Algorithm:

1. Construct a mapping `under_arcs[arc_id] = [under_at_start, under_at_end]` exactly as in the current implementation.

2. For each arc’s final geometry (a `LineString` or spline sample):

   * Let `L` be its length.
   * Let `t = undercrossing_offset` (e.g. `0.1`).
   * If under at both ends: take substring from `t` to `L − t`.
   * If under at start only: substring from `t` to `L`.
   * If under at end only: substring from `0` to `L − t`.
   * Otherwise: keep as is.

3. Plot over/under arcs in the correct z-order (`zorder` for under arcs lower than over arcs).

Note: we no longer interpret `undercrossing_offset` in “pixels”; in this normalized scheme it is a geometric length in the same units as the layout.

---

## 10. Multi-Component Diagrams

For links or diagrams with several connected components:

1. Use the underlying multigraph (from `PDCode`’s associated `networkx` graph) to compute connected components.

2. For each component:

   * Extract the induced PD (nodes, crossings, arcs).
   * Build a separate `PlanarDiagramLayout` instance.
   * Run the force-layout independently.

3. Packing:

   * For each optimized component, compute its bounding box.
   * Place components in a simple “packing layout”, e.g. side-by-side along the x-axis with fixed margins or in a grid.
   * Optionally rescale each component slightly to harmonize sizes.
   * Apply the packing translations to all control points of each component.
   * Ensure packing translations are large compared to `vertex_radius`/`edge_point_radius` so components do not touch.

We don’t need to preserve any relative positioning between components; only their internal topology matters.

---

## 11. Rendering: Straight vs. Spline Edges

The layout itself is defined on control points. Rendering should support:

1. **Straight segments** (default):

   * For each arc, connect its control points piecewise-linearly and then apply trimming.
   * This reproduces current behavior with improved positions.

2. **Spline curves** (optional, `use_splines=True`):

   * For each arc, take its (ordered) control points.
   * Fit a 2D spline (e.g. cubic B-spline) to these points.

     * Implementation detail:

       * Prefer a dependency-light solution; if `scipy` is available, use `splprep/splev`.
       * Otherwise, implement a simple Catmull–Rom or cubic Bezier subdivision.
   * Sample the spline at a fixed number of points (e.g. 50) to produce a smooth polyline.
   * Feed this polyline into the undercrossing trimming logic and then plot.

Expose this via `PlanarDiagramLayout.plot(use_splines: bool, ...)`.

---

## 12. Hyperparameters & Config

Expose a configuration object or keyword arguments to `PlanarDiagramLayout` for easy tuning:

* Normalization:

  * `scale_box: (xmin, xmax, ymin, ymax)` – default `(-10, 10, -10, 10)`.
* Radii:

  * `vertex_radius`, `crossing_radius`, `edge_point_radius`.
* Forces:

  * `k_rep`, `k_att`, exponents `α`, comfortable distance `d0`.
  * Spring constants along arcs, curvature penalties.
* Time integration:

  * Initial `dt`, damping factor, max iterations.
  * `alpha_step` (fraction of `d_min` used as max step).
* Topology checks:

  * Intersection tolerance.
  * Angle tolerance when comparing CCW orderings.

Defaults should be chosen so that typical diagrams converge in a few hundred iterations without self-intersections.

---

## 13. Testing & Validation

Suggested tests:

1. **Topological invariance**:

   * For a suite of known knots/links (e.g. trefoil, figure-8, Hopf link, multi-component links):

     * Run layout.
     * Verify:

       * No additional crossings added or removed.
       * Over/under pattern matches original.
2. **Planarity**:

   * After layout, check with shapely that no non-crossing intersections exist.
3. **CCW order**:

   * Assert that CCW orders at all vertices/crossings are preserved (up to rotation).
4. **Regression tests**:

   * Confirm that `plot_planar_diagram` (old) + `plot_pretty_planar_diagram` (new) produce consistent PDCode combinatorics and that old code still works.

---

## 14. Summary

* We introduce a **PlanarDiagramLayout** abstraction that:

  * Takes a `PDCode` as input.
  * Builds a control-point-based geometric model.
  * Performs a constrained, force-field-driven relaxation in a normalized coordinate box.
  * Enforces ambient isotopy via intersection checks and CCW order preservation.
  * Optionally renders edges as splines.
* The original `PDCode` and `NodalSkeleton` APIs remain intact; we only add a new, “pretty” plotting path and keep the “raw” plotting path for debugging.