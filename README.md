# `knotted_graph`: Analyzing Non-Hermitian Topological Nodal Structures

<a target="_blank" href="https://colab.research.google.com/github/sarinstein-yan/Nodal-Knot/blob/main/getting_started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
![pre-alpha](https://img.shields.io/badge/status-pre--alpha-red?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/knotted_graph)](https://pypi.org/project/knotted_graph/)

<div align="center" style="display: flex; flex-direction: row; justify-content: center; gap: 20px;">
    <div>
        <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/threelink_gamma=0.1.svg" 
        width="600" alt="Exceptional Surface at γ=0.1"/>
        <br>
        <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/threelink_gamma=0.1.html" 
        target="_blank" style="text-decoration:underline;">
        Click here to view the interactive 3D plot</a>
    </div>
    <div>
        <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/threelink_gamma=0.1.gif" 
        width="600" alt="Exceptional Surface Animation at γ=0.1"/>
    </div>
</div>

<strong>Figure: Non-Hermitian Nodal Phases —</strong>
Exceptional Surface (lightgreen) and *Exceptional Skeleton Graph* (the graph within)

`knotted_graph` is a package designed to analyze and visualize the topological features of 2-band, 3-D non-Hermitian nodal systems. In these systems, the eigen-energies become complex, and points in momentum space where the Hamiltonian's eigenvalues and eigenvectors coalesce simultaneously are known as **exceptional points (EPs)**.

In 3D non-Hermitian nodal systems, these EPs usually form an **exceptional surfaces (ES)**. The **skeleton** (i.e. **medial axis**) serves as a topological fingerprint for the non-Hermitian nodal phase. The `NodalSkeleton` class helps in:

1. Calculating the complex energy spectrum.
2. Visualizing the 3D exceptional surface.
3. Extracting the *medial axis (skeleton)* of the ES, which forms a *spatial multigraph*.
4. Analyzing and visualizing the topology of this skeleton graph.

This guide will walk you through the process of using the `NodalSkeleton` class, from defining a Hamiltonian to analyzing its exceptional skeleton graph.

## Installation

> [!NOTE] 
> The development is still in pre-alpha stage, expect bugs and rapid API changes.

You can install the package via pip:

```bash
$ pip install knotted_graph
```

or clone the repository and install it manually:

```bash
$ git clone https://github.com/sarinstein-yan/Nodal-Knot.git
$ cd Nodal-Knot
$ pip install -e .
```

This module is tested on `Python >= 3.11`.
Check the installation:

```python
import knotted_graph as kg
print(kg.__version__)
```

## Usage

### Initializing the `NodalSkeleton` Class

1. First, one needs to define a (non-interacting) 2-band non-Hermitian Hamiltonian in terms of the momentum vector $\vec{k} = (k_x, k_y, k_z)$.

The class accepts the Hamiltonian "`Characteristic`" in two forms:

- either as a 2x2 `sympy.Matrix`, 

$$H(\vec{k}) = \vec{d}(\vec{k}) \cdot \vec{\sigma}$$

where $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ are the Pauli matrices,

- or directly as a tuple the components of the Bloch vector `(sympy.Expr, sympy.Expr, sympy.Expr)`:

$$\vec{d}(\vec{k}) = ( d_x(\vec{k}), d_y(\vec{k}), d_z(\vec{k}) )$$

The non-Hermiticity arises from complex terms in $\vec{d}(\vec{k})$.

2. Next, optionally, specify the k-space region of interest (the `span` parameter) and the resolution of the k-space grid (the `dimension` parameter).

3. If the $k$ `sympy` symbols in the input Hamiltonian `sp.Matrix` or `(d_x, d_y, d_z)` are named unconventionally, you need to specify them in the `k_symbols` parameter. Otherwise, the `k_symbols` are inferred from the input Hamiltonian `characteristic`.

Let's define a model that is known to produce a **Hopf link** nodal lines in the Hermitian limit. When the non-Hermiticity is introduced, the nodal line (exceptional *line*) will expand into a exceptional *surface*.

```python
import numpy as np
import sympy as sp
from knotted_graph import NodalSkeleton

# Define momentum symbols
kx, ky, kz = sp.symbols('k_x k_y k_z', real=True)

# Use a non-Hermitian Bloch vector that can form a Hopf link
from knotted_graph.examples import hopf_link_bloch_vector

gamma = 0.1  # Non-Hermitian strength
d_x, d_y, d_z = hopf_link_bloch_vector(gamma)

# Initialize the `NodalSkeleton` with the Hamiltonian characteristic
ske = NodalSkeleton(
    char = (d_x, d_y, d_z),
    # k_symbols = (kx, ky, kz), # optional, we have named them *conventionally*
    # span = ((-np.pi, np.pi), (-np.pi, np.pi), (0, np.pi))
    # dimension = 200
)

print(f"Hamiltonian is Hermitian: {ske.is_Hermitian}")
print(f"Hamiltonian is PT-symmetric: {ske.is_PT_symmetric}")
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
Hamiltonian is Hermitian: False
Hamiltonian is PT-symmetric: False
```

-----
### Properties


- Hamiltonian matrix (`sympy.Matrix`):

```python
ske.h_k
```

<span style="color:#d73a49;font-weight:bold">>>></span>

$$\left[\begin{matrix}(2 \cos{(2 k_{z})} + 1) (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2) - 2 \sin{(k_{x})} \sin{(k_{y})} & \frac{(2 \cos{(2 k_{z})} + 1)^{2}}{4} - (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2)^{2} - \sin^{2}{(k_{x})} + \sin^{2}{(k_{y})} + 0.1 \\\\ \frac{(2 \cos{(2 k_{z})} + 1)^{2}}{4} - (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2)^{2} - \sin^{2}{(k_{x})} + \sin^{2}{(k_{y})} - 0.1 & - (2 \cos{(2 k_{z})} + 1) (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2) + 2 \sin{(k_{x})} \sin{(k_{y})}\end{matrix}\right]$$


- Bloch vector (`(sp.Expr, sp.Expr, sp.Expr)`):

```python
ske.bloch_vec
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
((2*cos(2*k_z) + 1)**2/4 - (cos(k_x) + cos(k_y) + cos(k_z) - 2)**2 - sin(k_x)**2 + sin(k_y)**2,
 0.1*I,
 (2*cos(2*k_z) + 1)*(cos(k_x) + cos(k_y) + cos(k_z) - 2) - 2*sin(k_x)*sin(k_y))
```

- $k$-space region information:

```python
print(f"self.dimension: {ske.dimension}")
print(f"self.spacing: {ske.spacing}")
print(f"self.origin: {ske.origin}")

print(f"self.span: {ske.span}")
print(f"self.kx_span: {ske.kx_span}")
print(f"self.ky_span: {ske.ky_span}")
print(f"self.kz_span: {ske.kz_span}")

# Below attributes are also available for y and z
print(f"self.kx_min: {ske.kx_min}")
print(f"self.kx_max: {ske.kx_max}")
print(f"self.kx_symbol: {ske.kx_symbol} | {type(ske.kx_symbol)}")
print(f"self.kx_vals: shape - {ske.kx_vals.shape} | dtype - {ske.kx_vals.dtype}")
print(f"self.kx_grid: shape - {ske.kx_grid.shape} | dtype - {ske.kx_grid.dtype}")
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
self.dimension: 200
self.spacing: [0.0315738 0.0315738 0.0157869]
self.origin: [-3.14159265 -3.14159265  0.        ]
self.span: [[-3.14159265  3.14159265]
 [-3.14159265  3.14159265]
 [ 0.          3.14159265]]
self.kx_span: (-3.141592653589793, 3.141592653589793)
self.ky_span: (-3.141592653589793, 3.141592653589793)
self.kz_span: (0, 3.141592653589793)
self.kx_min: -3.141592653589793
self.kx_max: 3.141592653589793
self.kx_symbol: k_x | <class 'sympy.core.symbol.Symbol'>
self.kx_vals: shape - (200,) | dtype - float64
self.kx_grid: shape - (200, 200, 200) | dtype - float64
```


- Energy spectrum (only the upper band) (`np.ndarray`):

```python
ske.spectrum.shape, ske.spectrum.dtype
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
((200, 200, 200), dtype('complex128'))
```


- Band gap (= `2 × |upper band spectrum|`) (`np.ndarray`):

```python
ske.band_gap.shape, ske.band_gap.dtype
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
((200, 200, 200), dtype('float64'))
```


#### **Skeleton graph** (`networkx.MultiGraph`):
-----

```python
graph = ske.skeleton_graph(
    # simplify = True,  # Topological simplification
    # smooth_epsilon = 4,  # Smoothness, unit is pixel
    # skeleton_image = ... # Can construct a skeleton graph from an skeletonized image
)
graph
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
<networkx.classes.multigraph.MultiGraph at 0x26d8b529b80>
```


- Check if the graph is trivalent

I.e. whether each vertex has degree <= 3. If trivalent, the *Yamada polynomial* is an isotopic invariant of the skeleton multigraph.

```
graph.graph['is_trivalent']
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
(True, True)
```


- Graph summary:

```python
ske.graph_summary()
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
| Property               | Value   |
|------------------------|---------|
| Number of nodes        | 2       |
| Number of edges        | 2       |
| Connected              | No      |
| # Connected components | 2       |
| Component 1 size       | 1       |
| Component 2 size       | 1       |

Degree distribution:
|   Degree |   Frequency |
|----------|-------------|
|        2 |           2 |
```


- Check graph minors:

```python
import networkx as nx

# Check if K_3 graph (a cycle of 3 nodes) is a minor of our skeleton graph
k3_graph = nx.complete_graph(3)
print("Checking for K3 minor...")
ske.check_minor(minor_graph=k3_graph)

# Now, let's try a more complex graph, K4 (complete graph of 4 nodes)
# A simple loop shouldn't contain a K4 minor.
k4_graph = nx.complete_graph(4)
print("\nChecking for K4 minor...")
ske.check_minor(k4_graph, host_graph=graph)
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
Checking for K3 minor...
The given graph DOES NOT contain the minor graph.

Checking for K4 minor...
The given graph DOES NOT contain the minor graph.
```



-----
### Visualization

`NodalSkeleton` uses `pyvista` for 3D plotting, creating interactive visualizations.

#### Plotting the Exceptional Surface
-----

The exceptional surface is the 3D surface in k-space where the band gap closes, defined by
$$
|d(\vec{k})| = 0 \Leftrightarrow d_x(\vec{k})^2 + d_y(\vec{k})^2 + d_z(\vec{k})^2 = 0
$$

The code cells below are meant to be run in a Jupyter notebook.

```python
import pyvista as pv
pv.set_jupyter_backend('client')
EXPORT_FIGS = True  # Set to True to export figures

plotter = ske.plot_exceptional_surface()
plotter.add_bounding_box()
plotter.show()
if EXPORT_FIGS:
    plotter.export_html(f'./assets/ES_gamma={gamma}.html')
    plotter.save_graphic(f'./assets/ES_gamma={gamma}.svg')
    plotter.save_graphic(f'./assets/ES_gamma={gamma}.pdf')
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_gamma=0.1.svg" 
    width="600" alt="Exceptional Surface at γ=0.1"/>
    <br>
    <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_gamma=0.1.html" 
    target="_blank" style="text-decoration:underline;">
    Click here to view the interactive 3D plot</a>
</p>


To add projected silhouettes of the exceptional surface onto the Surface Brillouin Zone (SBZ) planes, set `add_silhouettes=True`:

```python
plotter = ske.plot_exceptional_surface(
    add_silhouettes=True,  # Add projected silhouettes onto the SBZ planes
    silh_origins=np.diag([-np.pi, -np.pi, 0]),  
    # ^ Origin of the planes that the silhouettes are projected onto
)
plotter.show_bounds(xtitle='kx', ytitle='ky', ztitle='kz')
plotter.add_bounding_box()
plotter.zoom_camera(1.2)
plotter.show()
if EXPORT_FIGS:
    plotter.export_html(f'./assets/ES_gamma={gamma}_silhouettes.html')
    plotter.save_graphic(f'./assets/ES_gamma={gamma}_silhouettes.svg')
    plotter.save_graphic(f'./assets/ES_gamma={gamma}_silhouettes.pdf')
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_gamma=0.1_silhouettes.svg" 
    width="600" alt="Exceptional Surface with Silhouettes at γ=0.1"/>
    <br>
    <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_gamma=0.1_silhouettes.html" 
    target="_blank" style="text-decoration:underline;">
    Click here to view the interactive 3D plot</a>
</p>



#### Plotting the ***Exceptional Skeleton Graph***
-----

The exceptional skeleton graph is the ***medial axis*** of the exceptional surface interior, where the *energy spectrum is purely imaginary*.

```python
plotter = ske.plot_skeleton_graph(
    add_nodes=False, # since the skeleton is essentially a Hopf link
    add_silhouettes=True,
    silh_origins=np.diag([-np.pi, -np.pi, 0]),
)
plotter.show_bounds(xtitle='kx', ytitle='ky', ztitle='kz')
plotter.add_bounding_box()
plotter.show()
if EXPORT_FIGS:
    plotter.export_html(f'./assets/SG_gamma={gamma}_silhouettes.html')
    plotter.save_graphic(f'./assets/SG_gamma={gamma}_silhouettes.svg')
    plotter.save_graphic(f'./assets/SG_gamma={gamma}_silhouettes.pdf')
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/SG_gamma=0.1_silhouettes.svg" 
    width="600" alt="Exceptional Skeleton Graph at γ=0.1"/>
    <br>
    <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/SG_gamma=0.1_silhouettes.html" 
    target="_blank" style="text-decoration:underline;">
    Click here to view the interactive 3D plot</a>
</p>


#### **Non-Hermiticity induced exceptional knotted graph**
-----

For nodal knot systems, in the Hermitian limit or when non-Hermitian perturbation is small, the original knot / link topology is preserved, as shown above (`gamma=0.1`).

When non-Hermiticity is prevalent enough, the exceptional surface starts to touch itself, leading to topological transitions --- the skeleton (i.e., medial axis) of the exceptional surface becomes a knotted graph (a.k.a. spatial multigraph).

As the non-Hermiticity evolves, the knotted graph topology evolves accordingly, leading to a plethora of exotic 3D spatial geometries in the momentum space.

E.g., let us set `gamma = [0.2, 0.5]`:

```python
for gamma in [0.2, 0.5]:
    print(f"With gamma = {gamma}:\n")
    
    ske_ = NodalSkeleton(hopf_bloch_vector(gamma))
    ske_.graph_summary(ske.skeleton_graph())

    plotter = ske_.plot_exceptional_surface(surf_opacity=.3, surf_color='lightgreen')
    plotter = ske_.plot_skeleton_graph(plotter=plotter)

    if EXPORT_FIGS:
        plotter.export_html(f'./assets/ES_SG_gamma={gamma}.html')    
        plotter.save_graphic(f'./assets/ES_SG_gamma={gamma}.svg')
        plotter.save_graphic(f'./assets/ES_SG_gamma={gamma}.pdf')
        plotter.show()
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
With gamma = 0.2:

| Property          | Value              |
|-------------------|--------------------|
| Number of nodes   | 4                  |
| Number of edges   | 6                  |
| Connected         | Yes                |
| Diameter          | 2                  |
| Avg shortest path | 1.3333333333333333 |

Degree distribution:
|   Degree |   Frequency |
|----------|-------------|
|        3 |           4 |
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_SG_gamma=0.2.svg" 
    width="600" alt="Exceptional Surface and Skeleton Graph at γ=0.2"/>
    <br>
    <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_SG_gamma=0.2.html" 
    target="_blank" style="text-decoration:underline;">
    Click here to view the interactive 3D plot</a>
</p>

```
With gamma = 0.5:

| Property          | Value   |
|-------------------|---------|
| Number of nodes   | 2       |
| Number of edges   | 3       |
| Connected         | Yes     |
| Diameter          | 1       |
| Avg shortest path | 1.0     |

Degree distribution:
|   Degree |   Frequency |
|----------|-------------|
|        3 |           2 |
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_SG_gamma=0.5.svg" 
    width="600" alt="Exceptional Surface and Skeleton Graph at γ=0.5"/>
    <br>
    <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_SG_gamma=0.5.html" 
    target="_blank" style="text-decoration:underline;">
    Click here to view the interactive 3D plot</a>
</p>


### Planar Diagram and Yamada Polynomial

If a skeleton graph is trivalent (all node degrees <= 3), the *Yamada polynomial* is an isotopic invariant of the spatial graph.

If not trivalent, the Yamada polynomial is still well-defined, but it is not an isotopic invariant, but rather a *rigid isotopy invariant* --- it depends on how one projects the 3D skeleton graph onto a 2D plane.

---
For a trivalent skeleton graph, `NodalSkeleton.yamada_polynomial(variable)` by default will sample `num_rotations=10` different projections that quotient out the rotational symmetry that produces the same planar diagram, and start from the planar diagram with the *least* number of crossings.

If it finds two Yamada polynomials agree, which usually happens right after computing from the best two projections, it will return the agreed Yamada polynomial.

If after `num_rotations` computations, no two Yamada polynomials agree, it will return the projection data and the corresponding Yamada polynomials.

```python
# define the variable of the Yamada polynomial
A = sp.symbols('A')

_ = ske.skeleton_graph() # Ensure the skeleton graph is computed and cached
print("Is the skeleton graph trivalent?", ske.is_graph_trivalent)

# Compute the Yamada polynomial for the Hopf Link
Y = ske.yamada_polynomial(
    variable=A, 
    # normalize=True, # Normalize the Yamada polynomial
    # n_jobs=-1, # Use all available cores for one view

    num_rotations=10, # ONLY for trivalent graphs
    
    # rotation_angles=(0., 0., 0.), # ONLY for non-trivalent graphs
    # rotation_order='ZYX' # ONLY for non-trivalent graphs
)
Y
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
Is the skeleton graph trivalent? True
Computing Yamada polynomial:  10%|█         | 1/10 [00:00<00:01,  5.72it/s]
```

$$- A^{7} - A^{6} - A^{5} + A^{3} + 2 A^{2} + 2 A + 1$$


There a few ways to compute the Yamada polynomial apart from the `NodalSkeleton.yamada_polynomial()` method. \
E.g., by a function call:

```python
kg.compute_yamada_safely(
    skeleton_graph=hopf_link,
    variable=A,
    # num_rotations=10,
    # normalize=True,
    # n_jobs=-1
)
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
Computing Yamada polynomial:  10%|█         | 1/10 [00:00<00:01,  5.72it/s]
```

$$- A^{7} - A^{6} - A^{5} + A^{3} + 2 A^{2} + 2 A + 1$$


Or from the planar diagram code:

```python
pd = kg.PDCode(skeleton_graph=hopf_link)

pd_code = pd.compute(
    # specify the projection angles and order if needed
    rotation_angles=(137.5, 81.4, 0.),
    # rotation_order='ZYX',
)
print(f"planar diagram code: {pd_code}")

pd.compute_yamada(A, normalize=True)
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
planar diagram code: V[0,2];V[3,5];X[4,1,3,2];X[4,0,5,1]
```

$$- A^{7} - A^{6} - A^{5} + A^{3} + 2 A^{2} + 2 A + 1$$


Or from a thinly wrapped function:

```python
kg.compute_yamada_polynomial(hopf_link, A, (137.5, 81.4, 0.))
```

<span style="color:#d73a49;font-weight:bold">>>></span>

$$- A^{7} - A^{6} - A^{5} + A^{3} + 2 A^{2} + 2 A + 1$$


---
For a non-trivalent skeleton graph, the `NodalSkeleton.yamada_polynomial(variable)` will only compute from one projection, specified by the 
`rotation_angles[=(0., 0., 0.)]` and `rotation_order[='ZYX']`
parameters (see `NodalSkeleton.util.get_rotation_matrix` for the meaning of these parameters).

One can call `knotted_graph.util.generate_isotopy_projections` to generate a list of projections sorted by the number of crossings in the planar diagram, and then call `NodalSkeleton.yamada_polynomial(variable, rotation_angles=best_proj['angles'])` to compute the Yamada polynomial from the best projection.

```python
projections = kg.generate_isotopy_projections(
    skeleton_graph=hopf_link, 
    num_rotations=10
)

best_proj = projections[0]
print(f"Keys of a projection: {best_proj.keys()}")
print(f"Number of crossings: {best_proj['num_crossings']}")
print(f"Angles: {best_proj['angles']}")
print(f"pd_code: {best_proj['pd_code']}")

kg.compute_yamada_polynomial(hopf_link, A, best_proj['angles'])
# Or `ske.yamada_polynomial(variable=A, rotation_angles=best_proj['angles'])`
```

<span style="color:#d73a49;font-weight:bold">>>></span>

```
Keys of a projection: dict_keys(['num_crossings', 'vertices', 'crossings', 'arcs', 'angles', 'pd_code'])
Number of crossings: 2
Angles: [0.0, 87.13401601740115, 0.0]
pd_code: V[0,2];V[3,5];X[2,4,1,3];X[4,0,5,1]
```

#### Visualization of the planar diagram

`NodalSkeleton.plot_planar_diagram` can be used to visualize the planar diagram of a given rotation angle for projection.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3,3))
ax = ske.plot_planar_diagram(
    ax = ax,
    rotation_angles = projections[0]['angles'], 
    # rotation_order = 'ZYX',
    # undercrossing_offset = 5.
    # mark_crossings = False
)
ax.set_aspect('equal')
ax.axis('off')
if EXPORT_FIGS:
    plt.savefig("./assets/planar_diagram.png", bbox_inches='tight')
plt.show()
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/planar_diagram.png" 
    width="200" alt="Planar Diagram Visualization"/>
</p>


### Physical Fields Visualization

One can plot physical vector / scalar fields within the Exceptional Surface or on the Skeleton Graph edges.

Fields data are stored in the property `NodalSkeleton.fields_pv` as a `pyvista.ImageData`

```python
bvec = kg.hopf_link_bloch_vector(.4, (kx, ky, kz))
ske = kg.NodalSkeleton(bvec)

ske.fields_pv
```


#### Energy Dispersion $\nabla_{\vec{k}}\text{Im}(E)$

```python
pl = pv.Plotter(shape=(1, 2), window_size=[1200, 600])
pl.subplot(0, 0)
pl = ske.plot_interior_dispersion(pl, glyph_factor=0.1, glyph_tolerance=0.015)
pl.subplot(0, 1)
pl = ske.plot_skeleton_graph(pl, add_edge_field=True, 
                             orient='im_disp', scale='log10(|im_disp|+1)',
                             field_cmap='BuPu', glyph_factor=1.2)
pl.link_views()
pl.camera_position = [[-0.75, -7.62, 3.67], [0.14, -0.08, 0.96], [0.01, 0.34, 0.94]]
if EXPORT_FIGS:
    pl.screenshot("./assets/field_dispersion.png")
pl.show()
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/field_dispersion.png" 
    width="1200" alt="Energy Dispersion Visualization"/>
</p>


#### Berry Curvature $\vec{\Omega} := \nabla_{\vec{k}} \times \vec{A}$

$\vec{A} := \text{Re}(i \left< \phi^L | \nabla_{\vec{k}} | \phi^R \right> )$ is the Berry connection.

```python
pl = pv.Plotter(shape=(1, 2), window_size=[1200, 600])
pl.subplot(0, 0)
pl = ske.plot_berry_curvature(pl, glyph_factor=0.03, glyph_tolerance=0.015)
pl.subplot(0, 1)
pl = ske.plot_skeleton_graph(pl, add_edge_field=True, 
                             orient='berry', scale='log10(|berry|+1)',
                             field_cmap='BuPu', glyph_factor=.3)
pl.link_views()
pl.camera_position = [[-0.75, -7.62, 3.67], [0.14, -0.08, 0.96], [0.01, 0.34, 0.94]]
if EXPORT_FIGS:
    pl.screenshot("./assets/field_berry.png")
pl.show()
```

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/field_berry.png" 
    width="1200" alt="Berry Curvature Visualization"/>
</p>


#### Custom Visualization

One can add custom vector / scalar field data.

E.g. here we add the gradient of the Im(E) and the gradient of Berry curvature magnitude.

```python
vol = ske.fields_pv.copy()
# add extra fields
vol = vol.compute_derivative(scalars='|berry|', gradient='∇|berry|')
vol.point_data['log10(|∇|berry||+1)'] = - np.log10( np.linalg.norm(vol.point_data['∇|berry|'], axis=-1) +1)
vol = vol.compute_derivative(scalars='|im_disp|', gradient='∇|im_disp|')
vol.point_data['log10(|∇|im_disp||+1)'] = - np.log10( np.linalg.norm(vol.point_data['∇|im_disp|'], axis=-1) +1)

vol
```

Visualizing the iso-surfaces of scalar fields with `pyvista` interactive widgets.

```python
scalars = ['imag', 
           'log10(|berry|+1)', 'log10(|im_disp|+1)', 
           '|berry|', '|im_disp|', 
           'log10(|∇|berry||+1)', 'log10(|∇|im_disp||+1)']

# null out the exterior points
mask = np.where(vol.point_data['imag'] == 0)
for s in scalars:
    vol.point_data[s][mask] = np.nan

pv.set_jupyter_backend('trame')
pl2 = pv.Plotter(window_size=(800, 600))
ske.plot_exceptional_surface(pl2, surf_opacity=0.05, surf_color='gray')
pl2.add_mesh_isovalue(vol, scalars=scalars[0], opacity=0.5, cmap='BuPu')
pl2.add_legend()
pl2.add_bounding_box()
pl2.show()
```



## TODO:
- [ ] Documentation website
- [x] Graph diagram visualization with parallel projection
- [ ] Batched processing. Move the spectrum calculation batch to GPU.
- [ ] Multi-band Hamiltonians support


<!-- ## Citation
If you find this work useful, please cite our paper:

```bibtex

``` -->