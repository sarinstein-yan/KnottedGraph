# `knotted_graph`: Analyzing Non-Hermitian Topological Nodal Structures

<a target="_blank" href="https://colab.research.google.com/github/sarinstein-yan/Nodal-Knot/blob/main/getting_started.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

`knotted_graph` is a package designed to analyze and visualize the topological features of 2-band, 3-D non-Hermitian nodal systems. In these systems, the eigen-energies become complex, and points in momentum space where the Hamiltonian's eigenvalues and eigenvectors coalesce simultaneously are known as **exceptional points (EPs)**.

In 3D non-Hermitian nodal systems, these EPs usually form an **exceptional surfaces (ES)**. The **skeleton** (i.e. **medial axis**) serves as a topological fingerprint for the non-Hermitian nodal phase. The `NodalSkeleton` class helps in:

1. Calculating the complex energy spectrum.
2. Visualizing the 3D exceptional surface.
3. Extracting the *medial axis (skeleton)* of the ES, which forms a *spatial multigraph*.
4. Analyzing and visualizing the topology of this skeleton graph.

This guide will walk you through the process of using the `NodalSkeleton` class, from defining a Hamiltonian to analyzing its exceptional skeleton graph.

## Installation

<!-- You can install the package via pip:
```bash
$ pip install knotted_graph
```
or  -->
clone the repository and install it manually:

```bash
$ git clone https://github.com/sarinstein-yan/Nodal-Knot.git
$ cd Nodal-Knot
$ pip install .
```

This module is tested on `Python >= 3.11`.
Check the installation:

```python
import knotted_graph as kg
print(kg.__version__)
```

## Usage

### Initializing the `NodalSkeleton` Class

1. First, one needs to define a 2-band non-Hermitian Hamiltonian in terms of the momentum vector $\vec{k} = (k_x, k_y, k_z)$.

    The class accepts the Hamiltonian "`Characteristic`" in two forms:

    - either as a 2x2 `sympy.Matrix`, 

    $$
    H(\vec{k}) = \vec{d}(\vec{k}) \cdot \vec{\sigma}
    $$

    where $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ are the Pauli matrices,

    - or directly as a tuple the components of the Bloch vector `(sympy.Expr, sympy.Expr, sympy.Expr)`:

    $$
    \vec{d}(\vec{k}) = [ d_x(\vec{k}), d_y(\vec{k}), d_z(\vec{k}) ]
    $$

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

# Define a non-Hermitian Bloch vector that can form a trefoil knot
def hopf_bloch_vector(gamma, k_symbols=(kx, ky, kz)):
    """Returns the Bloch vector components for a Hopf link."""
    kx, ky, kz = k_symbols
    z = sp.cos(2*kz) + sp.Rational(1, 2) \
        + sp.I*(sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - 2)
    w = sp.sin(kx) + sp.I*sp.sin(ky)
    f = z**2 - w**2 
    cx = sp.simplify(sp.re(f))
    cz = sp.simplify(sp.im(f))
    return (cx, gamma * sp.I, cz)

gamma = 0.8  # Non-Hermitian strength
d_x, d_y, d_z = hopf_bloch_vector(gamma)

# Initialize the NodalSkeleton with the Hamioltonian characteristic
ske = NodalSkeleton(
    char = (d_x, d_y, d_z),
    # k_symbols = (kx, ky, kz), # optional, we are naming them *conventionally*
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

$$
\left[\begin{matrix}(2 \cos{(2 k_{z})} + 1) (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2) - 2 \sin{(k_{x})} \sin{(k_{y})} & 
\frac{(2 \cos{(2 k_{z})} + 1)^{2}}{4} - (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2)^{2} - \sin^{2}{(k_{x})} + \sin^{2}{(k_{y})} + 0.1
\\
\frac{(2 \cos{(2 k_{z})} + 1)^{2}}{4} - (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2)^{2} - \sin^{2}{(k_{x})} + \sin^{2}{(k_{y})} - 0.1 & 
- (2 \cos{(2 k_{z})} + 1) (\cos{(k_{x})} + \cos{(k_{y})} + \cos{(k_{z})} - 2) + 2 \sin{(k_{x})} \sin{(k_{y})}\end{matrix}\right]
$$


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

```python
graph = ske.skeleton_graph(
    simplify = True,  # Topological simplification
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
True
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
The given graph does not contain the minor graph.

Checking for K4 minor...
The given graph does not contain the minor graph.
```



-----
### Visualization

`NodalSkeleton` uses `pyvista` for 3D plotting, creating interactive visualizations.

#### Plotting the Exceptional Surface

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
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_gamma=0.1.svg" width="600" />
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
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_gamma=0.1_silhouettes.svg" width="600" />
</p>



#### Plotting the ***Exceptional Skeleton Graph***

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
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/SG_gamma=0.1_silhouettes.svg" width="600" />
</p>


#### **Non-Hermiticity induced exceptional knotted graph**

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
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_SG_gamma=0.2.svg" width="600" />
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
    <img src="https://raw.githubusercontent.com/sarinstein-yan/Nodal-Knot/main/assets/ES_SG_gamma=0.5.svg" width="600" />
</p>