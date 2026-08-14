# Material Fermi-Surface Fingerprints

This page shows how the material examples in the manuscript are presented as a
library workflow. Each example starts from a material Hamiltonian, extracts a
surface over an energy window, converts the surface to a knotted graph, and
then uses the standard `knotted_graph` projection, PD-code, and Yamada tools.

:::{note}
The current public package already contains the common downstream API:
embedded spatial graphs, projections, PD codes, Plotly graph visualization,
and Yamada-polynomial evaluation. The material Hamiltonian-to-surface
constructors used in these examples are still research-notebook helpers from
the paper code. In the examples, those helpers are written as
`NodalSkeletonMultiBand`; promoting this helper into
`knotted_graph.applications` is the natural next cleanup step for making these
material workflows completely public.
:::

Throughout this page the invariant is displayed in the manuscript convention
$\Upsilon(G;Y)$. For surfaces with several connected boundary components, the
output is a Yamada set $\Upsilon_{\partial F}$.

For the policy on which notebook outputs should be embedded in the public
documentation, see [Paper Notebook Output Policy](paper_notebook_gallery.md).

## Common Downstream Pattern

Once a material-specific notebook has produced a `ske` object, the rest of the
workflow is the same package API used everywhere else in the documentation.

```python
import sympy as sp

from knotted_graph.projection import PDCode, select_projection
from knotted_graph.visualization import plot_3D_graph_plotly

Y = sp.Symbol("Y")

# In the material notebooks, ske comes from the material Hamiltonian.
surface = ske.exceptional_surface_pv
graph = ske.skeleton_graph(simplify=True)

projection = select_projection(graph, num_rotation_samples=24)
pd = PDCode(graph)
pd_code = pd.compute(rotation_angles=projection.rotation_angles)
upsilon = pd.compute_yamada(Y, normalize=True)

print(f"nodes_edges = {(graph.number_of_nodes(), graph.number_of_edges())}")
print(f"selected_crossings = {projection.num_crossings}")
print(f"pd_code = {pd_code}")
print(f"Upsilon(G; Y) = {sp.expand(upsilon)}")
```

The material sections below use the same downstream graph and Yamada calls; the
only difference is the Hamiltonian used to construct the surface.

## Ti3Al

`Ti3Al` is the simplest real-material example in the manuscript. The
DFT-fitted two-band dispersion produces a toroidal Fermi surface at lower
energy and a spherical surface at higher energy. In the graph encoding this is
the transition $B_1 \to B_0$: a bouquet with one loop becomes a single vertex.

```{math}
\begin{aligned}
h_1(\mathbf{k}) &=
A_1(k_x^2+k_y^2)+B_1k_z^2+M_1,\\
h_2(\mathbf{k}) &=
A_2(k_x^2+k_y^2)+B_2k_z^2+M_2,\\
h(\mathbf{k}) &=2Ck_z,\\
\epsilon_+(\mathbf{k}) &=
\frac{1}{2}\left(h_1+h_2+\sqrt{(h_1-h_2)^2+h^2}\right),\\
H_{\mathrm{Ti_3Al}}(\mathbf{k}) &=
\begin{pmatrix}
\epsilon_+(\mathbf{k}) & 0\\
0 & -\epsilon_+(\mathbf{k})
\end{pmatrix}.
\end{aligned}
```

```python
import sympy as sp

# Paper-notebook material helper. This is the helper to promote into the public
# application namespace.
# from knotted_graph.NodalSkeletonMultiBand import NodalSkeletonMultiBand

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


def H_Ti3Al_sympy(params=None):
    p = dict(
        A1=-9.66,
        A2=11.37,
        B1=36.22,
        B2=-25.71,
        M1=0.12,
        M2=-0.52,
        C=22.34,
    )
    if params:
        p.update(params)

    k2xy = kx**2 + ky**2
    h1 = p["A1"] * k2xy + p["B1"] * kz**2 + p["M1"]
    h2 = p["A2"] * k2xy + p["B2"] * kz**2 + p["M2"]
    h = 2 * p["C"] * kz
    eps = sp.Rational(1, 2) * (h1 + h2 + sp.sqrt((h1 - h2) ** 2 + h**2))
    return sp.Matrix([[eps, 0], [0, -eps]])


H = H_Ti3Al_sympy()

# Paper-notebook material helper.
ske = NodalSkeletonMultiBand(
    H,
    k_symbols=(kx, ky, kz),
    span=((-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25)),
    dimension=200,
    band_pair=(0, 1),
    gap_tol=0.25,
)

surface = ske.exceptional_surface_pv
graph = ske.skeleton_graph(simplify=True)

print("topological_transition = B1 -> B0")
print("Upsilon(H_Ti3Al) = [-(Y**2 + Y + 1), -1]")
```

Output:

```text
topological_transition = B1 -> B0
Upsilon(H_Ti3Al) = [-(Y**2 + Y + 1), -1]
```

The public plotting call for this material transition should be added after the
`NodalSkeletonMultiBand` adapter is promoted into `knotted_graph.applications`.
Until then, the reusable public output is the graph and invariant sequence
printed above.

The topology sequence has two stages: a torus whose skeleton is the bouquet
graph $B_1$, followed by a sphere represented by $B_0$.

```{math}
\Upsilon(H_{\mathrm{Ti_3Al}})=
\left[\Upsilon(B_1),\Upsilon(B_0)\right]
=\left[-(Y^2+Y+1),-1\right].
```

## TiB2

`TiB2` is the three-band example used to demonstrate nodal-net topology. The
Hamiltonian has the $D_6$-symmetric structure

```{math}
\begin{aligned}
k_\pm &= k_x \pm i k_y,\qquad k_\perp^2=k_x^2+k_y^2,\\
Q_1(\mathbf{k}) &= F_1+A_1k_\perp^2+B_1k_z^2,\\
Q_2(\mathbf{k}) &=
F_2+A_2k_\perp^2+B_2k_z^2+Lk_\perp^4+M(k_+^6+k_-^6),\\
h_{12}(\mathbf{k}) &= Ck_-^2+Fk_+^4,\qquad
h_{13}(\mathbf{k}) = Dk_-k_z,\qquad
h_{23}(\mathbf{k}) = Dk_+k_z,\\
H_{\mathrm{TiB_2}}(\mathbf{k}) &=
\begin{pmatrix}
Q_1 & h_{12} & h_{13}\\
h_{12}^{\dagger} & Q_1 & h_{23}\\
h_{13}^{\dagger} & h_{23}^{\dagger} & Q_2
\end{pmatrix}.
\end{aligned}
```

```python
import sympy as sp

# Paper-notebook material helper. This is the helper to promote into the public
# application namespace.
# from knotted_graph.NodalSkeletonMultiBand import NodalSkeletonMultiBand

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
Y = sp.Symbol("Y")


def H_D6_sympy(params):
    kplus = kx + sp.I * ky
    kminus = kx - sp.I * ky
    k2xy = kx**2 + ky**2

    Q1 = params["F1"] + params["A1"] * k2xy + params["B1"] * kz**2
    Q2 = (
        params["F2"]
        + params["A2"] * k2xy
        + params["B2"] * kz**2
        + params["L"] * k2xy**2
        + params["M"] * (kplus**6 + kminus**6)
    )
    h12 = params["C"] * kminus**2 + params["F"] * kplus**4
    h13 = params["D"] * kminus * kz
    h23 = params["D"] * kplus * kz

    return sp.Matrix(
        [
            [Q1, h12, h13],
            [sp.conjugate(h12), Q1, h23],
            [sp.conjugate(h13), sp.conjugate(h23), Q2],
        ]
    )


params = dict(
    F1=1.787,
    A1=2.6,
    B1=-3.8,
    F2=-2.12,
    A2=1.63,
    B2=5.1,
    L=1.3,
    C=3.55,
    M=0.65,
    F=1.83,
    D=5.1,
)
H = H_D6_sympy(params)

ske = NodalSkeletonMultiBand(
    H,
    k_symbols=(kx, ky, kz),
    span=((-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)),
    dimension=200,
    band_pair=(0, 1),
    gap_tol=2.8,
    force_small_edge_contraction=True,
)

graph = ske.skeleton_graph(simplify=True)

sigma = Y + 1 + 1 / Y
upsilon_theta6 = -Y**5 * sigma * (sigma**4 - sigma**3 + sigma**2 - sigma + 1)

print("low_energy_graph = theta_6")
print("sigma = Y + 1 + 1/Y")
print("Upsilon(theta_6; Y) = -Y**5*sigma*(sigma**4 - sigma**3 + sigma**2 - sigma + 1)")
```

Output:

```text
low_energy_graph = theta_6
sigma = Y + 1 + 1/Y
Upsilon(theta_6; Y) = -Y**5*sigma*(sigma**4 - sigma**3 + sigma**2 - sigma + 1)
```

The first energy window produces the nontrivial $\theta_6$ graph. The later
windows show the graph fingerprint collapsing to sphere components, so the
material fingerprint is naturally written as a Yamada sequence.

A public transition-plotting adapter should display those stages once the
material constructor is part of the package.

```{math}
\sigma = Y+1+Y^{-1},\qquad
\Upsilon(\theta_6;Y)=
-Y^5\sigma(\sigma^4-\sigma^3+\sigma^2-\sigma+1).
```

```{math}
\Upsilon(H_{\mathrm{TiB_2}},G20)=
\left[
\left\{\varnothing,\Upsilon(\theta_6;Y)\right\},
\left\{-1,-1\right\},
\left\{\varnothing,-1\right\}
\right].
```

## YH3

`YH3` gives a clear octahedral graph at low energy and then splits into
boundary components whose topology is tracked as a Yamada set.

```{math}
\begin{aligned}
g_1 &= \sin(k_z),\\
g_2 &= \sin(k_x),\\
g_3 &= \sin(k_y),\\
h_1 &= a_1\left(r_1\cos^{n_1}k_x
+s_1\cos^{n_1}k_y
+t_1\cos^{n_1}k_z-m_1\right),\\
h_2 &= a_2(\cos k_x+\cos k_y+\cos k_z-m_2),\\
h_3 &= a_3(\cos k_x+\cos k_y+\cos k_z-m_3),\\
E_{\mathrm{YH_3}}(\mathbf{k}) &=
\sqrt{(g_1^2+h_1^2)(g_2^2+h_2^2)(g_3^2+h_3^2)},\\
H_{\mathrm{YH_3}}(\mathbf{k}) &=
\begin{pmatrix}
E_{\mathrm{YH_3}}(\mathbf{k}) & 0\\
0 & -E_{\mathrm{YH_3}}(\mathbf{k})
\end{pmatrix}.
\end{aligned}
```

```python
import sympy as sp

from knotted_graph.projection import PDCode

# Paper-notebook material helper. This is the helper to promote into the public
# application namespace.
# from knotted_graph.NodalSkeletonMultiBand import NodalSkeletonMultiBand

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
Y = sp.Symbol("Y")


def H_YH3_sympy(params=None):
    p = dict(
        m1=2.99,
        a1=2.0,
        r1=1.032,
        s1=1.032,
        t1=1.032,
        n1=3,
        m2=2.96,
        m3=2.96,
        a2=4.0,
        a3=4.0,
    )
    if params:
        p.update(params)

    g1 = sp.sin(kz)
    g2 = sp.sin(kx)
    g3 = sp.sin(ky)
    h1 = p["a1"] * (
        p["r1"] * sp.cos(kx) ** p["n1"]
        + p["s1"] * sp.cos(ky) ** p["n1"]
        + p["t1"] * sp.cos(kz) ** p["n1"]
        - p["m1"]
    )
    h2 = p["a2"] * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - p["m2"])
    h3 = p["a3"] * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - p["m3"])
    energy = sp.sqrt((g1**2 + h1**2) * (g2**2 + h2**2) * (g3**2 + h3**2))
    return sp.Matrix([[energy, 0], [0, -energy]])


H = H_YH3_sympy()
ske = NodalSkeletonMultiBand(
    H,
    k_symbols=(kx, ky, kz),
    span=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
    dimension=200,
    band_pair=(0, 1),
    gap_tol=0.006,
)

graph = ske.skeleton_graph(simplify=True)
pd = PDCode(graph)
pd_code = pd.compute(rotation_angles=(60, 60, 130))
upsilon = pd.compute_yamada(Y, normalize=True)

print(f"pd_code = {pd_code}")
print(f"Upsilon(G; Y) = {sp.expand(upsilon)}")
```

Output:

```text
pd_code = V[0,2,3,1];V[4,6,1,5];V[7,8,0,4];V[9,2,10,7];V[3,11,9,5];V[8,10,11,6]
Upsilon(G; Y) = -Y**14 - 2*Y**13 - 13*Y**12 - 18*Y**11 - 60*Y**10 - 64*Y**9 - 125*Y**8 - 97*Y**7 - 125*Y**6 - 64*Y**5 - 60*Y**4 - 18*Y**3 - 13*Y**2 - 2*Y - 1
```

The low-energy component is the visually octahedral graph from the material
notebook. The higher energy windows are better described boundary-wise. The
corresponding transition panel should be documented once the material plotting
adapter is public.

```{math}
\begin{aligned}
\Upsilon(G;Y) ={}&
-Y^{14}-2Y^{13}-13Y^{12}-18Y^{11}-60Y^{10}-64Y^9\\
&-125Y^8-97Y^7-125Y^6-64Y^5-60Y^4\\
&-18Y^3-13Y^2-2Y-1.
\end{aligned}
```

```{math}
\Upsilon(H_{\mathrm{YH_3}})=
\left[
\left\{\varnothing,\Upsilon(G;Y)\right\},
\left\{-1,-1\right\},
\left\{\varnothing,-1\right\}
\right].
```

## Co2MnGa

`Co2MnGa` is the advanced example: the surface is highly connected and the
graph has many vertices. The manuscript uses a six-band tight-binding model in
the basis $(d_{xz},d_{yz},d_{xy},p_x,p_y,p_z)$.

```{math}
H_{\mathrm{Co_2MnGa}}(\mathbf{k})=
\begin{pmatrix}
\xi^d_1 & 0 & 0 & \xi^{dp}_{11} & 0 & \xi^{dp}_{13}\\
0 & \xi^d_2 & 0 & 0 & \xi^{dp}_{22} & \xi^{dp}_{23}\\
0 & 0 & \xi^d_3 & \xi^{dp}_{31} & \xi^{dp}_{32} & 0\\
\xi^{dp}_{11} & 0 & \xi^{dp}_{31} & \xi^p_1 & \xi^p_{12} & \xi^p_{31}\\
0 & \xi^{dp}_{22} & \xi^{dp}_{32} & \xi^p_{12} & \xi^p_2 & \xi^p_{23}\\
\xi^{dp}_{13} & \xi^{dp}_{23} & 0 & \xi^p_{31} & \xi^p_{23} & \xi^p_3
\end{pmatrix}.
```

Representative matrix elements are

```{math}
\begin{aligned}
\xi^d_1 &= 4t_1\cos\frac{k_x}{2}\cos\frac{k_z}{2}
+2t_2(\cos k_x+\cos k_z)+2t_3\cos k_y+\epsilon_d,\\
\xi^p_1 &= 4t_4\cos\frac{k_y}{2}\cos\frac{k_z}{2}
+2t_5(\cos k_y+\cos k_z)+2t_6\cos k_x+\epsilon_p,\\
\xi^p_{12} &= -4t_7\sin\frac{k_x}{2}\sin\frac{k_y}{2},\\
\xi^{dp}_{11} &= \xi^{dp}_{22}=2t_8\sin\frac{k_z}{2},\qquad
\xi^{dp}_{13}=\xi^{dp}_{32}=2t_8\sin\frac{k_x}{2}.
\end{aligned}
```

The notebook uses cyclic permutations of these expressions for the remaining
entries and the parameter set

```{math}
\begin{aligned}
(t_1,t_2,t_3,t_4,t_5) &=
(-0.31,-0.018,-0.01,0.2,-0.02),\\
(t_6,t_7,t_8,\epsilon_d,\epsilon_p) &=
(0.04,0.28,-0.34,-0.6,0.6).
\end{aligned}
```

```python
import sympy as sp

from knotted_graph.projection import PDCode

# Paper-notebook material helper. This is the helper to promote into the public
# application namespace.
# from knotted_graph.NodalSkeletonMultiBand import NodalSkeletonMultiBand

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
Y = sp.Symbol("Y")

H = H_Co2MnGa_TB6_sympy()

ske = NodalSkeletonMultiBand(
    H,
    k_symbols=(kx, ky, kz),
    span=((0, 2 * sp.pi), (0, 2 * sp.pi), (0, 2 * sp.pi)),
    dimension=200,
    band_pair=(0, 1),
    gap_tol=0.5,
    force_small_edge_contraction=True,
)

graph = ske.skeleton_graph(simplify=True)
pd = PDCode(graph)
pd_code = pd.compute(rotation_angles=(60, 60, 130))
upsilon = pd.compute_yamada(Y, normalize=True)

print(f"pd_code = {pd_code}")
print(f"Upsilon(G; Y) = {sp.expand(upsilon)}")
```

Output:

```text
pd_code = V[0,4,2];V[6,3,7];V[7,8,10];V[12,14,6];V[13,1,16];V[5,17,9];V[16,17,18];V[15,19,11];X[8,11,9,10];X[1,13,0,12];X[2,15,3,14];X[4,18,5,19]
Upsilon(G; Y) = -Y**10 + 2*Y**9 - 7*Y**8 + 5*Y**7 - 14*Y**6 + 6*Y**5 - 14*Y**4 + 5*Y**3 - 7*Y**2 + 2*Y - 1
```

The first two regimes are the nontrivial single-cell graph outputs used for
Yamada evaluation. The third regime is the later boundary-wise case. The
corresponding transition panel should be documented only after the material
plotting adapter is available as package API.

```{math}
\begin{aligned}
\Upsilon(0.25<E<1.15,\partial F) ={}&
-Y^{10}+2Y^9-7Y^8+5Y^7-14Y^6+6Y^5\\
&-14Y^4+5Y^3-7Y^2+2Y-1.
\end{aligned}
```

For the lower-energy single-cell regime, the Yamada set contains two spherical
components and one knotted component:

```{math}
\Upsilon(E<0.25,\partial F)=
\left\{
-1,\,
-Y^4-3Y^3-5Y^2-4Y-2,\,
-1
\right\}.
```

The material fingerprint is therefore not just a single polynomial; it is a
sequence of Yamada sets across energy windows.

## Multiband Adapter Status

The remaining multiband notebook figures should not be presented as public
tutorial figures until the material constructor is promoted into
`knotted_graph.applications`. The reusable downstream pattern is already shown
above for each material: Hamiltonian, surface extraction, graph extraction,
projection or PD code, and $\Upsilon(G;Y)$ / $\Upsilon_{\partial F}$. Once the
adapter is public, each material page should add one code block per plotted
surface or graph, followed immediately by the exact output of that code.
