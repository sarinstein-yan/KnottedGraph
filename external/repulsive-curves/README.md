# Repulsive Curves
Christopher Yu, Henrik Schumacher, Keenan Crane
ACM Transactions on Graphics 2020 (accepted)

## Quick setup instructions

First, clone the project and all its dependencies:
```
git clone --recursive https://github.com/icethrush/repulsive-curves.git
```

If the recursive flag was not used to clone, then one can also get the dependencies by running:
```
git submodule update --init --recursive
```

From there, the project can be built using CMake.
```
cd repulsive-curves
mkdir build
cd build
cmake ..
make -j4
```
We highly recommend using Clang to build the project. Building with GCC/G++ is possible, but will require a different set of warnings to be suppressed.

The code can then be run:
```
./bin/rcurves_app path/to/scene.txt
```

For best performance, you should make sure that OpenMP is supported on your system.

Note that the file `scene.txt` has a particular format that describes where to find the curve data, as well as what constraints will be used. See `scenes/FORMATS.md` for details.

## Headless graph layout

This repo now also includes a non-interactive executable, a shared library entry point, and Python utilities for 3D spatial graph layout.

Build the solver targets:
```
cmake -S . -B build
cmake --build build -j 6 --target rcurves_headless rcurves_shared rcurves_polyscope_render
```

## Python usage

If you only want to call the solver from Python, you do not need to modify the C++ code. In that case you only need:
- a Python environment
- the Python packages listed below
- prebuilt solver binaries, or a local C++ build in `build/bin`

Install the base Python dependencies:
```
pip install -r python/requirements.txt
```

Optional packages:
```
pip install plotly
pip install kaleido
```
Use `plotly` for interactive HTML rendering. Use `kaleido` if you want Plotly figures exported directly to PNG.

Run a graph layout from an edge list / JSON / GraphML file:
```
python python/repulsive_graph_layout.py your_graph.edgelist \
  --workspace build/run1 \
  --steps 50 \
  --samples-per-edge 16 \
  --solver build/bin/rcurves_headless.exe
```

This `subprocess` path writes:
- `curve.obj`: initial poly-curve network
- `scene.txt`: solver scene
- `final.obj`: optimized curve network
- `layout.json`: structured output with final node coordinates and per-edge 3D polylines

Or call the shared library directly from Python via `ctypes`:
```
python python/repulsive_graph_layout.py your_graph.edgelist \
  --engine ctypes \
  --solver build/bin/librcurves_shared.dll \
  --workspace build/run1_native \
  --steps 50
```

This `ctypes` path writes:
- `final.obj`: optimized curve network
- `metadata.json`: bookkeeping data for the graph-to-curve mapping
- `layout.json`: structured output with final node coordinates and per-edge 3D polylines

There are also built-in examples for `K5` and `K3,3`:
```
python python/run_examples.py --solver build/bin/rcurves_headless.exe --workspace build/examples
```

Batch-process many graphs:
```
python python/batch_layout.py path/to/graphs_dir \
  --recursive \
  --workspace build/batch \
  --solver build/bin/rcurves_headless.exe \
  --steps 25
```

To batch through the shared library instead of subprocesses:
```
python python/batch_layout.py path/to/graphs_dir \
  --recursive \
  --engine ctypes \
  --workspace build/batch_native \
  --solver build/bin/librcurves_shared.dll \
  --steps 25
```

This writes one subdirectory per graph plus a `summary.csv`.

Render a static figure from `layout.json`:
```
python python/render_layout.py build/examples/k5/layout.json \
  --output build/examples/k5/layout.png
```

Render a repo-native screenshot using the same Polyscope material pipeline as the original interactive viewer:
```
python python/render_layout_repo_native.py build/examples/k5/layout.json \
  --renderer build/bin/rcurves_polyscope_render.exe \
  --output build/examples/k5/repo_native.png \
  --palette k5
```

Render an interactive HTML figure:
```
python python/render_layout_material.py build/examples/k5/layout.json \
  --output build/examples/k5/material.html \
  --palette k5
```

Notes:
- On the current Windows/GCC setup, the headless solver defaults to exact gradients and keeps Barnes-Hut off for stability.
- The Python wrapper initializes each graph edge as a slightly bent 3D polyline rather than a perfectly straight segment, which gives more stable starts for non-planar graph drawing.

## C++ development dependencies

If you plan to modify the solver, renderer, or bindings, install a full C++ toolchain. The repository vendors most library dependencies under `deps/`, so you do not need to install Eigen, Polyscope, or geometry-central separately, but you do need the standard build tools.

Required tools:
- `git`
- `cmake` 3.10 or newer
- a C++ compiler with OpenMP support
- Python is optional for the pure C++ targets, but recommended if you also want to test the wrappers

Typical setup by platform:

Ubuntu / Debian:
```
sudo apt update
sudo apt install git cmake build-essential clang libomp-dev python3 python3-pip
```

macOS (Homebrew):
```
brew install cmake llvm libomp python
```

Windows:
- install `Git`
- install `CMake`
- install one supported compiler toolchain:
  - Visual Studio 2022 Build Tools with Desktop C++
  - LLVM/Clang
  - MinGW-w64

Clone the repository with submodules:
```
git clone --recursive https://github.com/Kaonashi-12/repulsive-curves-python.git
cd repulsive-curves-python
```

If you forgot `--recursive`, fetch the submodules afterwards:
```
git submodule update --init --recursive
```

Build everything needed for graph layout development:
```
cmake -S . -B build
cmake --build build -j 6 --target rcurves_app rcurves_headless rcurves_shared rcurves_polyscope_render
```

Main binaries:
- `build/bin/rcurves_app`: original interactive viewer
- `build/bin/rcurves_headless`: non-interactive scene runner
- `build/bin/rcurves_polyscope_render`: repo-native screenshot renderer
- `build/bin/librcurves_shared.dll` on Windows, or the corresponding shared library on Linux/macOS: shared library entry point for Python `ctypes`

If you only want to run the Python workflow and already have prebuilt binaries, you can skip the full C++ toolchain installation.

## Using the project

The important options for manipulating curves are all under the "Curve options" panel in the system. These options are:

+ Run TPE: While checked, the system will run the gradient flow of the tangent-point energy.
+ Output frames: If checked, screenshots of every frame of the gradient flow will be saved as PNG images in the `./frames` directory; note that this is relative to the working directory from which the executable is run.
+ Normalize view: If checked, the objects will be visually rescaled to fit within the camera frame every timestep. This rescaling is purely visual and does not affect the flow.
+ Output OBJs: If checked, OBJs of the curve on every frame will be output to the `./objs` directory.
+ Use Sobolev: If checked, our fractional Sobolev preconditioner will be used. If unchecked, the L2 flow is used instead.
+ Use backprojection: If checked, the system will perform a projection step to enforce hard constraints, correctin for drift. If unchecked, no such step is performed.
+ Use Barnes-Hut: If checked, hierarchical Barnes-Hut approximation is used for energy and gradient evaluations. If unchecked, the energy and gradient are evaluated exactly (and slowly).
+ Use multigrid: If checked, multigrid is used to perform linear solves. If unchecked, dense linear solves are performed.

