from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UG = ROOT / "User_guide"

YAMADA_MARKERS = (
    "compute_yamada_polynomial(",
    "compute_graph_yamada_polynomial(",
    "Yamada(",
)

PLOTTING_ONLY: set[Path] = set()

BACKEND_PROVENANCE_PAIRS = (
    ("native_available", "native_import_error"),
    ("native_factorized_available", "factorized_import_error"),
)

SOURCE_OVERRIDE_FRAGMENTS = (
    "sys.path.insert(0, str(SRC_ROOT))",
    "sys.path.insert(0,str(SRC_ROOT))",
    "sys.path.insert(0, str(SRC))",
    "sys.path.insert(0,str(SRC))",
    "env[\"PYTHONPATH\"] = str(SRC)",
    "env['PYTHONPATH'] = str(SRC)",
)


def text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_text(cell, value):
    value = value.rstrip("\n")
    if text(cell).rstrip("\n") == value:
        return
    cell["source"] = [line + "\n" for line in value.splitlines()]


def upsert_markdown_after(nb, anchor_id, cell_id, value):
    """Update or insert one explanatory Markdown cell after a stable anchor."""
    matches = [cell for cell in nb["cells"] if cell.get("id") == cell_id]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate notebook cell id: {cell_id}")
    if matches:
        set_text(matches[0], value)
        return
    for index, cell in enumerate(nb["cells"]):
        if cell.get("id") == anchor_id:
            new_cell = {
                "cell_type": "markdown",
                "id": cell_id,
                "metadata": {},
                "source": [],
            }
            set_text(new_cell, value)
            nb["cells"].insert(index + 1, new_cell)
            return
    raise RuntimeError(f"notebook anchor cell was not found: {anchor_id}")


def ensure_unique_cell_ids(path, nb):
    """Assign stable IDs to missing/duplicate cells for modern nbformat readers."""
    used = set()
    stem = path.stem.replace("_", "-")
    for index, cell in enumerate(nb.get("cells", [])):
        cell_id = cell.get("id")
        if not cell_id or cell_id in used:
            cell_id = f"kg-{stem}-{index:03d}"[:64]
            suffix = 1
            while cell_id in used:
                tail = f"-{suffix}"
                cell_id = f"kg-{stem}-{index:03d}"[: 64 - len(tail)] + tail
                suffix += 1
            cell["id"] = cell_id
        used.add(cell_id)


def clear_transient_execution_state(nb):
    """Keep notebooks portable; accepted results live in tracked result files."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def find_root_block(var="PROJECT_ROOT"):
    return f'''{var} = Path.cwd().resolve()
while {var} != {var}.parent and not ({var} / "pyproject.toml").exists():
    {var} = {var}.parent
if not ({var} / "pyproject.toml").exists():
    raise RuntimeError(
        "Could not locate the KnottedGraph repository root. "
        "Run this notebook from inside the repository checkout."
    )'''


def backend_block():
    return '''from pathlib import Path
import sys

import knotted_graph
from knotted_graph.invariants.yamada.native import (
    native_available,
    native_import_error,
)

print("Python executable:", sys.executable)
print("KnottedGraph:", Path(knotted_graph.__file__).resolve())
print("Native Yamada backend:", native_available())
print("Native import error:", native_import_error())'''


def installed_package_block(*, report_backend=False):
    block = '''if importlib.util.find_spec("knotted_graph") is None:
    raise RuntimeError(
        "KnottedGraph is not installed in this notebook kernel. "
        "Install this checkout in an isolated environment, restart Jupyter, "
        "and select that environment as the kernel."
    )

missing_api = [
    name
    for name in ("knotted_graph.core", "knotted_graph.projection")
    if importlib.util.find_spec(name) is None
]
if missing_api:
    raise RuntimeError(
        "The installed knotted_graph package does not provide the current API: "
        f"{', '.join(missing_api)}. Install this repository checkout instead of "
        "the legacy PyPI 0.1.2 release."
    )

import knotted_graph

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "knottedgraph-mpl"),
)

print(f"KnottedGraph version: {knotted_graph.__version__}")
print("Python executable:", sys.executable)
print("KnottedGraph:", Path(knotted_graph.__file__).resolve())'''
    if report_backend:
        block += '''

from knotted_graph.invariants.yamada.native import (
    native_available,
    native_import_error,
)

print("Native Yamada backend:", native_available())
print("Native import error:", native_import_error())'''
    return block


def replace_setup_source(source):
    starts = [
        'PROJECT_ROOT = Path.cwd().resolve()\nwhile (\n    not (PROJECT_ROOT / "src").exists()',
        'PROJECT_ROOT = Path.cwd()\nwhile not (PROJECT_ROOT / "src").exists()',
    ]
    if not any(s in source for s in starts):
        return source

    prefix = source.split("PROJECT_ROOT =", 1)[0]
    tail = ""
    marker = 'os.environ.setdefault('
    if marker in source:
        tail = marker + source.split(marker, 1)[1]
        tail = tail.replace('print("project paths configured")\n', '')
    return prefix + find_root_block() + "\n\nDOC_ROOT = PROJECT_ROOT / \"doc\"\n\n" + tail


def ensure_backend_in_import_cell(nb):
    if has_backend_provenance(nb):
        return
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        s = text(c)
        if "from knotted_graph" in s or "import knotted_graph" in s:
            insertion = backend_block() + "\n\n"
            pos = s.find("from knotted_graph")
            if pos < 0:
                pos = s.find("import knotted_graph")
            s = s[:pos] + insertion + s[pos:]
            set_text(c, s)
            return
    raise RuntimeError("Yamada notebook has no KnottedGraph import cell")


def repair_getting_started(nb):
    # Keep an already-normalized notebook current without rewriting its code cells.
    for c in nb["cells"]:
        if c.get("cell_type") != "markdown":
            continue
        s = text(c)
        s = s.replace(
            'python -m pip install -e ".[notebook]"',
            'python -m pip install -e ".[notebook,viz]"',
        )
        s = s.replace(
            "python -m pip install knotted_graph",
            'python -m pip install "knotted_graph[notebook,viz]"',
        )
        set_text(c, s)

    install = '''## 1.1 Install KnottedGraph

### Recommended: source checkout in an isolated environment

From the repository root (the directory containing `pyproject.toml`), create and
activate a virtual environment, then install this checkout in editable mode with the
Jupyter and visualization extras used by this notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
python -m pip install -U pip
python -m pip install -e ".[notebook,viz]"
```

On macOS, Homebrew Python may reject a system-wide `pip install` with
`externally-managed-environment` (PEP 668). **Do not use `--break-system-packages`**
for this workflow; use the virtual environment above.

For all optional application and benchmark dependencies, use:

```bash
python -m pip install -e ".[all]"
```

The indexed PyPI package is currently the legacy 0.1.2 API and cannot run this
0.2 development notebook. Until 0.2 is released, install a pinned source checkout
as shown above rather than using an unpinned `pip install knotted_graph`.

After installing from source, restart Jupyter and select the kernel belonging to
the same environment. If needed, register it explicitly:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name knottedgraph --display-name "KnottedGraph"
```

The native C++ Yamada backend is optional for correctness but strongly recommended
for performance. The verification cell below reports whether it is active.'''

    verify_md = '''## 1.2 Verify the installation

Run this cell **before the tutorial**. It checks the active Python interpreter,
where `knotted_graph` is imported from, core dependencies, and the compiled Yamada
backend. An editable install may correctly report a Python file under `src/`; what
matters is that the environment was installed with `pip -e` and the native extension
is discoverable through that installation.

For a performance-ready source installation, the desired backend result is:

```text
Native Yamada backend: True
Native import error: None
```

If the backend is `False`, Yamada results remain exact through the Python fallback,
but larger calculations can be substantially slower.'''

    verify_code = '''from pathlib import Path
import importlib.util
import os
import sys
import tempfile

''' + installed_package_block(report_backend=True) + '''

print("\\nDependencies:")
for package in ["numpy", "networkx", "sympy", "shapely", "matplotlib", "plotly", "pyvista"]:
    print(f"{package:12s}: {importlib.util.find_spec(package) is not None}")

if native_available():
    import knotted_graph.invariants.yamada._yamada_native as _yamada_native
    print("Native extension:", Path(_yamada_native.__file__).resolve())
else:
    print("WARNING: exact Python Yamada fallback is active; high-crossing calculations may be slow.")'''

    for c in nb["cells"]:
        if c.get("cell_type") == "markdown" and text(c).startswith("## 1.1 Install"):
            set_text(c, install)
        elif c.get("cell_type") == "markdown" and text(c).startswith("## 1.2 Verify"):
            set_text(c, verify_md)
        elif c.get("id") == "b9105f36":
            # This cell is deliberately reset to the canonical verification cell.
            set_text(c, verify_code)


def repair_user_guide_map(nb):
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        source = text(cell)
        source = source.replace("| Notebook | Use it when... |", "| Resource | Use it when... |")
        input_row = (
            "| **[Input handling guide](../doc/user_guide/input_adapters.md)** | "
            "you have coordinates, PDB/mmCIF, GRO/LAMMPS, paired spatial CSV, "
            "or a supported surface mesh |\n"
        )
        marker = (
            "| **[01 - Getting Started](01_getting_started.ipynb)** | "
            "you want the shortest complete example |\n"
        )
        if input_row not in source and marker in source:
            source = source.replace(marker, marker + input_row)
        source = source.replace(
            "| **[Yamada Formula Discovery](applications/05_yamada_formula_discovery.ipynb)** | you want exact datasets and symbolic checks for conjecturing Yamada formulas |",
            "| **[Yamada Formula Discovery](applications/05_yamada_formula_discovery.ipynb)** | you are reproducing an advanced exact-data and held-out symbolic study, after learning the single-graph API |",
        )
        source = source.replace(
            "| **[Hamiltonian Yamada Phase Maps](applications/06_hamiltonian_yamada_phase_maps.ipynb)** | you want parameter-space regions, knot transitions, and their representative Hamiltonian skeletons |",
            "| **[Hamiltonian Yamada Phase Maps](applications/06_hamiltonian_yamada_phase_maps.ipynb)** | you are running a cached, compute-intensive two-parameter Hamiltonian study |",
        )
        first_use = "If this is your first use of the library, start with **01 - Getting Started**."
        advanced_note = (
            "Formula Discovery and Hamiltonian Phase Maps are advanced "
            "reproduction notebooks, not the next step after installation."
        )
        if advanced_note not in source:
            source = source.replace(first_use, first_use + " " + advanced_note)
        external_heading = "### I have external coordinate or structure data"
        surface_heading = "### I have a 3D surface or physical model"
        if external_heading not in source and surface_heading in source:
            source = source.replace(
                surface_heading,
                external_heading
                + "\nRead the **Input handling guide** first. It distinguishes "
                "implemented public adapters from application-specific conversions.\n\n"
                + surface_heading,
            )
        set_text(cell, source)



def repair_primary_tutorial_setup(nb):
    """Use the active installed/editable package without prepending raw ``src``."""
    if any(
        "KnottedGraph version:" in text(c) and "SRC_ROOT" not in text(c)
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    ):
        return
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        s = text(c)
        if "installation_mode" not in s or "SRC_ROOT" not in s:
            continue
        canonical = '''from pathlib import Path
import importlib.util
import os
import sys
import tempfile

''' + installed_package_block(report_backend=False) + '''

print("\\nOptional dependencies:")
for package in ["numpy", "networkx", "sympy", "plotly", "matplotlib", "pyvista"]:
    print(f"{package:10s} = {importlib.util.find_spec(package) is not None}")'''
        set_text(c, canonical)
        return
    raise RuntimeError("Primary tutorial has no recognizable environment setup cell")


def repair_sanity(nb):
    for c in nb["cells"]:
        if c.get("id") != "published":
            continue
        s = '''from pathlib import Path
import os
import subprocess
import sys

''' + find_root_block("ROOT") + '''

''' + backend_block() + '''

script = ROOT / "dev" / "run_yamada_sanity_checks.py"
env = dict(os.environ)
env.pop("PYTHONPATH", None)
proc = subprocess.run(
    [sys.executable, str(script)],
    cwd=ROOT,
    env=env,
    text=True,
    capture_output=True,
)
print(proc.stdout)
if proc.returncode:
    raise RuntimeError(
        f"Sanity checks failed.\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}"
    )
assert "PASS: all published/independent Yamada sanity checks succeeded." in proc.stdout
'''
        set_text(c, s)


def repair_regression(nb):
    # Historical comparison is the only intentional source-worktree exception.
    for c in nb["cells"]:
        if c.get("id") == "setup":
            s = text(c)
            if "Current notebook environment" not in s:
                insert = '''\nimport knotted_graph\nfrom knotted_graph.invariants.yamada.native import native_available, native_import_error\nprint('Current notebook environment:', sys.executable)\nprint('Current KnottedGraph:', Path(knotted_graph.__file__).resolve())\nprint('Native Yamada backend:', native_available())\nprint('Native import error:', native_import_error())\nprint('NOTE: the historical regression subprocesses intentionally import each detached source worktree; this notebook is a correctness regression, not a performance benchmark.')\n'''
                s = s.replace(
                    "DRIVER = ROOT / 'dev' / 'application_yamada_regression.py'\n",
                    "DRIVER = ROOT / 'dev' / 'application_yamada_regression.py'\n" + insert,
                )
            set_text(c, s)


def repair_thick_handlebody(nb):
    for c in nb["cells"]:
        if c.get("id") != "af32d418":
            continue
        s = text(c)
        s = s.replace(
            '''    sibling = candidate / "KnottedGraph_1earlier"
    if _is_knotted_graph_repo(sibling):
        ROOT = sibling
        break
''',
            "",
        )
        s = s.replace(
            '''src_path = ROOT / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

''',
            "",
        )
        set_text(c, s)
        return
    raise RuntimeError("Thick-handlebody notebook setup cell was not found")


def repair_hamiltonian_phase_maps(nb):
    introduction = r'''# Hamiltonian Yamada Phase Maps

> **Role:** advanced, compute-intensive application and publication-reproduction
> notebook. It is not a beginner tutorial and is not executed in the ordinary
> pull-request notebook matrix.

Use this notebook when your input is a family of in-memory nodal Hamiltonians or
Bloch-vector fields and you want to turn a two-parameter scan into an auditable
Yamada-topology phase map. Install the `nodal`, `viz`, and `notebook` extras and
use a compute environment for full regeneration.

The production route is

\[
(\lambda,\Gamma) \longrightarrow H(\mathbf{k};\lambda,\Gamma)
\longrightarrow \text{filled exceptional region}
\longrightarrow \text{skeleton} \longrightarrow G\subset\mathbb R^3
\longrightarrow \Upsilon(G;Y).
\]

The configured study evaluates five transitions on up to 60 lambda samples and
50 candidate Gamma samples with a $120^3$ volume per evaluated cell. Row caches
reduce repeated work, but a clean full run remains a substantial workload. Do
not run all cells on a login node.

The notebook keeps audit steps visible: per-cell error records, connected-region
stabilization, classic and contraction-equivalent classifications, endpoint
checks, static figures, and an interactive Plotly view of representative
exceptional surfaces and their graph skeletons. Generated outputs live below
`User_guide/applications/results/06_hamiltonian_yamada_phase_maps`; execution
outputs are deliberately not stored in the notebook JSON.

For a guided explanation and a directly viewable interactive artifact, read the
website page **Hamiltonian Yamada Phase Maps** before regenerating this notebook.'''
    set_text(nb["cells"][0], introduction)

    upsert_markdown_after(
        nb,
        "c2a4aff8",
        "phase-map-configuration-guide",
        r'''## Configuration, cost, and provenance

The setup cell fixes numerical-thread limits, resolves the repository, reports
the imported package and native Yamada backend, defines output/cache locations,
and declares the finite sampling grid. Inspect its printed values before any
scan. In particular:

- `PHASE_LAMBDAS` and `PHASE_GAMMAS` define the finite grid; changing them changes
  the phase-map evidence.
- `SKELETON_DIMENSION` controls the three-dimensional voxel resolution and is a
  dominant time/memory parameter.
- `N_JOBS` controls row-level concurrency, while each Yamada calculation uses an
  explicit single-worker setting to avoid nested oversubscription.
- cached rows are accepted only through a digest of the transition, axes,
  resolution, and retry policy.

A colored cell is therefore a recorded finite-resolution calculation, not an
analytic continuum phase boundary.''',
    )
    upsert_markdown_after(
        nb,
        "88b7e0e8",
        "phase-map-record-guide",
        r'''## What one phase-map record means

Each successful cell stores the sampled parameters, graph size, connected
components, cycle rank, degree sequence, total edge-geometry samples, exact
Yamada expression, and a stable phase signature. A failed extraction stores an
error instead of silently becoming a zero invariant.

`make_yamada_phase_map(...)` is the reusable application API. The longer code
below adds the publication-specific extraction retries, cache layout,
stabilization, figures, and validation rules. Keep that distinction in mind when
copying code into a new project.''',
    )
    upsert_markdown_after(
        nb,
        "83faf9d8",
        "phase-map-audit-guide",
        r'''## Audit the records before plotting

The scan summary must be read before the figures. Check the number of evaluated
cells, data source used for each cell, explicit errors, terminal Gamma retained
for each transition, and endpoint distinctions. The following helper cells write
stable CSV tables and verify that leaves, failed cells, or unstable small regions
have not been hidden by plotting logic.''',
    )
    upsert_markdown_after(
        nb,
        "34f9946d",
        "phase-map-figure-guide",
        r'''## Static phase figures and two classifications

The **classic** map labels stabilized connected regions by their exact Yamada
signature. The **up to contraction moves** map additionally groups selected
representative cores under the notebook's stated contraction convention. These
are different questions: a merged contraction class does not assert literal
equality of all classic signatures.

Read phase boundaries as nearest-neighbor boundaries on the sampled grid. A
single-cell island, a boundary-touching surface, or a cell produced only after a
fallback requires inspection rather than automatic physical interpretation.''',
    )

    for c in nb["cells"]:
        if c.get("id") != "c2a4aff8":
            continue
        s = text(c)
        s = s.replace(
            '"Could not find the KnottedGraph_1earlier repository root above "',
            '"Could not locate the KnottedGraph repository root above "',
        )
        s = s.replace('sys.path.insert(0, str(ROOT / "src"))\n\n', "")
        s = s.replace(
            '''kg_file = Path(knotted_graph.__file__).resolve()
if not kg_file.is_relative_to(ROOT):
    raise RuntimeError(f"Imported knotted_graph from outside this checkout: {kg_file}")
''',
            '''kg_file = Path(knotted_graph.__file__).resolve()
''',
        )
        set_text(c, s)
        if not has_backend_provenance(nb):
            ensure_backend_in_import_cell(nb)
        break
    else:
        raise RuntimeError("Hamiltonian phase-map setup cell was not found")

    for c in nb["cells"]:
        if c.get("id") == "interactive-plotly-qa-markdown":
            set_text(
                c,
                '''## Interactive Plotly geometry QA

The stable two-dimensional phase partitions are clickable. Select a mode and a
transition, then click a region to load one representative exceptional surface
and the corresponding simplified Yamada skeleton.

Use the view as a geometry audit, not only as decoration: confirm that the
surface, black skeleton, red graph vertices, component counts, cycle rank, and
displayed polynomial describe the same representative record. The HTML is a
self-contained data snapshot except for its pinned Plotly CDN script.''',
            )
        elif c.get("id") == "finalnodal":
            set_text(
                c,
                '''## Final phase-map verification

The final cell checks package provenance, transition coverage, error totals, and
the existence of the generated interactive artifact. A passing final cell means
the configured finite-grid workflow completed its stated checks; it does not
replace resolution convergence or an analytic phase-boundary proof.''',
            )
        elif c.get("id") == "finalcheck":
            s = text(c)
            s = s.replace(
                'print("expected source root:", (ROOT / "src" / "knotted_graph").resolve())\n',
                'print("repository root:", ROOT)\n',
            )
            s = s.replace(
                'assert Path(knotted_graph.__file__).resolve().is_relative_to((ROOT / "src").resolve())\n',
                '''assert knotted_graph.__version__.startswith("0.2"), (
    "This reproduction notebook requires the current 0.2 development API."
)
''',
            )
            set_text(c, s)


def _relax_formula_discovery_guard(source):
    source = source.replace(
        'EXPECTED_BRANCH = "integration/arbitrary-knot-fields-final-audit"',
        '''AUDITED_SOURCE_COMMIT = "49e34e47ca5c182f55ef5f0ea0906220df59befb"
STRICT_PUBLICATION_REGENERATION = (
    os.environ.get("KNOTTEDGRAPH_STRICT_PUBLICATION_REGENERATION", "0") == "1"
)''',
    )
    if "import warnings\n" not in source:
        source = source.replace("import time\n", "import time\nimport warnings\n", 1)

    branch_checks = (
        '''if CURRENT_BRANCH != EXPECTED_BRANCH:
    raise RuntimeError(
        f"This notebook is audited for {EXPECTED_BRANCH!r}, but HEAD is on {CURRENT_BRANCH!r}."
    )''',
        '''if CURRENT_BRANCH != EXPECTED_BRANCH:
    raise RuntimeError(
        f"This experiment is audited for {EXPECTED_BRANCH!r}, "
        f"but HEAD is on {CURRENT_BRANCH!r}."
    )''',
        '''if CURRENT_BRANCH != EXPECTED_BRANCH:
    raise RuntimeError(
        f"Expected branch {EXPECTED_BRANCH!r}; found {CURRENT_BRANCH!r}."
    )''',
    )
    revision_policy = '''AUDITED_SOURCE_PRESENT = subprocess.run(
    ["git", "merge-base", "--is-ancestor", AUDITED_SOURCE_COMMIT, "HEAD"],
    cwd=ROOT,
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0

if STRICT_PUBLICATION_REGENERATION and not AUDITED_SOURCE_PRESENT:
    raise RuntimeError(
        "Strict publication regeneration requires audited source commit "
        f"{AUDITED_SOURCE_COMMIT}. Current HEAD is {GIT_COMMIT}."
    )
if not AUDITED_SOURCE_PRESENT:
    warnings.warn(
        "The audited source commit is not an ancestor of this checkout. "
        "Exploration may continue, but do not treat regenerated tables as "
        "publication-audited outputs.",
        RuntimeWarning,
    )'''
    for old in branch_checks:
        source = source.replace(old, revision_policy)

    source = source.replace(
        '''if library_status:
    raise RuntimeError(
        "The KnottedGraph source tree has uncommitted changes. "
        "Commit or stash them before generating a publication dataset:\n"
        + library_status
    )''',
        '''if library_status:
    message = (
        "The KnottedGraph source tree has uncommitted changes. "
        "Publication regeneration requires a clean source tree:\n" + library_status
    )
    if STRICT_PUBLICATION_REGENERATION:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning)''',
    )
    source = source.replace(
        '''if library_status:
    raise RuntimeError(
        "The KnottedGraph source tree has uncommitted changes. "
        "Commit or stash them before generating an auditable dataset:\n"
        + library_status
    )''',
        '''if library_status:
    message = (
        "The KnottedGraph source tree has uncommitted changes. "
        "Publication regeneration requires a clean source tree:\n" + library_status
    )
    if STRICT_PUBLICATION_REGENERATION:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning)''',
    )
    source = source.replace(
        '''if library_status:
    raise RuntimeError(
        "Commit/stash changes under src/knotted_graph before an audited run:\n"
        + library_status
    )''',
        '''if library_status:
    message = (
        "Publication regeneration requires a clean src/knotted_graph tree:\n"
        + library_status
    )
    if STRICT_PUBLICATION_REGENERATION:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning)''',
    )

    package_checks = (
        '''if ROOT not in KG_FILE.parents:
    raise RuntimeError(
        "Python imported knotted_graph from outside this checkout.\n"
        f"Repository root: {ROOT}\nImported package: {KG_FILE}"
    )''',
        '''if ROOT not in KG_FILE.parents:
    raise RuntimeError(
        f"Imported knotted_graph outside checkout: {KG_FILE}"
    )''',
    )
    package_policy = '''if ROOT not in KG_FILE.parents:
    message = (
        "Python imported knotted_graph from outside this checkout: "
        f"{KG_FILE}. Exploration may continue, but strict publication "
        "regeneration requires the audited editable checkout."
    )
    if STRICT_PUBLICATION_REGENERATION:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning)'''
    for old in package_checks:
        source = source.replace(old, package_policy)

    native_checks = (
        '''if not native_factorized_available():
    raise RuntimeError(
        "The optimized native factorized Yamada backend is unavailable. "
        "Rebuild this checkout before running the dataset.\n"
        f"Import error: {factorized_import_error()!r}"
    )''',
        '''if not native_factorized_available():
    raise RuntimeError(
        "Optimized factorized Yamada backend unavailable: "
        f"{factorized_import_error()!r}"
    )''',
    )
    native_policy = '''if not native_factorized_available():
    message = (
        "The optimized factorized Yamada backend is unavailable. "
        f"Heavy formula-discovery cells cannot run: {factorized_import_error()!r}"
    )
    if STRICT_PUBLICATION_REGENERATION:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning)'''
    for old in native_checks:
        source = source.replace(old, native_policy)
    return source


def repair_formula_discovery(nb):
    introduction = r'''# Yamada Formula Discovery Applications

> **Role:** advanced exact-computation and publication-reproduction notebook.
> Learn the single-graph Yamada API first; do not use **Run All** as a first test
> of the package.

Use this notebook when you want to use KnottedGraph as an exact-computation
engine for discovering and testing Yamada-polynomial formulae across
parameterized spatial-graph families.

The notebook asks one practical research question:

> How much information about a local motif word survives in the exact Yamada
> polynomial of the resulting spatial graph?

It progresses through three regimes:

\[
\boxed{\text{homogeneous repetition}
\longrightarrow \text{Abelian count-only mixing}
\longrightarrow \text{non-Abelian order-sensitive mixing}.}
\]

Each part defines a graph family, constructs embedded graphs, evaluates exact
Laurent polynomials, exports only the data needed for fitting or audit, freezes a
candidate identity, and then runs separate held-out checks. Outputs live under
`User_guide/applications/results/05_yamada_formula_discovery`; execution outputs
are deliberately not committed in the notebook.

The notebook is browsable on an ordinary review branch. Set
`KNOTTEDGRAPH_STRICT_PUBLICATION_REGENERATION=1` only for an audited clean
regeneration with the required source ancestry and factorized native backend.
Held-out exact agreement is strong computational evidence, but it is not a
substitute for a mathematical proof.'''
    set_text(nb["cells"][0], introduction)
    for c in nb["cells"]:
        if c.get("cell_type") == "code" and "EXPECTED_BRANCH" in text(c):
            set_text(c, _relax_formula_discovery_guard(text(c)))
        elif c.get("cell_type") == "markdown":
            set_text(c, text(c).replace(r"\$$0.35em]", r"\\[0.35em]"))


def repair_generic(nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            set_text(c, replace_setup_source(text(c)))


def has_yamada(nb):
    return any(
        any(m in text(c) for m in YAMADA_MARKERS)
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )


def has_backend_provenance(nb):
    joined = "\n".join(
        text(c) for c in nb["cells"] if c.get("cell_type") == "code"
    )
    return any(all(name in joined for name in pair) for pair in BACKEND_PROVENANCE_PAIRS)


def validate(path, nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            s = text(c)
            parseable = "\n".join(
                line for line in s.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            ast.parse(parseable or "pass")
            if path.name != "02_application_regression_checks.ipynb":
                for frag in SOURCE_OVERRIDE_FRAGMENTS:
                    if frag in s:
                        raise AssertionError(f"{path}: stale source override: {frag}")
    if has_yamada(nb) and path not in PLOTTING_ONLY:
        if not has_backend_provenance(nb):
            raise AssertionError(
                f"{path}: Yamada evaluation lacks native-backend provenance"
            )


def process(path, check=False):
    nb = json.loads(path.read_text())
    before = json.dumps(nb, sort_keys=True)

    ensure_unique_cell_ids(path, nb)

    if path.name == "00_user_guide.ipynb":
        repair_user_guide_map(nb)
    elif path.name == "01_getting_started.ipynb":
        repair_getting_started(nb)
    elif path.name in {"02_core_workflows.ipynb", "03_advanced_and_reproduction.ipynb"}:
        repair_primary_tutorial_setup(nb)
    elif path.name == "01_yamada_sanity_checks.ipynb":
        repair_sanity(nb)
    elif path.name == "02_application_regression_checks.ipynb":
        repair_regression(nb)
    elif path.name == "04_thick_handlebody_validation.ipynb":
        repair_thick_handlebody(nb)
    elif path.name == "06_hamiltonian_yamada_phase_maps.ipynb":
        repair_hamiltonian_phase_maps(nb)
    elif path.name == "05_yamada_formula_discovery.ipynb":
        repair_formula_discovery(nb)
    else:
        repair_generic(nb)
        if (
            has_yamada(nb)
            and path not in PLOTTING_ONLY
            and not has_backend_provenance(nb)
        ):
            ensure_backend_in_import_cell(nb)

    clear_transient_execution_state(nb)

    validate(path, nb)
    after = json.dumps(nb, sort_keys=True)
    if check:
        if before != after:
            raise AssertionError(
                f"{path} is not normalized; run this script without --check"
            )
    elif before != after:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("updated", path.relative_to(ROOT))
    else:
        print("ok     ", path.relative_to(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    paths = sorted(UG.rglob("*.ipynb"))
    for path in paths:
        process(path, check=args.check)
    print(f"validated {len(paths)} notebooks")


if __name__ == "__main__":
    main()
