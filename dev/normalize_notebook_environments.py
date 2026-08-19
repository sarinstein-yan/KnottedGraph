from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UG = ROOT / "User_guide"

YAMADA_MARKERS = (
    "compute_yamada_polynomial(",
    "compute_yamada_polynomial_recursive(",
    "Yamada(",
)

PLOTTING_ONLY = {
    UG / "benchmarks" / "06_paper_scaling_publication_figure.ipynb",
}

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
    cell["source"] = [line + "\n" for line in value.rstrip("\n").splitlines()]


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
    return '''import knotted_graph
from knotted_graph.invariants.yamada.native import (
    native_available,
    native_import_error,
)

print("Python executable:", sys.executable)
print("KnottedGraph:", Path(knotted_graph.__file__).resolve())
print("Native Yamada backend:", native_available())
print("Native import error:", native_import_error())'''


def replace_setup_source(source):
    # Common old setup used by main/application notebooks.
    starts = [
        'PROJECT_ROOT = Path.cwd().resolve()\nwhile (\n    not (PROJECT_ROOT / "src").exists()',
        'PROJECT_ROOT = Path.cwd()\nwhile not (PROJECT_ROOT / "src").exists()',
    ]
    if not any(s in source for s in starts):
        return source

    prefix = source.split("PROJECT_ROOT =", 1)[0]
    # Preserve MPLCONFIGDIR and dependency reporting but remove source injection.
    tail = ""
    marker = 'os.environ.setdefault('
    if marker in source:
        tail = marker + source.split(marker, 1)[1]
        # Strip stale status message if present; a later import cell does the package check.
        tail = tail.replace('print("project paths configured")\n', '')
    return prefix + find_root_block() + "\n\nDOC_ROOT = PROJECT_ROOT / \"doc\"\n\n" + tail


def ensure_backend_in_import_cell(nb):
    if any("native_available" in text(c) for c in nb["cells"] if c.get("cell_type") == "code"):
        return
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        s = text(c)
        if "from knotted_graph" in s or "import knotted_graph" in s:
            insertion = backend_block() + "\n\n"
            # Put provenance before other KnottedGraph imports, after stdlib/third-party imports.
            pos = s.find("from knotted_graph")
            if pos < 0:
                pos = s.find("import knotted_graph")
            s = s[:pos] + insertion + s[pos:]
            set_text(c, s)
            return
    raise RuntimeError("Yamada notebook has no KnottedGraph import cell")


def repair_getting_started(nb):
    install = '''## 1.1 Install KnottedGraph

### Recommended: source checkout in an isolated environment

From the repository root (the directory containing `pyproject.toml`), create and
activate a virtual environment, then install this checkout in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
python -m pip install -U pip
python -m pip install -e ".[notebook]"
```

On macOS, Homebrew Python may reject a system-wide `pip install` with
`externally-managed-environment` (PEP 668). **Do not use `--break-system-packages`**
for this workflow; use the virtual environment above.

For all optional application and benchmark dependencies, use:

```bash
python -m pip install -e ".[all]"
```

For a released PyPI installation instead of a source checkout:

```bash
python -m pip install knotted_graph
```

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

''' + find_root_block() + '''

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "knottedgraph-mpl"),
)

''' + backend_block() + '''

print("\\nDependencies:")
for package in ["numpy", "networkx", "sympy", "shapely", "matplotlib", "plotly", "pyvista"]:
    print(f"{package:12s}: {importlib.util.find_spec(package) is not None}")

if native_available():
    import knotted_graph.invariants.yamada._yamada_native as _yamada_native
    print("Native extension:", Path(_yamada_native.__file__).resolve())
else:
    print("WARNING: exact Python Yamada fallback is active; high-crossing calculations may be slow.")'''

    # Replace old installation markdown and setup markdown/cell.
    for c in nb["cells"]:
        if c.get("cell_type") == "markdown" and "## 1.1 Install the package" in text(c):
            set_text(c, install)
        elif c.get("cell_type") == "markdown" and "## Set up the notebook" in text(c):
            set_text(c, verify_md)
        elif c.get("id") == "b9105f36":
            set_text(c, verify_code)

    # Renumber tutorial headings that followed the old 1.1 setup section.
    # Apply in descending order to avoid 1.2 -> 1.3 -> 1.4 cascading.
    mapping = (
        ("## 1.7 Continue", "## 1.8 Continue"),
        ("## 1.6 Compute", "## 1.7 Compute"),
        ("## 1.5 Choose", "## 1.6 Choose"),
        ("## 1.4 Apply", "## 1.5 Apply"),
        ("## 1.3 Inspect", "## 1.4 Inspect"),
        ("## 1.2 Build", "## 1.3 Build"),
    )
    for c in nb["cells"]:
        if c.get("cell_type") == "markdown":
            s = text(c)
            for a, b in mapping:
                s = s.replace(a, b)
            set_text(c, s)


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


def repair_scaling(nb):
    for c in nb["cells"]:
        if c.get("id") == "setup":
            s = '''from pathlib import Path
import csv, importlib.util, json, os, subprocess, sys
from tqdm.auto import tqdm

''' + find_root_block("ROOT") + '''
DEV = ROOT / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

''' + backend_block() + '''
if not native_available():
    raise RuntimeError(
        "This performance benchmark requires the compiled native Yamada backend. "
        "Install the checkout with `python -m pip install -e .` in the active environment."
    )

try:
    import topoly
except ImportError as exc:
    raise ImportError("Install benchmark dependencies with: python -m pip install -e '.[benchmark]'") from exc

OUT = ROOT / "User_guide" / "benchmarks"
RES = OUT / "results_latest"
FIG = OUT / "figures_latest"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
print("Topoly:", Path(topoly.__file__).resolve())
'''
            set_text(c, s)
        elif c.get("id") == "run":
            s = text(c)
            s = s.replace('env = dict(os.environ)\nenv["PYTHONPATH"] = os.pathsep.join([str(SRC), str(DEV)])\nenv["PYTHONNOUSERSITE"] = "1"\n', 'env = dict(os.environ)\nenv.pop("PYTHONPATH", None)\n')
            set_text(c, s)


def repair_synthetic(nb):
    for c in nb["cells"]:
        if c.get("id") != "setup":
            continue
        s = text(c)
        old = "ROOT=Path.cwd().resolve()\nwhile ROOT!=ROOT.parent and not (ROOT/'pyproject.toml').exists(): ROOT=ROOT.parent\nSRC=ROOT/'src'\nif not (SRC/'knotted_graph').exists(): raise RuntimeError('Run inside the KnottedGraph checkout.')\nif str(SRC) not in sys.path: sys.path.insert(0,str(SRC))\n\nimport knotted_graph\n"
        new = "ROOT=Path.cwd().resolve()\nwhile ROOT!=ROOT.parent and not (ROOT/'pyproject.toml').exists(): ROOT=ROOT.parent\nif not (ROOT/'pyproject.toml').exists(): raise RuntimeError('Run inside the KnottedGraph checkout.')\n\nimport knotted_graph\nfrom knotted_graph.invariants.yamada.native import native_available, native_import_error\nprint('Python executable:', sys.executable)\nprint('KnottedGraph:', Path(knotted_graph.__file__).resolve())\nprint('Native Yamada backend:', native_available())\nprint('Native import error:', native_import_error())\n"
        s = s.replace(old, new)
        s = s.replace("\nkg_path=Path(knotted_graph.__file__).resolve()\nif SRC not in kg_path.parents: raise RuntimeError(f'Stale knotted_graph import: {kg_path}')\n", "\nkg_path=Path(knotted_graph.__file__).resolve()\n")
        set_text(c, s)


def repair_distinct(nb):
    # Already uses installed-package semantics; only add interpreter provenance.
    for c in nb["cells"]:
        if c.get("id") == "imports":
            s = text(c)
            if 'print("Python executable:"' not in s:
                s = s.replace('print("Native Yamada backend:", native_available())', 'print("Python executable:", sys.executable)\nprint("Native Yamada backend:", native_available())')
            set_text(c, s)


def repair_regression(nb):
    # Historical branch comparison intentionally uses source worktrees; make that explicit.
    for c in nb["cells"]:
        if c.get("id") == "setup":
            s = text(c)
            if "Current notebook environment" not in s:
                insert = '''\nimport knotted_graph\nfrom knotted_graph.invariants.yamada.native import native_available, native_import_error\nprint('Current notebook environment:', sys.executable)\nprint('Current KnottedGraph:', Path(knotted_graph.__file__).resolve())\nprint('Native Yamada backend:', native_available())\nprint('Native import error:', native_import_error())\nprint('NOTE: the historical regression subprocesses intentionally import each detached source worktree; this notebook is a correctness regression, not a performance benchmark.')\n'''
                s = s.replace("DRIVER = ROOT / 'dev' / 'application_yamada_regression.py'\n", "DRIVER = ROOT / 'dev' / 'application_yamada_regression.py'\n" + insert)
            set_text(c, s)


def repair_generic(nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            s = replace_setup_source(text(c))
            set_text(c, s)


def has_yamada(nb):
    return any(any(m in text(c) for m in YAMADA_MARKERS) for c in nb["cells"] if c.get("cell_type") == "code")


def validate(path, nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            s = text(c)
            ast.parse(s or "pass")
            if path.name != "02_application_output_regression.ipynb":
                for frag in SOURCE_OVERRIDE_FRAGMENTS:
                    if frag in s:
                        raise AssertionError(f"{path}: stale source override: {frag}")
    if has_yamada(nb) and path not in PLOTTING_ONLY:
        joined = "\n".join(text(c) for c in nb["cells"] if c.get("cell_type") == "code")
        if "native_available" not in joined or "native_import_error" not in joined:
            raise AssertionError(f"{path}: Yamada evaluation lacks native-backend provenance")


def process(path, check=False):
    nb = json.loads(path.read_text())
    before = json.dumps(nb, sort_keys=True)

    if path.name == "01_getting_started.ipynb":
        repair_getting_started(nb)
    elif path.name == "01_sanity_checks.ipynb":
        repair_sanity(nb)
    elif path.name == "02_application_output_regression.ipynb":
        repair_regression(nb)
    elif path.name == "03_knottedgraph_vs_topoly_scaling.ipynb":
        repair_scaling(nb)
    elif path.name == "04_synthetic_ground_truth_validation.ipynb":
        repair_synthetic(nb)
    elif path.name == "05_same_abstract_graph_distinct_embeddings.ipynb":
        repair_distinct(nb)
    else:
        repair_generic(nb)
        if has_yamada(nb) and path not in PLOTTING_ONLY:
            ensure_backend_in_import_cell(nb)

    validate(path, nb)
    after = json.dumps(nb, sort_keys=True)
    if check:
        if before != after:
            raise AssertionError(f"{path} is not normalized; run this script without --check")
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
