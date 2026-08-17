from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebook = (root / args.notebook).resolve()
    if root not in notebook.parents:
        raise ValueError(f"Notebook must be inside repository: {notebook}")
    if not notebook.exists():
        raise FileNotFoundError(notebook)

    # Force every subprocess/import launched by a notebook to see this checkout
    # before site-packages. Individual notebooks retain their own local-source
    # bootstrap as a second independent guard.
    src = root / "src"
    os.environ["PYTHONPATH"] = str(src) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("PLOTLY_RENDERER", "json")

    with notebook.open("r", encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute(cwd=str(root))
    print(f"PASS {notebook.relative_to(root)}")


if __name__ == "__main__":
    main()
