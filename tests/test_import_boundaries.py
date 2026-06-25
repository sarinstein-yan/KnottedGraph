import json
import subprocess
import sys
import textwrap


def _run_python(source: str) -> dict[str, bool]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_top_level_import_does_not_load_application_stack():
    loaded = _run_python(
        """
        import json
        import sys

        import knotted_graph

        names = [
            "poly2graph",
            "pyvista",
            "skimage",
            "skimage.morphology",
            "knotted_graph.NodalSkeleton",
            "knotted_graph.surface_modes",
            "knotted_graph.vis",
        ]
        print(json.dumps({name: name in sys.modules for name in names}))
        """
    )

    assert loaded == {
        "poly2graph": False,
        "pyvista": False,
        "skimage": False,
        "skimage.morphology": False,
        "knotted_graph.NodalSkeleton": False,
        "knotted_graph.surface_modes": False,
        "knotted_graph.vis": False,
    }


def test_yamada_polynomial_import_does_not_load_application_stack():
    loaded = _run_python(
        """
        import json
        import sys

        from knotted_graph.yamada.polynomial import Yamada, compute_negami

        names = [
            "poly2graph",
            "pyvista",
            "skimage",
            "skimage.morphology",
            "knotted_graph.NodalSkeleton",
            "knotted_graph.surface_modes",
            "knotted_graph.vis",
        ]
        print(json.dumps({name: name in sys.modules for name in names}))
        """
    )

    assert loaded == {
        "poly2graph": False,
        "pyvista": False,
        "skimage": False,
        "skimage.morphology": False,
        "knotted_graph.NodalSkeleton": False,
        "knotted_graph.surface_modes": False,
        "knotted_graph.vis": False,
    }


def test_input_package_import_does_not_load_surface_stack():
    loaded = _run_python(
        """
        import json
        import sys

        import knotted_graph.inputs

        names = [
            "pyvista",
            "knotted_graph.inputs.surface_mesh",
        ]
        print(json.dumps({name: name in sys.modules for name in names}))
        """
    )

    assert loaded == {
        "pyvista": False,
        "knotted_graph.inputs.surface_mesh": False,
    }
