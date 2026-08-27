import pytest

import knotted_graph.inputs as inputs


def test_surface_adapter_missing_pyvista_error_is_actionable(monkeypatch):
    monkeypatch.delitem(inputs.__dict__, "from_surface_mesh", raising=False)

    def missing_pyvista(module_name):
        raise ModuleNotFoundError(
            "No module named 'pyvista'",
            name="pyvista",
        )

    monkeypatch.setattr(inputs, "import_module", missing_pyvista)

    with pytest.raises(ImportError, match=r"knotted_graph\[surface\]") as exc_info:
        inputs.from_surface_mesh

    assert "uv sync --extra surface" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_surface_adapter_does_not_mask_unrelated_import_errors(monkeypatch):
    monkeypatch.delitem(inputs.__dict__, "from_surface_mesh", raising=False)
    unrelated_error = ModuleNotFoundError(
        "No module named 'unrelated_dependency'",
        name="unrelated_dependency",
    )

    def fail_with_unrelated_error(module_name):
        raise unrelated_error

    monkeypatch.setattr(inputs, "import_module", fail_with_unrelated_error)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        inputs.from_surface_mesh

    assert exc_info.value is unrelated_error
