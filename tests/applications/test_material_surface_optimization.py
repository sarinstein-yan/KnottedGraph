from types import SimpleNamespace

import numpy as np
import skimage.morphology as morph

from knotted_graph.applications.material_surface import MaterialFermiSurface
from knotted_graph.applications.materials import MaterialFermiSurface as LazyMaterialFermiSurface
from knotted_graph.applications.materials import H_Ti3Al_sympy, H_YH3_sympy


def test_material_public_import_resolves_optimized_class():
    assert LazyMaterialFermiSurface is MaterialFermiSurface
    assert MaterialFermiSurface.__module__ == "knotted_graph.applications.material_surface"


def test_material_cropped_lee_skeleton_is_byte_identical_to_full_volume():
    mask = np.zeros((42, 39, 45), dtype=bool)
    mask[9:31, 8:28, 11:34] = True
    mask[16:23, 4:34, 17:24] = True

    expected = morph.skeletonize(mask, method="lee")
    stub = SimpleNamespace(_interior_mask=mask)
    actual = MaterialFermiSurface._skeleton_image.func(stub)

    assert np.array_equal(actual, expected)


def test_material_cropped_skeleton_preserves_global_voxel_coordinates():
    mask = np.zeros((64, 61, 59), dtype=bool)
    mask[37:53, 31:47, 28:44] = True
    mask[42:48, 26:52, 33:39] = True

    full = morph.skeletonize(mask, method="lee")
    stub = SimpleNamespace(_interior_mask=mask)
    cropped = MaterialFermiSurface._skeleton_image.func(stub)

    assert np.array_equal(np.argwhere(cropped), np.argwhere(full))


def test_builtin_square_root_materials_pass_hermitian_check():
    import sympy as sp

    k_symbols = sp.symbols("kx ky kz", real=True)
    for hamiltonian in (
        H_Ti3Al_sympy(k_symbols=k_symbols),
        H_YH3_sympy(k_symbols=k_symbols),
    ):
        surface = MaterialFermiSurface(
            hamiltonian,
            k_symbols=k_symbols,
            dimension=5,
            band_pair=(0, 1),
            gap_tol=0.1,
        )
        assert surface.is_Hermitian is True
