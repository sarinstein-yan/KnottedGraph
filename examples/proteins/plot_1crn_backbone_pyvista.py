"""Render the 1CRN C-alpha backbone in the generic PyVista tube style."""

from __future__ import annotations

from plot_protein_backbone_pyvista import run


def main() -> None:
    run("1CRN", chain_id="A", model_id=1, output_prefix="1crn_backbone")


if __name__ == "__main__":
    main()
