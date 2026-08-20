"""Focused tests for the current Main/S1/S2 figure-generation package."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageChops

from examples.input_gallery.task2_figures.cli import build_parser, main as cli_main
from examples.input_gallery.task2_figures.common import (
    DEFAULT_ASSET_ROOT,
    resolve_asset,
    save_figure,
    validate_assets,
)
from examples.input_gallery.task2_figures.main import render_main
from examples.input_gallery.task2_figures.s1 import format_yamada_lines, render_s1
from examples.input_gallery.task2_figures.s2 import render_s2
from examples.input_gallery.task2_figures.specs import (
    FIGURES,
    MAIN,
    S1,
    S2,
    FigureSpec,
    PanelSpec,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "examples" / "input_gallery" / "task2_figures"


def _canonical_polynomial(expression: str) -> tuple[tuple[int, int], ...]:
    """Canonicalize the expanded integer polynomial without an algebra dependency."""

    compact = expression.replace(" ", "")
    coefficients: dict[int, int] = {}
    for token in re.findall(r"[+-]?[^+-]+", compact):
        sign = -1 if token.startswith("-") else 1
        body = token.lstrip("+-")
        if "A" not in body:
            exponent = 0
            coefficient = int(body)
        else:
            coefficient_text = body.split("*A", 1)[0] if "*A" in body else ""
            coefficient = int(coefficient_text) if coefficient_text else 1
            exponent = int(body.split("**", 1)[1]) if "**" in body else 1
        coefficients[exponent] = coefficients.get(exponent, 0) + sign * coefficient
    return tuple(sorted((degree, value) for degree, value in coefficients.items() if value))


class Task2FigureTests(unittest.TestCase):
    def test_current_figure_contract_is_exact(self) -> None:
        self.assertEqual(tuple(FIGURES), ("main", "s1", "s2"))
        self.assertEqual((MAIN.rows, MAIN.cols, len(MAIN.panels)), (2, 4, 8))
        self.assertEqual((S1.rows, S1.cols, len(S1.panels)), (3, 5, 15))
        self.assertEqual((S2.rows, S2.cols, len(S2.panels)), (3, 4, 12))
        self.assertTrue(MAIN.show_formats and not MAIN.show_yamada)
        self.assertTrue(S1.show_yamada and S1.tight_bbox)
        self.assertTrue(not S2.show_yamada and S2.tight_bbox)
        self.assertEqual(S2.output_stem, "appendix_s3_skeletonization_beyond_yamada_v9")

    def test_specs_are_portable_and_hash_locked(self) -> None:
        all_panels = tuple(panel for spec in FIGURES.values() for panel in spec.panels)
        self.assertEqual(len(all_panels), 35)
        self.assertEqual(
            len({(spec.key, panel.key) for spec in FIGURES.values() for panel in spec.panels}),
            35,
        )
        for panel in all_panels:
            self.assertFalse(panel.asset.is_absolute())
            self.assertNotIn("..", panel.asset.parts)
            self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", panel.sha256))

    def test_s1_polynomials_are_nonzero_and_pairwise_distinct(self) -> None:
        polynomials = [panel.polynomial for panel in S1.panels]
        self.assertTrue(all(polynomial and polynomial.strip() != "0" for polynomial in polynomials))
        canonical = [_canonical_polynomial(polynomial or "") for polynomial in polynomials]
        self.assertEqual(len(canonical), 15)
        self.assertEqual(len(set(canonical)), 15)

        fingerprint_payload = "\n".join(
            f"{panel.key}\t{panel.title}\t{panel.polynomial}\t"
            f"{panel.asset.as_posix()}\t{panel.sha256}"
            for panel in S1.panels
        )
        self.assertEqual(
            hashlib.sha256(fingerprint_payload.encode()).hexdigest(),
            "05475a3791ac78b8be903dc36b2ae675fbd90b8d44ed2af6dfc15d4f11e30fcd",
        )

    def test_s1_uses_computer_modern_upsilon_in_at_most_two_lines(self) -> None:
        for panel in S1.panels:
            lines = format_yamada_lines(panel.polynomial or "")
            self.assertLessEqual(len(lines), 2)
            self.assertGreaterEqual(len(lines), 1)
            self.assertIn(r"\Upsilon(G;Y)", lines[0])
            self.assertNotIn("A", "".join(lines))
            self.assertNotIn("ϒ", "".join(lines))

    def test_missing_or_changed_assets_fail_closed(self) -> None:
        missing = PanelSpec(
            key="missing",
            title="Missing",
            asset=Path("panel.png"),
            sha256="0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Missing accepted panel"):
                resolve_asset(missing, root)

            path = root / "panel.png"
            path.write_bytes(b"not the accepted panel")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_asset(missing, root)

    def test_cli_exposes_only_current_figures_and_verification(self) -> None:
        parser = build_parser()
        for target in ("main", "s1", "s2", "all", "verify"):
            self.assertEqual(parser.parse_args([target]).target, target)
        target_action = next(action for action in parser._actions if action.dest == "target")
        self.assertEqual(set(target_action.choices or ()), {"main", "s1", "s2", "all", "verify"})

    def test_cli_validates_every_selected_figure_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch(
                    "examples.input_gallery.task2_figures.cli.validate_assets",
                    side_effect=((), FileNotFoundError("missing S1")),
                ),
                mock.patch("examples.input_gallery.task2_figures.cli.render_main") as render_main_mock,
                mock.patch("examples.input_gallery.task2_figures.cli.render_s1") as render_s1_mock,
                mock.patch("examples.input_gallery.task2_figures.cli.render_s2") as render_s2_mock,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "missing S1"):
                    cli_main(["all", "--asset-root", directory, "--output-dir", directory])
                render_main_mock.assert_not_called()
                render_s1_mock.assert_not_called()
                render_s2_mock.assert_not_called()

    def test_format_failure_does_not_publish_partial_outputs(self) -> None:
        panel = PanelSpec("test", "Test", Path("test.png"), "0" * 64)
        spec = FigureSpec("test", "test_figure", 1, 1, (1.0, 1.0), (panel,))

        class FailingFigure:
            def savefig(self, path: Path, **_kwargs: object) -> None:
                if Path(path).suffix == ".svg":
                    raise RuntimeError("synthetic SVG failure")
                Path(path).write_bytes(b"staged output")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "synthetic SVG failure"):
                save_figure(FailingFigure(), spec, output_dir)
            self.assertFalse(list(output_dir.glob("test_figure.*")))

    def test_package_contains_no_scheduler_or_personal_path_content(self) -> None:
        self.assertFalse(list(PACKAGE_DIR.rglob("*.pbs")))
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md"}:
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"/(?:scratch|nfs)/", text))

    def test_local_accepted_bundle_when_available(self) -> None:
        if not DEFAULT_ASSET_ROOT.is_dir():
            self.skipTest("accepted panel bundle is not present in this checkout")
        self.assertEqual(
            {key: len(validate_assets(spec, DEFAULT_ASSET_ROOT)) for key, spec in FIGURES.items()},
            {"main": 8, "s1": 15, "s2": 12},
        )

    @unittest.skipUnless(
        os.environ.get("TASK2_FIGURES_INTEGRATION") == "1",
        "set TASK2_FIGURES_INTEGRATION=1 for full publication-resolution render",
    )
    def test_full_render_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            results = (
                render_main(asset_root=DEFAULT_ASSET_ROOT, output_dir=output_dir),
                render_s1(asset_root=DEFAULT_ASSET_ROOT, output_dir=output_dir),
                render_s2(asset_root=DEFAULT_ASSET_ROOT, output_dir=output_dir),
            )
            expected_sizes = {
                "main": (3262, 1777),
                "s1": (5370, 3032),
                "s2": (3223, 2423),
            }
            accepted_pngs = {
                "main": DEFAULT_ASSET_ROOT / "main_text_input_gallery_framed_v9.png",
                "s1": DEFAULT_ASSET_ROOT / "appendix_s1_yamada_nonzero_v9.png",
                "s2": DEFAULT_ASSET_ROOT / "appendix_s3_skeletonization_beyond_yamada_v9.png",
            }
            for result in results:
                self.assertTrue(result.summary.is_file())
                self.assertEqual({path.suffix for path in result.outputs}, {".png", ".svg", ".pdf"})
                png = next(path for path in result.outputs if path.suffix == ".png")
                with Image.open(png) as image:
                    self.assertEqual(image.size, expected_sizes[result.figure])
                    accepted_path = accepted_pngs[result.figure]
                    if accepted_path.is_file():
                        with Image.open(accepted_path) as accepted:
                            difference = ImageChops.difference(
                                image.convert("RGBA"), accepted.convert("RGBA")
                            )
                            self.assertIsNone(
                                difference.getbbox(),
                                f"{result.figure} pixels changed from the accepted figure",
                            )
                for path in result.outputs:
                    self.assertGreater(path.stat().st_size, 0)
                    self.assertEqual(len(hashlib.sha256(path.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
