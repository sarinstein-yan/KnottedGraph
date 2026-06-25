from __future__ import annotations

from types import SimpleNamespace

from knotted_graph.layout.repulsive import cli


def _fake_result(tmp_path):
    return SimpleNamespace(
        workspace=tmp_path,
        metadata={
            "certificate": {},
            "history_summary": {},
            "decimation": {},
            "elapsed_seconds": 0.0,
        },
    )


def test_examples_cli_defaults_skip_independent_verifier(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run_protein_example(**kwargs):
        captured.update(kwargs)
        return _fake_result(tmp_path)

    monkeypatch.setattr(cli, "run_protein_example", fake_run_protein_example)
    args = cli.build_parser().parse_args(["examples", "--sample", "1aoc", "--out", str(tmp_path)])

    assert cli.run_examples(args) == 0
    assert captured["save_steps"] is False
    assert captured["verify_topology"] is False
    assert captured["solver_options"].topology_check is True


def test_examples_cli_verify_topology_implies_saved_steps(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run_protein_example(**kwargs):
        captured.update(kwargs)
        return _fake_result(tmp_path)

    monkeypatch.setattr(cli, "run_protein_example", fake_run_protein_example)
    args = cli.build_parser().parse_args(
        ["examples", "--sample", "1aoc", "--out", str(tmp_path), "--verify-topology"]
    )

    assert cli.run_examples(args) == 0
    assert captured["save_steps"] is True
    assert captured["verify_topology"] is True
