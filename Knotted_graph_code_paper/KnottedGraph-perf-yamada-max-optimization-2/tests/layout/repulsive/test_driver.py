from __future__ import annotations

import subprocess

from knotted_graph.layout.repulsive import driver


def test_run_driver_keeps_topology_check_enabled_by_default(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def fake_run_command(command: list[str], verbose: bool = True) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(driver, "run_command", fake_run_command)

    config = driver.DriverConfig(
        driver_binary=tmp_path / "repulsor_curve_driver",
        use_wsl=False,
        verbose=False,
    )
    driver.run_driver(
        tmp_path / "input.curve",
        tmp_path / "output.obj",
        tmp_path / "history.csv",
        driver.SolverOptions(steps=1),
        config,
    )

    command = captured["command"]
    assert "--topology-tolerance" in command
    assert "--no-topology-check" not in command
    assert "--save-steps-dir" not in command


def test_run_driver_passes_topology_options(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def fake_run_command(command: list[str], verbose: bool = True) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(driver, "run_command", fake_run_command)

    options = driver.SolverOptions(
        steps=1,
        topology_check=False,
        topology_tolerance=2e-6,
    )
    config = driver.DriverConfig(
        driver_binary=tmp_path / "repulsor_curve_driver",
        use_wsl=False,
        verbose=False,
    )
    driver.run_driver(
        tmp_path / "input.curve",
        tmp_path / "output.obj",
        tmp_path / "history.csv",
        options,
        config,
    )

    command = captured["command"]
    assert "--topology-tolerance" in command
    assert command[command.index("--topology-tolerance") + 1] == "2e-06"
    assert "--no-topology-check" in command
