from __future__ import annotations

import subprocess

import pytest

from knotted_graph.layout.repulsive import driver


def test_run_command_surfaces_native_diagnostics(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            ["g++"],
            stderr="first line\nfatal error: fmt/format.h not found\n",
        )

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="fmt/format.h not found"):
        driver.run_command(["g++"], verbose=False)


def test_macos_driver_build_uses_system_accelerate(monkeypatch, tmp_path):
    root = tmp_path / "Repulsor"
    root.mkdir()
    (root / "Repulsor.hpp").write_text("// header")
    source = tmp_path / "driver.cpp"
    source.write_text("int main() {}")
    captured: dict[str, list[str]] = {}

    def fake_run(command, verbose=True):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(driver.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(driver, "run_command", fake_run)
    config = driver.DriverConfig(
        repulsor_root=root,
        driver_source=source,
        driver_binary=tmp_path / "repulsor_driver",
        use_wsl=False,
        verbose=False,
    )

    driver.build_driver(config, force=True)

    command = captured["command"]
    assert command[0] == "c++"
    assert "-fenable-matrix" in command
    assert command[command.index("-framework") + 1] == "Accelerate"
    assert "-lfmt" not in command
    assert "-lopenblas" not in command


def test_run_driver_keeps_topology_check_enabled_by_default(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def fake_run_command(
        command: list[str], verbose: bool = True
    ) -> subprocess.CompletedProcess[str]:
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

    def fake_run_command(
        command: list[str], verbose: bool = True
    ) -> subprocess.CompletedProcess[str]:
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
