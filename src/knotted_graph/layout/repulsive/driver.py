from __future__ import annotations

import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPULSOR_ROOT = Path.cwd() / "external" / "Repulsor"
DEFAULT_DRIVER_SOURCE = Path(__file__).resolve().with_name("repulsor_curve_driver.cpp")
DEFAULT_DRIVER_BINARY = (
    Path.cwd() / "build" / "repulsor_driver" / "repulsor_curve_driver"
)


@dataclass
class DriverConfig:
    repulsor_root: Path = DEFAULT_REPULSOR_ROOT
    driver_source: Path = DEFAULT_DRIVER_SOURCE
    driver_binary: Path = DEFAULT_DRIVER_BINARY
    use_wsl: bool | None = None
    verbose: bool = True


@dataclass
class SolverOptions:
    steps: int = 100
    q: float = 4.0
    p: float = 8.0
    threads: int = 1
    max_time: float = 1.0
    safe_fraction: float = 0.95
    max_backtracks: int = 12
    max_iter: int = 60
    tolerance: float = 1e-4
    repulsion_weight: float = 1.0
    length_weight: float = 0.0
    curve_length_floor_weight: float = 0.0
    bend_weight: float = 0.0
    tube_radius: float = 0.0
    tube_gap: float = 0.0
    tube_weight: float = 0.0
    topology_check: bool = True
    topology_tolerance: float = 1e-7
    free_special_vertices: bool = False


def should_use_wsl(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.name == "nt"


def effective_repulsor_root(config: DriverConfig) -> Path:
    return Path(os.environ.get("REPULSOR_ROOT", config.repulsor_root)).resolve()


def wsl_path(path: Path) -> str:
    windows_path = path.resolve().as_posix()
    result = subprocess.run(
        ["wsl", "wslpath", "-a", windows_path],
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return result.stdout.strip()


def solver_path(path: Path, use_wsl: bool) -> str:
    return wsl_path(path) if use_wsl else str(path.resolve())


def run_command(
    command: list[str], verbose: bool = True
) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("+", " ".join(shlex.quote(str(part)) for part in command), flush=True)
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        diagnostics = (
            exc.stderr or exc.stdout or "no compiler/solver diagnostics"
        ).strip()
        tail = "\n".join(diagnostics.splitlines()[-40:])
        raise RuntimeError(
            f"External command failed with exit code {exc.returncode}:\n{tail}"
        ) from exc


def build_driver(config: DriverConfig, force: bool = False) -> Path:
    repulsor_root = effective_repulsor_root(config)
    driver_source = config.driver_source.resolve()
    driver_binary = config.driver_binary.resolve()
    use_wsl = should_use_wsl(config.use_wsl)

    if not repulsor_root.joinpath("Repulsor.hpp").exists():
        raise FileNotFoundError(
            f"Repulsor root not found or incomplete: {repulsor_root}. "
            "Clone Repulsor and/or set REPULSOR_ROOT."
        )
    if not driver_source.exists():
        raise FileNotFoundError(f"Repulsor driver source not found: {driver_source}")

    driver_binary.parent.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and driver_binary.exists()
        and driver_binary.stat().st_mtime >= driver_source.stat().st_mtime
    ):
        if config.verbose:
            print("Using existing Repulsor driver:", driver_binary)
        return driver_binary

    if platform.system() == "Darwin" and not use_wsl:
        compile_command = [
            "c++",
            "-std=c++20",
            "-O2",
            "-fenable-matrix",
            "-I",
            solver_path(repulsor_root, use_wsl),
            solver_path(driver_source, use_wsl),
            "-o",
            solver_path(driver_binary, use_wsl),
            "-framework",
            "Accelerate",
            "-pthread",
        ]
    else:
        compile_command = [
            "g++",
            "-std=c++20",
            "-O2",
            "-I",
            solver_path(repulsor_root, use_wsl),
            solver_path(driver_source, use_wsl),
            "-o",
            solver_path(driver_binary, use_wsl),
            "-lopenblas",
            "-llapack",
            "-llapacke",
            "-lfmt",
            "-pthread",
            "-lamd",
            "-fpermissive",
        ]
    if use_wsl:
        compile_command = ["wsl", *compile_command]
    run_command(compile_command, verbose=config.verbose)
    return driver_binary


def run_driver(
    input_curve: Path,
    output_obj: Path,
    history_csv: Path,
    options: SolverOptions,
    config: DriverConfig,
    save_steps_dir: Path | None = None,
    pinned_vertices: Path | None = None,
    curve_length_floors: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    driver_binary = config.driver_binary.resolve()
    use_wsl = should_use_wsl(config.use_wsl)

    command = [
        solver_path(driver_binary, use_wsl),
        "--input",
        solver_path(input_curve, use_wsl),
        "--output",
        solver_path(output_obj, use_wsl),
        "--steps",
        str(options.steps),
        "--q",
        str(options.q),
        "--p",
        str(options.p),
        "--threads",
        str(options.threads),
        "--max-time",
        str(options.max_time),
        "--safe-fraction",
        str(options.safe_fraction),
        "--max-backtracks",
        str(options.max_backtracks),
        "--max-iter",
        str(options.max_iter),
        "--tolerance",
        str(options.tolerance),
        "--repulsion-weight",
        str(options.repulsion_weight),
        "--length-weight",
        str(options.length_weight),
        "--curve-length-floor-weight",
        str(options.curve_length_floor_weight),
        "--bend-weight",
        str(options.bend_weight),
        "--tube-radius",
        str(options.tube_radius),
        "--tube-gap",
        str(options.tube_gap),
        "--tube-weight",
        str(options.tube_weight),
        "--topology-tolerance",
        str(options.topology_tolerance),
        "--history",
        solver_path(history_csv, use_wsl),
    ]
    if not options.topology_check:
        command.append("--no-topology-check")
    if save_steps_dir is not None:
        command.extend(["--save-steps-dir", solver_path(save_steps_dir, use_wsl)])
    if pinned_vertices is not None:
        command.extend(["--pinned-vertices", solver_path(pinned_vertices, use_wsl)])
    if curve_length_floors is not None:
        command.extend(
            ["--curve-length-floors", solver_path(curve_length_floors, use_wsl)]
        )
    if options.free_special_vertices:
        command.append("--free-special-vertices")
    if use_wsl:
        command = ["wsl", *command]
    return run_command(command, verbose=config.verbose)
