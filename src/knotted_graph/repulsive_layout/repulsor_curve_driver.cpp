#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#define TOOLS_NO_STDFORMAT
#include <fmt/format.h>

#include "submodules/Tensors/OpenBLAS.hpp"
#include "Repulsor.hpp"

using Int = std::int32_t;
using LInt = std::int64_t;
using Real = double;

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    int steps = 50;
    int max_iter = 60;
    int threads = 1;
    int max_backtracks = 12;
    Real q = 4.0;
    Real p = 8.0;
    Real max_time = 1.0;
    Real safe_fraction = 0.95;
    Real armijo = 1e-4;
    Real tolerance = 1e-4;
    Real min_step = 1e-10;
    bool pin_special_vertices = true;
    std::filesystem::path history;
    std::filesystem::path save_steps_dir;
};

struct CurveData {
    std::vector<Real> vertices;
    std::vector<Int> edges;
};

void PrintUsage() {
    std::cerr
        << "Usage: repulsor_curve_driver --input curve.txt --output final.obj [options]\n"
        << "Options:\n"
        << "  --steps N\n"
        << "  --q Q --p P\n"
        << "  --threads N\n"
        << "  --max-time T\n"
        << "  --safe-fraction F\n"
        << "  --max-backtracks N\n"
        << "  --tolerance TOL\n"
        << "  --history history.csv\n"
        << "  --save-steps-dir DIR\n"
        << "  --free-special-vertices\n";
}

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--input") {
            opts.input = require_value(arg);
        } else if (arg == "--output") {
            opts.output = require_value(arg);
        } else if (arg == "--steps") {
            opts.steps = std::stoi(require_value(arg));
        } else if (arg == "--q") {
            opts.q = std::stod(require_value(arg));
        } else if (arg == "--p") {
            opts.p = std::stod(require_value(arg));
        } else if (arg == "--threads") {
            opts.threads = std::stoi(require_value(arg));
        } else if (arg == "--max-time") {
            opts.max_time = std::stod(require_value(arg));
        } else if (arg == "--safe-fraction") {
            opts.safe_fraction = std::stod(require_value(arg));
        } else if (arg == "--armijo") {
            opts.armijo = std::stod(require_value(arg));
        } else if (arg == "--max-backtracks") {
            opts.max_backtracks = std::stoi(require_value(arg));
        } else if (arg == "--max-iter") {
            opts.max_iter = std::stoi(require_value(arg));
        } else if (arg == "--tolerance") {
            opts.tolerance = std::stod(require_value(arg));
        } else if (arg == "--min-step") {
            opts.min_step = std::stod(require_value(arg));
        } else if (arg == "--history") {
            opts.history = require_value(arg);
        } else if (arg == "--save-steps-dir") {
            opts.save_steps_dir = require_value(arg);
        } else if (arg == "--free-special-vertices") {
            opts.pin_special_vertices = false;
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.input.empty() || opts.output.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    if (opts.steps < 0 || opts.threads < 1 || opts.max_backtracks < 0) {
        throw std::runtime_error("Invalid non-positive integer option");
    }
    if (opts.q <= 0 || opts.p <= 0 || opts.max_time <= 0 || opts.safe_fraction <= 0) {
        throw std::runtime_error("Invalid non-positive real option");
    }
    return opts;
}

CurveData ReadCurve(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open input: " + path.string());
    }

    CurveData data;
    std::string token;
    Int vertex_count = -1;
    Int edge_count = -1;
    while (in >> token) {
        if (token == "#") {
            std::string ignored;
            std::getline(in, ignored);
        } else if (token == "vertices") {
            in >> vertex_count;
            data.vertices.resize(static_cast<size_t>(vertex_count) * 3);
            for (Int i = 0; i < vertex_count; ++i) {
                in >> data.vertices[3 * i + 0]
                   >> data.vertices[3 * i + 1]
                   >> data.vertices[3 * i + 2];
            }
        } else if (token == "edges") {
            in >> edge_count;
            data.edges.resize(static_cast<size_t>(edge_count) * 2);
            for (Int i = 0; i < edge_count; ++i) {
                in >> data.edges[2 * i + 0] >> data.edges[2 * i + 1];
            }
        } else {
            throw std::runtime_error("Unexpected token in curve file: " + token);
        }
    }

    if (vertex_count <= 0 || edge_count <= 0) {
        throw std::runtime_error("Input file must contain vertices and edges sections");
    }
    return data;
}

void WriteObj(const std::filesystem::path& path, const std::vector<Real>& vertices, const std::vector<Int>& edges) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not write output: " + path.string());
    }
    const Int vertex_count = static_cast<Int>(vertices.size() / 3);
    const Int edge_count = static_cast<Int>(edges.size() / 2);
    out.setf(std::ios::fixed);
    out.precision(9);
    for (Int i = 0; i < vertex_count; ++i) {
        out << "v " << vertices[3 * i + 0] << " " << vertices[3 * i + 1] << " " << vertices[3 * i + 2] << "\n";
    }
    for (Int i = 0; i < edge_count; ++i) {
        out << "l " << edges[2 * i + 0] + 1 << " " << edges[2 * i + 1] + 1 << "\n";
    }
}

std::vector<Int> Degrees(Int vertex_count, const std::vector<Int>& edges) {
    std::vector<Int> degrees(vertex_count, 0);
    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int i = 0; i < edge_count; ++i) {
        ++degrees[edges[2 * i + 0]];
        ++degrees[edges[2 * i + 1]];
    }
    return degrees;
}

std::filesystem::path StepObjPath(const std::filesystem::path& dir, int step) {
    std::ostringstream name;
    name << "step_" << std::setw(4) << std::setfill('0') << step << ".obj";
    return dir / name.str();
}

Real Dot(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

Real Norm(const std::vector<Real>& a) {
    return std::sqrt(Dot(a, a));
}

int main(int argc, char** argv) {
    try {
        const Options opts = ParseArgs(argc, argv);
        CurveData curve = ReadCurve(opts.input);

        using Mesh_T = Repulsor::SimplicialMeshBase<Real, Int, LInt>;
        using Energy_T = Repulsor::EnergyBase<Mesh_T>;
        using Metric_T = Repulsor::MetricBase<Mesh_T>;

        Repulsor::SimplicialMesh_Factory<Mesh_T, 1, 1, 3, 3> mesh_factory;
        Repulsor::TangentPointEnergy0_Factory<Mesh_T, 1, 1, 3, 3> energy_factory;
        Repulsor::TangentPointMetric0_Factory<Mesh_T, 1, 1, 3, 3> metric_factory;

        const Int vertex_count = static_cast<Int>(curve.vertices.size() / 3);
        const Int edge_count = static_cast<Int>(curve.edges.size() / 2);
        const std::vector<Int> degrees = Degrees(vertex_count, curve.edges);

        auto mesh_ptr = mesh_factory.Make(
            curve.vertices.data(), vertex_count, 3, false,
            curve.edges.data(), edge_count, 2, false,
            opts.threads
        );
        if (!mesh_ptr) {
            throw std::runtime_error("Failed to construct Repulsor mesh");
        }
        auto& mesh = *mesh_ptr;
        mesh.cluster_tree_settings.split_threshold = 1;
        mesh.block_cluster_tree_settings.far_field_separation_parameter = 0.25;
        mesh.adaptivity_settings.theta = 10.0;

        std::unique_ptr<Energy_T> tpe_ptr = energy_factory.Make(1, 3, opts.q, opts.p);
        std::unique_ptr<Metric_T> tpm_ptr = metric_factory.Make(1, 3, opts.q, opts.p);
        auto& tpe = *tpe_ptr;
        auto& tpm = *tpm_ptr;

        std::vector<Real> coords = curve.vertices;
        std::vector<Real> direction(coords.size(), 0);
        std::vector<Real> trial(coords.size(), 0);
        Mesh_T::CotangentVector_T diff(vertex_count, 3);
        Mesh_T::TangentVector_T gradient(vertex_count, 3);

        std::ofstream history;
        if (!opts.history.empty()) {
            if (opts.history.has_parent_path()) {
                std::filesystem::create_directories(opts.history.parent_path());
            }
            history.open(opts.history);
            if (!history) {
                throw std::runtime_error("Could not write history: " + opts.history.string());
            }
            history << "step,accepted,energy_before,energy_after,safe_t,step_size,margin,backtracks,dir_norm,dir_derivative\n";
        }

        if (!opts.save_steps_dir.empty()) {
            std::filesystem::create_directories(opts.save_steps_dir);
            WriteObj(StepObjPath(opts.save_steps_dir, 0), coords, curve.edges);
        }

        int accepted = 0;
        for (int step = 1; step <= opts.steps; ++step) {
            const Real energy = tpe.Differential(mesh, diff.data());
            tpm.Solve(
                mesh,
                Real(1), diff.data(), mesh.AmbDim(),
                Real(0), gradient.data(), mesh.AmbDim(),
                mesh.AmbDim(), opts.max_iter, opts.tolerance
            );

            for (Int i = 0; i < vertex_count; ++i) {
                const bool pinned = opts.pin_special_vertices && degrees[i] != 2;
                for (Int k = 0; k < 3; ++k) {
                    direction[3 * i + k] = pinned ? Real(0) : -gradient(i, k);
                }
            }

            const Real dir_norm = Norm(direction);
            const Real dir_derivative = Dot(std::vector<Real>(diff.data(), diff.data() + coords.size()), direction);
            if (dir_norm <= 1e-14 || dir_derivative >= 0) {
                std::cout << "Stopped at step " << step << ": non-descent direction\n";
                break;
            }

            Real safe_t = mesh.MaximumSafeStepSize(direction.data(), opts.max_time);
            Real step_size = opts.safe_fraction * safe_t;
            if (step_size <= opts.min_step) {
                std::cout << "Stopped at step " << step << ": safe step too small (" << step_size << ")\n";
                break;
            }

            bool accepted_step = false;
            Real trial_energy = energy;
            for (int attempt = 0; attempt <= opts.max_backtracks; ++attempt) {
                for (size_t i = 0; i < coords.size(); ++i) {
                    trial[i] = coords[i] + step_size * direction[i];
                }

                mesh.SemiStaticUpdate(trial.data());
                trial_energy = tpe.Value(mesh);

                const Real target = energy + opts.armijo * step_size * dir_derivative;
                if (trial_energy <= target) {
                    coords = trial;
                    ++accepted;
                    accepted_step = true;
                    if (history) {
                        history << step << ",1,"
                                << energy << ","
                                << trial_energy << ","
                                << safe_t << ","
                                << step_size << ","
                                << (safe_t - step_size) << ","
                                << attempt << ","
                                << dir_norm << ","
                                << dir_derivative << "\n";
                    }
                    if (!opts.save_steps_dir.empty()) {
                        WriteObj(StepObjPath(opts.save_steps_dir, accepted), coords, curve.edges);
                    }
                    std::cout << "step " << step
                              << " accepted energy " << energy << " -> " << trial_energy
                              << " safe_t " << safe_t
                              << " step_size " << step_size
                              << " backtracks " << attempt << "\n";
                    break;
                }

                mesh.SemiStaticUpdate(coords.data());
                step_size *= 0.5;
                if (step_size <= opts.min_step) {
                    break;
                }
            }

            if (!accepted_step) {
                mesh.SemiStaticUpdate(coords.data());
                if (history) {
                    history << step << ",0,"
                            << energy << ","
                            << trial_energy << ","
                            << safe_t << ","
                            << step_size << ","
                            << (safe_t - step_size) << ","
                            << opts.max_backtracks << ","
                            << dir_norm << ","
                            << dir_derivative << "\n";
                }
                std::cout << "Stopped at step " << step << ": line search failed\n";
                break;
            }
        }

        WriteObj(opts.output, coords, curve.edges);
        std::cout << "accepted_steps " << accepted << "\n";
        std::cout << "wrote " << opts.output.string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "repulsor_curve_driver error: " << e.what() << "\n";
        PrintUsage();
        return 1;
    }
}
