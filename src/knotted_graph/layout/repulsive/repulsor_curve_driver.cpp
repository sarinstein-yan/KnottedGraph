#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include "submodules/Tensors/Accelerate.hpp"
#else
#define TOOLS_NO_STDFORMAT
#include <fmt/format.h>
#include "submodules/Tensors/OpenBLAS.hpp"
#endif
#include "Repulsor.hpp"

using Int = std::int32_t;
using LInt = std::int64_t;
using Real = double;
using Vec3 = std::array<Real, 3>;

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
    Real repulsion_weight = 1.0;
    Real length_weight = 0.0;
    Real curve_length_floor_weight = 0.0;
    Real bend_weight = 0.0;
    Real tube_radius = 0.0;
    Real tube_gap = 0.0;
    Real tube_weight = 0.0;
    bool topology_check = true;
    Real topology_tolerance = 1e-7;
    bool pin_special_vertices = true;
    std::filesystem::path pinned_vertices;
    std::filesystem::path curve_length_floors;
    std::filesystem::path history;
    std::filesystem::path save_steps_dir;
};

struct CurveData {
    std::vector<Real> vertices;
    std::vector<Int> edges;
};

struct CurveLengthFloor {
    Real floor = 0.0;
    std::vector<Int> vertices;
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
        << "  --repulsion-weight W\n"
        << "  --length-weight W\n"
        << "  --curve-length-floor-weight W\n"
        << "  --curve-length-floors floors.txt\n"
        << "  --bend-weight W\n"
        << "  --tube-radius R\n"
        << "  --tube-gap G\n"
        << "  --tube-weight W\n"
        << "  --topology-tolerance TOL\n"
        << "  --no-topology-check\n"
        << "  --history history.csv\n"
        << "  --save-steps-dir DIR\n"
        << "  --pinned-vertices pinned.txt\n"
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
        } else if (arg == "--repulsion-weight") {
            opts.repulsion_weight = std::stod(require_value(arg));
        } else if (arg == "--length-weight") {
            opts.length_weight = std::stod(require_value(arg));
        } else if (arg == "--curve-length-floor-weight") {
            opts.curve_length_floor_weight = std::stod(require_value(arg));
        } else if (arg == "--curve-length-floors") {
            opts.curve_length_floors = require_value(arg);
        } else if (arg == "--bend-weight") {
            opts.bend_weight = std::stod(require_value(arg));
        } else if (arg == "--tube-radius") {
            opts.tube_radius = std::stod(require_value(arg));
        } else if (arg == "--tube-gap") {
            opts.tube_gap = std::stod(require_value(arg));
        } else if (arg == "--tube-weight") {
            opts.tube_weight = std::stod(require_value(arg));
        } else if (arg == "--topology-tolerance") {
            opts.topology_tolerance = std::stod(require_value(arg));
        } else if (arg == "--no-topology-check") {
            opts.topology_check = false;
        } else if (arg == "--history") {
            opts.history = require_value(arg);
        } else if (arg == "--save-steps-dir") {
            opts.save_steps_dir = require_value(arg);
        } else if (arg == "--pinned-vertices") {
            opts.pinned_vertices = require_value(arg);
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
    if (
        opts.repulsion_weight < 0
        || opts.length_weight < 0
        || opts.curve_length_floor_weight < 0
        || opts.bend_weight < 0
    ) {
        throw std::runtime_error(
            "--repulsion-weight, --length-weight, --curve-length-floor-weight, and --bend-weight must be non-negative"
        );
    }
    if (opts.tube_radius < 0 || opts.tube_gap < 0 || opts.tube_weight < 0) {
        throw std::runtime_error("--tube-radius, --tube-gap, and --tube-weight must be non-negative");
    }
    if (opts.topology_tolerance < 0) {
        throw std::runtime_error("--topology-tolerance must be non-negative");
    }
    if (opts.curve_length_floor_weight > 0 && opts.curve_length_floors.empty()) {
        throw std::runtime_error("--curve-length-floors is required when --curve-length-floor-weight is positive");
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

std::vector<char> ReadPinnedVertices(const std::filesystem::path& path, Int vertex_count) {
    std::vector<char> pinned(vertex_count, 0);
    if (path.empty()) {
        return pinned;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open pinned vertices file: " + path.string());
    }

    std::string token;
    while (in >> token) {
        if (token == "#") {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        const Int index = static_cast<Int>(std::stoi(token));
        if (index < 0 || index >= vertex_count) {
            throw std::runtime_error("Pinned vertex index out of range: " + token);
        }
        pinned[index] = 1;
    }
    return pinned;
}

std::vector<CurveLengthFloor> ReadCurveLengthFloors(const std::filesystem::path& path, Int vertex_count) {
    std::vector<CurveLengthFloor> floors;
    if (path.empty()) {
        return floors;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open curve length floors file: " + path.string());
    }

    std::string token;
    while (in >> token) {
        if (token == "#") {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        CurveLengthFloor floor;
        floor.floor = std::stod(token);
        Int count = 0;
        in >> count;
        if (!in || count < 2) {
            throw std::runtime_error("Each curve length floor row must contain a floor, count, and at least two vertices");
        }
        if (floor.floor < Real(0)) {
            throw std::runtime_error("Curve length floor must be non-negative");
        }
        floor.vertices.resize(static_cast<size_t>(count));
        for (Int i = 0; i < count; ++i) {
            in >> floor.vertices[static_cast<size_t>(i)];
            const Int index = floor.vertices[static_cast<size_t>(i)];
            if (index < 0 || index >= vertex_count) {
                throw std::runtime_error("Curve length floor vertex index out of range");
            }
        }
        floors.push_back(std::move(floor));
    }
    return floors;
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

std::vector<std::vector<Int>> Adjacency(Int vertex_count, const std::vector<Int>& edges) {
    std::vector<std::vector<Int>> adjacency(vertex_count);
    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int i = 0; i < edge_count; ++i) {
        const Int a = edges[2 * i + 0];
        const Int b = edges[2 * i + 1];
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);
    }
    return adjacency;
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

Real Clamp01(const Real x) {
    if (x < Real(0)) {
        return Real(0);
    }
    if (x > Real(1)) {
        return Real(1);
    }
    return x;
}

Real Coord(const std::vector<Real>& vertices, const Int index, const Int axis) {
    return vertices[3 * index + axis];
}

Vec3 VertexAt(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const Int index,
    const Real tau
) {
    Vec3 value{};
    for (Int k = 0; k < 3; ++k) {
        const Real a = Coord(start, index, k);
        value[k] = a + tau * (Coord(end, index, k) - a);
    }
    return value;
}

struct SegmentClosestData {
    Real distance;
    Real s;
    Real t;
    Vec3 delta;
};

SegmentClosestData SegmentClosestPointsRaw(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const Vec3& d
) {
    Vec3 u{};
    Vec3 v{};
    Vec3 w{};
    for (Int k = 0; k < 3; ++k) {
        u[k] = b[k] - a[k];
        v[k] = d[k] - c[k];
        w[k] = a[k] - c[k];
    }

    auto dot3 = [](const Vec3& left, const Vec3& right) -> Real {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };

    const Real uu = dot3(u, u);
    const Real vv = dot3(v, v);
    const Real uv = dot3(u, v);
    const Real uw = dot3(u, w);
    const Real vw = dot3(v, w);
    const Real eps = 1e-20;

    Real s = 0;
    Real t = 0;
    if (uu <= eps && vv <= eps) {
        s = 0;
        t = 0;
    } else if (uu <= eps) {
        s = 0;
        t = Clamp01(vw / vv);
    } else if (vv <= eps) {
        s = Clamp01(-uw / uu);
        t = 0;
    } else {
        const Real denom = uu * vv - uv * uv;
        if (denom > eps) {
            s = Clamp01((uv * vw - vv * uw) / denom);
        } else {
            s = 0;
        }

        t = (uv * s + vw) / vv;
        if (t < Real(0)) {
            t = 0;
            s = Clamp01(-uw / uu);
        } else if (t > Real(1)) {
            t = 1;
            s = Clamp01((uv - uw) / uu);
        }
    }

    Vec3 delta{};
    Real dist2 = 0;
    for (Int k = 0; k < 3; ++k) {
        const Real x = a[k] + s * u[k];
        const Real y = c[k] + t * v[k];
        delta[k] = x - y;
        dist2 += delta[k] * delta[k];
    }

    return SegmentClosestData{std::sqrt(dist2), s, t, delta};
}

SegmentClosestData SegmentClosestPoints(
    const std::vector<Real>& vertices,
    const Int a,
    const Int b,
    const Int c,
    const Int d
) {
    return SegmentClosestPointsRaw(
        Vec3{Coord(vertices, a, 0), Coord(vertices, a, 1), Coord(vertices, a, 2)},
        Vec3{Coord(vertices, b, 0), Coord(vertices, b, 1), Coord(vertices, b, 2)},
        Vec3{Coord(vertices, c, 0), Coord(vertices, c, 1), Coord(vertices, c, 2)},
        Vec3{Coord(vertices, d, 0), Coord(vertices, d, 1), Coord(vertices, d, 2)}
    );
}

bool EdgesShareVertex(const std::vector<Int>& edges, const Int left, const Int right) {
    const Int a = edges[2 * left + 0];
    const Int b = edges[2 * left + 1];
    const Int c = edges[2 * right + 0];
    const Int d = edges[2 * right + 1];
    return a == c || a == d || b == c || b == d;
}

struct TopologyCheckResult {
    bool safe = true;
    Real min_distance = std::numeric_limits<Real>::infinity();
    Int min_left_edge = -1;
    Int min_right_edge = -1;
    Real min_tau = 0;
    int tested_pairs = 0;
    int checked_events = 0;
    int degenerate_rejections = 0;
};

void IncludePointInBounds(const Vec3& p, Vec3& lower, Vec3& upper) {
    for (Int k = 0; k < 3; ++k) {
        lower[k] = std::min(lower[k], p[k]);
        upper[k] = std::max(upper[k], p[k]);
    }
}

void SegmentIntervalBounds(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const std::vector<Int>& edges,
    const Int edge,
    const Real tau0,
    const Real tau1,
    Vec3& lower,
    Vec3& upper
) {
    lower = Vec3{
        std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::infinity()
    };
    upper = Vec3{
        -std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity()
    };

    const Int a = edges[2 * edge + 0];
    const Int b = edges[2 * edge + 1];
    IncludePointInBounds(VertexAt(start, end, a, tau0), lower, upper);
    IncludePointInBounds(VertexAt(start, end, b, tau0), lower, upper);
    IncludePointInBounds(VertexAt(start, end, a, tau1), lower, upper);
    IncludePointInBounds(VertexAt(start, end, b, tau1), lower, upper);
}

bool SweptAabbSeparated(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const std::vector<Int>& edges,
    const Int left,
    const Int right,
    const Real tau0,
    const Real tau1,
    const Real eps
) {
    Vec3 left_lower{};
    Vec3 left_upper{};
    Vec3 right_lower{};
    Vec3 right_upper{};
    SegmentIntervalBounds(start, end, edges, left, tau0, tau1, left_lower, left_upper);
    SegmentIntervalBounds(start, end, edges, right, tau0, tau1, right_lower, right_upper);

    for (Int k = 0; k < 3; ++k) {
        if (left_upper[k] + eps < right_lower[k] || right_upper[k] + eps < left_lower[k]) {
            return true;
        }
    }
    return false;
}

SegmentClosestData SegmentClosestPointsAt(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const std::vector<Int>& edges,
    const Int left,
    const Int right,
    const Real tau
) {
    const Int a = edges[2 * left + 0];
    const Int b = edges[2 * left + 1];
    const Int c = edges[2 * right + 0];
    const Int d = edges[2 * right + 1];
    return SegmentClosestPointsRaw(
        VertexAt(start, end, a, tau),
        VertexAt(start, end, b, tau),
        VertexAt(start, end, c, tau),
        VertexAt(start, end, d, tau)
    );
}

void NoteTopologyDistance(
    TopologyCheckResult& result,
    const Real distance,
    const Int left,
    const Int right,
    const Real tau
) {
    if (distance < result.min_distance) {
        result.min_distance = distance;
        result.min_left_edge = left;
        result.min_right_edge = right;
        result.min_tau = tau;
    }
}

Vec3 Sub3(const Vec3& left, const Vec3& right) {
    return Vec3{left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

Real Dot3(const Vec3& left, const Vec3& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

Vec3 Cross3(const Vec3& left, const Vec3& right) {
    return Vec3{
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]
    };
}

Real Triple3(const Vec3& a, const Vec3& b, const Vec3& c) {
    return Dot3(a, Cross3(b, c));
}

Real EvalCubic(const std::array<Real, 4>& c, const Real t) {
    return ((c[3] * t + c[2]) * t + c[1]) * t + c[0];
}

void AppendRoot(std::vector<Real>& roots, const Real root, const Real tol = Real(1e-10)) {
    if (root < -tol || root > Real(1) + tol || !std::isfinite(root)) {
        return;
    }
    const Real clamped = Clamp01(root);
    for (const Real existing : roots) {
        if (std::abs(existing - clamped) <= Real(1e-7)) {
            return;
        }
    }
    roots.push_back(clamped);
}

std::vector<Real> QuadraticRootsInUnitInterval(const Real a, const Real b, const Real c, const Real tol) {
    std::vector<Real> roots;
    if (std::abs(a) <= tol) {
        if (std::abs(b) > tol) {
            AppendRoot(roots, -c / b);
        }
        return roots;
    }

    const Real disc = b * b - Real(4) * a * c;
    if (disc < -tol) {
        return roots;
    }
    if (std::abs(disc) <= tol) {
        AppendRoot(roots, -b / (Real(2) * a));
        return roots;
    }
    const Real sqrt_disc = std::sqrt(std::max(disc, Real(0)));
    AppendRoot(roots, (-b - sqrt_disc) / (Real(2) * a));
    AppendRoot(roots, (-b + sqrt_disc) / (Real(2) * a));
    return roots;
}

Real BisectCubicRoot(const std::array<Real, 4>& c, Real lo, Real hi) {
    Real flo = EvalCubic(c, lo);
    for (int i = 0; i < 80; ++i) {
        const Real mid = Real(0.5) * (lo + hi);
        const Real fmid = EvalCubic(c, mid);
        if (std::abs(fmid) <= Real(1e-14) || (hi - lo) <= Real(1e-13)) {
            return mid;
        }
        if ((flo <= Real(0) && fmid >= Real(0)) || (flo >= Real(0) && fmid <= Real(0))) {
            hi = mid;
        } else {
            lo = mid;
            flo = fmid;
        }
    }
    return Real(0.5) * (lo + hi);
}

std::vector<Real> CubicRootsInUnitInterval(const std::array<Real, 4>& c) {
    std::vector<Real> roots;
    Real scale = Real(0);
    for (const Real value : c) {
        scale = std::max(scale, std::abs(value));
    }
    const Real tol = std::max(Real(1e-12) * scale, Real(1e-14));

    if (std::abs(c[3]) <= tol) {
        if (std::abs(c[2]) <= tol) {
            if (std::abs(c[1]) > tol) {
                AppendRoot(roots, -c[0] / c[1]);
            }
            return roots;
        }
        return QuadraticRootsInUnitInterval(c[2], c[1], c[0], tol);
    }

    std::vector<Real> cuts{Real(0), Real(1)};
    for (const Real root : QuadraticRootsInUnitInterval(Real(3) * c[3], Real(2) * c[2], c[1], tol)) {
        if (root > Real(0) && root < Real(1)) {
            cuts.push_back(root);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(
        std::unique(cuts.begin(), cuts.end(), [](const Real a, const Real b) {
            return std::abs(a - b) <= Real(1e-10);
        }),
        cuts.end()
    );

    for (const Real cut : cuts) {
        if (std::abs(EvalCubic(c, cut)) <= tol) {
            AppendRoot(roots, cut);
        }
    }
    for (size_t i = 0; i + 1 < cuts.size(); ++i) {
        const Real lo = cuts[i];
        const Real hi = cuts[i + 1];
        const Real flo = EvalCubic(c, lo);
        const Real fhi = EvalCubic(c, hi);
        if ((flo < -tol && fhi > tol) || (flo > tol && fhi < -tol)) {
            AppendRoot(roots, BisectCubicRoot(c, lo, hi));
        }
    }
    std::sort(roots.begin(), roots.end());
    return roots;
}

Vec3 VertexDisplacement(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const Int index
) {
    return Vec3{
        Coord(end, index, 0) - Coord(start, index, 0),
        Coord(end, index, 1) - Coord(start, index, 1),
        Coord(end, index, 2) - Coord(start, index, 2)
    };
}

std::array<Real, 4> CoplanarityCubic(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const std::vector<Int>& edges,
    const Int left,
    const Int right
) {
    const Int a = edges[2 * left + 0];
    const Int b = edges[2 * left + 1];
    const Int c = edges[2 * right + 0];
    const Int d = edges[2 * right + 1];

    const Vec3 a0 = VertexAt(start, end, a, Real(0));
    const Vec3 b0 = VertexAt(start, end, b, Real(0));
    const Vec3 c0 = VertexAt(start, end, c, Real(0));
    const Vec3 d0 = VertexAt(start, end, d, Real(0));
    const Vec3 da = VertexDisplacement(start, end, a);
    const Vec3 db = VertexDisplacement(start, end, b);
    const Vec3 dc = VertexDisplacement(start, end, c);
    const Vec3 dd = VertexDisplacement(start, end, d);

    const Vec3 u0 = Sub3(b0, a0);
    const Vec3 u1 = Sub3(db, da);
    const Vec3 v0 = Sub3(d0, c0);
    const Vec3 v1 = Sub3(dd, dc);
    const Vec3 w0 = Sub3(c0, a0);
    const Vec3 w1 = Sub3(dc, da);

    return std::array<Real, 4>{
        Triple3(w0, u0, v0),
        Triple3(w1, u0, v0) + Triple3(w0, u1, v0) + Triple3(w0, u0, v1),
        Triple3(w1, u1, v0) + Triple3(w1, u0, v1) + Triple3(w0, u1, v1),
        Triple3(w1, u1, v1)
    };
}

bool MovingSegmentPairIntersectionSafe(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const std::vector<Int>& edges,
    const Int left,
    const Int right,
    const Real eps,
    TopologyCheckResult& result
) {
    if (SweptAabbSeparated(start, end, edges, left, right, Real(0), Real(1), eps)) {
        return true;
    }

    std::vector<Real> candidates{Real(0), Real(0.5), Real(1)};
    const std::array<Real, 4> cubic = CoplanarityCubic(start, end, edges, left, right);
    Real scale = Real(0);
    for (const Real value : cubic) {
        scale = std::max(scale, std::abs(value));
    }

    if (scale <= Real(1e-14)) {
        ++result.degenerate_rejections;
    } else {
        for (const Real root : CubicRootsInUnitInterval(cubic)) {
            AppendRoot(candidates, root);
        }
    }

    for (const Real tau : candidates) {
        ++result.checked_events;
        const SegmentClosestData closest = SegmentClosestPointsAt(start, end, edges, left, right, tau);
        NoteTopologyDistance(result, closest.distance, left, right, tau);
        if (closest.distance <= eps) {
            result.safe = false;
            return false;
        }
    }

    if (scale <= Real(1e-14)) {
        result.safe = false;
        return false;
    }
    return true;
}

TopologyCheckResult CheckMovingSegmentTopology(
    const std::vector<Real>& start,
    const std::vector<Real>& end,
    const std::vector<Int>& edges,
    const Real eps
) {
    TopologyCheckResult result;
    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int left = 0; left < edge_count; ++left) {
        for (Int right = left + 1; right < edge_count; ++right) {
            if (EdgesShareVertex(edges, left, right)) {
                continue;
            }
            ++result.tested_pairs;
            if (!MovingSegmentPairIntersectionSafe(
                    start,
                    end,
                    edges,
                    left,
                    right,
                    eps,
                    result
                )) {
                return result;
            }
        }
    }
    return result;
}

Real SquaredLengthEnergy(const std::vector<Real>& vertices, const std::vector<Int>& edges) {
    Real energy = 0;
    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int e = 0; e < edge_count; ++e) {
        const Int a = edges[2 * e + 0];
        const Int b = edges[2 * e + 1];
        for (Int k = 0; k < 3; ++k) {
            const Real d = vertices[3 * a + k] - vertices[3 * b + k];
            energy += d * d;
        }
    }
    return energy;
}

void AddSquaredLengthDifferential(
    const std::vector<Real>& vertices,
    const std::vector<Int>& edges,
    const Real weight,
    Real* diff
) {
    if (weight == Real(0)) {
        return;
    }

    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int e = 0; e < edge_count; ++e) {
        const Int a = edges[2 * e + 0];
        const Int b = edges[2 * e + 1];
        for (Int k = 0; k < 3; ++k) {
            const Real d = vertices[3 * a + k] - vertices[3 * b + k];
            diff[3 * a + k] += Real(2) * weight * d;
            diff[3 * b + k] -= Real(2) * weight * d;
        }
    }
}

Real CurvePolylineLength(const std::vector<Real>& vertices, const std::vector<Int>& chain) {
    Real length = 0;
    for (size_t i = 0; i + 1 < chain.size(); ++i) {
        Real squared = 0;
        const Int a = chain[i];
        const Int b = chain[i + 1];
        for (Int k = 0; k < 3; ++k) {
            const Real d = Coord(vertices, a, k) - Coord(vertices, b, k);
            squared += d * d;
        }
        length += std::sqrt(squared);
    }
    return length;
}

Real CurveLengthFloorEnergy(
    const std::vector<Real>& vertices,
    const std::vector<CurveLengthFloor>& floors
) {
    Real energy = 0;
    for (const CurveLengthFloor& floor : floors) {
        const Real length = CurvePolylineLength(vertices, floor.vertices);
        if (length > floor.floor) {
            const Real excess = length - floor.floor;
            energy += excess * excess;
        }
    }
    return energy;
}

void AddCurveLengthFloorDifferential(
    const std::vector<Real>& vertices,
    const std::vector<CurveLengthFloor>& floors,
    const Real weight,
    Real* diff
) {
    if (weight == Real(0) || floors.empty()) {
        return;
    }

    for (const CurveLengthFloor& floor : floors) {
        const Real length = CurvePolylineLength(vertices, floor.vertices);
        if (length <= floor.floor) {
            continue;
        }
        const Real coeff = Real(2) * weight * (length - floor.floor);
        for (size_t i = 0; i + 1 < floor.vertices.size(); ++i) {
            const Int a = floor.vertices[i];
            const Int b = floor.vertices[i + 1];
            Real segment_length_squared = 0;
            for (Int k = 0; k < 3; ++k) {
                const Real d = Coord(vertices, a, k) - Coord(vertices, b, k);
                segment_length_squared += d * d;
            }
            const Real segment_length = std::sqrt(segment_length_squared);
            if (segment_length <= Real(1e-12)) {
                continue;
            }
            for (Int k = 0; k < 3; ++k) {
                const Real d = Coord(vertices, a, k) - Coord(vertices, b, k);
                const Real g = coeff * d / segment_length;
                diff[3 * a + k] += g;
                diff[3 * b + k] -= g;
            }
        }
    }
}

Real BendingEnergy(
    const std::vector<Real>& vertices,
    const std::vector<std::vector<Int>>& adjacency
) {
    Real energy = 0;
    const Int vertex_count = static_cast<Int>(adjacency.size());
    for (Int i = 0; i < vertex_count; ++i) {
        if (adjacency[i].size() != 2) {
            continue;
        }
        const Int a = adjacency[i][0];
        const Int b = adjacency[i][1];
        for (Int k = 0; k < 3; ++k) {
            const Real r = Coord(vertices, a, k) - Real(2) * Coord(vertices, i, k) + Coord(vertices, b, k);
            energy += r * r;
        }
    }
    return energy;
}

void AddBendingDifferential(
    const std::vector<Real>& vertices,
    const std::vector<std::vector<Int>>& adjacency,
    const Real weight,
    Real* diff
) {
    if (weight == Real(0)) {
        return;
    }

    const Int vertex_count = static_cast<Int>(adjacency.size());
    for (Int i = 0; i < vertex_count; ++i) {
        if (adjacency[i].size() != 2) {
            continue;
        }
        const Int a = adjacency[i][0];
        const Int b = adjacency[i][1];
        for (Int k = 0; k < 3; ++k) {
            const Real r = Coord(vertices, a, k) - Real(2) * Coord(vertices, i, k) + Coord(vertices, b, k);
            diff[3 * a + k] += Real(2) * weight * r;
            diff[3 * i + k] -= Real(4) * weight * r;
            diff[3 * b + k] += Real(2) * weight * r;
        }
    }
}

Real TubeDistanceEnergy(
    const std::vector<Real>& vertices,
    const std::vector<Int>& edges,
    const Real tube_radius,
    const Real tube_gap
) {
    const Real target = Real(2) * tube_radius + tube_gap;
    if (target <= Real(0)) {
        return Real(0);
    }

    Real energy = 0;
    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int left = 0; left < edge_count; ++left) {
        const Int a = edges[2 * left + 0];
        const Int b = edges[2 * left + 1];
        for (Int right = left + 1; right < edge_count; ++right) {
            if (EdgesShareVertex(edges, left, right)) {
                continue;
            }
            const Int c = edges[2 * right + 0];
            const Int d = edges[2 * right + 1];
            const SegmentClosestData closest = SegmentClosestPoints(vertices, a, b, c, d);
            if (closest.distance < target) {
                const Real deficit = target - closest.distance;
                energy += deficit * deficit;
            }
        }
    }
    return energy;
}

void AddTubeDistanceDifferential(
    const std::vector<Real>& vertices,
    const std::vector<Int>& edges,
    const Real tube_radius,
    const Real tube_gap,
    const Real weight,
    Real* diff
) {
    const Real target = Real(2) * tube_radius + tube_gap;
    if (weight == Real(0) || target <= Real(0)) {
        return;
    }

    const Int edge_count = static_cast<Int>(edges.size() / 2);
    for (Int left = 0; left < edge_count; ++left) {
        const Int a = edges[2 * left + 0];
        const Int b = edges[2 * left + 1];
        for (Int right = left + 1; right < edge_count; ++right) {
            if (EdgesShareVertex(edges, left, right)) {
                continue;
            }

            const Int c = edges[2 * right + 0];
            const Int d = edges[2 * right + 1];
            const SegmentClosestData closest = SegmentClosestPoints(vertices, a, b, c, d);
            if (closest.distance >= target) {
                continue;
            }

            const Real distance = std::max(closest.distance, Real(1e-12));
            const Real deficit = target - closest.distance;
            const Real coeff = -Real(2) * weight * deficit / distance;
            for (Int k = 0; k < 3; ++k) {
                const Real gx = coeff * closest.delta[k];
                diff[3 * a + k] += (Real(1) - closest.s) * gx;
                diff[3 * b + k] += closest.s * gx;
                diff[3 * c + k] -= (Real(1) - closest.t) * gx;
                diff[3 * d + k] -= closest.t * gx;
            }
        }
    }
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
        const std::vector<std::vector<Int>> adjacency = Adjacency(vertex_count, curve.edges);
        const std::vector<char> explicit_pins = ReadPinnedVertices(opts.pinned_vertices, vertex_count);
        const std::vector<CurveLengthFloor> curve_length_floors =
            ReadCurveLengthFloors(opts.curve_length_floors, vertex_count);

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
            history
                << "step,accepted,energy_before,energy_after,safe_t,step_size,margin,backtracks,"
                << "dir_norm,dir_derivative,topology_enabled,topology_safe,topology_min_distance,"
                << "topology_rejections,topology_checked_pairs,topology_checked_events,"
                << "topology_min_left_edge,topology_min_right_edge,topology_min_tau,"
                << "topology_degenerate_rejections\n";
        }

        if (!opts.save_steps_dir.empty()) {
            std::filesystem::create_directories(opts.save_steps_dir);
            WriteObj(StepObjPath(opts.save_steps_dir, 0), coords, curve.edges);
        }

        int accepted = 0;
        for (int step = 1; step <= opts.steps; ++step) {
            const Real tpe_energy = tpe.Differential(mesh, diff.data());
            Real* diff_data = diff.data();
            if (opts.repulsion_weight != Real(1)) {
                for (size_t i = 0; i < coords.size(); ++i) {
                    diff_data[i] *= opts.repulsion_weight;
                }
            }
            const Real length_energy = SquaredLengthEnergy(coords, curve.edges);
            const Real curve_length_floor_energy =
                CurveLengthFloorEnergy(coords, curve_length_floors);
            const Real bend_energy = BendingEnergy(coords, adjacency);
            const Real tube_energy = TubeDistanceEnergy(coords, curve.edges, opts.tube_radius, opts.tube_gap);
            AddSquaredLengthDifferential(coords, curve.edges, opts.length_weight, diff_data);
            AddCurveLengthFloorDifferential(
                coords,
                curve_length_floors,
                opts.curve_length_floor_weight,
                diff_data
            );
            AddBendingDifferential(coords, adjacency, opts.bend_weight, diff_data);
            AddTubeDistanceDifferential(
                coords,
                curve.edges,
                opts.tube_radius,
                opts.tube_gap,
                opts.tube_weight,
                diff_data
            );
            const Real energy = opts.repulsion_weight * tpe_energy
                + opts.length_weight * length_energy
                + opts.curve_length_floor_weight * curve_length_floor_energy
                + opts.bend_weight * bend_energy
                + opts.tube_weight * tube_energy;
            tpm.Solve(
                mesh,
                Real(1), diff_data, mesh.AmbDim(),
                Real(0), gradient.data(), mesh.AmbDim(),
                mesh.AmbDim(), opts.max_iter, opts.tolerance
            );

            for (Int i = 0; i < vertex_count; ++i) {
                const bool pinned = explicit_pins[i] || (opts.pin_special_vertices && degrees[i] != 2);
                for (Int k = 0; k < 3; ++k) {
                    direction[3 * i + k] = pinned ? Real(0) : -gradient(i, k);
                }
            }

            const Real dir_norm = Norm(direction);
            const Real dir_derivative = Dot(std::vector<Real>(diff_data, diff_data + coords.size()), direction);
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
            int topology_rejections = 0;
            TopologyCheckResult topology_result;
            for (int attempt = 0; attempt <= opts.max_backtracks; ++attempt) {
                for (size_t i = 0; i < coords.size(); ++i) {
                    trial[i] = coords[i] + step_size * direction[i];
                }

                topology_result = TopologyCheckResult{};
                if (opts.topology_check) {
                    topology_result = CheckMovingSegmentTopology(
                        coords,
                        trial,
                        curve.edges,
                        opts.topology_tolerance
                    );
                    if (!topology_result.safe) {
                        ++topology_rejections;
                        step_size *= 0.5;
                        if (step_size <= opts.min_step) {
                            break;
                        }
                        continue;
                    }
                }

                mesh.SemiStaticUpdate(trial.data());
                trial_energy = opts.repulsion_weight * tpe.Value(mesh)
                    + opts.length_weight * SquaredLengthEnergy(trial, curve.edges)
                    + opts.curve_length_floor_weight
                        * CurveLengthFloorEnergy(trial, curve_length_floors)
                    + opts.bend_weight * BendingEnergy(trial, adjacency)
                    + opts.tube_weight * TubeDistanceEnergy(trial, curve.edges, opts.tube_radius, opts.tube_gap);

                const Real target = energy + opts.armijo * step_size * dir_derivative;
                if (trial_energy <= target) {
                    coords = trial;
                    ++accepted;
                    accepted_step = true;
                    if (history) {
                        const Real topology_min_distance =
                            std::isfinite(topology_result.min_distance) ? topology_result.min_distance : Real(-1);
                        history << step << ",1,"
                                << energy << ","
                                << trial_energy << ","
                                << safe_t << ","
                                << step_size << ","
                                << (safe_t - step_size) << ","
                                << attempt << ","
                                << dir_norm << ","
                                << dir_derivative << ","
                                << (opts.topology_check ? 1 : 0) << ","
                                << (topology_result.safe ? 1 : 0) << ","
                                << topology_min_distance << ","
                                << topology_rejections << ","
                                << topology_result.tested_pairs << ","
                                << topology_result.checked_events << ","
                                << topology_result.min_left_edge << ","
                                << topology_result.min_right_edge << ","
                                << topology_result.min_tau << ","
                                << topology_result.degenerate_rejections << "\n";
                    }
                    if (!opts.save_steps_dir.empty()) {
                        WriteObj(StepObjPath(opts.save_steps_dir, accepted), coords, curve.edges);
                    }
                    std::cout << "step " << step
                              << " accepted energy " << energy << " -> " << trial_energy
                              << " safe_t " << safe_t
                              << " step_size " << step_size
                              << " backtracks " << attempt
                              << " topology_rejections " << topology_rejections << "\n";
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
                    const Real topology_min_distance =
                        std::isfinite(topology_result.min_distance) ? topology_result.min_distance : Real(-1);
                    history << step << ",0,"
                            << energy << ","
                            << trial_energy << ","
                            << safe_t << ","
                            << step_size << ","
                            << (safe_t - step_size) << ","
                            << opts.max_backtracks << ","
                            << dir_norm << ","
                            << dir_derivative << ","
                            << (opts.topology_check ? 1 : 0) << ","
                            << (topology_result.safe ? 1 : 0) << ","
                            << topology_min_distance << ","
                            << topology_rejections << ","
                            << topology_result.tested_pairs << ","
                            << topology_result.checked_events << ","
                            << topology_result.min_left_edge << ","
                            << topology_result.min_right_edge << ","
                            << topology_result.min_tau << ","
                            << topology_result.degenerate_rejections << "\n";
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
