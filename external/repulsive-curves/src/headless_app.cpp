#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <omp.h>

#include "curve_io.h"
#include "extra_potentials.h"
#include "obstacles/mesh_obstacle.h"
#include "obstacles/plane_obstacle.h"
#include "scene_file.h"
#include "tpe_flow_sc.h"

#include "geometrycentral/surface/meshio.h"

using geometrycentral::Vector3;
using geometrycentral::surface::HalfedgeMesh;
using geometrycentral::surface::VertexPositionGeometry;
using geometrycentral::surface::loadMesh;

namespace {

struct CliOptions {
  std::string scene_file;
  std::string output_file = "output.obj";
  int steps = -1;
  bool use_sobolev = true;
  bool use_multigrid = false;
  bool use_barnes_hut = false;
  bool use_backprojection = true;
};

void printUsage() {
  std::cout
      << "Usage: rcurves_headless <scene.txt> [--output final.obj] [--steps N]\n"
      << "       [--l2] [--multigrid] [--barnes-hut] [--no-backprojection]\n";
}

CliOptions parseArgs(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    std::exit(1);
  }

  CliOptions opts;
  opts.scene_file = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) {
      opts.output_file = argv[++i];
    } else if (arg == "--steps" && i + 1 < argc) {
      opts.steps = std::stoi(argv[++i]);
    } else if (arg == "--l2") {
      opts.use_sobolev = false;
    } else if (arg == "--multigrid") {
      opts.use_multigrid = true;
    } else if (arg == "--barnes-hut") {
      opts.use_barnes_hut = true;
    } else if (arg == "--no-backprojection") {
      opts.use_backprojection = false;
    } else if (arg == "-h" || arg == "--help") {
      printUsage();
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      printUsage();
      std::exit(1);
    }
  }

  return opts;
}

std::string normalizeScenePath(std::string path) {
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '\\') {
      path[i] = '/';
    }
  }
  return path;
}

std::unique_ptr<LWS::PolyCurveNetwork> loadCurveNetwork(const std::string& filename) {
  std::vector<Vector3> all_positions;
  std::vector<std::array<size_t, 2>> all_edges;
  LWS::CurveIO::readVerticesAndEdges(filename, all_positions, all_edges);

  if (all_edges.empty()) {
    std::cout << "Did not find OBJ line elements; reading edges from faces instead" << std::endl;
    LWS::CurveIO::readFaces(filename, all_edges);
  }

  return std::unique_ptr<LWS::PolyCurveNetwork>(new LWS::PolyCurveNetwork(all_positions, all_edges));
}

void applySceneConstraints(LWS::PolyCurveNetwork& curves, const LWS::SceneData& scene) {
  for (LWS::ConstraintType type : scene.constraints) {
    curves.appliedConstraints.push_back(type);
  }
  for (int i : scene.pinnedVertices) {
    curves.PinVertex(i);
  }
  for (int i : scene.pinnedTangents) {
    curves.PinTangent(i);
  }

  curves.pinnedAllToSurface = false;

  if (scene.pinSpecialVertices) {
    curves.PinAllSpecialVertices(scene.pinSpecialTangents);
  } else if (scene.pinEndpointVertices) {
    curves.PinAllEndpoints(scene.pinSpecialTangents);
  }

  if (scene.constraintSurface) {
    curves.constraintSurface = scene.constraintSurface;
  }

  if (scene.constrainAllToSurface) {
    for (int i = 0; i < curves.NumVertices(); ++i) {
      curves.PinToSurface(i);
    }
    curves.pinnedAllToSurface = true;
  } else if (scene.constrainEndpointsToSurface) {
    for (int i = 0; i < curves.NumVertices(); ++i) {
      auto* v = curves.GetVertex(i);
      if (v->numEdges() == 1) {
        curves.PinToSurface(i);
      }
    }
  } else {
    for (int i : scene.surfaceConstrainedVertices) {
      curves.PinToSurface(i);
    }
  }
}

void addSceneObjectives(LWS::TPEFlowSolverSC& solver, const LWS::SceneData& scene) {
  const double p_exp = scene.tpe_beta - scene.tpe_alpha;

  for (const auto& plane : scene.planes) {
    solver.obstacles.push_back(new LWS::PlaneObstacle(plane.center, plane.normal, p_exp, plane.weight));
  }

  for (const auto& obstacle : scene.obstacles) {
    std::shared_ptr<HalfedgeMesh> mesh;
    std::shared_ptr<VertexPositionGeometry> geom;
    std::tie(mesh, geom) = loadMesh(obstacle.filename);
    geom->requireVertexPositions();
    solver.obstacles.push_back(new LWS::MeshObstacle(mesh, geom, p_exp, obstacle.weight));
  }

  for (const auto& potential : scene.extraPotentials) {
    switch (potential.type) {
      case LWS::PotentialType::Length:
        solver.potentials.push_back(new LWS::TotalLengthPotential(potential.weight));
        break;
      case LWS::PotentialType::LengthDiff:
        solver.potentials.push_back(new LWS::LengthDifferencePotential(potential.weight));
        break;
      case LWS::PotentialType::PinAngles:
        solver.potentials.push_back(new LWS::PinBendingPotential(potential.weight));
        break;
      case LWS::PotentialType::VectorField:
        if (potential.extraInfo == "constant") {
          solver.potentials.push_back(
              new LWS::VectorFieldPotential(potential.weight, new LWS::ConstantVectorField(Vector3{1, 0, 1})));
        } else if (potential.extraInfo == "circular") {
          solver.potentials.push_back(new LWS::VectorFieldPotential(potential.weight, new LWS::CircularVectorField()));
        } else if (potential.extraInfo == "interesting") {
          solver.potentials.push_back(
              new LWS::VectorFieldPotential(potential.weight, new LWS::InterestingVectorField()));
        } else {
          std::cerr << "Invalid vector field: " << potential.extraInfo << std::endl;
          std::exit(1);
        }
        break;
      case LWS::PotentialType::Area:
        std::cerr << "Area potential is not implemented" << std::endl;
        std::exit(1);
    }
  }

  if (scene.useLengthScale && scene.edgeLengthScale != 1) {
    solver.SetEdgeLengthScaleTarget(scene.edgeLengthScale);
  } else if (scene.useTotalLengthScale && scene.totalLengthScale != 1) {
    solver.SetTotalLengthScaleTarget(scene.totalLengthScale);
  }
}

void ensureParentDirectory(const std::string& filename) {
  std::string::size_type pos = filename.find_last_of("/\\");
  if (pos == std::string::npos) {
    return;
  }
  std::string dir = filename.substr(0, pos);
  if (dir.empty()) {
    return;
  }
  std::string command = "mkdir \"" + dir + "\" 2>nul";
  std::system(command.c_str());
}

void writeCurve(LWS::PolyCurveNetwork& network, const std::string& filename) {
  std::vector<Vector3> positions(network.NumVertices());
  std::vector<std::vector<size_t>> edges(network.NumEdges());

  for (int i = 0; i < network.NumVertices(); ++i) {
    positions[i] = network.GetVertex(i)->Position();
  }
  for (int i = 0; i < network.NumEdges(); ++i) {
    auto* edge = network.GetEdge(i);
    edges[i] = {static_cast<size_t>(edge->prevVert->GlobalIndex()), static_cast<size_t>(edge->nextVert->GlobalIndex())};
  }

  ensureParentDirectory(filename);
  LWS::CurveIO::writeOBJLineElements(filename.c_str(), positions, edges);
}

}  // namespace

int main(int argc, char** argv) {
  int default_threads = omp_get_max_threads();
  int used_threads = default_threads / 2 + 2;
  omp_set_num_threads(used_threads);
  std::cout << "OMP autodetected " << default_threads << " max threads; using " << used_threads << std::endl;

  CliOptions opts = parseArgs(argc, argv);

  LWS::SceneData scene = LWS::ParseSceneFile(normalizeScenePath(opts.scene_file));
  auto curves = loadCurveNetwork(scene.curve_filename);
  applySceneConstraints(*curves, scene);

  if (curves->appliedConstraints.empty()) {
    curves->appliedConstraints.push_back(LWS::ConstraintType::Barycenter);
    curves->appliedConstraints.push_back(LWS::ConstraintType::EdgeLengths);
  }

  LWS::TPEFlowSolverSC solver(curves.get(), scene.tpe_alpha, scene.tpe_beta);
  addSceneObjectives(solver, scene);

  int step_limit = opts.steps > 0 ? opts.steps : scene.iterationLimit;
  if (step_limit <= 0) {
    step_limit = 500;
  }
  std::cout << "Solver options: sobolev=" << opts.use_sobolev
            << " multigrid=" << opts.use_multigrid
            << " barnes_hut=" << opts.use_barnes_hut
            << " backprojection=" << opts.use_backprojection << std::endl;

  int stuck_iterations = 0;
  for (int step = 1; step <= step_limit; ++step) {
    bool good_step = false;
    if (opts.use_sobolev) {
      if (opts.use_multigrid) {
        good_step = solver.StepSobolevLSIterative(0.0, opts.use_backprojection);
      } else {
        good_step = solver.StepSobolevLS(opts.use_barnes_hut, opts.use_backprojection);
      }
    } else {
      good_step = solver.StepLS(opts.use_barnes_hut);
    }

    if (solver.soboNormZero) {
      std::cout << "Stopped at step " << step << ": near local minimum" << std::endl;
      break;
    }

    if (!good_step) {
      ++stuck_iterations;
      if (stuck_iterations >= 5 && solver.TargetLengthReached()) {
        std::cout << "Stopped at step " << step << ": no progress" << std::endl;
        break;
      }
    } else {
      stuck_iterations = 0;
    }

    if (step == step_limit) {
      std::cout << "Stopped at step limit " << step_limit << std::endl;
    }
  }

  writeCurve(*curves, opts.output_file);
  std::cout << "Wrote final curve to " << opts.output_file << std::endl;
  return 0;
}
