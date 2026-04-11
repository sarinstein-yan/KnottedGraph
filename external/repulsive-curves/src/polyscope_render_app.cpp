#include "json/json.hpp"
#include "polyscope/curve_network.h"
#include "polyscope/gl/ground_plane.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/view.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct RenderOptions {
  std::string layoutPath;
  std::string outputPath;
  std::string palette = "default";
  float curveRadius = 0.015f;
  float nodeRadius = 0.035f;
  int width = 1600;
  int height = 1200;
  double yawDegrees = 0.0;
  double pitchDegrees = 0.0;
  bool transparent = false;
};

struct LayoutData {
  std::vector<std::string> nodeOrder;
  std::map<std::string, glm::vec3> nodePositions;
  std::vector<std::array<std::string, 2>> edgeOrder;
  std::map<std::string, std::vector<glm::vec3>> edgePolylines;
};

glm::vec3 rgb(unsigned int r, unsigned int g, unsigned int b) {
  return glm::vec3(r / 255.f, g / 255.f, b / 255.f);
}

std::vector<glm::vec3> paletteColors(const std::string& paletteName, size_t count) {
  std::vector<glm::vec3> colors;
  if (paletteName == "k5") {
    colors = {
        rgb(181, 154, 24),
        rgb(39, 95, 178),
        rgb(139, 44, 168),
        rgb(177, 34, 46),
        rgb(34, 168, 75),
    };
  } else if (paletteName == "k33") {
    colors = {
        rgb(177, 34, 46),
        rgb(177, 34, 46),
        rgb(177, 34, 46),
        rgb(39, 95, 178),
        rgb(39, 95, 178),
        rgb(39, 95, 178),
    };
  } else {
    colors = {
        rgb(31, 119, 180),
        rgb(255, 127, 14),
        rgb(44, 160, 44),
        rgb(214, 39, 40),
        rgb(148, 103, 189),
        rgb(140, 86, 75),
        rgb(227, 119, 194),
        rgb(127, 127, 127),
        rgb(188, 189, 34),
        rgb(23, 190, 207),
    };
  }

  if (colors.empty()) {
    return colors;
  }
  std::vector<glm::vec3> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.push_back(colors[i % colors.size()]);
  }
  return out;
}

std::string edgeKey(const std::string& u, const std::string& v) {
  return u + "::" + v;
}

glm::vec3 parseVec3(const json& entry) {
  if (!entry.is_array() || entry.size() != 3) {
    throw std::runtime_error("Expected a length-3 coordinate array");
  }
  return glm::vec3(entry[0].get<float>(), entry[1].get<float>(), entry[2].get<float>());
}

double dist2(const glm::vec3& a, const glm::vec3& b) {
  glm::vec3 d = a - b;
  return glm::dot(d, d);
}

LayoutData loadLayout(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open layout file: " + path);
  }

  json data;
  in >> data;

  LayoutData layout;
  for (const auto& node : data.at("node_order")) {
    layout.nodeOrder.push_back(node.get<std::string>());
  }

  const json& nodePositions = data.at("node_positions_final");
  for (const std::string& node : layout.nodeOrder) {
    layout.nodePositions[node] = parseVec3(nodePositions.at(node));
  }

  for (const auto& edge : data.at("edge_order")) {
    if (!edge.is_array() || edge.size() != 2) {
      throw std::runtime_error("Expected edge_order entries to have length 2");
    }
    std::array<std::string, 2> e = {edge[0].get<std::string>(), edge[1].get<std::string>()};
    layout.edgeOrder.push_back(e);
  }

  const json& edgePolylines = data.at("edge_polylines_final");
  for (const auto& edge : layout.edgeOrder) {
    std::string key = edgeKey(edge[0], edge[1]);
    if (!edgePolylines.count(key)) {
      std::string reverseKey = edgeKey(edge[1], edge[0]);
      if (!edgePolylines.count(reverseKey)) {
        throw std::runtime_error("Missing polyline for edge: " + key);
      }
      key = reverseKey;
    }

    std::vector<glm::vec3> polyline;
    for (const auto& point : edgePolylines.at(key)) {
      polyline.push_back(parseVec3(point));
    }
    if (polyline.size() < 2) {
      throw std::runtime_error("Polyline has fewer than 2 points for edge: " + key);
    }
    layout.edgePolylines[edgeKey(edge[0], edge[1])] = polyline;
  }

  return layout;
}

void appendCurveNetworkGeometry(const LayoutData& layout, std::vector<glm::vec3>& curveNodes,
                                std::vector<std::array<size_t, 2>>& curveEdges) {
  curveNodes.clear();
  curveEdges.clear();

  std::map<std::string, size_t> nodeIndex;
  for (const std::string& node : layout.nodeOrder) {
    nodeIndex[node] = curveNodes.size();
    curveNodes.push_back(layout.nodePositions.at(node));
  }

  for (const auto& edge : layout.edgeOrder) {
    const std::string& u = edge[0];
    const std::string& v = edge[1];

    std::vector<glm::vec3> polyline = layout.edgePolylines.at(edgeKey(u, v));
    const glm::vec3& uPos = layout.nodePositions.at(u);
    const glm::vec3& vPos = layout.nodePositions.at(v);

    double directCost = dist2(polyline.front(), uPos) + dist2(polyline.back(), vPos);
    double reverseCost = dist2(polyline.front(), vPos) + dist2(polyline.back(), uPos);
    if (reverseCost < directCost) {
      std::reverse(polyline.begin(), polyline.end());
    }

    size_t prevIndex = nodeIndex.at(u);
    for (size_t i = 1; i + 1 < polyline.size(); ++i) {
      curveNodes.push_back(polyline[i]);
      size_t currentIndex = curveNodes.size() - 1;
      curveEdges.push_back({prevIndex, currentIndex});
      prevIndex = currentIndex;
    }
    curveEdges.push_back({prevIndex, nodeIndex.at(v)});
  }
}

void applyTurntableRotation(double yawDegrees, double pitchDegrees) {
  const double kPi = 3.14159265358979323846;
  if (std::abs(yawDegrees) < 1e-9 && std::abs(pitchDegrees) < 1e-9) {
    return;
  }

  glm::vec3 lookDir, upDir, rightDir;
  polyscope::view::getCameraFrame(lookDir, upDir, rightDir);

  polyscope::view::viewMat = glm::translate(polyscope::view::viewMat, polyscope::state::center);
  if (std::abs(pitchDegrees) >= 1e-9) {
    polyscope::view::viewMat =
        polyscope::view::viewMat *
        glm::rotate(glm::mat4x4(1.f), static_cast<float>(-pitchDegrees * kPi / 180.0), rightDir);
  }
  if (std::abs(yawDegrees) >= 1e-9) {
    polyscope::view::viewMat =
        polyscope::view::viewMat *
        glm::rotate(glm::mat4x4(1.f), static_cast<float>(yawDegrees * kPi / 180.0), glm::vec3(0.f, 1.f, 0.f));
  }
  polyscope::view::viewMat = glm::translate(polyscope::view::viewMat, -polyscope::state::center);
  polyscope::requestRedraw();
}

void printUsage() {
  std::cerr << "Usage: rcurves_polyscope_render <layout.json> --output <out.png> "
               "[--palette default|k5|k33] [--curve-radius 0.015] [--node-radius 0.035] "
               "[--width 1600] [--height 1200] [--yaw 0] [--pitch 0] [--transparent]"
            << std::endl;
}

RenderOptions parseArgs(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    throw std::runtime_error("Missing layout path");
  }

  RenderOptions options;
  options.layoutPath = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto requireValue = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--output") {
      options.outputPath = requireValue(arg);
    } else if (arg == "--palette") {
      options.palette = requireValue(arg);
    } else if (arg == "--curve-radius") {
      options.curveRadius = std::stof(requireValue(arg));
    } else if (arg == "--node-radius") {
      options.nodeRadius = std::stof(requireValue(arg));
    } else if (arg == "--width") {
      options.width = std::stoi(requireValue(arg));
    } else if (arg == "--height") {
      options.height = std::stoi(requireValue(arg));
    } else if (arg == "--yaw") {
      options.yawDegrees = std::stod(requireValue(arg));
    } else if (arg == "--pitch") {
      options.pitchDegrees = std::stod(requireValue(arg));
    } else if (arg == "--transparent") {
      options.transparent = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.outputPath.empty()) {
    throw std::runtime_error("Missing required --output");
  }
  return options;
}

} // namespace

int main(int argc, char** argv) {
  try {
    RenderOptions options = parseArgs(argc, argv);
    LayoutData layout = loadLayout(options.layoutPath);

    std::vector<glm::vec3> curveNodes;
    std::vector<std::array<size_t, 2>> curveEdges;
    appendCurveNetworkGeometry(layout, curveNodes, curveEdges);

    std::vector<glm::vec3> pointNodes;
    pointNodes.reserve(layout.nodeOrder.size());
    for (const std::string& node : layout.nodeOrder) {
      pointNodes.push_back(layout.nodePositions.at(node));
    }
    std::vector<glm::vec3> pointColors = paletteColors(options.palette, pointNodes.size());

    polyscope::options::programName = "Repulsive Curves Renderer";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::errorsThrowExceptions = true;
    polyscope::options::openImGuiWindowForUserCallback = false;
    polyscope::gl::groundPlaneEnabled = false;
    polyscope::view::windowWidth = options.width;
    polyscope::view::windowHeight = options.height;
    polyscope::view::bgColor = {{1.f, 1.f, 1.f, 1.f}};

    polyscope::init();

    polyscope::CurveNetwork* curve = polyscope::registerCurveNetwork("repulsive curves", curveNodes, curveEdges);
    curve->baseColor = glm::vec3(0.74f, 0.74f, 0.74f);
    curve->radius = options.curveRadius;

    polyscope::PointCloud* points = polyscope::registerPointCloud("graph nodes", pointNodes);
    points->pointRadius = options.nodeRadius;
    if (!pointColors.empty()) {
      points->addColorQuantity("node colors", pointColors)->setEnabled(true);
    }

    polyscope::view::resetCameraToHomeView();
    applyTurntableRotation(options.yawDegrees, options.pitchDegrees);
    polyscope::screenshot(options.outputPath, options.transparent);

    std::cout << "Wrote: " << options.outputPath << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Render failed: " << ex.what() << std::endl;
    return 1;
  }
}
