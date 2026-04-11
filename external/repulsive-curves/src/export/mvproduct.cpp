#include "export/mvproduct.h"

#include <Eigen/Core>
#include "geometrycentral/utilities/vector3.h"
#include "poly_curve_network.h"
#include "spatial/tpe_bvh.h"
#include "product/block_cluster_tree.h"
#include "tpe_flow_sc.h"
#include "flow/gradient_constraint_enum.h"

using namespace geometrycentral;

namespace LWS
{
PolyCurveNetwork *createCurveNetwork(std::vector<std::array<double, 3>> &positions, std::vector<std::array<size_t, 2>> &edges)
{
    std::vector<Vector3> vectors(positions.size());
    for (size_t i = 0; i < positions.size(); i++)
    {
        vectors[i] = Vector3{positions[i][0], positions[i][1], positions[i][2]};
    }
    return new PolyCurveNetwork(vectors, edges);
}

BVHNode3D *createBVHForEnergy(PolyCurveNetwork *curves)
{
    return CreateBVHFromCurve(curves);
}

BlockClusterTree *createBlockClusterTree(PolyCurveNetwork *curves, double sep, double alpha, double beta)
{
    BVHNode3D *edgeBVH = CreateEdgeBVHFromCurve(curves);
    BlockClusterTree *tree = new BlockClusterTree(curves, edgeBVH, sep, alpha, beta, 0);
    // This block cluster tree is only going to multiply the dense upper-left block, with no duplication of entries.
    tree->SetBlockTreeMode(BlockTreeMode::MatrixOnly);
    return tree;
}

void multiplyMetricWithVector(BlockClusterTree *tree, std::vector<double> &vec, std::vector<double> &output)
{
    // Copy input to an Eigen matrix
    Eigen::VectorXd in(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
    {
        in(i) = vec[i];
    }
    // Set up an Eigen matrix as output
    Eigen::VectorXd out(vec.size());
    out.setZero();

    // Call the block cluster tree routine
    tree->Multiply(in, out);

    // Copy result to the output
    for (size_t i = 0; i < vec.size(); i++)
    {
        output[i] = out(i);
    }
}

double evaluateEnergy(PolyCurveNetwork *curve, BVHNode3D *root, double alpha, double beta)
{
    return SpatialTree::TPEnergyBH(curve, root, alpha, beta);
}

void evaluateGradient(PolyCurveNetwork *curve, BVHNode3D *root, std::vector<std::array<double, 3>> &out, double alpha, double beta)
{
    // Set up an Eigen matrix for the computation to use
    Eigen::MatrixXd grad(out.size(), 3);
    grad.setZero();

    // Use the BVH routine
    SpatialTree::TPEGradientBarnesHut(curve, root, grad, alpha, beta);

    // Copy result to the output
    for (size_t i = 0; i < out.size(); i++)
    {
        out[i][0] = grad(i, 0);
        out[i][1] = grad(i, 1);
        out[i][2] = grad(i, 2);
    }
}

} // namespace LWS

extern "C" {

int runRepulsiveGraphLayout(
    double* positions,
    size_t num_vertices,
    const size_t* edges,
    size_t num_edges,
    int steps,
    double alpha,
    double beta,
    int use_sobolev,
    int use_multigrid,
    int use_barnes_hut,
    int use_backprojection)
{
    using namespace LWS;

    if (positions == nullptr || edges == nullptr || num_vertices == 0) {
        return -1;
    }

    std::vector<Vector3> vectors(num_vertices);
    for (size_t i = 0; i < num_vertices; ++i) {
        vectors[i] = Vector3{
            positions[3 * i + 0],
            positions[3 * i + 1],
            positions[3 * i + 2]
        };
    }

    std::vector<std::array<size_t, 2>> edge_list(num_edges);
    for (size_t i = 0; i < num_edges; ++i) {
        edge_list[i] = {edges[2 * i + 0], edges[2 * i + 1]};
    }

    PolyCurveNetwork curves(vectors, edge_list);
    curves.appliedConstraints.push_back(ConstraintType::Barycenter);
    curves.appliedConstraints.push_back(ConstraintType::EdgeLengths);

    TPEFlowSolverSC solver(&curves, alpha, beta);

    int step_limit = (steps > 0) ? steps : 1;
    int stuck_iterations = 0;

    for (int step = 0; step < step_limit; ++step) {
        bool good_step = false;

        if (use_sobolev) {
            if (use_multigrid) {
                good_step = solver.StepSobolevLSIterative(0.0, use_backprojection != 0);
            } else {
                good_step = solver.StepSobolevLS(use_barnes_hut != 0, use_backprojection != 0);
            }
        } else {
            good_step = solver.StepLS(use_barnes_hut != 0);
        }

        if (solver.soboNormZero) {
            break;
        }

        if (!good_step) {
            ++stuck_iterations;
            if (stuck_iterations >= 5 && solver.TargetLengthReached()) {
                break;
            }
        } else {
            stuck_iterations = 0;
        }
    }

    for (size_t i = 0; i < num_vertices; ++i) {
        Vector3 p = curves.GetVertex(i)->Position();
        positions[3 * i + 0] = p.x;
        positions[3 * i + 1] = p.y;
        positions[3 * i + 2] = p.z;
    }

    return 0;
}

}
