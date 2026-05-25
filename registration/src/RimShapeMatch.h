#ifndef RIM_SHAPE_MATCH_H
#define RIM_SHAPE_MATCH_H

// =====================================================================
// RimShapeMatch.h
//   Phase 7b "RIM Shape Match" — global initial-pose search by aligning
//   source rim curve to target boundary curve.
//
//   Implementation is staged behind the W-key family:
//     Plain W        : Step 1  (this file: walkRimChain)
//     Shift+W        : Step 2  (planned: target boundary 2D trace + lift)
//     Ctrl+W         : Step 3  (planned: chamfer cost sampling)
//     Ctrl+Shift+W   : Step 4  (planned: full Shape Match + Live bridge)
//
//   Step 1 (Plain W):
//     Given the source mesh and its per-vertex region labels (after
//     LiverRegionLabel::labelVertices + cleanupRimCC), order the RIM
//     vertices into a 1-D vertex-index chain via a greedy walk along
//     the rim subgraph. The chain is used by the green debug overlay
//     in main.cpp and, in later steps, as the input to the Shape-Match
//     cost function.
//
//   Why a chain (and not just a point set):
//     The Shape-Match cost (chamfer + tangent agreement) needs an
//     *ordering* on the source rim so that local tangents can be
//     defined as (pos[i+1] - pos[i-1]). The order also makes the
//     visualisation interpretable — adjacent verts in the array are
//     adjacent in screen space, so any "kinks" expose ordering bugs.
//
//   Dependencies:
//     mCutMesh.h           — source mesh storage layout
//     LiverRegionLabel.h   — RIM enum value, buildUniqueEdges,
//                            buildCSRAdjacency
// =====================================================================

#include <vector>
#include <ostream>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>
#include <cmath>
#include <limits>
#include <queue>           // GN: BFS for unsigned boundary distance map
#include <string>          // Phase 7b Step 3d: SilhouetteSweepState::fail_reason

#include <Eigen/Dense>
#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "LiverRegionLabel.h"
#include "DepthUtils.h"            // BoundaryDistMap, g_boundaryDistMap (Phase 7b Step 3a 2D)

namespace RimShape {

// ---------------------------------------------------------------------
// walkRimChain
//   Orders the RIM vertices of `mesh` into a 1-D vertex-index chain by
//   greedily following the RIM subgraph.
//
// Preconditions
//   - labels.size() == mesh vertex count
//   - LiverRegionLabel::cleanupRimCC has been applied, so verts marked
//     LiverRegionLabel::RIM form (essentially) one connected band — the
//     largest rim CC. Small CCs have been re-classified to ANTERIOR or
//     POSTERIOR already.
//
// Outputs
//   chain_out          : ordered vertex indices forming the chain
//   closed_out         : true iff the last chain vertex is a rim-graph
//                        neighbour of the seed (i.e. a closed loop)
//   n_rim_total_out    : total RIM-labelled vertices in the mesh
//                        (chain may be smaller — see "branchy bands")
//   seed_out           : the seed vertex picked (for diagnostics)
//
// Algorithm
//   1. Build full-mesh CSR adjacency.
//   2. Compute per-vertex "rim-degree" = number of neighbours that are
//      themselves RIM. Most rim verts on a clean band have rim-degree
//      exactly 2 (chain interior); endpoints have rim-degree 1 (open
//      chain); junctions in a thicker band have rim-degree >= 3.
//   3. Seed pick — prefer rim-degree 1 (an endpoint walks the full
//      chain in a single pass), else rim-degree 2 (closed loop), else
//      higher degree, else give up.
//   4. Greedy walk from seed: at each step move to the unvisited
//      rim-neighbour with the smallest rim-degree. Tie-break by
//      smallest vertex index (deterministic). This biases the walk
//      down skinny tails first instead of plunging into a junction.
//   5. If the seed is in the chain interior (i.e. the seed had >=2
//      rim-neighbours, both got visited from the same side because
//      greedy picked one first), do a second walk in the opposite
//      direction by reversing what we have and continuing from the
//      old seed into its unvisited rim-neighbour, if any.
//   6. Mark "closed" if the last chain vertex is a rim-graph neighbour
//      of the seed.
//
// Branchy bands (rim-degree >= 3 verts present) are handled gracefully:
// the walk takes one branch through each junction, so the returned
// chain is a single thread through the band. Any leftover rim verts
// (visible as "n_rim_total - chain_size > 0" in logs) are reported
// but not chained — they will appear as missing dots in the green
// overlay, which is the desired diagnostic signal.
//
// Cost: O(nV + nE) for adjacency, O(|chain|) for the walks.
// Returns false iff labels.size() mismatches or no RIM vertices exist.
// ---------------------------------------------------------------------
inline bool walkRimChain(const mCutMesh& mesh,
                         const std::vector<uint8_t>& labels,
                         std::vector<int>& chain_out,
                         bool& closed_out,
                         int& n_rim_total_out,
                         int& seed_out,
                         std::ostream* log = nullptr)
{
    chain_out.clear();
    closed_out = false;
    n_rim_total_out = 0;
    seed_out = -1;

    const int nV = (int)(mesh.mVertices.size() / 3);
    if ((int)labels.size() != nV) {
        if (log) (*log) << "[RimChain] ERROR labels.size=" << labels.size()
                   << " != nV=" << nV << std::endl;
        return false;
    }

    // 1. Full-mesh CSR adjacency (re-use helpers in LiverRegionLabel)
    auto edges = LiverRegionLabel::buildUniqueEdges(mesh);
    std::vector<int> off, ne;
    LiverRegionLabel::buildCSRAdjacency(nV, edges, off, ne);

    // 2. rim-degree per vertex (only meaningful for RIM verts)
    std::vector<int> rim_deg(nV, 0);
    int n_rim = 0;
    for (int i = 0; i < nV; i++) {
        if (labels[i] != LiverRegionLabel::RIM) continue;
        n_rim++;
        for (int k = off[i]; k < off[i + 1]; k++) {
            if (labels[ne[k]] == LiverRegionLabel::RIM) rim_deg[i]++;
        }
    }
    n_rim_total_out = n_rim;
    if (n_rim == 0) {
        if (log) (*log) << "[RimChain] no RIM vertices in labels" << std::endl;
        return false;
    }

    // 3. seed pick: prefer low rim-degree (endpoint > clean-chain >
    // junction > isolated). Isolated (rim_deg==0) verts cannot start
    // a meaningful chain, so they are last resort.
    int seed = -1;
    for (int target_deg : {1, 2, 3, 4, 5, 0}) {
        for (int i = 0; i < nV; i++) {
            if (labels[i] != LiverRegionLabel::RIM) continue;
            if (rim_deg[i] != target_deg) continue;
            seed = i;
            break;
        }
        if (seed >= 0) break;
    }
    if (seed < 0) {
        if (log) (*log) << "[RimChain] ERROR no usable seed" << std::endl;
        return false;
    }
    seed_out = seed;

    // 4. Greedy forward walk
    std::vector<uint8_t> visited(nV, 0);
    chain_out.reserve(n_rim);
    {
        int cur = seed;
        while (cur >= 0) {
            chain_out.push_back(cur);
            visited[cur] = 1;
            int best = -1, best_deg = INT_MAX;
            for (int k = off[cur]; k < off[cur + 1]; k++) {
                int v = ne[k];
                if (labels[v] != LiverRegionLabel::RIM) continue;
                if (visited[v]) continue;
                int d = rim_deg[v];
                // tie-break by smaller vertex index for determinism
                if (d < best_deg || (d == best_deg && (best < 0 || v < best))) {
                    best_deg = d; best = v;
                }
            }
            cur = best;
        }
    }

    // 5. If seed had unvisited rim-neighbours (i.e. seed was in chain
    // interior, e.g. rim_deg==2 and one neighbour got picked first),
    // reverse the current chain and continue from the (originally
    // seed) end into its other rim-neighbour.
    {
        int other = -1, other_deg = INT_MAX;
        for (int k = off[seed]; k < off[seed + 1]; k++) {
            int v = ne[k];
            if (labels[v] != LiverRegionLabel::RIM) continue;
            if (visited[v]) continue;
            int d = rim_deg[v];
            if (d < other_deg || (d == other_deg && (other < 0 || v < other))) {
                other_deg = d; other = v;
            }
        }
        if (other >= 0) {
            std::reverse(chain_out.begin(), chain_out.end());
            int cur = other;
            while (cur >= 0) {
                chain_out.push_back(cur);
                visited[cur] = 1;
                int best = -1, best_deg2 = INT_MAX;
                for (int k = off[cur]; k < off[cur + 1]; k++) {
                    int v = ne[k];
                    if (labels[v] != LiverRegionLabel::RIM) continue;
                    if (visited[v]) continue;
                    int d = rim_deg[v];
                    if (d < best_deg2 || (d == best_deg2 && (best < 0 || v < best))) {
                        best_deg2 = d; best = v;
                    }
                }
                cur = best;
            }
        }
    }

    // 6. closed?
    if ((int)chain_out.size() >= 3) {
        const int last = chain_out.back();
        for (int k = off[last]; k < off[last + 1]; k++) {
            if (ne[k] == seed) { closed_out = true; break; }
        }
    }

    // Diagnostics
    if (log) {
        int d0=0, d1=0, d2=0, d3=0, d4p=0;
        for (int i = 0; i < nV; i++) {
            if (labels[i] != LiverRegionLabel::RIM) continue;
            int d = rim_deg[i];
            if      (d == 0) d0++;
            else if (d == 1) d1++;
            else if (d == 2) d2++;
            else if (d == 3) d3++;
            else             d4p++;
        }
        (*log) << "[RimChain] nV=" << nV
               << " n_rim=" << n_rim
               << " chain=" << chain_out.size()
               << " seed=" << seed
               << " seed_rimdeg=" << rim_deg[seed]
               << " closed=" << (closed_out ? "Y" : "N")
               << "  rimdeg{0,1,2,3,>=4}=" << d0 << "/" << d1
               << "/" << d2 << "/" << d3 << "/" << d4p
               << "  unchained=" << (n_rim - (int)chain_out.size())
               << std::endl;
    }

    return true;
}


// ---------------------------------------------------------------------
// sortRimChainByAngle  — adopted approach for Step 1+
//
// Order RIM vertices by their atan2 angle around the centroid, in the
// PCA-derived major plane of the patch.
//
// Why this is preferred over walkRimChain (above):
//   The "rim" produced by `g_liverRegion ∩ quadrant ∩ caudal ∩ arvis`
//   is a *thick patch* (rim-degree >= 4 dominates) on the liver's
//   anterior shell — not a 1D curve. The graph walk in walkRimChain
//   stops at the first rim-graph junction and returns ~20% of the
//   verts. But after projecting onto the patch's major plane (PCA's
//   2 largest eigenvectors), the patch is approximately annular: the
//   smallest eigenvector is normal to the patch, and the angle
//   atan2(v, u) around the centroid increases monotonically (modulo
//   wrap) along the patch's circumferential direction.
//
//   Sorting by this angle gives:
//     - **All** RIM verts in the chain (no junction loss)
//     - Spatial coherence: adjacent indices = spatially nearby verts
//       → enables `(pos[i+1] - pos[i-1])` tangents in Step 3
//     - Deterministic ordering (PCA is rotation-stable up to evec
//       sign; we don't enforce a specific handedness, only consistency)
//     - Linear cost: O(n_rim · log n_rim) for the sort, ~0.5 ms for
//       254 verts
//
// Failure modes:
//   - Genuine non-planar rim (e.g. wrap-around with fold) → planarity
//     ratio (eval[0]/eval[1]) approaches 1; sort becomes unstable.
//     Logged as "POOR planar fit" for visual diagnosis.
//   - Empty/degenerate input → returns false with a log message.
//
// Outputs
//   chain_out               : RIM vertex indices, ordered by angle
//   n_rim_total_out         : count of RIM vertices in `labels`
//   centroid_out            : the centroid used as the sort origin
//   major_normal_out        : the patch normal (smallest PCA evec)
//   planarity_ratio_out     : eval[0] / eval[1]; <0.1 = good, <0.3 OK
// ---------------------------------------------------------------------
inline bool sortRimChainByAngle(const mCutMesh& mesh,
                                const std::vector<uint8_t>& labels,
                                std::vector<int>& chain_out,
                                int& n_rim_total_out,
                                glm::vec3& centroid_out,
                                glm::vec3& major_normal_out,
                                glm::vec3& principal_axis_out,
                                double& planarity_ratio_out,
                                std::ostream* log = nullptr)
{
    chain_out.clear();
    n_rim_total_out = 0;
    centroid_out = glm::vec3(0.0f);
    major_normal_out = glm::vec3(0.0f, 0.0f, 1.0f);
    principal_axis_out = glm::vec3(1.0f, 0.0f, 0.0f);
    planarity_ratio_out = 1.0;

    const int nV = (int)(mesh.mVertices.size() / 3);
    if ((int)labels.size() != nV) {
        if (log) (*log) << "[RimSort] ERROR labels.size=" << labels.size()
                   << " != nV=" << nV << std::endl;
        return false;
    }

    // 1. Collect RIM vertex indices and positions
    std::vector<int>       rim_idx;
    std::vector<glm::vec3> rim_pos;
    rim_idx.reserve(nV / 10);
    rim_pos.reserve(nV / 10);
    for (int i = 0; i < nV; i++) {
        if (labels[i] != LiverRegionLabel::RIM) continue;
        rim_idx.push_back(i);
        rim_pos.emplace_back(mesh.mVertices[i*3],
                             mesh.mVertices[i*3+1],
                             mesh.mVertices[i*3+2]);
    }
    n_rim_total_out = (int)rim_idx.size();
    if (rim_idx.size() < 3) {
        if (log) (*log) << "[RimSort] too few RIM vertices ("
                   << rim_idx.size() << ")" << std::endl;
        return false;
    }

    // 2. Centroid (double precision throughout to avoid Eigen drift)
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : rim_pos) {
        mean += Eigen::Vector3d(p.x, p.y, p.z);
    }
    mean /= double(rim_pos.size());
    const glm::vec3 c((float)mean.x(), (float)mean.y(), (float)mean.z());
    centroid_out = c;

    // 3. Covariance matrix → symmetric 3x3 eigendecomposition
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : rim_pos) {
        Eigen::Vector3d d(double(p.x) - mean.x(),
                          double(p.y) - mean.y(),
                          double(p.z) - mean.z());
        cov += d * d.transpose();
    }
    cov /= double(std::max<size_t>(rim_pos.size() - 1, 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success) {
        if (log) (*log) << "[RimSort] ERROR PCA eigendecomposition failed"
                   << std::endl;
        return false;
    }
    const Eigen::Vector3d evals = eig.eigenvalues();        // ascending
    const Eigen::Matrix3d evecs = eig.eigenvectors();
    //   col 0 (smallest evalue) = patch normal
    //   col 1 (mid evalue)      = in-plane v axis
    //   col 2 (largest evalue)  = in-plane u axis (primary spread)
    const glm::vec3 n_axis((float)evecs(0,0), (float)evecs(1,0), (float)evecs(2,0));
    const glm::vec3 v_axis((float)evecs(0,1), (float)evecs(1,1), (float)evecs(2,1));
    const glm::vec3 u_axis((float)evecs(0,2), (float)evecs(1,2), (float)evecs(2,2));
    major_normal_out = n_axis;
    principal_axis_out = u_axis;

    // Planarity: smallest / mid. <0.1 good (patch flat-ish), >0.3 poor
    // (patch is more like a blob — angle sort unstable).
    if (evals(1) > 1e-12) {
        planarity_ratio_out = evals(0) / evals(1);
    }

    // 4. Project each rim vertex onto (u, v) plane, compute angle
    std::vector<std::pair<float, int>> angle_idx;
    angle_idx.reserve(rim_pos.size());
    for (size_t k = 0; k < rim_pos.size(); k++) {
        const glm::vec3 d = rim_pos[k] - c;
        const float u = glm::dot(d, u_axis);
        const float v = glm::dot(d, v_axis);
        const float ang = std::atan2(v, u);     // [-pi, pi]
        angle_idx.emplace_back(ang, rim_idx[k]);
    }

    // 5. Sort ascending by angle (consistent rotation around n_axis,
    // direction unspecified — any consistent ordering works for chamfer
    // matching + (pos[i+1]-pos[i-1]) tangent. If a specific handedness
    // is needed later, multiply atan2 by sign(n_axis · world_up) etc.)
    std::sort(angle_idx.begin(), angle_idx.end(),
              [](const std::pair<float,int>& a,
                 const std::pair<float,int>& b){ return a.first < b.first; });

    chain_out.reserve(angle_idx.size());
    for (const auto& ai : angle_idx) chain_out.push_back(ai.second);

    if (log) {
        const char* fit_tag = (planarity_ratio_out < 0.10) ? "[GOOD planar fit]"
                              : (planarity_ratio_out < 0.30) ? "[OK planar fit]"
                                                             :                                "[POOR planar fit]";
        (*log) << "[RimSort] nV=" << nV
               << " n_rim=" << rim_pos.size()
               << " chain=" << chain_out.size()
               << " centroid=(" << c.x << "," << c.y << "," << c.z << ")"
               << "  evals=(" << evals(0) << "," << evals(1) << "," << evals(2) << ")"
               << "  planarity=" << planarity_ratio_out
               << " " << fit_tag
               << std::endl;
    }

    return true;
}


// =====================================================================
// Phase 7b Step 3 — Shape Match COARSE search helpers
// =====================================================================
//
// All functions below operate on plain 3D point clouds (no mesh/labels
// dependency), so they're reusable for both source-derived RIM points
// and target-derived boundary points alike.
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
// sortPointsByPCAAngle  — generic version of sortRimChainByAngle
//
// Same algorithm (PCA-major-plane atan2 ordering) but takes a plain
// std::vector<glm::vec3> instead of (mesh + labels). Used for target
// boundary points which have no mesh-side label structure.
//
// chain_out is the *ordered* sequence of input point indices (sorted by
// angle around the patch centroid in the PCA major plane). Caller can
// retrieve the actual 3D positions by indexing back into `pts`.
// ---------------------------------------------------------------------
inline bool sortPointsByPCAAngle(const std::vector<glm::vec3>& pts,
                                 std::vector<glm::vec3>& ordered_pts_out,
                                 glm::vec3& centroid_out,
                                 glm::vec3& major_normal_out,
                                 glm::vec3& principal_axis_out,
                                 double& planarity_ratio_out,
                                 std::ostream* log = nullptr)
{
    ordered_pts_out.clear();
    centroid_out = glm::vec3(0.0f);
    major_normal_out = glm::vec3(0.0f, 0.0f, 1.0f);
    principal_axis_out = glm::vec3(1.0f, 0.0f, 0.0f);
    planarity_ratio_out = 1.0;

    if (pts.size() < 3) {
        if (log) (*log) << "[PtsSort] too few points (" << pts.size()
                   << ")" << std::endl;
        return false;
    }

    // 1. Centroid
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : pts) {
        mean += Eigen::Vector3d(p.x, p.y, p.z);
    }
    mean /= double(pts.size());
    const glm::vec3 c((float)mean.x(), (float)mean.y(), (float)mean.z());
    centroid_out = c;

    // 2. Covariance + PCA
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : pts) {
        Eigen::Vector3d d(double(p.x) - mean.x(),
                          double(p.y) - mean.y(),
                          double(p.z) - mean.z());
        cov += d * d.transpose();
    }
    cov /= double(std::max<size_t>(pts.size() - 1, 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success) {
        if (log) (*log) << "[PtsSort] ERROR PCA failed" << std::endl;
        return false;
    }
    const Eigen::Vector3d evals = eig.eigenvalues();
    const Eigen::Matrix3d evecs = eig.eigenvectors();
    const glm::vec3 n_axis((float)evecs(0,0), (float)evecs(1,0), (float)evecs(2,0));
    const glm::vec3 v_axis((float)evecs(0,1), (float)evecs(1,1), (float)evecs(2,1));
    const glm::vec3 u_axis((float)evecs(0,2), (float)evecs(1,2), (float)evecs(2,2));
    major_normal_out = n_axis;
    principal_axis_out = u_axis;
    if (evals(1) > 1e-12) {
        planarity_ratio_out = evals(0) / evals(1);
    }

    // 3. Project + angle + sort
    std::vector<std::pair<float, int>> angle_idx;
    angle_idx.reserve(pts.size());
    for (size_t k = 0; k < pts.size(); k++) {
        const glm::vec3 d = pts[k] - c;
        const float u = glm::dot(d, u_axis);
        const float v = glm::dot(d, v_axis);
        angle_idx.emplace_back(std::atan2(v, u), (int)k);
    }
    std::sort(angle_idx.begin(), angle_idx.end(),
              [](const std::pair<float,int>& a,
                 const std::pair<float,int>& b){ return a.first < b.first; });

    ordered_pts_out.reserve(pts.size());
    for (const auto& ai : angle_idx) ordered_pts_out.push_back(pts[ai.second]);

    if (log) {
        const char* fit_tag = (planarity_ratio_out < 0.10) ? "[GOOD]"
                              : (planarity_ratio_out < 0.30) ? "[OK]"
                                                             :                                "[POOR]";
        (*log) << "[PtsSort] n=" << pts.size()
               << " centroid=(" << c.x << "," << c.y << "," << c.z << ")"
               << " evals=(" << evals(0) << "," << evals(1) << "," << evals(2) << ")"
               << " planarity=" << planarity_ratio_out << " " << fit_tag
               << std::endl;
    }
    return true;
}


// ---------------------------------------------------------------------
// solveTwoAxisAlignment
//   Build a 3x3 rotation R such that R aligns (s_tangent, s_normal) to
//   (t_tangent, t_normal). Inputs need not be exactly orthogonal — the
//   function Gram-Schmidt-orthogonalises each pair (projecting normal
//   onto the plane perpendicular to tangent), then constructs an
//   orthonormal triad [tangent | bitangent | normal] on each side and
//   solves R = T_target · T_source⁻¹ = T_target · T_sourceᵀ (since the
//   triad is orthonormal).
//
//   Numerical safety:
//     - Zero-length inputs default to (1,0,0).
//     - If s_normal is exactly parallel to s_tangent (degenerate),
//       Gram-Schmidt produces zero; we fall back to an arbitrary
//       perpendicular axis.
// ---------------------------------------------------------------------
inline glm::mat3 solveTwoAxisAlignment(glm::vec3 s_tangent,
                                       glm::vec3 s_normal,
                                       glm::vec3 t_tangent,
                                       glm::vec3 t_normal)
{
    auto normSafe = [](glm::vec3 v) {
        float n = glm::length(v);
        return (n > 1e-9f) ? (v / n) : glm::vec3(1.0f, 0.0f, 0.0f);
    };
    auto orthoNorm = [&](glm::vec3 to_ortho, glm::vec3 ref_unit) {
        glm::vec3 r = to_ortho - glm::dot(to_ortho, ref_unit) * ref_unit;
        if (glm::length(r) < 1e-9f) {
            // degenerate (parallel) — pick any axis perpendicular to ref
            glm::vec3 a = (std::abs(ref_unit.x) < 0.9f)
                              ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
            r = a - glm::dot(a, ref_unit) * ref_unit;
        }
        return normSafe(r);
    };
    s_tangent = normSafe(s_tangent);
    t_tangent = normSafe(t_tangent);
    s_normal  = orthoNorm(s_normal, s_tangent);
    t_normal  = orthoNorm(t_normal, t_tangent);
    const glm::vec3 s_bitan = glm::cross(s_normal, s_tangent);
    const glm::vec3 t_bitan = glm::cross(t_normal, t_tangent);
    // mat3(col0, col1, col2) — columns are the axes
    const glm::mat3 S(s_tangent, s_bitan, s_normal);
    const glm::mat3 T(t_tangent, t_bitan, t_normal);
    // R · S = T  →  R = T · Sᵀ (S orthonormal)
    return T * glm::transpose(S);
}


// ---------------------------------------------------------------------
// chamferBruteForce
//   One-sided mean nearest-neighbour distance from src_pts to tgt_pts.
//   O(|src| · |tgt|). For Phase 7b's 254 × 30k = ~7.7 M ops per
//   evaluation, ~50 ms each on a single thread — acceptable for a 30-
//   evaluation coarse pass at Ctrl+W press. Step 4 will replace with
//   KDTree if needed.
//
//   src_pts must be pre-transformed to target coordinates (i.e. apply
//   T to source verts *before* calling). Returns -1 if either input
//   is empty.
// ---------------------------------------------------------------------
inline double chamferBruteForce(const std::vector<glm::vec3>& src_pts,
                                const std::vector<glm::vec3>& tgt_pts)
{
    if (src_pts.empty() || tgt_pts.empty()) return -1.0;
    double total = 0.0;
    const int Ns = (int)src_pts.size();
    const int Nt = (int)tgt_pts.size();
    for (int i = 0; i < Ns; i++) {
        const glm::vec3& sp = src_pts[i];
        float best_d2 = std::numeric_limits<float>::max();
        for (int j = 0; j < Nt; j++) {
            const glm::vec3& tp = tgt_pts[j];
            const float dx = sp.x - tp.x;
            const float dy = sp.y - tp.y;
            const float dz = sp.z - tp.z;
            const float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d2) best_d2 = d2;
        }
        total += std::sqrt(double(best_d2));
    }
    return total / double(Ns);
}

// ---------------------------------------------------------------------
// chamferSymmetric
//   Two-sided (= bidirectional, "symmetric") chamfer distance:
//     = (mean nearest-A→B distance + mean nearest-B→A distance) / 2
//
//   Why this differs from one-sided for Phase 7b Step 4b axis sweep:
//     One-sided (A→B) only checks that A finds something nearby in B.
//     When A = source full vertices and B = target, this asks "does every
//     source vertex have a target near it?" — but internal source verts
//     find SOME target point nearby at almost any rotation, washing out
//     the 1D rotation signal (~67/4028 rim verts are 1.7% of A, so the
//     1D-sensitive part contributes only ~1.7% to the cost).
//
//     Two-sided also asks "does every target point have a source near
//     it?" — flipped-source poses have huge gaps where target points
//     find no nearby source vertex. This is what gives the symmetric
//     chamfer its sign-discriminating power.
//
//   Computational cost: 2× one-sided (~1.5-2× wall-clock with cache
//   effects). For Step 4b 36 × 4028 × 5000 ≈ 725 M ops one-sided, the
//   symmetric variant is ~1.4 G ops ~ 2-3 s.
// ---------------------------------------------------------------------
inline double chamferSymmetric(const std::vector<glm::vec3>& A,
                               const std::vector<glm::vec3>& B)
{
    if (A.empty() || B.empty()) return -1.0;
    const int Na = (int)A.size();
    const int Nb = (int)B.size();

    // A → B
    double sum_ab = 0.0;
    for (int i = 0; i < Na; i++) {
        const glm::vec3& a = A[i];
        float best_d2 = std::numeric_limits<float>::max();
        for (int j = 0; j < Nb; j++) {
            const float dx = a.x - B[j].x;
            const float dy = a.y - B[j].y;
            const float dz = a.z - B[j].z;
            const float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d2) best_d2 = d2;
        }
        sum_ab += std::sqrt(double(best_d2));
    }

    // B → A
    double sum_ba = 0.0;
    for (int j = 0; j < Nb; j++) {
        const glm::vec3& b = B[j];
        float best_d2 = std::numeric_limits<float>::max();
        for (int i = 0; i < Na; i++) {
            const float dx = b.x - A[i].x;
            const float dy = b.y - A[i].y;
            const float dz = b.z - A[i].z;
            const float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d2) best_d2 = d2;
        }
        sum_ba += std::sqrt(double(best_d2));
    }

    return (sum_ab / double(Na) + sum_ba / double(Nb)) * 0.5;
}


// ---------------------------------------------------------------------
// CoarseCandidate + runShapeMatchCoarse
//
// Coarse search per Phase 7b detailed design §2 Stage 1, simplified to
// our point-cloud + PCA-axes formulation:
//
//   For k = 0..N_coarse-1:
//     idx_k = round(k * M / N_coarse)                  (M = |tgt_chain|)
//     t_anchor      = tgt_chain[idx_k]
//     t_tangent_loc = normalize(tgt_chain[idx+1] - tgt_chain[idx-1])
//     R = solveTwoAxisAlignment(s_axis, s_normal, t_tangent_loc, t_normal)
//     t = t_anchor - R · s_centroid
//     cost = chamferBruteForce(R·s_pts + t, tgt_chain)
//
// Returns the full candidate list; caller sorts and picks top-K.
//
// Note on s_axis vs t_tangent_loc:
//   The source side uses its *global* PCA principal axis (one axis for
//   all 30 anchors). The target side uses a *local* tangent estimated
//   at each anchor. This is the asymmetry intended by the design: the
//   source's "long axis" is well-defined (whole rim ring), while each
//   target anchor effectively says "if my local tangent matches yours,
//   we may be aligned here".
// ---------------------------------------------------------------------
struct CoarseCandidate {
    glm::mat4 transform;        // 4x4 SE(3): R + translation
    double    cost;
    int       target_anchor_k;  // 0..N_coarse-1
    int       target_idx;       // index into tgt_chain
    int       sign_code;        // 0..3: bit0=tangent flipped, bit1=normal flipped
};

inline bool runShapeMatchCoarse(
    const std::vector<glm::vec3>& src_pts_world,   // current pose
    const glm::vec3& src_centroid,
    const glm::vec3& src_principal_axis,
    const glm::vec3& src_major_normal,
    const std::vector<glm::vec3>& tgt_chain_ordered,
    const glm::vec3& tgt_major_normal,
    int N_coarse,
    std::vector<CoarseCandidate>& candidates_out,
    std::ostream* log = nullptr,
    uint8_t sign_mask = 0xF)              // bit0=sign0(t+,n+), bit1=sign1(t-,n+),
    // bit2=sign2(t+,n-), bit3=sign3(t-,n-)
{
    candidates_out.clear();
    if (src_pts_world.empty()) {
        if (log) (*log) << "[CoarseSearch] empty source" << std::endl;
        return false;
    }
    const int M = (int)tgt_chain_ordered.size();
    if (M < 3) {
        if (log) (*log) << "[CoarseSearch] target chain too short ("
                   << M << ")" << std::endl;
        return false;
    }

    // Count enabled signs for capacity allocation
    int n_signs_enabled = 0;
    for (int s = 0; s < 4; s++) if (sign_mask & (1 << s)) n_signs_enabled++;
    if (n_signs_enabled == 0) {
        if (log) (*log) << "[CoarseSearch] sign_mask=0 disables all signs"
                   << std::endl;
        return false;
    }

    candidates_out.reserve(N_coarse * n_signs_enabled);
    std::vector<glm::vec3> src_transformed;
    src_transformed.reserve(src_pts_world.size());

    for (int k = 0; k < N_coarse; k++) {
        const int idx = (k * M) / N_coarse;
        const int im  = (idx - 1 + M) % M;
        const int ip  = (idx + 1) % M;
        const glm::vec3& t_anchor = tgt_chain_ordered[idx];
        const glm::vec3 t_tangent_local_raw = tgt_chain_ordered[ip] - tgt_chain_ordered[im];

        // sign variants (filtered by sign_mask):
        //   sign 0 (t+,n+) = identity-like (small rotation, "natural" pose)
        //   sign 1 (t-,n+) = 180° around target_normal
        //   sign 2 (t+,n-) = 180° around target_tangent
        //   sign 3 (t-,n-) = 180° around target_bitangent
        // Phase 7b "Apply Init Pose を信じる" 運用では sign_mask=0x1 (sign=0 only)
        // で反転候補を完全排除できる。
        for (int sign = 0; sign < 4; sign++) {
            if (!(sign_mask & (1 << sign))) continue;
            const glm::vec3 t_tangent_local =
                (sign & 1) ? -t_tangent_local_raw : t_tangent_local_raw;
            const glm::vec3 t_normal_signed =
                (sign & 2) ? -tgt_major_normal   : tgt_major_normal;

            // 2-axis SE(3) construction
            const glm::mat3 R = solveTwoAxisAlignment(
                src_principal_axis, src_major_normal,
                t_tangent_local,    t_normal_signed);
            const glm::vec3 t_vec = t_anchor - R * src_centroid;

            // Apply T to source for chamfer evaluation
            src_transformed.clear();
            for (const auto& p : src_pts_world) {
                src_transformed.push_back(R * p + t_vec);
            }
            const double cost = chamferBruteForce(src_transformed,
                                                  tgt_chain_ordered);

            // Compose 4x4 (column-major: T = mat4(col0, col1, col2, col3))
            glm::mat4 T4(1.0f);
            T4[0] = glm::vec4(R[0], 0.0f);
            T4[1] = glm::vec4(R[1], 0.0f);
            T4[2] = glm::vec4(R[2], 0.0f);
            T4[3] = glm::vec4(t_vec, 1.0f);

            candidates_out.push_back({T4, cost, k, idx, sign});
        }
    }

    if (log) {
        (*log) << "[CoarseSearch] N=" << N_coarse << "*" << n_signs_enabled
               << "_signs(mask=0x" << std::hex << (int)sign_mask << std::dec << ")"
               << "  src=" << src_pts_world.size()
               << "  tgt=" << M
               << "  candidates=" << candidates_out.size()
               << std::endl;
    }
    return true;
}

// =====================================================================
//  Phase 7b Step 4b — Rim Axis Rotation Sweep
//
//  Shape Match で得た baseline_T は rim 同士を 5 DOF で合わせる。
//  残る 1 DOF (= rim patch normal 軸まわりの回転) は rim-only chamfer
//  では決まらない。これを「全頂点 chamfer」で詰めるのがこの関数。
//
//  なぜ反転 (sign=1/2/3) が排除されるか:
//    - rim だけなら反転姿勢でも cost 低い (rim ぴたり合う)
//    - 全頂点で見ると、source 内部が target の裏側に行くため大幅高 cost
//    - rim 軸まわり 360° を全頂点で評価すれば、正しい姿勢が必然的に勝つ
//
//  なぜ「rim 絶対保持」になるか:
//    - rim 軸まわりの回転 = rim 上の点を動かさない (axis が rim plane の
//      法線で、rim 上の点は axis に直交する円上を動く...
//      実際には pivot=rim_centroid なので rim 中心は不動、rim 上の各点
//      は微小に動くが、rim 平面内なので target rim から大きく離れない)
//    - 厳密には「rim chain の chain order を保ったまま rotate」
//    - rim 帯の幅 (~2-3 vertex) 内での微移動のみ
//
//  計算量:
//    N_angles × |src_full| × |tgt_subsample|
//    36 × 4028 × 5000 = 725 M ops → ~1-2s on modern CPU
//    target subsample は uniform stride で OK (random shuffle 不要)
// =====================================================================

struct AxisSweepResult {
    glm::mat4 axis_rotation_T;  // pure rotation T about pivot (rim_centroid)
    double    cost;
    int       angle_idx;
    float     angle_deg;
};

inline bool runRimAxisRotationSweep(
    const std::vector<glm::vec3>& src_full_world_after_T,  // src 全頂点 (baseline_T 適用後 world)
    const std::vector<glm::vec3>& tgt_subsample,            // target 部分集合
    const glm::vec3& rim_centroid_world_after_T,            // 回転中心 (T 適用後)
    const glm::vec3& rim_axis_world_after_T,                // 回転軸 (T 適用後)
    int N_angles,
    std::vector<AxisSweepResult>& results_out,
    std::ostream* log = nullptr)
{
    results_out.clear();
    if (src_full_world_after_T.empty() || tgt_subsample.empty()) {
        if (log) (*log) << "[AxisSweep] empty input" << std::endl;
        return false;
    }
    if (N_angles < 4) {
        if (log) (*log) << "[AxisSweep] N_angles too small ("
                   << N_angles << ")" << std::endl;
        return false;
    }
    const float axis_len = glm::length(rim_axis_world_after_T);
    if (axis_len < 1e-6f) {
        if (log) (*log) << "[AxisSweep] rim axis is zero vector" << std::endl;
        return false;
    }
    const glm::vec3 axis = rim_axis_world_after_T / axis_len;

    const glm::mat4 to_pivot   = glm::translate(glm::mat4(1.0f),
                                              -rim_centroid_world_after_T);
    const glm::mat4 from_pivot = glm::translate(glm::mat4(1.0f),
                                                rim_centroid_world_after_T);

    results_out.reserve(N_angles);
    std::vector<glm::vec3> src_rotated;
    src_rotated.reserve(src_full_world_after_T.size());

    for (int i = 0; i < N_angles; i++) {
        const float angle_rad = (float(i) * 2.0f * 3.14159265358979f)
        / float(N_angles);
        // Rotation about pivot (rim centroid) around the rim axis:
        //   T_rot = Translate(+pivot) * Rotate(axis, angle) * Translate(-pivot)
        const glm::mat4 R_axis = glm::rotate(glm::mat4(1.0f), angle_rad, axis);
        const glm::mat4 T_rot  = from_pivot * R_axis * to_pivot;

        // Apply T_rot to all source vertices
        src_rotated.clear();
        for (const auto& p : src_full_world_after_T) {
            const glm::vec4 v = T_rot * glm::vec4(p, 1.0f);
            src_rotated.emplace_back(v.x, v.y, v.z);
        }

        // Chamfer: SYMMETRIC (two-sided) — see chamferSymmetric for rationale.
        //   One-sided was found to be flat (~1.4× ratio between best/worst
        //   angles) because source internal verts find some nearby target
        //   at any rotation. Symmetric chamfer asks the converse "does
        //   target also find nearby source?" which flipped poses cannot
        //   satisfy, giving a much stronger 1D rotation signal.
        const double cost = chamferSymmetric(src_rotated, tgt_subsample);

        AxisSweepResult r;
        r.axis_rotation_T = T_rot;
        r.cost            = cost;
        r.angle_idx       = i;
        r.angle_deg       = angle_rad * 180.0f / 3.14159265358979f;
        results_out.push_back(r);
    }

    // Sort by cost (best first)
    std::sort(results_out.begin(), results_out.end(),
              [](const AxisSweepResult& a, const AxisSweepResult& b) {
                  return a.cost < b.cost;
              });

    if (log) {
        (*log) << "[AxisSweep] N=" << N_angles
               << "  src=" << src_full_world_after_T.size()
               << "  tgt=" << tgt_subsample.size()
               << "  best_angle=" << results_out[0].angle_deg << "deg"
               << "  best_cost=" << results_out[0].cost
               << "  worst_cost=" << results_out.back().cost
               << "  ratio=" << (results_out.back().cost / std::max(1e-9, results_out[0].cost))
               << std::endl;
        // log top-3 and bottom-3 for diagnostic
        (*log) << "[AxisSweep] top-3:" << std::endl;
        for (int j = 0; j < std::min(3, (int)results_out.size()); j++) {
            (*log) << "    angle=" << results_out[j].angle_deg
                   << "deg  cost=" << results_out[j].cost
                   << std::endl;
        }
        (*log) << "[AxisSweep] worst:" << std::endl;
        for (int j = std::max(0, (int)results_out.size() - 1);
             j < (int)results_out.size(); j++) {
            (*log) << "    angle=" << results_out[j].angle_deg
                   << "deg  cost=" << results_out[j].cost
                   << std::endl;
        }
    }
    return true;
}

// =====================================================================
//  Phase 7b Step 3a — Full-2D matching (depth-free)
// =====================================================================
//
//  Motivation
//    The original Step 3 used 3D chamfer between source rim chain and
//    target boundary points that were *lifted* from screenMesh depth.
//    Depth estimates from depth-anything-v2 carry several-cm noise in
//    typical liver scenes, which contaminated the target 3D positions
//    and the cost ranking. The "ground truth" the user actually has is
//    the 2D SAM2 segmentation boundary; lifting it to 3D throws away
//    information.
//
//  Full-2D pipeline (Ctrl+W when g_shapeMatchUse2DCost == true)
//    - Source: stays 3D (CT mesh + walkRimChain order, depth-reliable)
//    - Target: 2D contour traced from g_boundaryDistMap, arc-length
//      resampled to N anchors in pixel space (no lift)
//    - Per anchor:
//        back-project pixel to 3D at source-centroid depth → 3D anchor
//        finite-diff neighbour anchors → 3D tangent at that depth
//        solveTwoAxisAlignment(src_axis_3D, src_normal_3D,
//                              t_tangent_3D, cam_axis_world)
//        T = R, translation = anchor_3D - R · src_centroid_3D
//    - Cost: forward-project transformed source 3D rim to 2D pixels and
//      look up g_boundaryDistMap. O(|src|) per evaluation.
//
//  Why "cam_axis_world" as target normal
//    The source PCA normal points roughly along the rim patch's normal.
//    For surgical liver views the rim normal is approximately parallel
//    to the camera viewing direction (we look at the liver "edge on").
//    Using cam_axis as the target normal forces R to bring src_normal
//    in line with the camera axis — i.e. makes the rim face the camera.
//    sign bit 1 (n flip) handles the case where src_normal is the
//    opposite PCA sign (PCA eigenvectors have ambiguous orientation).
//
//  Why "max_dist_cap" matters
//    g_boundaryDistMap stores BFS distance to boundary INSIDE the mask
//    and a sentinel 9999.0f OUTSIDE the mask. For Shape Match cost we
//    cap any sample to max_dist_cap_px to bound the influence of points
//    landing outside the mask (otherwise one outlier dominates the
//    mean). The cap also acts as the "out-of-mask" penalty.
// =====================================================================


// ---------------------------------------------------------------------
// traceContour2D
//   2D Moore-Neighbor-style 8-connectivity walk on boundary pixels.
//
//   A "boundary pixel" is bdy.data[i] < bdy_px_thresh (typically 1.5f =
//   the inner 1-px ring of the mask boundary as BFS-distance < 1.5).
//   Pixels where the optional instrument distance map is below
//   inst_px_thresh are excluded (instruments break a single rim into
//   multiple visible arcs; we want separate segments).
//
//   Algorithm
//     1. Build is_boundary mask
//     2. For each unvisited boundary pixel, start a greedy walk:
//          - At each step, pick the first unvisited 8-neighbour that is
//            also boundary; if none, end this segment
//     3. Filter segments with fewer than 10 pixels
//     4. Sort segments by size descending (largest first)
//
//   Returns true iff at least one segment remained after filtering.
//   Segments are *ordered* polylines (chain[i+1] is adjacent to chain[i]).
// ---------------------------------------------------------------------
inline bool traceContour2D(const BoundaryDistMap& bdy,
                           float bdy_px_thresh,
                           const BoundaryDistMap* inst_or_null,
                           float inst_px_thresh,
                           std::vector<std::vector<glm::vec2>>& segments_out,
                           std::ostream* log = nullptr)
{
    segments_out.clear();
    if (!bdy.valid || bdy.width <= 0 || bdy.height <= 0) {
        if (log) (*log) << "[TraceContour2D] boundary map invalid"
                   << std::endl;
        return false;
    }
    const int W = bdy.width, H = bdy.height;

    // Build is_boundary mask (exclude instrument-occluded boundary if
    // instrument map is provided AND its size matches)
    std::vector<uint8_t> is_b(W * H, 0);
    int n_boundary = 0, n_inst_excluded = 0;
    const bool use_inst = (inst_or_null != nullptr) && inst_or_null->valid
                          && inst_or_null->width  == W
                          && inst_or_null->height == H;
    for (int i = 0; i < W * H; i++) {
        const float bd = bdy.data[i];
        if (bd >= bdy_px_thresh) continue;
        if (bd >= 9000.0f)        continue;            // outside-mask sentinel
        if (use_inst && inst_or_null->data[i] < inst_px_thresh) {
            n_inst_excluded++;
            continue;
        }
        is_b[i] = 1;
        n_boundary++;
    }

    if (n_boundary == 0) {
        if (log) (*log) << "[TraceContour2D] no boundary pixels (bdy_th="
                   << bdy_px_thresh << ", inst_th=" << inst_px_thresh
                   << ", inst_excluded=" << n_inst_excluded << ")"
                   << std::endl;
        return false;
    }

    // 8-connectivity: order chosen so 4-conn neighbours come first for
    // cleaner chains where possible.
    const int dx8[] = { 1, -1,  0,  0,  1,  1, -1, -1};
    const int dy8[] = { 0,  0,  1, -1,  1, -1,  1, -1};
    std::vector<uint8_t> visited(W * H, 0);

    for (int sy = 0; sy < H; sy++) {
        for (int sx = 0; sx < W; sx++) {
            if (!is_b[sy * W + sx] || visited[sy * W + sx]) continue;
            std::vector<glm::vec2> seg;
            int cx = sx, cy = sy;
            visited[cy * W + cx] = 1;
            seg.emplace_back(float(cx), float(cy));
            while (true) {
                int next_x = -1, next_y = -1;
                for (int d = 0; d < 8; d++) {
                    int nx = cx + dx8[d], ny = cy + dy8[d];
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    if (!is_b[ny * W + nx] || visited[ny * W + nx]) continue;
                    next_x = nx; next_y = ny;
                    break;
                }
                if (next_x < 0) break;
                cx = next_x; cy = next_y;
                visited[cy * W + cx] = 1;
                seg.emplace_back(float(cx), float(cy));
            }
            if ((int)seg.size() >= 10) {
                segments_out.push_back(std::move(seg));
            }
        }
    }

    std::sort(segments_out.begin(), segments_out.end(),
              [](const std::vector<glm::vec2>& a,
                 const std::vector<glm::vec2>& b) {
                  return a.size() > b.size();
              });

    if (log) {
        int total = 0;
        for (auto& s : segments_out) total += (int)s.size();
        (*log) << "[TraceContour2D] " << W << "x" << H
               << "  boundary_pixels=" << n_boundary
               << "  inst_excluded=" << n_inst_excluded
               << "  segments=" << segments_out.size()
               << "  total_traced=" << total
               << "  largest=" << (segments_out.empty()
                                       ? 0 : (int)segments_out[0].size())
               << std::endl;
    }
    return !segments_out.empty();
}


// ---------------------------------------------------------------------
// resampleArcLength2D
//   Resample a 2D polyline to N evenly-spaced points along arc length.
//   Mid-bucket parametrisation (t_k = (k+0.5) * L / N) gives endpoint-
//   symmetric distribution.
// ---------------------------------------------------------------------
inline void resampleArcLength2D(const std::vector<glm::vec2>& contour,
                                int N,
                                std::vector<glm::vec2>& resampled_out)
{
    resampled_out.clear();
    if (contour.size() < 2 || N < 1) return;

    std::vector<float> cum(contour.size(), 0.0f);
    for (size_t i = 1; i < contour.size(); i++) {
        cum[i] = cum[i-1] + glm::length(contour[i] - contour[i-1]);
    }
    const float total_len = cum.back();
    if (total_len < 1e-6f) return;

    resampled_out.reserve(N);
    for (int k = 0; k < N; k++) {
        const float t = (float(k) + 0.5f) * total_len / float(N);
        auto it = std::upper_bound(cum.begin(), cum.end(), t);
        if (it == cum.begin()) {
            resampled_out.push_back(contour.front());
            continue;
        }
        if (it == cum.end()) {
            resampled_out.push_back(contour.back());
            continue;
        }
        const size_t idx = std::distance(cum.begin(), it);
        const float  seg_len = cum[idx] - cum[idx-1];
        const float  alpha = (seg_len > 1e-6f)
                                 ? (t - cum[idx-1]) / seg_len
                                 : 0.0f;
        resampled_out.push_back(
            contour[idx-1] * (1.0f - alpha) + contour[idx] * alpha);
    }
}


// ---------------------------------------------------------------------
// unprojectPixelAtWorldDepth
//   Back-project pixel (px, py) along the view ray to the world plane
//   that contains depth_ref_world.
//
//   Implementation: forward-project depth_ref_world to find its NDC.z,
//   then back-project (u_ndc, v_ndc, ndc_z, 1) through inv(proj*view).
//   This works for any view/proj combination, not just our AR fixed
//   camera (matches the inv-VP pattern in CmaesUtils::extractSourceContour).
// ---------------------------------------------------------------------
inline glm::vec3 unprojectPixelAtWorldDepth(
    const glm::vec2& pixel,
    const glm::vec3& depth_ref_world,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H)
{
    const glm::mat4 VP = proj * view;
    const glm::vec4 ref_clip = VP * glm::vec4(depth_ref_world, 1.0f);
    if (std::abs(ref_clip.w) < 1e-9f) return depth_ref_world;
    const float ndc_z = ref_clip.z / ref_clip.w;

    // Pixel → NDC (image origin top-left, Y growing down → NDC y flip)
    const float u_ndc =  2.0f * pixel.x / float(W) - 1.0f;
    const float v_ndc =  1.0f - 2.0f * pixel.y / float(H);

    const glm::mat4 invVP = glm::inverse(VP);
    const glm::vec4 world_h =
        invVP * glm::vec4(u_ndc, v_ndc, ndc_z, 1.0f);
    if (std::abs(world_h.w) < 1e-9f) return depth_ref_world;
    return glm::vec3(world_h) / world_h.w;
}


// ---------------------------------------------------------------------
// project2DBoundaryDistance
//   Forward-project src points to 2D AR camera, mean(boundary distance)
//   via g_boundaryDistMap lookup. Returns -1 if invalid input.
//
//   Per-point cost rule:
//     - clip.w <= 0  → behind camera   → out_of_frame_dist_px
//     - NDC outside [-1,+1]            → out_of_frame_dist_px
//     - inside frame, bd >= 9000       → max_dist_cap_px   (outside mask)
//     - inside frame, bd <  9000       → min(bd, max_dist_cap_px)
//
//   The cap bounds the influence of "wildly off" candidates so the
//   mean stays interpretable. Without the cap, a single point landing
//   on a 9999 sentinel pixel would drown out the other 250+ in the rim.
// ---------------------------------------------------------------------
inline double project2DBoundaryDistance(
    const std::vector<glm::vec3>& src_pts_world,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    const BoundaryDistMap& bdy,
    float out_of_frame_dist_px,
    float max_dist_cap_px,
    int* n_in_frame_out = nullptr,
    int* n_in_mask_out  = nullptr)
{
    if (src_pts_world.empty() || !bdy.valid) {
        if (n_in_frame_out) *n_in_frame_out = 0;
        if (n_in_mask_out)  *n_in_mask_out  = 0;
        return -1.0;
    }
    const glm::mat4 VP = proj * view;
    double sum = 0.0;
    int n_in_frame = 0;
    int n_in_mask  = 0;
    const int Nv = (int)src_pts_world.size();
    const float cap = max_dist_cap_px;

    for (int i = 0; i < Nv; i++) {
        const glm::vec4 clip = VP * glm::vec4(src_pts_world[i], 1.0f);
        if (clip.w <= 1e-9f) {
            sum += out_of_frame_dist_px;
            continue;
        }
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        if (ndcx < -1.0f || ndcx > 1.0f ||
            ndcy < -1.0f || ndcy > 1.0f) {
            sum += out_of_frame_dist_px;
            continue;
        }
        n_in_frame++;
        const float u = (ndcx + 1.0f) * 0.5f;
        const float v = (1.0f - ndcy) * 0.5f;
        int px = int(u * float(bdy.width));
        int py = int(v * float(bdy.height));
        if (px < 0) px = 0; else if (px >= bdy.width)  px = bdy.width  - 1;
        if (py < 0) py = 0; else if (py >= bdy.height) py = bdy.height - 1;
        const float bd = bdy.data[py * bdy.width + px];
        if (bd < 9000.0f) {
            n_in_mask++;
            sum += std::min(bd, cap);
        } else {
            sum += cap;
        }
    }
    if (n_in_frame_out) *n_in_frame_out = n_in_frame;
    if (n_in_mask_out)  *n_in_mask_out  = n_in_mask;
    return sum / double(Nv);
}


// ---------------------------------------------------------------------
// runShapeMatchCoarse2D
//   Full-2D coarse search per Phase 7b Step 3a design.
//
//   Inputs (3D, from source side):
//     src_pts_world           : source rim points in world space (current pose)
//     src_centroid_3D         : source rim centroid (PCA cache)
//     src_principal_axis_3D   : source rim principal axis (PCA cache)
//     src_major_normal_3D     : source rim patch normal (PCA cache)
//
//   Inputs (2D, from target side):
//     tgt_anchors_2D          : N evenly-spaced pixels along contour
//
//   Per anchor k (and per enabled sign):
//     a_3D       = unprojectPixelAtWorldDepth(tgt_anchors[k], src_centroid)
//     am_3D, ap_3D = same for neighbours k-1, k+1   (chain order)
//     t_tangent_3D = ap_3D - am_3D                  (in image plane)
//     t_normal_3D  = camera_axis_world              (image plane normal)
//     R = solveTwoAxisAlignment(s_axis_3D, s_normal_3D,
//                                  t_tangent_3D, t_normal_3D)
//     T_translation = a_3D - R · src_centroid_3D
//     cost = project2DBoundaryDistance(R·src_pts_world + T_translation, ...)
//
//   Candidates with in-frame fraction below min_in_frame_rate get a
//   large sortable rejection cost (1e9 + in_rate).
// ---------------------------------------------------------------------
inline bool runShapeMatchCoarse2D(
    const std::vector<glm::vec3>& src_pts_world,
    const glm::vec3& src_centroid_3D,
    const glm::vec3& src_principal_axis_3D,
    const glm::vec3& src_major_normal_3D,
    const std::vector<glm::vec2>& tgt_anchors_2D,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    const BoundaryDistMap& bdy,
    float out_of_frame_dist_px,
    float max_dist_cap_px,
    float min_in_frame_rate,
    std::vector<CoarseCandidate>& candidates_out,
    std::ostream* log = nullptr,
    uint8_t sign_mask = 0x1,
    float max_rot_deg = 180.0f)   // Step 3b: hard reject rotations above this
{
    candidates_out.clear();
    if (src_pts_world.empty()) {
        if (log) (*log) << "[Coarse2D] empty source" << std::endl;
        return false;
    }
    const int M = (int)tgt_anchors_2D.size();
    if (M < 3) {
        if (log) (*log) << "[Coarse2D] need >=3 anchors, got " << M
                   << std::endl;
        return false;
    }
    if (!bdy.valid) {
        if (log) (*log) << "[Coarse2D] boundary map invalid" << std::endl;
        return false;
    }

    int n_signs = 0;
    for (int s = 0; s < 4; s++) if (sign_mask & (1 << s)) n_signs++;
    if (n_signs == 0) {
        if (log) (*log) << "[Coarse2D] sign_mask=0 disables all signs"
                   << std::endl;
        return false;
    }

    // Camera viewing direction in world ("into the scene"). For our AR
    // fixed camera (lookAt(0, +Z)) this is approximately (0, 0, +1).
    // We use the inverse view to transform the camera-space forward
    // vector (0,0,-1, 0) back to world space — that's the direction
    // the camera looks toward in world coordinates.
    const glm::mat4 view_inv = glm::inverse(view);
    glm::vec3 cam_axis_world =
        glm::vec3(view_inv * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));
    const float cam_axis_len = glm::length(cam_axis_world);
    if (cam_axis_len < 1e-6f) {
        if (log) (*log) << "[Coarse2D] zero camera axis" << std::endl;
        return false;
    }
    cam_axis_world /= cam_axis_len;

    candidates_out.reserve(size_t(M) * size_t(n_signs));
    std::vector<glm::vec3> src_transformed;
    src_transformed.reserve(src_pts_world.size());

    int n_rejected_inframe = 0;
    int n_tried            = 0;
    int n_skipped_tangent  = 0;
    int n_rejected_rot     = 0;     // Step 3b: hard reject above max_rot_deg

    for (int k = 0; k < M; k++) {
        const int im = (k - 1 + M) % M;
        const int ip = (k + 1) % M;
        const glm::vec3 a_3D  = unprojectPixelAtWorldDepth(
            tgt_anchors_2D[k],  src_centroid_3D, view, proj, W, H);
        const glm::vec3 am_3D = unprojectPixelAtWorldDepth(
            tgt_anchors_2D[im], src_centroid_3D, view, proj, W, H);
        const glm::vec3 ap_3D = unprojectPixelAtWorldDepth(
            tgt_anchors_2D[ip], src_centroid_3D, view, proj, W, H);
        const glm::vec3 t_tangent_raw = ap_3D - am_3D;
        if (glm::length(t_tangent_raw) < 1e-9f) {
            n_skipped_tangent++;
            continue;
        }

        for (int sign = 0; sign < 4; sign++) {
            if (!(sign_mask & (1 << sign))) continue;
            const glm::vec3 t_tangent_signed =
                (sign & 1) ? -t_tangent_raw : t_tangent_raw;
            const glm::vec3 t_normal_signed =
                (sign & 2) ? -cam_axis_world : cam_axis_world;

            const glm::mat3 R = solveTwoAxisAlignment(
                src_principal_axis_3D, src_major_normal_3D,
                t_tangent_signed,       t_normal_signed);
            const glm::vec3 t_vec = a_3D - R * src_centroid_3D;

            src_transformed.clear();
            for (const auto& p : src_pts_world) {
                src_transformed.push_back(R * p + t_vec);
            }
            int n_in_frame = 0, n_in_mask = 0;
            const double cost = project2DBoundaryDistance(
                src_transformed, view, proj, W, H, bdy,
                out_of_frame_dist_px, max_dist_cap_px,
                &n_in_frame, &n_in_mask);
            n_tried++;

            double final_cost = cost;
            const float in_rate = (src_pts_world.empty()
                                       ? 0.0f
                                       : float(n_in_frame)
                                             / float(src_pts_world.size()));
            if (in_rate < min_in_frame_rate) {
                final_cost = 1e9 + double(1.0f - in_rate);
                n_rejected_inframe++;
            }

            // Step 3b: hard reject candidates whose rotation exceeds
            //   max_rot_deg from identity. Trusts the user's Init Pose
            //   as the rotational anchor; Coarse2D only refines within
            //   the cone. Skip the candidate entirely (not even kept
            //   as a "penalized" rank, so it can never win).
            //   cos(angle) = (trace(R) - 1) / 2
            const float cos_angle_cand =
                0.5f * (R[0][0] + R[1][1] + R[2][2] - 1.0f);
            const float cos_thresh_hard =
                std::cos(glm::radians(max_rot_deg));
            if (cos_angle_cand < cos_thresh_hard) {
                n_rejected_rot++;
                continue;   // skip this candidate entirely
            }

            glm::mat4 T4(1.0f);
            T4[0] = glm::vec4(R[0], 0.0f);
            T4[1] = glm::vec4(R[1], 0.0f);
            T4[2] = glm::vec4(R[2], 0.0f);
            T4[3] = glm::vec4(t_vec, 1.0f);

            candidates_out.push_back({T4, final_cost, k, k, sign});
        }
    }

    if (log) {
        (*log) << "[Coarse2D] M_anchors=" << M
               << "  signs=" << n_signs
               << "(mask=0x" << std::hex << (int)sign_mask << std::dec
               << ")  src=" << src_pts_world.size()
               << "  cands=" << candidates_out.size()
               << "  tried=" << n_tried
               << "  rejected_inframe=" << n_rejected_inframe
               << "  rejected_rot=" << n_rejected_rot
               << " (max=" << max_rot_deg << "°)"
               << "  skipped_tangent=" << n_skipped_tangent
               << "  cam_axis=(" << cam_axis_world.x << ","
               << cam_axis_world.y << "," << cam_axis_world.z << ")"
               << std::endl;
    }

    return !candidates_out.empty();
}


// =====================================================================
//  Phase 7b Step 3b — Gauss-Newton refinement (Alt+W)
// =====================================================================
//
//  Treats rim alignment as a PnP-style nonlinear least squares:
//
//    Unknown: ξ ∈ se(3) = [ρ; ω] ∈ ℝ⁶  (translation 3 + rotation 3)
//    Observations: rim 3D points p_i (CT, reliable)
//    Target: unsigned distance-to-boundary field bdy(pixel)
//
//    Residual:  r_i(ξ) = bdy( π(view · exp(ξ)·T_0 · p_i) )
//    Cost:      F(ξ)  = Σ_i r_i²
//    Update:    ξ ← ξ − (JᵀJ + λI)⁻¹ Jᵀ r          (Levenberg-Marquardt)
//
//  Jacobian chain (left perturbation, T_{k+1} = exp(Δξ)·T_k):
//
//    ∂r/∂ξ = (∂bdy/∂pixel) · (∂pixel/∂p_world) · (∂p_world/∂Δξ)
//             1×2              2×3                  3×6
//
//    ∂p_world/∂Δξ = [I_3 | −[p_w]_×]
//
//    Letting M = projMat · viewMat, w = clip.w, ndc = clip.xyz/w:
//      ∂pixel.x/∂p_world =  W/(2w) · ( M_row0_xyz − ndc.x · M_row3_xyz )
//      ∂pixel.y/∂p_world = −H/(2w) · ( M_row1_xyz − ndc.y · M_row3_xyz )
//
//    ∂bdy/∂pixel via exact bilinear gradient at the sub-pixel sample.
//
//  Why unsigned distance map (not g_boundaryDistMap directly):
//    g_boundaryDistMap is piecewise constant (sentinel 9999) outside the
//    SAM2 mask → no gradient there → GN can't pull outside points back.
//    We BFS from the boundary contour without the mask gate, producing
//    a smooth field that pulls rim points toward the boundary from
//    either side. One-time build per Shift+W (or invalidate on Run Depth).
//
//  Performance: ~1 ms per iteration with 254 rim points + 6×6 ldlt
//               solve. Typical convergence in 5–15 iters → ~10–20 ms.
//               Coarse2D (1 ms) + GN (15 ms) ≈ 20 ms total per Alt+W.
//
//  Convention: ξ = [ρ; ω] (rho first 3, omega last 3). T_new = exp(Δξ)·T.
// =====================================================================


// ---------------------------------------------------------------------
// skewSym
//   Build the 3×3 skew-symmetric matrix [v]× such that [v]× · w = v × w.
//   GLM is column-major, so we construct by columns:
//     [v]× = [[0,-v.z, v.y], [v.z, 0,-v.x], [-v.y, v.x, 0]]
// ---------------------------------------------------------------------
inline glm::mat3 skewSym(const glm::vec3& v)
{
    return glm::mat3(
        glm::vec3( 0.0f,  v.z, -v.y),   // column 0
        glm::vec3(-v.z,  0.0f,  v.x),   // column 1
        glm::vec3( v.y, -v.x,  0.0f));  // column 2
}


// ---------------------------------------------------------------------
// expSE3
//   Lie-algebra exponential map: ξ = [ρ; ω] ∈ se(3) → T ∈ SE(3).
//
//     θ = ||ω||
//     R = I + A·[ω]× + B·[ω]×²   (Rodrigues)
//     V = I + B·[ω]× + C·[ω]×²
//     t = V · ρ
//     T = [R t; 0 1]
//
//   Coefficients (Taylor-stable near θ=0):
//     A = sin(θ)/θ                    ≈ 1 − θ²/6 + …
//     B = (1 − cos(θ))/θ²             ≈ 1/2 − θ²/24 + …
//     C = (θ − sin(θ))/θ³             ≈ 1/6 − θ²/120 + …
// ---------------------------------------------------------------------
inline glm::mat4 expSE3(const glm::vec3& rho, const glm::vec3& omega)
{
    const float theta2 = glm::dot(omega, omega);
    const float theta  = std::sqrt(theta2);
    float A, B, C;
    if (theta < 1e-4f) {
        const float t4 = theta2 * theta2;
        A = 1.0f   - theta2 / 6.0f   + t4 / 120.0f;
        B = 0.5f   - theta2 / 24.0f  + t4 / 720.0f;
        C = 1.0f/6.0f - theta2 / 120.0f + t4 / 5040.0f;
    } else {
        const float s = std::sin(theta);
        const float c = std::cos(theta);
        A = s / theta;
        B = (1.0f - c) / theta2;
        C = (theta - s) / (theta2 * theta);
    }
    const glm::mat3 W   = skewSym(omega);
    const glm::mat3 W2  = W * W;
    const glm::mat3 I3  = glm::mat3(1.0f);
    const glm::mat3 R   = I3 + A * W + B * W2;
    const glm::mat3 V   = I3 + B * W + C * W2;
    const glm::vec3 t   = V * rho;

    glm::mat4 T(1.0f);
    T[0] = glm::vec4(R[0], 0.0f);
    T[1] = glm::vec4(R[1], 0.0f);
    T[2] = glm::vec4(R[2], 0.0f);
    T[3] = glm::vec4(t,    1.0f);
    return T;
}


// ---------------------------------------------------------------------
// buildUnsignedBoundaryMap
//   BFS-propagated distance to the SAM2 boundary contour, filling BOTH
//   sides (inside and outside the mask). Output values:
//      0      = on the boundary
//      1..N   = pixel-distance to nearest boundary pixel
//
//   Boundary seeds = pixels where bdy_inside.data[i] < 1.5f and not the
//   9999 sentinel (i.e. inner 1-px ring of the mask boundary).
//
//   Disconnected regions get 1000.0f (a soft "far away" cap; the BFS
//   should normally reach everything in a connected image).
//
//   Cost: O(W·H), one full-image BFS pass. ~10–20 ms on 1920×1080.
// ---------------------------------------------------------------------
inline bool buildUnsignedBoundaryMap(
    const BoundaryDistMap& bdy_inside,
    std::vector<float>& dist_out,
    int& W_out, int& H_out,
    std::ostream* log = nullptr)
{
    if (!bdy_inside.valid || bdy_inside.width <= 0 || bdy_inside.height <= 0) {
        if (log) (*log) << "[GN/UnsignedBdy] input bdy invalid" << std::endl;
        return false;
    }
    W_out = bdy_inside.width;
    H_out = bdy_inside.height;
    dist_out.assign(size_t(W_out) * size_t(H_out), -1.0f);

    std::queue<std::pair<int,int>> q;
    int n_seeds = 0;
    for (int y = 0; y < H_out; y++) {
        for (int x = 0; x < W_out; x++) {
            const float v = bdy_inside.data[y * W_out + x];
            if (v >= 0.0f && v < 1.5f && v < 9000.0f) {
                dist_out[y * W_out + x] = 0.0f;
                q.push({x, y});
                n_seeds++;
            }
        }
    }
    if (n_seeds == 0) {
        if (log) (*log) << "[GN/UnsignedBdy] no boundary seeds" << std::endl;
        return false;
    }

    const int dx4[] = { 1, -1,  0,  0 };
    const int dy4[] = { 0,  0,  1, -1 };
    while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        const float cd = dist_out[cy * W_out + cx];
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx4[d], ny = cy + dy4[d];
            if (nx < 0 || nx >= W_out || ny < 0 || ny >= H_out) continue;
            int ni = ny * W_out + nx;
            if (dist_out[ni] < 0.0f) {
                dist_out[ni] = cd + 1.0f;
                q.push({nx, ny});
            }
        }
    }
    for (size_t i = 0; i < dist_out.size(); i++) {
        if (dist_out[i] < 0.0f) dist_out[i] = 1000.0f;
    }
    if (log) {
        float maxd = 0.0f;
        for (float v : dist_out) {
            if (v < 999.0f && v > maxd) maxd = v;
        }
        (*log) << "[GN/UnsignedBdy] built " << W_out << "x" << H_out
               << "  seeds=" << n_seeds << "  reached_max=" << maxd << "px"
               << std::endl;
    }
    return true;
}


// ---------------------------------------------------------------------
// GNResult
//   Returned by runShapeMatchGN. cost_history contains the cost at the
//   start and after each ACCEPTED LM step (rejected steps not logged).
// ---------------------------------------------------------------------
struct GNResult {
    glm::mat4 final_T       = glm::mat4(1.0f);
    double    initial_cost  = 0.0;
    double    final_cost    = 0.0;
    int       n_iter        = 0;
    bool      converged     = false;
    int       reason        = 2;   // 0=step, 1=rel_cost, 2=max_iter, 3=lm_fail
    int       n_in_frame    = 0;   // at final pose
    std::vector<double> cost_history;
};


// ---------------------------------------------------------------------
// runShapeMatchGN
//   Levenberg-Marquardt refinement of T such that src_pts_world
//   project onto the boundary contour. T_init is the starting transform
//   (typically the best Coarse2D candidate).
//
//   The implementation samples bdy_unsigned with exact bilinear
//   gradient at each sub-pixel sample, builds a 6×6 normal equation per
//   iteration, and applies left perturbation T_new = exp(Δξ) · T.
//
//   max_iter        : hard cap on iterations (default 30)
//   lm_lambda_init  : initial LM damping (default 1e-3)
//   eps_step        : converge if ||Δξ|| < eps_step
//   eps_rel         : converge if |ΔF/F| < eps_rel
//
//   On reject: λ ×= 4 (up to 1e9). On accept: λ /= 2.
// ---------------------------------------------------------------------
inline bool runShapeMatchGN(
    const std::vector<glm::vec3>& src_pts_world,
    const glm::mat4& T_init,
    const glm::mat4& view, const glm::mat4& proj,
    const std::vector<float>& bdy_unsigned,
    int bdy_W, int bdy_H,
    int   max_iter,
    float lm_lambda_init,
    float eps_step,
    float eps_rel,
    GNResult& result_out,
    std::ostream* log = nullptr,
    bool  translation_only = false,    // Step 3b: lock rotation (Alt+W default)
    float lm_lambda_min    = 1.0e-3f,  // Step 3b: prevent depth runaway
    float max_step_norm    = 0.05f)    // Step 3b: trust-region on ||Δξ||
{
    result_out = GNResult{};
    result_out.final_T = T_init;
    if (src_pts_world.empty() || bdy_unsigned.empty()) {
        if (log) (*log) << "[GN] empty input" << std::endl;
        return false;
    }
    if (bdy_W <= 2 || bdy_H <= 2 ||
        (int)bdy_unsigned.size() < bdy_W * bdy_H) {
        if (log) (*log) << "[GN] bdy_unsigned size mismatch" << std::endl;
        return false;
    }

    const glm::mat4 M = proj * view;     // VP matrix
    // Row slices of M (xyz only) for the perspective Jacobian.
    // GLM is column-major: M[col][row] = element (row, col).
    const glm::vec3 Mr0_xyz(M[0][0], M[1][0], M[2][0]);
    const glm::vec3 Mr1_xyz(M[0][1], M[1][1], M[2][1]);
    const glm::vec3 Mr3_xyz(M[0][3], M[1][3], M[2][3]);
    const int N = (int)src_pts_world.size();

    // Bilinear sample with exact bilinear gradient (in pixel units).
    auto sample_with_grad = [&](float px, float py,
                                float& gx, float& gy) -> float {
        if (px < 0.0f) px = 0.0f;
        if (py < 0.0f) py = 0.0f;
        if (px > float(bdy_W - 2)) px = float(bdy_W - 2);
        if (py > float(bdy_H - 2)) py = float(bdy_H - 2);
        const int x0 = int(std::floor(px));
        const int y0 = int(std::floor(py));
        const float fx = px - float(x0);
        const float fy = py - float(y0);
        const float v00 = bdy_unsigned[y0     * bdy_W + x0];
        const float v10 = bdy_unsigned[y0     * bdy_W + (x0 + 1)];
        const float v01 = bdy_unsigned[(y0+1) * bdy_W + x0];
        const float v11 = bdy_unsigned[(y0+1) * bdy_W + (x0 + 1)];
        const float v = (1.0f - fx) * (1.0f - fy) * v00
                      +         fx  * (1.0f - fy) * v10
                      + (1.0f - fx) *         fy  * v01
                      +         fx  *         fy  * v11;
        gx = (1.0f - fy) * (v10 - v00) + fy * (v11 - v01);
        gy = (1.0f - fx) * (v01 - v00) + fx * (v11 - v10);
        return v;
    };

    // Cost evaluation only (no Jacobian). Out-of-frame penalty = 100 px,
    // matches the Coarse2D outside cap so the LM "cost" stays comparable.
    auto eval_cost = [&](const glm::mat4& T, int* n_in_frame_out) -> double {
        double sum = 0.0;
        int n_in_frame = 0;
        for (int i = 0; i < N; i++) {
            const glm::vec4 p_T  = T * glm::vec4(src_pts_world[i], 1.0f);
            const glm::vec4 clip = M * p_T;
            if (clip.w <= 1e-9f) { sum += 100.0; continue; }
            const float inv_w = 1.0f / clip.w;
            const float ndcx = clip.x * inv_w;
            const float ndcy = clip.y * inv_w;
            if (ndcx < -1.0f || ndcx > 1.0f ||
                ndcy < -1.0f || ndcy > 1.0f) {
                sum += 100.0;
                continue;
            }
            n_in_frame++;
            const float px = (ndcx + 1.0f) * 0.5f * float(bdy_W);
            const float py = (1.0f - ndcy) * 0.5f * float(bdy_H);
            float gx, gy;
            sum += sample_with_grad(px, py, gx, gy);
        }
        if (n_in_frame_out) *n_in_frame_out = n_in_frame;
        return sum / double(N);
    };

    glm::mat4 T = T_init;
    int n_in_frame_curr = 0;
    double cost = eval_cost(T, &n_in_frame_curr);
    result_out.initial_cost = cost;
    result_out.cost_history.push_back(cost);
    result_out.n_in_frame   = n_in_frame_curr;

    if (log) {
        (*log) << "[GN] start cost=" << cost << "px in_frame="
               << n_in_frame_curr << "/" << N
               << " lambda=" << lm_lambda_init
               << " λ_min=" << lm_lambda_min
               << " step_max=" << max_step_norm
               << " mode=" << (translation_only ? "TRANS_ONLY(3-DoF)" : "FULL(6-DoF)")
               << std::endl;
    }

    double lambda = lm_lambda_init;
    int    iter = 0;
    bool   converged = false;
    int    reason = 2;  // default = max_iter

    for (iter = 1; iter <= max_iter; iter++) {
        // Build 6×6 normal equations from in-frame points only.
        Eigen::Matrix<double, 6, 6> JTJ = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> JTr = Eigen::Matrix<double, 6, 1>::Zero();
        int n_used = 0;

        for (int i = 0; i < N; i++) {
            const glm::vec4 p_T  = T * glm::vec4(src_pts_world[i], 1.0f);
            const glm::vec3 p_w(p_T.x, p_T.y, p_T.z);
            const glm::vec4 clip = M * p_T;
            if (clip.w <= 1e-9f) continue;
            const float inv_w = 1.0f / clip.w;
            const float ndcx = clip.x * inv_w;
            const float ndcy = clip.y * inv_w;
            if (ndcx < -1.0f || ndcx > 1.0f ||
                ndcy < -1.0f || ndcy > 1.0f) continue;
            const float px = (ndcx + 1.0f) * 0.5f * float(bdy_W);
            const float py = (1.0f - ndcy) * 0.5f * float(bdy_H);
            float gx_px, gy_px;
            const float r = sample_with_grad(px, py, gx_px, gy_px);

            // ∂pixel/∂p_world  ∈ ℝ²ˣ³
            const float sx =  0.5f * float(bdy_W) * inv_w;
            const float sy = -0.5f * float(bdy_H) * inv_w;
            const glm::vec3 dpx_dp = sx * (Mr0_xyz - ndcx * Mr3_xyz);
            const glm::vec3 dpy_dp = sy * (Mr1_xyz - ndcy * Mr3_xyz);

            // ∂r/∂p_world = gx · dpx_dp + gy · dpy_dp  ∈ ℝ¹ˣ³
            const glm::vec3 L = gx_px * dpx_dp + gy_px * dpy_dp;

            // Left perturbation: ∂p_world/∂ξ = [ I | -[p_w]_× ]
            // J_i = L · [ I | -[p_w]_× ] = [ L , -L · [p_w]_× ]
            //     = [ L , L × p_w ]    (row-vector × skew identity)
            // Because L · skew(p) = -p × L = L × (-p) → −L·skew(p) = p × L … hmm.
            // Derivation: L · skew(p) = -(p × L) (treating L as row). So
            // -L · skew(p) = p × L. Equivalently L × p has different sign.
            // We need to double-check; pick the sign that physically pulls
            // p toward the gradient. We'll use: -L · skew(p_w) = cross(p_w, L)
            const glm::vec3 J_w = glm::cross(p_w, L);

            Eigen::Matrix<double, 6, 1> Ji;
            Ji(0) = L.x;   Ji(1) = L.y;   Ji(2) = L.z;
            Ji(3) = J_w.x; Ji(4) = J_w.y; Ji(5) = J_w.z;

            JTJ.noalias() += Ji * Ji.transpose();
            JTr.noalias() += Ji * double(r);
            n_used++;
        }

        if (n_used < 6) {
            if (log) (*log) << "[GN] iter " << iter
                       << " too few used points (" << n_used
                       << ") — abort" << std::endl;
            reason = 3;
            break;
        }

        // Solve (JTJ + λI) Δξ = −JTr
        // For translation_only mode: zero out the rotation block (last 3
        //   rows/cols of JTJ and last 3 rows of JTr) so the solver only
        //   moves the translation 3-vector. Equivalent to a separate 3×3
        //   solve, but kept in one path for code unity.
        Eigen::Matrix<double, 6, 6> H = JTJ;
        Eigen::Matrix<double, 6, 1> g = JTr;
        if (translation_only) {
            // Lock rotation: set rotation block to identity·1, rotation
            //   gradient to 0. Solver returns Δξ_rot = 0 exactly.
            for (int i = 3; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    H(i, j) = (i == j) ? 1.0 : 0.0;
                    H(j, i) = (i == j) ? 1.0 : 0.0;
                }
                g(i) = 0.0;
            }
        }
        for (int d = 0; d < 6; d++) H(d, d) += lambda;
        Eigen::Matrix<double, 6, 1> dxi = H.ldlt().solve(-g);

        // Step 3b: trust-region clamp on ||Δξ||
        //   Prevents the huge first-step explosion (||dxi||=0.62 seen in
        //   the log) that lets PnP escape into depth-degenerate minima.
        //   max_step_norm=0.05 ≈ 5cm + 3°, comfortable per LM iter.
        const double dxi_norm_raw = dxi.norm();
        if (dxi_norm_raw > double(max_step_norm)) {
            dxi *= double(max_step_norm) / dxi_norm_raw;
        }

        const glm::vec3 rho(   float(dxi(0)), float(dxi(1)), float(dxi(2)));
        const glm::vec3 omega( float(dxi(3)), float(dxi(4)), float(dxi(5)));
        const glm::mat4 T_trial = expSE3(rho, omega) * T;

        int n_in_frame_trial = 0;
        const double cost_trial = eval_cost(T_trial, &n_in_frame_trial);
        const double dxi_norm   = dxi.norm();

        if (log) {
            (*log) << "[GN] iter " << iter
                   << "  cost " << cost << "→" << cost_trial << "px"
                   << "  ||dxi||=" << dxi_norm
                   << "  λ=" << lambda
                   << "  used=" << n_used << "/" << N
                   << "  in_frame=" << n_in_frame_trial << "/" << N
                   << ((cost_trial < cost) ? " [accept]" : " [reject]")
                   << std::endl;
        }

        if (cost_trial < cost) {
            const double rel = std::abs(cost_trial - cost)
                               / std::max(1e-9, cost);
            cost = cost_trial;
            T = T_trial;
            lambda = std::max((double)lm_lambda_min, lambda * 0.5);
            result_out.cost_history.push_back(cost);
            result_out.n_in_frame = n_in_frame_trial;

            if (dxi_norm < eps_step) {
                converged = true; reason = 0; break;
            }
            if (rel < eps_rel) {
                converged = true; reason = 1; break;
            }
        } else {
            lambda *= 4.0;
            if (lambda > 1e9) {
                if (log) (*log) << "[GN] λ exploded — abort" << std::endl;
                reason = 3;
                break;
            }
        }
    }

    result_out.final_T   = T;
    result_out.final_cost = cost;
    result_out.n_iter    = iter;
    result_out.converged = converged;
    result_out.reason    = reason;

    if (log) {
        const char* reason_str[] = {"step", "rel", "max_iter", "lm_fail"};
        (*log) << "[GN] done iters=" << iter
               << "  cost " << result_out.initial_cost
               << " → " << cost << "px"
               << "  Δ=" << (result_out.initial_cost - cost)
               << "  converged=" << (converged ? "Y" : "N")
               << "  reason=" << reason_str[std::min(3, reason)]
               << "  in_frame=" << result_out.n_in_frame << "/" << N
               << std::endl;
    }
    return true;
}


// =====================================================================
//  Phase 7b Step 3c — Contour Sweep (Ctrl+Alt+W)
// =====================================================================
//
//  User-proposed "Shift+P 風" algorithm: discretize both the target
//  contour and the source rim into arc-length anchors, then sweep
//  every (target_anchor, source_pivot, rotation) triple as discrete
//  correspondence candidates. Replaces the gradient-driven GN with a
//  combinatorial search whose every move is geometrically grounded
//  (no depth-degeneracy because Z is locked by construction).
//
//  Math (one candidate (i, j, θ)):
//    P_s   = src_pivots_3D[j]                          // current Z preserved
//    P_t   = unproject(tgt_anchors_2D[i], depth=P_s.z) // lift at P_s's depth
//    R     = world-Z rotation by θ (camera-forward axis in AR view)
//    T(p)  = R · (p − P_s) + P_t                       // = R(p−P_s) + P_s + (P_t−P_s)
//
//  Because P_t.z = P_s.z and R is a Z-axis rotation, the translation
//  (P_t − P_s) has Z=0 and R preserves Z. The world Z of every source
//  point is invariant. Depth runaway is structurally impossible.
//
//  Cost-C (double chamfer over the 20-vs-20 pivot sets):
//    src2D_i = π(T · src_pivots_3D[i])                  // i=0..N_src
//    fwd  = (1/N_src) Σ_i  bdy_unsigned[src2D_i]        // src pulled to boundary
//    bwd  = (1/N_tgt) Σ_j  min_i ||tgt_anchors[j] − src2D_i||
//    cost = fwd + bwd
//
//  Phase 1 (broad): N_tgt × N_src × N_rot ≈ 20×20×36 = 14,400 candidates
//  Phase 2 (fine):  same dims but re-anchored around Phase 1 best with
//                   narrower arc-length / angle range (≈ ±1 phase-1 step)
//
//  Frame-by-frame ticking: caller chooses candidates_per_frame so that
//  total / batch = target_frames (60fps × 2.5sec ≈ 150 frames / phase).
// =====================================================================


// ---------------------------------------------------------------------
// arclengthResampleRim3D
//   Treat rim_chain (closed loop) as a 3D polyline; uniformly resample
//   to N points by arc length. The chain is assumed sorted in geometric
//   order (sortRimChainByAngle ensures this); closure between last and
//   first index is included in the arc length.
//
//   --- Phase 7b Step 3c++ (label-based orientation lock) ----------------
//   out_nearest_vert_idx (optional):
//     For each output pivot k, the *mesh vertex index* of the closer
//     rim_chain endpoint of the polyline segment it was interpolated
//     within. Lets the caller look up per-pivot anatomical labels
//     (LiverLeftRightLabel, LiverCranioCaudalLabel) without rebuilding
//     the chain elsewhere. Pass nullptr to skip.
// ---------------------------------------------------------------------
inline void arclengthResampleRim3D(
    const std::vector<int>& rim_chain,
    const std::vector<float>& mesh_verts,   // flat x,y,z,x,y,z,...
    int N,
    std::vector<glm::vec3>& resampled_out,
    std::vector<int>* out_nearest_vert_idx = nullptr)
{
    resampled_out.clear();
    if (out_nearest_vert_idx) out_nearest_vert_idx->clear();
    const int K = (int)rim_chain.size();
    if (K < 2 || N < 1) return;
    const int nV3 = (int)mesh_verts.size();

    // Polyline points + parallel vertex-index array (so callers can
    // look up labels without re-walking the chain).
    std::vector<glm::vec3> pts;
    std::vector<int>       pts_vidx;
    pts.reserve(K + 1);
    pts_vidx.reserve(K + 1);
    for (int idx : rim_chain) {
        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
        pts.emplace_back(mesh_verts[idx*3],
                         mesh_verts[idx*3+1],
                         mesh_verts[idx*3+2]);
        pts_vidx.push_back(idx);
    }
    if (pts.size() < 2) return;
    pts.push_back(pts.front());  // close the loop
    pts_vidx.push_back(pts_vidx.front());

    // Cumulative arc length
    std::vector<float> cum(pts.size(), 0.0f);
    for (size_t i = 1; i < pts.size(); i++) {
        cum[i] = cum[i-1] + glm::length(pts[i] - pts[i-1]);
    }
    const float total_len = cum.back();
    if (total_len < 1e-6f) return;

    resampled_out.reserve(N);
    if (out_nearest_vert_idx) out_nearest_vert_idx->reserve(N);
    for (int k = 0; k < N; k++) {
        const float t = (float(k) + 0.5f) * total_len / float(N);
        auto it = std::upper_bound(cum.begin(), cum.end(), t);
        if (it == cum.begin())   {
            resampled_out.push_back(pts.front());
            if (out_nearest_vert_idx) out_nearest_vert_idx->push_back(pts_vidx.front());
            continue;
        }
        if (it == cum.end())     {
            resampled_out.push_back(pts.back());
            if (out_nearest_vert_idx) out_nearest_vert_idx->push_back(pts_vidx.back());
            continue;
        }
        const size_t idx     = std::distance(cum.begin(), it);
        const float  seg_len = cum[idx] - cum[idx-1];
        const float  alpha   = (seg_len > 1e-6f)
                                   ? (t - cum[idx-1]) / seg_len
                                   : 0.0f;
        resampled_out.push_back(glm::mix(pts[idx-1], pts[idx], alpha));
        if (out_nearest_vert_idx) {
            const size_t closer = (alpha < 0.5f) ? (idx - 1) : idx;
            out_nearest_vert_idx->push_back(pts_vidx[closer]);
        }
    }
}


// ---------------------------------------------------------------------
// arclengthResampleContour2DSubrange
//   Like resampleArcLength2D but only resamples the [t_center-t_radius,
//   t_center+t_radius] subrange of the arc-length parameter (t ∈ [0,1]).
//   Used by Phase 2 to refine around Phase 1's best target anchor.
//   Boundary handling: clamp (no wrap) — target contour is an open
//   polyline (the largest connected segment from traceContour2D).
// ---------------------------------------------------------------------
inline void arclengthResampleContour2DSubrange(
    const std::vector<glm::vec2>& contour,
    float t_center, float t_radius,
    int N,
    std::vector<glm::vec2>& resampled_out)
{
    resampled_out.clear();
    if (contour.size() < 2 || N < 1) return;

    std::vector<float> cum(contour.size(), 0.0f);
    for (size_t i = 1; i < contour.size(); i++) {
        cum[i] = cum[i-1] + glm::length(contour[i] - contour[i-1]);
    }
    const float total_len = cum.back();
    if (total_len < 1e-6f) return;

    const float t0 = std::max(0.0f, t_center - t_radius) * total_len;
    const float t1 = std::min(1.0f, t_center + t_radius) * total_len;
    const float range = t1 - t0;
    if (range < 1e-6f) return;

    resampled_out.reserve(N);
    for (int k = 0; k < N; k++) {
        const float t = t0 + (float(k) + 0.5f) * range / float(N);
        auto it = std::upper_bound(cum.begin(), cum.end(), t);
        if (it == cum.begin())   { resampled_out.push_back(contour.front()); continue; }
        if (it == cum.end())     { resampled_out.push_back(contour.back());  continue; }
        const size_t idx     = std::distance(cum.begin(), it);
        const float  seg_len = cum[idx] - cum[idx-1];
        const float  alpha   = (seg_len > 1e-6f)
                                   ? (t - cum[idx-1]) / seg_len
                                   : 0.0f;
        resampled_out.push_back(glm::mix(contour[idx-1], contour[idx], alpha));
    }
}


// ---------------------------------------------------------------------
// arclengthResampleRim3DSubrange
//   Closed-loop subrange resample. t_center ± t_radius wraps modulo 1.
//   If the wrap straddles index 0, the parameter range is split (the
//   sampling is still uniform in arc length).
//
//   --- Phase 7b Step 3c++ ----------------------------------------------
//   out_nearest_vert_idx (optional): mesh vertex idx of the closer
//   chain endpoint for each output pivot, matching arclengthResampleRim3D.
// ---------------------------------------------------------------------
inline void arclengthResampleRim3DSubrange(
    const std::vector<int>& rim_chain,
    const std::vector<float>& mesh_verts,
    float t_center, float t_radius,
    int N,
    std::vector<glm::vec3>& resampled_out,
    std::vector<int>* out_nearest_vert_idx = nullptr)
{
    resampled_out.clear();
    if (out_nearest_vert_idx) out_nearest_vert_idx->clear();
    const int K = (int)rim_chain.size();
    if (K < 2 || N < 1) return;
    const int nV3 = (int)mesh_verts.size();

    std::vector<glm::vec3> pts;
    std::vector<int>       pts_vidx;
    pts.reserve(K + 1);
    pts_vidx.reserve(K + 1);
    for (int idx : rim_chain) {
        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
        pts.emplace_back(mesh_verts[idx*3],
                         mesh_verts[idx*3+1],
                         mesh_verts[idx*3+2]);
        pts_vidx.push_back(idx);
    }
    if (pts.size() < 2) return;
    pts.push_back(pts.front());
    pts_vidx.push_back(pts_vidx.front());

    std::vector<float> cum(pts.size(), 0.0f);
    for (size_t i = 1; i < pts.size(); i++) {
        cum[i] = cum[i-1] + glm::length(pts[i] - pts[i-1]);
    }
    const float total_len = cum.back();
    if (total_len < 1e-6f) return;

    auto sample = [&](float t, int* out_vidx) -> glm::vec3 {
        // Wrap t to [0, total_len]
        while (t < 0.0f)         t += total_len;
        while (t > total_len)    t -= total_len;
        auto it = std::upper_bound(cum.begin(), cum.end(), t);
        if (it == cum.begin())   {
            if (out_vidx) *out_vidx = pts_vidx.front();
            return pts.front();
        }
        if (it == cum.end())     {
            if (out_vidx) *out_vidx = pts_vidx.back();
            return pts.back();
        }
        const size_t idx     = std::distance(cum.begin(), it);
        const float  seg_len = cum[idx] - cum[idx-1];
        const float  alpha   = (seg_len > 1e-6f)
                                   ? (t - cum[idx-1]) / seg_len
                                   : 0.0f;
        if (out_vidx) {
            const size_t closer = (alpha < 0.5f) ? (idx - 1) : idx;
            *out_vidx = pts_vidx[closer];
        }
        return glm::mix(pts[idx-1], pts[idx], alpha);
    };

    const float t_lo = (t_center - t_radius) * total_len;
    const float t_hi = (t_center + t_radius) * total_len;
    const float range = t_hi - t_lo;
    resampled_out.reserve(N);
    if (out_nearest_vert_idx) out_nearest_vert_idx->reserve(N);
    for (int k = 0; k < N; k++) {
        const float t = t_lo + (float(k) + 0.5f) * range / float(N);
        int vi = -1;
        resampled_out.push_back(sample(t, out_nearest_vert_idx ? &vi : nullptr));
        if (out_nearest_vert_idx) out_nearest_vert_idx->push_back(vi);
    }
}


// ---------------------------------------------------------------------
// evaluateContourSweepCandidate
//   Single (i_tgt, j_src, θ) evaluation. Builds the 4×4 transform that
//   pins src_pivots[j_src] to tgt_anchors[i_tgt] at the source's current
//   depth, rotates around world-Z (camera forward) by θ degrees, then
//   computes the bidirectional chamfer between the transformed source
//   pivots and target anchors.
//
//   Returns cost; writes T_out. Cost convention: lower is better.
//   Out-of-frame source points contribute a large penalty (1000 px).
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// filterBoundaryByRim2DBbox
//
//   Anatomical / "stay near initial pose" filter. Computes the 2D
//   bounding box of the projected source rim (in pixel space), expands
//   it by margin_px on all sides, and keeps only the boundary 3D
//   points whose 2D projection falls inside that box.
//
//   This is what makes the Contour Sweep respect the source's initial
//   pose: by limiting target anchors to the source's vicinity, the
//   "pin pivot to anchor" translation can no longer fling the source
//   to anatomically nonsensical locations (e.g. caudal source pivot
//   onto cranial target anchor).
// ---------------------------------------------------------------------
inline void filterBoundaryByRim2DBbox(
    const std::vector<glm::vec3>& boundary_3D,
    const std::vector<glm::vec3>& src_rim_3D,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    float margin_px,
    std::vector<glm::vec3>& out_filtered)
{
    out_filtered.clear();
    if (boundary_3D.empty() || src_rim_3D.empty() || margin_px < 0.0f) {
        out_filtered = boundary_3D;
        return;
    }

    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);

    auto project_2D = [&](const glm::vec3& p, glm::vec2& out) -> bool {
        const glm::vec4 clip = M * glm::vec4(p, 1.0f);
        if (std::abs(clip.w) < 1e-9f) return false;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        out = glm::vec2(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
        return true;
    };

    // 1. Compute source rim 2D bbox
    glm::vec2 src_min( 1e9f,  1e9f);
    glm::vec2 src_max(-1e9f, -1e9f);
    bool any_valid = false;
    glm::vec2 p2d;
    for (const auto& p : src_rim_3D) {
        if (!project_2D(p, p2d)) continue;
        src_min = glm::min(src_min, p2d);
        src_max = glm::max(src_max, p2d);
        any_valid = true;
    }
    if (!any_valid) {
        out_filtered = boundary_3D;
        return;
    }
    src_min -= glm::vec2(margin_px);
    src_max += glm::vec2(margin_px);

    // 2. Filter boundary points by inclusion in expanded bbox
    out_filtered.reserve(boundary_3D.size() / 2);
    for (const auto& p : boundary_3D) {
        if (!project_2D(p, p2d)) continue;
        if (p2d.x >= src_min.x && p2d.x <= src_max.x
            && p2d.y >= src_min.y && p2d.y <= src_max.y)
        {
            out_filtered.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------
// Phase 7b Step 3c++: ScreenSide enums (label-based orientation lock)
//
//   After Apply Init Pose, the canonical viewing convention is:
//     screen +x  ↔  patient's RIGHT  (anatomical)
//     screen -y  ↔  patient's CRANIAL (top of image = head)
//
//   These enums classify each target anchor by which side of the
//   locked-centroid neutral band it falls in. AMBIGUOUS means "inside
//   the band" — used to relax the match constraint at the image
//   midline where the source's BOUNDARY-labeled pivots also live.
// ---------------------------------------------------------------------
enum ScreenSideLR : uint8_t {
    SS_LR_RIGHT     = 0,
    SS_LR_LEFT      = 1,
    SS_LR_AMBIGUOUS = 2
};
enum ScreenSideCC : uint8_t {
    SS_CC_CRANIAL   = 0,
    SS_CC_CAUDAL    = 1,
    SS_CC_AMBIGUOUS = 2
};

// ---------------------------------------------------------------------
// tagAnchorsInsideSourceBbox
//
//   Plan A (anchor-完全固定) helper. Given target anchors that were
//   extracted from the FULL boundary (no pre-filter, so locations are
//   invariant under source pose), tag each anchor true/false for
//   inclusion in the source rim's 2D bbox (expanded by margin_px).
//
//   Use:
//     - sweep can skip i_tgt with tag=false (anchors far from the
//       current source) without ever moving the anchor itself
//     - preview can render outside-bbox anchors in dim gray so the
//       user can see the bbox boundary visually
//
//   When margin_px < 0 or src_rim is empty, all anchors are tagged true
//   (filter effectively off — caller decides whether to skip).
// ---------------------------------------------------------------------
inline void tagAnchorsInsideSourceBbox(
    const std::vector<glm::vec3>& anchors_3D,
    const std::vector<glm::vec3>& src_rim_3D,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    float margin_px,
    std::vector<uint8_t>& out_inside_mask)
{
    out_inside_mask.assign(anchors_3D.size(), (uint8_t)1);
    if (anchors_3D.empty() || src_rim_3D.empty() || margin_px < 0.0f) {
        return;
    }

    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);

    auto project_2D = [&](const glm::vec3& p, glm::vec2& out) -> bool {
        const glm::vec4 clip = M * glm::vec4(p, 1.0f);
        if (std::abs(clip.w) < 1e-9f) return false;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        out = glm::vec2(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
        return true;
    };

    // Source rim 2D bbox
    glm::vec2 src_min( 1e9f,  1e9f);
    glm::vec2 src_max(-1e9f, -1e9f);
    bool any_valid = false;
    glm::vec2 p2d;
    for (const auto& p : src_rim_3D) {
        if (!project_2D(p, p2d)) continue;
        src_min = glm::min(src_min, p2d);
        src_max = glm::max(src_max, p2d);
        any_valid = true;
    }
    if (!any_valid) return;   // leave everything as "inside" (1)
    src_min -= glm::vec2(margin_px);
    src_max += glm::vec2(margin_px);

    for (size_t i = 0; i < anchors_3D.size(); i++) {
        if (!project_2D(anchors_3D[i], p2d)) {
            out_inside_mask[i] = 0;
            continue;
        }
        const bool inside = (p2d.x >= src_min.x && p2d.x <= src_max.x
                          && p2d.y >= src_min.y && p2d.y <= src_max.y);
        out_inside_mask[i] = inside ? (uint8_t)1 : (uint8_t)0;
    }
}

// ---------------------------------------------------------------------
// extractSectorBasedTargetAnchors3D
//
//   Replacement for arc-length resampling of 2D contour traces. Takes
//   the full set of 3D target boundary points (e.g. 55,000 from
//   g_debugTargetBoundaryPoints), projects each to 2D using the AR
//   camera, partitions them into N angular sectors around the 2D
//   centroid, and returns the medoid (point closest to its sector's
//   3D centroid) of each non-empty sector.
//
//   This MATCHES the approach used by Shift+P (Cyclic Boundary
//   Registration) and is robust to:
//     - Boundary fragmentation (SAM mask holes, instrument cuts)
//     - Disconnected segments
//     - Non-uniform point density along the boundary
//
//   Anchors are returned in CCW angular order. Empty sectors are
//   skipped, so out_anchors_3D.size() may be less than n_sectors.
//
//   --- Phase 7b Step 3c bug fix (centroid lock) -----------------------
//   full_boundary_3D_for_centroid (optional, default nullptr):
//     When provided, the 2D centroid that anchors the sector partition
//     is computed from THIS point set instead of `boundary_points_3D`.
//
//     Typical use: pass the *unfiltered* g_debugTargetBoundaryPoints
//     here while `boundary_points_3D` carries the source-bbox-filtered
//     subset. The result:
//       - Centroid is locked to the target image (invariant to source
//         pose changes — bbox filter no longer drags it around).
//       - Sector boundary lines are therefore also locked in image space.
//       - Only the subset of boundary points that get binned into each
//         sector depends on the source bbox filter; medoid selection
//         picks the locally most-representative point of each sector.
//
//     This is the fix for "source 移動で target anchor 位置が変わる"
//     reported in HANDOVER_PHASE_7B_STEP3C.md §bug 1.
// ---------------------------------------------------------------------
inline void extractSectorBasedTargetAnchors3D(
    const std::vector<glm::vec3>& boundary_points_3D,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    int n_sectors,
    std::vector<glm::vec3>& out_anchors_3D,
    std::vector<int>* out_sector_idx = nullptr,
    glm::vec2* out_centroid_2D = nullptr,
    const std::vector<glm::vec3>* full_boundary_3D_for_centroid = nullptr)
{
    out_anchors_3D.clear();
    if (out_sector_idx) out_sector_idx->clear();
    if (boundary_points_3D.empty() || n_sectors < 2) return;

    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);

    auto project_2D_inframe = [&](const glm::vec3& p,
                                  glm::vec2& out2D) -> bool
    {
        const glm::vec4 clip = M * glm::vec4(p, 1.0f);
        if (std::abs(clip.w) < 1e-9f) return false;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        if (ndcx < -1.5f || ndcx > 1.5f || ndcy < -1.5f || ndcy > 1.5f)
            return false;
        out2D = glm::vec2(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
        return true;
    };

    // --- Project + per-point validity for the bin-target points ------
    std::vector<glm::vec2> pts_2D(boundary_points_3D.size());
    std::vector<bool>      pts_valid(boundary_points_3D.size(), false);
    for (size_t i = 0; i < boundary_points_3D.size(); i++) {
        if (project_2D_inframe(boundary_points_3D[i], pts_2D[i])) {
            pts_valid[i] = true;
        }
    }

    // --- Centroid: from full_boundary (if provided) OR from binned set
    //     This is the key "lock" that makes the partition source-pose-
    //     invariant when the caller passes the unfiltered target set.
    glm::vec2 centroid_2D(0.0f);
    int n_valid_for_centroid = 0;
    if (full_boundary_3D_for_centroid != nullptr
        && !full_boundary_3D_for_centroid->empty())
    {
        glm::vec2 p2;
        for (const auto& p : *full_boundary_3D_for_centroid) {
            if (project_2D_inframe(p, p2)) {
                centroid_2D += p2;
                n_valid_for_centroid++;
            }
        }
        if (n_valid_for_centroid == 0) {
            // Full-boundary projection had no in-frame points;
            // fall back to binned-set centroid for robustness.
            for (size_t i = 0; i < pts_2D.size(); i++) {
                if (!pts_valid[i]) continue;
                centroid_2D += pts_2D[i];
                n_valid_for_centroid++;
            }
        }
    } else {
        for (size_t i = 0; i < pts_2D.size(); i++) {
            if (!pts_valid[i]) continue;
            centroid_2D += pts_2D[i];
            n_valid_for_centroid++;
        }
    }
    if (n_valid_for_centroid == 0) return;
    centroid_2D /= float(n_valid_for_centroid);
    if (out_centroid_2D) *out_centroid_2D = centroid_2D;

    const float kTwoPi = 6.283185307179586f;
    const float sector_size = kTwoPi / float(n_sectors);
    std::vector<std::vector<size_t>> sectorIdx(n_sectors);
    for (size_t i = 0; i < pts_2D.size(); i++) {
        if (!pts_valid[i]) continue;
        const glm::vec2 d = pts_2D[i] - centroid_2D;
        if (glm::length(d) < 1e-6f) continue;
        float theta = std::atan2(d.y, d.x);
        if (theta < 0.0f) theta += kTwoPi;
        int s = int(theta / sector_size);
        if (s >= n_sectors) s = n_sectors - 1;
        if (s < 0) s = 0;
        sectorIdx[s].push_back(i);
    }

    out_anchors_3D.reserve(n_sectors);
    if (out_sector_idx) out_sector_idx->reserve(n_sectors);
    for (int s = 0; s < n_sectors; s++) {
        if (sectorIdx[s].empty()) continue;
        glm::vec3 sec_c(0.0f);
        for (size_t idx : sectorIdx[s]) sec_c += boundary_points_3D[idx];
        sec_c /= float(sectorIdx[s].size());
        size_t best_idx = sectorIdx[s][0];
        float best_d = glm::length(boundary_points_3D[best_idx] - sec_c);
        for (size_t idx : sectorIdx[s]) {
            const float d = glm::length(boundary_points_3D[idx] - sec_c);
            if (d < best_d) { best_d = d; best_idx = idx; }
        }
        out_anchors_3D.push_back(boundary_points_3D[best_idx]);
        if (out_sector_idx) out_sector_idx->push_back(s);
    }
}

// ---------------------------------------------------------------------
// extractSectorBasedTargetAnchors3DSubrange
//
//   Phase 2 variant: refine target anchors within a narrower angular
//   window around the Phase 1 best sector.
//
//   --- Phase 7b Step 3c bug fix (centroid lock) -----------------------
//   full_boundary_3D_for_centroid: see extractSectorBasedTargetAnchors3D
//   docstring. Passing the unfiltered target boundary here keeps the
//   subrange centered on the same locked centroid that Phase 1 used,
//   so Phase 2's anchors don't slide as the source moves either.
// ---------------------------------------------------------------------
inline void extractSectorBasedTargetAnchors3DSubrange(
    const std::vector<glm::vec3>& boundary_points_3D,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    float center_angle,
    float half_range,
    int n_subsectors,
    std::vector<glm::vec3>& out_anchors_3D,
    const std::vector<glm::vec3>* full_boundary_3D_for_centroid = nullptr)
{
    out_anchors_3D.clear();
    if (boundary_points_3D.empty() || n_subsectors < 2) return;

    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);
    const float kTwoPi = 6.283185307179586f;

    auto project_2D_inframe = [&](const glm::vec3& p,
                                  glm::vec2& out2D) -> bool
    {
        const glm::vec4 clip = M * glm::vec4(p, 1.0f);
        if (std::abs(clip.w) < 1e-9f) return false;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        if (ndcx < -1.5f || ndcx > 1.5f || ndcy < -1.5f || ndcy > 1.5f)
            return false;
        out2D = glm::vec2(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
        return true;
    };

    // --- Project + per-point validity for the bin-target points ------
    std::vector<glm::vec2> pts_2D(boundary_points_3D.size());
    std::vector<bool>      pts_valid(boundary_points_3D.size(), false);
    for (size_t i = 0; i < boundary_points_3D.size(); i++) {
        if (project_2D_inframe(boundary_points_3D[i], pts_2D[i])) {
            pts_valid[i] = true;
        }
    }

    // --- Centroid: from full_boundary if provided, else from binned set
    glm::vec2 centroid_2D(0.0f);
    int n_valid_for_centroid = 0;
    if (full_boundary_3D_for_centroid != nullptr
        && !full_boundary_3D_for_centroid->empty())
    {
        glm::vec2 p2;
        for (const auto& p : *full_boundary_3D_for_centroid) {
            if (project_2D_inframe(p, p2)) {
                centroid_2D += p2;
                n_valid_for_centroid++;
            }
        }
        if (n_valid_for_centroid == 0) {
            for (size_t i = 0; i < pts_2D.size(); i++) {
                if (!pts_valid[i]) continue;
                centroid_2D += pts_2D[i];
                n_valid_for_centroid++;
            }
        }
    } else {
        for (size_t i = 0; i < pts_2D.size(); i++) {
            if (!pts_valid[i]) continue;
            centroid_2D += pts_2D[i];
            n_valid_for_centroid++;
        }
    }
    if (n_valid_for_centroid == 0) return;
    centroid_2D /= float(n_valid_for_centroid);

    const float a_lo = center_angle - half_range;
    const float subsec_size = (2.0f * half_range) / float(n_subsectors);
    std::vector<std::vector<size_t>> binIdx(n_subsectors);

    for (size_t i = 0; i < pts_2D.size(); i++) {
        if (!pts_valid[i]) continue;
        const glm::vec2 d = pts_2D[i] - centroid_2D;
        if (glm::length(d) < 1e-6f) continue;
        float theta = std::atan2(d.y, d.x);
        if (theta < 0.0f) theta += kTwoPi;
        float t = theta - a_lo;
        while (t < 0.0f)    t += kTwoPi;
        while (t >= kTwoPi) t -= kTwoPi;
        if (t > 2.0f * half_range) continue;
        int s = int(t / subsec_size);
        if (s >= n_subsectors) s = n_subsectors - 1;
        if (s < 0) s = 0;
        binIdx[s].push_back(i);
    }

    out_anchors_3D.reserve(n_subsectors);
    for (int s = 0; s < n_subsectors; s++) {
        if (binIdx[s].empty()) continue;
        glm::vec3 sec_c(0.0f);
        for (size_t idx : binIdx[s]) sec_c += boundary_points_3D[idx];
        sec_c /= float(binIdx[s].size());
        size_t best_idx = binIdx[s][0];
        float best_d = glm::length(boundary_points_3D[best_idx] - sec_c);
        for (size_t idx : binIdx[s]) {
            const float d = glm::length(boundary_points_3D[idx] - sec_c);
            if (d < best_d) { best_d = d; best_idx = idx; }
        }
        out_anchors_3D.push_back(boundary_points_3D[best_idx]);
    }
}

// ---------------------------------------------------------------------
// project3DAnchorsTo2D
// ---------------------------------------------------------------------
inline void project3DAnchorsTo2D(
    const std::vector<glm::vec3>& anchors_3D,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    std::vector<glm::vec2>& out_anchors_2D)
{
    out_anchors_2D.clear();
    out_anchors_2D.reserve(anchors_3D.size());
    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);
    for (const auto& p : anchors_3D) {
        const glm::vec4 clip = M * glm::vec4(p, 1.0f);
        if (std::abs(clip.w) < 1e-9f) {
            out_anchors_2D.emplace_back(-1e6f, -1e6f);
            continue;
        }
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        out_anchors_2D.emplace_back(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
    }
}

inline double evaluateContourSweepCandidate(
    const std::vector<glm::vec2>& tgt_anchors_2D,
    const std::vector<glm::vec3>& src_pivots_3D,
    int   target_anchor_idx,
    int   source_pivot_idx,
    float rotation_deg,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    const std::vector<float>& bdy_unsigned,
    int bdy_W, int bdy_H,
    glm::mat4& T_out)
{
    const int N_tgt = (int)tgt_anchors_2D.size();
    const int N_src = (int)src_pivots_3D.size();
    if (N_tgt < 1 || N_src < 1 ||
        target_anchor_idx < 0 || target_anchor_idx >= N_tgt ||
        source_pivot_idx  < 0 || source_pivot_idx  >= N_src) {
        T_out = glm::mat4(1.0f);
        return 1e18;
    }

    const glm::vec3 P_s = src_pivots_3D[source_pivot_idx];
    const glm::vec2 P_t_2D = tgt_anchors_2D[target_anchor_idx];
    const glm::vec3 P_t_3D =
        unprojectPixelAtWorldDepth(P_t_2D, P_s, view, proj, W, H);

    // Build T = Translate(P_t_3D) · R_z(θ) · Translate(-P_s)
    const float c = std::cos(glm::radians(rotation_deg));
    const float s = std::sin(glm::radians(rotation_deg));
    glm::mat4 R(1.0f);
    R[0][0] =  c; R[0][1] =  s; R[0][2] = 0.0f;
    R[1][0] = -s; R[1][1] =  c; R[1][2] = 0.0f;
    R[2][0] = 0.0f; R[2][1] = 0.0f; R[2][2] = 1.0f;
    const glm::mat4 T_to    = glm::translate(glm::mat4(1.0f),  P_t_3D);
    const glm::mat4 T_from  = glm::translate(glm::mat4(1.0f), -P_s);
    T_out = T_to * R * T_from;

    // Project transformed source pivots to 2D
    std::vector<glm::vec2> src2D(N_src);
    int n_in_frame = 0;
    const glm::mat4 M = proj * view;
    for (int i = 0; i < N_src; i++) {
        const glm::vec4 p4 = T_out * glm::vec4(src_pivots_3D[i], 1.0f);
        const glm::vec4 clip = M * p4;
        if (clip.w <= 1e-9f) { src2D[i] = glm::vec2(-1e6, -1e6); continue; }
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        if (ndcx < -1.0f || ndcx > 1.0f ||
            ndcy < -1.0f || ndcy > 1.0f) {
            src2D[i] = glm::vec2(-1e6, -1e6);
            continue;
        }
        src2D[i] = glm::vec2((ndcx + 1.0f) * 0.5f * float(bdy_W),
                             (1.0f - ndcy) * 0.5f * float(bdy_H));
        n_in_frame++;
    }
    if (n_in_frame < N_src / 2) {
        return 1e9 + double(N_src - n_in_frame);
    }

    // Forward chamfer: each src pivot → bdy distance
    const float kOOFPenalty = 1000.0f;
    double sum_fwd = 0.0;
    for (int i = 0; i < N_src; i++) {
        if (src2D[i].x < 0.0f) { sum_fwd += kOOFPenalty; continue; }
        int px = int(src2D[i].x), py = int(src2D[i].y);
        if (px < 0) px = 0; else if (px >= bdy_W) px = bdy_W - 1;
        if (py < 0) py = 0; else if (py >= bdy_H) py = bdy_H - 1;
        sum_fwd += double(bdy_unsigned[py * bdy_W + px]);
    }
    const double fwd = sum_fwd / double(N_src);

    // Backward chamfer: each target anchor → nearest transformed src pivot
    double sum_bwd = 0.0;
    for (int j = 0; j < N_tgt; j++) {
        float min_d2 = 1e18f;
        for (int i = 0; i < N_src; i++) {
            if (src2D[i].x < 0.0f) continue;
            const float dx = tgt_anchors_2D[j].x - src2D[i].x;
            const float dy = tgt_anchors_2D[j].y - src2D[i].y;
            const float d2 = dx * dx + dy * dy;
            if (d2 < min_d2) min_d2 = d2;
        }
        sum_bwd += double(std::sqrt(std::max(0.0f, min_d2)));
    }
    const double bwd = sum_bwd / double(N_tgt);

    return fwd + bwd;
}


// ---------------------------------------------------------------------
// ContourSweepState
//   State machine for the multi-frame Ctrl+Alt+W animation. Owned by
//   the caller (RegistrationActions/main.cpp); RimShapeMatch provides
//   only the per-candidate evaluator above. The state struct is here
//   for type unity.
// ---------------------------------------------------------------------
struct ContourSweepState {
    bool                       active = false;
    int                        phase = 0;             // 0=idle, 1=broad, 2=fine, 3=done
    int                        candidate_idx = 0;
    int                        total_candidates = 0;
    int                        candidates_per_frame = 100;
    int                        current_frame = 0;
    int                        total_frames_phase1 = 150;
    int                        total_frames_phase2 = 150;
    int                        n_target = 20;
    int                        n_source = 20;
    int                        n_rotation = 36;

    // Discretization caches (rebuilt per phase)
    std::vector<glm::vec2>     tgt_anchors_2D;
    std::vector<glm::vec3>     src_pivots_3D;

    // Best found this run
    int                        best_i_tgt = -1;
    int                        best_j_src = -1;
    float                      best_theta_deg = 0.0f;
    double                     best_cost = 1e18;
    glm::mat4                  best_T = glm::mat4(1.0f);

    // Phase 1 best (used to seed Phase 2 discretization)
    int                        phase1_best_i_tgt = -1;
    int                        phase1_best_j_src = -1;
    float                      phase1_best_theta_deg = 0.0f;

    // Endpoint-constraint state (Step 3c+): the source rim chain used
    // for sweep may be reversed from g_debugSourceRimChain when the
    // endpoint correspondence demands it, so Phase 2 must use this
    // same (possibly reversed) chain for subrange resampling.
    std::vector<int>           src_rim_chain_used;
    bool                       dir_reversed = false;
    bool                       endpoint_constraint_active = false;

    // -----------------------------------------------------------------
    // Phase 7b Step 3c++: Orientation lock + Plan A (anchor固定 + bbox tag)
    //
    //   This block replaces the earlier angle-based scheme. The two
    //   problems addressed:
    //
    //   (1) "Source を動かすと target anchor も動いた"
    //       Old design pre-filtered the boundary by source bbox then
    //       extracted sectors → centroid + bin + medoid ALL depended on
    //       the moving filter. Plan A passes the FULL boundary to the
    //       extractor (so anchors are completely target-only and
    //       invariant) and applies the bbox as a per-anchor TAG used
    //       only to skip candidates during sweep / dim-render during
    //       preview. Field: tgt_anchor_inside_bbox.
    //
    //   (2) "左右がひっくり返る (θ=210° best)"
    //       The angle-based index check was synthetic. Plan B uses the
    //       existing anatomical PCA labels:
    //         - src_pivot_lr_label[j]: PURE_RIGHT / PURE_LEFT / BOUNDARY
    //           inherited from LiverLeftRightLabel via the nearest
    //           rim_chain vertex of pivot j.
    //         - src_pivot_cc_label[j]: CRANIAL / CAUDAL from
    //           LiverCranioCaudalLabel.
    //         - tgt_anchor_screen_lr[i]: RIGHT / LEFT / AMBIGUOUS by
    //           anchor_2D.x vs locked_centroid_2D.x ± neutral band.
    //         - tgt_anchor_screen_cc[i]: CRANIAL / CAUDAL / AMBIGUOUS
    //           by anchor_2D.y vs centroid.y ± band.
    //       Mismatch (PURE_RIGHT ↔ SS_LR_LEFT etc.) → candidate skipped.
    //       BOUNDARY / AMBIGUOUS values always pass (this is where the
    //       "屈曲が強い垂直 rim" edge case is handled — those pivots
    //       are BOUNDARY by design of LiverLeftRightLabel).
    //
    //   orientation_lock_active = true iff
    //       g_shapeMatchSweepUseOrientationLock AND
    //       both g_liverLR and g_liverCC are valid at sweep start.
    //
    //   The independent θ_rotation magnitude cap (Check B) is unchanged
    //   and applies whenever orientation_lock_active.
    // -----------------------------------------------------------------
    std::vector<uint8_t>       src_pivot_lr_label;     // [N_source]  PURE_RIGHT/LEFT/BOUNDARY
    std::vector<uint8_t>       src_pivot_cc_label;     // [N_source]  CRANIAL/CAUDAL
    std::vector<uint8_t>       tgt_anchor_screen_lr;   // [N_target]  SS_LR_*
    std::vector<uint8_t>       tgt_anchor_screen_cc;   // [N_target]  SS_CC_*
    std::vector<uint8_t>       tgt_anchor_inside_bbox; // [N_target]  Plan A tag (1=inside)
    glm::vec2                  locked_centroid_2D = glm::vec2(0.0f);
    bool                       orientation_lock_active = false;

    // Sector-based target anchors (Step 3c+, Shift+P-style approach).
    // tgt_anchors_3D is the source of truth — derived from
    // g_debugTargetBoundaryPoints by polar-sector medoid extraction.
    // tgt_anchors_2D (already declared above) is the per-frame
    // projection of tgt_anchors_3D for cost computation. This avoids
    // re-projecting on every candidate evaluation while still using
    // the AR-camera 2D distance as the cost metric.
    std::vector<glm::vec3>     tgt_anchors_3D;
    // Center angle of best Phase 1 sector (for Phase 2 subrange)
    float                      phase1_best_center_angle = 0.0f;

    // Phase 2 search radii (in arc-length t param ∈ [0,1])
    float                      phase2_radius_target_t = 0.05f;   // 1/n_target
    float                      phase2_radius_source_t = 0.05f;
    float                      phase2_radius_rot_deg  = 10.0f;

    // Progress history for plotting
    std::vector<double>        cost_history;
};


// =====================================================================
// Phase 7b Step 3d: Silhouette 2D Dense Sweep
//   New, independent sweep path. Replaces the sector-based contour
//   sweep with a pure 2D dense chamfer over:
//     - source rim chain projected to image plane (~254 px curve)
//     - target rim 2D lower-half (~150 px curve, from
//       g_debugTargetContour2D filtered by centroid.y)
//   Pivot/anchor index is fixed-correspondence (pivot[i] ↔ anchor[i]).
//   Right-side start point for both curves is determined by:
//     - Source: 2D centroid of PURE_RIGHT-labeled mesh vertices
//     - Target: lower-half pixel with max x
//   The two curves are then re-oriented clockwise (right → bottom →
//   left → top) and arc-length resampled to 20 pivots/anchors.
//
//   No LR constraint, no bbox filter, no rot cap. The right-start
//   convention is what locks the orientation.
// =====================================================================

// ---------------------------------------------------------------------
// arclengthResample2D
//   Arc-length resample of a 2D open polyline to N evenly-spaced
//   points. Used to discretize both src_rim_2D (after re-orientation)
//   and tgt_lower_2D into 20 pivots/anchors.
//
//   If close_loop=true, the closing segment (last → first) is included
//   in the arc length; useful for the closed source rim. Default is
//   open (target lower-half is a polyline segment).
//
//   out_nearest_src_idx (optional): for each output sample, the index
//   in the INPUT curve of the polyline-segment endpoint closer to the
//   sample. Lets the caller carry parallel arrays (3D positions,
//   labels) without recomputing arc length.
// ---------------------------------------------------------------------
inline void arclengthResample2D(
    const std::vector<glm::vec2>& curve,
    int N,
    std::vector<glm::vec2>& resampled_out,
    bool close_loop = false,
    std::vector<int>* out_nearest_src_idx = nullptr)
{
    resampled_out.clear();
    if (out_nearest_src_idx) out_nearest_src_idx->clear();
    const int K = (int)curve.size();
    if (K < 2 || N < 1) return;

    // Build extended polyline (with optional closure)
    std::vector<glm::vec2> pts;
    std::vector<int>       pts_orig_idx;
    pts.reserve(K + (close_loop ? 1 : 0));
    pts_orig_idx.reserve(K + (close_loop ? 1 : 0));
    for (int i = 0; i < K; i++) {
        pts.push_back(curve[i]);
        pts_orig_idx.push_back(i);
    }
    if (close_loop) {
        pts.push_back(curve[0]);
        pts_orig_idx.push_back(0);
    }

    // Cumulative arc length
    std::vector<float> cum(pts.size(), 0.0f);
    for (size_t i = 1; i < pts.size(); i++) {
        cum[i] = cum[i-1] + glm::length(pts[i] - pts[i-1]);
    }
    const float total_len = cum.back();
    if (total_len < 1e-6f) return;

    resampled_out.reserve(N);
    if (out_nearest_src_idx) out_nearest_src_idx->reserve(N);
    for (int k = 0; k < N; k++) {
        const float t = (float(k) + 0.5f) * total_len / float(N);
        auto it = std::upper_bound(cum.begin(), cum.end(), t);
        if (it == cum.begin()) {
            resampled_out.push_back(pts.front());
            if (out_nearest_src_idx) out_nearest_src_idx->push_back(pts_orig_idx.front());
            continue;
        }
        if (it == cum.end()) {
            resampled_out.push_back(pts.back());
            if (out_nearest_src_idx) out_nearest_src_idx->push_back(pts_orig_idx.back());
            continue;
        }
        const size_t idx     = std::distance(cum.begin(), it);
        const float  seg_len = cum[idx] - cum[idx-1];
        const float  alpha   = (seg_len > 1e-6f)
                                   ? (t - cum[idx-1]) / seg_len
                                   : 0.0f;
        resampled_out.push_back(glm::mix(pts[idx-1], pts[idx], alpha));
        if (out_nearest_src_idx) {
            const size_t closer = (alpha < 0.5f) ? (idx - 1) : idx;
            out_nearest_src_idx->push_back(pts_orig_idx[closer]);
        }
    }
}


// ---------------------------------------------------------------------
// project_RimChainTo2D
//   Project each mesh vertex referenced by rim_chain to image-plane
//   pixel coords using (proj * view). Pixels with clip.w <= 0 are
//   emitted as (-1e6, -1e6) sentinels; callers should filter these.
//   Returns the same number of points as rim_chain.size().
// ---------------------------------------------------------------------
inline void project_RimChainTo2D(
    const std::vector<int>& rim_chain,
    const std::vector<float>& mesh_verts,      // flat x,y,z,...
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    std::vector<glm::vec2>& out_curve_2D)
{
    out_curve_2D.clear();
    const int K  = (int)rim_chain.size();
    const int nV3 = (int)mesh_verts.size();
    if (K < 1) return;
    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);
    out_curve_2D.reserve(K);
    for (int idx : rim_chain) {
        if (idx < 0 || idx*3 + 2 >= nV3) {
            out_curve_2D.emplace_back(-1e6f, -1e6f);
            continue;
        }
        const glm::vec3 p3(mesh_verts[idx*3],
                           mesh_verts[idx*3+1],
                           mesh_verts[idx*3+2]);
        const glm::vec4 clip = M * glm::vec4(p3, 1.0f);
        if (clip.w <= 1e-9f) {
            out_curve_2D.emplace_back(-1e6f, -1e6f);
            continue;
        }
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        out_curve_2D.emplace_back(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
    }
}


// ---------------------------------------------------------------------
// findRightStartIndex
//   For the source rim chain: compute the 3D centroid of all vertices
//   labeled PURE_RIGHT (LiverLeftRightLabel), project that centroid
//   to 2D, then return the chain index whose 2D position is closest.
//
//   Returns -1 if there are no PURE_RIGHT vertices, or if rim_chain
//   is empty / all sentinels.
// ---------------------------------------------------------------------
inline int findRightStartIndex(
    const std::vector<glm::vec2>& src_rim_2D,
    const std::vector<int>& rim_chain,
    const std::vector<uint8_t>& lr_labels,    // size = nV
    const std::vector<float>& mesh_verts,     // flat x,y,z,...
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    glm::vec2* out_right_centroid_2D = nullptr)
{
    if (src_rim_2D.empty()) return -1;
    if (src_rim_2D.size() != rim_chain.size()) return -1;
    if (lr_labels.empty() || mesh_verts.empty()) return -1;

    // Accumulate PURE_RIGHT 3D centroid (mesh-space). We use a 3D
    // centroid then project once, to avoid sentinels biasing the mean.
    const int nV  = (int)lr_labels.size();
    const int nV3 = (int)mesh_verts.size();
    glm::dvec3 sum3(0.0);
    int        n_R = 0;
    for (int i = 0; i < nV; i++) {
        if (i*3 + 2 >= nV3) break;
        if (lr_labels[i] != LiverLeftRightLabel::PURE_RIGHT) continue;
        sum3 += glm::dvec3(mesh_verts[i*3],
                           mesh_verts[i*3+1],
                           mesh_verts[i*3+2]);
        n_R++;
    }
    if (n_R == 0) return -1;
    const glm::vec3 right_centroid_3D(sum3 / double(n_R));

    // Project to 2D
    const glm::mat4 M = proj * view;
    const glm::vec4 clip = M * glm::vec4(right_centroid_3D, 1.0f);
    if (clip.w <= 1e-9f) return -1;
    const float ndcx = clip.x / clip.w;
    const float ndcy = clip.y / clip.w;
    const glm::vec2 c2D(
        (ndcx + 1.0f) * 0.5f * float(W),
        (1.0f - ndcy) * 0.5f * float(H));
    if (out_right_centroid_2D) *out_right_centroid_2D = c2D;

    // Find nearest chain index (skip sentinels)
    int   best_idx = -1;
    float best_d2  = std::numeric_limits<float>::infinity();
    for (int i = 0; i < (int)src_rim_2D.size(); i++) {
        if (src_rim_2D[i].x < -1e5f) continue;     // sentinel
        const float dx = src_rim_2D[i].x - c2D.x;
        const float dy = src_rim_2D[i].y - c2D.y;
        const float d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best_idx = i; }
    }
    return best_idx;
}


// ---------------------------------------------------------------------
// rotateAndOrientCurve
//   Given a closed-loop 2D curve and a start index, return a re-ordered
//   curve that begins at start_idx and proceeds in a chosen direction.
//
//   clockwise_on_screen:
//     true → the next step from start_idx should move predominantly
//            downward (+y on screen). If not, the chain is reversed
//            (in-place, keeping start_idx at position 0).
//
//   Outputs (must be different vectors from `curve_2D`):
//     out_curve              : re-ordered curve, size K
//     parallel_index_array   : if non-null, parallel array where
//                              element[i] = original index of out_curve[i]
//                              in curve_2D (lets caller reorder parallel
//                              data like rim chain vertex indices).
// ---------------------------------------------------------------------
inline void rotateAndOrientCurve(
    const std::vector<glm::vec2>& curve_2D,
    int start_idx,
    bool clockwise_on_screen,
    std::vector<glm::vec2>& out_curve,
    std::vector<int>* parallel_index_array = nullptr)
{
    out_curve.clear();
    if (parallel_index_array) parallel_index_array->clear();
    const int K = (int)curve_2D.size();
    if (K < 2 || start_idx < 0 || start_idx >= K) {
        out_curve = curve_2D;
        if (parallel_index_array) {
            parallel_index_array->resize(K);
            for (int i = 0; i < K; i++) (*parallel_index_array)[i] = i;
        }
        return;
    }

    // Step 1: rotate so start_idx becomes index 0
    std::vector<glm::vec2> rot(K);
    std::vector<int>       rot_orig(K);
    for (int i = 0; i < K; i++) {
        const int j = (start_idx + i) % K;
        rot[i] = curve_2D[j];
        rot_orig[i] = j;
    }

    // Step 2: decide direction (only if clockwise requested).
    if (clockwise_on_screen) {
        // Look at next ~10 points and pick a non-degenerate candidate
        glm::vec2 p0 = rot[0];
        glm::vec2 p1(0.0f);
        bool found = false;
        if (p0.x >= -1e5f) {
            for (int s = 1; s < std::min(K, 10); s++) {
                if (rot[s].x < -1e5f) continue;
                const float dx = rot[s].x - p0.x;
                const float dy = rot[s].y - p0.y;
                if (dx*dx + dy*dy > 1.0f) {   // > 1 px movement
                    p1 = rot[s];
                    found = true;
                    break;
                }
            }
        }
        if (found) {
            // We start at the right-side of the silhouette. To proceed
            // clockwise (right → bottom → left → top), the *next* step
            // should move predominantly downward (+y on screen). If
            // instead it moves upward (CCW), reverse.
            if (p1.y < p0.y) {
                // Reverse keeping index 0 fixed: 0, K-1, K-2, ..., 1
                std::vector<glm::vec2> rev(K);
                std::vector<int>       rev_orig(K);
                rev[0]      = rot[0];
                rev_orig[0] = rot_orig[0];
                for (int i = 1; i < K; i++) {
                    rev[i]      = rot[K - i];
                    rev_orig[i] = rot_orig[K - i];
                }
                rot.swap(rev);
                rot_orig.swap(rev_orig);
            }
        }
    }

    out_curve = std::move(rot);
    if (parallel_index_array) *parallel_index_array = std::move(rot_orig);
}


// ---------------------------------------------------------------------
// extractLowerHalfRimCurve
//   From the (open) target contour 2D polyline, extract the longest
//   maximal subsequence of consecutive points satisfying p.y >
//   centroid_y. The target contour from traceContour2D is the longest
//   connected boundary segment; on a 1920x1080 image, screen +y points
//   downward, so "lower half" = pixel.y > centroid.y.
//
//   If multiple disjoint runs satisfy the predicate, the longest one
//   (in point count) is returned.
// ---------------------------------------------------------------------
inline void extractLowerHalfRimCurve(
    const std::vector<glm::vec2>& contour_2D,
    float centroid_y,
    std::vector<glm::vec2>& out_lower)
{
    out_lower.clear();
    const int K = (int)contour_2D.size();
    if (K < 2) return;

    // Find maximal runs of consecutive indices where y > centroid_y.
    int best_s = -1, best_e = -1;     // inclusive [s, e]
    int cur_s = -1;
    for (int i = 0; i < K; i++) {
        const bool in_lower = (contour_2D[i].y > centroid_y);
        if (in_lower) {
            if (cur_s < 0) cur_s = i;
            const int cur_e = i;
            if ((cur_e - cur_s) > (best_e - best_s)) {
                best_s = cur_s;
                best_e = cur_e;
            }
        } else {
            cur_s = -1;
        }
    }
    if (best_s < 0) return;

    out_lower.reserve(best_e - best_s + 1);
    for (int i = best_s; i <= best_e; i++) out_lower.push_back(contour_2D[i]);
}


// ---------------------------------------------------------------------
// orientTargetLowerByRightStart
//   Given the target lower-half 2D curve (open polyline), pick the
//   endpoint with greater x as the "right start" and reverse if
//   needed so output[0] is the right end.
// ---------------------------------------------------------------------
inline void orientTargetLowerByRightStart(
    const std::vector<glm::vec2>& lower_in,
    std::vector<glm::vec2>& out_lower)
{
    out_lower.clear();
    const int K = (int)lower_in.size();
    if (K < 2) { out_lower = lower_in; return; }
    if (lower_in.front().x >= lower_in.back().x) {
        out_lower = lower_in;
    } else {
        out_lower.reserve(K);
        for (int i = K - 1; i >= 0; i--) out_lower.push_back(lower_in[i]);
    }
}


// ---------------------------------------------------------------------
// denseChamfer2D
//   Bidirectional dense chamfer distance between two 2D curves.
//   Returns mean(A→B) + mean(B→A). Sentinels (x < -1e5) are skipped.
//   Brute O(N*M). For N=254, M=150 that is ~38k distances; 720
//   candidates total ~27M ops — typically well under 1 sec.
// ---------------------------------------------------------------------
inline double denseChamfer2D(
    const std::vector<glm::vec2>& A,
    const std::vector<glm::vec2>& B)
{
    if (A.empty() || B.empty()) return 1e18;

    double sum_ab = 0.0;
    int    n_a    = 0;
    for (const auto& a : A) {
        if (a.x < -1e5f) continue;
        float dmin = std::numeric_limits<float>::infinity();
        for (const auto& b : B) {
            if (b.x < -1e5f) continue;
            const float dx = a.x - b.x;
            const float dy = a.y - b.y;
            const float d2 = dx*dx + dy*dy;
            if (d2 < dmin) dmin = d2;
        }
        if (std::isfinite(dmin)) { sum_ab += std::sqrt(dmin); n_a++; }
    }
    if (n_a == 0) return 1e18;

    double sum_ba = 0.0;
    int    n_b    = 0;
    for (const auto& b : B) {
        if (b.x < -1e5f) continue;
        float dmin = std::numeric_limits<float>::infinity();
        for (const auto& a : A) {
            if (a.x < -1e5f) continue;
            const float dx = a.x - b.x;
            const float dy = a.y - b.y;
            const float d2 = dx*dx + dy*dy;
            if (d2 < dmin) dmin = d2;
        }
        if (std::isfinite(dmin)) { sum_ba += std::sqrt(dmin); n_b++; }
    }
    if (n_b == 0) return 1e18;

    return (sum_ab / double(n_a)) + (sum_ba / double(n_b));
}


// ---------------------------------------------------------------------
// evaluateSilhouetteSweepCandidate
//   For one (i, θ) candidate:
//     1. Pin src_pivots_3D[i] to tgt_anchors_2D[i] at the source's
//        current depth (camera ray + unproject), giving translation.
//     2. Rotate the source by θ around the camera axis through the
//        pinned source pivot point.
//     3. Project the entire transformed source rim curve (3D) to 2D.
//     4. Return dense bidirectional 2D chamfer vs tgt_lower_2D.
//
//   The transform T is also written to T_out so the caller can adopt
//   it on finish.
// ---------------------------------------------------------------------
inline double evaluateSilhouetteSweepCandidate(
    const std::vector<glm::vec3>& src_rim_3D_dense,   // full chain in 3D (mesh-space)
    const std::vector<glm::vec3>& src_pivots_3D,      // 20 pivots in 3D
    const std::vector<glm::vec2>& tgt_anchors_2D,     // 20 anchors in 2D
    const std::vector<glm::vec2>& tgt_lower_2D,       // dense target lower-half
    int   pivot_idx,
    float rotation_deg,
    const glm::mat4& view, const glm::mat4& proj,
    int W, int H,
    glm::mat4& T_out)
{
    const int N_pivot = (int)src_pivots_3D.size();
    if (pivot_idx < 0 || pivot_idx >= N_pivot ||
        pivot_idx >= (int)tgt_anchors_2D.size() ||
        src_rim_3D_dense.empty() || tgt_lower_2D.empty())
    {
        T_out = glm::mat4(1.0f);
        return 1e18;
    }

    const glm::vec3 P_s    = src_pivots_3D[pivot_idx];
    const glm::vec2 P_t_2D = tgt_anchors_2D[pivot_idx];
    const glm::vec3 P_t_3D =
        unprojectPixelAtWorldDepth(P_t_2D, P_s, view, proj, W, H);

    // T = Translate(P_t_3D) · R_z(θ) · Translate(-P_s)
    //   R_z = rotation around world-Z (camera axis in this AR setup).
    //   Matches evaluateContourSweepCandidate exactly.
    const float c = std::cos(glm::radians(rotation_deg));
    const float s = std::sin(glm::radians(rotation_deg));
    glm::mat4 R(1.0f);
    R[0][0] =  c; R[0][1] =  s; R[0][2] = 0.0f;
    R[1][0] = -s; R[1][1] =  c; R[1][2] = 0.0f;
    R[2][0] = 0.0f; R[2][1] = 0.0f; R[2][2] = 1.0f;
    const glm::mat4 T_to    = glm::translate(glm::mat4(1.0f),  P_t_3D);
    const glm::mat4 T_from  = glm::translate(glm::mat4(1.0f), -P_s);
    T_out = T_to * R * T_from;

    // Project transformed dense src rim → 2D
    const glm::mat4 M = proj * view;
    const float W_f = float(W);
    const float H_f = float(H);
    std::vector<glm::vec2> src_2D;
    src_2D.reserve(src_rim_3D_dense.size());
    for (const auto& p : src_rim_3D_dense) {
        const glm::vec4 p4 = T_out * glm::vec4(p, 1.0f);
        const glm::vec4 cp = M * p4;
        if (cp.w <= 1e-9f) { src_2D.emplace_back(-1e6f, -1e6f); continue; }
        const float ndcx = cp.x / cp.w;
        const float ndcy = cp.y / cp.w;
        src_2D.emplace_back(
            (ndcx + 1.0f) * 0.5f * W_f,
            (1.0f - ndcy) * 0.5f * H_f);
    }

    return denseChamfer2D(src_2D, tgt_lower_2D);
}


// ---------------------------------------------------------------------
// SilhouetteSweepState
//   State machine for the multi-frame Ctrl+Alt+W Step 3d sweep.
//   Independent from ContourSweepState (Step 3c) — the two coexist so
//   the old method remains the default fallback when the new toggle
//   is off.
// ---------------------------------------------------------------------
struct SilhouetteSweepState {
    bool                       active = false;
    int                        phase  = 0;   // 0=idle,1=broad,2=fine,3=done

    // Frame-pacing controls (mirror ContourSweepState)
    int                        candidate_idx        = 0;
    int                        total_candidates     = 0;
    int                        candidates_per_frame = 60;
    int                        current_frame        = 0;
    int                        total_frames_phase1  = 60;
    int                        total_frames_phase2  = 30;

    // Discretization sizes (fixed per spec)
    int                        n_pivot     = 20;  // = n_anchor
    int                        n_rotation  = 36;  // Phase 1: 0..350° step 10°

    // ---- Phase 0 caches (set in startSilhouetteSweep) ----
    //
    // Source side (dense + sampled, indexed in the canonical
    // right-start clockwise order):
    std::vector<glm::vec2>     src_rim_2D;          // dense, projected (~254 pts)
    std::vector<glm::vec3>     src_rim_3D_oriented; // dense, mesh-space, reordered
    std::vector<int>           src_rim_orig_idx;    // parallel: index in original chain
    std::vector<glm::vec2>     src_pivots_2D;       // 20 pivots in 2D (debug/draw)
    std::vector<glm::vec3>     src_pivots_3D;       // 20 pivots in 3D (mesh-space)
    std::vector<uint8_t>       src_pivots_lr_label; // 20 pivots LR label (debug)
    glm::vec2                  src_right_centroid_2D = glm::vec2(0.0f);
    int                        src_start_idx_in_chain = -1;   // index into original chain
    bool                       src_dir_reversed = false;

    // Target side:
    std::vector<glm::vec2>     tgt_lower_2D;        // dense lower-half (~150 pts)
    std::vector<glm::vec2>     tgt_anchors_2D;      // 20 anchors in 2D
    glm::vec2                  tgt_centroid_2D = glm::vec2(0.0f);

    // ---- Phase 1 / 2 results ----
    int                        best_i_pivot   = -1;
    float                      best_theta_deg = 0.0f;
    double                     best_cost      = 1e18;
    glm::mat4                  best_T         = glm::mat4(1.0f);

    int                        phase1_best_i_pivot   = -1;
    float                      phase1_best_theta_deg = 0.0f;

    // Phase 2 search ranges
    int                        phase2_pivot_radius = 2;     // ±2 around best
    float                      phase2_theta_radius_deg = 10.0f;

    // History (for ImGui PlotLines)
    std::vector<double>        cost_history;

    // Failure capture for UI
    std::string                fail_reason;
};


} // namespace RimShape

#endif // RIM_SHAPE_MATCH_H
