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

#include <Eigen/Dense>
#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "LiverRegionLabel.h"

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

} // namespace RimShape

#endif // RIM_SHAPE_MATCH_H
