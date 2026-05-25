#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>       // [Phase 0/3] std::clamp for anchorBlend clamping
#include <limits>          // [Phase 0/3] std::numeric_limits for anchor NaN sentinel
#include <unordered_map>   // [Phase 0/3] full-idx -> compact-idx inverse map

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "NoOpen3DRegistration.h"
#include "mCutMesh.h"

namespace NormalRefine {

using Reg3DCustom::PointCloud;
using Reg3DCustom::NoOpen3DRegistration;
using Reg3DCustom::NanoflannAdaptorD;
using Reg3DCustom::KDTree3DD;
using Reg3DCustom::buildKDTreeD;
using Reg3DCustom::searchKNN1D;

// ============================================================
//  Ablation: two refinement methods
//    Key N) NORMAL_COMPAT  = Normal-Compatible Weighted ICP
//           weight = sigmoid(dist) × |dot(n_src, n_tgt)|
//    Key B) SRT_VARIANCE   = KDTree ICP + SRT variance weight
//           correspondence: same KDTree + normal filter as Key N
//           weight = 1/stddev from normal-line proximity distribution
// ============================================================
enum RefineMethod {
    NORMAL_COMPAT,
    SRT_VARIANCE
};

inline const char* methodTag(RefineMethod m) {
    return (m == SRT_VARIANCE) ? "[SRT-V]" : "[Refine]";
}

inline const char* methodName(RefineMethod m) {
    return (m == SRT_VARIANCE) ? "SRT Variance-Weighted" : "Normal-Compatible";
}

struct RefineParams {
    float distanceThreshold   = 0.15f;
    float minNormalAngleCos   = 0.30f;
    float maxNormalAngleCos   = 0.90f;
    float sigmaSoftDistStart  = 0.08f;
    float sigmaSoftDistEnd    = 0.03f;
    float tikhonovLambda      = 1e-3f;
    int   itersPerFrame       = 2;
    int   annealEndIteration  = 60;
    float convergenceRMSE     = 1e-6f;
    int   convergenceWindow   = 5;
    int   maxTotalIterations  = 200;

    // --- SRT variance params (Key B only) ---
    int   nSamples            = 11;
    float sampleRange         = 0.10f;
    float srtSlope            = 0.02f;
    float srtMinVariance      = 0.5f;

    // --- Boundary Z-weight params ---
    bool  useZWeight          = false;
    float zWeightBoundary     = 0.05f;
    float zWeightInterior     = 0.30f;
    float boundaryWidth       = 8.0f;
    float boundaryBoost       = 3.0f;

    // ============================================================
    //  [Phase 0/2] L1: RIM-weighted multiplicative boost
    //    The per-correspondence weight gets multiplied by
    //      (1 + betaRimSrc * isRimSrc[i]) * (1 + betaRimTgt * isRimTgt[nnIdx])
    //    When both beta are 0.0 (default), the weight is unchanged →
    //    byte-identical to the pre-Phase-2 behaviour.
    //    Source rim mask is populated from LiverRegionLabel::RIM in
    //    the wrapper; target rim mask is populated from
    //    boundaryDist < rim_tgt_thresh_px.
    // ============================================================
    float betaRimSrc          = 0.0f;
    float betaRimTgt          = 0.0f;

    // ============================================================
    //  [Phase 0/3] L2: Anchor-pair initialisation
    //    During the first anchorPhaseIter iterations, source vertices
    //    that have an anchor pair (= populated from the previous
    //    Ctrl+G's g_lastRimPair*) use the anchor target position
    //    instead of the runtime KDTree nearest neighbour. After
    //    anchorPhaseIter iterations, the anchor is dropped and the
    //    method falls through to the standard NN behaviour.
    //
    //    anchorBlend lets the wrapper soft-mix anchor and NN inside
    //    the anchor phase: blend=1.0 (default) → pure anchor;
    //    blend=0.0 → ignore anchors entirely. Useful for ablation.
    //
    //    Vertices WITHOUT an anchor (= non-rim) use NN both phases.
    //
    //    All three knobs default to a no-op (useAnchorPairs=false),
    //    so the byte-identical contract is preserved.
    // ============================================================
    bool  useAnchorPairs      = false;
    int   anchorPhaseIter     = 20;
    float anchorBlend         = 1.0f;

    // ============================================================
    //  [Phase 7a] Pure RIM target filter
    //    When true, refineStep DROPS correspondences whose target NN
    //    is not a rim point (isRimTgt[nnIdx] == 0). This makes the
    //    refinement use ONLY rim-to-rim matches, complementing the
    //    source-side rim filter in the wrapper.
    //
    //    Requires isRimTgt to be populated (state.isRimTgt non-empty
    //    and sized == target cloud). When the mask isn't available,
    //    the flag is silently ignored and full-target refinement
    //    resumes — same as Phase 2 with betaRimTgt = 0.
    //
    //    Default false → byte-identical to Phase 6 behaviour.
    // ============================================================
    bool  pureRimTarget       = false;
};

struct RefineStepResult {
    glm::dmat4 incrementalTransform = glm::dmat4(1.0);
    float rmse                = 0.0f;
    int   correspondenceCount = 0;
    bool  converged           = false;
};

struct RefineState {
    bool active = false;
    RefineMethod method = NORMAL_COMPAT;
    int  totalIterations = 0;
    int  convergeCount   = 0;
    float prevRMSE       = 1e10f;
    float currentRMSE    = 0.0f;

    std::shared_ptr<PointCloud> sourceCloud;
    std::shared_ptr<PointCloud> targetCloud;
    std::vector<glm::dvec3>     targetPtsD;
    std::unique_ptr<NanoflannAdaptorD> targetAdaptor;
    std::unique_ptr<KDTree3DD>         targetTree;
    std::vector<glm::dvec3>     sourcePtsD;
    std::vector<glm::dvec3>     sourceNormalsD;

    std::vector<mCutMesh*> organMeshes;
    RefineParams params;

    // ============================================================
    //  [Phase 0/2/3] Per-source-vertex auxiliary arrays
    //    All four are populated by initRefine() in the same loop
    //    that fills sourcePtsD / sourceNormalsD, so their indexing
    //    is identical (i = compact source index in [0, sourcePtsD.size())).
    //
    //    isRimSrc[i]            : 1 if the source vertex is on the
    //                             mesh-intrinsic RIM band (Phase 2 L1).
    //                             Size 0 means "no rim info" (legacy
    //                             behaviour, beta_src treated as 0).
    //    isRimTgt[j]            : per-TARGET point flag, populated
    //                             from rim_tgt_mask. Size 0 means
    //                             "no rim info" (beta_tgt -> 0).
    //                             Indexed by KDTree nnIdx, NOT i.
    //    subsetToFullVertIdx[i] : maps compact source index back to
    //                             liverMesh3D full-mesh vertex index.
    //                             Needed to look up anchor pairs
    //                             by full-mesh index (Phase 3 L2).
    //    anchorTgtPos[i]        : per-source-vertex anchor target.
    //                             glm::vec3(NAN) sentinel means "no
    //                             anchor for this vertex" (= not rim,
    //                             or the rim vertex was not in the
    //                             last Ctrl+G pair set). All NaN
    //                             means anchor mode is fully inactive
    //                             this session.
    // ============================================================
    std::vector<uint8_t>   isRimSrc;
    std::vector<uint8_t>   isRimTgt;
    std::vector<int>       subsetToFullVertIdx;
    std::vector<glm::dvec3> anchorTgtPos;     // dvec to avoid float drift in JTJ
    int                    anchorCount = 0;   // count of vertices that DO have anchor

    // --- Best state tracking ---
    float initialRMSE    = 1e10f;
    float bestRMSE       = 1e10f;
    int   bestIteration  = 0;
    int   worseCount     = 0;
    glm::dmat4 cumulativeTransform = glm::dmat4(1.0);
    glm::dmat4 bestCumulativeTransform = glm::dmat4(1.0);

    // --- Initial mesh backup (for revert) ---
    struct MeshBackup {
        std::vector<float> vertices;
        std::vector<float> normals;
    };
    std::vector<MeshBackup> initialMeshBackup;

    void reset() {
        active = false;
        method = NORMAL_COMPAT;
        totalIterations = 0;
        convergeCount = 0;
        prevRMSE = 1e10f;
        currentRMSE = 0.0f;
        sourceCloud.reset();
        targetCloud.reset();
        targetPtsD.clear();
        targetAdaptor.reset();
        targetTree.reset();
        sourcePtsD.clear();
        sourceNormalsD.clear();
        organMeshes.clear();
        // [Phase 0/2/3] reset new aux arrays
        isRimSrc.clear();
        isRimTgt.clear();
        subsetToFullVertIdx.clear();
        anchorTgtPos.clear();
        anchorCount = 0;
        initialRMSE = 1e10f;
        bestRMSE = 1e10f;
        bestIteration = 0;
        worseCount = 0;
        cumulativeTransform = glm::dmat4(1.0);
        bestCumulativeTransform = glm::dmat4(1.0);
        initialMeshBackup.clear();
    }

    void backupMeshes() {
        initialMeshBackup.resize(organMeshes.size());
        for (size_t i = 0; i < organMeshes.size(); i++) {
            initialMeshBackup[i].vertices = organMeshes[i]->mVertices;
            initialMeshBackup[i].normals  = organMeshes[i]->mNormals;
        }
    }

    void restoreMeshes() {
        for (size_t i = 0; i < organMeshes.size() && i < initialMeshBackup.size(); i++) {
            organMeshes[i]->mVertices = initialMeshBackup[i].vertices;
            organMeshes[i]->mNormals  = initialMeshBackup[i].normals;
            setUp(*organMeshes[i]);
        }
    }
};

// ============================================================
//  Math utilities
// ============================================================

inline bool solve6x6(double A[6][6], double b[6], double x[6]) {
    const int N = 6;
    double L[N][N] = {};
    double D[N] = {};
    for (int j = 0; j < N; j++) {
        double sum = A[j][j];
        for (int k = 0; k < j; k++) sum -= L[j][k] * L[j][k] * D[k];
        D[j] = sum;
        L[j][j] = 1.0;
        if (std::abs(D[j]) < 1e-20) {
            D[j] = 0.0;
            for (int i = j + 1; i < N; i++) L[i][j] = 0.0;
        } else {
            for (int i = j + 1; i < N; i++) {
                double s = A[i][j];
                for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k] * D[k];
                L[i][j] = s / D[j];
            }
        }
    }
    double y[N];
    for (int i = 0; i < N; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i][k] * y[k];
        y[i] = s;
    }
    double z[N];
    for (int i = 0; i < N; i++) {
        z[i] = (std::abs(D[i]) < 1e-20) ? 0.0 : y[i] / D[i];
    }
    for (int i = N - 1; i >= 0; i--) {
        double s = z[i];
        for (int k = i + 1; k < N; k++) s -= L[k][i] * x[k];
        x[i] = s;
    }
    return true;
}

inline glm::dmat4 transformVector6dToMatrix4d(const double result[6]) {
    double alpha = result[0], beta = result[1], gamma = result[2];
    double ca = std::cos(alpha), sa = std::sin(alpha);
    double cb = std::cos(beta),  sb = std::sin(beta);
    double cg = std::cos(gamma), sg = std::sin(gamma);
    glm::dmat4 mat(1.0);
    mat[0][0] = cg*cb;  mat[0][1] = sg*cb;  mat[0][2] = -sb;
    mat[1][0] = cg*sb*sa - sg*ca; mat[1][1] = sg*sb*sa + cg*ca; mat[1][2] = cb*sa;
    mat[2][0] = cg*sb*ca + sg*sa; mat[2][1] = sg*sb*ca - cg*sa; mat[2][2] = cb*ca;
    mat[3][0] = result[3]; mat[3][1] = result[4]; mat[3][2] = result[5]; mat[3][3] = 1.0;
    return mat;
}

inline float anneal(float startVal, float endVal, int iteration, int endIter) {
    float t = std::min(1.0f, static_cast<float>(iteration) / static_cast<float>(std::max(1, endIter)));
    return startVal + (endVal - startVal) * t;
}

inline float sigmoidWeight(float dist, float threshold, float sigma) {
    return 1.0f / (1.0f + std::exp((dist - threshold) / std::max(sigma, 1e-8f)));
}

// ============================================================
//  Shared: solve 6DoF, update source, check convergence
// ============================================================

inline bool solvePoseAndUpdate(
    double JTJ[6][6], double JTr_vec[6],
    RefineState& state, RefineStepResult& result,
    int corrCount, double totalError2)
{
    for (int a = 0; a < 6; a++)
        JTJ[a][a] += state.params.tikhonovLambda;

    double neg_JTr[6];
    for (int a = 0; a < 6; a++) neg_JTr[a] = -JTr_vec[a];

    double x[6] = {};
    if (!solve6x6(JTJ, neg_JTr, x)) return false;

    glm::dmat4 update = transformVector6dToMatrix4d(x);
    result.incrementalTransform = update * result.incrementalTransform;

    glm::dmat3 rotUpdate = glm::dmat3(update);
    for (size_t i = 0; i < state.sourcePtsD.size(); i++) {
        glm::dvec4 p4(state.sourcePtsD[i], 1.0);
        state.sourcePtsD[i] = glm::dvec3(update * p4);
        state.sourceNormalsD[i] = glm::normalize(rotUpdate * state.sourceNormalsD[i]);
    }

    result.correspondenceCount = corrCount;
    result.rmse = static_cast<float>(
        std::sqrt(totalError2 / std::max(1.0, static_cast<double>(corrCount))));
    state.totalIterations++;
    return true;
}

inline void checkConvergence(RefineState& state, RefineStepResult& result) {
    state.currentRMSE = result.rmse;
    if (std::abs(state.prevRMSE - state.currentRMSE) < state.params.convergenceRMSE) {
        state.convergeCount++;
    } else {
        state.convergeCount = 0;
    }
    state.prevRMSE = state.currentRMSE;

    if (state.convergeCount >= state.params.convergenceWindow ||
        state.totalIterations >= state.params.maxTotalIterations) {
        result.converged = true;
    }
}

// ============================================================
//  SRT variance computation along source normal
//  Samples N points along normal, measures target proximity,
//  returns variance of the resulting distribution.
//  Low variance = target surface clearly present = high confidence.
// ============================================================
inline double computeSRTVariance(
    const glm::dvec3& center,
    const glm::dvec3& normal,
    const KDTree3DD& tree,
    int N, double range, double slope, double thresh, double minVar)
{
    double halfN = static_cast<double>(N - 1) * 0.5;
    double proxySum = 0.0;

    std::vector<double> proximity(N);
    for (int k = 0; k < N; k++) {
        double t = (static_cast<double>(k) - halfN) / halfN * range;
        glm::dvec3 samplePt = center + t * normal;

        size_t nnIdx; double nnDistSq;
        searchKNN1D(tree, samplePt, nnIdx, nnDistSq);
        double dist = std::sqrt(nnDistSq);

        proximity[k] = 0.5 + 0.5 * std::tanh((thresh - dist) / (2.0 * slope));
        proxySum += proximity[k];
    }

    if (proxySum < 1e-8) return minVar * 10.0;

    for (int k = 0; k < N; k++) proximity[k] /= proxySum;

    double meanIdx = 0.0;
    for (int k = 0; k < N; k++) meanIdx += static_cast<double>(k) * proximity[k];

    double variance = 0.0;
    for (int k = 0; k < N; k++) {
        double diff = static_cast<double>(k) - meanIdx;
        variance += diff * diff * proximity[k];
    }

    return std::max(variance, static_cast<double>(minVar));
}

// ============================================================
//  Shared initialization (same for both methods)
//
//  [Phase 0/2/3] Optional rim / anchor inputs (default empty):
//    rim_src_mask          : per-full-mesh-vertex flag (size == liver
//                            vertex count). Populated from
//                            LiverRegionLabel::RIM. Empty -> Phase 2
//                            beta_src is silently treated as 0.
//    rim_tgt_mask          : per-extracted-target-point flag (must
//                            match the target cloud size produced by
//                            extractFrontFacePoints). Empty -> Phase 2
//                            beta_tgt is silently treated as 0.
//    anchor_pair_src_full_idx / anchor_pair_tgt_pos:
//                            paired arrays (same length) from
//                            g_lastRimPair* of the previous Ctrl+G
//                            session. The vertex indices are
//                            full-mesh space, so we use
//                            subsetToFullVertIdx[] as the inverse map.
//                            Either empty -> anchor mode disabled this
//                            session (preserves byte-identical
//                            contract when useAnchorPairs is also off).
//
//  All four trailing arguments are optional. When omitted, behaviour
//  matches the pre-feature header (modulo the per-step rim/anchor
//  branches inside refineStep, which detect empty arrays and short-
//  circuit back to the original code path).
// ============================================================
inline bool initRefine(
    RefineState& state,
    mCutMesh* liverMesh,
    const std::vector<size_t>& visibleVertexIndices,
    mCutMesh* screenMesh,
    std::vector<mCutMesh*>& organs,
    int gridW, int gridH, float zThreshold,
    const RefineParams& params,
    RefineMethod method,
    // [Phase 0/2/3] Optional inputs — see header comment above.
    const std::vector<uint8_t>* rim_src_mask                    = nullptr,
    const std::vector<uint8_t>* rim_tgt_mask                    = nullptr,
    const std::vector<int>*     anchor_pair_src_full_idx        = nullptr,
    const std::vector<glm::vec3>* anchor_pair_tgt_pos           = nullptr)
{
    state.reset();
    state.params = params;
    state.method = method;
    state.organMeshes = organs;

    const char* tag = methodTag(method);

    NoOpen3DRegistration reg;
    state.targetCloud = reg.extractFrontFacePoints(*screenMesh, gridW, gridH, zThreshold);
    if (!state.targetCloud || state.targetCloud->size() < 50) {
        std::cerr << tag << " Not enough target points: "
                  << (state.targetCloud ? state.targetCloud->size() : 0) << std::endl;
        return false;
    }
    if (!state.targetCloud->hasNormals()) {
        std::cerr << tag << " Target has no normals" << std::endl;
        return false;
    }

    state.targetPtsD.resize(state.targetCloud->size());
    for (size_t i = 0; i < state.targetCloud->size(); i++) {
        state.targetPtsD[i] = glm::dvec3(state.targetCloud->points[i]);
    }
    state.targetAdaptor = std::make_unique<NanoflannAdaptorD>(state.targetPtsD);
    state.targetTree = buildKDTreeD(*state.targetAdaptor);

    state.sourceCloud = std::make_shared<PointCloud>();
    size_t liverVertCount = liverMesh->mVertices.size() / 3;
    bool hasNormals = !liverMesh->mNormals.empty() &&
                      liverMesh->mNormals.size() >= liverVertCount * 3;

    // [Phase 0/2/3] Build subsetToFullVertIdx in lockstep with sourceCloud
    //   so the compact source index i corresponds exactly to a full-mesh
    //   vertex index. Always populated (cost is trivial, simplifies the
    //   anchor lookup below).
    state.subsetToFullVertIdx.clear();
    state.subsetToFullVertIdx.reserve(visibleVertexIndices.size());

    // [Phase 0/2] Build isRimSrc in the same loop so it stays aligned with
    //   sourceCloud[i]. Only populated when rim_src_mask is provided AND
    //   its size matches the full-mesh vertex count; otherwise left empty
    //   (the refineStep code path treats empty as "no rim info").
    const bool have_rim_src = (rim_src_mask != nullptr)
                              && (rim_src_mask->size() == liverVertCount);
    state.isRimSrc.clear();
    if (have_rim_src) state.isRimSrc.reserve(visibleVertexIndices.size());

    int skippedOOB = 0;
    for (size_t idx : visibleVertexIndices) {
        if (idx >= liverVertCount) { skippedOOB++; continue; }
        glm::vec3 v(liverMesh->mVertices[idx * 3],
                    liverMesh->mVertices[idx * 3 + 1],
                    liverMesh->mVertices[idx * 3 + 2]);
        glm::vec3 n(0, 0, 1);
        if (hasNormals) {
            n = glm::vec3(liverMesh->mNormals[idx * 3],
                          liverMesh->mNormals[idx * 3 + 1],
                          liverMesh->mNormals[idx * 3 + 2]);
            float len = glm::length(n);
            if (len > 1e-6f) n /= len;
            else n = glm::vec3(0, 0, 1);
        }
        state.sourceCloud->addPointWithNormal(v, n);
        // [Phase 0/2/3] Keep aux arrays aligned with sourceCloud[i].
        state.subsetToFullVertIdx.push_back(static_cast<int>(idx));
        if (have_rim_src) {
            state.isRimSrc.push_back((*rim_src_mask)[idx] ? 1 : 0);
        }
    }

    // [Phase 0/2] Build isRimTgt from rim_tgt_mask if supplied and sized
    //   to match the extracted target cloud. We accept the mask only when
    //   it's exactly aligned to extractFrontFacePoints' output so we don't
    //   silently use the wrong indexing.
    if (rim_tgt_mask != nullptr &&
        rim_tgt_mask->size() == state.targetCloud->size())
    {
        state.isRimTgt = *rim_tgt_mask;
    } else {
        state.isRimTgt.clear();
        if (rim_tgt_mask != nullptr && !rim_tgt_mask->empty()) {
            std::cerr << tag << " rim_tgt_mask size mismatch: got "
                      << rim_tgt_mask->size()
                      << ", expected " << state.targetCloud->size()
                      << " -> beta_rim_tgt will be silently ignored." << std::endl;
        }
    }

    std::cout << tag << " Visible indices: " << visibleVertexIndices.size()
              << " extracted: " << state.sourceCloud->size()
              << " skippedOOB: " << skippedOOB << std::endl;

    if (state.sourceCloud->size() < 20) {
        std::cerr << tag << " Not enough source points: " << state.sourceCloud->size() << std::endl;
        return false;
    }

    {
        glm::vec3 srcMin(1e10f), srcMax(-1e10f);
        glm::vec3 tgtMin(1e10f), tgtMax(-1e10f);
        for (const auto& p : state.sourceCloud->points) {
            srcMin = glm::min(srcMin, p); srcMax = glm::max(srcMax, p);
        }
        for (const auto& p : state.targetCloud->points) {
            tgtMin = glm::min(tgtMin, p); tgtMax = glm::max(tgtMax, p);
        }
        std::cout << tag << " Source AABB: ("
                  << std::fixed << std::setprecision(3)
                  << srcMin.x << "," << srcMin.y << "," << srcMin.z << ") - ("
                  << srcMax.x << "," << srcMax.y << "," << srcMax.z << ")" << std::endl;
        std::cout << tag << " Target AABB: ("
                  << tgtMin.x << "," << tgtMin.y << "," << tgtMin.z << ") - ("
                  << tgtMax.x << "," << tgtMax.y << "," << tgtMax.z << ")" << std::endl;
    }

    state.sourcePtsD.resize(state.sourceCloud->size());
    state.sourceNormalsD.resize(state.sourceCloud->size());
    for (size_t i = 0; i < state.sourceCloud->size(); i++) {
        state.sourcePtsD[i] = glm::dvec3(state.sourceCloud->points[i]);
        state.sourceNormalsD[i] = glm::dvec3(state.sourceCloud->normals[i]);
    }

    // [Phase 0/3] Build anchorTgtPos from optional anchor pair arrays.
    //   Each entry default-initialised to NaN so the refineStep code can
    //   detect "no anchor for this vertex" by checking !std::isfinite(x).
    //   Population walks the anchor_pair_* arrays (full-mesh idx -> world
    //   pos), and the inverse map subsetToFullVertIdx[] is searched via
    //   an unordered_map so the lookup is O(N_anchor + N_source) total.
    state.anchorTgtPos.assign(
        state.sourcePtsD.size(),
        glm::dvec3(std::numeric_limits<double>::quiet_NaN()));
    state.anchorCount = 0;

    if (params.useAnchorPairs &&
        anchor_pair_src_full_idx != nullptr &&
        anchor_pair_tgt_pos      != nullptr &&
        !anchor_pair_src_full_idx->empty() &&
        anchor_pair_src_full_idx->size() == anchor_pair_tgt_pos->size())
    {
        // Build inverse map: full-mesh vertex index -> compact source index.
        std::unordered_map<int, int> full_to_compact;
        full_to_compact.reserve(state.subsetToFullVertIdx.size() * 2);
        for (int ci = 0; ci < static_cast<int>(state.subsetToFullVertIdx.size()); ci++) {
            full_to_compact[state.subsetToFullVertIdx[ci]] = ci;
        }

        // For each (full_idx, tgt_pos) pair, if the full_idx is among our
        // source subset, install the anchor for the compact index.
        // If multiple anchor entries point to the same source vertex, the
        // LAST one wins (deterministic since input order is the
        // g_lastRimPair* order from compute_rim_only_rmse_diag).
        for (size_t k = 0; k < anchor_pair_src_full_idx->size(); k++) {
            const int full_idx = (*anchor_pair_src_full_idx)[k];
            auto it = full_to_compact.find(full_idx);
            if (it == full_to_compact.end()) continue;
            const int ci = it->second;
            // Replace; if it was already set we just overwrite (last wins).
            if (!std::isfinite(state.anchorTgtPos[ci].x)) state.anchorCount++;
            state.anchorTgtPos[ci] = glm::dvec3((*anchor_pair_tgt_pos)[k]);
        }
        std::cout << tag << " Anchor pairs: " << state.anchorCount
                  << " / " << state.sourcePtsD.size()
                  << " source verts have anchors"
                  << "  (input pairs=" << anchor_pair_src_full_idx->size()
                  << ", phaseIter=" << params.anchorPhaseIter
                  << ", blend=" << params.anchorBlend << ")" << std::endl;
    } else if (params.useAnchorPairs) {
        std::cout << tag << " Anchor pairs requested but inputs empty / mismatched"
                            " -> running without anchors." << std::endl;
    }

    // [Phase 0/2] Rim counts log (helps the operator see at a glance
    //   how many rim points are participating).
    if (!state.isRimSrc.empty() && params.betaRimSrc > 0.0f) {
        int n_rim_src = 0;
        for (uint8_t v : state.isRimSrc) if (v) n_rim_src++;
        std::cout << tag << " Rim src: " << n_rim_src
                  << " / " << state.isRimSrc.size()
                  << "  betaSrc=" << params.betaRimSrc << std::endl;
    }
    if (!state.isRimTgt.empty() && params.betaRimTgt > 0.0f) {
        int n_rim_tgt = 0;
        for (uint8_t v : state.isRimTgt) if (v) n_rim_tgt++;
        std::cout << tag << " Rim tgt: " << n_rim_tgt
                  << " / " << state.isRimTgt.size()
                  << "  betaTgt=" << params.betaRimTgt << std::endl;
    }

    {
        int distPass = 0, normalPass = 0;
        double maxDistSq = static_cast<double>(params.distanceThreshold) * params.distanceThreshold;
        double dotSum = 0.0;
        int dotCount = 0;
        double closestDist = 1e10;
        for (size_t i = 0; i < state.sourcePtsD.size(); i++) {
            size_t nnIdx; double nnDistSq;
            searchKNN1D(*state.targetTree, state.sourcePtsD[i], nnIdx, nnDistSq);
            double d = std::sqrt(nnDistSq);
            if (d < closestDist) closestDist = d;
            if (nnDistSq < maxDistSq) {
                distPass++;
                glm::dvec3 srcN = glm::normalize(state.sourceNormalsD[i]);
                glm::dvec3 tgtN = glm::normalize(glm::dvec3(state.targetCloud->normals[nnIdx]));
                double dot = glm::dot(srcN, tgtN);
                dotSum += dot; dotCount++;
                if (std::abs(dot) > static_cast<double>(params.minNormalAngleCos))
                    normalPass++;
            }
        }
        double avgDot = dotCount > 0 ? dotSum / dotCount : 0.0;
        std::cout << tag << " Diagnostics: closestDist=" << std::fixed << std::setprecision(4) << closestDist
                  << " distPass=" << distPass
                  << " normalPass=" << normalPass
                  << " avgNormalDot=" << std::setprecision(3) << avgDot << std::endl;
    }

    // Compute initial raw RMSE (all source points)
    {
        double totalDist2 = 0.0;
        for (size_t i = 0; i < state.sourcePtsD.size(); i++) {
            size_t nnIdx; double nnDistSq;
            searchKNN1D(*state.targetTree, state.sourcePtsD[i], nnIdx, nnDistSq);
            totalDist2 += nnDistSq;
        }
        state.initialRMSE = static_cast<float>(
            std::sqrt(totalDist2 / static_cast<double>(state.sourcePtsD.size())));
        state.bestRMSE = state.initialRMSE;
        state.prevRMSE = state.initialRMSE;
        std::cout << tag << " Initial raw RMSE: " << std::fixed << std::setprecision(4)
                  << state.initialRMSE << std::endl;
    }

    // Backup all mesh vertices for revert
    state.backupMeshes();

    state.active = true;
    std::cout << tag << " Initialized: source=" << state.sourceCloud->size()
              << " target=" << state.targetCloud->size()
              << " method=" << (method == SRT_VARIANCE ? "SRT_VARIANCE" : "NORMAL_COMPAT")
              << std::endl;
    if (method == SRT_VARIANCE) {
        std::cout << tag << " Params: nSamples=" << params.nSamples
                  << " sampleRange=" << params.sampleRange
                  << " slope=" << params.srtSlope << std::endl;
    }
    return true;
}

// ============================================================
//  Key N: Normal-Compatible Weighted ICP (Soft Weighting)
//    ALL source points participate — no hard cutoffs.
//    weight = sigmoid(dist) × normalCompat²
//    Far points / misaligned normals get near-zero weight
//    but still anchor the global pose.
//
//  [Phase 2] L1 multiplicative rim boost:
//    w *= (1 + betaSrc * isRimSrc[i]) * (1 + betaTgt * isRimTgt[nnIdx])
//    Empty masks or beta=0 → identity multiplier (byte-identical).
//
//  [Phase 3] L2 anchor-phase override:
//    For iter < anchorPhaseIter, source vertices that own an anchor
//    use the anchor target instead of the runtime KDTree NN. The
//    target normal still comes from the KDTree's nearest target
//    point (the anchor stores a position only, not a normal), so the
//    point-to-plane residual is well-defined. After anchorPhaseIter,
//    anchors are silently ignored and the original NN path resumes.
// ============================================================
inline RefineStepResult refineStepNormalCompat(
    RefineState& state,
    const glm::vec3& viewDirection)
{
    RefineStepResult result;
    if (!state.active) return result;

    const auto& params = state.params;
    float sigma = anneal(params.sigmaSoftDistStart, params.sigmaSoftDistEnd,
                         state.totalIterations, params.annealEndIteration);

    bool hasBd = params.useZWeight && state.targetCloud->hasBoundaryDist();

    // [Phase 2] Detect rim availability per side; cached locally so the
    //   inner loop pays no per-iteration cost when rim is off.
    const bool useRimSrc = (params.betaRimSrc > 0.0f) &&
                           (state.isRimSrc.size() == state.sourcePtsD.size());
    const bool useRimTgt = (params.betaRimTgt > 0.0f) &&
                           (state.isRimTgt.size() == state.targetPtsD.size());

    // [Phase 7a] Pure-RIM target filter: when ON, drop NN whose target
    //   point is not a rim point. Independent of betaRimTgt — the L1
    //   beta still applies among the rim points that survive.
    const bool pureRimTgt = params.pureRimTarget &&
                            (state.isRimTgt.size() == state.targetPtsD.size());

    // [Phase 3] Anchor-phase detection. We compare the OUTER iteration
    //   counter (state.totalIterations) which increments by 1 per
    //   solvePoseAndUpdate. Sub-iterations within itersPerFrame share
    //   the same anchor state.
    const bool anchorPhase = params.useAnchorPairs
                             && state.anchorCount > 0
                             && state.totalIterations < params.anchorPhaseIter;
    const double anchorBlendD = static_cast<double>(
        std::clamp(params.anchorBlend, 0.0f, 1.0f));

    for (int subIter = 0; subIter < params.itersPerFrame; subIter++) {

        struct Corr { int srcIdx, tgtIdx; double weight; float bdist; bool anchor; };
        std::vector<Corr> correspondences;
        double totalWeightedError2 = 0.0;
        double totalWeight = 0.0;
        double totalRawError2 = 0.0;
        int totalPoints = 0;
        int effectiveCorr = 0;
        int anchorUsed = 0;

        for (int i = 0; i < static_cast<int>(state.sourcePtsD.size()); i++) {
            size_t nnIdx; double nnDistSq;
            searchKNN1D(*state.targetTree, state.sourcePtsD[i], nnIdx, nnDistSq);

            // [Phase 7a] Pure-RIM target filter: skip non-rim target NN.
            //   Applied AFTER the KDTree query so the NN is well-defined
            //   even when most candidates are filtered out. Anchored
            //   correspondences are exempted (anchor overrides the NN by
            //   fiat, and the anchor target is always a rim point by
            //   construction in g_lastRimPair*).
            if (pureRimTgt && !state.isRimTgt[nnIdx]) {
                const bool anchorOverride = anchorPhase
                                            && std::isfinite(state.anchorTgtPos[i].x);
                if (!anchorOverride) continue;
            }

            // [Phase 3] Anchor override. If this vertex has a finite
            //   anchor and we're still in the anchor phase, replace
            //   the NN target with the anchor position blended toward
            //   the NN by anchorBlend. The KDTree's nnIdx is kept for
            //   the target NORMAL only (anchor stores position).
            bool useAnchorHere = false;
            glm::dvec3 effective_vt;  // residual is computed against this
            if (anchorPhase && std::isfinite(state.anchorTgtPos[i].x)) {
                useAnchorHere = true;
                const glm::dvec3 anchor_vt = state.anchorTgtPos[i];
                const glm::dvec3 nn_vt     = state.targetPtsD[nnIdx];
                effective_vt = anchorBlendD * anchor_vt
                               + (1.0 - anchorBlendD) * nn_vt;
                // Recompute distance using the effective target so the
                //   sigmoid weight reflects the residual the optimiser
                //   actually sees, not the NN distance.
                nnDistSq = glm::dot(state.sourcePtsD[i] - effective_vt,
                                    state.sourcePtsD[i] - effective_vt);
                anchorUsed++;
            } else {
                effective_vt = state.targetPtsD[nnIdx];
            }

            float dist = static_cast<float>(std::sqrt(nnDistSq));
            double wDist = static_cast<double>(sigmoidWeight(dist, params.distanceThreshold, sigma));

            glm::dvec3 srcN = glm::normalize(state.sourceNormalsD[i]);
            glm::dvec3 tgtN = glm::normalize(glm::dvec3(state.targetCloud->normals[nnIdx]));
            double normalCompat = std::abs(glm::dot(srcN, tgtN));

            double w = wDist * normalCompat * normalCompat;

            // [Phase 2] L1 rim multiplicative boost.
            if (useRimSrc && state.isRimSrc[i])
                w *= (1.0 + static_cast<double>(params.betaRimSrc));
            if (useRimTgt && state.isRimTgt[nnIdx])
                w *= (1.0 + static_cast<double>(params.betaRimTgt));

            float bd = hasBd ? state.targetCloud->boundaryDist[nnIdx] : 9999.0f;
            if (hasBd && bd < params.boundaryWidth) {
                w *= static_cast<double>(params.boundaryBoost);
            }

            totalRawError2 += nnDistSq;
            totalPoints++;

            if (w < 1e-6) continue;

            correspondences.push_back({i, static_cast<int>(nnIdx), w, bd, useAnchorHere});
            totalWeightedError2 += w * nnDistSq;
            totalWeight += w;
            if (w > 0.01) effectiveCorr++;
        }

        if (correspondences.size() < 6) { result.converged = true; break; }

        double JTJ[6][6] = {};
        double JTr_vec[6] = {};

        for (const auto& c : correspondences) {
            glm::dvec3 vs = state.sourcePtsD[c.srcIdx];
            // [Phase 3] Use the anchor position for the residual point
            //   when this correspondence was anchored; otherwise use
            //   the KDTree NN target. The target NORMAL always comes
            //   from the KDTree NN (anchor stores position only).
            glm::dvec3 vt;
            if (c.anchor) {
                const glm::dvec3 anchor_vt = state.anchorTgtPos[c.srcIdx];
                const glm::dvec3 nn_vt     = state.targetPtsD[c.tgtIdx];
                vt = anchorBlendD * anchor_vt + (1.0 - anchorBlendD) * nn_vt;
            } else {
                vt = state.targetPtsD[c.tgtIdx];
            }
            glm::dvec3 nt = glm::normalize(glm::dvec3(state.targetCloud->normals[c.tgtIdx]));
            if (glm::dot(glm::normalize(state.sourceNormalsD[c.srcIdx]), nt) < 0.0) nt = -nt;

            if (hasBd) {
                float alphaZ = computeAlphaZ(c.bdist, params.boundaryWidth,
                                             params.zWeightBoundary, params.zWeightInterior);
                nt.z *= static_cast<double>(alphaZ);
                double len = glm::length(nt);
                if (len > 1e-12) nt /= len;
            }

            double r = glm::dot(vs - vt, nt);
            double sw = std::sqrt(c.weight);
            double J[6];
            J[0] = (vs.y * nt.z - vs.z * nt.y) * sw;
            J[1] = (vs.z * nt.x - vs.x * nt.z) * sw;
            J[2] = (vs.x * nt.y - vs.y * nt.x) * sw;
            J[3] = nt.x * sw; J[4] = nt.y * sw; J[5] = nt.z * sw;
            r *= sw;

            for (int a = 0; a < 6; a++) {
                JTr_vec[a] += J[a] * r;
                for (int b = 0; b < 6; b++) JTJ[a][b] += J[a] * J[b];
            }
        }

        double rawRmse = (totalPoints > 0)
                             ? std::sqrt(totalRawError2 / static_cast<double>(totalPoints))
                             : 0.0;

        if (!solvePoseAndUpdate(JTJ, JTr_vec, state, result,
                                effectiveCorr, totalWeightedError2)) {
            result.converged = true; break;
        }

        result.rmse = static_cast<float>(rawRmse);
        result.correspondenceCount = effectiveCorr;

        // [Phase 3] Per-sub-iter trace: useful to see anchor fade-out
        //   right after anchorPhaseIter is crossed.
        if (anchorPhase) {
            std::cout << methodTag(state.method)
            << " iter=" << state.totalIterations
            << " anchor=" << anchorUsed
            << "/" << correspondences.size()
            << " rmse=" << std::fixed << std::setprecision(5) << result.rmse
            << std::defaultfloat << std::endl;
        }
    }

    checkConvergence(state, result);
    return result;
}

// ============================================================
//  Key B: SRT Variance-Weighted ICP
//    Correspondence: KDTree + normal filter (same as Key N)
//    Weight: 1/stddev from normal-line proximity distribution
//    Residual: point-to-plane (same as Key N)
//
//    The SRT variance measures "how clearly is a target surface
//    present along this source normal?" Low variance = sharp peak
//    = confident correspondence. High variance = ambiguous.
//
//    [Phase 2/3] Identical L1 rim boost and L2 anchor-phase override
//    as refineStepNormalCompat. See those comments for semantics.
// ============================================================
inline RefineStepResult refineStepSRTVariance(
    RefineState& state,
    const glm::vec3& viewDirection)
{
    RefineStepResult result;
    if (!state.active) return result;

    const auto& params = state.params;
    float cosThresh = anneal(params.minNormalAngleCos, params.maxNormalAngleCos,
                             state.totalIterations, params.annealEndIteration);
    double maxDistSq = static_cast<double>(params.distanceThreshold) * params.distanceThreshold;

    const double range = static_cast<double>(anneal(params.sampleRange, params.sampleRange * 0.5f,
                                                    state.totalIterations, params.annealEndIteration));
    const double slope = static_cast<double>(anneal(params.srtSlope, params.srtSlope * 0.3f,
                                                    state.totalIterations, params.annealEndIteration));
    const double thresh = static_cast<double>(params.distanceThreshold);
    const double minVar = static_cast<double>(params.srtMinVariance);

    bool hasBd = params.useZWeight && state.targetCloud->hasBoundaryDist();

    // [Phase 2] Rim availability (same as Normal-Compat).
    const bool useRimSrc = (params.betaRimSrc > 0.0f) &&
                           (state.isRimSrc.size() == state.sourcePtsD.size());
    const bool useRimTgt = (params.betaRimTgt > 0.0f) &&
                           (state.isRimTgt.size() == state.targetPtsD.size());

    // [Phase 3] Anchor phase.
    const bool anchorPhase = params.useAnchorPairs
                             && state.anchorCount > 0
                             && state.totalIterations < params.anchorPhaseIter;
    const double anchorBlendD = static_cast<double>(
        std::clamp(params.anchorBlend, 0.0f, 1.0f));

    // [Phase 7a] Pure-RIM target filter (same semantics as NormalCompat).
    const bool pureRimTgt = params.pureRimTarget &&
                            (state.isRimTgt.size() == state.targetPtsD.size());

    for (int subIter = 0; subIter < params.itersPerFrame; subIter++) {

        struct Corr { int srcIdx, tgtIdx; double weight; float bdist; bool anchor; };
        std::vector<Corr> correspondences;
        double totalError2 = 0.0;
        int anchorUsed = 0;

        for (int i = 0; i < static_cast<int>(state.sourcePtsD.size()); i++) {
            size_t nnIdx; double nnDistSq;
            searchKNN1D(*state.targetTree, state.sourcePtsD[i], nnIdx, nnDistSq);

            // [Phase 7a] Pure-RIM target filter: drop non-rim NN.
            //   Anchored verts exempt (anchor target is rim by
            //   construction). Placed BEFORE the maxDistSq gate so the
            //   gate doesn't fire on filtered points (subtle but matters
            //   for the early-out optimisation).
            if (pureRimTgt && !state.isRimTgt[nnIdx]) {
                const bool anchorOverride = anchorPhase
                                            && std::isfinite(state.anchorTgtPos[i].x);
                if (!anchorOverride) continue;
            }

            // [Phase 3] Anchor override (only applies if vertex has anchor
            //   AND we're inside the anchor phase). Note: for SRT_VARIANCE
            //   we KEEP the normal-cos filter against the NN-target normal
            //   (target NORMAL is still from KDTree) but allow the residual
            //   to be against the anchor position. We do NOT apply the
            //   max-dist gate to anchored vertices, because the gate would
            //   reject the anchor early in the anchor phase exactly when
            //   we want it to participate.
            bool useAnchorHere = false;
            glm::dvec3 effective_vt;
            if (anchorPhase && std::isfinite(state.anchorTgtPos[i].x)) {
                useAnchorHere = true;
                const glm::dvec3 anchor_vt = state.anchorTgtPos[i];
                const glm::dvec3 nn_vt     = state.targetPtsD[nnIdx];
                effective_vt = anchorBlendD * anchor_vt
                               + (1.0 - anchorBlendD) * nn_vt;
                nnDistSq = glm::dot(state.sourcePtsD[i] - effective_vt,
                                    state.sourcePtsD[i] - effective_vt);
                anchorUsed++;
            } else {
                effective_vt = state.targetPtsD[nnIdx];
                if (nnDistSq >= maxDistSq) continue;
            }

            glm::dvec3 srcN = glm::normalize(state.sourceNormalsD[i]);
            glm::dvec3 tgtN = glm::normalize(glm::dvec3(state.targetCloud->normals[nnIdx]));
            double normalCompat = std::abs(glm::dot(srcN, tgtN));
            // Anchored verts skip the cosThresh filter — the anchor
            //   defines the correspondence by fiat in this phase.
            if (!useAnchorHere && normalCompat < static_cast<double>(cosThresh)) continue;

            double variance = computeSRTVariance(
                state.sourcePtsD[i], srcN, *state.targetTree,
                params.nSamples, range, slope, thresh, minVar);
            double stddev = std::sqrt(variance);
            double w = 1.0 / stddev;

            // [Phase 2] L1 rim boost.
            if (useRimSrc && state.isRimSrc[i])
                w *= (1.0 + static_cast<double>(params.betaRimSrc));
            if (useRimTgt && state.isRimTgt[nnIdx])
                w *= (1.0 + static_cast<double>(params.betaRimTgt));

            float bd = hasBd ? state.targetCloud->boundaryDist[nnIdx] : 9999.0f;
            if (hasBd && bd < params.boundaryWidth) {
                w *= static_cast<double>(params.boundaryBoost);
            }

            correspondences.push_back({i, static_cast<int>(nnIdx), w, bd, useAnchorHere});
            totalError2 += nnDistSq;
        }

        if (correspondences.size() < 6) { result.converged = true; break; }

        double JTJ[6][6] = {};
        double JTr_vec[6] = {};

        for (const auto& c : correspondences) {
            glm::dvec3 vs = state.sourcePtsD[c.srcIdx];
            glm::dvec3 vt;
            if (c.anchor) {
                const glm::dvec3 anchor_vt = state.anchorTgtPos[c.srcIdx];
                const glm::dvec3 nn_vt     = state.targetPtsD[c.tgtIdx];
                vt = anchorBlendD * anchor_vt + (1.0 - anchorBlendD) * nn_vt;
            } else {
                vt = state.targetPtsD[c.tgtIdx];
            }
            glm::dvec3 nt = glm::normalize(glm::dvec3(state.targetCloud->normals[c.tgtIdx]));
            if (glm::dot(glm::normalize(state.sourceNormalsD[c.srcIdx]), nt) < 0.0) nt = -nt;

            if (hasBd) {
                float alphaZ = computeAlphaZ(c.bdist, params.boundaryWidth,
                                             params.zWeightBoundary, params.zWeightInterior);
                nt.z *= static_cast<double>(alphaZ);
                double len = glm::length(nt);
                if (len > 1e-12) nt /= len;
            }

            double r = glm::dot(vs - vt, nt);
            double sw = std::sqrt(c.weight);
            double J[6];
            J[0] = (vs.y * nt.z - vs.z * nt.y) * sw;
            J[1] = (vs.z * nt.x - vs.x * nt.z) * sw;
            J[2] = (vs.x * nt.y - vs.y * nt.x) * sw;
            J[3] = nt.x * sw; J[4] = nt.y * sw; J[5] = nt.z * sw;
            r *= sw;

            for (int a = 0; a < 6; a++) {
                JTr_vec[a] += J[a] * r;
                for (int b = 0; b < 6; b++) JTJ[a][b] += J[a] * J[b];
            }
        }

        if (!solvePoseAndUpdate(JTJ, JTr_vec, state, result,
                                static_cast<int>(correspondences.size()), totalError2)) {
            result.converged = true; break;
        }

        if (anchorPhase) {
            std::cout << methodTag(state.method)
            << " iter=" << state.totalIterations
            << " anchor=" << anchorUsed
            << "/" << correspondences.size()
            << " rmse=" << std::fixed << std::setprecision(5) << result.rmse
            << std::defaultfloat << std::endl;
        }
    }

    checkConvergence(state, result);
    return result;
}

// ============================================================
//  Dispatch
// ============================================================
inline RefineStepResult refineStep(
    RefineState& state,
    const glm::vec3& viewDirection)
{
    if (state.method == SRT_VARIANCE)
        return refineStepSRTVariance(state, viewDirection);
    else
        return refineStepNormalCompat(state, viewDirection);
}

// ============================================================
//  Apply transform to all organ meshes
// ============================================================
inline void applyIncrementalTransform(
    const glm::dmat4& transform,
    std::vector<mCutMesh*>& meshes)
{
    glm::mat4 T = glm::mat4(transform);
    glm::mat3 R = glm::mat3(glm::dmat3(transform));

    for (auto* mesh : meshes) {
        size_t vertCount = mesh->mVertices.size() / 3;
        for (size_t i = 0; i < vertCount; i++) {
            glm::vec4 v(mesh->mVertices[i * 3],
                        mesh->mVertices[i * 3 + 1],
                        mesh->mVertices[i * 3 + 2], 1.0f);
            v = T * v;
            mesh->mVertices[i * 3]     = v.x;
            mesh->mVertices[i * 3 + 1] = v.y;
            mesh->mVertices[i * 3 + 2] = v.z;
        }
        if (!mesh->mNormals.empty() && mesh->mNormals.size() >= vertCount * 3) {
            for (size_t i = 0; i < vertCount; i++) {
                glm::vec3 n(mesh->mNormals[i * 3],
                            mesh->mNormals[i * 3 + 1],
                            mesh->mNormals[i * 3 + 2]);
                n = glm::normalize(R * n);
                mesh->mNormals[i * 3]     = n.x;
                mesh->mNormals[i * 3 + 1] = n.y;
                mesh->mNormals[i * 3 + 2] = n.z;
            }
        }
        setUp(*mesh);
    }
}

}
