#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

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
// ============================================================
inline bool initRefine(
    RefineState& state,
    mCutMesh* liverMesh,
    const std::vector<size_t>& visibleVertexIndices,
    mCutMesh* screenMesh,
    std::vector<mCutMesh*>& organs,
    int gridW, int gridH, float zThreshold,
    const RefineParams& params,
    RefineMethod method)
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

    for (int subIter = 0; subIter < params.itersPerFrame; subIter++) {

        struct Corr { int srcIdx, tgtIdx; double weight; float bdist; };
        std::vector<Corr> correspondences;
        double totalWeightedError2 = 0.0;
        double totalWeight = 0.0;
        double totalRawError2 = 0.0;
        int totalPoints = 0;
        int effectiveCorr = 0;

        for (int i = 0; i < static_cast<int>(state.sourcePtsD.size()); i++) {
            size_t nnIdx; double nnDistSq;
            searchKNN1D(*state.targetTree, state.sourcePtsD[i], nnIdx, nnDistSq);

            float dist = static_cast<float>(std::sqrt(nnDistSq));
            double wDist = static_cast<double>(sigmoidWeight(dist, params.distanceThreshold, sigma));

            glm::dvec3 srcN = glm::normalize(state.sourceNormalsD[i]);
            glm::dvec3 tgtN = glm::normalize(glm::dvec3(state.targetCloud->normals[nnIdx]));
            double normalCompat = std::abs(glm::dot(srcN, tgtN));

            double w = wDist * normalCompat * normalCompat;

            float bd = hasBd ? state.targetCloud->boundaryDist[nnIdx] : 9999.0f;
            if (hasBd && bd < params.boundaryWidth) {
                w *= static_cast<double>(params.boundaryBoost);
            }

            totalRawError2 += nnDistSq;
            totalPoints++;

            if (w < 1e-6) continue;

            correspondences.push_back({i, static_cast<int>(nnIdx), w, bd});
            totalWeightedError2 += w * nnDistSq;
            totalWeight += w;
            if (w > 0.01) effectiveCorr++;
        }

        if (correspondences.size() < 6) { result.converged = true; break; }

        double JTJ[6][6] = {};
        double JTr_vec[6] = {};

        for (const auto& c : correspondences) {
            glm::dvec3 vs = state.sourcePtsD[c.srcIdx];
            glm::dvec3 vt = state.targetPtsD[c.tgtIdx];
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

    for (int subIter = 0; subIter < params.itersPerFrame; subIter++) {

        struct Corr { int srcIdx, tgtIdx; double weight; float bdist; };
        std::vector<Corr> correspondences;
        double totalError2 = 0.0;

        for (int i = 0; i < static_cast<int>(state.sourcePtsD.size()); i++) {
            size_t nnIdx; double nnDistSq;
            searchKNN1D(*state.targetTree, state.sourcePtsD[i], nnIdx, nnDistSq);
            if (nnDistSq >= maxDistSq) continue;

            glm::dvec3 srcN = glm::normalize(state.sourceNormalsD[i]);
            glm::dvec3 tgtN = glm::normalize(glm::dvec3(state.targetCloud->normals[nnIdx]));
            double normalCompat = std::abs(glm::dot(srcN, tgtN));
            if (normalCompat < static_cast<double>(cosThresh)) continue;

            double variance = computeSRTVariance(
                state.sourcePtsD[i], srcN, *state.targetTree,
                params.nSamples, range, slope, thresh, minVar);
            double stddev = std::sqrt(variance);
            double w = 1.0 / stddev;

            float bd = hasBd ? state.targetCloud->boundaryDist[nnIdx] : 9999.0f;
            if (hasBd && bd < params.boundaryWidth) {
                w *= static_cast<double>(params.boundaryBoost);
            }

            correspondences.push_back({i, static_cast<int>(nnIdx), w, bd});
            totalError2 += nnDistSq;
        }

        if (correspondences.size() < 6) { result.converged = true; break; }

        double JTJ[6][6] = {};
        double JTr_vec[6] = {};

        for (const auto& c : correspondences) {
            glm::dvec3 vs = state.sourcePtsD[c.srcIdx];
            glm::dvec3 vt = state.targetPtsD[c.tgtIdx];
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
