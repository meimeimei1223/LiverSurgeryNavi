#ifndef AUTO_DEFORM_H
#define AUTO_DEFORM_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>
#include <glm/glm.hpp>
#include "NoOpen3DRegistration.h"
#include "mCutMesh.h"
#include "SoftBody.h"
#include "Sphere.h"
#include "ShaderProgram.h"

namespace AutoDeform {

enum Category {
    CAT_NONE    = 0,
    CAT_INLIER  = 1,
    CAT_MOVER   = 2,
    CAT_OUTLIER = 3
};

struct Correspondence {
    int       srcIdx;
    glm::vec3 srcPos;
    glm::vec3 srcNormal;
    glm::vec3 tgtPos;
    glm::vec3 tgtNormal;
    float     dist;
    float     distProjected;
    float     boundaryDist;
    float     ratio;
    bool      ratioOK;
    bool      isSilhouette;
    bool      bidirConsistent;
    int       category;
};

struct Stats {
    float median = 0.0f;
    float mad    = 0.0f;
    float minV   = 0.0f;
    float maxV   = 0.0f;
    float mean   = 0.0f;
    int   count  = 0;
};

enum SrcVisibility {
    VIS_VISIBLE   = 0,
    VIS_BUFFER    = 1,
    VIS_HIDDEN    = 2,
    VIS_OUTSCREEN = 3,
};

struct Handle {
    glm::vec3 center;
    glm::vec3 target;
    float     radius;
    bool      isFixed;
};

struct Preset {
    const char* name;
    int   K_fix;
    int   K_move;
    float rFixScale;
    float rMoveScale;
    float tauLowScale;
    float tauHighScale;
    float minSepScale;
    float fixCompliance;
    float moveCompliance;
};

inline const std::vector<Preset>& getPresets() {
    static const std::vector<Preset> presets = {
                                                { "P0 Sequential",   4, 3, 0.10f, 0.08f, 0.5f, 1.5f, 0.20f, 0.0f, 1e-6f },
                                                { "P1 Conservative", 4, 3, 0.08f, 0.06f, 0.5f, 1.5f, 0.15f, 0.0f, 1e-7f },
                                                { "P2 Standard",     3, 5, 0.10f, 0.10f, 0.5f, 1.5f, 0.15f, 0.0f, 1e-6f },
                                                { "P3 Aggressive",   2, 6, 0.10f, 0.14f, 0.5f, 1.4f, 0.15f, 0.0f, 1e-5f },
                                                { "P4 Extreme",      2, 8, 0.08f, 0.18f, 0.5f, 1.3f, 0.12f, 0.0f, 1e-4f },
                                                };
    return presets;
}

struct State {
    std::vector<Correspondence> correspondences;
    Stats statsAll;
    Stats statsSilhouette;
    Stats statsInterior;

    float silhouetteThreshold = 12.0f;
    float thrInlierSil  = 0.0f;
    float thrOutlierSil = 0.0f;
    float thrInlierInt  = 0.0f;
    float thrOutlierInt = 0.0f;

    int stage = 0;
    int nSilInlier = 0, nSilMover = 0, nSilOutlier = 0;
    int nIntInlier = 0, nIntMover = 0, nIntOutlier = 0;
    int nBidirInconsistent = 0;
    int nRatioRejected = 0;
    int srcDownsampledCount = 0;

    std::vector<glm::vec3> fieldVertexPos;
    std::vector<float>     fieldScoreRaw;
    std::vector<float>     fieldScoreSmooth;
    std::vector<bool>      visMeshValidMask;
    int   nVisMeshValid   = 0;
    int   nVisMeshMasked  = 0;
    float fieldMin    = 0.0f;
    float fieldMax    = 0.0f;
    float fieldMedian = 0.0f;
    bool  fieldReady  = false;

    std::vector<Handle> fixHandles;
    std::vector<Handle> moveHandles;
    float fieldBboxDiag = 0.0f;

    // Step 1 (classifySrcVisibility) で計測される source liver の AABB 対角線。
    // scale=1/scale=10 どちらでも全 draw 関数のスフィア半径をこれに比例させる
    // ことで「bbox の 0.5-1% の点」という相対サイズを保つ。
    // 0 のときは旧来の絶対半径(scale=10 想定値)を使う。
    float srcBboxDiag = 0.0f;

    std::vector<int>       seqMoveQueue;
    std::vector<bool>      seqMoveUsed;
    std::vector<int>       seqVisIdxOf;
    std::vector<int>       seqPinnedParticleIdx;
    std::vector<glm::vec3> seqPinnedTargets;
    Handle                 seqLastMove;
    bool                   seqHasLastMove = false;
    int                    seqAppliedCount = 0;
    int                    seqLastQueueSlot = -1;
    int                    seqLastCorrIdx   = -1;
    Handle                 seqAttachedMove;
    bool                   seqHasAttachedMove = false;
    float                  seqProgress = 0.0f;

    std::vector<glm::vec3> srcPosCache;
    std::vector<int>       srcVisibility;
    std::vector<glm::vec3> tgtPosCache;
    int nVisVisible   = 0;
    int nVisBuffer    = 0;
    int nVisHidden    = 0;
    int nVisOutscreen = 0;
    float visTolerance = 0.0f;

    float rmseBeforeDeform = -1.0f;
    float rmseLastMeasured = -1.0f;
    int   rmseLastMatched  = 0;
    int   rmseLastTotal    = 0;

    // SHIFT+Key1/2/3 で 1個だけ描くデバッグモード (0=通常)
    int   debugMaxPoints   = 0;

    void clear() {
        correspondences.clear();
        statsAll = statsSilhouette = statsInterior = Stats{};
        stage = 0;
        thrInlierSil = thrOutlierSil = thrInlierInt = thrOutlierInt = 0.0f;
        nSilInlier = nSilMover = nSilOutlier = 0;
        nIntInlier = nIntMover = nIntOutlier = 0;
        nBidirInconsistent = 0;
        nRatioRejected = 0;
        srcDownsampledCount = 0;
        fieldVertexPos.clear();
        fieldScoreRaw.clear();
        fieldScoreSmooth.clear();
        visMeshValidMask.clear();
        nVisMeshValid = nVisMeshMasked = 0;
        fieldMin = fieldMax = fieldMedian = 0.0f;
        fieldReady = false;
        fixHandles.clear();
        moveHandles.clear();
        fieldBboxDiag = 0.0f;
        srcBboxDiag   = 0.0f;
        seqMoveQueue.clear();
        seqMoveUsed.clear();
        seqVisIdxOf.clear();
        seqPinnedParticleIdx.clear();
        seqPinnedTargets.clear();
        seqHasLastMove = false;
        seqAppliedCount = 0;
        seqLastQueueSlot = -1;
        seqLastCorrIdx   = -1;
        seqHasAttachedMove = false;
        seqProgress = 0.0f;
        srcPosCache.clear();
        srcVisibility.clear();
        tgtPosCache.clear();
        nVisVisible = nVisBuffer = nVisHidden = nVisOutscreen = 0;
        visTolerance = 0.0f;
        rmseBeforeDeform = -1.0f;
        rmseLastMeasured = -1.0f;
        rmseLastMatched  = 0;
        rmseLastTotal    = 0;
    }
};

inline Stats computeStatsFromValues(const std::vector<float>& vals) {
    Stats s;
    if (vals.empty()) return s;
    std::vector<float> d = vals;
    std::sort(d.begin(), d.end());
    s.count  = static_cast<int>(d.size());
    s.minV   = d.front();
    s.maxV   = d.back();
    s.median = d[d.size() / 2];
    double sum = 0.0;
    for (float v : d) sum += v;
    s.mean = static_cast<float>(sum / d.size());
    std::vector<float> absDev(d.size());
    for (size_t i = 0; i < d.size(); i++) absDev[i] = std::fabs(d[i] - s.median);
    std::sort(absDev.begin(), absDev.end());
    s.mad = absDev[absDev.size() / 2];
    return s;
}

inline void classifySrcVisibility(
    State& st,
    mCutMesh* liverMesh,
    mCutMesh* screenMesh,
    int gridW, int gridH,
    const glm::vec3& cameraPos,
    const glm::vec3& cameraTarget,
    float depthScale = 1.0f)
{
    if (!liverMesh || !screenMesh) return;
    if (liverMesh->mVertices.empty() || screenMesh->mVertices.empty()) return;

    const auto& V = liverMesh->mVertices;
    size_t Nsrc = V.size() / 3;

    glm::vec3 mn(1e10f), mx(-1e10f);
    for (size_t i = 0; i + 2 < V.size(); i += 3) {
        glm::vec3 p(V[i], V[i+1], V[i+2]);
        mn = glm::min(mn, p); mx = glm::max(mx, p);
    }
    float diag = glm::length(mx - mn);
    st.srcBboxDiag = diag;   // ★ 全 draw 関数のスフィア半径正規化に使う ★
    st.visTolerance = 0.0f;

    Reg3D::BVHTree bvh;
    bvh.build(liverMesh->mVertices, liverMesh->mIndices);

    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh, bvh, cameraPos, cameraTarget);

    st.srcPosCache.resize(Nsrc);
    for (size_t i = 0; i < Nsrc; i++) {
        st.srcPosCache[i] = glm::vec3(V[i*3], V[i*3+1], V[i*3+2]);
    }

    st.srcVisibility.assign(Nsrc, VIS_HIDDEN);
    for (size_t idx : vis.vertexIndices) {
        if (idx < Nsrc) st.srcVisibility[idx] = VIS_VISIBLE;
    }
    st.nVisVisible   = vis.visibleCount;
    st.nVisBuffer    = 0;
    st.nVisHidden    = vis.occludedCount + vis.backfaceCount;
    st.nVisOutscreen = 0;

    int postDemoted = 0;
    {
        const auto& sv = screenMesh->mVertices;
        int frontN = (gridW + 1) * (gridH + 1);
        int sN = static_cast<int>(sv.size() / 3);
        if (frontN > sN) frontN = sN;

        if (frontN > 0) {
            glm::vec3 viewDir = glm::normalize(cameraTarget - cameraPos);
            glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
            glm::vec3 right = glm::cross(viewDir, worldUp);
            if (glm::length(right) < 1e-6f) right = glm::vec3(1.0f, 0.0f, 0.0f);
            right = glm::normalize(right);
            glm::vec3 up = glm::cross(right, viewDir);

            auto camXY = [&](const glm::vec3& p) -> glm::vec2 {
                glm::vec3 d = p - cameraPos;
                return glm::vec2(glm::dot(d, right), glm::dot(d, up));
            };
            auto camDepth = [&](const glm::vec3& p) -> float {
                return glm::dot(p - cameraPos, viewDir);
            };

            std::vector<glm::vec3> screenPts;
            std::vector<float>     screenDepths;
            screenPts.reserve(frontN);
            screenDepths.reserve(frontN);
            for (int i = 0; i < frontN; i++) {
                glm::vec3 p(sv[i*3], sv[i*3+1], sv[i*3+2]);
                glm::vec2 xy = camXY(p);
                screenPts.emplace_back(xy.x, xy.y, 0.0f);
                screenDepths.push_back(camDepth(p));
            }

            Reg3DCustom::NanoflannAdaptor adaptor(screenPts);
            auto tree = Reg3DCustom::buildKDTree(adaptor);

            float depthTol = diag * 0.05f;
            for (size_t idx = 0; idx < Nsrc; idx++) {
                if (st.srcVisibility[idx] != VIS_VISIBLE) continue;
                glm::vec3 p = st.srcPosCache[idx];
                glm::vec2 xy = camXY(p);
                glm::vec3 query(xy.x, xy.y, 0.0f);
                size_t nnIdx; float distSq;
                if (!Reg3DCustom::searchKNN1(*tree, query, nnIdx, distSq)) continue;
                float srcD    = camDepth(p);
                float screenD = screenDepths[nnIdx];
                if (srcD > screenD + depthTol) {
                    st.srcVisibility[idx] = VIS_HIDDEN;
                    st.nVisVisible--;
                    st.nVisHidden++;
                    postDemoted++;
                }
            }
        }
    }

    Reg3DCustom::NoOpen3DRegistration reg;
    auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridW, gridH, 0.05f);
    st.tgtPosCache.clear();
    if (targetCloud) {
        st.tgtPosCache.reserve(targetCloud->size());
        for (size_t i = 0; i < targetCloud->size(); i++)
            st.tgtPosCache.push_back(targetCloud->points[i]);
    }

    st.stage = 1;

    std::cout << "\n=== [AutoDeform Step1] Src/Target Visibility (Raycast+DepthFilter) ===" << std::endl;
    std::cout << "  bbox diag       : " << diag << std::endl;
    std::cout << "  src vertices    : " << Nsrc << std::endl;
    std::cout << "    raycast vis.  : " << vis.visibleCount << std::endl;
    std::cout << "    occluded      : " << vis.occludedCount << std::endl;
    std::cout << "    backface      : " << vis.backfaceCount << std::endl;
    std::cout << "    post demoted  : " << postDemoted << " (depthTol=" << diag * 0.05f << ")" << std::endl;
    std::cout << "  --- final ---" << std::endl;
    std::cout << "    visible  (G)  : " << st.nVisVisible
              << " (" << (Nsrc > 0 ? 100.0f * st.nVisVisible / Nsrc : 0.0f) << "%)" << std::endl;
    std::cout << "    hidden   (R)  : " << st.nVisHidden
              << " (" << (Nsrc > 0 ? 100.0f * st.nVisHidden / Nsrc : 0.0f) << "%)" << std::endl;
    std::cout << "  target points   : " << st.tgtPosCache.size() << std::endl;
    std::cout << "======================================================================\n" << std::endl;
}

inline void drawSrcVisibility(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float srcRadius = 0.04f,
    float tgtRadius = 0.05f,
    bool  drawHidden = false,
    bool  drawOutscreen = false,
    int   maxPoints = 0)
{
    if (st.srcPosCache.empty()) return;

    static const glm::vec3 colVisible (0.1f, 1.0f, 0.1f);
    static const glm::vec3 colBuffer  (1.0f, 1.0f, 0.0f);
    static const glm::vec3 colHidden  (1.0f, 0.1f, 0.1f);
    static const glm::vec3 colOutscr  (0.4f, 0.4f, 0.4f);
    static const glm::vec3 colTarget  (0.2f, 0.4f, 1.0f);

    int drawn = 0;
    for (size_t i = 0; i < st.srcPosCache.size(); i++) {
        int v = st.srcVisibility[i];
        if (v == VIS_HIDDEN && !drawHidden) continue;
        if (v == VIS_OUTSCREEN && !drawOutscreen) continue;
        glm::vec3 c;
        switch (v) {
        case VIS_VISIBLE:   c = colVisible; break;
        case VIS_BUFFER:    c = colBuffer;  break;
        case VIS_HIDDEN:    c = colHidden;  break;
        default:            c = colOutscr;  break;
        }
        sphere.draw(shader, st.srcPosCache[i], c, srcRadius, view, proj, camPos);
        drawn++;
        if (maxPoints > 0 && drawn >= maxPoints) break;
    }

    if (maxPoints <= 0 || drawn < maxPoints) {
        int tgtDrawn = 0;
        for (const auto& p : st.tgtPosCache) {
            sphere.draw(shader, p, colTarget, tgtRadius, view, proj, camPos);
            tgtDrawn++;
            if (maxPoints > 0 && (drawn + tgtDrawn) >= maxPoints) break;
        }
    }
}

inline void extractCorrespondences(
    State& st,
    mCutMesh* liverMesh,
    mCutMesh* screenMesh,
    int gridW, int gridH,
    float depthScale,
    float maxDist = 1.0f,
    float silhouetteThreshold = 12.0f,
    float ratioThreshold = 0.8f,
    float srcVoxelSize = 0.0f)
{
    auto savedVisibility   = st.srcVisibility;
    auto savedSrcPosCache  = st.srcPosCache;
    auto savedTgtPosCache  = st.tgtPosCache;
    int  savedNVisVisible  = st.nVisVisible;
    int  savedNVisBuffer   = st.nVisBuffer;
    int  savedNVisHidden   = st.nVisHidden;
    int  savedNVisOutscr   = st.nVisOutscreen;
    float savedVisToler    = st.visTolerance;
    float savedSrcBboxDiag = st.srcBboxDiag;   // ★ Key1で計測した値を draw() スフィア半径正規化用に保持 ★

    st.clear();

    st.srcVisibility   = savedVisibility;
    st.srcPosCache     = savedSrcPosCache;
    st.tgtPosCache     = savedTgtPosCache;
    st.nVisVisible     = savedNVisVisible;
    st.nVisBuffer      = savedNVisBuffer;
    st.nVisHidden      = savedNVisHidden;
    st.nVisOutscreen   = savedNVisOutscr;
    st.visTolerance    = savedVisToler;
    st.srcBboxDiag     = savedSrcBboxDiag;     // ★ 復元(Key1→Key2 で 0 に落ちる問題の修正)★

    st.silhouetteThreshold = silhouetteThreshold;
    if (!liverMesh || !screenMesh) return;
    if (liverMesh->mVertices.empty()) return;

    Reg3DCustom::NoOpen3DRegistration reg;
    float zThresh = std::max(0.01f, depthScale * 0.05f);
    auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridW, gridH, zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[AutoDeform::Phase0] target cloud empty" << std::endl;
        return;
    }
    bool hasTgtNormals = targetCloud->hasNormals();
    bool hasBd         = targetCloud->hasBoundaryDist();

    auto sourceCloudFull = std::make_shared<Reg3DCustom::PointCloud>();
    const auto& V = liverMesh->mVertices;
    const auto& N = liverMesh->mNormals;
    bool hasSrcNormals = !N.empty() && N.size() == V.size();
    bool useVisFilter = !st.srcVisibility.empty() && (st.srcVisibility.size() * 3 == V.size());
    int filteredOut = 0;
    for (size_t i = 0; i + 2 < V.size(); i += 3) {
        size_t vidx = i / 3;
        if (useVisFilter) {
            int v = st.srcVisibility[vidx];
            if (v == VIS_HIDDEN || v == VIS_OUTSCREEN) { filteredOut++; continue; }
        }
        glm::vec3 p(V[i], V[i+1], V[i+2]);
        if (hasSrcNormals) {
            glm::vec3 n(N[i], N[i+1], N[i+2]);
            float ln = glm::length(n);
            if (ln > 1e-6f) n /= ln;
            else n = glm::vec3(0, 0, 1);
            sourceCloudFull->addPointWithNormal(p, n);
        } else {
            sourceCloudFull->addPoint(p);
        }
    }
    if (sourceCloudFull->empty()) return;
    if (useVisFilter) {
        std::cout << "[AutoDeform Step2] visibility filter: dropped " << filteredOut
                  << " src vertices (hidden/outscreen)" << std::endl;
    }

    std::shared_ptr<Reg3DCustom::PointCloud> sourceCloud;
    if (srcVoxelSize > 0.0f) {
        sourceCloud = reg.voxelDownSample(sourceCloudFull, srcVoxelSize);
    } else {
        glm::vec3 mn(1e10f), mx(-1e10f);
        for (const auto& p : sourceCloudFull->points) {
            mn = glm::min(mn, p); mx = glm::max(mx, p);
        }
        float diag = glm::length(mx - mn);
        size_t Ntgt = targetCloud->size();
        float autoVox = (Ntgt > 0) ? diag / std::cbrt(static_cast<float>(Ntgt) * 2.0f) : diag * 0.05f;
        sourceCloud = reg.voxelDownSample(sourceCloudFull, autoVox);
    }
    if (!sourceCloud || sourceCloud->empty()) sourceCloud = sourceCloudFull;
    st.srcDownsampledCount = static_cast<int>(sourceCloud->size());

    Reg3DCustom::NanoflannAdaptor srcAdaptor(sourceCloud->points);
    auto srcTree = Reg3DCustom::buildKDTree(srcAdaptor);
    Reg3DCustom::NanoflannAdaptor tgtAdaptor(targetCloud->points);
    auto tgtTree = Reg3DCustom::buildKDTree(tgtAdaptor);

    float maxDistSq = maxDist * maxDist;

    st.correspondences.reserve(targetCloud->size());
    std::vector<float> distsAll, distsSil, distsInt;

    for (size_t i = 0; i < targetCloud->size(); i++) {
        const glm::vec3& tgt = targetCloud->points[i];

        std::vector<size_t> knnIdx; std::vector<float> knnD2;
        size_t found = Reg3DCustom::searchKNN(*srcTree, tgt, 2, knnIdx, knnD2);
        if (found < 1) continue;
        if (knnD2[0] > maxDistSq) continue;

        float d1 = std::sqrt(knnD2[0]);
        float d2 = (found >= 2) ? std::sqrt(knnD2[1]) : d1 * 10.0f;
        float ratio = (d2 > 1e-6f) ? (d1 / d2) : 0.0f;
        bool ratioOK = (ratio < ratioThreshold) || (found < 2);
        if (!ratioOK) st.nRatioRejected++;

        size_t nnSrcIdx = knnIdx[0];
        glm::vec3 src = sourceCloud->points[nnSrcIdx];

        size_t nnTgtIdx; float distSq2;
        bool bidir = false;
        if (Reg3DCustom::searchKNN1(*tgtTree, src, nnTgtIdx, distSq2)) {
            bidir = (nnTgtIdx == i);
        }

        glm::vec3 srcN = hasSrcNormals && sourceCloud->hasNormals() ? sourceCloud->normals[nnSrcIdx] : glm::vec3(0,0,1);
        glm::vec3 tgtN = hasTgtNormals ? targetCloud->normals[i]   : glm::vec3(0,0,1);
        glm::vec3 nAvg;
        if (hasSrcNormals && hasTgtNormals) {
            if (glm::dot(srcN, tgtN) < 0.0f) tgtN = -tgtN;
            nAvg = srcN + tgtN;
            float ln = glm::length(nAvg);
            if (ln > 1e-6f) nAvg /= ln;
            else nAvg = srcN;
        } else if (hasSrcNormals) nAvg = srcN;
        else if (hasTgtNormals)   nAvg = tgtN;
        else                      nAvg = glm::vec3(0, 0, 1);

        float d = d1;
        float dProj = std::fabs(glm::dot(tgt - src, nAvg));
        float bd    = hasBd ? targetCloud->boundaryDist[i] : 9999.0f;
        bool isSil  = (bd < silhouetteThreshold);

        Correspondence c;
        c.srcIdx          = static_cast<int>(nnSrcIdx);
        c.srcPos          = src;
        c.srcNormal       = srcN;
        c.tgtPos          = tgt;
        c.tgtNormal       = tgtN;
        c.dist            = d;
        c.distProjected   = dProj;
        c.boundaryDist    = bd;
        c.ratio           = ratio;
        c.ratioOK         = ratioOK;
        c.isSilhouette    = isSil;
        c.bidirConsistent = bidir;
        c.category        = CAT_NONE;
        st.correspondences.push_back(c);

        distsAll.push_back(dProj);
        if (isSil) distsSil.push_back(dProj);
        else       distsInt.push_back(dProj);
        if (!bidir) st.nBidirInconsistent++;
    }

    st.statsAll        = computeStatsFromValues(distsAll);
    st.statsSilhouette = computeStatsFromValues(distsSil);
    st.statsInterior   = computeStatsFromValues(distsInt);
    st.stage = 2;

    int total = static_cast<int>(st.correspondences.size());
    std::cout << "\n=== [AutoDeform Step2] Correspondence Extraction ===" << std::endl;
    std::cout << "  source (full)   : " << sourceCloudFull->size() << std::endl;
    std::cout << "  source (downsmp): " << st.srcDownsampledCount << std::endl;
    std::cout << "  target points   : " << targetCloud->size() << std::endl;
    std::cout << "  correspondences : " << total << std::endl;
    std::cout << "  silhouette thr  : " << silhouetteThreshold << std::endl;
    std::cout << "  ratio threshold : " << ratioThreshold << std::endl;
    std::cout << "  ratio rejected  : " << st.nRatioRejected
              << " (" << (total > 0 ? 100.0f * st.nRatioRejected / total : 0.0f) << "%)" << std::endl;
    std::cout << "  bidir inconsist : " << st.nBidirInconsistent
              << " (" << (total > 0 ? 100.0f * st.nBidirInconsistent / total : 0.0f) << "%)" << std::endl;
    std::cout << "  -- ALL (projected distance) --" << std::endl;
    std::cout << "    count=" << st.statsAll.count
              << " median=" << st.statsAll.median
              << " MAD=" << st.statsAll.mad
              << " mean=" << st.statsAll.mean
              << " [" << st.statsAll.minV << ", " << st.statsAll.maxV << "]" << std::endl;
    std::cout << "  -- SILHOUETTE (projected distance) --" << std::endl;
    std::cout << "    count=" << st.statsSilhouette.count
              << " median=" << st.statsSilhouette.median
              << " MAD=" << st.statsSilhouette.mad
              << " mean=" << st.statsSilhouette.mean
              << " [" << st.statsSilhouette.minV << ", " << st.statsSilhouette.maxV << "]" << std::endl;
    std::cout << "  -- INTERIOR (projected distance) --" << std::endl;
    std::cout << "    count=" << st.statsInterior.count
              << " median=" << st.statsInterior.median
              << " MAD=" << st.statsInterior.mad
              << " mean=" << st.statsInterior.mean
              << " [" << st.statsInterior.minV << ", " << st.statsInterior.maxV << "]" << std::endl;
    std::cout << "==========================================================\n" << std::endl;
}

inline int refreshCorrespondencesAfterDeform(
    State& st,
    const std::vector<float>& visVerts,
    const std::vector<float>& origVerts)
{
    if (st.correspondences.empty()) return 0;
    if (visVerts.empty() || visVerts.size() != origVerts.size()) return 0;
    size_t N = visVerts.size() / 3;
    if (N == 0) return 0;

    std::vector<glm::vec3> origPts(N);
    for (size_t i = 0; i < N; i++)
        origPts[i] = glm::vec3(origVerts[i*3], origVerts[i*3+1], origVerts[i*3+2]);

    Reg3DCustom::NanoflannAdaptor adaptor(origPts);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    int updated = 0;
    for (auto& c : st.correspondences) {
        size_t nnIdx; float distSq;
        if (!Reg3DCustom::searchKNN1(*tree, c.srcPos, nnIdx, distSq)) continue;
        glm::vec3 newSrc(visVerts[nnIdx*3], visVerts[nnIdx*3+1], visVerts[nnIdx*3+2]);
        c.srcPos = newSrc;
        c.dist = glm::length(c.tgtPos - newSrc);
        glm::vec3 n = c.srcNormal + c.tgtNormal;
        float ln = glm::length(n);
        if (ln > 1e-6f) n /= ln;
        else n = c.srcNormal;
        c.distProjected = std::abs(glm::dot(c.tgtPos - newSrc, n));
        updated++;
    }

    std::vector<float> distAll, distSil, distInt;
    distAll.reserve(st.correspondences.size());
    for (const auto& c : st.correspondences) {
        if (!c.ratioOK) continue;
        distAll.push_back(c.distProjected);
        if (c.isSilhouette) distSil.push_back(c.distProjected);
        else                distInt.push_back(c.distProjected);
    }
    st.statsAll        = computeStatsFromValues(distAll);
    st.statsSilhouette = computeStatsFromValues(distSil);
    st.statsInterior   = computeStatsFromValues(distInt);

    std::cout << "[AutoDeform] refreshed " << updated << "/" << st.correspondences.size()
              << " correspondences from current VisMesh"
              << "  median(all)=" << st.statsAll.median << std::endl;
    return updated;
}

inline void classify(
    State& st,
    float inlierMadScale  = 1.0f,
    float outlierMadScale = 3.0f,
    bool  dropByRatio = true)
{
    if (st.correspondences.empty()) {
        std::cerr << "[AutoDeform::Phase0 Step2] no correspondences" << std::endl;
        return;
    }

    st.thrInlierSil  = st.statsSilhouette.median + inlierMadScale  * st.statsSilhouette.mad;
    st.thrOutlierSil = st.statsSilhouette.median + outlierMadScale * st.statsSilhouette.mad;
    st.thrInlierInt  = st.statsInterior.median   + inlierMadScale  * st.statsInterior.mad;
    st.thrOutlierInt = st.statsInterior.median   + outlierMadScale * st.statsInterior.mad;

    st.nSilInlier = st.nSilMover = st.nSilOutlier = 0;
    st.nIntInlier = st.nIntMover = st.nIntOutlier = 0;

    for (auto& c : st.correspondences) {
        if (dropByRatio && !c.ratioOK) {
            c.category = CAT_OUTLIER;
            if (c.isSilhouette) st.nSilOutlier++; else st.nIntOutlier++;
            continue;
        }
        float thrIn  = c.isSilhouette ? st.thrInlierSil  : st.thrInlierInt;
        float thrOut = c.isSilhouette ? st.thrOutlierSil : st.thrOutlierInt;
        float d = c.distProjected;

        if (d < thrIn) {
            c.category = CAT_INLIER;
            if (c.isSilhouette) st.nSilInlier++; else st.nIntInlier++;
        } else if (d < thrOut) {
            c.category = CAT_MOVER;
            if (c.isSilhouette) st.nSilMover++; else st.nIntMover++;
        } else {
            c.category = CAT_OUTLIER;
            if (c.isSilhouette) st.nSilOutlier++; else st.nIntOutlier++;
        }
    }
    st.stage = 3;

    int total   = static_cast<int>(st.correspondences.size());
    int totSil  = st.nSilInlier + st.nSilMover + st.nSilOutlier;
    int totInt  = st.nIntInlier + st.nIntMover + st.nIntOutlier;
    auto pct = [](int n, int t) { return t > 0 ? 100.0f * n / t : 0.0f; };

    std::cout << "\n=== [AutoDeform Step3] Dual Classification ===" << std::endl;
    std::cout << "  inlier  MAD scale : " << inlierMadScale << std::endl;
    std::cout << "  outlier MAD scale : " << outlierMadScale << std::endl;
    std::cout << "  drop by ratio     : " << (dropByRatio ? "YES" : "NO") << std::endl;
    std::cout << "  -- SILHOUETTE (total=" << totSil << ") --" << std::endl;
    std::cout << "    thr inlier  : " << st.thrInlierSil << std::endl;
    std::cout << "    thr outlier : " << st.thrOutlierSil << std::endl;
    std::cout << "    inlier  : " << st.nSilInlier  << " (" << pct(st.nSilInlier,  totSil) << "%)" << std::endl;
    std::cout << "    mover   : " << st.nSilMover   << " (" << pct(st.nSilMover,   totSil) << "%)" << std::endl;
    std::cout << "    outlier : " << st.nSilOutlier << " (" << pct(st.nSilOutlier, totSil) << "%)" << std::endl;
    std::cout << "  -- INTERIOR (total=" << totInt << ") --" << std::endl;
    std::cout << "    thr inlier  : " << st.thrInlierInt << std::endl;
    std::cout << "    thr outlier : " << st.thrOutlierInt << std::endl;
    std::cout << "    inlier  : " << st.nIntInlier  << " (" << pct(st.nIntInlier,  totInt) << "%)" << std::endl;
    std::cout << "    mover   : " << st.nIntMover   << " (" << pct(st.nIntMover,   totInt) << "%)" << std::endl;
    std::cout << "    outlier : " << st.nIntOutlier << " (" << pct(st.nIntOutlier, totInt) << "%)" << std::endl;
    std::cout << "=====================================================\n" << std::endl;
    (void)total;
}

inline glm::vec3 distanceToColor(float dist, float vmin, float vmax) {
    float t = (vmax > vmin) ? (dist - vmin) / (vmax - vmin) : 0.0f;
    t = std::max(0.0f, std::min(1.0f, t));
    if (t < 0.5f) {
        float u = t * 2.0f;
        return glm::vec3(0.0f, u, 1.0f - u);
    } else {
        float u = (t - 0.5f) * 2.0f;
        return glm::vec3(u, 1.0f - u, 0.0f);
    }
}

inline glm::vec3 categoryColor(int cat, bool isSilhouette) {
    if (isSilhouette) {
        switch (cat) {
        case CAT_INLIER:  return glm::vec3(0.0f, 1.0f, 1.0f);
        case CAT_MOVER:   return glm::vec3(1.0f, 0.0f, 1.0f);
        case CAT_OUTLIER: return glm::vec3(0.6f, 0.0f, 0.0f);
        default:          return glm::vec3(0.5f);
        }
    } else {
        switch (cat) {
        case CAT_INLIER:  return glm::vec3(0.1f, 0.8f, 0.1f);
        case CAT_MOVER:   return glm::vec3(0.9f, 0.8f, 0.0f);
        case CAT_OUTLIER: return glm::vec3(0.5f, 0.2f, 0.2f);
        default:          return glm::vec3(0.5f);
        }
    }
}

inline void drawStep1(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float srcRadius = 0.04f,
    float tgtRadius = 0.04f,
    int   maxPoints = 2000)
{
    if (st.stage != 2 || st.correspondences.empty()) return;
    float vmin = st.statsAll.minV;
    float vmax = st.statsAll.median + 3.0f * st.statsAll.mad;
    if (vmax <= vmin) vmax = st.statsAll.maxV;

    // maxPoints<=0 のときは間引き無し(全対応点を描画)
    size_t N = st.correspondences.size();
    int step = (maxPoints > 0) ? std::max<int>(1, static_cast<int>(N) / maxPoints) : 1;

    for (size_t i = 0; i < N; i += step) {
        const auto& c = st.correspondences[i];
        glm::vec3 col = distanceToColor(c.distProjected, vmin, vmax);
        float rSrc = c.isSilhouette ? srcRadius * 1.3f : srcRadius;
        float rTgt = c.isSilhouette ? tgtRadius * 1.3f : tgtRadius;
        sphere.draw(shader, c.srcPos, col, rSrc, view, proj, camPos);
        sphere.draw(shader, c.tgtPos, col * 0.6f, rTgt, view, proj, camPos);
    }
}

inline void drawStep2(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float srcRadius = 0.05f,
    float tgtRadius = 0.04f,
    bool  drawOutlier = true,
    int   maxPoints = 2000)
{
    if (st.stage != 3 || st.correspondences.empty()) return;

    // maxPoints<=0 のときは間引き無し(全対応点を描画)
    size_t N = st.correspondences.size();
    int step = (maxPoints > 0) ? std::max<int>(1, static_cast<int>(N) / maxPoints) : 1;

    for (size_t i = 0; i < N; i += step) {
        const auto& c = st.correspondences[i];
        if (!drawOutlier && c.category == CAT_OUTLIER) continue;
        glm::vec3 col = categoryColor(c.category, c.isSilhouette);
        float rSrc = c.isSilhouette ? srcRadius * 1.3f : srcRadius;
        float rTgt = c.isSilhouette ? tgtRadius * 1.3f : tgtRadius;
        sphere.draw(shader, c.srcPos, col, rSrc, view, proj, camPos);
        sphere.draw(shader, c.tgtPos, col * 0.5f, rTgt, view, proj, camPos);
    }
}

inline void laplacianSmoothField(
    std::vector<float>& values,
    const std::vector<int>& triIds,
    int    iterations = 10,
    float  alpha      = 0.5f,
    const std::vector<bool>* validMask = nullptr)
{
    size_t N = values.size();
    if (N == 0 || triIds.empty()) return;
    bool useMask = (validMask && validMask->size() == N);

    std::vector<std::vector<int>> adj(N);
    std::vector<std::set<int>> adjSet(N);
    for (size_t i = 0; i + 2 < triIds.size(); i += 3) {
        int a = triIds[i], b = triIds[i+1], c = triIds[i+2];
        if (a < 0 || b < 0 || c < 0) continue;
        if ((size_t)a >= N || (size_t)b >= N || (size_t)c >= N) continue;
        adjSet[a].insert(b); adjSet[a].insert(c);
        adjSet[b].insert(a); adjSet[b].insert(c);
        adjSet[c].insert(a); adjSet[c].insert(b);
    }
    for (size_t i = 0; i < N; i++) adj[i].assign(adjSet[i].begin(), adjSet[i].end());

    std::vector<float> work = values;
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < N; i++) {
            if (useMask && !(*validMask)[i]) { work[i] = values[i]; continue; }
            if (adj[i].empty())              { work[i] = values[i]; continue; }
            double sum = 0.0; int cnt = 0;
            for (int nb : adj[i]) {
                if (useMask && !(*validMask)[nb]) continue;
                sum += values[nb]; cnt++;
            }
            if (cnt == 0) { work[i] = values[i]; continue; }
            float avg = static_cast<float>(sum / cnt);
            work[i] = (1.0f - alpha) * values[i] + alpha * avg;
        }
        values.swap(work);
    }
}

inline void computeFieldOnVisMesh(
    State& st,
    const std::vector<float>& visVerts,
    const std::vector<int>&   visTriIds,
    int   kNearest = 8,
    float gaussianSigma = 0.0f,
    bool  useOnlyRatioOK = true,
    int   smoothIterations = 10,
    float smoothAlpha = 0.5f,
    bool  dropOutlier = false)  // 2-way 時のみ true: CAT_OUTLIER を field 寄与から除外
{
    if (st.correspondences.empty()) {
        std::cerr << "[AutoDeform::Step3a] run Step1 first" << std::endl;
        return;
    }
    if (visVerts.empty() || visVerts.size() % 3 != 0) {
        std::cerr << "[AutoDeform::Step3a] invalid visVerts" << std::endl;
        return;
    }

    std::vector<glm::vec3> corrSrcPts;
    std::vector<float>     corrScores;
    corrSrcPts.reserve(st.correspondences.size());
    corrScores.reserve(st.correspondences.size());
    int nDroppedOutlier = 0;
    for (const auto& c : st.correspondences) {
        if (useOnlyRatioOK && !c.ratioOK) continue;
        // 2-way モード (dropOutlier=true) のときだけ、classify で OUTLIER 判定された
        // 遠い対応点を field 計算から除外する。
        //   3-way (dropOutlier=false, 既定) : 全 ratioOK 点で field を作る → handle は
        //          フルセット由来 (green/red=68/73, move[0]=(-0.12108,...))。本来の3-way。
        //   2-way (dropOutlier=true)        : OUTLIER を抜く → field が INLIER/MOVER のみで
        //          構築され handle 配置が変わる (green/red=51/87, move[0]=(0.25352,...))。
        //   classify 未実行時は category=CAT_NONE なのでここは発火せず全点が入る (後方互換)。
        if (dropOutlier && c.category == CAT_OUTLIER) { nDroppedOutlier++; continue; }
        corrSrcPts.push_back(c.srcPos);
        corrScores.push_back(c.distProjected);
    }
    if (corrSrcPts.empty()) {
        std::cerr << "[AutoDeform::Step3a] no valid correspondences after filter" << std::endl;
        return;
    }

    glm::vec3 bbMin(1e10f), bbMax(-1e10f);
    for (const auto& p : corrSrcPts) { bbMin = glm::min(bbMin, p); bbMax = glm::max(bbMax, p); }
    float diag = glm::length(bbMax - bbMin);
    float sigma = (gaussianSigma > 0.0f) ? gaussianSigma : diag * 0.05f;
    float sigma2 = sigma * sigma;
    st.fieldBboxDiag = diag;

    Reg3DCustom::NanoflannAdaptor corrAdaptor(corrSrcPts);
    auto corrTree = Reg3DCustom::buildKDTree(corrAdaptor);

    size_t Nvis = visVerts.size() / 3;
    st.fieldVertexPos.resize(Nvis);
    st.fieldScoreRaw.assign(Nvis, 0.0f);
    int K = std::min<int>(kNearest, static_cast<int>(corrSrcPts.size()));

    st.visMeshValidMask.assign(Nvis, true);
    st.nVisMeshValid = static_cast<int>(Nvis);
    st.nVisMeshMasked = 0;
    if (st.srcVisibility.size() == Nvis) {
        st.nVisMeshValid = 0;
        for (size_t i = 0; i < Nvis; i++) {
            bool ok = (st.srcVisibility[i] == VIS_VISIBLE);
            st.visMeshValidMask[i] = ok;
            if (ok) st.nVisMeshValid++;
            else    st.nVisMeshMasked++;
        }
    }

    for (size_t i = 0; i < Nvis; i++) {
        glm::vec3 v(visVerts[i*3], visVerts[i*3+1], visVerts[i*3+2]);
        st.fieldVertexPos[i] = v;
        if (!st.visMeshValidMask[i]) continue;

        std::vector<size_t> knnIdx; std::vector<float> knnD2;
        Reg3DCustom::searchKNN(*corrTree, v, K, knnIdx, knnD2);

        double wSum = 0.0, sSum = 0.0;
        for (size_t j = 0; j < knnIdx.size(); j++) {
            float w = std::exp(-knnD2[j] / (2.0f * sigma2));
            wSum += w;
            sSum += w * corrScores[knnIdx[j]];
        }
        st.fieldScoreRaw[i] = (wSum > 1e-12) ? static_cast<float>(sSum / wSum) : 0.0f;
    }

    st.fieldScoreSmooth = st.fieldScoreRaw;
    if (smoothIterations > 0 && !visTriIds.empty()) {
        laplacianSmoothField(st.fieldScoreSmooth, visTriIds, smoothIterations, smoothAlpha,
                             &st.visMeshValidMask);
    }

    auto computeStats = [&](const std::vector<float>& vals, float& mn, float& md, float& mx) {
        std::vector<float> v;
        v.reserve(vals.size());
        for (size_t i = 0; i < vals.size(); i++) {
            if (!st.visMeshValidMask[i]) continue;
            v.push_back(vals[i]);
        }
        if (v.empty()) { mn = md = mx = 0.0f; return; }
        std::sort(v.begin(), v.end());
        mn = v.front(); mx = v.back(); md = v[v.size() / 2];
    };

    float rawMin, rawMax, rawMedian;
    computeStats(st.fieldScoreRaw, rawMin, rawMedian, rawMax);
    computeStats(st.fieldScoreSmooth, st.fieldMin, st.fieldMedian, st.fieldMax);
    st.fieldReady  = true;
    st.stage = 4;

    std::cout << "\n=== [AutoDeform Step4] Continuous Field on VisMesh ===" << std::endl;
    std::cout << "  vis vertices       : " << Nvis << std::endl;
    std::cout << "    valid (visible)  : " << st.nVisMeshValid << std::endl;
    std::cout << "    masked (hidden)  : " << st.nVisMeshMasked << std::endl;
    std::cout << "  valid corresp used : " << corrSrcPts.size()
              << "  (dropped outlier=" << nDroppedOutlier << ")" << std::endl;
    std::cout << "  bbox diag          : " << diag << std::endl;
    std::cout << "  gaussian sigma     : " << sigma << std::endl;
    std::cout << "  K nearest          : " << K << std::endl;
    std::cout << "  smooth iterations  : " << smoothIterations
              << " (alpha=" << smoothAlpha << ")" << std::endl;
    std::cout << "  raw    min/med/max : " << rawMin << " / " << rawMedian << " / " << rawMax << std::endl;
    std::cout << "  smooth min/med/max : " << st.fieldMin << " / " << st.fieldMedian << " / " << st.fieldMax << std::endl;
    std::cout << "=======================================================\n" << std::endl;
}

inline void drawFieldPoints(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float radius = 0.03f,
    int   maxPoints = 500)
{
    if (!st.fieldReady || st.fieldVertexPos.empty()) return;

    float vmin = st.fieldMin;
    float vmax = st.fieldMedian * 2.0f;
    if (vmax <= vmin) vmax = st.fieldMax;

    size_t N = st.fieldVertexPos.size();
    int step = std::max<int>(1, static_cast<int>(N) / maxPoints);
    bool useMask = (st.visMeshValidMask.size() == N);

    const std::vector<float>& src = st.fieldScoreSmooth.empty() ? st.fieldScoreRaw : st.fieldScoreSmooth;
    for (size_t i = 0; i < N; i += step) {
        if (useMask && !st.visMeshValidMask[i]) continue;
        glm::vec3 col = distanceToColor(src[i], vmin, vmax);
        sphere.draw(shader, st.fieldVertexPos[i], col, radius, view, proj, camPos);
    }
}

inline std::vector<int> farthestPointSampling(
    const std::vector<glm::vec3>& pts,
    const std::vector<int>& candidateIdx,
    int K,
    int seed = 0)
{
    std::vector<int> result;
    if (candidateIdx.empty() || K <= 0) return result;
    int firstIdx = candidateIdx[seed % static_cast<int>(candidateIdx.size())];
    result.push_back(firstIdx);

    std::vector<float> minDist(candidateIdx.size(), std::numeric_limits<float>::max());
    for (size_t i = 0; i < candidateIdx.size(); i++) {
        const glm::vec3& p = pts[candidateIdx[i]];
        float d2 = glm::length(p - pts[firstIdx]);
        minDist[i] = d2 * d2;
    }

    while (static_cast<int>(result.size()) < K && result.size() < candidateIdx.size()) {
        float maxD = -1.0f;
        int   bestI = -1;
        for (size_t i = 0; i < candidateIdx.size(); i++) {
            if (minDist[i] > maxD) {
                maxD = minDist[i];
                bestI = static_cast<int>(i);
            }
        }
        if (bestI < 0 || maxD <= 1e-12f) break;
        int chosen = candidateIdx[bestI];
        result.push_back(chosen);
        for (size_t i = 0; i < candidateIdx.size(); i++) {
            const glm::vec3& p = pts[candidateIdx[i]];
            float d2 = glm::length(p - pts[chosen]);
            d2 = d2 * d2;
            if (d2 < minDist[i]) minDist[i] = d2;
        }
    }
    return result;
}

inline void generateHandles(
    State& st,
    int   K_fix       = 4,
    int   K_move      = 3,
    float tauLowScale = 0.5f,
    float tauHighScale= 1.5f,
    float rFixScale   = 0.08f,
    float rMoveScale  = 0.06f,
    float minSepScale = 0.15f)
{
    st.fixHandles.clear();
    st.moveHandles.clear();
    if (!st.fieldReady || st.fieldVertexPos.empty() || st.correspondences.empty()) {
        std::cerr << "[AutoDeform::Step3b] field not ready" << std::endl;
        return;
    }

    float tauLow  = st.fieldMedian * tauLowScale;
    float tauHigh = st.fieldMedian * tauHighScale;
    float diag = (st.fieldBboxDiag > 0.0f) ? st.fieldBboxDiag : 1.0f;
    float rFix    = diag * rFixScale;
    float rMove   = diag * rMoveScale;
    float minSep  = diag * minSepScale;

    std::vector<int> greenSet, redSet;
    size_t N = st.fieldVertexPos.size();
    bool useMask = (st.visMeshValidMask.size() == N);
    const std::vector<float>& src = st.fieldScoreSmooth.empty() ? st.fieldScoreRaw : st.fieldScoreSmooth;
    for (size_t i = 0; i < N; i++) {
        if (useMask && !st.visMeshValidMask[i]) continue;
        float s = src[i];
        if (s < tauLow)  greenSet.push_back(static_cast<int>(i));
        if (s > tauHigh) redSet.push_back(static_cast<int>(i));
    }

    auto fixSel = farthestPointSampling(st.fieldVertexPos, greenSet, K_fix, 0);

    std::vector<int> redSorted = redSet;
    std::sort(redSorted.begin(), redSorted.end(),
              [&src](int a, int b) { return src[a] > src[b]; });
    auto moveSel = farthestPointSampling(st.fieldVertexPos, redSorted, K_move, 0);

    std::vector<glm::vec3> corrSrcPts, corrTgtPts;
    for (const auto& c : st.correspondences) {
        if (!c.ratioOK) continue;
        corrSrcPts.push_back(c.srcPos);
        corrTgtPts.push_back(c.tgtPos);
    }

    for (int idx : fixSel) {
        Handle h;
        h.center  = st.fieldVertexPos[idx];
        h.target  = h.center;
        h.radius  = rFix;
        h.isFixed = true;
        st.fixHandles.push_back(h);
    }

    for (int idx : moveSel) {
        glm::vec3 center = st.fieldVertexPos[idx];
        glm::vec3 tgtSum(0.0f);
        float     wSum = 0.0f;
        float sigma = rMove;
        float sigma2 = sigma * sigma;
        for (size_t j = 0; j < corrSrcPts.size(); j++) {
            float d2 = glm::length(corrSrcPts[j] - center);
            d2 = d2 * d2;
            float w = std::exp(-d2 / (2.0f * sigma2));
            tgtSum += corrTgtPts[j] * w;
            wSum += w;
        }
        glm::vec3 target = (wSum > 1e-6f) ? (tgtSum / wSum) : center;

        Handle h;
        h.center  = center;
        h.target  = target;
        h.radius  = rMove;
        h.isFixed = false;
        st.moveHandles.push_back(h);
    }

    auto dropTooClose = [&](std::vector<Handle>& list, const std::vector<Handle>& other) {
        std::vector<Handle> kept;
        for (const auto& h : list) {
            bool tooClose = false;
            for (const auto& o : other) {
                if (glm::length(h.center - o.center) < minSep) { tooClose = true; break; }
            }
            if (!tooClose) kept.push_back(h);
        }
        list = kept;
    };
    dropTooClose(st.fixHandles,  st.moveHandles);
    dropTooClose(st.moveHandles, st.fixHandles);

    st.stage = 5;

    std::cout << "\n=== [AutoDeform Step5] Handle Candidate Generation ===" << std::endl;
    std::cout << "  bbox diag      : " << diag << std::endl;
    std::cout << "  tau low/high   : " << tauLow << " / " << tauHigh << std::endl;
    std::cout << "  green set size : " << greenSet.size() << std::endl;
    std::cout << "  red set size   : " << redSet.size() << std::endl;
    std::cout << "  fix handles    : " << st.fixHandles.size()
              << " (radius=" << rFix << ")" << std::endl;
    std::cout << "  move handles   : " << st.moveHandles.size()
              << " (radius=" << rMove << ")" << std::endl;
    std::cout << "  min separation : " << minSep << std::endl;
    std::cout << "=======================================================\n" << std::endl;
}

inline void drawHandles(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos)
{
    if (st.stage < 5) return;

    for (const auto& h : st.fixHandles) {
        sphere.draw(shader, h.center, glm::vec3(0.2f, 0.5f, 1.0f), h.radius * 0.5f, view, proj, camPos);
    }
}

inline int applyHandlesToSoftBody(
    const State& st,
    SoftBody* body,
    float fixCompliance  = 0.0f,
    float moveCompliance = 1e-7f)
{
    if (!body) return 0;
    if (st.fixHandles.empty() && st.moveHandles.empty()) return 0;

    body->clearAttachmentConstraints();

    const auto& positions = body->getPositions();
    size_t Npart = body->getNumParticles();
    int nAdded = 0;

    auto addWithinRadius = [&](const Handle& h, float compliance, bool anchorAtCurrent) {
        int cnt = 0;
        for (size_t i = 0; i < Npart; i++) {
            glm::vec3 p(positions[i*3], positions[i*3+1], positions[i*3+2]);
            float d = glm::length(p - h.center);
            if (d > h.radius) continue;
            glm::vec3 tgt;
            if (anchorAtCurrent) {
                tgt = p;
            } else {
                glm::vec3 offset = p - h.center;
                tgt = h.target + offset;
            }
            body->addAttachmentConstraint(static_cast<int>(i), tgt, compliance);
            cnt++;
        }
        return cnt;
    };

    int nFixVerts = 0, nMoveVerts = 0;
    for (const auto& h : st.fixHandles) {
        nFixVerts += addWithinRadius(h, fixCompliance, true);
    }
    for (const auto& h : st.moveHandles) {
        nMoveVerts += addWithinRadius(h, moveCompliance, false);
    }
    nAdded = nFixVerts + nMoveVerts;

    std::cout << "\n=== [AutoDeform Step3c] Apply Handles to SoftBody ===" << std::endl;
    std::cout << "  fix handles       : " << st.fixHandles.size() << std::endl;
    std::cout << "  move handles      : " << st.moveHandles.size() << std::endl;
    std::cout << "  fix vert attached : " << nFixVerts
              << " (compliance=" << fixCompliance << ")" << std::endl;
    std::cout << "  move vert attached: " << nMoveVerts
              << " (compliance=" << moveCompliance << ")" << std::endl;
    std::cout << "  total attachments : " << body->getNumAttachments() << std::endl;
    std::cout << "======================================================\n" << std::endl;

    return nAdded;
}

// ============================================================================
// 描画ディスパッチャ — スフィア半径を bbox 比例に正規化
// ----------------------------------------------------------------------------
// scale=10 時代のハードコード半径(0.04, 0.05, 0.03 等)は bbox≈6.3 の世界で
// 「bbox の 0.5-0.8% 程度」という想定だった。scale=1 になって bbox が 0.63
// になると同じ絶対値では 6-8% で太すぎる。
//
// st.srcBboxDiag(Step1 で計測)が有効なら、kRefBbox = 6.33 を基準とした
// 比率で各サブ関数のデフォルト半径をスケールする。bbox 未計測の場合は
// 旧来の絶対値で動く(後方互換)。
//
// さらに視覚バランスのため:
//  - Stage 2/3 は対応点が 87,000+ あって全部描くと塗りつぶされるので
//    maxPoints=2000 で間引き(Step 5 と同じ思想)
//  - 全体の半径も控えめ(kVizShrink)に絞ってある
// ============================================================================
inline void draw(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos)
{
    constexpr float kRefBbox    = 6.33f;   // scale=10 時代の典型的な liver bbox diag
    constexpr float kVizShrink  = 0.5f;    // 旧基準の半分にして「点」感を出す
    constexpr float kDotRadius  = 0.04f;   // 全 stage 統一の基本半径(旧 scale=10)

    const float r = (st.srcBboxDiag > 1e-6f) ? (st.srcBboxDiag / kRefBbox) : 1.0f;
    const float dot = kDotRadius * r * kVizShrink;

    // SHIFT+Key1/2/3 で debugMaxPoints=1 → 1個だけ描画(サイズ比較デバッグ)
    const bool single = (st.debugMaxPoints > 0);
    const int mp1 = single ? 1 : 0;       // stage 1: 0=全部
    const int mp2 = single ? 1 : 0;       // stage 2/3: 0=全部 (間引き廃止)
    const int mp4 = single ? 1 : 500;     // stage 4
    const int mp5 = single ? 1 : 300;     // stage 5

    // ================================================================
    // SHIFT+Key デバッグ: 黄色の1個だけ描画してサイズ比較する
    // ================================================================
    if (st.debugMaxPoints > 0) {
        glm::vec3 pos(0);
        bool found = false;
        const char* label = "?";

        if (st.stage == 1 && !st.srcPosCache.empty()) {
            for (size_t i = 0; i < st.srcPosCache.size(); i++) {
                if (st.srcVisibility[i] == VIS_VISIBLE)
                { pos = st.srcPosCache[i]; found = true; break; }
            }
            label = "Stage1(srcVis)";
        } else if ((st.stage == 2 || st.stage == 3) && !st.correspondences.empty()) {
            pos = st.correspondences[0].srcPos; found = true;
            label = (st.stage == 2) ? "Stage2(corr)" : "Stage3(class)";
        }

        if (found) {
            sphere.draw(shader, pos, glm::vec3(1.0f, 1.0f, 0.0f), dot,
                        view, proj, camPos);
            static int dbgLastStage = -1;
            if (dbgLastStage != st.stage) {
                std::cout << "[debug] " << label
                          << "  dot=" << dot
                          << "  bbox=" << st.srcBboxDiag
                          << "  r=" << r
                          << "  pos=(" << pos.x << "," << pos.y << "," << pos.z << ")"
                          << std::endl;
                dbgLastStage = st.stage;
            }
        }
        return;
    }

    // ================================================================
    // 通常描画 — 全 stage 統一 dot サイズ
    // debugMaxPoints>0 時は上のブロックで return 済み
    // ================================================================
    if (st.stage == 1) {
        drawSrcVisibility(st, sphere, shader, view, proj, camPos,
                          /*srcRadius=*/dot, /*tgtRadius=*/dot,
                          /*drawHidden=*/false, /*drawOutscreen=*/false,
                          /*maxPoints=*/mp1);
    } else if (st.stage == 2) {
        drawStep1(st, sphere, shader, view, proj, camPos,
                  /*srcRadius=*/dot, /*tgtRadius=*/dot, /*maxPoints=*/mp2);
    } else if (st.stage == 3) {
        drawStep2(st, sphere, shader, view, proj, camPos,
                  /*srcRadius=*/dot, /*tgtRadius=*/dot,
                  /*drawOutlier=*/true, /*maxPoints=*/mp2);
    } else if (st.stage == 4) {
        drawFieldPoints(st, sphere, shader, view, proj, camPos,
                        /*radius=*/dot * 0.75f, mp4);
    } else if (st.stage == 5) {
        drawFieldPoints(st, sphere, shader, view, proj, camPos,
                        /*radius=*/dot * 0.5f, /*maxPoints=*/mp5);
        drawHandles(st, sphere, shader, view, proj, camPos);
    }
}

inline float measureRMSE(
    State& st,
    const std::vector<float>& visVerts,
    mCutMesh* screenMesh,
    int gridW, int gridH,
    float depthScale,
    bool  storeAsBefore = false,
    float maxDist = 1.0f,
    bool  verbose = true)
{
    if (!screenMesh || visVerts.empty() || visVerts.size() % 3 != 0) {
        std::cerr << "[AutoDeform::measureRMSE] invalid input" << std::endl;
        return -1.0f;
    }

    Reg3DCustom::NoOpen3DRegistration reg;
    float zThresh = std::max(0.01f, depthScale * 0.05f);

    std::streambuf* oldBuf = nullptr;
    std::ostringstream devNull;
    if (!verbose) oldBuf = std::cout.rdbuf(devNull.rdbuf());
    auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridW, gridH, zThresh);
    if (!verbose && oldBuf) std::cout.rdbuf(oldBuf);

    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[AutoDeform::measureRMSE] target empty" << std::endl;
        return -1.0f;
    }

    std::vector<glm::vec3> srcPts;
    srcPts.reserve(visVerts.size() / 3);
    for (size_t i = 0; i + 2 < visVerts.size(); i += 3) {
        srcPts.emplace_back(visVerts[i], visVerts[i+1], visVerts[i+2]);
    }

    Reg3DCustom::NanoflannAdaptor srcAdaptor(srcPts);
    auto tree = Reg3DCustom::buildKDTree(srcAdaptor);

    float maxDistSq = maxDist * maxDist;
    float sumSq = 0.0f, sumErr = 0.0f, maxErr = 0.0f;
    int   matched = 0;
    for (size_t i = 0; i < targetCloud->size(); i++) {
        size_t nnIdx; float distSq;
        if (!Reg3DCustom::searchKNN1(*tree, targetCloud->points[i], nnIdx, distSq)) continue;
        if (distSq > maxDistSq) continue;
        float d = std::sqrt(distSq);
        sumSq  += distSq;
        sumErr += d;
        if (d > maxErr) maxErr = d;
        matched++;
    }
    float rmse = (matched > 0) ? std::sqrt(sumSq / matched) : -1.0f;
    float avg  = (matched > 0) ? (sumErr / matched) : 0.0f;

    st.rmseLastMeasured = rmse;
    st.rmseLastMatched  = matched;
    st.rmseLastTotal    = static_cast<int>(targetCloud->size());
    if (storeAsBefore) st.rmseBeforeDeform = rmse;

    if (verbose) {
        std::cout << "\n=== [AutoDeform RMSE on VisMesh 0] ===" << std::endl;
        std::cout << "  target points : " << targetCloud->size() << std::endl;
        std::cout << "  vis vertices  : " << srcPts.size() << std::endl;
        std::cout << "  matched       : " << matched << std::endl;
        std::cout << "  RMSE          : " << rmse << std::endl;
        std::cout << "  AvgErr        : " << avg << std::endl;
        std::cout << "  MaxErr        : " << maxErr << std::endl;
        if (st.rmseBeforeDeform > 0.0f && !storeAsBefore) {
            float delta = rmse - st.rmseBeforeDeform;
            float pct = (st.rmseBeforeDeform > 1e-6f) ? (100.0f * delta / st.rmseBeforeDeform) : 0.0f;
            std::cout << "  before deform : " << st.rmseBeforeDeform << std::endl;
            std::cout << "  delta         : " << delta << "  (" << pct << "%)" << std::endl;
            std::cout << "  status        : " << (delta < 0.0f ? "IMPROVED" : "WORSE") << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }
    return rmse;
}

inline void initSequentialQueue(
    State& st,
    const std::vector<float>& origVisVerts)
{
    st.seqMoveQueue.clear();
    st.seqMoveUsed.clear();
    st.seqVisIdxOf.assign(st.correspondences.size(), -1);
    st.seqPinnedParticleIdx.clear();
    st.seqPinnedTargets.clear();
    st.seqHasLastMove = false;
    st.seqAppliedCount = 0;

    if (st.correspondences.empty()) return;

    if (origVisVerts.size() >= 3) {
        size_t Nv = origVisVerts.size() / 3;
        std::vector<glm::vec3> vpts(Nv);
        for (size_t i = 0; i < Nv; i++)
            vpts[i] = glm::vec3(origVisVerts[i*3], origVisVerts[i*3+1], origVisVerts[i*3+2]);
        Reg3DCustom::NanoflannAdaptor adaptor(vpts);
        auto tree = Reg3DCustom::buildKDTree(adaptor);
        for (size_t i = 0; i < st.correspondences.size(); i++) {
            size_t nnIdx; float distSq;
            if (Reg3DCustom::searchKNN1(*tree, st.correspondences[i].srcPos, nnIdx, distSq))
                st.seqVisIdxOf[i] = static_cast<int>(nnIdx);
        }
    }

    std::vector<int> filtered;
    filtered.reserve(st.correspondences.size());
    for (size_t i = 0; i < st.correspondences.size(); i++) {
        const auto& c = st.correspondences[i];
        if (!c.ratioOK) continue;
        if (c.category == CAT_OUTLIER) continue;
        filtered.push_back(static_cast<int>(i));
    }

    std::unordered_map<int, int> bestBySrc;
    bestBySrc.reserve(filtered.size());
    for (int idx : filtered) {
        int s = st.correspondences[idx].srcIdx;
        auto it = bestBySrc.find(s);
        if (it == bestBySrc.end() ||
            st.correspondences[it->second].distProjected < st.correspondences[idx].distProjected) {
            bestBySrc[s] = idx;
        }
    }
    std::vector<int> dedup;
    dedup.reserve(bestBySrc.size());
    for (const auto& kv : bestBySrc) dedup.push_back(kv.second);
    int nBefore = static_cast<int>(filtered.size());
    int nAfter  = static_cast<int>(dedup.size());

    std::sort(dedup.begin(), dedup.end(), [&st](int a, int b) {
        return st.correspondences[a].distProjected > st.correspondences[b].distProjected;
    });

    st.seqMoveQueue = dedup;
    st.seqMoveUsed.assign(dedup.size(), false);

    std::cout << "[AutoDeform Sequential] queue built : " << nAfter
              << " correspondences (deduped " << nBefore << " -> " << nAfter
              << " by srcIdx, sorted desc)" << std::endl;
    if (!dedup.empty()) {
        std::cout << "  top dist  : " << st.correspondences[dedup.front()].distProjected << std::endl;
        std::cout << "  bot dist  : " << st.correspondences[dedup.back()].distProjected << std::endl;
    }
}

inline int pickNextSequentialMove(
    State& st,
    float percentileLo,
    float percentileHi,
    float minSeparation = 0.0f,
    const std::vector<float>* curVisVerts = nullptr,
    int* outSlot = nullptr)
{
    if (outSlot) *outSlot = -1;
    if (st.seqMoveQueue.empty()) return -1;
    int N = static_cast<int>(st.seqMoveQueue.size());
    int loIdx = std::max(0, static_cast<int>(std::floor(percentileLo * N)));
    int hiIdx = std::min(N - 1, static_cast<int>(std::floor(percentileHi * N)));
    if (loIdx > hiIdx) return -1;

    auto getCenter = [&](int corrIdx) -> glm::vec3 {
        if (corrIdx < 0) return glm::vec3(0);
        if (curVisVerts && corrIdx < (int)st.seqVisIdxOf.size()) {
            int v = st.seqVisIdxOf[corrIdx];
            if (v >= 0 && (size_t)(v*3+2) < curVisVerts->size()) {
                return glm::vec3((*curVisVerts)[v*3], (*curVisVerts)[v*3+1], (*curVisVerts)[v*3+2]);
            }
        }
        return st.correspondences[corrIdx].srcPos;
    };

    std::vector<glm::vec3> placedPositions;
    if (minSeparation > 0.0f) {
        placedPositions.reserve(N);
        for (int i = 0; i < N; i++) {
            if (st.seqMoveUsed[i]) placedPositions.push_back(getCenter(st.seqMoveQueue[i]));
        }
    }
    float minSep2 = minSeparation * minSeparation;

    for (int i = loIdx; i <= hiIdx; i++) {
        if (st.seqMoveUsed[i]) continue;
        if (minSeparation > 0.0f) {
            glm::vec3 c = getCenter(st.seqMoveQueue[i]);
            bool tooClose = false;
            for (const auto& p : placedPositions) {
                glm::vec3 dv = c - p;
                if (glm::dot(dv, dv) < minSep2) { tooClose = true; break; }
            }
            if (tooClose) continue;
        }
        st.seqMoveUsed[i] = true;
        if (outSlot) *outSlot = i;
        return st.seqMoveQueue[i];
    }
    return -1;
}

inline int pinParticlesFromHandle(
    State& st,
    SoftBody* body,
    const Handle& h)
{
    if (!body) return 0;
    const auto& positions = body->getPositions();
    size_t Npart = body->getNumParticles();
    int added = 0;
    for (size_t i = 0; i < Npart; i++) {
        glm::vec3 p(positions[i*3], positions[i*3+1], positions[i*3+2]);
        if (glm::length(p - h.center) > h.radius) continue;
        st.seqPinnedParticleIdx.push_back(static_cast<int>(i));
        st.seqPinnedTargets.push_back(p);
        added++;
    }
    return added;
}

inline int peekNextSequentialMove(
    const State& st,
    float percentileLo,
    float percentileHi,
    float minSeparation,
    int fromSlot,
    const std::vector<float>* curVisVerts,
    int* outSlot)
{
    if (outSlot) *outSlot = -1;
    if (st.seqMoveQueue.empty()) return -1;
    int N = static_cast<int>(st.seqMoveQueue.size());
    int loIdx = std::max(0, static_cast<int>(std::floor(percentileLo * N)));
    int hiIdx = std::min(N - 1, static_cast<int>(std::floor(percentileHi * N)));
    if (loIdx > hiIdx) return -1;

    auto getCenter = [&](int corrIdx) -> glm::vec3 {
        if (corrIdx < 0) return glm::vec3(0);
        if (curVisVerts && corrIdx < (int)st.seqVisIdxOf.size()) {
            int v = st.seqVisIdxOf[corrIdx];
            if (v >= 0 && (size_t)(v*3+2) < curVisVerts->size()) {
                return glm::vec3((*curVisVerts)[v*3], (*curVisVerts)[v*3+1], (*curVisVerts)[v*3+2]);
            }
        }
        return st.correspondences[corrIdx].srcPos;
    };

    std::vector<glm::vec3> placedPositions;
    if (minSeparation > 0.0f) {
        placedPositions.reserve(N);
        for (int i = 0; i < N; i++) {
            if (st.seqMoveUsed[i]) placedPositions.push_back(getCenter(st.seqMoveQueue[i]));
        }
    }
    float minSep2 = minSeparation * minSeparation;

    int startFrom = std::max(loIdx, fromSlot + 1);
    auto tryRange = [&](int a, int b) -> int {
        for (int i = a; i <= b; i++) {
            if (st.seqMoveUsed[i]) continue;
            if (i == fromSlot) continue;
            if (minSeparation > 0.0f) {
                glm::vec3 c = getCenter(st.seqMoveQueue[i]);
                bool tooClose = false;
                for (const auto& p : placedPositions) {
                    glm::vec3 dv = c - p;
                    if (glm::dot(dv, dv) < minSep2) { tooClose = true; break; }
                }
                if (tooClose) continue;
            }
            if (outSlot) *outSlot = i;
            return st.seqMoveQueue[i];
        }
        return -1;
    };

    int r = tryRange(startFrom, hiIdx);
    if (r >= 0) return r;
    if (fromSlot >= loIdx) {
        int end = std::min(hiIdx, fromSlot - 1);
        r = tryRange(loIdx, end);
        if (r >= 0) return r;
    }
    return -1;
}

inline int applySequential(
    State& st,
    SoftBody* body,
    const Handle* newMove,
    float fixCompliance,
    float moveCompliance,
    bool  verbose = false)
{
    if (!body) return 0;
    body->clearAttachmentConstraints();

    for (size_t i = 0; i < st.seqPinnedParticleIdx.size(); i++) {
        body->addAttachmentConstraint(
            st.seqPinnedParticleIdx[i],
            st.seqPinnedTargets[i],
            fixCompliance);
    }

    int nMove = 0;
    if (newMove) {
        const auto& positions = body->getPositions();
        size_t Npart = body->getNumParticles();
        for (size_t i = 0; i < Npart; i++) {
            glm::vec3 p(positions[i*3], positions[i*3+1], positions[i*3+2]);
            if (glm::length(p - newMove->center) > newMove->radius) continue;
            glm::vec3 offset = p - newMove->center;
            glm::vec3 tgt = newMove->target + offset;
            body->addAttachmentConstraint(static_cast<int>(i), tgt, moveCompliance);
            nMove++;
        }
    }

    int total = static_cast<int>(st.seqPinnedParticleIdx.size()) + nMove;
    if (verbose) {
        std::cout << "[Apply] pinned=" << st.seqPinnedParticleIdx.size()
        << " move=" << nMove << " total=" << total << std::endl;
    }
    return total;
}

inline bool buildHandleFromCorrespondence(
    const State& st,
    int corrIdx,
    const std::vector<float>& curVisVerts,
    float radius,
    Handle& outHandle,
    float progress = 0.1f)
{
    if (corrIdx < 0 || corrIdx >= static_cast<int>(st.correspondences.size())) return false;
    const auto& c = st.correspondences[corrIdx];
    glm::vec3 center = c.srcPos;
    if (corrIdx < static_cast<int>(st.seqVisIdxOf.size())) {
        int vIdx = st.seqVisIdxOf[corrIdx];
        if (vIdx >= 0 && (size_t)(vIdx*3+2) < curVisVerts.size()) {
            center = glm::vec3(
                curVisVerts[vIdx*3], curVisVerts[vIdx*3+1], curVisVerts[vIdx*3+2]);
        }
    }
    float p = std::max(0.0f, std::min(1.0f, progress));
    outHandle.center = center;
    outHandle.target = c.srcPos + (c.tgtPos - c.srcPos) * p;
    outHandle.radius = radius;
    outHandle.isFixed = false;
    return true;
}

inline void drawSequentialCandidates(
    const State& st,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    const std::vector<float>& curVisVerts,
    float percentileLo,
    float percentileHi,
    float minSeparation,
    float radius = 0.04f)
{
    if (st.seqMoveQueue.empty()) return;

    int N = static_cast<int>(st.seqMoveQueue.size());
    int loIdx = std::max(0, static_cast<int>(std::floor(percentileLo * N)));
    int hiIdx = std::min(N - 1, static_cast<int>(std::floor(percentileHi * N)));

    static const glm::vec3 colDone    (0.35f, 0.35f, 0.35f);
    static const glm::vec3 colDoneTgt (0.20f, 0.20f, 0.20f);
    static const glm::vec3 colCurrent (1.00f, 0.55f, 0.00f);
    static const glm::vec3 colCurTgt  (1.00f, 0.10f, 0.10f);
    static const glm::vec3 colPlanned (1.00f, 1.00f, 0.10f);
    static const glm::vec3 colPlanTgt (0.10f, 0.50f, 1.00f);

    auto getCenter = [&](int corrIdx) -> glm::vec3 {
        if (corrIdx < 0 || corrIdx >= (int)st.correspondences.size())
            return glm::vec3(0);
        if (corrIdx < (int)st.seqVisIdxOf.size()) {
            int vIdx = st.seqVisIdxOf[corrIdx];
            if (vIdx >= 0 && (size_t)(vIdx*3+2) < curVisVerts.size()) {
                return glm::vec3(curVisVerts[vIdx*3], curVisVerts[vIdx*3+1], curVisVerts[vIdx*3+2]);
            }
        }
        return st.correspondences[corrIdx].srcPos;
    };

    auto drawArrow = [&](const glm::vec3& a, const glm::vec3& b,
                         const glm::vec3& cA, const glm::vec3& cB,
                         int nSeg, float r) {
        for (int s = 1; s < nSeg; s++) {
            float t = static_cast<float>(s) / static_cast<float>(nSeg);
            glm::vec3 mid = glm::mix(a, b, t);
            glm::vec3 col = glm::mix(cA, cB, t);
            sphere.draw(shader, mid, col, r, view, proj, camPos);
        }
    };

    std::vector<glm::vec3> placedPositions;
    placedPositions.reserve(N);
    for (int i = 0; i < N; i++) {
        if (!st.seqMoveUsed[i]) continue;
        if (i == st.seqLastQueueSlot && st.seqHasLastMove) continue;
        int corrIdx = st.seqMoveQueue[i];
        glm::vec3 center = getCenter(corrIdx);
        const auto& c = st.correspondences[corrIdx];
        sphere.draw(shader, center,   colDone,    radius * 0.30f, view, proj, camPos);
        sphere.draw(shader, c.tgtPos, colDoneTgt, radius * 0.22f, view, proj, camPos);
        placedPositions.push_back(center);
    }
    if (st.seqHasLastMove) {
        placedPositions.push_back(st.seqLastMove.center);
    }

    float minSep2 = minSeparation * minSeparation;
    int displayedPlanned = 0;
    for (int i = loIdx; i <= hiIdx; i++) {
        if (st.seqMoveUsed[i]) continue;
        if (i == st.seqLastQueueSlot && st.seqHasLastMove) continue;
        int corrIdx = st.seqMoveQueue[i];
        glm::vec3 center = getCenter(corrIdx);
        if (minSeparation > 0.0f) {
            bool tooClose = false;
            for (const auto& p : placedPositions) {
                glm::vec3 dv = center - p;
                if (glm::dot(dv, dv) < minSep2) { tooClose = true; break; }
            }
            if (tooClose) continue;
        }
        placedPositions.push_back(center);

        const auto& c = st.correspondences[corrIdx];
        sphere.draw(shader, center,   colPlanned, radius * 0.50f, view, proj, camPos);
        sphere.draw(shader, c.tgtPos, colPlanTgt, radius * 0.35f, view, proj, camPos);
        drawArrow(center, c.tgtPos, colPlanned, colPlanTgt, 4, radius * 0.15f);
        displayedPlanned++;
    }

    if (st.seqHasLastMove) {
        glm::vec3 curCenter = st.seqLastMove.center;
        if (st.seqLastCorrIdx >= 0 && st.seqLastCorrIdx < (int)st.seqVisIdxOf.size()) {
            int v = st.seqVisIdxOf[st.seqLastCorrIdx];
            if (v >= 0 && (size_t)(v*3+2) < curVisVerts.size()) {
                curCenter = glm::vec3(curVisVerts[v*3], curVisVerts[v*3+1], curVisVerts[v*3+2]);
            }
        }
        sphere.draw(shader, curCenter,            colCurrent, radius * 1.0f, view, proj, camPos);
        sphere.draw(shader, st.seqLastMove.target, colCurTgt, radius * 0.75f, view, proj, camPos);
        drawArrow(curCenter, st.seqLastMove.target, colCurrent, colCurTgt, 6, radius * 0.20f);
    }
}

inline int autoConsumeResolvedCorrespondences(
    State& st,
    const std::vector<float>& curVisVerts,
    float resolveThreshold)
{
    if (st.seqMoveQueue.empty()) return 0;
    if (curVisVerts.empty()) return 0;

    int consumed = 0;
    for (size_t k = 0; k < st.seqMoveQueue.size(); k++) {
        if (st.seqMoveUsed[k]) continue;
        int corrIdx = st.seqMoveQueue[k];
        if (corrIdx < 0 || corrIdx >= (int)st.correspondences.size()) continue;
        const auto& c = st.correspondences[corrIdx];

        glm::vec3 curSrc = c.srcPos;
        if (corrIdx < (int)st.seqVisIdxOf.size()) {
            int vIdx = st.seqVisIdxOf[corrIdx];
            if (vIdx >= 0 && (size_t)(vIdx*3+2) < curVisVerts.size()) {
                curSrc = glm::vec3(curVisVerts[vIdx*3], curVisVerts[vIdx*3+1], curVisVerts[vIdx*3+2]);
            }
        }
        glm::vec3 n = c.srcNormal + c.tgtNormal;
        float ln = glm::length(n);
        if (ln > 1e-6f) n /= ln;
        else n = c.srcNormal;
        float newDistProj = std::abs(glm::dot(c.tgtPos - curSrc, n));
        if (newDistProj < resolveThreshold) {
            st.seqMoveUsed[k] = true;
            consumed++;
        }
    }
    return consumed;
}

inline int applySequentialMulti(
    State& st,
    SoftBody* body,
    const Handle* newMove,
    const std::vector<float>& curVisVerts,
    float fixCompliance,
    float moveCompliance,
    bool  includeCorrespWithinRadius = true)
{
    if (!body) return 0;
    body->clearAttachmentConstraints();

    for (size_t i = 0; i < st.seqPinnedParticleIdx.size(); i++) {
        body->addAttachmentConstraint(
            st.seqPinnedParticleIdx[i],
            st.seqPinnedTargets[i],
            fixCompliance);
    }

    int nMoveRaw = 0, nMoveExtra = 0;
    if (newMove) {
        const auto& positions = body->getPositions();
        size_t Npart = body->getNumParticles();
        for (size_t i = 0; i < Npart; i++) {
            glm::vec3 p(positions[i*3], positions[i*3+1], positions[i*3+2]);
            if (glm::length(p - newMove->center) > newMove->radius) continue;
            glm::vec3 offset = p - newMove->center;
            glm::vec3 tgt = newMove->target + offset;
            body->addAttachmentConstraint(static_cast<int>(i), tgt, moveCompliance);
            nMoveRaw++;
        }

        if (includeCorrespWithinRadius && !curVisVerts.empty()) {
            float r2 = newMove->radius * newMove->radius;
            for (size_t k = 0; k < st.seqMoveQueue.size(); k++) {
                if (st.seqMoveUsed[k]) continue;
                int corrIdx = st.seqMoveQueue[k];
                if (corrIdx < 0 || corrIdx >= (int)st.correspondences.size()) continue;
                const auto& c = st.correspondences[corrIdx];

                int vIdx = (corrIdx < (int)st.seqVisIdxOf.size()) ? st.seqVisIdxOf[corrIdx] : -1;
                glm::vec3 curSrc = c.srcPos;
                if (vIdx >= 0 && (size_t)(vIdx*3+2) < curVisVerts.size()) {
                    curSrc = glm::vec3(curVisVerts[vIdx*3], curVisVerts[vIdx*3+1], curVisVerts[vIdx*3+2]);
                }
                glm::vec3 dv = curSrc - newMove->center;
                if (glm::dot(dv, dv) > r2) continue;

                int bestPart = -1; float bestD2 = 1e30f;
                const auto& positions2 = body->getPositions();
                size_t Npart2 = body->getNumParticles();
                for (size_t i = 0; i < Npart2; i++) {
                    glm::vec3 p(positions2[i*3], positions2[i*3+1], positions2[i*3+2]);
                    float d = glm::length(p - curSrc);
                    float d2 = d * d;
                    if (d2 < bestD2) { bestD2 = d2; bestPart = (int)i; }
                }
                if (bestPart < 0) continue;
                body->addAttachmentConstraint(bestPart, c.tgtPos, moveCompliance);
                nMoveExtra++;
                st.seqMoveUsed[k] = true;
            }
        }
    }

    int total = static_cast<int>(st.seqPinnedParticleIdx.size()) + nMoveRaw + nMoveExtra;
    std::cout << "\n=== [AutoDeform Sequential Apply #" << st.seqAppliedCount << "] ===" << std::endl;
    std::cout << "  pinned (cumulative fix) : " << st.seqPinnedParticleIdx.size()
              << " (compliance=" << fixCompliance << ")" << std::endl;
    std::cout << "  new move particles      : " << nMoveRaw << std::endl;
    std::cout << "  extra corresp in radius : " << nMoveExtra
              << " (bundled & consumed)" << std::endl;
    std::cout << "  total attachments       : " << total << std::endl;
    std::cout << "=====================================\n" << std::endl;
    return total;
}

inline bool buildConsolidatedMoveHandle(
    State& st,
    int primaryCorrIdx,
    const std::vector<float>& curVisVerts,
    float radius,
    float minConsensusCos,
    Handle& outHandle,
    std::vector<int>& outConsumedQueueSlots)
{
    outConsumedQueueSlots.clear();
    if (primaryCorrIdx < 0 || primaryCorrIdx >= (int)st.correspondences.size()) return false;
    const auto& primary = st.correspondences[primaryCorrIdx];

    auto getCenter = [&](int corrIdx) -> glm::vec3 {
        if (corrIdx < 0) return glm::vec3(0);
        if (corrIdx < (int)st.seqVisIdxOf.size()) {
            int v = st.seqVisIdxOf[corrIdx];
            if (v >= 0 && (size_t)(v*3+2) < curVisVerts.size()) {
                return glm::vec3(curVisVerts[v*3], curVisVerts[v*3+1], curVisVerts[v*3+2]);
            }
        }
        return st.correspondences[corrIdx].srcPos;
    };

    glm::vec3 primaryCenter = getCenter(primaryCorrIdx);
    glm::vec3 primaryDisp   = primary.tgtPos - primaryCenter;
    float primaryLen = glm::length(primaryDisp);
    if (primaryLen < 1e-8f) return false;
    glm::vec3 primaryDir = primaryDisp / primaryLen;

    float r2 = radius * radius;

    std::vector<glm::vec3> memberCenters;
    std::vector<glm::vec3> memberDisps;
    std::vector<int>       memberQueueSlots;
    int primarySlot = -1;

    for (size_t k = 0; k < st.seqMoveQueue.size(); k++) {
        int corrIdx = st.seqMoveQueue[k];
        if (corrIdx == primaryCorrIdx) { primarySlot = (int)k; break; }
    }

    memberCenters.push_back(primaryCenter);
    memberDisps.push_back(primaryDisp);
    if (primarySlot >= 0) memberQueueSlots.push_back(primarySlot);

    for (size_t k = 0; k < st.seqMoveQueue.size(); k++) {
        if (st.seqMoveUsed[k]) continue;
        int corrIdx = st.seqMoveQueue[k];
        if (corrIdx == primaryCorrIdx) continue;
        if (corrIdx < 0 || corrIdx >= (int)st.correspondences.size()) continue;
        const auto& c = st.correspondences[corrIdx];

        glm::vec3 center = getCenter(corrIdx);
        glm::vec3 dv = center - primaryCenter;
        if (glm::dot(dv, dv) > r2) continue;

        glm::vec3 disp = c.tgtPos - center;
        float dLen = glm::length(disp);
        if (dLen < 1e-8f) continue;
        glm::vec3 dir = disp / dLen;

        if (glm::dot(dir, primaryDir) < minConsensusCos) continue;

        memberCenters.push_back(center);
        memberDisps.push_back(disp);
        memberQueueSlots.push_back(static_cast<int>(k));
    }

    glm::vec3 centroid(0.0f), avgDisp(0.0f);
    for (const auto& p : memberCenters) centroid += p;
    for (const auto& d : memberDisps)   avgDisp  += d;
    centroid /= static_cast<float>(memberCenters.size());
    avgDisp  /= static_cast<float>(memberDisps.size());

    outHandle.center  = centroid;
    outHandle.target  = centroid + avgDisp;
    outHandle.radius  = radius;
    outHandle.isFixed = false;
    outConsumedQueueSlots = memberQueueSlots;

    std::cout << "[Consolidated] members=" << memberCenters.size()
              << " (within r=" << radius << ", minCos=" << minConsensusCos << ")"
              << "  avgDispLen=" << glm::length(avgDisp) << std::endl;
    return true;
}

} // namespace AutoDeform

#endif
