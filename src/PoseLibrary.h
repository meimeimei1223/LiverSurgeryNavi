#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <iostream>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "RegistrationActions.h"
#include "LiverLeftRightLabel.h"   // V3R: PoseEntry::quadrant_mask 用 (quadrantMaskString)

struct PoseEntry {
    int id = -1;

    enum Method { FULL_AUTO=0, HEMI_AUTO=1, UMEYAMA=2, BIPOP_CMAES=3, SILHOUETTE_ALIGN=4 };
    Method baseMethod = HEMI_AUTO;

    int sessionId   = 1;
    int bipopCount  = 0;
    int silhouetteCount = 0;
    float elapsedSec = 0.0f;

    std::string timestamp;

    float baseFitness    = 0.0f;
    float baseIcpRmse    = 0.0f;
    float baseAvgError   = 0.0f;
    float baseRmse       = 0.0f;
    float baseMaxError   = 0.0f;
    float baseScale      = 1.0f;

    float compRmse     = 0.0f;
    float compAvgError = 0.0f;
    float compMaxError = 0.0f;
    float compIoU2D    = 0.0f;
    // [Phase B] IoU computed with rasterize_iou2d_v3rs while excluding pixels
    // covered by the instrument distance map. Recorded as a parallel diagnostic
    // alongside compIoU2D (which is method-neutral full-image IoU). When the
    // Pose Library's "Use IoU_occluded for accept" toggle is OFF (default),
    // this field is purely informational; when ON, it becomes the criterion
    // used in Layer 4 acceptance. See
    // DESIGN_Occlusion_Aware_Silhouette_Anchor.md §4.2.
    float compIoU2D_occluded = 0.0f;
    int   compCount    = 0;

    // V3R rim diagnostic: RMSE measured ONLY over rim-rim pairs (src
    // vertices on LiverRegionLabel::RIM, tgt points with boundaryDist <
    // rim_thresh, both gated by max_dist_sq = (sceneDiag/7.36)^2).
    // Populated by Ctrl+G (runBipopCmaesV3R Phase F.5);
    //   -1.0f       = not measured (non-Ctrl+G entry, or rim N/A)
    //    >= 0.0f    = rim-only RMSE in liver units (metres)
    //
    // Display-only diagnostic; NOT used in the Layer 4 acceptance gate.
    // The point of these fields is the "rim overfitting indicator":
    // rimMatched / rimTgtTotal gives the fraction of target rim that
    // actually has a source-rim NN within the 22cm gate. When the optimiser
    // overfits to the dense interior, this fraction collapses (e.g.
    // 17633/42184 → 1852/42184 across 9 Ctrl+G sessions on LIVER01) even
    // though compRmse continues to improve. See
    // HANDOVER_Measurement_Inconsistency_Analysis.md §1.
    float compRmseRim = -1.0f;
    int   rimMatched  = 0;
    int   rimTgtTotal = 0;
    int   rimSrcTotal = 0;

    // [Phase B] V3R rim PAIR storage for the colored-pairs visualizer.
    // Captured by Ctrl+G / Ctrl+Shift+G Phase F.5 (post-Apply pose) via
    // the g_lastRimPair* globals and copied here by buildEntryFromCurrent.
    // Size invariant: rimPairSrcVertIdx.size() == rimPairTgtPos.size()
    //                                          == rimMatched
    //                                          (== 0 for non-Ctrl+G saves).
    //
    // src is a liverMesh3D->mVertices index (full-mesh; "follow" semantics
    // so the marker tracks the mesh through subsequent ICP/Apply updates).
    // tgt is an immutable world coordinate (image-side; never moves).
    //
    // Display-only; NEVER read by the Layer 4 acceptance gate. Not yet
    // serialised to CSV (Phase B keeps these RAM-only; cross-session
    // persistence would require a sidecar binary file, deferred).
    std::vector<int>       rimPairSrcVertIdx;
    std::vector<glm::vec3> rimPairTgtPos;

    // [NEW V3RS-CONTAIN] Containment-direction diagnostic (precision/recall
    // of the post-Apply silhouette vs. SAM2 target mask, instrument-aware).
    // -1.0f sentinel = "not measured" (= HemiAuto or other non-Ctrl+G save,
    // or the rasterize bailed out because the boundary map was unavailable).
    // recall > precision  → source overshoots target (oversized silhouette)
    // recall < precision  → source undershoots target (undersized)
    // Display-only — same lifecycle as compRmseRim.
    float compIoUOccPrecision = -1.0f;
    float compIoUOccRecall    = -1.0f;

    std::vector<glm::vec3> corrSource;
    std::vector<glm::vec3> corrTarget;

    std::string initOrientation = "Base";   // Phase 2: 旧 "Front" → "Base"
    int orientRunCount = 1;

    // Phase 2: シード再現性のため、entry 保存時の (trialSeed, callIdx) を記録。
    // Apply 時にこれらを g_trialSeed / g_callIdx に書き戻すことで、
    // この entry の姿勢から Shift+V (BIPOP-CMA-ES) を実行した結果が
    // AutoProbe 中に同じ probe で実行されたケースと bit-identical になる。
    uint32_t savedTrialSeed = 0;
    uint32_t savedCallIdx   = 0;

    // V3R (Ctrl+G) で記録された 4 象限選択ビットマスク。
    // 0xFF (QUAD_LEGACY_FULL) が初期値で、Ctrl+G 以外のメソッド
    // (HemiAuto, Shift+V, Shift+F, Shift+G, Shift+E など) のエントリは
    // この値のまま保存される。Ctrl+G セッションだけ実 mask
    // (LiverLeftRightLabel::QuadrantMask の 0x01..0x0F) を入れる。
    uint8_t quadrant_mask = 0xFF;  // QUAD_LEGACY_FULL

    glm::mat4 transform = glm::mat4(1.0f);

    float finalRmse() const { return compRmse; }

    const char* methodStr() const {
        switch (baseMethod) {
        case FULL_AUTO:        return "FullAuto";
        case HEMI_AUTO:        return "HemiAuto";
        case UMEYAMA:          return "Umeyama";
        case BIPOP_CMAES:      return "BIPOP";
        case SILHOUETTE_ALIGN: return "SilAln";
        }
        return "Unknown";
    }

    std::string sessionLabel() const {
        return initOrientation + "#" + std::to_string(orientRunCount);
    }

    std::string label() const {
        std::string s = methodStr();
        s += "/" + sessionLabel();
        if (bipopCount > 0)      s += "+Bx" + std::to_string(bipopCount);
        if (silhouetteCount > 0) s += "+Sx" + std::to_string(silhouetteCount);
        // V3R: quadrant_mask は legacy 値 (0xFF) では非表示。Ctrl+G セッション
        // (実 mask 0x01..0x0F が記録されたエントリ) のみ "Q:AR+AL" 等を付加。
        if (quadrant_mask != 0xFF) {
            s += " " + LiverLeftRightLabel::quadrantMaskString(quadrant_mask);
        }
        return s;
    }
};

class PoseLibrary {
public:
    std::vector<PoseEntry> entries;
    int maxEntries = 500;   // Pose Library FIFO 上限。AutoProbe N=108 を
        // 4 回連続実行しても溢れないサイズを確保。
    int nextId     = 1;

    PoseEntry lastRegistration;
    bool hasLastRegistration = false;

    int activeEntryId = -1;
    bool showWindow   = false;

    static std::string nowTimestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
        return std::string(buf);
    }

    static glm::mat4 computeTransformFromLiver(
        const std::vector<GLfloat>& initVerts,
        const std::vector<GLfloat>& curVerts)
    {
        size_t n = initVerts.size() / 3;
        if (n == 0) return glm::mat4(1.0f);

        glm::vec3 srcCenter(0), dstCenter(0);
        for (size_t i = 0; i < n; i++) {
            srcCenter += glm::vec3(initVerts[i*3], initVerts[i*3+1], initVerts[i*3+2]);
            dstCenter += glm::vec3(curVerts[i*3],  curVerts[i*3+1],  curVerts[i*3+2]);
        }
        srcCenter /= (float)n;
        dstCenter /= (float)n;

        glm::mat3 H(0.0f);
        float srcVar = 0.0f;
        for (size_t i = 0; i < n; i++) {
            glm::vec3 s = glm::vec3(initVerts[i*3], initVerts[i*3+1], initVerts[i*3+2]) - srcCenter;
            glm::vec3 d = glm::vec3(curVerts[i*3],  curVerts[i*3+1],  curVerts[i*3+2])  - dstCenter;
            H += glm::outerProduct(d, s);
            srcVar += glm::dot(s, s);
        }

        glm::mat3 R = H;
        for (int iter = 0; iter < 200; iter++) {
            glm::mat3 Rinv = glm::inverse(R);
            R = 0.5f * (R + glm::transpose(Rinv));
        }

        glm::mat3 RtH = glm::transpose(R) * H;
        float traceRtH = RtH[0][0] + RtH[1][1] + RtH[2][2];
        float scale = (srcVar > 1e-8f) ? (traceRtH / srcVar) : 1.0f;
        // Issue 2 fix (HANDOVER V3 §3.2): 旧コードは clamp(scale, 0.5, 2.0)
        // で囲んでいたが、長セッション (>20 BIPOP iter) では累積 scale が
        // 1.05^21 ≈ 2.8 まで成長することが実機で確認された (Bx21 で
        // det(R)=8, col_norms=(2,2,2) → clamp に当たり情報損失、Apply 時に
        // CompRMSE 0.04〜0.05 drift)。実用範囲を狭めず、NaN/Inf 防止のみ
        // 目的とした緩い clamp に置き換える。
        scale = glm::clamp(scale, 0.01f, 100.0f);
        // Reflection guard: H = sum d*s^T で det(H) < 0 (反射) の場合、
        // polar 反復後の R も det=-1 になり scale が負に出る。Procrustes は
        // 通常 det>0 (正しい SRT で構成された mesh ペア) なので、安全側で
        // 正の値に倒しておく。
        if (scale < 0.0f) scale = -scale;

        glm::vec3 t = dstCenter - scale * R * srcCenter;

        glm::mat4 T(1.0f);
        T[0] = glm::vec4(scale * R[0], 0.0f);
        T[1] = glm::vec4(scale * R[1], 0.0f);
        T[2] = glm::vec4(scale * R[2], 0.0f);
        T[3] = glm::vec4(t, 1.0f);
        return T;
    }

    static void applyTransformToMeshes(
        const glm::mat4& T,
        const std::vector<std::vector<GLfloat>>& initVerts,
        const std::vector<std::vector<GLfloat>>& initNormals,
        std::vector<mCutMesh*>& organs)
    {
        glm::mat3 R = glm::mat3(T);
        glm::mat3 normalMat = glm::transpose(glm::inverse(R));
        for (size_t m = 0; m < organs.size() && m < initVerts.size(); m++) {
            const auto& iv = initVerts[m];
            const auto& in_ = initNormals[m];
            auto* mesh = organs[m];
            size_t nv = iv.size() / 3;
            mesh->mVertices.resize(iv.size());
            mesh->mNormals.resize(in_.size());
            for (size_t i = 0; i < nv; i++) {
                glm::vec4 v(iv[i*3], iv[i*3+1], iv[i*3+2], 1.0f);
                v = T * v;
                mesh->mVertices[i*3]   = v.x;
                mesh->mVertices[i*3+1] = v.y;
                mesh->mVertices[i*3+2] = v.z;
            }
            size_t nn = in_.size() / 3;
            for (size_t i = 0; i < nn; i++) {
                glm::vec3 n(in_[i*3], in_[i*3+1], in_[i*3+2]);
                n = glm::normalize(normalMat * n);
                mesh->mNormals[i*3]   = n.x;
                mesh->mNormals[i*3+1] = n.y;
                mesh->mNormals[i*3+2] = n.z;
            }
            setUp(*mesh);
        }
    }

    void autoSaveLastRegistration(const glm::mat4& transform) {
        lastRegistration = PoseEntry();
        lastRegistration.transform = transform;
        lastRegistration.timestamp = nowTimestamp();
        hasLastRegistration = true;
        std::cout << "[PoseLibrary] Undo snapshot saved" << std::endl;
    }

    bool undoToLast(
        const std::vector<std::vector<GLfloat>>& initVerts,
        const std::vector<std::vector<GLfloat>>& initNormals,
        std::vector<mCutMesh*>& organs)
    {
        if (!hasLastRegistration) {
            std::cout << "[PoseLibrary] Nothing to undo" << std::endl;
            return false;
        }
        applyTransformToMeshes(lastRegistration.transform, initVerts, initNormals, organs);
        activeEntryId = -1;
        std::cout << "[PoseLibrary] Undo: restored" << std::endl;
        return true;
    }

    PoseEntry buildEntryFromCurrent(
        PoseEntry::Method method,
        int sessionId,
        int bipopCount,
        int silhouetteCount,
        float elapsedSec,
        float baseFitness, float baseIcpRmse,
        float baseAvgError, float baseRmse, float baseMaxError,
        float baseScale,
        float compRmse, float compAvgError, float compMaxError,
        float compIoU2D,
        float compIoU2D_occluded,   // [Phase B] parallel IoU_occluded
        int compCount,
        const std::vector<glm::vec3>& compSrc,
        const std::vector<glm::vec3>& compTgt,
        const glm::mat4& transform,
        const std::string& initOrientation = "Base",
        int orientRunCount = 1,
        uint32_t savedTrialSeed = 0,
        uint32_t savedCallIdx   = 0,
        uint8_t  quadrant_mask  = 0xFF,   // V3R: 0xFF = legacy (Ctrl+G 以外)
        // V3R rim diagnostic (display-only). Defaults preserve N/A for
        // non-Ctrl+G call sites; Ctrl+G fills these in poseSaveToLibrary
        // by snapshotting the g_lastRim* globals set in Phase F.5.
        float compRmseRim = -1.0f,
        int   rimMatched  = 0,
        int   rimTgtTotal = 0,
        int   rimSrcTotal = 0,
        // [NEW V3RS-CONTAIN] Containment direction (display-only). Same
        // N/A-by-default policy: Ctrl+G snapshots g_lastIoUOcc* in
        // poseSaveToLibrary, other call sites leave them as -1.
        float compIoUOccPrecision = -1.0f,
        float compIoUOccRecall    = -1.0f,
        // [Phase B] Rim pair vectors for the colored-pairs viewer. Default
        // empty for non-Ctrl+G call sites; Ctrl+G fills these in
        // poseSaveToLibrary by snapshotting g_lastRimPair* (set in
        // Phase F.5). Size invariant: src.size() == tgt.size() == rimMatched.
        const std::vector<int>&       rimPairSrcVertIdx = {},
        const std::vector<glm::vec3>& rimPairTgtPos     = {})
    {
        PoseEntry e;
        e.id              = nextId++;
        e.baseMethod      = method;
        e.sessionId       = sessionId;
        e.bipopCount      = bipopCount;
        e.silhouetteCount = silhouetteCount;
        e.elapsedSec      = elapsedSec;
        e.timestamp       = nowTimestamp();
        e.baseFitness     = baseFitness;
        e.baseIcpRmse     = baseIcpRmse;
        e.baseAvgError    = baseAvgError;
        e.baseRmse        = baseRmse;
        e.baseMaxError    = baseMaxError;
        e.baseScale       = baseScale;
        e.compRmse        = compRmse;
        e.compAvgError    = compAvgError;
        e.compMaxError    = compMaxError;
        e.compIoU2D       = compIoU2D;
        e.compIoU2D_occluded = compIoU2D_occluded;   // [Phase B]
        e.compCount       = compCount;
        e.corrSource      = compSrc;
        e.corrTarget      = compTgt;
        e.transform       = transform;
        e.initOrientation = initOrientation;
        e.orientRunCount  = orientRunCount;
        e.savedTrialSeed  = savedTrialSeed;
        e.savedCallIdx    = savedCallIdx;
        e.quadrant_mask   = quadrant_mask;   // V3R
        e.compRmseRim     = compRmseRim;     // V3R rim diag
        e.rimMatched      = rimMatched;
        e.rimTgtTotal     = rimTgtTotal;
        e.rimSrcTotal     = rimSrcTotal;
        e.compIoUOccPrecision = compIoUOccPrecision;   // V3RS-CONTAIN
        e.compIoUOccRecall    = compIoUOccRecall;
        e.rimPairSrcVertIdx   = rimPairSrcVertIdx;     // [Phase B]
        e.rimPairTgtPos       = rimPairTgtPos;
        return e;
    }

    void addEntry(const PoseEntry& entry) {
        entries.push_back(entry);
        while ((int)entries.size() > maxEntries)
            entries.erase(entries.begin());
        std::cout << "[PoseLibrary] Added entry #" << entry.id
                  << " (" << entry.label()
                  << ", CompRMSE=" << entry.compRmse
                  << ", IoU2D=" << entry.compIoU2D
                  << ", IoU_occ=" << entry.compIoU2D_occluded   // [Phase B]
                  << ", RIM=";
        // V3R rim diagnostic: pretty-print as "rmse (matched/tgt_total)"
        // when measured (Ctrl+G entries), or "N/A" for other methods.
        // [Phase B] When rim pairs were captured, append "+Npairs" so the
        // log surfaces that the colored-pairs viewer has data for this
        // entry. Size invariant: pair count == rimMatched (verified on
        // capture in Phase F.5).
        if (entry.compRmseRim >= 0.0f) {
            std::cout << entry.compRmseRim
                      << " (" << entry.rimMatched
                      << "/" << entry.rimTgtTotal << ")";
            if (!entry.rimPairSrcVertIdx.empty()) {
                std::cout << " +" << entry.rimPairSrcVertIdx.size() << "pairs";
            }
        } else {
            std::cout << "N/A";
        }
        // [NEW V3RS-CONTAIN] Pretty-print containment direction inline.
        // [NEW V3RS-CONTAIN-RATIO] Lead with size_ratio so the [Added
        // entry] line surfaces the same magnitude info as the table.
        std::cout << ", Contain=";
        if (entry.compIoUOccPrecision >= 0.0f &&
            entry.compIoUOccRecall    >= 0.0f) {
            const float sr = (entry.compIoUOccPrecision > 1e-6f)
            ? entry.compIoUOccRecall / entry.compIoUOccPrecision
            : 0.0f;
            std::cout << "size=" << std::fixed << std::setprecision(2) << sr << "x"
                      << "/r=" << entry.compIoUOccRecall
                      << std::defaultfloat;
        } else {
            std::cout << "N/A";
        }
        std::cout << ", elapsed=" << entry.elapsedSec << "s"
                  << "). Library size: " << entries.size() << std::endl;
    }

    void saveCurrentToLibrary(
        PoseEntry::Method method,
        int sessionId,
        int bipopCount,
        int silhouetteCount,
        float elapsedSec,
        float baseFitness, float baseIcpRmse,
        float baseAvgError, float baseRmse, float baseMaxError,
        float baseScale,
        float compRmse, float compAvgError, float compMaxError,
        float compIoU2D,
        float compIoU2D_occluded,   // [Phase B] parallel IoU_occluded
        int compCount,
        const std::vector<glm::vec3>& compSrc,
        const std::vector<glm::vec3>& compTgt,
        const glm::mat4& transform,
        const std::string& initOrientation = "Base",
        int orientRunCount = 1,
        uint32_t savedTrialSeed = 0,
        uint32_t savedCallIdx   = 0,
        uint8_t  quadrant_mask  = 0xFF,   // V3R: 0xFF = legacy (Ctrl+G 以外)
        // V3R rim diagnostic (display-only). Defaults preserve N/A for
        // call sites that don't fill these in.
        float compRmseRim = -1.0f,
        int   rimMatched  = 0,
        int   rimTgtTotal = 0,
        int   rimSrcTotal = 0,
        // [NEW V3RS-CONTAIN] Containment direction (display-only).
        float compIoUOccPrecision = -1.0f,
        float compIoUOccRecall    = -1.0f,
        // [Phase B] Rim pair vectors for the colored-pairs viewer.
        // Default empty for non-Ctrl+G call sites.
        const std::vector<int>&       rimPairSrcVertIdx = {},
        const std::vector<glm::vec3>& rimPairTgtPos     = {})
    {
        PoseEntry e = buildEntryFromCurrent(
            method, sessionId, bipopCount, silhouetteCount, elapsedSec,
            baseFitness, baseIcpRmse, baseAvgError, baseRmse, baseMaxError, baseScale,
            compRmse, compAvgError, compMaxError,
            compIoU2D, compIoU2D_occluded,   // [Phase B]
            compCount,
            compSrc, compTgt, transform,
            initOrientation, orientRunCount,
            savedTrialSeed, savedCallIdx,
            quadrant_mask,   // V3R
            compRmseRim, rimMatched, rimTgtTotal, rimSrcTotal,   // V3R rim diag
            compIoUOccPrecision, compIoUOccRecall,   // V3RS-CONTAIN
            rimPairSrcVertIdx, rimPairTgtPos);   // [Phase B] rim pair viz
        addEntry(e);
    }

    bool applyEntry(
        int entryId,
        const std::vector<std::vector<GLfloat>>& initVerts,
        const std::vector<std::vector<GLfloat>>& initNormals,
        std::vector<mCutMesh*>& organs)
    {
        for (auto& e : entries) {
            if (e.id == entryId) {
                lastRegistration = PoseEntry();
                lastRegistration.transform = e.transform;
                lastRegistration.timestamp = nowTimestamp();
                hasLastRegistration = true;

                {
                    glm::mat3 R = glm::mat3(e.transform);
                    float det = glm::determinant(R);
                    float c0 = glm::length(R[0]);
                    float c1 = glm::length(R[1]);
                    float c2 = glm::length(R[2]);
                    std::cout << "[PoseLibrary] Apply transform debug:" << std::endl;
                    std::cout << "  det(R)=" << det
                              << "  col_norms=(" << c0 << ", " << c1 << ", " << c2 << ")" << std::endl;
                    std::cout << "  T[3]= (" << e.transform[3][0] << ", "
                              << e.transform[3][1] << ", " << e.transform[3][2] << ")" << std::endl;
                }
                auto meshBBox = [](const std::vector<GLfloat>& v, const std::string& tag) {
                    if (v.size() < 3) return;
                    float mn[3]={v[0],v[1],v[2]}, mx[3]={v[0],v[1],v[2]};
                    for (size_t i=0; i+2<v.size(); i+=3) {
                        for(int k=0;k<3;k++){mn[k]=std::min(mn[k],v[i+k]);mx[k]=std::max(mx[k],v[i+k]);}
                    }
                    std::cout << tag
                              << " size=(" << (mx[0]-mn[0]) << ", " << (mx[1]-mn[1]) << ", " << (mx[2]-mn[2]) << ")"
                              << " center=(" << (mn[0]+mx[0])*0.5f << ", " << (mn[1]+mx[1])*0.5f << ", " << (mn[2]+mx[2])*0.5f << ")" << std::endl;
                };
                if (!initVerts.empty()) meshBBox(initVerts[0], "[DEBUG] initVerts[0]");
                applyTransformToMeshes(e.transform, initVerts, initNormals, organs);
                if (organs[0] && !organs[0]->mVertices.empty()) meshBBox(organs[0]->mVertices, "[DEBUG] after Apply");
                activeEntryId = entryId;
                std::cout << "[PoseLibrary] Applied entry #" << entryId
                          << " (" << e.label() << ")" << std::endl;
                return true;
            }
        }
        std::cout << "[PoseLibrary] Entry #" << entryId << " not found" << std::endl;
        return false;
    }

    void deleteEntry(int entryId) {
        entries.erase(
            std::remove_if(entries.begin(), entries.end(),
                           [entryId](const PoseEntry& e) { return e.id == entryId; }),
            entries.end());
        if (activeEntryId == entryId) activeEntryId = -1;
    }

    bool exportToCsv(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "[PoseLibrary] Cannot open " << filepath << std::endl;
            return false;
        }

        ofs << "id,session,session_id,method,bipop_count,elapsed_sec,timestamp,"
            << "base_fitness,base_icp_rmse,base_corr_avg_error,base_corr_rmse,base_corr_max_error,base_scale,"
            << "comp_rmse,comp_avg_error,comp_max_error,comp_count,"
            << "init_orientation,orient_run,"
            << "m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33,"
            << "comp_iou_2d,silhouette_count,"
            << "saved_trial_seed,saved_call_idx,"
            << "quadrant_mask,"          // V3R: 0xFF=255=legacy, 0x0F=15=ALL, 0x01..0x08=single
            << "comp_iou_2d_occluded"    // [Phase B] parallel IoU_occluded column
            << std::endl;

        ofs << std::fixed << std::setprecision(6);

        for (const auto& e : entries) {
            ofs << e.id << ","
                << e.sessionLabel() << ","
                << e.sessionId << ","
                << e.methodStr() << ","
                << e.bipopCount << ","
                << e.elapsedSec << ","
                << e.timestamp << ","
                << e.baseFitness << ","
                << e.baseIcpRmse << ","
                << e.baseAvgError << ","
                << e.baseRmse << ","
                << e.baseMaxError << ","
                << e.baseScale << ","
                << e.compRmse << ","
                << e.compAvgError << ","
                << e.compMaxError << ","
                << e.compCount << ","
                << e.initOrientation << ","
                << e.orientRunCount;
            for (int col = 0; col < 4; col++)
                for (int row = 0; row < 4; row++)
                    ofs << "," << e.transform[col][row];
            ofs << "," << e.compIoU2D
                << "," << e.silhouetteCount
                << "," << e.savedTrialSeed
                << "," << e.savedCallIdx
                << "," << (unsigned)e.quadrant_mask   // V3R: uint8_t を整数として出力
                << "," << e.compIoU2D_occluded       // [Phase B]
                << "," << e.compRmseRim              // V3R rim diag
                << "," << e.rimMatched
                << "," << e.rimTgtTotal
                << "," << e.rimSrcTotal
                << "," << e.compIoUOccPrecision      // V3RS-CONTAIN
                << "," << e.compIoUOccRecall;
            ofs << std::endl;
        }

        std::cout << "[PoseLibrary] Exported " << entries.size()
                  << " entries to " << filepath << std::endl;
        return true;
    }

    bool importFromCsv(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            std::cerr << "[PoseLibrary] Cannot open " << filepath << std::endl;
            return false;
        }

        std::string line;
        std::getline(ifs, line);

        // Column count: 19 fixed + 16 transform = 35; +2 optional (iou,silhouette) = 37.
        // (Old format with refine columns had 38/40 — those columns are now ignored.)
        const int kFixedCols     = 19;
        const int kTransformCols = 16;
        const int kMinCols       = kFixedCols + kTransformCols; // 35

        int count = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<std::string> tok;
            std::string field;
            while (std::getline(ss, field, ',')) tok.push_back(field);
            if ((int)tok.size() < kMinCols) continue;

            PoseEntry e;
            e.id = nextId++;
            std::string mstr = tok[3];
            if      (mstr == "HemiAuto")   e.baseMethod = PoseEntry::HEMI_AUTO;
            else if (mstr == "BIPOP")      e.baseMethod = PoseEntry::BIPOP_CMAES;
            else if (mstr == "Umeyama")    e.baseMethod = PoseEntry::UMEYAMA;
            else if (mstr == "SilAln")     e.baseMethod = PoseEntry::SILHOUETTE_ALIGN;
            else                           e.baseMethod = PoseEntry::FULL_AUTO;
            e.sessionId           = std::stoi(tok[2]);
            e.bipopCount          = std::stoi(tok[4]);
            e.elapsedSec          = std::stof(tok[5]);
            e.timestamp           = tok[6];
            e.baseFitness         = std::stof(tok[7]);
            e.baseIcpRmse         = std::stof(tok[8]);
            e.baseAvgError        = std::stof(tok[9]);
            e.baseRmse            = std::stof(tok[10]);
            e.baseMaxError        = std::stof(tok[11]);
            e.baseScale           = std::stof(tok[12]);
            e.compRmse            = std::stof(tok[13]);
            e.compAvgError        = std::stof(tok[14]);
            e.compMaxError        = std::stof(tok[15]);
            e.compCount           = std::stoi(tok[16]);
            e.initOrientation     = tok[17];
            e.orientRunCount      = std::stoi(tok[18]);
            int ti = kFixedCols;
            for (int col = 0; col < 4; col++)
                for (int row = 0; row < 4; row++)
                    e.transform[col][row] = std::stof(tok[ti++]);
            /* Optional columns appended at end (backward compat: missing → 0) */
            if ((int)tok.size() > ti)     e.compIoU2D       = std::stof(tok[ti]);
            if ((int)tok.size() > ti + 1) e.silhouetteCount = std::stoi(tok[ti + 1]);
            if ((int)tok.size() > ti + 2) e.savedTrialSeed  = (uint32_t)std::stoul(tok[ti + 2]);
            if ((int)tok.size() > ti + 3) e.savedCallIdx    = (uint32_t)std::stoul(tok[ti + 3]);
            if ((int)tok.size() > ti + 4) e.quadrant_mask   = (uint8_t)std::stoul(tok[ti + 4]);  // V3R: optional, default 0xFF (legacy)
            // [Phase B] comp_iou_2d_occluded - optional, default 0.0f
            if ((int)tok.size() > ti + 5) e.compIoU2D_occluded = std::stof(tok[ti + 5]);
            // V3R rim diag - optional, defaults (-1.0, 0, 0, 0) = "not measured"
            if ((int)tok.size() > ti + 6) e.compRmseRim = std::stof(tok[ti + 6]);
            if ((int)tok.size() > ti + 7) e.rimMatched  = std::stoi(tok[ti + 7]);
            if ((int)tok.size() > ti + 8) e.rimTgtTotal = std::stoi(tok[ti + 8]);
            if ((int)tok.size() > ti + 9) e.rimSrcTotal = std::stoi(tok[ti + 9]);
            // [NEW V3RS-CONTAIN] precision/recall - optional, default -1 = N/A
            if ((int)tok.size() > ti + 10) e.compIoUOccPrecision = std::stof(tok[ti + 10]);
            if ((int)tok.size() > ti + 11) e.compIoUOccRecall    = std::stof(tok[ti + 11]);

            entries.push_back(e);
            count++;
        }

        std::cout << "[PoseLibrary] Imported " << count
                  << " entries from " << filepath << std::endl;
        return count > 0;
    }

    bool exportCorrespondences(int entryId, const std::string& filepath) const {
        for (const auto& e : entries) {
            if (e.id == entryId) {
                std::ofstream ofs(filepath);
                if (!ofs.is_open()) return false;
                ofs << "source_x,source_y,source_z,target_x,target_y,target_z,distance" << std::endl;
                ofs << std::fixed << std::setprecision(8);
                for (size_t i = 0; i < e.corrSource.size() && i < e.corrTarget.size(); i++) {
                    float d = glm::distance(e.corrSource[i], e.corrTarget[i]);
                    ofs << e.corrSource[i].x << "," << e.corrSource[i].y << "," << e.corrSource[i].z << ","
                        << e.corrTarget[i].x << "," << e.corrTarget[i].y << "," << e.corrTarget[i].z << ","
                        << d << std::endl;
                }
                return true;
            }
        }
        return false;
    }
};

// =========================================================================
//  Application glue
//  -----------------------------------------------------------------------
//  以下は元の長いmain.cppから移植したセッション管理＋UIのコード。
//  アプリケーション固有のため PoseLibrary クラス本体とは分離してある。
//  Refine 関連は完全に削除し、それ以外は元の挙動を忠実に再現する。
// =========================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "imgui.h"
#include "RegistrationCore.h"           // RegistrationData
#include "RegistrationImGuiManager.h"   // RegUIState (gUI.state.regMethod 参照)
#include "PathConfig.h"                 // REG_MODEL_PATH
#include "FullSphereCameraWithTarget.h" // FullSphereCamera 型 (runAutoProbe の AR 固定用)

#ifdef HAS_TINYFILEDIALOGS
#include "tinyfiledialogs.h"
#endif

// ----- main.cpp / RegistrationActions.h 側のグローバル参照 -----
extern mCutMesh* liverMesh3D;
extern mCutMesh* portalMesh3D;
extern mCutMesh* veinMesh3D;
extern mCutMesh* tumorMesh3D;
extern mCutMesh* segmentMesh3D;
extern mCutMesh* gbMesh3D;
extern RegistrationData         registrationHandle;
extern RegistrationImGuiManager gUI;
extern std::vector<std::vector<GLfloat>> g_initOrganVertices;
extern std::vector<std::vector<GLfloat>> g_initOrganNormals;
extern FullSphereCamera         OrbitCam;   // runAutoProbe で AR 固定するため

// computeUnifiedMetrics() は RegistrationActions.h で inline 定義されている。
// CmaesUtils.h は内部で `static void computeUnifiedMetrics();` を持っているので、
// PoseLibrary.h 側で重複前方宣言すると -fpermissive でも衝突するため宣言しない。
// その代わり、main.cpp 側で PoseLibrary.h を RegistrationActions.h より後ろに
// include することで、PoseLibrary.h の inline 関数群が定義時点で
// computeUnifiedMetrics の宣言を参照できるようにする。

// ----- 採択基準 -----
// RMSE   : compRmse が session-best 以下なら accept (HemiAuto / BIPOP / Umeyama 既定)
// IOU    : compIoU2D が session-best より大きければ accept (Shift+E 用)
// EITHER : どちらかが改善すれば accept
enum class SaveCriterion { RMSE, IOU, EITHER };

// ----- inline グローバル (C++17) -----
inline PoseLibrary g_poseLibrary;

inline int          g_sessionId           = 1;
inline int          g_sessionBipopN       = 0;
inline int          g_sessionSilhouetteN  = 0;
inline std::string  g_currentOrientLabel  = "Base";   // Phase 2: 旧 "Front" -> "Base"
inline int          g_currentOrientRunCount = 0;
inline std::chrono::steady_clock::time_point g_stepStartTime
    = std::chrono::steady_clock::now();

inline float g_bestSessionCompRmse = FLT_MAX;
inline float g_bestSessionIoU2D    = 0.0f;
// [Phase B] Parallel session-best for IoU_occluded. Updated alongside
// g_bestSessionIoU2D whenever an entry is accepted; the choice of which
// to compare against is governed by g_poseLibraryUseOccludedForAccept
// (see below). Keeping both series in parallel means toggling the UI
// checkbox mid-session does not invalidate the historical baseline -
// each series stays internally consistent.
inline float g_bestSessionIoU2D_occluded = 0.0f;

// [Phase B] Pose Library acceptance gate (Layer 4) toggle. When false
// (default), the Layer 4 check uses registrationHandle.compIoU2D, i.e.
// the method-neutral full-image IoU. When true, the gate uses
// g_lastSilOccludedIoU2D (set by the Ctrl+Shift+G wrapper during Phase
// E). This switches Layer 4 between "method-neutral acceptance" and
// "internally-consistent acceptance with Layers 1-3". Default OFF for
// legacy bit-identical behaviour.
inline bool  g_poseLibraryUseOccludedForAccept = false;

// [Phase B] Last IoU_occluded value computed by Ctrl+Shift+G Phase E
// (= candidate pose evaluated via rasterize_iou2d_v3rs with the same
// instrument mask + threshold the optimiser uses). Set immediately
// before poseSaveToLibrary is called by main.cpp keybinding; consumed
// inside poseSaveToLibrary when the toggle above is ON. Defaulted to
// 0.0 so non-V3RS code paths (HemiAuto, Shift+V, etc.) cleanly fall
// through to "no IoU_occluded available" semantics.
inline float g_lastSilOccludedIoU2D = 0.0f;

// V3R rim-only RMSE diagnostic publication globals. Set by Ctrl+G
// (runBipopCmaesV3R Phase F.5) immediately before poseSaveToLibrary
// is called; consumed-and-cleared inside poseSaveToLibrary so non-
// Ctrl+G saves see "N/A" (compRmseRim = -1.0f).
//
// Display-only — these NEVER participate in the Layer 4 acceptance
// gate. They exist so the operator can see whether the rim band has
// drifted away from the target rim while compRmse is still improving
// (= rim overfitting indicator; see
// HANDOVER_Measurement_Inconsistency_Analysis.md §1).
//
// Restored from PoseEntry by poseApplyEntry so that re-applying a
// Ctrl+G entry repopulates the in-memory diagnostic too.
inline float g_lastRimRmse     = -1.0f;   // -1.0 = not measured / N/A
inline int   g_lastRimMatched  = 0;
inline int   g_lastRimTgtTotal = 0;
inline int   g_lastRimSrcTotal = 0;

// =================================================================
// [Phase A] V3R rim PAIR visualization (Show colored RIM pairs).
// Mirrors the g_lastRim* publish/consume pattern: Ctrl+G / Ctrl+Shift+G
// Phase F.5 captures the (src_full_vert_idx, tgt_world_pos) for every
// rim pair that passed the max_dist_sq gate. poseSaveToLibrary copies
// them onto the PoseEntry (Phase B) and clears the globals; applyEntry
// restores them so re-applying an entry repopulates the in-memory
// state used by the colored-pair viewer.
//
// Size invariant after Phase F.5:
//   g_lastRimPairSrcVertIdx.size() == g_lastRimPairTgtPos.size()
//                                  == g_lastRimMatched
//
// Source: liverMesh3D->mVertices index (full-mesh). Drawing dereferences
//         the current mVertices so the marker follows ICP / Apply updates.
// Target: immutable world coordinate (image-side; never moves).
inline std::vector<int>       g_lastRimPairSrcVertIdx;
inline std::vector<glm::vec3> g_lastRimPairTgtPos;

// =================================================================
// [NEW V3RS-CONTAIN] Containment-direction diagnostic publication.
// Same publish-and-consume pattern as the rim globals above.
// runBipopCmaesV3R Phase F.5b writes the freshly-computed values
// just before the function returns; poseSaveToLibrary reads them
// out and clears the slot (sentinel -1.0f). Non-Ctrl+G save paths
// see "N/A" (compIoUOccPrecision/Recall = -1.0f).
//
// precision = |src∩tgt| / |src|  → 1 means src ⊆ tgt   (no overshoot)
// recall    = |src∩tgt| / |tgt|  → 1 means tgt ⊆ src   (full coverage)
//
// recall > precision  → "source overshoots target"  ← the case the
//                       operator wanted a held-out signal for.
// recall < precision  → "source undershoots target"
//
// Restored from PoseEntry by poseApplyEntry so re-applying an entry
// repopulates the in-memory diagnostic.
inline float g_lastIoUOccPrecision = -1.0f;  // -1.0 = N/A
inline float g_lastIoUOccRecall    = -1.0f;  // -1.0 = N/A

inline std::vector<std::vector<GLfloat>> g_bestSessionVertices;
inline std::vector<std::vector<GLfloat>> g_bestSessionNormals;

// ----- 補助関数 -----
inline std::vector<mCutMesh*> poseGetOrganList() {
    return { liverMesh3D, portalMesh3D, veinMesh3D,
            tumorMesh3D, segmentMesh3D, gbMesh3D };
}

inline glm::mat4 computeCurrentTransform() {
    if (g_initOrganVertices.empty() ||
        g_initOrganVertices[0].empty() ||
        !liverMesh3D) {
        return glm::mat4(1.0f);
    }
    return PoseLibrary::computeTransformFromLiver(
        g_initOrganVertices[0], liverMesh3D->mVertices);
}

inline void poseAutoSaveBeforeRegistration() {
    g_poseLibrary.autoSaveLastRegistration(computeCurrentTransform());
}

inline void poseStartNewSession() {
    g_sessionId++;
    g_sessionBipopN       = 0;
    g_sessionSilhouetteN  = 0;
    g_bestSessionCompRmse = FLT_MAX;
    g_bestSessionIoU2D    = 0.0f;
    g_bestSessionIoU2D_occluded = 0.0f;   // [Phase B] parallel reset
    g_bestSessionVertices.clear();
    g_bestSessionNormals.clear();
    g_currentOrientRunCount = 0;
    g_stepStartTime = std::chrono::steady_clock::now();
    std::cout << "[Session] New session #" << g_sessionId << std::endl;
}

// =========================================================================
//  poseSaveToLibrary
//  -----------------------------------------------------------------------
//  HemiAuto / BIPOP / Umeyama / Shift+E (Silhouette) 完了直後に呼ばれる。
//  - 採択基準 (RMSE/IOU/EITHER) で session-best 改善判定
//  - accept なら entry を追加し、g_bestSession* を更新
//  - reject なら g_bestSession の頂点に巻き戻す
// =========================================================================
inline void poseSaveToLibrary(SaveCriterion crit = SaveCriterion::RMSE,
                              uint8_t quadrant_mask = 0xFF) {   // V3R: 0xFF = legacy
    if (registrationHandle.state != RegistrationData::REGISTERED) {
        std::cout << "[PoseLibrary] No registration to save" << std::endl;
        return;
    }

    float elapsedSec = std::chrono::duration<float>(
                           std::chrono::steady_clock::now() - g_stepStartTime).count();

    float currentRmse        = registrationHandle.compRmse;
    float currentIoU         = registrationHandle.compIoU2D;
    // [Phase B] Snapshot the IoU_occluded value the V3RS Phase E wrapper
    // set before invoking us. Non-V3RS code paths leave this at 0 so the
    // toggle-OFF path behaves bit-identically to the legacy gate.
    // Consume-and-clear semantics: each call gets a fresh value (or 0
    // for non-V3RS paths). Phase E always re-publishes before the next
    // V3RS save; non-V3RS callers see 0 (= "no IoU_occluded available").
    float currentIoU_occluded = g_lastSilOccludedIoU2D;
    g_lastSilOccludedIoU2D    = 0.0f;

    // V3R rim diagnostic: same consume-and-clear pattern as IoU_occluded.
    // Non-Ctrl+G call sites will see g_lastRimRmse == -1.0f (= N/A) which
    // gets recorded onto the entry verbatim. Display-only; never gates.
    float currentRimRmse     = g_lastRimRmse;
    int   currentRimMatched  = g_lastRimMatched;
    int   currentRimTgtTotal = g_lastRimTgtTotal;
    int   currentRimSrcTotal = g_lastRimSrcTotal;
    g_lastRimRmse     = -1.0f;
    g_lastRimMatched  = 0;
    g_lastRimTgtTotal = 0;
    g_lastRimSrcTotal = 0;

    // [NEW V3RS-CONTAIN] Same consume-and-clear pattern. -1 = N/A for
    // non-Ctrl+G paths. Display-only.
    float currentIoUOccPrecision = g_lastIoUOccPrecision;
    float currentIoUOccRecall    = g_lastIoUOccRecall;
    g_lastIoUOccPrecision = -1.0f;
    g_lastIoUOccRecall    = -1.0f;

    // [Phase B FIX] Rim PAIR vectors — DIFFERENT lifecycle from the
    // scalar diagnostics above:
    //
    // The colored-pairs viewer reads `g_lastRimPair*` to render markers
    // every frame. The plain consume-and-clear pattern used for the
    // scalar diagnostics would BLANK the viewer immediately after each
    // Ctrl+G save — which is exactly when the user wants to see the
    // pairs. So we deviate from the pattern here:
    //
    //   Ctrl+G save (currentRimRmse >= 0):
    //     - Snapshot pairs onto the entry (for applyEntry replay)
    //     - KEEP globals populated so the viewer continues to render
    //       the just-saved capture until the next Ctrl+G or applyEntry
    //
    //   Non-Ctrl+G save (currentRimRmse < 0, i.e. Phase F.5 didn't run):
    //     - Save entry with EMPTY pairs (consistent with rimRmse=N/A)
    //     - CLEAR globals so the viewer doesn't render stale Ctrl+G
    //       pairs against a different (post-HemiAuto/etc.) pose
    //
    // currentRimRmse is the canonical "Phase F.5 just captured fresh
    // data" signal (set alongside the pairs, snapshot above just before
    // its own consume-and-clear). Gating on it keeps the two in sync.
    std::vector<int>       currentRimPairSrcVertIdx;
    std::vector<glm::vec3> currentRimPairTgtPos;
    if (currentRimRmse >= 0.0f) {
        // Ctrl+G / Ctrl+Shift+G branch: globals were freshly populated
        // by Phase F.5. Copy onto the entry; keep globals for viewer.
        currentRimPairSrcVertIdx = g_lastRimPairSrcVertIdx;
        currentRimPairTgtPos     = g_lastRimPairTgtPos;
    } else {
        // Non-Ctrl+G branch (HemiAuto, Shift+V, Shift+E, etc.). Entry
        // gets empty pairs; clear globals to invalidate the viewer
        // (mesh is at a new pose; prior Ctrl+G pairs are stale).
        g_lastRimPairSrcVertIdx.clear();
        g_lastRimPairTgtPos.clear();
    }

    auto  organs             = poseGetOrganList();

    PoseEntry::Method method;
    int rm = gUI.state.regMethod;
    if      (rm == 0) method = PoseEntry::FULL_AUTO;
    else if (rm == 1) method = PoseEntry::HEMI_AUTO;
    else if (rm == 2) method = PoseEntry::UMEYAMA;
    else if (rm == 3) method = PoseEntry::BIPOP_CMAES;
    else if (rm == 5) method = PoseEntry::SILHOUETTE_ALIGN;
    else              method = PoseEntry::HEMI_AUTO;

    // [Phase B] Pose Library acceptance gate ("Layer 4")
    //
    // Decide whether this candidate pose improves on the running
    // session-best per the chosen SaveCriterion (RMSE / IOU / EITHER).
    // The IoU series used for the IOU check is selected by
    // g_poseLibraryUseOccludedForAccept:
    //
    //   OFF (default): use registrationHandle.compIoU2D (full IoU,
    //                  method-neutral). Bit-identical to the legacy
    //                  behaviour - both g_bestSessionIoU2D and
    //                  currentIoU come from computeUnifiedMetrics.
    //
    //   ON           : use currentIoU_occluded (set by V3RS Phase E
    //                  via g_lastSilOccludedIoU2D). The corresponding
    //                  session-best, g_bestSessionIoU2D_occluded, is
    //                  maintained in parallel below so the two series
    //                  stay internally consistent regardless of toggle
    //                  changes mid-session.
    //
    // Both series (full / occluded) are updated together on accept so
    // the toggle can be flipped at any time without breaking the
    // session anchor.
    //
    // See DESIGN_Occlusion_Aware_Silhouette_Anchor.md §4.2.
    const bool  useOccluded = g_poseLibraryUseOccludedForAccept;
    const float iouForGate  = useOccluded ? currentIoU_occluded : currentIoU;
    const float iouBest     = useOccluded ? g_bestSessionIoU2D_occluded
                                      : g_bestSessionIoU2D;

    // 採択判定
    bool rmseImproved = (currentRmse <= g_bestSessionCompRmse);
    bool iouImproved  = (iouForGate > iouBest + 1e-4f) && (iouForGate > 0.0f);
    bool accept = false;
    switch (crit) {
    case SaveCriterion::RMSE:   accept = rmseImproved; break;
    case SaveCriterion::IOU:    accept = iouImproved;  break;
    case SaveCriterion::EITHER: accept = rmseImproved || iouImproved; break;
    }

    const char* critName = (crit == SaveCriterion::RMSE)   ? "RMSE"
                           : (crit == SaveCriterion::IOU)    ? "IOU"
                                                          : "EITHER";
    const char* iouSeriesName = useOccluded ? "IoU_occluded" : "IoU_full";

    if (accept) {
        g_currentOrientRunCount++;

        // 「直近accept時」の参照値として更新 (= 現状 = 次回 revert アンカ)
        // [Phase B] Both IoU series advance in parallel so neither
        // baseline goes stale relative to the other.
        float prevRmseRef         = g_bestSessionCompRmse;
        float prevIouRefFull      = g_bestSessionIoU2D;
        float prevIouRefOccluded  = g_bestSessionIoU2D_occluded;
        g_bestSessionCompRmse = currentRmse;
        if (currentIoU > 0.0f)          g_bestSessionIoU2D          = currentIoU;
        if (currentIoU_occluded > 0.0f) g_bestSessionIoU2D_occluded = currentIoU_occluded;

        if (currentRmse < prevRmseRef)
            std::cout << "[Session] CompRMSE reference: " << prevRmseRef
                      << " -> " << currentRmse << " [improved]" << std::endl;
        else if (currentRmse > prevRmseRef)
            std::cout << "[Session] CompRMSE reference: " << prevRmseRef
                      << " -> " << currentRmse
                      << " [regressed but accepted via " << critName << "]" << std::endl;
        // [Phase B] Report whichever IoU series drove the decision; the
        // other one is shown afterwards as a reference so the log is
        // self-contained regardless of toggle state.
        const float prevDecisionRef = useOccluded ? prevIouRefOccluded : prevIouRefFull;
        if (iouForGate > 0.0f) {
            if (iouForGate > prevDecisionRef + 1e-4f)
                std::cout << "[Session] " << iouSeriesName << " reference: "
                          << prevDecisionRef << " -> " << iouForGate << " [improved]" << std::endl;
            else if (iouForGate < prevDecisionRef - 1e-4f)
                std::cout << "[Session] " << iouSeriesName << " reference: "
                          << prevDecisionRef << " -> " << iouForGate
                          << " [regressed but accepted via " << critName << "]" << std::endl;
        }
        if (useOccluded) {
            std::cout << "[Session]   IoU_full[ref]: "
                      << prevIouRefFull << " -> " << currentIoU << std::endl;
        } else if (currentIoU_occluded > 0.0f) {
            std::cout << "[Session]   IoU_occluded[ref]: "
                      << prevIouRefOccluded << " -> " << currentIoU_occluded << std::endl;
        }

        // accept スナップショット (revert アンカ)
        g_bestSessionVertices.resize(organs.size());
        g_bestSessionNormals.resize(organs.size());
        for (size_t i = 0; i < organs.size(); i++) {
            g_bestSessionVertices[i] = organs[i]->mVertices;
            g_bestSessionNormals[i]  = organs[i]->mNormals;
        }

        glm::mat4 T = computeCurrentTransform();
        g_poseLibrary.saveCurrentToLibrary(
            method,
            g_sessionId,
            g_sessionBipopN,
            g_sessionSilhouetteN,
            elapsedSec,
            registrationHandle.fitness,
            registrationHandle.icpRmse,
            registrationHandle.averageError,
            registrationHandle.rmse,
            registrationHandle.maxError,
            registrationHandle.scaleFactor,
            registrationHandle.compRmse,
            registrationHandle.compAvgError,
            registrationHandle.compMaxError,
            registrationHandle.compIoU2D,
            currentIoU_occluded,   // [Phase B] parallel IoU_occluded column
            registrationHandle.compCount,
            registrationHandle.compSource,
            registrationHandle.compTarget,
            T,
            g_currentOrientLabel,
            g_currentOrientRunCount,
            g_trialSeed,    // Phase 2: 保存時の trial seed
            g_callIdx,      // Phase 2: 保存時の call index
            quadrant_mask,  // V3R: Ctrl+G なら 0x01..0x0F、それ以外なら 0xFF
            // V3R rim diag (display-only; -1/0/0/0 for non-Ctrl+G saves)
            currentRimRmse, currentRimMatched,
            currentRimTgtTotal, currentRimSrcTotal,
            // [NEW V3RS-CONTAIN] precision/recall (display-only; -1 = N/A)
            currentIoUOccPrecision, currentIoUOccRecall,
            // [Phase B] rim pair vectors (empty for non-Ctrl+G saves)
            currentRimPairSrcVertIdx, currentRimPairTgtPos);
        // (Apply 後に Shift+V で同じ seed を再現するために使用)

    } else {
        // [Phase B] Reject log shows both IoU series so the user can see
        // why the toggle did or did not save this candidate.
        std::cout << "[Session] Rejected by criterion=" << critName
                  << "  (gate-IoU=" << iouSeriesName << ")"
                  << " : RMSE " << currentRmse << " (best " << g_bestSessionCompRmse << ")"
                  << ", " << iouSeriesName << " " << iouForGate
                  << " (best " << iouBest << ")";
        if (useOccluded) {
            std::cout << "  [ref IoU_full " << currentIoU
                      << " best " << g_bestSessionIoU2D << "]";
        } else if (currentIoU_occluded > 0.0f) {
            std::cout << "  [ref IoU_occluded " << currentIoU_occluded
                      << " best " << g_bestSessionIoU2D_occluded << "]";
        }
        std::cout << " -> reverting" << std::endl;


        if (!g_bestSessionVertices.empty() &&
            g_bestSessionVertices.size() == organs.size()) {
            for (size_t i = 0; i < organs.size(); i++) {
                organs[i]->mVertices = g_bestSessionVertices[i];
                organs[i]->mNormals  = g_bestSessionNormals[i];
                setUp(*organs[i]);
            }
            registrationHandle.state           = RegistrationData::REGISTERED;
            registrationHandle.useRegistration = true;
            computeUnifiedMetrics();
            std::cout << "[Session] Reverted. CompRMSE=" << registrationHandle.compRmse
                      << " IoU2D=" << registrationHandle.compIoU2D << std::endl;
        }
    }
}

// =========================================================================
//  poseApplyEntry
//  -----------------------------------------------------------------------
//  Pose Library のエントリを現在の臓器メッシュに適用。
//  メトリクスは entry に保存済みの値を registrationHandle に書き戻したあと、
//  computeUnifiedMetrics() で再計算して再現性 (saved vs reproduced) をログ出力。
// =========================================================================
inline void poseApplyEntry(int entryId) {
    auto organs = poseGetOrganList();
    if (!g_poseLibrary.applyEntry(entryId, g_initOrganVertices, g_initOrganNormals, organs)) {
        return;
    }
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;

    float savedCompRmse = 0.0f;
    for (auto& e : g_poseLibrary.entries) {
        if (e.id != entryId) continue;
        savedCompRmse                   = e.compRmse;
        registrationHandle.fitness      = e.baseFitness;
        registrationHandle.icpRmse      = e.baseIcpRmse;
        registrationHandle.averageError = e.baseAvgError;
        registrationHandle.rmse         = e.baseRmse;
        registrationHandle.maxError     = e.baseMaxError;
        registrationHandle.scaleFactor  = e.baseScale;
        registrationHandle.compRmse     = e.compRmse;
        registrationHandle.compAvgError = e.compAvgError;
        registrationHandle.compMaxError = e.compMaxError;
        registrationHandle.compIoU2D    = e.compIoU2D;
        registrationHandle.compCount    = e.compCount;
        registrationHandle.compSource   = e.corrSource;
        registrationHandle.compTarget   = e.corrTarget;
        if      (e.baseMethod == PoseEntry::FULL_AUTO)        gUI.state.regMethod = 0;
        else if (e.baseMethod == PoseEntry::HEMI_AUTO)        gUI.state.regMethod = 1;
        else if (e.baseMethod == PoseEntry::UMEYAMA)          gUI.state.regMethod = 2;
        else if (e.baseMethod == PoseEntry::BIPOP_CMAES)      gUI.state.regMethod = 3;
        else if (e.baseMethod == PoseEntry::SILHOUETTE_ALIGN) gUI.state.regMethod = 5;
        else                                                  gUI.state.regMethod = 1;

        // Phase 2: シード復元
        // この entry が保存された時点の (g_trialSeed, g_callIdx) を書き戻す。
        // これにより Apply 直後に Shift+V (BIPOP-CMA-ES) を実行すると、
        // AutoProbe 中に同じ probe で CMA-ES を呼んだ場合と同じシードが
        // 使われ、bit-identical な結果が得られる。
        // (旧 CSV からインポートしたエントリは savedTrialSeed=0,
        //  savedCallIdx=0 のままなので、この場合のみ書き戻しをスキップして
        //  現状の値を維持する)
        if (e.savedTrialSeed != 0 || e.savedCallIdx != 0) {
            const uint32_t prev_trial = g_trialSeed;
            const uint32_t prev_call  = g_callIdx;
            g_trialSeed = e.savedTrialSeed;
            g_callIdx   = e.savedCallIdx;
            std::cout << "[PoseLibrary] Seed restored from entry #" << entryId
                      << "  trial_seed: " << prev_trial << " -> " << g_trialSeed
                      << "  call_idx: " << prev_call << " -> " << g_callIdx
                      << std::endl;
        } else {
            std::cout << "[PoseLibrary] Entry #" << entryId
                      << " has no saved seed (legacy entry); g_trialSeed/g_callIdx unchanged"
                      << std::endl;
        }
        break;
    }

    // 再現性チェック: 保存時の compRmse と再計算した compRmse を比較
    computeUnifiedMetrics();
    float reproRmse = registrationHandle.compRmse;
    float diff      = std::abs(reproRmse - savedCompRmse);
    std::cout << "[PoseLibrary] Reproduction check entry #" << entryId << std::endl;
    std::cout << "  Saved  CompRMSE: " << savedCompRmse << std::endl;
    std::cout << "  Repro  CompRMSE: " << reproRmse << std::endl;
    std::cout << "  Diff:            " << diff
              << (diff < 1e-4f ? "  [OK]" : "  [WARN: drift detected]") << std::endl;

    // session-best を「適用したエントリ」にリセット
    // (これをやらないと別ポーズの古い best と比較されて誤った reject が起きる)
    // [Phase B] g_bestSessionIoU2D_occluded もパラレルにリセット。
    // poseApplyEntry は computeUnifiedMetrics() で compIoU2D (full) を
    // 再計算するが、IoU_occluded は entry に保存された値を信頼する
    // (Apply 直後に Ctrl+Shift+G を打てば g_lastSilOccludedIoU2D が
    // 上書きされるため、apply 後の immediate state は entry のもの)。
    g_bestSessionCompRmse        = registrationHandle.compRmse;
    g_bestSessionIoU2D           = registrationHandle.compIoU2D;
    g_bestSessionIoU2D_occluded  = 0.0f;   // [Phase B] reset; reads from entry below
    g_lastSilOccludedIoU2D       = 0.0f;   // [Phase B] stale value cleared
    // V3R rim diag: same pattern -- repopulate from entry so subsequent
    // poseSaveToLibrary calls (e.g. after Apply+Ctrl+G chain) see the
    // applied entry's rim state as the baseline.
    g_lastRimRmse                = -1.0f;
    g_lastRimMatched             = 0;
    g_lastRimTgtTotal            = 0;
    g_lastRimSrcTotal            = 0;
    // [NEW V3RS-CONTAIN] Reset containment globals to N/A; loop below
    // overwrites with the applied entry's stored values when found.
    g_lastIoUOccPrecision        = -1.0f;
    g_lastIoUOccRecall           = -1.0f;
    // [Phase B] Reset rim PAIR globals; loop below restores from entry
    // so the colored-pairs viewer sees the applied entry's saved pairs
    // immediately after Apply (no need to re-run Ctrl+G).
    g_lastRimPairSrcVertIdx.clear();
    g_lastRimPairTgtPos.clear();
    for (auto& e : g_poseLibrary.entries) {
        if (e.id == entryId) {
            g_bestSessionIoU2D_occluded = e.compIoU2D_occluded;
            g_lastSilOccludedIoU2D      = e.compIoU2D_occluded;
            g_lastRimRmse               = e.compRmseRim;
            g_lastRimMatched            = e.rimMatched;
            g_lastRimTgtTotal           = e.rimTgtTotal;
            g_lastRimSrcTotal           = e.rimSrcTotal;
            g_lastIoUOccPrecision       = e.compIoUOccPrecision;
            g_lastIoUOccRecall          = e.compIoUOccRecall;
            // [Phase B] Restore rim pair vectors so the colored-pairs
            // viewer can draw the applied entry's pairs in their captured
            // configuration. Empty for non-Ctrl+G entries (viewer skips).
            g_lastRimPairSrcVertIdx     = e.rimPairSrcVertIdx;
            g_lastRimPairTgtPos         = e.rimPairTgtPos;
            break;
        }
    }
    g_bestSessionVertices.resize(organs.size());
    g_bestSessionNormals.resize(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            g_bestSessionVertices[i] = organs[i]->mVertices;
            g_bestSessionNormals[i]  = organs[i]->mNormals;
        }
    }
    std::cout << "[Session] Reset session-best to applied entry: "
              << "CompRMSE=" << g_bestSessionCompRmse
              << " IoU2D=" << g_bestSessionIoU2D
              << " IoU_occ=" << g_bestSessionIoU2D_occluded
              << std::endl;

    // [NEW V3RS-POSE-APPLY-F9] Refresh F9 Final slot so the user sees
    // the applied entry's silhouette without needing to re-run Ctrl+G
    // or Ctrl+Shift+G. Mirrors Ctrl+G's Phase F.5c exactly:
    //
    // - Uses Ctrl+G-style 3-panel layout (rim_sil_max_px = 0) regardless
    //   of which method originally produced the entry. This keeps F9
    //   layout STABLE across consecutive Applies of mixed-method entries
    //   (3-panel = Ctrl+G, 6-panel = Ctrl+Shift+G would flip every click
    //   otherwise, which would be jarring). Trade-off documented:
    //   Ctrl+Shift+G entries lose their rim diagnostic bottom row on
    //   Apply re-display, but the top-row silhouette + IoU + Containment
    //   (which is what users mostly look at) is preserved.
    //
    // - Uses full-mesh indices (not quadrant-filtered). The applied pose
    //   committed the WHOLE liver to liverMesh3D, so showing the whole
    //   silhouette matches the mesh state on screen.
    //
    // - Instrument-occlusion: AUTO (same policy as Ctrl+G F.5c and the
    //   Containment F.5b in both V3RS methods). Stable semantics across
    //   methods.
    //
    // - scale_value: derived from the entry's cumulative transform via
    //   cube root of the 3x3 sub-matrix determinant. For a pure-rotation
    //   entry this yields 1.0; for a V3RS solution with scale 1.08 it
    //   yields 1.08. Display-only.
    //
    // Cost: one rasterize_iou2d_v3rs pass at step=8 (~15 ms). Skipped
    // silently when SAM2 boundary map is unavailable.
    if (g_boundaryDistMap.valid
        && g_boundaryDistMap.width  > 1
        && g_boundaryDistMap.height > 1
        && g_boundaryDistMap.data.size() ==
               (size_t)g_boundaryDistMap.width *
                   (size_t)g_boundaryDistMap.height
        && liverMesh3D
        && !liverMesh3D->mIndices.empty())
    {
        const glm::mat4 sil_view_pa = buildSilhouetteView();
        const glm::mat4 sil_proj_pa = buildSilhouetteProj();
        const int sil_w_pa = g_boundaryDistMap.width;
        const int sil_h_pa = g_boundaryDistMap.height;

        std::vector<uint32_t> sil_indices_pa(
            liverMesh3D->mIndices.begin(),
            liverMesh3D->mIndices.end());

        // Instrument-occlusion mask: AUTO. Identical policy to Ctrl+G F.5c.
        const std::vector<float>* inst_ptr_pa = nullptr;
        float inst_thresh_pa = 0.0f;
        const bool inst_loaded_pa = ensureInstrumentDistMap();
        if (inst_loaded_pa
            && g_instrumentDistMap.valid
            && g_instrumentDistMap.width  == sil_w_pa
            && g_instrumentDistMap.height == sil_h_pa
            && g_instrumentDistMap.data.size() ==
                   (size_t)sil_w_pa * (size_t)sil_h_pa)
        {
            inst_ptr_pa    = &g_instrumentDistMap.data;
            inst_thresh_pa = std::max(0.0f, g_instrumentPxThresh);
        }

        // Cumulative scale of the applied entry transform (cosmetic, F9
        // shows "scale = X.XX" next to IoU). lastRegistration.transform
        // was set by applyEntry above to e.transform, so we can read it
        // directly without re-iterating the entries vector.
        const float det_pa = glm::determinant(
            glm::mat3(g_poseLibrary.lastRegistration.transform));
        const float entry_scale_pa = (det_pa > 0.0f)
                                         ? std::cbrt(det_pa) : 1.0f;

        // best_run_idx = -1: there is no per-Run captioning for Apply
        // (the F9 "Best Run was N" line just hides). rim_sil_max_px = 0
        // forces 3-panel layout (see comment block above).
        SilOverlay::captureFinal(
            SilOverlay::g_silOverlay, /*best_run_idx=*/-1, liverMesh3D,
            sil_indices_pa, sil_view_pa, sil_proj_pa, g_boundaryDistMap.data,
            sil_w_pa, sil_h_pa, /*step=*/8, entry_scale_pa,
            inst_ptr_pa, inst_thresh_pa,
            /*rim_sil_max_px=*/0.0f,
            /*is_rim_anatomic_per_vertex=*/nullptr);

        std::cout << "[PoseLibrary/F9] Refreshed Final slot for entry #"
                  << entryId << "  (scale=" << entry_scale_pa
                  << ", inst=" << (inst_ptr_pa ? "ON" : "OFF") << ")"
                  << std::endl;
    } else {
        std::cout << "[PoseLibrary/F9] Skipped Final slot refresh for "
                  << "entry #" << entryId
                  << "  (boundary map invalid or mesh empty)" << std::endl;
    }
}

inline void poseUndo() {
    auto organs = poseGetOrganList();
    g_poseLibrary.undoToLast(g_initOrganVertices, g_initOrganNormals, organs);
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    computeUnifiedMetrics();
}

// =========================================================================
//  runAutoProbe
//  -----------------------------------------------------------------------
//  Fibonacci 球面 N 点を初期姿勢として、各点で HemiAuto を実行し、結果を
//  Pose Library に保存する。論文用評価プロトコルの基盤。
//
//  N の選択について:
//    - N=36  : 最低限のカバー (~33° 間隔)。プレリミナリ用。
//    - N=72  : 標準 (~24° 間隔)。
//    - N=108 : ★現在のデフォルト。~19° 間隔で FGR の capture radius
//              (~15-25°) を確実にカバー。論文の主結果に十分。
//    - N=144 : 厚め (~17° 間隔)。
//  N は constexpr int N で関数冒頭に定義してあるので、値だけ変えれば
//  他のスケジュールにも切り替えられる。将来 N をスイープして
//  success rate vs N をプロットする場合も、この値を変えるだけで対応可能。
//
//  - 開始時のメッシュ状態をスナップショット → 各 probe で復元 → 回転 → HemiAuto
//  - 全 N entry を「強制保存」(session-best を毎回 FLT_MAX にリセット)
//  - Undo snapshot は AutoProbe 全体で 1 個 (X キーで全体取り消し)
//  - g_currentOrientLabel = "AutoProbe" → エントリは "AutoProbe#1..#N"
//
//  ★ AR カメラ固定 (再現性確保) ★
//  runHemiAuto は OrbitCam.cameraPos / cameraTarget を読んで
//  extractVisibleVerticesCustom に渡し、可視頂点集合を決める。これらは
//  マウス操作 (回転/パン/ズーム) で run 間に微妙に変動するため、同じ入力で
//  AutoProbe を 2 回実行しても結果が一部ズレる原因になっていた。
//
//  ここでは AutoProbe ループ突入前に OrbitCam.cameraPos / cameraTarget を
//  AR カメラと同じ固定値 (cameraPos=(0,0,0), cameraTarget=(0,0,1)) に
//  上書きし、ループ終了後に元の値に復元する。これにより:
//    - マウス位置に依存せず N probe 全部が同じ AR 視点から可視判定される
//    - run 間で完全に bit-identical な結果が得られる (再現性 100%)
//    - SAM2 マスク取得時のカメラ (= AR カメラ) と一貫した評価
//  IoU2D は既に buildSilhouetteView/Proj で AR 固定なので変更不要。
//  3D RMSE は OrbitCam に非依存 (extractFrontFacePoints 経由) なので不要。
//  performRegistrationSingleMesh の camera_position 引数は dead parameter
//  なのでそちらの上書きも不要。
// =========================================================================
inline void runAutoProbe() {
    constexpr int N = 12;   // Fibonacci 球面の点数 (~19° 間隔。値変更可)

    auto organs = poseGetOrganList();
    if (organs.empty() || !organs[0]) {
        std::cerr << "[AutoProbe] No liver mesh available" << std::endl;
        return;
    }

    // -------------------------------------------------------------------
    //  AR カメラ固定 (Phase 2: 再現性確保)
    //  -------------------------------------------------------------------
    //  OrbitCam.cameraPos / cameraTarget を保存 → AR 固定値で上書き。
    //  ループ終了後 (関数末尾) で必ず復元する。
    //  AR モード中 (gApp.arMode == true) でも冪等に上書きする
    //  (AR モード中はマウスコールバック early-return で rotation は凍結
    //   するが、cameraPos/cameraTarget は AR 突入前の orbit 値が残っている
    //   だけで (0,0,0)/(0,0,1) ではないため、明示的な上書きが必要)。
    // -------------------------------------------------------------------
    const glm::vec3 saved_cameraPos    = OrbitCam.cameraPos;
    const glm::vec3 saved_cameraTarget = OrbitCam.cameraTarget;
    OrbitCam.cameraPos    = glm::vec3(0.0f, 0.0f, 0.0f);
    OrbitCam.cameraTarget = glm::vec3(0.0f, 0.0f, 1.0f);
    std::cout << "[AutoProbe] OrbitCam locked to AR view  pos=(0,0,0)  target=(0,0,1)"
              << "  (saved pos=(" << saved_cameraPos.x << "," << saved_cameraPos.y
              << "," << saved_cameraPos.z << ")"
              << "  target=(" << saved_cameraTarget.x << "," << saved_cameraTarget.y
              << "," << saved_cameraTarget.z << "))" << std::endl;

    // -------------------------------------------------------------------
    //  callIdx は呼び出し元が管理:
    //    - 単体 AutoProbe ボタン: 事前に g_callIdx = 0
    //    - IterAutoProbe:         事前に g_callIdx = k * N
    //  これにより IterAutoProbe の各サイクルで異なるシード範囲が使われる。
    // -------------------------------------------------------------------
    const uint32_t callIdx_start = g_callIdx;
    std::cout << "[Seed] AutoProbe start  trial_seed=" << g_trialSeed
              << "  callIdx=" << callIdx_start
              << ".." << (callIdx_start + N - 1) << std::endl;

    // -------------------------------------------------------------------
    //  ベースライン: g_initOrganVertices (prealignment 直後の初期姿勢)
    //  ★ 現在のメッシュ状態ではなく初期姿勢を使うことで、
    //    - IterAutoProbe の各サイクルが独立した試行になる
    //    - どの InitRot プリセットからでも同じ結果になる
    // -------------------------------------------------------------------
    if (g_initOrganVertices.empty() || g_initOrganVertices.size() != organs.size()) {
        std::cerr << "[AutoProbe] g_initOrganVertices not available" << std::endl;
        OrbitCam.cameraPos = saved_cameraPos;
        OrbitCam.cameraTarget = saved_cameraTarget;
        return;
    }

    std::vector<std::vector<GLfloat>> baseV(organs.size());
    std::vector<std::vector<GLfloat>> baseN(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            baseV[i] = g_initOrganVertices[i];
            baseN[i] = g_initOrganNormals[i];
        }
    }

    // Undo snapshot は AutoProbe 全体で 1 個
    poseAutoSaveBeforeRegistration();

    glm::vec3 centroid = Reg3DCustom::computeMeshCenter(*organs[0]);

    std::cout << "\n=== AutoProbe " << N << " (Fibonacci sphere) ===" << std::endl;
    std::cout << "[AutoProbe] Centroid: ("
              << centroid.x << ", " << centroid.y << ", " << centroid.z
              << ")" << std::endl;

    // 新セッション開始 (AutoProbe 全体で 1 セッション)
    poseStartNewSession();
    g_currentOrientLabel = "AutoProbe";
    g_currentOrientRunCount = 0;  // poseSaveToLibrary が accept ごとに ++

    // Fibonacci 球面 N 方向
    constexpr float kPI         = 3.14159265358979f;
    constexpr float kSqrt5      = 2.2360679774997896964f;
    constexpr float kGoldenAngle = kPI * (3.0f - kSqrt5);  // π·(3-√5) [rad]

    int success_count = 0;
    int saved_count   = 0;

    for (int i = 0; i < N; i++) {
        // 球面上の i 番目の点 (y: +1 → -1, theta: 黄金角 step)
        float y      = 1.0f - (2.0f * i) / float(N - 1);
        float radius = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float theta  = i * kGoldenAngle;
        glm::vec3 dir(std::cos(theta) * radius, y, std::sin(theta) * radius);

        // 初期メッシュ状態を復元
        for (size_t j = 0; j < organs.size(); j++) {
            if (organs[j]) {
                organs[j]->mVertices = baseV[j];
                organs[j]->mNormals  = baseN[j];
                setUp(*organs[j]);
            }
        }

        // 標準正面 (+Z) → dir への回転を計算
        glm::vec3 from(0.0f, 0.0f, 1.0f);
        float dotVal = glm::dot(from, dir);
        glm::mat4 R(1.0f);
        if (dotVal > 0.9999f) {
            R = glm::mat4(1.0f);
        } else if (dotVal < -0.9999f) {
            // 反対向き: Y 軸まわり 180°
            R = glm::rotate(glm::mat4(1.0f), kPI, glm::vec3(0.0f, 1.0f, 0.0f));
        } else {
            glm::vec3 axis = glm::normalize(glm::cross(from, dir));
            float angle = std::acos(glm::clamp(dotVal, -1.0f, 1.0f));
            R = glm::rotate(glm::mat4(1.0f), angle, axis);
        }

        // 重心ピボットで適用: T(c) * R * T(-c)
        glm::mat4 toOrigin   = glm::translate(glm::mat4(1.0f), -centroid);
        glm::mat4 fromOrigin = glm::translate(glm::mat4(1.0f),  centroid);
        glm::mat4 transform  = fromOrigin * R * toOrigin;
        for (auto* m : organs) {
            if (m) applyMatrixToMeshVerticesAndNormals(m, transform);
        }

        g_stepStartTime = std::chrono::steady_clock::now();

        std::cout << "\n[AutoProbe " << (i + 1) << "/" << N << "]  dir=("
                  << dir.x << ", " << dir.y << ", " << dir.z << ")" << std::endl;

        // HemiAuto 実行
        runHemiAuto();

        // visible vertex 不足等で REGISTERED にならなかった場合はスキップ
        if (registrationHandle.state != RegistrationData::REGISTERED) {
            std::cout << "[AutoProbe " << (i + 1) << "/" << N
                      << "] HemiAuto failed (not REGISTERED), skipping save"
                      << std::endl;
            continue;
        }

        // 全 probe を強制保存するため session-best をリセット
        // (こうすることで rmseImproved = (rmse <= FLT_MAX) = true となり必ず accept)
        g_bestSessionCompRmse        = FLT_MAX;
        g_bestSessionIoU2D           = 0.0f;
        g_bestSessionIoU2D_occluded  = 0.0f;   // [Phase B] parallel reset

        // Save (RMSE 基準で必ず accept される)
        gUI.state.regMethod = 1;  // HemiAuto
        poseSaveToLibrary(SaveCriterion::RMSE);
        saved_count++;

        if (registrationHandle.compRmse > 0.0f && registrationHandle.compRmse < 0.05f) {
            success_count++;
        }
    }

    std::cout << "\n=== AutoProbe Complete ===" << std::endl;
    std::cout << "[AutoProbe] Saved: " << saved_count << "/" << N << " entries"
              << std::endl;
    std::cout << "[AutoProbe] RMSE < 0.05: " << success_count << "/" << N
              << " runs" << std::endl;

    // -------------------------------------------------------------------
    //  AR カメラ固定の解除 (Phase 2)
    //  -------------------------------------------------------------------
    //  保存しておいた OrbitCam.cameraPos / cameraTarget を復元。
    //  この後 main ループの OrbitCam.UpdateCamera(dt) が呼ばれると、
    //  rotation/gRadius/cameraTarget から cameraPos が再計算されるが、
    //  rotation 等はこの関数中で触っていないので元の視点に戻る。
    // -------------------------------------------------------------------
    OrbitCam.cameraPos    = saved_cameraPos;
    OrbitCam.cameraTarget = saved_cameraTarget;
    std::cout << "[AutoProbe] OrbitCam restored  pos=("
              << saved_cameraPos.x << "," << saved_cameraPos.y << ","
              << saved_cameraPos.z << ")  target=("
              << saved_cameraTarget.x << "," << saved_cameraTarget.y << ","
              << saved_cameraTarget.z << ")" << std::endl;
}

// =========================================================================
//  runIterativeAutoProbe(int K)
// -------------------------------------------------------------------------
//  N 方向 × K 独立シード = N*K 回の独立試行を実行する。
//  全サイクルが g_initOrganVertices (初期姿勢) から開始するため、
//  baseline-drift ではなく multi-start 確率的最適化として機能する。
//
//  シードレイアウト (N=20 の場合):
//    Cycle 0: g_callIdx =   0..19  → FGR seeds trialSeed+0  .. +19
//    Cycle 1: g_callIdx =  20..39  → FGR seeds trialSeed+20 .. +39
//    ...
//    Cycle k: g_callIdx = k*N .. (k+1)*N-1
//
//  再現性: 同じ (trialSeed, K, N) → bit-identical な結果。
//          PoseLibrary の各エントリに savedCallIdx が保存されるため、
//          Apply → Shift+V でも一意なシードが再現される。
// =========================================================================
inline void runIterativeAutoProbe(int K) {
    if (K <= 0) {
        std::cerr << "[IterAutoProbe] Invalid K=" << K << std::endl;
        return;
    }

    constexpr int N = 20;  // runAutoProbe 内の N と一致させること

    auto t_start = std::chrono::steady_clock::now();
    std::cout << "\n##########  IterAutoProbe START  K=" << K
              << "  N=" << N << "  total=" << K * N << " trials"
              << "  trialSeed=" << g_trialSeed
              << "  ##########" << std::endl;

    for (int k = 0; k < K; k++) {
        g_callIdx = (uint32_t)(k * N);  // サイクルごとに異なるシード範囲
        std::cout << "\n----------  cycle " << (k + 1) << "/" << K
                  << "  callIdx=" << g_callIdx << ".." << (g_callIdx + N - 1)
                  << "  ----------" << std::endl;
        runAutoProbe();
    }

    float dt = std::chrono::duration<float>(
                   std::chrono::steady_clock::now() - t_start).count();
    std::cout << "\n##########  IterAutoProbe DONE  " << K * N
              << " trials in " << dt << " s  ##########" << std::endl;
}

// =========================================================================
//  drawPoseLibraryWindow
//  -----------------------------------------------------------------------
//  ImGui ウィンドウ。元コード (line 2602-) を忠実に移植。
//  Refine カラムのみ削除 (Refine 自体が無くなったため)。
//  カラム構成: # | Session | Method | BIPOP | Silh | CompRMSE | IoU2D | N | Time | Apply
//  (元は Refine カラムを含む 11 列、Refine削除で 10 列)
// =========================================================================
inline void drawPoseLibraryWindow() {
    if (!g_poseLibrary.showWindow) return;

    ImGui::SetNextWindowSize(ImVec2(640, 420), ImGuiCond_FirstUseEver);
    ImGui::PushStyleColor(ImGuiCol_WindowBg,      ImVec4(0.06f,0.06f,0.08f,0.95f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg,       ImVec4(0.12f,0.10f,0.18f,1.0f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.20f,0.15f,0.30f,1.0f));

    if (ImGui::Begin("Pose Library", &g_poseLibrary.showWindow)) {
        ImGui::Text("Entries: %d / %d  |  Session #%d",
                    (int)g_poseLibrary.entries.size(),
                    g_poseLibrary.maxEntries, g_sessionId);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 248);
        {
            static bool s_importGuard = false;
            if (ImGui::Button("Import CSV", ImVec2(120,0)) && !s_importGuard) {
                s_importGuard = true;
#ifdef HAS_TINYFILEDIALOGS
                const char* filters[] = {"*.csv"};
                const char* sel = tinyfd_openFileDialog(
                    "Import Pose Library CSV", "", 1, filters, "CSV Files (*.csv)", 0);
                if (sel) g_poseLibrary.importFromCsv(std::string(sel));
#else
                std::cerr << "[PoseLibrary] Build with -DHAS_TINYFILEDIALOGS for file picker." << std::endl;
#endif
            } else { s_importGuard = false; }
        }
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
        if (ImGui::Button("Export CSV", ImVec2(120,0))) {
            auto now = std::chrono::system_clock::now();
            auto tt  = std::chrono::system_clock::to_time_t(now);
            auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) % 1000;
            std::tm tm = *std::localtime(&tt);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "pose_library_%04d%02d%02d_%02d%02d%02d_%03d.csv",
                          tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
                          tm.tm_hour, tm.tm_min, tm.tm_sec, (int)ms.count());
            g_poseLibrary.exportToCsv(buf);
        }
        // [Phase B] Pose Library acceptance gate (Layer 4) toggle.
        // OFF (default): legacy bit-identical behaviour using IoU_full.
        // ON           : V3RS Phase A consistency - Layer 4 uses
        //                IoU_occluded set by the Ctrl+Shift+G wrapper.
        // The toggle is per-process (not persisted) so each run starts in
        // the documented default. See DESIGN_*.md §4.2 for trade-offs.
        ImGui::Checkbox("Use IoU_occluded for accept gate (Layer 4)",
                        &g_poseLibraryUseOccludedForAccept);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(
                "OFF (default): Pose Library compares IoU_full (method-neutral).\n"
                "ON           : Pose Library compares IoU_occluded (matches\n"
                "               Layers 1/2/3 of V3RS internal optimization).\n\n"
                "Toggle effect is immediate; both IoU series are recorded on\n"
                "every accepted entry regardless of this setting.");
            ImGui::EndTooltip();
        }
        ImGui::Separator();

        // # | Session | Method | BIPOP | Silh | CompRMSE | IoU2D | IoU_occ | RIM | N | Time | [Apply]
        // [Phase B] IoU_occ inserted as column 7; subsequent indices shifted by +1.
        // V3R rim diag: RIM inserted as column 8 (after IoU_occ);
        // subsequent indices shifted by +1 again.
        // [NEW V3RS-CONTAIN] CONT inserted as column 9 (after RIM); N/Time/Apply
        // indices shifted by +1.
        const int kCols = 13;
        auto setupColWidths = [](){
            ImGui::SetColumnWidth(0,  26);   // #
            ImGui::SetColumnWidth(1,  76);   // Session
            ImGui::SetColumnWidth(2,  60);   // Method (60 to fit "SilAln")
            ImGui::SetColumnWidth(3,  36);   // BIPOP
            ImGui::SetColumnWidth(4,  36);   // Silh
            ImGui::SetColumnWidth(5,  70);   // CompRMSE
            ImGui::SetColumnWidth(6,  60);   // IoU2D (= IoU_full)
            ImGui::SetColumnWidth(7,  60);   // IoU_occ (= IoU_occluded)  [Phase B]
            ImGui::SetColumnWidth(8,  92);   // RIM (rmse + match%)  [V3R rim diag]
            ImGui::SetColumnWidth(9,  84);   // CONT (recall/prec)  [V3RS-CONTAIN]
            ImGui::SetColumnWidth(10, 40);   // N
            ImGui::SetColumnWidth(11, 44);   // Time
            ImGui::SetColumnWidth(12, 46);   // Apply
        };

        ImGui::Columns(kCols, "pose_cols", true);
        setupColWidths();

        auto hc = ImVec4(0.7f,0.7f,0.7f,1);
        ImGui::TextColored(hc, "#");        ImGui::NextColumn();
        ImGui::TextColored(hc, "Session");  ImGui::NextColumn();
        ImGui::TextColored(hc, "Method");   ImGui::NextColumn();
        ImGui::TextColored(hc, "BIPOP");    ImGui::NextColumn();
        ImGui::TextColored(hc, "Silh");     ImGui::NextColumn();
        ImGui::TextColored(hc, "CompRMSE"); ImGui::NextColumn();
        ImGui::TextColored(hc, "IoU2D");    ImGui::NextColumn();
        ImGui::TextColored(hc, "IoU_occ");  ImGui::NextColumn();   // [Phase B]
        ImGui::TextColored(hc, "RIM");      ImGui::NextColumn();   // V3R rim diag
        ImGui::TextColored(hc, "CONT");     ImGui::NextColumn();   // V3RS-CONTAIN
        ImGui::TextColored(hc, "N");        ImGui::NextColumn();
        ImGui::TextColored(hc, "Time");     ImGui::NextColumn();
        ImGui::TextColored(hc, "");         ImGui::NextColumn();
        ImGui::Separator();

        int applyId       = -1;
        int prevSessionId = -1;

        for (size_t i = 0; i < g_poseLibrary.entries.size(); i++) {
            auto& e = g_poseLibrary.entries[i];
            bool isActive = (e.id == g_poseLibrary.activeEntryId);

            if (prevSessionId >= 0 && e.sessionId != prevSessionId) {
                ImGui::Columns(1);
                ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.3f,0.3f,0.5f,0.8f));
                ImGui::Separator();
                ImGui::PopStyleColor();
                ImGui::Columns(kCols, "pose_cols", true);
                setupColWidths();
            }
            prevSessionId = e.sessionId;

            if (isActive) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f,1.0f,0.3f,1.0f));

            ImGui::Text("%d", (int)(i+1)); ImGui::NextColumn();

            ImGui::TextColored(ImVec4(0.55f,0.80f,1.0f,1.0f), "%s", e.sessionLabel().c_str());
            ImGui::NextColumn();

            {
                ImVec4 mc = ImVec4(0.85f,0.85f,0.85f,1);
                if (e.baseMethod == PoseEntry::HEMI_AUTO)        mc = ImVec4(0.94f,0.56f,0.19f,1);
                if (e.baseMethod == PoseEntry::BIPOP_CMAES)      mc = ImVec4(0.94f,0.56f,0.19f,1);
                if (e.baseMethod == PoseEntry::UMEYAMA)          mc = ImVec4(0.55f,0.80f,1.0f,1);
                if (e.baseMethod == PoseEntry::SILHOUETTE_ALIGN) mc = ImVec4(0.85f,0.40f,0.95f,1);
                ImGui::TextColored(mc, "%s", e.methodStr());
                // V3R: Ctrl+G セッションは "Q:AR+AL" 等のタグを Method 名の
                // 右側に淡色で並べて表示する。Legacy エントリ (quadrant_mask
                // == 0xFF) は何も表示しない。
                if (e.quadrant_mask != 0xFF) {
                    ImGui::SameLine(0.0f, 6.0f);
                    ImGui::TextColored(ImVec4(0.55f, 0.80f, 1.0f, 1.0f),
                                       "%s",
                                       LiverLeftRightLabel::quadrantMaskString(
                                           e.quadrant_mask).c_str());
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("ID: %d  Session: %s", e.id, e.sessionLabel().c_str());
                    ImGui::Text("Timestamp: %s", e.timestamp.c_str());
                    ImGui::Text("Elapsed: %.3f s", e.elapsedSec);
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(1,1,0.5f,1), "=== Unified Metrics ===");
                    ImGui::Text("Comp RMSE:   %.6f  (%d pairs)", e.compRmse, e.compCount);
                    ImGui::Text("Comp AvgErr: %.6f", e.compAvgError);
                    ImGui::Text("Comp MaxErr: %.6f", e.compMaxError);
                    if (e.compIoU2D > 0.0f)
                        ImGui::Text("Comp IoU2D:  %.4f", e.compIoU2D);
                    else
                        ImGui::Text("Comp IoU2D:  (not measured)");
                    // [Phase B] Show parallel IoU_occluded if available
                    if (e.compIoU2D_occluded > 0.0f)
                        ImGui::Text("Comp IoU_occ: %.4f", e.compIoU2D_occluded);
                    // V3R rim diagnostic block (Ctrl+G entries only).
                    if (e.compRmseRim >= 0.0f && e.rimTgtTotal > 0) {
                        const float pct = 100.0f * (float)e.rimMatched
                                          / (float)e.rimTgtTotal;
                        ImGui::Text("RIM RMSE:    %.6f m  (%d/%d = %.2f%%)",
                                    e.compRmseRim,
                                    e.rimMatched, e.rimTgtTotal, pct);
                        ImGui::Text("Source rim:  %d verts", e.rimSrcTotal);
                    } else {
                        ImGui::Text("RIM:         (not measured)");
                    }
                    // [NEW V3RS-CONTAIN] Containment block — only shown
                    // when measured (Ctrl+G or Ctrl+Shift+G entries with
                    // F.5b populated).
                    if (e.compIoUOccPrecision >= 0.0f &&
                        e.compIoUOccRecall    >= 0.0f) {
                        const float dir = e.compIoUOccRecall
                                          - e.compIoUOccPrecision;
                        const char* tag =
                            (std::fabs(dir) < 0.05f) ? "balanced"
                            : (dir > 0.0f)       ? "overshoot src>tgt"
                                                     : "undershoot src<tgt";
                        // [NEW V3RS-CONTAIN-RATIO] Lead with size_ratio +
                        // overshoot_fraction; recall/precision below as
                        // breakdown.
                        const float sr = (e.compIoUOccPrecision > 1e-6f)
                                             ? e.compIoUOccRecall / e.compIoUOccPrecision
                                             : 0.0f;
                        const float of = (e.compIoUOccPrecision > 1e-6f)
                                             ? e.compIoUOccRecall *
                                                   (1.0f - e.compIoUOccPrecision) /
                                                   e.compIoUOccPrecision
                                             : 0.0f;
                        ImGui::Text("size_ratio:  %.3fx", sr);
                        ImGui::Text("overshoot:   %.1f%%", 100.0f * of);
                        ImGui::Text("Recall:      %.4f", e.compIoUOccRecall);
                        ImGui::Text("Precision:   %.4f", e.compIoUOccPrecision);
                        ImGui::Text("Containment: %s", tag);
                    } else {
                        ImGui::Text("Containment: (not measured)");
                    }
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "--- Base Registration ---");
                    ImGui::Text("Fitness (ICP): %.6f", e.baseFitness);
                    ImGui::Text("ICP RMSE:      %.6f", e.baseIcpRmse);
                    ImGui::Text("Corr. RMSE:    %.6f", e.baseRmse);
                    ImGui::Text("Corr. AvgErr:  %.6f", e.baseAvgError);
                    ImGui::Text("Corr. MaxErr:  %.6f", e.baseMaxError);
                    ImGui::Text("Scale:         %.4f", e.baseScale);
                    if (e.silhouetteCount > 0) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.85f,0.40f,0.95f,1), "--- Silhouette ---");
                        ImGui::Text("Count:        %d",   e.silhouetteCount);
                        ImGui::Text("IoU_full:     %.4f", e.compIoU2D);
                        if (e.compIoU2D_occluded > 0.0f)
                            ImGui::Text("IoU_occluded: %.4f", e.compIoU2D_occluded);   // [Phase B]
                    }
                    ImGui::EndTooltip();
                }
            }
            ImGui::NextColumn();

            if (e.bipopCount > 0)
                ImGui::TextColored(ImVec4(0.94f,0.56f,0.19f,0.9f), "x%d", e.bipopCount);
            else
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "---");
            ImGui::NextColumn();

            if (e.silhouetteCount > 0)
                ImGui::TextColored(ImVec4(0.85f,0.40f,0.95f,0.9f), "x%d", e.silhouetteCount);
            else
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "---");
            ImGui::NextColumn();

            ImGui::Text("%.4f", e.compRmse); ImGui::NextColumn();

            if (e.compIoU2D > 0.0f) {
                /* IoU 高いほど緑寄り、低いほどグレー寄り */
                float t = std::min(1.0f, std::max(0.0f, (e.compIoU2D - 0.7f) / 0.3f));
                ImVec4 ic = ImVec4(0.55f + (0.0f  - 0.55f) * t,
                                   0.55f + (0.85f - 0.55f) * t,
                                   0.55f + (0.30f - 0.55f) * t, 1.0f);
                ImGui::TextColored(ic, "%.3f", e.compIoU2D);
            } else {
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "—");
            }
            ImGui::NextColumn();

            // [Phase B] IoU_occluded cell (Phase A consistent with Layers 1-3).
            // Renders in a slightly more saturated cyan-green ramp so the two
            // IoU columns are visually distinguishable at a glance. Threshold
            // 0.75 (vs 0.70 for IoU_full) reflects that IoU_occluded values
            // typically run higher when instruments occlude the liver.
            if (e.compIoU2D_occluded > 0.0f) {
                float t = std::min(1.0f, std::max(0.0f, (e.compIoU2D_occluded - 0.75f) / 0.25f));
                ImVec4 ic = ImVec4(0.50f + (0.10f - 0.50f) * t,
                                   0.55f + (0.90f - 0.55f) * t,
                                   0.60f + (0.70f - 0.60f) * t, 1.0f);
                ImGui::TextColored(ic, "%.3f", e.compIoU2D_occluded);
            } else {
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "—");
            }
            ImGui::NextColumn();

            // V3R rim diagnostic cell: "<rmse> (<match%>)" when measured.
            // Match% = rimMatched / rimTgtTotal — fraction of the target
            // rim pool that has a source-rim NN within the gate. Drops
            // when the rim band drifts; useful as an overfitting indicator.
            // Color: high match% + low rmse -> orange (= rim well aligned);
            // low match% -> dim grey (= rim drifted / not measured).
            if (e.compRmseRim >= 0.0f && e.rimTgtTotal > 0) {
                const float pct = 100.0f * (float)e.rimMatched
                                  / (float)e.rimTgtTotal;
                // Color ramp on match%: 0% grey -> 40%+ orange (clamp).
                float t = std::min(1.0f, std::max(0.0f, pct / 40.0f));
                ImVec4 rc = ImVec4(0.45f + (0.94f - 0.45f) * t,
                                   0.45f + (0.56f - 0.45f) * t,
                                   0.45f + (0.19f - 0.45f) * t, 1.0f);
                ImGui::TextColored(rc, "%.3f (%.0f%%)",
                                   e.compRmseRim, pct);
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(ImVec4(1,1,0.5f,1),
                                       "=== RIM-only diagnostic ===");
                    ImGui::Text("RIM RMSE:    %.6f m", e.compRmseRim);
                    ImGui::Text("Matched:     %d", e.rimMatched);
                    ImGui::Text("Target rim:  %d  (match rate = %.2f%%)",
                                e.rimTgtTotal, pct);
                    ImGui::Text("Source rim:  %d", e.rimSrcTotal);
                    ImGui::Separator();
                    ImGui::TextWrapped(
                        "Rim-only RMSE is a display-only diagnostic. "
                        "It measures the alignment of the anatomical rim "
                        "band (LiverRegionLabel::RIM) against the target "
                        "boundary band (boundaryDist < threshold). "
                        "Acceptance is unchanged; this is only for judging "
                        "convergence (Ctrl+G, Ctrl+Shift+G).");
                    ImGui::EndTooltip();
                }
            } else {
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "—");
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("RIM-only diagnostic not measured.");
                    ImGui::Text("Populated by Ctrl+G and Ctrl+Shift+G entries.");
                    ImGui::EndTooltip();
                }
            }
            ImGui::NextColumn();

            // [NEW V3RS-CONTAIN] Containment cell: shows "r=X.XX/p=X.XX"
            // (recall / precision) with a 3-state color ramp on the
            // recall - precision delta:
            //   |dir| < 0.05 → green   (balanced; close to ideal)
            //   dir > +0.05  → orange  (overshoot: src > tgt — the case
            //                  the operator most wanted a held-out
            //                  signal for, since IoU alone hides it)
            //   dir < -0.05  → blue    (undershoot: src < tgt)
            // Tooltip explains the metric and shows raw inter/src/tgt
            // cell counts. -1 sentinels → "—" with a help tooltip.
            if (e.compIoUOccPrecision >= 0.0f &&
                e.compIoUOccRecall    >= 0.0f) {
                const float dir = e.compIoUOccRecall - e.compIoUOccPrecision;
                ImVec4 cc;
                const char* glyph;
                if (std::fabs(dir) < 0.05f) {
                    cc = ImVec4(0.45f, 0.80f, 0.50f, 1.0f);   // balanced
                    glyph = "=";
                } else if (dir > 0.0f) {
                    cc = ImVec4(0.90f, 0.55f, 0.35f, 1.0f);   // overshoot
                    glyph = "↑";    // src bigger than tgt
                } else {
                    cc = ImVec4(0.45f, 0.65f, 0.95f, 1.0f);   // undershoot
                    glyph = "↓";
                }
                // [NEW V3RS-CONTAIN-RATIO] size_ratio = |src|/|tgt|
                //   = recall / precision (with 0-guard).
                // overshoot_fraction = |src - inter|/|tgt|
                //   = recall * (1 - precision) / precision.
                // recall is preserved as the coverage axis; size_ratio
                // is the new magnitude axis that recall (saturated at 1)
                // cannot express. The pair (recall, size_ratio) is
                // strictly equivalent in information content to
                // (recall, precision) but reads off as
                // "coverage + how-much-overshoot" instead of two ratios
                // whose magnitudes are coupled through inter.
                const float size_ratio = (e.compIoUOccPrecision > 1e-6f)
                                             ? e.compIoUOccRecall / e.compIoUOccPrecision
                                             : 0.0f;
                const float overshoot_frac =
                    (e.compIoUOccPrecision > 1e-6f)
                        ? e.compIoUOccRecall *
                              (1.0f - e.compIoUOccPrecision) /
                              e.compIoUOccPrecision
                        : 0.0f;
                ImGui::TextColored(cc, "%s r=%.2f x%.2f",
                                   glyph,
                                   e.compIoUOccRecall,
                                   size_ratio);
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(ImVec4(1,1,0.5f,1),
                                       "=== Containment direction ===");
                    // [NEW V3RS-CONTAIN-RATIO] Lead with the headline
                    // metrics (size_ratio + overshoot_fraction); keep
                    // recall/precision below as the breakdown.
                    ImGui::Text("size_ratio = |src|/|tgt|      = %.3fx",
                                size_ratio);
                    ImGui::Text("overshoot  = |src-inter|/|tgt|= %.1f%%",
                                100.0f * overshoot_frac);
                    ImGui::Separator();
                    ImGui::Text("Recall    = |src∩tgt|/|tgt| = %.4f",
                                e.compIoUOccRecall);
                    ImGui::Text("Precision = |src∩tgt|/|src| = %.4f",
                                e.compIoUOccPrecision);
                    ImGui::Text("Δ (recall - prec)            = %+.4f",
                                dir);
                    ImGui::Separator();
                    if (std::fabs(dir) < 0.05f) {
                        ImGui::TextColored(cc,
                                           "Balanced (|Δ| < 0.05).");
                        ImGui::TextWrapped(
                            "Source and target shapes are similarly sized; "
                            "any residual IoU loss is from positional / "
                            "rotational error, not from a size mismatch.");
                    } else if (dir > 0.0f) {
                        ImGui::TextColored(cc,
                                           "OVERSHOOT (size_ratio > 1.0)");
                        ImGui::TextWrapped(
                            "Source silhouette is LARGER than target — "
                            "source covers the target well (high recall) "
                            "but spills outside it. size_ratio reads off "
                            "how much: e.g. 1.44x means source projects "
                            "44% bigger than target. This is the failure "
                            "mode the IoU scalar (and recall alone) "
                            "cannot identify.");
                    } else {
                        ImGui::TextColored(cc,
                                           "UNDERSHOOT (size_ratio < 1.0)");
                        ImGui::TextWrapped(
                            "Source silhouette is SMALLER than target — "
                            "what source covers is mostly correct (high "
                            "precision) but it doesn't reach all of the "
                            "target (low recall). Typical of an undersized "
                            "fit or a scale-too-small solution.");
                    }
                    ImGui::Separator();
                    ImGui::TextDisabled(
                        "Held-out diagnostic. Display only — never gates "
                        "Pose Library acceptance.");
                    ImGui::EndTooltip();
                }
            } else {
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "—");
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Containment direction not measured.");
                    ImGui::Text("Populated by Ctrl+G and Ctrl+Shift+G "
                                "entries with F.5b IoU_occ rasterize.");
                    ImGui::EndTooltip();
                }
            }
            ImGui::NextColumn();

            ImGui::Text("%d", e.compCount); ImGui::NextColumn();

            // Time カラム (元コード line 2787-)
            {
                char tbuf[16];
                if (e.elapsedSec < 100.0f)
                    std::snprintf(tbuf, sizeof(tbuf), "%.1fs", e.elapsedSec);
                else
                    std::snprintf(tbuf, sizeof(tbuf), "%ds", (int)e.elapsedSec);
                ImGui::TextColored(ImVec4(0.6f,0.6f,0.6f,1), "%s", tbuf);
            }
            ImGui::NextColumn();

            ImGui::PushID(e.id);
            if (ImGui::SmallButton("Apply")) applyId = e.id;
            ImGui::PopID();
            ImGui::NextColumn();

            if (isActive) ImGui::PopStyleColor();
        }

        ImGui::Columns(1);

        if (applyId >= 0) poseApplyEntry(applyId);

        ImGui::Separator();
        ImGui::Spacing();

        float bw = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;
        bool canUndo = g_poseLibrary.hasLastRegistration;
        if (!canUndo) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.4f);
        if (ImGui::Button("Undo", ImVec2(bw, 28))) { if (canUndo) poseUndo(); }
        if (!canUndo) ImGui::PopStyleVar();
        ImGui::SameLine();
        if (ImGui::Button("Clear All", ImVec2(bw, 28))) {
            g_poseLibrary.entries.clear();
            g_poseLibrary.activeEntryId = -1;
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(3);
}
