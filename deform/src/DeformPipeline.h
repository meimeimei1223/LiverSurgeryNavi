// DeformPipeline.h
// 変形パイプラインの初期化／更新／描画／マウス操作を関数化したヘッダ。
// すべて inline 関数で .cpp 不要。
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "DeformGlobals.h"
#include "PathConfig.h"
#include "ShaderProgram.h"
#include "MeshDrawing.h"
#include "MeshDrawingSoftBody.h"   // [deform-sep Step 2] SoftBody draw helper
#include "Sphere.h"
#include "SoftBody.h"
#include "TetoMeshData.h"
#include "CentVoxTetrahedralizerHybrid.h"
#include "mCutMesh.h"
#include "ScreenMeshPoints.h"   // ScreenMeshPointCache (shared GL_POINTS draw)
#include "MeshCleanup.h"             // Reg3DCustom::cleanupOBJMesh
#include "FullSphereCameraWithTarget.h"

// AutoDeform 関連 (段階1〜)
#include "OBJTargetExtraction.h"     // setupOBJTarget, mirrorMeshAndCloudX, scaleMeshAndCloud
#include "NoOpen3DRegistration.h"    // setCachedTargetCloud, PointCloud

// ============================================================================
// reset() の本体(SoftBody 必須なのでここで定義)
// ============================================================================
inline void DeformHandlPlaceData::reset() {
    softbodyPoints.clear();
    state = RIGID_MODE;
    if (multiBody) multiBody->fullReset();
    std::cout << "[Deform] reset (state=RIGID_MODE)" << std::endl;
}

namespace DeformPipeline {

// ============================================================================
// Target cloud をグローバルキャッシュに注入
//   Path A: setupOBJTarget (segmentation_mask.png 必須、boundaryDist 付き、原コード忠実)
//   Path B: gTargetMesh の頂点から自前で PointCloud を作成 (mask 不要、silhouette判定なし)
//
// 注入後は extractFrontFacePoints が cache を返すため、AutoDeform の grid 引数は
// 全て無視される(L260 の screen depth bounds 用ブロックを除く)。
//
// 呼び出しは loadReferenceMeshes 内、gTargetMesh の cleanup 後・transform 前。
//   理由: setupOBJTarget は K (image-space) で頂点を再投影してマスク参照するので、
//        mirror+scale 前のメトリック空間で実行する必要がある。
// ============================================================================
inline std::shared_ptr<Reg3DCustom::PointCloud> setupTargetCloud_PathA(
    mCutMesh& tgt, const Reg3DCustom::CameraIntrinsics& K)
{
    auto cloud = Reg3DCustom::setupOBJTarget(
        tgt, K, Reg3DCustom::OBJ_Y_SIGN_OPENGL);
    if (cloud && !cloud->empty()) {
        std::cout << "[Deform/setupTargetCloud] Path A OK ("
                  << cloud->size() << " pts, boundaryDist="
                  << (cloud->hasBoundaryDist() ? "YES" : "NO") << ")" << std::endl;
        return cloud;
    }
    std::cout << "[Deform/setupTargetCloud] Path A failed (segmentation_mask.png missing?)"
              << std::endl;
    return nullptr;
}

inline void setupTargetCloud_PathB(mCutMesh& tgt) {
    // gTargetMesh の頂点をそのまま cloud にする。mask フィルタ無し。
    auto fb = std::make_shared<Reg3DCustom::PointCloud>();
    const auto& V = tgt.mVertices;
    const auto& N = tgt.mNormals;
    bool hasN = (N.size() == V.size());
    fb->points.reserve(V.size() / 3);
    if (hasN) fb->normals.reserve(V.size() / 3);
    for (size_t i = 0; i + 2 < V.size(); i += 3) {
        fb->points.emplace_back(V[i], V[i + 1], V[i + 2]);
        if (hasN) fb->normals.emplace_back(N[i], N[i + 1], N[i + 2]);
    }
    Reg3DCustom::setCachedTargetCloud(fb);
    std::cout << "[Deform/setupTargetCloud] Path B OK ("
              << fb->size() << " pts, boundaryDist=NO)" << std::endl;
}

// ============================================================================
// ⚠️ TEMPORARY ADAPTER — 段階7 で除去予定 ⚠️
// ============================================================================
// AutoDeform.h の grid 引数(grid mesh 時代の名残)を OBJ 用に埋めるアダプタ。
//
// 既知の意味論的問題:
//   classifySrcVisibility L260-310 は「mesh.mVertices の先頭 N 頂点」を screen
//   depth bounds 用に直接アクセスする設計。grid mesh では [0..N)=前面で意味が
//   通るが、OBJ では頂点順序保証なし。frontN=sN クランプで全頂点使用となり
//   「結果的に」正しく動作するが、設計的には壊れている。
//
// 段階7 で AutoDeform.h に no-grid overload を追加してこのアダプタを除去する。
// ============================================================================
inline std::pair<int, int> noGridFor(const mCutMesh* m) {
    if (!m || m->mVertices.empty()) return {1, 1};
    int sN   = static_cast<int>(m->mVertices.size() / 3);
    int side = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(sN)))) + 1;
    return {side, side};
}

// === No-grid ファサード(呼出側から grid/depthScale を完全に隠蔽) ===
//   全て cache 注入済みであることを前提とする(setupTargetCloud で実施)

inline void autoClassifyVisibility(
    mCutMesh* liver, mCutMesh* tgt,
    const glm::vec3& camPos, const glm::vec3& camTgt)
{
    auto [gW, gH] = noGridFor(tgt);
    AutoDeform::classifySrcVisibility(
        gAutoDeform, liver, tgt, gW, gH, camPos, camTgt, /*depthScale=*/1.0f);
}

inline void autoExtractCorrespondences(
    mCutMesh* liver, mCutMesh* tgt, float maxDist = 1.0f)
{
    auto [gW, gH] = noGridFor(tgt);
    AutoDeform::extractCorrespondences(
        gAutoDeform, liver, tgt, gW, gH, /*depthScale=*/1.0f, maxDist);
}

inline float autoMeasureRMSE(
    const std::vector<float>& visVerts, mCutMesh* tgt,
    bool storeAsBefore = false, float maxDist = 1.0f, bool verbose = true)
{
    auto [gW, gH] = noGridFor(tgt);
    return AutoDeform::measureRMSE(
        gAutoDeform, visVerts, tgt, gW, gH, /*depthScale=*/1.0f,
        storeAsBefore, maxDist, verbose);
}

// ============================================================================
// Target / Board / 静的 Liver メッシュをロード
//
// Target 処理順序(原 setupObjScene と互換):
//   1) loadMeshFromFile (raw OBJ in meters)
//   2) cleanupOBJMesh   (silhouette/縮退三角形除去)
//   3) setupTargetCloud (Path A → B fallback) ← cache 注入はここ、mirror前!
//   4) mirrorMeshAndCloudX  (Path A: mesh+cloud 同期, X 軸反転のみ)
//      または applyMirrorOnly (Path B: mesh のみ, X 軸反転)
//   5) setUp (GL バッファ更新)
//
// Board は AutoDeform に関与しないので mirror のみ。
// gLiverStaticMesh は reg_liver.obj から1回ロード(以降不変, registration
// 側で pre-align + post-registration を適用済みなので transform 不要)。
//
// ★ scale=10 撤廃 ★
//   registration 側が g_objDisplayScale=1 (= scale 撤廃)を採用したため、
//   deform 側もそれに合わせて kDisplayScale を撤廃。target/board は raw
//   meters のまま mirror だけ適用する。reg_*.obj は registration の最終
//   状態(target スケール ≒ 0.7m 程度)で保存されているのでそのまま読む。
//
//   旧コードを残したい場合は kDisplayScale を 10.0f に戻して
//   applyMirrorAndScale 経路を復活させればよい(下にコメントで残してある)。
// ============================================================================

// [obj-migration p4] Resolve a depth OBJ path: prefer the canonical (tag-less)
// name REG now writes; fall back to the legacy _k4a-tagged name for pre-migration
// data. Returns "" if neither exists.
inline std::string resolveDeformObjPath(const std::string& canonical,
                                        const std::string& legacyK4a) {
    namespace fs = std::filesystem;
    if (fs::exists(DEPTH_OUTPUT_PATH + canonical)) return DEPTH_OUTPUT_PATH + canonical;
    if (fs::exists(DEPTH_OUTPUT_PATH + legacyK4a)) {
        std::cout << "[Deform] (warn) canonical " << canonical
                  << " not found; using legacy " << legacyK4a
                  << " (re-run Run Depth in REG to update)" << std::endl;
        return DEPTH_OUTPUT_PATH + legacyK4a;
    }
    return std::string();
}

// [obj-migration p4] Load the camera K that produced the depth OBJ. Canonical
// intrinsics.txt (REG-written) first, legacy intrinsics_k4a.txt next, then the
// K4A 720p hardcode as last resort. Shared by loadReferenceMeshes() and --dry-run.
inline Reg3DCustom::CameraIntrinsics loadDeformIntrinsics() {
    Reg3DCustom::CameraIntrinsics K;
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics.txt", K)
        && K.valid()) {
        std::cout << "[Deform] intrinsics.txt: " << K.width << "x" << K.height
                  << "  fx=" << K.fx << " fy=" << K.fy
                  << " cx=" << K.cx << " cy=" << K.cy
                  << " (name=" << (K.name.empty() ? "?" : K.name) << ")" << std::endl;
        return K;
    }
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt", K)
        && K.valid()) {
        std::cout << "[Deform] (warn) intrinsics.txt not found; using legacy "
                     "intrinsics_k4a.txt (re-run Run Depth in REG to update)" << std::endl;
        return K;
    }
    K = Reg3DCustom::CameraIntrinsics::k4a_color_720p();
    std::cout << "[Deform] intrinsics.txt & intrinsics_k4a.txt missing/invalid "
                 "-> K4A 720p hardcode fallback" << std::endl;
    return K;
}

// [intrinsics-step-1] Project board vertices to texture coords using the source
// camera K (the exact pre-transform pinhole projection). Fills board.mTexCoords
// and returns the resulting UV bounds {uMin, uMax, vMin, vMax} (for diagnostics).
inline glm::vec4 computeBoardUV(mCutMesh& board, const Reg3DCustom::CameraIntrinsics& K) {
    const float fx = K.fx, fy = K.fy;
    const float cx = K.cx, cy = K.cy;
    const int   W  = K.width, H = K.height;
    const auto& V = board.mVertices;
    int nVerts = (int)(V.size() / 3);
    board.mTexCoords.resize(nVerts * 2);
    float uMin = 1e30f, uMax = -1e30f, vMin = 1e30f, vMax = -1e30f;
    for (int i = 0; i < nVerts; i++) {
        float x = V[i*3], y = V[i*3+1], z = V[i*3+2];
        if (std::abs(z) < 1e-6f) z = 1e-6f;
        float u = (fx * x / z + cx) / (float)W;
        float v = (fy * y / z + cy) / (float)H;
        board.mTexCoords[i*2]     = u;
        board.mTexCoords[i*2 + 1] = v;
        uMin = std::min(uMin, u); uMax = std::max(uMax, u);
        vMin = std::min(vMin, v); vMax = std::max(vMax, v);
    }
    return glm::vec4(uMin, uMax, vMin, vMax);
}

// [intrinsics-step-1] CPU-only verification of the Step-1 path (no GL, no
// reg_*.obj needed): loads K, loads the board OBJ, computes UV, logs results.
// Driven by lsn_deform --dry-run.
inline void dryRunStep1() {
    std::cout << "[DryRun] === Step 1 intrinsics check (no GL) ===" << std::endl;
    Reg3DCustom::CameraIntrinsics K = loadDeformIntrinsics();
    std::cout << "[DryRun] deformK fx=" << K.fx << " fy=" << K.fy
              << " cx=" << K.cx << " cy=" << K.cy
              << " width=" << K.width << " height=" << K.height << std::endl;

    const std::string p = resolveDeformObjPath("pc_metric_pinhole_full_light.obj",
                                               "pc_metric_pinhole_full_k4a_light.obj");
    if (p.empty()) {
        std::cout << "[DryRun] board OBJ not found (canonical nor legacy)"
                  << " (UV range check skipped)" << std::endl;
        return;
    }
    mCutMesh board = mCutMesh().loadMeshFromFile(p.c_str());
    glm::vec4 uv = computeBoardUV(board, K);
    std::cout << "[DryRun] board verts=" << board.mVertices.size() / 3
              << " UV u=[" << uv.x << "," << uv.y << "]"
              << " v=[" << uv.z << "," << uv.w << "]" << std::endl;
}

// ============================================================================
inline bool loadReferenceMeshes() {
    // mirror-only 版(registration 側 mirrorMeshAndCloudX と等価, mesh 専用):
    //   1) vertex X 反転  2) 法線 X 反転  3) winding flip
    // ※ 法線X反転を忘れるとライティングが裏返って真っ黒に見える
    auto applyMirrorOnly = [](mCutMesh& m) {
        for (size_t i = 0; i < m.mVertices.size(); i += 3)
            m.mVertices[i] = -m.mVertices[i];                 // ① mirror X
        for (size_t i = 0; i < m.mNormals.size(); i += 3)
            m.mNormals[i] = -m.mNormals[i];                   // ② normal X 反転
        for (size_t i = 0; i + 2 < m.mIndices.size(); i += 3)
            std::swap(m.mIndices[i + 1], m.mIndices[i + 2]);  // ③ winding flip
    };

    // ---- gLiverStaticMesh (= 元の長いmain.cpp の liverMesh3D) ----
    // Reg_TARGET_FILE_PATH = registration_model/reg_liver.obj
    // SoftBody の vis(0) と同じファイルだが、AutoDeform 用に独立に保持する。
    // DEFORM_MODE 中は更新せず、起動時の状態のまま。
    {
        if (gLiverStaticMesh) { gLiverStaticMesh->cleanup(); delete gLiverStaticMesh; }
        if (std::filesystem::exists(Reg_TARGET_FILE_PATH)) {
            gLiverStaticMesh = new mCutMesh(
                mCutMesh().loadMeshFromFile(Reg_TARGET_FILE_PATH.c_str()));
            setUp(*gLiverStaticMesh);
            std::cout << "[Deform] LiverStatic loaded: " << Reg_TARGET_FILE_PATH
                      << " (verts=" << gLiverStaticMesh->mVertices.size() / 3
                      << " tris=" << gLiverStaticMesh->mIndices.size() / 3 << ")" << std::endl;
        } else {
            std::cerr << "[Deform] (warn) reg_liver.obj not found at "
                      << Reg_TARGET_FILE_PATH
                      << " — AutoDeform will not work" << std::endl;
        }
    }

    // [intrinsics-step-1] Load the camera K that produced the depth OBJ
    // (matches the _k4a-tagged mesh loaded below) instead of hardcoding K4A
    // 720p, so 1080p / other resolutions get a correct target cloud + board UV.
    // Shared helper (also used by the --dry-run check).
    Reg3DCustom::CameraIntrinsics deformK = loadDeformIntrinsics();

    // ---- Target (= screenMesh) + cache 注入 ----
    {
        const std::string p = resolveDeformObjPath("pc_metric_pinhole_masked.obj",
                                                    "pc_metric_pinhole_masked_k4a.obj");
        if (!p.empty()) {
            if (gTargetMesh) { gTargetMesh->cleanup(); delete gTargetMesh; }
            gTargetMesh = new mCutMesh(mCutMesh().loadMeshFromFile(p.c_str()));
            gTargetMesh->mColor = glm::vec3(0.3f, 0.6f, 0.9f);

            // 髭状エッジ除去(setupObjScene と同じ処理)
            Reg3DCustom::cleanupOBJMesh(*gTargetMesh, /*cosThresh=*/0.10f, 0.0f);

            // === ここで cache 注入(mirror+scale 前!) ===
            // [intrinsics-step-1] use the loaded deformK instead of hardcoded K4A 720p
            auto cloud = setupTargetCloud_PathA(*gTargetMesh, deformK);
            if (cloud) {
                // Path A 成功: mesh と cloud を同期して mirror のみ
                // (registration 側の運用と完全一致, scale=10 は撤廃)
                Reg3DCustom::mirrorMeshAndCloudX(*gTargetMesh, *cloud);
            } else {
                // Path B フォールバック: mesh のみ mirror、その後自前で cloud 生成
                applyMirrorOnly(*gTargetMesh);
                setupTargetCloud_PathB(*gTargetMesh);
            }
            setUp(*gTargetMesh);

            std::cout << "[Deform] Target loaded: " << p
                      << " (verts=" << gTargetMesh->mVertices.size() / 3
                      << " tris=" << gTargetMesh->mIndices.size() / 3 << ")" << std::endl;
        } else {
            std::cout << "[Deform] (skip) Target not found: " << p << std::endl;
        }
    }

    // ---- Board (= boardMesh3D, テクスチャ付き) ----
    {
        const std::string objPath = resolveDeformObjPath("pc_metric_pinhole_full_light.obj",
                                                          "pc_metric_pinhole_full_k4a_light.obj");
        const std::string texPath = DEPTH_OUTPUT_PATH + "texture.png";
        if (!objPath.empty()) {
            if (gBoardMesh) { gBoardMesh->cleanup(); delete gBoardMesh; }
            gBoardMesh = new mCutMesh(mCutMesh().loadMeshFromFile(objPath.c_str()));
            gBoardMesh->mColor = glm::vec3(1.0f, 1.0f, 1.0f);

            // [intrinsics-step-1] UV 生成(変換前の座標で投影)。OBJ 生成時の K
            // (deformK) を使用 — 旧 K4A 720p ハードコード撤廃。共有ヘルパー。
            computeBoardUV(*gBoardMesh, deformK);

            applyMirrorOnly(*gBoardMesh);

            if (std::filesystem::exists(texPath)) {
                gBoardMesh->loadTextureFromFile(texPath);
                std::cout << "[Deform] Board loaded with texture: " << texPath << std::endl;
            } else {
                std::cout << "[Deform] Board loaded (no texture)" << std::endl;
            }
            setUp(*gBoardMesh);
        } else {
            std::cout << "[Deform] (skip) Board not found: " << objPath << std::endl;
        }
    }

    // ============================================================================
    // ★ シーンスケール計測 ★
    //   全メッシュロード後、gTargetMesh の AABB 対角線で gSceneDiag を確定。
    //   以降のカメラ感度・距離・scale 依存しきい値は全てこの値に比例させる。
    //   registration 側の同名処理と完全一致(将来共通化想定)。
    //
    //   gTargetMesh が無い場合(target ファイル欠損時)は gLiverStaticMesh で
    //   フォールバック計測する(registration が pre-align 後に保存している
    //   ので組織サイズは target に近い)。
    // ============================================================================
    {
        float diag = 0.0f;
        const char* src = "none";
        if (gTargetMesh && !gTargetMesh->mVertices.empty()) {
            diag = Reg3DCustom::computeMeshDiag(*gTargetMesh);
            src  = "target";
        } else if (gLiverStaticMesh && !gLiverStaticMesh->mVertices.empty()) {
            diag = Reg3DCustom::computeMeshDiag(*gLiverStaticMesh);
            src  = "liver(fallback)";
        }
        if (diag > 1e-6f) {
            gSceneDiag = diag;
            std::cout << "[Deform] sceneDiag = " << gSceneDiag
                      << "  (src=" << src
                      << ", kRefSceneDiag=" << kRefSceneDiag
                      << ", ratio=" << (gSceneDiag / kRefSceneDiag)
                      << ")" << std::endl;
        } else {
            std::cerr << "[Deform] (warn) sceneDiag could not be measured; "
                         "using kRefSceneDiag as fallback" << std::endl;
            gSceneDiag = kRefSceneDiag;   // 安全側: 旧来の挙動相当
        }
    }

    // ============================================================================
    // ★ 位置ズレ診断ログ ★ (registration が target と reg_liver を同じ
    //   座標フレームで保存できているかを実測で切り分けるためのもの)
    //
    //   target (mirror済み) と reg_liver (raw load) が同じ世界座標に
    //   いるべき。bbox の center が大きく(>10% sceneDiag)離れていれば
    //   registration 側の保存が壊れている可能性が高い。
    // ============================================================================
    auto bboxOf = [](const mCutMesh& m, glm::vec3& mn, glm::vec3& mx) {
        if (m.mVertices.empty()) { mn = mx = glm::vec3(0); return; }
        mn = glm::vec3(FLT_MAX); mx = glm::vec3(-FLT_MAX);
        for (size_t i = 0; i + 2 < m.mVertices.size(); i += 3) {
            glm::vec3 v(m.mVertices[i], m.mVertices[i+1], m.mVertices[i+2]);
            mn = glm::min(mn, v); mx = glm::max(mx, v);
        }
    };
    {
        glm::vec3 tMn(0), tMx(0), lMn(0), lMx(0);
        bool haveT = (gTargetMesh && !gTargetMesh->mVertices.empty());
        bool haveL = (gLiverStaticMesh && !gLiverStaticMesh->mVertices.empty());
        if (haveT) bboxOf(*gTargetMesh, tMn, tMx);
        if (haveL) bboxOf(*gLiverStaticMesh, lMn, lMx);
        glm::vec3 tC = (tMn + tMx) * 0.5f, tS = tMx - tMn;
        glm::vec3 lC = (lMn + lMx) * 0.5f, lS = lMx - lMn;
        std::cout << "\n[Deform] === alignment diagnostics ===" << std::endl;
        if (haveT) std::cout << "  target  center=(" << tC.x << "," << tC.y << "," << tC.z
                             << ")  size=(" << tS.x << "," << tS.y << "," << tS.z
                             << ")  diag=" << glm::length(tS) << std::endl;
        if (haveL) std::cout << "  liver   center=(" << lC.x << "," << lC.y << "," << lC.z
                             << ")  size=(" << lS.x << "," << lS.y << "," << lS.z
                             << ")  diag=" << glm::length(lS) << std::endl;
        if (haveT && haveL) {
            float dCenter = glm::length(tC - lC);
            float diagRef = glm::length(tS);
            float ratio   = (diagRef > 1e-6f) ? dCenter / diagRef : 0.0f;
            std::cout << "  delta_center = " << dCenter
                      << "   (= " << (ratio * 100.0f) << "% of target diag)" << std::endl;
            if (ratio > 0.10f) {
                std::cout << "  >>> WARNING: target and reg_liver are MISALIGNED.\n"
                             "      Deform side applies NO position transform to either —\n"
                             "      this means registration_model/reg_liver.obj was NOT\n"
                             "      saved in the same frame as the (mirrored) target.\n"
                             "      Check the registration save path: it must write the\n"
                             "      MESH STATE AFTER prealignSourceToTarget() AND AFTER\n"
                             "      ICP/FGR rigid transform, in the (mirrored) target frame.\n"
                          << std::endl;
            } else {
                std::cout << "  alignment OK (centers within 10% of target diag)"
                          << std::endl;
            }
        }
        std::cout << std::endl;
    }
    return true;
}

// ============================================================================
// メッシュ中心ユーティリティ(レジストレーション側 Reg3DCustom::computeMeshCenter 相当)
// ============================================================================
inline glm::vec3 computeMeshCenter(const std::vector<float>& verts) {
    if (verts.empty()) return glm::vec3(0.0f);
    glm::vec3 sum(0.0f);
    size_t n = verts.size() / 3;
    for (size_t i = 0; i < n; i++)
        sum += glm::vec3(verts[i*3], verts[i*3+1], verts[i*3+2]);
    return sum / (float)n;
}
inline glm::vec3 computeMeshCenter(const mCutMesh& m) {
    return computeMeshCenter(m.mVertices);
}

// ============================================================================
// シーンスケールに応じてカメラパラメータを動的調整
// ----------------------------------------------------------------------------
// registration 側の同名関数と完全一致(将来共通ヘッダ化を想定)。
//
// スケール対応:
//   InitialRadius / gRadius / minRadius / maxRadius : 距離なので比例
//   LIGHT_MOUSE_SENSITIVITY (パン)                  : 距離なので比例
//   ZOOM_SENSITIVITY                                : 距離なので比例
//   MOUSE_SENSITIVITY (回転)                        : 角度なので比例しない
//   SCALE_SPEED                                     : 比なので比例しない
//
// 呼び出し順序の制約:
//   loadReferenceMeshes() で gSceneDiag を確定させた後、
//   applyRegistrationCameraPose() で初期 pose を決める前に呼ぶこと。
//   (applyRegistrationCameraPose は cam.InitialRadius を起点に位置を計算する)
// ============================================================================
inline void applySceneScaleToCamera(FullSphereCamera& cam, float sceneDiag) {
    if (sceneDiag <= 0.0f) {
        std::cerr << "[Camera] applySceneScaleToCamera: invalid sceneDiag, skipping"
                  << std::endl;
        return;
    }
    const float r = sceneDiag / kRefSceneDiag;

    // 距離系 — 旧 scale=10 時代のデフォルト値を r で正規化
    cam.InitialRadius           = 11.35f * r;
    cam.gRadius                 = cam.InitialRadius;   // Reset() でも追従
    cam.minRadius               = 2.0f   * r;
    cam.maxRadius               = 80.0f  * r;

    // 移動感度系
    cam.LIGHT_MOUSE_SENSITIVITY = 0.01f  * r;
    cam.ZOOM_SENSITIVITY        = -1.0   * (double)r;

    // ※ MOUSE_SENSITIVITY (回転、ラジアン/pixel) は変更しない
    // ※ SCALE_SPEED (倍率) は変更しない

    std::cout << "[Camera] applySceneScaleToCamera:"
              << "  sceneDiag=" << sceneDiag
              << "  ratio="     << r
              << "  InitialRadius=" << cam.InitialRadius
              << "  minR="      << cam.minRadius
              << "  maxR="      << cam.maxRadius
              << "  panSens="   << cam.LIGHT_MOUSE_SENSITIVITY
              << "  zoomSens="  << cam.ZOOM_SENSITIVITY
              << std::endl;
}

// ============================================================================
// scale 依存しきい値ヘルパ
// ----------------------------------------------------------------------------
// AutoDeform Step 2 (extractCorrespondences) の maxDist は「絶対距離」で
// 与える必要があるが、scale=10 と scale=1 で挙動が変わると困る。
// kRefSceneDiag を基準とした既知値(旧 1.0f)を現在のシーンスケールに
// 比例させて返す。
//
// 旧コード:
//     autoExtractCorrespondences(liver, tgt, /*maxDist=*/1.0f);
// 新コード:
//     autoExtractCorrespondences(liver, tgt, /*maxDist=*/scaledMaxDist());
//
// 1.0f は「scale=10 時代に 7.36 のシーンに対する 0.136 倍 ≒ シーンの 13.6%」
// という意味だったので、これを r 倍した値を返せば挙動が保存される。
// ============================================================================
inline float scaledMaxDist(float refValue = 1.0f) {
    if (gSceneDiag <= 0.0f) return refValue;   // 未初期化なら旧挙動
    return refValue * (gSceneDiag / kRefSceneDiag);
}

// ============================================================================
// カメラを「レジストレーション側 setupObjScene 末尾」と同じ向き・対象にする
// ============================================================================
inline void applyRegistrationCameraPose(FullSphereCamera& cam) {
    cam.rotation = glm::angleAxis(glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    cam.currentTarget = TARGET_TEXTURE;
}

// 毎フレーム呼ぶ:liver / texture の中心をカメラに通知
inline void updateCameraTargets(FullSphereCamera& cam) {
    glm::vec3 liverCenter = bunnyPos;
    if (multiBody && multiBody->getNumVisParticles() > 0) {
        liverCenter = bunnyPos + computeMeshCenter(multiBody->getVisPositions(0));
    }
    glm::vec3 textureCenter = gTargetMesh
        ? computeMeshCenter(*gTargetMesh)
        : bunnyPos;
    cam.updateTargetPositions(liverCenter, textureCenter);
}

// ============================================================================
// 内部ヘルパ: BEFORE/AFTER 検査モードから編集モードへ自動復帰
// ----------------------------------------------------------------------------
// Key 7 (snapshot toggle) は安全に切替するため gInspectMode=true で物理を停止する。
// しかしユーザーが続けて Key 6/Backspace/N で編集再開しようとすると、inspect の
// ままなので物理が動かない。この関数は Key 6/Backspace/N の冒頭で呼び、
// 「inspect 中かつ snapshot あり」状態なら自動で復帰する:
//   1) BEFORE 表示中なら AFTER 状態を復元(編集の積み重ねを保全)
//   2) gInspectMode=false で物理 solver を再開
//
// snapshot は破棄しないので、次に Key 7 を押せば再び inspect に入れる。
// (注:setRigidMode は触らない。元から rigid だったかどうかは保たれる)
// ============================================================================
inline bool resumeEditingFromInspect() {
    if (!multiBody) return false;
    if (!gInspectMode) return false;  // editing 中なら何もしない

    if (gSnapAfterValid && !gShowingAfter) {
        // BEFORE 表示中 → AFTER 状態を復元して編集の積み重ねを失わないようにする
        multiBody->setPositions(gSnapAfterPositions);
        multiBody->setAllVisPositions(gSnapAfterVisPositions);
        multiBody->updateTetMeshes();
        multiBody->updateVisMeshes();
        gShowingAfter = true;
        std::cout << "[resume] restored AFTER state (was showing BEFORE)" << std::endl;
    }
    gInspectMode = false;
    std::cout << "[resume] released inspect mode for editing" << std::endl;
    return true;
}

// ============================================================================
// ハンドル描画(物理影響範囲と視覚を一致させるバージョン)
// ----------------------------------------------------------------------------
// 元の AutoHandleController.h の drawAutoHandles は描画スフィアを 0.5 倍してて
// 「物理 vs 視覚」が乖離していた。本関数は center sphere を物理半径そのまま
// 描画し、target marker と arrow は概念マーカーなので相対的に小さく描く。
//
//   center sphere : moveRadius (= 物理影響半径そのまま)
//   target marker : moveRadius * 0.30f (位置参考点)
//   arrow segs    : moveRadius * 0.10f (矢印の線の太さ相当)
//
//   active / edited / untouched は色のみで区別(球サイズは同一 = 物理に忠実)
// ============================================================================
inline void drawAutoHandlesPhysical(
    const AutoHandleController& ctrl,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float fixRadius,     // 物理影響半径(× 倍率なし)
    float moveRadius)    // 物理影響半径(× 倍率なし)
{
    static const glm::vec3 colFix     (0.20f, 0.50f, 1.00f);   // 青  : fix
    static const glm::vec3 colUnt     (1.00f, 1.00f, 0.10f);   // 黄  : 未編集
    static const glm::vec3 colEdit    (0.10f, 0.85f, 0.30f);   // 緑  : 編集済み
    static const glm::vec3 colActive  (1.00f, 0.55f, 0.00f);   // 橙  : アクティブ
    static const glm::vec3 colTgt     (1.00f, 0.10f, 0.10f);   // 赤  : ターゲット

    // Fix ハンドル: 物理半径そのままで描画
    for (int i = 0; i < ctrl.numFix(); i++) {
        const auto& h = ctrl.fixHandle(i);
        sphere.draw(shader, h.initialCenter, colFix, fixRadius, view, proj, camPos);
    }

    // Move ハンドル: 物理半径そのままで描画(色のみで状態区別)
    for (int i = 0; i < ctrl.numMove(); i++) {
        const auto& h = ctrl.moveHandle(i);
        glm::vec3 curCenter = h.initialCenter + h.dirVec * h.progress;
        glm::vec3 tgtFinal  = h.initialCenter + h.dirVec;
        bool isActive = (i == ctrl.activeMoveIdx());

        glm::vec3 centerCol;
        if (isActive)                centerCol = colActive;     // 橙: 編集中
        else if (h.progress > 1e-5f) centerCol = colEdit;        // 緑: 編集済み
        else                         centerCol = colUnt;         // 黄: 未編集

        // メイン中心球: 物理半径そのまま
        sphere.draw(shader, curCenter, centerCol, moveRadius, view, proj, camPos);

        // ターゲットマーカー: 概念点なので小さく(物理半径の 30%)
        sphere.draw(shader, tgtFinal, colTgt, moveRadius * 0.30f, view, proj, camPos);

        // 矢印(目印用、物理半径の 10%)
        for (int s = 1; s < 6; s++) {
            float t = static_cast<float>(s) / 6.0f;
            glm::vec3 mid = glm::mix(curCenter, tgtFinal, t);
            glm::vec3 col = glm::mix(centerCol, colTgt, t);
            sphere.draw(shader, mid, col, moveRadius * 0.10f, view, proj, camPos);
        }
    }
}

// ============================================================================
// reg_*.obj から四面体化＆SoftBody を作る
// ============================================================================
inline bool initFromRegistered() {
    if (deformInit) return true;

    if (!std::filesystem::exists(Reg_TARGET_FILE_PATH)) {
        std::cerr << "[Deform] reg_liver.obj not found: "
                  << Reg_TARGET_FILE_PATH << std::endl;
        return false;
    }

    // 1) liver を四面体化(20分割HYBRIDモード、スムージング無効=元コード準拠)
    {
        CentVoxTetrahedralizerHybrid tet(
            20, Reg_TARGET_FILE_PATH, OUTPUT_TET_FILE,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID, 1, 1);
        CentVoxTetrahedralizerHybrid::SmoothingSettings ss;
        ss.enabled = false; ss.iterations = 0; ss.smoothFactor = 0.0f;
        ss.preserveVolume = true; ss.rescaleToOriginal = true;
        tet.setSmoothingSettings(ss);
        tet.execute();
    }
    std::cout << "[Deform] tet generated -> " << OUTPUT_TET_FILE << std::endl;

    // 2) tet と各臓器(可視メッシュ)をロード
    SoftBody::MeshData tetmesh = SoftBody::loadTetMesh(OUTPUT_TET_FILE);

    auto loadVis = [](const std::string& p) -> SoftBody::MeshData {
        if (std::filesystem::exists(p)) return TetoMeshData::ReadVetexAndFace(p);
        std::cout << "[Deform] (skip) " << p << std::endl;
        return SoftBody::MeshData{};
    };

    std::vector<SoftBody::MeshData> visMeshes;
    visMeshes.push_back(loadVis(Reg_TARGET_FILE_PATH));   // liver  (必須)
    visMeshes.push_back(loadVis(Reg_PORTAL_FILE_PATH));   // portal
    visMeshes.push_back(loadVis(Reg_VEIN_FILE_PATH));     // vein
    visMeshes.push_back(loadVis(Reg_TUMOR_FILE_PATH));    // tumor
    visMeshes.push_back(loadVis(Reg_SEGMENT_FILE_PATH));  // segment
    visMeshes.push_back(loadVis(Reg_GB_FILE_PATH));       // gb

    // 3) SoftBody 生成
    multiBody = new SoftBody(tetmesh, visMeshes, /*edgeComp=*/0.001f, /*volComp=*/0.0f);
    multiBody->setRigidMode(true);   // 起動直後 RIGID_MODE
    deformInit = true;

    // sphere marker (ハンドル可視化用)
    deformSphereMarker.generate(1.0f, 16, 16);
    deformSphereMarker.setup();

    std::cout << "[Deform] SoftBody created (rigid=on, "
              << visMeshes.size() << " vis meshes)" << std::endl;
    return true;
}

// ============================================================================
// 毎フレーム：物理ステップ + 描画
// ============================================================================
inline void updateAndDraw(float dt,
                          ShaderProgram& shader,
                          ShaderProgram& shaderTex,
                          const glm::mat4& viewMat,
                          const glm::mat4& projMat,
                          const glm::vec3& camPos)
{
    if (!deformInit || !multiBody) return;

    if (gGrabber) gGrabber->update(dt);

    // モデル行列(SoftBody 全体の平行移動)
    glm::mat4 model = glm::translate(glm::mat4(1.0f), bunnyPos);
    multiBody->setModelMatrix(model);

    // ハンドルの可視化(planecut以外)
    if (deformHandlPlace.state != DeformHandlPlaceData::PLANECUT_MODE) {
        for (size_t g = 0; g < multiBody->handleGroups.size(); ++g) {
            glm::vec3 center   = multiBody->handleGroups[g].centerPosition;
            float     radius   = multiBody->handleGroups[g].radius;
            glm::vec3 worldPos = glm::vec3(model * glm::vec4(center, 1.0f));
            glm::vec3 color    = getPointColor((int)g, true);
            deformSphereMarker.draw(shader, worldPos, color, radius,
                                    viewMat, projMat, camPos);
        }
    }

    // 物理ステップ
    shader.use();
    shader.setUniform("model",      model);
    shader.setUniform("view",       viewMat);
    shader.setUniform("projection", projMat);
    shader.setUniform("lightPos",   camPos);
    shader.setUniform("lightColor", glm::vec3(1.0f));
    shader.setUniform("vertColor",  glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

    // ---- 物理 step ----
    //   gInspectMode=true (Key 7 で snapshot 表示中) のときは物理を完全停止。
    //   これにより rigidMode のような restPosition 上書き問題を回避でき、
    //   setPositions(snapshot) で書き込んだ状態がそのまま保たれる。
    if (!gInspectMode) {
        multiBody->preSolve(dt, gravity);
        multiBody->solve(dt);
        multiBody->postSolve(dt);
        multiBody->updateVisMeshes();
    }

    // Pending RMSE 表示(Key 6/Backspace 後の物理 settle 計測)
    //   step 直後: gPendingRmseImmediate / gPendingRmseFrames=60 がセットされる
    //   60 フレーム後: settle した状態で再測定し、step delta / base delta を表示
    if (gPendingRmseFrames > 0 && !gInspectMode) {
        gPendingRmseFrames--;
        if (gPendingRmseFrames == 0 && gTargetMesh) {
            const auto& visV = multiBody->getVisPositions(0);
            float settled = DeformPipeline::autoMeasureRMSE(
                visV, gTargetMesh,
                /*storeAsBefore=*/false, scaledMaxDist(), /*verbose=*/false);
            float base = gAutoDeform.rmseBeforeDeform;
            float stepDelta = settled - gPendingRmseImmediate;
            float baseDelta = (base > 0.0f) ? (settled - base) : 0.0f;
            std::cout << "[settled] RMSE=" << settled
                      << "  step:" << (stepDelta>=0?"+":"") << stepDelta << (stepDelta<0?" DOWN":" UP")
                      << "  base:"  << (baseDelta>=0?"+":"") << baseDelta << (baseDelta<0?" DOWN":" UP")
                      << std::endl;
        }
    }

    // 各 vis メッシュの色(liver/portal/vein/tumor/segment/gb) — gOrganAlpha 一括
    const float oa = gOrganAlpha;
    std::vector<glm::vec4> visColors = {
        glm::vec4(0.8f, 0.2f, 0.2f, oa),
        glm::vec4(0.2f, 0.2f, 0.8f, oa),
        glm::vec4(0.2f, 0.5f, 0.5f, oa),
        glm::vec4(0.8f, 0.5f, 0.5f, oa),
        glm::vec4(0.2f, 0.8f, 0.5f, oa),
        glm::vec4(0.2f, 0.8f, 0.2f, oa),
    };

    // ---- (1) Target / Board を背景として draw_AllmCutMeshes で描画 ----
    {
        std::vector<mCutMesh*> extras;
        std::vector<glm::vec4> extraColors;
        int textureMeshIdx = -1;

        // Target mesh (= screenMesh equivalent) is very heavy; draw it as a
        // point cloud (shared ScreenMeshPointCache) instead of rasterizing its
        // ~1.7M triangles every frame. Always points in DEFORM mode.
        if (gTargetAlpha > 0.01f && gTargetMesh) {
            static ScreenMeshPointCache s_targetPC;
            s_targetPC.draw(gTargetMesh, shader, model, viewMat, projMat, camPos,
                            glm::vec4(0.3f, 0.6f, 0.9f, gTargetAlpha),
                            /*pointSize=*/2.0f, /*densityPct=*/1.0f);
        }
        if (gBoardAlpha > 0.01f && gBoardMesh) {
            extras.push_back(gBoardMesh);
            extraColors.push_back(glm::vec4(1.0f, 1.0f, 1.0f, gBoardAlpha));
            textureMeshIdx = (int)extras.size() - 1;
        }
        if (!extras.empty()) {
            draw_AllmCutMeshes(extras, shader, shaderTex, camPos, extraColors,
                               model, viewMat, projMat, textureMeshIdx);
        }
    }

    // ---- (2) SoftBody (臓器) を手前に draw_AllVisMeshesWithExtraMesh で ----
    if (oa > 0.01f) {
        draw_AllVisMeshesWithExtraMesh(
            multiBody, shader, shaderTex,
            /*extraMesh=*/nullptr, camPos,
            visColors, glm::vec4(0.0f),
            model, viewMat, projMat);
    }

    // ---- (2.5) TetMesh wireframe (Key 0 でトグル) ----
    //   showTetMesh フラグ内蔵なので無条件呼び出しで OK
    multiBody->updateTetMeshes();
    shader.setUniform("model", model);
    multiBody->drawTetMesh(shader);

    // ---- (3) AutoDeform 可視化 (stage に応じて自動分岐) ----
    //   stage=0 (Key 未押): 何も描かれない
    //   stage=1 (Key 1):    src visibility 色分け (visible=緑 / hidden=赤)
    //   stage=2 (Key 2):    対応点 (src↔tgt 線分)
    //   stage=3 (Key 3):    分類カテゴリ色 (INLIER/MOVER/OUTLIER × silhouette/interior)
    //   stage=4 (Key 4):    フィールド点群 (vis mesh 上のスコア)
    //   stage=5 (Key 5):    フィールド点群 + 生成ハンドル位置 (drawHandles)
    AutoDeform::draw(gAutoDeform, deformSphereMarker, shader,
                     viewMat, projMat, camPos);

    // ---- (4) Auto ハンドル(コントローラ状態の可視化) ----
    //   gAutoCtrl が initialize 済みのとき、現在の進捗(active=橙、edited=緑、未編集=黄)
    //   と矢印・ターゲット球を表示。
    //   ★ 描画スフィア = 物理影響半径 × 0.5(scale=10 時代の元コードと同じ見た目)★
    //   drawAutoHandlesPhysical の中で target/arrow は moveRadius 基準で
    //   相対比率(0.30/0.10)が掛かるので、外で 0.5 倍すれば全体が等比で縮む。
    if (gAutoCtrl.numFix() + gAutoCtrl.numMove() > 0) {
        const auto& presetCur = AutoDeform::getPresets()[gAutoDeformPresetIdx];
        float fixRadius  = gAutoDeform.fieldBboxDiag * presetCur.rFixScale  * 0.5f;
        float moveRadius = gAutoDeform.fieldBboxDiag * presetCur.rMoveScale * 0.5f;
        DeformPipeline::drawAutoHandlesPhysical(
            gAutoCtrl, deformSphereMarker, shader,
            viewMat, projMat, camPos,
            fixRadius, moveRadius);
    }
}

// ============================================================================
// マウスイベント
// ============================================================================
inline void onMousePress(int button, double xpos, double ypos)
{
    if (!gGrabber || !multiBody) return;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    using S = DeformHandlPlaceData;
    switch (deformHandlPlace.state) {
    case S::HANDLE_PLACE_MODE: {
        if (multiBody->handleGroups.size() >= SoftBody::MAX_HANDLE_GROUPS) {
            std::cout << "[Deform] handle max reached. press C to clear." << std::endl;
            return;
        }
        gGrabber->placeSphere((float)xpos, (float)ypos, gGroupRadius);
        if (hit_index >= 0)
            deformHandlPlace.softbodyPoints.push_back(hit_position);
        if (deformHandlPlace.softbodyPoints.size() >= SoftBody::MAX_HANDLE_GROUPS) {
            deformHandlPlace.state = S::DEFORM_MODE;
            multiBody->setRigidMode(false);
            std::cout << "[Deform] -> DEFORM_MODE (max handles placed)" << std::endl;
        }
        isDragging = false;
        hit_index  = -1;
    } break;

    case S::RIGID_MODE: {
        if (gGrabber->hitTest((float)xpos, (float)ypos))
            isDragging = true;
    } break;

    case S::DEFORM_MODE: {
        gGrabber->startGrab((float)xpos, (float)ypos);
    } break;

    default: break;
    }
}

inline void onMouseRelease(int button)
{
    if (!gGrabber) return;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    using S = DeformHandlPlaceData;
    if (deformHandlPlace.state == S::DEFORM_MODE) {
        gGrabber->endGrab();
    }
    isDragging = false;
    hit_index  = -1;
}

inline void onMouseMove(double xpos, double ypos, float dt)
{
    if (!gGrabber) return;
    if (deformHandlPlace.state != DeformHandlPlaceData::DEFORM_MODE) return;
    if (!isDragging) return;
    gGrabber->moveGrab((float)xpos, (float)ypos, dt);
}

// ============================================================================
// キーボード(統合 main.cpp 用に独立化)
//   DEFORM_MODE のとき呼ぶ想定。GLFW の key / mods をそのまま渡す。
//   ※ 1〜0 は AutoDeform 用に予約済み(段階2 以降で接続)
// ============================================================================
inline void onKey(int key, int mods)
{
    using S = DeformHandlPlaceData;
    switch (key) {
    case GLFW_KEY_R:
        deformHandlPlace.state = S::RIGID_MODE;
        if (multiBody) multiBody->setRigidMode(true);
        std::cout << "[Mode] RIGID_MODE" << std::endl;
        break;
    case GLFW_KEY_H:
        deformHandlPlace.state = S::HANDLE_PLACE_MODE;
        std::cout << "[Mode] HANDLE_PLACE_MODE (LMB to place, max 5)" << std::endl;
        break;
    case GLFW_KEY_D:
        deformHandlPlace.state = S::DEFORM_MODE;
        if (multiBody) multiBody->setRigidMode(false);
        std::cout << "[Mode] DEFORM_MODE (LMB drag to deform)" << std::endl;
        break;
    case GLFW_KEY_C:
        deformHandlPlace.reset();
        break;
    case GLFW_KEY_V:
        cycleAlpha(gOrganAlpha);
        std::cout << "[Toggle] Organs = " << alphaLabel(gOrganAlpha) << std::endl;
        break;
    case GLFW_KEY_T:
        cycleAlpha(gTargetAlpha);
        std::cout << "[Toggle] Target = " << alphaLabel(gTargetAlpha) << std::endl;
        break;
    case GLFW_KEY_B:
        cycleAlpha(gBoardAlpha);
        std::cout << "[Toggle] Board  = " << alphaLabel(gBoardAlpha) << std::endl;
        break;

    // ========================================================================
    // === AUTO Step 1〜4 (元の長いmain.cpp の Key 1〜4 と同等) ===
    //   1: classifySrcVisibility (src 頂点の visible/hidden 分類)
    //   2: extractCorrespondences (src↔tgt 対応点)
    //   3: classify (INLIER/MOVER/OUTLIER 分類, 必要なら 2 を先に呼ぶ)
    //   4: computeFieldOnVisMesh (変形フィールド,   必要なら 2 を先に呼ぶ)
    // ========================================================================
    case GLFW_KEY_1:
        if (currentMainMode == DEFORM_MODE && gLiverStaticMesh && gTargetMesh) {
            if (mods & GLFW_MOD_SHIFT) {
                // SHIFT+1: 表示モードだけ切替(パイプライン再実行なし)
                gAutoDeform.debugMaxPoints = gAutoDeform.debugMaxPoints ? 0 : 1;
                std::cout << "[debug] single sphere mode: "
                          << (gAutoDeform.debugMaxPoints ? "ON" : "OFF") << std::endl;
                break;
            }
            gAutoDeform.debugMaxPoints = 0;
            // KeyA (AR モード) と同一の固定カメラポーズを使用:
            //   - cameraPos    = 原点 (= original.jpg を撮ったカメラ位置)
            //   - cameraTarget = +Z 方向
            //   AR で見えている面と Step1 visibility 判定を一致させるため、
            //   OrbitCam の現在/初期ポーズではなく AR の固定ポーズで raycast する。
            //   レジストレーション側 main.cpp の KeyA と同一の (origin, +Z) ペア。
            glm::vec3 arCamPos(0.0f, 0.0f, 0.0f);
            glm::vec3 arCamTgt(0.0f, 0.0f, 1.0f);
            std::cout << "[AutoDeform Step1] using AR camera pose (0,0,0) -> (0,0,1)"
                      << std::endl;
            DeformPipeline::autoClassifyVisibility(
                gLiverStaticMesh, gTargetMesh, arCamPos, arCamTgt);
        }
        break;

    case GLFW_KEY_2:
        if (currentMainMode == DEFORM_MODE && gLiverStaticMesh && gTargetMesh) {
            if (mods & GLFW_MOD_SHIFT) {
                gAutoDeform.debugMaxPoints = gAutoDeform.debugMaxPoints ? 0 : 1;
                std::cout << "[debug] single sphere mode: "
                          << (gAutoDeform.debugMaxPoints ? "ON" : "OFF") << std::endl;
                break;
            }
            gAutoDeform.debugMaxPoints = 0;
            DeformPipeline::autoExtractCorrespondences(
                gLiverStaticMesh, gTargetMesh, /*maxDist=*/scaledMaxDist());
        }
        break;

    case GLFW_KEY_3:
        if (currentMainMode == DEFORM_MODE && gLiverStaticMesh && gTargetMesh) {
            if (mods & GLFW_MOD_SHIFT) {
                gAutoDeform.debugMaxPoints = gAutoDeform.debugMaxPoints ? 0 : 1;
                std::cout << "[debug] single sphere mode: "
                          << (gAutoDeform.debugMaxPoints ? "ON" : "OFF") << std::endl;
                break;
            }
            gAutoDeform.debugMaxPoints = 0;
            if (gAutoDeform.correspondences.empty()) {
                DeformPipeline::autoExtractCorrespondences(
                    gLiverStaticMesh, gTargetMesh, scaledMaxDist());
            }
            AutoDeform::classify(gAutoDeform, /*inlierMadScale=*/1.0f, /*outlierMadScale=*/3.0f);
        }
        break;

    case GLFW_KEY_4:
        if (currentMainMode == DEFORM_MODE && multiBody && gLiverStaticMesh && gTargetMesh) {
            if (gAutoDeform.correspondences.empty()) {
                DeformPipeline::autoExtractCorrespondences(
                    gLiverStaticMesh, gTargetMesh, scaledMaxDist());
            }
            const auto& visVerts  = multiBody->getVisPositions(0);
            const auto& visTriIds = multiBody->getVisSurfaceTriIds(0);
            AutoDeform::computeFieldOnVisMesh(
                gAutoDeform, visVerts, visTriIds,
                /*kNearest=*/8, /*gaussianSigma=*/0.0f,
                /*useOnlyRatioOK=*/true,
                /*smoothIterations=*/10, /*smoothAlpha=*/0.5f);
        }
        break;

    // ========================================================================
    // === AUTO Step 5: ハンドル自動生成 + DEFORM_MODE 自動切替 ===
    //   - 必要なら Step 2/4 を先に呼ぶ(extractCorrespondences / computeFieldOnVisMesh)
    //   - generateHandles でプリセットに従って fix/move ハンドル配置
    //   - gAutoCtrl.initialize で SoftBody にハンドル登録
    //   - setRigidMode(false) + state=DEFORM_MODE で物理駆動開始
    // ========================================================================
    case GLFW_KEY_5:
        if (currentMainMode == DEFORM_MODE && multiBody && gLiverStaticMesh && gTargetMesh) {
            // 元コード踏襲: フィールド未計算なら Step 2/4 相当を先に呼ぶ
            if (!gAutoDeform.fieldReady) {
                if (gAutoDeform.correspondences.empty()) {
                    DeformPipeline::autoExtractCorrespondences(
                        gLiverStaticMesh, gTargetMesh, scaledMaxDist());
                }
                const auto& visVerts  = multiBody->getVisPositions(0);
                const auto& visTriIds = multiBody->getVisSurfaceTriIds(0);
                AutoDeform::computeFieldOnVisMesh(
                    gAutoDeform, visVerts, visTriIds,
                    8, 0.0f, true, 10, 0.5f);
            }

            const auto& p = AutoDeform::getPresets()[gAutoDeformPresetIdx];
            std::cout << "[AutoDeform] Using preset: " << p.name << std::endl;
            AutoDeform::generateHandles(
                gAutoDeform,
                p.K_fix, p.K_move,
                p.tauLowScale, p.tauHighScale,
                p.rFixScale, p.rMoveScale,
                p.minSepScale);

            // Debug print: fix / move ハンドルの詳細
            std::cout << "--- [Debug Key5] after generateHandles ---" << std::endl;
            for (size_t i = 0; i < gAutoDeform.fixHandles.size(); i++) {
                const auto& h = gAutoDeform.fixHandles[i];
                std::cout << "  fix[" << i << "]"
                          << " center=(" << h.center.x << "," << h.center.y << "," << h.center.z << ")"
                          << " radius=" << h.radius << std::endl;
            }
            for (size_t i = 0; i < gAutoDeform.moveHandles.size(); i++) {
                const auto& h = gAutoDeform.moveHandles[i];
                glm::vec3 disp = h.target - h.center;
                std::cout << "  move[" << i << "]"
                          << " center=(" << h.center.x << "," << h.center.y << "," << h.center.z << ")"
                          << " target=(" << h.target.x << "," << h.target.y << "," << h.target.z << ")"
                          << " |disp|=" << glm::length(disp)
                          << " radius=" << h.radius << std::endl;
            }

            // SoftBody に登録 → DEFORM_MODE で物理駆動開始
            gAutoCtrl.initialize(multiBody, gAutoDeform);
            gAutoDeformFirstCommit = false;
            multiBody->setRigidMode(false);
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
            std::cout << "[Key5] auto-switched to DEFORM_MODE (rigidMode=false)" << std::endl;
        }
        break;

    // ========================================================================
    // === HEMI 駆動 (Keys 6, A, Backspace, -, =) ===
    //   Key 5 で生成・配置されたハンドルを、人がキー押すたびに 1 ステップずつ駆動
    //   元の長いmain.cpp の Key 6 / Bksp / A / -/= と完全一致(レジストレーションの
    //   HemiAuto とは無関係。あちらは FGR+ICP の自動位置合わせ)
    // ========================================================================

    // Key 6: アクティブな move ハンドルを +gMoveScale 進める
    case GLFW_KEY_6:
        if (currentMainMode == DEFORM_MODE && multiBody && gTargetMesh) {
            if (gAutoDeform.stage < 5 || gAutoCtrl.numMove() == 0) {
                std::cout << "[AutoDeform] Run handle gen (key 5) first" << std::endl;
                break;
            }
            if (gAutoCtrl.activeMoveIdx() < 0) {
                std::cout << "[Key6] no active move (press A to select first)" << std::endl;
                break;
            }

            // Key 7 inspection モードからの自動復帰
            DeformPipeline::resumeEditingFromInspect();

            std::cout << "[Key6 debug] rigid=" << multiBody->isRigidMode()
                      << " att=" << multiBody->getNumAttachments()
                      << " inspect=" << gInspectMode
                      << " state=" << (int)deformHandlPlace.state << std::endl;

            // First commit: BEFORE スナップショット + baseline RMSE 計測
            bool isFirst = !gAutoDeformFirstCommit;
            if (isFirst) {
                gSnapBeforePositions    = multiBody->getPositions();
                gSnapBeforeVisPositions = multiBody->getAllVisPositions();
                gSnapBeforeValid = true;
                gSnapAfterValid  = false;
                gShowingAfter    = true;
                const auto& visVerts0 = multiBody->getVisPositions(0);
                DeformPipeline::autoMeasureRMSE(
                    visVerts0, gTargetMesh,
                    /*storeAsBefore=*/true, scaledMaxDist(), /*verbose=*/false);
                std::cout << "[Key6] baseline RMSE=" << gAutoDeform.rmseBeforeDeform << std::endl;
                gAutoDeformFirstCommit = true;
            }

            int   activeIdx = gAutoCtrl.activeMoveIdx();
            float prevProg  = gAutoCtrl.moveHandle(activeIdx).progress;
            float prevRMSE  = gAutoDeform.rmseLastMeasured;

            bool changed = gAutoCtrl.stepActive(multiBody, +gMoveScale);
            if (!changed) {
                std::cout << "[Key6] slot=" << activeIdx << " already at goal, no-op" << std::endl;
                break;
            }
            gAutoCtrl.runBoost(multiBody, gAutoDeformBoostIter, gAutoDeformBoostDamping);

            float newProg = gAutoCtrl.moveHandle(activeIdx).progress;
            const auto& visVertsForRMSE = multiBody->getVisPositions(0);
            float immRMSE = DeformPipeline::autoMeasureRMSE(
                visVertsForRMSE, gTargetMesh,
                /*storeAsBefore=*/false, scaledMaxDist(), /*verbose=*/false);
            float stepD = (prevRMSE > 0.0f) ? (immRMSE - prevRMSE) : 0.0f;
            float baseD = (gAutoDeform.rmseBeforeDeform > 0.0f) ? (immRMSE - gAutoDeform.rmseBeforeDeform) : 0.0f;
            std::cout << "[6] move=" << activeIdx << " progress " << prevProg << " -> " << newProg
                      << "  RMSE=" << immRMSE
                      << "  step:" << (stepD>=0?"+":"") << stepD << (stepD<0?" DOWN":" UP")
                      << "  base:" << (baseD>=0?"+":"") << baseD << (baseD<0?" DOWN":" UP")
                      << std::endl;

            // settled RMSE 用に 60 フレーム後の遅延計測を予約
            gPendingRmseImmediate = immRMSE;
            gPendingRmseFrames    = 60;
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
        }
        break;

    // Backspace: アクティブ move ハンドルを -gMoveScale 戻す
    case GLFW_KEY_BACKSPACE:
        if (currentMainMode == DEFORM_MODE && multiBody && gTargetMesh) {
            if (gAutoCtrl.activeMoveIdx() < 0) {
                std::cout << "[Bksp] no active move" << std::endl;
                break;
            }
            int activeIdx  = gAutoCtrl.activeMoveIdx();
            float prevProg = gAutoCtrl.moveHandle(activeIdx).progress;
            if (prevProg <= 1e-5f) {
                std::cout << "[Bksp] move=" << activeIdx << " already at origin, no-op" << std::endl;
                break;
            }

            // Key 7 inspection モードからの自動復帰
            DeformPipeline::resumeEditingFromInspect();

            float prevRMSE = gAutoDeform.rmseLastMeasured;
            bool changed = gAutoCtrl.stepActive(multiBody, -gMoveScale);
            if (!changed) {
                std::cout << "[Bksp] no change" << std::endl;
                break;
            }
            gAutoCtrl.runBoost(multiBody, gAutoDeformBoostIter, gAutoDeformBoostDamping);

            float newProg = gAutoCtrl.moveHandle(activeIdx).progress;
            const auto& visVertsForRMSE = multiBody->getVisPositions(0);
            float immRMSE = DeformPipeline::autoMeasureRMSE(
                visVertsForRMSE, gTargetMesh,
                /*storeAsBefore=*/false, scaledMaxDist(), /*verbose=*/false);
            float stepD = (prevRMSE > 0.0f) ? (immRMSE - prevRMSE) : 0.0f;
            float baseD = (gAutoDeform.rmseBeforeDeform > 0.0f) ? (immRMSE - gAutoDeform.rmseBeforeDeform) : 0.0f;
            std::cout << "[Bksp] move=" << activeIdx << " progress " << prevProg << " -> " << newProg
                      << (newProg <= 0.0f ? "  [AT ORIGIN]" : "")
                      << "  RMSE=" << immRMSE
                      << "  step:" << (stepD>=0?"+":"") << stepD << (stepD<0?" DOWN":" UP")
                      << "  base:" << (baseD>=0?"+":"") << baseD << (baseD<0?" DOWN":" UP")
                      << std::endl;

            gPendingRmseImmediate = immRMSE;
            gPendingRmseFrames    = 60;
        }
        break;

    // Key N: 次の move ハンドルへアクティブ切替
    //   旧 Key A から移動（A は main.cpp 側で AR 背景オーバーレイ切替に使うため）。
    //   "N" = Next move handle のニーモニック。
    case GLFW_KEY_N:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            if (gAutoCtrl.numMove() == 0) {
                std::cout << "[KeyN] run handle gen (key 5) first" << std::endl;
                break;
            }

            // Key 7 inspection モードからの自動復帰
            DeformPipeline::resumeEditingFromInspect();

            gAutoCtrl.selectNextMove();
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
        }
        break;

    // Key -: gMoveScale -= 0.1 (clamp 0.1〜2.0)
    case GLFW_KEY_MINUS:
        if (currentMainMode == DEFORM_MODE) {
            gMoveScale = std::max(0.1f, gMoveScale - 0.1f);
            std::cout << "[MoveScale] = " << gMoveScale
                      << "  (handle will pull " << (gMoveScale * 100.0f) << "% of corresp distance)"
                      << std::endl;
        }
        break;

    // Key =: gMoveScale += 0.1
    case GLFW_KEY_EQUAL:
        if (currentMainMode == DEFORM_MODE) {
            gMoveScale = std::min(2.0f, gMoveScale + 0.1f);
            std::cout << "[MoveScale] = " << gMoveScale
                      << "  (handle will pull " << (gMoveScale * 100.0f) << "% of corresp distance)"
                      << std::endl;
        }
        break;

    // ========================================================================
    // === Key 7: BEFORE/AFTER スナップショット toggle ===
    //   Key 6 で baseline (BEFORE) は既にキャプチャ済み
    //
    //   状態遷移:
    //     editing  (rigid=false) → 7 押下: 現在の状態を AFTER として再 capture + AFTER 表示
    //     inspect  (rigid=true)  → 7 押下: BEFORE/AFTER 切替のみ
    //
    //   これにより「編集 → 7 で確認 → 編集再開 → 7 で再確認」のサイクルで、
    //   AFTER が常に最新の編集状態を反映する。
    // ========================================================================
    case GLFW_KEY_7:
        if (currentMainMode == DEFORM_MODE && multiBody && gTargetMesh) {
            if (!gSnapBeforeValid) {
                const auto& visVerts0 = multiBody->getVisPositions(0);
                DeformPipeline::autoMeasureRMSE(
                    visVerts0, gTargetMesh, false, scaledMaxDist(), true);
                std::cout << "[AutoDeform] No BEFORE snapshot "
                             "(press key 6 at least once first)" << std::endl;
                break;
            }

            // editing(gInspectMode=false) からの遷移なら AFTER を再 capture
            bool wasEditing = !gInspectMode;

            if (wasEditing) {
                bool isFirstCapture = !gSnapAfterValid;
                gSnapAfterPositions    = multiBody->getPositions();
                gSnapAfterVisPositions = multiBody->getAllVisPositions();
                gSnapAfterValid = true;

                // ★ UX: 編集→inspect 遷移時は BEFORE を表示する
                //   ユーザーは「現在の編集状態 (= AFTER) は既に見ている」ので、
                //   Key 7 押下の瞬間に BEFORE を見せて差分を即座に見せる方が自然。
                //   (toggle してまた 7 押せば AFTER に戻る)
                gShowingAfter = false;

                // BEFORE と現在の差(編集量)を表示
                float maxD = 0.0f, sumD = 0.0f; int cnt = 0;
                if (gSnapAfterPositions.size() == gSnapBeforePositions.size()) {
                    for (size_t k = 0; k + 2 < gSnapAfterPositions.size(); k += 3) {
                        glm::vec3 a(gSnapAfterPositions[k],  gSnapAfterPositions[k+1],  gSnapAfterPositions[k+2]);
                        glm::vec3 b(gSnapBeforePositions[k], gSnapBeforePositions[k+1], gSnapBeforePositions[k+2]);
                        float d = glm::length(a - b);
                        if (d > maxD) maxD = d;
                        sumD += d; cnt++;
                    }
                }
                std::cout << "[Key7] AFTER " << (isFirstCapture ? "captured" : "RE-captured")
                          << "  maxParticleDisp=" << maxD
                          << "  avgParticleDisp=" << (cnt > 0 ? sumD / cnt : 0.0f)
                          << "  -> showing BEFORE for instant comparison"
                          << std::endl;
            } else {
                // inspect 中(rigid=true): BEFORE/AFTER toggle
                if (!gSnapAfterValid) {
                    std::cout << "[Key7] no AFTER snapshot yet (edit first, then press 7)" << std::endl;
                    break;
                }
                gShowingAfter = !gShowingAfter;
            }

            // 物理を完全停止して安全に snapshot 表示
            //   ※ setRigidMode(true) は使わない。
            //     rigid mode の preSolve 内で updateRigidTransform() が
            //     restPositionsRelative (rest pose) で positions を上書きするため、
            //     snapshot が消えてしまう。代わりに gInspectMode で物理を完全停止する。
            gInspectMode = true;
            if (gShowingAfter) {
                multiBody->setPositions(gSnapAfterPositions);
                multiBody->setAllVisPositions(gSnapAfterVisPositions);
                std::cout << "[Toggle] Showing AFTER (deformed)" << std::endl;
            } else {
                multiBody->setPositions(gSnapBeforePositions);
                multiBody->setAllVisPositions(gSnapBeforeVisPositions);
                std::cout << "[Toggle] Showing BEFORE (original)" << std::endl;
            }
            multiBody->updateTetMeshes();
            multiBody->updateVisMeshes();

            // 切替後の RMSE 計測(verbose=true で表示)
            const auto& visVerts0 = multiBody->getVisPositions(0);
            DeformPipeline::autoMeasureRMSE(
                visVerts0, gTargetMesh,
                /*storeAsBefore=*/false, scaledMaxDist(), /*verbose=*/true);
        }
        break;

    // ========================================================================
    // === Key P: プリセット切替 (P0..P4 サイクル) ===
    //   Key 5 の前に押すと次回 Key 5 で新プリセットでハンドル生成。
    //   既にハンドル生成済みの場合は描画半径のみ更新(物理は次回再生成まで旧プリセット)。
    // ========================================================================
    case GLFW_KEY_P:
        if (currentMainMode == DEFORM_MODE) {
            const auto& presets = AutoDeform::getPresets();
            gAutoDeformPresetIdx = (gAutoDeformPresetIdx + 1) % static_cast<int>(presets.size());
            const auto& p = presets[gAutoDeformPresetIdx];
            std::cout << "[Preset] -> " << p.name
                      << "  K_fix=" << p.K_fix
                      << "  K_move=" << p.K_move
                      << "  rFixScale="  << p.rFixScale
                      << "  rMoveScale=" << p.rMoveScale
                      << "  tau=[" << p.tauLowScale << "," << p.tauHighScale << "]"
                      << "  minSep=" << p.minSepScale
                      << std::endl;
            std::cout << "       (press 5 to regenerate handles with this preset)" << std::endl;
        }
        break;

    // ========================================================================
    // === Key 0: TetMesh wireframe toggle ===
    // ========================================================================
    case GLFW_KEY_0:
        if (multiBody) {
            multiBody->toggleTetMeshVisible();
            std::cout << "[Wireframe] TetMesh visible: "
                      << (multiBody->isTetMeshVisible() ? "ON" : "OFF") << std::endl;
        }
        break;

    default: break;
    }
}

// ============================================================================
// 後片付け
// ============================================================================
inline void cleanup()
{
    if (multiBody)        { delete multiBody;                                multiBody        = nullptr; }
    if (gTargetMesh)      { gTargetMesh->cleanup();      delete gTargetMesh;      gTargetMesh      = nullptr; }
    if (gBoardMesh)       { gBoardMesh->cleanup();       delete gBoardMesh;       gBoardMesh       = nullptr; }
    if (gLiverStaticMesh) { gLiverStaticMesh->cleanup(); delete gLiverStaticMesh; gLiverStaticMesh = nullptr; }
    deformInit = false;
}

}  // namespace DeformPipeline
