#pragma once
// RegistrationActions.h
// HemiAuto / BIPOP-CMA-ES / Shift+E (SilhouetteAlign) / Metrics
// main.cppのグローバル変数をexternで参照する。

// [COMPILE-CONTEXT SENTINEL] Dependent headers that rely on this file's
// globals (e.g. SilComparePanel.h, which uses liverMesh3D / g_ctrlgs* /
// runShiftE and is meant to be #included only AFTER this one inside
// main.cpp) test for this macro to compile to an empty TU when an IDE or
// build system feeds them in isolation. Defining it here is the single
// source of truth that "RegistrationActions.h has been seen in this TU".
#define REGISTRATION_ACTIONS_H_INCLUDED 1

#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>      // std::memcpy for Step 3d2 pose-hash cache
#include <chrono>      // V3-2 timing diagnostics for runBipopCmaesV3
#include <unordered_set>   // QuadAuto (Shift+O): filter_AR ∩ filter_Quad の hash set

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>

#include "mCutMesh.h"
#include "MeshDrawing.h"
#include "RegistrationCore.h"
#include "NoOpen3DRegistration.h"
#include "OBJTargetExtraction.h"   // for computeVertexNormalsFromFaces (used by runSilhouetteHemi)
#include "CmaesUtils.h"
#include "CmaesRefineV2.h"   // V2: Shift+F entry point (runV2)
#include "CmaesRefineV3.h"   // V3: Shift+G entry point (runBipopCmaesV3)
#include "CmaesRefineV3R.h"  // V3R: Ctrl+G entry point (runBipopCmaesV3R, 4-quadrant region-aware)
#include "CmaesRefineV3RS.h" // V3RS: Ctrl+Shift+G entry point (silhouette anchor, builds on V3R)
#include "IoUDebugDump.h"   // V3RS Phase 2: per-Run hitmap/target/composite PNG dumps
#include "SilOverlayDebug.h" // V3RS Phase 2: ImGui overlay window (KeyD/ARSave pattern)
#include "LiverRegionLabel.h"      // V3R: g_liverRegion 参照のため (anterior/rim/posterior)
#include "RimShapeMatch.h"         // Phase 7b: walkRimChain (Plain W)
#include "LiverLeftRightLabel.h"   // V3R: g_liverLR 参照のため (PURE_R/BOUNDARY/PURE_L) + QuadrantMask
#include "LiverCranioCaudalLabel.h" // V3R-W: g_liverCC 参照のため (CRANIAL/CAUDAL, Ctrl+G Only-Caudal filter)
#include "FullSphereCameraWithTarget.h"
#include "NormalCompatibleRefine.h" // Shift+N / Ctrl+Shift+N: Normal-Compatible refine (Phase 0/2/3 L1+L2)
#include "SoftPartition.h"          // Phase U-1
#include "CpdRigid.h"               // Phase U-CPD: anchor-free pure rigid CPD

// =========================================================
//  main.cpp のグローバル変数への extern 参照
// =========================================================
extern mCutMesh* liverMesh3D;
extern mCutMesh* portalMesh3D;
extern mCutMesh* veinMesh3D;
extern mCutMesh* tumorMesh3D;
extern mCutMesh* segmentMesh3D;
extern mCutMesh* gbMesh3D;
extern mCutMesh* screenMesh;

// Phase U-1: main.cpp で定義
extern SoftPartition::AnchorSet       g_softAnchors;
extern SoftPartition::PartitionField  g_softTgtField;
extern SoftPartition::PartitionField  g_softSrcField;
extern SoftPartition::PartitionParams g_softPartParams;
extern std::vector<float>             g_softGroupRadii;
extern float                          g_softUmeyamaScale;
extern bool                           g_softPartReady;

// Phase U-2: soft-weighted ICP state (main.cpp で定義)
extern SoftPartition::SoftICPParams   g_softIcpParams;
extern int                            g_softIcpLastIters;
extern float                          g_softIcpRmseBefore;
extern float                          g_softIcpRmseAfter;
extern float                          g_softIcpIoUBefore;
extern float                          g_softIcpIoUAfter;
extern glm::dmat4                     g_softIcpAppliedT;
extern bool                           g_softIcpCanRevert;
extern bool                           g_softShowGroups;

// =========================================================
//  Phase U-CPD: pure rigid CPD (anchor-free).
//  ALL CPD state is defined INLINE here (not split into main.cpp
//  globals + extern). RegistrationActions.h is included into main.cpp
//  at line ~664, and every CPD reference in main.cpp (the Ctrl+U
//  dispatch and the U-tab UI) is AFTER that include, so these inline
//  variables are visible there with no extra #include and no extern
//  split. Matches the existing `inline g_fgrSeed` pattern in
//  NoOpen3DRegistration.h and keeps the CPD logic + state co-located.
// =========================================================
enum RegMethod { METHOD_SOFT_ICP = 0, METHOD_CPD = 1 };
inline int               g_regMethod = METHOD_SOFT_ICP;   // which method Ctrl+U runs
inline CpdRigid::Params  g_cpdParams;                     // tunables (see UI)

// [LIVE] Frame-driven convergence animation for CPD and soft-ICP, mirroring
//   g_normRefineLiveMode (Shift+N). When ON, the one-shot run paths (Ctrl+U,
//   "Run all CPD", "Run Soft-weighted ICP") record the per-iteration transform
//   during the (fast) solve and then replay it frame-by-frame so the mesh
//   visibly converges. OFF = the original blocking one-shot behaviour.
inline bool g_regLiveMode           = true;   // animate CPD / soft-ICP
inline int  g_regLiveStepsPerFrame  = 1;      // trajectory entries replayed per render frame

// Staged pipeline state — checkbox 1 -> 5 advance in order.
inline std::shared_ptr<Reg3DCustom::PointCloud> g_cpdTgtCloud; // stage 1 output (raw front-face cloud)
inline std::vector<glm::dvec3> g_cpdSrcD;                 // source (liver) points fed to the solver
inline std::vector<glm::dvec3> g_cpdTgtD;                 // target points fed to the solver (post stage 2)
inline CpdRigid::Result  g_cpdResult;                     // stage 4 output
inline bool              g_cpdStageDone[6] = { false, false, false, false, false, false }; // index 1..5
inline double            g_cpdSigma2Init = 0.0;           // stage 3 preview
inline int               g_cpdSrcCount = 0, g_cpdTgtCountRaw = 0, g_cpdTgtCountDS = 0;
inline int               g_cpdLastIters = 0;

// Before/after metrics + applied transform (for revert).
inline float             g_cpdRmseBefore = 0.0f, g_cpdRmseAfter = 0.0f;
inline float             g_cpdIoUBefore  = 0.0f, g_cpdIoUAfter  = 0.0f;
inline glm::dmat4        g_cpdAppliedT   = glm::dmat4(1.0);
inline bool              g_cpdCanRevert  = false;

extern FullSphereCamera OrbitCam;
extern RegistrationData registrationHandle;

extern glm::mat4 model, view, projection;
extern glm::vec3 objPos;
extern int   gWindowWidth, gWindowHeight;
extern int   gGridWidth;
extern float gDepthScale;
extern float g_voxelSize;

// CT-truth liver size reference (defined in main.cpp, set once at startup
// from diag(liver.obj) in DICOM mm). Used by Shift+M for SCALE_RESTORE and
// by Shift+I (runDilationRmseRefine) as the "implied_scale" diagnostic —
// the ratio of the DA3-frame liver size to the CT-mm size. ~1 means the
// DA3 metric depth and the CT model agree on absolute size.
extern float g_originalLiverDiagMm;
extern bool  g_hasOriginalDiags;

// [Phase B] Forward declaration: actual definition lives in
// PoseLibrary.h as an inline (C++17 inline variable). PoseLibrary.h is
// included AFTER RegistrationActions.h in main.cpp, so this extern is
// needed to publish iou_cand_occluded from Phase E into the Pose
// Library Layer 4 gate. See DESIGN_*.md §4.2.
extern float g_lastSilOccludedIoU2D;

// V3R rim-only RMSE diagnostic publication globals; defined in
// PoseLibrary.h. Set by runBipopCmaesV3R Phase F.5; consumed-and-cleared
// inside poseSaveToLibrary. Display-only — never participates in the
// Layer 4 acceptance gate.
extern float g_lastRimRmse;       // -1.0f = N/A (non-Ctrl+G saves)
extern int   g_lastRimMatched;
extern int   g_lastRimTgtTotal;
extern int   g_lastRimSrcTotal;

// [NEW V3RS-CONTAIN] Containment-direction diagnostic publication globals;
// defined in PoseLibrary.h. Set by runBipopCmaesV3R Phase F.5b alongside
// the IoU_occluded computation; consumed-and-cleared inside
// poseSaveToLibrary. Sentinel -1.0f = N/A (non-Ctrl+G saves, or rasterize
// failed). recall = |src∩tgt| / |tgt|, precision = |src∩tgt| / |src|.
// Display-only.
extern float g_lastIoUOccPrecision;
extern float g_lastIoUOccRecall;

// [Phase A] V3R rim PAIR visualization globals; defined in PoseLibrary.h
// as inline (C++17). Both vectors are size == g_lastRimMatched after a
// successful Ctrl+G / Ctrl+Shift+G session. Consume-and-clear by
// poseSaveToLibrary; restore by applyEntry. Drawn by the colored-pairs
// viewer (Phase D) and saved onto PoseEntry (Phase B).
extern std::vector<int>       g_lastRimPairSrcVertIdx;
extern std::vector<glm::vec3> g_lastRimPairTgtPos;

extern std::vector<glm::vec3> g_cluster1Points;
extern std::vector<glm::vec3> g_cluster2Points;
extern std::vector<glm::vec3> g_targetPoints;
extern std::vector<glm::vec3> g_rejectedBoundaryPoints;   // 器具マスクで棄却された偽境界
extern std::vector<glm::vec3> g_visibleSourcePoints;      // 全可視ソース頂点 (debug)
extern std::vector<glm::vec3> g_silhouetteSourcePoints;   // |n·v| フィルタ後 (debug + SilhouetteHemi で実使用)

// Phase 7b Step 1 (Plain W): source RIM chain debug overlay.
//   `g_debugSourceRimChain` は RimShape::sortRimChainByAngle が生成する、
//   source RIM patch を PCA 主面上の atan2 角度で順序付けした「頂点
//   index」列。描画ループでは liverMesh3D->mVertices からその index で
//   現在位置を fetch するので、ICP / Live tracking で organ が動いても
//   マーカーが追従する (Cyclic correspondence と同じパターン)。
//   Plain W で populate + 表示 ON、再度 W で OFF。
extern std::vector<int> g_debugSourceRimChain;
extern bool             g_showDebugSourceRimChain;

// Phase 7b Step 1 PCA cache.
//   populateDebugSourceRimChain が sortRimChainByAngle を呼ぶ際に同時
//   計算される PCA 量を、Step 3 (Shape Match) で 2 回計算しないよう
//   キャッシュ。Step 3 の solveTwoAxisAlignment では:
//     s_centroid = g_debugSourceRimCentroid
//     s_normal   = g_debugSourceRimMajorNormal      (patch 法線)
//     s_tangent  = g_debugSourceRimPrincipalAxis    (主軸 = 最大固有ベクトル)
//   `g_debugSourceRimPlanarity` は eval[0]/eval[1] で <0.1 = GOOD,
//   <0.3 = OK, それ以上は POOR (角度 sort 順序の信頼性低下サイン)。
extern glm::vec3 g_debugSourceRimCentroid;
extern glm::vec3 g_debugSourceRimMajorNormal;
extern glm::vec3 g_debugSourceRimPrincipalAxis;
extern double    g_debugSourceRimPlanarity;

// Phase 7b Step 2 (Shift+W): target boundary points debug overlay.
//   `g_debugTargetBoundaryPoints` は targetCloud->boundaryDist[i] <
//   g_ctrlgRimTgtThreshPx かつ instrumentDist[i] >= g_instrumentPxThresh
//   を満たす vertex の 3D 座標群。target は静的 (ICP で動かない) なので
//   index ではなく座標を直接保存。順序付け (chain 化) は Step 3 で必要に
//   なったときに source 同様 PCA 角度 sort で行う想定。
extern std::vector<glm::vec3> g_debugTargetBoundaryPoints;
extern bool                   g_showDebugTargetBoundary;

// Phase 7c (REDGE/稜線) overlays — defined in main.cpp.
//   source ridge = 境界(silhouette) ∩ ANTERIOR_CORE (非RIM・非後面)。
//   頂点 index 保持 (rim chain と同じ、mesh 追従)。
//   target ridge = g_debugTargetBoundaryPoints の上半分 (3D 直接保持)。
extern std::vector<int>       g_debugSourceRidge;
extern bool                   g_showDebugSourceRidge;
extern std::vector<glm::vec3> g_debugTargetRidgePoints;
extern bool                   g_showDebugTargetRidge;
extern float                  g_ctrlgRidgeCosBand;
// Phase 7c — target ridge outlier removal (debug).
extern bool  g_ridgeTgtRemoveOutliers;
extern int   g_ridgeTgtOutlierMode;     // 0=SOR, 1=largest connected component
extern int   g_ridgeTgtSorK;
extern float g_ridgeTgtSorStd;
extern float g_ridgeTgtCcRadius;        // 0 → auto
extern int   g_ridgeTgtCcMinPts;
extern bool  g_ridgeTgtSplitLongEdge;   // false=up/down, true=bbox長辺(PCA)

// Phase 7b Step 3 (Ctrl+W): Shape Match coarse search results.
//   `g_debugShapeMatchBestSrc` は best 候補の R, t を source RIM の現在
//   位置に適用して得た 3D 点群。赤点で描画して「ベスト姿勢ならここに
//   行く」予測を可視化 (実際の mesh は動かない — Step 4 で動かす)。
//   `g_debugShapeMatchBestCost` / `g_debugShapeMatchBestK` は診断用。
//   `g_debugShapeMatchBestTransform` は best 候補の 4x4 SE(3)。Step 4
//   (Ctrl+Shift+W) で実 mesh への applyIncrementalTransform に渡す。
extern std::vector<glm::vec3> g_debugShapeMatchBestSrc;
extern bool                   g_showDebugShapeMatch;
extern double                 g_debugShapeMatchBestCost;
extern int                    g_debugShapeMatchBestK;
extern glm::mat4              g_debugShapeMatchBestTransform;

// Phase 7b Step 3 — Shape Match rotation-angle constraint.
//   Apply Init Pose で source 解剖向きは既に target に合っていると仮定
//   し、Shape Match の R は「小さな調整回転のみ」に制限する。R の全体
//   回転角度を cos_angle = (trace(R) - 1) / 2 で測り、cos_angle <
//   thresh の候補に penalty を加算 (= 反転候補 sign=1,2,3 を実質除外)。
//
//     cos_angle = (R[0][0] + R[1][1] + R[2][2] - 1) / 2  ← [-1, +1]
//       +1 = 0°    rotation (identity)
//        0 = 90°   rotation (boundary)
//       -1 = 180°  rotation (full flip)
//     if (cos_angle < g_shapeMatchAnatomyThresh)
//         cost += g_shapeMatchAnatomyLambda * (thresh - cos_angle)
//
//   default: thresh=0.0  (90° 超えを penalize、sign=1/2/3 を確実に除外)
//            lambda=1.0  (180° flip で penalty=1.0、chamfer 0.05-0.15 に
//                          対し圧倒的)
//   どちらも 0 にすれば拘束なし。
//   名残: 「Anatomy」命名は旧版 (d_lr/d_cc dot 拘束) の名残だが、UI 既存
//   スライダ互換性のため維持。
inline float  g_shapeMatchAnatomyThresh  = 0.0f;   // cos_angle threshold
inline double g_shapeMatchAnatomyLambda  = 1.0;    // 0 to disable

// Phase 7b Step 4a — sign mask + Live ICP iter cap.
//
// `g_shapeMatchSignMode`:
//   bit0 = sign 0 (t+, n+) "identity-like"
//   bit1 = sign 1 (t-, n+) "180° around target normal"
//   bit2 = sign 2 (t+, n-) "180° around target tangent"
//   bit3 = sign 3 (t-, n-) "180° around target bitangent"
//   default 0xF = all 4 tried; 0x1 = sign=0 only ("trust Apply Init Pose
//   orientation, allow only small rotations"). Set via Ctrl+G panel
//   checkbox.
//
// `g_shapeMatchLiveMaxIter`:
//   Ctrl+Shift+W (Step 4) で Live ICP の最大 iter を一時的に override
//   する値。default 20 (= Shape Match の解を「絶対維持」に近づける)。
//   Ctrl+Shift+W は g_normRefineMaxIter を save → これに override →
//   startNormalCompatRefineLive (内部で値をコピー) → 即 restore する。
//   0 や負値で「override しない」(= g_normRefineMaxIter 既定値 200 を使う)。
inline uint8_t g_shapeMatchSignMode    = 0xF;
inline int     g_shapeMatchLiveMaxIter = 20;

// Phase 7b Step 4b — Rim Axis Rotation Sweep
//
//   Shape Match で baseline_T 取得後、rim 軸まわりに 360° / N_angles
//   stride で sweep し、全頂点 chamfer 最小の角度を採用する。これに
//   より:
//     - rim 同士の合致を絶対保持 (rim 軸の回転は rim を動かさない)
//     - 反転姿勢を物理的に排除 (全頂点で見ると反転は必ず高 cost)
//     - 残り 1 DOF が全頂点視点で詰まる (= ICP 不要)
//
//   `g_shapeMatchAxisSweepEnabled`:
//     true (default) = Ctrl+Shift+W で Shape Match → axis sweep → ICP skip
//     false           = Ctrl+Shift+W で Shape Match → Live ICP (Step 4a)
//
//   `g_shapeMatchAxisSweepN`: stride 角度数 (default 36 = 10° step)
//
//   `g_shapeMatchAxisSweepTgtSubN`: target 部分集合のサイズ
//     default 5000 = 全 ~180k から uniform stride で約 36 倍ダウンサンプル
//     → 36 angles × 4028 src × 5000 tgt = 725 M ops ~ 1-2 s
inline bool g_shapeMatchAxisSweepEnabled = true;
inline int  g_shapeMatchAxisSweepN       = 36;
inline int  g_shapeMatchAxisSweepTgtSubN = 5000;

// Phase 7b Step 4b — Dual-variant comparison mode
//   true  (default) = Variant A (full vertex sym chamfer) AND
//                     Variant B (rim chain vs target boundary)
//                     を両方走らせ、各 best rotation を試しに適用し、
//                     0-iter session で CompRMSE を計測。RMSE 良い方を
//                     最終採用。デバッグログに両 variant の値出力。
//   false           = Variant A のみ (Step 4b 旧挙動)
//
//   コスト: variant B が ~0.5s、追加 0-iter session 2 回が ~150ms。
//   合計 ~700ms 追加だが信頼性向上。
inline bool g_shapeMatchAxisSweepCompare = true;

// =====================================================================
// Phase 7b Step 3a — Full-2D Ctrl+W cost (depth-free)
// =====================================================================
//   When ON (default), runDebugShapeMatchCoarse switches its cost
//   evaluation from 3D chamfer (legacy Step 3) to 2D AR-projected
//   boundary distance lookup. Target side is the 2D contour traced
//   directly from g_boundaryDistMap — no screenMesh depth lift.
//
//   `g_shapeMatchUse2DCost`        : master toggle (Use 2D AR matching).
//   `g_shapeMatchContourN2D`       : arc-length resample count for the
//                                    2D contour → coarse anchor count.
//                                    default 200 (vs legacy 30 in 3D)
//                                    since 2D evaluation is ~1000× faster.
//   `g_shapeMatchMinInFrameRate`   : reject candidates whose projection
//                                    has < this fraction of source rim
//                                    inside the image frame.
//   `g_shapeMatchOutOfFrameDistPx` : penalty value assigned to source
//                                    points projecting outside the
//                                    viewport. Used inside cost mean.
//   `g_shapeMatchMaxDistCapPx`     : per-point cost cap. Caps both the
//                                    inside-mask distance values and
//                                    the "outside mask" sentinel.
//                                    Bounds the influence of outliers.
//   `g_shapeMatch2DInstThreshPx`   : instrument distance threshold for
//                                    excluding target boundary pixels
//                                    near surgical instruments. 0 to
//                                    disable. Reuses pixel-domain
//                                    convention of g_ctrlgsInstrumentThreshPx.
inline bool  g_shapeMatchUse2DCost         = true;
inline int   g_shapeMatchContourN2D        = 200;
inline float g_shapeMatchMinInFrameRate    = 0.30f;
inline float g_shapeMatchOutOfFrameDistPx  = 100.0f;
inline float g_shapeMatchMaxDistCapPx      = 100.0f;
inline float g_shapeMatch2DInstThreshPx    = 40.0f;

// Phase 7b Step 3a — 2D contour cache + last cost diagnostics.
//   `g_debugTargetContour2D`         : the largest contour segment from
//                                      Shift+W's last trace, in pixel
//                                      space. Used by Ctrl+W (2D mode)
//                                      and for the on-screen overlay.
//   `g_debugTargetContour2DSegSizes` : per-segment sizes (segments[0]
//                                      is the one stored above; later
//                                      ones are dropped for now).
//   `g_debugShapeMatchBestInFrame`   : in-frame fraction of best cand
//                                      (post-2D-eval). For UI display.
//   `g_debugShapeMatchBestInMask`    : ditto, in-mask fraction.
extern std::vector<glm::vec2> g_debugTargetContour2D;
extern std::vector<int>       g_debugTargetContour2DSegSizes;
inline float g_debugShapeMatchBestInFrame = 0.0f;
inline float g_debugShapeMatchBestInMask  = 0.0f;

// =====================================================================
// Phase 7b Step 3b — Gauss-Newton refinement (Alt+W)
// =====================================================================
//
//   Alt+W runs Coarse2D first to get a good initial T, then refines
//   it with PnP-style Levenberg-Marquardt against an UNSIGNED
//   boundary distance field (smooth on both sides of the mask).
//   Sub-pixel rim/contour alignment, ~10–20 ms refine.
//
//   Source-normal sign normalization (default ON, "Idea A"):
//     PCA eigenvectors carry a ± sign ambiguity. We flip the source
//     rim's PCA "normal" so it faces the camera before passing it to
//     solveTwoAxisAlignment. Without this, sign=0 candidates can
//     systematically resolve to a 180°-flipped pose. This benefits
//     both Ctrl+W and Alt+W (toggle for legacy reproduction).
//
//   GN parameters:
//     g_shapeMatchGNMaxIter      : LM iteration cap
//     g_shapeMatchGNLambdaInit   : initial Levenberg damping
//     g_shapeMatchGNEpsStep      : converge if ||Δξ|| below this
//     g_shapeMatchGNEpsRel       : converge if |ΔF/F| below this
//
//   Cached unsigned-distance map:
//     g_gnUnsignedBdy  / W / H / Valid — built once, reused across
//     Alt+W calls. Invalidated when g_boundaryDistMap.invalidate()
//     fires (Run Depth, segmentation mask reload).
inline bool   g_shapeMatchFlipNormalToCamera = true;
inline int    g_shapeMatchGNMaxIter          = 30;
inline float  g_shapeMatchGNLambdaInit       = 1.0e-3f;
inline float  g_shapeMatchGNEpsStep          = 1.0e-5f;
inline float  g_shapeMatchGNEpsRel           = 1.0e-4f;

// Step 3b (revised) — robustness controls (default values mirror the
// fix for the depth-runaway observed in the first Alt+W trial).
//
//   g_shapeMatchCoarseMaxRotDeg    : Coarse2D の hard rotation cap.
//       Init Pose を回転基準として絶対視するなら 45° 以下に。Step 3a
//       の rot cos threshold は「soft penalty」だったため大幅回転候補
//       が cost で勝ってしまっていた。これは hard reject (continue) で
//       絶対通さない。Ctrl+W / Alt+W 双方の Coarse2D 探索に効く。
//       180° = 制約なし (legacy 動作)。45° 推奨。
//   g_shapeMatchGNTranslationOnly  : Alt+W で回転を完全ロックする。
//       PnP の depth 縮退を構造的に排除。Init Pose の回転を信頼すれば
//       並進 3-DoF のみで rim を合わせれば十分。
//   g_shapeMatchGNLambdaMin        : LM ダンピング下限。1e-3 が
//       depth 暴走を防ぐ最小値 (実測。1e-9 まで落とすと破綻)。
//   g_shapeMatchGNStepMax          : trust region cap on ||Δξ|| per
//       iteration. 0.05 ≈ 5 cm + 3° / iter。初回ステップの大暴れを抑止。
//   g_shapeMatchAltWSkipCoarse     : Alt+W から Coarse2D を完全スキップ。
//       「現在 mesh 姿勢からの局所 refine のみ」モード。Init Pose 後の
//       polish 用途として推奨 (default ON)。
//       OFF = 旧動作 (Coarse2D → GN)。
inline float  g_shapeMatchCoarseMaxRotDeg    = 45.0f;
inline bool   g_shapeMatchGNTranslationOnly  = true;
inline float  g_shapeMatchGNLambdaMin        = 1.0e-3f;
inline float  g_shapeMatchGNStepMax          = 0.05f;
inline bool   g_shapeMatchAltWSkipCoarse     = true;

// =====================================================================
// Phase 7b Step 3c — Contour Sweep (Ctrl+Alt+W)
// =====================================================================
//
//   Live "Shift+P 風" sweep: arc-length partition both target contour
//   and source rim, then iterate all (target_anchor, source_pivot,
//   rotation) triples as discrete correspondences. Each candidate
//   anchors a rigid transform that preserves Z by construction
//   (rotation around camera-forward, translation in image plane).
//
//   Driven from main loop via tickContourSweep() — one batch of
//   candidates per rendered frame so the convergence is visible.
//
//   Tunables:
//     g_shapeMatchSweepNTarget   : target contour partition count
//     g_shapeMatchSweepNSource   : source rim partition count
//     g_shapeMatchSweepNRotation : rotation discretization (10° steps)
//     g_shapeMatchSweepFrames1/2 : frame budget for Phase 1 / Phase 2
//                                  (60fps × 2.5s = 150 frames default)
//
//   Returns to PoseLibrary via the standard refine-session flow,
//   identical to Ctrl+W and Alt+W (Apply → undo snapshot → refine →
//   poseSaveToLibrary).
inline int    g_shapeMatchSweepNTarget       = 20;
inline int    g_shapeMatchSweepNSource       = 20;
inline int    g_shapeMatchSweepNRotation     = 36;
inline int    g_shapeMatchSweepFrames1       = 300;   // ~5s @ 60fps
inline int    g_shapeMatchSweepFrames2       = 300;   // total 10s default
inline bool   g_shapeMatchSweepLog           = true;  // verbose per-batch log
inline bool   g_shapeMatchSweepAnimate       = true;  // OFF = one-shot (no animation)

// Endpoint constraint: RIM and target contour are typically OPEN chains
// (caudal-only source + largest-segment target contour). Both have two
// physical endpoints. Knowing which source endpoint maps to which target
// endpoint collapses ~half the search space (no left↔right or anterior↔
// posterior mismatches) and removes the 360° direction ambiguity.
//
//   g_shapeMatchSweepUseEndpointConstraint :
//       ON  = determine endpoint correspondence at sweep start, reverse
//             source chain if needed so that i_tgt=0 ↔ j_src=0 means
//             "same end of curve". Then in the sweep, only allow
//             candidates with |j_src - i_tgt| <= tolerance.
//       OFF = legacy unconstrained sweep (good for closed-loop sources
//             like full rim without caudal filter).
//   g_shapeMatchSweepEndpointTolerance :
//       Allowed misalignment in arc-length steps. 0 = strict diagonal
//       (j_src must equal i_tgt). 3 default = ±3 step slack accommodates
//       arc-length parameterization speed differences. Higher = wider
//       search, but proportionally less benefit.
inline bool   g_shapeMatchSweepUseEndpointConstraint = true;
inline int    g_shapeMatchSweepEndpointTolerance     = 3;
// Diagnostic readback (filled at sweep start)
inline bool   g_shapeMatchSweepDirReversedDiag       = false;
inline bool   g_shapeMatchSweepSrcIsOpenDiag         = true;

// ---------------------------------------------------------------------
// Phase 7b Step 3c++: Orientation lock — anatomical L/R + C/C label
// based (Plan B, replaces the earlier angle-based scheme).
//
//   Premise: Apply Init Pose puts the source's anatomical RIGHT on
//   the screen +x side and CRANIAL on the screen -y side (this is
//   exactly what getPresetRotation does via the PCA-derived d_lr,
//   d_cc unit vectors). The label-based lock just enforces that the
//   sweep doesn't unwind that correspondence.
//
//   g_shapeMatchSweepUseOrientationLock:
//     Master toggle (default ON). Drives BOTH:
//       Check A (label) — each candidate (i_tgt, j_src) must satisfy
//                         the LR/CC label compatibility table below.
//       Check B (rot)   — |wrap_180(θ_rotation_deg)| ≤ RotationLockDeg.
//     Auto-degraded to false at sweep start if g_liverLR or g_liverCC
//     is not yet computed (with diagnostic log).
//
//   g_shapeMatchSweepNeutralBandPx:
//     Half-width of the "AMBIGUOUS" band around the locked target
//     centroid in pixel space. Anchors with |dx| ≤ band_px are
//     SS_LR_AMBIGUOUS (pair with any source LR label). Similarly for
//     dy and CC. Default 80 px ≈ 4% of a 1920-wide frame — comparable
//     to the BOUNDARY-label band on the source side.
//
//   g_shapeMatchSweepRotationLockDeg:
//     Check B cap on the candidate's θ in degrees. Default 90°. This
//     is the absolute brake — even if the label check accidentally
//     passes (LR-symmetric rim shapes can fool it), |θ| > cap is
//     exactly the image-plane flip the user is complaining about.
//
//   Label compatibility table (skip iff mismatched):
//     | src LR \ tgt LR      | RIGHT  | LEFT   | AMBIGUOUS |
//     |-----------------------|--------|--------|-----------|
//     | PURE_RIGHT            |  OK    |  skip  |   OK      |
//     | PURE_LEFT             |  skip  |  OK    |   OK      |
//     | BOUNDARY (rim 屈曲帯) |  OK    |  OK    |   OK      |
//     CC same with CRANIAL/CAUDAL.
//
//   Diagnostic counters logged each batch:
//     ep_skip   = endpoint constraint (OPEN-rim only)
//     lbl_skip  = label compatibility (Check A)
//     rot_skip  = θ_rotation magnitude (Check B)
//     bbox_skip = source-bbox tag (Plan A active-mask)
// ---------------------------------------------------------------------
inline bool   g_shapeMatchSweepUseOrientationLock   = true;
inline float  g_shapeMatchSweepNeutralBandPx        = 80.0f;
inline float  g_shapeMatchSweepRotationLockDeg      = 90.0f;
// Per-sweep diagnostic readbacks (filled at sweep start).
inline bool   g_shapeMatchSweepLabelLockReadyDiag   = false;

// Anatomical / "stay near initial pose" filter for target boundary.
//   g_shapeMatchSweepFilterByRim:
//     ON  = filter target boundary points to source rim's 2D bbox
//           (expanded by margin) before sector extraction. This is
//           THE essential anatomical constraint: it prevents the
//           sweep from pinning source's caudal pivot onto target's
//           cranial anchor (which would translate source far from
//           its initial pose, violating the Apply Init Pose prior).
//     OFF = use all boundary points (legacy / debugging only).
//   g_shapeMatchSweepFilterMarginPx:
//     Expansion of the bbox in pixels. 100-200 px is a reasonable
//     range. Larger → more pose-correction freedom, but also more
//     anatomically risky.
inline bool  g_shapeMatchSweepFilterByRim    = true;
inline float g_shapeMatchSweepFilterMarginPx = 150.0f;

// Trial-pose visualization (rendered yellow during sweep). Distinct
// from g_debugShapeMatchBestSrc (red = global best so far) so the user
// can see both: red converges, yellow dances through current trials.
inline std::vector<glm::vec3> g_contourSweepTrialSrc;
inline bool                   g_contourSweepShowTrial = false;

// ----- Anchor/pivot visualization (verifies sweep is indexing both sides) ----
// Target anchors: 20 contour samples back-projected to 3D at the source
// rim's centroid depth so they appear on the AR image plane in the scene.
// Source pivots: 20 source rim samples transformed by the current
// best-in-batch T so they sit on the yellow trial rim.
// Highlighted dot (cyan) marks the i_tgt / j_src of the current best-in-
// batch candidate — this is the pair the algorithm is currently
// proposing as a correspondence. Watching the highlighted dots walk
// through the 20-point discretization confirms the animation is exercising
// both sides.
inline bool                   g_shapeMatchSweepShowAnchors = true;
inline std::vector<glm::vec3> g_contourSweepTgtAnchors3D;     // 20 pts, fixed
inline std::vector<glm::vec3> g_contourSweepSrcPivotsTrial;   // 20 pts, animates
inline int                    g_contourSweepCurrentITgt = -1;
inline int                    g_contourSweepCurrentJSrc = -1;

// Phase 7b Step 3c+ "preview anchors": independent toggle that
// recomputes the 20 target anchors (rainbow-coloured) and the 20
// source pivots (rainbow-coloured) from the current target contour /
// source rim chain — works OUTSIDE of an active sweep, so the user
// can verify the discretization placement BEFORE pressing Ctrl+Alt+W.
//
// Colour: HSV around the full hue circle so anchor 0 is red, anchor
// 5 is yellow-green, anchor 10 is cyan, anchor 15 is magenta — making
// the curve direction obvious at a glance and exposing any clustering
// or maldistribution.
inline bool g_shapeMatchSweepPreviewAnchors = false;

// Defined inline (C++17) so the single instance lives in the header,
// avoiding the include-order trap (main.cpp uses RimShape symbols
// before RimShapeMatch.h is included if we define the global in main.cpp).
inline RimShape::ContourSweepState g_contourSweepState;

// =====================================================================
// Phase 7b Step 3d: Silhouette 2D Dense Sweep — globals & UI toggles
// =====================================================================
// Master switch for new method. When ON, Ctrl+Alt+W runs the
// silhouette dense sweep instead of the old sector-based contour
// sweep. Both paths coexist; only one is active at a time.
inline bool g_silhouetteSweepEnable = false;

// ---------------------------------------------------------------------
// CB1 / sweep source-rim discretization method (Step 3d Stage A).
//   Selects WHICH algorithm populates g_silSwSrcRim2DPreview /
//   g_silSwSrcPivots2DPreview / g_silSwSrcRim3DPreview etc.
//   Both methods coexist; the popup CB1 + Ctrl+Alt+W sweep both
//   consume whichever method is active.
//
//   ENVELOPE        = legacy angle-bin + max-radius envelope (closed
//                     loop topology, wrong for the caudal-RIM arch but
//                     kept as a fallback / comparison).
//   MST_LONGEST_PATH = CB0.2 result (grid + KNN + MST + longest path,
//                      open polyline topology — anatomically correct).
//
//   Default is MST_LONGEST_PATH because CB0/CB0.1/CB0.2 visualization
//   confirmed the source RIM is an open arch; envelope is wrong by
//   construction.
// ---------------------------------------------------------------------
enum SrcRimMethod : int {
    SRC_RIM_METHOD_ENVELOPE         = 0,
    SRC_RIM_METHOD_MST_LONGEST_PATH = 1,
};
inline int g_silSwSrcRimMethod = SRC_RIM_METHOD_MST_LONGEST_PATH;

// Debug popup toggles (independent of sweep activation).
//   CB0: raw RIM 2D projection (points only, no ordering / envelope)
//   CB1: source 2D-projection popup window (envelope + pivots)
//   CB2: target lower-half + anchors popup window
// Useful before pressing Ctrl+Alt+W to verify discretization.
//
// CB0 rationale (Step 3d, 2026-05-23):
//   CB1 applies an angle-bin + max-radius "envelope" step that forces
//   a closed loop. If the actual anatomical RIM is open (an arch — the
//   typical caudal-RIM case), the envelope will incorrectly close it
//   across the top of the screen. CB0 bypasses ALL ordering /
//   filtering / envelope logic — it simply projects every
//   g_debugSourceRimChain vertex to 2D pixel space and dots them on a
//   canvas — so the user can visually answer the question:
//      "Is the anatomical source RIM, as the camera sees it, an open
//       arch or a closed loop?"
inline bool g_debugShow2DProjPopup_RawRim = false;   // [0] CB0 — points-only debug

// CB0.1 (Stage 0.1): smoothed RIM 2D projection.
//   Takes the same raw projection as CB0 and applies two cleanup steps:
//     1. Grid-cell aggregation: bin points into G×G px cells, replace
//        each cell's points with a single centroid (LR label = majority).
//        This kills density-driven noise — clusters become one point.
//     2. KNN smoothing: for each centroid, find K nearest neighbours
//        in 2D and replace with their mean position. Iterate N times.
//        This smooths the resulting curve without imposing topology.
//   NEITHER step orders the points; the output is still an unordered
//   point set (so it can be honestly compared against CB0). This is the
//   stage between CB0 (raw) and CB1 (envelope-closed).
inline bool  g_debugShow2DProjPopup_RawRimSmoothed = false;   // [0.1]
inline float g_rawRimSmooth_GridPx       = 15.0f;             // bin cell size
inline int   g_rawRimSmooth_KnnK         = 5;                 // KNN neighbours
inline int   g_rawRimSmooth_KnnIters     = 2;                 // KNN passes
inline bool  g_rawRimSmooth_ShowRawOverlay = true;            // overlay raw

// CB0.2 (Stage 0.2): ordered RIM via MST + longest path.
//   Takes the cleaned points from CB0.1 (same grid/KNN parameters)
//   and orders them into an OPEN polyline:
//     1. Build MST on the cleaned points (Prim, O(N²)).
//        Edges longer than g_rawRimOrder_MaxEdgePx are filtered out;
//        if this disconnects the graph, only the largest connected
//        component is used.
//     2. Find the longest path in the MST via two-pass BFS.
//     3. Orient so the start endpoint is closer to PURE_RIGHT 3D
//        centroid → 2D.
//     4. Arc-length resample N pivots along the open path.
//   This is the correct topology for the caudal RIM arch observed
//   in CB0: no forced loop closure, no envelope artefact.
inline bool  g_debugShow2DProjPopup_RawRimOrdered = false;    // [0.2]
inline float g_rawRimOrder_MaxEdgePx   = 100.0f;              // MST edge cutoff
inline int   g_rawRimOrder_NPivots     = 20;                  // pivots on path
inline bool  g_rawRimOrder_ShowMST     = false;               // overlay MST
inline bool  g_rawRimOrder_ShowCleaned = true;                // overlay cleaned

// CB0.3 (Stage 0.3): Manual Sweep Probe — interactive sweep candidate viewer.
//   Lets the user pick (PIVOT i, ANCHOR j, rotation step k) via sliders
//   and see the resulting source-on-target overlay in real time. Reuses
//   the exact same transform formula as evaluateSilhouetteSweepCandidate
//   so the geometry matches the sweep byte-for-byte.
//
//   When `lock_j_eq_i` is ON, j follows i (matches the actual sweep
//   convention where pivot[i] is paired with anchor[i]). When OFF, the
//   user can freely explore i ≠ j combinations (which the sweep never
//   tries, but is useful for debugging).
//
//   Rotation is exposed as a timestep (k = 0..n_rotation-1, with
//   θ = k * (360/n_rotation)). An "animate" toggle steps k forward
//   automatically so the user can watch the rotation sweep through.
//
//   Adds CC orientation indicator (arrow + head-up/FLIPPED text) using
//   g_liverCC.d_cc to diagnose the cranio-caudal flip issue observed
//   in Phase 1.
inline bool  g_debugShow2DProjPopup_OverlayProbe = false;     // [0.3]
inline int   g_overlayProbe_PivotI        = 0;       // source pivot 0..N-1
inline int   g_overlayProbe_AnchorJ       = 0;       // target anchor 0..M-1
inline int   g_overlayProbe_RotStep       = 0;       // rotation step 0..n_rot-1
inline int   g_overlayProbe_NRotation     = 36;      // matches sweep default
inline bool  g_overlayProbe_LockJI        = false;   // OFF by default →
                                                     // independent i / j slider operation;
                                                     // user can flip ON to match sweep convention
inline bool  g_overlayProbe_ShowCCArrow   = true;    // CC direction arrow
inline bool  g_overlayProbe_ShowSrcOverlay = true;   // transformed source
inline bool  g_overlayProbe_ShowTgtOverlay = true;   // target lower + anchors
inline bool  g_overlayProbe_AutoAnimate   = false;   // auto-advance k
inline int   g_overlayProbe_AnimFramesPerStep = 5;   // throttle for animation
inline int   g_overlayProbe_AnimFrameCounter  = 0;   // internal frame tally

// ---- CB0.3 simulate flags: apply sweep本番's Check A / Check B in CB0.3 ----
// When ON, CB0.3 shows "REJECTED BY A/B" + tints canvas red for the
// candidates that sweep本番 would reject. Lets the user verify the
// guard logic visually before trusting sweep本番.
inline bool  g_overlayProbe_SimulateCheckA = true;
inline bool  g_overlayProbe_SimulateCheckB = true;

// =====================================================================
// Step 3d sweep candidate guards (本番 Ctrl+Alt+W)
// =====================================================================
// Check A: rotation magnitude cap. Wraps θ to (-180°, +180°] and
//   rejects any candidate whose |θ| exceeds g_silSwCheckA_RotCapDeg.
//   Independent of CC labels — works whenever sweep runs.
// Check B: CC orientation guard. Requires g_liverCC.valid(). Projects
//   the CRANIAL→CAUDAL direction (g_liverCC.d_cc) under the candidate
//   transform and rejects candidates where the on-screen CC vector
//   deviates from "pointing down" (90° = 6 o'clock = +y_pixel) by
//   more than g_silSwCheckB_CCToleranceDeg.
//
//   Both default ON. ±30° / ±15° per user spec (= "前後30°" and
//   "5:55-7:05 ≈ ±5° but ±15° gives realistic working tolerance").
// =====================================================================
inline bool  g_silSwCheckA_Enable        = true;
inline float g_silSwCheckA_RotCapDeg     = 30.0f;
inline bool  g_silSwCheckB_Enable        = true;
inline float g_silSwCheckB_CCToleranceDeg = 15.0f;

inline bool g_debugShow2DProjPopup_Source = false;
inline bool g_debugShow2DProjPopup_Target = false;

// Sweep state machine. Same inline-singleton pattern as
// g_contourSweepState.
inline RimShape::SilhouetteSweepState g_silhouetteSweep;

// Frame-pacing tuning (mirrors the Step 3c slider semantics).
inline int  g_silhouetteSweepFrames1 = 60;     // Phase 1 frame budget
inline int  g_silhouetteSweepFrames2 = 30;     // Phase 2 frame budget
inline bool g_silhouetteSweepLog     = false;     // default OFF (was true);
                                                  // turn ON only when debugging
inline bool g_silhouetteSweepAnimate = true;

// Preview cache: source 2D projection + 20 pivots (rebuilt every frame
// the source-popup is visible, since the source mesh can move between
// sweeps and the popup must reflect the current pose).
// Target cache is rebuilt only when populateDebugTargetBoundary runs
// (target is static during a session).
inline std::vector<glm::vec2> g_silSwSrcRim2DPreview;
inline std::vector<glm::vec2> g_silSwSrcPivots2DPreview;
inline std::vector<uint8_t>   g_silSwSrcRim2DPreviewLR;   // per dense point
inline std::vector<uint8_t>   g_silSwSrcPivotsLRPreview;  // per pivot
inline int                    g_silSwSrcStartIdxPreview = -1;
inline glm::vec2              g_silSwSrcRightCentroid2DPreview = glm::vec2(0.0f);

// [Stage A] 3D pivot positions computed inline with the MST method.
// When non-empty, startSilhouetteSweep uses this directly instead of
// its own closed-loop resampling (which is wrong for an open polyline).
// Always cleared on a cache miss; populated only by the MST dispatcher;
// stays empty for the envelope path so the legacy resampling fires.
inline std::vector<glm::vec3> g_silSwSrcPivots3DPreview;

// [Step 3d revised] AR-silhouette source dense data, parallel to
// g_silSwSrcRim2DPreview. Populated by silSwBuildSrcPreview, consumed
// by startSilhouetteSweep to seed src_rim_3D_oriented + src_pivots_3D.
// Replaces the old g_debugSourceRimChain-based dense source.
inline std::vector<glm::vec3> g_silSwSrcRim3DPreview;     // dense 3D (mesh-space), oriented
inline std::vector<int>       g_silSwSrcRimVIdxPreview;   // dense vertex idx, parallel

// [Step 3d2] Source preview cache key. silSwBuildSrcPreview is called
// every frame the popup is open, but recomputation is expensive
// (BVH build + extractVisibleVerticesCustom prints ~50ms + log spam).
// We hash a few mesh-vertex coordinates as a proxy for rigid-pose
// change and skip the rebuild when the hash matches.
inline uint64_t g_silSwSrcPreviewCacheHash = 0;
inline bool     g_silSwSrcPreviewCacheValid = false;

inline std::vector<glm::vec2> g_silSwTgtLower2DPreview;
inline std::vector<glm::vec2> g_silSwTgtAnchors2DPreview;
inline glm::vec2              g_silSwTgtCentroid2DPreview = glm::vec2(0.0f);

extern std::vector<float> g_gnUnsignedBdy;
extern int                g_gnUnsignedBdyW;
extern int                g_gnUnsignedBdyH;
extern bool               g_gnUnsignedBdyValid;

// GN diagnostics (filled by last Alt+W call, displayed in panel + log)
inline double g_debugShapeMatchGNInitCost   = 0.0;
inline double g_debugShapeMatchGNFinalCost  = 0.0;
inline int    g_debugShapeMatchGNIters      = 0;
inline bool   g_debugShapeMatchGNConverged  = false;
inline int    g_debugShapeMatchGNReason     = -1;     // 0=step,1=rel,2=max,3=fail
inline int    g_debugShapeMatchGNInFrame    = 0;

// Debug snapshot of last axis sweep (visualizers / logging)
inline glm::mat4 g_debugShapeMatchAxisSweepT       = glm::mat4(1.0f);
inline double    g_debugShapeMatchAxisSweepCost    = 0.0;
inline float     g_debugShapeMatchAxisSweepAngle   = 0.0f;
extern bool g_showClusterVisualization;
extern bool g_quietMetrics;

// V3R: 4-quadrant region-aware Ctrl+G で参照する頂点ラベル。
// main.cpp で定義され、HemiAuto (O) 直後に LiverRegionLabel::labelVertices /
// LiverLeftRightLabel::labelVertices で計算される。runBipopCmaesV3R は
// これらを ParamsV3R::region_labels / lr_labels にコピーして V3R driver
// に渡す。再ロード時のみ再計算 (HANDOVER §8.4)。
extern LiverRegionLabel::Result    g_liverRegion;
extern LiverLeftRightLabel::Result g_liverLR;

// V3R-W: caudal-only filter (Ctrl+G Only-Caudal) で参照する解剖学的
// CC ラベル。main.cpp で定義され、Shift+H または applyInitRotation の
// auto-trigger で計算される。Ctrl+G の caudal-only フィルタが ON のとき
// だけ参照する。未計算なら警告を出して caudal-only を当該回限り無効化
// (Region/LR と違って fatal abort はしない: 他のフィルタは無関係に動く)。
extern LiverCranioCaudalLabel::Result g_liverCC;

// QuadAuto (Shift+O) で参照する解剖象限ビットマスク。
// main.cpp で定義され、Initial Orientation panel / Ctrl+G panel と共有する
// (LiverLeftRightLabel::QUAD_AR | QUAD_AL | QUAD_PR | QUAD_PL の OR、
//  QUAD_ALL = 0x0F で全象限選択)。
extern uint8_t g_activeQuadrantMask;

// =========================================================
// =========================================================
//  Instrument-aware boundary rejection threshold (pixels in mask space).
//  Vertices whose reprojected pixel sits within this radius of any
//  instrument-mask pixel are NOT counted as liver boundary, even if their
//  boundaryDist < kBoundaryPxTh. Adjustable via the sidebar slider
//  (range 0-50 px; see RegistrationImGuiManager.h:1349).
//  Default 20 px — operational value used during development. The slider
//  caps at 50 px, so larger code-side values are out of range; previously
//  this was set to 80 which silently bypassed the UI's design intent and
//  rejected ~35% of boundary candidates in our scenes (Ctrl+G log
//  [Boundary3D] rejected_by_instrument vs accepted). See B-key
//  visualization for per-scene tuning.
// =========================================================
inline float g_instrumentPxThresh = 40.0f;

// =========================================================
//  Source-side silhouette threshold (used by SilhouetteHemi / Key P).
//  A visible vertex is treated as a silhouette vertex when
//      |dot(normal, viewDir)| < g_silhouetteSrcCosThresh
//  i.e. the surface there is nearly tangent to the line of sight.
//  Range: 0 (only exact silhouette) -> 1 (all visible). Practical range
//  0.3 - 0.5; default 0.40.
// =========================================================
inline float g_silhouetteSrcCosThresh = 0.40f;

// =========================================================
//  Ctrl+G (V3-R) Rim-weighted extension — UI state + viz storage
// ---------------------------------------------------------
//  Three opt-in features stacked on Ctrl+G's existing 4-quadrant
//  region-aware BIPOP-CMA-ES:
//
//    1. AR-camera visibility filter (g_ctrlgUseArVisFilter)
//       Restricts the source subset to vertices visible from the
//       fixed AR camera (cam_pos = (0,0,0), look-at = (0,0,1)).
//       Computed once per Ctrl+G session via raycastVisibilityBVH.
//       Default OFF: subset_idx_voxel matches the original V3-R
//       behavior, byte-identical to Shift+G at QUAD_ALL.
//
//    2. Rim-rim multiplicative weight (g_ctrlgBetaRimWeight)
//       The inner CMA-ES cost replaces RMSE = sqrt(sum d^2 / count)
//       with sqrt(sum w*d^2 / sum w) where
//           w_i = 1 + beta * is_rim_src[j] * is_rim_tgt[i].
//       beta=0 -> uniform (byte-identical accumulator).
//       Source rim = LiverRegionLabel::RIM band (mesh-intrinsic).
//       Target rim = boundaryDist < g_ctrlgRimTgtThreshPx (image-side).
//
//    3. RIM-pair visualization (g_ctrlgShowRimPairs)
//       Draws orange spheres at source RIM vertices and magenta
//       spheres at target RIM points. Populated at Ctrl+G session
//       entry; persists until next session.
// =========================================================
inline bool  g_ctrlgUseArVisFilter   = false;
inline float g_ctrlgBetaRimWeight    = 5.0f;
inline float g_ctrlgGammaRedgeWeight = 0.0f;   // [Phase 7c] REDGE 稜線重み (0=OFF, 従来一致)
inline bool  g_ctrlgBidirMatching    = false;  // [Phase 7c] 双方向(対称)マッチング (B方式: RIM/REDGEのみ, OFF=従来一致)
inline bool  g_ctrlgBidirAllPoints   = false;  // [Phase 7c] (A) 双方向を全subset点へ (要 g_ctrlgBidirMatching=ON)
inline float g_ctrlgRimTgtThreshPx   = 12.0f;   // matches Shift+P kBoundaryPxTh
// [UI整理] Default OFF: the RIM correspondence-pair overlay is opt-in
// (it draws orange/magenta spheres for every rim pair, which clutters the
// default Ctrl+G / Ctrl+I view). Turn it on in the Debug Panel when needed.
inline bool  g_ctrlgShowRimPairs     = false;

// [Phase D] Colored RIM pairs viewer (the K-representative variant).
//   g_ctrlgShowColoredRimPairs : tick to overlay K colored sphere pairs
//                                (src + tgt, same HSV hue per pair) on
//                                top of the rim sets drawn by
//                                g_ctrlgShowRimPairs. Source spheres
//                                follow liverMesh3D->mVertices, so the
//                                pairs track ICP/Apply updates.
//                                Default OFF — backwards compatible.
//   g_ctrlgColoredRimN         : how many pairs to draw, 5..30.
//                                Default 10 (small enough to
//                                differentiate 10-bucket HSV by eye).
//   g_ctrlgColoredRimMode      : sampling strategy (cast to
//                                RimPairSampling::Mode at call site).
//                                0=ArcUniform, 1=WorstK, 2=BestK, 3=Random.
//                                ArcUniform default — stable visual layout
//                                across runs makes Pose Library entries
//                                directly comparable.
//   g_ctrlgColoredRimSeed      : reshuffle counter. Increment on
//                                "Reshuffle" button click to force a
//                                fresh sample (only matters for Random;
//                                deterministic modes ignore it).
//
// Data source: g_lastRimPair* globals (set by Phase F.5, restored by
// applyEntry). When those are empty the viewer silently draws nothing.
inline bool     g_ctrlgShowColoredRimPairs = false;
inline int      g_ctrlgColoredRimN         = 10;
inline int      g_ctrlgColoredRimMode      = 0;     // 0 = ArcUniform
inline uint32_t g_ctrlgColoredRimSeed      = 1u;

// V3R-W (R-feat-2): caudal-only filter (anatomical CC axis).
//   g_ctrlgUseCaudalOnly       : tick to AND-restrict source subset to
//                                LiverCranioCaudalLabel::CAUDAL vertices.
//   g_ctrlgArvisCaudalCombine  : when BOTH AR-vis and Caudal-only are ON,
//                                this picks the combine mode (0=AND, 1=OR).
//                                When only one is ON, this is ignored.
// Both default to a state that preserves V3R / V3 byte-identical behaviour:
//   OFF + (default AND mode is irrelevant since the filter is off).
inline bool    g_ctrlgUseCaudalOnly       = true;
inline uint8_t g_ctrlgArvisCaudalCombine  = 0;   // 0=AND (default), 1=OR

// =========================================================
//  [NEW V3R/SearchMode] Reduced-DoF search mode for Ctrl+G.
//  --------------------------------------------------------
//  Selects the CMA-ES decision-vector dimension for V3R:
//    SEVEN_DOF        : tx,ty,tz, rx,ry,rz, scale  (V3R byte-identical)
//    SIX_DOF_RIGID    : tx,ty,tz, rx,ry,rz         (scale frozen at 1)
//    FOUR_DOF_XYRXRY  : tx,ty,    rx,ry            (tz/rz/scale frozen)
//
//  Controlled by the "Search dimension (Ctrl+G)" radio group inside
//  the Ctrl+G Quadrant Selector window. SEVEN_DOF default preserves
//  byte-identical behaviour. Reduced modes also pre-scale the t/r
//  ranges and BIPOP jitter inside runBipopCmaesV3R(): SIX_DOF_RIGID
//  uses 0.7x and FOUR_DOF_XYRXRY uses 0.5x.
//
//  Motivation (HANDOVER_UNIFIED_ALL.md §III/3.1): scale blowup is the
//  failure mode where repeated Ctrl+G sessions inflate the source
//  mesh. SIX_DOF_RIGID and FOUR_DOF_XYRXRY remove scale (and the
//  TZ/RZ axes coupled to apparent-scale in pinhole projection) as
//  hard constraints rather than soft penalties.
// =========================================================
inline CmaesRefineV3R::SearchMode g_ctrlgSearchMode =
    CmaesRefineV3R::SearchMode::SEVEN_DOF;

// CMA-ES min-match-ratio override for Ctrl+G. ParamsV3 / V3R default
// is 0.30; lower values (0.15-0.25) can help reduced-DoF modes if the
// constrained search starts from a poor pose that would otherwise hit
// the penalty floor at Gen 0.
inline float g_ctrlgMinMatchRatio = 0.30f;

// =========================================================
//  Shift+N / Ctrl+Shift+N : Normal-Compatible Refine controls.
// ---------------------------------------------------------
//  Independent from Ctrl+G / Ctrl+Shift+G. The wrapper
//  runNormalCompatRefineSession reads these globals and feeds them
//  into NormalRefine::RefineParams. Defaults match the header struct
//  defaults so the very first press of Shift+N produces a sensible
//  refinement with rim L1 = 0 and anchor L2 = OFF (= pure NormalCompat
//  ICP). Turn knobs on to layer in rim weighting / anchor pairs.
//
//  Designed as the "finishing pass" after Ctrl+G:
//    Apply Init Pose → Ctrl+P → Ctrl+G → Shift+N (NormalCompat polish)
//                              ↘ Ctrl+Shift+N (SRT-Variance polish)
//
//  Source / target filters are SHARED with Ctrl+G via the existing
//  g_ctrlgUseArVisFilter / g_ctrlgUseCaudalOnly / g_activeQuadrantMask
//  globals. So whatever the user has ticked in the Ctrl+G panel is
//  what Shift+N uses too — no separate filter UI to keep in sync.
// =========================================================

// Master enable. When false, Shift+N still runs but produces no
//   actual pose change (the wrapper early-returns). Useful for
//   quickly disabling the feature without unbinding the key.
inline bool  g_normRefineEnabled         = true;

// Per-iteration solver knobs (mirror NormalRefine::RefineParams).
//   distanceThreshold : sigmoid centre; larger -> more far points
//                       contribute (good when ICP needs long-range
//                       attraction). Scales with scene size.
//   minNormalCos      : NormalCompat ignores; SRT_VARIANCE uses as
//                       the annealed start cosine threshold. 0.30
//                       means accept up to ~72.5° between normals.
//   maxIter           : hard cap on outer iterations across the
//                       blocking loop (the inner header runs
//                       itersPerFrame=2 sub-steps per outer step).
//   itersPerFrame     : sub-steps inside one refineStep call.
//                       Header default 2 = "two Gauss-Newton steps
//                       between recomputing correspondences".
inline float g_normRefineDistThresh      = 0.15f;
inline float g_normRefineMinNormalCos    = 0.30f;
inline int   g_normRefineMaxIter         = 200;
inline int   g_normRefineItersPerFrame   = 2;

// [Phase 2] L1 rim multiplicative weights.
//   betaSrc = 0.0  → source-rim points get the same weight as interior
//   betaSrc = 1.0  → source-rim points get DOUBLE weight (1+1)
//   betaTgt is analogous on the target side (boundaryDist<thresh).
// Default 0 keeps the byte-identical contract with the pre-Phase-2
// wrapper (any later regression chase starts from this baseline).
inline float g_normRefineBetaRimSrc      = 0.0f;
inline float g_normRefineBetaRimTgt      = 0.0f;

// [Phase 3] L2 anchor pair usage.
//   useAnchor       : master toggle. When OFF, NormalRefine sees
//                     an empty anchor array and runs in pure-NN
//                     mode regardless of phaseIter/blend.
//   anchorPhaseIter : number of OUTER iterations (totalIterations)
//                     during which anchored vertices use the anchor
//                     target. After this, anchors are silently
//                     ignored and the loop converges via pure NN.
//   anchorBlend     : 1.0 = pure anchor, 0.0 = ignore anchor, 0.5
//                     = halfway between anchor and current NN.
// The actual pair data comes from g_lastRimPair* (populated by the
// most-recent Ctrl+G / Ctrl+Shift+G Phase F.5 and restored by
// Pose Library apply). When empty, useAnchor has no effect.
inline bool  g_normRefineUseAnchor       = true;
inline int   g_normRefineAnchorPhaseIter = 20;
inline float g_normRefineAnchorBlend     = 1.0f;

// Method selection mirror. main.cpp's KEY_N handler sets this to the
// method that was pressed (Shift+N → 0, Ctrl+Shift+N → 1) so the
// floating panel can show "Last method: NORMAL_COMPAT" without
// duplicating state. Not read by the wrapper itself (the wrapper
// takes method as an explicit argument).
inline int   g_normRefineLastMethod      = 0;  // 0 = NORMAL_COMPAT, 1 = SRT_VARIANCE

// Last-session status mirror (read by the UI for the status line).
inline int   g_normRefineLastIter        = 0;
inline float g_normRefineLastInitialRMSE = -1.0f;
inline float g_normRefineLastBestRMSE    = -1.0f;
inline bool  g_normRefineLastAccepted    = false;
inline bool  g_normRefineLastConverged   = false;

// =========================================================
//  Live mode controls (Phase 6).
// ---------------------------------------------------------
//  When g_normRefineLiveMode is TRUE (default), Shift+N / Ctrl+Shift+N
//  start a frame-driven refinement: each render frame runs ONE
//  refineStep call, applies the resulting incremental transform to
//  organMeshes immediately, and re-renders. The user sees the mesh
//  "track" the target like an SRT-3D object tracker, instead of just
//  snapping to the final pose after a 4-8 second block.
//
//  When FALSE, both keys fall back to the blocking wrapper used until
//  Phase 5 — finished in one frame, mesh moves once at the end.
// =========================================================
inline bool  g_normRefineLiveMode        = true;
// Mirror values updated each tick so the UI panel can show progress
// while a live session is running.
inline bool  g_normRefineLiveActive      = false;
inline float g_normRefineLiveCurrentRMSE = -1.0f;
inline int   g_normRefineLiveAnchorPhase = 0;  // -1=not anchored, 0=inactive, 1=active

// =========================================================
//  [Phase 7a] Pure RIM mode
// ---------------------------------------------------------
//  When TRUE, restrict the refinement to RIM correspondences only:
//    source = liver verts AND LiverRegionLabel::RIM
//    target = boundaryDist < g_ctrlgRimTgtThreshPx (instrument-aware)
//
//  This is a HARD filter, not a weight (= different from Phase 2 L1
//  betas). The intent is to test "rim-curve to rim-curve" alignment
//  without interior support. Expected to be fast (765 src verts vs
//  3270 in the implicit Q:AR case observed in the validation log) but
//  prone to in-plane rotation drift (rim is roughly a 1D curve, so
//  rotation perpendicular to it is under-constrained).
//
//  Best used after Ctrl+M (Shape Match coarse-to-fine, Phase 7b) has
//  found a good initial rotation. When the curves are roughly aligned,
//  Pure RIM Live polishes the residual much faster than full ICP.
//
//  Default OFF: byte-identical to Phase 6 behaviour when unticked.
// =========================================================
inline bool g_normRefinePureRim = false;

// [Phase 6 UX] How many refineStep calls per render frame.
//   1  = ~30-60 outer iters / sec at 60 FPS (slow, dramatic motion).
//   2-3 = faster animation, still visible.
//   5+ = effectively "blocking but spread over a few frames".
// Slider in the Normal-Compatible Refine panel adjusts this live.
inline int   g_normRefineLiveStepsPerFrame = 1;

// =========================================================
//  Ctrl+Shift+G (V3RS, silhouette-anchored) controls.
//  COMPLETELY INDEPENDENT from Ctrl+G (V3R-W) controls above.
//  These are read only by runBipopCmaesV3RS; runBipopCmaesV3R never
//  touches them. Defaults give a sensible first run (lambda=0.3)
//  the moment Ctrl+Shift+G is pressed.
//
//   g_ctrlgsLambdaSil       : silhouette anchor strength (0 = no
//                             silhouette, falls back to V3R-W path).
//                             **Phase 2 default 4.0**: the cost is
//                             RMSE_W + lambda * (1-IoU) * |scale-1|,
//                             so the typical penalty magnitude is
//                             lambda * 0.4 * 0.05 = 0.02*lambda. To
//                             make this comparable to RMSE_W ~0.05,
//                             lambda ~= 4 is the right order. (The
//                             old Phase 1 default of 0.3 went with
//                             the unconditional (1-IoU) cost; both
//                             defaults match their cost shapes.)
//
// The silhouette loss itself is hardcoded to:
//   - Rasterize the full mesh through pre-built sil_view/sil_proj
//   - Sample dist_map at grid centres (binary mask, < 9000 = inside)
//   - Penalty = (1 - IoU2D) * |scale - 1.0|  (zero when either is OK)
// Earlier rim-based variants were structurally broken; see
// HANDOVER_V3RS_silhouette_pivot_to_2D_IoU.md.
// =========================================================
inline float g_ctrlgsLambdaSil          = 0.2f;

// =========================================================
//  [V3I / Ctrl+I] Pure-IoU mode flag.
//  When true, the V3RS engine optimizes cost = (1 - IoU2D)
//  ONLY (no RMSE_W / outside / rim_sil), the Run selector
//  decides by IoU2D, and the wrapper accept gate skips the
//  RMSE cap and accepts on IoU gain. This is the
//  "Ctrl+G mechanism, objective = squash-IoU" experiment.
//  runBipopCmaesV3I() flips this on for one call; Ctrl+Shift+G
//  leaves it false => byte-identical V3RS behaviour.
// =========================================================
inline bool g_ctrlgsPureIoUMode         = false;

// =========================================================
//  [NEW] Asymmetric outside-ratio penalty for Ctrl+Shift+G.
//  When ON, the inner cost adds lambda_out * outside_ratio,
//  where outside_ratio = (source raster cells outside target
//  mask) / (source raster cells). Directly penalises mask
//  expansion (source ⊃ target), which is symmetric IoU's
//  blind spot. The Layer 3 score in this file also includes
//  this term so accept/reject stays consistent across layers.
//
//   g_ctrlgsUseOutsideRatio : master toggle.
//   g_ctrlgsLambdaOut       : weight when toggle is ON.
// Default OFF → byte-identical to pre-feature behaviour.
// =========================================================
inline bool  g_ctrlgsUseOutsideRatio    = false;
inline float g_ctrlgsLambdaOut          = 0.5f;

// =========================================================
//  [NEW] RIM silhouette penalty for Ctrl+Shift+G.
//  When ON, the inner cost adds lambda_rim_sil * rim_sil_loss,
//  where rim_sil_loss is the mean normalised distance from each
//  SOURCE-BOUNDARY raster cell to the target silhouette
//  boundary. Cells outside the target mask contribute 1.0;
//  cells inside contribute min(d / rim_sil_max_px, 1.0).
//  The silhouette-space analogue of Ctrl+G's beta-rim weighting:
//  forces source rim ↔ target rim alignment in image space
//  rather than mere area overlap.
//
//   g_ctrlgsUseRimSil      : master toggle.
//   g_ctrlgsLambdaRimSil   : weight when toggle is ON.
//   g_ctrlgsRimSilMaxPx    : normalisation cap (image pixels).
// Default OFF → byte-identical to pre-feature behaviour.
// =========================================================
inline bool  g_ctrlgsUseRimSil          = false;
inline float g_ctrlgsLambdaRimSil       = 0.3f;
inline float g_ctrlgsRimSilMaxPx        = 100.0f;
// [NEW V3RS-RIM-ANAT] When ON, rim_sil_loss is computed over anatomical
// RIM vertices (LiverRegionLabel::RIM, filtered by quadrant + AR-vis +
// Caudal-only the same way the Ctrl+G "Show RIM pairs" checkbox filters
// them). Source rim = the orange spheres the user sees in the AR view
// when RimViz is on. When OFF, rim_sil falls back to the per-cell raster-
// boundary mode (every silhouette outline cell, including detached blobs).
inline bool  g_ctrlgsRimSilAnatomic     = false;

// =========================================================
//  [NEW] Dynamic RMSE acceptance cap for Ctrl+Shift+G Phase E.
//  The legacy fixed factor (1.05x) rejected solutions that
//  improved IoU but raised RMSE by more than 5% -- exactly the
//  "recover from mask expansion" move we WANT after Ctrl+G has
//  drifted. When the dynamic cap is ON, the cap interpolates
//  linearly between RmseCapBase (at diou=0) and RmseCapMax
//  (at diou >= RmseCapDiouFull), where
//     diou = iou_cand_occluded - init_iou_occluded.
//
//   g_ctrlgsUseDynamicCap   : when OFF, cap = RmseCapBase (fixed).
//                              When ON, interpolates to RmseCapMax.
//   g_ctrlgsRmseCapBase     : cap when no IoU improvement.
//   g_ctrlgsRmseCapMax      : cap at full IoU improvement.
//   g_ctrlgsRmseCapDiouFull : diou at which cap saturates.
// Default OFF + Base=1.05 → legacy 1.05 behaviour preserved.
// =========================================================
inline bool  g_ctrlgsUseDynamicCap      = false;
inline float g_ctrlgsRmseCapBase        = 1.05f;
inline float g_ctrlgsRmseCapMax         = 1.15f;
inline float g_ctrlgsRmseCapDiouFull    = 0.05f;

// =========================================================
//  Ctrl+Shift+G Instrument Occlusion Filter (NEW).
//  Controls whether IoU computation in rasterize_iou2d_v3rs
//  Step 3 excludes grid cells that lie under an instrument
//  (rasterized through g_instrumentDistMap). When the source
//  mesh projects onto an area covered by an instrument, the
//  SAM2 target mask has correctly NO liver there (occluded by
//  the tool), so any source overshoot in that area would be
//  spuriously counted as IoU loss. Excluding those cells from
//  BOTH numerator (intersection) and denominator (union) is
//  the correct fix.
//
//   g_ctrlgsIgnoreInstrument   : master toggle. Defaults OFF
//                                so pre-feature behaviour is
//                                preserved byte-for-byte. Enable
//                                via the Ctrl+Shift+G ImGui
//                                panel checkbox.
//   g_ctrlgsInstrumentThreshPx : pixel-distance threshold. Cells
//                                whose centre falls on an
//                                instrument pixel with
//                                inst_dist < thresh are excluded.
//                                  0  = exclude only INSIDE the
//                                       instrument region.
//                                  5  = also exclude within 5px
//                                       of the instrument boundary
//                                       (compensates for SAM2 mask
//                                       edge slop). Recommended.
//                                 10+ = aggressive; risk of dropping
//                                       legitimate silhouette near
//                                       the tool.
//
// Dependency: ensureInstrumentDistMap() must succeed (i.e.
// instrument_segmentation_mask.png must exist and match the
// liver-mask image dimensions). When that fails at session
// start, the wrapper falls back to OFF for that session and
// logs the reason. When that succeeds but the master toggle
// is OFF, the filter is still disabled (no behavioural change).
// =========================================================
inline bool  g_ctrlgsIgnoreInstrument    = false;
inline float g_ctrlgsInstrumentThreshPx  = 0.0f;

// =========================================================
//  Ctrl+Shift+G Phase 2 diagnostic: per-Run PNG dumps.
//  When g_silDumpPerRunEnabled is true, every Ctrl+Shift+G
//  press writes 10 sets of {hitmap.png, target_mask.png,
//  composite.png, boundary_map.png} to
//    <DEPTH_OUTPUT_PATH>/v3rs_dump/session_NNN/run_MM_*.png
//  using IoUDebug::dump. NNN auto-increments per session so
//  earlier dumps are preserved. Default OFF; UI toggles it.
//  Cost: ~50ms per Run = 500ms per session of PNG IO.
// =========================================================
inline bool         g_silDumpPerRunEnabled = false;
inline int          g_silDumpSessionCounter = 0;

// =========================================================
//  Ctrl+Shift+G silhouette debug visualization (case A).
//  Populated once per Ctrl+Shift+G press, in runBipopCmaesV3RS
//  wrapper AFTER the apply phase (so liverMesh3D is at the
//  optimised pose).  Each point stores its WORLD-SPACE 3D
//  position and the bilinear-sampled 2D distance (in original
//  image pixels) at its projection.  Rendering in main.cpp:
//  draw sphere markers at world_pos using g_sphereMarker; color
//  comes from dist_px (green=on boundary, red=far inside/outside).
//
//  Points are subsampled to ~600 for render perf.  Points whose
//  z-buffer projection lost (occluded or off-screen) are dropped.
//  Points with sentinel dist (>9000) are kept in storage but
//  ignored by mean_dist_px statistics and may be hidden at
//  render time depending on g_silProjShowSentinel.
// =========================================================
struct SilProjDebugPoint {
    glm::vec3 world_pos;   // 3D position (in the captured pose)
    float     dist_px;     // bilinear sample of g_boundaryDistMap at proj(u,v)
};
struct SilProjDebug {
    std::vector<SilProjDebugPoint> pts;
    bool  valid           = false;   // true after a successful capture
    int   img_w           = 0;
    int   img_h           = 0;
    int   n_visible       = 0;       // voxels that passed z-buffer
    int   n_with_signal   = 0;       // visible AND dist < sentinel
    float mean_dist_px    = 0.0f;    // averaged over points with signal
    float mean_dist_norm  = 0.0f;    // mean_dist_px / image_diagonal
};
inline SilProjDebug g_silProjDebug;
// [UI整理] Default OFF: the silhouette-projection debug overlay is opt-in.
inline bool         g_silProjShow = false;  // UI toggle

// [Phase A/F optimization] True when registrationHandle.{compRmse,compCount,compIoU}
// reflect the CURRENT mesh pose. Set true after any computeUnifiedMetrics() call;
// cleared to false by Phase A (consumed) and by external pose changes
// (InitRot Apply, Undo) via g_metricsValid = false in main.cpp.
inline bool g_metricsValid = false;

// Visualization buffers populated by runBipopCmaesV3R wrapper.
//   src: liverMesh3D vertex indices -> follows mesh through ICP/CMA-ES
//   tgt: target 3D positions (immutable, target never moves)
inline std::vector<int>       g_ctrlgRimSrcVertIdx;
inline std::vector<glm::vec3> g_ctrlgRimTgtPos;
inline bool                   g_ctrlgRimVizAvailable = false;

// =========================================================
//  Cyclic Boundary Registration (Shift+P) — 対応点可視化用ストレージ
// ---------------------------------------------------------
//  runCyclicBoundaryReg が best 確定後にここを populate する。
//  Shift+B (g_showCyclicCorrespondence) でセクターごとに HSV 着色した
//  source / target ペアを球マーカーで表示。
//  src は liverMesh3D の頂点 index 保存 → ICP で organ が動いても
//  常に最新位置で描画される。target は不変なので 3D 点直接保存。
//  i 番目の (src, tgt) は同じ HSV 色 → ペアであることが視認可能。
// =========================================================
inline std::vector<int>       g_cyclicPairSrcVertIdx;  // liverMesh3D 頂点 index (-1=空)
inline std::vector<glm::vec3> g_cyclicPairTgtPos;      // target 3D 位置 (不変)
inline std::vector<char>      g_cyclicPairValid;       // 各セクターのペア有効フラグ
inline int  g_cyclicSectors   = 24;
inline int  g_cyclicBestShift = 0;
inline int  g_cyclicBestRev   = 0;
inline bool g_cyclicAvailable = false;

// =========================================================
//  QuadCyclic-RANSAC (Shift+Ctrl+P) — Subset RANSAC ハイパラ
// ---------------------------------------------------------
//  K=3 minimum sample で 24 セクター medoid の中から 3 つを選んで
//  Umeyama (相似変換) → 内部 RMSE で top-K 候補絞り込み → Stage 2 で
//  full chamfer 評価 → ICP 精錬。v1 はハードコードだが globals 経由
//  にして将来 UI スライダで露出可能 (Cyclic Tuning panel 想定)。
// =========================================================
inline int   g_qcrSubsetK        = 5;     // K=3/4/5 (slider). 3=Fischler-Bolles min sample, 4-5=more stable
inline int   g_qcrMinSpreadSec   = 4;     // 採用 K sector の角度間隔下限 (=60°)
inline int   g_qcrTopKCandidates = 20;    // Stage 1 → Stage 2 で残す候補数
inline int   g_qcrMaxTrials      = 100000; // Stage 1 trial 数の上限 (超えたら等間隔サンプリング)

// =========================================================
//  [v1.2/v1.3/v1.4/v1.5] Init-pose prior + Inlier Refinement
// ---------------------------------------------------------
//  ユーザは Initial Orientation panel で「Up-L @ Q:AL」等の preset を
//  押してから Shift+Ctrl+P を実行する想定。この init pose は「ある
//  程度正しい」前提が成り立つので、RANSAC は「init pose から
//  大きく動く解」(例: 鏡像 mirror=1 reverse(CCW)、scale=2 等) よりも
//  「init pose から小さく補正する解」を優先すべき。
//
//  5 つのレイヤで実装:
//    (A1) Stage 1 hard limit (v1.3) — Umeyama 直後に各軸 (X/Y/Z) 回転を
//         チェックし、|angle| > g_qcrMaxAxisRotDeg の trial は
//         top-K 候補から完全除外。これにより鏡像解 (X 軸 ~180°等)
//         が top-K に入らなくなる。
//    (A2) Stage 2 hard limit (v1.2) — Stage 1 で生き残った top-K 候補も
//         念のため再チェック (Stage 1 のキャッシュ + 安全網)。
//    (B1) Stage 1 light penalty (v1.3) — `score_pair_total =
//         score_pair + λ_lite × disp_medoid` で top-K 選別。
//         medoid 22 点だけの安価な displacement なので Stage 1 で
//         init prior を効かせるのに最適。λ_lite = g_qcrStage1DispWeight。
//    (B2) Stage 2 full penalty (v1.2) — Stage 2 final score:
//           score_total = score_chamfer + λ × displacement_full
//         displacement_full = mean over silh∩quad points of ||T·p - p||
//         λ = g_qcrInitDispWeight。
//    (C)  Inlier Refinement (v1.4) — Stage 2 で best 候補を選んだ後、
//         「3 点 exact fit T」を「inlier consensus 上の over-determined
//         Umeyama T」に置き換える。これが Plan A の核心:
//           1. bestC.T を 24 sector medoid 全ペアに適用
//           2. residual <= max(median×2, score_pair×1.5) を inlier 判定
//           3. inlier ≥ 4 点なら Umeyama 再計算 → T_refined
//           4. T_refined が sane (rot/scale check) かつ
//              [v1.5] total (= chamfer + λ × disp) が improve したら採用。
//              v1.4 までは chamfer 単独 (≤ +5 %) で判定していたが、ログから
//              「chamfer 微改善・disp 大悪化」failure mode が観測されたため
//              v1.5 で Stage 2 と同じ total を採用基準に変更 (strict improve)。
//         3 点 → 15-22 点に増えるので Ctrl+P と同等の安定性 + ICP 収束性
//         が期待できる。RANSAC の本来の使い方。
//
//  λ=0 にすると displacement penalty 無効 = 純粋な data fit。
//  g_qcrMaxAxisRotDeg=180 にすると hard limit 無効 = どんな回転も許可。
//
//  v1.3 の動機: v1.2 ログで「top-20 全部が鏡像解 → 全 REJECT → fallback で
//  鏡像が選ばれる」という failure mode が発覚。Stage 1 自体に init prior を
//  入れないと「まともな候補が top-K に入らない」問題が解決しない。
//  v1.4 の動機: v1.3 でranking は正しくなったが「3 点 exact fit による
//  noise」で ICP が diverge して final RMSE が Ctrl+P を超えない。
//  Ctrl+P 相当の over-determined fit を RANSAC で選んだ inlier に対して
//  適用することで構造的限界を破る。
// =========================================================
inline float g_qcrStage1DispWeight = 0.3f;  // λ_lite (Stage 1, medoid disp)
inline float g_qcrInitDispWeight   = 0.5f;  // λ (Stage 2, full silh∩quad disp)
// --------------------------------------------------------------------
//  Rotation hard limits (post-Umeyama)
// --------------------------------------------------------------------
//  Umeyama / RANSAC subset fit が出す T が「init pose から見て」どこまで
//  回転を許容するかの上限。Apply Init Pose の 15 preset の最大回転量は
//  ±40° (FAR) なので、Umeyama drift をその範囲内に抑える設計。
//
//   - g_qcrMaxAxisRotDeg : 各軸 (X/Y/Z) Euler 角の |angle| の MAX 上限。
//     30° → 単軸成分はどれも 30° を超えない。
//   - g_qcrMaxTotalRotDeg: axis-angle 表現 (= rotation matrix の "回転角"
//     成分; arccos((trace(R)-1)/2)) で見た総回転量の上限。
//     per-axis では (89°,89°,89°) のように個別はクリアしても合計で
//     ~150° 回ってしまうケースがあるため、両方を AND で課す。
//
//  AutoQCR (= runAutoQuadCyclicRansac) は内部で runQuadCyclicRansac を
//  9 回呼ぶので、ここで設定した上限は AutoQCR の全 trial に効く。
//
//  Edit (チャット, 2026-05-21): デフォルトを 90°→30° に絞り、さらに
//  total-axis-angle ガードを追加。AutoQCR が「15 preset の範囲を大きく
//  超える」失敗ケースを抑えるための変更。
inline float g_qcrMaxAxisRotDeg    = 30.0f; // hard limit per X/Y/Z axis rotation (deg)
inline float g_qcrMaxTotalRotDeg   = 30.0f; // hard limit on total axis-angle rotation (deg)

// HSV -> RGB (h, s, v in [0, 1]); h は wrap される
inline glm::vec3 cyclicHsv2rgb(float h, float s, float v) {
    h = h - std::floor(h);
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    glm::vec3 rgb(0.0f);
    if      (h < 1.0f/6.0f) rgb = glm::vec3(c, x, 0);
    else if (h < 2.0f/6.0f) rgb = glm::vec3(x, c, 0);
    else if (h < 3.0f/6.0f) rgb = glm::vec3(0, c, x);
    else if (h < 4.0f/6.0f) rgb = glm::vec3(0, x, c);
    else if (h < 5.0f/6.0f) rgb = glm::vec3(x, 0, c);
    else                    rgb = glm::vec3(c, 0, x);
    return rgb + glm::vec3(m);
}

// =========================================================
//  Phase 1: Reproducibility — 決定論的シード固定機構
// ---------------------------------------------------------
//  論文の Reproducibility 節で「同じ trial_seed なら同じ結果」を主張する
//  ための仕組み。3 つの乱数源にオフセットで分配する。
//
//    g_trialSeed  : master seed (画像 × 条件ごとに変える)
//    g_callIdx    : trial 内の連番 (run* が呼ばれるたびに ++)
//
//    FGR tuple test  ← g_trialSeed + g_callIdx
//    BIPOP outer rng ← g_trialSeed + 1000u + g_callIdx * 97u
//    CMA-ES srand    ← g_trialSeed + 2000u + g_callIdx * 10u + run
//
//  オフセット (+1000, +2000) は 3 箇所同シードによる相関リスクを避けるため。
//  乗数 (97, 10) は call 間で乱数列が独立するように選んだ素数 / 小整数。
//
//  使い方:
//    resetTrialSeed(42);           // 新 trial 開始 (シード明示)
//    runHemiAuto();                 // → callIdx=0 で実行、++ されて 1
//    runBipopCmaes();               // → callIdx=1 で実行、++ されて 2
//    runShiftE();                   // → callIdx=2 で実行、++ されて 3
//
//    resetTrialSeed(42);            // 同じ trial 再現 → bit-identical な結果
//
//  AutoProbe は内部で g_callIdx をリセットして 0..N-1 で N 回 HemiAuto を
//  呼ぶので、各 probe が独立した FGR シードを使う (N は PoseLibrary.h の
//  runAutoProbe 内の定数で決まる、現在 N=108)。
// =========================================================
inline uint32_t g_trialSeed = 20260420u;  // master seed (デフォルト値)
inline uint32_t g_callIdx   = 0;           // trial 内の連番 (run* で auto-increment)

// =========================================================
//  AutoQCR (Alt+Ctrl+P) consume-and-clear globals
// ---------------------------------------------------------
//  Phase 1 観察用: runQuadCyclicRansac が末尾で publish する RANSAC prealign の
//  scale 値。AutoQCR の loop で各 trial 前に -1 (sentinel) にクリアし、
//  trial 後に値を読み取って TrialResult.prealignScale に記録する。
//  [0.85, 1.10] 外の trial は top-5 ログで [scale-bias?] と注記される。
//
//  [Opt A: AutoQCR fast metrics path] g_lastQcrChamfer:
//  single_mesh_only=true で runQuadCyclicRansac を呼んだとき (AutoQCR loop) は
//  computeUnifiedMetrics をスキップし、Stage 2 の bestScore (chamfer) を
//  ここに publish する。AutoQCR loop はこれを TrialResult.bestChamfer に
//  記録し、winner replay 後の Determinism check で「loop と replay の
//  chamfer が一致するか」を検証する (chamfer は pipeline 決定性の良い
//  invariant; computeUnifiedMetrics も決定的なので、chamfer 一致は全体の
//  決定性を担保する)。fast/full 両 path で publish される (-1 sentinel)。
// =========================================================
inline float g_lastQcrPrealignScale = -1.0f;
inline float g_lastQcrChamfer       = -1.0f;

// trial を新規開始 (master seed リセット + callIdx=0)
inline void resetTrialSeed(uint32_t s = 20260420u) {
    g_trialSeed = s;
    g_callIdx   = 0;
    std::cout << "[Seed] Trial reset: trial_seed=" << g_trialSeed << std::endl;
}

// =========================================================
//  Scene-scale-relative parameter ratios
// ---------------------------------------------------------
//  All registration parameters are derived from g_sceneDiag
//  (target AABB diagonal after prealignSourceToTarget).
//  Reference values were tuned and measured at sceneDiag ≈ 7.36
//  (target diagonal at g_objDisplayScale = 10, k4a 720p FOV).
//
//  NOTE on L = median NN distance (extern g_referenceL):
//  We initially attempted to switch voxel size to the standard
//  literature ratio voxel = 5L (Open3D / Zhou 2018), but found
//  it caused FGR's tuple test to fail (tuple_scale=0.95 is too
//  strict for our dense, low-noise depth-anything reconstructions).
//  The empirical voxel ≈ 0.30 at sceneDiag=7.36 corresponds to
//  voxel ≈ 27.6L, much coarser than literature defaults.
//  This will be defended in the paper via sensitivity analysis.
// =========================================================
extern float g_sceneDiag;
extern float g_referenceL;     // median NN distance — for diagnostics, not used yet

namespace RegRatios {
// Reference scene diag where the original constants were tuned.
// Empirically measured: target AABB diagonal at g_objDisplayScale=10
// with k4a 720p intrinsics.
constexpr float kRefSceneDiag    = 7.36f;

// Original (tuned) absolute values at kRefSceneDiag:
constexpr float kRefVoxel        = 0.30f;   // voxel ≈ 27.6 * L (much coarser than 5L literature default)
constexpr float kRefZThresh      = 0.015f;
constexpr float kRefMaxDist      = 1.00f;
constexpr float kRefCmaLocalT    = 0.50f;
constexpr float kRefCmaGlobalT   = 1.50f;
constexpr float kRefConvergence  = 0.005f;

// Visualization marker radii (in length units), tuned at kRefSceneDiag = 7.36.
// Without scaleRatio() they appear ~8x too large at sceneDiag ≈ 0.91 (metric).
constexpr float kRefMarkerCluster        = 0.08f;
constexpr float kRefMarkerTarget         = 0.12f;
constexpr float kRefMarkerCorrespondence = 0.30f;

inline float scaleRatio()  { return g_sceneDiag / kRefSceneDiag; }

inline float voxel()       { return g_voxelSize     * scaleRatio(); }
inline float zThresh()     { return std::max(0.001f, kRefZThresh * scaleRatio()); }
inline float maxDist()     { return kRefMaxDist     * scaleRatio(); }
inline float maxDistSq()   { float d = maxDist(); return d * d; }
inline float cmaLocalT()   { return kRefCmaLocalT   * scaleRatio(); }
inline float cmaGlobalT()  { return kRefCmaGlobalT  * scaleRatio(); }
inline float convergence() { return kRefConvergence * scaleRatio(); }

inline float markerCluster()        { return kRefMarkerCluster        * scaleRatio(); }
inline float markerTarget()         { return kRefMarkerTarget         * scaleRatio(); }
inline float markerCorrespondence() { return kRefMarkerCorrespondence * scaleRatio(); }
}

// =========================================================
//  ヘルパ
// =========================================================
inline std::vector<mCutMesh*> getOrganList() {
    return { liverMesh3D, portalMesh3D, veinMesh3D,
            tumorMesh3D, segmentMesh3D, gbMesh3D };
}

inline int gGridHeight() {
    if (screenMesh && screenMesh->loadedImageWidth > 0 && screenMesh->loadedImageHeight > 0)
        return gGridWidth * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
    return gGridWidth * 9 / 16;
}

// AR固定カメラ行列を構築（シルエットアライメント用）
// Kカメラ = RGB画像取得時のカメラ = 原点から+Z方向
inline glm::mat4 buildSilhouetteView() {
    return glm::lookAt(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));
}

// intrinsicsベースのprojection（キャリブレーション解像度で構築）
inline glm::mat4 buildSilhouetteProj() {
    int w = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
    int h = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

    glm::mat4 P(0.0f);
    P[0][0] =  2.0f * OrbitCam.fx / w;
    P[1][1] =  2.0f * OrbitCam.fy / h;
    P[2][0] =  1.0f - 2.0f * OrbitCam.cx / w;
    P[2][1] =  2.0f * OrbitCam.cy / h - 1.0f;
    P[2][2] = -(OrbitCam.farPlane + OrbitCam.nearPlane)
              / (OrbitCam.farPlane - OrbitCam.nearPlane);
    P[2][3] = -1.0f;
    P[3][2] = -2.0f * OrbitCam.farPlane * OrbitCam.nearPlane
              / (OrbitCam.farPlane - OrbitCam.nearPlane);
    return P;
}

// =========================================================
//  Unified Metrics（境界/内部色分け込み）
// =========================================================
inline void computeUnifiedMetrics() {
    Reg3DCustom::NoOpen3DRegistration reg;
    float zThresh = RegRatios::zThresh();

    std::streambuf*    oldBuf = nullptr;
    std::ostringstream devNull;
    if (g_quietMetrics) oldBuf = std::cout.rdbuf(devNull.rdbuf());
    auto targetCloud = reg.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (g_quietMetrics && oldBuf) std::cout.rdbuf(oldBuf);

    // ターゲットクラウドを境界/内部で色分け（Cluster ON / Boundary Candidates 用）
    if (targetCloud->hasBoundaryDist()) {
        g_targetPoints.clear();
        g_cluster2Points.clear();
        g_rejectedBoundaryPoints.clear();

        const bool hasInst = targetCloud->hasInstrumentDist();
        const float instPxTh = g_instrumentPxThresh;
        constexpr float kBoundaryPxTh = 12.0f;

        for (size_t i = 0; i < targetCloud->size(); i++) {
            float bd = targetCloud->boundaryDist[i];
            if (bd >= 9000.0f) continue;
            float idd = hasInst ? targetCloud->instrumentDist[i] : 9999.0f;

            const bool isBoundaryRaw  = (bd  < kBoundaryPxTh);
            const bool nearInstrument = (idd < instPxTh);

            if (isBoundaryRaw && !nearInstrument)
                g_targetPoints.push_back(targetCloud->points[i]);            // 採用境界（黄）
            else if (isBoundaryRaw && nearInstrument)
                g_rejectedBoundaryPoints.push_back(targetCloud->points[i]);  // 器具で棄却（赤）
            else
                g_cluster2Points.push_back(targetCloud->points[i]);          // 内部（青）
        }
        if (!g_quietMetrics) {
            std::cout << "[Boundary3D] accepted=" << g_targetPoints.size()
            << " rejected_by_instrument=" << g_rejectedBoundaryPoints.size()
            << " interior=" << g_cluster2Points.size()
            << "  (instPxTh=" << g_instrumentPxThresh
            << ", instMask=" << (hasInst ? "YES" : "NO")
            << ")" << std::endl;
        }
    }

    auto sourceCloud = std::make_shared<Reg3DCustom::PointCloud>();
    const auto& verts = liverMesh3D->mVertices;
    for (size_t i = 0; i + 2 < verts.size(); i += 3)
        sourceCloud->addPoint(glm::vec3(verts[i], verts[i+1], verts[i+2]));

    Reg3DCustom::NanoflannAdaptor adaptor(sourceCloud->points);
    auto  tree        = Reg3DCustom::buildKDTree(adaptor);
    float max_dist_sq = RegRatios::maxDistSq();

    float totalErr = 0.0f, sumSq = 0.0f, maxErr = 0.0f;
    int   count = 0;
    for (size_t i = 0; i < targetCloud->size(); i++) {
        size_t nn;  float d2;
        if (Reg3DCustom::searchKNN1(*tree, targetCloud->points[i], nn, d2)) {
            if (d2 < max_dist_sq) {
                float d = std::sqrt(d2);
                totalErr += d;
                sumSq    += d*d;
                if (d > maxErr) maxErr = d;
                count++;
            }
        }
    }
    float n = count ? (float)count : 1.0f;
    registrationHandle.compRmse     = std::sqrt(sumSq / n);
    registrationHandle.compAvgError = totalErr / n;
    registrationHandle.compMaxError = maxErr;
    registrationHandle.compCount    = count;

    if (!g_quietMetrics) {
        std::cout << "[Metrics] matched=" << count
                  << "  RMSE=" << registrationHandle.compRmse
                  << "  avg="  << registrationHandle.compAvgError
                  << "  max="  << registrationHandle.compMaxError << std::endl;
    }

    /* ----- 2D Hausdorff: シルエット境界の双方向最大距離 (画像 px 単位) -----
       (元の長いmain.cpp line 1564-1587 と同じ実装。
        Refine 関連削除以外は完全に忠実な移植。)
       g_boundaryDistMap が valid な時だけ計算 (depth マスクが必要)。

       ⚠️ 重要: g_boundaryDistMap は SAM2 マスクを AR カメラ (intrinsics ベースの
       固定カメラ、原点から +Z 方向) で投影した結果。したがって IoU は必ず
       同じ AR カメラから計測しないと意味がない (OrbitCam を動かすたびに値が
       変わってしまうため)。runShiftE の measureIoU ラムダと同じく、計算前に
       グローバル view/projection/gWindowWidth/gWindowHeight を一時差し替え。
       computeSilhouette2DObjectiveFast の内部はこれらのグローバルを参照する
       ため、引数で view/projection を渡してもグローバル差し替えが必要。 */
    if (g_boundaryDistMap.valid && liverMesh3D) {
        // AR 固定カメラ行列を構築 (runShiftE と同じ)
        glm::mat4 silView = buildSilhouetteView();
        glm::mat4 silProj = buildSilhouetteProj();
        int silW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
        int silH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

        // グローバル一時差し替え
        glm::mat4 sv = view;          glm::mat4 sp = projection;
        int       sw = gWindowWidth;  int       sh = gWindowHeight;
        view = silView;       projection = silProj;
        gWindowWidth = silW;  gWindowHeight = silH;

        float h2d = 0.0f;
        int dInter = 0, dUnion = 0;
        double dMs = 0.0;
        bool wasQuiet = g_quietMetrics;
        g_quietMetrics = true;  /* IoU の Sil2D-Fast ログを抑制 (二重出力防止) */
        CmaesRefine::computeSilhouette2DObjectiveFast(
            liverMesh3D, view, projection, /*step=*/8,
            &dInter, &dUnion, &dMs, &h2d);
        g_quietMetrics = wasQuiet;

        // グローバル復元
        view = sv;             projection = sp;
        gWindowWidth = sw;     gWindowHeight = sh;

        registrationHandle.sil2DHausdorff = h2d;
        registrationHandle.compIoU2D = (dUnion > 0) ? (float)dInter / (float)dUnion : 0.0f;

        if (!g_quietMetrics) {
            std::cout << "[Hausdorff2D] IoU=" << registrationHandle.compIoU2D
                      << "  H2D=" << std::fixed << std::setprecision(1)
                      << h2d << "px"
                      << "  (cost=" << std::setprecision(2) << dMs << "ms, step=8)"
                      << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }
    } else {
        /* boundaryDistMap が未 build の時は IoU を 0 にしておく
           (poseSaveToLibrary の IoU 比較は currentIoU > 0 で守る) */
        registrationHandle.compIoU2D = 0.0f;
    }
}

// =========================================================
//  HemiAuto Registration (Key O)
// =========================================================
inline void runHemiAuto() {
    std::cout << "\n=== HemiAuto Registration (Key O) ===" << std::endl;

    // Phase 1: FGR tuple test シード固定
    const uint32_t fgr_seed = g_trialSeed + g_callIdx;
    Reg3DCustom::setFgrSeed(fgr_seed);
    std::cout << "[Seed] FGR=" << fgr_seed
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;

    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
    if (vis.cloud->size() < 50) {
        std::cerr << "[O] Not enough visible points ("
                  << vis.cloud->size() << ")" << std::endl;
        g_callIdx++;  // 失敗時もインクリメント (シード列を進める)
        return;
    }

    g_cluster1Points      = vis.points;
    g_cluster2Points.clear();

    auto organs = getOrganList();
    Reg3DCustom::performRegistrationSingleMesh(
        organs, liverMesh3D, vis.vertexIndices,
        screenMesh, OrbitCam.cameraPos,
        gGridWidth, gGridHeight(),
        1,
        RegRatios::convergence(),    // convergence_threshold (was 0.005)
        0.35f,                        // min_fitness — ratio, scale-invariant
        true, 0.03f,                  // min_scale_threshold — ratio, scale-invariant
        RegRatios::zThresh(),        // zThresh (was gDepthScale)
        RegRatios::voxel());         // voxel_size (was g_voxelSize)

    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== HemiAuto Complete  RMSE=" << registrationHandle.compRmse
              << " ===" << std::endl;

    g_callIdx++;  // Phase 1: 末尾でインクリメント
}

// =========================================================
//  QuadAuto Registration (Shift+O)
// ---------------------------------------------------------
//  HemiAuto と同じ FGR+ICP パイプラインを使うが、source 頂点の
//  絞り込みフィルタを以下の二段で構成する:
//
//    filter_AR    : AR 固定視点 cameraPos=(0,0,0), target=(0,0,1)
//                   から extractVisibleVerticesCustom() で得た
//                   可視頂点 (raycast 遮蔽除去 + dot(rayDir,viewDir)>0.3)
//    filter_Quad  : LiverLeftRightLabel::makeQuadrantSubsetIdx() で
//                   g_activeQuadrantMask に属する解剖象限頂点
//                   (Initial Orientation panel / Ctrl+G と共有)
//
//    source_final = filter_AR ∩ filter_Quad   ← FGR/ICP の source 入力
//
//  動機:
//    - Key O (HemiAuto) は OrbitCam の現在マウス位置に依存するため
//      AutoProbe 等で再現性確保のために OrbitCam を swap する必要が
//      あった。QuadAuto は最初から AR 固定視点を直接渡すので
//      OrbitCam に副作用なし、いつでも同じ視点で評価可能。
//    - 同時に Initial Orientation panel の quadrant 選択と意味的に
//      一貫する: 「右葉に置いた」のなら右葉だけで合わせる。
//    - QUAD_ALL (= 0x0F) のときは filter_Quad が全頂点なので、
//      結果は「AR 固定視点 hemi」になる。Key O (現 OrbitCam 視点 hemi)
//      とは視点が違う独立の動作。
//
//  ラベル未計算ガード:
//    g_liverRegion / g_liverLR が未計算なら abort し、明示エラー出力。
//    呼び出し元 (main.cpp の Shift+O dispatch / a.onQuadAuto lambda)
//    で recomputeLiver* を auto-trigger する想定 (applyInitRotation と
//    同じ流儀)。RegistrationActions.h からは main.cpp の recompute*
//    関数が見えないので、ここではガードに留める。
//
//  target / FGR/ICP パラメータは Key O と完全同一 (RegRatios::*)。
//  performRegistrationSingleMesh の camera_position 引数は dead
//  parameter (PoseLibrary.h §AutoProbe コメント参照) なので arPos を
//  そのまま渡す。
// =========================================================
inline void runQuadAuto() {
    std::cout << "\n=== QuadAuto Registration (Shift+O) ===" << std::endl;

    // --- 0. Seed (HemiAuto と完全同形: 比較しやすさのため) ---
    const uint32_t fgr_seed = g_trialSeed + g_callIdx;
    Reg3DCustom::setFgrSeed(fgr_seed);
    std::cout << "[Seed] FGR=" << fgr_seed
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;

    // --- 1. Quadrant label validity guard ---
    //         呼び出し元で recomputeLiver{Region,LR}() を auto-trigger
    //         している想定だが、念のためここでも防御。
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[Shift+O] Region/LR labels not computed: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << "  LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << ". Press R/Y to compute, or use Initial Orientation panel."
                  << std::endl;
        g_callIdx++;
        return;
    }

    // --- 2. Quadrant subset (anatomy-based source filter) ---
    const uint8_t mask = g_activeQuadrantMask;
    std::vector<int> quadIdx = LiverLeftRightLabel::makeQuadrantSubsetIdx(
        g_liverRegion.labels, g_liverLR.labels, mask);
    std::cout << "[Shift+O] quadrant_mask = "
              << LiverLeftRightLabel::quadrantMaskString(mask)
              << "  (0x" << std::hex << (unsigned)mask << std::dec << ")"
              << "  subset_size=" << quadIdx.size() << "/"
              << g_liverRegion.labels.size() << std::endl;
    if (quadIdx.empty()) {
        std::cerr << "[Shift+O] Empty quadrant subset (mask=0x"
                  << std::hex << (unsigned)mask << std::dec
                  << "). Select at least one quadrant in Initial Orientation"
                  << " panel (or set QUAD_ALL = 0x0F)." << std::endl;
        g_callIdx++;
        return;
    }

    // --- 3. AR-fixed-camera visibility (view-independent reproducibility) ---
    //         OrbitCam には触らない -- 引数として直接 AR 固定値を渡す。
    //         AutoProbe では OrbitCam を swap する必要があったが、
    //         QuadAuto は extractVisibleVerticesCustom に直接渡すだけで
    //         十分。
    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    const glm::vec3 arPos(0.0f, 0.0f, 0.0f);
    const glm::vec3 arTgt(0.0f, 0.0f, 1.0f);
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, arPos, arTgt);
    if (vis.cloud->size() < 50) {
        std::cerr << "[Shift+O] Not enough AR-visible points ("
                  << vis.cloud->size() << "). Liver may be off-screen "
                     "from the AR camera." << std::endl;
        g_callIdx++;
        return;
    }

    // --- 4. Intersection: filter_AR ∩ filter_Quad ---
    //         quadIdx を hash 化 → vis.vertexIndices を線形走査。
    //         vis.vertexIndices と vis.points は parallel array なので
    //         同じ k で照合。
    std::unordered_set<int> quadSet(quadIdx.begin(), quadIdx.end());
    std::vector<size_t>     finalIdx;
    std::vector<glm::vec3>  finalPoints;
    finalIdx.reserve(std::min(vis.vertexIndices.size(), quadIdx.size()));
    finalPoints.reserve(finalIdx.capacity());
    for (size_t k = 0; k < vis.vertexIndices.size(); k++) {
        const int vi = static_cast<int>(vis.vertexIndices[k]);
        if (quadSet.count(vi)) {
            finalIdx.push_back(vis.vertexIndices[k]);
            finalPoints.push_back(vis.points[k]);
        }
    }
    std::cout << "[Shift+O] AR-visible=" << vis.vertexIndices.size()
              << "  quad=" << quadIdx.size()
              << "  intersection=" << finalIdx.size() << std::endl;
    if (finalIdx.size() < 50) {
        std::cerr << "[Shift+O] Intersection too small ("
                  << finalIdx.size() << "). The selected quadrant may "
                     "be facing away from the AR camera, or try a "
                     "different mask." << std::endl;
        g_callIdx++;
        return;
    }

    // --- 5. Debug viz: cluster1 = 交差後の source 頂点 (Key O 同形) ---
    g_cluster1Points = finalPoints;
    g_cluster2Points.clear();

    // --- 6. Run existing FGR+ICP pipeline (Key O と完全同一パラメータ) ---
    //         arPos は dead parameter (camera_position 引数) として渡す。
    auto organs = getOrganList();
    Reg3DCustom::performRegistrationSingleMesh(
        organs, liverMesh3D, finalIdx,
        screenMesh, arPos,
        gGridWidth, gGridHeight(),
        1,
        RegRatios::convergence(),    // convergence_threshold
        0.35f,                        // min_fitness
        true, 0.03f,                  // estimate_scale, min_scale_threshold
        RegRatios::zThresh(),        // zThresh
        RegRatios::voxel());         // voxel_size

    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== QuadAuto Complete  RMSE=" << registrationHandle.compRmse
              << "  mask=" << LiverLeftRightLabel::quadrantMaskString(mask)
              << " ===" << std::endl;

    g_callIdx++;  // HemiAuto と同じく末尾でインクリメント
}

// =========================================================
//  SilhouetteHemi Registration (Key P)
// ---------------------------------------------------------
//  HemiAuto と同じパイプラインだが、登録に使う点群を以下の通り絞り込む:
//
//     Source side:
//        extractVisibleVerticesCustom で得た可視頂点のうち、
//        |dot(normal, viewDir)| < g_silhouetteSrcCosThresh のものだけ
//        (シルエット近傍の頂点のみ)
//
//     Target side:
//        boundaryDist < 12 かつ instrumentDist >= g_instrumentPxThresh
//        のものだけ (採用境界点のみ)
//
//  両者がシルエット/境界に絞り込まれることで、FPFH 特徴量の識別性が
//  上がり、FGR の tuple test が通る確率が高まる、という仮説を試す。
//
//  既存の runHemiAuto には一切手を加えない。target は計算中だけ
//  キャッシュを差し替え、終了時に元に戻すので副作用なし。
// =========================================================
inline void runSilhouetteHemi() {
    std::cout << "\n=== SilhouetteHemi Registration (Key P) ===" << std::endl;

    // Phase 1: FGR tuple test シード固定 (HemiAuto と同じ仕組み)
    const uint32_t fgr_seed = g_trialSeed + g_callIdx;
    Reg3DCustom::setFgrSeed(fgr_seed);
    std::cout << "[Seed] FGR=" << fgr_seed
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;

    // ---- 0. Source mesh の法線を必要なら計算（loadMeshFromFile は法線を読まないため） ----
    //         一度計算すれば performRegistrationSingleMesh が transform 時に
    //         同じ rotation で更新してくれるので、複数回 P を押しても OK。
    if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
        std::cout << "[P] liverMesh3D normals missing -- computing from faces..." << std::endl;
        Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
    }

    // ---- 1. Source 側: 全可視頂点を取得 ----
    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
    if (vis.cloud->size() < 50) {
        std::cerr << "[P] Not enough visible points ("
                  << vis.cloud->size() << ")" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 2. Source 側: シルエット頂点に絞り込み ----
    glm::vec3 viewDir = glm::normalize(OrbitCam.cameraTarget - OrbitCam.cameraPos);
    g_visibleSourcePoints    = vis.points;       // for visualization (cyan)
    g_silhouetteSourcePoints.clear();

    std::vector<size_t> silhouetteIndices;
    silhouetteIndices.reserve(vis.cloud->size());
    const bool hasN = vis.cloud->hasNormals();
    if (!hasN) {
        std::cerr << "[P] visible cloud has no normals -- falling back to all visible"
                  << std::endl;
        silhouetteIndices = vis.vertexIndices;
        g_silhouetteSourcePoints = vis.points;
    } else {
        for (size_t k = 0; k < vis.cloud->size(); k++) {
            const glm::vec3& n = vis.cloud->normals[k];
            float s = std::abs(glm::dot(n, viewDir));
            if (s < g_silhouetteSrcCosThresh) {
                silhouetteIndices.push_back(vis.vertexIndices[k]);
                g_silhouetteSourcePoints.push_back(vis.cloud->points[k]);
            }
        }
        std::cout << "[Silhouette/Source] visible=" << vis.cloud->size()
                  << "  silhouette=" << silhouetteIndices.size()
                  << "  cosThresh=" << g_silhouetteSrcCosThresh << std::endl;
        if (silhouetteIndices.size() < 50) {
            std::cerr << "[P] Too few silhouette vertices ("
                      << silhouetteIndices.size()
                      << ") -- falling back to all visible" << std::endl;
            silhouetteIndices = vis.vertexIndices;
            g_silhouetteSourcePoints = vis.points;
        }
    }

    g_cluster1Points = g_silhouetteSourcePoints;   // 既存 Cluster 表示にも反映
    g_cluster2Points.clear();

    // ---- 3. Target 側: 採用境界点だけからなるフィルタ済みクラウドを作る ----
    auto savedFullTarget = Reg3DCustom::getCachedTargetCloud();
    if (!savedFullTarget || savedFullTarget->empty()) {
        std::cerr << "[P] No cached target cloud" << std::endl;
        g_callIdx++;
        return;
    }
    auto filteredTarget = std::make_shared<Reg3DCustom::PointCloud>();
    constexpr float kBoundaryPxTh = 12.0f;
    const float instTh    = g_instrumentPxThresh;
    const bool  hasInst   = savedFullTarget->hasInstrumentDist();
    const bool  copyN     = savedFullTarget->hasNormals();
    const bool  copyBd    = savedFullTarget->hasBoundaryDist();

    filteredTarget->points.reserve(savedFullTarget->size() / 4);
    if (copyN)   filteredTarget->normals.reserve(savedFullTarget->size() / 4);
    if (copyBd)  filteredTarget->boundaryDist.reserve(savedFullTarget->size() / 4);
    if (hasInst) filteredTarget->instrumentDist.reserve(savedFullTarget->size() / 4);

    for (size_t i = 0; i < savedFullTarget->size(); i++) {
        float bd  = savedFullTarget->boundaryDist[i];
        float idd = hasInst ? savedFullTarget->instrumentDist[i] : 9999.0f;
        if (bd < kBoundaryPxTh && idd >= instTh) {
            filteredTarget->points.push_back(savedFullTarget->points[i]);
            if (copyN)   filteredTarget->normals.push_back(savedFullTarget->normals[i]);
            if (copyBd)  filteredTarget->boundaryDist.push_back(bd);
            if (hasInst) filteredTarget->instrumentDist.push_back(idd);
        }
    }
    filteredTarget->colors.assign(filteredTarget->points.size(),
                                  glm::vec3(0.0f, 1.0f, 0.0f));

    std::cout << "[Silhouette/Target] full=" << savedFullTarget->size()
              << "  boundary_kept=" << filteredTarget->size()
              << "  (boundaryPx<" << kBoundaryPxTh
              << ", instPx>=" << instTh << ")" << std::endl;

    if (filteredTarget->size() < 50) {
        std::cerr << "[P] Too few boundary points in target (" << filteredTarget->size()
        << ") -- aborting (try lowering instrumentPxThresh)" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 4. キャッシュをフィルタ済み target に一時的に差し替え ----
    Reg3DCustom::setCachedTargetCloud(filteredTarget);

    // ---- 5. 既存の registration パイプラインを呼ぶ ----
    auto organs = getOrganList();
    bool ok = true;
    try {
        Reg3DCustom::performRegistrationSingleMesh(
            organs, liverMesh3D, silhouetteIndices,
            screenMesh, OrbitCam.cameraPos,
            gGridWidth, gGridHeight(),
            1,
            RegRatios::convergence(),
            0.35f,
            true, 0.03f,
            RegRatios::zThresh(),
            RegRatios::voxel());
    } catch (const std::exception& e) {
        std::cerr << "[P] performRegistrationSingleMesh threw: " << e.what() << std::endl;
        ok = false;
    }

    // ---- 6. 必ず元の target を復元 (例外時も保証) ----
    Reg3DCustom::setCachedTargetCloud(savedFullTarget);

    if (!ok) { g_callIdx++; return; }

    // ---- 7. メトリクスは元のフル target でやる (HemiAuto と同じ評価軸を保つ) ----
    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== SilhouetteHemi Complete  RMSE=" << registrationHandle.compRmse
              << " ===" << std::endl;

    g_callIdx++;
}

// =========================================================
//  Cyclic Boundary Registration (Shift+P)
// ---------------------------------------------------------
//  Key P (SilhouetteHemi) と同じ前処理を使うが、FPFH/FGR の代わりに
//  「重心を中心に時計回り N セクター分割 → 巡回シフト × ミラー反転で
//   2N パターンの対応点候補を試行 → 全 source × 全 target の
//   chamfer RMSE で最良 T を選ぶ」 という古典的 cyclic Procrustes
//   approach で初期 T を作る。その後 performRegistrationSingleMesh
//   で ICP 精錬する。
//
//  動機: silhouette/boundary の点数が少ないと FPFH の識別性が出ず、
//        FGR が tuple test 0 corres → identity に落ちる。境界形状の
//        トポロジーを直接利用すれば特徴量を介さず初期姿勢が出せる。
//
//  挙動 (user 指定):
//    - 重心を中心に 2D 平面で N 分割し時計回りにセクター index を付与。
//    - target にそのセクターの境界点が無ければ、source 側の同セクター
//      の対応点も無視 (= ペアから除外)。
//
//  パラメータ (v1 ハードコード):
//    kSectors        = 24    セクター数 (=15° 刻み)
//    kMinValidPairs  = 4     Umeyama に必要な最小対応点数
//    kScale Lo/Hi    = 0.5/2 許容スケール範囲 (sanity check)
//    試行数          = 2N    (forward N + reverse N: ミラー対応)
// =========================================================
inline void runCyclicBoundaryReg() {
    std::cout << "\n=== Cyclic Boundary Registration (Shift+P) ===" << std::endl;

    // Phase 1: シード固定 (Key P と同じ仕組み)
    const uint32_t fgr_seed = g_trialSeed + g_callIdx;
    Reg3DCustom::setFgrSeed(fgr_seed);
    std::cout << "[Seed] FGR=" << fgr_seed
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;

    // ---- 0. Source mesh の法線確認 ----
    if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
        std::cout << "[Shift+P] liverMesh3D normals missing -- computing from faces..."
                  << std::endl;
        Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
    }

    // ---- 1. Source: 全可視頂点を取得 ----
    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
    if (vis.cloud->size() < 50) {
        std::cerr << "[Shift+P] Not enough visible points ("
                  << vis.cloud->size() << ")" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 2. Source: silhouette filtering (Key P と同じ) ----
    glm::vec3 viewDir = glm::normalize(OrbitCam.cameraTarget - OrbitCam.cameraPos);
    g_visibleSourcePoints = vis.points;
    g_silhouetteSourcePoints.clear();

    std::vector<glm::vec3> sourceSilhPts;
    std::vector<size_t>    silhouetteIndices;
    sourceSilhPts.reserve(vis.cloud->size());
    silhouetteIndices.reserve(vis.cloud->size());

    const bool hasN = vis.cloud->hasNormals();
    if (!hasN) {
        std::cerr << "[Shift+P] visible cloud has no normals -- using all visible"
                  << std::endl;
        sourceSilhPts     = vis.points;
        silhouetteIndices = vis.vertexIndices;
    } else {
        for (size_t k = 0; k < vis.cloud->size(); k++) {
            const glm::vec3& n = vis.cloud->normals[k];
            float s = std::abs(glm::dot(n, viewDir));
            if (s < g_silhouetteSrcCosThresh) {
                sourceSilhPts.push_back(vis.cloud->points[k]);
                silhouetteIndices.push_back(vis.vertexIndices[k]);
            }
        }
        if (sourceSilhPts.size() < 50) {
            std::cerr << "[Shift+P] Too few silhouette vertices ("
                      << sourceSilhPts.size()
                      << ") -- falling back to all visible" << std::endl;
            sourceSilhPts     = vis.points;
            silhouetteIndices = vis.vertexIndices;
        }
    }
    g_silhouetteSourcePoints = sourceSilhPts;
    g_cluster1Points         = sourceSilhPts;
    g_cluster2Points.clear();

    std::cout << "[Shift+P/Source] visible=" << vis.cloud->size()
              << "  silhouette=" << sourceSilhPts.size()
              << "  cosThresh=" << g_silhouetteSrcCosThresh << std::endl;

    // ---- 3. Target: boundary filtering (Key P と同じ) ----
    auto savedFullTarget = Reg3DCustom::getCachedTargetCloud();
    if (!savedFullTarget || savedFullTarget->empty()) {
        std::cerr << "[Shift+P] No cached target cloud" << std::endl;
        g_callIdx++;
        return;
    }

    constexpr float kBoundaryPxTh = 12.0f;
    const float instTh   = g_instrumentPxThresh;
    const bool  hasInst  = savedFullTarget->hasInstrumentDist();
    const bool  hasBdy   = savedFullTarget->hasBoundaryDist();

    if (!hasBdy) {
        std::cerr << "[Shift+P] target cloud has no boundaryDist -- aborting"
                  << std::endl;
        g_callIdx++;
        return;
    }

    std::vector<glm::vec3> targetBdyPts;
    targetBdyPts.reserve(savedFullTarget->size() / 4);
    for (size_t i = 0; i < savedFullTarget->size(); i++) {
        float bd  = savedFullTarget->boundaryDist[i];
        float idd = hasInst ? savedFullTarget->instrumentDist[i] : 9999.0f;
        if (bd < kBoundaryPxTh && idd >= instTh) {
            targetBdyPts.push_back(savedFullTarget->points[i]);
        }
    }
    std::cout << "[Shift+P/Target] full=" << savedFullTarget->size()
              << "  boundary_kept=" << targetBdyPts.size()
              << "  (boundaryPx<" << kBoundaryPxTh
              << ", instPx>=" << instTh << ")" << std::endl;

    if (targetBdyPts.size() < 50) {
        std::cerr << "[Shift+P] Too few boundary points in target ("
                  << targetBdyPts.size() << ") -- aborting" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 4. View plane を構築 (camera viewDir に直交する 2D 基底) ----
    //         vRight, vUp は projection 平面内の正規直交基底。
    //         向きは任意だが source/target で同じ基底を使えば良い。
    glm::vec3 vUp_world = (std::abs(viewDir.y) > 0.9f)
                              ? glm::vec3(1.0f, 0.0f, 0.0f)
                              : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 vRight = glm::normalize(glm::cross(vUp_world, viewDir));
    glm::vec3 vUp    = glm::normalize(glm::cross(viewDir, vRight));

    auto project2D = [&](const glm::vec3& p) -> glm::vec2 {
        glm::vec3 d = p - OrbitCam.cameraPos;
        return glm::vec2(glm::dot(d, vRight), glm::dot(d, vUp));
    };

    // ---- 5. 2D 重心を source / target でそれぞれ計算 ----
    glm::vec2 srcCentroid2D(0.0f), tgtCentroid2D(0.0f);
    for (const auto& p : sourceSilhPts) srcCentroid2D += project2D(p);
    for (const auto& p : targetBdyPts)  tgtCentroid2D += project2D(p);
    srcCentroid2D /= float(sourceSilhPts.size());
    tgtCentroid2D /= float(targetBdyPts.size());

    // ---- 6. N セクター分割 (重心からの角度で時計回り) ----
    constexpr int   kSectors       = 24;
    constexpr int   kMinValidPairs = 4;
    constexpr float kScaleLo       = 0.5f;
    constexpr float kScaleHi       = 2.0f;
    const float kTwoPi = 6.28318530717958f;

    auto assignSector = [&](const glm::vec2& p2d, const glm::vec2& center) -> int {
        glm::vec2 d = p2d - center;
        // 時計回り正方向: -atan2(y, x) を採用。
        // [0, 2π) に正規化してセクター index 化。
        float theta = -std::atan2(d.y, d.x);
        if (theta < 0.0f) theta += kTwoPi;
        int idx = int(theta / (kTwoPi / float(kSectors)));
        if (idx >= kSectors) idx = kSectors - 1;
        if (idx < 0)         idx = 0;
        return idx;
    };

    std::vector<std::vector<size_t>> srcSectorIdx(kSectors);
    std::vector<std::vector<size_t>> tgtSectorIdx(kSectors);

    for (size_t i = 0; i < sourceSilhPts.size(); i++) {
        int s = assignSector(project2D(sourceSilhPts[i]), srcCentroid2D);
        srcSectorIdx[s].push_back(i);
    }
    for (size_t i = 0; i < targetBdyPts.size(); i++) {
        int s = assignSector(project2D(targetBdyPts[i]), tgtCentroid2D);
        tgtSectorIdx[s].push_back(i);
    }

    // 各セクターの medoid (3D 点) を選ぶ
    auto medoidOfSector = [](const std::vector<glm::vec3>& pts,
                             const std::vector<size_t>& idxList) -> int {
        if (idxList.empty()) return -1;
        glm::vec3 mean(0.0f);
        for (size_t i : idxList) mean += pts[i];
        mean /= float(idxList.size());
        int   best  = (int)idxList[0];
        float bestD = glm::distance(pts[best], mean);
        for (size_t i : idxList) {
            float d = glm::distance(pts[i], mean);
            if (d < bestD) { bestD = d; best = (int)i; }
        }
        return best;
    };

    std::vector<int> srcMedoid(kSectors, -1);
    std::vector<int> tgtMedoid(kSectors, -1);
    int nSrcOcc = 0, nTgtOcc = 0;
    for (int k = 0; k < kSectors; k++) {
        srcMedoid[k] = medoidOfSector(sourceSilhPts, srcSectorIdx[k]);
        tgtMedoid[k] = medoidOfSector(targetBdyPts,  tgtSectorIdx[k]);
        if (srcMedoid[k] >= 0) nSrcOcc++;
        if (tgtMedoid[k] >= 0) nTgtOcc++;
    }
    std::cout << "[Sectors] N=" << kSectors
              << "  src_occupied=" << nSrcOcc << "/" << kSectors
              << "  tgt_occupied=" << nTgtOcc << "/" << kSectors << std::endl;

    if (nSrcOcc < kMinValidPairs || nTgtOcc < kMinValidPairs) {
        std::cerr << "[Shift+P] Too few occupied sectors -- aborting" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 7. Chamfer 評価用 KD tree (target 側) ----
    Reg3DCustom::NanoflannAdaptor tgtAdaptor(targetBdyPts);
    auto tgtTree = Reg3DCustom::buildKDTree(tgtAdaptor);

    auto chamferRMSE = [&](const glm::mat4& T) -> float {
        float sumSq = 0.0f;
        int   cnt   = 0;
        for (const auto& p : sourceSilhPts) {
            glm::vec4 v(p, 1.0f);
            glm::vec3 tp = glm::vec3(T * v);
            size_t nn; float d2;
            if (Reg3DCustom::searchKNN1(*tgtTree, tp, nn, d2)) {
                sumSq += d2;
                cnt++;
            }
        }
        if (cnt == 0) return 1e9f;
        return std::sqrt(sumSq / float(cnt));
    };

    // ---- 8. 巡回シフト × ミラー反転 = 2N パターンを試行 ----
    glm::mat4 bestT(1.0f);
    float bestScore = 1e9f;
    int   bestShift = 0, bestRev = 0;
    int   triedCount = 0, validCount = 0;

    for (int rev = 0; rev < 2; rev++) {
        for (int shift = 0; shift < kSectors; shift++) {
            std::vector<glm::vec3> srcPairs, tgtPairs;
            srcPairs.reserve(kSectors);
            tgtPairs.reserve(kSectors);

            for (int i = 0; i < kSectors; i++) {
                int j;
                if (rev == 0) {
                    j = (i + shift) % kSectors;                          // forward
                } else {
                    j = ((kSectors - 1 - i) + shift) % kSectors;         // reverse
                    if (j < 0) j += kSectors;
                }
                // ターゲットが空なら、対応するソース対応点も無視 (user 指定の挙動)
                if (srcMedoid[i] < 0 || tgtMedoid[j] < 0) continue;
                srcPairs.push_back(sourceSilhPts[srcMedoid[i]]);
                tgtPairs.push_back(targetBdyPts [tgtMedoid[j]]);
            }
            triedCount++;
            if ((int)srcPairs.size() < kMinValidPairs) continue;

            glm::mat4 T = Reg3D::UmeyamaRegistration(srcPairs, tgtPairs);

            // Sanity: 行列要素の有限性
            bool finiteT = true;
            for (int c = 0; c < 4 && finiteT; c++)
                for (int r = 0; r < 4 && finiteT; r++)
                    if (!std::isfinite(T[c][r])) finiteT = false;
            if (!finiteT) continue;

            // Sanity: scale が極端なら捨てる
            float scale = glm::length(glm::vec3(T[0]));
            if (!std::isfinite(scale) || scale < kScaleLo || scale > kScaleHi) continue;

            float score = chamferRMSE(T);
            validCount++;
            if (score < bestScore) {
                bestScore = score;
                bestT     = T;
                bestShift = shift;
                bestRev   = rev;
            }
        }
    }

    std::cout << "[Cyclic] tried=" << triedCount
              << "  valid=" << validCount
              << "  best_shift=" << bestShift
              << "  best_dir=" << (bestRev ? "reverse(CCW)" : "forward(CW)")
              << "  best_chamfer_rmse=" << bestScore << std::endl;

    if (validCount == 0) {
        std::cerr << "[Shift+P] No valid cyclic match found -- aborting" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 8.5. 対応点を Shift+B 可視化用に保存 ----
    //          src は liverMesh3D 頂点 index で保存 (ICP 後も最新位置を反映)。
    //          target は不変なので 3D 点をそのまま保存。
    g_cyclicSectors   = kSectors;
    g_cyclicBestShift = bestShift;
    g_cyclicBestRev   = bestRev;
    g_cyclicPairSrcVertIdx.assign(kSectors, -1);
    g_cyclicPairTgtPos.assign(kSectors, glm::vec3(0.0f));
    g_cyclicPairValid.assign(kSectors, 0);
    for (int i = 0; i < kSectors; i++) {
        int j;
        if (bestRev == 0) {
            j = (i + bestShift) % kSectors;
        } else {
            j = ((kSectors - 1 - i) + bestShift) % kSectors;
            if (j < 0) j += kSectors;
        }
        if (srcMedoid[i] < 0 || tgtMedoid[j] < 0) continue;
        int srcLocal = srcMedoid[i];
        if (srcLocal >= 0 && srcLocal < (int)silhouetteIndices.size()) {
            g_cyclicPairSrcVertIdx[i] = (int)silhouetteIndices[srcLocal];
        }
        g_cyclicPairTgtPos[i] = targetBdyPts[tgtMedoid[j]];
        g_cyclicPairValid[i]  = (g_cyclicPairSrcVertIdx[i] >= 0) ? 1 : 0;
    }
    g_cyclicAvailable = true;

    // ---- 9. bestT を全 organMeshes に適用 (verts + normals 同時) ----
    auto organs = getOrganList();
    float estScale = glm::length(glm::vec3(bestT[0]));
    glm::mat3 R3pure = (estScale > 1e-6f) ? glm::mat3(
                                                glm::vec3(bestT[0]) / estScale,
                                                glm::vec3(bestT[1]) / estScale,
                                                glm::vec3(bestT[2]) / estScale
                                                ) : glm::mat3(bestT);

    for (auto* m : organs) {
        if (!m) continue;
        for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
            glm::vec4 v(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2], 1.0f);
            v = bestT * v;
            m->mVertices[i]     = v.x;
            m->mVertices[i + 1] = v.y;
            m->mVertices[i + 2] = v.z;
        }
        if (!m->mNormals.empty()) {
            for (size_t i = 0; i + 2 < m->mNormals.size(); i += 3) {
                glm::vec3 nrm(m->mNormals[i], m->mNormals[i+1], m->mNormals[i+2]);
                nrm = glm::normalize(R3pure * nrm);
                m->mNormals[i]     = nrm.x;
                m->mNormals[i + 1] = nrm.y;
                m->mNormals[i + 2] = nrm.z;
            }
        }
        setUp(*m);
    }

    std::cout << "[Shift+P] Applied cyclic prealign T (scale=" << estScale
              << ")  -> proceeding to ICP refinement" << std::endl;

    // ---- 10. フィルタ済み target に temp swap して既存パイプラインで ICP 精錬 ----
    //          初期 T が近い状態なら FGR は identity を返し ICP が仕上げる、
    //          という流れを期待。Key P と同じパラメータ。
    auto filteredTarget = std::make_shared<Reg3DCustom::PointCloud>();
    const bool copyN  = savedFullTarget->hasNormals();
    filteredTarget->points.reserve(targetBdyPts.size());
    if (copyN)   filteredTarget->normals.reserve(targetBdyPts.size());
    filteredTarget->boundaryDist.reserve(targetBdyPts.size());
    if (hasInst) filteredTarget->instrumentDist.reserve(targetBdyPts.size());

    for (size_t i = 0; i < savedFullTarget->size(); i++) {
        float bd  = savedFullTarget->boundaryDist[i];
        float idd = hasInst ? savedFullTarget->instrumentDist[i] : 9999.0f;
        if (bd < kBoundaryPxTh && idd >= instTh) {
            filteredTarget->points.push_back(savedFullTarget->points[i]);
            if (copyN)   filteredTarget->normals.push_back(savedFullTarget->normals[i]);
            filteredTarget->boundaryDist.push_back(bd);
            if (hasInst) filteredTarget->instrumentDist.push_back(idd);
        }
    }
    filteredTarget->colors.assign(filteredTarget->points.size(),
                                  glm::vec3(0.0f, 1.0f, 0.0f));

    Reg3DCustom::setCachedTargetCloud(filteredTarget);

    bool ok = true;
    try {
        Reg3DCustom::performRegistrationSingleMesh(
            organs, liverMesh3D, silhouetteIndices,
            screenMesh, OrbitCam.cameraPos,
            gGridWidth, gGridHeight(),
            1,
            RegRatios::convergence(),
            0.35f,
            true, 0.03f,
            RegRatios::zThresh(),
            RegRatios::voxel());
    } catch (const std::exception& e) {
        std::cerr << "[Shift+P] performRegistrationSingleMesh threw: " << e.what()
        << std::endl;
        ok = false;
    }

    // 必ず元の target を復元 (例外時も保証)
    Reg3DCustom::setCachedTargetCloud(savedFullTarget);

    if (!ok) { g_callIdx++; return; }

    // ---- 11. メトリクスは元のフル target で評価 (Key O / P と同じ評価軸) ----
    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== Shift+P (Cyclic) Complete  RMSE=" << registrationHandle.compRmse
              << "  (cyclic chamfer was " << bestScore << ")"
              << " ===" << std::endl;

    g_callIdx++;
}

// =========================================================
//  QuadCyclicMedoids — Ctrl+P / Shift+Ctrl+P 共通の前処理結果
// ---------------------------------------------------------
//  runQuadCyclic (Ctrl+P) と runQuadCyclicRansac (Shift+Ctrl+P) の
//  共通前処理 (Step 1-7: AR 可視性 → silhouette ∩ quad → target boundary →
//  view plane → 2D 重心 → N セクター medoid) の出力を 1 つの struct に
//  集約する。両関数で Step 1-7 は完全同一なので、差し替えるのは Step 8
//  以降の matching algorithm だけになる。
//
//  ライフタイム注意:
//    KDTree は targetBdyPts の参照を握る (NanoflannAdaptor 経由)。
//    呼び出し側はこの struct を「値で受けてスコープ内で保持し続ける」
//    こと。move されると targetBdyPts のアドレスが変わって KDTree が
//    dangling になるため、return 後の代入や再配置は禁止 (RVO 任せで OK)。
//    そのため KDTree 自体はこの struct には入れず、呼び出し側が
//    extractQuadCyclicMedoids() の戻り値を受け取った後で別途構築する。
//
//  失敗時:
//    ok=false で返る。エラーログは関数内で出力済み。呼び出し側は
//    g_callIdx++ してから return する想定 (既存 runQuadCyclic と同様)。
// =========================================================
struct QuadCyclicMedoids {
    // --- ok flag (false なら何もしないで abort) ---
    bool ok = false;

    // --- 入力スナップショット (logging / Step 11 で参照) ---
    uint8_t mask         = 0;     // g_activeQuadrantMask snapshot
    bool    fallbackUsed = false; // silhouette ∩ quad < 50 で silhouette を外したか

    // --- Source side (AR ∩ silh ∩ quad の最終点群) ---
    std::vector<glm::vec3> sourceSilhPts;     // 最終 source 点群 (= g_cluster1Points 相当)
    std::vector<size_t>    silhouetteIndices; // sourceSilhPts[i] に対応する liverMesh3D 頂点 index
    std::vector<int>       srcMedoid;         // [kSectors], -1 if sector empty
    int                    nSrcOcc = 0;

    // --- Target side (boundary ∩ instrument filter 適用後) ---
    std::vector<glm::vec3> targetBdyPts;
    std::vector<int>       tgtMedoid;         // [kSectors], -1 if sector empty
    int                    nTgtOcc = 0;
    std::shared_ptr<Reg3DCustom::PointCloud> savedFullTarget; // ICP swap の復元用

    // --- Target filter params (Step 10 で filteredTarget 再構築に使う) ---
    float kBoundaryPxTh = 12.0f;
    float instTh        = 40.0f;
    bool  hasInst       = false;

    // --- View frame (AR 固定 cam_pos=(0,0,0), look-at=(0,0,1)) ---
    glm::vec3 arPos     {0.0f, 0.0f, 0.0f};
    glm::vec3 arTgt     {0.0f, 0.0f, 1.0f};
    glm::vec3 arViewDir {0.0f, 0.0f, 1.0f};

    // --- Sector / sanity constants (Shift+P / Ctrl+P と同じ) ---
    int   kSectors       = 24;
    int   kMinValidPairs = 4;
    float kScaleLo       = 0.5f;
    float kScaleHi       = 2.0f;
};

// =========================================================
//  extractQuadCyclicMedoids — Step 1-7 の共通前処理
// ---------------------------------------------------------
//  入力 (暗黙):
//    - liverMesh3D                 (source mesh)
//    - g_liverRegion / g_liverLR   (quadrant labels)
//    - g_activeQuadrantMask        (UI 選択象限)
//    - g_silhouetteSrcCosThresh    (silhouette 閾値)
//    - g_instrumentPxThresh        (target 器具距離閾値)
//    - Reg3DCustom::getCachedTargetCloud()  (target cloud)
//
//  副作用 (Shift+B 等の可視化のため既存挙動を維持):
//    - g_visibleSourcePoints     := AR 可視全頂点
//    - g_silhouetteSourcePoints  := silh ∩ quad ∩ AR の最終点群
//    - g_cluster1Points          := 同上
//    - g_cluster2Points.clear()
//
//  出力: QuadCyclicMedoids (ok=true なら sourceSilhPts / targetBdyPts /
//    srcMedoid / tgtMedoid / savedFullTarget 等がすべて populated)
//
//  tagBare は "Ctrl+P" や "Shift+Ctrl+P" 等の素の文字列を渡す。
//  ログ prefix は内部で "[" + tag + "]" / "[" + tag + "/Source]" /
//  "[" + tag + "/Target]" の 3 系統に整形する (既存 runQuadCyclic の
//  log フォーマットと完全に byte-identical)。
// =========================================================
inline QuadCyclicMedoids extractQuadCyclicMedoids(const char* tagBare) {
    const std::string tag    = std::string("[") + tagBare + "]";
    const std::string tagSrc = std::string("[") + tagBare + "/Source]";
    const std::string tagTgt = std::string("[") + tagBare + "/Target]";

    QuadCyclicMedoids out;
    out.kBoundaryPxTh = 12.0f;
    out.instTh        = g_instrumentPxThresh;
    out.kSectors      = 24;
    out.kMinValidPairs= 4;
    out.kScaleLo      = 0.5f;
    out.kScaleHi      = 2.0f;
    out.arPos         = glm::vec3(0.0f, 0.0f, 0.0f);
    out.arTgt         = glm::vec3(0.0f, 0.0f, 1.0f);
    out.arViewDir     = glm::vec3(0.0f, 0.0f, 1.0f);

    // ---- Quadrant label validity guard ----
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << tag << " Region/LR labels not computed: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << "  LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << ". Press R/Y or use Initial Orientation panel." << std::endl;
        return out;  // ok=false
    }

    // ---- Quadrant subset (anatomy-based source filter) ----
    const uint8_t mask = g_activeQuadrantMask;
    out.mask = mask;
    std::vector<int> quadIdx = LiverLeftRightLabel::makeQuadrantSubsetIdx(
        g_liverRegion.labels, g_liverLR.labels, mask);
    std::cout << tag << " quadrant_mask = "
              << LiverLeftRightLabel::quadrantMaskString(mask)
              << "  (0x" << std::hex << (unsigned)mask << std::dec << ")"
              << "  subset_size=" << quadIdx.size() << "/"
              << g_liverRegion.labels.size() << std::endl;
    if (quadIdx.empty()) {
        std::cerr << tag << " Empty quadrant subset (mask=0x"
                  << std::hex << (unsigned)mask << std::dec
                  << "). Select at least one quadrant." << std::endl;
        return out;
    }
    std::unordered_set<int> quadSet(quadIdx.begin(), quadIdx.end());

    // ---- 0. Source mesh の法線確認 ----
    if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
        std::cout << tag << " liverMesh3D normals missing -- computing from faces..."
                  << std::endl;
        Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
    }

    // ---- 1. Source: AR 固定視点で可視頂点を取得 (OrbitCam 非依存) ----
    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    const glm::vec3 arPos     = out.arPos;
    const glm::vec3 arTgt     = out.arTgt;
    const glm::vec3 arViewDir = out.arViewDir;
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, arPos, arTgt);
    if (vis.cloud->size() < 50) {
        std::cerr << tag << " Not enough AR-visible points ("
                  << vis.cloud->size() << "). Liver may be off-screen "
                     "from the AR camera." << std::endl;
        return out;
    }

    // ---- 2. Source: silhouette ∩ quadrant filter (3-way intersection) ----
    //         AR-visible cloud に対し silhouette (|n·arViewDir|<th) + quad を
    //         両方満たす頂点を選ぶ。一回のループで実装。
    //         silhouette ∩ quad が < 50 なら quadrant のみに fallback。
    g_visibleSourcePoints = vis.points;
    g_silhouetteSourcePoints.clear();

    std::vector<glm::vec3> sourceSilhPts;
    std::vector<size_t>    silhouetteIndices;
    sourceSilhPts.reserve(vis.cloud->size());
    silhouetteIndices.reserve(vis.cloud->size());

    int n_quadOnly = 0, n_silhOnly = 0, n_both = 0;
    const bool hasN = vis.cloud->hasNormals();
    if (!hasN) {
        std::cerr << tag << " visible cloud has no normals -- using AR ∩ quad only"
                  << std::endl;
        for (size_t k = 0; k < vis.cloud->size(); k++) {
            const int vi = static_cast<int>(vis.vertexIndices[k]);
            if (!quadSet.count(vi)) continue;
            sourceSilhPts.push_back(vis.cloud->points[k]);
            silhouetteIndices.push_back(vis.vertexIndices[k]);
        }
    } else {
        for (size_t k = 0; k < vis.cloud->size(); k++) {
            const int vi = static_cast<int>(vis.vertexIndices[k]);
            const glm::vec3& n = vis.cloud->normals[k];
            const float s = std::abs(glm::dot(n, arViewDir));
            const bool inQuad = (quadSet.count(vi) > 0);
            const bool inSilh = (s < g_silhouetteSrcCosThresh);
            if (inQuad && inSilh) n_both++;
            else if (inQuad)      n_quadOnly++;
            else if (inSilh)      n_silhOnly++;
            if (inQuad && inSilh) {
                sourceSilhPts.push_back(vis.cloud->points[k]);
                silhouetteIndices.push_back(vis.vertexIndices[k]);
            }
        }
        if (sourceSilhPts.size() < 50) {
            // Fallback: quadrant のみ (silhouette を緩める)
            std::cerr << tag << " Triple intersection too small ("
                      << sourceSilhPts.size()
                      << ") -- falling back to AR ∩ quad only" << std::endl;
            sourceSilhPts.clear();
            silhouetteIndices.clear();
            for (size_t k = 0; k < vis.cloud->size(); k++) {
                const int vi = static_cast<int>(vis.vertexIndices[k]);
                if (!quadSet.count(vi)) continue;
                sourceSilhPts.push_back(vis.cloud->points[k]);
                silhouetteIndices.push_back(vis.vertexIndices[k]);
            }
            out.fallbackUsed = true;
        }
    }
    g_silhouetteSourcePoints = sourceSilhPts;
    g_cluster1Points         = sourceSilhPts;
    g_cluster2Points.clear();

    std::cout << tagSrc << " AR-visible=" << vis.cloud->size()
              << "  silh∩quad=" << n_both
              << "  quad_only=" << n_quadOnly
              << "  silh_only=" << n_silhOnly
              << "  final=" << sourceSilhPts.size()
              << "  cosThresh=" << g_silhouetteSrcCosThresh << std::endl;

    if (sourceSilhPts.size() < 50) {
        std::cerr << tag << " Source too small after triple intersection ("
                  << sourceSilhPts.size() << "). Try a wider mask." << std::endl;
        return out;
    }

    // ---- 3. Target: boundary filtering (Shift+P と完全同じ) ----
    auto savedFullTarget = Reg3DCustom::getCachedTargetCloud();
    if (!savedFullTarget || savedFullTarget->empty()) {
        std::cerr << tag << " No cached target cloud" << std::endl;
        return out;
    }

    const float kBoundaryPxTh = out.kBoundaryPxTh;
    const float instTh        = out.instTh;
    const bool  hasInst       = savedFullTarget->hasInstrumentDist();
    const bool  hasBdy        = savedFullTarget->hasBoundaryDist();
    out.hasInst = hasInst;

    if (!hasBdy) {
        std::cerr << tag << " target cloud has no boundaryDist -- aborting"
                  << std::endl;
        return out;
    }

    std::vector<glm::vec3> targetBdyPts;
    targetBdyPts.reserve(savedFullTarget->size() / 4);
    for (size_t i = 0; i < savedFullTarget->size(); i++) {
        float bd  = savedFullTarget->boundaryDist[i];
        float idd = hasInst ? savedFullTarget->instrumentDist[i] : 9999.0f;
        if (bd < kBoundaryPxTh && idd >= instTh) {
            targetBdyPts.push_back(savedFullTarget->points[i]);
        }
    }
    std::cout << tagTgt << " full=" << savedFullTarget->size()
              << "  boundary_kept=" << targetBdyPts.size()
              << "  (boundaryPx<" << kBoundaryPxTh
              << ", instPx>=" << instTh << ")" << std::endl;

    if (targetBdyPts.size() < 50) {
        std::cerr << tag << " Too few boundary points in target ("
                  << targetBdyPts.size() << ") -- aborting" << std::endl;
        return out;
    }

    // ---- 4. View plane を構築 (AR 固定 viewDir に直交する 2D 基底) ----
    glm::vec3 vUp_world = (std::abs(arViewDir.y) > 0.9f)
                              ? glm::vec3(1.0f, 0.0f, 0.0f)
                              : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 vRight = glm::normalize(glm::cross(vUp_world, arViewDir));
    glm::vec3 vUp    = glm::normalize(glm::cross(arViewDir, vRight));

    auto project2D = [&](const glm::vec3& p) -> glm::vec2 {
        glm::vec3 d = p - arPos;   // arPos = (0,0,0) so d == p
        return glm::vec2(glm::dot(d, vRight), glm::dot(d, vUp));
    };

    // ---- 5. 2D 重心 (source / target それぞれ; Shift+P と同じ) ----
    glm::vec2 srcCentroid2D(0.0f), tgtCentroid2D(0.0f);
    for (const auto& p : sourceSilhPts) srcCentroid2D += project2D(p);
    for (const auto& p : targetBdyPts)  tgtCentroid2D += project2D(p);
    srcCentroid2D /= float(sourceSilhPts.size());
    tgtCentroid2D /= float(targetBdyPts.size());

    // ---- 6. N セクター分割 (Shift+P と同じ) ----
    const int   kSectors = out.kSectors;
    const float kTwoPi   = 6.28318530717958f;

    auto assignSector = [&](const glm::vec2& p2d, const glm::vec2& center) -> int {
        glm::vec2 d = p2d - center;
        float theta = -std::atan2(d.y, d.x);
        if (theta < 0.0f) theta += kTwoPi;
        int idx = int(theta / (kTwoPi / float(kSectors)));
        if (idx >= kSectors) idx = kSectors - 1;
        if (idx < 0)         idx = 0;
        return idx;
    };

    std::vector<std::vector<size_t>> srcSectorIdx(kSectors);
    std::vector<std::vector<size_t>> tgtSectorIdx(kSectors);

    for (size_t i = 0; i < sourceSilhPts.size(); i++) {
        int s = assignSector(project2D(sourceSilhPts[i]), srcCentroid2D);
        srcSectorIdx[s].push_back(i);
    }
    for (size_t i = 0; i < targetBdyPts.size(); i++) {
        int s = assignSector(project2D(targetBdyPts[i]), tgtCentroid2D);
        tgtSectorIdx[s].push_back(i);
    }

    auto medoidOfSector = [](const std::vector<glm::vec3>& pts,
                             const std::vector<size_t>& idxList) -> int {
        if (idxList.empty()) return -1;
        glm::vec3 mean(0.0f);
        for (size_t i : idxList) mean += pts[i];
        mean /= float(idxList.size());
        int   best  = (int)idxList[0];
        float bestD = glm::distance(pts[best], mean);
        for (size_t i : idxList) {
            float d = glm::distance(pts[i], mean);
            if (d < bestD) { bestD = d; best = (int)i; }
        }
        return best;
    };

    std::vector<int> srcMedoid(kSectors, -1);
    std::vector<int> tgtMedoid(kSectors, -1);
    int nSrcOcc = 0, nTgtOcc = 0;
    for (int k = 0; k < kSectors; k++) {
        srcMedoid[k] = medoidOfSector(sourceSilhPts, srcSectorIdx[k]);
        tgtMedoid[k] = medoidOfSector(targetBdyPts,  tgtSectorIdx[k]);
        if (srcMedoid[k] >= 0) nSrcOcc++;
        if (tgtMedoid[k] >= 0) nTgtOcc++;
    }
    std::cout << "[Sectors] N=" << kSectors
              << "  src_occupied=" << nSrcOcc << "/" << kSectors
              << "  tgt_occupied=" << nTgtOcc << "/" << kSectors << std::endl;

    if (nSrcOcc < out.kMinValidPairs || nTgtOcc < out.kMinValidPairs) {
        std::cerr << tag << " Too few occupied sectors -- aborting" << std::endl;
        return out;
    }

    // ---- 7. Populate out (KDTree は呼び出し側が targetBdyPts から構築) ----
    out.sourceSilhPts     = std::move(sourceSilhPts);
    out.silhouetteIndices = std::move(silhouetteIndices);
    out.srcMedoid         = std::move(srcMedoid);
    out.nSrcOcc           = nSrcOcc;
    out.targetBdyPts      = std::move(targetBdyPts);
    out.tgtMedoid         = std::move(tgtMedoid);
    out.nTgtOcc           = nTgtOcc;
    out.savedFullTarget   = savedFullTarget;
    out.ok                = true;
    return out;
}

// =========================================================
//  QuadCyclic Registration (Ctrl+P)
// ---------------------------------------------------------
//  Shift+P (runCyclicBoundaryReg) と同じ 24 セクター巡回シフト ×
//  Umeyama + ICP パイプラインを使うが、source 頂点の絞り込みフィルタを
//  以下の三段で構成する:
//
//    filter_AR    : AR 固定視点 cameraPos=(0,0,0), target=(0,0,1)
//                   から extractVisibleVerticesCustom() で得た可視頂点
//                   (raycast 遮蔽除去 + dot(rayDir,viewDir)>0.3)
//    filter_Silh  : |dot(normal, arViewDir)| < g_silhouetteSrcCosThresh
//                   (シルエット近傍頂点; 既存 Shift+P と同じ閾値)
//    filter_Quad  : LiverLeftRightLabel::makeQuadrantSubsetIdx() で
//                   g_activeQuadrantMask に属する解剖象限頂点
//                   (Initial Orientation panel / Ctrl+G と共有)
//
//    source_final = filter_AR ∩ filter_Silh ∩ filter_Quad
//                   ↓
//    24 セクター分割 + medoid + 巡回シフト × ミラー = 48 パターン
//                   ↓
//    各パターンで Umeyama → chamfer RMSE で best 選択 → ICP 精錬
//
//  動機:
//    - Shift+P の cyclic boundary アルゴリズムは「境界形状の rim」を
//      前提に設計されている (sector medoid が rim 上に乗ることで
//      回転推定が幾何的に意味を持つ)。silhouette フィルタは必須なので残す。
//    - その上で、Initial Orientation で選んだ象限に source を絞ることで、
//      局所形状ベースの cyclic match を試みる:
//        例: Q:AL (左前葉) なら、左前葉の rim だけが target 境界に
//            cyclic shift で当てに行く形になる。
//    - AR 固定視点採用で OrbitCam に副作用なし、再現性向上 (QuadAuto と同形)。
//
//  target 側:
//    Shift+P と完全同一 (boundaryDist < 12px ∩ instrumentDist >= threshold)。
//    target は 2D 画像由来なので象限投影は不可、ここは無変更。
//
//  Visualization (Shift+B):
//    g_cyclic* globals を Shift+P と同じように書き込む。Ctrl+P 実行後に
//    Shift+B を押すと、Ctrl+P が選んだ対応点ペアが可視化される
//    (Shift+P と同じ可視化機構を共有)。
//
//  Triple intersection fallback:
//    silhouette ∩ quad が < 50 点なら quadrant のみに緩める (silhouette
//    を一段外す)。完全に空ならエラー abort。
//
//  ラベル未計算ガード:
//    Shift+O / QuadAuto と同じく呼び出し元で recompute を auto-trigger
//    する想定。関数内では valid() チェックのみ。
//
//  [Refactor 2026-05] Step 1-7 (前処理) は extractQuadCyclicMedoids ヘルパー
//    に切り出した。Shift+Ctrl+P (runQuadCyclicRansac) と共通化するため。
//    本関数の挙動 (RMSE / 適用変換 / Shift+B 可視化 / ログ出力) は
//    refactor 前と完全に byte-identical を維持する。
// =========================================================
inline void runQuadCyclic() {
    std::cout << "\n=== QuadCyclic Registration (Ctrl+P) ===" << std::endl;

    // Phase 1: シード固定 (Shift+P と同じ)
    const uint32_t fgr_seed = g_trialSeed + g_callIdx;
    Reg3DCustom::setFgrSeed(fgr_seed);
    std::cout << "[Seed] FGR=" << fgr_seed
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;

    // ---- Step 1-7: 共通前処理 (AR ∩ silh ∩ quad → medoids) ----
    QuadCyclicMedoids med = extractQuadCyclicMedoids("Ctrl+P");
    if (!med.ok) { g_callIdx++; return; }

    // Chamfer 評価用 KD tree (target 側; med.targetBdyPts は move しないこと)
    Reg3DCustom::NanoflannAdaptor tgtAdaptor(med.targetBdyPts);
    auto tgtTree = Reg3DCustom::buildKDTree(tgtAdaptor);

    auto chamferRMSE = [&](const glm::mat4& T) -> float {
        float sumSq = 0.0f;
        int   cnt   = 0;
        for (const auto& p : med.sourceSilhPts) {
            glm::vec4 v(p, 1.0f);
            glm::vec3 tp = glm::vec3(T * v);
            size_t nn; float d2;
            if (Reg3DCustom::searchKNN1(*tgtTree, tp, nn, d2)) {
                sumSq += d2;
                cnt++;
            }
        }
        if (cnt == 0) return 1e9f;
        return std::sqrt(sumSq / float(cnt));
    };

    // =====================================================
    //  [Ctrl+P init prior] (Shift+Ctrl+P から移植)
    // ---------------------------------------------------------
    //  Ctrl+P は 22 ペア over-determined Umeyama を 48 パターン
    //  (shift × dir) 試して best chamfer を取る。これに init pose prior を
    //  追加すれば、たまに鏡像系の不安定な解 (X 軸 ~180°) を出すのを抑え、
    //  「init から離れすぎない解」 を優先するようにできる。
    //
    //  Shift+Ctrl+P と同じ 2 つのレイヤを使う:
    //    (1) hard limit: 各軸 (X/Y/Z) の Euler 角 |angle| > g_qcrMaxAxisRotDeg
    //        の T は採用しない (init からの相対回転 = T が回してる量)。
    //    (2) light penalty: best 選別を score_chamfer 単独ではなく
    //          score_total = score_chamfer + λ × displacement
    //        で行う。λ = g_qcrInitDispWeight (Shift+Ctrl+P と共有)。
    //
    //  Shift+Ctrl+P が 3 点 exact fit の noise で勝てないのに対し、Ctrl+P は
    //  最初から 22 ペア over-determined fit なので、prior を足すだけで安定。
    //  Shift+Ctrl+P (RANSAC 版) は並列存在として残す (削除しない)。
    // =====================================================
    auto displacementOf = [&](const glm::mat4& T) -> float {
        if (med.sourceSilhPts.empty()) return 0.0f;
        float sum = 0.0f;
        int   n   = 0;
        for (const auto& p : med.sourceSilhPts) {
            const glm::vec3 tp = glm::vec3(T * glm::vec4(p, 1.0f));
            const float dx = tp.x - p.x;
            const float dy = tp.y - p.y;
            const float dz = tp.z - p.z;
            sum += std::sqrt(dx*dx + dy*dy + dz*dz);
            n++;
        }
        return (n > 0) ? sum / float(n) : 0.0f;
    };
    // T からスケール除去 → R 抽出 → X-Y-Z Euler (deg) を返す。
    // GLM は column-major: T[col][row]; R3pure[col] が R の col 列 (basis vec)。
    auto extractAxisRotDeg = [&](const glm::mat4& T, float out[3]) {
        const float sx = glm::length(glm::vec3(T[0]));
        const float sy = glm::length(glm::vec3(T[1]));
        const float sz = glm::length(glm::vec3(T[2]));
        if (sx < 1e-6f || sy < 1e-6f || sz < 1e-6f) {
            out[0] = out[1] = out[2] = 0.0f;
            return;
        }
        const glm::mat3 R3pure(
            glm::vec3(T[0]) / sx,
            glm::vec3(T[1]) / sy,
            glm::vec3(T[2]) / sz
        );
        // X-Y-Z (extrinsic) Euler: x = atan2(R[2][1], R[2][2]); etc.
        // GLM access: R3pure[col][row] → row=R3pure[col].r, etc.
        const float r00 = R3pure[0].x;
        const float r10 = R3pure[0].y;
        const float r20 = R3pure[0].z;
        const float r21 = R3pure[1].z;
        const float r22 = R3pure[2].z;
        const float rad2deg = 180.0f / 3.14159265358979f;
        float rx, ry, rz;
        ry = std::asin(std::max(-1.0f, std::min(1.0f, -r20)));
        if (std::abs(r20) < 0.99999f) {
            rx = std::atan2(r21, r22);
            rz = std::atan2(r10, r00);
        } else {
            rx = std::atan2(-R3pure[2].y, R3pure[1].y);
            rz = 0.0f;
        }
        out[0] = rx * rad2deg;
        out[1] = ry * rad2deg;
        out[2] = rz * rad2deg;
    };
    // T からスケール除去 → R 抽出 → axis-angle 表現の総回転量 (deg) を返す。
    //   trace(R) = 1 + 2 cos(θ)  ⇒  θ = arccos((trace(R) - 1) / 2)
    //   per-axis Euler では (35°,-35°,35°) のような off-axis 大回転を
    //   見逃すので、こちらを併用して total angle で防御する。
    auto extractTotalRotDeg = [&](const glm::mat4& T) -> float {
        const float sx = glm::length(glm::vec3(T[0]));
        const float sy = glm::length(glm::vec3(T[1]));
        const float sz = glm::length(glm::vec3(T[2]));
        if (sx < 1e-6f || sy < 1e-6f || sz < 1e-6f) return 0.0f;
        // R3pure column = unit basis; trace = c0.x + c1.y + c2.z
        const float r00 = T[0].x / sx;
        const float r11 = T[1].y / sy;
        const float r22 = T[2].z / sz;
        const float trace = r00 + r11 + r22;
        float cosA = (trace - 1.0f) * 0.5f;
        if (cosA >  1.0f) cosA =  1.0f;
        if (cosA < -1.0f) cosA = -1.0f;
        const float rad2deg = 180.0f / 3.14159265358979f;
        return std::acos(cosA) * rad2deg;
    };
    const float maxRotDeg      = g_qcrMaxAxisRotDeg;    // hard limit per axis (deg)
    const float maxTotalRotDeg = g_qcrMaxTotalRotDeg;   // hard limit on total axis-angle (deg)
    const float lambdaDisp = g_qcrInitDispWeight;   // λ for total = cham + λ × disp
    const int   kSectors       = med.kSectors;
    const int   kMinValidPairs = med.kMinValidPairs;
    const float kScaleLo       = med.kScaleLo;
    const float kScaleHi       = med.kScaleHi;

    // ---- 8. 巡回シフト × ミラー = 2N パターン (Shift+P と同じ枠組み + init prior) ----
    glm::mat4 bestT(1.0f);
    float bestScore     = 1e9f;   // best score_total (chamfer + λ × disp)
    float bestChamfer   = 1e9f;   // 採用した best の chamfer 単独
    float bestDisp      = 0.0f;   // 採用した best の disp 単独
    float bestRot[3]    = {0.0f, 0.0f, 0.0f};
    int   bestShift     = 0, bestRev = 0;
    int   triedCount    = 0;
    int   validCount    = 0;
    int   rotFiltered   = 0;      // hard limit で落ちた count

    for (int rev = 0; rev < 2; rev++) {
        for (int shift = 0; shift < kSectors; shift++) {
            std::vector<glm::vec3> srcPairs, tgtPairs;
            srcPairs.reserve(kSectors);
            tgtPairs.reserve(kSectors);

            for (int i = 0; i < kSectors; i++) {
                int j;
                if (rev == 0) {
                    j = (i + shift) % kSectors;
                } else {
                    j = ((kSectors - 1 - i) + shift) % kSectors;
                    if (j < 0) j += kSectors;
                }
                if (med.srcMedoid[i] < 0 || med.tgtMedoid[j] < 0) continue;
                srcPairs.push_back(med.sourceSilhPts[med.srcMedoid[i]]);
                tgtPairs.push_back(med.targetBdyPts [med.tgtMedoid[j]]);
            }
            triedCount++;
            if ((int)srcPairs.size() < kMinValidPairs) continue;

            glm::mat4 T = Reg3D::UmeyamaRegistration(srcPairs, tgtPairs);

            bool finiteT = true;
            for (int c = 0; c < 4 && finiteT; c++)
                for (int r = 0; r < 4 && finiteT; r++)
                    if (!std::isfinite(T[c][r])) finiteT = false;
            if (!finiteT) continue;

            float scale = glm::length(glm::vec3(T[0]));
            if (!std::isfinite(scale) || scale < kScaleLo || scale > kScaleHi) continue;

            // [init prior レイヤ 1] hard limit: 各軸 (X/Y/Z) 回転 |angle| > maxRotDeg は除外
            float axisDeg[3];
            extractAxisRotDeg(T, axisDeg);
            const float maxAxisDeg = std::max({std::abs(axisDeg[0]),
                                               std::abs(axisDeg[1]),
                                               std::abs(axisDeg[2])});
            if (maxAxisDeg > maxRotDeg) {
                rotFiltered++;
                continue;
            }
            // [init prior レイヤ 1b] total axis-angle 回転量で追加ガード:
            //   per-axis では off-axis (e.g. (1,1,1) 軸まわり大回転) を捉えられない
            //   ケースがあるため、rotation matrix の総回転量 (axis-angle) も
            //   閾値 maxTotalRotDeg 以下に縛る。
            const float totalDeg = extractTotalRotDeg(T);
            if (totalDeg > maxTotalRotDeg) {
                rotFiltered++;
                continue;
            }

            // [init prior レイヤ 2] light penalty: score_total = chamfer + λ × disp
            const float chamfer = chamferRMSE(T);
            const float disp    = displacementOf(T);
            const float total   = chamfer + lambdaDisp * disp;

            validCount++;
            if (total < bestScore) {
                bestScore   = total;
                bestT       = T;
                bestChamfer = chamfer;
                bestDisp    = disp;
                bestRot[0]  = axisDeg[0];
                bestRot[1]  = axisDeg[1];
                bestRot[2]  = axisDeg[2];
                bestShift   = shift;
                bestRev     = rev;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(4)
              << "[Cyclic] tried=" << triedCount
              << "  rot_filtered=" << rotFiltered
              << "  valid=" << validCount
              << "  best_shift=" << bestShift
              << "  best_dir=" << (bestRev ? "reverse(CCW)" : "forward(CW)")
              << "  chamfer=" << bestChamfer
              << "  disp=" << bestDisp
              << "  total=" << bestScore
              << "  rot=[" << std::setprecision(1)
              << bestRot[0] << "," << bestRot[1] << "," << bestRot[2] << "]deg"
              << "  (lambda=" << std::setprecision(3) << lambdaDisp
              << ", max_rot=" << std::setprecision(1) << maxRotDeg << "deg"
              << ", max_total=" << std::setprecision(1) << maxTotalRotDeg << "deg)"
              << std::defaultfloat << std::setprecision(6) << std::endl;

    if (validCount == 0) {
        // Fallback: hard limit ですべて除外された場合に限り、init prior を切って
        // chamfer 単独で再評価。Shift+Ctrl+P で観測された「Stage 1 で全 reject」型
        // failure mode を Ctrl+P でも防ぐ。本来 Ctrl+P は 22 ペア fit なので
        // hard limit に引っかかる解は稀だが、念のため。
        if (rotFiltered > 0) {
            std::cout << "[Ctrl+P] All " << rotFiltered << " rot-filtered; "
                      << "retrying WITHOUT hard limit (chamfer-only)" << std::endl;
            float bestChamFallback = 1e9f;
            int   bestShiftFallback = 0, bestRevFallback = 0;
            glm::mat4 bestTFallback(1.0f);
            for (int rev = 0; rev < 2; rev++) {
                for (int shift = 0; shift < kSectors; shift++) {
                    std::vector<glm::vec3> srcPairs, tgtPairs;
                    srcPairs.reserve(kSectors);
                    tgtPairs.reserve(kSectors);
                    for (int i = 0; i < kSectors; i++) {
                        int j;
                        if (rev == 0) j = (i + shift) % kSectors;
                        else { j = ((kSectors - 1 - i) + shift) % kSectors; if (j < 0) j += kSectors; }
                        if (med.srcMedoid[i] < 0 || med.tgtMedoid[j] < 0) continue;
                        srcPairs.push_back(med.sourceSilhPts[med.srcMedoid[i]]);
                        tgtPairs.push_back(med.targetBdyPts [med.tgtMedoid[j]]);
                    }
                    if ((int)srcPairs.size() < kMinValidPairs) continue;
                    glm::mat4 T = Reg3D::UmeyamaRegistration(srcPairs, tgtPairs);
                    bool finiteT = true;
                    for (int c = 0; c < 4 && finiteT; c++)
                        for (int r = 0; r < 4 && finiteT; r++)
                            if (!std::isfinite(T[c][r])) finiteT = false;
                    if (!finiteT) continue;
                    float scale = glm::length(glm::vec3(T[0]));
                    if (!std::isfinite(scale) || scale < kScaleLo || scale > kScaleHi) continue;
                    float ch = chamferRMSE(T);
                    if (ch < bestChamFallback) {
                        bestChamFallback  = ch;
                        bestTFallback     = T;
                        bestShiftFallback = shift;
                        bestRevFallback   = rev;
                    }
                }
            }
            if (bestChamFallback < 1e8f) {
                float fbRot[3];
                extractAxisRotDeg(bestTFallback, fbRot);
                bestT       = bestTFallback;
                bestChamfer = bestChamFallback;
                bestDisp    = displacementOf(bestTFallback);
                bestScore   = bestChamfer + lambdaDisp * bestDisp;
                bestRot[0]  = fbRot[0]; bestRot[1] = fbRot[1]; bestRot[2] = fbRot[2];
                bestShift   = bestShiftFallback;
                bestRev     = bestRevFallback;
                validCount  = 1;
                std::cout << std::fixed << std::setprecision(4)
                          << "[Ctrl+P] Fallback found: shift=" << bestShift
                          << " dir=" << (bestRev ? "reverse(CCW)" : "forward(CW)")
                          << " chamfer=" << bestChamfer
                          << " rot=[" << std::setprecision(1)
                          << bestRot[0] << "," << bestRot[1] << "," << bestRot[2] << "]"
                          << std::defaultfloat << std::setprecision(6) << std::endl;
            }
        }
    }

    if (validCount == 0) {
        std::cerr << "[Ctrl+P] No valid cyclic match found -- aborting" << std::endl;
        g_callIdx++;
        return;
    }

    // ---- 8.5. Shift+B 可視化用に対応点を保存 (Shift+P と同じ globals) ----
    g_cyclicSectors   = kSectors;
    g_cyclicBestShift = bestShift;
    g_cyclicBestRev   = bestRev;
    g_cyclicPairSrcVertIdx.assign(kSectors, -1);
    g_cyclicPairTgtPos.assign(kSectors, glm::vec3(0.0f));
    g_cyclicPairValid.assign(kSectors, 0);
    for (int i = 0; i < kSectors; i++) {
        int j;
        if (bestRev == 0) {
            j = (i + bestShift) % kSectors;
        } else {
            j = ((kSectors - 1 - i) + bestShift) % kSectors;
            if (j < 0) j += kSectors;
        }
        if (med.srcMedoid[i] < 0 || med.tgtMedoid[j] < 0) continue;
        int srcLocal = med.srcMedoid[i];
        if (srcLocal >= 0 && srcLocal < (int)med.silhouetteIndices.size()) {
            g_cyclicPairSrcVertIdx[i] = (int)med.silhouetteIndices[srcLocal];
        }
        g_cyclicPairTgtPos[i] = med.targetBdyPts[med.tgtMedoid[j]];
        g_cyclicPairValid[i]  = (g_cyclicPairSrcVertIdx[i] >= 0) ? 1 : 0;
    }
    g_cyclicAvailable = true;

    // ---- 9. bestT を全 organMeshes に適用 (Shift+P と同じ) ----
    auto organs = getOrganList();
    float estScale = glm::length(glm::vec3(bestT[0]));
    glm::mat3 R3pure = (estScale > 1e-6f) ? glm::mat3(
                                                glm::vec3(bestT[0]) / estScale,
                                                glm::vec3(bestT[1]) / estScale,
                                                glm::vec3(bestT[2]) / estScale
                                                ) : glm::mat3(bestT);

    for (auto* m : organs) {
        if (!m) continue;
        for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
            glm::vec4 v(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2], 1.0f);
            v = bestT * v;
            m->mVertices[i]     = v.x;
            m->mVertices[i + 1] = v.y;
            m->mVertices[i + 2] = v.z;
        }
        if (!m->mNormals.empty()) {
            for (size_t i = 0; i + 2 < m->mNormals.size(); i += 3) {
                glm::vec3 nrm(m->mNormals[i], m->mNormals[i+1], m->mNormals[i+2]);
                nrm = glm::normalize(R3pure * nrm);
                m->mNormals[i]     = nrm.x;
                m->mNormals[i + 1] = nrm.y;
                m->mNormals[i + 2] = nrm.z;
            }
        }
        setUp(*m);
    }

    std::cout << "[Ctrl+P] Applied cyclic prealign T (scale=" << estScale
              << ")  -> proceeding to ICP refinement" << std::endl;

    // ---- 10. フィルタ済み target に temp swap → ICP 精錬 (Shift+P と同じ) ----
    auto filteredTarget = std::make_shared<Reg3DCustom::PointCloud>();
    const bool  copyN         = med.savedFullTarget->hasNormals();
    const bool  hasInst       = med.hasInst;
    const float kBoundaryPxTh = med.kBoundaryPxTh;
    const float instTh        = med.instTh;
    filteredTarget->points.reserve(med.targetBdyPts.size());
    if (copyN)   filteredTarget->normals.reserve(med.targetBdyPts.size());
    filteredTarget->boundaryDist.reserve(med.targetBdyPts.size());
    if (hasInst) filteredTarget->instrumentDist.reserve(med.targetBdyPts.size());

    for (size_t i = 0; i < med.savedFullTarget->size(); i++) {
        float bd  = med.savedFullTarget->boundaryDist[i];
        float idd = hasInst ? med.savedFullTarget->instrumentDist[i] : 9999.0f;
        if (bd < kBoundaryPxTh && idd >= instTh) {
            filteredTarget->points.push_back(med.savedFullTarget->points[i]);
            if (copyN)   filteredTarget->normals.push_back(med.savedFullTarget->normals[i]);
            filteredTarget->boundaryDist.push_back(bd);
            if (hasInst) filteredTarget->instrumentDist.push_back(idd);
        }
    }
    filteredTarget->colors.assign(filteredTarget->points.size(),
                                  glm::vec3(0.0f, 1.0f, 0.0f));

    Reg3DCustom::setCachedTargetCloud(filteredTarget);

    bool ok = true;
    try {
        Reg3DCustom::performRegistrationSingleMesh(
            organs, liverMesh3D, med.silhouetteIndices,
            screenMesh, med.arPos,   // dead parameter (AR 固定で渡す)
            gGridWidth, gGridHeight(),
            1,
            RegRatios::convergence(),
            0.35f,
            true, 0.03f,
            RegRatios::zThresh(),
            RegRatios::voxel());
    } catch (const std::exception& e) {
        std::cerr << "[Ctrl+P] performRegistrationSingleMesh threw: " << e.what()
                  << std::endl;
        ok = false;
    }

    // 必ず元の target を復元 (例外時も保証)
    Reg3DCustom::setCachedTargetCloud(med.savedFullTarget);

    if (!ok) { g_callIdx++; return; }

    // ---- 11. メトリクスは元のフル target で評価 (Shift+P と同じ) ----
    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== QuadCyclic Complete  RMSE=" << registrationHandle.compRmse
              << "  mask=" << LiverLeftRightLabel::quadrantMaskString(med.mask)
              << "  (cyclic chamfer was " << bestChamfer
              << ", total " << bestScore << ")"
              << " ===" << std::endl;

    g_callIdx++;
}

// =========================================================
//  QcrCandidate — Subset RANSAC 候補 1 件
// ---------------------------------------------------------
//  Stage 1 で生成された 1 つの仮説。subset (3 sector index) × shift ×
//  mirror で確定する Umeyama T と、その 3 ペアでの内部 RMSE を保持。
//  Stage 2 で full chamfer 評価 + init prior penalty が走り、
//  score_chamfer / displacement / score_total が埋まる。
// =========================================================
struct QcrCandidate {
    glm::mat4 T            = glm::mat4(1.0f);
    float     score_pair   = 1e9f;   // Stage 1: 全 paired sector の T 適用後 RMSE (medoid consensus)
    float     score_chamfer= 1e9f;   // Stage 2: full source × target chamfer RMSE
    float     displacement = 0.0f;   // Stage 2: source 平均移動量 (init pose=identity 基準)
    float     score_total  = 1e9f;   // Stage 2: score_chamfer + λ * displacement (final ranking)
    float     axisRotDeg[3]= {0.0f, 0.0f, 0.0f};  // Stage 2: T 回転成分の Euler 角 (X, Y, Z 軸; deg)
    bool      rotOK        = true;   // Stage 2: axis rot hard limit を通過したか
    int       subset[5]    = {-1, -1, -1, -1, -1};  // 採用 sector index (source 側; 先頭 K 個が有効)
    int       K_used       = 3;      // この候補の K (3/4/5)
    int       shift        = 0;
    int       mirror       = 0;      // 0=forward(CW), 1=reverse(CCW)
};

// QcrSubsetKey: kSectors の中の K sector を昇順で保持。
// K=3 のときは [3]/[4] = -1、K=4 のときは [4] = -1。
struct QcrSubsetKey { int idx[5]; int K; };  // 0 <= idx[0] < idx[1] < ... < idx[K-1] < kSectors

// =========================================================
//  getValidSubsetsK3/4/5 — minSpread 制約を満たす K sector 部分集合の列挙
// ---------------------------------------------------------
//  kSectors セクターの中から K つを選ぶ組合せ C(kSectors, K) のうち、
//  「隣接 K 対 (cyclic 含む) の循環距離すべてが minSpread 以上」の
//  ものだけ返す。K sector を分散させて Umeyama の数値安定性を確保。
//
//  Note: 厳密には「全 C(K,2) ペア」ではなく「ソート後の隣接 K ペア」で
//  チェックすればよい (循環距離が連結 → 非隣接ペアは隣接ペアの和)。
//  K=3 ならどちらも同じ。K=4/5 では隣接 only で十分。
//
//  典型値 (kSectors=24, minSpread=4):
//    K=3 → ~770 組 (C(24,3)=2024 中 ~38%)
//    K=4 → ~1800 組 (C(24,4)=10626 中 ~17%)
//    K=5 → ~2400 組 (C(24,5)=42504 中 ~6%)
//
//  ビルドは K=5 でも 1ms 以下。同じパラメータの 2 回目以降は cache。
//  K ごとに別の static cache を持つ。
// =========================================================

// minSpread の cyclic adjacency check: idx[0..K-1] が昇順前提。
// cyclic 隣接距離 = (idx[1]-idx[0]), (idx[2]-idx[1]), ..., (idx[K-1]-idx[K-2]),
//                   (kSectors - idx[K-1] + idx[0])
inline bool checkMinSpreadCyclic(const int* idx, int K, int kSectors, int minSpread) {
    int minD = kSectors + 1;
    for (int p = 0; p < K - 1; p++) {
        const int d = idx[p+1] - idx[p];
        if (d < minD) minD = d;
    }
    const int dWrap = kSectors - idx[K-1] + idx[0];
    if (dWrap < minD) minD = dWrap;
    return minD >= minSpread;
}

inline const std::vector<QcrSubsetKey>&
getValidSubsetsK3(int kSectors, int minSpread) {
    static int                       cached_kSectors  = -1;
    static int                       cached_minSpread = -1;
    static std::vector<QcrSubsetKey> cached_subsets;
    if (cached_kSectors == kSectors && cached_minSpread == minSpread) {
        return cached_subsets;
    }
    cached_subsets.clear();
    cached_subsets.reserve(2048);
    for (int i = 0; i <= kSectors - 3; i++) {
        for (int j = i + 1; j <= kSectors - 2; j++) {
            for (int k = j + 1; k <= kSectors - 1; k++) {
                int idx[3] = {i, j, k};
                if (checkMinSpreadCyclic(idx, 3, kSectors, minSpread)) {
                    QcrSubsetKey s;
                    s.K = 3;
                    s.idx[0] = i; s.idx[1] = j; s.idx[2] = k;
                    s.idx[3] = -1; s.idx[4] = -1;
                    cached_subsets.push_back(s);
                }
            }
        }
    }
    cached_kSectors  = kSectors;
    cached_minSpread = minSpread;
    return cached_subsets;
}

inline const std::vector<QcrSubsetKey>&
getValidSubsetsK4(int kSectors, int minSpread) {
    static int                       cached_kSectors  = -1;
    static int                       cached_minSpread = -1;
    static std::vector<QcrSubsetKey> cached_subsets;
    if (cached_kSectors == kSectors && cached_minSpread == minSpread) {
        return cached_subsets;
    }
    cached_subsets.clear();
    cached_subsets.reserve(4096);
    for (int i = 0; i <= kSectors - 4; i++) {
        for (int j = i + 1; j <= kSectors - 3; j++) {
            for (int k = j + 1; k <= kSectors - 2; k++) {
                for (int l = k + 1; l <= kSectors - 1; l++) {
                    int idx[4] = {i, j, k, l};
                    if (checkMinSpreadCyclic(idx, 4, kSectors, minSpread)) {
                        QcrSubsetKey s;
                        s.K = 4;
                        s.idx[0] = i; s.idx[1] = j; s.idx[2] = k; s.idx[3] = l;
                        s.idx[4] = -1;
                        cached_subsets.push_back(s);
                    }
                }
            }
        }
    }
    cached_kSectors  = kSectors;
    cached_minSpread = minSpread;
    return cached_subsets;
}

inline const std::vector<QcrSubsetKey>&
getValidSubsetsK5(int kSectors, int minSpread) {
    static int                       cached_kSectors  = -1;
    static int                       cached_minSpread = -1;
    static std::vector<QcrSubsetKey> cached_subsets;
    if (cached_kSectors == kSectors && cached_minSpread == minSpread) {
        return cached_subsets;
    }
    cached_subsets.clear();
    cached_subsets.reserve(8192);
    for (int i = 0; i <= kSectors - 5; i++) {
        for (int j = i + 1; j <= kSectors - 4; j++) {
            for (int k = j + 1; k <= kSectors - 3; k++) {
                for (int l = k + 1; l <= kSectors - 2; l++) {
                    for (int m = l + 1; m <= kSectors - 1; m++) {
                        int idx[5] = {i, j, k, l, m};
                        if (checkMinSpreadCyclic(idx, 5, kSectors, minSpread)) {
                            QcrSubsetKey s;
                            s.K = 5;
                            s.idx[0] = i; s.idx[1] = j; s.idx[2] = k;
                            s.idx[3] = l; s.idx[4] = m;
                            cached_subsets.push_back(s);
                        }
                    }
                }
            }
        }
    }
    cached_kSectors  = kSectors;
    cached_minSpread = minSpread;
    return cached_subsets;
}

// =========================================================
//  QuadCyclic-RANSAC Registration (Shift+Ctrl+P)
// ---------------------------------------------------------
//  Ctrl+P (runQuadCyclic) と同じ前処理 (AR ∩ silh ∩ quad → 24 sector
//  medoid) を使うが、matching を「24 sector 全部対 cyclic shift × mirror
//  = 48 パターン」ではなく「K=3 minimum sample subset RANSAC」に
//  置き換える:
//
//    Stage 1: ~770 valid_subsets × 24 shift × 2 mirror ≈ 37K trials
//             各 trial:
//               (a) 3 ペアで Umeyama → T
//               (b) Pairing chamfer score (同じ shift/mirror で paired
//                   17-24 sector 全部の T 適用後 RMSE) を inlier
//                   consensus として計算
//             → top-K (=20) 候補を保持 (sorted insert by score_pair)
//    Stage 2: top-K を full chamfer RMSE で再評価 → best 確定
//    Stage 3-6: bestT を organ に適用 → ICP 精錬 → metrics → poseSave
//
//  動機:
//    Ctrl+P (全 24 sector 巡回シフト) は「全 sector が rim 上に乗って
//    いる」前提が崩れた瞬間 (一部 sector が肝臓内部に逸れた場合) に
//    全試行が汚染される (= Umeyama の最小二乗が外れ点に引っ張られる)。
//    Subset RANSAC は 3 sector だけ使うので、24 sector のうち良い 3 つ
//    が見つかれば inlier として吸い上げられる。
//
//    [v1.1] Stage 1 scoring を「K=3 内部残差」から「全 paired sector の
//    chamfer」に変更。K=3 内部残差は 3 点が co-linear かどうかしか判定
//    できず、top-K がほぼランダムに選ばれていたため (実測: 全 trial で
//    score_inner=0.002〜0.04 でほぼ無差別)。新 scoring は古典的 RANSAC
//    の inlier consensus を連続距離で実装したもの — 3 点で T を作って
//    残り 14-21 inlier 候補で確認するスキーム。
//
//    2 段階評価 (pairing chamfer → full chamfer) は、medoid 上では
//    良く見えても source 全点で評価すると外れる candidate を Stage 2
//    で弾くため。medoid だけだと 17-24 点の表現でしかないので、
//    Stage 2 で 100-1400 source silh 点 vs 41K target boundary の
//    full chamfer に上げて精度を上げる。チャット5-7 V3-3 案 D'
//    (top-3 screening) と同じ哲学。
//
//  v1 ハイパラ (globals 経由、将来 UI 露出予定):
//    g_qcrSubsetK         = 3     K=3 minimum sample (Fischler-Bolles 標準)
//    g_qcrMinSpreadSec    = 4     採用 3 sector の角度間隔下限 (=60°)
//    g_qcrTopKCandidates  = 20    Stage 1 → Stage 2 で残す候補数
//
//  byte-identical 保証なし: Ctrl+P とは matching algorithm が異なる。
//  ただし前処理 (extractQuadCyclicMedoids) は完全共通なので、source/
//  target 点群 / sector medoid は Ctrl+P と bit-identical。
//
//  Shift+B 可視化:
//    採用 3 sector のみ g_cyclicPairValid[i]=1 で書き込み、他 21 sector
//    は valid=0。Shift+P / Ctrl+P と同じ viz loop が 3 つだけ描画する。
// =========================================================
inline void runQuadCyclicRansac(bool lock_scale = false, bool single_mesh_only = false) {
    std::cout << "\n=== QuadCyclic-RANSAC Registration (Shift+Ctrl+P)"
              << (lock_scale ? "  [6-DoF rigid]" : "  [7-DoF T+R+S]")
              << (single_mesh_only ? "  [single-mesh: liver only]" : "")
              << " ===" << std::endl;

    // Phase 1: シード固定 (Ctrl+P と同じ)
    const uint32_t fgr_seed = g_trialSeed + g_callIdx;
    Reg3DCustom::setFgrSeed(fgr_seed);
    std::cout << "[Seed] FGR=" << fgr_seed
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;

    // ---- Step 1-7: 共通前処理 (Ctrl+P と同じヘルパー) ----
    QuadCyclicMedoids med = extractQuadCyclicMedoids("Shift+Ctrl+P");
    if (!med.ok) { g_callIdx++; return; }

    // Chamfer 評価用 KD tree (med.targetBdyPts は move しないこと)
    Reg3DCustom::NanoflannAdaptor tgtAdaptor(med.targetBdyPts);
    auto tgtTree = Reg3DCustom::buildKDTree(tgtAdaptor);

    auto chamferRMSE = [&](const glm::mat4& T) -> float {
        float sumSq = 0.0f;
        int   cnt   = 0;
        for (const auto& p : med.sourceSilhPts) {
            glm::vec4 v(p, 1.0f);
            glm::vec3 tp = glm::vec3(T * v);
            size_t nn; float d2;
            if (Reg3DCustom::searchKNN1(*tgtTree, tp, nn, d2)) {
                sumSq += d2;
                cnt++;
            }
        }
        if (cnt == 0) return 1e9f;
        return std::sqrt(sumSq / float(cnt));
    };

    // ---- 8. Subset RANSAC ----
    const int   kSectors  = med.kSectors;
    const float kScaleLo  = med.kScaleLo;
    const float kScaleHi  = med.kScaleHi;
    int         K         = std::max(3, std::min(5, g_qcrSubsetK));  // clamp to [3,5]
    const int   minSpread = std::max(1, g_qcrMinSpreadSec);
    const int   topKCap   = std::max(1, g_qcrTopKCandidates);
    const int   maxTrials = std::max(1000, g_qcrMaxTrials);  // floor 1000

    if (g_qcrSubsetK != K) {
        std::cerr << "[Shift+Ctrl+P] K=" << g_qcrSubsetK
                  << " out of supported range [3,5]; clamped to K=" << K << "." << std::endl;
    }

    // K に応じて subset list を取得 (cached)
    const std::vector<QcrSubsetKey>* subsetsPtr = nullptr;
    if      (K == 3) subsetsPtr = &getValidSubsetsK3(kSectors, minSpread);
    else if (K == 4) subsetsPtr = &getValidSubsetsK4(kSectors, minSpread);
    else             subsetsPtr = &getValidSubsetsK5(kSectors, minSpread);
    const auto& subsets = *subsetsPtr;

    // Trial cap: 全部回すと K=4/5 で 100K-400K trials になり遅い。
    // cap を超える場合は trial 全体に均等な stride を入れて間引く。
    // stride=1 なら全列挙、stride=N なら 1/N に間引き。
    const long long totalTrials = (long long)subsets.size() * (long long)kSectors * 2LL;
    long long stride = 1;
    if (totalTrials > (long long)maxTrials) {
        stride = (totalTrials + (long long)maxTrials - 1) / (long long)maxTrials;
    }
    const long long plannedTrials = (totalTrials + stride - 1) / stride;
    std::cout << "[QCR] valid_subsets=" << subsets.size()
              << "  (kSectors=" << kSectors
              << ", K=" << K << ", MIN_SPREAD=" << minSpread << " sec)"
              << "  trials_total=" << totalTrials
              << "  trials_planned=" << plannedTrials
              << "  stride=" << stride
              << "  (cap=" << maxTrials << ")" << std::endl;

    if (subsets.empty()) {
        std::cerr << "[Shift+Ctrl+P] No valid subsets (minSpread="
                  << minSpread << " too strict?) -- aborting" << std::endl;
        g_callIdx++;
        return;
    }

    // Stage 1: ~37K trials, keep top-K by inner RMSE
    auto stage1_t0 = std::chrono::steady_clock::now();
    std::vector<QcrCandidate> topK;
    topK.reserve(topKCap + 1);

    auto cmpInner = [](const QcrCandidate& a, const QcrCandidate& b) {
        return a.score_pair < b.score_pair;
    };

    // ---- v1.3 shared helpers (used in both Stage 1 and Stage 2) ----
    //   maxRotDeg, lambdaS1Disp / lambdaDisp はそれぞれ Stage 1 / Stage 2 用。
    //   displacementOf  : full silh∩quad 点群での平均移動量 (Stage 2 用)
    //   displacementMedoid: 22 sector medoid 点群での平均移動量 (Stage 1 用、軽量)
    //   extractAxisRotDeg: T からスケール除去 → R 抽出 → X-Y-Z Euler (deg)
    //
    //   Stage 1 では medoid displacement (~22 点) を使うことで Stage 1 の
    //   コストを最小限に抑えつつ init prior を効かせる。Stage 2 では
    //   silh∩quad 全点 (100-400点) で正確に評価。
    const float maxRotDeg    = std::max(1.0f, g_qcrMaxAxisRotDeg);
    const float maxTotalRotDeg = std::max(1.0f, g_qcrMaxTotalRotDeg);
    const float lambdaS1Disp = std::max(0.0f, g_qcrStage1DispWeight);
    const float lambdaDisp   = std::max(0.0f, g_qcrInitDispWeight);
    const float rad2deg      = 57.2957795f;  // 180/π

    auto displacementOf = [&](const glm::mat4& T) -> float {
        // mean over silh∩quad source points of ||T·p - p||
        if (med.sourceSilhPts.empty()) return 0.0f;
        double sumD = 0.0;
        for (const auto& p : med.sourceSilhPts) {
            const glm::vec3 tp = glm::vec3(T * glm::vec4(p, 1.0f));
            const glm::vec3 d  = tp - p;
            sumD += std::sqrt((double)glm::dot(d, d));
        }
        return float(sumD / double(med.sourceSilhPts.size()));
    };

    auto displacementMedoid = [&](const glm::mat4& T) -> float {
        // mean over up-to-22 medoid points of ||T·p - p||。
        // Stage 1 で各 trial ごとに呼ばれるので、ループは src medoid のみ
        // (med.srcMedoid[i] >= 0 のもの)。
        double sumD = 0.0;
        int    cnt  = 0;
        for (int i = 0; i < kSectors; i++) {
            if (med.srcMedoid[i] < 0) continue;
            const glm::vec3& p = med.sourceSilhPts[med.srcMedoid[i]];
            const glm::vec3  tp = glm::vec3(T * glm::vec4(p, 1.0f));
            const glm::vec3  d  = tp - p;
            sumD += std::sqrt((double)glm::dot(d, d));
            cnt++;
        }
        if (cnt == 0) return 0.0f;
        return float(sumD / double(cnt));
    };

    auto extractAxisRotDeg = [&](const glm::mat4& T, float out[3]) {
        // T = scale * R * (translate) なので、まず column 長で scale を割って R を抽出。
        glm::vec3 c0(T[0]), c1(T[1]), c2(T[2]);
        const float s0 = glm::length(c0);
        const float s1 = glm::length(c1);
        const float s2 = glm::length(c2);
        if (s0 < 1e-6f || s1 < 1e-6f || s2 < 1e-6f) {
            out[0] = out[1] = out[2] = 0.0f; return;
        }
        c0 /= s0; c1 /= s1; c2 /= s2;
        // R は column-major: c0=R[:,0], c1=R[:,1], c2=R[:,2]。
        // Euler X-Y-Z (intrinsic): sy = -R[2][0] = -c0.z
        const float sy = -c0.z;
        float ax, ay, az;
        if (std::abs(sy) < 0.99999f) {
            ay = std::asin(sy);
            ax = std::atan2(c1.z, c2.z);
            az = std::atan2(c0.y, c0.x);
        } else {
            ay = (sy > 0.0f ? 1.57079632f : -1.57079632f);
            ax = std::atan2(-c2.y, c1.y);
            az = 0.0f;
        }
        auto wrap = [](float a) {
            while (a >  3.14159265f) a -= 6.28318530f;
            while (a < -3.14159265f) a += 6.28318530f;
            return a;
        };
        out[0] = wrap(ax) * rad2deg;
        out[1] = wrap(ay) * rad2deg;
        out[2] = wrap(az) * rad2deg;
    };
    // axis-angle total rotation (deg): θ = arccos((trace(R)-1)/2)
    //   per-axis では off-axis 大回転を見逃すケースがあるため、こちらを併用。
    auto extractTotalRotDeg = [&](const glm::mat4& T) -> float {
        glm::vec3 c0(T[0]), c1(T[1]), c2(T[2]);
        const float s0 = glm::length(c0);
        const float s1 = glm::length(c1);
        const float s2 = glm::length(c2);
        if (s0 < 1e-6f || s1 < 1e-6f || s2 < 1e-6f) return 0.0f;
        // unit basis の trace (= c0.x/s0 + c1.y/s1 + c2.z/s2)
        const float trace = c0.x/s0 + c1.y/s1 + c2.z/s2;
        float cosA = (trace - 1.0f) * 0.5f;
        if (cosA >  1.0f) cosA =  1.0f;
        if (cosA < -1.0f) cosA = -1.0f;
        return std::acos(cosA) * rad2deg;
    };

    int triedCount = 0, validCount = 0;
    int rotFilteredS1 = 0;  // Stage 1 で axis rotation hard limit に弾かれた数

    // ---- Stage 1: flat enumeration with stride sampling ----
    //   trials are indexed as (subset_idx, shift, mirror) → t = subset_idx * 2*kSectors + shift * 2 + mirror
    //   stride サンプリング: t = 0, stride, 2*stride, ... を踏む。
    //   stride=1 なら全列挙。K=4/5 で totalTrials が cap を超える場合のみ stride > 1。
    const long long shiftMirrorPerSubset = (long long)kSectors * 2LL;
    for (long long t = 0; t < totalTrials; t += stride) {
        const long long sub_idx = t / shiftMirrorPerSubset;
        const long long rem     = t - sub_idx * shiftMirrorPerSubset;
        const int       shift   = (int)(rem / 2);
        const int       mirror  = (int)(rem & 1LL);
        const auto&     sub     = subsets[(size_t)sub_idx];

        triedCount++;

        // K ペアの (src, tgt) を sector index から引く
        glm::vec3 srcPts[5], tgtPts[5];
        bool valid = true;
        for (int p = 0; p < K; p++) {
            const int si = sub.idx[p];
            int tj;
            if (mirror == 0) {
                tj = (si + shift) % kSectors;
            } else {
                tj = ((kSectors - 1 - si) + shift) % kSectors;
                if (tj < 0) tj += kSectors;
            }
            if (med.srcMedoid[si] < 0 || med.tgtMedoid[tj] < 0) {
                valid = false; break;
            }
            srcPts[p] = med.sourceSilhPts[med.srcMedoid[si]];
            tgtPts[p] = med.targetBdyPts [med.tgtMedoid[tj]];
        }
        if (!valid) continue;

        // Umeyama (K=3 は exact fit、K=4/5 は over-determined。退化は sanity でガード)
        std::vector<glm::vec3> srcVec(srcPts, srcPts + K);
        std::vector<glm::vec3> tgtVec(tgtPts, tgtPts + K);
        glm::mat4 T = Reg3D::UmeyamaRegistration(srcVec, tgtVec);

        bool finiteT = true;
        for (int c = 0; c < 4 && finiteT; c++)
            for (int r = 0; r < 4 && finiteT; r++)
                if (!std::isfinite(T[c][r])) finiteT = false;
        if (!finiteT) continue;

        float scale = glm::length(glm::vec3(T[0]));
        if (!std::isfinite(scale) || scale < kScaleLo || scale > kScaleHi) continue;

        // ---- v1.3 (A1): Stage 1 axis rotation hard limit ----
        float axisDeg[3];
        extractAxisRotDeg(T, axisDeg);
        const float maxAxisS1 = std::max({std::abs(axisDeg[0]),
                                          std::abs(axisDeg[1]),
                                          std::abs(axisDeg[2])});
        if (maxAxisS1 > maxRotDeg) {
            rotFilteredS1++;
            continue;
        }
        // ---- (A1b): total axis-angle 回転量による追加ガード (新規) ----
        //   per-axis では off-axis (e.g. (1,1,1) 軸まわり大回転) を捉えられない
        //   ケースがあるため、rotation matrix の総回転量を閾値以下に縛る。
        const float totalDegS1 = extractTotalRotDeg(T);
        if (totalDegS1 > maxTotalRotDeg) {
            rotFilteredS1++;
            continue;
        }

        // Pairing chamfer (medoid consensus): 全 sector ペアで T 適用後残差
        float sumSq = 0.0f;
        int   cntPairs = 0;
        for (int i = 0; i < kSectors; i++) {
            if (med.srcMedoid[i] < 0) continue;
            int jj;
            if (mirror == 0) {
                jj = (i + shift) % kSectors;
            } else {
                jj = ((kSectors - 1 - i) + shift) % kSectors;
                if (jj < 0) jj += kSectors;
            }
            if (med.tgtMedoid[jj] < 0) continue;
            const glm::vec3& sp  = med.sourceSilhPts[med.srcMedoid[i]];
            const glm::vec3& tgp = med.targetBdyPts [med.tgtMedoid[jj]];
            const glm::vec3  tsp = glm::vec3(T * glm::vec4(sp, 1.0f));
            const float dx = tsp.x - tgp.x;
            const float dy = tsp.y - tgp.y;
            const float dz = tsp.z - tgp.z;
            sumSq += dx*dx + dy*dy + dz*dz;
            cntPairs++;
        }
        if (cntPairs < 3) continue;
        const float score_pair = std::sqrt(sumSq / float(cntPairs));

        // ---- v1.3 (B1): Stage 1 light displacement penalty ----
        const float dispS1 = displacementMedoid(T);
        const float score_pair_total = score_pair + lambdaS1Disp * dispS1;
        validCount++;

        // top-K に挿入 (sorted vector, ascending score_pair_total)
        if ((int)topK.size() < topKCap ||
            score_pair_total < topK.back().score_pair)
        {
            QcrCandidate c;
            c.T             = T;
            c.score_pair    = score_pair_total;
            c.axisRotDeg[0] = axisDeg[0];
            c.axisRotDeg[1] = axisDeg[1];
            c.axisRotDeg[2] = axisDeg[2];
            c.rotOK         = true;
            for (int p = 0; p < K; p++)  c.subset[p] = sub.idx[p];
            for (int p = K; p < 5; p++)  c.subset[p] = -1;
            c.K_used        = K;
            c.shift         = shift;
            c.mirror        = mirror;
            auto pos = std::upper_bound(topK.begin(), topK.end(), c, cmpInner);
            topK.insert(pos, c);
            if ((int)topK.size() > topKCap) topK.pop_back();
        }
    }
    auto stage1_t1 = std::chrono::steady_clock::now();
    const double stage1_ms =
        std::chrono::duration<double, std::milli>(stage1_t1 - stage1_t0).count();

    std::cout << "[QCR] Stage 1: tried=" << triedCount
              << "  rot_filtered=" << rotFilteredS1
              << "  valid=" << validCount
              << "  topK=" << topK.size()
              << "  lambda_s1=" << std::fixed << std::setprecision(3) << lambdaS1Disp
              << "  max_rot=" << std::setprecision(1) << maxRotDeg << "deg"
              << "  max_total=" << std::setprecision(1) << maxTotalRotDeg << "deg"
              << "  (" << std::setprecision(1) << stage1_ms << " ms)"
              << std::defaultfloat << std::setprecision(6) << std::endl;

    if (topK.empty()) {
        std::cerr << "[Shift+Ctrl+P] No valid Stage 1 candidates -- aborting"
                  << std::endl;
        g_callIdx++;
        return;
    }

    // ---- Stage 2: top-K を full chamfer + init prior penalty で再評価 ----
    //   Stage 1 で axis rotation hard limit (v1.3) と medoid displacement
    //   penalty (v1.3) を通過した候補に対し、ここでは full silh∩quad
    //   chamfer + full silh∩quad displacement で精密ランキング。
    //   v1.3 では Stage 1 で大半の鏡像解が排除されているはずなので、
    //   rot_rejected はほぼ 0 になる想定 (Stage 1 と Stage 2 で同じ
    //   threshold を使うため)。lambdaS1Disp / lambdaDisp は別パラメータ
    //   (Stage 1 は軽量 medoid, Stage 2 は full)。
    auto stage2_t0 = std::chrono::steady_clock::now();

    int   bestIdx     = -1;
    float bestTotal   = 1e9f;
    int   nRotRejected = 0;
    for (size_t i = 0; i < topK.size(); i++) {
        auto& c = topK[i];
        // (A) Axis rotation hard limit (per-axis + total axis-angle)
        extractAxisRotDeg(c.T, c.axisRotDeg);
        const float maxAxis = std::max({std::abs(c.axisRotDeg[0]),
                                        std::abs(c.axisRotDeg[1]),
                                        std::abs(c.axisRotDeg[2])});
        const float totalAxis = extractTotalRotDeg(c.T);
        c.rotOK = (maxAxis <= maxRotDeg) && (totalAxis <= maxTotalRotDeg);
        // (B) Chamfer (常に計算; 後段の log 用)
        c.score_chamfer = chamferRMSE(c.T);
        // (B') Displacement
        c.displacement  = displacementOf(c.T);
        // (C) Total score (rejected candidate は +∞ にする)
        if (c.rotOK) {
            c.score_total = c.score_chamfer + lambdaDisp * c.displacement;
        } else {
            c.score_total = 1e9f;
            nRotRejected++;
        }
        if (c.score_total < bestTotal) {
            bestTotal = c.score_total;
            bestIdx   = (int)i;
        }
    }

    auto stage2_t1 = std::chrono::steady_clock::now();
    const double stage2_ms =
        std::chrono::duration<double, std::milli>(stage2_t1 - stage2_t0).count();

    std::cout << "[QCR] Stage 2: " << topK.size() << " candidates eval"
              << "  rot_rejected=" << nRotRejected
              << "  lambda_disp=" << std::fixed << std::setprecision(3) << lambdaDisp
              << "  max_axis_rot=" << std::setprecision(1) << maxRotDeg << "deg"
              << "  max_total_rot=" << std::setprecision(1) << maxTotalRotDeg << "deg"
              << "  (" << std::setprecision(1) << stage2_ms << " ms)"
              << std::defaultfloat << std::setprecision(6) << std::endl;

    // Stage 2 が全 reject されたら fallback: hard limit を無視して
    // score_chamfer のみで best を選ぶ (緊急脱出)。
    if (bestIdx < 0) {
        std::cerr << "[QCR] All " << topK.size() << " candidates rejected by "
                     "axis rotation limit (per-axis=" << maxRotDeg
                  << "deg, total=" << maxTotalRotDeg
                  << "deg). Falling back to chamfer-only ranking." << std::endl;
        float bestCh = 1e9f;
        for (size_t i = 0; i < topK.size(); i++) {
            if (topK[i].score_chamfer < bestCh) {
                bestCh  = topK[i].score_chamfer;
                bestIdx = (int)i;
            }
        }
        if (bestIdx < 0) bestIdx = 0;  // 最終保険
    }

    const QcrCandidate& bestC = topK[bestIdx];
    const glm::mat4& bestT_initial = bestC.T;
    const float bestScore   = bestC.score_chamfer;

    // ---- v1.4 (Inlier Refinement): K=3 exact-fit → over-determined Umeyama ----
    //   Plan A の核心。3 点 Umeyama は exact fit なので 3 点ノイズが T に
    //   直接乗る → ICP refinement で diverge する。Ctrl+P が 22 ペア
    //   over-determined fit で安定してるのを、RANSAC で選んだ best T を
    //   起点に再現する:
    //
    //     1. bestC.T を 24 sector medoid 全ペアに適用
    //     2. 各ペアで residual ||T·src - tgt|| を計算
    //     3. residual <= threshold な inlier だけ抽出
    //     4. inlier ≥ 4 点なら Umeyama 再計算 → T_refined
    //     5. T_refined が sane (rot < 90°, scale in [0.4, 2.5]) なら採用
    //     6. 失敗時は bestC.T を fallback
    //
    //   inlier threshold は bestC.score_pair (Stage 1 score) × 1.5 or
    //   median residual × 2 のうち大きい方。鏡像解はすでに Stage 1 で
    //   排除されているので、ここでは「良い T の周辺の微調整」になる。
    //   3 → ~15-22 ペアで fit するので Ctrl+P と同等の安定性が期待できる。
    glm::mat4 bestT     = bestT_initial;
    bool refineApplied  = false;
    float refineChamfer = bestScore;
    float refineDisp    = bestC.displacement;
    float refineRot[3]  = {bestC.axisRotDeg[0], bestC.axisRotDeg[1], bestC.axisRotDeg[2]};
    int   nInliers      = 0;
    float inlierThr     = 0.0f;
    float medianResid   = 0.0f;
    do {
        // (1) (2) 全 sector ペアで T_initial 適用後の residual を収集
        std::vector<std::pair<float, int>> residSorted;  // (residual, sector_i)
        residSorted.reserve(kSectors);
        for (int i = 0; i < kSectors; i++) {
            if (med.srcMedoid[i] < 0) continue;
            int jj;
            if (bestC.mirror == 0) {
                jj = (i + bestC.shift) % kSectors;
            } else {
                jj = ((kSectors - 1 - i) + bestC.shift) % kSectors;
                if (jj < 0) jj += kSectors;
            }
            if (med.tgtMedoid[jj] < 0) continue;
            const glm::vec3& sp  = med.sourceSilhPts[med.srcMedoid[i]];
            const glm::vec3& tgp = med.targetBdyPts [med.tgtMedoid[jj]];
            const glm::vec3  tsp = glm::vec3(bestT_initial * glm::vec4(sp, 1.0f));
            const float dx = tsp.x - tgp.x;
            const float dy = tsp.y - tgp.y;
            const float dz = tsp.z - tgp.z;
            const float r  = std::sqrt(dx*dx + dy*dy + dz*dz);
            residSorted.emplace_back(r, i);
        }
        if ((int)residSorted.size() < 4) break;  // 不足: refinement 不可

        // (3) threshold: max(median × 2, score_pair × 1.5)
        std::vector<float> residOnly;
        residOnly.reserve(residSorted.size());
        for (const auto& p : residSorted) residOnly.push_back(p.first);
        std::sort(residOnly.begin(), residOnly.end());
        medianResid = residOnly[residOnly.size() / 2];
        inlierThr   = std::max(medianResid * 2.0f, bestC.score_pair * 1.5f);

        // (4) inlier 点群収集
        std::vector<glm::vec3> srcIn, tgtIn;
        srcIn.reserve(residSorted.size());
        tgtIn.reserve(residSorted.size());
        for (const auto& pr : residSorted) {
            if (pr.first > inlierThr) continue;
            const int i  = pr.second;
            int jj;
            if (bestC.mirror == 0) {
                jj = (i + bestC.shift) % kSectors;
            } else {
                jj = ((kSectors - 1 - i) + bestC.shift) % kSectors;
                if (jj < 0) jj += kSectors;
            }
            srcIn.push_back(med.sourceSilhPts[med.srcMedoid[i]]);
            tgtIn.push_back(med.targetBdyPts [med.tgtMedoid[jj]]);
        }
        nInliers = (int)srcIn.size();
        if (nInliers < 4) break;  // 不足: refinement 不可

        // (5) Umeyama 再計算 (over-determined fit)
        glm::mat4 T_refined = Reg3D::UmeyamaRegistration(srcIn, tgtIn);
        bool finiteT = true;
        for (int c = 0; c < 4 && finiteT; c++)
            for (int r = 0; r < 4 && finiteT; r++)
                if (!std::isfinite(T_refined[c][r])) finiteT = false;
        if (!finiteT) break;

        // (6) sanity: scale, axis rotation (per-axis + total axis-angle)
        const float refScale = glm::length(glm::vec3(T_refined[0]));
        if (!std::isfinite(refScale) || refScale < kScaleLo || refScale > kScaleHi) break;
        float refDeg[3];
        extractAxisRotDeg(T_refined, refDeg);
        const float refMaxAxis = std::max({std::abs(refDeg[0]),
                                           std::abs(refDeg[1]),
                                           std::abs(refDeg[2])});
        if (refMaxAxis > maxRotDeg) break;  // 暴走: refinement 結果が per-axis hard limit 違反
        const float refTotalAxis = extractTotalRotDeg(T_refined);
        if (refTotalAxis > maxTotalRotDeg) break;  // 暴走: total axis-angle hard limit 違反

        // (7) refined T の chamfer / disp 計算
        const float refCham = chamferRMSE(T_refined);
        const float refDisp = displacementOf(T_refined);

        // 採用基準 (v1.5 案 A): chamfer 単独ではなく total (= chamfer + λ × disp)
        //   が improve しないと reject。v1.4 までは chamfer ≤ bestScore × 1.05 だけ
        //   見ていたが、ログから「refinement で chamfer は微改善 (≤2%) なのに
        //   disp が大きく悪化 (+29 %～+110 %)」する failure mode が観測された。
        //   inlier consensus 上の least-squares が「init から遠い真の boundary fit」
        //   に引きずられて起こる症状で、ICP refinement の収束性をかえって悪化させる。
        //
        //   v1.5: Stage 2 と同じ score_total = chamfer + λ × displacement を採用
        //   基準に使う → 「init pose から離れる refinement」は reject。
        //   λ は Stage 2 と同じ g_qcrInitDispWeight (lambdaDisp) を共有。
        //   厳密 improve だけ要求 (slack 0%) — 微妙な改善で disp が暴走するのを防ぐ。
        const float initTotal = bestScore + lambdaDisp * bestC.displacement;
        const float refTotal  = refCham   + lambdaDisp * refDisp;
        const bool acceptOK   = (refTotal <= initTotal);
        if (!acceptOK) break;

        // 採用
        bestT          = T_refined;
        refineApplied  = true;
        refineChamfer  = refCham;
        refineDisp     = refDisp;
        refineRot[0]   = refDeg[0];
        refineRot[1]   = refDeg[1];
        refineRot[2]   = refDeg[2];
    } while (false);

    // ---- 詳細 best ログ (precision を明示) ----
    std::cout << std::fixed << std::setprecision(4)
              << "[QCR] Best: K=" << bestC.K_used << "  subset={";
    for (int p = 0; p < bestC.K_used; p++) {
        if (p > 0) std::cout << ",";
        std::cout << bestC.subset[p];
    }
    std::cout << "}"
              << "  shift=" << bestC.shift
              << "  dir=" << (bestC.mirror ? "reverse(CCW)" : "forward(CW)")
              << "  score_s1=" << bestC.score_pair
              << "  chamfer=" << bestC.score_chamfer
              << "  disp=" << bestC.displacement
              << "  total=" << bestC.score_total
              << "  rot=[" << std::setprecision(1)
              << bestC.axisRotDeg[0] << ","
              << bestC.axisRotDeg[1] << ","
              << bestC.axisRotDeg[2] << "]deg"
              << std::defaultfloat << std::setprecision(6) << std::endl;

    // ---- v1.4 refinement 結果ログ ----
    if (refineApplied) {
        std::cout << std::fixed << std::setprecision(4)
                  << "[QCR] Refine: APPLIED  inliers=" << nInliers << "/" << kSectors
                  << "  thr=" << inlierThr
                  << "  (median_resid=" << medianResid << ")"
                  << "  chamfer: " << bestScore << " -> " << refineChamfer
                  << "  disp: " << bestC.displacement << " -> " << refineDisp
                  << "  rot=[" << std::setprecision(1)
                  << refineRot[0] << "," << refineRot[1] << "," << refineRot[2] << "]deg"
                  << std::defaultfloat << std::setprecision(6) << std::endl;
    } else {
        std::cout << std::fixed << std::setprecision(4)
                  << "[QCR] Refine: SKIPPED  (inliers=" << nInliers
                  << ", thr=" << inlierThr
                  << ", median_resid=" << medianResid
                  << ") -> using initial 3-pt T  [reason: total worsened or insane refinement]"
                  << std::defaultfloat << std::setprecision(6) << std::endl;
    }

    // ---- 全 top-K の score 分布 (debug; rejection 状況 + ranking 透明性) ----
    std::cout << std::fixed << std::setprecision(4)
              << "[QCR] top" << topK.size() << " (rank by score_total):" << std::endl;
    // total 昇順インデックス
    std::vector<int> rankIdx(topK.size());
    for (size_t i = 0; i < topK.size(); i++) rankIdx[i] = (int)i;
    std::sort(rankIdx.begin(), rankIdx.end(), [&](int a, int b) {
        return topK[a].score_total < topK[b].score_total;
    });
    const int kShow = std::min((int)topK.size(), 10);
    for (int r = 0; r < kShow; r++) {
        const auto& c = topK[rankIdx[r]];
        std::cout << "    #" << (r+1) << " sub={";
        for (int p = 0; p < c.K_used; p++) {
            if (p > 0) std::cout << ",";
            std::cout << c.subset[p];
        }
        std::cout << "} sh=" << c.shift << "/" << (c.mirror ? "R" : "F")
                  << "  s1=" << c.score_pair
                  << " cham=" << c.score_chamfer
                  << " disp=" << c.displacement
                  << " tot=" << c.score_total
                  << "  rot=[" << std::setprecision(1)
                  << c.axisRotDeg[0] << ","
                  << c.axisRotDeg[1] << ","
                  << c.axisRotDeg[2] << "]"
                  << (c.rotOK ? "" : " [REJECT]")
                  << std::setprecision(4) << std::endl;
    }
    std::cout << std::defaultfloat << std::setprecision(6);

    // ---- 8.5. Shift+B 可視化: 採用 K sector だけ valid=1 ----
    g_cyclicSectors   = kSectors;
    g_cyclicBestShift = bestC.shift;
    g_cyclicBestRev   = bestC.mirror;
    g_cyclicPairSrcVertIdx.assign(kSectors, -1);
    g_cyclicPairTgtPos.assign(kSectors, glm::vec3(0.0f));
    g_cyclicPairValid.assign(kSectors, 0);
    for (int p = 0; p < bestC.K_used; p++) {
        const int si = bestC.subset[p];
        int tj;
        if (bestC.mirror == 0) {
            tj = (si + bestC.shift) % kSectors;
        } else {
            tj = ((kSectors - 1 - si) + bestC.shift) % kSectors;
            if (tj < 0) tj += kSectors;
        }
        if (med.srcMedoid[si] < 0 || med.tgtMedoid[tj] < 0) continue;
        const int srcLocal = med.srcMedoid[si];
        if (srcLocal >= 0 && srcLocal < (int)med.silhouetteIndices.size()) {
            g_cyclicPairSrcVertIdx[si] = (int)med.silhouetteIndices[srcLocal];
        }
        g_cyclicPairTgtPos[si] = med.targetBdyPts[med.tgtMedoid[tj]];
        g_cyclicPairValid[si]  = (g_cyclicPairSrcVertIdx[si] >= 0) ? 1 : 0;
    }
    g_cyclicAvailable = true;

    // ---- 9. bestT を organMeshes に適用 (Ctrl+P と同じ) ----
    //   single_mesh_only=true (AutoQCR loop からの呼び出し) のときは
    //   liver のみ。non-liver organ は loop 終了後の winner replay で
    //   1 回だけ full-organ 適用する。これにより segment (~132K vert) の
    //   ICP iter 内 vertex 変換が 9 trial 中 1 回まで削減される。
    //   下流の performRegistrationSingleMesh は organMeshes 配列を
    //   そのまま受け取って ICP iter ごとに各 mesh を変換するので、
    //   この時点で organs を絞っておけば ICP 全体が liver-only で走る。
    auto organs = single_mesh_only
        ? std::vector<mCutMesh*>{liverMesh3D}
        : getOrganList();
    float estScale = glm::length(glm::vec3(bestT[0]));
    glm::mat3 R3pure = (estScale > 1e-6f) ? glm::mat3(
                                                glm::vec3(bestT[0]) / estScale,
                                                glm::vec3(bestT[1]) / estScale,
                                                glm::vec3(bestT[2]) / estScale
                                                ) : glm::mat3(bestT);

    // ---- 9.5. 6-DoF lock (AutoQCR の checkbox=ON から伝搬) ----
    //   lock_scale=true のとき: bestT から scale を剥がし、estScale=1.0 に上書き。
    //   これで rigid (DICOM mm + metric depth → SE(3)) として扱われ、Ctrl+G を
    //   何度叩いても size_ratio が暴走しない(=論文 defensible)。
    if (lock_scale && estScale > 1e-6f) {
        const float origScale = estScale;
        // bestT を R3pure + translation のみで再構築 (scale 列を unit に)
        glm::mat4 rigid(1.0f);
        rigid[0] = glm::vec4(R3pure[0], 0.0f);
        rigid[1] = glm::vec4(R3pure[1], 0.0f);
        rigid[2] = glm::vec4(R3pure[2], 0.0f);
        rigid[3] = bestT[3];   // translation 列は保持
        bestT    = rigid;
        estScale = 1.0f;
        std::cout << "[Shift+Ctrl+P/6-DoF] Scale lock: estScale "
                  << origScale << " -> 1.000" << std::endl;
    }

    for (auto* m : organs) {
        if (!m) continue;
        for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
            glm::vec4 v(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2], 1.0f);
            v = bestT * v;
            m->mVertices[i]     = v.x;
            m->mVertices[i + 1] = v.y;
            m->mVertices[i + 2] = v.z;
        }
        if (!m->mNormals.empty()) {
            for (size_t i = 0; i + 2 < m->mNormals.size(); i += 3) {
                glm::vec3 nrm(m->mNormals[i], m->mNormals[i+1], m->mNormals[i+2]);
                nrm = glm::normalize(R3pure * nrm);
                m->mNormals[i]     = nrm.x;
                m->mNormals[i + 1] = nrm.y;
                m->mNormals[i + 2] = nrm.z;
            }
        }
        setUp(*m);
    }

    std::cout << "[Shift+Ctrl+P] Applied RANSAC prealign T (scale=" << estScale
              << ")  -> proceeding to ICP refinement" << std::endl;

    // ---- 10. ICP 精錬 (Ctrl+P と同じ filteredTarget swap) ----
    auto filteredTarget = std::make_shared<Reg3DCustom::PointCloud>();
    const bool  copyN         = med.savedFullTarget->hasNormals();
    const bool  hasInst       = med.hasInst;
    const float kBoundaryPxTh = med.kBoundaryPxTh;
    const float instTh        = med.instTh;
    filteredTarget->points.reserve(med.targetBdyPts.size());
    if (copyN)   filteredTarget->normals.reserve(med.targetBdyPts.size());
    filteredTarget->boundaryDist.reserve(med.targetBdyPts.size());
    if (hasInst) filteredTarget->instrumentDist.reserve(med.targetBdyPts.size());

    for (size_t i = 0; i < med.savedFullTarget->size(); i++) {
        float bd  = med.savedFullTarget->boundaryDist[i];
        float idd = hasInst ? med.savedFullTarget->instrumentDist[i] : 9999.0f;
        if (bd < kBoundaryPxTh && idd >= instTh) {
            filteredTarget->points.push_back(med.savedFullTarget->points[i]);
            if (copyN)   filteredTarget->normals.push_back(med.savedFullTarget->normals[i]);
            filteredTarget->boundaryDist.push_back(bd);
            if (hasInst) filteredTarget->instrumentDist.push_back(idd);
        }
    }
    filteredTarget->colors.assign(filteredTarget->points.size(),
                                  glm::vec3(0.0f, 1.0f, 0.0f));

    Reg3DCustom::setCachedTargetCloud(filteredTarget);

    bool ok = true;
    try {
        Reg3DCustom::performRegistrationSingleMesh(
            organs, liverMesh3D, med.silhouetteIndices,
            screenMesh, med.arPos,
            gGridWidth, gGridHeight(),
            1,
            RegRatios::convergence(),
            0.35f,
            !lock_scale, 0.03f,    // estimate_scale: 6-DoF lock 時は false
            RegRatios::zThresh(),
            RegRatios::voxel());
    } catch (const std::exception& e) {
        std::cerr << "[Shift+Ctrl+P] performRegistrationSingleMesh threw: "
                  << e.what() << std::endl;
        ok = false;
    }
    Reg3DCustom::setCachedTargetCloud(med.savedFullTarget);
    if (!ok) { g_callIdx++; return; }

    // ---- 11. メトリクス --------------------------------------------------
    //   [Opt A: AutoQCR fast metrics path]
    //   single_mesh_only=true (AutoQCR loop) のときは computeUnifiedMetrics
    //   (~150-200ms: target 879K × source KD tree KNN + boundary 分類 +
    //   Hausdorff2D) を完全スキップする。代わりに Stage 2 の chamfer を
    //   registrationHandle.compRmse に書き込んで AutoQCR loop の ranking
    //   proxy にする。実 RMSE は winner replay (single_mesh_only=false) で
    //   1 回だけ計算され、最終的な registrationHandle.compRmse は real RMSE
    //   になる。
    //
    //   trade-off: chamfer ranking と real RMSE ranking で winner が変わる
    //   可能性 (top 付近の僅差 trial が入れ替わる)。AutoQCR は「おおまかな
    //   位置合わせ」が用途なので許容。後段 Ctrl+G refinement で最終収束させる
    //   前提。期待短縮: ~150-200ms × 8 trial = ~1.2-1.6 秒。
    const float trialChamfer = refineApplied ? refineChamfer : bestScore;
    if (single_mesh_only) {
        registrationHandle.compRmse        = trialChamfer;  // ranking proxy
        registrationHandle.compAvgError    = trialChamfer;
        registrationHandle.compMaxError    = 0.0f;
        registrationHandle.compCount       = 0;
        registrationHandle.compIoU2D       = 0.0f;
        registrationHandle.sil2DHausdorff  = 0.0f;
        g_metricsValid                     = false;  // 不完全 — winner replay で補完
        std::cout << "[QCR/fast] Skipped computeUnifiedMetrics  "
                  << "(compRmse <- chamfer=" << trialChamfer
                  << " as AutoQCR ranking proxy)" << std::endl;
    } else {
        computeUnifiedMetrics();
        g_metricsValid = true;
    }
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== QuadCyclic-RANSAC Complete  RMSE=" << registrationHandle.compRmse
              << "  mask=" << LiverLeftRightLabel::quadrantMaskString(med.mask)
              << "  (RANSAC chamfer was " << bestScore << ")"
              << " ===" << std::endl;

    // Phase 1 観察用: RANSAC prealign の estScale を publish。
    // AutoQCR の loop 側が consume-and-clear で読み取る。
    g_lastQcrPrealignScale = estScale;
    // [Opt A] AutoQCR の Determinism check 用に best chamfer を publish。
    // fast path / full path 両方とも同じ値が出るべき (Stage 1/2 は決定的)。
    g_lastQcrChamfer       = trialChamfer;

    g_callIdx++;
}

// =========================================================
//  BIPOP-CMA-ES (Shift+V)
// =========================================================
inline void runBipopCmaes() {
    std::cout << "\n=== BIPOP-CMA-ES (Shift+V) ===" << std::endl;
    if (registrationHandle.compRmse == 0.0f) {
        std::cerr << "[Shift+V] No registration yet. Run HemiAuto (O) first."
                  << std::endl;
        return;
    }

    // Phase 1: シード固定
    const uint32_t outer_seed = g_trialSeed + 1000u + g_callIdx * 97u;
    const uint32_t cma_base   = g_trialSeed + 2000u + g_callIdx * 10u;
    std::cout << "[Seed] BIPOP outer=" << outer_seed
              << "  CMA-ES base=" << cma_base
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    auto organs = getOrganList();
    computeUnifiedMetrics();
    g_metricsValid = true;
    float rmse_before = registrationHandle.compRmse;
    std::cout << "[Shift+V] start RMSE=" << rmse_before << std::endl;

    std::vector<std::vector<GLfloat>> start_v(organs.size()), start_n(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            start_v[i] = organs[i]->mVertices;
            start_n[i] = organs[i]->mNormals;
        }
    }

    float best_rmse = rmse_before;
    auto  best_v = start_v;
    auto  best_n = start_n;

    const int N_STARTS = 10;
    std::mt19937 rng(outer_seed);  // Phase 1: trial_seed 連動 (was random_device)
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    for (int run = 0; run < N_STARTS; run++) {
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                organs[i]->mVertices = start_v[i];
                organs[i]->mNormals  = start_n[i];
                setUp(*organs[i]);
            }
        }

        CmaesRefine::Params p;
        p.verbose        = true;
        p.log_every      = 100;
        p.save_debug_jpg = false;
        // CMA-ES sampling range is in length units -> scale by sceneDiag
        p.tx_range = RegRatios::cmaLocalT();
        p.ty_range = RegRatios::cmaLocalT();
        p.tz_range = RegRatios::cmaLocalT();
        // Phase 1: CMA-ES 内部 srand を固定 (cmaes_init() の time(NULL) 上書き)
        p.rng_seed = cma_base + (uint32_t)run;

        float tx=0,ty=0,tz=0,rx=0,ry=0,rz=0,sc=1.0f;
        std::string regime;

        // 元のコードと同じ: 偶数=local, 奇数=global の交互配分
        if (run % 2 == 0) {
            // Regime2 (local)
            p.sigma0 = 0.3 + d01(rng) * 0.4;   // 0.3 ~ 0.7 (CMA-ES内部正規化なのでscale-free)
            const float lt = RegRatios::cmaLocalT();
            tx = (d01(rng)*2-1)*lt; ty = (d01(rng)*2-1)*lt;
            tz = (d01(rng)*2-1)*lt;
            rx = (d01(rng)*2-1)*10.f; ry = (d01(rng)*2-1)*10.f;
            rz = (d01(rng)*2-1)*10.f;
            sc = 0.95f + d01(rng)*0.10f;
            regime = "Local";
        } else {
            // Regime1 (global)
            p.sigma0 = 0.5 + d01(rng) * 0.5;   // 0.5 ~ 1.0
            const float gt = RegRatios::cmaGlobalT();
            tx = (d01(rng)*2-1)*gt; ty = (d01(rng)*2-1)*gt;
            tz = (d01(rng)*2-1)*gt;
            rx = (d01(rng)*2-1)*30.f; ry = (d01(rng)*2-1)*30.f;
            rz = (d01(rng)*2-1)*30.f;
            sc = 0.90f + d01(rng)*0.20f;
            regime = "Global";
        }

        if (run > 0) {
            CmaesRefine::applyIncrementalSRT(organs, tx,ty,tz, rx,ry,rz, sc);
            for (auto* m : organs) if (m) setUp(*m);
        }

        std::cout << "[Shift+V] Run " << (run+1) << "/" << N_STARTS
                  << "  " << regime << "  sigma0=" << p.sigma0
                  << "  cma_seed=" << p.rng_seed << std::endl;

        CmaesRefine::Result r = CmaesRefine::run(
            organs, screenMesh,
            gGridWidth, gGridHeight(),
            RegRatios::zThresh(), p);
        computeUnifiedMetrics();
        g_metricsValid = true;
        float rmse_run = registrationHandle.compRmse;
        // CMA-ES内部でstd::fixedが設定されるのでリセット
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "[Shift+V] Run " << (run+1)
                  << "  RMSE=" << rmse_run
                  << (r.improved ? " [+]" : " [-]") << std::endl;

        if (rmse_run < best_rmse) {
            best_rmse = rmse_run;
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) {
                    best_v[i] = organs[i]->mVertices;
                    best_n[i] = organs[i]->mNormals;
                }
            }
        }
    }

    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            organs[i]->mVertices = best_v[i];
            organs[i]->mNormals  = best_n[i];
            setUp(*organs[i]);
        }
    }
    computeUnifiedMetrics();
    g_metricsValid = true;
    std::cout << std::defaultfloat << std::setprecision(6);
    float improvement = rmse_before - best_rmse;
    std::cout << "[Shift+V] Best: " << rmse_before << " -> " << best_rmse
              << " (delta=" << improvement << ")"
              << (improvement > 0.001f ? " [IMPROVED]" : " [NO CHANGE]")
              << std::endl;

    g_callIdx++;  // Phase 1: 末尾でインクリメント
}

// =========================================================
//  BIPOP-CMA-ES V2 (Shift+F)  -- "Fast"
//  -----------------------------------------------------------------
//  V2 entry point. Functionally equivalent to runBipopCmaes() above
//  (Shift+V) for plain RMSE objectives; replaces the inner CMA-ES
//  call with CmaesRefine::runV2(), which uses the pure-function
//  evaluate_one() path defined in CmaesRefineV2.h.
//
//  Phase 1 contract:
//   - eval_mode forced to FULL_MESH; intended to produce CompRMSE
//     bit-identical to V1 from the same trial_seed/callIdx state.
//   - parallel_population=false; single-threaded inner loop.
//   - Phase 2 will add SUBSET_RMSE (3-5x speed at <=5% RMSE drift).
//   - Phase 3 will add OpenMP over the population.
//
//  Determinism contract: every formula that consumes the BIPOP rng
//  or sets a CMA-ES seed is COPIED VERBATIM from runBipopCmaes(),
//  so that calling Shift+V vs Shift+F from the same
//  (g_trialSeed, g_callIdx) state walks the same initial-jitter
//  trajectory and feeds CMA-ES the same seed sequence. Only the
//  log-prefix strings, the Params type, and the inner-call entry
//  point differ.
// =========================================================
inline void runBipopCmaesV2() {
    std::cout << "\n=== BIPOP-CMA-ES V2 (Shift+F) ===" << std::endl;
    if (registrationHandle.compRmse == 0.0f) {
        std::cerr << "[Shift+F] No registration yet. Run HemiAuto (O) first."
                  << std::endl;
        return;
    }

    // Phase 1: シード固定 -- formulas IDENTICAL to runBipopCmaes()
    const uint32_t outer_seed = g_trialSeed + 1000u + g_callIdx * 97u;
    const uint32_t cma_base   = g_trialSeed + 2000u + g_callIdx * 10u;
    std::cout << "[Seed Shift+F] BIPOP outer=" << outer_seed
              << "  CMA-ES base=" << cma_base
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    auto organs = getOrganList();
    computeUnifiedMetrics();
    g_metricsValid = true;
    float rmse_before = registrationHandle.compRmse;
    std::cout << "[Shift+F] start RMSE=" << rmse_before << std::endl;

    std::vector<std::vector<GLfloat>> start_v(organs.size()), start_n(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            start_v[i] = organs[i]->mVertices;
            start_n[i] = organs[i]->mNormals;
        }
    }

    float best_rmse = rmse_before;
    auto  best_v = start_v;
    auto  best_n = start_n;

    const int N_STARTS = 10;
    std::mt19937 rng(outer_seed);  // same seed type/value as Shift+V
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    for (int run = 0; run < N_STARTS; run++) {
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                organs[i]->mVertices = start_v[i];
                organs[i]->mNormals  = start_n[i];
                setUp(*organs[i]);
            }
        }

        // ----- Params: ParamsV2 instead of Params; V1 fields copied --
        CmaesRefine::ParamsV2 p;
        p.verbose        = true;
        p.log_every      = 100;
        p.save_debug_jpg = false;
        p.tx_range = RegRatios::cmaLocalT();
        p.ty_range = RegRatios::cmaLocalT();
        p.tz_range = RegRatios::cmaLocalT();
        p.rng_seed = cma_base + (uint32_t)run;

        // ----- V2-only knobs (Phase 2: AUTO -> SUBSET_RMSE) ----------
        // AUTO is resolved by runV2 via resolve_eval_mode():
        //   plain RMSE (this code path)         -> SUBSET_RMSE
        //   silhouette/boundary (not used here) -> FULL_MESH
        // To force FULL_MESH for V1 bit-identical reproduction, set
        // p.eval_mode = CmaesRefine::EvalMode::FULL_MESH explicitly.
        p.eval_mode           = CmaesRefine::EvalMode::AUTO;
        p.parallel_population = false;
        p.update_interval     = 10;   // matches V1's UPDATE_INTERVAL

        float tx=0,ty=0,tz=0,rx=0,ry=0,rz=0,sc=1.0f;
        std::string regime;

        // d01(rng) consumption order MUST match runBipopCmaes() exactly:
        //   Local:  sigma0, tx, ty, tz, rx, ry, rz, sc
        //   Global: sigma0, tx, ty, tz, rx, ry, rz, sc
        if (run % 2 == 0) {
            // Regime2 (local)
            p.sigma0 = 0.3 + d01(rng) * 0.4;   // 0.3 ~ 0.7
            const float lt = RegRatios::cmaLocalT();
            tx = (d01(rng)*2-1)*lt; ty = (d01(rng)*2-1)*lt;
            tz = (d01(rng)*2-1)*lt;
            rx = (d01(rng)*2-1)*10.f; ry = (d01(rng)*2-1)*10.f;
            rz = (d01(rng)*2-1)*10.f;
            sc = 0.95f + d01(rng)*0.10f;
            regime = "Local";
        } else {
            // Regime1 (global)
            p.sigma0 = 0.5 + d01(rng) * 0.5;   // 0.5 ~ 1.0
            const float gt = RegRatios::cmaGlobalT();
            tx = (d01(rng)*2-1)*gt; ty = (d01(rng)*2-1)*gt;
            tz = (d01(rng)*2-1)*gt;
            rx = (d01(rng)*2-1)*30.f; ry = (d01(rng)*2-1)*30.f;
            rz = (d01(rng)*2-1)*30.f;
            sc = 0.90f + d01(rng)*0.20f;
            regime = "Global";
        }

        if (run > 0) {
            CmaesRefine::applyIncrementalSRT(organs, tx,ty,tz, rx,ry,rz, sc);
            for (auto* m : organs) if (m) setUp(*m);
        }

        std::cout << "[Shift+F] Run " << (run+1) << "/" << N_STARTS
                  << "  " << regime << "  sigma0=" << p.sigma0
                  << "  cma_seed=" << p.rng_seed << std::endl;

        // ----- The single line that differs in spirit ----------------
        // runV2 (V2 path) instead of run (V1 path). Same arguments.
        CmaesRefine::Result r = CmaesRefine::runV2(
            organs, screenMesh,
            gGridWidth, gGridHeight(),
            RegRatios::zThresh(), p);
        computeUnifiedMetrics();
        g_metricsValid = true;
        float rmse_run = registrationHandle.compRmse;
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "[Shift+F] Run " << (run+1)
                  << "  RMSE=" << rmse_run
                  << (r.improved ? " [+]" : " [-]") << std::endl;

        if (rmse_run < best_rmse) {
            best_rmse = rmse_run;
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) {
                    best_v[i] = organs[i]->mVertices;
                    best_n[i] = organs[i]->mNormals;
                }
            }
        }
    }

    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            organs[i]->mVertices = best_v[i];
            organs[i]->mNormals  = best_n[i];
            setUp(*organs[i]);
        }
    }
    computeUnifiedMetrics();
    g_metricsValid = true;
    std::cout << std::defaultfloat << std::setprecision(6);
    float improvement = rmse_before - best_rmse;
    std::cout << "[Shift+F] Best: " << rmse_before << " -> " << best_rmse
              << " (delta=" << improvement << ")"
              << (improvement > 0.001f ? " [IMPROVED]" : " [NO CHANGE]")
              << std::endl;

    g_callIdx++;  // Phase 1: 末尾でインクリメント (matches Shift+V)
}

// =========================================================
//  BIPOP-CMA-ES V3 (Shift+G)  -- "Good performance"
//  -----------------------------------------------------------------
//  V3 entry point. Caller-side wrapper around CmaesRefine::
//  runBipopCmaesV3. Responsibilities live exclusively here:
//    - read globals (organs, screenMesh, registrationHandle,
//      g_sceneDiag, g_trialSeed, g_callIdx),
//    - convert mCutMesh vertex/normal arrays to vec3,
//    - extract the target cloud,
//    - hand a pure ParamsV3 + vec3 buffers to CmaesRefineV3.h,
//    - apply the returned best_world_matrix to all 6 organs,
//    - read back compRmse via computeUnifiedMetrics for logging
//      and PoseLibrary.
//
//  CmaesRefineV3.h itself is OpenGL-aware via NoOpen3DRegistration.h
//  but never writes to globals from within its inner loop, so this
//  wrapper is the ONLY translation unit that touches registration
//  state during a Shift+G session. That structure is what V3-5
//  (run-level OMP) needs.
//
//  Determinism contract: outer_seed and cma_base formulas IDENTICAL
//  to runBipopCmaes() (V1, Shift+V) and runBipopCmaesV2() (V2,
//  Shift+F). Calling V/F/G from the same (g_trialSeed, g_callIdx)
//  state walks the same initial-jitter trajectory and feeds CMA-ES
//  the same seed sequence. Only the eval path (V3 pure-function)
//  and the writeback path (matrix-based, single application) differ
//  from V1.
//
//  V3-1 ships with src_voxel_ratio = tgt_voxel_ratio = 0, which
//  makes the V3 numerics bit-identical to V1 / V2 FULL_MESH for the
//  same seed state. V3-2 will set both to 0.015f.
// =========================================================
inline void runBipopCmaesV3() {
    std::cout << "\n=== BIPOP-CMA-ES V3 (Shift+G) ===" << std::endl;
    if (registrationHandle.compRmse == 0.0f) {
        std::cerr << "[Shift+G] No registration yet. Run HemiAuto (O) first."
                  << std::endl;
        return;
    }

    // ----- V3 timing diagnostics (chat 7) ---------------------------
    // Six checkpoints around the major phases. Goal is to identify
    // which phase outside the CMA-ES inner loop is responsible for
    // the elapsed time gap (ResultV3 reports ~870ms of CMA-ES core
    // for the heavy mesh, but PoseLibrary records ~16s elapsed).
    using clk    = std::chrono::high_resolution_clock;
    auto   ms_dur = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };
    const auto t_phase_start = clk::now();
    auto t_prev = t_phase_start;
    auto stamp  = [&](const char* label) {
        const auto t_now = clk::now();
        std::cout << "[ShiftG/Time] " << label << " : "
                  << std::fixed << std::setprecision(1)
                  << ms_dur(t_now - t_prev) << " ms"
                  << "  (cumulative " << ms_dur(t_now - t_phase_start)
                  << " ms)" << std::defaultfloat << std::endl;
        t_prev = t_now;
    };

    // ----- Seed determination (formulas IDENTICAL to V1 / V2) ------
    const uint32_t outer_seed = g_trialSeed + 1000u + g_callIdx * 97u;
    const uint32_t cma_base   = g_trialSeed + 2000u + g_callIdx * 10u;
    std::cout << "[Seed Shift+G] BIPOP outer=" << outer_seed
              << "  CMA-ES base=" << cma_base
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    auto organs = getOrganList();

    // ----- Phase A: Pre-session computeUnifiedMetrics ---------------
    // Read once, BEFORE entering the pure V3 driver. V3 does not call
    // computeUnifiedMetrics during its 10-Run loop -- this snapshot
    // is the only screening reference for r.improved.
    //
    // [Phase A skip 撤回 — 2026-05-21]
    // 旧コード: 前回 Phase F (または RANSAC) が g_metricsValid=true で
    // 終わっていれば、cached registrationHandle.compRmse をそのまま
    // 使って 135-225 ms 節約していた。
    //
    // 問題: コード内で g_metricsValid=true を立てる箇所が 13 箇所以上
    // あり、その一部に「pose は変化しているのに metrics は再計算せず
    // フラグだけ true にする」経路が紛れている。さらに pose を変える
    // ハンドラ (Undo / 手動移動 / AR overlay) が g_metricsValid=false
    // を確実に立てていない可能性もある。結果として Phase A は古い
    // compRmse を読み、Phase F (今回の修正で常に再計算) の真値と
    // 大幅に乖離するケースが現場ログで確認された
    // (例: Phase A=0.0316795, Phase F=0.115469 で apply は skip)。
    // この乖離は PoseLibrary の RMSE 採択ゲートを直撃し、
    // 「劣化したように見える → reject → revert」を誘発する。
    //
    // 安全側に倒して毎回再計算する。コスト 135-225 ms 増。
    // Phase F の修正 (常時再計算) と対称な振る舞いになり、
    // rmse_before と rmse_after が同じ関数 / 同じタイミング条件で
    // 計測した値同士の比較になる。
    computeUnifiedMetrics();
    g_metricsValid = false;  // consumed; Phase F will re-validate
    const float rmse_before  = registrationHandle.compRmse;
    const int   init_matched = registrationHandle.compCount;
    std::cout << "[Shift+G] start RMSE=" << rmse_before
              << "  init_matched=" << init_matched << std::endl;
    stamp("A. pre_computeUnifiedMetrics");

    if (!liverMesh3D) {
        std::cerr << "[Shift+G] liverMesh3D is null; aborting." << std::endl;
        return;
    }

    // ----- Phase B: Convert liver vertices / normals to vec3 --------
    // Vertex order is preserved exactly (i+=3 stride, x/y/z columns
    // -> .x/.y/.z components), so the KDTree input order inside V3
    // matches V1's runtime liverMesh3D->mVertices order. This is the
    // bit-identical hook from V1 to V3 for the V3-1 voxel=0 path.
    std::vector<glm::vec3> start_liver_verts;
    std::vector<glm::vec3> start_liver_normals;
    {
        const auto& v = liverMesh3D->mVertices;
        const auto& n = liverMesh3D->mNormals;
        start_liver_verts.reserve(v.size() / 3);
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            start_liver_verts.emplace_back(v[i], v[i+1], v[i+2]);
        }
        start_liver_normals.reserve(n.size() / 3);
        for (size_t i = 0; i + 2 < n.size(); i += 3) {
            start_liver_normals.emplace_back(n[i], n[i+1], n[i+2]);
        }
    }
    stamp("B. vec3_conversion");

    // ----- Phase C: Extract target cloud ----------------------------
    // Same call site V1 / V2 use inside the CMA-ES driver. Caller-
    // side here so V3 stays free of NoOpen3DRegistration calls in
    // its hot path (build_eval_context_v3 already takes the cloud
    // by const ref).
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = std::max(0.001f, RegRatios::zThresh());
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[Shift+G] empty target cloud; aborting." << std::endl;
        return;
    }
    const std::vector<glm::vec3>& tgt_points = targetCloud->points;
    stamp("C. extractFrontFacePoints");

    // ----- Build ParamsV3 -------------------------------------------
    // Range fields mirror V1's `Params` defaults that runBipopCmaes()
    // does NOT override (rotation, scale, penalty, screening). V1
    // sets tx/ty/tz_range = cmaLocalT() unconditionally; V3 follows.
    CmaesRefine::ParamsV3 p;
    p.verbose         = true;
    p.log_every       = 100;
    p.save_debug_jpg  = false;

    p.tx_range        = RegRatios::cmaLocalT();
    p.ty_range        = RegRatios::cmaLocalT();
    p.tz_range        = RegRatios::cmaLocalT();
    // Defaults already match V1: rx/ry/rz_range = 10.0, scale ∈ [0.90, 1.10],
    // min_match_ratio = 0.30, penalty_value = 9.9.

    // V3-specific: scene scale + jitter ranges. V3 does not extern
    // g_sceneDiag, so we copy it once here.
    p.scene_diag      = g_sceneDiag;
    p.jitter_local_t  = RegRatios::cmaLocalT();
    p.jitter_global_t = RegRatios::cmaGlobalT();

    // V3-2: voxel downsample enabled. case C (chat 6 plan):
    // CmaesRefine::runBipopCmaesV3 voxelizes both src and tgt ONCE
    // per Shift+G session BEFORE the 10-Run loop, then jitters the
    // small voxel cloud per Run. This keeps the downsampled cloud
    // independent of jitter (deterministic across Runs and across
    // Shift+G invocations on the same machine) and avoids 9x of
    // wasted voxel work compared to a per-Run "voxelize-after-jitter"
    // design.
    //
    // 0.015 ratio @ scene_diag ~= 2.21m -> voxel_size ~= 3.3 cm.
    // This is the value Phase 2.5 (V2) settled on after empirical
    // tuning; expected effect on heavy mesh (132k src, 800k tgt):
    //   src: 132k -> ~3-5k post-voxel
    //   tgt: 800k -> ~5-10k post-voxel
    // RMSE acceptance: V1 +/- 5% over all 10 Runs.
    p.src_voxel_ratio = 0.015f;
    p.tgt_voxel_ratio = 0.015f;

    // ----- Phase D: Call into the pure V3 driver --------------------
    // Internally voxelizes src+tgt once, then runs 10 BIPOP CMA-ES
    // restarts. The driver already prints per-Run time breakdown
    // ([V3] Run N Time Breakdown lines); Phase D timestamp here is
    // the wall-clock of the entire driver call (= sum of Run totals
    // + voxel + housekeeping inside runBipopCmaesV3).
    CmaesRefine::ResultV3 r = CmaesRefine::runBipopCmaesV3(
        start_liver_verts, start_liver_normals, tgt_points,
        p, rmse_before, init_matched,
        outer_seed, cma_base);
    stamp("D. CMA-ES driver (10 runs + voxel)");

    // ----- Phase E: Apply best_world_matrix to all organs -----------
    // Single application across all 6 organs; replaces V1's "snapshot
    // + restore + applyIncrementalSRT" double-pass per Run plus the
    // outer best_v[]/best_n[] write-back. The matrix already encodes
    // (jitter then best_srt), each built around the correct centroid.
    //
    // Normal transform mirrors CmaesUtils.h::applyIncrementalSRT
    // line 1115, 1121-1126: normalMat = mat3(transpose(inverse(M))),
    // followed by per-component normalize. For a similarity transform
    // (rotation + uniform scale + translation) this gives normals
    // free of skew / non-unit scale artefacts.
    //
    // Per-organ apply / setUp breakdown logged separately to identify
    // whether the GL buffer rebuild dominates the apply cost.
    if (r.improved && r.best_run_idx >= 0) {
        const glm::mat4& M = r.best_world_matrix;
        const glm::mat3  normalMat =
            glm::mat3(glm::transpose(glm::inverse(M)));

        double t_apply_sum = 0.0;
        double t_setup_sum = 0.0;
        int    organ_idx   = 0;
        for (auto* mesh : organs) {
            if (!mesh) { organ_idx++; continue; }

            const auto t_apply_start = clk::now();
            auto& v = mesh->mVertices;
            auto& n = mesh->mNormals;
            for (size_t i = 0; i + 2 < v.size(); i += 3) {
                glm::vec4 p4(v[i], v[i+1], v[i+2], 1.0f);
                glm::vec4 tp = M * p4;
                v[i]   = tp.x; v[i+1] = tp.y; v[i+2] = tp.z;
            }
            for (size_t i = 0; i + 2 < n.size(); i += 3) {
                glm::vec3 nm(n[i], n[i+1], n[i+2]);
                glm::vec3 tn = normalMat * nm;
                float len = glm::length(tn);
                if (len > 1e-8f) tn /= len;
                n[i]   = tn.x; n[i+1] = tn.y; n[i+2] = tn.z;
            }
            const auto t_apply_end = clk::now();
            t_apply_sum += ms_dur(t_apply_end - t_apply_start);

            const auto t_setup_start = clk::now();
            setUp(*mesh);
            const auto t_setup_end = clk::now();
            const double t_one_setup = ms_dur(t_setup_end - t_setup_start);
            t_setup_sum += t_one_setup;
            std::cout << "[ShiftG/Time]   organ[" << organ_idx
                      << "] verts=" << (v.size() / 3)
                      << " apply=" << std::fixed << std::setprecision(1)
                      << ms_dur(t_apply_end - t_apply_start) << "ms"
                      << " setUp=" << t_one_setup << "ms"
                      << std::defaultfloat << std::endl;
            organ_idx++;
        }
        std::cout << "[ShiftG/Time]   apply_sum=" << std::fixed
                  << std::setprecision(1) << t_apply_sum << "ms"
                  << "  setUp_sum=" << t_setup_sum << "ms"
                  << std::defaultfloat << std::endl;
    } else {
        std::cout << "[ShiftG/Time]   (no improvement, skipped apply)"
                  << std::endl;
    }
    stamp("E. apply_matrix + setUp x6");

    // ----- Phase F: Confirm via computeUnifiedMetrics ---------------
    // [Phase F skip 撤回 — 2026-05-21]
    // 旧コード: r.improved==false なら pose 不変だから computeUnifiedMetrics を
    // スキップして 122-155 ms 節約していた。
    //
    // 問題: g_metricsValid=true を立てて帰った後、ユーザが (Shift+)Ctrl+G を
    // 再度押すまでの間に Undo / PoseLibrary 復元 / Ctrl+Shift+G の accept /
    // 手動移動 / AR overlay 切替 などで pose が動く経路があると、
    // registrationHandle.compRmse は古い値のまま固まり、次の Phase A も
    // キャッシュを読んで rmse_before が古い値で凍結する。
    // 結果として「改善判定の基準が古い」状態となり、新しい pose では本来
    // 採択されるはずの候補が常に「劣化」と判定されて永久 NO CHANGE になる
    // 既知のリグレッション (FULL RMSE 高速化チャット 2026-05-15 で導入、
    // V3R も同じパターンを共有)。
    //
    // 安全側に倒して r.improved に関係なく毎回再計算する。コスト 135-225ms
    // 増 (V3 driver の数 sec に対して数 % で許容範囲)。
    // 高速化を再導入する場合は、5/1 の Phase V3-1 完了後計画 §7.3 で議論
    // された session_id ガード、もしくは pose を変える全箇所での明示
    // g_metricsValid=false 徹底 (grep が必要) のどちらかが必要。
    computeUnifiedMetrics();
    g_metricsValid = true;
    const float rmse_after = registrationHandle.compRmse;
    stamp("F. post_computeUnifiedMetrics");

    std::cout << std::defaultfloat << std::setprecision(6);
    const float improvement = rmse_before - rmse_after;
    std::cout << "[Shift+G] Best: " << rmse_before << " -> " << rmse_after
              << " (delta=" << improvement << ")"
              << "  best_run="
              << (r.best_run_idx < 0 ? std::string("none")
                                     : std::to_string(r.best_run_idx + 1))
              << "  total_gens=" << r.total_generations
              << (improvement > 0.001f ? "  [IMPROVED]" : "  [NO CHANGE]")
              << std::endl;

    // Total wall-clock summary (compares to PoseLibrary elapsed)
    const double t_grand_total = ms_dur(clk::now() - t_phase_start);
    std::cout << "[ShiftG/Time] === GRAND TOTAL: "
              << std::fixed << std::setprecision(1) << t_grand_total
              << " ms ===" << std::defaultfloat << std::endl;

    g_callIdx++;  // V3: 末尾でインクリメント (matches V1 / V2)
}

// =========================================================
//  RIM-only RMSE diagnostic (Ctrl+G の最終 FINAL ログ用)
//  ---------------------------------------------------------
//  RMSE を rim 帯だけで測る診断指標。最適化には参加させず、
//  セッション最後の [Ctrl+G] サマリーに 1 行だけ出す。
//
//  方向: rim_tgt -> rim_src (KDTree を src 側に張り tgt 側を巡回)。
//  これは compute_full_rmse_local と同じ向きで、computeUnifiedMetrics
//  の compRmse とも同じ向き。max_dist_sq も V1/V3 と同じ
//  (g_sceneDiag / 7.36)^2 ゲート。
//
//  戻り値: マッチが 1 個もないか rim 集合が空のときは -1.0f を返す
//  (caller 側で "N/A" 表示にフォールバック)。
//
//  論文化に向けた背景: HANDOVER_Measurement_Inconsistency_Analysis.md
//  で指摘されている通り、Ctrl+G で出る既存の RMSE 4 種はいずれも
//  "rim 限定" 集計を持たない。RIM-rim weighted RMSE (beta>0 時) も
//  全頂点ペアの重み付け加算であって rim 単独の数値ではない。
//  この関数は純粋な diagnostic で、最適化には一切参加しない。
// =========================================================
inline float compute_rim_only_rmse_diag(
    const std::vector<glm::vec3>& liver_src_now,    // 現姿勢の肝臓頂点 (full mesh)
    const std::vector<uint8_t>&   is_rim_src,       // size == liver_src_now.size()
    const std::vector<glm::vec3>& tgt_rim_points,   // tgt rim 部分集合
    float                         max_dist_sq,
    int&                          n_src_rim_out,
    int&                          matched_out,
    // [Phase A] Optional pair capture for the colored-pairs visualization.
    // When BOTH pointers are non-null, every (tgt_i, src_NN) pair that
    // passes the max_dist_sq gate is appended as
    //     (full-mesh vertex index of src_NN, tgt world position).
    // Both pointers nullptr (default) → byte-identical legacy behaviour;
    // used by the "before" call sites that need only the scalar RMSE.
    //
    // Caller is responsible for clear()ing the output vectors before the
    // call if a fresh capture is required (this function only appends).
    // The scalar return value, n_src_rim_out, and matched_out are
    // unaffected by whether capture is enabled.
    std::vector<int>*             pair_src_full_idx_out = nullptr,
    std::vector<glm::vec3>*       pair_tgt_pos_out      = nullptr)
{
    n_src_rim_out = 0;
    matched_out   = 0;

    const bool capture_pairs =
        (pair_src_full_idx_out != nullptr) && (pair_tgt_pos_out != nullptr);

    // 1. Source side: extract rim subset from current full-mesh pose.
    //    [Phase A] When pair output is requested, also build a
    //    subset_to_full[] map so the KDTree's nnIdx (which is into the
    //    compact src_rim cloud) can be rewritten to full-mesh vertex
    //    index before publication.
    std::vector<glm::vec3> src_rim;
    std::vector<int>       subset_to_full;
    src_rim.reserve(liver_src_now.size() / 8);
    if (capture_pairs) subset_to_full.reserve(liver_src_now.size() / 8);
    const size_t N = std::min(liver_src_now.size(), is_rim_src.size());
    for (size_t i = 0; i < N; i++) {
        if (is_rim_src[i]) {
            src_rim.push_back(liver_src_now[i]);
            if (capture_pairs) subset_to_full.push_back((int)i);
        }
    }
    n_src_rim_out = (int)src_rim.size();

    if (src_rim.empty() || tgt_rim_points.empty()) return -1.0f;

    // 2. KDTree on the small src_rim cloud (~765 verts on LIVER01).
    //    Cost: ~0.5 ms build + ~0.5 ms query at this scale.
    Reg3DCustom::NanoflannAdaptor adaptor(src_rim);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    // 3. tgt_rim を巡回し max_dist_sq ゲート以下のペアだけ集計。
    //    V1 fastComputeRMSE と同じ accumulation 順 (sumSq += d*d).
    //    [Phase A] capture_pairs==true なら同じ gate を通った瞬間に
    //    (full-mesh idx, tgt_pos) を out vector に追記。scalar の
    //    RMSE / count / matched_out への寄与は引数なしの場合と完全に等価。
    float sumSq = 0.0f;
    int   count = 0;
    for (size_t i = 0; i < tgt_rim_points.size(); i++) {
        size_t nnIdx;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, tgt_rim_points[i], nnIdx, dist_sq)
            && dist_sq < max_dist_sq)
        {
            sumSq += dist_sq;
            count++;
            if (capture_pairs) {
                pair_src_full_idx_out->push_back(subset_to_full[nnIdx]);
                pair_tgt_pos_out->push_back(tgt_rim_points[i]);
            }
        }
    }
    matched_out = count;
    if (count == 0) return -1.0f;
    return std::sqrt(sumSq / (float)count);
}

// rim_src マスク + rim_tgt 部分集合の構築 (compute_rim_only_rmse_diag 入力)。
// Phase C2b と同じフィルタ規則 (boundaryDist < threshold かつ instrument
// 除外) を使うので、診断値は最適化路径が見ている rim 定義と一致する。
inline void build_rim_only_rmse_inputs(
    size_t                                          N_full,
    const std::shared_ptr<Reg3DCustom::PointCloud>& targetCloud,
    const std::vector<glm::vec3>&                   tgt_points,
    float                                           rim_tgt_thresh_px,
    std::vector<uint8_t>&                           is_rim_src_out,
    std::vector<glm::vec3>&                         tgt_rim_points_out)
{
    is_rim_src_out.assign(N_full, 0);
    if (g_liverRegion.valid() && g_liverRegion.labels.size() == N_full) {
        for (size_t i = 0; i < N_full; i++) {
            if (g_liverRegion.labels[i] == LiverRegionLabel::RIM) {
                is_rim_src_out[i] = 1;
            }
        }
    }
    tgt_rim_points_out.clear();
    if (targetCloud && targetCloud->hasBoundaryDist() &&
        targetCloud->boundaryDist.size() == tgt_points.size())
    {
        const bool useInst = targetCloud->hasInstrumentDist() &&
                             targetCloud->instrumentDist.size() == tgt_points.size();
        tgt_rim_points_out.reserve(tgt_points.size() / 16);
        for (size_t i = 0; i < tgt_points.size(); i++) {
            if (targetCloud->boundaryDist[i] >= rim_tgt_thresh_px) continue;
            if (useInst &&
                targetCloud->instrumentDist[i] < g_instrumentPxThresh)
            {
                continue;
            }
            tgt_rim_points_out.push_back(tgt_points[i]);
        }
    }
}

// =========================================================
//  publishCtrlGStyleDiagnostics
// ---------------------------------------------------------
//  Compute the diagnostic metrics that Ctrl+G's Phase F.5 normally
//  publishes (rim RMSE + IoU_occluded + containment precision/recall)
//  AT THE CURRENT liverMesh3D POSE, and write them to the
//  g_lastRim* / g_lastSilOccludedIoU2D / g_lastIoUOcc* globals so the
//  immediately-following poseSaveToLibrary call records them onto the
//  PoseLibrary entry — same columns as Ctrl+G entries.
//
//  Use case: AutoQCR (Alt+Ctrl+P) wants its PoseLibrary entries to
//  carry the same metric columns as Ctrl+G (IoU_occ, RIM, Contain).
//  Without this helper, those columns show 0 / N/A for AutoQCR entries.
//
//  Preconditions (gracefully degraded if not met):
//    - liverMesh3D at final pose                  → required
//    - screenMesh                                 → required
//    - g_liverRegion.valid()                      → required for RIM
//    - g_boundaryDistMap.valid                    → required for IoU_occ
//    - g_instrumentDistMap (optional)             → used if available
//
//  Side effects (publish to consume-and-clear globals):
//    - g_lastRimRmse, g_lastRimMatched, g_lastRimTgtTotal,
//      g_lastRimSrcTotal, g_lastRimPairSrcVertIdx, g_lastRimPairTgtPos
//    - g_lastSilOccludedIoU2D
//    - g_lastIoUOccPrecision, g_lastIoUOccRecall
//
//  This is a verbatim extraction of Ctrl+G Phase C.5 + F.5 + F.5b logic
//  (rim diag inputs + rim RMSE compute + IoU2D rasterize + precision/recall).
//  Phase F.5c (F9 overlay capture) is intentionally omitted (display-only,
//  and not needed for PoseLibrary entries).
//
//  Cost: ~20-30ms (target extraction ~5-10ms + rim KD tree ~1ms +
//                  rasterize IoU2D ~5ms + precision/recall ~0.3ms)
// =========================================================
inline void publishCtrlGStyleDiagnostics() {
    // ----- Step 1: Reset all output globals to sentinel ----------------
    //   N/A defaults consistent with non-Ctrl+G call sites: rim=-1.0 (N/A),
    //   IoU_occ=0 ("not measured" per Ctrl+Shift+G convention), prec/rec=-1.
    g_lastRimRmse          = -1.0f;
    g_lastRimMatched       = 0;
    g_lastRimTgtTotal      = 0;
    g_lastRimSrcTotal      = 0;
    g_lastRimPairSrcVertIdx.clear();
    g_lastRimPairTgtPos.clear();
    g_lastSilOccludedIoU2D = 0.0f;
    g_lastIoUOccPrecision  = -1.0f;
    g_lastIoUOccRecall     = -1.0f;

    // ----- Step 2: Sanity ----------------------------------------------
    if (!liverMesh3D || liverMesh3D->mVertices.empty() || !screenMesh) {
        std::cerr << "[CtrlGDiag] liver/screen mesh missing; skip publish."
                  << std::endl;
        return;
    }

    // ----- Step 3: Extract target cloud (front-facing, with boundary) -
    //   This is the same extraction Ctrl+G Phase C does (and the same
    //   computeUnifiedMetrics does internally). The target cache (set
    //   by runQuadCyclicRansac just before) means this is mostly free.
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = std::max(0.001f, RegRatios::zThresh());
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[CtrlGDiag] empty target cloud; skip publish."
                  << std::endl;
        return;
    }
    const std::vector<glm::vec3>& tgt_points = targetCloud->points;

    // ----- Step 4: Rim diagnostic inputs + rim RMSE --------------------
    //   Build is_rim_src (from g_liverRegion.labels == RIM) and
    //   tgt_rim_points (from target boundaryDist < threshold).
    //   Mirrors Phase C.5 + F.5 of Ctrl+G verbatim.
    std::vector<glm::vec3> liver_verts_now;
    liver_verts_now.reserve(liverMesh3D->mVertices.size() / 3);
    for (size_t i = 0; i + 2 < liverMesh3D->mVertices.size(); i += 3) {
        liver_verts_now.emplace_back(
            liverMesh3D->mVertices[i],
            liverMesh3D->mVertices[i+1],
            liverMesh3D->mVertices[i+2]);
    }

    std::vector<uint8_t>   is_rim_src;
    std::vector<glm::vec3> tgt_rim_points;
    build_rim_only_rmse_inputs(
        liver_verts_now.size(), targetCloud, tgt_points,
        g_ctrlgRimTgtThreshPx,
        is_rim_src, tgt_rim_points);

    // max_dist_sq: V1 unified-metrics と同じ (g_sceneDiag/7.36)^2 = max_dist_sq_rim_diag。
    constexpr float kRefSceneDiag = 7.36f;
    const float max_dist_rim    = g_sceneDiag * (1.0f / kRefSceneDiag);
    const float max_dist_sq_rim = max_dist_rim * max_dist_rim;

    int n_src_rim = 0, n_matched = 0;
    const float rim_rmse = compute_rim_only_rmse_diag(
        liver_verts_now, is_rim_src, tgt_rim_points,
        max_dist_sq_rim, n_src_rim, n_matched,
        &g_lastRimPairSrcVertIdx, &g_lastRimPairTgtPos);

    g_lastRimRmse     = rim_rmse;          // -1.0f if matched==0 (compute_rim returns N/A)
    g_lastRimMatched  = n_matched;
    g_lastRimTgtTotal = (int)tgt_rim_points.size();
    g_lastRimSrcTotal = n_src_rim;

    // ----- Step 5: IoU_occluded + Containment (Phase F.5b) -------------
    //   Rasterize liver silhouette via AR camera, compare against target
    //   SAM2 mask (boundaryDist map), with instrument-occlusion exclusion
    //   when the instrument mask is loadable. Returns scalar IoU + cell-
    //   level hitmap/tmask for precision/recall computation.
    if (g_boundaryDistMap.valid
        && g_boundaryDistMap.width  > 1
        && g_boundaryDistMap.height > 1
        && g_boundaryDistMap.data.size() ==
               (size_t)g_boundaryDistMap.width *
               (size_t)g_boundaryDistMap.height
        && !liverMesh3D->mIndices.empty())
    {
        const glm::mat4 sil_view = buildSilhouetteView();
        const glm::mat4 sil_proj = buildSilhouetteProj();
        const glm::mat4 sil_mvp  = sil_proj * sil_view;
        const int sil_w = g_boundaryDistMap.width;
        const int sil_h = g_boundaryDistMap.height;

        std::vector<uint32_t> sil_indices_full(
            liverMesh3D->mIndices.begin(),
            liverMesh3D->mIndices.end());

        const std::vector<float>* inst_ptr = nullptr;
        float inst_thresh = 0.0f;
        const bool inst_loaded = ensureInstrumentDistMap();
        if (inst_loaded
            && g_instrumentDistMap.valid
            && g_instrumentDistMap.width  == sil_w
            && g_instrumentDistMap.height == sil_h
            && g_instrumentDistMap.data.size() ==
                   (size_t)sil_w * (size_t)sil_h)
        {
            inst_ptr    = &g_instrumentDistMap.data;
            inst_thresh = std::max(0.0f, g_instrumentPxThresh);
        }

        std::vector<uint8_t> hitmap_occ, tmask_occ;
        int gw_occ = 0, gh_occ = 0;
        const float iou_occ = CmaesRefineV3RS::rasterize_iou2d_v3rs(
            liver_verts_now, sil_indices_full, sil_mvp,
            g_boundaryDistMap.data, sil_w, sil_h, /*step=*/8,
            &hitmap_occ, &tmask_occ, &gw_occ, &gh_occ,
            nullptr, nullptr, nullptr,
            /*raster_mode=*/0,
            inst_ptr, inst_thresh);

        g_lastSilOccludedIoU2D = (iou_occ >= 0.0f) ? iou_occ : 0.0f;

        // Containment precision/recall from captured cell maps.
        if (gw_occ > 0 && gh_occ > 0) {
            int inter_c = 0, src_c = 0, tgt_c = 0;
            const size_t N_c = (size_t)gw_occ * (size_t)gh_occ;
            for (size_t i = 0; i < N_c; ++i) {
                const bool s = (hitmap_occ[i] != 0);
                const bool t = (tmask_occ[i]  != 0);
                if (s)         ++src_c;
                if (t)         ++tgt_c;
                if (s && t)    ++inter_c;
            }
            g_lastIoUOccPrecision =
                (src_c > 0) ? (float)inter_c / (float)src_c : 0.0f;
            g_lastIoUOccRecall    =
                (tgt_c > 0) ? (float)inter_c / (float)tgt_c : 0.0f;
        }
    }

    // ----- Step 6: Summary log -----------------------------------------
    std::cout << "[CtrlGDiag] Published Ctrl+G-style metrics: "
              << "RIM rmse=" << g_lastRimRmse
              << " (matched " << g_lastRimMatched
              << "/" << g_lastRimTgtTotal << " tgt, "
              << g_lastRimSrcTotal << " src)"
              << "  IoU_occ=" << g_lastSilOccludedIoU2D
              << "  P=" << g_lastIoUOccPrecision
              << "  R=" << g_lastIoUOccRecall
              << std::endl;
}

// =========================================================
//  BIPOP-CMA-ES V3-R (Ctrl+G)  -- 4-quadrant region-aware
//  -----------------------------------------------------------------
//  V3-R entry point. Caller-side wrapper around CmaesRefineV3R::
//  runBipopCmaesV3R. Same caller-side responsibilities as
//  runBipopCmaesV3 (read globals, vec3 conversion, target extraction,
//  matrix application, computeUnifiedMetrics readback), with one
//  additional duty:
//    - copy g_liverRegion.labels and g_liverLR.labels into ParamsV3R
//      so the V3R driver can derive subset_idx_voxel during its
//      session-level Phase C/D (HANDOVER §4.5).
//
//  Determinism contract: outer_seed and cma_base formulas IDENTICAL
//  to runBipopCmaes() (V1) / runBipopCmaesV2() (V2) / runBipopCmaesV3
//  (V3). Calling V3R from the same (g_trialSeed, g_callIdx) state
//  with QUAD_ALL produces a digit-for-digit identical per-Run /
//  per-Gen log to V3 (Shift+G), which is the S4 acceptance gate
//  (HANDOVER §2.6 / §4.4).
//
//  runNormalCompatRefineSession (Shift+N / Ctrl+Shift+N)
// ---------------------------------------------------------
//  Wrapper for the Normal-Compatible refinement.
//
//  Two execution modes (selected at the main.cpp dispatch level by
//  g_normRefineLiveMode):
//    - BLOCKING: this function runs the whole loop in one frame and
//                returns when finished. Behaviour through Phase 5.
//    - LIVE    : main.cpp calls startNormalCompatRefineLive instead,
//                then ticks one step per render frame via
//                tickNormalCompatRefineLive (Phase 6, default).
//
//  Both paths share the same setup (prepareNormalRefineSession) and
//  finalisation (finalizeNormalRefineSession) helpers, so the math
//  is byte-identical — only the loop placement differs.
// =========================================================

// Forward declarations so the live functions can be defined in any order.
inline void finishNormalCompatRefineLive();
inline void cancelNormalCompatRefineLive(const char* reason);

// =========================================================
//  prepareNormalRefineSession
// ---------------------------------------------------------
//  Phase 0+A+B+C+D from the original wrapper, extracted so both the
//  blocking and live paths reuse the exact same setup code.
//
//  Inputs : method, quadrant_mask (= g_activeQuadrantMask normally).
//  Outputs:
//    nrs_out  - populated RefineState ready to be stepped.
//    t0_out   - session start time (for "session done: total=..." log).
//
//  Returns: true if nrs_out is initialised and refineStep is safe to
//           call; false on any abort (with the same error log lines as
//           the legacy wrapper).
// =========================================================
inline bool prepareNormalRefineSession(NormalRefine::RefineMethod method,
                                        uint8_t quadrant_mask,
                                        NormalRefine::RefineState& nrs_out,
                                        std::chrono::steady_clock::time_point& t0_out)
{
    const char* tag = NormalRefine::methodTag(method);

    if (!g_normRefineEnabled) {
        std::cout << tag << " disabled via g_normRefineEnabled=false; "
                     "abort." << std::endl;
        return false;
    }

    std::cout << "\n=== " << NormalRefine::methodName(method)
              << " Refine (Shift+N) ===" << std::endl;
    std::cout << tag << " quadrant_mask = "
              << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
              << "  (0x" << std::hex << (int)quadrant_mask << std::dec
              << ")" << std::endl;

    // ----- Mirror status globals defaulted to "no session yet" -------
    g_normRefineLastMethod      = (method == NormalRefine::SRT_VARIANCE) ? 1 : 0;
    g_normRefineLastIter        = 0;
    g_normRefineLastInitialRMSE = -1.0f;
    g_normRefineLastBestRMSE    = -1.0f;
    g_normRefineLastAccepted    = false;
    g_normRefineLastConverged   = false;

    // ----- Step A: reset diag globals (mirrors Ctrl+G Phase A) ----
    g_lastRimRmse     = -1.0f;
    g_lastRimMatched  = 0;
    g_lastRimTgtTotal = 0;
    g_lastRimSrcTotal = 0;
    // NOTE: do NOT clear g_lastRimPair* here — we need them as anchor
    //   input. publishCtrlGStyleDiagnostics() will rewrite them at
    //   end-of-session with the final-pose pair set.
    g_lastIoUOccPrecision = -1.0f;
    g_lastIoUOccRecall    = -1.0f;
    g_lastSilOccludedIoU2D = 0.0f;
    g_metricsValid = false;

    // ----- Sanity gates ------------------------------------------------
    if (!liverMesh3D || liverMesh3D->mVertices.empty()) {
        std::cerr << tag << " liverMesh3D missing/empty; abort." << std::endl;
        return false;
    }
    if (!screenMesh) {
        std::cerr << tag << " screenMesh missing (run Depth first); abort."
                  << std::endl;
        return false;
    }
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << tag << " Region/LR labels not computed: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << " LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << ". Press Apply Init Pose first." << std::endl;
        return false;
    }

    const size_t N_full = liverMesh3D->mVertices.size() / 3;
    using clk = std::chrono::steady_clock;
    t0_out = clk::now();

    // ----- Step B: Build source visible-vertex indices ----------------
    //   Quadrant subset → AR-vis raycast → Caudal-only → compose.
    std::vector<size_t> visible_indices;
    {
        const auto t_filter0 = clk::now();

        std::vector<int> quad_subset = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels, g_liverLR.labels, quadrant_mask);
        if (quad_subset.empty()) {
            std::cerr << tag << " Empty quadrant subset for mask=0x"
                      << std::hex << (int)quadrant_mask << std::dec
                      << "; abort." << std::endl;
            return false;
        }

        std::vector<uint8_t> arvis_mask;
        if (g_ctrlgUseArVisFilter) {
            if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
                Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
            }
            Reg3D::BVHTree bvh;
            bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
            const float diag = (g_liverRegion.bbox_diag > 0.0f)
                                   ? g_liverRegion.bbox_diag : g_sceneDiag;
            const glm::vec3 ar_cam_pos(0.0f, 0.0f, -0.2f * diag);
            LiverRegionLabel::raycastVisibilityBVH(
                *liverMesh3D, bvh, ar_cam_pos, diag, arvis_mask);
            int n_vis = 0;
            for (uint8_t v : arvis_mask) if (v) n_vis++;
            std::cout << tag << " AR-vis raycast: "
                      << n_vis << "/" << N_full << " visible" << std::endl;
        }

        std::vector<uint8_t> caudal_mask;
        if (g_ctrlgUseCaudalOnly) {
            if (!g_liverCC.valid() || g_liverCC.labels.size() != N_full) {
                std::cout << tag << " Caudal-only requested but g_liverCC "
                             "not computed; ignoring caudal filter this run."
                          << std::endl;
            } else {
                caudal_mask.assign(N_full, 0);
                for (size_t i = 0; i < N_full; i++) {
                    if (g_liverCC.labels[i] == LiverCranioCaudalLabel::CAUDAL)
                        caudal_mask[i] = 1;
                }
            }
        }

        visible_indices.reserve(quad_subset.size());
        const bool useAr  = !arvis_mask.empty();
        const bool useCau = !caudal_mask.empty();
        const bool combineOR = (g_ctrlgArvisCaudalCombine == 1);
        for (int qi : quad_subset) {
            if (qi < 0 || qi >= (int)N_full) continue;
            bool ar_ok  = useAr  ? (arvis_mask[qi]  != 0) : true;
            bool cau_ok = useCau ? (caudal_mask[qi] != 0) : true;
            bool keep;
            if (useAr && useCau) {
                keep = combineOR ? (ar_ok || cau_ok) : (ar_ok && cau_ok);
            } else {
                keep = ar_ok && cau_ok;
            }
            if (keep) visible_indices.push_back(static_cast<size_t>(qi));
        }

        const auto t_filter1 = clk::now();
        const double filter_ms = std::chrono::duration<double, std::milli>(
                                     t_filter1 - t_filter0).count();
        std::cout << tag << " Source filter: " << visible_indices.size()
                  << " / " << N_full
                  << "  (quad=" << quad_subset.size()
                  << ", AR-vis=" << (useAr ? "ON" : "OFF")
                  << ", Caudal=" << (useCau ? "ON" : "OFF")
                  << ", combine=" << (useAr && useCau
                                      ? (combineOR ? "OR" : "AND") : "n/a")
                  << ")  time=" << std::fixed << std::setprecision(1)
                  << filter_ms << std::defaultfloat << " ms" << std::endl;

        // [Phase 7a] Pure RIM mode: intersect visible_indices with
        //   LiverRegionLabel::RIM. Applied AFTER the standard compose
        //   (quad/AR-vis/caudal) so the RIM filter respects whatever
        //   quadrant the user has selected. Drops source verts that
        //   are not rim; the matching target-side drop happens inside
        //   refineStep via the pureRimTarget RefineParams flag (set
        //   in Step D below).
        if (g_normRefinePureRim) {
            const size_t before = visible_indices.size();
            if (g_liverRegion.labels.size() != N_full) {
                std::cout << tag << " Pure-RIM requested but g_liverRegion "
                             "size mismatch; ignoring." << std::endl;
            } else {
                std::vector<size_t> rim_only;
                rim_only.reserve(before);
                for (size_t qi : visible_indices) {
                    if (g_liverRegion.labels[qi] == LiverRegionLabel::RIM)
                        rim_only.push_back(qi);
                }
                visible_indices.swap(rim_only);
                std::cout << tag << " Pure-RIM source filter: "
                          << visible_indices.size() << " / " << before
                          << " (kept rim verts only)" << std::endl;
            }
        }

        if (visible_indices.size() < 20) {
            std::cerr << tag << " Source set too small ("
                      << visible_indices.size()
                      << " verts); abort." << std::endl;
            return false;
        }
    }

    // ----- Step C: rim_src_mask + rim_tgt_mask (Phase 2 L1) -----------
    std::vector<uint8_t> rim_src_mask;
    std::vector<uint8_t> rim_tgt_mask;
    {
        rim_src_mask.assign(N_full, 0);
        if (g_liverRegion.labels.size() == N_full) {
            for (size_t i = 0; i < N_full; i++) {
                if (g_liverRegion.labels[i] == LiverRegionLabel::RIM)
                    rim_src_mask[i] = 1;
            }
        }

        Reg3DCustom::NoOpen3DRegistration reg_extract;
        const float zT = std::max(0.001f, RegRatios::zThresh());
        auto preview = reg_extract.extractFrontFacePoints(
            *screenMesh, gGridWidth, gGridHeight(), zT);
        if (preview && preview->hasBoundaryDist() &&
            preview->boundaryDist.size() == preview->points.size())
        {
            const bool useInst = preview->hasInstrumentDist() &&
                                 preview->instrumentDist.size() == preview->points.size();
            const float thresh = g_ctrlgRimTgtThreshPx;
            rim_tgt_mask.assign(preview->points.size(), 0);
            int n_tgt_rim = 0;
            for (size_t i = 0; i < preview->points.size(); i++) {
                if (preview->boundaryDist[i] >= thresh) continue;
                if (useInst &&
                    preview->instrumentDist[i] < g_instrumentPxThresh) continue;
                rim_tgt_mask[i] = 1;
                n_tgt_rim++;
            }
            std::cout << tag << " Target rim mask: " << n_tgt_rim
                      << " / " << preview->points.size()
                      << "  thresh=" << thresh << "px" << std::endl;
        }
    }

    // ----- Step D: build RefineParams + initRefine --------------------
    NormalRefine::RefineParams nrp;
    nrp.distanceThreshold   = g_normRefineDistThresh;
    nrp.minNormalAngleCos   = g_normRefineMinNormalCos;
    nrp.maxTotalIterations  = g_normRefineMaxIter;
    nrp.itersPerFrame       = std::max(1, g_normRefineItersPerFrame);
    nrp.betaRimSrc          = std::max(0.0f, g_normRefineBetaRimSrc);
    nrp.betaRimTgt          = std::max(0.0f, g_normRefineBetaRimTgt);
    nrp.useAnchorPairs      = g_normRefineUseAnchor
                              && !g_lastRimPairSrcVertIdx.empty()
                              && (g_lastRimPairSrcVertIdx.size()
                                  == g_lastRimPairTgtPos.size());
    nrp.anchorPhaseIter     = std::max(0, g_normRefineAnchorPhaseIter);
    nrp.anchorBlend         = std::clamp(g_normRefineAnchorBlend, 0.0f, 1.0f);
    nrp.pureRimTarget       = g_normRefinePureRim;   // [Phase 7a] target-side filter

    auto organs = getOrganList();
    const float zT_main = std::max(0.001f, RegRatios::zThresh());

    bool init_ok = NormalRefine::initRefine(
        nrs_out, liverMesh3D, visible_indices, screenMesh, organs,
        gGridWidth, gGridHeight(), zT_main,
        nrp, method,
        &rim_src_mask, &rim_tgt_mask,
        nrp.useAnchorPairs ? &g_lastRimPairSrcVertIdx : nullptr,
        nrp.useAnchorPairs ? &g_lastRimPairTgtPos     : nullptr);
    if (!init_ok) {
        std::cerr << tag << " initRefine failed; abort." << std::endl;
        return false;
    }

    g_normRefineLastInitialRMSE = nrs_out.initialRMSE;
    nrs_out.cumulativeTransform = glm::dmat4(1.0);
    nrs_out.bestCumulativeTransform = glm::dmat4(1.0);
    return true;
}

// =========================================================
//  finalizeNormalRefineSession
// ---------------------------------------------------------
//  Step F + G + g_callIdx++ from the original wrapper. Called by
//  both the blocking wrapper (after its loop) and the live tick
//  (after convergence / max-iter).
//
//  Behavior:
//    - Accept gate: bestRMSE < initialRMSE * 0.999 ?
//    - If accept: restoreMeshes + apply best cumulative transform.
//    - If reject: restoreMeshes only (pose returns to pre-press state).
//    - computeUnifiedMetrics + publishCtrlGStyleDiagnostics.
//    - Increment g_callIdx.
//
//  The caller (main.cpp for blocking, the live tick / cancel for
//  live mode) is responsible for invoking poseSaveToLibrary, which
//  is not in scope here (PoseLibrary.h is included after this file).
// =========================================================
inline void finalizeNormalRefineSession(NormalRefine::RefineMethod method,
                                         NormalRefine::RefineState& nrs,
                                         std::chrono::steady_clock::time_point t0,
                                         const char* mode_tag)
{
    using clk = std::chrono::steady_clock;
    const char* tag = NormalRefine::methodTag(method);

    g_normRefineLastIter     = nrs.totalIterations;
    g_normRefineLastBestRMSE = nrs.bestRMSE;

    const float accept_thresh = nrs.initialRMSE * 0.999f;
    const bool  accept = (nrs.bestRMSE < accept_thresh)
                         && std::isfinite(nrs.bestRMSE);
    g_normRefineLastAccepted = accept;

    auto organs = getOrganList();
    if (accept) {
        // Restore from backup then apply the best cumulative transform
        // so the final mesh state sits at the best-RMSE pose (not the
        // possibly worse final pose).
        nrs.restoreMeshes();
        NormalRefine::applyIncrementalTransform(
            nrs.bestCumulativeTransform, organs);
        std::cout << tag << " " << mode_tag << " ACCEPTED  initial="
                  << std::fixed << std::setprecision(5) << nrs.initialRMSE
                  << "  best=" << nrs.bestRMSE
                  << "  iters=" << nrs.totalIterations
                  << "  (best @iter=" << nrs.bestIteration << ")"
                  << std::defaultfloat << std::endl;
    } else {
        nrs.restoreMeshes();
        std::cout << tag << " " << mode_tag << " REJECTED  initial="
                  << std::fixed << std::setprecision(5) << nrs.initialRMSE
                  << "  best=" << nrs.bestRMSE
                  << "  iters=" << nrs.totalIterations
                  << "  (no significant improvement)"
                  << std::defaultfloat << std::endl;
    }

    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    publishCtrlGStyleDiagnostics();

    const auto t1 = clk::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << tag << " session done: total=" << std::fixed
              << std::setprecision(1) << total_ms << " ms"
              << "  compRmse=" << std::setprecision(5) << registrationHandle.compRmse
              << "  compIoU2D=" << std::setprecision(4) << registrationHandle.compIoU2D
              << std::defaultfloat << std::endl;

    g_callIdx++;
}

// =========================================================
//  runNormalCompatRefineSession (BLOCKING)
// ---------------------------------------------------------
//  Original Shift+N / Ctrl+Shift+N entry: prepare + loop + finalize
//  in a single call. Kept for the case where g_normRefineLiveMode is
//  OFF, and used as the reference implementation for tests.
//
//  The caller (main.cpp KEY_N handler) is responsible for invoking
//  poseAutoSaveBeforeRegistration() before and
//  poseSaveToLibrary(SaveCriterion::EITHER) after.
// =========================================================
inline void runNormalCompatRefineSession(NormalRefine::RefineMethod method,
                                          uint8_t quadrant_mask)
{
    NormalRefine::RefineState nrs;
    std::chrono::steady_clock::time_point t0;
    if (!prepareNormalRefineSession(method, quadrant_mask, nrs, t0)) {
        return;  // prepare logged the abort reason.
    }

    // ----- Step E: blocking refinement loop ---------------------------
    const glm::vec3 viewDir(0.0f, 0.0f, 1.0f);   // AR fixed view-dir
    while (nrs.totalIterations < nrs.params.maxTotalIterations) {
        NormalRefine::RefineStepResult step =
            NormalRefine::refineStep(nrs, viewDir);
        nrs.cumulativeTransform =
            step.incrementalTransform * nrs.cumulativeTransform;
        if (step.rmse > 0.0f && step.rmse < nrs.bestRMSE) {
            nrs.bestRMSE = step.rmse;
            nrs.bestIteration = nrs.totalIterations;
            nrs.bestCumulativeTransform = nrs.cumulativeTransform;
        }
        if (step.converged) {
            g_normRefineLastConverged = true;
            break;
        }
    }

    finalizeNormalRefineSession(method, nrs, t0, "[blocking]");
}

// =========================================================
//  Live mode (Phase 6) — Frame-driven refinement
// ---------------------------------------------------------
//  Object-tracking-style visualisation: each render frame runs ONE
//  refineStep call, applies the resulting incremental transform to
//  organMeshes immediately, and re-renders. The user sees the mesh
//  smoothly converge toward the target instead of snapping at the end.
//
//  Control flow:
//    main.cpp KEY_N         -> poseAutoSaveBeforeRegistration()
//                              + startNormalCompatRefineLive(...)
//    main.cpp render loop   -> tickNormalCompatRefineLive() each frame
//                              [last tick triggers finishNormalCompatRefineLive]
//    finish sets pendingSave -> main.cpp consumes flag and calls
//                              poseSaveToLibrary(SaveCriterion::EITHER, mask)
//
//  Concurrency: single session at a time. Pressing Shift+N or
//  Ctrl+Shift+N while one is in flight cancels (rejects) the current
//  session before starting the new one.
// =========================================================
namespace NormalRefineLive {
    inline NormalRefine::RefineState           state;
    inline bool                                 active      = false;
    inline bool                                 pendingSave = false;
    inline NormalRefine::RefineMethod           method      = NormalRefine::NORMAL_COMPAT;
    inline uint8_t                              mask        = 0xFF;
    inline std::chrono::steady_clock::time_point t0;

    // [Live early-stop] How many consecutive iterations RMSE has NOT
    //   improved over best. Matches the SRT-3D tracking reference's
    //   worseCount field. Reset to 0 in start, incremented in tick
    //   when step.rmse >= bestRMSE, reset to 0 when a new best is
    //   found. Termination at LIVE_WORSE_MAX iters of stagnation.
    inline int worseCount = 0;
    inline constexpr int LIVE_WORSE_MAX = 30;
}

inline bool startNormalCompatRefineLive(NormalRefine::RefineMethod method,
                                         uint8_t quadrant_mask)
{
    // If something is already running, cancel it (force reject) first
    // so the previous session lands in PoseLibrary cleanly before we
    // start a new one. Without this we would silently leak the old
    // state.
    if (NormalRefineLive::active) {
        cancelNormalCompatRefineLive("restart-requested");
    }
    NormalRefineLive::state.reset();

    if (!prepareNormalRefineSession(method, quadrant_mask,
                                     NormalRefineLive::state,
                                     NormalRefineLive::t0))
    {
        // prepare logged the abort reason. Leave active=false so the
        // main loop's tick remains a no-op.
        return false;
    }

    NormalRefineLive::method      = method;
    NormalRefineLive::mask        = quadrant_mask;
    NormalRefineLive::active      = true;
    NormalRefineLive::pendingSave = false;
    NormalRefineLive::worseCount  = 0;   // [Live early-stop] reset
    g_normRefineLiveActive        = true;

    std::cout << NormalRefine::methodTag(method)
              << " LIVE START  initialRMSE="
              << std::fixed << std::setprecision(5)
              << NormalRefineLive::state.initialRMSE
              << "  anchors=" << NormalRefineLive::state.anchorCount
              << std::defaultfloat
              << std::endl;
    return true;
}

inline void tickNormalCompatRefineLive() {
    if (!NormalRefineLive::active) return;
    auto& nrs = NormalRefineLive::state;

    // [Phase 6 UX] Honour steps-per-frame slider. We unroll the body
    //   below into an inner loop, terminating early on any early-stop
    //   condition. This is just a multiplier — Live mode with steps=1
    //   gives the slowest, most visually dramatic animation.
    const int steps = std::max(1, g_normRefineLiveStepsPerFrame);
    for (int s = 0; s < steps; s++) {
        if (!NormalRefineLive::active) return;   // finish() inside the loop

        // Termination check (max-iter reached).
        if (nrs.totalIterations >= nrs.params.maxTotalIterations) {
            finishNormalCompatRefineLive();
            return;
        }

        // ONE refineStep per inner iter (the header itself runs
        //   itersPerFrame sub-iterations inside each call, so this
        //   advances by itersPerFrame outer iters — default 2).
        const glm::vec3 viewDir(0.0f, 0.0f, 1.0f);
        NormalRefine::RefineStepResult step =
            NormalRefine::refineStep(nrs, viewDir);

        // [Reference parity] Guard against degenerate sub-iter that
        //   returned <6 correspondences (e.g. all source verts went past
        //   distanceThreshold). Treat as converged so we don't apply a
        //   garbage transform.
        if (step.correspondenceCount < 6) {
            std::cout << NormalRefine::methodTag(NormalRefineLive::method)
                      << " LIVE early stop: correspondenceCount="
                      << step.correspondenceCount << " < 6" << std::endl;
            g_normRefineLastConverged = true;
            finishNormalCompatRefineLive();
            return;
        }

        // Compose into cumulative & apply to organ meshes for live viz.
        nrs.cumulativeTransform = step.incrementalTransform * nrs.cumulativeTransform;
        auto organs = getOrganList();
        NormalRefine::applyIncrementalTransform(step.incrementalTransform, organs);

        // Track best + worseCount (drives the 30-iter stagnation early stop).
        if (step.rmse > 0.0f && step.rmse < nrs.bestRMSE) {
            nrs.bestRMSE                = step.rmse;
            nrs.bestIteration           = nrs.totalIterations;
            nrs.bestCumulativeTransform = nrs.cumulativeTransform;
            NormalRefineLive::worseCount = 0;     // [Live early-stop] reset
        } else {
            NormalRefineLive::worseCount++;       // [Live early-stop] tick
        }

        // [Reference parity] Per-10-iter progress log so the operator can
        //   sanity-check the trajectory in the terminal while watching the
        //   ImGui panel.
        if (nrs.totalIterations > 0 && nrs.totalIterations % 10 == 0) {
            std::cout << NormalRefine::methodTag(NormalRefineLive::method)
                      << " LIVE iter=" << nrs.totalIterations
                      << " corr=" << step.correspondenceCount
                      << " rmse=" << std::fixed << std::setprecision(4)
                      << step.rmse
                      << " best=" << nrs.bestRMSE
                      << "@" << nrs.bestIteration
                      << " worse=" << NormalRefineLive::worseCount
                      << std::defaultfloat << std::endl;
        }

        // UI status mirror.
        g_normRefineLastIter        = nrs.totalIterations;
        g_normRefineLastBestRMSE    = nrs.bestRMSE;
        g_normRefineLiveCurrentRMSE = step.rmse;
        g_normRefineLiveAnchorPhase =
            (nrs.anchorCount > 0 &&
             nrs.totalIterations < nrs.params.anchorPhaseIter) ? 1 :
            (nrs.anchorCount > 0 ? 0 : -1);

        // [Reference parity] Early stop if RMSE has been worsening for
        //   LIVE_WORSE_MAX (= 30) consecutive iterations. This is the
        //   "stagnation detector" — the optimiser plateaued, no point
        //   running to the maxIter cap.
        if (NormalRefineLive::worseCount >= NormalRefineLive::LIVE_WORSE_MAX) {
            std::cout << NormalRefine::methodTag(NormalRefineLive::method)
                      << " LIVE early stop: RMSE worsened for "
                      << NormalRefineLive::LIVE_WORSE_MAX
                      << " consecutive iters  (last best @iter="
                      << nrs.bestIteration << ")" << std::endl;
            g_normRefineLastConverged = true;
            finishNormalCompatRefineLive();
            return;
        }

        if (step.converged) {
            g_normRefineLastConverged = true;
            finishNormalCompatRefineLive();
            return;
        }
    } // for steps
}

inline void finishNormalCompatRefineLive() {
    if (!NormalRefineLive::active) return;
    finalizeNormalRefineSession(NormalRefineLive::method,
                                 NormalRefineLive::state,
                                 NormalRefineLive::t0,
                                 "[LIVE]");
    NormalRefineLive::pendingSave = true;   // main.cpp consumes & calls poseSaveToLibrary
    NormalRefineLive::active      = false;
    g_normRefineLiveActive        = false;
    g_normRefineLiveCurrentRMSE   = -1.0f;
    // Note: NormalRefineLive::state.reset() is intentionally deferred
    //   until the next startNormalCompatRefineLive so that the UI can
    //   still display "ACCEPTED / REJECTED" status until then.
}

inline void cancelNormalCompatRefineLive(const char* reason) {
    if (!NormalRefineLive::active) return;
    std::cout << NormalRefine::methodTag(NormalRefineLive::method)
              << " LIVE CANCELLED (" << reason << ")" << std::endl;
    // Force the accept gate to fail by inflating bestRMSE so the
    // finalize step takes the REJECT path (revert to backup).
    NormalRefineLive::state.bestRMSE =
        NormalRefineLive::state.initialRMSE * 10.0f + 1.0f;
    finishNormalCompatRefineLive();
}



// =========================================================
//  runBipopCmaesV3R (Ctrl+G — Region-aware BIPOP-CMA-ES)
// ---------------------------------------------------------
//  Argument:
//    quadrant_mask -- LiverLeftRightLabel::QuadrantMask bitmask
//                     (QUAD_AR | QUAD_AL | QUAD_PR | QUAD_PL combined,
//                     QUAD_ALL = 0x0F). Caller (main.cpp Ctrl+G handler)
//                     supplies g_activeQuadrantMask.
// =========================================================
inline void runBipopCmaesV3R(uint8_t quadrant_mask) {
    std::cout << "\n=== BIPOP-CMA-ES V3-R (Ctrl+G) ===" << std::endl;
    std::cout << "[Ctrl+G] quadrant_mask = "
              << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
              << "  (0x" << std::hex << (int)quadrant_mask << std::dec
              << ")" << std::endl;

    // V3R rim diagnostic: reset published values so a session that bails
    // out before Phase F.5 (early return below, target cloud empty, etc.)
    // does not leak a stale value into the next poseSaveToLibrary call.
    // Phase F.5 re-publishes the fresh values immediately before save.
    g_lastRimRmse     = -1.0f;
    g_lastRimMatched  = 0;
    g_lastRimTgtTotal = 0;
    g_lastRimSrcTotal = 0;
    // [Phase A] Same defensive reset for the rim PAIR vectors so a
    // session that bails before Phase F.5 doesn't leak stale pairs
    // into the next save. Phase F.5 below repopulates them on success.
    g_lastRimPairSrcVertIdx.clear();
    g_lastRimPairTgtPos.clear();
    // [NEW V3RS-CONTAIN] Same reset pattern for precision/recall.
    g_lastIoUOccPrecision = -1.0f;
    g_lastIoUOccRecall    = -1.0f;

    // V3R F9 overlay diagnostic: reset the SilOverlay state so a session
    // that bails out before Phase F.5c does not leave a stale capture
    // from a prior Ctrl+G / Ctrl+Shift+G run visible in the F9 window.
    // F9 slot is SHARED with Ctrl+Shift+G — design decision "whoever
    // pressed last wins" (HANDOVER A·A·A: minimal cost, plural held-out
    // diagnostics). Phase F.5c repopulates this slot with the final
    // pose composite immediately before the function returns.
    //
    // The matching IoU_occluded scalar is published to
    // g_lastSilOccludedIoU2D in Phase F.5b (just below); poseSaveToLibrary
    // already consumes that into the entry's compIoU2D_occluded column,
    // so no PoseLibrary-side change is needed for the IoU number to land
    // on Ctrl+G entries' rows in the table.
    SilOverlay::reset(SilOverlay::g_silOverlay);
    g_lastSilOccludedIoU2D = 0.0f;

    // [UI整理] 旧コード:
    //   if (registrationHandle.compRmse == 0.0f) {
    //       std::cerr << "[Ctrl+G] No registration yet. Run HemiAuto (O) first." ...
    //       return;
    //   }
    // を削除。このガードは「事前に他の登録が走っていないと弾く」だけで
    // 冗長だった。Apply Init Pose 直後 (compRmse==0) から Ctrl+G を呼んでも、
    // 下の Phase A で computeUnifiedMetrics() が現在 pose から rmse_before を
    // 生成する (2026-05-21 以降は cache を信用せず無条件で再計算)。
    // → Ctrl+G が Apply Init Pose 直後から単独で使えるようになる。

    // ----- V3R label validity gate ----------------------------------
    // Region/LR labels must be populated before any non-QUAD_ALL run;
    // for QUAD_ALL we still verify them so a misconfigured run cannot
    // silently fall back to "no filtering" semantics that the user
    // didn't intend.
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[Ctrl+G] Region/LR labels not computed: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << "  LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << ". Press Apply Init Pose first (auto-triggers labels)."
                  << std::endl;
        return;
    }

    // ----- V3R timing diagnostics (mirrors V3 phase stamps) ----------
    using clk    = std::chrono::high_resolution_clock;
    auto   ms_dur = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };
    const auto t_phase_start = clk::now();
    auto t_prev = t_phase_start;
    auto stamp  = [&](const char* label) {
        const auto t_now = clk::now();
        std::cout << "[CtrlG/Time] " << label << " : "
                  << std::fixed << std::setprecision(1)
                  << ms_dur(t_now - t_prev) << " ms"
                  << "  (cumulative " << ms_dur(t_now - t_phase_start)
                  << " ms)" << std::defaultfloat << std::endl;
        t_prev = t_now;
    };

    // ----- Seed determination (formulas IDENTICAL to V1 / V2 / V3) -
    const uint32_t outer_seed = g_trialSeed + 1000u + g_callIdx * 97u;
    const uint32_t cma_base   = g_trialSeed + 2000u + g_callIdx * 10u;
    std::cout << "[Seed Ctrl+G] BIPOP outer=" << outer_seed
              << "  CMA-ES base=" << cma_base
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    auto organs = getOrganList();

    // ----- Phase A: Pre-session computeUnifiedMetrics ---------------
    // [Phase A skip 撤回 — 2026-05-21]
    // 旧コード: g_metricsValid=true なら cache を流用していた。
    // 詳細は V3 (Shift+G) Phase A の同コメント参照。要旨:
    // g_metricsValid を立てる箇所が分散しており、pose 変化と
    // フラグの一貫性が保てない経路があるため、Phase A の rmse_before
    // が stale になり Phase F の真値と乖離して PoseLibrary の
    // RMSE 採択ゲートを誤発火させる現場ログを確認 (callIdx=17/20
    // で Phase A=0.0316795 vs Phase F=0.0508 / 0.1154、apply は skip)。
    // Phase F (常時再計算) と対称に毎回再計算する。
    computeUnifiedMetrics();
    g_metricsValid = false;  // consumed; Phase F will re-validate
    const float rmse_before  = registrationHandle.compRmse;
    const int   init_matched = registrationHandle.compCount;
    std::cout << "[Ctrl+G] start RMSE=" << rmse_before
              << "  init_matched=" << init_matched << std::endl;
    stamp("A. pre_computeUnifiedMetrics");

    if (!liverMesh3D) {
        std::cerr << "[Ctrl+G] liverMesh3D is null; aborting." << std::endl;
        return;
    }

    // ----- Phase B: Convert liver vertices / normals to vec3 --------
    std::vector<glm::vec3> start_liver_verts;
    std::vector<glm::vec3> start_liver_normals;
    {
        const auto& v = liverMesh3D->mVertices;
        const auto& n = liverMesh3D->mNormals;
        start_liver_verts.reserve(v.size() / 3);
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            start_liver_verts.emplace_back(v[i], v[i+1], v[i+2]);
        }
        start_liver_normals.reserve(n.size() / 3);
        for (size_t i = 0; i + 2 < n.size(); i += 3) {
            start_liver_normals.emplace_back(n[i], n[i+1], n[i+2]);
        }
    }
    stamp("B. vec3_conversion");

    // Label vs. vertex size sanity check (HANDOVER §8.4).
    if (g_liverRegion.labels.size() != start_liver_verts.size() ||
        g_liverLR.labels.size()     != start_liver_verts.size()) {
        std::cerr << "[Ctrl+G] Label-size mismatch: verts="
                  << start_liver_verts.size()
                  << "  region.labels=" << g_liverRegion.labels.size()
                  << "  lr.labels=" << g_liverLR.labels.size()
                  << "; aborting (mesh may have been reloaded "
                     "without re-labeling)." << std::endl;
        return;
    }

    // ----- Phase C: Extract target cloud ----------------------------
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = std::max(0.001f, RegRatios::zThresh());
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[Ctrl+G] empty target cloud; aborting." << std::endl;
        return;
    }
    const std::vector<glm::vec3>& tgt_points = targetCloud->points;
    stamp("C. extractFrontFacePoints");

    // ----- Phase C.5: RIM-only RMSE diagnostic inputs + "before" -----
    // セッション最後の [Ctrl+G] RIM-only サマリー用。最適化には不参加。
    // is_rim_src は Phase C2b で再構築されるが、ここで作るのは Phase C2b
    // が rim weighting OFF のときスキップされる (= beta=0 のとき empty
    // になる) ため。診断は beta によらず常に出したい。
    std::vector<uint8_t>   rim_diag_is_rim_src;
    std::vector<glm::vec3> rim_diag_tgt_rim_points;
    build_rim_only_rmse_inputs(
        start_liver_verts.size(), targetCloud, tgt_points,
        g_ctrlgRimTgtThreshPx,
        rim_diag_is_rim_src, rim_diag_tgt_rim_points);

    // max_dist_sq は V1 unified-metrics と同じ (g_sceneDiag/7.36)^2。
    constexpr float kRefSceneDiag_rim_diag = 7.36f;
    const float max_dist_rim_diag    = g_sceneDiag * (1.0f / kRefSceneDiag_rim_diag);
    const float max_dist_sq_rim_diag = max_dist_rim_diag * max_dist_rim_diag;

    int rim_diag_n_src_rim_before = 0;
    int rim_diag_matched_before   = 0;
    const float rmse_rim_before = compute_rim_only_rmse_diag(
        start_liver_verts, rim_diag_is_rim_src, rim_diag_tgt_rim_points,
        max_dist_sq_rim_diag,
        rim_diag_n_src_rim_before, rim_diag_matched_before);
    stamp("C.5. rim_only_rmse_before");

    // ----- Build ParamsV3R ------------------------------------------
    // Inherits ParamsV3 fields from V3 (kept VERBATIM identical to
    // runBipopCmaesV3 above so the determinism contract is preserved
    // at QUAD_ALL). The only V3R-specific additions are the four
    // quadrant fields appended at the end.
    CmaesRefineV3R::ParamsV3R p;

    // ----- ParamsV3 (base class) fields, identical to V3 -----------
    p.verbose         = true;
    p.log_every       = 100;
    p.save_debug_jpg  = false;

    p.tx_range        = RegRatios::cmaLocalT();
    p.ty_range        = RegRatios::cmaLocalT();
    p.tz_range        = RegRatios::cmaLocalT();
    // Defaults match V1: rx/ry/rz_range=10.0, scale ∈ [0.90,1.10],
    // min_match_ratio=0.30, penalty_value=9.9.

    p.scene_diag      = g_sceneDiag;
    p.jitter_local_t  = RegRatios::cmaLocalT();
    p.jitter_global_t = RegRatios::cmaGlobalT();

    p.src_voxel_ratio = 0.015f;
    p.tgt_voxel_ratio = 0.015f;

    // ----- ParamsV3R-specific fields ---------------------------------
    p.quadrant_mask        = quadrant_mask;
    p.region_labels        = g_liverRegion.labels;   // copy (~10K bytes)
    p.lr_labels            = g_liverLR.labels;       // copy (~10K bytes)
    p.full_rmse_use_subset = false;  // S4 default (HANDOVER §1, plan E1)
    // voxel_to_orig and subset_idx_voxel are filled by runBipopCmaesV3R
    // at session level (Phase C/D inside the driver).

    // ----- ParamsV3R-W (rim-weighted extension, opt-in) --------------
    // All three switches default to OFF. When all are off, the wrapper
    // passes empty arrays / zero beta to V3R, which falls through to
    // the original evaluate_one_v3 verbatim → byte-identical with V3
    // at QUAD_ALL preserved.
    p.use_arvis_filter      = g_ctrlgUseArVisFilter;
    p.beta_rim_weight       = std::max(0.0f, g_ctrlgBetaRimWeight);
    p.rim_tgt_threshold_px  = g_ctrlgRimTgtThreshPx;
    // R-feat-2: caudal-only filter (anatomical, orthogonal to AR-vis).
    // is_caudal_orig itself is filled below in Phase C2c (only when the
    // toggle is ON and g_liverCC is valid). The combine mode is always
    // passed; it takes effect only if BOTH arvis_voxel and caudal_voxel
    // end up non-empty inside runBipopCmaesV3R.
    p.use_caudal_only       = g_ctrlgUseCaudalOnly;
    p.arvis_caudal_combine  = g_ctrlgArvisCaudalCombine;

    // ----- [NEW V3R/SearchMode] Reduced-DoF mode wiring --------------
    // Wire the UI-selected search dimension and min_match_ratio. The
    // search_mode field is read inside run_one_bipop_v3r to pick DIM
    // (7/6/4) for cmaes_init and to dispatch the population decode.
    //
    // For SIX_DOF_RIGID and FOUR_DOF_XYRXRY we additionally pre-scale
    // the t/r ranges AND the BIPOP outer-loop jitter so the reduced
    // search stays appropriately local. This is the simplest knob
    // that preserves CMA-ES's [-1, 1] sigma semantics: shrinking the
    // physical range with sigma0 unchanged is mathematically equivalent
    // to shrinking sigma0 with range unchanged, and keeps the jitter
    // / range relationship consistent.
    //
    // SEVEN_DOF (k==1.0) leaves every field untouched, preserving the
    // V3R byte-identical contract at QUAD_ALL with all other knobs
    // (AR-vis, caudal, beta, etc.) also at defaults.
    p.search_mode     = g_ctrlgSearchMode;
    p.min_match_ratio = g_ctrlgMinMatchRatio;
    {
        float k = 1.0f;
        if (p.search_mode == CmaesRefineV3R::SearchMode::SIX_DOF_RIGID)
            k = 0.7f;
        else if (p.search_mode == CmaesRefineV3R::SearchMode::FOUR_DOF_XYRXRY)
            k = 0.5f;
        if (k != 1.0f) {
            p.tx_range        *= k;
            p.ty_range        *= k;
            p.tz_range        *= k;   // unused by FOUR_DOF (tz=0) but harmless
            p.rx_range        *= k;
            p.ry_range        *= k;
            p.rz_range        *= k;   // unused by FOUR_DOF (rz=0) but harmless
            p.jitter_local_t  *= k;
            p.jitter_global_t *= k;
        }
    }

    // ----- Phase C2a: AR visibility (raycast + rim CC rescue) -------
    // Two-stage:
    //   (a) BVH raycast from origin -> base visibility for the entire
    //       organ. This is the trustworthy front/back classifier.
    //   (b) Rim-subgraph BFS rescue: rim vertices wrongly tagged
    //       occluded by raycast self-occlusion (the rim band blocks
    //       its own adjacent triangles where curvature is sharp) are
    //       rehydrated. Expansion walks rim-rim edges only, gated by
    //       a per-vertex normal test so the BFS cannot wrap around to
    //       the genuinely back-facing rim.
    //
    // Why this is preferred over the standalone normal test:
    //   - Raycast handles the global front/back partition correctly
    //     (including non-convex / concave pockets), which the normal
    //     test cannot.
    //   - The rescue is local (rim-only, 1+ hop adjacency, normal
    //     gate) so the back rim never propagates in.
    //   - Cost ~16 ms (raycast) + ~1 ms (rescue), once per Ctrl+G.
    if (p.use_arvis_filter) {
        const auto t_arv0 = clk::now();

        // Ensure normals are available (used by the rescue gate).
        if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
            std::cout << "[Ctrl+G/AR-vis] WARN: normals missing -- "
                      << "computing from faces..." << std::endl;
            Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
        }

        // ---- Stage (a): BVH raycast from origin (base visibility) ----
        Reg3D::BVHTree bvh;
        bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
        const float diag = (g_liverRegion.bbox_diag > 0.0f)
                               ? g_liverRegion.bbox_diag : g_sceneDiag;
        // AR camera nominally at origin, look-at +Z (PoseLibrary
        // AutoProbe convention). We pull the raycast origin back
        // slightly along -Z so that even when a previous Ctrl+G
        // produced a scale-blowup transform (mesh straddles or wraps
        // the true origin), the raycast still has the whole organ in
        // front of it. Direction +Z is unchanged so front/back
        // semantics are preserved. Pull-back is kept small (0.2*diag)
        // because raycastVisibilityBVH's tol is fixed at 1e-3*diag,
        // and a longer ray path means stricter "true" occlusion —
        // too much pull-back legitimately culls back-side vertices.
        const glm::vec3 ar_cam_pos(0.0f, 0.0f, -0.2f * diag);
        LiverRegionLabel::raycastVisibilityBVH(
            *liverMesh3D, bvh, ar_cam_pos, diag, p.arvis_orig);
        int n_vis_raycast = 0;
        for (uint8_t v : p.arvis_orig) if (v) n_vis_raycast++;

        // ---- Stage (b): rim subgraph BFS rescue ----
        // For each rim vertex left occluded by raycast, promote to
        // visible if it is reachable in the rim-only adjacency graph
        // from a currently-visible rim vertex AND its normal still
        // faces the camera (cos(n, -pos) > kGateCos). The cosine gate
        // prevents the BFS from leaking into back-side rim, which
        // along a continuous rim loop has strongly negative cos.
        const size_t N = liverMesh3D->mVertices.size() / 3;
        int n_rescued = 0;
        if (g_liverRegion.labels.size() == N) {
            // Rim<->Rim edges from triangles. mIndices is flat.
            std::vector<std::vector<int>> rim_adj(N);
            const auto& idx = liverMesh3D->mIndices;
            auto isRim = [&](int v) {
                return g_liverRegion.labels[v] == LiverRegionLabel::RIM;
            };
            auto addEdge = [&](int u, int v) {
                rim_adj[u].push_back(v);
                rim_adj[v].push_back(u);
            };
            for (size_t t = 0; t + 2 < idx.size(); t += 3) {
                const int a = (int)idx[t + 0];
                const int b = (int)idx[t + 1];
                const int c = (int)idx[t + 2];
                const bool ra = isRim(a);
                const bool rb = isRim(b);
                const bool rc = isRim(c);
                if (ra && rb) addEdge(a, b);
                if (rb && rc) addEdge(b, c);
                if (rc && ra) addEdge(c, a);
            }

            // Normal gate: cos angle between normal and (cam - pos).
            // Pure front: cos > 0;  exact silhouette: cos == 0;
            // back-facing: cos < 0. We allow up to a small negative
            // value to catch silhouette / slight-back-facing rim that
            // is part of the "front rim band" the user wants to keep.
            constexpr float kGateCos = -0.2f;

            // Seed BFS with already-visible rim vertices.
            std::vector<int> queue;
            queue.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                if (p.arvis_orig[i] && isRim((int)i)) {
                    queue.push_back((int)i);
                }
            }
            size_t qhead = 0;
            while (qhead < queue.size()) {
                const int u = queue[qhead++];
                for (int w : rim_adj[u]) {
                    if (p.arvis_orig[w]) continue;  // already visible
                    const glm::vec3 pw(
                        liverMesh3D->mVertices[3*w + 0],
                        liverMesh3D->mVertices[3*w + 1],
                        liverMesh3D->mVertices[3*w + 2]);
                    const glm::vec3 nw(
                        liverMesh3D->mNormals[3*w + 0],
                        liverMesh3D->mNormals[3*w + 1],
                        liverMesh3D->mNormals[3*w + 2]);
                    // View direction from the (pulled-back) AR camera
                    // to this vertex. Keep gate consistent with the
                    // raycast origin used above.
                    const glm::vec3 view = pw - ar_cam_pos;
                    const float lv = glm::length(view);
                    const float ln = glm::length(nw);
                    if (lv < 1e-6f || ln < 1e-6f) continue;
                    const float cos_nv = glm::dot(nw, -view) / (ln * lv);
                    if (cos_nv > kGateCos) {
                        p.arvis_orig[w] = 1;
                        ++n_rescued;
                        queue.push_back(w);
                    }
                }
            }
        }

        const auto t_arv1 = clk::now();
        int n_vis_total = 0;
        for (uint8_t v : p.arvis_orig) if (v) n_vis_total++;
        std::cout << "[Ctrl+G/AR-vis] visible=" << n_vis_total
                  << "/" << N
                  << "  (raycast=" << n_vis_raycast
                  << "  +rim_rescued=" << n_rescued << ")"
                  << "  time=" << ms_dur(t_arv1 - t_arv0) << " ms"
                  << "  diag=" << diag
                  << std::endl;
        stamp("C2a. AR visibility (raycast+rim rescue)");
    }

    // ----- Phase C2b: rim/redge-flag arrays (opt-in; beta>0 or gamma>0) ---
    p.gamma_redge_weight = std::max(0.0f, g_ctrlgGammaRedgeWeight);  // [Phase 7c]
    p.bidirectional_matching = g_ctrlgBidirMatching;                 // [Phase 7c] (V3R only)
    p.bidirectional_all_points = g_ctrlgBidirAllPoints;              // [Phase 7c] (A)
    if (p.beta_rim_weight > 0.0f || p.gamma_redge_weight > 0.0f) {
        // Source RIM flag (mesh-intrinsic, LiverRegionLabel::RIM).
        p.is_rim_orig.assign(g_liverRegion.labels.size(), 0);
        for (size_t i = 0; i < g_liverRegion.labels.size(); i++) {
            if (g_liverRegion.labels[i] == LiverRegionLabel::RIM) {
                p.is_rim_orig[i] = 1;
            }
        }

        // [Phase 7c] Source REDGE flag: ANTERIOR_CORE のオクルーディング
        //   輪郭 = 境界(grazing) ∩ 非RIM ∩ 非後面。populateDebugSourceRidge
        //   と同じ判定を run 開始ポーズで一度計算 (固定)。RIM とは排他なので
        //   is_rim_orig と同時に 1 にならない → 重みで二重計上しない。
        int n_redge_src = 0;
        if (p.gamma_redge_weight > 0.0f) {
            if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
                Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
            }
            const float rdiag = (g_liverRegion.bbox_diag > 0.0f)
                                   ? g_liverRegion.bbox_diag : g_sceneDiag;
            const glm::vec3 rcam(0.0f, 0.0f, -0.2f * rdiag);
            const float rband = std::max(0.02f, g_ctrlgRidgeCosBand);
            p.is_redge_orig.assign(g_liverRegion.labels.size(), 0);
            const auto& RV = liverMesh3D->mVertices;
            const auto& RN = liverMesh3D->mNormals;
            for (size_t i = 0; i < g_liverRegion.labels.size(); i++) {
                if (g_liverRegion.labels[i] != LiverRegionLabel::ANTERIOR_CORE) continue;
                if (i*3+2 >= RV.size() || i*3+2 >= RN.size()) continue;
                const glm::vec3 pv(RV[i*3], RV[i*3+1], RV[i*3+2]);
                const glm::vec3 nv(RN[i*3], RN[i*3+1], RN[i*3+2]);
                const glm::vec3 vv = pv - rcam;
                const float lv = glm::length(vv), ln = glm::length(nv);
                if (lv < 1e-6f || ln < 1e-6f) continue;
                const float cos_nv = glm::dot(nv, -vv) / (ln * lv);
                if (std::fabs(cos_nv) < rband) { p.is_redge_orig[i] = 1; n_redge_src++; }
            }
        }

        // Target band (rim/redge 共有): boundaryDist from cached cloud.
        // May be empty if extracted via the legacy grid path; the V3R
        // driver then sees empty tgt and skips weighting silently.
        if (targetCloud->hasBoundaryDist() &&
            targetCloud->boundaryDist.size() == tgt_points.size())
        {
            p.tgt_boundary_dist_full = targetCloud->boundaryDist;
        } else {
            std::cout << "[Ctrl+G/rim] target cloud has no boundaryDist "
                         "(legacy path?). Rim/redge weighting will be inactive."
                      << std::endl;
            p.tgt_boundary_dist_full.clear();
        }
        int n_rim_src = 0;
        for (uint8_t v : p.is_rim_orig) if (v) n_rim_src++;
        std::cout << "[Ctrl+G/rim] beta=" << p.beta_rim_weight
                  << " gamma=" << p.gamma_redge_weight
                  << "  src_rim=" << n_rim_src
                  << "/" << p.is_rim_orig.size()
                  << "  src_redge=" << n_redge_src
                  << "  tgt_bdist_avail="
                  << (p.tgt_boundary_dist_full.empty() ? "NO" : "YES")
                  << "  thresh=" << p.rim_tgt_threshold_px << "px"
                  << std::endl;
    }

    // ----- Phase C2c: caudal-only flag (R-feat-2; anatomical filter) -
    // Source-side filter that complements AR-vis on a different axis:
    //   AR-vis = view-based (raycast from AR camera; depends on current
    //            mesh pose / scale, needs camera tuning).
    //   Caudal = anatomy-based (LiverCranioCaudalLabel; bound to vertex
    //            index, transform-invariant, zero compute cost here).
    //
    // The two filters compose inside V3R driver via the AND/OR combine
    // mode (p.arvis_caudal_combine). When only one is ON, that one
    // alone applies. When neither is ON, the V3 byte-identical contract
    // is preserved as before.
    //
    // Degradation policy: if user requested caudal-only but g_liverCC
    // is not yet computed (Shift+H or Apply Init Pose not pressed), we
    // disable the filter for THIS run (warn, then continue). Aborting
    // would be hostile because the rest of Ctrl+G (quadrant, arvis,
    // beta) is unaffected.
    if (p.use_caudal_only) {
        if (!g_liverCC.valid()) {
            std::cerr << "[Ctrl+G/caudal] Only-Caudal requested but "
                         "g_liverCC not yet computed. Press Shift+H "
                         "(or Apply Init Pose) to populate. "
                         "Disabling caudal filter for this run."
                      << std::endl;
            p.use_caudal_only = false;
            p.is_caudal_orig.clear();
        } else if (g_liverCC.labels.size() != start_liver_verts.size()) {
            std::cerr << "[Ctrl+G/caudal] g_liverCC.labels size mismatch "
                      << "(cc=" << g_liverCC.labels.size()
                      << " vs verts=" << start_liver_verts.size()
                      << "). Disabling caudal filter for this run; "
                         "re-run Shift+H after the mesh change."
                      << std::endl;
            p.use_caudal_only = false;
            p.is_caudal_orig.clear();
        } else {
            // Build per-original-vertex 0/1 mask. The voxel-side
            // mapping (is_caudal_voxel) is derived later inside
            // runBipopCmaesV3R via derive_is_caudal_voxel, parallel
            // to how arvis_orig -> arvis_voxel is built.
            p.is_caudal_orig.assign(g_liverCC.labels.size(), 0);
            for (size_t i = 0; i < g_liverCC.labels.size(); i++) {
                if (g_liverCC.labels[i] == LiverCranioCaudalLabel::CAUDAL) {
                    p.is_caudal_orig[i] = 1;
                }
            }
            int n_caudal = 0;
            for (uint8_t v : p.is_caudal_orig) if (v) n_caudal++;
            std::cout << "[Ctrl+G/caudal] caudal_only=ON"
                      << "  n_caudal=" << n_caudal
                      << "/" << p.is_caudal_orig.size()
                      << "  cc_confidence="
                      << std::fixed << std::setprecision(3)
                      << g_liverCC.cc.confidence
                      << std::defaultfloat
                      << (g_liverCC.cc.weak ? "  [WEAK]" : "  [OK]")
                      << "  flipped_manual="
                      << (g_liverCC.cc.flipped_manual ? "true" : "false")
                      << "  combine="
                      << (p.arvis_caudal_combine == 0 ? "AND" : "OR")
                      << std::endl;
            stamp("C2c. caudal-only flag (mesh-intrinsic CC)");
        }
    } else {
        p.is_caudal_orig.clear();
    }

    // ----- Populate RIM-pair visualization buffers (opt-in) ---------
    // Independent from the optimization itself: even with beta=0 the
    // user may want to see what the rim sets look like. Buffers are
    // (re)populated at every Ctrl+G entry so they reflect the current
    // mesh / target / threshold.
    g_ctrlgRimSrcVertIdx.clear();
    g_ctrlgRimTgtPos.clear();
    g_ctrlgRimVizAvailable = false;
    if (g_ctrlgShowRimPairs) {
        // Source: vertices that pass (a) the user-selected quadrant
        // mask, (b) RIM region label, and (c) the SAME subset filter
        // that CMA-ES sees -- i.e. arvis and/or caudal combined via
        // p.arvis_caudal_combine. Without mirroring this, the rim
        // spheres would advertise vertices that the optimiser never
        // actually uses, which is confusing.
        //
        // p.use_caudal_only may have been disabled above (degradation
        // path: g_liverCC missing / size mismatch); reflect the
        // post-degradation state, not the raw UI toggle, so what we
        // draw is exactly what CMA-ES uses.
        const bool a_on = p.use_arvis_filter && !p.arvis_orig.empty();
        const bool c_on = p.use_caudal_only  && !p.is_caudal_orig.empty();
        const uint8_t cmode = p.arvis_caudal_combine;  // 0=AND, 1=OR
        std::vector<int> quadAllowed = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels,
            g_liverLR.labels,
            quadrant_mask);
        for (int idx : quadAllowed) {
            if (idx < 0 ||
                (size_t)idx >= g_liverRegion.labels.size()) continue;
            if (g_liverRegion.labels[idx] != LiverRegionLabel::RIM) continue;
            // Empty array = "all pass" for that filter (matches V3R
            // filter_by_quadrant_with_arvis_caudal semantics).
            const bool a = (!a_on) ||
                           ((size_t)idx < p.arvis_orig.size() &&
                            p.arvis_orig[idx]);
            const bool c = (!c_on) ||
                           ((size_t)idx < p.is_caudal_orig.size() &&
                            p.is_caudal_orig[idx]);
            bool pass;
            if (a_on && c_on) {
                pass = (cmode == 0) ? (a && c) : (a || c);
            } else {
                pass = a && c;  // one is unconditionally true via shortcut
            }
            if (!pass) continue;
            g_ctrlgRimSrcVertIdx.push_back(idx);
        }
        // Target: points with boundaryDist below threshold, also
        // exclude instrument-occluded points consistent with Shift+P.
        if (targetCloud->hasBoundaryDist()) {
            const bool useInst = targetCloud->hasInstrumentDist() &&
                                 targetCloud->instrumentDist.size() == tgt_points.size();
            for (size_t i = 0; i < tgt_points.size(); i++) {
                if (targetCloud->boundaryDist[i] >= g_ctrlgRimTgtThreshPx) continue;
                if (useInst &&
                    targetCloud->instrumentDist[i] < g_instrumentPxThresh)
                {
                    continue;
                }
                g_ctrlgRimTgtPos.push_back(tgt_points[i]);
            }
        }
        g_ctrlgRimVizAvailable = !g_ctrlgRimSrcVertIdx.empty() ||
                                 !g_ctrlgRimTgtPos.empty();
        std::cout << "[Ctrl+G/RimViz] src=" << g_ctrlgRimSrcVertIdx.size()
                  << "  tgt=" << g_ctrlgRimTgtPos.size()
                  << "  quad=" << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
                  << "  (AR-vis filter "
                  << (a_on ? "ON" : "OFF")
                  << ", Caudal "
                  << (c_on ? "ON" : "OFF")
                  << ", combine "
                  << ((a_on && c_on) ? (cmode == 0 ? "AND" : "OR") : "n/a")
                  << ")"
                  << std::endl;
    }

    // ----- Phase D: Call into the pure V3R driver -------------------
    // Internally voxelizes src+tgt once (same as V3), then computes
    // voxel_to_orig (NN reverse map, ~1 ms) and subset_idx_voxel
    // (label filter, <1 ms), then runs 10 BIPOP CMA-ES restarts with
    // a subset KDTree per Run.
    CmaesRefine::ResultV3 r = CmaesRefineV3R::runBipopCmaesV3R(
        start_liver_verts, start_liver_normals, tgt_points,
        p, rmse_before, init_matched,
        outer_seed, cma_base);
    stamp("D. CMA-ES V3R driver (10 runs + voxel + subset)");

    // ----- Phase E: Apply best_world_matrix to all organs -----------
    // Verbatim copy of V3's Phase E. The matrix already encodes
    // (jitter then best_srt) and is applied uniformly to all organs;
    // the subset filtering only affected the inner-loop KDTree, not
    // the final transform.
    if (r.improved && r.best_run_idx >= 0) {
        const glm::mat4& M = r.best_world_matrix;
        const glm::mat3  normalMat =
            glm::mat3(glm::transpose(glm::inverse(M)));

        double t_apply_sum = 0.0;
        double t_setup_sum = 0.0;
        int    organ_idx   = 0;
        for (auto* mesh : organs) {
            if (!mesh) { organ_idx++; continue; }

            const auto t_apply_start = clk::now();
            auto& v = mesh->mVertices;
            auto& n = mesh->mNormals;
            for (size_t i = 0; i + 2 < v.size(); i += 3) {
                glm::vec4 p4(v[i], v[i+1], v[i+2], 1.0f);
                glm::vec4 tp = M * p4;
                v[i]   = tp.x; v[i+1] = tp.y; v[i+2] = tp.z;
            }
            for (size_t i = 0; i + 2 < n.size(); i += 3) {
                glm::vec3 nm(n[i], n[i+1], n[i+2]);
                glm::vec3 tn = normalMat * nm;
                float len = glm::length(tn);
                if (len > 1e-8f) tn /= len;
                n[i]   = tn.x; n[i+1] = tn.y; n[i+2] = tn.z;
            }
            const auto t_apply_end = clk::now();
            t_apply_sum += ms_dur(t_apply_end - t_apply_start);

            const auto t_setup_start = clk::now();
            setUp(*mesh);
            const auto t_setup_end = clk::now();
            const double t_one_setup = ms_dur(t_setup_end - t_setup_start);
            t_setup_sum += t_one_setup;
            std::cout << "[CtrlG/Time]   organ[" << organ_idx
                      << "] verts=" << (v.size() / 3)
                      << " apply=" << std::fixed << std::setprecision(1)
                      << ms_dur(t_apply_end - t_apply_start) << "ms"
                      << " setUp=" << t_one_setup << "ms"
                      << std::defaultfloat << std::endl;
            organ_idx++;
        }
        std::cout << "[CtrlG/Time]   apply_sum=" << std::fixed
                  << std::setprecision(1) << t_apply_sum << "ms"
                  << "  setUp_sum=" << t_setup_sum << "ms"
                  << std::defaultfloat << std::endl;
    } else {
        std::cout << "[CtrlG/Time]   (no improvement, skipped apply)"
                  << std::endl;
    }
    stamp("E. apply_matrix + setUp x6");

    // ----- Phase F: Confirm via computeUnifiedMetrics ---------------
    // [Phase F skip 撤回 — 2026-05-21]
    // 旧コード: r.improved==false なら pose 不変だから computeUnifiedMetrics を
    // スキップして 122-155 ms 節約していた (FULL RMSE 高速化チャット
    // 2026-05-15 で V3 と同じパターンが導入された)。
    //
    // 問題: g_metricsValid=true を立てて帰った後、次の Ctrl+G までの間に
    // Undo / PoseLibrary 復元 / Ctrl+Shift+G の accept / 手動移動 /
    // AR overlay 切替 などで pose が動く経路があると、
    // registrationHandle.compRmse は古い値のまま固まり、次の Phase A も
    // キャッシュを読んで rmse_before が古い値で凍結する。
    // 結果として「改善判定の基準が古い」状態となり、Ctrl+G が永久に
    // NO CHANGE で反応しないように見える既知のリグレッション。
    // (現場ログ: callIdx 46-57 連続で start RMSE=0.0278526 ビット完全一致、
    // 一方で RIM-only RMSE / IoU_occluded / Containment は変動継続、
    // matched=762146 で target 全数 saturate しているのも一因)。
    //
    // 安全側に倒して r.improved に関係なく毎回再計算する。コスト 135-225ms
    // 増 (V3R driver 420ms に対して +30-50%、許容範囲)。
    // 高速化を再導入する場合は、5/1 の Phase V3-1 完了後計画 §7.3 で議論
    // された session_id ガード、もしくは pose を変える全箇所での明示
    // g_metricsValid=false 徹底 (grep が必要) のどちらかが必要。
    computeUnifiedMetrics();
    g_metricsValid = true;
    const float rmse_after = registrationHandle.compRmse;
    stamp("F. post_computeUnifiedMetrics");

    // ----- Phase F.5: RIM-only RMSE "after" -------------------------
    // Phase E で liverMesh3D->mVertices が更新済み (improved 時)。NO
    // CHANGE 時は start_liver_verts と内容一致 → before==after が出る。
    // どちらでも問題ない。
    std::vector<glm::vec3> liver_verts_after;
    {
        const auto& v = liverMesh3D->mVertices;
        liver_verts_after.reserve(v.size() / 3);
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            liver_verts_after.emplace_back(v[i], v[i+1], v[i+2]);
        }
    }
    int rim_diag_n_src_rim_after = 0;
    int rim_diag_matched_after   = 0;
    // [Phase A] Capture (src_full_idx, tgt_pos) pairs into the publish-
    // and-consume globals so poseSaveToLibrary can copy them onto this
    // entry's PoseEntry (Phase B). The "before" call (Phase C.5) above
    // intentionally omits these args — pairs are only meaningful at the
    // final pose, and the legacy 6-arg call there stays byte-identical.
    // Defensive clear() in case Phase F.5 is ever re-entered in one
    // session (currently it isn't, but matches the function's
    // append-only contract).
    g_lastRimPairSrcVertIdx.clear();
    g_lastRimPairTgtPos.clear();
    const float rmse_rim_after = compute_rim_only_rmse_diag(
        liver_verts_after, rim_diag_is_rim_src, rim_diag_tgt_rim_points,
        max_dist_sq_rim_diag,
        rim_diag_n_src_rim_after, rim_diag_matched_after,
        &g_lastRimPairSrcVertIdx, &g_lastRimPairTgtPos);
    stamp("F.5. rim_only_rmse_after");

    // ----- Phase F.5b: IoU_occluded diagnostic (display-only) -------
    // Mirrors Ctrl+Shift+G Phase E candidate eval (rasterize_iou2d_v3rs)
    // at the post-Apply pose so the same IoU number the V3RS optimiser
    // optimises against can be read off Ctrl+G entries too. This is
    // strictly held-out — never participates in the optimiser cost or
    // the PoseLibrary acceptance gate. Cost: ~5 ms at step=8.
    //
    // Gating policy (HANDOVER A·A·A): "always when measurable".
    //   - g_boundaryDistMap.valid (SAM2 mask loaded) → compute
    //   - otherwise → publish 0.0f and PoseLibrary shows "—"
    // Instrument-occlusion is applied AUTOMATICALLY when the
    // instrument distance map loads cleanly — the Ctrl+Shift+G UI
    // toggle is NOT consulted, so PoseLibrary's IoU_occ column has
    // stable semantics regardless of how V3RS happens to be
    // configured. F9 has its own visualisation toggle for switching
    // the displayed composite (see SilOverlayDebug.h).
    float iou_occluded_after  = -1.0f;
    bool  iou_occ_inst_active = false;
    // [NEW V3RS-CONTAIN] precision = |src∩tgt|/|src|, recall = |src∩tgt|/|tgt|.
    // Together they identify the asymmetric overlap failure modes that
    // a scalar IoU hides:
    //   recall ≈ 1, precision << 1  → source overshoots target (oversized)
    //   recall << 1, precision ≈ 1  → source undershoots target (undersized)
    //   both << 1                   → positional / rotational mismatch
    // Held-out diagnostic; never gates acceptance. Sentinel -1 = N/A.
    float iou_occ_precision  = -1.0f;
    float iou_occ_recall     = -1.0f;
    int   iou_occ_src_cells  = 0;
    int   iou_occ_tgt_cells  = 0;
    int   iou_occ_int_cells  = 0;
    if (g_boundaryDistMap.valid
        && g_boundaryDistMap.width  > 1
        && g_boundaryDistMap.height > 1
        && g_boundaryDistMap.data.size() ==
               (size_t)g_boundaryDistMap.width *
               (size_t)g_boundaryDistMap.height
        && liverMesh3D
        && !liverMesh3D->mIndices.empty())
    {
        const glm::mat4 sil_view = buildSilhouetteView();
        const glm::mat4 sil_proj = buildSilhouetteProj();
        const glm::mat4 sil_mvp  = sil_proj * sil_view;
        const int sil_w = g_boundaryDistMap.width;
        const int sil_h = g_boundaryDistMap.height;

        // mIndices is std::vector<GLuint>; rasterizer wants
        // std::vector<uint32_t>. On every platform we ship to these
        // are the same width type, but vector types are distinct, so
        // we copy through the iterator constructor — same pattern
        // Ctrl+Shift+G uses to build p.sil_indices.
        std::vector<uint32_t> sil_indices_full(
            liverMesh3D->mIndices.begin(),
            liverMesh3D->mIndices.end());

        // Instrument-occlusion mask: AUTO. We always try to load and
        // use the instrument mask when it's available — the IoU_occ
        // diagnostic is held-out display data, "use the best available
        // signal" is the right policy regardless of how V3RS is
        // configured. The Ctrl+Shift+G UI toggle
        // (g_ctrlgsIgnoreInstrument) governs only V3RS's cost function
        // and is deliberately NOT consulted here, so the value in
        // PoseLibrary's IoU_occ column has a stable, well-defined
        // meaning across sessions: "IoU with instrument occlusion
        // applied if the mask was loadable".
        //
        // F9 has its own user-facing checkbox (in SilOverlayDebug.h)
        // for switching the *displayed* composite between occlusion-on
        // and occlusion-off — that's purely a visualisation toggle and
        // does not affect the number stored in the PoseLibrary entry.
        const std::vector<float>* inst_ptr = nullptr;
        float inst_thresh = 0.0f;
        const bool inst_loaded = ensureInstrumentDistMap();
        if (inst_loaded
            && g_instrumentDistMap.valid
            && g_instrumentDistMap.width  == sil_w
            && g_instrumentDistMap.height == sil_h
            && g_instrumentDistMap.data.size() ==
                   (size_t)sil_w * (size_t)sil_h)
        {
            inst_ptr            = &g_instrumentDistMap.data;
            // Use the master instrument threshold (g_instrumentPxThresh,
            // default 80 px) — the same value the boundary cloud and
            // the rest of the system rely on. The Ctrl+Shift+G UI
            // slider g_ctrlgsInstrumentThreshPx is intentionally NOT
            // consulted here (it's a cost-function tuning knob for
            // V3RS and defaults to 0, which would effectively disable
            // the filter for the Ctrl+G diagnostic).
            inst_thresh         = std::max(0.0f, g_instrumentPxThresh);
            iou_occ_inst_active = true;
        }

        // Single eval at post-Apply pose. liver_verts_after already
        // incorporates r.best_world_matrix (Phase E wrote it back into
        // liverMesh3D->mVertices, and we re-built liver_verts_after
        // from those vertices in Phase F.5). So mvp = proj * view * I.
        //
        // [NEW V3RS-CONTAIN] We enable the hitmap/tmask capture buffers
        // so a single rasterize pass yields both the IoU scalar and the
        // raw cell-occupancy maps used to compute precision/recall in
        // the loop just below. step=8 → ~14k cells, capture cost is ~0.3 ms
        // on top of the rasterize itself (negligible vs. the 5 ms eval).
        std::vector<uint8_t> hitmap_occ, tmask_occ;
        int gw_occ = 0, gh_occ = 0;
        iou_occluded_after = CmaesRefineV3RS::rasterize_iou2d_v3rs(
            liver_verts_after, sil_indices_full, sil_mvp,
            g_boundaryDistMap.data, sil_w, sil_h, /*step=*/8,
            &hitmap_occ, &tmask_occ, &gw_occ, &gh_occ,
            nullptr, nullptr, nullptr,
            /*raster_mode=*/0,
            inst_ptr, inst_thresh);

        // Compute precision / recall in the same pass over the captured
        // cell maps. The rasterizer already cleared instrument-occluded
        // cells when inst_ptr was non-null, so these counts are consistent
        // with the returned IoU regardless of the filter state.
        if (gw_occ > 0 && gh_occ > 0) {
            int inter_c = 0, src_c = 0, tgt_c = 0;
            const size_t N_c = (size_t)gw_occ * (size_t)gh_occ;
            for (size_t i = 0; i < N_c; ++i) {
                const bool s = (hitmap_occ[i] != 0);
                const bool t = (tmask_occ[i]  != 0);
                if (s)         ++src_c;
                if (t)         ++tgt_c;
                if (s && t)    ++inter_c;
            }
            iou_occ_int_cells = inter_c;
            iou_occ_src_cells = src_c;
            iou_occ_tgt_cells = tgt_c;
            iou_occ_precision = (src_c > 0)
                ? (float)inter_c / (float)src_c : 0.0f;
            iou_occ_recall    = (tgt_c > 0)
                ? (float)inter_c / (float)tgt_c : 0.0f;
        }
    }
    stamp("F.5b. iou_occluded_diag");

    // Publish to PoseLibrary's IoU_occluded global. Uses the same
    // sentinel convention as Ctrl+Shift+G:
    //   > 0.0f  → measured value; PoseLibrary column shows the number
    //     0.0f  → "not measured"; PoseLibrary column shows "—"
    // Display-only — does NOT affect g_poseLibraryUseOccludedForAccept
    // gate semantics (that toggle still applies whichever method the
    // user runs, but for Ctrl+G the user-selected acceptance criterion
    // stays RMSE per the HANDOVER A·A·A decision).
    g_lastSilOccludedIoU2D =
        (iou_occluded_after >= 0.0f) ? iou_occluded_after : 0.0f;

    // [NEW V3RS-CONTAIN] Publish containment metrics. Sentinel -1 means
    // "not measured" (e.g. boundary map invalid, rasterize failed) — the
    // PoseLibrary consumer treats that as N/A and shows "—" in the table.
    g_lastIoUOccPrecision = iou_occ_precision;
    g_lastIoUOccRecall    = iou_occ_recall;

    // ----- Phase F.5c: F9 silhouette overlay capture ----------------
    // Pushes a Final-slot composite to SilOverlay::g_silOverlay so the
    // F9 window shows what Ctrl+G actually applied (rim + silhouette
    // + target overlay) without re-running anything. Shares the slot
    // with Ctrl+Shift+G's Final capture — the most recent press wins.
    // Cost: ~15 ms.
    //
    // Same gating as F.5b: needs SAM2 mask + a liver mesh with indices.
    // When skipped, the F9 window simply shows whatever the previous
    // capture left there (or empty, after the reset at function head).
    if (g_boundaryDistMap.valid
        && g_boundaryDistMap.width  > 1
        && g_boundaryDistMap.height > 1
        && g_boundaryDistMap.data.size() ==
               (size_t)g_boundaryDistMap.width *
               (size_t)g_boundaryDistMap.height
        && liverMesh3D
        && !liverMesh3D->mIndices.empty())
    {
        const glm::mat4 sil_view_f = buildSilhouetteView();
        const glm::mat4 sil_proj_f = buildSilhouetteProj();
        const int sil_w_f = g_boundaryDistMap.width;
        const int sil_h_f = g_boundaryDistMap.height;

        // Rebuild sil_indices (the F.5b copy went out of scope when
        // its `if` block closed). Cheap (~µs for 20k tris).
        std::vector<uint32_t> sil_indices_f(
            liverMesh3D->mIndices.begin(),
            liverMesh3D->mIndices.end());

        // Reuse F.5b's instrument-filter decision so the F9 composite
        // matches the IoU number we just published. If F.5b decided
        // the instrument filter is unusable (mask missing / size
        // mismatch / toggle off), F.5c uses the same plain path.
        const std::vector<float>* inst_ptr_f =
            iou_occ_inst_active ? &g_instrumentDistMap.data : nullptr;
        const float inst_thresh_f =
            iou_occ_inst_active ? std::max(0.0f, g_instrumentPxThresh)
                                : 0.0f;

        // best_run_idx for the "Best Run was N" caption. scale_value
        // comes from r.best_srt.scale, same as Ctrl+Shift+G captureFinal.
        // rim_sil_max_px = 0 because Ctrl+G does not use rim_sil_loss
        // in its cost — the bottom row of the 6-panel layout will
        // simply not render that diagnostic; expected.
        SilOverlay::captureFinal(
            SilOverlay::g_silOverlay, r.best_run_idx, liverMesh3D,
            sil_indices_f, sil_view_f, sil_proj_f, g_boundaryDistMap.data,
            sil_w_f, sil_h_f, /*step=*/8, r.best_srt.scale,
            inst_ptr_f, inst_thresh_f,
            /*rim_sil_max_px=*/0.0f,
            /*is_rim_anatomic_per_vertex=*/nullptr);
    }
    stamp("F.5c. f9_overlay_capture");

    // V3R rim diagnostic: publish the post-CMA-ES rim metrics to the
    // PoseLibrary globals so poseSaveToLibrary can record them onto
    // this session's entry. Display-only; never gates acceptance.
    //
    // The "after" snapshot is the right value to publish here because
    // poseSaveToLibrary runs AFTER this function returns, and the entry
    // reflects the final (post-Apply) pose of liverMesh3D. The "before"
    // value is consumed only by the [Ctrl+G] RIM-only log line below.
    //
    // -1.0f when matched==0 or compute returned the N/A sentinel. The
    // consume-and-clear inside poseSaveToLibrary protects subsequent
    // non-Ctrl+G saves from seeing this value.
    g_lastRimRmse     = rmse_rim_after;   // -1.0f if N/A
    g_lastRimMatched  = rim_diag_matched_after;
    g_lastRimTgtTotal = (int)rim_diag_tgt_rim_points.size();
    g_lastRimSrcTotal = rim_diag_n_src_rim_after;

    std::cout << std::defaultfloat << std::setprecision(6);
    const float improvement = rmse_before - rmse_after;
    std::cout << "[Ctrl+G] Best: " << rmse_before << " -> " << rmse_after
              << " (delta=" << improvement << ")"
              << "  best_run="
              << (r.best_run_idx < 0 ? std::string("none")
                                     : std::to_string(r.best_run_idx + 1))
              << "  total_gens=" << r.total_generations
              << "  Q=" << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
              << (improvement > 0.001f ? "  [IMPROVED]" : "  [NO CHANGE]")
              << std::endl;

    // ----- RIM-only RMSE diagnostic line. 1 line per Ctrl+G session. -
    // -1.0f は N/A (target に boundaryDist が無いか、rim 集合が空)。
    if (rmse_rim_before > 0.0f && rmse_rim_after > 0.0f) {
        const float rim_delta = rmse_rim_before - rmse_rim_after;
        std::cout << "[Ctrl+G] RIM-only: " << rmse_rim_before
                  << " -> " << rmse_rim_after
                  << " (delta=" << rim_delta << ")"
                  << "  src_rim=" << rim_diag_n_src_rim_before
                  << "  tgt_rim=" << rim_diag_tgt_rim_points.size()
                  << "  matched(before/after)="
                  << rim_diag_matched_before << "/" << rim_diag_matched_after
                  << (rim_delta >  0.0001f ? "  [RIM IMPROVED]"  :
                      rim_delta < -0.0001f ? "  [RIM WORSE]"     :
                                             "  [RIM NO CHANGE]")
                  << std::endl;
    } else {
        std::cout << "[Ctrl+G] RIM-only: N/A"
                  << "  src_rim=" << rim_diag_n_src_rim_before
                  << "  tgt_rim=" << rim_diag_tgt_rim_points.size()
                  << "  (no boundaryDist or empty rim set)" << std::endl;
    }

    // ----- IoU_occluded diagnostic line. 1 line per Ctrl+G session. -
    // Same semantics as the RIM line: display-only, never gates
    // acceptance. Reports whether the instrument filter was active
    // so the number can be compared against Ctrl+Shift+G's IoU_occ
    // for the same scene (same filter setting → byte-identical IoU).
    if (iou_occluded_after >= 0.0f) {
        std::cout << "[Ctrl+G] IoU_occ: " << iou_occluded_after
                  << "  instrument:"
                  << (iou_occ_inst_active ? "ON" : "OFF")
                  << "  (display-only)" << std::endl;
    } else {
        std::cout << "[Ctrl+G] IoU_occ: N/A"
                  << "  (no boundary distance map)" << std::endl;
    }

    // ----- Containment direction (precision / recall). 1 line per Ctrl+G.
    // Identifies which asymmetric failure mode the post-Apply silhouette
    // exhibits — overshoot (src too large), undershoot (src too small),
    // or balanced. Helps the operator decide whether the optimiser is
    // running into the typical "src ⊇ tgt" overfit before rim drift
    // shows up. Display-only.
    if (iou_occ_precision >= 0.0f && iou_occ_recall >= 0.0f) {
        const float dir = iou_occ_recall - iou_occ_precision;
        const char* tag = (std::fabs(dir) < 0.05f) ? "[BALANCED]"
                       : (dir > 0.0f)              ? "[OVERSHOOT src>tgt]"
                                                   : "[UNDERSHOOT src<tgt]";
        // [NEW V3RS-CONTAIN-RATIO] size_ratio = |src|/|tgt| (recall/prec),
        // overshoot_fraction = |src-inter|/|tgt|. These preserve magnitude
        // info that recall saturates and hides. Computed from cell counts
        // so they're sentinel-free when src/tgt are positive.
        const float size_ratio = (iou_occ_tgt_cells > 0)
            ? (float)iou_occ_src_cells / (float)iou_occ_tgt_cells : 0.0f;
        const float overshoot_frac = (iou_occ_tgt_cells > 0)
            ? (float)std::max(0, iou_occ_src_cells - iou_occ_int_cells)
              / (float)iou_occ_tgt_cells
            : 0.0f;
        std::cout << "[Ctrl+G] Containment: size=" << std::fixed
                  << std::setprecision(2) << size_ratio << "x"
                  << "  overshoot=" << std::setprecision(0)
                  << 100.0f * overshoot_frac << "%"
                  << std::defaultfloat << std::setprecision(6)
                  << "  recall=" << iou_occ_recall
                  << "  precision=" << iou_occ_precision
                  << "  (inter=" << iou_occ_int_cells
                  << " src=" << iou_occ_src_cells
                  << " tgt=" << iou_occ_tgt_cells << " cells)  "
                  << tag << std::endl;
    } else {
        std::cout << "[Ctrl+G] Containment: N/A"
                  << "  (IoU_occ not computed)" << std::endl;
    }

    const double t_grand_total = ms_dur(clk::now() - t_phase_start);
    std::cout << "[CtrlG/Time] === GRAND TOTAL: "
              << std::fixed << std::setprecision(1) << t_grand_total
              << " ms ===" << std::defaultfloat << std::endl;

    // [PoseLibrary save fix] HemiAuto / AutoQCR と同様に Ctrl+G 完走時にも
    // state を REGISTERED に上げ、useRegistration を立てる。これがないと
    // poseSaveToLibrary 冒頭の "state != REGISTERED" ガードで初期 pose 直後
    // からの Ctrl+G が PoseLibrary に保存されない問題が起きる (従来は
    // 「Ctrl+G の前に必ず HemiAuto/AutoQCR が走っていて REGISTERED が立って
    //  いる」前提だったため顕在化していなかった)。
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;

    g_callIdx++;  // V3R: 末尾でインクリメント (V1 / V2 / V3 と同じ)
}


// =========================================================
//  Ctrl+Shift+G: V3RS (silhouette-anchored) wrapper.
//  ---------------------------------------------------------
//  STRAIGHT FORK of runBipopCmaesV3R above with:
//    * Params type ParamsV3RS (silhouette fields added)
//    * Phase C2c.5: pull AR camera intrinsics + boundary-dist
//      map; gate by g_boundaryDistMap.valid
//    * Final driver call goes to CmaesRefineV3RS::runBipopCmaesV3RS
//    * Log prefix "[Ctrl+Shift+G/*]" so logs are diffable
//  Everything else is verbatim from the V3R wrapper. Ctrl+G is
//  completely unaffected: it still calls runBipopCmaesV3R and
//  uses ParamsV3R with no silhouette fields touched.
// =========================================================

inline void runBipopCmaesV3RS(uint8_t quadrant_mask) {
    std::cout << "\n=== BIPOP-CMA-ES V3-RS (Ctrl+Shift+G) ===" << std::endl;
    std::cout << "[Ctrl+Shift+G] quadrant_mask = "
              << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
              << "  (0x" << std::hex << (int)quadrant_mask << std::dec
              << ")" << std::endl;

    // [Phase B] Reset the published IoU_occluded so a session that
    // bails out early (e.g. before Phase E) does not leak a stale value
    // into the next poseSaveToLibrary call. Phase E re-publishes the
    // fresh post-apply value before the save call below.
    g_lastSilOccludedIoU2D = 0.0f;

    if (registrationHandle.compRmse == 0.0f) {
        std::cerr << "[Ctrl+Shift+G] No registration yet. Run HemiAuto (O) first."
                  << std::endl;
        return;
    }

    // ----- V3R label validity gate ----------------------------------
    // Region/LR labels must be populated before any non-QUAD_ALL run;
    // for QUAD_ALL we still verify them so a misconfigured run cannot
    // silently fall back to "no filtering" semantics that the user
    // didn't intend.
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[Ctrl+Shift+G] Region/LR labels not computed: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << "  LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << ". Run HemiAuto (O) first to populate them."
                  << std::endl;
        return;
    }

    // ----- V3R timing diagnostics (mirrors V3 phase stamps) ----------
    using clk    = std::chrono::high_resolution_clock;
    auto   ms_dur = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };
    const auto t_phase_start = clk::now();
    auto t_prev = t_phase_start;
    auto stamp  = [&](const char* label) {
        const auto t_now = clk::now();
        std::cout << "[CtrlGS/Time] " << label << " : "
                  << std::fixed << std::setprecision(1)
                  << ms_dur(t_now - t_prev) << " ms"
                  << "  (cumulative " << ms_dur(t_now - t_phase_start)
                  << " ms)" << std::defaultfloat << std::endl;
        t_prev = t_now;
    };

    // ----- Seed determination (formulas IDENTICAL to V1 / V2 / V3) -
    const uint32_t outer_seed = g_trialSeed + 1000u + g_callIdx * 97u;
    const uint32_t cma_base   = g_trialSeed + 2000u + g_callIdx * 10u;
    std::cout << "[Seed Ctrl+Shift+G] BIPOP outer=" << outer_seed
              << "  CMA-ES base=" << cma_base
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    auto organs = getOrganList();

    // ----- Phase A: Pre-session computeUnifiedMetrics ---------------
    // [Phase A skip 撤回 — 2026-05-21]
    // 旧コード: g_metricsValid=true なら cache を流用していた。
    // 詳細は V3 (Shift+G) Phase A の同コメント参照。V3RS では
    // Phase E の中で必ず computeUnifiedMetrics が走るので Phase F は
    // 元から再計算されていたが、Phase A 側の cache 嘘は同じ問題を
    // 引き起こす (init_iou2d も同様に stale 化する)。
    // Phase F と対称に毎回再計算する。
    computeUnifiedMetrics();
    g_metricsValid = false;  // consumed; Phase F will re-validate
    const float rmse_before  = registrationHandle.compRmse;
    const float init_iou2d   = registrationHandle.compIoU2D;   // composite gate baseline
    const int   init_matched = registrationHandle.compCount;
    std::cout << "[Ctrl+Shift+G] start RMSE=" << rmse_before
              << "  IoU2D=" << init_iou2d
              << "  init_matched=" << init_matched << std::endl;
    stamp("A. pre_computeUnifiedMetrics");

    if (!liverMesh3D) {
        std::cerr << "[Ctrl+Shift+G] liverMesh3D is null; aborting." << std::endl;
        return;
    }

    // ----- Phase B: Convert liver vertices / normals to vec3 --------
    std::vector<glm::vec3> start_liver_verts;
    std::vector<glm::vec3> start_liver_normals;
    {
        const auto& v = liverMesh3D->mVertices;
        const auto& n = liverMesh3D->mNormals;
        start_liver_verts.reserve(v.size() / 3);
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            start_liver_verts.emplace_back(v[i], v[i+1], v[i+2]);
        }
        start_liver_normals.reserve(n.size() / 3);
        for (size_t i = 0; i + 2 < n.size(); i += 3) {
            start_liver_normals.emplace_back(n[i], n[i+1], n[i+2]);
        }
    }
    stamp("B. vec3_conversion");

    // Label vs. vertex size sanity check (HANDOVER §8.4).
    if (g_liverRegion.labels.size() != start_liver_verts.size() ||
        g_liverLR.labels.size()     != start_liver_verts.size()) {
        std::cerr << "[Ctrl+Shift+G] Label-size mismatch: verts="
                  << start_liver_verts.size()
                  << "  region.labels=" << g_liverRegion.labels.size()
                  << "  lr.labels=" << g_liverLR.labels.size()
                  << "; aborting (mesh may have been reloaded "
                     "without re-labeling)." << std::endl;
        return;
    }

    // ----- Phase C: Extract target cloud ----------------------------
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = std::max(0.001f, RegRatios::zThresh());
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[Ctrl+Shift+G] empty target cloud; aborting." << std::endl;
        return;
    }
    const std::vector<glm::vec3>& tgt_points = targetCloud->points;
    stamp("C. extractFrontFacePoints");

    // ----- Phase C.5: RIM-only RMSE diagnostic inputs + "before" -----
    // [NEW V3RS-RIM-DIAG] Mirror of Ctrl+G's Phase C.5. Builds the
    // rim_src mask + rim_tgt point subset used by compute_rim_only_rmse_diag.
    // Held-out diagnostic; never participates in the V3RS cost function.
    //
    // Threshold (g_ctrlgRimTgtThreshPx) is intentionally shared with
    // Ctrl+G so the RIM-only column in PoseLibrary has identical
    // semantics across both methods — a Ctrl+G entry and a Ctrl+Shift+G
    // entry at the same scene-pose are directly comparable.
    //
    // The diagnostic uses LiverRegionLabel::RIM as the source-side rim
    // definition (anatomic), not Ctrl+Shift+G's voxel-space rim used in
    // the cost surface. That keeps the diagnostic method-neutral.
    std::vector<uint8_t>   rim_diag_is_rim_src;
    std::vector<glm::vec3> rim_diag_tgt_rim_points;
    build_rim_only_rmse_inputs(
        start_liver_verts.size(), targetCloud, tgt_points,
        g_ctrlgRimTgtThreshPx,
        rim_diag_is_rim_src, rim_diag_tgt_rim_points);

    // max_dist_sq matches V1 unified-metrics and Ctrl+G F.5b
    // ((g_sceneDiag/7.36)^2 gate) so values are directly comparable.
    constexpr float kRefSceneDiag_rim_diag_shg = 7.36f;
    const float max_dist_rim_diag_shg    =
        g_sceneDiag * (1.0f / kRefSceneDiag_rim_diag_shg);
    const float max_dist_sq_rim_diag_shg =
        max_dist_rim_diag_shg * max_dist_rim_diag_shg;

    int rim_diag_n_src_rim_before = 0;
    int rim_diag_matched_before   = 0;
    const float rmse_rim_before = compute_rim_only_rmse_diag(
        start_liver_verts, rim_diag_is_rim_src, rim_diag_tgt_rim_points,
        max_dist_sq_rim_diag_shg,
        rim_diag_n_src_rim_before, rim_diag_matched_before);
    stamp("C.5. rim_only_rmse_before");

    // ----- Build ParamsV3R ------------------------------------------
    // Inherits ParamsV3 fields from V3 (kept VERBATIM identical to
    // runBipopCmaesV3 above so the determinism contract is preserved
    // at QUAD_ALL). The only V3R-specific additions are the four
    // quadrant fields appended at the end.
    CmaesRefineV3RS::ParamsV3RS p;

    // ----- ParamsV3 (base class) fields, identical to V3 -----------
    p.verbose         = true;
    p.log_every       = 100;
    p.save_debug_jpg  = false;

    p.tx_range        = RegRatios::cmaLocalT();
    p.ty_range        = RegRatios::cmaLocalT();
    p.tz_range        = RegRatios::cmaLocalT();
    // Defaults match V1: rx/ry/rz_range=10.0, scale ∈ [0.90,1.10],
    // min_match_ratio=0.30, penalty_value=9.9.

    p.scene_diag      = g_sceneDiag;
    p.jitter_local_t  = RegRatios::cmaLocalT();
    p.jitter_global_t = RegRatios::cmaGlobalT();

    p.src_voxel_ratio = 0.015f;
    p.tgt_voxel_ratio = 0.015f;

    // ----- ParamsV3R-specific fields ---------------------------------
    p.quadrant_mask        = quadrant_mask;
    p.region_labels        = g_liverRegion.labels;   // copy (~10K bytes)
    p.lr_labels            = g_liverLR.labels;       // copy (~10K bytes)
    p.full_rmse_use_subset = false;  // S4 default (HANDOVER §1, plan E1)
    // voxel_to_orig and subset_idx_voxel are filled by runBipopCmaesV3R
    // at session level (Phase C/D inside the driver).

    // ----- ParamsV3R-W (rim-weighted extension, opt-in) --------------
    // All three switches default to OFF. When all are off, the wrapper
    // passes empty arrays / zero beta to V3R, which falls through to
    // the original evaluate_one_v3 verbatim → byte-identical with V3
    // at QUAD_ALL preserved.
    p.use_arvis_filter      = g_ctrlgUseArVisFilter;
    p.beta_rim_weight       = std::max(0.0f, g_ctrlgBetaRimWeight);
    p.rim_tgt_threshold_px  = g_ctrlgRimTgtThreshPx;
    // R-feat-2: caudal-only filter (anatomical, orthogonal to AR-vis).
    // is_caudal_orig itself is filled below in Phase C2c (only when the
    // toggle is ON and g_liverCC is valid). The combine mode is always
    // passed; it takes effect only if BOTH arvis_voxel and caudal_voxel
    // end up non-empty inside runBipopCmaesV3R.
    p.use_caudal_only       = g_ctrlgUseCaudalOnly;
    p.arvis_caudal_combine  = g_ctrlgArvisCaudalCombine;

    // ----- Phase C2a: AR visibility (raycast + rim CC rescue) -------
    // Two-stage:
    //   (a) BVH raycast from origin -> base visibility for the entire
    //       organ. This is the trustworthy front/back classifier.
    //   (b) Rim-subgraph BFS rescue: rim vertices wrongly tagged
    //       occluded by raycast self-occlusion (the rim band blocks
    //       its own adjacent triangles where curvature is sharp) are
    //       rehydrated. Expansion walks rim-rim edges only, gated by
    //       a per-vertex normal test so the BFS cannot wrap around to
    //       the genuinely back-facing rim.
    //
    // Why this is preferred over the standalone normal test:
    //   - Raycast handles the global front/back partition correctly
    //     (including non-convex / concave pockets), which the normal
    //     test cannot.
    //   - The rescue is local (rim-only, 1+ hop adjacency, normal
    //     gate) so the back rim never propagates in.
    //   - Cost ~16 ms (raycast) + ~1 ms (rescue), once per Ctrl+G.
    if (p.use_arvis_filter) {
        const auto t_arv0 = clk::now();

        // Ensure normals are available (used by the rescue gate).
        if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
            std::cout << "[Ctrl+Shift+G/AR-vis] WARN: normals missing -- "
                      << "computing from faces..." << std::endl;
            Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
        }

        // ---- Stage (a): BVH raycast from origin (base visibility) ----
        Reg3D::BVHTree bvh;
        bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
        const float diag = (g_liverRegion.bbox_diag > 0.0f)
                               ? g_liverRegion.bbox_diag : g_sceneDiag;
        // AR camera nominally at origin, look-at +Z (PoseLibrary
        // AutoProbe convention). We pull the raycast origin back
        // slightly along -Z so that even when a previous Ctrl+G
        // produced a scale-blowup transform (mesh straddles or wraps
        // the true origin), the raycast still has the whole organ in
        // front of it. Direction +Z is unchanged so front/back
        // semantics are preserved. Pull-back is kept small (0.2*diag)
        // because raycastVisibilityBVH's tol is fixed at 1e-3*diag,
        // and a longer ray path means stricter "true" occlusion —
        // too much pull-back legitimately culls back-side vertices.
        const glm::vec3 ar_cam_pos(0.0f, 0.0f, -0.2f * diag);
        LiverRegionLabel::raycastVisibilityBVH(
            *liverMesh3D, bvh, ar_cam_pos, diag, p.arvis_orig);
        int n_vis_raycast = 0;
        for (uint8_t v : p.arvis_orig) if (v) n_vis_raycast++;

        // ---- Stage (b): rim subgraph BFS rescue ----
        // For each rim vertex left occluded by raycast, promote to
        // visible if it is reachable in the rim-only adjacency graph
        // from a currently-visible rim vertex AND its normal still
        // faces the camera (cos(n, -pos) > kGateCos). The cosine gate
        // prevents the BFS from leaking into back-side rim, which
        // along a continuous rim loop has strongly negative cos.
        const size_t N = liverMesh3D->mVertices.size() / 3;
        int n_rescued = 0;
        if (g_liverRegion.labels.size() == N) {
            // Rim<->Rim edges from triangles. mIndices is flat.
            std::vector<std::vector<int>> rim_adj(N);
            const auto& idx = liverMesh3D->mIndices;
            auto isRim = [&](int v) {
                return g_liverRegion.labels[v] == LiverRegionLabel::RIM;
            };
            auto addEdge = [&](int u, int v) {
                rim_adj[u].push_back(v);
                rim_adj[v].push_back(u);
            };
            for (size_t t = 0; t + 2 < idx.size(); t += 3) {
                const int a = (int)idx[t + 0];
                const int b = (int)idx[t + 1];
                const int c = (int)idx[t + 2];
                const bool ra = isRim(a);
                const bool rb = isRim(b);
                const bool rc = isRim(c);
                if (ra && rb) addEdge(a, b);
                if (rb && rc) addEdge(b, c);
                if (rc && ra) addEdge(c, a);
            }

            // Normal gate: cos angle between normal and (cam - pos).
            // Pure front: cos > 0;  exact silhouette: cos == 0;
            // back-facing: cos < 0. We allow up to a small negative
            // value to catch silhouette / slight-back-facing rim that
            // is part of the "front rim band" the user wants to keep.
            constexpr float kGateCos = -0.2f;

            // Seed BFS with already-visible rim vertices.
            std::vector<int> queue;
            queue.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                if (p.arvis_orig[i] && isRim((int)i)) {
                    queue.push_back((int)i);
                }
            }
            size_t qhead = 0;
            while (qhead < queue.size()) {
                const int u = queue[qhead++];
                for (int w : rim_adj[u]) {
                    if (p.arvis_orig[w]) continue;  // already visible
                    const glm::vec3 pw(
                        liverMesh3D->mVertices[3*w + 0],
                        liverMesh3D->mVertices[3*w + 1],
                        liverMesh3D->mVertices[3*w + 2]);
                    const glm::vec3 nw(
                        liverMesh3D->mNormals[3*w + 0],
                        liverMesh3D->mNormals[3*w + 1],
                        liverMesh3D->mNormals[3*w + 2]);
                    // View direction from the (pulled-back) AR camera
                    // to this vertex. Keep gate consistent with the
                    // raycast origin used above.
                    const glm::vec3 view = pw - ar_cam_pos;
                    const float lv = glm::length(view);
                    const float ln = glm::length(nw);
                    if (lv < 1e-6f || ln < 1e-6f) continue;
                    const float cos_nv = glm::dot(nw, -view) / (ln * lv);
                    if (cos_nv > kGateCos) {
                        p.arvis_orig[w] = 1;
                        ++n_rescued;
                        queue.push_back(w);
                    }
                }
            }
        }

        const auto t_arv1 = clk::now();
        int n_vis_total = 0;
        for (uint8_t v : p.arvis_orig) if (v) n_vis_total++;
        std::cout << "[Ctrl+Shift+G/AR-vis] visible=" << n_vis_total
                  << "/" << N
                  << "  (raycast=" << n_vis_raycast
                  << "  +rim_rescued=" << n_rescued << ")"
                  << "  time=" << ms_dur(t_arv1 - t_arv0) << " ms"
                  << "  diag=" << diag
                  << std::endl;
        stamp("C2a. AR visibility (raycast+rim rescue)");
    }

    // ----- Phase C2b: rim-flag arrays --------------------------------
    // Populated when EITHER:
    //   (a) beta_rim_weight > 0 (V3R-W rim weighting in RMSE), or
    //   (b) silhouette will be active in this session
    //       (g_ctrlgsLambdaSil > 0 AND boundary dist map valid).
    // Silhouette eval iterates over rim ∩ subset voxels for the
    // anchor term, which needs is_rim_orig populated upstream and
    // is_rim_src_voxel built downstream in CmaesRefineV3RS.h. The
    // previous "if (beta>0)" gate caused a silent bypass when the
    // user pressed Ctrl+Shift+G with beta=0 + lambda_sil>0: rim
    // arrays stayed empty, rim_subset was empty, sil_active gated
    // false, and Ctrl+Shift+G effectively did the same thing as
    // Ctrl+G (silhouette contributed nothing).
    //
    // Note: p.lambda_sil is set later (Phase C2c.5), so we check
    // the global g_ctrlgsLambdaSil directly here to determine intent.
    const bool sil_will_be_active_for_rim =
        (g_ctrlgsLambdaSil > 0.0f) && g_boundaryDistMap.valid;

    if (p.beta_rim_weight > 0.0f || sil_will_be_active_for_rim) {
        // Source: per-original-vertex RIM flag from LiverRegionLabel.
        p.is_rim_orig.assign(g_liverRegion.labels.size(), 0);
        for (size_t i = 0; i < g_liverRegion.labels.size(); i++) {
            if (g_liverRegion.labels[i] == LiverRegionLabel::RIM) {
                p.is_rim_orig[i] = 1;
            }
        }
        // Target: copy boundaryDist from the cached cloud parallel to
        // tgt_points. May be empty if the target was extracted via the
        // legacy grid path; in that case the V3R driver will see empty
        // tgt_boundary_dist_full and skip rim weighting silently.
        if (targetCloud->hasBoundaryDist() &&
            targetCloud->boundaryDist.size() == tgt_points.size())
        {
            p.tgt_boundary_dist_full = targetCloud->boundaryDist;
        } else {
            std::cout << "[Ctrl+Shift+G/rim] target cloud has no boundaryDist "
                         "(legacy path?). Rim weighting will be inactive."
                      << std::endl;
            p.tgt_boundary_dist_full.clear();
        }
        int n_rim_src = 0;
        for (uint8_t v : p.is_rim_orig) if (v) n_rim_src++;
        const char* why =
            (p.beta_rim_weight > 0.0f && sil_will_be_active_for_rim) ? "beta+sil" :
            (p.beta_rim_weight > 0.0f)                                ? "beta"     :
                                                                        "sil";
        std::cout << "[Ctrl+Shift+G/rim] populated (" << why << "): beta="
                  << p.beta_rim_weight
                  << "  src_rim=" << n_rim_src
                  << "/" << p.is_rim_orig.size()
                  << "  tgt_bdist_avail="
                  << (p.tgt_boundary_dist_full.empty() ? "NO" : "YES")
                  << "  thresh=" << p.rim_tgt_threshold_px << "px"
                  << std::endl;
    }

    // ----- Phase C2c: caudal-only flag (R-feat-2; anatomical filter) -
    // Source-side filter that complements AR-vis on a different axis:
    //   AR-vis = view-based (raycast from AR camera; depends on current
    //            mesh pose / scale, needs camera tuning).
    //   Caudal = anatomy-based (LiverCranioCaudalLabel; bound to vertex
    //            index, transform-invariant, zero compute cost here).
    //
    // The two filters compose inside V3R driver via the AND/OR combine
    // mode (p.arvis_caudal_combine). When only one is ON, that one
    // alone applies. When neither is ON, the V3 byte-identical contract
    // is preserved as before.
    //
    // Degradation policy: if user requested caudal-only but g_liverCC
    // is not yet computed (Shift+H or Apply Init Pose not pressed), we
    // disable the filter for THIS run (warn, then continue). Aborting
    // would be hostile because the rest of Ctrl+G (quadrant, arvis,
    // beta) is unaffected.
    if (p.use_caudal_only) {
        if (!g_liverCC.valid()) {
            std::cerr << "[Ctrl+Shift+G/caudal] Only-Caudal requested but "
                         "g_liverCC not yet computed. Press Shift+H "
                         "(or Apply Init Pose) to populate. "
                         "Disabling caudal filter for this run."
                      << std::endl;
            p.use_caudal_only = false;
            p.is_caudal_orig.clear();
        } else if (g_liverCC.labels.size() != start_liver_verts.size()) {
            std::cerr << "[Ctrl+Shift+G/caudal] g_liverCC.labels size mismatch "
                      << "(cc=" << g_liverCC.labels.size()
                      << " vs verts=" << start_liver_verts.size()
                      << "). Disabling caudal filter for this run; "
                         "re-run Shift+H after the mesh change."
                      << std::endl;
            p.use_caudal_only = false;
            p.is_caudal_orig.clear();
        } else {
            // Build per-original-vertex 0/1 mask. The voxel-side
            // mapping (is_caudal_voxel) is derived later inside
            // runBipopCmaesV3R via derive_is_caudal_voxel, parallel
            // to how arvis_orig -> arvis_voxel is built.
            p.is_caudal_orig.assign(g_liverCC.labels.size(), 0);
            for (size_t i = 0; i < g_liverCC.labels.size(); i++) {
                if (g_liverCC.labels[i] == LiverCranioCaudalLabel::CAUDAL) {
                    p.is_caudal_orig[i] = 1;
                }
            }
            int n_caudal = 0;
            for (uint8_t v : p.is_caudal_orig) if (v) n_caudal++;
            std::cout << "[Ctrl+Shift+G/caudal] caudal_only=ON"
                      << "  n_caudal=" << n_caudal
                      << "/" << p.is_caudal_orig.size()
                      << "  cc_confidence="
                      << std::fixed << std::setprecision(3)
                      << g_liverCC.cc.confidence
                      << std::defaultfloat
                      << (g_liverCC.cc.weak ? "  [WEAK]" : "  [OK]")
                      << "  flipped_manual="
                      << (g_liverCC.cc.flipped_manual ? "true" : "false")
                      << "  combine="
                      << (p.arvis_caudal_combine == 0 ? "AND" : "OR")
                      << std::endl;
            stamp("C2c. caudal-only flag (mesh-intrinsic CC)");
        }
    } else {
        p.is_caudal_orig.clear();
    }

    // ----- Phase C2c.5 [V3RS]: silhouette loss setup (scale anchor) --
    // The Ctrl+Shift+G UI exposes a slider g_ctrlgsLambdaSil (and a
    // master checkbox is not used here: pressing Ctrl+Shift+G means
    // the user asked for silhouette. If lambda is 0 we still run but
    // print a notice -- equivalent to V3R-W).
    //
    // The 2D distance map and AR-camera intrinsics are pulled from
    // g_boundaryDistMap and OrbitCam respectively. If the SAM2 mask
    // hasn't been built (g_boundaryDistMap.valid == false) we disable
    // silhouette for this run rather than failing -- V3R-W result is
    // still useful.
    p.lambda_sil = std::max(0.0f, g_ctrlgsLambdaSil);
    // [V3I] Pure-IoU mode pass-through. When ON, the engine ignores
    //   RMSE in the cost; lambda_sil is still used only to gate IoU
    //   computation, so force it positive if the user left it at 0.
    p.pure_iou_mode = g_ctrlgsPureIoUMode;
    if (p.pure_iou_mode && p.lambda_sil <= 0.0f) p.lambda_sil = 1.0f;
    if (p.lambda_sil <= 0.0f) {
        std::cout << "[Ctrl+Shift+G/sil] lambda_sil=0; running V3RS as V3R-W "
                     "(use Ctrl+G for that; this is a courtesy fallback)."
                  << std::endl;
        p.sil_dist_map_2d.clear();
    } else if (!g_boundaryDistMap.valid ||
               g_boundaryDistMap.width  <= 1 ||
               g_boundaryDistMap.height <= 1) {
        std::cerr << "[Ctrl+Shift+G/sil] lambda_sil=" << p.lambda_sil
                  << " requested but g_boundaryDistMap is not valid "
                     "(SAM2 mask missing?). Disabling silhouette for "
                     "this run." << std::endl;
        p.lambda_sil = 0.0f;
        p.sil_dist_map_2d.clear();
    } else if (g_boundaryDistMap.data.size() !=
               (size_t)g_boundaryDistMap.width *
               (size_t)g_boundaryDistMap.height) {
        std::cerr << "[Ctrl+Shift+G/sil] g_boundaryDistMap size mismatch "
                     "(data=" << g_boundaryDistMap.data.size()
                  << " vs " << g_boundaryDistMap.width << "x"
                  << g_boundaryDistMap.height
                  << "). Disabling silhouette for this run."
                  << std::endl;
        p.lambda_sil = 0.0f;
        p.sil_dist_map_2d.clear();
    } else {
        p.sil_fx     = OrbitCam.fx;
        p.sil_fy     = OrbitCam.fy;
        p.sil_cx     = OrbitCam.cx;
        p.sil_cy     = OrbitCam.cy;
        p.sil_img_w  = g_boundaryDistMap.width;
        p.sil_img_h  = g_boundaryDistMap.height;
        p.sil_dist_map_2d = g_boundaryDistMap.data;   // value copy

        // ----- Instrument occlusion fields (NEW, opt-in) ------------
        // Populated only when the master toggle g_ctrlgsIgnoreInstrument
        // is ON. When OFF, sil_instrument_dist_map_2d stays empty
        // (ParamsV3RS default) and the rasterizer skips the filter
        // entirely -- byte-identical to the pre-feature path.
        //
        // Degradation policy: if the toggle is ON but the instrument
        // mask can't be loaded (file missing) or size-mismatches the
        // liver mask, we log a warning and disable the filter for
        // this session. The Ctrl+Shift+G run still proceeds, just
        // without occlusion filtering. The toggle stays ON so the
        // user can re-try after fixing the file.
        if (g_ctrlgsIgnoreInstrument) {
            const bool inst_loaded = ensureInstrumentDistMap();
            const bool size_ok =
                inst_loaded
                && g_instrumentDistMap.valid
                && g_instrumentDistMap.width  == g_boundaryDistMap.width
                && g_instrumentDistMap.height == g_boundaryDistMap.height
                && g_instrumentDistMap.data.size() ==
                       (size_t)g_instrumentDistMap.width *
                       (size_t)g_instrumentDistMap.height;
            if (size_ok) {
                p.sil_instrument_dist_map_2d = g_instrumentDistMap.data;
                p.sil_instrument_thresh_px   = std::max(0.0f,
                                                  g_ctrlgsInstrumentThreshPx);
                int n_excluded = 0;
                for (float d : p.sil_instrument_dist_map_2d) {
                    if (d < p.sil_instrument_thresh_px) ++n_excluded;
                }
                const double pct =
                    100.0 * (double)n_excluded /
                    (double)p.sil_instrument_dist_map_2d.size();
                std::cout << std::fixed << std::setprecision(3)
                          << "[Ctrl+Shift+G/occ] instrument filter ON"
                          << "  thresh=" << p.sil_instrument_thresh_px
                          << "px"
                          << "  excluded_px=" << n_excluded
                          << " (" << pct << "% of image)"
                          << std::defaultfloat << std::setprecision(6)
                          << std::endl;
            } else {
                std::cerr << "[Ctrl+Shift+G/occ] toggle ON but instrument "
                             "mask unavailable (file missing or size "
                             "mismatch). Filter DISABLED for this session."
                          << std::endl;
                p.sil_instrument_dist_map_2d.clear();
                p.sil_instrument_thresh_px = 0.0f;
            }
        } else {
            // Toggle OFF: leave the field empty, filter is disabled.
            p.sil_instrument_dist_map_2d.clear();
            p.sil_instrument_thresh_px = 0.0f;
        }

        // [V3RS Phase 1] Full-mesh triangle indices + pre-built AR
        // silhouette view/proj. These let runBipopCmaesV3RS rasterize
        // the full mesh in the CMA-ES inner loop (path B3) without
        // touching any globals. liverMesh3D->mIndices is GLuint, which
        // is uint32_t.
        //
        // [V3RS Phase 2c] Quadrant filter for silhouette.
        //   The RMSE_W path is already filtered by quadrant_mask (via
        //   subset_idx_voxel). For the silhouette to be semantically
        //   matched, we filter triangles by the same quadrant: a
        //   triangle is kept iff ALL 3 of its vertices belong to the
        //   active quadrant per makeQuadrantSubsetIdx. Triangles
        //   straddling the boundary are dropped. Backface culling
        //   inside the rasterizer further removes posterior parts
        //   that face away from the camera.
        //   When the quadrant is QUAD_ALL, the filter degenerates to
        //   "all triangles" -- byte-identical to the unfiltered path.
        {
            const std::vector<int> quad_vert_ids =
                LiverLeftRightLabel::makeQuadrantSubsetIdx(
                    g_liverRegion.labels,
                    g_liverLR.labels,
                    quadrant_mask);
            const size_t nV = liverMesh3D->mVertices.size() / 3;
            std::vector<uint8_t> in_quad(nV, 0);
            for (int idx : quad_vert_ids) {
                if (idx >= 0 && (size_t)idx < nV) in_quad[(size_t)idx] = 1;
            }

            const auto& src_idx = liverMesh3D->mIndices;
            p.sil_indices.clear();
            p.sil_indices.reserve(src_idx.size());
            int n_tris_total = 0;
            int n_tris_kept  = 0;
            for (size_t t = 0; t + 2 < src_idx.size(); t += 3) {
                ++n_tris_total;
                const uint32_t i0 = src_idx[t + 0];
                const uint32_t i1 = src_idx[t + 1];
                const uint32_t i2 = src_idx[t + 2];
                if (i0 >= nV || i1 >= nV || i2 >= nV) continue;
                if (in_quad[i0] && in_quad[i1] && in_quad[i2]) {
                    p.sil_indices.push_back(i0);
                    p.sil_indices.push_back(i1);
                    p.sil_indices.push_back(i2);
                    ++n_tris_kept;
                }
            }
            std::cout << "[Ctrl+Shift+G/sil] quad-filter tris: "
                      << n_tris_kept << " / " << n_tris_total
                      << "  (Q=" << LiverLeftRightLabel::quadrantMaskString(
                                        quadrant_mask)
                      << ")" << std::endl;
        }

        p.sil_view = buildSilhouetteView();
        p.sil_proj = buildSilhouetteProj();
        // p.sil_eval_interval / p.sil_raster_step keep their defaults
        // (1 and 8) from ParamsV3RS unless the UI overrides them
        // later. Both can be edited in the next code-cycle.

        // ----- Target-mask cache (source-parity raster) -------------
        // Build / hit the app-wide squashed-and-haloed target mask so
        // that rasterize_iou2d_v3rs's Step 3 uses the cache path
        // instead of legacy centre-sample. The cache is content-
        // fingerprinted against the dist_map data; same SAM2 mask =
        // same fingerprint = cache hit (no rebuild) even across
        // multiple Ctrl+Shift+G presses. SAM2 re-load -> content
        // changes -> fingerprint mismatch -> automatic rebuild.
        //
        // Toggle g_silTargetSquashEnabled (default ON) gates whether
        // rasterize_iou2d_v3rs actually USES the cache. We still build
        // it here either way so flipping the toggle later is instant
        // (no rebuild). When the toggle is OFF, the rasterizer falls
        // back to the legacy asymmetric centre-sample path.
        //
        // Cost on miss: ~5 ms one-time per SAM2 mask change. Cost on
        // hit: ~5 reads + 5 compares (~10 ns). Negligible.
        CmaesRefineV3RS::ensureSilTargetMaskCache(
            p.sil_dist_map_2d,
            p.sil_img_w, p.sil_img_h,
            p.sil_raster_step);

        // ----- [NEW RA-2] Asymmetric / rim-sil pass-through -----------
        // Globals are read at wrapper time so subsequent edits in the
        // ImGui panel take effect on the NEXT Ctrl+Shift+G press.
        // When the corresponding toggle is OFF we explicitly zero the
        // weight so the V3RS inner loop falls through to the Phase 3
        // path (no per-eval overhead for the disabled term).
        p.lambda_out     = g_ctrlgsUseOutsideRatio ? g_ctrlgsLambdaOut    : 0.0f;
        p.lambda_rim_sil = g_ctrlgsUseRimSil      ? g_ctrlgsLambdaRimSil : 0.0f;
        p.rim_sil_max_px = std::max(1.0f, g_ctrlgsRimSilMaxPx);

        // ----- [NEW RA-RIM-ANAT] Populate is_rim_anatomic_full --------
        // Same filter the Ctrl+G "Show RIM pairs" checkbox uses (see
        // existing g_ctrlgRimSrcVertIdx block ~70 lines below). We
        // produce a per-vertex uint8_t flag here (rather than an index
        // list) because rasterize_iou2d_v3rs's anatomic-mode loop is
        // indexed by positions[i] (the FULL mesh vertex i), so a flag
        // array of the same size is the natural form.
        //
        // Only populate when BOTH rim_sil is enabled AND anatomic mode
        // is selected -- otherwise the empty vector triggers the legacy
        // raster-boundary fallback in V3RS.
        p.is_rim_anatomic_full.clear();
        const bool want_anatomic = g_ctrlgsUseRimSil && g_ctrlgsRimSilAnatomic;
        if (want_anatomic
            && g_liverRegion.valid()
            && !g_liverRegion.labels.empty())
        {
            const size_t nV = g_liverRegion.labels.size();
            p.is_rim_anatomic_full.assign(nV, 0);

            // Match the RimViz filter exactly: quadrant ∩ RIM-label ∩
            // (post-degradation) arvis ∩ caudal, combined via cmode.
            // p.use_*_only/arvis flags were set earlier in this wrapper;
            // we read them (not the raw UI globals) so the resolved
            // post-degradation state is what governs the filter.
            const bool a_on = p.use_arvis_filter && !p.arvis_orig.empty();
            const bool c_on = p.use_caudal_only  && !p.is_caudal_orig.empty();
            const uint8_t cmode = p.arvis_caudal_combine;
            const auto quadAllowed = LiverLeftRightLabel::makeQuadrantSubsetIdx(
                g_liverRegion.labels, g_liverLR.labels, quadrant_mask);

            int n_kept = 0;
            for (int idx : quadAllowed) {
                if (idx < 0 || (size_t)idx >= nV) continue;
                if (g_liverRegion.labels[idx] != LiverRegionLabel::RIM) continue;
                const bool a = (!a_on) ||
                               ((size_t)idx < p.arvis_orig.size() &&
                                p.arvis_orig[idx]);
                const bool c = (!c_on) ||
                               ((size_t)idx < p.is_caudal_orig.size() &&
                                p.is_caudal_orig[idx]);
                bool pass;
                if (a_on && c_on) {
                    pass = (cmode == 0) ? (a && c) : (a || c);
                } else {
                    pass = a && c;   // unconditional-true via shortcut
                }
                if (pass) {
                    p.is_rim_anatomic_full[(size_t)idx] = 1;
                    ++n_kept;
                }
            }
            std::cout << "[Ctrl+Shift+G/rim-anat] anatomic_mode=ON  "
                         "rim_vertices=" << n_kept
                      << "/" << nV
                      << "  (quadrant + RIM-label"
                      << (a_on ? " + AR-vis" : "")
                      << (c_on ? " + Caudal" : "")
                      << ((a_on && c_on)
                          ? (cmode == 0 ? "/AND" : "/OR")
                          : "")
                      << ")"
                      << std::endl;
        } else if (g_ctrlgsRimSilAnatomic && !g_ctrlgsUseRimSil) {
            std::cout << "[Ctrl+Shift+G/rim-anat] anatomic toggle ON but "
                         "rim_sil_penalty toggle OFF -- anatomic mode "
                         "ignored." << std::endl;
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "[Ctrl+Shift+G/sil] ON  lambda_sil=" << p.lambda_sil
                  << "  lambda_out=" << p.lambda_out
                  << "  lambda_rim_sil=" << p.lambda_rim_sil
                  << "  rim_sil_max_px=" << p.rim_sil_max_px
                  << "  rim_anatomic="
                  << (p.is_rim_anatomic_full.empty() ? "OFF" : "ON")
                  << "  intrinsics=(" << p.sil_fx << ","
                                       << p.sil_fy << ","
                                       << p.sil_cx << ","
                                       << p.sil_cy << ")"
                  << "  img=" << p.sil_img_w << "x" << p.sil_img_h
                  << "  tris=" << (p.sil_indices.size() / 3)
                  << "  eval_interval=" << p.sil_eval_interval
                  << "  raster_step=" << p.sil_raster_step
                  << "  target_squash="
                  << (CmaesRefineV3RS::g_silTargetSquashEnabled ? "ON" : "OFF")
                  << std::defaultfloat << std::setprecision(6)
                  << std::endl;
        stamp("C2c.5. silhouette intrinsics + dist map");
    }

    // ----- Populate RIM-pair visualization buffers (opt-in) ---------
    // Independent from the optimization itself: even with beta=0 the
    // user may want to see what the rim sets look like. Buffers are
    // (re)populated at every Ctrl+G entry so they reflect the current
    // mesh / target / threshold.
    g_ctrlgRimSrcVertIdx.clear();
    g_ctrlgRimTgtPos.clear();
    g_ctrlgRimVizAvailable = false;
    if (g_ctrlgShowRimPairs) {
        // Source: vertices that pass (a) the user-selected quadrant
        // mask, (b) RIM region label, and (c) the SAME subset filter
        // that CMA-ES sees -- i.e. arvis and/or caudal combined via
        // p.arvis_caudal_combine. Without mirroring this, the rim
        // spheres would advertise vertices that the optimiser never
        // actually uses, which is confusing.
        //
        // p.use_caudal_only may have been disabled above (degradation
        // path: g_liverCC missing / size mismatch); reflect the
        // post-degradation state, not the raw UI toggle, so what we
        // draw is exactly what CMA-ES uses.
        const bool a_on = p.use_arvis_filter && !p.arvis_orig.empty();
        const bool c_on = p.use_caudal_only  && !p.is_caudal_orig.empty();
        const uint8_t cmode = p.arvis_caudal_combine;  // 0=AND, 1=OR
        std::vector<int> quadAllowed = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels,
            g_liverLR.labels,
            quadrant_mask);
        for (int idx : quadAllowed) {
            if (idx < 0 ||
                (size_t)idx >= g_liverRegion.labels.size()) continue;
            if (g_liverRegion.labels[idx] != LiverRegionLabel::RIM) continue;
            // Empty array = "all pass" for that filter (matches V3R
            // filter_by_quadrant_with_arvis_caudal semantics).
            const bool a = (!a_on) ||
                           ((size_t)idx < p.arvis_orig.size() &&
                            p.arvis_orig[idx]);
            const bool c = (!c_on) ||
                           ((size_t)idx < p.is_caudal_orig.size() &&
                            p.is_caudal_orig[idx]);
            bool pass;
            if (a_on && c_on) {
                pass = (cmode == 0) ? (a && c) : (a || c);
            } else {
                pass = a && c;  // one is unconditionally true via shortcut
            }
            if (!pass) continue;
            g_ctrlgRimSrcVertIdx.push_back(idx);
        }
        // Target: points with boundaryDist below threshold, also
        // exclude instrument-occluded points consistent with Shift+P.
        if (targetCloud->hasBoundaryDist()) {
            const bool useInst = targetCloud->hasInstrumentDist() &&
                                 targetCloud->instrumentDist.size() == tgt_points.size();
            for (size_t i = 0; i < tgt_points.size(); i++) {
                if (targetCloud->boundaryDist[i] >= g_ctrlgRimTgtThreshPx) continue;
                if (useInst &&
                    targetCloud->instrumentDist[i] < g_instrumentPxThresh)
                {
                    continue;
                }
                g_ctrlgRimTgtPos.push_back(tgt_points[i]);
            }
        }
        g_ctrlgRimVizAvailable = !g_ctrlgRimSrcVertIdx.empty() ||
                                 !g_ctrlgRimTgtPos.empty();
        std::cout << "[Ctrl+Shift+G/RimViz] src=" << g_ctrlgRimSrcVertIdx.size()
                  << "  tgt=" << g_ctrlgRimTgtPos.size()
                  << "  quad=" << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
                  << "  (AR-vis filter "
                  << (a_on ? "ON" : "OFF")
                  << ", Caudal "
                  << (c_on ? "ON" : "OFF")
                  << ", combine "
                  << ((a_on && c_on) ? (cmode == 0 ? "AND" : "OR") : "n/a")
                  << ")"
                  << std::endl;
    }

    // ----- Phase D: Call into the pure V3R driver -------------------
    // Internally voxelizes src+tgt once (same as V3), then computes
    // voxel_to_orig (NN reverse map, ~1 ms) and subset_idx_voxel
    // (label filter, <1 ms), then runs 10 BIPOP CMA-ES restarts with
    // a subset KDTree per Run.

    // ----- Phase 0 Run-selector callback installation ----------------
    // When the silhouette path is active and a SAM2 mask is available,
    // install a callback that V3RS will replay for each of the 10 Run
    // candidates after its Run loop. The callback evaluates 2D IoU of
    // the candidate pose against the SAM2 mask using the AR fixed
    // camera, and is the cheap "is IoU a useful signal?" diagnostic
    // for Phase 0. Selection is NOT changed -- V3RS still picks the
    // best Run by argmin(rmse_full). See
    // HANDOVER_V3RS_silhouette_pivot_to_2D_IoU.md §5.1 / §9.3.
    //
    // Implementation note: computeSilhouette2DObjectiveFast reads
    // liverMesh3D->mVertices directly, so the callback applies the
    // candidate world-matrix to a one-time backup written into
    // liverMesh3D's flat array, evaluates, and leaves liverMesh3D in
    // the just-evaluated state. After runBipopCmaesV3RS returns, we
    // restore from backup so Phase E sees the original pre-Phase-D
    // vertices (Phase E applies its own M_world).
    std::vector<float> liver_verts_backup;
    if (p.lambda_sil > 0.0f
        && g_boundaryDistMap.valid
        && g_boundaryDistMap.width  > 0
        && g_boundaryDistMap.height > 0
        && liverMesh3D
        && !liverMesh3D->mVertices.empty())
    {
        liver_verts_backup = liverMesh3D->mVertices;   // ~120 KB, one-time

        const glm::mat4 sel_view = buildSilhouetteView();
        const glm::mat4 sel_proj = buildSilhouetteProj();
        const int sel_w = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
        const int sel_h = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

        // [Phase A] Callback now evaluates IoU_occluded via
        // rasterize_iou2d_v3rs (the same routine Layer 1 calls per-eval
        // inside CMA-ES). This unifies Layer 1 / Layer 2 on the same
        // IoU definition; the wrapper-level Layer 3 gate (Phase E
        // below) does the same call so all three internal layers see
        // an identical c(x) = RMSE + lambda_sil*(1 - IoU_occluded).
        //
        // Capture strategy:
        //   sel_view / sel_proj / sel_w / sel_h : value (small mat4/int)
        //   liver_verts_backup                  : reference (the only
        //       writable surface -- this is the pre-Phase-D snapshot
        //       that gets restored after the driver returns)
        //   &p                                  : reference (read-only;
        //       captures sil_indices, sil_dist_map_2d,
        //       sil_instrument_dist_map_2d, sil_instrument_thresh_px,
        //       and the raster geometry fields without value-copying
        //       the large vectors)
        p.sil_iou2d_eval_fn =
            [sel_view, sel_proj, sel_w, sel_h,
             &liver_verts_backup, &p]
            (const glm::mat4& M_world) -> float
            {
                if (!liverMesh3D) return 0.0f;
                auto& V = liverMesh3D->mVertices;
                const size_t N = liver_verts_backup.size();
                if (V.size() != N) return 0.0f;

                // Apply M_world to the backup -> liverMesh3D->mVertices.
                // Identity M_world means "no transform" -> evaluate the
                // pre-Phase-D pose (used by the Phase A selector to
                // compute baseline_iou_occluded).
                std::vector<glm::vec3> positions;
                const size_t nV = N / 3;
                positions.resize(nV);
                for (size_t i = 0; i < nV; ++i) {
                    glm::vec4 p4(liver_verts_backup[i*3 + 0],
                                 liver_verts_backup[i*3 + 1],
                                 liver_verts_backup[i*3 + 2], 1.0f);
                    glm::vec4 tp = M_world * p4;
                    V[i*3 + 0] = tp.x;
                    V[i*3 + 1] = tp.y;
                    V[i*3 + 2] = tp.z;
                    positions[i] = glm::vec3(tp.x, tp.y, tp.z);
                }

                // Build the same MVP the per-eval rasterizer uses.
                const glm::mat4 mvp = sel_proj * sel_view;

                // Forward the params-side instrument fields. Empty
                // vector -> no occlusion filter (byte-identical to the
                // pre-feature path) -- this preserves degradation
                // semantics when the master toggle is OFF.
                const std::vector<float>* inst_ptr =
                    p.sil_instrument_dist_map_2d.empty()
                        ? nullptr
                        : &p.sil_instrument_dist_map_2d;

                return CmaesRefineV3RS::rasterize_iou2d_v3rs(
                    positions, p.sil_indices, mvp,
                    p.sil_dist_map_2d,
                    p.sil_img_w, p.sil_img_h, p.sil_raster_step,
                    /*out_hitmap     */ nullptr,
                    /*out_target_mask*/ nullptr,
                    /*out_gw         */ nullptr,
                    /*out_gh         */ nullptr,
                    /*out_step1_us   */ nullptr,
                    /*out_step2_us   */ nullptr,
                    /*out_step3_us   */ nullptr,
                    /*raster_mode    */ 0,
                    inst_ptr,
                    p.sil_instrument_thresh_px);
            };

        // [NEW V3RS-SEL] Extended Selector callback. Returns IoU,
        // outside_ratio and rim_sil_loss together so the Phase A
        // Selector can compare Runs on the SAME composite Phase E
        // gates on. Replicates the projection step from
        // sil_iou2d_eval_fn (above) and just asks rasterize_iou2d_v3rs
        // for the extra scalars. Only the toggled-ON penalty pointers
        // are passed in so when a feature is OFF the rasterizer skips
        // the extra work entirely. The Selector prefers this callback
        // over sil_iou2d_eval_fn; when this is null the legacy IoU-only
        // path is used (byte-identical to pre-feature behaviour).
        p.sil_metrics_eval_fn =
            [sel_view, sel_proj, sel_w, sel_h,
             &liver_verts_backup, &p]
            (const glm::mat4& M_world)
                -> CmaesRefineV3RS::ParamsV3RS::SelectorMetrics
            {
                CmaesRefineV3RS::ParamsV3RS::SelectorMetrics m;
                if (!liverMesh3D) return m;
                auto& V = liverMesh3D->mVertices;
                const size_t N = liver_verts_backup.size();
                if (V.size() != N) return m;

                std::vector<glm::vec3> positions;
                const size_t nV = N / 3;
                positions.resize(nV);
                for (size_t i = 0; i < nV; ++i) {
                    glm::vec4 p4(liver_verts_backup[i*3 + 0],
                                 liver_verts_backup[i*3 + 1],
                                 liver_verts_backup[i*3 + 2], 1.0f);
                    glm::vec4 tp = M_world * p4;
                    V[i*3 + 0] = tp.x;
                    V[i*3 + 1] = tp.y;
                    V[i*3 + 2] = tp.z;
                    positions[i] = glm::vec3(tp.x, tp.y, tp.z);
                }

                const glm::mat4 mvp = sel_proj * sel_view;

                const std::vector<float>* inst_ptr =
                    p.sil_instrument_dist_map_2d.empty()
                        ? nullptr
                        : &p.sil_instrument_dist_map_2d;

                // Toggle-gated outputs: only request what the user has
                // enabled. This matches Phase E (see
                // RegistrationActions.h Phase E body) so Selector and
                // Phase E ask for exactly the same scalars in exactly
                // the same configurations.
                float outside_ratio = 0.0f;
                float rim_sil_loss  = 0.0f;
                const bool want_out = (p.lambda_out      > 0.0f);
                const bool want_rs  = (p.lambda_rim_sil  > 0.0f);

                m.iou_occluded = CmaesRefineV3RS::rasterize_iou2d_v3rs(
                    positions, p.sil_indices, mvp,
                    p.sil_dist_map_2d,
                    p.sil_img_w, p.sil_img_h, p.sil_raster_step,
                    /*out_hitmap     */ nullptr,
                    /*out_target_mask*/ nullptr,
                    /*out_gw         */ nullptr,
                    /*out_gh         */ nullptr,
                    /*out_step1_us   */ nullptr,
                    /*out_step2_us   */ nullptr,
                    /*out_step3_us   */ nullptr,
                    /*raster_mode    */ 0,
                    inst_ptr,
                    p.sil_instrument_thresh_px,
                    /*out_outside_ratio*/
                        want_out ? &outside_ratio : nullptr,
                    /*out_rim_sil_loss */
                        want_rs  ? &rim_sil_loss  : nullptr,
                    p.rim_sil_max_px,
                    // Anatomic-mode RIM source: pass only when both the
                    // rim_sil toggle is ON AND the wrapper has populated
                    // is_rim_anatomic_full. Null/empty -> legacy raster-
                    // boundary mode (byte-identical to pre-anat path).
                    (want_rs && !p.is_rim_anatomic_full.empty())
                        ? &p.is_rim_anatomic_full
                        : nullptr,
                    /*out_rim_cell_mask*/ nullptr);
                m.outside_ratio = outside_ratio;
                m.rim_sil_loss  = rim_sil_loss;
                return m;
            };

        // [V3RS Phase 2 diagnostic] Per-Run capture callback. Called
        // RIGHT AFTER sil_iou2d_eval_fn for the same Run; liverMesh3D
        // is in the just-evaluated state so the capture reads it
        // directly (no re-apply). The callback ALWAYS uploads to the
        // SilOverlay ImGui texture (cheap: ~5 ms / Run including GL).
        // It ALSO writes PNGs via IoUDebug::dump iff
        // g_silDumpPerRunEnabled is true (~50 ms / Run PNG IO).
        //
        // Reset SilOverlay slots at session start so stale data from a
        // previous Ctrl+Shift+G doesn't linger in the UI.
        SilOverlay::reset(SilOverlay::g_silOverlay);

        std::string session_dir;   // populated only if dump enabled
        if (g_silDumpPerRunEnabled) {
            ++g_silDumpSessionCounter;
            std::ostringstream session_dir_oss;
            session_dir_oss << DEPTH_OUTPUT_PATH << "v3rs_dump/session_"
                            << std::setw(3) << std::setfill('0')
                            << g_silDumpSessionCounter;
            session_dir = session_dir_oss.str();
            std::error_code ec;
            std::filesystem::create_directories(session_dir, ec);
            if (ec) {
                std::cerr << "[V3RS/Dump] failed to create "
                          << session_dir << ": " << ec.message()
                          << "; PNG dump disabled for this session"
                          << std::endl;
                session_dir.clear();   // disables IoUDebug::dump branch
            } else {
                std::cout << "[V3RS/Dump] PNG dump enabled -> "
                          << session_dir << std::endl;
            }
        }

        const glm::mat4 cap_view = sel_view;
        const glm::mat4 cap_proj = sel_proj;
        const int       cap_w    = sel_w;
        const int       cap_h    = sel_h;
        // p.sil_indices contains the QUADRANT-FILTERED triangle list
        // that the optimizer is using -- capture the viz against the
        // same triangle set so what the user sees matches the cost.
        const std::vector<uint32_t> cap_indices = p.sil_indices;
        const std::vector<float>    cap_dist    = p.sil_dist_map_2d;

        // Capture the instrument occlusion fields too so the F9 viz
        // shows the SAME IoU and excluded cells the CMA-ES cost saw.
        // Both are value-copied into the closure; empty vector means
        // filter OFF (default ParamsV3RS default) and the capture path
        // is byte-identical to the pre-feature behaviour.
        const std::vector<float>    cap_inst_dist  = p.sil_instrument_dist_map_2d;
        const float                 cap_inst_thresh = p.sil_instrument_thresh_px;
        // [NEW V3RS-VIZ] Capture rim_sil_max_px so the F9 6-panel layout
        // is built with the same normalisation cap the user is testing.
        // Always >0 here (clamped to >=1 at wrapper setup), so passing
        // unconditionally always enables the bottom row in the viz --
        // useful even when lambda_rim_sil is 0 because the panels then
        // SHOW the user what rim_sil WOULD penalise if turned on.
        const float                 cap_rim_sil_max = p.rim_sil_max_px;
        // [NEW V3RS-RIM-ANAT] Value-copy the anatomic-mode flag array so
        // the viz uses the SAME RIM definition as the cost. When the
        // anatomic toggle is OFF this vector is empty, and the viz falls
        // back to raster-boundary mode automatically.
        const std::vector<uint8_t>  cap_rim_anatomic = p.is_rim_anatomic_full;

        p.sil_per_run_dump_fn =
            [cap_view, cap_proj, cap_w, cap_h, cap_indices,
             cap_dist, cap_inst_dist, cap_inst_thresh, cap_rim_sil_max,
             cap_rim_anatomic, session_dir]
            (int run_idx, float scale_value)
            {
                if (!liverMesh3D) return;

                // (1) Always: upload to SilOverlay GL texture slot.
                //     Pass instrument-occlusion args so the F9 IoU
                //     and composite match what CMA-ES evaluated.
                SilOverlay::capture(
                    SilOverlay::g_silOverlay, run_idx, liverMesh3D,
                    cap_indices, cap_view, cap_proj, cap_dist,
                    cap_w, cap_h, /*step=*/8, scale_value,
                    cap_inst_dist.empty() ? nullptr : &cap_inst_dist,
                    cap_inst_thresh,
                    // [NEW V3RS-VIZ] enable bottom row (rim diagnostic)
                    cap_rim_sil_max,
                    // [NEW V3RS-RIM-ANAT] anatomic mode flag (empty → legacy)
                    cap_rim_anatomic.empty() ? nullptr : &cap_rim_anatomic);

                // (2) Optional: also write the 4 PNG files for this
                //     Run via IoUDebug::dump. Skipped when toggle off
                //     or session_dir failed to create. Uses the FULL
                //     mesh indices (liverMesh3D->mIndices) inside
                //     IoUDebug::dump -- that's the unfiltered viz, by
                //     design (PNG comparison shows full-vs-mask, GL
                //     overlay shows filtered-vs-mask).
                if (!session_dir.empty()) {
                    std::ostringstream prefix;
                    prefix << "run_" << std::setw(2)
                           << std::setfill('0') << (run_idx + 1);
                    const int sw_g = gWindowWidth;
                    const int sh_g = gWindowHeight;
                    gWindowWidth  = cap_w;
                    gWindowHeight = cap_h;
                    IoUDebug::dump(session_dir + "/", prefix.str(),
                                   liverMesh3D,
                                   cap_view, cap_proj,
                                   cap_w, cap_h, /*step=*/8);
                    gWindowWidth  = sw_g;
                    gWindowHeight = sh_g;
                }
            };
    }

    CmaesRefine::ResultV3 r = CmaesRefineV3RS::runBipopCmaesV3RS(
        start_liver_verts, start_liver_normals, tgt_points,
        p, rmse_before, init_matched,
        outer_seed, cma_base);

    // Tear down Phase 0 selector. Order matters: drop the closure
    // FIRST (so the captured reference to liver_verts_backup is gone
    // before we move it), then restore liverMesh3D->mVertices to
    // its pre-Phase-D state so Phase E starts from the same vertices
    // it would have without the selector.
    p.sil_iou2d_eval_fn     = nullptr;
    p.sil_metrics_eval_fn   = nullptr;   // [NEW V3RS-SEL]
    p.sil_per_run_dump_fn   = nullptr;
    if (!liver_verts_backup.empty() && liverMesh3D
        && liverMesh3D->mVertices.size() == liver_verts_backup.size())
    {
        liverMesh3D->mVertices = std::move(liver_verts_backup);
    }
    stamp("D. CMA-ES V3R driver (10 runs + voxel + subset)");

    // ----- Phase E: Apply best_world_matrix + composite-score gate ----
    // [Phase A — unified internal cost]
    //
    // Layer 3 (this gate) now optimises the SAME scalar as Layer 1
    // (per-eval cost) and Layer 2 (Run selector):
    //
    //     score(x) = RMSE(x) + lambda_sil * (1 - IoU_occluded(x))
    //
    // where IoU_occluded comes from rasterize_iou2d_v3rs (the same
    // routine the optimiser hits per-eval) and `lambda_sil` is the
    // single hyperparameter that drives all three layers.
    //
    // The legacy `computeUnifiedMetrics()` call is intentionally KEPT
    // and still populates `registrationHandle.compIoU2D` (= IoU_full,
    // no occlusion). That value remains the Pose Library's external
    // metric so cross-method comparisons stay neutral. See
    // DESIGN_Occlusion_Aware_Silhouette_Anchor.md §3 / §4 for the
    // responsibility split.
    //
    // RMSE safety cap (5%) is preserved as an orthogonal guard against
    // degenerate IoU-only solutions that drop 3D fit quality.
    // [REMOVED] kRmseCapFactor (was 1.05f).
    //   Replaced by the dynamic cap computed at the accept/reject
    //   decision point below (g_ctrlgsRmseCapBase / Max / DiouFull).
    //   When g_ctrlgsUseDynamicCap is OFF, the cap reduces to
    //   g_ctrlgsRmseCapBase (default 1.05f → byte-identical to the
    //   legacy behaviour).
    const float lambda_layer3 = std::max(0.0f, p.lambda_sil);
    // [NEW RA-3] Layer 3 also weights the asymmetric terms identically
    // to the per-eval cost, so accept/reject is consistent with what the
    // optimiser scored. Zero when the respective toggle is OFF.
    const float lambda_out_l3     = g_ctrlgsUseOutsideRatio
                                    ? std::max(0.0f, g_ctrlgsLambdaOut)    : 0.0f;
    const float lambda_rim_sil_l3 = g_ctrlgsUseRimSil
                                    ? std::max(0.0f, g_ctrlgsLambdaRimSil) : 0.0f;
    const float rim_sil_max_l3    = std::max(1.0f, g_ctrlgsRimSilMaxPx);

    // Phase A baseline: IoU_occluded at the pre-Phase-D pose. The same
    // rasterize_iou2d_v3rs path the optimiser uses, evaluated at the
    // current liverMesh3D vertices. When the silhouette path is
    // disabled (lambda=0 or no dist map), we fall back to using
    // init_iou2d (IoU_full) so behaviour matches the legacy gate.
    float init_iou_occluded = init_iou2d;
    // [NEW RA-4] Capture the asymmetric / rim-sil metrics at baseline
    // so score_before contains the same terms as score_after. When the
    // corresponding lambda is 0 (toggle OFF), the rasterizer returns
    // 0 for that metric without doing the extra work.
    float init_outside_ratio = 0.0f;
    float init_rim_sil_loss  = 0.0f;
    if (lambda_layer3 > 0.0f
        && !p.sil_dist_map_2d.empty()
        && liverMesh3D
        && !liverMesh3D->mVertices.empty())
    {
        const glm::mat4 vp_view = buildSilhouetteView();
        const glm::mat4 vp_proj = buildSilhouetteProj();
        const glm::mat4 mvp_baseline = vp_proj * vp_view;

        const size_t nV = liverMesh3D->mVertices.size() / 3;
        std::vector<glm::vec3> positions(nV);
        for (size_t i = 0; i < nV; ++i) {
            positions[i] = glm::vec3(
                liverMesh3D->mVertices[i*3 + 0],
                liverMesh3D->mVertices[i*3 + 1],
                liverMesh3D->mVertices[i*3 + 2]);
        }
        const std::vector<float>* inst_ptr =
            p.sil_instrument_dist_map_2d.empty()
                ? nullptr : &p.sil_instrument_dist_map_2d;
        init_iou_occluded = CmaesRefineV3RS::rasterize_iou2d_v3rs(
            positions, p.sil_indices, mvp_baseline,
            p.sil_dist_map_2d, p.sil_img_w, p.sil_img_h, p.sil_raster_step,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            /*raster_mode*/0,
            inst_ptr, p.sil_instrument_thresh_px,
            // [NEW RA-4] capture outside_ratio + rim_sil_loss baseline
            (lambda_out_l3     > 0.0f) ? &init_outside_ratio : nullptr,
            (lambda_rim_sil_l3 > 0.0f) ? &init_rim_sil_loss  : nullptr,
            rim_sil_max_l3,
            // [NEW RA-RIM-ANAT] Forward anatomic-mode flag and skip the
            // cell-mask output (Phase A only needs scalars).
            (lambda_rim_sil_l3 > 0.0f && !p.is_rim_anatomic_full.empty())
                ? &p.is_rim_anatomic_full
                : nullptr,
            /*out_rim_cell_mask*/ nullptr);
    }

    const float score_before =
        rmse_before
        + lambda_layer3     * (1.0f - init_iou_occluded)
        + lambda_out_l3     * init_outside_ratio
        + lambda_rim_sil_l3 * init_rim_sil_loss;

    // [Phase B fix 2026-05-XX] Default-publish init_iou_occluded as the
    // applied-pose IoU. This is the correct value for two scenarios:
    //   (a) Phase E is skipped entirely because the V3RS selector found
    //       no Run that beat baseline_combo (r.best_run_idx == -1) --
    //       the mesh stays at the pre-V3RS pose, so its IoU_occluded is
    //       init_iou_occluded by definition.
    //   (b) Phase E runs but REJECTS, and the wrapper reverts the mesh
    //       back to the snapshot below. After revert the pose is again
    //       the pre-V3RS pose, so init_iou_occluded again.
    // The ACCEPT branch below overwrites this with iou_cand_occluded.
    //
    // Previously (buggy): the publish was done unconditionally with
    // iou_cand_occluded BEFORE the accept/reject decision. That made
    // the Pose Library record a stale candidate IoU on REJECT cases
    // (mesh reverted but IoU_occluded recorded the candidate), and
    // recorded 0 on skip cases (Phase E never reached). The CSV from
    // 2026-05-17 showed both pathologies: entries 8/9/12/13 with
    // IoU_occ=0 (skip) and entries 10/11 with IoU_occ=0.81 (stale
    // candidate, transform matrix identical to the rejected baseline).
    g_lastSilOccludedIoU2D = init_iou_occluded;

    if (r.best_run_idx >= 0) {
        // --- snapshot for revert ---
        std::vector<std::vector<GLfloat>> snap_v(organs.size());
        std::vector<std::vector<GLfloat>> snap_n(organs.size());
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                snap_v[i] = organs[i]->mVertices;
                snap_n[i] = organs[i]->mNormals;
            }
        }

        // --- apply matrix to all organs ---
        const glm::mat4& M = r.best_world_matrix;
        const glm::mat3  normalMat =
            glm::mat3(glm::transpose(glm::inverse(M)));

        double t_apply_sum = 0.0;
        double t_setup_sum = 0.0;
        int    organ_idx   = 0;
        for (auto* mesh : organs) {
            if (!mesh) { organ_idx++; continue; }

            const auto t_apply_start = clk::now();
            auto& v = mesh->mVertices;
            auto& n = mesh->mNormals;
            for (size_t i = 0; i + 2 < v.size(); i += 3) {
                glm::vec4 p4(v[i], v[i+1], v[i+2], 1.0f);
                glm::vec4 tp = M * p4;
                v[i]   = tp.x; v[i+1] = tp.y; v[i+2] = tp.z;
            }
            for (size_t i = 0; i + 2 < n.size(); i += 3) {
                glm::vec3 nm(n[i], n[i+1], n[i+2]);
                glm::vec3 tn = normalMat * nm;
                float len = glm::length(tn);
                if (len > 1e-8f) tn /= len;
                n[i]   = tn.x; n[i+1] = tn.y; n[i+2] = tn.z;
            }
            const auto t_apply_end = clk::now();
            t_apply_sum += ms_dur(t_apply_end - t_apply_start);

            const auto t_setup_start = clk::now();
            setUp(*mesh);
            const auto t_setup_end = clk::now();
            const double t_one_setup = ms_dur(t_setup_end - t_setup_start);
            t_setup_sum += t_one_setup;
            std::cout << "[CtrlGS/Time]   organ[" << organ_idx
                      << "] verts=" << (v.size() / 3)
                      << " apply=" << std::fixed << std::setprecision(1)
                      << ms_dur(t_apply_end - t_apply_start) << "ms"
                      << " setUp=" << t_one_setup << "ms"
                      << std::defaultfloat << std::endl;
            organ_idx++;
        }
        std::cout << "[CtrlGS/Time]   apply_sum=" << std::fixed
                  << std::setprecision(1) << t_apply_sum << "ms"
                  << "  setUp_sum=" << t_setup_sum << "ms"
                  << std::defaultfloat << std::endl;

        // --- post-apply metrics ---
        // computeUnifiedMetrics keeps populating registrationHandle.
        // compIoU2D with the LEGACY (no-occlusion) IoU so the Pose
        // Library / external reports stay method-neutral. The Layer 3
        // gate below ignores this value and uses iou_cand_occluded
        // computed via rasterize_iou2d_v3rs instead.
        computeUnifiedMetrics();
        g_metricsValid = true;
        const float rmse_cand        = registrationHandle.compRmse;
        const float iou_cand_full    = registrationHandle.compIoU2D;

        // Phase A: candidate IoU_occluded -- the rasterize_iou2d_v3rs
        // path used by Layer 1/2. When the silhouette path is OFF
        // (lambda=0 or no dist map) we fall back to iou_cand_full so
        // the gate reduces to the legacy composite-with-full-IoU
        // behaviour (which itself reduces to RMSE-only when lambda=0).
        float iou_cand_occluded = iou_cand_full;
        // [NEW RA-5] Capture candidate outside_ratio + rim_sil_loss
        // alongside iou_cand_occluded so score_after has the same terms
        // as score_before. Defaults to 0 when the corresponding toggle
        // is OFF, which means the term contributes zero to the score
        // (and the rasterizer skipped the extra computation).
        float cand_outside_ratio = 0.0f;
        float cand_rim_sil_loss  = 0.0f;
        if (lambda_layer3 > 0.0f
            && !p.sil_dist_map_2d.empty()
            && liverMesh3D
            && !liverMesh3D->mVertices.empty())
        {
            const glm::mat4 vp_view = buildSilhouetteView();
            const glm::mat4 vp_proj = buildSilhouetteProj();
            const glm::mat4 mvp_after = vp_proj * vp_view;

            const size_t nV = liverMesh3D->mVertices.size() / 3;
            std::vector<glm::vec3> positions(nV);
            for (size_t i = 0; i < nV; ++i) {
                positions[i] = glm::vec3(
                    liverMesh3D->mVertices[i*3 + 0],
                    liverMesh3D->mVertices[i*3 + 1],
                    liverMesh3D->mVertices[i*3 + 2]);
            }
            const std::vector<float>* inst_ptr =
                p.sil_instrument_dist_map_2d.empty()
                    ? nullptr : &p.sil_instrument_dist_map_2d;
            iou_cand_occluded = CmaesRefineV3RS::rasterize_iou2d_v3rs(
                positions, p.sil_indices, mvp_after,
                p.sil_dist_map_2d, p.sil_img_w, p.sil_img_h,
                p.sil_raster_step,
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                /*raster_mode*/0,
                inst_ptr, p.sil_instrument_thresh_px,
                // [NEW RA-5] candidate outside_ratio + rim_sil_loss
                (lambda_out_l3     > 0.0f) ? &cand_outside_ratio : nullptr,
                (lambda_rim_sil_l3 > 0.0f) ? &cand_rim_sil_loss  : nullptr,
                rim_sil_max_l3,
                // [NEW RA-RIM-ANAT] Forward anatomic-mode flag.
                (lambda_rim_sil_l3 > 0.0f && !p.is_rim_anatomic_full.empty())
                    ? &p.is_rim_anatomic_full
                    : nullptr,
                /*out_rim_cell_mask*/ nullptr);
        }

        const float score_after =
            rmse_cand
            + lambda_layer3     * (1.0f - iou_cand_occluded)
            + lambda_out_l3     * cand_outside_ratio
            + lambda_rim_sil_l3 * cand_rim_sil_loss;

        // [Phase B fix] The unconditional publish that used to be here
        // is now MOVED into the ACCEPT branch below. See the comment at
        // the default-publish above for the full semantic correction.

        // [NEW RA-5] Dynamic RMSE cap.
        // The legacy fixed cap (1.05x) blocked silhouette-improving
        // candidates that traded a few percent of RMSE for a meaningful
        // IoU jump -- exactly the recovery from Ctrl+G mask expansion.
        // When g_ctrlgsUseDynamicCap is ON, the cap factor interpolates
        // linearly between RmseCapBase (diou=0) and RmseCapMax
        // (diou>=RmseCapDiouFull). When OFF, cap is just RmseCapBase
        // (default 1.05 → legacy behaviour).
        const float diou      = iou_cand_occluded - init_iou_occluded;
        const float diou_full = std::max(0.001f, g_ctrlgsRmseCapDiouFull);
        const float cap_t     = g_ctrlgsUseDynamicCap
            ? std::clamp(diou / diou_full, 0.0f, 1.0f)
            : 0.0f;
        const float cap_factor = g_ctrlgsRmseCapBase
            + cap_t * (g_ctrlgsRmseCapMax - g_ctrlgsRmseCapBase);
        // [V3I] Pure-IoU (Ctrl+I): no RMSE cap, and accept on IoU gain
        //   rather than the RMSE-blended composite. Default path
        //   (g_ctrlgsPureIoUMode == false) is the unchanged Ctrl+Shift+G
        //   gate below.
        const bool rmse_cap_ok = g_ctrlgsPureIoUMode
            ? true
            : (rmse_cand < rmse_before * cap_factor);
        const bool score_improves = g_ctrlgsPureIoUMode
            ? (iou_cand_occluded > init_iou_occluded)
            : (score_after < score_before);

        if (score_improves && rmse_cap_ok) {
            r.improved = true;
            // [Phase B fix] Publish iou_cand_occluded ONLY on accept.
            // The pose has been applied, so the candidate's IoU is the
            // correct value for the current mesh state. On reject, this
            // line is skipped and the default-publish (init_iou_occluded
            // at the top of Phase E) remains in effect.
            g_lastSilOccludedIoU2D = iou_cand_occluded;
            std::cout << std::fixed << std::setprecision(6)
                      << "[Ctrl+Shift+G] ACCEPTED"
                      << "  score: " << score_before << " -> " << score_after
                      << "  RMSE: " << rmse_before   << " -> " << rmse_cand
                      << "  IoU_occluded: " << init_iou_occluded
                      << " -> " << iou_cand_occluded
                      << "  outside: " << init_outside_ratio
                      << " -> " << cand_outside_ratio
                      << "  rim_sil: " << init_rim_sil_loss
                      << " -> " << cand_rim_sil_loss
                      << "  (cap=" << cap_factor << ", diou=" << diou
                      << ", IoU_full[ref]: " << init_iou2d
                      << " -> " << iou_cand_full
                      << ", lambdas=[" << lambda_layer3
                      << "," << lambda_out_l3
                      << "," << lambda_rim_sil_l3 << "])"
                      << std::defaultfloat << std::endl;
        } else {
            if (score_after < score_before && !rmse_cap_ok) {
                std::cout << std::fixed << std::setprecision(6)
                          << "[Ctrl+Shift+G] REJECTED (RMSE cap: "
                          << rmse_cand << " >= " << rmse_before
                          << " * " << cap_factor << "="
                          << rmse_before * cap_factor << ")"
                          << "  score would have been: "
                          << score_before << " -> " << score_after
                          << "  IoU_occluded: " << init_iou_occluded
                          << " -> " << iou_cand_occluded
                          << "  diou=" << diou
                          << "  (dynamic_cap="
                          << (g_ctrlgsUseDynamicCap ? "ON" : "OFF")
                          << ")"
                          << std::defaultfloat << std::endl;
            }
            r.improved = false;
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i] && !snap_v[i].empty()) {
                    organs[i]->mVertices = snap_v[i];
                    organs[i]->mNormals  = snap_n[i];
                    setUp(*organs[i]);
                }
            }
            computeUnifiedMetrics();
            g_metricsValid = true;
            std::cout << std::fixed << std::setprecision(6)
                      << "[Ctrl+Shift+G] REJECTED (composite no improvement)"
                      << "  score: " << score_before << " -> " << score_after
                      << "  RMSE: " << rmse_before   << " -> " << rmse_cand
                      << "  IoU_occluded: " << init_iou_occluded
                      << " -> " << iou_cand_occluded
                      << "  outside: " << init_outside_ratio
                      << " -> " << cand_outside_ratio
                      << "  rim_sil: " << init_rim_sil_loss
                      << " -> " << cand_rim_sil_loss
                      << "  (cap=" << cap_factor << ", diou=" << diou
                      << ", IoU_full[ref]: " << init_iou2d
                      << " -> " << iou_cand_full
                      << ", lambdas=[" << lambda_layer3
                      << "," << lambda_out_l3
                      << "," << lambda_rim_sil_l3 << "])"
                      << std::defaultfloat << std::endl;
        }
    } else {
        std::cout << "[CtrlGS/Time]   (no best_run found, skipped apply)"
                  << std::endl;
    }
    stamp("E. apply_matrix + setUp x6");

    // ----- Phase E.5: Capture silhouette projection (debug viz) -----
    // After apply, liverMesh3D is at the optimised pose. We capture the
    // SAME set the silhouette eval iterated over: the rim ∩ quadrant
    // subset (typically 50-150 pts). Each rim_subset voxel index is
    // mapped back to its original mesh vertex via p.voxel_to_orig, then
    // we read the post-apply world position from liverMesh3D->mVertices.
    //
    // Old version captured ~6500 voxels with z-buffer occlusion and
    // sub-sampled to 600. The new eval only looks at rim_subset, and
    // rim_subset is already small enough to display all points without
    // sub-sampling -- which makes the overlay show EXACTLY what the
    // optimiser was scored against.
    //
    // Skipped when sil was inactive (lambda_sil == 0 or dist map
    // missing) OR no improvement (so the stale capture from the
    // previous run stays visible).
    if (r.improved
        && p.lambda_sil > 0.0f
        && !p.sil_dist_map_2d.empty()
        && p.sil_img_w > 0 && p.sil_img_h > 0
        && !p.voxel_to_orig.empty()
        && !p.is_rim_src_voxel.empty()
        && !p.subset_idx_voxel.empty()
        && liverMesh3D)
    {
        const auto& V  = liverMesh3D->mVertices;
        const size_t nV = V.size() / 3;
        const auto& vox_to_orig = p.voxel_to_orig;

        // Re-derive rim_subset_indices (same logic as run_one_bipop_v3rs)
        // so the viz set is byte-identical to what eval iterated over.
        std::vector<int> rim_subset_indices;
        rim_subset_indices.reserve(p.subset_idx_voxel.size() / 4);
        for (int idx : p.subset_idx_voxel) {
            if (idx < 0) continue;
            if ((size_t)idx >= p.is_rim_src_voxel.size()) continue;
            if (p.is_rim_src_voxel[idx]) rim_subset_indices.push_back(idx);
        }

        // Build world positions from rim_subset voxels only.
        std::vector<glm::vec3> liver_world;
        liver_world.reserve(rim_subset_indices.size());
        for (int vIdx_voxel : rim_subset_indices) {
            if (vIdx_voxel < 0 ||
                (size_t)vIdx_voxel >= vox_to_orig.size()) continue;
            int vIdx = vox_to_orig[(size_t)vIdx_voxel];
            if (vIdx < 0 || (size_t)vIdx >= nV) continue;
            liver_world.emplace_back(
                V[(size_t)vIdx * 3 + 0],
                V[(size_t)vIdx * 3 + 1],
                V[(size_t)vIdx * 3 + 2]);
        }

        std::vector<glm::vec3> dbg_world;
        std::vector<float>     dbg_dist;
        int n_vis = 0, n_sig = 0;
        CmaesRefineV3RS::captureSilProjectionDebug(
            liver_world,
            p.sil_fx, p.sil_fy, p.sil_cx, p.sil_cy,
            p.sil_img_w, p.sil_img_h,
            p.sil_dist_map_2d,
            dbg_world, dbg_dist, n_vis, n_sig);

        // No subsampling needed -- rim_subset is already small.
        g_silProjDebug.pts.clear();
        g_silProjDebug.pts.reserve(dbg_world.size());

        double sum_dist = 0.0;
        int    n_sig_kept = 0;
        for (size_t i = 0; i < dbg_world.size(); ++i) {
            SilProjDebugPoint pt;
            pt.world_pos = dbg_world[i];
            pt.dist_px   = dbg_dist [i];
            g_silProjDebug.pts.push_back(pt);
            if (pt.dist_px < 9000.0f) {
                sum_dist  += (double)pt.dist_px;
                ++n_sig_kept;
            }
        }

        const float diag = std::sqrt(
            (float)(p.sil_img_w * p.sil_img_w +
                    p.sil_img_h * p.sil_img_h));

        g_silProjDebug.valid          = true;
        g_silProjDebug.img_w          = p.sil_img_w;
        g_silProjDebug.img_h          = p.sil_img_h;
        g_silProjDebug.n_visible      = n_vis;
        g_silProjDebug.n_with_signal  = n_sig;
        g_silProjDebug.mean_dist_px   = (n_sig_kept > 0)
                                        ? (float)(sum_dist / (double)n_sig_kept)
                                        : 0.0f;
        g_silProjDebug.mean_dist_norm = (diag > 1e-6f)
                                        ? g_silProjDebug.mean_dist_px / diag
                                        : 0.0f;

        std::cout << "[Ctrl+Shift+G/SilViz] captured "
                  << g_silProjDebug.pts.size()
                  << " pts (rim_subset, no subsample)"
                  << "  in_mask=" << n_sig_kept
                  << "  out_of_mask=" << (g_silProjDebug.pts.size() - n_sig_kept)
                  << "  mean_dist=" << std::fixed << std::setprecision(2)
                  << g_silProjDebug.mean_dist_px << "px ("
                  << std::setprecision(4) << g_silProjDebug.mean_dist_norm
                  << " norm)"
                  << std::defaultfloat << std::endl;
    }
    stamp("E5. capture sil projection (viz)");

    // ----- Phase E.6: Final-pose silhouette capture (debug viz) ------
    // The applied final pose (best Run's M_world) is now in
    // liverMesh3D. Capture it to the SilOverlay "Final" slot so the
    // ImGui window can show what V3RS actually applied vs each of the
    // 10 candidate Runs. Additionally write PNGs to disk if the dump
    // toggle is on (PNG output is for paper figures / offline diff).
    if (p.lambda_sil > 0.0f
        && g_boundaryDistMap.valid
        && liverMesh3D)
    {
        const glm::mat4 sil_view_f = buildSilhouetteView();
        const glm::mat4 sil_proj_f = buildSilhouetteProj();
        const int sil_w_f = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
        const int sil_h_f = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

        // (1) Always: SilOverlay ImGui capture for the Final slot.
        //     Pass instrument-occlusion args so the F9 Final IoU and
        //     composite match what CMA-ES evaluated. p.sil_instrument_*
        //     is the same data the per-Run callback captured above;
        //     empty -> filter OFF, byte-identical to pre-feature path.
        SilOverlay::captureFinal(
            SilOverlay::g_silOverlay, r.best_run_idx, liverMesh3D,
            p.sil_indices, sil_view_f, sil_proj_f, p.sil_dist_map_2d,
            sil_w_f, sil_h_f, /*step=*/8, r.best_srt.scale,
            p.sil_instrument_dist_map_2d.empty()
                ? nullptr
                : &p.sil_instrument_dist_map_2d,
            p.sil_instrument_thresh_px,
            // [NEW V3RS-VIZ] enable bottom row (rim diagnostic) for
            // the Final slot too. Always >0 so the 6-panel layout is
            // built regardless of whether rim_sil_loss is in the cost.
            p.rim_sil_max_px,
            // [NEW V3RS-RIM-ANAT] anatomic mode flag (empty → legacy)
            p.is_rim_anatomic_full.empty()
                ? nullptr
                : &p.is_rim_anatomic_full);

        // (2) Optional: write final_*.png to the same session dir
        //     used by the per-Run callback.
        if (g_silDumpPerRunEnabled && g_silDumpSessionCounter > 0) {
            std::ostringstream session_dir_oss;
            session_dir_oss << DEPTH_OUTPUT_PATH << "v3rs_dump/session_"
                            << std::setw(3) << std::setfill('0')
                            << g_silDumpSessionCounter;
            const std::string session_dir = session_dir_oss.str();

            const int sw_g = gWindowWidth;
            const int sh_g = gWindowHeight;
            gWindowWidth  = sil_w_f;
            gWindowHeight = sil_h_f;
            IoUDebug::dump(session_dir + "/", "final",
                           liverMesh3D,
                           sil_view_f, sil_proj_f,
                           sil_w_f, sil_h_f, /*step=*/8);
            gWindowWidth  = sw_g;
            gWindowHeight = sh_g;

            std::cout << "[V3RS/Dump] wrote final-pose PNGs to "
                      << session_dir << "/final_*.png" << std::endl;
        }
    }
    stamp("E6. final-pose hitmap dump (viz)");

    // ----- Phase F: metrics already updated in Phase E ----------------
    // Phase E で apply/revert 後に必ず computeUnifiedMetrics() を呼んでいるので
    // ここでは再計測不要。g_metricsValid だけ保証する。
    g_metricsValid = true;
    const float rmse_after = registrationHandle.compRmse;
    const float iou_after  = registrationHandle.compIoU2D;
    stamp("F. post_computeUnifiedMetrics");

    // ----- Build post-Apply / post-revert vertex array (shared) -----
    // [NEW V3RS-RIM-DIAG / V3RS-CONTAIN-RATIO] Built unconditionally so
    // both Phase F.5 (RIM-only RMSE) and Phase F.5b (Containment) can
    // share it. On accept: post-Apply pose; on reject: snap-restored
    // pre-Apply pose. Either way it matches what PoseLibrary will record.
    std::vector<glm::vec3> liver_verts_now;
    if (liverMesh3D) {
        const auto& v = liverMesh3D->mVertices;
        liver_verts_now.reserve(v.size() / 3);
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            liver_verts_now.emplace_back(v[i], v[i+1], v[i+2]);
        }
    }

    // ----- Phase F.5: RIM-only RMSE "after" -------------------------
    // [NEW V3RS-RIM-DIAG] Mirror of Ctrl+G's Phase F.5. Uses the same
    // rim_diag inputs built in Phase C.5 (shared rim_src mask + rim_tgt
    // subset) so before/after measurements are directly comparable.
    int rim_diag_n_src_rim_after = 0;
    int rim_diag_matched_after   = 0;
    // [Phase A] Mirror of Ctrl+G Phase F.5 pair capture. Same publish-
    // and-consume contract — globals get copied onto the PoseEntry by
    // poseSaveToLibrary and cleared. liver_verts_now (built just above)
    // reflects post-Apply on accept or reverted pose on reject, so the
    // captured pairs always match what PoseLibrary will record.
    g_lastRimPairSrcVertIdx.clear();
    g_lastRimPairTgtPos.clear();
    const float rmse_rim_after = compute_rim_only_rmse_diag(
        liver_verts_now, rim_diag_is_rim_src, rim_diag_tgt_rim_points,
        max_dist_sq_rim_diag_shg,
        rim_diag_n_src_rim_after, rim_diag_matched_after,
        &g_lastRimPairSrcVertIdx, &g_lastRimPairTgtPos);
    stamp("F.5. rim_only_rmse_after");

    // ----- Phase F.5b: IoU_occluded Containment diagnostic ----------
    // [NEW V3RS-CONTAIN-RATIO] Mirror of Ctrl+G's F.5b. Single
    // post-Apply rasterize at step=8 (~3-5 ms) to extract cell-level
    // src/tgt/inter counts so precision, recall, size_ratio and
    // overshoot_fraction can be published to the PoseLibrary entry.
    //
    // liver_verts_now (built just above) reflects the post-Apply pose
    // on accept (Phase E wrote it back) and the reverted pose on
    // reject (Phase E restored snap_v[]). Either way it matches what
    // the new PoseLibrary entry will record, so the diagnostic is
    // correct for both branches without conditioning on r.improved.
    //
    // Held-out diagnostic — display only, never gates acceptance. Same
    // semantics, gating policy and instrument-mask AUTO behaviour as
    // Ctrl+G's F.5b so the CONT column in PoseLibrary has stable
    // meaning across both methods.
    float iou_occ_precision_shg = -1.0f;
    float iou_occ_recall_shg    = -1.0f;
    int   iou_occ_src_cells_shg = 0;
    int   iou_occ_tgt_cells_shg = 0;
    int   iou_occ_int_cells_shg = 0;
    if (g_boundaryDistMap.valid
        && g_boundaryDistMap.width  > 1
        && g_boundaryDistMap.height > 1
        && g_boundaryDistMap.data.size() ==
               (size_t)g_boundaryDistMap.width *
               (size_t)g_boundaryDistMap.height
        && liverMesh3D
        && !liverMesh3D->mIndices.empty())
    {
        const glm::mat4 sil_view_d = buildSilhouetteView();
        const glm::mat4 sil_proj_d = buildSilhouetteProj();
        const glm::mat4 sil_mvp_d  = sil_proj_d * sil_view_d;
        const int sil_w_d = g_boundaryDistMap.width;
        const int sil_h_d = g_boundaryDistMap.height;

        std::vector<uint32_t> sil_indices_full_d(
            liverMesh3D->mIndices.begin(),
            liverMesh3D->mIndices.end());

        // Instrument-occlusion mask: AUTO (same policy as Ctrl+G F.5b).
        // The Ctrl+Shift+G UI toggle g_ctrlgsIgnoreInstrument is NOT
        // consulted here so PoseLibrary CONT semantics are stable.
        const std::vector<float>* inst_ptr_d = nullptr;
        float inst_thresh_d = 0.0f;
        const bool inst_loaded_d = ensureInstrumentDistMap();
        if (inst_loaded_d
            && g_instrumentDistMap.valid
            && g_instrumentDistMap.width  == sil_w_d
            && g_instrumentDistMap.height == sil_h_d
            && g_instrumentDistMap.data.size() ==
                   (size_t)sil_w_d * (size_t)sil_h_d)
        {
            inst_ptr_d    = &g_instrumentDistMap.data;
            inst_thresh_d = std::max(0.0f, g_instrumentPxThresh);
        }

        std::vector<uint8_t> hitmap_d, tmask_d;
        int gw_d = 0, gh_d = 0;
        (void)CmaesRefineV3RS::rasterize_iou2d_v3rs(
            liver_verts_now, sil_indices_full_d, sil_mvp_d,
            g_boundaryDistMap.data, sil_w_d, sil_h_d, /*step=*/8,
            &hitmap_d, &tmask_d, &gw_d, &gh_d,
            nullptr, nullptr, nullptr,
            /*raster_mode=*/0,
            inst_ptr_d, inst_thresh_d);

        if (gw_d > 0 && gh_d > 0) {
            int inter_c = 0, src_c = 0, tgt_c = 0;
            const size_t N_c = (size_t)gw_d * (size_t)gh_d;
            for (size_t i = 0; i < N_c; ++i) {
                const bool s = (hitmap_d[i] != 0);
                const bool t = (tmask_d[i]  != 0);
                if (s)         ++src_c;
                if (t)         ++tgt_c;
                if (s && t)    ++inter_c;
            }
            iou_occ_int_cells_shg = inter_c;
            iou_occ_src_cells_shg = src_c;
            iou_occ_tgt_cells_shg = tgt_c;
            iou_occ_precision_shg = (src_c > 0)
                ? (float)inter_c / (float)src_c : 0.0f;
            iou_occ_recall_shg    = (tgt_c > 0)
                ? (float)inter_c / (float)tgt_c : 0.0f;
        }
    }
    stamp("F.5b. iou_occluded_diag");

    // Publish to PoseLibrary globals. Same sentinel convention as
    // Ctrl+G's F.5b (-1 = N/A → PoseLibrary shows "—" in CONT column).
    g_lastIoUOccPrecision = iou_occ_precision_shg;
    g_lastIoUOccRecall    = iou_occ_recall_shg;

    // [NEW V3RS-RIM-DIAG] Publish RIM-only RMSE values for PoseLibrary
    // entry. -1 = N/A (same convention as Ctrl+G F.5 publish). The
    // poseSaveToLibrary consume-and-clear pattern protects subsequent
    // non-RIM-diagnostic saves from seeing this value.
    g_lastRimRmse     = rmse_rim_after;   // -1.0f if N/A
    g_lastRimMatched  = rim_diag_matched_after;
    g_lastRimTgtTotal = (int)rim_diag_tgt_rim_points.size();
    g_lastRimSrcTotal = rim_diag_n_src_rim_after;

    std::cout << std::defaultfloat << std::setprecision(6);
    const float improvement = rmse_before - rmse_after;
    std::cout << "[Ctrl+Shift+G] Best: " << rmse_before << " -> " << rmse_after
              << " (delta=" << improvement << ")"
              << "  IoU2D: " << init_iou2d << " -> " << iou_after
              << "  best_run="
              << (r.best_run_idx < 0 ? std::string("none")
                                     : std::to_string(r.best_run_idx + 1))
              << "  total_gens=" << r.total_generations
              << "  Q=" << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
              << (r.improved ? "  [ACCEPTED]" : "  [NO CHANGE]")
              << std::endl;

    // [NEW V3RS-RIM-DIAG] RIM-only RMSE diagnostic line (parallels
    // Ctrl+G). Format matches Ctrl+G's so a paper-ready log dump can
    // interleave both sessions with the same regex.
    // -1.0f means N/A (target has no boundaryDist or rim set is empty).
    if (rmse_rim_before > 0.0f && rmse_rim_after > 0.0f) {
        const float rim_delta = rmse_rim_before - rmse_rim_after;
        std::cout << "[Ctrl+Shift+G] RIM-only: " << rmse_rim_before
                  << " -> " << rmse_rim_after
                  << " (delta=" << rim_delta << ")"
                  << "  src_rim=" << rim_diag_n_src_rim_before
                  << "  tgt_rim=" << rim_diag_tgt_rim_points.size()
                  << "  matched(before/after)="
                  << rim_diag_matched_before << "/" << rim_diag_matched_after
                  << (rim_delta >  0.0001f ? "  [RIM IMPROVED]"  :
                      rim_delta < -0.0001f ? "  [RIM WORSE]"     :
                                             "  [RIM NO CHANGE]")
                  << std::endl;
    } else {
        std::cout << "[Ctrl+Shift+G] RIM-only: N/A"
                  << "  src_rim=" << rim_diag_n_src_rim_before
                  << "  tgt_rim=" << rim_diag_tgt_rim_points.size()
                  << "  (no boundaryDist or empty rim set)" << std::endl;
    }

    // [NEW V3RS-CONTAIN-RATIO] Containment line (parallels Ctrl+G).
    // Display-only; never gates acceptance. Mirrors the format used by
    // Ctrl+G so a paper-ready log dump can interleave both sessions
    // and parse with the same regex.
    if (iou_occ_precision_shg >= 0.0f && iou_occ_recall_shg >= 0.0f) {
        const float dir = iou_occ_recall_shg - iou_occ_precision_shg;
        const char* tag = (std::fabs(dir) < 0.05f) ? "[BALANCED]"
                       : (dir > 0.0f)              ? "[OVERSHOOT src>tgt]"
                                                   : "[UNDERSHOOT src<tgt]";
        const float size_ratio = (iou_occ_tgt_cells_shg > 0)
            ? (float)iou_occ_src_cells_shg / (float)iou_occ_tgt_cells_shg
            : 0.0f;
        const float overshoot_frac = (iou_occ_tgt_cells_shg > 0)
            ? (float)std::max(0,
                              iou_occ_src_cells_shg - iou_occ_int_cells_shg)
              / (float)iou_occ_tgt_cells_shg
            : 0.0f;
        std::cout << "[Ctrl+Shift+G] Containment: size=" << std::fixed
                  << std::setprecision(2) << size_ratio << "x"
                  << "  overshoot=" << std::setprecision(0)
                  << 100.0f * overshoot_frac << "%"
                  << std::defaultfloat << std::setprecision(6)
                  << "  recall=" << iou_occ_recall_shg
                  << "  precision=" << iou_occ_precision_shg
                  << "  (inter=" << iou_occ_int_cells_shg
                  << " src=" << iou_occ_src_cells_shg
                  << " tgt=" << iou_occ_tgt_cells_shg << " cells)  "
                  << tag << std::endl;
    } else {
        std::cout << "[Ctrl+Shift+G] Containment: N/A"
                  << "  (IoU_occ not computed)" << std::endl;
    }

    const double t_grand_total = ms_dur(clk::now() - t_phase_start);
    std::cout << "[CtrlGS/Time] === GRAND TOTAL: "
              << std::fixed << std::setprecision(1) << t_grand_total
              << " ms ===" << std::defaultfloat << std::endl;

    // [PoseLibrary save fix] V3R と同じ理由で末尾で state を REGISTERED に。
    // Ctrl+Shift+G が初期 pose 直後から走ったとき (= state が IDLE のまま)、
    // 完走後の poseSaveToLibrary が冒頭ガードで bail しないようにする。
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;

    g_callIdx++;  // V3R: 末尾でインクリメント (V1 / V2 / V3 と同じ)
}


// =========================================================
//  [V3I / Ctrl+I] runBipopCmaesV3I -- pure squash-IoU registration.
//  ---------------------------------------------------------
//  This is literally "Ctrl+Shift+G with the objective variable
//  swapped to IoU-only": it reuses the ENTIRE V3RS pipeline
//  (BIPOP, 10 restarts, sigma adaptation, SRT space, voxel prep,
//  the rasterize_iou2d_v3rs squash/step16 silhouette) and only
//  changes the cost from  RMSE_W + lambda*(1-IoU)  to  (1 - IoU).
//
//  Mechanism: flip the g_ctrlgsPureIoUMode global for the duration
//  of one runBipopCmaesV3RS() call. The V3RS engine reads it via
//  ParamsV3RS::pure_iou_mode (set in the wrapper) and:
//    - inner cost            = (1 - IoU2D)        [no RMSE term]
//    - Run selector decides  by IoU2D
//    - wrapper accept gate    skips the RMSE cap, accepts on IoU gain
//  Default (flag false) leaves Ctrl+Shift+G byte-identical, and
//  Ctrl+G (V3R) is untouched entirely.
//
//  Bound to Ctrl+I in main.cpp.
// =========================================================
inline void runBipopCmaesV3I(uint8_t quadrant_mask) {
    const bool prev = g_ctrlgsPureIoUMode;
    g_ctrlgsPureIoUMode = true;
    std::cout << "[Ctrl+I] pure squash-IoU mode ON (V3RS mechanism, "
                 "cost = 1 - IoU2D)" << std::endl;
    runBipopCmaesV3RS(quadrant_mask);   // engine reads the flag via params
    g_ctrlgsPureIoUMode = prev;
}


//  ---------------------------------------------------------
//  Measurement-only. Does NOT run CMA-ES, does NOT touch the optimiser
//  or liverMesh3D's pose. Rasterizes the quadrant-filtered liver
//  silhouette at the CURRENT static pose with BOTH the hot-path
//  triangle-bbox splat and a plain vertex-squash 3x3, then:
//    (1) prints the [V3RS/vsq-diag] console block (hole count, write
//        counts, projected edge-length histogram, ASCII A/B map) via
//        CmaesRefineV3RS::diagnoseVertexSquashOnce
//    (2) uploads both composites to the F9 overlay's Diagnostic slot
//        (SilOverlay::captureDiag) so the result is visible with F9
//
//  This is the Phase A diagnostic for the vertex-squash / adaptive-
//  subdivision line of work (HANDOVER §5.2). It follows the §4.3 "new
//  diagnostic function" pattern: compare the current hot path against
//  a candidate raster BEFORE touching the hot path.
//
//  Bound to F10 in main.cpp. Independent of Ctrl+Shift+G; reads the
//  same globals the Ctrl+Shift+G prep reads (g_liverRegion / g_liverLR
//  for the quadrant filter, g_boundaryDistMap for the SAM2 target,
//  OrbitCam via buildSilhouetteProj for intrinsics) but builds nothing
//  the optimiser needs and writes no optimiser state.
// =========================================================
inline void diagnoseVertexSquashV3RS(uint8_t quadrant_mask) {
    std::cout << "\n=== V3RS vertex-squash diagnostic (F10) ===" << std::endl;
    std::cout << "[F10] quadrant_mask = "
              << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
              << "  (0x" << std::hex << (int)quadrant_mask << std::dec
              << ")" << std::endl;

    // ----- Guards ---------------------------------------------------
    if (!liverMesh3D) {
        std::cerr << "[F10] liverMesh3D is null; aborting." << std::endl;
        return;
    }
    if (liverMesh3D->mVertices.empty() || liverMesh3D->mIndices.empty()) {
        std::cerr << "[F10] liver mesh has no geometry; aborting." << std::endl;
        return;
    }
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[F10] Region/LR labels not computed "
                  << "(Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << ", LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << "). Run HemiAuto (O) first." << std::endl;
        return;
    }
    if (!g_boundaryDistMap.valid) {
        std::cerr << "[F10] g_boundaryDistMap not valid -- need a SAM2 "
                     "boundary distance map first. Aborting." << std::endl;
        return;
    }
    const int img_w = g_boundaryDistMap.width;
    const int img_h = g_boundaryDistMap.height;
    if (img_w <= 0 || img_h <= 0 ||
        g_boundaryDistMap.data.size() != (size_t)img_w * (size_t)img_h) {
        std::cerr << "[F10] g_boundaryDistMap size mismatch (data="
                  << g_boundaryDistMap.data.size() << " vs "
                  << img_w << "x" << img_h << "). Aborting." << std::endl;
        return;
    }

    // Label-size sanity (mirrors the Ctrl+Shift+G prep check).
    const size_t nV = liverMesh3D->mVertices.size() / 3;
    if (g_liverRegion.labels.size() != nV ||
        g_liverLR.labels.size()     != nV) {
        std::cerr << "[F10] Label-size mismatch: verts=" << nV
                  << "  region.labels=" << g_liverRegion.labels.size()
                  << "  lr.labels=" << g_liverLR.labels.size()
                  << " (mesh may have been reloaded without re-labeling)."
                  << std::endl;
        return;
    }

    // ----- Quadrant-filtered triangle list --------------------------
    // IDENTICAL filter to the Ctrl+Shift+G prep (a triangle is kept iff
    // all 3 of its vertices are in the active quadrant) so the
    // diagnostic rasterizes EXACTLY the triangle set the cost function
    // would see.
    std::vector<uint32_t> sil_indices;
    {
        const std::vector<int> quad_vert_ids =
            LiverLeftRightLabel::makeQuadrantSubsetIdx(
                g_liverRegion.labels, g_liverLR.labels, quadrant_mask);
        std::vector<uint8_t> in_quad(nV, 0);
        for (int idx : quad_vert_ids) {
            if (idx >= 0 && (size_t)idx < nV) in_quad[(size_t)idx] = 1;
        }
        const auto& src_idx = liverMesh3D->mIndices;
        sil_indices.reserve(src_idx.size());
        int n_tris_total = 0, n_tris_kept = 0;
        for (size_t t = 0; t + 2 < src_idx.size(); t += 3) {
            ++n_tris_total;
            const uint32_t i0 = src_idx[t + 0];
            const uint32_t i1 = src_idx[t + 1];
            const uint32_t i2 = src_idx[t + 2];
            if (i0 >= nV || i1 >= nV || i2 >= nV) continue;
            if (in_quad[i0] && in_quad[i1] && in_quad[i2]) {
                sil_indices.push_back(i0);
                sil_indices.push_back(i1);
                sil_indices.push_back(i2);
                ++n_tris_kept;
            }
        }
        std::cout << "[F10] quad-filter tris: " << n_tris_kept << " / "
                  << n_tris_total << "  (Q="
                  << LiverLeftRightLabel::quadrantMaskString(quadrant_mask)
                  << ")" << std::endl;
        if (sil_indices.empty()) {
            std::cerr << "[F10] quadrant filter left 0 triangles; aborting."
                      << std::endl;
            return;
        }
    }

    // ----- View / proj / step (hot-path-consistent) -----------------
    const glm::mat4 sil_view = buildSilhouetteView();
    const glm::mat4 sil_proj = buildSilhouetteProj();
    // Use the hot-path raster step (ParamsV3RS default) so the holes
    // and write counts reflect what the optimiser actually sees.
    const int diag_step = CmaesRefineV3RS::ParamsV3RS{}.sil_raster_step;

    // ----- Target-mask cache (source-parity raster) -----------------
    // Build / hit the same squashed-and-haloed target mask the hot path
    // uses, so diagnoseVertexSquashOnce scores IoU(A) and IoU(B)
    // against the cache path (matches rasterize_iou2d_v3rs Step 3 when
    // g_silTargetSquashEnabled is on). Content-fingerprinted: a cache
    // built by a prior Ctrl+Shift+G press for the same SAM2 mask is
    // hit, not rebuilt.
    CmaesRefineV3RS::ensureSilTargetMaskCache(
        g_boundaryDistMap.data, img_w, img_h, diag_step);

    // ----- liverMesh3D vertices -> vec3 -----------------------------
    // liverMesh3D->mVertices is already at the current pose, so the
    // model matrix is identity and mvp = proj * view.
    std::vector<glm::vec3> positions(nV);
    {
        const auto& V = liverMesh3D->mVertices;
        for (size_t i = 0; i < nV; ++i) {
            positions[i] = glm::vec3(V[i*3 + 0], V[i*3 + 1], V[i*3 + 2]);
        }
    }
    const glm::mat4 mvp = sil_proj * sil_view;

    // ----- (1) Console A/B diagnostic -------------------------------
    CmaesRefineV3RS::diagnoseVertexSquashOnce(
        positions, sil_indices, mvp,
        g_boundaryDistMap.data, img_w, img_h, diag_step,
        /*run_index=*/-1);   // -1 = F10 on-demand (current static pose)

    // ----- (2) F9 overlay capture (both raster modes) ---------------
    // captureDiag rasterizes twice (bbox + vtx-squash), uploads both
    // composites, and jumps the F9 selector to the Diagnostic entry.
    const float iou_bbox = SilOverlay::captureDiag(
        SilOverlay::g_silOverlay, liverMesh3D, sil_indices,
        sil_view, sil_proj, g_boundaryDistMap.data,
        img_w, img_h, diag_step, /*scale_value=*/1.0f);

    std::cout << "[F10] F9 Diagnostic slot updated (bbox IoU="
              << std::fixed << std::setprecision(4) << iou_bbox
              << std::defaultfloat
              << ").  Press F9 to view; the checkbox toggles "
                 "bbox <-> vtx-squash." << std::endl;
}


// =========================================================
//  Silhouette Alignment (Shift+E)
//  SAM2マスクのシルエットとソースメッシュのラスタライズ投影を
//  Kカメラ固定行列でIoU最大化。buildProjectedLiverMaskは不要。
// =========================================================
// Alt+P (Silhouette Align / runShiftE) tunables — surfaced in Ctrl+D > G tab.
// Were hardcoded locals (N_STARTS=5, raster step=8) before the G/W panel split.
// raster step drives BOTH the IoU measurement and the CMA-ES objective (linked).
inline int g_shiftE_NStarts    = 5;
inline int g_shiftE_RasterStep = 8;

// [UI整理] Alt+P "pure IoU" default. runShiftE blends a small 3D term
// (alpha_3d) into the silhouette objective. ON (default) drops it
// (alpha_3d = 0) so Alt+P is driven purely by the 2D-IoU silhouette
// match — the same intent as Ctrl+I, but via the lighter/faster V1
// CmaesRefine path (use_silhouette_2d_fast stays ON regardless). OFF
// restores the legacy alpha_3d = 0.3 blend.
inline bool g_shiftE_pureIoU   = true;

// [QUAD-SIL] Alt+P quadrant-aware silhouette (default ON).
//   ON  : when the Ctrl+G Quadrant Selector (g_activeQuadrantMask) is NOT
//         QUAD_ALL, runShiftE rasterizes only the triangles whose 3 vertices
//         all belong to the active quadrants (same Phase-2c rule the V3RS
//         wrapper applies for Ctrl+Shift+G / Ctrl+I). The CMA-ES inner
//         objective, the V1 accept gate, and the before/after IoU printout
//         all use this filtered raster (one switch — see
//         CmaesRefine::g_silTriOverride). Intended for the occlusion
//         scenario: partial SAM2 mask vs the matching liver quadrant(s).
//   OFF : legacy full-mesh raster regardless of the quadrant selection
//         (A/B comparison baseline).
//   At QUAD_ALL the filter is bypassed entirely (override stays null), so
//   the default state is byte-identical to the pre-feature build in BOTH
//   toggle positions.
inline bool g_shiftE_quadrantAware = true;

// [QUAD-SIL BRAKE] Mask-expansion brake weight for Alt+P quadrant mode.
//   Added to the V1 silhouette cost as lambda * outside_ratio (the share of
//   source cells lying outside the target mask). Mirrors the V3RS
//   g_ctrlgsLambdaOut default of 0.5. Only consulted in quadrant mode
//   (runShiftE arms CmaesRefine::g_silOutsideLambda from this); the legacy
//   full-mesh path never sets the brake, so it stays byte-identical.
//   Raise to penalise inflation harder; 0 disables the brake (back to raw
//   symmetric IoU, i.e. the "blanket the target" failure mode).
inline float g_shiftE_outsideLambda = 0.5f;

// [Alt+P found-pose viz] ON -> runShiftE captures EACH run's inner-loop best
// (found) pose into an F9 Run slot BEFORE the V1 accept gate reverts it, plus
// the applied pose into Final. Lets F9 show the IoU-0.96 poses the gate throws
// away. Scored with rasterize_iou2d_v3rs at g_shiftECaptureStep (16 = same
// yardstick as SilCompare / Ctrl+I, so the panel IoU is comparable to method
// 4 -- and != runShiftE's full-mesh IoU on purpose). OFF = byte-identical.
inline bool g_shiftECaptureFound = false;
inline int  g_shiftECaptureStep  = 16;

// [QUAD-SIL] quadrant_mask: the Ctrl+G Quadrant Selector bitmask
// (caller passes g_activeQuadrantMask). Default QUAD_ALL keeps every
// legacy call site (SilComparePanel etc.) byte-identical.
inline void runShiftE(uint8_t quadrant_mask = LiverLeftRightLabel::QUAD_ALL) {
    std::cout << "\n=== 2D Silhouette BIPOP-CMA-ES (Shift+E) ===" << std::endl;

    if (!registrationHandle.useRegistration) {
        std::cerr << "[Shift+E] Run HemiAuto (O) first." << std::endl;
        return;
    }
    if (!g_boundaryDistMap.valid) {
        std::cerr << "[Shift+E] boundary map invalid" << std::endl;
        return;
    }

    auto organs = getOrganList();

    // ----------------------------------------------------------------
    // [QUAD-SIL] Build the quadrant-filtered triangle list and arm the
    // session override. Same selection rule as the V3RS Phase 2c filter
    // (Ctrl+Shift+G / Ctrl+I): a triangle is kept iff ALL 3 of its
    // vertices belong to the active quadrants per makeQuadrantSubsetIdx;
    // original triangle order is preserved. quadTris must outlive the
    // whole optimisation (the override holds a pointer to it), hence
    // function scope.
    //
    // Engagement conditions (all must hold, otherwise legacy full-mesh):
    //   - g_shiftE_quadrantAware toggle ON
    //   - quadrant_mask != QUAD_ALL  (ALL == full mesh by definition;
    //     skipping the override keeps the default path bit-identical)
    //   - region/LR labels valid and sized to the liver vertex count
    //     (the dispatch auto-computes them; this is a belt-and-braces
    //     fallback for callers that don't)
    // ----------------------------------------------------------------
    std::vector<GLuint> quadTris;
    std::vector<int>     quadPivotIdx;   // subset vertex ids for the SRT pivot
    bool quadFilterActive = false;
    if (g_shiftE_quadrantAware
        && quadrant_mask != LiverLeftRightLabel::QUAD_ALL
        && liverMesh3D)
    {
        const size_t nV = liverMesh3D->mVertices.size() / 3;
        const bool labelsOk =
            g_liverRegion.valid() && g_liverLR.valid() &&
            g_liverRegion.labels.size() == nV &&
            g_liverLR.labels.size()     == nV;
        if (!labelsOk) {
            std::cerr << "[Shift+E/quad] labels unavailable or size "
                         "mismatch; falling back to FULL mesh."
                      << std::endl;
        } else {
            const std::vector<int> quad_vert_ids =
                LiverLeftRightLabel::makeQuadrantSubsetIdx(
                    g_liverRegion.labels, g_liverLR.labels, quadrant_mask);
            std::vector<uint8_t> in_quad(nV, 0);
            for (int idx : quad_vert_ids) {
                if (idx >= 0 && (size_t)idx < nV) in_quad[(size_t)idx] = 1;
            }
            const auto& src_idx = liverMesh3D->mIndices;
            quadTris.reserve(src_idx.size());
            int n_tris_total = 0;
            int n_tris_kept  = 0;
            for (size_t t = 0; t + 2 < src_idx.size(); t += 3) {
                ++n_tris_total;
                const GLuint i0 = src_idx[t + 0];
                const GLuint i1 = src_idx[t + 1];
                const GLuint i2 = src_idx[t + 2];
                if (i0 >= nV || i1 >= nV || i2 >= nV) continue;
                if (in_quad[i0] && in_quad[i1] && in_quad[i2]) {
                    quadTris.push_back(i0);
                    quadTris.push_back(i1);
                    quadTris.push_back(i2);
                    ++n_tris_kept;
                }
            }
            if (quadTris.empty()) {
                std::cerr << "[Shift+E/quad] quadrant subset has no whole "
                             "triangles (Q="
                          << LiverLeftRightLabel::quadrantMaskString(
                                 quadrant_mask)
                          << "); falling back to FULL mesh." << std::endl;
            } else {
                CmaesRefine::g_silTriOverride = &quadTris;
                // [QUAD-SIL PIVOT] Rotate/scale about the subset's own
                // centroid (Ctrl+G parity). Use the makeQuadrantSubsetIdx
                // vertex set — the same subset Ctrl+G's S.centroid is built
                // from. applyIncrementalSRT recomputes the centroid from
                // these vertices at the CURRENT pose every call, so the
                // pivot tracks the mesh as CMA-ES moves it.
                quadPivotIdx = quad_vert_ids;
                CmaesRefine::g_silPivotVertIdx = &quadPivotIdx;
                // [QUAD-SIL BRAKE] Arm the mask-expansion brake (port of
                // V3RS outside_ratio). Without it the symmetric IoU lets the
                // optimiser "win" by inflating the subset to blanket the
                // target instead of aligning it. Default weight matches the
                // V3RS Ctrl+Shift+G default (g_ctrlgsLambdaOut = 0.5).
                CmaesRefine::g_silOutsideLambda = g_shiftE_outsideLambda;
                quadFilterActive = true;
                std::cout << "[Shift+E/quad] quad-filter tris: "
                          << n_tris_kept << " / " << n_tris_total
                          << "  (Q="
                          << LiverLeftRightLabel::quadrantMaskString(
                                 quadrant_mask)
                          << ")  — objective/gate/IoU + SRT pivot use this "
                             "subset; expansion brake lambda="
                          << g_shiftE_outsideLambda << std::endl;
            }
        }
    }

    // --- AR固定カメラ行列を構築（OrbitCamに依存しない） ---
    glm::mat4 silView = buildSilhouetteView();
    glm::mat4 silProj = buildSilhouetteProj();
    int silW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
    int silH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

    // IoU計測ラムダ（固定行列 + 固定解像度を使用）
    auto measureIoU = [&]() -> float {
        // グローバルを一時差し替え
        glm::mat4 sv = view;  glm::mat4 sp = projection;
        int sw = gWindowWidth; int sh = gWindowHeight;
        view = silView;  projection = silProj;
        gWindowWidth = silW;  gWindowHeight = silH;

        float fval = CmaesRefine::computeSilhouette2DObjectiveFast(
            liverMesh3D, view, projection, g_shiftE_RasterStep);

        view = sv;  projection = sp;
        gWindowWidth = sw;  gWindowHeight = sh;
        return 1.0f - fval;
    };

    float iou_before = measureIoU();
    std::cout << "[Shift+E] initial IoU=" << iou_before << std::endl;

    // 起点スナップショット
    std::vector<std::vector<GLfloat>> start_v(organs.size()), start_n(organs.size());
    for (size_t i = 0; i < organs.size(); i++)
        if (organs[i]) {
            start_v[i] = organs[i]->mVertices;
            start_n[i] = organs[i]->mNormals;
        }

    float best_iou = iou_before;
    auto  best_v = start_v;
    auto  best_n = start_n;

    const int N_STARTS = g_shiftE_NStarts;

    // Phase 1: シード固定 (旧: rng(20260425u) 固定値)
    const uint32_t outer_seed = g_trialSeed + 1000u + g_callIdx * 97u;
    const uint32_t cma_base   = g_trialSeed + 2000u + g_callIdx * 10u;
    std::cout << "[Seed] Shift+E outer=" << outer_seed
              << "  CMA-ES base=" << cma_base
              << "  (trial=" << g_trialSeed << ", callIdx=" << g_callIdx << ")"
              << std::endl;

    std::mt19937 rng(outer_seed);
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    for (int run = 0; run < N_STARTS; run++) {
        // 起点に巻き戻し
        for (size_t i = 0; i < organs.size(); i++)
            if (organs[i]) {
                organs[i]->mVertices = start_v[i];
                organs[i]->mNormals  = start_n[i];
                setUp(*organs[i]);
            }

        CmaesRefine::Params p;
        p.verbose                = true;
        p.log_every              = 100;
        p.save_debug_jpg         = false;
        p.use_silhouette_2d      = true;
        p.use_silhouette_2d_fast = true;        // fast raster path (always on)
        p.alpha_silhouette       = 1.0f;
        // [UI整理] pure IoU (default) -> drop the 3D blend term.
        p.alpha_3d               = g_shiftE_pureIoU ? 0.0f : 0.3f;
        p.silhouette_step        = g_shiftE_RasterStep;
        p.maxgen = 300;
        p.tolfun = 1e-4;
        // CMA-ES sampling range scaled by sceneDiag (was 1.0f)
        const float gtE = g_sceneDiag * 0.5f;
        p.tx_range = gtE; p.ty_range = gtE; p.tz_range = gtE;
        p.rx_range = 20.0f; p.ry_range = 20.0f; p.rz_range = 20.0f;
        p.scale_lo = 0.85f; p.scale_hi = 1.15f;
        // Phase 1: CMA-ES 内部 srand を固定
        p.rng_seed = cma_base + (uint32_t)run;

        // [Alt+P found-pose viz] capture this run's found (pre-revert) pose.
        if (g_shiftECaptureFound && g_boundaryDistMap.valid
                                 && run < SilOverlay::kNumRuns) {
            const int run_slot = run;
            p.on_best_candidate = [run_slot]() {
                if (!liverMesh3D || !g_boundaryDistMap.valid) return;
                const int cw = g_boundaryDistMap.width;
                const int ch = g_boundaryDistMap.height;
                const glm::mat4 cv = buildSilhouetteView();
                const glm::mat4 cp = buildSilhouetteProj();
                // [QUAD-SIL] Show what the optimiser actually rasterized:
                // when the session override is armed, capture the filtered
                // triangle list; otherwise the full mesh (legacy).
                std::vector<uint32_t> ctris;
                std::vector<uint32_t> full_for_viz;   // yellow overlay source
                if (CmaesRefine::g_silTriOverride
                    && !CmaesRefine::g_silTriOverride->empty()) {
                    ctris.assign(CmaesRefine::g_silTriOverride->begin(),
                                 CmaesRefine::g_silTriOverride->end());
                    // Full mesh so buildComposite can paint the discarded
                    // (non-subset) quadrants yellow in the Source panel.
                    full_for_viz.assign(liverMesh3D->mIndices.begin(),
                                        liverMesh3D->mIndices.end());
                } else {
                    ctris.assign(liverMesh3D->mIndices.begin(),
                                 liverMesh3D->mIndices.end());
                    // No subset → no yellow overlay (leave full_for_viz empty;
                    // captureImpl no-ops when full==filtered size).
                }
                const float iou = SilOverlay::capture(
                    SilOverlay::g_silOverlay, run_slot,
                    liverMesh3D, ctris, cv, cp,
                    g_boundaryDistMap.data, cw, ch,
                    g_shiftECaptureStep, /*scale_value=*/1.0f,
                    /*instrument_dist_map=*/nullptr,
                    /*instrument_thresh_px=*/0.0f,
                    /*rim_sil_max_px=*/0.0f,
                    /*is_rim_anatomic_per_vertex=*/nullptr,
                    /*full_indices (yellow discarded overlay)=*/&full_for_viz);
                std::cout << "[Shift+E/found] Run " << (run_slot + 1)
                          << " found-pose -> F9 Run " << (run_slot + 1)
                          << "  (squash IoU=" << iou
                          << ", capture tris=" << (ctris.size() / 3)
                          << (CmaesRefine::g_silTriOverride
                              && !CmaesRefine::g_silTriOverride->empty()
                                  ? ", QUAD" : ", full")
                          << ")" << std::endl;
            };
        }

        float tx=0,ty=0,tz=0, rx=0,ry=0,rz=0, sc=1.0f;
        std::string regime;
        if (run == 0) {
            p.sigma0 = 0.2;  regime = "Baseline";
        } else if (run <= 2) {
            p.sigma0 = 0.05f + d01(rng)*0.25f;
            const float lt = RegRatios::cmaLocalT();
            tx = (d01(rng)*2-1)*lt; ty = (d01(rng)*2-1)*lt;
            tz = (d01(rng)*2-1)*lt;
            rx = (d01(rng)*2-1)*10.f; ry = (d01(rng)*2-1)*10.f;
            rz = (d01(rng)*2-1)*10.f;
            sc = 0.95f + d01(rng)*0.10f;
            regime = "Local";
        } else {
            p.sigma0 = 0.30f + d01(rng)*0.50f;
            const float gt = RegRatios::cmaGlobalT();
            tx = (d01(rng)*2-1)*gt; ty = (d01(rng)*2-1)*gt;
            tz = (d01(rng)*2-1)*gt;
            rx = (d01(rng)*2-1)*20.f; ry = (d01(rng)*2-1)*20.f;
            rz = (d01(rng)*2-1)*20.f;
            sc = 0.85f + d01(rng)*0.30f;
            regime = "Global";
        }

        if (run > 0) {
            CmaesRefine::applyIncrementalSRT(organs, tx,ty,tz, rx,ry,rz, sc);
            for (auto* m : organs) if (m) setUp(*m);
        }

        std::cout << "[Shift+E] Run " << (run+1) << "/" << N_STARTS
                  << "  " << regime << "  sigma0=" << p.sigma0 << std::endl;

        // --- CMA-ES中はグローバル行列をK固定カメラに差し替え ---
        glm::mat4 sv = view;  glm::mat4 sp = projection;
        int sw = gWindowWidth; int sh = gWindowHeight;
        view = silView;  projection = silProj;
        gWindowWidth = silW;  gWindowHeight = silH;

        CmaesRefine::run(organs, screenMesh,
                         gGridWidth, gGridHeight(),
                         RegRatios::zThresh(), p);

        // グローバル復元
        view = sv;  projection = sp;
        gWindowWidth = sw;  gWindowHeight = sh;

        float iou_run = measureIoU();
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "[Shift+E] Run " << (run+1)
                  << "  IoU=" << iou_run
                  << (iou_run > best_iou + 1e-4f ? " [+]" : " [-]")
                  << std::endl;

        if (iou_run > best_iou + 1e-4f) {
            best_iou = iou_run;
            for (size_t i = 0; i < organs.size(); i++)
                if (organs[i]) {
                    best_v[i] = organs[i]->mVertices;
                    best_n[i] = organs[i]->mNormals;
                }
        }
    }

    // ベスト姿勢を適用
    for (size_t i = 0; i < organs.size(); i++)
        if (organs[i]) {
            organs[i]->mVertices = best_v[i];
            organs[i]->mNormals  = best_n[i];
            setUp(*organs[i]);
        }

    // [QUAD-SIL] Disarm the override BEFORE the final computeUnifiedMetrics
    // so the app-wide reported metrics (registrationHandle.compIoU2D, the
    // Hausdorff readout, and the PoseLibrary Layer-4 IoU series) stay on the
    // full-mesh yardstick — the same convention Ctrl+I follows. Intermediate
    // registrationHandle values written during the loop (engine-internal
    // computeUnifiedMetrics calls saw the subset) are overwritten here.
    if (quadFilterActive) {
        CmaesRefine::g_silTriOverride  = nullptr;
        CmaesRefine::g_silPivotVertIdx = nullptr;   // [QUAD-SIL PIVOT] disarm
        CmaesRefine::g_silOutsideLambda = 0.0f;     // [QUAD-SIL BRAKE] disarm
        std::cout << "[Shift+E/quad] override disarmed; final metrics below "
                     "are FULL-mesh (session IoU printout above was subset)"
                  << std::endl;
    }

    computeUnifiedMetrics();
    g_metricsValid = true;

    // [QUAD-SIL ACCEPT] In quadrant mode, publish the occluded-IoU /
    // RIM / containment diagnostics at the final pose (exactly what
    // Ctrl+G / Ctrl+I / AutoQCR publish via this same helper). This sets
    // g_lastSilOccludedIoU2D so the following poseSaveToLibrary gate can
    // judge on the occluded IoU — the SAME criterion Ctrl+I uses — instead
    // of the full-mesh IoU2D, which a subset alignment necessarily lowers.
    // Only in quadrant mode: the legacy full-mesh Alt+P path (Q:ALL /
    // toggle OFF) leaves this untouched and keeps its byte-identical
    // compIoU2D-based gate.
    if (quadFilterActive) {
        publishCtrlGStyleDiagnostics();
        std::cout << "[Shift+E/quad] published occluded-IoU diagnostics "
                     "(IoU_occ=" << g_lastSilOccludedIoU2D
                  << ") for the accept gate — Ctrl+I parity" << std::endl;
    }

    // [Alt+P found-pose viz] capture the applied (kept) pose into Final.
    if (g_shiftECaptureFound && g_boundaryDistMap.valid && liverMesh3D) {
        const glm::mat4 cv = buildSilhouetteView();
        const glm::mat4 cp = buildSilhouetteProj();
        // [QUAD-SIL] Final capture mirrors what the optimiser saw: the
        // filtered list when the quadrant filter was active this session
        // (quadTris is still alive — function scope), else the full mesh.
        std::vector<uint32_t> ctris;
        std::vector<uint32_t> full_for_viz;   // yellow discarded-quadrant overlay
        if (quadFilterActive) {
            ctris.assign(quadTris.begin(), quadTris.end());
            full_for_viz.assign(liverMesh3D->mIndices.begin(),
                                liverMesh3D->mIndices.end());
        } else {
            ctris.assign(liverMesh3D->mIndices.begin(),
                         liverMesh3D->mIndices.end());
        }
        SilOverlay::captureFinal(
            SilOverlay::g_silOverlay, /*best_run_idx=*/-1,
            liverMesh3D, ctris, cv, cp,
            g_boundaryDistMap.data, g_boundaryDistMap.width,
            g_boundaryDistMap.height, g_shiftECaptureStep, /*scale_value=*/1.0f,
            /*instrument_dist_map=*/nullptr,
            /*instrument_thresh_px=*/0.0f,
            /*rim_sil_max_px=*/0.0f,
            /*is_rim_anatomic_per_vertex=*/nullptr,
            /*full_indices (yellow overlay)=*/&full_for_viz);
        SilOverlay::g_silOverlay.showWindow = true;
        std::cout << "[Shift+E/found] applied (kept) pose -> F9 Final"
                  << "  (capture tris=" << (ctris.size() / 3)
                  << (quadFilterActive ? ", QUAD-FILTERED" : ", full-mesh")
                  << ", step=" << g_shiftECaptureStep << ")" << std::endl;
    }

    std::cout << std::defaultfloat << std::setprecision(6);
    float iou_delta = best_iou - iou_before;
    std::cout << "[Shift+E] IoU: " << iou_before << " -> " << best_iou
              << " (delta=" << iou_delta << ")"
              << (iou_delta > 0.001f ? " [IMPROVED]" : " [NO CHANGE]")
              << std::endl;

    g_callIdx++;  // Phase 1: 末尾でインクリメント
}

// =====================================================================
//  Shift+I : Silhouette-fixed dilation 1-D RMSE refine ("stage 2")
// ---------------------------------------------------------------------
//  Runs AFTER Alt+P (runShiftE, 2D-IoU max) or Ctrl+I (V3I, 1-IoU2D)
//  have aligned the silhouette. Those stages constrain everything the
//  SAM2 mask can see — in-plane translation, in-plane rotation, apparent
//  size — but they are blind to ONE direction: uniform dilation about
//  the camera optical center C, i.e. p -> C + k(p - C). Under a pinhole
//  projection that map moves every vertex along its own line of sight,
//  so the projected pixel (hence the silhouette and the 2D-IoU) is
//  EXACTLY invariant. k trades depth against scale. Alt+P / Ctrl+I
//  therefore leave k undetermined (whatever value CMA-ES happened to
//  stop at on that invariant ray); this step pins it down by minimising
//  the 3D RMSE against the DA3 depth cloud.
//
//  Because the silhouette is held exactly fixed, the search is strictly
//  1-D in k — not a 2-DOF (Z, scale) search (those would each move the
//  silhouette and reduce to a soft IoU penalty, i.e. V3RS). A 1-D line
//  search (coarse log bracket + golden section) is enough; no CMA-ES.
//
//  Fast eval (the key trick): the dilation maps source s -> C + k(s-C).
//  For a fixed target q the squared distance obeys the identity
//        | q - (C + k(s-C)) |^2  ==  k^2 * | (C + (q-C)/k) - s |^2
//  so instead of dilating the source (which would force a KDTree rebuild
//  per k) we INVERSE-dilate the query, q' = C + (q-C)/k, search the
//  k=1 source tree once, and scale the squared distance by k^2. The NN
//  index is preserved (scaling all distances by k^2 doesn't change the
//  argmin), and the physical max_dist_sq gate is on the TRUE distance,
//  so it is k-invariant:  k^2 * dsq' < max_dist_sq  <=>  dsq' < gate/k^2.
//
//  Source / target / gate / direction are IDENTICAL to computeUnified-
//  Metrics (and Ctrl+G's F-phase): source = liver full mesh, target =
//  extractFrontFacePoints (DA3 cloud), gate = (sceneDiag/7.36)^2,
//  direction = tgt -> src (KDTree on src). So eval(1.0) equals the
//  current registrationHandle.compRmse and the optimisation moves the
//  same number the rest of the app reports.
//
//  Save criterion MUST be RMSE: IoU is invariant by construction, so an
//  IoU gate would read zero change and reject every result (handled in
//  the main.cpp Shift+I dispatch).
//
//  Operating flow:  O -> Alt+P or Ctrl+I (IoU) -> Shift+I (this).
// =====================================================================
inline void runDilationRmseRefine() {
    std::cout << "\n=== Silhouette-fixed Dilation RMSE refine (Shift+I) ==="
              << std::endl;

    // ----- Preconditions (mirror runShiftE) --------------------------
    if (!registrationHandle.useRegistration) {
        std::cerr << "[Shift+I] Run HemiAuto (O) + Alt+P / Ctrl+I first."
                  << std::endl;
        return;
    }
    if (!liverMesh3D || liverMesh3D->mVertices.size() < 9) {
        std::cerr << "[Shift+I] liver mesh empty; aborting." << std::endl;
        return;
    }
    if (!screenMesh) {
        std::cerr << "[Shift+I] no screen mesh; aborting." << std::endl;
        return;
    }

    auto organs = getOrganList();

    // ----- Optical center C (camera center in world) -----------------
    // buildSilhouetteView() == lookAt(eye=(0,0,0), ...), so C == origin.
    // We still derive C from the inverse view so the invariance assert
    // below also validates that the silhouette camera really is centered
    // where we assume (a non-origin eye would surface here as IoU drift).
    const glm::mat4 silView = buildSilhouetteView();
    const glm::vec3 C = glm::vec3(glm::inverse(silView)[3]);

    // ----- IoU measure (same path as runShiftE.measureIoU) -----------
    // computeSilhouette2DObjectiveFast reads the GLOBAL view/projection/
    // gWindowWidth/Height, so they must be swapped to the AR fixed camera
    // for the call and restored afterwards.
    const glm::mat4 silProj = buildSilhouetteProj();
    const int silW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
    const int silH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;
    auto measureIoU = [&]() -> float {
        glm::mat4 sv = view;          glm::mat4 sp = projection;
        int       sw = gWindowWidth;  int       sh = gWindowHeight;
        view = silView;       projection = silProj;
        gWindowWidth = silW;  gWindowHeight = silH;
        bool wasQuiet = g_quietMetrics; g_quietMetrics = true;
        float fval = CmaesRefine::computeSilhouette2DObjectiveFast(
            liverMesh3D, view, projection, /*step=*/8);
        g_quietMetrics = wasQuiet;
        view = sv;             projection = sp;
        gWindowWidth = sw;     gWindowHeight = sh;
        return 1.0f - fval;   // fval is the cost (1 - IoU2D)
    };
    const float iou_before = measureIoU();

    // ----- Target cloud (DA3), identical to computeUnifiedMetrics -----
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = std::max(0.001f, RegRatios::zThresh());
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[Shift+I] empty target cloud; aborting." << std::endl;
        return;
    }
    const std::vector<glm::vec3>& tgt_points = targetCloud->points;

    // ----- Source cloud: liver full mesh (identical to compRmse) ------
    // NanoflannAdaptor holds a REFERENCE to src_pts, so src_pts must
    // outlive `tree`. Tree is built ONCE at k=1; eval() inverse-dilates
    // the query rather than rebuilding (see header identity).
    std::vector<glm::vec3> src_pts;
    {
        const auto& V = liverMesh3D->mVertices;
        src_pts.reserve(V.size() / 3);
        for (size_t i = 0; i + 2 < V.size(); i += 3)
            src_pts.emplace_back(V[i], V[i + 1], V[i + 2]);
    }
    if (src_pts.empty()) {
        std::cerr << "[Shift+I] source cloud empty; aborting." << std::endl;
        return;
    }
    Reg3DCustom::NanoflannAdaptor src_adaptor(src_pts);
    auto tree = Reg3DCustom::buildKDTree(src_adaptor);

    const float max_dist_sq = RegRatios::maxDistSq();  // == (sceneDiag/7.36)^2

    // ----- Fast eval: true RMSE of dilation-by-k, no tree rebuild -----
    auto eval = [&](float k) -> float {
        const float invk = 1.0f / k;
        const float gate = max_dist_sq * invk * invk;   // gate / k^2
        float sumSq = 0.0f;
        int   cnt   = 0;
        for (size_t i = 0; i < tgt_points.size(); ++i) {
            const glm::vec3 qp = C + (tgt_points[i] - C) * invk;
            size_t nn;  float dsq;
            if (Reg3DCustom::searchKNN1(*tree, qp, nn, dsq) && dsq < gate) {
                sumSq += dsq;   // dsq' ; true sq-dist = k^2 * dsq'
                ++cnt;
            }
        }
        // true RMSE = sqrt( mean( k^2 * dsq' ) ) = k * sqrt( mean dsq' ).
        return cnt ? k * std::sqrt(sumSq / (float)cnt) : 9.9f;
    };

    const float rmse_before = eval(1.0f);

    // ----- 1-D search: coarse log bracket, then golden section --------
    constexpr float kLo = 0.5f, kHi = 2.0f;
    constexpr int   kCoarse = 11;
    int eval_count = 0;
    auto evalC = [&](float k) -> float { ++eval_count; return eval(k); };

    // Coarse scan (log-spaced) to bracket the minimum without assuming
    // unimodality / sticking to an endpoint.
    float gridK[kCoarse], gridF[kCoarse];
    int   gridMin = 0;
    const float lnLo = std::log(kLo), lnHi = std::log(kHi);
    for (int i = 0; i < kCoarse; ++i) {
        const float t = (float)i / (float)(kCoarse - 1);
        gridK[i] = std::exp(lnLo + t * (lnHi - lnLo));
        gridF[i] = evalC(gridK[i]);
        if (gridF[i] < gridF[gridMin]) gridMin = i;
    }

    // Golden section in u = log(k) on the bracket around the coarse min.
    const int   lo_i = (gridMin > 0)            ? gridMin - 1 : 0;
    const int   hi_i = (gridMin < kCoarse - 1)  ? gridMin + 1 : kCoarse - 1;
    double ulo = std::log(gridK[lo_i]);
    double uhi = std::log(gridK[hi_i]);
    const double gr = 0.6180339887498949;            // (sqrt(5)-1)/2
    double u1 = uhi - gr * (uhi - ulo);
    double u2 = ulo + gr * (uhi - ulo);
    float  f1 = evalC((float)std::exp(u1));
    float  f2 = evalC((float)std::exp(u2));
    const double uTol = std::log(1.0 + 0.005);        // ~0.5% in k
    int gs_iter = 0;
    while ((uhi - ulo) > uTol && gs_iter < 40) {
        if (f1 < f2) { uhi = u2; u2 = u1; f2 = f1;
                       u1 = uhi - gr * (uhi - ulo); f1 = evalC((float)std::exp(u1)); }
        else         { ulo = u1; u1 = u2; f1 = f2;
                       u2 = ulo + gr * (uhi - ulo); f2 = evalC((float)std::exp(u2)); }
        ++gs_iter;
    }
    const float kMid = (float)std::exp(0.5 * (ulo + uhi));
    const float fMid = evalC(kMid);

    // Pick the best candidate; seed with k=1 (no-op) so the refine can
    // NEVER worsen the RMSE — at worst it leaves the pose untouched.
    float kStar = 1.0f, rmse_after = rmse_before;
    const float candK[] = { (float)std::exp(u1), (float)std::exp(u2),
                            kMid, gridK[gridMin] };
    const float candF[] = { f1, f2, fMid, gridF[gridMin] };
    for (int i = 0; i < 4; ++i)
        if (candF[i] < rmse_after) { rmse_after = candF[i]; kStar = candK[i]; }

    // ----- Apply dilation s -> C + k*(s-C) to all organs --------------
    // Uniform scale about C: normal DIRECTIONS are invariant, so only
    // vertices change; setUp() refreshes the GL buffers. Skip the no-op
    // at k==1 to avoid any reassociation rounding when C is non-origin.
    if (std::fabs(kStar - 1.0f) > 1e-7f) {
        for (auto* m : organs) {
            if (!m) continue;
            auto& V = m->mVertices;
            for (size_t i = 0; i + 2 < V.size(); i += 3) {
                V[i]     = C.x + kStar * (V[i]     - C.x);
                V[i + 1] = C.y + kStar * (V[i + 1] - C.y);
                V[i + 2] = C.z + kStar * (V[i + 2] - C.z);
            }
            setUp(*m);
        }
    }

    // ----- Metrics (overwrites compRmse with the post-dilation value) -
    computeUnifiedMetrics();
    g_metricsValid = true;
    const float iou_after = measureIoU();

    // ----- implied_scale diagnostic (DA3-frame liver vs CT-mm) --------
    // = current_liver_diag / g_originalLiverDiagMm (reciprocal of Shift+M
    // SCALE_RESTORE). ~1 => DA3 metric depth and the CT model agree on
    // absolute size; far from 1 => DA3 absolute-scale error, which rides
    // on this very (projection-invariant) dilation ray and therefore is
    // never visible in the overlay. This is a read-out of DA3 metric
    // quality, NOT a correction applied during registration.
    float cur_diag = 0.0f;
    {
        const auto& V = liverMesh3D->mVertices;
        glm::vec3 mn(V[0], V[1], V[2]), mx = mn;
        for (size_t i = 0; i + 2 < V.size(); i += 3) {
            glm::vec3 v(V[i], V[i + 1], V[i + 2]);
            mn = glm::min(mn, v);
            mx = glm::max(mx, v);
        }
        cur_diag = glm::length(mx - mn);
    }
    const float implied_scale =
        (g_hasOriginalDiags && g_originalLiverDiagMm > 1e-6f)
            ? cur_diag / g_originalLiverDiagMm : -1.0f;

    // ----- Log (Ctrl+G-style, plus the invariance assert) -------------
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "[Shift+I] k*=" << kStar
              << "  (search [" << kLo << "," << kHi << "], evals=" << eval_count
              << ", coarse_min_k=" << gridK[gridMin] << ")" << std::endl;

    const float rmse_delta = rmse_before - rmse_after;
    std::cout << "[Shift+I] RMSE: " << rmse_before << " -> " << rmse_after
              << " (delta=" << rmse_delta << ")"
              << (rmse_delta > 1e-6f ? "  [IMPROVED]" : "  [NO CHANGE]")
              << std::endl;

    if (implied_scale > 0.0f)
        std::cout << "[Shift+I] implied_scale (DA3/CT-mm) = " << implied_scale
                  << "  (current_liver_diag=" << cur_diag
                  << ", g_originalLiverDiagMm=" << g_originalLiverDiagMm << ")"
                  << std::endl;
    else
        std::cout << "[Shift+I] implied_scale: N/A "
                     "(CT-mm reference not captured at startup)" << std::endl;

    // Invariance assert: dilation about the optical center must leave the
    // rasterized silhouette unchanged. A non-trivial delta means C / the
    // silhouette view is off — a correctness bug, not a tuning knob.
    const float iou_delta = std::fabs(iou_after - iou_before);
    std::cout << "[Shift+I] IoU(invariance check): " << iou_before
              << " -> " << iou_after << "  |delta|=" << iou_delta
              << (iou_delta > 1e-3f ? "  [WARN: silhouette moved -- check C/view]"
                                    : "  [OK: silhouette fixed]")
              << std::endl;

    g_callIdx++;  // match V1 / V3 / Shift+E: increment at the end
}


// =====================================================================
// Phase 7b Step 1 helper — Plain W key wrapper
// =====================================================================
//
// populateDebugSourceRimChain
//   Ctrl+G が実際に使う source RIM subset と完全に同じ vertex 集合に対
//   して RimShape::walkRimChain を実行し、g_debugSourceRimChain に格納。
//   Plain W 押下のたびに現在のパネル設定 (g_activeQuadrantMask,
//   g_ctrlgUseArVisFilter, g_ctrlgUseCaudalOnly, g_ctrlgArvisCaudalCombine)
//   を動的に読み、Ctrl+G の line 5700-5738 compose と byte-equivalent な
//   filtering を行う。
//
// Pipeline (Ctrl+G の line 5700-5738 と同等):
//   1. quadAllowed = makeQuadrantSubsetIdx(region_labels, lr_labels, mask)
//   2. base = quadAllowed ∩ {labels[i] == RIM}
//   3. AR-vis filter ON → BVH raycast from (0,0,-0.2*diag), 可視のみ
//      (rim CC rescue は Step 1 では省略、~10ms 軽量化のため)
//   4. caudal-only ON → labels[i] == CAUDAL のみ
//   5. (a_on && c_on) なら g_ctrlgArvisCaudalCombine (0=AND, 1=OR) で合成
//
// Failure modes (degradation):
//   - g_liverRegion / g_liverLR 未計算 → 失敗 (呼び出し側で auto-trigger)
//   - caudal-only ON だが g_liverCC 未計算 → caudal フィルタ無効化、警告
//   - quadrant が QUAD_ALL なら全頂点 (= 通常の RIM 全部に縮約)
//
// 描画側との約束:
//   g_debugSourceRimChain は「頂点 index」を保持。描画ループでは
//   liverMesh3D->mVertices[idx*3..idx*3+2] から現在位置を fetch するの
//   で、ICP / Live tracking で organ が動いてもマーカーが追従する。
//
inline bool populateDebugSourceRimChain()
{
    g_debugSourceRimChain.clear();

    // ---- Preconditions ----------------------------------------------
    if (!liverMesh3D) {
        std::cout << "[W/RimChain] no scene loaded (liverMesh3D == null)"
                  << std::endl;
        return false;
    }
    if (!g_liverRegion.valid()) {
        std::cout << "[W/RimChain] g_liverRegion not yet computed"
                  << " — caller should recomputeLiverRegion() first"
                  << std::endl;
        return false;
    }
    if (!g_liverLR.valid()) {
        std::cout << "[W/RimChain] g_liverLR not yet computed"
                  << " — caller should recomputeLiverLR() first"
                  << std::endl;
        return false;
    }
    const size_t N = g_liverRegion.labels.size();
    if (N == 0 || g_liverLR.labels.size() != N) {
        std::cout << "[W/RimChain] label size mismatch (region="
                  << N << " LR=" << g_liverLR.labels.size() << ")"
                  << std::endl;
        return false;
    }

    // ---- Step 1: quadrant subset (uses current panel state) ---------
    auto quadAllowed = LiverLeftRightLabel::makeQuadrantSubsetIdx(
        g_liverRegion.labels, g_liverLR.labels, g_activeQuadrantMask);

    // ---- Step 2: AR-vis raycast (optional, ~16ms one-shot) ----------
    //   Ctrl+G の Phase C2a/Stage(a) と同じ raycastVisibilityBVH。Stage(b)
    //   の rim CC rescue は Step 1 では省略 (BFS+normal_gate で +~10ms、
    //   かつ rescue ロジックは ParamsV3R::rim_adj 構築が必要で複雑)。
    //   実機で「rim 表側の cup 状凹みで raycast が rim を自己遮蔽する」
    //   ケースが目立つようなら Step 4 で rescue 追加。
    std::vector<uint8_t> arvis;
    const bool a_on = g_ctrlgUseArVisFilter;
    if (a_on) {
        Reg3D::BVHTree bvh;
        bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
        const float diag = (g_liverRegion.bbox_diag > 0.0f)
                              ? g_liverRegion.bbox_diag : g_sceneDiag;
        const glm::vec3 ar_cam_pos(0.0f, 0.0f, -0.2f * diag);
        LiverRegionLabel::raycastVisibilityBVH(
            *liverMesh3D, bvh, ar_cam_pos, diag, arvis);
    }

    // ---- Step 3: caudal-only (optional, degradation-safe) -----------
    bool c_on = g_ctrlgUseCaudalOnly;
    if (c_on) {
        if (!g_liverCC.valid() || g_liverCC.labels.size() != N) {
            std::cout << "[W/RimChain] WARN caudal-only requested but"
                      << " g_liverCC missing/mismatched (cc.valid="
                      << (g_liverCC.valid() ? "Y" : "N")
                      << " size=" << g_liverCC.labels.size()
                      << ") — disabling caudal filter for this call"
                      << std::endl;
            c_on = false;
        }
    }

    // ---- Step 4: compose AND/OR (mirrors Ctrl+G line 5711-5736) -----
    const uint8_t cmode = g_ctrlgArvisCaudalCombine;  // 0=AND, 1=OR
    std::vector<uint8_t> filtered_labels(N, LiverRegionLabel::ANTERIOR_CORE);
    int n_after_quad = 0, n_after_filters = 0;
    for (int idx : quadAllowed) {
        if (idx < 0 || (size_t)idx >= N) continue;
        if (g_liverRegion.labels[idx] != LiverRegionLabel::RIM) continue;
        n_after_quad++;
        const bool a = (!a_on) ||
                       ((size_t)idx < arvis.size() && arvis[idx]);
        const bool c = (!c_on) ||
                       (g_liverCC.labels[idx] == LiverCranioCaudalLabel::CAUDAL);
        bool pass;
        if (a_on && c_on) {
            pass = (cmode == 0) ? (a && c) : (a || c);
        } else {
            pass = (a && c);  // shortcut: empty side is unconditionally true
        }
        if (!pass) continue;
        filtered_labels[idx] = LiverRegionLabel::RIM;
        n_after_filters++;
    }

    std::cout << "[W/RimChain] filter compose: a_on=" << (a_on ? "Y" : "N")
              << " c_on=" << (c_on ? "Y" : "N")
              << " cmode=" << (cmode == 0 ? "AND" : "OR")
              << "  quadrant=0x" << std::hex << (int)g_activeQuadrantMask << std::dec
              << "  rim_in_quad=" << n_after_quad
              << "  rim_after_all=" << n_after_filters
              << std::endl;

    if (n_after_filters == 0) {
        std::cout << "[W/RimChain] no RIM verts pass the filter -- abort"
                  << std::endl;
        return false;
    }

    // ---- Step 5: order the 254 verts by PCA-major-plane angle ------
    //   Why not walkRimChain: the filtered rim patch has rim-degree>=4
    //   dominant, so a degree-2 graph walk gets stuck at junctions and
    //   returns only ~20% of the verts. PCA-plane angle sort instead
    //   uses every vertex, gives spatial coherence (adjacent idx =
    //   adjacent in space) sufficient for tangent estimation in Step 3,
    //   and is O(n log n) ~0.5 ms for 254 verts.
    //   Cache PCA outputs (centroid / patch normal / principal axis /
    //   planarity) into globals for Step 3's solveTwoAxisAlignment to
    //   reuse without recomputing.
    glm::vec3 centroid, major_normal, principal_axis;
    double    planarity = 1.0;
    int       n_rim_total = 0;
    const bool ok = RimShape::sortRimChainByAngle(
        *liverMesh3D, filtered_labels,
        g_debugSourceRimChain,
        n_rim_total, centroid, major_normal, principal_axis, planarity,
        &std::cout);
    if (!ok) {
        g_debugSourceRimChain.clear();
        std::cout << "[W/RimChain] sortRimChainByAngle failed" << std::endl;
        return false;
    }
    g_debugSourceRimCentroid      = centroid;
    g_debugSourceRimMajorNormal   = major_normal;
    g_debugSourceRimPrincipalAxis = principal_axis;
    g_debugSourceRimPlanarity     = planarity;
    return true;
}


// =====================================================================
// Phase 7c (REDGE/稜線) — source ridge populate
// =====================================================================
//   populateDebugSourceRimChain の双子。RIM 帯のかわりに「ドームの
//   オクルーディング輪郭」を集める。
//     条件: 境界(silhouette) ∩ 非RIM ∩ 非後面
//       - 非RIM・非後面  ⇔ label == ANTERIOR_CORE (3ラベルなので等価)
//       - 境界(silhouette) ⇔ |cos(n, -view)| < g_ctrlgRidgeCosBand
//   quadrant / caudal フィルタは rim chain と同じ (Q:* 選択を尊重)。
//   AR-vis raycast は使わない (稜線は grazing なので raw raycast が自己
//   遮蔽で落とす — RIM が rescue を要したのと同じ。grazing が可視判定代用)。
//   view依存=動的 → 押下時点の現在ポーズで一度計算して固定。ポーズを
//   動かしたら再 toggle でリフレッシュ (rim chain と同じ契約)。
//
inline bool populateDebugSourceRidge()
{
    g_debugSourceRidge.clear();

    if (!liverMesh3D) {
        std::cout << "[Ridge/src] no scene loaded (liverMesh3D == null)" << std::endl;
        return false;
    }
    if (!g_liverRegion.valid()) {
        std::cout << "[Ridge/src] g_liverRegion not yet computed" << std::endl;
        return false;
    }
    if (!g_liverLR.valid()) {
        std::cout << "[Ridge/src] g_liverLR not yet computed" << std::endl;
        return false;
    }
    const size_t N = g_liverRegion.labels.size();
    if (N == 0 || g_liverLR.labels.size() != N) {
        std::cout << "[Ridge/src] label size mismatch" << std::endl;
        return false;
    }

    // normals 必須 (grazing テスト)。無ければ面から計算 (AR-vis と同じ)。
    if (liverMesh3D->mNormals.size() != liverMesh3D->mVertices.size()) {
        std::cout << "[Ridge/src] normals missing — computing from faces..." << std::endl;
        Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
    }

    // quadrant subset (rim chain と同一)
    auto quadAllowed = LiverLeftRightLabel::makeQuadrantSubsetIdx(
        g_liverRegion.labels, g_liverLR.labels, g_activeQuadrantMask);

    // caudal-only (rim chain と同一、degradation-safe)
    bool c_on = g_ctrlgUseCaudalOnly;
    if (c_on && (!g_liverCC.valid() || g_liverCC.labels.size() != N)) {
        std::cout << "[Ridge/src] WARN caudal-only requested but g_liverCC"
                  << " missing — disabling for this call" << std::endl;
        c_on = false;
    }

    // AR camera (rim chain / Ctrl+G と同一)
    const float diag = (g_liverRegion.bbox_diag > 0.0f)
                          ? g_liverRegion.bbox_diag : g_sceneDiag;
    const glm::vec3 ar_cam_pos(0.0f, 0.0f, -0.2f * diag);
    const float band = std::max(0.02f, g_ctrlgRidgeCosBand);

    const auto& V  = liverMesh3D->mVertices;
    const auto& Nr = liverMesh3D->mNormals;

    int n_ant_in_quad = 0;
    for (int idx : quadAllowed) {
        if (idx < 0 || (size_t)idx >= N) continue;
        if (g_liverRegion.labels[idx] != LiverRegionLabel::ANTERIOR_CORE) continue;
        if (c_on && g_liverCC.labels[idx] != LiverCranioCaudalLabel::CAUDAL) continue;
        n_ant_in_quad++;

        const glm::vec3 p(V[idx*3], V[idx*3+1], V[idx*3+2]);
        const glm::vec3 n(Nr[idx*3], Nr[idx*3+1], Nr[idx*3+2]);
        const glm::vec3 viewv = p - ar_cam_pos;
        const float lv = glm::length(viewv);
        const float ln = glm::length(n);
        if (lv < 1e-6f || ln < 1e-6f) continue;
        const float cos_nv = glm::dot(n, -viewv) / (ln * lv);
        if (std::fabs(cos_nv) < band) {
            g_debugSourceRidge.push_back(idx);
        }
    }

    std::cout << "[Ridge/src] quadrant=0x" << std::hex
              << (int)g_activeQuadrantMask << std::dec
              << "  caudal=" << (c_on ? "Y" : "N")
              << "  ANTERIOR_in_quad=" << n_ant_in_quad
              << "  ridge(|cos|<" << band << ")=" << g_debugSourceRidge.size()
              << std::endl;

    if (g_debugSourceRidge.empty()) {
        std::cout << "[Ridge/src] no ridge verts — widen the cos band" << std::endl;
        return false;
    }
    return true;
}


// =====================================================================
// Phase 7b Step 2 helper — Shift+W key wrapper
// =====================================================================
//
// populateDebugTargetBoundary
//   target side の "rim band" 3D 点群を抽出して
//   g_debugTargetBoundaryPoints に格納。Ctrl+G の line 5740-5750 の
//   target side build と等価:
//       targetCloud->boundaryDist[i] < g_ctrlgRimTgtThreshPx (default 12px)
//     AND targetCloud->instrumentDist[i] >= g_instrumentPxThresh (default 20px)
//
// なぜ詳細設計 §1B の 2D Moore-tracing を使わないか:
//   既存 targetCloud->boundaryDist は depth grid 上の per-vertex distance
//   なので、distance < threshold で filter するだけで 3D rim points が
//   取れる。tracing は "instrument で切れた multi-segment 検出" に強い
//   利点があるが、Step 3 の chamfer cost は順序関係なし、Step 3 の
//   tangent も必要に応じて source 同様 PCA 角度 sort で代用可能。
//   2D tracing は MVP では over-engineering と判断、必要時に Step 3 で
//   追加。
//
// 計算量: extractFrontFacePoints ~100ms (Ctrl+G と同じコスト)、filter
//   loop ~1ms。Shift+W 押下のたびに毎回実行 (target は static なのに
//   毎回再実行するのは無駄だが、source 側と対称的に「現在の Ctrl+G
//   設定を反映」させたい — g_ctrlgRimTgtThreshPx や g_instrumentPxThresh
//   をスライダで変えたら即反映される) という意図。
//
inline bool populateDebugTargetBoundary()
{
    g_debugTargetBoundaryPoints.clear();

    // ---- Preconditions ---------------------------------------------
    if (!screenMesh) {
        std::cout << "[W/TgtBound] no scene loaded (screenMesh == null)"
                  << std::endl;
        return false;
    }

    // ---- Extract target cloud (same pattern as Ctrl+G) -------------
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = std::max(0.001f, RegRatios::zThresh());
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->empty()) {
        std::cout << "[W/TgtBound] empty target cloud" << std::endl;
        return false;
    }
    const auto& tgt_points = targetCloud->points;

    if (!targetCloud->hasBoundaryDist() ||
        targetCloud->boundaryDist.size() != tgt_points.size())
    {
        std::cout << "[W/TgtBound] targetCloud missing/mismatched"
                  << " boundaryDist — abort" << std::endl;
        return false;
    }

    const bool useInst = targetCloud->hasInstrumentDist() &&
                         targetCloud->instrumentDist.size() == tgt_points.size();
    const float boundary_thresh = g_ctrlgRimTgtThreshPx;
    const float inst_thresh     = g_instrumentPxThresh;

    // ---- Filter ----------------------------------------------------
    int n_total       = (int)tgt_points.size();
    int n_pass_bound  = 0;
    int n_inst_reject = 0;
    g_debugTargetBoundaryPoints.reserve(tgt_points.size() / 16);
    for (size_t i = 0; i < tgt_points.size(); i++) {
        if (targetCloud->boundaryDist[i] >= boundary_thresh) continue;
        n_pass_bound++;
        if (useInst && targetCloud->instrumentDist[i] < inst_thresh) {
            n_inst_reject++;
            continue;
        }
        g_debugTargetBoundaryPoints.push_back(tgt_points[i]);
    }

    std::cout << "[W/TgtBound] tgt_total=" << n_total
              << " boundary<" << boundary_thresh << ":" << n_pass_bound
              << " inst_reject=" << n_inst_reject
              << " final=" << g_debugTargetBoundaryPoints.size()
              << "  (useInst=" << (useInst ? "Y" : "N")
              << " instThresh=" << inst_thresh << "px)"
              << std::endl;

    if (g_debugTargetBoundaryPoints.empty()) {
        std::cout << "[W/TgtBound] no points pass filter — abort" << std::endl;
        return false;
    }

    // -----------------------------------------------------------------
    // Phase 7b Step 3a — also extract 2D contour directly from
    // g_boundaryDistMap. This is the "real" target lying in 2D pixel
    // space, used by Ctrl+W when g_shapeMatchUse2DCost == true.
    //
    // We populate this alongside the 3D points so Shift+W gives the
    // 2D path everything it needs without an extra key press. The 3D
    // points above stay for legacy 3D Ctrl+W and for the purple-dot
    // debug overlay (depth-lifted, debug-only).
    // -----------------------------------------------------------------
    g_debugTargetContour2D.clear();
    g_debugTargetContour2DSegSizes.clear();
    if (g_boundaryDistMap.valid && g_boundaryDistMap.width  > 0
                                && g_boundaryDistMap.height > 0)
    {
        std::vector<std::vector<glm::vec2>> segments;
        const float bdy_th  = 1.5f;
        const float inst_th = g_shapeMatch2DInstThreshPx;
        const BoundaryDistMap* inst_ptr =
            (inst_th > 0.0f && g_instrumentDistMap.valid)
                ? &g_instrumentDistMap : nullptr;
        const bool traced = RimShape::traceContour2D(
            g_boundaryDistMap, bdy_th, inst_ptr, inst_th,
            segments, &std::cout);
        if (traced && !segments.empty()) {
            g_debugTargetContour2D = std::move(segments[0]);
            g_debugTargetContour2DSegSizes.reserve(segments.size() + 1);
            g_debugTargetContour2DSegSizes.push_back(
                (int)g_debugTargetContour2D.size());
            for (size_t i = 1; i < segments.size(); i++) {
                g_debugTargetContour2DSegSizes.push_back(
                    (int)segments[i].size());
            }
            std::cout << "[W/TgtBound/2D] using largest segment: "
                      << g_debugTargetContour2D.size() << " pixels"
                      << "  (total segments=" << segments.size() << ")"
                      << std::endl;
        } else {
            std::cout << "[W/TgtBound/2D] no 2D contour traced —"
                      << " Ctrl+W 2D mode will fail" << std::endl;
        }
    } else {
        std::cout << "[W/TgtBound/2D] g_boundaryDistMap invalid"
                  << " — Ctrl+W 2D mode unavailable" << std::endl;
    }

    return true;
}


// =====================================================================
// Phase 7c (REDGE/稜線) — target ridge outlier removal helpers
// =====================================================================
//   Debug-only / one-shot (toggle・スライダ変更時に再実行)。target 稜線
//   (g_debugTargetRidgePoints) にだけ適用し、既存の purple/RIM 表示や
//   beta 重み付けには一切触れない。深度復元面の「後ろに伸びる尻尾」など、
//   2D 境界帯をすり抜ける幾何的外れ値を除く用。
//   O(N^2) brute force (N~15k で one-shot なら可、SOR は OMP 込み)。
//   ※ FLT_MAX は本ヘッダ未使用のため numeric_limits を使用。
namespace RidgeOutlier {

// SOR: 各点の k 近傍平均距離が (全体平均 + std_mul*σ) を超えたら除外。
inline void statisticalOutlierRemoval(std::vector<glm::vec3>& pts,
                                      int k, float std_mul,
                                      std::ostream* log = nullptr)
{
    const int n = (int)pts.size();
    if (n <= k + 1 || k < 1) return;

    std::vector<float> meanKnn(n, 0.0f);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> d2buf;
        d2buf.reserve(n);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n; i++) {
            d2buf.clear();
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                const glm::vec3 d = pts[j] - pts[i];
                d2buf.push_back(glm::dot(d, d));
            }
            const int kk = std::min(k, (int)d2buf.size());
            std::nth_element(d2buf.begin(), d2buf.begin() + kk, d2buf.end());
            double s = 0.0;
            for (int t = 0; t < kk; t++) s += std::sqrt((double)d2buf[t]);
            meanKnn[i] = (kk > 0) ? (float)(s / kk) : 0.0f;
        }
    }

    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i++) {
        sum  += meanKnn[i];
        sum2 += (double)meanKnn[i] * meanKnn[i];
    }
    const double mean   = sum / n;
    const double var    = std::max(0.0, sum2 / n - mean * mean);
    const double sd     = std::sqrt(var);
    const double thresh = mean + (double)std_mul * sd;

    std::vector<glm::vec3> kept;
    kept.reserve(n);
    for (int i = 0; i < n; i++)
        if (meanKnn[i] <= thresh) kept.push_back(pts[i]);

    if (log) {
        *log << "[Ridge/tgt/SOR] k=" << k << " std_mul=" << std_mul
             << "  meanNN=" << mean << " sd=" << sd << " thresh=" << thresh
             << "  kept=" << kept.size() << "/" << n
             << " (dropped " << (n - (int)kept.size()) << ")" << std::endl;
    }
    pts.swap(kept);
}

// Euclidean clustering (union-find): 最大クラスタだけ残す。
//   radius<=0 → auto = (最近傍距離の中央値) * 2.5。
//   最大クラスタが min_pts 未満でも空にはしない (一番大きいものは残す)。
inline void keepLargestCluster(std::vector<glm::vec3>& pts,
                               float radius, int min_pts,
                               std::ostream* log = nullptr)
{
    const int n = (int)pts.size();
    if (n < 2) return;
    const float kBig = std::numeric_limits<float>::max();

    // radius<=0 → auto: 各点の最近傍距離の中央値 * 2.5
    float r = radius;
    if (r <= 0.0f) {
        std::vector<float> nn(n, kBig);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; i++) {
            float best = kBig;
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                const glm::vec3 d = pts[j] - pts[i];
                const float d2 = glm::dot(d, d);
                if (d2 < best) best = d2;
            }
            nn[i] = std::sqrt(best);
        }
        std::vector<float> tmp = nn;
        std::nth_element(tmp.begin(), tmp.begin() + n / 2, tmp.end());
        const float med = tmp[n / 2];
        r = (med > 0.0f) ? med * 2.5f : 1e-3f;
    }
    const float r2 = r * r;

    // union-find (path halving、std::function 不使用)
    std::vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;
    auto findRoot = [&parent](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&parent, &findRoot](int a, int b) {
        const int ra = findRoot(a), rb = findRoot(b);
        if (ra != rb) parent[ra] = rb;
    };

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            const glm::vec3 d = pts[j] - pts[i];
            if (glm::dot(d, d) <= r2) unite(i, j);
        }
    }

    std::vector<int> csize(n, 0);
    for (int i = 0; i < n; i++) csize[findRoot(i)]++;
    int best_root = 0, best_size = -1, n_clusters = 0;
    for (int i = 0; i < n; i++) {
        if (csize[i] > 0) n_clusters++;
        if (csize[i] > best_size) { best_size = csize[i]; best_root = i; }
    }

    // min_pts 以上のクラスタを全部残す (尻尾=小クラスタを落とす)。
    // それで空になるなら最大クラスタだけ残す (空回避)。
    std::vector<glm::vec3> kept;
    kept.reserve(n);
    for (int i = 0; i < n; i++)
        if (csize[findRoot(i)] >= min_pts) kept.push_back(pts[i]);
    bool fellback = false;
    if (kept.empty()) {
        fellback = true;
        for (int i = 0; i < n; i++)
            if (findRoot(i) == best_root) kept.push_back(pts[i]);
    }

    if (log) {
        *log << "[Ridge/tgt/CC] radius=" << r << " min_pts=" << min_pts
             << "  clusters=" << n_clusters
             << " largest=" << best_size << "/" << n
             << "  kept=" << kept.size()
             << (fellback ? " (all < min_pts; kept largest only)" : "")
             << std::endl;
    }
    pts.swap(kept);
}

} // namespace RidgeOutlier


// =====================================================================
// Phase 7c (REDGE/稜線) — target ridge populate (upper half)
// =====================================================================
//   populateDebugTargetBoundary が作った g_debugTargetBoundaryPoints
//   (= rim band 3D) を AR カメラへ投影し、2D centroid より上
//   (screen +y=down なので p.y <= centroid.y) を上半分=稜線として 3D の
//   まま抽出。silSwBuildTgtPreview の Step 1-3 と同じ投影・centroid。
//   下半分 (p.y > centroid.y) は従来どおり RIM。target は静止なので
//   3D 点を直接保持 (purple rim band と同じ流儀)。
//   前提: 先に Shift+W で g_debugTargetBoundaryPoints が埋まっていること
//   (空なら自動で populateDebugTargetBoundary を呼ぶ)。
//
inline bool populateDebugTargetRidge()
{
    g_debugTargetRidgePoints.clear();

    if (g_debugTargetBoundaryPoints.empty()) {
        std::cout << "[Ridge/tgt] g_debugTargetBoundaryPoints empty —"
                  << " auto-running populateDebugTargetBoundary..." << std::endl;
        if (!populateDebugTargetBoundary() || g_debugTargetBoundaryPoints.empty()) {
            std::cout << "[Ridge/tgt] still empty — abort" << std::endl;
            return false;
        }
    }

    const glm::mat4 view_m = buildSilhouetteView();
    const glm::mat4 proj_m = buildSilhouetteProj();
    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
    const glm::mat4 M = proj_m * view_m;

    // Step 1: 全点を 2D pixel へ投影 (parallel に元 index を保持)
    std::vector<glm::vec2> all_2D;
    std::vector<int>       all_src;
    all_2D.reserve(g_debugTargetBoundaryPoints.size());
    all_src.reserve(g_debugTargetBoundaryPoints.size());
    for (int i = 0; i < (int)g_debugTargetBoundaryPoints.size(); i++) {
        const glm::vec4 clip = M * glm::vec4(g_debugTargetBoundaryPoints[i], 1.0f);
        if (clip.w < 1e-9f) continue;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        if (ndcx < -1.5f || ndcx > 1.5f || ndcy < -1.5f || ndcy > 1.5f) continue;
        all_2D.emplace_back((ndcx + 1.0f) * 0.5f * float(W_img),
                            (1.0f - ndcy) * 0.5f * float(H_img));
        all_src.push_back(i);
    }
    if (all_2D.size() < 10) {
        std::cout << "[Ridge/tgt] too few on-screen boundary points (<10)" << std::endl;
        return false;
    }

    // Step 2: 2D centroid
    glm::dvec2 sum2d(0.0);
    for (const auto& p : all_2D) sum2d += glm::dvec2(p);
    const glm::vec2 centroid(sum2d / double(all_2D.size()));

    // Step 3: 上半分=稜線 / 下半分=RIM を分ける軸を決める。
    //   split_normal = 各点を射影して centroid と比較する「下方向ベクトル」
    //                  (s = dot(p-centroid, split_normal); s<=0 → 上半分=稜線)。
    //   - mode 0 (既定): screen +y (=下方向)。p.y<=centroid.y → 上半分。
    //   - mode 1 (bbox長辺): 2D 点群の主軸(=長辺)を PCA で求め、その短軸を
    //     分割法線にする。横長/やや傾いた肝臓でも長辺に平行な線で「上側の
    //     長辺(稜線)/下側の長辺(RIM)」に割れる。向きは screen-up(-y) に合わせ
    //     るので ridge=上 のまま。長辺が水平なら mode 0 に一致。
    glm::vec2 split_normal(0.0f, 1.0f);   // mode0: +y(down)
    if (g_ridgeTgtSplitLongEdge) {
        double cxx = 0.0, cxy = 0.0, cyy = 0.0;
        for (const auto& p : all_2D) {
            const double dx = (double)p.x - centroid.x;
            const double dy = (double)p.y - centroid.y;
            cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
        }
        // 主軸(最大分散)角: theta = 0.5*atan2(2*cxy, cxx-cyy)
        const double theta = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);
        // 短軸(=分割法線) = 主軸 + 90deg
        glm::vec2 minor((float)(-std::sin(theta)), (float)(std::cos(theta)));
        if (minor.y > 0.0f) minor = -minor;   // screen-up(-y) に向ける
        split_normal = -minor;                 // 下方向に戻す (上=ridge を保つ)
    }

    int n_upper = 0, n_lower = 0;
    g_debugTargetRidgePoints.reserve(all_2D.size() / 2);
    for (size_t k = 0; k < all_2D.size(); k++) {
        const glm::vec2 rel = all_2D[k] - centroid;
        const float s = glm::dot(rel, split_normal);
        if (s <= 0.0f) {   // 上半分 = 稜線
            g_debugTargetRidgePoints.push_back(
                g_debugTargetBoundaryPoints[all_src[k]]);
            n_upper++;
        } else {
            n_lower++;
        }
    }

    std::cout << "[Ridge/tgt] on_screen=" << all_2D.size()
              << "  split=" << (g_ridgeTgtSplitLongEdge ? "bbox-long-edge" : "up/down")
              << "  upper(ridge)=" << n_upper
              << "  lower(rim)=" << n_lower << std::endl;

    // ---- (Phase 7c) optional outlier removal (debug toggle) ----------
    //   稜線(上半分)にだけ適用。purple RIM / beta 重み付けは不変。
    if (g_ridgeTgtRemoveOutliers && g_debugTargetRidgePoints.size() > 4) {
        const auto t0 = std::chrono::steady_clock::now();
        const int n_before = (int)g_debugTargetRidgePoints.size();
        if (g_ridgeTgtOutlierMode == 1) {
            RidgeOutlier::keepLargestCluster(
                g_debugTargetRidgePoints,
                g_ridgeTgtCcRadius, g_ridgeTgtCcMinPts, &std::cout);
        } else {
            RidgeOutlier::statisticalOutlierRemoval(
                g_debugTargetRidgePoints,
                g_ridgeTgtSorK, g_ridgeTgtSorStd, &std::cout);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[Ridge/tgt] outlier removal ("
                  << (g_ridgeTgtOutlierMode == 1 ? "largest-CC" : "SOR")
                  << "): " << n_before << " -> "
                  << g_debugTargetRidgePoints.size()
                  << "  (" << ms << " ms)" << std::endl;
    }

    if (g_debugTargetRidgePoints.empty()) {
        std::cout << "[Ridge/tgt] upper half empty — abort" << std::endl;
        return false;
    }
    return true;
}


// =====================================================================
// Phase 7b Step 3 helper — Ctrl+W key wrapper
// =====================================================================
//
// runDebugShapeMatchCoarse
//   詳細設計 §2 Stage 1 (COARSE search N=30) を実機適用:
//     1. source/target が未 populate なら自動 populate (Plain W/Shift+W
//        と同じフロー)
//     2. target 側を sortPointsByPCAAngle で順序付け (PCA → atan2 sort)
//     3. target chain 上の arc-length 等間隔 30 anchor で T 候補生成
//     4. 各 T で source を変換 → chamfer cost 計算
//     5. best 候補の予測 source 位置を g_debugShapeMatchBestSrc に保存
//     6. top-3 を log
//
// 計算量見積もり:
//   30 候補 × (source 254 + chamfer 254×30k = ~7.7M) = ~230M ops
//   single thread brute-force = ~1.5s. Ctrl+G の ~5s より速い。
//
// 結果適用 (mesh への transform 反映) はしない: Step 3 は「予測表示
// + cost 評価」までで、Step 4 (Ctrl+Shift+W) で Live bridge 経由で
// 実際に source mesh を動かす。
//
inline bool runDebugShapeMatchCoarse()
{
    g_debugShapeMatchBestSrc.clear();
    g_debugShapeMatchBestCost = 1e18;
    g_debugShapeMatchBestK    = -1;

    // ---- Auto-populate source/target if missing ---------------------
    if (g_debugSourceRimChain.empty()) {
        std::cout << "[Ctrl+W] source chain empty — auto-populating..."
                  << std::endl;
        if (!populateDebugSourceRimChain()) {
            std::cout << "[Ctrl+W] source populate failed — abort"
                      << std::endl;
            return false;
        }
    }
    if (g_debugTargetBoundaryPoints.empty()) {
        std::cout << "[Ctrl+W] target points empty — auto-populating..."
                  << std::endl;
        if (!populateDebugTargetBoundary()) {
            std::cout << "[Ctrl+W] target populate failed — abort"
                      << std::endl;
            return false;
        }
    }
    if (!liverMesh3D) {
        std::cout << "[Ctrl+W] no liverMesh3D — abort" << std::endl;
        return false;
    }

    // ---- Collect source positions at current pose -------------------
    std::vector<glm::vec3> src_pts;
    src_pts.reserve(g_debugSourceRimChain.size());
    const auto& V = liverMesh3D->mVertices;
    const int nV3 = (int)V.size();
    for (int idx : g_debugSourceRimChain) {
        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
        src_pts.emplace_back(V[idx*3], V[idx*3+1], V[idx*3+2]);
    }
    if (src_pts.empty()) {
        std::cout << "[Ctrl+W] no valid source positions — abort"
                  << std::endl;
        return false;
    }

    // =================================================================
    // Phase 7b Step 3a — Branch on 2D vs 3D cost
    //   2D path: depth-free, uses g_boundaryDistMap directly.
    //   3D path: legacy chamfer between depth-lifted target and source.
    // =================================================================
    if (g_shapeMatchUse2DCost) {
        // ---- 2D path: full-2D coarse search ------------------------
        if (!g_boundaryDistMap.valid) {
            std::cout << "[Ctrl+W/2D] g_boundaryDistMap invalid — "
                      << "fall back to 3D path" << std::endl;
            // fall through to legacy 3D below via flag flip
            // (so user gets a result instead of nothing)
            goto LEGACY_3D_PATH;
        }
        if (g_debugTargetContour2D.size() < 8) {
            std::cout << "[Ctrl+W/2D] target 2D contour too short ("
                      << g_debugTargetContour2D.size()
                      << ") — fall back to 3D path" << std::endl;
            goto LEGACY_3D_PATH;
        }

        // Resample 2D contour to N anchors
        const int N_anchors = std::max(8, g_shapeMatchContourN2D);
        std::vector<glm::vec2> anchors2D;
        RimShape::resampleArcLength2D(g_debugTargetContour2D,
                                      N_anchors, anchors2D);
        if ((int)anchors2D.size() < 8) {
            std::cout << "[Ctrl+W/2D] resample produced too few anchors ("
                      << anchors2D.size() << ") — fall back to 3D"
                      << std::endl;
            goto LEGACY_3D_PATH;
        }

        // AR fixed camera (same convention as runShiftE / IoU metrics)
        const glm::mat4 silView = buildSilhouetteView();
        const glm::mat4 silProj = buildSilhouetteProj();
        const int silW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
        const int silH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

        // ---- Source normal sign normalization (Idea A) -------------
        // PCA eigenvectors carry ± ambiguity. Flip the source rim's
        // PCA "patch normal" so it faces the camera; otherwise sign=0
        // candidates systematically produce a 180°-flipped pose
        // (observed: cos_range=[-0.96, -0.78] = ~140-165° rotation).
        // Controlled by g_shapeMatchFlipNormalToCamera (default ON).
        glm::vec3 src_normal_adj = g_debugSourceRimMajorNormal;
        if (g_shapeMatchFlipNormalToCamera) {
            const glm::vec3 cam_axis_world = glm::vec3(
                glm::inverse(silView) * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));
            if (glm::dot(src_normal_adj, cam_axis_world) < 0.0f) {
                src_normal_adj = -src_normal_adj;
                std::cout << "[Ctrl+W/2D] src_normal flipped to face camera"
                          << "  (cam_axis · orig_normal < 0)" << std::endl;
            }
        }

        std::cout << "[Ctrl+W/2D] silW=" << silW << " silH=" << silH
                  << "  bdy=" << g_boundaryDistMap.width
                  << "x" << g_boundaryDistMap.height
                  << "  anchors=" << anchors2D.size()
                  << "  src=" << src_pts.size()
                  << "  oof_dist=" << g_shapeMatchOutOfFrameDistPx << "px"
                  << "  cap=" << g_shapeMatchMaxDistCapPx << "px"
                  << "  min_inframe=" << g_shapeMatchMinInFrameRate
                  << "  flipN2cam=" << (g_shapeMatchFlipNormalToCamera ? "Y" : "N")
                  << std::endl;

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<RimShape::CoarseCandidate> candidates;
        if (!RimShape::runShapeMatchCoarse2D(
                src_pts,
                g_debugSourceRimCentroid,
                g_debugSourceRimPrincipalAxis,
                src_normal_adj,
                anchors2D,
                silView, silProj, silW, silH,
                g_boundaryDistMap,
                g_shapeMatchOutOfFrameDistPx,
                g_shapeMatchMaxDistCapPx,
                g_shapeMatchMinInFrameRate,
                candidates, &std::cout,
                g_shapeMatchSignMode,
                g_shapeMatchCoarseMaxRotDeg))
        {
            std::cout << "[Ctrl+W/2D] coarse search failed —"
                      << " fall back to 3D" << std::endl;
            goto LEGACY_3D_PATH;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(
                              t1 - t0).count();

        if (candidates.empty()) {
            std::cout << "[Ctrl+W/2D] no candidates produced —"
                      << " fall back to 3D" << std::endl;
            goto LEGACY_3D_PATH;
        }

        // Rotation-angle penalty (same as 3D path, sign=1/2/3 control)
        const float  anat_thresh = g_shapeMatchAnatomyThresh;
        const double anat_lambda = g_shapeMatchAnatomyLambda;
        int n_penalized = 0;
        float min_cos = 1.0f, max_cos = -1.0f;
        if (anat_lambda > 0.0) {
            for (auto& cand : candidates) {
                if (cand.cost >= 1e9) continue;        // already rejected
                const glm::mat3 R(cand.transform);
                const float trace = R[0][0] + R[1][1] + R[2][2];
                float ca = (trace - 1.0f) * 0.5f;
                if (ca > 1.0f) ca = 1.0f;
                if (ca < -1.0f) ca = -1.0f;
                if (ca < min_cos) min_cos = ca;
                if (ca > max_cos) max_cos = ca;
                if (ca < anat_thresh) {
                    cand.cost += anat_lambda * double(anat_thresh - ca);
                    n_penalized++;
                }
            }
        }
        std::cout << "[Ctrl+W/2D] Rotation penalty: thresh=" << anat_thresh
                  << " lambda=" << anat_lambda
                  << " cos_range=[" << min_cos << "," << max_cos << "]"
                  << " penalized=" << n_penalized << "/" << candidates.size()
                  << std::endl;

        // Best + top-5
        int best_i = 0;
        for (int i = 1; i < (int)candidates.size(); i++) {
            if (candidates[i].cost < candidates[best_i].cost) best_i = i;
        }
        g_debugShapeMatchBestCost = candidates[best_i].cost;
        g_debugShapeMatchBestK    = candidates[best_i].target_anchor_k;
        g_debugShapeMatchBestTransform = candidates[best_i].transform;

        // Re-evaluate best candidate to fill in-frame/in-mask diagnostics
        {
            std::vector<glm::vec3> src_best;
            src_best.reserve(src_pts.size());
            const glm::mat4& Tb = candidates[best_i].transform;
            for (const auto& p : src_pts) {
                const glm::vec4 v4 = Tb * glm::vec4(p, 1.0f);
                src_best.emplace_back(v4.x, v4.y, v4.z);
            }
            int n_in_frame = 0, n_in_mask = 0;
            (void)RimShape::project2DBoundaryDistance(
                src_best, silView, silProj, silW, silH,
                g_boundaryDistMap,
                g_shapeMatchOutOfFrameDistPx,
                g_shapeMatchMaxDistCapPx,
                &n_in_frame, &n_in_mask);
            g_debugShapeMatchBestInFrame =
                (src_pts.empty()) ? 0.0f
                                  : float(n_in_frame) / float(src_pts.size());
            g_debugShapeMatchBestInMask =
                (src_pts.empty()) ? 0.0f
                                  : float(n_in_mask)  / float(src_pts.size());

            g_debugShapeMatchBestSrc = std::move(src_best);
        }

        // Top-K log
        std::vector<int> idx_sorted(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) idx_sorted[i] = (int)i;
        std::sort(idx_sorted.begin(), idx_sorted.end(),
                  [&](int a, int b){ return candidates[a].cost < candidates[b].cost; });

        std::cout << "[Ctrl+W/2D] ShapeMatch coarse2D"
                  << "  elapsed=" << ms << "ms"
                  << "  best_cost=" << g_debugShapeMatchBestCost << "px"
                  << "  best_k=" << g_debugShapeMatchBestK
                  << "  best_sign=" << candidates[best_i].sign_code
                  << "(t" << ((candidates[best_i].sign_code & 1) ? "-" : "+")
                  << ",n" << ((candidates[best_i].sign_code & 2) ? "-" : "+")
                  << ")  in_frame=" << g_debugShapeMatchBestInFrame
                  << "  in_mask="   << g_debugShapeMatchBestInMask
                  << std::endl;
        const int show = std::min<int>(5, (int)idx_sorted.size());
        for (int t_rank = 0; t_rank < show; t_rank++) {
            const int i = idx_sorted[t_rank];
            std::cout << "    rank " << (t_rank + 1)
                      << ": k=" << candidates[i].target_anchor_k
                      << " sign=" << candidates[i].sign_code
                      << "(t" << ((candidates[i].sign_code & 1) ? "-" : "+")
                      << ",n" << ((candidates[i].sign_code & 2) ? "-" : "+")
                      << ")  cost=" << candidates[i].cost << "px"
                      << std::endl;
        }
        return true;
    }

LEGACY_3D_PATH:

    // ---- Sort target by PCA angle ----------------------------------
    glm::vec3 tgt_centroid, tgt_normal, tgt_axis;
    double tgt_planarity = 1.0;
    std::vector<glm::vec3> tgt_chain;
    if (!RimShape::sortPointsByPCAAngle(
            g_debugTargetBoundaryPoints,
            tgt_chain, tgt_centroid, tgt_normal, tgt_axis, tgt_planarity,
            &std::cout))
    {
        std::cout << "[Ctrl+W] target sort failed — abort" << std::endl;
        return false;
    }

    // ---- Coarse search 30 candidates -------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    const int N_coarse = 30;
    std::vector<RimShape::CoarseCandidate> candidates;
    if (!RimShape::runShapeMatchCoarse(
            src_pts,
            g_debugSourceRimCentroid,
            g_debugSourceRimPrincipalAxis,
            g_debugSourceRimMajorNormal,
            tgt_chain, tgt_normal,
            N_coarse, candidates, &std::cout,
            g_shapeMatchSignMode))   // Phase 7b Step 4a: sign filter
    {
        std::cout << "[Ctrl+W] coarse search failed — abort" << std::endl;
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (candidates.empty()) {
        std::cout << "[Ctrl+W] no candidates produced — abort" << std::endl;
        return false;
    }

    // ---- Rotation-angle penalty -----------------------------------
    //   sign=1/2/3 候補は target frame の 180°回転を含むので、R 全体の
    //   回転角度 (= cos_angle = (trace-1)/2) で大回転を penalize する
    //   ことで「Apply Init Pose を信じる」小調整に Shape Match を制限
    //   する。これは旧版 (d_lr/d_cc dot 拘束) で sign=3(t-,n-) が
    //   bitangent 軸回転で解剖軸を偶然保持する case をすり抜けたのを
    //   防ぐ確実な手段。
    //
    //   cos_angle ∈ [-1,+1]
    //     +1 = 0° rotation (identity)
    //      0 = 90° rotation (thresh default)
    //     -1 = 180° rotation (sign=1/2/3 はだいたい -1 近辺)
    const float  anat_thresh = g_shapeMatchAnatomyThresh;
    const double anat_lambda = g_shapeMatchAnatomyLambda;

    int n_penalized = 0;
    float min_cos_angle = 1.0f, max_cos_angle = -1.0f;
    if (anat_lambda > 0.0) {
        for (auto& cand : candidates) {
            const glm::mat3 R(cand.transform);
            const float trace = R[0][0] + R[1][1] + R[2][2];
            float cos_angle = (trace - 1.0f) * 0.5f;
            // numerical safety
            if (cos_angle > 1.0f) cos_angle = 1.0f;
            if (cos_angle < -1.0f) cos_angle = -1.0f;
            if (cos_angle < min_cos_angle) min_cos_angle = cos_angle;
            if (cos_angle > max_cos_angle) max_cos_angle = cos_angle;
            if (cos_angle < anat_thresh) {
                const double pen = anat_lambda * double(anat_thresh - cos_angle);
                cand.cost += pen;
                n_penalized++;
            }
        }
    }
    std::cout << "[Ctrl+W] Rotation penalty: thresh=" << anat_thresh
              << " lambda=" << anat_lambda
              << " cos_angle_range=[" << min_cos_angle << "," << max_cos_angle << "]"
              << " penalized=" << n_penalized << "/" << candidates.size()
              << std::endl;

    // ---- Find best + top-5 -----------------------------------------
    int best_i = 0;
    for (int i = 1; i < (int)candidates.size(); i++) {
        if (candidates[i].cost < candidates[best_i].cost) best_i = i;
    }
    g_debugShapeMatchBestCost = candidates[best_i].cost;
    g_debugShapeMatchBestK    = candidates[best_i].target_anchor_k;
    g_debugShapeMatchBestTransform = candidates[best_i].transform;  // for Step 4

    // Apply best transform to source points → predicted positions
    const glm::mat4& T = candidates[best_i].transform;
    g_debugShapeMatchBestSrc.reserve(src_pts.size());
    for (const auto& p : src_pts) {
        const glm::vec4 v4 = T * glm::vec4(p, 1.0f);
        g_debugShapeMatchBestSrc.emplace_back(v4.x, v4.y, v4.z);
    }

    // Log top-3 (sort indices only — avoid copying CoarseCandidate)
    std::vector<int> idx_sorted(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) idx_sorted[i] = (int)i;
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b){ return candidates[a].cost < candidates[b].cost; });

    std::cout << "[Ctrl+W] ShapeMatch coarse N=" << N_coarse << "*4_signs"
              << "  elapsed=" << ms << "ms"
              << "  best_cost=" << g_debugShapeMatchBestCost
              << "  best_k=" << g_debugShapeMatchBestK
              << "  best_sign=" << candidates[best_i].sign_code
              << "(t" << ((candidates[best_i].sign_code & 1) ? "-" : "+")
              << ",n" << ((candidates[best_i].sign_code & 2) ? "-" : "+")
              << ")" << std::endl;
    const int show = std::min<int>(5, (int)idx_sorted.size());   // top-5 for more visibility
    for (int t_rank = 0; t_rank < show; t_rank++) {
        const int i = idx_sorted[t_rank];
        std::cout << "    rank " << (t_rank + 1)
                  << ": k=" << candidates[i].target_anchor_k
                  << " sign=" << candidates[i].sign_code
                  << "(t" << ((candidates[i].sign_code & 1) ? "-" : "+")
                  << ",n" << ((candidates[i].sign_code & 2) ? "-" : "+")
                  << ")  cost=" << candidates[i].cost
                  << std::endl;
    }
    return true;
}


// =====================================================================
// Phase 7b Step 3b — Gauss-Newton refinement (Alt+W)
// =====================================================================
//
// runDebugShapeMatchGN
//   Alt+W's core function:
//     1. Run Coarse2D (forced 2D path) → initial T_coarse + src/tgt populate
//     2. Build / cache unsigned boundary distance map (BFS from contour
//        in both directions, so GN gets a gradient outside the SAM2 mask)
//     3. Levenberg-Marquardt refine: T_GN = arg min Σ bdy(π(T·p_i))²
//     4. Update g_debugShapeMatchBestTransform / BestCost / BestSrc with
//        the refined result. The main.cpp Alt+W handler then applies it.
//
// Returns true if Coarse2D succeeded (GN failure still returns true with
// the coarse T retained — best-effort).
//
// Why we call runDebugShapeMatchCoarse() directly rather than duplicating
// its body: that function already handles auto-populate, sign-mask, PCA
// cache reuse, and the Idea-A source-normal flip. We just bolt GN on top.
//
// Performance: coarse ~1ms + unsigned bdy build ~15ms (cached after 1st
// call) + GN 5-15 iters × ~1ms = ~20ms typical refresh, ~2ms repeated.
// =====================================================================
inline bool runDebugShapeMatchGN()
{
    // =================================================================
    // Alt+W (Step 3b revised)
    // -----------------------------------------------------------------
    // Two modes, selected by g_shapeMatchAltWSkipCoarse:
    //
    //   SkipCoarse=ON  (default, recommended):
    //     Trust the current mesh pose (Apply Init Pose result) and run
    //     GN refine STARTING FROM IDENTITY on the current world-space
    //     rim. Coarse2D is NOT invoked. The 6-DoF (or 3-DoF translation-
    //     only) LM solves a tiny perturbation on top of the user's
    //     init pose. Pose break-down is structurally impossible because
    //     the trust region cap on ||Δξ|| is 0.05 (≈ 5cm + 3° / iter).
    //
    //   SkipCoarse=OFF (legacy):
    //     Coarse2D first (forces 2D path), then GN refine from coarse
    //     best transform. Old Step 3b behavior. Kept for comparison
    //     and for situations where the init pose is bad.
    //
    // Both modes feed g_shapeMatchGNTranslationOnly through to the
    // solver. With TRANS_ONLY=ON the rotation block of JTJ is locked
    // to identity so Δω = 0 by construction; rim alignment becomes a
    // pure 3-DoF translation fit, immune to depth-degeneracy.
    // =================================================================

    if (!liverMesh3D) {
        std::cout << "[Alt+W] no liverMesh3D — abort" << std::endl;
        return false;
    }
    if (!g_boundaryDistMap.valid) {
        std::cout << "[Alt+W] g_boundaryDistMap invalid — load a depth"
                  << " scene first" << std::endl;
        return false;
    }

    // Auto-populate prerequisites
    if (g_debugSourceRimChain.empty()) {
        std::cout << "[Alt+W] auto-running populateDebugSourceRimChain()..."
                  << std::endl;
        if (!populateDebugSourceRimChain()) {
            std::cout << "[Alt+W] failed to populate source rim chain"
                      << std::endl;
            return false;
        }
    }

    double coarse_cost = -1.0;
    glm::mat4 T_init(1.0f);

    if (g_shapeMatchAltWSkipCoarse) {
        // ---- SkipCoarse mode: refine from identity on current pose --
        std::cout << "[Alt+W] SKIP_COARSE mode — refining current pose"
                  << " (no Coarse2D)" << std::endl;
    } else {
        // ---- Legacy mode: Coarse2D first --------------------------
        const bool saved_use_2d = g_shapeMatchUse2DCost;
        g_shapeMatchUse2DCost = true;
        const bool coarse_ok = runDebugShapeMatchCoarse();
        g_shapeMatchUse2DCost = saved_use_2d;
        if (!coarse_ok) {
            std::cout << "[Alt+W] Coarse2D failed — abort" << std::endl;
            g_debugShapeMatchGNInitCost  = 0.0;
            g_debugShapeMatchGNFinalCost = 0.0;
            g_debugShapeMatchGNIters     = 0;
            g_debugShapeMatchGNConverged = false;
            g_debugShapeMatchGNReason    = 3;
            return false;
        }
        T_init      = g_debugShapeMatchBestTransform;
        coarse_cost = g_debugShapeMatchBestCost;
    }

    // ---- Build / cache unsigned boundary distance map --------------
    if (!g_gnUnsignedBdyValid) {
        std::cout << "[Alt+W] building unsigned boundary distance map..."
                  << std::endl;
        if (!RimShape::buildUnsignedBoundaryMap(
                g_boundaryDistMap,
                g_gnUnsignedBdy, g_gnUnsignedBdyW, g_gnUnsignedBdyH,
                &std::cout))
        {
            std::cout << "[Alt+W] unsigned bdy build failed — abort"
                      << std::endl;
            return false;
        }
        g_gnUnsignedBdyValid = true;
    }

    // ---- Collect current rim source points (world space) ----------
    std::vector<glm::vec3> src_pts;
    src_pts.reserve(g_debugSourceRimChain.size());
    const auto& V = liverMesh3D->mVertices;
    const int nV3 = (int)V.size();
    for (int idx : g_debugSourceRimChain) {
        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
        src_pts.emplace_back(V[idx*3], V[idx*3+1], V[idx*3+2]);
    }
    if (src_pts.empty()) {
        std::cout << "[Alt+W] no source rim points — abort" << std::endl;
        return false;
    }

    // ---- Levenberg-Marquardt refine -------------------------------
    const glm::mat4 silView = buildSilhouetteView();
    const glm::mat4 silProj = buildSilhouetteProj();

    RimShape::GNResult gn;
    const auto t0 = std::chrono::steady_clock::now();
    const bool gn_ok = RimShape::runShapeMatchGN(
        src_pts, T_init,
        silView, silProj,
        g_gnUnsignedBdy, g_gnUnsignedBdyW, g_gnUnsignedBdyH,
        g_shapeMatchGNMaxIter,
        g_shapeMatchGNLambdaInit,
        g_shapeMatchGNEpsStep,
        g_shapeMatchGNEpsRel,
        gn, &std::cout,
        g_shapeMatchGNTranslationOnly,
        g_shapeMatchGNLambdaMin,
        g_shapeMatchGNStepMax);
    const auto t1 = std::chrono::steady_clock::now();
    const double gn_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!gn_ok) {
        std::cout << "[Alt+W] GN reported failure" << std::endl;
        g_debugShapeMatchGNInitCost  = coarse_cost > 0 ? coarse_cost : 0.0;
        g_debugShapeMatchGNFinalCost = coarse_cost > 0 ? coarse_cost : 0.0;
        g_debugShapeMatchGNIters     = 0;
        g_debugShapeMatchGNConverged = false;
        g_debugShapeMatchGNReason    = 3;
        return false;
    }

    // ---- Adopt GN result ------------------------------------------
    g_debugShapeMatchBestTransform = gn.final_T;
    g_debugShapeMatchBestCost      = gn.final_cost;
    g_debugShapeMatchGNInitCost    = gn.initial_cost;
    g_debugShapeMatchGNFinalCost   = gn.final_cost;
    g_debugShapeMatchGNIters       = gn.n_iter;
    g_debugShapeMatchGNConverged   = gn.converged;
    g_debugShapeMatchGNReason      = gn.reason;
    g_debugShapeMatchGNInFrame     = gn.n_in_frame;

    g_debugShapeMatchBestSrc.clear();
    g_debugShapeMatchBestSrc.reserve(src_pts.size());
    for (const auto& p : src_pts) {
        const glm::vec4 v4 = gn.final_T * glm::vec4(p, 1.0f);
        g_debugShapeMatchBestSrc.emplace_back(v4.x, v4.y, v4.z);
    }

    const char* reason_str[] = {"step", "rel_cost", "max_iter", "lm_fail"};
    std::cout << "[Alt+W] GN refined ("
              << (g_shapeMatchAltWSkipCoarse ? "SkipCoarse" : "Coarse+GN")
              << "/"
              << (g_shapeMatchGNTranslationOnly ? "TransOnly3DoF" : "Full6DoF")
              << "):"
              << "  cost " << gn.initial_cost << "→" << gn.final_cost << "px"
              << "  Δ=" << (gn.initial_cost - gn.final_cost) << "px"
              << "  iters=" << gn.n_iter
              << "  " << reason_str[std::min(3, gn.reason)]
              << "  conv=" << (gn.converged ? "Y" : "N")
              << "  in_frame=" << gn.n_in_frame << "/" << src_pts.size()
              << "  t=" << gn_ms << "ms"
              << std::endl;

    return true;
}


// =====================================================================
// Phase 7b Step 3c — Contour Sweep (Ctrl+Alt+W) driver
// =====================================================================
//
// Three entry points:
//   startContourSweep()  — initialize state, populate target/source,
//                          enter Phase 1.
//   tickContourSweep()   — called every main-loop frame; processes a
//                          batch of candidates, advances state. Returns
//                          true while sweep is ongoing.
//   finishContourSweep() — adopt best T, apply to organ meshes, save
//                          to PoseLibrary. Mirrors Ctrl+W / Alt+W tail.
//
// The mesh is NOT touched during ticks — only the visualization
// (g_debugShapeMatchBestSrc) updates so the user sees the best-so-far
// pose live. The mesh is moved exactly once, on finishContourSweep.
// =====================================================================
inline bool startContourSweep()
{
    auto& S = g_contourSweepState;
    S = RimShape::ContourSweepState{};   // reset

    if (!liverMesh3D) {
        std::cout << "[Ctrl+Alt+W] no liverMesh3D — abort" << std::endl;
        return false;
    }
    if (!g_boundaryDistMap.valid) {
        std::cout << "[Ctrl+Alt+W] g_boundaryDistMap invalid — load depth first"
                  << std::endl;
        return false;
    }

    // Auto-populate source rim chain
    if (g_debugSourceRimChain.empty()) {
        std::cout << "[Ctrl+Alt+W] auto-running populateDebugSourceRimChain..."
                  << std::endl;
        if (!populateDebugSourceRimChain()) {
            std::cout << "[Ctrl+Alt+W] failed to populate source rim" << std::endl;
            return false;
        }
    }

    // Auto-populate target contour (2D pixel polyline)
    if (g_debugTargetContour2D.empty()) {
        std::cout << "[Ctrl+Alt+W] auto-running populateDebugTargetBoundary..."
                  << std::endl;
        if (!populateDebugTargetBoundary()) {
            std::cout << "[Ctrl+Alt+W] failed to populate target boundary"
                      << std::endl;
            return false;
        }
    }
    if (g_debugTargetContour2D.size() < 3) {
        std::cout << "[Ctrl+Alt+W] target contour too short ("
                  << g_debugTargetContour2D.size() << " px) — abort"
                  << std::endl;
        return false;
    }

    // Build / cache unsigned boundary distance map (reuse Alt+W's cache)
    if (!g_gnUnsignedBdyValid) {
        if (!RimShape::buildUnsignedBoundaryMap(
                g_boundaryDistMap,
                g_gnUnsignedBdy, g_gnUnsignedBdyW, g_gnUnsignedBdyH,
                &std::cout))
        {
            std::cout << "[Ctrl+Alt+W] unsigned bdy build failed — abort"
                      << std::endl;
            return false;
        }
        g_gnUnsignedBdyValid = true;
    }

    // Configure state
    S.n_target            = std::max(4, g_shapeMatchSweepNTarget);
    S.n_source            = std::max(4, g_shapeMatchSweepNSource);
    S.n_rotation          = std::max(4, g_shapeMatchSweepNRotation);
    S.total_frames_phase1 = std::max(1, g_shapeMatchSweepFrames1);
    S.total_frames_phase2 = std::max(1, g_shapeMatchSweepFrames2);
    S.total_candidates    = S.n_target * S.n_source * S.n_rotation;
    S.candidates_per_frame =
        std::max(1, (S.total_candidates + S.total_frames_phase1 - 1)
                       / S.total_frames_phase1);

    // Phase 1: full-range SECTOR-BASED target anchor extraction
    //   Plan A (Step 3c++): ANCHOR-COMPLETELY-FIXED.
    //     The full target boundary is passed to the extractor — centroid,
    //     sector binning, and medoid selection ALL use target-only data.
    //     Anchors are therefore invariant under source pose changes
    //     ("source を動かしても anchor は動かない" guarantee).
    //
    //     The "stay near initial pose" filter (bbox by source rim) is
    //     applied as a per-anchor TAG (tagAnchorsInsideSourceBbox) instead
    //     of a pre-filter, so it still gates sweep candidates without
    //     ever moving the anchor positions.
    {
        const glm::mat4 view_pre = buildSilhouetteView();
        const glm::mat4 proj_pre = buildSilhouetteProj();
        const int W_pre = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
        const int H_pre = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;

        // 1. Extract anchors from the FULL boundary (no pre-filter).
        RimShape::extractSectorBasedTargetAnchors3D(
            g_debugTargetBoundaryPoints,
            view_pre, proj_pre, W_pre, H_pre,
            S.n_target,
            S.tgt_anchors_3D,
            /*out_sector_idx=*/        nullptr,
            /*out_centroid_2D=*/       &S.locked_centroid_2D,
            /*full_boundary_3D_for_centroid=*/ nullptr);
        RimShape::project3DAnchorsTo2D(
            S.tgt_anchors_3D, view_pre, proj_pre, W_pre, H_pre,
            S.tgt_anchors_2D);

        // 2. Tag each anchor inside / outside source bbox + margin.
        //    Used by tick() to skip out-of-bbox candidates and by
        //    preview() to render outside anchors in dim gray. Source
        //    bbox is recomputed each sweep start from the CURRENT pose,
        //    so the tag set responds to user pose adjustments while the
        //    anchors themselves stay put.
        if (g_shapeMatchSweepFilterByRim
            && !g_debugSourceRimChain.empty() && liverMesh3D)
        {
            std::vector<glm::vec3> src_rim_3D;
            src_rim_3D.reserve(g_debugSourceRimChain.size());
            const auto& V = liverMesh3D->mVertices;
            const int nV3 = (int)V.size();
            for (int idx : g_debugSourceRimChain) {
                if (idx * 3 + 2 < nV3)
                    src_rim_3D.emplace_back(V[idx*3], V[idx*3+1], V[idx*3+2]);
            }
            RimShape::tagAnchorsInsideSourceBbox(
                S.tgt_anchors_3D, src_rim_3D,
                view_pre, proj_pre, W_pre, H_pre,
                g_shapeMatchSweepFilterMarginPx,
                S.tgt_anchor_inside_bbox);
        } else {
            S.tgt_anchor_inside_bbox.assign(S.tgt_anchors_3D.size(), (uint8_t)1);
        }
        int n_inside = 0;
        for (uint8_t b : S.tgt_anchor_inside_bbox) if (b) n_inside++;
        std::cout << "[Ctrl+Alt+W] sector-based target anchors: "
                  << S.tgt_anchors_3D.size() << " / " << S.n_target
                  << " sectors non-empty"
                  << "  bbox_inside=" << n_inside << "/"
                  << S.tgt_anchor_inside_bbox.size()
                  << " (margin=" << g_shapeMatchSweepFilterMarginPx << "px)"
                  << std::endl;
    }

    // -----------------------------------------------------------------
    // Endpoint constraint setup:
    //   1. Detect if source rim chain is OPEN (caudal-only) or CLOSED
    //      (full rim). Closed loops have no meaningful "endpoint" so
    //      the constraint is disabled in that case.
    //   2. For open chains, project both source endpoints to 2D and
    //      determine which target endpoint each maps closer to.
    //   3. If reversed, build a reversed chain so that after resampling
    //      src_pivots[0] is geometrically near tgt_anchors[0].
    // -----------------------------------------------------------------
    std::vector<int> src_chain_for_sweep = g_debugSourceRimChain;
    {
        const auto& V = liverMesh3D->mVertices;
        const int K = (int)g_debugSourceRimChain.size();
        bool src_open = true;     // default open
        bool dir_reversed = false;

        if (K >= 3) {
            const int idx_A = g_debugSourceRimChain.front();
            const int idx_B = g_debugSourceRimChain.back();
            if (idx_A * 3 + 2 < (int)V.size() &&
                idx_B * 3 + 2 < (int)V.size())
            {
                const glm::vec3 src_A_3D(V[idx_A*3],   V[idx_A*3+1], V[idx_A*3+2]);
                const glm::vec3 src_B_3D(V[idx_B*3],   V[idx_B*3+1], V[idx_B*3+2]);
                const float endpoint_dist = glm::length(src_A_3D - src_B_3D);

                // Average adjacent-segment length over the chain
                float sum_seg = 0.0f;
                int   n_seg = 0;
                for (int i = 1; i < K; i++) {
                    const int ia = g_debugSourceRimChain[i-1];
                    const int ib = g_debugSourceRimChain[i];
                    if (ia*3+2 >= (int)V.size() || ib*3+2 >= (int)V.size()) continue;
                    const glm::vec3 pa(V[ia*3], V[ia*3+1], V[ia*3+2]);
                    const glm::vec3 pb(V[ib*3], V[ib*3+1], V[ib*3+2]);
                    sum_seg += glm::length(pa - pb);
                    n_seg++;
                }
                const float avg_seg = (n_seg > 0) ? (sum_seg / float(n_seg)) : 0.0f;
                // Open if endpoint distance >> typical adjacent step
                src_open = (avg_seg > 1e-6f && endpoint_dist > avg_seg * 5.0f);

                if (g_shapeMatchSweepUseEndpointConstraint && src_open
                    && S.tgt_anchors_2D.size() >= 2)
                {
                    // Project source endpoints to 2D using AR camera
                    const glm::mat4 view_m = buildSilhouetteView();
                    const glm::mat4 proj_m = buildSilhouetteProj();
                    auto project_2d = [&](const glm::vec3& p) -> glm::vec2 {
                        const glm::vec4 clip = (proj_m * view_m) * glm::vec4(p, 1.0f);
                        if (std::abs(clip.w) < 1e-9f) return glm::vec2(-1e6f, -1e6f);
                        const float ndcx = clip.x / clip.w;
                        const float ndcy = clip.y / clip.w;
                        return glm::vec2(
                            (ndcx + 1.0f) * 0.5f * float(g_boundaryDistMap.width),
                            (1.0f - ndcy) * 0.5f * float(g_boundaryDistMap.height));
                    };
                    const glm::vec2 src_A_2D = project_2d(src_A_3D);
                    const glm::vec2 src_B_2D = project_2d(src_B_3D);
                    // tgt_anchors_2D[0] and back() are the anchors closest
                    // to the contour's two ends after arc-length resampling.
                    const glm::vec2 tgt_A_2D = S.tgt_anchors_2D.front();
                    const glm::vec2 tgt_B_2D = S.tgt_anchors_2D.back();

                    const float d_fwd =
                        glm::length(src_A_2D - tgt_A_2D)
                        + glm::length(src_B_2D - tgt_B_2D);
                    const float d_rev =
                        glm::length(src_A_2D - tgt_B_2D)
                        + glm::length(src_B_2D - tgt_A_2D);
                    dir_reversed = (d_rev < d_fwd);

                    if (dir_reversed) {
                        std::reverse(src_chain_for_sweep.begin(),
                                     src_chain_for_sweep.end());
                    }

                    std::cout << "[Ctrl+Alt+W] endpoint constraint:"
                              << "  src_open=Y  dir="
                              << (dir_reversed ? "REVERSED" : "forward")
                              << "  d_fwd=" << d_fwd << "px"
                              << "  d_rev=" << d_rev << "px"
                              << "  tolerance=" << g_shapeMatchSweepEndpointTolerance
                              << std::endl;
                } else if (!src_open) {
                    std::cout << "[Ctrl+Alt+W] endpoint constraint: src is CLOSED"
                              << " (endpoint_dist=" << endpoint_dist
                              << ", avg_seg=" << avg_seg
                              << ") — constraint disabled"
                              << std::endl;
                }
            }
        }
        g_shapeMatchSweepSrcIsOpenDiag   = src_open;
        g_shapeMatchSweepDirReversedDiag = dir_reversed;
        // Persist for Phase 2
        S.src_rim_chain_used = src_chain_for_sweep;
        S.dir_reversed = dir_reversed;
        S.endpoint_constraint_active =
            g_shapeMatchSweepUseEndpointConstraint && src_open;
    }

    // Resample source pivots and capture each pivot's nearest rim-chain
    // mesh vertex index — that gives us the source LR/CC label per pivot
    // for the Phase 7b Step 3c++ label-based orientation lock below.
    std::vector<int> src_pivot_vert_idx;
    RimShape::arclengthResampleRim3D(
        src_chain_for_sweep, liverMesh3D->mVertices,
        S.n_source, S.src_pivots_3D,
        &src_pivot_vert_idx);

    // -----------------------------------------------------------------
    // Phase 7b Step 3c++: Label-based orientation lock setup.
    //   Source side: inherit PURE_RIGHT / PURE_LEFT / BOUNDARY from
    //                LiverLeftRightLabel and CRANIAL/CAUDAL from
    //                LiverCranioCaudalLabel via src_pivot_vert_idx.
    //   Target side: classify each anchor as SS_LR_RIGHT / LEFT /
    //                AMBIGUOUS by 2D x vs locked_centroid_2D ± band,
    //                and CC equivalent on y. AMBIGUOUS = inside the
    //                neutral band — the image-midline counterpart to
    //                source's BOUNDARY label.
    //
    // If either label set is missing, auto-degrade the lock to OFF
    // with a diagnostic line — Check B (rot cap) still applies but
    // Check A becomes a pass-through.
    // -----------------------------------------------------------------
    {
        const bool lr_ready =
            g_liverLR.valid()
            && (int)g_liverLR.labels.size()
                   == (int)(liverMesh3D->mVertices.size() / 3);
        const bool cc_ready =
            g_liverCC.valid()
            && (int)g_liverCC.labels.size()
                   == (int)(liverMesh3D->mVertices.size() / 3);
        g_shapeMatchSweepLabelLockReadyDiag = (lr_ready && cc_ready);

        S.src_pivot_lr_label.clear();
        S.src_pivot_cc_label.clear();
        S.src_pivot_lr_label.reserve(S.src_pivots_3D.size());
        S.src_pivot_cc_label.reserve(S.src_pivots_3D.size());
        if (lr_ready && cc_ready
            && src_pivot_vert_idx.size() == S.src_pivots_3D.size())
        {
            for (int vi : src_pivot_vert_idx) {
                if (vi < 0 || vi >= (int)g_liverLR.labels.size()
                           || vi >= (int)g_liverCC.labels.size())
                {
                    // Defensive — treat unknown as BOUNDARY/CAUDAL so it
                    // pairs with anything (passes Check A trivially).
                    S.src_pivot_lr_label.push_back(
                        (uint8_t)LiverLeftRightLabel::BOUNDARY);
                    S.src_pivot_cc_label.push_back(
                        (uint8_t)LiverCranioCaudalLabel::CAUDAL);
                    continue;
                }
                S.src_pivot_lr_label.push_back(g_liverLR.labels[vi]);
                S.src_pivot_cc_label.push_back(g_liverCC.labels[vi]);
            }
        } else {
            // No valid labels — fill with BOUNDARY/CAUDAL so the
            // skip-check is a no-op even if it accidentally fires.
            S.src_pivot_lr_label.assign(
                S.src_pivots_3D.size(),
                (uint8_t)LiverLeftRightLabel::BOUNDARY);
            S.src_pivot_cc_label.assign(
                S.src_pivots_3D.size(),
                (uint8_t)LiverCranioCaudalLabel::CAUDAL);
        }

        // Classify target anchors by their 2D screen position relative
        // to the locked centroid. Screen +x → patient's RIGHT, screen
        // -y (smaller y in pixel coords) → CRANIAL, by Apply-Init-Pose
        // convention.
        const float band = std::max(0.0f, g_shapeMatchSweepNeutralBandPx);
        S.tgt_anchor_screen_lr.clear();
        S.tgt_anchor_screen_cc.clear();
        S.tgt_anchor_screen_lr.reserve(S.tgt_anchors_2D.size());
        S.tgt_anchor_screen_cc.reserve(S.tgt_anchors_2D.size());
        int n_R = 0, n_L = 0, n_Ax = 0, n_C = 0, n_K = 0, n_Ay = 0;
        for (const auto& a2d : S.tgt_anchors_2D) {
            const float dx = a2d.x - S.locked_centroid_2D.x;
            const float dy = a2d.y - S.locked_centroid_2D.y;
            uint8_t lr = RimShape::SS_LR_AMBIGUOUS;
            if      (dx >  band) { lr = RimShape::SS_LR_RIGHT;   n_R++; }
            else if (dx < -band) { lr = RimShape::SS_LR_LEFT;    n_L++; }
            else                 {                                n_Ax++; }
            uint8_t cc = RimShape::SS_CC_AMBIGUOUS;
            if      (dy < -band) { cc = RimShape::SS_CC_CRANIAL; n_C++; }
            else if (dy >  band) { cc = RimShape::SS_CC_CAUDAL;  n_K++; }
            else                 {                                n_Ay++; }
            S.tgt_anchor_screen_lr.push_back(lr);
            S.tgt_anchor_screen_cc.push_back(cc);
        }

        S.orientation_lock_active =
            g_shapeMatchSweepUseOrientationLock && lr_ready && cc_ready;
        std::cout << "[Ctrl+Alt+W] orientation lock: "
                  << (S.orientation_lock_active ? "ON (label)" : "OFF")
                  << "  rot_cap=" << g_shapeMatchSweepRotationLockDeg << "°"
                  << "  band=" << band << "px"
                  << "  LR_ready=" << (lr_ready ? "Y" : "N")
                  << "  CC_ready=" << (cc_ready ? "Y" : "N")
                  << "  centroid_2D=(" << S.locked_centroid_2D.x
                  << "," << S.locked_centroid_2D.y << ")"
                  << "  tgt LR R/L/Amb=" << n_R << "/" << n_L << "/" << n_Ax
                  << "  CC C/K/Amb="    << n_C << "/" << n_K << "/" << n_Ay
                  << std::endl;
    }

    // -----------------------------------------------------------------
    // Backproject target anchors (2D pixel coords) to world 3D on the
    // plane z = source_rim_centroid.z so they can be drawn next to the
    // source pivots in the AR scene. The exact depth doesn't matter
    // for the visualization — only that target dots appear at the
    // expected screen positions and roughly the right scene depth.
    // -----------------------------------------------------------------

    // ---- Target anchor visualization (sector-based) -----------------
    // The new sector-based extraction already gives us 3D world points
    // (medoids of 3D boundary points within each angular sector), so
    // no back-projection is needed — just copy them out for rendering.
    g_contourSweepTgtAnchors3D = S.tgt_anchors_3D;
    g_contourSweepSrcPivotsTrial.clear();   // populated each tick
    g_contourSweepCurrentITgt = -1;
    g_contourSweepCurrentJSrc = -1;

    if ((int)S.tgt_anchors_2D.size() < 3 ||
        (int)S.src_pivots_3D.size()  < 3) {
        std::cout << "[Ctrl+Alt+W] resample failed (tgt="
                  << S.tgt_anchors_2D.size()
                  << " src=" << S.src_pivots_3D.size() << ") — abort"
                  << std::endl;
        return false;
    }

    S.active = true;
    S.phase  = 1;
    S.candidate_idx = 0;
    S.current_frame = 0;
    S.best_cost = 1e18;
    S.cost_history.clear();
    S.cost_history.reserve(S.total_frames_phase1 + S.total_frames_phase2);

    std::cout << "[Ctrl+Alt+W] sweep started.  Phase 1: "
              << S.n_target << "×" << S.n_source << "×" << S.n_rotation
              << " = " << S.total_candidates << " candidates,  "
              << S.candidates_per_frame << " per frame,  ~"
              << S.total_frames_phase1 << " frames"
              << std::endl;
    return true;
}


// ---------------------------------------------------------------------
// initPhase2InternalContourSweep
//   Refine discretization around Phase 1 best.
// ---------------------------------------------------------------------
inline void initPhase2InternalContourSweep()
{
    auto& S = g_contourSweepState;

    // Phase 1 best position in t ∈ [0, 1]
    const float t_tgt_best = (float(S.phase1_best_i_tgt) + 0.5f)
                                 / float(S.n_target);
    const float t_src_best = (float(S.phase1_best_j_src) + 0.5f)
                                 / float(S.n_source);

    // Phase 2: re-anchor with narrower range (±1 phase-1 step)
    S.phase2_radius_target_t = 1.0f / float(S.n_target);
    S.phase2_radius_source_t = 1.0f / float(S.n_source);
    S.phase2_radius_rot_deg  =
        std::max(2.0f, 360.0f / float(S.n_rotation));

    // Sector-based Phase 2 target anchor refinement.
    //   Use angular subrange centered on the Phase 1 best sector.
    //   half_range = full sector size of Phase 1 → ±1 sector slack.
    {
        const glm::mat4 view_p2 = buildSilhouetteView();
        const glm::mat4 proj_p2 = buildSilhouetteProj();
        const int W_p2 = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
        const int H_p2 = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
        const float kTwoPi = 6.283185307179586f;
        float center_angle = S.phase1_best_center_angle;
        if (center_angle <= 0.0f && S.n_target > 0) {
            center_angle = (float(S.phase1_best_i_tgt) + 0.5f)
                              * (kTwoPi / float(S.n_target));
        }
        const float half_range = (kTwoPi / float(S.n_target));

        // Plan A: pass FULL boundary subrange (no pre-filter); the
        // bbox tag step further down handles the spatial gate.
        RimShape::extractSectorBasedTargetAnchors3DSubrange(
            g_debugTargetBoundaryPoints,
            view_p2, proj_p2, W_p2, H_p2,
            center_angle, half_range,
            S.n_target,
            S.tgt_anchors_3D,
            /*full_boundary_3D_for_centroid=*/ nullptr);
        RimShape::project3DAnchorsTo2D(
            S.tgt_anchors_3D, view_p2, proj_p2, W_p2, H_p2,
            S.tgt_anchors_2D);

        // Re-tag bbox membership for the new anchor set.
        if (g_shapeMatchSweepFilterByRim
            && !g_debugSourceRimChain.empty() && liverMesh3D)
        {
            std::vector<glm::vec3> src_rim_3D;
            src_rim_3D.reserve(g_debugSourceRimChain.size());
            const auto& V = liverMesh3D->mVertices;
            const int nV3 = (int)V.size();
            for (int idx : g_debugSourceRimChain) {
                if (idx * 3 + 2 < nV3)
                    src_rim_3D.emplace_back(V[idx*3], V[idx*3+1], V[idx*3+2]);
            }
            RimShape::tagAnchorsInsideSourceBbox(
                S.tgt_anchors_3D, src_rim_3D,
                view_p2, proj_p2, W_p2, H_p2,
                g_shapeMatchSweepFilterMarginPx,
                S.tgt_anchor_inside_bbox);
        } else {
            S.tgt_anchor_inside_bbox.assign(S.tgt_anchors_3D.size(), (uint8_t)1);
        }

        std::cout << "[Ctrl+Alt+W] Phase 2 sector subrange: center="
                  << center_angle * 57.29578f << "°  half_range="
                  << half_range * 57.29578f << "°  anchors="
                  << S.tgt_anchors_3D.size() << std::endl;
    }
    // Phase 2 must use the SAME (possibly reversed) chain that Phase 1
    // used so t_src_best (a parameter in Phase 1's chain coordinate) is
    // interpreted correctly.
    const std::vector<int>& chain_for_phase2 =
        S.src_rim_chain_used.empty() ? g_debugSourceRimChain
                                     : S.src_rim_chain_used;
    std::vector<int> phase2_src_vert_idx;
    RimShape::arclengthResampleRim3DSubrange(
        chain_for_phase2, liverMesh3D->mVertices,
        t_src_best, S.phase2_radius_source_t,
        S.n_source, S.src_pivots_3D,
        &phase2_src_vert_idx);

    // Refresh anchor visualization for Phase 2: sector-based extraction
    // gave us 3D anchors directly; just copy them out for rendering.
    g_contourSweepTgtAnchors3D = S.tgt_anchors_3D;

    // -----------------------------------------------------------------
    // Phase 7b Step 3c++: Recompute label arrays + screen-side for the
    // narrower Phase-2 set. Source mesh has NOT moved (sweep applies
    // best T only in finishContourSweep), so anatomical labels carry
    // over via vertex index; target-anchor screen-side is recomputed
    // since the anchors are different points.
    // -----------------------------------------------------------------
    if (S.orientation_lock_active)
    {
        const bool lr_ready =
            g_liverLR.valid()
            && (int)g_liverLR.labels.size()
                   == (int)(liverMesh3D->mVertices.size() / 3);
        const bool cc_ready =
            g_liverCC.valid()
            && (int)g_liverCC.labels.size()
                   == (int)(liverMesh3D->mVertices.size() / 3);

        S.src_pivot_lr_label.clear();
        S.src_pivot_cc_label.clear();
        S.src_pivot_lr_label.reserve(S.src_pivots_3D.size());
        S.src_pivot_cc_label.reserve(S.src_pivots_3D.size());
        if (lr_ready && cc_ready
            && phase2_src_vert_idx.size() == S.src_pivots_3D.size())
        {
            for (int vi : phase2_src_vert_idx) {
                if (vi < 0 || vi >= (int)g_liverLR.labels.size()
                           || vi >= (int)g_liverCC.labels.size())
                {
                    S.src_pivot_lr_label.push_back(
                        (uint8_t)LiverLeftRightLabel::BOUNDARY);
                    S.src_pivot_cc_label.push_back(
                        (uint8_t)LiverCranioCaudalLabel::CAUDAL);
                    continue;
                }
                S.src_pivot_lr_label.push_back(g_liverLR.labels[vi]);
                S.src_pivot_cc_label.push_back(g_liverCC.labels[vi]);
            }
        } else {
            S.src_pivot_lr_label.assign(
                S.src_pivots_3D.size(),
                (uint8_t)LiverLeftRightLabel::BOUNDARY);
            S.src_pivot_cc_label.assign(
                S.src_pivots_3D.size(),
                (uint8_t)LiverCranioCaudalLabel::CAUDAL);
        }

        const float band = std::max(0.0f, g_shapeMatchSweepNeutralBandPx);
        S.tgt_anchor_screen_lr.clear();
        S.tgt_anchor_screen_cc.clear();
        S.tgt_anchor_screen_lr.reserve(S.tgt_anchors_2D.size());
        S.tgt_anchor_screen_cc.reserve(S.tgt_anchors_2D.size());
        for (const auto& a2d : S.tgt_anchors_2D) {
            const float dx = a2d.x - S.locked_centroid_2D.x;
            const float dy = a2d.y - S.locked_centroid_2D.y;
            uint8_t lr = RimShape::SS_LR_AMBIGUOUS;
            if      (dx >  band) lr = RimShape::SS_LR_RIGHT;
            else if (dx < -band) lr = RimShape::SS_LR_LEFT;
            uint8_t cc = RimShape::SS_CC_AMBIGUOUS;
            if      (dy < -band) cc = RimShape::SS_CC_CRANIAL;
            else if (dy >  band) cc = RimShape::SS_CC_CAUDAL;
            S.tgt_anchor_screen_lr.push_back(lr);
            S.tgt_anchor_screen_cc.push_back(cc);
        }
    }

    S.phase = 2;
    S.candidate_idx = 0;
    S.current_frame = 0;
    S.total_candidates = S.n_target * S.n_source * S.n_rotation;
    S.candidates_per_frame =
        std::max(1, (S.total_candidates + S.total_frames_phase2 - 1)
                       / S.total_frames_phase2);

    std::cout << "[Ctrl+Alt+W] Phase 2 start.  fine around"
              << "  t_tgt=" << t_tgt_best
              << "  t_src=" << t_src_best
              << "  θ=" << S.phase1_best_theta_deg << "°"
              << "  rot_radius=" << S.phase2_radius_rot_deg << "°"
              << std::endl;
}


// ---------------------------------------------------------------------
// tickContourSweep
//   Process one frame's worth of candidates. Returns true if more
//   work to do (caller should keep calling), false if done.
// ---------------------------------------------------------------------
inline bool tickContourSweep()
{
    auto& S = g_contourSweepState;
    if (!S.active) return false;
    if (S.phase < 1 || S.phase > 2) { S.active = false; return false; }

    const glm::mat4 view = buildSilhouetteView();
    const glm::mat4 proj = buildSilhouetteProj();
    const int W = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
    const int H = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;

    const int batch_start = S.candidate_idx;
    // When animation is OFF, finish the phase in a single tick (no
    // per-frame batching). Otherwise honor candidates_per_frame.
    const int batch_end   = g_shapeMatchSweepAnimate
        ? std::min(S.candidate_idx + S.candidates_per_frame, S.total_candidates)
        : S.total_candidates;

    int n_evaluated = 0;
    int n_improved  = 0;
    int n_skipped_endpoint = 0;
    int n_skipped_label    = 0;   // Phase 7b Step 3c++: Check A label LR/CC
    int n_skipped_rotation = 0;   // Phase 7b Step 3c++: Check B θ cap
    int n_skipped_bbox     = 0;   // Phase 7b Step 3c++: Plan A bbox tag
    // Pre-compute Check B threshold in radians once per batch.
    const float kPi    = 3.141592653589793f;
    const float kTwoPi = 6.283185307179586f;
    const float rot_lock_rad =
        g_shapeMatchSweepRotationLockDeg * (kPi / 180.0f);
    // Track best-in-batch separately so we can show a trial pose
    // (yellow) every frame even if no global improvement happened.
    glm::mat4 best_in_batch_T = S.best_T;
    double    best_in_batch_cost = 1e18;
    int       best_in_batch_i_tgt = -1;
    int       best_in_batch_j_src = -1;
    for (int c = batch_start; c < batch_end; c++) {
        // Decode (i_tgt outermost, j_src middle, k_rot innermost) so the
        // animation sweeps rotations fastest, source pivots next, target
        // anchors slowest — visually intuitive.
        const int i_tgt = c / (S.n_source * S.n_rotation);
        const int j_src = (c / S.n_rotation) % S.n_source;
        const int k_rot = c % S.n_rotation;

        // Endpoint constraint (Phase 1 only). After Phase 1's resample
        // (possibly with reversed chain), i_tgt=0 and j_src=0 both
        // correspond to "end A" of the curve. So |j_src - i_tgt|
        // measures the arc-length offset of the implied correspondence.
        // tolerance=0 means strict diagonal, tolerance=N means no
        // constraint.
        if (S.phase == 1 && S.endpoint_constraint_active) {
            if (std::abs(j_src - i_tgt) > g_shapeMatchSweepEndpointTolerance) {
                n_skipped_endpoint++;
                continue;
            }
        }

        // -------------------------------------------------------------
        // Phase 7b Step 3c++ Plan A — Check C: bbox active-mask.
        // Anchors are FIXED (target-only); the source-bbox filter is
        // applied here as a per-anchor skip. Source movement updates
        // S.tgt_anchor_inside_bbox at sweep start but leaves the anchor
        // positions themselves alone.
        // -------------------------------------------------------------
        if (g_shapeMatchSweepFilterByRim
            && i_tgt < (int)S.tgt_anchor_inside_bbox.size()
            && !S.tgt_anchor_inside_bbox[i_tgt])
        {
            n_skipped_bbox++;
            continue;
        }

        // -------------------------------------------------------------
        // Phase 7b Step 3c++ — Check A: label-based LR/CC orientation
        // lock. Source pivot's anatomical PCA label vs target anchor's
        // screen-side classification. BOUNDARY (source LR) and
        // AMBIGUOUS (target screen) always pass — those are the
        // designed-in "neutral zones" that handle 屈曲帯 / midline.
        // -------------------------------------------------------------
        if (S.orientation_lock_active
            && j_src < (int)S.src_pivot_lr_label.size()
            && i_tgt < (int)S.tgt_anchor_screen_lr.size())
        {
            const uint8_t lr_s = S.src_pivot_lr_label[j_src];
            const uint8_t lr_t = S.tgt_anchor_screen_lr[i_tgt];
            const uint8_t cc_s = S.src_pivot_cc_label[j_src];
            const uint8_t cc_t = S.tgt_anchor_screen_cc[i_tgt];

            // LR mismatch (PURE_RIGHT ↔ SS_LR_LEFT or vice versa)
            const bool lr_bad =
                (lr_s == (uint8_t)LiverLeftRightLabel::PURE_RIGHT
                    && lr_t == RimShape::SS_LR_LEFT)
             || (lr_s == (uint8_t)LiverLeftRightLabel::PURE_LEFT
                    && lr_t == RimShape::SS_LR_RIGHT);
            // CC mismatch (CRANIAL ↔ SS_CC_CAUDAL or vice versa)
            const bool cc_bad =
                (cc_s == (uint8_t)LiverCranioCaudalLabel::CRANIAL
                    && cc_t == RimShape::SS_CC_CAUDAL)
             || (cc_s == (uint8_t)LiverCranioCaudalLabel::CAUDAL
                    && cc_t == RimShape::SS_CC_CRANIAL);

            if (lr_bad || cc_bad) {
                n_skipped_label++;
                continue;
            }
        }

        float theta_deg = 0.0f;
        if (S.phase == 1) {
            theta_deg = float(k_rot) * 360.0f / float(S.n_rotation);
        } else {
            // Phase 2: ±radius_rot_deg around phase 1 best
            const float t = (float(k_rot) + 0.5f) / float(S.n_rotation);
            theta_deg = S.phase1_best_theta_deg
                          + (2.0f * t - 1.0f) * S.phase2_radius_rot_deg;
        }

        // -------------------------------------------------------------
        // Phase 7b Step 3c++ — Check B: θ_rotation magnitude cap.
        // Independent secondary defense: even if Check A passes (LR-
        // symmetric rim shapes can fool it), a θ magnitude above the
        // cap is exactly the image-plane flip the user is complaining
        // about. Applies in Phase 1 AND Phase 2.
        // -------------------------------------------------------------
        if (S.orientation_lock_active) {
            float th_rad = theta_deg * (kPi / 180.0f);
            while (th_rad >  kPi) th_rad -= kTwoPi;
            while (th_rad < -kPi) th_rad += kTwoPi;
            if (std::abs(th_rad) > rot_lock_rad) {
                n_skipped_rotation++;
                continue;
            }
        }

        glm::mat4 T;
        const double cost = RimShape::evaluateContourSweepCandidate(
            S.tgt_anchors_2D, S.src_pivots_3D,
            i_tgt, j_src, theta_deg,
            view, proj, W, H,
            g_gnUnsignedBdy, g_gnUnsignedBdyW, g_gnUnsignedBdyH,
            T);
        n_evaluated++;

        if (cost < best_in_batch_cost) {
            best_in_batch_cost  = cost;
            best_in_batch_T     = T;
            best_in_batch_i_tgt = i_tgt;
            best_in_batch_j_src = j_src;
        }

        if (cost < S.best_cost) {
            S.best_cost      = cost;
            S.best_T         = T;
            S.best_i_tgt     = i_tgt;
            S.best_j_src     = j_src;
            S.best_theta_deg = theta_deg;
            n_improved++;
        }
    }
    S.candidate_idx = batch_end;
    S.cost_history.push_back(S.best_cost);
    S.current_frame++;

    // ---- Trial-pose visualization (yellow) ------------------------
    // Updated EVERY frame regardless of global-best improvement so the
    // mesh always shows motion during the sweep. Best-in-batch may be
    // worse than the global best (since the global best from earlier
    // batches isn't redone), but that's the point — the user sees what
    // is currently being tried at this batch.
    if (best_in_batch_cost < 1e17) {
        g_contourSweepTrialSrc.clear();
        g_contourSweepTrialSrc.reserve(g_debugSourceRimChain.size());
        const auto& V = liverMesh3D->mVertices;
        const int nV3 = (int)V.size();
        for (int idx : g_debugSourceRimChain) {
            if (idx < 0 || idx * 3 + 2 >= nV3) continue;
            const glm::vec4 p4 = best_in_batch_T *
                glm::vec4(V[idx*3], V[idx*3+1], V[idx*3+2], 1.0f);
            g_contourSweepTrialSrc.emplace_back(p4.x, p4.y, p4.z);
        }
        g_contourSweepShowTrial = true;

        // Source pivot trial positions: 20 pivots transformed by
        // best-in-batch T so they land on/around the yellow trial rim.
        g_contourSweepSrcPivotsTrial.clear();
        g_contourSweepSrcPivotsTrial.reserve(S.src_pivots_3D.size());
        for (const auto& p : S.src_pivots_3D) {
            const glm::vec4 p4 = best_in_batch_T * glm::vec4(p, 1.0f);
            g_contourSweepSrcPivotsTrial.emplace_back(p4.x, p4.y, p4.z);
        }
        // Current correspondence indices for cyan highlight
        g_contourSweepCurrentITgt = best_in_batch_i_tgt;
        g_contourSweepCurrentJSrc = best_in_batch_j_src;
    }

    // ---- Global-best visualization (red) --------------------------
    // Updated only when a new global best is found, so the user sees
    // a stable "current converged answer" that snaps forward on
    // improvements.
    if (n_improved > 0) {
        g_debugShapeMatchBestSrc.clear();
        g_debugShapeMatchBestSrc.reserve(g_debugSourceRimChain.size());
        const auto& V = liverMesh3D->mVertices;
        const int nV3 = (int)V.size();
        for (int idx : g_debugSourceRimChain) {
            if (idx < 0 || idx * 3 + 2 >= nV3) continue;
            const glm::vec4 p4 = S.best_T *
                glm::vec4(V[idx*3], V[idx*3+1], V[idx*3+2], 1.0f);
            g_debugShapeMatchBestSrc.emplace_back(p4.x, p4.y, p4.z);
        }
        g_debugShapeMatchBestTransform = S.best_T;
        g_debugShapeMatchBestCost      = S.best_cost;
        g_showDebugShapeMatch          = true;  // ensure red dots visible
    }

    if (g_shapeMatchSweepLog) {
        // Throttle to keep terminal readable on 600-frame sweeps:
        //   - always log when improved (something interesting happened)
        //   - always log on the last batch of the phase (sets up
        //     "Phase X done." line nicely)
        //   - otherwise every 30 frames so progress is still visible
        const bool is_last_batch = (S.candidate_idx >= S.total_candidates);
        const bool periodic      = (S.current_frame % 30 == 0);
        const bool worth_logging = (n_improved > 0) || is_last_batch || periodic;
        if (worth_logging) {
            std::cout << "[Ctrl+Alt+W] phase=" << S.phase
                      << "  frame=" << S.current_frame
                      << "  c=" << batch_start << "-" << batch_end
                      << "/" << S.total_candidates
                      << "  best=" << S.best_cost << "px"
                      << "  i=" << S.best_i_tgt
                      << "  j=" << S.best_j_src
                      << "  θ=" << S.best_theta_deg
                      << "  improved+" << n_improved << "/" << n_evaluated
                      << "  ep_skip=" << n_skipped_endpoint
                      << "  lbl_skip=" << n_skipped_label
                      << "  rot_skip=" << n_skipped_rotation
                      << "  bbox_skip=" << n_skipped_bbox
                      << std::endl;
        }
    }

    // Phase transition
    if (S.candidate_idx >= S.total_candidates) {
        if (S.phase == 1) {
            S.phase1_best_i_tgt     = S.best_i_tgt;
            S.phase1_best_j_src     = S.best_j_src;
            S.phase1_best_theta_deg = S.best_theta_deg;
            // Recover the angular center of the best sector by
            // re-projecting the best 3D anchor and computing its
            // angle from the boundary centroid. This is what Phase 2
            // re-sectors around.
            //
            // Plan A simplification: S.locked_centroid_2D already holds
            // the full-boundary centroid (target-only, source-invariant)
            // from Phase 1 setup. No re-extraction needed.
            if (S.best_i_tgt >= 0
                && S.best_i_tgt < (int)S.tgt_anchors_3D.size())
            {
                const glm::mat4 view_t = buildSilhouetteView();
                const glm::mat4 proj_t = buildSilhouetteProj();
                const int W_t = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
                const int H_t = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
                std::vector<glm::vec2> best_anchor_2D;
                RimShape::project3DAnchorsTo2D(
                    { S.tgt_anchors_3D[S.best_i_tgt] },
                    view_t, proj_t, W_t, H_t, best_anchor_2D);
                if (!best_anchor_2D.empty()) {
                    const glm::vec2 d =
                        best_anchor_2D[0] - S.locked_centroid_2D;
                    float theta = std::atan2(d.y, d.x);
                    const float kTwoPi = 6.283185307179586f;
                    if (theta < 0.0f) theta += kTwoPi;
                    S.phase1_best_center_angle = theta;
                }
            }
            std::cout << "[Ctrl+Alt+W] Phase 1 done.  best cost=" << S.best_cost
                      << " at (i=" << S.best_i_tgt
                      << ", j=" << S.best_j_src
                      << ", θ=" << S.best_theta_deg << "°)"
                      << "  center_angle="
                      << S.phase1_best_center_angle * 57.29578f << "°"
                      << std::endl;
            initPhase2InternalContourSweep();
            return true;
        } else {
            std::cout << "[Ctrl+Alt+W] Phase 2 done.  final cost="
                      << S.best_cost
                      << " at (i=" << S.best_i_tgt
                      << ", j=" << S.best_j_src
                      << ", θ=" << S.best_theta_deg << "°)"
                      << std::endl;
            S.phase = 3;       // done flag
            S.active = false;
            return false;
        }
    }
    return true;
}


// ---------------------------------------------------------------------
// finishContourSweep
//   Sweep completion lives in main.cpp because the post-apply pose-
//   save flow uses gUI / g_stepStartTime / SaveCriterion / poseSave*
//   which are defined in main.cpp / PoseLibrary.h (included AFTER this
//   header). The actual logic mirrors Ctrl+W / Alt+W tail exactly;
//   see the tickContourSweep path inside main.cpp's render loop.
// ---------------------------------------------------------------------


// =====================================================================
// Phase 7b Step 3d: Silhouette 2D Dense Sweep — sweep functions
//
//   startSilhouetteSweep()  — Phase 0: project source rim, find right
//                             start, re-orient clockwise, resample 20
//                             pivots; build target lower-half, find
//                             right end, resample 20 anchors.
//   tickSilhouetteSweep()   — Phase 1 (720 cands) → Phase 2 (180 cands).
//                             Returns true while sweep is ongoing.
//   finishSilhouetteSweep() — adopt best T, apply to organ meshes,
//                             save to PoseLibrary. Lives in main.cpp
//                             (same reason as finishContourSweep).
//
//   The mesh is NOT touched during ticks — only g_debugShapeMatchBestSrc
//   updates so the user sees the best-so-far pose. Mesh is moved exactly
//   once, on finishSilhouetteSweep.
// =====================================================================

// ---------------------------------------------------------------------
// silSwBuildDenseSrcRim3DOriented
//   Helper: given a chain, an ordered list of original-chain indices
//   (output of rotateAndOrientCurve's parallel_index_array), and the
//   liver mesh, build a dense 3D point vector matching that order.
// ---------------------------------------------------------------------
inline void silSwBuildDenseSrcRim3DOriented(
    const std::vector<int>& rim_chain,
    const std::vector<int>& chain_order_into_rim_chain,
    const std::vector<float>& mesh_verts,
    std::vector<glm::vec3>& out_dense_3D)
{
    out_dense_3D.clear();
    const int nV3 = (int)mesh_verts.size();
    const int K   = (int)chain_order_into_rim_chain.size();
    out_dense_3D.reserve(K);
    for (int i = 0; i < K; i++) {
        const int chain_pos = chain_order_into_rim_chain[i];
        if (chain_pos < 0 || chain_pos >= (int)rim_chain.size()) {
            out_dense_3D.emplace_back(0.0f);   // sentinel; should not happen
            continue;
        }
        const int vidx = rim_chain[chain_pos];
        if (vidx < 0 || vidx*3 + 2 >= nV3) {
            out_dense_3D.emplace_back(0.0f);
            continue;
        }
        out_dense_3D.emplace_back(mesh_verts[vidx*3],
                                  mesh_verts[vidx*3+1],
                                  mesh_verts[vidx*3+2]);
    }
}

// ---------------------------------------------------------------------
// =====================================================================
// Phase 7b Step 3d Stage 0.1/0.2 — Shared smoothed RIM 2D builder
// =====================================================================
// buildSmoothedRim2D
//   Helper used by CB0.2 popup (and intended for CB0.1 popup refactor).
//   Pure function of:
//     - liverMesh3D vertex coords (current source pose)
//     - g_debugSourceRimChain      (W-key rim chain)
//     - g_liverLR                  (per-vertex left/right label)
//     - AR camera view + projection (buildSilhouetteView/Proj)
//     - grid_px, knn_k, knn_iters  (smoothing parameters)
//
//   Pipeline:
//     1. Auto-populate g_debugSourceRimChain if empty
//     2. Project every chain vertex to 2D pixel coords (drop clip.w<=0)
//     3. Bin into grid_px × grid_px cells; emit centroid + majority LR
//     4. Optional KNN smoothing (k nearest + self, iterated)
//     5. Compute 2D centroid + PURE_RIGHT 3D centroid → 2D
//
//   Returns SmoothedRim2DResult with .ok set; caller checks .ok and
//   .fail_reason before consuming the point arrays.
// ---------------------------------------------------------------------
struct SmoothedRim2DResult {
    bool        ok = false;
    std::string fail_reason;

    // Raw projected points (pre-smoothing), parallel arrays
    std::vector<glm::vec2> raw_pts;
    std::vector<uint8_t>   raw_lr;
    std::vector<glm::vec3> raw_pts_3D;     // mesh-space, parallel to raw_pts
    std::vector<int>       raw_vidx;       // vertex idx, parallel to raw_pts

    // Cleaned (grid + KNN) points, parallel arrays + cell occupancy
    std::vector<glm::vec2> smo_pts;
    std::vector<uint8_t>   smo_lr;
    std::vector<int>       smo_cnt;
    std::vector<glm::vec3> smo_pts_3D;     // mesh-space centroid per cell
                                           // (KNN-smoothed in lockstep with smo_pts)
    std::vector<int>       smo_repr_vidx;  // representative vertex idx per cell
                                           // (first vidx that fell in the bin)

    int W_img = 1920;
    int H_img = 1080;

    glm::vec2 smoothed_centroid_2D = glm::vec2(0.0f);
    glm::vec2 right_centroid_2D    = glm::vec2(-1e6f);
    bool      right_centroid_valid = false;
};

inline SmoothedRim2DResult buildSmoothedRim2D(
    float grid_px, int knn_k, int knn_iters)
{
    SmoothedRim2DResult R;

    if (!liverMesh3D) {
        R.fail_reason = "liverMesh3D is null";
        return R;
    }
    if (g_debugSourceRimChain.empty()) {
        if (!populateDebugSourceRimChain()) {
            R.fail_reason = "populateDebugSourceRimChain failed (press W first)";
            return R;
        }
    }
    if (g_debugSourceRimChain.empty()) {
        R.fail_reason = "g_debugSourceRimChain still empty";
        return R;
    }

    const glm::mat4 view_m = buildSilhouetteView();
    const glm::mat4 proj_m = buildSilhouetteProj();
    R.W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
    R.H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
    const glm::mat4 M = proj_m * view_m;

    const auto& Vmesh = liverMesh3D->mVertices;
    const int nV3 = (int)Vmesh.size();
    const bool lrValid = g_liverLR.valid();
    const int  nLR     = lrValid ? (int)g_liverLR.labels.size() : 0;

    // --- Step 1: project all rim chain to 2D ---
    R.raw_pts.reserve(g_debugSourceRimChain.size());
    R.raw_lr.reserve(g_debugSourceRimChain.size());
    R.raw_pts_3D.reserve(g_debugSourceRimChain.size());
    R.raw_vidx.reserve(g_debugSourceRimChain.size());
    for (int idx : g_debugSourceRimChain) {
        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
        const glm::vec3 p3(Vmesh[idx*3], Vmesh[idx*3+1], Vmesh[idx*3+2]);
        const glm::vec4 clip = M * glm::vec4(p3, 1.0f);
        if (clip.w < 1e-9f) continue;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        R.raw_pts.emplace_back(
            (ndcx + 1.0f) * 0.5f * float(R.W_img),
            (1.0f - ndcy) * 0.5f * float(R.H_img));
        uint8_t lr = 255;
        if (lrValid && idx < nLR) lr = g_liverLR.labels[idx];
        R.raw_lr.push_back(lr);
        R.raw_pts_3D.push_back(p3);
        R.raw_vidx.push_back(idx);
    }
    const int N_raw = (int)R.raw_pts.size();
    if (N_raw == 0) {
        R.fail_reason = "no points projected on screen";
        return R;
    }

    // --- Step 2: grid aggregation ---
    const float grid = std::max(1.0f, grid_px);
    struct Cell {
        glm::dvec2 sum;
        glm::dvec3 sum_3D;
        int        cnt;
        int        n_R, n_L, n_B, n_U;
        int        first_vidx;
    };
    auto cellKey = [grid](float x, float y) -> long long {
        const int cx = int(std::floor(x / grid));
        const int cy = int(std::floor(y / grid));
        const long long ux = (long long)(cx + (1 << 15));
        const long long uy = (long long)(cy + (1 << 15));
        return (uy << 20) | (ux & 0xFFFFFLL);
    };
    std::unordered_map<long long, Cell> bins;
    bins.reserve(size_t(N_raw));
    for (int i = 0; i < N_raw; i++) {
        const long long k = cellKey(R.raw_pts[i].x, R.raw_pts[i].y);
        auto& C = bins[k];
        if (C.cnt == 0) {
            C.sum = glm::dvec2(0.0);
            C.sum_3D = glm::dvec3(0.0);
            C.n_R = C.n_L = C.n_B = C.n_U = 0;
            C.first_vidx = R.raw_vidx[i];
        }
        C.sum    += glm::dvec2(R.raw_pts[i]);
        C.sum_3D += glm::dvec3(R.raw_pts_3D[i]);
        C.cnt++;
        switch (R.raw_lr[i]) {
            case LiverLeftRightLabel::PURE_RIGHT: C.n_R++; break;
            case LiverLeftRightLabel::PURE_LEFT:  C.n_L++; break;
            case LiverLeftRightLabel::BOUNDARY:   C.n_B++; break;
            default:                              C.n_U++; break;
        }
    }
    R.smo_pts.reserve(bins.size());
    R.smo_lr.reserve(bins.size());
    R.smo_cnt.reserve(bins.size());
    R.smo_pts_3D.reserve(bins.size());
    R.smo_repr_vidx.reserve(bins.size());
    for (auto& kv : bins) {
        const Cell& C = kv.second;
        R.smo_pts.emplace_back(C.sum / double(C.cnt));
        R.smo_pts_3D.emplace_back(C.sum_3D / double(C.cnt));
        R.smo_repr_vidx.push_back(C.first_vidx);
        uint8_t maj = 255;
        int best = -1;
        if (C.n_R > best) { best = C.n_R; maj = LiverLeftRightLabel::PURE_RIGHT; }
        if (C.n_L > best) { best = C.n_L; maj = LiverLeftRightLabel::PURE_LEFT;  }
        if (C.n_B > best) { best = C.n_B; maj = LiverLeftRightLabel::BOUNDARY;   }
        if (C.n_U > best) { best = C.n_U; maj = 255; }
        R.smo_lr.push_back(maj);
        R.smo_cnt.push_back(C.cnt);
    }
    const int N_cells = (int)R.smo_pts.size();

    // --- Step 3: KNN smoothing ---
    const int K     = std::max(0, knn_k);
    const int iters = std::max(0, knn_iters);
    if (K > 0 && iters > 0 && N_cells >= 2) {
        const int Keff = std::min(K, N_cells - 1);
        std::vector<glm::vec2> tmp(N_cells);
        std::vector<glm::vec3> tmp3D(N_cells);
        std::vector<std::pair<float, int>> dist_idx;
        dist_idx.reserve(N_cells);
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < N_cells; i++) {
                dist_idx.clear();
                for (int j = 0; j < N_cells; j++) {
                    if (j == i) continue;
                    const float dx = R.smo_pts[j].x - R.smo_pts[i].x;
                    const float dy = R.smo_pts[j].y - R.smo_pts[i].y;
                    dist_idx.emplace_back(dx*dx + dy*dy, j);
                }
                std::nth_element(
                    dist_idx.begin(),
                    dist_idx.begin() + Keff,
                    dist_idx.end(),
                    [](const std::pair<float,int>& a,
                       const std::pair<float,int>& b){
                        return a.first < b.first;
                    });
                glm::dvec2 sum2(0.0);
                glm::dvec3 sum3(0.0);
                for (int n = 0; n < Keff; n++) {
                    const int j = dist_idx[n].second;
                    sum2 += glm::dvec2(R.smo_pts[j]);
                    sum3 += glm::dvec3(R.smo_pts_3D[j]);
                }
                sum2 += glm::dvec2(R.smo_pts[i]);     // self prevents drift
                sum3 += glm::dvec3(R.smo_pts_3D[i]);
                tmp  [i] = glm::vec2(sum2 / double(Keff + 1));
                tmp3D[i] = glm::vec3(sum3 / double(Keff + 1));
            }
            R.smo_pts   .swap(tmp);
            R.smo_pts_3D.swap(tmp3D);
        }
    }

    // --- Step 4: centroids ---
    glm::dvec2 sum2d(0.0);
    for (const auto& p : R.smo_pts) sum2d += glm::dvec2(p);
    R.smoothed_centroid_2D = R.smo_pts.empty()
        ? glm::vec2(float(R.W_img) * 0.5f, float(R.H_img) * 0.5f)
        : glm::vec2(sum2d / double(R.smo_pts.size()));

    if (lrValid) {
        glm::dvec3 sumR(0.0);
        int nR3 = 0;
        for (int i = 0; i < nLR; i++) {
            if (i * 3 + 2 >= nV3) break;
            if (g_liverLR.labels[i] != LiverLeftRightLabel::PURE_RIGHT) continue;
            sumR += glm::dvec3(Vmesh[i*3], Vmesh[i*3+1], Vmesh[i*3+2]);
            nR3++;
        }
        if (nR3 > 0) {
            const glm::vec3 rc3(sumR / double(nR3));
            const glm::vec4 clip = M * glm::vec4(rc3, 1.0f);
            if (clip.w > 1e-9f) {
                const float ndcx = clip.x / clip.w;
                const float ndcy = clip.y / clip.w;
                R.right_centroid_2D = glm::vec2(
                    (ndcx + 1.0f) * 0.5f * float(R.W_img),
                    (1.0f - ndcy) * 0.5f * float(R.H_img));
                R.right_centroid_valid = true;
            }
        }
    }

    R.ok = true;
    return R;
}

// ---------------------------------------------------------------------
// buildOrderedRim2D — Stage 0.2 ordering via MST + longest path
// ---------------------------------------------------------------------
//   Input:  cleaned 2D point set (from buildSmoothedRim2D) + start hint
//           (PURE_RIGHT 3D centroid projected to 2D, optional)
//   Output: oriented path + arc-length resampled pivots
//
//   Pipeline:
//     1. Prim's MST O(N²) on full pairwise 2D distances
//        Edges longer than max_edge_px are excluded (set to +inf)
//        → may disconnect the graph; we keep the LARGEST CC only.
//     2. Two-pass BFS on the chosen CC:
//          a. BFS from arbitrary seed → farthest node = endpoint A
//          b. BFS from endpoint A     → farthest node = endpoint B
//          → path = endpoint B ← ... ← endpoint A (via parent[])
//     3. Orient: if start hint valid, ensure path[0] is the endpoint
//        closer (in 2D) to the start hint; else leave as-is.
//     4. Arc-length resample n_pivots evenly along the oriented path.
//
//   Open-curve safe: the longest path in a tree is, by definition, an
//   open chain — no forced loop closure. This is the correct topology
//   for the caudal RIM arch we observed in CB0.
// ---------------------------------------------------------------------
struct OrderedRim2DResult {
    bool        ok = false;
    std::string fail_reason;

    // MST adjacency (indexed into input cleaned_pts; nodes outside the
    // largest CC still appear here but with edges only to their CC).
    std::vector<std::vector<int>> mst_adj;
    int   n_rejected_edges  = 0;
    float mst_total_length_px = 0.0f;

    // Indices in the largest connected component
    std::vector<int> component;

    // Oriented longest path (indices into input cleaned_pts)
    std::vector<int> path;
    int   endpoint_a = -1;
    int   endpoint_b = -1;
    float arc_length_px = 0.0f;
    bool  start_oriented_to_right = false;

    // Arc-length resampled pivots (size = n_pivots)
    std::vector<glm::vec2> pivots;
};

inline OrderedRim2DResult buildOrderedRim2D(
    const std::vector<glm::vec2>& cleaned_pts,
    const glm::vec2& right_centroid_2D,
    bool  right_centroid_valid,
    float max_edge_px,
    int   n_pivots)
{
    OrderedRim2DResult R;
    const int N = (int)cleaned_pts.size();
    if (N < 2) {
        R.fail_reason = "need at least 2 cleaned points";
        return R;
    }

    // --- 1. Prim's MST O(N²) with max_edge filter ---
    const float kInf = std::numeric_limits<float>::infinity();
    const float maxEdge2 = (max_edge_px > 0.0f)
        ? max_edge_px * max_edge_px : kInf;
    std::vector<float> minDist2(N, kInf);
    std::vector<int>   parent  (N, -1);
    std::vector<bool>  inTree  (N, false);

    minDist2[0] = 0.0f;
    for (int it = 0; it < N; it++) {
        int u = -1;
        float best = kInf;
        for (int v = 0; v < N; v++) {
            if (!inTree[v] && minDist2[v] < best) {
                best = minDist2[v];
                u = v;
            }
        }
        if (u < 0) break;            // remaining nodes unreachable under filter
        inTree[u] = true;
        for (int v = 0; v < N; v++) {
            if (inTree[v]) continue;
            const float dx = cleaned_pts[v].x - cleaned_pts[u].x;
            const float dy = cleaned_pts[v].y - cleaned_pts[u].y;
            const float d2 = dx*dx + dy*dy;
            if (d2 > maxEdge2) continue;
            if (d2 < minDist2[v]) {
                minDist2[v] = d2;
                parent[v]   = u;
            }
        }
    }

    R.mst_adj.assign(N, {});
    int n_edges_kept = 0;
    for (int v = 0; v < N; v++) {
        if (parent[v] < 0) continue;
        R.mst_adj[v].push_back(parent[v]);
        R.mst_adj[parent[v]].push_back(v);
        const float dx = cleaned_pts[v].x - cleaned_pts[parent[v]].x;
        const float dy = cleaned_pts[v].y - cleaned_pts[parent[v]].y;
        R.mst_total_length_px += std::sqrt(dx*dx + dy*dy);
        n_edges_kept++;
    }
    R.n_rejected_edges = (N - 1) - n_edges_kept;

    // --- 2. Largest connected component ---
    std::vector<int> comp_id(N, -1);
    std::vector<int> comp_size;
    for (int seed = 0; seed < N; seed++) {
        if (comp_id[seed] >= 0) continue;
        const int cid = (int)comp_size.size();
        int sz = 0;
        std::vector<int> stack = { seed };
        comp_id[seed] = cid;
        while (!stack.empty()) {
            int u = stack.back(); stack.pop_back();
            sz++;
            for (int v : R.mst_adj[u]) {
                if (comp_id[v] < 0) {
                    comp_id[v] = cid;
                    stack.push_back(v);
                }
            }
        }
        comp_size.push_back(sz);
    }
    int biggest_cid = 0;
    for (size_t i = 1; i < comp_size.size(); i++) {
        if (comp_size[i] > comp_size[biggest_cid]) biggest_cid = (int)i;
    }
    R.component.reserve(comp_size[biggest_cid]);
    for (int i = 0; i < N; i++) {
        if (comp_id[i] == biggest_cid) R.component.push_back(i);
    }
    if (R.component.size() < 2) {
        R.fail_reason = "largest CC has <2 nodes (max_edge_px too small?)";
        return R;
    }

    // --- 3. Two-pass BFS for longest path on the chosen CC ---
    auto bfs_farthest = [&](int src,
                            std::vector<int>& par_out,
                            std::vector<float>& dist_out) -> int {
        par_out.assign(N, -1);
        dist_out.assign(N, kInf);
        std::vector<int> q;
        q.reserve(R.component.size());
        dist_out[src] = 0.0f;
        q.push_back(src);
        size_t head = 0;
        while (head < q.size()) {
            int u = q[head++];
            for (int v : R.mst_adj[u]) {
                if (dist_out[v] != kInf) continue;
                par_out[v] = u;
                const float dx = cleaned_pts[v].x - cleaned_pts[u].x;
                const float dy = cleaned_pts[v].y - cleaned_pts[u].y;
                dist_out[v] = dist_out[u] + std::sqrt(dx*dx + dy*dy);
                q.push_back(v);
            }
        }
        int best = src;
        float best_d = 0.0f;
        for (int i : R.component) {
            if (dist_out[i] != kInf && dist_out[i] > best_d) {
                best_d  = dist_out[i];
                best    = i;
            }
        }
        return best;
    };

    std::vector<int>   par1, par2;
    std::vector<float> d1, d2;
    R.endpoint_a = bfs_farthest(R.component[0], par1, d1);
    R.endpoint_b = bfs_farthest(R.endpoint_a,   par2, d2);
    R.arc_length_px = d2[R.endpoint_b];

    // Reconstruct path: endpoint_b → ... → endpoint_a (via par2)
    std::vector<int> path;
    {
        int cur = R.endpoint_b;
        int guard = 0;
        while (cur != -1 && guard < N + 1) {
            path.push_back(cur);
            if (cur == R.endpoint_a) break;
            cur = par2[cur];
            guard++;
        }
        if (path.empty() || path.back() != R.endpoint_a) {
            R.fail_reason = "path reconstruction failed";
            return R;
        }
    }
    R.path = std::move(path);

    // --- 4. Orient by PURE_RIGHT proximity ---
    if (right_centroid_valid) {
        auto d2_from = [&](int idx) -> float {
            const float dx = cleaned_pts[idx].x - right_centroid_2D.x;
            const float dy = cleaned_pts[idx].y - right_centroid_2D.y;
            return dx*dx + dy*dy;
        };
        const float dF = d2_from(R.path.front());
        const float dB = d2_from(R.path.back());
        if (dB < dF) std::reverse(R.path.begin(), R.path.end());
        R.start_oriented_to_right = true;
    }

    // --- 5. Arc-length resample n_pivots ---
    const int Np = std::max(2, n_pivots);
    R.pivots.reserve(Np);
    if (R.path.size() == 1) {
        for (int i = 0; i < Np; i++)
            R.pivots.push_back(cleaned_pts[R.path[0]]);
    } else {
        std::vector<float> cum(R.path.size(), 0.0f);
        for (size_t i = 1; i < R.path.size(); i++) {
            const float dx = cleaned_pts[R.path[i]].x   - cleaned_pts[R.path[i-1]].x;
            const float dy = cleaned_pts[R.path[i]].y   - cleaned_pts[R.path[i-1]].y;
            cum[i] = cum[i-1] + std::sqrt(dx*dx + dy*dy);
        }
        const float total = cum.back();
        for (int p = 0; p < Np; p++) {
            const float t = float(p) / float(Np - 1);
            const float ss = t * total;
            auto it = std::upper_bound(cum.begin(), cum.end(), ss);
            int seg = (int)(it - cum.begin()) - 1;
            if (seg < 0) seg = 0;
            if (seg >= (int)R.path.size() - 1) {
                R.pivots.push_back(cleaned_pts[R.path.back()]);
                continue;
            }
            const float seg_len = cum[seg+1] - cum[seg];
            const float u = (seg_len > 1e-6f)
                          ? (ss - cum[seg]) / seg_len : 0.0f;
            const glm::vec2 a = cleaned_pts[R.path[seg]];
            const glm::vec2 b = cleaned_pts[R.path[seg+1]];
            R.pivots.push_back(a + u * (b - a));
        }
    }

    R.ok = true;
    return R;
}

// ---------------------------------------------------------------------
// silSwBuildSrcPreview
//   Rebuilds the source-side cache vars used by the popup window
//   (CB1) AND by Phase 0 of startSilhouetteSweep. Pure function of:
// ---------------------------------------------------------------------
// silSwBuildSrcPreview  [REVISED — Step 3d2]
//   Build the source-side preview using the SAME silhouette extraction
//   as Shift+E (runSilhouetteHemi):
//     1. extractVisibleVerticesCustom (BVH-based AR visibility)
//     2. |dot(normal, viewDir)| < g_silhouetteSrcCosThresh
//        → silhouette-edge vertices
//     3. Project to 2D using buildSilhouetteView/Proj
//     4. Compute 2D centroid of silhouette
//     5. atan2-sort around 2D centroid → smooth closed 2D contour
//     6. Find PURE_RIGHT 3D centroid, project to 2D, locate the
//        nearest silhouette point → start_idx
//     7. Rotate to put start_idx at index 0, enforce clockwise (next
//        step moves +y on screen)
//     8. arc-length resample 20 pivots
//
//   This replaces the previous g_debugSourceRimChain-based path which
//   yielded a jagged 2D curve (chain ordered by PCA-plane angle, not
//   by AR screen angle). The new curve is by construction the visible
//   silhouette as the AR camera actually sees it.
//
//   Caches populated:
//     g_silSwSrcRim2DPreview         (dense, sorted+oriented)
//     g_silSwSrcRim3DPreview         (dense, mesh-space, parallel)
//     g_silSwSrcRimVIdxPreview       (dense, vertex idx, parallel)
//     g_silSwSrcRim2DPreviewLR       (dense, LR label, parallel)
//     g_silSwSrcPivots2DPreview      (20)
//     g_silSwSrcPivotsLRPreview      (20)
//     g_silSwSrcStartIdxPreview      (index in sorted-not-rotated array)
//     g_silSwSrcRightCentroid2DPreview
// ---------------------------------------------------------------------
inline bool silSwBuildSrcPreview(std::string* out_fail_reason = nullptr)
{
    if (!liverMesh3D) {
        if (out_fail_reason) *out_fail_reason = "liverMesh3D is null";
        g_silSwSrcPreviewCacheValid = false;
        return false;
    }

    // ---- Step 3d2: pose-hash short-circuit -------------------------
    // BVH build + extractVisibleVerticesCustom is ~30-70ms and emits
    // log spam ("Camera position: ...") every call. Skip the entire
    // pipeline when the mesh hasn't moved since last build.
    //
    // Hash strategy: rigid transforms move ALL vertices by the same
    // T, so checking a handful of vertex coords (cast as raw bits)
    // is sufficient to detect any pose change. We sample 4 vertices
    // spread across the mesh-vertex array.
    {
        const auto& V = liverMesh3D->mVertices;
        const int nV3 = (int)V.size();
        if (nV3 >= 12) {
            uint64_t h = 1469598103934665603ULL;   // FNV-1a offset
            const int sample_offsets[4] = {
                0,
                std::max(0, nV3 / 4 - (nV3 / 4 % 3)),
                std::max(0, nV3 / 2 - (nV3 / 2 % 3)),
                std::max(0, (nV3 * 3 / 4) - ((nV3 * 3 / 4) % 3))
            };
            for (int s = 0; s < 4; s++) {
                const int o = sample_offsets[s];
                for (int k = 0; k < 3; k++) {
                    if (o + k >= nV3) continue;
                    uint32_t bits;
                    const float fv = V[o + k];
                    std::memcpy(&bits, &fv, sizeof(bits));
                    h ^= uint64_t(bits);
                    h *= 1099511628211ULL;          // FNV-1a prime
                }
            }
            // (No early return here — params get mixed in below and the
            // combined check decides cache validity.)
            g_silSwSrcPreviewCacheHash = h;
        }
    }

    // [Stage A] Mix the source-rim method + CB0.1/CB0.2 tuning parameters
    // into the cache key so that toggling the method or moving any slider
    // forces a rebuild rather than returning stale results. Done AFTER
    // the vertex hash above so the combined key reflects both pose and
    // parameter state.
    {
        uint64_t hp = g_silSwSrcPreviewCacheHash;
        auto mix32 = [&](uint32_t bits) {
            hp ^= uint64_t(bits);
            hp *= 1099511628211ULL;
        };
        auto mixF = [&](float v) {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            mix32(bits);
        };
        mix32(uint32_t(g_silSwSrcRimMethod));
        mixF (g_rawRimSmooth_GridPx);
        mix32(uint32_t(g_rawRimSmooth_KnnK));
        mix32(uint32_t(g_rawRimSmooth_KnnIters));
        mixF (g_rawRimOrder_MaxEdgePx);
        mix32(uint32_t(g_rawRimOrder_NPivots));
        if (g_silSwSrcPreviewCacheValid &&
            g_silSwSrcPreviewCacheHash == hp &&
            !g_silSwSrcRim2DPreview.empty())
        {
            return true;       // both verts AND params unchanged
        }
        g_silSwSrcPreviewCacheHash = hp;
    }

    g_silSwSrcRim2DPreview.clear();
    g_silSwSrcRim3DPreview.clear();
    g_silSwSrcRimVIdxPreview.clear();
    g_silSwSrcPivots2DPreview.clear();
    g_silSwSrcRim2DPreviewLR.clear();
    g_silSwSrcPivotsLRPreview.clear();
    g_silSwSrcPivots3DPreview.clear();          // [Stage A] open-path safe
    g_silSwSrcStartIdxPreview = -1;
    g_silSwSrcPreviewCacheValid = false;

    if (!g_liverLR.valid()) {
        if (out_fail_reason) *out_fail_reason = "g_liverLR not computed (run HemiAuto/O first)";
        return false;
    }
    if (!g_liverRegion.valid()) {
        if (out_fail_reason) *out_fail_reason =
            "g_liverRegion not computed (run HemiAuto/O first; need RIM label)";
        return false;
    }
    int n_PR = 0;
    for (uint8_t L : g_liverLR.labels) {
        if (L == LiverLeftRightLabel::PURE_RIGHT) n_PR++;
    }
    if (n_PR == 0) {
        if (out_fail_reason) *out_fail_reason = "no PURE_RIGHT vertices in g_liverLR";
        return false;
    }

    // =================================================================
    // METHOD DISPATCHER — Stage A integration of CB0.2 (MST + longest path).
    //   The legacy path below (angle-bin envelope) is still reachable
    //   when g_silSwSrcRimMethod == ENVELOPE, kept for comparison and
    //   for fallback if MST_LONGEST_PATH ever has issues. Default is
    //   MST_LONGEST_PATH because the CB0 visualization confirmed the
    //   source RIM is an open arch, not a closed loop, and the envelope
    //   step would close it incorrectly.
    // =================================================================
    if (g_silSwSrcRimMethod == SRC_RIM_METHOD_MST_LONGEST_PATH) {
        // --- 1. Build cleaned points via CB0.1 pipeline ---
        //   Uses the SAME parameters the user is tuning in the CB0.1
        //   popup; results are byte-identical to what CB0.2 popup shows.
        SmoothedRim2DResult S = buildSmoothedRim2D(
            g_rawRimSmooth_GridPx,
            g_rawRimSmooth_KnnK,
            g_rawRimSmooth_KnnIters);
        if (!S.ok) {
            if (out_fail_reason) {
                *out_fail_reason = std::string("[MST] smoothing failed: ")
                                 + S.fail_reason;
            }
            return false;
        }

        // --- 2. Order via CB0.2 pipeline ---
        OrderedRim2DResult O = buildOrderedRim2D(
            S.smo_pts,
            S.right_centroid_2D,
            S.right_centroid_valid,
            g_rawRimOrder_MaxEdgePx,
            g_rawRimOrder_NPivots);
        if (!O.ok) {
            if (out_fail_reason) {
                *out_fail_reason = std::string("[MST] ordering failed: ")
                                 + O.fail_reason;
            }
            return false;
        }

        // --- 3. Populate dense arrays (cleaned points on the path) ---
        //   These feed the sweep's cost calculation; ordering matters.
        const int M = (int)O.path.size();
        g_silSwSrcRim2DPreview.resize(M);
        g_silSwSrcRim3DPreview.resize(M);
        g_silSwSrcRimVIdxPreview.resize(M);
        g_silSwSrcRim2DPreviewLR.resize(M);
        for (int i = 0; i < M; i++) {
            const int ci = O.path[i];
            g_silSwSrcRim2DPreview[i]   = S.smo_pts   [ci];
            g_silSwSrcRim3DPreview[i]   = S.smo_pts_3D[ci];
            g_silSwSrcRimVIdxPreview[i] = S.smo_repr_vidx[ci];
            g_silSwSrcRim2DPreviewLR[i] = S.smo_lr   [ci];
        }

        // --- 4. Populate 20 pivots (arc-length resampled on path) ---
        //   Pivot LR is inherited from the nearest dense point. Index 0
        //   is the PURE_RIGHT-oriented start by construction of O.path.
        const int Np = (int)O.pivots.size();
        g_silSwSrcPivots2DPreview.resize(Np);
        g_silSwSrcPivotsLRPreview .resize(Np);
        for (int i = 0; i < Np; i++) {
            g_silSwSrcPivots2DPreview[i] = O.pivots[i];
            float best_d2 = std::numeric_limits<float>::infinity();
            int   best_j  = 0;
            for (int j = 0; j < M; j++) {
                const float dx = O.pivots[i].x - g_silSwSrcRim2DPreview[j].x;
                const float dy = O.pivots[i].y - g_silSwSrcRim2DPreview[j].y;
                const float d2 = dx*dx + dy*dy;
                if (d2 < best_d2) { best_d2 = d2; best_j = j; }
            }
            g_silSwSrcPivotsLRPreview[i] = (M > 0)
                ? g_silSwSrcRim2DPreviewLR[best_j]
                : (uint8_t)255;
        }

        // --- 4b. Compute 3D pivots using SAME 2D arc-length parameterization
        //   as O.pivots so g_silSwSrcPivots3DPreview[i] is the 3D point
        //   whose perspective projection matches g_silSwSrcPivots2DPreview[i].
        //   Open curve — NO loop closure (would mis-place pivots for the
        //   anatomical arch topology).
        g_silSwSrcPivots3DPreview.assign(Np, glm::vec3(0.0f));
        if (M >= 2) {
            std::vector<float> cum2D(M, 0.0f);
            for (int i = 1; i < M; i++) {
                const float dx = g_silSwSrcRim2DPreview[i].x
                               - g_silSwSrcRim2DPreview[i-1].x;
                const float dy = g_silSwSrcRim2DPreview[i].y
                               - g_silSwSrcRim2DPreview[i-1].y;
                cum2D[i] = cum2D[i-1] + std::sqrt(dx*dx + dy*dy);
            }
            const float total2D = cum2D[M - 1];
            for (int p = 0; p < Np; p++) {
                const float t  = float(p) / float(std::max(1, Np - 1));
                const float ss = t * total2D;
                auto it = std::upper_bound(cum2D.begin(), cum2D.end(), ss);
                int seg = (int)(it - cum2D.begin()) - 1;
                if (seg < 0) seg = 0;
                if (seg >= M - 1) {
                    g_silSwSrcPivots3DPreview[p] = g_silSwSrcRim3DPreview[M - 1];
                    continue;
                }
                const float seg_len = cum2D[seg + 1] - cum2D[seg];
                const float u = (seg_len > 1e-6f)
                              ? (ss - cum2D[seg]) / seg_len : 0.0f;
                g_silSwSrcPivots3DPreview[p] =
                    g_silSwSrcRim3DPreview[seg] +
                    u * (g_silSwSrcRim3DPreview[seg + 1] -
                         g_silSwSrcRim3DPreview[seg]);
            }
        } else if (M == 1) {
            for (int p = 0; p < Np; p++) {
                g_silSwSrcPivots3DPreview[p] = g_silSwSrcRim3DPreview[0];
            }
        }

        g_silSwSrcStartIdxPreview        = 0;     // path[0] is start by construction
        g_silSwSrcRightCentroid2DPreview = S.right_centroid_2D;

        if (g_silhouetteSweepLog) {
            // Rate-limit to avoid log spam when this function is called
            // every frame (e.g. CB1 popup open while Live ICP is running
            // and constantly moving the mesh). Print at most once per
            // ~60 calls (≈ once per second at 60 FPS).
            static int s_mst_log_skip_counter = 0;
            if ((s_mst_log_skip_counter++ % 60) == 0) {
                std::cout << "[3d/SrcPreview/MST]"
                          << " cleaned=" << (int)S.smo_pts.size()
                          << " largest_CC=" << (int)O.component.size()
                          << " path_nodes=" << M
                          << " arc=" << O.arc_length_px << "px"
                          << " pivots=" << Np
                          << " start_to_R=" << (O.start_oriented_to_right ? "YES" : "no")
                          << " rejected_edges=" << O.n_rejected_edges
                          << "  (every 60th call shown)"
                          << std::endl;
            }
        }

        g_silSwSrcPreviewCacheValid = true;
        return true;
    }
    // Otherwise: fall through to the legacy ENVELOPE method below.

    // --- Step 1: Use g_debugSourceRimChain directly --------------
    //   These are the GREEN dots the user sees with the W key —
    //   anatomical RIM vertices that already passed the user's
    //   active filter chain (quadrant / caudal / etc). Auto-populate
    //   if missing.
    //
    //   No AR-visibility filter on top: the angle-bin + max-radius
    //   envelope step below naturally selects on-screen outer
    //   verts and rejects back-projected interior points (they sit
    //   near the 2D centroid because perspective foreshortens
    //   far-side rim verts toward the screen center).
    if (g_debugSourceRimChain.empty()) {
        if (!populateDebugSourceRimChain()) {
            if (out_fail_reason) *out_fail_reason =
                "populateDebugSourceRimChain failed";
            return false;
        }
    }
    if (g_debugSourceRimChain.size() < 20) {
        if (out_fail_reason) *out_fail_reason =
            "g_debugSourceRimChain too short (<20 verts) — adjust W filters";
        return false;
    }

    const auto& Vmesh = liverMesh3D->mVertices;
    const int nV3 = (int)Vmesh.size();

    // Project every RIM-chain vertex to 2D pixel coordinates.
    std::vector<glm::vec2> all_2D;
    std::vector<glm::vec3> all_3D;
    std::vector<int>       all_vidx;
    all_2D.reserve(g_debugSourceRimChain.size());
    all_3D.reserve(g_debugSourceRimChain.size());
    all_vidx.reserve(g_debugSourceRimChain.size());
    for (int idx : g_debugSourceRimChain) {
        if (idx < 0 || idx*3 + 2 >= nV3) continue;
        const glm::vec3 p3(Vmesh[idx*3], Vmesh[idx*3+1], Vmesh[idx*3+2]);
        // Will project below (M is built earlier in this function but
        // before this rewrite the order had to change). Project here.
        // Note: glm::mat4 M is built later in original code; we move
        // its construction up via the build below.
        all_3D.push_back(p3);
        all_vidx.push_back(idx);
    }
    // (2D projection happens after M is built below)

    const glm::mat4 view_m = buildSilhouetteView();
    const glm::mat4 proj_m = buildSilhouetteProj();
    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
    const glm::mat4 M = proj_m * view_m;

    // --- Step 2: project all_3D → 2D pixels (drop off-screen) -----
    const int Nraw = (int)all_3D.size();
    all_2D.reserve(Nraw);
    {
        std::vector<glm::vec3> kept3;
        std::vector<int>       keptVI;
        std::vector<glm::vec2> kept2;
        kept3.reserve(Nraw);
        keptVI.reserve(Nraw);
        kept2.reserve(Nraw);
        for (int i = 0; i < Nraw; i++) {
            const glm::vec4 clip = M * glm::vec4(all_3D[i], 1.0f);
            if (clip.w < 1e-9f) continue;
            const float ndcx = clip.x / clip.w;
            const float ndcy = clip.y / clip.w;
            if (ndcx < -1.5f || ndcx > 1.5f ||
                ndcy < -1.5f || ndcy > 1.5f) continue;
            kept3.push_back(all_3D[i]);
            keptVI.push_back(all_vidx[i]);
            kept2.emplace_back(
                (ndcx + 1.0f) * 0.5f * float(W_img),
                (1.0f - ndcy) * 0.5f * float(H_img));
        }
        all_3D.swap(kept3);
        all_vidx.swap(keptVI);
        all_2D.swap(kept2);
    }
    const int Ns = (int)all_2D.size();
    if (Ns < 20) {
        if (out_fail_reason) *out_fail_reason =
            "too few on-screen rim-chain points (<20)";
        return false;
    }

    // --- Step 3: 2D centroid (used as ray origin) -------------------
    glm::dvec2 sum2d(0.0);
    for (const auto& p : all_2D) sum2d += glm::dvec2(p);
    const glm::vec2 cen2D(sum2d / double(Ns));

    // --- Step 4: PURE_RIGHT 2D centroid (for start anchor) ----------
    glm::dvec3 sum3(0.0);
    int n_R = 0;
    for (int i = 0; i < (int)g_liverLR.labels.size(); i++) {
        if (i*3 + 2 >= nV3) break;
        if (g_liverLR.labels[i] != LiverLeftRightLabel::PURE_RIGHT) continue;
        sum3 += glm::dvec3(Vmesh[i*3], Vmesh[i*3+1], Vmesh[i*3+2]);
        n_R++;
    }
    const glm::vec3 right_centroid_3D(sum3 / double(n_R));
    glm::vec2 right_centroid_2D(-1e6f);
    {
        const glm::vec4 clip = M * glm::vec4(right_centroid_3D, 1.0f);
        if (clip.w > 1e-9f) {
            const float ndcx = clip.x / clip.w;
            const float ndcy = clip.y / clip.w;
            right_centroid_2D = glm::vec2(
                (ndcx + 1.0f) * 0.5f * float(W_img),
                (1.0f - ndcy) * 0.5f * float(H_img));
        }
    }
    if (right_centroid_2D.x < -1e5f) {
        if (out_fail_reason) *out_fail_reason =
            "PURE_RIGHT centroid projects behind camera";
        return false;
    }
    g_silSwSrcRightCentroid2DPreview = right_centroid_2D;

    // --- Step 5: angle-bin envelope (radial ray cast) ---------------
    //   User's idea: cast rays from the 2D centroid outward in every
    //   direction; the farthest projected point in each angular bin
    //   IS that direction's outer rim envelope. This naturally rejects
    //   back-side rim verts (which foreshorten toward the 2D center
    //   under perspective) and produces a clean closed silhouette
    //   contour in angle-sorted order.
    //
    //   N_BINS = 360 (1° per bin). Empty bins are skipped, so the
    //   resulting envelope can have fewer than 360 points but is
    //   always angle-monotonic = on-screen clockwise (atan2 ascending
    //   under screen +y-down convention).
    constexpr int   N_BINS = 360;
    constexpr float kTwoPi = 6.2831853f;
    constexpr float kPi    = 3.14159265f;
    std::vector<float> bin_max_r2(N_BINS, -1.0f);
    std::vector<int>   bin_winner(N_BINS, -1);
    for (int i = 0; i < Ns; i++) {
        const float dx = all_2D[i].x - cen2D.x;
        const float dy = all_2D[i].y - cen2D.y;
        const float r2 = dx*dx + dy*dy;
        const float a  = std::atan2(dy, dx);    // [-π, π]
        int b = int((a + kPi) / kTwoPi * float(N_BINS));
        if (b < 0) b = 0;
        if (b >= N_BINS) b = N_BINS - 1;
        if (r2 > bin_max_r2[b]) {
            bin_max_r2[b] = r2;
            bin_winner[b] = i;
        }
    }

    // Determine start bin from PURE_RIGHT centroid's angle.
    const float right_angle = std::atan2(
        right_centroid_2D.y - cen2D.y,
        right_centroid_2D.x - cen2D.x);
    const int right_bin = std::min(N_BINS - 1, std::max(0,
        int((right_angle + kPi) / kTwoPi * float(N_BINS))));

    // Find the filled bin nearest to right_bin (circular distance).
    int start_bin = -1;
    int min_dist = N_BINS;
    for (int b = 0; b < N_BINS; b++) {
        if (bin_winner[b] < 0) continue;
        int d = std::abs(b - right_bin);
        if (d > N_BINS / 2) d = N_BINS - d;
        if (d < min_dist) { min_dist = d; start_bin = b; }
    }
    if (start_bin < 0) {
        if (out_fail_reason) *out_fail_reason =
            "all bins empty (impossible?)";
        return false;
    }

    // Collect filled bins starting from start_bin, walking +b (CW).
    std::vector<glm::vec2> rot2D;
    std::vector<glm::vec3> rot3D;
    std::vector<int>       rotVI;
    rot2D.reserve(N_BINS);
    rot3D.reserve(N_BINS);
    rotVI.reserve(N_BINS);
    int n_filled = 0;
    float env_sum_step = 0.0f;
    float env_max_step = 0.0f;
    glm::vec2 prev_pt(0.0f);
    bool prev_set = false;
    for (int k = 0; k < N_BINS; k++) {
        const int b = (start_bin + k) % N_BINS;
        const int wi = bin_winner[b];
        if (wi < 0) continue;
        rot2D.push_back(all_2D[wi]);
        rot3D.push_back(all_3D[wi]);
        rotVI.push_back(all_vidx[wi]);
        if (prev_set) {
            const float dx = all_2D[wi].x - prev_pt.x;
            const float dy = all_2D[wi].y - prev_pt.y;
            const float d  = std::sqrt(dx*dx + dy*dy);
            env_sum_step += d;
            if (d > env_max_step) env_max_step = d;
        }
        prev_pt = all_2D[wi];
        prev_set = true;
        n_filled++;
    }
    const int Ks = (int)rot2D.size();
    if (Ks < 20) {
        if (out_fail_reason) {
            *out_fail_reason = "envelope too sparse ("
                             + std::to_string(Ks) + " filled bins)";
        }
        return false;
    }
    g_silSwSrcStartIdxPreview = 0;

    const float env_avg_step = (Ks > 1)
        ? env_sum_step / float(Ks - 1) : 0.0f;
    if (g_silhouetteSweepLog) {
        std::cout << "[3d/SrcPreview/AngleBin] rim_chain=" << Nraw
                  << "  on_screen=" << Ns
                  << "  filled_bins=" << Ks << "/" << N_BINS
                  << "  start_bin=" << start_bin
                  << "  (right_bin=" << right_bin << ")"
                  << "  avg_step=" << env_avg_step << "px"
                  << "  max_step=" << env_max_step << "px"
                  << std::endl;
    }

    // Commit to globals
    g_silSwSrcRim2DPreview   = rot2D;
    g_silSwSrcRim3DPreview   = rot3D;
    g_silSwSrcRimVIdxPreview = rotVI;

    // Parallel LR label
    g_silSwSrcRim2DPreviewLR.assign(Ks, LiverLeftRightLabel::BOUNDARY);
    for (int i = 0; i < Ks; i++) {
        const int vidx = rotVI[i];
        if (vidx < 0 || vidx >= (int)g_liverLR.labels.size()) continue;
        g_silSwSrcRim2DPreviewLR[i] = g_liverLR.labels[vidx];
    }

    // --- Step 6: arc-length resample 20 pivots (CLOSED loop) --------
    //   The angle-bin envelope IS a closed contour around the 2D
    //   centroid by construction, so close_loop=true gives evenly-
    //   distributed pivots around the entire silhouette.
    const int N_pivot = g_silhouetteSweep.n_pivot;
    std::vector<int> pivot_nearest_idx;
    RimShape::arclengthResample2D(
        g_silSwSrcRim2DPreview, N_pivot,
        g_silSwSrcPivots2DPreview,
        /*close_loop=*/ true,
        &pivot_nearest_idx);

    g_silSwSrcPivotsLRPreview.assign(N_pivot, LiverLeftRightLabel::BOUNDARY);
    for (int p = 0; p < N_pivot; p++) {
        if (p < (int)pivot_nearest_idx.size()) {
            const int ii = pivot_nearest_idx[p];
            if (ii >= 0 && ii < (int)g_silSwSrcRim2DPreviewLR.size())
                g_silSwSrcPivotsLRPreview[p] = g_silSwSrcRim2DPreviewLR[ii];
        }
    }

    // [Step 3d2] mark cache valid for this pose-hash. Subsequent calls
    // with the same liverMesh3D pose will short-circuit at the top.
    g_silSwSrcPreviewCacheValid = true;
    return true;
}

// ---------------------------------------------------------------------
// silSwBuildTgtPreview  [REVISED — Step 3d2]
//   Build target lower-half curve + 20 anchors from
//   g_debugTargetBoundaryPoints (dense 3D point cloud filtered by
//   boundaryDist<12px ∧ instDist>=instThresh, populated by Shift+W).
//
//   Why not g_debugTargetContour2D:
//     traceContour2D often shatters the silhouette into hundreds of
//     short segments (209 / longest 303px in current scene), giving
//     a degenerate curve that doesn't span the liver. Direct use of
//     the filtered 3D boundary point set yields ~50000 points densely
//     covering the silhouette band — much more reliable.
//
//   Pipeline:
//     1. Project all 3D boundary points to 2D pixels (drop sentinels)
//     2. Compute 2D centroid
//     3. Filter to lower half (p.y > centroid.y on screen)
//     4. Bin by x into N_BINS=200 columns; for each filled bin keep
//        the point with max y → smooth lower envelope
//     5. Reverse (envelope is left-to-right by bin index; the desired
//        order starts from right-end / max x)
//     6. arc-length resample 20 anchors
// ---------------------------------------------------------------------
inline bool silSwBuildTgtPreview(std::string* out_fail_reason = nullptr)
{
    g_silSwTgtLower2DPreview.clear();
    g_silSwTgtAnchors2DPreview.clear();

    if (g_debugTargetBoundaryPoints.size() < 50) {
        if (out_fail_reason) *out_fail_reason =
            "g_debugTargetBoundaryPoints too short (<50 pts); run Shift+W first";
        return false;
    }

    const glm::mat4 view_m = buildSilhouetteView();
    const glm::mat4 proj_m = buildSilhouetteProj();
    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
    const glm::mat4 M = proj_m * view_m;

    // --- Step 1: project all boundary 3D points to 2D pixels ---
    std::vector<glm::vec2> all_2D;
    all_2D.reserve(g_debugTargetBoundaryPoints.size());
    for (const auto& p : g_debugTargetBoundaryPoints) {
        const glm::vec4 clip = M * glm::vec4(p, 1.0f);
        if (clip.w < 1e-9f) continue;
        const float ndcx = clip.x / clip.w;
        const float ndcy = clip.y / clip.w;
        if (ndcx < -1.5f || ndcx > 1.5f ||
            ndcy < -1.5f || ndcy > 1.5f) continue;
        all_2D.emplace_back(
            (ndcx + 1.0f) * 0.5f * float(W_img),
            (1.0f - ndcy) * 0.5f * float(H_img));
    }
    if ((int)all_2D.size() < 50) {
        if (out_fail_reason) *out_fail_reason =
            "too few on-screen boundary points (<50)";
        return false;
    }

    // --- Step 2: 2D centroid ---
    glm::dvec2 sum2d(0.0);
    for (const auto& p : all_2D) sum2d += glm::dvec2(p);
    const glm::vec2 centroid(sum2d / double(all_2D.size()));
    g_silSwTgtCentroid2DPreview = centroid;

    // --- Step 3: lower half (screen +y = down) ---
    std::vector<glm::vec2> lower_raw;
    lower_raw.reserve(all_2D.size() / 2);
    for (const auto& p : all_2D) {
        if (p.y > centroid.y) lower_raw.push_back(p);
    }
    if ((int)lower_raw.size() < 30) {
        if (out_fail_reason) *out_fail_reason =
            "too few lower-half boundary points (<30)";
        return false;
    }

    // --- Step 4: x-bin / max-y envelope ---
    float xmin = lower_raw[0].x, xmax = lower_raw[0].x;
    for (const auto& p : lower_raw) {
        if (p.x < xmin) xmin = p.x;
        if (p.x > xmax) xmax = p.x;
    }
    const float xrange = xmax - xmin;
    if (xrange < 10.0f) {
        if (out_fail_reason) *out_fail_reason =
            "lower-half x extent too small (<10px)";
        return false;
    }
    const int N_BINS = 200;
    std::vector<float> bin_max_y (N_BINS, -1.0f);
    std::vector<float> bin_at_x  (N_BINS,  0.0f);
    for (const auto& p : lower_raw) {
        int b = int((p.x - xmin) / xrange * float(N_BINS - 1));
        if (b < 0) b = 0;
        if (b >= N_BINS) b = N_BINS - 1;
        if (p.y > bin_max_y[b]) {
            bin_max_y[b] = p.y;
            bin_at_x[b]  = p.x;
        }
    }
    std::vector<glm::vec2> envelope;
    envelope.reserve(N_BINS);
    for (int b = 0; b < N_BINS; b++) {
        if (bin_max_y[b] > 0.0f) {
            envelope.emplace_back(bin_at_x[b], bin_max_y[b]);
        }
    }
    if ((int)envelope.size() < 20) {
        if (out_fail_reason) *out_fail_reason =
            "envelope too short after x-bin (<20 bins filled)";
        return false;
    }

    // --- Step 5: reverse to right-start (envelope is increasing x) ---
    std::vector<glm::vec2> ordered;
    ordered.reserve(envelope.size());
    for (int i = (int)envelope.size() - 1; i >= 0; i--) {
        ordered.push_back(envelope[i]);
    }
    g_silSwTgtLower2DPreview = ordered;

    // --- Step 6: arc-length resample 20 anchors ---
    const int N = g_silhouetteSweep.n_pivot;
    RimShape::arclengthResample2D(
        g_silSwTgtLower2DPreview, N,
        g_silSwTgtAnchors2DPreview,
        /*close_loop=*/ false);
    if ((int)g_silSwTgtAnchors2DPreview.size() < N) {
        if (out_fail_reason) *out_fail_reason =
            "arclengthResample2D produced too few anchors";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// startSilhouetteSweep
//   Phase 0 setup: build src/tgt caches, populate SilhouetteSweepState,
//   transition to phase=1. Logs diagnostics. Returns false on any
//   failure (sweep not started).
// ---------------------------------------------------------------------
inline bool startSilhouetteSweep()
{
    auto& S = g_silhouetteSweep;
    S = RimShape::SilhouetteSweepState{};   // reset

    // ---- Auto-trigger upstream populates (Step 3d2: still need
    //      g_debugTargetBoundaryPoints from Shift+W) ----
    if (g_debugTargetBoundaryPoints.empty()) {
        std::cout << "[Ctrl+Alt+W/3d] auto-running populateDebugTargetBoundary..."
                  << std::endl;
        if (!populateDebugTargetBoundary()) {
            S.fail_reason = "populateDebugTargetBoundary failed";
            std::cout << "[Ctrl+Alt+W/3d] " << S.fail_reason << std::endl;
            return false;
        }
    }

    // ---- Build source + target previews ---
    // (Step 3d2: src preview uses AR silhouette directly; tgt preview
    //  uses g_debugTargetBoundaryPoints. No longer depends on
    //  g_debugSourceRimChain or g_debugTargetContour2D.)
    std::string srcFail;
    if (!silSwBuildSrcPreview(&srcFail)) {
        S.fail_reason = "src setup: " + srcFail;
        std::cout << "[Ctrl+Alt+W/3d] failed — " << S.fail_reason << std::endl;
        return false;
    }
    std::string tgtFail;
    if (!silSwBuildTgtPreview(&tgtFail)) {
        S.fail_reason = "tgt setup: " + tgtFail;
        std::cout << "[Ctrl+Alt+W/3d] failed — " << S.fail_reason << std::endl;
        return false;
    }

    // ---- Copy preview caches into state machine ----
    S.src_rim_2D             = g_silSwSrcRim2DPreview;
    S.src_rim_3D_oriented    = g_silSwSrcRim3DPreview;     // Step 3d2: from preview directly
    S.src_pivots_2D          = g_silSwSrcPivots2DPreview;
    S.src_pivots_lr_label    = g_silSwSrcPivotsLRPreview;
    S.src_start_idx_in_chain = g_silSwSrcStartIdxPreview;
    S.src_right_centroid_2D  = g_silSwSrcRightCentroid2DPreview;
    S.tgt_lower_2D           = g_silSwTgtLower2DPreview;
    S.tgt_anchors_2D         = g_silSwTgtAnchors2DPreview;
    S.tgt_centroid_2D        = g_silSwTgtCentroid2DPreview;

    // Build pivot 3D from dense 3D by re-resampling along the same arc
    // length parameterisation. This ensures src_pivots_3D[i] is the
    // exact 3D position whose 2D projection matches src_pivots_2D[i].
    //
    // [Stage A] If g_silSwSrcPivots3DPreview is populated (= MST source
    // rim method is active), the dispatcher has ALREADY computed the
    // 3D pivots using the open-curve arc-length parameterisation that
    // matches src_pivots_2D. Use those directly and skip the legacy
    // closed-loop resampling, which assumes a loop topology that the
    // MST open polyline does not have.
    if (!g_silSwSrcPivots3DPreview.empty() &&
        (int)g_silSwSrcPivots3DPreview.size() == (int)S.src_pivots_2D.size())
    {
        S.src_pivots_3D = g_silSwSrcPivots3DPreview;
    } else {
        // Legacy closed-loop resampling — runs for ENVELOPE method.
        const int N = S.n_pivot;
        const int M = (int)S.src_rim_3D_oriented.size();
        if (M >= 2 && N >= 1) {
            std::vector<glm::vec3> pts3 = S.src_rim_3D_oriented;
            pts3.push_back(pts3.front());     // close loop
            std::vector<float> cum(pts3.size(), 0.0f);
            for (size_t i = 1; i < pts3.size(); i++)
                cum[i] = cum[i-1] + glm::length(pts3[i] - pts3[i-1]);
            const float total = cum.back();
            S.src_pivots_3D.clear();
            S.src_pivots_3D.reserve(N);
            if (total > 1e-6f) {
                for (int k = 0; k < N; k++) {
                    const float t = (float(k) + 0.5f) * total / float(N);
                    auto it = std::upper_bound(cum.begin(), cum.end(), t);
                    if (it == cum.begin()) { S.src_pivots_3D.push_back(pts3.front()); continue; }
                    if (it == cum.end())   { S.src_pivots_3D.push_back(pts3.back());  continue; }
                    const size_t idx = std::distance(cum.begin(), it);
                    const float seg = cum[idx] - cum[idx-1];
                    const float alpha = (seg > 1e-6f) ? (t - cum[idx-1]) / seg : 0.0f;
                    S.src_pivots_3D.push_back(glm::mix(pts3[idx-1], pts3[idx], alpha));
                }
            }
        }
    }

    // ---- Configure phase machine ----
    S.n_pivot    = (int)S.tgt_anchors_2D.size();
    S.n_rotation = 36;

    // Total candidate counts
    const int p1_cands = S.n_pivot * S.n_rotation;             // 20 * 36 = 720
    const int p2_cands = (2 * S.phase2_pivot_radius + 1)
                       * (2 * int(S.phase2_theta_radius_deg) + 1);
    // Phase 2 will recompute exact total after Phase 1 done — initial
    // value here is just for diagnostics.
    (void)p2_cands;

    S.total_frames_phase1 = g_silhouetteSweepAnimate
                              ? std::max(1, g_silhouetteSweepFrames1)
                              : 1;
    S.total_frames_phase2 = g_silhouetteSweepAnimate
                              ? std::max(1, g_silhouetteSweepFrames2)
                              : 1;
    S.candidates_per_frame = std::max(1,
        (p1_cands + S.total_frames_phase1 - 1) / S.total_frames_phase1);

    S.total_candidates = p1_cands;
    S.candidate_idx    = 0;
    S.current_frame    = 0;
    S.phase            = 1;
    S.active           = true;
    S.best_cost        = 1e18;
    S.best_i_pivot     = -1;
    S.best_theta_deg   = 0.0f;
    S.best_T           = glm::mat4(1.0f);
    S.cost_history.clear();

    std::cout << "[Ctrl+Alt+W/3d] sweep START\n"
              << "  src_dense=" << S.src_rim_2D.size()
              << "  src_pivots=" << S.src_pivots_3D.size()
              << "  tgt_lower=" << S.tgt_lower_2D.size()
              << "  tgt_anchors=" << S.tgt_anchors_2D.size()
              << "  start_idx=" << S.src_start_idx_in_chain
              << "  reversed=" << (S.src_dir_reversed ? "Y" : "N")
              << "\n  Phase 1: " << p1_cands << " cands, "
              << S.candidates_per_frame << "/frame, "
              << S.total_frames_phase1 << " frames"
              << std::endl;

    return true;
}


// ---------------------------------------------------------------------
// tickSilhouetteSweep
//   Process one frame's batch of candidates. Returns true while sweep
//   is ongoing; false when complete (caller then runs the apply/save
//   tail in main.cpp).
//
//   Phase 1: 20 pivots * 36 rotations = 720 candidates
//   Phase 2: (2*radius+1) pivots * (2*range_deg+1) rotations around best
// ---------------------------------------------------------------------
inline bool tickSilhouetteSweep()
{
    auto& S = g_silhouetteSweep;
    if (!S.active) return false;

    const glm::mat4 view_m = buildSilhouetteView();
    const glm::mat4 proj_m = buildSilhouetteProj();
    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;

    int processed_this_frame = 0;
    const int budget = std::max(1, S.candidates_per_frame);

    int last_pivot_i = -1;
    while (processed_this_frame < budget
           && S.candidate_idx < S.total_candidates)
    {
        int pivot_i;
        float theta_deg;

        if (S.phase == 1) {
            // candidate_idx → (i, k_rot)
            const int i_p   = S.candidate_idx / S.n_rotation;
            const int k_rot = S.candidate_idx % S.n_rotation;
            pivot_i   = i_p;
            theta_deg = 360.0f * float(k_rot) / float(S.n_rotation);
        } else if (S.phase == 2) {
            const int n_rot_p2 = 2 * int(S.phase2_theta_radius_deg) + 1;
            const int i_p     = S.candidate_idx / n_rot_p2;
            const int k_rot   = S.candidate_idx % n_rot_p2;
            pivot_i = S.phase1_best_i_pivot - S.phase2_pivot_radius + i_p;
            // Wrap pivot_i mod n_pivot (closed-loop indexing)
            while (pivot_i < 0) pivot_i += S.n_pivot;
            while (pivot_i >= S.n_pivot) pivot_i -= S.n_pivot;
            theta_deg = S.phase1_best_theta_deg
                      - S.phase2_theta_radius_deg
                      + float(k_rot);
        } else {
            break;
        }

        glm::mat4 T_cand(1.0f);
        double cost = RimShape::evaluateSilhouetteSweepCandidate(
            S.src_rim_3D_oriented, S.src_pivots_3D,
            S.tgt_anchors_2D, S.tgt_lower_2D,
            pivot_i, theta_deg,
            view_m, proj_m, W_img, H_img,
            T_cand);

        // === Check A: rotation magnitude cap (independent of CC) ===
        //   Wrap θ to (-180°, +180°], reject if |θ| > cap.
        if (g_silSwCheckA_Enable) {
            float th = theta_deg;
            while (th >  180.0f) th -= 360.0f;
            while (th < -180.0f) th += 360.0f;
            if (std::fabs(th) > g_silSwCheckA_RotCapDeg) {
                cost = 1e18;
            }
        }

        // === Check B: CC orientation guard (CRANIAL→CAUDAL must point
        //     screen-down within ±tolerance, i.e. ~6 o'clock) ===
        if (cost < 1e17 && g_silSwCheckB_Enable && g_liverCC.valid()) {
            const glm::vec3 d_cc(
                float(g_liverCC.d_cc.x),
                float(g_liverCC.d_cc.y),
                float(g_liverCC.d_cc.z));
            const glm::vec3 P_s = S.src_pivots_3D[pivot_i];
            const float step = 60.0f;     // mesh units (rough liver scale)
            const glm::vec3 cranial_w = P_s + step * d_cc;
            const glm::vec3 caudal_w  = P_s - step * d_cc;
            const glm::mat4 VP = proj_m * view_m;
            auto project = [&](const glm::vec3& p3) -> glm::vec2 {
                const glm::vec4 cp = VP * (T_cand * glm::vec4(p3, 1.0f));
                if (cp.w <= 1e-9f) return glm::vec2(-1e6f, -1e6f);
                return glm::vec2(
                    (cp.x / cp.w + 1.0f) * 0.5f * float(W_img),
                    (1.0f - cp.y / cp.w) * 0.5f * float(H_img));
            };
            const glm::vec2 cranial_2D = project(cranial_w);
            const glm::vec2 caudal_2D  = project(caudal_w);
            if (cranial_2D.x > -1e5f && caudal_2D.x > -1e5f) {
                const float dx = caudal_2D.x - cranial_2D.x;
                const float dy = caudal_2D.y - cranial_2D.y;
                const float angle_deg =
                    std::atan2(dy, dx) * 180.0f / 3.14159265f;
                // Target: 90° (= +y screen-down = 6 o'clock)
                float delta = angle_deg - 90.0f;
                while (delta >  180.0f) delta -= 360.0f;
                while (delta < -180.0f) delta += 360.0f;
                if (std::fabs(delta) > g_silSwCheckB_CCToleranceDeg) {
                    cost = 1e18;
                }
            }
            // If projection failed (NaN / behind camera), no reject —
            // let the candidate through and rely on dense chamfer cost.
        }

        if (cost < S.best_cost) {
            S.best_cost      = cost;
            S.best_i_pivot   = pivot_i;
            S.best_theta_deg = theta_deg;
            S.best_T         = T_cand;
        }

        last_pivot_i = pivot_i;
        S.candidate_idx++;
        processed_this_frame++;
    }

    // ----- Trial visualization (Step 3d2: animate sweep on AR scene) -----
    //   Mirrors the Step 3c trial-pose visualization. Yellow dots =
    //   current best transformed source rim (whole curve). Pivots
    //   shown in gray with the *last-tried* index in cyan so the user
    //   can see the algorithm walking through the discretization.
    //   Cleared by main.cpp's finish path.
    {
        g_contourSweepShowTrial = true;
        g_contourSweepTrialSrc.clear();
        g_contourSweepTrialSrc.reserve(S.src_rim_3D_oriented.size());
        for (const auto& p : S.src_rim_3D_oriented) {
            const glm::vec4 p4 = S.best_T * glm::vec4(p, 1.0f);
            g_contourSweepTrialSrc.emplace_back(p4.x, p4.y, p4.z);
        }
        g_contourSweepSrcPivotsTrial.clear();
        g_contourSweepSrcPivotsTrial.reserve(S.src_pivots_3D.size());
        for (const auto& p : S.src_pivots_3D) {
            const glm::vec4 p4 = S.best_T * glm::vec4(p, 1.0f);
            g_contourSweepSrcPivotsTrial.emplace_back(p4.x, p4.y, p4.z);
        }
        // Use existing pivot-anchor highlight slots:
        //   CurrentJSrc highlights the source pivot in cyan.
        //   CurrentITgt would highlight the target anchor, but Step 3d
        //   keeps target as a 2D-only curve (no 3D back-projection in
        //   the scene), so we leave it -1.
        g_contourSweepCurrentJSrc = last_pivot_i;
        g_contourSweepCurrentITgt = -1;
        // Note: g_contourSweepTgtAnchors3D is not populated by Step 3d
        // (target is purely 2D), so the existing draw code will simply
        // skip the target-anchor dots.
        g_contourSweepTgtAnchors3D.clear();
    }

    // Push history once per frame for ImGui plot
    S.cost_history.push_back(S.best_cost);
    S.current_frame++;

    // -------- Phase transition --------
    if (S.candidate_idx >= S.total_candidates) {
        if (S.phase == 1) {
            // Lock Phase 1 best, set up Phase 2 sweep
            S.phase1_best_i_pivot   = S.best_i_pivot;
            S.phase1_best_theta_deg = S.best_theta_deg;

            const int n_pivot_p2 = 2 * S.phase2_pivot_radius + 1;
            const int n_rot_p2   = 2 * int(S.phase2_theta_radius_deg) + 1;
            const int p2_cands   = n_pivot_p2 * n_rot_p2;

            S.total_candidates    = p2_cands;
            S.candidate_idx       = 0;
            S.current_frame       = 0;
            S.candidates_per_frame = std::max(1,
                (p2_cands + S.total_frames_phase2 - 1) / S.total_frames_phase2);
            S.phase = 2;

            if (g_silhouetteSweepLog) {
                std::cout << "[Ctrl+Alt+W/3d] Phase 1 done."
                          << "  best i=" << S.best_i_pivot
                          << "  θ=" << S.best_theta_deg << "°"
                          << "  cost=" << S.best_cost << "px\n"
                          << "  Phase 2: " << p2_cands << " cands, "
                          << S.candidates_per_frame << "/frame"
                          << std::endl;
            }
            return true;
        } else if (S.phase == 2) {
            S.phase  = 3;
            S.active = false;
            if (g_silhouetteSweepLog) {
                std::cout << "[Ctrl+Alt+W/3d] Phase 2 done."
                          << "  final best i=" << S.best_i_pivot
                          << "  θ=" << S.best_theta_deg << "°"
                          << "  cost=" << S.best_cost << "px"
                          << std::endl;
            }
            return false;
        }
    }

    return true;
}

// =========================================================================
//  Phase U-1: soft partition の計算ドライバ
// =========================================================================
inline void runSoftPartitionCompute() {
    if (!g_softAnchors.valid()) {
        std::cerr << "[Soft U-1] anchors invalid (need >="
                  << SoftPartition::MIN_GROUPS << ", have "
                  << g_softAnchors.count() << ")" << std::endl;
        g_softPartReady = false; return;
    }
    if (!screenMesh || SoftPartition::vertexCount(*screenMesh) == 0 ||
        screenMesh->mIndices.empty()) {
        std::cerr << "[Soft U-1] screenMesh empty or has no triangles" << std::endl;
        g_softPartReady = false; return;
    }
    if (!liverMesh3D || SoftPartition::vertexCount(*liverMesh3D) == 0 ||
        liverMesh3D->mIndices.empty()) {
        std::cerr << "[Soft U-1] liverMesh3D empty or has no triangles" << std::endl;
        g_softPartReady = false; return;
    }

    std::cout << "[Soft U-1] Computing partition: N=" << g_softAnchors.count()
              << " s_ume=" << g_softUmeyamaScale
              << " sigma=" << (g_softPartParams.autoSigma ? "auto" : "manual")
              << " baseline_none=" << g_softPartParams.baselineNone << std::endl;

    bool okT = SoftPartition::computeTargetPartition(
        *screenMesh, g_softAnchors, g_softPartParams,
        g_softTgtField, g_softGroupRadii);
    if (okT) {
        std::cout << "[Soft U-1] Target OK: "
                  << SoftPartition::vertexCount(*screenMesh) << " verts. group radii=";
        for (size_t i = 0; i < g_softGroupRadii.size(); ++i)
            std::cout << g_softGroupRadii[i]
                      << (i + 1 < g_softGroupRadii.size() ? "," : "");
        std::cout << std::endl;
    } else {
        std::cerr << "[Soft U-1] Target FAILED" << std::endl;
    }

    // U-1: registration 後は liver も world 単位なので s_ume=1 を渡して
    // r_src == r_tgt にする (二重補正回避)。g_softUmeyamaScale は表示用に保持。
    bool okS = SoftPartition::computeSourcePartition(
        *liverMesh3D, g_softAnchors, g_softGroupRadii,
        1.0f, g_softPartParams, g_softSrcField);
    if (okS) {
        std::cout << "[Soft U-1] Source OK: "
                  << SoftPartition::vertexCount(*liverMesh3D) << " verts." << std::endl;
    } else {
        std::cerr << "[Soft U-1] Source FAILED" << std::endl;
    }

    g_softPartReady = okT && okS;
}

// =========================================================================
//  Phase U-2: soft-weighted ICP  (Ctrl+U)
//  Reuses: extractFrontFacePoints (target cloud), Reg3DCustom KDTree (NN),
//          NormalRefine::solve6x6 / transformVector6dToMatrix4d (point-to-plane),
//          NormalRefine::applyIncrementalTransform (apply to all organs),
//          computeUnifiedMetrics (RMSE/IoU).
//  New here: the soft per-correspondence weight  w = agree * (1 - p_none),
//            agree = <p_src[j], p_tgt[k]>, and the target-cloud -> screenMesh
//            probability mapping (PointCloud carries no screenMesh index).
// =========================================================================
inline void runSoftWeightedICP(std::vector<glm::dmat4>* outTraj = nullptr,
                               bool applyResult = true) {
    if (!g_softPartReady || !g_softSrcField.valid || !g_softTgtField.valid) {
        std::cerr << "[Soft U-2] partition not ready — run Shift+U first" << std::endl;
        return;
    }
    if (!screenMesh || !liverMesh3D) {
        std::cerr << "[Soft U-2] screenMesh / liverMesh3D missing" << std::endl;
        return;
    }

    // ----- target cloud (same path as Ctrl+G / Ctrl+N) -----
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = RegRatios::zThresh();
    auto targetCloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!targetCloud || targetCloud->size() < 6 || !targetCloud->hasNormals()) {
        std::cerr << "[Soft U-2] target cloud invalid (need normals, >=6 pts)" << std::endl;
        return;
    }

    // Optional voxel downsample (debug / large-scale). 1920x1080 dense clouds
    // can exceed 300k points; reuse Ctrl+G's voxelDownSample (keeps normals).
    if (g_softIcpParams.downsampleTarget) {
        const float vox = std::max(1e-6f, g_softIcpParams.voxelFac * (float)g_sceneDiag);
        auto ds = reg_extract.voxelDownSample(targetCloud, vox);
        if (ds && ds->size() >= 6 && ds->hasNormals()) {
            std::cout << "[Soft U-2] target downsampled "
                      << targetCloud->size() << " -> " << ds->size()
                      << "  (voxel=" << vox << ")" << std::endl;
            targetCloud = ds;
        } else {
            std::cerr << "[Soft U-2] downsample gave too few/normless pts; "
                         "using full cloud" << std::endl;
        }
    }

    const int nT = (int)targetCloud->size();
    std::vector<glm::dvec3> tgtD(nT);
    for (int i = 0; i < nT; ++i) tgtD[i] = glm::dvec3(targetCloud->points[i]);
    Reg3DCustom::NanoflannAdaptorD tgtAdaptor(tgtD);
    auto tgtTree = Reg3DCustom::buildKDTreeD(tgtAdaptor);

    // ----- NEW: target-cloud point -> nearest screenMesh vertex -> p_tgt -----
    const int nSV = SoftPartition::vertexCount(*screenMesh);
    if ((int)g_softTgtField.probs.size() != nSV) {
        std::cerr << "[Soft U-2] target field size mismatch" << std::endl;
        return;
    }
    std::vector<glm::dvec3> svD(nSV);
    for (int v = 0; v < nSV; ++v)
        svD[v] = glm::dvec3(SoftPartition::vertexAt(*screenMesh, v));
    Reg3DCustom::NanoflannAdaptorD svAdaptor(svD);
    auto svTree = Reg3DCustom::buildKDTreeD(svAdaptor);

    const int G = g_softTgtField.numGroups;
    std::vector<std::array<float, SoftPartition::MAX_GROUPS + 1>> tgtCloudProb(nT);
    for (int k = 0; k < nT; ++k) {
        size_t vidx = 0; double dsq = 0.0;
        if (Reg3DCustom::searchKNN1D(*svTree, tgtD[k], vidx, dsq))
            tgtCloudProb[k] = g_softTgtField.probs[vidx];
        else
            tgtCloudProb[k].fill(0.0f);
    }

    // ----- source = liver vertices (world coords) + normals -----
    const int nS = SoftPartition::vertexCount(*liverMesh3D);
    if ((int)g_softSrcField.probs.size() != nS) {
        std::cerr << "[Soft U-2] source field size mismatch" << std::endl;
        return;
    }
    const bool haveSrcN = ((int)(liverMesh3D->mNormals.size() / 3) == nS);
    std::vector<glm::dvec3> srcD(nS), srcN(nS, glm::dvec3(0, 0, 1));
    for (int j = 0; j < nS; ++j) {
        srcD[j] = glm::dvec3(SoftPartition::vertexAt(*liverMesh3D, j));
        if (haveSrcN) {
            glm::dvec3 n(liverMesh3D->mNormals[j * 3],
                         liverMesh3D->mNormals[j * 3 + 1],
                         liverMesh3D->mNormals[j * 3 + 2]);
            double l = glm::length(n);
            if (l > 1e-12) srcN[j] = n / l;
        }
    }

    // ----- before metrics -----
    computeUnifiedMetrics();
    g_softIcpRmseBefore = registrationHandle.compRmse;
    g_softIcpIoUBefore  = registrationHandle.compIoU2D;

    const double maxCorr2 =
        std::pow((double)g_softIcpParams.maxCorrDistFac * (double)g_sceneDiag, 2.0);

    std::cout << "[Soft U-2] start: src=" << nS << " tgt=" << nT
              << " groups=" << G << " maxIters=" << g_softIcpParams.maxIters
              << " mode=" << (g_softIcpParams.uniformWeight ? "UNIFORM(ablation)" : "SOFT")
              << " RMSE_before=" << g_softIcpRmseBefore
              << " IoU_before=" << g_softIcpIoUBefore << std::endl;

    glm::dmat4 T_total(1.0);
    int iter = 0;
    for (; iter < g_softIcpParams.maxIters; ++iter) {
        double JTJ[6][6] = {};
        double JTr[6]    = {};
        int    used = 0;
        double wsum = 0.0, rawErr2 = 0.0;
        int    rawN = 0;

        const glm::dmat3 Rtot = glm::dmat3(T_total);
        for (int j = 0; j < nS; ++j) {
            const auto& ps = g_softSrcField.probs[j];
            const float pnone = ps[SoftPartition::NONE_IDX];
            if (pnone > g_softIcpParams.noneSkip) continue;

            glm::dvec3 vs = glm::dvec3(T_total * glm::dvec4(srcD[j], 1.0));
            size_t k = 0; double dsq = 0.0;
            if (!Reg3DCustom::searchKNN1D(*tgtTree, vs, k, dsq)) continue;
            rawErr2 += dsq; ++rawN;
            if (dsq > maxCorr2) continue;

            const auto& pt = tgtCloudProb[k];
            double w;
            if (g_softIcpParams.uniformWeight) {
                // ablation: plain ICP. Same correspondence set (none-skip still
                // applies above), but every match counts equally.
                w = 1.0;
            } else {
                double agree = 0.0;
                for (int g = 0; g < G; ++g) agree += (double)ps[g] * (double)pt[g];
                w = agree * (1.0 - (double)pnone);
                if (w < (double)g_softIcpParams.minWeight) continue;
            }

            glm::dvec3 vt = tgtD[k];
            glm::dvec3 nt = glm::normalize(glm::dvec3(targetCloud->normals[k]));
            if (haveSrcN) {
                glm::dvec3 ns = glm::normalize(Rtot * srcN[j]);
                if (glm::dot(ns, nt) < 0.0) nt = -nt;
            }

            const double r  = glm::dot(vs - vt, nt);
            const double sw = std::sqrt(w);
            double J[6];
            J[0] = (vs.y * nt.z - vs.z * nt.y) * sw;
            J[1] = (vs.z * nt.x - vs.x * nt.z) * sw;
            J[2] = (vs.x * nt.y - vs.y * nt.x) * sw;
            J[3] = nt.x * sw; J[4] = nt.y * sw; J[5] = nt.z * sw;
            const double rr = r * sw;
            for (int a = 0; a < 6; ++a) {
                JTr[a] += J[a] * rr;
                for (int b = 0; b < 6; ++b) JTJ[a][b] += J[a] * J[b];
            }
            ++used; wsum += w;
        }

        if (used < 6) {
            std::cout << "[Soft U-2] iter=" << iter << " too few corr ("
                      << used << "), stopping" << std::endl;
            break;
        }

        for (int a = 0; a < 6; ++a) JTJ[a][a] += (double)g_softIcpParams.tikhonov;
        double negJTr[6];
        for (int a = 0; a < 6; ++a) negJTr[a] = -JTr[a];
        double x[6] = {};
        if (!NormalRefine::solve6x6(JTJ, negJTr, x)) {
            std::cout << "[Soft U-2] iter=" << iter << " solve6x6 failed" << std::endl;
            break;
        }

        glm::dmat4 dT = NormalRefine::transformVector6dToMatrix4d(x);
        T_total = dT * T_total;
        if (outTraj) outTraj->push_back(T_total);   // [LIVE] record for replay

        const double dnorm = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] +
                                       x[3]*x[3] + x[4]*x[4] + x[5]*x[5]);
        const double rawRmse = rawN ? std::sqrt(rawErr2 / (double)rawN) : 0.0;
        std::cout << "[Soft U-2] iter=" << iter << " used=" << used
                  << " wsum=" << wsum << " rawRMSE=" << rawRmse
                  << " |dx|=" << dnorm << std::endl;
        if (dnorm < (double)g_softIcpParams.convergeEps) { ++iter; break; }
    }

    g_softIcpLastIters = iter;

    if (!applyResult) {
        // [LIVE record] caller will replay outTraj frame-by-frame and finalize
        // (apply + metrics + revert-state) once the animation completes.
        std::cout << "[Soft U-2] (live record) iters=" << iter
                  << " frames=" << (outTraj ? outTraj->size() : 0) << std::endl;
        return;
    }

    // ----- apply accumulated refinement to all organs -----
    auto organs = getOrganList();
    NormalRefine::applyIncrementalTransform(T_total, organs);
    g_softIcpAppliedT  = T_total;   // remember so we can revert
    g_softIcpCanRevert = true;

    // ----- after metrics -----
    computeUnifiedMetrics();
    g_softIcpRmseAfter = registrationHandle.compRmse;
    g_softIcpIoUAfter  = registrationHandle.compIoU2D;

    std::cout << "[Soft U-2] DONE iters=" << iter
              << "  RMSE " << g_softIcpRmseBefore << " -> " << g_softIcpRmseAfter
              << "  IoU "  << g_softIcpIoUBefore  << " -> " << g_softIcpIoUAfter
              << std::endl;
}

// =========================================================================
//  Phase U-2: revert the last soft-weighted ICP (apply inverse of T_total)
// =========================================================================
inline void revertSoftICP() {
    if (!g_softIcpCanRevert) {
        std::cerr << "[Soft U-2] nothing to revert" << std::endl;
        return;
    }
    glm::dmat4 inv = glm::inverse(g_softIcpAppliedT);
    auto organs = getOrganList();
    NormalRefine::applyIncrementalTransform(inv, organs);
    g_softIcpCanRevert = false;
    g_softIcpAppliedT  = glm::dmat4(1.0);
    computeUnifiedMetrics();
    g_softIcpRmseAfter = registrationHandle.compRmse;
    g_softIcpIoUAfter  = registrationHandle.compIoU2D;
    std::cout << "[Soft U-2] reverted to pre-ICP pose.  RMSE="
              << g_softIcpRmseAfter << " IoU=" << g_softIcpIoUAfter << std::endl;
}

// =========================================================================
//  Phase U-CPD: pure rigid CPD (anchor-free), staged so the U-tab UI can
//  drive it as a sequential checklist (check 1 -> 5 to advance).
//
//  Stage 1  extract target cloud   (front-face points of screenMesh)
//  Stage 2  downsample target      (voxel grid; the solve size)
//  Stage 3  build source           (liver mVertices, read DIRECTLY — no
//                                    SoftPartition / no anchor) + before-metrics
//  Stage 4  run CPD EM             (CpdRigid::runRigidCPD)
//  Stage 5  apply + after-metrics  (applyIncrementalTransform of the result)
//
//  Each stage sets g_cpdStageDone[i] on success. The drivers below
//  (runCpdRegistration / revertCpd) and cpdUncheckFrom keep the staged
//  state and the actually-applied transform consistent.
// =========================================================================

// Undo the applied transform (if any) and clear stages >= `stage`.
inline void cpdUncheckFrom(int stage) {
    if (stage <= 5 && g_cpdStageDone[5] && g_cpdCanRevert) {
        glm::dmat4 inv = glm::inverse(g_cpdAppliedT);
        auto organs = getOrganList();
        NormalRefine::applyIncrementalTransform(inv, organs);
        g_cpdCanRevert = false;
        g_cpdAppliedT  = glm::dmat4(1.0);
        computeUnifiedMetrics();
        g_cpdRmseAfter = registrationHandle.compRmse;
        g_cpdIoUAfter  = registrationHandle.compIoU2D;
        std::cout << "[CPD] stage 5 apply reverted (RMSE=" << g_cpdRmseAfter
                  << " IoU=" << g_cpdIoUAfter << ")" << std::endl;
    }
    for (int j = stage; j < 6; ++j) g_cpdStageDone[j] = false;
    if (stage <= 1) { g_cpdTgtCloud.reset(); g_cpdTgtCountRaw = 0; }
    if (stage <= 2) { g_cpdTgtD.clear();     g_cpdTgtCountDS  = 0; }
    if (stage <= 3) { g_cpdSrcD.clear();     g_cpdSrcCount = 0; g_cpdSigma2Init = 0.0; }
    if (stage <= 4) { g_cpdResult = CpdRigid::Result{}; g_cpdLastIters = 0; }
}

// Full reset (reverts apply too).
inline void cpdResetPipeline() {
    cpdUncheckFrom(1);
    std::cout << "[CPD] pipeline reset" << std::endl;
}

// --- Stage 1: extract target front-face cloud --------------------------
inline bool cpdStage1_extractTarget() {
    if (!screenMesh || !liverMesh3D) {
        std::cerr << "[CPD-1] screenMesh / liverMesh3D missing" << std::endl;
        return false;
    }
    Reg3DCustom::NoOpen3DRegistration reg_extract;
    const float zThresh = RegRatios::zThresh();
    auto cloud = reg_extract.extractFrontFacePoints(
        *screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (!cloud || cloud->size() < 10) {
        std::cerr << "[CPD-1] target cloud invalid (>=10 pts needed)" << std::endl;
        return false;
    }
    g_cpdTgtCloud    = cloud;
    g_cpdTgtCountRaw = (int)cloud->size();
    g_cpdStageDone[1] = true;
    std::cout << "[CPD-1] target extracted: " << g_cpdTgtCountRaw << " pts" << std::endl;
    return true;
}

// --- Stage 2: voxel downsample target ----------------------------------
inline bool cpdStage2_downsample() {
    if (!g_cpdTgtCloud) {
        std::cerr << "[CPD-2] no target cloud (run stage 1 first)" << std::endl;
        return false;
    }
    auto cloud = g_cpdTgtCloud;
    if (g_cpdParams.downsample) {
        const float vox = std::max(1e-6f, g_cpdParams.tgtVoxelFac * (float)g_sceneDiag);
        Reg3DCustom::NoOpen3DRegistration reg_extract;
        auto ds = reg_extract.voxelDownSample(cloud, vox);
        if (ds && ds->size() >= 10) {
            std::cout << "[CPD-2] target downsampled " << cloud->size()
                      << " -> " << ds->size() << "  (voxel=" << vox << ")" << std::endl;
            cloud = ds;
        } else {
            std::cerr << "[CPD-2] downsample gave too few pts; using full cloud" << std::endl;
        }
    } else {
        std::cout << "[CPD-2] downsample OFF; using full target ("
                  << cloud->size() << " pts)" << std::endl;
    }
    const int n = (int)cloud->size();
    g_cpdTgtD.resize(n);
    for (int i = 0; i < n; ++i) g_cpdTgtD[i] = glm::dvec3(cloud->points[i]);
    g_cpdTgtCountDS = n;
    g_cpdStageDone[2] = true;
    return true;
}

// --- Stage 3: build source (anchor-free) + before-metrics + sigma2 -----
inline bool cpdStage3_buildSource() {
    if (!liverMesh3D) {
        std::cerr << "[CPD-3] liverMesh3D missing" << std::endl;
        return false;
    }
    // Source = liver vertices in world coords, read DIRECTLY from mVertices.
    // (Deliberately NOT SoftPartition::vertexAt — keeps the CPD path free of
    //  the U-1 anchor / probability fields, so this is a clean baseline.)
    const int nV = (int)(liverMesh3D->mVertices.size() / 3);
    if (nV < 10) {
        std::cerr << "[CPD-3] liver has too few vertices" << std::endl;
        return false;
    }
    std::vector<glm::dvec3> src(nV);
    for (int j = 0; j < nV; ++j)
        src[j] = glm::dvec3(liverMesh3D->mVertices[j * 3 + 0],
                            liverMesh3D->mVertices[j * 3 + 1],
                            liverMesh3D->mVertices[j * 3 + 2]);

    // Optional source thinning (compute-only; the final T still applies to the
    // full mesh). Off by default (srcVoxelFac == 0).
    if (g_cpdParams.srcVoxelFac > 0.0f) {
        auto sc = std::make_shared<Reg3DCustom::PointCloud>();
        sc->points.reserve(nV);
        for (const auto& p : src) sc->points.push_back(glm::vec3(p));
        Reg3DCustom::NoOpen3DRegistration reg_extract;
        const float vox = std::max(1e-6f, g_cpdParams.srcVoxelFac * (float)g_sceneDiag);
        auto ds = reg_extract.voxelDownSample(sc, vox);
        if (ds && ds->size() >= 10) {
            std::cout << "[CPD-3] source downsampled " << src.size()
                      << " -> " << ds->size() << "  (voxel=" << vox << ")" << std::endl;
            src.resize(ds->size());
            for (size_t i = 0; i < ds->size(); ++i) src[i] = glm::dvec3(ds->points[i]);
        }
    }

    g_cpdSrcD     = std::move(src);
    g_cpdSrcCount = (int)g_cpdSrcD.size();

    // Before-metrics (current pose, = the shared Umeyama init).
    computeUnifiedMetrics();
    g_cpdRmseBefore = registrationHandle.compRmse;
    g_cpdIoUBefore  = registrationHandle.compIoU2D;

    if (!g_cpdTgtD.empty() && !g_cpdSrcD.empty())
        g_cpdSigma2Init = CpdRigid::initialSigma2(g_cpdSrcD, g_cpdTgtD);

    g_cpdStageDone[3] = true;
    std::cout << "[CPD-3] source=" << g_cpdSrcCount
              << "  sigma2_init=" << g_cpdSigma2Init
              << "  RMSE_before=" << g_cpdRmseBefore
              << "  IoU_before=" << g_cpdIoUBefore << std::endl;
    return true;
}

// --- Stage 4: run the CPD EM -------------------------------------------
inline bool cpdStage4_runEM(std::vector<glm::dmat4>* outTraj = nullptr) {
    if (g_cpdSrcD.empty() || g_cpdTgtD.empty()) {
        std::cerr << "[CPD-4] source/target empty (run stages 1-3 first)" << std::endl;
        return false;
    }
    std::cout << "[CPD-4] start: src=" << g_cpdSrcD.size()
              << " tgt=" << g_cpdTgtD.size()
              << " w=" << g_cpdParams.w_outlier
              << " solveScale=" << (g_cpdParams.solveScale ? 1 : 0)
              << " maxIters=" << g_cpdParams.maxIters << std::endl;
    g_cpdResult    = CpdRigid::runRigidCPD(g_cpdSrcD, g_cpdTgtD, g_cpdParams, outTraj);
    g_cpdLastIters = g_cpdResult.iters;
    if (!g_cpdResult.ok) {
        if (g_cpdResult.collapsed) {
            // N_P guard tripped: the GMM shrank onto a small subset of the
            // target (overfit). With a good init this usually means the
            // Umeyama pose is already at the data noise floor and there is no
            // rigid move left to make. Keep the current pose.
            std::cerr << "[CPD-4] REFUSED: N_P collapsed to " << g_cpdResult.N_P
                      << " / " << g_cpdTgtD.size() << " ("
                      << (g_cpdTgtD.empty() ? 0.0
                            : 100.0 * g_cpdResult.N_P / (double)g_cpdTgtD.size())
                      << "% explained) -- overfit; init likely already optimal. "
                      << "Lower 'N_P collapse frac' or raise 'sigma2 floor' to override."
                      << std::endl;
        } else {
            std::cerr << "[CPD-4] CPD failed (ok=false: too few correspondences)" << std::endl;
        }
        return false;
    }
    std::cout << "[CPD-4] done: iters=" << g_cpdResult.iters
              << " sigma2=" << g_cpdResult.sigma2
              << " (floor=" << g_cpdResult.sigma2Floor << ")"
              << " scale=" << g_cpdResult.finalScale
              << " N_P=" << g_cpdResult.N_P
              << " / " << g_cpdTgtD.size() << std::endl;
    g_cpdStageDone[4] = true;
    return true;
}

// --- Stage 5: apply result + after-metrics -----------------------------
inline bool cpdStage5_apply() {
    if (!g_cpdStageDone[4] || !g_cpdResult.ok) {
        std::cerr << "[CPD-5] no valid result (run stage 4 first)" << std::endl;
        return false;
    }
    if (g_cpdCanRevert) {
        std::cerr << "[CPD-5] already applied; revert first" << std::endl;
        return false;
    }
    auto organs = getOrganList();
    NormalRefine::applyIncrementalTransform(g_cpdResult.T, organs);
    g_cpdAppliedT  = g_cpdResult.T;
    g_cpdCanRevert = true;
    computeUnifiedMetrics();
    g_cpdRmseAfter = registrationHandle.compRmse;
    g_cpdIoUAfter  = registrationHandle.compIoU2D;
    g_cpdStageDone[5] = true;
    std::cout << "[CPD-5] DONE  RMSE " << g_cpdRmseBefore << " -> " << g_cpdRmseAfter
              << "  IoU " << g_cpdIoUBefore << " -> " << g_cpdIoUAfter << std::endl;
    return true;
}

// --- One-shot driver: run stages 1 -> 5 (used by Ctrl+U and "Run all") --
inline void runCpdRegistration() {
    cpdResetPipeline();
    if (!cpdStage1_extractTarget()) return;
    if (!cpdStage2_downsample())    return;
    if (!cpdStage3_buildSource())   return;
    if (!cpdStage4_runEM())         return;
    cpdStage5_apply();
}

// Revert the applied CPD transform (leaves stages 1-4 intact for re-apply).
inline void revertCpd() { cpdUncheckFrom(5); }

// =========================================================================
//  [LIVE] Frame-driven convergence animation for CPD and soft-ICP.
//
//  Mirrors NormalRefineLive (Shift+N) but uses a record-then-replay scheme:
//  the solver runs once (fast), recording the cumulative transform after every
//  iteration; the main render loop then replays that trajectory one (or
//  g_regLiveStepsPerFrame) entries per frame via applyIncrementalTransform, so
//  the mesh visibly converges. The validated CPD/ICP math is untouched.
//
//  Lifecycle (parallels NormalRefineLive):
//    caller: poseAutoSaveBeforeRegistration()  + startXxxLive()
//    main loop: tickXxxLive() each frame  -> [last tick] finishXxxLive()
//    finish sets pendingSave -> main.cpp consumes it and calls poseSaveToLibrary
//
//  Concurrency: a single live session at a time across all three (NormalRefine,
//  CPD, soft-ICP). Starting one while another runs is refused.
// =========================================================================
namespace CpdLive {
    inline bool                    active      = false;
    inline bool                    pendingSave = false;
    inline std::vector<glm::dmat4> traj;                 // per-iteration cumulative T
    inline glm::dmat4              applied     = glm::dmat4(1.0); // T currently on the mesh
    inline int                     pi          = 0;       // entries replayed so far
}
namespace SoftIcpLive {
    inline bool                    active      = false;
    inline bool                    pendingSave = false;
    inline std::vector<glm::dmat4> traj;
    inline glm::dmat4              applied     = glm::dmat4(1.0);
    inline int                     pi          = 0;
}

inline void finishCpdLive();
inline void finishSoftIcpLive();

// True if any frame-driven session is currently animating.
inline bool anyRegLiveActive() {
    return NormalRefineLive::active || CpdLive::active || SoftIcpLive::active;
}

// Advance a replay by g_regLiveStepsPerFrame and apply the composite delta.
// Returns true when the trajectory has been fully replayed.
inline bool replayAdvance(std::vector<glm::dmat4>& traj, glm::dmat4& applied, int& pi) {
    const int N = (int)traj.size();
    if (N == 0) return true;
    const int steps  = std::max(1, g_regLiveStepsPerFrame);
    const int target = std::min(pi + steps, N);
    if (target > pi) {
        const glm::dmat4 Tt    = traj[target - 1];
        const glm::dmat4 delta = Tt * glm::inverse(applied);
        auto organs = getOrganList();
        NormalRefine::applyIncrementalTransform(delta, organs);
        applied = Tt;
        pi      = target;
    }
    return pi >= N;
}

// ---- CPD live ----------------------------------------------------------
inline bool startCpdLive() {
    if (CpdLive::active) { finishCpdLive(); return false; } // press-to-stop -> jump to end
    if (anyRegLiveActive()) {
        std::cerr << "[CPD LIVE] another live session is active — stop it first" << std::endl;
        return false;
    }
    // Run stages 1-4 (these do NOT move the mesh) with trajectory recording.
    cpdResetPipeline();
    if (!cpdStage1_extractTarget()) return false;
    if (!cpdStage2_downsample())    return false;
    if (!cpdStage3_buildSource())   return false;
    CpdLive::traj.clear();
    if (!cpdStage4_runEM(&CpdLive::traj)) return false;     // logs collapse/fail; sets g_cpdResult
    if (CpdLive::traj.empty()) {
        // Degenerate (0 iterations). Apply directly via the blocking stage 5.
        std::cout << "[CPD LIVE] no trajectory (0 iters) — applying directly" << std::endl;
        cpdStage5_apply();
        return false;
    }
    CpdLive::applied     = glm::dmat4(1.0);
    CpdLive::pi          = 0;
    CpdLive::active      = true;
    CpdLive::pendingSave = false;
    std::cout << "[CPD LIVE] start: " << CpdLive::traj.size()
              << " frames  RMSE_before=" << std::fixed << std::setprecision(5)
              << g_cpdRmseBefore << std::defaultfloat << std::endl;
    return true;
}
inline void tickCpdLive() {
    if (!CpdLive::active) return;
    if (replayAdvance(CpdLive::traj, CpdLive::applied, CpdLive::pi))
        finishCpdLive();
}
inline void finishCpdLive() {
    if (!CpdLive::active) return;
    // Mesh now sits at the final CPD pose (= traj.back() == g_cpdResult.T).
    g_cpdAppliedT  = CpdLive::traj.empty() ? g_cpdResult.T : CpdLive::traj.back();
    g_cpdCanRevert = true;
    computeUnifiedMetrics();
    g_cpdRmseAfter = registrationHandle.compRmse;
    g_cpdIoUAfter  = registrationHandle.compIoU2D;
    g_cpdStageDone[5] = true;
    std::cout << "[CPD LIVE] DONE  RMSE " << std::fixed << std::setprecision(5)
              << g_cpdRmseBefore << " -> " << g_cpdRmseAfter
              << "  IoU " << g_cpdIoUBefore << " -> " << g_cpdIoUAfter
              << std::defaultfloat << std::endl;
    CpdLive::pendingSave = true;
    CpdLive::active      = false;
}

// ---- soft-ICP live -----------------------------------------------------
inline bool startSoftIcpLive() {
    if (SoftIcpLive::active) { finishSoftIcpLive(); return false; }
    if (anyRegLiveActive()) {
        std::cerr << "[Soft U-2 LIVE] another live session is active — stop it first" << std::endl;
        return false;
    }
    SoftIcpLive::traj.clear();
    runSoftWeightedICP(&SoftIcpLive::traj, /*applyResult=*/false); // records, does NOT move mesh
    if (SoftIcpLive::traj.empty()) {
        std::cerr << "[Soft U-2 LIVE] no iterations recorded (not started)" << std::endl;
        return false;
    }
    SoftIcpLive::applied     = glm::dmat4(1.0);
    SoftIcpLive::pi          = 0;
    SoftIcpLive::active      = true;
    SoftIcpLive::pendingSave = false;
    std::cout << "[Soft U-2 LIVE] start: " << SoftIcpLive::traj.size()
              << " frames  RMSE_before=" << std::fixed << std::setprecision(5)
              << g_softIcpRmseBefore << std::defaultfloat << std::endl;
    return true;
}
inline void tickSoftIcpLive() {
    if (!SoftIcpLive::active) return;
    if (replayAdvance(SoftIcpLive::traj, SoftIcpLive::applied, SoftIcpLive::pi))
        finishSoftIcpLive();
}
inline void finishSoftIcpLive() {
    if (!SoftIcpLive::active) return;
    g_softIcpAppliedT  = SoftIcpLive::traj.empty() ? glm::dmat4(1.0)
                                                   : SoftIcpLive::traj.back();
    g_softIcpCanRevert = true;
    computeUnifiedMetrics();
    g_softIcpRmseAfter = registrationHandle.compRmse;
    g_softIcpIoUAfter  = registrationHandle.compIoU2D;
    std::cout << "[Soft U-2 LIVE] DONE  RMSE " << std::fixed << std::setprecision(5)
              << g_softIcpRmseBefore << " -> " << g_softIcpRmseAfter
              << "  IoU " << g_softIcpIoUBefore << " -> " << g_softIcpIoUAfter
              << std::defaultfloat << std::endl;
    SoftIcpLive::pendingSave = true;
    SoftIcpLive::active      = false;
}
