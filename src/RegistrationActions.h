#pragma once
// RegistrationActions.h
// HemiAuto / BIPOP-CMA-ES / Shift+E (SilhouetteAlign) / Metrics
// main.cppのグローバル変数をexternで参照する。

#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>
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
#include "LiverLeftRightLabel.h"   // V3R: g_liverLR 参照のため (PURE_R/BOUNDARY/PURE_L) + QuadrantMask
#include "LiverCranioCaudalLabel.h" // V3R-W: g_liverCC 参照のため (CRANIAL/CAUDAL, Ctrl+G Only-Caudal filter)
#include "FullSphereCameraWithTarget.h"

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

extern FullSphereCamera OrbitCam;
extern RegistrationData registrationHandle;

extern glm::mat4 model, view, projection;
extern glm::vec3 objPos;
extern int   gWindowWidth, gWindowHeight;
extern int   gGridWidth;
extern float gDepthScale;
extern float g_voxelSize;

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
inline float g_instrumentPxThresh = 20.0f;

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
inline float g_ctrlgBetaRimWeight    = 0.0f;
inline float g_ctrlgRimTgtThreshPx   = 12.0f;   // matches Shift+P kBoundaryPxTh
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
inline bool    g_ctrlgUseCaudalOnly       = false;
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
inline bool         g_silProjShow = true;  // UI toggle (default ON)

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
inline int   g_qcrSubsetK        = 3;     // K=3/4/5 (slider). 3=Fischler-Bolles min sample, 4-5=more stable
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
inline float g_qcrMaxAxisRotDeg    = 90.0f; // hard limit per X/Y/Z axis rotation (deg)

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
// =========================================================
inline float g_lastQcrPrealignScale = -1.0f;

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
    const float maxRotDeg = g_qcrMaxAxisRotDeg;     // hard limit per axis (deg)
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
              << ", max_rot=" << std::setprecision(1) << maxRotDeg << "deg)"
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
inline void runQuadCyclicRansac(bool lock_scale = false) {
    std::cout << "\n=== QuadCyclic-RANSAC Registration (Shift+Ctrl+P)"
              << (lock_scale ? "  [6-DoF rigid]" : "  [7-DoF T+R+S]")
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
        // (A) Axis rotation hard limit
        extractAxisRotDeg(c.T, c.axisRotDeg);
        const float maxAxis = std::max({std::abs(c.axisRotDeg[0]),
                                        std::abs(c.axisRotDeg[1]),
                                        std::abs(c.axisRotDeg[2])});
        c.rotOK = (maxAxis <= maxRotDeg);
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
              << "  (" << std::setprecision(1) << stage2_ms << " ms)"
              << std::defaultfloat << std::setprecision(6) << std::endl;

    // Stage 2 が全 reject されたら fallback: hard limit を無視して
    // score_chamfer のみで best を選ぶ (緊急脱出)。
    if (bestIdx < 0) {
        std::cerr << "[QCR] All " << topK.size() << " candidates rejected by "
                     "axis rotation limit (" << maxRotDeg
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

        // (6) sanity: scale, axis rotation
        const float refScale = glm::length(glm::vec3(T_refined[0]));
        if (!std::isfinite(refScale) || refScale < kScaleLo || refScale > kScaleHi) break;
        float refDeg[3];
        extractAxisRotDeg(T_refined, refDeg);
        const float refMaxAxis = std::max({std::abs(refDeg[0]),
                                           std::abs(refDeg[1]),
                                           std::abs(refDeg[2])});
        if (refMaxAxis > maxRotDeg) break;  // 暴走: refinement 結果が hard limit 違反

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

    // ---- 9. bestT を全 organMeshes に適用 (Ctrl+P と同じ) ----
    auto organs = getOrganList();
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

    // ---- 11. メトリクス (Ctrl+P と同じ) ----
    computeUnifiedMetrics();
    g_metricsValid = true;
    registrationHandle.state           = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    std::cout << "=== QuadCyclic-RANSAC Complete  RMSE=" << registrationHandle.compRmse
              << "  mask=" << LiverLeftRightLabel::quadrantMaskString(med.mask)
              << "  (RANSAC chamfer was " << bestScore << ")"
              << " ===" << std::endl;

    // Phase 1 観察用: RANSAC prealign の estScale を publish。
    // AutoQCR の loop 側が consume-and-clear で読み取る。
    g_lastQcrPrealignScale = estScale;

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
    // [Phase A skip] If the previous Phase F (or RANSAC) already ran
    // computeUnifiedMetrics for the current pose, skip the call and
    // read from registrationHandle directly. Saves 135-225 ms per call.
    // g_metricsValid is cleared here (consumed) so Phase F must re-set it.
    if (!g_metricsValid || registrationHandle.compRmse <= 0.0f) {
        computeUnifiedMetrics();
        g_metricsValid = true;
    }
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
    // [Phase F skip] If the driver found no improvement (r.improved==false),
    // the pose is unchanged from Phase A → registrationHandle already
    // holds the correct values. Skip computeUnifiedMetrics and save
    // 122-155 ms. Set g_metricsValid=true so the next Phase A can skip.
    // If improved, always run to get fresh post-apply RMSE and IoU2D.
    if (r.improved) {
        computeUnifiedMetrics();
        g_metricsValid = true;
    }
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

    if (registrationHandle.compRmse == 0.0f) {
        std::cerr << "[Ctrl+G] No registration yet. Run HemiAuto (O) first."
                  << std::endl;
        return;
    }

    // ----- V3R label validity gate ----------------------------------
    // Region/LR labels must be populated before any non-QUAD_ALL run;
    // for QUAD_ALL we still verify them so a misconfigured run cannot
    // silently fall back to "no filtering" semantics that the user
    // didn't intend.
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[Ctrl+G] Region/LR labels not computed: "
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
    // [Phase A skip] Same logic as runBipopCmaesV3: reuse cached
    // registrationHandle values if g_metricsValid is set.
    if (!g_metricsValid || registrationHandle.compRmse <= 0.0f) {
        computeUnifiedMetrics();
        g_metricsValid = true;
    }
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

    // ----- Phase C2b: rim-flag arrays (opt-in; needed iff beta>0) ---
    if (p.beta_rim_weight > 0.0f) {
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
            std::cout << "[Ctrl+G/rim] target cloud has no boundaryDist "
                         "(legacy path?). Rim weighting will be inactive."
                      << std::endl;
            p.tgt_boundary_dist_full.clear();
        }
        int n_rim_src = 0;
        for (uint8_t v : p.is_rim_orig) if (v) n_rim_src++;
        std::cout << "[Ctrl+G/rim] beta=" << p.beta_rim_weight
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
    // [Phase F skip] NO CHANGE → pose unchanged → skip. IMPROVED → run.
    // g_metricsValid=true so next Phase A can skip.
    if (r.improved) {
        computeUnifiedMetrics();
        g_metricsValid = true;
    }
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
    // [Phase A skip] Same logic as runBipopCmaesV3: reuse cached
    // registrationHandle values if g_metricsValid is set.
    if (!g_metricsValid || registrationHandle.compRmse <= 0.0f) {
        computeUnifiedMetrics();
        g_metricsValid = true;
    }
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
        const bool rmse_cap_ok = (rmse_cand < rmse_before * cap_factor);

        if (score_after < score_before && rmse_cap_ok) {
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

    g_callIdx++;  // V3R: 末尾でインクリメント (V1 / V2 / V3 と同じ)
}


// =========================================================
//  diagnoseVertexSquashV3RS (F10) -- vertex-squash A/B raster diagnostic
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
inline void runShiftE() {
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
            liverMesh3D, view, projection, 8);

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

    const int N_STARTS = 5;

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
        p.use_silhouette_2d_fast = true;
        p.alpha_silhouette       = 1.0f;
        p.alpha_3d               = 0.3f;
        p.silhouette_step        = 8;
        p.maxgen = 300;
        p.tolfun = 1e-4;
        // CMA-ES sampling range scaled by sceneDiag (was 1.0f)
        const float gtE = g_sceneDiag * 0.5f;
        p.tx_range = gtE; p.ty_range = gtE; p.tz_range = gtE;
        p.rx_range = 20.0f; p.ry_range = 20.0f; p.rz_range = 20.0f;
        p.scale_lo = 0.85f; p.scale_hi = 1.15f;
        // Phase 1: CMA-ES 内部 srand を固定
        p.rng_seed = cma_base + (uint32_t)run;

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
    computeUnifiedMetrics();
    g_metricsValid = true;

    std::cout << std::defaultfloat << std::setprecision(6);
    float iou_delta = best_iou - iou_before;
    std::cout << "[Shift+E] IoU: " << iou_before << " -> " << best_iou
              << " (delta=" << iou_delta << ")"
              << (iou_delta > 0.001f ? " [IMPROVED]" : " [NO CHANGE]")
              << std::endl;

    g_callIdx++;  // Phase 1: 末尾でインクリメント
}
