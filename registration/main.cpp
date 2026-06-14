#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <functional>
#include <random>
#include <algorithm>
#include <numeric>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ShaderProgram.h"
#include "Sphere.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <ctime>

// SimpleCameraをSTB_IMAGE_IMPLEMENTATIONが有効な間にインクルード
#include "SimpleCamera.hpp"

#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "CameraPreview.h"
#include "mCutMesh.h"
#include "MeshDrawing.h"
#include "ScreenMeshPoints.h"   // ScreenMeshPointCache (shared GL_POINTS draw)
#include "FullSphereCameraWithTarget.h"
#include "DepthUtils.h"
#include "IoUDebugDump.h"   // IoUDebug::dump (Shift+I)
#include "RegistrationCore.h"
#include "NoOpen3DRegistration.h"

#include "OBJTargetExtraction.h"
#include "IntrinsicsSource.h"
#include "IntrinsicsPresets.h"
#include "DepthToObjExport.h"   // obj-migration Phase 3: REG-side OBJ export
#include "IntrinsicsScaling.h"  // scaleIntrinsics() (FEATURE Task 1/2)
#include "ReconstructFromBin.h" // Reconstruct-from-BIN (FEATURE Task 3/4)
#include "MeshCleanup.h"
#include "AR.h"
#include "SilOverlayDebug.h"   // V3RS Phase 2: silhouette IoU ImGui overlay (F9 toggle)

#include <filesystem>
#include <fstream>     // Shift+M snapshot README.txt 生成用
#include <chrono>      // Shift+M snapshot timestamp 用 (transitively 入っているはずだが明示)
#include "AppContext.h"
#include "ImageSession.h"
#include "MaskPicker.h"
#include "DepthRunner.h"
#include "PathConfig.h"
#include "LiverRegionLabel.h"     // Shift+R/Shift+T: 肝臓領域 (anterior/rim/posterior) ラベリング
#include "LiverLeftRightLabel.h"  // Y/Shift+Y: 肝臓左右ラベリング (pure-R/boundary/pure-L)
#include "LiverCranioCaudalLabel.h"  // Shift+H: 肝臓頭尾ラベリング (cranial/caudal) — Phase 1

#ifdef HAS_TINYFILEDIALOGS
#include "tinyfiledialogs.h"
#endif

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "RegistrationImGuiManager.h"
#include "DebugPanel.h"
#include "StlExport.h"
#include "UmeyamaController.h"
#include "SoftPartition.h"   // Phase U-1 (グローバル定義のため明示 include)
// PoseLibrary.h は RegistrationActions.h の後で include する
// (RegistrationActions.h 内の inline computeUnifiedMetrics 定義を見えてからで
//  ないと、PoseLibrary.h の inline 関数本体が unresolved になるため)

// キャリブレーション結果（外部exe calibration_tool から読み込み）
struct CalibResult {
    double fx=0, fy=0, cx=0, cy=0, k1=0, k2=0;
    int width=0, height=0;
    double rmsError=0;
    int numImages=0;
    bool valid=false;
    std::string message;
};

int gWindowWidth  = 1280;
int gWindowHeight = 720;
GLFWwindow* gWindow = nullptr;

glm::mat4 model(1.0f), view(1.0f), projection(1.0f);
glm::vec3 objPos(0.0f);

FullSphereCamera OrbitCam;

RegistrationData registrationHandle;
std::function<void(float, const char*)> g_progressCallback = nullptr;

std::vector<glm::vec3> g_cluster1Points;
std::vector<glm::vec3> g_cluster2Points;
std::vector<glm::vec3> g_targetPoints;
std::vector<glm::vec3> g_rejectedBoundaryPoints;   // 器具マスクで棄却された偽境界
std::vector<glm::vec3> g_visibleSourcePoints;      // SilhouetteHemi 全可視ソース頂点
std::vector<glm::vec3> g_silhouetteSourcePoints;   // SilhouetteHemi シルエット絞り込み後

// Phase 7b Step 1 (Plain W) — extern 宣言は RegistrationActions.h
std::vector<int> g_debugSourceRimChain;     // RIM 頂点 index chain (順序付け済)
bool             g_showDebugSourceRimChain = false;

// Phase 7b Step 1 PCA cache (populate 関数で書き込まれる、Step 3 で再利用)
glm::vec3 g_debugSourceRimCentroid      = glm::vec3(0.0f);
glm::vec3 g_debugSourceRimMajorNormal   = glm::vec3(0.0f, 0.0f, 1.0f);
glm::vec3 g_debugSourceRimPrincipalAxis = glm::vec3(1.0f, 0.0f, 0.0f);
double    g_debugSourceRimPlanarity     = 1.0;

// Phase 7b Step 2 (Shift+W) — target side rim band 3D point cloud
std::vector<glm::vec3> g_debugTargetBoundaryPoints;
bool                   g_showDebugTargetBoundary = false;

// Phase 7c (REDGE/稜線) — extern 宣言は RegistrationActions.h
//   source ridge: ANTERIOR_CORE のうち AR カメラに対し法線が grazing な
//   頂点 = ドームのオクルーディング輪郭 (境界 ∩ 非RIM ∩ 非後面)。頂点
//   index 保持 → mVertices から fetch して mesh 追従 (rim chain と同じ)。
std::vector<int> g_debugSourceRidge;
bool             g_showDebugSourceRidge = false;
//   target ridge: g_debugTargetBoundaryPoints の上半分 (p.y <= 2D
//   centroid.y)。target は静止なので 3D 直接保持。
std::vector<glm::vec3> g_debugTargetRidgePoints;
bool                   g_showDebugTargetRidge = false;
//   source grazing band: |cos(n, -view)| < これ → silhouette とみなす。
float g_ctrlgRidgeCosBand = 0.30f;   // 既定 ~±17.5deg around edge-on
//   target ridge outlier removal (debug toggle, 稜線にだけ適用)。
bool  g_ridgeTgtRemoveOutliers = false;   // 既定 OFF (生を基準に残す)
int   g_ridgeTgtOutlierMode    = 0;       // 0=SOR, 1=largest connected component
int   g_ridgeTgtSorK           = 16;      // SOR: k nearest neighbors
float g_ridgeTgtSorStd         = 2.0f;    // SOR: mean + std*sigma 超えを除外
float g_ridgeTgtCcRadius       = 0.0f;    // CC: 0 → auto (median NN dist * 2.5)
int   g_ridgeTgtCcMinPts       = 50;      // CC: この点数未満のクラスタを捨てる
bool  g_ridgeTgtSplitLongEdge  = false;   // 上下分割: false=up/down(y), true=bbox長辺(PCA短軸)

// Phase 7b Step 3 (Ctrl+W) — Shape Match coarse search outputs
std::vector<glm::vec3> g_debugShapeMatchBestSrc;
bool                   g_showDebugShapeMatch     = false;
double                 g_debugShapeMatchBestCost = 0.0;
int                    g_debugShapeMatchBestK    = -1;
glm::mat4              g_debugShapeMatchBestTransform = glm::mat4(1.0f);

// Phase 7b Step 3a (Ctrl+W, 2D mode) — target 2D contour cache.
//   populated by Shift+W (populateDebugTargetBoundary), consumed by
//   Ctrl+W when g_shapeMatchUse2DCost is true. Pixel coordinates in
//   the g_boundaryDistMap frame.
std::vector<glm::vec2> g_debugTargetContour2D;
std::vector<int>       g_debugTargetContour2DSegSizes;

// Phase 7b Step 3b (Alt+W) — Gauss-Newton refinement cache.
//   g_gnUnsignedBdy: unsigned distance-to-boundary field for the entire
//   image (BFS from boundary contour in both directions, providing a
//   smooth gradient field for LM optimization). Built lazily on first
//   Alt+W call, invalidated by g_boundaryDistMap.invalidate() callers.
std::vector<float> g_gnUnsignedBdy;
int                g_gnUnsignedBdyW     = 0;
int                g_gnUnsignedBdyH     = 0;
bool               g_gnUnsignedBdyValid = false;

// Phase 7b Step 3c (Ctrl+Alt+W) — Contour Sweep state lives in
// RegistrationActions.h as an inline variable; no definition needed here.
bool g_showClusterVisualization = false;
bool g_showBoundaryCandidates   = false;
bool g_showSourceVisualization  = false;
bool g_quietMetrics             = false;

mCutMesh* liverMesh3D   = nullptr;
mCutMesh* portalMesh3D  = nullptr;
mCutMesh* veinMesh3D    = nullptr;
mCutMesh* tumorMesh3D   = nullptr;
mCutMesh* segmentMesh3D = nullptr;
mCutMesh* gbMesh3D      = nullptr;
mCutMesh* screenMesh    = nullptr;
mCutMesh* boardMesh3D   = nullptr;  // テクスチャ付き表示用メッシュ（full OBJ）

int   gGridWidth  = 128;
float gDepthScale = 0.3f;
float g_voxelSize = 0.3f;


bool      isDragging   = false;
int       hit_index    = -1;
glm::vec3 hit_position(0.0f);

// ---- Phase U-1: SoftPartition グローバル状態 (ここで定義、ヘッダで extern) ----
SoftPartition::AnchorSet       g_softAnchors;
SoftPartition::PartitionField  g_softTgtField;
SoftPartition::PartitionField  g_softSrcField;
SoftPartition::PartitionParams g_softPartParams;
std::vector<float>             g_softGroupRadii;
float                          g_softUmeyamaScale = 1.0f;
bool                           g_softShowHeatmap  = false;
bool                           g_softPartReady    = false;
// ヒートマップ描画は既存 g_sphereMarker を再利用 (新 marker 不要)

// ---- Phase U-2: soft-weighted ICP state ----
SoftPartition::SoftICPParams   g_softIcpParams;
int                            g_softIcpLastIters  = 0;
float                          g_softIcpRmseBefore = 0.0f;
float                          g_softIcpRmseAfter  = 0.0f;
float                          g_softIcpIoUBefore  = 0.0f;
float                          g_softIcpIoUAfter   = 0.0f;
glm::dmat4                     g_softIcpAppliedT   = glm::dmat4(1.0); // last applied (for revert)
bool                           g_softIcpCanRevert  = false;
bool                           g_softShowGroups    = true;            // heatmap color groups visible

// Per-group heatmap visibility (G0..G7). A checkbox next to each Gi in the U
// tab toggles these. Default all-on => identical to the previous behaviour.
bool g_softGroupVisible[SoftPartition::MAX_GROUPS] =
    { true, true, true, true, true, true, true, true };

// Visibility of the "none" points (vertices that don't belong to any group,
// i.e. none-mass > 0.5). Toggled by the extra "none" checkbox in the legend.
// Default ON => identical to the previous behaviour (none points always drawn).
bool g_softShowNone = true;

// Decide whether a heatmap vertex should be drawn given the per-group / none /
// show-groups toggles. `p` is the soft prob vector, numGroups the active group
// count. A vertex is "grouped" when its none-mass <= 0.5 (matches the legacy
// show-groups test): grouped points obey show-groups AND their dominant group's
// checkbox; none points (none-mass > 0.5) obey the "none" checkbox. The heatmap
// master toggle still gates the whole overlay upstream.
static inline bool softHeatmapVertexVisible(
        const std::array<float, SoftPartition::MAX_GROUPS + 1>& p, int numGroups) {
    const bool grouped = (p[SoftPartition::NONE_IDX] <= 0.5f);
    if (!grouped)            return g_softShowNone;  // none points: own toggle
    if (!g_softShowGroups)   return false;    // colored points hidden (legacy)
    int dom = 0; float best = -1.0f;
    for (int i = 0; i < numGroups && i < SoftPartition::MAX_GROUPS; ++i)
        if (p[i] > best) { best = p[i]; dom = i; }
    return g_softGroupVisible[dom];           // per-group filter
}
std::vector<mCutMesh*> allMeshes;
float scaleSpeed = 1.1f;

// AR variables moved to AppContext

float                          g_silhouetteCosThreshold = 0.02f;
std::string                    g_objSourcePath;
Reg3DCustom::CameraIntrinsics  g_intrinsics;
// Distorted calibration kept aside when an image is rectified: g_intrinsics then
// holds the distortion-free K matching original_rectified.jpg, while
// g_intrinsics_raw retains the original (distortion-bearing) K so a subsequent
// image load can still decide/perform rectification. Invalidated whenever K is
// (re)loaded from a source. See loadImageRectifyAware().
Reg3DCustom::CameraIntrinsics  g_intrinsics_raw;
// Reconstruct-from-BIN drop state (FEATURE Task 3/4). Filled by file/folder drops
// of depth_metric.bin + segmentation_mask.png (+ original.jpg); consumed by the
// Reconstruct action (Task 6) + UI (Task 5).
ReconstructFromBin::State     g_reconstructState;

// =========================================================
//  Scene scale (for size-invariant registration parameters)
// ---------------------------------------------------------
//  After prealignSourceToTarget() runs, source and target are
//  approximately the same size. g_sceneDiag captures that size
//  (AABB diagonal of the target, in metric units). All distance
//  / voxel / search parameters in the registration pipeline are
//  derived from it via constants in RegistrationActions.h, so
//  the scene works directly in meters with no display scaling.
// =========================================================
float       g_sceneDiag         = 1.0f;
glm::mat4   g_lastOrganTransform = glm::mat4(1.0f);  // similarity transform applied at last setup
bool        g_hasLastOrganTransform = false;

// =========================================================
//  [CT-handoff] Source .obj of the CURRENT liverMesh3D in its raw
//  (pre-registration / pre-prealign) coordinates.
// ---------------------------------------------------------
//  CT-handoff (StlExport::exportRegisteredObjs) recovers the
//  CT->registered similarity T via Umeyama, which needs a 1:1 vertex
//  correspondence between the registered liverMesh3D and the raw CT
//  liver it came from. It used to assume that raw CT is always
//  MODEL_PATH + "liver.obj" (the startup model). That breaks when a
//  different organ set is dropped in mid-session via
//  swapOrganSetFromDrop: the dropped liver may have a different vertex
//  count, the counts mismatch, and reg_transform.txt / ct_roundtrip get
//  skipped (the REG->DEFORM handoff silently does not happen).
//
//  We therefore record where the current liver actually came from:
//    - startup / setupObjScene load  -> MODEL_PATH + "liver.obj"
//    - NEW CT organ-set drop          -> the dropped liver.obj path
//    - preserve-pose (deform round-trip) drop -> NOT updated: that mesh
//      is in registered coords, not raw CT, so the previous CT source
//      stays valid.
//  Empty string -> fall back to MODEL_PATH + "liver.obj".
// =========================================================
std::string g_liverCtSourcePath;   // raw-CT source of current liverMesh3D ("" = fallback)

// =========================================================
//  Original CT-mm diagonals of liver/tumor as loaded from model/*.obj
// ---------------------------------------------------------
//  Captured ONCE at startup, immediately after the .obj files are
//  loaded into liverMesh3D / tumorMesh3D and BEFORE any prealign or
//  registration touches them. Used by Shift+M to compute the inverse
//  scale needed to bring the registered mesh back to true CT-mm size:
//
//      SCALE_RESTORE = g_originalLiverDiagMm / current_liver_diag
//
//  This is invariant across:
//    - prealignSourceToTarget (uniform scale s_prealign)
//    - any subsequent registration (CMA-ES, hand alignment, ICP)
//    - manual user nudges
//  as long as the operations stay in the rigid-similarity family.
//  We use the LIVER diag (not tumor) because liver is ~10x larger and
//  therefore numerically more stable; the same scale factor is applied
//  to all organs since they all share the same transform chain.
// =========================================================
float       g_originalLiverDiagMm = 0.0f;  // diag(liver.obj) in CT mm, set once at startup
float       g_originalTumorDiagMm = 0.0f;  // diag(tumor.obj) in CT mm, set once at startup
bool        g_hasOriginalDiags    = false;

// =========================================================
//  [deform round-trip] reg -> CT-mm 固定スケール
// ---------------------------------------------------------
//  上の SCALE_RESTORE = g_originalLiverDiagMm / current_liver_diag は
//  「現在の liver が CT の相似変換である」前提(rigid-similarity family)で
//  のみ正しい。DEFORM で非剛体変形した liver を戻すとこの前提が崩れ、
//  current_liver_diag に変形分が混ざって Shift+M の mm スケールがずれる。
//  そこで「まだ未変形(=相似変換)の段階」で reg->mm スケールを一度だけ
//  捕捉して固定し、以降の Shift+M はこれを使う。捕捉は最初の「Deform >>」
//  (exportRegisteredObjs) で行い、生CTモデルを新規ドロップしたらリセットする。
//  pure REG 運用(Deform >> を押さない)では未捕捉のまま従来式にフォールバック。
// =========================================================
float       g_regToMmScale    = 0.0f;
bool        g_hasRegToMmScale = false;

// =========================================================
//  Target subset AABB (Phase 2 拡張: 重心 Position に応じた initial scale)
//  ---------------------------------------------------------
//  target cloud (screenMesh の頂点) を world X 軸の中点で左右に分割し、
//  3 つの AABB を保持する:
//    g_targetAabbFull : 全体 (Position=Center で使用)
//    g_targetAabbXneg : world -X 半分 (画面の左 = radiology 慣例「患者の右」 = Position=Right)
//    g_targetAabbXpos : world +X 半分 (画面の右 = radiology 慣例「患者の左」 = Position=Left)
//
//  Position 選択時、liver mesh を「g_targetAabbFull.diag → subset.diag」の
//  比率でスケールダウン + 重心を subset.center に平行移動する。
//
//  これにより、UI で "Right" を選ぶと、肝臓は target の left half (= 画面の左) の
//  AABB に合わせてスケール + 配置される (radiology: 患者の右側)。
//
//  下流影響:
//   - Shift+M: g_originalLiverDiagMm / current_liver_diag で復元するので
//     Position 変更で liver_diag が変わっても自動的に正しい scale で復元される
//   - AutoProbe / HemiAuto / CMA-ES: 現在の liver state を起点に動作するので影響なし
//   - PoseLibrary: 現在の transform を記録するので影響なし
//                  g_currentOrientLabel に "Base @ Right" の形式で position も記録
// =========================================================
struct TargetSubsetAabb {
    glm::vec3 min    = glm::vec3(0.0f);   // ★チャット 10: AABB 最小コーナー (BB 描画用)
    glm::vec3 max    = glm::vec3(0.0f);   // ★チャット 10: AABB 最大コーナー
    glm::vec3 center = glm::vec3(0.0f);   // AABB midpoint = 0.5*(min+max)
    glm::vec3 mean   = glm::vec3(0.0f);   // vertex mean (info)
    float     diag   = 0.0f;
    bool      valid  = false;
};
TargetSubsetAabb g_targetAabbFull;
TargetSubsetAabb g_targetAabbXpos;   // world +X half (画面右)
TargetSubsetAabb g_targetAabbXneg;   // world -X half (画面左)

// 初期化時 (prealign 後) または target 再構築時に呼ぶ。
// screenMesh.mVertices から AABB を計算する。
static void computeTargetSubsetAabbs() {
    g_targetAabbFull.valid = false;
    g_targetAabbXpos.valid = false;
    g_targetAabbXneg.valid = false;

    if (!screenMesh || screenMesh->mVertices.empty()) {
        std::cerr << "[TargetAABB] screenMesh empty; skipping AABB computation"
                  << std::endl;
        return;
    }

    const auto& V = screenMesh->mVertices;
    const int nV  = (int)(V.size() / 3);
    if (nV < 3) {
        std::cerr << "[TargetAABB] too few vertices (" << nV << "); skipping"
                  << std::endl;
        return;
    }

    // 全体の AABB + mean (vertex 平均 = 真の重心)
    glm::vec3 mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum_all(0.0f);
    for (int i = 0; i < nV; i++) {
        glm::vec3 p(V[i*3], V[i*3+1], V[i*3+2]);
        mn = glm::min(mn, p);
        mx = glm::max(mx, p);
        sum_all += p;
    }
    g_targetAabbFull.min    = mn;
    g_targetAabbFull.max    = mx;
    g_targetAabbFull.center = 0.5f * (mn + mx);
    g_targetAabbFull.mean   = sum_all / (float)nV;
    g_targetAabbFull.diag   = glm::length(mx - mn);
    g_targetAabbFull.valid  = (g_targetAabbFull.diag > 1e-9f);

    // X 軸中点 (= AABB midpoint の x) で 2 分割
    const float x_mid = g_targetAabbFull.center.x;

    glm::vec3 mn_P( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx_P(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum_P(0.0f);
    glm::vec3 mn_N( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx_N(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum_N(0.0f);
    int n_P = 0, n_N = 0;
    for (int i = 0; i < nV; i++) {
        glm::vec3 p(V[i*3], V[i*3+1], V[i*3+2]);
        if (p.x >= x_mid) {
            mn_P = glm::min(mn_P, p);
            mx_P = glm::max(mx_P, p);
            sum_P += p;
            n_P++;
        } else {
            mn_N = glm::min(mn_N, p);
            mx_N = glm::max(mx_N, p);
            sum_N += p;
            n_N++;
        }
    }
    if (n_P > 0) {
        g_targetAabbXpos.center = 0.5f * (mn_P + mx_P);
        g_targetAabbXpos.mean   = sum_P / (float)n_P;
        g_targetAabbXpos.diag   = glm::length(mx_P - mn_P);
        g_targetAabbXpos.valid  = (g_targetAabbXpos.diag > 1e-9f);
    }
    if (n_N > 0) {
        g_targetAabbXneg.center = 0.5f * (mn_N + mx_N);
        g_targetAabbXneg.mean   = sum_N / (float)n_N;
        g_targetAabbXneg.diag   = glm::length(mx_N - mn_N);
        g_targetAabbXneg.valid  = (g_targetAabbXneg.diag > 1e-9f);
    }

    std::cout << "[TargetAABB] full : c_aabb=(" << g_targetAabbFull.center.x << ","
              << g_targetAabbFull.center.y << "," << g_targetAabbFull.center.z
              << ") mean=(" << g_targetAabbFull.mean.x << ","
              << g_targetAabbFull.mean.y << "," << g_targetAabbFull.mean.z
              << ") diag=" << g_targetAabbFull.diag << "  n=" << nV << std::endl;
    std::cout << "[TargetAABB] +X   : c_aabb=(" << g_targetAabbXpos.center.x << ","
              << g_targetAabbXpos.center.y << "," << g_targetAabbXpos.center.z
              << ") mean=(" << g_targetAabbXpos.mean.x << ","
              << g_targetAabbXpos.mean.y << "," << g_targetAabbXpos.mean.z
              << ") diag=" << g_targetAabbXpos.diag << "  n=" << n_P
              << "  (= Position 'Left' in radiology)" << std::endl;
    std::cout << "[TargetAABB] -X   : c_aabb=(" << g_targetAabbXneg.center.x << ","
              << g_targetAabbXneg.center.y << "," << g_targetAabbXneg.center.z
              << ") mean=(" << g_targetAabbXneg.mean.x << ","
              << g_targetAabbXneg.mean.y << "," << g_targetAabbXneg.mean.z
              << ") diag=" << g_targetAabbXneg.diag << "  n=" << n_N
              << "  (= Position 'Right' in radiology)" << std::endl;
}

// =========================================================
//  Source liver subset AABB (Phase 2 拡張 v2)
//  ---------------------------------------------------------
//  Source mesh (liver) の頂点を world X 軸の中点で左右に分割し、
//  3 通りの AABB を保持する。target subset AABB との「対応する半分どうし」
//  でスケール・位置をマッチさせるために使う:
//
//    Position=Right (患者の右):
//      scale = target_-X.diag / source_-X.diag
//      → source の右半分のサイズを、target の右半分 (world -X) に合わせる
//
//    Position=Left (患者の左):
//      scale = target_+X.diag / source_+X.diag
//      → source の左半分のサイズを、target の左半分 (world +X) に合わせる
//
//    Position=Center:
//      scale = target_full.diag / source_full.diag = 1.0 (prealign 済みのため)
//
//  これによって「ターゲットが片側だけしか見えない (occlusion 等)」状況でも、
//  source の対応する半分のサイズに合わせるので、source 全体の縮尺が崩れない。
//
//  ※ snapshotInitialPose() の直後 (= prealign 済みの state) で計算する。
//  ※ world X 軸の中点で分割するのは、prealign 後 liver centroid ≈ target centroid
//     のため、source の +X/-X 半分が解剖学的な左右半分にほぼ対応する。
// =========================================================
struct SourceSubsetAabb {
    glm::vec3 min    = glm::vec3(0.0f);   // ★チャット 10: AABB 最小コーナー (回転後 AABB 計算用)
    glm::vec3 max    = glm::vec3(0.0f);   // ★チャット 10: AABB 最大コーナー
    glm::vec3 center = glm::vec3(0.0f);   // AABB midpoint = 0.5*(min+max)
    glm::vec3 mean   = glm::vec3(0.0f);   // vertex mean (旧コード後方互換のため残置)
    float     diag   = 0.0f;
    bool      valid  = false;
};
SourceSubsetAabb g_sourceLiverAabbFull;
SourceSubsetAabb g_sourceLiverAabbXpos;   // world +X half (画面右、患者の左)
SourceSubsetAabb g_sourceLiverAabbXneg;   // world -X half (画面左、患者の右)

// g_initOrganVertices は後の行で正式定義されるが、computeSourceLiverSubsetAabbs()
// から参照するため、ここで前方宣言する (extern)。
extern std::vector<std::vector<GLfloat>> g_initOrganVertices;

static void computeSourceLiverSubsetAabbs() {
    g_sourceLiverAabbFull.valid = false;
    g_sourceLiverAabbXpos.valid = false;
    g_sourceLiverAabbXneg.valid = false;

    // organs[0] = liver と仮定 (Phase 1 から一貫)
    if (g_initOrganVertices.empty() || g_initOrganVertices[0].empty()) {
        std::cerr << "[SourceAABB] g_initOrganVertices empty; skipping" << std::endl;
        return;
    }

    const auto& V = g_initOrganVertices[0];
    const int nV  = (int)(V.size() / 3);
    if (nV < 3) {
        std::cerr << "[SourceAABB] too few vertices (" << nV << "); skipping"
                  << std::endl;
        return;
    }

    // 全体の AABB + mean (vertex 平均 = 真の重心)
    glm::vec3 mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum_all(0.0f);
    for (int i = 0; i < nV; i++) {
        glm::vec3 p(V[i*3], V[i*3+1], V[i*3+2]);
        mn = glm::min(mn, p);
        mx = glm::max(mx, p);
        sum_all += p;
    }
    g_sourceLiverAabbFull.min    = mn;
    g_sourceLiverAabbFull.max    = mx;
    g_sourceLiverAabbFull.center = 0.5f * (mn + mx);
    g_sourceLiverAabbFull.mean   = sum_all / (float)nV;
    g_sourceLiverAabbFull.diag   = glm::length(mx - mn);
    g_sourceLiverAabbFull.valid  = (g_sourceLiverAabbFull.diag > 1e-9f);

    const float x_mid = g_sourceLiverAabbFull.center.x;

    glm::vec3 mn_P( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx_P(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum_P(0.0f);
    glm::vec3 mn_N( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx_N(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum_N(0.0f);
    int n_P = 0, n_N = 0;
    for (int i = 0; i < nV; i++) {
        glm::vec3 p(V[i*3], V[i*3+1], V[i*3+2]);
        if (p.x >= x_mid) {
            mn_P = glm::min(mn_P, p);
            mx_P = glm::max(mx_P, p);
            sum_P += p;
            n_P++;
        } else {
            mn_N = glm::min(mn_N, p);
            mx_N = glm::max(mx_N, p);
            sum_N += p;
            n_N++;
        }
    }
    if (n_P > 0) {
        g_sourceLiverAabbXpos.min    = mn_P;
        g_sourceLiverAabbXpos.max    = mx_P;
        g_sourceLiverAabbXpos.center = 0.5f * (mn_P + mx_P);
        g_sourceLiverAabbXpos.mean   = sum_P / (float)n_P;
        g_sourceLiverAabbXpos.diag   = glm::length(mx_P - mn_P);
        g_sourceLiverAabbXpos.valid  = (g_sourceLiverAabbXpos.diag > 1e-9f);
    }
    if (n_N > 0) {
        g_sourceLiverAabbXneg.min    = mn_N;
        g_sourceLiverAabbXneg.max    = mx_N;
        g_sourceLiverAabbXneg.center = 0.5f * (mn_N + mx_N);
        g_sourceLiverAabbXneg.mean   = sum_N / (float)n_N;
        g_sourceLiverAabbXneg.diag   = glm::length(mx_N - mn_N);
        g_sourceLiverAabbXneg.valid  = (g_sourceLiverAabbXneg.diag > 1e-9f);
    }

    std::cout << "[SourceAABB] full : c_aabb=(" << g_sourceLiverAabbFull.center.x << ","
              << g_sourceLiverAabbFull.center.y << "," << g_sourceLiverAabbFull.center.z
              << ") mean=(" << g_sourceLiverAabbFull.mean.x << ","
              << g_sourceLiverAabbFull.mean.y << "," << g_sourceLiverAabbFull.mean.z
              << ") diag=" << g_sourceLiverAabbFull.diag << "  n=" << nV << std::endl;
    std::cout << "[SourceAABB] +X   : c_aabb=(" << g_sourceLiverAabbXpos.center.x << ","
              << g_sourceLiverAabbXpos.center.y << "," << g_sourceLiverAabbXpos.center.z
              << ") mean=(" << g_sourceLiverAabbXpos.mean.x << ","
              << g_sourceLiverAabbXpos.mean.y << "," << g_sourceLiverAabbXpos.mean.z
              << ") diag=" << g_sourceLiverAabbXpos.diag << "  n=" << n_P
              << "  (= source half for Position 'Left')" << std::endl;
    std::cout << "[SourceAABB] -X   : c_aabb=(" << g_sourceLiverAabbXneg.center.x << ","
              << g_sourceLiverAabbXneg.center.y << "," << g_sourceLiverAabbXneg.center.z
              << ") mean=(" << g_sourceLiverAabbXneg.mean.x << ","
              << g_sourceLiverAabbXneg.mean.y << "," << g_sourceLiverAabbXneg.mean.z
              << ") diag=" << g_sourceLiverAabbXneg.diag << "  n=" << n_N
              << "  (= source half for Position 'Right')" << std::endl;
}

// =========================================================
//  Source liver subset AABB (Step 1: dynamic mask-based version)
//  ---------------------------------------------------------
//  任意の 4-quadrant ビットマスクから on-demand に subset AABB を計算する。
//  Ctrl+G の g_activeQuadrantMask を流用するため、Initial Orientation 側で
//  も同じ解剖学的 4 象限分割を使えるようにする (チャット 9 で確定した
//  「チェックボックス方式」)。
//
//  入力:
//    mask          : QUAD_AR / QUAD_AL / QUAD_PR / QUAD_PL の OR (任意組合せ)
//    region_labels : g_liverRegion.labels (ANTERIOR_CORE / RIM / POSTERIOR)
//    lr_labels     : g_liverLR.labels     (PURE_RIGHT / BOUNDARY / PURE_LEFT)
//
//  戻り値:
//    SourceSubsetAabb (mask が示す頂点集合の center / mean / diag / valid)
//
//  特例:
//    mask == QUAD_ALL  : 高速パス、g_sourceLiverAabbFull をそのまま返す
//                        (旧 POS_CENTER と byte-identical を保証)
//    mask == QUAD_NONE : valid=false の空 AABB を返す (UI 側で警告)
//    subset 頂点数 = 0 : valid=false の空 AABB を返す
//
//  ※ g_initOrganVertices[0] (prealign 済みの初期姿勢) から計算する。
//    現在の liverMesh3D が transform 済みでも、結果は initial pose 基準。
// =========================================================
static SourceSubsetAabb computeSourceLiverSubsetAabbFromMask(
    uint8_t mask,
    const std::vector<uint8_t>& region_labels,
    const std::vector<uint8_t>& lr_labels)
{
    SourceSubsetAabb out;   // default: valid=false

    // QUAD_ALL 高速パス: 既存の g_sourceLiverAabbFull をそのまま返す。
    // → 旧 POS_CENTER と完全同一の数値を返すので byte-identical 検証可能。
    if (mask == LiverLeftRightLabel::QUAD_ALL) {
        return g_sourceLiverAabbFull;
    }

    // 入力 sanity check
    if (g_initOrganVertices.empty() || g_initOrganVertices[0].empty()) {
        std::cerr << "[SourceAABB/Mask] g_initOrganVertices empty" << std::endl;
        return out;
    }
    if (region_labels.empty() || lr_labels.empty() ||
        region_labels.size() != lr_labels.size()) {
        std::cerr << "[SourceAABB/Mask] labels not ready (region="
                  << region_labels.size() << " lr=" << lr_labels.size() << ")"
                  << std::endl;
        return out;
    }
    if ((mask & LiverLeftRightLabel::QUAD_ALL) == 0) {
        // QUAD_NONE: 空集合
        return out;
    }

    // mask に基づく頂点 index 集合を取得 (Ctrl+G と同じ関数で同じ集合)
    auto subset_idx = LiverLeftRightLabel::makeQuadrantSubsetIdx(
        region_labels, lr_labels, mask);
    if (subset_idx.empty()) {
        return out;
    }

    const auto& V = g_initOrganVertices[0];
    const int   nV = (int)(V.size() / 3);
    if ((int)region_labels.size() != nV) {
        std::cerr << "[SourceAABB/Mask] label size mismatch: labels="
                  << region_labels.size() << " nV=" << nV << std::endl;
        return out;
    }

    // AABB + mean を計算
    glm::vec3 mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    glm::vec3 sum(0.0f);
    int n_used = 0;
    for (int idx : subset_idx) {
        if (idx < 0 || idx >= nV) continue;
        glm::vec3 p(V[idx*3], V[idx*3+1], V[idx*3+2]);
        mn = glm::min(mn, p);
        mx = glm::max(mx, p);
        sum += p;
        n_used++;
    }
    if (n_used == 0) {
        return out;
    }

    out.min    = mn;
    out.max    = mx;
    out.center = 0.5f * (mn + mx);
    out.mean   = sum / (float)n_used;
    out.diag   = glm::length(mx - mn);
    out.valid  = (out.diag > 1e-9f);
    return out;
}

// =========================================================
//  Reference length L (median NN distance of target cloud)
// ---------------------------------------------------------
//  L characterizes the local sampling resolution of the target
//  point cloud. It is the natural reference length for:
//      - voxel size (5L, Open3D / Zhou 2018)
//      - FPFH search radius (25L = 5*voxel, Rusu 2009)
//      - FGR distance threshold (2.5L = 0.5*voxel, Zhou 2016)
//      - ICP correspondence distance (~7.5L = 1.5*voxel, Open3D)
//
//  Computed once after target cloud setup; updated whenever the
//  target is rebuilt.
// =========================================================
float       g_referenceL        = 0.01f;  // safe default; overwritten in setupObjScene

// =========================================================
//  Apply scene-scale-relative camera/UI parameters
// ---------------------------------------------------------
//  Camera radius, pan/zoom sensitivities, and clamp ranges are
//  in mesh length units. Originally tuned at sceneDiag ≈ 7.36;
//  we re-scale them whenever sceneDiag changes (e.g. after
//  setupObjScene). Angular sensitivities (rotation, scaleSpeed)
//  are NOT touched because they are unitless ratios.
// =========================================================
constexpr float kRefSceneDiagCamera = 7.36f;  // matches RegRatios::kRefSceneDiag

inline void applySceneScaleToCamera() {
    const float r = g_sceneDiag / kRefSceneDiagCamera;

    // Camera radius (current view distance + initial value used on Reset)
    OrbitCam.gRadius          = 11.35f * r;
    OrbitCam.InitialRadius    = 11.35f * r;
    OrbitCam.minRadius        =  2.0f  * r;
    OrbitCam.maxRadius        = 80.0f  * r;

    // Pan & zoom step sizes (length per pixel / scroll tick)
    OrbitCam.LIGHT_MOUSE_SENSITIVITY = 0.01f * r;
    OrbitCam.ZOOM_SENSITIVITY        = -1.0  * r;

    // MOUSE_SENSITIVITY (rotation, rad/pixel) and SCALE_SPEED (multiplier)
    // are scale-invariant -- DO NOT touch.

    std::cout << "[applySceneScaleToCamera] r=" << r
              << "  gRadius=" << OrbitCam.gRadius
              << "  pan=" << OrbitCam.LIGHT_MOUSE_SENSITIVITY
              << "  zoom=" << OrbitCam.ZOOM_SENSITIVITY
              << "  clamp=[" << OrbitCam.minRadius << "," << OrbitCam.maxRadius << "]"
              << std::endl;
}

extern mCutMesh* liverMesh3D;
#include "RegistrationActions.h"
#include "PoseLibrary.h"   // RegistrationActions.h の後ろで include
#include "RimPairSampling.h"   // [Phase C] colored RIM pairs sampler
#include "SilComparePanel.h"   // G tab: silhouette method A/B/C compare (reuses runShiftE / runBipopCmaesV3RS / SilOverlay F9)
    // (inline computeUnifiedMetrics の宣言が必要なため)

// ----------------------------------------------------------------------
// [QUAD-SIL ACCEPT] Save an Alt+P (runShiftE) result to the Pose Library
// with the SAME accept criterion Ctrl+I uses, when in quadrant mode.
// Defined here (after PoseLibrary.h) because it calls poseSaveToLibrary
// and g_poseLibraryUseOccludedForAccept, both declared in PoseLibrary.h.
//
//   Legacy Alt+P (full mesh, Q:ALL or toggle OFF):
//     poseSaveToLibrary(IOU) — gate on full-mesh IoU2D, byte-identical
//     to the pre-feature behaviour.
//
//   Quadrant Alt+P:
//     A subset alignment necessarily lowers the FULL-mesh IoU (only the
//     selected quadrant matches; the rest drifts), so a full-mesh-IoU
//     gate would reject every good result. Ctrl+I avoids this by judging
//     on the OCCLUDED IoU. runShiftE already published g_lastSilOccludedIoU2D
//     via publishCtrlGStyleDiagnostics, so here we just flip the gate to
//     the occluded series for this one save (restored immediately after),
//     mirroring Ctrl+I exactly. EITHER (RMSE-or-IoU), same as Ctrl+I.
//
//   The toggle flip is local and reverted, so the user's Pose-Library
//   panel setting is untouched after the call.
// ----------------------------------------------------------------------
inline void saveSilAlignToLibrary() {
    const bool quadMode =
        g_shiftE_quadrantAware &&
        g_activeQuadrantMask != LiverLeftRightLabel::QUAD_ALL;
    if (!quadMode) {
        // Legacy path — unchanged.
        poseSaveToLibrary(SaveCriterion::IOU);
        return;
    }
    // Quadrant path — judge on occluded IoU like Ctrl+I.
    const bool prevOcc = g_poseLibraryUseOccludedForAccept;
    g_poseLibraryUseOccludedForAccept = true;
    poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
    g_poseLibraryUseOccludedForAccept = prevOcc;
}

// SimpleCameraPreview は CameraPreview.h に移動

static SimpleCameraPreview gCamera;
static AppContext             gApp;
static MaskPicker::Renderer   gMaskRenderer;
RegistrationImGuiManager gUI;          // PoseLibrary.h からも extern 参照されるため非static
static DebugPanel::State g_debugPanel;  // Ctrl+D で開閉する統合デバッグパネル
static UmeyamaController gUmeyama;
// 臓器6個 + board(6) + target(7)  ≥0.75="ON", 0.01~0.74="50%", <0.01="OFF"
//   起動デフォルト: liver と target だけ ON、それ以外は全 OFF
//     [0] liver   ON  (0.8)
//     [1] portal  OFF (0.0)
//     [2] vein    OFF (0.0)
//     [3] tumor   OFF (0.0)
//     [4] segment OFF (0.0)
//     [5] gb      OFF (0.0)
//     [6] board   OFF (0.0)
//     [7] target  ON  (0.8) — 点群表示
//
// static を外しているのは Umeyama 2画面モードの右ビューで board / target の
// 表示判定にこの値を参照するため (UmeyamaController.h が extern する)。
float g_meshAlpha[8] = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f};

// =========================================================
//  screenMesh の描画モード切替
// ---------------------------------------------------------
//  screenMesh (= masked OBJ, full-res target) は頂点が多く、
//  三角形描画では overdraw で重い。デフォルトで点群描画にし、
//  必要なら従来の三角形描画にも戻せるようにする。
//
//  g_screenMeshAsPoints  : true=点群 (デフォルト)、false=三角形
//  g_screenMeshPointSize : GL_POINTS のサイズ [px]、ImGui で調整可
//  g_screenMeshDensity   : 描画する頂点の割合 [%], 100 = 全頂点
//
//  static を外しているのは Umeyama 2画面モードの右ビューポートから
//  同じ点群描画パスを呼ぶため (UmeyamaController.h が extern 参照する)。
//  これがないと Umeyama 中は screenMesh が常に full-triangle で描画され、
//  数十万頂点のターゲットでフレームレートが大幅低下する。
// =========================================================
bool  g_screenMeshAsPoints  = true;
float g_screenMeshPointSize = 2.0f;
float g_screenMeshDensity   = 1.0f;
static SphereMesh g_sphereMarker;  // クラスタ・対応点描画用スフィア
static ARSave::State g_arSave;    // AR保存＋プレビュー状態
static ShaderProgram* g_pShader     = nullptr;  // AR保存用シェーダ参照
static ShaderProgram* g_pShaderCube = nullptr;
// Intrinsics source selector (enum in common/src/IntrinsicsSource.h).
// Default Preset = the only factory preset currently in the registry
// (Azure Kinect 1080p). Step 4 (autoSelect) overrides this at startup based on
// available files.
static IntrinsicsSource g_intrinsicsSource = IntrinsicsSource::Preset;
static std::string      g_currentPresetKey = "azure_kinect_1080p";
static CalibResult    g_calibResult;             // キャリブレーション結果
// Step 7: Settings タブの Calibration パラメータ (UI から編集される)
static std::string      g_chessboardFolder   = "../../../chessboard/";
static int              g_chessboardCols     = 9;
static int              g_chessboardRows     = 6;
static float            g_chessboardSquareMm = 22.0f;

// ---- Live Calibration session state (Take Picture tab) ----
//   Frames are captured into g_liveCalibFolder; onLiveCalibRun points
//   g_chessboardFolder at it and delegates to the existing onRunCalibration
//   (external calibration_tool), which writes intrinsics_calib.txt and flips
//   the active source to Calib on success.
static std::string              g_liveCalibFolder;       // <DEPTH_OUTPUT_PATH>chessboard_live_<ts>/
static std::vector<std::string> g_liveCalibFiles;        // full paths of saved JPEGs
static int                      g_liveCalibCounter = 0;  // next frame number
static bool                     g_liveCalibActive  = false;
static std::string              g_liveCalibLastMsg;      // latest UI feedback line

// Cross-platform "YYYYMMDD_HHMMSS" timestamp (mirrors the Shift+M snapshot code).
static std::string makeTimestampString() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tmLocal{};
#ifdef _WIN32
    localtime_s(&tmLocal, &t);
#else
    localtime_r(&t, &tmLocal);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmLocal);
    return std::string(buf);
}

// カメラ開始前の状態を保存
static struct {
    AppMode previousMode = AppMode::kEmpty;
    bool hadImage = false;
    std::string previousImagePath;
    int previousImageWidth = 0;
    int previousImageHeight = 0;
} g_cameraBackupState;
bool g_showCorrespondencePoints = false;

// =========================================================
//  Debug: AABB 可視化 (チャット 10)
// ---------------------------------------------------------
//  Initial Orientation の重心合わせを目視確認するため、
//  - target_full AABB の 8 コーナー + center を 赤  (固定)
//  - source subset AABB の 8 コーナー + center を 緑 (Apply 直後に保存)
//  を球マーカーで描画する。toggle で ON/OFF。
//  applyInitRotation() で post-transform AABB を g_dbgSourceBB に保存し、
//  drawScene の最後で sphere を描く。
// =========================================================
bool      g_showDebugBB = false;         // 既定で OFF。ImGui で toggle 可
glm::vec3 g_dbgSourceBB_min(0.0f);
glm::vec3 g_dbgSourceBB_max(0.0f);
glm::vec3 g_dbgSourceBB_center(0.0f);
bool      g_dbgSourceBB_valid = false;
bool g_showCyclicCorrespondence = false;  // Shift+B: Shift+P 対応点の可視化

// =========================================================
//  LiverRegion (Shift+R / Shift+T): 肝臓を anterior(前面)/
//  rim(ヘリ)/posterior(後面) の 3 領域に分けて可視化する。
//  Python 版 liver_raycast_4patients.py と
//  liver_anterior_rim_4patients.py の手法を C++ で再実装。
//
//  - Shift+R: 表示トグル (初回 ON 時に自動計算 ~数百 ms)
//  - Shift+T: 現在の g_rimTargetMm で再計算
//
//  ラベルは頂点 index に紐づくので、registration の transform
//  で動いても再計算不要 (球マーカーは mVertices[idx*3] で
//  現在位置を取得 → 自動追従)。
// =========================================================
LiverRegionLabel::Result g_liverRegion;
bool                     g_showLiverRegion = false;
float                    g_rimTargetMm     = 8.0f;   // Shift+T で再計算時の rim 厚さ (mm)
std::vector<int>         g_regionVizIdxAnt;          // 球マーカー描画用 subsample (anterior)
std::vector<int>         g_regionVizIdxRim;          //                            (rim)
std::vector<int>         g_regionVizIdxPost;         //                            (posterior)

// Shift+R 初回 ON 時、または Shift+T で呼ばれる。
// liverMesh3D の現在の頂点座標で計算する。
//
// rim 厚さは「物理 mm」で指定したいが、現在の liverMesh3D は
// prealignSourceToTarget で scale ≈ 0.005-0.007 に縮小されている。
// labelVertices() に「元 CT mesh の bbox diag (mm)」を渡せば、
// 関数内で mesh_units_per_mm = curDiag / origDiag(mm) を計算して
// 換算してくれる。startup 時に g_originalLiverDiagMm として保持済み。
inline void recomputeLiverRegion() {
    if (!liverMesh3D) {
        std::cerr << "[Region] liverMesh3D not loaded yet" << std::endl;
        return;
    }
    if (liverMesh3D->mVertices.empty() || liverMesh3D->mIndices.empty()) {
        std::cerr << "[Region] liverMesh3D has no geometry" << std::endl;
        return;
    }

    float orig_diag_mm = (g_hasOriginalDiags ? g_originalLiverDiagMm : 0.0f);
    std::cout << "[Region] calling labelVertices with target_rim_mm="
              << g_rimTargetMm << "  origDiag_mm=" << orig_diag_mm
              << std::endl;

    g_liverRegion = LiverRegionLabel::labelVertices(
        *liverMesh3D, g_rimTargetMm, /*smooth_iters=*/40, orig_diag_mm);
    if (!g_liverRegion.valid()) {
        std::cerr << "[Region] labeling failed" << std::endl;
        g_regionVizIdxAnt.clear();
        g_regionVizIdxRim.clear();
        g_regionVizIdxPost.clear();
        return;
    }
    g_regionVizIdxAnt  = LiverRegionLabel::sampleVertexIndices(
        g_liverRegion.labels, LiverRegionLabel::ANTERIOR_CORE, 1500);
    g_regionVizIdxRim  = LiverRegionLabel::sampleVertexIndices(
        g_liverRegion.labels, LiverRegionLabel::RIM,            800);
    g_regionVizIdxPost = LiverRegionLabel::sampleVertexIndices(
        g_liverRegion.labels, LiverRegionLabel::POSTERIOR,     1200);
    std::cout << "[Region] viz subsample: ant=" << g_regionVizIdxAnt.size()
              << " rim=" << g_regionVizIdxRim.size()
              << " post=" << g_regionVizIdxPost.size() << std::endl;
}

// =========================================================
//  LiverLeftRight (Y / Shift+Y): 肝臓を pure-right(右葉) /
//  boundary(鎌状間膜) / pure-left(左葉) の 3 領域に分けて可視化。
//  Python 版 liver_leftright_4patients.py の手法を C++ で再実装。
//
//  Shift+R/Shift+T (anterior/rim/posterior) と完全パラレルの仕組み:
//  - Y       : 表示トグル (初回 ON 時に自動計算 ~数百 ms)
//  - Shift+Y : 現在の g_lrPureFrac, g_lrFullFrac で再計算
//
//  ラベルは頂点 index に紐づくので、registration の transform で
//  動いても再計算不要 (球マーカーは mVertices[idx*3] で現在位置を
//  取得 → 自動追従)。
//
//  スケール変換は不要 (閾値が無次元 mass 比なのでスケール不変)。
// =========================================================
LiverLeftRightLabel::Result g_liverLR;
bool                        g_showLiverLR  = false;
float                       g_lrPureFrac   = 0.60f;   // Python デフォルト
float                       g_lrFullFrac   = 0.70f;   // Python デフォルト
bool                        g_lrFlipManual = false;   // Python の FLIP_OVERRIDE 相当
std::vector<int>            g_lrVizIdxR;              // 球マーカー描画用 subsample (pure right)
std::vector<int>            g_lrVizIdxBoundary;       //                              (boundary)
std::vector<int>            g_lrVizIdxL;              //                              (pure left)

// Y キー初回 ON または Shift+Y で呼ばれる。
inline void recomputeLiverLR() {
    if (!liverMesh3D) {
        std::cerr << "[LR] liverMesh3D not loaded yet" << std::endl;
        return;
    }
    if (liverMesh3D->mVertices.empty() || liverMesh3D->mIndices.empty()) {
        std::cerr << "[LR] liverMesh3D has no geometry" << std::endl;
        return;
    }

    std::cout << "[LR] calling labelVertices with right_pure_fraction="
              << g_lrPureFrac << "  right_full_fraction=" << g_lrFullFrac
              << "  flip_manual=" << (g_lrFlipManual ? "true" : "false")
              << std::endl;

    g_liverLR = LiverLeftRightLabel::labelVertices(
        *liverMesh3D, g_lrPureFrac, g_lrFullFrac, g_lrFlipManual);

    if (!g_liverLR.valid()) {
        std::cerr << "[LR] labeling failed" << std::endl;
        g_lrVizIdxR.clear();
        g_lrVizIdxBoundary.clear();
        g_lrVizIdxL.clear();
        return;
    }
    g_lrVizIdxR        = LiverLeftRightLabel::sampleVertexIndices(
        g_liverLR.labels, LiverLeftRightLabel::PURE_RIGHT, 1500);
    g_lrVizIdxBoundary = LiverLeftRightLabel::sampleVertexIndices(
        g_liverLR.labels, LiverLeftRightLabel::BOUNDARY,    600);
    g_lrVizIdxL        = LiverLeftRightLabel::sampleVertexIndices(
        g_liverLR.labels, LiverLeftRightLabel::PURE_LEFT,  1200);
    std::cout << "[LR] viz subsample: R=" << g_lrVizIdxR.size()
              << " boundary=" << g_lrVizIdxBoundary.size()
              << " L=" << g_lrVizIdxL.size() << std::endl;
}

// =========================================================
//  LiverCranioCaudal (Shift+H): 肝臓を cranial (頭側) / caudal (足側)
//  の 2 領域に分けて可視化。Python v7 アルゴリズム (RIM 上の
//  area-weighted dihedral roughness) を C++ で再実装したもの。
//
//  Shift+R (anterior/rim/posterior) と Y (pure_R/boundary/pure_L) の
//  ラベルに依存する (RIM mask + lr_axis_idx を借りる)。未計算なら
//  Shift+H 押下時に自動的に計算する (Quad と同じ流儀)。
//
//  キー操作:
//    Shift+H : 可視化トグル (初回 ON 時に自動計算)
//              黄=CRANIAL(頭側), 青=CAUDAL(足側)
//              confidence < 5% で std::cout に [WEAK] 警告を出力。
//
//  Phase 1 段階では registration には未統合 (CC ラベル参照箇所なし)。
//  Phase 2 (Initial Orientation 幾何ベース化) で d_cc を使う予定。
//
//  ラベルは頂点 index に紐づくので、registration の transform で
//  動いても再計算不要 (球マーカーは mVertices[idx*3] で現在位置を
//  取得 → 自動追従)。
// =========================================================
LiverCranioCaudalLabel::Result g_liverCC;
bool                           g_showLiverCC  = false;
bool                           g_ccFlipManual = false;   // Python の FLIP_OVERRIDE 相当 (Phase 1 では UI 無し)
std::vector<int>               g_ccVizIdxCranial;        // 球マーカー描画用 subsample (黄)
std::vector<int>               g_ccVizIdxCaudal;         //                              (青)

// Shift+H 初回 ON または明示再計算で呼ばれる。
inline void recomputeLiverCC() {
    if (!liverMesh3D) {
        std::cerr << "[CC] liverMesh3D not loaded yet" << std::endl;
        return;
    }
    if (liverMesh3D->mVertices.empty() || liverMesh3D->mIndices.empty()) {
        std::cerr << "[CC] liverMesh3D has no geometry" << std::endl;
        return;
    }

    // 依存: Shift+R / Y のラベルが未計算なら auto-trigger (Quad と同じ流儀)
    if (!g_liverRegion.valid()) {
        std::cout << "[CC] LiverRegion (Shift+R) not yet computed, auto-running..." << std::endl;
        recomputeLiverRegion();
    }
    if (!g_liverLR.valid()) {
        std::cout << "[CC] LiverLR (Y) not yet computed, auto-running..." << std::endl;
        recomputeLiverLR();
    }
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[CC] Cannot compute CC labels: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << "  LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << std::endl;
        g_ccVizIdxCranial.clear();
        g_ccVizIdxCaudal.clear();
        return;
    }

    std::cout << "[CC] calling labelVertices  flip_manual="
              << (g_ccFlipManual ? "true" : "false") << std::endl;

    g_liverCC = LiverCranioCaudalLabel::labelVertices(
        *liverMesh3D, g_liverRegion, g_liverLR, g_ccFlipManual);

    if (!g_liverCC.valid()) {
        std::cerr << "[CC] labeling failed" << std::endl;
        g_ccVizIdxCranial.clear();
        g_ccVizIdxCaudal.clear();
        return;
    }

    g_ccVizIdxCranial = LiverCranioCaudalLabel::sampleVertexIndices(
        g_liverCC.labels, LiverCranioCaudalLabel::CRANIAL, 1500);
    g_ccVizIdxCaudal  = LiverCranioCaudalLabel::sampleVertexIndices(
        g_liverCC.labels, LiverCranioCaudalLabel::CAUDAL,  1500);
    std::cout << "[CC] viz subsample: cranial=" << g_ccVizIdxCranial.size()
              << " caudal=" << g_ccVizIdxCaudal.size() << std::endl;
}

// =========================================================
//  LiverQuad (H): 肝臓を4象限に分けて可視化。
//  Shift+R (anterior/rim/posterior) と Y (pure_R/boundary/pure_L)
//  の結果を AND 合成して4象限ラベルを作る。
//
//  rim は前後どちらにも、boundary は左右どちらにも所属させる
//  「重複所属」方式 (案D):
//    ant_right = (anterior OR rim) ∩ (pure_right OR boundary)   緑
//    ant_left  = (anterior OR rim) ∩ (pure_left  OR boundary)   紫
//    pos_right = (posterior OR rim) ∩ (pure_right OR boundary)  青
//    pos_left  = (posterior OR rim) ∩ (pure_left  OR boundary)  橙
//
//  例: rim ∩ pure_right の頂点 → ant_right と pos_right の両方に描画
//      anterior ∩ boundary の頂点 → ant_right と ant_left の両方に描画
//      rim ∩ boundary の頂点 → 4象限全部に描画
//
//  キー操作:
//    H : 4象限可視化トグル (初回 ON 時に自動計算)
//        Shift+R / Y のラベルが未計算なら自動的に計算する。
//
//  ラベルは頂点 index に紐づくので、registration の transform で
//  動いても再計算不要 (球マーカーは mVertices[idx*3] で現在位置を
//  取得 → 自動追従)。
// =========================================================
bool             g_showLiverQuad = false;
std::vector<int> g_quadVizIdxAR;   // ant_right (緑)
std::vector<int> g_quadVizIdxAL;   // ant_left  (紫)
std::vector<int> g_quadVizIdxPR;   // pos_right (青)
std::vector<int> g_quadVizIdxPL;   // pos_left  (橙)

// [bugfix] Clear the liver-label visualization caches — the subsample
// vertex-index arrays for the region / LR / CC / 4-quadrant marker
// overlays. These hold indices into the CURRENT liver mesh; when the
// mesh is swapped for one with a different vertex count (organ drop /
// scene reload), the stale indices draw markers at wrong positions or
// out of range. The label Result structs are reset alongside this; both
// repopulate on the next recompute*. Call wherever liverMesh3D topology
// may change.
inline void clearLiverLabelVizCaches() {
    g_regionVizIdxAnt.clear(); g_regionVizIdxRim.clear();  g_regionVizIdxPost.clear();
    g_lrVizIdxR.clear();       g_lrVizIdxBoundary.clear(); g_lrVizIdxL.clear();
    g_ccVizIdxCranial.clear(); g_ccVizIdxCaudal.clear();
    g_quadVizIdxAR.clear();    g_quadVizIdxAL.clear();
    g_quadVizIdxPR.clear();    g_quadVizIdxPL.clear();
}

// ----------------------------------------------------------------------
//  Ctrl+G (V3-R) 用の 4象限選択ビットマスク
//  ---------------------------------------------------------------------
//  ImGui 2×2 grid checkbox (Ctrl+G Quadrant Selector パネル) で操作。
//  デフォルト QUAD_ALL = 0x0F (全象限選択) は V3 と同じ subset =全頂点 を
//  意味するので、起動直後に Ctrl+G を押した場合の挙動は (S4 完了時点で)
//  Shift+G と byte-identical になる検証ハンドルになる。
//
//  本変数は H キー可視化 (g_showLiverQuad / g_quadVizIdx*) と完全独立。
//  Ctrl+G 実行時にのみ参照される。
// ----------------------------------------------------------------------
uint8_t g_activeQuadrantMask = LiverLeftRightLabel::QUAD_ALL;  // 0x0F

// ----------------------------------------------------------------------
// [QUAD-SIL] Alt+P dispatch boilerplate — mirrors the Ctrl+I preamble.
//   Shared by all three Alt+P entry points (GLFW key branch, G-tab
//   button, sidebar onSilhouetteAlign lambda) so the guard logic can't
//   drift between them.
//   - Always logs the active quadrant mask (parity with Ctrl+I logging).
//   - When the quadrant filter will actually engage
//     (g_shiftE_quadrantAware && mask != QUAD_ALL):
//       * auto-computes region/LR labels on demand (no HemiAuto needed),
//       * aborts (returns false) when labels fail or the subset is empty
//         — same hard-abort semantics as the Ctrl+I dispatch.
//   - When the filter is bypassed (toggle OFF or QUAD_ALL), returns true
//     without touching labels: legacy Alt+P never required them, and we
//     keep that robustness.
// ----------------------------------------------------------------------
inline bool prepareQuadrantForSilAlign(const char* tag) {
    const auto maskStr =
        LiverLeftRightLabel::quadrantMaskString(g_activeQuadrantMask);
    std::cout << "[" << tag << "] quadrant_mask = " << maskStr
              << "  (0x" << std::hex << (unsigned)g_activeQuadrantMask
              << std::dec << ")"
              << (g_shiftE_quadrantAware ? "" : "  [quad-aware OFF]")
              << std::endl;

    const bool willEngage =
        g_shiftE_quadrantAware &&
        g_activeQuadrantMask != LiverLeftRightLabel::QUAD_ALL;
    if (!willEngage) return true;

    if (!g_liverRegion.valid()) recomputeLiverRegion();
    if (!g_liverLR.valid())     recomputeLiverLR();
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[" << tag << "] ERROR: label computation failed "
                     "(is the liver mesh loaded?)." << std::endl;
        return false;
    }
    const auto subset = LiverLeftRightLabel::makeQuadrantSubsetIdx(
        g_liverRegion.labels, g_liverLR.labels, g_activeQuadrantMask);
    if (subset.empty()) {
        std::cerr << "[" << tag << "] ERROR: quadrant subset is empty."
                  << std::endl;
        return false;
    }
    return true;
}
// NOTE: saveSilAlignToLibrary() is defined AFTER the PoseLibrary.h include
// (it calls poseSaveToLibrary / g_poseLibraryUseOccludedForAccept, both from
// PoseLibrary.h which is included further down). See its definition there.

// H キー初回 ON で呼ばれる。Shift+R / Y のラベルを利用して
// 4象限の subsample 配列を構築する。
inline void recomputeLiverQuad() {
    if (!liverMesh3D) {
        std::cerr << "[Quad] liverMesh3D not loaded yet" << std::endl;
        return;
    }
    // Shift+R / Y のラベルが未計算なら自動的に計算
    if (!g_liverRegion.valid()) {
        std::cout << "[Quad] LiverRegion (Shift+R) not yet computed, auto-running..." << std::endl;
        recomputeLiverRegion();
    }
    if (!g_liverLR.valid()) {
        std::cout << "[Quad] LiverLR (Y) not yet computed, auto-running..." << std::endl;
        recomputeLiverLR();
    }
    if (!g_liverRegion.valid() || !g_liverLR.valid()) {
        std::cerr << "[Quad] Cannot build 4-quadrant labels: "
                  << "Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                  << "  LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << std::endl;
        g_quadVizIdxAR.clear();
        g_quadVizIdxAL.clear();
        g_quadVizIdxPR.clear();
        g_quadVizIdxPL.clear();
        return;
    }

    const int nV = (int)liverMesh3D->mVertices.size() / 3;
    if ((int)g_liverRegion.labels.size() != nV ||
        (int)g_liverLR.labels.size()     != nV) {
        std::cerr << "[Quad] label size mismatch: "
                  << "region=" << g_liverRegion.labels.size()
                  << "  LR=" << g_liverLR.labels.size()
                  << "  nV=" << nV
                  << " (mesh changed? recomputing both)" << std::endl;
        recomputeLiverRegion();
        recomputeLiverLR();
        if ((int)g_liverRegion.labels.size() != nV ||
            (int)g_liverLR.labels.size()     != nV) {
            std::cerr << "[Quad] still mismatched, abort" << std::endl;
            return;
        }
    }

    // 4象限の所属判定 (重複可)
    //   AP: anterior(0) / posterior(1) / rim(2)
    //   LR: pure_R(0)   / pure_L(1)    / boundary(2)
    std::vector<uint8_t> in_AR(nV, 0), in_AL(nV, 0),
        in_PR(nV, 0), in_PL(nV, 0);
    for (int i = 0; i < nV; i++) {
        uint8_t ap = g_liverRegion.labels[i];
        uint8_t lr = g_liverLR.labels[i];
        bool is_ant = (ap == LiverRegionLabel::ANTERIOR_CORE) ||
                      (ap == LiverRegionLabel::RIM);
        bool is_pos = (ap == LiverRegionLabel::POSTERIOR) ||
                      (ap == LiverRegionLabel::RIM);
        bool is_R   = (lr == LiverLeftRightLabel::PURE_RIGHT) ||
                    (lr == LiverLeftRightLabel::BOUNDARY);
        bool is_L   = (lr == LiverLeftRightLabel::PURE_LEFT)  ||
                    (lr == LiverLeftRightLabel::BOUNDARY);
        if (is_ant && is_R) in_AR[i] = 1;
        if (is_ant && is_L) in_AL[i] = 1;
        if (is_pos && is_R) in_PR[i] = 1;
        if (is_pos && is_L) in_PL[i] = 1;
    }

    // 集計
    int n_AR = 0, n_AL = 0, n_PR = 0, n_PL = 0;
    for (int i = 0; i < nV; i++) {
        if (in_AR[i]) n_AR++;
        if (in_AL[i]) n_AL++;
        if (in_PR[i]) n_PR++;
        if (in_PL[i]) n_PL++;
    }
    std::cout << "[Quad] 4-quadrant membership counts (with overlap):" << std::endl;
    std::cout << "[Quad]   ant_right  (緑) : " << n_AR << " verts" << std::endl;
    std::cout << "[Quad]   ant_left   (紫) : " << n_AL << " verts" << std::endl;
    std::cout << "[Quad]   pos_right  (青) : " << n_PR << " verts" << std::endl;
    std::cout << "[Quad]   pos_left   (橙) : " << n_PL << " verts" << std::endl;
    std::cout << "[Quad]   sum=" << (n_AR + n_AL + n_PR + n_PL)
              << "  (>= " << nV << " due to rim/boundary overlap)" << std::endl;

    // subsample (各象限 1000 点)
    auto sample = [&](const std::vector<uint8_t>& mask, int max_points) {
        std::vector<int> all;
        all.reserve(mask.size() / 3 + 1);
        for (int i = 0; i < (int)mask.size(); i++) {
            if (mask[i]) all.push_back(i);
        }
        if ((int)all.size() <= max_points) return all;
        std::vector<int> out;
        out.reserve(max_points);
        double step = double(all.size()) / double(max_points);
        for (int k = 0; k < max_points; k++) {
            int idx = (int)std::floor((k + 0.5) * step);
            if (idx >= (int)all.size()) idx = (int)all.size() - 1;
            out.push_back(all[idx]);
        }
        return out;
    };
    g_quadVizIdxAR = sample(in_AR, 1000);
    g_quadVizIdxAL = sample(in_AL, 1000);
    g_quadVizIdxPR = sample(in_PR, 1000);
    g_quadVizIdxPL = sample(in_PL, 1000);
    std::cout << "[Quad] viz subsample: AR=" << g_quadVizIdxAR.size()
              << "  AL=" << g_quadVizIdxAL.size()
              << "  PR=" << g_quadVizIdxPR.size()
              << "  PL=" << g_quadVizIdxPL.size() << std::endl;
}

// 前方宣言
static bool setupObjScene();
static bool runDepthAndUpdateScene(AppContext& ctx);
static void loadIntrinsicsFromCurrentSource();   // defined near main(); used by sidecar import
// Forward decl: kind selects which mask the preview pops up for.
//   MaskKind::Liver       -> uses ctx.maskPoints, writes segmentation_mask.png
//   MaskKind::Instrument  -> uses ctx.instrumentMaskPoints, writes
//                            instrument_segmentation_mask.png
// Called by the Segment 1 / Instrument buttons via the onSegment1 /
// onSegment2 lambdas.
static bool runSegmentOnly(AppContext& ctx, MaskKind kind);
static void syncUIState();
static void setupUICallbacks();
static void showFPS(GLFWwindow* window);
static void snapshotInitialPose();
static void restoreInitialPose();
static void applyInitRotation(bool startNewSession, bool liver_only = false);   // Phase 2 拡張: preset + position 適用。liver_only=true で AutoQCR loop 用の liver-only モード (non-liver organ は触らない)。
static void runAutoQuadCyclicRansac(bool lock_scale = false);   // Alt+Ctrl+P / AutoQCR ボタン: 9 ORIENT preset 自動探索 + ベスト 1 採用 (lock_scale=true で 6-DoF rigid)
static void drawCtrlGRimRaycastControls();                       // Advanced セクション用 helper: Ctrl+G / Ctrl+Shift+G の RIM-weighted + RIM silhouette penalty コントロールを描画
static void computeTargetSubsetAabbs();                // Phase 2 拡張: target cloud の AABB を 3 通り (full/+X/-X) 計算
static void computeSourceLiverSubsetAabbs();           // Phase 2 拡張 v2: source liver の AABB を 3 通り計算

// グローバル変数
// PoseLibrary.h からも extern 参照されるため static を外す
std::vector<std::vector<GLfloat>> g_initOrganVertices;
std::vector<std::vector<GLfloat>> g_initOrganNormals;

// Registration用3Dビューポート（キャリブレーション解像度に合わせる）
static struct {
    int x = 0, y = 0, w = 1280, h = 720;
} g_3dViewport;

// キャリブレーション解像度のアスペクト比を維持したビューポートを計算
static void compute3DViewport(int windowW, int windowH, int sidebarW) {
    int availW = windowW - sidebarW;
    int availH = windowH;
    if (availW <= 0 || availH <= 0) {
        g_3dViewport = {0, 0, windowW, windowH};
        return;
    }

    int cw = OrbitCam.calibWidth;
    int ch = OrbitCam.calibHeight;
    if (cw <= 0 || ch <= 0) {
        // キャリブレーション未設定時はサイドバー分だけ除外
        g_3dViewport = {0, 0, availW, availH};
        return;
    }

    float calibAspect = (float)cw / (float)ch;
    float availAspect = (float)availW / (float)availH;

    if (availAspect > calibAspect) {
        // 横余裕あり → 高さに合わせて中央配置
        int vpH = availH;
        int vpW = (int)(vpH * calibAspect);
        g_3dViewport = {(availW - vpW) / 2, 0, vpW, vpH};
    } else {
        // 縦余裕あり → 幅に合わせて中央配置
        int vpW = availW;
        int vpH = (int)(vpW / calibAspect);
        g_3dViewport = {0, (availH - vpH) / 2, vpW, vpH};
    }
}

// カメラから深度推定を実行する関数
static bool runCameraDepthEstimation() {
    if (!gCamera.active) {
        std::cerr << "[DepthEstimation] Camera is not active" << std::endl;
        return false;
    }

    // カメラフレームをJPEGとして保存
    std::string jpegFile = gCamera.saveForDepthEstimation();
    if (jpegFile.empty()) {
        std::cerr << "[DepthEstimation] Failed to save camera frame" << std::endl;
        return false;
    }

    std::cout << "[DepthEstimation] Saved camera frame to: " << jpegFile << std::endl;

    // AppContextを設定（画像サイズも設定）
    gApp.image.path = jpegFile;
    gApp.image.loaded = true;
    gApp.image.width = gCamera.width;
    gApp.image.height = gCamera.height;

    // 深度推定を実行
    if (!runDepthAndUpdateScene(gApp)) {
        std::cerr << "[DepthEstimation] Failed to run depth estimation" << std::endl;
        return false;
    }

    std::cout << "[DepthEstimation] Successfully generated depth mesh from camera" << std::endl;

    // 深度生成が成功したら、カメラを停止する
    // Registrationモードに移行するため、カメラは不要になる
    gCamera.stop();
    std::cout << "[DepthEstimation] Camera stopped after successful depth generation" << std::endl;

    return true;
}

// gGridHeight(), getOrganList() は RegistrationActions.h に移動

static glm::vec3 liverCenter() {
    glm::vec3 c(0.0f);
    size_t n = liverMesh3D->mVertices.size() / 3;
    if (n == 0) return c;
    for (size_t i = 0; i < liverMesh3D->mVertices.size(); i += 3) {
        c.x += liverMesh3D->mVertices[i];
        c.y += liverMesh3D->mVertices[i+1];
        c.z += liverMesh3D->mVertices[i+2];
    }
    return c / (float)n;
}

static bool rayTri(const glm::vec3& O, const glm::vec3& D,
                   const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                   float& t) {
    const float EPS = 1e-7f;
    glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
    glm::vec3 h = glm::cross(D, e2);
    float a = glm::dot(e1, h);
    if (std::abs(a) < EPS) return false;
    float f = 1.0f / a;
    glm::vec3 s = O - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(D, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * glm::dot(e2, q);
    return t > EPS;
}

static bool hitTestMesh(float sx, float sy, mCutMesh* mesh, glm::vec3& outPos) {
    if (!mesh || mesh->mIndices.empty()) return false;
    // Registration時は3Dビューポート座標を使用
    float vpX = (float)g_3dViewport.x;
    float vpY = (float)g_3dViewport.y;
    float vpW = (float)g_3dViewport.w;
    float vpH = (float)g_3dViewport.h;
    float ndcX = 2.0f * (sx - vpX) / vpW  - 1.0f;
    float ndcY = 1.0f - 2.0f * (sy - vpY) / vpH;
    glm::mat4 invVP = glm::inverse(projection * view);
    glm::vec4 nr = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 fr = invVP * glm::vec4(ndcX, ndcY,  1.0f, 1.0f);
    nr /= nr.w; fr /= fr.w;
    glm::vec3 O = glm::vec3(nr);
    glm::vec3 D = glm::normalize(glm::vec3(fr - nr));

    const auto& V = mesh->mVertices;
    const auto& I = mesh->mIndices;
    float bestT = std::numeric_limits<float>::max();
    bool  hit   = false;
    for (size_t i = 0; i + 2 < I.size(); i += 3) {
        GLuint a = I[i], b = I[i+1], c = I[i+2];
        glm::vec3 v0(V[a*3], V[a*3+1], V[a*3+2]);
        glm::vec3 v1(V[b*3], V[b*3+1], V[b*3+2]);
        glm::vec3 v2(V[c*3], V[c*3+1], V[c*3+2]);
        float t;
        if (rayTri(O, D, v0, v1, v2, t) && t < bestT) {
            bestT  = t;
            outPos = O + D * t;
            hit    = true;
        }
    }
    return hit;
}

#include "InteractionHelpers.h"

// computeUnifiedMetrics(), runHemiAuto(), runBipopCmaes(), runShiftE()
// は RegistrationActions.h に移動

static glm::vec3 g_lastOrganOffset(0.0f);

// =========================================================
//  ScreenMeshPointCache - 点群描画用のシャッフル済みインデックス
// ---------------------------------------------------------
//  事前に頂点インデックス [0..N-1] をランダムシャッフルして EBO に
//  上げておき、density% に応じて先頭から K 個だけ glDrawElements する。
//  シャッフルは 1 回限り (mesh が変わるまで) なので、density スライダを
//  動かしても点はチラつかない (常に同じシャッフル順の prefix を描く)。
//
//  VAO は専用に作る理由: mesh->VAO に GL_ELEMENT_ARRAY_BUFFER を bind
//  してしまうと VAO 状態が破壊され、次フレームの三角形描画が壊れる。
//  VBO/NBO は mesh のものを共有 (アトリビュート設定だけ別 VAO に貼る)。
//
//  cachedVBO で mesh の VBO ID を覚え、setUp() で VBO が再生成された
//  ときに自動的にリビルドする (ID の変化で検出)。
// =========================================================
//  ScreenMeshPointCache の定義は common/src/ScreenMeshPoints.h に移動
//  (REG と DEFORM で共有)。
static ScreenMeshPointCache g_screenMeshPC;

// =========================================================
//  drawScreenMeshAsPoints
// ---------------------------------------------------------
//  screenMesh を点群として描画する。
//  - g_screenMeshDensity [%] で描画頂点数を制御 (ランダムサンプリング)
//  - 三角形ラスタライズの overdraw が消えるので頂点数が多くても軽量
//
//  シェーダは shaderProgram (basic) を流用。useTexture=false にして
//  単色 (vertColor) で描く。lighting uniforms も一応セットしておく。
//
//  static を外している理由は UmeyamaController::render が右画面で
//  screenMesh 描画にこの関数を呼ぶため。
// =========================================================
void drawScreenMeshAsPoints(
    mCutMesh* mesh,
    ShaderProgram& shader,
    const glm::mat4& model,
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::vec3& camPos,
    const glm::vec4& color,
    float pointSize)
{
    // 共有の ScreenMeshPointCache に委譲。density は REG の UI 値を渡す。
    g_screenMeshPC.draw(mesh, shader, model, view, projection, camPos,
                        color, pointSize, g_screenMeshDensity);
}

static void rebuildOBJWithCurrentThreshold() {
    if (!screenMesh) return;
    std::cout << "\n=== Rebuilding OBJ with cosThreshold="
              << g_silhouetteCosThreshold << " ===" << std::endl;

    delete screenMesh;
    screenMesh = new mCutMesh(mCutMesh().loadMeshFromFile(g_objSourcePath.c_str()));
    gApp.screen = screenMesh;

    Reg3DCustom::clearCachedTargetCloud();

    auto stats = Reg3DCustom::cleanupOBJMesh(*screenMesh,
                                             g_silhouetteCosThreshold,
                                             0.0f);
    (void)stats;

    auto targetCloud = Reg3DCustom::setupOBJTarget(
        *screenMesh, g_intrinsics, Reg3DCustom::OBJ_Y_SIGN_OPENGL);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[Rebuild] target cloud empty, aborting" << std::endl;
        return;
    }

    Reg3DCustom::mirrorMeshAndCloudX(*screenMesh, *targetCloud);

    // Undo previous prealignment (similarity), if any. This restores organs
    // to their model-space pose so we can prealign again to the new target.
    std::vector<mCutMesh*> organs = { liverMesh3D, portalMesh3D, veinMesh3D,
                                      tumorMesh3D, segmentMesh3D, gbMesh3D };
    if (g_hasLastOrganTransform) {
        glm::mat4 invT = glm::inverse(g_lastOrganTransform);
        Reg3DCustom::applyTransformToMeshes(organs, invT);
    } else if (glm::dot(g_lastOrganOffset, g_lastOrganOffset) > 0.0f) {
        // Legacy translation-only undo (older state)
        for (auto* m : organs) {
            if (m) {
                for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
                    m->mVertices[i    ] -= g_lastOrganOffset.x;
                    m->mVertices[i + 1] -= g_lastOrganOffset.y;
                    m->mVertices[i + 2] -= g_lastOrganOffset.z;
                }
            }
        }
    }

    glm::vec3 organCenter = Reg3DCustom::computeMeshCenter(*liverMesh3D);
    glm::vec3 objCenter   = Reg3DCustom::computeMeshCenter(*screenMesh);
    g_lastOrganOffset     = objCenter - organCenter;  // legacy

    // NEW: similarity prealignment again
    g_lastOrganTransform     = Reg3DCustom::prealignSourceToTarget(
        organs, *screenMesh);
    g_hasLastOrganTransform  = true;

    g_sceneDiag = std::max(Reg3DCustom::computeMeshDiag(*screenMesh), 1e-3f);
    std::cout << "[SceneDiag/rebuild] " << g_sceneDiag << std::endl;

    // Recompute L for the new target cloud
    if (targetCloud && !targetCloud->empty()) {
        float L_new = Reg3DCustom::computeMedianNNDistance(*targetCloud);
        g_referenceL = std::max(L_new, 1e-6f);
        std::cout << "[L/rebuild] " << g_referenceL << std::endl;
    }

    applySceneScaleToCamera();

    setUp(*screenMesh);
    setUp(*liverMesh3D);
    setUp(*portalMesh3D);
    setUp(*veinMesh3D);
    setUp(*tumorMesh3D);
    setUp(*segmentMesh3D);
    setUp(*gbMesh3D);

    Reg3DCustom::printMeshBBox(*screenMesh,  "OBJ rebuilt");
    Reg3DCustom::printMeshBBox(*liverMesh3D, "liver rebuilt");
    std::cout << "=== Rebuild done ===" << std::endl;
}

// =============================================================================
// [CT-handoff] REG -> DEFORM -> REG round-trip plumbing (ideas (1) and (5)).
//   (1) Recover T (CT model/*.obj  ->  registered reg_*.obj) EMPIRICALLY via
//       Umeyama on corresponding vertices, instead of composing the transform
//       chain. organ meshes are never remeshed during registration (cleanupOBJMesh
//       runs only on screenMesh), so re-loading model/liver.obj from disk gives a
//       1:1 vertex correspondence with the current registered liverMesh3D. Umeyama
//       then yields the exact similarity, det(R) detects any reflection for free,
//       and the residual tells us whether every stage really was a similarity.
//   (5) Zero-deform CT round-trip: write T^-1(reg_<organ>.obj) to ct_roundtrip/
//       (== what DEFORM would output with zero deformation). Dropping that folder
//       back into REG (NORMAL drop -> prealign + register#2) lets us verify the
//       coordinate plumbing in one app, isolated from any physics.
// =============================================================================
namespace CtHandoff {

// When true, exportRegisteredObjs() also writes ct_roundtrip/ (the zero-deform
// stand-in). Flip to false to disable the test scaffold (reg_transform.txt is
// always written; it is part of the real handoff).
inline bool g_writeZeroDeformRoundtrip = true;

// Umeyama (1991) similarity. Solves q ~= s*R*p + t for corresponding flat xyz
// arrays src(p) and dst(q) of equal length. Returns column-major glm::mat4 and
// reports scale, det(R) (should be +1, no reflection), RMSE and max residual in
// dst units, and the pair count.
inline glm::mat4 umeyamaSimilarity(const std::vector<GLfloat>& src,
                                   const std::vector<GLfloat>& dst,
                                   double& outScale, double& outDetR,
                                   double& outRmse,  double& outMaxResid,
                                   int& outN) {
    outScale = 1.0; outDetR = 1.0; outRmse = -1.0; outMaxResid = -1.0; outN = 0;
    if (src.size() != dst.size() || src.size() < 9) return glm::mat4(1.0f);
    const int N = static_cast<int>(src.size() / 3);
    outN = N;

    Eigen::Vector3d mp = Eigen::Vector3d::Zero(), mq = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; ++i) {
        mp += Eigen::Vector3d(src[3*i], src[3*i+1], src[3*i+2]);
        mq += Eigen::Vector3d(dst[3*i], dst[3*i+1], dst[3*i+2]);
    }
    mp /= N; mq /= N;

    Eigen::Matrix3d Sigma = Eigen::Matrix3d::Zero();
    double varP = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d pc(src[3*i]-mp.x(), src[3*i+1]-mp.y(), src[3*i+2]-mp.z());
        Eigen::Vector3d qc(dst[3*i]-mq.x(), dst[3*i+1]-mq.y(), dst[3*i+2]-mq.z());
        Sigma += qc * pc.transpose();
        varP  += pc.squaredNorm();
    }
    Sigma /= N; varP /= N;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Vector3d D = svd.singularValues();
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if (U.determinant() * V.determinant() < 0.0) S(2, 2) = -1.0;
    Eigen::Matrix3d R = U * S * V.transpose();
    const double trDS = D(0)*S(0,0) + D(1)*S(1,1) + D(2)*S(2,2);
    const double s = (varP > 1e-20) ? trDS / varP : 1.0;
    Eigen::Vector3d t = mq - s * R * mp;

    outScale = s; outDetR = R.determinant();
    double se = 0.0, mx = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p(src[3*i], src[3*i+1], src[3*i+2]);
        Eigen::Vector3d q(dst[3*i], dst[3*i+1], dst[3*i+2]);
        const double e = (s * R * p + t - q).norm();
        se += e * e; mx = std::max(mx, e);
    }
    outRmse = std::sqrt(se / std::max(1, N)); outMaxResid = mx;

    glm::mat4 T(1.0f);                       // glm is column-major: T[col][row]
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            T[c][r] = static_cast<float>(s * R(r, c));
    T[3][0] = static_cast<float>(t.x());
    T[3][1] = static_cast<float>(t.y());
    T[3][2] = static_cast<float>(t.z());
    return T;
}

// Apply a 4x4 to a flat xyz vertex array in place.
inline void applyMat4(std::vector<GLfloat>& verts, const glm::mat4& M) {
    for (size_t i = 0; i + 2 < verts.size(); i += 3) {
        const glm::vec4 p = M * glm::vec4(verts[i], verts[i+1], verts[i+2], 1.0f);
        verts[i] = p.x; verts[i+1] = p.y; verts[i+2] = p.z;
    }
}

} // namespace CtHandoff

// [key-reorg Phase 12] STL/OBJ export, moved verbatim out of the old
// GLFW_KEY_M switch case. Declared in StlExport.h; called by the sidebar
// Export buttons (onExportStl / onExportStlFlipped).
void StlExport::exportRegisteredObjs() {
    if (gApp.mode != AppMode::kRegistration) {
        std::cout << "[ExportObj] only valid in Registration mode" << std::endl;
        return;
    }
    std::filesystem::create_directories(REG_MODEL_PATH);
    if (liverMesh3D)   liverMesh3D->exportObjFile(Reg_TARGET_FILE_PATH);
    if (portalMesh3D)  portalMesh3D->exportObjFile(Reg_PORTAL_FILE_PATH);
    if (veinMesh3D)    veinMesh3D->exportObjFile(Reg_VEIN_FILE_PATH);
    if (tumorMesh3D)   tumorMesh3D->exportObjFile(Reg_TUMOR_FILE_PATH);
    if (segmentMesh3D) segmentMesh3D->exportObjFile(Reg_SEGMENT_FILE_PATH);
    if (gbMesh3D)      gbMesh3D->exportObjFile(Reg_GB_FILE_PATH);
    std::cout << "[ExportObj] Registered OBJs exported to " << REG_MODEL_PATH << std::endl;

    // -------------------------------------------------------------------------
    // [CT-handoff] (1) write reg_transform.txt = Umeyama( model/liver.obj raw CT
    //   -> current registered liverMesh3D ). Same vertex order/count because
    //   organ meshes are not remeshed during registration.
    //   (5) write ct_roundtrip/<organ>.obj = T^-1(reg_<organ>.obj), the
    //   zero-deform CT-space stand-in for DEFORM. NOTE: the folder is named
    //   "ct_roundtrip" (NOT "deformed") on purpose so that dropping it back into
    //   REG takes the NORMAL drop path (prealign + register#2), per the new
    //   design -- a "deformed" folder would trigger preserve-pose and skip both.
    // -------------------------------------------------------------------------
    if (liverMesh3D && !liverMesh3D->mVertices.empty()) {
        // [CT-handoff] Umeyama needs the raw-CT liver that the CURRENT
        // liverMesh3D actually came from -- not the startup model when a
        // different organ set was dropped mid-session. Use the tracked
        // source path; fall back to model/liver.obj only if unset.
        const std::string ctSrc = g_liverCtSourcePath.empty()
            ? (MODEL_PATH + "liver.obj") : g_liverCtSourcePath;
        mCutMesh rawCt = mCutMesh().loadMeshFromFile(ctSrc.c_str());
        if (rawCt.mVertices.size() != liverMesh3D->mVertices.size() ||
            rawCt.mVertices.empty()) {
            std::cerr << "[CT-handoff] vertex-count mismatch: " << ctSrc << "("
                      << rawCt.mVertices.size() / 3 << ") vs registered liver("
                      << liverMesh3D->mVertices.size() / 3
                      << "). Skipping reg_transform.txt / ct_roundtrip." << std::endl;
        } else {
            double s, detR, rmse, maxResid; int npair;
            const glm::mat4 T = CtHandoff::umeyamaSimilarity(
                rawCt.mVertices, liverMesh3D->mVertices,
                s, detR, rmse, maxResid, npair);
            const glm::mat4 Tinv = glm::inverse(T);

            // reg_transform.txt (self-describing header + 16 floats column-major)
            const std::string tp = REG_MODEL_PATH + "reg_transform.txt";
            std::ofstream tf(tp);
            if (tf.is_open()) {
                tf << "# reg_transform.txt : CT(model/*.obj) -> registered(reg_*.obj) similarity\n";
                tf << "# layout: column-major 4x4 (glm). 16 floats below, 4 per line.\n";
                tf << "# scale_s=" << s << " det_R=" << detR
                   << " umeyama_rmse=" << rmse << " max_resid=" << maxResid
                   << " npairs=" << npair << " src=liver.obj\n";
                const float* P = glm::value_ptr(T);
                tf << std::setprecision(10);
                for (int i = 0; i < 16; ++i) tf << P[i] << ((i % 4 == 3) ? '\n' : ' ');
                tf.close();
                std::cout << "[CT-handoff] wrote " << tp
                          << "  s=" << s << " det(R)=" << detR
                          << " rmse=" << rmse << " maxResid=" << maxResid
                          << " (N=" << npair << ")" << std::endl;
                if (detR < 0.0)
                    std::cerr << "[CT-handoff] WARNING: det(R)<0 -> reflection baked "
                                 "into reg_*.obj. Mirror handling needs review." << std::endl;
            } else {
                std::cerr << "[CT-handoff] cannot open " << tp << " for write" << std::endl;
            }

            // ct_roundtrip/<organ>.obj = T^-1(reg organ)  (zero-deform CT stand-in)
            if (CtHandoff::g_writeZeroDeformRoundtrip) {
                const std::string dDir = REG_MODEL_PATH + "ct_roundtrip/";
                std::filesystem::create_directories(dDir);
                struct OrganOut { mCutMesh* m; const char* name; };
                const OrganOut outs[6] = {
                    { liverMesh3D,   "liver.obj"   }, { portalMesh3D,  "portal.obj"  },
                    { veinMesh3D,    "vein.obj"    }, { tumorMesh3D,   "tumor.obj"   },
                    { segmentMesh3D, "segment.obj" }, { gbMesh3D,      "gb.obj"      },
                };
                int wrote = 0;
                for (const auto& o : outs) {
                    if (!o.m || o.m->mVertices.empty()) continue;
                    mCutMesh cp = *o.m;          // keep faces/texcoords
                    cp.mNormals.clear();         // drop stale post-transform normals
                    CtHandoff::applyMat4(cp.mVertices, Tinv);  // registered -> CT
                    cp.exportObjFile(dDir + o.name);
                    ++wrote;
                }

                // self-check: T^-1(reg_liver) should reproduce model/liver.obj raw CT.
                double devMax = 0.0, devSum = 0.0;
                std::vector<GLfloat> chk = liverMesh3D->mVertices;
                CtHandoff::applyMat4(chk, Tinv);
                for (size_t i = 0; i + 2 < chk.size() && i + 2 < rawCt.mVertices.size(); i += 3) {
                    const double dx = chk[i]   - rawCt.mVertices[i];
                    const double dy = chk[i+1] - rawCt.mVertices[i+1];
                    const double dz = chk[i+2] - rawCt.mVertices[i+2];
                    const double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
                    devMax = std::max(devMax, d); devSum += d;
                }
                const double devMean = devSum / std::max<size_t>(1, chk.size() / 3);
                std::cout << "[CT-handoff] ct_roundtrip/: wrote " << wrote
                          << " organ(s) to " << dDir << std::endl;
                std::cout << "[CT-handoff] self-check T^-1(reg_liver) vs model/liver.obj (CT mm): "
                          << "maxDev=" << devMax << " meanDev=" << devMean
                          << "  (should be ~0; equals Umeyama residual / scale)" << std::endl;
                std::cout << "[CT-handoff] NEXT: drop the ct_roundtrip/ folder into REG "
                             "(normal drop), then run registration #2 and Shift+M; the tumor "
                             "should land where cycle #1 did." << std::endl;
            }
        }
    } else {
        std::cerr << "[CT-handoff] no registered liver mesh; skipping T export." << std::endl;
    }

    // [deform round-trip] 最初の「Deform >>」時(= liver はまだ未変形の相似変換)に
    //   reg->CT-mm スケールを一度だけ固定する。以降の反復では reg_liver.obj が
    //   変形済みになるため再捕捉しない(2回目以降は rigid refine 前提でスケール不変)。
    //   Shift+M はこの値を使い、変形に依らず CT-mm スケールを保つ。
    if (!g_hasRegToMmScale && g_hasOriginalDiags && liverMesh3D &&
        !liverMesh3D->mVertices.empty()) {
        glm::vec3 mn(liverMesh3D->mVertices[0], liverMesh3D->mVertices[1],
                     liverMesh3D->mVertices[2]), mx = mn;
        for (size_t i = 0; i + 2 < liverMesh3D->mVertices.size(); i += 3) {
            glm::vec3 v(liverMesh3D->mVertices[i], liverMesh3D->mVertices[i + 1],
                        liverMesh3D->mVertices[i + 2]);
            mn = glm::min(mn, v); mx = glm::max(mx, v);
        }
        float curDiag = glm::length(mx - mn);
        if (curDiag > 1e-9f) {
            g_regToMmScale    = g_originalLiverDiagMm / curDiag;
            g_hasRegToMmScale = true;
            std::cout << "[ExportObj] captured reg->mm scale = " << g_regToMmScale
                      << " (frozen for Shift+M across the deform round-trip)" << std::endl;
        }
    }
}
void StlExport::exportCamMmStlWithSnapshot() {
    if (gApp.mode != AppMode::kRegistration) {
        std::cout << "[ExportSTL] only valid in Registration mode" << std::endl;
        return;
    }
    std::filesystem::create_directories(REG_MODEL_PATH);

    auto bboxDiag = [](const mCutMesh* m) -> float {
        if (!m || m->mVertices.size() < 3) return 0.0f;
        glm::vec3 mn(m->mVertices[0], m->mVertices[1], m->mVertices[2]), mx = mn;
        for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
            glm::vec3 v(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2]);
            mn = glm::min(mn, v);
            mx = glm::max(mx, v);
        }
        return glm::length(mx - mn);
    };

    float SCALE_RESTORE = 1.0f;
    // [deform round-trip] 変形に依らない固定 reg->mm スケールがあれば最優先で使う。
    //   無い(pure REG 運用 / 未捕捉)ときだけ従来の対角長比にフォールバックする。
    //   従来式は「現在 liver が CT の相似変換」前提なので、変形後の liver では
    //   bbox 対角に変形が混ざり mm スケールがずれる(固定スケールならこれを回避)。
    if (g_hasRegToMmScale) {
        SCALE_RESTORE = g_regToMmScale;
        std::cout << "[Shift+M] using frozen reg->mm scale = " << SCALE_RESTORE
                  << " (deformation-independent)" << std::endl;
    } else if (g_hasOriginalDiags && liverMesh3D) {
        float current_liver_diag = bboxDiag(liverMesh3D);
        if (current_liver_diag < 1e-9f) {
            std::cerr << "[Shift+M] current liver diag near zero -- using 1.0 fallback"
                      << std::endl;
        } else {
            SCALE_RESTORE = g_originalLiverDiagMm / current_liver_diag;
        }
    } else {
        std::cerr << "[Shift+M] WARNING: original CT diagonals not captured at startup; "
                     "using 1.0 (output will likely be in wrong scale)" << std::endl;
    }
    std::cout << "[Shift+M] g_originalLiverDiagMm=" << g_originalLiverDiagMm
              << "  current_liver_diag=" << (liverMesh3D ? bboxDiag(liverMesh3D) : 0.0f)
              << "  SCALE_RESTORE=" << SCALE_RESTORE
              << "  (CT-truth based, X+Z flip)" << std::endl;
    auto exportCamMmStl =
        [&](const mCutMesh* src, const std::string& outPath, const char* label) {
            if (!src || src->mVertices.empty()) {
                std::cout << "[Shift+M] Skip " << label << " (mesh empty)" << std::endl;
                return;
            }
            mCutMesh out = *src;
            for (size_t i = 0; i + 2 < out.mVertices.size(); i += 3) {
                out.mVertices[i  ] = -out.mVertices[i  ] * SCALE_RESTORE;  // X flip
                out.mVertices[i+1] =  out.mVertices[i+1] * SCALE_RESTORE;  // Y keep
                out.mVertices[i+2] = -out.mVertices[i+2] * SCALE_RESTORE;  // Z flip
            }
            out.exportStlFile(outPath);
            Reg3DCustom::printMeshBBox(out, label);
        };

    exportCamMmStl(tumorMesh3D, REG_MODEL_PATH + "tumor_cam_mm.stl",
                   "tumor_cam_mm (Adagolodjo-format)");
    exportCamMmStl(liverMesh3D, REG_MODEL_PATH + "liver_cam_mm.stl",
                   "liver_cam_mm (Adagolodjo-format)");

    std::cout << "[Shift+M] Exported tumor/liver STL in cam-mm (Adagolodjo format)"
              << "  (v4: CT-truth scale, X+Z flip, no index swap)" << std::endl;

    try {
        // obj-migration Phase 6: snapshot the canonical (tag-less) outputs.
        const std::string srcName = intrinsicsSourceToTag(g_intrinsicsSource,
                                                          g_currentPresetKey);

        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tmLocal{};
#ifdef _WIN32
        localtime_s(&tmLocal, &t);
#else
        localtime_r(&t, &tmLocal);
#endif
        char tsBuf[32];
        std::strftime(tsBuf, sizeof(tsBuf), "%Y%m%d_%H%M%S", &tmLocal);

        std::string snapDir = REG_MODEL_PATH + "snapshot_" + tsBuf + "/";
        std::filesystem::create_directories(snapDir);

        const std::vector<std::string> srcFiles = {
            DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj",
            DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.mtl",
            DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_light.obj",
            DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_light.mtl",
            DEPTH_OUTPUT_PATH + "intrinsics.txt",
            DEPTH_OUTPUT_PATH + "intrinsics_da3.txt",
            DEPTH_OUTPUT_PATH + "depth_metric.bin",
            DEPTH_OUTPUT_PATH + "original_rectified.jpg",
            DEPTH_OUTPUT_PATH + "segmentation_mask.png",
            DEPTH_OUTPUT_PATH + "instrument_segmentation_mask.png",
            DEPTH_OUTPUT_PATH + "texture.png",
            REG_MODEL_PATH + "tumor_cam_mm.stl",
            REG_MODEL_PATH + "liver_cam_mm.stl",
        };

        int copiedCount = 0;
        int missingCount = 0;
        std::vector<std::string> missing;
        for (const auto& src : srcFiles) {
            std::filesystem::path srcPath(src);
            if (!std::filesystem::exists(srcPath)) {
                ++missingCount;
                missing.push_back(srcPath.filename().string());
                continue;
            }
            std::filesystem::path dstPath =
                std::filesystem::path(snapDir) / srcPath.filename();
            try {
                std::filesystem::copy_file(
                    srcPath, dstPath,
                    std::filesystem::copy_options::overwrite_existing);
                ++copiedCount;
            } catch (const std::exception& e) {
                std::cerr << "[Shift+M] copy failed: " << src
                          << " -> " << dstPath
                          << " : " << e.what() << std::endl;
            }
        }

        {
            std::ofstream readme(snapDir + "README.txt");
            if (readme.is_open()) {
                readme << "Snapshot created: " << tsBuf << "\n"
                       << "Intrinsics source (g_intrinsicsSource="
                       << intrinsicsSourceToLegacyInt(g_intrinsicsSource) << "): " << srcName << "\n"
                       << "g_originalLiverDiagMm = " << g_originalLiverDiagMm << " mm\n"
                       << "g_originalTumorDiagMm = " << g_originalTumorDiagMm << " mm\n"
                       << "SCALE_RESTORE used     = " << SCALE_RESTORE << "\n"
                       << "\n"
                       << "Source paths at capture time:\n"
                       << "  DEPTH_OUTPUT_PATH = " << DEPTH_OUTPUT_PATH << "\n"
                       << "  REG_MODEL_PATH    = " << REG_MODEL_PATH << "\n";
                readme.close();
                ++copiedCount;
            }
        }

        std::cout << "[Shift+M] Snapshot saved: " << snapDir
                  << "  (" << copiedCount << " files copied, "
                  << missingCount << " missing)" << std::endl;
        if (missingCount > 0) {
            std::cout << "[Shift+M] Missing in snapshot (skipped):";
            for (const auto& m : missing) std::cout << "\n  - " << m;
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Shift+M] Snapshot creation failed: " << e.what()
                  << "  (STL export was successful, only snapshot failed)"
                  << std::endl;
    }
}

static void glfw_onKey(GLFWwindow* win, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    // [key-reorg Phase 5] isShiftV / isShiftF removed (Shift+V/F -> Alt+G/Alt+Shift+G).
    // [key-reorg Phase 9] isShiftE removed (Shift+E -> Alt+P; key==P already in needsScene).
    const bool isShiftG     = (key == GLFW_KEY_G) && (mods & GLFW_MOD_SHIFT) && !(mods & GLFW_MOD_CONTROL);  // V3 BIPOP-CMA-ES (Good performance)
    const bool isCtrlShiftG = (key == GLFW_KEY_G) && (mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT);   // V3-RS BIPOP-CMA-ES (silhouette anchor)
    const bool isCtrlG      = (key == GLFW_KEY_G) && (mods & GLFW_MOD_CONTROL) && !(mods & GLFW_MOD_SHIFT);  // V3-R BIPOP-CMA-ES (Region-aware, 4-quadrant subset)
    const bool isAltG       = (key == GLFW_KEY_G) && (mods & GLFW_MOD_ALT) && !(mods & GLFW_MOD_CONTROL) && !(mods & GLFW_MOD_SHIFT);  // V1 BIPOP-CMA-ES (旧 Shift+V)
    const bool isAltShiftG  = (key == GLFW_KEY_G) && (mods & GLFW_MOD_ALT) && (mods & GLFW_MOD_SHIFT) && !(mods & GLFW_MOD_CONTROL);   // V2 BIPOP-CMA-ES Fast (旧 Shift+F)
    // Shift+N        : Normal-Compatible refine (finishing pass after Ctrl+G)
    // Ctrl+Shift+N   : SRT Variance-Weighted refine (ablation alt)
    // Plain N        : kept as the SourceVis toggle (no scene requirement)
    const bool isShiftN     = (key == GLFW_KEY_N) && (mods & GLFW_MOD_SHIFT) && !(mods & GLFW_MOD_CONTROL);
    const bool isCtrlShiftN = (key == GLFW_KEY_N) && (mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT);
    const bool needsScene = (key == GLFW_KEY_O      ||
                             key == GLFW_KEY_P      ||
                             isShiftG               ||
                             isCtrlG                ||
                             isCtrlShiftG           ||
                             key == GLFW_KEY_I      ||   // [V3I] Ctrl+I pure squash-IoU
                             isAltG                 ||  // [key-reorg] V1 (旧 Shift+V)
                             isAltShiftG            ||  // [key-reorg] V2 (旧 Shift+F)
                             isShiftN               ||
                             isCtrlShiftN           ||
                             key == GLFW_KEY_D      ||
                             key == GLFW_KEY_COMMA  ||
                             key == GLFW_KEY_PERIOD);
    if (needsScene && gApp.mode != AppMode::kRegistration) {
        std::cout << "[Key] '" << (char)key
                  << "' requires a loaded scene -- drop an image and"
                     " press R to run the depth pipeline." << std::endl;
        return;
    }

    // Ctrl+D — toggle the consolidated Debug Panel. Handled before the switch
    // so it does not fall through to the plain-D (AR save) case below.
    if (key == GLFW_KEY_D && (mods & GLFW_MOD_CONTROL) && !(mods & GLFW_MOD_SHIFT)) {
        g_debugPanel.showWindow = !g_debugPanel.showWindow;
        std::cout << "[DebugPanel] " << (g_debugPanel.showWindow ? "ON" : "OFF") << std::endl;
        return;
    }

    // Space — Live Calibration shutter hotkey. Active only while a session
    // is in progress AND the camera is in live (not captured) mode. Mirrors
    // the on-screen shutter button so the user can hold the chessboard in
    // both hands and tap Space to take a frame. No conflict with the rest
    // of the app because Space is otherwise unused.
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS && !mods &&
        g_liveCalibActive && gCamera.active && !gCamera.captured) {
        if (gUI.actions.onLiveCalibCapture) gUI.actions.onLiveCalibCapture();
        return;
    }

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        break;
    case GLFW_KEY_O:
        if (mods & GLFW_MOD_SHIFT) {
            // ---- Shift+O : QuadAuto -----------------------------------------
            //   AR 固定視点 ∩ g_activeQuadrantMask の解剖象限、の交差集合を
            //   source として FGR+ICP。Key O (HemiAuto) と独立、副作用なし。
            //   Region/LR labels が未計算なら auto-trigger
            //   (applyInitRotation と同じ流儀)。
            if (!g_liverRegion.valid()) {
                std::cout << "[Shift+O] LiverRegion (Shift+R) not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverRegion();
            }
            if (!g_liverLR.valid()) {
                std::cout << "[Shift+O] LiverLR (Y) not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverLR();
            }
            gUI.state.regMethod = 1;        // HemiAuto と同じ表示扱い
            g_stepStartTime  = std::chrono::steady_clock::now();
            g_sessionBipopN  = 0;            // 新規試行扱い
            poseAutoSaveBeforeRegistration();
            runQuadAuto();
            poseSaveToLibrary(SaveCriterion::RMSE);
        } else {
            // ---- O : HemiAuto (既存) ----------------------------------------
            //         元コード line 3213-3238 (a.onHemiAuto) の順序に合わせる
            gUI.state.regMethod = 1;
            g_stepStartTime  = std::chrono::steady_clock::now();
            g_sessionBipopN  = 0;          // HemiAuto は新規試行扱い: BIPOP カウンタをリセット
            poseAutoSaveBeforeRegistration();
            runHemiAuto();
            poseSaveToLibrary(SaveCriterion::RMSE);
        }
        break;
    // [key-reorg Phase 5] GLFW_KEY_V / GLFW_KEY_F removed:
    //   Shift+V (V1 BIPOP) -> Alt+G  ;  Shift+F (V2 BIPOP) -> Alt+Shift+G
    //   plain V (cluster viz) -> Ctrl+D > Viz tab
    // ============================================================
    // [V3I] Ctrl+I : pure squash-IoU registration. Objective-only
    //   change of Ctrl+Shift+G -- reuses the entire V3RS pipeline via
    //   runBipopCmaesV3I() (cost = 1 - IoU2D, RMSE cap bypassed,
    //   selector by IoU). Ctrl+G / Ctrl+Shift+G are untouched.
    // ============================================================
    case GLFW_KEY_I:
        if ((mods & GLFW_MOD_CONTROL) && !(mods & GLFW_MOD_SHIFT)
                                      && !(mods & GLFW_MOD_ALT)) {
            std::cout << "[Ctrl+I] V3I pure squash-IoU session start" << std::endl;
            const auto maskStrI = LiverLeftRightLabel::quadrantMaskString(
                g_activeQuadrantMask);
            std::cout << "[Ctrl+I] quadrant_mask = " << maskStrI
                      << "  (0x" << std::hex << (unsigned)g_activeQuadrantMask
                      << std::dec << ")" << std::endl;
            // [UI整理] Auto-compute labels on demand (no HemiAuto needed) so
            // Ctrl+I runs straight from Apply Init Pose / after manual drag.
            if (!g_liverRegion.valid()) recomputeLiverRegion();
            if (!g_liverLR.valid())     recomputeLiverLR();
            if (!g_liverRegion.valid() || !g_liverLR.valid()) {
                std::cerr << "[Ctrl+I] ERROR: label computation failed "
                             "(is the liver mesh loaded?)." << std::endl;
                break;
            }
            auto subsetI = LiverLeftRightLabel::makeQuadrantSubsetIdx(
                g_liverRegion.labels, g_liverLR.labels,
                g_activeQuadrantMask);
            if (subsetI.empty()) {
                std::cerr << "[Ctrl+I] ERROR: subset is empty." << std::endl;
                break;
            }
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;   // same lane as V3RS
            poseAutoSaveBeforeRegistration();
            runBipopCmaesV3I(g_activeQuadrantMask);
            poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
        } else if ((mods & GLFW_MOD_SHIFT) && !(mods & GLFW_MOD_CONTROL)
                                           && !(mods & GLFW_MOD_ALT)) {
            // ----- Shift+I : silhouette-fixed dilation RMSE refine (stage 2) -----
            //   Sequel to Alt+P / Ctrl+I. Those align the silhouette
            //   (maximise 2D-IoU) but cannot constrain the one DOF the
            //   pinhole projection is blind to: uniform dilation about the
            //   optical center (depth <-> apparent size). Shift+I sweeps
            //   exactly that 1-D ray to minimise the 3D RMSE against the
            //   DA3 depth cloud, leaving the silhouette (IoU) fixed by
            //   construction. Save by RMSE: IoU is invariant here, so an
            //   IoU gate would see zero change and reject every result.
            std::cout << "[Shift+I] dispatching Dilation RMSE refine..." << std::endl;
            if (!registrationHandle.useRegistration) {
                std::cerr << "[Shift+I] Run HemiAuto (O) + Alt+P / Ctrl+I first."
                          << std::endl;
                break;
            }
            g_stepStartTime = std::chrono::steady_clock::now();
            poseAutoSaveBeforeRegistration();
            runDilationRmseRefine();
            gUI.state.regMethod = 3;   // refine lane (same as Ctrl+I / V3RS)
            poseSaveToLibrary(SaveCriterion::RMSE, g_activeQuadrantMask);
        }
        break;

    case GLFW_KEY_G:
        // Shift+G: BIPOP-CMA-ES V3 ("Good performance"). Pure-function
        // refactor of V2 with liver-only snapshot, matrix-based per-Run
        // result, and zero global writes inside the inner loop --
        // foundation for V3-4 (population OMP) and V3-5 (run OMP).
        // V3-1 ships with src_voxel_ratio = tgt_voxel_ratio = 0, which
        // makes CompRMSE bit-identical to Shift+V (V1) and Shift+F (V2
        // FULL_MESH) from the same (g_trialSeed, g_callIdx) state --
        // the validation hook for the V3 architectural refactor.
        // V3-2 will set both ratios to 0.015f for the speedup.
        // Plain G (no shift) is intentionally left unbound.
        //
        // Ctrl+G : BIPOP-CMA-ES V3-R (Region-aware) — 4 象限選択
        //          (g_activeQuadrantMask) で対応点候補を絞った CMA-ES。
        //          S4 完了: CmaesRefineV3R::runBipopCmaesV3R を呼ぶ。
        //          QUAD_ALL のとき Shift+G (V3) と数値 byte-identical
        //          (HANDOVER §2.6 受け入れ基準)。
        // ----- Alt+G : V1 BIPOP-CMA-ES (旧 Shift+V) -----
        //   byte-identical to the legacy Shift+V branch (same order:
        //   g_stepStartTime / g_sessionBipopN++ / regMethod=3 /
        //   poseAutoSaveBeforeRegistration / runBipopCmaes / poseSaveToLibrary).
        //   Must be tested BEFORE the Ctrl+Shift+G branch (mod-order).
        if ((mods & GLFW_MOD_ALT) && !(mods & GLFW_MOD_CONTROL)
                                  && !(mods & GLFW_MOD_SHIFT)) {
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            runBipopCmaes();
            poseSaveToLibrary(SaveCriterion::RMSE);
            break;
        }
        // ----- Alt+Shift+G : V2 BIPOP-CMA-ES Fast (旧 Shift+F) -----
        if ((mods & GLFW_MOD_ALT) && (mods & GLFW_MOD_SHIFT)
                                  && !(mods & GLFW_MOD_CONTROL)) {
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;   // BIPOP method (same as Shift+V)
            poseAutoSaveBeforeRegistration();
            runBipopCmaesV2();
            poseSaveToLibrary(SaveCriterion::RMSE);
            break;
        }
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT)) {
            // ----- Ctrl+Shift+G : V3-RS (silhouette-anchored) -----
            // Independent of Ctrl+G; calls a fresh wrapper that reads
            // a single sil-specific global g_ctrlgsLambdaSil. Quadrant
            // mask + Caudal + AR-vis + beta all still apply via the
            // shared g_ctrlg* globals because ParamsV3RS inherits
            // ParamsV3R.
            std::cout << "[Ctrl+Shift+G] V3-RS (silhouette anchor) session start"
                      << std::endl;
            const auto maskStr = LiverLeftRightLabel::quadrantMaskString(
                g_activeQuadrantMask);
            std::cout << "[Ctrl+Shift+G] quadrant_mask = " << maskStr
                      << "  (0x" << std::hex << (unsigned)g_activeQuadrantMask
                      << std::dec << ")" << std::endl;

            if (!g_liverRegion.valid() || !g_liverLR.valid()) {
                std::cerr << "[Ctrl+Shift+G] ERROR: labels not computed"
                          << " (Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                          << ", LR.valid="    << (g_liverLR.valid() ? "Y" : "N")
                          << "). Run HemiAuto (O) first."
                          << std::endl;
                break;
            }
            auto subsetCS = LiverLeftRightLabel::makeQuadrantSubsetIdx(
                g_liverRegion.labels, g_liverLR.labels,
                g_activeQuadrantMask);
            std::cout << "[Ctrl+Shift+G] subset_size = " << subsetCS.size()
                      << " / " << g_liverRegion.labels.size()
                      << " vertices (original-index space)"
                      << std::endl;
            if (subsetCS.empty()) {
                std::cerr << "[Ctrl+Shift+G] ERROR: subset is empty."
                          << std::endl;
                break;
            }

            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            runBipopCmaesV3RS(g_activeQuadrantMask);
            poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
        } else if (mods & GLFW_MOD_CONTROL) {
            // ----- Ctrl+G : V3-R (region-aware, S4 implemented) -----
            // g_activeQuadrantMask が示す 4 象限部分集合だけを KDTree
            // 入力とする region-aware BIPOP-CMA-ES。実体は
            // CmaesRefineV3R::runBipopCmaesV3R にあり、ParamsV3R に
            // region/lr ラベルと quadrant_mask を渡して呼び出す。
            std::cout << "[Ctrl+G] V3-R (region-aware) session start" << std::endl;
            const auto maskStr = LiverLeftRightLabel::quadrantMaskString(
                g_activeQuadrantMask);
            std::cout << "[Ctrl+G] quadrant_mask = " << maskStr
                      << "  (0x" << std::hex << (unsigned)g_activeQuadrantMask
                      << std::dec << ")" << std::endl;

            // [UI整理] Auto-compute labels on demand (no HemiAuto needed) so
            // Ctrl+G runs straight from Apply Init Pose / after manual drag.
            if (!g_liverRegion.valid()) recomputeLiverRegion();
            if (!g_liverLR.valid())     recomputeLiverLR();
            if (!g_liverRegion.valid() || !g_liverLR.valid()) {
                std::cerr << "[Ctrl+G] ERROR: label computation failed"
                          << " (Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                          << ", LR.valid="    << (g_liverLR.valid() ? "Y" : "N")
                          << "). Is the liver mesh loaded?"
                          << std::endl;
                break;
            }
            auto subset = LiverLeftRightLabel::makeQuadrantSubsetIdx(
                g_liverRegion.labels, g_liverLR.labels,
                g_activeQuadrantMask);
            std::cout << "[Ctrl+G] subset_size = " << subset.size()
                      << " / " << g_liverRegion.labels.size()
                      << " vertices (original-index space)";
            if (g_activeQuadrantMask == LiverLeftRightLabel::QUAD_ALL) {
                std::cout << "  (QUAD_ALL: byte-identical to V3 expected)";
            }
            std::cout << std::endl;

            if (subset.empty()) {
                std::cerr << "[Ctrl+G] ERROR: subset is empty for mask=0x"
                          << std::hex << (unsigned)g_activeQuadrantMask
                          << std::dec
                          << ". Select at least one quadrant in the UI panel."
                          << std::endl;
                break;
            }

            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;   // BIPOP method (same as Shift+V/F/G)
            poseAutoSaveBeforeRegistration();
            // S4: V3-R driver dispatch (HANDOVER §4.6 切替).
            runBipopCmaesV3R(g_activeQuadrantMask);
            // S5 (V3R): PoseEntry に quadrant_mask を記録するため、
            // Ctrl+G 経路でのみ第 2 引数 (g_activeQuadrantMask) を渡す。
            // 他の経路 (Shift+G/V/F/E、HemiAuto 等) はデフォルト 0xFF (legacy)。
            poseSaveToLibrary(SaveCriterion::RMSE, g_activeQuadrantMask);
        } else if (mods & GLFW_MOD_SHIFT) {
            // ----- Shift+G : V3 (既存) ----------------------------------
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;   // BIPOP method (same as Shift+V/F)
            poseAutoSaveBeforeRegistration();
            runBipopCmaesV3();
            poseSaveToLibrary(SaveCriterion::RMSE);
        }
        break;
    // [key-reorg Phase 4] GLFW_KEY_B removed: B (boundary candidates) and
    //   Shift+B (cyclic correspondence) viz toggles moved to Ctrl+D > Viz tab.
    case GLFW_KEY_N:
        // [LIVE] Don't let Shift+N start while a CPD/soft-ICP replay is
        //   animating (both would apply transforms to the mesh the same
        //   frame). The reverse is already guarded by anyRegLiveActive().
        if ((mods & GLFW_MOD_SHIFT) && (CpdLive::active || SoftIcpLive::active)) {
            std::cout << "[Shift+N] a CPD/soft-ICP live session is animating — "
                         "wait for it to finish" << std::endl;
            break;
        }
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT)) {
            // ---- Ctrl+Shift+N : SRT Variance-Weighted Refine -----------
            //   Same wrapper as Shift+N but method = SRT_VARIANCE. Uses
            //   the same source filter (AR-vis + quad + caudal) and the
            //   same L1/L2 rim & anchor controls. Intended for ablation
            //   against Shift+N (NormalCompat). See
            //   NormalCompatibleRefine.h for the per-correspondence
            //   weighting differences.
            if (gApp.mode != AppMode::kRegistration) {
                std::cout << "[Ctrl+Shift+N] only valid in Registration mode"
                          << std::endl;
                break;
            }
            // If a Live session is already running, this keypress STOPS
            // it (finalize -> the main render loop consumes pendingSave
            // on the next frame and calls poseSaveToLibrary).
            if (NormalRefineLive::active) {
                std::cout << "[Ctrl+Shift+N] stopping active Live session"
                          << std::endl;
                finishNormalCompatRefineLive();
                break;
            }
            std::cout << "[Ctrl+Shift+N] SRT-Variance refine session start  "
                      << "mode=" << (g_normRefineLiveMode ? "LIVE (frame-driven)"
                                                          : "BLOCKING (one shot)")
                      << std::endl;
            g_stepStartTime = std::chrono::steady_clock::now();
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            if (g_normRefineLiveMode) {
                // Live (frame-driven, object-tracking-style visualisation).
                // The main loop will tick this until convergence/max-iter
                // and then consume pendingSave to call poseSaveToLibrary.
                startNormalCompatRefineLive(
                    NormalRefine::SRT_VARIANCE, g_activeQuadrantMask);
            } else {
                runNormalCompatRefineSession(
                    NormalRefine::SRT_VARIANCE, g_activeQuadrantMask);
                poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
            }
        } else if (mods & GLFW_MOD_SHIFT) {
            // ---- Shift+N : Normal-Compatible Refine (default polish) ---
            //   "Finishing pass" after Ctrl+G has done the global search.
            //   In LIVE mode (g_normRefineLiveMode=true, default), the
            //   mesh visibly moves frame-by-frame like an SRT-3D object
            //   tracker. In BLOCKING mode, the wrapper runs to completion
            //   in one frame.
            //
            //   Source/target filters share the Ctrl+G panel toggles
            //   (g_ctrlgUseArVisFilter, g_ctrlgUseCaudalOnly,
            //    g_activeQuadrantMask) — no separate filter UI here.
            if (gApp.mode != AppMode::kRegistration) {
                std::cout << "[Shift+N] only valid in Registration mode"
                          << std::endl;
                break;
            }
            // Same Live-session "press to stop" semantic as Ctrl+Shift+N.
            if (NormalRefineLive::active) {
                std::cout << "[Shift+N] stopping active Live session"
                          << std::endl;
                finishNormalCompatRefineLive();
                break;
            }
            std::cout << "[Shift+N] Normal-Compatible refine session start  "
                      << "mode=" << (g_normRefineLiveMode ? "LIVE (frame-driven)"
                                                          : "BLOCKING (one shot)")
                      << std::endl;
            g_stepStartTime = std::chrono::steady_clock::now();
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            if (g_normRefineLiveMode) {
                startNormalCompatRefineLive(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
            } else {
                runNormalCompatRefineSession(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
            }
        }
        // [key-reorg Phase 4] plain N (source visualization toggle) moved to
        // Ctrl+D > Viz tab. Shift+N / Ctrl+Shift+N (refine) kept above.
        break;
    case GLFW_KEY_W:
        // ---- Phase 7b: W-family (RIM Shape Match) ----
        //   Plain W       : source RIM chain overlay toggle (green dots)
        //   Shift+W       : target boundary overlay toggle (purple dots)
        //   Ctrl+W        : Shape Match → mesh apply → PoseLibrary save
        //                   (推奨運用: → 続けてユーザが手動で Ctrl+G)
        //   Ctrl+Shift+W  : Shape Match + rim axis sweep + save (Step 4b)
        //                   ICP は skip、rim 絶対保持優先
        //   Alt+W         : Shape Match Coarse2D + Gauss-Newton refine
        //                   (Phase 7b Step 3b — PnP-style sub-pixel refine)
        //   Ctrl+Alt+W    : Contour Sweep (Phase 7b Step 3c)
        //                   discrete arc-length × rotation grid search,
        //                   Z-axis-preserving by construction
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_ALT)
                                     && !(mods & GLFW_MOD_SHIFT)) {
            // ==== Ctrl+Alt+W : Step 3c or 3d (toggle on g_silhouetteSweepEnable) ====
            //   Starts a multi-frame live sweep. Per-frame batch is
            //   processed in the main loop after rendering; mesh is
            //   applied only on completion (no oscillation during sweep).
            //
            //   When g_silhouetteSweepEnable is OFF (default), runs the
            //   Step 3c sector-based contour sweep (legacy).
            //   When ON, runs the Step 3d silhouette 2D dense sweep
            //   (no LR constraint, no bbox filter, no rot cap; the
            //    right-start convention locks the orientation).
            if (g_contourSweepState.active || g_silhouetteSweep.active) {
                std::cout << "[Ctrl+Alt+W] sweep already running"
                          << "  (3c.phase=" << g_contourSweepState.phase
                          << "  3d.phase=" << g_silhouetteSweep.phase
                          << ")  — ignored" << std::endl;
                break;
            }
            if (gApp.mode != AppMode::kRegistration) {
                std::cout << "[Ctrl+Alt+W] requires a loaded scene"
                          << std::endl;
                break;
            }
            // Auto-trigger labels (same as Ctrl+W).
            // Step 3c++ label-based orientation lock needs BOTH LR and
            // CC labels — unlike Ctrl+W which only needs CC when
            // caudal-only is requested, here we always pull CC so the
            // sweep can enforce screen-up ↔ patient-cranial alignment.
            // Step 3d only needs LR (PURE_RIGHT centroid), but we
            // pull both for parity / safety.
            if (!g_liverRegion.valid()) recomputeLiverRegion();
            if (!g_liverLR.valid())     recomputeLiverLR();
            if (!g_liverCC.valid())     recomputeLiverCC();
            if (g_silhouetteSweepEnable) {
                startSilhouetteSweep();
            } else {
                startContourSweep();
            }
            break;
        }
        if ((mods & GLFW_MOD_ALT) && !(mods & GLFW_MOD_CONTROL)
                                  && !(mods & GLFW_MOD_SHIFT)) {
            // ==== Alt+W : Coarse2D + Gauss-Newton refine + apply + save ==
            //   Flow:
            //     1. runDebugShapeMatchGN() → runs Coarse2D then LM refine
            //        (updates g_debugShapeMatchBestTransform with refined T)
            //     2. Apply refined T to all organ meshes
            //     3. Save to PoseLibrary via 0-iter pseudo-session
            //   Mirrors Ctrl+W's structure but with the GN-refined transform.
            if (NormalRefineLive::active) {
                std::cout << "[Alt+W] stopping active Live session"
                          << std::endl;
                finishNormalCompatRefineLive();
                break;
            }
            if (gApp.mode != AppMode::kRegistration) {
                std::cout << "[Alt+W] requires a loaded scene (Registration mode)"
                          << std::endl;
                break;
            }
            // Auto-trigger labels (same as Ctrl+W)
            if (!g_liverRegion.valid()) {
                std::cout << "[Alt+W] auto-running recomputeLiverRegion()..."
                          << std::endl;
                recomputeLiverRegion();
            }
            if (!g_liverLR.valid()) {
                std::cout << "[Alt+W] auto-running recomputeLiverLR()..."
                          << std::endl;
                recomputeLiverLR();
            }
            if (g_ctrlgUseCaudalOnly && !g_liverCC.valid()) {
                std::cout << "[Alt+W] auto-running recomputeLiverCC()..."
                          << std::endl;
                recomputeLiverCC();
            }
            // 1. Run Coarse2D + GN
            std::cout << "[Alt+W] running Coarse2D + Gauss-Newton refine..."
                      << std::endl;
            if (!runDebugShapeMatchGN()) {
                std::cout << "[Alt+W] failed — abort, mesh unchanged"
                          << std::endl;
                break;
            }
            std::cout << "[Alt+W] refined T:"
                      << "  cost=" << g_debugShapeMatchBestCost << "px"
                      << "  GN iters=" << g_debugShapeMatchGNIters
                      << "  Δ=" << (g_debugShapeMatchGNInitCost
                                    - g_debugShapeMatchGNFinalCost) << "px"
                      << std::endl;

            // 2. Undo snapshot
            poseAutoSaveBeforeRegistration();

            // 3. Apply refined T to all organ meshes
            {
                auto organs = getOrganList();
                int n_valid = 0;
                for (auto* m : organs) if (m) n_valid++;
                std::cout << "[Alt+W] applying refined T to "
                          << n_valid << " organ meshes..." << std::endl;
                NormalRefine::applyIncrementalTransform(
                    glm::dmat4(g_debugShapeMatchBestTransform), organs);
            }

            // 4. Clear preview state
            g_showDebugShapeMatch = false;
            g_debugShapeMatchBestSrc.clear();

            // 5. Save to PoseLibrary
            std::cout << "[Alt+W] saving pose to PoseLibrary"
                      << " (max_iter=0 pseudo-session)..." << std::endl;
            const int saved_max_iter = g_normRefineMaxIter;
            g_normRefineMaxIter = 0;
            g_stepStartTime  = std::chrono::steady_clock::now();
            gUI.state.regMethod = 3;
            if (g_normRefineLiveMode) {
                startNormalCompatRefineLive(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
            } else {
                runNormalCompatRefineSession(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                poseSaveToLibrary(SaveCriterion::EITHER,
                                  g_activeQuadrantMask);
            }
            g_normRefineMaxIter = saved_max_iter;
            break;
        }
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT)) {
            // ==== Ctrl+Shift+W : Step 4 — apply best T + Live bridge ====
            //   詳細設計 §3-4 の本番フロー:
            //     1. coarse search (auto-trigger source/target populate)
            //     2. best T を全 organ mesh に applyIncrementalTransform
            //     3. poseAutoSaveBeforeRegistration() で Undo 用 snapshot
            //     4. startNormalCompatRefineLive で Live ICP に bridge
            //   失敗時は Ctrl+Z (Undo) で Apply Init Pose 状態に戻れる。
            //
            //   Stop semantic: Live が active なら停止 (Shift+N と同じ流儀)
            if (NormalRefineLive::active) {
                std::cout << "[Ctrl+Shift+W] stopping active Live session"
                          << std::endl;
                finishNormalCompatRefineLive();
                break;
            }
            if (gApp.mode != AppMode::kRegistration) {
                std::cout << "[Ctrl+Shift+W] requires a loaded scene"
                          << " (Registration mode)" << std::endl;
                break;
            }
            // Same label auto-trigger as Plain W / Ctrl+W
            if (!g_liverRegion.valid()) {
                std::cout << "[Ctrl+Shift+W] g_liverRegion not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverRegion();
            }
            if (!g_liverLR.valid()) {
                std::cout << "[Ctrl+Shift+W] g_liverLR not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverLR();
            }
            if (g_ctrlgUseCaudalOnly && !g_liverCC.valid()) {
                std::cout << "[Ctrl+Shift+W] caudal-only ON but g_liverCC"
                          << " not yet computed, auto-running..." << std::endl;
                recomputeLiverCC();
            }
            // 1. Coarse search (this also populates source/target if missing)
            std::cout << "[Ctrl+Shift+W] Step 4 start: running coarse search..."
                      << std::endl;
            if (!runDebugShapeMatchCoarse()) {
                std::cout << "[Ctrl+Shift+W] coarse search failed — abort,"
                          << " mesh unchanged" << std::endl;
                break;
            }
            std::cout << "[Ctrl+Shift+W] best T: cost="
                      << g_debugShapeMatchBestCost
                      << "  k=" << g_debugShapeMatchBestK
                      << std::endl;

            // 2. Undo snapshot BEFORE moving mesh
            poseAutoSaveBeforeRegistration();

            // 3. Apply best T to all organ meshes
            //    (everything moves rigidly: liver, tumor, vessels, ...)
            auto organs = getOrganList();
            int n_valid = 0;
            for (auto* m : organs) if (m) n_valid++;
            std::cout << "[Ctrl+Shift+W] applying best T to "
                      << n_valid << " organ meshes..." << std::endl;
            NormalRefine::applyIncrementalTransform(
                glm::dmat4(g_debugShapeMatchBestTransform), organs);

            // 4. Turn off Step 3 preview since mesh is now AT that position
            //    (the red dots would overlap the green dots awkwardly)
            g_showDebugShapeMatch = false;
            g_debugShapeMatchBestSrc.clear();

            // ---- Phase 7b Step 4b — Rim Axis Rotation Sweep -----------
            //   既存の Live ICP は rim 合致を維持しないので、ユーザの
            //   「RIM フィットは絶対保持」要件に応えるべく rim 軸まわり
            //   1D 回転で全頂点最適を取り、ICP を skip する。
            //
            //   g_shapeMatchAxisSweepCompare=true なら Variant A (full)
            //   と Variant B (rim-only) を両方走らせ、それぞれの best
            //   rotation を一旦適用 → 0-iter session で CompRMSE 計測 →
            //   逆 rotation で元に戻し → RMSE 良い方を最終採用する。
            if (g_shapeMatchAxisSweepEnabled) {
                std::cout << "[Ctrl+Shift+W] Step 4b: rim axis rotation"
                          << " sweep starting..." << std::endl;

                // (a) rim axis / centroid を baseline T で world frame に
                //     変換 (g_debugSourceRim* は Apply Init Pose 状態 = T 前)
                const glm::mat3 R_best(g_debugShapeMatchBestTransform);
                const glm::vec3 rim_axis_after_T = glm::normalize(
                    R_best * g_debugSourceRimMajorNormal);
                const glm::vec3 rim_centroid_after_T = glm::vec3(
                    g_debugShapeMatchBestTransform
                    * glm::vec4(g_debugSourceRimCentroid, 1.0f));

                // ==== Variant A: full vertex sweep =====================
                std::vector<glm::vec3> src_full;
                if (liverMesh3D) {
                    const size_t n = liverMesh3D->mVertices.size() / 3;
                    src_full.reserve(n);
                    for (size_t i = 0; i < n; i++) {
                        src_full.emplace_back(
                            liverMesh3D->mVertices[i*3],
                            liverMesh3D->mVertices[i*3+1],
                            liverMesh3D->mVertices[i*3+2]);
                    }
                }
                std::vector<glm::vec3> tgt_sub;
                {
                    auto cachedTgt = Reg3DCustom::getCachedTargetCloud();
                    if (cachedTgt && !cachedTgt->points.empty()) {
                        const size_t N_tgt  = cachedTgt->points.size();
                        const int    N_want = std::max(100,
                                                       g_shapeMatchAxisSweepTgtSubN);
                        const size_t stride = std::max<size_t>(1,
                                                               N_tgt / (size_t)N_want);
                        tgt_sub.reserve(N_tgt / stride + 1);
                        for (size_t i = 0; i < N_tgt; i += stride) {
                            tgt_sub.push_back(cachedTgt->points[i]);
                        }
                    }
                }

                std::vector<RimShape::AxisSweepResult> results_A;
                bool A_ok = false;
                if (!src_full.empty() && !tgt_sub.empty()) {
                    std::cout << "[Ctrl+Shift+W] === Variant A (full vertex"
                              << " symmetric chamfer) ===" << std::endl;
                    A_ok = RimShape::runRimAxisRotationSweep(
                        src_full, tgt_sub,
                        rim_centroid_after_T, rim_axis_after_T,
                        g_shapeMatchAxisSweepN,
                        results_A, &std::cout);
                }

                // ==== Variant B: rim-only sweep ========================
                std::vector<RimShape::AxisSweepResult> results_B;
                bool B_ok = false;
                if (g_shapeMatchAxisSweepCompare) {
                    std::vector<glm::vec3> src_rim;
                    if (liverMesh3D && !g_debugSourceRimChain.empty()) {
                        src_rim.reserve(g_debugSourceRimChain.size());
                        for (int idx : g_debugSourceRimChain) {
                            if (idx < 0) continue;
                            const size_t k = size_t(idx) * 3;
                            if (k + 2 < liverMesh3D->mVertices.size()) {
                                src_rim.emplace_back(
                                    liverMesh3D->mVertices[k],
                                    liverMesh3D->mVertices[k+1],
                                    liverMesh3D->mVertices[k+2]);
                            }
                        }
                    }
                    if (!src_rim.empty()
                        && !g_debugTargetBoundaryPoints.empty())
                    {
                        std::cout << "[Ctrl+Shift+W] === Variant B (rim-only"
                                  << " chamfer, src=" << src_rim.size()
                                  << " tgt=" << g_debugTargetBoundaryPoints.size()
                                  << ") ===" << std::endl;
                        B_ok = RimShape::runRimAxisRotationSweep(
                            src_rim, g_debugTargetBoundaryPoints,
                            rim_centroid_after_T, rim_axis_after_T,
                            g_shapeMatchAxisSweepN,
                            results_B, &std::cout);
                    } else {
                        std::cout << "[Ctrl+Shift+W] Variant B skipped:"
                                  << " src_rim=" << src_rim.size()
                                  << " tgt_boundary="
                                  << g_debugTargetBoundaryPoints.size()
                                  << std::endl;
                    }
                }

                const glm::mat4 rot_A = (A_ok && !results_A.empty())
                    ? results_A[0].axis_rotation_T : glm::mat4(1.0f);
                const glm::mat4 rot_B = (B_ok && !results_B.empty())
                    ? results_B[0].axis_rotation_T : glm::mat4(1.0f);

                // ==== RMSE 比較: 各 variant を適用 → 0-iter session で
                //     CompRMSE 計測 → 逆 rotation で元に戻す
                //     (poseSaveToLibrary は呼ばないので PoseLibrary には
                //      評価セッションのエントリは追加されない)
                auto measure_rmse = [&](const glm::mat4& R)->double {
                    NormalRefine::applyIncrementalTransform(
                        glm::dmat4(R), organs);
                    const int saved = g_normRefineMaxIter;
                    g_normRefineMaxIter = 0;
                    runNormalCompatRefineSession(
                        NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                    g_normRefineMaxIter = saved;
                    const double rmse = registrationHandle.compRmse;
                    // 初期値に戻す (Shape Match best T 適用後の状態)
                    NormalRefine::applyIncrementalTransform(
                        glm::dmat4(glm::inverse(R)), organs);
                    return rmse;
                };

                const double rmse_A = A_ok ? measure_rmse(rot_A)
                                            : std::numeric_limits<double>::infinity();
                const double rmse_B = B_ok ? measure_rmse(rot_B)
                                            : std::numeric_limits<double>::infinity();

                bool A_wins = (rmse_A <= rmse_B);
                if (!A_ok && !B_ok) {
                    std::cout << "[Ctrl+Shift+W] both variants failed,"
                              << " keeping baseline T" << std::endl;
                } else {
                    if (!A_ok) A_wins = false;
                    if (!B_ok) A_wins = true;
                    const glm::mat4 winner = A_wins ? rot_A : rot_B;
                    const float win_angle = A_wins
                        ? (results_A.empty() ? 0.0f : results_A[0].angle_deg)
                        : (results_B.empty() ? 0.0f : results_B[0].angle_deg);
                    const double win_cost = A_wins
                        ? (results_A.empty() ? 0.0 : results_A[0].cost)
                        : (results_B.empty() ? 0.0 : results_B[0].cost);

                    std::cout << "[Ctrl+Shift+W] === Variant comparison ==="
                              << "  A(full) RMSE="
                              << (A_ok ? std::to_string(rmse_A) : "N/A")
                              << "  B(rim) RMSE="
                              << (B_ok ? std::to_string(rmse_B) : "N/A")
                              << "  winner=" << (A_wins ? "A(full)" : "B(rim)")
                              << "  angle=" << win_angle << "deg"
                              << "  sweep_cost=" << win_cost
                              << std::endl;

                    g_debugShapeMatchAxisSweepT     = winner;
                    g_debugShapeMatchAxisSweepCost  = win_cost;
                    g_debugShapeMatchAxisSweepAngle = win_angle;

                    NormalRefine::applyIncrementalTransform(
                        glm::dmat4(winner), organs);
                }

                // (f) Final save: max_iter=0 で session 再実行 + 確定 save
                std::cout << "[Ctrl+Shift+W] saving final pose to PoseLibrary"
                          << " (max_iter=0 pseudo-session)..." << std::endl;
                const int saved_max_iter = g_normRefineMaxIter;
                g_normRefineMaxIter = 0;
                g_stepStartTime  = std::chrono::steady_clock::now();
                gUI.state.regMethod = 3;
                if (g_normRefineLiveMode) {
                    startNormalCompatRefineLive(
                        NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                } else {
                    runNormalCompatRefineSession(
                        NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                    poseSaveToLibrary(SaveCriterion::EITHER,
                                      g_activeQuadrantMask);
                }
                g_normRefineMaxIter = saved_max_iter;
                break;
            }

            // ---- Phase 7b Step 4a — fallback to original Live ICP -----
            //     (g_shapeMatchAxisSweepEnabled = false のとき)
            // 5. Bridge to Live ICP (Phase 6 既存配線)
            //    Phase 7b Step 4a: g_shapeMatchLiveMaxIter で「Shape Match
            //    を絶対維持に近づける」ため max_iter を一時的に override。
            //    prepareNormalRefineSession 内で nrp.maxTotalIterations =
            //    g_normRefineMaxIter としてキャプチャされるので、
            //    start 後すぐ復元しても session には override 値が残る。
            const int saved_max_iter = g_normRefineMaxIter;
            if (g_shapeMatchLiveMaxIter > 0) {
                g_normRefineMaxIter = g_shapeMatchLiveMaxIter;
                std::cout << "[Ctrl+Shift+W] Live ICP max_iter override:"
                          << " " << saved_max_iter << " -> "
                          << g_normRefineMaxIter
                          << " (Phase 7b absolute-maintenance mode)"
                          << std::endl;
            }
            g_stepStartTime  = std::chrono::steady_clock::now();
            gUI.state.regMethod = 3;
            if (g_normRefineLiveMode) {
                std::cout << "[Ctrl+Shift+W] bridging to NormalCompat LIVE"
                          << " (frame-driven)..." << std::endl;
                startNormalCompatRefineLive(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                // main loop ticks until finish; poseSaveToLibrary fires
                // via pendingSave (existing Phase 6 wiring).
            } else {
                std::cout << "[Ctrl+Shift+W] bridging to NormalCompat"
                          << " BLOCKING (one-shot)..." << std::endl;
                runNormalCompatRefineSession(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
            }
            // Restore: session already captured the override value.
            // Subsequent Shift+N etc. use the user's normal default.
            g_normRefineMaxIter = saved_max_iter;
            break;
        }
        if ((mods & GLFW_MOD_CONTROL) && !(mods & GLFW_MOD_SHIFT)) {
            // ==== Ctrl+W : Shape Match → mesh apply → PoseLibrary save ====
            //   旧 Step 3 (preview のみ) を廃止し、mesh を実際に Shape
            //   Match best T で動かして PoseLibrary に直接保存する。
            //   ユーザ運用想定:
            //     Apply Init Pose → Ctrl+W → (ユーザが手動で) Ctrl+G
            //   Ctrl+W で Shape Match による良い初期姿勢を確定し、
            //   その状態から Ctrl+G の V3-R BIPOP-CMA-ES を走らせる流れ。
            //   ICP / axis sweep は Ctrl+Shift+W (Step 4b) が担当。
            //
            //   Stop semantic: Live が active なら停止
            if (NormalRefineLive::active) {
                std::cout << "[Ctrl+W] stopping active Live session"
                          << std::endl;
                finishNormalCompatRefineLive();
                break;
            }
            if (gApp.mode != AppMode::kRegistration) {
                std::cout << "[Ctrl+W] requires a loaded scene"
                          << " (Registration mode)" << std::endl;
                break;
            }
            // Auto-trigger labels (same as Plain W / Ctrl+Shift+W)
            if (!g_liverRegion.valid()) {
                std::cout << "[Ctrl+W] g_liverRegion not yet computed,"
                          << " auto-running recomputeLiverRegion()..."
                          << std::endl;
                recomputeLiverRegion();
            }
            if (!g_liverLR.valid()) {
                std::cout << "[Ctrl+W] g_liverLR not yet computed,"
                          << " auto-running recomputeLiverLR()..."
                          << std::endl;
                recomputeLiverLR();
            }
            if (g_ctrlgUseCaudalOnly && !g_liverCC.valid()) {
                std::cout << "[Ctrl+W] caudal-only ON but g_liverCC not"
                          << " yet computed, auto-running..." << std::endl;
                recomputeLiverCC();
            }
            // 1. Run coarse search → best T
            std::cout << "[Ctrl+W] running Shape Match coarse search..."
                      << std::endl;
            if (!runDebugShapeMatchCoarse()) {
                std::cout << "[Ctrl+W] coarse search failed — abort,"
                          << " mesh unchanged" << std::endl;
                break;
            }
            std::cout << "[Ctrl+W] best T: cost="
                      << g_debugShapeMatchBestCost
                      << "  k=" << g_debugShapeMatchBestK
                      << std::endl;

            // 2. Undo snapshot (for Ctrl+Z safety)
            poseAutoSaveBeforeRegistration();

            // 3. Apply best T to all organ meshes
            {
                auto organs = getOrganList();
                int n_valid = 0;
                for (auto* m : organs) if (m) n_valid++;
                std::cout << "[Ctrl+W] applying best T to "
                          << n_valid << " organ meshes..." << std::endl;
                NormalRefine::applyIncrementalTransform(
                    glm::dmat4(g_debugShapeMatchBestTransform), organs);
            }

            // 4. Clear any leftover preview state (from older code path)
            g_showDebugShapeMatch = false;
            g_debugShapeMatchBestSrc.clear();

            // 5. Save to PoseLibrary via max_iter=0 pseudo-session
            //    (mesh の現在位置で metrics 計算 + PoseLibrary に save)
            std::cout << "[Ctrl+W] saving pose to PoseLibrary"
                      << " (max_iter=0 pseudo-session, no ICP)..."
                      << std::endl;
            const int saved_max_iter = g_normRefineMaxIter;
            g_normRefineMaxIter = 0;
            g_stepStartTime  = std::chrono::steady_clock::now();
            gUI.state.regMethod = 3;
            if (g_normRefineLiveMode) {
                startNormalCompatRefineLive(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                // main loop ticks → 0 iters → finish → save via pendingSave
            } else {
                runNormalCompatRefineSession(
                    NormalRefine::NORMAL_COMPAT, g_activeQuadrantMask);
                poseSaveToLibrary(SaveCriterion::EITHER,
                                  g_activeQuadrantMask);
            }
            g_normRefineMaxIter = saved_max_iter;
            break;
        }
        // [key-reorg Phase 4] Shift+W (target boundary overlay) and plain W
        // (source rim chain overlay) viz toggles moved to Ctrl+D > Viz tab
        // ("Debug Target Boundary" / "Debug Source Rim Chain"). The W-family
        // ACTION shortcuts (Ctrl+W, Ctrl+Shift+W, Alt+W, Ctrl+Alt+W) are kept
        // above.
        break;
    case GLFW_KEY_P:
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_ALT)) {
            // ---- Alt+Ctrl+P : AutoQuadCyclic-RANSAC (Auto Orient Sweep) ----
            //   Apply Init Pose で確定した QUADRANT を固定したまま、ORIENT
            //   (preset) を 9 個試行し、最も compRmse が良いものを採用して
            //   PoseLibrary に 1 件だけ追加する。
            //
            //   9 個の preset は QUADRANT の左右成分で選ばれる:
            //     右側のみ → Right 中心 9 (画面左 3 列、FAR 含)
            //     左側のみ → Left  中心 9 (画面右 3 列、FAR 含)
            //     両側     → Base  中心 9 (画面中央 3 列)
            //
            //   loop 中は PoseLibrary に一切触らず、Session reject 機構の
            //   介入もないため、連打しても Library が荒れない。
            //
            //   注意: mods 判定順序は Alt+Ctrl > Shift+Ctrl > Ctrl > Shift > 単独。
            //         Alt+Ctrl の組合せが先に拾われるよう必ずこの位置 (一番先)
            //         に置くこと。
            //
            //   想定ワークフロー:
            //     UI で preset/quadrant 設定 → Apply Init Pose
            //       ↓
            //     Alt+Ctrl+P  (この関数。9 ORIENT × 1 QUAD = 9 trial)
            //       ↓
            //     Ctrl+G      (V3R refinement)
            //
            //   lock_scale: UI チェックボックス (Hemi Auto の下の AutoQCR 行の
            //               "6-DoF" チェック) の値を見る。チェック ON で rigid。
            runAutoQuadCyclicRansac(gUI.state.autoQcrLockScale);
        } else if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT)) {
            // ---- Shift+Ctrl+P : QuadCyclic-RANSAC --------------------------
            //   Ctrl+P と同じ AR ∩ silh ∩ quad 前処理を使うが、matching を
            //   「24 sector 全部対 48 パターン cyclic shift」ではなく
            //   「K=3 subset RANSAC + 2 段階評価 (内部 RMSE → chamfer)」に
            //   置き換える。3 sector だけ使うので一部の sector が rim から
            //   逸れていても inlier 3 つで初期姿勢を確保できる。
            //   ハイパラは v1 ハードコード (g_qcrSubsetK / g_qcrMinSpreadSec
            //   / g_qcrTopKCandidates)、将来 UI 露出予定。
            //   注意: mods の判定順は Alt+Ctrl > Shift+Ctrl > Ctrl > Shift > 単独で、
            //   Shift+Ctrl の組合せが Ctrl 単独より先に拾われるよう必ずこの位置
            //   (Ctrl 単独より先) に置くこと。
            if (!g_liverRegion.valid()) {
                std::cout << "[Shift+Ctrl+P] LiverRegion (Shift+R) not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverRegion();
            }
            if (!g_liverLR.valid()) {
                std::cout << "[Shift+Ctrl+P] LiverLR (Y) not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverLR();
            }
            gUI.state.regMethod = 1;
            g_stepStartTime  = std::chrono::steady_clock::now();
            g_sessionBipopN  = 0;
            poseAutoSaveBeforeRegistration();
            runQuadCyclicRansac();
            poseSaveToLibrary(SaveCriterion::RMSE);
        } else if (mods & GLFW_MOD_CONTROL) {
            // ---- Ctrl+P : QuadCyclic ---------------------------------------
            //   Shift+P と同じ cyclic boundary + Umeyama + ICP のパイプラインを
            //   使うが、source を AR 固定視点 ∩ silhouette ∩ g_activeQuadrantMask
            //   の三段交差集合で絞り込む。Initial Orientation で選んだ象限の rim
            //   だけが target 境界に当てに行く動作。
            //   Region/LR labels が未計算なら auto-trigger
            //   (applyInitRotation / Shift+O と同じ流儀)。
            if (!g_liverRegion.valid()) {
                std::cout << "[Ctrl+P] LiverRegion (Shift+R) not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverRegion();
            }
            if (!g_liverLR.valid()) {
                std::cout << "[Ctrl+P] LiverLR (Y) not yet computed,"
                          << " auto-running..." << std::endl;
                recomputeLiverLR();
            }
            gUI.state.regMethod = 1;   // SilhouetteHemi と同じ表示扱い
            g_stepStartTime  = std::chrono::steady_clock::now();
            g_sessionBipopN  = 0;
            poseAutoSaveBeforeRegistration();
            runQuadCyclic();
            poseSaveToLibrary(SaveCriterion::RMSE);
        } else if ((mods & GLFW_MOD_ALT) && !(mods & GLFW_MOD_CONTROL)
                                         && !(mods & GLFW_MOD_SHIFT)) {
            // [key-reorg Phase 9] Alt+P : Silhouette Align (was Shift+E).
            // byte-identical to the old GLFW_KEY_E Shift branch. Placed after
            // the Ctrl* branches and before Shift+P / Plain P (mod precedence).
            // [QUAD-SIL] Now quadrant-aware: mask logging + label guard via
            // prepareQuadrantForSilAlign (Ctrl+I parity), mask handed to
            // runShiftE (triangle filter) and poseSaveToLibrary (entry
            // metadata). At QUAD_ALL this is byte-identical to before.
            std::cout << "[Alt+P] dispatching Silhouette Align..." << std::endl;
            if (!prepareQuadrantForSilAlign("Alt+P")) break;
            g_stepStartTime = std::chrono::steady_clock::now();
            poseAutoSaveBeforeRegistration();
            runShiftE(g_activeQuadrantMask);
            g_sessionSilhouetteN++;
            gUI.state.regMethod = 5;
            saveSilAlignToLibrary();   // [QUAD-SIL ACCEPT] Ctrl+I-parity gate
        } else if (mods & GLFW_MOD_SHIFT) {
            // Shift+P: Cyclic Boundary Registration
            //  Key P と同じ前処理 (source silhouette + target boundary) を使うが、
            //  FGR/FPFH の代わりに重心まわり N=24 セクターで巡回シフト × ミラー
            //  反転 (合計 2N=48 パターン) を試し、全 source × 全 target の
            //  chamfer RMSE が最小の T を初期姿勢として ICP で精錬する。
            //  動機: silhouette/boundary の点数が少ない場面では FPFH 識別性が
            //  不十分で FGR が tuple test 0 → identity に落ちるため、
            //  特徴量を介さない幾何ベースの初期化を試す。
            gUI.state.regMethod = 1;   // SilhouetteHemi と同じ表示扱い
            g_stepStartTime  = std::chrono::steady_clock::now();
            g_sessionBipopN  = 0;
            poseAutoSaveBeforeRegistration();
            runCyclicBoundaryReg();
            poseSaveToLibrary(SaveCriterion::RMSE);
        } else {
            // P: SilhouetteHemi (既存) — HemiAuto と並ぶ独立の登録ボタン
            //                            (Source/Target 双方境界優先)
            gUI.state.regMethod = 1;   // HemiAuto と同じ表示扱い
            g_stepStartTime  = std::chrono::steady_clock::now();
            g_sessionBipopN  = 0;
            poseAutoSaveBeforeRegistration();
            runSilhouetteHemi();
            poseSaveToLibrary(SaveCriterion::RMSE);
        }
        break;
    // [key-reorg Phase 9] GLFW_KEY_E removed: Shift+E (Silhouette Align) -> Alt+P.
    // [key-reorg Phase 4] GLFW_KEY_I removed: Shift+I (IoU debug dump) moved to
    //   Ctrl+D > Viz tab "Dump IoU debug PNG" button.
    case GLFW_KEY_Q:
        // Pose Library ウィンドウ開閉
        g_poseLibrary.showWindow = !g_poseLibrary.showWindow;
        std::cout << "[PoseLibrary] Window "
                  << (g_poseLibrary.showWindow ? "ON" : "OFF") << std::endl;
        break;
    case GLFW_KEY_X:
        // 元コード line 7039-7042 通り: REGISTRATION_MODE && hasLastRegistration
        if (gApp.mode == AppMode::kRegistration && g_poseLibrary.hasLastRegistration) {
            poseUndo();
        }
        break;
    // [key-reorg Phase 11] GLFW_KEY_F2 removed: camera reset via sidebar
    //   "Cam Init" button (onResetCamera lambda is equivalent, incl. the
    //   Registration-mode 180-deg Y rotation + currentTarget=TARGET_TEXTURE).
    case GLFW_KEY_A:
        gApp.arMode = !gApp.arMode;
        std::cout << "[AR] background overlay: "
                  << (gApp.arMode ? "ON" : "OFF") << std::endl;
        break;
    case GLFW_KEY_D:
        if (gApp.mode == AppMode::kRegistration && g_pShader) {
            std::vector<mCutMesh*> organs = {
                liverMesh3D, portalMesh3D, veinMesh3D,
                tumorMesh3D, segmentMesh3D, gbMesh3D
            };
            int imgW = OrbitCam.calibWidth  > 0 ? OrbitCam.calibWidth  : 1280;
            int imgH = OrbitCam.calibHeight > 0 ? OrbitCam.calibHeight : 720;
            ARSave::capture(g_arSave, OrbitCam, *g_pShader, *g_pShaderCube,
                            gApp.arBg, organs, g_meshAlpha, objPos,
                            imgW, imgH,
                            DEPTH_OUTPUT_PATH, gWindowWidth, gWindowHeight);
        } else {
            std::cout << "[D] AR save requires Registration mode" << std::endl;
        }
        break;

    // [key-reorg Phase 11] GLFW_KEY_F9 / GLFW_KEY_F10 removed:
    //   F9 (silhouette IoU overlay window) -> Ctrl+D > Viz tab checkbox
    //     "Show Silhouette Overlay window".
    //   F10 (vertex-squash diagnose) -> Ctrl+D > Viz tab button (added Phase 1).
    case GLFW_KEY_COMMA: {
        const float step = (mods & GLFW_MOD_SHIFT) ? 0.05f : 0.01f;
        g_silhouetteCosThreshold = std::max(0.0f,
                                            g_silhouetteCosThreshold - step);
        rebuildOBJWithCurrentThreshold();
        break;
    }
    case GLFW_KEY_PERIOD: {
        const float step = (mods & GLFW_MOD_SHIFT) ? 0.05f : 0.01f;
        g_silhouetteCosThreshold = std::min(0.99f,
                                            g_silhouetteCosThreshold + step);
        rebuildOBJWithCurrentThreshold();
        break;
    }
    // [key-reorg Phase 10] GLFW_KEY_R removed: Run depth via sidebar "Run Depth" button.
    // [key-reorg Phase 4] GLFW_KEY_T / GLFW_KEY_Y / GLFW_KEY_H removed:
    //   Shift+T (region recompute), Shift+Y (LR recompute), Y (LR viz),
    //   Shift+H (CC viz), H (4-quadrant viz) all moved to Ctrl+D > Viz tab
    //   (checkboxes with auto-recompute + "Recompute Region/LR" buttons).
    case GLFW_KEY_U:
        if (gApp.mode == AppMode::kImageOnly) {
            MaskPicker::undo(gApp);                       // 既存動作 (byte-identical)
        } else if (gApp.mode == AppMode::kRegistration &&
                   (mods & GLFW_MOD_SHIFT) && !(mods & GLFW_MOD_CONTROL)) {
            // Phase U-1: soft partition 計算 + ヒートマップ toggle
            if (!g_softAnchors.valid()) {
                std::cerr << "[Shift+U] Need >=" << SoftPartition::MIN_GROUPS
                          << " Umeyama anchors first (have "
                          << g_softAnchors.count() << ")" << std::endl;
            } else {
                if (!g_softPartReady) runSoftPartitionCompute();
                if (g_softPartReady) {
                    g_softShowHeatmap = !g_softShowHeatmap;
                    std::cout << "[Shift+U] heatmap "
                              << (g_softShowHeatmap ? "ON" : "OFF")
                              << "  groups=" << g_softAnchors.count() << std::endl;
                }
            }
        } else if (gApp.mode == AppMode::kRegistration &&
                   (mods & GLFW_MOD_CONTROL) && !(mods & GLFW_MOD_SHIFT)) {
            // Ctrl+U: run the selected registration method (both start from
            // the same Umeyama init). Phase U-CPD added the CPD branch.
            if (g_regMethod == METHOD_CPD) {
                // Phase U-CPD: pure rigid CPD (anchor-free). LIVE = animate
                //   stages 1-4 then replay; the render loop saves on finish.
                if (g_regLiveMode) {
                    poseAutoSaveBeforeRegistration();
                    startCpdLive();
                } else {
                    poseAutoSaveBeforeRegistration();
                    runCpdRegistration();          // stages 1->5 (blocking)
                    poseSaveToLibrary(SaveCriterion::RMSE);
                    if (!g_cpdCanRevert) g_cpdStageDone[5] = false; // session reverted it
                }
            } else if (!g_softPartReady) {
                std::cerr << "[Ctrl+U] run Shift+U (soft partition) first"
                          << std::endl;
            } else {
                // Phase U-2: soft-weighted ICP (LIVE animates the convergence).
                if (g_regLiveMode) {
                    poseAutoSaveBeforeRegistration();
                    startSoftIcpLive();
                } else {
                    poseAutoSaveBeforeRegistration();
                    runSoftWeightedICP();
                    poseSaveToLibrary(SaveCriterion::RMSE);
                }
            }
        }
        break;
    case GLFW_KEY_C:
        if (gApp.mode == AppMode::kImageOnly) MaskPicker::clear(gApp);
        break;
    case GLFW_KEY_UP:
        if (gApp.mode == AppMode::kRegistration) {
            g_voxelSize += 0.05f;
            std::cout << "[VoxelSize] " << g_voxelSize << std::endl;
        }
        break;
    case GLFW_KEY_DOWN:
        if (gApp.mode == AppMode::kRegistration) {
            g_voxelSize = std::max(0.0f, g_voxelSize - 0.05f);
            std::cout << "[VoxelSize] " << g_voxelSize << std::endl;
        }
        break;
    // [key-reorg Phase 10] GLFW_KEY_K removed: camera depth via sidebar
    //   "Run Depth" button (camera mode uses the same button).
    // [key-reorg Phase 12] GLFW_KEY_J removed: camera_frame_temp.jpg is written
    //   automatically by "Run Depth" (camera mode); standalone save not needed.
    // [key-reorg Phase 10] GLFW_KEY_S / GLFW_KEY_L removed: snapshot / live-view
    //   handled by the sidebar camera toggle ("Capture" / "Re-Capture").
    // [key-reorg Phase 12] GLFW_KEY_M removed: OBJ/STL export moved to the
    //   sidebar Export buttons (StlExport::exportRegisteredObjs /
    //   exportCamMmStlWithSnapshot, wired via onExportStl / onExportStlFlipped).
    }
}

static void glfw_OnFramebufferSize(GLFWwindow*, int w, int h) {
    gWindowWidth  = w;
    gWindowHeight = h;
    gApp.windowW  = w;
    gApp.windowH  = h;
    glViewport(0, 0, w, h);
    OrbitCam.onWindowResize(w, h);
}

// Load an image, rectifying if the active K has distortion, and keep the
// downstream intrinsics consistent with the rectified pixels.
//
//   - Rectification is decided/performed with the FULL (distorted) calibration.
//     g_intrinsics may have had its distortion zeroed by a previous rectify, so
//     restore it from g_intrinsics_raw first (when valid) for the decision.
//   - If rectification ran, depth_output/original_rectified.jpg is now the live
//     image and is already undistorted. We stash the distorted K in
//     g_intrinsics_raw and ZERO the distortion in g_intrinsics so everything
//     downstream (SAM2 unprojection, OBJTargetExtraction, OrbitCam) uses a K
//     that matches the rectified pixels. This also makes a second pass a no-op
//     (hasDistortion() == false), preventing double rectification.
//   - If no rectification ran, there is no raw/effective split to track.
// Sidecar import: if an intrinsics_custom.txt sits in the SAME directory as the
// image being loaded, copy it into depth_output/intrinsics_custom.txt and make
// Custom the active source. This lets per-image (per-dataset) calibration files
// take effect on load without a manual copy. No-op if absent, or if the sidecar
// already IS depth_output/intrinsics_custom.txt.
static void maybeImportSidecarIntrinsics(const std::string& imagePath) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path sidecar = fs::path(imagePath).parent_path() / "intrinsics_custom.txt";
    if (!fs::exists(sidecar, ec)) return;

    const std::string dst = DEPTH_OUTPUT_PATH + "intrinsics_custom.txt";
    fs::path srcAbs = fs::weakly_canonical(sidecar, ec);
    fs::path dstAbs = fs::weakly_canonical(fs::path(dst), ec);
    if (!srcAbs.empty() && srcAbs == dstAbs) return;   // would copy onto itself

    fs::create_directories(DEPTH_OUTPUT_PATH, ec);
    fs::copy_file(sidecar, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "[Sidecar] copy failed (" << sidecar << " -> " << dst
                  << "): " << ec.message() << std::endl;
        return;
    }
    std::cout << "[Sidecar] imported " << sidecar << " -> " << dst
              << "; selecting Custom source" << std::endl;
    g_intrinsicsSource = IntrinsicsSource::Custom;
    g_intrinsics_raw   = Reg3DCustom::CameraIntrinsics{};  // force a fresh decision below
    loadIntrinsicsFromCurrentSource();                     // load the imported 4K K
}

// Drag-and-drop import of a camera-intrinsics (内部パラメータ) file: install it as
// the Custom source and apply immediately. This is the drop-driven sibling of
// maybeImportSidecarIntrinsics() above -- instead of auto-detecting an
// intrinsics_custom.txt sitting next to the image being loaded, it is triggered
// by the file itself being dropped onto the window. Reuses the exact same
// loader / destination / source enum as the rest of the intrinsics system.
//
// Returns true iff the path was recognised as a valid intrinsics file (and thus
// consumed). A non-.txt, or a .txt that does not parse to a valid K, returns
// false so handleFileDrop() can fall through to its other routes / rejection.
static bool maybeImportDroppedIntrinsics(const std::string& path) {
    namespace fs = std::filesystem;

    // 拡張子チェック: 正準フォーマットは .txt。
    // (画像 / .obj / .bin は別ルートで処理されるので奪わない)
    size_t dot = path.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? "" : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    if (ext != ".txt") return false;

    // 中身チェック: アプリ本体と同じローダーで「有効な K」として読めるものだけ採用。
    // ただのメモ .txt はここで弾かれ、下のフォールスルー/拒否に回る。
    Reg3DCustom::CameraIntrinsics K;
    if (!Reg3DCustom::loadCameraIntrinsics(path, K) || !K.valid()) return false;

    std::error_code ec;
    const std::string dst = DEPTH_OUTPUT_PATH + "intrinsics_custom.txt";
    fs::path srcAbs = fs::weakly_canonical(fs::path(path), ec);
    fs::path dstAbs = fs::weakly_canonical(fs::path(dst), ec);

    if (srcAbs.empty() || srcAbs != dstAbs) {        // 自分自身へのコピーは回避
        fs::create_directories(DEPTH_OUTPUT_PATH, ec);
        fs::copy_file(path, dst, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            std::cerr << "[FileDrop/Intrinsics] copy failed ("
                      << path << " -> " << dst << "): " << ec.message() << std::endl;
            return true;   // 内部パラメータとは認識した(=画像扱いさせない)。設置のみ失敗
        }
    }

    g_intrinsicsSource = IntrinsicsSource::Custom;
    g_intrinsics_raw   = Reg3DCustom::CameraIntrinsics{};   // rectify-raw を破棄
    loadIntrinsicsFromCurrentSource();                      // 即時適用 (g_intrinsics / OrbitCam)

    std::cout << "[FileDrop/Intrinsics] imported " << path << " -> " << dst
              << "  (Custom 選択)  K " << K.width << "x" << K.height
              << " fx=" << K.fx << " fy=" << K.fy
              << " cx=" << K.cx << " cy=" << K.cy << std::endl;
    return true;
}

static bool loadImageRectifyAware(AppContext& ctx, const std::string& path) {
    maybeImportSidecarIntrinsics(path);
    if (g_intrinsics_raw.valid()) g_intrinsics = g_intrinsics_raw;
    bool rectified = false;
    if (!ImageSession::loadWithIntrinsics(ctx, path, g_intrinsics, &rectified))
        return false;
    if (rectified) {
        g_intrinsics_raw = g_intrinsics;                 // keep distorted calibration
        g_intrinsics.k1 = g_intrinsics.k2 = g_intrinsics.k3 = g_intrinsics.k4 = 0.0f;
        g_intrinsics.p1 = g_intrinsics.p2 = 0.0f;        // rectified -> no distortion
        gApp.intrinsics = g_intrinsics;
        std::cout << "[Intrinsics] image rectified -> downstream K distortion "
                     "zeroed (distorted K kept in g_intrinsics_raw)" << std::endl;
    } else {
        g_intrinsics_raw = Reg3DCustom::CameraIntrinsics{};  // no active rectify
    }
    return true;
}

// =========================================================================
//  臓器セットのドラッグ&ドロップ（CTモデル差し替え）
//  -----------------------------------------------------------------------
//  フォルダ（標準）または複数 .obj をまとめてドロップして、現在の臓器セット
//  (liver/portal/vein/tumor/segment/gb) を差し替え、既存の深度ターゲットへ
//  再 prealign する。ルール:
//    - liver.obj は必須。無いセットは拒否（門脈単独などもこれで弾ける）。
//    - ドロップに含まれない臓器は空メッシュにクリア＝完全置換。よって
//      liver.obj だけ落とせば肝臓だけが残る。
//    - 深度ターゲット(screenMesh)が既にロード済みであること。無い場合は
//      整列できない(生CT座標で画面外)ため、エラーで拒否する。
//    - ターゲットは据え置きなので g_sceneDiag / voxel は変わらない。再計算
//      されるのは prealign 変換と肝臓の幾何ラベル(Region/LR/CC)のみ。
//  照合するcanonicalファイル名(大文字小文字無視):
//    liver.obj portal.obj vein.obj tumor.obj segment.obj gb.obj
// =========================================================================
static const char* kOrganCanonical[6] = {
    "liver.obj", "portal.obj", "vein.obj",
    "tumor.obj", "segment.obj", "gb.obj"
};

struct OrganDropScan {
    bool                      anyOrgan = false;  // canonical臓器objを1つ以上検出
    std::array<std::string,6> path{};            // path[role]; 空=不在
    bool preserveRegisteredPose = false;         // [deform round-trip] "deformed"
                                                 // フォルダ由来なら prealign を
                                                 // スキップし REG#1 姿勢を保つ
    bool hasLiver() const { return !path[0].empty(); }
};

// ドロップされたパス群(ファイル/フォルダ混在可)を走査し role->path に解決
static OrganDropScan scanOrganDrop(int count, const char** paths) {
    OrganDropScan scan;
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return (char)std::tolower(c); });
        return s;
    };
    // [deform round-trip] このフォルダ名(大文字小文字無視)から来た organ は、
    //   既に REG#1 の登録座標系にある変形モデルとみなし prealign をスキップする。
    auto isDeformedDir = [&](const std::filesystem::path& dir) {
        return lower(dir.filename().string()) == "deformed";
    };
    std::error_code ec;
    for (int i = 0; i < count; ++i) {
        const std::string p = paths[i];
        if (std::filesystem::is_directory(p, ec)) {
            // フォルダドロップ（標準）: 中の各canonical名を探す
            bool foundHere = false;
            for (int r = 0; r < 6; ++r) {
                std::string cand =
                    (std::filesystem::path(p) / kOrganCanonical[r]).string();
                if (std::filesystem::exists(cand, ec)) {
                    scan.path[r]  = cand;
                    scan.anyOrgan = true;
                    foundHere     = true;
                }
            }
            if (foundHere && isDeformedDir(std::filesystem::path(p)))
                scan.preserveRegisteredPose = true;
        } else {
            // 個別ファイル: basename を canonical名と照合
            const std::string fn =
                lower(std::filesystem::path(p).filename().string());
            for (int r = 0; r < 6; ++r) {
                if (fn == kOrganCanonical[r]) {
                    scan.path[r]  = p;   // 重複は後勝ち
                    scan.anyOrgan = true;
                    if (isDeformedDir(std::filesystem::path(p).parent_path()))
                        scan.preserveRegisteredPose = true;
                    break;
                }
            }
        }
    }
    return scan;
}
// 臓器セットを差し替え、既存ターゲットへ再 prealign する。
// ターゲット未ロードなら false（ログ出力）。
static bool swapOrganSetFromDrop(const std::array<std::string,6>& srcPaths,
                                 bool preserveRegisteredPose = false) {
    if (!screenMesh || screenMesh->mVertices.empty()) {
        std::cerr << "[OrganDrop] REFUSED: no depth target loaded. "
                     "先に画像ロード / Run Depth(または bin から再構成)してから、"
                     "臓器セットをドロップしてターゲットへ登録してください。"
                  << std::endl;
        return false;
    }

    struct Spec { mCutMesh** ptr; int role; glm::vec3 defColor; };
    Spec specs[6] = {
        { &liverMesh3D,   0, glm::vec3(0.8f, 0.2f, 0.2f) },
        { &portalMesh3D,  1, glm::vec3(0.2f, 0.2f, 0.8f) },
        { &veinMesh3D,    2, glm::vec3(0.2f, 0.5f, 0.5f) },
        { &tumorMesh3D,   3, glm::vec3(0.8f, 0.5f, 0.5f) },
        { &segmentMesh3D, 4, glm::vec3(0.2f, 0.8f, 0.5f) },
        { &gbMesh3D,      5, glm::vec3(0.2f, 0.8f, 0.2f) },
    };

    // 1. メッシュ置換: ドロップにあればロード、無ければ空にクリア（完全置換）
    int loaded = 0;
    for (auto& s : specs) {
        glm::vec3 keepColor = (*s.ptr) ? (*s.ptr)->mColor : s.defColor;
        delete *s.ptr;
        const std::string& src = srcPaths[s.role];
        if (!src.empty()) {
            *s.ptr = new mCutMesh(mCutMesh().loadMeshFromFile(src.c_str()));
            if (!(*s.ptr)->mVertices.empty()) {
                ++loaded;
                std::cout << "[OrganDrop]   " << kOrganCanonical[s.role]
                          << "  V:" << (*s.ptr)->mVertices.size()/3
                          << "  <- " << src << std::endl;
            } else {
                std::cerr << "[OrganDrop]   " << kOrganCanonical[s.role]
                          << "  load FAILED (" << src << ") -> empty" << std::endl;
            }
        } else {
            *s.ptr = new mCutMesh();   // セットに無い → クリア
            std::cout << "[OrganDrop]   " << kOrganCanonical[s.role]
                      << "  cleared (not in dropped set)" << std::endl;
        }
        (*s.ptr)->mColor = keepColor;
        setUp(**s.ptr);
    }

    // 1b. Shift+M / Region ラベルの物理スケール基準を新しい raw メッシュから取り直す。
    //     [deform round-trip] preserve-pose のドロップは登録スケール(~0.7m)であって
    //     CT-mm ではないため、ここを更新すると Region リム幅換算と Shift+M が壊れる。
    //     同セッションなら起動時の値が正しいので据え置く。新規CTドロップ時のみ更新し、
    //     固定 reg->mm スケールもリセットして次の Deform >> で再捕捉させる。
    if (!preserveRegisteredPose) {
        auto bboxDiag = [](const mCutMesh* m) -> float {
            if (!m || m->mVertices.size() < 3) return 0.0f;
            glm::vec3 mn(m->mVertices[0], m->mVertices[1], m->mVertices[2]), mx = mn;
            for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
                glm::vec3 v(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2]);
                mn = glm::min(mn, v); mx = glm::max(mx, v);
            }
            return glm::length(mx - mn);
        };
        g_originalLiverDiagMm = bboxDiag(liverMesh3D);
        g_originalTumorDiagMm = bboxDiag(tumorMesh3D);
        g_hasOriginalDiags    = (g_originalLiverDiagMm > 1e-6f);
        g_hasRegToMmScale     = false;   // 新症例 → reg->mm を次の Deform >> で再捕捉
        // [CT-handoff] この liver が由来する raw-CT .obj を記録。以後 Shift+M /
        //   Reg->Deform の Umeyama は起動時 model/liver.obj ではなくこれと突き合わせる
        //   (頂点数の違うモデルをドロップしても mismatch で skip されなくなる)。
        g_liverCtSourcePath   = srcPaths[0];  // role 0 = liver (空ならフォールバック)
        std::cout << "[OrganDrop] CT-mm ref refreshed: liver=" << g_originalLiverDiagMm
                  << " tumor=" << g_originalTumorDiagMm
                  << "  (Region rim scale + Shift+M; reg->mm scale reset)" << std::endl;
        std::cout << "[OrganDrop] CT-handoff source set: " << g_liverCtSourcePath << std::endl;
    } else {
        // preserve-pose: ドロップは登録座標系のモデルで raw CT ではないため、
        // CT-handoff source は更新しない(直前の新規CTドロップ/起動時の値が正しい)。
        std::cout << "[OrganDrop] preserve-pose: CT-mm 基準 / reg->mm スケールを据え置き"
                     "(dropは登録座標系のモデル)。" << std::endl;
    }

    // 2. AppContext / allMeshes の再ポイント（上で delete したため必須）
    gApp.liver = liverMesh3D; gApp.portal = portalMesh3D; gApp.vein = veinMesh3D;
    gApp.tumor = tumorMesh3D; gApp.segment = segmentMesh3D; gApp.gb = gbMesh3D;
    allMeshes = { liverMesh3D, portalMesh3D, veinMesh3D,
                  tumorMesh3D, segmentMesh3D, gbMesh3D };
    gApp.all = allMeshes;

    // 3. 新規はCT空間そのまま: 直前の変換とラベルを破棄。
    g_lastOrganTransform    = glm::mat4(1.0f);
    g_hasLastOrganTransform = false;
    g_lastOrganOffset       = glm::vec3(0.0f);
    g_liverRegion = LiverRegionLabel::Result();
    g_liverLR     = LiverLeftRightLabel::Result();
    g_liverCC     = LiverCranioCaudalLabel::Result();
    // [bugfix] Also drop the marker viz caches — they index the OLD liver
    // mesh, so a swapped-in model with a different vertex count would draw
    // the 4-quadrant / region / LR / CC markers at stale positions until
    // the next recompute. (Label Results above are cleared; this matches.)
    clearLiverLabelVizCaches();
    //  ★ターゲット雲キャッシュは絶対にクリアしない★
    //  ターゲット雲(=深度 screenMesh 由来の ~28万点 + boundaryDist/normals)は臓器とは
    //  独立で、臓器スワップでは一切変化しない。ここで clearCachedTargetCloud() すると
    //  注入済みフル解像度ターゲットが消え、Ctrl+G が front-face フォールバック(~1267点)に
    //  落ちて全くフィットしなくなる(レジストレーションが合わない症状の原因)。
    //  setupObjScene / rebuild はクリア直後に setupOBJTarget で必ず再構築するが、スワップは
    //  screenMesh を作り直さないので、クリアせずキャッシュを温存するのが正しい。

    // 4. prealign。
    //    [deform round-trip] preserve-pose のときはスキップ。既に REG#1 の登録
    //    座標系にあるので、2回目の register は「ゼロから当てはめ」ではなく REG#1
    //    を出発点とした局所 refine になり、座標が暴れない。
    std::vector<mCutMesh*> organs = { liverMesh3D, portalMesh3D, veinMesh3D,
                                      tumorMesh3D, segmentMesh3D, gbMesh3D };
    if (preserveRegisteredPose) {
        g_lastOrganTransform    = glm::mat4(1.0f);
        g_hasLastOrganTransform = false;
        std::cout << "[OrganDrop] preserve-pose: prealign SKIP "
                     "(model は REG#1 登録座標系のまま)。" << std::endl;
    } else {
        g_lastOrganTransform    = Reg3DCustom::prealignSourceToTarget(organs, *screenMesh);
        g_hasLastOrganTransform = true;
    }
    for (auto* m : organs) if (m) setUp(*m);

    // 4b. 初期姿勢スナップショットを撮り直す（重要）。
    //     Apply Init Pose / Reset は g_initOrganVertices から頂点を復元するため、
    //     撮り直さないと旧トポロジ(頂点数)の snapshot を新メッシュに上書きし、
    //     "Index out of range" / "region.labels size mismatch" を起こす。
    snapshotInitialPose();

    // 5. 差し替えが見えるように: 現在ロード中の臓器が1つも表示されていなければ
    //    肝臓(必ず存在)を表示する。
    bool anyVisible = false;
    for (auto& s : specs) {
        if (*s.ptr && !(*s.ptr)->mVertices.empty() && g_meshAlpha[s.role] > 0.01f) {
            anyVisible = true; break;
        }
    }
    if (!anyVisible) g_meshAlpha[0] = 0.8f;

    Reg3DCustom::printMeshBBox(*liverMesh3D, "liver (organ-drop, aligned)");
    std::cout << "[OrganDrop] swapped organ set (" << loaded
              << " organ(s) loaded); "
              << (preserveRegisteredPose ? "pose preserved (no prealign)."
                                         : "prealigned to existing target.")
              << " sceneDiag/voxel unchanged." << std::endl;
    return true;
}
// FileDropHandler.hのglfw_onFileDropを使用するように変更
static void handleFileDrop(GLFWwindow* win, int count, const char** paths) {
    if (count <= 0 || !paths) return;
    auto* ctx = (AppContext*)glfwGetWindowUserPointer(win);
    if (!ctx) {
        std::cerr << "[FileDrop] ERROR: No AppContext found" << std::endl;
        return;
    }

    std::cout << "[FileDrop] Before: mode=" << (int)ctx->mode << std::endl;

    const std::string filePath = paths[0];
    std::cout << "[FileDrop] Attempting to load: " << filePath << std::endl;

    // -------------------------------------------------------------------
    //  臓器セットのドロップ（CTモデル差し替え）。フォルダ内の臓器OBJが下の
    //  bin-folder スキャンに吸われないよう Reconstruct/画像より「先」に判定。
    //  liver.obj 必須。無いセット(門脈単独など)は拒否。
    // -------------------------------------------------------------------
    {
        OrganDropScan organScan = scanOrganDrop(count, paths);
        if (organScan.anyOrgan) {
            if (!organScan.hasLiver()) {
                std::cerr << "[OrganDrop] REJECTED: the dropped set has no liver.obj. "
                             "liver.obj が必須です(セット一式、または liver.obj を含む"
                             "フォルダをドロップしてください)。非肝臓OBJ単独は不可。"
                          << std::endl;
                return;   // 無効なセット → 何もしない
            }
            // [deform round-trip] "deformed" フォルダ由来なら preserve-pose で受ける
            //   (prealign スキップ → REG#1 姿勢のまま 2回目 register の出発点にする)。
            swapOrganSetFromDrop(organScan.path, organScan.preserveRegisteredPose);
            return;       // 処理済み(or ターゲット無しで拒否)。下に流さない
        }
    }
    // FEATURE Task 4: route Reconstruct-from-BIN inputs (depth_metric.bin /
    // segmentation_mask.png / original.jpg, or a depth_output/ folder) to the
    // reconstruct state machine instead of the normal image-load path. Each
    // dropped path is handled independently (multi-file / folder drops supported).
    {
        // main.cpp owns the stb_image implementation; provide it as the decoder.
        ReconstructFromBin::ImageDecoder decode =
            [](const std::string& p, int channels,
               std::vector<uint8_t>& out, int& w, int& h) -> bool {
            int c = 0;
            unsigned char* px = stbi_load(p.c_str(), &w, &h, &c, channels);
            if (!px) return false;
            out.assign(px, px + (size_t)w * h * channels);
            stbi_image_free(px);
            return true;
        };
        std::error_code ec;
        bool consumed = false;
        for (int i = 0; i < count; ++i) {
            const std::string p = paths[i];
            if (std::filesystem::is_directory(p, ec)) {
                ReconstructFromBin::onFolderDropped(g_reconstructState, p, decode);
                consumed = true;
            } else if (ReconstructFromBin::isReconstructFile(p)) {
                ReconstructFromBin::onFileDropped(g_reconstructState, p, decode);
                consumed = true;
            }
        }
        if (consumed) return;   // handled by Reconstruct; do not fall through to image load
    }

    // 内部パラメータ(.txt)のドロップ → Custom として取り込み即適用。
    // 画像拡張子チェックの「前」に置くことで、キャリブファイル(intrinsics_custom.txt
    // 等)をそのまま投げ込めば反映される。有効な K として読めない .txt はここを
    // 素通りして下の通常ルート/拒否に回る。
    if (maybeImportDroppedIntrinsics(filePath)) {
        std::cout << "[FileDrop] handled as Custom intrinsics" << std::endl;
        return;
    }

    if (!ImageSession::isSupportedExtension(filePath)) {
        std::cerr << "[FileDrop] Unsupported format: " << filePath
                  << "  (expected .png .jpg .jpeg .bmp .ppm)" << std::endl;
        return;
    }

    // カメラが動作している場合は停止
    if (gCamera.active) {
        gCamera.stop();
        std::cout << "[FileDrop] Stopping camera to load image" << std::endl;
    }

    if (!loadImageRectifyAware(*ctx, filePath)) {
        std::cerr << "[FileDrop] Failed to load: " << filePath << std::endl;
        return;
    }

    std::cout << "[FileDrop] After: mode=" << (int)ctx->mode << " (0=Empty, 1=ImageOnly, 2=MaskSelection, 3=Registration)" << std::endl;
}

static void mouse_button_callback(GLFWwindow* win, int button, int action, int mods) {
    // ImGuiがマウスをキャプチャしている場合は処理しない
    if (ImGui::GetIO().WantCaptureMouse) return;

    // AR モード中は 3D シーン操作 (カメラ回転 / メッシュドラッグ / ライト方向)
    // のみ無効化する。
    // — これがないと OrbitCam の cameraPos が更新され続けてライト方向が動く。
    // ただし kMaskSelection / kImageOnly の 2D マスク点打ちは AR とは無関係なので
    // ここでは止めずに通す。
    //   旧コードはここで無条件に return しており、A キーで AR を ON にしたまま
    //   LOAD IMAGE すると mode==kMaskSelection でも arMode==true が残るため、
    //   セグメンテーションのクリックがこの行で全て握り潰され「マウスが反応しない」
    //   状態になっていた (arMode は load/mode 変更でリセットされない)。
    if (gApp.arMode &&
        gApp.mode != AppMode::kMaskSelection &&
        gApp.mode != AppMode::kImageOnly) {
        return;
    }

    // Umeyama 2画面モード
    if (gUmeyama.active && action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(win, &x, &y);
        if (gUmeyama.handleMouse((float)x, (float)y, button,
                                 gWindowWidth, gWindowHeight,
                                 registrationHandle, liverMesh3D, screenMesh))
            return;
    }

    if (action == GLFW_RELEASE) {
        isDragging = false;
        hit_index  = -1;
        return;
    }
    if (action != GLFW_PRESS) return;

    double x, y;
    glfwGetCursorPos(win, &x, &y);

    if (gApp.mode == AppMode::kMaskSelection || gApp.mode == AppMode::kImageOnly) {
        // マスク選択モードでは固定アスペクト比を考慮した座標変換
        if (gApp.mode == AppMode::kMaskSelection && gApp.image.loaded) {
            float imgAspect = (float)gApp.image.width / (float)gApp.image.height;
            float winAspect = (float)gWindowWidth / (float)gWindowHeight;

            int viewW = gWindowWidth;
            int viewH = gWindowHeight;
            int viewX = 0;
            int viewY = 0;

            if (imgAspect > winAspect) {
                viewH = gWindowWidth / imgAspect;
                viewY = (gWindowHeight - viewH) / 2;
            } else {
                viewW = gWindowHeight * imgAspect;
                viewX = (gWindowWidth - viewW) / 2;
            }

            // ビューポート外のクリックは無視
            if (x < viewX || x > viewX + viewW || y < viewY || y > viewY + viewH) {
                return;
            }

            // ビューポート内の座標を画像座標に変換
            float u = ((float)(x - viewX) / viewW) * gApp.image.width;
            float v = ((float)(y - viewY) / viewH) * gApp.image.height;

            // Pick the destination list based on which mask the user is
            // currently editing. Without this branch (the original code
            // pushed unconditionally to gApp.maskPoints), Instrument-mode
            // clicks ended up in the Liver list, the Renderer never drew
            // any cyan/orange points, and the [MaskPicker] log was missing
            // the "Liver" / "Instrument" prefix.
            const bool isInstrument = (gApp.activeMaskKind == MaskKind::Instrument);
            std::vector<MaskPoint>& dst = isInstrument
                                              ? gApp.instrumentMaskPoints
                                              : gApp.maskPoints;
            const char* kindName = isInstrument ? "Instrument" : "Liver";
            dst.push_back({u, v, button == GLFW_MOUSE_BUTTON_LEFT});

            int nFg = 0, nBg = 0;
            for (const auto& p : dst) (p.fg ? nFg : nBg)++;
            std::cout << "[MaskPicker] " << kindName << " "
                      << (button == GLFW_MOUSE_BUTTON_LEFT ? "FG" : "BG")
                      << " (" << (int)u << "," << (int)v << ")"
                      << "  fg=" << nFg << " bg=" << nBg << std::endl;
        } else {
            // 通常のImageOnlyモード
            if (button == GLFW_MOUSE_BUTTON_LEFT)
                MaskPicker::addFromScreen(gApp, (float)x, (float)y, true);
            else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                MaskPicker::addFromScreen(gApp, (float)x, (float)y, false);
        }
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (hitTestMesh((float)x, (float)y, liverMesh3D, hit_position))
            isDragging = true;
    }
}

static void glfw_onMouseMoveOrbit(GLFWwindow* win, double posX, double posY) {
    static glm::vec2 last(0.0f);

    // 元コード line 4844-4848 と同じ: ImGui がマウスを使っている間は
    // カメラ操作をスキップ。ただし last は更新しないと UI を離れた瞬間に
    // 巨大な delta でカメラが飛んでしまう。
    if (ImGui::GetIO().WantCaptureMouse) {
        last = glm::vec2((float)posX, (float)posY);
        return;
    }

    // AR モード中は観測専用 (3D カメラ・メッシュ操作を無効化)。
    // last だけ更新しないと AR モード OFF の瞬間に巨大な delta でカメラが飛ぶ。
    if (gApp.arMode) {
        last = glm::vec2((float)posX, (float)posY);
        return;
    }

    float dx = (float)posX - last.x;
    float dy = (float)posY - last.y;

    // Umeyama 2画面モード — 左右独立カメラ操作
    if (gUmeyama.active) {
        bool L = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
        bool R = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        gUmeyama.handleMouseMove((float)posX, dx, dy, L, R, gWindowWidth);
        last = glm::vec2((float)posX, (float)posY);
        return;
    }

    bool L = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
    bool R = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (!isDragging) {
        if (L && !R) OrbitCam.Rotate(dx, dy);
        else if (R && !L) OrbitCam.Pan(dx, -dy);
    } else {
        if (L && R) {
            glm::vec3 mv = OrbitCam.cameraDirection * (dy * OrbitCam.LIGHT_MOUSE_SENSITIVITY);
            translateAllMeshes(mv);
        } else if (R && !L) {
            float mdx =  dx * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
            float mdy = -dy * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
            glm::vec3 mv = OrbitCam.cameraRight * mdx + OrbitCam.cameraUp * mdy;
            translateAllMeshes(mv);
        } else if (L && !R) {
            float rx = dy * 0.01f;
            float ry = dx * 0.01f;
            rotateAllMeshes(liverCenter(),
                            OrbitCam.cameraRight, rx,
                            OrbitCam.cameraUp,    ry);
        }
    }

    last = glm::vec2((float)posX, (float)posY);
}

static void glfw_onMouseScroll(GLFWwindow* win, double, double deltaY) {
    // 元コード line 4971 と同じ: ImGui がマウスを使っている間はスクロールを無視
    if (ImGui::GetIO().WantCaptureMouse) return;

    // AR モード中は観測専用 (ズームも無効化)
    if (gApp.arMode) return;

    // Umeyama 2画面モード
    if (gUmeyama.active) {
        double x, y;
        glfwGetCursorPos(win, &x, &y);
        gUmeyama.handleScroll((float)x, (float)deltaY, gWindowWidth);
        return;
    }

    bool R = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (R) {
        float s = (deltaY > 0) ? scaleSpeed : (1.0f / scaleSpeed);
        scaleAllMeshes(liverCenter(), s);
    } else {
        OrbitCam.gRadius += (float)deltaY * OrbitCam.ZOOM_SENSITIVITY;
        OrbitCam.gRadius  = glm::clamp(OrbitCam.gRadius,
                                      OrbitCam.minRadius,
                                      OrbitCam.maxRadius);
    }
}

static bool setupObjScene() {
    if (g_objSourcePath.empty() || !std::filesystem::exists(g_objSourcePath)) {
        std::cerr << "[setupObjScene] OBJ missing: " << g_objSourcePath << std::endl;
        return false;
    }

    // ---------------------------------------------------------------------
    //  Reload all organ meshes from the CT model (.obj) so we ALWAYS restart
    //  from CT/model space. Previously this block only undid the last prealign
    //  (g_lastOrganTransform), which left Apply-Init-Pose rotation / CMA-ES /
    //  Umeyama / manual transforms baked into the vertices. Reloading discards
    //  the entire transform chain, so the snapshotInitialPose() taken right
    //  after the new prealign reflects the TRUE initial pose for the new depth
    //  target (otherwise loading depth mid-session corrupts the initial pose
    //  and every later Apply Init Pose / d_lr / d_cc derived from it).
    // ---------------------------------------------------------------------
    {
        struct OrganSpec { mCutMesh** ptr; const char* file; glm::vec3 color; };
        OrganSpec specs[] = {
            { &liverMesh3D,   "liver.obj",  glm::vec3(0.8f, 0.2f, 0.2f) },
            { &portalMesh3D,  "portal.obj", glm::vec3(0.2f, 0.2f, 0.8f) },
            { &veinMesh3D,    "vein.obj",   glm::vec3(0.2f, 0.5f, 0.5f) },
            { &tumorMesh3D,   "tumor.obj",  glm::vec3(0.8f, 0.5f, 0.5f) },
            { &segmentMesh3D, "segment.obj", glm::vec3(0.2f, 0.8f, 0.5f) },
            { &gbMesh3D,      "gb.obj",     glm::vec3(0.2f, 0.8f, 0.2f) },
        };
        for (auto& s : specs) {
            // Preserve any runtime color tweak; fall back to the default color.
            glm::vec3 keepColor = (*s.ptr) ? (*s.ptr)->mColor : s.color;
            delete *s.ptr;
            *s.ptr = new mCutMesh(mCutMesh().loadMeshFromFile((MODEL_PATH + s.file).c_str()));
            (*s.ptr)->mColor = keepColor;
            setUp(**s.ptr);
        }

        // Re-point AppContext caches (otherwise getOrganList() is fine but
        // gApp.liver / gApp.all hold dangling pointers to the freed meshes).
        gApp.liver   = liverMesh3D;  gApp.portal  = portalMesh3D; gApp.vein = veinMesh3D;
        gApp.tumor   = tumorMesh3D;  gApp.segment = segmentMesh3D; gApp.gb  = gbMesh3D;
        allMeshes = { liverMesh3D, portalMesh3D, veinMesh3D,
                      tumorMesh3D, segmentMesh3D, gbMesh3D };
        gApp.all = allMeshes;

        // No transform is applied to the freshly loaded vertices yet.
        g_lastOrganTransform    = glm::mat4(1.0f);
        g_hasLastOrganTransform = false;
        g_lastOrganOffset       = glm::vec3(0.0f);

        // [CT-handoff] liver came straight from MODEL_PATH; reset the tracked
        // raw-CT source so the Umeyama handoff uses the startup model again.
        g_liverCtSourcePath     = MODEL_PATH + "liver.obj";

        // Geometry-derived labels were computed on the old pose; invalidate so
        // they auto-recompute (vertex indexing is identical after reload, so
        // this is a safety reset rather than strictly required).
        g_liverRegion = LiverRegionLabel::Result();
        g_liverLR     = LiverLeftRightLabel::Result();
        g_liverCC     = LiverCranioCaudalLabel::Result();
        clearLiverLabelVizCaches();   // [bugfix] drop stale marker viz indices too

        std::cout << "[setupObjScene] Reloaded all 6 organ meshes from " << MODEL_PATH
                  << "  (CT space; all prior transforms discarded)" << std::endl;
    }
    Reg3DCustom::clearCachedTargetCloud();

    if (screenMesh) { delete screenMesh; screenMesh = nullptr; }
    screenMesh = new mCutMesh(mCutMesh().loadMeshFromFile(g_objSourcePath.c_str()));
    gApp.screen = screenMesh;
    setUp(*screenMesh);
    Reg3DCustom::printMeshBBox(*screenMesh, "OBJ raw (meters)");

    Reg3DCustom::CameraIntrinsics K;
    // intrinsicsSource に応じて候補リストを切り替える。Step 10 で Run Depth が
    // 使う srcTag (Custom/Calib/Preset=key/DA3) と対応させ、SAM2 が書き出す
    // intrinsics_<tag>.txt を読む。
    std::vector<std::string> intrinsicsCandidates;
    if (g_intrinsicsSource == IntrinsicsSource::Custom) {
        intrinsicsCandidates = {
            DEPTH_OUTPUT_PATH + "intrinsics_custom.txt",
        };
    } else if (g_intrinsicsSource == IntrinsicsSource::Calib) {
        intrinsicsCandidates = {
            DEPTH_OUTPUT_PATH + "intrinsics_calib_last.txt",
            DEPTH_OUTPUT_PATH + "intrinsics_calib.txt",
        };
    } else if (g_intrinsicsSource == IntrinsicsSource::Preset) {
        // SAM2 round-trips the passed preset K to intrinsics_<key>.txt; fall
        // back to legacy intrinsics_k4a.txt for older k4a-tagged OBJs.
        intrinsicsCandidates = {
            DEPTH_OUTPUT_PATH + "intrinsics_" + g_currentPresetKey + ".txt",
            DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt",
        };
    } else {
        // DA3: the OBJ is built with the K SAM2 actually used (intrinsics_da3.txt
        // / legacy intrinsics_k4a.txt); intrinsics_da3_last.txt / intrinsics.txt
        // hold the DA3 estimate.
        intrinsicsCandidates = {
            DEPTH_OUTPUT_PATH + "intrinsics_da3.txt",
            DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt",
            DEPTH_OUTPUT_PATH + "intrinsics_da3_last.txt",
            DEPTH_OUTPUT_PATH + "intrinsics.txt",
        };
    }

    if (!Reg3DCustom::loadCameraIntrinsicsAny(intrinsicsCandidates, K)) {
        const char* labels[] = {"DA3", "Kinect", "Custom", "Calib"};
        std::cerr << "[Intrinsics] "
                  << labels[std::clamp(intrinsicsSourceToLegacyInt(g_intrinsicsSource), 0, 3)]
                  << " selected but no matching file under " << DEPTH_OUTPUT_PATH
                  << "; falling back to k4a 720p" << std::endl;
        K = Reg3DCustom::CameraIntrinsics::k4a_color_720p();
    }
    // 解像度チェックは「画像」と K を比較する (ウィンドウではない)。
    // 画像が未ロードの場合はスキップ。
    if (gApp.image.loaded && gApp.image.width > 0 && gApp.image.height > 0) {
        Reg3DCustom::checkIntrinsicsResolution(K, gApp.image.width, gApp.image.height);
    }
    g_intrinsics = K;
    g_intrinsics_raw = Reg3DCustom::CameraIntrinsics{};  // fresh source K; drop any rectify-raw

    Reg3DCustom::printEdgeLengthStats(*screenMesh, "OBJ raw");

    auto cleanupStats = Reg3DCustom::cleanupOBJMesh(
        *screenMesh, g_silhouetteCosThreshold, 0.0f);
    (void)cleanupStats;
    setUp(*screenMesh);
    Reg3DCustom::printMeshBBox(*screenMesh, "OBJ after cleanup");

    OrbitCam.setIntrinsics(K.fx, K.fy, K.cx, K.cy, K.width, K.height);
    std::cout << "[OrbitCam] intrinsics -> " << K.name
              << " (fx=" << K.fx << " fy=" << K.fy
              << " cx=" << K.cx << " cy=" << K.cy
              << " res=" << K.width << "x" << K.height << ")"
              << std::endl;

    // OBJ-projection K: REG writes the K it ACTUALLY used to build the OBJ
    // (scaled to the depth resolution, Task 2) to the canonical intrinsics.txt.
    // The target cloud must be projected with that same K to match the OBJ
    // geometry, so prefer intrinsics.txt; fall back to the legacy pre-migration
    // intrinsics_custom_used.txt, else the display K. OrbitCam / AR background
    // above keep the display K (matches the full-res rectified image).
    Reg3DCustom::CameraIntrinsics K_obj = K;
    {
        Reg3DCustom::CameraIntrinsics Kobj2;
        if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics.txt", Kobj2)
            && Kobj2.valid()) {
            K_obj = Kobj2;
            std::cout << "[OBJ Setup] OBJ-projection K from intrinsics.txt: "
                      << K_obj.width << "x" << K_obj.height << " fx=" << K_obj.fx
                      << "  (display/OrbitCam keeps " << K.width << "x" << K.height
                      << " fx=" << K.fx << ")" << std::endl;
        } else if (g_intrinsicsSource == IntrinsicsSource::Custom &&
                   Reg3DCustom::loadCameraIntrinsics(
                       DEPTH_OUTPUT_PATH + "intrinsics_custom_used.txt", Kobj2)
                   && Kobj2.valid()) {
            K_obj = Kobj2;  // legacy (pre-migration) used-K
            std::cout << "[OBJ Setup] OBJ-projection K from legacy intrinsics_custom_used.txt"
                      << std::endl;
        }
    }

    auto targetCloud = Reg3DCustom::setupOBJTarget(
        *screenMesh, K_obj, Reg3DCustom::OBJ_Y_SIGN_OPENGL);
    if (!targetCloud || targetCloud->empty()) {
        std::cerr << "[OBJ Setup] FAILED to build target cloud" << std::endl;
        return false;
    }

    Reg3DCustom::mirrorMeshAndCloudX(*screenMesh, *targetCloud);

    std::vector<mCutMesh*> organs_for_move = { liverMesh3D, portalMesh3D, veinMesh3D,
                                               tumorMesh3D, segmentMesh3D, gbMesh3D };
    glm::vec3 organCenterPre = Reg3DCustom::computeMeshCenter(*liverMesh3D);
    glm::vec3 objCenterPre   = Reg3DCustom::computeMeshCenter(*screenMesh);
    g_lastOrganOffset        = objCenterPre - organCenterPre;  // legacy (translation only)

    // NEW: similarity prealignment (scale + translation)
    g_lastOrganTransform     = Reg3DCustom::prealignSourceToTarget(
        organs_for_move, *screenMesh);
    g_hasLastOrganTransform  = true;

    // After prealignment, source ≈ target in size. Cache the diagonal as the
    // single reference length for all downstream registration parameters.
    g_sceneDiag = std::max(Reg3DCustom::computeMeshDiag(*screenMesh), 1e-3f);
    std::cout << "[SceneDiag] " << g_sceneDiag
              << "  (target AABB diagonal, used for parameter normalization)"
              << std::endl;

    // ===== Diagnostic: median NN distance L (used for paper-time analysis) =====
    // Compute L of target cloud and store it. Currently NOT used by registration
    // (we tried voxel = 5L per Open3D / Zhou 2018 but it failed FGR tuple test
    // on our dense depth-anything reconstructions; reverted to empirical voxel).
    // L is logged for future sensitivity analysis and paper writing.
    if (targetCloud && !targetCloud->empty()) {
        float L_target = Reg3DCustom::computeMedianNNDistance(*targetCloud);
        g_referenceL = std::max(L_target, 1e-6f);

        // Show the actual parameters in use vs literature-derived values for context
        float voxel_used     = g_sceneDiag * (0.30f / 7.36f);     // currently active
        float voxel_5L       = 5.0f  * g_referenceL;              // Open3D / Zhou 2018
        float voxel_to_L     = voxel_used / g_referenceL;

        std::cout << "[L-info] median NN distance and active params:" << std::endl;
        std::cout << "    L (median NN)          : " << g_referenceL << std::endl;
        std::cout << "    sceneDiag / L          : " << (g_sceneDiag / g_referenceL) << std::endl;
        std::cout << "    voxel (active)         : " << voxel_used
                  << "    (= " << voxel_to_L << " L)" << std::endl;
        std::cout << "    voxel (literature 5L)  : " << voxel_5L
                  << "    [not used: too fine for dense depth-anything clouds]" << std::endl;
    }
    // ===== END diagnostic =====

    // Adjust camera/UI parameters to match new scene scale
    applySceneScaleToCamera();

    setUp(*screenMesh);
    setUp(*liverMesh3D);
    setUp(*portalMesh3D);
    setUp(*veinMesh3D);
    setUp(*tumorMesh3D);
    setUp(*segmentMesh3D);
    setUp(*gbMesh3D);

    Reg3DCustom::printMeshBBox(*screenMesh,  "OBJ final  (camera-space)");
    Reg3DCustom::printMeshBBox(*liverMesh3D, "liverMesh3D (moved)");

    // --- テクスチャ付きboardメッシュのロード（表示専用） ---
    {
        // Board (textured display mesh) — must match the source of screenMesh,
        // otherwise we render OLD geometry textured with NEW K, producing a
        // visible misalignment with the segmentation overlay (this used to be
        // hard-coded to "_k4a_light.obj" which left stale 1280x720 geometry
        // sitting around when the user ran depth in custom/calib mode).
        //
        // Strategy: derive the suffix from g_objSourcePath itself. The screen
        // mesh path was set just before setupObjScene was called and looks
        // like "pc_metric_pinhole_masked_<tag>.obj" -- swap "masked" for
        // "full" and add "_light".
        std::string fullObjPath;
        {
            const std::string& src = g_objSourcePath;
            // Build an ordered list of candidate "full" OBJ paths and pick the
            // first that actually exists on disk. The source mesh name comes in
            // two flavours and the old code only handled the tagged one:
            //   (a) tagged:   pc_metric_pinhole_masked_<tag>.obj
            //                  -> pc_metric_pinhole_full_<tag>_light.obj
            //   (b) tagless:  pc_metric_pinhole_masked.obj   (current depth export)
            //                  -> pc_metric_pinhole_full_light.obj
            // Falling back to a hard-coded "_k4a_light.obj" left the tagless
            // export (the common case now) with no board mesh at all.
            std::vector<std::string> candidates;

            auto posT = src.find("pc_metric_pinhole_masked_");   // tagged form
            if (posT != std::string::npos) {
                std::string head = src.substr(0, posT);
                std::string tail = src.substr(posT + std::string("pc_metric_pinhole_masked_").size());
                auto dotPos = tail.find(".obj");
                std::string tag = (dotPos != std::string::npos) ? tail.substr(0, dotPos) : tail;
                candidates.push_back(head + "pc_metric_pinhole_full_" + tag + "_light.obj");
            }

            auto posU = src.find("pc_metric_pinhole_masked.obj"); // tagless form
            if (posU != std::string::npos) {
                std::string head = src.substr(0, posU);
                candidates.push_back(head + "pc_metric_pinhole_full_light.obj");
            }

            // Generic fallbacks rooted at DEPTH_OUTPUT_PATH (cover both names).
            candidates.push_back(DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_light.obj");
            candidates.push_back(DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_k4a_light.obj");

            fullObjPath = candidates.empty() ? std::string() : candidates.front();
            for (const auto& c : candidates) {
                if (std::filesystem::exists(c)) { fullObjPath = c; break; }
            }
        }
        std::string texPath = DEPTH_OUTPUT_PATH + "texture.png";
        if (std::filesystem::exists(fullObjPath)) {
            if (boardMesh3D) { boardMesh3D->cleanup(); delete boardMesh3D; }
            boardMesh3D = new mCutMesh(mCutMesh().loadMeshFromFile(fullObjPath.c_str()));
            boardMesh3D->mColor = glm::vec3(1.0f, 1.0f, 1.0f);

            // K intrinsicsからUV座標を生成（変換前の座標系で投影 - 過去の
            // 意図的な仕様。X反転と組み合わせて鏡像表示する旧来の挙動）。
            //
            // 注意: 過去のコードは V を `py / H` で計算していたが、これは僅かに
            // 間違っており (1280x720 で 1.8% ずれ、長年気づかれず)、1920x1080 で
            // 8.5% ずれて GL_REPEAT で「画像上部に帯」として顕在化した。
            //
            // 真の関係:
            //   ・obj_exporter は flipY=true で書き出すので、Y_3D = -(y_pixel-cy)*Z/fy
            //     → y_pixel = cy - fy*Y_3D/Z = 2*cy - py  (py は existing 計算式の値)
            //   ・mCutMesh::loadTextureFromData はアップロード時に画像を垂直反転
            //     する (line 114-118) ので GL の t=0 は元画像の下端、t=1 は元画像の上端
            //   ・正しいテクスチャサンプリング: v = 1 - y_pixel/H
            //
            // この補正で V は必ず [0, 1] に収まる (旧式は (2cy-y)/H で
            // 2cy > H のとき 1.0 を超える)。GL_REPEAT のラップが完全消滅し、
            // 帯状の異常表示も解消する。U は flipY の影響を受けないので変更なし。
            {
                const auto& V = boardMesh3D->mVertices;
                int nVerts = (int)(V.size() / 3);
                boardMesh3D->mTexCoords.resize(nVerts * 2);
                const float invW = 1.0f / (float)K.width;
                const float invH = 1.0f / (float)K.height;
                for (int i = 0; i < nVerts; i++) {
                    float x = V[i*3+0];
                    float y = V[i*3+1];
                    float z = V[i*3+2];
                    if (std::abs(z) < 1e-6f) z = 1e-6f;
                    float px = K.fx * x / z + K.cx;
                    float py = K.fy * y / z + K.cy;
                    // U: 既存通り (flipY の影響を受けない)
                    boardMesh3D->mTexCoords[i*2+0] = px * invW;
                    // V: flipY と texture upload-flip の両方を考慮した正しい式
                    //    y_pixel = 2*cy - py;  v = 1 - y_pixel/H
                    float y_pixel = 2.0f * K.cy - py;
                    boardMesh3D->mTexCoords[i*2+1] = 1.0f - y_pixel * invH;
                }
                std::cout << "[Board] UV from intrinsics (" << nVerts
                          << " verts, V corrected for flipY+upload-flip)" << std::endl;
            }

            // screenMeshと同じ変換を適用（mirrorX）
            for (size_t i = 0; i < boardMesh3D->mVertices.size(); i += 3)
                boardMesh3D->mVertices[i] = -boardMesh3D->mVertices[i];

            // X反転でワインディング順が逆転するので修正（法線を正しく向ける）
            for (size_t i = 0; i + 2 < boardMesh3D->mIndices.size(); i += 3)
                std::swap(boardMesh3D->mIndices[i+1], boardMesh3D->mIndices[i+2]);

            // テクスチャロード
            if (std::filesystem::exists(texPath)) {
                boardMesh3D->loadTextureFromFile(texPath);
                std::cout << "[Board] Loaded with texture: " << texPath << std::endl;
            } else {
                std::cout << "[Board] No texture: " << texPath << std::endl;
            }

            setUp(*boardMesh3D);
            Reg3DCustom::printMeshBBox(*boardMesh3D, "boardMesh3D");
        } else {
            std::cout << "[Board] Full OBJ not found: " << fullObjPath << std::endl;
            boardMesh3D = nullptr;
        }
    }

    OrbitCam.rotation = glm::angleAxis(glm::radians(180.0f),
                                       glm::vec3(0.0f, 1.0f, 0.0f));
    OrbitCam.currentTarget = TARGET_TEXTURE;

    // ウィンドウサイズ: K の解像度 + サイドバーが希望、ただしディスプレイに
    // 収まる範囲でクランプ (案A auto-fit)。1920x1080 の Custom intrinsics でも
    // 1080p ディスプレイで動くようにする。3D viewport は compute3DViewport()
    // が K のアスペクト比でフィットさせるため、ウィンドウが K より小さくても
    // 描画自体は壊れない (ピクセル 1:1 表示が崩れるだけ)。
    {
        const int sidebarW = 400;
        int recW = K.width + sidebarW;
        int recH = K.height;

        // ディスプレイ作業領域でクランプ (タスクバー等を除いた領域)
        GLFWmonitor* mon = glfwGetPrimaryMonitor();
        if (mon) {
            int mx, my, mw, mh;
            glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh);
            int maxW = (int)(mw * 0.95f);
            int maxH = (int)(mh * 0.90f);
            if (recW > maxW) recW = maxW;
            if (recH > maxH) recH = maxH;
        }

        glfwSetWindowSize(gWindow, recW, recH);
        gApp.windowW = recW;
        gApp.windowH = recH;

        std::cout << "[Window] Resized to " << recW << "x" << recH
                  << " (calib " << K.width << "x" << K.height
                  << " + sidebar " << sidebarW << ")" << std::endl;
        if (recW < K.width + sidebarW || recH < K.height) {
            std::cout << "[Window] Note: clamped to display work area; "
                         "AR preview is shown at reduced size, "
                         "but ARSave outputs at native " << K.width << "x"
                      << K.height << "." << std::endl;
        }
    }

    gApp.objSourcePath = g_objSourcePath;
    gApp.intrinsics    = K;
    gApp.mode          = AppMode::kRegistration;

    std::cout << "[OBJ Setup] target cloud ready: "
              << targetCloud->size() << " points; mode=Registration"
              << std::endl;
    return true;
}

// obj-migration Phase 5: applyIntrinsicsToRunnerConfig() removed. REG no longer
// passes K to sam2 (sam2 owns no K). REG builds the OBJ from its own K after
// Run Depth via the export hook below (DepthToObjExport), which uses
// intrinsicsSourceToTag(g_intrinsicsSource, ...) directly.

static bool runDepthAndUpdateScene(AppContext& ctx) {
    if (ctx.image.path.empty() || !ctx.image.loaded) {
        std::cerr << "[RunDepth] no image loaded" << std::endl;
        return false;
    }

    DepthRunner runner;
    initDepthRunnerConfig(runner);

    // Propagate the UI checkbox to the external pipeline. When OFF, the
    // runner adds --no-vignette-detect so instrument_segmentation_mask.png
    // contains only the SAM2 instrument result without the auto-detected
    // FOV vignette merged in.
    runner.config.detectVignette = ctx.detectVignette;

    // CUDA / GPU acceleration. When ON, the runner adds --cuda so the
    // external pipeline registers the CUDAExecutionProvider. Harmless
    // when sam2_da3_lite is a CPU-only build (silent CPU fallback).
    runner.config.useCuda = ctx.useCuda;

    // Phase 5: no K is passed to sam2. REG builds the canonical OBJ after the run.

    if (!runner.isAvailable()) {
        std::cerr << "[RunDepth] exe not found: " << runner.config.exePath
                  << std::endl;
        runner.printDiagnostics();
        return false;
    }
    if (!runner.areModelsAvailable()) {
        std::cerr << "[RunDepth] ONNX models missing" << std::endl;
        runner.printDiagnostics();
        return false;
    }

    std::vector<DepthRunnerPoint> pts;
    if (ctx.maskPoints.empty()) {
        // マスクポイントがない場合はデフォルトを生成
        std::cout << "[RunDepth] no mask points; using default (center FG + 4 corner BG)"
                  << std::endl;
        float cx = ctx.image.width / 2.0f;
        float cy = ctx.image.height / 2.0f;
        float marginX = ctx.image.width * 0.1f;
        float marginY = ctx.image.height * 0.1f;

        pts.emplace_back(cx, cy, true);  // 前景: 中心
        pts.emplace_back(marginX, marginY, false);  // 背景: 左上
        pts.emplace_back(ctx.image.width - marginX, marginY, false);  // 背景: 右上
        pts.emplace_back(marginX, ctx.image.height - marginY, false);  // 背景: 左下
        pts.emplace_back(ctx.image.width - marginX, ctx.image.height - marginY, false);  // 背景: 右下
    } else {
        pts.reserve(ctx.maskPoints.size());
        for (const auto& p : ctx.maskPoints) {
            pts.emplace_back(p.u, p.v, p.fg);
        }
        int nFg = 0, nBg = 0;
        for (const auto& p : pts) (p.isForeground ? nFg : nBg)++;
        std::cout << "[RunDepth] passing " << pts.size()
                  << " points (fg=" << nFg << " bg=" << nBg << ")" << std::endl;
    }

    // ---- Instrument prompts (optional) ----
    // When the user has placed any Instrument-mask points, ship them to
    // the external pipeline as --instrument-point / --instrument-bg-point
    // so it runs a second SAM2 pass and writes
    //   <output>/instrument_segmentation_mask.png
    // alongside the liver mask. When the list is empty, we pass an empty
    // vector and the pipeline behaves exactly as before (no second pass,
    // no instrument outputs).
    std::vector<DepthRunnerPoint> instPts;
    instPts.reserve(ctx.instrumentMaskPoints.size());
    for (const auto& p : ctx.instrumentMaskPoints) {
        instPts.emplace_back(p.u, p.v, p.fg);
    }
    if (!instPts.empty()) {
        int nFg = 0, nBg = 0;
        for (const auto& p : instPts) (p.isForeground ? nFg : nBg)++;
        std::cout << "[RunDepth] passing " << instPts.size()
                  << " instrument points (fg=" << nFg << " bg=" << nBg << ")"
                  << std::endl;
    } else {
        // Stale instrument-mask cleanup. The external pipeline only writes
        // instrument_segmentation_mask.png when --instrument-point is
        // supplied, so an empty instPts means "this run produces no
        // instrument mask". Delete any leftover from a previous Run Depth
        // so the downstream ensureInstrumentDistMap() doesn't accidentally
        // pick up a mask that belongs to a different image / prompts and
        // silently corrupt boundary rejection during registration.
        std::error_code ec;
        std::string stalePng = DEPTH_OUTPUT_PATH
                               + "instrument_segmentation_mask.png";
        std::string staleJpg = DEPTH_OUTPUT_PATH
                               + "instrument_segmentation_overlay.jpg";
        if (std::filesystem::remove(stalePng, ec)) {
            std::cout << "[RunDepth] removed stale instrument mask: "
                      << stalePng << std::endl;
        }
        // Overlay removal is silent; it's purely a debugging artifact.
        std::filesystem::remove(staleJpg, ec);
    }

    auto rr = runner.run(ctx.image.path, pts, nullptr, instPts);
    if (!rr.success) {
        std::cerr << "[RunDepth] runner failed (exit=" << rr.exitCode << ")"
                  << std::endl;
        return false;
    }
    if (rr.hasInstrumentMask()) {
        std::cout << "[RunDepth] instrument mask written to "
                  << rr.instrumentSegmentationMaskPath << std::endl;
    }

    // ---- persist the DA3 estimate as the DA3 source ----
    // Phase 3: sam2 now writes its DA3-estimated K to intrinsics_da3.txt (NOT
    // intrinsics.txt, which REG owns as canonical). Promote it to
    // intrinsics_da3_last.txt so the file-driven "DA3 (last estimate)" source
    // stays selectable/re-loadable, symmetric with Custom/Calib.
    {
        Reg3DCustom::CameraIntrinsics da3K;
        const std::string da3Src = DEPTH_OUTPUT_PATH + "intrinsics_da3.txt";
        const std::string da3Dst = DEPTH_OUTPUT_PATH + "intrinsics_da3_last.txt";
        if (Reg3DCustom::loadCameraIntrinsics(da3Src, da3K) && da3K.valid()) {
            da3K.name = "da3_last";
            if (Reg3DCustom::saveCameraIntrinsics(da3Dst, da3K))
                std::cout << "[RunDepth] DA3 estimate -> intrinsics_da3_last.txt ("
                          << da3K.width << "x" << da3K.height
                          << " fx=" << da3K.fx << ")" << std::endl;
        } else {
            std::cout << "[RunDepth] no usable intrinsics_da3.txt to promote (skipped)"
                      << std::endl;
        }
    }

    // ---- obj-migration Phase 3: REG-side OBJ export (routes A/B/C) ----
    // sam2 still writes its tagged OBJ in parallel (Phase 3); REG additionally
    // writes the canonical OBJ + intrinsics.txt from its OWN K, overwriting the
    // tagged copy (last-writer-wins, REG後勝ち) for the diff comparison.
    {
        namespace fs = std::filesystem;
        const std::string binPath  = DEPTH_OUTPUT_PATH + "depth_metric.bin";
        const std::string canonObj = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj";

        if (fs::exists(binPath)) {
            // Route A: metric depth present -> REG builds the OBJ from its own K.
            depthexport::DepthBin db;
            if (depthexport::loadDepthMetricBin(binPath, db)) {
                int rw = 0, rh = 0, rc = 0, mw = 0, mh = 0, mc = 0;
                unsigned char* rgb  = stbi_load(rr.originalPath.c_str(), &rw, &rh, &rc, 3);
                unsigned char* mraw = stbi_load(rr.segmentationMaskPath.c_str(), &mw, &mh, &mc, 1);
                if (rgb && mraw && rw == db.width && rh == db.height
                    && mw == db.width && mh == db.height) {
                    std::vector<uint8_t> mask((size_t)mw * mh);
                    for (size_t i = 0; i < mask.size(); ++i) mask[i] = (mraw[i] > 127) ? 255 : 0;
                    std::vector<uint8_t> rgbv(rgb, rgb + (size_t)rw * rh * 3);

                    depthexport::Request req;
                    req.rgbPixels   = rgbv.data();
                    req.width       = db.width;
                    req.height      = db.height;
                    req.depthMetric = &db.depth;
                    req.mask        = &mask;
                    // Task 2: scale K to the depth resolution. sam2 may have
                    // downscaled the image (e.g. 4K -> 1080p) and depth_metric.bin
                    // is at that processed resolution; unprojecting with the
                    // unscaled K would misproject (and shed ~half the vertices).
                    req.K           = Reg3DCustom::scaleIntrinsics(g_intrinsics,
                                                                   db.width, db.height);
                    if (g_intrinsics.width != db.width || g_intrinsics.height != db.height)
                        std::cout << "[RunDepth] K scaled to depth res "
                                  << db.width << "x" << db.height
                                  << " (from " << g_intrinsics.width << "x"
                                  << g_intrinsics.height << ", fx " << g_intrinsics.fx
                                  << " -> " << req.K.fx << ")" << std::endl;
                    req.K.name      = intrinsicsSourceToTag(g_intrinsicsSource, g_currentPresetKey);
                    req.tag         = req.K.name;
                    req.outDir      = DEPTH_OUTPUT_PATH;
                    auto er = depthexport::exportDepthArtifacts(req);
                    if (!er.ok) std::cerr << "[RunDepth] REG exportDepthArtifacts failed" << std::endl;
                } else {
                    std::cerr << "[RunDepth] REG export skipped (res mismatch): bin="
                              << db.width << "x" << db.height << " rgb=" << rw << "x" << rh
                              << " mask=" << mw << "x" << mh << std::endl;
                }
                if (rgb)  stbi_image_free(rgb);
                if (mraw) stbi_image_free(mraw);
            }
        } else if (fs::exists(canonObj)) {
            // Route B: external OBJ injected (no metric depth). Keep it as-is;
            // just emit intrinsics.txt from the current K so downstream matches.
            std::cout << "[RunDepth] depth_metric.bin absent; using existing OBJ as-is: "
                      << canonObj << std::endl;
            depthexport::saveIntrinsicsFile(DEPTH_OUTPUT_PATH + "intrinsics.txt",
                                            g_intrinsics, g_intrinsics.width, g_intrinsics.height);
        } else {
            // Route C: no metric depth and no canonical OBJ. In Phase 3 sam2 still
            // writes a tagged OBJ in parallel, so objPath below can still load it;
            // we log rather than hard-fail (post-Phase-5 this becomes a real error).
            std::cerr << "[RunDepth] WARNING: no depth_metric.bin and no canonical OBJ; "
                         "relying on sam2 tagged OBJ (Phase 3 parallel)" << std::endl;
        }
    }

    // Phase 5: REG wrote the canonical OBJ above (export hook). Load it; fall back
    // to the legacy _k4a name only for pre-migration data on disk.
    std::string objPath = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj";
    if (!std::filesystem::exists(objPath)) {
        std::string legacy = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked_k4a.obj";
        if (std::filesystem::exists(legacy)) {
            std::cerr << "[RunDepth] canonical OBJ missing; using legacy "
                      << legacy << std::endl;
            objPath = legacy;
        } else {
            std::cerr << "[RunDepth] no OBJ found (canonical nor legacy)" << std::endl;
            return false;
        }
    }

    // ---------------------------------------------------------------------
    //  Invalidate cached masks tied to the previous segmentation_mask.png.
    //  The depth runner has just overwritten that file on disk, but
    //  ensureBoundaryMap() short-circuits when g_boundaryDistMap.valid is
    //  still true, so without this step setupOBJTarget() reuses the
    //  PREVIOUS image's boundary map -- which silently corrupts both
    //  extractTargetFromOBJ() (per-vertex boundaryDist) and the IoU
    //  computed by computeSilhouette2DObjectiveFast() (which reads
    //  g_boundaryDistMap.data directly). g_projectedLiverMask is the
    //  rendered SAM2-on-screenMesh cache used by Shift+E and is also
    //  derived from the same mask, so drop it too.
    // ---------------------------------------------------------------------
    g_boundaryDistMap.invalidate();
    g_instrumentDistMap.invalidate();    // 器具マスクも再読み込みさせる
    g_projectedLiverMask.invalidate();
    g_gnUnsignedBdyValid = false;        // Alt+W GN cache: depends on boundary
    std::cout << "[RunDepth] invalidated boundary map & projected liver mask"
              << "  (will be rebuilt from new segmentation_mask.png)"
              << std::endl;

    g_objSourcePath = objPath;
    if (!setupObjScene()) {
        std::cerr << "[RunDepth] setupObjScene failed" << std::endl;
        return false;
    }

    // 初期メッシュ状態をバックアップ
    snapshotInitialPose();

    // Phase 2 拡張: target cloud の subset AABB を 3 通り (全体/+X/-X) 計算
    // Position selector が「Right なら右だけのバウンディング」を実現するため。
    computeTargetSubsetAabbs();

    // Phase 2 拡張 v2: source liver の subset AABB も 3 通り計算
    // 「source の左半分を target の左半分にマッチ」させるため。
    computeSourceLiverSubsetAabbs();

    // マスク選択モードから抜けてRegistrationモードへ
    ctx.mode = AppMode::kRegistration;

    std::cout << "[RunDepth] OK in " << rr.elapsedMs
              << " ms; mode=Registration" << std::endl;
    return true;
}

// =============================================================================
//  runSegmentOnly: Run the SAM2 stage of the depth pipeline only and pop up
//  a preview of segmentation_overlay.jpg in the UI. Used by the "Segment 1"
//  button to let the user sanity-check the mask BEFORE paying for depth.
//
//  This is a stripped-down sibling of runDepthAndUpdateScene():
//    - same DepthRunner config / intrinsics handling / mask-point handling
//    - sets runner.config.stage = DepthStage::Segment so the external
//      executable returns after writing segmentation_mask.png and
//      segmentation_overlay.jpg
//    - does NOT touch the OBJ scene, mode, or registration state
//    - loads segmentation_overlay.jpg into a GL texture and hands it to
//      gUI.state for the popup to render
//
//  Failure mode: any error along the way leaves gUI.state.segPreviewOpen
//  unchanged (i.e. the popup doesn't open) and logs to stderr. The user
//  can retry or just skip Segment 1 entirely and use Run Depth as before.
// =============================================================================

// Helper: free any prior preview texture so we don't leak GL handles when
// the user clicks Segment 1 multiple times in a row.
static void releaseSegPreviewTexture() {
    if (gUI.state.segPreviewTexId != 0) {
        GLuint t = (GLuint)gUI.state.segPreviewTexId;
        glDeleteTextures(1, &t);
        gUI.state.segPreviewTexId = 0;
        gUI.state.segPreviewW = 0;
        gUI.state.segPreviewH = 0;
    }
}

// Helper: load a JPG/PNG into a fresh GL texture. Returns 0 on failure.
// Forces 3-channel RGB (matches stbi_load's RGB request elsewhere in the
// file). Mirrors the GL setup in mCutMesh::loadTextureFromFile.
static GLuint loadImageAsGLTexture(const std::string& path,
                                   int* outW, int* outH)
{
    int w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data) {
        std::cerr << "[SegPreview] stbi_load failed: " << path
                  << " (" << stbi_failure_reason() << ")" << std::endl;
        return 0;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(data);
    if (outW) *outW = w;
    if (outH) *outH = h;
    return tex;
}

// Run the SAM2-only stage of the depth pipeline for the requested mask
// kind. For Liver this writes segmentation_mask.png; for Instrument it
// writes instrument_segmentation_mask.png (so the liver mask on disk is
// not overwritten by an Instrument preview). The popup state is shared:
// segPreviewTexId points at whichever overlay was just produced.
static bool runSegmentOnly(AppContext& ctx, MaskKind kind) {
    const bool isInstrument = (kind == MaskKind::Instrument);
    const char* tag = isInstrument ? "Instrument" : "Segment1";

    if (ctx.image.path.empty() || !ctx.image.loaded) {
        std::cerr << "[" << tag << "] no image loaded" << std::endl;
        return false;
    }

    DepthRunner runner;
    initDepthRunnerConfig(runner);

    // Propagate the UI checkbox to the external pipeline. Same logic
    // as runDepthAndUpdateScene -- both code paths produce
    // instrument_segmentation_mask.png so both must respect the toggle.
    runner.config.detectVignette = ctx.detectVignette;

    // CUDA / GPU acceleration. Same logic as runDepthAndUpdateScene.
    runner.config.useCuda = ctx.useCuda;

    // Phase 5: no K passed to sam2 (Segment stage only writes a mask anyway).

    // Stage selector: SAM2 only.
    runner.config.stage = DepthStage::Segment;
    // Output filename selector. For Instrument we ask the external pipeline
    // to write to instrument_segmentation_*.png so the liver mask isn't
    // clobbered. For Liver we leave maskOutputName empty (legacy names).
    runner.config.maskOutputName = isInstrument ? "instrument" : "";

    if (!runner.isAvailable()) {
        std::cerr << "[" << tag << "] exe not found: " << runner.config.exePath
                  << std::endl;
        return false;
    }
    if (!runner.areModelsAvailable()) {
        std::cerr << "[" << tag << "] ONNX models missing" << std::endl;
        return false;
    }

    // Pick the right point list. The Liver path keeps its legacy default-
    // points fallback (center FG + 4 corner BG). For Instrument we *don't*
    // fall back because there is no sensible default position for "the
    // tool" -- instead we just refuse if the user hasn't placed any
    // prompts yet, and the UI button enabling logic should match.
    const std::vector<MaskPoint>& srcPts =
        isInstrument ? ctx.instrumentMaskPoints : ctx.maskPoints;

    std::vector<DepthRunnerPoint> pts;
    if (srcPts.empty()) {
        if (isInstrument) {
            std::cerr << "[Instrument] no instrument mask points; "
                         "click on the tool first" << std::endl;
            return false;
        }
        std::cout << "[Segment1] no mask points; using default "
                     "(center FG + 4 corner BG)" << std::endl;
        float cx = ctx.image.width / 2.0f;
        float cy = ctx.image.height / 2.0f;
        float marginX = ctx.image.width  * 0.1f;
        float marginY = ctx.image.height * 0.1f;
        pts.emplace_back(cx, cy, true);
        pts.emplace_back(marginX, marginY, false);
        pts.emplace_back(ctx.image.width - marginX, marginY, false);
        pts.emplace_back(marginX, ctx.image.height - marginY, false);
        pts.emplace_back(ctx.image.width - marginX,
                         ctx.image.height - marginY, false);
    } else {
        pts.reserve(srcPts.size());
        for (const auto& p : srcPts) pts.emplace_back(p.u, p.v, p.fg);
        int nFg = 0, nBg = 0;
        for (const auto& p : pts) (p.isForeground ? nFg : nBg)++;
        std::cout << "[" << tag << "] passing " << pts.size()
                  << " points (fg=" << nFg << " bg=" << nBg << ")" << std::endl;
    }

    auto rr = runner.run(ctx.image.path, pts);
    if (!rr.success) {
        std::cerr << "[" << tag << "] runner failed (exit=" << rr.exitCode << ")"
                  << std::endl;
        return false;
    }

    // ---- Pick the right overlay/mask file for the popup ----
    // The external pipeline's maskOutputName="instrument" puts the outputs
    // under instrument_segmentation_*.png; otherwise the legacy names.
    std::string maskPath, overlayPath;
    if (isInstrument) {
        maskPath    = rr.instrumentSegmentationMaskPath;
        overlayPath = rr.instrumentSegmentationOverlayPath;
    } else {
        maskPath    = rr.segmentationMaskPath;
        overlayPath = rr.segmentationOverlayPath;
    }
    if (maskPath.empty() || !std::filesystem::exists(maskPath)) {
        std::cerr << "[" << tag << "] expected mask missing: "
                  << maskPath << std::endl;
        return false;
    }
    std::string previewPath = overlayPath;
    if (previewPath.empty() || !std::filesystem::exists(previewPath)) {
        previewPath = maskPath;
    }

    releaseSegPreviewTexture();
    int w = 0, h = 0;
    GLuint tex = loadImageAsGLTexture(previewPath, &w, &h);
    if (tex == 0) {
        std::cerr << "[" << tag << "] failed to upload preview texture from "
                  << previewPath << std::endl;
        return false;
    }
    gUI.state.segPreviewTexId    = (unsigned int)tex;
    gUI.state.segPreviewW        = w;
    gUI.state.segPreviewH        = h;
    gUI.state.segPreviewOpen     = true;
    gUI.state.segPreviewScore    = 0.0f;
    gUI.state.segPreviewFgPixels = 0;

    std::cout << "[" << tag << "] OK in " << rr.elapsedMs
              << " ms; preview=" << previewPath
              << " (" << w << "x" << h << ")" << std::endl;
    return true;
}

static bool initOpenGL() {
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight,
                               "Simple Registration", nullptr, nullptr);
    if (!gWindow) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(gWindow);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;

    glfwSetKeyCallback(gWindow, glfw_onKey);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);
    glfwSetFramebufferSizeCallback(gWindow, glfw_OnFramebufferSize);
    glfwSetCursorPosCallback(gWindow, glfw_onMouseMoveOrbit);
    glfwSetScrollCallback(gWindow, glfw_onMouseScroll);
    glfwSetWindowUserPointer(gWindow, &gApp);
    glfwSetDropCallback(gWindow, handleFileDrop);
    glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPos(gWindow, gWindowWidth/2.0, gWindowHeight/2.0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glViewport(0, 0, gWindowWidth, gWindowHeight);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.075f, 0.094f, 1.0f);

    {
        const float fontSize = 18.0f;
        bool fontLoaded = false;
        const char* fontPaths[] = {
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            nullptr
        };
        for (int i = 0; fontPaths[i]; i++) {
            FILE* f = fopen(fontPaths[i], "rb");
            if (f) {
                fclose(f);
                io.Fonts->AddFontFromFileTTF(fontPaths[i], fontSize);
                fontLoaded = true;
                printf("[ImGui] Font loaded: %s (%.0fpx)\n", fontPaths[i], fontSize);
                break;
            }
        }
        if (!fontLoaded) {
            ImFontConfig cfg;
            cfg.SizePixels = fontSize;
            io.Fonts->AddFontDefault(&cfg);
            printf("[ImGui] Using default font (%.0fpx)\n", fontSize);
        }
    }

    ImGui_ImplGlfw_InitForOpenGL(gWindow, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    return true;
}

// =============================================================================
//  Intrinsics startup auto-selection (Step 4 / step7-cleanup)
//  File-driven sources are tried by existence; Preset is the always-available
//  default fallback:
//      Custom : intrinsics_custom.txt valid
//      Calib  : intrinsics_calib_last.txt valid (legacy intrinsics_calib.txt
//               also accepted until the Step 6 rename lands)
//      DA3    : intrinsics_da3_last.txt valid (DA3 estimate from the last
//               Run Depth)
//      Preset : default fallback (g_currentPresetKey, currently
//               azure_kinect_1080p) when no source file exists
//  Note: DA3 sits above the Preset fallback so "da3_last only -> DA3" holds;
//  the nominal "Preset > DA3" ordering only applied to a persisted preset
//  choice, which is not implemented.
// =============================================================================
static IntrinsicsSource autoSelectIntrinsicsSource() {
    namespace fs = std::filesystem;
    const std::string base = DEPTH_OUTPUT_PATH;
    auto fileValid = [&](const char* f) {
        Reg3DCustom::CameraIntrinsics K;
        return fs::exists(base + f)
            && Reg3DCustom::loadCameraIntrinsics(base + f, K) && K.valid();
    };

    // 1. Custom — highest priority
    if (fileValid("intrinsics_custom.txt")) return IntrinsicsSource::Custom;
    // 2. Calib (prefer _last; tolerate legacy name until Step 6)
    if (fileValid("intrinsics_calib_last.txt") || fileValid("intrinsics_calib.txt"))
        return IntrinsicsSource::Calib;
    // 3. DA3 (last estimate, file-driven)
    if (fileValid("intrinsics_da3_last.txt")) return IntrinsicsSource::DA3;
    // 4. Preset — default fallback (always available)
    return IntrinsicsSource::Preset;
}

// Load K for the currently-selected source into g_intrinsics / gApp.intrinsics /
// OrbitCam. All four sources are file/preset driven and symmetric; on failure
// OrbitCam is left unset (warned).
static void loadIntrinsicsFromCurrentSource() {
    Reg3DCustom::CameraIntrinsics K;
    bool ok = false;
    switch (g_intrinsicsSource) {
        case IntrinsicsSource::Custom:
            ok = Reg3DCustom::loadCameraIntrinsics(
                     DEPTH_OUTPUT_PATH + "intrinsics_custom.txt", K);
            break;
        case IntrinsicsSource::Calib:
            ok = Reg3DCustom::loadCameraIntrinsics(
                     DEPTH_OUTPUT_PATH + "intrinsics_calib_last.txt", K)
              || Reg3DCustom::loadCameraIntrinsics(
                     DEPTH_OUTPUT_PATH + "intrinsics_calib.txt", K);
            break;
        case IntrinsicsSource::Preset:
            ok = Reg3DCustom::lookupPreset(g_currentPresetKey, K);
            break;
        case IntrinsicsSource::DA3:
            ok = Reg3DCustom::loadCameraIntrinsics(
                     DEPTH_OUTPUT_PATH + "intrinsics_da3_last.txt", K);
            break;
    }
    if (ok && K.valid()) {
        g_intrinsics    = K;
        gApp.intrinsics = K;
        g_intrinsics_raw = Reg3DCustom::CameraIntrinsics{};  // fresh source K; drop any rectify-raw
        OrbitCam.setIntrinsics(K.fx, K.fy, K.cx, K.cy, K.width, K.height);
        std::cout << "[Intrinsics] startup source="
                  << intrinsicsSourceToLegacyInt(g_intrinsicsSource)
                  << " key=" << g_currentPresetKey
                  << "  K " << K.width << "x" << K.height
                  << " fx=" << K.fx << " fy=" << K.fy
                  << " cx=" << K.cx << " cy=" << K.cy << std::endl;
    } else {
        std::cerr << "[Intrinsics] startup: failed to load K for selected source "
                  << intrinsicsSourceToLegacyInt(g_intrinsicsSource)
                  << "; leaving OrbitCam unset (no K)" << std::endl;
    }
}

int main(int argc, char** argv) {
    initPaths();
    initFilePaths();

    // Headless self-check for Step 4 autoSelect (no GUI/OpenGL). Runs the real
    // autoSelectIntrinsicsSource() + loadIntrinsicsFromCurrentSource() against
    // whatever intrinsics_*.txt files exist under DEPTH_OUTPUT_PATH, prints the
    // chosen source + resulting K, then exits. Used by test scripts.
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--check-intrinsics") {
            const char* names[] = {"DA3", "Kinect/Preset", "Custom", "Calib"};
            g_intrinsicsSource = autoSelectIntrinsicsSource();
            int li = intrinsicsSourceToLegacyInt(g_intrinsicsSource);
            std::cout << "[CheckIntrinsics] autoSelect -> source=" << li
                      << " (" << names[std::clamp(li, 0, 3)] << ")"
                      << " presetKey=" << g_currentPresetKey << std::endl;
            loadIntrinsicsFromCurrentSource();
            std::cout << "[CheckIntrinsics] g_intrinsics valid="
                      << (g_intrinsics.valid() ? 1 : 0)
                      << " " << g_intrinsics.width << "x" << g_intrinsics.height
                      << " fx=" << g_intrinsics.fx << " fy=" << g_intrinsics.fy
                      << std::endl;
            return 0;
        }
    }

    if (!initOpenGL()) {
        std::cerr << "GLFW initialization failed" << std::endl;
        return -1;
    }

    OrbitCam.setWindowSizePointers(&gWindowWidth, &gWindowHeight);
    OrbitCam.setGlobalMatrixPointers(&view, &projection, &model, &objPos);
    // Step 4: 起動時に Custom > Calib > Preset > Auto の優先順位で自動選択し、
    // 選ばれたソースから K をロードする。旧ハードコード K4A 720p は撤廃
    // (default preset = azure_kinect_720p なので、ファイルが何も無い環境では
    //  従来同様 K4A 720p が入る)。Auto のときは OrbitCam を未設定のままにする。
    g_intrinsicsSource = autoSelectIntrinsicsSource();
    loadIntrinsicsFromCurrentSource();
    OrbitCam.printCameraInfo();

    ShaderProgram shaderProgram;
    shaderProgram.loadShaders("../../../shaders/basic.vert",
                              "../../../shaders/basic.frag");
    ShaderProgram shaderProgramCube;
    shaderProgramCube.loadShaders("../../../shaders/texture.vert",
                                  "../../../shaders/texture.frag");
    g_pShader     = &shaderProgram;
    g_pShaderCube = &shaderProgramCube;

    gApp.arBg.initGL();
    if (!gApp.arBg.loadTexture(DEPTH_OUTPUT_PATH + "original.jpg")) {
        std::cerr << "[AR] background image missing -- overlay disabled"
                  << std::endl;
    }
    gMaskRenderer.initGL();

    // スフィアマーカーの初期化（クラスタ・対応点描画用）
    g_sphereMarker.generate(1.0f, 16, 16);
    g_sphereMarker.setup();
    gUmeyama.init();

    liverMesh3D = new mCutMesh(liverMesh3D->loadMeshFromFile((MODEL_PATH + "liver.obj").c_str()));
    liverMesh3D->mColor = glm::vec3(0.8f, 0.2f, 0.2f);
    setUp(*liverMesh3D);

    portalMesh3D = new mCutMesh(portalMesh3D->loadMeshFromFile((MODEL_PATH + "portal.obj").c_str()));
    portalMesh3D->mColor = glm::vec3(0.2f, 0.2f, 0.8f);
    setUp(*portalMesh3D);

    veinMesh3D = new mCutMesh(veinMesh3D->loadMeshFromFile((MODEL_PATH + "vein.obj").c_str()));
    veinMesh3D->mColor = glm::vec3(0.2f, 0.5f, 0.5f);
    setUp(*veinMesh3D);

    tumorMesh3D = new mCutMesh(tumorMesh3D->loadMeshFromFile((MODEL_PATH + "tumor.obj").c_str()));
    tumorMesh3D->mColor = glm::vec3(0.8f, 0.5f, 0.5f);
    setUp(*tumorMesh3D);

    segmentMesh3D = new mCutMesh(segmentMesh3D->loadMeshFromFile((MODEL_PATH + "segment.obj").c_str()));
    segmentMesh3D->mColor = glm::vec3(0.2f, 0.8f, 0.5f);
    setUp(*segmentMesh3D);

    gbMesh3D = new mCutMesh(gbMesh3D->loadMeshFromFile((MODEL_PATH + "gb.obj").c_str()));
    gbMesh3D->mColor = glm::vec3(0.2f, 0.8f, 0.2f);
    setUp(*gbMesh3D);

    // -----------------------------------------------------------------
    //  Capture original CT-mm diagonals of liver and tumor.
    //  These are the invariants used by Shift+M to undo the full scale
    //  chain (prealign × registration × manual). They must be recorded
    //  HERE -- right after the .obj load -- because setupObjScene below
    //  will run prealignSourceToTarget which mutates the vertices.
    // -----------------------------------------------------------------
    {
        auto bboxDiag = [](const mCutMesh* m) -> float {
            if (!m || m->mVertices.size() < 3) return 0.0f;
            glm::vec3 mn(m->mVertices[0], m->mVertices[1], m->mVertices[2]), mx = mn;
            for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
                glm::vec3 v(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2]);
                mn = glm::min(mn, v);
                mx = glm::max(mx, v);
            }
            return glm::length(mx - mn);
        };
        g_originalLiverDiagMm = bboxDiag(liverMesh3D);
        g_originalTumorDiagMm = bboxDiag(tumorMesh3D);
        g_hasOriginalDiags    = (g_originalLiverDiagMm > 1e-6f);
        // [CT-handoff] startup liver came from MODEL_PATH; record it as the
        // raw-CT source for the Umeyama handoff (overridden later if a
        // different organ set is dropped in).
        g_liverCtSourcePath   = MODEL_PATH + "liver.obj";
        std::cout << "[OriginalDiag] liver=" << g_originalLiverDiagMm
                  << " mm, tumor=" << g_originalTumorDiagMm
                  << " mm  (CT-mm reference for Shift+M scale restoration)"
                  << std::endl;
    }

    // Sync with AppContext
    gApp.liver = liverMesh3D;
    gApp.portal = portalMesh3D;
    gApp.vein = veinMesh3D;
    gApp.tumor = tumorMesh3D;
    gApp.segment = segmentMesh3D;
    gApp.gb = gbMesh3D;

    allMeshes.push_back(liverMesh3D);
    allMeshes.push_back(portalMesh3D);
    allMeshes.push_back(veinMesh3D);
    allMeshes.push_back(tumorMesh3D);
    allMeshes.push_back(segmentMesh3D);
    allMeshes.push_back(gbMesh3D);

    gApp.all = allMeshes;

    // Initialize other AppContext members
    gApp.window = gWindow;
    gApp.windowW = gWindowWidth;
    gApp.windowH = gWindowHeight;
    gApp.orbitCam = OrbitCam;
    gApp.model = model;
    gApp.view = view;
    gApp.projection = projection;
    gApp.objPos = objPos;
    gApp.reg = registrationHandle;
    gApp.silhouetteCosThreshold = g_silhouetteCosThreshold;
    gApp.objSourcePath = g_objSourcePath;
    gApp.intrinsics = g_intrinsics;

    setupUICallbacks();

    // 起動時に既存 OBJ が残っていれば即ロード。
    //
    // 重要: 前回 Run Depth したときの結果は「画像（original.jpg, segmentation_mask.png）
    // + メッシュ（pc_metric_pinhole_masked_<tag>.obj） + K（intrinsics_<tag>.txt）」
    // のセットで一貫している必要がある。Custom と Kinect (k4a) を行き来した
    // 履歴があると、ディスクに両方の OBJ が残り得て、現在の画像と整合しない
    // 古いメッシュを選んでしまうリスクがある。
    //
    // そのため候補を「修正時刻の新しい順」で並べ直し、**最新** の OBJ を選ぶ。
    // 直近の Run Depth が書き出したものが常に正解。
    {
        // obj-migration Phase 6: prefer the canonical (tag-less) OBJ that REG now
        // writes. Fall back to legacy tagged OBJs (most-recent wins) only for
        // pre-migration data left on disk.
        g_objSourcePath.clear();
        const std::string canon = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj";
        std::error_code ec0;
        if (std::filesystem::exists(canon, ec0)) {
            g_objSourcePath = canon;
            std::cout << "[main] Using canonical OBJ: " << canon << std::endl;
            // Align g_intrinsicsSource with intrinsics.txt's name field so
            // setupObjScene loads the matching K.
            Reg3DCustom::CameraIntrinsics k;
            if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics.txt", k)) {
                if      (k.name == "custom") g_intrinsicsSource = IntrinsicsSource::Custom;
                else if (k.name == "calib")  g_intrinsicsSource = IntrinsicsSource::Calib;
                else if (k.name == "da3" || k.name == "da3_last")
                                             g_intrinsicsSource = IntrinsicsSource::DA3;
                else                         g_intrinsicsSource = IntrinsicsSource::Preset;
                std::cout << "[main] Intrinsics source from intrinsics.txt name='"
                          << k.name << "' (g_intrinsicsSource="
                          << intrinsicsSourceToLegacyInt(g_intrinsicsSource) << ")" << std::endl;
            }
        } else {
            // Legacy fallback: tagged OBJs, most-recent first.
            struct CandObj { std::string path; std::string tag;
                             std::filesystem::file_time_type mtime; };
            std::vector<CandObj> objCandidates;
            for (const auto& tag : {"k4a", "custom", "calib"}) {
                std::string p = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked_" + tag + ".obj";
                std::error_code ec;
                if (std::filesystem::exists(p, ec)) {
                    auto t = std::filesystem::last_write_time(p, ec);
                    if (!ec) objCandidates.push_back({p, tag, t});
                }
            }
            std::sort(objCandidates.begin(), objCandidates.end(),
                      [](const CandObj& a, const CandObj& b){ return a.mtime > b.mtime; });
            if (!objCandidates.empty()) {
                const auto& chosen = objCandidates.front();
                g_objSourcePath = chosen.path;
                if (objCandidates.size() > 1) {
                    std::cout << "[main] (legacy) Multiple tagged OBJs, picking most recent:"
                              << std::endl;
                    for (const auto& c : objCandidates)
                        std::cout << "[main]   " << (c.path == chosen.path ? "* " : "  ")
                                  << c.path << std::endl;
                }
                std::cout << "[main] Using legacy tagged OBJ: " << chosen.path << std::endl;
                if      (chosen.tag == "custom") g_intrinsicsSource = IntrinsicsSource::Custom;
                else if (chosen.tag == "calib")  g_intrinsicsSource = IntrinsicsSource::Calib;
                else                             g_intrinsicsSource = IntrinsicsSource::Preset;  // k4a
                std::cout << "[main] Intrinsics source aligned to legacy OBJ tag: "
                          << chosen.tag << std::endl;
            }
        }
    }
    if (!g_objSourcePath.empty()) {
        if (!setupObjScene()) return -1;
        snapshotInitialPose();  // 初期ポーズをバックアップ（プリセット回転用）
        computeTargetSubsetAabbs();   // Phase 2: target cloud の AABB 3 通り (full/+X/-X)
        computeSourceLiverSubsetAabbs();   // Phase 2 v2: source liver の AABB 3 通り
    } else {
        std::cout << "[main] No existing pc_metric_pinhole_masked_*.obj found" << std::endl;
        std::cout << "[main] Image-only mode."
                     " Drop image, click FG/BG points, press R to run depth."
                  << std::endl;
        gApp.mode = AppMode::kImageOnly;

        const std::vector<std::string> candidates = {
            DEPTH_OUTPUT_PATH + "original.jpg",
            gDepthInputImage
        };
        bool fallbackLoaded = false;
        for (const auto& p : candidates) {
            if (p.empty() || !std::filesystem::exists(p)) continue;
            if (loadImageRectifyAware(gApp, p)) { fallbackLoaded = true; break; }
        }
        if (!fallbackLoaded) {
            std::cout << "[main] No fallback image -- viewport stays black"
                         " until you drop a file." << std::endl;
        }
    }

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(gWindow)) {
        double now = glfwGetTime();
        float  dt  = (float)(now - lastTime);
        lastTime   = now;

        showFPS(gWindow);
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        syncUIState();

        // ----------------------------------------------------------------
        //  [Phase 6 LIVE] Normal-Compatible Refine tick.
        //  -------------------------------------------------------------
        //  When a live session is active (= user pressed Shift+N or
        //  Ctrl+Shift+N with g_normRefineLiveMode=true), advance ONE
        //  refineStep per render frame. Each tick:
        //    - runs one refineStep on the persistent NormalRefineLive::state
        //    - applies the resulting incremental transform to organMeshes
        //    - tracks bestRMSE / bestCumulativeTransform
        //    - if converged / max-iter, calls finishNormalCompatRefineLive
        //      which sets pendingSave for the consumer immediately below.
        //
        //  No-op when no live session is in flight (returns immediately).
        //  Placed BEFORE rendering so the mesh update is visible the same
        //  frame.
        // ----------------------------------------------------------------
        tickNormalCompatRefineLive();
        // Consume pendingSave: after finish() flagged it, call
        //   poseSaveToLibrary here (the function is only reachable from
        //   main.cpp because PoseLibrary.h includes after the actions
        //   header). Mirrors the blocking wrapper's post-save call.
        if (NormalRefineLive::pendingSave) {
            poseSaveToLibrary(SaveCriterion::EITHER, NormalRefineLive::mask);
            NormalRefineLive::pendingSave = false;
        }

        // [LIVE] CPD + soft-ICP frame-driven replay (same pattern as above).
        //   When a live session is active, replay one batch of trajectory
        //   entries per frame; finish() flags pendingSave, consumed here.
        tickCpdLive();
        if (CpdLive::pendingSave) {
            poseSaveToLibrary(SaveCriterion::RMSE);
            CpdLive::pendingSave = false;
            if (!g_cpdCanRevert) g_cpdStageDone[5] = false; // session reverted it
        }
        tickSoftIcpLive();
        if (SoftIcpLive::pendingSave) {
            poseSaveToLibrary(SaveCriterion::RMSE);
            SoftIcpLive::pendingSave = false;
        }

        // Phase 7b Step 3c — Contour Sweep tick.
        //   While active, process one batch of candidates per frame.
        //   The mesh stays put; only the visualization (best-so-far
        //   transformed rim) updates so the user sees convergence
        //   live. On the frame the sweep completes, the apply / save
        //   tail runs here (same as Ctrl+W / Alt+W tail).
        if (g_contourSweepState.active) {
            const bool more = tickContourSweep();
            if (!more) {
                // ---- Finish: apply best T + save to PoseLibrary ----
                auto& S = g_contourSweepState;
                if (S.best_cost < 1e17) {
                    std::cout << "[Ctrl+Alt+W] finishing sweep,"
                              << " applying best T:  cost="
                              << S.best_cost << "px" << std::endl;

                    poseAutoSaveBeforeRegistration();
                    {
                        auto organs = getOrganList();
                        int n_valid = 0;
                        for (auto* m : organs) if (m) n_valid++;
                        std::cout << "[Ctrl+Alt+W] applying best T to "
                                  << n_valid << " organ meshes..."
                                  << std::endl;
                        NormalRefine::applyIncrementalTransform(
                            glm::dmat4(S.best_T), organs);
                    }

                    g_showDebugShapeMatch = false;
                    g_debugShapeMatchBestSrc.clear();
                    g_debugShapeMatchBestTransform = S.best_T;
                    g_debugShapeMatchBestCost      = S.best_cost;

                    // Clear trial visualization (was yellow during sweep)
                    g_contourSweepShowTrial = false;
                    g_contourSweepTrialSrc.clear();
                    g_contourSweepTgtAnchors3D.clear();
                    g_contourSweepSrcPivotsTrial.clear();
                    g_contourSweepCurrentITgt = -1;
                    g_contourSweepCurrentJSrc = -1;

                    std::cout << "[Ctrl+Alt+W] saving pose to PoseLibrary"
                              << " (max_iter=0 pseudo-session)..."
                              << std::endl;
                    const int saved_max_iter = g_normRefineMaxIter;
                    g_normRefineMaxIter = 0;
                    g_stepStartTime = std::chrono::steady_clock::now();
                    gUI.state.regMethod = 3;
                    if (g_normRefineLiveMode) {
                        startNormalCompatRefineLive(
                            NormalRefine::NORMAL_COMPAT,
                            g_activeQuadrantMask);
                    } else {
                        runNormalCompatRefineSession(
                            NormalRefine::NORMAL_COMPAT,
                            g_activeQuadrantMask);
                        poseSaveToLibrary(SaveCriterion::EITHER,
                                          g_activeQuadrantMask);
                    }
                    g_normRefineMaxIter = saved_max_iter;
                } else {
                    std::cout << "[Ctrl+Alt+W] sweep complete but no valid"
                              << " best found — mesh unchanged"
                              << std::endl;
                }
            }
        }

        // Phase 7b Step 3d — Silhouette 2D Sweep tick (new path).
        //   Independent state from the Step 3c sweep above; both
        //   functions are mutually exclusive (start dispatch picks
        //   one based on g_silhouetteSweepEnable). On completion,
        //   the apply / save tail mirrors the Step 3c tail exactly,
        //   but reads from g_silhouetteSweep instead of g_contourSweepState.
        if (g_silhouetteSweep.active) {
            const bool more = tickSilhouetteSweep();
            if (!more) {
                auto& S = g_silhouetteSweep;
                if (S.best_cost < 1e17) {
                    std::cout << "[Ctrl+Alt+W/3d] finishing sweep,"
                              << " applying best T:  cost="
                              << S.best_cost << "px" << std::endl;

                    poseAutoSaveBeforeRegistration();
                    {
                        auto organs = getOrganList();
                        int n_valid = 0;
                        for (auto* m : organs) if (m) n_valid++;
                        std::cout << "[Ctrl+Alt+W/3d] applying best T to "
                                  << n_valid << " organ meshes..."
                                  << std::endl;
                        NormalRefine::applyIncrementalTransform(
                            glm::dmat4(S.best_T), organs);
                    }

                    g_showDebugShapeMatch = false;
                    g_debugShapeMatchBestSrc.clear();
                    g_debugShapeMatchBestTransform = S.best_T;
                    g_debugShapeMatchBestCost      = S.best_cost;

                    // Clear trial visualization (was yellow / cyan
                    // during sweep). Mirrors the Step 3c finish path.
                    g_contourSweepShowTrial = false;
                    g_contourSweepTrialSrc.clear();
                    g_contourSweepTgtAnchors3D.clear();
                    g_contourSweepSrcPivotsTrial.clear();
                    g_contourSweepCurrentITgt = -1;
                    g_contourSweepCurrentJSrc = -1;

                    std::cout << "[Ctrl+Alt+W/3d] saving pose to PoseLibrary"
                              << " (max_iter=0 pseudo-session)..."
                              << std::endl;
                    const int saved_max_iter = g_normRefineMaxIter;
                    g_normRefineMaxIter = 0;
                    g_stepStartTime = std::chrono::steady_clock::now();
                    gUI.state.regMethod = 3;
                    if (g_normRefineLiveMode) {
                        startNormalCompatRefineLive(
                            NormalRefine::NORMAL_COMPAT,
                            g_activeQuadrantMask);
                    } else {
                        runNormalCompatRefineSession(
                            NormalRefine::NORMAL_COMPAT,
                            g_activeQuadrantMask);
                        poseSaveToLibrary(SaveCriterion::EITHER,
                                          g_activeQuadrantMask);
                    }
                    g_normRefineMaxIter = saved_max_iter;
                } else {
                    std::cout << "[Ctrl+Alt+W/3d] sweep complete but no valid"
                              << " best found — mesh unchanged"
                              << " (reason: " << S.fail_reason << ")"
                              << std::endl;
                    // Even on failure, clean up trial visualization
                    g_contourSweepShowTrial = false;
                    g_contourSweepTrialSrc.clear();
                    g_contourSweepTgtAnchors3D.clear();
                    g_contourSweepSrcPivotsTrial.clear();
                    g_contourSweepCurrentITgt = -1;
                    g_contourSweepCurrentJSrc = -1;
                }
            }
        }

        // マスク選択モードでは通常のUIを表示しない
        if (gApp.mode != AppMode::kMaskSelection) {
            gUI.draw(gWindowWidth, gWindowHeight);
        }

        // screenMesh 描画モード切替（点群 / 三角形）と点サイズ調整
        //   - レジストレーションモード時のみ表示（screenMesh 自体がそのときだけ意味を持つ）
        //   - チェックを外すと従来の三角形描画に戻る（緊急時のフォールバック）

        // ----------------------------------------------------------------
        //  Ctrl+G (V3-R) Quadrant Selector
        //  -------------------------------------------------------------
        //  Region (Shift+R) と LR (Y) のラベルを合成した 4 象限から、
        //  Ctrl+G の最適化対象とする象限をユーザが選択する。
        //  - 解剖学的配置: 上=anterior, 下=posterior, 左=患者の右, 右=患者の左
        //  - subset 頂点数 (重複排除後) をリアルタイム表示
        //  - ラベル未計算の場合は警告表示 (Shift+R / Y を促す)
        //  - Umeyama 2画面モード中は非表示 (画面中央の overlay と被るため、
        //    また毎フレーム O(N) の countByQuadrant / makeQuadrantSubsetIdx
        //    が走るので Umeyama 時の体感カクつき要因にもなる)
        // ----------------------------------------------------------------
        // [PHASE-3] The full Ctrl+G Quadrant Selector content is no longer a
        // standalone floating window. It is registered here as a hook (a lambda
        // capturing frame-loop locals by reference) and rendered inside
        // Debug Panel > G tab by DebugPanel::draw() below, under the same
        // kRegistration && !gUmeyama.active guard. All functionality preserved.
        g_debugPanel.drawGBody = [&]() {

            const bool labelsReady = g_liverRegion.valid() && g_liverLR.valid();
            if (!labelsReady) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                   "Labels not computed yet.");
                ImGui::TextWrapped("Press Shift+R (anterior/rim/posterior)"
                                   " and Y (left/right) first,"
                                   " or H (auto-compute both).");
            }

            // 象限ごとの頂点数 (ラベル準備済みのときだけ計算)
            int n_AR = 0, n_AL = 0, n_PR = 0, n_PL = 0;
            if (labelsReady) {
                LiverLeftRightLabel::countByQuadrant(
                    g_liverRegion.labels, g_liverLR.labels,
                    n_AR, n_AL, n_PR, n_PL);
            }

            ImGui::Text("Anatomical orientation:");
            ImGui::Text("  Top=anterior, Left=patient's right");
            ImGui::Separator();

            // 2×2 grid: BeginTable で枠線付きの 2×2 を作る
            const ImGuiTableFlags tableFlags =
                ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame;
            if (ImGui::BeginTable("##quad_grid", 2, tableFlags)) {
                using LR = LiverLeftRightLabel::QuadrantMask;
                auto checkbox_cell = [](const char* shortName,
                                        int   nv,
                                        uint8_t bit,
                                        uint8_t& mask)
                {
                    bool on = (mask & bit) != 0;
                    char label[64];
                    if (nv > 0) {
                        std::snprintf(label, sizeof(label),
                                      "%s\n(%d v)", shortName, nv);
                    } else {
                        std::snprintf(label, sizeof(label),
                                      "%s\n(--)", shortName);
                    }
                    if (ImGui::Checkbox(label, &on)) {
                        if (on)  mask |= bit;
                        else     mask &= static_cast<uint8_t>(~bit);
                    }
                };

                // Row 1: anterior
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("ant_R", n_AR, LR::QUAD_AR, g_activeQuadrantMask);
                ImGui::TableNextColumn();
                checkbox_cell("ant_L", n_AL, LR::QUAD_AL, g_activeQuadrantMask);

                // Row 2: posterior
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("pos_R", n_PR, LR::QUAD_PR, g_activeQuadrantMask);
                ImGui::TableNextColumn();
                checkbox_cell("pos_L", n_PL, LR::QUAD_PL, g_activeQuadrantMask);

                ImGui::EndTable();
            }

            ImGui::Separator();

            // subset 頂点数のプレビュー (ラベル準備済みのときだけ)
            if (labelsReady) {
                auto subset = LiverLeftRightLabel::makeQuadrantSubsetIdx(
                    g_liverRegion.labels, g_liverLR.labels,
                    g_activeQuadrantMask);
                const int total = static_cast<int>(g_liverRegion.labels.size());
                ImGui::Text("Subset: %d / %d unique vertices",
                            (int)subset.size(), total);
                std::string maskStr = LiverLeftRightLabel::quadrantMaskString(
                    g_activeQuadrantMask);
                ImGui::Text("Mask: %s (0x%02X)",
                            maskStr.c_str(), (unsigned)g_activeQuadrantMask);
            } else {
                ImGui::Text("Subset: (n/a)");
                std::string maskStr = LiverLeftRightLabel::quadrantMaskString(
                    g_activeQuadrantMask);
                ImGui::Text("Mask: %s (0x%02X)",
                            maskStr.c_str(), (unsigned)g_activeQuadrantMask);
            }

            // 便利ボタン
            if (ImGui::Button("All")) {
                g_activeQuadrantMask = LiverLeftRightLabel::QUAD_ALL;
            }
            ImGui::SameLine();
            if (ImGui::Button("Anterior only")) {
                g_activeQuadrantMask = static_cast<uint8_t>(
                    LiverLeftRightLabel::QUAD_AR |
                    LiverLeftRightLabel::QUAD_AL);
            }
            ImGui::SameLine();
            if (ImGui::Button("Right only")) {
                g_activeQuadrantMask = static_cast<uint8_t>(
                    LiverLeftRightLabel::QUAD_AR |
                    LiverLeftRightLabel::QUAD_PR);
            }
            ImGui::SameLine();
            if (ImGui::Button("None")) {
                g_activeQuadrantMask = LiverLeftRightLabel::QUAD_NONE;
            }

            // -----------------------------------------------------------
            //  [NEW V3R/SearchMode] Reduced-DoF search dimension
            //  --------------------------------------------------------
            //  Switches the CMA-ES decision-vector dimension for Ctrl+G:
            //    7-DoF (default): tx,ty,tz, rx,ry,rz, scale
            //                     (V3R byte-identical when other Ctrl+G
            //                      knobs are also at defaults)
            //    6-DoF rigid    : tx,ty,tz, rx,ry,rz       (scale fixed=1)
            //    4-DoF XY+RX+RY : tx,ty, rx,ry             (tz/rz/scale=0/0/1)
            //
            //  Motivation (HANDOVER_UNIFIED_ALL.md §III/3.1): repeated
            //  Ctrl+G presses can drift into scale blowup -- each
            //  session adopts a scale slightly >1.0 which accumulates
            //  until the mask is inflated. SIX_DOF_RIGID and
            //  FOUR_DOF_XYRXRY remove the offending DoFs as a hard
            //  constraint rather than a soft penalty.
            //
            //  Recommended workflow:
            //    1) 7-DoF Ctrl+G    (coarse, 1-2 sessions)
            //    2) 6-DoF rigid     (after scale converged, 1-2 sessions)
            //    3) 4-DoF XY+RX+RY  (final polish; no scale/Z/roll drift)
            //
            //  The wrapper in RegistrationActions.h additionally
            //  pre-scales tx_range / ry_range / jitter_local_t etc. by
            //  0.7x (SIX_DOF_RIGID) or 0.5x (FOUR_DOF_XYRXRY) so the
            //  reduced-DoF search stays appropriately local without
            //  needing a separate sigma0 knob.
            // -----------------------------------------------------------
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.85f, 1.0f, 0.7f, 1.0f),
                               "Search dimension (Ctrl+G)");

            {
                int mode = (int)g_ctrlgSearchMode;
                if (ImGui::RadioButton("7-DoF: T+R+Scale (default)##ctrlg_search_mode",
                                       &mode, 0)) {
                    g_ctrlgSearchMode =
                        CmaesRefineV3R::SearchMode::SEVEN_DOF;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Full 7-DoF search:\n"
                        "  tx, ty, tz, rx, ry, rz, scale.\n"
                        "Default. Byte-identical to original V3R when\n"
                        "other Ctrl+G knobs (AR-vis, caudal, beta) are\n"
                        "also at defaults.");
                }

                if (ImGui::RadioButton("6-DoF rigid: T+R, scale=1##ctrlg_search_mode",
                                       &mode, 1)) {
                    g_ctrlgSearchMode =
                        CmaesRefineV3R::SearchMode::SIX_DOF_RIGID;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Rigid 6-DoF: tx, ty, tz, rx, ry, rz; scale\n"
                        "frozen at 1.0.\n"
                        "Use after the first Ctrl+G has settled scale\n"
                        "to prevent mask-expansion drift on repeated runs.\n"
                        "Range and jitter auto-shrunk to 0.7x.");
                }

                if (ImGui::RadioButton("4-DoF: TX, TY, RX, RY only##ctrlg_search_mode",
                                       &mode, 2)) {
                    g_ctrlgSearchMode =
                        CmaesRefineV3R::SearchMode::FOUR_DOF_XYRXRY;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Minimal 4-DoF: tx, ty, rx, ry only.\n"
                        "tz, rz, scale all locked at identity.\n"
                        "Final polish stage; assumes a fixed-AR-camera\n"
                        "workflow where scale/roll/depth have already\n"
                        "converged.\n"
                        "Range and jitter auto-shrunk to 0.5x.");
                }

                // Mode-aware diagnostic line
                const char* sigma_note = "ranges 1.0x, jitter 1.0x (V3R default)";
                switch (g_ctrlgSearchMode) {
                case CmaesRefineV3R::SearchMode::SEVEN_DOF:
                    sigma_note = "ranges 1.0x, jitter 1.0x (V3R default)";
                    break;
                case CmaesRefineV3R::SearchMode::SIX_DOF_RIGID:
                    sigma_note = "ranges 0.7x, jitter 0.7x";
                    break;
                case CmaesRefineV3R::SearchMode::FOUR_DOF_XYRXRY:
                    sigma_note = "ranges 0.5x, jitter 0.5x";
                    break;
                }
                ImGui::TextDisabled("  auto: %s", sigma_note);

                ImGui::SliderFloat("min_match_ratio##ctrlg_search_mode",
                                   &g_ctrlgMinMatchRatio,
                                   0.10f, 0.50f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Minimum fraction of source voxel cells that must\n"
                        "find a target NN within max_dist_sq; otherwise\n"
                        "penalty_value (9.9) is returned for that sample.\n"
                        "V3R / ParamsV3 default is 0.30.\n"
                        "Lower values (0.15 - 0.25) can help 4-DoF /\n"
                        "6-DoF modes when the constrained search starts\n"
                        "from a poor pose that would otherwise hit the\n"
                        "penalty floor at Gen 0.");
                }
            }

            // -----------------------------------------------------------
            //  Rim-weighted V3R extension (opt-in)
            //  --------------------------------------------------------
            //  Three controls stacked on top of the 4-quadrant selector.
            //  When all three are at their defaults
            //    (AR-vis = OFF, β = 0.0, Show RIM pairs = OFF)
            //  Ctrl+G behaves identically to the original V3R (and at
            //  QUAD_ALL, byte-identical to Shift+G). Ticking any of
            //  them activates the corresponding extension.
            // -----------------------------------------------------------
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                               "Rim-weighted (opt-in)");

            ImGui::Checkbox("AR-visible only (filter source subset)",
                            &g_ctrlgUseArVisFilter);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Restrict source vertices to those "
                                  "visible from the fixed AR camera "
                                  "(cam_pos=(0,0,0), look-at=+Z).\n"
                                  "Removes back-side mesh vertices that have no "
                                  "counterpart in the single-sided depth target.\n"
                                  "OFF = byte-identical to Shift+G at QUAD_ALL.");
            }

            // -----------------------------------------------------------
            //  Only-Caudal (R-feat-2): anatomical CC-axis filter.
            //  Independent of AR-vis. Uses g_liverCC labels (Shift+H).
            //  Weak/uncomputed states only show a warning here -- the
            //  CC sign Flip toggle lives in the Initial Orientation
            //  panel (drawAnatomicalAxesStatus) to keep one source of
            //  truth; avoid duplicating it on Ctrl+G.
            // -----------------------------------------------------------
            ImGui::Checkbox("Only Caudal rim (mesh-intrinsic)",
                            &g_ctrlgUseCaudalOnly);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Restrict source vertices to the caudal "
                                  "(foot-side) half of the mesh, classified by "
                                  "LiverCranioCaudalLabel (Shift+H).\n"
                                  "Anatomical axis; transform-invariant (no "
                                  "camera tuning).\n"
                                  "Orthogonal to AR-visible: ticking both "
                                  "applies the Combine mode below.\n"
                                  "OFF = no caudal filter applied.");
            }
            // CC state notice (no Flip button here -- handled in
            // Initial Orientation panel).
            if (g_ctrlgUseCaudalOnly && !g_liverCC.valid()) {
                ImGui::TextColored(
                    ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
                    "  CC labels not yet computed - press Shift+H "
                    "or Apply Init Pose.");
            } else if (g_ctrlgUseCaudalOnly && g_liverCC.valid() &&
                       g_liverCC.cc.weak) {
                ImGui::TextColored(
                    ImVec4(0.96f, 0.32f, 0.32f, 1.0f),
                    "  [WEAK %.1f%%] verify CC sign in Initial "
                    "Orientation panel; use Flip CC if reversed.",
                    g_liverCC.cc.confidence * 100.0f);
            }

            // Combine mode (effective only when BOTH checkboxes are on).
            // Always shown; greyed out when not applicable.
            {
                const bool both_on = g_ctrlgUseArVisFilter && g_ctrlgUseCaudalOnly;
                if (!both_on) ImGui::BeginDisabled();
                ImGui::Text("  Combine when both ON:");
                ImGui::SameLine();
                int mode = (int)g_ctrlgArvisCaudalCombine;
                if (ImGui::RadioButton("AND##arvis_caudal", &mode, 0)) {
                    g_ctrlgArvisCaudalCombine = 0;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("OR##arvis_caudal",  &mode, 1)) {
                    g_ctrlgArvisCaudalCombine = 1;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("AND: vertex must be AR-visible AND caudal "
                                      "(strict; smallest subset, default).\n"
                                      "OR : vertex passes if AR-visible OR caudal "
                                      "(lenient; mutual rescue).\n"
                                      "Effective only when both filters above are ON.");
                }
                if (!both_on) ImGui::EndDisabled();
            }

            ImGui::SliderFloat("beta (rim-rim weight boost)",
                               &g_ctrlgBetaRimWeight, 0.0f, 10.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Multiplicative weight for pairs where\n"
                                  "  source vertex is on the LiverRegion::RIM band\n"
                                  "  AND target point's boundaryDist < threshold.\n"
                                  "w_i = 1 + beta * is_rim_src * is_rim_tgt.\n"
                                  "0.0 = uniform RMSE (byte-identical accumulator).\n"
                                  "1.0 = rim-rim pairs counted twice.\n"
                                  "3.0 = rim-rim pairs counted 4x.");
            }

            ImGui::SliderFloat("rim threshold [px]",
                               &g_ctrlgRimTgtThreshPx, 4.0f, 30.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Target boundaryDist < threshold -> "
                                  "treated as RIM (image-side rim membership).\n"
                                  "Same semantic as Shift+P kBoundaryPxTh (default 12).");
            }

            // Phase 7b Step 2 (Shift+W) で紫点が instrument の境界に
            // 漏れる場合はここで threshold を上げる。Shift+N / V3R-W /
            // V3R / Phase 7b 全部が同じ g_instrumentPxThresh を参照する
            // ので、ここで変えると Ctrl+G の RIM 抽出にも即時反映される。
            // 既存の Inst Px スライダ (HemiAuto モード regMethod==1 のみ
            // 表示) と完全に同じ global を共有。
            ImGui::SliderFloat("inst threshold [px]",
                               &g_instrumentPxThresh, 0.0f, 50.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Target points with instrumentDist < threshold\n"
                                  "are excluded from RIM (instrument-aware\n"
                                  "boundary rejection).\n"
                                  "Shared with Shift+N / V3R / Phase 7b Shift+W.\n"
                                  "Increase if instrument outline leaks into the\n"
                                  "RIM band (visible as purple dots in Shift+W).");
            }

            // ===============================================================
            //  Alt+P  Silhouette Align (2D-IoU BIPOP-CMA-ES, was Shift+E)
            // ---------------------------------------------------------------
            //  Migrated into the G tab because runShiftE is algorithmically a
            //  BIPOP-CMA-ES sibling of Ctrl+G / Ctrl+Shift+G (it maximises 2D
            //  silhouette IoU). The button reproduces the Alt+P key dispatch
            //  byte-identically (poseAutoSaveBeforeRegistration -> runShiftE
            //  -> poseSaveToLibrary(IOU)). N_STARTS / raster step are now
            //  global (g_shiftE_*) so these sliders actually take effect.
            // ===============================================================
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f),
                               "Alt+P  Silhouette Align (2D-IoU BIPOP)");

            ImGui::SliderInt("restarts N (Alt+P)", &g_shiftE_NStarts, 1, 20);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("runShiftE outer restarts (was hardcoded 5).\n"
                                  "run 0 = baseline, 1-2 = local, 3+ = global jitter.");
            }
            ImGui::SliderInt("raster step (Alt+P)", &g_shiftE_RasterStep, 1, 16);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("2D-IoU rasterization step (was hardcoded 8).\n"
                                  "Drives BOTH the IoU readout and the CMA-ES\n"
                                  "objective (linked). Lower = finer / slower.");
            }

            ImGui::Checkbox("pure IoU (Alt+P)", &g_shiftE_pureIoU);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON (default): Alt+P is driven purely by the 2D-IoU\n"
                    "silhouette match (alpha_3d = 0) -- same intent as Ctrl+I,\n"
                    "but via the lighter / faster V1 silhouette path.\n"
                    "OFF: restore the legacy alpha_3d = 0.3 blend (mixes a\n"
                    "small 3D RMSE term into the silhouette objective).");
            }

            // [QUAD-SIL] Alt+P quadrant-aware toggle. Shares the Ctrl+G
            // Quadrant Selector above (g_activeQuadrantMask) — no separate
            // quadrant UI. At Q:ALL the filter is bypassed either way.
            ImGui::Checkbox("quadrant-aware (Alt+P)", &g_shiftE_quadrantAware);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON (default): when the Quadrant Selector above is not\n"
                    "Q:ALL, Alt+P rasterizes ONLY the selected quadrant's\n"
                    "triangles (3-vertex rule, same as Ctrl+I / Ctrl+Shift+G)\n"
                    "and the objective + accept gate + IoU printout all use\n"
                    "that partial silhouette. For occluded targets: select\n"
                    "the visible quadrant(s) and register partial-vs-partial.\n"
                    "OFF: legacy full-mesh raster regardless of selection\n"
                    "(A/B baseline). At Q:ALL both positions are identical.");
            }
            if (g_shiftE_quadrantAware
                && g_activeQuadrantMask != LiverLeftRightLabel::QUAD_ALL) {
                ImGui::SameLine();
                ImGui::TextColored(
                    ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "(%s active)",
                    LiverLeftRightLabel::quadrantMaskString(
                        g_activeQuadrantMask).c_str());
            }

            // [QUAD-SIL BRAKE] Mask-expansion brake weight. Only matters in
            // quadrant mode; mirrors V3RS outside_ratio (g_ctrlgsLambdaOut).
            if (g_shiftE_quadrantAware) {
                ImGui::SliderFloat("expansion brake (Alt+P quad)",
                                   &g_shiftE_outsideLambda, 0.0f, 2.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Penalises the subset ballooning past the target mask\n"
                        "(cost += lambda * outside_ratio). Port of Ctrl+I's\n"
                        "V3RS outside_ratio brake. Without it the symmetric\n"
                        "IoU rewards 'inflate to cover everything' instead of\n"
                        "aligning. 0.5 = V3RS default. 0 = brake off (raw IoU).\n"
                        "Only active in quadrant mode.");
                }
            }

            // [PERF/APPROX] Alt+P: 内側探索雲の voxel ダウンサンプル。
            //  OFF = フル雲 (レガシー基準)。ON = CMA-ES 内側の RMSE ゲート雲
            //  だけ間引く (近似)。外側 computeUnifiedMetrics / 最終 IoU /
            //  PoseLibrary は常にフル雲なので、OFF と ON の最終 full-IoU を
            //  直接比較してデバッグできる。実体は CmaesUtils.h の
            //  g_silDownsample* 群 (run() 内 extractFrontFacePoints 直後で発火)。
            ImGui::Checkbox("downsample search cloud (Alt+P)",
                            &CmaesRefine::g_silDownsampleEnable);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "OFF (default): full target cloud (~317k pts) for the\n"
                    "inner CMA-ES gate -- legacy baseline.\n"
                    "ON: voxel-downsample the inner search cloud only.\n"
                    "Final IoU / PoseLibrary always use the FULL cloud,\n"
                    "so compare final IoU OFF vs ON to validate.\n"
                    "APPROXIMATE: can move the optimum. If final IoU\n"
                    "degrades, lower the voxel (more points).");
            }
            if (CmaesRefine::g_silDownsampleEnable) {
                ImGui::SliderFloat("ds voxel x sceneDiag (Alt+P)",
                                   &CmaesRefine::g_silDownsampleVoxelRatio,
                                   0.002f, 0.05f, "%.4f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "voxel_size = ratio * sceneDiag.\n"
                        "Bigger = fewer points = faster / riskier.\n"
                        "Start at 0.01 and tune so the count below lands\n"
                        "in the low thousands.");
                }
                if (CmaesRefine::g_silDownsampleFullCount > 0) {
                    ImGui::SameLine();
                    ImGui::TextColored(
                        ImVec4(0.5f, 0.9f, 0.6f, 1.0f), "%ld -> %ld pts",
                        CmaesRefine::g_silDownsampleFullCount,
                        CmaesRefine::g_silDownsampleDownCount);
                } else {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(run Alt+P to populate)");
                }
            }

            // [PERF/EXACT] 1+3 厳密系: 値未使用の computeUnifiedMetrics を退避値
            //  復元で置換 + voxel ダウンサンプル結果のセッション越しキャッシュ。
            //  どちらも結果不変 (OFF/ON で最終 IoU・姿勢がビット一致するはず)。
            ImGui::Checkbox("Alt+P: skip redundant metrics + cache ds (exact)",
                            &CmaesRefine::g_silExactPerf);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "OFF (default): recompute outer metrics every time and\n"
                    "re-downsample per run -- legacy baseline.\n"
                    "ON: (1) skip the 3 computeUnifiedMetrics calls whose\n"
                    "values are never read (restore entry snapshot instead;\n"
                    "pose is byte-identical so values are bit-identical),\n"
                    "(2) reuse the voxel-downsampled cloud across runs\n"
                    "(key = full point-list + voxel equality -- exact).\n"
                    "EXACT: final IoU / pose / PoseLibrary must bit-match\n"
                    "OFF. If they ever differ, report it as a bug.");
            }

            // --- occluded-gate trap (the Alt+P PoseLibrary rejection bug) ---
            //  runShiftE never fills g_lastSilOccludedIoU2D (only Ctrl+Shift+G
            //  Phase E does), so it stays 0. If the accept gate is set to use
            //  occluded IoU, every Alt+P result is rejected (0 vs 0). Surface
            //  the same global here so it can be turned OFF in one place.
            ImGui::Checkbox("Pose accept uses occluded IoU",
                            &g_poseLibraryUseOccludedForAccept);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Same global as the Pose Library panel.\n"
                                  "Alt+P evaluates FULL 2D IoU only and leaves\n"
                                  "occluded IoU = 0. If this is ON, Alt+P saves\n"
                                  "are rejected by the occluded gate (0 vs 0).\n"
                                  "Turn OFF for IoU-only methods like Alt+P.");
            }
            if (g_poseLibraryUseOccludedForAccept) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.4f, 1.0f),
                                   "(Alt+P will be rejected!)");
            }

            if (ImGui::Button("Run Silhouette Align (Alt+P)")) {
                // byte-identical to the Alt+P branch in the GLFW_KEY_P case
                // [QUAD-SIL] incl. the quadrant guard + mask pass-through.
                std::cout << "[Alt+P] dispatching Silhouette Align (button)..."
                          << std::endl;
                if (prepareQuadrantForSilAlign("Alt+P")) {
                    g_stepStartTime = std::chrono::steady_clock::now();
                    poseAutoSaveBeforeRegistration();
                    runShiftE(g_activeQuadrantMask);
                    g_sessionSilhouetteN++;
                    gUI.state.regMethod = 5;
                    saveSilAlignToLibrary();   // [QUAD-SIL ACCEPT] Ctrl+I-parity gate
                }
            }

            // [PERF TOGGLE] Alt+P fast path (default OFF = legacy baseline).
            ImGui::Checkbox("Alt+P fast path (perf; OFF = legacy baseline)",
                            &CmaesRefine::g_silFastPath);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "OFF (default): original Alt+P (serial RMSE, per-eval\n"
                    "target mask rebuild, serial IoU) -- legacy baseline.\n"
                    "ON: parallel fastComputeRMSE + cached target mask +\n"
                    "parallel IoU. Result-invariant, ~2-3x faster/eval.\n"
                    "Re-run Alt+P from the same start pose to verify the\n"
                    "final IoU matches.");

            // [EXPERIMENT] Alt+P accept gate -- the reason it found 0.96 but
            // reverted to ~0.72. OFF/1.20 = legacy.
            ImGui::Checkbox("Alt+P: pure IoU (ignore RMSE accept cap)",
                            &CmaesRefine::g_silAcceptIgnoreRmse);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "OFF (default): legacy gate -- keep found pose only if\n"
                    "compRMSE < before * cap (below). Reverts IoU-optimal\n"
                    "poses that raise 3D RMSE (the 0.96->revert behaviour).\n"
                    "ON: accept on IoU gain alone. Test whether Alt+P keeps\n"
                    "the 0.96 it already finds.");
            if (!CmaesRefine::g_silAcceptIgnoreRmse) {
                ImGui::SliderFloat("Alt+P RMSE accept cap (x before)",
                                   &CmaesRefine::g_silAcceptRmseCap,
                                   1.0f, 5.0f, "%.2f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "accept needs compRMSE_after < compRMSE_before * this.\n"
                        "1.20 = legacy (20%% worsening; old comment said 50%%).\n"
                        "The IoU=0.96 poses need ~2.3x here.");
            }

            ImGui::Checkbox("Alt+P: capture found poses -> F9 (per run)",
                            &g_shiftECaptureFound);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "ON: runShiftE snapshots each run's inner-loop best pose\n"
                    "into an F9 Run slot BEFORE the gate reverts it, applied\n"
                    "pose -> Final. SEE the 0.96 poses the gate discards.\n"
                    "Squash-raster scored at the step below (16 = Ctrl+I /\n"
                    "SilCompare yardstick; panel IoU != full-mesh on purpose).\n"
                    "OFF (default): no capture (runShiftE byte-identical).");
            if (g_shiftECaptureFound) {
                ImGui::SliderInt("Alt+P found capture step",
                                 &g_shiftECaptureStep, 1, 32);
            }

            // --- Silhouette method compare (A/B/C -> F9) ----------------
            //  Runs the EXISTING runShiftE (legacy Alt+P = method 1) and/or
            //  runBipopCmaesV3RS (methods 2/3) from the same start pose,
            //  scores each with the same 2D-IoU yardstick, captures each to
            //  F9 Run 1/2/3. Non-destructive. Alt+P controls above untouched.
            SilCompare::drawSection();

            // --- [PARALLEL] Run-loop parallelization toggle -------------
            //  One switch governs BOTH BIPOP engines' run loops:
            //    - V3RS (Ctrl+I, Ctrl+Shift+G): silhouette engine
            //    - V3R  (Ctrl+G)              : region-aware ICP engine
            //  Parallelizes the 10 restarts across CPU cores (one thread
            //  per run). Determinism is preserved (same poses / IoU /
            //  best-run), so it's a pure speed lever -- toggle OFF to
            //  reproduce the legacy serial timing for A/B comparison.
            ImGui::Separator();
            ImGui::Checkbox("Parallel runs (Ctrl+I / Ctrl+G / Ctrl+Shift+G)",
                            &CmaesRefineV3RS::g_v3rsParallelRuns);
            // Mirror the single UI switch onto the V3R engine's toggle so
            // one checkbox controls both run loops.
            CmaesRefineV3R::g_v3rParallelRuns = CmaesRefineV3RS::g_v3rsParallelRuns;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "OFF = legacy serial run loops (byte-identical results).\n"
                    "ON  = run the 10 BIPOP restarts concurrently, one OpenMP\n"
                    "      thread per run.\n"
                    "  - V3RS (Ctrl+I / Ctrl+Shift+G): inner raster forced\n"
                    "    serial so there's no nested-parallel overhead.\n"
                    "  - V3R (Ctrl+G): no inner OpenMP, so runs fork cleanly.\n"
                    "Same poses / IoU / best-run as serial (jitter is\n"
                    "pre-generated in run order), so only wall-clock changes.\n"
                    "Run a method once each way and compare the\n"
                    "'runs loop wall-clock' console line.");
            }
            // Per-engine last-run readouts (compare without the console).
            if (CmaesRefineV3RS::g_v3rsLastRunLoopMs > 0.0) {
                ImGui::TextDisabled("  V3RS (Ctrl+I / Ctrl+Shift+G) last: %.0f ms  (%s x%d)",
                    CmaesRefineV3RS::g_v3rsLastRunLoopMs,
                    CmaesRefineV3RS::g_v3rsParallelRuns ? "par" : "ser",
                    CmaesRefineV3RS::g_v3rsLastRunThreads);
            }
            if (CmaesRefineV3R::g_v3rLastRunLoopMs > 0.0) {
                ImGui::TextDisabled("  V3R  (Ctrl+G)     last: %.0f ms  (%s x%d)",
                    CmaesRefineV3R::g_v3rLastRunLoopMs,
                    CmaesRefineV3R::g_v3rParallelRuns ? "par" : "ser",
                    CmaesRefineV3R::g_v3rLastRunThreads);
            }

            ImGui::Checkbox("Show RIM pairs",
                            &g_ctrlgShowRimPairs);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Visualize the rim sets used by beta "
                                  "weighting:\n  orange = source RIM (mesh-intrinsic)\n"
                                  "  magenta = target RIM (image boundary).\n"
                                  "Buffers are populated at next Ctrl+G press.");
            }

            // 状態表示: viz バッファの中身が分かるとデバッグしやすい
            if (g_ctrlgRimVizAvailable) {
                ImGui::Text("RimViz buffers: src=%d  tgt=%d",
                            (int)g_ctrlgRimSrcVertIdx.size(),
                            (int)g_ctrlgRimTgtPos.size());
            } else if (g_ctrlgShowRimPairs) {
                ImGui::TextDisabled(
                    "RimViz: press Ctrl+G to populate");
            }

            // -----------------------------------------------------------
            // REDGE (稜線) — experimental  [1][2]  ※表示のみ
            //   RIM とは別に、ドームのオクルーディング輪郭を可視化。
            //   source = 境界 ∩ ANTERIOR_CORE (非RIM・非後面) の grazing 帯。
            //   target = 境界の上半分 (RIM=下半分の対)。
            //   レジストレーション [3] は別途追加予定。
            // -----------------------------------------------------------
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.6f, 0.95f, 1.0f, 1.0f),
                               "REDGE (稜線) — experimental");

            // [1] target ridge (upper half) — yellow
            if (ImGui::Checkbox("[1] Show target REDGE (upper half)",
                                &g_showDebugTargetRidge)) {
                if (g_showDebugTargetRidge) {
                    if (!populateDebugTargetRidge()) g_showDebugTargetRidge = false;
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "yellow = target ridge = boundary UPPER half\n"
                    "(p.y <= 2D centroid.y; RIM is the lower half).\n"
                    "Splits g_debugTargetBoundaryPoints; auto-runs\n"
                    "Shift+W populate if empty. Re-toggle to refresh.");
            }

            // [1] outlier removal sub-controls (debug-only, target 稜線のみ)
            ImGui::Indent(16);
            {
                bool changed = false;
                changed |= ImGui::Checkbox("split by bbox long edge (PCA, 横長対応)",
                                           &g_ridgeTgtSplitLongEdge);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "OFF = simple up/down (p.y <= centroid.y).\n"
                        "ON  = split parallel to the 2D long axis (PCA):\n"
                        "the dividing line follows the elongated silhouette,\n"
                        "so a wide/tilted liver splits into upper(ridge) /\n"
                        "lower(RIM) sensibly. Reduces to up/down when the\n"
                        "long axis is horizontal.");
                }
                changed |= ImGui::Checkbox("remove outliers (target ridge)",
                                           &g_ridgeTgtRemoveOutliers);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Drop geometric outliers (e.g. the depth-surface\n"
                        "tail that slips through the 2D boundary band).\n"
                        "Applies ONLY to the yellow target ridge; the\n"
                        "purple RIM band / beta weighting are untouched.\n"
                        "Default OFF so raw stays the baseline.");
                }
                if (g_ridgeTgtRemoveOutliers) {
                    changed |= ImGui::RadioButton("SOR", &g_ridgeTgtOutlierMode, 0);
                    ImGui::SameLine();
                    changed |= ImGui::RadioButton("largest CC", &g_ridgeTgtOutlierMode, 1);
                    if (g_ridgeTgtOutlierMode == 0) {
                        ImGui::SetNextItemWidth(140);
                        changed |= ImGui::SliderInt("SOR k", &g_ridgeTgtSorK, 4, 48);
                        ImGui::SetNextItemWidth(140);
                        changed |= ImGui::SliderFloat("SOR std x sigma",
                                                      &g_ridgeTgtSorStd, 0.5f, 4.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Larger = more permissive (drops fewer).\n"
                                              "Smaller = aggressive (drops the tail harder).");
                        }
                    } else {
                        ImGui::SetNextItemWidth(140);
                        changed |= ImGui::SliderFloat("CC radius (0=auto)",
                                                      &g_ridgeTgtCcRadius, 0.0f, 5.0f, "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Neighbor radius for Euclidean clustering.\n"
                                              "0 = auto (median NN dist * 2.5).");
                        }
                        ImGui::SetNextItemWidth(140);
                        changed |= ImGui::SliderInt("CC min pts", &g_ridgeTgtCcMinPts, 1, 500);
                    }
                }
                // live 反映: [1] が ON のとき変更で即再 populate (切替を見やすく)
                if (changed && g_showDebugTargetRidge) {
                    populateDebugTargetRidge();
                }
            }
            ImGui::Unindent(16);

            // [2] source ridge (occluding contour, non-RIM/non-posterior) — cyan
            if (ImGui::Checkbox("[2] Show source REDGE (boundary, non-RIM/non-posterior)",
                                &g_showDebugSourceRidge)) {
                if (g_showDebugSourceRidge) {
                    if (!populateDebugSourceRidge()) g_showDebugSourceRidge = false;
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "cyan = source ridge = ANTERIOR_CORE verts whose\n"
                    "normal is grazing to the AR camera (|cos|<band) =\n"
                    "the dome's occluding contour. Excludes RIM and\n"
                    "POSTERIOR by label. View-dependent: computed at\n"
                    "toggle for the current pose; re-toggle after moving.");
            }
            ImGui::SetNextItemWidth(160);
            if (ImGui::SliderFloat("REDGE cos band", &g_ctrlgRidgeCosBand,
                                   0.05f, 0.60f, "%.2f")) {
                if (g_showDebugSourceRidge) populateDebugSourceRidge();  // live 反映
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Source silhouette band width:\n"
                                  "|cos(normal, -view)| < band counts as ridge.\n"
                                  "Larger = thicker ridge band.");
            }

            if (g_showDebugSourceRidge || g_showDebugTargetRidge) {
                ImGui::Text("REDGE buffers: src=%d  tgt=%d",
                            (int)g_debugSourceRidge.size(),
                            (int)g_debugTargetRidgePoints.size());
            }

            // [3] REDGE -> registration weight (Ctrl+G / V3R).
            //   w_i = 1 + beta*(rim_src & rim_tgt) + gamma*(redge_src & rim_tgt)
            //   gamma=0 -> 従来の Ctrl+G と完全一致。次の Ctrl+G から反映。
            ImGui::SetNextItemWidth(180);
            ImGui::SliderFloat("[3] REDGE weight gamma (Ctrl+G)",
                               &g_ctrlgGammaRedgeWeight, 0.0f, 10.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Ridge-ridge weight added to Ctrl+G (V3R) on top of\n"
                    "beta. 0 = OFF (registration byte-identical to before).\n"
                    "Source = ANTERIOR_CORE grazing contour (uses REDGE cos\n"
                    "band above); target = same boundary band as the rim term.\n"
                    "Source rim/redge are disjoint, so each correspondence\n"
                    "gets at most ONE boost (no double counting). Takes effect\n"
                    "on the NEXT Ctrl+G. Log: [Ctrl+G/rim] beta=.. gamma=..\n"
                    "src_redge=..  and  [V3R-W] weighting: ...");
            }

            // [3b] bidirectional (symmetric) matching for rim/redge ((B)方式).
            ImGui::Checkbox("bidirectional rim/redge matching (Ctrl+G)",
                            &g_ctrlgBidirMatching);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Add the reverse term (source -> nearest target) for\n"
                    "RIM/REDGE source points ONLY, so those classes are\n"
                    "constrained BOTH ways. Resists the source ballooning up\n"
                    "in scale to cover the target (one-directional matching\n"
                    "only rewards coverage, never penalizes overshoot).\n"
                    "Plain (non rim/redge) points stay one-directional.\n"
                    "OFF = byte-identical. Needs beta>0 or gamma>0.\n"
                    "Takes effect on the NEXT Ctrl+G. Log: [V3R] bidir\n"
                    "src_to_eval: N reverse corr ...");
            }
            if (g_ctrlgBidirMatching) {
                ImGui::Indent(16);
                ImGui::Checkbox("...also non rim/redge points ((A) full symmetric)",
                                &g_ctrlgBidirAllPoints);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "(A) Extend the reverse term to ALL subset source\n"
                        "points, not just rim/redge. Penalizes the whole\n"
                        "surface's overshoot (closer to symmetric ICP), so\n"
                        "the source should shrink further toward 1.0x.\n"
                        "Unchecked = (B) rim/redge only (current behavior).");
                }
                ImGui::Unindent(16);
            }

            // -----------------------------------------------------------
            // [Phase D] Colored RIM pairs (K representatives).
            //   Independent from Show RIM pairs above. Shows K paired
            //   source+target spheres in matching HSV colors so the
            //   operator can see WHICH rim vertex maps to WHICH target
            //   point at the current pose. Pairs are sampled from the
            //   ~20k captured at Ctrl+G Phase F.5 (or restored from a
            //   PoseLibrary entry after Apply). 4 sampling modes:
            //     - ArcUniform: even spacing around tgt centroid (default)
            //     - WorstK    : K longest src-tgt distances (diagnostic)
            //     - BestK     : K shortest distances
            //     - Random    : seeded uniform (reshufflable)
            // -----------------------------------------------------------
            ImGui::Checkbox("Show colored pairs (K)",
                            &g_ctrlgShowColoredRimPairs);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Overlay K representative rim-rim pairs "
                                  "drawn in matching HSV colors so each "
                                  "src↔tgt mapping is visually identifiable.\n"
                                  "Data source: g_lastRimPair* (set by "
                                  "Ctrl+G / Ctrl+Shift+G Phase F.5; restored "
                                  "by Pose Library Apply).\n"
                                  "Pairs follow the mesh through subsequent "
                                  "ICP/Apply (source is a full-mesh vertex "
                                  "index; target is fixed world coords).\n"
                                  "Independent from Show RIM pairs above — "
                                  "leave both ON for max info.");
            }
            if (g_ctrlgShowColoredRimPairs) {
                ImGui::Indent();
                ImGui::SliderInt("K pairs", &g_ctrlgColoredRimN, 5, 30);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("How many representative pairs to draw "
                                      "(5–30). 10 is the sweet spot — small "
                                      "enough to differentiate by HSV hue, "
                                      "large enough to span the rim.");
                }
                const char* modeItems =
                    "ArcUniform\0WorstK\0BestK\0Random\0";
                ImGui::Combo("Sample mode",
                             &g_ctrlgColoredRimMode, modeItems);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "ArcUniform : evenly spaced around tgt centroid "
                        "(stable across runs — default).\n"
                        "WorstK     : K longest src↔tgt distances "
                        "(diagnostic: where is the rim misaligned?).\n"
                        "BestK      : K shortest distances "
                        "(sanity check).\n"
                        "Random     : seeded uniform sample. Use the "
                        "Reshuffle button to draw a new sample.");
                }
                // Reshuffle only affects Random mode (other modes are
                // deterministic). Disable the button outside Random so
                // the UI signals this clearly instead of accepting clicks
                // that produce no visible change.
                const bool reshuffleActive =
                    (g_ctrlgColoredRimMode ==
                     (int)RimPairSampling::Mode::Random);
                if (!reshuffleActive) ImGui::BeginDisabled();
                if (ImGui::Button("Reshuffle")) {
                    g_ctrlgColoredRimSeed++;
                }
                if (!reshuffleActive) ImGui::EndDisabled();
                if (ImGui::IsItemHovered()) {
                    if (reshuffleActive) {
                        ImGui::SetTooltip(
                            "Draw a new K-sample with a fresh seed.\n"
                            "Active only in Random mode.");
                    } else {
                        ImGui::SetTooltip(
                            "Reshuffle is only available in Random mode.\n"
                            "ArcUniform / WorstK / BestK are deterministic — "
                            "pressing this button would have no effect.");
                    }
                }
                ImGui::SameLine();
                if (!g_lastRimPairSrcVertIdx.empty()) {
                    ImGui::TextDisabled("(%d pairs avail.)",
                                        (int)g_lastRimPairSrcVertIdx.size());
                } else {
                    ImGui::TextDisabled("(no pairs — press Ctrl+G)");
                }
                ImGui::Unindent();
            }

            // -----------------------------------------------------------
            //  Ctrl+Shift+G (V3-RS, silhouette anchor) - inline section.
            //  Placed inside the Ctrl+G panel so it is always visible
            //  alongside beta / AR-vis / Caudal. These controls are
            //  read only when Ctrl+Shift+G is pressed; plain Ctrl+G
            //  ignores them entirely.
            // -----------------------------------------------------------
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.5f, 1.0f),
                               "Ctrl+Shift+G silhouette anchor");

            ImGui::SliderFloat("lambda_sil",
                               &g_ctrlgsLambdaSil, 0.0f, 1.0f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Silhouette anchor strength for Ctrl+Shift+G.\n"
                    "  0.00 : V3R-W behaviour (use Ctrl+G instead).\n"
                    "  0.10 : weak.\n"
                    "  0.30 : recommended starting point.\n"
                    "  1.00 : silhouette-dominant.\n"
                    "Plain Ctrl+G ignores this slider.");
            }

            // ----- [NEW UI-1a] Asymmetric outside-ratio penalty -------
            ImGui::Checkbox("Asymmetric outside-ratio penalty (mask-expansion brake)",
                            &g_ctrlgsUseOutsideRatio);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Add an ASYMMETRIC penalty to the Ctrl+Shift+G cost:\n"
                    "  cost += lambda_out * (source AND NOT target) / source\n"
                    "\n"
                    "Symmetric (1-IoU) penalises source-contains-target only\n"
                    "weakly: a fully-containing source has IoU < 1 but the\n"
                    "gradient toward shrinking is small. This term directly\n"
                    "measures the FRACTION of source raster outside the\n"
                    "target, putting a one-sided pull toward source-in-target.\n"
                    "  0   : source is inside target (no penalty).\n"
                    "  1   : no overlap (max penalty).\n"
                    "\n"
                    "Default OFF -- byte-identical to pre-feature behaviour.\n"
                    "Recommended when Ctrl+G has drifted into mask expansion.");
            }
            ImGui::SliderFloat("lambda_out",
                               &g_ctrlgsLambdaOut, 0.0f, 2.0f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Weight for the asymmetric outside-ratio penalty.\n"
                    "  0.0  : no effect (same as checkbox OFF).\n"
                    "  0.3  : weak brake on mask expansion.\n"
                    "  0.5  : recommended starting point.\n"
                    "  1.0+ : aggressive shrink toward source inside target.\n"
                    "Only active when the checkbox above is ON.");
            }

            // ----- [NEW UI-1b] RIM silhouette penalty ----------------
            ImGui::Checkbox("RIM silhouette penalty (boundary-to-boundary)",
                            &g_ctrlgsUseRimSil);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Add a boundary-alignment penalty to the cost:\n"
                    "  cost += lambda_rim_sil * mean(dist_to_target_boundary)\n"
                    "\n"
                    "Evaluated only at SOURCE-BOUNDARY raster cells (source\n"
                    "cells with at least one non-source 4-neighbour). For\n"
                    "each such cell:\n"
                    "  outside target mask -> contribute 1.0 (max).\n"
                    "  inside target mask  -> contribute min(d/max_px, 1.0)\n"
                    "where d is the image-pixel distance to the target\n"
                    "silhouette boundary, from the SAM2 distance map.\n"
                    "\n"
                    "Silhouette-space analogue of Ctrl+G's beta-rim weighting:\n"
                    "forces source RIM to target RIM coincidence rather than\n"
                    "mere area overlap. Catches drift patterns where source\n"
                    "covers target area well but with bulges/dents at the rim.\n"
                    "\n"
                    "Default OFF.");
            }
            ImGui::SliderFloat("lambda_rim_sil",
                               &g_ctrlgsLambdaRimSil, 0.0f, 2.0f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Weight for the RIM silhouette penalty.\n"
                    "  0.0  : no effect.\n"
                    "  0.2  : weak.\n"
                    "  0.3  : recommended starting point.\n"
                    "  1.0+ : boundary-dominant.\n"
                    "Only active when 'RIM silhouette penalty' is ON.");
            }
            ImGui::SliderFloat("rim_sil_max_px",
                               &g_ctrlgsRimSilMaxPx, 10.0f, 300.0f, "%.0f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Image-pixel normalisation cap for the RIM penalty.\n"
                    "Source-boundary cells AT the target boundary contribute\n"
                    "0; cells >= max_px away from the boundary saturate to 1.\n"
                    "  50  : tight (small drift heavily penalised).\n"
                    "  100 : recommended starting point.\n"
                    "  200 : loose (only large drift penalised).\n"
                    "Only active when 'RIM silhouette penalty' is ON.");
            }

            // [NEW UI-RIM-ANAT] Anatomic-mode toggle
            ImGui::Checkbox("Use anatomical RIM (vs. raster boundary)",
                            &g_ctrlgsRimSilAnatomic);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Switch the source-rim definition between two modes:\n"
                    "\n"
                    "OFF (legacy raster boundary):\n"
                    "  Source RIM = every cell on the rasterised silhouette\n"
                    "  outline (4-neighbour test on the source hitmap).\n"
                    "  Includes the outlines of detached blobs / artefacts;\n"
                    "  doesn't know anything about anatomy. Pure geometric.\n"
                    "\n"
                    "ON (anatomical RIM):\n"
                    "  Source RIM = vertices labelled LiverRegionLabel::RIM,\n"
                    "  filtered by quadrant + AR-vis + Caudal the SAME way\n"
                    "  the Ctrl+G 'Show RIM pairs' checkbox filters them.\n"
                    "  These are exactly the orange spheres you see in the\n"
                    "  AR view when RimViz is enabled. rim_sil is the mean\n"
                    "  distance from each VISIBLE projected RIM vertex to\n"
                    "  the target silhouette boundary.\n"
                    "\n"
                    "F9 viz: in anatomic mode, panels 4 & 6 highlight cells\n"
                    "where any anatomical RIM vertex projected, NOT the full\n"
                    "silhouette outline. Lets you check whether the rim\n"
                    "Ctrl+G already cares about coincides with the SAM2\n"
                    "boundary.\n"
                    "\n"
                    "Only active when 'RIM silhouette penalty' is ON.\n"
                    "Default OFF -- legacy behaviour.");
            }
            if (g_ctrlgsRimSilAnatomic && !g_ctrlgsUseRimSil) {
                ImGui::TextColored(
                    ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
                    "  NOTE: anatomic toggle ON but rim_sil penalty OFF -- has no effect");
            }

            // ----- [NEW UI-1c] Dynamic RMSE cap for Phase E ----------
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                               "Phase E RMSE acceptance cap");
            ImGui::Checkbox("Dynamic cap (loosen on IoU gain)",
                            &g_ctrlgsUseDynamicCap);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Layer 3 (Phase E) rejects candidates whose RMSE exceeds\n"
                    "rmse_before * cap_factor. The legacy fixed cap (1.05x)\n"
                    "can block silhouette-improving candidates whose RMSE\n"
                    "rose 6-10%% while IoU jumped 0.05-0.10 -- exactly the\n"
                    "recovery move we WANT after Ctrl+G mask expansion.\n"
                    "\n"
                    "OFF: cap = RmseCapBase (legacy 1.05x behaviour).\n"
                    "ON : cap interpolates linearly between RmseCapBase\n"
                    "     (at diou=0) and RmseCapMax (at diou>=DiouFull).\n"
                    "     diou is the IoU gain reported in the ACCEPTED /\n"
                    "     REJECTED log line.\n"
                    "\n"
                    "Default OFF -- preserves legacy behaviour.");
            }
            ImGui::SliderFloat("RmseCapBase",
                               &g_ctrlgsRmseCapBase, 1.00f, 1.20f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Cap factor with no IoU improvement (and the only one\n"
                    "used when Dynamic cap is OFF).\n"
                    "  1.00 : strict (RMSE cannot increase at all).\n"
                    "  1.05 : legacy default (5%% tolerance).\n"
                    "  1.20 : very lenient.\n"
                    "Always active.");
            }
            ImGui::SliderFloat("RmseCapMax",
                               &g_ctrlgsRmseCapMax, 1.00f, 1.50f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Cap factor when IoU gain saturates (diou>=DiouFull).\n"
                    "  1.05 : same as base (no loosening).\n"
                    "  1.15 : recommended starting point.\n"
                    "  1.30 : aggressive recovery from mask expansion.\n"
                    "Only effective when 'Dynamic cap' is ON.");
            }
            ImGui::SliderFloat("RmseCapDiouFull",
                               &g_ctrlgsRmseCapDiouFull, 0.00f, 0.20f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "IoU gain at which the dynamic cap reaches RmseCapMax.\n"
                    "Linear interpolation: (diou=0, cap=Base) to\n"
                    "(diou>=DiouFull, cap=Max).\n"
                    "  0.02 : aggressive loosening on tiny IoU gains.\n"
                    "  0.05 : recommended starting point.\n"
                    "  0.10 : conservative -- only big IoU gains relax cap.\n"
                    "Only effective when 'Dynamic cap' is ON.");
            }
            ImGui::Separator();

            // Target-mask squash toggle. When ON, the SAM2 target mask
            // is rasterized through the same triangle-bbox + 1-cell halo
            // system the source mesh uses, so IoU compares equally-
            // inflated shapes (fair). When OFF, target uses legacy
            // per-cell centre sample (asymmetric -- the diagnostic
            // path for A/B comparison). Default ON.
            ImGui::Checkbox("Target squash (source-parity raster)",
                            &CmaesRefineV3RS::g_silTargetSquashEnabled);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Rasterize the SAM2 target mask through the SAME\n"
                    "raster system the source mesh uses (step x step OR\n"
                    "coverage + 1-cell halo). Removes the asymmetric bias\n"
                    "that was capping IoU around 0.63.\n"
                    "\n"
                    "  ON  (default): fair, cached app-wide.\n"
                    "  OFF          : legacy centre-sample (A/B reference).\n"
                    "\n"
                    "Cache survives across Ctrl+Shift+G presses and is\n"
                    "rebuilt only when the SAM2 mask changes.");
            }

            // Instrument occlusion filter (NEW). When ON, grid cells
            // covered by instruments (per g_instrumentDistMap) are
            // excluded from BOTH union and intersection of the IoU
            // computation. Fixes the asymmetric error where the source
            // mesh extends behind an instrument occluder but the SAM2
            // target mask correctly has no liver there. Default OFF so
            // pre-feature behaviour is preserved byte-for-byte.
            ImGui::Checkbox("Ignore instrument-occluded pixels",
                            &g_ctrlgsIgnoreInstrument);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Exclude IoU grid cells that lie under an instrument\n"
                    "(rasterized through g_instrumentDistMap).\n"
                    "\n"
                    "Why: when the source mesh projects onto an area\n"
                    "covered by a tool, SAM2 correctly has NO liver mask\n"
                    "there. Without this filter, the source overshoot in\n"
                    "that occluded area is counted as IoU loss, biasing\n"
                    "the optimiser toward poses that shrink the mesh\n"
                    "behind tools (containment failure variant).\n"
                    "\n"
                    "Requires instrument_segmentation_mask.png to exist\n"
                    "and match the liver-mask dimensions. If unavailable,\n"
                    "the filter is silently disabled for that session.\n"
                    "\n"
                    "Default OFF -- pre-feature behaviour preserved.");
            }

            ImGui::SliderFloat("instrument ignore thresh [px]",
                               &g_ctrlgsInstrumentThreshPx,
                               3.0f, 20.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Pixel-distance threshold for the instrument filter.\n"
                    "Cells whose centre pixel has inst_dist < thresh are\n"
                    "excluded.\n"
                    "\n"
                    "  0.0  : exclude only INSIDE the instrument region.\n"
                    "  5.0  : also exclude within 5 px of the boundary\n"
                    "         (compensates for SAM2 mask edge slop).\n"
                    "         Recommended starting point.\n"
                    " 10.0+ : aggressive; risk of dropping legitimate\n"
                    "         silhouette near tools.\n"
                    "\n"
                    "Only active when the checkbox above is ON.");
            }
            if (g_ctrlgsIgnoreInstrument && !g_instrumentDistMap.valid) {
                ImGui::TextColored(
                    ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
                    "  WARNING: instrument mask not loaded -- filter inactive");
            }

            ImGui::Checkbox("Show sil projection (after Ctrl+Shift+G)",
                            &g_silProjShow);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Draw the rim ∩ quadrant subset (the points the\n"
                    "silhouette loss actually evaluates) as coloured\n"
                    "spheres in the AR view. No subsampling -- you see\n"
                    "every point the optimiser was scored against.\n"
                    "Colour scale by 2D boundary distance:\n"
                    "  RED  : projection OUTSIDE the SAM mask\n"
                    "         -> contributes image-diagonal penalty.\n"
                    "  GREEN  (< 5 px)   : on the silhouette boundary\n"
                    "  YELLOW (5-30 px)  : near the boundary\n"
                    "  BLUE   (>= 30 px) : inside the mask\n"
                    "                       (rim voxel that drifted in)\n"
                    "Captured once per Ctrl+Shift+G press.");
            }

            if (g_silProjDebug.valid) {
                const int n_in  = g_silProjDebug.n_with_signal;
                const int n_tot = g_silProjDebug.n_visible;
                const int n_out = std::max(0, n_tot - n_in);
                const float pct_out = (n_tot > 0)
                    ? 100.0f * (float)n_out / (float)n_tot : 0.0f;
                ImGui::Text("  sil viz: %d pts  mean_dist=%.1f px (%.3f norm)",
                            (int)g_silProjDebug.pts.size(),
                            g_silProjDebug.mean_dist_px,
                            g_silProjDebug.mean_dist_norm);
                ImGui::Text("  out-of-mask: %d / %d (%.1f%%)  in-mask: %d",
                            n_out, n_tot, pct_out, n_in);
            } else {
                ImGui::TextDisabled(
                    "  sil viz: (press Ctrl+Shift+G to populate)");
            }

            if (g_silProjShow && !g_boundaryDistMap.valid) {
                ImGui::TextColored(
                    ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
                    "  WARNING: g_boundaryDistMap invalid - sil will be skipped");
            }

            ImGui::Separator();
            ImGui::TextDisabled("Press Ctrl+G to run V3-R with this selection");
            ImGui::TextDisabled("Press Ctrl+Shift+G to run V3-RS (silhouette anchor)");
            ImGui::TextDisabled("Press Shift+N / Ctrl+Shift+N for Normal-Compat / SRT polish");

            // [PHASE-5] F9 silhouette IoU toggle — also available from W tab.
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.85f, 1.0f, 0.7f, 1.0f),
                               "Silhouette IoU diagnostic:");
            if (ImGui::Button("F9: Toggle Silhouette IoU window##g_f9")) {
                SilOverlay::g_silOverlay.showWindow =
                    !SilOverlay::g_silOverlay.showWindow;
            }
            ImGui::SameLine();
            ImGui::TextDisabled(SilOverlay::g_silOverlay.showWindow
                                    ? "(currently open)"
                                    : "(currently closed)");

        };  // end g_debugPanel.drawGBody (migrated Ctrl+G Quadrant Selector)

        // ----------------------------------------------------------------
        //  Normal-Compatible Refine (Shift+N) panel
        //  -------------------------------------------------------------
        //  Companion panel to "Ctrl+G Quadrant Selector". Configures the
        //  finishing-pass refinement that runs after Ctrl+G:
        //    Apply Init Pose → Ctrl+P → Ctrl+G → Shift+N (polish)
        //                                       ↘ Ctrl+Shift+N (SRT-Variance polish)
        //
        //  Source/target filters are SHARED with Ctrl+G (AR-vis / Caudal /
        //  Quadrant globals are read by the wrapper directly). Only the
        //  rim weights (Phase 2 L1) and anchor controls (Phase 3 L2) live
        //  here, plus per-iteration solver knobs.
        // ----------------------------------------------------------------
        // [PHASE-4] Normal-Compatible Refine content relocated into Debug Panel
        // > N tab. Registered as a hook (lambda capturing frame-loop locals by
        // reference); rendered by DebugPanel::draw() under the same guard.
        g_debugPanel.drawNBody = [&]() {

            ImGui::TextColored(ImVec4(0.7f, 0.95f, 0.7f, 1.0f),
                               "Finishing-pass refinement after Ctrl+G");
            ImGui::Spacing();

            // Master enable.
            ImGui::Checkbox("Enabled", &g_normRefineEnabled);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Master switch. When OFF, pressing Shift+N or "
                    "Ctrl+Shift+N logs the abort and does not touch the\n"
                    "pose. Useful for quickly disabling the feature\n"
                    "without unbinding the key.");
            }

            // ---- Live mode toggle (Phase 6) -----------------------------
            //   ON  : object-tracking-style visualisation — mesh moves
            //         frame-by-frame as the optimisation progresses.
            //   OFF : blocking wrapper — mesh snaps to final pose at
            //         the end of a 4-8s pause. Same math either way.
            ImGui::SameLine();
            ImGui::Checkbox("Live mode", &g_normRefineLiveMode);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON  (default): frame-driven refinement.\n"
                    "  Each render frame runs refineStep, the mesh\n"
                    "  visibly moves toward the target. Like an SRT-3D\n"
                    "  object tracker. Press Shift+N again to stop.\n"
                    "\n"
                    "OFF: blocking wrapper.\n"
                    "  The whole loop runs in one frame; the mesh moves\n"
                    "  once at the end. Faster total wall-clock but no\n"
                    "  intermediate visualisation. Useful when you only\n"
                    "  care about the final pose.");
            }
            // [Phase 6 UX] Steps/frame slider — directly controls
            //   animation speed. Only meaningful when Live is ON.
            if (g_normRefineLiveMode) {
                ImGui::SliderInt("Steps/frame",
                                 &g_normRefineLiveStepsPerFrame, 1, 10);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "How many refineStep calls per render frame in\n"
                        "Live mode. Each refineStep does itersPerFrame\n"
                        "internal sub-iterations (default 2).\n"
                        "  1  : slowest, most dramatic animation.\n"
                        "  3-5: faster but motion still visible.\n"
                        "  10 : effectively blocking spread over frames.");
                }
            }

            // ---- LIVE running banner + Cancel button --------------------
            if (g_normRefineLiveActive) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f),
                                   "● LIVE TRACKING — iter %d / %d",
                                   g_normRefineLastIter,
                                   g_normRefineMaxIter);
                if (g_normRefineLiveCurrentRMSE >= 0.0f) {
                    ImGui::Text("  current RMSE = %.5f   best = %.5f",
                                g_normRefineLiveCurrentRMSE,
                                g_normRefineLastBestRMSE);
                }
                ImGui::Text("  initial RMSE = %.5f   gain = %.1f%%",
                            g_normRefineLastInitialRMSE,
                            g_normRefineLastInitialRMSE > 0.0f
                                ? 100.0f * (1.0f - g_normRefineLastBestRMSE
                                                 / g_normRefineLastInitialRMSE)
                                : 0.0f);
                if (g_normRefineLiveAnchorPhase == 1) {
                    ImGui::TextColored(ImVec4(0.96f, 0.82f, 0.30f, 1.0f),
                                       "  anchor phase ACTIVE");
                } else if (g_normRefineLiveAnchorPhase == 0) {
                    ImGui::TextDisabled("  anchor phase ended (pure NN)");
                }
                ImGui::TextDisabled("  (press Shift+N to stop early)");
                if (ImGui::Button("Cancel (revert)", ImVec2(-1, 0))) {
                    cancelNormalCompatRefineLive("user-cancel-button");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Force-abort the live session, revert the mesh\n"
                        "to its pre-press pose, and save a REJECTED\n"
                        "entry to the Pose Library.");
                }
            }

            // ---- Source filter mirror (read-only status, lives in Ctrl+G panel) ----
            ImGui::Separator();
            ImGui::TextDisabled("Source filter (shared with Ctrl+G panel):");
            {
                const auto maskStr = LiverLeftRightLabel::quadrantMaskString(
                    g_activeQuadrantMask);
                ImGui::Text("  quadrant = %s  (0x%X)",
                            maskStr.c_str(), (unsigned)g_activeQuadrantMask);
                ImGui::Text("  AR-vis = %s,  Caudal-only = %s",
                            g_ctrlgUseArVisFilter ? "ON" : "OFF",
                            g_ctrlgUseCaudalOnly  ? "ON" : "OFF");
                if (g_ctrlgUseArVisFilter && g_ctrlgUseCaudalOnly) {
                    ImGui::Text("  combine = %s",
                                g_ctrlgArvisCaudalCombine == 1 ? "OR" : "AND");
                }
            }

            // ---- Per-iteration solver knobs ----
            ImGui::Separator();
            ImGui::TextDisabled("Solver");
            ImGui::SliderFloat("distanceThreshold",
                               &g_normRefineDistThresh, 0.01f, 0.50f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Sigmoid centre for the per-correspondence weight.\n"
                    "Larger -> more far points contribute (good when ICP\n"
                    "needs long-range attraction). Scales with scene size.\n"
                    "Header default 0.15.");
            }
            ImGui::SliderFloat("minNormalCos",
                               &g_normRefineMinNormalCos, 0.0f, 0.9f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Annealed start cosine threshold. NormalCompat\n"
                    "(Shift+N) ignores this; SRT_Variance (Ctrl+Shift+N)\n"
                    "uses it to reject correspondences whose normals\n"
                    "disagree more than acos(this) radians at iter 0.\n"
                    "Header default 0.30 (~72.5° accepted at iter 0).");
            }
            ImGui::SliderInt("maxIter", &g_normRefineMaxIter, 20, 500);
            ImGui::SliderInt("itersPerFrame", &g_normRefineItersPerFrame, 1, 5);

            // ---- Phase 7a: Pure RIM mode -----------------------------
            //   HARD filter (vs L1's soft weight). When ON, only rim-to-rim
            //   correspondences are used. Source = liver verts AND
            //   LiverRegionLabel::RIM (intersected with current quadrant
            //   / AR-vis / Caudal selection). Target = boundaryDist <
            //   g_ctrlgRimTgtThreshPx (instrument-aware).
            //
            //   Recommended workflow: enable AFTER Ctrl+M (Shape Match,
            //   Phase 7b) has aligned the rims globally, then Shift+N
            //   polishes the residual using rim-only ICP.
            ImGui::Separator();
            ImGui::Checkbox("RIM-only mode (hard filter)", &g_normRefinePureRim);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Phase 7a: restrict refinement to RIM-to-RIM matches.\n"
                    "\n"
                    "Source = liver verts AND LiverRegionLabel::RIM\n"
                    "         (intersected with current Ctrl+G filters)\n"
                    "Target = boundaryDist < g_ctrlgRimTgtThreshPx\n"
                    "         (instrument-aware)\n"
                    "\n"
                    "Pros: 4-5x faster, focuses on the curve we care about.\n"
                    "Cons: rim is roughly 1D, so in-plane rotation is\n"
                    "      under-constrained — works best AFTER a good\n"
                    "      initial alignment (Ctrl+G or Ctrl+M).\n"
                    "\n"
                    "Independent of beta_rim_src/tgt (L1 weight). L1 betas\n"
                    "still apply AMONG the rim points that survive the\n"
                    "filter when both are on.");
            }
            if (g_normRefinePureRim) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "[ACTIVE]");
            }

            // ---- Phase 2 L1: Rim multiplicative weights ----
            ImGui::Separator();
            ImGui::TextDisabled("L1: Rim weight (Phase 2)");
            ImGui::SliderFloat("beta_rim_src",
                               &g_normRefineBetaRimSrc, 0.0f, 3.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Multiplicative boost for source-rim correspondences.\n"
                    "  0.0  : source-rim points get the same weight as\n"
                    "         interior points (header byte-identical).\n"
                    "  1.0  : rim points contribute DOUBLE.\n"
                    "  3.0  : rim points contribute 4x.\n"
                    "Source rim = LiverRegionLabel::RIM (mesh-intrinsic),\n"
                    "available iff g_liverRegion is computed.");
            }
            ImGui::SliderFloat("beta_rim_tgt",
                               &g_normRefineBetaRimTgt, 0.0f, 3.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Multiplicative boost for target-rim correspondences.\n"
                    "Target rim = boundaryDist < g_ctrlgRimTgtThreshPx\n"
                    "(image-side rim membership; same threshold the\n"
                    "Ctrl+G panel uses, so changing the slider there\n"
                    "also changes what counts as 'rim' here).");
            }
            ImGui::TextDisabled("Tgt rim threshold (Ctrl+G panel): %.1f px",
                                g_ctrlgRimTgtThreshPx);

            // ---- Phase 3 L2: Anchor pair carry-over from last Ctrl+G ----
            ImGui::Separator();
            ImGui::TextDisabled("L2: Anchor pair (Phase 3)");

            const bool anchorAvail =
                !g_lastRimPairSrcVertIdx.empty() &&
                (g_lastRimPairSrcVertIdx.size() == g_lastRimPairTgtPos.size());
            if (anchorAvail) {
                ImGui::TextColored(ImVec4(0.7f, 0.95f, 0.7f, 1.0f),
                                   "  %d anchor pairs available",
                                   (int)g_lastRimPairSrcVertIdx.size());
            } else {
                ImGui::TextColored(ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
                                   "  No anchor pairs (run Ctrl+G first)");
            }

            ImGui::Checkbox("Use anchor pairs", &g_normRefineUseAnchor);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Use the rim-pair correspondences captured by the\n"
                    "most-recent Ctrl+G / Ctrl+Shift+G (Phase F.5) as a\n"
                    "warm start. During the first anchorPhaseIter outer\n"
                    "iterations, anchored vertices use the anchor target\n"
                    "instead of the runtime KDTree nearest neighbour.\n"
                    "After the phase ends, anchors are silently dropped\n"
                    "and the loop converges via pure NN.\n"
                    "\n"
                    "When NO anchors are available (e.g. fresh session),\n"
                    "this toggle has no effect — the wrapper passes empty\n"
                    "anchor arrays and runs in pure-NN mode.");
            }
            ImGui::SliderInt("anchorPhaseIter",
                             &g_normRefineAnchorPhaseIter, 0, 100);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Number of outer iterations during which anchors\n"
                    "override KDTree NN for vertices that own an anchor.\n"
                    "0 = effectively disable anchors; 20 = default;\n"
                    "100 = run anchored to the very end (rarely useful).");
            }
            ImGui::SliderFloat("anchorBlend",
                               &g_normRefineAnchorBlend, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Blend factor inside the anchor phase:\n"
                    "  1.0 : pure anchor (target = anchor position).\n"
                    "  0.5 : midpoint between anchor and current NN.\n"
                    "  0.0 : ignore anchor entirely (= toggle OFF).");
            }

            // ---- Last-session status ----
            ImGui::Separator();
            ImGui::TextDisabled("Status (last session)");
            if (g_normRefineLastInitialRMSE >= 0.0f) {
                const char* methodName =
                    (g_normRefineLastMethod == 1)
                        ? "SRT_VARIANCE" : "NORMAL_COMPAT";
                ImGui::Text("  method: %s", methodName);
                ImGui::Text("  iter: %d / %d   initialRMSE: %.5f",
                            g_normRefineLastIter, g_normRefineMaxIter,
                            g_normRefineLastInitialRMSE);
                ImGui::Text("  bestRMSE: %.5f   %s   %s",
                            g_normRefineLastBestRMSE,
                            g_normRefineLastConverged ? "converged" : "max-iter",
                            g_normRefineLastAccepted  ? "ACCEPTED"  : "REJECTED");
            } else {
                ImGui::TextDisabled("  (no session yet — press Shift+N)");
            }

            ImGui::Separator();
            ImGui::TextDisabled("Shift+N: Normal-Compat   |   "
                                "Ctrl+Shift+N: SRT-Variance");

        };  // end g_debugPanel.drawNBody (migrated Normal-Compatible Refine)

        // [PHASE-7] ScreenMesh Display content relocated into Debug Panel > Viz
        // tab (rendered after the Phase-2 Cluster/CorresPoints section).
        // Registered as a hook; also surfaces the B/N-key visualization toggles.
        // Standalone floating window removed.
        g_debugPanel.drawVizExtra = [&]() {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Other markers:");
            ImGui::Checkbox("Boundary candidates (was B)##viz_b",
                            &g_showBoundaryCandidates);
            ImGui::Checkbox("Source visualization (was N)##viz_n",
                            &g_showSourceVisualization);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Screen mesh rendering:");
            ImGui::Checkbox("Draw as points (lightweight)", &g_screenMeshAsPoints);
            if (g_screenMeshAsPoints) {
                ImGui::SliderFloat("Point size [px]",
                                   &g_screenMeshPointSize,
                                   1.0f, 8.0f, "%.1f");
                ImGui::SliderFloat("Density [%]",
                                   &g_screenMeshDensity,
                                   1.0f, 100.0f, "%.0f");
                if (ImGui::Button("Reshuffle")) {
                    g_screenMeshPC.requestReshuffle();
                }
                // 表示中の頂点数の情報
                const size_t total = g_screenMeshPC.totalVerts;
                if (total > 0) {
                    const size_t drawn = std::max<size_t>(
                        1, (size_t)((double)total
                                  * (double)g_screenMeshDensity / 100.0));
                    ImGui::SameLine();
                    ImGui::Text("(%zu / %zu pts)", drawn, total);
                }
            }
            ImGui::Separator();
            ImGui::Checkbox("Show debug AABB (red=target, green=source)",
                            &g_showDebugBB);
            if (g_showDebugBB) {
                ImGui::TextDisabled("Source AABB is post-Apply state");
                if (g_dbgSourceBB_valid && g_targetAabbFull.valid) {
                    glm::vec3 err = g_dbgSourceBB_center - g_targetAabbFull.center;
                    float d = glm::length(err);
                    ImGui::Text("|err| = %.4f m  (%.1f mm)", d, d * 1000.0f);
                }
            }

            // ---- [Phase 1] viz toggles migrated from keyboard ---------------
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                               "Visualization toggles (formerly keyboard):");

            // [Phase 5.5] Cluster viz (was V) already exists above as
            // drawTabViz's "Cluster markers" (onToggleClusterVis). Duplicate
            // checkbox removed to avoid two toggles for the same global.

            // B key family (cyclic correspondence: pure toggle; needs Shift+P
            // to have run first to have pairs to show).
            ImGui::Checkbox("Cyclic Correspondence - Shift+P pairs (was Shift+B)##viz_cyclic",
                            &g_showCyclicCorrespondence);

            // W key family — enabling must POPULATE the overlay data (mirror the
            // old plain-W / Shift+W keys), otherwise the flag is on but nothing
            // is drawn. On populate failure the toggle is reverted.
            if (ImGui::Checkbox("Debug Source Rim Chain - green dots (was W)##viz_rim_src",
                                &g_showDebugSourceRimChain)) {
                if (g_showDebugSourceRimChain) {
                    if (!g_liverRegion.valid()) recomputeLiverRegion();
                    if (!g_liverLR.valid())     recomputeLiverLR();
                    if (g_ctrlgUseCaudalOnly && !g_liverCC.valid()) recomputeLiverCC();
                    if (!populateDebugSourceRimChain()) g_showDebugSourceRimChain = false;
                }
            }
            if (ImGui::Checkbox("Debug Target Boundary - purple dots (was Shift+W)##viz_rim_tgt",
                                &g_showDebugTargetBoundary)) {
                if (g_showDebugTargetBoundary) {
                    if (!populateDebugTargetBoundary()) g_showDebugTargetBoundary = false;
                }
            }

            // Liver label viz
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Liver labels:");
            if (ImGui::Checkbox("Liver Region (anterior/rim/posterior) - was Shift+R##viz_region",
                                &g_showLiverRegion)) {
                if (g_showLiverRegion && !g_liverRegion.valid()) recomputeLiverRegion();
            }
            if (ImGui::Checkbox("Liver Left/Right - was Y##viz_lr",
                                &g_showLiverLR)) {
                if (g_showLiverLR && !g_liverLR.valid()) recomputeLiverLR();
            }
            if (ImGui::Checkbox("Liver Cranio/Caudal - was Shift+H##viz_cc",
                                &g_showLiverCC)) {
                if (g_showLiverCC && !g_liverCC.valid()) recomputeLiverCC();
            }
            if (ImGui::Checkbox("Liver 4-Quadrant overlay - was H##viz_quad",
                                &g_showLiverQuad)) {
                if (g_showLiverQuad &&
                    g_quadVizIdxAR.empty() && g_quadVizIdxAL.empty() &&
                    g_quadVizIdxPR.empty() && g_quadVizIdxPL.empty()) {
                    recomputeLiverQuad();
                }
            }

            // Recompute buttons (formerly Shift+T / Shift+Y)
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Recompute labels:");
            if (ImGui::Button("Recompute Region  (was Shift+T)##btn_recompute_region")) {
                std::cout << "[Region] recomputing with target_rim_mm = "
                          << g_rimTargetMm << std::endl;
                recomputeLiverRegion();
            }
            ImGui::SameLine();
            if (ImGui::Button("Recompute LR  (was Shift+Y)##btn_recompute_lr")) {
                std::cout << "[LR] recomputing  right_pure_fraction = "
                          << g_lrPureFrac << "  right_full_fraction = "
                          << g_lrFullFrac << std::endl;
                recomputeLiverLR();
            }

            // Debug dumps (formerly Shift+I / F10)
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Debug dumps:");
            if (ImGui::Button("Dump IoU debug PNG  (was Shift+I)##btn_iou_dump")) {
                if (gApp.mode == AppMode::kRegistration) {
                    glm::mat4 silView = buildSilhouetteView();
                    glm::mat4 silProj = buildSilhouetteProj();
                    int silW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
                    int silH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;
                    IoUDebug::dump(DEPTH_OUTPUT_PATH, "iou_debug",
                                   liverMesh3D, silView, silProj, silW, silH, 8);
                } else {
                    std::cout << "[IoU dump] requires Registration mode" << std::endl;
                }
            }
            if (ImGui::Button("Vertex-Squash diagnose  (was F10)##btn_vsq_diag")) {
                diagnoseVertexSquashV3RS(g_activeQuadrantMask);
            }
            // [key-reorg Phase 11] F9 -> checkbox for the Silhouette IoU window.
            ImGui::Checkbox("Show Silhouette Overlay window  (was F9)##viz_sil_overlay",
                            &SilOverlay::g_silOverlay.showWindow);
            // ---- end Phase 1 migration -------------------------------------
        };  // end g_debugPanel.drawVizExtra (migrated ScreenMesh Display + B/N viz)

        g_debugPanel.drawWExtra = [&]() {
            // Migrated from drawGBody (G/W split): Shape Match parameters for
            // Ctrl+W / Alt+W / Ctrl+Shift+W / Ctrl+Alt+W. Same [&] capture
            // scope as drawGBody, so every g_shapeMatch* / g_silhouetteSweep* /
            // g_ctrlgRim* / g_debugShapeMatch* symbol resolves identically.
            // Includes the Shape Match CollapsingHeader (Steps 3a-3d) AND the
            // rot / sign / ICP / rim-axis-sweep params that used to sit below it.

            // ===============================================================
            // Phase 7b Step 3a/3b — Shape Match (Ctrl+W / Alt+W) panel
            // ---------------------------------------------------------------
            // Wrapped in a CollapsingHeader so the panel is easy to find
            // inside the long Ctrl+G Quadrant Selector window. The header
            // is open by default the first time the window is built.
            // ===============================================================
            ImGui::Separator();
            ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            if (ImGui::CollapsingHeader("Shape Match (Ctrl+W / Alt+W)##phase7b")) {
            // Phase 7b Step 3a — Full-2D Ctrl+W cost (depth-free)
            //   Uses g_boundaryDistMap directly + AR-camera projection.
            //   Recommended when depth-anything-v2 depth is unreliable.
            ImGui::TextUnformatted("[Step 3a] 2D AR-projected Ctrl+W cost");
            ImGui::Checkbox("Use 2D AR matching (Ctrl+W)",
                            &g_shapeMatchUse2DCost);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Phase 7b Step 3a: when ON, Ctrl+W's\n"
                                  "cost is mean(g_boundaryDistMap[project(src)])\n"
                                  "  - depth-free (target lives in 2D pixel space)\n"
                                  "  - O(|src|) lookup, ~1000x faster than 3D chamfer\n"
                                  "When OFF: legacy 3D chamfer between\n"
                                  "  depth-lifted target and source (original\n"
                                  "  Step 3 implementation).");
            }
            if (g_shapeMatchUse2DCost) {
                ImGui::SliderInt("2D contour anchors N",
                                 &g_shapeMatchContourN2D, 8, 500);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Arc-length resample target 2D\n"
                                      "contour to N evenly-spaced anchors.\n"
                                      "Each anchor produces 1..4 candidates\n"
                                      "(× sign_mask). 200 default, fast even\n"
                                      "with 200×4=800 evals (~50ms total).");
                }
                ImGui::SliderFloat("2D out-of-frame penalty (px)",
                                   &g_shapeMatchOutOfFrameDistPx,
                                   10.0f, 500.0f, "%.0f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Per-point cost assigned when a source\n"
                                      "vertex projects outside the AR camera\n"
                                      "viewport (or behind the camera).");
                }
                ImGui::SliderFloat("2D max distance cap (px)",
                                   &g_shapeMatchMaxDistCapPx,
                                   10.0f, 500.0f, "%.0f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Clamp per-point boundary distance to\n"
                                      "this cap. Bounds influence of points\n"
                                      "landing outside the mask (which\n"
                                      "otherwise get the 9999 sentinel from\n"
                                      "g_boundaryDistMap).");
                }
                ImGui::SliderFloat("2D min in-frame rate",
                                   &g_shapeMatchMinInFrameRate,
                                   0.0f, 1.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Reject candidates where fewer than\n"
                                      "this fraction of source rim points\n"
                                      "project inside the AR viewport.\n"
                                      "0.30 = default, 0.0 = no rejection.");
                }
                ImGui::SliderFloat("2D instrument exclude (px)",
                                   &g_shapeMatch2DInstThreshPx,
                                   0.0f, 60.0f, "%.0f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Exclude target boundary pixels with\n"
                                      "instrumentDist < threshold from the\n"
                                      "2D contour trace. 0 = disable.\n"
                                      "(Reapply with Shift+W after change.)");
                }
                // Diagnostic readback
                if (g_debugShapeMatchBestK >= 0) {
                    ImGui::Text("Last Ctrl+W/2D: in_frame=%.0f%%  in_mask=%.0f%%  contour=%d px",
                                100.0f * g_debugShapeMatchBestInFrame,
                                100.0f * g_debugShapeMatchBestInMask,
                                (int)g_debugTargetContour2D.size());
                }
            }
            ImGui::Separator();

            // Phase 7b Step 3b — Gauss-Newton refine (Alt+W)
            //   Revised: SkipCoarse (trust init pose) + TransOnly (3-DoF)
            //   by default. Prevents the depth-runaway observed when
            //   full 6-DoF GN escapes into a depth-degenerate minimum.
            ImGui::TextUnformatted("[Step 3b] Gauss-Newton refine (Alt+W)");
            ImGui::Checkbox("Flip source normal to camera (Idea A)",
                            &g_shapeMatchFlipNormalToCamera);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Step 3b: flip the source rim's PCA\n"
                                  "patch normal to face the camera before\n"
                                  "passing to Coarse2D. Fixes 180° flip.\n"
                                  "Affects BOTH Ctrl+W and Alt+W.");
            }
            ImGui::SliderFloat("Coarse max rot [deg] (Ctrl+W)",
                               &g_shapeMatchCoarseMaxRotDeg, 0.0f, 180.0f, "%.0f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Step 3b Plan A: HARD reject Coarse2D\n"
                                  "candidates whose rotation exceeds this\n"
                                  "many degrees from identity (trace-based\n"
                                  "angle). 45° = trust Init Pose as the\n"
                                  "rotation reference. 180° = no constraint.\n"
                                  "This is a hard reject (continue), NOT\n"
                                  "the old soft penalty.");
            }
            ImGui::Separator();
            ImGui::Checkbox("Alt+W: skip Coarse2D (trust current pose)",
                            &g_shapeMatchAltWSkipCoarse);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Step 3b Plan B: Alt+W bypasses Coarse2D\n"
                                  "entirely and refines the CURRENT mesh\n"
                                  "pose with GN. Recommended after\n"
                                  "Apply Init Pose has set a good baseline.\n"
                                  "OFF = legacy (Coarse2D → GN, can break\n"
                                  "      a good init pose).");
            }
            ImGui::Checkbox("Alt+W: translation-only refine (3-DoF)",
                            &g_shapeMatchGNTranslationOnly);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Step 3b Plan B: lock rotation, only\n"
                                  "translate (tx, ty, tz) in GN.\n"
                                  "Eliminates depth-degeneracy of 6-DoF PnP.\n"
                                  "Best paired with SkipCoarse=ON for\n"
                                  "a pure translation polish after Init Pose.");
            }
            ImGui::SliderInt("GN max iter (Alt+W)",
                             &g_shapeMatchGNMaxIter, 1, 100);
            ImGui::SliderFloat("GN λ init (Alt+W)",
                               &g_shapeMatchGNLambdaInit,
                               1.0e-6f, 1.0e0f, "%.0e",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("GN λ MIN (Alt+W)",
                               &g_shapeMatchGNLambdaMin,
                               1.0e-9f, 1.0e0f, "%.0e",
                               ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Step 3b: LM damping floor. Prevents the\n"
                                  "depth-degenerate escape (observed:\n"
                                  "λ → 1e-9 → ||dxi|| spike). 1e-3 keeps\n"
                                  "the solver always partly gradient-descent.\n"
                                  "Lower = more aggressive GN.");
            }
            ImGui::SliderFloat("GN step max ||Δξ|| (Alt+W)",
                               &g_shapeMatchGNStepMax, 0.001f, 1.0f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Step 3b: trust-region clamp on the\n"
                                  "per-iteration update norm. 0.05 ≈\n"
                                  "5 cm + 3°. Caps the first-step explosion.");
            }
            ImGui::SliderFloat("GN eps step (Alt+W)",
                               &g_shapeMatchGNEpsStep,
                               1.0e-8f, 1.0e-2f, "%.0e",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("GN eps rel (Alt+W)",
                               &g_shapeMatchGNEpsRel,
                               1.0e-8f, 1.0e-2f, "%.0e",
                               ImGuiSliderFlags_Logarithmic);
            // Diagnostic readback for last Alt+W call
            if (g_debugShapeMatchGNIters > 0) {
                const char* reason_str[] = {"step", "rel_cost", "max_iter", "lm_fail"};
                const int r = (g_debugShapeMatchGNReason >= 0
                               && g_debugShapeMatchGNReason < 4)
                              ? g_debugShapeMatchGNReason : 3;
                ImGui::Text("Last Alt+W: %.2f → %.2f px  Δ=%.2f  iters=%d  %s  %s",
                            g_debugShapeMatchGNInitCost,
                            g_debugShapeMatchGNFinalCost,
                            g_debugShapeMatchGNInitCost - g_debugShapeMatchGNFinalCost,
                            g_debugShapeMatchGNIters,
                            g_debugShapeMatchGNConverged ? "[conv]" : "[no-conv]",
                            reason_str[r]);
                ImGui::Text("            in_frame=%d  bdy_cache=%s",
                            g_debugShapeMatchGNInFrame,
                            g_gnUnsignedBdyValid ? "valid" : "invalid");
            }

            // Phase 7b Step 3c — Contour Sweep (Ctrl+Alt+W)
            //   Live arc-length × rotation grid search. Z-axis is
            //   preserved structurally; sweep runs over multiple
            //   frames so user sees convergence visually.
            ImGui::Separator();
            ImGui::TextUnformatted("[Step 3c] Contour Sweep (Ctrl+Alt+W)");
            ImGui::SliderInt("Sweep N target anchors",
                             &g_shapeMatchSweepNTarget, 4, 80);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Number of arc-length divisions on the\n"
                                  "target contour. 20 default. Each is a\n"
                                  "candidate \"pin point\" for the source rim.");
            }
            ImGui::SliderInt("Sweep N source pivots",
                             &g_shapeMatchSweepNSource, 4, 80);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Arc-length divisions on the source rim.\n"
                                  "Each pivot tries to match every target\n"
                                  "anchor. Total candidates = N_tgt × N_src × N_rot.");
            }
            ImGui::SliderInt("Sweep N rotations",
                             &g_shapeMatchSweepNRotation, 4, 180);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Rotation discretization. 36 default =\n"
                                  "10° steps. Rotates source rim around\n"
                                  "world Z (camera-forward in AR view).");
            }
            ImGui::Checkbox("Sweep: show anchors / pivots",
                            &g_shapeMatchSweepShowAnchors);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Show 20 target anchors (gray, fixed on\n"
                                  "image plane) + 20 source pivots (gray,\n"
                                  "transformed by trial pose every frame).\n"
                                  "The current (i_tgt, j_src) pair is drawn\n"
                                  "larger in cyan, walking along both curves\n"
                                  "in lockstep so you can see the discrete\n"
                                  "correspondence being evaluated.");
            }
            ImGui::Checkbox("Preview: 20 anchor/pivot dots (rainbow)",
                            &g_shapeMatchSweepPreviewAnchors);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Show 20 target anchors AND 20 source pivots\n"
                                  "as rainbow-coloured dots (red=0 ... violet=19)\n"
                                  "regardless of whether a sweep is active.\n"
                                  "Lets you verify, BEFORE pressing Ctrl+Alt+W,\n"
                                  "that the 20 candidates are evenly distributed\n"
                                  "along the entire target boundary / source rim.\n"
                                  "Anchor 0 (red) and anchor 19 (violet) reveal\n"
                                  "the endpoints. Clustering reveals problems.");
            }
            ImGui::Checkbox("Filter target by source bbox (anatomical)##sweepfilter",
                            &g_shapeMatchSweepFilterByRim);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Restrict target boundary points to the 2D\n"
                                  "bounding box of the projected source rim,\n"
                                  "expanded by margin px on all sides.\n"
                                  "PURPOSE: prevent the sweep from pinning\n"
                                  "source's caudal pivot to target's cranial\n"
                                  "anchor (which would translate source far\n"
                                  "from its Apply-Init-Pose position).\n"
                                  "Strongly recommended when source uses a\n"
                                  "Caudal-only filter (or any partial rim).");
            }
            ImGui::SliderFloat("Filter margin (px)##sweepfiltermargin",
                               &g_shapeMatchSweepFilterMarginPx,
                               20.0f, 500.0f, "%.0f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Bbox expansion on each side, in pixels.\n"
                                  "Smaller (50-100): stricter, stays closer to\n"
                                  "initial pose. Default 150 = moderate slack.\n"
                                  "Larger (300+): more pose-correction freedom\n"
                                  "but allows anatomically loose matches.");
            }
            ImGui::Checkbox("Sweep: endpoint constraint (RIM direction)",
                            &g_shapeMatchSweepUseEndpointConstraint);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("ON: detect which source RIM endpoint maps\n"
                                  "to which target contour endpoint, possibly\n"
                                  "reverse the source chain so j_src=0 ↔ i_tgt=0\n"
                                  "means same physical end. Then only allow\n"
                                  "sweep candidates with |j_src - i_tgt| <=\n"
                                  "tolerance (eliminates left↔right and\n"
                                  "forward↔reverse anatomical mismatches).\n"
                                  "OFF for closed-loop sources (full rim,\n"
                                  "no caudal filter).");
            }
            ImGui::SliderInt("Sweep endpoint tolerance",
                             &g_shapeMatchSweepEndpointTolerance, 0, 19);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Allowed |j_src - i_tgt| arc-length steps.\n"
                                  "0 = strict diagonal (j_src must equal i_tgt).\n"
                                  "3 = ±3 step slack (default).\n"
                                  "19 = effectively no constraint.");
            }
            if (g_shapeMatchSweepUseEndpointConstraint) {
                ImGui::Text("  src_open=%s  dir=%s  (last sweep)",
                            g_shapeMatchSweepSrcIsOpenDiag ? "Y" : "N (closed)",
                            g_shapeMatchSweepDirReversedDiag ? "REVERSED" : "forward");
            }

            // ---- Phase 7b Step 3c++: Label-based orientation lock UI -
            // The endpoint constraint only works on OPEN chains. For
            // CLOSED rims we use the LiverLeftRightLabel / LiverCranio-
            // CaudalLabel PCA labels directly: a PURE_RIGHT source pivot
            // can only pair with a target anchor on screen +x side, etc.
            // BOUNDARY-labeled pivots (rim 屈曲帯) pass through freely.
            ImGui::Separator();
            ImGui::TextDisabled("Orientation lock (label LR/CC + θ cap)");
            ImGui::Checkbox("Sweep: orientation lock (anatomical L/R/CC)",
                            &g_shapeMatchSweepUseOrientationLock);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON: enforce two checks during the sweep —\n"
                    "  (A) anatomical label match —\n"
                    "      source pivot j carries the LR/CC label of its\n"
                    "      nearest rim-chain vertex (PCA-derived). Target\n"
                    "      anchor i is classified by 2D position vs locked\n"
                    "      centroid: screen +x → RIGHT, screen -y → CRANIAL.\n"
                    "      Mismatch (PURE_RIGHT ↔ SS_LR_LEFT or CRANIAL ↔\n"
                    "      SS_CC_CAUDAL) → candidate skipped. BOUNDARY (src)\n"
                    "      and AMBIGUOUS (tgt, inside neutral band) always\n"
                    "      pass — handles the 屈曲帯 / midline edge case.\n"
                    "  (B) θ magnitude cap |wrap_180(θ)| ≤ rot_cap°.\n"
                    "Auto-degrades to OFF if g_liverLR / g_liverCC missing\n"
                    "(diagnostic line at sweep start).\n"
                    "Default ON.");
            }
            ImGui::SliderFloat("orientation neutral band [px]",
                               &g_shapeMatchSweepNeutralBandPx,
                               0.0f, 400.0f, "%.0f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Check (A) target side AMBIGUOUS-band half-width in\n"
                    "pixels around the locked centroid.\n"
                    "Anchors with |dx| ≤ band or |dy| ≤ band are tagged\n"
                    "AMBIGUOUS and pair with any source label — the image-\n"
                    "midline analog of source BOUNDARY pivots.\n"
                    "80 px ≈ 4% of 1920w (default).\n"
                    "0 px = strictest (zero AMBIGUOUS zone).\n"
                    "400 px = effectively disables Check (A); only B applies.");
            }
            ImGui::SliderFloat("orientation rot_cap [deg]",
                               &g_shapeMatchSweepRotationLockDeg,
                               15.0f, 180.0f, "%.0f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Check (B) cap on the candidate's image-plane θ\n"
                    "rotation. |wrap_180(θ)| ≤ rot_cap.\n"
                    "90° = ±90° from identity (default).\n"
                    "45° = strict, only fine-tuning rotations.\n"
                    "180° = no rotation cap (only check A applies).\n"
                    "This directly blocks the 180°-class flips the user\n"
                    "observes when running Ctrl+Alt+W on a CLOSED rim.");
            }
            // Diagnostic readback of the last sweep's label-lock status
            if (g_shapeMatchSweepUseOrientationLock) {
                ImGui::Text("  label_lock ready=%s  (LR & CC both valid)",
                            g_shapeMatchSweepLabelLockReadyDiag ? "Y" : "N");
            }

            ImGui::Checkbox("Sweep: show anchor/pivot dots (cyan=current)",
                            &g_shapeMatchSweepShowAnchors);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Visualize the 20 target anchors (back-projected\n"
                                  "to AR image plane) and 20 source pivots\n"
                                  "(transformed to trial pose). The current\n"
                                  "(i_tgt, j_src) pair being tested is highlighted\n"
                                  "as a large CYAN dot on each side — same color\n"
                                  "means corresponding pair.\n"
                                  "Use this to confirm that the sweep is actually\n"
                                  "exercising the target-side indexing every frame.");
            }
            ImGui::Checkbox("Animate sweep##animatesweep",
                            &g_shapeMatchSweepAnimate);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("ON  = sweep runs over multiple frames\n"
                                  "       (visual animation, ~10 sec @ 300+300\n"
                                  "       frames). Yellow dots = current trial,\n"
                                  "       red dots = global best so far.\n"
                                  "OFF = one-shot batch (~20 ms, no animation,\n"
                                  "       result applied immediately).");
            }
            ImGui::SliderInt("Sweep frames Phase 1",
                             &g_shapeMatchSweepFrames1, 30, 1200);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("How many frames to spend on Phase 1\n"
                                  "(broad sweep over full ranges). 300 ≈\n"
                                  "5 sec at 60 fps. Higher = slower\n"
                                  "animation, smaller per-frame batches.\n"
                                  "Ignored if Animate sweep = OFF.");
            }
            ImGui::SliderInt("Sweep frames Phase 2",
                             &g_shapeMatchSweepFrames2, 30, 1200);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Frame budget for Phase 2 (fine refine\n"
                                  "around Phase 1 best with narrower\n"
                                  "arc-length / angle range).\n"
                                  "Ignored if Animate sweep = OFF.");
            }
            ImGui::Checkbox("Sweep verbose log##sweeplog",
                            &g_shapeMatchSweepLog);
            // Diagnostic readback
            if (g_contourSweepState.active ||
                g_contourSweepState.phase == 3)
            {
                const auto& S = g_contourSweepState;
                ImGui::Text("Sweep status: phase=%d  frame=%d  cands=%d/%d",
                            S.phase, S.current_frame,
                            S.candidate_idx, S.total_candidates);
                ImGui::Text("  best cost=%.2f px  (i=%d, j=%d, θ=%.1f°)",
                            S.best_cost, S.best_i_tgt, S.best_j_src,
                            S.best_theta_deg);
                if (!S.cost_history.empty()) {
                    std::vector<float> hist_f(S.cost_history.size());
                    for (size_t i = 0; i < S.cost_history.size(); i++)
                        hist_f[i] = float(S.cost_history[i]);
                    ImGui::PlotLines("##sweep_cost",
                                     hist_f.data(), (int)hist_f.size(),
                                     0, nullptr,
                                     0.0f, FLT_MAX,
                                     ImVec2(0.0f, 50.0f));
                }
            }

            // -----------------------------------------------------------
            // Phase 7b Step 3d — Silhouette 2D Sweep (NEW, separate path)
            //   When [3] is ON, Ctrl+Alt+W routes to the new method:
            //     - Source rim 2D projection + PURE_RIGHT centroid start
            //     - Target contour lower-half + right-end start
            //     - 20 pivots / 20 anchors (fixed index correspondence)
            //     - Dense 2D chamfer cost
            //   [1] / [2] are debug popups; they can be enabled before
            //   sweep start to verify discretization is correct.
            // -----------------------------------------------------------
            ImGui::Separator();
            ImGui::TextDisabled("[Step 3d] Silhouette 2D Sweep (NEW)");
            ImGui::Checkbox("[0] Raw RIM 2D projection (points only, no ordering)",
                            &g_debugShow2DProjPopup_RawRim);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "STAGE 0 — minimum-viable debug view.\n"
                    "Projects every vertex of g_debugSourceRimChain\n"
                    "(the GREEN dots from the W key) to 2D pixel space\n"
                    "using the AR camera intrinsics + view matrix.\n"
                    "\n"
                    "Renders POINTS ONLY:\n"
                    "  - no polyline (no ordering assumed)\n"
                    "  - no angle-bin envelope\n"
                    "  - no pivot resampling\n"
                    "  - color = LR label\n"
                    "      red   = PURE_RIGHT\n"
                    "      blue  = PURE_LEFT\n"
                    "      gray  = BOUNDARY\n"
                    "  - yellow cross = 2D centroid of all rim points\n"
                    "  - cyan  cross = PURE_RIGHT 3D centroid projected to 2D\n"
                    "\n"
                    "Purpose: visually answer 'Is the source anatomical\n"
                    "RIM an open arch or a closed loop?' before CB1's\n"
                    "envelope step forces it into a closed contour.\n"
                    "Rebuilds every frame from the current source pose.");
            }
            ImGui::Checkbox("[0.1] Smoothed RIM 2D projection (grid + KNN)",
                            &g_debugShow2DProjPopup_RawRimSmoothed);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "STAGE 0.1 — noise-removed point set.\n"
                    "Same input as CB0, but applies:\n"
                    "  1. GRID AGGREGATION: bin into G×G px cells,\n"
                    "     replace each cell with the centroid of its\n"
                    "     points (LR label = majority vote).\n"
                    "  2. KNN SMOOTHING: for each centroid, find K\n"
                    "     nearest neighbours in 2D and replace with\n"
                    "     their mean; repeat N times.\n"
                    "Output is STILL UNORDERED — no envelope, no\n"
                    "polyline, no pivots. Just a cleaner point set\n"
                    "you can compare against the CB0 raw scatter.\n"
                    "\n"
                    "Optional overlay: original raw dots in faint gray\n"
                    "underneath the smoothed dots (toggle in popup).");
            }
            // [0.1] tuning sliders (shown inline so they're discoverable
            // even when the popup isn't open; they live under the
            // checkbox like the [3d] sweep sliders).
            ImGui::SliderFloat("[0.1] grid cell px",
                               &g_rawRimSmooth_GridPx, 3.0f, 80.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Side length of the 2D grid cell used to bin\n"
                    "rim points. Larger = fewer / smoother output\n"
                    "points. Typical 10-25 px for 1920x1080.");
            }
            ImGui::SliderInt("[0.1] KNN K",
                             &g_rawRimSmooth_KnnK, 0, 30);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "K nearest neighbours used in the smoothing pass.\n"
                    "0 = no KNN smoothing (grid-only result).\n"
                    "Larger K = smoother but blurrier; small (3-7)\n"
                    "preserves curvature.");
            }
            ImGui::SliderInt("[0.1] KNN iters",
                             &g_rawRimSmooth_KnnIters, 0, 8);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Number of KNN smoothing passes. Each pass\n"
                    "averages every point with its K neighbours.\n"
                    "1-3 is usually enough.");
            }
            ImGui::Checkbox("[0.1] show raw overlay",
                            &g_rawRimSmooth_ShowRawOverlay);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "When ON, draw the original raw points in faint\n"
                    "gray underneath the smoothed dots — quick visual\n"
                    "check that smoothing followed the data.");
            }
            ImGui::Checkbox("[0.2] Ordered RIM (MST + longest path)",
                            &g_debugShow2DProjPopup_RawRimOrdered);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "STAGE 0.2 — ordered open polyline.\n"
                    "Takes the cleaned CB0.1 points (uses the same\n"
                    "grid / KNN parameters) and orders them via:\n"
                    "  1. Prim MST on cleaned points (O(N²))\n"
                    "  2. Two-pass BFS → longest path in tree\n"
                    "  3. Orient so start endpoint is closer to the\n"
                    "     PURE_RIGHT 3D centroid (projected to 2D)\n"
                    "  4. Arc-length resample N pivots along the path\n"
                    "Output is an OPEN polyline — correct topology\n"
                    "for the caudal RIM arch (CB1's envelope forced\n"
                    "a closed loop, which is wrong).");
            }
            ImGui::SliderFloat("[0.2] max edge px",
                               &g_rawRimOrder_MaxEdgePx,
                               10.0f, 500.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "MST edges longer than this are filtered out.\n"
                    "Use this to prevent the MST from bridging\n"
                    "unrelated point clusters (e.g. main rim arch\n"
                    "vs. a stray BOUNDARY blob).\n"
                    "If the filter disconnects the graph, the largest\n"
                    "connected component is used.\n"
                    "Set to 500 to effectively disable.");
            }
            ImGui::SliderInt("[0.2] N pivots",
                             &g_rawRimOrder_NPivots, 5, 40);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Number of arc-length spaced pivots placed on\n"
                    "the longest path. 20 matches the existing CB1\n"
                    "convention so visual comparison is direct.");
            }
            ImGui::Checkbox("[0.2] show MST edges",
                            &g_rawRimOrder_ShowMST);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Draw all MST edges (faint gray) under the\n"
                    "longest path. Useful for diagnosing why a\n"
                    "particular ordering was chosen.");
            }
            ImGui::Checkbox("[0.2] show cleaned overlay",
                            &g_rawRimOrder_ShowCleaned);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Draw the cleaned (CB0.1) points faintly under\n"
                    "the path so you can see which ones were skipped\n"
                    "or bridged.");
            }
            // ---- CB0.3: Manual Sweep Probe -----------------------------
            ImGui::Checkbox("[0.3] Overlay probe (manual sweep candidate)",
                            &g_debugShow2DProjPopup_OverlayProbe);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "STAGE 0.3 — Manual sweep candidate viewer.\n"
                    "Reproduces ONE silhouette-sweep candidate at a time:\n"
                    "  - PIVOT i slider  : which source pivot to pin\n"
                    "  - ANCHOR j slider : which target anchor to pin to\n"
                    "  - rotation step k : θ = k * (360 / n_rotation)\n"
                    "  - 'lock j=i'      : matches sweep (i:i pairing)\n"
                    "  - animate         : auto-advance k as a timestep\n"
                    "\n"
                    "Uses evaluateSilhouetteSweepCandidate's exact formula,\n"
                    "so the displayed geometry is byte-identical to what\n"
                    "the sweep computes.\n"
                    "\n"
                    "Shows a CC direction arrow on the transformed source,\n"
                    "with a head-up ✓ / FLIPPED ✗ indicator — exactly\n"
                    "what's needed to diagnose Phase 1 flip behavior.");
            }
            ImGui::SliderInt("[0.3] PIVOT i",
                             &g_overlayProbe_PivotI, 0, 39);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Source pivot index to pin to ANCHOR j.\n"
                    "Range hard-capped at 39 for the slider; actual\n"
                    "valid range is 0..N-1 where N = pivot count\n"
                    "(typically 20). Out-of-range values are clamped\n"
                    "automatically in the popup.");
            }
            // ANCHOR j slider is disabled (grayed out) when lock is ON,
            // since j follows i in that mode. This avoids the confusing
            // "I'm moving the slider but nothing changes" state.
            if (g_overlayProbe_LockJI) ImGui::BeginDisabled();
            ImGui::SliderInt("[0.3] ANCHOR j",
                             &g_overlayProbe_AnchorJ, 0, 39);
            if (g_overlayProbe_LockJI) ImGui::EndDisabled();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Target anchor index to receive the pinned source\n"
                    "pivot. DISABLED when 'lock j=i' is ON (j follows i).\n"
                    "Turn lock OFF to explore i≠j combinations that the\n"
                    "actual sweep never tries (sweep uses i:i pairing only).");
            }
            ImGui::Checkbox("[0.3] lock j=i (match sweep convention)",
                            &g_overlayProbe_LockJI);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "OFF (default) → i and j are independent.\n"
                    "                Lets you probe what would happen\n"
                    "                with cross combinations.\n"
                    "ON             → j follows i. Mirrors the real\n"
                    "                sweep (which uses i:i pairing only).\n"
                    "                The ANCHOR j slider grays out.");
            }
            ImGui::SliderInt("[0.3] rotation step k",
                             &g_overlayProbe_RotStep,
                             0, std::max(1, g_overlayProbe_NRotation - 1));
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Rotation index. θ = k * (360 / n_rotation).\n"
                    "Default n_rotation=36 → 10° per step → k=0..35.\n"
                    "Treat this as the timestep slider — drag it or\n"
                    "enable animate to watch the rotation sweep.");
            }
            ImGui::SliderInt("[0.3] n_rotation (sweep steps)",
                             &g_overlayProbe_NRotation, 4, 360);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Number of discrete rotation steps. The actual\n"
                    "sweep uses 36 (= 10° per step). Lowering this\n"
                    "makes k coarser; raising it gives finer θ control\n"
                    "for diagnosis.");
            }
            ImGui::Checkbox("[0.3] auto-animate k",
                            &g_overlayProbe_AutoAnimate);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Auto-advances rotation step k every N frames.\n"
                    "Watch how the source rotates through all candidate\n"
                    "orientations for the chosen (i, j).");
            }
            ImGui::SliderInt("[0.3] frames per step",
                             &g_overlayProbe_AnimFramesPerStep, 1, 60);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Animation speed control. Larger = slower.\n"
                    "1 step/frame ≈ 60 steps per second at 60 FPS.");
            }
            ImGui::Checkbox("[0.3] show CC direction arrow",
                            &g_overlayProbe_ShowCCArrow);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Draws an arrow on the transformed source pointing\n"
                    "from CRANIAL (head) to CAUDAL (foot). Header text\n"
                    "shows 'head-up ✓' if the arrow points downward on\n"
                    "screen (normal anatomy), or 'FLIPPED ✗' otherwise.");
            }
            ImGui::Checkbox("[0.3] show source overlay",
                            &g_overlayProbe_ShowSrcOverlay);
            ImGui::Checkbox("[0.3] show target overlay",
                            &g_overlayProbe_ShowTgtOverlay);
            ImGui::Checkbox("[0.3] simulate Check A (rotation cap)",
                            &g_overlayProbe_SimulateCheckA);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "When ON, CB0.3 paints the canvas reddish and shows\n"
                    "'REJECTED BY A' if the current (i, j, k) would be\n"
                    "rejected by sweep本番's Check A (rotation cap).\n"
                    "Uses the same cap value as the sweep ([3d] slider).\n"
                    "Turn OFF to see the raw candidate even if rejected.");
            }
            ImGui::Checkbox("[0.3] simulate Check B (CC guard)",
                            &g_overlayProbe_SimulateCheckB);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "When ON, CB0.3 paints the canvas reddish and shows\n"
                    "'REJECTED BY B' if the current (i, j, k) would be\n"
                    "rejected by sweep本番's Check B (CC orientation).\n"
                    "Uses the same tolerance as the sweep ([3d] slider).\n"
                    "Requires g_liverCC valid (run Shift+H first).");
            }
            ImGui::Checkbox("[1] Source 2D projection popup",
                            &g_debugShow2DProjPopup_Source);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Open a popup window showing the source rim chain\n"
                    "projected to 2D, with:\n"
                    "  - dense polyline tinted by LR label\n"
                    "    (red=PURE_RIGHT, gray=BOUNDARY, blue=PURE_LEFT)\n"
                    "  - 20 pivots in rainbow color (red=0 → violet=19)\n"
                    "  - PURE_RIGHT centroid (yellow cross)\n"
                    "  - right-start chain vertex (large yellow marker)\n"
                    "Rebuilds every frame from current source pose.\n"
                    "Use before pressing Ctrl+Alt+W to confirm orientation.");
            }
            ImGui::Checkbox("[2] Target rim lower-half popup",
                            &g_debugShow2DProjPopup_Target);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Open a popup window showing target contour:\n"
                    "  - upper half (gray, p.y <= centroid.y)\n"
                    "  - lower half (purple, p.y >  centroid.y)\n"
                    "  - centroid (yellow cross)\n"
                    "  - 20 anchors in rainbow color (red=0 → violet=19)\n"
                    "  - right-end (max x) start point (large yellow marker)\n"
                    "Cached; rebuilt only on Shift+W.");
            }
            ImGui::Checkbox("[3] Enable silhouette sweep (Ctrl+Alt+W uses NEW method)",
                            &g_silhouetteSweepEnable);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON  → Ctrl+Alt+W runs the Step 3d silhouette sweep.\n"
                    "OFF → Ctrl+Alt+W runs the legacy Step 3c sector sweep.\n"
                    "Independent state; the two paths never interfere.");
            }
            // ---- Source-rim discretization method (Step 3d Stage A) ----
            //   Drives CB1 popup AND Ctrl+Alt+W sweep (when [3] is ON).
            //   MST (default) = CB0.2 result (open polyline).
            //   ENVELOPE      = legacy angle-bin closed loop (kept for
            //                   comparison; anatomically wrong for the
            //                   caudal RIM arch).
            const char* srcRimMethodItems =
                "ENVELOPE (legacy closed loop)\0"
                "MST + longest path (CB0.2, open polyline)\0";
            ImGui::Combo("Source rim method", &g_silSwSrcRimMethod,
                         srcRimMethodItems);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Selects which algorithm populates the source rim\n"
                    "pivots used by CB1 popup AND the silhouette sweep:\n"
                    "  ENVELOPE: angle-bin + max-radius envelope.\n"
                    "    Produces a CLOSED loop. Legacy method; wrong\n"
                    "    for the caudal-RIM arch (CB0 confirmed this).\n"
                    "    Kept available as a fallback / comparison.\n"
                    "  MST + longest path:\n"
                    "    Uses the cleaned CB0.1 points (same grid / KNN\n"
                    "    parameters), builds an MST, takes the longest\n"
                    "    path (open polyline), orients toward PURE_RIGHT,\n"
                    "    arc-length resamples N pivots. Anatomically\n"
                    "    correct for an open arch. This is the default.");
            }
            ImGui::SliderInt("3d frames Phase 1",
                             &g_silhouetteSweepFrames1, 1, 600);
            ImGui::SliderInt("3d frames Phase 2",
                             &g_silhouetteSweepFrames2, 1, 300);
            ImGui::Checkbox("3d animate##silswanim",
                            &g_silhouetteSweepAnimate);
            ImGui::Checkbox("3d verbose log##silswlog",
                            &g_silhouetteSweepLog);

            // ---- Step 3d candidate guards (本番 sweep) ----
            ImGui::Separator();
            ImGui::TextDisabled("[3d] Candidate guards (本番 sweep)");
            ImGui::Checkbox("[3d] enable Check A (rotation cap)",
                            &g_silSwCheckA_Enable);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Reject sweep candidates whose |θ| exceeds the cap.\n"
                    "θ is wrapped into (-180°, +180°] first.\n"
                    "Independent of CC labels; works whenever sweep runs.\n"
                    "Default ON, cap = 30° (= sweep explores ±30° around 0).");
            }
            ImGui::SliderFloat("[3d] Check A: rotation cap ±deg",
                               &g_silSwCheckA_RotCapDeg, 5.0f, 180.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Hard cap on rotation magnitude. Candidates with\n"
                    "|θ| > this value get cost = +∞ (rejected).\n"
                    "Smaller = stricter (faster to converge but may\n"
                    "miss valid poses if initial alignment is off).");
            }
            ImGui::Checkbox("[3d] enable Check B (CC orientation guard)",
                            &g_silSwCheckB_Enable);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Reject sweep candidates whose CRANIAL→CAUDAL axis\n"
                    "(g_liverCC.d_cc, projected through candidate T)\n"
                    "doesn't point screen-down (= 6 o'clock = +y_pixel)\n"
                    "within tolerance.\n"
                    "Requires g_liverCC valid (run Shift+H first).\n"
                    "If not valid, this check is silently skipped.");
            }
            ImGui::SliderFloat("[3d] Check B: CC tolerance ±deg",
                               &g_silSwCheckB_CCToleranceDeg, 5.0f, 90.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Allowed deviation from 6 o'clock for the on-screen\n"
                    "CC direction. ±15° default = clock face 5:30..6:30.\n"
                    "(User originally specified 5:55-7:05 = ±5° but ±15°\n"
                    "gives realistic working tolerance.)");
            }

            if (g_silhouetteSweep.active || g_silhouetteSweep.phase == 3) {
                const auto& S3d = g_silhouetteSweep;
                ImGui::Text("3d status: phase=%d  frame=%d  cands=%d/%d",
                            S3d.phase, S3d.current_frame,
                            S3d.candidate_idx, S3d.total_candidates);
                ImGui::Text("  best cost=%.2f px  (i=%d, θ=%.1f°)  rev=%s",
                            S3d.best_cost, S3d.best_i_pivot,
                            S3d.best_theta_deg,
                            S3d.src_dir_reversed ? "Y" : "N");
                if (!S3d.cost_history.empty()) {
                    std::vector<float> hist_f(S3d.cost_history.size());
                    for (size_t i = 0; i < S3d.cost_history.size(); i++)
                        hist_f[i] = float(S3d.cost_history[i]);
                    ImGui::PlotLines("##silsw_cost",
                                     hist_f.data(), (int)hist_f.size(),
                                     0, nullptr,
                                     0.0f, FLT_MAX,
                                     ImVec2(0.0f, 50.0f));
                }
            }
            if (!g_silhouetteSweep.fail_reason.empty()
                && !g_silhouetteSweep.active)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                                   "3d last failure: %s",
                                   g_silhouetteSweep.fail_reason.c_str());
            }

            } // close: if (CollapsingHeader("Shape Match..."))
            ImGui::Separator();

            // Phase 7b Step 3 — Rotation-angle constraint for Ctrl+W
            // Shape Match. cos_angle = (trace(R)-1)/2 below threshold
            // incurs a penalty. Effectively limits Shape Match to small
            // adjustment rotations, trusting Apply Init Pose for the
            // overall anatomy orientation.
            ImGui::SliderFloat("rot cos thresh (Ctrl+W)",
                               &g_shapeMatchAnatomyThresh, -1.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Phase 7b Ctrl+W: cos((trace(R)-1)/2) below this\n"
                                  "threshold incurs a penalty.\n"
                                  "  +1.0 = strict (R must be identity)\n"
                                  "  +0.5 = allow up to 60° rotation\n"
                                  "   0.0 = allow up to 90° rotation (default,\n"
                                  "         excludes sign=1/2/3 180° flip candidates)\n"
                                  "  -1.0 = no constraint");
            }
            {
                float lam = (float)g_shapeMatchAnatomyLambda;
                if (ImGui::SliderFloat("rot lambda (Ctrl+W)",
                                       &lam, 0.0f, 5.0f, "%.2f")) {
                    g_shapeMatchAnatomyLambda = (double)lam;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Phase 7b Ctrl+W: rotation-angle penalty\n"
                                      "weight. 0 = disable constraint.\n"
                                      "1 = 180° flip incurs penalty 1.0 (= 10×\n"
                                      "typical chamfer 0.1, effective rejection).");
                }
            }

            // Phase 7b Step 4a — sign-mask filter for Shape Match.
            // sign=0 only = "trust Apply Init Pose, allow only identity-
            // like rotations". Sign 1/2/3 are 180° flip candidates which
            // can hit wrong poses even with rotation penalty.
            {
                bool sign0_only = (g_shapeMatchSignMode == 0x1);
                if (ImGui::Checkbox("sign=0 only (Ctrl+W: no flip)",
                                    &sign0_only))
                {
                    g_shapeMatchSignMode = sign0_only ? 0x1 : 0xF;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Phase 7b Ctrl+W: when ON, try only\n"
                                      "sign=0 (t+,n+) = identity-like rotation,\n"
                                      "30 candidates instead of 120.\n"
                                      "When OFF, try all 4 signs (180° flips\n"
                                      "included, 120 candidates).\n"
                                      "ON = trust Apply Init Pose orientation.");
                }
            }

            // Phase 7b Step 4a — Live ICP iter cap for Ctrl+Shift+W.
            // When Shape Match is followed by Live ICP, cap the iters so
            // the ICP stays close to the Shape Match solution instead of
            // wandering off and getting rescued by early-stop.
            ImGui::SliderInt("live max_iter (Ctrl+Shift+W)",
                             &g_shapeMatchLiveMaxIter, 0, 200);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Phase 7b Ctrl+Shift+W (Step 4a fallback):\n"
                                  "temporarily override g_normRefineMaxIter\n"
                                  "for Live ICP after Shape Match.\n"
                                  "  0 or negative = no override (use Shift+N default 200)\n"
                                  "  20 = strong absolute-maintenance (default)\n"
                                  "  5-10 = even stronger\n"
                                  "Only used when 'use rim-axis sweep' is OFF.");
            }

            // Phase 7b Step 4b — Rim Axis Rotation Sweep (default ON)
            // Replaces Live ICP with a 1D sweep around the rim normal
            // axis. Preserves rim alignment absolutely and excludes
            // flipped poses via all-vertex chamfer.
            ImGui::Separator();
            ImGui::Checkbox("Use rim-axis sweep (Ctrl+Shift+W: skip ICP)",
                            &g_shapeMatchAxisSweepEnabled);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Phase 7b Ctrl+Shift+W (Step 4b, default ON):\n"
                                  "After Shape Match best T applied, sweep\n"
                                  "rotations around rim normal axis with\n"
                                  "all-vertex chamfer. Picks best angle and\n"
                                  "SKIPS Live ICP entirely.\n"
                                  "  ON  = rim fit absolutely preserved\n"
                                  "  OFF = fallback to Live ICP (Step 4a)");
            }
            ImGui::SliderInt("axis sweep N angles",
                             &g_shapeMatchAxisSweepN, 8, 90);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Number of rotation samples in 360°.\n"
                                  "36 = 10° step (default).\n"
                                  "Higher = finer but slower.");
            }
            ImGui::SliderInt("axis sweep tgt subN",
                             &g_shapeMatchAxisSweepTgtSubN, 500, 20000);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Target points uniformly downsampled to\n"
                                  "this count before chamfer.\n"
                                  "5000 = balanced (~1-2s for default N=36).\n"
                                  "Higher = more accurate but slower.");
            }
            ImGui::Checkbox("dual-variant compare (A:full vs B:rim)",
                            &g_shapeMatchAxisSweepCompare);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Phase 7b Ctrl+Shift+W (default ON):\n"
                                  "Run both Variant A (full vertex sym\n"
                                  "chamfer) AND Variant B (rim chain vs\n"
                                  "target boundary) sweeps. Apply each\n"
                                  "winner-candidate rotation, measure\n"
                                  "CompRMSE via 0-iter session, restore,\n"
                                  "then pick the variant with lower RMSE.\n"
                                  "OFF = Variant A only (~700ms faster).");
            }

        };  // end g_debugPanel.drawWExtra (migrated Shape Match params from G tab)

        // Consolidated Debug Panel (Ctrl+D). Same registration / non-Umeyama
        // guard as the legacy floating panels above.
        if (gApp.mode == AppMode::kRegistration && !gUmeyama.active) {
            DebugPanel::draw(g_debugPanel, gUI);
        }

        // Pose Library / AR preview / SilOverlay preview
        //   Umeyama 2画面モード中は全部スキップ。drawUmeyamaOverlay が画面中央に
        //   ステータス + 大きなボタン群 (Undo/Execute/Cancel) を出し、サイドバーも
        //   消えるので、その上にこれらのフローティングウィンドウが被ると
        //   クリック取りこぼし・視認性低下・体感カクつきの原因になる。
        if (!gUmeyama.active) {
            // Pose Library ウィンドウ (元コード line 4699 通り、mode guard 無しで毎フレーム呼ぶ。
            // showWindow == false なら drawPoseLibraryWindow 側で early-return)
            drawPoseLibraryWindow();

            // ARスクリーンショットのプレビューウィンドウ
            float vpW = gUI.getViewportWidth(gWindowWidth);
            ARSave::drawPreviewWindow(g_arSave, vpW, (float)gWindowHeight);

            // V3RS Phase 2 diagnostic: silhouette IoU overlay window.
            // Toggled via F9. Reuses the same viewport width / window
            // height computed for ARSave above.
            SilOverlay::drawPreviewWindow(
                SilOverlay::g_silOverlay, vpW, (float)gWindowHeight);
        }

        OrbitCam.UpdateCamera(dt);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ---- Live Calibration preview branch ---------------------------------
        // Active session + open camera -> dedicated render path that wins over
        // whatever AppMode the user came from (typically kRegistration). This
        // is the ONLY place that pumps gCamera.capture() during a live calib
        // session, so originalFrame stays fresh for saveFrameAsJPEG().
        // Critically: no MaskPicker, no drawDepthOverlay. The viewport shows
        // the camera; the Take Picture tab in the sidebar handles control.
        if (g_liveCalibActive && gCamera.active) {
            gCamera.capture(gApp.arBg);

            // Letterbox to camera aspect ratio (same math as kMaskSelection).
            float imgAspect = (float)gCamera.width  / (float)gCamera.height;
            float winAspect = (float)gWindowWidth   / (float)gWindowHeight;
            int viewW = gWindowWidth, viewH = gWindowHeight, viewX = 0, viewY = 0;
            if (imgAspect > winAspect) {
                viewH = (int)(gWindowWidth / imgAspect);
                viewY = (gWindowHeight - viewH) / 2;
            } else {
                viewW = (int)(gWindowHeight * imgAspect);
                viewX = (gWindowWidth - viewW) / 2;
            }
            glViewport(viewX, viewY, viewW, viewH);
            gApp.arBg.draw();
            glViewport(0, 0, gWindowWidth, gWindowHeight);
        }
        else if (gApp.mode == AppMode::kMaskSelection) {
            // カメラモードの場合はフレームをキャプチャ
            if (gCamera.active) {
                gCamera.capture(gApp.arBg);
            }

            // マスク選択モード：固定アスペクト比で画像を表示
            if (gApp.image.loaded || gCamera.active) {
                // 画像のアスペクト比を保持して中央に表示
                int imgWidth = gCamera.active ? gCamera.width : gApp.image.width;
                int imgHeight = gCamera.active ? gCamera.height : gApp.image.height;
                float imgAspect = (float)imgWidth / (float)imgHeight;
                float winAspect = (float)gWindowWidth / (float)gWindowHeight;

                int viewW = gWindowWidth;
                int viewH = gWindowHeight;
                int viewX = 0;
                int viewY = 0;

                if (imgAspect > winAspect) {
                    // 画像の方が横長
                    viewH = gWindowWidth / imgAspect;
                    viewY = (gWindowHeight - viewH) / 2;
                } else {
                    // 画像の方が縦長
                    viewW = gWindowHeight * imgAspect;
                    viewX = (gWindowWidth - viewW) / 2;
                }

                glViewport(viewX, viewY, viewW, viewH);
                // JPEG経由でテクスチャは更新済みなので、単に描画するだけ
                gApp.arBg.draw();
                glViewport(0, 0, gWindowWidth, gWindowHeight);

                // マスクポイントの描画（調整が必要）
                gMaskRenderer.draw(gApp);
            }

            // マスク選択モード用のUIオーバーレイを常に表示
            gUI.drawDepthOverlay(gWindowWidth, gWindowHeight);
        }
        else if (gApp.mode == AppMode::kImageOnly) {
            // カメラモードの場合はフレームを更新
            if (gCamera.active && !gCamera.captured) {
                gCamera.capture(gApp.arBg);
            }
            if (gApp.image.loaded) gApp.arBg.draw();
            gMaskRenderer.draw(gApp);
        }
        else if (gApp.mode == AppMode::kRegistration) {
            if (gUmeyama.active) {
                // ---- Umeyama 2画面モード ----
                gUmeyama.render(shaderProgram, shaderProgramCube,
                                registrationHandle,
                                {liverMesh3D, portalMesh3D, veinMesh3D,
                                 tumorMesh3D, segmentMesh3D, gbMesh3D},
                                screenMesh, gWindowWidth, gWindowHeight);
            } else {
                // ---- 通常1画面描画 ----
                const int sidebarW = 400;
                compute3DViewport(gWindowWidth, gWindowHeight, sidebarW);
                glViewport(g_3dViewport.x, g_3dViewport.y,
                           g_3dViewport.w, g_3dViewport.h);

                glm::vec3 liverCenter   = Reg3DCustom::computeMeshCenter(*liverMesh3D);
                glm::vec3 textureCenter = Reg3DCustom::computeMeshCenter(*screenMesh);
                OrbitCam.updateTargetPositions(liverCenter, textureCenter);
                model = glm::translate(glm::mat4(1.0f), objPos);

                if (gApp.arMode) {
                    view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),
                                       glm::vec3(0.0f, 0.0f, 1.0f),
                                       glm::vec3(0.0f, 1.0f, 0.0f));
                }

                // ★ライティング用カメラ位置: ARモードでは固定AR視点(原点)に合わせる。
                //   上で view を AR視点(eye=原点, look +Z)に上書きしているのに、ライティング
                //   (lightPos/viewPos = ヘッドライト)に orbit のカメラ位置を渡すと、オービット
                //   回転後の位置から照らされて照明が破綻する。view と光源を一致させる。
                const glm::vec3 lightCamPos = gApp.arMode
                                                  ? glm::vec3(0.0f, 0.0f, 0.0f)
                                                  : OrbitCam.cameraPos;

                if (gApp.arMode) gApp.arBg.draw();

                std::vector<mCutMesh*> meshesToDraw;
                std::vector<glm::vec4> dynamicColors;

                mCutMesh* organs[6] = { liverMesh3D, portalMesh3D, veinMesh3D,
                                       tumorMesh3D, segmentMesh3D, gbMesh3D };
                glm::vec4 organColors[6] = {
                    glm::vec4(0.8f, 0.2f, 0.2f, 1.0f),  // liver - red
                    glm::vec4(0.2f, 0.2f, 0.8f, 1.0f),  // portal - blue
                    glm::vec4(0.2f, 0.5f, 0.5f, 1.0f),  // vein - teal
                    glm::vec4(0.8f, 0.5f, 0.5f, 1.0f),  // tumor - pink
                    glm::vec4(0.2f, 0.8f, 0.5f, 1.0f),  // segment - green
                    glm::vec4(0.2f, 0.8f, 0.2f, 1.0f)    // gb - yellow-green
                };

                int textureMeshIdx = -1;
                for (int i = 0; i < 6; i++) {
                    if (organs[i] && g_meshAlpha[i] > 0.01f) {
                        meshesToDraw.push_back(organs[i]);
                        glm::vec4 c = organColors[i];
                        c.a = g_meshAlpha[i];
                        dynamicColors.push_back(c);
                    }
                }
                // screenMesh（レジストレーション用、非テクスチャ、alpha=g_meshAlpha[7]）
                //   - g_screenMeshAsPoints=false : 既存の三角形描画パス（meshesToDraw に追加）
                //   - g_screenMeshAsPoints=true  : ここでは push せず、後段で点群として別描画
                if (g_meshAlpha[7] > 0.01f && !g_screenMeshAsPoints) {
                    meshesToDraw.push_back(screenMesh);
                    dynamicColors.push_back(glm::vec4(0.3f, 0.6f, 0.9f, g_meshAlpha[7]));
                }

                // boardMesh（テクスチャ付き表示用、alpha=g_meshAlpha[6]）
                if (boardMesh3D && g_meshAlpha[6] > 0.01f) {
                    meshesToDraw.push_back(boardMesh3D);
                    dynamicColors.push_back(glm::vec4(1.0f, 1.0f, 1.0f, g_meshAlpha[6]));
                    textureMeshIdx = (int)meshesToDraw.size() - 1;
                }

                draw_AllmCutMeshes(meshesToDraw, shaderProgram, shaderProgramCube,
                                   lightCamPos, dynamicColors,
                                   model, view, projection, textureMeshIdx);

                // screenMesh を点群として描画（フラグが立っている場合）
                //   - 三角形描画より軽量（overdraw なし）
                //   - 既存 VBO を共有するので追加メモリなし
                if (g_screenMeshAsPoints && g_meshAlpha[7] > 0.01f && screenMesh) {
                    drawScreenMeshAsPoints(
                        screenMesh, shaderProgram,
                        model, view, projection, lightCamPos,
                        glm::vec4(0.3f, 0.6f, 0.9f, g_meshAlpha[7]),
                        g_screenMeshPointSize);
                }

                // クラスタ可視化の描画
                // ソース (cluster1) は明るく大きく目立たせ、ターゲット (targetPoints) は
                // 多数あるので暗めで小さくして視覚的に「源点 vs 先点」を区別する。
                if (g_showClusterVisualization) {
                    const float rBase = RegRatios::markerCluster();
                    const float rSrc  = rBase * 1.4f;   // Source: 大きい
                    const float rTgt  = rBase * 0.55f;  // Target accepted: 小さい
                    const float rInt  = rBase * 0.40f;  // Target interior: もっと小さい

                    // SOURCE (cluster1) — bright green, large
                    for (size_t i = 0; i < g_cluster1Points.size(); i++) {
                        g_sphereMarker.draw(shaderProgram, g_cluster1Points[i],
                                            glm::vec3(0.30f, 1.00f, 0.20f),  // bright green
                                            rSrc, view, projection, lightCamPos);
                    }
                    // TARGET interior (cluster2) — dim blue, small (rare; only when used as interior cloud)
                    for (size_t i = 0; i < g_cluster2Points.size(); i++) {
                        g_sphereMarker.draw(shaderProgram, g_cluster2Points[i],
                                            glm::vec3(0.05f, 0.20f, 0.45f),  // very dim blue
                                            rInt, view, projection, lightCamPos);
                    }
                    // TARGET accepted boundary — dim yellow, small (massive count)
                    for (size_t i = 0; i < g_targetPoints.size(); i++) {
                        g_sphereMarker.draw(shaderProgram, g_targetPoints[i],
                                            glm::vec3(0.55f, 0.50f, 0.05f),  // dim yellow / mustard
                                            rTgt, view, projection, lightCamPos);
                    }
                }

                // 境界候補の可視化（B キー）: 採用=緑, 棄却(器具)=赤
                if (g_showBoundaryCandidates) {
                    const float rTarget = RegRatios::markerTarget();
                    for (const auto& p : g_targetPoints) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(0.0f, 1.0f, 0.2f),
                                            rTarget, view, projection, lightCamPos);
                    }
                    for (const auto& p : g_rejectedBoundaryPoints) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(1.0f, 0.1f, 0.1f),
                                            rTarget, view, projection, lightCamPos);
                    }
                }

                // ソース側の可視化（N キー）: シアン=全可視, マゼンタ=シルエット絞り込み後
                // 全可視は半径やや小さめで奥に、シルエット集合は大きめで手前に描画。
                if (g_showSourceVisualization) {
                    const float rBase = RegRatios::markerCluster();
                    const float rVis  = rBase * 0.8f;
                    const float rSil  = rBase * 1.1f;
                    for (const auto& p : g_visibleSourcePoints) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(0.0f, 0.8f, 1.0f),  // cyan
                                            rVis, view, projection, lightCamPos);
                    }
                    for (const auto& p : g_silhouetteSourcePoints) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(1.0f, 0.2f, 0.9f),  // magenta
                                            rSil, view, projection, lightCamPos);
                    }
                }

                // ---- Phase U-1: soft partition heatmap (g_sphereMarker 再利用) ----
                if (g_softShowHeatmap && g_softPartReady) {
                    // ターゲットは数十万頂点なので draw 数を ~6k/~4k に自動間引き。
                    // 本格的には per-vertex color VBO 化 (Phase U-1.5)。
                    const float kRadius = 0.004f * g_sceneDiag;

                    if (screenMesh && g_softTgtField.valid) {
                        const int numV = SoftPartition::vertexCount(*screenMesh);
                        if ((int)g_softTgtField.probs.size() == numV) {
                            const int stride = std::max(1, numV / 6000);
                            for (int v = 0; v < numV; v += stride) {
                                glm::vec4 col = SoftPartition::blendVertexColor(
                                    g_softTgtField.probs[v], g_softTgtField.numGroups);
                                if (!softHeatmapVertexVisible(
                                        g_softTgtField.probs[v], g_softTgtField.numGroups))
                                    continue;   // per-group + show-groups filter
                                glm::vec3 wp = SoftPartition::vertexAt(*screenMesh, v);
                                g_sphereMarker.draw(shaderProgram, wp, glm::vec3(col),
                                                    kRadius, view, projection,
                                                    lightCamPos);
                            }
                        }
                    }
                    if (liverMesh3D && g_softSrcField.valid) {
                        const int numV = SoftPartition::vertexCount(*liverMesh3D);
                        if ((int)g_softSrcField.probs.size() == numV) {
                            const int stride = std::max(1, numV / 4000);
                            for (int v = 0; v < numV; v += stride) {
                                glm::vec4 col = SoftPartition::blendVertexColor(
                                    g_softSrcField.probs[v], g_softSrcField.numGroups);
                                if (!softHeatmapVertexVisible(
                                        g_softSrcField.probs[v], g_softSrcField.numGroups))
                                    continue;   // per-group + show-groups filter
                                glm::vec3 wp = SoftPartition::vertexAt(*liverMesh3D, v);
                                g_sphereMarker.draw(shaderProgram, wp, glm::vec3(col),
                                                    kRadius, view, projection,
                                                    lightCamPos);
                            }
                        }
                    }
                }

                // Phase 7b Step 1 (Plain W): source RIM chain debug overlay.
                //   `g_debugSourceRimChain` は頂点 index の chain。毎フレーム
                //   liverMesh3D->mVertices から現在位置を fetch するので、
                //   ICP / Live tracking で organ が動いてもマーカーが追従する
                //   (Cyclic correspondence と同じパターン)。
                if (g_showDebugSourceRimChain && liverMesh3D &&
                    !g_debugSourceRimChain.empty())
                {
                    const float rRim = RegRatios::markerCluster() * 1.0f;
                    const auto& V = liverMesh3D->mVertices;
                    const int nV3 = (int)V.size();
                    for (int idx : g_debugSourceRimChain) {
                        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
                        glm::vec3 p(V[idx*3], V[idx*3+1], V[idx*3+2]);
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(0.2f, 1.0f, 0.2f),  // green
                                            rRim, view, projection, lightCamPos);
                    }
                }

                // Phase 7c (REDGE/稜線): source ridge overlay (cyan).
                //   頂点 index 保持 → mVertices から fetch して mesh 追従。
                if (g_showDebugSourceRidge && liverMesh3D &&
                    !g_debugSourceRidge.empty())
                {
                    const float rRdg = RegRatios::markerCluster() * 1.0f;
                    const auto& V = liverMesh3D->mVertices;
                    const int nV3 = (int)V.size();
                    for (int idx : g_debugSourceRidge) {
                        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
                        glm::vec3 p(V[idx*3], V[idx*3+1], V[idx*3+2]);
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(0.10f, 0.90f, 1.0f),  // cyan
                                            rRdg, view, projection, lightCamPos);
                    }
                }

                // Phase 7b Step 2 (Shift+W): target boundary debug overlay.
                //   `g_debugTargetBoundaryPoints` は 3D 座標を直接保持。
                //   target は ICP で動かないので index lookup 不要 (source
                //   側との非対称はそのため)。
                if (g_showDebugTargetBoundary &&
                    !g_debugTargetBoundaryPoints.empty())
                {
                    const float rTgt = RegRatios::markerCluster() * 0.9f;
                    for (const auto& p : g_debugTargetBoundaryPoints) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(0.7f, 0.2f, 1.0f),  // purple
                                            rTgt, view, projection, lightCamPos);
                    }
                }

                // Phase 7c (REDGE/稜線): target ridge overlay (yellow, upper half).
                //   3D 直接保持 (target は静止)。
                if (g_showDebugTargetRidge &&
                    !g_debugTargetRidgePoints.empty())
                {
                    const float rRdg = RegRatios::markerCluster() * 0.9f;
                    for (const auto& p : g_debugTargetRidgePoints) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(1.0f, 0.85f, 0.10f),  // yellow
                                            rRdg, view, projection, lightCamPos);
                    }
                }

                // Phase 7b Step 3 (Ctrl+W): Shape Match best candidate
                //   predicted source position (赤点).
                //   `g_debugShapeMatchBestSrc` は best T を source 現在
                //   位置に適用して得た 3D 点群。mesh は動かないので、
                //   緑 (source 現在) vs 赤 (best 予測) を視覚比較すれば
                //   「Shape Match で source がどこに行くか」が一目で分
                //   かる。
                if (g_showDebugShapeMatch && !g_debugShapeMatchBestSrc.empty())
                {
                    const float rRed = RegRatios::markerCluster() * 1.1f;
                    for (const auto& p : g_debugShapeMatchBestSrc) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(1.0f, 0.15f, 0.15f),  // red
                                            rRed, view, projection, lightCamPos);
                    }
                }

                // Phase 7b Step 3c (Ctrl+Alt+W): Contour Sweep trial pose
                //   黄色点: 現在 sweep が試行中の "batch best" pose。
                //   赤 (global best) と共存して描画され、sweep の進行中
                //   毎フレーム位置が更新されるためアニメーションが見える。
                //   sweep 終了時に clear される。
                if (g_contourSweepShowTrial && !g_contourSweepTrialSrc.empty())
                {
                    const float rYel = RegRatios::markerCluster() * 0.95f;
                    for (const auto& p : g_contourSweepTrialSrc) {
                        g_sphereMarker.draw(shaderProgram, p,
                                            glm::vec3(1.0f, 0.95f, 0.15f),  // yellow
                                            rYel, view, projection, lightCamPos);
                    }
                }

                // Phase 7b Step 3c: Anchor / pivot visualization
                //   灰色小球: 20 target anchor (固定, AR 画面 plane 上に back-project)
                //            + 20 source pivot (best-in-batch T 適用後、黄色 rim 上)
                //   シアン大球: 現在 batch の i_tgt 番目 target anchor
                //              + j_src 番目 source pivot
                //   毎フレーム i_tgt/j_src が更新 → 動いている対応が確認できる。
                //   同色 (シアン) なので「どの target がどの source とペアか」が判る。
                if (g_shapeMatchSweepShowAnchors && g_contourSweepShowTrial)
                {
                    const float rSmall = RegRatios::markerCluster() * 0.80f;
                    const float rLarge = RegRatios::markerCluster() * 2.20f;
                    const glm::vec3 colGray(0.78f, 0.78f, 0.85f);
                    const glm::vec3 colHi  (0.10f, 0.95f, 1.00f);   // cyan

                    // Target anchors (back-projected on image plane)
                    for (size_t i = 0; i < g_contourSweepTgtAnchors3D.size(); i++) {
                        const bool hi = (int(i) == g_contourSweepCurrentITgt);
                        g_sphereMarker.draw(shaderProgram,
                            g_contourSweepTgtAnchors3D[i],
                            hi ? colHi : colGray,
                            hi ? rLarge : rSmall,
                            view, projection, lightCamPos);
                    }
                    // Source pivots (transformed to trial pose)
                    for (size_t j = 0; j < g_contourSweepSrcPivotsTrial.size(); j++) {
                        const bool hi = (int(j) == g_contourSweepCurrentJSrc);
                        g_sphereMarker.draw(shaderProgram,
                            g_contourSweepSrcPivotsTrial[j],
                            hi ? colHi : colGray,
                            hi ? rLarge : rSmall,
                            view, projection, lightCamPos);
                    }
                }

                // Phase 7b Step 3c++: Preview anchor/pivot dots.
                //   Plan A (anchor固定) means anchors are always extracted
                //   from the FULL target boundary — they don't move when
                //   the user adjusts the source pose. The source bbox is
                //   recomputed each frame and applied as a per-anchor
                //   TAG: anchors INSIDE bbox render in rainbow (matches
                //   what the sweep will see), OUTSIDE in dim gray with
                //   a smaller radius so the user can visually confirm
                //   the spatial gate. Source pivots get LR/CC label tint:
                //     PURE_RIGHT → green-tinted
                //     PURE_LEFT  → magenta-tinted
                //     BOUNDARY   → gray (屈曲帯 = 反転防止 free pass)
                //   So at a glance the user can spot reversed pairings.
                if (g_shapeMatchSweepPreviewAnchors)
                {
                    const int Np = std::max(2, g_shapeMatchSweepNTarget);
                    const int Ms = std::max(2, g_shapeMatchSweepNSource);
                    const float rPrev = RegRatios::markerCluster() * 1.30f;
                    const glm::vec3 colDim(0.55f, 0.55f, 0.55f);   // bbox-outside
                    const glm::vec3 colBoundary(0.70f, 0.70f, 0.72f);

                    const int W_img = (OrbitCam.calibWidth > 0)  ? OrbitCam.calibWidth
                                    : (g_boundaryDistMap.width > 0) ? g_boundaryDistMap.width
                                    : 1920;
                    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight
                                    : (g_boundaryDistMap.height > 0) ? g_boundaryDistMap.height
                                    : 1080;

                    // --- Target anchors (Plan A: always full boundary) -
                    std::vector<glm::vec3> preview_tgt_3D;
                    std::vector<uint8_t>   preview_tgt_inside;
                    if (!g_debugTargetBoundaryPoints.empty()) {
                        const glm::mat4 view_m = buildSilhouetteView();
                        const glm::mat4 proj_m = buildSilhouetteProj();

                        RimShape::extractSectorBasedTargetAnchors3D(
                            g_debugTargetBoundaryPoints,
                            view_m, proj_m, W_img, H_img,
                            Np, preview_tgt_3D);

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
                                preview_tgt_3D, src_rim_3D,
                                view_m, proj_m, W_img, H_img,
                                g_shapeMatchSweepFilterMarginPx,
                                preview_tgt_inside);
                        } else {
                            preview_tgt_inside.assign(
                                preview_tgt_3D.size(), (uint8_t)1);
                        }
                    }

                    // --- Source pivots (arc-length + LR label) ---------
                    std::vector<glm::vec3> preview_src_3D;
                    std::vector<int>       preview_src_vidx;
                    if (!g_debugSourceRimChain.empty() && liverMesh3D) {
                        RimShape::arclengthResampleRim3D(
                            g_debugSourceRimChain,
                            liverMesh3D->mVertices,
                            Ms, preview_src_3D,
                            &preview_src_vidx);
                    }

                    // Diagnostic (≤ once per ~2 sec)
                    static int dbg_cnt = 0;
                    if ((++dbg_cnt % 120) == 1) {
                        int n_in = 0;
                        for (uint8_t b : preview_tgt_inside) if (b) n_in++;
                        std::cout << "[Preview] anchor-fixed"
                                  << "  tgt_boundary_pts=" << g_debugTargetBoundaryPoints.size()
                                  << "  tgt_anchors=" << preview_tgt_3D.size()
                                  << " / " << Np
                                  << "  bbox_in=" << n_in << "/" << preview_tgt_inside.size()
                                  << "  src_pivots=" << preview_src_3D.size()
                                  << "  img=" << W_img << "x" << H_img
                                  << std::endl;
                    }

                    // Render target anchors: rainbow if inside bbox,
                    // dim gray (smaller) if outside.
                    for (size_t i = 0; i < preview_tgt_3D.size(); i++) {
                        const bool inside = (i < preview_tgt_inside.size())
                                            ? (bool)preview_tgt_inside[i]
                                            : true;
                        const float h_hue = float(i)
                            / float(std::max((size_t)1, preview_tgt_3D.size()));
                        const glm::vec3 col = inside
                            ? cyclicHsv2rgb(h_hue, 0.90f, 1.0f)
                            : colDim;
                        const float r_use = inside ? rPrev : (rPrev * 0.55f);
                        g_sphereMarker.draw(shaderProgram,
                            preview_tgt_3D[i], col, r_use,
                            view, projection, lightCamPos);
                    }

                    // Render source pivots with LR-label tint (rainbow
                    // hue from i still, but saturation/value modulated
                    // by label so reversal is visually obvious).
                    const bool lr_ok =
                        g_liverLR.valid()
                        && liverMesh3D
                        && (int)g_liverLR.labels.size()
                               == (int)(liverMesh3D->mVertices.size() / 3)
                        && preview_src_vidx.size() == preview_src_3D.size();
                    for (size_t i = 0; i < preview_src_3D.size(); i++) {
                        const float h_hue = float(i)
                            / float(std::max((size_t)1, preview_src_3D.size()));
                        glm::vec3 col = cyclicHsv2rgb(h_hue, 0.90f, 1.0f);
                        if (lr_ok) {
                            const int vi = preview_src_vidx[i];
                            if (vi >= 0 && vi < (int)g_liverLR.labels.size()) {
                                const uint8_t lbl = g_liverLR.labels[vi];
                                if (lbl == (uint8_t)LiverLeftRightLabel::BOUNDARY) {
                                    col = colBoundary;  // gray = 屈曲帯
                                }
                                // PURE_RIGHT / PURE_LEFT keep their
                                // rainbow hue so the user can still
                                // match anchor by index.
                            }
                        }
                        const float r_use = rPrev;
                        g_sphereMarker.draw(shaderProgram,
                            preview_src_3D[i], col, r_use,
                            view, projection, lightCamPos);
                    }
                }

                // Cyclic correspondence visualization (Shift+B):
                //   24 セクターを HSV で円環着色し、source (大球) と target
                //   (中球) を同色で描画 → 同じ色のペアが対応関係を示す。
                //   src は liverMesh3D の頂点 index 経由で現在位置を取得するので、
                //   後段の ICP で organ が動いてもマーカーが追従する。
                if (g_showCyclicCorrespondence && g_cyclicAvailable && liverMesh3D) {
                    const float rBase = RegRatios::markerCluster();
                    const float rSrc  = rBase * 1.6f;   // source: 大きい
                    const float rTgt  = rBase * 1.0f;   // target: 中サイズ
                    const auto& V = liverMesh3D->mVertices;
                    const int N = g_cyclicSectors;

                    for (int i = 0; i < N; i++) {
                        if ((int)g_cyclicPairValid.size() <= i) continue;
                        if (!g_cyclicPairValid[i]) continue;

                        float h = float(i) / float(std::max(1, N));
                        glm::vec3 col = cyclicHsv2rgb(h, 0.85f, 1.0f);

                        // Source: 現在の頂点位置 (ICP 後の姿勢を反映)
                        int vIdx = g_cyclicPairSrcVertIdx[i];
                        if (vIdx >= 0 && (size_t)vIdx * 3 + 2 < V.size()) {
                            glm::vec3 srcPos(V[vIdx*3], V[vIdx*3+1], V[vIdx*3+2]);
                            g_sphereMarker.draw(shaderProgram, srcPos, col, rSrc,
                                                view, projection, lightCamPos);
                        }

                        // Target: 不変なので保存済み 3D 位置を使用
                        g_sphereMarker.draw(shaderProgram, g_cyclicPairTgtPos[i],
                                            col, rTgt,
                                            view, projection, lightCamPos);
                    }
                }

                // -----------------------------------------------------------
                // Ctrl+G RIM pair visualization (g_ctrlgShowRimPairs)
                //   orange  = source RIM (LiverRegion::RIM band) — uses
                //             liverMesh3D vertex index, so the marker
                //             follows the mesh through ICP/CMA-ES updates.
                //   magenta = target RIM (image boundary; immutable).
                //   Populated by RegistrationActions::runBipopCmaesV3R
                //   wrapper at every Ctrl+G entry; persists until next
                //   session. AND-filtered with AR-vis when enabled.
                // -----------------------------------------------------------
                if (g_ctrlgShowRimPairs && g_ctrlgRimVizAvailable && liverMesh3D) {
                    const float rBase = RegRatios::markerCluster();
                    const float rSrc  = rBase * 1.2f;   // source: 中〜大
                    const float rTgt  = rBase * 0.9f;   // target: 中
                    const glm::vec3 colSrc(1.0f, 0.55f, 0.10f);  // orange
                    const glm::vec3 colTgt(1.0f, 0.20f, 0.85f);  // magenta
                    const auto&  V  = liverMesh3D->mVertices;
                    const size_t nV = V.size() / 3;

                    for (int vIdx : g_ctrlgRimSrcVertIdx) {
                        if (vIdx < 0 || (size_t)vIdx >= nV) continue;
                        glm::vec3 p(V[vIdx*3], V[vIdx*3+1], V[vIdx*3+2]);
                        g_sphereMarker.draw(shaderProgram, p, colSrc, rSrc,
                                            view, projection,
                                            lightCamPos);
                    }
                    for (const auto& p : g_ctrlgRimTgtPos) {
                        g_sphereMarker.draw(shaderProgram, p, colTgt, rTgt,
                                            view, projection,
                                            lightCamPos);
                    }
                }

                // -----------------------------------------------------------
                // [Phase D] Colored RIM pairs (K representatives).
                //   Same data-follows-mesh contract as Shift+B Cyclic viz:
                //   src is a liverMesh3D vertex index so spheres track
                //   subsequent ICP/Apply moves; tgt is fixed world coords.
                //   Sampling is done every frame (fast enough at N≈20k,
                //   K≤30) so the K pairs stay sensibly distributed even
                //   if WorstK/BestK rankings shift as the mesh moves.
                //
                //   Source data: g_lastRimPair* (publish/consume globals).
                //   - Set by Ctrl+G / Ctrl+Shift+G Phase F.5 [Phase A].
                //   - Restored from PoseEntry by poseApplyEntry [Phase B].
                //   When both vectors are empty (e.g. session bailed out
                //   before F.5, or an applied entry was non-Ctrl+G), the
                //   block silently draws nothing.
                //
                //   Colour: cyclicHsv2rgb(k/K, 0.85, 1.0) — same palette
                //   as Shift+B Cyclic correspondence so the two viewers
                //   are visually consistent (Shift+B's 24-sector wheel
                //   and our K=10 wheel both use HSV ramps).
                // -----------------------------------------------------------
                if (g_ctrlgShowColoredRimPairs &&
                    !g_lastRimPairSrcVertIdx.empty() &&
                    g_lastRimPairSrcVertIdx.size() == g_lastRimPairTgtPos.size() &&
                    liverMesh3D)
                {
                    const float rBase = RegRatios::markerCluster();
                    const float rSrc  = rBase * 1.6f;   // source: 大 (matches Shift+B)
                    const float rTgt  = rBase * 1.0f;   // target: 中
                    const auto& V  = liverMesh3D->mVertices;
                    const size_t nV = V.size() / 3;

                    const auto mode =
                        static_cast<RimPairSampling::Mode>(
                            g_ctrlgColoredRimMode);
                    std::vector<int> sel = RimPairSampling::sampleRimPairIndices(
                        g_lastRimPairSrcVertIdx,
                        g_lastRimPairTgtPos,
                        V,
                        mode,
                        g_ctrlgColoredRimN,
                        g_ctrlgColoredRimSeed);

                    const int K = (int)sel.size();
                    for (int k = 0; k < K; k++) {
                        const int i = sel[k];
                        if (i < 0 ||
                            (size_t)i >= g_lastRimPairSrcVertIdx.size()) continue;

                        // Colour by position in the selection (0..K-1),
                        // NOT by original pair index, so the K colours
                        // span the wheel cleanly regardless of how the
                        // sampler chose them.
                        const float h = float(k) / float(std::max(1, K));
                        const glm::vec3 col = cyclicHsv2rgb(h, 0.85f, 1.0f);

                        // Source: current vertex position (follows mesh).
                        const int vIdx = g_lastRimPairSrcVertIdx[i];
                        if (vIdx >= 0 && (size_t)vIdx * 3 + 2 < V.size()) {
                            glm::vec3 srcPos(V[vIdx*3],
                                             V[vIdx*3+1],
                                             V[vIdx*3+2]);
                            g_sphereMarker.draw(shaderProgram, srcPos, col, rSrc,
                                                view, projection,
                                                lightCamPos);
                        }

                        // Target: immutable world coord from the capture.
                        const glm::vec3& tgtPos = g_lastRimPairTgtPos[i];
                        g_sphereMarker.draw(shaderProgram, tgtPos, col, rTgt,
                                            view, projection,
                                            lightCamPos);
                    }
                    (void)nV;   // silence unused-warn if compiler complains
                }

                // -----------------------------------------------------------
                // Ctrl+Shift+G silhouette projection visualization.
                //   Captured ONCE per Ctrl+Shift+G press (see Phase E.5 in
                //   runBipopCmaesV3RS). Each stored point keeps its world
                //   position from the captured pose, so the markers stay
                //   put even if the liver is moved by a subsequent
                //   registration -- they show "where the previous sil
                //   result placed the points".
                //
                //   Colour semantics (revised):
                //     RED (sentinel >= 9000)  : projection landed OUTSIDE
                //                               the SAM mask entirely.
                //                               These points contribute
                //                               nothing to the silhouette
                //                               loss -- a HIGH count here
                //                               is the real failure mode
                //                               (scale too large / pose
                //                               translated off the organ).
                //     GREEN  (< 5  px)        : on the silhouette boundary
                //                               -> silhouette match.
                //     YELLOW (5-30 px)        : near boundary.
                //     BLUE   (>= 30 px)       : inside the mask, deep
                //                               interior. EXPECTED for
                //                               non-rim voxels; not a
                //                               problem.
                //   Sentinels are drawn slightly larger to surface them
                //   over the inside-mask voxels.
                // -----------------------------------------------------------
                if (g_silProjShow && g_silProjDebug.valid) {
                    const float rBase = RegRatios::markerCluster();
                    const float rSil  = rBase * 0.55f;
                    const float rOut  = rBase * 0.70f;  // out-of-mask: bigger
                    for (const auto& pt : g_silProjDebug.pts) {
                        glm::vec3 col;
                        float radius = rSil;
                        if (pt.dist_px >= 9000.0f) {
                            // OUTSIDE mask -- the real diagnostic signal
                            col = glm::vec3(0.95f, 0.18f, 0.18f);             // red
                            radius = rOut;
                        } else if (pt.dist_px < 5.0f) {
                            col = glm::vec3(0.20f, 0.95f, 0.20f);             // green
                        } else if (pt.dist_px < 30.0f) {
                            const float t = (pt.dist_px - 5.0f) / 25.0f;
                            col = glm::vec3(0.20f + 0.75f * t,
                                            0.95f,
                                            0.20f * (1.0f - t));              // green->yellow
                        } else {
                            // inside the mask, deep interior -- expected
                            col = glm::vec3(0.25f, 0.55f, 0.95f);             // blue / cyan
                        }
                        g_sphereMarker.draw(shaderProgram, pt.world_pos,
                                            col, radius,
                                            view, projection,
                                            lightCamPos);
                    }
                }

                // -----------------------------------------------------------
                // Liver region visualization (Shift+R):
                //   赤=anterior_core, 橙=rim, 青=posterior の subsample を
                //   球マーカーで描画。liverMesh3D の頂点 index 経由で現在
                //   位置を取るので registration 後も追従する。
                // -----------------------------------------------------------
                if (g_showLiverRegion && g_liverRegion.valid() && liverMesh3D) {
                    const float rBase = RegRatios::markerCluster();
                    const float rAnt  = rBase * 1.0f;
                    const float rRim  = rBase * 1.4f;   // rim を強調
                    const float rPost = rBase * 0.7f;
                    const auto& V = liverMesh3D->mVertices;
                    const size_t nV = V.size() / 3;

                    auto drawIdx = [&](const std::vector<int>& idxs,
                                       const glm::vec3& col, float r) {
                        for (int vIdx : idxs) {
                            if (vIdx < 0 || (size_t)vIdx >= nV) continue;
                            glm::vec3 p(V[vIdx*3], V[vIdx*3+1], V[vIdx*3+2]);
                            g_sphereMarker.draw(shaderProgram, p, col, r,
                                                view, projection,
                                                lightCamPos);
                        }
                    };

                    // 赤: anterior_core
                    drawIdx(g_regionVizIdxAnt,  glm::vec3(0.86f, 0.22f, 0.22f), rAnt);
                    // 橙: rim
                    drawIdx(g_regionVizIdxRim,  glm::vec3(0.97f, 0.65f, 0.10f), rRim);
                    // 青: posterior
                    drawIdx(g_regionVizIdxPost, glm::vec3(0.20f, 0.45f, 0.86f), rPost);
                }

                // -----------------------------------------------------------
                //  LiverLeftRight (Y / Shift+Y) 球マーカー描画
                //   緑=pure_right, 黄=boundary, 紫=pure_left の subsample を
                //   球マーカーで描画。liverMesh3D の頂点 index 経由で現在
                //   位置を取るので registration 後も追従する。
                //
                //   Shift+R (anterior/rim/posterior) と同時 ON も可能。
                //   その場合、同じ頂点に複数の球マーカーが重なって描画される。
                // -----------------------------------------------------------
                if (g_showLiverLR && g_liverLR.valid() && liverMesh3D) {
                    const float rBase = RegRatios::markerCluster();
                    const float rR    = rBase * 1.0f;
                    const float rBnd  = rBase * 1.4f;   // 境界帯を強調
                    const float rL    = rBase * 1.0f;
                    const auto& V = liverMesh3D->mVertices;
                    const size_t nV = V.size() / 3;

                    auto drawIdxLR = [&](const std::vector<int>& idxs,
                                         const glm::vec3& col, float r) {
                        for (int vIdx : idxs) {
                            if (vIdx < 0 || (size_t)vIdx >= nV) continue;
                            glm::vec3 p(V[vIdx*3], V[vIdx*3+1], V[vIdx*3+2]);
                            g_sphereMarker.draw(shaderProgram, p, col, r,
                                                view, projection,
                                                lightCamPos);
                        }
                    };

                    // 緑: pure right (右葉)
                    drawIdxLR(g_lrVizIdxR,        glm::vec3(0.20f, 0.70f, 0.30f), rR);
                    // 黄: boundary (鎌状間膜)
                    drawIdxLR(g_lrVizIdxBoundary, glm::vec3(0.95f, 0.85f, 0.10f), rBnd);
                    // 紫: pure left (左葉)
                    drawIdxLR(g_lrVizIdxL,        glm::vec3(0.65f, 0.25f, 0.75f), rL);
                }

                // -----------------------------------------------------------
                //  LiverQuad (H) 4象限球マーカー描画
                //   緑=ant_right, 紫=ant_left, 青=pos_right, 橙=pos_left
                //   重複所属方式 (案D): rim と boundary はそれぞれ前後・左右
                //   両方に所属するため、該当頂点は複数のマーカーが重なって描画される。
                //   球サイズを4種で変えて識別しやすくする (AR大、AL中、PR中、PL小)。
                //
                //   Shift+R / Y / H すべて同時 ON できるが、視認性が悪くなるため
                //   通常は H のみ ON で運用するのが推奨。
                // -----------------------------------------------------------
                if (g_showLiverQuad && liverMesh3D &&
                    !(g_quadVizIdxAR.empty() && g_quadVizIdxAL.empty() &&
                      g_quadVizIdxPR.empty() && g_quadVizIdxPL.empty())) {
                    const float rBase = RegRatios::markerCluster();
                    // 重複描画されるので、球のサイズを変えて識別性を上げる
                    const float rAR = rBase * 1.4f;   // 緑: 大
                    const float rAL = rBase * 1.1f;   // 紫: 中大
                    const float rPR = rBase * 0.9f;   // 青: 中小
                    const float rPL = rBase * 0.7f;   // 橙: 小
                    const auto& V = liverMesh3D->mVertices;
                    const size_t nV = V.size() / 3;

                    auto drawIdxQuad = [&](const std::vector<int>& idxs,
                                           const glm::vec3& col, float r) {
                        for (int vIdx : idxs) {
                            if (vIdx < 0 || (size_t)vIdx >= nV) continue;
                            glm::vec3 p(V[vIdx*3], V[vIdx*3+1], V[vIdx*3+2]);
                            g_sphereMarker.draw(shaderProgram, p, col, r,
                                                view, projection,
                                                lightCamPos);
                        }
                    };

                    // 緑: ant_right
                    drawIdxQuad(g_quadVizIdxAR, glm::vec3(0.20f, 0.70f, 0.30f), rAR);
                    // 紫: ant_left
                    drawIdxQuad(g_quadVizIdxAL, glm::vec3(0.65f, 0.25f, 0.75f), rAL);
                    // 青: pos_right
                    drawIdxQuad(g_quadVizIdxPR, glm::vec3(0.20f, 0.45f, 0.86f), rPR);
                    // 橙: pos_left
                    drawIdxQuad(g_quadVizIdxPL, glm::vec3(0.97f, 0.55f, 0.10f), rPL);
                }

                // -----------------------------------------------------------
                //  LiverCranioCaudal (Shift+H) 球マーカー描画
                //   黄=CRANIAL(頭側), 青=CAUDAL(足側) を subsample 球で表示。
                //   liverMesh3D の頂点 index 経由で現在位置を取るので
                //   registration 後も追従する。
                //
                //   Shift+R / Y / H / Shift+H すべて同時 ON できるが、
                //   視認性が悪くなるため通常は Shift+H のみ ON で運用するのが推奨。
                //
                //   Phase 1 では registration には影響しない (可視化のみ)。
                // -----------------------------------------------------------
                if (g_showLiverCC && liverMesh3D &&
                    !(g_ccVizIdxCranial.empty() && g_ccVizIdxCaudal.empty()))
                {
                    const float rBase = RegRatios::markerCluster();
                    const float rCr   = rBase * 1.1f;   // 識別性を上げるため LR より少し大きく
                    const float rCa   = rBase * 1.1f;
                    const auto& V = liverMesh3D->mVertices;
                    const size_t nV = V.size() / 3;

                    auto drawIdxCC = [&](const std::vector<int>& idxs,
                                         const glm::vec3& col, float r) {
                        for (int vIdx : idxs) {
                            if (vIdx < 0 || (size_t)vIdx >= nV) continue;
                            glm::vec3 p(V[vIdx*3], V[vIdx*3+1], V[vIdx*3+2]);
                            g_sphereMarker.draw(shaderProgram, p, col, r,
                                                view, projection,
                                                lightCamPos);
                        }
                    };

                    // 黄: CRANIAL (頭側)
                    drawIdxCC(g_ccVizIdxCranial, glm::vec3(0.95f, 0.85f, 0.10f), rCr);
                    // 青: CAUDAL (足側)
                    drawIdxCC(g_ccVizIdxCaudal,  glm::vec3(0.15f, 0.45f, 0.90f), rCa);
                }

                // 対応点の描画
                {
                    bool activeSelection =
                        (registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS ||
                         registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS ||
                         registrationHandle.state == RegistrationData::READY_TO_REGISTER);
                    if (activeSelection || g_showCorrespondencePoints) {
                        const float rCorr = RegRatios::markerCorrespondence();
                        for (size_t i = 0; i < registrationHandle.boardPoints.size(); i++) {
                            glm::vec3 color = getPointColor(i, true);
                            g_sphereMarker.draw(shaderProgram, registrationHandle.boardPoints[i],
                                                color, rCorr, view, projection, lightCamPos);
                        }
                        for (size_t i = 0; i < registrationHandle.objectPoints.size(); i++) {
                            glm::vec3 color = getPointColor(i, false);
                            g_sphereMarker.draw(shaderProgram, registrationHandle.objectPoints[i],
                                                color, rCorr, view, projection, lightCamPos);
                        }
                    }
                }

                // ============================================================
                //  デバッグ AABB 描画 (チャット 10):
                //   赤 = target_full の AABB (8 コーナー + center 計 9 点)
                //   緑 = source の post-transform AABB (Apply Init Pose 後の状態)
                //   両者の center が重なっていれば OK。
                //  toggle: g_showDebugBB (ImGui のチェックボックスで切替)
                // ============================================================
                if (g_showDebugBB) {
                    const float r_marker = 0.012f;   // 小さめ
                    auto drawBoxMarkers = [&](const glm::vec3& mn,
                                              const glm::vec3& mx,
                                              const glm::vec3& ctr,
                                              const glm::vec3& corner_color,
                                              const glm::vec3& center_color,
                                              float r_center_scale)
                    {
                        const glm::vec3 corners[8] = {
                                                      {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z},
                                                      {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
                                                      {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
                                                      {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z},
                                                      };
                        for (int i = 0; i < 8; i++) {
                            g_sphereMarker.draw(shaderProgram, corners[i],
                                                corner_color, r_marker,
                                                view, projection, lightCamPos);
                        }
                        // center は少し大きく目立たせる
                        g_sphereMarker.draw(shaderProgram, ctr,
                                            center_color, r_marker * r_center_scale,
                                            view, projection, lightCamPos);
                    };

                    if (g_targetAabbFull.valid) {
                        drawBoxMarkers(g_targetAabbFull.min,
                                       g_targetAabbFull.max,
                                       g_targetAabbFull.center,
                                       glm::vec3(0.95f, 0.20f, 0.20f),   // 赤コーナー
                                       glm::vec3(1.00f, 0.05f, 0.05f),   // 濃赤センター
                                       1.6f);
                    }
                    if (g_dbgSourceBB_valid) {
                        drawBoxMarkers(g_dbgSourceBB_min,
                                       g_dbgSourceBB_max,
                                       g_dbgSourceBB_center,
                                       glm::vec3(0.30f, 0.95f, 0.30f),   // 緑コーナー
                                       glm::vec3(0.05f, 1.00f, 0.05f),   // 濃緑センター
                                       1.6f);
                    }
                }

                // ビューポートをフルウィンドウに復元（ImGui描画用）
                glViewport(0, 0, gWindowWidth, gWindowHeight);
            } // else（通常1画面描画）
        } // kRegistration

        // ================================================================
        // Phase 7b Step 3d — 2D projection debug popups (CB0 / CB1 / CB2)
        //   Three independent ImGui windows that visualize the source rim
        //   2D projection (with pivots + LR tinting + right-start marker)
        //   and the target contour lower-half (with anchors + right-end
        //   marker). Each is a self-contained canvas using ImDrawList;
        //   no GL state is touched, so they coexist with all 3D rendering.
        //
        //   CB0 = raw RIM 2D projection (points only)   ← STAGE 0 debug
        //   CB1 = envelope + pivots + start marker
        //   CB2 = target lower-half + anchors
        // ================================================================
        // --- CB0: Raw RIM 2D projection popup (points only, no ordering) ---
        //   STAGE 0: project every g_debugSourceRimChain vertex to 2D pixel
        //   space using the AR camera, and render as colored dots. No
        //   ordering, no envelope, no pivots. Lets the user see the actual
        //   point distribution and decide whether the anatomical source
        //   RIM is an open arch or a closed loop, BEFORE any algorithm
        //   tries to impose a topology on it.
        if (g_debugShow2DProjPopup_RawRim && gApp.mode == AppMode::kRegistration) {
            ImGui::SetNextWindowSize(ImVec2(820, 540), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Raw RIM 2D projection (Step 3d Stage 0)##silsw_rawrim_popup",
                             &g_debugShow2DProjPopup_RawRim))
            {
                // ---- 1. Gather rim chain (auto-populate if empty) ----
                bool can_render = true;
                std::string fail_reason;

                if (!liverMesh3D) {
                    can_render = false;
                    fail_reason = "liverMesh3D is null";
                }
                if (can_render && g_debugSourceRimChain.empty()) {
                    if (!populateDebugSourceRimChain()) {
                        can_render = false;
                        fail_reason = "populateDebugSourceRimChain failed (press W first)";
                    }
                }
                if (can_render && g_debugSourceRimChain.empty()) {
                    can_render = false;
                    fail_reason = "g_debugSourceRimChain still empty";
                }

                if (!can_render) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                       "Cannot render: %s", fail_reason.c_str());
                } else {
                    // ---- 2. Build view/proj and project ----
                    const glm::mat4 view_m = buildSilhouetteView();
                    const glm::mat4 proj_m = buildSilhouetteProj();
                    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
                    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
                    const glm::mat4 M = proj_m * view_m;

                    const auto& Vmesh = liverMesh3D->mVertices;
                    const int nV3 = (int)Vmesh.size();
                    const bool lrValid = g_liverLR.valid();
                    const int  nLR     = lrValid ? (int)g_liverLR.labels.size() : 0;

                    struct Pt2 {
                        glm::vec2 p;
                        uint8_t   lr;     // 0=PURE_RIGHT, 1=PURE_LEFT, 2=BOUNDARY, 255=unknown
                        bool      onscreen;
                    };
                    std::vector<Pt2> pts;
                    pts.reserve(g_debugSourceRimChain.size());

                    int n_input    = (int)g_debugSourceRimChain.size();
                    int n_projected = 0;
                    int n_onscreen  = 0;
                    int n_R = 0, n_L = 0, n_B = 0, n_U = 0;

                    for (int idx : g_debugSourceRimChain) {
                        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
                        const glm::vec3 p3(Vmesh[idx*3],
                                           Vmesh[idx*3+1],
                                           Vmesh[idx*3+2]);
                        const glm::vec4 clip = M * glm::vec4(p3, 1.0f);
                        if (clip.w < 1e-9f) continue;
                        n_projected++;
                        const float ndcx = clip.x / clip.w;
                        const float ndcy = clip.y / clip.w;
                        const float px = (ndcx + 1.0f) * 0.5f * float(W_img);
                        const float py = (1.0f - ndcy) * 0.5f * float(H_img);
                        Pt2 q;
                        q.p = glm::vec2(px, py);
                        // On-screen test (allow a small margin so the
                        // header count matches what the canvas can show)
                        q.onscreen = (px >= 0.0f && px <= float(W_img) &&
                                      py >= 0.0f && py <= float(H_img));
                        if (q.onscreen) n_onscreen++;
                        q.lr = 255;
                        if (lrValid && idx < nLR) {
                            q.lr = g_liverLR.labels[idx];
                            switch (q.lr) {
                                case LiverLeftRightLabel::PURE_RIGHT: n_R++; break;
                                case LiverLeftRightLabel::PURE_LEFT:  n_L++; break;
                                case LiverLeftRightLabel::BOUNDARY:   n_B++; break;
                                default:                              n_U++; break;
                            }
                        } else {
                            n_U++;
                        }
                        pts.push_back(q);
                    }

                    // ---- 3. 2D centroid of projected rim ----
                    glm::dvec2 sum2d(0.0);
                    for (const auto& q : pts) sum2d += glm::dvec2(q.p);
                    const glm::vec2 cen2D = (pts.empty())
                        ? glm::vec2(float(W_img) * 0.5f, float(H_img) * 0.5f)
                        : glm::vec2(sum2d / double(pts.size()));

                    // ---- 4. PURE_RIGHT 3D centroid → 2D ----
                    glm::vec2 rightCen2D(-1e6f);
                    bool      rightCen2DValid = false;
                    if (lrValid) {
                        glm::dvec3 sumR(0.0);
                        int nR3 = 0;
                        for (int i = 0; i < nLR; i++) {
                            if (i * 3 + 2 >= nV3) break;
                            if (g_liverLR.labels[i] !=
                                LiverLeftRightLabel::PURE_RIGHT) continue;
                            sumR += glm::dvec3(Vmesh[i*3],
                                               Vmesh[i*3+1],
                                               Vmesh[i*3+2]);
                            nR3++;
                        }
                        if (nR3 > 0) {
                            const glm::vec3 rc3(sumR / double(nR3));
                            const glm::vec4 clip = M * glm::vec4(rc3, 1.0f);
                            if (clip.w > 1e-9f) {
                                const float ndcx = clip.x / clip.w;
                                const float ndcy = clip.y / clip.w;
                                rightCen2D = glm::vec2(
                                    (ndcx + 1.0f) * 0.5f * float(W_img),
                                    (1.0f - ndcy) * 0.5f * float(H_img));
                                rightCen2DValid = true;
                            }
                        }
                    }

                    // ---- 5. 2D bbox of projected rim ----
                    float bb_minx = 1e30f, bb_miny = 1e30f;
                    float bb_maxx = -1e30f, bb_maxy = -1e30f;
                    for (const auto& q : pts) {
                        bb_minx = std::min(bb_minx, q.p.x);
                        bb_miny = std::min(bb_miny, q.p.y);
                        bb_maxx = std::max(bb_maxx, q.p.x);
                        bb_maxy = std::max(bb_maxy, q.p.y);
                    }
                    if (pts.empty()) {
                        bb_minx = bb_miny = bb_maxx = bb_maxy = 0.0f;
                    }

                    // ---- 6. Header info ----
                    ImGui::Text("Raw AR-projection of g_debugSourceRimChain (no ordering)");
                    ImGui::Text("  img=%dx%d   input=%d   projected=%d   on-screen=%d",
                                W_img, H_img, n_input, n_projected, n_onscreen);
                    if (lrValid) {
                        ImGui::Text("  LR labels:  PURE_RIGHT=%d  PURE_LEFT=%d  "
                                    "BOUNDARY=%d  unknown=%d",
                                    n_R, n_L, n_B, n_U);
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                            "  [!] g_liverLR not valid — points drawn in neutral gray");
                    }
                    ImGui::Text("  2D bbox px:  x=[%.0f, %.0f]  y=[%.0f, %.0f]  "
                                "(w=%.0f, h=%.0f)",
                                bb_minx, bb_maxx, bb_miny, bb_maxy,
                                bb_maxx - bb_minx, bb_maxy - bb_miny);
                    ImGui::Text("  2D centroid: (%.0f, %.0f)",
                                cen2D.x, cen2D.y);
                    if (rightCen2DValid) {
                        ImGui::Text("  PURE_RIGHT 3D centroid → 2D: (%.0f, %.0f)",
                                    rightCen2D.x, rightCen2D.y);
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                            "  [!] PURE_RIGHT 3D centroid unavailable");
                    }
                    ImGui::Separator();

                    // ---- 7. Canvas ----
                    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    if (avail.x < 100.0f) avail.x = 100.0f;
                    if (avail.y < 100.0f) avail.y = 100.0f;
                    const float canvas_w = avail.x;
                    const float canvas_h = avail.y;
                    ImVec2 canvas_p1(canvas_p0.x + canvas_w, canvas_p0.y + canvas_h);

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(20, 20, 26, 255));
                    dl->AddRect      (canvas_p0, canvas_p1, IM_COL32(120, 120, 130, 255));

                    // Image → canvas scale (preserve aspect, fit)
                    const float sx = canvas_w / float(W_img);
                    const float sy = canvas_h / float(H_img);
                    const float s  = (sx < sy) ? sx : sy;
                    const float ox = canvas_p0.x + 0.5f * (canvas_w - s * float(W_img));
                    const float oy = canvas_p0.y + 0.5f * (canvas_h - s * float(H_img));
                    auto px2cv = [&](const glm::vec2& p) -> ImVec2 {
                        return ImVec2(ox + p.x * s, oy + p.y * s);
                    };

                    // Image frame outline (= AR image extent)
                    dl->AddRect(
                        px2cv(glm::vec2(0.0f,        0.0f)),
                        px2cv(glm::vec2(float(W_img), float(H_img))),
                        IM_COL32(80, 80, 90, 255));

                    // ---- 8. Draw points (NO lines) ----
                    //   PURE_RIGHT = red, PURE_LEFT = blue,
                    //   BOUNDARY = gray, unknown = neutral light gray.
                    //   Filled circles 3 px radius; outline visible
                    //   over both bg and image-frame area.
                    for (const auto& q : pts) {
                        ImU32 col = IM_COL32(200, 200, 200, 230);   // unknown
                        switch (q.lr) {
                            case LiverLeftRightLabel::PURE_RIGHT:
                                col = IM_COL32(230,  70,  70, 240); break;
                            case LiverLeftRightLabel::PURE_LEFT:
                                col = IM_COL32( 80, 110, 240, 240); break;
                            case LiverLeftRightLabel::BOUNDARY:
                                col = IM_COL32(170, 170, 175, 220); break;
                            default: break;
                        }
                        const ImVec2 c = px2cv(q.p);
                        dl->AddCircleFilled(c, 3.0f, col);
                    }

                    // ---- 9. 2D centroid (yellow cross) ----
                    {
                        const ImVec2 c = px2cv(cen2D);
                        const float d = 9.0f;
                        const ImU32 yc = IM_COL32(255, 240, 80, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y), ImVec2(c.x + d, c.y), yc, 2.0f);
                        dl->AddLine(ImVec2(c.x, c.y - d), ImVec2(c.x, c.y + d), yc, 2.0f);
                        dl->AddText(ImVec2(c.x + 10, c.y - 18), yc, "2D centroid");
                    }

                    // ---- 10. PURE_RIGHT 3D centroid → 2D (cyan cross) ----
                    if (rightCen2DValid) {
                        const ImVec2 c = px2cv(rightCen2D);
                        const float d = 9.0f;
                        const ImU32 cy = IM_COL32(100, 230, 230, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y - d),
                                    ImVec2(c.x + d, c.y + d), cy, 2.0f);
                        dl->AddLine(ImVec2(c.x - d, c.y + d),
                                    ImVec2(c.x + d, c.y - d), cy, 2.0f);
                        dl->AddCircle(c, 7.0f, cy, 0, 1.5f);
                        dl->AddText(ImVec2(c.x + 10, c.y + 8),
                                    cy, "PURE_RIGHT 3D centroid → 2D");
                    }

                    // ---- 11. Legend (small, top-left of canvas) ----
                    {
                        const float lx = canvas_p0.x + 8.0f;
                        float       ly = canvas_p0.y + 8.0f;
                        auto legLine = [&](ImU32 col, const char* txt) {
                            dl->AddCircleFilled(ImVec2(lx + 6.0f, ly + 7.0f),
                                                3.0f, col);
                            dl->AddText(ImVec2(lx + 16.0f, ly), col, txt);
                            ly += 16.0f;
                        };
                        legLine(IM_COL32(230,  70,  70, 240), "PURE_RIGHT");
                        legLine(IM_COL32( 80, 110, 240, 240), "PURE_LEFT");
                        legLine(IM_COL32(170, 170, 175, 220), "BOUNDARY");
                    }

                    ImGui::Dummy(ImVec2(canvas_w, canvas_h));
                }
                ImGui::End();
            }
        }

        // --- CB0.1: Smoothed RIM 2D projection (grid + KNN) ---
        //   STAGE 0.1: same raw projection as CB0, but with two cleanup
        //   passes. Output is still an unordered point set so the
        //   comparison against CB0 is honest.
        //
        //   Pass 1 (grid aggregation):
        //     bin all projected points into G×G px cells; emit one
        //     centroid per non-empty cell, with LR label = majority
        //     vote. Kills density-driven jaggedness and reduces the
        //     point count from O(254) to typically O(30-80).
        //   Pass 2 (KNN smoothing, optional):
        //     for each centroid, replace its position with the mean of
        //     its K nearest 2D neighbours; iterate N times. K=0 or
        //     iters=0 skips this pass entirely.
        if (g_debugShow2DProjPopup_RawRimSmoothed &&
            gApp.mode == AppMode::kRegistration)
        {
            ImGui::SetNextWindowSize(ImVec2(820, 560), ImGuiCond_FirstUseEver);
            if (ImGui::Begin(
                    "Smoothed RIM 2D projection (Step 3d Stage 0.1)"
                    "##silsw_rawrim_smoothed_popup",
                    &g_debugShow2DProjPopup_RawRimSmoothed))
            {
                bool can_render = true;
                std::string fail_reason;

                if (!liverMesh3D) {
                    can_render = false;
                    fail_reason = "liverMesh3D is null";
                }
                if (can_render && g_debugSourceRimChain.empty()) {
                    if (!populateDebugSourceRimChain()) {
                        can_render = false;
                        fail_reason = "populateDebugSourceRimChain failed "
                                      "(press W first)";
                    }
                }
                if (can_render && g_debugSourceRimChain.empty()) {
                    can_render = false;
                    fail_reason = "g_debugSourceRimChain still empty";
                }

                if (!can_render) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                       "Cannot render: %s", fail_reason.c_str());
                } else {
                    // ---- 1. Project all rim chain to 2D (same as CB0) ----
                    const glm::mat4 view_m = buildSilhouetteView();
                    const glm::mat4 proj_m = buildSilhouetteProj();
                    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
                    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
                    const glm::mat4 M = proj_m * view_m;

                    const auto& Vmesh = liverMesh3D->mVertices;
                    const int nV3 = (int)Vmesh.size();
                    const bool lrValid = g_liverLR.valid();
                    const int  nLR     = lrValid ? (int)g_liverLR.labels.size() : 0;

                    std::vector<glm::vec2> raw_pts;
                    std::vector<uint8_t>   raw_lr;
                    raw_pts.reserve(g_debugSourceRimChain.size());
                    raw_lr.reserve(g_debugSourceRimChain.size());

                    for (int idx : g_debugSourceRimChain) {
                        if (idx < 0 || idx * 3 + 2 >= nV3) continue;
                        const glm::vec3 p3(Vmesh[idx*3],
                                           Vmesh[idx*3+1],
                                           Vmesh[idx*3+2]);
                        const glm::vec4 clip = M * glm::vec4(p3, 1.0f);
                        if (clip.w < 1e-9f) continue;
                        const float ndcx = clip.x / clip.w;
                        const float ndcy = clip.y / clip.w;
                        const float px = (ndcx + 1.0f) * 0.5f * float(W_img);
                        const float py = (1.0f - ndcy) * 0.5f * float(H_img);
                        raw_pts.emplace_back(px, py);
                        uint8_t lr = 255;
                        if (lrValid && idx < nLR) lr = g_liverLR.labels[idx];
                        raw_lr.push_back(lr);
                    }
                    const int N_raw = (int)raw_pts.size();

                    // ---- 2. Grid aggregation ----
                    //   Hash bin = (cellX, cellY). Per bin: accumulate
                    //   sum + count + LR-label tally; emit centroid.
                    const float grid = std::max(1.0f,
                                                g_rawRimSmooth_GridPx);
                    struct Cell {
                        glm::dvec2 sum;
                        int        cnt;
                        int        n_R;
                        int        n_L;
                        int        n_B;
                        int        n_U;
                    };
                    // Use long long key (32-bit cellX in low, cellY in high)
                    auto cellKey = [grid](float x, float y) -> long long {
                        // Offset by +2^15 to keep both halves non-negative
                        // for typical 1920x1080 with grid up to 80 px.
                        const int cx = int(std::floor(x / grid));
                        const int cy = int(std::floor(y / grid));
                        const long long ux = (long long)(cx + (1 << 15));
                        const long long uy = (long long)(cy + (1 << 15));
                        return (uy << 20) | (ux & 0xFFFFFLL);
                    };
                    std::unordered_map<long long, Cell> bins;
                    bins.reserve(size_t(N_raw));
                    for (int i = 0; i < N_raw; i++) {
                        const long long k = cellKey(raw_pts[i].x,
                                                    raw_pts[i].y);
                        auto& C = bins[k];
                        if (C.cnt == 0) {
                            C.sum = glm::dvec2(0.0);
                            C.n_R = C.n_L = C.n_B = C.n_U = 0;
                        }
                        C.sum += glm::dvec2(raw_pts[i]);
                        C.cnt++;
                        switch (raw_lr[i]) {
                            case LiverLeftRightLabel::PURE_RIGHT: C.n_R++; break;
                            case LiverLeftRightLabel::PURE_LEFT:  C.n_L++; break;
                            case LiverLeftRightLabel::BOUNDARY:   C.n_B++; break;
                            default:                              C.n_U++; break;
                        }
                    }
                    std::vector<glm::vec2> smo_pts;
                    std::vector<uint8_t>   smo_lr;
                    std::vector<int>       smo_cnt;
                    smo_pts.reserve(bins.size());
                    smo_lr.reserve(bins.size());
                    smo_cnt.reserve(bins.size());
                    for (auto& kv : bins) {
                        const Cell& C = kv.second;
                        smo_pts.emplace_back(C.sum / double(C.cnt));
                        // Majority LR (ties broken: R > L > B > U)
                        uint8_t maj = 255;
                        int best = -1;
                        if (C.n_R > best) { best = C.n_R; maj = LiverLeftRightLabel::PURE_RIGHT; }
                        if (C.n_L > best) { best = C.n_L; maj = LiverLeftRightLabel::PURE_LEFT;  }
                        if (C.n_B > best) { best = C.n_B; maj = LiverLeftRightLabel::BOUNDARY;   }
                        if (C.n_U > best) { best = C.n_U; maj = 255; }
                        smo_lr.push_back(maj);
                        smo_cnt.push_back(C.cnt);
                    }
                    const int N_cells = (int)smo_pts.size();

                    // ---- 3. KNN smoothing ----
                    //   For each point, find K nearest in 2D (brute
                    //   force; N is small after binning), replace with
                    //   their mean position. Iterate.
                    const int K     = std::max(0, g_rawRimSmooth_KnnK);
                    const int iters = std::max(0, g_rawRimSmooth_KnnIters);
                    if (K > 0 && iters > 0 && N_cells >= 2) {
                        const int Keff = std::min(K, N_cells - 1);
                        std::vector<glm::vec2> tmp(N_cells);
                        for (int it = 0; it < iters; it++) {
                            // For each i find Keff nearest js, average
                            // their (current) positions.
                            std::vector<std::pair<float, int>> dist_idx;
                            dist_idx.reserve(N_cells);
                            for (int i = 0; i < N_cells; i++) {
                                dist_idx.clear();
                                for (int j = 0; j < N_cells; j++) {
                                    if (j == i) continue;
                                    const float dx = smo_pts[j].x - smo_pts[i].x;
                                    const float dy = smo_pts[j].y - smo_pts[i].y;
                                    dist_idx.emplace_back(dx*dx + dy*dy, j);
                                }
                                // Partial nth_element for Keff smallest
                                std::nth_element(
                                    dist_idx.begin(),
                                    dist_idx.begin() + Keff,
                                    dist_idx.end(),
                                    [](const std::pair<float,int>& a,
                                       const std::pair<float,int>& b){
                                        return a.first < b.first;
                                    });
                                glm::dvec2 sum(0.0);
                                for (int n = 0; n < Keff; n++) {
                                    sum += glm::dvec2(smo_pts[dist_idx[n].second]);
                                }
                                // Include self with weight 1 to avoid
                                // run-away drift (1 + Keff weighting)
                                sum += glm::dvec2(smo_pts[i]);
                                tmp[i] = glm::vec2(sum / double(Keff + 1));
                            }
                            smo_pts.swap(tmp);
                        }
                    }

                    // ---- 4. Stats on smoothed point set ----
                    glm::dvec2 sum2d(0.0);
                    for (const auto& p : smo_pts) sum2d += glm::dvec2(p);
                    const glm::vec2 cen2D = (smo_pts.empty())
                        ? glm::vec2(float(W_img) * 0.5f, float(H_img) * 0.5f)
                        : glm::vec2(sum2d / double(smo_pts.size()));

                    // PURE_RIGHT 3D centroid → 2D (same as CB0, for ref)
                    glm::vec2 rightCen2D(-1e6f);
                    bool      rightCen2DValid = false;
                    if (lrValid) {
                        glm::dvec3 sumR(0.0);
                        int nR3 = 0;
                        for (int i = 0; i < nLR; i++) {
                            if (i * 3 + 2 >= nV3) break;
                            if (g_liverLR.labels[i] !=
                                LiverLeftRightLabel::PURE_RIGHT) continue;
                            sumR += glm::dvec3(Vmesh[i*3],
                                               Vmesh[i*3+1],
                                               Vmesh[i*3+2]);
                            nR3++;
                        }
                        if (nR3 > 0) {
                            const glm::vec3 rc3(sumR / double(nR3));
                            const glm::vec4 clip = M * glm::vec4(rc3, 1.0f);
                            if (clip.w > 1e-9f) {
                                const float ndcx = clip.x / clip.w;
                                const float ndcy = clip.y / clip.w;
                                rightCen2D = glm::vec2(
                                    (ndcx + 1.0f) * 0.5f * float(W_img),
                                    (1.0f - ndcy) * 0.5f * float(H_img));
                                rightCen2DValid = true;
                            }
                        }
                    }

                    int n_R_out = 0, n_L_out = 0, n_B_out = 0, n_U_out = 0;
                    int cnt_min = std::numeric_limits<int>::max();
                    int cnt_max = 0, cnt_sum = 0;
                    for (int i = 0; i < N_cells; i++) {
                        switch (smo_lr[i]) {
                            case LiverLeftRightLabel::PURE_RIGHT: n_R_out++; break;
                            case LiverLeftRightLabel::PURE_LEFT:  n_L_out++; break;
                            case LiverLeftRightLabel::BOUNDARY:   n_B_out++; break;
                            default:                              n_U_out++; break;
                        }
                        cnt_min = std::min(cnt_min, smo_cnt[i]);
                        cnt_max = std::max(cnt_max, smo_cnt[i]);
                        cnt_sum += smo_cnt[i];
                    }
                    if (N_cells == 0) cnt_min = 0;

                    // ---- 5. Header info ----
                    ImGui::Text("Smoothed RIM 2D points (grid + KNN)");
                    ImGui::Text("  img=%dx%d   raw=%d   "
                                "cells=%d  (grid=%.1f px)   K=%d  iters=%d",
                                W_img, H_img, N_raw, N_cells,
                                g_rawRimSmooth_GridPx, K, iters);
                    if (N_cells > 0) {
                        ImGui::Text("  cell occupancy:  min=%d  max=%d  "
                                    "avg=%.1f  total=%d",
                                    cnt_min, cnt_max,
                                    float(cnt_sum) / float(N_cells),
                                    cnt_sum);
                        ImGui::Text("  LR (majority):  PURE_RIGHT=%d  "
                                    "PURE_LEFT=%d  BOUNDARY=%d  unknown=%d",
                                    n_R_out, n_L_out, n_B_out, n_U_out);
                    }
                    ImGui::Text("  smoothed 2D centroid: (%.0f, %.0f)",
                                cen2D.x, cen2D.y);
                    if (rightCen2DValid) {
                        ImGui::Text("  PURE_RIGHT 3D centroid → 2D: (%.0f, %.0f)",
                                    rightCen2D.x, rightCen2D.y);
                    }
                    ImGui::Separator();

                    // ---- 6. Canvas ----
                    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    if (avail.x < 100.0f) avail.x = 100.0f;
                    if (avail.y < 100.0f) avail.y = 100.0f;
                    const float canvas_w = avail.x;
                    const float canvas_h = avail.y;
                    ImVec2 canvas_p1(canvas_p0.x + canvas_w,
                                     canvas_p0.y + canvas_h);

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(canvas_p0, canvas_p1,
                                      IM_COL32(20, 20, 26, 255));
                    dl->AddRect      (canvas_p0, canvas_p1,
                                      IM_COL32(120, 120, 130, 255));

                    const float sx = canvas_w / float(W_img);
                    const float sy = canvas_h / float(H_img);
                    const float s  = (sx < sy) ? sx : sy;
                    const float ox = canvas_p0.x + 0.5f * (canvas_w - s * float(W_img));
                    const float oy = canvas_p0.y + 0.5f * (canvas_h - s * float(H_img));
                    auto px2cv = [&](const glm::vec2& p) -> ImVec2 {
                        return ImVec2(ox + p.x * s, oy + p.y * s);
                    };

                    dl->AddRect(
                        px2cv(glm::vec2(0.0f,        0.0f)),
                        px2cv(glm::vec2(float(W_img), float(H_img))),
                        IM_COL32(80, 80, 90, 255));

                    // ---- 7. Raw overlay (optional, faint gray, small) ----
                    if (g_rawRimSmooth_ShowRawOverlay) {
                        const ImU32 rawCol = IM_COL32(140, 140, 145, 110);
                        for (const auto& p : raw_pts) {
                            dl->AddCircleFilled(px2cv(p), 1.8f, rawCol);
                        }
                    }

                    // ---- 8. Smoothed points (colored by majority LR) ----
                    //   Size scales gently with cell occupancy so dense
                    //   regions are visually distinguishable.
                    for (int i = 0; i < N_cells; i++) {
                        ImU32 col = IM_COL32(200, 200, 200, 240);
                        switch (smo_lr[i]) {
                            case LiverLeftRightLabel::PURE_RIGHT:
                                col = IM_COL32(230,  70,  70, 250); break;
                            case LiverLeftRightLabel::PURE_LEFT:
                                col = IM_COL32( 80, 110, 240, 250); break;
                            case LiverLeftRightLabel::BOUNDARY:
                                col = IM_COL32(170, 170, 175, 230); break;
                            default: break;
                        }
                        // radius 3.5..5.5 px based on occupancy
                        float r = 3.5f;
                        if (cnt_max > 1) {
                            r += 2.0f * float(smo_cnt[i] - 1)
                                      / float(cnt_max - 1);
                        }
                        const ImVec2 c = px2cv(smo_pts[i]);
                        dl->AddCircleFilled(c, r, col);
                        dl->AddCircle(c, r + 0.8f,
                                      IM_COL32(0, 0, 0, 200), 0, 1.0f);
                    }

                    // ---- 9. 2D centroid (yellow cross) ----
                    {
                        const ImVec2 c = px2cv(cen2D);
                        const float d = 9.0f;
                        const ImU32 yc = IM_COL32(255, 240, 80, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y),
                                    ImVec2(c.x + d, c.y), yc, 2.0f);
                        dl->AddLine(ImVec2(c.x, c.y - d),
                                    ImVec2(c.x, c.y + d), yc, 2.0f);
                        dl->AddText(ImVec2(c.x + 10, c.y - 18),
                                    yc, "smoothed centroid");
                    }

                    // ---- 10. PURE_RIGHT 3D centroid → 2D (cyan ×) ----
                    if (rightCen2DValid) {
                        const ImVec2 c = px2cv(rightCen2D);
                        const float d = 9.0f;
                        const ImU32 cy = IM_COL32(100, 230, 230, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y - d),
                                    ImVec2(c.x + d, c.y + d), cy, 2.0f);
                        dl->AddLine(ImVec2(c.x - d, c.y + d),
                                    ImVec2(c.x + d, c.y - d), cy, 2.0f);
                        dl->AddCircle(c, 7.0f, cy, 0, 1.5f);
                        dl->AddText(ImVec2(c.x + 10, c.y + 8),
                                    cy, "PURE_RIGHT 3D centroid → 2D");
                    }

                    // ---- 11. Legend ----
                    {
                        const float lx = canvas_p0.x + 8.0f;
                        float       ly = canvas_p0.y + 8.0f;
                        auto legLine = [&](ImU32 col, const char* txt) {
                            dl->AddCircleFilled(ImVec2(lx + 6.0f, ly + 7.0f),
                                                3.5f, col);
                            dl->AddText(ImVec2(lx + 16.0f, ly), col, txt);
                            ly += 16.0f;
                        };
                        legLine(IM_COL32(230,  70,  70, 250), "PURE_RIGHT (smoothed)");
                        legLine(IM_COL32( 80, 110, 240, 250), "PURE_LEFT  (smoothed)");
                        legLine(IM_COL32(170, 170, 175, 230), "BOUNDARY   (smoothed)");
                        if (g_rawRimSmooth_ShowRawOverlay) {
                            legLine(IM_COL32(140, 140, 145, 180), "raw (overlay)");
                        }
                    }

                    ImGui::Dummy(ImVec2(canvas_w, canvas_h));
                }
                ImGui::End();
            }
        }

        // --- CB0.2: Ordered RIM (MST + longest path) popup ---
        //   STAGE 0.2: same cleaned point set as CB0.1, but with an
        //   explicit ORDERING via MST + longest path, then arc-length
        //   resampled pivots. Output is an open polyline — the correct
        //   topology for the caudal RIM arch we observed in CB0.
        if (g_debugShow2DProjPopup_RawRimOrdered &&
            gApp.mode == AppMode::kRegistration)
        {
            ImGui::SetNextWindowSize(ImVec2(820, 580), ImGuiCond_FirstUseEver);
            if (ImGui::Begin(
                    "Ordered RIM (Step 3d Stage 0.2)##silsw_rawrim_ordered_popup",
                    &g_debugShow2DProjPopup_RawRimOrdered))
            {
                // Reuse the same smoothing pipeline as CB0.1 so the
                // two popups show the exact same cleaned input.
                SmoothedRim2DResult S = buildSmoothedRim2D(
                    g_rawRimSmooth_GridPx,
                    g_rawRimSmooth_KnnK,
                    g_rawRimSmooth_KnnIters);

                if (!S.ok) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                       "Cannot build cleaned points: %s",
                                       S.fail_reason.c_str());
                } else {
                    OrderedRim2DResult O = buildOrderedRim2D(
                        S.smo_pts,
                        S.right_centroid_2D,
                        S.right_centroid_valid,
                        g_rawRimOrder_MaxEdgePx,
                        g_rawRimOrder_NPivots);

                    // ---- Header info ----
                    ImGui::Text("Ordered RIM (MST + longest path)");
                    ImGui::Text("  img=%dx%d   cleaned=%d  "
                                "(grid=%.1fpx, K=%d, iters=%d)   "
                                "max_edge=%.0fpx",
                                S.W_img, S.H_img, (int)S.smo_pts.size(),
                                g_rawRimSmooth_GridPx,
                                g_rawRimSmooth_KnnK,
                                g_rawRimSmooth_KnnIters,
                                g_rawRimOrder_MaxEdgePx);
                    if (!O.ok) {
                        ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                           "  ordering failed: %s",
                                           O.fail_reason.c_str());
                    } else {
                        ImGui::Text("  MST: total=%.0fpx   rejected_edges=%d"
                                    "   largest_CC=%d/%d nodes",
                                    O.mst_total_length_px,
                                    O.n_rejected_edges,
                                    (int)O.component.size(),
                                    (int)S.smo_pts.size());
                        const float path_step = (O.path.size() > 1)
                            ? O.arc_length_px / float(O.path.size() - 1)
                            : 0.0f;
                        const float pivot_step = (O.pivots.size() > 1)
                            ? O.arc_length_px / float(O.pivots.size() - 1)
                            : 0.0f;
                        ImGui::Text("  longest path: nodes=%d  arc=%.0fpx"
                                    "  step≈%.1fpx",
                                    (int)O.path.size(),
                                    O.arc_length_px, path_step);
                        ImGui::Text("  pivots: N=%d  step=%.1fpx  start_to_R=%s",
                                    (int)O.pivots.size(),
                                    pivot_step,
                                    O.start_oriented_to_right ? "YES" : "no");
                    }
                    ImGui::Separator();

                    // ---- Canvas ----
                    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    if (avail.x < 100.0f) avail.x = 100.0f;
                    if (avail.y < 100.0f) avail.y = 100.0f;
                    const float canvas_w = avail.x;
                    const float canvas_h = avail.y;
                    ImVec2 canvas_p1(canvas_p0.x + canvas_w,
                                     canvas_p0.y + canvas_h);

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(canvas_p0, canvas_p1,
                                      IM_COL32(20, 20, 26, 255));
                    dl->AddRect      (canvas_p0, canvas_p1,
                                      IM_COL32(120, 120, 130, 255));

                    const float sx = canvas_w / float(S.W_img);
                    const float sy = canvas_h / float(S.H_img);
                    const float s  = (sx < sy) ? sx : sy;
                    const float ox = canvas_p0.x + 0.5f * (canvas_w - s * float(S.W_img));
                    const float oy = canvas_p0.y + 0.5f * (canvas_h - s * float(S.H_img));
                    auto px2cv = [&](const glm::vec2& p) -> ImVec2 {
                        return ImVec2(ox + p.x * s, oy + p.y * s);
                    };

                    dl->AddRect(
                        px2cv(glm::vec2(0.0f,        0.0f)),
                        px2cv(glm::vec2(float(S.W_img), float(S.H_img))),
                        IM_COL32(80, 80, 90, 255));

                    // ---- 1. Cleaned points overlay (LR-tinted, faint) ----
                    if (g_rawRimOrder_ShowCleaned) {
                        for (size_t i = 0; i < S.smo_pts.size(); i++) {
                            ImU32 col = IM_COL32(120, 120, 130, 150);
                            switch (S.smo_lr[i]) {
                                case LiverLeftRightLabel::PURE_RIGHT:
                                    col = IM_COL32(180,  80,  80, 160); break;
                                case LiverLeftRightLabel::PURE_LEFT:
                                    col = IM_COL32( 80,  90, 180, 160); break;
                                case LiverLeftRightLabel::BOUNDARY:
                                    col = IM_COL32(150, 150, 155, 140); break;
                                default: break;
                            }
                            dl->AddCircleFilled(px2cv(S.smo_pts[i]), 2.8f, col);
                        }
                    }

                    if (O.ok) {
                        // ---- 2. MST edges (optional, thin gray) ----
                        if (g_rawRimOrder_ShowMST) {
                            const ImU32 mst_col = IM_COL32(110, 110, 130, 200);
                            for (int u = 0; u < (int)O.mst_adj.size(); u++) {
                                for (int v : O.mst_adj[u]) {
                                    if (v <= u) continue;       // each edge once
                                    dl->AddLine(px2cv(S.smo_pts[u]),
                                                px2cv(S.smo_pts[v]),
                                                mst_col, 1.0f);
                                }
                            }
                        }

                        // ---- 3. Longest path (thick orange polyline) ----
                        const ImU32 path_col = IM_COL32(255, 180, 70, 230);
                        for (size_t i = 1; i < O.path.size(); i++) {
                            dl->AddLine(px2cv(S.smo_pts[O.path[i-1]]),
                                        px2cv(S.smo_pts[O.path[i]]),
                                        path_col, 2.5f);
                        }

                        // ---- 4. Pivots (rainbow, indexed) ----
                        const int Np = (int)O.pivots.size();
                        for (int p = 0; p < Np; p++) {
                            const float hue = (Np > 1)
                                ? (float(p) / float(Np)) * 0.83f : 0.0f;
                            const glm::vec3 c3 = cyclicHsv2rgb(hue, 0.95f, 1.0f);
                            const ImU32 col = IM_COL32(int(c3.r * 255),
                                                       int(c3.g * 255),
                                                       int(c3.b * 255), 255);
                            const ImVec2 c = px2cv(O.pivots[p]);
                            dl->AddCircleFilled(c, 5.0f, col);
                            dl->AddCircle(c, 6.0f,
                                          IM_COL32(255, 255, 255, 255), 0, 1.0f);
                            char idbuf[8];
                            std::snprintf(idbuf, sizeof(idbuf), "%d", p);
                            dl->AddText(ImVec2(c.x + 7, c.y - 14), col, idbuf);
                        }

                        // ---- 5. Start marker (large yellow circle on pivot 0) ----
                        if (!O.pivots.empty()) {
                            const ImVec2 c = px2cv(O.pivots[0]);
                            dl->AddCircle(c, 11.0f,
                                          IM_COL32(255, 240, 80, 255), 0, 3.0f);
                            dl->AddText(ImVec2(c.x + 12, c.y + 6),
                                        IM_COL32(255, 240, 80, 255), "start");
                        }

                        // ---- 6. End marker (yellow X on last pivot) ----
                        if (O.pivots.size() > 1) {
                            const ImVec2 c = px2cv(O.pivots.back());
                            const float d = 7.0f;
                            const ImU32 yc = IM_COL32(255, 200, 80, 230);
                            dl->AddLine(ImVec2(c.x - d, c.y - d),
                                        ImVec2(c.x + d, c.y + d), yc, 2.0f);
                            dl->AddLine(ImVec2(c.x - d, c.y + d),
                                        ImVec2(c.x + d, c.y - d), yc, 2.0f);
                            dl->AddText(ImVec2(c.x + 9, c.y - 16), yc, "end");
                        }
                    }

                    // ---- 7. PURE_RIGHT 3D centroid → 2D (cyan ×) ----
                    if (S.right_centroid_valid) {
                        const ImVec2 c = px2cv(S.right_centroid_2D);
                        const float d = 9.0f;
                        const ImU32 cy = IM_COL32(100, 230, 230, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y - d),
                                    ImVec2(c.x + d, c.y + d), cy, 2.0f);
                        dl->AddLine(ImVec2(c.x - d, c.y + d),
                                    ImVec2(c.x + d, c.y - d), cy, 2.0f);
                        dl->AddCircle(c, 7.0f, cy, 0, 1.5f);
                        dl->AddText(ImVec2(c.x + 10, c.y + 8),
                                    cy, "PURE_RIGHT 3D centroid → 2D");
                    }

                    // ---- 8. Legend ----
                    {
                        const float lx = canvas_p0.x + 8.0f;
                        float       ly = canvas_p0.y + 8.0f;
                        if (g_rawRimOrder_ShowCleaned) {
                            dl->AddCircleFilled(ImVec2(lx + 6.0f, ly + 7.0f),
                                                3.0f,
                                                IM_COL32(180, 80, 80, 220));
                            dl->AddText(ImVec2(lx + 16.0f, ly),
                                        IM_COL32(200, 200, 200, 255),
                                        "cleaned (CB0.1, faint)");
                            ly += 16.0f;
                        }
                        if (g_rawRimOrder_ShowMST) {
                            dl->AddLine(ImVec2(lx, ly + 7),
                                        ImVec2(lx + 14, ly + 7),
                                        IM_COL32(110, 110, 130, 220), 1.5f);
                            dl->AddText(ImVec2(lx + 16, ly),
                                        IM_COL32(160, 160, 170, 255),
                                        "MST edges");
                            ly += 16.0f;
                        }
                        dl->AddLine(ImVec2(lx, ly + 7),
                                    ImVec2(lx + 14, ly + 7),
                                    IM_COL32(255, 180, 70, 230), 2.5f);
                        dl->AddText(ImVec2(lx + 16, ly),
                                    IM_COL32(255, 200, 100, 255),
                                    "longest path");
                        ly += 16.0f;
                        dl->AddCircleFilled(ImVec2(lx + 6.0f, ly + 7.0f),
                                            3.5f,
                                            IM_COL32(230, 70, 70, 255));
                        dl->AddText(ImVec2(lx + 16.0f, ly),
                                    IM_COL32(220, 220, 220, 255),
                                    "pivots 0..N (rainbow)");
                    }

                    ImGui::Dummy(ImVec2(canvas_w, canvas_h));
                }
                ImGui::End();
            }
        }

        // --- CB0.3: Manual Sweep Probe popup ---
        //   STAGE 0.3: interactive single-candidate viewer that uses the
        //   SAME source/target globals as the actual sweep, so geometry
        //   matches byte-for-byte. Reproduces ONE evaluateSilhouette-
        //   SweepCandidate evaluation per frame.
        if (g_debugShow2DProjPopup_OverlayProbe &&
            gApp.mode == AppMode::kRegistration)
        {
            ImGui::SetNextWindowSize(ImVec2(900, 640), ImGuiCond_FirstUseEver);
            if (ImGui::Begin(
                    "Overlay probe — manual sweep candidate"
                    " (Step 3d Stage 0.3)##silsw_overlay_probe_popup",
                    &g_debugShow2DProjPopup_OverlayProbe))
            {
                // ---- 1. Ensure source preview is up to date ----
                //   Uses whichever method is active (ENVELOPE / MST) so
                //   CB0.3 reflects what the actual sweep would see.
                std::string srcFail;
                const bool src_ok = silSwBuildSrcPreview(&srcFail);

                // ---- 2. Ensure target preview is up to date ----
                std::string tgtFail;
                bool tgt_ok = true;
                if (g_silSwTgtAnchors2DPreview.empty() ||
                    g_silSwTgtLower2DPreview.empty())
                {
                    tgt_ok = silSwBuildTgtPreview(&tgtFail);
                }

                if (!src_ok) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                        "Cannot build source preview: %s\n"
                        "(press W to populate g_debugSourceRimChain first)",
                        srcFail.c_str());
                }
                if (!tgt_ok) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                        "Cannot build target preview: %s\n"
                        "(press Shift+W to populate target boundary)",
                        tgtFail.c_str());
                }

                if (src_ok && tgt_ok) {
                    // ---- 3. Resolve i / j / k against actual sizes ----
                    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
                    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
                    const glm::mat4 view_m = buildSilhouetteView();
                    const glm::mat4 proj_m = buildSilhouetteProj();
                    const glm::mat4 VP     = proj_m * view_m;

                    const int N_pivot  = (int)g_silSwSrcPivots3DPreview.size();
                    const int N_anchor = (int)g_silSwTgtAnchors2DPreview.size();
                    const int N_rot    = std::max(2, g_overlayProbe_NRotation);

                    if (N_pivot == 0 || N_anchor == 0) {
                        ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1),
                            "Pivots/anchors not populated yet "
                            "(N_pivot=%d, N_anchor=%d)",
                            N_pivot, N_anchor);
                    } else {
                        // ---- 4. Auto-animate k ----
                        if (g_overlayProbe_AutoAnimate) {
                            g_overlayProbe_AnimFrameCounter++;
                            if (g_overlayProbe_AnimFrameCounter >=
                                std::max(1, g_overlayProbe_AnimFramesPerStep))
                            {
                                g_overlayProbe_AnimFrameCounter = 0;
                                g_overlayProbe_RotStep =
                                    (g_overlayProbe_RotStep + 1) % N_rot;
                            }
                        }

                        // ---- 5. Clamp and resolve indices ----
                        const int i = std::max(0,
                            std::min(g_overlayProbe_PivotI, N_pivot - 1));
                        int j = g_overlayProbe_LockJI ? i
                            : std::max(0,
                                std::min(g_overlayProbe_AnchorJ, N_anchor - 1));
                        if (g_overlayProbe_LockJI) {
                            j = std::min(j, N_anchor - 1);
                        }
                        const int k = std::max(0,
                            std::min(g_overlayProbe_RotStep, N_rot - 1));
                        const float theta_deg = float(k) * 360.0f / float(N_rot);
                        const float c_th = std::cos(glm::radians(theta_deg));
                        const float s_th = std::sin(glm::radians(theta_deg));

                        // ---- 6. Build T using evaluateSilhouetteSweepCandidate
                        //   formula but with INDEPENDENT i / j (sweep uses
                        //   i == j; CB0.3 allows i ≠ j for exploration).
                        const glm::vec3 P_s    = g_silSwSrcPivots3DPreview[i];
                        const glm::vec2 P_t_2D = g_silSwTgtAnchors2DPreview[j];
                        const glm::vec3 P_t_3D = RimShape::unprojectPixelAtWorldDepth(
                            P_t_2D, P_s, view_m, proj_m, W_img, H_img);

                        glm::mat4 R(1.0f);
                        R[0][0] =  c_th; R[0][1] =  s_th; R[0][2] = 0.0f;
                        R[1][0] = -s_th; R[1][1] =  c_th; R[1][2] = 0.0f;
                        R[2][0] = 0.0f;  R[2][1] = 0.0f;  R[2][2] = 1.0f;
                        const glm::mat4 T_to   = glm::translate(glm::mat4(1.0f),  P_t_3D);
                        const glm::mat4 T_from = glm::translate(glm::mat4(1.0f), -P_s);
                        const glm::mat4 T      = T_to * R * T_from;

                        // ---- 7. Transform source dense + pivots to 2D ----
                        auto project = [&](const glm::vec3& p3) -> glm::vec2 {
                            const glm::vec4 cp = VP * (T * glm::vec4(p3, 1.0f));
                            if (cp.w <= 1e-9f) return glm::vec2(-1e6f, -1e6f);
                            return glm::vec2(
                                (cp.x/cp.w + 1.0f) * 0.5f * float(W_img),
                                (1.0f - cp.y/cp.w) * 0.5f * float(H_img));
                        };

                        std::vector<glm::vec2> src_dense_2D;
                        src_dense_2D.reserve(g_silSwSrcRim3DPreview.size());
                        for (const auto& p : g_silSwSrcRim3DPreview)
                            src_dense_2D.push_back(project(p));

                        std::vector<glm::vec2> src_pivots_2D(N_pivot);
                        for (int p = 0; p < N_pivot; p++)
                            src_pivots_2D[p] = project(g_silSwSrcPivots3DPreview[p]);

                        // ---- 8. Cost (chamfer to target lower) ----
                        const double cost = RimShape::denseChamfer2D(
                            src_dense_2D, g_silSwTgtLower2DPreview);

                        // ---- 9. CC orientation diagnosis ----
                        //   Transform g_liverCC.d_cc by R only (translation
                        //   doesn't affect direction). Project the CRANIAL
                        //   tip and CAUDAL tip from the pivot point.
                        glm::vec2 cc_arrow_head(-1e6f);   // CAUDAL screen pt
                        glm::vec2 cc_arrow_tail(-1e6f);   // CRANIAL screen pt
                        bool      cc_valid    = false;
                        bool      cc_head_up  = false;
                        float     cc_angle_deg = 0.0f;    // on-screen CRANIAL→CAUDAL
                        float     cc_delta_from_6oclock = 0.0f;
                        if (g_liverCC.valid()) {
                            // d_cc points toward CRANIAL by convention.
                            const glm::vec3 d_cc_world(
                                g_liverCC.d_cc.x,
                                g_liverCC.d_cc.y,
                                g_liverCC.d_cc.z);
                            // Use a small step so both endpoints project
                            // close to the pivot for a visible arrow.
                            const float step = 60.0f;   // mesh units (rough liver scale)
                            const glm::vec3 cranial_world = P_s + step * d_cc_world;
                            const glm::vec3 caudal_world  = P_s - step * d_cc_world;
                            const glm::vec2 cranial_2D = project(cranial_world);
                            const glm::vec2 caudal_2D  = project(caudal_world);
                            if (cranial_2D.x > -1e5f && caudal_2D.x > -1e5f) {
                                cc_arrow_tail = cranial_2D;
                                cc_arrow_head = caudal_2D;
                                // "head-up" means: CRANIAL projects ABOVE
                                // CAUDAL on screen (smaller pixel-y is up).
                                cc_head_up = (cranial_2D.y < caudal_2D.y);
                                cc_valid = true;
                                // On-screen CC angle (CRANIAL→CAUDAL).
                                // +y is screen-down → 90° = 6 o'clock.
                                const float dx = caudal_2D.x - cranial_2D.x;
                                const float dy = caudal_2D.y - cranial_2D.y;
                                cc_angle_deg = std::atan2(dy, dx) * 180.0f / 3.14159265f;
                                cc_delta_from_6oclock = cc_angle_deg - 90.0f;
                                while (cc_delta_from_6oclock >  180.0f) cc_delta_from_6oclock -= 360.0f;
                                while (cc_delta_from_6oclock < -180.0f) cc_delta_from_6oclock += 360.0f;
                            }
                        }

                        // ---- 9b. Check A (rotation cap) — same logic as sweep本番 ----
                        float th_wrapped = theta_deg;
                        while (th_wrapped >  180.0f) th_wrapped -= 360.0f;
                        while (th_wrapped < -180.0f) th_wrapped += 360.0f;
                        const bool checkA_pass =
                            std::fabs(th_wrapped) <= g_silSwCheckA_RotCapDeg;
                        const bool checkA_reject =
                            g_overlayProbe_SimulateCheckA && !checkA_pass;

                        // ---- 9c. Check B (CC guard) — same logic as sweep本番 ----
                        bool checkB_pass    = true;
                        bool checkB_skipped = true;   // skipped when g_liverCC invalid
                        if (cc_valid) {
                            checkB_skipped = false;
                            checkB_pass =
                                std::fabs(cc_delta_from_6oclock) <=
                                g_silSwCheckB_CCToleranceDeg;
                        }
                        const bool checkB_reject =
                            g_overlayProbe_SimulateCheckB &&
                            !checkB_skipped && !checkB_pass;

                        // ---- 10. LR alignment check ----
                        //   Source pivot[0] is by construction the
                        //   PURE_RIGHT-oriented end. After T, where does
                        //   it land relative to PURE_RIGHT centroid?
                        bool lr_ok = true;
                        if (g_silSwSrcRightCentroid2DPreview.x > -1e5f &&
                            N_pivot > 0)
                        {
                            const glm::vec2 p0_after = src_pivots_2D[0];
                            const glm::vec2 pN_after = src_pivots_2D[N_pivot - 1];
                            const glm::vec2 R_ref    = g_silSwSrcRightCentroid2DPreview;
                            // After T, pivot[0] should still be closer
                            // to the source's PURE_RIGHT centroid than
                            // pivot[N-1] is. (LR is internal to source.)
                            const float d0 = glm::length(p0_after - R_ref);
                            const float dN = glm::length(pN_after - R_ref);
                            lr_ok = (d0 <= dN);
                        }

                        // ==== Header ====
                        ImGui::Text("Manual sweep candidate viewer");
                        ImGui::Text("  i=%d  j=%d  k=%d (θ=%.1f°)   "
                                    "lock=%s   cost=%.1f px",
                                    i, j, k, theta_deg,
                                    g_overlayProbe_LockJI ? "ON" : "OFF",
                                    cost);
                        ImGui::Text("  N_pivot=%d   N_anchor=%d   N_rot=%d",
                                    N_pivot, N_anchor, N_rot);
                        if (cc_valid) {
                            if (cc_head_up) {
                                ImGui::TextColored(ImVec4(0.4f, 0.95f, 0.4f, 1),
                                    "  CC: head-up ✓  (angle from 6h = %.1f°)",
                                    cc_delta_from_6oclock);
                            } else {
                                ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1),
                                    "  CC: FLIPPED ✗  (angle from 6h = %.1f°)",
                                    cc_delta_from_6oclock);
                            }
                        } else {
                            ImGui::TextColored(ImVec4(0.95f, 0.7f, 0.3f, 1),
                                "  CC: unavailable (g_liverCC not valid — run Shift+H)");
                        }
                        ImGui::Text("  LR (pivot 0 vs N-1 to PURE_RIGHT centroid): %s",
                                    lr_ok ? "aligned ✓" : "mirrored ✗");

                        // ---- Check A / Check B status (sweep本番 guard simulation) ----
                        if (checkA_pass) {
                            ImGui::TextColored(ImVec4(0.4f, 0.95f, 0.4f, 1),
                                "  Check A (rot cap ±%.0f°): pass ✓  (|θ wrap|=%.1f°)",
                                g_silSwCheckA_RotCapDeg, std::fabs(th_wrapped));
                        } else {
                            const ImVec4 col = g_overlayProbe_SimulateCheckA
                                ? ImVec4(0.95f, 0.4f, 0.4f, 1)
                                : ImVec4(0.95f, 0.7f, 0.3f, 1);
                            ImGui::TextColored(col,
                                "  Check A (rot cap ±%.0f°): %s  (|θ wrap|=%.1f°)",
                                g_silSwCheckA_RotCapDeg,
                                g_overlayProbe_SimulateCheckA ? "REJECT ✗" : "would reject (sim OFF)",
                                std::fabs(th_wrapped));
                        }
                        if (checkB_skipped) {
                            ImGui::TextColored(ImVec4(0.95f, 0.7f, 0.3f, 1),
                                "  Check B (CC ±%.0f°): skipped (no CC data)",
                                g_silSwCheckB_CCToleranceDeg);
                        } else if (checkB_pass) {
                            ImGui::TextColored(ImVec4(0.4f, 0.95f, 0.4f, 1),
                                "  Check B (CC ±%.0f°): pass ✓  (|Δ from 6h|=%.1f°)",
                                g_silSwCheckB_CCToleranceDeg,
                                std::fabs(cc_delta_from_6oclock));
                        } else {
                            const ImVec4 col = g_overlayProbe_SimulateCheckB
                                ? ImVec4(0.95f, 0.4f, 0.4f, 1)
                                : ImVec4(0.95f, 0.7f, 0.3f, 1);
                            ImGui::TextColored(col,
                                "  Check B (CC ±%.0f°): %s  (|Δ from 6h|=%.1f°)",
                                g_silSwCheckB_CCToleranceDeg,
                                g_overlayProbe_SimulateCheckB ? "REJECT ✗" : "would reject (sim OFF)",
                                std::fabs(cc_delta_from_6oclock));
                        }

                        const bool any_reject = checkA_reject || checkB_reject;
                        if (any_reject) {
                            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1),
                                "  >>> REJECTED — sweep本番 would set cost = +∞ <<<");
                        }
                        ImGui::Separator();

                        // ==== Canvas ====
                        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                        ImVec2 avail = ImGui::GetContentRegionAvail();
                        if (avail.x < 100.0f) avail.x = 100.0f;
                        if (avail.y < 100.0f) avail.y = 100.0f;
                        const float canvas_w = avail.x;
                        const float canvas_h = avail.y;
                        ImVec2 canvas_p1(canvas_p0.x + canvas_w,
                                         canvas_p0.y + canvas_h);

                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        // Reddish tint when any simulated check rejects;
                        // normal dark background otherwise.
                        const ImU32 bg_col = any_reject
                            ? IM_COL32(60, 18, 18, 255)
                            : IM_COL32(20, 20, 26, 255);
                        dl->AddRectFilled(canvas_p0, canvas_p1, bg_col);
                        dl->AddRect      (canvas_p0, canvas_p1,
                                          any_reject
                                            ? IM_COL32(200, 80, 80, 255)
                                            : IM_COL32(120, 120, 130, 255));

                        const float sx = canvas_w / float(W_img);
                        const float sy = canvas_h / float(H_img);
                        const float s  = (sx < sy) ? sx : sy;
                        const float ox = canvas_p0.x + 0.5f * (canvas_w - s * float(W_img));
                        const float oy = canvas_p0.y + 0.5f * (canvas_h - s * float(H_img));
                        auto px2cv = [&](const glm::vec2& p) -> ImVec2 {
                            return ImVec2(ox + p.x * s, oy + p.y * s);
                        };

                        dl->AddRect(
                            px2cv(glm::vec2(0.0f,        0.0f)),
                            px2cv(glm::vec2(float(W_img), float(H_img))),
                            IM_COL32(80, 80, 90, 255));

                        // ---- 11a. Target overlay (purple dense + rainbow anchors) ----
                        if (g_overlayProbe_ShowTgtOverlay) {
                            // Dense target lower-half (purple dots)
                            const ImU32 tcol = IM_COL32(180, 120, 200, 180);
                            for (const auto& p : g_silSwTgtLower2DPreview) {
                                dl->AddCircleFilled(px2cv(p), 1.4f, tcol);
                            }
                            // 20 target anchors (rainbow ×)
                            for (int a = 0; a < N_anchor; a++) {
                                const float hue = (N_anchor > 1)
                                    ? (float(a) / float(N_anchor)) * 0.83f : 0.0f;
                                const glm::vec3 c3 = cyclicHsv2rgb(hue, 0.95f, 1.0f);
                                const ImU32 col = IM_COL32(int(c3.r * 255),
                                                           int(c3.g * 255),
                                                           int(c3.b * 255), 255);
                                const ImVec2 cp = px2cv(g_silSwTgtAnchors2DPreview[a]);
                                const float d = 5.0f;
                                dl->AddLine(ImVec2(cp.x-d, cp.y-d),
                                            ImVec2(cp.x+d, cp.y+d), col, 1.5f);
                                dl->AddLine(ImVec2(cp.x-d, cp.y+d),
                                            ImVec2(cp.x+d, cp.y-d), col, 1.5f);
                                char idbuf[8];
                                std::snprintf(idbuf, sizeof(idbuf), "%d", a);
                                dl->AddText(ImVec2(cp.x + 7, cp.y + 4), col, idbuf);
                            }
                        }

                        // ---- 11b. Source overlay (transformed dense path + pivots) ----
                        if (g_overlayProbe_ShowSrcOverlay) {
                            // Dense path as connected polyline
                            const int M = (int)src_dense_2D.size();
                            for (int p = 1; p < M; p++) {
                                if (src_dense_2D[p-1].x < -1e5f) continue;
                                if (src_dense_2D[p].x   < -1e5f) continue;
                                ImU32 col = IM_COL32(220, 180, 180, 200);
                                if (p - 1 < (int)g_silSwSrcRim2DPreviewLR.size()) {
                                    switch (g_silSwSrcRim2DPreviewLR[p - 1]) {
                                        case LiverLeftRightLabel::PURE_RIGHT:
                                            col = IM_COL32(230, 80, 80, 230); break;
                                        case LiverLeftRightLabel::PURE_LEFT:
                                            col = IM_COL32(80, 110, 240, 230); break;
                                        case LiverLeftRightLabel::BOUNDARY:
                                            col = IM_COL32(180, 180, 185, 200); break;
                                        default: break;
                                    }
                                }
                                dl->AddLine(px2cv(src_dense_2D[p-1]),
                                            px2cv(src_dense_2D[p]), col, 2.0f);
                            }
                            // Source pivots (rainbow ●)
                            for (int p = 0; p < N_pivot; p++) {
                                if (src_pivots_2D[p].x < -1e5f) continue;
                                const float hue = (N_pivot > 1)
                                    ? (float(p) / float(N_pivot)) * 0.83f : 0.0f;
                                const glm::vec3 c3 = cyclicHsv2rgb(hue, 0.95f, 1.0f);
                                const ImU32 col = IM_COL32(int(c3.r * 255),
                                                           int(c3.g * 255),
                                                           int(c3.b * 255), 255);
                                const ImVec2 cp = px2cv(src_pivots_2D[p]);
                                dl->AddCircleFilled(cp, 4.5f, col);
                                dl->AddCircle(cp, 5.5f,
                                              IM_COL32(0, 0, 0, 220), 0, 1.0f);
                                char idbuf[8];
                                std::snprintf(idbuf, sizeof(idbuf), "%d", p);
                                dl->AddText(ImVec2(cp.x + 6, cp.y - 14), col, idbuf);
                            }
                        }

                        // ---- 12. Highlight selected pivot[i] and anchor[j] ----
                        if (i >= 0 && i < (int)src_pivots_2D.size() &&
                            src_pivots_2D[i].x > -1e5f)
                        {
                            const ImVec2 cp = px2cv(src_pivots_2D[i]);
                            dl->AddCircle(cp, 11.0f,
                                          IM_COL32(255, 240, 80, 255), 0, 3.0f);
                            dl->AddText(ImVec2(cp.x + 13, cp.y + 7),
                                        IM_COL32(255, 240, 80, 255), "pivot[i]");
                        }
                        if (j >= 0 && j < (int)g_silSwTgtAnchors2DPreview.size()) {
                            const ImVec2 cp = px2cv(g_silSwTgtAnchors2DPreview[j]);
                            const float d = 10.0f;
                            const ImU32 yc = IM_COL32(255, 220, 100, 255);
                            dl->AddLine(ImVec2(cp.x-d, cp.y-d),
                                        ImVec2(cp.x+d, cp.y+d), yc, 3.0f);
                            dl->AddLine(ImVec2(cp.x-d, cp.y+d),
                                        ImVec2(cp.x+d, cp.y-d), yc, 3.0f);
                            dl->AddText(ImVec2(cp.x + 12, cp.y + 7), yc, "anchor[j]");
                        }

                        // ---- 13. CC direction arrow ----
                        if (g_overlayProbe_ShowCCArrow && cc_valid) {
                            const ImVec2 tail = px2cv(cc_arrow_tail);  // CRANIAL
                            const ImVec2 head = px2cv(cc_arrow_head);  // CAUDAL
                            const ImU32 col = cc_head_up
                                ? IM_COL32(120, 240, 120, 255)
                                : IM_COL32(240, 100, 100, 255);
                            dl->AddLine(tail, head, col, 3.0f);
                            // Arrowhead (perpendicular triangle at head)
                            const float dx = head.x - tail.x;
                            const float dy = head.y - tail.y;
                            const float len = std::sqrt(dx*dx + dy*dy);
                            if (len > 1.0f) {
                                const float ux = dx / len, uy = dy / len;
                                const float px = -uy, py = ux;
                                const float a = 10.0f, b = 6.0f;
                                const ImVec2 p1(head.x - a*ux + b*px,
                                                head.y - a*uy + b*py);
                                const ImVec2 p2(head.x - a*ux - b*px,
                                                head.y - a*uy - b*py);
                                dl->AddTriangleFilled(head, p1, p2, col);
                            }
                            dl->AddText(tail,
                                        IM_COL32(180, 240, 180, 230), "CRANIAL");
                            dl->AddText(head,
                                        IM_COL32(240, 180, 180, 230), "CAUDAL");
                        }

                        // ---- 14. Legend (top-left) ----
                        {
                            const float lx = canvas_p0.x + 8.0f;
                            float       ly = canvas_p0.y + 8.0f;
                            auto legCircle = [&](ImU32 col, const char* txt) {
                                dl->AddCircleFilled(ImVec2(lx + 6.0f, ly + 7.0f),
                                                    3.5f, col);
                                dl->AddText(ImVec2(lx + 16.0f, ly), col, txt);
                                ly += 16.0f;
                            };
                            auto legLine = [&](ImU32 col, const char* txt) {
                                dl->AddLine(ImVec2(lx, ly + 7),
                                            ImVec2(lx + 14, ly + 7), col, 2.0f);
                                dl->AddText(ImVec2(lx + 16, ly), col, txt);
                                ly += 16.0f;
                            };
                            if (g_overlayProbe_ShowTgtOverlay) {
                                legCircle(IM_COL32(180, 120, 200, 220),
                                          "target lower-half (purple)");
                                legCircle(IM_COL32(255, 100, 100, 230),
                                          "target anchors (rainbow ×)");
                            }
                            if (g_overlayProbe_ShowSrcOverlay) {
                                legLine(IM_COL32(230, 80, 80, 230),
                                        "source dense (LR-tinted)");
                                legCircle(IM_COL32(255, 220, 100, 230),
                                          "source pivots (rainbow ●)");
                            }
                            if (g_overlayProbe_ShowCCArrow && cc_valid) {
                                legLine(cc_head_up
                                    ? IM_COL32(120, 240, 120, 255)
                                    : IM_COL32(240, 100, 100, 255),
                                    cc_head_up ? "CC head-up ✓"
                                               : "CC FLIPPED ✗");
                            }
                        }

                        // ---- 15. REJECTED banner (top center, large) ----
                        if (any_reject) {
                            const char* msg =
                                (checkA_reject && checkB_reject) ? "REJECTED BY A + B" :
                                checkA_reject                    ? "REJECTED BY A (rotation cap)" :
                                                                   "REJECTED BY B (CC guard)";
                            const ImU32 col = IM_COL32(255, 80, 80, 255);
                            // Position near top, horizontally centered over canvas.
                            const ImVec2 tsize = ImGui::CalcTextSize(msg);
                            const ImVec2 pos(
                                canvas_p0.x + 0.5f * (canvas_w - tsize.x),
                                canvas_p0.y + 24.0f);
                            // Soft dark backing for readability
                            dl->AddRectFilled(
                                ImVec2(pos.x - 6, pos.y - 3),
                                ImVec2(pos.x + tsize.x + 6, pos.y + tsize.y + 3),
                                IM_COL32(20, 0, 0, 200), 4.0f);
                            dl->AddText(pos, col, msg);
                        }

                        ImGui::Dummy(ImVec2(canvas_w, canvas_h));
                    }
                }
                ImGui::End();
            }
        }

        // --- CB1: Source 2D projection popup ---
        if (g_debugShow2DProjPopup_Source && gApp.mode == AppMode::kRegistration) {
            // Rebuild source cache every frame so the popup reflects
            // the current source pose. silSwBuildSrcPreview is cheap.
            std::string srcFail;
            const bool ok = silSwBuildSrcPreview(&srcFail);

            ImGui::SetNextWindowSize(ImVec2(820, 540), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Source rim 2D projection (Step 3d)##silsw_src_popup",
                             &g_debugShow2DProjPopup_Source))
            {
                if (!ok) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                       "Cannot build preview: %s",
                                       srcFail.c_str());
                } else {
                    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
                    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
                    ImGui::Text("AR-image-plane projection of source rim chain");
                    ImGui::Text("  dense=%d pts   pivots=%d   start_idx=%d   img=%dx%d",
                                (int)g_silSwSrcRim2DPreview.size(),
                                (int)g_silSwSrcPivots2DPreview.size(),
                                g_silSwSrcStartIdxPreview, W_img, H_img);

                    // [Step 3d2] angle-bin envelope diagnostic: step stats
                    {
                        const auto& curveD = g_silSwSrcRim2DPreview;
                        const int Kd = (int)curveD.size();
                        if (Kd >= 2) {
                            float dsum = 0.0f, dmax = 0.0f, dmin = 1e30f;
                            int   n_jump = 0;
                            std::vector<float> ds;
                            ds.reserve(Kd);
                            for (int i = 0; i < Kd; i++) {
                                const int j = (i + 1) % Kd;
                                const float dx = curveD[j].x - curveD[i].x;
                                const float dy = curveD[j].y - curveD[i].y;
                                const float d = std::sqrt(dx*dx + dy*dy);
                                ds.push_back(d);
                                dsum += d;
                                if (d > dmax) dmax = d;
                                if (d < dmin) dmin = d;
                            }
                            std::sort(ds.begin(), ds.end());
                            const float dmed = ds[ds.size() / 2];
                            for (float d : ds) if (d > 4.0f * dmed) n_jump++;
                            const float davg = dsum / float(Kd);
                            ImGui::Text("  step px: avg=%.1f  med=%.1f  min=%.1f  max=%.1f"
                                        "   jumps(>4×med)=%d",
                                        davg, dmed, dmin, dmax, n_jump);
                            if (n_jump > 0) {
                                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                                    "  [!] %d step(s) longer than 4× median — "
                                    "shown in orange below",
                                    n_jump);
                            }
                        }
                    }
                    ImGui::Separator();

                    // Reserve canvas region
                    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    if (avail.x < 100.0f) avail.x = 100.0f;
                    if (avail.y < 100.0f) avail.y = 100.0f;
                    const float canvas_w = avail.x;
                    const float canvas_h = avail.y;
                    ImVec2 canvas_p1(canvas_p0.x + canvas_w, canvas_p0.y + canvas_h);

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(20, 20, 26, 255));
                    dl->AddRect      (canvas_p0, canvas_p1, IM_COL32(120, 120, 130, 255));

                    // Image → canvas scale (preserve aspect, fit)
                    const float sx = canvas_w / float(W_img);
                    const float sy = canvas_h / float(H_img);
                    const float s  = (sx < sy) ? sx : sy;
                    // Centered offset
                    const float ox = canvas_p0.x + 0.5f * (canvas_w - s * float(W_img));
                    const float oy = canvas_p0.y + 0.5f * (canvas_h - s * float(H_img));
                    auto px2cv = [&](const glm::vec2& p) -> ImVec2 {
                        return ImVec2(ox + p.x * s, oy + p.y * s);
                    };

                    // Draw image frame outline
                    dl->AddRect(
                        px2cv(glm::vec2(0.0f,        0.0f)),
                        px2cv(glm::vec2(float(W_img), float(H_img))),
                        IM_COL32(80, 80, 90, 255));

                    // Dense rim drawn as CLOSED polyline (last→first
                    // segment included). The angle-bin envelope is a
                    // closed contour by construction. Color by LR
                    // label. Segments more than 4× median length are
                    // drawn in orange to expose any anomaly (typically
                    // there will be none for the envelope method).
                    const auto& curve = g_silSwSrcRim2DPreview;
                    const auto& labels = g_silSwSrcRim2DPreviewLR;
                    const int Kc = (int)curve.size();
                    float jump_thr = 1e9f;
                    if (Kc >= 2) {
                        std::vector<float> ds;
                        ds.reserve(Kc);
                        for (int i = 0; i < Kc; i++) {
                            const int j = (i + 1) % Kc;
                            const float dx = curve[j].x - curve[i].x;
                            const float dy = curve[j].y - curve[i].y;
                            ds.push_back(std::sqrt(dx*dx + dy*dy));
                        }
                        std::sort(ds.begin(), ds.end());
                        jump_thr = 4.0f * ds[ds.size() / 2];
                    }
                    for (int i = 0; i < Kc; i++) {
                        if (curve[i].x < -1e5f) continue;
                        const int j = (i + 1) % Kc;     // closed loop
                        if (curve[j].x < -1e5f) continue;
                        const float dx = curve[j].x - curve[i].x;
                        const float dy = curve[j].y - curve[i].y;
                        const float dseg = std::sqrt(dx*dx + dy*dy);

                        ImU32 col = IM_COL32(180, 180, 180, 255);
                        if (i < (int)labels.size()) {
                            switch (labels[i]) {
                                case LiverLeftRightLabel::PURE_RIGHT:
                                    col = IM_COL32(230, 70, 70, 255);   break;
                                case LiverLeftRightLabel::PURE_LEFT:
                                    col = IM_COL32(80, 110, 240, 255);  break;
                                case LiverLeftRightLabel::BOUNDARY:
                                    col = IM_COL32(170, 170, 175, 255); break;
                            }
                        }
                        if (dseg > jump_thr) {
                            col = IM_COL32(255, 150, 50, 200);   // orange = jump
                        }
                        dl->AddLine(px2cv(curve[i]), px2cv(curve[j]), col, 1.5f);
                    }

                    // PURE_RIGHT centroid (yellow cross)
                    {
                        const ImVec2 c = px2cv(g_silSwSrcRightCentroid2DPreview);
                        const float d = 8.0f;
                        const ImU32 yc = IM_COL32(255, 240, 80, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y), ImVec2(c.x + d, c.y), yc, 2.0f);
                        dl->AddLine(ImVec2(c.x, c.y - d), ImVec2(c.x, c.y + d), yc, 2.0f);
                        dl->AddText(ImVec2(c.x + 10, c.y - 18), yc, "PURE_RIGHT centroid");
                    }

                    // Right-start chain vertex (large yellow marker)
                    if (!curve.empty() && curve[0].x > -1e5f) {
                        const ImVec2 c = px2cv(curve[0]);
                        dl->AddCircle(c, 10.0f, IM_COL32(255, 240, 80, 255), 0, 3.0f);
                        dl->AddText(ImVec2(c.x + 12, c.y + 6),
                                    IM_COL32(255, 240, 80, 255), "start");
                    }

                    // 20 pivots (rainbow)
                    const auto& pv = g_silSwSrcPivots2DPreview;
                    const int Np = (int)pv.size();
                    for (int p = 0; p < Np; p++) {
                        if (pv[p].x < -1e5f) continue;
                        // Cyclic HSV rainbow over [0, 5/6] (matches the
                        // existing Preview anchors so visual matching by
                        // index works between popup and 3D scene).
                        const float hue = (Np > 1) ? (float(p) / float(Np)) * 0.83f : 0.0f;
                        const glm::vec3 c3 = cyclicHsv2rgb(hue, 0.95f, 1.0f);
                        const ImU32 col = IM_COL32(int(c3.r * 255),
                                                   int(c3.g * 255),
                                                   int(c3.b * 255), 255);
                        const ImVec2 c = px2cv(pv[p]);
                        dl->AddCircleFilled(c, 5.0f, col);
                        dl->AddCircle(c, 6.0f, IM_COL32(255, 255, 255, 255), 0, 1.0f);
                        char idbuf[8];
                        std::snprintf(idbuf, sizeof(idbuf), "%d", p);
                        dl->AddText(ImVec2(c.x + 7, c.y - 14), col, idbuf);
                    }

                    ImGui::Dummy(ImVec2(canvas_w, canvas_h));
                }
                ImGui::End();
            }
        }

        // --- CB2: Target rim lower-half + anchors popup ---
        if (g_debugShow2DProjPopup_Target && gApp.mode == AppMode::kRegistration) {
            // Target is static once Shift+W ran; cache survives. Rebuild
            // is also cheap (~150 pts max).
            std::string tgtFail;
            const bool ok = silSwBuildTgtPreview(&tgtFail);

            ImGui::SetNextWindowSize(ImVec2(820, 540), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Target rim lower-half + anchors (Step 3d)##silsw_tgt_popup",
                             &g_debugShow2DProjPopup_Target))
            {
                if (!ok) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                                       "Cannot build preview: %s",
                                       tgtFail.c_str());
                } else {
                    const int W_img = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1920;
                    const int H_img = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 1080;
                    ImGui::Text("AR-image-plane target contour (largest segment)");
                    ImGui::Text("  total=%d  lower=%d  anchors=%d  img=%dx%d",
                                (int)g_debugTargetContour2D.size(),
                                (int)g_silSwTgtLower2DPreview.size(),
                                (int)g_silSwTgtAnchors2DPreview.size(),
                                W_img, H_img);
                    ImGui::Text("  centroid.y=%.1f  (pixels with y > this = lower)",
                                g_silSwTgtCentroid2DPreview.y);
                    ImGui::Separator();

                    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    if (avail.x < 100.0f) avail.x = 100.0f;
                    if (avail.y < 100.0f) avail.y = 100.0f;
                    const float canvas_w = avail.x;
                    const float canvas_h = avail.y;
                    ImVec2 canvas_p1(canvas_p0.x + canvas_w, canvas_p0.y + canvas_h);

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(20, 20, 26, 255));
                    dl->AddRect      (canvas_p0, canvas_p1, IM_COL32(120, 120, 130, 255));

                    const float sx = canvas_w / float(W_img);
                    const float sy = canvas_h / float(H_img);
                    const float s  = (sx < sy) ? sx : sy;
                    const float ox = canvas_p0.x + 0.5f * (canvas_w - s * float(W_img));
                    const float oy = canvas_p0.y + 0.5f * (canvas_h - s * float(H_img));
                    auto px2cv = [&](const glm::vec2& p) -> ImVec2 {
                        return ImVec2(ox + p.x * s, oy + p.y * s);
                    };

                    // Image frame
                    dl->AddRect(
                        px2cv(glm::vec2(0.0f,        0.0f)),
                        px2cv(glm::vec2(float(W_img), float(H_img))),
                        IM_COL32(80, 80, 90, 255));

                    // Full target contour: above-centroid gray, below purple.
                    // (g_debugTargetContour2D is an open polyline; multiple
                    //  segments are possible but traceContour2D returns the
                    //  longest, so a single polyline draw is fine.)
                    const float cy = g_silSwTgtCentroid2DPreview.y;
                    const auto& full = g_debugTargetContour2D;
                    const int Kt = (int)full.size();
                    for (int i = 0; i + 1 < Kt; i++) {
                        const bool lower = (full[i].y > cy);
                        const ImU32 col = lower
                            ? IM_COL32(170, 100, 220, 255)   // purple
                            : IM_COL32(110, 110, 115, 255);  // gray
                        dl->AddLine(px2cv(full[i]), px2cv(full[i + 1]), col, 1.5f);
                    }

                    // Centroid line + cross
                    {
                        const ImVec2 a = px2cv(glm::vec2(0.0f, cy));
                        const ImVec2 b = px2cv(glm::vec2(float(W_img), cy));
                        dl->AddLine(a, b, IM_COL32(90, 90, 100, 160), 1.0f);
                        const ImVec2 c = px2cv(g_silSwTgtCentroid2DPreview);
                        const float d = 8.0f;
                        const ImU32 yc = IM_COL32(255, 240, 80, 255);
                        dl->AddLine(ImVec2(c.x - d, c.y), ImVec2(c.x + d, c.y), yc, 2.0f);
                        dl->AddLine(ImVec2(c.x, c.y - d), ImVec2(c.x, c.y + d), yc, 2.0f);
                        dl->AddText(ImVec2(c.x + 10, c.y - 18), yc, "centroid");
                    }

                    // Right-end (max x) start marker
                    if (!g_silSwTgtLower2DPreview.empty()) {
                        const ImVec2 c = px2cv(g_silSwTgtLower2DPreview.front());
                        dl->AddCircle(c, 10.0f, IM_COL32(255, 240, 80, 255), 0, 3.0f);
                        dl->AddText(ImVec2(c.x + 12, c.y + 6),
                                    IM_COL32(255, 240, 80, 255), "start");
                    }

                    // 20 anchors (rainbow), same coloring as source pivots
                    const auto& an = g_silSwTgtAnchors2DPreview;
                    const int Na = (int)an.size();
                    for (int p = 0; p < Na; p++) {
                        const float hue = (Na > 1) ? (float(p) / float(Na)) * 0.83f : 0.0f;
                        const glm::vec3 c3 = cyclicHsv2rgb(hue, 0.95f, 1.0f);
                        const ImU32 col = IM_COL32(int(c3.r * 255),
                                                   int(c3.g * 255),
                                                   int(c3.b * 255), 255);
                        const ImVec2 c = px2cv(an[p]);
                        dl->AddCircleFilled(c, 5.0f, col);
                        dl->AddCircle(c, 6.0f, IM_COL32(255, 255, 255, 255), 0, 1.0f);
                        char idbuf[8];
                        std::snprintf(idbuf, sizeof(idbuf), "%d", p);
                        dl->AddText(ImVec2(c.x + 7, c.y - 14), col, idbuf);
                    }

                    ImGui::Dummy(ImVec2(canvas_w, canvas_h));
                }
                ImGui::End();
            }
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(gWindow);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(gWindow);
    glfwTerminate();
    return 0;
}

void showFPS(GLFWwindow* window) {
    static double previousSeconds = 0.0;
    static int frameCount = 0;
    double elapsedSeconds;
    double currentSeconds = glfwGetTime();

    elapsedSeconds = currentSeconds - previousSeconds;

    if (elapsedSeconds > 0.25) {
        previousSeconds = currentSeconds;
        double fps = (double)frameCount / elapsedSeconds;
        double msPerFrame = 1000.0 / fps;

        int gcd = 1;  // Simplified without std::gcd
        int aspectWidth = gWindowWidth / gcd;
        int aspectHeight = gWindowHeight / gcd;
        double aspectRatio = (double)gWindowWidth / (double)gWindowHeight;

        std::ostringstream outs;
        outs.precision(3);
        outs << std::fixed
             << "FPS: " << fps << "    "
             << "Frame Time: " << msPerFrame << " (ms)    "
             << "Window: " << gWindowWidth << "x" << gWindowHeight << "    "
             << "Aspect: " << aspectWidth << ":" << aspectHeight
             << " (" << aspectRatio << ")";
        glfwSetWindowTitle(window, outs.str().c_str());
        frameCount = 0;
    }
    frameCount++;
}

static void syncUIState() {
    auto& s = gUI.state;
    s.mainMode          = 0;
    s.depthRunning      = false;
    // MaskSelectionとImageOnlyモードの時はdepthDone=falseにしてDepthセクションを表示
    s.depthDone         = (gApp.mode == AppMode::kRegistration);
    s.hasLocalImage     = gApp.image.loaded;

    if (gApp.image.loaded) {
        auto pos = gApp.image.path.find_last_of("/\\");
        s.localImageName = (pos == std::string::npos)
                               ? gApp.image.path : gApp.image.path.substr(pos + 1);
    } else {
        s.localImageName.clear();
    }

    int nFg = 0, nBg = 0;
    for (const auto& p : gApp.maskPoints) (p.fg ? nFg : nBg)++;
    s.segFG = nFg;
    s.segBG = nBg;
    // Instrument-mask counters and active-kind indicator. Mirroring these
    // every frame keeps the UI live without needing an event when points
    // are added / removed via MaskPicker.
    int iFg = 0, iBg = 0;
    for (const auto& p : gApp.instrumentMaskPoints) (p.fg ? iFg : iBg)++;
    s.instSegFG = iFg;
    s.instSegBG = iBg;
    s.activeMaskKind = (gApp.activeMaskKind == MaskKind::Instrument) ? 1 : 0;

    s.depthScale       = gDepthScale;
    // RegistrationData の状態を UI にマッピング
    if (gApp.mode == AppMode::kImageOnly) {
        s.regState = 0;
    } else {
        switch (registrationHandle.state) {
        case RegistrationData::IDLE:                    s.regState = 0; break;
        case RegistrationData::SELECTING_BOARD_POINTS:  s.regState = 1; break;
        case RegistrationData::SELECTING_OBJECT_POINTS: s.regState = 2; break;
        case RegistrationData::READY_TO_REGISTER:       s.regState = 3; break;
        case RegistrationData::REGISTERED:              s.regState = 4; break;
        case RegistrationData::REFINING:                s.regState = 5; break;
        }
    }
    s.boardPtCount     = (int)registrationHandle.boardPoints.size();
    s.objPtCount       = (int)registrationHandle.objectPoints.size();
    s.targetPtCount    = registrationHandle.targetPointCount;
    s.useRegistration  = registrationHandle.useRegistration;
    s.rmse             = registrationHandle.compRmse;
    s.avgError         = registrationHandle.compAvgError;
    s.maxError         = registrationHandle.compMaxError;

    // Pose Library の状態をUIに反映
    s.poseLibraryOpen    = g_poseLibrary.showWindow;
    s.poseUndoAvailable  = g_poseLibrary.hasLastRegistration;
    s.poseEntryCount     = (int)g_poseLibrary.entries.size();

    s.splitScreen      = gUmeyama.active;
    s.depthSplitScreen = false;
    // カメラの状態を反映 0:未起動, 1:ライブビュー, 2:キャプチャ済み
    if (!gCamera.active) {
        s.cameraState = 0;
    } else if (!gCamera.captured) {
        s.cameraState = 1;  // ライブビュー中（Captureボタンを表示）
    } else {
        s.cameraState = 2;  // キャプチャ済み（Restart Cameraボタンを表示）
    }

    s.depthModelIdx    = gCurrentDepthModel;
    for (int i = 0; i < 3; i++) s.depthModelAvail[i] = isDepthModelAvailable(i);

    s.clusterVis        = g_showClusterVisualization;
    s.correspondenceVis = g_showCorrespondencePoints;

    s.hemiVoxelSize     = g_voxelSize;

    // 器具マスク連動: スライダー値と「マスクが有効か」フラグを UI 側へ
    s.instrumentPxThresh   = g_instrumentPxThresh;
    s.instrumentMaskActive = g_instrumentDistMap.valid;

    // Vignette toggle: mirror AppContext state into the UI so the
    // checkbox reflects the value used by the next Run Depth.
    s.detectVignette = gApp.detectVignette;

    // GPU toggle: same mirroring pattern as the vignette flag.
    s.useCuda = gApp.useCuda;

    // 臓器+board+targetのアルファ値をUIに反映
    for (int i = 0; i < 6; i++)
        s.organs[i].alpha = g_meshAlpha[i];
    s.boardAlpha  = g_meshAlpha[6];
    s.targetAlpha = g_meshAlpha[7];

    // ---- Step 7: intrinsics source dropdown + Active card ----
    s.intrinsicsSource  = g_intrinsicsSource;          // enum, direct
    s.currentPresetKey  = g_currentPresetKey;
    s.currentFx = g_intrinsics.fx; s.currentFy = g_intrinsics.fy;
    s.currentCx = g_intrinsics.cx; s.currentCy = g_intrinsics.cy;
    s.currentWidth = g_intrinsics.width; s.currentHeight = g_intrinsics.height;

    // Availability of file-backed sources.
    {
        namespace fs = std::filesystem;
        const std::string base = DEPTH_OUTPUT_PATH;
        s.customAvailable    = fs::exists(base + "intrinsics_custom.txt");
        s.calibLastAvailable = fs::exists(base + "intrinsics_calib_last.txt")
                            || fs::exists(base + "intrinsics_calib.txt");
        s.da3LastAvailable   = fs::exists(base + "intrinsics_da3_last.txt");
    }

    // Human-readable Active card label.
    switch (g_intrinsicsSource) {
        case IntrinsicsSource::Custom:
            s.currentDisplayName = "Custom (intrinsics_custom.txt)"; break;
        case IntrinsicsSource::Calib:
            s.currentDisplayName = "Calib (last)"; break;
        case IntrinsicsSource::Preset: {
            s.currentDisplayName = g_currentPresetKey;
            for (auto& p : Reg3DCustom::presetRegistry())
                if (g_currentPresetKey == p.key) { s.currentDisplayName = p.displayName; break; }
            break;
        }
        case IntrinsicsSource::DA3:
            s.currentDisplayName = "DA3 (last estimate)"; break;
    }

    // Preset list for the dropdown (rebuilt each frame; small + cheap).
    s.presetList.clear();
    for (auto& p : Reg3DCustom::presetRegistry()) {
        RegUIState::PresetEntry e;
        e.key = p.key; e.displayName = p.displayName;
        e.available = p.K.valid(); e.dynamic = false;  // no dynamic presets anymore
        s.presetList.push_back(std::move(e));
    }

    // Reconstruct-from-BIN slot mirror (FEATURE Task 5).
    {
        auto cp = [](RegUIState::ReconSlot& d, const ReconstructFromBin::Slot& sl) {
            d.status = (int)sl.status; d.width = sl.width; d.height = sl.height;
            d.err = sl.errorMessage;
        };
        cp(s.reconBin,  g_reconstructState.bin);
        cp(s.reconMask, g_reconstructState.mask);
        cp(s.reconRgb,  g_reconstructState.rgb);
        s.reconReady       = g_reconstructState.isReconstructReady();
        s.reconHasAny      = g_reconstructState.hasAny();
        s.reconKWidth      = g_intrinsics.width;
        s.reconKHeight     = g_intrinsics.height;
        s.reconKSourceName = intrinsicsSourceToTag(g_intrinsicsSource, g_currentPresetKey);
    }

    // Settings tab — calibration parameters.
    s.chessboardFolder    = g_chessboardFolder;
    s.chessboardBoardCols = g_chessboardCols;
    s.chessboardBoardRows = g_chessboardRows;
    s.chessboardSquareMm  = g_chessboardSquareMm;

    // Calibration result state.
    s.calibDone     = g_calibResult.valid;
    s.calibMessage  = g_calibResult.message;
    s.calibFx       = (float)g_calibResult.fx;
    s.calibFy       = (float)g_calibResult.fy;
    s.calibCx       = (float)g_calibResult.cx;
    s.calibCy       = (float)g_calibResult.cy;
    s.calibRms      = (float)g_calibResult.rmsError;
    s.calibImgCount = g_calibResult.numImages;

    // Live Calibration (Take Picture tab) state.
    s.liveCalibSessionActive = g_liveCalibActive;
    s.liveCalibFolder        = g_liveCalibFolder;
    s.liveCalibCapturedCount = (int)g_liveCalibFiles.size();
    s.liveCalibLastMessage   = g_liveCalibLastMsg;
    s.liveCalibCamW          = gCamera.width;
    s.liveCalibCamH          = gCamera.height;
    s.liveCalibCapturedFiles.clear();
    for (const auto& fp : g_liveCalibFiles) {
        s.liveCalibCapturedFiles.push_back(std::filesystem::path(fp).filename().string());
    }

    // チャット 9: 4-quadrant mask 連動 (Ctrl+G ↔ Initial Orientation で完全共有)
    //   g_activeQuadrantMask を毎フレーム UI 側にコピー → 別 panel で変えても
    //   両方の checkbox が同期して見える。
    //   ラベルが未計算なら頂点数 = 0 を渡し、UI 側で "labels not computed" 表示。
    s.activeQuadrantMask = g_activeQuadrantMask;
    const bool quadReady = g_liverRegion.valid() && g_liverLR.valid();
    s.quadLabelsReady = quadReady;

    // Ctrl+G の 6-DoF/7-DoF checkbox 用同期: g_ctrlgSearchMode (enum) を
    // bool に折りたたむ。SIX_DOF_RIGID → true、SEVEN_DOF/FOUR_DOF → false。
    // 左 floating パネルで radio button を操作した場合もこの経路で反映される。
    s.ctrlgLockScale =
        (g_ctrlgSearchMode == CmaesRefineV3R::SearchMode::SIX_DOF_RIGID);
    if (quadReady) {
        LiverLeftRightLabel::countByQuadrant(
            g_liverRegion.labels, g_liverLR.labels,
            s.quadNAR, s.quadNAL, s.quadNPR, s.quadNPL);
        auto subset = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels, g_liverLR.labels,
            g_activeQuadrantMask);
        s.quadSubsetCount = (int)subset.size();
        s.quadTotalCount  = (int)g_liverRegion.labels.size();
    } else {
        s.quadNAR = s.quadNAL = s.quadNPR = s.quadNPL = 0;
        s.quadSubsetCount = 0;
        s.quadTotalCount  = 0;
    }

    // ---- Anatomical Axes Status (Preview OBJ Anatomical Pose) ----
    //   3 軸ラベルの状態を毎フレーム UI 側にコピー。confidence と weak flag、
    //   flip 状態の現在値も同期する。Apply Init Pose で auto-trigger される
    //   経路と、Shift+R / Y / Shift+H で個別に計算する経路、どちらでも
    //   毎フレーム反映される (UI 側は読み取り専用)。
    s.apAxisValid  = g_liverRegion.valid();
    s.lrAxisValid  = g_liverLR.valid();
    s.ccAxisValid  = g_liverCC.valid();
    if (s.lrAxisValid) {
        // LR confidence: |a_vis_pos - a_vis_neg| / a_avg。decisive = >= 2%。
        const auto& e = g_liverLR.eclipse;
        float a_avg = 0.5f * (e.a_vis_pos + e.a_vis_neg);
        s.lrConfidence = (a_avg > 1e-9f)
            ? (std::fabs(e.a_vis_pos - e.a_vis_neg) / a_avg)
            : 0.0f;
        if (s.lrConfidence > 1.0f) s.lrConfidence = 1.0f;
        s.lrDecisive = e.decisive;
    } else {
        s.lrConfidence = 0.0f;
        s.lrDecisive   = false;
    }
    if (s.ccAxisValid) {
        s.ccConfidence = g_liverCC.cc.confidence;
        s.ccWeak       = g_liverCC.cc.weak;
    } else {
        s.ccConfidence = 0.0f;
        s.ccWeak       = false;
    }
    s.lrFlipped = g_lrFlipManual;
    s.ccFlipped = g_ccFlipManual;

    if (s.arSavedTimer > 0) s.arSavedTimer -= ImGui::GetIO().DeltaTime;
}

// =========================================================================
//  Rebuild the point cloud (OBJ) from the depth already loaded in
//  g_reconstructState, using the CURRENTLY ACTIVE intrinsics -- WITHOUT
//  re-running DA3/SAM2. Shared by:
//    - a.onReconstruct            (slots filled by a bin/mask/rgb drop)
//    - a.onRebuildFromCurrentDepth (slots auto-filled from depth_output/)
//  Crucially, after the OBJ is rebuilt this reloads intrinsics.txt (the K just
//  baked into the OBJ) into the display/OrbitCam K, so display K == OBJ K and
//  the AR overlay lines up with the photo again.
// =========================================================================
static void reconstructFromLoadedDepthWithCurrentK() {
    // K source-of-truth = current g_intrinsics; tag from the source enum.
    Reg3DCustom::CameraIntrinsics K = g_intrinsics;
    K.name = intrinsicsSourceToTag(g_intrinsicsSource, g_currentPresetKey);
    auto res = ReconstructFromBin::execute(g_reconstructState, DEPTH_OUTPUT_PATH, K);
    if (!res.ok) {
        std::cerr << "[Reconstruct] aborted: " << res.errorMessage << std::endl;
        return;
    }
    // AR background = the (processed-resolution) original.jpg in depth_output/.
    {
        const std::string orig = DEPTH_OUTPUT_PATH + "original.jpg";
        if (std::filesystem::exists(orig)) ImageSession::load(gApp, orig);
    }
    // Mask-derived caches depend on segmentation_mask.png.
    g_boundaryDistMap.invalidate();
    g_instrumentDistMap.invalidate();
    g_projectedLiverMask.invalidate();
    g_gnUnsignedBdyValid = false;

    // Load the reconstructed canonical OBJ into the scene.
    g_objSourcePath = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj";
    if (!setupObjScene()) {
        std::cerr << "[Reconstruct] setupObjScene failed" << std::endl;
        return;
    }
    // Sync display/OrbitCam K to the K actually baked into the OBJ
    // (intrinsics.txt) so AR overlay == photo (no display-K vs OBJ-K mismatch).
    Reg3DCustom::CameraIntrinsics Kc;
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics.txt", Kc)
        && Kc.valid()) {
        g_intrinsics = Kc; gApp.intrinsics = Kc;
        OrbitCam.setIntrinsics(Kc.fx, Kc.fy, Kc.cx, Kc.cy, Kc.width, Kc.height);
    }
    snapshotInitialPose();
    computeTargetSubsetAabbs();
    computeSourceLiverSubsetAabbs();
    gApp.mode = AppMode::kRegistration;
    std::cout << "[Reconstruct] scene updated; mode=Registration" << std::endl;
}

static void setupUICallbacks() {
    auto& a = gUI.actions;

    a.onLoadLocalImage = []() {
#ifdef HAS_TINYFILEDIALOGS
        const char* filters[] = {"*.png", "*.jpg", "*.jpeg", "*.ppm", "*.bmp"};
        const char* selected = tinyfd_openFileDialog(
            "Load Image for Depth",
            "",
            5, filters,
            "Image Files (png/jpg/ppm/bmp)",
            0
            );
        if (selected) {
            std::cout << "[FilePicker] Selected: " << selected << std::endl;
            loadImageRectifyAware(gApp, selected);
            if (gApp.mode == AppMode::kRegistration) {
                gApp.mode = AppMode::kImageOnly;
            }
        } else {
            std::cout << "[FilePicker] Cancelled" << std::endl;
        }
#else
        std::cerr << "[FilePicker] tinyfiledialogs not available." << std::endl;
#endif
    };

    a.onUndoSegPoint = []() {
        MaskPicker::undo(gApp);
    };

    a.onClearPoints = []() {
        MaskPicker::clear(gApp);
    };

    a.onRunDepth = []() {
        AppMode previousMode = gApp.mode;
        std::cout << "[UI] Run Depth" << std::endl;

        // カメラモードかどうかチェック
        if (gCamera.active && (gApp.image.path == "[Camera Live]" || gApp.image.path == "[Camera Captured]")) {
            // カメラの場合、マスクポイントの座標変換が必要
            // 画面上のクリック位置（左右反転済み）を元の画像座標に変換
            std::vector<MaskPoint> originalMaskPoints       = gApp.maskPoints;
            std::vector<MaskPoint> originalInstrumentPoints = gApp.instrumentMaskPoints;
            for (auto& p : gApp.maskPoints) {
                p.u = gApp.image.width - p.u;  // X座標を反転
            }
            for (auto& p : gApp.instrumentMaskPoints) {
                p.u = gApp.image.width - p.u;  // X座標を反転
            }

            // カメラから深度推定を実行
            bool result = runCameraDepthEstimation();

            // マスクポイントを元に戻す
            gApp.maskPoints           = originalMaskPoints;
            gApp.instrumentMaskPoints = originalInstrumentPoints;

        } else {
            // 通常の画像から深度推定を実行
            runDepthAndUpdateScene(gApp);
        }
    };

    // ---- Segment 1 / Instrument button handlers -----------------------------
    // Each handler does two things:
    //   (1) make its mask kind the active one (so future clicks land in the
    //       right list and the renderer picks the right colors);
    //   (2) if that list already has a foreground point, run a SAM2-only
    //       preview popup so the user can sanity-check the mask before
    //       committing to Run Depth.
    //
    // Camera-live mode mirrors the click coordinates left/right because the
    // preview is also flipped; we restore the original click coords after
    // the runner returns so the user's edits aren't disturbed.
    auto runSegmentForCamera = [](AppContext& ctx, MaskKind kind) {
        // The external SAM2 tool loads the image from disk via fopen(), but in
        // camera mode ctx.image.path is a placeholder ("[Camera Captured]" /
        // "[Camera Live]"), which can't be opened. Save the current frame to a
        // real file first (same as Run Depth via runCameraDepthEstimation),
        // then restore the placeholder afterwards so other code paths are
        // unaffected.
        std::string savedPath = gCamera.saveForDepthEstimation();
        if (savedPath.empty()) {
            std::cerr << "[Segment] failed to save camera frame for preview"
                      << std::endl;
            return;
        }
        std::string origPath = ctx.image.path;
        ctx.image.path = savedPath;

        // Pick which list to flip based on the kind being previewed.
        std::vector<MaskPoint>& target =
            (kind == MaskKind::Instrument) ? ctx.instrumentMaskPoints
                                           : ctx.maskPoints;
        std::vector<MaskPoint> backup = target;
        for (auto& p : target) p.u = ctx.image.width - p.u;
        runSegmentOnly(ctx, kind);
        target = backup;

        ctx.image.path = origPath;  // restore camera placeholder
    };

    a.onSegment1 = [runSegmentForCamera]() {
        std::cout << "[UI] Segment 1 (Liver)" << std::endl;
        // (1) Activate Liver mask (next click lands in maskPoints).
        gApp.activeMaskKind = MaskKind::Liver;
        // (2) Preview only if there's at least one foreground point.
        bool hasFg = false;
        for (const auto& p : gApp.maskPoints) if (p.fg) { hasFg = true; break; }
        if (!hasFg) {
            std::cout << "[Segment1] activated; click a foreground point "
                         "and press Segment 1 again to preview" << std::endl;
            return;
        }
        if (gCamera.active &&
            (gApp.image.path == "[Camera Live]" ||
             gApp.image.path == "[Camera Captured]"))
        {
            runSegmentForCamera(gApp, MaskKind::Liver);
        } else {
            runSegmentOnly(gApp, MaskKind::Liver);
        }
    };

    a.onSegment2 = [runSegmentForCamera]() {
        std::cout << "[UI] Instrument (Segment 2)" << std::endl;
        // (1) Activate Instrument mask (next click lands in instrumentMaskPoints).
        gApp.activeMaskKind = MaskKind::Instrument;
        // (2) Preview only if there's at least one foreground point.
        bool hasFg = false;
        for (const auto& p : gApp.instrumentMaskPoints)
            if (p.fg) { hasFg = true; break; }
        if (!hasFg) {
            std::cout << "[Instrument] activated; click a foreground point "
                         "(cyan) and press Instrument again to preview"
                      << std::endl;
            return;
        }
        if (gCamera.active &&
            (gApp.image.path == "[Camera Live]" ||
             gApp.image.path == "[Camera Captured]"))
        {
            runSegmentForCamera(gApp, MaskKind::Instrument);
        } else {
            runSegmentOnly(gApp, MaskKind::Instrument);
        }
    };

    a.onResetDefaultImage = []() {
        loadImageRectifyAware(gApp, DEPTH_OUTPUT_PATH + "original.jpg");
        gApp.maskPoints.clear();
    };

    a.onHemiAuto = []() {
        // 元コード line 3213-3238 通り
        gUI.state.regMethod = 1;
        g_stepStartTime  = std::chrono::steady_clock::now();
        g_sessionBipopN  = 0;
        poseAutoSaveBeforeRegistration();
        runHemiAuto();
        poseSaveToLibrary(SaveCriterion::RMSE);
    };

    a.onQuadAuto = []() {
        // Shift+O / QuadAuto: labels auto-trigger してから runQuadAuto。
        //   a.onHemiAuto と並列のラッパー。UI ボタン経由でも同等の動作を保証
        //   (現時点で UI ボタンは未配線 — 将来 HemiAuto ボタン横に追加可能)。
        if (!g_liverRegion.valid()) {
            std::cout << "[QuadAuto] LiverRegion not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverRegion();
        }
        if (!g_liverLR.valid()) {
            std::cout << "[QuadAuto] LiverLR not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverLR();
        }
        gUI.state.regMethod = 1;
        g_stepStartTime  = std::chrono::steady_clock::now();
        g_sessionBipopN  = 0;
        poseAutoSaveBeforeRegistration();
        runQuadAuto();
        poseSaveToLibrary(SaveCriterion::RMSE);
    };

    a.onQuadCyclic = []() {
        // Ctrl+P / QuadCyclic: labels auto-trigger してから runQuadCyclic。
        //   a.onQuadAuto と並列のラッパー。UI ボタン経由でも同等の動作を保証
        //   (現時点で UI ボタンは未配線 — 将来追加可能)。
        if (!g_liverRegion.valid()) {
            std::cout << "[QuadCyclic] LiverRegion not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverRegion();
        }
        if (!g_liverLR.valid()) {
            std::cout << "[QuadCyclic] LiverLR not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverLR();
        }
        gUI.state.regMethod = 1;
        g_stepStartTime  = std::chrono::steady_clock::now();
        g_sessionBipopN  = 0;
        poseAutoSaveBeforeRegistration();
        runQuadCyclic();
        poseSaveToLibrary(SaveCriterion::RMSE);
    };

    a.onQuadCyclicRansac = []() {
        // Shift+Ctrl+P / QuadCyclic-RANSAC: labels auto-trigger → runQuadCyclicRansac。
        //   a.onQuadCyclic と同形のラッパー。UI ボタン経由でも同等の動作を保証
        //   (現時点で UI ボタンは未配線 — 将来 Cyclic Tuning panel で追加予定)。
        if (!g_liverRegion.valid()) {
            std::cout << "[QuadCyclic-RANSAC] LiverRegion not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverRegion();
        }
        if (!g_liverLR.valid()) {
            std::cout << "[QuadCyclic-RANSAC] LiverLR not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverLR();
        }
        gUI.state.regMethod = 1;
        g_stepStartTime  = std::chrono::steady_clock::now();
        g_sessionBipopN  = 0;
        poseAutoSaveBeforeRegistration();
        runQuadCyclicRansac();
        poseSaveToLibrary(SaveCriterion::RMSE);
    };

    a.onBipopCmaes = []() {
        // 元コード line 3338-3445 通り
        g_stepStartTime = std::chrono::steady_clock::now();
        g_sessionBipopN++;
        gUI.state.regMethod = 3;
        poseAutoSaveBeforeRegistration();
        runBipopCmaes();
        poseSaveToLibrary(SaveCriterion::RMSE);
    };

    a.onSaveAR = []() {
        if (gApp.mode != AppMode::kRegistration || !screenMesh || !g_pShader) {
            std::cout << "[AR] Not in Registration mode or missing data" << std::endl;
            return;
        }
        std::vector<mCutMesh*> organs = {
            liverMesh3D, portalMesh3D, veinMesh3D,
            tumorMesh3D, segmentMesh3D, gbMesh3D
        };
        // 入力画像サイズ（キャリブレーション解像度）で保存
        int imgW = OrbitCam.calibWidth  > 0 ? OrbitCam.calibWidth  : 1280;
        int imgH = OrbitCam.calibHeight > 0 ? OrbitCam.calibHeight : 720;
        ARSave::capture(g_arSave, OrbitCam, *g_pShader, *g_pShaderCube,
                        gApp.arBg, organs, g_meshAlpha, objPos,
                        imgW, imgH,
                        DEPTH_OUTPUT_PATH, gWindowWidth, gWindowHeight);
        gUI.state.arSavedTimer = 2.0f;
    };

    a.onResetCamera = []() {
        OrbitCam.resetToInitialState();
        // OBJワークフローではカメラを180° Y回転が正しい初期位置
        if (gApp.mode == AppMode::kRegistration) {
            OrbitCam.rotation = glm::angleAxis(glm::radians(180.0f),
                                               glm::vec3(0.0f, 1.0f, 0.0f));
            OrbitCam.currentTarget = TARGET_TEXTURE;
        }
    };

    // [key-reorg Phase 12] Export buttons (were M / Shift+M keys).
    a.onExportStl        = []() { StlExport::exportRegisteredObjs(); };
    a.onExportStlFlipped = []() { StlExport::exportCamMmStlWithSnapshot(); };

    // [deform round-trip FIX] 「Load Deformed (ct_deformed/) [REG #2]」ボタン。
    // これまで未配線だったため、UI 側の `if (actions.onLoadDeformedCT)` ガードに
    // 弾かれて無言で何もしなかった（ログも出ない＝処理自体が走っていない）。
    // DEFORM が書き出す registration_model/ct_deformed/ を走査し、手動の
    // 「通常ドロップ」と完全に同じ経路 swapOrganSetFromDrop(..., false) で受ける。
    //   - ct_deformed/ は CT 空間なので preserveRegisteredPose は必ず false
    //     （prealign でターゲットに再フィット → register#2 の出発点）。
    //     true にすると CT スケール(肝 diag≈8.6mm)のまま置かれてズレる。
    //   - 深度ターゲット未ロード時は swapOrganSetFromDrop 側が [OrganDrop] REFUSED
    //     を出して false を返す。
    a.onLoadDeformedCT = []() {
        const std::string dir = REG_MODEL_PATH + "ct_deformed/";
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            std::cerr << "[LoadDeformedCT] folder not found: " << dir
                      << "  (DEFORM 側で Export Deformed (CT) を先に実行してください)"
                      << std::endl;
            return;
        }
        std::array<std::string, 6> paths{};
        bool any = false;
        for (int r = 0; r < 6; ++r) {
            const std::string cand =
                (std::filesystem::path(dir) / kOrganCanonical[r]).string();
            if (std::filesystem::exists(cand, ec)) {
                paths[r] = cand;
                any = true;
                std::cout << "[LoadDeformedCT] found: " << kOrganCanonical[r]
                          << std::endl;
            }
        }
        if (!any) {
            std::cerr << "[LoadDeformedCT] no canonical organ obj "
                         "(liver/portal/vein/tumor/segment/gb) in " << dir
                      << std::endl;
            return;
        }
        const bool ok = swapOrganSetFromDrop(paths, /*preserveRegisteredPose=*/false);
        std::cout << "[LoadDeformedCT] " << (ok ? "loaded" : "FAILED")
                  << " from " << dir
                  << (ok ? "  -> prealigned to target. 続けて register#2 を実行" : "")
                  << std::endl;
    };

    a.onToggleClusterVis = []() {
        g_showClusterVisualization = !g_showClusterVisualization;
    };

    a.onSwitchDepthModel = [](int i) {
        DepthRunner dummy;
        switchDepthModel(dummy, i);
    };

    a.onDepthScaleChanged = [](float v) {
        gDepthScale = v;
    };

    a.onIntrinsicsSourceChanged = [](int i) {
        // i is the legacy 4-button index (0=DA3,1=Kinect,2=Custom,3=Calib);
        // bridge to the enum. Kinect maps to the (sole) Azure Kinect preset.
        g_intrinsicsSource = intrinsicsSourceFromLegacyInt(i);
        if (g_intrinsicsSource == IntrinsicsSource::Preset)
            g_currentPresetKey = "azure_kinect_1080p";
        const char* names[] = {"DA3", "Kinect", "Custom", "Calibrated"};
        std::cout << "[Intrinsics] Source: " << names[std::clamp(i,0,3)] << std::endl;

        // ボタン押下と同時に g_intrinsics を更新する。これがないと Custom を
        // 選んだ直後にドロップした画像が rectify 経路を通らない
        // (loadWithIntrinsics は g_intrinsics を見るが、setupObjScene が
        // 呼ばれるまで g_intrinsics は古い K のまま) というバグになる。
        Reg3DCustom::CameraIntrinsics K;
        bool loaded = false;
        if (i == 2) {
            loaded = Reg3DCustom::loadCameraIntrinsics(
                DEPTH_OUTPUT_PATH + "intrinsics_custom.txt", K);
        } else if (i == 1) {
            loaded = Reg3DCustom::loadCameraIntrinsics(
                DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt", K);
            if (!loaded) {
                K = Reg3DCustom::CameraIntrinsics::k4a_color_720p();
                loaded = true;
            }
        } else if (i == 0) {
            loaded = Reg3DCustom::loadCameraIntrinsics(
                DEPTH_OUTPUT_PATH + "intrinsics.txt", K);
        } else if (i == 3) {
            if (g_calibResult.valid) {
                K.fx = (float)g_calibResult.fx;
                K.fy = (float)g_calibResult.fy;
                K.cx = (float)g_calibResult.cx;
                K.cy = (float)g_calibResult.cy;
                K.width  = g_calibResult.width;
                K.height = g_calibResult.height;
                K.name   = "calibrated";
                loaded = true;
            }
        }

        if (loaded && K.valid()) {
            g_intrinsics    = K;
            gApp.intrinsics = K;
            g_intrinsics_raw = Reg3DCustom::CameraIntrinsics{};  // fresh source K; drop any rectify-raw
            OrbitCam.setIntrinsics(K.fx, K.fy, K.cx, K.cy, K.width, K.height);
            std::cout << "[Intrinsics] Live K updated: "
                      << (K.name.empty() ? "(unnamed)" : K.name)
                      << "  " << K.width << "x" << K.height
                      << (K.hasDistortion() ? "  [has distortion]" : "")
                      << std::endl;
        } else {
            std::cerr << "[Intrinsics] Failed to load K for source "
                      << names[std::clamp(i,0,3)]
                      << " -- keeping previous K" << std::endl;
        }
    };

    // ---- Step 7: new intrinsics source dropdown callbacks (案 Y) ----
    a.onSourceChanged = [](IntrinsicsSource s) {
        g_intrinsicsSource = s;
        const char* nm[] = {"Custom", "Calib", "Preset", "DA3"};
        std::cout << "[Intrinsics] Source -> " << nm[std::clamp((int)s, 0, 3)] << std::endl;
        loadIntrinsicsFromCurrentSource();
    };
    a.onPresetChanged = [](const std::string& key) {
        g_currentPresetKey = key;
        g_intrinsicsSource = IntrinsicsSource::Preset;
        std::cout << "[Intrinsics] Preset -> " << key << std::endl;
        loadIntrinsicsFromCurrentSource();
    };
    a.onSaveAsCustom = []() {
        if (!g_intrinsics.valid()) {
            std::cerr << "[Intrinsics] Save as Custom: current K invalid; ignored" << std::endl;
            return;
        }
        Reg3DCustom::CameraIntrinsics K = g_intrinsics;
        K.name = "custom";
        const std::string path = DEPTH_OUTPUT_PATH + "intrinsics_custom.txt";
        if (Reg3DCustom::saveCameraIntrinsics(path, K)) {
            g_intrinsicsSource = IntrinsicsSource::Custom;
            loadIntrinsicsFromCurrentSource();
            std::cout << "[Intrinsics] Saved current K as Custom -> " << path << std::endl;
        }
    };
    a.onChessboardFolderChanged = [](const std::string& f) { g_chessboardFolder = f; };
    a.onBoardSizeChanged = [](int c, int r) { g_chessboardCols = c; g_chessboardRows = r; };
    a.onSquareSizeChanged = [](float mm) { g_chessboardSquareMm = mm; };

    // ---- Reconstruct from BIN (FEATURE Task 7) ----
    a.onReconstructClear = []() {
        g_reconstructState.clear();
        std::cout << "[Reconstruct] cleared all slots" << std::endl;
    };
    a.onReconstruct = []() { reconstructFromLoadedDepthWithCurrentK(); };

    // One-click: rebuild the cloud from the CURRENT session depth
    // (depth_output/depth_metric.bin + segmentation_mask.png [+ original.jpg])
    // with the active K, no DA3/SAM2. Use after changing the intrinsics source
    // (drop a custom K, or Run Calibration) so the point cloud is re-unprojected
    // with the new K and the AR overlay lines up with the photo again.
    a.onRebuildFromCurrentDepth = []() {
        // main.cpp owns the stb_image implementation; provide it as the decoder.
        ReconstructFromBin::ImageDecoder decode =
            [](const std::string& p, int channels,
               std::vector<uint8_t>& out, int& w, int& h) -> bool {
            int c = 0;
            unsigned char* px = stbi_load(p.c_str(), &w, &h, &c, channels);
            if (!px) return false;
            out.assign(px, px + (size_t)w * h * channels);
            stbi_image_free(px);
            return true;
        };
        // Auto-load depth_output/{depth_metric.bin, segmentation_mask.png,
        // original.jpg} into the reconstruct slots (clear first so a stale prior
        // drop can't leak in).
        g_reconstructState.clear();
        ReconstructFromBin::onFolderDropped(g_reconstructState, DEPTH_OUTPUT_PATH, decode);
        if (!g_reconstructState.isReconstructReady()) {
            std::cerr << "[Rebuild] depth_output/ has no usable depth_metric.bin + "
                         "segmentation_mask.png pair. Run Depth first." << std::endl;
            return;
        }
        reconstructFromLoadedDepthWithCurrentK();
        std::cout << "[Rebuild] cloud rebuilt from current depth with active K"
                  << std::endl;
    };

    a.onRunCalibration = []() {
        std::cout << "[Calib] Running calibration_tool..." << std::endl;

        // Find calibration_tool executable (same dir as sam2_da3_lite)
        std::string exeDir = std::filesystem::path(DEPTH_EXE_PATH).parent_path().string();
        std::string exe;
        const std::vector<std::string> candidates = {
            "./calibration_tool",
            exeDir + "/calibration_tool",
        };
        for (auto& c : candidates) {
            if (std::filesystem::exists(c)) { exe = c; break; }
        }
        if (exe.empty()) {
            std::cerr << "[Calib] calibration_tool not found!\n"
                      << "  CWD: " << std::filesystem::current_path() << "\n"
                      << "  Searched:" << std::endl;
            for (auto& c : candidates) {
                std::string abs;
                try { abs = std::filesystem::absolute(c).string(); } catch (...) { abs = c; }
                std::cerr << "    " << abs << "  "
                          << (std::filesystem::exists(c) ? "[OK]" : "[NOT FOUND]") << std::endl;
            }
            g_calibResult.message = "calibration_tool not found (check build)";
            return;
        }

        try { exe = std::filesystem::absolute(exe).string(); } catch (...) {}
        std::cout << "[Calib] exe: " << exe << std::endl;

        // Step 7: Calibration パラメータは Settings タブから編集可能な globals を使う。
        std::string folder  = g_chessboardFolder;
        std::string outFile = DEPTH_OUTPUT_PATH + "intrinsics_calib.txt";
        std::string board   = std::to_string(g_chessboardCols) + ","
                            + std::to_string(g_chessboardRows);
        std::ostringstream sq; sq << g_chessboardSquareMm;

        std::string cmd = "\"" + exe + "\" \"" + folder + "\""
                          + " --board " + board + " --square " + sq.str()
                          + " --output \"" + outFile + "\""
                          + " 2>&1";

        std::cout << "[Calib] " << cmd << std::endl;
        FILE* pipe = PLATFORM_POPEN(cmd.c_str(), "r");
        if (!pipe) {
            g_calibResult.message = "popen failed";
            std::cerr << "[Calib] " << g_calibResult.message << std::endl;
            return;
        }
        char buf[512];
        while (fgets(buf, sizeof(buf), pipe)) std::cout << buf;
        int exitCode = PLATFORM_PCLOSE(pipe);

        if (exitCode != 0) {
            g_calibResult.message = "calibration_tool exit code " + std::to_string(exitCode);
            std::cerr << "[Calib] " << g_calibResult.message << std::endl;
            return;
        }

        // Read result file
        std::ifstream ifs(outFile);
        if (!ifs.is_open()) {
            g_calibResult.message = "Cannot open " + outFile;
            return;
        }
        std::string key; double val;
        while (ifs >> key >> val) {
            if      (key == "fx")     g_calibResult.fx = val;
            else if (key == "fy")     g_calibResult.fy = val;
            else if (key == "cx")     g_calibResult.cx = val;
            else if (key == "cy")     g_calibResult.cy = val;
            else if (key == "k1")     g_calibResult.k1 = val;
            else if (key == "k2")     g_calibResult.k2 = val;
            else if (key == "width")  g_calibResult.width  = (int)val;
            else if (key == "height") g_calibResult.height = (int)val;
            else if (key == "rms")    g_calibResult.rmsError = val;
        }
        g_calibResult.valid = (g_calibResult.fx > 0 && g_calibResult.fy > 0);
        g_calibResult.message = g_calibResult.valid ? "OK" : "Invalid result";
        if (g_calibResult.valid) {
            g_intrinsicsSource = IntrinsicsSource::Calib;
            // Pre-existing bug fix: switching the source enum is NOT enough.
            // The in-memory K (g_intrinsics) is what the depth runner reads
            // at L3495 (req.K = scaleIntrinsics(g_intrinsics, ...)). Without
            // this reload, the next Run Depth would unproject with the stale
            // previous-source K (e.g. Factory 3840x2160 fx=2606 -> scaled
            // 1280x720 fx=868) and bake the wrong intrinsics into the OBJ
            // vertex positions, even though g_intrinsicsSource says "Calib"
            // and intrinsics.txt is later overwritten. loadIntrinsicsFrom-
            // CurrentSource reads the right file for the current enum and
            // updates g_intrinsics / gApp.intrinsics / OrbitCam in one shot.
            loadIntrinsicsFromCurrentSource();
        }

        std::cout << "[Calib] Result: fx=" << g_calibResult.fx
                  << " fy=" << g_calibResult.fy
                  << " cx=" << g_calibResult.cx
                  << " cy=" << g_calibResult.cy
                  << " rms=" << g_calibResult.rmsError << std::endl;
    };

    // ---- Live Calibration handlers (Take Picture tab) ----
    a.onLiveCalibStart = []() {
        namespace fs = std::filesystem;
        g_liveCalibFolder = std::string(DEPTH_OUTPUT_PATH)
                          + "chessboard_live_" + makeTimestampString() + "/";
        std::error_code ec;
        fs::create_directories(g_liveCalibFolder, ec);
        if (ec) {
            std::cerr << "[LiveCalib] Failed to create folder: " << ec.message() << std::endl;
            g_liveCalibLastMsg = "Folder creation failed";
            return;
        }
        g_liveCalibFiles.clear();
        g_liveCalibCounter = 0;
        g_liveCalibActive  = true;
        g_liveCalibLastMsg.clear();
        std::cout << "[LiveCalib] Session started: " << g_liveCalibFolder << std::endl;

        // Open the camera DIRECTLY here. We must NOT route through
        // onToggleCamera(): its start branch sets
        //   gApp.mode         = AppMode::kImageOnly;
        //   gApp.image.loaded = true;
        //   gApp.image.path   = "[Camera Live]";
        // which (a) hides the Take Picture tab — drawIntrinsicsSource is
        // gated on state.depthDone == (gApp.mode == kRegistration) — and
        // (b) makes drawDepthOverlay() paint the "[Liver] L-click = FG"
        // segmentation overlay (gated on hasLocalImage && !depthDone).
        // gApp.mode is left untouched so the workflow phase (typically
        // kRegistration when the tab is reachable) survives the session.
        // The dedicated live-calib branch in the main render loop pumps
        // frames into originalFrame and draws the preview full-screen
        // without MaskPicker / overlay.
        if (!gCamera.active) {
            if (!gCamera.start()) {
                std::cerr << "[LiveCalib] Camera failed to start" << std::endl;
                g_liveCalibLastMsg = "Camera start failed";
                g_liveCalibActive  = false;   // roll back so Stop is not needed
            }
        }
    };

    a.onLiveCalibStop = []() {
        g_liveCalibActive = false;
        g_liveCalibLastMsg.clear();
        // Symmetric with onLiveCalibStart: we did not touch gApp.mode or
        // gApp.image.loaded on the start side, so we do not touch them
        // here either. Clobbering them would yank the user out of e.g.
        // the registration phase they came from. Just stop the camera.
        if (gCamera.active) {
            gCamera.stop();
        }
        std::cout << "[LiveCalib] Session stopped" << std::endl;
    };

    a.onLiveCalibCapture = []() {
        if (!g_liveCalibActive) {
            std::cerr << "[LiveCalib] No active session" << std::endl;
            return;
        }
        if (!gCamera.active) {
            std::cerr << "[LiveCalib] Camera not running" << std::endl;
            g_liveCalibLastMsg = "Camera not running";
            return;
        }
        char fname[64];
        snprintf(fname, sizeof(fname), "frame_%03d.jpg", g_liveCalibCounter + 1);
        std::string fullPath = g_liveCalibFolder + fname;
        if (gCamera.saveFrameAsJPEG(fullPath)) {
            ++g_liveCalibCounter;
            g_liveCalibFiles.push_back(fullPath);
            g_liveCalibLastMsg = std::string("Captured ") + fname;
            std::cout << "[LiveCalib] Captured: " << fullPath << std::endl;
        } else {
            std::cerr << "[LiveCalib] Save failed: " << fullPath << std::endl;
            g_liveCalibLastMsg = "Save failed";
        }
    };

    a.onLiveCalibClear = []() {
        namespace fs = std::filesystem;
        if (!g_liveCalibActive) return;
        for (const auto& f : g_liveCalibFiles) {
            std::error_code ec;
            fs::remove(f, ec);
        }
        g_liveCalibFiles.clear();
        g_liveCalibCounter = 0;
        g_liveCalibLastMsg = "Cleared all captures";
        std::cout << "[LiveCalib] Cleared all captures" << std::endl;
    };

    a.onLiveCalibRun = []() {
        if (!g_liveCalibActive || g_liveCalibFiles.size() < 3) {
            std::cerr << "[LiveCalib] Need >= 3 captures" << std::endl;
            return;
        }
        // Reset the success flag BEFORE delegating. onRunCalibration only
        // writes g_calibResult.valid on the full happy path (L9875); any
        // early return (calibration_tool not found, popen fail, non-zero
        // exit, unreadable output) leaves valid at its previous value.
        // Without this reset, a prior successful session would make the
        // current failure look like success to our post-call check.
        g_calibResult.valid = false;

        // Point the shared calibration folder at this session and delegate to
        // the existing Run Calibration path (external calibration_tool). It
        // writes intrinsics_calib.txt and flips the source to Calib on success.
        // Per design decision: g_chessboardFolder stays on the live folder
        // afterwards, so the Settings tab shows where the frames came from.
        g_chessboardFolder = g_liveCalibFolder;
        std::cout << "[LiveCalib] Running calibration on " << g_liveCalibFolder
                  << " (" << g_liveCalibFiles.size() << " frames)" << std::endl;
        if (gUI.actions.onRunCalibration) gUI.actions.onRunCalibration();

        // Always end the session after Run Calibration, success OR failure.
        // Rationale: user explicitly wants to "return to the original image"
        // after pressing the button -- they should not be stuck in the
        // camera overlay if the calibration_tool exits non-zero (e.g. Zhang
        // method couldn't separate intrinsics). The captured JPEGs stay on
        // disk under g_liveCalibFolder so they can be re-run via the
        // Settings tab if desired. On success g_intrinsicsSource is
        // already switched to Calib by onRunCalibration (so the new K
        // takes effect immediately); state.calib* shows the result in the
        // Live Calibration tab. On failure g_liveCalibLastMsg explains
        // what happened so the user can start a new session.
        bool ok = g_calibResult.valid;
        g_liveCalibActive  = false;
        g_liveCalibLastMsg = ok
            ? std::string("Calibration OK - returned to scene")
            : std::string("Calibration failed - retry with new frames");
        if (gCamera.active) gCamera.stop();
        std::cout << "[LiveCalib] Session ended ("
                  << (ok ? "success" : "failure") << ")" << std::endl;
    };

    a.onLiveCalibOpenFolder = []() {
        if (g_liveCalibFolder.empty()) return;
#if defined(_WIN32)
        std::string cmd = "explorer \"" + g_liveCalibFolder + "\"";
#elif defined(__APPLE__)
        std::string cmd = "open \"" + g_liveCalibFolder + "\"";
#else
        std::string cmd = "xdg-open \"" + g_liveCalibFolder + "\" &";
#endif
        int rc = std::system(cmd.c_str());
        (void)rc;  // best-effort; file manager launch failure is non-fatal
    };

    a.onToggleOrgan = [](int i) {
        if (i < 0 || i >= 8) return;
        float a = g_meshAlpha[i];
        // ON(≥0.75) → OFF(0) → 50%(0.5) → ON(0.8)
        if (a >= 0.75f)      g_meshAlpha[i] = 0.0f;
        else if (a < 0.01f)  g_meshAlpha[i] = 0.5f;
        else                 g_meshAlpha[i] = 0.8f;
    };

    a.onToggleCorrespondenceVis = []() {
        g_showCorrespondencePoints = !g_showCorrespondencePoints;
    };

    a.onToggleCamera = []() {
        if (!gCamera.active) {
            // カメラ開始前に現在の状態を保存
            g_cameraBackupState.previousMode = gApp.mode;
            g_cameraBackupState.hadImage = gApp.image.loaded;
            g_cameraBackupState.previousImagePath = gApp.image.path;
            g_cameraBackupState.previousImageWidth = gApp.image.width;
            g_cameraBackupState.previousImageHeight = gApp.image.height;

            // カメラを開始（ライブビューモード、通常UIのまま）
            if (gCamera.start()) {
                // ImageOnlyモードでライブビュー表示（通常のUIが表示される）
                gApp.mode = AppMode::kImageOnly;
                gApp.maskPoints.clear();
                gApp.image.loaded = true;
                gApp.image.width = gCamera.width;
                gApp.image.height = gCamera.height;
                gApp.image.path = "[Camera Live]";
                std::cout << "[Camera] Started in live view mode (ImageOnly mode)" << std::endl;
            }
        } else if (!gCamera.captured) {
            // ライブビュー中 → キャプチャして静止画にする、マスク選択モードへ
            // 少し待ってから最初のフレームをキャプチャ
            for (int i = 0; i < 5; i++) {
                gCamera.capture(gApp.arBg);
            }
            gCamera.captureCurrentFrame();  // 静止画キャプチャ
            gApp.image.path = "[Camera Captured]";
            gApp.mode = AppMode::kMaskSelection;  // キャプチャ後にマスク選択モードへ
            std::cout << "[Camera] Frame captured, switched to mask selection mode" << std::endl;
        } else {
            // キャプチャ済み → カメラを停止
            gCamera.stop();
            gApp.mode = AppMode::kEmpty;
            gApp.image.loaded = false;
            std::cout << "[Camera] Stopped and returned to empty mode" << std::endl;
        }
    };

    a.onCameraBack = []() {
        // カメラを停止
        gCamera.stop();

        // 前の状態を復元
        gApp.mode = g_cameraBackupState.previousMode;
        gApp.image.loaded = g_cameraBackupState.hadImage;
        gApp.image.path = g_cameraBackupState.previousImagePath;
        gApp.image.width = g_cameraBackupState.previousImageWidth;
        gApp.image.height = g_cameraBackupState.previousImageHeight;

        // マスクポイントはクリア（新しい操作として開始）
        gApp.maskPoints.clear();

        std::cout << "[Camera] Returned to previous mode: "
                  << static_cast<int>(g_cameraBackupState.previousMode)
                  << (g_cameraBackupState.hadImage ? " (with previous image)" : " (no image)")
                  << std::endl;
    };

    a.onFullAuto = []() {
        std::cout << "[stub] onFullAuto" << std::endl;
    };

    a.onHemiVoxelChanged = [](float v) {
        g_voxelSize = v;
        std::cout << "[HemiVoxel] Changed to " << g_voxelSize << std::endl;
    };

    a.onInstrumentPxThreshChanged = [](float v) {
        g_instrumentPxThresh = std::max(0.0f, v);
        std::cout << "[InstrumentMask] threshold -> "
                  << g_instrumentPxThresh << " px"
                  << "  (next HemiAuto / O will reclassify)" << std::endl;
    };

    a.onDetectVignetteChanged = [](bool include) {
        // Update the AppContext flag. The next Instrument-preview or
        // Run-Depth invocation will pass this through to
        // DepthRunnerConfig::detectVignette which controls whether the
        // external pipeline merges the auto-detected FOV vignette into
        // instrument_segmentation_mask.png.
        gApp.detectVignette = include;
        std::cout << "[Occluder] include_vignette_in_occluder="
                  << (include ? "ON" : "OFF")
                  << "  (applies to NEXT Run Depth / Instrument preview)"
                  << std::endl;
    };

    a.onUseCudaChanged = [](bool useGpu) {
        // Update the AppContext flag. The next Instrument-preview or
        // Run-Depth invocation passes this through to
        // DepthRunnerConfig::useCuda which adds --cuda to the CLI so the
        // external pipeline registers the CUDAExecutionProvider.
        // The flag is harmless when sam2_da3_lite was built with
        // USE_CUDA=OFF -- the pipeline prints a "CUDA not available"
        // warning and falls back to CPU.
        gApp.useCuda = useGpu;
        std::cout << "[Depth] use_cuda="
                  << (useGpu ? "ON" : "OFF")
                  << "  (applies to NEXT Run Depth / Instrument preview)"
                  << std::endl;
    };

    a.onStartUmeyama = []() {
        gUmeyama.start(registrationHandle, OrbitCam, gWindowWidth, gWindowHeight,
                       gUI.state.umeyamaPtCount);
        gUI.state.regMethod = 2;
    };

    a.onExecuteUmeyama = []() {
        // 元コード line 3475-3489 通り
        // (g_stepStartTime は触らない; regMethod=2 は onStartUmeyama で設定済み)
        poseAutoSaveBeforeRegistration();
        auto organs = getOrganList();
        gUmeyama.execute(registrationHandle, organs, OrbitCam, gWindowWidth, gWindowHeight);
        computeUnifiedMetrics();
        poseSaveToLibrary(SaveCriterion::RMSE);
    };

    a.onUndoUmeyamaPoint = []() {
        gUmeyama.undoPoint(registrationHandle);
    };

    // Phase U-1: Soft Partition controls live in the Debug Panel (Ctrl+D) "U" tab.
    // Body uses only globals + runSoftPartitionCompute, so a no-capture lambda is
    // safe; assigned once here, rendered by DebugPanel::draw() via the drawUExtra hook.
    g_debugPanel.drawUExtra = []() {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f),
                           "Soft Partition (U-1)  [Shift+U]");

        // --- Step 1: Umeyama anchors ---
        const int  n           = g_softAnchors.count();
        const bool haveAnchors = g_softAnchors.valid();
        ImGui::TextColored(haveAnchors ? ImVec4(0.4f, 0.9f, 0.4f, 1)
                                       : ImVec4(0.9f, 0.6f, 0.3f, 1),
                           "1. Anchors: %d  (need >=%d)",
                           n, SoftPartition::MIN_GROUPS);
        ImGui::Text("   s_ume = %.4f", g_softUmeyamaScale);

        // --- Step 2: compute ---
        if (!haveAnchors) ImGui::BeginDisabled();
        if (ImGui::Button("2. Compute Soft Partition")) runSoftPartitionCompute();
        if (!haveAnchors) ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextColored(g_softPartReady ? ImVec4(0.4f, 0.9f, 0.4f, 1)
                                           : ImVec4(0.6f, 0.6f, 0.6f, 1),
                           g_softPartReady ? "ready" : "not ready");

        // --- params ---
        ImGui::Checkbox("auto sigma_tgt", &g_softPartParams.autoSigma);
        if (g_softPartParams.autoSigma)
            ImGui::SliderFloat("sigma factor",
                               &g_softPartParams.sigmaTgtAutoFactor, 0.2f, 1.5f);
        else
            ImGui::SliderFloat("sigma manual",
                               &g_softPartParams.sigmaTgtManual, 0.001f, 1.0f, "%.4f");
        ImGui::SliderFloat("baseline none",
                           &g_softPartParams.baselineNone, 0.05f, 0.95f);
        ImGui::SliderFloat("radius %tile",
                           &g_softPartParams.groupRadiusPercentile, 0.5f, 0.95f);

        // --- Step 3: recompute + show ---
        if (!g_softPartReady) ImGui::BeginDisabled();
        if (ImGui::Button("Recompute")) runSoftPartitionCompute();
        ImGui::SameLine();
        ImGui::Checkbox("3. Show heatmap", &g_softShowHeatmap);
        ImGui::SameLine();
        ImGui::Checkbox("show groups", &g_softShowGroups);
        if (!g_softPartReady) ImGui::EndDisabled();

        // --- group radii readout ---
        if (g_softPartReady && !g_softGroupRadii.empty()) {
            ImGui::Spacing();
            ImGui::Text("group radii (world units):");
            for (size_t i = 0; i < g_softGroupRadii.size(); ++i) {
                glm::vec3 c = SoftPartition::paletteColor((int)i);
                if (i < (size_t)SoftPartition::MAX_GROUPS) {
                    ImGui::Checkbox(("##gv" + std::to_string(i)).c_str(),
                                    &g_softGroupVisible[i]);   // toggle this group's heatmap points
                    ImGui::SameLine();
                }
                ImGui::ColorButton(("##rad" + std::to_string(i)).c_str(),
                                   ImVec4(c.r, c.g, c.b, 1.0f),
                                   ImGuiColorEditFlags_NoTooltip |
                                   ImGuiColorEditFlags_NoPicker,
                                   ImVec2(14, 14));
                ImGui::SameLine();
                ImGui::Text("G%d : %.4f", (int)i, g_softGroupRadii[i]);
            }
            // "none" row: source points not belonging to any group (none-mass
            // > 0.5). Uncheck to hide them; the swatch is the neutral none grey.
            ImGui::Checkbox("##gvnone", &g_softShowNone);
            ImGui::SameLine();
            ImGui::ColorButton("##radnone",
                               ImVec4(0.35f, 0.35f, 0.38f, 1.0f),
                               ImGuiColorEditFlags_NoTooltip |
                               ImGuiColorEditFlags_NoPicker,
                               ImVec2(14, 14));
            ImGui::SameLine();
            ImGui::Text("none (no group)");
        }

        // --- Step 4: Soft-weighted ICP (Ctrl+U) ---
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f),
                           "Step 4: Soft-weighted ICP  [Ctrl+U]");
        if (!g_softPartReady) ImGui::BeginDisabled();
        ImGui::SliderInt("ICP iters", &g_softIcpParams.maxIters, 1, 80);
        ImGui::SliderFloat("max corr (xDiag)",
                           &g_softIcpParams.maxCorrDistFac, 0.005f, 0.2f, "%.3f");
        ImGui::SliderFloat("none skip", &g_softIcpParams.noneSkip, 0.1f, 0.95f);
        ImGui::SliderFloat("min weight", &g_softIcpParams.minWeight, 0.0f, 0.3f, "%.3f");
        ImGui::Checkbox("uniform weight (ablation / plain ICP)",
                        &g_softIcpParams.uniformWeight);
        ImGui::Checkbox("downsample target (large-scale)",
                        &g_softIcpParams.downsampleTarget);
        if (g_softIcpParams.downsampleTarget)
            ImGui::SliderFloat("voxel (xDiag)",
                               &g_softIcpParams.voxelFac, 0.002f, 0.05f, "%.3f");
        if (ImGui::Button("4. Run Soft-weighted ICP")) {
            if (g_regLiveMode) {
                poseAutoSaveBeforeRegistration();
                startSoftIcpLive();          // animates; render loop saves on finish
            } else {
                runSoftWeightedICP();
                poseSaveToLibrary(SaveCriterion::RMSE);
            }
        }
        if (!g_softPartReady) ImGui::EndDisabled();
        ImGui::SameLine();
        // Capture the disabled state in a local: revertSoftICP() flips
        // g_softIcpCanRevert mid-frame, so testing the global twice would call
        // EndDisabled() without a matching BeginDisabled() (imgui assert).
        const bool icpRevDisabled = !g_softIcpCanRevert;
        if (icpRevDisabled) ImGui::BeginDisabled();
        if (ImGui::Button("Revert ICP")) { revertSoftICP(); poseSyncSessionBestToCurrent(); }
        if (icpRevDisabled) ImGui::EndDisabled();
        if (g_softIcpLastIters > 0) {
            ImGui::Text("  iters = %d", g_softIcpLastIters);
            ImGui::Text("  RMSE  %.5f -> %.5f",
                        g_softIcpRmseBefore, g_softIcpRmseAfter);
            ImGui::Text("  IoU   %.4f -> %.4f",
                        g_softIcpIoUBefore, g_softIcpIoUAfter);
        }

        // --- Phase U-CPD: method selector + pure rigid CPD (anchor-free) ---
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f),
                           "Registration method  (Ctrl+U; both from Umeyama init)");
        ImGui::RadioButton("Soft-weighted ICP (U-2, uses anchor)", &g_regMethod, METHOD_SOFT_ICP);
        ImGui::RadioButton("Pure rigid CPD (no anchor)",           &g_regMethod, METHOD_CPD);

        if (g_regMethod == METHOD_CPD) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f), "[CPD] parameters");
            ImGui::SliderInt  ("CPD max iters",      &g_cpdParams.maxIters,   1, 150);
            ImGui::SliderFloat("CPD outlier w",      &g_cpdParams.w_outlier,  0.0f, 0.5f, "%.3f");
            ImGui::Checkbox   ("CPD solve scale",    &g_cpdParams.solveScale);
            ImGui::Checkbox   ("CPD downsample tgt", &g_cpdParams.downsample);
            if (g_cpdParams.downsample)
                ImGui::SliderFloat("CPD tgt voxel (xDiag)", &g_cpdParams.tgtVoxelFac, 0.005f, 0.05f, "%.3f");
            ImGui::SliderFloat("CPD src voxel (xDiag, 0=off)", &g_cpdParams.srcVoxelFac, 0.0f, 0.05f, "%.3f");
            // Robustness (stops the sigma^2 -> 0 collapse on a too-good init):
            ImGui::SliderFloat("CPD sigma2 floor (xSpacing, 0=off)", &g_cpdParams.sigmaFloorFac, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("CPD N_P collapse frac (0=off)",      &g_cpdParams.npCollapseFrac, 0.0f, 0.9f, "%.2f");

            // [LIVE] Shared by CPD and soft-ICP (mirrors Shift+N's "Live mode").
            ImGui::Separator();
            ImGui::Checkbox("Live animate (CPD + soft-ICP)", &g_regLiveMode);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "ON (default): Ctrl+U / 'Run all CPD' / 'Run Soft-weighted ICP'\n"
                    "  replay the solve frame-by-frame and the mesh visibly\n"
                    "  converges, like the Shift+N tracker. Watch CPD (now\n"
                    "  floored) barely move on a good init while soft-ICP\n"
                    "  drifts off it.\n"
                    "OFF: blocking one-shot (mesh jumps to the final pose).");
            }
            if (g_regLiveMode)
                ImGui::SliderInt("Live steps/frame", &g_regLiveStepsPerFrame, 1, 10);

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.6f, 1.0f),
                               "CPD pipeline — check 1 -> 5 in order:");

            const char* cpdStageLabels[6] = {
                "",
                "1. Extract target cloud",
                "2. Downsample target",
                "3. Build source + before-metrics",
                "4. Run CPD (EM)",
                "5. Apply + after-metrics"
            };
            for (int i = 1; i <= 5; ++i) {
                const bool enabled = (i == 1) || g_cpdStageDone[i - 1];
                if (!enabled) ImGui::BeginDisabled();
                bool v = g_cpdStageDone[i];
                if (ImGui::Checkbox(cpdStageLabels[i], &v)) {
                    if (v && !g_cpdStageDone[i]) {
                        switch (i) {
                            case 1: cpdStage1_extractTarget(); break;
                            case 2: cpdStage2_downsample();    break;
                            case 3: cpdStage3_buildSource();   break;
                            case 4: cpdStage4_runEM();         break;
                            case 5: cpdStage5_apply();
                                    if (g_cpdStageDone[5]) {
                                        poseSaveToLibrary(SaveCriterion::RMSE);
                                        // session may have rejected & reverted it
                                        if (!g_cpdCanRevert) g_cpdStageDone[5] = false;
                                    }
                                    break;
                        }
                    } else if (!v && g_cpdStageDone[i]) {
                        cpdUncheckFrom(i);   // cascades: unchecks >= i (5 also reverts)
                    }
                }
                if (!enabled) ImGui::EndDisabled();

                // per-stage readout
                if (i == 1 && g_cpdStageDone[1])
                    ImGui::Text("      target: %d pts", g_cpdTgtCountRaw);
                if (i == 2 && g_cpdStageDone[2])
                    ImGui::Text("      -> %d pts (solve size)", g_cpdTgtCountDS);
                if (i == 3 && g_cpdStageDone[3]) {
                    ImGui::Text("      source: %d pts   sigma2_init: %.5g",
                                g_cpdSrcCount, g_cpdSigma2Init);
                    ImGui::Text("      RMSE before %.5f   IoU before %.4f",
                                g_cpdRmseBefore, g_cpdIoUBefore);
                }
                if (i == 4 && g_cpdStageDone[4])
                    ImGui::Text("      iters=%d  sigma2=%.5g (floor %.5g)  scale=%.4f  N_P=%.1f/%d",
                                g_cpdResult.iters, g_cpdResult.sigma2, g_cpdResult.sigma2Floor,
                                g_cpdResult.finalScale, g_cpdResult.N_P, g_cpdTgtCountDS);
                if (i == 5 && g_cpdStageDone[5]) {
                    ImGui::Text("      RMSE %.5f -> %.5f", g_cpdRmseBefore, g_cpdRmseAfter);
                    ImGui::Text("      IoU  %.4f -> %.4f", g_cpdIoUBefore,  g_cpdIoUAfter);
                }
            }

            ImGui::Spacing();
            if (ImGui::Button("Run all CPD (1 -> 5)")) {
                if (g_regLiveMode) {
                    poseAutoSaveBeforeRegistration();
                    startCpdLive();          // animates; render loop saves on finish
                } else {
                    poseAutoSaveBeforeRegistration();
                    runCpdRegistration();
                    poseSaveToLibrary(SaveCriterion::RMSE);
                    if (!g_cpdCanRevert) g_cpdStageDone[5] = false; // session reverted it
                }
            }
            ImGui::SameLine();
            const bool cpdRevDisabled = !g_cpdCanRevert;
            if (cpdRevDisabled) ImGui::BeginDisabled();
            if (ImGui::Button("Revert CPD")) { revertCpd(); poseSyncSessionBestToCurrent(); }
            if (cpdRevDisabled) ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Reset CPD pipeline")) cpdResetPipeline();
        }

        // --- Step 5: Soft BIPOP (Ctrl+Shift+U) — U-3, not yet ---
        ImGui::Spacing();
        ImGui::BeginDisabled();
        ImGui::Button("5. Soft BIPOP          (Ctrl+Shift+U)");
        ImGui::EndDisabled();
    };

    a.onResetRegistration = []() {
        AppMode previousMode = gApp.mode;
        std::cout << "[UI] Reset Registration" << std::endl;

        // 元コード line 3491-3513 通り: startNewSession を最初に
        poseStartNewSession();

        // Phase 1: trial シード/callIdx もリセット (再現性の "fresh trial")
        resetTrialSeed();

        // Umeyamaがアクティブなら閉じる
        if (gUmeyama.active) {
            gUmeyama.cancel(registrationHandle, OrbitCam);
        }

        // レジストレーションの変換のみリセット（ポイントは保持）
        registrationHandle.resetTransformOnly();
        gUI.state.regMethod = -1;

        // 初期ポーズを復元（元のコードと同じ）
        restoreInitialPose();

        // 初期回転プリセットをリセット (Phase 2: 幾何ベース BASE + Position CENTER)
        registrationHandle.initRotPreset = RegistrationData::PRESET_BASE;
        registrationHandle.initRotPosition = RegistrationData::POS_CENTER;
        gUI.state.initRotPreset = 0;
        gUI.state.initRotPosition = 0;
        // チャット 9: 4-quadrant mask も QUAD_ALL にリセット
        //   (= 旧 POS_CENTER と byte-identical な初期状態)
        g_activeQuadrantMask = LiverLeftRightLabel::QUAD_ALL;
        gUI.state.activeQuadrantMask = LiverLeftRightLabel::QUAD_ALL;
        g_currentOrientLabel    = "Base";   // Phase 2: 旧 "Front" → "Base"

        // クラスタ可視化をクリア
        g_cluster1Points.clear();
        g_cluster2Points.clear();
        g_targetPoints.clear();
        g_rejectedBoundaryPoints.clear();
        g_visibleSourcePoints.clear();
        g_silhouetteSourcePoints.clear();
        g_debugSourceRimChain.clear();           // Phase 7b Step 1 (Plain W)
        g_debugTargetBoundaryPoints.clear();     // Phase 7b Step 2 (Shift+W)
        g_debugShapeMatchBestSrc.clear();        // Phase 7b Step 3 (Ctrl+W)
        g_debugSourceRidge.clear();              // Phase 7c (REDGE/稜線)
        g_debugTargetRidgePoints.clear();        // Phase 7c (REDGE/稜線)
        g_showClusterVisualization = false;
        g_showBoundaryCandidates   = false;
        g_showSourceVisualization  = false;
        g_showDebugSourceRimChain  = false;      // Phase 7b Step 1 (Plain W)
        g_showDebugTargetBoundary  = false;      // Phase 7b Step 2 (Shift+W)
        g_showDebugShapeMatch      = false;      // Phase 7b Step 3 (Ctrl+W)
        g_showDebugSourceRidge     = false;      // Phase 7c (REDGE/稜線)
        g_showDebugTargetRidge     = false;      // Phase 7c (REDGE/稜線)
        g_showCorrespondencePoints = false;
        g_showCyclicCorrespondence = false;
        g_cyclicAvailable          = false;

        // Ctrl+G rim-pair viz buffers (V3R-W). Toggle stays as user
        // last set it; only the populated buffers are cleared.
        g_ctrlgRimSrcVertIdx.clear();
        g_ctrlgRimTgtPos.clear();
        g_ctrlgRimVizAvailable     = false;

        // Cranio-caudal viz buffers (Shift+H, Phase 1). Toggle stays as
        // user last set it; only the populated buffers are cleared
        // (LR/Region と同じ流儀。labels 自体はメッシュ再ロード時に
        //  recomputeLiverCC の size mismatch チェックで再計算される)。
        g_ccVizIdxCranial.clear();
        g_ccVizIdxCaudal.clear();

        // AppModeは変更しない！3Dシーンはそのまま表示される
        // UIのregPhaseActive_は呼び出し元（Back Depthボタン）で制御される

        std::cout << "[InitRot] Reset to Base" << std::endl;
        std::cout << "[ResetReg] Registration state and clusters cleared (mode unchanged: "
                  << static_cast<int>(previousMode) << ")" << std::endl;
    };

    a.onRigidMode = []() {
        std::cout << "[stub] onRigidMode" << std::endl;
    };

    a.onHandlePlaceMode = []() {
        std::cout << "[stub] onHandlePlaceMode" << std::endl;
    };

    a.onDeformMode = []() {
        std::cout << "[stub] onDeformMode" << std::endl;
    };

    a.onFullReset = []() {
        std::cout << "[stub] onFullReset" << std::endl;
    };

    a.onHandleRadiusChanged = [](float) {
        std::cout << "[stub] onHandleRadiusChanged" << std::endl;
    };

    a.onStartFromDepth = []() {
        // Registration状態をリセットしてDepthフェーズに戻る
        registrationHandle.reset();
        gUI.resetToDepthPhase();
        gApp.mode = AppMode::kEmpty;
        gApp.image.loaded = false;

        // 臓器の可視性をリセット（デフォルトalpha値に戻す）
        // liver と target だけ ON、それ以外は全 OFF
        const float defaultAlpha[8] = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f};
        for (int i = 0; i < 8; i++) {
            g_meshAlpha[i] = defaultAlpha[i];
        }

        std::cout << "[StartFromDepth] Back to Depth phase" << std::endl;
    };

    a.onSwitchToDeformMode = []() {
        // 1) DEFORM が読む reg_*.obj スナップショットを出力（同期。return 時点で
        //    ディスクに出ているので、子プロセス側の読み込みと競合しない）。
        std::cout << "[Reg->Deform] Exporting reg_*.obj..." << std::endl;
        StlExport::exportRegisteredObjs();

        // 2) lsn_deform を detached 起動。DEFORM_EXE_PATH は initPaths() が
        //    findExe で解決済み（bin/ の ./lsn_deform、env override は
        //    AAA_DEFORM_EXE_PATH）。素の std::system は使わない
        //    （Qt Creator 経由だと継承した stdout パイプの閉鎖で子が SIGPIPE 死する）。
        //    platform_launch_detached は nohup + stdio を /dev/null に切り離す対策込み。
        if (!std::filesystem::exists(DEFORM_EXE_PATH)) {
            std::cerr << "[Reg->Deform] lsn_deform not found: '" << DEFORM_EXE_PATH
                      << "'  -- build lsn_deform first (or set AAA_DEFORM_EXE_PATH)" << std::endl;
            return;
        }
        std::cout << "[Reg->Deform] Launching: " << DEFORM_EXE_PATH << std::endl;
        int rc = platform_launch_detached(DEFORM_EXE_PATH);
        std::cout << "[Reg->Deform] launch rc=" << rc << std::endl;
    };

    a.onRefine = []() {
        std::cout << "[stub] onRefine" << std::endl;
    };

    a.onSilhouetteAlign = []() {
        // 元コード line 2854 (runShiftE 内冒頭) と line 3073-3075
        // g_stepStartTime をリセットして各Shift+E呼出しごとの所要時間を記録
        // [QUAD-SIL] quadrant guard + mask pass-through (keyboard parity).
        std::cout << "[Alt+P/UI] dispatching Silhouette Align..." << std::endl;
        if (!prepareQuadrantForSilAlign("Alt+P/UI")) return;
        g_stepStartTime = std::chrono::steady_clock::now();
        poseAutoSaveBeforeRegistration();
        runShiftE(g_activeQuadrantMask);
        g_sessionSilhouetteN++;
        gUI.state.regMethod = 5;
        saveSilAlignToLibrary();   // [QUAD-SIL ACCEPT] Ctrl+I-parity gate
    };

    a.onPoseLibraryToggle = []() {
        g_poseLibrary.showWindow = !g_poseLibrary.showWindow;
        std::cout << "[PoseLibrary] Window "
                  << (g_poseLibrary.showWindow ? "ON" : "OFF") << std::endl;
    };

    a.onPoseUndo = []() {
        g_metricsValid = false;  // pose about to be restored; Phase A must remeasure
        poseUndo();
    };

    a.onAutoProbe = []() {
        g_callIdx = 0;  // 単体 AutoProbe: シード範囲 0..N-1
        runAutoProbe();
    };

    a.onIterativeAutoProbe = [](int K) {
        runIterativeAutoProbe(K);
    };

    // AutoQCR ボタン: チェック値 (lock_scale) を渡して 9 trial sweep を実行。
    //   - lock_scale=true  : 6-DoF rigid (scale=1 固定、論文推奨)
    //   - lock_scale=false : 7-DoF (scale 推定 ON、現状互換)
    a.onAutoQCR = [](bool lockScale) {
        runAutoQuadCyclicRansac(lockScale);
    };

    // --------------------------------------------------------------------
    //  onInitRotPresetSilent : POSITION 変更時の自動 Orient 同期
    // --------------------------------------------------------------------
    //  ユーザが POSITION の 2x2 grid checkbox / Quick Presets を操作したとき、
    //  RegistrationImGuiManager 側で mapQuadrantToOrientPreset によって
    //  対応する Orient (Right/Base/Left) が算出され、その値が ここに渡ってくる。
    //  registrationHandle.initRotPreset を更新するだけで、applyInitRotation
    //  は呼ばない (Apply Init Pose ボタン押下時にまとめて適用される)。
    a.onInitRotPresetSilent = [](int preset) {
        registrationHandle.initRotPreset =
            (RegistrationData::InitRotPreset)preset;
        std::cout << "[InitRot] Auto-set by quadrant: "
                  << RegistrationData::presetName(registrationHandle.initRotPreset)
                  << "  (silent; applyInitRotation not called -- press Apply Init Pose)"
                  << std::endl;
    };

    // --------------------------------------------------------------------
    //  onDrawAdvancedCtrlG : Advanced セクションへ RIM/Raycast パネル統合
    // --------------------------------------------------------------------
    //  REGISTRATION → Advanced CollapsingHeader を開いたとき、
    //  drawCtrlGRimRaycastControls がインラインで描画される。Floating
    //  "Ctrl+G Quadrant Selector" 窓と同じ g_ctrlg* globals を操作するので、
    //  両方の表示は自動的に同期する (ImGui ID stack は window 単位で独立)。
    a.onDrawAdvancedCtrlG = []() {
        drawCtrlGRimRaycastControls();
    };

    // ----------------------------------------------------------------
    //  onCtrlG : V3-R BIPOP-CMA-ES (Region-aware refinement)
    // ----------------------------------------------------------------
    //  REGISTRATION セクション内 "Ctrl+G  V3-R  [Refine]" ボタンの
    //  押下時に呼ばれる。キーボードの Ctrl+G dispatch
    //  (GLFW_KEY_G case, "isCtrlG" 分岐) と等価のシーケンスを実行する。
    //  両経路で同じ (trial_seed, callIdx) 状態から開始すれば結果は
    //  byte-identical になる。
    //
    //  失敗時 (ラベル未計算 / subset 空) は本体に到達せず early-return。
    //  UI ボタン側でも quadLabelsReady と activeQuadrantMask!=0 を
    //  チェックして disabled 化するが、二重ガードしておく。
    a.onCtrlG = []() {
        std::cout << "[Ctrl+G/UI] V3-R (region-aware) session start" << std::endl;
        const auto maskStr = LiverLeftRightLabel::quadrantMaskString(
            g_activeQuadrantMask);
        std::cout << "[Ctrl+G/UI] quadrant_mask = " << maskStr
                  << "  (0x" << std::hex << (unsigned)g_activeQuadrantMask
                  << std::dec << ")" << std::endl;

        // [UI整理] Auto-compute labels on demand (no HemiAuto needed) so the
        // button works from the default state / after Apply Init Pose / drag.
        if (!g_liverRegion.valid()) recomputeLiverRegion();
        if (!g_liverLR.valid())     recomputeLiverLR();
        if (!g_liverRegion.valid() || !g_liverLR.valid()) {
            std::cerr << "[Ctrl+G/UI] ERROR: label computation failed"
                      << " (Region.valid=" << (g_liverRegion.valid() ? "Y" : "N")
                      << ", LR.valid="    << (g_liverLR.valid() ? "Y" : "N")
                      << "). Is the liver mesh loaded?"
                      << std::endl;
            return;
        }
        auto subset = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels, g_liverLR.labels,
            g_activeQuadrantMask);
        std::cout << "[Ctrl+G/UI] subset_size = " << subset.size()
                  << " / " << g_liverRegion.labels.size()
                  << " vertices (original-index space)";
        if (g_activeQuadrantMask == LiverLeftRightLabel::QUAD_ALL) {
            std::cout << "  (QUAD_ALL: byte-identical to V3 expected)";
        }
        std::cout << std::endl;

        if (subset.empty()) {
            std::cerr << "[Ctrl+G/UI] ERROR: subset is empty for mask=0x"
                      << std::hex << (unsigned)g_activeQuadrantMask
                      << std::dec
                      << ". Select at least one quadrant in the POSITION panel."
                      << std::endl;
            return;
        }

        g_stepStartTime = std::chrono::steady_clock::now();
        g_sessionBipopN++;
        gUI.state.regMethod = 3;   // BIPOP method (same slot as Shift+V/F/G)
        poseAutoSaveBeforeRegistration();
        runBipopCmaesV3R(g_activeQuadrantMask);
        poseSaveToLibrary(SaveCriterion::RMSE, g_activeQuadrantMask);
    };

    // ----------------------------------------------------------------
    //  onCtrlI : V3I pure squash-IoU (silhouette align via V3RS engine)
    // ----------------------------------------------------------------
    //  Mirrors the keyboard Ctrl+I dispatch (GLFW_KEY_I case). Pairs with
    //  the "Silhouette (Ctrl+I)" Refine button. Parallel runs are ON by
    //  default (g_v3rsParallelRuns); toggle in Tuning & Advanced.
    a.onCtrlI = []() {
        std::cout << "[Ctrl+I/UI] V3I pure squash-IoU session start" << std::endl;
        // [UI整理] Auto-compute labels on demand (no HemiAuto needed).
        if (!g_liverRegion.valid()) recomputeLiverRegion();
        if (!g_liverLR.valid())     recomputeLiverLR();
        if (!g_liverRegion.valid() || !g_liverLR.valid()) {
            std::cerr << "[Ctrl+I/UI] ERROR: label computation failed "
                         "(is the liver mesh loaded?)." << std::endl;
            return;
        }
        auto subsetI = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels, g_liverLR.labels, g_activeQuadrantMask);
        if (subsetI.empty()) {
            std::cerr << "[Ctrl+I/UI] ERROR: subset is empty." << std::endl;
            return;
        }
        g_stepStartTime = std::chrono::steady_clock::now();
        g_sessionBipopN++;
        gUI.state.regMethod = 3;
        poseAutoSaveBeforeRegistration();
        runBipopCmaesV3I(g_activeQuadrantMask);
        poseSaveToLibrary(SaveCriterion::EITHER, g_activeQuadrantMask);
    };

    // ----------------------------------------------------------------
    //  onShiftI : silhouette-fixed dilation RMSE refine (depth post-step)
    // ----------------------------------------------------------------
    //  Mirrors the keyboard Shift+I dispatch. Pairs with the "Depth
    //  Dilation (Shift+I)" Refine button. Save by RMSE: IoU is invariant
    //  under the dilation, so an IoU gate would reject every result.
    a.onShiftI = []() {
        std::cout << "[Shift+I/UI] Dilation RMSE refine" << std::endl;
        if (!registrationHandle.useRegistration) {
            std::cerr << "[Shift+I/UI] Run HemiAuto (O) + Alt+P / Ctrl+I first."
                      << std::endl;
            return;
        }
        g_stepStartTime = std::chrono::steady_clock::now();
        poseAutoSaveBeforeRegistration();
        runDilationRmseRefine();
        gUI.state.regMethod = 3;
        poseSaveToLibrary(SaveCriterion::RMSE, g_activeQuadrantMask);
    };

    // ----------------------------------------------------------------
    //  onCtrlgLockScaleChanged : サイドバー Ctrl+G 横の 6-DoF checkbox
    // ----------------------------------------------------------------
    //  チェック ON  → g_ctrlgSearchMode = SIX_DOF_RIGID
    //  チェック OFF → g_ctrlgSearchMode = SEVEN_DOF
    //  4-DoF (FOUR_DOF_XYRXRY) への切替は左 floating "Ctrl+G Quadrant
    //  Selector" パネルの radio button 経由でのみ可能 (3 択 → 2 択に潰す
    //  ことで主動線をシンプルに保つ)。floating パネル側で 4-DoF を選んで
    //  いる状態でこの checkbox を触ると、6-DoF または 7-DoF に上書きされる。
    a.onCtrlgLockScaleChanged = [](bool lock) {
        g_ctrlgSearchMode = lock
            ? CmaesRefineV3R::SearchMode::SIX_DOF_RIGID
            : CmaesRefineV3R::SearchMode::SEVEN_DOF;
        std::cout << "[Ctrl+G/UI] SearchMode → "
                  << (lock ? "SIX_DOF_RIGID (scale=1)"
                           : "SEVEN_DOF (T+R+Scale)")
                  << std::endl;
    };

    a.onInitRotPresetChanged = [](int preset) {
        // Phase 2: 幾何ベース Initial Orientation (preset = 9 通りの回転)。
        //   - 患者ごとに肝臓自身の解剖軸を基準にするので、同じプリセットが
        //     患者間で同じ解剖学的意味を持つ。
        //   - 重心位置 (initRotPosition) は別の selector で独立に管理。
        //     9 × 3 = 27 通りの初期姿勢が生成可能。
        //   - 実際の回転・平行移動の適用は applyInitRotation() に集約。
        registrationHandle.initRotPreset = (RegistrationData::InitRotPreset)preset;
        std::cout << "[InitRot] Preset selected: "
                  << RegistrationData::presetName(registrationHandle.initRotPreset)
                  << std::endl;
        applyInitRotation(/*startNewSession=*/true);
    };

    a.onInitRotPositionChanged = [](int position) {
        // Phase 2 拡張 (legacy): 重心の画面配置 selector (Right / Center / Left)。
        //   UI ボタンは削除済みだが、API 互換のため残置。チャット 9 以降は
        //   onQuadrantMaskChanged + onApplyInitPose 経由が標準。
        registrationHandle.initRotPosition = (RegistrationData::InitRotPosition)position;
        std::cout << "[InitRot] (legacy) Position selected: "
                  << RegistrationData::positionName(registrationHandle.initRotPosition)
                  << std::endl;
        applyInitRotation(/*startNewSession=*/false);
    };

    // -------------------------------------------------------------------
    //  チャット 9: 4-quadrant 連動 callback
    //  ------------------------------------------------------------------
    //  onQuadrantMaskChanged:
    //    Initial Orientation panel の 2x2 grid checkbox or Quick Preset 押下時、
    //    または Ctrl+G panel の checkbox 押下時に呼ばれる。
    //    main.cpp 側で g_activeQuadrantMask を更新するのみ — 副作用なし。
    //    実際の姿勢適用は onApplyInitPose 経由で明示的に行う (判断 A-2)。
    //
    //  ※ Ctrl+G panel 側 checkbox はこの callback を経由せず g_activeQuadrantMask
    //    を直接書き換える既存実装のまま。これは Ctrl+G 操作中に initial pose を
    //    勝手に動かさないため (subset 調整時の意図しない姿勢変化を防止)。
    //    どちらの panel から変えても g_activeQuadrantMask は同期する。
    a.onQuadrantMaskChanged = [](uint8_t mask) {
        g_activeQuadrantMask = mask;
        std::cout << "[InitRot] Quadrant mask changed: "
                  << LiverLeftRightLabel::quadrantMaskString(mask)
                  << " (0x" << std::hex << (unsigned)mask << std::dec << ")"
                  << "  -- press 'Apply Init Pose' to apply, or 'Ctrl+G' to run V3-R"
                  << std::endl;
    };

    //  onApplyInitPose:
    //    Initial Orientation panel の Apply Init Pose ボタン押下時に呼ばれる。
    //    現在の preset + g_activeQuadrantMask で applyInitRotation を実行し、
    //    新規 PoseLibrary session を開始する (preset 変更時と同じ扱い)。
    a.onApplyInitPose = []() {
        std::cout << "[InitRot] Apply Init Pose button pressed: preset="
                  << RegistrationData::presetName(registrationHandle.initRotPreset)
                  << "  mask=" << LiverLeftRightLabel::quadrantMaskString(g_activeQuadrantMask)
                  << " (0x" << std::hex << (unsigned)g_activeQuadrantMask << std::dec << ")"
                  << std::endl;
        g_metricsValid = false;  // pose about to change; Phase A must remeasure
        applyInitRotation(/*startNewSession=*/true);
    };

    //  onFlipLR (Preview OBJ Anatomical Pose):
    //    LR axis の sign を反転する。**in-place** で d_lr を negate + ラベル配列を
    //    swap するだけで、PCA / eclipse の再計算は行わない。
    //
    //    なぜ recompute しないか:
    //      Apply Init Pose は mesh を回転させる。次に Flip を押したとき、
    //      labelVertices() を呼ぶと「変形後の mesh」で PCA を実行する → 固有
    //      ベクトルが変わる → 2 回 Flip しても元の d_lr に戻らない。
    //      この症状は実機で確認済 (Flip true → Apply → Flip false で sign が
    //      非対称になる)。
    //    In-place 実装の対称性:
    //      g_lrFlipManual を toggle するのみ + g_liverLR の中身を sign-反転する。
    //      PCA 結果 (eclipse.a_vis_pos 等) は保持されるので、2 回押せば必ず
    //      元の状態に戻る。Apply Init Pose が間に何回入っても無関係。
    //    CC への影響:
    //      CC は LR が選んだ PCA 軸 (idx) ではなく「残りの軸」を使う。LR の
    //      sign 反転は CC の d_cc には何ら影響しない (LiverCranioCaudalLabel
    //      の実装を参照)。したがって CC は触らない。
    //    Bootstrap:
    //      g_liverLR がまだ未計算なら通常の recompute 経路に fallback する。
    //      最初の 1 回だけは PCA を回す必要があるため。
    a.onFlipLR = []() {
        g_lrFlipManual = !g_lrFlipManual;
        if (!g_liverLR.valid()) {
            std::cout << "[Axes] Flip LR -> "
                      << (g_lrFlipManual ? "true" : "false")
                      << "  (bootstrap: full recompute)" << std::endl;
            recomputeLiverLR();
            if (g_liverCC.valid()) recomputeLiverCC();
            return;
        }
        std::cout << "[Axes] Flip LR -> "
                  << (g_lrFlipManual ? "true" : "false")
                  << "  (in-place: PCA preserved -> 2x flip = identity)"
                  << std::endl;
        // d_lr 反転
        g_liverLR.d_lr = -g_liverLR.d_lr;
        g_liverLR.eclipse.flipped_manual = g_lrFlipManual;
        // ラベル swap (PURE_R ↔ PURE_L、BOUNDARY はそのまま)
        for (auto& lbl : g_liverLR.labels) {
            if (lbl == LiverLeftRightLabel::PURE_RIGHT)
                lbl = LiverLeftRightLabel::PURE_LEFT;
            else if (lbl == LiverLeftRightLabel::PURE_LEFT)
                lbl = LiverLeftRightLabel::PURE_RIGHT;
        }
        std::swap(g_liverLR.n_pure_right, g_liverLR.n_pure_left);
        // 診断情報も swap (UI confidence は |Δ|/avg で対称式なので不変)
        std::swap(g_liverLR.eclipse.n_vis_pos, g_liverLR.eclipse.n_vis_neg);
        std::swap(g_liverLR.eclipse.a_vis_pos, g_liverLR.eclipse.a_vis_neg);
        g_liverLR.eclipse.lean_area = -g_liverLR.eclipse.lean_area;
        // Y キーの球マーカー (viz subsample) を新しいラベル割り当てに合わせて再構築
        g_lrVizIdxR = LiverLeftRightLabel::sampleVertexIndices(
            g_liverLR.labels, LiverLeftRightLabel::PURE_RIGHT, 1500);
        g_lrVizIdxL = LiverLeftRightLabel::sampleVertexIndices(
            g_liverLR.labels, LiverLeftRightLabel::PURE_LEFT, 1200);
    };

    //  onFlipCC (Preview OBJ Anatomical Pose):
    //    CC axis の sign を反転する。onFlipLR と同じ in-place 戦略。
    //    LR axis には影響しない (CC は LR の axis idx を借りるが、d_cc は
    //    独立に sign 決定されているため)。
    a.onFlipCC = []() {
        g_ccFlipManual = !g_ccFlipManual;
        if (!g_liverCC.valid()) {
            std::cout << "[Axes] Flip CC -> "
                      << (g_ccFlipManual ? "true" : "false")
                      << "  (bootstrap: full recompute)" << std::endl;
            recomputeLiverCC();
            return;
        }
        std::cout << "[Axes] Flip CC -> "
                  << (g_ccFlipManual ? "true" : "false")
                  << "  (in-place: PCA preserved -> 2x flip = identity)"
                  << std::endl;
        // d_cc 反転
        g_liverCC.d_cc = -g_liverCC.d_cc;
        g_liverCC.cc.flipped_manual = g_ccFlipManual;
        // ラベル swap (CRANIAL ↔ CAUDAL)
        for (auto& lbl : g_liverCC.labels) {
            lbl = (lbl == LiverCranioCaudalLabel::CRANIAL)
                ? (uint8_t)LiverCranioCaudalLabel::CAUDAL
                : (uint8_t)LiverCranioCaudalLabel::CRANIAL;
        }
        std::swap(g_liverCC.n_cranial, g_liverCC.n_caudal);
        // 診断情報も swap (confidence は |Δ|/sum で対称式なので不変)
        std::swap(g_liverCC.cc.mean_plus, g_liverCC.cc.mean_minus);
        std::swap(g_liverCC.cc.area_plus, g_liverCC.cc.area_minus);
        std::swap(g_liverCC.cc.n_rim_plus, g_liverCC.cc.n_rim_minus);
        // Shift+H の球マーカー (viz subsample) も更新
        g_ccVizIdxCranial = LiverCranioCaudalLabel::sampleVertexIndices(
            g_liverCC.labels, LiverCranioCaudalLabel::CRANIAL, 1500);
        g_ccVizIdxCaudal = LiverCranioCaudalLabel::sampleVertexIndices(
            g_liverCC.labels, LiverCranioCaudalLabel::CAUDAL, 1500);
    };
}

// === 関数定義 ===

// =========================================================
//  applyInitRotation (Phase 2 拡張):
//    現在の registrationHandle.initRotPreset と .initRotPosition に基づき、
//    初期ポーズから幾何ベース回転 + 重心オフセットを適用する。
//    onInitRotPresetChanged と onInitRotPositionChanged の両方から呼ばれる。
//
//    startNewSession=true : poseStartNewSession() を実行 (preset 変更時)
//    startNewSession=false: 既存セッション内での再適用 (position 変更時)
//
//    依存ラベル (LiverRegion / LiverLR / LiverCC) が未計算なら auto-trigger
//    する (Quad と同じ流儀)。失敗時は identity (回転スキップ) で安全側。
// =========================================================
static void applyInitRotation(bool startNewSession, bool liver_only) {
    if (startNewSession) {
        poseStartNewSession();
    }

    // Pose Library label を更新: preset 名 + quadrant mask 名 (QUAD_ALL 以外のみ)
    //   - 旧: "Base @ Right" / "Base @ Left" / "Base"
    //   - 新: "Base @ Q:AR+PR" / "Base @ Q:AL+PL" / "Base"
    //   QUAD_ALL は省略 (旧 POS_CENTER 相当、byte-identical の意味で "Base" のまま)。
    //   旧 PoseLibrary エントリの "Base @ Right" 等は読み込み時にそのまま文字列保持。
    g_currentOrientLabel = RegistrationData::presetName(registrationHandle.initRotPreset);
    if (g_activeQuadrantMask != LiverLeftRightLabel::QUAD_ALL) {
        g_currentOrientLabel += " @ ";
        g_currentOrientLabel +=
            LiverLeftRightLabel::quadrantMaskString(g_activeQuadrantMask);
    }
    if (startNewSession) {
        std::cout << "[Session] New session: " << g_currentOrientLabel
                  << "  (mask=0x" << std::hex << (unsigned)g_activeQuadrantMask
                  << std::dec << ")" << std::endl;
    } else {
        std::cout << "[InitRot] re-applying within session: " << g_currentOrientLabel
                  << "  (mask=0x" << std::hex << (unsigned)g_activeQuadrantMask
                  << std::dec << ")" << std::endl;
    }

    // 初期ポーズ復元
    auto organs = getOrganList();
    if (g_initOrganVertices.empty() ||
        g_initOrganVertices.size() != organs.size()) {
        std::cerr << "[InitRot] g_initOrganVertices not snapshotted yet" << std::endl;
        return;
    }
    for (size_t i = 0; i < organs.size(); i++) {
        // AutoQCR loop からの呼び出し (liver_only=true) では non-liver organ を
        // 一切触らない。loop 中 non-liver は g_initOrganVertices の状態のまま
        // 据え置かれ、winner 確定後の full-organ replay で 1 回だけ復元 + 変換される。
        if (liver_only && organs[i] != liverMesh3D) continue;
        organs[i]->mVertices = g_initOrganVertices[i];
        organs[i]->mNormals  = g_initOrganNormals[i];
        setUp(*organs[i]);
    }

    if (!liverMesh3D || liverMesh3D->mVertices.empty()) {
        std::cerr << "[InitRot] liverMesh3D unavailable; skipping rotation."
                  << std::endl;
        return;
    }

    // d_LR / d_CC が未計算なら auto-trigger (Quad と同じ流儀)。
    if (!g_liverCC.valid()) {
        std::cout << "[InitRot] LiverCC (Shift+H) not yet computed, auto-running..."
                  << std::endl;
        recomputeLiverCC();
    }
    if (!g_liverLR.valid() || !g_liverCC.valid()) {
        std::cerr << "[InitRot] Cannot apply: "
                  << "LR.valid=" << (g_liverLR.valid() ? "Y" : "N")
                  << "  CC.valid=" << (g_liverCC.valid() ? "Y" : "N")
                  << "  -> identity (no rotation applied)." << std::endl;
        return;
    }

    // Step 1+: mask != QUAD_ALL のときは Region/LR ラベルが必須
    //   (動的 subset AABB を計算するため)。未計算なら auto-trigger。
    //   mask == QUAD_ALL は g_sourceLiverAabbFull 直行なのでラベル不要。
    if (g_activeQuadrantMask != LiverLeftRightLabel::QUAD_ALL) {
        if (!g_liverRegion.valid()) {
            std::cout << "[InitRot] LiverRegion (Shift+R) not yet computed, auto-running..."
                      << std::endl;
            recomputeLiverRegion();
        }
        // g_liverLR は既に valid (上で確認済み) なので再確認不要
        if (!g_liverRegion.valid()) {
            std::cerr << "[InitRot] LiverRegion still not valid; falling back to "
                      << "QUAD_ALL behavior (full-mesh AABB)." << std::endl;
            // Continue with QUAD_ALL-equivalent: srcSubset = g_sourceLiverAabbFull,
            // scale=1, t_pos=0 (handled below by the "QUAD_ALL or invalid" branch)
        }
    }

    // 回転 + mask に応じた scale + 平行移動の適用 (Step 2: mask-based selection)。
    //   target cloud は target_full に固定。source 側のみ mask が示す 4-quadrant
    //   subset の AABB を使う:
    //     mask = QUAD_ALL    → 全体 ↔ 全体 (scale=1, t_pos=0、prealign 済)
    //                          = 旧 POS_CENTER と byte-identical
    //     mask = AR+PR (0x05) → 右葉 (= 旧 POS_RIGHT 相当だが解剖学的に厳密)
    //     mask = AL+PL (0x0A) → 左葉 (= 旧 POS_LEFT 相当)
    //     mask = AR+AL (0x03) → 前面 (新規)
    //     mask = PR+PL (0x0C) → 後面 (新規)
    //     mask = AR (0x01) のみ → 右前 4 象限 (新規)
    //     ... etc
    //
    //   利点 (前バージョンとの差):
    //   - 解剖学的 4 象限ラベル (Ctrl+G と同じ) を流用するので「右葉」が
    //     world X 軸ではなく d_LR PCA 軸ベースで定義される
    //   - Ctrl+G の g_activeQuadrantMask と完全共有 → checkbox を変えるだけで
    //     初期姿勢と CMA-ES subset の両方が同時に変わる
    //
    //   下流影響: Shift+M は g_originalLiverDiagMm / current_liver_diag で
    //   復元するので scale が変わっても自動対応。他の登録系も影響なし。
    glm::vec3 centroid = computeMeshCentroidFromVertices(liverMesh3D->mVertices);

    // mask から動的に source subset AABB を計算 (QUAD_ALL は高速パスで full を返す)
    const bool   use_full = (g_activeQuadrantMask == LiverLeftRightLabel::QUAD_ALL)
                          || !g_liverRegion.valid();   // fallback safety
    SourceSubsetAabb subset_storage;
    const SourceSubsetAabb* srcSubset = nullptr;
    if (use_full) {
        srcSubset = &g_sourceLiverAabbFull;
    } else {
        subset_storage = computeSourceLiverSubsetAabbFromMask(
            g_activeQuadrantMask,
            g_liverRegion.labels,
            g_liverLR.labels);
        srcSubset = &subset_storage;

        // Safety fallback: subset が極端に小さい場合は QUAD_ALL 動作にフォールバック。
        //   閾値:
        //     - subset.diag < 1e-6f (実質ゼロサイズ)
        //     - subset.valid == false (頂点 0 個)
        //   将来的に「subset 頂点数 < 50 で fallback」を追加する場所もここ。
        if (!srcSubset->valid || srcSubset->diag < 1e-6f) {
            std::cerr << "[InitRot] subset AABB invalid for mask=0x"
                      << std::hex << (unsigned)g_activeQuadrantMask << std::dec
                      << " (valid=" << srcSubset->valid
                      << " diag=" << srcSubset->diag
                      << "); falling back to full-mesh AABB." << std::endl;
            srcSubset = &g_sourceLiverAabbFull;
        }
    }

    // Scale factor + translation 計算 (チャット 10 v2: 実頂点回転 → AABB):
    //   - 8 コーナー回転方式は誤り (元 AABB の外接 box = looser bound、実頂点 AABB と
    //     center が異なるため、PostApply で error が残る)。
    //   - 正しくは: subset 頂点 (初期姿勢) を pivot 回転 → 新 AABB → その center を
    //     target_full.center に合わせる平行移動を計算する。
    //   - 頂点数は subset で 2k-10k 程度、毎 Apply で数 ms。
    //   - QUAD_ALL: scale=1 (prealign 済み)、subset: scale = target.diag / subset.diag。
    //
    //   定式:
    //     final point: p' = s * R*(p - centroid) + centroid + t_pos
    //     rotated_aabb_center は回転後の頂点群から AABB を取り直して 0.5*(min+max)
    //     scale を centroid 中心で適用: q = s*(rotated_aabb_center - centroid) + centroid
    //     t_pos = target_full.center - q
    //
    //   なぜ mean ではなく AABB center を使うか:
    //     target は depth cloud で前面のみ頂点があり mean が偏る。AABB center は
    //     形状サイズベースで両者一貫している (ユーザー指示: チャット 10)。
    //
    //   下流影響: Shift+M は g_originalLiverDiagMm / current_liver_diag で
    //   復元するので scale が変わっても自動対応 (相似変換)。
    float     scale_factor = 1.0f;
    glm::vec3 t_pos(0.0f);

    const bool is_quad_all_or_fallback =
        (srcSubset == &g_sourceLiverAabbFull);   // = QUAD_ALL or fallback

    // 回転後 AABB (デバッグ表示 + g_debugBB 描画用に関数スコープに昇格)
    glm::vec3 rotated_aabb_center(0.0f);
    glm::vec3 rotated_aabb_min   (0.0f);
    glm::vec3 rotated_aabb_max   (0.0f);
    int       rotated_n_used = 0;

    // subset 頂点インデックス (subset の場合のみ使用)
    std::vector<int> subset_idx_for_rot;
    if (!is_quad_all_or_fallback && g_liverRegion.valid()) {
        subset_idx_for_rot = LiverLeftRightLabel::makeQuadrantSubsetIdx(
            g_liverRegion.labels, g_liverLR.labels,
            g_activeQuadrantMask);
    }

    if (srcSubset->valid
        && g_targetAabbFull.valid
        && srcSubset->diag > 1e-9f
        && !g_initOrganVertices.empty()
        && !g_initOrganVertices[0].empty())
    {
        // scale 決定
        if (is_quad_all_or_fallback) {
            scale_factor = 1.0f;
        } else {
            scale_factor = g_targetAabbFull.diag / srcSubset->diag;
        }

        // 回転 R を identity scale + zero translation で取得
        glm::mat4 M_rot = getPresetRotation(
            registrationHandle.initRotPreset,
            centroid,
            g_liverLR.d_lr, g_liverCC.d_cc,
            1.0f, glm::vec3(0.0f));

        // ★ 実頂点を pivot 回転 → 新 AABB を計算
        //   初期姿勢の頂点 (g_initOrganVertices[0]) を使う (snapshotInitialPose で
        //   prealign 直後の状態が保存されている。これが srcSubset の参照系と一致する)
        const auto& V0 = g_initOrganVertices[0];
        const int   nV = (int)(V0.size() / 3);
        glm::vec3 new_mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
        glm::vec3 new_mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        int n_used = 0;

        if (is_quad_all_or_fallback) {
            // QUAD_ALL: 全頂点を回す
            for (int i = 0; i < nV; i++) {
                glm::vec3 p0(V0[i*3], V0[i*3+1], V0[i*3+2]);
                glm::vec3 q  = glm::vec3(M_rot * glm::vec4(p0, 1.0f));
                new_mn = glm::min(new_mn, q);
                new_mx = glm::max(new_mx, q);
                n_used++;
            }
        } else {
            // subset: 該当 idx のみ
            for (int idx : subset_idx_for_rot) {
                if (idx < 0 || idx >= nV) continue;
                glm::vec3 p0(V0[idx*3], V0[idx*3+1], V0[idx*3+2]);
                glm::vec3 q  = glm::vec3(M_rot * glm::vec4(p0, 1.0f));
                new_mn = glm::min(new_mn, q);
                new_mx = glm::max(new_mx, q);
                n_used++;
            }
        }

        if (n_used > 0) {
            rotated_aabb_min    = new_mn;
            rotated_aabb_max    = new_mx;
            rotated_aabb_center = 0.5f * (new_mn + new_mx);
            rotated_n_used      = n_used;

            // scale を centroid 中心で適用: s*(p - c) + c
            glm::vec3 src_rotated_scaled =
                (rotated_aabb_center - centroid) * scale_factor + centroid;
            // target_full center に合わせる translation
            t_pos = g_targetAabbFull.center - src_rotated_scaled;
        } else {
            std::cerr << "[InitRot] rotated AABB: no vertices used (mask=0x"
                      << std::hex << (unsigned)g_activeQuadrantMask << std::dec
                      << "); falling back to t_pos=0" << std::endl;
        }
    } else {
        std::cerr << "[InitRot] AABB invalid (src.valid="
                  << srcSubset->valid
                  << " tgt_full.valid=" << g_targetAabbFull.valid
                  << "); falling back to scale=1, no translation." << std::endl;
    }

    // Pre-apply ログ
    std::cout << "[InitRot/PreApply] BEFORE transform:" << std::endl;
    std::cout << "    centroid (rotation pivot) = ("
              << centroid.x << "," << centroid.y << "," << centroid.z << ")"
              << std::endl;
    if (srcSubset) {
        std::cout << "    src_subset.center (orig AABB midpoint) = ("
                  << srcSubset->center.x << "," << srcSubset->center.y << ","
                  << srcSubset->center.z << ")  diag=" << srcSubset->diag
                  << std::endl;
        std::cout << "    rotated AABB (real vertices, n=" << rotated_n_used << "): min=("
                  << rotated_aabb_min.x << "," << rotated_aabb_min.y << ","
                  << rotated_aabb_min.z << ")  max=("
                  << rotated_aabb_max.x << "," << rotated_aabb_max.y << ","
                  << rotated_aabb_max.z << ")  center=("
                  << rotated_aabb_center.x << "," << rotated_aabb_center.y << ","
                  << rotated_aabb_center.z << ")"
                  << std::endl;
    }
    std::cout << "    target_full.center (固定参照点 = AABB midpoint) = ("
              << g_targetAabbFull.center.x << "," << g_targetAabbFull.center.y << ","
              << g_targetAabbFull.center.z << ")  diag=" << g_targetAabbFull.diag
              << std::endl;
    std::cout << "    computed: scale=" << scale_factor
              << "  t_pos=(" << t_pos.x << "," << t_pos.y << "," << t_pos.z << ")"
              << std::endl;

    glm::mat4 R = getPresetRotation(
        registrationHandle.initRotPreset,
        centroid,
        g_liverLR.d_lr, g_liverCC.d_cc,
        scale_factor, t_pos);

    for (auto* m : organs) {
        if (liver_only && m != liverMesh3D) continue;
        applyMatrixToMeshVerticesAndNormals(m, R);
        setUp(*m);
    }

    std::cout << "[InitRot] applied: preset="
              << RegistrationData::presetName(registrationHandle.initRotPreset)
              << "  mask=" << LiverLeftRightLabel::quadrantMaskString(g_activeQuadrantMask)
              << " (0x" << std::hex << (unsigned)g_activeQuadrantMask << std::dec << ")"
              << "  scale=" << scale_factor
              << "  t_pos=(" << t_pos.x << "," << t_pos.y << "," << t_pos.z << ")"
              << "  d_lr=[" << g_liverLR.d_lr.x << "," << g_liverLR.d_lr.y << "," << g_liverLR.d_lr.z << "]"
              << "  d_cc=[" << g_liverCC.d_cc.x << "," << g_liverCC.d_cc.y << "," << g_liverCC.d_cc.z << "]"
              << std::endl;

    // ====================================================================
    //  PostApply 検証ログ (チャット 10 改訂):
    //  transform 適用後の liverMesh3D で実際の subset 頂点から AABB を再計算し、
    //  その center が target_full.center と一致するか確認する。
    //  一致していれば「画面上で source 重心が target 重心に重なっている」
    //  ことを意味する (AR カメラは world Z 方向 → world AABB = view AABB)。
    // ====================================================================
    if (liverMesh3D && !liverMesh3D->mVertices.empty()
        && srcSubset && srcSubset->valid) {
        const auto& Vnow = liverMesh3D->mVertices;
        const int nV = (int)(Vnow.size() / 3);

        // subset 頂点だけから AABB を再計算
        // - QUAD_ALL: 全頂点
        // - subset: makeQuadrantSubsetIdx で選ばれた頂点
        glm::vec3 post_mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
        glm::vec3 post_mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        int n_used = 0;

        if (is_quad_all_or_fallback || !g_liverRegion.valid()) {
            for (int i = 0; i < nV; i++) {
                glm::vec3 p(Vnow[i*3], Vnow[i*3+1], Vnow[i*3+2]);
                post_mn = glm::min(post_mn, p);
                post_mx = glm::max(post_mx, p);
                n_used++;
            }
        } else {
            auto subset_idx = LiverLeftRightLabel::makeQuadrantSubsetIdx(
                g_liverRegion.labels, g_liverLR.labels,
                g_activeQuadrantMask);
            for (int idx : subset_idx) {
                if (idx < 0 || idx >= nV) continue;
                glm::vec3 p(Vnow[idx*3], Vnow[idx*3+1], Vnow[idx*3+2]);
                post_mn = glm::min(post_mn, p);
                post_mx = glm::max(post_mx, p);
                n_used++;
            }
        }

        if (n_used > 0) {
            glm::vec3 post_center = 0.5f * (post_mn + post_mx);
            glm::vec3 err = post_center - g_targetAabbFull.center;
            float err_dist = glm::length(err);
            std::cout << "[InitRot/PostApply] VERIFICATION (AABB center):" << std::endl;
            std::cout << "    post-transform subset AABB center = ("
                      << post_center.x << "," << post_center.y << ","
                      << post_center.z << ")  (n_used=" << n_used << ")" << std::endl;
            std::cout << "    target_full.center                ("
                      << g_targetAabbFull.center.x << "," << g_targetAabbFull.center.y << ","
                      << g_targetAabbFull.center.z << ")" << std::endl;
            std::cout << "    error vector = (" << err.x << "," << err.y << "," << err.z << ")"
                      << "  |err|=" << err_dist
                      << (err_dist < 1e-4f ? "  [OK: subset AABB center == target center]"
                                           : "  [WARN: mismatch]")
                      << std::endl;

            // ★ デバッグ BB 可視化用に保存 (drawScene で球マーカー描画)
            g_dbgSourceBB_min    = post_mn;
            g_dbgSourceBB_max    = post_mx;
            g_dbgSourceBB_center = post_center;
            g_dbgSourceBB_valid  = true;
        }
    }
}

// =========================================================
//  runAutoQuadCyclicRansac (Alt+Ctrl+P)
// ---------------------------------------------------------
//  想定ワークフロー:
//      [Apply Init Pose] (preset + quadrant を確定)
//          ↓
//      [Alt+Ctrl+P]      ← この関数: ORIENT を 9 通り試行 → ベスト採用
//          ↓
//      [Ctrl+G]          (V3R refinement で詰める)
//
//  ユーザが Apply Init Pose で選んだ QUADRANT は loop 中も固定し、
//  ORIENT (preset) だけを QUADRANT の左右成分に応じて選ばれた 9 個に対して
//  試行し、unified compRmse が最小のものを採択する。
//
//  QUADRANT → 中心 preset グループ:
//    右側のみ (AR/PR/AR+PR=0x1/0x4/0x5)  → Right 中心 (画面左 3 列、FAR 含)
//    左側のみ (AL/PL/AL+PL=0x2/0x8/0xA)  → Left  中心 (画面右 3 列、FAR 含)
//    両側 (QUAD_ALL=0xF や混合)          → Base  中心 (画面中央 3 列)
//
//  UI レイアウト (Anatomical: Left=patient's right):
//     行1: Up-R+  Up-R   Up    Up-L   Up-L+
//     行2: Right+ Right  Base  Left   Left+
//     行3: Dn-R+  Dn-R   Down  Dn-L   Dn-L+
//
//  AutoProbe (PoseLibrary.h::runAutoProbe) と同じ「baseV/baseN から復元 →
//  姿勢変更 → 登録実行」のループ構造を借りるが、AutoProbe が全 probe を
//  PoseLibrary に投入するのに対し、こちらは loop 中は PoseLibrary に一切
//  触らず、最後にベスト pose だけ 1 件追加する。これにより:
//    - PoseLibrary の汚染がゼロ (9 件のゴミが残らない)
//    - Session reject 機構の介入もない (loop 終了後の 1 回だけ通る)
//    - 連打しても Library が荒れない
//
//  採択指標 (現状): registrationHandle.compRmse 最小 (low is better)。
//  top-5 は std::cout にダンプするのみで Library には入れない。
//
//  実行時間目安: 1 trial ~0.26s × 9 = 約 2.5s (実機ログから推定)。
//
//  lock_scale 引数 (Phase 2):
//      false : 7-DoF (T+R+Scale) — 現状互換、scale 推定 ON
//      true  : 6-DoF (rigid SE(3)) — scale を 1 に固定。Init Pose で AABB
//              を合わせた前提で、以後 scale は推定しない。
//              論文的に defensible (DICOM mm + metric depth → rigid)
//              + CMA-ES の拡大発散 failure mode を回避する。
//
//  注: 既存の Shift+Ctrl+P key handler は一切変更しない (互換性)。
//      AutoQCR ボタン / Alt+Ctrl+P / UI チェックボックスで 6/7-DoF 切替。
// =========================================================
static void runAutoQuadCyclicRansac(bool lock_scale) {
    std::cout << "\n=== AutoQuadCyclic-RANSAC (Alt+Ctrl+P)"
              << (lock_scale ? "  [6-DoF rigid]" : "  [7-DoF T+R+Scale]")
              << " ===" << std::endl;

    // ---- 0. 前提条件 -----------------------------------------------------
    auto organs = getOrganList();
    if (organs.empty() || !organs[0] || !liverMesh3D) {
        std::cerr << "[AutoQCR] No liver mesh available" << std::endl;
        return;
    }
    if (g_initOrganVertices.empty() ||
        g_initOrganVertices.size() != organs.size()) {
        std::cerr << "[AutoQCR] g_initOrganVertices not snapshotted. "
                  << "Press 'Apply Init Pose' first to capture initial state."
                  << std::endl;
        return;
    }

    // 解剖ラベルは applyInitRotation 内で auto-trigger されるが、
    // 15 回のループに入る前に一度だけ走らせておく方が無駄な再計算を
    // 避けられる (Region/LR/CC は preset/mask に独立)。
    if (!g_liverRegion.valid()) {
        std::cout << "[AutoQCR] LiverRegion (Shift+R) not yet computed, auto-running..."
                  << std::endl;
        recomputeLiverRegion();
    }
    if (!g_liverLR.valid()) {
        std::cout << "[AutoQCR] LiverLR (Y) not yet computed, auto-running..."
                  << std::endl;
        recomputeLiverLR();
    }
    if (!g_liverCC.valid()) {
        std::cout << "[AutoQCR] LiverCC (Shift+H) not yet computed, auto-running..."
                  << std::endl;
        recomputeLiverCC();
    }

    // ---- 1. 状態スナップ (loop 終了後の Library save 前に best 復元用) ---
    //   ・元の Shift+Ctrl+P と同じく Undo snapshot は 1 個 (loop 全体で 1 件)
    //   ・QUADRANT は Apply Init Pose で確定された値を loop 中固定。
    //     applyInitRotation 自体は g_activeQuadrantMask を変更しないので
    //     原則として書き換わらないはずだが、各 trial 前に明示復元する
    //     ことで安全側に倒す (将来の applyInitRotation 改修への防御)。
    //   ・preset は trial ごとに切り替え、終了後に元に戻す。
    poseAutoSaveBeforeRegistration();

    const RegistrationData::InitRotPreset saved_preset =
        registrationHandle.initRotPreset;
    const uint8_t  saved_mask          = g_activeQuadrantMask;
    const uint32_t saved_callIdx_start = g_callIdx;

    // ---- 2. パターン定義 (QUADRANT に応じて 9 パターンを選択) -----------
    //   UI レイアウト (Anatomical: Left=patient's right):
    //     行1: Up-R+  Up-R   Up    Up-L   Up-L+
    //     行2: Right+ Right  Base  Left   Left+
    //     行3: Dn-R+  Dn-R   Down  Dn-L   Dn-L+
    //
    //   選択されている QUADRANT mask の左右成分を見て中心グループを選ぶ:
    //     右側のみ (AR/PR/AR+PR)        → Right 中心: 左の 3 列 (FAR 含む)
    //     左側のみ (AL/PL/AL+PL)        → Left  中心: 右の 3 列 (FAR 含む)
    //     両側含む (QUAD_ALL や混合)     → Base  中心: 中央の 3 列
    //
    //   bit 配置: AR=0x1 (bit0), AL=0x2 (bit1), PR=0x4 (bit2), PL=0x8 (bit3)
    static const RegistrationData::InitRotPreset kPresetsRight[] = {
        // Right 中心: 画面左 3 列 (患者の右側 / FAR 含む)
        RegistrationData::PRESET_BASE_UP_R_FAR,  // Up-R+
        RegistrationData::PRESET_BASE_UP_R,      // Up-R
        RegistrationData::PRESET_BASE_UP,        // Up
        RegistrationData::PRESET_BASE_R_FAR,     // Right+
        RegistrationData::PRESET_BASE_R,         // Right
        RegistrationData::PRESET_BASE,           // Base
        RegistrationData::PRESET_BASE_DN_R_FAR,  // Dn-R+
        RegistrationData::PRESET_BASE_DN_R,      // Dn-R
        RegistrationData::PRESET_BASE_DN,        // Down
    };
    static const RegistrationData::InitRotPreset kPresetsLeft[] = {
        // Left 中心: 画面右 3 列 (患者の左側 / FAR 含む)
        RegistrationData::PRESET_BASE_UP,        // Up
        RegistrationData::PRESET_BASE_UP_L,      // Up-L
        RegistrationData::PRESET_BASE_UP_L_FAR,  // Up-L+
        RegistrationData::PRESET_BASE,           // Base
        RegistrationData::PRESET_BASE_L,         // Left
        RegistrationData::PRESET_BASE_L_FAR,     // Left+
        RegistrationData::PRESET_BASE_DN,        // Down
        RegistrationData::PRESET_BASE_DN_L,      // Dn-L
        RegistrationData::PRESET_BASE_DN_L_FAR,  // Dn-L+
    };
    static const RegistrationData::InitRotPreset kPresetsBase[] = {
        // Base 中心: 画面中央 3 列 (左右両側ある / QUAD_ALL の標準探索)
        RegistrationData::PRESET_BASE_UP_R,      // Up-R
        RegistrationData::PRESET_BASE_UP,        // Up
        RegistrationData::PRESET_BASE_UP_L,      // Up-L
        RegistrationData::PRESET_BASE_R,         // Right
        RegistrationData::PRESET_BASE,           // Base
        RegistrationData::PRESET_BASE_L,         // Left
        RegistrationData::PRESET_BASE_DN_R,      // Dn-R
        RegistrationData::PRESET_BASE_DN,        // Down
        RegistrationData::PRESET_BASE_DN_L,      // Dn-L
    };

    // QUADRANT mask の左右成分判定
    const bool hasRight = (saved_mask & (LiverLeftRightLabel::QUAD_AR
                                       | LiverLeftRightLabel::QUAD_PR)) != 0;
    const bool hasLeft  = (saved_mask & (LiverLeftRightLabel::QUAD_AL
                                       | LiverLeftRightLabel::QUAD_PL)) != 0;

    const RegistrationData::InitRotPreset* kPresets = nullptr;
    const char* groupName = "";
    if (hasRight && !hasLeft) {
        kPresets  = kPresetsRight;
        groupName = "Right-centered (image left 3 cols)";
    } else if (hasLeft && !hasRight) {
        kPresets  = kPresetsLeft;
        groupName = "Left-centered (image right 3 cols)";
    } else {
        // 両側含む or QUAD_ALL or 空 (空は本来上で弾かれるが防御)
        kPresets  = kPresetsBase;
        groupName = "Base-centered (center 3 cols)";
    }
    constexpr int kNumPresets = 9;  // 3 グループとも 9 個固定

    std::cout << "[AutoQCR] Locked QUADRANT = "
              << LiverLeftRightLabel::quadrantMaskString(saved_mask)
              << "  (0x" << std::hex << (unsigned)saved_mask << std::dec << ")"
              << "  -- preset group: " << groupName
              << "  (" << kNumPresets << " trials)"
              << std::endl;

    // 新セッション (Auto loop 全体で 1 つ) を開始
    poseStartNewSession();
    g_currentOrientLabel = lock_scale ? "AutoQCR6" : "AutoQCR7";
    g_currentOrientRunCount = 0;

    // ---- 2.5. [Opt C] Liver normals を 1 回だけ pre-compute -------------
    //   現状、各 trial の extractQuadCyclicMedoids が liverMesh3D->mNormals
    //   が空 (or サイズ不一致) を検出して computeVertexNormalsFromFaces を
    //   毎回実行している (~30ms × 9 = ~270ms 無駄)。
    //
    //   原因: g_initOrganNormals[0] が空のため、applyInitRotation の復元で
    //   liverMesh3D->mNormals も空のままになる。applyMatrixToMeshVerticesAndNormals
    //   は normals が空のときは何もしないので、結果として extractQuadCyclicMedoids
    //   で初めて normals が computed される。
    //
    //   修正: snapshot 状態の liver geometry で 1 回だけ normals を計算し、
    //   g_initOrganNormals[0] に焼き込む。以降の applyInitRotation 復元 +
    //   applyMatrixToMeshVerticesAndNormals は正しい normals を保持する。
    //
    //   注: liverMesh3D の現在 state (Apply Init Pose 直後) を一時的に
    //   clobber するが、すぐ下の loop の applyInitRotation で完全に上書き
    //   されるので安全。
    if (!g_initOrganNormals.empty() &&
        g_initOrganNormals[0].size() != g_initOrganVertices[0].size()) {
        std::cout << "[AutoQCR/OptC] Pre-computing liver normals once "
                  << "(was missing in g_initOrganNormals[0])..." << std::endl;
        liverMesh3D->mVertices = g_initOrganVertices[0];
        liverMesh3D->mNormals.clear();
        Reg3DCustom::computeVertexNormalsFromFaces(*liverMesh3D);
        g_initOrganNormals[0] = liverMesh3D->mNormals;
        std::cout << "[AutoQCR/OptC] cached "
                  << (g_initOrganNormals[0].size() / 3)
                  << " vertex normals into snapshot "
                  << "(saves ~30ms × " << kNumPresets << " trials)" << std::endl;
    }

    // ---- 3. Best 候補保持用バッファ -------------------------------------
    struct TrialResult {
        int             preset_idx     = -1;
        const char*     preset_name    = "";
        RegistrationData::InitRotPreset preset = RegistrationData::PRESET_BASE;
        bool            valid          = false;  // REGISTERED reached?
        float           compRmse       = FLT_MAX;  // [Opt A] fast mode では chamfer proxy
        float           compIoU2D      = 0.0f;     // [Opt A] fast mode では 0
        // [Opt A] Determinism check 用の best chamfer (Stage 2 score)。
        // g_lastQcrChamfer から consume。fast/full 両 path で同じ値が出るはず。
        float           bestChamfer    = FLT_MAX;
        // Phase 1 観察用: runQuadCyclicRansac が publish した RANSAC prealign の
        // estScale。1.0 から離れているほど scale bias がかかっている疑いが強い。
        // 論文的合理範囲は [0.85, 1.10]、CMA-ES 救済特性を加味すると下方向は
        // 寛容、上方向は厳しめ。
        float           prealignScale  = -1.0f;
        // (verts/norms の保存はやめた: loop 中 liver-only モードで動かし、
        //  winner 確定後に full-organ replay で再現する方式に変更したため。)
        // registrationHandle を best に書き戻すための値 (replay 失敗時の
        // フォールバック用に保持)。
        float baseFitness = 0.0f, baseIcpRmse = 0.0f;
        float baseAvgError = 0.0f, baseRmse = 0.0f;
        float baseMaxError = 0.0f, baseScale = 1.0f;
        float compAvgError = 0.0f, compMaxError = 0.0f;
        int   compCount    = 0;
    };

    TrialResult              bestTrial;     // 最良 1 個
    std::vector<TrialResult> topNSummary;   // ログ用 (top-5)
    topNSummary.reserve(8);

    auto wall_t0 = std::chrono::steady_clock::now();
    int  reg_count = 0;  // REGISTERED 状態に達した試行数

    // ---- 4. 9 trial ループ (ORIENT preset を 9 個、QUADRANT 固定) -------
    for (int pi = 0; pi < kNumPresets; pi++) {
        const RegistrationData::InitRotPreset preset = kPresets[pi];
        const char* presetName = RegistrationData::presetName(preset);
        const int trial_idx = pi + 1;

        std::cout << "\n[AutoQCR " << trial_idx << "/" << kNumPresets << "]"
                  << "  ORIENT=" << presetName
                  << "  QUAD=" << LiverLeftRightLabel::quadrantMaskString(saved_mask)
                  << std::endl;

        // (a) preset を設定、quadrant は固定値で常に復元
        registrationHandle.initRotPreset = preset;
        g_activeQuadrantMask             = saved_mask;

        // (b) g_initOrganVertices から liver のみ復元 + 姿勢適用
        //     startNewSession=false  (loop 全体で 1 セッション)
        //     liver_only=true        (non-liver organ は触らない、winner replay で適用)
        applyInitRotation(/*startNewSession=*/false, /*liver_only=*/true);

        // (c) Shift+Ctrl+P と同じ流儀で stepStartTime をリセット
        g_stepStartTime  = std::chrono::steady_clock::now();
        g_sessionBipopN  = 0;

        // (d) registration 実行 (PoseLibrary には一切触らない)
        //     single_mesh_only=true で QCR を liver-only モードで実行。
        //     ICP の per-iter vertex 変換が 6 organ → 1 organ に縮小される。
        //     最重量の segment (~132K vert × ~15 iter) が消えるのが効く。
        //     Phase 1 観察用: runQuadCyclicRansac が末尾で publish する
        //     g_lastQcrPrealignScale を新規 trial 前に sentinel (-1) にクリア。
        //     trial 内で値が書き換わらなかった場合は -1 のまま読まれる (anomaly
        //     検出用)。
        //     lock_scale=true のときは内部で scale=1 固定 (rigid 6-DoF)。
        g_lastQcrPrealignScale = -1.0f;
        try {
            runQuadCyclicRansac(lock_scale, /*single_mesh_only=*/true);
        } catch (const std::exception& e) {
            std::cerr << "[AutoQCR " << trial_idx << "/" << kNumPresets
                      << "] runQuadCyclicRansac threw: " << e.what()
                      << "  (skipping this trial)" << std::endl;
            continue;
        }

        // (e) REGISTERED に達したか確認
        if (registrationHandle.state != RegistrationData::REGISTERED) {
            std::cout << "[AutoQCR " << trial_idx << "/" << kNumPresets
                      << "] Not REGISTERED  (likely empty quadrant or "
                      << "insufficient visible verts); skipping." << std::endl;
            continue;
        }
        reg_count++;

        // (f) 結果を取り出して bestTrial と比較
        //     consume-and-clear: scale + chamfer を取得後、global を即クリアして
        //     次 trial に持ち越さない (PoseLibrary の g_lastRimRmse パターンと同じ)。
        //     [Opt A] fast mode では registrationHandle.compRmse 自体が chamfer
        //     proxy になっているが、Determinism check 用に bestChamfer も
        //     別途明示的に記録する (winner replay 後の比較に使う)。
        const float trialScale   = g_lastQcrPrealignScale;
        const float trialChamfer = g_lastQcrChamfer;
        g_lastQcrPrealignScale   = -1.0f;
        g_lastQcrChamfer         = -1.0f;

        TrialResult tr;
        tr.preset_idx    = pi;
        tr.preset_name   = presetName;
        tr.preset        = preset;
        tr.valid         = true;
        tr.compRmse      = registrationHandle.compRmse;   // [Opt A] fast mode では chamfer
        tr.compIoU2D     = registrationHandle.compIoU2D;  // [Opt A] fast mode では 0
        tr.bestChamfer   = trialChamfer;                  // [Opt A] Determinism check 用
        tr.prealignScale = trialScale;
        tr.baseFitness   = registrationHandle.fitness;
        tr.baseIcpRmse   = registrationHandle.icpRmse;
        tr.baseAvgError  = registrationHandle.averageError;
        tr.baseRmse      = registrationHandle.rmse;
        tr.baseMaxError  = registrationHandle.maxError;
        tr.baseScale     = registrationHandle.scaleFactor;
        tr.compAvgError  = registrationHandle.compAvgError;
        tr.compMaxError  = registrationHandle.compMaxError;
        tr.compCount     = registrationHandle.compCount;

        std::cout << "[AutoQCR " << trial_idx << "/" << kNumPresets
                  << "] chamfer=" << tr.compRmse   // [Opt A] fast mode で chamfer proxy
                  << "  scale=" << tr.prealignScale
                  << "  (IoU2D=" << tr.compIoU2D << " — fast mode では 0)"
                  << std::endl;

        const bool isBest = (tr.compRmse < bestTrial.compRmse);

        // top-N サマリ更新 (compRmse 昇順、最大 5 件)
        {
            // TrialResult から verts/norms メンバは削除済 (liver-only loop +
            // winner replay 方式に変更したため、メッシュ snapshot は不要)。
            // tr をそのまま push してよい (メタデータのみで軽量)。
            topNSummary.push_back(tr);
            std::sort(topNSummary.begin(), topNSummary.end(),
                      [](const TrialResult& a, const TrialResult& b) {
                          return a.compRmse < b.compRmse;
                      });
            if (topNSummary.size() > 5) topNSummary.resize(5);
        }

        if (isBest) {
            // verts/norms の保存はやめた: loop 後に winner trial を full-organ で
            // 再実行する方式に変更したため、ここではメタデータだけ保存する。
            bestTrial = std::move(tr);
            std::cout << "[AutoQCR " << trial_idx << "/" << kNumPresets
                      << "] *** NEW BEST *** chamfer=" << bestTrial.compRmse
                      << "  scale=" << bestTrial.prealignScale
                      << "  (ORIENT=" << bestTrial.preset_name << ")"
                      << std::endl;
        }
    }

    auto wall_t1 = std::chrono::steady_clock::now();
    const float wall_sec = std::chrono::duration<float>(wall_t1 - wall_t0).count();

    // ---- 5. 結果サマリ ---------------------------------------------------
    std::cout << "\n=== AutoQCR Complete ===" << std::endl;
    std::cout << "[AutoQCR] Trials: " << reg_count << "/" << kNumPresets
              << " reached REGISTERED  (wall=" << wall_sec << "s, ~"
              << (reg_count > 0 ? (wall_sec / reg_count) : 0.0f)
              << "s/trial)" << std::endl;

    if (!bestTrial.valid) {
        std::cerr << "[AutoQCR] No valid trial; nothing to apply. "
                  << "Restoring saved preset/mask." << std::endl;
        registrationHandle.initRotPreset = saved_preset;
        g_activeQuadrantMask             = saved_mask;
        return;
    }

    // top-N ログ (PoseLibrary には載せない — 診断用)
    //   scale 列を追加: Phase 1 観察用。1.0 付近が論文的に望ましく、
    //   scale > 1.0 寄りの解は後段の CMA-ES で救えない傾向がある。
    //   "[scale-bias?]" 注記は scale が論文的合理範囲 [0.85, 1.10] 外の
    //   trial に付ける (まだ採択判定には使わない — 観察のみ)。
    //   [Opt A] fast mode では "compRmse" 列は chamfer proxy。winner 確定後の
    //   replay で real RMSE が registrationHandle に書き込まれる。
    std::cout << "[AutoQCR] top-" << topNSummary.size()
              << " (rank by chamfer proxy [Opt A fast mode]):" << std::endl;
    for (size_t r = 0; r < topNSummary.size(); r++) {
        const auto& t = topNSummary[r];
        const bool scaleOutOfRange = (t.prealignScale > 0.0f) &&
                                     (t.prealignScale < 0.85f || t.prealignScale > 1.10f);
        std::cout << "    #" << (r + 1) << "  ORIENT=" << t.preset_name
                  << "  chamfer=" << t.compRmse
                  << "  scale=" << t.prealignScale
                  << (scaleOutOfRange ? "  [scale-bias?]" : "")
                  << std::endl;
    }
    std::cout << "[AutoQCR] *** WINNER *** ORIENT=" << bestTrial.preset_name
              << "  QUAD=" << LiverLeftRightLabel::quadrantMaskString(saved_mask)
              << "  chamfer=" << bestTrial.compRmse
              << "  scale=" << bestTrial.prealignScale
              << "  (real RMSE computed in upcoming winner replay)"
              << std::endl;

    // ---- 6. Winner trial を full-organ モードで再実行 --------------------
    //   loop 中は liver-only で走らせていたので、non-liver organ
    //   (tumor/portal/vein/segment/gb) はまだ g_initOrganVertices 状態のまま。
    //   winner の preset で applyInitRotation + runQuadCyclicRansac を
    //   full-organ モードで再実行し、全 6 organ を winner pose に持っていく。
    //
    //   決定性: FGR seed は g_trialSeed + g_callIdx 依存。winner trial の
    //   開始時 callIdx (= saved_callIdx_start + bestTrial.preset_idx) に
    //   巻き戻してから呼ぶことで、Stage 1/Stage 2 enumeration と ICP は
    //   loop 内で観測した winner と byte-identical な結果を再現する。
    //   差異があれば WARN ログ (Stage 1/2 は決定的、ICP も入力同じなら同じ)。
    //
    //   コスト: ~1 full trial (~350-500ms)。loop 8 trial 分の full-organ
    //   負荷 (~2-3s) と相殺して大幅な短縮になる。
    std::cout << "\n[AutoQCR] Replaying winner trial (preset_idx="
              << bestTrial.preset_idx << ", ORIENT=" << bestTrial.preset_name
              << ") with full-organ propagation..." << std::endl;

    const uint32_t saved_callIdx_after_loop = g_callIdx;
    // [Opt A] loop の compRmse は fast mode で chamfer proxy になっているので、
    // replay の real RMSE とは直接比較できない。代わりに「loop と replay の
    // best chamfer (g_lastQcrChamfer publish 値)」が一致するかで決定性を検証する。
    // chamfer は QCR Stage 1/Stage 2 enumeration の出力で、両 path で同じ
    // pipeline で計算されるので、これが一致すれば pipeline 決定性 OK。
    const float loop_recorded_chamfer = bestTrial.bestChamfer;

    // Winner trial の callIdx に巻き戻す (FGR seed 再現のため)。
    g_callIdx = saved_callIdx_start + (uint32_t)bestTrial.preset_idx;
    registrationHandle.initRotPreset = bestTrial.preset;
    g_activeQuadrantMask             = saved_mask;
    g_stepStartTime                  = std::chrono::steady_clock::now();
    g_lastQcrPrealignScale           = -1.0f;
    g_lastQcrChamfer                 = -1.0f;

    applyInitRotation(/*startNewSession=*/false, /*liver_only=*/false);
    bool replayOk = true;
    try {
        runQuadCyclicRansac(lock_scale, /*single_mesh_only=*/false);
    } catch (const std::exception& e) {
        std::cerr << "[AutoQCR] winner replay threw: " << e.what() << std::endl;
        replayOk = false;
    }

    // callIdx を loop 終了直後の値に戻す (downstream の seed 連続性を保つ)。
    g_callIdx = saved_callIdx_after_loop;
    // replay 後の g_lastQcrChamfer を consume (Determinism check 後に必要)。
    const float replay_published_chamfer = g_lastQcrChamfer;
    g_lastQcrChamfer = -1.0f;

    if (!replayOk || registrationHandle.state != RegistrationData::REGISTERED) {
        std::cerr << "[AutoQCR] WARN: winner replay did not reach REGISTERED. "
                  << "Falling back to loop-recorded metrics; non-liver organs "
                  << "may be stuck at init pose. compRmse will hold chamfer "
                  << "proxy (not real RMSE)." << std::endl;
        // bestTrial のメタデータで registrationHandle を上書き (旧 writeback 流儀)
        // 注: [Opt A] により bestTrial.compRmse 等は chamfer proxy。real RMSE
        // はこの fallback path では取得できない (winner replay 失敗のため)。
        registrationHandle.state           = RegistrationData::REGISTERED;
        registrationHandle.useRegistration = true;
        registrationHandle.fitness         = bestTrial.baseFitness;
        registrationHandle.icpRmse         = bestTrial.baseIcpRmse;
        registrationHandle.averageError    = bestTrial.baseAvgError;
        registrationHandle.rmse            = bestTrial.baseRmse;
        registrationHandle.maxError        = bestTrial.baseMaxError;
        registrationHandle.scaleFactor     = bestTrial.baseScale;
        registrationHandle.compRmse        = bestTrial.compRmse;
        registrationHandle.compAvgError    = bestTrial.compAvgError;
        registrationHandle.compMaxError    = bestTrial.compMaxError;
        registrationHandle.compIoU2D       = bestTrial.compIoU2D;
        registrationHandle.compCount       = bestTrial.compCount;
    } else {
        // Replay 成功: 決定性チェック (loop と replay の best chamfer 一致確認)。
        // chamfer は QCR Stage 1/Stage 2 の出力で、pipeline が決定的なら一致。
        // 一致すれば computeUnifiedMetrics (KD tree KNN + boundary 分類) も
        // 決定的に走ったはず → 全体の決定性 OK。
        const float chamferDelta = std::abs(replay_published_chamfer - loop_recorded_chamfer);
        if (chamferDelta > 1e-4f) {
            std::cerr << "[AutoQCR] WARN: replay chamfer drift from loop record. "
                      << "loop chamfer=" << loop_recorded_chamfer
                      << " replay chamfer=" << replay_published_chamfer
                      << "  (Δ=" << chamferDelta << "). "
                      << "Determinism assumption may be broken." << std::endl;
        } else {
            std::cout << "[AutoQCR] Replay chamfer matches loop record "
                      << "(chamfer=" << replay_published_chamfer
                      << ", real RMSE from full metrics=" << registrationHandle.compRmse
                      << ", real IoU2D=" << registrationHandle.compIoU2D
                      << "). Determinism OK." << std::endl;
        }
    }

    // preset / mask / orient label は winner で確定。
    registrationHandle.initRotPreset = bestTrial.preset;
    g_activeQuadrantMask             = saved_mask;
    g_currentOrientLabel = std::string(lock_scale ? "AutoQCR6/" : "AutoQCR7/")
                         + bestTrial.preset_name;
    if (saved_mask != LiverLeftRightLabel::QUAD_ALL) {
        g_currentOrientLabel += " @ ";
        g_currentOrientLabel +=
            LiverLeftRightLabel::quadrantMaskString(saved_mask);
    }

    // runQuadCyclicRansac 内部で computeUnifiedMetrics 済みなので再計算不要だが、
    // replay 失敗時の fallback パスを通った場合に備えて mark only。
    g_metricsValid = true;

    // ---- 6.5. Ctrl+G スタイル診断 metrics を publish ----------------------
    //   Ctrl+G の Phase F.5 が publish する診断指標 (RIM rmse / IoU_occluded /
    //   Containment precision/recall) を AutoQCR の winner pose で計算し、
    //   g_lastRim* / g_lastSilOccludedIoU2D / g_lastIoUOcc* globals に書き込む。
    //   直後の poseSaveToLibrary がこれらを consume-and-clear で吸い上げ、
    //   PoseLibrary entry に Ctrl+G 同等の metric 列を残す。
    //   コスト: ~20-30ms。Determinism には影響しない (publish のみ)。
    publishCtrlGStyleDiagnostics();

    // ---- 7. PoseLibrary に 1 件だけ追加 ---------------------------------
    //   - poseSaveToLibrary は session-best と比較するが、session を loop
    //     開始時に reset しているので必ず accept される。
    //   - mask 引数で entry のラベルが "AutoQCR/<orient>" 形式になる。
    //   - elapsedSec は g_stepStartTime から計算される。winner replay 内で
    //     g_stepStartTime が上書きされているので、ここで wall_t0 (AutoQCR
    //     開始時刻) に巻き戻す。これで PoseLibrary entry の elapsed が
    //     「ユーザが Alt+Ctrl+P を押してから完了するまでの実時間」になる。
    gUI.state.regMethod = 1;  // HemiAuto と同じ表示扱い
    g_stepStartTime = wall_t0;
    poseSaveToLibrary(SaveCriterion::RMSE, saved_mask);

    std::cout << "[AutoQCR] PoseLibrary entry added "
              << "(ORIENT=" << bestTrial.preset_name
              << ", QUAD=" << LiverLeftRightLabel::quadrantMaskString(saved_mask)
              << ", callIdx range=" << saved_callIdx_start
              << ".." << (g_callIdx - 1) << ")" << std::endl;
}

// ==========================================================
//  drawCtrlGRimRaycastControls (Advanced セクション用 helper)
// ----------------------------------------------------------
//  REGISTRATION → Advanced CollapsingHeader (sidebar) から
//  onDrawAdvancedCtrlG callback 経由で呼ばれる。
//  Ctrl+G の Rim-weighted (opt-in) + Ctrl+Shift+G の RIM
//  silhouette penalty 2 セクションを同じ globals 越しに表示。
//
//  既存の floating 'Ctrl+G Quadrant Selector' 窓と SAME globals
//  を操作するので、片方を変えればもう片方の表示も同期する
//  (ImGui ID stack は window ごとに独立なので collision なし)。
// ==========================================================
static void drawCtrlGRimRaycastControls() {
    //  Rim-weighted V3R extension (opt-in)
    //  --------------------------------------------------------
    //  Three controls stacked on top of the 4-quadrant selector.
    //  When all three are at their defaults
    //    (AR-vis = OFF, β = 0.0, Show RIM pairs = OFF)
    //  Ctrl+G behaves identically to the original V3R (and at
    //  QUAD_ALL, byte-identical to Shift+G). Ticking any of
    //  them activates the corresponding extension.
    // -----------------------------------------------------------
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Rim-weighted (opt-in)");

    ImGui::Checkbox("AR-visible only (filter source subset)",
                    &g_ctrlgUseArVisFilter);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Restrict source vertices to those "
                          "visible from the fixed AR camera "
                          "(cam_pos=(0,0,0), look-at=+Z).\n"
                          "Removes back-side mesh vertices that have no "
                          "counterpart in the single-sided depth target.\n"
                          "OFF = byte-identical to Shift+G at QUAD_ALL.");
    }

    // -----------------------------------------------------------
    //  Only-Caudal (R-feat-2): anatomical CC-axis filter.
    //  Independent of AR-vis. Uses g_liverCC labels (Shift+H).
    //  Weak/uncomputed states only show a warning here -- the
    //  CC sign Flip toggle lives in the Initial Orientation
    //  panel (drawAnatomicalAxesStatus) to keep one source of
    //  truth; avoid duplicating it on Ctrl+G.
    // -----------------------------------------------------------
    ImGui::Checkbox("Only Caudal rim (mesh-intrinsic)",
                    &g_ctrlgUseCaudalOnly);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Restrict source vertices to the caudal "
                          "(foot-side) half of the mesh, classified by "
                          "LiverCranioCaudalLabel (Shift+H).\n"
                          "Anatomical axis; transform-invariant (no "
                          "camera tuning).\n"
                          "Orthogonal to AR-visible: ticking both "
                          "applies the Combine mode below.\n"
                          "OFF = no caudal filter applied.");
    }
    // CC state notice (no Flip button here -- handled in
    // Initial Orientation panel).
    if (g_ctrlgUseCaudalOnly && !g_liverCC.valid()) {
        ImGui::TextColored(
            ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
            "  CC labels not yet computed - press Shift+H "
            "or Apply Init Pose.");
    } else if (g_ctrlgUseCaudalOnly && g_liverCC.valid() &&
               g_liverCC.cc.weak) {
        ImGui::TextColored(
            ImVec4(0.96f, 0.32f, 0.32f, 1.0f),
            "  [WEAK %.1f%%] verify CC sign in Initial "
            "Orientation panel; use Flip CC if reversed.",
            g_liverCC.cc.confidence * 100.0f);
    }

    // Combine mode (effective only when BOTH checkboxes are on).
    // Always shown; greyed out when not applicable.
    {
        const bool both_on = g_ctrlgUseArVisFilter && g_ctrlgUseCaudalOnly;
        if (!both_on) ImGui::BeginDisabled();
        ImGui::Text("  Combine when both ON:");
        ImGui::SameLine();
        int mode = (int)g_ctrlgArvisCaudalCombine;
        if (ImGui::RadioButton("AND##arvis_caudal", &mode, 0)) {
            g_ctrlgArvisCaudalCombine = 0;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("OR##arvis_caudal",  &mode, 1)) {
            g_ctrlgArvisCaudalCombine = 1;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("AND: vertex must be AR-visible AND caudal "
                              "(strict; smallest subset, default).\n"
                              "OR : vertex passes if AR-visible OR caudal "
                              "(lenient; mutual rescue).\n"
                              "Effective only when both filters above are ON.");
        }
        if (!both_on) ImGui::EndDisabled();
    }

    ImGui::SliderFloat("beta (rim-rim weight boost)",
                       &g_ctrlgBetaRimWeight, 0.0f, 10.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Multiplicative weight for pairs where\n"
                          "  source vertex is on the LiverRegion::RIM band\n"
                          "  AND target point's boundaryDist < threshold.\n"
                          "w_i = 1 + beta * is_rim_src * is_rim_tgt.\n"
                          "0.0 = uniform RMSE (byte-identical accumulator).\n"
                          "1.0 = rim-rim pairs counted twice.\n"
                          "3.0 = rim-rim pairs counted 4x.");
    }

    ImGui::SliderFloat("rim threshold [px]",
                       &g_ctrlgRimTgtThreshPx, 4.0f, 30.0f, "%.1f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Target boundaryDist < threshold -> "
                          "treated as RIM (image-side rim membership).\n"
                          "Same semantic as Shift+P kBoundaryPxTh (default 12).");
    }

    // Phase 7b Step 2 — companion to the slider in the floating "Ctrl+G
    // Quadrant Selector" window. Shares g_instrumentPxThresh, so changes
    // here propagate everywhere (Shift+N / V3R / Phase 7b).
    ImGui::SliderFloat("inst threshold [px]",
                       &g_instrumentPxThresh, 0.0f, 50.0f, "%.1f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Target points with instrumentDist < threshold\n"
                          "are excluded from RIM (instrument-aware\n"
                          "boundary rejection).\n"
                          "Shared with Shift+N / V3R / Phase 7b Shift+W.\n"
                          "Increase if instrument outline leaks into the\n"
                          "RIM band (visible as purple dots in Shift+W).");
    }

    // Phase 7b Step 3 — Rotation-angle constraint
    // (companion to the floating Ctrl+G window sliders).
    ImGui::SliderFloat("rot cos thresh (Ctrl+W)",
                       &g_shapeMatchAnatomyThresh, -1.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Phase 7b Ctrl+W: cos((trace(R)-1)/2) below this\n"
                          "incurs penalty. 0 = 90° max (default), -1 = off.");
    }
    {
        float lam = (float)g_shapeMatchAnatomyLambda;
        if (ImGui::SliderFloat("rot lambda (Ctrl+W)",
                               &lam, 0.0f, 5.0f, "%.2f")) {
            g_shapeMatchAnatomyLambda = (double)lam;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rotation penalty weight. 0 = off, 1 = default.");
        }
    }
    // Phase 7b Step 4a — companion sliders
    {
        bool sign0_only = (g_shapeMatchSignMode == 0x1);
        if (ImGui::Checkbox("sign=0 only (Ctrl+W: no flip)",
                            &sign0_only))
        {
            g_shapeMatchSignMode = sign0_only ? 0x1 : 0xF;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("ON = sign=0 (no flip) only, 30 candidates.\n"
                              "OFF = all 4 signs incl. 180° flips, 120 cand.");
        }
    }
    ImGui::SliderInt("live max_iter (Ctrl+Shift+W)",
                     &g_shapeMatchLiveMaxIter, 0, 200);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cap Live ICP iters (Step 4a fallback).\n"
                          "Only used when 'use rim-axis sweep' is OFF.");
    }
    // Phase 7b Step 4b — Axis sweep toggles
    ImGui::Checkbox("Use rim-axis sweep (Ctrl+Shift+W)",
                    &g_shapeMatchAxisSweepEnabled);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("ON (default) = skip ICP, sweep rim axis.\n"
                          "OFF = use Live ICP (Step 4a).");
    }
    ImGui::SliderInt("axis sweep N",
                     &g_shapeMatchAxisSweepN, 8, 90);
    ImGui::SliderInt("axis sweep tgt subN",
                     &g_shapeMatchAxisSweepTgtSubN, 500, 20000);
    ImGui::Checkbox("dual-variant compare (A:full vs B:rim)",
                    &g_shapeMatchAxisSweepCompare);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run both variants, pick lower CompRMSE.\n"
                          "OFF = Variant A (full vertex) only.");
    }

    ImGui::Checkbox("Show RIM pairs",
                    &g_ctrlgShowRimPairs);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Visualize the rim sets used by beta "
                          "weighting:\n  orange = source RIM (mesh-intrinsic)\n"
                          "  magenta = target RIM (image boundary).\n"
                          "Buffers are populated at next Ctrl+G press.");
    }

    // 状態表示: viz バッファの中身が分かるとデバッグしやすい
    if (g_ctrlgRimVizAvailable) {
        ImGui::Text("RimViz buffers: src=%d  tgt=%d",
                    (int)g_ctrlgRimSrcVertIdx.size(),
                    (int)g_ctrlgRimTgtPos.size());
    } else if (g_ctrlgShowRimPairs) {
        ImGui::TextDisabled(
            "RimViz: press Ctrl+G to populate");
    }

    // -----------------------------------------------------------
    // [Phase D] Colored RIM pairs (K representatives).
    //   Independent from Show RIM pairs above. Shows K paired
    //   source+target spheres in matching HSV colors so the
    //   operator can see WHICH rim vertex maps to WHICH target
    //   point at the current pose. Pairs are sampled from the
    //   ~20k captured at Ctrl+G Phase F.5 (or restored from a
    //   PoseLibrary entry after Apply). 4 sampling modes:
    //     - ArcUniform: even spacing around tgt centroid (default)
    //     - WorstK    : K longest src-tgt distances (diagnostic)
    //     - BestK     : K shortest distances
    //     - Random    : seeded uniform (reshufflable)
    // -----------------------------------------------------------
    ImGui::Checkbox("Show colored pairs (K)",
                    &g_ctrlgShowColoredRimPairs);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Overlay K representative rim-rim pairs "
                          "drawn in matching HSV colors so each "
                          "src↔tgt mapping is visually identifiable.\n"
                          "Data source: g_lastRimPair* (set by "
                          "Ctrl+G / Ctrl+Shift+G Phase F.5; restored "
                          "by Pose Library Apply).\n"
                          "Pairs follow the mesh through subsequent "
                          "ICP/Apply (source is a full-mesh vertex "
                          "index; target is fixed world coords).\n"
                          "Independent from Show RIM pairs above — "
                          "leave both ON for max info.");
    }
    if (g_ctrlgShowColoredRimPairs) {
        ImGui::Indent();
        ImGui::SliderInt("K pairs", &g_ctrlgColoredRimN, 5, 30);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("How many representative pairs to draw "
                              "(5–30). 10 is the sweet spot — small "
                              "enough to differentiate by HSV hue, "
                              "large enough to span the rim.");
        }
        const char* modeItems =
            "ArcUniform\0WorstK\0BestK\0Random\0";
        ImGui::Combo("Sample mode",
                     &g_ctrlgColoredRimMode, modeItems);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "ArcUniform : evenly spaced around tgt centroid "
                "(stable across runs — default).\n"
                "WorstK     : K longest src↔tgt distances "
                "(diagnostic: where is the rim misaligned?).\n"
                "BestK      : K shortest distances "
                "(sanity check).\n"
                "Random     : seeded uniform sample. Use the "
                "Reshuffle button to draw a new sample.");
        }
        // Reshuffle only affects Random mode (other modes are
        // deterministic). Disable the button outside Random so
        // the UI signals this clearly instead of accepting clicks
        // that produce no visible change.
        const bool reshuffleActive =
            (g_ctrlgColoredRimMode ==
             (int)RimPairSampling::Mode::Random);
        if (!reshuffleActive) ImGui::BeginDisabled();
        if (ImGui::Button("Reshuffle")) {
            g_ctrlgColoredRimSeed++;
        }
        if (!reshuffleActive) ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            if (reshuffleActive) {
                ImGui::SetTooltip(
                    "Draw a new K-sample with a fresh seed.\n"
                    "Active only in Random mode.");
            } else {
                ImGui::SetTooltip(
                    "Reshuffle is only available in Random mode.\n"
                    "ArcUniform / WorstK / BestK are deterministic — "
                    "pressing this button would have no effect.");
            }
        }
        ImGui::SameLine();
        if (!g_lastRimPairSrcVertIdx.empty()) {
            ImGui::TextDisabled("(%d pairs avail.)",
                                (int)g_lastRimPairSrcVertIdx.size());
        } else {
            ImGui::TextDisabled("(no pairs — press Ctrl+G)");
        }
        ImGui::Unindent();
    }

    // ----- [NEW UI-1b] RIM silhouette penalty ----------------
    ImGui::Checkbox("RIM silhouette penalty (boundary-to-boundary)",
                    &g_ctrlgsUseRimSil);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Add a boundary-alignment penalty to the cost:\n"
            "  cost += lambda_rim_sil * mean(dist_to_target_boundary)\n"
            "\n"
            "Evaluated only at SOURCE-BOUNDARY raster cells (source\n"
            "cells with at least one non-source 4-neighbour). For\n"
            "each such cell:\n"
            "  outside target mask -> contribute 1.0 (max).\n"
            "  inside target mask  -> contribute min(d/max_px, 1.0)\n"
            "where d is the image-pixel distance to the target\n"
            "silhouette boundary, from the SAM2 distance map.\n"
            "\n"
            "Silhouette-space analogue of Ctrl+G's beta-rim weighting:\n"
            "forces source RIM to target RIM coincidence rather than\n"
            "mere area overlap. Catches drift patterns where source\n"
            "covers target area well but with bulges/dents at the rim.\n"
            "\n"
            "Default OFF.");
    }
    ImGui::SliderFloat("lambda_rim_sil",
                       &g_ctrlgsLambdaRimSil, 0.0f, 2.0f, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Weight for the RIM silhouette penalty.\n"
            "  0.0  : no effect.\n"
            "  0.2  : weak.\n"
            "  0.3  : recommended starting point.\n"
            "  1.0+ : boundary-dominant.\n"
            "Only active when 'RIM silhouette penalty' is ON.");
    }
    ImGui::SliderFloat("rim_sil_max_px",
                       &g_ctrlgsRimSilMaxPx, 10.0f, 300.0f, "%.0f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Image-pixel normalisation cap for the RIM penalty.\n"
            "Source-boundary cells AT the target boundary contribute\n"
            "0; cells >= max_px away from the boundary saturate to 1.\n"
            "  50  : tight (small drift heavily penalised).\n"
            "  100 : recommended starting point.\n"
            "  200 : loose (only large drift penalised).\n"
            "Only active when 'RIM silhouette penalty' is ON.");
    }

    // [NEW UI-RIM-ANAT] Anatomic-mode toggle
    ImGui::Checkbox("Use anatomical RIM (vs. raster boundary)",
                    &g_ctrlgsRimSilAnatomic);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Switch the source-rim definition between two modes:\n"
            "\n"
            "OFF (legacy raster boundary):\n"
            "  Source RIM = every cell on the rasterised silhouette\n"
            "  outline (4-neighbour test on the source hitmap).\n"
            "  Includes the outlines of detached blobs / artefacts;\n"
            "  doesn't know anything about anatomy. Pure geometric.\n"
            "\n"
            "ON (anatomical RIM):\n"
            "  Source RIM = vertices labelled LiverRegionLabel::RIM,\n"
            "  filtered by quadrant + AR-vis + Caudal the SAME way\n"
            "  the Ctrl+G 'Show RIM pairs' checkbox filters them.\n"
            "  These are exactly the orange spheres you see in the\n"
            "  AR view when RimViz is enabled. rim_sil is the mean\n"
            "  distance from each VISIBLE projected RIM vertex to\n"
            "  the target silhouette boundary.\n"
            "\n"
            "F9 viz: in anatomic mode, panels 4 & 6 highlight cells\n"
            "where any anatomical RIM vertex projected, NOT the full\n"
            "silhouette outline. Lets you check whether the rim\n"
            "Ctrl+G already cares about coincides with the SAM2\n"
            "boundary.\n"
            "\n"
            "Only active when 'RIM silhouette penalty' is ON.\n"
            "Default OFF -- legacy behaviour.");
    }
    if (g_ctrlgsRimSilAnatomic && !g_ctrlgsUseRimSil) {
        ImGui::TextColored(
            ImVec4(0.96f, 0.72f, 0.28f, 1.0f),
            "  NOTE: anatomic toggle ON but rim_sil penalty OFF -- has no effect");
    }
}

static void snapshotInitialPose() {
    auto organs = getOrganList();
    g_initOrganVertices.resize(organs.size());
    g_initOrganNormals.resize(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        g_initOrganVertices[i] = organs[i]->mVertices;
        g_initOrganNormals[i] = organs[i]->mNormals;
    }
    std::cout << "[MeshBackup] Initial pose snapshot saved (" << organs.size() << " organs)" << std::endl;
}

static void restoreInitialPose() {
    auto organs = getOrganList();
    if (!g_initOrganVertices.empty() && g_initOrganVertices.size() == organs.size()) {
        for (size_t i = 0; i < organs.size(); i++) {
            organs[i]->mVertices = g_initOrganVertices[i];
            organs[i]->mNormals = g_initOrganNormals[i];
            setUp(*organs[i]);
        }
        std::cout << "[MeshBackup] Initial pose restored (" << organs.size() << " organs)" << std::endl;
    }
}
