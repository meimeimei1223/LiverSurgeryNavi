// main.cpp (deform-only minimal)
//
// レジストレーション後のOBJ(registration_model/reg_*.obj)を読み込み、
// 四面体化→SoftBody生成→マウスでグラブ変形できるだけの最小実装。
// UIなし。今後 RegistrationImGuiManager 等を段階的に上乗せする想定。
//
// キー:
//   R: RIGID_MODE (起動時)
//   H: HANDLE_PLACE_MODE  (左クリックでハンドル配置 / 最大5個)
//   D: DEFORM_MODE        (左ドラッグで変形)
//   C: ハンドル＆形状リセット → RIGID_MODE
//   ESC: 終了

#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>     // AR save: timestamp 生成
#include <cstring>    // AR save: memcpy (行反転)
#include <vector>     // AR save: pixel buffer

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// stb_image の実装は1つのTUに必要(mCutMesh.h 等が間接的に参照)
//   ※ implementation フラグは下流 header (DepthUtils.h 等) が再 include する
//      前に必ず undef する。さもないと stb_image.h の実装ブロックが多重展開され、
//      stbi__err 等の redefinition エラーになる(レジストレーション側 main.cpp 同様)。
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "ShaderProgram.h"
#include "FullSphereCameraWithTarget.h"
#include "PathConfig.h"

// 変形関連
#include "Grabber.h"          // Grabber クラス
#include "DeformGlobals.h"    // multiBody, gGrabber, bunnyPos, deformHandlPlace 等
// ↑ AutoDeform.h → NoOpen3DRegistration.h → DepthUtils.h →
//   stb_image.h を transitively include するが、
//   STB_IMAGE_IMPLEMENTATION は既に undef 済みなので安全
#include "DeformPipeline.h"   // initFromRegistered, updateAndDraw, onMouse*
#include "AR.h"               // AR::Background (KeyA で背景オーバーレイ)

// ImGui (DEFORM UI 統合). RegistrationImGuiManager は REG/DEFORM 共通の UI
// 本体で、state.mainMode==1 を見て DEFORM 専用表示に切り替わる。
#include "imgui.h"
#include "AppImGuiBoot.h"
#include "RegistrationImGuiManager.h"
#include "AutoDeformDebugPanel.h"   // AutoDeform 専用デバッグサブパネル(フローティング)

// ============================================================================
// グローバル(RayCast.h / Grabber.h が extern で要求するもの)
// ============================================================================
int          gWindowWidth  = 1280;
int          gWindowHeight = 720;
GLFWwindow*  gWindow       = nullptr;

glm::mat4    model(1.0f), view(1.0f), projection(1.0f);
glm::vec3    objPos(0.0f);

bool         isDragging   = false;
int          hit_index    = -1;
glm::vec3    hit_position(0.0f);

// RIGID_MODE のマウス押下時、クリックがオブジェクト(肝臓メッシュ)に当たったかを
// 記録するフラグ。押下 (onMouseButton) でセットし、ドラッグ (onMouseMove) 中
// 参照する。
//   true  : オブジェクト上 → 左=肝臓回転 / 右=肝臓平行移動
//   false : 何もない所     → 左=カメラオービット / 右=カメラパン
// 左右どちらのボタンでも使うため isDragging とは別管理。
static bool  g_rigidHitObject = false;

// ============================================================================
// Fine-tune モード (AUTO DEFORM 後の手動微調整)
//   モード ON/OFF フラグ gFineTuneMode は DeformGlobals.h で定義し、サブパネル
//   と共有する。ここ (main.cpp) ではグラブの一時状態だけ保持する。
//   ON のとき DEFORM_MODE の通常メッシュグラブを止め、AUTO move 球
//   (gAutoCtrl の move handle) を直接クリックして掴み、カメラ平面内で
//   ドラッグできる。各 handle の manualOffset を更新し、毎フレーム物理を回す。
// ============================================================================
static int       g_fineGrabIdx     = -1;      // 掴み中の move handle idx (-1=なし)
static glm::vec3 g_fineGrabStartOffset(0.0f); // 掴んだ瞬間の manualOffset
static glm::vec3 g_fineGrabAnchor(0.0f);      // 掴んだ瞬間のマウスレイ上の点 (depth 固定面上)
    //   ドラッグ中はこの点からの移動量を offset に積む。
    //   クリック位置と球中心のズレで飛ばないための基準点。
static float     g_fineGrabDepth   = 1.0f;    // 掴んだ点のカメラからの距離 (射影距離)
    //   (これを固定 = カメラ平面内ドラッグ)

FullSphereCamera OrbitCam;

// UI manager: REG/DEFORM 共通クラス。DEFORM 側は state.mainMode=1 固定で起動する。
RegistrationImGuiManager gUI;

// ============================================================================
// AR モード（KeyA で切替）
//   レジストレーション側 main.cpp と同一ロジック:
//     - g_arMode を bool トグル
//     - 描画ループで view を「原点から +Z」に上書き
//     - g_arBg.draw() で original.jpg を背景描画
//     - projection は OrbitCam の intrinsics ベースをそのまま使う
// ============================================================================
static AR::Background g_arBg;
static bool           g_arMode = false;

// ============================================================================
// [REG-parity AR] レターボックス・シーンビューポート
//   REG (compute3DViewport) と同じく calib アスペクトを保持した矩形にシーン
//   (AR背景 + メッシュ) を描く。投影は intrinsics 固定(calib アスペクト)なので、
//   ビューポートも同アスペクトにしないとウィンドウ変形で背景/メッシュが歪む。
//   ImGui はフルウィンドウ座標で描く。
//     g_sceneRect     : GL ビューポート矩形 (x,y は左下基準オフセット)
//     g_sceneViewport : ピッキング用 (0,0,w,h)。Grabber / RayCast::screenToRay は
//                       viewport のオフセットを無視するため、座標側を矩形ローカルへ
//                       変換して渡す(windowToScene)。REG の Umeyama と同じ規約。
// ============================================================================
static struct { int x = 0, y = 0, w = 1280, h = 720; } g_sceneRect;
glm::vec4 g_sceneViewport(0.0f, 0.0f, 1280.0f, 720.0f);   // Grabber.h で extern 宣言

// 右側に確保する UI 幅 (REG と同じ 400)。0 にすれば全幅レターボックス。
static const int kSidebarW = 400;

// calib アスペクトを保持したレターボックス矩形を g_sceneRect / g_sceneViewport に反映
static void updateSceneViewport() {
    int availW = gWindowWidth - kSidebarW;
    int availH = gWindowHeight;
    int vx = 0, vy = 0;
    int vw = (availW > 0) ? availW : gWindowWidth;
    int vh = (availH > 0) ? availH : gWindowHeight;
    int cw = OrbitCam.calibWidth, ch = OrbitCam.calibHeight;
    if (availW > 0 && availH > 0 && cw > 0 && ch > 0) {
        float ca = (float)cw / (float)ch;
        float aa = (float)availW / (float)availH;
        if (aa > ca) { vh = availH; vw = (int)(vh * ca); vx = (availW - vw) / 2; vy = 0; }
        else         { vw = availW; vh = (int)(vw / ca); vx = 0; vy = (availH - vh) / 2; }
    }
    g_sceneRect.x = vx; g_sceneRect.y = vy; g_sceneRect.w = vw; g_sceneRect.h = vh;
    g_sceneViewport = glm::vec4(0.0f, 0.0f, (float)vw, (float)vh);
}

// ウィンドウ座標 (左上原点, px) → シーンビューポート・ローカル座標。
//   g_sceneViewport=(0,0,w,h) と組み合わせて screenToRay に渡すと、レターボックス
//   の中央寄せオフセットを正しく考慮したレイになる。GL ビューポートの y は左下基準
//   なので、ウィンドウ上端からの矩形上端を引いて top-down ローカルへ直す。
static void windowToScene(double winX, double winY, double& sx, double& sy) {
    sx = winX - (double)g_sceneRect.x;
    double vpTopFromWindowTop = (double)gWindowHeight - (double)(g_sceneRect.y + g_sceneRect.h);
    sy = winY - vpTopFromWindowTop;
}

// AR Save Image 用: メインループ内のローカル shaderProgram / shaderProgramCube を
// onSaveAR ラムダから参照するためのポインタ。main() の shader 初期化直後にセットする。
static ShaderProgram* g_shaderPtr     = nullptr;
static ShaderProgram* g_shaderCubePtr = nullptr;

// QCR Tuning globals (REG-only feature). RegistrationImGuiManager.h's
// QCR slider panel short-returns before referencing these when
// state.mainMode==1, so the values are never read in DEFORM; but the
// extern declarations still need definitions at link time.
int   g_qcrSubsetK        = 3;
int   g_qcrMaxTrials      = 100000;
float g_qcrMaxAxisRotDeg  = 45.0f;
float g_qcrMaxTotalRotDeg = 90.0f;

// Forward decls for the two functions that wire / sync the shared UI.
static void setupUICallbacks();
static void syncUIState();

// ============================================================================
// AR Save Image (DEFORM 版)
// ----------------------------------------------------------------------------
// REG 側の ARSave::capture (AR.h) は organMeshes (mCutMesh*) を描く専用なので
// DEFORM の SoftBody VisMesh には使えない。ここでは DEFORM の実描画関数
// DeformPipeline::updateAndDraw() を FBO 内で呼び、メッシュ + ハンドル +
// auto handle を丸ごと入力画像サイズでキャプチャして PNG 保存する。
//
//   - 画像サイズ: AR 背景テクスチャ (original.jpg) の texW/texH を使用。
//     未ロードならウィンドウサイズにフォールバック。
//   - view: AR overlay と同じ「原点 → +Z」固定 (g_arMode の有無に関わらず、
//     保存画像は常に AR 視点で出す。これが intrinsics と整合する唯一の視点)。
//   - projection: OrbitCam.fx/fy/cx/cy から入力画像サイズ基準で構築。
//   - 背景: original.jpg を必ず合成 (AR 画像として意味があるのは背景つき)。
//
// 戻り値: 保存成功なら true。
// ============================================================================
// [CT-handoff AR] DEFORM の臓器は CT 空間(固定 mm)にある。AR オーバーレイ / Save
//   は「原点→+Z 固定 view + depth intrinsics 投影 + original.jpg 背景」で描くので、
//   CT スケール(肝 diag≈8.6)のまま描くと巨大・ズレ位置になり写真に乗らない。
//   そこで AR パスだけ、臓器を実カメラ frame へ戻してから投影する:
//       CT --T--> registered --un-mirror--> camera/photo
//   (REG/DEFORM は depth target を X ミラーするので、写真(未ミラー)に重ねるには
//    registered を X 反転で戻す。)updateAndDraw の model は単位、カリングは無効なので
//   view に反射行列を掛けても面が消えたり裏返ったりしない。
//   reg_transform.txt が無ければ arView をそのまま返す(従来挙動)。
//   T は実行中不変なので初回だけ読み、以降はキャッシュ(毎フレーム I/O / ログ抑止)。
//
//   ※ 反転軸は経験則(DEFORM の mirror は X のみ)。実 RGB に左右/上下が合わない
//     場合は kArFlip を 1 行変えるだけで調整できる。
// ============================================================================
static glm::mat4 arViewForCtOrgans(const glm::mat4& arView, glm::vec3& outCamPos) {
    // [CT-handoff AR] CT 空間の臓器/ターゲット/ボードを「実カメラ(写真)フレーム」へ
    //   戻して intrinsics 投影に乗せるための合成行列 M = flip · T (T=CT->registered)。
    //   臓器・ターゲット・ボードは updateAndDraw で同一 model/view/proj で描かれるため、
    //   M が正しければ3者は必ず一緒に写真へ乗る(個別にズレることはない)。
    //
    //   ★ flip は IDENTITY (1,1,1)。これが正しい理由 —
    //     depth cloud の X ミラー(mirrorMeshAndCloudX)は「registered フレームを
    //     lookAt(原点→+Z)+intrinsics で描くと元写真にピタリ一致する」ようにするための
    //     ミラーで、registered 空間に *既に焼き込まれている*。REG 側 Key A(ARSave)が
    //     registered の臓器を origin→+Z + intrinsics + flip 無し で原写真に正しく重ねて
    //     成立しているのがその証拠。DEFORM は臓器を T^-1 で CT へ移しただけなので、AR では
    //     T で registered へ戻せば REG と完全同一フレームになる(= M = T、追加 flip 不要)。
    //     ここに更に Xflip を掛けると「二重ミラー」となり、メッシュが左右反転して
    //     裏返って見える(= これまでの症状の正体。カメラ位置ではなくハンドネスの問題)。
    //     観測との整合: (-1,1,1)=Xのみ→1回余分にミラー→反転 / (-1,1,-1)=X+Z→ミラー+Z反転
    //     で物体がカメラ背面へ→消失 / (1,1,1)=ミラー無し→REG と一致→正しい、で全て説明可。
    //   ※ Z は厳禁(反転すると消える)。万一 RGB に乗らない時の調整は下の1行のみだが、
    //     まず Board ON + Key A で「ボードの写真が背景写真に重なるか」を見て確定する:
    //     (1,1,1)=flip 無し=REG と同一(既定/本命) / (-1,1,1)=X のみ(旧・二重ミラー)
    static const glm::vec3 kArFlip(1.0f, 1.0f, 1.0f);   // identity: REG Key A と同一フレーム

    static bool      s_init  = false;
    static bool      s_haveT = false;
    static glm::mat4 s_M(1.0f);
    static glm::mat4 s_Minv(1.0f);
    if (!s_init) {
        s_init = true;
        const glm::mat4 Tinv = DeformPipeline::loadRegToCtInverse(&s_haveT); // registered->CT
        if (s_haveT) {
            const glm::mat4 T     = glm::inverse(Tinv);                      // CT->registered
            const glm::mat4 Xflip = glm::scale(glm::mat4(1.0f), kArFlip);    // registered->camera
            s_M    = Xflip * T;                                             // CT -> camera/photo
            s_Minv = glm::inverse(s_M);
            std::cout << "[CT-handoff AR] organs will be rendered in camera frame "
                         "for AR (flip=(" << kArFlip.x << "," << kArFlip.y << ","
                      << kArFlip.z << "))." << std::endl;
        } else {
            std::cout << "[CT-handoff AR] no reg_transform.txt -> AR uses organs as-is."
                      << std::endl;
        }
    }
    if (!s_haveT) { outCamPos = glm::vec3(0.0f); return arView; }
    outCamPos = glm::vec3(s_Minv * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)); // eye in CT space (lighting)
    return arView * s_M;
}

static bool saveDeformARImage() {
    if (!g_shaderPtr || !g_shaderCubePtr) {
        std::cerr << "[ARSave/DEFORM] shaders not ready" << std::endl;
        return false;
    }

    // --- 画像サイズ決定 ---
    //   REG の ARSave::capture と同じく calib 解像度を基準にする(intrinsics の
    //   fx/fy/cx/cy と一致させ、投影が狂わないようにする)。未設定時は背景テクスチャ
    //   → ウィンドウの順でフォールバック。
    int imgW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth
               : (g_arBg.texW > 0) ? g_arBg.texW : gWindowWidth;
    int imgH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight
               : (g_arBg.texH > 0) ? g_arBg.texH : gWindowHeight;
    if (imgW <= 0 || imgH <= 0) {
        std::cerr << "[ARSave/DEFORM] invalid image size " << imgW << "x" << imgH << std::endl;
        return false;
    }

    // --- FBO セットアップ (color tex + depth rbo) ---
    GLuint fbo = 0, colorTex = 0, depthRbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgW, imgH, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, colorTex, 0);

    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, imgW, imgH);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, depthRbo);

    bool ok = false;

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        // viewport を画像サイズに
        GLint prevVp[4];
        glGetIntegerv(GL_VIEWPORT, prevVp);
        glViewport(0, 0, imgW, imgH);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- AR view (原点 → +Z) ---
        glm::mat4 arView = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),
                                       glm::vec3(0.0f, 0.0f, 1.0f),
                                       glm::vec3(0.0f, 1.0f, 0.0f));

        // --- intrinsics projection (入力画像サイズ基準) ---
        //   [black-screen fix] AR は arViewForCtOrgans が臓器を registered(メートル系,
        //   z≈0.5m, diag≈数m)へ戻して描く。OrbitCam.near/far は CT 空間表示用に
        //   sceneDiag 比例で r 倍(near≈2.4 等)されているため、これを AR に使うと
        //   メートル系の臓器が near 面の手前に来て全クリップ → 真っ黒。AR は scale 非依存
        //   の固定 metric near/far(REG Key A と同じ 0.1/100)で組む。intrinsics 項
        //   (fx/fy/cx/cy) は比率なのでスケール非依存でそのまま使える。
        const float kArNear = 0.1f, kArFar = 100.0f;
        glm::mat4 arProj(0.0f);
        arProj[0][0] =  2.0f * OrbitCam.fx / imgW;
        arProj[1][1] =  2.0f * OrbitCam.fy / imgH;
        arProj[2][0] =  1.0f - 2.0f * OrbitCam.cx / imgW;
        arProj[2][1] =  2.0f * OrbitCam.cy / imgH - 1.0f;
        arProj[2][2] = -(kArFar + kArNear) / (kArFar - kArNear);
        arProj[2][3] = -1.0f;
        arProj[3][2] = -2.0f * kArFar * kArNear / (kArFar - kArNear);

        // 1) 背景画像
        g_arBg.draw();

        // 2) DEFORM 実描画 (メッシュ + ハンドル + auto handle)
        //    camPos は AR 視点 = 原点。
        //
        //    ★ 重要: updateAndDraw は内部で物理 step を回す (preSolve/solve/postSolve)。
        //      solveEdges は alpha=compliance/(dt*dt)、postSolve は velocity=diff/dt
        //      を計算するので dt=0 を渡すとゼロ除算で NaN → メッシュ破壊。
        //      そこで gInspectMode=true にして物理 step 自体をスキップさせる
        //      (updateAndDraw の `if (!gInspectMode)` ガードを利用)。
        //      描画のみ実行され、物理状態は一切変化しない。
        const bool savedInspect = gInspectMode;
        gInspectMode = true;
        // [CT-handoff AR] CT 臓器を実カメラ frame へ戻して投影(写真に重なるように)。
        glm::vec3 arCamPos;
        const glm::mat4 arViewOrgans = arViewForCtOrgans(arView, arCamPos);
        DeformPipeline::updateAndDraw(
            0.0f, *g_shaderPtr, *g_shaderCubePtr,
            arViewOrgans, arProj, arCamPos);
        gInspectMode = savedInspect;

        // --- pixel 読み取り + 上下反転 ---
        std::vector<unsigned char> pixels(static_cast<size_t>(imgW) * imgH * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, imgW, imgH, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        const int stride = imgW * 3;
        std::vector<unsigned char> flipped(pixels.size());
        for (int y = 0; y < imgH; y++) {
            std::memcpy(&flipped[y * stride],
                        &pixels[(imgH - 1 - y) * stride], stride);
        }

        // --- timestamp ---
        auto nowc = std::chrono::system_clock::now();
        auto tt   = std::chrono::system_clock::to_time_t(nowc);
        struct tm lt;
#ifdef _WIN32
        localtime_s(&lt, &tt);
#else
        localtime_r(&tt, &lt);
#endif
        char stamp[64];
        std::snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d",
                      lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday,
                      lt.tm_hour, lt.tm_min, lt.tm_sec);

        std::string saveDir = DEPTH_OUTPUT_PATH;
        if (!saveDir.empty() && saveDir.back() != '/') saveDir += "/";
        saveDir += "deform_screenshots/";
        std::filesystem::create_directories(saveDir);

        std::string path = saveDir + "deform_ar_" + stamp + ".png";
        if (stbi_write_png(path.c_str(), imgW, imgH, 3, flipped.data(), stride)) {
            std::cout << "[ARSave/DEFORM] " << imgW << "x" << imgH << " -> "
                      << std::filesystem::absolute(path).string() << std::endl;
            ok = true;
        } else {
            std::cerr << "[ARSave/DEFORM] stbi_write_png failed: " << path << std::endl;
        }

        glViewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);
    } else {
        std::cerr << "[ARSave/DEFORM] FBO incomplete" << std::endl;
    }

    // --- cleanup ---
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (colorTex) glDeleteTextures(1, &colorTex);
    if (depthRbo) glDeleteRenderbuffers(1, &depthRbo);
    if (fbo)      glDeleteFramebuffers(1, &fbo);

    return ok;
}


// ============================================================================
// GLFW コールバック
// ============================================================================
static void onFramebufferSize(GLFWwindow*, int w, int h) {
    gWindowWidth  = w;
    gWindowHeight = h;
    glViewport(0, 0, w, h);
    OrbitCam.onWindowResize(w, h);
}

static void onKey(GLFWwindow* win, int key, int, int action, int mods) {
    // While the UI has keyboard focus (text fields etc), swallow the key so
    // it does not also drive scene shortcuts.
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        return;
    }
    // KeyA: AR背景オーバーレイ切替（レジストレーション側 main.cpp と同一ロジック）
    //   旧 DeformPipeline 側 Key A の "select next move handle" は Key N に移動済み。
    //   API（DeformPipeline.h）には手を加えず、ここで先取りして return。
    if (key == GLFW_KEY_A) {
        g_arMode = !g_arMode;
        // AR を ON にした瞬間、グローバル view を即 AR 視点に切り替える。
        //   hitTest / grab は glfwPollEvents 内 (= updateAndDraw より前) で
        //   グローバル view を参照するため、ここで先にセットしておかないと
        //   「AR ON 直後の最初のクリック」が 1 フレーム前のオービット view で
        //   レイを飛ばしてしまい、AR 表示とマウス位置がズレる。
        if (g_arMode) {
            view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f));
        }
        std::cout << "[AR] background overlay: "
                  << (g_arMode ? "ON" : "OFF") << std::endl;
        return;
    }

    // [Phase 2] DeformPipeline::onKey へのディスパッチを停止。
    //   旧 R/H/D/C/V/T/B/1〜7/N/P/0/-/=/Bksp の各 KEY は全て GUI に
    //   移行済み:
    //     - sub-mode (R/H/D), reset (C), visibility (V/T/B) → side panel
    //     - AutoDeform (1-7/N/P/0/-/=/Bksp/SHIFT+1)        → AutoDeformDebugPanel
    //   DeformPipeline::onKey() 自体は GUI(AutoDeformDebugPanel::tapKey)から
    //   呼ばれてロジックを共有しているため関数定義は残す。ここからの
    //   キーボード由来の呼び出しのみを止める。
    //
    //   将来 KEY を復活させたい場合はこのコメントごと差し替える形で、
    //   下記のような dispatch を再導入できる:
    //     switch (currentMainMode) {
    //     case DEFORM_MODE:        DeformPipeline::onKey(key, mods); break;
    //     case REGISTRATION_MODE:  /* TODO Phase 3 */                  break;
    //     }
    (void)key;
    (void)mods;
}

static void onMouseButton(GLFWwindow* win, int button, int action, int) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    double x, y; glfwGetCursorPos(win, &x, &y);
    // [REG-parity AR] レターボックス・ビューポートのローカル座標へ変換してから
    //   ピッキング(Grabber / fine-tune は g_sceneViewport=(0,0,w,h) を使う)。
    windowToScene(x, y, x, y);

    using S = DeformHandlPlaceData;

    // RIGID_MODE は左右ボタン両方を肝臓操作に使う (左=回転 / 右=平行移動)。
    //   DeformPipeline::onMousePress は LEFT しか処理しない作りなので、
    //   RIGID のオブジェクト命中判定はここで左右まとめて行う。
    //     命中  → isDragging=true, g_rigidHitObject=true (肝臓を操作)
    //     非命中→ isDragging=false (onMouseMove のカメラ分岐へ流れる)
    if (deformHandlPlace.state == S::RIGID_MODE && multiBody && gGrabber) {
        if (action == GLFW_PRESS &&
            (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT)) {
            bool hit = gGrabber->hitTest((float)x, (float)y);
            g_rigidHitObject = hit;
            isDragging       = hit;   // 命中時のみドラッグ開始 (非命中はカメラ操作)
            if (button == GLFW_MOUSE_BUTTON_RIGHT && hit) {
                // 肝臓を右クリックで平行移動するためにグラブ起点を記録。
                // (左クリック回転は rigidRotateAroundCenter を使うので不要)
                gGrabber->startGrab((float)x, (float)y);
            }
        } else if (action == GLFW_RELEASE) {
            if (g_rigidHitObject) gGrabber->endGrab();
            isDragging       = false;
            g_rigidHitObject = false;
            hit_index        = -1;
        }
        return;
    }

    // Fine-tune モード (DEFORM_MODE 中、パネルで ON):
    //   通常メッシュグラブの代わりに AUTO ハンドル球 (fix + move 両方) を直接
    //   クリックして掴む。fix/move は区別しない (通常グラブが種類を区別しないのと
    //   同様)。左ボタンのみ。球に当たれば通し番号 (unifiedIdx) と起点を記録し、
    //   カメラ平面内ドラッグのために「掴んだ center のカメラ距離」を固定保存する。
    if (gFineTuneMode && deformHandlPlace.state == S::DEFORM_MODE && multiBody) {
        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
            RayCast::Ray ray = RayCast::screenToRay(
                (float)x, (float)y, view, projection,
                g_sceneViewport);

            // 全ハンドル (fix+move を通し番号で) について、レイ上の最近接点で当たり判定。
            //   レイ上で handle center に最も近い点を求め、球内かどうかで判定する。
            int   bestIdx = -1;
            float bestT   = 1e30f;
            for (int i = 0; i < gAutoCtrl.numHandles(); i++) {
                glm::vec3 c = gAutoCtrl.unifiedHandleCenter(i);
                float t = glm::dot(c - ray.origin, ray.direction);   // レイ上の射影距離
                if (t <= 0.0f) continue;
                glm::vec3 closest = ray.origin + ray.direction * t;
                float d2 = glm::dot(c - closest, c - closest);
                float r  = gAutoCtrl.unifiedHandleRadius(i);
                if (d2 <= r * r && t < bestT) {
                    bestT   = t;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0) {
                g_fineGrabIdx         = bestIdx;
                g_fineGrabStartOffset = gAutoCtrl.unifiedManualOffset(bestIdx);
                // カメラ距離 = 球 center をレイに射影した距離 bestT で固定。
                // これを保つことで「カメラ平面内」(奥行き固定) ドラッグになる。
                g_fineGrabDepth       = bestT;
                // アンカー = レイ上で bestT の点 (= レイ上で球 center に最も近い点)。
                // クリック位置は球中心から少しずれるが、ドラッグ量は「現在のレイ上
                // の点 − このアンカー」で測るので、掴んだ瞬間は差分ゼロ → 飛ばない。
                g_fineGrabAnchor      = ray.origin + ray.direction * bestT;
                bool isFix = gAutoCtrl.isUnifiedFix(bestIdx);
                std::cout << "[FineTune] grabbed " << (isFix ? "fix=" : "move=")
                          << (isFix ? bestIdx : bestIdx - gAutoCtrl.numFix())
                          << "  depth=" << g_fineGrabDepth
                          << "  startOffset=(" << g_fineGrabStartOffset.x << ","
                          << g_fineGrabStartOffset.y << "," << g_fineGrabStartOffset.z << ")"
                          << std::endl;
            } else {
                g_fineGrabIdx = -1;   // 球を外した → 何もしない (カメラは別途固定)
            }
        } else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT) {
            if (g_fineGrabIdx >= 0) {
                std::cout << "[FineTune] released handle uIdx=" << g_fineGrabIdx << std::endl;
            }
            g_fineGrabIdx = -1;
        }
        return;   // Fine-tune 中は通常グラブに渡さない
    }

    // RIGID 以外 (HANDLE_PLACE / DEFORM) は従来通り DeformPipeline に委譲。
    if (action == GLFW_PRESS)        DeformPipeline::onMousePress(button, x, y);
    else if (action == GLFW_RELEASE) DeformPipeline::onMouseRelease(button);
}

static void onMouseMove(GLFWwindow* win, double x, double y) {
    // [REG-parity AR] 以降の x,y はレターボックス・ビューポートのローカル座標。
    //   Grabber / fine-tune は g_sceneViewport=(0,0,w,h) を使うため、座標側で
    //   中央寄せオフセットと GL ビューポートの y(左下基準)を吸収する。
    windowToScene(x, y, x, y);
    // Keep the last position fresh even while the UI captures the mouse,
    // otherwise the first frame back in the scene jumps by a large dx/dy.
    static glm::vec2 last(0.0f);
    if (ImGui::GetIO().WantCaptureMouse) {
        last = glm::vec2((float)x, (float)y);
        return;
    }
    float dx = (float)x - last.x;
    float dy = (float)y - last.y;
    last = glm::vec2((float)x, (float)y);

    bool L = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
    bool R = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    // RIGID_MODE drag:
    //   オブジェクト命中時 (g_rigidHitObject):
    //     L : 肝臓を回転 (rigidRotateAroundCenter, camera right/up 軸まわり)
    //     R : 肝臓を平行移動 (grab した点がカーソルを追従)
    //   非命中時は下のカメラ操作 (オービット / パン) に流れる。
    if (deformHandlPlace.state == DeformHandlPlaceData::RIGID_MODE
        && multiBody && g_rigidHitObject) {
        // 回転軸・奥行き軸は通常 / AR とも OrbitCam の右・上・視線ベクトルを使う。
        //   登録 pose が 180°Y 回転のため cameraRight≈(-1,0,0), cameraDir≈(0,0,-1)。
        //   AR 中も OrbitCam はフリーズしているだけで値は同じ向きを保持しているので、
        //   ここで AR 用の固定軸 (+X/+Y/+Z) に差し替えると右と視線の符号が反転し、
        //   通常モードと AR モードで回転方向が逆になってしまう。OrbitCam の軸で
        //   統一すれば通常 / AR で同じ操作感になる。
        const glm::vec3 axisRight = OrbitCam.cameraRight;
        const glm::vec3 axisUp    = OrbitCam.cameraUp;
        const glm::vec3 axisFwd   = OrbitCam.cameraDirection;

        if (L && !R) {
            float rotX = dy * 0.01f;
            float rotY = dx * 0.01f;
            if (std::abs(rotX) > 1e-5f) multiBody->rigidRotateAroundCenter(axisRight, rotX);
            if (std::abs(rotY) > 1e-5f) multiBody->rigidRotateAroundCenter(axisUp,    rotY);
        } else if (R && !L) {
            // 右クリックで startGrab 済み。grab 点をカーソルに追従させる。
            if (gGrabber) gGrabber->moveGrab((float)x, (float)y, 1.0f / 60.0f);
        } else if (L && R) {
            // 左右同時: カメラ奥行き方向に平行移動
            glm::vec3 mv = axisFwd * (-dy) * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
            multiBody->rigidTranslate(mv);
        }
        return;
    }

    // Fine-tune ドラッグ: 掴んだ move 球をカメラ平面内で動かす。
    //   掴んだ center のカメラ距離 (g_fineGrabDepth) を固定し、現在のマウス
    //   レイ上でその距離の点を新しい目標 center とする。これがカメラ平面内
    //   ドラッグ (奥行き固定) になる。カメラを回せば別角度から、Key A 中なら
    //   AR 視点平面で動く (view が AR に切替わっているため自動対応)。
    //   差分を manualOffset に積み、毎フレーム物理を回してリアルタイム反映。
    if (gFineTuneMode && g_fineGrabIdx >= 0
        && deformHandlPlace.state == DeformHandlPlaceData::DEFORM_MODE
        && multiBody && (L && !R)) {
        RayCast::Ray ray = RayCast::screenToRay(
            (float)x, (float)y, view, projection,
            g_sceneViewport);

        // 掴んだ深度の点 = レイ起点 + 方向 * 固定距離。
        glm::vec3 targetCenter = ray.origin + ray.direction * g_fineGrabDepth;
        // 掴んだ瞬間のアンカー (レイ上の点) からの移動量を、起点 offset に加算。
        //   掴み直後は targetCenter == g_fineGrabAnchor なので delta=0 → 飛ばない。
        glm::vec3 deltaWorld   = targetCenter - g_fineGrabAnchor;
        glm::vec3 newOffset    = g_fineGrabStartOffset + deltaWorld;

        gAutoCtrl.setUnifiedManualOffset(multiBody, g_fineGrabIdx, newOffset);

        // ★ FPS 対策: ここでは solve を呼ばない。
        //   setUnifiedManualOffset 内の applyProgress/applyFixHandle が handle 頂点を
        //   目標位置に固定 (position=prev=目標, velocity=0) するので、あとは
        //   メインループの updateAndDraw が毎フレーム回す通常 solve (重力ゼロ・1 回)
        //   が周囲を馴染ませる。通常のメッシュグラブ (moveGrab) と全く同じ構造。
        //
        //   以前はここで runBoost(15) を呼んでいたが、onMouseMove は 1 フレーム
        //   内にマウスイベントの数だけ複数回呼ばれるため、1 フレームで
        //   15 iter × 数回 + updateAndDraw の solve = 100+ iter となり激重だった。
        //   位置を書くだけにすれば通常グラブ同様に軽い。
        return;
    }

    // DEFORM_MODE で左ドラッグ中ならグラブ移動
    if (isDragging && deformHandlPlace.state == DeformHandlPlaceData::DEFORM_MODE) {
        DeformPipeline::onMouseMove(x, y, 1.0f / 60.0f);
        return;
    }

    // それ以外はカメラ操作(回転 / パン)
    //   RIGID で非命中の場合もここに来る → ボード中心オービット / パン。
    //   ★ ただし AR モード中はカメラを固定する(視点を intrinsics に合わせて
    //     背景画像に重ねるのが目的なので、オブジェクト外ドラッグでカメラを
    //     動かさない)。OrbitCam を回すと cameraPos が変わり、AR を抜けたとき
    //     視点が飛ぶ・ライティングが変わる原因になるため、ここで抑止する。
    if (g_arMode) return;
    if (L && !R) OrbitCam.Rotate(dx, dy);
    else if (R && !L) OrbitCam.Pan(dx, -dy);
}

static void onMouseScroll(GLFWwindow*, double, double dy) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    // AR モード中はカメラ固定 (ズームで radius が変わると intrinsics 視点が
    // ずれて背景と合わなくなるため抑止)。
    if (g_arMode) return;
    OrbitCam.Zoom(dy);
}

// ============================================================================
// FPS + モード をタイトルバーに表示(元 main.cpp の showFPS 互換)
// ============================================================================
static void showFPS(GLFWwindow* win) {
    static double prevTime = 0.0;
    static int    frames   = 0;
    double now = glfwGetTime();
    double dt  = now - prevTime;
    frames++;
    if (dt > 0.25) {
        double fps = frames / dt;
        double ms  = 1000.0 / fps;
        std::ostringstream s;
        s << std::fixed << std::setprecision(1)
          << "FPS " << fps << " (" << ms << "ms)"
          << "  [" << mainModeName(currentMainMode)
          << " / " << deformStateName(deformHandlPlace.state) << "]"
          << "  handles=" << (multiBody ? multiBody->handleGroups.size() : 0)
          << "/" << SoftBody::MAX_HANDLE_GROUPS
          << "  V:" << alphaLabel(gOrganAlpha)
          << " T:" << alphaLabel(gTargetAlpha)
          << " B:" << alphaLabel(gBoardAlpha)
          << "  " << gWindowWidth << "x" << gWindowHeight;
        glfwSetWindowTitle(win, s.str().c_str());
        prevTime = now;
        frames   = 0;
    }
}

// ============================================================================
// OpenGL 初期化
// ============================================================================
static bool initOpenGL() {
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight,
                               "Liver Deform (minimal)", nullptr, nullptr);
    if (!gWindow) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(gWindow);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;

    glfwSetKeyCallback        (gWindow, onKey);
    glfwSetMouseButtonCallback(gWindow, onMouseButton);
    glfwSetCursorPosCallback  (gWindow, onMouseMove);
    glfwSetScrollCallback     (gWindow, onMouseScroll);
    glfwSetFramebufferSizeCallback(gWindow, onFramebufferSize);

    glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
    glViewport(0, 0, gWindowWidth, gWindowHeight);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

// ============================================================================
// UI Actions wire-up. Only the actions that DEFORM mode actually triggers
// are wired; the rest of the RegUIActions table is left as default-constructed
// std::function (the UI side guards with `if (actions.xxx)` before calling).
// ============================================================================
static void setupUICallbacks() {
    auto& a = gUI.actions;

    a.onRigidMode = []{
        if (currentMainMode != DEFORM_MODE) return;
        deformHandlPlace.state = DeformHandlPlaceData::RIGID_MODE;
        if (multiBody) multiBody->setRigidMode(true);
        std::cout << "[UI] Rigid mode" << std::endl;
    };

    a.onHandlePlaceMode = []{
        if (currentMainMode != DEFORM_MODE) return;
        if (multiBody) {
            multiBody->setRigidMode(true);
            multiBody->initPhysics();
            multiBody->reapplyHandleConstraints();
        }
        deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
        std::cout << "[UI] Handle Place mode" << std::endl;
    };

    a.onDeformMode = []{
        if (currentMainMode != DEFORM_MODE) return;
        if (multiBody) multiBody->setRigidMode(false);
        deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
        std::cout << "[UI] Deform mode" << std::endl;
    };

    a.onFullReset = []{
        if (currentMainMode != DEFORM_MODE) return;
        deformHandlPlace.reset();
        if (multiBody) {
            multiBody->fullReset();
            multiBody->setRigidMode(true);
            multiBody->initPhysics();
        }
        // ★ AutoHandleController が掴んでいた auto handles (fix/move スフィア) も消去。
        //   multiBody->fullReset() で invMass を含む particle 状態は既に initial に
        //   戻っているので、ここでは body=nullptr を渡して「saved invMass 書き戻し」
        //   をスキップし、内部 handle 配列だけ空にする。
        //   描画は gAutoCtrl.numFix()/numMove() を走査しているのでスフィアも消える。
        gAutoCtrl.clear(nullptr);
        // AutoDeform の Step 進捗も初期化 (再度 Stage 1 から走らせ直す前提)。
        gAutoDeform.stage = 0;
        // AutoOpt の表示キャッシュも消去 (古い RMSE 履歴が残らないように)
        AutoDeformOpt::gLastSingle = AutoDeformOpt::SingleResult{};
        AutoDeformOpt::gLastAll    = AutoDeformOpt::AllResult{};

        deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
        std::cout << "[UI] Full Reset (incl. auto handles & stage)" << std::endl;
    };

    a.onHandleRadiusChanged = [](float r) {
        gGroupRadius = r;
    };

    // Visibility toggle: REG side has per-organ g_meshAlpha[8]; DEFORM only
    // has 3 single values (gOrganAlpha / gBoardAlpha / gTargetAlpha). For
    // Phase 1 we accept "all organs together" -- i=0..5 all cycle gOrganAlpha,
    // i=6 cycles gBoardAlpha, i=7 cycles gTargetAlpha. syncUIState() mirrors
    // the same value into all 6 organ UI slots so the buttons stay in sync.
    a.onToggleOrgan = [](int i) {
        if (i >= 0 && i <= 5) {
            cycleAlpha(gOrganAlpha);
            std::cout << "[UI] Organs = " << alphaLabel(gOrganAlpha) << std::endl;
        } else if (i == 6) {
            cycleAlpha(gBoardAlpha);
            std::cout << "[UI] Board = " << alphaLabel(gBoardAlpha) << std::endl;
        } else if (i == 7) {
            cycleAlpha(gTargetAlpha);
            std::cout << "[UI] Target = " << alphaLabel(gTargetAlpha) << std::endl;
        }
    };

    // AR snapshot save (DEFORM 版): FBO に AR 視点で描画して PNG 保存。
    a.onSaveAR = []{
        if (currentMainMode != DEFORM_MODE) return;
        bool ok = saveDeformARImage();
        std::cout << "[UI] Save AR Image (DEFORM) "
                  << (ok ? "OK" : "FAILED") << std::endl;
    };

    a.onResetCamera = []{
        if (gOrbitCamPtr) {
            // resetToInitialState() は rotation=identity に戻すだけで、起動時に
            // applyRegistrationCameraPose() が設定した「180度Y回転 + TARGET_TEXTURE」
            // を捨ててしまう。そのため reset 後に同じ pose を再適用し、
            // 「起動直後とまったく同じ見た目」に戻す。
            gOrbitCamPtr->resetToInitialState();
            DeformPipeline::applyRegistrationCameraPose(*gOrbitCamPtr);
            std::cout << "[UI] Camera reset (restored registration pose)" << std::endl;
        }
    };

    // DEFORM stays an independent process. The button just acknowledges the
    // intent in the log; launching lsn_registration is left to the user
    // (Qt Creator, terminal, ...). Phase 3 unified binary will replace this
    // with an in-process currentMainMode = REGISTRATION_MODE switch.
    a.onStartFromDepth = []{
        std::cout << "[Deform] Start From Depth requested -- launch lsn_registration manually" << std::endl;
    };

    // We are already in DEFORM mode; the REG-side "Deform >>" button is what
    // got us here. Logging is enough.
    a.onSwitchToDeformMode = []{
        std::cout << "[UI] Already in DEFORM mode" << std::endl;
    };

    // [deform round-trip FIX] メインパネル Export セクションの
    // 「Export Deformed (CT) -> ct_deformed/」ボタン。
    // これまで未配線だったため、UI 側の `if (actions.onExportDeformedCT)`
    // ガードに弾かれて無言で何もしなかった。AutoDeformDebugPanel の
    // ボタン / キー E と同じ exportDeformedOrgans() に接続する。
    a.onExportDeformedCT = []{
        bool ok = DeformPipeline::exportDeformedOrgans();   // -> registration_model/ct_deformed/
        std::cout << "[UI] Export Deformed (CT) "
                  << (ok ? "OK -> registration_model/ct_deformed/" : "FAILED")
                  << std::endl;
    };
}

// ============================================================================
// UI State sync (globals -> gUI.state, per frame).
//   - Keep state.mainMode pinned to 1 (DEFORM) so the UI stays in DEFORM
//     presentation no matter what the user fiddles with.
//   - Mirror DEFORM sub-mode and the 3 alpha values so toggles done via the
//     V/T/B keyboard shortcuts stay in sync with the side panel.
//   - REG-side state fields (regState, depthRunning, cameraState, ...) are
//     set once at startup and never touched here; the UI short-returns on
//     them because state.mainMode==1.
// ============================================================================
static void syncUIState() {
    auto& s = gUI.state;
    s.mainMode  = 1;
    s.depthDone = true;
    s.regState  = 4;

    s.deformState     = (int)deformHandlPlace.state;
    s.handleGroups    = multiBody ? (int)multiBody->handleGroups.size() : 0;
    s.maxHandleGroups = SoftBody::MAX_HANDLE_GROUPS;
    s.handleRadius    = gGroupRadius;

    for (int k = 0; k < 6; k++) {
        s.organs[k].alpha = gOrganAlpha;
    }
    s.boardAlpha  = gBoardAlpha;
    s.targetAlpha = gTargetAlpha;

    if (s.arSavedTimer > 0) s.arSavedTimer -= ImGui::GetIO().DeltaTime;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    initPaths();
    initFilePaths();

    // [intrinsics-step-1-test] --dry-run: CPU-only verification of the Step-1
    // intrinsics path (load K from intrinsics_k4a.txt + board UV). No GL window,
    // no reg_*.obj required. Logs deformK + board UV range, then exits.
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dry-run") {
            DeformPipeline::dryRunStep1();
            return 0;
        }
    }

    if (!initOpenGL()) {
        std::cerr << "OpenGL init failed" << std::endl;
        return -1;
    }

    // カメラ設定: intrinsics.txt から読み込んで REG と同じ K を使う(AR の重なりを
    //   REG と一致させるため)。読めなければ従来の K4A 720p ハードコードにフォールバック。
    OrbitCam.setWindowSizePointers(&gWindowWidth, &gWindowHeight);
    OrbitCam.setGlobalMatrixPointers(&view, &projection, &model, &objPos);
    {
        Reg3DCustom::CameraIntrinsics K = DeformPipeline::loadDeformIntrinsics();
        if (K.valid())
            OrbitCam.setIntrinsics(K.fx, K.fy, K.cx, K.cy, K.width, K.height);
        else
            OrbitCam.setIntrinsics(918.234f, 918.112f, 640.152f, 366.447f, 1280, 720);
    }
    OrbitCam.printCameraInfo();

    // DeformPipeline の Key 1 で OrbitCam.InitialRadius を参照するためのポインタ登録
    gOrbitCamPtr = &OrbitCam;

    // シェーダ
    ShaderProgram shaderProgram;
    shaderProgram.loadShaders((SHADERS_PATH + "basic.vert").c_str(),
                              (SHADERS_PATH + "basic.frag").c_str());
    ShaderProgram shaderProgramCube;
    shaderProgramCube.loadShaders((SHADERS_PATH + "texture.vert").c_str(),
                                  (SHADERS_PATH + "texture.frag").c_str());

    // AR Save Image 用に shader をグローバル公開 (onSaveAR ラムダから参照)
    g_shaderPtr     = &shaderProgram;
    g_shaderCubePtr = &shaderProgramCube;

    // AR背景の初期化（レジストレーション側 main.cpp と同じ運用）
    //   loadTexture が失敗してもアプリは継続。draw() は ready=false のとき no-op。
    g_arBg.initGL();
    if (!g_arBg.loadTexture(DEPTH_OUTPUT_PATH + "original.jpg")) {
        std::cerr << "[AR] background image missing -- overlay disabled"
                  << std::endl;
    }

    // ImGui (REG/DEFORM 共通の AppImGuiBoot 経由)
    AppImGuiBoot::init(gWindow);

    // DEFORM アプリは常に DEFORM フェーズで起動。UI 側は state.mainMode==1
    // を見て DEFORM 専用表示に切り替わる(DEPTH セクション compact、REG セク
    // ション "Registration: Done"、DEFORM/Visibility セクションが active)。
    currentMainMode           = DEFORM_MODE;   // default だが明示
    gUI.state.mainMode        = 1;
    gUI.state.depthDone       = true;
    gUI.state.regState        = 4;             // 4 = REGISTERED
    gUI.state.maxHandleGroups = SoftBody::MAX_HANDLE_GROUPS;

    setupUICallbacks();

    // 変形パイプラインの初期化
    if (!DeformPipeline::initFromRegistered()) {
        std::cerr << "[main] DeformPipeline init failed. "
                     "Make sure registration_model/reg_*.obj exist." << std::endl;
        glfwDestroyWindow(gWindow);
        glfwTerminate();
        return -1;
    }

    // Target / Board 参照メッシュ(任意)を読み込む
    DeformPipeline::loadReferenceMeshes();

    // ★ シーンスケールに応じてカメラパラメータを調整 ★
    //   loadReferenceMeshes() で gSceneDiag が確定済み。これを基に
    //   InitialRadius / minRadius / maxRadius / pan・zoom 感度を比例調整。
    //   applyRegistrationCameraPose は InitialRadius を起点に位置を決めるので
    //   この順序が必須(scale 適用 → pose 適用)。
    DeformPipeline::applySceneScaleToCamera(OrbitCam, gSceneDiag);

    // レジストレーション側 main.cpp と同じカメラ向き
    //   (Y軸180°回転 + TARGET_TEXTURE 注視)
    DeformPipeline::applyRegistrationCameraPose(OrbitCam);

    // Grabber を構築して接続
    Grabber grabber;
    gGrabber = &grabber;
    gGrabber->setPhysicsObject(multiBody);

    std::cout << "\n=== Ready ==="
              << "\n  Mouse  = orbit / pan / zoom (RIGID_MODE: drag = move rigid body)"
              << "\n  A      = toggle AR background overlay"
              << "\n  ESC    = quit"
              << "\n  --- All other ops moved to GUI ---"
              << "\n  Side panel       : Rigid / Handle / Deform, Sphere Radius, Visibility, Reset"
              << "\n  AutoDeform panel : Stages 1-5, HEMI drive, Inspect, Preset, Wireframe"
              << std::endl;

    // メインループ
    double last = glfwGetTime();
    while (!glfwWindowShouldClose(gWindow)) {
        double now = glfwGetTime();
        float  dt  = (float)(now - last);
        last = now;

        showFPS(gWindow);
        glfwPollEvents();

        AppImGuiBoot::beginFrame();
        syncUIState();

        // liver / texture の中心をカメラに通知(レジストレーション側と同じ運用)
        DeformPipeline::updateCameraTargets(OrbitCam);
        OrbitCam.UpdateCamera(dt);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // [REG-parity AR] シーンは calib アスペクト保持のレターボックス・ビュー
        //   ポートに描く(REG の compute3DViewport と同じ)。ImGui の前に全幅へ戻す。
        updateSceneViewport();
        glViewport(g_sceneRect.x, g_sceneRect.y, g_sceneRect.w, g_sceneRect.h);

        // === AR モード（KeyA で切替）===
        //   レジストレーション側 main.cpp の描画パスと同一ロジック:
        //     1) view を「原点 → +Z 方向」固定で上書き
        //        (projection は OrbitCam の intrinsics ベースをそのまま使う)
        //     2) 背景画像（original.jpg）を NDC z=0.999 で描画
        //   ※ updateAndDraw に渡す view 参照はグローバル `view` を指しているため、
        //     ここでグローバルを書き換えれば後段の描画にそのまま反映される。
        //
        //   ★ camPos も AR 視点の原点 (0,0,0) に固定する。
        //     updateAndDraw 内で camPos は lightPos (ライティング光源) に使われる。
        //     view だけ固定して camPos を OrbitCam.cameraPos のままにすると、
        //     マウスでカメラを回した際に「視点は固定なのに光源だけ動く」
        //     → ライティングだけ変化する不具合になる。AR save 関数とも揃える。
        glm::vec3 renderCamPos = OrbitCam.cameraPos;
        glm::mat4 renderProjection = projection;   // 既定: CT 空間表示用(orbit, r倍near/far)
        if (g_arMode) {
            // [CT-handoff AR] CT 臓器を実カメラ frame へ戻して描画(背景写真に重なる)。
            const glm::mat4 arViewBase = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),
                                                     glm::vec3(0.0f, 0.0f, 1.0f),
                                                     glm::vec3(0.0f, 1.0f, 0.0f));
            view = arViewForCtOrgans(arViewBase, renderCamPos);

            // [black-screen fix] AR は臓器を registered(メートル系)へ戻すので、
            //   CT 用に r 倍した projection(near≈2.4)では near クリップで全消し→真っ黒。
            //   AR 専用に固定 metric near/far(0.1/100, REG Key A と同じ)で intrinsics
            //   投影を組み直す。fx/fy/cx/cy は比率なのでスケール非依存。
            const int   pw = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : gWindowWidth;
            const int   ph = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : gWindowHeight;
            const float kArNear = 0.1f, kArFar = 100.0f;
            glm::mat4 p(0.0f);
            p[0][0] =  2.0f * OrbitCam.fx / pw;
            p[1][1] =  2.0f * OrbitCam.fy / ph;
            p[2][0] =  1.0f - 2.0f * OrbitCam.cx / pw;
            p[2][1] =  2.0f * OrbitCam.cy / ph - 1.0f;
            p[2][2] = -(kArFar + kArNear) / (kArFar - kArNear);
            p[2][3] = -1.0f;
            p[3][2] = -2.0f * kArFar * kArNear / (kArFar - kArNear);
            renderProjection = p;

            // [AR pick fix] ピッキング(RayCast::screenToRay / Grabber)はグローバル
            //   `projection` を参照する。AR 中は view だけ AR 用(registered メートル系
            //   eye)に書き換わるのに projection が orbit 用(CT 用 r 倍 near≈2.4)の
            //   ままだと、不整合ペアで unproject され ray.origin がシーン全体(~0.3m)
            //   を通り過ぎた位置に置かれる。その結果 Fine-tune のヒットテスト
            //   (t = dot(c - origin, dir) <= 0 で skip)が全ハンドルを外し、
            //   「AR 中にハンドルが掴めない/動かない」症状になる。
            //   描画に使う AR projection をグローバルにも書き戻し、view と同じ
            //   「描画時に更新 → 次フレームのマウスイベントが参照」の規約に揃える。
            //   (AR OFF 時は OrbitCam.UpdateCamera が毎フレーム orbit projection を
            //    グローバルへ書き戻すので、ここでの上書きは自動的に元に戻る。)
            projection = p;

            g_arBg.draw();
        }

        DeformPipeline::updateAndDraw(
            dt, shaderProgram, shaderProgramCube,
            view, renderProjection, renderCamPos);

        // [REG-parity AR] ImGui はフルウィンドウ座標で描くのでビューポートを戻す。
        glViewport(0, 0, gWindowWidth, gWindowHeight);
        gUI.draw(gWindowWidth, gWindowHeight);
        AutoDeformDebugPanel::draw();   // AutoDeform 専用デバッグサブパネル

        AppImGuiBoot::endFrame();
        glfwSwapBuffers(gWindow);
    }

    AppImGuiBoot::shutdown();
    DeformPipeline::cleanup();
    glfwDestroyWindow(gWindow);
    glfwTerminate();
    return 0;
}
