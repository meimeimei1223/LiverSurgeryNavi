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
        std::cout << "[AR] background overlay: "
                  << (g_arMode ? "ON" : "OFF") << std::endl;
        return;
    }

    // Mode-dispatch scaffold. Phase 1 only fires DEFORM_MODE; the
    // REGISTRATION_MODE branch is a placeholder for the Phase 3 unified app.
    switch (currentMainMode) {
    case DEFORM_MODE:
        DeformPipeline::onKey(key, mods);
        break;
    case REGISTRATION_MODE:
        // Phase 3: dispatch to REG-side handlers here.
        break;
    }
}

static void onMouseButton(GLFWwindow* win, int button, int action, int) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    double x, y; glfwGetCursorPos(win, &x, &y);
    if (action == GLFW_PRESS)        DeformPipeline::onMousePress(button, x, y);
    else if (action == GLFW_RELEASE) DeformPipeline::onMouseRelease(button);
}

static void onMouseMove(GLFWwindow* win, double x, double y) {
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

    // DEFORM_MODE で左ドラッグ中ならグラブ移動
    if (isDragging && deformHandlPlace.state == DeformHandlPlaceData::DEFORM_MODE) {
        DeformPipeline::onMouseMove(x, y, 1.0f / 60.0f);
        return;
    }

    // それ以外はカメラ操作(回転 / パン)
    if (L && !R) OrbitCam.Rotate(dx, dy);
    else if (R && !L) OrbitCam.Pan(dx, -dy);
}

static void onMouseScroll(GLFWwindow*, double, double dy) {
    if (ImGui::GetIO().WantCaptureMouse) return;
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
        deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
        std::cout << "[UI] Full Reset" << std::endl;
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

    // AR snapshot save: not yet ported to DEFORM, no-op for now.
    a.onSaveAR = []{
        std::cout << "[UI] Save AR Image (not yet implemented in DEFORM)" << std::endl;
    };

    a.onResetCamera = []{
        if (gOrbitCamPtr) {
            gOrbitCamPtr->resetToInitialState();
            std::cout << "[UI] Camera reset" << std::endl;
        }
    };

    // DEFORM -> REG transition. Spawns lsn_registration detached and tears
    // this window down. The REG side will read intrinsics_calib.txt etc as
    // usual; nothing is passed via argv.
    a.onStartFromDepth = []{
        std::cout << "[Deform] Restart from Depth -> spawning lsn_registration" << std::endl;
        platform_launch_detached(REG_EXE_PATH);
        glfwSetWindowShouldClose(gWindow, GLFW_TRUE);
    };

    // We are already in DEFORM mode; the REG-side "Deform >>" button is what
    // got us here. Logging is enough.
    a.onSwitchToDeformMode = []{
        std::cout << "[UI] Already in DEFORM mode" << std::endl;
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

    // カメラ設定(現状の正しい K4A 720p)
    OrbitCam.setWindowSizePointers(&gWindowWidth, &gWindowHeight);
    OrbitCam.setGlobalMatrixPointers(&view, &projection, &model, &objPos);
    OrbitCam.setIntrinsics(918.234f, 918.112f, 640.152f, 366.447f, 1280, 720);
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
              << "\n  R/H/D = Rigid / HandlePlace / Deform"
              << "\n  C     = reset handles & shape"
              << "\n  V     = Organs visibility cycle (ON / 50% / OFF)"
              << "\n  T     = Target  visibility cycle (ON / 50% / OFF)"
              << "\n  B     = Board   visibility cycle (ON / 50% / OFF)"
              << "\n  --- AUTO Step 1-4 ---"
              << "\n  1     = classify Src visibility (visible/hidden)"
              << "\n  2     = extract correspondences"
              << "\n  3     = classify (INLIER/MOVER/OUTLIER)"
              << "\n  4     = compute field on vis mesh"
              << "\n  --- AUTO Step 5 ---"
              << "\n  5     = generate handles + auto-switch to DEFORM_MODE"
              << "\n  --- HEMI drive (after key 5) ---"
              << "\n  6     = step active move +gMoveScale"
              << "\n  Bksp  = step active move -gMoveScale"
              << "\n  N     = select next move handle"
              << "\n  A     = toggle AR background overlay (camera intrinsics view)"
              << "\n  - / = = decrease / increase gMoveScale (0.1..2.0)"
              << "\n  --- Inspect ---"
              << "\n  7     = BEFORE/AFTER snapshot toggle"
              << "\n  0     = TetMesh wireframe toggle"
              << "\n  --- Preset (cycle P0->P1->P2->P3->P4) ---"
              << "\n  P     = next preset (then press 5 to regenerate handles)"
              << "\n  ESC   = quit"
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

        // === AR モード（KeyA で切替）===
        //   レジストレーション側 main.cpp の描画パスと同一ロジック:
        //     1) view を「原点 → +Z 方向」固定で上書き
        //        (projection は OrbitCam の intrinsics ベースをそのまま使う)
        //     2) 背景画像（original.jpg）を NDC z=0.999 で描画
        //   ※ updateAndDraw に渡す view 参照はグローバル `view` を指しているため、
        //     ここでグローバルを書き換えれば後段の描画にそのまま反映される。
        if (g_arMode) {
            view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f));
            g_arBg.draw();
        }

        DeformPipeline::updateAndDraw(
            dt, shaderProgram, shaderProgramCube,
            view, projection, OrbitCam.cameraPos);

        gUI.draw(gWindowWidth, gWindowHeight);

        AppImGuiBoot::endFrame();
        glfwSwapBuffers(gWindow);
    }

    AppImGuiBoot::shutdown();
    DeformPipeline::cleanup();
    glfwDestroyWindow(gWindow);
    glfwTerminate();
    return 0;
}
