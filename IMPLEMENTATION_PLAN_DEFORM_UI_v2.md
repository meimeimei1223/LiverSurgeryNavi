# 実装計画書 v2: DEFORM アプリへの UI 統合 (Phase 1)

> **対象**: AAA_LiverSurgeryNaviComb プロジェクト
> **実装担当**: Claude Code
> **改訂日**: 2026-05-27
> **改訂理由**: REG 側プロジェクトの大幅アップデート後の再調査結果を反映
> **前提**: 現状は `lsn_registration` / `lsn_deform` の 2 プロジェクト分離開発。最終的に 1 つの実行ファイルに統合予定。
>
> **v1 からの主な変更点**:
> - REG 側 `onSwitchToDeformMode` は **stub のまま** が確認された → 「置換」ではなく「実装」と明示
> - UI 変数名は **`gUI`** (REG 側準拠) に統一 (旧 `gUIManager`)
> - クラス名 `RegistrationImGuiManager` は **rename しない** (REG 側破壊回避、ファイル移動のみ)
> - Step B (AppMode.h 共通化) は **削除** (REG 側に MainMode enum がそもそも存在しないため ODR 違反は発生しない)
> - DEFORM 側 alpha 管理 (gOrganAlpha 単一値) と REG 側 (g_meshAlpha[8] 配列) の差異を明示
> - REG 側の `setupUICallbacks()` / `syncUIState()` 関数化パターンに揃える

---

## 0. 全体方針

### 0.1 戦略
- DEFORM アプリは現在 UI なし (`deform/main.cpp` 337 行、コメント L5 に「UIなし。今後 RegistrationImGuiManager 等を段階的に上乗せする想定」と明記)。
- 旧 `元の長いmain.cpp` の DEFORM UI 動線を再現する。
- UI ヘッダ `RegistrationImGuiManager.h` は **すでに `state.mainMode` で REG↔DEFORM 切替できる設計** になっているため、そのまま流用する。
- DEFORM アプリは起動時に `state.mainMode = 1` 固定で起動し、UI 側はそれを見て DEFORM 専用表示になる。

### 0.2 確認済み事実 (調査結果)

#### REG 側
- **UI manager 変数名**: `RegistrationImGuiManager gUI;` (グローバル, 関数名 `gUIManager` ではない)
- **クラス名**: `RegistrationImGuiManager` (この名前のまま使用)
- **ImGui 構造**: `main.cpp` の `initOpenGL()` 末尾で `IMGUI_CHECKVERSION/CreateContext/style/font/Impl_Init` を一括処理
- **メインループパターン**: `ImGui_ImplOpenGL3_NewFrame() → ImGui_ImplGlfw_NewFrame() → ImGui::NewFrame() → syncUIState() → 描画 → gUI.draw() → ImGui::Render() → ImGui_ImplOpenGL3_RenderDrawData() → glfwSwapBuffers()`
- **`syncUIState()` 関数**: 毎フレーム呼ばれ、グローバル → UI state へ同期する。**冒頭で `s.mainMode = 0;` ハードコード** されている (REG アプリは常に REG モード)
- **`setupUICallbacks()` 関数**: actions の wire-up を集約。**多くの DEFORM 関連 actions が stub のまま**:
  ```cpp
  a.onRigidMode         = []() { std::cout << "[stub] onRigidMode" << std::endl; };
  a.onHandlePlaceMode   = []() { std::cout << "[stub] onHandlePlaceMode" << std::endl; };
  a.onDeformMode        = []() { std::cout << "[stub] onDeformMode" << std::endl; };
  a.onFullReset         = []() { std::cout << "[stub] onFullReset" << std::endl; };
  a.onHandleRadiusChanged = [](float) { std::cout << "[stub] onHandleRadiusChanged" << std::endl; };
  a.onSwitchToDeformMode = []() { std::cout << "[stub] onSwitchToDeformMode" << std::endl; };  // ← 重要
  // onStartFromDepth は実装済み(REG 内部リセット動線)
  ```
- **`MainMode` enum**: REG 側に**存在しない**。`DeformGlobals.h` のみに定義
- **`AppMode` enum**: REG 内部の状態管理用 (`kEmpty/kImageOnly/kMaskSelection/kRegistration`)。`MainMode` とは無関係
- **alpha 管理**: `g_meshAlpha[8]` グローバル配列 (0-5: organs, 6: board, 7: screenMesh)

#### DEFORM 側
- **MainMode enum**: `DeformGlobals.h` に定義 (`REGISTRATION_MODE`, `DEFORM_MODE`)
- **`currentMainMode`**: グローバル inline 変数、default = `DEFORM_MODE`
- **alpha 管理**: `gOrganAlpha`/`gBoardAlpha`/`gTargetAlpha` の単一値 3 つ
- **DeformPipeline::onKey()**: 既に R/H/D/V/T/B/1-7/0/P/Bksp/N/-/= キーをすべて処理
- **AR モード**: `static AR::Background g_arBg; static bool g_arMode = false;` で実装済み
- **現状**: 完全に ImGui 抜き、すべてキーボード操作

### 0.3 統合容易性のための共通化

統合時の二重初期化を未然防止するため、以下を common/src へ移動:

| 移動対象 | 移動先 | 理由 |
|---|---|---|
| ImGui 初期化/終了/フレーム制御 | `common/src/AppImGuiBoot.h` (新規) | 両 main.cpp で重複コードになる |
| `RegistrationImGuiManager.h` | `common/src/RegistrationImGuiManager.h` (移動のみ) | UI を共通利用するため。**名前変更しない** |

**やらないこと**:
- `MainMode` の共通化 (Step B 削除): REG 側に同 enum が存在しないため ODR 違反は発生しない。Phase 3 統合時に対応
- クラス名 `RegistrationImGuiManager` → `AppImGuiManager` の rename: REG 側 main.cpp を触らずに済ませるため

### 0.4 やらないこと (Phase 1 スコープ外)

- **AutoDeform Step 1-5 の UI 化**: キーボード操作のまま据え置き。将来 DebugPanel に統合
- **HEMI drive / Preset / Inspect の UI 化**: 同上
- **alpha 管理の統一**: REG `g_meshAlpha[8]` ↔ DEFORM 単一値 3 個。DEFORM 側内部の運用に合わせる
- **Phase 3 統合作業そのもの**: 本計画書は Phase 1 (UI 導入) + Phase 2 (subprocess 起動) のみ

---

## 1. ファイル変更マトリクス

| # | ファイル | 変更種別 | 行数目安 |
|---|---|---|---|
| 1 | `common/src/PlatformCompat.h` | 追記 | +18 行 |
| 2 | `common/src/PathConfig.h` | 追記 | +28 行 |
| 3 | `common/src/AppImGuiBoot.h` | **新規** | +60 行 |
| 4 | `registration/src/RegistrationImGuiManager.h` | **移動 → `common/src/RegistrationImGuiManager.h`** (名前変更なし) | 0 (移動のみ) |
| 5 | `registration/CMakeLists.txt` | include path 確認 | 0-3 行 |
| 6 | `deform/CMakeLists.txt` | ImGui 追加 + include path 確認 | +5-10 行 |
| 7 | `deform/main.cpp` | 大幅追記 (ImGui 導入 + actions wire-up) | +200 行 |
| 8 | `registration/main.cpp` | `setupUICallbacks()` 内の `onSwitchToDeformMode` を stub から実装に置換 | ±15 行 |

**注意点**:
- ファイル `RegistrationImGuiManager.h` は **移動するだけ** で中身は無修正。REG 側 main.cpp の `#include "RegistrationImGuiManager.h"` も無修正でよい (include path に common/src があれば解決する)
- DEFORM 側 `DeformGlobals.h` の `MainMode` enum はそのまま据え置き

---

## 2. Step A: subprocess 起動基盤 (5 分)

### 2.1 `common/src/PlatformCompat.h` への追記

**追記位置**: ファイル末尾 (最後の `#endif` の後)

**追記内容**:
```cpp

// =============================================================================
// Sibling-process launch (detached). Used by REG → DEFORM and DEFORM → REG
// transition buttons. Returns std::system's return code (informational only;
// the caller typically closes its own GLFW window right after).
// =============================================================================
#include <cstdlib>   // std::system

inline int platform_launch_detached(const std::string& exePath) {
#ifdef _WIN32
    // start "" /B = タイトル空・別ウィンドウなしでバックグラウンド起動
    //   "" は start のタイトル引数プレースホルダ(これが無いと exePath が
    //   タイトル扱いになる罠を回避)
    std::string cmd = "start \"\" /B \"" + exePath + "\"";
#else
    // 末尾 & で fork-and-detach(親 shell は子を待たない)
    std::string cmd = exePath + " &";
#endif
    return std::system(cmd.c_str());
}
```

### 2.2 `common/src/PathConfig.h` への追記

#### (a) 宣言部 (L17-22 の `DEPTH_EXE_PATH` ブロックを以下で置換):

```cpp
#ifdef _WIN32
inline std::string DEPTH_EXE_PATH   = "./sam2_da3_lite.exe";
inline std::string REG_EXE_PATH     = "./lsn_registration.exe";
inline std::string DEFORM_EXE_PATH  = "./lsn_deform.exe";
#else
inline std::string DEPTH_EXE_PATH   = "./sam2_da3_lite";
inline std::string REG_EXE_PATH     = "./lsn_registration";
inline std::string DEFORM_EXE_PATH  = "./lsn_deform";
#endif
```

#### (b) `initPaths()` 内、`DEPTH_EXE_PATH = findExe(...)` 直後に追記:

```cpp
    REG_EXE_PATH = findExe("REG_EXE_PATH", {
#ifdef _WIN32
        "./lsn_registration.exe",
        "lsn_registration.exe",
#else
        "./lsn_registration",
        "lsn_registration",
#endif
    });

    DEFORM_EXE_PATH = findExe("DEFORM_EXE_PATH", {
#ifdef _WIN32
        "./lsn_deform.exe",
        "lsn_deform.exe",
#else
        "./lsn_deform",
        "lsn_deform",
#endif
    });
```

#### (c) `initPaths()` 末尾のサマリログに追記:

`std::cout << "  DEPTH_EXE_PATH:   " << DEPTH_EXE_PATH << std::endl;` の直後:

```cpp
    std::cout << "  REG_EXE_PATH:     " << REG_EXE_PATH << std::endl;
    std::cout << "  DEFORM_EXE_PATH:  " << DEFORM_EXE_PATH << std::endl;
```

### 2.3 検証

両アプリビルド後、起動して標準出力に以下が出ることを確認:
```
[Path] REG_EXE_PATH (auto): ./lsn_registration
[Path] DEFORM_EXE_PATH (auto): ./lsn_deform
```

`NOT FOUND` が出た場合は CMake の `RUNTIME_OUTPUT_DIRECTORY` が両ターゲットで揃っていない可能性 → Step 5/6 で確認。

---

## 3. Step B: AppImGuiBoot.h 共通化 (30 分)

### 3.1 `common/src/AppImGuiBoot.h` を新規作成

**全文**:
```cpp
// AppImGuiBoot.h
// ImGui コンテキストのライフサイクルを 1 ヶ所に集約。
// REG / DEFORM どちらの main.cpp でも同じ関数を呼ぶことで、
// Phase 3 統合時の二重初期化を回避する。
//
// REG 側 main.cpp の initOpenGL() 末尾 (L〜) にあった以下と同等:
//   IMGUI_CHECKVERSION(); ImGui::CreateContext();
//   ImGuiIO& io = ImGui::GetIO();
//   io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
//   ImGui::StyleColorsDark();
//   style.WindowRounding=0, FrameRounding=4, GrabRounding=3, ScrollbarRounding=3
//   style.Colors[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.075f, 0.094f, 1.0f);
//   フォント読み込み (4 候補)
//   ImGui_ImplGlfw_InitForOpenGL(gWindow, true);
//   ImGui_ImplOpenGL3_Init("#version 330");
#pragma once

#include <cstdio>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

namespace AppImGuiBoot {

inline void loadFont() {
    ImGuiIO& io = ImGui::GetIO();
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

inline void init(GLFWwindow* win) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 0.0f;
    style.FrameRounding     = 4.0f;
    style.GrabRounding      = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.075f, 0.094f, 1.0f);

    loadFont();

    ImGui_ImplGlfw_InitForOpenGL(win, true);  // true = install chained callbacks
    ImGui_ImplOpenGL3_Init("#version 330");
}

inline void shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

inline void beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

inline void endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace AppImGuiBoot
```

### 3.2 (オプション) REG 側も AppImGuiBoot を使うように移行

**今回は触らない** が、Phase 3 統合に向けて REG 側も将来こちらに揃えるのが望ましい。REG 側の既存 ImGui 初期化コードはそのまま残す (動作確認済みの実績がある)。

### 3.3 検証

ファイル単独ではコンパイル不可 (使う側が必要)。次の Step で使用するため、ここでは作成のみ。

---

## 4. Step C: RegistrationImGuiManager.h を common 化 (5 分)

### 4.1 ファイル移動 (中身無修正)

```bash
mv registration/src/RegistrationImGuiManager.h common/src/RegistrationImGuiManager.h
```

**クラス名 `RegistrationImGuiManager` は変更しない**。REG 側 main.cpp の `#include "RegistrationImGuiManager.h"` と `RegistrationImGuiManager gUI;` がそのまま動作する。

### 4.2 include path 確認

`registration/CMakeLists.txt` の `target_include_directories` に `common/src` が含まれていることを確認 (PROJECT_STRUCTURE.md L37 によれば既に通っているはず)。

含まれていなければ追加:
```cmake
target_include_directories(lsn_registration PRIVATE
    ${CMAKE_SOURCE_DIR}/common/src
    ${CMAKE_SOURCE_DIR}/registration/src
    # ...
)
```

### 4.3 QCR extern の扱い

`RegistrationImGuiManager.h` L17-20 の以下:
```cpp
extern int   g_qcrSubsetK;
extern int   g_qcrMaxTrials;
extern float g_qcrMaxAxisRotDeg;
extern float g_qcrMaxTotalRotDeg;
```

これは `drawRegistrationSection` 内の Tuning & Advanced ヘッダで使われるが、`state.mainMode == 1` (DEFORM 時) は L1528 で短絡 return するため**実行時には到達しない**。

しかし**コンパイル時には extern 解決が必要**。DEFORM 側 main.cpp に stub 定義を追加して解決する (Step 5 で記載)。

### 4.4 検証

REG アプリだけビルドして起動。UI が以前と同一に表示されることを確認:
```bash
cd build/Desktop-Release && cmake --build . --target lsn_registration
./bin/lsn_registration
```

---

## 5. Step D: deform/main.cpp に UI 導入 (2-3 時間、最大の作業)

### 5.1 include 追加

`deform/main.cpp` の冒頭、既存の `#include "AR.h"` (L46) の直後に追加:

```cpp
// ImGui (DEFORM UI 統合)
#include "imgui.h"
#include "AppImGuiBoot.h"
#include "RegistrationImGuiManager.h"
```

### 5.2 グローバル UI manager 宣言

L62 (FullSphereCamera OrbitCam; の直後) に追加:

```cpp
// UI manager (REG 側と同じクラス名で統一。state.mainMode=1 にすることで DEFORM 専用表示)
RegistrationImGuiManager gUI;
```

### 5.3 QCR extern の stub 定義

L74 (g_arMode 宣言の直後) に追加:

```cpp
// =============================================================================
// QCR Tuning globals (REG-only feature).
// DEFORM mode では UI 側で state.mainMode==1 によって到達しないが、
// RegistrationImGuiManager.h の extern 宣言を満たすため stub 定義を提供する。
// 値は使われない。
// =============================================================================
int   g_qcrSubsetK       = 3;
int   g_qcrMaxTrials     = 100000;
float g_qcrMaxAxisRotDeg = 45.0f;
float g_qcrMaxTotalRotDeg = 90.0f;
```

### 5.4 GLFW コールバックを mode 分岐に変更

**現状の `onKey` (L85-102) を以下で置換**:

```cpp
static void onKey(GLFWwindow* win, int key, int sc, int action, int mods) {
    // ImGui がキー入力を消費している場合は何もしない (UI フォーカス時)
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) return;

    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        return;
    }
    // KeyA: AR背景オーバーレイ切替 (mode に依存しないグローバル機能)
    if (key == GLFW_KEY_A) {
        g_arMode = !g_arMode;
        std::cout << "[AR] background overlay: "
                  << (g_arMode ? "ON" : "OFF") << std::endl;
        return;
    }

    // === mode 分岐 (Phase 3 統合時はここに REG 用 case を追加するだけ) ===
    switch (currentMainMode) {
    case DEFORM_MODE:
        DeformPipeline::onKey(key, mods);
        break;
    case REGISTRATION_MODE:
        // Phase 3 統合時に追加
        break;
    }
}
```

### 5.5 マウスコールバックにも ImGui ガードを追加

**`onMouseButton` (L104-108) を以下で置換**:
```cpp
static void onMouseButton(GLFWwindow* win, int button, int action, int) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    double x, y; glfwGetCursorPos(win, &x, &y);
    if (action == GLFW_PRESS)        DeformPipeline::onMousePress(button, x, y);
    else if (action == GLFW_RELEASE) DeformPipeline::onMouseRelease(button);
}
```

**`onMouseMove` (L110-128) の冒頭に追加** (既存の `static glm::vec2 last(0.0f);` の前):
```cpp
static void onMouseMove(GLFWwindow* win, double x, double y) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        // それでも last は更新しないと次回の dx/dy が飛ぶ
        return;
    }
    static glm::vec2 last(0.0f);
    // ... (既存の処理続き)
```

**`onMouseScroll` (L130-132) を以下で置換**:
```cpp
static void onMouseScroll(GLFWwindow*, double dx, double dy) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    OrbitCam.Zoom(dy);
}
```

### 5.6 ImGui ライフサイクル統合

#### (a) `initOpenGL()` (L166-192) を一切変更しない。
ImGui 初期化は `main()` 内で OrbitCam セットアップ後に行う (REG 側のパターンに揃える)。

#### (b) `main()` 内、L235 の `g_arBg.initGL();` の **後** (L240 あたり) に ImGui 初期化を追加:

```cpp
    // === ImGui 初期化 (REG 側と完全同等。AppImGuiBoot 経由) ===
    AppImGuiBoot::init(gWindow);

    // === DEFORM アプリは常に DEFORM フェーズで起動 ===
    // UI 側は state.mainMode==1 を見て DEFORM 専用表示に切り替わる:
    //   - drawDepthSection:     compact summary 表示 (intrinsics source のみ)
    //   - drawRegistrationSection: "Registration: Done" のみ
    //   - drawDeformSection:    通常表示 (Rigid/Handle/Deform サブモード)
    //   - drawVisibility:       通常表示
    currentMainMode = DEFORM_MODE;   // 既に default だが明示
    gUI.state.mainMode    = 1;       // 1 = DEFORM
    gUI.state.depthDone   = true;    // depth 完了扱い
    gUI.state.regState    = 4;       // reg 完了扱い (4 = REGISTERED)
    gUI.state.maxHandleGroups = SoftBody::MAX_HANDLE_GROUPS;  // = 5 (UI default は 4)

    // === UI Actions wire-up ===
    setupUICallbacks();
```

#### (c) `setupUICallbacks()` 関数を `main()` の **前** に追加 (REG 側パターンに揃える)。
内容は次の Section 5.7 参照。

#### (d) main loop 内 (L298-331) を以下で置換:

```cpp
    // メインループ
    double last = glfwGetTime();
    while (!glfwWindowShouldClose(gWindow)) {
        double now = glfwGetTime();
        float  dt  = (float)(now - last);
        last = now;

        showFPS(gWindow);
        glfwPollEvents();

        // === ImGui フレーム開始 ===
        AppImGuiBoot::beginFrame();

        // === UI 状態同期 ===
        syncUIState();

        // liver / texture の中心をカメラに通知
        DeformPipeline::updateCameraTargets(OrbitCam);
        OrbitCam.UpdateCamera(dt);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // === AR モード ===
        if (g_arMode) {
            view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f));
            g_arBg.draw();
        }

        // === メインシーン描画 ===
        DeformPipeline::updateAndDraw(
            dt, shaderProgram, shaderProgramCube,
            view, projection, OrbitCam.cameraPos);

        // === UI 描画 ===
        gUI.draw(gWindowWidth, gWindowHeight);

        // === ImGui フレーム終了 + バッファスワップ ===
        AppImGuiBoot::endFrame();
        glfwSwapBuffers(gWindow);
    }

    // === シャットダウン ===
    AppImGuiBoot::shutdown();
    DeformPipeline::cleanup();
    glfwDestroyWindow(gWindow);
    glfwTerminate();
    return 0;
}
```

#### (e) `syncUIState()` 関数を `main()` の **前** に追加。内容は Section 5.8 参照。

### 5.7 setupUICallbacks() の中身

`main()` 関数の **直前** に以下を追加:

```cpp
// ============================================================================
// UI Actions wire-up
//   DEFORM mode で実際に押されるボタンのみ wire。それ以外は nullptr のまま
//   (UI 側で if(actions.xxx) チェック後に呼ぶので落ちない)。
//   旧 元の長いmain.cpp L3573-3604 のロジックをそのまま移植。
// ============================================================================
static void setupUICallbacks() {
    auto& a = gUI.actions;

    // --- DEFORM サブモード切替 ---
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

    // --- Reset All ---
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

    // --- Sphere Radius slider ---
    a.onHandleRadiusChanged = [](float r) {
        gGroupRadius = r;
    };

    // --- Visibility toggle ---
    //   REG 側は g_meshAlpha[0..7] 配列。DEFORM 側は単一値 3 個 (gOrganAlpha/
    //   gBoardAlpha/gTargetAlpha) なので、個別 organ 切替えはできない (全臓器一括)。
    //
    //   i=0..5: organ index → 全 organ 一括で gOrganAlpha cycle
    //   i=6:    Board (gBoardAlpha cycle)
    //   i=7:    Target (gTargetAlpha cycle)
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

    // --- AR snapshot save (DEFORM 側に AR snapshot 未実装、no-op) ---
    a.onSaveAR = []{
        std::cout << "[UI] Save AR Image (not yet implemented in DEFORM)" << std::endl;
    };

    // --- Camera reset ---
    a.onResetCamera = []{
        if (gOrbitCamPtr) {
            gOrbitCamPtr->resetToInitialState();
            std::cout << "[UI] Camera reset" << std::endl;
        }
    };

    // --- Start From Depth (DEFORM → REG プロセス起動) ---
    a.onStartFromDepth = []{
        std::cout << "[Deform] Restart from Depth -> spawning lsn_registration" << std::endl;
        platform_launch_detached(REG_EXE_PATH);
        glfwSetWindowShouldClose(gWindow, GLFW_TRUE);
    };

    // --- Switch to Deform (DEFORM 内では既に DEFORM mode なので no-op) ---
    a.onSwitchToDeformMode = []{
        std::cout << "[UI] Already in DEFORM mode" << std::endl;
    };

    // --- 以下は DEFORM mode で UI 側が短絡 return するため呼ばれない ---
    // (UI 側で if(actions.xxx) チェック済みなので null のまま放置で OK)
    //   a.onRunDepth, a.onSegment1, a.onSegment2, a.onLoadLocalImage,
    //   a.onResetDefaultImage, a.onUndoSegPoint, a.onDepthScaleChanged,
    //   a.onFullAuto, a.onBipopCmaes, a.onHemiAuto, a.onQuadAuto,
    //   a.onQuadCyclic, a.onQuadCyclicRansac, a.onHemiVoxelChanged,
    //   a.onStartUmeyama, a.onExecuteUmeyama, a.onResetRegistration,
    //   a.onClearPoints, a.onUndoUmeyamaPoint, a.onToggleClusterVis,
    //   a.onToggleCorrespondenceVis, a.onRefine, a.onSilhouetteAlign,
    //   a.onCtrlG, a.onCtrlgLockScaleChanged, a.onPoseLibraryToggle,
    //   a.onPoseUndo, a.onAutoProbe, a.onIterativeAutoProbe, a.onAutoQCR,
    //   a.onSwitchDepthModel, a.onInitRotPresetChanged, a.onInitRotPresetSilent,
    //   a.onInitRotPositionChanged, a.onIntrinsicsSourceChanged,
    //   a.onRunCalibration, a.onSourceChanged, a.onPresetChanged,
    //   a.onSaveAsCustom, a.onChessboardFolderChanged, a.onBoardSizeChanged,
    //   a.onSquareSizeChanged, a.onInstrumentPxThreshChanged,
    //   a.onDetectVignetteChanged, a.onUseCudaChanged, a.onQuadrantMaskChanged,
    //   a.onApplyInitPose, a.onFlipLR, a.onFlipCC, a.onRecomputeAxes,
    //   a.onDrawAdvancedCtrlG, a.onLiveCalibCapture, a.onLiveCalibRun,
    //   a.onToggleCamera, a.onCameraBack, a.onExportStl, a.onExportStlFlipped
}
```

### 5.8 syncUIState() の中身

`setupUICallbacks()` の直後に追加:

```cpp
// ============================================================================
// UI State 同期 (グローバル → state)
//   REG 側は毎フレーム mainMode=0 をハードコードしている。DEFORM 側は逆に
//   mainMode=1 を維持する。それ以外は DEFORM 関連の最小限を同期。
//   REG 関連の state field (regState, depthRunning, cameraState, …) は
//   起動時に初期化したまま据え置き (UI 側で短絡 return される)。
// ============================================================================
static void syncUIState() {
    auto& s = gUI.state;
    s.mainMode  = 1;            // 1 = DEFORM (常時)
    s.depthDone = true;         // depth 完了扱い (REG セクションを compact 表示)
    s.regState  = 4;            // reg 完了扱い (DEFORM セクションを active 化)

    // DEFORM サブモード
    s.deformState     = (int)deformHandlPlace.state;
    s.handleGroups    = multiBody ? (int)multiBody->handleGroups.size() : 0;
    s.maxHandleGroups = SoftBody::MAX_HANDLE_GROUPS;
    s.handleRadius    = gGroupRadius;

    // Visibility alpha (キーボード V/T/B から変更されることがあるので毎フレーム同期)
    // REG 側は state.organs[6] が個別配列だが、DEFORM 側は全 organ 一括 (gOrganAlpha)
    // なので 6 個ともこの値で埋める
    for (int k = 0; k < 6; k++) {
        s.organs[k].alpha = gOrganAlpha;
    }
    s.boardAlpha  = gBoardAlpha;
    s.targetAlpha = gTargetAlpha;

    // arSavedTimer のデクリメント (REG 側と同じ処理。Save AR Image ボタンが
    // 押されたあと "Saved!" 表示が 2 秒間出る)
    if (s.arSavedTimer > 0) s.arSavedTimer -= ImGui::GetIO().DeltaTime;
}
```

### 5.9 検証

DEFORM アプリビルド + 起動:

```bash
cd build/Desktop-Release && cmake --build . --target lsn_deform
./bin/lsn_deform
```

確認事項:
1. ウィンドウ右側にサイドバーが表示される (幅 ~400px)
2. 「DEPTH GENERATION」セクションは compact (intrinsics source のみ表示)
3. 「REGISTRATION」セクションは "Registration: Done" のみ
4. 「DEFORM」セクションが Rigid / Handle / Deform の 3 ボタンを表示
5. 「VISIBILITY」セクションが organ 6 つ + Board + Target ボタンを表示
6. Rigid / Handle / Deform ボタン押下でモードが切り替わる (FPS タイトルバー表示でも確認可能)
7. キーボード R / H / D も従来通り動作
8. Visibility ボタン押下で gOrganAlpha が cycle (全 organ 一括) → 6 ボタン全部 ON/50%/OFF が同期表示
9. Board / Target ボタン押下で gBoardAlpha / gTargetAlpha が cycle
10. キーボード V/T/B も従来通り動作 (UI と同期する)
11. Sphere Radius スライダで gGroupRadius が変化
12. 「Reset All」ボタンで Full Reset (state RIGID_MODE→HANDLE_PLACE_MODE 経由)
13. 「Start From Depth」ボタン押下 → 確認ポップアップ「Restart from Depth?」 → 確認すると `lsn_registration` が起動 + DEFORM アプリは終了

---

## 6. Step E: REG 側 onSwitchToDeformMode 実装 (10 分)

### 6.1 registration/main.cpp の修正

`setupUICallbacks()` 関数内の現在の stub:
```cpp
a.onSwitchToDeformMode = []() {
    std::cout << "[stub] onSwitchToDeformMode" << std::endl;
};
```

を以下で **置換** (stub を実装に置き換え):

```cpp
a.onSwitchToDeformMode = []() {
    std::cout << "[Reg] Switch to Deform -> exporting reg_*.obj + spawning lsn_deform" << std::endl;

    // === reg_*.obj 出力 (旧 元の長いmain.cpp L3619-3625 と同等) ===
    // DEFORM 側 DeformPipeline::initFromRegistered() がこれらを読む
    std::filesystem::create_directories(REG_MODEL_PATH);
    if (liverMesh3D)   liverMesh3D->exportObjFile(Reg_TARGET_FILE_PATH);
    if (portalMesh3D)  portalMesh3D->exportObjFile(Reg_PORTAL_FILE_PATH);
    if (veinMesh3D)    veinMesh3D->exportObjFile(Reg_VEIN_FILE_PATH);
    if (tumorMesh3D)   tumorMesh3D->exportObjFile(Reg_TUMOR_FILE_PATH);
    if (segmentMesh3D) segmentMesh3D->exportObjFile(Reg_SEGMENT_FILE_PATH);
    if (gbMesh3D)      gbMesh3D->exportObjFile(Reg_GB_FILE_PATH);

    // === Phase 2: 分離プロセス起動 ===
    // (Phase 3 統合時は platform_launch_detached を削除し、
    //  currentMainMode = DEFORM_MODE に戻す)
    platform_launch_detached(DEFORM_EXE_PATH);
    glfwSetWindowShouldClose(gWindow, GLFW_TRUE);
};
```

### 6.2 include 確認

`registration/main.cpp` で `platform_launch_detached` が解決できる必要があるが、`PathConfig.h` → `DepthRunner.h` → `PlatformCompat.h` の include chain で transitively 通っているはず。

確実にしたければ `#include "PlatformCompat.h"` を明示追加 (該当ヘッダの include は冪等なので安全)。

### 6.3 検証

REG アプリ起動 → 通常通り depth → registration → 「Deform >>」ボタンを押下:
1. ログ `[Reg] Switch to Deform -> exporting reg_*.obj + spawning lsn_deform` が出る
2. `registration_model/reg_*.obj` 群がディレクトリに出力される
3. `lsn_deform` がバックグラウンド起動する
4. REG アプリは自分のウィンドウを閉じる
5. 数秒後 DEFORM アプリのウィンドウが立ち上がる + サイドバー表示

---

## 7. CMakeLists.txt の確認・調整

### 7.1 RUNTIME_OUTPUT_DIRECTORY

ルート `CMakeLists.txt` (または各サブプロジェクトの CMakeLists.txt) で両ターゲットが同じディレクトリに出力されていることを確認:

```cmake
set_target_properties(lsn_registration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(lsn_deform PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
```

なければ追加。同じディレクトリにいないと `platform_launch_detached("./lsn_xxx")` が見つけられない。

### 7.2 deform/CMakeLists.txt の ImGui 追加

`lsn_deform` ターゲットに ImGui のソースをリンクする。REG 側に既存設定があるはずなので、それをコピー:

```cmake
# REG 側で動いている設定をコピー (推測。実際の設定を確認すること)
target_sources(lsn_deform PRIVATE
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(lsn_deform PRIVATE
    ${CMAKE_SOURCE_DIR}/common/src
    ${CMAKE_SOURCE_DIR}/deform/src
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
```

### 7.3 Phase 3 統合用 placeholder コメント

ルート CMakeLists.txt 末尾あたりに、将来用のコメントを残す:

```cmake
# =============================================================================
# [Phase 3 統合用 placeholder]
# 統合時はここに lsn_unified ターゲットを追加する。
# - main.cpp は registration/main.cpp と deform/main.cpp を統合した app/main.cpp
# - include path は common/src + registration/src + deform/src
# - 既存の lsn_registration / lsn_deform は当面残してもよい (実験用)
# add_executable(lsn_unified app/main.cpp ${COMMON_SOURCES} ${REG_SOURCES} ${DEFORM_SOURCES})
# =============================================================================
```

---

## 8. 動作確認手順 (全 Step 完了後)

### 8.1 ビルド

```bash
cd build/Desktop-Release
cmake --build . --target lsn_registration lsn_deform -j
```

エラーゼロを確認。

### 8.2 REG アプリ単体動作

```bash
cd bin && ./lsn_registration
```

確認:
- 起動・UI 表示・カメラ動作が変更前と同一
- depth → registration の動線が動く
- 「Deform >>」ボタンを押すと:
  - `[Reg] Switch to Deform -> exporting reg_*.obj + spawning lsn_deform` ログ
  - `registration_model/reg_*.obj` が出力される
  - `lsn_deform` が起動して REG は閉じる

### 8.3 DEFORM アプリ単体動作

`reg_*.obj` が `registration_model/` に存在することを前提に:

```bash
cd bin && ./lsn_deform
```

確認:
- UI サイドバーが表示される
- DEFORM セクションの Rigid / Handle / Deform ボタンが動く
- VISIBILITY セクションの各ボタンが動く (キーボード V/T/B と同期する)
- 「Start From Depth」で `lsn_registration` が起動して自分は閉じる
- 既存のキーボード操作 (R/H/D/V/T/B/1-7/0/P/A/N/-/=) が全て従来通り動く

### 8.4 双方向遷移テスト

```
lsn_registration 起動
  → depth → registration 完了
  → 「Deform >>」押下
  → lsn_deform 起動 (REG は閉じる)
  → DEFORM サブモードを試す
  → 「Start From Depth」押下 → 確認ポップアップ → "Restart"
  → lsn_registration 起動 (DEFORM は閉じる)
  → 元の状態に戻っているか確認
```

---

## 9. 既知の問題と対処

### 9.1 ImGui がマウス・キーボード入力を奪う

Step 5.5 の `WantCaptureMouse` / `WantCaptureKeyboard` チェックで対処済み。これは REG 側でも同様の対策が必要だが、`ImGui_ImplGlfw_InitForOpenGL(win, true)` の chain installation で自動的に動くので問題ない。

### 9.2 Visibility ボタンの個別 OFF/ON ができない

REG 側は `g_meshAlpha[i]` 配列で個別 organ ごとに切替可能だが、DEFORM 側は `gOrganAlpha` 単一値なので **6 個の organ ボタンを押すと全 organ が同時に切り替わる**。UI 表示も 6 個すべて同じ alpha 値で同期される。

これは Phase 1 では受容する。将来個別切替が必要なら DEFORM 側にも `gMeshAlpha[8]` 相当の配列を導入する (Phase 3 統合時に REG 側と統一する候補)。

### 9.3 「Start From Depth」のメッシュ再ロード問題

DEFORM アプリ単独起動の場合、`reg_*.obj` は既存ファイルを読むだけ。「Start From Depth」で REG プロセスに切り替わるとき、REG 側が `preReg_*.obj` を再ロードして depth phase に戻るのは **REG 側の責務**。DEFORM 側で深いリセットをする必要はない (プロセスごと作り直すので)。

### 9.4 REG 側 onStartFromDepth の重複問題

REG 側にも同名 lambda `a.onStartFromDepth` が実装済み (registration 内部リセット動線)。DEFORM 側で新規に `a.onStartFromDepth = [platform_launch_detached]` を wire するが、これは **DEFORM アプリの actions インスタンスにだけ wire される**ので、REG 側の `onStartFromDepth` 実装に影響しない (それぞれの gUI.actions は別物)。

統合時 (Phase 3) は両者の実装をマージする必要がある:
- REG mode のときは内部リセット動線
- DEFORM mode のときはメッシュ再ロード + currentMainMode = REGISTRATION_MODE

これは Phase 3 マターとして `Section 11` の TODO リストに含まれる。

### 9.5 (新) ImGui コールバック chain の二重 install リスク

`ImGui_ImplGlfw_InitForOpenGL(win, true)` の 2 番目引数 `true` は ImGui が GLFW コールバックを chain で install する意味。

DEFORM 側 `initOpenGL()` (L180-184) で先に `glfwSetKeyCallback`, `glfwSetMouseButtonCallback` 等を install したあと、ImGui が chain install することになる。

**ImGui が後勝ち + 既存コールバックを chain する** ので、UI フォーカス時は ImGui が先に処理し、その後アプリ側コールバックが呼ばれる (`WantCaptureXxx` チェックで吸収するパターン)。

Step 5.5 でこのチェックを各コールバックに追加済み。

---

## 10. Phase 3 統合時のチェックリスト (将来用、本計画書では実行しない)

```
[ ] registration/main.cpp と deform/main.cpp を 1 つに統合 (app/main.cpp)
[ ] CMakeLists.txt に lsn_unified ターゲット追加
[ ] onSwitchToDeformMode の platform_launch_detached を削除し、
    currentMainMode = DEFORM_MODE + DeformPipeline::initFromRegistered() に置換
[ ] onStartFromDepth の DEFORM 側 wire (platform_launch_detached 版) を削除し、
    旧 元の長いmain.cpp L3634-3690 のメッシュ再ロード + currentMainMode = REGISTRATION_MODE に置換
[ ] GLFW コールバックの mode 分岐 (Step 5.4 の足場) に REGISTRATION_MODE case を実装
[ ] AR モードグローバル (g_arMode in DEFORM, gApp.arMode in REG) を統一
    (DEFORM の static g_arMode を gApp.arMode に揃える)
[ ] syncUIState() の `s.mainMode = 0;` ハードコードを
    `s.mainMode = (currentMainMode == DEFORM_MODE) ? 1 : 0;` に変更
[ ] DEFORM 側 alpha 単一値 (gOrganAlpha) と REG 側 alpha 配列 (g_meshAlpha[8]) の統一
    候補: REG 側に揃える → DEFORM 側 onKey V/T/B も g_meshAlpha[0..7] 操作に変更
[ ] MainMode enum を common/src に移動 (この時点で REG 側にも必要になる)
[ ] ImGui 初期化を AppImGuiBoot 経由に統一済み → 何もしなくてよい
[ ] RegistrationImGuiManager.h は common/src に集約済み → 何もしなくてよい
[ ] DEFORM 側 actions (onRigidMode 等) の lambda 実装を REG 側の stub に上書きコピー
```

**変更箇所がほぼ main.cpp のみで済む状態** がゴール。

---

## 11. 実装順序とコミット推奨単位

| コミット | 内容 | 動作確認 |
|---|---|---|
| 1 | Step A (PathConfig + PlatformCompat) | 両アプリ起動、ログに REG_EXE_PATH / DEFORM_EXE_PATH が出ることを確認 |
| 2 | Step B (AppImGuiBoot.h 新規作成のみ) | コンパイル確認のみ |
| 3 | Step C (RegistrationImGuiManager.h を common/src に移動) | REG アプリで UI が変わらず動くことを確認 |
| 4 | Step D (deform/main.cpp に UI 導入) | DEFORM アプリで UI が出て Rigid/Handle/Deform が動くことを確認 |
| 5 | Step E (REG 側 onSwitchToDeformMode 実装) | 双方向遷移テスト (Section 8.4) |

各コミットで前 Step が壊れていないことを動作確認してから次へ。

---

## 12. 付録 A: ファイル間依存関係 (Phase 1 完了時)

```
common/src/
├── AppImGuiBoot.h                   ← ImGui ライフサイクル (新規)
├── RegistrationImGuiManager.h       ← UI 本体 (registration/src から移動。クラス名変更なし)
├── PathConfig.h                     ← REG_EXE_PATH / DEFORM_EXE_PATH 追加
├── PlatformCompat.h                 ← platform_launch_detached 追加
└── (その他 22 ファイル)

registration/src/
├── (RegistrationImGuiManager.h は削除済み)
├── RegistrationActions.h
├── (その他 27 ファイル)

deform/src/
├── DeformGlobals.h                  ← 無修正 (MainMode enum はここに残る)
├── DeformPipeline.h                 ← 無修正
├── (その他 14 ファイル)

registration/main.cpp                ← setupUICallbacks() 内の onSwitchToDeformMode を stub から実装に
deform/main.cpp                      ← ImGui 全面導入 + setupUICallbacks() + syncUIState() + QCR extern stub
```

---

## 13. 付録 B: トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `platform_launch_detached` が見つからない | PlatformCompat.h の include が解決していない | `#include "PlatformCompat.h"` を直接追加 |
| `REG_EXE_PATH NOT FOUND` | CMake の出力ディレクトリが揃っていない | Section 7.1 を確認 |
| ImGui のシンボル未解決 | CMakeLists.txt に ImGui sources が追加されていない | Section 7.2 を確認 |
| `g_qcrSubsetK` 未定義リンクエラー | DEFORM 側 main.cpp に stub 定義がない | Step 5.3 を確認 |
| `RegistrationImGuiManager.h: No such file` | common/src への移動が完了していない | Step 4.1 の `mv` を確認 |
| ボタン押下で何も起きない | actions が wire されていない、または mainMode が違う | Step 5.7 / 5.6(b) の確認、`std::cout` ログを actions lambda に入れて確認 |
| DEFORM サブモードボタンが grayed out | `state.mainMode != 1` になっている | Step 5.6(b) の `gUI.state.mainMode = 1;` が抜けていないか、また Step 5.8 の `syncUIState()` で毎フレーム 1 を維持しているか確認 |
| 「Deform >>」ボタンを押しても DEFORM が起動しない | REG 側で stub のまま | Step 6.1 で stub を新規実装に置換したか確認 |
| ImGui ウィンドウが半透明で見えにくい | アルファ設定漏れ | Step 3.1 の `ImGui::StyleColorsDark()` + `ImGuiCol_WindowBg = ImVec4(0.067f, 0.075f, 0.094f, 1.0f)` を確認 |
| マウス操作で UI とシーンの両方が反応する | WantCaptureMouse チェック漏れ | Step 5.5 のガードを各マウスコールバックに入れたか確認 |

---

**実装計画書 v2 終わり**

---

## 主な v1 → v2 変更まとめ

1. **削除**: Step B (`AppMode.h` 共通化) → 不要 (REG 側に MainMode enum がそもそも存在しない)
2. **置換**: クラス名 rename → ファイル移動のみ、クラス名・include 文の変更なし
3. **置換**: `gUIManager` → `gUI` (REG 側準拠)
4. **明示**: REG 側 `onSwitchToDeformMode` は stub → 「置換」ではなく「実装」
5. **追加**: REG 側 `setupUICallbacks()` / `syncUIState()` 関数化パターンに揃える
6. **追加**: REG 側 `g_meshAlpha[8]` vs DEFORM 側 alpha 単一値 3 個の差異と、UI で全 organ 一括切替になる仕様の明示
7. **追加**: ImGui 初期化のフォント読み込みコード (REG 側既存実装) を `AppImGuiBoot::loadFont()` に集約
8. **追加**: `state.maxHandleGroups = SoftBody::MAX_HANDLE_GROUPS` (= 5、UI default 4) の同期
9. **追加**: ImGui コールバック chain install の挙動説明 (Section 9.5)
