#pragma once

#include "imgui.h"
#include <string>
#include <functional>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "PathConfig.h"
#include "IntrinsicsSource.h"   // enum IntrinsicsSource (lightweight, no GL deps)

// Forward declarations for Shift+Ctrl+P tuning globals defined in RegistrationActions.h.
// (RegistrationActions.h is included after this header in main.cpp, so we need these
//  extern decls to compile the QCR Tuning slider panel below.)
extern int   g_qcrSubsetK;
extern int   g_qcrMaxTrials;
extern float g_qcrMaxAxisRotDeg;     // hard limit per X/Y/Z axis rotation (deg)
extern float g_qcrMaxTotalRotDeg;    // hard limit on total axis-angle rotation (deg)

struct RegUIActions {
    std::function<void()> onToggleCamera;
    std::function<void()> onCameraBack;
    std::function<void()> onRunDepth;
    // Segment-only previews. onSegment1 runs the SAM2 stage of the depth
    // pipeline and pops up a small preview of segmentation_overlay.jpg so
    // the user can sanity-check the mask before paying for depth inference.
    // onSegment2 is reserved for a future second mask (instrument / occluder
    // identification) — currently a no-op stub.
    std::function<void()> onSegment1;
    std::function<void()> onSegment2;
    std::function<void()> onResetDefaultImage;
    std::function<void()> onLoadLocalImage;
    std::function<void()> onUndoSegPoint;
    std::function<void(float)> onDepthScaleChanged;
    std::function<void()> onFullAuto;
    std::function<void()> onBipopCmaes;
    std::function<void()> onHemiAuto;
    std::function<void()> onQuadAuto;       // Shift+O: AR-fixed-view ∩ quadrant intersection
    std::function<void()> onQuadCyclic;     // Ctrl+P : AR-fixed ∩ silhouette ∩ quadrant + Umeyama cyclic
    std::function<void()> onQuadCyclicRansac; // Shift+Ctrl+P : 同前処理 + K=3 subset RANSAC (2 段階評価)
    std::function<void(float)> onHemiVoxelChanged;
    std::function<void()> onStartUmeyama;
    std::function<void()> onExecuteUmeyama;
    std::function<void()> onResetRegistration;
    std::function<void()> onClearPoints;
    std::function<void()> onUndoUmeyamaPoint;
    std::function<void()> onToggleClusterVis;
    std::function<void()> onToggleCorrespondenceVis;
    std::function<void()> onRigidMode;
    std::function<void()> onHandlePlaceMode;
    std::function<void()> onDeformMode;
    std::function<void()> onFullReset;
    std::function<void(float)> onHandleRadiusChanged;
    std::function<void()> onStartFromDepth;
    std::function<void()> onSaveAR;
    std::function<void(int)> onToggleOrgan;
    std::function<void()> onSwitchToDeformMode;
    std::function<void()> onResetCamera;
    std::function<void()> onExportStl;         // [key-reorg P12] was M (export registered OBJs)
    std::function<void()> onExportStlFlipped;  // [key-reorg P12] was Shift+M (cam-mm STL + snapshot)
    std::function<void()> onRefine;
    std::function<void()> onSilhouetteAlign;
    // ---- Ctrl+G : V3-R BIPOP-CMA-ES (Region-aware, main refinement) ----
    //   Initial Orientation panel + AutoQCR で姿勢を寄せた後の主動線。
    //   main.cpp 側で以下と等価のシーケンスを実装する:
    //     1) g_liverRegion / g_liverLR 未計算なら ERROR 出力して abort
    //     2) makeQuadrantSubsetIdx で subset を計算、空なら abort
    //     3) g_stepStartTime 更新、g_sessionBipopN++、regMethod=3
    //     4) poseAutoSaveBeforeRegistration()
    //     5) runBipopCmaesV3R(g_activeQuadrantMask)
    //     6) poseSaveToLibrary(SaveCriterion::RMSE, g_activeQuadrantMask)
    //   キーボードの Ctrl+G dispatch (main.cpp GLFW_KEY_G case) と
    //   結果 byte-identical になることが期待される。
    std::function<void()> onCtrlG;
    // onCtrlgLockScaleChanged: Ctrl+G の 6-DoF/7-DoF チェックボックス変更時。
    //   true  → main.cpp が g_ctrlgSearchMode = SIX_DOF_RIGID に
    //   false → main.cpp が g_ctrlgSearchMode = SEVEN_DOF に
    //   (4-DoF への切替は左 floating パネル経由のみ。サイドバーからは触れない)
    std::function<void(bool)> onCtrlgLockScaleChanged;
    std::function<void()> onPoseLibraryToggle;
    std::function<void()> onPoseUndo;
    std::function<void()> onAutoProbe;
    std::function<void(int)> onIterativeAutoProbe;  // (K) — calls runAutoProbe() K times
    // AutoQCR (Alt+Ctrl+P): 9-preset 自動 sweep。bool 引数は lock_scale
    // (true=6-DoF rigid、false=7-DoF T+R+S)。RegUIState::autoQcrLockScale
    // から渡る。チェックボックス default ON = 6-DoF (論文推奨、CMA-ES 発散回避)。
    std::function<void(bool)> onAutoQCR;
    std::function<void(int)> onSwitchDepthModel;
    std::function<void(int)> onInitRotPresetChanged;
    // onInitRotPresetSilent: registrationHandle.initRotPreset を更新するが
    //   applyInitRotation は呼ばない。POSITION 変更時の自動 Orient 設定で使う:
    //   ユーザがクアドラントを選んだ時点で対応する Orient (Right/Base/Left)
    //   に切り替わるが、Apply Init Pose を押すまで実適用しない。
    std::function<void(int)> onInitRotPresetSilent;
    std::function<void(int)> onInitRotPositionChanged;   // Phase 2: 重心位置 selector (legacy, deprecated)
    std::function<void(int)> onIntrinsicsSourceChanged;   // legacy 4-button (unused by Step 7 UI)
    std::function<void()>    onRunCalibration;
    // ---- Step 7: intrinsics source dropdown (案 Y) ----
    //   onSourceChanged : Custom / Calib / Auto を選んだとき。main.cpp が
    //     g_intrinsicsSource を更新し loadIntrinsicsFromCurrentSource()。
    //   onPresetChanged : ドロップダウンでプリセットを選んだとき (key 指定)。
    //     main.cpp が source=Preset + g_currentPresetKey=key にして load。
    //   onSaveAsCustom  : 現在の K を intrinsics_custom.txt に保存し source=Custom。
    //   onChessboardFolderChanged / onBoardSizeChanged / onSquareSizeChanged :
    //     Settings タブの Calibration パラメータ編集。
    std::function<void(IntrinsicsSource)>   onSourceChanged;
    std::function<void(const std::string&)> onPresetChanged;
    std::function<void()>                    onSaveAsCustom;
    std::function<void(const std::string&)> onChessboardFolderChanged;
    std::function<void(int,int)>             onBoardSizeChanged;
    std::function<void(float)>               onSquareSizeChanged;
    std::function<void(float)> onInstrumentPxThreshChanged;   // ★追加
    // Vignette auto-detection toggle. Called when the checkbox in the
    // DEPTH GENERATION section is toggled. main.cpp side updates
    // gApp.detectVignette which is consulted when building the next
    // DepthRunner config (Instrument preview or Run Depth).
    std::function<void(bool)> onDetectVignetteChanged;
    // CUDA / GPU toggle. main.cpp 側で gApp.useCuda を更新し、次回の
    // Run Depth / Instrument preview の CLI に --cuda を付与する。
    // sam2_da3_lite が USE_CUDA=OFF でビルドされている場合は CPU
    // fallback されるため、ON でも害はない。
    std::function<void(bool)> onUseCudaChanged;
    // ---- チャット 9: 4-quadrant 連動 (Initial Orientation = Ctrl+G mask) ----
    //   onQuadrantMaskChanged: 2x2 grid checkbox or Quick Preset で
    //     mask が変更されたとき呼ばれる。main.cpp 側で g_activeQuadrantMask を
    //     更新する (副作用なし — apply は別ボタン)。
    //   onApplyInitPose: Apply Init Pose ボタン押下時。main.cpp 側で
    //     applyInitRotation(startNewSession=true) を呼ぶ。
    std::function<void(uint8_t)> onQuadrantMaskChanged;
    std::function<void()>        onApplyInitPose;

    // ---- Anatomical Axes Status (Preview OBJ Anatomical Pose) ----
    //   Flip ボタン押下 → main.cpp 側で g_lrFlipManual / g_ccFlipManual を
    //   トグルし、それぞれ recomputeLiverLR() / recomputeLiverCC() を実行。
    //   LR の sign 反転は CC の計算結果にも影響するので、onFlipLR は内部で
    //   CC も再計算する (main.cpp 側で連鎖呼出)。
    std::function<void()> onFlipLR;
    std::function<void()> onFlipCC;
    // (Reserved) 全ラベルを強制再計算するボタン。Phase 1 では未配線。
    std::function<void()> onRecomputeAxes;

    // ---- Advanced section integration ----
    //   onDrawAdvancedCtrlG:
    //     REGISTRATION セクションの "Advanced" CollapsingHeader 内で呼ばれる。
    //     main.cpp 側で Ctrl+G / Ctrl+Shift+G の RIM-weighted / raycast 関連
    //     コントロール (g_ctrlgUseArVisFilter, g_ctrlgBetaRimWeight,
    //     g_ctrlgsLambdaRimSil 等) を ImGui 描画する。callback 方式により
    //     この header に extern decl を 10 個以上撒くのを回避。
    //
    //     既存の floating "Ctrl+G Quadrant Selector" 窓と同じ globals を
    //     操作するので、片方を変えればもう片方の表示も即時同期する。
    std::function<void()> onDrawAdvancedCtrlG;
};

struct RegUIState {
    int mainMode = 0;
    int cameraState = 0;
    bool depthRunning = false;
    bool depthDone = false;
    float depthScale = 0.3f;
    int segFG = 0, segBG = 0;
    // Per-mask point counts. segFG/segBG above are kept for the Liver
    // mask so existing layout code that just shows "FG: %d  BG: %d" stays
    // backwards compatible. instSegFG/instSegBG are the same numbers for
    // the Instrument mask, displayed alongside on a second row when the
    // user has any instrument prompts.
    int instSegFG = 0, instSegBG = 0;

    // Which mask is currently being edited (mirrors AppContext::activeMaskKind).
    // 0 = Liver, 1 = Instrument. main.cpp's syncUIState sets this each frame.
    // Used by the button row to highlight the active button and by the
    // legend hint text to switch between green/red and cyan/orange.
    int activeMaskKind = 0;
    bool hasLocalImage = false;
    std::string localImageName;

    // ---- Segmentation preview popup (Segment 1 button) ----
    // Owned by main.cpp: runSegmentOnly() loads segmentation_overlay.jpg into
    // an OpenGL texture, fills these fields, and sets segPreviewOpen=true to
    // raise the popup. The UI code in this header just renders the popup;
    // it does not allocate or free the texture.
    unsigned int segPreviewTexId = 0;   // GL texture id (0 = none)
    int   segPreviewW = 0, segPreviewH = 0;
    bool  segPreviewOpen = false;       // set by runSegmentOnly, cleared on close
    float segPreviewScore = 0.0f;       // SAM2 IoU prediction (0 = not set)
    int   segPreviewFgPixels = 0;       // foreground pixel count (0 = not set)

    int regState = 0;
    int regMethod = -1;
    bool refineEnabled = false;
    bool poseLibraryOpen = false;
    bool poseUndoAvailable = false;
    int  poseEntryCount = 0;
    int  depthModelIdx = 0;
    bool depthModelAvail[3] = {false, false, false};
    int boardPtCount = 0, objPtCount = 0, targetPtCount = 5;
    bool splitScreen = false;
    bool depthSplitScreen = false;

    float regMatrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float avgError = 0.0f;
    float rmse = 0.0f;
    float maxError = 0.0f;
    float scaleFactor = 1.0f;
    float modelBBoxDiag = 1.0f;
    bool useRegistration = false;

    int deformState = 0;
    int handleGroups = 0, maxHandleGroups = 4;
    float handleRadius = 0.5f;

    struct OrganInfo { const char* name; float alpha; ImVec4 color; };
    OrganInfo organs[6] = {
        {"Liver",   0.8f, {0.95f,0.30f,0.25f,1}},
        {"Portal",  0.9f, {0.95f,0.40f,0.70f,1}},
        {"Vein",    0.9f, {0.30f,0.85f,0.95f,1}},
        {"Tumor",   0.5f, {0.90f,0.65f,0.15f,1}},
        {"Segment", 0.5f, {0.92f,0.82f,0.20f,1}},
        {"GB",      0.7f, {0.30f,0.85f,0.40f,1}},
        };
    unsigned int organIconTex[6] = {0,0,0,0,0,0};

    enum BtnIcon {
        ICON_CAMERA=0, ICON_LOAD_IMAGES, ICON_DEPTH,
        ICON_FULL_AUTO, ICON_HEMI_AUTO, ICON_UMEYAMA,
        ICON_RIGID, ICON_HANDLE, ICON_DEFORM,
        ICON_COUNT
    };
    unsigned int btnIconTex[ICON_COUNT] = {};

    bool clusterVis = false;
    bool correspondenceVis = false;
    float arSavedTimer = 0.0f;
    float hemiVoxelSize = 0.5f;
    float idealVoxel1to1  = 0.0f;
    float idealVoxel1to15 = 0.0f;
    float idealVoxel1to2  = 0.0f;
    int   iterCycles = 9;   // IterAutoProbe: number of cycles
    // AutoQCR (Alt+Ctrl+P) 6-DoF/7-DoF 切替。
    //   true  (default) = 6-DoF rigid (論文推奨、CMA-ES 発散回避)
    //   false           = 7-DoF T+R+Scale (Shift+Ctrl+P 互換挙動)
    bool  autoQcrLockScale = true;
    // Ctrl+G (V3-R) 6-DoF/7-DoF 切替。サイドバーのチェックボックスから操作。
    //   true  = 6-DoF rigid (scale=1 固定、SIX_DOF_RIGID)
    //   false (default) = 7-DoF T+R+Scale (SEVEN_DOF、g_ctrlgSearchMode の元 default)
    //   ※ 4-DoF (FOUR_DOF_XYRXRY) は左 floating パネルの radio button 経由のみ。
    //     その状態のときはこの bool は false 扱いで表示する (4-DoF != 6-DoF)。
    //   main.cpp 側 syncUIState で g_ctrlgSearchMode から毎フレーム同期する。
    bool  ctrlgLockScale = false;
    float boardAlpha = 0.7f;
    float targetAlpha = 0.5f;
    unsigned int boardIconTex = 0;

    // ★追加: 器具マスク連動
    float instrumentPxThresh   = 15.0f;
    bool  instrumentMaskActive = false;

    // Vignette auto-detection toggle (mirrors AppContext::detectVignette).
    // Affects what the NEXT Instrument-preview or Run-Depth invocation
    // writes into instrument_segmentation_mask.png. Decision is baked at
    // mask-creation time, so toggling after Run Depth has no effect until
    // the pipeline is re-run.
    bool detectVignette = true;
    // CUDA / GPU toggle (mirrors AppContext::useCuda). Adds --cuda to the
    // CLI for the next Run Depth / Instrument preview.
    bool useCuda = false;

    int initRotPreset = 0;
    int initRotPosition = 0;   // Phase 2: 重心位置 (0=Center, 1=Right, 2=Left) — legacy, UI 表示廃止

    // ---- チャット 9: 4-quadrant 連動 (Initial Orientation = Ctrl+G と同じ mask) ----
    //   main.cpp 側 syncUIState() で g_activeQuadrantMask を毎フレーム反映。
    //   UI チェックボックスは表示更新のみ (副作用なし)。変更は onQuadrantMaskChanged
    //   経由で main.cpp が g_activeQuadrantMask を更新する。
    //   labels が未計算のときは quadLabelsReady=false で頂点数表示を抑制。
    uint8_t activeQuadrantMask = 0x0F;   // QUAD_ALL (= 旧 POS_CENTER と同じ動作)
    bool    quadLabelsReady    = false;
    int     quadNAR = 0, quadNAL = 0, quadNPR = 0, quadNPL = 0;
    int     quadSubsetCount    = 0;      // makeQuadrantSubsetIdx(mask) の出力サイズ
    int     quadTotalCount     = 0;      // labels.size() (= 全頂点数)

    // ---- Anatomical Axes Status (Preview OBJ Anatomical Pose) ----
    //   3 軸ラベル (AP / LR / CC) の状態を毎フレーム syncUIState() で
    //   main.cpp 側 g_liverRegion / g_liverLR / g_liverCC からコピーする。
    //   confidence は表示用 (0..1)。weak フラグは赤バッジ表示の判定に使う。
    //
    //   AP:  g_liverRegion.valid() のみ (raycast ベースなので sign 反転概念なし)
    //   LR:  valid + |a_pos-a_neg|/a_avg を confidence として扱う + decisive
    //        フラグも併記。flipped は g_lrFlipManual の現在値。
    //   CC:  valid + g_liverCC.cc.confidence + g_liverCC.cc.weak (conf<5%)
    //        flipped は g_ccFlipManual の現在値。
    bool  apAxisValid   = false;
    bool  lrAxisValid   = false;
    bool  ccAxisValid   = false;
    float lrConfidence  = 0.0f;   // [0,1]
    bool  lrDecisive    = false;  // EclipseInfo::decisive
    float ccConfidence  = 0.0f;   // [0,1]
    bool  ccWeak        = false;  // confidence < 5%
    bool  lrFlipped     = false;  // g_lrFlipManual
    bool  ccFlipped     = false;  // g_ccFlipManual

    // ---- Step 7: intrinsics source (案 Y dropdown + Active card) ----
    //   syncUIState() が main.cpp の g_intrinsicsSource / g_intrinsics 等から毎フレーム反映。
    IntrinsicsSource intrinsicsSource = IntrinsicsSource::DA3;
    std::string currentPresetKey;      // Preset 選択時の key
    std::string currentDisplayName;    // Active カードの表示名
    float currentFx=0, currentFy=0, currentCx=0, currentCy=0;
    int   currentWidth=0, currentHeight=0;
    bool  customAvailable=false;       // intrinsics_custom.txt が存在し valid
    bool  calibLastAvailable=false;    // intrinsics_calib_last.txt (or legacy) が存在し valid
    bool  da3LastAvailable=false;      // intrinsics_da3_last.txt が存在 (DA3 推定結果)
    // Factory/dynamic presets (main.cpp が presetRegistry() から syncUIState で詰める)
    struct PresetEntry { std::string key; std::string displayName; bool available=false; bool dynamic=false; };
    std::vector<PresetEntry> presetList;
    // Settings タブ: Calibration パラメータ
    std::string chessboardFolder = "../../../chessboard/";
    int   chessboardBoardCols = 9, chessboardBoardRows = 6;
    float chessboardSquareMm   = 22.0f;

    bool calibDone = false;
    std::string calibMessage;
    float calibFx=0, calibFy=0, calibCx=0, calibCy=0;
    float calibRms = 0;
    int   calibImgCount = 0;
    std::string calibFolder = "../../../chessboard/";
};

class RegistrationImGuiManager {
public:
    RegUIActions actions;
    RegUIState   state;

private:
    bool infoExpanded_ = false;
    bool showRestartConfirm_ = false;
    bool regPhaseActive_ = false;
    bool initOrientShouldCollapse_ = false;   // [Phase 9c] One-shot: set by the
                                              // Apply Init Pose button, consumed
                                              // next frame to fold INITIAL ORIENT.
    float sidebarWidth_ = 400.0f;
    bool  intrinsicsWantSettingsTab_ = false;  // Step 7: "Run Calibration…" -> Settings tab

    // ---- INITIAL ORIENTATION panel: CollapsingHeader open states ----
    //   ImGui の CollapsingHeader は自身で開閉状態を保持するが、
    //   childH の動的計算 (BeginChild 開始前) には事前にサイズを
    //   知っておく必要があるため、前フレームの値をここにキャッシュ。
    //   1 フレームの遅延が発生するが視認では検出不可。
    //
    //   anatAxesExpanded_:
    //     - 通常は false (折りたたみ)
    //     - WEAK (state.ccWeak == true) 時は SetNextItemOpen で
    //       強制展開され、ユーザが閉じても次フレームで再展開される
    //   orientExpanded_:
    //     - 既定 false (折りたたみ)。AutoQCR が ORIENT を 9 通り自動
    //       sweep するため、通常は触らない。ヘッダに現在の preset 名
    //       (state.initRotPreset → presetLabel) を表示するので、AutoQCR
    //       後の選択結果は折りたたんだままでも確認できる。
    bool anatAxesExpanded_ = false;
    bool orientExpanded_   = false;

    static ImVec4 colDepth()  { return {0.055f,0.83f,0.66f,1}; }
    static ImVec4 colReg()    { return {0.94f,0.56f,0.19f,1}; }
    static ImVec4 colDeform() { return {0.66f,0.33f,0.97f,1}; }
    static ImVec4 colGreen()  { return {0.13f,0.77f,0.37f,1}; }
    static ImVec4 colRed()    { return {0.94f,0.27f,0.27f,1}; }
    static ImVec4 colBlue()   { return {0.23f,0.51f,0.96f,1}; }
    static ImVec4 colYellow() { return {0.92f,0.70f,0.03f,1}; }
    static ImVec4 colDim()    { return {0.35f,0.38f,0.44f,1}; }
    static ImVec4 colMuted()  { return {0.22f,0.24f,0.29f,1}; }
    static ImVec4 colVis()    { return {0.85f,0.55f,0.15f,1}; }

    static ImU32 toU32(const ImVec4& c, float a=1.0f) {
        return IM_COL32((int)(c.x*255),(int)(c.y*255),(int)(c.z*255),(int)(a*255));
    }

    // -----------------------------------------------------------------
    //  presetLabel:
    //    initRotPreset (int, 0..14) → 表示用ラベル文字列。
    //    drawInitOrientationPanel の 3x5 grid に書かれているラベルと
    //    完全一致させること。ORIENTATION CollapsingHeader のヘッダ
    //    タイトルに "[Base]" / "[Up-R+]" のように現在値を出して、
    //    折りたたんだまま AutoQCR の結果が確認できるようにする。
    // -----------------------------------------------------------------
    static const char* presetLabel(int pid) {
        switch (pid) {
        case  0: return "Base";
        case  1: return "Up";
        case  2: return "Up-R";
        case  3: return "Right";
        case  4: return "Dn-R";
        case  5: return "Down";
        case  6: return "Dn-L";
        case  7: return "Left";
        case  8: return "Up-L";
        case  9: return "Up-R+";
        case 10: return "Right+";
        case 11: return "Dn-R+";
        case 12: return "Up-L+";
        case 13: return "Left+";
        case 14: return "Dn-L+";
        default: return "?";
        }
    }

    // -----------------------------------------------------------------
    //  mapQuadrantToOrientPreset:
    //    POSITION 2x2 grid / Quick Presets でクアドラントが変更されたとき、
    //    対応する Orient preset id を返す。AutoQCR の preset グループ分け
    //    (main.cpp::runAutoQuadCyclicRansac のコメント) と同じ方針:
    //
    //      右側のみ (AR / PR / AR+PR = 0x1 / 0x4 / 0x5)   → Right (preset 3)
    //      左側のみ (AL / PL / AL+PL = 0x2 / 0x8 / 0xA)   → Left  (preset 7)
    //      両側 (QUAD_ALL=0xF や混合) / 空                → Base  (preset 0)
    //
    //    これにより、ユーザは POSITION を選ぶだけで妥当な Orient が自動セットされ、
    //    そのまま Apply Init Pose を押せる。手動で別の preset を試したい場合は
    //    ORIENTATION CollapsingHeader を開いて選び直す。
    // -----------------------------------------------------------------
    static int mapQuadrantToOrientPreset(uint8_t mask) {
        constexpr uint8_t kMaskAR = 0x01;
        constexpr uint8_t kMaskAL = 0x02;
        constexpr uint8_t kMaskPR = 0x04;
        constexpr uint8_t kMaskPL = 0x08;
        const bool hasR = (mask & (kMaskAR | kMaskPR)) != 0;
        const bool hasL = (mask & (kMaskAL | kMaskPL)) != 0;
        if (hasR && !hasL) return 3;   // Right
        if (hasL && !hasR) return 7;   // Left
        return 0;                       // Base (両側 / 空)
    }

    int currentPhase() const {
        if (state.mainMode == 1) return 2;
        if (regPhaseActive_) return 1;
        return 0;
    }

    void drawButtonIcon(unsigned int tex, ImVec2 btnPos, float btnH, bool disabled=false) {
        if (!tex) return;
        float iconSz = btnH * 0.85f;
        if (iconSz < 28.0f) iconSz = 28.0f;
        float iconY = btnPos.y + (btnH - iconSz) * 0.5f;
        float iconAlpha = disabled ? 0.3f : 1.0f;
        ImGui::GetWindowDrawList()->AddImage(
            (ImTextureID)(intptr_t)tex,
            ImVec2(btnPos.x + 4, iconY),
            ImVec2(btnPos.x + 4 + iconSz, iconY + iconSz),
            ImVec2(0,0), ImVec2(1,1),
            IM_COL32(255,255,255,(int)(iconAlpha*255)));
    }

    bool colorButton(const char* label, ImVec4 col, bool active=false, bool disabled=false, float w=-1, float h=0, unsigned int iconTex=0) {
        ImGui::PushID(label);
        if (disabled) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.18f,0.19f,0.22f,1));
        } else if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(col.x*0.18f,col.y*0.18f,col.z*0.18f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(col.x*0.25f,col.y*0.25f,col.z*0.25f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(col.x*0.32f,col.y*0.32f,col.z*0.32f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, col);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(col.x*0.10f,col.y*0.10f,col.z*0.10f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(col.x*0.18f,col.y*0.18f,col.z*0.18f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(col.x*0.25f,col.y*0.25f,col.z*0.25f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, col);
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        char paddedLabel[128];
        if (iconTex) snprintf(paddedLabel, sizeof(paddedLabel), "       %s", label);
        else snprintf(paddedLabel, sizeof(paddedLabel), "%s", label);
        ImVec2 btnPos = ImGui::GetCursorScreenPos();
        float btnH = h > 0 ? h : ImGui::GetFrameHeight();
        bool clicked = ImGui::Button(paddedLabel, ImVec2(w, h));
        if (iconTex) drawButtonIcon(iconTex, btnPos, btnH, disabled);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
        ImGui::PopID();
        return clicked && !disabled;
    }

    bool glowButton(const char* label, ImVec4 col, bool disabled=false, float w=-1, float h=36, unsigned int iconTex=0) {
        ImGui::PushID(label);
        if (disabled) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.18f,0.19f,0.22f,1));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(col.x*0.20f,col.y*0.20f,col.z*0.20f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(col.x*0.30f,col.y*0.30f,col.z*0.30f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(col.x*0.40f,col.y*0.40f,col.z*0.40f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1,1,1,0.95f));
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, (h - ImGui::GetFontSize()) * 0.5f));
        char paddedLabel[128];
        if (iconTex) snprintf(paddedLabel, sizeof(paddedLabel), "       %s", label);
        else snprintf(paddedLabel, sizeof(paddedLabel), "%s", label);
        ImVec2 btnPos = ImGui::GetCursorScreenPos();
        bool clicked = ImGui::Button(paddedLabel, ImVec2(w, 0));
        if (iconTex) drawButtonIcon(iconTex, btnPos, h, disabled);
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
        ImGui::PopID();
        return clicked && !disabled;
    }

    bool methodButton(const char* label, const char* sc, bool isSel, int rState, bool disabled, unsigned int iconTex=0, float btnW=-1) {
        ImGui::PushID(label);
        ImVec4 c = colReg();
        if (disabled) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.06f,0.065f,0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.18f,0.19f,0.22f,1));
        } else if (isSel && rState > 0) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.18f,c.y*0.18f,c.z*0.18f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(c.x*0.25f,c.y*0.25f,c.z*0.25f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(c.x*0.32f,c.y*0.32f,c.z*0.32f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, c);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.07f,c.y*0.07f,c.z*0.07f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(c.x*0.14f,c.y*0.14f,c.z*0.14f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(c.x*0.22f,c.y*0.22f,c.z*0.22f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(c.x*0.7f,c.y*0.7f,c.z*0.7f,1));
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
        char paddedLabel[128];
        if (iconTex) snprintf(paddedLabel, sizeof(paddedLabel), "       %s", label);
        else snprintf(paddedLabel, sizeof(paddedLabel), "%s", label);
        ImVec2 btnPos = ImGui::GetCursorScreenPos();
        bool clicked = ImGui::Button(paddedLabel, ImVec2(btnW, 36));
        if (iconTex) drawButtonIcon(iconTex, btnPos, 36, disabled);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
        ImGui::PopID();
        return clicked && !disabled;
    }

    void drawProgress(const char* label, int cur, int total, ImVec4 col) {
        ImGui::TextColored(col, "%s", label);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 40);
        ImGui::Text("%d / %d", cur, total);
        float frac = total > 0 ? (float)cur / total : 0.0f;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f,0.1f,0.13f,1));
        ImGui::ProgressBar(frac, ImVec2(-1, 4), "");
        ImGui::PopStyleColor(2);
    }

    static ImVec4 pointColor(int i) {
        const ImVec4 cols[] = {
            {1.0f,0.0f,0.0f,1}, {0.0f,1.0f,0.0f,1}, {0.0f,0.0f,1.0f,1},
            {1.0f,1.0f,0.0f,1}, {1.0f,0.0f,1.0f,1}, {0.0f,1.0f,1.0f,1}
        };
        return cols[i % 6];
    }

    void drawColoredPointProgress(const char* label, int cur, int total, bool isActive, float scale=1.0f) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x;
        float dotR = 10.0f * scale;
        float spacing = 8.0f * scale;
        float labelW = ImGui::CalcTextSize(label).x;
        float countW = 50.0f * scale;
        float dotsStartX = p.x + labelW + 20.0f * scale;
        float rowH = dotR * 2 + 4.0f;

        ImGui::TextColored(isActive ? colReg() : (cur >= total ? colGreen() : colMuted()), "%s", label);
        ImGui::SameLine(w - countW);
        ImGui::Text("%d / %d", cur, total);

        float dotsY = p.y + rowH * 0.5f;
        for (int i = 0; i < total; i++) {
            float cx = dotsStartX + i * (dotR * 2 + spacing) + dotR;
            ImVec4 c = pointColor(i);
            bool filled = (i < cur);
            if (filled) {
                dl->AddCircleFilled(ImVec2(cx, dotsY), dotR,
                                    IM_COL32((int)(c.x*255),(int)(c.y*255),(int)(c.z*255),255));
                dl->AddCircle(ImVec2(cx, dotsY), dotR + 1,
                              IM_COL32((int)(c.x*200),(int)(c.y*200),(int)(c.z*200),120), 0, 2.0f);
            } else if (isActive && i == cur) {
                float t = (float)ImGui::GetTime();
                float pulse = 0.4f + 0.3f * sinf(t * 4.0f);
                dl->AddCircleFilled(ImVec2(cx, dotsY), dotR,
                                    IM_COL32((int)(c.x*80),(int)(c.y*80),(int)(c.z*80),255));
                dl->AddCircle(ImVec2(cx, dotsY), dotR + 2,
                              IM_COL32((int)(c.x*255),(int)(c.y*255),(int)(c.z*255),(int)(pulse*255)), 0, 2.0f);
            } else {
                dl->AddCircleFilled(ImVec2(cx, dotsY), dotR,
                                    IM_COL32((int)(c.x*40),(int)(c.y*40),(int)(c.z*40),255));
                dl->AddCircle(ImVec2(cx, dotsY), dotR,
                              IM_COL32((int)(c.x*80),(int)(c.y*80),(int)(c.z*80),180));
            }
            char num[4]; snprintf(num, sizeof(num), "%d", i + 1);
            ImVec2 ts = ImGui::CalcTextSize(num);
            dl->AddText(ImVec2(cx - ts.x * 0.5f, dotsY - ts.y * 0.5f),
                        filled ? IM_COL32(255,255,255,230) : IM_COL32(255,255,255,60), num);
        }
        ImGui::Spacing();
    }

    void drawSectionWithBar(const char* label, ImVec4 col, bool done, bool active, bool processing) {
        ImGui::Spacing();
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float barH = ImGui::GetFontSize() + 10;

        float barAlpha = active ? 1.0f : (done ? 0.5f : 0.12f);
        dl->AddRectFilled(ImVec2(p.x, p.y), ImVec2(p.x + 4, p.y + barH),
                          toU32(col, barAlpha), 2.0f);

        if (active && !done) {
            dl->AddRectFilled(ImVec2(p.x + 4, p.y), ImVec2(p.x + sidebarWidth_, p.y + barH),
                              toU32(col, 0.05f));
        }

        ImGui::Indent(16);
        if (done) {
            ImGui::TextColored(col, ">> %s", label);
            ImGui::SameLine();
            ImGui::TextColored(col, " DONE");
        } else if (processing) {
            ImGui::TextColored(col, "> %s ...", label);
        } else {
            ImGui::TextColored(active ? col : colMuted(), "  %s", label);
        }
        ImGui::Unindent(16);
        ImGui::Spacing();
    }

public:
    void resetToDepthPhase() {
        regPhaseActive_ = false;
        showRestartConfirm_ = false;
        state.regMethod = -1;
    }

    void draw(int windowWidth, int windowHeight) {
        if (state.arSavedTimer > 0) state.arSavedTimer -= ImGui::GetIO().DeltaTime;

        if (state.regState > 0 || state.regMethod >= 0) regPhaseActive_ = true;
        if (state.mainMode == 1) regPhaseActive_ = true;

        bool umeyamaSplit = (state.splitScreen && state.regMethod == 2
                             && state.regState >= 1 && state.regState <= 3);
        if (umeyamaSplit) { drawUmeyamaOverlay(windowWidth, windowHeight); return; }
        // 画像がロードされていて、まだdepth処理が完了していない場合もオーバーレイを表示
        if (state.depthSplitScreen || (state.hasLocalImage && !state.depthDone && state.mainMode == 0)) {
            drawDepthOverlay(windowWidth, windowHeight);
            if (state.depthSplitScreen) return;  // Split screenモードの時だけ早期リターン
        }

        ImGui::SetNextWindowPos(ImVec2(windowWidth - sidebarWidth_, 0));
        ImGui::SetNextWindowSize(ImVec2(sidebarWidth_, (float)windowHeight));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize
                                 | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.067f,0.075f,0.094f,1));

        if (ImGui::Begin("##RegSidebar", nullptr, flags)) {
            drawWorkflowStepper();
            drawDepthSection();
            drawRegistrationSection();
            drawDeformSection();
            drawExport();
            drawSaveAR();
            drawVisibility();
            drawInfoPanel();
        }
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(3);
    }

    float getViewportWidth(int windowWidth) const {
        bool umeyamaSplit = (state.splitScreen && state.regMethod == 2
                             && state.regState >= 1 && state.regState <= 3);
        if (umeyamaSplit || state.depthSplitScreen) return (float)windowWidth;
        return windowWidth - sidebarWidth_;
    }

private:
    void drawWorkflowStepper() {
        int phase = currentPhase();

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f,0.055f,0.07f,1));
        ImGui::BeginChild("##stepper", ImVec2(0, 78), false);
        ImGui::Spacing(); ImGui::Spacing();

        float w = ImGui::GetContentRegionAvail().x;
        float cx1=w*0.20f, cx2=w*0.50f, cx3=w*0.80f, cy=24.0f;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 wp = ImGui::GetWindowPos();

        auto stepColor = [&](int step) -> ImU32 {
            if (step == 0) {
                return toU32(colDepth());
            }
            if (step == 1) {
                if (state.regState == 4) return toU32(colReg());
                if (regPhaseActive_) return toU32(colReg());
                return toU32(colMuted(), 0.3f);
            }
            if (phase == 2) return toU32(colDeform());
            return toU32(colMuted(), 0.3f);
        };

        dl->AddLine(ImVec2(wp.x+cx1+14,wp.y+cy), ImVec2(wp.x+cx2-14,wp.y+cy),
                    regPhaseActive_ ? toU32(colDepth(),0.5f) : toU32(colMuted(),0.2f), 2);
        dl->AddLine(ImVec2(wp.x+cx2+14,wp.y+cy), ImVec2(wp.x+cx3-14,wp.y+cy),
                    (state.mainMode==1) ? toU32(colReg(),0.5f) : toU32(colMuted(),0.2f), 2);

        dl->AddCircleFilled(ImVec2(wp.x+cx1,wp.y+cy), 14, stepColor(0));
        dl->AddCircleFilled(ImVec2(wp.x+cx2,wp.y+cy), 14, stepColor(1));
        dl->AddCircleFilled(ImVec2(wp.x+cx3,wp.y+cy), 14, stepColor(2));

        auto dn = [&](float cx, int step) {
            const char* txt;
            if (step == 0 && state.depthDone) txt = "ok";
            else if (step == 1 && state.regState == 4) txt = "ok";
            else if (step == 0) txt = "1";
            else if (step == 1) txt = "2";
            else txt = "3";
            ImVec2 ts = ImGui::CalcTextSize(txt);
            dl->AddText(ImVec2(wp.x+cx-ts.x*0.5f, wp.y+cy-ts.y*0.5f), IM_COL32(255,255,255,220), txt);
        };
        dn(cx1, 0); dn(cx2, 1); dn(cx3, 2);

        auto lbl = [&](float cx, const char* t, ImVec4 c, bool lit) {
            ImVec2 ts = ImGui::CalcTextSize(t);
            dl->AddText(ImVec2(wp.x+cx-ts.x*0.5f, wp.y+cy+18),
                        lit ? toU32(c) : toU32(colMuted(),0.45f), t);
        };
        lbl(cx1, "Depth",        colDepth(),  true);
        lbl(cx2, "Registration", colReg(),    regPhaseActive_ || state.regState==4);
        lbl(cx3, "Deform",       colDeform(), phase==2);

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Separator();
    }

    void drawDepthSection() {
        int phase = currentPhase();
        drawSectionWithBar("DEPTH GENERATION", colDepth(), state.depthDone, phase==0, state.depthRunning);

        if (regPhaseActive_ || state.mainMode == 1) {
            ImGui::Indent(16);
            if (state.depthDone) {
                drawIntrinsicsSource("reg");
            } else {
                ImGui::TextColored(colMuted(), "  Depth: Not generated");
            }
            ImGui::Unindent(16);
            ImGui::Spacing(); ImGui::Separator();
            return;
        }

        ImGui::Indent(16); ImGui::PushItemWidth(-16);

        ImGui::TextColored(colDepth(), "DEPTH MODEL");
        ImGui::Spacing();
        {
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f,0.1f,0.13f,1));
            ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.08f,0.08f,0.11f,1));
            const char* preview = depthModelName(state.depthModelIdx);
            if (ImGui::BeginCombo("##depthmodel", preview)) {
                for (int i = 0; i < DEPTH_MODEL_COUNT; i++) {
                    bool avail = state.depthModelAvail[i];
                    bool selected = (state.depthModelIdx == i);
                    char label[128];
                    if (avail)
                        snprintf(label, sizeof(label), "%s", depthModelName(i));
                    else
                        snprintf(label, sizeof(label), "%s  [not found]", depthModelName(i));
                    if (!avail) ImGui::PushStyleColor(ImGuiCol_Text, colDim());
                    if (ImGui::Selectable(label, selected)) {
                        if (avail && actions.onSwitchDepthModel) {
                            actions.onSwitchDepthModel(i);
                        }
                    }
                    if (!avail) ImGui::PopStyleColor();
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        ImVec2 srcStart = ImGui::GetCursorScreenPos();
        ImGui::TextColored(colDepth(), "IMAGE SOURCE");
        ImGui::Spacing();

        {
            const char* cl;
            ImVec4 cc2;
            bool isActive = state.cameraState != 0;
            switch(state.cameraState) {
            case 0:  cl = "> Start Camera";  cc2 = colDepth(); break;
            case 1:  cl = "Capture";        cc2 = colGreen(); break;
            case 2:  cl = "> Restart Camera"; cc2 = colBlue();  break;
            default: cl = "Camera";           cc2 = colDim();   break;
            }
            if (state.cameraState == 0 && !state.depthDone) {
                if (glowButton(cl, colDepth(), false, -1, 36, state.btnIconTex[RegUIState::ICON_CAMERA])) { if(actions.onToggleCamera) actions.onToggleCamera(); }
            } else {
                if (colorButton(cl, cc2, isActive, false, -1, 0, state.btnIconTex[RegUIState::ICON_CAMERA])) { if(actions.onToggleCamera) actions.onToggleCamera(); }
            }
        }

        ImGui::Spacing();

        if (state.cameraState == 0 && !state.depthDone) {
            if (glowButton("Load Local Image", colDepth(), false, -1, 36, state.btnIconTex[RegUIState::ICON_LOAD_IMAGES])) {
                if(actions.onLoadLocalImage) actions.onLoadLocalImage();
            }
        } else {
            if (colorButton("Load Local Image", colDepth(), false, false, -1, 0, state.btnIconTex[RegUIState::ICON_LOAD_IMAGES])) {
                if(actions.onLoadLocalImage) actions.onLoadLocalImage();
            }
        }

        if (state.hasLocalImage) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.01f,0.25f,0.20f,0.15f));
            ImGui::BeginChild("##localimg", ImVec2(-1,24), true);
            ImGui::TextColored(colDepth(), "  %s", state.localImageName.c_str());
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 10);
            if(ImGui::SmallButton("x")) { if(actions.onResetDefaultImage) actions.onResetDefaultImage(); }
            ImGui::EndChild(); ImGui::PopStyleColor();
        }

        ImGui::Spacing();
        ImGui::TextColored(colMuted(), "  or drag & drop PNG/JPG onto viewport");

        if (!state.depthDone && state.cameraState == 0 && !state.hasLocalImage) {
            float t = (float)ImGui::GetTime();
            float pulse = 0.3f + 0.35f * sinf(t * 3.0f);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImVec2 srcEnd = ImGui::GetCursorScreenPos();
            dl->AddRect(ImVec2(srcStart.x - 4, srcStart.y - 4),
                        ImVec2(srcStart.x + ImGui::GetContentRegionAvail().x + 4, srcEnd.y + 2),
                        toU32(colDepth(), pulse), 6.0f, 0, 2.0f);
        }

        if (state.cameraState == 1) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.02f,0.09f,0.04f,0.5f));
            ImGui::BeginChild("##camst", ImVec2(-1,24), true);
            ImGui::TextColored(colGreen(), "  * LIVE");
            ImGui::EndChild(); ImGui::PopStyleColor();
        } else if (state.cameraState == 2) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f,0.06f,0.12f,0.5f));
            ImGui::BeginChild("##camst", ImVec2(-1,24), true);
            ImGui::TextColored(colBlue(), "  FROZEN - add SegPoints below");
            ImGui::EndChild(); ImGui::PopStyleColor();
        }

        // キャプチャ済みの場合のみDepthオーバーレイを表示
        if (state.cameraState == 2) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f,0.06f,0.14f,0.6f));
            ImGui::BeginChild("##segpts", ImVec2(-1, 72), true);
            ImGui::SetWindowFontScale(1.4f);
            ImGui::TextColored(colBlue(), " L-click = FG   R-click = BG");
            ImGui::Spacing();
            ImGui::TextColored(colDepth(), "  FG: %d", state.segFG);
            ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.4f);
            ImGui::TextColored(colRed(), "BG: %d", state.segBG);
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 75);
            {
                bool noP = (state.segFG + state.segBG == 0);
                if(colorButton("Undo", noP ? colDim() : colReg(), false, noP, 75)) {
                    if(actions.onUndoSegPoint) actions.onUndoSegPoint();
                }
            }
            ImGui::SetWindowFontScale(1.0f);
            ImGui::EndChild(); ImGui::PopStyleColor();
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        if (state.depthRunning) {
            ImVec4 dc = colDepth();
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0,12));
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(dc.x*0.08f,dc.y*0.08f,dc.z*0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(dc.x*0.08f,dc.y*0.08f,dc.z*0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(dc.x*0.08f,dc.y*0.08f,dc.z*0.08f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, dc);
            ImGui::Button("Processing...", ImVec2(-1,0));
            ImGui::PopStyleColor(4); ImGui::PopStyleVar(2);
        }

        if (state.depthDone) {
            drawIntrinsicsSource("dep");

            ImGui::Spacing();
            if(colorButton("Reset to Default Image", colDim())) {
                if(actions.onResetDefaultImage) actions.onResetDefaultImage();
            }
        }

        ImGui::PopItemWidth(); ImGui::Unindent(16);
        ImGui::Spacing(); ImGui::Separator();
    }

    // =====================================================================
    //  drawAnatomicalAxesContent (Preview OBJ Anatomical Pose):
    //    Initial Orientation panel の ANATOMICAL AXES CollapsingHeader
    //    が「展開」状態の時にヘッダ直下に描く 3 軸 (AP / LR / CC) の
    //    確認 UI。レジ前に解剖軸の sign が正しいかを目視 + 数値で確認し、
    //    Flip ボタンで g_lrFlipManual / g_ccFlipManual を即時トグルする。
    //
    //    WEAK 状態 (CC confidence < 5%) や未計算状態のステータスバッジは
    //    drawInitOrientationPanel 側で CollapsingHeader のタイトルに
    //    埋め込まれる (例: "ANATOMICAL AXES  [WEAK 3.2%]")。ここでは
    //    純粋に内容 (3 行) のみを担当する。
    //
    //    レイアウト (1 行 = 20px):
    //      [name(40px)] [● status] [conf %] [note]            [Flip btn(70px)]
    // =====================================================================
    void drawAnatomicalAxesContent() {
        const float rowH = 20.0f;
        const float rowSpacing = 3.0f;

        // ---- axis 1 行を描く lambda ----
        auto drawRow = [&](const char* name,
                           bool valid,
                           bool hasConfidence,
                           float confidence,
                           bool warn,           // true なら conf を赤で表示
                           const char* note,    // 末尾の注釈 (nullable)
                           bool flipped,
                           bool hasFlipBtn,
                           const std::function<void()>& onFlip)
        {
            ImGui::PushID(name);
            // axis 名
            ImGui::TextColored(ImVec4(0.70f, 0.75f, 0.85f, 1.0f), "  %s", name);
            ImGui::SameLine(40.0f);

            // status dot
            ImVec4 dotCol = valid ? ImVec4(0.20f, 0.85f, 0.40f, 1.0f)
                                  : ImVec4(0.50f, 0.52f, 0.55f, 1.0f);
            // 丸記号は環境依存なので '*' を使い色で区別。
            ImGui::TextColored(dotCol, valid ? "*" : "-");
            ImGui::SameLine(0, 4);

            if (valid) {
                ImGui::TextColored(ImVec4(0.80f, 0.84f, 0.90f, 1.0f), "valid");
            } else {
                ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.62f, 1.0f), "n/a");
            }

            // confidence (LR, CC のみ)
            if (valid && hasConfidence) {
                ImGui::SameLine(135.0f);
                ImVec4 conCol;
                if (warn) {
                    conCol = ImVec4(0.96f, 0.32f, 0.32f, 1.0f);   // red
                } else if (confidence < 0.15f) {
                    conCol = ImVec4(0.95f, 0.75f, 0.25f, 1.0f);   // yellow
                } else {
                    conCol = ImVec4(0.30f, 0.85f, 0.45f, 1.0f);   // green
                }
                if (warn) {
                    ImGui::TextColored(conCol, "[!] %.1f%%", confidence * 100.0f);
                } else {
                    ImGui::TextColored(conCol, "%.1f%%", confidence * 100.0f);
                }
                if (note && note[0]) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.62f, 0.65f, 0.72f, 1.0f), " %s", note);
                }
            }

            // Flip ボタン (右端固定)
            //   従来は GetContentRegionAvail() を見て条件付き SameLine していたが、
            //   前段の "lean (weak)" や confidence% が長いと availW が btnW を
            //   下回って SameLine を skip → 次行に折り返し → childH の clip rect
            //   に食われて見えなくなるバグがあった。修正: コンテンツ右端からの
            //   絶対位置で SameLine し、必ず同じ行の右端に出るようにする。
            //   視認性のため: ダーク背景上で埋もれないよう、unflipped はビビッドな青、
            //   flipped はアンバーで明示する。border 1px + 角丸 3px + 白文字。
            if (hasFlipBtn) {
                const float btnW = 72.0f;
                // GetContentRegionMax().x = 現在のグループ / カラムの右端 (window-local)。
                // BeginChild 内では child window の右端と等価。ここから btnW + 2 を
                // 引いた位置を SameLine の絶対 x に渡せば、行の右端にボタンが固定で
                // 配置される (前段テキストの長さに依存しない)。
                float rightX = ImGui::GetContentRegionMax().x;
                ImGui::SameLine(rightX - btnW - 2.0f);

                ImVec4 btnCol = flipped
                                    ? ImVec4(0.78f, 0.45f, 0.12f, 1.0f)   // bright amber: flipped
                                    : ImVec4(0.22f, 0.45f, 0.72f, 1.0f);  // bright blue:  default
                ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                      ImVec4(btnCol.x*1.2f + 0.05f,
                                             btnCol.y*1.2f + 0.05f,
                                             btnCol.z*1.2f + 0.05f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                      ImVec4(btnCol.x*1.4f + 0.10f,
                                             btnCol.y*1.4f + 0.10f,
                                             btnCol.z*1.4f + 0.10f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text,
                                      ImVec4(1.00f, 1.00f, 1.00f, 1.0f));   // 白
                ImGui::PushStyleColor(ImGuiCol_Border,
                                      ImVec4(btnCol.x*1.5f + 0.10f,
                                             btnCol.y*1.5f + 0.10f,
                                             btnCol.z*1.5f + 0.10f, 1.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
                bool dis = !valid;
                if (dis) ImGui::BeginDisabled();
                const char* btnLbl = flipped ? "Flipped" : "Flip";
                if (ImGui::Button(btnLbl, ImVec2(btnW, rowH - 2.0f))) {
                    if (onFlip) onFlip();
                }
                if (dis) ImGui::EndDisabled();
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(5);
            }

            ImGui::PopID();
            ImGui::Dummy(ImVec2(0, rowSpacing - 2.0f));
        };

        // ---- 3 行: AP / LR / CC ----
        drawRow("AP",
                state.apAxisValid,
                /*hasConfidence=*/false, 0.0f, /*warn=*/false,
                /*note=*/nullptr,
                /*flipped=*/false, /*hasFlipBtn=*/false,
                /*onFlip=*/{});

        drawRow("LR",
                state.lrAxisValid,
                /*hasConfidence=*/true, state.lrConfidence,
                /*warn=*/(state.lrAxisValid && !state.lrDecisive),
                /*note=*/(state.lrAxisValid
                              ? (state.lrDecisive ? "dec." : "lean!")
                              : nullptr),
                state.lrFlipped, /*hasFlipBtn=*/true,
                [this]() { if (actions.onFlipLR) actions.onFlipLR(); });

        drawRow("CC",
                state.ccAxisValid,
                /*hasConfidence=*/true, state.ccConfidence,
                /*warn=*/state.ccWeak,
                /*note=*/(state.ccAxisValid
                              ? (state.ccWeak ? "WEAK" : "ok")
                              : nullptr),
                state.ccFlipped, /*hasFlipBtn=*/true,
                [this]() { if (actions.onFlipCC) actions.onFlipCC(); });
    }

    // =====================================================================
    //  drawVoxelInfo:
    //    Voxel size 表示パネル (Voxel 数値 + クリック式バー + ideal markers)。
    //    g_voxelSize (extern, main.cpp:119) と直接結びついており、Hemi Auto
    //    だけでなく全 FGR+ICP 系登録 (Shift+O, Ctrl+P, Shift+Ctrl+P, AutoQCR
    //    含む) の voxel downsample 解像度を決める。default 0.30 で運用しても
    //    支障はないが、論文の感度解析等で変更したい場合のためここに残す。
    //
    //    [UI整理] 以前は state.regMethod == 1 (Hemi Auto 選択時) のみ表示して
    //    いたが、影響範囲が全方式なのと、日常運用では触らないことから、
    //    REGISTRATION 末尾の Advanced CollapsingHeader に移動した。
    //
    //    キーボードの UP / DOWN でも +/- 0.05 単位で変更可能 (main.cpp 側)。
    // =====================================================================
    void drawVoxelInfo() {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.02f,0.08f,0.04f,0.4f));
        ImGui::BeginChild("##voxelinfo", ImVec2(-1, 62), false);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
        ImGui::TextColored(ImVec4(0.4f,0.8f,0.5f,0.8f), "  Voxel");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.6f,1.0f,0.7f,1.0f), "%.2f", state.hemiVoxelSize);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.3f,0.4f,0.32f,0.7f), "  [UP] +0.05  [DOWN] -0.05");
        ImGui::Spacing();
        {
            const float vMax = 0.6f;
            ImVec2 p = ImGui::GetCursorScreenPos();
            float avail = ImGui::GetContentRegionAvail().x - 8;
            const float barH = 14.0f;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImGui::InvisibleButton("##voxelbar", ImVec2(avail, barH));
            bool hovered = ImGui::IsItemHovered();
            bool held    = ImGui::IsItemActive();
            if ((held && ImGui::IsMouseDown(0)) || ImGui::IsItemClicked(0)) {
                float mx = ImGui::GetIO().MousePos.x;
                float ratio = (mx - p.x) / avail;
                ratio = std::max(0.0f, std::min(1.0f, ratio));
                float newVal = ratio * vMax;
                newVal = std::round(newVal / 0.05f) * 0.05f;
                newVal = std::max(0.05f, newVal);
                if (actions.onHemiVoxelChanged) actions.onHemiVoxelChanged(newVal);
            }
            ImU32 bgCol  = hovered ? IM_COL32(40, 80, 45, 220) : IM_COL32(30, 60, 35, 200);
            dl->AddRectFilled(ImVec2(p.x, p.y), ImVec2(p.x + avail, p.y + barH), bgCol, 3.0f);
            float fillRatio = std::min(state.hemiVoxelSize, vMax) / vMax;
            float curX = p.x + avail * fillRatio;
            ImU32 fillCol = held ? IM_COL32(100, 210, 120, 200) : IM_COL32(80, 180, 100, 160);
            dl->AddRectFilled(ImVec2(p.x, p.y), ImVec2(curX, p.y + barH), fillCol, 3.0f);
            if (hovered || held) {
                float mx = ImGui::GetIO().MousePos.x;
                float ratio = std::max(0.0f, std::min(1.0f, (mx - p.x) / avail));
                float preview = std::round(ratio * vMax / 0.05f) * 0.05f;
                float px2 = p.x + avail * (std::min(preview, vMax) / vMax);
                dl->AddLine(ImVec2(px2, p.y), ImVec2(px2, p.y + barH), IM_COL32(255,255,255,160), 1.0f);
            }
            struct IdealLine { float val; ImU32 col; };
            IdealLine ideals[3] = {
                                   { state.idealVoxel1to2,  IM_COL32(255,220, 60,220) },
                                   { state.idealVoxel1to15, IM_COL32(255,160, 40,220) },
                                   { state.idealVoxel1to1,  IM_COL32(255, 80, 60,220) },
                                   };
            for (auto& il : ideals) {
                if (il.val <= 0.0f || il.val > vMax) continue;
                float lx = p.x + avail * (il.val / vMax);
                dl->AddLine(ImVec2(lx, p.y - 2), ImVec2(lx, p.y + barH + 2), il.col, 2.0f);
                char buf[8]; snprintf(buf, sizeof(buf), "%.2f", il.val);
                ImVec2 ts = ImGui::CalcTextSize(buf);
                float tx = lx - ts.x * 0.5f;
                if (tx < p.x) tx = p.x;
                if (tx + ts.x > p.x + avail) tx = p.x + avail - ts.x;
                dl->AddText(ImVec2(tx, p.y + barH + 3),
                            (il.col & 0x00FFFFFF) | 0xCC000000, buf);
            }
            ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + barH + 14));
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    void drawInitOrientationPanel() {
        // [UI整理 - 新レイアウト]
        //   ANATOMICAL AXES と ORIENTATION を CollapsingHeader 化して
        //   POSITION + Apply Init Pose を主動線として前面に出す。
        //
        //   配置順:
        //     INITIAL ORIENTATION (固定ヘッダ)
        //       ANATOMICAL AXES [OK/WEAK/...] (折りたたみ, WEAK 時自動展開)
        //         └ AP / LR / CC の 3 行 (展開時のみ)
        //       POSITION (常時表示)
        //         └ 2x2 grid + Quick Presets
        //       ORIENTATION [Base/Up/...] (折りたたみ, 既定 OFF)
        //         └ 3x5 grid (展開時のみ)
        //       Apply Init Pose (常時表示)
        //
        //   AutoQCR が ORIENT を 9 通り自動 sweep するため、ORIENTATION は
        //   ヘッダのラベルで現在の preset 名 ([Base] / [Up-R+] 等) を確認
        //   できれば十分。手動指定が必要な時だけ展開する。
        //
        //   ANATOMICAL AXES は WEAK (CC confidence < 5%) のとき
        //   SetNextItemOpen で強制展開し、赤テキストで警告する。
        //
        //   childH は前フレームの anatAxesExpanded_ / orientExpanded_ から
        //   動的に計算 (1 フレーム遅延あるが視認不可)。
        const float fontH    = ImGui::GetFontSize();
        const float headerH  = ImGui::GetFrameHeight();  // CollapsingHeader 一行分

        float childH = 4.0f + fontH + 4.0f            // INITIAL ORIENTATION title
                       + headerH + 2.0f;                 // ANATOMICAL AXES collapsing header
        if (anatAxesExpanded_) {
            childH += 3.0f * (20.0f + 3.0f);           // axes 3 行
        }
        childH += 6.0f                                 // section separator
                  + fontH + 2.0f                         // POSITION sub-header
                  + fontH                                // anatomy hint
                  + 32.0f                                // 1x4 grid (Phase 9a, was 64 for 2x2)
                  + fontH + 4.0f                         // mask text
                  + (22.0f + 4.0f)                       // quick presets
                  + 6.0f                                 // spacing before ORIENTATION
                  + headerH + 2.0f;                      // ORIENTATION collapsing header
        if (orientExpanded_) {
            childH += 3.0f * (22.0f + 4.0f);           // orientation 3 行
        }
        childH += 8.0f;                                // tail padding only
                                                       // (Phase 9d: Apply moved out)

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.04f, 0.06f, 0.10f, 0.5f));
        ImGui::BeginChild("##initOrient", ImVec2(-1, childH), false);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 1.0f), "  INITIAL ORIENTATION");
        ImGui::Spacing();

        // ============================================================
        //  [§A] ANATOMICAL AXES (CollapsingHeader, WEAK 時自動展開)
        // ============================================================
        {
            // ヘッダタイトル + 表示色を状態で決める。
            //   WEAK    -> 赤 + SetNextItemOpen(true) で強制展開
            //   未計算  -> 黄
            //   全 OK   -> 緑
            // "###anat_axes" 部分で固有 ID を固定 (label が変わっても開閉状態が
            // 維持されるように)。
            ImVec4 hdrCol;
            char hdrLabel[160];
            if (state.ccAxisValid && state.ccWeak) {
                hdrCol = ImVec4(0.96f, 0.32f, 0.32f, 1.0f);  // red
                std::snprintf(hdrLabel, sizeof(hdrLabel),
                              "ANATOMICAL AXES  [WEAK %.1f%% - check Flip CC]###anat_axes",
                              state.ccConfidence * 100.0f);
                ImGui::SetNextItemOpen(true, ImGuiCond_Always);
            } else if (!state.apAxisValid || !state.lrAxisValid || !state.ccAxisValid) {
                hdrCol = ImVec4(0.95f, 0.72f, 0.28f, 1.0f);  // yellow
                std::snprintf(hdrLabel, sizeof(hdrLabel),
                              "ANATOMICAL AXES  [not all computed - press Apply Init Pose]"
                              "###anat_axes");
            } else {
                hdrCol = ImVec4(0.30f, 0.85f, 0.45f, 1.0f);  // green
                std::snprintf(hdrLabel, sizeof(hdrLabel),
                              "ANATOMICAL AXES  [OK]###anat_axes");
            }

            ImGui::PushStyleColor(ImGuiCol_Text, hdrCol);
            // Header 自体の背景は薄めに (主動線ではないので主張させない)。
            ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(0.10f, 0.12f, 0.16f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.15f, 0.18f, 0.24f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive,  ImVec4(0.20f, 0.24f, 0.32f, 0.8f));
            bool anatOpen = ImGui::CollapsingHeader(hdrLabel);
            ImGui::PopStyleColor(4);
            anatAxesExpanded_ = anatOpen;

            if (anatOpen) {
                drawAnatomicalAxesContent();
            }
        }
        ImGui::Spacing();

        float totalW = ImGui::GetContentRegionAvail().x;
        float btnW   = (totalW - 8.0f) / 3.0f;
        float btnH   = 22.0f;

        ImVec4 colActive   = ImVec4(0.23f, 0.51f, 0.96f, 1.0f);
        ImVec4 colInactive = ImVec4(0.20f, 0.22f, 0.28f, 1.0f);
        ImVec4 colHover    = ImVec4(0.18f, 0.28f, 0.48f, 1.0f);

        // ============================================================
        //  [§B] POSITION (主動線、常時表示)
        // ============================================================
        //  Ctrl+G Quadrant Selector パネルと g_activeQuadrantMask を完全共有。
        //  この panel で checkbox を変更すると、Ctrl+G panel の同じ checkbox も
        //  自動で連動する (ImGui immediate mode + 同じ state を毎フレーム参照)。
        //
        //  bit 定数 (LiverLeftRightLabel::QuadrantMask と完全同期、ここでは
        //  軽量 header を維持するためミラー定義):
        //    QUAD_AR  = 0x01  ant_R (前面 ∩ 右葉)
        //    QUAD_AL  = 0x02  ant_L (前面 ∩ 左葉)
        //    QUAD_PR  = 0x04  pos_R (後面 ∩ 右葉)
        //    QUAD_PL  = 0x08  pos_L (後面 ∩ 左葉)
        //    QUAD_ALL = 0x0F  全選択 (= 旧 POS_CENTER と byte-identical)
        // -----------------------------------------------------------------
        constexpr uint8_t kMaskAR  = 0x01;
        constexpr uint8_t kMaskAL  = 0x02;
        constexpr uint8_t kMaskPR  = 0x04;
        constexpr uint8_t kMaskPL  = 0x08;
        constexpr uint8_t kMaskAll = 0x0F;
        constexpr uint8_t kMaskNone= 0x00;

        // ---- applyMaskChange (POSITION 変更時の共通ハンドラ) ----
        //   2x2 grid checkbox / Quick Presets ボタンのどちらから呼んでも
        //   同じ副作用を持つようまとめた lambda:
        //     1) state.activeQuadrantMask を更新 (UI 即時反映)
        //     2) onQuadrantMaskChanged 経由で main.cpp 側 g_activeQuadrantMask 同期
        //     3) mapQuadrantToOrientPreset で対応 Orient を引いて、現在値と
        //        違うときだけ state.initRotPreset と registrationHandle.initRotPreset
        //        を更新 (onInitRotPresetSilent 経由)。applyInitRotation は呼ばない。
        //
        //   結果: クアドラントを選んだ瞬間に ORIENTATION ヘッダの [Base/Right/Left]
        //   表示が切り替わり、Apply Init Pose をすぐ押せる状態になる。
        auto applyMaskChange = [&](uint8_t newMask) {
            state.activeQuadrantMask = newMask;
            if (actions.onQuadrantMaskChanged)
                actions.onQuadrantMaskChanged(newMask);

            const int autoPreset = mapQuadrantToOrientPreset(newMask);
            if (state.initRotPreset != autoPreset) {
                state.initRotPreset = autoPreset;
                if (actions.onInitRotPresetSilent)
                    actions.onInitRotPresetSilent(autoPreset);
            }
        };

        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 0.85f), "  POSITION");
        // [Phase 9a] Removed "Top=anterior, Left=patient's right" hint — the
        // 2x2 spatial mapping it described no longer applies in the 1x4
        // horizontal layout (just reading order: ant_R, ant_L, pos_R, pos_L).

        // ---- 1x4 grid (horizontal, Phase 9a; was 2x2) ----
        {
            const ImGuiTableFlags tableFlags =
                ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame;
            if (ImGui::BeginTable("##initorient_quad_grid", 4, tableFlags)) {
                auto checkbox_cell = [&](const char* shortName,
                                         int nv,
                                         uint8_t bit)
                {
                    bool on = (state.activeQuadrantMask & bit) != 0;
                    char label[64];
                    if (state.quadLabelsReady && nv >= 0) {
                        std::snprintf(label, sizeof(label),
                                      "%s\n(%d v)", shortName, nv);
                    } else {
                        std::snprintf(label, sizeof(label),
                                      "%s\n(--)", shortName);
                    }
                    ImGui::PushID(bit + 400);   // 衝突回避: Ctrl+G UI と別 ID 空間
                    if (ImGui::Checkbox(label, &on)) {
                        uint8_t newMask = state.activeQuadrantMask;
                        if (on)  newMask |= bit;
                        else     newMask = static_cast<uint8_t>(newMask & ~bit);
                        applyMaskChange(newMask);
                    }
                    ImGui::PopID();
                };

                // [Phase 9a] Single row, 4 columns. Same callbacks as 2x2.
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("ant_R", state.quadNAR, kMaskAR);
                ImGui::TableNextColumn();
                checkbox_cell("ant_L", state.quadNAL, kMaskAL);
                ImGui::TableNextColumn();
                checkbox_cell("pos_R", state.quadNPR, kMaskPR);
                ImGui::TableNextColumn();
                checkbox_cell("pos_L", state.quadNPL, kMaskPL);

                ImGui::EndTable();
            }
        }

        // ---- Mask 表示 + subset 頂点数 ----
        {
            const uint8_t m = state.activeQuadrantMask;
            // quadrantMaskString と同じ整形 (UI 軽量化のためここで簡易再実装)
            char maskBuf[32] = {0};
            if (m == kMaskNone) {
                std::snprintf(maskBuf, sizeof(maskBuf), "NONE");
            } else if (m == kMaskAll) {
                std::snprintf(maskBuf, sizeof(maskBuf), "ALL");
            } else {
                int pos = 0;
                pos += std::snprintf(maskBuf + pos, sizeof(maskBuf) - pos, "Q:");
                bool first = true;
                auto add = [&](const char* s) {
                    if (!first) pos += std::snprintf(maskBuf + pos, sizeof(maskBuf) - pos, "+");
                    pos += std::snprintf(maskBuf + pos, sizeof(maskBuf) - pos, "%s", s);
                    first = false;
                };
                if (m & kMaskAR) add("AR");
                if (m & kMaskAL) add("AL");
                if (m & kMaskPR) add("PR");
                if (m & kMaskPL) add("PL");
            }
            if (state.quadLabelsReady) {
                ImGui::Text("  Mask: %s (0x%02X)  Subset: %d / %d",
                            maskBuf, (unsigned)m,
                            state.quadSubsetCount, state.quadTotalCount);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                   "  Mask: %s (0x%02X)  [labels not computed - press Shift+R, Y]",
                                   maskBuf, (unsigned)m);
            }
        }

        // ---- Quick Presets (1 行) ----
        //   旧 Right / Center / Left ボタンの後方互換 + Anterior / Posterior 追加。
        //   合計 5 ボタンを横並びで配置。
        ImGui::Spacing();
        {
            struct Preset { const char* label; uint8_t mask; };
            Preset presets[5] = {
                {"All",       kMaskAll},                  // = 旧 Center
                {"Right",     (uint8_t)(kMaskAR|kMaskPR)}, // = 旧 Right (患者右葉)
                {"Left",      (uint8_t)(kMaskAL|kMaskPL)}, // = 旧 Left  (患者左葉)
                {"Anterior",  (uint8_t)(kMaskAR|kMaskAL)}, // 前面のみ (新規)
                {"Posterior", (uint8_t)(kMaskPR|kMaskPL)}, // 後面のみ (新規)
            };
            float qpW = (totalW - 4.0f * 4.0f) / 5.0f;   // 5 ボタン、間隔 4px
            for (int i = 0; i < 5; i++) {
                if (i > 0) ImGui::SameLine(0, 4);
                bool isSel = (state.activeQuadrantMask == presets[i].mask);
                ImGui::PushID(i + 500);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
                if (isSel) {
                    ImGui::PushStyleColor(ImGuiCol_Button,
                                          ImVec4(colActive.x*0.25f, colActive.y*0.25f, colActive.z*0.25f, 1));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                          ImVec4(colActive.x*0.35f, colActive.y*0.35f, colActive.z*0.35f, 1));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                          ImVec4(colActive.x*0.45f, colActive.y*0.45f, colActive.z*0.45f, 1));
                    ImGui::PushStyleColor(ImGuiCol_Text, colActive);
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button,
                                          ImVec4(colInactive.x, colInactive.y, colInactive.z, 1));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                          ImVec4(colHover.x, colHover.y, colHover.z, 1));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                          ImVec4(colHover.x*1.2f, colHover.y*1.2f, colHover.z*1.2f, 1));
                    ImGui::PushStyleColor(ImGuiCol_Text,
                                          ImVec4(0.55f, 0.60f, 0.70f, 1.0f));
                }
                if (ImGui::Button(presets[i].label, ImVec2(qpW, btnH))) {
                    applyMaskChange(presets[i].mask);
                }
                ImGui::PopStyleColor(4);
                ImGui::PopStyleVar();
                ImGui::PopID();
            }
        }

        // ============================================================
        //  [§C] ORIENTATION (CollapsingHeader, 既定 OFF)
        // ============================================================
        //  AutoQCR が 9 通り自動 sweep するため通常は閉じたまま。
        //  ヘッダタイトルに現在の preset 名 ([Base] / [Up-R+] 等) を出すので、
        //  AutoQCR 実行後の選択結果は折りたたんだまま確認できる。
        //  手動で別 preset を試したい時だけ展開する。
        ImGui::Spacing();
        {
            char orientHdr[64];
            std::snprintf(orientHdr, sizeof(orientHdr),
                          "ORIENTATION  [%s]###orient_hdr",
                          presetLabel(state.initRotPreset));

            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(0.45f, 0.55f, 0.70f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(0.10f, 0.12f, 0.16f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.15f, 0.18f, 0.24f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive,  ImVec4(0.20f, 0.24f, 0.32f, 0.8f));
            bool orientOpen = ImGui::CollapsingHeader(orientHdr);
            ImGui::PopStyleColor(4);
            orientExpanded_ = orientOpen;

            if (orientOpen) {
                struct PresetBtn { int id; const char* label; };
                // Radiology 慣例: 患者の右 (Right) を画面左、患者の左 (Left)
                // を画面右に配置する。enum 番号は維持して動作は変えない
                // (動作は getPresetRotation 側で d_lr に基づいて決まる)。
                // ボタンの位置とラベルの対応だけ反転。
                //
                // チャット 11 拡張: 外側 2 列 (Right+/Left+ ファミリ) を追加し、
                //   3x5 グリッドに。外側 = ±40° (dx=±2)、内側 = ±20° (dx=±1)、
                //   中央列 = 既存 (dx=0)。ラベルは末尾 "+" で強めを明示。
                PresetBtn grid[3][5] = {
                                        { { 9,"Up-R+"}, {2,"Up-R"},  {1,"Up"},   {8,"Up-L"}, {12,"Up-L+"} },
                                        { {10,"Right+"},{3,"Right"}, {0,"Base"}, {7,"Left"}, {13,"Left+"} },
                                        { {11,"Dn-R+"}, {4,"Dn-R"},  {5,"Down"}, {6,"Dn-L"}, {14,"Dn-L+"} },
                                        };

                // 5 列用ローカル width。4 個の spacing (4px each) 分を totalW
                // から控除して 5 等分。
                float btnW_orient = (totalW - 4.0f * 4.0f) / 5.0f;

                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 5; col++) {
                        if (col > 0) ImGui::SameLine(0, 4);
                        int pid    = grid[row][col].id;
                        bool isSel = (state.initRotPreset == pid);
                        ImGui::PushID(pid + 200);
                        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
                        if (isSel) {
                            ImGui::PushStyleColor(ImGuiCol_Button,
                                                  ImVec4(colActive.x*0.25f, colActive.y*0.25f, colActive.z*0.25f, 1));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                                  ImVec4(colActive.x*0.35f, colActive.y*0.35f, colActive.z*0.35f, 1));
                            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                                  ImVec4(colActive.x*0.45f, colActive.y*0.45f, colActive.z*0.45f, 1));
                            ImGui::PushStyleColor(ImGuiCol_Text, colActive);
                        } else {
                            ImGui::PushStyleColor(ImGuiCol_Button,
                                                  ImVec4(colInactive.x, colInactive.y, colInactive.z, 1));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                                  ImVec4(colHover.x, colHover.y, colHover.z, 1));
                            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                                  ImVec4(colHover.x*1.2f, colHover.y*1.2f, colHover.z*1.2f, 1));
                            ImGui::PushStyleColor(ImGuiCol_Text,
                                                  ImVec4(0.55f, 0.60f, 0.70f, 1.0f));
                        }
                        if (ImGui::Button(grid[row][col].label, ImVec2(btnW_orient, btnH))) {
                            state.initRotPreset = pid;
                            if (actions.onInitRotPresetChanged)
                                actions.onInitRotPresetChanged(pid);
                        }
                        ImGui::PopStyleColor(4);
                        ImGui::PopStyleVar();
                        ImGui::PopID();
                    }
                }
            }
        }

        // ============================================================
        //  [§D] Apply Init Pose moved to drawRegistrationSection (Phase 9d):
        //  it now lives OUTSIDE this CollapsingHeader so it stays visible even
        //  when INITIAL ORIENTATION is folded.
        // ============================================================

        ImGui::Spacing();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    void drawRegistrationSection() {
        int phase = currentPhase();
        bool regDone = state.regState == 4;
        bool regActive = regPhaseActive_ && !regDone;
        bool processing = (state.regState > 0 && state.regState < 4 && state.regMethod >= 0 && state.regMethod != 2);

        if (state.mainMode == 1) {
            drawSectionWithBar("REGISTRATION", colReg(), true, false, false);
            ImGui::Indent(16);
            ImGui::TextColored(colReg(), "  Registration: Done");
            ImGui::Unindent(16);
            ImGui::Spacing(); ImGui::Separator();
            return;
        }

        if (!regPhaseActive_ && state.mainMode == 0) {
            drawSectionWithBar("REGISTRATION", colReg(), false, false, false);
            ImGui::Indent(16);
            if (!state.depthDone) {
                ImGui::TextColored(ImVec4(0.18f,0.19f,0.22f,1), "  Run Depth first");
            } else {
                ImGui::Spacing();
                if (glowButton("Proceed to Registration >>", colReg(), false, -1, 38)) {
                    regPhaseActive_ = true;
                }
                ImGui::Spacing();
            }
            ImGui::Unindent(16);
            ImGui::Spacing(); ImGui::Separator();
            return;
        }

        drawSectionWithBar("REGISTRATION", colReg(), regDone, regActive || regDone, processing);

        bool depthBusy = (state.cameraState == 1 || state.cameraState == 2 || state.depthRunning);
        if (depthBusy && state.regState < 1) {
            ImGui::Indent(16);
            ImGui::TextColored(ImVec4(0.18f,0.19f,0.22f,1), "  Complete Depth first");
            ImGui::Unindent(16);
            ImGui::Spacing(); ImGui::Separator();
            return;
        }

        ImGui::Indent(16); ImGui::PushItemWidth(-16);

        bool anyP = (state.regState > 0 && state.regState < 4);

        // [Phase 8] INITIAL ORIENTATION as a collapsing header. Label shows
        // current Q-mask + Orient preset so it stays readable when collapsed.
        {
            constexpr uint8_t kMaskNone = 0x00;
            constexpr uint8_t kMaskAll  = 0x0F;
            char qstr[24];
            if (state.activeQuadrantMask == kMaskNone)      std::snprintf(qstr, sizeof(qstr), "NONE");
            else if (state.activeQuadrantMask == kMaskAll)  std::snprintf(qstr, sizeof(qstr), "ALL");
            else std::snprintf(qstr, sizeof(qstr), "0x%02X", (unsigned)state.activeQuadrantMask);

            char hdrLabel[96];
            std::snprintf(hdrLabel, sizeof(hdrLabel),
                          "INITIAL ORIENTATION  [Q:%s | %s]###initorient_hdr",
                          qstr, presetLabel(state.initRotPreset));

            // [Phase 9c] One-shot fold from Apply Init Pose. Cleared on consume
            // so the user can manually re-open the header any time afterwards.
            if (initOrientShouldCollapse_) {
                ImGui::SetNextItemOpen(false, ImGuiCond_Always);
                initOrientShouldCollapse_ = false;
            }
            // Auto-open before any registration / labels exist; fold up after.
            else if (!state.useRegistration && !state.quadLabelsReady) {
                ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            }

            ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(0.10f, 0.12f, 0.16f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.15f, 0.18f, 0.24f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive,  ImVec4(0.20f, 0.24f, 0.32f, 0.9f));
            ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.55f, 0.70f, 0.90f, 1.0f));
            bool ioOpen = ImGui::CollapsingHeader(hdrLabel);
            ImGui::PopStyleColor(4);

            if (ioOpen) {
                drawInitOrientationPanel();  // contents only (Apply moved out, Phase 9d)
            }
        }

        // [Phase 9d] Apply Init Pose — main-line button, ALWAYS visible (outside
        // the CollapsingHeader). Disabled if no quadrant selected. Pressing it
        // also triggers the Phase 9c one-shot auto-fold on the next frame.
        {
            constexpr uint8_t kMaskNone_apply = 0x00;
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.10f, 0.30f, 0.55f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.40f, 0.70f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.20f, 0.50f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.95f, 0.97f, 1.00f, 1.0f));
            bool empty = (state.activeQuadrantMask == kMaskNone_apply);
            if (empty) ImGui::BeginDisabled();
            if (ImGui::Button("Apply Init Pose", ImVec2(-1, 28.0f))) {
                if (actions.onApplyInitPose) actions.onApplyInitPose();
                initOrientShouldCollapse_ = true;   // [Phase 9c]
            }
            if (empty) ImGui::EndDisabled();
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar();
        }
        ImGui::Spacing();

        // --- STAGE 1: Hemi Quad (Shift+O, 2/3 width) + Probe (1/3 width) ---
        // [Phase 8 rename] Hemi Auto -> Hemi Quad (Shift+O): calls onQuadAuto
        // (AR-fixed-view ∩ quadrant intersection, the improved method) instead
        // of onHemiAuto. onHemiAuto is preserved in RegUIActions for keyboard
        // and other code paths; only this sidebar button is rerouted. The
        // v:0.50 voxel readout is dropped (same value lives in Debug Panel O
        // tab and sidebar Advanced).
        {
            const float gap     = 4.0f;
            const float availW  = ImGui::GetContentRegionAvail().x;
            const float hemiW   = (availW - gap) * 0.66f;

            if(glowButton("Hemi Quad (Shift+O)", colReg(),
                          anyP && state.regMethod!=1, hemiW, 52,
                          state.btnIconTex[RegUIState::ICON_HEMI_AUTO])) {
                state.regMethod = 1;
                if(actions.onQuadAuto) actions.onQuadAuto();
            }

            ImGui::SameLine(0.0f, gap);
            if(glowButton("Probe", colReg(), anyP, -1, 52, 0)) {
                if(actions.onAutoProbe) actions.onAutoProbe();
            }
        }
        ImGui::Spacing();

        // ----------------------------------------------------------------
        //  AutoQCR (Alt+Ctrl+P): 9-preset 自動 sweep + 6-DoF/7-DoF 切替
        // ----------------------------------------------------------------
        //  Hemi Auto / AutoProbe 行の直下に同じ寸法で配置。左 2/3 が
        //  AutoQCR ボタン、右 1/3 が 6-DoF lock チェックボックス。
        //  チェック ON (default) = rigid (DICOM mm + metric depth)、
        //  チェック OFF = 7-DoF T+R+Scale (Shift+Ctrl+P 互換)。
        // [Phase 8] AutoQCR is now a full-width button. The 6-DoF/7-DoF lock
        // checkbox moved into the QCR Tuning collapsing header below (keeps the
        // main button clean). `anyP` disables it only while another
        // registration is running; prerequisites are guarded inside
        // runAutoQuadCyclicRansac.
        {
            if(glowButton("AutoQCR", colReg(), anyP, -1, 52, 0)) {
                if(actions.onAutoQCR) actions.onAutoQCR(state.autoQcrLockScale);
            }
        }
        ImGui::Spacing();

        // [Phase 9b] QCR Tuning block merged into the "Tuning & Advanced"
        // collapsing header near the bottom of this section.

        // ----------------------------------------------------------------
        //  [Ctrl+G] V3-R BIPOP-CMA-ES (主動線、メイン Refinement)
        // ----------------------------------------------------------------
        //  Apply Init Pose → AutoQCR の後、g_activeQuadrantMask で絞った
        //  4-quadrant subset に対し region-aware BIPOP-CMA-ES を回す。
        //  キーボード Ctrl+G と等価。実体は main.cpp 側 onCtrlG lambda が
        //  poseAutoSaveBeforeRegistration → runBipopCmaesV3R(mask) →
        //  poseSaveToLibrary(RMSE, mask) を順に呼ぶ。
        //
        //  [UI整理] 以前ここには BIPOP-CMA-ES [Shift+V] と Silhouette
        //  Alignment [Shift+E] の 2 ボタンが並んでいたが、日常運用で
        //  使われないため REGISTRATION 末尾の Advanced CollapsingHeader
        //  に退避し、ここを Ctrl+G 専用とした。
        {
            // 有効条件: registration phase 中で、quadrant ラベルが計算済み。
            // useRegistration (= 既存 reg 完了済み) は問わない: Ctrl+G は
            // Apply Init Pose 直後でも (refinement なしの状態から) 走らせる
            // ことがある。
            bool ctrlgDisabled = !state.quadLabelsReady
                                 || state.activeQuadrantMask == 0x00;

            // AutoQCR と同じ 2/3 + 1/3 レイアウト: ボタン本体 (左) +
            // 6-DoF チェックボックス (右)。
            // [Phase 8] Ctrl+G full-width button (shorter label). The 6-DoF
            // lock checkbox moved into the Advanced collapsing header below
            // (the Debug Panel G tab covers the 4-DoF / search-dimension radio).
            if (glowButton("Ctrl+G", colReg(), ctrlgDisabled, -1, 56))
            {
                state.regMethod = 3;   // BIPOP method (Shift+V/F/G と同じ slot)
                if (actions.onCtrlG) actions.onCtrlG();
            }
        }

        // [Phase 8] Instrument Px Threshold slider relocated to Debug Panel
        // O tab (always accessible there, not conditional). Sidebar stays clean.

        // --- Iter Probe (K回 AutoProbe を連続呼び出し) ---
        //   [UI整理 - 削除済] AutoQCR で 9 preset 自動 sweep ができるため、
        //   Iter Probe (= AutoProbe を K 回繰り返し) はもはや日常動線では
        //   使われない。UI からは外したが、actions.onIterativeAutoProbe と
        //   state.iterCycles は将来の再評価実験用に温存している。
        //   復活させるときはここに button + DragInt を再配置する。

        // --- STAGE 2-3: BIPOP-CMA-ES / Silhouette Alignment ---
        //   [UI整理 - Advanced 移動済] 旧 Shift+V / Shift+E ボタンは、Ctrl+G
        //   がメイン refinement になった現運用ではほぼ使われない。
        //   REGISTRATION 末尾の Advanced CollapsingHeader 内に退避済み
        //   (キーボードショートカット Shift+V / Shift+E は引き続き有効)。

        // --- Manual Registration: Umeyama ---
        if(methodButton("Umeyama Manual", "", state.regMethod==2, state.regState, anyP && state.regMethod!=2, state.btnIconTex[RegUIState::ICON_UMEYAMA])) {
            state.regMethod = 2; if(actions.onStartUmeyama) actions.onStartUmeyama();
        }
        if (state.regMethod == 2 && state.regState >= 1 && state.regState <= 3 && !state.splitScreen) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.04f,0.02f,0.01f,1));
            ImGui::BeginChild("##umeya", ImVec2(-1,190), true);
            if(state.regState == 1)      ImGui::TextColored(colMuted(), "  RIGHT: click board points");
            else if(state.regState == 2) ImGui::TextColored(colMuted(), "  LEFT: click liver points");
            else if(state.regState == 3) ImGui::TextColored(colGreen(), "  Ready! Click Execute");
            ImGui::Spacing();
            drawColoredPointProgress("Board", state.boardPtCount, state.targetPtCount, state.regState == 1);
            ImGui::Spacing();
            drawColoredPointProgress("Object", state.objPtCount, state.targetPtCount, state.regState == 2);
            ImGui::Spacing();
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            bool canUndo = (state.boardPtCount + state.objPtCount) > 0;
            if(colorButton("Undo", colDim(), false, !canUndo, bw2)) {
                if(actions.onUndoUmeyamaPoint) actions.onUndoUmeyamaPoint();
            }
            ImGui::SameLine();
            if(colorButton("Execute", colGreen(), false, state.regState != 3, bw2)) {
                if(actions.onExecuteUmeyama) actions.onExecuteUmeyama();
            }
            ImGui::EndChild(); ImGui::PopStyleColor();
        }

        ImGui::Spacing();
        // [Phase 8] Pose Library label includes entry count when > 0.
        {
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            char libLabel[48];
            if (state.poseEntryCount > 0) {
                std::snprintf(libLabel, sizeof(libLabel),
                              state.poseLibraryOpen ? "Pose Library (%d) ON"
                                                    : "Pose Library (%d)",
                              state.poseEntryCount);
            } else {
                std::snprintf(libLabel, sizeof(libLabel),
                              state.poseLibraryOpen ? "Pose Library ON"
                                                    : "Pose Library");
            }
            if(colorButton(libLabel,
                           state.poseLibraryOpen ? colGreen() : colReg(), false, false, bw2)) {
                if(actions.onPoseLibraryToggle) actions.onPoseLibraryToggle();
            }
            ImGui::SameLine();
            if(colorButton("Pose Undo", state.poseUndoAvailable ? colRed() : colDim(),
                            false, !state.poseUndoAvailable, bw2)) {
                if(actions.onPoseUndo) actions.onPoseUndo();
            }
        }

        // [Phase 8] Reset is now part of the 3-column footer at the bottom of
        // this section (Deform / Depth / Reset). Reset no longer auto-toggles
        // cluster viz off — that belongs to the user (Debug Panel Viz tab).
        // [PHASE-2] Cluster / CorresPoints viz toggles relocated to
        // Debug Panel > Viz tab (Ctrl+D).

        // ---- QCR Tuning は AutoQCR の直下に移動済み (上部参照) ----
        ImGui::Spacing();

        // ====================================================================
        //  Advanced (CollapsingHeader, 既定 OFF)
        // --------------------------------------------------------------------
        //  日常運用ではほぼ触らない項目をここに集約:
        //    - Voxel size (= g_voxelSize): 全 FGR+ICP 系登録の解像度
        //    - Ctrl+G / Ctrl+Shift+G の RIM-weighted / raycast コントロール
        //      (main.cpp 側 drawCtrlGRimRaycastControls を onDrawAdvancedCtrlG
        //      callback 経由で呼ぶ。floating "Ctrl+G Quadrant Selector" 窓と
        //      同じ globals を操作するので両方表示でも同期する。)
        //
        //  [整理履歴]
        //    - BIPOP-CMA-ES [Shift+V] / Silhouette Alignment [Shift+E] ボタン
        //      は現運用で使われないため一旦削除 (キーボード Shift+V / Shift+E
        //      は引き続き有効)。
        //    - Voxel UI も regMethod==1 (Hemi Auto 選択時) 限定だった旧表示を
        //      ここに常時表示として移動済み。
        // ====================================================================
        // [Phase 9b] Merged "QCR Tuning" + "Advanced" into a single end-of-
        // section header so the two debug/expert headers no longer occupy two
        // separate rows. Layout: AutoQCR section -> separator -> Ctrl+G section.
        if (ImGui::CollapsingHeader("Tuning & Advanced")) {
            ImGui::Indent(8);

            // ===== AutoQCR section =====
            ImGui::TextColored(colMuted(), "AutoQCR (Alt+Ctrl+P / Shift+Ctrl+P):");
            {
                bool lockScale = state.autoQcrLockScale;
                if (ImGui::Checkbox("AutoQCR 6-DoF lock (scale=1)##qcr_lock", &lockScale)) {
                    const_cast<RegUIState&>(state).autoQcrLockScale = lockScale;
                }
            }
            ImGui::Spacing();
            ImGui::TextColored(colMuted(), "Subset size K:");
            ImGui::SliderInt("##qcrK", &g_qcrSubsetK, 3, 5, "K = %d");
            ImGui::TextColored(ImVec4(0.45f,0.45f,0.5f,1),
                               "  K=3: exact fit  K=4: balanced  K=5: stable");
            ImGui::Spacing();
            ImGui::TextColored(colMuted(), "Max trials (Stage 1 cap):");
            ImGui::SliderInt("##qcrCap", &g_qcrMaxTrials, 10000, 500000, "%d");
            ImGui::Spacing();
            ImGui::TextColored(colMuted(), "Max axis rotation (per-axis):");
            ImGui::SliderFloat("##qcrMaxAxis", &g_qcrMaxAxisRotDeg,
                               5.0f, 90.0f, "%.1f deg");
            ImGui::TextColored(colMuted(), "Max total rotation (axis-angle):");
            ImGui::SliderFloat("##qcrMaxTotal", &g_qcrMaxTotalRotDeg,
                               5.0f, 90.0f, "%.1f deg");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // ===== Ctrl+G section =====
            ImGui::TextColored(colMuted(), "Ctrl+G / Ctrl+Shift+G (V3-R):");
            {
                bool lockScale = state.ctrlgLockScale;
                if (ImGui::Checkbox("Ctrl+G 6-DoF lock (scale=1)##ctrlg_lock_adv", &lockScale)) {
                    const_cast<RegUIState&>(state).ctrlgLockScale = lockScale;
                    if (actions.onCtrlgLockScaleChanged)
                        actions.onCtrlgLockScaleChanged(lockScale);
                }
            }
            ImGui::Spacing();
            drawVoxelInfo();
            ImGui::Spacing();
            if (actions.onDrawAdvancedCtrlG) {
                ImGui::PushID("##adv_ctrlg");
                actions.onDrawAdvancedCtrlG();
                ImGui::PopID();
            }

            ImGui::Unindent(8);
        }
        ImGui::Spacing();

        // [Phase 8] Footer: 3-column nav (Deform / Depth / Reset). All three
        // are "leave this section" actions; grouping them communicates that.
        // Reset stays red as a warning.
        ImGui::Spacing(); ImGui::Spacing();
        {
            float bw3 = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
            bool canDeform = (state.regState == 4 && state.mainMode == 0);

            if(canDeform) {
                if(glowButton("Deform >>", colDeform(), false, bw3, 36)) {
                    if (actions.onSwitchToDeformMode) actions.onSwitchToDeformMode();
                }
            } else {
                colorButton("Deform >>", colDim(), false, true, bw3, 36);
            }
            ImGui::SameLine();
            if(colorButton("<< Depth", colDepth(), false, false, bw3, 36)) {
                regPhaseActive_ = false;
                if (actions.onResetRegistration) actions.onResetRegistration();
            }
            ImGui::SameLine();
            if(colorButton("Reset", colRed(), false, false, bw3, 36)) {
                if (actions.onResetRegistration) actions.onResetRegistration();
            }
        }

        ImGui::PopItemWidth(); ImGui::Unindent(16);
        ImGui::Spacing(); ImGui::Separator();
    }

    void drawDeformSection() {
        int phase = currentPhase();
        drawSectionWithBar("DEFORM", colDeform(), false, phase==2, false);

        if (state.mainMode != 1) {
            ImGui::Indent(16);
            if (state.regState == 4)
                ImGui::TextColored(colMuted(), "  Complete Registration to proceed");
            else
                ImGui::TextColored(ImVec4(0.18f,0.19f,0.22f,1), "  Complete Registration first");
            ImGui::Unindent(16);
            ImGui::Spacing(); ImGui::Separator();
            return;
        }

        ImGui::Indent(16); ImGui::PushItemWidth(-16);
        ImGui::TextColored(colDeform(), "SUB MODE");
        ImGui::Spacing();
        auto dmBtn = [&](const char* l, const char* sc, int mv, float w, unsigned int iconTex=0) -> bool {
            bool isA = state.deformState == mv;
            ImGui::PushID(l); ImVec4 c = colDeform();
            if(isA) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.15f,c.y*0.15f,c.z*0.15f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(c.x*0.22f,c.y*0.22f,c.z*0.22f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(c.x*0.30f,c.y*0.30f,c.z*0.30f,1));
                ImGui::PushStyleColor(ImGuiCol_Text, c);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.06f,0.065f,0.08f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.10f,0.11f,0.14f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.14f,0.15f,0.19f,1));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.25f,0.26f,0.32f,1));
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
            char paddedLabel[128];
            if (iconTex) snprintf(paddedLabel, sizeof(paddedLabel), "       %s", l);
            else snprintf(paddedLabel, sizeof(paddedLabel), "%s", l);
            ImVec2 btnPos = ImGui::GetCursorScreenPos();
            bool cl = ImGui::Button(paddedLabel, ImVec2(w, 36));
            if (iconTex) drawButtonIcon(iconTex, btnPos, 36);
            ImGui::PopStyleVar(); ImGui::PopStyleColor(4); ImGui::PopID();
            return cl;
        };
        if(dmBtn("Rigid","",0,-1,state.btnIconTex[RegUIState::ICON_RIGID])) { if(actions.onRigidMode) actions.onRigidMode(); }
        ImGui::Spacing();
        if(dmBtn("Handle","",1,-1,state.btnIconTex[RegUIState::ICON_HANDLE])) { if(actions.onHandlePlaceMode) actions.onHandlePlaceMode(); }
        ImGui::Spacing();
        if(dmBtn("Deform","",2,-1,state.btnIconTex[RegUIState::ICON_DEFORM])) { if(actions.onDeformMode) actions.onDeformMode(); }
        if(state.deformState == 1) {
            ImGui::Spacing();
            drawProgress("Handle Groups", state.handleGroups, state.maxHandleGroups, colDeform());
            ImGui::Spacing();
            bool locked = state.handleGroups > 0;
            ImGui::TextColored(locked ? colMuted() : colDeform(), "Sphere Radius");
            if (locked) {
                ImGui::PushStyleColor(ImGuiCol_SliderGrab,       ImVec4(0.25f,0.26f,0.32f,1));
                ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.25f,0.26f,0.32f,1));
                ImGui::PushStyleColor(ImGuiCol_FrameBg,          ImVec4(0.08f,0.08f,0.10f,1));
                ImGui::PushStyleColor(ImGuiCol_Text,             colMuted());
                float r = state.handleRadius;
                ImGui::SliderFloat("##hr", &r, 0.3f, 3.0f, "%.2f");
                ImGui::PopStyleColor(4);
            } else {
                ImVec4 cd = colDeform();
                ImGui::PushStyleColor(ImGuiCol_SliderGrab,       cd);
                ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, cd);
                ImGui::PushStyleColor(ImGuiCol_FrameBg,          ImVec4(0.1f,0.1f,0.13f,1));
                float r = state.handleRadius;
                if (ImGui::SliderFloat("##hr", &r, 0.3f, 3.0f, "%.2f")) {
                    if (actions.onHandleRadiusChanged) actions.onHandleRadiusChanged(r);
                }
                ImGui::PopStyleColor(3);
            }
        }
        ImGui::Spacing(); ImGui::Spacing();
        if(colorButton("Reset All", colRed())) { if(actions.onFullReset) actions.onFullReset(); }

        ImGui::Spacing();
        if(colorButton("Start From Depth", ImVec4(0.9f,0.4f,0.1f,1))) {
            showRestartConfirm_ = true;
            ImGui::OpenPopup("##RestartConfirm");
        }

        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.08f, 0.08f, 0.10f, 0.95f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 20));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        if(ImGui::BeginPopupModal("##RestartConfirm", &showRestartConfirm_,
                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Restart from Depth?");
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                               "All deform and registration progress\nwill be lost. Meshes will be reloaded\nfrom original files.");
            ImGui::Spacing(); ImGui::Spacing();
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.15f, 0.1f, 1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.2f, 0.15f, 1));
            if(ImGui::Button("Yes, Restart", ImVec2(bw2, 36))) {
                if(actions.onStartFromDepth) actions.onStartFromDepth();
                showRestartConfirm_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            if(ImGui::Button("Cancel", ImVec2(bw2, 36))) {
                showRestartConfirm_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();

        ImGui::PopItemWidth(); ImGui::Unindent(16);
        ImGui::Spacing(); ImGui::Separator();
    }

    // ---- Step 7: Intrinsics panel (案 Y: dropdown + Active card + 3 tabs) ----
    void drawIntrinsicsSource(const char* suffix = "") {
        ImGui::Spacing();
        char barId[64]; snprintf(barId, sizeof(barId), "##intrTabs%s", suffix);
        if (ImGui::BeginTabBar(barId, ImGuiTabBarFlags_None)) {
            char id[64];

            snprintf(id, sizeof(id), "Source##%s", suffix);
            if (ImGui::BeginTabItem(id)) {
                drawIntrinsicsSourceTab_(suffix);
                ImGui::EndTabItem();
            }

            // "Run Calibration…" を押すと次フレームで Settings タブへ遷移する。
            ImGuiTabItemFlags setFlags = intrinsicsWantSettingsTab_
                                       ? ImGuiTabItemFlags_SetSelected : 0;
            intrinsicsWantSettingsTab_ = false;
            snprintf(id, sizeof(id), "Settings##%s", suffix);
            if (ImGui::BeginTabItem(id, nullptr, setFlags)) {
                drawIntrinsicsSettingsTab_(suffix);
                ImGui::EndTabItem();
            }

            snprintf(id, sizeof(id), "Take Picture##%s", suffix);
            if (ImGui::BeginTabItem(id)) {
                ImGui::Spacing();
                ImGui::TextColored(colDim(), "Coming soon");
                ImGui::TextWrapped("In-app chessboard capture then Run Calibration "
                                   "(Step 8).");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::Spacing();
    }

    // Active source -> short label + pill color for the Active card.
    void intrinsicsSourceLabelColor_(const char*& label, ImVec4& col) const {
        switch (state.intrinsicsSource) {
            case IntrinsicsSource::Custom: label = "Custom"; col = ImVec4(0.9f,0.7f,0.2f,1); break;
            case IntrinsicsSource::Calib:  label = "Calib";  col = ImVec4(0.9f,0.4f,0.6f,1); break;
            case IntrinsicsSource::Preset: label = "Preset"; col = ImVec4(0.2f,0.6f,1.0f,1); break;
            case IntrinsicsSource::DA3:    label = "DA3";    col = ImVec4(0.3f,0.8f,0.6f,1); break;
            default:                       label = "?";      col = colDim(); break;
        }
    }

    void drawIntrinsicsSourceTab_(const char* suffix) {
        ImGui::Spacing();
        // ---- Active intrinsics card ----
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07f,0.075f,0.095f,1));
        ImGui::BeginChild("##activeK", ImVec2(0, 100), true);
        {
            const char* srcLabel; ImVec4 pill;
            intrinsicsSourceLabelColor_(srcLabel, pill);
            ImGui::TextColored(colDim(), "Active intrinsics");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(pill.x*0.30f,pill.y*0.30f,pill.z*0.30f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(pill.x*0.30f,pill.y*0.30f,pill.z*0.30f,1));
            ImGui::PushStyleColor(ImGuiCol_Text, pill);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
            ImGui::SmallButton(srcLabel);
            ImGui::PopStyleVar(); ImGui::PopStyleColor(3);

            if (!state.currentDisplayName.empty())
                ImGui::TextColored(colDim(), "%s", state.currentDisplayName.c_str());
            bool haveK = (state.currentWidth > 0 && state.currentHeight > 0);
            if (!haveK) {
                ImGui::TextColored(colMuted(), "(no K loaded; run depth to populate)");
            } else {
                ImGui::Text("fx %.2f   fy %.2f", state.currentFx, state.currentFy);
                ImGui::Text("cx %.2f   cy %.2f", state.currentCx, state.currentCy);
                ImGui::TextColored(colDim(), "res %d x %d",
                                   state.currentWidth, state.currentHeight);
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        ImGui::Spacing();
        // ---- Source dropdown (priority order) ----
        ImGui::TextColored(colDim(), "Source");
        const std::string preview = state.currentDisplayName.empty()
                                   ? std::string("(select source)") : state.currentDisplayName;
        // green "o" = available, dim "-" = missing; then a Selectable.
        // Unavailable entries are disabled (greyed + non-clickable).
        auto entry = [&](const char* label, bool available, bool selected) -> bool {
            ImGui::PushStyleColor(ImGuiCol_Text, available ? colGreen() : colMuted());
            ImGui::TextUnformatted(available ? "o" : "-");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGuiSelectableFlags flags = available ? 0 : ImGuiSelectableFlags_Disabled;
            return ImGui::Selectable(label, selected, flags);
        };
        char comboId[64]; snprintf(comboId, sizeof(comboId), "##srcCombo%s", suffix);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo(comboId, preview.c_str())) {
            // 1. Custom
            if (entry("Custom (intrinsics_custom.txt)", state.customAvailable,
                      state.intrinsicsSource == IntrinsicsSource::Custom)) {
                if (actions.onSourceChanged) actions.onSourceChanged(IntrinsicsSource::Custom);
            }
            // 2. Calib (last)
            if (entry("Calib (last)", state.calibLastAvailable,
                      state.intrinsicsSource == IntrinsicsSource::Calib)) {
                if (actions.onSourceChanged) actions.onSourceChanged(IntrinsicsSource::Calib);
            }
            // 3. Factory presets
            ImGui::Separator();
            ImGui::TextColored(colDim(), "Factory presets");
            for (auto& p : state.presetList) {
                if (p.dynamic) continue;
                bool sel = (state.intrinsicsSource == IntrinsicsSource::Preset
                            && state.currentPresetKey == p.key);
                char lbl[112]; snprintf(lbl, sizeof(lbl), "%s##preset_%s",
                                        p.displayName.c_str(), p.key.c_str());
                if (entry(lbl, p.available, sel)) {
                    if (actions.onPresetChanged) actions.onPresetChanged(p.key);
                }
            }
            // Dynamic presets
            bool anyDyn = false;
            for (auto& p : state.presetList) if (p.dynamic) { anyDyn = true; break; }
            if (anyDyn) {
                ImGui::Separator();
                ImGui::TextColored(colDim(), "Dynamic");
                for (auto& p : state.presetList) {
                    if (!p.dynamic) continue;
                    bool sel = (state.intrinsicsSource == IntrinsicsSource::Preset
                                && state.currentPresetKey == p.key);
                    char lbl[112]; snprintf(lbl, sizeof(lbl), "%s##dyn_%s",
                                            p.displayName.c_str(), p.key.c_str());
                    if (entry(lbl, p.available, sel)) {
                        if (actions.onPresetChanged) actions.onPresetChanged(p.key);
                    }
                }
            }
            // Saved calibrations (Step 6 placeholder)
            ImGui::Separator();
            ImGui::TextColored(colMuted(), "Saved calibrations: (Step 6)");
            // 4. DA3 (file-driven: intrinsics_da3_last.txt). Disabled until a
            //    Run Depth has produced an estimate.
            ImGui::Separator();
            if (entry("DA3 (last estimate)", state.da3LastAvailable,
                      state.intrinsicsSource == IntrinsicsSource::DA3)) {
                if (actions.onSourceChanged) actions.onSourceChanged(IntrinsicsSource::DA3);
            }
            ImGui::EndCombo();
        }

        ImGui::Spacing();
        // ---- Buttons ----
        float bw = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        bool noK = (state.currentWidth <= 0 || state.currentHeight <= 0);  // no valid K loaded
        if (colorButton("Save current K as Custom", colYellow(), false, noK, bw)) {
            if (actions.onSaveAsCustom) actions.onSaveAsCustom();
        }
        ImGui::SameLine();
        if (colorButton("Run Calibration...", ImVec4(0.9f,0.4f,0.6f,1), false, false, bw)) {
            intrinsicsWantSettingsTab_ = true;
        }
    }

    void drawIntrinsicsSettingsTab_(const char* suffix) {
        ImGui::Spacing();
        // ---- Custom intrinsics ----
        ImGui::TextColored(colYellow(), "Custom intrinsics");
        ImGui::TextWrapped("File: intrinsics_custom.txt  (edit externally, then Reload)");
        float bw = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        bool noK = (state.currentWidth <= 0 || state.currentHeight <= 0);  // no valid K loaded
        if (colorButton("Save current K", colYellow(), false, noK, bw)) {
            if (actions.onSaveAsCustom) actions.onSaveAsCustom();
        }
        ImGui::SameLine();
        if (colorButton("Reload Custom", colDim(), false, !state.customAvailable, bw)) {
            if (actions.onSourceChanged) actions.onSourceChanged(IntrinsicsSource::Custom);
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        // ---- Calibration parameters ----
        ImGui::TextColored(ImVec4(0.9f,0.4f,0.6f,1), "Calibration parameters");

        char folderBuf[256];
        std::strncpy(folderBuf, state.chessboardFolder.c_str(), sizeof(folderBuf)-1);
        folderBuf[sizeof(folderBuf)-1] = 0;
        char fId[48]; snprintf(fId, sizeof(fId), "Chessboard folder##%s", suffix);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText(fId, folderBuf, sizeof(folderBuf))) {
            if (actions.onChessboardFolderChanged) actions.onChessboardFolderChanged(folderBuf);
        }

        int cols = state.chessboardBoardCols, rows = state.chessboardBoardRows;
        bool boardChanged = false;
        ImGui::SetNextItemWidth(110);
        char cId[32]; snprintf(cId, sizeof(cId), "cols##%s", suffix);
        boardChanged |= ImGui::InputInt(cId, &cols);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        char rId[32]; snprintf(rId, sizeof(rId), "rows##%s", suffix);
        boardChanged |= ImGui::InputInt(rId, &rows);
        if (boardChanged && actions.onBoardSizeChanged) {
            if (cols < 2) cols = 2;
            if (rows < 2) rows = 2;
            actions.onBoardSizeChanged(cols, rows);
        }

        float mm = state.chessboardSquareMm;
        ImGui::SetNextItemWidth(160);
        char sId[40]; snprintf(sId, sizeof(sId), "Square size (mm)##%s", suffix);
        if (ImGui::InputFloat(sId, &mm, 0.5f, 1.0f, "%.2f")) {
            if (mm < 0.1f) mm = 0.1f;
            if (actions.onSquareSizeChanged) actions.onSquareSizeChanged(mm);
        }

        ImGui::Spacing();
        if (colorButton("Run Calibration", ImVec4(0.9f,0.4f,0.6f,1))) {
            if (actions.onRunCalibration) actions.onRunCalibration();
        }
        if (state.calibDone) {
            ImGui::TextColored(colDim(), "Last: fx=%.1f fy=%.1f  RMS=%.3f (%d imgs)",
                               state.calibFx, state.calibFy, state.calibRms, state.calibImgCount);
        }
        if (!state.calibMessage.empty() && state.calibMessage != "OK") {
            ImGui::TextColored(ImVec4(1,0.4f,0.4f,1), "%s", state.calibMessage.c_str());
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        // ---- Saved calibrations (Step 6) ----
        ImGui::TextColored(colMuted(), "Saved calibrations: (Step 6)");
    }

    // [key-reorg Phase 12] Export section — replaces the removed M / Shift+M
    // keys. "Export Reg OBJs" = registered organ OBJs to registration_model/.
    // "Export cam-mm STL" = cam-mm tumor/liver STL (X+Z flip) + input snapshot.
    void drawExport() {
        if (!ImGui::CollapsingHeader("Export")) return;
        ImGui::Indent(16); ImGui::Spacing();
        float halfW = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        if (colorButton("Export Reg OBJs", colReg(), false, false, halfW)) {
            if (actions.onExportStl) actions.onExportStl();
        }
        ImGui::SameLine();
        if (colorButton("Export cam-mm STL", colYellow(), false, false, halfW)) {
            if (actions.onExportStlFlipped) actions.onExportStlFlipped();
        }
        ImGui::Spacing(); ImGui::Unindent(16); ImGui::Separator();
    }

    void drawSaveAR() {
        ImGui::Indent(16); ImGui::Spacing();
        float halfW = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        // Cam Init (left)
        {
            ImVec4 cc = {0.5f, 0.7f, 1.0f, 1};
            if(colorButton("Cam Init", cc, false, false, halfW)) {
                if(actions.onResetCamera) actions.onResetCamera();
            }
        }
        ImGui::SameLine();
        // Save AR Image (right)
        if(state.arSavedTimer > 0) { colorButton("Saved!", colYellow(), true, false, halfW); }
        else {
            if(colorButton("Save AR Image", colYellow(), false, false, halfW)) {
                if(actions.onSaveAR) actions.onSaveAR();
                state.arSavedTimer = 2.0f;
            }
        }
        ImGui::Spacing(); ImGui::Unindent(16); ImGui::Separator();
    }

    void drawVisibility() {
        ImGui::Spacing();
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(ImVec2(p.x, p.y), ImVec2(p.x + 4, p.y + ImGui::GetFontSize() + 10),
                              toU32(colVis(), 0.7f), 2.0f);
        }
        ImGui::Indent(16); ImGui::Spacing();
        ImGui::TextColored(colVis(), "VISIBILITY");
        ImGui::Spacing();
        float bw = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        float bh = 44.0f;
        float iconSz = 32.0f;
        for(int i = 0; i < 6; i++) {
            if(i % 2 != 0) ImGui::SameLine();
            auto& o = state.organs[i]; bool vis = o.alpha > 0.01f;
            ImGui::PushID(i);
            if(vis) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(o.color.x*0.14f,o.color.y*0.14f,o.color.z*0.14f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(o.color.x*0.22f,o.color.y*0.22f,o.color.z*0.22f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(o.color.x*0.30f,o.color.y*0.30f,o.color.z*0.30f,1));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.05f,0.055f,0.07f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.08f,0.085f,0.11f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.11f,0.12f,0.15f,1));
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            const char* aStr = o.alpha < 0.01f ? "OFF" : (o.alpha < 0.75f ? "50%" : "ON");
            ImVec2 btnPos = ImGui::GetCursorScreenPos();
            char lbl[64]; snprintf(lbl, sizeof(lbl), "       %s %s", o.name, aStr);
            if (o.alpha < 0.01f) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.42f, 0.48f, 1));
            } else {
                float op = o.alpha < 0.75f ? 0.55f : 1.0f;
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(o.color.x*op, o.color.y*op, o.color.z*op, 1));
            }
            bool clicked = ImGui::Button(lbl, ImVec2(bw, bh));
            ImGui::PopStyleColor();
            if (state.organIconTex[i] != 0) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                float iconY = btnPos.y + (bh - iconSz) * 0.5f;
                float iconAlpha = o.alpha < 0.01f ? 0.3f : 1.0f;
                dl->AddImage(
                    (ImTextureID)(intptr_t)state.organIconTex[i],
                    ImVec2(btnPos.x + 6, iconY),
                    ImVec2(btnPos.x + 6 + iconSz, iconY + iconSz),
                    ImVec2(0,0), ImVec2(1,1),
                    IM_COL32(255, 255, 255, (int)(iconAlpha * 255)));
            }
            if(clicked) { if(actions.onToggleOrgan) actions.onToggleOrgan(i); }
            ImGui::PopStyleVar(); ImGui::PopStyleColor(3); ImGui::PopID();
        }

        ImGui::Spacing();
        {
            float halfW = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
            float halfH = bh;
            bool bVis = state.boardAlpha > 0.01f;
            ImVec4 bc = {0.75f, 0.75f, 0.75f, 1};
            if(bVis) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(bc.x*0.14f,bc.y*0.14f,bc.z*0.14f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(bc.x*0.22f,bc.y*0.22f,bc.z*0.22f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(bc.x*0.30f,bc.y*0.30f,bc.z*0.30f,1));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.05f,0.055f,0.07f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.08f,0.085f,0.11f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.11f,0.12f,0.15f,1));
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            const char* bStr = state.boardAlpha < 0.01f ? "OFF" : (state.boardAlpha < 0.75f ? "50%" : "ON");
            char bLbl[64]; snprintf(bLbl, sizeof(bLbl), "Board %s", bStr);
            if (state.boardAlpha < 0.01f) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.42f, 0.48f, 1));
            } else {
                float bOp = state.boardAlpha < 0.75f ? 0.55f : 1.0f;
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(bc.x*bOp, bc.y*bOp, bc.z*bOp, 1));
            }
            bool bClicked = ImGui::Button(bLbl, ImVec2(halfW, halfH));
            ImGui::PopStyleColor();
            if(bClicked) { if(actions.onToggleOrgan) actions.onToggleOrgan(6); }
            ImGui::PopStyleVar(); ImGui::PopStyleColor(3);

            ImGui::SameLine();

            // Target mesh toggle (screenMesh)
            bool tVis = state.targetAlpha > 0.01f;
            ImVec4 tc = {0.3f, 0.6f, 0.9f, 1};
            if(tVis) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(tc.x*0.14f,tc.y*0.14f,tc.z*0.14f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(tc.x*0.22f,tc.y*0.22f,tc.z*0.22f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(tc.x*0.30f,tc.y*0.30f,tc.z*0.30f,1));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.05f,0.055f,0.07f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.08f,0.085f,0.11f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.11f,0.12f,0.15f,1));
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            const char* tStr = state.targetAlpha < 0.01f ? "OFF" : (state.targetAlpha < 0.75f ? "50%" : "ON");
            char tLbl[64]; snprintf(tLbl, sizeof(tLbl), "Target %s", tStr);
            if (state.targetAlpha < 0.01f) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.42f, 0.48f, 1));
            } else {
                float tOp = state.targetAlpha < 0.75f ? 0.55f : 1.0f;
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(tc.x*tOp, tc.y*tOp, tc.z*tOp, 1));
            }
            bool tClicked = ImGui::Button(tLbl, ImVec2(halfW, halfH));
            ImGui::PopStyleColor();
            if(tClicked) { if(actions.onToggleOrgan) actions.onToggleOrgan(7); }
            ImGui::PopStyleVar(); ImGui::PopStyleColor(3);
        }

        ImGui::Spacing(); ImGui::Unindent(16); ImGui::Separator();
    }

    void drawInfoPanel() {
        ImGui::Indent(16); ImGui::Spacing();
        if(ImGui::TreeNodeEx("Info", ImGuiTreeNodeFlags_NoTreePushOnOpen
                                          | (infoExpanded_ ? ImGuiTreeNodeFlags_DefaultOpen : 0))) {
            infoExpanded_ = true; ImGui::Spacing();
            ImGui::TextColored(colDim(), "Split Screen");
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
            ImGui::TextColored(state.splitScreen ? colGreen() : colMuted(), state.splitScreen ? "ON" : "OFF");
            ImGui::TextColored(colDim(), "Intrinsics");
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 60);
            const char* isrcName = "DA3";
            switch (state.intrinsicsSource) {
                case IntrinsicsSource::Custom: isrcName = "Custom"; break;
                case IntrinsicsSource::Calib:  isrcName = "Calib";  break;
                case IntrinsicsSource::Preset: isrcName = "Preset"; break;
                case IntrinsicsSource::DA3:    isrcName = "DA3";    break;
            }
            ImGui::TextColored(colDepth(), "%s", isrcName);
            ImGui::TextColored(colDim(), "Image Source");
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 50);
            if(state.hasLocalImage) ImGui::TextColored(colDepth(), "Local");
            else if(state.cameraState > 0) ImGui::TextColored(colGreen(), "Camera");
            else ImGui::TextColored(colMuted(), "Default");
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            if(state.useRegistration) {
                ImGui::Spacing();
                float diag = state.modelBBoxDiag > 0.0f ? state.modelBBoxDiag : 1.0f;
                ImGui::TextColored(colDim(), "Comp AvgErr");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
                ImGui::Text("%.4f (%.2f%%)", state.avgError, state.avgError / diag * 100.0f);
                ImGui::TextColored(colDim(), "Comp RMSE");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
                ImGui::Text("%.4f (%.2f%%)", state.rmse, state.rmse / diag * 100.0f);
                ImGui::TextColored(colDim(), "Comp MaxErr");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
                ImGui::Text("%.4f (%.2f%%)", state.maxError, state.maxError / diag * 100.0f);
                ImGui::TextColored(colDim(), "Scale");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 45);
                ImGui::Text("%.4f", state.scaleFactor);
                ImGui::TextColored(colDim(), "Model Size");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 45);
                ImGui::Text("%.2f", diag);
            }
        } else { infoExpanded_ = false; }
        ImGui::Unindent(16);
    }

    void drawUmeyamaOverlay(int windowWidth, int windowHeight) {
        ImGuiWindowFlags ov = ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize
                              |ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar
                              |ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoFocusOnAppearing|ImGuiWindowFlags_NoNav;
        const float sc = 2.0f;
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.5f,30),0,ImVec2(0.5f,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(30,16));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,12.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0,0,0,0.75f));
            ImGui::Begin("##ov_top",nullptr,ov);
            ImGui::SetWindowFontScale(sc);
            if(state.regState==1)
                ImGui::TextColored(colReg(),"Select board points on RIGHT screen  (%d / %d)",
                                   state.boardPtCount,state.targetPtCount);
            else if(state.regState==2)
                ImGui::TextColored(colReg(),"Select corresponding points on LEFT screen  (%d / %d)",
                                   state.objPtCount,state.targetPtCount);
            else if(state.regState==3)
                ImGui::TextColored(colGreen(),"All points selected!  Press Execute");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }
        {
            ImGuiWindowFlags lf = ov|ImGuiWindowFlags_NoInputs;
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.25f,windowHeight-50.0f),0,ImVec2(0.5f,1));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(16,8));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,8.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0,0,0,0.5f));
            ImGui::Begin("##lbl_L",nullptr,lf);
            ImGui::SetWindowFontScale(sc);
            ImGui::TextColored(colDim(),"LEFT: 3D Liver");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);

            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.75f,windowHeight-50.0f),0,ImVec2(0.5f,1));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(16,8));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,8.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0,0,0,0.5f));
            ImGui::Begin("##lbl_R",nullptr,lf);
            ImGui::SetWindowFontScale(sc);
            ImGui::TextColored(colDim(),"RIGHT: Texture Board");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.5f,windowHeight-100.0f),0,ImVec2(0.5f,1));
            ImGui::SetNextWindowSize(ImVec2(600,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(20,14));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,12.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0.05f,0.055f,0.07f,0.9f));
            ImGui::Begin("##ov_btm",nullptr,ov);
            ImGui::SetWindowFontScale(sc);
            drawColoredPointProgress("Board",state.boardPtCount,state.targetPtCount,
                                     state.regState==1, sc);
            ImGui::Spacing();
            drawColoredPointProgress("Object",state.objPtCount,state.targetPtCount,
                                     state.regState==2, sc);
            ImGui::Spacing();
            float bw3=(ImGui::GetContentRegionAvail().x-16)/3.0f;
            bool canUndo=(state.boardPtCount+state.objPtCount)>0;
            if(colorButton("Undo",colDim(),false,!canUndo,bw3)) {
                if(actions.onUndoUmeyamaPoint) actions.onUndoUmeyamaPoint();
            }
            ImGui::SameLine();
            if(colorButton("Execute",colGreen(),false,state.regState!=3,bw3)) {
                if(actions.onExecuteUmeyama) actions.onExecuteUmeyama();
            }
            ImGui::SameLine();
            if(colorButton("Cancel",colRed(),false,false,bw3)) {
                if(actions.onResetRegistration) actions.onResetRegistration();
            }
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }
    }

public:  // drawDepthOverlayをpublicに変更（マスク選択モードから呼ぶため）
    void drawDepthOverlay(int windowWidth, int windowHeight) {
        const float sc = 2.0f;

        // 上部の説明文のみ表示（シンプルに）
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.5f,8),ImGuiCond_Always,ImVec2(0.5f,0));
            ImGui::SetNextWindowSize(ImVec2(0,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(24*sc,10*sc));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,12.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0,0.05f,0.15f,0.85f));
            ImGui::Begin("##depthInstruction",nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
                             ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoInputs|ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::SetWindowFontScale(sc);
            // Hint colors track the active mask kind so the user always knows
            // what color their next click will produce.
            //   Liver       FG=green,  BG=red
            //   Instrument  FG=cyan,   BG=orange
            if (state.activeMaskKind == 1) {
                // Cyan / orange (matches MaskPicker draw colors).
                ImGui::TextColored(ImVec4(0.20f,0.85f,1.00f,1),
                                   "[Instrument]  L-click = FG (cyan)  "
                                   "R-click = BG (orange)");
            } else {
                ImGui::TextColored(colDepth(),
                                   "[Liver]  L-click = FG (green)  "
                                   "R-click = BG (red)");
            }
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }

        // LEFT/RIGHTラベルは削除（不要）
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.5f,windowHeight-10),ImGuiCond_Always,ImVec2(0.5f,1.0f));
            // Widened from 600 -> 720 to fit 4 buttons (Undo / Segment 1 /
            // Instrument / Run Depth) on one row without crowding.
            ImGui::SetNextWindowSize(ImVec2(720,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(20*sc,12*sc));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,16.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0.03f,0.04f,0.08f,0.92f));
            ImGui::Begin("##depthPanel",nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
                             ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::SetWindowFontScale(sc);
            // Liver point counts (existing). Always shown so the user sees
            // there's a primary mask under construction.
            ImGui::TextColored(colGreen(),"FG: %d",state.segFG);
            ImGui::SameLine(0,30);
            ImGui::TextColored(colRed(),"BG: %d",state.segBG);
            ImGui::SameLine(0,30);
            ImGui::TextColored(colDim(),"Total: %d",state.segFG+state.segBG);
            // Instrument counts (only when there are any, to keep the
            // common single-mask case visually quiet).
            if (state.instSegFG + state.instSegBG > 0) {
                ImGui::SameLine(0, 40);
                ImGui::TextColored(ImVec4(0.20f,0.85f,1.00f,1),
                                   "[Inst FG: %d", state.instSegFG);
                ImGui::SameLine(0, 12);
                ImGui::TextColored(ImVec4(1.00f,0.70f,0.20f,1),
                                   "BG: %d]", state.instSegBG);
            }
            ImGui::Spacing();

            // カメラがキャプチャされた状態、または画像がロードされた状態でボタン表示
            if (state.cameraState == 2 || (state.hasLocalImage && state.cameraState == 0)) {
                // Vignette auto-detection toggle. Affects what the NEXT
                // Instrument-preview or Run-Depth invocation writes into
                // instrument_segmentation_mask.png. The decision is baked
                // at mask-creation time, so toggling afterward has no
                // effect until the pipeline is re-run -- this is why the
                // checkbox lives in the DEPTH GENERATION section adjacent
                // to the buttons that trigger that pipeline.
                {
                    bool inclVig = state.detectVignette;
                    if (ImGui::Checkbox("Include vignette in occluder mask",
                                        &inclVig)) {
                        if (actions.onDetectVignetteChanged) {
                            actions.onDetectVignetteChanged(inclVig);
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("ON: 黒い視野 vignette を occluder mask に含める");
                    }
                    ImGui::Spacing();

                    // CUDA / GPU toggle. main.cpp 側で gApp.useCuda を更新し、
                    // 次回の Run Depth / Instrument preview の CLI に --cuda
                    // を付与する。sam2_da3_lite が USE_CUDA=OFF でビルド
                    // されている場合は CPU fallback されるため害はない。
                    {
                        bool useGpu = state.useCuda;
                        if (ImGui::Checkbox("Use CUDA (GPU)", &useGpu)) {
                            if (actions.onUseCudaChanged) {
                                actions.onUseCudaChanged(useGpu);
                            }
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("次回 Run Depth に --cuda 付与 (CPU fallback あり)");
                        }
                        ImGui::Spacing();
                    }
                }

                // Row 1: 4 buttons — Undo / Segment 1 / Instrument / Run Depth.
                // bw4 = (avail - 3*spacing) / 4. 16 ≈ 3 * default item spacing.
                float bw4 = (ImGui::GetContentRegionAvail().x - 24) / 4.0f;

                // Undo affects whichever mask is currently active. Enabled
                // when *that* list has any points; disabled otherwise.
                bool canUndoLiver      = (state.segFG    + state.segBG)    > 0;
                bool canUndoInstrument = (state.instSegFG+ state.instSegBG)> 0;
                bool canUndo = (state.activeMaskKind == 1)
                                   ? canUndoInstrument : canUndoLiver;

                // Both Segment buttons are always clickable. The handlers
                // (onSegment1 / onSegment2) decide whether to just activate
                // or to also preview, based on whether their own list has
                // an FG point yet. See the long comment in the button
                // rendering code below for why we don't gate on FG count
                // via colorButton's `disabled` parameter.

                bool isLiverActive = (state.activeMaskKind == 0);
                bool isInstActive  = (state.activeMaskKind == 1);

                if(colorButton("Undo",colDim(),false,!canUndo,bw4)) {
                    if(actions.onUndoSegPoint) actions.onUndoSegPoint();
                }
                ImGui::SameLine();
                // ---- Segment 1 (Liver) / Instrument toggle buttons ----
                //
                // Root cause of the "active button looks darker" bug we hit
                // earlier: the previous code passed `disabled = !hasFgPoints
                // && isActive` to colorButton, but colorButton renders
                // disabled buttons in nearly black (0.06,0.065,0.08) — which
                // visually overrides everything else. So clicking to activate
                // a kind that had no FG points yet *darkened* the button,
                // making active look like "off".
                //
                // The fix is to stop using disabled here entirely. Both
                // buttons are always clickable; the handlers (onSegment1 /
                // onSegment2) themselves decide between
                //   (a) activating only (when no FG point yet), or
                //   (b) activating + previewing (when FG point present).
                //
                // Visual feedback for which kind is active:
                //   - ">> " prefix on the active button's label
                //   - active button uses its native color (green / cyan),
                //     inactive is rendered in dim gray
                //   - colorButton's own active flag boosts the background
                //     tint a bit on top
                ImVec4 dimCol = colDim();
                ImVec4 liverActiveCol = colDepth();
                ImVec4 instActiveCol  = ImVec4(0.20f, 0.85f, 1.00f, 1.0f);
                const char* segLabel  = isLiverActive ? ">> Segment 1"  : "Segment 1";
                const char* instLabel = isInstActive  ? ">> Instrument" : "Instrument";
                ImVec4 segCol  = isLiverActive ? liverActiveCol : dimCol;
                ImVec4 instCol = isInstActive  ? instActiveCol  : dimCol;
                // disabled = false unconditionally — see comment above.
                if(colorButton(segLabel, segCol, isLiverActive, false, bw4)) {
                    if(actions.onSegment1) actions.onSegment1();
                }
                ImGui::SameLine();
                if(colorButton(instLabel, instCol, isInstActive, false, bw4)) {
                    if(actions.onSegment2) actions.onSegment2();
                }
                ImGui::SameLine();
                if(colorButton("Run Depth",colDepth(),false,false,bw4)) {
                    if(actions.onRunDepth) actions.onRunDepth();
                }
                // Row 2: Re-Capture (camera-captured mode only).
                if (state.cameraState == 2) {
                    ImGui::Spacing();
                    float bwFull = ImGui::GetContentRegionAvail().x - 16;
                    if(colorButton("Re-Capture",colYellow(),false,false,bwFull)) {
                        if(actions.onToggleCamera) actions.onToggleCamera();
                    }
                }
            }
            // カメラライブビュー状態（cameraState == 1）での戻るボタン
            else if (state.cameraState == 1) {
                float bw = ImGui::GetContentRegionAvail().x - 16;
                if(colorButton("< Back",colMuted(),false,false,bw)) {
                    if(actions.onCameraBack) actions.onCameraBack();
                }
            }
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }

        // -----------------------------------------------------------------
        // Segmentation preview popup (raised by runSegmentOnly via the
        // Segment 1 button). Renders the SAM2 overlay produced by the
        // external pipeline (--stage=segment). No mask editing, just a
        // confidence check before the user commits to depth inference.
        // -----------------------------------------------------------------
        if (state.segPreviewOpen) {
            ImGui::OpenPopup("Segmentation Preview");
            // Center on the next frame; cleared after BeginPopupModal so
            // that subsequent frames don't re-open after the user closes.
            ImGui::SetNextWindowPos(
                ImVec2(windowWidth * 0.5f, windowHeight * 0.5f),
                ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        }
        bool popupAlive = state.segPreviewOpen;
        if (ImGui::BeginPopupModal("Segmentation Preview",
                                   &popupAlive,
                                   ImGuiWindowFlags_AlwaysAutoResize))
        {
            if (state.segPreviewTexId != 0 &&
                state.segPreviewW > 0 && state.segPreviewH > 0)
            {
                // Cap the displayed image at 640 px wide while preserving
                // aspect. Bigger feels intrusive on 1280x720 calibrations.
                float dispW = 640.0f;
                float dispH = dispW * (float)state.segPreviewH
                              / (float)state.segPreviewW;
                ImGui::Image((ImTextureID)(intptr_t)state.segPreviewTexId,
                             ImVec2(dispW, dispH));
            } else {
                ImGui::Text("(no preview image loaded)");
            }
            ImGui::Separator();
            if (state.segPreviewScore > 0.0f) {
                ImGui::Text("SAM2 score: %.3f    FG pixels: %d",
                            state.segPreviewScore,
                            state.segPreviewFgPixels);
            } else {
                ImGui::TextDisabled("(stats unavailable)");
            }
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(120, 0))) {
                popupAlive = false;
            }
            ImGui::SameLine();
            ImGui::TextDisabled(
                "Add / remove points then click Segment 1 again to retry.");
            ImGui::EndPopup();
        }
        if (!popupAlive) {
            // The 'X' or Close button fired; let main.cpp release the
            // texture next frame if it wants to (state owns the id, but
            // we just signal closed here).
            state.segPreviewOpen = false;
        }
    }
};
