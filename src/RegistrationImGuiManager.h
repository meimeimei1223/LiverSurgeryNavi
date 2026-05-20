#pragma once

#include "imgui.h"
#include <string>
#include <functional>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include "PathConfig.h"

// Forward declarations for Shift+Ctrl+P tuning globals defined in RegistrationActions.h.
// (RegistrationActions.h is included after this header in main.cpp, so we need these
//  extern decls to compile the QCR Tuning slider panel below.)
extern int g_qcrSubsetK;
extern int g_qcrMaxTrials;

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
    std::function<void()> onRefine;
    std::function<void()> onSilhouetteAlign;
    std::function<void()> onPoseLibraryToggle;
    std::function<void()> onPoseUndo;
    std::function<void()> onAutoProbe;
    std::function<void(int)> onIterativeAutoProbe;  // (K) — calls runAutoProbe() K times
    std::function<void(int)> onSwitchDepthModel;
    std::function<void(int)> onInitRotPresetChanged;
    std::function<void(int)> onInitRotPositionChanged;   // Phase 2: 重心位置 selector (legacy, deprecated)
    std::function<void(int)> onIntrinsicsSourceChanged;
    std::function<void()>    onRunCalibration;
    std::function<void(float)> onInstrumentPxThreshChanged;   // ★追加
    // Vignette auto-detection toggle. Called when the checkbox in the
    // DEPTH GENERATION section is toggled. main.cpp side updates
    // gApp.detectVignette which is consulted when building the next
    // DepthRunner config (Instrument preview or Run Depth).
    std::function<void(bool)> onDetectVignetteChanged;
    // CUDA / GPU toggle. main.cpp 側で gApp.useCuda を更新し、次回の
    // Run Depth / Instrument preview の CLI に --cuda を付与する。
    // medsam2_da3_lite が USE_CUDA=OFF でビルドされている場合は CPU
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

    // Intrinsics source: 0=DA3, 1=Kinect, 2=Custom, 3=Calibrated
    int  intrinsicsSource = 1;
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
    float sidebarWidth_ = 400.0f;

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
    //  drawAnatomicalAxesStatus (Preview OBJ Anatomical Pose):
    //    Initial Orientation panel の先頭に表示する 3 軸 (AP / LR / CC) の
    //    確認 UI。レジ前に解剖軸の sign が正しいかを目視 + 数値で確認し、
    //    weak case (CC confidence < 5%) を赤バッジで強調する。
    //    Flip ボタンで g_lrFlipManual / g_ccFlipManual を即時トグルし、
    //    main.cpp 側で対応するラベルを再計算する。
    //
    //    レイアウト (1 行 = 20px):
    //      [name(40px)] [● status] [conf %] [note]            [Flip btn(70px)]
    // =====================================================================
    void drawAnatomicalAxesStatus() {
        const float rowH = 20.0f;
        const float rowSpacing = 3.0f;

        // ---- サブヘッダ + 赤/黄バッジ ----
        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 0.85f), "  ANATOMICAL AXES");
        ImGui::SameLine();
        if (state.ccAxisValid && state.ccWeak) {
            // 赤バッジ: CC confidence < 5% (Python の WEAK 閾値と同じ)
            ImGui::TextColored(ImVec4(0.96f, 0.32f, 0.32f, 1.0f),
                               "  [WEAK %.1f%% - inspect with Shift+H, Flip CC if reversed]",
                               state.ccConfidence * 100.0f);
        } else if (!state.apAxisValid || !state.lrAxisValid || !state.ccAxisValid) {
            // 黄バッジ: いずれかが未計算
            ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.28f, 1.0f),
                               "  [not all computed - press Apply Init Pose]");
        } else {
            // 緑チェック: 全 OK
            ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.45f, 1.0f),  "  [OK]");
        }

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

    void drawInitOrientationPanel() {
        // チャット 9: ORIENTATION 9 ボタン + POSITION (4-quadrant checkbox + Quick Presets
        // + Apply) の 2 セクション構成。POSITION 部は Ctrl+G と完全に連動。
        // [追加] 先頭に ANATOMICAL AXES (Preview OBJ Anatomical Pose) を配置:
        //        AP / LR / CC の状態 + confidence + Flip ボタンを表示し、weak
        //        case (CC conf < 5%) を赤バッジで前面警告。
        //   高さ内訳:
        //     header (4 + fontH + 4)
        //     ANATOMICAL AXES sub-header + badge (fontH + 2)
        //     3 行 (3 × (20 + 3))
        //     section separator spacing (6)
        //     POSITION sub-header (fontH + 2)
        //     anatomical orientation text (fontH)
        //     2x2 grid (~ 2 行 × 28px + table padding ≈ 64)
        //     mask string text (fontH)
        //     Quick Presets row (22 + 4)
        //     Apply button row (24 + 6)
        //     ORIENTATION sub-header (fontH + 2)
        //     Orientation 3 行 (3 × (22 + 4))
        //     末尾 padding 8
        const float fontH = ImGui::GetFontSize();
        float childH = 4.0f + fontH + 4.0f                        // header
                       + fontH + 2.0f                              // ANATOMICAL AXES sub
                       + 3.0f * (20.0f + 3.0f)                     // axes 3 行
                       + 6.0f                                      // section separator
                       + fontH + 2.0f                              // POSITION sub
                       + fontH                                     // anatomy hint
                       + 64.0f                                     // 2x2 grid
                       + fontH + 4.0f                              // mask text
                       + (22.0f + 4.0f)                            // quick presets
                       + (24.0f + 6.0f)                            // apply button
                       + fontH + 2.0f                              // ORIENTATION sub
                       + 3.0f * (22.0f + 4.0f)                     // orientation 3 行
                       + 8.0f;                                     // tail

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.04f, 0.06f, 0.10f, 0.5f));
        ImGui::BeginChild("##initOrient", ImVec2(-1, childH), false);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 1.0f), "  INITIAL ORIENTATION");
        ImGui::Spacing();

        // ---- ANATOMICAL AXES (Preview OBJ Anatomical Pose) ----
        drawAnatomicalAxesStatus();
        ImGui::Spacing();

        float totalW = ImGui::GetContentRegionAvail().x;
        float btnW   = (totalW - 8.0f) / 3.0f;
        float btnH   = 22.0f;

        ImVec4 colActive   = ImVec4(0.23f, 0.51f, 0.96f, 1.0f);
        ImVec4 colInactive = ImVec4(0.20f, 0.22f, 0.28f, 1.0f);
        ImVec4 colHover    = ImVec4(0.18f, 0.28f, 0.48f, 1.0f);

        // -----------------------------------------------------------------
        //  POSITION (チャット 9: 4-quadrant チェックボックス + Quick Presets + Apply)
        //  ----------------------------------------------------------------
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

        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 0.85f), "  POSITION");
        ImGui::TextColored(ImVec4(0.60f, 0.65f, 0.75f, 0.85f),
                           "  Top=anterior, Left=patient's right");

        // ---- 2x2 grid (Ctrl+G と同じレイアウト) ----
        {
            const ImGuiTableFlags tableFlags =
                ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame;
            if (ImGui::BeginTable("##initorient_quad_grid", 2, tableFlags)) {
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
                        state.activeQuadrantMask = newMask;
                        if (actions.onQuadrantMaskChanged)
                            actions.onQuadrantMaskChanged(newMask);
                    }
                    ImGui::PopID();
                };

                // Row 1: anterior
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("ant_R", state.quadNAR, kMaskAR);
                ImGui::TableNextColumn();
                checkbox_cell("ant_L", state.quadNAL, kMaskAL);

                // Row 2: posterior
                ImGui::TableNextRow();
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
                    state.activeQuadrantMask = presets[i].mask;
                    if (actions.onQuadrantMaskChanged)
                        actions.onQuadrantMaskChanged(presets[i].mask);
                }
                ImGui::PopStyleColor(4);
                ImGui::PopStyleVar();
                ImGui::PopID();
            }
        }

        // ---- Apply Init Pose ボタン (mask 確定後に明示的に姿勢を適用) ----
        ImGui::Spacing();
        {
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.10f, 0.30f, 0.55f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.15f, 0.40f, 0.70f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  ImVec4(0.20f, 0.50f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(0.95f, 0.97f, 1.00f, 1.0f));
            bool empty = (state.activeQuadrantMask == kMaskNone);
            if (empty) ImGui::BeginDisabled();
            if (ImGui::Button("Apply Init Pose", ImVec2(-1, 24.0f))) {
                if (actions.onApplyInitPose) actions.onApplyInitPose();
            }
            if (empty) ImGui::EndDisabled();
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar();
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 0.85f), "  ORIENTATION");

        struct PresetBtn { int id; const char* label; };
        // Radiology 慣例: 患者の右 (Right) を画面左、患者の左 (Left) を画面右に
        // 配置する。enum 番号は維持して動作は変えない (動作は getPresetRotation
        // 側で d_lr に基づいて決まる)。ボタンの位置とラベルの対応だけ反転。
        //
        // チャット 11 拡張: 外側 2 列 (Right+/Left+ ファミリ) を追加し、3x5 グリッドに。
        //   外側 = ±40° (dx=±2)、内側 = ±20° (dx=±1)、中央列 = 既存 (dx=0)。
        //   ラベルは末尾 "+" で強めを明示。
        PresetBtn grid[3][5] = {
                                { { 9,"Up-R+"}, {2,"Up-R"},  {1,"Up"},   {8,"Up-L"}, {12,"Up-L+"} },
                                { {10,"Right+"},{3,"Right"}, {0,"Base"}, {7,"Left"}, {13,"Left+"} },
                                { {11,"Dn-R+"}, {4,"Dn-R"},  {5,"Down"}, {6,"Dn-L"}, {14,"Dn-L+"} },
                                };

        // 5 列用ローカル width (POSITION 2x2 と Apply は既存の btnW/qpW を継続)。
        // 4 個の spacing (4px each) 分を totalW から控除して 5 等分。
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

        drawInitOrientationPanel();

        // --- STAGE 1: Hemi Auto (2/3 width) + AutoProbe (1/3 width) ---
        {
            const float gap     = 4.0f;
            const float availW  = ImGui::GetContentRegionAvail().x;
            const float hemiW   = (availW - gap) * 0.66f;

            if(glowButton("Hemi Auto", colReg(), anyP && state.regMethod!=1, hemiW, 52, state.btnIconTex[RegUIState::ICON_HEMI_AUTO])) {
                state.regMethod = 1; if(actions.onHemiAuto) actions.onHemiAuto();
            }
            {
                ImVec2 p = ImGui::GetItemRectMin();
                ImVec2 sz = ImGui::GetItemRectSize();
                char buf[16]; snprintf(buf, sizeof(buf), "v:%.2f", state.hemiVoxelSize);
                ImVec2 ts = ImGui::CalcTextSize(buf);
                ImGui::GetWindowDrawList()->AddText(
                    ImVec2(p.x + sz.x - ts.x - 6, p.y + (sz.y - ts.y) * 0.5f),
                    IM_COL32(120,220,160,180), buf);
            }

            // AutoProbe: HemiAuto の右隣 (残り 1/3)
            ImGui::SameLine(0.0f, gap);
            if(glowButton("AutoProbe", colReg(), anyP, -1, 52, 0)) {
                if(actions.onAutoProbe) actions.onAutoProbe();
            }
        }
        ImGui::Spacing();
        if (state.regMethod == 1) {
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
            ImGui::Spacing();
        }

        // --- Instrument Px Threshold ---
        // 器具マスクが有効な時だけ表示。スライダー値が変わったら次の HemiAuto
        // (Key O) で再分類されるので、Bキーで赤点を見ながら調整できる。
        if (state.regMethod == 1 && state.instrumentMaskActive) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.04f, 0.04f, 0.4f));
            ImGui::BeginChild("##instinfo", ImVec2(-1, 42), false);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
            ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.45f, 1.0f), "  Inst Px");
            ImGui::SameLine();
            float v = state.instrumentPxThresh;
            ImGui::SetNextItemWidth(-8);
            if (ImGui::SliderFloat("##instpx", &v, 0.0f, 50.0f, "%.0f px")) {
                if (actions.onInstrumentPxThreshChanged) {
                    actions.onInstrumentPxThreshChanged(v);
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::Spacing();
        }

        // --- Iter Probe (K回 AutoProbe を連続呼び出し) ---
        {
            const float gap   = 4.0f;
            const float availW = ImGui::GetContentRegionAvail().x;
            const float halfW  = (availW - gap) * 0.66f;
            const float ctrlW  = availW - halfW - gap;

            char iterLabel[48];
            snprintf(iterLabel, sizeof(iterLabel), "Iter Probe x%d", state.iterCycles);
            if(glowButton(iterLabel, colReg(), anyP, halfW, 52, 0)) {
                if(actions.onIterativeAutoProbe)
                    actions.onIterativeAutoProbe(state.iterCycles);
            }
            ImGui::SameLine(0.0f, gap);
            ImGui::PushItemWidth(ctrlW);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 14);
            ImGui::DragInt("##iterK", &state.iterCycles, 0.2f, 1, 64, "K=%d");
            ImGui::PopItemWidth();
        }
        ImGui::Spacing();

        // --- STAGE 2: BIPOP-CMA-ES ---
        {
            bool bipopDisabled = !state.useRegistration;
            if(glowButton("BIPOP-CMA-ES  [Shift+V]", colReg(), bipopDisabled, -1, 52)) {
                state.regMethod = 0; if(actions.onBipopCmaes) actions.onBipopCmaes();
            }
        }
        ImGui::Spacing();

        // --- STAGE 3: Silhouette Alignment ---
        {
            bool silhDisabled = !state.useRegistration;
            if(glowButton("Silhouette Alignment  [Shift+E]", colReg(), silhDisabled, -1, 52)) {
                if(actions.onSilhouetteAlign) actions.onSilhouetteAlign();
            }
        }
        ImGui::Spacing();

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
        {
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            if(colorButton(state.poseLibraryOpen ? "Pose Library ON" : "Pose Library",
                            state.poseLibraryOpen ? colGreen() : colReg(), false, false, bw2)) {
                if(actions.onPoseLibraryToggle) actions.onPoseLibraryToggle();
            }
            ImGui::SameLine();
            if(colorButton("Undo", state.poseUndoAvailable ? colRed() : colDim(),
                            false, !state.poseUndoAvailable, bw2)) {
                if(actions.onPoseUndo) actions.onPoseUndo();
            }
        }

        ImGui::Spacing();
        {
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            if(colorButton("Reset Reg", colRed(), false, false, bw2)) {
                if(actions.onResetRegistration) actions.onResetRegistration();
                if(state.clusterVis && actions.onToggleClusterVis) actions.onToggleClusterVis();
            }
            ImGui::SameLine();
            bool hasPoints = (state.boardPtCount + state.objPtCount) > 0;
            if(colorButton("Clear CorresPoints", hasPoints ? colRed() : colDim(), false, !hasPoints, bw2)) {
                if(actions.onClearPoints) actions.onClearPoints();
            }
        }
        ImGui::Spacing();
        {
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            if(colorButton(state.correspondenceVis ? "CorresPoints ON" : "CorresPoints OFF",
                            state.correspondenceVis ? colGreen() : colDim(), false, false, bw2)) {
                if(actions.onToggleCorrespondenceVis) actions.onToggleCorrespondenceVis();
            }
            ImGui::SameLine();
            if(colorButton(state.clusterVis ? "Cluster ON" : "Cluster OFF",
                            state.clusterVis ? colGreen() : colDim(), false, false, bw2)) {
                if(actions.onToggleClusterVis) actions.onToggleClusterVis();
            }
        }

        // ---- Shift+Ctrl+P (QuadCyclic-RANSAC) tuning ----
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Shift+Ctrl+P Tuning")) {
            ImGui::Indent(8);
            // K subset size: 3 (Fischler-Bolles min), 4-5 (more stable, over-determined)
            ImGui::TextColored(colMuted(), "Subset size K:");
            if (ImGui::SliderInt("##qcrK", &g_qcrSubsetK, 3, 5, "K = %d")) {
                // value clamped inside runQuadCyclicRansac if out of range
            }
            ImGui::TextColored(ImVec4(0.45f,0.45f,0.5f,1),
                               "  K=3: %d-pt exact fit (max variety)\n"
                               "  K=4: over-det. (balanced)\n"
                               "  K=5: over-det. (most stable)",
                               g_qcrSubsetK);
            // Trial count cap (K=4/5 expensive: stride sample to stay under cap)
            ImGui::Spacing();
            ImGui::TextColored(colMuted(), "Max trials (Stage 1 cap):");
            ImGui::SliderInt("##qcrCap", &g_qcrMaxTrials, 10000, 500000, "%d");
            ImGui::Unindent(8);
        }
        ImGui::Spacing();

        ImGui::Spacing(); ImGui::Spacing();
        {
            float bw2 = (ImGui::GetContentRegionAvail().x - 4) / 2.0f;
            bool canDeform = (state.regState == 4 && state.mainMode == 0);
            if(canDeform) {
                if(glowButton("Proceed Deform >>", colDeform(), false, bw2, 36)) {
                    if (actions.onSwitchToDeformMode) actions.onSwitchToDeformMode();
                }
            } else {
                colorButton("Proceed Deform >>", colDim(), false, true, bw2, 36);
            }
            ImGui::SameLine();
            if(colorButton("<< Back Depth", colDepth(), false, false, bw2, 36)) {
                regPhaseActive_ = false;
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

    // ---- Intrinsics Source selector (replaces Depth Scale) ----
    void drawIntrinsicsSource(const char* suffix = "") {
        ImGui::Spacing();
        ImGui::TextColored(colDim(), "Intrinsics Source");
        ImGui::Spacing();

        const char* labels[] = {"DA3", "Kinect", "Custom", "Calib"};
        ImVec4 colors[] = {
            {0.3f,0.8f,0.6f,1}, {0.2f,0.6f,1.0f,1},
            {0.9f,0.7f,0.2f,1}, {0.9f,0.4f,0.6f,1}
        };
        float bw = (ImGui::GetContentRegionAvail().x - 18) / 4.0f;

        for (int i = 0; i < 4; i++) {
            if (i > 0) ImGui::SameLine();
            bool sel = (state.intrinsicsSource == i);
            ImVec4 c = colors[i];
            if (sel) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(c.x*0.25f,c.y*0.25f,c.z*0.25f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(c.x*0.35f,c.y*0.35f,c.z*0.35f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(c.x*0.45f,c.y*0.45f,c.z*0.45f,1));
                ImGui::PushStyleColor(ImGuiCol_Text,           c);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.06f,0.065f,0.08f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.10f,0.11f,0.14f,1));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(0.14f,0.15f,0.18f,1));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.47f,0.52f,1));
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
            char id[64]; snprintf(id, sizeof(id), "%s##isrc%d%s", labels[i], i, suffix);
            if (ImGui::Button(id, ImVec2(bw, 30))) {
                if (actions.onIntrinsicsSourceChanged) actions.onIntrinsicsSourceChanged(i);
            }
            ImGui::PopStyleVar(); ImGui::PopStyleColor(4);
        }

        // Show calibration sub-panel when Calib is selected
        if (state.intrinsicsSource == 3) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f,0.06f,0.10f,1));
            ImGui::BeginChild("##calibPanel", ImVec2(0, state.calibDone ? 110 : 60), true);
            ImGui::TextColored(ImVec4(0.9f,0.4f,0.6f,1), "Calibration");
            ImGui::SameLine();
            ImGui::TextColored(colDim(), "(%s)", state.calibFolder.c_str());

            if (!state.calibDone) {
                ImGui::Spacing();
                if (colorButton("Run Calibration##calib", ImVec4(0.9f,0.4f,0.6f,1))) {
                    if (actions.onRunCalibration) actions.onRunCalibration();
                }
            } else {
                ImGui::TextColored(colDim(),
                                   "fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f",
                                   state.calibFx, state.calibFy, state.calibCx, state.calibCy);
                ImGui::TextColored(colDim(),
                                   "RMS=%.3f px  (%d images)", state.calibRms, state.calibImgCount);
                ImGui::Spacing();
                if (colorButton("Re-run##recalib", ImVec4(0.9f,0.4f,0.6f,1))) {
                    if (actions.onRunCalibration) actions.onRunCalibration();
                }
            }
            if (!state.calibMessage.empty() && state.calibMessage != "OK") {
                ImGui::TextColored(ImVec4(1,0.4f,0.4f,1), "%s", state.calibMessage.c_str());
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();
        }
        ImGui::Spacing();
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
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 50);
            const char* isrcNames[] = {"DA3", "Kinect", "Custom", "Calib"};
            int si = std::clamp(state.intrinsicsSource, 0, 3);
            ImGui::TextColored(colDepth(), "%s", isrcNames[si]);
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
                        ImGui::SetTooltip(
                            "When ON (default), the depth pipeline auto-detects\n"
                            "the black FOV vignette and OR-merges it into\n"
                            "instrument_segmentation_mask.png.\n"
                            "\n"
                            "When OFF, the saved occluder mask contains only the\n"
                            "SAM2 instrument result (or no file at all if no\n"
                            "instrument prompts were given). Equivalent to\n"
                            "passing --no-vignette-detect to the external\n"
                            "pipeline.\n"
                            "\n"
                            "The choice is baked at mask creation; press Run\n"
                            "Depth (or the Instrument preview) to apply.");
                    }
                    ImGui::Spacing();

                    // CUDA / GPU toggle. main.cpp 側で gApp.useCuda を更新し、
                    // 次回の Run Depth / Instrument preview の CLI に --cuda
                    // を付与する。medsam2_da3_lite が USE_CUDA=OFF でビルド
                    // されている場合は CPU fallback されるため害はない。
                    {
                        bool useGpu = state.useCuda;
                        if (ImGui::Checkbox("Use CUDA (GPU)", &useGpu)) {
                            if (actions.onUseCudaChanged) {
                                actions.onUseCudaChanged(useGpu);
                            }
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "When ON, the next Run Depth / Instrument preview\n"
                                "adds --cuda to the CLI so medsam2_da3_lite uses\n"
                                "CUDAExecutionProvider.\n"
                                "\n"
                                "Harmless when the pipeline was built with\n"
                                "USE_CUDA=OFF: it prints a warning and falls back\n"
                                "to CPU.\n"
                                "\n"
                                "Applies to NEXT Run Depth, not the current frame.");
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
