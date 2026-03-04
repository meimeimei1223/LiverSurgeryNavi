#pragma once

#include "imgui.h"
#include <string>
#include <functional>
#include <vector>
#include <cstdio>
#include <cmath>
#include "PathConfig.h"

struct RegUIActions {
    std::function<void()> onToggleCamera;
    std::function<void()> onRunDepth;
    std::function<void()> onResetDefaultImage;
    std::function<void()> onLoadLocalImage;
    std::function<void()> onUndoSegPoint;
    std::function<void(float)> onDepthScaleChanged;
    std::function<void()> onFullAuto;
    std::function<void()> onHemiAuto;
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
    std::function<void()> onStartFromDepth;
    std::function<void()> onSaveAR;
    std::function<void(int)> onToggleOrgan;
    std::function<void()> onSwitchToDeformMode;
    std::function<void()> onResetCamera;
    std::function<void()> onRefine;
    std::function<void()> onPoseLibraryToggle;
    std::function<void()> onPoseUndo;
    std::function<void(int)> onSwitchDepthModel;
};

struct RegUIState {
    int mainMode = 0;
    int cameraState = 0;
    bool depthRunning = false;
    bool depthDone = false;
    float depthScale = 0.3f;
    int segFG = 0, segBG = 0;
    bool hasLocalImage = false;
    std::string localImageName;

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
    float boardAlpha = 0.7f;
    unsigned int boardIconTex = 0;
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
        if (state.depthSplitScreen) { drawDepthOverlay(windowWidth, windowHeight); return; }

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
                ImGui::Spacing();
                ImGui::TextColored(colDim(), "Depth Scale");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 35);
                ImGui::Text("%.2f", state.depthScale);
                ImGui::PushItemWidth(-16);
                ImGui::PushStyleColor(ImGuiCol_SliderGrab, colDepth());
                ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, colDepth());
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f,0.1f,0.13f,1));
                float ns = state.depthScale;
                if(ImGui::SliderFloat("##ds_reg", &ns, 0.0f, 2.0f, "")) {
                    if(actions.onDepthScaleChanged) actions.onDepthScaleChanged(ns);
                }
                ImGui::PopStyleColor(3);
                ImGui::PopItemWidth();
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
            ImGui::Spacing();
            ImGui::TextColored(colDim(), "Depth Scale");
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 35);
            ImGui::Text("%.2f", state.depthScale);
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, colDepth());
            ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, colDepth());
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f,0.1f,0.13f,1));
            float ns = state.depthScale;
            if(ImGui::SliderFloat("##ds", &ns, 0.0f, 2.0f, "")) {
                if(actions.onDepthScaleChanged) actions.onDepthScaleChanged(ns);
            }
            ImGui::PopStyleColor(3);

            ImGui::Spacing();
            if(colorButton("Reset to Default Image", colDim())) {
                if(actions.onResetDefaultImage) actions.onResetDefaultImage();
            }
        }

        ImGui::PopItemWidth(); ImGui::Unindent(16);
        ImGui::Spacing(); ImGui::Separator();
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

        if(methodButton("Full Auto", "", state.regMethod==0, state.regState, anyP && state.regMethod!=0, state.btnIconTex[RegUIState::ICON_FULL_AUTO])) {
            state.regMethod = 0; if(actions.onFullAuto) actions.onFullAuto();
        }
        ImGui::Spacing();
        {
            float totalW = ImGui::GetContentRegionAvail().x;
            float hemiW = (totalW - 4) * 0.64f;
            float refW  = (totalW - 4) * 0.36f;
            if(methodButton("Hemi Auto", "", state.regMethod==1, state.regState, anyP && state.regMethod!=1, state.btnIconTex[RegUIState::ICON_HEMI_AUTO], hemiW)) {
                state.regMethod = 1; if(actions.onHemiAuto) actions.onHemiAuto();
            }
            ImGui::SameLine();
            bool isRefining = (state.regState == 5);
            bool refineClickable = state.refineEnabled || isRefining;
            const char* refLabel = isRefining ? "Stop" : "Refine";
            ImVec4 refCol = isRefining ? colRed() : colReg();
            if(colorButton(refLabel, refineClickable ? refCol : colDim(), false, !refineClickable, refW, 36)) {
                if(actions.onRefine) actions.onRefine();
            }
        }
        ImGui::Spacing();
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
            drawColoredPointProgress("Board", state.boardPtCount, state.targetPtCount,
                                     state.regState == 1);
            ImGui::Spacing();
            drawColoredPointProgress("Object", state.objPtCount, state.targetPtCount,
                                     state.regState == 2);
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

    void drawSaveAR() {
        ImGui::Indent(16); ImGui::Spacing();
        if(state.arSavedTimer > 0) { colorButton("Saved!", colYellow(), true); }
        else {
            if(colorButton("Save AR Image", colYellow())) {
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
                    IM_COL32(255, 255, 255, (int)(iconAlpha * 255))
                    );
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

            ImVec4 cc = {0.5f, 0.7f, 1.0f, 1};
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(cc.x*0.14f,cc.y*0.14f,cc.z*0.14f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(cc.x*0.22f,cc.y*0.22f,cc.z*0.22f,1));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   ImVec4(cc.x*0.30f,cc.y*0.30f,cc.z*0.30f,1));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(cc.x, cc.y, cc.z, 0.85f));
            if (ImGui::Button("Cam Init", ImVec2(halfW, halfH))) {
                if(actions.onResetCamera) actions.onResetCamera();
            }
            ImGui::PopStyleColor();
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
            ImGui::TextColored(colDim(), "Depth Scale");
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 35);
            ImGui::Text("%.2f", state.depthScale);
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

    void drawDepthOverlay(int windowWidth, int windowHeight) {
        const float sc = 2.0f;
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
            ImGui::TextColored(colDepth(),"RIGHT screen: L-click = FG (green)  R-click = BG (red)");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.25f,windowHeight-50*sc),ImGuiCond_Always,ImVec2(0.5f,0.5f));
            ImGui::SetNextWindowSize(ImVec2(0,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(16,6));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,8.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0,0,0,0.5f));
            ImGui::Begin("##depthLeftLabel",nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
                             ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoInputs|ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::SetWindowFontScale(sc*0.8f);
            ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1),"LEFT: 3D Model (reference)");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);

            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.75f,windowHeight-50*sc),ImGuiCond_Always,ImVec2(0.5f,0.5f));
            ImGui::SetNextWindowSize(ImVec2(0,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(16,6));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,8.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0,0,0,0.5f));
            ImGui::Begin("##depthRightLabel",nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
                             ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoInputs|ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::SetWindowFontScale(sc*0.8f);
            ImGui::TextColored(colDepth(),"RIGHT: Texture Board (click here)");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidth*0.5f,windowHeight-10),ImGuiCond_Always,ImVec2(0.5f,1.0f));
            ImGui::SetNextWindowSize(ImVec2(600,0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(20*sc,12*sc));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,16.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg,ImVec4(0.03f,0.04f,0.08f,0.92f));
            ImGui::Begin("##depthPanel",nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
                             ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::SetWindowFontScale(sc);
            ImGui::TextColored(colGreen(),"FG: %d",state.segFG);
            ImGui::SameLine(0,30);
            ImGui::TextColored(colRed(),"BG: %d",state.segBG);
            ImGui::SameLine(0,30);
            ImGui::TextColored(colDim(),"Total: %d",state.segFG+state.segBG);
            ImGui::Spacing();
            float bw3=(ImGui::GetContentRegionAvail().x-16)/3.0f;
            bool canUndo=(state.segFG+state.segBG)>0;
            if(colorButton("Undo",colDim(),false,!canUndo,bw3)) {
                if(actions.onUndoSegPoint) actions.onUndoSegPoint();
            }
            ImGui::SameLine();
            if(colorButton("Run Depth",colDepth(),false,false,bw3)) {
                if(actions.onRunDepth) actions.onRunDepth();
            }
            ImGui::SameLine();
            if(colorButton("Re-Capture",colYellow(),false,false,bw3)) {
                if(actions.onToggleCamera) actions.onToggleCamera();
            }
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar(2);
        }
    }
};
