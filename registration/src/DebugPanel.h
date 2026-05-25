#pragma once

// =============================================================================
//  DebugPanel.h —  Consolidated tabbed debug/diagnostic window for Registration
//
//  Replaces (incrementally, over Phases 2-7) three always-on floating windows:
//    - "Ctrl+G Quadrant Selector"          -> tab G  (Phase 3)
//    - "Normal-Compatible Refine (Shift+N)" -> tab N  (Phase 4)
//    - "ScreenMesh Display"                 -> tab Viz (Phase 7)
//
//  Tabs organize controls by keyboard-shortcut family:
//    G    — Ctrl+G / Ctrl+Shift+G  (V3-R BIPOP-CMA-ES region-aware refine)
//    O    — Shift+O                (HemiAuto / QuadAuto, voxel, instrument)
//    N    — Shift+N / Ctrl+Shift+N (Normal-Compatible finishing refine)
//    W    — Shift+W series         (RIM 2D projection debug popups)
//    U    — Umeyama Manual         (point-pair list, error stats)
//    Viz  — General visualization  (Cluster, CorresPoints, AABB, screen mesh)
//
//  Toggle key: Ctrl+D (handled in main.cpp keyboard dispatch; plain D = AR save).
//
//  Lifecycle:
//    static DebugPanel::State g_debugPanel;            // in main.cpp
//    if (Ctrl+D pressed) g_debugPanel.showWindow ^= 1; // in key handler
//    // In ImGui frame, only during kRegistration && !gUmeyama.active:
//    DebugPanel::draw(g_debugPanel, gUI);
// =============================================================================

#include "imgui.h"
#include "RegistrationImGuiManager.h"   // for RegUIState / RegUIActions

namespace DebugPanel {

enum Tab {
    TAB_G = 0,   // Ctrl+G
    TAB_O,       // Shift+O
    TAB_N,       // Shift+N
    TAB_W,       // Shift+W
    TAB_U,       // Umeyama
    TAB_VIZ,     // Visualization
    TAB_COUNT
};

struct State {
    bool showWindow   = false;
    int  activeTab    = TAB_G;  // initial tab on first open
};

// Internal tab draw functions (later phases populate these).
inline void drawTabG  (RegUIState& s, RegUIActions& a); // Phase 3
inline void drawTabO  (RegUIState& s, RegUIActions& a); // Phase 6
inline void drawTabN  (RegUIState& s, RegUIActions& a); // Phase 4
inline void drawTabW  (RegUIState& s, RegUIActions& a); // Phase 5
inline void drawTabU  (RegUIState& s, RegUIActions& a); // Phase 6
inline void drawTabViz(RegUIState& s, RegUIActions& a); // Phase 2 + 7

inline void draw(State& st, RegistrationImGuiManager& gUI) {
    if (!st.showWindow) return;

    ImGui::SetNextWindowSize(ImVec2(560.0f, 480.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);

    if (ImGui::Begin("Debug Panel  [Ctrl+D]", &st.showWindow,
                     ImGuiWindowFlags_NoCollapse))
    {
        if (ImGui::BeginTabBar("##debugpanel_tabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("G")) {
                st.activeTab = TAB_G;
                drawTabG(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("O")) {
                st.activeTab = TAB_O;
                drawTabO(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("N")) {
                st.activeTab = TAB_N;
                drawTabN(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("W")) {
                st.activeTab = TAB_W;
                drawTabW(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("U")) {
                st.activeTab = TAB_U;
                drawTabU(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Viz")) {
                st.activeTab = TAB_VIZ;
                drawTabViz(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// ----- Phase 1 stubs (filled in later phases) -----
inline void drawTabG  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("G tab — Phase 3 will populate"); }
inline void drawTabO  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("O tab — Phase 6 will populate"); }
inline void drawTabN  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("N tab — Phase 4 will populate"); }
inline void drawTabW  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("W tab — Phase 5 will populate"); }
inline void drawTabU  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("U tab — Phase 6 will populate"); }
inline void drawTabViz(RegUIState& s, RegUIActions& a) {
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Visualization toggles  (keyboard shortcuts in parens)");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ----- Registration-result viz -----
    ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Registration cluster (key J):");
    {
        bool clusterVis = s.clusterVis;
        if (ImGui::Checkbox("Cluster markers  (green=src, blue=tgt-int, yellow=tgt-bnd)##viz_cluster",
                            &clusterVis)) {
            if (a.onToggleClusterVis) a.onToggleClusterVis();
        }
    }
    {
        bool corresVis = s.correspondenceVis;
        if (ImGui::Checkbox("Correspondence points  (Umeyama board+object)##viz_corres",
                            &corresVis)) {
            if (a.onToggleCorrespondenceVis) a.onToggleCorrespondenceVis();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextDisabled("More visualizations (B / N keys) — Phase 7 will surface.");
    ImGui::TextDisabled("Screen-mesh point density / debug AABB — Phase 7 will move here.");
}

} // namespace DebugPanel
