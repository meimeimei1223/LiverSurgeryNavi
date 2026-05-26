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
//    Viz  — General visualization  (Cluster, CorresPoints, AABB, screen mesh,
//           + the toggles/buttons migrated off the keyboard by the key-reorg
//           pass: formerly V/B/Shift+B/N/W/Shift+W/Shift+R/Y/H/Shift+H/Shift+T/
//           Shift+Y/Shift+I/F9/F10. See docs/KEY_REFERENCE.md.)
//
//  Toggle key: Ctrl+D (handled in main.cpp keyboard dispatch; plain D = AR save).
//
//  Lifecycle:
//    static DebugPanel::State g_debugPanel;            // in main.cpp
//    if (Ctrl+D pressed) g_debugPanel.showWindow ^= 1; // in key handler
//    // In ImGui frame, only during kRegistration && !gUmeyama.active:
//    DebugPanel::draw(g_debugPanel, gUI);
// =============================================================================

#include <functional>

#include "imgui.h"
#include "RegistrationImGuiManager.h"   // for RegUIState / RegUIActions
#include "SilOverlayDebug.h"            // for SilOverlay::g_silOverlay (F9)

// =============================================================================
// Phase 5 (W tab) — externs for silhouette-sweep RIM debug globals.
// These are `inline bool/float/int` at GLOBAL scope in RegistrationActions.h,
// which main.cpp includes AFTER this header, so the W tab needs extern decls to
// read/write them. (SilOverlay::g_silOverlay is namespace-scoped and provided
// by the SilOverlayDebug.h include above — NOT redeclared here.)
// =============================================================================
extern bool g_debugShow2DProjPopup_RawRim;
extern bool g_debugShow2DProjPopup_RawRimSmoothed;
extern bool g_debugShow2DProjPopup_RawRimOrdered;
extern bool g_debugShow2DProjPopup_Source;
extern bool g_debugShow2DProjPopup_Target;

extern float g_rawRimSmooth_GridPx;
extern int   g_rawRimSmooth_KnnK;
extern int   g_rawRimSmooth_KnnIters;
extern bool  g_rawRimSmooth_ShowRawOverlay;

extern float g_rawRimOrder_MaxEdgePx;
extern int   g_rawRimOrder_NPivots;
extern bool  g_rawRimOrder_ShowMST;
extern bool  g_rawRimOrder_ShowCleaned;

extern int g_silSwSrcRimMethod;          // 0=ENVELOPE, 1=MST_LONGEST_PATH

extern int  g_silhouetteSweepFrames1;
extern int  g_silhouetteSweepFrames2;
extern bool g_silhouetteSweepLog;
extern bool g_silhouetteSweepAnimate;

extern bool  g_silSwCheckA_Enable;
extern float g_silSwCheckA_RotCapDeg;
extern bool  g_silSwCheckB_Enable;
extern float g_silSwCheckB_CCToleranceDeg;

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

    // Optional per-tab body hooks set from main.cpp. When set, the hook is
    // rendered inside that tab instead of the local stub. Used for tabs whose
    // content depends on many main.cpp-local symbols (e.g. the migrated Ctrl+G
    // Quadrant Selector panel — Phase 3) and is therefore registered as a
    // lambda capturing frame-loop locals by reference.
    std::function<void()> drawGBody;    // Phase 3: full Ctrl+G panel body
    std::function<void()> drawNBody;    // Phase 4: Normal-Compatible Refine panel
    std::function<void()> drawVizExtra; // Phase 7: ScreenMesh Display + B/N viz
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
                if (st.drawGBody) st.drawGBody();   // Phase 3: migrated Ctrl+G panel
                else              drawTabG(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("O")) {
                st.activeTab = TAB_O;
                drawTabO(gUI.state, gUI.actions);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("N")) {
                st.activeTab = TAB_N;
                if (st.drawNBody) st.drawNBody();   // Phase 4: migrated Normal-Compat Refine
                else              drawTabN(gUI.state, gUI.actions);
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
                drawTabViz(gUI.state, gUI.actions);   // Phase 2: Cluster / CorresPoints
                if (st.drawVizExtra) st.drawVizExtra(); // Phase 7: ScreenMesh + B/N viz
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// ----- Phase 1 stubs (filled in later phases) -----
inline void drawTabG  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("G tab content is provided by main.cpp (drawGBody hook)."); }
inline void drawTabO(RegUIState& s, RegUIActions& a) {
    ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.3f, 1.0f),
                       "O — Shift+O — HemiAuto / QuadAuto");
    ImGui::TextWrapped(
        "Settings consumed by the next Shift+O run. Sidebar Advanced > Voxel "
        "slider mirrors the voxel value here (same global).");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Voxel size =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Voxel size (downsampling):");
    {
        float v = s.hemiVoxelSize;
        if (ImGui::SliderFloat("##o_voxel", &v, 0.1f, 2.0f, "%.2f")) {
            if (a.onHemiVoxelChanged) a.onHemiVoxelChanged(v);
        }
        ImGui::TextDisabled("  Recommended ratios:  1:1=%.2f  1:1.5=%.2f  1:2=%.2f",
                            s.idealVoxel1to1, s.idealVoxel1to15, s.idealVoxel1to2);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Instrument Px threshold =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Instrument-mask Px threshold:");
    {
        float v = s.instrumentPxThresh;
        if (ImGui::SliderFloat("##o_instpx", &v, 0.0f, 50.0f, "%.0f px")) {
            if (a.onInstrumentPxThreshChanged) a.onInstrumentPxThreshChanged(v);
        }
        ImGui::TextDisabled(
            "  Affects next Shift+O run. Use B key to see rejection points.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== AutoProbe iter cycles =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "AutoProbe iterations:");
    ImGui::SliderInt("##o_iter", &s.iterCycles, 1, 25, "K = %d");
    ImGui::Spacing();
    if (ImGui::Button("Run AutoProbe x K##o_runiter", ImVec2(-1, 24.0f))) {
        if (a.onIterativeAutoProbe) a.onIterativeAutoProbe(s.iterCycles);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Related viz shortcuts =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Related visualizations:");
    ImGui::TextDisabled("  Cluster (green/blue/yellow) - see Viz tab (J key).");
    ImGui::TextDisabled("  Boundary candidates / source viz - see Viz tab (B / N keys).");
}
inline void drawTabN  (RegUIState&, RegUIActions&) { ImGui::TextDisabled("N tab content is provided by main.cpp (drawNBody hook)."); }
inline void drawTabW(RegUIState& /*s*/, RegUIActions& /*a*/) {
    ImGui::TextColored(ImVec4(0.85f, 0.75f, 1.0f, 1.0f),
                       "W — Shift+W series — RIM 2D projection debug");
    ImGui::TextWrapped(
        "Each checkbox opens a separate ImGui popup window in the main frame. "
        "Keyboard shortcuts (Shift+W base + suffix) continue to work; these "
        "toggles are just a single place to find them.");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Section 1: Stage 0 — Raw/Smoothed/Ordered RIM popups =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Stage 0 — Source RIM discretization debug:");
    ImGui::Checkbox("CB0  Raw RIM 2D projection (points only)##w_raw",
                    &g_debugShow2DProjPopup_RawRim);
    ImGui::Checkbox("CB0.1  Smoothed RIM (grid + KNN)##w_sm",
                    &g_debugShow2DProjPopup_RawRimSmoothed);
    if (g_debugShow2DProjPopup_RawRimSmoothed) {
        ImGui::Indent(16);
        ImGui::SliderFloat("Grid px##w_sm_grid",
                           &g_rawRimSmooth_GridPx, 5.0f, 50.0f, "%.1f");
        ImGui::SliderInt("KNN K##w_sm_k", &g_rawRimSmooth_KnnK, 1, 20);
        ImGui::SliderInt("KNN iters##w_sm_iter", &g_rawRimSmooth_KnnIters, 1, 10);
        ImGui::Checkbox("Show raw overlay##w_sm_raw",
                        &g_rawRimSmooth_ShowRawOverlay);
        ImGui::Unindent(16);
    }

    ImGui::Checkbox("CB0.2  Ordered RIM (MST + longest path)##w_ord",
                    &g_debugShow2DProjPopup_RawRimOrdered);
    if (g_debugShow2DProjPopup_RawRimOrdered) {
        ImGui::Indent(16);
        ImGui::SliderFloat("MST edge max px##w_ord_edge",
                           &g_rawRimOrder_MaxEdgePx, 20.0f, 300.0f, "%.0f");
        ImGui::SliderInt("Pivots##w_ord_pivots", &g_rawRimOrder_NPivots, 5, 50);
        ImGui::Checkbox("Show MST overlay##w_ord_mst", &g_rawRimOrder_ShowMST);
        ImGui::Checkbox("Show cleaned overlay##w_ord_clean",
                        &g_rawRimOrder_ShowCleaned);
        ImGui::Unindent(16);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Section 2: Stage 3d — Source / Target rim popups =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Stage 3d — Source / Target rim popups:");
    ImGui::Checkbox("CB1  Source 2D projection (envelope + pivots)##w_src",
                    &g_debugShow2DProjPopup_Source);
    ImGui::Checkbox("CB2  Target lower-half + anchors##w_tgt",
                    &g_debugShow2DProjPopup_Target);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Section 3: Source RIM method =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Source RIM method (Ctrl+Alt+W sweep):");
    {
        int m = g_silSwSrcRimMethod;
        if (ImGui::RadioButton("MST longest path  (default, open arch)##w_method_mst",
                               m == 1)) g_silSwSrcRimMethod = 1;
        ImGui::SameLine();
        if (ImGui::RadioButton("Envelope (legacy)##w_method_env",
                               m == 0)) g_silSwSrcRimMethod = 0;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Section 4: Sweep safety checks =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                       "Sweep candidate filters:");
    ImGui::Checkbox("Check A: rotation cap##w_chka", &g_silSwCheckA_Enable);
    if (g_silSwCheckA_Enable) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("##w_chka_deg", &g_silSwCheckA_RotCapDeg,
                           5.0f, 90.0f, "+/-%.0f deg");
    }
    ImGui::Checkbox("Check B: CC orientation##w_chkb", &g_silSwCheckB_Enable);
    if (g_silSwCheckB_Enable) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("##w_chkb_deg", &g_silSwCheckB_CCToleranceDeg,
                           5.0f, 45.0f, "+/-%.0f deg");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Section 5: Sweep animation =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Sweep animation:");
    ImGui::Checkbox("Animate##w_anim", &g_silhouetteSweepAnimate);
    ImGui::SameLine();
    ImGui::Checkbox("Verbose log##w_log", &g_silhouetteSweepLog);
    ImGui::Spacing();
    ImGui::SliderInt("Phase 1 frames##w_f1", &g_silhouetteSweepFrames1, 1, 240);
    ImGui::SliderInt("Phase 2 frames##w_f2", &g_silhouetteSweepFrames2, 1, 120);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Section 6: F9 silhouette IoU button (also in G tab) =====
    ImGui::TextColored(ImVec4(0.85f, 1.0f, 0.7f, 1.0f),
                       "Silhouette IoU diagnostic:");
    if (ImGui::Button("F9: Toggle Silhouette IoU window##w_f9")) {
        SilOverlay::g_silOverlay.showWindow = !SilOverlay::g_silOverlay.showWindow;
    }
    ImGui::SameLine();
    ImGui::TextDisabled(SilOverlay::g_silOverlay.showWindow
                            ? "(currently open)"
                            : "(currently closed)");
}
inline void drawTabU(RegUIState& s, RegUIActions& /*a*/) {
    ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f),
                       "U — Umeyama Manual diagnostics");
    ImGui::TextWrapped(
        "Read-only view of current Umeyama session and last result. "
        "Operate via sidebar 'Umeyama Manual' button (manual mode opens "
        "a 2-screen overlay).");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Current session point counts =====
    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Current session:");
    {
        float boardFrac = (s.targetPtCount > 0)
            ? (float)s.boardPtCount / (float)s.targetPtCount : 0.0f;
        float objFrac   = (s.targetPtCount > 0)
            ? (float)s.objPtCount   / (float)s.targetPtCount : 0.0f;
        ImGui::Text("Board points:");
        ImGui::SameLine(160);
        ImGui::Text("%d / %d", s.boardPtCount, s.targetPtCount);
        ImGui::ProgressBar(boardFrac, ImVec2(-1, 6), "");
        ImGui::Spacing();
        ImGui::Text("Object points:");
        ImGui::SameLine(160);
        ImGui::Text("%d / %d", s.objPtCount, s.targetPtCount);
        ImGui::ProgressBar(objFrac, ImVec2(-1, 6), "");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ===== Last result =====
    if (s.useRegistration) {
        ImGui::TextColored(ImVec4(0.7f, 0.95f, 0.7f, 1.0f), "Last registration result:");
        const float diag = (s.modelBBoxDiag > 0.0f) ? s.modelBBoxDiag : 1.0f;
        ImGui::Text("  Avg error:");   ImGui::SameLine(180);
        ImGui::Text("%.4f  (%.2f%%)", s.avgError, s.avgError / diag * 100.0f);
        ImGui::Text("  RMSE:");        ImGui::SameLine(180);
        ImGui::Text("%.4f  (%.2f%%)", s.rmse, s.rmse / diag * 100.0f);
        ImGui::Text("  Max error:");   ImGui::SameLine(180);
        ImGui::Text("%.4f  (%.2f%%)", s.maxError, s.maxError / diag * 100.0f);
        ImGui::Text("  Scale factor:"); ImGui::SameLine(180);
        ImGui::Text("%.4f", s.scaleFactor);
        ImGui::Text("  Model size:");  ImGui::SameLine(180);
        ImGui::Text("%.4f", diag);
    } else {
        ImGui::TextDisabled("No registration applied yet.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Related visualization:");
    ImGui::TextDisabled("  CorresPoints (board+object spheres) - see Viz tab.");
}
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
