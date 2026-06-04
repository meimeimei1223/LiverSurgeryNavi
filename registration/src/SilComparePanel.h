#ifndef SIL_COMPARE_PANEL_H
#define SIL_COMPARE_PANEL_H
// =====================================================================
// SilComparePanel.h
//   Ctrl+D > G tab subsection: run up to 3 silhouette-alignment methods
//   from the SAME start pose, score each with the SAME 2D-IoU yardstick,
//   and capture each result into a SilOverlay (F9) slot so the existing
//   F9 window can flip between the silhouette composites for A/B/C
//   visual comparison.
//
//   NOTHING new is optimised here. All three methods are existing
//   project entry points run with different toggle states:
//
//     [1] Legacy Alt+P  : runShiftE()                       (V1 full-mesh
//                          raster IoU; the slow baseline we're profiling)
//     [2] IoU Ctrl+G    : runBipopCmaesV3RS(mask) with rim-sil OFF and a
//                          dominant lambda_sil  -> area-IoU folded onto the
//                          V3R BIPOP machinery (cached target, step16,
//                          parallel projection)
//     [3] Rim chamfer   : runBipopCmaesV3RS(mask) with the ANATOMIC rim-sil
//                          penalty ON -> boundary-distance objective (the
//                          rim->g_boundaryDistMap lookup we discussed; the
//                          plumbing already exists in rasterize_iou2d_v3rs)
//
//     scoring : SilOverlay::capture()  (rasterize_iou2d_v3rs, identical step)
//     viewer  : SilOverlay F9 window   -> Run 1 = method 1, Run 2 = method 2,
//                                         Run 3 = method 3
//
//   NON-DESTRUCTIVE. Snapshots every organ before the first method,
//   restores that snapshot before each method, and restores it again at
//   the end (unless "keep best" is ticked), so comparing never moves the
//   patient and never writes the Pose Library (the dispatch-site
//   poseAutoSaveBeforeRegistration / poseSaveToLibrary wrappers are NOT
//   invoked from here).
//
//   SLOT-CLOBBER NOTE. runBipopCmaesV3RS writes g_silOverlay Run slots
//   during its own run. To avoid clobbering, we RUN all checked methods
//   first (stashing each result mesh), then CAPTURE all results in one
//   burst with no optimiser call in between. The final 3 captures win,
//   so no edit to SilOverlayDebug.h is needed.
//
//   INCLUDE ORDER. This header calls inline functions / globals defined in
//   RegistrationActions.h and SilOverlayDebug.h (which itself pulls in
//   CmaesRefineV3RS.h). #include it in main.cpp AFTER both, e.g. right
//   below the RegistrationActions.h include. It deliberately does NOT
//   re-include those big headers to avoid ordering / circular issues —
//   it relies on them already being visible in the translation unit,
//   exactly like main.cpp's drawGBody / drawWExtra lambdas do.
//
//   INTEGRATION (2 lines in main.cpp):
//     1) #include "SilComparePanel.h"     // after RegistrationActions.h
//     2) inside the drawGBody lambda (near the Alt+P section):
//            SilCompare::drawSection();
// =====================================================================

#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <iostream>   // [ROBUSTNESS] std::cout/cerr/endl used in run(); do not
                      // rely on a prior header having pulled it transitively.

#include <glm/glm.hpp>
#include "imgui.h"

// [COMPILE-CONTEXT GUARD] This header is designed to be #included in main.cpp
// AFTER RegistrationActions.h + SilOverlayDebug.h (see the header comment
// above). It deliberately does NOT re-include those big headers. If a tool or
// build rule compiles this file in isolation (its own translation unit), none
// of liverMesh3D / g_ctrlgs* / SilOverlay / runShiftE are visible and every
// reference is an "undeclared identifier" error. To make that harmless, the
// entire body is gated on the sentinel RegistrationActions.h defines. In a
// standalone TU the sentinel is absent => this file expands to nothing (0
// errors); in main.cpp's TU it is present => full functionality, byte-for-byte
// as before. Nothing about the main.cpp build path changes.
#ifdef REGISTRATION_ACTIONS_H_INCLUDED

namespace SilCompare {

// ----- Panel state (file-scope global, mirrors g_shiftE_* / g_ctrlgs*) -----
struct State {
    // Which methods to run when "Run checked" is pressed.
    bool run_m1 = true;    // [1] Legacy Alt+P  (runShiftE)
    bool run_m2 = false;   // [2] IoU Ctrl+G    (V3RS, rim-sil OFF, RMSE-blended)
    bool run_m3 = false;   // [3] Rim chamfer   (V3RS, anatomic rim-sil ON)
    bool run_m4 = true;    // [4] Pure squash-IoU (V3RS, pure_iou_mode = Ctrl+I)

    // Scoring rasteriser stride (the yardstick). 16 matches the V3RS
    // hot-path default so method [2]/[3]/[4]'s internal score lines up
    // with this post-hoc score. Lower = finer / slower.
    int  score_step = 16;

    // lambda_sil to force on method [2] so IoU actually dominates the
    // V3R chamfer term (the stock default 0.2 is too weak for a pure-IoU
    // comparison). Only applied for the duration of the [2] run.
    float m2_lambda_sil = 4.0f;

    // After comparing, leave the mesh at the best-IoU method's pose
    // instead of restoring the original start pose. OFF = fully
    // non-destructive (recommended; commit the winner via its own hotkey).
    bool keep_best = false;

    // ----- Results (filled by run()) -----
    struct Res { bool valid = false; float iou = 0.0f; double ms = 0.0; int slot = -1; };
    Res res[4];
    int  best = -1;
    bool has_run = false;
};

inline State g_state;

// Human labels for the three methods (also used in the results table).
inline const char* methodName(int m) {
    switch (m) {
    case 0: return "1) Legacy Alt+P (V1 full-mesh IoU)";
    case 1: return "2) IoU Ctrl+G (V3RS rim-sil OFF, RMSE-blended)";
    case 2: return "3) Rim chamfer (V3RS, anatomic rim)";
    case 3: return "4) Pure squash-IoU (V3RS pure_iou = Ctrl+I)";
    default: return "?";
    }
}

// =====================================================================
// run() -- execute the checked methods and capture each into an F9 slot.
//   Phase A: run every checked method from the shared start pose, stash
//            its result mesh (NO F9 capture yet -> no slot clobber).
//   Phase B: re-apply each stashed result and capture it into Run slot m,
//            scored with rasterize_iou2d_v3rs at score_step (same for all).
//   Phase C: restore original pose (or the best method's, if keep_best).
// =====================================================================
inline void run() {
    using clk = std::chrono::steady_clock;
    auto& S = g_state;
    for (int m = 0; m < 4; ++m) S.res[m] = State::Res{};
    S.best = -1;

    if (!liverMesh3D || !g_boundaryDistMap.valid) {
        std::cerr << "[SilCompare] need a loaded scene + valid boundary map "
                     "(run HemiAuto (O) and have a SAM2 mask first)." << std::endl;
        return;
    }

    auto organs = getOrganList();
    if (organs.empty()) return;
    const size_t NO = organs.size();

    // ---- snapshot start pose (all organs) ----
    std::vector<std::vector<GLfloat>> snap_v(NO), snap_n(NO);
    for (size_t i = 0; i < NO; ++i)
        if (organs[i]) { snap_v[i] = organs[i]->mVertices; snap_n[i] = organs[i]->mNormals; }

    auto restoreStart = [&]() {
        for (size_t i = 0; i < NO; ++i)
            if (organs[i]) {
                organs[i]->mVertices = snap_v[i];
                organs[i]->mNormals  = snap_n[i];
                setUp(*organs[i]);
            }
    };

    // per-method result stash (vertices+normals of every organ)
    struct Stash { bool valid = false; std::vector<std::vector<GLfloat>> v, n; double ms = 0.0; };
    Stash st[4];
    auto stashCurrent = [&](int m, double ms) {
        st[m].valid = true; st[m].ms = ms;
        st[m].v.assign(NO, {}); st[m].n.assign(NO, {});
        for (size_t i = 0; i < NO; ++i)
            if (organs[i]) { st[m].v[i] = organs[i]->mVertices; st[m].n[i] = organs[i]->mNormals; }
    };
    auto applyStash = [&](int m) {
        for (size_t i = 0; i < NO; ++i)
            if (organs[i] && i < st[m].v.size()) {
                organs[i]->mVertices = st[m].v[i];
                organs[i]->mNormals  = st[m].n[i];
                setUp(*organs[i]);
            }
    };

    // ---- save V3RS toggles so the compare doesn't change user settings ----
    const bool  sv_useRim  = g_ctrlgsUseRimSil;
    const bool  sv_rimAnat = g_ctrlgsRimSilAnatomic;
    const float sv_lamSil  = g_ctrlgsLambdaSil;
    const float sv_lamRim  = g_ctrlgsLambdaRimSil;
    const bool  sv_pureIoU = g_ctrlgsPureIoUMode;

    // ===== Phase A: run (no capture) =====
    if (S.run_m1) {
        restoreStart();
        auto t0 = clk::now();
        runShiftE();                                   // [1] legacy
        stashCurrent(0, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    if (S.run_m2) {
        restoreStart();
        g_ctrlgsUseRimSil      = false;                // [2] area-IoU, RMSE-blended
        g_ctrlgsRimSilAnatomic = false;
        g_ctrlgsPureIoUMode    = false;
        g_ctrlgsLambdaSil      = S.m2_lambda_sil;      // force IoU to dominate
        auto t0 = clk::now();
        runBipopCmaesV3RS(g_activeQuadrantMask);
        stashCurrent(1, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    if (S.run_m3) {
        restoreStart();
        g_ctrlgsUseRimSil      = true;                 // [3] boundary-distance
        g_ctrlgsRimSilAnatomic = true;                 // anatomic RIM
        g_ctrlgsPureIoUMode    = false;
        if (g_ctrlgsLambdaRimSil <= 0.0f) g_ctrlgsLambdaRimSil = 0.3f;
        auto t0 = clk::now();
        runBipopCmaesV3RS(g_activeQuadrantMask);
        stashCurrent(2, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    if (S.run_m4) {
        restoreStart();
        // [4] Pure squash-IoU = exactly the Ctrl+I objective: V3RS pipeline
        //   with pure_iou_mode ON -> cost = (1 - IoU2D), RMSE cap bypassed,
        //   selector by IoU. rim-sil OFF so it is the clean IoU experiment.
        g_ctrlgsUseRimSil      = false;
        g_ctrlgsRimSilAnatomic = false;
        g_ctrlgsPureIoUMode    = true;
        auto t0 = clk::now();
        runBipopCmaesV3RS(g_activeQuadrantMask);
        stashCurrent(3, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }

    // restore toggles immediately (before capture/return)
    g_ctrlgsUseRimSil      = sv_useRim;
    g_ctrlgsRimSilAnatomic = sv_rimAnat;
    g_ctrlgsLambdaSil      = sv_lamSil;
    g_ctrlgsLambdaRimSil   = sv_lamRim;
    g_ctrlgsPureIoUMode    = sv_pureIoU;

    // ===== Phase B: capture burst (no optimiser between -> slots stable) =====
    // rasterize_iou2d_v3rs REQUIRES dist_map.size() == imgW*imgH, so the
    // image dims MUST come from the boundary map itself (NOT OrbitCam.calib).
    const int imgW = g_boundaryDistMap.width;
    const int imgH = g_boundaryDistMap.height;
    const int step = (S.score_step < 1) ? 1 : S.score_step;
    const glm::mat4 silView = buildSilhouetteView();
    const glm::mat4 silProj = buildSilhouetteProj();

    // full-mesh indices, GLuint -> uint32_t (capture() wants uint32_t)
    std::vector<uint32_t> tris(liverMesh3D->mIndices.begin(),
                               liverMesh3D->mIndices.end());

    // warm the shared target-mask cache once so all methods score against the
    // identical target (no-op if squash disabled; harmless otherwise).
    CmaesRefineV3RS::ensureSilTargetMaskCache(g_boundaryDistMap.data, imgW, imgH, step);

    // F9 (SilOverlay) only has 3 Run slots (0,1,2). We score ALL checked
    // methods, but pack their composites into the available slots in method
    // order (so if all 4 are checked the 4th method's IoU is still printed
    // and scored; it just shares the visual slot rota). The console line is
    // the source of truth for the IoU/ms table; F9 is the visual aid.
    int next_slot = 0;
    for (int m = 0; m < 4; ++m) {
        if (!st[m].valid) continue;
        applyStash(m);
        const int slot = (next_slot < 3) ? next_slot : (3 - 1);  // clamp to last
        const float iou = SilOverlay::capture(
            SilOverlay::g_silOverlay, /*run_idx=*/slot,
            liverMesh3D, tris, silView, silProj,
            g_boundaryDistMap.data, imgW, imgH, step,
            /*scale_value=*/1.0f);
        S.res[m].valid = true;
        S.res[m].iou   = iou;
        S.res[m].ms    = st[m].ms;
        S.res[m].slot  = slot;
        std::cout << "[SilCompare] " << methodName(m)
                  << "  IoU=" << iou << "  (" << st[m].ms << " ms)"
                  << "  -> F9 Run " << (slot + 1) << std::endl;
        next_slot++;
    }

    // pick winner by scored IoU
    float bi = -1.0f;
    for (int m = 0; m < 4; ++m)
        if (S.res[m].valid && S.res[m].iou > bi) { bi = S.res[m].iou; S.best = m; }

    // ===== Phase C: final pose =====
    if (S.keep_best && S.best >= 0) applyStash(S.best);
    else                            restoreStart();

    // open F9 on the winner's slot so the comparison is visible
    if (S.best >= 0) SilOverlay::g_silOverlay.currentSlot = S.res[S.best].slot;
    // clear any stale [WINNER] tag left by the last real V3RS run so the
    // F9 combo doesn't mislabel an unrelated Run slot during a compare.
    SilOverlay::g_silOverlay.bestRunIdx = -1;
    SilOverlay::g_silOverlay.showWindow = true;
    S.has_run = true;
}

// =====================================================================
// drawSection() -- the ImGui block. Call inside drawGBody.
// =====================================================================
inline void drawSection() {
    auto& S = g_state;

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f),
                       "Silhouette method compare -> F9");

    const bool ready = (liverMesh3D != nullptr) && g_boundaryDistMap.valid;
    if (!ready) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f),
                           "Run HemiAuto (O) + load a SAM2 mask first.");
    }

    ImGui::Checkbox("1) Legacy Alt+P (runShiftE)", &S.run_m1);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("V1 engine, full-mesh CPU raster IoU rebuilt every\n"
                          "eval. The accuracy baseline (reaches ~0.96).");
    ImGui::Checkbox("2) IoU Ctrl+G (V3RS rim-sil OFF, RMSE-blended)", &S.run_m2);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("runBipopCmaesV3RS, rim-sil OFF, cost = RMSE_W +\n"
                          "lambda_sil*(1-IoU). The RMSE term + RMSE accept cap\n"
                          "cap IoU around ~0.7 (it rejects IoU gains that raise\n"
                          "3D RMSE). Kept for reference / ablation.");
    ImGui::Checkbox("3) Rim chamfer (V3RS, anatomic rim)", &S.run_m3);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("runBipopCmaesV3RS with the ANATOMIC rim-sil penalty\n"
                          "ON: rim vertices -> g_boundaryDistMap lookup. A\n"
                          "point-set boundary-distance objective. Objective is\n"
                          "boundary distance, not area IoU.");
    ImGui::Checkbox("4) Pure squash-IoU (V3RS pure_iou = Ctrl+I)", &S.run_m4);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("THE key experiment: V3RS pipeline (BIPOP, step16\n"
                          "squash raster) with objective swapped to pure IoU.\n"
                          "cost = (1 - IoU2D), RMSE cap bypassed, selector by\n"
                          "IoU. This is the 'Ctrl+G mechanism, squash-IoU\n"
                          "objective' run -- compare its IoU to method 1.");

    ImGui::SliderInt("score step", &S.score_step, 1, 32);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Yardstick rasteriser stride. SAME for all methods so\n"
                          "the IoU readouts are comparable. 16 = V3RS default.");
    ImGui::SliderFloat("m2 lambda_sil", &S.m2_lambda_sil, 0.5f, 16.0f, "%.1f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Forced onto method 2 only, for its run. Stock 0.2 is\n"
                          "too weak to make IoU dominate the V3R chamfer term.");
    ImGui::Checkbox("keep best pose (else restore original)", &S.keep_best);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("OFF = fully non-destructive: pose is restored after\n"
                          "comparing. ON = leave the best-IoU method applied.");

    ImGui::BeginDisabled(!ready ||
                         (!S.run_m1 && !S.run_m2 && !S.run_m3 && !S.run_m4));
    if (ImGui::Button("Run checked & compare (capture to F9)")) {
        run();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Open F9##silcmp")) {
        SilOverlay::g_silOverlay.showWindow = true;
    }

    // ----- [CHECKPOINT] one-click Alt+P vs Ctrl+I, side-by-side in F9 -----
    //  Runs ONLY method 1 (Alt+P) and method 4 (Ctrl+I) from the same start
    //  pose, scores both with the identical squash yardstick (Phase B), and
    //  flips the F9 window into compare mode pointed at the two slots we just
    //  filled. Full-mesh IoU for each still prints to the console during the
    //  run ([Shift+E] for Alt+P, [Ctrl+Shift+G] for Ctrl+I), so the 2x2
    //  (full-mesh vs squash) is readable from one click.
    ImGui::Spacing();
    ImGui::BeginDisabled(!ready);
    if (ImGui::Button("Checkpoint: Alt+P vs Ctrl+I  (run + side-by-side)")) {
        const bool s1 = S.run_m1, s2 = S.run_m2, s3 = S.run_m3, s4 = S.run_m4;
        S.run_m1 = true;  S.run_m2 = false; S.run_m3 = false; S.run_m4 = true;
        run();                              // backend: both methods, captured to F9
        S.run_m1 = s1; S.run_m2 = s2; S.run_m3 = s3; S.run_m4 = s4;

        auto& ov = SilOverlay::g_silOverlay;
        ov.compareSlotA  = (S.res[0].valid ? S.res[0].slot : 0);   // method 1 = Alt+P
        ov.compareSlotB  = (S.res[3].valid ? S.res[3].slot : 1);   // method 4 = Ctrl+I
        ov.compareLabelA = "Alt+P  (V1 full-mesh raster)";
        ov.compareLabelB = "Ctrl+I (V3RS pure squash-IoU)";
        ov.compareMode   = true;
        ov.showWindow    = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Runs method 1 and method 4 from the SAME start pose,\n"
                          "scores both with the SAME squash yardstick, and opens\n"
                          "F9 in side-by-side mode (Alt+P top, Ctrl+I bottom).\n"
                          "Full-mesh IoU for each prints to the console.");
    ImGui::EndDisabled();

    ImGui::TextDisabled("(F9 has 3 view slots; if 4 methods are checked the\n"
                        "IoU/ms table below still scores all of them.)");

    // ----- results -----
    if (S.has_run) {
        ImGui::Separator();
        ImGui::TextDisabled("results (scored at step %d, identical yardstick):", S.score_step);
        for (int m = 0; m < 4; ++m) {
            if (!S.res[m].valid) {
                ImGui::TextDisabled("  %-44s  (not run)", methodName(m));
                continue;
            }
            const bool win = (m == S.best);
            ImVec4 col = win ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f)
                             : ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
            ImGui::TextColored(col, "  %-44s  IoU=%.4f  %7.0f ms  -> F9 Run %d%s",
                               methodName(m), S.res[m].iou, S.res[m].ms,
                               S.res[m].slot + 1, win ? "  [WIN]" : "");
        }
        ImGui::TextDisabled("Flip Run slots in the F9 window to compare silhouettes.");
    }
}

}  // namespace SilCompare

#endif  // REGISTRATION_ACTIONS_H_INCLUDED (compile-context guard)

#endif  // SIL_COMPARE_PANEL_H
