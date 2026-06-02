// =============================================================================
//  SilOverlayDebug.h —  ImGui preview window for V3RS silhouette IoU debug
//
//  Mirrors the AR.h / ARSave pattern (State + capture + drawPreviewWindow).
//  Captures per-Run AND final-pose silhouette comparison composites during a
//  Ctrl+Shift+G session, uploads them as GL textures, and renders them in an
//  ImGui window selectable via a "Run 1..10 / Final" combo box.
//
//  Composite color scheme (matches IoUDebugDump.h):
//      green : intersection  (source AND target -- aligned)
//      red   : target only   (mask says liver, source missed)
//      blue  : source only   (source overshoots beyond mask)
//      black : neither
//
//  Lifecycle:
//      static SilOverlay::State g_silOverlay;
//
//      // During Ctrl+Shift+G (Phase 0 selector callback, per-Run):
//      SilOverlay::capture(g_silOverlay, run_idx,
//                          liverMesh3D, indices, view, proj,
//                          dist_map, imgW, imgH, step=8,
//                          scale_value);
//
//      // After Phase E (final pose applied):
//      SilOverlay::captureFinal(g_silOverlay, best_run_idx,
//                               liverMesh3D, indices, view, proj,
//                               dist_map, imgW, imgH, step=8,
//                               scale_value);
//
//      // F9 key handler in main.cpp:
//      g_silOverlay.showWindow = !g_silOverlay.showWindow;
//
//      // In the ImGui frame pass in main.cpp:
//      SilOverlay::drawPreviewWindow(g_silOverlay, vpW, gWindowHeight);
//
//  Notes:
//      - capture() must be called from the GL thread (main thread). The Phase
//        0 selector callback is invoked synchronously inside the wrapper, so
//        we're on the main thread there.
//      - GL textures persist across sessions; capture() reuses the same
//        texture object per slot (glTexImage2D replaces contents).
//      - The captured texture size is gw x gh (e.g. 240x135 at step=8). ImGui
//        upscales for display; that's intentional so the composite is sharp
//        without inflating GPU memory.
// =============================================================================

#pragma once

#include <algorithm>
#include <array>      // [NEW V3RS-VIZ] for std::array<uint8_t,3> heat colours
#include <cmath>      // [NEW V3RS-CONTAIN] for std::fabs
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include <GL/glew.h>

#include "imgui.h"

#include "mCutMesh.h"
#include "CmaesRefineV3RS.h"     // rasterize_iou2d_v3rs with capture outputs

namespace SilOverlay {

constexpr int kNumRuns   = 10;
constexpr int kFinalSlot = kNumRuns;           // index of the Final slot
// Diagnostic slots (F10 vertex-squash A/B). Two GL textures so the F9
// window can toggle bbox <-> vtx-squash without re-capturing.
constexpr int kDiagBBoxSlot = kNumRuns + 1;    // F10: triangle-bbox splat
constexpr int kDiagVtxSlot  = kNumRuns + 2;    // F10: vertex squash 3x3
constexpr int kNumSlots     = kNumRuns + 3;    // 10 Runs + Final + 2 diag
// Combo entries in the F9 selector: Run 1..10, Final, Diagnostic. The
// two diag slots share ONE combo entry; a checkbox switches between
// them, so the combo has one fewer entry than there are slots.
constexpr int kDiagComboIdx    = kNumRuns + 1; // "Diagnostic (F10)" entry
constexpr int kNumComboEntries = kNumRuns + 2;

struct RunSlot {
    GLuint tex      = 0;
    int    w        = 0;       // grid width (gw)
    int    h        = 0;       // grid height (gh)
    float  iou      = 0.0f;    // IoU2D score for this slot
    float  scale    = 1.0f;    // SRT scale for this slot
    int    inter_px = 0;       // intersection pixel count
    int    union_px = 0;       // union pixel count
    // ----- [NEW V3RS-CONTAIN] Precision / recall counters --------------
    // source_px = cells with hitmap=1 (occ-cleared if filter active)
    // target_px = cells with tmask =1 (occ-cleared if filter active)
    // precision = inter_px / source_px
    // recall    = inter_px / target_px
    // Together they distinguish overshoot (precision<<recall) from
    // undershoot (recall<<precision) from positional error (both low).
    int    source_px = 0;
    int    target_px = 0;
    bool   has_data = false;   // false until first capture
    // ----- [NEW V3RS-VIZ] Rim-row metadata -------------------------------
    // Set when captureImpl was called with rim diagnostic inputs (dist_map
    // + img dims + step + rim_sil_max_px). Used by drawPreviewWindow to
    // (a) report rim_sil_loss alongside IoU, and (b) toggle the bottom-row
    // legend on. has_rim_row = false → texture is the legacy 3-panel
    // layout and the UI hides the rim legend.
    bool   has_rim_row     = false;
    float  rim_sil_loss    = 0.0f;   // mean per-cell penalty in [0, 1]
    int    rim_count       = 0;      // source-boundary cells contributing
    float  rim_sil_max_px  = 100.0f; // normalisation cap used at build time
    // Breakdown for diagnosis: which kind of rim cell dominates?
    int    rim_outside_count = 0;    // d>=9000 (source rim BEYOND target)
    int    rim_near_count    = 0;    // d<=5 (aligned with target boundary)
    int    rim_inside_count  = 0;    // 5<d<max (inside target mask, capped)
    // ----- [NEW V3RS-OCC-VIZ] Instrument-occluded variant ----------------
    // Populated when captureImpl received a non-null instrument_dist_map.
    // The primary (no-occ) fields above always reflect the IoU computed
    // WITHOUT the instrument filter, regardless of how the function was
    // called — this keeps the slot's "raw" meaning unambiguous. The alt
    // fields below mirror them with the instrument filter active. F9's
    // checkbox switches the displayed variant; the toggle does not
    // re-rasterize since both variants are cached at capture time.
    //
    // When has_alt = false (instrument map was null / unusable / sizes
    // didn't match), the checkbox is disabled and the F9 window behaves
    // identically to the pre-feature path.
    bool   has_alt              = false;
    GLuint tex_alt              = 0;
    float  iou_alt              = 0.0f;
    int    inter_alt_px         = 0;
    int    union_alt_px         = 0;
    // [NEW V3RS-CONTAIN] Same precision/recall counters for alt variant.
    int    source_alt_px        = 0;
    int    target_alt_px        = 0;
    float  rim_sil_loss_alt     = 0.0f;
    int    rim_count_alt        = 0;
    int    rim_outside_count_alt = 0;
    int    rim_near_count_alt    = 0;
    int    rim_inside_count_alt  = 0;
};

struct State {
    RunSlot slots[kNumSlots];
    bool    showWindow       = false;   // toggled by F9
    int     currentSlot      = kFinalSlot;  // combo selection index
    int     bestRunIdx       = -1;      // 0..9, set by captureFinal
    int     captureSessionId = 0;       // increments each session
    // F10 vertex-squash diagnostic state.
    bool    hasDiag          = false;   // true once captureDiag has run
    bool    showVtxSquash    = false;   // F9 checkbox: bbox <-> vtx-squash
    // ----- [NEW V3RS-OCC-VIZ] Instrument-occlusion display toggle --------
    // F9 checkbox state. Defaults ON so users who have an instrument
    // mask available see the occluded variant by default (it's the
    // "scientifically meaningful" IoU — V3RS's cost-function variant).
    // When a slot has has_alt=false (no instrument data captured), the
    // checkbox is greyed out and the no-occ variant is shown regardless.
    bool    showWithOcc      = true;
    // ----- [COMPARE] Side-by-side A|B compare view ----------------------
    // Toggled by the F9 "Compare A|B" checkbox or set by the SilCompare
    // checkpoint button. compareSlotA/B index into slots[] (0-based:
    // 0..9 = Run 1..10, kFinalSlot = Final). The squash IoU printed is
    // slot.iou -- the score of the very composite displayed, identical
    // yardstick for both rows. Full-mesh IoU is in the console per method.
    bool        compareMode   = false;
    int         compareSlotA  = 0;     // top row  (e.g. Alt+P)
    int         compareSlotB  = 1;     // bottom row (e.g. Ctrl+I)
    std::string compareLabelA = "A";
    std::string compareLabelB = "B";
};

// -----------------------------------------------------------------------------
//  Build the composite RGB image from the hitmap and target_mask buffers.
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
//  Build a 3-panel triptych:  [ Source | Target | Overlay ]
//  Layout:
//      panel 0 (left):    source-only.  white where hitmap[i]=1, else dark grey
//      panel 1 (middle):  target-only.  white where tmask[i]=1, else dark grey
//      panel 2 (right):   overlay (green=match, red=miss, blue=overshoot)
//      separators        : 2px vertical mid-grey strips between panels
//
//  Total image size: (3*gw + 4) x gh.  Dark grey (instead of black) makes the
//  background distinguishable from "no signal" pixels inside the panels --
//  important to spot raster holes vs. background.
//
//  Out dimensions are written back so the caller knows the actual texture
//  size to upload.
// -----------------------------------------------------------------------------
constexpr int kPanelSep   = 2;                 // pixels between panels
constexpr int kPanelCount = 3;                 // source / target / overlay (TOP row)
constexpr int kPanelRows  = 2;                 // [NEW] top: SRC|TGT|OVL, bottom: RIM|DIST|ALN

inline int tripletWidth(int gw) {
    return kPanelCount * gw + (kPanelCount + 1) * kPanelSep;
}
// [NEW] Composite height depends on whether the rim-diagnostic bottom row
// is requested. Layout:
//   has_rim=false: H = gh                          (legacy 1-row)
//   has_rim=true : H = gh + kPanelSep + gh        (2 rows separated by 1 sep strip)
inline int compositeHeightFor(int gh, bool has_rim) {
    return has_rim ? (2 * gh + kPanelSep) : gh;
}

inline void buildComposite(const std::vector<uint8_t>& hitmap,
                           const std::vector<uint8_t>& tmask,
                           int gw, int gh,
                           std::vector<uint8_t>& out_rgb,
                           int* out_w = nullptr,
                           int* out_h = nullptr,
                           // [NEW V3RS-VIZ] Rim-diagnostic bottom row inputs.
                           // When dist_map_ptr is non-null AND its size matches
                           // img_w*img_h, the function builds a 2-row 2x3 layout:
                           //   row 0: Source | Target | Overlay  (legacy)
                           //   row 1: RimSrc | DistMap | RimAln  (new)
                           // When any rim arg is missing/invalid, the function
                           // falls back to the legacy 1-row 3-panel layout
                           // (byte-identical to pre-feature behaviour).
                           const std::vector<float>* dist_map_ptr = nullptr,
                           int img_w = 0,
                           int img_h = 0,
                           int step  = 1,
                           float rim_sil_max_px = 100.0f,
                           const std::vector<float>* inst_dist_map = nullptr,
                           float inst_thresh_px = 0.0f,
                           // [NEW V3RS-RIM-ANAT] Optional per-cell mask of
                           // anatomic RIM cells (size gw*gh; 1 = cell contains
                           // at least one anatomical RIM vertex projection).
                           // When non-null AND size-matched, the bottom row's
                           // RimSrc / RimAln panels are driven by this mask
                           // instead of the per-cell 4-neighbour boundary.
                           // When null, the legacy raster-boundary mode runs.
                           const std::vector<uint8_t>* rim_cell_mask = nullptr)
{
    const int W = tripletWidth(gw);
    // Decide whether to draw the bottom row.
    const bool has_rim =
        (dist_map_ptr != nullptr)
        && !dist_map_ptr->empty()
        && (img_w > 0) && (img_h > 0)
        && ((size_t)dist_map_ptr->size() == (size_t)img_w * (size_t)img_h)
        && (step >= 1);
    const int H = compositeHeightFor(gh, has_rim);
    // Background = dark grey so individual panels are visible against the
    // window background; separators darker still.
    constexpr uint8_t kBgGrey  = 28;
    constexpr uint8_t kSepGrey = 12;
    constexpr uint8_t kFgWhite = 235;
    constexpr uint8_t kGreen   = 230;
    constexpr uint8_t kRed     = 230;
    constexpr uint8_t kBlue    = 230;

    out_rgb.assign((size_t)W * (size_t)H * 3, kBgGrey);

    // Pre-fill the separator columns.
    auto setSepColumn = [&](int x) {
        if (x < 0 || x >= W) return;
        for (int y = 0; y < H; ++y) {
            const size_t off = ((size_t)y * (size_t)W + (size_t)x) * 3;
            out_rgb[off + 0] = kSepGrey;
            out_rgb[off + 1] = kSepGrey;
            out_rgb[off + 2] = kSepGrey;
        }
    };
    // separator x-positions: 0..kPanelSep-1, then between each pair, then trailing
    for (int s = 0; s <= kPanelCount; ++s) {
        const int x0 = s * (gw + kPanelSep);  // 0, gw+sep, 2gw+2sep, 3gw+3sep
        for (int dx = 0; dx < kPanelSep; ++dx) setSepColumn(x0 + dx);
    }

    // [NEW V3RS-VIZ] Horizontal separator strip between rows. The fill
    // colour matches kSepGrey so it visually blends with the vertical
    // separators where they cross. Done as a row sweep so the separator
    // is solid across the full image width.
    if (has_rim) {
        for (int dy = 0; dy < kPanelSep; ++dy) {
            const int y = gh + dy;
            const size_t row_off = (size_t)y * (size_t)W * 3;
            for (int x = 0; x < W; ++x) {
                out_rgb[row_off + (size_t)x * 3 + 0] = kSepGrey;
                out_rgb[row_off + (size_t)x * 3 + 1] = kSepGrey;
                out_rgb[row_off + (size_t)x * 3 + 2] = kSepGrey;
            }
        }
    }

    // Panel x-offsets (where the per-panel pixel data starts in W).
    const int x_off_src = 0 * (gw + kPanelSep) + kPanelSep;
    const int x_off_tgt = 1 * (gw + kPanelSep) + kPanelSep;
    const int x_off_ovl = 2 * (gw + kPanelSep) + kPanelSep;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < gw; ++x) {
            const size_t i = (size_t)y * (size_t)gw + (size_t)x;
            const bool s = (i < hitmap.size()) ? (hitmap[i] != 0) : false;
            const bool t = (i < tmask.size())  ? (tmask[i]  != 0) : false;

            // Source panel (white where source pixel exists).
            if (s) {
                const size_t off = ((size_t)y * (size_t)W +
                                    (size_t)(x_off_src + x)) * 3;
                out_rgb[off + 0] = kFgWhite;
                out_rgb[off + 1] = kFgWhite;
                out_rgb[off + 2] = kFgWhite;
            }
            // Target panel (white where target pixel exists).
            if (t) {
                const size_t off = ((size_t)y * (size_t)W +
                                    (size_t)(x_off_tgt + x)) * 3;
                out_rgb[off + 0] = kFgWhite;
                out_rgb[off + 1] = kFgWhite;
                out_rgb[off + 2] = kFgWhite;
            }
            // Overlay panel (green=intersection / red=miss / blue=overshoot).
            uint8_t r = 0, g = 0, b = 0;
            if (s && t)      { g = kGreen; }
            else if (t)      { r = kRed;   }
            else if (s)      { b = kBlue;  }
            if (r | g | b) {
                const size_t off = ((size_t)y * (size_t)W +
                                    (size_t)(x_off_ovl + x)) * 3;
                out_rgb[off + 0] = r;
                out_rgb[off + 1] = g;
                out_rgb[off + 2] = b;
            }
        }
    }

    // ----- [NEW V3RS-VIZ] Bottom row: rim diagnostic panels ----------
    // Same horizontal x-offsets as the top row; bottom row starts at
    // y_off_bot in image space. For each cell we evaluate:
    //   (a) source-rim flag (4-neighbour boundary test on the hitmap)
    //   (b) distance to target boundary (sample dist_map at centre pixel)
    //   (c) instrument-occlusion flag (matches the IoU rasterizer's gate)
    // and paint three panels:
    //   Panel 4 (RimSrc): hitmap with source-boundary cells highlighted
    //   Panel 5 (DistMap): heatmap of dist values at each cell
    //   Panel 6 (RimAln): per-cell rim_sil penalty for source-rim cells
    //
    // This block runs only when has_rim is true; otherwise the bottom
    // half of the image stays at the background colour (which it is by
    // default since out_rgb was assigned kBgGrey, and H==gh so there is
    // no bottom half).
    if (has_rim) {
        const int y_off_bot = gh + kPanelSep;
        const float D_MAX = (rim_sil_max_px > 0.0f) ? rim_sil_max_px : 100.0f;
        const bool occ_enabled =
            (inst_dist_map != nullptr)
            && (inst_dist_map->size() == (size_t)img_w * (size_t)img_h)
            && (inst_thresh_px >= 0.0f);
        const float* occ_data = occ_enabled ? inst_dist_map->data() : nullptr;
        const float* dist_data = dist_map_ptr->data();

        // [NEW V3RS-RIM-ANAT] Anatomic-cell-mask mode.
        // When the caller supplied a per-cell anatomic RIM mask (built by
        // the rasterizer via is_rim_anatomic_per_vertex), use it to decide
        // which cells light up as "RIM" in panels 4 & 6. Otherwise fall
        // back to the per-cell 4-neighbour raster-boundary test.
        const bool anatomic_cells =
            (rim_cell_mask != nullptr)
            && !rim_cell_mask->empty()
            && (rim_cell_mask->size() == (size_t)gw * (size_t)gh);

        // Per-cell heat colours: utility lambdas for readability.
        auto heat_dist = [&](float d) -> std::array<uint8_t, 3> {
            // Panel 5 colourisation:
            //   d >= 9000 (outside mask) -> dark blue: clearly distinct
            //   0 <= d <= D_MAX          -> gradient: cyan -> yellow -> red
            //   d > D_MAX                -> saturated dark red
            if (d >= 9000.0f) return {10, 20, 80};
            const float t = std::min(d / D_MAX, 1.0f);
            // Linear gradient through cyan(0,200,200) -> yellow(255,220,0)
            //                       -> red(200,40,40).
            if (t < 0.5f) {
                const float u = t * 2.0f;                  // 0..1
                const uint8_t R = (uint8_t)(  0 + u * (255 -   0));
                const uint8_t G = (uint8_t)(200 + u * (220 - 200));
                const uint8_t B = (uint8_t)(200 + u * (  0 - 200));
                return {R, G, B};
            } else {
                const float u = (t - 0.5f) * 2.0f;         // 0..1
                const uint8_t R = (uint8_t)(255 + u * (200 - 255));
                const uint8_t G = (uint8_t)(220 + u * ( 40 - 220));
                const uint8_t B = (uint8_t)(  0 + u * ( 40 -   0));
                return {R, G, B};
            }
        };
        auto heat_align = [&](float d) -> std::array<uint8_t, 3> {
            // Panel 6 colourisation for source-rim cells:
            //   d >= 9000 (outside mask)  -> MAGENTA: source rim BEYOND target
            //   d <= 5 px                 -> GREEN: aligned to target boundary
            //   5 < d < D_MAX             -> green -> yellow -> red gradient
            //   d >= D_MAX                -> dark red (max in-mask penalty)
            if (d >= 9000.0f)  return {220,  60, 220};      // magenta
            if (d <=    5.0f)  return { 40, 220,  60};      // green
            const float t = std::min((d - 5.0f) / std::max(1.0f, D_MAX - 5.0f), 1.0f);
            if (t < 0.5f) {
                const float u = t * 2.0f;                   // 0..1: green->yellow
                const uint8_t R = (uint8_t)( 40 + u * (240 -  40));
                const uint8_t G = (uint8_t)(220 + u * (220 - 220));
                const uint8_t B = (uint8_t)( 60 + u * ( 30 -  60));
                return {R, G, B};
            } else {
                const float u = (t - 0.5f) * 2.0f;          // 0..1: yellow->red
                const uint8_t R = (uint8_t)(240 + u * (200 - 240));
                const uint8_t G = (uint8_t)(220 + u * ( 30 - 220));
                const uint8_t B = (uint8_t)( 30 + u * ( 30 -  30));
                return {R, G, B};
            }
        };
        constexpr uint8_t kRimInteriorGrey = 70;   // source but not on its boundary
        // [FIXED V3RS-VIZ] Occluded cell colour. Was kRimOccludedGrey=18,
        // visually indistinguishable from kBgGrey=28 -- user couldn't tell
        // whether instrument-shadow regions were actually being excluded
        // from the rim_sil computation. Bumped to dark purple so excluded
        // cells stand out clearly in all three bottom panels.
        constexpr uint8_t kRimOcclR = 60;
        constexpr uint8_t kRimOcclG =  0;
        constexpr uint8_t kRimOcclB = 80;

        for (int gy = 0; gy < gh; ++gy) {
            const size_t row_grd = (size_t)gy * (size_t)gw;
            const int ipy = gy * step + step / 2;
            const int my  = (ipy < 0) ? 0
                                     : (ipy >= img_h ? img_h - 1 : ipy);
            const size_t row_img = (size_t)my * (size_t)img_w;
            const int y_img = y_off_bot + gy;
            for (int gx = 0; gx < gw; ++gx) {
                const size_t idx = row_grd + (size_t)gx;
                const bool s_self = (idx < hitmap.size()) ? (hitmap[idx] != 0) : false;

                // RIM-cell determination. Two modes:
                //   ANATOMIC: cell is RIM iff rim_cell_mask[idx] == 1.
                //     The mask comes from the rasterizer's projection of
                //     anatomical RIM vertices through the same MVP -- so a
                //     "rim" cell here is one that contains at least one
                //     projected RIM vertex.
                //   RASTER-BOUNDARY (legacy): cell is RIM iff it is source
                //     AND has at least one non-source 4-neighbour. Grid
                //     edges count as non-source.
                bool is_rim;
                if (anatomic_cells) {
                    is_rim = (idx < rim_cell_mask->size()) &&
                             ((*rim_cell_mask)[idx] != 0);
                } else {
                    const bool nl = (gx > 0)      ? (hitmap[idx - 1]            != 0) : false;
                    const bool nr = (gx + 1 < gw) ? (hitmap[idx + 1]            != 0) : false;
                    const bool nu = (gy > 0)      ? (hitmap[idx - (size_t)gw]   != 0) : false;
                    const bool nd = (gy + 1 < gh) ? (hitmap[idx + (size_t)gw]   != 0) : false;
                    is_rim = s_self && !(nl && nr && nu && nd);
                }
                // is_int = source cell that is NOT a rim cell. In anatomic
                // mode this is still meaningful (any source cell that isn't
                // anatomic-rim is "interior context"); the visual context
                // helps the user see where the source silhouette lies.
                const bool is_int = s_self && !is_rim;

                const int ipx = gx * step + step / 2;
                const int mx  = (ipx < 0) ? 0
                                         : (ipx >= img_w ? img_w - 1 : ipx);

                bool occluded = false;
                if (occ_enabled) {
                    const float idist = occ_data[row_img + (size_t)mx];
                    occluded = (idist < inst_thresh_px);
                }
                const float d = dist_data[row_img + (size_t)mx];

                // ----- Panel 4: RIM cells (source-boundary highlight) ------
                {
                    const size_t off =
                        ((size_t)y_img * (size_t)W +
                         (size_t)(x_off_src + gx)) * 3;
                    if (occluded) {
                        out_rgb[off + 0] = kRimOcclR;
                        out_rgb[off + 1] = kRimOcclG;
                        out_rgb[off + 2] = kRimOcclB;
                    } else if (is_rim) {
                        out_rgb[off + 0] = kFgWhite;
                        out_rgb[off + 1] = kFgWhite;
                        out_rgb[off + 2] = kFgWhite;
                    } else if (is_int) {
                        out_rgb[off + 0] = kRimInteriorGrey;
                        out_rgb[off + 1] = kRimInteriorGrey;
                        out_rgb[off + 2] = kRimInteriorGrey;
                    } /* else: leave kBgGrey */
                }

                // ----- Panel 5: Target distance heatmap --------------------
                {
                    const size_t off =
                        ((size_t)y_img * (size_t)W +
                         (size_t)(x_off_tgt + gx)) * 3;
                    if (occluded) {
                        out_rgb[off + 0] = kRimOcclR;
                        out_rgb[off + 1] = kRimOcclG;
                        out_rgb[off + 2] = kRimOcclB;
                    } else {
                        auto c = heat_dist(d);
                        out_rgb[off + 0] = c[0];
                        out_rgb[off + 1] = c[1];
                        out_rgb[off + 2] = c[2];
                    }
                }

                // ----- Panel 6: RIM alignment (per-cell penalty) -----------
                {
                    const size_t off =
                        ((size_t)y_img * (size_t)W +
                         (size_t)(x_off_ovl + gx)) * 3;
                    if (occluded) {
                        out_rgb[off + 0] = kRimOcclR;
                        out_rgb[off + 1] = kRimOcclG;
                        out_rgb[off + 2] = kRimOcclB;
                    } else if (is_rim) {
                        auto c = heat_align(d);
                        out_rgb[off + 0] = c[0];
                        out_rgb[off + 1] = c[1];
                        out_rgb[off + 2] = c[2];
                    } else if (is_int) {
                        out_rgb[off + 0] = kRimInteriorGrey;
                        out_rgb[off + 1] = kRimInteriorGrey;
                        out_rgb[off + 2] = kRimInteriorGrey;
                    } /* else: leave kBgGrey */
                }
            }
        }
    }

    if (out_w) *out_w = W;
    if (out_h) *out_h = H;
}

// -----------------------------------------------------------------------------
//  Internal: upload an RGB buffer into the slot's GL texture.
// -----------------------------------------------------------------------------
// Low-level: upload an RGB buffer to a GL texture id (allocates if 0).
// Used by uploadToSlot (primary variant) and directly by captureImpl
// when uploading the instrument-occlusion variant to slot.tex_alt.
inline void uploadTextureRGB(GLuint& tex,
                             const std::vector<uint8_t>& rgb,
                             int gw, int gh)
{
    if (tex == 0) {
        glGenTextures(1, &tex);
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 gw, gh, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

inline void uploadToSlot(RunSlot& slot,
                         const std::vector<uint8_t>& rgb,
                         int gw, int gh)
{
    uploadTextureRGB(slot.tex, rgb, gw, gh);
    slot.w = gw;
    slot.h = gh;
}

// -----------------------------------------------------------------------------
//  Core capture: rasterize liver at its CURRENT pose vs dist_map,
//  build composite, upload to slot.
//
//  `liver_indices_filtered` should be the SAME quadrant-filtered triangle
//  list that the optimizer used (so the viz matches the cost function).
//
//  Optional instrument occlusion: when `instrument_dist_map` is non-null
//  and its size matches dist_map, cells whose centre pixel has
//  instrument-distance < `instrument_thresh_px` are excluded from the
//  IoU computation AND cleared in the composite (shown as black).
//  Default (null) preserves the pre-feature path byte-for-byte.
// -----------------------------------------------------------------------------
inline float captureImpl(State& st,
                         int slot_idx,
                         const mCutMesh*               liver,
                         const std::vector<uint32_t>&  liver_indices_filtered,
                         const glm::mat4&              view_mat,
                         const glm::mat4&              proj_mat,
                         const std::vector<float>&     dist_map,
                         int   imgW, int imgH,
                         int   step,
                         float scale_value,
                         int   raster_mode = 0,
                         const std::vector<float>* instrument_dist_map = nullptr,
                         float instrument_thresh_px = 0.0f,
                         // [NEW V3RS-VIZ] When > 0 (and dist_map non-empty),
                         // the composite texture includes the rim diagnostic
                         // bottom row; the slot's rim_sil_loss + breakdown
                         // counters are populated. When <= 0, legacy 3-panel
                         // layout and rim metadata stays default.
                         float rim_sil_max_px = 0.0f,
                         // [NEW V3RS-RIM-ANAT] Per-vertex anatomical RIM flag.
                         // When non-null and size-matched to the full mesh,
                         // the rasterizer switches rim_sil to ANATOMIC mode:
                         //   - rim_cell_mask is produced (cells receiving any
                         //     RIM vertex projection)
                         //   - rim_sil_loss is averaged over visible RIM
                         //     vertices, not over source-boundary cells
                         //   - panels 4 & 6 highlight the anatomic mask
                         // When null (default), the legacy raster-boundary
                         // mode runs. Forwarded from the wrapper's
                         // p.is_rim_anatomic_full.
                         const std::vector<uint8_t>* is_rim_anatomic_per_vertex = nullptr)
{
    if (!liver || slot_idx < 0 || slot_idx >= kNumSlots) return 0.0f;
    if (liver->mVertices.empty() || liver_indices_filtered.empty()) return 0.0f;

    // Convert flat float vertex array to glm::vec3 (one-time per capture,
    // ~120 KB / call -- 11 captures per session worst case = ~1.3 MB).
    const size_t nV = liver->mVertices.size() / 3;
    std::vector<glm::vec3> positions(nV);
    for (size_t i = 0; i < nV; ++i) {
        positions[i] = glm::vec3(
            liver->mVertices[i*3 + 0],
            liver->mVertices[i*3 + 1],
            liver->mVertices[i*3 + 2]);
    }
    // liverMesh3D->mVertices is already at the target pose, so world = I.
    const glm::mat4 mvp = proj_mat * view_mat;

    // ----- [NEW V3RS-VIZ + RIM-ANAT] Rim-row capture --------------------
    // When the rim row is desired (rim_sil_max_px > 0), we ask the
    // rasterizer to ALSO compute rim_sil_loss and rim_cell_mask in one
    // go. That keeps the cell-mode decision (anatomic vs raster-boundary)
    // in a single place inside the rasterizer; the viz then just renders
    // whatever mask comes back.
    const bool want_rim_row =
        (rim_sil_max_px > 0.0f)
        && !dist_map.empty()
        && (dist_map.size() == (size_t)imgW * (size_t)imgH);
    const std::vector<uint8_t>* anat_in =
        (is_rim_anatomic_per_vertex && !is_rim_anatomic_per_vertex->empty()
         && is_rim_anatomic_per_vertex->size() == positions.size())
            ? is_rim_anatomic_per_vertex
            : nullptr;
    // [NEW V3RS-OCC-VIZ] Detect whether the caller supplied a usable
    // instrument mask. When yes we run TWO rasterize passes — one
    // without the filter (primary slot fields) and one with (alt
    // fields) — so F9's "Apply instrument occlusion" checkbox can
    // flip between them without re-rendering. When no, we run a
    // single no-occ pass and leave has_alt = false; F9 disables the
    // checkbox and behaviour is byte-identical to the pre-feature
    // path.
    const bool has_inst_input =
        (instrument_dist_map != nullptr)
        && (instrument_dist_map->size() == (size_t)imgW * (size_t)imgH);

    RunSlot& slot = st.slots[slot_idx];

    // ===== Variant 1: WITHOUT instrument occlusion (primary) ============
    // Always captured. Fills slot.tex / slot.iou / slot.inter_px /
    // slot.union_px / slot.has_rim_row + rim breakdown. Forces the
    // rasterizer's instrument args to null so this represents the
    // "raw" IoU regardless of how the caller invoked us.
    float ret_iou = 0.0f;
    {
        std::vector<uint8_t> hitmap_no, tmask_no;
        int gw_no = 0, gh_no = 0;
        float rim_loss_no = 0.0f;
        std::vector<uint8_t> rim_mask_no;
        const float iou_no = CmaesRefineV3RS::rasterize_iou2d_v3rs(
            positions, liver_indices_filtered, mvp,
            dist_map, imgW, imgH, step,
            &hitmap_no, &tmask_no, &gw_no, &gh_no,
            /*out_step1*/ nullptr, /*out_step2*/ nullptr, /*out_step3*/ nullptr,
            raster_mode,
            /*instrument_dist_map*/ nullptr,
            /*instrument_thresh_px*/ 0.0f,
            /*out_outside_ratio*/ nullptr,
            want_rim_row ? &rim_loss_no : nullptr,
            rim_sil_max_px,
            anat_in,
            want_rim_row ? &rim_mask_no : nullptr);

        if (gw_no <= 0 || gh_no <= 0) {
            // Rasterize failed (degenerate geometry, empty mesh in
            // viewport, etc.). Don't touch slot fields beyond a
            // minimal mark; the caller's defaults stay valid.
            return iou_no;
        }

        // Inter/union for slot metadata.
        // [NEW V3RS-CONTAIN] Source and target cell counts added in the
        // same pass — precision = inter/source, recall = inter/target.
        int inter = 0, uni = 0, src_n = 0, tgt_n = 0;
        const size_t N = (size_t)gw_no * (size_t)gh_no;
        for (size_t i = 0; i < N; ++i) {
            const bool s = (hitmap_no[i] != 0);
            const bool t = (tmask_no[i]  != 0);
            if (s) ++src_n;
            if (t) ++tgt_n;
            if (s || t) ++uni;
            if (s && t) ++inter;
        }

        // Rim breakdown (no occlusion → no occ-cell skip in classifier).
        float rim_sil_loss = rim_loss_no;
        int   rim_count    = 0;
        int   rim_outside  = 0;
        int   rim_near     = 0;
        int   rim_inside   = 0;
        if (want_rim_row && !rim_mask_no.empty()) {
            const float* dist_data = dist_map.data();
            for (int gy = 0; gy < gh_no; ++gy) {
                const size_t row_grd = (size_t)gy * (size_t)gw_no;
                const int ipy = gy * step + step / 2;
                const int my  = (ipy < 0) ? 0
                                         : (ipy >= imgH ? imgH - 1 : ipy);
                const size_t row_img = (size_t)my * (size_t)imgW;
                for (int gx = 0; gx < gw_no; ++gx) {
                    const size_t idx = row_grd + (size_t)gx;
                    if (!rim_mask_no[idx]) continue;
                    const int ipx = gx * step + step / 2;
                    const int mx  = (ipx < 0) ? 0
                                             : (ipx >= imgW ? imgW - 1 : ipx);
                    const float d = dist_data[row_img + (size_t)mx];
                    if (d >= 9000.0f) ++rim_outside;
                    else if (d <= 5.0f) ++rim_near;
                    else                ++rim_inside;
                    ++rim_count;
                }
            }
        }

        // Composite (instrument args forced null → "raw" view).
        std::vector<uint8_t> rgb_no;
        int trip_w_no = 0, trip_h_no = 0;
        buildComposite(hitmap_no, tmask_no, gw_no, gh_no,
                       rgb_no, &trip_w_no, &trip_h_no,
                       want_rim_row ? &dist_map : nullptr,
                       imgW, imgH, step, rim_sil_max_px,
                       /*inst_dist_map*/ nullptr, /*inst_thresh_px*/ 0.0f,
                       want_rim_row ? &rim_mask_no : nullptr);

        uploadToSlot(slot, rgb_no, trip_w_no, trip_h_no);
        slot.iou       = iou_no;
        slot.scale     = scale_value;
        slot.inter_px  = inter;
        slot.union_px  = uni;
        slot.source_px = src_n;
        slot.target_px = tgt_n;
        slot.has_data  = true;
        slot.has_rim_row       = want_rim_row;
        slot.rim_sil_loss      = rim_sil_loss;
        slot.rim_count         = rim_count;
        slot.rim_sil_max_px    = rim_sil_max_px;
        slot.rim_outside_count = rim_outside;
        slot.rim_near_count    = rim_near;
        slot.rim_inside_count  = rim_inside;
        // Default: assume no alt variant. The next block flips this
        // when the with-occ pass succeeds.
        slot.has_alt = false;
        ret_iou = iou_no;
    }

    // ===== Variant 2: WITH instrument occlusion (alt) ===================
    // Only when the caller supplied a usable instrument mask. Fills
    // slot.tex_alt + alt-suffixed metric fields. Setting has_alt = true
    // is what enables the F9 checkbox.
    if (has_inst_input) {
        std::vector<uint8_t> hitmap_alt, tmask_alt;
        int gw_alt = 0, gh_alt = 0;
        float rim_loss_alt = 0.0f;
        std::vector<uint8_t> rim_mask_alt;
        const float iou_alt = CmaesRefineV3RS::rasterize_iou2d_v3rs(
            positions, liver_indices_filtered, mvp,
            dist_map, imgW, imgH, step,
            &hitmap_alt, &tmask_alt, &gw_alt, &gh_alt,
            /*out_step1*/ nullptr, /*out_step2*/ nullptr, /*out_step3*/ nullptr,
            raster_mode,
            instrument_dist_map,
            instrument_thresh_px,
            /*out_outside_ratio*/ nullptr,
            want_rim_row ? &rim_loss_alt : nullptr,
            rim_sil_max_px,
            anat_in,
            want_rim_row ? &rim_mask_alt : nullptr);

        if (gw_alt > 0 && gh_alt > 0) {
            int inter_alt = 0, uni_alt = 0, src_alt = 0, tgt_alt = 0;
            const size_t N_alt = (size_t)gw_alt * (size_t)gh_alt;
            for (size_t i = 0; i < N_alt; ++i) {
                const bool s = (hitmap_alt[i] != 0);
                const bool t = (tmask_alt[i]  != 0);
                if (s) ++src_alt;
                if (t) ++tgt_alt;
                if (s || t) ++uni_alt;
                if (s && t) ++inter_alt;
            }

            // Rim breakdown (occ enabled → re-skip occluded cells
            // defensively, matching the legacy single-pass code).
            float rim_sil_loss_a = rim_loss_alt;
            int   rim_count_a    = 0;
            int   rim_outside_a  = 0;
            int   rim_near_a     = 0;
            int   rim_inside_a   = 0;
            if (want_rim_row && !rim_mask_alt.empty()) {
                const float* occ_data  = instrument_dist_map->data();
                const float* dist_data = dist_map.data();
                for (int gy = 0; gy < gh_alt; ++gy) {
                    const size_t row_grd = (size_t)gy * (size_t)gw_alt;
                    const int ipy = gy * step + step / 2;
                    const int my  = (ipy < 0) ? 0
                                             : (ipy >= imgH ? imgH - 1 : ipy);
                    const size_t row_img = (size_t)my * (size_t)imgW;
                    for (int gx = 0; gx < gw_alt; ++gx) {
                        const size_t idx = row_grd + (size_t)gx;
                        if (!rim_mask_alt[idx]) continue;
                        const int ipx = gx * step + step / 2;
                        const int mx  = (ipx < 0) ? 0
                                                 : (ipx >= imgW ? imgW - 1 : ipx);
                        const float idist = occ_data[row_img + (size_t)mx];
                        if (idist < instrument_thresh_px) continue;
                        const float d = dist_data[row_img + (size_t)mx];
                        if (d >= 9000.0f) ++rim_outside_a;
                        else if (d <= 5.0f) ++rim_near_a;
                        else                ++rim_inside_a;
                        ++rim_count_a;
                    }
                }
            }

            std::vector<uint8_t> rgb_alt;
            int trip_w_alt = 0, trip_h_alt = 0;
            buildComposite(hitmap_alt, tmask_alt, gw_alt, gh_alt,
                           rgb_alt, &trip_w_alt, &trip_h_alt,
                           want_rim_row ? &dist_map : nullptr,
                           imgW, imgH, step, rim_sil_max_px,
                           instrument_dist_map, instrument_thresh_px,
                           want_rim_row ? &rim_mask_alt : nullptr);

            uploadTextureRGB(slot.tex_alt, rgb_alt, trip_w_alt, trip_h_alt);
            slot.iou_alt              = iou_alt;
            slot.inter_alt_px         = inter_alt;
            slot.union_alt_px         = uni_alt;
            slot.source_alt_px        = src_alt;
            slot.target_alt_px        = tgt_alt;
            slot.rim_sil_loss_alt     = rim_sil_loss_a;
            slot.rim_count_alt        = rim_count_a;
            slot.rim_outside_count_alt = rim_outside_a;
            slot.rim_near_count_alt    = rim_near_a;
            slot.rim_inside_count_alt  = rim_inside_a;
            slot.has_alt              = true;
            // Caller-facing return: when an instrument mask was
            // supplied the "meaningful" IoU is the occluded one, so
            // the existing callers (which pass instrument args to
            // request occlusion-aware scoring) keep getting that
            // number from this function's return value — preserves
            // the pre-feature contract.
            ret_iou = iou_alt;
        }
    }

    return ret_iou;
}

// -----------------------------------------------------------------------------
//  Public: capture a single Run slot (0..9).
//
//  Optional instrument occlusion args are forwarded as-is to captureImpl.
//  Default (null map) preserves the pre-feature behaviour byte-for-byte.
// -----------------------------------------------------------------------------
inline float capture(State& st, int run_idx,
                     const mCutMesh* liver,
                     const std::vector<uint32_t>& indices,
                     const glm::mat4& view_mat, const glm::mat4& proj_mat,
                     const std::vector<float>& dist_map,
                     int imgW, int imgH, int step,
                     float scale_value,
                     const std::vector<float>* instrument_dist_map = nullptr,
                     float instrument_thresh_px = 0.0f,
                     // [NEW V3RS-VIZ] Forward to captureImpl. Set from
                     // the V3RS wrapper using p.rim_sil_max_px when the
                     // RIM diagnostic bottom row is desired. 0 (default)
                     // → legacy 3-panel layout for all other callers.
                     float rim_sil_max_px = 0.0f,
                     // [NEW V3RS-RIM-ANAT] Forward to captureImpl. When
                     // non-null and size-matched, switches the rim panels
                     // to ANATOMIC mode. Default null → legacy raster-
                     // boundary mode.
                     const std::vector<uint8_t>* is_rim_anatomic_per_vertex = nullptr)
{
    if (run_idx < 0 || run_idx >= kNumRuns) return 0.0f;
    return captureImpl(st, run_idx, liver, indices,
                       view_mat, proj_mat, dist_map,
                       imgW, imgH, step, scale_value,
                       /*raster_mode=*/0,
                       instrument_dist_map,
                       instrument_thresh_px,
                       rim_sil_max_px,
                       is_rim_anatomic_per_vertex);
}

// -----------------------------------------------------------------------------
//  Public: capture the Final pose slot. Also sets bestRunIdx and bumps
//  captureSessionId so the UI knows which Run got applied.
//
//  Optional instrument occlusion args are forwarded as-is to captureImpl.
//  Default (null map) preserves the pre-feature behaviour byte-for-byte.
// -----------------------------------------------------------------------------
inline float captureFinal(State& st, int best_run_idx,
                          const mCutMesh* liver,
                          const std::vector<uint32_t>& indices,
                          const glm::mat4& view_mat, const glm::mat4& proj_mat,
                          const std::vector<float>& dist_map,
                          int imgW, int imgH, int step,
                          float scale_value,
                          const std::vector<float>* instrument_dist_map = nullptr,
                          float instrument_thresh_px = 0.0f,
                          // [NEW V3RS-VIZ] See capture() for semantics.
                          float rim_sil_max_px = 0.0f,
                          // [NEW V3RS-RIM-ANAT] See capture() for semantics.
                          const std::vector<uint8_t>* is_rim_anatomic_per_vertex = nullptr)
{
    st.bestRunIdx = best_run_idx;
    ++st.captureSessionId;
    // Default the selector to the Final slot so the user sees the
    // actually-applied result first.
    st.currentSlot = kFinalSlot;
    return captureImpl(st, kFinalSlot, liver, indices,
                       view_mat, proj_mat, dist_map,
                       imgW, imgH, step, scale_value,
                       /*raster_mode=*/0,
                       instrument_dist_map,
                       instrument_thresh_px,
                       rim_sil_max_px,
                       is_rim_anatomic_per_vertex);
}

// -----------------------------------------------------------------------------
//  Public: F10 vertex-squash diagnostic capture. Rasterizes the CURRENT
//  static pose TWICE -- once with the triangle-bbox splat (hot path,
//  -> kDiagBBoxSlot) and once with the plain vertex-squash 3x3 raster
//  (-> kDiagVtxSlot) -- so the F9 window can flip between them with the
//  checkbox without re-capturing. Sets hasDiag, jumps the F9 selector
//  to the Diagnostic entry, and bumps captureSessionId.
//
//  Returns the bbox-splat IoU (the hot-path-equivalent number). The
//  per-cell hole / write-count / edge-length comparison is printed to
//  the console by CmaesRefineV3RS::diagnoseVertexSquashOnce, which the
//  F10 wrapper calls alongside this.
// -----------------------------------------------------------------------------
inline float captureDiag(State& st,
                         const mCutMesh* liver,
                         const std::vector<uint32_t>& indices,
                         const glm::mat4& view_mat, const glm::mat4& proj_mat,
                         const std::vector<float>& dist_map,
                         int imgW, int imgH, int step,
                         float scale_value)
{
    const float iou_bbox = captureImpl(st, kDiagBBoxSlot, liver, indices,
                                       view_mat, proj_mat, dist_map,
                                       imgW, imgH, step, scale_value,
                                       /*raster_mode=*/0);
    captureImpl(st, kDiagVtxSlot, liver, indices,
                view_mat, proj_mat, dist_map,
                imgW, imgH, step, scale_value,
                /*raster_mode=*/1);
    st.hasDiag     = true;
    st.currentSlot = kDiagComboIdx;   // jump the F9 selector to Diagnostic
    ++st.captureSessionId;
    return iou_bbox;
}

// -----------------------------------------------------------------------------
//  Clear the Run + Final slots (call from session start if you want a
//  clean slate; optional, since captures overwrite). Does NOT touch the
//  F10 diagnostic slots -- those are driven independently of any
//  Ctrl+Shift+G session and should survive one. Does NOT free GL
//  textures so the IDs are reused next session.
// -----------------------------------------------------------------------------
inline void reset(State& st) {
    for (int i = 0; i <= kFinalSlot; ++i) {
        st.slots[i].has_data  = false;
        st.slots[i].iou       = 0.0f;
        st.slots[i].scale     = 1.0f;
        st.slots[i].inter_px  = 0;
        st.slots[i].union_px  = 0;
        st.slots[i].source_px = 0;
        st.slots[i].target_px = 0;
        // [NEW V3RS-OCC-VIZ] Clear alt variant metadata. The GL texture
        // handles (tex / tex_alt) are intentionally preserved across
        // resets — they're reused by the next uploadTextureRGB call,
        // same lifecycle as slot.tex.
        st.slots[i].has_alt        = false;
        st.slots[i].iou_alt        = 0.0f;
        st.slots[i].inter_alt_px   = 0;
        st.slots[i].union_alt_px   = 0;
        st.slots[i].source_alt_px  = 0;
        st.slots[i].target_alt_px  = 0;
    }
    st.bestRunIdx = -1;
}

// -----------------------------------------------------------------------------
//  ImGui preview window. Call once per frame in main.cpp inside the
//  ImGui frame, after all other windows. Honors `st.showWindow`.
// -----------------------------------------------------------------------------
// =====================================================================
// [COMPARE] One row of the side-by-side view: a method's triptych
// ([Source | Target | Overlay]) drawn aspect-correct and capped to
// row_h_budget, with a coloured title + its squash IoU above it.
// =====================================================================
inline void drawCompareRow(const RunSlot& slot, const char* title,
                           const ImVec4& title_col, float row_h_budget) {
    ImGui::PushID(title);
    ImGui::TextColored(title_col, "%s", title);
    ImGui::SameLine();
    if (!slot.has_data || slot.tex == 0) {
        ImGui::TextDisabled("  (no data -- run the checkpoint)");
        ImGui::PopID();
        return;
    }
    ImGui::Text("  squash IoU = %.4f   |   inter %d / union %d px",
                slot.iou, slot.inter_px, slot.union_px);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    const int tex_h_px = slot.has_rim_row ? (2 * slot.h + kPanelSep) : slot.h;
    float iw = avail.x;
    float ih = (slot.w > 0) ? iw * (float)tex_h_px / (float)slot.w : row_h_budget;
    if (ih > row_h_budget) {
        ih = row_h_budget;
        iw = (tex_h_px > 0) ? ih * (float)slot.w / (float)tex_h_px : avail.x;
    }
    float off = (avail.x - iw) * 0.5f;
    if (off < 0) off = 0;

    // per-panel column headers (Source | Target | Overlay), same idiom as
    // the single-slot view so the labels line up over each third.
    const float scale_disp = (slot.w > 0) ? (iw / (float)slot.w) : 1.0f;
    const float panel_px   = scale_disp *
        (float)((slot.w - (kPanelCount + 1) * kPanelSep) / kPanelCount);
    const float sep_px     = scale_disp * (float)kPanelSep;
    const float region_x   = ImGui::GetCursorPosX() + off;
    auto drawHeader = [&](float left_local, const char* txt) {
        float tw = ImGui::CalcTextSize(txt).x;
        float cx = left_local + (panel_px - tw) * 0.5f;
        if (cx < left_local) cx = left_local;
        ImGui::SetCursorPosX(cx);
        ImGui::TextUnformatted(txt);
    };
    drawHeader(region_x + sep_px, "Source");
    ImGui::SameLine();
    drawHeader(region_x + sep_px + panel_px + sep_px, "Target");
    ImGui::SameLine();
    drawHeader(region_x + sep_px + (panel_px + sep_px) * 2.0f, "Overlay");

    if (off > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
    ImGui::Image((ImTextureID)(intptr_t)slot.tex, ImVec2(iw, ih));
    ImGui::PopID();
}

// =====================================================================
// [COMPARE] The full side-by-side body: header + delta + two stacked
// rows (A on top, B below) + the GREEN/RED/BLUE legend. Rendered in
// place of the single-slot view when st.compareMode is on.
// =====================================================================
inline void drawCompareBody(State& st) {
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f),
                       "Checkpoint compare  (same start pose, same squash yardstick)");
    ImGui::TextDisabled("Top = A, bottom = B.  Full-mesh IoU prints to the console "
                        "([Shift+E] / [Ctrl+Shift+G]).");
    ImGui::Separator();

    const RunSlot& A = st.slots[std::clamp(st.compareSlotA, 0, kNumSlots - 1)];
    const RunSlot& B = st.slots[std::clamp(st.compareSlotB, 0, kNumSlots - 1)];

    if (A.has_data && B.has_data) {
        ImGui::Text("squash IoU:   A = %.4f    B = %.4f    (B - A = %+.4f)",
                    A.iou, B.iou, B.iou - A.iou);
        ImGui::Separator();
    }

    float avail_y = ImGui::GetContentRegionAvail().y;
    float row_h   = std::max(80.0f, (avail_y - 90.0f) * 0.5f);

    drawCompareRow(A, st.compareLabelA.empty() ? "A" : st.compareLabelA.c_str(),
                   ImVec4(0.5f, 1.0f, 0.5f, 1.0f), row_h);
    ImGui::Spacing();
    drawCompareRow(B, st.compareLabelB.empty() ? "B" : st.compareLabelB.c_str(),
                   ImVec4(1.0f, 0.8f, 0.4f, 1.0f), row_h);

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.30f, 1.0f), "[GREEN]");
    ImGui::SameLine(); ImGui::Text("match");
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.85f, 0.30f, 0.30f, 1.0f), "  [RED]");
    ImGui::SameLine(); ImGui::Text("target only (miss)");
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.30f, 0.40f, 0.95f, 1.0f), "  [BLUE]");
    ImGui::SameLine(); ImGui::Text("source only (overshoot)");
}

inline void drawPreviewWindow(State& st, float viewportW, float windowH) {
    if (!st.showWindow) return;

    // [FIXED V3RS-VIZ] Bumped default window size from 960x380 to 1100x720
    // to fit the 2-row 6-panel layout without forcing the user to resize
    // on first open. ImGuiCond_FirstUseEver means user-resizing still
    // persists across frames; only the first-ever-open uses this default.
    ImGui::SetNextWindowSize(ImVec2(1100.0f, 720.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(
        ImVec2(viewportW * 0.5f, windowH * 0.5f),
        ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.04f, 0.04f, 0.06f, 0.95f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));

    if (ImGui::Begin("V3RS Silhouette IoU [F9]", &st.showWindow,
                     ImGuiWindowFlags_NoCollapse))
    {
        // [COMPARE] Side-by-side A|B view. When ON we render two stacked
        // triptychs and skip the single-slot combo path entirely. Set by
        // the SilCompare "Checkpoint" button, or toggled manually here.
        ImGui::Checkbox("Compare A|B side-by-side", &st.compareMode);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Show two captured slots stacked for direct mask\n"
                              "comparison (top = A, bottom = B). Driven by the\n"
                              "'Checkpoint: Alt+P vs Ctrl+I' button, or pick\n"
                              "slots manually below (0-based: 0..9 = Run 1..10).");
        if (st.compareMode) {
            int a = st.compareSlotA, b = st.compareSlotB;
            ImGui::SetNextItemWidth(90);
            if (ImGui::InputInt("A slot", &a)) st.compareSlotA = std::clamp(a, 0, kNumSlots - 1);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90);
            if (ImGui::InputInt("B slot", &b)) st.compareSlotB = std::clamp(b, 0, kNumSlots - 1);
            drawCompareBody(st);
            ImGui::End();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
            return;
        }

        // Slot selector: "Run 1" ... "Run 10", "Final", "Diagnostic (F10)".
        std::string label_storage[kNumComboEntries];
        const char* labels[kNumComboEntries];
        for (int i = 0; i < kNumRuns; ++i) {
            std::ostringstream oss;
            oss << "Run " << (i + 1);
            if (st.bestRunIdx == i) oss << "  [WINNER]";
            if (!st.slots[i].has_data) oss << "  (no data)";
            label_storage[i] = oss.str();
            labels[i] = label_storage[i].c_str();
        }
        {
            std::ostringstream oss;
            oss << "Final";
            if (st.bestRunIdx >= 0 && st.bestRunIdx < kNumRuns) {
                oss << "  (= Run " << (st.bestRunIdx + 1) << ")";
            }
            if (!st.slots[kFinalSlot].has_data) oss << "  (no data)";
            label_storage[kFinalSlot] = oss.str();
            labels[kFinalSlot] = label_storage[kFinalSlot].c_str();
        }
        {
            std::ostringstream oss;
            oss << "Diagnostic (F10)";
            if (!st.hasDiag) oss << "  (no data)";
            label_storage[kDiagComboIdx] = oss.str();
            labels[kDiagComboIdx] = label_storage[kDiagComboIdx].c_str();
        }

        int sel = std::clamp(st.currentSlot, 0, kNumComboEntries - 1);
        if (ImGui::Combo("##slotsel", &sel, labels, kNumComboEntries)) {
            st.currentSlot = sel;
        }

        // The Diagnostic combo entry maps to one of the two diag slots;
        // the checkbox picks which (bbox splat vs. plain vertex squash).
        // Run/Final entries map 1:1 to their slot and ignore the checkbox.
        const bool on_diag = (st.currentSlot == kDiagComboIdx);
        if (on_diag) {
            ImGui::Checkbox("Vtx-squash raster (vs. bbox splat)",
                            &st.showVtxSquash);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "OFF = triangle-bbox splat (the current hot path)\n"
                    "ON  = plain vertex-squash 3x3 (diagnostic)\n"
                    "Both are captured by F10 at the current static pose.\n"
                    "See the [V3RS/vsq-diag] console block for the\n"
                    "hole / write-count / edge-length numbers.");
            }
        }

        int actualSlot;
        if (st.currentSlot < kDiagComboIdx) {
            actualSlot = st.currentSlot;            // Run 0..9 or Final
        } else {
            actualSlot = st.showVtxSquash ? kDiagVtxSlot : kDiagBBoxSlot;
        }

        const RunSlot& slot = st.slots[actualSlot];
        if (!slot.has_data || slot.tex == 0) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                               on_diag
                                 ? "No diagnostic data yet. Press F10 to capture."
                                 : "No data for this slot yet. Run Ctrl+Shift+G to capture.");
        } else {
            // [NEW V3RS-OCC-VIZ] Instrument occlusion display toggle.
            // Visible only when this slot was captured with both
            // variants (has_alt). Otherwise we still show a disabled
            // checkbox as a hint that the option exists but the data
            // wasn't supplied.
            if (slot.has_alt) {
                ImGui::Checkbox("Apply instrument occlusion",
                                &st.showWithOcc);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "ON  = composite + IoU with instrument-occluded\n"
                        "      cells cleared (matches V3RS cost function\n"
                        "      and Ctrl+G PoseLibrary IoU_occ).\n"
                        "OFF = raw composite + IoU without occlusion.\n"
                        "Both variants are rasterised at capture time;\n"
                        "toggling is free (no re-render).");
                }
            } else {
                ImGui::BeginDisabled();
                bool dummy = false;
                ImGui::Checkbox(
                    "Apply instrument occlusion (no mask captured)",
                    &dummy);
                ImGui::EndDisabled();
            }

            // Pick the variant to display + report.
            const bool effective_with_occ = st.showWithOcc && slot.has_alt;
            const float  iou_disp        = effective_with_occ ? slot.iou_alt          : slot.iou;
            const int    inter_disp      = effective_with_occ ? slot.inter_alt_px     : slot.inter_px;
            const int    union_disp      = effective_with_occ ? slot.union_alt_px     : slot.union_px;
            const int    src_disp        = effective_with_occ ? slot.source_alt_px    : slot.source_px;
            const int    tgt_disp        = effective_with_occ ? slot.target_alt_px    : slot.target_px;
            const float  rim_loss_disp   = effective_with_occ ? slot.rim_sil_loss_alt : slot.rim_sil_loss;
            const int    rim_count_disp  = effective_with_occ ? slot.rim_count_alt    : slot.rim_count;
            const int    rim_out_disp    = effective_with_occ ? slot.rim_outside_count_alt : slot.rim_outside_count;
            const int    rim_near_disp   = effective_with_occ ? slot.rim_near_count_alt    : slot.rim_near_count;
            const int    rim_in_disp     = effective_with_occ ? slot.rim_inside_count_alt  : slot.rim_inside_count;

            // [NEW V3RS-CONTAIN] precision / recall, with 0-guard.
            const float precision_disp = (src_disp > 0)
                ? (float)inter_disp / (float)src_disp : 0.0f;
            const float recall_disp    = (tgt_disp > 0)
                ? (float)inter_disp / (float)tgt_disp : 0.0f;
            // [NEW V3RS-CONTAIN-RATIO] size_ratio = |source| / |target|.
            // Identical information to (recall / precision) but read off as
            // "source is N× the size of target": 1.0 = same size, >1 = src
            // oversized (overshoot), <1 = src undersized (undershoot). The
            // primary indicator the operator wanted, because recall
            // saturates at 1.0 and hides "how much" the source overshoots
            // even when it dramatically engulfs the target.
            //
            // overshoot_fraction = |source - intersection| / |target|
            //   = (1 - precision) / precision * recall
            //   = recall / precision - recall
            // Reads as "the blue (source-only) region is NN% the size of
            // the target". 0 = no overshoot; can exceed 1 when source is
            // much bigger than target.
            const float size_ratio_disp = (tgt_disp > 0)
                ? (float)src_disp / (float)tgt_disp : 0.0f;
            const float overshoot_frac_disp = (tgt_disp > 0)
                ? (float)std::max(0, src_disp - inter_disp) / (float)tgt_disp
                : 0.0f;

            // IoU label gets a short annotation so the user always
            // knows which variant they're looking at, even with the
            // checkbox visually adjacent.
            const char* iou_tag =
                slot.has_alt
                    ? (effective_with_occ ? "  [occ ON]"
                                          : "  [occ OFF]")
                    : "";
            ImGui::Text("IoU = %.4f%s", iou_disp, iou_tag);
            ImGui::SameLine();
            ImGui::Text("  |  scale = %.4f", slot.scale);
            ImGui::Text("intersection = %d px   union = %d px",
                        inter_disp, union_disp);
            // [NEW V3RS-CONTAIN] Containment row: precision / recall +
            // a one-glance text tag for which direction the mismatch
            // leans. epsilon=0.05 keeps trivial differences quiet.
            const float dir = recall_disp - precision_disp;
            const char* contain_tag =
                (std::fabs(dir) < 0.05f) ? "balanced"
                    : (dir > 0.0f)       ? "overshoot (src > tgt)"
                                         : "undershoot (src < tgt)";
            ImVec4 contain_color =
                (std::fabs(dir) < 0.05f) ? ImVec4(0.50f, 0.85f, 0.50f, 1.0f)
                    : (dir > 0.0f)       ? ImVec4(0.85f, 0.55f, 0.45f, 1.0f)
                                         : ImVec4(0.55f, 0.65f, 0.95f, 1.0f);
            // [NEW V3RS-CONTAIN-RATIO] Headline row: size_ratio +
            // overshoot_fraction. These are the user-facing primary
            // indicators because they retain magnitude info that
            // recall (saturated at 1) hides. Direction tag uses the
            // same recall-vs-precision delta as before.
            ImGui::TextColored(
                contain_color,
                "size = %.2fx   overshoot = %.0f%%   -> %s",
                size_ratio_disp,
                100.0f * overshoot_frac_disp,
                contain_tag);
            // Reference row: keep recall / precision for breakdown.
            // De-emphasised (greyer) since the headline carries the
            // primary signal.
            ImGui::TextColored(
                ImVec4(0.70f, 0.70f, 0.70f, 1.0f),
                "  (recall = %.3f   precision = %.3f)",
                recall_disp, precision_disp);
            // [NEW V3RS-VIZ] Rim metrics line. Shown only when the rim row
            // was actually built into this slot's texture; we don't want to
            // claim a 0.0 rim_sil_loss on slots that were captured before
            // the rim feature was enabled.
            if (slot.has_rim_row) {
                ImGui::TextColored(
                    ImVec4(0.85f, 0.75f, 0.50f, 1.0f),
                    "rim_sil_loss = %.3f   (rim cells: %d total = %d outside + "
                    "%d near + %d inside)  max_px=%.0f",
                    rim_loss_disp,
                    rim_count_disp,
                    rim_out_disp,
                    rim_near_disp,
                    rim_in_disp,
                    slot.rim_sil_max_px);
            }
            ImGui::Separator();

            // The texture is either a 3-panel triptych (legacy) or a 2x3
            // grid (rim row included). The texture height tells us which
            // layout we have (slot.h matches the grid height gh; the actual
            // pixel height of the texture is 2*gh+sep when rim row is on).
            // Compute display size preserving the full aspect ratio.
            ImVec2 avail = ImGui::GetContentRegionAvail();
            const float legend_h = slot.has_rim_row ? 48.0f : 24.0f;
            const float header_h = slot.has_rim_row ? 40.0f : 20.0f;
            float region_h = std::max(48.0f, avail.y - legend_h - header_h);
            // Effective texture pixel height: 2*gh + sep when rim row on.
            const int tex_h_px = slot.has_rim_row
                                 ? (2 * slot.h + kPanelSep)
                                 : slot.h;
            float iw = avail.x;
            float ih = (slot.w > 0)
                           ? iw * (float)tex_h_px / (float)slot.w
                           : region_h;
            if (ih > region_h) {
                ih = region_h;
                iw = (tex_h_px > 0) ? ih * (float)slot.w / (float)tex_h_px : avail.x;
            }
            float off = (avail.x - iw) * 0.5f;
            if (off < 0) off = 0;

            // Per-panel display width (matches the per-panel pixel width
            // up to the same uniform scale factor we apply to the image).
            const float scale_disp =
                (slot.w > 0) ? (iw / (float)slot.w) : 1.0f;
            const float panel_px = scale_disp *
                                   (float)((slot.w - (kPanelCount + 1) * kPanelSep) / kPanelCount);
            const float sep_px   = scale_disp * (float)kPanelSep;

            // Headers: "Source" / "Target" / "Overlay" each centred above
            // the corresponding panel column. Implemented as three
            // separate Selectable-free TextUnformatted calls; SameLine(x)
            // with absolute x-position is the cleanest ImGui idiom that
            // survives DPI / styling. x positions are local to the
            // window's content region.
            const float region_x = ImGui::GetCursorPosX() + off;
            auto drawHeader = [&](float panel_left_local, const char* txt) {
                float tw = ImGui::CalcTextSize(txt).x;
                float cx = panel_left_local + (panel_px - tw) * 0.5f;
                if (cx < panel_left_local) cx = panel_left_local;
                ImGui::SetCursorPosX(cx);
                ImGui::TextUnformatted(txt);
            };
            // Source header.
            drawHeader(region_x + sep_px, "Source");
            // Target header on same line.
            ImGui::SameLine();
            drawHeader(region_x + sep_px + panel_px + sep_px, "Target");
            // Overlay header on same line.
            ImGui::SameLine();
            drawHeader(region_x + sep_px + (panel_px + sep_px) * 2.0f, "Overlay");

            // [NEW V3RS-VIZ] Second header row for the rim diagnostic panels.
            // Drawn only when the slot has the rim row. Placed AFTER the
            // top headers so it appears just above the image, splitting
            // visually into "top-row labels | image | bottom-row labels"
            // is awkward in ImGui, so we put both header rows above the
            // single image and rely on the legend below to clarify.
            if (slot.has_rim_row) {
                drawHeader(region_x + sep_px,                                "Rim cells");
                ImGui::SameLine();
                drawHeader(region_x + sep_px + panel_px + sep_px,             "Dist heatmap");
                ImGui::SameLine();
                drawHeader(region_x + sep_px + (panel_px + sep_px) * 2.0f,    "Rim alignment");
                ImGui::TextDisabled(
                    "  (top row: source/target overlap   |   bottom row: RIM diagnostic)");
            }

            // Image row.
            if (off > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
            // [NEW V3RS-OCC-VIZ] Switch displayed texture based on the
            // F9 occlusion checkbox. effective_with_occ is computed
            // above; tex_alt is always populated when slot.has_alt is
            // true (the upload happens unconditionally inside the
            // captureImpl Variant 2 block when the rasterize succeeds).
            const GLuint tex_disp =
                (effective_with_occ && slot.tex_alt != 0)
                    ? slot.tex_alt
                    : slot.tex;
            ImGui::Image((ImTextureID)(intptr_t)tex_disp, ImVec2(iw, ih));

            ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.30f, 1.0f), "[GREEN]");
            ImGui::SameLine(); ImGui::Text("match");
            ImGui::SameLine(); ImGui::TextColored(ImVec4(0.85f, 0.30f, 0.30f, 1.0f), "  [RED]");
            ImGui::SameLine(); ImGui::Text("target only (miss)");
            ImGui::SameLine(); ImGui::TextColored(ImVec4(0.30f, 0.40f, 0.95f, 1.0f), "  [BLUE]");
            ImGui::SameLine(); ImGui::Text("source only (overshoot)");

            // [NEW V3RS-VIZ] Bottom-row legend for the rim panels.
            if (slot.has_rim_row) {
                ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "Rim cells:");
                ImGui::SameLine();
                ImGui::Text("white=source-boundary  grey=interior");
                ImGui::SameLine();
                ImGui::TextDisabled("  |  Dist: cyan=on-boundary -> yellow -> red=far inside; ");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.30f, 1.0f), "  [GREEN]");
                ImGui::SameLine(); ImGui::Text("rim @ boundary (d<=5)");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.10f, 1.0f), "  [YELLOW]");
                ImGui::SameLine(); ImGui::Text("rim inside (medium)");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.85f, 0.30f, 0.30f, 1.0f), "  [RED]");
                ImGui::SameLine(); ImGui::Text("rim deep inside (high penalty)");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.85f, 0.30f, 0.85f, 1.0f), "  [MAGENTA]");
                ImGui::SameLine(); ImGui::Text("rim OUTSIDE target (worst)");
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

inline State g_silOverlay;   // singleton; wired into main.cpp

}  // namespace SilOverlay
