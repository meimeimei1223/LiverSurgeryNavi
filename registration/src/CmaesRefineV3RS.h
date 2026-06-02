#pragma once
#ifndef CMAES_REFINE_V3RS_H
#define CMAES_REFINE_V3RS_H
/*
 * CmaesRefineV3RS.h
 * ----------------------------------------------------------------------
 * V3R-S: Silhouette-anchor variant of V3R. Bound to Ctrl+Shift+G.
 *
 * Design contract (HANDOVER V6 + this chat):
 *   - V3R (Ctrl+G, header CmaesRefineV3R.h) is FROZEN. This file does
 *     NOT modify V3R; it only #includes it and calls its public helpers.
 *     V3R-W (rim weighting + AR-vis + Caudal-only) behaviour is
 *     preserved bit-for-bit at the Ctrl+G entry point.
 *   - ParamsV3RS inherits from ParamsV3R and adds silhouette fields.
 *   - evaluate_one_v3rs_silhouette calls V3R's evaluate_one_v3r_weighted
 *     internally for the RMSE_W term, then adds lambda_sil * sil_loss.
 *     There is NO duplicated implementation of the rim-weighted
 *     accumulator -- V3R is the single source of truth.
 *   - run_one_bipop_v3rs and runBipopCmaesV3RS are STRAIGHT FORKS of
 *     V3R's run_one_bipop_v3r and runBipopCmaesV3R with [V3RS-DIFF]
 *     markers at the (small) intentional differences:
 *       * params type ParamsV3RS&
 *       * 3-way eval dispatch (sil > weighted > plain)
 *       * tolfun override when lambda_sil > 0
 *       * per-run silhouette summary log
 *     All other code is verbatim-copy and calls back into the
 *     CmaesRefineV3R:: namespace for the session-setup helpers
 *     (build_voxel_to_orig, derive_arvis_voxel, derive_is_caudal_voxel,
 *     filter_by_quadrant_with_arvis_caudal, build_eval_context_v3r,
 *     rebuild_correspondences_v3r).
 *
 * Why fork-and-mark rather than refactor V3R:
 *   - Ctrl+G stability is non-negotiable until S6 (paper data).
 *   - The diffs are small and clearly bounded by [V3RS-DIFF] tags.
 *   - If/when V3R is refactored to take an eval policy, this file
 *     collapses to ~150 lines. Until then, the copy is the cost of
 *     keeping V3R untouched.
 *
 * Silhouette loss summary (Phase 3):
 *   cost = RMSE_W + lambda_sil * (1 - IoU2D)
 *
 *   IoU2D is the 2D Intersection-over-Union between the full-mesh
 *   triangle-bbox rasterization (Q:AR+AL quad-filtered triangles) and
 *   the SAM2 target mask (squash-ON path, symmetric source/target raster).
 *
 *   Phase 3 change from Phase 2:
 *     Phase 2 used  cost = RMSE_W + lambda * (1-IoU) * |scale-1|
 *     which was permanently zero because scale is always 1.0000.
 *     Phase 3 removes the |scale-1| multiplier so the IoU penalty is
 *     unconditional, preventing the containment failure mode where
 *     RMSE_W improves while the source silhouette expands to enclose
 *     the SAM2 target mask.
 *
 *   Scale of terms (typical operating point):
 *     RMSE_W    ≈ 0.03 – 0.05
 *     (1-IoU)   ≈ 0.35 – 0.45
 *   → lambda_sil ≈ 0.05–0.15 keeps both terms comparable.
 *     lambda_sil = 0.2 weights IoU roughly 2–3x stronger than RMSE.
 *     (RegistrationActions.h default: g_ctrlgsLambdaSil = 0.2f)
 * ----------------------------------------------------------------------
 */

#include "CmaesRefineV3R.h"       // V3R Params, helpers, evaluators reused as-is

#include <vector>
#include <cstdint>
#include <cfloat>      // FLT_MAX (z-buffer init)
#include <cmath>       // std::floor, std::sqrt (bilinear sample, diag norm)
#include <algorithm>   // std::count (sil-aware diagnostics)
#include <functional>  // std::function (Phase 0 Run-selector callback)
#include <string>      // std::string (ASCII silhouette map in cov-diag)
#include <iostream>    // std::cout (timing + coverage diagnostics)
#include <iomanip>     // std::setprecision (diagnostic formatting)
#include <chrono>      // std::chrono (per-step timing instrumentation)

#ifdef _OPENMP
#include <omp.h>     // parallel triangle-bbox splat in rasterize_iou2d_v3rs
#endif

namespace CmaesRefineV3RS {

// =====================================================================
// ParamsV3RS -- V3R params + silhouette anchor fields.
// ---------------------------------------------------------------------
// Inherits every field of CmaesRefineV3R::ParamsV3R so V3RS can reuse
// V3R's helpers (build_voxel_to_orig, filter_by_quadrant_with_arvis_caudal,
// build_eval_context_v3r, evaluate_one_v3r_weighted, ...) without any
// shim. Silhouette-only fields are added at the end.
//
// Defaults preserve "no silhouette" behaviour: when lambda_sil == 0 OR
// sil_dist_map_2d is empty, runBipopCmaesV3RS still works but the inner
// dispatch falls back to V3R-W / V3 (the user is better served by
// pressing Ctrl+G in that case; V3RS just remains correct).
// =====================================================================
struct ParamsV3RS : public CmaesRefineV3R::ParamsV3R {
    // ----- Silhouette anchor (caller-set) ---------------------------
    // lambda_sil == 0.0f -> V3R-W behaviour (no silhouette term).
    float lambda_sil = 0.0f;

    // [V3I] Pure-IoU mode (Ctrl+I). Default false => V3RS unchanged.
    //   When true the inner cost becomes  cost = (1 - IoU2D)  ONLY
    //   (RMSE_W / outside / rim_sil terms dropped), and the Run selector
    //   decides by IoU2D instead of the RMSE-blended combo. This is the
    //   "Ctrl+G mechanism, objective swapped to squash-IoU" experiment.
    //   The wrapper (RegistrationActions.h) also skips the RMSE accept
    //   cap when this is set, so an IoU-improving pose is never rejected
    //   for raising 3D RMSE. Ctrl+Shift+G leaves this false => byte-
    //   identical behaviour.
    bool  pure_iou_mode = false;

    // AR-camera pinhole intrinsics. cam_pos = (0,0,0), look-at = +Z.
    //   u = sil_fx * p.x / p.z + sil_cx
    //   v = sil_fy * p.y / p.z + sil_cy
    float sil_fx     = 0.0f;
    float sil_fy     = 0.0f;
    float sil_cx     = 0.0f;
    float sil_cy     = 0.0f;
    int   sil_img_w  = 0;
    int   sil_img_h  = 0;

    // Per-pixel distance-to-mask-boundary, row-major, size = w*h.
    // Pixel value >= 9000 is the DepthUtils sentinel ("unreachable").
    std::vector<float> sil_dist_map_2d;

    // Tolfun override when lambda_sil > 0. The default V3 tolfun (1e-4)
    // is calibrated against RMSE_W ~ 0.05; adding a sil_loss term of
    // comparable magnitude shifts the cost scale and tolfun fires at
    // Gen 0..8. Override to a tighter threshold so CMA-ES actually
    // explores. Set to <=0 to disable the override.
    float sil_tolfun_override = 1e-5f;

    // [V3RS Phase 0] Run-selector callback. Optional. When non-null,
    // runBipopCmaesV3RS calls this 10 times after the Run loop (once per
    // Run candidate), passing the world-matrix that would be applied to
    // the source mesh for that Run. The callback returns IoU2D in [0,1]
    // (higher = better overlap). Selection is NOT changed in Phase 0:
    // best_run is still chosen by argmin(rmse_full); the IoU2D values
    // are logged for diagnostic comparison only. See
    // HANDOVER_V3RS_silhouette_pivot_to_2D_IoU.md §5.1 for the full
    // design. Wrapper installs this when lambda_sil>0 and the SAM2
    // boundary distance map is valid. Kept null when sil is inactive
    // (selector skipped, no overhead). The closure typically wraps
    // CmaesUtils::computeSilhouette2DObjectiveFast against a shadow
    // application of M to liverMesh3D vertices; V3RS itself never
    // references any of the globals that work depends on, so the
    // dependency graph stays as documented in HANDOVER §9.2.
    std::function<float(const glm::mat4&)> sil_iou2d_eval_fn;

    // [NEW V3RS-SEL] Extended Selector callback. Optional. When non-null,
    // the Phase A Selector PREFERS this over sil_iou2d_eval_fn so the Run
    // comparison can include the same composite cost Layer 1 (per-eval)
    // and Layer 3 (Phase E gate) use:
    //   combo = RMSE_W
    //         + lambda_sil      * (1 - IoU_occluded)
    //         + lambda_out      * outside_ratio
    //         + lambda_rim_sil  * rim_sil_loss
    // Without this, Selector only saw IoU and would pick a Run that won
    // on (RMSE + lambda_sil*(1-IoU)) but lost on outside/rim_sil, then
    // Phase E would REJECT it because the composite hadn't improved.
    // Returns the three metrics for the pose M_world applied to the
    // pre-Phase-D backup of liverMesh3D. When this callback is null
    // (legacy / wrapper didn't install it), the Selector falls back to
    // sil_iou2d_eval_fn with outside_ratio = rim_sil_loss = 0 so the
    // behaviour is byte-identical to the pre-feature path.
    struct SelectorMetrics {
        float iou_occluded   = 0.0f;   // [0,1], higher better
        float outside_ratio  = 0.0f;   // [0,1], lower  better
        float rim_sil_loss   = 0.0f;   // [0,1], lower  better
    };
    std::function<SelectorMetrics(const glm::mat4&)> sil_metrics_eval_fn;

    // [V3RS Phase 2 diagnostic] Per-Run capture callback. Called
    // RIGHT AFTER sil_iou2d_eval_fn for the same Run, so liverMesh3D
    // is already in the just-evaluated state (no re-apply needed).
    // Receives run_idx (0..9) and the scale value of that Run's SRT
    // for UI display. Wrapper uses this to upload the GL texture for
    // the ImGui overlay AND optionally write PNG dumps.
    std::function<void(int /*run_idx*/, float /*scale*/)> sil_per_run_dump_fn;

    // ----- [V3RS Phase 1] In-loop IoU2D fields -----------------------
    // Phase 1 replaces the old rim-anchored sil_loss (which was
    // structurally broken; see HANDOVER §2) with an every-Nth-eval
    // rasterized 2D-IoU term folded into the CMA-ES cost. The wrapper
    // populates the following four fields once per Ctrl+Shift+G call;
    // V3RS uses them in run_one_bipop_v3rs to call rasterize_iou2d_v3rs
    // without touching any globals (HANDOVER §9.2, path B3).

    // Full-mesh triangle indices, copied from liverMesh3D->mIndices.
    // 3*N_tris entries (~60K for the standard 9992-vert liver mesh).
    // Empty -> Phase 1 in-loop IoU2D path falls through to RMSE_W only.
    std::vector<uint32_t> sil_indices;

    // Pre-built AR fixed-camera silhouette view & projection. Set by
    // the wrapper from buildSilhouetteView() / buildSilhouetteProj().
    // Defaults to identity so legacy callers that don't set them
    // produce a well-defined (but useless) projection.
    glm::mat4 sil_view = glm::mat4(1.0f);
    glm::mat4 sil_proj = glm::mat4(1.0f);

    // Sampling cadence for the in-loop IoU2D term. **Phase 2 default
    // is 1 (per-eval).** The Phase 1 default of 10 (10% sampling) was
    // found to be too weak to prevent scale-cheating, since the
    // optimizer could move 9 of every 10 evals without ever feeling
    // the silhouette penalty. Per-eval cost is ~3.75ms at step=8,
    // giving ~50s/session (HANDOVER §4.2). Set to 0 (or <=0) to
    // disable the in-loop IoU2D fold entirely (cost = RMSE_W only).
    int sil_eval_interval = 1;

    // Rasterizer stride. After the rasterizer switched from triangle
    // raster to vertex-splat 3x3 (CmaesRefineV3RS.h Step 2), the
    // bottleneck moved from per-triangle inner work to the gw*gh IoU
    // loop and the gw*gh hitmap allocation. step=16 keeps the IoU
    // computation meaningful (240x135 cells -> 120x68 = 8K cells,
    // boundary precision 16 image px = ~6 mm @ 1.27 m, below the
    // typical late-CMA-ES translation variance) while quartering
    // both loops. Also incidentally raises per-cell vert density
    // (1 cell at step=16 covers 4x the area of 1 cell at step=8),
    // which fills the residual sub-cell gaps that 3x3 splat leaves
    // when 2D vert density is sparse. Step is overridable from the
    // wrapper if the user wants finer/coarser sampling.
    int sil_raster_step = 16;

    // ----- Instrument occlusion mask (optional) ----------------------
    // Per-pixel distance map to the instrument-segmentation region,
    // row-major, size = sil_img_w * sil_img_h. Built upstream from
    // instrument_segmentation_mask.png via DepthUtils::g_instrumentDistMap
    // (BFS from instrument pixels; 0 inside instrument, growing outward).
    //
    // When non-empty AND its size matches sil_dist_map_2d, IoU cells
    // satisfying `instrument_dist_map[px] < sil_instrument_thresh_px`
    // are EXCLUDED from both numerator (intersection) and denominator
    // (union) of the IoU computation. This fixes the asymmetric error
    // where source mesh extends behind an instrument occluder but the
    // SAM2 target mask correctly has no liver there -- without this
    // filter, the source overshoot would be counted as IoU loss even
    // though it's physically plausible.
    //
    // When empty (default): occlusion filter OFF; IoU is byte-identical
    // to the pre-feature path. Same effect as thresh = 0 with empty map.
    //
    // Thresh semantics (pixels):
    //   thresh = 0  : exclude only cells INSIDE the instrument region
    //                 (dist == 0). Recommended starting point.
    //   thresh = 5  : also exclude cells within 5 px of an instrument
    //                 (compensates for SAM2 mask edge slop / liver-edge
    //                 segmentation noise around the tool boundary).
    //   thresh ~10  : aggressive; useful when SAM2 leaks liver pixels
    //                 onto the tool body. Risk: drops legitimate
    //                 silhouette boundary near the tool.
    std::vector<float> sil_instrument_dist_map_2d;
    float              sil_instrument_thresh_px = 0.0f;

    // ----- [NEW] Asymmetric outside-ratio penalty --------------------
    // Set by RegistrationActions.h wrapper from g_ctrlgsLambdaOut when
    // g_ctrlgsUseOutsideRatio is ON. When > 0, the inner cost adds:
    //   + lambda_out * outside_ratio
    // where outside_ratio = (source cells outside target) / (source cells).
    // This directly penalises mask expansion (source ⊃ target), which is
    // symmetric IoU's blind spot: containing-source has IoU < 1 but the
    // gradient toward shrinking is weak. Default 0 → no behaviour change.
    float lambda_out = 0.0f;

    // ----- [NEW] RIM silhouette: source-boundary cell distance penalty
    // Set by RegistrationActions.h wrapper from g_ctrlgsLambdaRimSil when
    // g_ctrlgsUseRimSil is ON. When > 0, the inner cost adds:
    //   + lambda_rim_sil * rim_sil_loss
    // where rim_sil_loss is the mean (clipped, normalised) distance from
    // each SOURCE-BOUNDARY raster cell (a source cell with at least one
    // non-source 4-neighbour) to the target silhouette boundary.
    //   outside target mask  -> contribute 1.0 (max)
    //   inside target mask   -> contribute min(d / rim_sil_max_px, 1.0)
    // This is the silhouette-space analogue of Ctrl+G's beta-rim weighting:
    // forces source rim ↔ target rim coincidence rather than mere overlap.
    // Default 0 → no behaviour change.
    float lambda_rim_sil  = 0.0f;
    float rim_sil_max_px  = 100.0f;

    // ----- [NEW V3RS-RIM-ANAT] Anatomic-mode RIM flag (per FULL vertex) -
    // When non-empty (size == liver_full_positions.size()), the inner
    // loop switches rim_sil computation to ANATOMIC mode: iterate over
    // these RIM vertices, project each, sample dist_map at projection,
    // aggregate per-vertex penalty. Otherwise (empty, the default),
    // rim_sil falls back to the per-cell raster-boundary mode.
    //
    // Populated by the RegistrationActions.h wrapper from the SAME filter
    // as the Ctrl+G "Show RIM pairs" checkbox uses:
    //   - LiverRegionLabel::RIM   (mesh-intrinsic anatomical rim)
    //   - active quadrant mask    (per the current Q:* selection)
    //   - AR-vis filter (opt-in)  (visible from the AR camera)
    //   - Caudal-only filter (opt-in) (cranio-caudal label = CAUDAL)
    // So the rim_sil "RIM" matches what the user already visualises in
    // the AR view with the existing rim-spheres checkbox.
    std::vector<uint8_t> is_rim_anatomic_full;
};

// =====================================================================
// EvalTimingV3RS -- per-Run timing accumulator for evaluate_one + raster.
// ---------------------------------------------------------------------
// Goal: split the "evaluate_one (sum)" line in the Run banner into its
// constituent parts so we can localise the real bottleneck instead of
// guessing. Filled by evaluate_one_v3rs_silhouette and propagated into
// run_one_bipop_v3rs's summary log.
//
//   eval_rmse_w_us       : V3R-W RMSE_W path inside evaluate_one
//                          (KDTree NN search; the V3 part).
//   eval_sil_total_us    : whole rasterize_iou2d_v3rs call
//                          ( = sil_proj_us + sil_splat_us + sil_iou_us
//                            + small bookkeeping inside the func).
//   eval_sil_proj_us     : Step 1 of rasterize_iou2d_v3rs
//                          (vertex projection through MVP).
//   eval_sil_splat_us    : Step 2 of rasterize_iou2d_v3rs
//                          (triangle-bbox splat into hitmap).
//   eval_sil_iou_us      : Step 3 of rasterize_iou2d_v3rs
//                          (IoU compute against target mask).
//   eval_other_us        : M_srt construction, penalty arithmetic, the
//                          gate checks. Should be tiny if measurement
//                          is correct.
//   n_iou_evals          : count of evals that actually folded IoU
//                          (per-eval default = total evals; with
//                          eval_interval>1 it's ~ evals/interval).
//
// All values in microseconds; the Run summary formats them as ms.
// Counts are int64 to be safe at the eval volume the driver runs at
// (10 Runs * 1350 evals = 13.5k; nowhere near overflow but the type
// matches the long-long counter conventions elsewhere in this file).
// =====================================================================
struct EvalTimingV3RS {
    double   eval_rmse_w_us    = 0.0;
    double   eval_sil_total_us = 0.0;
    double   eval_sil_proj_us  = 0.0;
    double   eval_sil_splat_us = 0.0;
    double   eval_sil_iou_us   = 0.0;
    double   eval_other_us     = 0.0;
    long long n_iou_evals      = 0;
    long long n_total_evals    = 0;
};

// =====================================================================
// Silhouette target-mask cache (app-wide, content-fingerprinted).
// ---------------------------------------------------------------------
// Problem this solves:
//   Source side (mesh) is rasterized with triangle-bbox splat + 1-cell
//   halo (see rasterize_iou2d_v3rs Step 2). Target side, in the legacy
//   path, was a per-grid-cell CENTRE SAMPLE of dist_map -- a totally
//   different raster system. Source got systematically inflated by the
//   bbox+halo, target didn't. The IoU therefore had an asymmetric bias
//   that capped achievable IoU and made the cost landscape biased.
//
// Fix: build the target mask through the SAME raster system as the
// source:
//   Step A (squash): for each grid cell (gw x gh), OR over the
//                    step x step window of dist_map pixels it covers.
//                    Equivalent to triangle-bbox splat at the source
//                    side: "if ANY mask pixel touches this cell, mark
//                    cell ON".
//   Step B (halo):   1-cell dilation (3x3 square, 8-connectivity).
//                    Same halo width the source bbox uses.
// Now source and target are constructed by the SAME conservative
// rasterization rule, the inflation is symmetric, and IoU measures
// agreement of equally-inflated shapes.
//
// Lifecycle: app-wide. The cache survives across Ctrl+Shift+G presses
// and is rebuilt only when (a) the source dist_map content changes
// (detected via a 5-sample content fingerprint -- this catches SAM2
// re-load even when the new mask happens to allocate at the same
// std::vector buffer address), or (b) any of step/img_w/img_h change.
//
// Why content fingerprint, not pointer-based:
//   The wrapper (RegistrationActions.h) value-copies g_boundaryDistMap
//   into ParamsV3RS::sil_dist_map_2d every press. The copies have
//   different addresses but identical content. A pointer-based
//   fingerprint would miss the cache every press (~5 ms wasted build);
//   a content-based fingerprint hits on the COPY too. Sample 5 floats
//   from deterministic positions -- collision probability of two
//   unrelated masks matching at all 5 sample points is ~1 in 2^160.
//
// Why this is preferred over invalidate-from-DepthUtils.h:
//   DepthUtils.h is the mask PRODUCER; V3RS is the CONSUMER. Pushing
//   invalidation from producer to consumer reverses the dependency
//   direction. The consumer-side pull-based fingerprint check keeps
//   DepthUtils.h ignorant of V3RS, no header coupling.
//
// Toggle: g_silTargetSquashEnabled (default ON). When OFF, the rasterize
// function falls back to legacy centre-sample. Useful for A/B
// comparisons in the F9 overlay.
// =====================================================================
inline bool g_silTargetSquashEnabled = true;

// =====================================================================
// [V3RS-PARALLEL] Run-loop parallelization toggle.
//
// When true, runBipopCmaesV3RS executes its N_STARTS independent BIPOP
// restarts CONCURRENTLY -- one OpenMP thread per run, each run doing a
// SERIAL raster (the inner per-eval raster parallelism is suppressed via
// omp_in_parallel() so we don't pay per-thread-hitmap alloc/reduce inside
// an already-parallel region). Default OFF = the legacy serial run loop,
// which is byte-identical to the pre-change behavior.
//
// Determinism is preserved in BOTH modes: per-run jitter and sigma0 are
// pre-generated from the outer RNG in run order (the exact 8-draw-per-run
// sequence the legacy in-loop code used) BEFORE the loop, so the RNG
// consumption is identical regardless of run execution order, and each
// run's CMA-ES is seeded by cma_base+run (RNG-independent). Result poses,
// IoU, best-run selection, and the accept decision are therefore
// identical to serial; only the wall-clock and the ordering of verbose
// per-run log lines differ (in parallel mode the per-run logs are
// suppressed during the loop and reprinted in run order afterward).
//
// Read once at the top of the run loop. Set from the UI (Debug Panel)
// via CmaesRefineV3RS::g_v3rsParallelRuns. Applies to both Ctrl+I
// (pure-IoU) and Ctrl+G since they share this engine.
inline bool g_v3rsParallelRuns = false;
// Filled by runBipopCmaesV3RS so the UI can show what the last run did.
inline int  g_v3rsLastRunThreads = 1;     // threads actually used last time
inline double g_v3rsLastRunLoopMs = 0.0;  // run-loop wall-clock last time

struct SilTargetMaskCache {
    std::vector<uint8_t> data;   // gw * gh, squashed + 1-cell halo
    int gw    = 0;
    int gh    = 0;
    int step  = 0;
    int img_w = 0;
    int img_h = 0;

    // Content fingerprint of the source dist_map.
    size_t src_n = 0;
    float  fp[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    bool empty() const { return data.empty() || gw <= 0 || gh <= 0; }
};

// Single app-wide cache instance. Inline to avoid header ODR issues.
inline SilTargetMaskCache g_silTargetMaskCache;

// Sample 5 values from a deterministic distribution across the buffer.
// Bit-exact: dist_map is computed by deterministic BFS, so the same
// underlying mask produces the same fingerprint every time.
inline void silMaskFingerprint(const std::vector<float>& dist_map,
                               float out_fp[5])
{
    const size_t n = dist_map.size();
    if (n == 0) {
        out_fp[0] = out_fp[1] = out_fp[2] = out_fp[3] = out_fp[4] = 0.0f;
        return;
    }
    if (n == 1) {
        out_fp[0] = out_fp[1] = out_fp[2] = out_fp[3] = out_fp[4] = dist_map[0];
        return;
    }
    out_fp[0] = dist_map[0];
    out_fp[1] = dist_map[n / 4];
    out_fp[2] = dist_map[n / 2];
    out_fp[3] = dist_map[(3 * n) / 4];
    out_fp[4] = dist_map[n - 1];
}

inline bool silFingerprintsEqual(const float a[5], const float b[5]) {
    // Bit-exact equality. Mask values are produced by deterministic
    // integer BFS distances stored as floats, so NaN / -0 don't appear.
    for (int i = 0; i < 5; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Returns true if cache is valid for the given (dist_map, img_w, img_h, step).
// Cheap (5 reads + 5 compares). Safe to call every rasterize_iou2d_v3rs eval.
inline bool isSilTargetMaskCacheValidFor(const std::vector<float>& dist_map,
                                         int img_w, int img_h, int step)
{
    const SilTargetMaskCache& c = g_silTargetMaskCache;
    if (c.empty())                                  return false;
    if (c.step  != step)                            return false;
    if (c.img_w != img_w || c.img_h != img_h)       return false;
    if (c.src_n != dist_map.size())                 return false;
    float fp_now[5];
    silMaskFingerprint(dist_map, fp_now);
    return silFingerprintsEqual(c.fp, fp_now);
}

// Builder. Idempotent: hits the cache when nothing changed.
// Cost on miss: ~5 ms for 1920x1080, step=16 (single-threaded squash
// + dilation). Returns true on success or hit, false on bad input.
inline bool ensureSilTargetMaskCache(const std::vector<float>& dist_map,
                                     int img_w, int img_h, int step)
{
    if (dist_map.empty() || img_w <= 0 || img_h <= 0 || step < 1) {
        g_silTargetMaskCache = SilTargetMaskCache{};
        return false;
    }
    if (dist_map.size() != (size_t)img_w * (size_t)img_h) {
        g_silTargetMaskCache = SilTargetMaskCache{};
        return false;
    }

    if (isSilTargetMaskCacheValidFor(dist_map, img_w, img_h, step)) {
        // Hit. No log: this is the common path (every Ctrl+Shift+G
        // press after the first, until the SAM2 mask is reloaded).
        return true;
    }

    // ----- Miss: (re)build ----------------------------------------
    SilTargetMaskCache c;
    c.gw    = (img_w + step - 1) / step;
    c.gh    = (img_h + step - 1) / step;
    c.step  = step;
    c.img_w = img_w;
    c.img_h = img_h;
    c.src_n = dist_map.size();
    silMaskFingerprint(dist_map, c.fp);

    if (c.gw <= 0 || c.gh <= 0) {
        g_silTargetMaskCache = SilTargetMaskCache{};
        return false;
    }

    // Step A: step x step OR coverage per cell -- matches the
    // semantics of the source-side triangle-bbox splat (which marks
    // any cell that the triangle's bbox touches).
    std::vector<uint8_t> raw((size_t)c.gw * (size_t)c.gh, 0);
    for (int gy = 0; gy < c.gh; ++gy) {
        const int py0 = gy * step;
        const int py1 = std::min(img_h, py0 + step);
        const size_t row_out = (size_t)gy * (size_t)c.gw;
        for (int gx = 0; gx < c.gw; ++gx) {
            const int px0 = gx * step;
            const int px1 = std::min(img_w, px0 + step);
            bool on = false;
            for (int py = py0; py < py1 && !on; ++py) {
                const size_t row_in = (size_t)py * (size_t)img_w;
                for (int px = px0; px < px1; ++px) {
                    if (dist_map[row_in + (size_t)px] < 9000.0f) {
                        on = true;
                        break;
                    }
                }
            }
            raw[row_out + (size_t)gx] = on ? (uint8_t)1 : (uint8_t)0;
        }
    }

    // Step B: 1-cell halo dilation (3x3 square, 8-connectivity).
    // Matches the +/- 1 cell halo the source-side bbox splat uses.
    c.data.assign((size_t)c.gw * (size_t)c.gh, 0);
    for (int gy = 0; gy < c.gh; ++gy) {
        const int y0 = std::max(0,        gy - 1);
        const int y1 = std::min(c.gh - 1, gy + 1);
        for (int gx = 0; gx < c.gw; ++gx) {
            const int x0 = std::max(0,        gx - 1);
            const int x1 = std::min(c.gw - 1, gx + 1);
            bool on = false;
            for (int y = y0; y <= y1 && !on; ++y) {
                const size_t row = (size_t)y * (size_t)c.gw;
                for (int x = x0; x <= x1; ++x) {
                    if (raw[row + (size_t)x]) { on = true; break; }
                }
            }
            c.data[(size_t)gy * (size_t)c.gw + (size_t)gx] =
                on ? (uint8_t)1 : (uint8_t)0;
        }
    }

    size_t n_on = 0;
    for (uint8_t v : c.data) if (v) ++n_on;
    std::cout << "[V3RS/sil-cache] (re)built: "
              << c.gw << "x" << c.gh
              << " step=" << step
              << "  cells_on=" << n_on << "/" << c.data.size()
              << "  (squashed + 1-cell halo, source-parity raster)"
              << std::endl;

    g_silTargetMaskCache = std::move(c);
    return true;
}

// Explicit invalidation -- useful e.g. if a caller wants to force a
// rebuild on the next ensureSilTargetMaskCache call. Not strictly
// needed (fingerprint mismatch triggers a rebuild on its own), but
// available for symmetry with other cache APIs.
inline void invalidateSilTargetMaskCache() {
    g_silTargetMaskCache = SilTargetMaskCache{};
}

// =====================================================================
// bilinear_sample -- 2D bilinear sample of a flat row-major float image.
// ---------------------------------------------------------------------
// Returns 9999.0f (sentinel) when (u,v) is out of range or any of the
// four taps holds the >=9000 sentinel; caller should drop those.
// =====================================================================
inline float bilinear_sample(const std::vector<float>& img,
                             int w, int h, float u, float v)
{
    if (w <= 1 || h <= 1) return 9999.0f;
    if (img.size() != (size_t)(w * h)) return 9999.0f;
    if (!(u >= 0.0f) || !(v >= 0.0f)) return 9999.0f;      // NaN-safe
    if (u >= (float)(w - 1) || v >= (float)(h - 1)) return 9999.0f;

    const int x0 = (int)std::floor(u);
    const int y0 = (int)std::floor(v);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = u - (float)x0;
    const float fy = v - (float)y0;

    const float d00 = img[(size_t)(y0 * w + x0)];
    const float d10 = img[(size_t)(y0 * w + x1)];
    const float d01 = img[(size_t)(y1 * w + x0)];
    const float d11 = img[(size_t)(y1 * w + x1)];

    if (d00 >= 9000.0f || d10 >= 9000.0f ||
        d01 >= 9000.0f || d11 >= 9000.0f) {
        return 9999.0f;
    }

    const float d0 = d00 * (1.0f - fx) + d10 * fx;
    const float d1 = d01 * (1.0f - fx) + d11 * fx;
    return d0 * (1.0f - fy) + d1 * fy;
}

// =====================================================================
// captureSilProjectionDebug
// ---------------------------------------------------------------------
// Debug visualization helper. Mirrors the projection + bilinear-sample
// pipeline inside evaluate_one_v3rs_silhouette (revised: no z-buffer,
// rim_subset only). Takes a small set of world-space positions (the
// rim ∩ quadrant subset, typically ~50-150 pts) already at the AR
// camera frame (origin, looking +Z) and records where each one lands
// on the 2D distance map.
//
// Output per point: (world_pos, dist_px).
//   dist_px >= 9000 = out-of-mask (sentinel, kept for colouring)
//   dist_px == 9999 also returned when projection is behind camera
//                   (z<=0), so the caller can show "off-screen" status.
// =====================================================================
inline void captureSilProjectionDebug(
    const std::vector<glm::vec3>& world_positions,
    float  fx, float  fy, float  cx, float  cy,
    int    img_w, int  img_h,
    const std::vector<float>& sil_dist_map_2d,   // (img_w * img_h) full-res
    std::vector<glm::vec3>& out_world,
    std::vector<float>&     out_dist_px,
    int& n_visible_out,
    int& n_with_signal_out)
{
    out_world.clear();
    out_dist_px.clear();
    n_visible_out    = 0;
    n_with_signal_out = 0;

    const size_t N = world_positions.size();
    if (N == 0) return;
    if (img_w  <= 0 || img_h  <= 0) return;

    out_world.reserve(N);
    out_dist_px.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        const glm::vec3& P = world_positions[i];
        float d;
        if (P.z <= 1e-6f) {
            d = 9999.0f;  // behind camera: viz-sentinel
        } else {
            const float u_full = fx * (P.x / P.z) + cx;
            const float v_full = fy * (P.y / P.z) + cy;
            d = bilinear_sample(sil_dist_map_2d,
                                img_w, img_h, u_full, v_full);
            // d == 9999 from out-of-image OR sentinel >= 9000 from mask map
        }
        out_world.push_back(P);
        out_dist_px.push_back(d);
        ++n_visible_out;          // "visible" = "captured" now; no zbuf cull
        if (d < 9000.0f) ++n_with_signal_out;
    }
}


// =====================================================================
// rasterize_iou2d_v3rs -- in-loop IoU2D computation (Phase 1, path B3).
// ---------------------------------------------------------------------
// Pure: no globals, all inputs by argument. Adapted from
// CmaesRefine::computeSilhouette2DObjectiveFast (CmaesUtils.h:385-702)
// to V3RS's data layout (vec3 positions, flat uint32_t indices) and
// trimmed to IoU-only -- Phase 1 doesn't need Hausdorff2D, which
// dominated the original function's runtime.
//
// The caller is expected to pass an MVP that already combines the
// silhouette projection, silhouette view, and the per-eval world
// matrix (M_srt * M_jit). Vertices are projected straight to grid
// coordinates and rasterized with backface culling (CCW in screen
// space). The hitmap is unioned/intersected against a binary target
// mask derived from `dist_map` (entries < 9000 are inside the mask;
// >= 9000 is the DepthUtils "unreachable" sentinel).
//
// Returns IoU2D in [0, 1] (higher = better overlap), or 0 on invalid
// input (empty mesh, mismatched dist_map, degenerate viewport).
//
// Cost (measured per Sil2DBench, HANDOVER §4.1):
//   step=8, 20k tris, 1920x1080 -> ~3.75 ms / call
//   step=4                       -> ~14    ms / call
// =====================================================================
inline float rasterize_iou2d_v3rs(
    const std::vector<glm::vec3>& positions,      // ORIGINAL mesh verts
    const std::vector<uint32_t>&  indices,        // 3 * N_tris
    const glm::mat4&              mvp,            // proj * view * M_world
    const std::vector<float>&     dist_map,       // size = img_w * img_h
    int                            img_w,
    int                            img_h,
    int                            step,
    // ----- Optional outputs for diagnostics (PNG dump / ImGui viz) ---
    // All four must be either all-null (hot path) or all-non-null
    // (capture path). When non-null, the function fills:
    //   *out_hitmap      size = (*out_gw)*(*out_gh), 1 = source pixel
    //   *out_target_mask size = same,                 1 = target pixel
    //   *out_gw, *out_gh : grid dimensions used
    // The IoU is the SAME number returned regardless of capture; the
    // outputs let viz code rebuild the composite without re-running.
    std::vector<uint8_t>*         out_hitmap      = nullptr,
    std::vector<uint8_t>*         out_target_mask = nullptr,
    int*                          out_gw          = nullptr,
    int*                          out_gh          = nullptr,
    // ----- Optional per-step timing outputs (microseconds) -----------
    // When non-null, the function writes the wall-clock time of each
    // internal step to the corresponding pointer. Independent: any
    // subset can be non-null. Used by the run-loop summary log to
    // localise the dominant cost inside this function. All three are
    // null on the typical capture path (no overhead at all then).
    double*                       out_step1_proj_us  = nullptr,
    double*                       out_step2_splat_us = nullptr,
    double*                       out_step3_iou_us   = nullptr,
    // ----- Optional raster-mode selector (F10 diagnostic / F9 toggle) -
    // 0 = triangle-bbox splat. This is the hot-path default; when the
    //     argument is omitted the function is byte-identical to the
    //     pre-F10 implementation, so evaluate_one_v3rs_silhouette and
    //     every other existing caller are unchanged.
    // 1 = vertex squash: project vertices (Step 1, shared), then splat
    //     each surviving vertex into a 3x3 (1-cell halo) cell stamp.
    //     DIAGNOSTIC ONLY -- used by the F9 overlay's Diagnostic slot
    //     so the long-edge hole pattern is eyeball-checkable. NOT a
    //     hot-path option: plain vertex squash has the long-edge holes
    //     documented in HANDOVER §3.5; the adaptive-subdivision idea
    //     (§5.2) is what would eventually make a squash raster viable.
    int                           raster_mode = 0,
    // ----- Optional instrument occlusion mask -------------------------
    // When non-null and its size matches dist_map (img_w * img_h),
    // any grid cell whose centre falls on an instrument-mask pixel
    // satisfying `(*instrument_dist_map)[px] < instrument_thresh_px`
    // is EXCLUDED from the IoU union and intersection counters. When
    // null (default), no exclusion is applied and the IoU computation
    // is byte-identical to the pre-feature path.
    //
    // Excluded cells are also cleared (set to 0) in the captured
    // out_hitmap / out_target_mask, so the F9 composite and any
    // downstream recount of inter/union from those buffers remain
    // consistent with the returned IoU value.
    //
    // Caller obligations:
    //   - same image dimensions as `dist_map` (size = img_w * img_h)
    //   - row-major layout matching DepthUtils::g_instrumentDistMap
    //   - threshold in image pixels (NOT grid cells)
    const std::vector<float>*     instrument_dist_map  = nullptr,
    float                          instrument_thresh_px = 0.0f,
    // ----- [NEW] Optional outside_ratio output (asymmetric IoU) ------
    // When non-null, *out_outside_ratio receives:
    //   (cells with source AND NOT target) / (cells with source)
    // Range [0, 1]; 0 = source ⊆ target, 1 = no overlap. The denominator
    // excludes instrument-occluded cells when the occlusion gate is
    // active, so the metric stays consistent with the IoU returned.
    // When null (default), no extra work is done.
    float*                        out_outside_ratio    = nullptr,
    // ----- [NEW] Optional rim_sil_loss output (boundary alignment) ----
    // When non-null, *out_rim_sil_loss receives the mean per-cell
    // penalty for each SOURCE-BOUNDARY raster cell. The penalty per cell:
    //   1.0                       when outside the target mask
    //   min(dist / rim_sil_max_px, 1.0)   when inside the mask
    // where `dist` comes from dist_map (image-pixel units). The final
    // scalar is the mean over all boundary cells; in [0, 1].
    // 0 = every source-boundary cell on the target boundary; 1 = all
    // outside. When the source raster has no boundary cells (degenerate),
    // the output is 0.0f. When null (default), no extra work is done.
    float*                        out_rim_sil_loss     = nullptr,
    float                         rim_sil_max_px       = 100.0f,
    // ----- [NEW V3RS-RIM-ANAT] Anatomic RIM mode -------------------------
    // When `is_rim_anatomic_per_vertex` is non-null AND its size matches
    // positions.size(), the rim_sil_loss output switches from "per source-
    // boundary cell" to "per projected anatomical RIM vertex":
    //   For each i with (*is_rim_anatomic_per_vertex)[i] == 1:
    //     skip if not visible (frustum cull, post-occlusion gate)
    //     project to (gx, gy) using the same MVP Step 1 computed
    //     d = dist_map at the cell's centre pixel
    //     per_vertex = (d>=9000) ? 1.0 : min(d/rim_sil_max_px, 1.0)
    //   rim_sil_loss = mean(per_vertex) over the kept set
    // This matches the spec the user requested: "use the same RIM region
    // as the Ctrl+G render checkbox" (anatomical LiverRegionLabel::RIM,
    // optionally filtered by quadrant/arvis/caudal upstream). When this
    // pointer is null/empty/size-mismatched, the function falls back to
    // the per-cell raster-boundary computation (byte-identical to the
    // pre-feature behaviour).
    //
    // out_rim_cell_mask (optional): when non-null AND we are in anatomic
    // mode, the function fills it (size gw*gh) with 1 wherever any
    // anatomical RIM vertex projected. Used by the F9 viz to visually
    // highlight anatomical-RIM cells in panels 4 / 6 instead of the
    // 4-neighbour boundary cells. Null = no extra work.
    const std::vector<uint8_t>*   is_rim_anatomic_per_vertex = nullptr,
    std::vector<uint8_t>*         out_rim_cell_mask          = nullptr)
{
    using clk = std::chrono::high_resolution_clock;
    const bool want_t1 = (out_step1_proj_us  != nullptr);
    const bool want_t2 = (out_step2_splat_us != nullptr);
    const bool want_t3 = (out_step3_iou_us   != nullptr);
    if (want_t1) *out_step1_proj_us  = 0.0;
    if (want_t2) *out_step2_splat_us = 0.0;
    if (want_t3) *out_step3_iou_us   = 0.0;
    if (positions.empty() || indices.empty())                return 0.0f;
    if (img_w <= 0 || img_h <= 0)                            return 0.0f;
    if (dist_map.size() != (size_t)img_w * (size_t)img_h)    return 0.0f;
    if (step < 1) step = 1;

    const int gw = (img_w + step - 1) / step;
    const int gh = (img_h + step - 1) / step;
    if (gw <= 0 || gh <= 0) return 0.0f;

    const int nVerts = (int)positions.size();
    const int nTris  = (int)(indices.size() / 3);
    if (nVerts == 0 || nTris == 0) return 0.0f;

    const float halfW  = (float)img_w * 0.5f;
    const float halfH  = (float)img_h * 0.5f;
    const float invStep = 1.0f / (float)step;

    // ----- Step 1: project all vertices to grid space ----------------
    // screen[i] = (gx, gy, ndcZ); inside[i] flags whether the vertex
    // is inside the view-frustum-with-slack (1.2 NDC margin matches
    // computeSilhouette2DObjectiveFast).
    //
    // Parallelization (schedule static, no reduction needed):
    //   After Step 2's false-sharing fix moved the splat down to
    //   ~110 us/eval, this projection loop became the single largest
    //   component of rasterize_iou2d_v3rs (~129 us/eval, ~49% of the
    //   raster cost) -- and it was still fully serial. Profiling made
    //   that obvious so we parallelize it here.
    //
    //   Unlike the Step 2 splat, this loop needs NO per-thread buffers
    //   and NO reduction: iteration i writes only screen[i] and
    //   inside[i], so different threads touch disjoint indices. There
    //   is no write-write sharing of data, only the unavoidable
    //   boundary effect where two threads' ranges meet inside the same
    //   64-byte cache line (screen is glm::vec3 = 12 bytes, so ~5
    //   elements per line). With schedule(static) each thread gets one
    //   contiguous block, so that boundary touches at most 2 cache
    //   lines per thread -- negligible, and each element is written
    //   exactly once (no repeated hammering like the splat did).
    //
    //   schedule(static) (not dynamic): every iteration does the same
    //   fixed work (one mat4*vec4 + a divide + the frustum compare),
    //   so static block partitioning is both the lowest-overhead and
    //   the most cache-friendly choice. dynamic's per-chunk handoff
    //   would only add cost here.
    //
    //   The `if (nVerts >= 2048)` guard skips the fork/join for tiny
    //   meshes where the thread setup would dominate; the full liver
    //   mesh (9992 verts) is comfortably above the threshold.
    const auto t_step1_a = want_t1 ? clk::now() : clk::time_point{};
    std::vector<glm::vec3> screen(nVerts);
    std::vector<uint8_t>   inside(nVerts, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (nVerts >= 2048 && !omp_in_parallel())
#endif
    for (int i = 0; i < nVerts; ++i) {
        const glm::vec3& p = positions[i];
        const glm::vec4 c  = mvp * glm::vec4(p.x, p.y, p.z, 1.0f);
        if (std::abs(c.w) < 1e-8f) {
            screen[i] = glm::vec3(0.0f, 0.0f, 2.0f);
            inside[i] = 0;
            continue;
        }
        const float ndcX = c.x / c.w;
        const float ndcY = c.y / c.w;
        const float ndcZ = c.z / c.w;
        const float px = (ndcX + 1.0f) * halfW;
        const float py = (1.0f - ndcY) * halfH;          // Y flip
        screen[i] = glm::vec3(px * invStep, py * invStep, ndcZ);
        inside[i] = (ndcX > -1.2f && ndcX < 1.2f &&
                     ndcY > -1.2f && ndcY < 1.2f &&
                     ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }
    if (want_t1) {
        const auto t_step1_b = clk::now();
        *out_step1_proj_us = std::chrono::duration<double, std::micro>(
                                 t_step1_b - t_step1_a).count();
    }

    // ----- Step 2: Triangle-bbox splat into hitmap ------------------
    // Why triangle-bbox instead of vertex splat:
    //   Vertex splat (the previous implementation) treats each
    //   surviving vertex as a 5x5 cell halo. That works when the
    //   inter-vertex distance is <= 2 * splat_radius (= 4 cells at
    //   radius 2). For edges longer than 4 cells, the splats from
    //   the two endpoint vertices fail to overlap at the edge's
    //   midpoint -- a 1-cell gap appears. The decimated 9992-vert
    //   liver mesh has p_max edge ~4.5+ cells (step=16), so a
    //   small fraction of long edges produce scattered "specks" of
    //   missing coverage inside the silhouette.
    //
    //   Triangle-bbox splat removes the long-edge failure mode at
    //   the source: each triangle's bbox is, by construction, a
    //   superset of the triangle itself. Marking the entire bbox
    //   guarantees coverage of every cell the triangle touches,
    //   regardless of triangle size or aspect ratio. A 1-cell halo
    //   around each bbox (a) absorbs the rounding error at cell
    //   boundaries for triangles flush with the grid and (b) keeps
    //   adjacent triangles' bboxes connected even when one triangle
    //   is narrow.
    //
    // Cost: O(n_tris * avg_bbox_cells). With 9151 surviving tris
    //   and ~9 cells per (1-cell-haloed) bbox, ~82 K marks per
    //   evaluate -- slightly cheaper than the 5x5 vertex path
    //   (118 K marks), and now the surface itself is the
    //   rasterization unit rather than its sample points.
    //
    // Boundary inflation: 1.5 cells (0.5 from bbox rounding + 1
    //   from halo) = 24 image px at step=16 = 1.25% of 1920 wide.
    //   Tighter than vertex splat's ~32 px.
    //
    // Parallelization (per-thread hitmap + OR reduction):
    //   The PREVIOUS implementation had every thread write 1 directly
    //   into a single shared hitmap. Although correct (byte writes
    //   are idempotent and atomic on x86_64), every write touched a
    //   64-byte cache line that other threads were also writing, so
    //   the line ping-ponged between cores' L1 caches -- classic
    //   false sharing. Per-Run profiling (the breakdown log) showed
    //   step2 splat at ~264 us/eval with a 2.8x run-to-run variance,
    //   the signature of a false-sharing-bound loop.
    //
    //   Fix: give each thread its OWN gw*gh hitmap. No two threads
    //   ever touch the same cache line during the splat. After the
    //   parallel region, the per-thread maps are OR-reduced into the
    //   final hitmap -- a single linear pass over n_threads * gw*gh
    //   bytes (~65 KB at 8 threads, step=16), trivially cheap and
    //   itself cache-friendly (sequential access).
    //
    //   Memory: n_threads * gw*gh bytes. At step=16, 1920x1080,
    //   8 threads: 8 * 8160 = ~65 KB. Allocated once per call. The
    //   alternative (a 2D bitset, or atomic ORs) is more complex and
    //   the 65 KB is well within budget.
    //
    //   schedule(dynamic, 64) is preserved: triangle bboxes vary in
    //   size so dynamic balancing still matters; only the false
    //   sharing on the write side is removed.
    //
    //   The std::min/std::max over the 3 vertices is also unrolled
    //   to explicit comparisons. The initializer_list overload of
    //   std::min does not always inline to branchless code; with
    //   exactly 3 elements the hand-rolled form is reliably faster
    //   and this is the hottest inner loop in the whole evaluator.
    std::vector<uint8_t> hitmap((size_t)gw * (size_t)gh, 0);

    const int n_tris = (int)(indices.size() / 3);
    const auto t_step2_a = want_t2 ? clk::now() : clk::time_point{};

    if (raster_mode == 1) {
        // ---- Vertex-squash raster (DIAGNOSTIC; HANDOVER §3.5 / §5.2) -
        // Project-then-splat: every vertex that survived the Step 1
        // frustum test stamps a 3x3 (1-cell halo) block into the
        // hitmap. Serial -- this path is only ever reached from the
        // F10 diagnostic / F9 Diagnostic-slot capture, never the
        // CMA-ES inner loop, so there is nothing to parallelise for.
        //
        // This is deliberately the "plain vertex squash" that §3.5
        // rejected for the hot path: it reproduces the long-edge hole
        // pattern faithfully so the F10 diagnostic can measure exactly
        // how bad (or not) it is at the hot-path raster step.
        for (int i = 0; i < nVerts; ++i) {
            if (!inside[i]) continue;
            const int cx = (int)std::floor(screen[i].x);
            const int cy = (int)std::floor(screen[i].y);
            const int x0 = std::max(0,      cx - 1);
            const int x1 = std::min(gw - 1, cx + 1);
            const int y0 = std::max(0,      cy - 1);
            const int y1 = std::min(gh - 1, cy + 1);
            if (x0 > x1 || y0 > y1) continue;
            for (int y = y0; y <= y1; ++y) {
                const size_t row = (size_t)y * (size_t)gw;
                for (int x = x0; x <= x1; ++x) {
                    hitmap[row + (size_t)x] = 1;
                }
            }
        }
    } else {
        // ---- Triangle-bbox splat (hot path, raster_mode == 0) ----------
#ifdef _OPENMP
        // [V3RS-PARALLEL] Also require NOT being inside an outer parallel
        // region: when runBipopCmaesV3RS parallelizes across runs, each
        // run should raster SERIALLY (one thread). Nested OpenMP is off by
        // default, so the inner region would run on 1 thread anyway -- but
        // without this guard we'd still allocate + zero + OR-reduce
        // n_threads private hitmaps every eval for nothing. omp_in_parallel()
        // makes the splat take the cheap serial path in that case.
        const int n_threads_splat = (n_tris >= 1024 && !omp_in_parallel())
                                        ? std::max(1, omp_get_max_threads())
                                        : 1;
#else
        const int n_threads_splat = 1;
#endif
        const size_t hitmap_cells = (size_t)gw * (size_t)gh;

        if (n_threads_splat <= 1) {
            // Serial path: write straight into the final hitmap. No
            // false sharing possible with one thread, no reduction cost.
            for (int ti = 0; ti < n_tris; ++ti) {
                const size_t t = (size_t)ti * 3;
                const uint32_t i0 = indices[t + 0];
                const uint32_t i1 = indices[t + 1];
                const uint32_t i2 = indices[t + 2];
                if (i0 >= (uint32_t)nVerts ||
                    i1 >= (uint32_t)nVerts ||
                    i2 >= (uint32_t)nVerts) continue;
                if (!inside[i0] && !inside[i1] && !inside[i2]) continue;

                const glm::vec3& s0 = screen[i0];
                const glm::vec3& s1 = screen[i1];
                const glm::vec3& s2 = screen[i2];

                float minX = s0.x, maxX = s0.x;
                if (s1.x < minX) minX = s1.x; else if (s1.x > maxX) maxX = s1.x;
                if (s2.x < minX) minX = s2.x; else if (s2.x > maxX) maxX = s2.x;
                float minY = s0.y, maxY = s0.y;
                if (s1.y < minY) minY = s1.y; else if (s1.y > maxY) maxY = s1.y;
                if (s2.y < minY) minY = s2.y; else if (s2.y > maxY) maxY = s2.y;

                const int x0 = std::max(0,      (int)std::floor(minX) - 1);
                const int x1 = std::min(gw - 1, (int)std::floor(maxX) + 1);
                const int y0 = std::max(0,      (int)std::floor(minY) - 1);
                const int y1 = std::min(gh - 1, (int)std::floor(maxY) + 1);
                if (x0 > x1 || y0 > y1) continue;

                for (int y = y0; y <= y1; ++y) {
                    const size_t row = (size_t)y * (size_t)gw;
                    for (int x = x0; x <= x1; ++x) {
                        hitmap[row + (size_t)x] = 1;
                    }
                }
            }
        } else {
            // Parallel path: per-thread private hitmap, then OR-reduce.
            std::vector<std::vector<uint8_t>> thread_hitmaps(
                (size_t)n_threads_splat,
                std::vector<uint8_t>(hitmap_cells, 0));

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads_splat)
#endif
            {
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif \ \
    // Each thread owns one private hitmap -- no two threads \ \
    // ever write the same cache line during the splat.
                uint8_t* my_hitmap = thread_hitmaps[(size_t)tid].data();

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
                for (int ti = 0; ti < n_tris; ++ti) {
                    const size_t t = (size_t)ti * 3;
                    const uint32_t i0 = indices[t + 0];
                    const uint32_t i1 = indices[t + 1];
                    const uint32_t i2 = indices[t + 2];
                    if (i0 >= (uint32_t)nVerts ||
                        i1 >= (uint32_t)nVerts ||
                        i2 >= (uint32_t)nVerts) continue;
                    // Skip triangles entirely outside the frustum.
                    if (!inside[i0] && !inside[i1] && !inside[i2]) continue;

                    const glm::vec3& s0 = screen[i0];
                    const glm::vec3& s1 = screen[i1];
                    const glm::vec3& s2 = screen[i2];

                    // Hand-unrolled 3-element min/max (see comment above).
                    float minX = s0.x, maxX = s0.x;
                    if (s1.x < minX) minX = s1.x; else if (s1.x > maxX) maxX = s1.x;
                    if (s2.x < minX) minX = s2.x; else if (s2.x > maxX) maxX = s2.x;
                    float minY = s0.y, maxY = s0.y;
                    if (s1.y < minY) minY = s1.y; else if (s1.y > maxY) maxY = s1.y;
                    if (s2.y < minY) minY = s2.y; else if (s2.y > maxY) maxY = s2.y;

                    // bbox in cells, with 1-cell halo on each side.
                    const int x0 = std::max(0,      (int)std::floor(minX) - 1);
                    const int x1 = std::min(gw - 1, (int)std::floor(maxX) + 1);
                    const int y0 = std::max(0,      (int)std::floor(minY) - 1);
                    const int y1 = std::min(gh - 1, (int)std::floor(maxY) + 1);
                    if (x0 > x1 || y0 > y1) continue;

                    for (int y = y0; y <= y1; ++y) {
                        const size_t row = (size_t)y * (size_t)gw;
                        for (int x = x0; x <= x1; ++x) {
                            my_hitmap[row + (size_t)x] = 1;
                        }
                    }
                }
            } // end parallel region

            // OR-reduce per-thread hitmaps into the final hitmap.
            // Single linear pass; sequential access, cache-friendly.
            // Outer loop over threads, inner over cells, so the final
            // hitmap is written once per cell-per-thread but always
            // streaming forward.
            for (int tt = 0; tt < n_threads_splat; ++tt) {
                const uint8_t* src = thread_hitmaps[(size_t)tt].data();
                for (size_t i = 0; i < hitmap_cells; ++i) {
                    hitmap[i] |= src[i];
                }
            }
        }
    }  // end raster_mode == 0 (triangle-bbox splat) branch
    if (want_t2) {
        const auto t_step2_b = clk::now();
        *out_step2_splat_us = std::chrono::duration<double, std::micro>(
                                  t_step2_b - t_step2_a).count();
    }

    // ----- Step 3: IoU vs binary target mask -------------------------
    // Two paths, gated by g_silTargetSquashEnabled and cache validity:
    //
    //   (a) CACHE path (preferred): the target mask was pre-built by
    //       ensureSilTargetMaskCache using the SAME raster system as
    //       the source side (step x step OR coverage + 1-cell halo).
    //       Source and target are now constructed symmetrically, so
    //       triangle-bbox inflation cancels out of the IoU. This is
    //       the fix for the asymmetry that was capping IoU at ~0.63.
    //
    //   (b) LEGACY path (fallback): per-grid-cell CENTRE SAMPLE of
    //       dist_map (the original implementation). Asymmetric vs the
    //       source side. Used when (i) the toggle is OFF for A/B
    //       comparison, (ii) ensureSilTargetMaskCache wasn't called,
    //       or (iii) the cache fingerprint doesn't match this call's
    //       (dist_map, img_w, img_h, step) tuple.
    //
    // Capture path (F9 ImGui overlay / PNG dump) honours the same
    // gating so the user sees in F9 exactly what the cost function
    // evaluates.
    const auto t_step3_a = want_t3 ? clk::now() : clk::time_point{};
    const bool want_capture = (out_hitmap != nullptr)
                              && (out_target_mask != nullptr)
                              && (out_gw != nullptr)
                              && (out_gh != nullptr);
    std::vector<uint8_t> tmask_local;
    if (want_capture) {
        tmask_local.assign((size_t)gw * (size_t)gh, 0);
    }

    // ----- Instrument occlusion mask gate ----------------------------
    // When `instrument_dist_map` is non-null AND its size matches
    // dist_map, we enable per-cell exclusion: grid cells whose centre
    // pixel has instrument-distance < threshold are skipped entirely
    // (not counted in union or intersection). When disabled (default,
    // pointer null OR size mismatch OR threshold negative), this code
    // path is short-circuited and IoU is byte-identical to the
    // pre-feature implementation.
    //
    // The mismatch fallback is intentionally silent: a caller that
    // wires up an instrument map of wrong dimensions gets the same
    // behaviour as not passing one. Logging is left to the wrapper.
    const bool occ_enabled =
        (instrument_dist_map != nullptr) &&
        (instrument_dist_map->size() == (size_t)img_w * (size_t)img_h) &&
        (instrument_thresh_px >= 0.0f);
    const float* occ_data = occ_enabled ? instrument_dist_map->data() : nullptr;

    const bool use_cache =
        g_silTargetSquashEnabled &&
        isSilTargetMaskCacheValidFor(dist_map, img_w, img_h, step) &&
        g_silTargetMaskCache.gw == gw &&
        g_silTargetMaskCache.gh == gh;

    int inter = 0, uni = 0;
    // [NEW V3RS-3] outside_ratio counters. Incremented only when the
    // caller requested out_outside_ratio; otherwise zeroed and ignored.
    // The branch is hoisted by the compiler (loop-invariant), so the
    // cost when the feature is off is ~zero.
    const bool want_outside = (out_outside_ratio != nullptr);
    int src_total = 0;
    int src_outside = 0;
    if (use_cache) {
        // (a) Cache path. The target mask is at the same gw x gh grid
        // as our hitmap, so the inner loop reduces to a flat sweep.
        //
        // Occlusion variant: the cache target mask is at grid resolution,
        // but the instrument mask is at image resolution -- we have to
        // re-derive the per-cell centre pixel coordinates to sample it.
        // The cost is gw*gh extra reads (~8K-40K depending on step) and
        // is negligible vs the Step 2 splat budget.
        const uint8_t* tmask_src = g_silTargetMaskCache.data.data();
        if (!occ_enabled) {
            // Original cache path (byte-identical to pre-feature).
            const size_t   N         = (size_t)gw * (size_t)gh;
            for (size_t i = 0; i < N; ++i) {
                const bool s = (hitmap[i]    != 0);
                const bool t = (tmask_src[i] != 0);
                if (s || t) ++uni;
                if (s && t) ++inter;
                // [NEW V3RS-3] outside_ratio accumulators
                if (want_outside) {
                    if (s)        ++src_total;
                    if (s && !t)  ++src_outside;
                }
                if (want_capture && t) tmask_local[i] = 1;
            }
        } else {
            // Occlusion-aware cache path. Same per-cell logic plus the
            // instrument-dist gate. Cells that fail the gate are
            // excluded from both inter and uni; in capture mode we also
            // zero hitmap[i] / tmask_local[i] so the F9 composite shows
            // black for occluded cells and any downstream recount of
            // inter/union from those buffers stays consistent.
            for (int gy = 0; gy < gh; ++gy) {
                const int ipy = gy * step + step / 2;
                const int my  = (ipy < 0) ? 0
                                         : (ipy >= img_h ? img_h - 1 : ipy);
                const size_t row_occ = (size_t)my * (size_t)img_w;
                const size_t row_grd = (size_t)gy * (size_t)gw;
                for (int gx = 0; gx < gw; ++gx) {
                    const int ipx = gx * step + step / 2;
                    const int mx  = (ipx < 0) ? 0
                                             : (ipx >= img_w ? img_w - 1 : ipx);
                    const float idist = occ_data[row_occ + (size_t)mx];
                    if (idist < instrument_thresh_px) {
                        if (want_capture) {
                            hitmap[row_grd + (size_t)gx] = 0;
                            // tmask_local stays 0 (already zeroed at assign)
                        }
                        continue;   // skip this cell entirely
                    }
                    const size_t i = row_grd + (size_t)gx;
                    const bool s = (hitmap[i]    != 0);
                    const bool t = (tmask_src[i] != 0);
                    if (s || t) ++uni;
                    if (s && t) ++inter;
                    // [NEW V3RS-3] outside_ratio accumulators (post-occlusion)
                    if (want_outside) {
                        if (s)        ++src_total;
                        if (s && !t)  ++src_outside;
                    }
                    if (want_capture && t) tmask_local[i] = 1;
                }
            }
        }
    } else {
        // (b) Legacy path. Centre-sample dist_map at each cell.
        // Occlusion gate is fused into the same nested loop because
        // the centre-pixel coordinates are computed here anyway --
        // adding the instrument test costs one extra read + compare
        // per cell, no extra arithmetic.
        for (int gy = 0; gy < gh; ++gy) {
            const int ipy = gy * step + step / 2;
            const int my  = (ipy < 0) ? 0
                                     : (ipy >= img_h ? img_h - 1 : ipy);
            const size_t row_dist = (size_t)my * (size_t)img_w;
            const size_t row_hit  = (size_t)gy * (size_t)gw;
            for (int gx = 0; gx < gw; ++gx) {
                const int ipx = gx * step + step / 2;
                const int mx  = (ipx < 0) ? 0
                                         : (ipx >= img_w ? img_w - 1 : ipx);
                // Instrument occlusion gate (no-op when disabled).
                if (occ_enabled) {
                    const float idist = occ_data[row_dist + (size_t)mx];
                    if (idist < instrument_thresh_px) {
                        if (want_capture) {
                            hitmap[row_hit + (size_t)gx] = 0;
                            // tmask_local stays 0
                        }
                        continue;   // skip this cell entirely
                    }
                }
                const bool t = (dist_map[row_dist + (size_t)mx] < 9000.0f);
                const bool s = (hitmap[row_hit + (size_t)gx] != 0);
                if (s || t) uni++;
                if (s && t) inter++;
                // [NEW V3RS-3] outside_ratio accumulators (legacy path)
                if (want_outside) {
                    if (s)        ++src_total;
                    if (s && !t)  ++src_outside;
                }
                if (want_capture && t) {
                    tmask_local[row_hit + (size_t)gx] = 1;
                }
            }
        }
    }

    // ----- [NEW V3RS-3] Publish outside_ratio --------------------------
    if (out_outside_ratio) {
        *out_outside_ratio = (src_total > 0)
            ? (float)src_outside / (float)src_total
            : 0.0f;
    }

    // ----- [NEW V3RS-3] rim_sil_loss output ---------------------------
    // Two modes (chosen by whether is_rim_anatomic_per_vertex was passed):
    //
    //   ANATOMIC mode: iterate over RIM vertices. For each visible RIM
    //     vertex, project to image space, sample dist_map. Anatomically
    //     meaningful penalty. Matches the Ctrl+G "Show RIM pairs"
    //     definition of the source rim. Cells touched are also written
    //     to out_rim_cell_mask when that pointer is non-null (for F9
    //     viz). Cost is O(N_rim) ~ a few hundred reads; very cheap.
    //
    //   LEGACY mode: per-cell 4-neighbour boundary detection on the
    //     hitmap. Treats the entire 2D silhouette boundary of the source
    //     raster as the rim. Cost is O(gw*gh) ~ 8K reads; still well
    //     under 1 ms but penalises ALL silhouette outline cells including
    //     those of detached blobs / artefacts.
    //
    // [NEW V3RS-RIM-ANAT] When is_rim_anatomic_per_vertex is provided,
    // the function switches to ANATOMIC mode (described in the param
    // block at the top of the signature).
    const bool anatomic_mode =
        (is_rim_anatomic_per_vertex != nullptr)
        && !is_rim_anatomic_per_vertex->empty()
        && (is_rim_anatomic_per_vertex->size() == (size_t)nVerts);
    // Always (optionally) clear the cell mask first so the caller sees
    // a clean state. In LEGACY mode we don't populate it.
    if (out_rim_cell_mask) {
        out_rim_cell_mask->assign((size_t)gw * (size_t)gh, 0);
    }
    if (out_rim_sil_loss && anatomic_mode) {
        // ----- ANATOMIC MODE: per-vertex projection penalty --------------
        double rim_sum = 0.0;
        int    rim_count = 0;
        const float D_MAX = (rim_sil_max_px > 0.0f) ? rim_sil_max_px : 100.0f;
        for (int i = 0; i < nVerts; ++i) {
            if (!(*is_rim_anatomic_per_vertex)[i]) continue;
            if (!inside[i]) continue;             // outside frustum-with-slack

            // screen[i] is already in grid-cell coordinates (Step 1
            // multiplied raw pixel coords by invStep). Round-to-nearest
            // is fine for cell membership; clamping keeps us in-bounds
            // for the rare frustum-boundary case.
            const int gx = (int)screen[i].x;
            const int gy = (int)screen[i].y;
            if (gx < 0 || gx >= gw || gy < 0 || gy >= gh) continue;

            const int ipy = gy * step + step / 2;
            const int ipx = gx * step + step / 2;
            const int my  = (ipy < 0) ? 0
                                     : (ipy >= img_h ? img_h - 1 : ipy);
            const int mx  = (ipx < 0) ? 0
                                     : (ipx >= img_w ? img_w - 1 : ipx);

            // Occlusion gate: skip vertices whose projection falls on an
            // instrument-occluded pixel. Mirrors the IoU rasterizer's
            // gate so the cost is internally consistent.
            if (occ_enabled) {
                const float idist = occ_data[(size_t)my * (size_t)img_w + (size_t)mx];
                if (idist < instrument_thresh_px) continue;
            }

            // Mark cell in the optional output mask (for F9 viz). Done
            // BEFORE the penalty compute so the mask exactly reflects
            // which vertices contributed to rim_sil_loss.
            if (out_rim_cell_mask) {
                (*out_rim_cell_mask)[(size_t)gy * (size_t)gw + (size_t)gx] = 1;
            }

            const float d = dist_map[(size_t)my * (size_t)img_w + (size_t)mx];
            float per_vertex;
            if (d >= 9000.0f) {
                per_vertex = 1.0f;   // projection outside target mask
            } else {
                per_vertex = std::min(d / D_MAX, 1.0f);
            }
            rim_sum += (double)per_vertex;
            ++rim_count;
        }
        *out_rim_sil_loss = (rim_count > 0)
            ? (float)(rim_sum / (double)rim_count)
            : 0.0f;
    } else if (out_rim_sil_loss) {
        // ----- LEGACY MODE: per-cell raster-boundary penalty -------------
        double rim_sum = 0.0;
        int    rim_count = 0;
        const float D_MAX = (rim_sil_max_px > 0.0f) ? rim_sil_max_px : 100.0f;
        for (int gy = 0; gy < gh; ++gy) {
            const size_t row_grd = (size_t)gy * (size_t)gw;
            const int ipy = gy * step + step / 2;
            const int my  = (ipy < 0) ? 0
                                     : (ipy >= img_h ? img_h - 1 : ipy);
            const size_t row_img = (size_t)my * (size_t)img_w;
            for (int gx = 0; gx < gw; ++gx) {
                const size_t idx = row_grd + (size_t)gx;
                if (!hitmap[idx]) continue;

                // 4-neighbour source-boundary test. Edge of the grid
                // counts as a non-source neighbour, so cells at the grid
                // border are treated as boundary cells (an open silhouette
                // has a rim there, conceptually).
                const bool nl = (gx > 0)      ? (hitmap[idx - 1]            != 0) : false;
                const bool nr = (gx + 1 < gw) ? (hitmap[idx + 1]            != 0) : false;
                const bool nu = (gy > 0)      ? (hitmap[idx - (size_t)gw]   != 0) : false;
                const bool nd = (gy + 1 < gh) ? (hitmap[idx + (size_t)gw]   != 0) : false;
                if (nl && nr && nu && nd) {
                    continue;  // interior source cell
                }

                const int ipx = gx * step + step / 2;
                const int mx  = (ipx < 0) ? 0
                                         : (ipx >= img_w ? img_w - 1 : ipx);

                // Occlusion gate: skip occluded boundary cells (they
                // are excluded from IoU too, for consistency).
                if (occ_enabled) {
                    const float idist = occ_data[row_img + (size_t)mx];
                    if (idist < instrument_thresh_px) continue;
                }

                // [NEW V3RS-RIM-ANAT] Mark cell in optional output mask
                // ALSO in legacy mode. Lets the F9 viz use a single
                // codepath: "highlight cells in rim_cell_mask" without
                // caring which mode produced them.
                if (out_rim_cell_mask) {
                    (*out_rim_cell_mask)[idx] = 1;
                }

                const float d = dist_map[row_img + (size_t)mx];
                float per_cell;
                if (d >= 9000.0f) {
                    per_cell = 1.0f;   // outside target mask → max penalty
                } else {
                    per_cell = std::min(d / D_MAX, 1.0f);
                }
                rim_sum += (double)per_cell;
                ++rim_count;
            }
        }
        *out_rim_sil_loss = (rim_count > 0)
            ? (float)(rim_sum / (double)rim_count)
            : 0.0f;
    }

    if (want_capture) {
        *out_hitmap      = hitmap;          // copy: ~32 KB
        *out_target_mask = std::move(tmask_local);
        *out_gw          = gw;
        *out_gh          = gh;
    }

    if (want_t3) {
        const auto t_step3_b = clk::now();
        *out_step3_iou_us = std::chrono::duration<double, std::micro>(
                                t_step3_b - t_step3_a).count();
    }

    return (uni > 0) ? (float)inter / (float)uni : 0.0f;
}

// =====================================================================
// diagnoseSilCoverageOnce -- one-shot raster coverage histogram.
// ---------------------------------------------------------------------
// Measurement-only. Re-runs the Step 1 projection + Step 2 triangle-bbox
// splat ONCE (serial, diagnostic), but instead of a 0/1 hitmap it keeps
// a per-cell COVERAGE COUNT: how many triangle bboxes touched each cell.
//
// Purpose: estimate the headroom for a flood-fill / scanline interior
// fill. The current splat writes every cell a triangle's bbox covers;
// interior cells get hit many times (overlapping bboxes), boundary
// cells typically once. A flood-fill interior fill would touch each
// on-cell exactly once. So:
//
//   current splat writes  = sum over cells of coverage[c]
//   flood-fill ideal      = number of on-cells (coverage >= 1)
//   potential speedup     = sum(coverage) / count(coverage>=1)
//                         = "avg coverage"
//
// A high avg coverage (say 3-5x) means the interior is thick and a
// flood-fill would pay off big. A low avg coverage (~1.5x) means the
// silhouette is mostly boundary and there's little to gain.
//
// Also reports the boundary/interior split (cov==1 vs cov>=2) and a
// histogram so we can see the coverage distribution shape.
//
// Called once per session (Run 1, initial pose) from run_one_bipop_v3rs.
// Does NOT touch optimizer state; the projection here is independent
// of the hot-path rasterizer.
// =====================================================================
inline void diagnoseSilCoverageOnce(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>&  indices,
    const glm::mat4&              mvp,
    int                            img_w,
    int                            img_h,
    int                            step,
    int                            run_index)
{
    if (positions.empty() || indices.empty()) return;
    if (img_w <= 0 || img_h <= 0)             return;
    if (step < 1) step = 1;

    const int gw = (img_w + step - 1) / step;
    const int gh = (img_h + step - 1) / step;
    if (gw <= 0 || gh <= 0) return;

    const int nVerts = (int)positions.size();
    const int nTris  = (int)(indices.size() / 3);
    if (nVerts == 0 || nTris == 0) return;

    const float halfW   = (float)img_w * 0.5f;
    const float halfH   = (float)img_h * 0.5f;
    const float invStep = 1.0f / (float)step;

    // ----- Step 1: project (serial; this is a one-shot diagnostic) ---
    std::vector<glm::vec3> screen(nVerts);
    std::vector<uint8_t>   inside(nVerts, 0);
    for (int i = 0; i < nVerts; ++i) {
        const glm::vec3& p = positions[i];
        const glm::vec4 c  = mvp * glm::vec4(p.x, p.y, p.z, 1.0f);
        if (std::abs(c.w) < 1e-8f) {
            screen[i] = glm::vec3(0.0f, 0.0f, 2.0f);
            inside[i] = 0;
            continue;
        }
        const float ndcX = c.x / c.w;
        const float ndcY = c.y / c.w;
        const float ndcZ = c.z / c.w;
        const float px = (ndcX + 1.0f) * halfW;
        const float py = (1.0f - ndcY) * halfH;          // Y flip
        screen[i] = glm::vec3(px * invStep, py * invStep, ndcZ);
        inside[i] = (ndcX > -1.2f && ndcX < 1.2f &&
                     ndcY > -1.2f && ndcY < 1.2f &&
                     ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }

    // ----- Step 2: coverage-count splat (serial) ---------------------
    // coverage[c]      = how many triangle bboxes (1-cell halo) wrote c.
    // coverage_core[c] = same WITHOUT the 1-cell halo (bbox only). The
    //                    halo is what inflates the boundary; comparing
    //                    the two tells us how much of "cells_on" is
    //                    halo bleed vs real triangle coverage.
    std::vector<uint32_t> coverage((size_t)gw * (size_t)gh, 0);
    std::vector<uint32_t> coverage_core((size_t)gw * (size_t)gh, 0);
    long long total_writes = 0;
    for (int ti = 0; ti < nTris; ++ti) {
        const size_t t = (size_t)ti * 3;
        const uint32_t i0 = indices[t + 0];
        const uint32_t i1 = indices[t + 1];
        const uint32_t i2 = indices[t + 2];
        if (i0 >= (uint32_t)nVerts ||
            i1 >= (uint32_t)nVerts ||
            i2 >= (uint32_t)nVerts) continue;
        if (!inside[i0] && !inside[i1] && !inside[i2]) continue;

        const glm::vec3& s0 = screen[i0];
        const glm::vec3& s1 = screen[i1];
        const glm::vec3& s2 = screen[i2];

        float minX = s0.x, maxX = s0.x;
        if (s1.x < minX) minX = s1.x; else if (s1.x > maxX) maxX = s1.x;
        if (s2.x < minX) minX = s2.x; else if (s2.x > maxX) maxX = s2.x;
        float minY = s0.y, maxY = s0.y;
        if (s1.y < minY) minY = s1.y; else if (s1.y > maxY) maxY = s1.y;
        if (s2.y < minY) minY = s2.y; else if (s2.y > maxY) maxY = s2.y;

        // halo version (matches the hot path exactly).
        const int x0 = std::max(0,      (int)std::floor(minX) - 1);
        const int x1 = std::min(gw - 1, (int)std::floor(maxX) + 1);
        const int y0 = std::max(0,      (int)std::floor(minY) - 1);
        const int y1 = std::min(gh - 1, (int)std::floor(maxY) + 1);
        if (!(x0 > x1 || y0 > y1)) {
            for (int y = y0; y <= y1; ++y) {
                const size_t row = (size_t)y * (size_t)gw;
                for (int x = x0; x <= x1; ++x) {
                    coverage[row + (size_t)x]++;
                    ++total_writes;
                }
            }
        }

        // core version (NO halo): bbox clamped, no +-1 expansion.
        const int cx0 = std::max(0,      (int)std::floor(minX));
        const int cx1 = std::min(gw - 1, (int)std::floor(maxX));
        const int cy0 = std::max(0,      (int)std::floor(minY));
        const int cy1 = std::min(gh - 1, (int)std::floor(maxY));
        if (!(cx0 > cx1 || cy0 > cy1)) {
            for (int y = cy0; y <= cy1; ++y) {
                const size_t row = (size_t)y * (size_t)gw;
                for (int x = cx0; x <= cx1; ++x) {
                    coverage_core[row + (size_t)x]++;
                }
            }
        }
    }

    // ----- Histogram + boundary/interior split -----------------------
    long long cells_on = 0, boundary_cells = 0, interior_cells = 0;
    long long h1 = 0, h2 = 0, h3 = 0, h4 = 0, h5 = 0, h6_10 = 0, h11p = 0;
    uint32_t  max_cov = 0;
    for (size_t i = 0; i < coverage.size(); ++i) {
        const uint32_t c = coverage[i];
        if (c == 0) continue;
        ++cells_on;
        if (c == 1) { ++boundary_cells; ++h1; }
        else {
            ++interior_cells;
            if      (c == 2)  ++h2;
            else if (c == 3)  ++h3;
            else if (c == 4)  ++h4;
            else if (c == 5)  ++h5;
            else if (c <= 10) ++h6_10;
            else              ++h11p;
        }
        if (c > max_cov) max_cov = c;
    }

    const double cells_on_d   = (cells_on > 0) ? (double)cells_on : 1.0;
    const double avg_cov      = (double)total_writes / cells_on_d;
    const double boundary_pct = 100.0 * (double)boundary_cells / cells_on_d;
    const double interior_pct = 100.0 * (double)interior_cells / cells_on_d;

    // ----- GEOMETRIC boundary (independent of coverage count) --------
    // The cov==1 metric only finds cells that exactly one bbox touched.
    // With 1-cell halos overlapping everywhere, that under-counts the
    // true outline. The real boundary is topological: an on-cell whose
    // 4-neighbourhood contains at least one off-cell. This is computed
    // straight from the binary hitmap shape, NOT from coverage, so it
    // cannot be fooled by halo overlap.
    long long geo_boundary = 0, geo_interior = 0;
    long long cells_on_core = 0;
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            const size_t idx = (size_t)y * (size_t)gw + (size_t)x;
            if (coverage[idx] == 0) continue;
            // 4-neighbour check; grid edge counts as "off" neighbour.
            bool has_off_neighbour = false;
            if (x == 0      || coverage[idx - 1]       == 0) has_off_neighbour = true;
            if (x == gw - 1 || coverage[idx + 1]       == 0) has_off_neighbour = true;
            if (y == 0      || coverage[idx - gw]      == 0) has_off_neighbour = true;
            if (y == gh - 1 || coverage[idx + (size_t)gw] == 0) has_off_neighbour = true;
            if (has_off_neighbour) ++geo_boundary;
            else                   ++geo_interior;
        }
    }
    for (size_t i = 0; i < coverage_core.size(); ++i) {
        if (coverage_core[i] != 0) ++cells_on_core;
    }

    // ----- Per-row span structure (scanline feasibility) -------------
    // For each grid row: how many DISJOINT runs of on-cells. A row with
    // exactly 1 run is "convex enough" for naive scanline fill. A row
    // with 2+ runs has a concavity/hole -- naive scanline would fill
    // the gap between runs and corrupt the silhouette there.
    //   xmin/xmax span = the naive-scanline footprint of that row.
    //   sum of (span - on_cells_in_row) over multi-run rows = how many
    //   cells naive scanline would WRONGLY fill.
    long long rows_with_on   = 0;
    long long rows_single    = 0;   // exactly 1 run
    long long rows_multi     = 0;   // 2+ runs
    long long max_runs_row   = 0;
    long long scanline_overfill = 0;  // cells naive scanline would add
    for (int y = 0; y < gh; ++y) {
        int runs = 0;
        int on_in_row = 0;
        int row_xmin = -1, row_xmax = -1;
        bool prev_on = false;
        for (int x = 0; x < gw; ++x) {
            const bool on = (coverage[(size_t)y * (size_t)gw + (size_t)x] != 0);
            if (on) {
                ++on_in_row;
                if (row_xmin < 0) row_xmin = x;
                row_xmax = x;
                if (!prev_on) ++runs;
            }
            prev_on = on;
        }
        if (on_in_row == 0) continue;
        ++rows_with_on;
        if (runs <= 1) ++rows_single;
        else {
            ++rows_multi;
            // naive scanline would fill row_xmin..row_xmax inclusive.
            const int span = row_xmax - row_xmin + 1;
            scanline_overfill += (long long)(span - on_in_row);
        }
        if (runs > max_runs_row) max_runs_row = runs;
    }

    // ----- Output ----------------------------------------------------
    std::cout << "[V3RS/cov-diag] === Coverage diagnostic (Run "
              << (run_index + 1) << ", initial pose) ===" << std::endl;
    std::cout << "[V3RS/cov-diag]   grid=" << gw << "x" << gh
              << "  tris=" << nTris
              << "  cells_on=" << cells_on << "/" << coverage.size()
              << "  cells_on(no-halo)=" << cells_on_core
              << std::endl;
    std::cout << std::fixed << std::setprecision(2)
              << "[V3RS/cov-diag]   total bbox writes : " << total_writes
              << "   avg coverage = " << avg_cov << " writes/on-cell"
              << std::endl;
    std::cout << std::setprecision(1)
              << "[V3RS/cov-diag]   cov==1 cells      : " << boundary_cells
              << "  (" << boundary_pct << "% of on-cells)"
              << "   [coverage-based, halo-fooled]" << std::endl;
    std::cout << "[V3RS/cov-diag]   cov>=2 cells      : " << interior_cells
              << "  (" << interior_pct << "% of on-cells)" << std::endl;
    // The honest boundary number:
    {
        const double gb_pct = 100.0 * (double)geo_boundary / cells_on_d;
        const double gi_pct = 100.0 * (double)geo_interior / cells_on_d;
        std::cout << "[V3RS/cov-diag]   GEOMETRIC boundary: " << geo_boundary
                  << "  (" << gb_pct << "% of on-cells)"
                  << "   [4-neighbour, halo-proof]" << std::endl;
        std::cout << "[V3RS/cov-diag]   GEOMETRIC interior: " << geo_interior
                  << "  (" << gi_pct << "% of on-cells)" << std::endl;
    }
    std::cout << "[V3RS/cov-diag]   coverage histogram (on-cells):"
              << std::endl;
    std::cout << "[V3RS/cov-diag]     cov==1 : " << h1 << std::endl;
    std::cout << "[V3RS/cov-diag]     cov==2 : " << h2 << std::endl;
    std::cout << "[V3RS/cov-diag]     cov==3 : " << h3 << std::endl;
    std::cout << "[V3RS/cov-diag]     cov==4 : " << h4 << std::endl;
    std::cout << "[V3RS/cov-diag]     cov==5 : " << h5 << std::endl;
    std::cout << "[V3RS/cov-diag]     cov 6-10 : " << h6_10 << std::endl;
    std::cout << "[V3RS/cov-diag]     cov 11+  : " << h11p
              << "   (max coverage = " << max_cov << ")" << std::endl;
    // Row span structure -- the scanline-feasibility evidence.
    std::cout << "[V3RS/cov-diag]   --- per-row span structure ---"
              << std::endl;
    std::cout << "[V3RS/cov-diag]     rows with on-cells : " << rows_with_on
              << std::endl;
    std::cout << "[V3RS/cov-diag]     single-run rows    : " << rows_single
              << "   (naive scanline OK)" << std::endl;
    std::cout << "[V3RS/cov-diag]     multi-run rows     : " << rows_multi
              << "   (naive scanline would overfill)" << std::endl;
    std::cout << "[V3RS/cov-diag]     max runs in a row  : " << max_runs_row
              << std::endl;
    std::cout << "[V3RS/cov-diag]     scanline overfill  : " << scanline_overfill
              << " cells would be wrongly filled" << std::endl;
    std::cout << std::setprecision(2)
              << "[V3RS/cov-diag]   flood-fill estimate: "
              << "ideal writes = " << cells_on
              << "  vs current = " << total_writes
              << "  -> up to " << avg_cov
              << "x fewer splat writes" << std::endl;

    // ----- ASCII map of the silhouette -------------------------------
    // Visual sanity check: print the hitmap as text so the actual shape
    // is eyeball-verifiable. Legend:
    //   ' ' off   '.' cov 1   ':' cov 2-3   '+' cov 4-10   '#' cov 11+
    // If gw is wide, this wraps per the terminal; that's fine, the
    // shape is still recognisable. Printed last so it doesn't push the
    // numbers off-screen.
    std::cout << "[V3RS/cov-diag]   --- ASCII silhouette map ("
              << gw << "x" << gh
              << ", legend: ' '=off .=1 :=2-3 +=4-10 #=11+) ---"
              << std::endl;
    for (int y = 0; y < gh; ++y) {
        std::string line = "[V3RS/cov-diag]   |";
        for (int x = 0; x < gw; ++x) {
            const uint32_t c = coverage[(size_t)y * (size_t)gw + (size_t)x];
            char ch;
            if      (c == 0)  ch = ' ';
            else if (c == 1)  ch = '.';
            else if (c <= 3)  ch = ':';
            else if (c <= 10) ch = '+';
            else              ch = '#';
            line += ch;
        }
        line += "|";
        std::cout << line << std::endl;
    }
    std::cout << std::defaultfloat << std::setprecision(6);
}

// =====================================================================
// diagnoseVertexSquashOnce -- one-shot A/B raster comparison.
// ---------------------------------------------------------------------
// Measurement-only. Re-runs Step 1 projection ONCE, then builds TWO
// hitmaps from the same projected vertices:
//   A = triangle-bbox splat   (the current hot path, raster_mode 0)
//   B = vertex squash 3x3     (raster_mode 1, plain project-then-splat)
// and reports, side by side:
//   - cells_on(A), cells_on(B)
//   - total writes A vs B           (per-write cost asymmetry, §2.5)
//   - hole count = cells in A but NOT in B  (the long-edge failure of
//     plain vertex squash; THIS is the number Phase B must drive to 0)
//   - overshoot  = cells in B but NOT in A
//   - IoU(A) and IoU(B) vs the SAME target mask the hot path uses
//     (cache path if g_silTargetSquashEnabled and the cache is valid,
//      else legacy centre-sample) so the IoU delta is apples-to-apples
//   - projected per-triangle max-edge-length histogram: long edges are
//     what open holes (HANDOVER §3.5), so this localises which
//     triangles a Phase B adaptive subdivision would need to target
//   - ASCII map with the hole pattern marked ('H')
//
// This is the §4.3 "new diagnostic function" for the vertex-squash /
// adaptive-subdivision line of work (the old diagnoseFloodFillOnce was
// deleted with the flood-fill 3-losses). It does NOT touch optimizer
// state and is invoked on demand by the F10 key (current static pose),
// independent of any Ctrl+Shift+G session.
//
// Decision rule (mirrors §2.5 / §5.2): vertex squash is only worth
// pursuing as a hot-path replacement if a candidate exists with
// hole_count == 0 AND total_writes(B) < total_writes(A). At the plain
// (un-subdivided) stage this will almost certainly FAIL the hole test;
// the value here is measuring HOW MANY holes and from HOW FEW big
// triangles, which is exactly what tells us whether Phase B
// (subdivide only the long-edge triangles) has a chance.
// =====================================================================
inline void diagnoseVertexSquashOnce(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>&  indices,
    const glm::mat4&              mvp,
    const std::vector<float>&     dist_map,
    int                            img_w,
    int                            img_h,
    int                            step,
    int                            run_index)   // -1 = F10 on-demand
{
    if (positions.empty() || indices.empty()) return;
    if (img_w <= 0 || img_h <= 0)             return;
    if (dist_map.size() != (size_t)img_w * (size_t)img_h) return;
    if (step < 1) step = 1;

    const int gw = (img_w + step - 1) / step;
    const int gh = (img_h + step - 1) / step;
    if (gw <= 0 || gh <= 0) return;

    const int nVerts = (int)positions.size();
    const int nTris  = (int)(indices.size() / 3);
    if (nVerts == 0 || nTris == 0) return;

    const float halfW   = (float)img_w * 0.5f;
    const float halfH   = (float)img_h * 0.5f;
    const float invStep = 1.0f / (float)step;

    const std::string tag = "[V3RS/vsq-diag]";

    // ----- Step 1: project (serial; one-shot diagnostic) -------------
    std::vector<glm::vec3> screen(nVerts);
    std::vector<uint8_t>   inside(nVerts, 0);
    for (int i = 0; i < nVerts; ++i) {
        const glm::vec3& p = positions[i];
        const glm::vec4 c  = mvp * glm::vec4(p.x, p.y, p.z, 1.0f);
        if (std::abs(c.w) < 1e-8f) {
            screen[i] = glm::vec3(0.0f, 0.0f, 2.0f);
            inside[i] = 0;
            continue;
        }
        const float ndcX = c.x / c.w;
        const float ndcY = c.y / c.w;
        const float ndcZ = c.z / c.w;
        const float px = (ndcX + 1.0f) * halfW;
        const float py = (1.0f - ndcY) * halfH;          // Y flip
        screen[i] = glm::vec3(px * invStep, py * invStep, ndcZ);
        inside[i] = (ndcX > -1.2f && ndcX < 1.2f &&
                     ndcY > -1.2f && ndcY < 1.2f &&
                     ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }

    // ----- Raster timing setup (measurement-first, Phase A follow-up) -
    // Both raster loops below are SERIAL. The real hot-path Step 2
    // (rasterize_iou2d_v3rs raster_mode 0) is OpenMP-parallel with
    // per-thread hitmaps + OR reduction; the vtx-squash path has never
    // been parallelized. So time(B) < time(A) HERE is the *algorithmic*
    // per-write comparison only -- necessary, NOT sufficient: a winning
    // candidate still has to survive parallelization (HANDOVER §2.5:
    // "reduce write count" schemes lose when per-write cost rises).
    // Each loop is repeated kRasterReps times and the per-rep average
    // is reported -- one pass of an ~0.1-0.3 ms loop is too noisy (cold
    // cache, scheduler jitter). The hitmap zero-fill is INSIDE the
    // timed region: it is part of producing one hitmap and is identical
    // for A and B, so it cancels in the ratio.
    using clk = std::chrono::high_resolution_clock;
    constexpr int kRasterReps = 64;
    double tA_us = 0.0, tB_us = 0.0;

    // ----- Hitmap A: triangle-bbox splat (hot-path replica) ----------
    std::vector<uint8_t> hitA((size_t)gw * (size_t)gh, 0);
    long long writesA = 0;
    {
        const auto _t0 = clk::now();
        for (int rep = 0; rep < kRasterReps; ++rep) {
            std::fill(hitA.begin(), hitA.end(), (uint8_t)0);
            writesA = 0;
            for (int ti = 0; ti < nTris; ++ti) {
                const size_t t = (size_t)ti * 3;
                const uint32_t i0 = indices[t + 0];
                const uint32_t i1 = indices[t + 1];
                const uint32_t i2 = indices[t + 2];
                if (i0 >= (uint32_t)nVerts ||
                    i1 >= (uint32_t)nVerts ||
                    i2 >= (uint32_t)nVerts) continue;
                if (!inside[i0] && !inside[i1] && !inside[i2]) continue;

                const glm::vec3& s0 = screen[i0];
                const glm::vec3& s1 = screen[i1];
                const glm::vec3& s2 = screen[i2];

                float minX = s0.x, maxX = s0.x;
                if (s1.x < minX) minX = s1.x; else if (s1.x > maxX) maxX = s1.x;
                if (s2.x < minX) minX = s2.x; else if (s2.x > maxX) maxX = s2.x;
                float minY = s0.y, maxY = s0.y;
                if (s1.y < minY) minY = s1.y; else if (s1.y > maxY) maxY = s1.y;
                if (s2.y < minY) minY = s2.y; else if (s2.y > maxY) maxY = s2.y;

                const int x0 = std::max(0,      (int)std::floor(minX) - 1);
                const int x1 = std::min(gw - 1, (int)std::floor(maxX) + 1);
                const int y0 = std::max(0,      (int)std::floor(minY) - 1);
                const int y1 = std::min(gh - 1, (int)std::floor(maxY) + 1);
                if (x0 > x1 || y0 > y1) continue;
                for (int y = y0; y <= y1; ++y) {
                    const size_t row = (size_t)y * (size_t)gw;
                    for (int x = x0; x <= x1; ++x) {
                        hitA[row + (size_t)x] = 1;
                        ++writesA;
                    }
                }
            }
        }  // for rep
        const auto _t1 = clk::now();
        tA_us = std::chrono::duration<double, std::micro>(_t1 - _t0).count()
                / (double)kRasterReps;
    }  // timed block A

    // ----- Hitmap B: vertex squash 3x3 (plain project-then-splat) ----
    std::vector<uint8_t> hitB((size_t)gw * (size_t)gh, 0);
    long long writesB = 0;
    long long n_inside_verts = 0;
    {
        const auto _t0 = clk::now();
        for (int rep = 0; rep < kRasterReps; ++rep) {
            std::fill(hitB.begin(), hitB.end(), (uint8_t)0);
            writesB = 0;
            n_inside_verts = 0;
            for (int i = 0; i < nVerts; ++i) {
                if (!inside[i]) continue;
                ++n_inside_verts;
                const int cx = (int)std::floor(screen[i].x);
                const int cy = (int)std::floor(screen[i].y);
                const int x0 = std::max(0,      cx - 1);
                const int x1 = std::min(gw - 1, cx + 1);
                const int y0 = std::max(0,      cy - 1);
                const int y1 = std::min(gh - 1, cy + 1);
                if (x0 > x1 || y0 > y1) continue;
                for (int y = y0; y <= y1; ++y) {
                    const size_t row = (size_t)y * (size_t)gw;
                    for (int x = x0; x <= x1; ++x) {
                        hitB[row + (size_t)x] = 1;
                        ++writesB;
                    }
                }
            }
        }  // for rep
        const auto _t1 = clk::now();
        tB_us = std::chrono::duration<double, std::micro>(_t1 - _t0).count()
                / (double)kRasterReps;
    }  // timed block B

    // ----- A vs B set difference -------------------------------------
    // hole      = on in A, off in B  -> the long-edge failure of squash
    // overshoot = on in B, off in A  -> squash halo spilling past bbox
    long long cells_on_A = 0, cells_on_B = 0;
    long long both = 0, hole = 0, overshoot = 0;
    const size_t N = (size_t)gw * (size_t)gh;
    for (size_t i = 0; i < N; ++i) {
        const bool a = (hitA[i] != 0);
        const bool b = (hitB[i] != 0);
        if (a) ++cells_on_A;
        if (b) ++cells_on_B;
        if (a && b)       ++both;
        else if (a && !b) ++hole;
        else if (b && !a) ++overshoot;
    }
    const double ab_iou = (both + hole + overshoot > 0)
                              ? (double)both / (double)(both + hole + overshoot)
                              : 0.0;

    // ----- Projected per-triangle max-edge-length histogram ----------
    // A hole opens when a triangle's projected edge is longer than the
    // squash stamp can bridge: the two endpoint 3x3 stamps stop
    // overlapping past ~2 cells of separation, so any edge > 2 cells is
    // a hole risk and the risk grows with length. Histogram of the
    // per-triangle MAX edge length tells us how many triangles are in
    // each risk band -- i.e. how many a Phase B adaptive subdivision
    // would have to touch, and whether that set is small (§5.2 says it
    // must be: "all triangles" subdivision loses to bbox splat).
    long long eh_le1 = 0, eh_le2 = 0, eh_le3 = 0, eh_le4 = 0;
    long long eh_le5 = 0, eh_le7 = 0, eh_le10 = 0, eh_gt10 = 0;
    double    max_edge_seen = 0.0;
    long long tris_considered = 0;
    auto edge_len = [](const glm::vec3& a, const glm::vec3& b) -> double {
        const double dx = (double)a.x - (double)b.x;
        const double dy = (double)a.y - (double)b.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    for (int ti = 0; ti < nTris; ++ti) {
        const size_t t = (size_t)ti * 3;
        const uint32_t i0 = indices[t + 0];
        const uint32_t i1 = indices[t + 1];
        const uint32_t i2 = indices[t + 2];
        if (i0 >= (uint32_t)nVerts ||
            i1 >= (uint32_t)nVerts ||
            i2 >= (uint32_t)nVerts) continue;
        if (!inside[i0] && !inside[i1] && !inside[i2]) continue;
        ++tris_considered;
        const double e0 = edge_len(screen[i0], screen[i1]);
        const double e1 = edge_len(screen[i1], screen[i2]);
        const double e2 = edge_len(screen[i2], screen[i0]);
        double em = e0;
        if (e1 > em) em = e1;
        if (e2 > em) em = e2;
        if (em > max_edge_seen) max_edge_seen = em;
        if      (em <= 1.0)  ++eh_le1;
        else if (em <= 2.0)  ++eh_le2;
        else if (em <= 3.0)  ++eh_le3;
        else if (em <= 4.0)  ++eh_le4;
        else if (em <= 5.0)  ++eh_le5;
        else if (em <= 7.0)  ++eh_le7;
        else if (em <= 10.0) ++eh_le10;
        else                 ++eh_gt10;
    }
    // Triangles that are hole-risk (max edge > 2 cells) -- the Phase B
    // subdivision candidate set.
    const long long tris_risk = eh_le3 + eh_le4 + eh_le5
                                + eh_le7 + eh_le10 + eh_gt10;

    // ----- IoU(A) and IoU(B) vs the hot-path target mask -------------
    // Same gating as rasterize_iou2d_v3rs Step 3: cache path when the
    // squash toggle is on and the cache is valid for this tuple, else
    // legacy centre-sample. Whichever it is, A and B are scored against
    // the SAME target, so the IoU delta is meaningful.
    const bool use_cache =
        g_silTargetSquashEnabled &&
        isSilTargetMaskCacheValidFor(dist_map, img_w, img_h, step) &&
        g_silTargetMaskCache.gw == gw &&
        g_silTargetMaskCache.gh == gh;

    long long interA = 0, uniA = 0, interB = 0, uniB = 0;
    long long tgt_on = 0;
    if (use_cache) {
        const uint8_t* tm = g_silTargetMaskCache.data.data();
        for (size_t i = 0; i < N; ++i) {
            const bool tt = (tm[i] != 0);
            if (tt) ++tgt_on;
            const bool a = (hitA[i] != 0);
            const bool b = (hitB[i] != 0);
            if (a || tt) ++uniA;
            if (a && tt) ++interA;
            if (b || tt) ++uniB;
            if (b && tt) ++interB;
        }
    } else {
        for (int gy = 0; gy < gh; ++gy) {
            const int ipy = gy * step + step / 2;
            const int my  = (ipy < 0) ? 0
                                     : (ipy >= img_h ? img_h - 1 : ipy);
            const size_t row_dist = (size_t)my * (size_t)img_w;
            const size_t row_hit  = (size_t)gy * (size_t)gw;
            for (int gx = 0; gx < gw; ++gx) {
                const int ipx = gx * step + step / 2;
                const int mx  = (ipx < 0) ? 0
                                         : (ipx >= img_w ? img_w - 1 : ipx);
                const bool tt = (dist_map[row_dist + (size_t)mx] < 9000.0f);
                if (tt) ++tgt_on;
                const bool a = (hitA[row_hit + (size_t)gx] != 0);
                const bool b = (hitB[row_hit + (size_t)gx] != 0);
                if (a || tt) ++uniA;
                if (a && tt) ++interA;
                if (b || tt) ++uniB;
                if (b && tt) ++interB;
            }
        }
    }
    const double iouA = (uniA > 0) ? (double)interA / (double)uniA : 0.0;
    const double iouB = (uniB > 0) ? (double)interB / (double)uniB : 0.0;

    // ----- Output ----------------------------------------------------
    std::cout << tag << " === Vertex-squash A/B diagnostic ("
              << (run_index >= 0
                      ? ("Run " + std::to_string(run_index + 1))
                      : std::string("F10 on-demand, current pose"))
              << ") ===" << std::endl;
    std::cout << tag << "   grid=" << gw << "x" << gh
              << "  step=" << step
              << "  tris=" << nTris << " (considered " << tris_considered << ")"
              << "  verts=" << nVerts << " (inside " << n_inside_verts << ")"
              << "  target_mask=" << (use_cache ? "CACHE" : "legacy-centre")
              << "  tgt_on=" << tgt_on << std::endl;
    std::cout << tag << "   A bbox-splat : cells_on=" << cells_on_A
              << "  writes=" << writesA << std::endl;
    std::cout << tag << "   B vtx-squash : cells_on=" << cells_on_B
              << "  writes=" << writesB << std::endl;
    {
        const double wr = (writesA > 0)
        ? (double)writesB / (double)writesA : 0.0;
        std::cout << std::fixed << std::setprecision(3)
                  << tag << "   writes(B)/writes(A) = " << wr
                  << "   (B is " << (wr < 1.0 ? "CHEAPER" : "more expensive")
                  << " per the raw write count; per-write cost differs, "
                     "see §2.5)" << std::endl;
    }
    {
        const double tr = (tA_us > 0.0) ? tB_us / tA_us : 0.0;
        std::cout << std::fixed << std::setprecision(1)
                  << tag << "   serial raster time: A=" << tA_us
                  << " us  B=" << tB_us
                  << " us  (per-rep avg of " << kRasterReps
                  << ", zero-fill included)" << std::endl;
        std::cout << std::setprecision(3)
                  << tag << "   time(B)/time(A) = " << tr
                  << "   [SERIAL replica: hot-path A is OpenMP-parallel, "
                     "B is not -- B must win here AND survive "
                     "parallelization (§2.5)]" << std::endl;
    }
    std::cout << std::setprecision(4)
              << tag << "   A vs B shape : both=" << both
              << "  hole(in A not B)=" << hole
              << "  overshoot(in B not A)=" << overshoot
              << "   A^B IoU=" << ab_iou << std::endl;
    std::cout << tag << "   IoU vs target: A=" << iouA
              << "   B=" << iouB
              << "   delta(B-A)=" << (iouB - iouA) << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << tag << "   --- projected per-triangle MAX edge length "
                        "(cells) ---" << std::endl;
    std::cout << tag << "     <=1 : " << eh_le1
              << "   <=2 : " << eh_le2
              << "   <=3 : " << eh_le3
              << "   <=4 : " << eh_le4 << std::endl;
    std::cout << tag << "     <=5 : " << eh_le5
              << "   <=7 : " << eh_le7
              << "   <=10 : " << eh_le10
              << "   >10 : " << eh_gt10 << std::endl;
    std::cout << std::fixed << std::setprecision(2)
              << tag << "     max edge = " << max_edge_seen << " cells"
              << "   hole-risk tris (max edge > 2) = " << tris_risk
              << " / " << tris_considered
              << " (" << (tris_considered > 0
                              ? 100.0 * (double)tris_risk / (double)tris_considered
                              : 0.0)
              << "%)" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);

    // ----- Verdict line (the §5.2 decision rule, stated explicitly) --
    {
        const bool pass_hole   = (hole == 0);
        const bool pass_writes = (writesB < writesA);
        std::cout << tag << "   VERDICT (plain vtx-squash, no subdiv): "
                  << "hole==0? " << (pass_hole ? "YES" : "NO")
                  << "   writes(B)<writes(A)? "
                  << (pass_writes ? "YES" : "NO")
                  << "   -> "
                  << ((pass_hole && pass_writes)
                          ? "would already beat bbox splat"
                          : "does NOT beat bbox splat as-is; Phase B "
                            "(subdivide the hole-risk tris) is the next "
                            "thing to diagnose")
                  << std::endl;
    }

    // ----- ASCII map: A vs B with the hole pattern marked ------------
    //   ' ' off in both        '#' on in both (agree)
    //   'H' hole: in A not B    'o' overshoot: in B not A
    std::cout << tag << "   --- ASCII A/B map (" << gw << "x" << gh
              << ", legend: ' '=off  #=both  H=hole(A\\B)  o=overshoot(B\\A))"
              << " ---" << std::endl;
    for (int y = 0; y < gh; ++y) {
        std::string line = tag + "   |";
        for (int x = 0; x < gw; ++x) {
            const size_t idx = (size_t)y * (size_t)gw + (size_t)x;
            const bool a = (hitA[idx] != 0);
            const bool b = (hitB[idx] != 0);
            char ch;
            if      (a && b)   ch = '#';
            else if (a && !b)  ch = 'H';
            else if (b && !a)  ch = 'o';
            else               ch = ' ';
            line += ch;
        }
        line += "|";
        std::cout << line << std::endl;
    }
}

// =====================================================================
// diagnoseAdaptiveSubdivOnce -- Phase B adaptive-subdivision diagnostic.
// ---------------------------------------------------------------------
// HANDOVER §5.2 Phase B. Follows the §4.3 "new diagnostic function"
// pattern: measurement-only, called from F11 (diagnoseAdaptiveSubdivV3RS
// in RegistrationActions.h). Does NOT touch optimizer state or the liver
// pose.
//
// Context (from F10 Phase A results):
//   Plain vertex squash (B) had 126 holes and time(B)/time(A)=0.207.
//   The holes come from triangles whose projected max edge > 2 cells
//   (2701/9151 = 29.52% of tris). This function (Phase B) asks:
//   "If we recursively subdivide those risk triangles until all sub-edges
//   ≤ threshold_cells, does the hole count reach 0, and does the raw
//   write budget stay below bbox splat (167k)?"
//
// Approach -- adaptive 1-to-4 midpoint subdivision:
//   For each triangle:
//     - Compute the 3 projected vertex positions.
//     - If max projected edge ≤ threshold_cells: stamp the 3 corners
//       (identical to plain vertex squash for this triangle).
//     - Else: push onto an explicit stack and recursively apply 1-to-4
//       midpoint subdivision (connect edge midpoints, producing 4
//       congruent sub-triangles) until all sub-edges ≤ threshold_cells
//       or max_depth is reached, then stamp every leaf vertex.
//   1-to-4 halves the edge length each level, so max_depth=4 handles
//   the worst case (max edge 14.2 cells: log2(14.2/2)=2.83 -> depth 3).
//
// Builds two hitmaps (same method as diagnoseVertexSquashOnce):
//   A  = triangle-bbox splat  (current hot path -- reference)
//   B' = adaptive vertex squash (stamp set from subdivision above)
//
// Reports:
//   - total_stamps (original vertices + added midpoints, with duplicates)
//   - writes(A) vs writes(B')  [raw write count including duplicates]
//   - hole (cells in A not B')  -- target is 0
//   - overshoot (cells in B' not A)
//   - A^B' IoU (shape agreement A vs B')
//   - IoU(A) and IoU(B') vs SAM2 target mask (same cache path as vsq-diag)
//   - serial timing A vs B'
//   - VERDICT: hole==0 AND writes(B') < writes(A)?
//
// Decision rule (§5.2):
//   B' is worth pursuing as a hot-path candidate iff
//     hole_count == 0  AND  writes(B') < writes(A=167k).
//   If hole_count > 0, either threshold needs lowering or large-interior
//   triangles need interior sample points too. If writes(B') >= writes(A),
//   adaptive subdivision loses to bbox splat on raw budget -- even though
//   per-write cost for vertex squash is cheaper (§2.5).
//
// Note on timing interpretation (same caveat as diagnoseVertexSquashOnce):
//   Both raster loops here are SERIAL. The hot-path step2 (bbox splat)
//   is OpenMP-parallel. So time(B')/time(A) < 1 here is the algorithmic
//   comparison only -- necessary, NOT sufficient. Winning candidates must
//   also survive the parallelization analysis in §2.5.
// =====================================================================
inline void diagnoseAdaptiveSubdivOnce(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>&  indices,
    const glm::mat4&              mvp,
    const std::vector<float>&     dist_map,
    int                            img_w,
    int                            img_h,
    int                            step,
    int                            run_index,            // -1 = F11 on-demand
    float                          threshold_cells = 2.0f,
    int                            max_depth       = 4)
{
    if (positions.empty() || indices.empty()) return;
    if (img_w <= 0 || img_h <= 0)             return;
    if (dist_map.size() != (size_t)img_w * (size_t)img_h) return;
    if (step < 1) step = 1;
    if (threshold_cells <= 0.0f) threshold_cells = 2.0f;
    if (max_depth < 1)           max_depth = 4;

    const int gw = (img_w + step - 1) / step;
    const int gh = (img_h + step - 1) / step;
    if (gw <= 0 || gh <= 0) return;

    const int nVerts = (int)positions.size();
    const int nTris  = (int)(indices.size() / 3);
    if (nVerts == 0 || nTris == 0) return;

    const float halfW   = (float)img_w * 0.5f;
    const float halfH   = (float)img_h * 0.5f;
    const float invStep = 1.0f / (float)step;
    const std::string tag = "[V3RS/asub-diag]";

    // ----- Step 1: project all vertices (serial) ----------------------
    // Store as vec2 (grid-space x,y only); the frustum test (inside[])
    // uses the full NDC coordinates from the projection.
    std::vector<glm::vec2> screen2(nVerts);
    std::vector<uint8_t>   inside(nVerts, 0);
    for (int i = 0; i < nVerts; ++i) {
        const glm::vec3& p = positions[i];
        const glm::vec4  c = mvp * glm::vec4(p.x, p.y, p.z, 1.0f);
        if (std::abs(c.w) < 1e-8f) { screen2[i] = glm::vec2(0.0f, 0.0f); continue; }
        const float ndcX = c.x / c.w;
        const float ndcY = c.y / c.w;
        const float ndcZ = c.z / c.w;
        screen2[i] = glm::vec2((ndcX + 1.0f) * halfW * invStep,
                               (1.0f - ndcY) * halfH * invStep);
        inside[i]  = (ndcX > -1.2f && ndcX < 1.2f &&
                     ndcY > -1.2f && ndcY < 1.2f &&
                     ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }

    // ----- Helper: stamp a 3x3 (1-cell halo) block at (pt) -----------
    // Matches the vtx-squash stamp in rasterize_iou2d_v3rs (raster_mode 1)
    // and in diagnoseVertexSquashOnce exactly.
    auto stamp3x3 = [&](std::vector<uint8_t>& hm, long long& writes,
                        const glm::vec2& pt) {
        const int cx = (int)std::floor(pt.x);
        const int cy = (int)std::floor(pt.y);
        const int x0 = std::max(0, cx - 1), x1 = std::min(gw - 1, cx + 1);
        const int y0 = std::max(0, cy - 1), y1 = std::min(gh - 1, cy + 1);
        if (x0 > x1 || y0 > y1) return;
        for (int y = y0; y <= y1; ++y) {
            const size_t row = (size_t)y * (size_t)gw;
            for (int x = x0; x <= x1; ++x) {
                hm[row + (size_t)x] = 1;
                ++writes;
            }
        }
    };

    // ----- Helper: squared 2D edge length -----------------------------
    auto elen2 = [](const glm::vec2& a, const glm::vec2& b) -> float {
        const float dx = a.x - b.x, dy = a.y - b.y;
        return dx * dx + dy * dy;
    };

    // ----- Build stamp list for B' via adaptive subdivision -----------
    // Collect all stamp points (with duplicates) into stamp_pts once.
    // Duplicates between adjacent triangles are intentional: they
    // produce the same inflated write count that the timing comparison
    // will measure, making it apples-to-apples with writesA.
    const float thr2 = threshold_cells * threshold_cells;

    struct SubTri { glm::vec2 v0, v1, v2; int depth; };
    std::vector<SubTri>   stk;
    stk.reserve(64);
    std::vector<glm::vec2> stamp_pts;
    stamp_pts.reserve((size_t)nVerts * 4);

    long long n_safe_tris = 0, n_risk_tris = 0;
    long long safe_stamps = 0, risk_stamps = 0;

    for (int ti = 0; ti < nTris; ++ti) {
        const size_t  t  = (size_t)ti * 3;
        const uint32_t i0 = indices[t + 0];
        const uint32_t i1 = indices[t + 1];
        const uint32_t i2 = indices[t + 2];
        if (i0 >= (uint32_t)nVerts ||
            i1 >= (uint32_t)nVerts ||
            i2 >= (uint32_t)nVerts) continue;
        if (!inside[i0] && !inside[i1] && !inside[i2]) continue;

        const glm::vec2& s0 = screen2[i0];
        const glm::vec2& s1 = screen2[i1];
        const glm::vec2& s2 = screen2[i2];
        const float me2 = std::max({elen2(s0, s1), elen2(s1, s2), elen2(s2, s0)});

        if (me2 <= thr2) {
            // Safe triangle: stamp 3 corners (same as plain vertex squash).
            stamp_pts.push_back(s0);
            stamp_pts.push_back(s1);
            stamp_pts.push_back(s2);
            ++n_safe_tris;
            safe_stamps += 3;
        } else {
            // Risk triangle: adaptive 1-to-4 midpoint subdivision.
            ++n_risk_tris;
            const long long before = (long long)stamp_pts.size();
            stk.clear();
            stk.push_back({s0, s1, s2, 0});
            while (!stk.empty()) {
                const SubTri st = stk.back(); stk.pop_back();
                const float sub_me2 = std::max({elen2(st.v0, st.v1),
                                                elen2(st.v1, st.v2),
                                                elen2(st.v2, st.v0)});
                if (sub_me2 <= thr2 || st.depth >= max_depth) {
                    // Leaf: stamp the 3 sub-triangle vertices.
                    stamp_pts.push_back(st.v0);
                    stamp_pts.push_back(st.v1);
                    stamp_pts.push_back(st.v2);
                } else {
                    // Subdivide into 4 congruent sub-triangles by connecting
                    // edge midpoints (standard 1-to-4 midpoint subdivision).
                    const glm::vec2 m01 = (st.v0 + st.v1) * 0.5f;
                    const glm::vec2 m12 = (st.v1 + st.v2) * 0.5f;
                    const glm::vec2 m20 = (st.v2 + st.v0) * 0.5f;
                    stk.push_back({st.v0, m01,  m20,  st.depth + 1});
                    stk.push_back({m01,  st.v1, m12,  st.depth + 1});
                    stk.push_back({m20,  m12,  st.v2, st.depth + 1});
                    stk.push_back({m01,  m12,  m20,  st.depth + 1});
                }
            }
            risk_stamps += (long long)stamp_pts.size() - before;
        }
    }
    const long long total_stamps = (long long)stamp_pts.size();

    // ----- Timed raster loops -----------------------------------------
    using clk = std::chrono::high_resolution_clock;
    constexpr int kRasterReps = 64;
    const size_t N = (size_t)gw * (size_t)gh;

    // ---- Hitmap A: triangle-bbox splat (hot-path replica) ------------
    std::vector<uint8_t> hitA(N, 0);
    long long writesA = 0;
    double    tA_us   = 0.0;
    {
        const auto t0 = clk::now();
        for (int rep = 0; rep < kRasterReps; ++rep) {
            std::fill(hitA.begin(), hitA.end(), (uint8_t)0);
            writesA = 0;
            for (int ti = 0; ti < nTris; ++ti) {
                const size_t  t  = (size_t)ti * 3;
                const uint32_t i0 = indices[t + 0];
                const uint32_t i1 = indices[t + 1];
                const uint32_t i2 = indices[t + 2];
                if (i0 >= (uint32_t)nVerts ||
                    i1 >= (uint32_t)nVerts ||
                    i2 >= (uint32_t)nVerts) continue;
                if (!inside[i0] && !inside[i1] && !inside[i2]) continue;
                const glm::vec2& s0 = screen2[i0];
                const glm::vec2& s1 = screen2[i1];
                const glm::vec2& s2 = screen2[i2];
                float minX = s0.x, maxX = s0.x;
                if (s1.x < minX) minX = s1.x; else if (s1.x > maxX) maxX = s1.x;
                if (s2.x < minX) minX = s2.x; else if (s2.x > maxX) maxX = s2.x;
                float minY = s0.y, maxY = s0.y;
                if (s1.y < minY) minY = s1.y; else if (s1.y > maxY) maxY = s1.y;
                if (s2.y < minY) minY = s2.y; else if (s2.y > maxY) maxY = s2.y;
                const int x0 = std::max(0,      (int)std::floor(minX) - 1);
                const int x1 = std::min(gw - 1, (int)std::floor(maxX) + 1);
                const int y0 = std::max(0,      (int)std::floor(minY) - 1);
                const int y1 = std::min(gh - 1, (int)std::floor(maxY) + 1);
                if (x0 > x1 || y0 > y1) continue;
                for (int y = y0; y <= y1; ++y) {
                    const size_t row = (size_t)y * (size_t)gw;
                    for (int x = x0; x <= x1; ++x) {
                        hitA[row + (size_t)x] = 1;
                        ++writesA;
                    }
                }
            }
        }
        tA_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count()
                / (double)kRasterReps;
    }

    // ---- Hitmap B': adaptive vertex squash ---------------------------
    std::vector<uint8_t> hitBp(N, 0);
    long long writesBp = 0;
    double    tBp_us   = 0.0;
    {
        const auto t0 = clk::now();
        for (int rep = 0; rep < kRasterReps; ++rep) {
            std::fill(hitBp.begin(), hitBp.end(), (uint8_t)0);
            writesBp = 0;
            for (const glm::vec2& pt : stamp_pts) {
                stamp3x3(hitBp, writesBp, pt);
            }
        }
        tBp_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count()
                 / (double)kRasterReps;
    }

    // ----- Cell comparison: hole / overshoot / both -------------------
    long long cells_on_A = 0, cells_on_Bp = 0;
    long long hole = 0, overshoot = 0, both = 0;
    for (size_t i = 0; i < N; ++i) {
        const bool a  = (hitA[i]  != 0);
        const bool bp = (hitBp[i] != 0);
        if (a)  ++cells_on_A;
        if (bp) ++cells_on_Bp;
        if (a && bp)   ++both;
        if (a && !bp)  ++hole;
        if (!a && bp)  ++overshoot;
    }
    const double ab_iou = (both + hole + overshoot > 0)
                              ? (double)both / (double)(both + hole + overshoot)
                              : 0.0;

    // ----- IoU(A) and IoU(B') vs SAM2 target -------------------------
    // Same gating as rasterize_iou2d_v3rs Step 3 and diagnoseVertexSquashOnce:
    // use the squashed target cache when valid, else legacy centre-sample.
    const bool use_cache =
        g_silTargetSquashEnabled &&
        isSilTargetMaskCacheValidFor(dist_map, img_w, img_h, step) &&
        g_silTargetMaskCache.gw == gw &&
        g_silTargetMaskCache.gh == gh;

    long long interA = 0, uniA = 0, interBp = 0, uniBp = 0;
    if (use_cache) {
        const uint8_t* tm = g_silTargetMaskCache.data.data();
        for (size_t i = 0; i < N; ++i) {
            const bool tt = (tm[i] != 0);
            const bool a  = (hitA[i]  != 0);
            const bool bp = (hitBp[i] != 0);
            if (a  || tt) ++uniA;
            if (a  && tt) ++interA;
            if (bp || tt) ++uniBp;
            if (bp && tt) ++interBp;
        }
    } else {
        for (int gy = 0; gy < gh; ++gy) {
            const int ipy = gy * step + step / 2;
            const int my  = (ipy < 0) ? 0 : (ipy >= img_h ? img_h - 1 : ipy);
            const size_t row_d = (size_t)my * (size_t)img_w;
            const size_t row_h = (size_t)gy * (size_t)gw;
            for (int gx = 0; gx < gw; ++gx) {
                const int ipx = gx * step + step / 2;
                const int mx  = (ipx < 0) ? 0 : (ipx >= img_w ? img_w - 1 : ipx);
                const bool tt = (dist_map[row_d + (size_t)mx] < 9000.0f);
                const bool a  = (hitA[row_h  + (size_t)gx] != 0);
                const bool bp = (hitBp[row_h + (size_t)gx] != 0);
                if (a  || tt) ++uniA;
                if (a  && tt) ++interA;
                if (bp || tt) ++uniBp;
                if (bp && tt) ++interBp;
            }
        }
    }
    const double iouA  = (uniA  > 0) ? (double)interA  / (double)uniA  : 0.0;
    const double iouBp = (uniBp > 0) ? (double)interBp / (double)uniBp : 0.0;

    // ----- Console output ---------------------------------------------
    const std::string ctx = (run_index >= 0)
                                ? ("Run " + std::to_string(run_index + 1))
                                : std::string("F11 on-demand, current pose");

    std::cout << std::fixed << std::setprecision(4);
    std::cout << tag << " === Adaptive Subdivision A/B' diagnostic ("
              << ctx << ") ===" << std::endl;
    std::cout << tag << "   grid=" << gw << "x" << gh
              << "  step=" << step
              << "  tris=" << nTris << " (considered " << (n_safe_tris + n_risk_tris) << ")"
              << "  verts=" << nVerts
              << "  target_mask=" << (use_cache ? "CACHE" : "centre-sample")
              << std::endl;
    std::cout << tag << "   threshold=" << std::setprecision(2)
              << threshold_cells << " cells"
              << "  max_depth=" << max_depth << std::endl;
    std::cout << tag << "   safe_tris (max_edge <= thr) = "
              << n_safe_tris << "  safe_stamps=" << safe_stamps << std::endl;
    std::cout << tag << "   risk_tris (max_edge >  thr) = "
              << n_risk_tris << " (" << std::setprecision(2)
              << 100.0 * (double)n_risk_tris / (double)(n_safe_tris + n_risk_tris)
              << "%)  risk_stamps=" << risk_stamps << std::endl;
    std::cout << tag << "   total_stamps = " << total_stamps
              << "  (original " << (n_safe_tris * 3) << " safe-vert stamps + "
              << risk_stamps << " from subdivision)" << std::endl;
    std::cout << std::setprecision(4);
    std::cout << tag << "   A  (bbox splat)    : cells_on=" << cells_on_A
              << "  writes=" << writesA
              << "  IoU(vs tgt)=" << iouA << std::endl;
    std::cout << tag << "   B' (adaptive subdiv): cells_on=" << cells_on_Bp
              << "  writes=" << writesBp
              << "  IoU(vs tgt)=" << iouBp << std::endl;
    std::cout << tag << "   writes(B')/writes(A) = "
              << std::setprecision(3)
              << (writesA > 0 ? (double)writesBp / (double)writesA : 0.0)
              << "  (B' is "
              << (writesBp < writesA ? "CHEAPER" : "MORE EXPENSIVE")
              << " per raw write count)" << std::endl;
    std::cout << tag << "   serial raster time: A=" << std::setprecision(1)
              << tA_us << " us  B'=" << tBp_us << " us"
              << "  (per-rep avg of " << kRasterReps << ", zero-fill included)"
              << std::endl;
    std::cout << tag << "   time(B')/time(A) = " << std::setprecision(3)
              << (tA_us > 0.0 ? tBp_us / tA_us : 0.0)
              << "   [SERIAL replica: hot-path A is OpenMP-parallel (§2.5)]"
              << std::endl;
    std::cout << tag << "   A vs B' shape: both=" << both
              << "  hole(in A not B')=" << hole
              << "  overshoot(in B' not A)=" << overshoot
              << "   A^B' IoU=" << std::setprecision(4) << ab_iou << std::endl;
    std::cout << tag << "   IoU vs target: A=" << iouA
              << "   B'=" << iouBp
              << "   delta(B'-A)=" << std::showpos << (iouBp - iouA)
              << std::noshowpos << std::endl;

    // Verdict
    const bool hole_ok   = (hole == 0);
    const bool writes_ok = (writesBp < writesA);
    std::cout << tag << "   VERDICT (hole==0? "
              << (hole_ok   ? "YES" : "NO")
              << "   writes(B')<writes(A)? "
              << (writes_ok ? "YES" : "NO")
              << ") -> ";
    if (hole_ok && writes_ok) {
        std::cout << "CANDIDATE for hot-path replacement. "
                     "Proceed to parallelization analysis (HANDOVER §2.5)."
                  << std::endl;
    } else if (hole_ok && !writes_ok) {
        std::cout << "Hole-free but write budget EXCEEDS bbox splat ("
                  << writesBp << " vs " << writesA << "). "
                  << "Try a larger threshold or accept the budget cost if "
                     "per-write timing (§2.5) compensates."
                  << std::endl;
    } else if (!hole_ok && writes_ok) {
        std::cout << hole << " holes remain. "
                  << "Lower threshold_cells (try 1.5) or add interior "
                     "sampling for large triangles."
                  << std::endl;
    } else {
        std::cout << hole << " holes remain AND write budget exceeds bbox ("
                  << writesBp << " vs " << writesA << "). "
                  << "Adaptive subdivision does not beat bbox splat; "
                     "see HANDOVER §5.2 for next options."
                  << std::endl;
    }

    // ----- ASCII map: A vs B' with hole pattern ('H') ----------------
    //   ' ' off in both  '#' on in both  'H' hole(A\B')  'o' overshoot(B'\A)
    std::cout << tag << "   --- ASCII A/B' map (" << gw << "x" << gh
              << ", legend: ' '=off  #=both  H=hole(A\\B')  o=overshoot(B'\\A))"
              << " ---" << std::endl;
    for (int y = 0; y < gh; ++y) {
        std::string line = tag + "   |";
        for (int x = 0; x < gw; ++x) {
            const size_t idx = (size_t)y * (size_t)gw + (size_t)x;
            const bool a  = (hitA[idx]  != 0);
            const bool bp = (hitBp[idx] != 0);
            char ch;
            if      (a && bp)   ch = '#';
            else if (a && !bp)  ch = 'H';
            else if (!a && bp)  ch = 'o';
            else                ch = ' ';
            line += ch;
        }
        line += "|";
        std::cout << line << std::endl;
    }
    std::cout << std::defaultfloat << std::setprecision(6);
}

// =====================================================================
// evaluate_one_v3rs_silhouette -- unconditional IoU penalty (Phase 3).
// ---------------------------------------------------------------------
// Phase 3 design:
//
//   cost = RMSE_W + lambda_sil * (1 - IoU2D)
//
// History:
//   Phase 1: cost = RMSE_W + lambda * (1-IoU)   [every-Nth sampling]
//     → scale-cheating: optimizer drove scale to ~1.05 to lower RMSE_W
//       at the cost of silhouette degradation.
//   Phase 2: cost = RMSE_W + lambda * (1-IoU) * |scale-1|
//     → Fixed scale-cheating, but accidentally broke IoU signalling:
//       scale is always constrained to 1.0000 in practice, so the
//       |scale-1| factor kept the IoU term permanently zero.
//       Result: optimizer only saw RMSE_W, and source silhouette
//       expanded to contain the SAM2 target (containment failure).
//   Phase 3 (current): remove |scale-1| → unconditional IoU term.
//     The IoU penalty now always reaches the CMA-ES cost landscape,
//     preventing both scale-cheating (IoU degrades immediately if
//     scale inflates the mesh) and the containment failure mode.
//
// eval_interval remains 1 (per-eval, established in Phase 2).
// Per-eval cost is ~3.75 ms at step=8, giving ~50 s/session.
//
// Caller contract (gated by run_one_bipop_v3rs):
//   - lambda_sil > 0
//   - sil_dist_map_2d, sil_indices populated
//   - sil_img_w / sil_img_h match dist_map dimensions
//   - liver_full_positions points to the ORIGINAL (untransformed)
//     full mesh; the eval applies M_world internally via the MVP.
//   - view_proj_jit pre-built: sil_proj * sil_view * M_jit (per Run).
//
// Outputs (besides return value):
//   matched_out      : V3R-W matched count (drives the bad-eval gate).
//   rmse_w_out       : RMSE_W alone (for diagnostics).
//   iou2d_out        : IoU2D in [0,1] when computed, -1 when skipped.
//   iou_computed_out : true if IoU was folded into cost on this call.
// =====================================================================
inline float evaluate_one_v3rs_silhouette(
    // ----- RMSE_W path (V3R-W, unchanged) ----------------------------
    const CmaesRefine::EvalContextStaticV3& S,
    CmaesRefine::EvalContextScratchV3&      W,
    const CmaesRefine::SRTParamsV3&         srt,
    const std::vector<uint8_t>&             is_rim_src_voxel,
    const std::vector<uint8_t>&             is_rim_tgt_voxel,
    float                                    beta,
    // ----- Phase 2 silhouette x scale inputs -------------------------
    long long                                eval_index,
    int                                      eval_interval,
    float                                    lambda_sil,
    const std::vector<glm::vec3>&           liver_full_positions,
    const std::vector<uint32_t>&            sil_indices,
    const glm::mat4&                         view_proj_jit,
    const std::vector<float>&               sil_dist_map_2d,
    int                                      sil_img_w,
    int                                      sil_img_h,
    int                                      sil_raster_step,
    // ----- Outputs ---------------------------------------------------
    int&                                     matched_out,
    float&                                   rmse_w_out,
    float&                                   iou2d_out,
    bool&                                    iou_computed_out,
    // ----- Optional per-eval timing accumulator ----------------------
    // When non-null, every component's wall-clock time (microseconds)
    // is added in-place. Caller resets the struct once per Run before
    // the first call; accumulation across the 1350 evals gives the
    // total per-Run breakdown the summary log prints.
    EvalTimingV3RS*                          timing_acc = nullptr,
    // ----- Optional instrument occlusion mask (forwarded as-is) -----
    // When `sil_instrument_dist_map_2d` is null or empty/mismatched,
    // the IoU rasterizer falls through to its pre-feature behaviour
    // -- this argument is byte-identically additive for callers that
    // don't supply it.
    const std::vector<float>*                sil_instrument_dist_map_2d = nullptr,
    float                                    sil_instrument_thresh_px   = 0.0f,
    // ----- [NEW V3RS-4] Asymmetric / rim-silhouette penalties --------
    // Each lambda controls its own term; setting both to 0 reproduces
    // the Phase 3 behaviour exactly (cost = RMSE_W + lambda_sil*(1-IoU)).
    float                                    lambda_out_in       = 0.0f,
    float                                    lambda_rim_sil_in   = 0.0f,
    float                                    rim_sil_max_px_in   = 100.0f,
    // Optional per-eval aggregates so the wrapper / per-Run summary
    // can compute averages for diagnostics. Null = no overhead.
    float*                                   outside_ratio_out   = nullptr,
    float*                                   rim_sil_loss_out    = nullptr,
    // ----- [NEW V3RS-RIM-ANAT] Per-vertex anatomical RIM flag ---------
    // When non-null and size-matched to liver_full_positions, rim_sil
    // is computed in ANATOMIC mode (over projected RIM vertices) rather
    // than the per-cell raster-boundary mode. Forwarded directly to
    // rasterize_iou2d_v3rs. Pass as nullptr (default) for legacy.
    const std::vector<uint8_t>*              is_rim_anatomic_per_vertex_in = nullptr,
    // [V3I] Pure-IoU mode. Default false => unchanged V3RS blend. When
    // true the returned cost is exactly (1 - IoU2D); RMSE_W is still
    // computed (matched gate / diagnostics) but excluded from the cost.
    bool                                     pure_iou_mode = false)
{
    using namespace CmaesRefine;
    using clk = std::chrono::high_resolution_clock;

    if (timing_acc) timing_acc->n_total_evals += 1;

    // ----- Step 1: RMSE_W via V3R-W (single source of truth) ---------
    // When is_rim_src_voxel / is_rim_tgt_voxel are empty (typical for
    // V3RS with beta=0), evaluate_one_v3r_weighted falls back to the
    // plain V3 RMSE -- same behaviour as Ctrl+G with beta=0.
    const auto t_rmse_a = timing_acc ? clk::now() : clk::time_point{};
    const float rmse_w = CmaesRefineV3R::evaluate_one_v3r_weighted(
        S, W, srt, is_rim_src_voxel, is_rim_tgt_voxel, beta, matched_out);
    if (timing_acc) {
        const auto t_rmse_b = clk::now();
        timing_acc->eval_rmse_w_us +=
            std::chrono::duration<double, std::micro>(t_rmse_b - t_rmse_a).count();
    }
    rmse_w_out       = rmse_w;
    iou2d_out        = -1.0f;
    iou_computed_out = false;

    // ----- Step 2: gate the IoU2D fold -------------------------------
    // Skip when disabled (interval <= 0), or this eval isn't on the
    // sampling boundary, or any required input is missing.
    const bool want_iou =
        (eval_interval > 0) &&
        (eval_index % (long long)eval_interval == 0) &&
        ((lambda_sil > 0.0f) || pure_iou_mode) &&
        !liver_full_positions.empty() &&
        !sil_indices.empty() &&
        !sil_dist_map_2d.empty() &&
        (sil_img_w > 0) && (sil_img_h > 0);

    if (!want_iou) return rmse_w;

    // ----- Step 3: build M_srt with the SAME centroid V3R-W used ----
    // S.centroid is the centroid of the JITTERED voxel cloud. Phase E
    // assembles M_world = build_srt_matrix_v3(best_srt, c_post) * M_jit
    // where c_post == S.centroid, then applies M_world to the original
    // FULL mesh. To stay numerically consistent with that assembly,
    // we use the same M_srt here.
    const auto t_other_a = timing_acc ? clk::now() : clk::time_point{};
    const glm::mat4 M_srt    = build_srt_matrix_v3(srt, S.centroid);
    const glm::mat4 full_mvp = view_proj_jit * M_srt;
    if (timing_acc) {
        const auto t_other_b = clk::now();
        timing_acc->eval_other_us +=
            std::chrono::duration<double, std::micro>(t_other_b - t_other_a).count();
    }

    // ----- Step 4: rasterize IoU2D ----------------------------------
    // Per-step micro-timers are only requested if the caller wants
    // them (timing_acc != null). nullptr fallback is the hot path
    // when timing is off, with zero clock-call overhead.
    //
    // Instrument occlusion: forwarded as-is to the rasterizer. When
    // `sil_instrument_dist_map_2d` is null or empty/mismatched, the
    // rasterizer treats it as disabled and the IoU is computed
    // byte-identically to the pre-feature path.
    double step1_us = 0.0, step2_us = 0.0, step3_us = 0.0;
    // [NEW V3RS-4] Outputs for the asymmetric / rim-sil penalties.
    // Pointers are passed only when the corresponding lambda > 0 so
    // the rasterizer skips the extra work otherwise.
    float outside_ratio = 0.0f;
    float rim_sil_loss  = 0.0f;
    const bool want_outside = (lambda_out_in     > 0.0f);
    const bool want_rim_sil = (lambda_rim_sil_in > 0.0f);
    const auto t_sil_a = timing_acc ? clk::now() : clk::time_point{};
    const float iou = rasterize_iou2d_v3rs(
        liver_full_positions, sil_indices,
        full_mvp, sil_dist_map_2d,
        sil_img_w, sil_img_h, sil_raster_step,
        /*out_hitmap*/      nullptr,
        /*out_target_mask*/ nullptr,
        /*out_gw*/          nullptr,
        /*out_gh*/          nullptr,
        timing_acc ? &step1_us : nullptr,
        timing_acc ? &step2_us : nullptr,
        timing_acc ? &step3_us : nullptr,
        /*raster_mode*/     0,
        sil_instrument_dist_map_2d,
        sil_instrument_thresh_px,
        // [NEW V3RS-4] outside_ratio + rim_sil_loss outputs
        want_outside ? &outside_ratio : nullptr,
        want_rim_sil ? &rim_sil_loss  : nullptr,
        rim_sil_max_px_in,
        // [NEW V3RS-RIM-ANAT] Per-vertex anatomical RIM flag (forwarded).
        // When non-null, rasterize_iou2d_v3rs switches rim_sil_loss to
        // ANATOMIC mode. The cell-mask output is unused in the hot loop
        // (we only need the scalar penalty), so pass nullptr for it.
        want_rim_sil ? is_rim_anatomic_per_vertex_in : nullptr,
        /*out_rim_cell_mask*/ nullptr);
    if (timing_acc) {
        const auto t_sil_b = clk::now();
        timing_acc->eval_sil_total_us +=
            std::chrono::duration<double, std::micro>(t_sil_b - t_sil_a).count();
        timing_acc->eval_sil_proj_us  += step1_us;
        timing_acc->eval_sil_splat_us += step2_us;
        timing_acc->eval_sil_iou_us   += step3_us;
        timing_acc->n_iou_evals       += 1;
    }

    iou2d_out        = iou;
    iou_computed_out = true;
    if (outside_ratio_out) *outside_ratio_out = outside_ratio;
    if (rim_sil_loss_out)  *rim_sil_loss_out  = rim_sil_loss;

    // ----- Step 5: composite penalty (Phase 3 + optional NEW terms) ---
    // Phase 3 baseline: cost = RMSE_W + lambda_sil * (1 - IoU2D)
    // [NEW V3RS-4] Two opt-in one-sided terms stacked on top:
    //   + lambda_out_in     * outside_ratio    (mask-expansion brake)
    //   + lambda_rim_sil_in * rim_sil_loss     (rim-to-rim alignment)
    // Each lambda is independent; both 0 → Phase 3 byte-identical.
    const auto t_pen_a = timing_acc ? clk::now() : clk::time_point{};
    // [V3I] Pure-IoU mode: drop RMSE_W / outside / rim_sil entirely so the
    //   inner objective is exactly (1 - IoU2D). Default path (pure_iou_mode
    //   == false) is the unchanged V3RS blend.
    const float cost = pure_iou_mode
                     ? (1.0f - iou)
                     : (rmse_w
                        + lambda_sil         * (1.0f - iou)
                        + lambda_out_in      * outside_ratio
                        + lambda_rim_sil_in  * rim_sil_loss);
    if (timing_acc) {
        const auto t_pen_b = clk::now();
        timing_acc->eval_other_us +=
            std::chrono::duration<double, std::micro>(t_pen_b - t_pen_a).count();
    }
    return cost;
}

// =====================================================================
// run_one_bipop_v3rs -- FORK of CmaesRefineV3R::run_one_bipop_v3r.
// ---------------------------------------------------------------------
// Differences from V3R (marked [V3RS-DIFF]):
//   - params type is ParamsV3RS (adds silhouette fields)
//   - eval dispatch adds a 3rd branch for evaluate_one_v3rs_silhouette
//     when params.lambda_sil > 0; otherwise behaviour mirrors V3R exactly.
//   - Per-run silhouette aggregates returned via out-parameters.
// Everything else is verbatim from V3R.
//
// Original V3R comment block:
// ---------------------------------------------------------------------
// Differences from V3 (in-line as "[V3RS]"):
//   - Step 2 calls CmaesRefineV3R::build_eval_context_v3r (subset-aware).
//   - Step 5's UPDATE_INTERVAL refresh calls CmaesRefineV3R::rebuild_correspondences_v3r.
//   - Step 6 dispatches to compute_full_rmse_local (V3, full-vertex)
//     OR a future subset-RMSE function based on
//     params.full_rmse_use_subset. At S4 only the V3 path is wired up.
//   - Log prefix "[V3]" -> "[V3RS]" throughout for log-diff clarity.
// All other code (jitter SRT, CMA-ES init, main loop, time logging)
// is verbatim from V3.
// =====================================================================
inline void run_one_bipop_v3rs(
    CmaesRefine::RunContext& rc,
    const ParamsV3RS&        params,
    double&                  sil_sum_loss_out,
    long long&               sil_sum_vis_out,
    long long&               sil_eval_count_out)
{
    using namespace CmaesRefine;

    // ----- 0. Validate inputs (verbatim from V3) ----------------------
    if (!rc.liver_voxel_positions || !rc.liver_full_positions
        || !rc.tgt_voxel_points    || !rc.tgt_full_points) {
        std::cerr << "[V3RS] run_one_bipop: null input pointer(s); "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "NullInput";
        return;
    }
    if (rc.liver_voxel_positions->empty() || rc.tgt_voxel_points->empty()) {
        std::cerr << "[V3RS] run_one_bipop: empty liver_voxel or tgt_voxel; "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "EmptyInput";
        return;
    }

    // ----- 1. Apply jitter to the VOXEL snapshot ---------------------
    // [V3R Issue 1 案A]: subset 縮小モード (非 QUAD_ALL) では post_jitter
    // 後に subset 全点が tgt KDTree の max_dist を超えて飛び (matched=0)、
    // CMA-ES が Gen 0 で best=0 を記録 → TolFun で即終了する現象が発生
    // (HANDOVER V3 §3.1)。これを防ぐため、matched 不足を検出したら
    // jitter を 0.5x で縮小して最大 3 回リトライし、それでも不十分なら
    // identity (jitter なし) で起点とする。QUAD_ALL では byte-identical
    // 契約を絶対に壊さないため retry を一切発動させない (二重ガード)。
    const glm::vec3 c_jitter = compute_centroid_v3(*rc.liver_voxel_positions);

    // ----- 1.5. Per-Step timing (verbatim from V3) --------------------
    auto step_now    = []{ return std::chrono::high_resolution_clock::now(); };
    using step_ms    = std::chrono::duration<double, std::milli>;
    const auto t_step_t0 = step_now();

    // ----- 2. [V3RS] Apply jitter + build eval context (with retry) ----
    SRTParamsV3            jitter_used = rc.jitter;   // rng 由来の値から開始
    std::vector<glm::vec3> liver_after_jitter;
    std::vector<glm::vec3> liver_normals_after_jitter;
    int                    post_jitter_matched = 0;

    const bool is_quad_all =
        (params.quadrant_mask == LiverLeftRightLabel::QUAD_ALL);
    // 二重ガードの 1 つ目: 閾値。
    // QUAD_ALL の subset_size = full voxel (~4076) → min_required ~204、
    // 実機で post_jitter_matched は数千レベルなので絶対に閾値割れしない。
    const int min_required_for_retry = std::max(
        10,
        (int)(params.subset_idx_voxel.size() * 0.05));

    int   jitter_retry = 0;
    float sigma_factor = 1.0f;
    while (true) {
        const glm::mat4 M_jitter_local =
            build_srt_matrix_v3(jitter_used, c_jitter);
        liver_after_jitter.clear();
        apply_srt_to_points(M_jitter_local,
                            *rc.liver_voxel_positions, liver_after_jitter);
        liver_normals_after_jitter.clear();   // V3 と同じく空のまま渡す

        rc.ctx = CmaesRefineV3R::build_eval_context_v3r(
            liver_after_jitter,
            liver_normals_after_jitter,
            *rc.tgt_voxel_points,
            params,
            rc.init_matched,
            params.subset_idx_voxel,
            &post_jitter_matched);   // V3R: matched 数を取得

        // 二重ガードの 2 つ目: QUAD_ALL では retry 厳禁。
        if (is_quad_all) break;

        // matched 十分なら break。
        if (post_jitter_matched >= min_required_for_retry) break;

        // 既に 3 回リトライ済みなら break (fallback で抜ける)。
        if (jitter_retry >= 3) break;

        // jitter を 0.5x に縮小して再試行。tx/ty/tz/rx/ry/rz_deg は
        // 単純に半減、scale は identity (1.0) との中点に寄せる。
        sigma_factor *= 0.5f;
        jitter_used.tx     *= 0.5f;
        jitter_used.ty     *= 0.5f;
        jitter_used.tz     *= 0.5f;
        jitter_used.rx_deg *= 0.5f;
        jitter_used.ry_deg *= 0.5f;
        jitter_used.rz_deg *= 0.5f;
        jitter_used.scale   = 1.0f + (jitter_used.scale - 1.0f) * 0.5f;
        jitter_retry++;
        std::cerr << "[V3RS] jitter retry " << jitter_retry
                  << "/3: sigma_factor=" << sigma_factor
                  << "  post_jitter_matched=" << post_jitter_matched
                  << " < " << min_required_for_retry
                  << "  (Q=" << LiverLeftRightLabel::quadrantMaskString(
                         params.quadrant_mask) << ")"
                  << std::endl;
    }

    // 3 回リトライしても依然 matched 不足なら identity (jitter なし) で起点
    // とする。これは "no jitter" run と同等で、最低限 HemiAuto 後の pose
    // から BIPOP を始められることを保証する。
    if (!is_quad_all
        && jitter_retry >= 3
        && post_jitter_matched < min_required_for_retry) {
        std::cerr << "[V3RS] WARNING: jitter retry exhausted, "
                  << "starting from un-jittered pose"
                  << "  (Q=" << LiverLeftRightLabel::quadrantMaskString(
                         params.quadrant_mask) << ")"
                  << std::endl;
        jitter_used = SRTParamsV3{};                       // identity
        liver_after_jitter = *rc.liver_voxel_positions;    // jitter なし
        liver_normals_after_jitter.clear();
        rc.ctx = CmaesRefineV3R::build_eval_context_v3r(
            liver_after_jitter,
            liver_normals_after_jitter,
            *rc.tgt_voxel_points,
            params,
            rc.init_matched,
            params.subset_idx_voxel,
            &post_jitter_matched);
    }

    // 重要: 後段の compute_full_rmse_local (line ~713) と driver の
    // best_jitter capture (line ~993) が rc.jitter を参照するため、
    // retry/fallback で実際に使った値を rc.jitter に書き戻す。
    // QUAD_ALL では retry が一切発動しないので jitter_used == rc.jitter
    // となり、書き戻しても V3 と byte-identical のまま。
    rc.jitter = jitter_used;

    const auto t_step_t1 = step_now();
    const double t_step_build_eval =
        step_ms(t_step_t1 - t_step_t0).count();

    if (rc.ctx.base_positions.empty() || rc.ctx.tgt_points.empty()) {
        std::cerr << "[V3RS] run_one_bipop: build_eval_context failed; "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "EmptyContext";
        return;
    }

    if (params.verbose) {
        std::cout << "[V3RS] Run " << (rc.run_index + 1)
        << "  " << (rc.is_local_regime ? "Local" : "Global")
        << "  sigma0=" << rc.sigma0
        << "  cma_seed=" << rc.cma_seed << std::endl;
    }

    // ----- 3. cmaes_init + deterministic srand (verbatim from V3) -----
    const int DIM = 7;
    double lb[DIM], ub[DIM], xstart[DIM];
    for (int d = 0; d < DIM; d++) {
        lb[d] = -1.0; ub[d] = 1.0; xstart[d] = 0.0;
    }
    cmaes_t* evo = cmaes_init(DIM, xstart, rc.sigma0,
                              params.lambda, lb, ub);
    if (rc.cma_seed != 0) {
        srand(rc.cma_seed);
        if (params.verbose) {
            std::cout << "[V3RS] Deterministic seed: " << rc.cma_seed
                      << std::endl;
        }
    }

    // ----- 4. CMA-ES state (verbatim from V3) -------------------------
    double best_x[DIM] = {0,0,0,0,0,0,0};
    float  best_rmse   = rc.ctx.initial_rmse;

    std::vector<EvalContextScratchV3> scratch_pool(1);

    // ----- 4.5 [V3RS Phase 1] Pre-build per-Run silhouette MVP --------
    // For the in-loop IoU2D term, the full mesh is rasterized at the
    // AR-fixed camera. The world transform per eval is M_world = M_srt
    // * M_jit, where M_jit is constant for this Run (the jitter we
    // just applied + retried). We factor that out: precompute
    // view_proj_jit = sil_proj * sil_view * M_jit ONCE here, and the
    // per-eval cost shrinks to one extra mat4*mat4 plus rasterize.
    //
    // M_jit uses c_jitter (centroid of UN-jittered voxel cloud) as
    // pivot -- the same value passed to build_eval_context above. We
    // rebuild the matrix from jitter_used here (rather than capturing
    // M_jitter_local from inside the retry loop) so that the fallback
    // "identity" path produces a clean identity M_jit too.
    //
    // The sil_active gate is also computed once: as long as all the
    // required Phase 1 inputs are populated and lambda_sil > 0, every
    // eval will check the eval_index % eval_interval gate inside the
    // evaluator. Empty sil_indices or empty dist_map -> sil inactive
    // -> path falls back to weighted/plain V3R behaviour.
    const bool sil_inputs_valid =
        (params.lambda_sil > 0.0f) &&
        (params.sil_eval_interval > 0) &&
        !params.sil_dist_map_2d.empty() &&
        !params.sil_indices.empty() &&
        !rc.liver_full_positions->empty() &&
        (params.sil_img_w > 0) && (params.sil_img_h > 0);

    glm::mat4 view_proj_jit(1.0f);
    if (sil_inputs_valid) {
        const glm::mat4 M_jit_final =
            build_srt_matrix_v3(jitter_used, c_jitter);
        view_proj_jit = params.sil_proj * params.sil_view * M_jit_final;
    }

    // Per-Run eval counter for the every-Nth gate. Starts at 0 so the
    // first eval (k=0, gen=0) does fire IoU2D, giving an immediate
    // baseline measurement at Gen 0.
    long long sil_eval_index = 0;

    // Per-Run timing accumulator. Filled by evaluate_one_v3rs_silhouette
    // on every call; reported in the Run-summary log alongside the
    // existing coarse "evaluate_one (sum)" line. Only active on the
    // sil_active branch; weighted_path / plain_v3 paths leave it at 0.
    EvalTimingV3RS run_timing;

    // ----- 4.6 [V3RS diag] one-shot raster coverage histogram --------
    // Runs ONCE per session (Run 1 only), before the CMA-ES loop, to
    // measure how thick the silhouette interior is: how many triangle
    // bboxes overlap each on-cell. This estimates the headroom for a
    // flood-fill / scanline interior fill (which would write each
    // on-cell exactly once instead of once-per-overlapping-bbox).
    //
    // The MVP used here is the initial-pose MVP: view_proj_jit times
    // the SRT matrix for an identity srt (scale=1, no rotation, no
    // translation), i.e. the pose the optimizer starts from. That is
    // a representative "typical" silhouette -- not a degenerate one --
    // so the coverage stats reflect what the hot path actually sees.
    // Measurement-only: does not touch optimizer state.
    //
    // **DISABLED by default** since 2026-05-XX. The flood-fill /
    // scanline optimization path was abandoned (3 attempts failed to
    // outperform bbox splat in practice), so this diagnostic is no
    // longer informative for routine operation. It produces ~80 lines
    // of ASCII map + histogram per Ctrl+Shift+G session, polluting the
    // log of substantive metrics (Selector, ACCEPTED, etc.). Function
    // definition is kept intact for future ad-hoc debugging; flip the
    // constexpr below to true and rebuild to re-enable.
    constexpr bool kEnableSilCoverageDiag = false;
    if (kEnableSilCoverageDiag && sil_inputs_valid && rc.run_index == 0) {
        SRTParamsV3 srt_identity;          // default-constructed = identity
        const glm::mat4 M_srt_id =
            build_srt_matrix_v3(srt_identity, rc.ctx.centroid);
        const glm::mat4 diag_mvp = view_proj_jit * M_srt_id;
        diagnoseSilCoverageOnce(
            *rc.liver_full_positions,
            params.sil_indices,
            diag_mvp,
            params.sil_img_w,
            params.sil_img_h,
            params.sil_raster_step,
            rc.run_index);
    }

    // ----- 5. CMA-ES main loop (verbatim from V3, except rebuild call)-
    const char* stop = nullptr;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    using ms_dur = std::chrono::duration<double, std::milli>;

    const auto t_loop_start = now();
    double t_eval = 0.0, t_rebuild = 0.0;

    int gen = 0;
    for (gen = 0; gen < params.maxgen && !stop; gen++) {
        auto tg0 = now();

        double**            pop = cmaes_SamplePopulation(evo);
        std::vector<double> fval(evo->lambda);

        for (int k = 0; k < evo->lambda; k++) {
            SRTParamsV3 srt = srt_from_population_v3(pop[k], params);
            int   matched = 0;
            // [V3RS-DIFF] 3-way eval dispatch:
            //   1. sil_active (Phase 2): evaluate_one_v3rs_silhouette
            //      adds lambda_sil * (1 - IoU2D) * |scale - 1| every
            //      eval (default per-eval), plus RMSE_W on every eval
            //      (via evaluate_one_v3r_weighted). Penalty is zero
            //      when scale=1 OR silhouette is perfect; otherwise
            //      scales jointly with both deviations.
            //   2. weighted_path (beta>0): V3R-W weighted RMSE only.
            //   3. default: evaluate_one_v3 (byte-identical to V3R / V3).
            // Priority is sil > weighted > plain so that lambda_sil=0
            // gives 1:1 fallback to V3R (use Ctrl+G instead if that's
            // what you want -- V3RS is intended for sil > 0).
            const bool sil_active   = sil_inputs_valid;
            const bool weighted_path =
                (params.beta_rim_weight > 0.0f) &&
                !params.is_rim_src_voxel.empty() &&
                !params.is_rim_tgt_voxel.empty();
            float rmse;
            if (sil_active) {
                float rmse_w        = 0.0f;
                float iou2d         = -1.0f;
                bool  iou_computed  = false;
                rmse = evaluate_one_v3rs_silhouette(
                    rc.ctx, scratch_pool[0], srt,
                    params.is_rim_src_voxel,
                    params.is_rim_tgt_voxel,
                    params.beta_rim_weight,
                    // ----- Phase 1 IoU2D inputs ------------------
                    sil_eval_index,
                    params.sil_eval_interval,
                    params.lambda_sil,
                    *rc.liver_full_positions,
                    params.sil_indices,
                    view_proj_jit,
                    params.sil_dist_map_2d,
                    params.sil_img_w, params.sil_img_h,
                    params.sil_raster_step,
                    // ----- Outputs --------------------------------
                    matched, rmse_w, iou2d, iou_computed,
                    // ----- Per-Run timing accumulator -------------
                    &run_timing,
                    // ----- Instrument occlusion (forwarded) -------
                    // params.sil_instrument_dist_map_2d is empty
                    // unless the wrapper opted in via the UI toggle
                    // (g_ctrlgsIgnoreInstrument). Empty -> filter
                    // OFF, no behavioural change.
                    params.sil_instrument_dist_map_2d.empty()
                        ? nullptr
                        : &params.sil_instrument_dist_map_2d,
                    params.sil_instrument_thresh_px,
                    // [NEW V3RS-5] Asymmetric / rim-sil pass-through.
                    // Both default to 0 in ParamsV3RS, so absent of the
                    // wrapper opting in, this call is byte-identical to
                    // the Phase 3 baseline.
                    params.lambda_out,
                    params.lambda_rim_sil,
                    params.rim_sil_max_px,
                    // outside_ratio_out / rim_sil_loss_out: not needed
                    // for the hot loop's accumulation (we only need the
                    // composite cost).
                    /*outside_ratio_out*/ nullptr,
                    /*rim_sil_loss_out*/  nullptr,
                    // [NEW V3RS-RIM-ANAT] Anatomic-mode RIM flag. Empty
                    // by default; non-empty only when the wrapper opts
                    // in by populating is_rim_anatomic_full. Forwarded
                    // by-pointer (.empty() check inside).
                    params.is_rim_anatomic_full.empty()
                        ? nullptr
                        : &params.is_rim_anatomic_full,
                    // [V3I] pure-IoU mode pass-through (default false).
                    params.pure_iou_mode);
                sil_eval_index++;
                if (iou_computed) {
                    // sil_sum_loss_out accumulates (1 - IoU2D) so that
                    // avg_iou = 1 - avg_sil_loss in the per-Run banner.
                    sil_sum_loss_out += (double)(1.0f - iou2d);
                    sil_eval_count_out += 1;
                }
            } else if (weighted_path) {
                rmse = CmaesRefineV3R::evaluate_one_v3r_weighted(
                    rc.ctx, scratch_pool[0], srt,
                    params.is_rim_src_voxel,
                    params.is_rim_tgt_voxel,
                    params.beta_rim_weight,
                    matched);
            } else {
                rmse = CmaesRefine::evaluate_one_v3(rc.ctx, scratch_pool[0],
                                                    srt, matched);
            }

            const bool bad = (matched < rc.ctx.matched_min_required)
                             || (rmse == 0.0f);
            fval[k] = bad ? (double)params.penalty_value : (double)rmse;

            if (fval[k] < best_rmse) {
                best_rmse = (float)fval[k];
                for (int d = 0; d < DIM; d++) best_x[d] = pop[k][d];
            }
        }

        auto tg1 = now();
        t_eval += ms_dur(tg1 - tg0).count();

        cmaes_UpdateDistribution(evo, fval.data());

        // UPDATE_INTERVAL refresh ([V3RS] subset-aware rebuild) ---------
        if (gen > 0 && gen % params.update_interval == 0) {
            auto tr0 = now();
            const SRTParamsV3 cur_best = srt_from_xvec_v3(best_x, params);
            CmaesRefineV3R::rebuild_correspondences_v3r(rc.ctx, cur_best, params,
                                                        params.subset_idx_voxel);
            auto tr1 = now();
            t_rebuild += ms_dur(tr1 - tr0).count();
        }

        if (params.verbose && (gen % params.log_every == 0)) {
            std::cout << "[V3RS] Gen " << std::setw(4) << gen
                      << "  best=" << std::fixed << std::setprecision(5)
                      << best_rmse
                      << "  sigma=" << std::setprecision(4) << evo->sigma
                      << std::endl;
        }

        stop = cmaes_TestForTermination(evo, params.maxgen,
                                        params.tolfun, params.tolx);
    }

    const double t_loop_total = ms_dur(now() - t_loop_start).count();
    const int    evo_lambda   = evo->lambda;
    rc.generations            = evo->gen;
    rc.stop_reason            = stop ? stop : "MaxGen";
    cmaes_exit(evo);

    // ----- 6. Decode best, compute screening RMSE ---------------------
    rc.best_srt        = srt_from_xvec_v3(best_x, params);
    rc.best_rmse_inner = best_rmse;

    constexpr float kRefSceneDiag_full = 7.36f;
    const float max_dist_full    = params.scene_diag
                                * (1.0f / kRefSceneDiag_full);
    const float max_dist_sq_full = max_dist_full * max_dist_full;

    const auto t_step_t2 = step_now();

    // [V3RS] Subset-RMSE flag dispatch.
    //   At S4: only full_rmse_use_subset=false is wired up (V3 path).
    //   Future: implement compute_subset_rmse_local_v3r when needed
    //   (would build KDTree on the subset of after_best, query tgt_full).
    if (params.full_rmse_use_subset &&
        params.quadrant_mask != LiverLeftRightLabel::QUAD_ALL) {
        // Future hook: subset-only screening RMSE.
        // For now, fall through to the V3 path with a one-time warning
        // so misuse during S4 is loud rather than silent.
        static bool warned_once = false;
        if (!warned_once) {
            std::cerr << "[V3RS] warning: full_rmse_use_subset=true "
                         "is reserved for future work; falling back to "
                         "full-vertex RMSE for this Run." << std::endl;
            warned_once = true;
        }
    }
    // Default path (V3 byte-identical screening RMSE):
    rc.best_rmse_full = compute_full_rmse_local(
        rc.jitter, rc.best_srt,
        *rc.liver_voxel_positions,
        *rc.liver_full_positions,
        *rc.tgt_full_points,
        max_dist_sq_full);

    const auto t_step_t3 = step_now();
    const double t_step_full_rmse =
        step_ms(t_step_t3 - t_step_t2).count();

    rc.improved = (rc.best_rmse_full < rc.rmse_before);

    // ----- 7. Time-breakdown log (verbatim from V3, V3R prefix) -------
    if (params.verbose) {
        const int    total_evals    = rc.generations * evo_lambda;
        const double t_total        = t_loop_total;
        const double t_other        = t_total - t_eval - t_rebuild;
        const int    rebuild_calls  = (rc.generations - 1)
                                  / params.update_interval;

        std::cout << std::fixed << std::setprecision(1)
                  << "[V3RS] === Run " << (rc.run_index + 1)
                  << " Time Breakdown (total " << (int)t_total
                  << " ms, " << total_evals << " evals) ===" << std::endl;
        if (t_total > 0.0) {
            std::cout << "[V3RS]   evaluate_one (sum) : " << (int)t_eval
                      << " ms (" << (int)(100*t_eval/t_total) << "%)"
                      << std::endl
                      << "[V3RS]   rebuild_corr (sum) : " << (int)t_rebuild
                      << " ms (" << (int)(100*t_rebuild/t_total) << "%)"
                      << "  [" << rebuild_calls << " calls]"
                      << std::endl
                      << "[V3RS]   cmaes/log/other    : " << (int)t_other
                      << " ms (" << (int)(100*t_other/t_total) << "%)"
                      << std::endl;
        }
        std::cout << "[V3RS]   build_eval_ctx     : "
                  << (int)t_step_build_eval << " ms"
                  << "   compute_full_rmse  : "
                  << (int)t_step_full_rmse << " ms"
                  << std::endl;

        // ----- Per-eval breakdown (sil path only) --------------------
        // Filled by evaluate_one_v3rs_silhouette through run_timing.
        // When the silhouette path was inactive this Run (lambda_sil=0
        // or sil_inputs_valid false), n_total_evals stays at 0 and we
        // skip the block entirely. Per-call cost averages are printed
        // in microseconds because the per-eval scale is sub-ms.
        if (run_timing.n_total_evals > 0) {
            const double inv_total =
                1.0 / (double)run_timing.n_total_evals;
            const double inv_iou =
                run_timing.n_iou_evals > 0
                    ? 1.0 / (double)run_timing.n_iou_evals
                    : 0.0;
            const double rmse_w_sum_ms     = run_timing.eval_rmse_w_us     * 1e-3;
            const double sil_total_sum_ms  = run_timing.eval_sil_total_us  * 1e-3;
            const double sil_proj_sum_ms   = run_timing.eval_sil_proj_us   * 1e-3;
            const double sil_splat_sum_ms  = run_timing.eval_sil_splat_us  * 1e-3;
            const double sil_iou_sum_ms    = run_timing.eval_sil_iou_us    * 1e-3;
            const double other_sum_ms      = run_timing.eval_other_us      * 1e-3;
            // Per-eval averages (microseconds).
            const double rmse_w_avg_us     = run_timing.eval_rmse_w_us     * inv_total;
            const double sil_total_avg_us  = run_timing.eval_sil_total_us  * inv_iou;
            const double sil_proj_avg_us   = run_timing.eval_sil_proj_us   * inv_iou;
            const double sil_splat_avg_us  = run_timing.eval_sil_splat_us  * inv_iou;
            const double sil_iou_avg_us    = run_timing.eval_sil_iou_us    * inv_iou;
            const double other_avg_us      = run_timing.eval_other_us      * inv_total;
            const double t_eval_safe       = (t_eval > 0.0) ? t_eval : 1.0;

            std::cout << "[V3RS]   --- per-eval breakdown ("
                      << run_timing.n_total_evals << " evals, "
                      << run_timing.n_iou_evals << " with IoU) ---"
                      << std::endl;
            std::cout << std::fixed << std::setprecision(1)
                      << "[V3RS]     rmse_w (V3R-W)    : "
                      << (int)rmse_w_sum_ms << " ms ("
                      << (int)(100.0 * rmse_w_sum_ms / t_eval_safe) << "%)"
                      << "  avg=" << rmse_w_avg_us << " us/eval"
                      << std::endl
                      << "[V3RS]     sil_total (raster): "
                      << (int)sil_total_sum_ms << " ms ("
                      << (int)(100.0 * sil_total_sum_ms / t_eval_safe) << "%)"
                      << "  avg=" << sil_total_avg_us << " us/eval"
                      << std::endl
                      << "[V3RS]       step1 proj      : "
                      << (int)sil_proj_sum_ms << " ms"
                      << "  avg=" << sil_proj_avg_us << " us/eval"
                      << std::endl
                      << "[V3RS]       step2 splat(bbox): "
                      << (int)sil_splat_sum_ms << " ms"
                      << "  avg=" << sil_splat_avg_us << " us/eval"
                      << std::endl
                      << "[V3RS]       step3 iou       : "
                      << (int)sil_iou_sum_ms << " ms"
                      << "  avg=" << sil_iou_avg_us << " us/eval"
                      << std::endl
                      << "[V3RS]     other (M_srt+pen) : "
                      << (int)other_sum_ms << " ms"
                      << "  avg=" << other_avg_us << " us/eval"
                      << std::endl;
        }

        std::cout << std::defaultfloat << std::setprecision(6)
                  << "[V3RS] Run " << (rc.run_index + 1)
                  << "  best_inner=" << rc.best_rmse_inner
                  << "  best_full="  << rc.best_rmse_full
                  << "  stop="       << rc.stop_reason
                  << (rc.improved ? "  [+]" : "  [-]")
                  << std::endl;
    }
}

// =====================================================================
// runBipopCmaesV3RS -- FORK of CmaesRefineV3R::runBipopCmaesV3R.
// ---------------------------------------------------------------------
// Differences from V3R (marked [V3RS-DIFF]):
//   - params type is ParamsV3RS
//   - Inner run loop calls run_one_bipop_v3rs (V3RS dispatch)
//   - Silhouette banner log + per-run sil aggregate report
//   - tolfun override when lambda_sil > 0 (sil makes cost surface
//     much larger in magnitude, so the default tolfun=1e-4 fires
//     way too early -- HANDOVER V6 follow-up).
// All V3R session-setup helpers are reused via the CmaesRefineV3R::
// namespace (build_voxel_to_orig, derive_arvis_voxel, etc.) so the
// V3R header stays the single source of truth for those helpers.
//
// Original V3R comment block:
// ---------------------------------------------------------------------
// Session driver. Delta from V3:
//   - Phase C (after voxel downsample): build voxel_to_orig via NN
//     reverse map (CmaesRefineV3R::build_voxel_to_orig).
//   - Phase D: derive subset_idx_voxel from voxel_to_orig + labels +
//     quadrant_mask (filter_by_quadrant). Stored on the (mutable)
//     params object so run_one_bipop_v3r can reach them without an
//     extra parameter slot.
//   - Run loop dispatches to run_one_bipop_v3r instead of run_one_bipop.
//
// Determinism: outer_seed and cma_base are passed in by the caller and
// used IDENTICALLY to V3 (mt19937, d01 sequence, jitter formulas, run==0
// gate). At QUAD_ALL the entire sequence of double draws is unchanged
// from V3.
//
// Note on the params&: we accept ParamsV3R by non-const reference so we
// can write the session-derived voxel_to_orig / subset_idx_voxel back
// onto it. The caller's instance is mutated in this single way; all
// other fields are read-only here.
// =====================================================================
inline CmaesRefine::ResultV3 runBipopCmaesV3RS(
    const std::vector<glm::vec3>& start_liver_verts,
    const std::vector<glm::vec3>& start_liver_normals,
    const std::vector<glm::vec3>& tgt_points,
    ParamsV3RS&                   params,
    float                         rmse_before,
    int                           init_matched,
    uint32_t                      outer_seed,
    uint32_t                      cma_base)
{
    using namespace CmaesRefine;

    ResultV3 r;
    r.rmse_before = rmse_before;
    r.rmse_after  = rmse_before;
    r.improved    = false;

    // ----- Empty-input early return (verbatim from V3) ----------------
    if (start_liver_verts.empty() || tgt_points.empty()) {
        std::cerr << "[V3RS] runBipopCmaesV3R: empty start_liver_verts ("
                  << start_liver_verts.size() << ") or tgt_points ("
                  << tgt_points.size() << "); aborting." << std::endl;
        return r;
    }

    // ----- Pre-compute session-wide max_dist_sq (verbatim from V3) ----
    constexpr float kRefSceneDiag_session = 7.36f;
    const float max_dist_session    = params.scene_diag
                                   * (1.0f / kRefSceneDiag_session);
    const float max_dist_sq_session = max_dist_session * max_dist_session;

    // ----- V3-2 case C: voxel-downsample (verbatim from V3) -----------
    auto sess_now    = []{ return std::chrono::high_resolution_clock::now(); };
    using sess_ms    = std::chrono::duration<double, std::milli>;
    auto t_sess_t0   = sess_now();

    std::vector<glm::vec3> session_voxel_liver;
    std::vector<glm::vec3> session_voxel_tgt;
    const float src_voxel_size = (params.src_voxel_ratio > 0.0f)
                                     ? (params.src_voxel_ratio * params.scene_diag) : 0.0f;
    const float tgt_voxel_size = (params.tgt_voxel_ratio > 0.0f)
                                     ? (params.tgt_voxel_ratio * params.scene_diag) : 0.0f;

    auto t_sess_voxel0 = sess_now();
    voxel_downsample_v3(start_liver_verts, src_voxel_size,
                        session_voxel_liver);
    auto t_sess_voxel1 = sess_now();
    voxel_downsample_v3(tgt_points,        tgt_voxel_size,
                        session_voxel_tgt);
    auto t_sess_voxel2 = sess_now();

    if (params.verbose) {
        std::cout << "[V3RS session/Time] voxel src ("
                  << start_liver_verts.size() << "->"
                  << session_voxel_liver.size() << ") : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_voxel1 - t_sess_voxel0).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3RS session/Time] voxel tgt ("
                  << tgt_points.size() << "->"
                  << session_voxel_tgt.size() << ") : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_voxel2 - t_sess_voxel1).count()
                  << " ms" << std::defaultfloat << std::endl;
    }

    // ----- Phase C [V3RS]: build voxel_to_orig via NN reverse map ------
    auto t_sess_v2o0 = sess_now();
    params.voxel_to_orig =
        CmaesRefineV3R::build_voxel_to_orig(start_liver_verts, session_voxel_liver);
    auto t_sess_v2o1 = sess_now();

    // ----- Phase C2 [V3RS-W]: derive arvis_voxel + rim arrays (opt-in) -
    // All three are session-derived from caller-provided original-space
    // arrays. Empty input -> empty output -> downstream falls back to
    // the standard V3R path (no visibility filter, no rim weighting).
    if (params.use_arvis_filter) {
        params.arvis_voxel = CmaesRefineV3R::derive_arvis_voxel(
            params.arvis_orig, params.voxel_to_orig);
        if (params.verbose) {
            int n_vis = 0;
            for (uint8_t v : params.arvis_voxel) if (v) n_vis++;
            std::cout << "[V3RS-W] arvis: visible voxels = "
                      << n_vis << "/" << params.arvis_voxel.size()
                      << "  (orig src visible="
                      << std::count(params.arvis_orig.begin(),
                                    params.arvis_orig.end(), (uint8_t)1)
                      << "/" << params.arvis_orig.size() << ")"
                      << std::endl;
        }
    } else {
        params.arvis_voxel.clear();
    }

    if (params.beta_rim_weight > 0.0f) {
        params.is_rim_src_voxel = CmaesRefineV3R::derive_is_rim_src_voxel(
            params.is_rim_orig, params.voxel_to_orig);
        params.is_rim_tgt_voxel = CmaesRefineV3R::derive_is_rim_tgt_voxel(
            tgt_points, params.tgt_boundary_dist_full,
            session_voxel_tgt, params.rim_tgt_threshold_px);
        if (params.verbose) {
            int n_rim_src = 0;
            for (uint8_t v : params.is_rim_src_voxel) if (v) n_rim_src++;
            int n_rim_tgt = 0;
            for (uint8_t v : params.is_rim_tgt_voxel) if (v) n_rim_tgt++;
            std::cout << "[V3RS-W] rim weighting: beta="
                      << params.beta_rim_weight
                      << "  src_rim=" << n_rim_src
                      << "/" << params.is_rim_src_voxel.size()
                      << "  tgt_rim=" << n_rim_tgt
                      << "/" << params.is_rim_tgt_voxel.size()
                      << "  thresh=" << params.rim_tgt_threshold_px << "px"
                      << std::endl;
        }
    } else {
        params.is_rim_src_voxel.clear();
        params.is_rim_tgt_voxel.clear();
    }

    // Caudal voxel mask (R-feat-2). Independent of arvis. Empty input
    // -> empty output -> CmaesRefineV3R::filter_by_quadrant_with_arvis_caudal treats
    // it as "not requested" (no caudal filtering).
    if (params.use_caudal_only) {
        params.is_caudal_voxel = CmaesRefineV3R::derive_is_caudal_voxel(
            params.is_caudal_orig, params.voxel_to_orig);
        if (params.verbose) {
            int n_caudal = 0;
            for (uint8_t v : params.is_caudal_voxel) if (v) n_caudal++;
            std::cout << "[V3RS-W] caudal: caudal voxels = "
                      << n_caudal << "/" << params.is_caudal_voxel.size()
                      << "  (orig src caudal="
                      << std::count(params.is_caudal_orig.begin(),
                                    params.is_caudal_orig.end(), (uint8_t)1)
                      << "/" << params.is_caudal_orig.size() << ")"
                      << "  combine="
                      << (params.arvis_caudal_combine == 0 ? "AND" : "OR")
                      << std::endl;
        }
    } else {
        params.is_caudal_voxel.clear();
    }

    // ----- Phase D [V3RS]: derive subset_idx_voxel ---------------------
    // Combined filter handles arvis + caudal in one pass with selectable
    // AND/OR mode. When both arvis_voxel and is_caudal_voxel are empty
    // (neither feature requested), it degenerates to filter_by_quadrant
    // verbatim, so QUAD_ALL byte-identical with V3 is preserved.
    params.subset_idx_voxel = CmaesRefineV3R::filter_by_quadrant_with_arvis_caudal(
        params.voxel_to_orig,
        params.region_labels,
        params.lr_labels,
        params.quadrant_mask,
        params.arvis_voxel,
        params.is_caudal_voxel,
        params.arvis_caudal_combine);
    auto t_sess_v2o2 = sess_now();

    if (params.verbose) {
        std::cout << "[V3RS session/Time] voxel_to_orig ("
                  << session_voxel_liver.size() << " NN lookups) : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_v2o1 - t_sess_v2o0).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3RS session/Time] subset_filter : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_v2o2 - t_sess_v2o1).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3RS] quadrant_mask="
                  << LiverLeftRightLabel::quadrantMaskString(params.quadrant_mask)
                  << "  arvis=" << (params.use_arvis_filter ? "ON" : "OFF")
                  << "  caudal=" << (params.use_caudal_only ? "ON" : "OFF")
                  << "  combine="
                  << (params.arvis_caudal_combine == 0 ? "AND" : "OR")
                  << "  subset_size=" << params.subset_idx_voxel.size()
                  << "/" << session_voxel_liver.size()
                  << " (voxel-space)" << std::endl;

        // [V3RS-DIFF] Silhouette session banner (Phase 1).
        // sil_ready now also requires sil_indices populated (Phase 1
        // rasterizes the full mesh, so triangle list is mandatory).
        const bool sil_ready =
            (params.lambda_sil > 0.0f) &&
            (params.sil_eval_interval > 0) &&
            !params.sil_dist_map_2d.empty() &&
            !params.sil_indices.empty() &&
            (params.sil_img_w > 0) && (params.sil_img_h > 0) &&
            (params.sil_dist_map_2d.size() ==
             (size_t)params.sil_img_w * (size_t)params.sil_img_h);
        std::cout << "[V3RS-W/sil] lambda_sil=" << params.lambda_sil
                  << "  voxel_total=" << session_voxel_liver.size()
                  << "  tris=" << (params.sil_indices.size() / 3)
                  << "  eval_interval=" << params.sil_eval_interval
                  << "  raster_step=" << params.sil_raster_step
                  << "  dist_map=" << (sil_ready ? "YES" : "NO")
                  << "  active=" << (sil_ready ? "ON" : "OFF")
                  << std::endl;
    }

    // Defensive: subset empty (e.g. QUAD_NONE) -> abort cleanly.
    if (params.subset_idx_voxel.empty()) {
        std::cerr << "[V3RS] subset_idx_voxel is empty (mask=0x"
                  << std::hex << (int)params.quadrant_mask << std::dec
                  << "); no vertices selected. Aborting session."
                  << std::endl;
        return r;
    }

    // ----- BIPOP outer rng init (verbatim from V3) --------------------
    std::mt19937 rng(outer_seed);
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    if (params.verbose) {
        std::cout << "[V3RS] === Starting BIPOP-CMA-ES V3R ===" << std::endl
                  << "[V3RS] outer_seed=" << outer_seed
                  << "  cma_base=" << cma_base
                  << "  rmse_before=" << rmse_before
                  << "  init_matched=" << init_matched
                  << "  scene_diag=" << params.scene_diag
                  << std::endl
                  << "[V3RS] src: " << start_liver_verts.size()
                  << " -> " << session_voxel_liver.size()
                  << " (voxel=" << src_voxel_size << ", ratio="
                  << params.src_voxel_ratio << ")" << std::endl
                  << "[V3RS] tgt: " << tgt_points.size()
                  << " -> " << session_voxel_tgt.size()
                  << " (voxel=" << tgt_voxel_size << ", ratio="
                  << params.tgt_voxel_ratio << ")" << std::endl;
    }

    // ----- 10 BIPOP runs (verbatim from V3, except dispatch target) ---
    const int N_STARTS = 10;

    float       best_rmse_full     = rmse_before;
    int         best_run_idx       = -1;
    SRTParamsV3 best_jitter;
    SRTParamsV3 best_srt;
    std::string best_stop_reason   = "NoImprovement";
    int         total_generations  = 0;

    auto t_sess_runs0 = sess_now();
    double t_run_outer_wall_sum = 0.0;

    // [V3RS-DIFF] Phase 0 Run-selector recording. Per-Run jitter and
    // best_srt are needed to reconstruct each Run's world matrix for
    // IoU2D evaluation after the Run loop. Allocate only when the
    // wrapper installed an IoU2D callback (no-op for legacy callers).
    const bool phase0_selector_on = static_cast<bool>(params.sil_iou2d_eval_fn);
    std::vector<SRTParamsV3> per_run_jitter;
    std::vector<SRTParamsV3> per_run_srt;
    std::vector<float>       per_run_full;
    std::vector<std::string> per_run_stop_reason;   // Phase A: combo selector
    if (phase0_selector_on) {
        per_run_jitter.resize(N_STARTS);
        per_run_srt.resize(N_STARTS);
        per_run_full.assign(N_STARTS, rmse_before);
        per_run_stop_reason.resize(N_STARTS);
    }

    // [V3RS-DIFF] Cost-surface scale changes when sil_loss is added,
    // so the default tolfun (1e-4 in V3R) fires far too early at Gen 0.
    // When sil is active, override tolfun to params.sil_tolfun_override
    // (default 1e-5). This affects only the V3RS path; the V3R caller's
    // params object is never observed here.
    const bool  sil_active_session = (params.lambda_sil > 0.0f) &&
                                    !params.sil_dist_map_2d.empty();
    const float saved_tolfun = params.tolfun;
    if (sil_active_session && params.sil_tolfun_override > 0.0f) {
        const_cast<ParamsV3RS&>(params).tolfun = params.sil_tolfun_override;
        if (params.verbose) {
            std::cout << "[V3RS-W/sil] tolfun override: "
                      << saved_tolfun << " -> " << params.tolfun
                      << "  (sil cost surface is at a different scale; "
                         "default would early-stop)" << std::endl;
        }
    }

    // ===== [V3RS-PARALLEL] Pre-generate per-run jitter / sigma0 =======
    // The outer RNG (rng, seeded by outer_seed) is consumed HERE in run
    // order -- the exact 8-draw-per-run sequence (sigma0, tx,ty,tz,
    // rx,ry,rz, scale) the legacy in-loop code used. Lifting it out of
    // the loop makes the run body RNG-free, so runs can execute in any
    // order (parallel) yet produce byte-identical jitter to the serial
    // path. cma_seed is RNG-independent (cma_base + run). This block runs
    // in BOTH modes, so legacy serial results are unchanged.
    std::vector<double>      pre_sigma0(N_STARTS, 0.0);
    std::vector<SRTParamsV3> pre_jitter(N_STARTS);
    std::vector<bool>        pre_is_local(N_STARTS, false);
    for (int run = 0; run < N_STARTS; run++) {
        const bool is_local = (run % 2 == 0);
        pre_is_local[run] = is_local;
        SRTParamsV3 jitter;
        if (is_local) {
            pre_sigma0[run] = 0.3 + d01(rng) * 0.4;
            const float lt = params.jitter_local_t;
            jitter.tx     = (d01(rng) * 2.0f - 1.0f) * lt;
            jitter.ty     = (d01(rng) * 2.0f - 1.0f) * lt;
            jitter.tz     = (d01(rng) * 2.0f - 1.0f) * lt;
            jitter.rx_deg = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            jitter.ry_deg = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            jitter.rz_deg = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            jitter.scale  = 0.95f + d01(rng) * 0.10f;
        } else {
            pre_sigma0[run] = 0.5 + d01(rng) * 0.5;
            const float gt = params.jitter_global_t;
            jitter.tx     = (d01(rng) * 2.0f - 1.0f) * gt;
            jitter.ty     = (d01(rng) * 2.0f - 1.0f) * gt;
            jitter.tz     = (d01(rng) * 2.0f - 1.0f) * gt;
            jitter.rx_deg = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            jitter.ry_deg = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            jitter.rz_deg = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            jitter.scale  = 0.90f + d01(rng) * 0.20f;
        }
        pre_jitter[run] = (run == 0) ? SRTParamsV3{} : jitter;
    }

    // Per-run result slots. Each run writes ONLY its own index, so the
    // parallel path has no shared-write races; reductions are done
    // serially after the loop in run order (identical to the legacy
    // in-loop reduction, since argmin with strict '<' over runs 0..N-1
    // keeps the first run achieving the min either way).
    std::vector<float>       pr_full(N_STARTS, rmse_before);
    std::vector<SRTParamsV3> pr_srt(N_STARTS);
    std::vector<SRTParamsV3> pr_jitter_used(N_STARTS);
    std::vector<std::string> pr_stop(N_STARTS);
    std::vector<int>         pr_gens(N_STARTS, 0);
    std::vector<double>      pr_sil_loss(N_STARTS, 0.0);
    std::vector<long long>   pr_sil_vis(N_STARTS, 0);
    std::vector<long long>   pr_sil_count(N_STARTS, 0);
    std::vector<double>      pr_wall(N_STARTS, 0.0);

    // Execute one run into slot `run`. run_params lets the parallel path
    // pass a verbose=false copy so the per-run inner logs don't interleave
    // across threads. All state touched here is either a fresh local
    // (RunContext rc + the scratch_pool/timing built inside
    // run_one_bipop_v3rs) or a disjoint pr_*[run] slot, so this is safe to
    // call concurrently for distinct `run`.
    auto exec_run = [&](int run, const ParamsV3RS& run_params) {
        const auto t0 = sess_now();
        RunContext rc;
        rc.run_index             = run;
        rc.is_local_regime       = pre_is_local[run];
        rc.cma_seed              = cma_base + (uint32_t)run;
        rc.rmse_before           = rmse_before;
        rc.init_matched          = init_matched;
        rc.max_dist_sq           = max_dist_sq_session;
        rc.liver_full_positions  = &start_liver_verts;
        rc.liver_full_normals    = &start_liver_normals;
        rc.liver_voxel_positions = &session_voxel_liver;
        rc.tgt_full_points       = &tgt_points;
        rc.tgt_voxel_points      = &session_voxel_tgt;
        rc.sigma0                = pre_sigma0[run];
        rc.jitter                = pre_jitter[run];

        double    sil_sum_loss   = 0.0;
        long long sil_sum_vis    = 0;
        long long sil_eval_count = 0;
        run_one_bipop_v3rs(rc, run_params, sil_sum_loss, sil_sum_vis, sil_eval_count);

        pr_full[run]        = rc.best_rmse_full;
        pr_srt[run]         = rc.best_srt;
        pr_jitter_used[run] = rc.jitter;
        pr_stop[run]        = rc.stop_reason;
        pr_gens[run]        = rc.generations;
        pr_sil_loss[run]    = sil_sum_loss;
        pr_sil_vis[run]     = sil_sum_vis;
        pr_sil_count[run]   = sil_eval_count;
        pr_wall[run]        = sess_ms(sess_now() - t0).count();
    };

    // Serial reduction of one run's recorded slot into the session bests +
    // selector arrays. Order-preserving: call for run=0..N-1.
    auto reduce_run = [&](int run) {
        if (pr_full[run] < best_rmse_full) {
            best_rmse_full   = pr_full[run];
            best_run_idx     = run;
            best_jitter      = pr_jitter_used[run];
            best_srt         = pr_srt[run];
            best_stop_reason = pr_stop[run];
        }
        total_generations += pr_gens[run];
        if (phase0_selector_on) {
            per_run_jitter[run]      = pr_jitter_used[run];
            per_run_srt[run]         = pr_srt[run];
            per_run_full[run]        = pr_full[run];
            per_run_stop_reason[run] = pr_stop[run];
        }
        t_run_outer_wall_sum += pr_wall[run];
    };

    auto print_run_summary = [&](int run) {
        if (!params.verbose) return;
        std::cout << std::fixed << std::setprecision(4)
                  << "[V3RS] Run " << (run+1) << "/" << N_STARTS
                  << "  " << (pre_is_local[run] ? "Local " : "Global")
                  << "  sigma0=" << pre_sigma0[run]
                  << "  cma_seed=" << (cma_base + (uint32_t)run)
                  << "  full=" << pr_full[run]
                  << "  gens=" << pr_gens[run]
                  << "  (" << std::setprecision(1) << pr_wall[run] << " ms)"
                  << std::defaultfloat << std::setprecision(6) << std::endl;
        if (pr_sil_count[run] > 0) {
            const double avg_sil = pr_sil_loss[run] / (double)pr_sil_count[run];
            std::cout << std::fixed << std::setprecision(4)
                      << "[V3RS/sil] Run " << (run+1)
                      << "  iou_evals=" << pr_sil_count[run]
                      << "  avg_iou2d=" << (1.0 - avg_sil)
                      << "  avg_sil_loss=" << avg_sil
                      << "  lambda=" << params.lambda_sil
                      << std::defaultfloat << std::setprecision(6) << std::endl;
        }
    };

#ifdef _OPENMP
    const int n_run_threads = g_v3rsParallelRuns
        ? std::max(1, std::min(N_STARTS, omp_get_max_threads()))
        : 1;
#else
    const int n_run_threads = 1;
#endif
    g_v3rsLastRunThreads = n_run_threads;

    if (g_v3rsParallelRuns && n_run_threads > 1) {
        // ---- PARALLEL run loop -------------------------------------
        if (params.verbose) {
            std::cout << "[V3RS] run loop mode: PARALLEL  ("
                      << n_run_threads << " threads x " << N_STARTS
                      << " runs; inner raster forced serial; per-run logs "
                         "deferred to after the loop)" << std::endl;
        }
        // One shared, read-only quiet copy of params: silences the
        // interleaving per-run verbose prints inside run_one_bipop_v3rs.
        // Made AFTER the tolfun override above so it inherits that value.
        ParamsV3RS params_quiet = params;
        params_quiet.verbose = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_run_threads)
#endif
        for (int run = 0; run < N_STARTS; run++) {
            exec_run(run, params_quiet);
        }
        // Serial post-pass: deferred verbose logs + order-preserving
        // reductions. (Reductions MUST be serial so best_run_idx tie-break
        // and per_run_* match the legacy path exactly.)
        for (int run = 0; run < N_STARTS; run++) {
            print_run_summary(run);
            reduce_run(run);
        }
    } else {
        // ---- SERIAL run loop (legacy; byte-identical results) ------
        if (params.verbose && g_v3rsParallelRuns) {
            std::cout << "[V3RS] run loop mode: SERIAL  (parallel requested "
                         "but <=1 thread/run available)" << std::endl;
        }
        for (int run = 0; run < N_STARTS; run++) {
            if (params.verbose) {
                std::cout << "[V3RS] Run " << (run+1) << "/" << N_STARTS
                          << "  " << (pre_is_local[run] ? "Local " : "Global")
                          << "  sigma0=" << std::fixed << std::setprecision(4)
                          << pre_sigma0[run]
                          << "  cma_seed=" << (cma_base + (uint32_t)run)
                          << std::defaultfloat << std::setprecision(6)
                          << std::endl;
            }
            exec_run(run, params);   // verbose inner logs print live here
            if (params.verbose && pr_sil_count[run] > 0) {
                const double avg_sil = pr_sil_loss[run] / (double)pr_sil_count[run];
                std::cout << std::fixed << std::setprecision(4)
                          << "[V3RS/sil] Run " << (run+1)
                          << "  iou_evals=" << pr_sil_count[run]
                          << "  avg_iou2d=" << (1.0 - avg_sil)
                          << "  avg_sil_loss=" << avg_sil
                          << "  lambda=" << params.lambda_sil
                          << std::defaultfloat << std::setprecision(6)
                          << std::endl;
            }
            reduce_run(run);
        }
    }

    auto t_sess_runs1 = sess_now();
    g_v3rsLastRunLoopMs = sess_ms(t_sess_runs1 - t_sess_runs0).count();
    if (params.verbose) {
        std::cout << "[V3RS session/Time] runs loop wall-clock : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_runs1 - t_sess_runs0).count()
                  << " ms"
                  << "  (sum of per-run outer = "
                  << t_run_outer_wall_sum << " ms)"
                  << "  [" << (g_v3rsParallelRuns ? "PARALLEL x" : "SERIAL x")
                  << g_v3rsLastRunThreads << "]"
                  << std::defaultfloat << std::endl;
    }

    // ----- [V3RS Phase A] Unified combo Run-selector ------------------
    // Replaces the previous "log-only" Phase 0 selector. The combo
    // criterion is now the authoritative decision and the same lambda_sil
    // that drives Layer 1 (per-eval cost) is used here so all three
    // decision layers (Layer 1: per-eval, Layer 2: Run selection,
    // Layer 3: session acceptance gate in the wrapper) optimise the
    // same scalar
    //
    //     c(x) = RMSE_W(x) + lambda_sil * (1 - IoU_occluded(x))
    //
    // See DESIGN_Occlusion_Aware_Silhouette_Anchor.md §3 for the
    // responsibility-split rationale.
    //
    // Pre-conditions: phase0_selector_on (wrapper installed callback)
    // AND lambda_sil > 0 AND at least one Run completed with an IoU
    // eval. Otherwise we fall back to the legacy argmin(rmse_full)
    // decision already taken inside the Run loop.
    //
    // The block has two purposes:
    //   (a) Compute per-Run IoU_occluded by replaying the callback
    //       against each Run's reconstructed world matrix.
    //   (b) Decide the final winner via argmin(combo) and override
    //       best_run_idx/best_jitter/best_srt/best_rmse_full/
    //       best_stop_reason when the combo winner differs from the
    //       rmse-only winner already picked inside the Run loop.
    //
    // Logging (per-Run lines, "best by full/IoU/combo", DECISION) is
    // emitted only when params.verbose is true; the decision itself
    // always runs when phase0_selector_on, so verbose mode does NOT
    // change which Run is applied.
    if (phase0_selector_on) {
        const auto t_sel_t0 = sess_now();

        const glm::vec3 c_pre = compute_centroid_v3(session_voxel_liver);
        std::vector<float> iou_per_run    (N_STARTS, 0.0f);
        // [NEW V3RS-SEL] Per-Run outside_ratio and rim_sil_loss. Populated
        // only when sil_metrics_eval_fn is installed; otherwise stay 0 so
        // the combo computation degrades cleanly to (RMSE + lambda_sil*
        // (1-IoU)) and is byte-identical to the pre-feature path.
        std::vector<float> outside_per_run(N_STARTS, 0.0f);
        std::vector<float> rim_sil_per_run(N_STARTS, 0.0f);
        const bool have_metrics_fn =
            static_cast<bool>(params.sil_metrics_eval_fn);

        // (a.0) Baseline metrics for the pre-Phase-D pose. Same shape as
        // Phase E "init_*" so the Selector compares Runs on the SAME
        // composite Phase E will gate on. When the extended callback is
        // not installed, we fall back to the IoU-only path and treat
        // outside / rim_sil as zero (their lambdas should also be zero
        // in that configuration, so the combo math is unchanged).
        float baseline_iou_occluded = 0.0f;
        float baseline_outside      = 0.0f;
        float baseline_rim_sil      = 0.0f;
        if (have_metrics_fn) {
            const auto bm =
                params.sil_metrics_eval_fn(glm::mat4(1.0f));
            baseline_iou_occluded = bm.iou_occluded;
            baseline_outside      = bm.outside_ratio;
            baseline_rim_sil      = bm.rim_sil_loss;
        } else {
            baseline_iou_occluded =
                params.sil_iou2d_eval_fn(glm::mat4(1.0f));
        }

        // (a.1) Per-Run metrics.
        for (int r_i = 0; r_i < N_STARTS; ++r_i) {
            const glm::mat4 M_jit_i =
                build_srt_matrix_v3(per_run_jitter[r_i], c_pre);
            std::vector<glm::vec3> voxel_after_jitter;
            apply_srt_to_points(M_jit_i, session_voxel_liver,
                                voxel_after_jitter);
            const glm::vec3 c_post = compute_centroid_v3(voxel_after_jitter);
            const glm::mat4 M_best_i =
                build_srt_matrix_v3(per_run_srt[r_i], c_post);
            const glm::mat4 M_world_i = M_best_i * M_jit_i;

            if (have_metrics_fn) {
                const auto rm =
                    params.sil_metrics_eval_fn(M_world_i);
                iou_per_run    [r_i] = rm.iou_occluded;
                outside_per_run[r_i] = rm.outside_ratio;
                rim_sil_per_run[r_i] = rm.rim_sil_loss;
            } else {
                iou_per_run[r_i] =
                    params.sil_iou2d_eval_fn(M_world_i);
            }

            // Per-Run capture callback (ImGui overlay + optional PNG).
            // The eval callback above has applied M_world_i to
            // liverMesh3D, so the capture callback can read that state
            // directly without re-applying. We pass per_run_srt[r_i]
            // .scale so the UI can display the SRT scale per slot --
            // useful for spotting scale-cheating regressions.
            if (params.sil_per_run_dump_fn) {
                params.sil_per_run_dump_fn(r_i, per_run_srt[r_i].scale);
            }
        }

        const auto t_sel_t1 = sess_now();

        // (b) Compute the three candidate criteria, using the SAME
        // lambdas Layer 1 (per-eval cost) and Layer 3 (Phase E gate)
        // use. This is the fix for the issue where Selector picked a
        // Run that won on (RMSE + lambda_sil*(1-IoU)) but lost on
        // outside/rim_sil -- Phase E then rejected because the
        // composite hadn't actually improved.
        const float L_sil = std::max(0.0f, params.lambda_sil);
        const float L_out = std::max(0.0f, params.lambda_out);
        const float L_rs  = std::max(0.0f, params.lambda_rim_sil);

        auto compute_combo = [&](int i) -> float {
            return per_run_full[i]
                 + L_sil * (1.0f - iou_per_run[i])
                 + L_out * outside_per_run[i]
                 + L_rs  * rim_sil_per_run[i];
        };

        int   idx_full   = 0;
        int   idx_iou    = 0;
        int   idx_combo  = 0;
        float min_full   = per_run_full[0];
        float max_iou    = iou_per_run[0];
        float min_combo  = compute_combo(0);
        for (int r_i = 1; r_i < N_STARTS; ++r_i) {
            if (per_run_full[r_i] < min_full) {
                min_full = per_run_full[r_i]; idx_full = r_i;
            }
            if (iou_per_run[r_i] > max_iou) {
                max_iou  = iou_per_run[r_i];  idx_iou  = r_i;
            }
            const float combo = compute_combo(r_i);
            if (combo < min_combo) {
                min_combo = combo; idx_combo = r_i;
            }
        }

        // (b.1) Baseline combo at the pre-Phase-D pose. Same shape as
        // compute_combo() / Phase E score_before so this comparison is
        // apples-to-apples. The combo winner only overrides best_run_idx
        // if its combo improves on this baseline.
        const float baseline_combo =
            rmse_before
            + L_sil * (1.0f - baseline_iou_occluded)
            + L_out * baseline_outside
            + L_rs  * baseline_rim_sil;

        // Decision: if the combo winner improves on baseline_combo,
        // override best_run_idx and friends so Phase E applies that
        // Run's world matrix. Otherwise leave best_run_idx as the
        // in-loop argmin(rmse_full) result (which may itself be -1
        // if no Run improved RMSE).
        const bool combo_improves =
            (min_combo < baseline_combo);
        const int  combo_decision_idx =
            combo_improves ? idx_combo : -1;

        // [V3I] In pure-IoU mode the winner is chosen by IoU2D (idx_iou)
        //   and accepted iff it beats the baseline IoU. RMSE is ignored
        //   here (and the wrapper skips the RMSE cap), so an IoU gain that
        //   happens to raise 3D RMSE is still applied. Default V3RS path
        //   (pure_iou_mode == false) keeps the RMSE-blended combo logic
        //   below byte-identically.
        const bool   pure_iou      = params.pure_iou_mode;
        const bool   iou_improves  = (max_iou > baseline_iou_occluded);
        (void)combo_decision_idx;

        // The previous winner (set inside the Run loop) was picked by
        // argmin(rmse_full). Remember it for the log so the diff is
        // visible to the reader.
        const int prev_best_run_idx = best_run_idx;

        if (pure_iou) {
            if (iou_improves) {
                best_run_idx     = idx_iou;
                best_jitter      = per_run_jitter[idx_iou];
                best_srt         = per_run_srt[idx_iou];
                best_rmse_full   = per_run_full[idx_iou];
                best_stop_reason = per_run_stop_reason[idx_iou];
            } else {
                best_run_idx     = -1;
            }
        } else if (combo_improves) {
            best_run_idx     = idx_combo;
            best_jitter      = per_run_jitter[idx_combo];
            best_srt         = per_run_srt[idx_combo];
            best_rmse_full   = per_run_full[idx_combo];
            best_stop_reason = per_run_stop_reason[idx_combo];
        } else {
            // No combo improvement → no winner. Phase E will treat
            // this as [NO CHANGE] and skip the apply.
            best_run_idx     = -1;
        }

        if (params.verbose) {
            // Per-Run lines.
            std::cout << std::fixed << std::setprecision(4);
            for (int r_i = 0; r_i < N_STARTS; ++r_i) {
                std::cout << "[V3RS/Selector] Run " << (r_i + 1)
                << ": full=" << per_run_full[r_i]
                << "  IoU2D=" << iou_per_run[r_i]
                << "  outside=" << outside_per_run[r_i]
                << "  rim_sil=" << rim_sil_per_run[r_i]
                << "  scale=" << per_run_srt[r_i].scale
                << "  jitter_scale=" << per_run_jitter[r_i].scale
                << std::endl;
            }

            std::cout << "[V3RS/Selector] baseline_iou_occluded="
                      << baseline_iou_occluded
                      << "  baseline_outside=" << baseline_outside
                      << "  baseline_rim_sil=" << baseline_rim_sil
                      << std::endl;
            std::cout << "[V3RS/Selector] rmse_before=" << rmse_before
                      << "  baseline_combo=" << baseline_combo
                      << "  lambdas=[sil=" << L_sil
                      << " out=" << L_out
                      << " rim_sil=" << L_rs << "]"
                      << std::endl;

            std::cout << "[V3RS/Selector] best by full  : Run "
                      << (idx_full + 1)
                      << "  (full=" << per_run_full[idx_full]
                      << "  IoU=" << iou_per_run[idx_full] << ")"
                      << std::endl;
            std::cout << "[V3RS/Selector] best by IoU   : Run "
                      << (idx_iou + 1)
                      << "  (full=" << per_run_full[idx_iou]
                      << "  IoU=" << iou_per_run[idx_iou] << ")"
                      << (idx_iou != idx_full ? "   <-- differs from full!" : "")
                      << std::endl;
            std::cout << "[V3RS/Selector] best by combo : Run "
                      << (idx_combo + 1)
                      << "  (combo=" << min_combo
                      << "  full=" << per_run_full[idx_combo]
                      << "  IoU=" << iou_per_run[idx_combo]
                      << "  outside=" << outside_per_run[idx_combo]
                      << "  rim_sil=" << rim_sil_per_run[idx_combo]
                      << ")"
                      << std::endl;

            std::cout << "[V3RS/Selector] DECISION: ";
            if (combo_improves) {
                std::cout << "applying Run " << (idx_combo + 1)
                << " (by combo, Phase A unified)";
                if (idx_combo != idx_full) {
                    std::cout << "  -- overrides argmin(full)=Run "
                              << (idx_full + 1);
                }
                std::cout << "  combo: " << baseline_combo
                          << " -> " << min_combo;
                if (prev_best_run_idx != idx_combo) {
                    std::cout << "  (rmse-loop winner was Run "
                              << (prev_best_run_idx + 1) << ")";
                }
            } else {
                std::cout << "no Run improved baseline_combo="
                          << baseline_combo
                          << " (best combo=" << min_combo
                          << " Run " << (idx_combo + 1) << ")";
            }
            std::cout << std::endl;

            std::cout << "[V3RS/Selector] selector cost : "
                      << std::fixed << std::setprecision(1)
                      << sess_ms(t_sel_t1 - t_sel_t0).count() << " ms"
                      << (have_metrics_fn
                          ? "  (10 + 1 metrics evals)"
                          : "  (10 + 1 IoU2D evals)")
                      << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }

        (void)combo_decision_idx;   // unused beyond log
    }

    // ----- Assemble best_world_matrix (verbatim from V3) --------------
    if (best_run_idx >= 0) {
        const glm::vec3 c_pre = compute_centroid_v3(session_voxel_liver);
        const glm::mat4 M_jit = build_srt_matrix_v3(best_jitter, c_pre);
        std::vector<glm::vec3> voxel_after_jitter;
        apply_srt_to_points(M_jit, session_voxel_liver, voxel_after_jitter);
        const glm::vec3 c_post = compute_centroid_v3(voxel_after_jitter);
        const glm::mat4 M_best = build_srt_matrix_v3(best_srt, c_post);

        r.best_world_matrix = M_best * M_jit;
        r.best_jitter       = best_jitter;
        r.best_srt          = best_srt;
        r.rmse_after        = best_rmse_full;
        r.improved          = (best_rmse_full < rmse_before);
        r.best_run_idx      = best_run_idx;
        r.last_stop_reason  = best_stop_reason;
        r.total_generations = total_generations;
    } else {
        r.best_world_matrix = glm::mat4(1.0f);
        r.rmse_after        = rmse_before;
        r.improved          = false;
        r.best_run_idx      = -1;
        r.last_stop_reason  = "NoImprovement";
        r.total_generations = total_generations;
    }

    if (params.verbose) {
        const double t_session_total =
            sess_ms(sess_now() - t_sess_t0).count();
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "[V3RS] === BIPOP-CMA-ES V3R DONE ==="
                  << "  best_run="
                  << (best_run_idx < 0 ? std::string("none")
                                       : std::to_string(best_run_idx + 1))
                  << "  RMSE: " << rmse_before << " -> " << r.rmse_after
                  << "  delta=" << (rmse_before - r.rmse_after)
                  << (r.improved ? "  [IMPROVED]" : "  [NO CHANGE]")
                  << "  total_gens=" << total_generations
                  << "  quadrant="
                  << LiverLeftRightLabel::quadrantMaskString(params.quadrant_mask)
                  << std::endl
                  << "[V3RS session/Time] DRIVER TOTAL : "
                  << std::fixed << std::setprecision(1)
                  << t_session_total << " ms"
                  << std::defaultfloat << std::endl;
    }

    // (g_callIdx++ is the caller's responsibility, performed in the
    //  RegistrationActions::runBipopCmaesV3R wrapper after this returns.)

    // [V3RS-DIFF] Restore tolfun (in case caller reuses params object).
    if (sil_active_session && params.sil_tolfun_override > 0.0f) {
        const_cast<ParamsV3RS&>(params).tolfun = saved_tolfun;
    }
    return r;
}


} // namespace CmaesRefineV3RS

#endif // CMAES_REFINE_V3RS_H
