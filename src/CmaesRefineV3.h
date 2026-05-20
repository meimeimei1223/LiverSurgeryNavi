#pragma once
#ifndef CMAES_REFINE_V3_H
#define CMAES_REFINE_V3_H
/*
 * CmaesRefineV3.h
 * ----------------------------------------------------------------------
 * V3 implementation of CMA-ES post-registration refinement.
 *
 * Goals (vs V2):
 *   1. Liver-only snapshot (V2 snaps all 6 organs every run = 4.7MB
 *      copy that is never touched by the inner loop).
 *   2. Matrix-based per-Run result: each Run returns SRTParams + a
 *      single mat4 instead of carrying full per-organ vertex/normal
 *      arrays around.
 *   3. Per-Run independence: no global writes inside run_one_bipop.
 *      computeUnifiedMetrics is touched only at session start/end by
 *      the caller; organs[m]->mVertices is written once at session
 *      end by the caller; g_quietMetrics is no longer toggled.
 *   4. Voxel downsample inherited from V2 phase 2.5 (HemiAuto-style,
 *      ratio * g_sceneDiag), gated to V3-2.
 *   5. Wall-clock time accounting from day 1 (V2 had a t_other double-
 *      counting bug that was patched mid-development).
 *   6. Foundation for parallelism: pure functions, per-Run RunContext,
 *      single global write at session end -- so V3-4 (population OMP)
 *      and V3-5 (run OMP) drop in without further refactoring.
 *
 * Phasing (see chat-6 plan, sections 4.x):
 *   V3-1: Skeleton + voxel = 0 (FULL_MESH compatible). Validation: V1
 *         (Shift+V) and V3 (Shift+G) produce CompRMSE matching to
 *         8 decimal places across 10 BIPOP runs from the same
 *         (g_trialSeed, g_callIdx) state.
 *   V3-2: Voxel downsample ON. Validation: RMSE within +/-5% of V1,
 *         speedup >= 1.5x on light meshes, >= 3x expected on 132k.
 *   V3-3: Heavy mesh (132k verts) confirmation.
 *   V3-4: OpenMP over the population. Bit-identical with the
 *         single-threaded V3-2 path; ~3-4x speedup.
 *   V3-5: OpenMP over runs. Bit-identical with V3-4; ~2x further.
 *   V3-6: Optional: deprecate V2 (Shift+F) and rebind Shift+G as the
 *         default BIPOP entry. PoseLibrary evalVersion column.
 *
 * Determinism contract (must hold across all phases):
 *   - outer_seed and cma_base formulas in runBipopCmaesV3() are
 *     IDENTICAL to runBipopCmaes() (V1) and runBipopCmaesV2() (V2):
 *         outer_seed = g_trialSeed + 1000u + g_callIdx * 97u
 *         cma_base   = g_trialSeed + 2000u + g_callIdx * 10u
 *   - p.rng_seed = cma_base + run; srand(rng_seed) called immediately
 *     after cmaes_init() (same timing as V1).
 *   - g_callIdx++ at the END of runBipopCmaesV3() (same as V1).
 *   - The d01(rng) consumption order in the BIPOP outer loop is
 *     IDENTICAL to V1 (sigma0, tx, ty, tz, rx, ry, rz, sc).
 *
 * Dependency policy:
 *   This header includes mCutMesh.h and NoOpen3DRegistration.h to
 *   reuse PointCloud, voxelDownSample, extractFrontFacePoints, and
 *   the nanoflann KDTree wrapper. That brings OpenGL transitively
 *   into translation units that include this header, which is fine
 *   for compilation but is NOT a license to call GL or to read/write
 *   global registration state from inside this namespace -- doing so
 *   would defeat the parallelization plan in V3-4 / V3-5.
 *
 *   API regulation (must hold):
 *     - No function in this header takes mCutMesh* as a parameter.
 *     - No function in this header reads or writes:
 *         registrationHandle, g_quietMetrics, g_targetPoints,
 *         liverMesh3D (or any other organ pointer), view, projection,
 *         gWindowWidth, gWindowHeight, computeUnifiedMetrics().
 *     - All inputs are passed by value/const-ref as std::vector<glm::vec3>
 *       (and friends). All outputs are returned in ResultV3 / RunContext.
 *
 *   The caller (RegistrationActions.h::runBipopCmaesV3()) owns:
 *     - mCutMesh* organ list assembly and vec3 conversion at entry.
 *     - extractFrontFacePoints invocation (target cloud extraction).
 *     - rmse_before / init_matched read via computeUnifiedMetrics.
 *     - Final mat4 application to all organs at session end.
 *     - g_callIdx++ at session end.
 *
 *   CmaesUtils.h (V1) is intentionally NOT included. Reusing
 *   applyIncrementalSRT (mCutMesh* signature) and Params/Result
 *   (no V3-specific fields) provides little benefit; SRT decode and
 *   matrix construction formulas are duplicated verbatim from V1
 *   below, with comments cross-referencing the source lines, to
 *   preserve byte-for-byte numerical agreement.
 * ----------------------------------------------------------------------
 */

#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <random>      // std::mt19937, std::uniform_real_distribution
#include <algorithm>   // std::sort (V3-2 voxel_downsample_v3)
// (BIPOP outer-loop rng, runBipopCmaesV3)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

extern "C" {
#include "third_party/c-cmaes/cmaes.h"
}

// PointCloud, voxelDownSample, NanoflannAdaptor, KDTree3D, searchKNN1.
// Pulled in for KDTree reuse and voxel downsample. mCutMesh.h comes
// transitively; this header still must not take mCutMesh* in any
// public signature (see "API regulation" above).
#include "mCutMesh.h"
#include "NoOpen3DRegistration.h"

// Note: this header intentionally does NOT extern-declare g_sceneDiag
// (or any other main.cpp global). Scene scale enters via
// ParamsV3::scene_diag, filled by the caller. This keeps every
// function in this namespace pure with respect to globals -- a
// requirement for V3-5 run-level parallelism.

namespace CmaesRefine {

// =====================================================================
// SRTParamsV3 -- 7-DoF parameters in physical units.
// Decoded from the [-1,1]-normalized CMA-ES population vector by
// srt_from_population_v3() (Step 2). Independent of V2's SRTParams
// to keep V3 self-contained; field-for-field identical so the
// numerical effect of any matching call site is the same.
// =====================================================================
struct SRTParamsV3 {
    float tx     = 0.0f;
    float ty     = 0.0f;
    float tz     = 0.0f;
    float rx_deg = 0.0f;
    float ry_deg = 0.0f;
    float rz_deg = 0.0f;
    float scale  = 1.0f;
};

// =====================================================================
// ParamsV3 -- run-time configuration for runBipopCmaesV3.
// Field set is the union of V1's Params and V3's voxel/parallel knobs.
// V3 explicitly does NOT inherit from V1's Params (would require
// including CmaesUtils.h, which transitively reintroduces global-state
// extern declarations we don't want propagated through this header).
// =====================================================================
struct ParamsV3 {
    // ----- 7-DoF search ranges (physical units; mirror V1 defaults) --
    // CMA-ES samples in normalized [-1, 1]; srt_from_population_v3
    // multiplies by these ranges to get physical units. Translation is
    // typically overridden by the caller per regime (RegRatios::cmaLocalT
    // for local runs, cmaGlobalT for global runs).
    float tx_range        = 0.5f;
    float ty_range        = 0.5f;
    float tz_range        = 0.5f;
    float rx_range        = 10.0f;   // degrees, applied directly
    float ry_range        = 10.0f;
    float rz_range        = 10.0f;
    float scale_lo        = 0.90f;   // scale ∈ [scale_lo, scale_hi]
    float scale_hi        = 1.10f;

    // ----- BIPOP outer-loop jitter ranges (NEW in V3) ---------------
    // Used by runBipopCmaesV3 to generate per-run initial-pose jitter.
    // Caller fills these from RegRatios::cmaLocalT() / cmaGlobalT()
    // (the same scene-scaled values V1 reads inline at runBipopCmaes
    // lines 1103, 1113). Keeping them in ParamsV3 lets V3 stay free of
    // any g_sceneDiag extern. Default 0 reproduces V1 behavior only by
    // accident (jitter values become 0 -> all runs use identity jitter,
    // CMA-ES still works but BIPOP regime variety is lost).
    float jitter_local_t  = 0.0f;
    float jitter_global_t = 0.0f;

    // ----- Penalty / matching gates (V1 plain-RMSE rules) ------------
    float min_match_ratio = 0.30f;
    float penalty_value   = 9.9f;

    // ----- CMA-ES core parameters -----------------------------------
    int      maxgen   = 150;
    int      lambda   = 0;        // 0 = c-cmaes default
    double   sigma0   = 0.3;
    double   tolfun   = 1e-6;
    double   tolx     = 1e-8;
    bool     verbose  = true;
    int      log_every    = 30;
    bool     save_debug_jpg = false;   // V3 doesn't generate per-eval debug JPGs

    // ----- Determinism control (matches V1 contract) -----------------
    // When non-zero, srand(rng_seed) is called immediately after
    // cmaes_init(). runBipopCmaesV3 fills this with cma_base + run.
    uint32_t rng_seed = 0;

    // ----- V3-specific: voxel downsample ratios ---------------------
    // Effective voxel size in meters = ratio * scene_diag. 0 disables
    // downsample entirely (V1 bit-identical path). V3-1 ships with 0
    // for both; V3-2 sets 0.015 (~3 cm at scene_diag=2.2 m).
    float src_voxel_ratio = 0.0f;
    float tgt_voxel_ratio = 0.0f;

    // ----- V3-specific: scene scale (caller fills from g_sceneDiag) -
    // Target AABB diagonal in meters. Used to:
    //   - derive effective voxel size (ratio * scene_diag),
    //   - derive max_dist_sq for the correspondence-distance gate
    //     (V1 constant: scene_diag * (1.0 / 7.36)).
    // Caller responsibility: assign p.scene_diag = g_sceneDiag before
    // calling runBipopCmaesV3. Leaving it 0 reproduces V1 behavior
    // only by accident (max_dist_sq becomes 0 -> all pairs gated out
    // -> penalty_value everywhere).
    float scene_diag = 0.0f;

    // ----- V3-specific: correspondence refresh interval -------------
    // V1 hard-codes 10. Exposed here for tuning experiments without
    // touching V1.
    int update_interval = 10;

    // ----- V3-specific: parallelism toggles (V3-4 / V3-5) -----------
    bool parallel_population = false;
    bool parallel_runs       = false;
    int  max_threads         = 0;   // 0 = omp_get_max_threads()
};

// =====================================================================
// ResultV3 -- result of a full BIPOP session (10 runs).
// Returned by runBipopCmaesV3 to the caller, which then applies
// best_world_matrix to all organs (see RegistrationActions.h).
// =====================================================================
struct ResultV3 {
    // ----- Combined transform from snapshot pose to best pose -------
    // Composition: best_world_matrix = M_best_inner * M_jitter, both
    // built around their respective centroids (see compute_full_rmse_local
    // for the ordering). Applying this matrix to every organ vertex /
    // normal at the snapshot pose reproduces the converged scene.
    glm::mat4 best_world_matrix = glm::mat4(1.0f);

    // ----- Decomposed best parameters (for logging / PoseLibrary) ---
    SRTParamsV3 best_jitter;
    SRTParamsV3 best_srt;

    // ----- Metrics --------------------------------------------------
    float rmse_before = 0.0f;
    float rmse_after  = 0.0f;
    bool  improved    = false;
    int   best_run_idx = -1;

    // ----- Aggregate diagnostics ------------------------------------
    int         total_generations = 0;
    std::string last_stop_reason;
};

// =====================================================================
// EvalContextStaticV3 -- read-only, shared across the inner loop and
// (in V3-4) across population threads. Built once per Run by
// build_eval_context_v3(); refreshed in place by
// rebuild_correspondences_v3() at UPDATE_INTERVAL boundaries.
// =====================================================================
struct EvalContextStaticV3 {
    // Base point set in pre-SRT (snapshot + jitter applied) coordinates.
    // V3-1 (voxel = 0): all liver vertices, original order.
    // V3-2 (voxel > 0): voxel-downsampled subset; order is fixed for
    //   the lifetime of one Run (plan D from chat 5: rebuild_correspondences
    //   only refreshes the KDTree, never re-voxelizes the base, so
    //   index meanings are stable across UPDATE_INTERVAL boundaries).
    std::vector<glm::vec3> base_positions;
    std::vector<glm::vec3> base_normals;

    // Around-centroid SRT origin. Computed from base_positions with the
    // same naive accumulation order used by V1's applyIncrementalSRT
    // (for v in verts: c += v; c /= cnt) so the FULL_MESH-equivalent
    // path matches V1 byte-for-byte.
    glm::vec3 centroid = glm::vec3(0.0f);

    // tgt[i] -> index into base_positions / base_normals. -1 means no
    // correspondence (this i contributes nothing to the RMSE sum).
    // Refreshed by rebuild_correspondences_v3 at UPDATE_INTERVAL
    // boundaries against the current best pose.
    std::vector<int> tgt_to_eval;

    // Target point cloud, captured once at build time and held constant
    // for the entire CMA-ES run. V3-1: full cloud. V3-2: voxel-downsampled.
    std::vector<glm::vec3> tgt_points;

    // Squared correspondence-distance gate. Pairs with squared distance
    // >= max_dist_sq are dropped from the RMSE accumulation. Set from
    // g_sceneDiag * (1.0 / kRefSceneDiag), the same scale-invariant
    // constant used by V1 / V2.
    float max_dist_sq = 0.0f;

    // Penalty gate. If matched < matched_min_required, the population
    // member is assigned params.penalty_value (V1 default 9.9).
    int matched_min_required = 10;

    // Post-jitter starting RMSE, computed at build time as the all-tgt
    // RMSE of the un-perturbed (identity SRT) base_positions against
    // tgt_points using the same max_dist_sq gate as the inner loop.
    // Used by run_one_bipop to seed CMA-ES's `best_rmse` so the first
    // population evaluation has a meaningful threshold to beat.
    //
    // V1 reference: CmaesRefine::run sets `best_rmse = result.rmse_before`,
    // and result.rmse_before is what computeUnifiedMetrics returns at
    // entry, AFTER jitter is applied. That value equals the post-jitter
    // base->tgt RMSE -- exactly what we compute here. Initialising V3's
    // best_rmse from a session-start RMSE (pre-jitter) would let Runs
    // with disruptive jitters never beat the threshold, so best_x is
    // never updated and the Run terminates with identity SRT (visible
    // in early V3 builds as best_inner == rmse_before for Global runs).
    float initial_rmse = 0.0f;
};

// =====================================================================
// EvalContextScratchV3 -- per-evaluation work memory.
// In V3-4 (parallel_population), each OMP thread owns one of these to
// keep work_positions writes thread-local. 64-byte alignment prevents
// false-sharing on the cacheline holding matched_count / last_rmse.
// =====================================================================
struct alignas(64) EvalContextScratchV3 {
    // SRT-applied base_positions, sized to base_positions.size() lazily
    // by evaluate_one_v3.
    std::vector<glm::vec3> work_positions;

    // SRT-applied base_normals. Plain RMSE (V3-1..V3-5 default
    // objective) does not read these; left for future point-to-plane
    // or normal-aware objectives.
    std::vector<glm::vec3> work_normals;

    // Diagnostics from the last evaluate_one_v3 call.
    int   matched_count = 0;
    float last_rmse     = 0.0f;
};

// =====================================================================
// RunContext -- complete state of one BIPOP run (the unit of V3-5
// run-level parallelization). Carries inputs, const refs to shared
// data, and outputs. No raw mCutMesh*, no global state.
// =====================================================================
struct RunContext {
    // ----- Inputs (filled by the caller before run_one_bipop) -------
    int      run_index     = 0;        // 0..N_STARTS-1
    bool     is_local_regime = true;   // run % 2 == 0 (matches V1)
    uint32_t cma_seed      = 0;        // cma_base + run
    double   sigma0        = 0.3;
    SRTParamsV3 jitter;                // initial-pose jitter for this run

    // ----- Session-wide values (set once by the caller; same for all
    //       runs in one Shift+G session) -----------------------------
    // Stored on the RunContext rather than passed as extra arguments
    // to run_one_bipop so the function signature stays narrow (one
    // ref + params) and so V3-5 run-level OpenMP can dispatch each
    // context independently without rebuilding extra argument lists
    // per worker.
    float rmse_before  = 0.0f;   // pre-session unified-metrics RMSE
        //   used as: (a) initial best_rmse for
        //   CMA-ES, (b) improved-gate threshold.
    int   init_matched = 0;      // pre-session compCount, feeds the
        //   matched_min_required formula in
        //   build_eval_context_v3.
    float max_dist_sq  = 0.0f;   // (scene_diag / 7.36)^2, precomputed
        // by the caller. Currently unused inside
        // run_one_bipop (which recomputes from
        // params.scene_diag for clarity), but
        // available for future call sites that
        // want a session-cached gate.

    // ----- Shared read-only data (owned by ShiftGSession-equivalent
    //       in runBipopCmaesV3, lifetime > run_one_bipop) -----------
    // Full liver vertices/normals at session start (pre-jitter,
    // NEVER voxel-downsampled). Used by compute_full_rmse_local for
    // the all-vertex screening RMSE, mirroring V1's per-Run
    // computeUnifiedMetrics evaluation against the full mesh.
    const std::vector<glm::vec3>* liver_full_positions = nullptr;
    const std::vector<glm::vec3>* liver_full_normals   = nullptr;
    // Voxel-downsampled liver positions (V3-2+; equals liver_full
    // when src_voxel_ratio == 0). Used inside the CMA-ES inner loop
    // for jitter application, build_eval_context_v3, and centroid
    // computation in matrix construction. Voxel-then-jitter ordering
    // (case C in chat 6 plan) keeps the downsampled cloud independent
    // of jitter and avoids re-voxelizing every Run.
    const std::vector<glm::vec3>* liver_voxel_positions = nullptr;
    // Full target cloud (NEVER voxel-downsampled; used by
    // compute_full_rmse_local for the all-vertex screening RMSE).
    const std::vector<glm::vec3>* tgt_full_points      = nullptr;
    // Voxel-downsampled target cloud (V3-2+; equals tgt_full when
    // tgt_voxel_ratio == 0). Used inside the CMA-ES inner loop only.
    const std::vector<glm::vec3>* tgt_voxel_points     = nullptr;

    // ----- Per-run mutable scratch (built inside run_one_bipop) -----
    EvalContextStaticV3 ctx;

    // ----- Outputs --------------------------------------------------
    SRTParamsV3 best_srt;            // CMA-ES converged x
    float       best_rmse_inner = 0.0f;   // voxel-scale RMSE (CMA-ES gauge)
    float       best_rmse_full  = 0.0f;   // all-vertex RMSE (run comparison)
    bool        improved        = false;
    int         generations     = 0;
    std::string stop_reason;
};

// =====================================================================
// Forward declarations for Steps 4-9 (implementations land in later
// commits). Stubs below let the header parse and link standalone.
// =====================================================================
inline EvalContextStaticV3 build_eval_context_v3(
    const std::vector<glm::vec3>& src_positions,
    const std::vector<glm::vec3>& src_normals,
    const std::vector<glm::vec3>& tgt_points,
    const ParamsV3& params,
    int             init_matched);

inline float evaluate_one_v3(
    const EvalContextStaticV3& S,
    EvalContextScratchV3&      W,
    const SRTParamsV3&         srt,
    int&                       matched_out);

inline void rebuild_correspondences_v3(
    EvalContextStaticV3& S,
    const SRTParamsV3&   cur_best,
    const ParamsV3&      params);

inline float compute_full_rmse_local(
    const SRTParamsV3&            jitter,
    const SRTParamsV3&            cma_best,
    const std::vector<glm::vec3>& liver_voxel,    // for centroid extraction
    const std::vector<glm::vec3>& liver_full,     // for RMSE evaluation
    const std::vector<glm::vec3>& tgt_full,
    float                         max_dist_sq);

// V3-2: Voxel-grid downsample with deterministic post-sort.
// When voxel_size <= 0 (or input empty) returns the input unchanged
// in V3-1 byte-identical compatible order.
inline void voxel_downsample_v3(
    const std::vector<glm::vec3>& in_points,
    float                         voxel_size,
    std::vector<glm::vec3>&       out_points);

inline void run_one_bipop(
    RunContext&     rc,
    const ParamsV3& params);

inline ResultV3 runBipopCmaesV3(
    const std::vector<glm::vec3>& start_liver_verts,
    const std::vector<glm::vec3>& start_liver_normals,
    const std::vector<glm::vec3>& tgt_points,
    const ParamsV3&               params,
    float                         rmse_before,
    int                           init_matched,
    uint32_t                      outer_seed,
    uint32_t                      cma_base);

// =====================================================================
// Step 2: SRT helpers.
// ---------------------------------------------------------------------
// Decode the CMA-ES normalized [-1, 1] vector into physical-unit
// SRTParamsV3, and build the around-centroid 4x4 SRT matrix.
//
// Expression evaluation order, cast timing, and matrix composition
// order are kept BYTE-FOR-BYTE identical to V1 (CmaesUtils.h::run()
// inner loop and applyIncrementalSRT(), lines 1099-1109). This is
// required for the V3-1 voxel-ratio=0 path to reproduce V1 results
// to the last bit.
// =====================================================================

// V1 reference (CmaesUtils.h::run() inner loop, summarized):
//     float tx = (float)(pop[k][0] * params.tx_range);
//     float ty = (float)(pop[k][1] * params.ty_range);
//     float tz = (float)(pop[k][2] * params.tz_range);
//     float rx = (float)(pop[k][3] * params.rx_range);
//     float ry = (float)(pop[k][4] * params.ry_range);
//     float rz = (float)(pop[k][5] * params.rz_range);
//     float sc = params.scale_lo
//              + (float)((pop[k][6]+1.0)*0.5)
//                    * (params.scale_hi - params.scale_lo);
inline SRTParamsV3 srt_from_population_v3(const double* pop_k,
                                          const ParamsV3& p)
{
    SRTParamsV3 s;
    s.tx     = (float)(pop_k[0] * p.tx_range);
    s.ty     = (float)(pop_k[1] * p.ty_range);
    s.tz     = (float)(pop_k[2] * p.tz_range);
    s.rx_deg = (float)(pop_k[3] * p.rx_range);
    s.ry_deg = (float)(pop_k[4] * p.ry_range);
    s.rz_deg = (float)(pop_k[5] * p.rz_range);
    s.scale  = p.scale_lo
              + (float)((pop_k[6] + 1.0) * 0.5)
                    * (p.scale_hi - p.scale_lo);
    return s;
}

// Body intentionally identical to srt_from_population_v3; the separate
// name documents the call site (best-so-far parameters, used at
// UPDATE_INTERVAL rebuild and final writeback). Matches V1's
// best_x decode at run() lines 1432, 1493, 1530.
inline SRTParamsV3 srt_from_xvec_v3(const double* x, const ParamsV3& p)
{
    SRTParamsV3 s;
    s.tx     = (float)(x[0] * p.tx_range);
    s.ty     = (float)(x[1] * p.ty_range);
    s.tz     = (float)(x[2] * p.tz_range);
    s.rx_deg = (float)(x[3] * p.rx_range);
    s.ry_deg = (float)(x[4] * p.ry_range);
    s.rz_deg = (float)(x[5] * p.rz_range);
    s.scale  = p.scale_lo
              + (float)((x[6] + 1.0) * 0.5)
                    * (p.scale_hi - p.scale_lo);
    return s;
}

// Build the around-centroid 4x4 similarity transform.
// V1 reference (CmaesUtils.h::applyIncrementalSRT, lines 1099-1109):
//     const float deg2rad = (float)(M_PI / 180.0);
//     glm::mat4 T  = glm::translate(glm::mat4(1.0f), glm::vec3(tx,ty,tz));
//     glm::mat4 Rx = glm::rotate(glm::mat4(1.0f), rx_deg * deg2rad,
//                                glm::vec3(1,0,0));
//     glm::mat4 Ry = glm::rotate(glm::mat4(1.0f), ry_deg * deg2rad,
//                                glm::vec3(0,1,0));
//     glm::mat4 Rz = glm::rotate(glm::mat4(1.0f), rz_deg * deg2rad,
//                                glm::vec3(0,0,1));
//     glm::mat4 S  = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
//     glm::mat4 toCentroid    = glm::translate(glm::mat4(1.0f), -centroid);
//     glm::mat4 fromCentroid  = glm::translate(glm::mat4(1.0f),  centroid);
//     glm::mat4 M = T * fromCentroid * Rz * Ry * Rx * S * toCentroid;
inline glm::mat4 build_srt_matrix_v3(const SRTParamsV3& srt,
                                     const glm::vec3&   centroid)
{
    const float deg2rad = (float)(M_PI / 180.0);
    glm::mat4 T  = glm::translate(glm::mat4(1.0f),
                                 glm::vec3(srt.tx, srt.ty, srt.tz));
    glm::mat4 Rx = glm::rotate(glm::mat4(1.0f),
                               srt.rx_deg * deg2rad,
                               glm::vec3(1, 0, 0));
    glm::mat4 Ry = glm::rotate(glm::mat4(1.0f),
                               srt.ry_deg * deg2rad,
                               glm::vec3(0, 1, 0));
    glm::mat4 Rz = glm::rotate(glm::mat4(1.0f),
                               srt.rz_deg * deg2rad,
                               glm::vec3(0, 0, 1));
    glm::mat4 S  = glm::scale(glm::mat4(1.0f),
                             glm::vec3(srt.scale));
    glm::mat4 toCentroid   = glm::translate(glm::mat4(1.0f), -centroid);
    glm::mat4 fromCentroid = glm::translate(glm::mat4(1.0f),  centroid);
    return T * fromCentroid * Rz * Ry * Rx * S * toCentroid;
}

// =====================================================================
// Step 3: Geometry utilities.
// ---------------------------------------------------------------------
// Reproduce V1 applyIncrementalSRT's centroid-and-apply logic on
// std::vector<glm::vec3> input. The numerical effect is bit-identical
// to V1 because:
//   - V1's flat float array v[i], v[i+1], v[i+2] is repacked vertex-
//     by-vertex into glm::vec3 by the caller; the per-vertex bytes are
//     unchanged, so reading them back through .x/.y/.z gives the same
//     float value as v[i+0/1/2].
//   - The accumulation order (sequential, naive +=) is preserved.
//   - The final divide is by (float)cnt with cnt as int -- same usual-
//     arithmetic-conversion rule as V1.
// =====================================================================

// V1 reference (applyIncrementalSRT lines 1088-1097):
//     glm::vec3 centroid(0.0f);
//     int cnt = 0;
//     for (size_t i = 0; i + 2 < verts.size(); i += 3) {
//         centroid += glm::vec3(verts[i], verts[i+1], verts[i+2]);
//         cnt++;
//     }
//     if (cnt > 0) centroid /= (float)cnt;
inline glm::vec3 compute_centroid_v3(const std::vector<glm::vec3>& verts)
{
    glm::vec3 centroid(0.0f);
    int cnt = 0;
    for (const glm::vec3& v : verts) {
        centroid += v;
        cnt++;
    }
    if (cnt > 0) centroid /= (float)cnt;
    return centroid;
}

// V1 reference (applyIncrementalSRT lines 1116-1120, vertices only):
//     for (size_t i = 0; i + 2 < v.size(); i += 3) {
//         glm::vec4 p(v[i], v[i+1], v[i+2], 1.0f);
//         glm::vec4 tp = M * p;
//         v[i] = tp.x; v[i+1] = tp.y; v[i+2] = tp.z;
//     }
// V3 writes into a separate output buffer (no in-place mutation of
// the input) so callers can reuse the snapshot positions in the next
// evaluation. The per-vertex .x/.y/.z assignment mirrors V1 verbatim.
inline void apply_srt_to_points(const glm::mat4&              M,
                                const std::vector<glm::vec3>& src,
                                std::vector<glm::vec3>&       dst)
{
    if (dst.size() != src.size()) dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        glm::vec4 p(src[i].x, src[i].y, src[i].z, 1.0f);
        glm::vec4 tp = M * p;
        dst[i] = glm::vec3(tp.x, tp.y, tp.z);
    }
}

// =====================================================================
// V3-2 helper: Voxel-grid downsample with deterministic post-sort.
// ---------------------------------------------------------------------
// Wraps Reg3DCustom::NoOpen3DRegistration::voxelDownSample (which uses
// std::unordered_map and therefore returns points in implementation-
// defined iteration order) and applies a lexicographic sort on the
// output coordinates so that the result is deterministic across STL
// implementations / build flavors.
//
// Why determinism matters:
//   - The KDTree built from this output is sensitive to input order
//     in tie / near-tie cases. Differences in input order produce
//     different tree structures, which in turn yield slightly
//     different best_rmse_inner values for the CMA-ES loop.
//   - For run-to-run reproducibility on a single machine and for
//     cross-machine reproducibility (different STL hash impls), a
//     canonical order is required.
//
// Approach:
//   1. Wrap input vec3 array into a Reg3DCustom::PointCloud (positions
//      only -- V3 plain RMSE does not consume normals).
//   2. Call existing voxelDownSample.
//   3. Sort the output by lexicographic (x, y, z) comparison. Because
//      voxel cell centroids are distinct (each represents a unique
//      cell), the sort is well-defined; ties between cells are
//      vanishingly rare and harmless when they occur (the points are
//      effectively duplicates).
//
// Edge cases:
//   - voxel_size <= 0 OR in_points empty: copy input verbatim. This
//     reproduces V3-1 (FULL_MESH) ordering bit-for-bit when ratios=0,
//     which is required for the V3-1 / V3-2 (ratio=0) compatibility
//     check.
// =====================================================================
inline void voxel_downsample_v3(
    const std::vector<glm::vec3>& in_points,
    float                         voxel_size,
    std::vector<glm::vec3>&       out_points)
{
    if (in_points.empty() || voxel_size <= 0.0f) {
        // Verbatim copy preserves V1 byte-identical input ordering
        // (matters because the inner-loop KDTree iterates in input
        // order and ties resolve by index).
        out_points = in_points;
        return;
    }

    // ----- 1. Wrap into PointCloud (positions only) ------------------
    auto in_pc = std::make_shared<Reg3DCustom::PointCloud>();
    in_pc->points = in_points;   // copy

    // ----- 2. Voxel-grid downsample (existing impl) ------------------
    // NoOpen3DRegistration::voxelDownSample is effectively stateless
    // (no this-> reads), so a local instance is safe. V2 keeps a
    // cached instance per EvalContextStaticV2, but here we voxelize
    // exactly twice per Shift+G session so the per-call construction
    // cost is negligible.
    Reg3DCustom::NoOpen3DRegistration reg;
    auto out_pc = reg.voxelDownSample(in_pc, voxel_size);

    // ----- 3. Lexicographic sort (cross-platform determinism) --------
    out_points = std::move(out_pc->points);
    if (out_points.size() > 1) {
        std::sort(out_points.begin(), out_points.end(),
                  [](const glm::vec3& a, const glm::vec3& b) {
                      if (a.x != b.x) return a.x < b.x;
                      if (a.y != b.y) return a.y < b.y;
                      return a.z < b.z;
                  });
    }
}

// =====================================================================
// STUBS for Steps 4-9 (filled in subsequent commits).
// Each returns a default-constructed value of the right type so the
// header parses and the linker is satisfied if main.cpp is wired up
// before the inner functions are written. Once Step 9 lands these
// stubs are replaced wholesale.
// =====================================================================

// =====================================================================
// Step 4: build_eval_context_v3.
// ---------------------------------------------------------------------
// Construct the read-only EvalContextStaticV3 used by every
// evaluate_one_v3 call inside one CMA-ES run. V3-1 covers the
// voxel-ratio = 0 path (V1 bit-identical). V3-2 will add the voxel
// downsample branch in the marked TODO blocks.
//
// V1 reference (CmaesUtils.h::run() lines 1335-1376, the cache build
// + initial updateCorrespondences):
//
//   constexpr float kRefSceneDiag_cache = 7.36f;
//   const float max_dist_for_cache = g_sceneDiag * (1.0f / kRefSceneDiag_cache);
//   const float max_dist_sq_cache  = max_dist_for_cache * max_dist_for_cache;
//
//   std::vector<int> corr_idx(tgt_size, -1);
//   auto updateCorrespondences = [&]() {
//       const auto& verts = liverMesh3D->mVertices;
//       std::vector<glm::vec3> srcPts;
//       srcPts.reserve(verts.size() / 3);
//       for (size_t i = 0; i + 2 < verts.size(); i += 3)
//           srcPts.emplace_back(verts[i], verts[i+1], verts[i+2]);
//       Reg3DCustom::NanoflannAdaptor adaptor(srcPts);
//       auto tree = Reg3DCustom::buildKDTree(adaptor);
//       for (size_t i = 0; i < tgt_size; i++) {
//           size_t nnIdx; float dist_sq;
//           if (Reg3DCustom::searchKNN1(*tree, tgt_points_cache[i], nnIdx, dist_sq)
//               && dist_sq < max_dist_sq_cache)
//               corr_idx[i] = (int)nnIdx;
//           else
//               corr_idx[i] = -1;
//       }
//   };
//   updateCorrespondences();
//
// V3 contract for bit-identical reproduction:
//   - src_positions arrives in the same vertex order as
//     liverMesh3D->mVertices (caller does the flat-float -> vec3
//     repacking with the same i+=3 stride V1 uses).
//   - tgt_points arrives as the .points field of the same
//     extractFrontFacePoints PointCloud V1 caches.
//   - max_dist_sq formula matches V1 verbatim (kRefSceneDiag = 7.36).
//   - The dist_sq < max_dist_sq comparison is STRICT less-than.
//   - On the no-correspondence branch tgt_to_eval[i] is left as -1.
// =====================================================================
inline EvalContextStaticV3 build_eval_context_v3(
    const std::vector<glm::vec3>& src_positions,
    const std::vector<glm::vec3>& src_normals,
    const std::vector<glm::vec3>& tgt_points,
    const ParamsV3&               params,
    int                           init_matched)
{
    EvalContextStaticV3 S;

    // ----- 0. Empty-input guard --------------------------------------
    // Caller may legitimately end up here with empty arrays (e.g.
    // first frame, before any geometry loads). Return a zero-sized
    // context; runV3 detects this via base_positions.empty() and
    // aborts the CMA-ES run cleanly with rmse_after = rmse_before.
    if (src_positions.empty() || tgt_points.empty()) {
        if (params.verbose) {
            std::cerr << "[V3] build_eval_context: empty src ("
                      << src_positions.size() << ") or tgt ("
                      << tgt_points.size() << "); aborting build."
                      << std::endl;
        }
        return S;
    }

    // ----- A. tgt_points (V3-2 case C: pre-voxelized by caller) -----
    // In V3-2 case C the caller (runBipopCmaesV3) voxel-downsamples
    // tgt ONCE per Shift+G session before entering the Run loop and
    // passes the small cloud here. This function therefore performs
    // no voxel work itself; it just copies the reference into the
    // context for KDTree iteration in step E below.
    //
    // V3-1 (ratio=0) path: caller passes the full tgt cloud, this
    // copy preserves the original order (required for the KDTree
    // input to match V1's srcPts iteration order bit-for-bit).
    S.tgt_points = tgt_points;

    // ----- B. base_positions / base_normals (V3-2 case C) -----------
    // Same case-C contract as A: caller pre-voxelizes liver ONCE per
    // session and applies jitter to the SMALL cloud before invoking
    // build_eval_context_v3. src_positions here is therefore
    // liver_voxel_after_jitter (small in V3-2, full in V3-1 ratio=0).
    // No voxel branch lives in this function.
    S.base_positions = src_positions;
    S.base_normals   = src_normals;

    // ----- C. Centroid (V1 byte-identical accumulation order) -------
    // compute_centroid_v3 uses the same naive +=/=cnt sequence as V1
    // (CmaesUtils.h::applyIncrementalSRT lines 1088-1097), so this
    // value is bit-identical to what V1 computes inside
    // applyIncrementalSRT on the same vertex array.
    S.centroid = compute_centroid_v3(S.base_positions);

    // ----- D. max_dist_sq (V1 constant; CmaesUtils.h lines 1340-1342)
    constexpr float kRefSceneDiag = 7.36f;
    const float max_dist = params.scene_diag * (1.0f / kRefSceneDiag);
    S.max_dist_sq = max_dist * max_dist;

    // ----- E. Initial tgt_to_eval (KDTree on base_positions) --------
    // Mirror V1's updateCorrespondences lambda. Same adaptor type,
    // same tree builder, same search call, same gate. Empty-input
    // already filtered out at step 0, so adaptor / tree construction
    // here cannot operate on empty input.
    //
    // V1 ordering note: this block runs FIRST in V3, before
    // matched_min_required is computed below in step F. V1 instead
    // computes min_ok upfront from a pre-stored init_matched (which
    // is registrationHandle.compCount measured AFTER jitter is
    // applied by runBipopCmaes). To reproduce that bit-for-bit, V3
    // ignores the init_matched argument's value when computing
    // min_ok and uses the matched count freshly measured against the
    // jitter-applied base_positions instead. See step F for details.
    Reg3DCustom::NanoflannAdaptor adaptor(S.base_positions);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    S.tgt_to_eval.assign(S.tgt_points.size(), -1);
    int   post_jitter_matched = 0;
    float post_jitter_sumSq   = 0.0f;
    for (size_t i = 0; i < S.tgt_points.size(); i++) {
        size_t nnIdx;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, S.tgt_points[i], nnIdx, dist_sq)
            && dist_sq < S.max_dist_sq) {
            S.tgt_to_eval[i] = (int)nnIdx;
            post_jitter_matched++;
            // V1 byte-identical accumulation: V1's computeUnifiedMetrics
            // (RegistrationActions.h lines 305-318) does
            //   float d = std::sqrt(d2);
            //   sumSq += d * d;
            // The d->d*d round-trip introduces a few-ulp difference vs
            // accumulating dist_sq directly, which propagates into
            // compRmse and hence into best_rmse / Gen 0 fval. Mirror
            // V1's exact form to keep initial_rmse byte-identical to
            // what V1's computeUnifiedMetrics writes into compRmse.
            const float d = std::sqrt(dist_sq);
            post_jitter_sumSq += d * d;
        }
        // else: leave as -1 (assigned by .assign above).
    }

    // ----- F-2. Post-jitter starting RMSE ---------------------------
    // V1 reference: in CmaesRefine::run, result.rmse_before is set
    // from registrationHandle.compRmse, which RegistrationActions.h
    // computeUnifiedMetrics computes as
    //   float n = count ? count : 1.0f;
    //   compRmse = std::sqrt(sumSq / n);
    // The (count ? count : 1.0f) ternary returns 0 when count==0 (i.e.
    // sqrt(sumSq/1) == sqrt(0) == 0); we mirror that exactly so the
    // edge case doesn't produce an inconsistent "9.9 starting RMSE"
    // initialisation in V3.
    if (post_jitter_matched == 0) {
        S.initial_rmse = 0.0f;   // V1 unified-metric semantics
    } else {
        S.initial_rmse = std::sqrt(post_jitter_sumSq
                                   / (float)post_jitter_matched);
    }

    // ----- F. matched_min_required (V1 byte-identical) --------------
    // CRITICAL: this MUST be computed from the JITTER-APPLIED matched
    // count, not the session-start init_matched value.
    //
    // V1 reference (CmaesUtils.h::run() lines 1216-1219, 1438-1441):
    //   computeUnifiedMetrics();
    //   result.rmse_before = registrationHandle.compRmse;
    //   int init_matched   = registrationHandle.compCount;
    //   ...
    //   int min_ok = (int)(init_matched * params.min_match_ratio);
    //   if (min_ok < 10) min_ok = 10;
    //
    // The crucial subtlety: runBipopCmaes() applies jitter to organs
    // BEFORE entering CmaesRefine::run(). So when run() reads
    // registrationHandle.compCount via computeUnifiedMetrics, it gets
    // the post-jitter matched count. V1 thus has a per-Run min_ok
    // that varies with how disruptive the Run's jitter was.
    //
    // V3 must mirror this. The post_jitter_matched we just computed
    // against S.base_positions (which is liver_after_jitter, supplied
    // by run_one_bipop) IS the post-jitter compCount-equivalent --
    // it uses the same KDTree, same max_dist_sq gate, same iteration
    // order as V1's computeUnifiedMetrics body in
    // RegistrationActions.h lines 305-316. The only structural
    // difference is V1 builds the KDTree on liverMesh3D->mVertices
    // post-jitter (which equals our S.base_positions vertex-for-
    // vertex when the caller does the conversion correctly), so the
    // count is bit-identical.
    //
    // The function's `init_matched` parameter is now unused for the
    // min_ok computation but kept in the signature for API stability
    // and log readability. (Earlier V3 builds used it to compute
    // min_ok directly, which broke V1 bit-identical reproduction
    // for Runs whose jitter significantly changed the matched count.)
    (void)init_matched;  // explicitly mark as unused for min_ok
    int min_ok = (int)(post_jitter_matched * params.min_match_ratio);
    if (min_ok < 10) min_ok = 10;
    S.matched_min_required = min_ok;

    // ----- G. Verbose log -------------------------------------------
    if (params.verbose) {
        std::cout << "[V3] build_eval_context"
                  << "  src=" << S.base_positions.size()
                  << "  tgt=" << S.tgt_points.size()
                  << "  session_init_matched=" << init_matched
                  << "  post_jitter_matched=" << post_jitter_matched
                  << "  matched_min=" << S.matched_min_required
                  << "  post_jitter_rmse=" << S.initial_rmse
                  << "  max_dist=" << std::sqrt(S.max_dist_sq) << "m"
                  << "  scene_diag=" << params.scene_diag
                  << std::endl;
    }

    return S;
}

// =====================================================================
// Step 5: evaluate_one_v3.
// ---------------------------------------------------------------------
// PURE FUNCTION. Transforms S.base_positions by `srt` into
// W.work_positions, then computes RMSE against S.tgt_points using the
// cached correspondence map S.tgt_to_eval. Reads no globals; writes
// only into the per-thread scratch W. This purity is what unlocks
// V3-4 (population OMP) and V3-5 (run OMP).
//
// Returns:
//   - 9.9f when count == 0 (no matched pairs), matching V1
//     fastComputeRMSE's early-return value;
//   - sqrt(sumSq / count) otherwise.
//
// matched_out: receives the number of matched pairs. The caller
// (run_one_bipop) uses this for V1's penalty rule
// (matched < S.matched_min_required -> fval = penalty_value).
//
// V1 reference (CmaesUtils.h::run() inner k-loop body, lines 1430-1432
// + fastComputeRMSE lambda lines 1378-1396):
//
//   organs[0]->mVertices = snap_v[0];                     // restore
//   organs[0]->mNormals  = snap_n[0];
//   applyIncrementalSRT(liver_only, tx, ty, tz,
//                       rx, ry, rz, sc);                  // mutate
//   float rmse    = fastComputeRMSE();                    // read mVertices
//   int   matched = registrationHandle.compCount;
//
//   auto fastComputeRMSE = [&]() -> float {
//       const auto& verts = liverMesh3D->mVertices;
//       float sumSq = 0.0f;  int count = 0;
//       for (size_t i = 0; i < tgt_size; i++) {
//           int j = corr_idx[i];
//           if (j < 0) continue;
//           size_t vi = (size_t)j * 3;
//           if (vi + 2 >= verts.size()) continue;
//           glm::vec3 srcPt(verts[vi], verts[vi+1], verts[vi+2]);
//           glm::vec3 d  = srcPt - tgt_points_cache[i];
//           float     sq = glm::dot(d, d);
//           if (sq < max_dist_sq_cache) { sumSq += sq; count++; }
//       }
//       if (count == 0) return 9.9f;
//       registrationHandle.compCount = count;             // global write
//       return std::sqrt(sumSq / count);
//   };
//
// V3 deviations from V1 (verified bit-identical-safe):
//   1. The (vi + 2 >= verts.size()) range check is dropped. KDTree
//      returns indices in [0, base_positions.size()), and base_positions
//      is sized once in build_eval_context_v3 and never resized after,
//      so the check would never fire on a valid context. (Even if it
//      did fire, both branches "skip this i" so dropping the check
//      can't change the V1 numerical result on valid inputs.)
//   2. The `registrationHandle.compCount = count` write is REMOVED.
//      It violates the V3 API regulation (no global writes inside
//      this namespace) and is unused at the V3 call site (the caller
//      reads matched_out by reference instead). V1's CompCount is
//      restored at session end by the caller's computeUnifiedMetrics.
//   3. Per-vertex .x/.y/.z assignment is replaced by a vec3 subtract
//      (W.work_positions[j] - S.tgt_points[i]). Both forms reduce to
//      the same three float subtractions in the same order; result is
//      bit-identical.
//
// Phase 1 note (intentional V1 deviation, no impact on RMSE):
// W.work_normals is not populated. V1's applyIncrementalSRT
// transforms normals in place, but fastComputeRMSE never reads them,
// so the resulting RMSE is identical. If a future objective
// (point-to-plane, normal-weighted) needs them, populate via
// glm::mat3(glm::transpose(glm::inverse(M))) -- equal to V1's
// normalMat construction at applyIncrementalSRT line 1115.
// =====================================================================
inline float evaluate_one_v3(const EvalContextStaticV3& S,
                             EvalContextScratchV3&      W,
                             const SRTParamsV3&         srt,
                             int&                       matched_out)
{
    // ----- 1. SRT matrix (around-centroid; identical to V1) ----------
    const glm::mat4 M = build_srt_matrix_v3(srt, S.centroid);

    // ----- 2. Apply M to base_positions -> W.work_positions ----------
    // apply_srt_to_points uses the same per-vertex
    // glm::vec4(p, 1.0f); tp = M*p; out = vec3(tp.x, tp.y, tp.z)
    // pattern as V1's applyIncrementalSRT loop, so corresponding
    // entries in V1's mutated mVertices and V3's work_positions are
    // float-equal.
    apply_srt_to_points(M, S.base_positions, W.work_positions);

    // ----- 3. RMSE over matched correspondences ---------------------
    // Mirror of V1 fastComputeRMSE, sans global write. Iteration
    // order, accumulation order, and the strict (sq < max_dist_sq)
    // gate match V1 verbatim, so float accumulation produces the
    // same sumSq sequence and the same count.
    const size_t T = S.tgt_points.size();
    float sumSq = 0.0f;
    int   count = 0;
    for (size_t i = 0; i < T; i++) {
        int j = S.tgt_to_eval[i];
        if (j < 0) continue;
        glm::vec3 d  = W.work_positions[(size_t)j] - S.tgt_points[i];
        float     sq = glm::dot(d, d);
        if (sq < S.max_dist_sq) {
            sumSq += sq;
            count++;
        }
    }

    matched_out     = count;
    W.matched_count = count;

    if (count == 0) {
        W.last_rmse = 9.9f;
        return 9.9f;
    }

    // sumSq / count: count is int, promotes to float per the C++
    // usual-arithmetic-conversion rule -- same expression effective
    // as V1's `std::sqrt(sumSq / count)`. The (float) cast is
    // explicit here for auditability; the result is bit-identical.
    W.last_rmse = std::sqrt(sumSq / (float)count);
    return W.last_rmse;
}

// =====================================================================
// Step 6: rebuild_correspondences_v3.
// ---------------------------------------------------------------------
// Refresh S.tgt_to_eval at UPDATE_INTERVAL boundaries. As CMA-ES
// converges the best_x drifts, so the snapshot-time KDTree no longer
// pairs each tgt with its true nearest src; periodically rebuilding
// the correspondence map under the current best pose pulls a stuck
// optimiser out of local minima.
//
// V1 reference (CmaesUtils.h::run() lines 1481-1497; the
// gen % UPDATE_INTERVAL == 0 branch):
//
//   if (gen > 0 && gen % UPDATE_INTERVAL == 0) {
//       float tx_b = (float)(best_x[0] * params.tx_range);  // ... etc
//       for (m: organs) organs[m]->mVertices = snap_v[m];   // restore
//       applyIncrementalSRT(organs, tx_b, ..., sc_b);       // mutate
//       updateCorrespondences();                            // rebuild
//       for (m: organs) organs[m]->mVertices = snap_v[m];   // restore
//   }
//
// The double restore is what V1 needs because applyIncrementalSRT
// mutates organs[m]->mVertices in place; updateCorrespondences then
// reads liverMesh3D->mVertices (post-mutation) to build srcPts; and
// the second restore puts the inner loop back to its snapshot pose
// before the next generation.
//
// V3 design (plan D, finalized end of chat 5):
//   - S.base_positions stays FIXED for the lifetime of one Run. It
//     holds the jitter-applied snapshot of the liver, established by
//     build_eval_context_v3 and never resized or reordered after.
//   - The SRT-applied points are written to a stack-local buffer
//     `transformed`, used only as KDTree input for this refresh.
//   - Only S.tgt_to_eval is overwritten; everything else in S is
//     intact, so evaluate_one_v3 keeps interpreting tgt_to_eval[i]
//     as an index into the same S.base_positions across the
//     UPDATE_INTERVAL boundary.
//   - organs[*] are not touched. The "double restore" in V1 has no
//     V3 counterpart because V3 never mutated anything to begin with.
//
// bit-identical claim (V3-1, voxel = 0):
//   - cur_best is decoded from best_x via srt_from_xvec_v3, whose
//     formulas are byte-for-byte the V1 best_x decode at line 1482.
//   - S.centroid was set in build_eval_context_v3 from S.base_positions,
//     using compute_centroid_v3's V1-mirroring accumulation order.
//     Because V1 calls applyIncrementalSRT after `organs[m]->mVertices
//     = snap_v[m]` (so organs[0]->mVertices == snap_v[0] == V3's
//     S.base_positions, vertex-for-vertex), V1's recomputed centroid
//     equals V3's cached S.centroid.
//   - apply_srt_to_points uses the same per-vertex
//     glm::vec4(p,1)*M -> .x/.y/.z assignment pattern as V1's
//     applyIncrementalSRT. The output `transformed` therefore equals
//     V1's mutated mVertices vertex-for-vertex.
//   - V1's updateCorrespondences then repacks those mutated
//     mVertices into a vec3 array `srcPts`; V3 already has them as
//     vec3 in `transformed`. The two arrays are float-equal.
//   - KDTree adaptor / builder / search call / max_dist_sq gate are
//     all reused from build_eval_context_v3, which itself mirrors V1
//     line-for-line.
//   So tgt_to_eval[i] == V1's corr_idx[i] after the refresh.
// =====================================================================
inline void rebuild_correspondences_v3(EvalContextStaticV3& S,
                                       const SRTParamsV3&   cur_best,
                                       const ParamsV3&      params)
{
    // Defensive guard. build_eval_context_v3 already returns an
    // empty context for empty inputs and run_one_bipop will skip the
    // CMA-ES loop in that case, so this branch is purely defensive.
    if (S.base_positions.empty() || S.tgt_points.empty()) {
        if (params.verbose) {
            std::cerr << "[V3] rebuild_correspondences: empty context"
                      << "; skipping refresh." << std::endl;
        }
        return;
    }

    // ----- 1. SRT matrix using S.centroid (frozen at build time) ---
    const glm::mat4 M = build_srt_matrix_v3(cur_best, S.centroid);

    // ----- 2. Stack-local SRT-applied buffer ------------------------
    // Sized once per call (constructor + apply_srt_to_points's
    // internal resize). Only this function holds it; goes out of
    // scope at return so memory is reclaimed. Allocation cost is
    // O(N_base_positions); on a 132k-vertex mesh that's ~1.5 MB,
    // acceptable for a 10-call-per-Run refresh path.
    //
    // V3-4/V3-5 note: when run_one_bipop is parallelized over runs
    // (V3-5), each thread's RunContext owns its own EvalContextStaticV3,
    // so this stack-local buffer is naturally thread-private.
    std::vector<glm::vec3> transformed;
    apply_srt_to_points(M, S.base_positions, transformed);

    // ----- 3. Rebuild KDTree on `transformed`; refresh tgt_to_eval -
    // Same adaptor / builder / search call / gate as
    // build_eval_context_v3 step F. Only the input array differs
    // (S.base_positions there, `transformed` here). max_dist_sq is
    // unchanged across UPDATE_INTERVAL boundaries -- it depends only
    // on params.scene_diag, which V3 does not mutate.
    Reg3DCustom::NanoflannAdaptor adaptor(transformed);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    // Overwrite every entry. We don't .assign(N, -1) first because
    // every i is unconditionally assigned below (matched -> nnIdx,
    // unmatched -> -1), so a separate clear is wasted work.
    for (size_t i = 0; i < S.tgt_points.size(); i++) {
        size_t nnIdx;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, S.tgt_points[i], nnIdx, dist_sq)
            && dist_sq < S.max_dist_sq) {
            S.tgt_to_eval[i] = (int)nnIdx;
        } else {
            S.tgt_to_eval[i] = -1;
        }
    }
}

// =====================================================================
// Step 7: compute_full_rmse_local.   (V3-2 case C, rolled back from V3-3)
// ---------------------------------------------------------------------
// V3-ONLY HELPER. Compute the all-vertex RMSE of (liver_full with
// jitter then cma_best applied) against tgt_full, without touching
// any global state. This is the per-Run screening metric used to
// pick the best Run inside runBipopCmaesV3 and to set rc.improved.
//
// Direction note (chat 7 V3-3 attempt and rollback):
//   V3-3 inverted the KDTree direction (build on tgt_full, query
//   on transformed liver) hoping to cut the per-Run cost from
//   ~745 ms to ~250 ms. The attempt failed for two reasons:
//
//   1. Speed regression. nanoflann's KDTree build cost on 774k
//      unstructured tgt points was ~600-1000 ms per call, far
//      higher than predicted (~110 ms based on N log N scaling
//      from the 132k case). The smaller query count (132k vs
//      774k) did not compensate. Per-Run cost increased to
//      ~930 ms average.
//
//   2. Correctness regression. The liver->tgt RMSE is
//      systematically larger than the tgt->liver RMSE for
//      monocular-depth target clouds (which represent only the
//      visible front face of the scene), because liver back-face
//      vertices have no nearby tgt and are clipped by the
//      max_dist gate. Concretely: rmse_before was measured by
//      computeUnifiedMetrics in tgt->liver direction, but the
//      V3-3 best_rmse_full was measured liver->tgt; the gate
//      rc.improved = (best_rmse_full < rmse_before) failed for
//      all 10 Runs, marking the entire session as "no change".
//
//   The fix would have required either re-measuring rmse_before
//   in the new direction (additional 500+ ms session cost) or
//   accepting a different gate semantics. Neither was a clean
//   solution. The direction is restored to V3-2 here while
//   alternative strategies are evaluated:
//     - best_inner -only screening (drop full-mesh confirmation)
//     - voxel-tgt KDTree sharing with the inner loop
//     - parallelization of the 10 per-Run KDTree builds
//
// Design rationale (chat 6 plan, section 3.2): V3 must not call
// computeUnifiedMetrics inside run_one_bipop because that function
// writes to registrationHandle, reads liverMesh3D, and is therefore
// neither reentrant nor parallelizable. Replacing it with a pure
// function on (liver_*, tgt_full) is the single change that
// unlocks V3-5 run-level OpenMP parallelism.
//
// V3-2 case-C dual-resolution input:
//   liver_voxel: SAME cloud the inner CMA-ES loop operates on. The
//     two-stage SRT centroids (c_jitter, c_best) are computed from
//     this cloud so that the matrices used here EXACTLY match the
//     matrices the inner loop optimized against.
//   liver_full:  full-resolution cloud (organs[0]->mVertices). The
//     RMSE is evaluated by transforming this cloud through the
//     centered-at-voxel-centroid matrices and querying tgt_full.
//   In V3-1 (ratio=0) mode, liver_voxel == liver_full pointwise, so
//   the centroids and the RMSE evaluation cloud coincide, giving
//   bit-identical results to the previous (V3-1) compute_full_rmse_local
//   that took only one liver argument.
//
// Two-stage SRT composition (mirrors V1's two applyIncrementalSRT
// calls within one BIPOP run):
//   stage 1 (runBipopCmaes side, V1 line 1411-1417):
//     organs[*]->mVertices = start_v[*];  // restore to pre-jitter
//     applyIncrementalSRT(organs, jitter)
//       -> centroid = compute_centroid(start_v[0])  // pre-jitter
//   stage 2 (CmaesRefine::run epilogue, V1 line 1525-1530):
//     organs[m]->mVertices = snap_v[m];   // restore to post-jitter
//     applyIncrementalSRT(organs, best_x)
//       -> centroid = compute_centroid(snap_v[0])   // post-jitter
//
// V3-2 mirror (case C):
//   stage 1: c_jitter = compute_centroid_v3(liver_voxel)   // pre-jitter
//            M_jitter = build_srt_matrix_v3(jitter, c_jitter)
//            voxel_after_jitter = apply M_jitter to liver_voxel
//            (voxel-only side trip; used for c_best below)
//   stage 2: c_best = compute_centroid_v3(voxel_after_jitter)  // post-jitter
//            M_best = build_srt_matrix_v3(cma_best, c_best)
//   The matrices M_jitter and M_best are then applied separately
//   (NOT pre-composed) to liver_full -- preserves V1 byte-identical
//   numerics in the V3-1 (ratio=0) compatibility path.
//
// max_dist_sq numerics (intentional V1 unified-metrics match):
//   The caller passes a max_dist_sq computed from params.scene_diag
//   via (scene_diag / 7.36)^2. RegRatios::maxDistSq() in V1 evaluates
//   to the same expression (kRefMaxDist=1.0, scaleRatio = sceneDiag/7.36),
//   so this RMSE uses the same gate as V1's computeUnifiedMetrics
//   compRmse, which is what runBipopCmaes (Shift+V) uses for its
//   per-Run rmse_run comparison. -> Identical Run-best selection.
//
// count==0 fallback:
//   Returns 9.9f, matching V1's fastComputeRMSE rather than V1's
//   computeUnifiedMetrics (which would return 0). The 9.9 fallback
//   is correct here because this RMSE participates directly in
//   "smaller-is-better" Run selection; a 0 return on a totally
//   diverged Run would falsely mark it as the new best. The
//   divergence in semantics from compRmse only matters when count
//   genuinely reaches 0, which on a non-empty input never happens
//   for a well-posed registration.
//
// Cost (V3-2 case C):
//   O(N_voxel) for the voxel side trip (small, ~500 verts)
//   + O(N_full) for two full-cloud SRT applications
//   + O(N_full log N_full) for one KDTree build         (~180 ms @ 132k)
//   + O(N_tgt log N_full) for the search.               (~500 ms @ 774k)
//   On 132k liver verts and 774k tgt that's ~745 ms per call.
//   Called once per Run (10 calls per Shift+G session) -> ~7.5 s.
//   This is the current bottleneck of Shift+G. See V3-3 attempt
//   notes above for failed inversion approach.
// =====================================================================
inline float compute_full_rmse_local(
    const SRTParamsV3&            jitter,
    const SRTParamsV3&            cma_best,
    const std::vector<glm::vec3>& liver_voxel,
    const std::vector<glm::vec3>& liver_full,
    const std::vector<glm::vec3>& tgt_full,
    float                         max_dist_sq)
{
    if (liver_full.empty() || tgt_full.empty()) {
        return 9.9f;
    }

    // ----- Stage 1: jitter centroid from voxel pre-jitter cloud -----
    // Mirrors runBipopCmaes() line 1411-1417 (restore to start_v +
    // applyIncrementalSRT(organs, jitter)). The centroid SOURCE is
    // liver_voxel: in V3-2 this is the same cloud the inner loop
    // built its M_jitter against, so the screening matrix matches
    // exactly. In V3-1 (ratio=0) liver_voxel == liver_full so this
    // collapses to V1 byte-identical behavior.
    const glm::vec3 c_jitter = compute_centroid_v3(liver_voxel);
    const glm::mat4 M_jitter = build_srt_matrix_v3(jitter, c_jitter);

    // ----- Voxel-only side trip: compute c_best ---------------------
    // We need the centroid of the post-jitter VOXEL cloud (matches
    // what the inner loop used for its stage-2 matrix construction).
    // This is cheap because liver_voxel is small in V3-2 (~500 verts).
    // In V3-1 (ratio=0) this duplicates the full-cloud transform we
    // do below, but the duplication is harmless: it produces
    // bit-identical c_best values either way, and the cost is one
    // extra full-cloud apply (negligible vs the 50ms KDTree build).
    std::vector<glm::vec3> voxel_after_jitter;
    apply_srt_to_points(M_jitter, liver_voxel, voxel_after_jitter);
    const glm::vec3 c_best = compute_centroid_v3(voxel_after_jitter);
    const glm::mat4 M_best = build_srt_matrix_v3(cma_best, c_best);

    // ----- Apply two-stage transform to FULL liver ------------------
    // Two separate applies (NOT pre-composed M = M_best * M_jitter).
    // Pre-composing would round differently and break V1 byte-identical
    // reproduction in the ratio=0 compatibility check. Two separate
    // applies match V1's organs->mVertices = M_jitter*organs;
    // organs->mVertices = M_best*organs sequence verbatim.
    std::vector<glm::vec3> after_jitter;
    apply_srt_to_points(M_jitter, liver_full, after_jitter);

    std::vector<glm::vec3> after_best;
    apply_srt_to_points(M_best, after_jitter, after_best);

    // ----- KDTree on after_best; iterate tgt_full -------------------
    // V3-2 (rollback from V3-3 attempt): KDTree input is the
    // transformed liver, queries iterate tgt_full points.
    // Direction matches V1's fastComputeRMSE and V1's
    // computeUnifiedMetrics, both of which iterate target points
    // and look up source points.
    //
    // V3-3 attempted to invert this direction (KDTree on tgt_full,
    // queries on after_best) for performance reasons, but the
    // attempt regressed both speed (KDTree build on 774k tgt was
    // heavier than predicted) and correctness (the liver->tgt RMSE
    // was systematically larger than rmse_before, which is measured
    // tgt->liver, breaking the rc.improved gate so all 10 Runs were
    // rejected as "no improvement"). Direction is restored to V3-2
    // semantics here while alternative speedup strategies (best_inner
    // -only screening, voxel-tgt KDTree sharing) are evaluated.
    Reg3DCustom::NanoflannAdaptor adaptor(after_best);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    // [V3-3 案H] OpenMP parallel query loop.
    // KDTree (nanoflann) is read-only after build and thread-safe for
    // concurrent queries. OMP reduction on sumSq/count avoids any
    // shared-state write. Loop index is long long (required by OMP
    // for signed/unsigned compatibility across compilers).
    // Falls back to serial when compiled without -fopenmp.
    float     sumSq = 0.0f;
    int       count = 0;
    const long long n_tgt = (long long)tgt_full.size();
    // Pre-reserve after_best is already done above (full-liver size).
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumSq) reduction(+:count) schedule(static)
#endif
    for (long long i = 0; i < n_tgt; i++) {
        size_t nnIdx;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, tgt_full[(size_t)i], nnIdx, dist_sq)
            && dist_sq < max_dist_sq) {
            sumSq += dist_sq;
            count++;
        }
    }

    if (count == 0) return 9.9f;
    return std::sqrt(sumSq / (float)count);
}

// =====================================================================
// Step 8: run_one_bipop.
// ---------------------------------------------------------------------
// Drive one BIPOP-CMA-ES run end-to-end as a pure function over rc.
// Reads no globals; writes only to rc.{ctx,best_srt,best_rmse_*,improved,
// generations,stop_reason}. This is the unit of V3-5 run-level
// parallelism.
//
// V1 reference (CmaesUtils.h::CmaesRefine::run, lines 1280-1530, the
// plain-RMSE path):
//
//   cmaes_t* evo = cmaes_init(DIM=7, xstart, sigma0, lambda, lb, ub);
//   if (rng_seed) srand(rng_seed);
//   double best_x[7] = {0,...,0};
//   float best_rmse = result.rmse_before;
//
//   for (gen = 0; gen < maxgen && !stop; gen++) {
//       pop = cmaes_SamplePopulation(evo);
//       for (k = 0; k < lambda; k++) {
//           // restore organs[0] to snap, apply SRT, fastComputeRMSE, ...
//           bad = matched < min_ok || rmse == 0;
//           fval[k] = bad ? penalty : rmse;
//           if (fval[k] < best_rmse) { best_rmse = fval[k]; best_x = pop[k]; }
//       }
//       cmaes_UpdateDistribution(evo, fval);
//       if (gen > 0 && gen % UPDATE_INTERVAL == 0) {
//           // restore + applyIncrementalSRT(best_x) + updateCorrespondences + restore
//       }
//       stop = cmaes_TestForTermination(...);
//   }
//
// V3 deviations from V1 inner loop (all bit-identical-safe, see
// per-step comments in evaluate_one_v3 / rebuild_correspondences_v3):
//   - The k-loop replaces the "snap restore + applyIncrementalSRT +
//     fastComputeRMSE + read registrationHandle.compCount" sequence
//     with a single evaluate_one_v3(ctx, scratch, srt, &matched).
//   - The UPDATE_INTERVAL branch replaces the "double restore +
//     applyIncrementalSRT + updateCorrespondences" sequence with a
//     single rebuild_correspondences_v3(ctx, cur_best, params).
//   - No global state is touched, so this function is reentrant per
//     RunContext (the prerequisite for V3-5 OpenMP parallel runs).
//   - Wall-clock timing uses `now()` deltas at each phase boundary so
//     t_other can be derived as (loop_total - eval - rebuild) without
//     the V2 double-counting bug.
//
// Phase 1 objective gate: silhouette/contour objectives are not
// supported in V3 (Shift+G is the plain-RMSE successor to Shift+V).
// If params somehow gets silhouette flags through (V3 ParamsV3 has no
// such fields, so this is impossible from clean code paths), there
// is no place in the inner loop that would honor them, so the
// behaviour is plain RMSE regardless.
// =====================================================================
inline void run_one_bipop(RunContext& rc, const ParamsV3& params)
{
    // ----- 0. Validate inputs --------------------------------------
    // V3-2 case-C inputs: the inner loop reads liver_voxel_positions
    // (small) and tgt_voxel_points (small); the screening RMSE at
    // step 6 reads liver_full_positions and tgt_full_points (large).
    // All four pointers must be set by the caller.
    if (!rc.liver_voxel_positions || !rc.liver_full_positions
        || !rc.tgt_voxel_points    || !rc.tgt_full_points) {
        std::cerr << "[V3] run_one_bipop: null input pointer(s); "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "NullInput";
        return;
    }
    if (rc.liver_voxel_positions->empty() || rc.tgt_voxel_points->empty()) {
        std::cerr << "[V3] run_one_bipop: empty liver_voxel or tgt_voxel; "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "EmptyInput";
        return;
    }

    // ----- 1. Apply jitter to the VOXEL snapshot --------------------
    // V3-2 case C: jitter is applied to the small voxel cloud, not
    // the full mesh. The voxel cloud has been pre-built once per
    // session in runBipopCmaesV3 (before this loop) and is shared by
    // all 10 Runs, so the downsample work is done exactly once per
    // Shift+G session rather than 10 times.
    //
    // V1 reference: runBipopCmaes() applies jitter to organs via
    // applyIncrementalSRT(organs, jitter) before calling run(). In
    // V3-2 we apply the equivalent transform to liver_voxel only;
    // the full mesh transform is replayed at session end via
    // best_world_matrix.
    const glm::vec3 c_jitter = compute_centroid_v3(*rc.liver_voxel_positions);
    const glm::mat4 M_jitter = build_srt_matrix_v3(rc.jitter, c_jitter);
    std::vector<glm::vec3> liver_after_jitter;
    apply_srt_to_points(M_jitter, *rc.liver_voxel_positions, liver_after_jitter);

    // V3-2 normals: kept empty. evaluate_one_v3 doesn't read normals
    // in plain-RMSE mode, so transforming them here would be wasted
    // work (V1 transforms normals in applyIncrementalSRT but
    // fastComputeRMSE never reads them either, so the resulting RMSE
    // is identical). When a future objective needs them, populate
    // here via normalMat = mat3(transpose(inverse(M_jitter))) and
    // pass voxel-downsampled normals through liver_voxel_normals on
    // the RunContext.
    std::vector<glm::vec3> liver_normals_after_jitter;

    // ----- 1.5. Per-Step timing (chat 7) ----------------------------
    // Define `now`/`ms_dur` here (earlier than the original Step 5
    // definition) so we can measure Step 2 (build_eval_context) and
    // Step 6 (compute_full_rmse_local) too. These two are called once
    // per Run (vs the inner-loop evaluate_one / rebuild_corr) and are
    // therefore not captured by t_loop_total. Goal is to pin down the
    // ~890ms/run gap between per-Run total and runs-loop wall-clock.
    auto step_now    = []{ return std::chrono::high_resolution_clock::now(); };
    using step_ms    = std::chrono::duration<double, std::milli>;
    const auto t_step_t0 = step_now();

    // ----- 2. Build the eval context for this Run -------------------
    rc.ctx = build_eval_context_v3(
        liver_after_jitter,
        liver_normals_after_jitter,
        *rc.tgt_voxel_points,
        params,
        rc.init_matched);

    const auto t_step_t1 = step_now();
    const double t_step_build_eval =
        step_ms(t_step_t1 - t_step_t0).count();

    if (rc.ctx.base_positions.empty() || rc.ctx.tgt_points.empty()) {
        std::cerr << "[V3] run_one_bipop: build_eval_context failed; "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "EmptyContext";
        return;
    }

    if (params.verbose) {
        std::cout << "[V3] Run " << (rc.run_index + 1)
        << "  " << (rc.is_local_regime ? "Local" : "Global")
        << "  sigma0=" << rc.sigma0
        << "  cma_seed=" << rc.cma_seed << std::endl;
    }

    // ----- 3. cmaes_init + deterministic srand override -------------
    // Order matches V1 (CmaesUtils.h lines 1299-1318): cmaes_init runs
    // its internal srand(time(NULL)), then we override with the
    // caller-provided seed. This is the determinism contract.
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
            std::cout << "[V3] Deterministic seed: " << rc.cma_seed
                      << std::endl;
        }
    }

    // ----- 4. CMA-ES state ------------------------------------------
    // V1 initialises best_x = {0..0} and best_rmse = result.rmse_before,
    // where result.rmse_before is registrationHandle.compRmse measured
    // AFTER jitter is applied (CmaesUtils.h lines 1217-1218, 1274, 1321).
    // best_x represents identity SRT; best_rmse is the RMSE at the
    // post-jitter pose (= rc.ctx.initial_rmse, freshly computed in
    // build_eval_context_v3 step F-2 with the V1 unified-metrics formula).
    //
    // Sourcing best_rmse from rc.rmse_before instead -- as earlier V3
    // builds did -- breaks bit-identical reproduction for Runs whose
    // jitter degrades the RMSE meaningfully: the inner loop never
    // beats the (lower) session-start threshold so best_x stays at
    // identity, the Run terminates with no improvement, and the final
    // best_full equals the post-jitter RMSE.
    double best_x[DIM] = {0,0,0,0,0,0,0};
    float  best_rmse   = rc.ctx.initial_rmse;

    // V3-1: single-threaded, scratch_pool size 1. V3-4 will resize
    // to omp_get_max_threads() and dispatch evaluate_one_v3 calls
    // from a parallel for over k.
    std::vector<EvalContextScratchV3> scratch_pool(1);

    // ----- 5. CMA-ES main loop --------------------------------------
    const char* stop = nullptr;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    using ms_dur = std::chrono::duration<double, std::milli>;

    // Wall-clock for the whole loop, used to derive "other" time as
    // (loop_total - eval - rebuild). V2 had a double-counting bug
    // when t_other was computed from per-iteration timestamps; this
    // residual approach is structurally immune.
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
            float rmse    = evaluate_one_v3(rc.ctx, scratch_pool[0],
                                         srt, matched);

            // V1 penalty rule (CmaesUtils.h line 1441): bad iff
            // matched < min_ok || rmse == 0. matched_min_required
            // was computed in build_eval_context_v3 with the same
            // formula V1 uses (max(10, init_matched * min_match_ratio)).
            const bool bad = (matched < rc.ctx.matched_min_required)
                             || (rmse == 0.0f);
            fval[k] = bad ? (double)params.penalty_value : (double)rmse;

            // Order-preserving best tracking. V3-4 will keep this
            // OUTSIDE the parallel-for (collected per thread, merged
            // sequentially) so the best_x state remains deterministic
            // under OMP.
            if (fval[k] < best_rmse) {
                best_rmse = (float)fval[k];
                for (int d = 0; d < DIM; d++) best_x[d] = pop[k][d];
            }
        }

        auto tg1 = now();
        t_eval += ms_dur(tg1 - tg0).count();

        cmaes_UpdateDistribution(evo, fval.data());

        // UPDATE_INTERVAL refresh. V1 uses a hard-coded constant 10;
        // V3 exposes it as params.update_interval (default 10) for
        // experiments without touching V1.
        if (gen > 0 && gen % params.update_interval == 0) {
            auto tr0 = now();
            const SRTParamsV3 cur_best = srt_from_xvec_v3(best_x, params);
            rebuild_correspondences_v3(rc.ctx, cur_best, params);
            auto tr1 = now();
            t_rebuild += ms_dur(tr1 - tr0).count();
        }

        if (params.verbose && (gen % params.log_every == 0)) {
            std::cout << "[V3] Gen " << std::setw(4) << gen
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

    // ----- 6. Decode best, compute full-mesh screening RMSE ---------
    rc.best_srt        = srt_from_xvec_v3(best_x, params);
    rc.best_rmse_inner = best_rmse;   // CMA-ES gauge (voxel-scale
        // when V3-2 is on; full-mesh
        // when V3-1 / voxel ratio = 0)

    // max_dist_sq is recomputed from params.scene_diag rather than
    // reused from rc.ctx.max_dist_sq. In V3-1 the two are equal, but
    // in V3-2 rc.ctx.max_dist_sq may be sized for the voxel-downsampled
    // scale (still derived from scene_diag, but conceptually distinct);
    // the all-vertex screening RMSE here always uses the all-vertex
    // gate, which matches V1 computeUnifiedMetrics and V1 fastComputeRMSE
    // (both use g_sceneDiag/7.36 unconditionally).
    constexpr float kRefSceneDiag_full = 7.36f;
    const float max_dist_full    = params.scene_diag
                                * (1.0f / kRefSceneDiag_full);
    const float max_dist_sq_full = max_dist_full * max_dist_full;

    const auto t_step_t2 = step_now();
    rc.best_rmse_full = compute_full_rmse_local(
        rc.jitter, rc.best_srt,
        *rc.liver_voxel_positions,   // centroid source (matches inner loop)
        *rc.liver_full_positions,    // RMSE evaluation cloud
        *rc.tgt_full_points,
        max_dist_sq_full);
    const auto t_step_t3 = step_now();
    const double t_step_full_rmse =
        step_ms(t_step_t3 - t_step_t2).count();

    // V1 screening compares the final RMSE to rmse_before * 1.001 to
    // skip noise-only "improvements". V3 uses a strict less-than
    // against rmse_before for run-level filtering; the session-level
    // best selection in runBipopCmaesV3 picks the run with the
    // smallest best_rmse_full regardless. Mirroring the V1 1.001
    // padding inside this gate would introduce a constant factor
    // that runBipopCmaesV3's MIN selection can't undo on a tied
    // session, so we leave it strict here.
    rc.improved = (rc.best_rmse_full < rc.rmse_before);

    // ----- 7. Time-breakdown log ------------------------------------
    if (params.verbose) {
        const int    total_evals    = rc.generations * evo_lambda;
        const double t_total        = t_loop_total;
        const double t_other        = t_total - t_eval - t_rebuild;
        const int    rebuild_calls  = (rc.generations - 1)
                                  / params.update_interval;

        std::cout << std::fixed << std::setprecision(1)
                  << "[V3] === Run " << (rc.run_index + 1)
                  << " Time Breakdown (total " << (int)t_total
                  << " ms, " << total_evals << " evals) ===" << std::endl;
        if (t_total > 0.0) {
            std::cout << "[V3]   evaluate_one (sum) : " << (int)t_eval
                      << " ms (" << (int)(100*t_eval/t_total) << "%)"
                      << std::endl
                      << "[V3]   rebuild_corr (sum) : " << (int)t_rebuild
                      << " ms (" << (int)(100*t_rebuild/t_total) << "%)"
                      << "  [" << rebuild_calls << " calls]"
                      << std::endl
                      << "[V3]   cmaes/log/other    : " << (int)t_other
                      << " ms (" << (int)(100*t_other/t_total) << "%)"
                      << std::endl;
        }
        // chat 7: per-Run "outside the CMA-ES loop" cost. The per-Run
        // wall-clock at the runBipopCmaesV3 caller side is approximately
        // (t_total + t_step_build_eval + t_step_full_rmse + ~few ms
        // RunContext setup). If t_step_full_rmse dominates, that's the
        // optimization target (full-mesh KDTree built once per Run).
        std::cout << "[V3]   build_eval_ctx     : "
                  << (int)t_step_build_eval << " ms"
                  << "   compute_full_rmse  : "
                  << (int)t_step_full_rmse << " ms"
                  << std::endl;
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "[V3] Run " << (rc.run_index + 1)
                  << "  best_inner=" << rc.best_rmse_inner
                  << "  best_full="  << rc.best_rmse_full
                  << "  stop="       << rc.stop_reason
                  << (rc.improved ? "  [+]" : "  [-]")
                  << std::endl;
    }
}

// =====================================================================
// Step 9: runBipopCmaesV3.
// ---------------------------------------------------------------------
// Session driver: run 10 BIPOP-CMA-ES instances against the same
// session-shared data and return a single ResultV3 describing the
// best Run's transform from snapshot pose to final pose.
//
// API contract:
//   - All inputs are passed by value/const-ref. No mCutMesh*, no
//     globals read or written here. The caller's RegistrationActions
//     wrapper (Shift+G entry point) is responsible for:
//       (i)  converting liverMesh3D->mVertices to vec3,
//       (ii) extracting tgt via extractFrontFacePoints,
//       (iii) reading rmse_before / init_matched via computeUnifiedMetrics
//             once at session start,
//       (iv) computing outer_seed / cma_base from g_trialSeed/g_callIdx
//             with the formulas matched to V1 / V2,
//       (v)  applying ResultV3::best_world_matrix to all 6 organs
//            after this returns,
//       (vi) calling g_callIdx++ at session end.
//
// Determinism contract (must match V1 Shift+V byte-for-byte for the
// V3-1 voxel=0 path):
//   - outer_seed and cma_base formulas (caller-side, passed in here):
//       outer_seed = g_trialSeed + 1000u + g_callIdx * 97u
//       cma_base   = g_trialSeed + 2000u + g_callIdx * 10u
//   - mt19937 is seeded with outer_seed.
//   - d01(rng) consumption order PER RUN is exactly 8 calls in the
//     order: sigma0, tx, ty, tz, rx, ry, rz, scale.
//     Both regimes (Local and Global) use the same 8-call ordering;
//     only the multipliers differ. This matches V1 line 1102-1108
//     and 1112-1118 verbatim.
//   - run = 0 still consumes its 8 d01 values, even though the
//     resulting jitter is overwritten with identity afterwards.
//     V1's `if (run > 0)` gate at line 1122-1125 skips the jitter
//     APPLY but not the jitter GENERATION, so the outer rng state
//     after run 0 is identical to V1's. Mirror this exactly.
//   - rc.cma_seed = cma_base + run, the same formula V1 / V2 use.
//
// Best-Run selection:
//   The Run with the smallest rc.best_rmse_full is chosen as best.
//   compute_full_rmse_local uses the same max_dist_sq gate as V1's
//   computeUnifiedMetrics (params.scene_diag / 7.36)^2, so for
//   non-pathological inputs (count > 0) this picks the same Run V1
//   would have picked via per-Run computeUnifiedMetrics calls.
//
// Output transform:
//   ResultV3::best_world_matrix = M_best * M_jit, where
//     M_jit  = build_srt_matrix_v3(best_jitter,
//                                  centroid_of(start_liver_verts))
//     M_best = build_srt_matrix_v3(best_srt,
//                                  centroid_of(start_liver_verts AFTER jit))
//   This is the same composition compute_full_rmse_local applies
//   internally for the screening RMSE, so the all-vertex RMSE
//   reported in r.rmse_after equals the full-mesh RMSE the caller
//   will measure via computeUnifiedMetrics after applying the matrix.
//
//   When no Run improves on rmse_before, best_world_matrix is
//   identity, r.rmse_after == r.rmse_before, r.improved == false.
//   The caller can short-circuit organ writeback in this case.
// =====================================================================
inline ResultV3 runBipopCmaesV3(
    const std::vector<glm::vec3>& start_liver_verts,
    const std::vector<glm::vec3>& start_liver_normals,
    const std::vector<glm::vec3>& tgt_points,
    const ParamsV3&               params,
    float                         rmse_before,
    int                           init_matched,
    uint32_t                      outer_seed,
    uint32_t                      cma_base)
{
    ResultV3 r;
    r.rmse_before = rmse_before;
    r.rmse_after  = rmse_before;
    r.improved    = false;

    // ----- Empty-input early return ---------------------------------
    if (start_liver_verts.empty() || tgt_points.empty()) {
        std::cerr << "[V3] runBipopCmaesV3: empty start_liver_verts ("
                  << start_liver_verts.size() << ") or tgt_points ("
                  << tgt_points.size() << "); aborting." << std::endl;
        return r;
    }

    // ----- Pre-compute session-wide max_dist_sq ---------------------
    // Used as a default fill on RunContext.max_dist_sq. run_one_bipop
    // currently re-derives this from params.scene_diag for clarity;
    // exposing it on RunContext anyway lets future call sites that
    // share session state (e.g. logging, alternate evaluators) avoid
    // a recomputation.
    constexpr float kRefSceneDiag_session = 7.36f;
    const float max_dist_session    = params.scene_diag
                                   * (1.0f / kRefSceneDiag_session);
    const float max_dist_sq_session = max_dist_session * max_dist_session;

    // ----- V3-2 case C: voxel-downsample ONCE per session -----------
    // Both src (liver) and tgt are voxel-downsampled here, BEFORE the
    // Run loop. The result is shared by all 10 Runs as const data:
    //   * Voxel work is done exactly once per Shift+G session, not
    //     once per Run as in a "voxelize-inside-build_eval_context"
    //     design. For the heavy mesh (132k src, 800k tgt) this saves
    //     ~9x on voxel cost.
    //   * The downsampled cloud is a property of the input geometry,
    //     not of any per-Run jitter. This keeps the Run-to-Run
    //     starting state numerically deterministic: the same vertex
    //     set is jittered identically by 10 different SRTs.
    //   * voxel_downsample_v3 lex-sorts its output, so even when the
    //     underlying voxelDownSample returns implementation-defined
    //     iteration order, downstream KDTrees see a canonical order.
    //
    // ratio == 0 contract: voxel_downsample_v3 short-circuits to a
    // verbatim copy, preserving V3-1 byte-identical numerics.
    //
    // Chat 7 timing diagnostics: stamp around the voxel work and the
    // 10-Run loop separately so we can attribute the Phase D wall-clock
    // (logged by RegistrationActions::runBipopCmaesV3) to its sub-parts.
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
        std::cout << "[V3 session/Time] voxel src ("
                  << start_liver_verts.size() << "->"
                  << session_voxel_liver.size() << ") : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_voxel1 - t_sess_voxel0).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3 session/Time] voxel tgt ("
                  << tgt_points.size() << "->"
                  << session_voxel_tgt.size() << ") : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_voxel2 - t_sess_voxel1).count()
                  << " ms" << std::defaultfloat << std::endl;
    }

    // ----- BIPOP outer rng init -------------------------------------
    std::mt19937 rng(outer_seed);
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    if (params.verbose) {
        std::cout << "[V3] === Starting BIPOP-CMA-ES V3 ===" << std::endl
                  << "[V3] outer_seed=" << outer_seed
                  << "  cma_base=" << cma_base
                  << "  rmse_before=" << rmse_before
                  << "  init_matched=" << init_matched
                  << "  scene_diag=" << params.scene_diag
                  << std::endl
                  << "[V3] src: " << start_liver_verts.size()
                  << " -> " << session_voxel_liver.size()
                  << " (voxel=" << src_voxel_size << ", ratio="
                  << params.src_voxel_ratio << ")" << std::endl
                  << "[V3] tgt: " << tgt_points.size()
                  << " -> " << session_voxel_tgt.size()
                  << " (voxel=" << tgt_voxel_size << ", ratio="
                  << params.tgt_voxel_ratio << ")" << std::endl;
    }

    // ----- 10 BIPOP runs --------------------------------------------
    const int N_STARTS = 10;

    float       best_rmse_full     = rmse_before;
    int         best_run_idx       = -1;
    SRTParamsV3 best_jitter;
    SRTParamsV3 best_srt;
    std::string best_stop_reason   = "NoImprovement";
    int         total_generations  = 0;

    // Wall-clock around the entire 10-Run loop. The per-Run "[V3] ===
    // Run k Time Breakdown" lines printed inside run_one_bipop measure
    // only the CMA-ES inner loop; this outer measurement also captures
    // RunContext setup, jitter generation, build_eval_context_v3, and
    // compute_full_rmse_local. Comparing this against the sum of the
    // per-Run "totals" reveals how much overhead lives outside the
    // CMA-ES inner loop.
    auto t_sess_runs0 = sess_now();
    double t_run_outer_wall_sum = 0.0;

    // ----------------------------------------------------------------
    // [V3 V3-5] Run-level parallelism: two-phase loop structure.
    // ----------------------------------------------------------------
    // Phase 1 (serial): Pre-generate sigma0 + jitter for all Runs.
    // mt19937 is NOT thread-safe; all d01/rng draws must be serial
    // and in the same order to preserve determinism.
    //
    // Phase 2 (parallel): Execute each Run independently with OMP.
    //   - best_rmse_full tracking: local per-run, merged after loop
    //   - total_generations / t_run_outer_wall_sum: #pragma omp atomic
    //   - cmaes.c thread safety: handled by cmaes_tls_wrapper.c
    //   - compute_full_rmse_local (called inside run_one_bipop) has
    //     an inner #pragma omp parallel for (案H). When called from
    //     within a parallel region, OMP nested parallelism is disabled
    //     by default, so compute_full_rmse runs single-threaded per
    //     Run. The 10x run-level parallelism still applies to the
    //     CMA-ES loop and the KDTree build in compute_full_rmse.
    // ----------------------------------------------------------------

    // ----- Phase 1: serial pre-generation of sigma0 + jitter --------
    struct RunPreGenV3 {
        double      sigma0;
        SRTParamsV3 jitter;
    };
    std::vector<RunPreGenV3> pre_gen(N_STARTS);
    for (int run = 0; run < N_STARTS; run++) {
        const bool is_local = (run % 2 == 0);
        RunPreGenV3& pg = pre_gen[run];
        if (is_local) {
            pg.sigma0         = 0.3 + d01(rng) * 0.4;
            const float lt    = params.jitter_local_t;
            pg.jitter.tx      = (d01(rng) * 2.0f - 1.0f) * lt;
            pg.jitter.ty      = (d01(rng) * 2.0f - 1.0f) * lt;
            pg.jitter.tz      = (d01(rng) * 2.0f - 1.0f) * lt;
            pg.jitter.rx_deg  = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            pg.jitter.ry_deg  = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            pg.jitter.rz_deg  = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            pg.jitter.scale   = 0.95f + d01(rng) * 0.10f;
        } else {
            pg.sigma0         = 0.5 + d01(rng) * 0.5;
            const float gt    = params.jitter_global_t;
            pg.jitter.tx      = (d01(rng) * 2.0f - 1.0f) * gt;
            pg.jitter.ty      = (d01(rng) * 2.0f - 1.0f) * gt;
            pg.jitter.tz      = (d01(rng) * 2.0f - 1.0f) * gt;
            pg.jitter.rx_deg  = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            pg.jitter.ry_deg  = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            pg.jitter.rz_deg  = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            pg.jitter.scale   = 0.90f + d01(rng) * 0.20f;
        }
        if (run == 0) pg.jitter = SRTParamsV3{};
    }

    // ----- Phase 2: parallel Run execution --------------------------
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(N_STARTS)
#endif
    for (int run = 0; run < N_STARTS; run++) {
        const auto t_one_run_t0 = sess_now();
        RunContext rc;
        rc.run_index             = run;
        rc.is_local_regime       = (run % 2 == 0);
        rc.cma_seed              = cma_base + (uint32_t)run;
        rc.rmse_before           = rmse_before;
        rc.init_matched          = init_matched;
        rc.max_dist_sq           = max_dist_sq_session;
        rc.liver_full_positions  = &start_liver_verts;
        rc.liver_full_normals    = &start_liver_normals;
        rc.liver_voxel_positions = &session_voxel_liver;
        rc.tgt_full_points       = &tgt_points;
        rc.tgt_voxel_points      = &session_voxel_tgt;

        rc.sigma0 = pre_gen[run].sigma0;
        rc.jitter = pre_gen[run].jitter;

        if (params.verbose) {
            std::cout << "[V3] Run " << (run+1) << "/" << N_STARTS
                      << "  " << (rc.is_local_regime ? "Local " : "Global")
                      << "  sigma0=" << std::fixed << std::setprecision(4)
                      << rc.sigma0
                      << "  cma_seed=" << rc.cma_seed
                      << std::defaultfloat << std::setprecision(6)
                      << std::endl;
        }

        // ----- Execute one Run (pure wrt globals) -------------------
        run_one_bipop(rc, params);

        // ----- Track best Run (critical section) --------------------
        // OMP critical is needed because best_rmse_full / best_run_idx
        // are read-modify-write across threads.
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (rc.best_rmse_full < best_rmse_full) {
                best_rmse_full   = rc.best_rmse_full;
                best_run_idx     = run;
                best_jitter      = rc.jitter;
                best_srt         = rc.best_srt;
                best_stop_reason = rc.stop_reason;
            }
        }

#ifdef _OPENMP
#pragma omp atomic
#endif
        total_generations += rc.generations;

        const auto t_one_run_t1 = sess_now();
        const double run_wall_ms = sess_ms(t_one_run_t1 - t_one_run_t0).count();
#ifdef _OPENMP
#pragma omp atomic
#endif
        t_run_outer_wall_sum += run_wall_ms;
    }  // end parallel for

    auto t_sess_runs1 = sess_now();
    if (params.verbose) {
        std::cout << "[V3 session/Time] runs loop wall-clock : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_runs1 - t_sess_runs0).count()
                  << " ms"
                  << "  (sum of per-run outer = "
                  << t_run_outer_wall_sum << " ms)"
                  << std::defaultfloat << std::endl;
    }

    // ----- Assemble best_world_matrix ------------------------------
    if (best_run_idx >= 0) {
        // Two-stage SRT composition. The centroids are sourced from
        // the VOXEL cloud (session_voxel_liver), MATCHING what the
        // CMA-ES inner loop optimized against and what
        // compute_full_rmse_local used to compute best_rmse_full.
        //
        // This consistency is critical: best_rmse_full is the
        // screening metric that determined which Run is "best", so
        // the matrix the caller applies to organs MUST produce
        // exactly that RMSE when re-measured. Sourcing centroids
        // from start_liver_verts (full mesh) instead would build a
        // slightly different matrix when src_voxel_ratio > 0, making
        // post-application computeUnifiedMetrics disagree with
        // r.rmse_after by a few ulps to a few mm depending on
        // voxel_ratio.
        //
        // V3-1 (ratio=0) compatibility: session_voxel_liver is a
        // verbatim copy of start_liver_verts in this mode, so
        // c_pre/c_post are bit-identical to the previous V3-1
        // implementation.
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
        // No Run improved on rmse_before. Identity transform.
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
        std::cout << "[V3] === BIPOP-CMA-ES V3 DONE ==="
                  << "  best_run="
                  << (best_run_idx < 0 ? std::string("none")
                                       : std::to_string(best_run_idx + 1))
                  << "  RMSE: " << rmse_before << " -> " << r.rmse_after
                  << "  delta=" << (rmse_before - r.rmse_after)
                  << (r.improved ? "  [IMPROVED]" : "  [NO CHANGE]")
                  << "  total_gens=" << total_generations
                  << std::endl
                  << "[V3 session/Time] DRIVER TOTAL : "
                  << std::fixed << std::setprecision(1)
                  << t_session_total << " ms"
                  << std::defaultfloat << std::endl;
    }

    // (g_callIdx++ is the caller's responsibility, performed in the
    //  RegistrationActions::runBipopCmaesV3 wrapper after this returns.)

    return r;
}

} // namespace CmaesRefine

#endif // CMAES_REFINE_V3_H
