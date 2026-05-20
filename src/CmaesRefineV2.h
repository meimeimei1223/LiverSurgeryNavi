#pragma once
#ifndef CMAES_REFINE_V2_H
#define CMAES_REFINE_V2_H
/*
 * CmaesRefineV2.h
 * ----------------------------------------------------------------------
 * V2 implementation of CMA-ES post-registration refinement.
 * V1 (CmaesUtils.h :: CmaesRefine::run) is preserved untouched and
 * remains dispatched by Shift+V; V2 is dispatched by Shift+F ("Fast").
 *
 * Goals (vs V1):
 *   1. Pure-function inner-loop evaluation (no global-state writes inside
 *      evaluate_one), so OpenMP-parallel population evaluation is safe.
 *   2. SUBSET_RMSE mode: evaluate SRT/RMSE only on the unique src vertices
 *      that have a tgt correspondence (typically a few thousand) instead of
 *      the full liver mesh (~132k verts).
 *   3. FULL_MESH mode: bit-identical reproducibility with V1, used as a
 *      reference path to verify the architectural refactor is correct.
 *
 * Phasing (see chat-5 plan, sections 4.1-4.4):
 *   Phase 1: Skeleton + FULL_MESH mode, single-threaded. Validation: V1
 *            and V2 produce CompRMSE matching to 8 decimal places.
 *   Phase 2: Add SUBSET_RMSE mode. Validation: CompRMSE within +/-5%,
 *            IoU2D within +/-0.02, runtime <= V1/3.
 *   Phase 3: OpenMP over the population (k = 0..lambda). Validation:
 *            still bit-identical to single-threaded V2 (best update is
 *            kept outside the parallel region for order preservation).
 *   Phase 4: PoseLibrary integration; dispatch migration. (Separate chat.)
 *
 * Determinism contract (must hold across all phases):
 *   - outer_seed and cma_base formulas in runBipopCmaesV2() are identical
 *     to runBipopCmaes() (V1).
 *   - p.rng_seed = cma_base + run, srand(rng_seed) called immediately
 *     after cmaes_init() (same timing as V1).
 *   - g_callIdx++ at the END of runBipopCmaesV2() (same as V1).
 *   - The d01(rng) consumption order in the BIPOP outer loop is identical
 *     to V1 (sigma0, tx, ty, tz, rx, ry, rz, sc).
 *
 * V1 inner-loop semantics that V2 reproduces:
 *   - Only organs[0] (liver) is transformed during evaluation; other
 *     organs are touched only at UPDATE_INTERVAL boundaries and at final
 *     writeback.
 *   - The around-centroid SRT uses the centroid of the pre-SRT (snapshot)
 *     liver vertices. In FULL_MESH mode V2 caches this centroid once,
 *     which is numerically identical to V1's per-call recomputation
 *     (same input, same naive accumulation order).
 *   - Correspondences (tgt -> src) are rebuilt every UPDATE_INTERVAL
 *     generations from snap_v[0] transformed by the current best_x.
 * ----------------------------------------------------------------------
 */

// ----------------------------------------------------------------------
// V1 include-order self-sufficiency fix.
// CmaesUtils.h::run() references the global `liverMesh3D` directly
// inside its inline lambdas (updateCorrespondences / fastComputeRMSE),
// WITHOUT an extern declaration of that symbol. The existing build
// works only because main.cpp happens to declare liverMesh3D BEFORE
// any header that pulls CmaesUtils.h in. Including this V2 header from
// an earlier point in main.cpp surfaces that fragile dependency as a
// parse error ("'liverMesh3D' was not declared in this scope" inside
// the V1 lambda body).
//
// Pull mCutMesh.h directly and extern-declare liverMesh3D BEFORE we
// include CmaesUtils.h, so this header is parseable from any TU
// regardless of include order. V1 source is left untouched.
// ----------------------------------------------------------------------
#include "mCutMesh.h"
extern mCutMesh* liverMesh3D;

#include "CmaesUtils.h"   // Reuse V1: Params, Result, applyIncrementalSRT

#include <vector>
#include <glm/glm.hpp>

namespace CmaesRefine {

// =====================================================================
// EvalMode -- evaluation strategy.
// =====================================================================
enum class EvalMode {
    // Resolved by resolve_eval_mode(): silhouette/contour objectives are
    // forced to FULL_MESH (those evaluators need the full transformed mesh,
    // not just the corresponded subset). Plain RMSE picks voxel-
    // downsampled processing per ParamsV2::{src,tgt}_voxel_ratio.
    AUTO,

    // V1-compatible path. src_voxel_ratio / tgt_voxel_ratio are forced
    // to 0 internally (no downsample), giving the entire liver mesh
    // and the entire tgt cloud. Used as the bit-identical validation
    // mode for Phase 1.
    FULL_MESH,

    // Deprecated since Phase 2.5. Originally: a correspondence-derived
    // unique-vertex subset of the src side. Replaced by symmetric voxel
    // downsample (HemiAuto-style) which is simpler, cheaper, and
    // addresses the actual bottleneck (the 774k-point tgt RMSE scan).
    // Treated as a synonym for AUTO at runtime; left in the enum so old
    // PoseLibrary entries / persisted Params can still parse.
    SUBSET_RMSE,
    };

// =====================================================================
// SRTParams -- 7-DoF parameters in physical units.
// Decoded from the [-1,1]-normalized CMA-ES population vector by
// srt_from_population() (defined in a later step).
// =====================================================================
struct SRTParams {
    float tx     = 0.0f;
    float ty     = 0.0f;
    float tz     = 0.0f;
    float rx_deg = 0.0f;
    float ry_deg = 0.0f;
    float rz_deg = 0.0f;
    float scale  = 1.0f;
};

// =====================================================================
// ParamsV2 -- extends V1 Params with V2-specific knobs.
// All V1 fields (tx_range, rng_seed, sigma0, ...) are inherited unchanged
// so existing setup code in runBipopCmaes() can be copied verbatim.
// =====================================================================
struct ParamsV2 : public Params {
    // Evaluation strategy.
    //   AUTO       (default): voxel-downsampled point clouds with the
    //                ratios below. Plain RMSE objective only; if
    //                use_silhouette_2d / use_boundary_weight is set,
    //                AUTO falls back to FULL_MESH (mesh topology kept).
    //   FULL_MESH: forces voxel ratios to 0 (no downsample) and uses
    //                the entire src/tgt point sets. Used to reproduce
    //                V1 bit-identically.
    //   SUBSET_RMSE: deprecated since Phase 2.5 (correspondence-based
    //                src subset). Treated as a synonym for AUTO; do
    //                not rely on the historical semantics.
    EvalMode eval_mode = EvalMode::AUTO;

    // Voxel downsample ratios (multipliers on g_sceneDiag).
    // 0 disables downsampling on that side. Reference values:
    //   HemiAuto uses g_sceneDiag * 0.0408 (= ~9 cm at sceneDiag=2.2m)
    //     for its FGR coarse alignment -- intentionally coarser than
    //     the literature 5L choice (see main.cpp::1349).
    //   Phase 2.5 uses ~0.015 (~3 cm at sceneDiag=2.2m), about 36%
    //     of HemiAuto's voxel, which is the right granularity for a
    //     refinement step (finer than coarse FGR, still coarse enough
    //     to give a large speedup vs the full 774k-point tgt scan).
    // Both are set to non-zero by runBipopCmaesV2 in AUTO mode.
    // FULL_MESH forces both to 0 internally.
    float src_voxel_ratio = 0.0f;
    float tgt_voxel_ratio = 0.0f;

    // OpenMP toggle for the population loop (k = 0..lambda).
    // Default off in Phase 1-2; flipped on in Phase 3 once bit-identical
    // behaviour vs the single-threaded V2 path is confirmed.
    bool parallel_population = false;

    // Thread cap. 0 = omp_get_max_threads(). Effective threads is
    // min(lambda, max_threads) when parallel_population == true.
    int max_threads = 0;

    // V1 hard-coded UPDATE_INTERVAL = 10. Exposed here so that experiments
    // can change correspondence refresh frequency without touching V1.
    int update_interval = 10;
};

// =====================================================================
// EvalContextStatic -- read-only, shared across threads and evaluations.
// Built once by build_eval_context() and refreshed in place by
// rebuild_correspondences() at UPDATE_INTERVAL boundaries.
// =====================================================================
struct EvalContextStatic {
    // Resolved evaluation mode (never AUTO after build).
    EvalMode mode = EvalMode::FULL_MESH;

    // Base point set in pre-SRT (snapshot) coordinates.
    //   FULL_MESH    : all liver vertices, in original mesh order.
    //   SUBSET_RMSE  : unique vertices that have a tgt correspondence,
    //                  sorted by full-vertex-index for determinism.
    // SRT is applied to these points fresh on every evaluation, into
    // EvalContextScratch::work_positions.
    std::vector<glm::vec3> base_positions;
    std::vector<glm::vec3> base_normals;

    // Around-centroid SRT origin. Computed from base_positions with the
    // same naive accumulation order used by V1's applyIncrementalSRT
    // (for i in 0..N : c += v[i]; c /= N), so FULL_MESH matches V1 bit-
    // for-bit. In SUBSET_RMSE mode this is the centroid of the subset
    // (intentionally different from V1 -- see plan section 3.6).
    glm::vec3 centroid = glm::vec3(0.0f);

    // tgt[i] -> index into base_positions / base_normals. -1 means no
    // correspondence (this i contributes nothing to the RMSE sum).
    //   FULL_MESH    : value = src vertex index (identity remapping).
    //   SUBSET_RMSE  : value = position within the subset.
    std::vector<int> tgt_to_eval;

    // Target point cloud, extracted once at build time and held constant
    // for the entire CMA-ES run (V1 caches this in targetCloud_cache).
    std::vector<glm::vec3> tgt_points;

    // Reverse mapping: subset_idx -> full vertex index in snap_v[0].
    // For FULL_MESH this is the identity (eval_to_full_idx[k] == k).
    // Used by rebuild_correspondences when refreshing the subset.
    std::vector<int> eval_to_full_idx;

    // Squared correspondence-distance gate. Pairs with squared distance
    // >= max_dist_sq are dropped from the RMSE accumulation. Set from
    // g_sceneDiag * (1.0 / kRefSceneDiag), the same scale-invariant
    // constant used by V1.
    float max_dist_sq = 0.0f;

    // Penalty gate. If matched < matched_min_required, the population
    // member is assigned params.penalty_value (V1 default 9.9).
    // Computed at build time as max(10, init_matched * min_match_ratio).
    int matched_min_required = 10;
};

// =====================================================================
// EvalContextScratch -- per-evaluation work memory.
// In Phase 3 (parallel_population), each OpenMP thread owns one of these
// to keep work_positions writes thread-local. 64-byte alignment prevents
// false-sharing on the cacheline that holds matched_count / last_rmse.
// =====================================================================
struct alignas(64) EvalContextScratch {
    // SRT-applied base_positions, sized to base_positions.size() lazily
    // by evaluate_one() (so Phase 2 / SUBSET subset-resize is automatic).
    std::vector<glm::vec3> work_positions;

    // SRT-applied base_normals. Only populated when the active objective
    // needs them (e.g. point-to-plane variants); plain RMSE leaves this
    // unused. Phase 1 keeps it unused.
    std::vector<glm::vec3> work_normals;

    // Diagnostic state from the last evaluate_one() call. matched_count
    // is also returned by reference for the penalty check.
    int   matched_count = 0;
    float last_rmse     = 0.0f;
};

// =====================================================================
// SRT helpers (Step 1-3).
// ---------------------------------------------------------------------
// Decode the CMA-ES normalized [-1, 1] vector into physical-unit
// SRTParams, and build the around-centroid 4x4 SRT matrix.
//
// Expression evaluation order, cast timing, and matrix composition
// order are kept BYTE-FOR-BYTE identical to V1 (see CmaesUtils.h
// :: run() inner loop and :: applyIncrementalSRT()). This is required
// for FULL_MESH mode to reproduce V1 results to the last bit, which
// is how Phase 1 validation confirms the architectural refactor is
// correct.
//
// Subsequent steps add: build_eval_context (1-4), evaluate_one (1-5),
// rebuild_correspondences (1-6), writeback_best_to_organs (1-7),
// runV2 (1-8).
// =====================================================================

// ----------------------------------------------------------------------
// Decode a CMA-ES population vector pop[k][0..6] into physical SRT.
// V1 reference (CmaesUtils.h::run() inner loop):
//     float tx = (float)(pop[k][0] * params.tx_range);
//     ...
//     float sc = params.scale_lo
//              + (float)((pop[k][6]+1.0)*0.5)
//                    * (params.scale_hi - params.scale_lo);
// ----------------------------------------------------------------------
inline SRTParams srt_from_population(const double* pop_k, const ParamsV2& p)
{
    SRTParams s;
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

// ----------------------------------------------------------------------
// Decode the CMA-ES best_x vector into physical SRT.
// Body is INTENTIONALLY identical to srt_from_population so the same
// numerics apply; the separate name documents the call site (best-so-
// far parameters, used for correspondence rebuild and final writeback).
// V1 reference (CmaesUtils.h::run() at update_interval boundary and
// final writeback):
//     float tx_b = (float)(best_x[0] * params.tx_range);
//     ...
// ----------------------------------------------------------------------
inline SRTParams srt_from_xvec(const double* x, const ParamsV2& p)
{
    SRTParams s;
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

// ----------------------------------------------------------------------
// Build the around-centroid 4x4 similarity transform (rigid + uniform
// scale). Vertices v are transformed as:
//     v' = M * v,   where  M = T * fromCentroid * Rz * Ry * Rx * S
//                                            * toCentroid
// i.e. translate to origin -> uniform scale -> Rx -> Ry -> Rz ->
//      translate back to centroid -> apply final translation.
//
// V1 reference (CmaesUtils.h::applyIncrementalSRT(), end of function):
//     glm::mat4 M = T * fromCentroid * Rz * Ry * Rx * S * toCentroid;
//
// The composition order, the per-axis rotation order (X then Y then Z),
// and the deg->rad constant ((float)(M_PI / 180.0)) are reproduced
// exactly. CmaesUtils.h is already pulled in above, so <cmath> /
// <glm/gtc/matrix_transform.hpp> are transitively available; M_PI
// resolves via _USE_MATH_DEFINES (set in CMakeLists.txt).
// ----------------------------------------------------------------------
inline glm::mat4 build_srt_matrix(const SRTParams& s,
                                  const glm::vec3& centroid)
{
    const float deg2rad = (float)(M_PI / 180.0);

    glm::mat4 T  = glm::translate(glm::mat4(1.0f),
                                 glm::vec3(s.tx, s.ty, s.tz));
    glm::mat4 Rx = glm::rotate   (glm::mat4(1.0f),
                               s.rx_deg * deg2rad, glm::vec3(1, 0, 0));
    glm::mat4 Ry = glm::rotate   (glm::mat4(1.0f),
                               s.ry_deg * deg2rad, glm::vec3(0, 1, 0));
    glm::mat4 Rz = glm::rotate   (glm::mat4(1.0f),
                               s.rz_deg * deg2rad, glm::vec3(0, 0, 1));
    glm::mat4 S  = glm::scale    (glm::mat4(1.0f),
                             glm::vec3(s.scale));

    glm::mat4 toCentroid   = glm::translate(glm::mat4(1.0f), -centroid);
    glm::mat4 fromCentroid = glm::translate(glm::mat4(1.0f),  centroid);

    return T * fromCentroid * Rz * Ry * Rx * S * toCentroid;
}

// =====================================================================
// build_eval_context (Step 1-4, FULL_MESH only).
// ---------------------------------------------------------------------
// One-shot initialiser for EvalContextStatic. Performs the work that V1
// scatters across CmaesUtils.h::run() before its CMA-ES loop:
//   - snapshot organs[0] (liver) into base_positions / base_normals,
//   - compute the around-centroid origin (same accumulation order as V1
//     applyIncrementalSRT) so FULL_MESH centroid matches V1 bit-for-bit,
//   - extract the target point cloud (V1 path: extractFrontFacePoints,
//     which transparently returns the OBJ-injected cache when set),
//   - build the squared correspondence-distance gate (V1 formula:
//     g_sceneDiag / 7.36),
//   - build tgt_to_eval via a KDTree over base_positions (iteration
//     order over tgt matches V1::updateCorrespondences exactly),
//   - set eval_to_full_idx to the identity map (FULL_MESH only),
//   - compute the penalty gate matched_min_required from init_matched.
//
// SUBSET_RMSE is added in Phase 2; if requested in Phase 1 we log and
// fall back to FULL_MESH so the surrounding plumbing can be developed
// without crashes.
// =====================================================================
inline EvalContextStatic build_eval_context(
    const std::vector<mCutMesh*>& organs,
    mCutMesh*                     screenMesh,
    int gridWidth, int gridHeight, float depthScale,
    EvalMode        resolved_mode,
    int             init_matched,
    const ParamsV2& params)
{
    EvalContextStatic S;
    S.mode = resolved_mode;

    // SUBSET_RMSE is deprecated since Phase 2.5: a correspondence-
    // based src subset, replaced by symmetric voxel downsample. We
    // accept it here as a synonym for AUTO so old call sites or
    // serialized Params still parse. AUTO must already be resolved
    // by the caller (resolve_eval_mode in runV2) before we get here;
    // any other value is a bug.
    if (S.mode == EvalMode::SUBSET_RMSE) {
        S.mode = EvalMode::AUTO;  // treat as AUTO
    }
    if (S.mode != EvalMode::FULL_MESH && S.mode != EvalMode::AUTO) {
        std::cerr << "[V2] build_eval_context: unexpected mode="
                  << (int)S.mode
                  << " (expected FULL_MESH or AUTO); "
                     "falling back to FULL_MESH." << std::endl;
        S.mode = EvalMode::FULL_MESH;
    }

    if (organs.empty() || !organs[0]) {
        std::cerr << "[V2] build_eval_context: organs[0] is null"
                  << std::endl;
        return S;  // empty context; caller must handle.
    }

    // ----- A. Pull full snapshot of organs[0] into vec3 arrays -------
    const auto&  src_verts = organs[0]->mVertices;
    const auto&  src_norms = organs[0]->mNormals;
    std::vector<glm::vec3> full_positions;
    std::vector<glm::vec3> full_normals;
    full_positions.reserve(src_verts.size() / 3);
    for (size_t i = 0; i + 2 < src_verts.size(); i += 3) {
        full_positions.emplace_back(src_verts[i],
                                    src_verts[i + 1],
                                    src_verts[i + 2]);
    }
    full_normals.reserve(src_norms.size() / 3);
    for (size_t i = 0; i + 2 < src_norms.size(); i += 3) {
        full_normals.emplace_back(src_norms[i],
                                  src_norms[i + 1],
                                  src_norms[i + 2]);
    }

    // ----- B. tgt_points via V1's extractFrontFacePoints --------------
    // Same constructor, same arguments, same zThresh floor as V1.
    Reg3DCustom::NoOpen3DRegistration reg_cache;
    float zThresh = std::max(0.001f, depthScale);
    auto  targetCloud = reg_cache.extractFrontFacePoints(
        *screenMesh, gridWidth, gridHeight, zThresh);
    if (targetCloud) {
        S.tgt_points = targetCloud->points;
    }

    // ----- C. max_dist_sq (scale-invariant gate, V1 constant) --------
    constexpr float kRefSceneDiag = 7.36f;
    const float max_dist = g_sceneDiag * (1.0f / kRefSceneDiag);
    S.max_dist_sq = max_dist * max_dist;

    // ----- D. Resolve effective voxel sizes ---------------------------
    // FULL_MESH forces voxel=0 (no downsample) for V1 bit-identical
    // reproduction. AUTO uses ratios from params, scaled by the active
    // sceneDiag so the same ratio gives a sensible absolute size
    // regardless of how Depth-Anything's metric output came out.
    float src_voxel = 0.0f, tgt_voxel = 0.0f;
    if (S.mode == EvalMode::AUTO) {
        if (params.src_voxel_ratio > 0.0f) {
            src_voxel = params.src_voxel_ratio * g_sceneDiag;
        }
        if (params.tgt_voxel_ratio > 0.0f) {
            tgt_voxel = params.tgt_voxel_ratio * g_sceneDiag;
        }
    }

    // ----- E. Optional src voxel downsample ---------------------------
    // voxelDownSample takes a shared<PointCloud>; we round-trip the
    // vec3 arrays through one. When voxel=0 the call returns the input
    // pointer unchanged (early-out in NoOpen3DRegistration.h).
    if (src_voxel > 0.0f && !full_positions.empty()) {
        auto src_pc = std::make_shared<Reg3DCustom::PointCloud>();
        src_pc->points  = full_positions;   // copy
        src_pc->normals = full_normals;     // copy (drives has_normals)
        const size_t before = src_pc->size();
        auto src_ds = reg_cache.voxelDownSample(src_pc, src_voxel);
        S.base_positions = std::move(src_ds->points);
        S.base_normals   = std::move(src_ds->normals);
        if (params.verbose) {
            std::cout << "[V2] src downsample: " << before << " -> "
                      << S.base_positions.size()
                      << " (voxel=" << src_voxel << " m)" << std::endl;
        }
    } else {
        S.base_positions = std::move(full_positions);
        S.base_normals   = std::move(full_normals);
    }

    // ----- F. Optional tgt voxel downsample ---------------------------
    if (tgt_voxel > 0.0f && !S.tgt_points.empty()) {
        auto tgt_pc = std::make_shared<Reg3DCustom::PointCloud>();
        tgt_pc->points = S.tgt_points;       // copy
        // boundaryDist / normals are preserved by voxelDownSample but
        // unused by Phase 2.5 evaluate_one (plain RMSE only).
        const size_t before = tgt_pc->size();
        auto tgt_ds = reg_cache.voxelDownSample(tgt_pc, tgt_voxel);
        S.tgt_points = std::move(tgt_ds->points);
        if (params.verbose) {
            std::cout << "[V2] tgt downsample: " << before << " -> "
                      << S.tgt_points.size()
                      << " (voxel=" << tgt_voxel << " m)" << std::endl;
        }
    }

    // ----- G. centroid of base_positions (around-centroid SRT origin)
    // Same naive accumulation order as V1::applyIncrementalSRT, so the
    // FULL_MESH code path stays bit-identical to V1. AUTO mode
    // intentionally uses the centroid of the downsampled cloud (which
    // is what evaluate_one actually applies SRT to), accepting a
    // small geometric shift vs the full-mesh centroid -- the same
    // budget that the Phase 2 acceptance criterion of +/- 5% RMSE
    // covers.
    glm::vec3 c(0.0f);
    int cnt = 0;
    for (size_t i = 0; i < S.base_positions.size(); i++) {
        c += S.base_positions[i];
        cnt++;
    }
    if (cnt > 0) c /= (float)cnt;
    S.centroid = c;

    // ----- H. tgt_to_eval via KDTree over base_positions --------------
    // Iteration order over tgt is identical to V1's
    // updateCorrespondences. Values are indices INTO base_positions
    // (which is post-voxel in AUTO, full in FULL_MESH).
    S.tgt_to_eval.assign(S.tgt_points.size(), -1);
    if (!S.base_positions.empty() && !S.tgt_points.empty()) {
        Reg3DCustom::NanoflannAdaptor adaptor(S.base_positions);
        auto tree = Reg3DCustom::buildKDTree(adaptor);
        for (size_t i = 0; i < S.tgt_points.size(); i++) {
            size_t nnIdx;
            float  dist_sq;
            if (Reg3DCustom::searchKNN1(*tree, S.tgt_points[i],
                                        nnIdx, dist_sq)
                && dist_sq < S.max_dist_sq)
            {
                S.tgt_to_eval[i] = (int)nnIdx;
            }
        }
    }

    // ----- I. eval_to_full_idx (identity; deprecated -- voxel cells
    //         have no 1:1 mapping to original vertices, but downstream
    //         code in Phase 2.5 doesn't read this field) -------------
    S.eval_to_full_idx.resize(S.base_positions.size());
    for (size_t k = 0; k < S.base_positions.size(); k++) {
        S.eval_to_full_idx[k] = (int)k;
    }

    // ----- J. matched_min_required (penalty gate) --------------------
    // FULL_MESH: keep V1's formula on init_matched (which was measured
    // against the full 774k tgt cloud) so the gate is bit-identical.
    // AUTO: init_matched no longer matches our (downsampled) tgt scale;
    // base the gate on the downsampled tgt size with the same ratio.
    int min_ok;
    if (S.mode == EvalMode::FULL_MESH) {
        min_ok = (int)(init_matched * params.min_match_ratio);
    } else {
        min_ok = (int)(S.tgt_points.size() * params.min_match_ratio);
    }
    if (min_ok < 10) min_ok = 10;
    S.matched_min_required = min_ok;

    return S;
}

// =====================================================================
// evaluate_one (Step 1-5).
// ---------------------------------------------------------------------
// PURE FUNCTION. Transforms S.base_positions by `srt` into
// W.work_positions, then computes RMSE against S.tgt_points using the
// cached correspondence map S.tgt_to_eval. Reads no globals; writes
// only into the per-thread scratch W. This purity is what enables
// OpenMP-parallel population evaluation in Phase 3 (each population
// member k uses its own W with no contention).
//
// Returns:
//   - 9.9f when count == 0 (no matched pairs), matching V1
//     fastComputeRMSE's early-return value;
//   - sqrt(sumSq / count) otherwise.
//
// matched_out: receives the number of matched pairs. The caller
// (runV2) uses this to apply V1's penalty rule
// (matched < S.matched_min_required -> fval = penalty_value).
//
// V1 reference -- CmaesUtils.h::run() inner k-loop does:
//   organs[0]->mVertices = snap_v[0];
//   organs[0]->mNormals  = snap_n[0];
//   applyIncrementalSRT(liver_only, tx,ty,tz, rx,ry,rz, sc);
//   float rmse    = fastComputeRMSE();
//   int   matched = registrationHandle.compCount;
// V2 condenses the restore + SRT + RMSE into this single pure
// function. The SRT matrix is built once via build_srt_matrix() (which
// matches V1::applyIncrementalSRT byte-for-byte), then applied to
// base_positions instead of mutating organs[0]->mVertices in place.
// The RMSE accumulation order, the squared-distance gate, and the
// count==0 fallback are all reproduced exactly.
//
// Phase 1 note (intentional V1 deviation, no impact on CompRMSE):
// normals are NOT transformed. V1 applyIncrementalSRT transforms them
// in place but V1 fastComputeRMSE never reads them, so the resulting
// rmse is the same. Skipping that work shaves a constant fraction off
// each eval. If a future phase wires up a normal-dependent objective
// (e.g. point-to-plane), populate W.work_normals here using
// glm::mat3(glm::transpose(glm::inverse(M))) -- matrix is identical
// to V1's normalMat.
// =====================================================================
inline float evaluate_one(const EvalContextStatic& S,
                          EvalContextScratch&      W,
                          const SRTParams&         srt,
                          int&                     matched_out)
{
    // ----- 1. SRT matrix (around-centroid; identical to V1) ----------
    const glm::mat4 M = build_srt_matrix(srt, S.centroid);

    // ----- 2. Apply M to base_positions -> W.work_positions ----------
    // Per-component (.x/.y/.z) assignment matches V1's
    // v[i]=tp.x; v[i+1]=tp.y; v[i+2]=tp.z
    // bit-for-bit; do NOT shorten to `glm::vec3(tp)` -- the implicit
    // truncation has the same numerical effect but the explicit form
    // mirrors V1 exactly and is easier to audit.
    const size_t N = S.base_positions.size();
    if (W.work_positions.size() != N) W.work_positions.resize(N);
    for (size_t i = 0; i < N; i++) {
        glm::vec4 p(S.base_positions[i], 1.0f);
        glm::vec4 tp = M * p;
        W.work_positions[i] = glm::vec3(tp.x, tp.y, tp.z);
    }

    // ----- 3. RMSE over matched correspondences ----------------------
    // V1 reference (fastComputeRMSE):
    //   for i in 0..tgt_size:
    //     j = corr_idx[i]; if (j < 0) continue;
    //     d  = srcPt[j] - tgt[i];
    //     sq = dot(d,d);
    //     if (sq < max_dist_sq) { sumSq += sq; count++; }
    //   return count==0 ? 9.9f : sqrt(sumSq / count);
    //
    // For FULL_MESH, S.tgt_to_eval[i] equals V1's corr_idx[i] (KDTree
    // built on the same point set with the same iteration order in
    // build_eval_context), so accumulation produces the same float
    // sequence and the same final RMSE.
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
    // sumSq / count: count is int, promotes to float per C++ usual
    // arithmetic conversions -- same effective expression as V1's
    // `std::sqrt(sumSq / count)`. The (float) cast is explicit here
    // for clarity; the result is bit-identical.
    W.last_rmse = std::sqrt(sumSq / (float)count);
    return W.last_rmse;
}

// =====================================================================
// rebuild_correspondences (Step 1-6, FULL_MESH only).
// ---------------------------------------------------------------------
// Refresh S.tgt_to_eval at UPDATE_INTERVAL boundaries. As CMA-ES
// converges the best_x drifts, so the snapshot-time KDTree no longer
// pairs each tgt with its true nearest src; periodically rebuilding
// the correspondence map under the current best pose pulls the local
// minimum out from under the optimiser.
//
// V1 reference (CmaesUtils.h::run() at gen % UPDATE_INTERVAL == 0):
//   for m in organs: organs[m]->mVertices = snap_v[m];      // restore
//   applyIncrementalSRT(organs, best.tx, ..., best.scale);  // apply best
//   updateCorrespondences();          // rebuild corr_idx via KDTree
//   for m in organs: organs[m]->mVertices = snap_v[m];      // restore again
//
// V2 leaves organs[*] alone. We synthesise the post-best-SRT liver
// vertex array into a stack-local buffer, build the KDTree on that,
// and write straight into S.tgt_to_eval. The integer mapping produced
// is bit-identical to V1's corr_idx[] because:
//   - the SRT matrix is built around S.centroid, which equals V1's
//     per-call centroid of snap_v[0] (same naive accumulation);
//   - the per-vertex transform uses the same per-component (.x/.y/.z)
//     write pattern as V1::applyIncrementalSRT;
//   - the KDTree input point sequence is therefore bit-identical to
//     liverMesh3D->mVertices after V1's in-place SRT;
//   - the tgt iteration order and the same searchKNN1 + max_dist_sq
//     gate are preserved.
//
// FULL_MESH note: S.base_positions and S.centroid stay unchanged here.
// In FULL_MESH the base set is always snap_v[0] and never shrinks, so
// only S.tgt_to_eval needs refreshing.
//
// AUTO note (Phase 2.5, plan D): S.base_positions and S.centroid are
// ALSO held fixed for the duration of a single runV2(). They are
// established once in build_eval_context() (snap_liver voxel-down-
// sampled), and rebuild_correspondences here only refreshes the
// KDTree-derived tgt_to_eval against the SRT(best)-applied base.
// This mirrors V1's FULL_MESH semantics ("base unchanged, only the
// correspondence map moves") and keeps the meaning of CMA-ES's
// best_x stable across UPDATE_INTERVAL boundaries -- a cleaner
// invariant than the earlier draft that re-voxelized the base on
// every refresh.
//
// Net effect on Phase 2.5 cost:
//   src voxel downsample is now called only in build (10x per
//   Shift+F, once per BIPOP run); rebuild_correspondences no
//   longer downsamples (was 14x per run = 140x per Shift+F).
// =====================================================================
inline void rebuild_correspondences(EvalContextStatic&        S,
                                    const std::vector<float>& snap_liver,
                                    const SRTParams&          cur_best,
                                    const ParamsV2&           params)
{
    if (S.mode == EvalMode::SUBSET_RMSE) S.mode = EvalMode::AUTO;
    if (S.mode != EvalMode::FULL_MESH && S.mode != EvalMode::AUTO) {
        std::cerr << "[V2] rebuild_correspondences: unexpected mode="
                  << (int)S.mode << "; skipping refresh." << std::endl;
        return;
    }

    // ----- 1. SRT matrix around the snap-time centroid ---------------
    // For both modes, S.centroid is what evaluate_one uses as its SRT
    // origin, so applying the same matrix here positions the base in
    // the same frame the optimiser most recently saw it -- the right
    // pose to redo the nearest-neighbour search against.
    const glm::mat4 M = build_srt_matrix(cur_best, S.centroid);

    // ----- 2. Choose the source of truth for "current liver pose" ----
    // FULL_MESH: snap_liver (full mesh, V1 bit-identical path).
    // AUTO     : S.base_positions (already voxel-downsampled in build),
    //            so we don't pay the voxel-downsample cost on every
    //            UPDATE_INTERVAL boundary.
    // (void)snap_liver in AUTO -- argument is kept for API symmetry
    // with FULL_MESH and to leave the door open if a future phase
    // wants to re-derive the base from a fresh voxelization.
    std::vector<glm::vec3> transformed;
    if (S.mode == EvalMode::FULL_MESH) {
        transformed.reserve(snap_liver.size() / 3);
        for (size_t i = 0; i + 2 < snap_liver.size(); i += 3) {
            glm::vec4 p(snap_liver[i],
                        snap_liver[i + 1],
                        snap_liver[i + 2],
                        1.0f);
            glm::vec4 tp = M * p;
            transformed.emplace_back(tp.x, tp.y, tp.z);
        }
    } else {
        // AUTO: apply M to the cached voxel-downsampled base. Indices
        // here align 1:1 with S.base_positions, so tgt_to_eval values
        // continue to mean "subset_idx into S.base_positions" -- the
        // same contract evaluate_one expects.
        transformed.resize(S.base_positions.size());
        for (size_t k = 0; k < S.base_positions.size(); k++) {
            glm::vec4 p(S.base_positions[k], 1.0f);
            glm::vec4 tp = M * p;
            transformed[k] = glm::vec3(tp.x, tp.y, tp.z);
        }
        (void)params;  // src_voxel_ratio not consumed in plan-D rebuild
    }

    // ----- 3. KDTree on the SRT-applied base, refresh tgt_to_eval ----
    // Iteration order over tgt is identical to V1's
    // updateCorrespondences. -1 for tgt points whose nearest neighbour
    // is farther than max_dist (gate matches V1 exactly).
    S.tgt_to_eval.assign(S.tgt_points.size(), -1);
    if (!transformed.empty() && !S.tgt_points.empty()) {
        Reg3DCustom::NanoflannAdaptor adaptor(transformed);
        auto tree = Reg3DCustom::buildKDTree(adaptor);
        for (size_t i = 0; i < S.tgt_points.size(); i++) {
            size_t nnIdx;
            float  dist_sq;
            if (Reg3DCustom::searchKNN1(*tree, S.tgt_points[i],
                                        nnIdx, dist_sq)
                && dist_sq < S.max_dist_sq)
            {
                S.tgt_to_eval[i] = (int)nnIdx;
            }
        }
    }

    // S.base_positions / S.centroid / S.eval_to_full_idx are all
    // unchanged (plan D: cached voxelization is reused for the entire
    // runV2 lifetime). The caller continues to evaluate fresh SRTs
    // against the same base, just with the refreshed correspondence
    // map.
}

// =====================================================================
// resolve_eval_mode (Step 2-3).
// ---------------------------------------------------------------------
// Map ParamsV2::eval_mode == AUTO to a concrete mode based on which
// objective is active. The rule (plan section 1, 3.1):
//   - silhouette / boundary objectives  -> FULL_MESH
//       (these evaluators want a full transformed mesh, not a
//        correspondence-derived subset; current evaluate_one does
//        plain RMSE only, but reserving FULL_MESH for these makes
//        the AUTO contract robust if those objectives are wired in
//        via this code path in future phases.)
//   - plain RMSE                         -> SUBSET_RMSE
//       (the speedup target: only the unique src vertices that
//        actually have a tgt correspondence enter the SRT/RMSE.)
//
// Non-AUTO modes are passed through unchanged.
// =====================================================================
inline EvalMode resolve_eval_mode(const ParamsV2& p)
{
    if (p.eval_mode != EvalMode::AUTO) {
        return p.eval_mode;
    }
    if (p.use_silhouette_2d || p.use_boundary_weight) {
        return EvalMode::FULL_MESH;
    }
    return EvalMode::SUBSET_RMSE;
}

// =====================================================================
// writeback_best_to_organs (Step 1-7).
// ---------------------------------------------------------------------
// Apply the final best SRT to ALL organs (liver, portal, vein, tumor,
// segment, gb -- not just the liver), so that the scene reflects the
// converged pose for every mesh the BIPOP outer loop will subsequently
// snapshot into best_v[m] / best_n[m].
//
// The CMA-ES inner loop only ever transformed organs[0] (liver) for
// speed; the other organs stayed at their snapshot pose throughout the
// run. This function is the single place where the converged pose is
// pushed onto every mesh.
//
// V1 reference (CmaesUtils.h::run() epilogue, after cmaes_exit):
//   for m: organs[m]->mVertices = snap_v[m];
//          organs[m]->mNormals  = snap_n[m];
//   applyIncrementalSRT(organs, tx_b, ty_b, tz_b, rx_b, ry_b, rz_b, sc_b);
//   for m: best_v[m] = organs[m]->mVertices;
//          best_n[m] = organs[m]->mNormals;
// V2 owns steps (1) and (2) here; the read-back into best_v / best_n
// is the caller's responsibility (runBipopCmaesV2 needs those for its
// outer-loop best tracking, see plan section 1.1).
//
// bit-identical notes:
//   - applyIncrementalSRT (V1's function, reused verbatim) computes
//     centroid from organs[0]->mVertices == snap_v[0] right after the
//     restore loop, matching V1's per-call recomputation exactly.
//   - SRTParams `best` is produced by srt_from_xvec(best_x, params),
//     whose expression order matches V1's per-axis decode line-by-line.
// =====================================================================
inline void writeback_best_to_organs(
    const std::vector<mCutMesh*>&            organs,
    const std::vector<std::vector<float>>&   snap_v,
    const std::vector<std::vector<float>>&   snap_n,
    const SRTParams&                         best)
{
    // ----- 1. Restore every organ to its snapshot pose ---------------
    for (size_t m = 0; m < organs.size(); m++) {
        if (organs[m]) {
            organs[m]->mVertices = snap_v[m];
            organs[m]->mNormals  = snap_n[m];
        }
    }

    // ----- 2. Apply best SRT to every organ in one shot --------------
    // Reuse V1's applyIncrementalSRT verbatim (defined in CmaesUtils.h,
    // same namespace, no qualifier needed). centroid is computed inside
    // from organs[0]->mVertices == snap_v[0], identical to V1.
    applyIncrementalSRT(organs,
                        best.tx,     best.ty,     best.tz,
                        best.rx_deg, best.ry_deg, best.rz_deg,
                        best.scale);
}

// =====================================================================
// runV2 (Step 1-8) -- main entry point, single-threaded.
// ---------------------------------------------------------------------
// V2 counterpart to CmaesUtils.h::CmaesRefine::run(). Same outward
// contract (same Params/Result types modulo ParamsV2 extensions, same
// caller in RegistrationActions::runBipopCmaesV2 -- to be added in
// step 1-9), same numerical effect on `organs` and on
// registrationHandle.compRmse.
//
// The major structural change vs V1 is that the inner population loop
// now calls evaluate_one() (a pure function over EvalContext + scratch)
// instead of mutating organs[0]->mVertices and reading
// registrationHandle.compCount inside the loop. That purity is what
// enables OpenMP-parallel populations in Phase 3.
//
// Phase 1 deliberately implements only the plain-RMSE objective. If
// use_silhouette_2d or use_boundary_weight is set, V2 logs a warning
// and falls back to plain RMSE. Use V1 (Shift+V) for silhouette /
// contour objectives.
//
// V1 epilogue is reproduced faithfully, including the two-gate
// accept logic:
//   (a) screening gate: best_rmse < initial_fval (= rmse_before*1.001
//       for plain RMSE), to skip the expensive computeUnifiedMetrics
//       when the inner-loop best didn't even break the noise floor;
//   (b) final gate: rmse_after_uniform < rmse_before, comparing the
//       unified-metric compRmse on the best-applied organs vs the
//       snapshot.
// Both gates use the same input numerics as V1, so the accept/reject
// decision is bit-identical and result.improved + result.rmse_after
// match V1 byte-for-byte.
// =====================================================================
inline Result runV2(
    const std::vector<mCutMesh*>& organs,
    mCutMesh*                     screenMesh,
    int gridWidth, int gridHeight, float depthScale,
    const ParamsV2&               params)
{
    Result result;

    // ----- Phase 1 objective gate -----------------------------------
    // V1 supports silhouette / boundary objectives via overrides on
    // fval[k] inside the inner loop. V2 inner loop returns plain RMSE
    // only (the silhouette evaluators read view/projection globals and
    // do GL work that's not thread-safe). Warn and continue.
    if (params.use_silhouette_2d || params.use_boundary_weight) {
        std::cerr << "[V2] Phase 1 supports plain RMSE objective only; "
                     "use_silhouette_2d / use_boundary_weight ignored. "
                     "Use V1 (Shift+V) for those objectives." << std::endl;
    }

    // ----- 1. Snapshot all organs ------------------------------------
    std::vector<std::vector<float>> snap_v(organs.size());
    std::vector<std::vector<float>> snap_n(organs.size());
    for (size_t m = 0; m < organs.size(); m++) {
        if (organs[m]) {
            snap_v[m] = organs[m]->mVertices;
            snap_n[m] = organs[m]->mNormals;
        }
    }

    // ----- 2. rmse_before via computeUnifiedMetrics ------------------
    computeUnifiedMetrics();
    result.rmse_before = registrationHandle.compRmse;
    const int init_matched = registrationHandle.compCount;

    // ----- 3. Resolve EvalMode (Phase 2: AUTO -> SUBSET_RMSE for plain
    //         RMSE; FULL_MESH still selectable explicitly for V1
    //         bit-identical reproduction). ------------------------------
    EvalMode resolved = resolve_eval_mode(params);

    // ----- 4. Build EvalContextStatic --------------------------------
    EvalContextStatic S = build_eval_context(
        organs, screenMesh, gridWidth, gridHeight, depthScale,
        resolved, init_matched, params);

    if (S.base_positions.empty() || S.tgt_points.empty()) {
        std::cerr << "[V2] empty base or tgt cloud; aborting CMA-ES"
                  << std::endl;
        result.improved   = false;
        result.rmse_after = result.rmse_before;
        return result;
    }

    if (params.verbose) {
        std::cout << "[V2] === Starting BIPOP-CMA-ES V2 ===" << std::endl
                  << "[V2] eval mode    : "
                  << (S.mode == EvalMode::FULL_MESH ? "FULL_MESH"
                                                    : "SUBSET_RMSE")
                  << "  (" << S.base_positions.size() << " src verts)"
                  << std::endl
                  << "[V2] tgt points   : " << S.tgt_points.size()
                  << std::endl
                  << "[V2] init_matched : " << init_matched
                  << "  min_required: " << S.matched_min_required
                  << std::endl
                  << "[V2] update_int   : " << params.update_interval
                  << "  parallel pop : "
                  << (params.parallel_population ? "YES" : "NO")
                  << std::endl;
    }

    // ----- 5. cmaes_init + deterministic srand override --------------
    const int DIM = 7;
    double lb[DIM], ub[DIM], xstart[DIM];
    for (int d = 0; d < DIM; d++) {
        lb[d] = -1.0; ub[d] = 1.0; xstart[d] = 0.0;
    }
    cmaes_t* evo = cmaes_init(DIM, xstart, params.sigma0,
                              params.lambda, lb, ub);
    if (params.rng_seed != 0) {
        srand(params.rng_seed);
        if (params.verbose) {
            std::cout << "[V2] Deterministic seed: "
                      << params.rng_seed << std::endl;
        }
    }

    // ----- 6. CMA-ES state -------------------------------------------
    double best_x[DIM] = {0,0,0,0,0,0,0};
    float  best_rmse   = result.rmse_before;
    g_quietMetrics = true;

    // ----- 7. Per-thread scratch (Phase 1 single-thread: size 1) -----
    std::vector<EvalContextScratch> scratch_pool(1);

    // ----- 8. CMA-ES main loop ---------------------------------------
    const char* stop = nullptr;
    double t_eval = 0.0, t_rebuild = 0.0;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    using ms = std::chrono::duration<double, std::milli>;

    // Wall-clock for the whole CMA-ES loop, used to derive "other"
    // time as (loop_total - eval - rebuild) so we don't double-count
    // by accident with overlapping start/end markers.
    const auto t_loop_start = now();

    for (int gen = 0; gen < params.maxgen && !stop; gen++) {
        auto tg0 = now();

        double**            pop = cmaes_SamplePopulation(evo);
        std::vector<double> fval(evo->lambda);

        for (int k = 0; k < evo->lambda; k++) {
            SRTParams srt = srt_from_population(pop[k], params);
            int   matched = 0;
            float rmse    = evaluate_one(S, scratch_pool[0], srt, matched);

            // V1 penalty rule: bad iff matched < min_required OR rmse==0.
            const bool bad = (matched < S.matched_min_required)
                             || (rmse == 0.0f);
            fval[k] = bad ? (double)params.penalty_value : (double)rmse;

            // Order-preserving best tracking (matches V1; will stay
            // single-threaded even after Phase 3 parallel population).
            if (fval[k] < best_rmse) {
                best_rmse = (float)fval[k];
                for (int d = 0; d < DIM; d++) best_x[d] = pop[k][d];
            }
        }
        auto tg1 = now();
        t_eval += ms(tg1 - tg0).count();

        cmaes_UpdateDistribution(evo, fval.data());

        // UPDATE_INTERVAL: refresh tgt->src correspondences against
        // the current best pose. V1 does this every UPDATE_INTERVAL=10
        // gens; V2 exposes the interval as params.update_interval.
        if (gen > 0 && gen % params.update_interval == 0) {
            auto tr0 = now();
            const SRTParams cur_best = srt_from_xvec(best_x, params);
            rebuild_correspondences(S, snap_v[0], cur_best, params);
            auto tr1 = now();
            t_rebuild += ms(tr1 - tr0).count();
        }

        if (params.verbose && (gen % params.log_every == 0)) {
            std::cout << "[V2] Gen " << std::setw(4) << gen
                      << "  best=" << std::fixed << std::setprecision(5)
                      << best_rmse
                      << "  sigma=" << std::setprecision(4) << evo->sigma
                      << std::endl;
        }
        stop = cmaes_TestForTermination(evo, params.maxgen,
                                        params.tolfun, params.tolx);
    }
    const double t_loop_total = ms(now() - t_loop_start).count();

    // Capture lambda before exit so we can compute total_evals.
    const int evo_lambda = evo->lambda;
    result.generations = evo->gen;
    result.stop_reason = stop ? stop : "MaxGen";
    cmaes_exit(evo);

    // ----- 9. Apply best to all organs; capture best_v/best_n --------
    // Same effect as V1's "build best_v from best_x" block: snap
    // restore + applyIncrementalSRT, then read mVertices/mNormals.
    const SRTParams best_srt = srt_from_xvec(best_x, params);
    writeback_best_to_organs(organs, snap_v, snap_n, best_srt);
    std::vector<std::vector<float>> best_v(organs.size());
    std::vector<std::vector<float>> best_n(organs.size());
    for (size_t m = 0; m < organs.size(); m++) {
        if (organs[m]) {
            best_v[m] = organs[m]->mVertices;
            best_n[m] = organs[m]->mNormals;
        }
    }

    // ----- 10. Time-breakdown log ------------------------------------
    if (params.verbose) {
        const int total_evals = result.generations * evo_lambda;
        // t_total = wall-clock of the CMA-ES loop. t_other is what's
        // left after subtracting eval and rebuild -- this represents
        // cmaes_SamplePopulation + cmaes_UpdateDistribution + log +
        // termination test. Computing it as a residual avoids the
        // double-counting bug where rebuild time was also included
        // in a naive end-of-iteration "other" timer.
        const double t_total = t_loop_total;
        const double t_other = t_total - t_eval - t_rebuild;
        // Count of rebuild calls (gen=update_interval, 2*ui, 3*ui, ...).
        const int rebuild_calls =
            (result.generations - 1) / params.update_interval;

        std::cout << std::fixed << std::setprecision(1)
                  << "[V2] === Time Breakdown (total " << (int)t_total
                  << " ms, " << total_evals << " evals) ===" << std::endl;
        if (t_total > 0.0) {
            std::cout << "[V2]   evaluate_one (sum) : " << (int)t_eval
                      << " ms (" << (int)(100*t_eval/t_total) << "%)"
                      << std::endl
                      << "[V2]   rebuild_corr (sum) : " << (int)t_rebuild
                      << " ms (" << (int)(100*t_rebuild/t_total) << "%)"
                      << "  [" << rebuild_calls << " calls"
                      << (rebuild_calls > 0
                              ? ", " + std::to_string((int)(t_rebuild/rebuild_calls)) + " ms/call"
                              : "")
                      << "]" << std::endl
                      << "[V2]   cmaes/log/other    : " << (int)t_other
                      << " ms (" << (int)(100*t_other/t_total) << "%)"
                      << std::endl;
        }
        if (total_evals > 0) {
            std::cout << "[V2]   per eval avg       : "
                      << std::setprecision(3)
                      << (t_total / total_evals) << " ms" << std::endl;
        }
    }

    // ----- 11. Restore organs to snap; measure for screening --------
    // V1 restores to snap here so initial_fval / rmse_before are
    // measured on the snapshot pose -- not the best pose. V2 mirrors
    // this exactly so both screening gates see the same input numbers.
    for (size_t m = 0; m < organs.size(); m++) {
        if (organs[m]) {
            organs[m]->mVertices = snap_v[m];
            organs[m]->mNormals  = snap_n[m];
        }
    }
    computeUnifiedMetrics();
    g_quietMetrics = false;

    // ----- 12. initial_fval (Phase 1 plain-RMSE branch only) --------
    // V1: for plain RMSE, initial_fval = result.rmse_before * 1.001f.
    // The 0.1% bias prevents accepting "improvements" that are just
    // noise. silhouette/contour have different formulas (out of scope).
    const float initial_fval = result.rmse_before * 1.001f;

    // ----- 13. Screening gate ---------------------------------------
    if (best_rmse < initial_fval) {
        // Re-apply best by copying back the cached best_v/best_n
        // (faster than recomputing applyIncrementalSRT; matches V1).
        for (size_t m = 0; m < organs.size(); m++) {
            if (organs[m]) {
                organs[m]->mVertices = best_v[m];
                organs[m]->mNormals  = best_n[m];
            }
        }
        computeUnifiedMetrics();
        const float rmse_after_uniform = registrationHandle.compRmse;

        // ----- 14. Final accept gate (Phase 1 plain RMSE) -----------
        const bool accepted = (rmse_after_uniform < result.rmse_before);

        if (accepted) {
            result.improved   = true;
            result.rmse_after = rmse_after_uniform;
            // Same float values as V1 (srt_from_xvec uses V1 expression
            // order); record into Result for caller diagnostics.
            result.delta_tx      = best_srt.tx;
            result.delta_ty      = best_srt.ty;
            result.delta_tz      = best_srt.tz;
            result.delta_rx_deg  = best_srt.rx_deg;
            result.delta_ry_deg  = best_srt.ry_deg;
            result.delta_rz_deg  = best_srt.rz_deg;
            result.scale_applied = best_srt.scale;

            if (params.verbose) {
                const float pct = 100.0f
                                  * (result.rmse_before - result.rmse_after)
                                  / result.rmse_before;
                std::cout << "[V2] *** IMPROVED ***" << std::endl
                          << "[V2] compRMSE: " << result.rmse_before
                          << " -> "            << result.rmse_after
                          << "  (" << std::fixed << std::setprecision(1)
                          << pct << "%)"       << std::endl
                          << "[V2] dT=("
                          << result.delta_tx     << ", "
                          << result.delta_ty     << ", "
                          << result.delta_tz     << ")" << std::endl
                          << "[V2] dR=("
                          << result.delta_rx_deg << ", "
                          << result.delta_ry_deg << ", "
                          << result.delta_rz_deg << ") deg" << std::endl
                          << "[V2] scale=" << result.scale_applied
                          << std::endl
                          << "[V2] Stop: "  << result.stop_reason
                          << "  gens=" << result.generations
                          << std::endl;
            }
        } else {
            // Final gate failed: revert to snap.
            result.improved   = false;
            result.rmse_after = result.rmse_before;
            for (size_t m = 0; m < organs.size(); m++) {
                if (organs[m]) {
                    organs[m]->mVertices = snap_v[m];
                    organs[m]->mNormals  = snap_n[m];
                }
            }
            computeUnifiedMetrics();
            if (params.verbose) {
                std::cout << "[V2] No improvement (compRMSE "
                          << result.rmse_before << " -> "
                          << rmse_after_uniform << "). Reverted."
                          << std::endl;
            }
        }
    } else {
        // Screening gate failed: best_rmse not better than the noise
        // floor. Skip the expensive computeUnifiedMetrics and revert.
        result.improved   = false;
        result.rmse_after = result.rmse_before;
        for (size_t m = 0; m < organs.size(); m++) {
            if (organs[m]) {
                organs[m]->mVertices = snap_v[m];
                organs[m]->mNormals  = snap_n[m];
            }
        }
        computeUnifiedMetrics();
        if (params.verbose) {
            std::cout << "[V2] No improvement ("
                      << result.rmse_before
                      << " -> best_tried=" << best_rmse
                      << "). Reverted." << std::endl
                      << "[V2] Stop: " << result.stop_reason
                      << std::endl;
        }
    }

    return result;
}

} // namespace CmaesRefine

#endif // CMAES_REFINE_V2_H
