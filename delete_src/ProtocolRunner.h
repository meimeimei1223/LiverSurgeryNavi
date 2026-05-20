#pragma once
/*
 * ProtocolRunner.h
 * ----------------
 * Shift+P automated protocol state machine for MICCAI-WS experiments.
 *
 * Responsible for executing a deterministic sequence of 10 trials
 * (5 × Condition B pose perturbations, 5 × Condition C seed variations)
 * on the currently loaded image, logging every HemiAuto / CMA-ES / Refine
 * call to a dedicated ProtocolLog (NOT the interactive pose library).
 *
 * Design contract (must be preserved across Step 4b-2 / 4b-3 / 4c):
 *
 *   1. The existing interactive UI path (onHemiAuto, onBipopCmaes, onRefine
 *      lambdas in main.cpp) is NEVER touched by this runner.
 *   2. While the protocol is active (active == true), the main render
 *      loop should invoke advance() exactly once per frame. One call to
 *      advance() performs at most one atomic action (H call, C call,
 *      R run, or trial setup/teardown) and returns. This ensures the
 *      screen updates naturally between actions.
 *   3. g_suppressPoseLibSave is held true for the entire protocol run,
 *      so that any internally-called poseSaveToLibrary() is a no-op.
 *   4. Seeds are derived from the master trial_seed:
 *        FGR tuple test  : trial_seed
 *        BIPOP outer rng : trial_seed + 1000
 *        CMA-ES internal : trial_seed + 2000 (+ call_idx offset per call)
 *
 * Staged implementation:
 *   Step 4b-1 (this file, current state): skeleton only — struct
 *     definitions, trial table, stubbed begin/advance/abort. Calling
 *     advance() is a no-op. This lets the build verify everything
 *     compiles before logic is wired in.
 *   Step 4b-2 : runHemiAutoOnce / runCmaesOnce synchronous helpers.
 *   Step 4b-3 : advance() state machine body populated.
 *   Step 4c   : main.cpp integration (#include, g_protocolState,
 *               render-loop hook, Shift+P handler, ImGui overlay).
 */

#ifndef PROTOCOL_RUNNER_H
#define PROTOCOL_RUNNER_H

#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iostream>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mCutMesh.h"
#include "ProtocolLog.h"
#include "RefineSyncTypes.h"
#include "NoOpen3DRegistration.h"
#include "CmaesUtils.h"
#include "RegistrationCore.h"       /* getPresetRotation, applyMatrixToMeshVerticesAndNormals */
#include "UmeyamaUtils.h"           /* resetRegistrationState */

/* ============================================================
 * Forward declarations of main.cpp helpers used by this runner.
 * These functions are defined in main.cpp with external linkage
 * (the 'static' qualifier was removed in Step 4b-1).
 * ============================================================ */
std::vector<mCutMesh*> getOrganList();
void                   computeUnifiedMetrics();
glm::mat4              buildPerturbation(float rxDeg, float ryDeg, float rzDeg,
                                         float tx, float ty, float tz,
                                         const glm::vec3& centroid);
RefineSyncResult       runRefineSync(int max_iter);


/* ============================================================
 * Extern references to main.cpp globals used by this runner.
 * (No new globals are defined here — this header is a pure
 *  consumer of main.cpp state.)
 * ============================================================ */
extern std::vector<std::vector<GLfloat>> g_initOrganVertices;
extern std::vector<std::vector<GLfloat>> g_initOrganNormals;
extern std::vector<size_t>               g_refineVertexIndices;
extern bool                              g_suppressPoseLibSave;
extern ProtocolLog                       g_protocolLog;

/* Additional main.cpp globals needed by runHemiAutoOnce / runCmaesOnce
 * (Step 4b-2). These are defined as non-static globals in main.cpp
 * and we simply reference them here. */
struct RegistrationData;
class  FullSphereCamera;
extern mCutMesh*          liverMesh3D;
extern mCutMesh*          screenMesh;
extern FullSphereCamera   OrbitCam;
extern int                gWindowWidth;
extern int                gGridWidth;
extern float              gDepthScale;
extern float              g_voxelSize;
extern RegistrationData   registrationHandle;
extern std::vector<glm::vec3> g_cluster1Points;
extern std::vector<glm::vec3> g_cluster2Points;

/* main.cpp helpers that are inline or non-static and available: */
int         gGridHeight();          /* inline in main.cpp around line 486 */
void        setUp(mCutMesh& m);     /* declared by MeshDataTypes / main  */


/* ============================================================
 * HemiAutoSyncResult — return type of runHemiAutoOnce()
 * ============================================================ */
struct HemiAutoSyncResult {
    bool  valid         = false;  /* false if visible extract returned <50 pts */
    float comp_rmse     = 0.0f;
    float comp_avg_error= 0.0f;
    float comp_max_error= 0.0f;
    int   comp_count    = 0;
    float base_fitness  = 0.0f;
    float base_icp_rmse = 0.0f;
    float base_scale    = 1.0f;
    float elapsed_sec   = 0.0f;
};

/* ============================================================
 * CmaesSyncResult — return type of runCmaesOnce()
 *   One call = one CmaesRefine::run() invocation (single restart).
 *   The outer BIPOP multi-start loop is owned by the caller
 *   (ProtocolState::advance()), so this function is the smallest
 *   unit of CMA-ES work.
 *   Note: this differs from the UI's onBipopCmaes which bundles
 *   10 internal restarts into one button press. The protocol
 *   instead does 10 outer calls to runCmaesOnce() per trial, each
 *   with a different sigma0 / perturbation from the outer rng.
 * ============================================================ */
struct CmaesSyncResult {
    bool        improved     = false;
    float       comp_rmse    = 0.0f;  /* after this single CMA-ES call   */
    float       comp_avg_error = 0.0f;
    float       comp_max_error = 0.0f;
    int         comp_count   = 0;
    float       elapsed_sec  = 0.0f;
    std::string regime_label;         /* "Regime1(global)" / "Regime2(local)" */
    double      sigma0_used  = 0.0;   /* for logging                      */
};


/* ============================================================
 * runHemiAutoOnce — synchronous HemiAuto executor.
 * ============================================================
 * Replicates the pure computational body of a.onHemiAuto from
 * main.cpp, WITHOUT the UI side-effects:
 *   - no poseAutoSaveBeforeRegistration()
 *   - no gUIManager.state.regMethod update
 *   - no poseSaveToLibrary() (the protocol logs elsewhere)
 *   - no computeIdealVoxelSizes() (UI telemetry only,
 *     does not influence the registration output)
 *
 * Caller must have already placed the mesh at the desired
 * starting pose (e.g. TOP + perturbation). This function:
 *   1. Extracts visible vertices from current camera view
 *   2. Updates g_refineVertexIndices (for a later Refine)
 *   3. Runs performRegistrationSingleMesh (FGR + ICP)
 *   4. Calls computeUnifiedMetrics()
 *   5. Packages metrics into HemiAutoSyncResult
 *
 * Determinism: if the caller sets Reg3DCustom::setFgrSeed(seed)
 * before invoking this function, the FGR tuple test RNG is
 * seeded and the run is bit-reproducible.
 * ============================================================ */
inline HemiAutoSyncResult runHemiAutoOnce()
{
    HemiAutoSyncResult out;
    auto t0 = std::chrono::steady_clock::now();

    /* Same preparation as the UI path, minus UI-only calls */
    resetRegistrationState();

    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);

    if (vis.cloud->size() < 50) {
        std::cerr << "[Protocol/HemiAuto] too few visible pts ("
                  << vis.cloud->size() << "); skipping." << std::endl;
        out.elapsed_sec = std::chrono::duration<float>(
                              std::chrono::steady_clock::now() - t0).count();
        return out;
    }

    g_cluster1Points = vis.points;
    g_cluster2Points.clear();
    g_refineVertexIndices = vis.vertexIndices;

    auto organs = getOrganList();
    Reg3DCustom::performRegistrationSingleMesh(
        organs, liverMesh3D, vis.vertexIndices,
        screenMesh, OrbitCam.cameraPos,
        gGridWidth, gGridHeight(),
        15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);

    computeUnifiedMetrics();

    out.valid          = true;
    out.comp_rmse      = registrationHandle.compRmse;
    out.comp_avg_error = registrationHandle.compAvgError;
    out.comp_max_error = registrationHandle.compMaxError;
    out.comp_count     = registrationHandle.compCount;
    out.base_fitness   = registrationHandle.fitness;       /* NOTE: RegistrationData member is named
                                                            * 'fitness', not 'baseFitness' — that naming
                                                            * is only used inside PoseEntry.           */
    out.base_icp_rmse  = registrationHandle.icpRmse;       /* same for icpRmse / base_icp_rmse.       */
    out.base_scale     = registrationHandle.scaleFactor;   /* and scaleFactor / base_scale.           */
    out.elapsed_sec    = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t0).count();
    return out;
}


/* ============================================================
 * runCmaesOnce — synchronous CMA-ES single-restart executor.
 * ============================================================
 * Replicates the per-run body of a.onBipopCmaes's for-loop (but
 * only ONE restart per call; the 10-restart outer loop is driven
 * by ProtocolState::advance()).
 *
 * Arguments:
 *   run_idx      : 0-based call index within this trial's Phase C.
 *                  Used to decide Regime (even = local, odd = global)
 *                  matching the UI's parity-based scheme.
 *   outer_rng    : caller-owned mt19937 seeded with (trial_seed+1000).
 *                  Drawn from to generate sigma0 and perturbation
 *                  magnitudes (identical distributions to the UI).
 *   cma_seed     : fed to CmaesRefine::Params::rng_seed. Typically
 *                  (trial_seed + 2000 + run_idx).
 *
 * The mesh is expected to be at "start of this restart" pose on
 * entry. On entry if run_idx > 0, the function applies the
 * perturbation relative to current pose (matching UI behavior
 * where restarts are chained; the caller is responsible for
 * resetting to start_v/start_n between calls if independence
 * is desired).
 *
 * Note: for the protocol we chain the calls (no reset between)
 * to mirror UI exploration. Trial-level reset happens in advance().
 * ============================================================ */
inline CmaesSyncResult runCmaesOnce(int                   run_idx,
                                    std::mt19937&         outer_rng,
                                    unsigned              cma_seed)
{
    CmaesSyncResult out;
    auto t0 = std::chrono::steady_clock::now();

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    /* Mirror UI's parameter derivation exactly. */
    CmaesRefine::Params p;
    p.verbose        = true;
    p.log_every      = 100;
    p.save_debug_jpg = false;
    p.rng_seed       = cma_seed;     /* Step 2 determinism hook */

    float tx_perturb = 0, ty_perturb = 0, tz_perturb = 0;
    float rx_perturb = 0, ry_perturb = 0, rz_perturb = 0;
    float sc_perturb = 1.0f;

    if (run_idx % 2 == 0) {
        /* Regime2 (local): smaller σ, smaller perturbations */
        p.sigma0   = 0.3 + dist01(outer_rng) * 0.4;
        tx_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 0.5f;
        ty_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 0.5f;
        tz_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 0.5f;
        rx_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 10.0f;
        ry_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 10.0f;
        rz_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 10.0f;
        sc_perturb = 0.95f + dist01(outer_rng) * 0.10f;
        out.regime_label = "Regime2(local)";
    } else {
        /* Regime1 (global): larger σ, larger perturbations */
        p.sigma0   = 0.5 + dist01(outer_rng) * 0.5;
        tx_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 1.5f;
        ty_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 1.5f;
        tz_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 1.5f;
        rx_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 30.0f;
        ry_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 30.0f;
        rz_perturb = (dist01(outer_rng) * 2.0f - 1.0f) * 30.0f;
        sc_perturb = 0.90f + dist01(outer_rng) * 0.20f;
        out.regime_label = "Regime1(global)";
    }
    out.sigma0_used = p.sigma0;

    auto organs = getOrganList();

    /* Apply perturbation on restarts > 0 (same behavior as UI).
     * Unlike the UI which resets vertices to start_v before each
     * run, the protocol chains perturbations from the current pose
     * so that each restart explores from a slightly different
     * start. The trial-level best tracking in advance() selects
     * the best result across all 10 restarts. */
    if (run_idx > 0) {
        CmaesRefine::applyIncrementalSRT(organs,
                                         tx_perturb, ty_perturb, tz_perturb,
                                         rx_perturb, ry_perturb, rz_perturb,
                                         sc_perturb);
        for (size_t i = 0; i < organs.size(); i++)
            if (organs[i]) setUp(*organs[i]);
    }

    std::cout << "[Protocol/CMA-ES] Run " << (run_idx + 1) << "/10  "
              << out.regime_label
              << "  sigma0=" << std::fixed << std::setprecision(2)
              << p.sigma0 << std::endl;

    CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                             gGridWidth, gGridHeight(),
                                             gDepthScale, p);
    computeUnifiedMetrics();

    out.improved       = r.improved;
    out.comp_rmse      = registrationHandle.compRmse;
    out.comp_avg_error = registrationHandle.compAvgError;
    out.comp_max_error = registrationHandle.compMaxError;
    out.comp_count     = registrationHandle.compCount;
    out.elapsed_sec    = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t0).count();

    std::cout << "[Protocol/CMA-ES] Run " << (run_idx + 1)
              << " compRMSE=" << std::setprecision(6) << out.comp_rmse
              << (r.improved ? " [IMPROVED]" : " [NO CHANGE]")
              << "  time=" << out.elapsed_sec << "s" << std::endl;
    return out;
}


/* ============================================================
 * TrialSpec — one row in the protocol's 10-trial table.
 * ============================================================
 * Condition B (pose perturbation, seed fixed @ 42):
 *   B0: TOP baseline
 *   B1: Rx +3°, tx +0.1
 *   B2: Rx -3°, tx -0.1
 *   B3: Ry +3°, ty +0.1
 *   B4: Rz +3°, tz +0.1
 *
 * Condition C (pose fixed @ TOP, seed variation):
 *   C0: seed 42    (duplicate of B0 — integrity check)
 *   C1: seed 123
 *   C2: seed 456
 *   C3: seed 789
 *   C4: seed 2024
 */
struct TrialSpec {
    char        condition;         /* 'B' or 'C'                              */
    int         trial_idx;         /* 0..4 within the condition               */
    float       rx_deg, ry_deg, rz_deg;  /* rotation perturbation (B only)    */
    float       tx, ty, tz;              /* translation perturbation (B only) */
    unsigned    trial_seed;        /* master seed for this trial              */
    std::string label;             /* human-readable perturbation descriptor  */
};


/* ============================================================
 * Progress snapshot — read by ImGui overlay (Step 4c).
 * Populated fresh at the start of each advance() invocation.
 * ============================================================ */
struct ProtocolProgress {
    bool        active           = false;
    char        cond             = '?';
    int         trial_idx        = 0;
    int         trial_total      = 10;
    char        phase            = '?';   /* 'H' / 'C' / 'R' / 'P' (pre/post) */
    int         call_idx         = 0;
    int         call_total       = 0;
    std::string perturbLabel;
    float       currentRmse      = 0.0f;
    float       bestInThisTrial  = 0.0f;
    int         entriesLogged    = 0;
};


/* ============================================================
 * ProtocolState — the full state machine.
 * ============================================================ */
struct ProtocolState {
    /* --- lifecycle --- */
    bool active = false;

    /* --- trial schedule --- */
    std::vector<TrialSpec> trials;     /* 10 entries: B0..B4, C0..C4         */

    /* --- cursor --- */
    int trial_i = 0;                   /* current index in trials[] (0..9)   */

    enum Phase {
        PRE_TRIAL = 0,                 /* setup: restore mesh, seed, perturb */
        PHASE_H,                       /* HemiAuto calls (call_total = 2)    */
        PHASE_C,                       /* CMA-ES calls   (call_total = 10)   */
        PHASE_R,                       /* Refine attempt (call_total = 1)    */
        POST_TRIAL,                    /* teardown, advance to next trial    */
        FINISHED                       /* all 10 trials done — save & stop   */
    };
    Phase phase = PRE_TRIAL;
    int   call_i = 0;                  /* 0-based index within current phase */

    /* --- per-trial transient state --- */
    /* HemiAuto best across the 2 H calls (vertex snapshot, matches
     * existing g_bestSessionVertices pattern). */
    std::vector<std::vector<GLfloat>> best_H_verts;
    std::vector<std::vector<GLfloat>> best_H_normals;
    float                             best_H_rmse     = 0.0f;
    int                               best_H_call_idx = -1;

    /* CMA-ES best across the 10 C calls. */
    std::vector<std::vector<GLfloat>> best_C_verts;
    std::vector<std::vector<GLfloat>> best_C_normals;
    float                             best_C_rmse     = 0.0f;
    int                               best_C_call_idx = -1;

    /* rmse captured right before Phase R runs (for refine_delta). */
    float before_R_rmse = 0.0f;

    /* Outer rng used by Phase C's internal BIPOP multi-start loop.
     * Re-seeded with (trial_seed + 1000) at the start of each trial. */
    std::mt19937 bipop_outer_rng;

    /* --- metadata --- */
    std::string imageName;
    std::string outDir;

    /* --- progress for UI --- */
    ProtocolProgress progress;

    /* ============================================================
     * Public API
     * ============================================================ */

    /* Begin a new protocol run. Clears state, builds the trial table,
     * and prepares the ProtocolLog. Must be called from the UI thread
     * (Shift+P handler). After begin(), the main render loop should
     * call advance() each frame until active becomes false. */
    void begin(const std::string& imageName_,
               const std::string& outDir_ = "./");

    /* Stop the protocol immediately. Mesh state is left as-is (no
     * auto-rollback). Logs collected so far are saved to disk. */
    void abort();

    /* One-action-per-frame state machine step. Safe to call every
     * frame — returns immediately if !active. */
    void advance();

    bool isActive() const { return active; }

    /* Helper: rebuild the 10-entry trial table. Called by begin(). */
    void buildTrialTable();
};


/* ============================================================
 * STUB implementations (Step 4b-1)
 * These allow the project to link as soon as Step 4c includes
 * this header. Real logic is added in Step 4b-2 / 4b-3.
 * ============================================================ */

inline void ProtocolState::buildTrialTable() {
    trials.clear();
    trials.reserve(10);

    const unsigned BSEED = 42u;

    /* ---- Condition B: pose perturbation, seed fixed at 42 ---- */
    trials.push_back({ 'B', 0,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   BSEED, "TOP"           });
    trials.push_back({ 'B', 1,  +3.0f, 0.0f, 0.0f,  +0.1f, 0.0f, 0.0f,   BSEED, "Rx+3_tx+0.1"   });
    trials.push_back({ 'B', 2,  -3.0f, 0.0f, 0.0f,  -0.1f, 0.0f, 0.0f,   BSEED, "Rx-3_tx-0.1"   });
    trials.push_back({ 'B', 3,   0.0f,+3.0f, 0.0f,   0.0f,+0.1f, 0.0f,   BSEED, "Ry+3_ty+0.1"   });
    trials.push_back({ 'B', 4,   0.0f, 0.0f,+3.0f,   0.0f, 0.0f,+0.1f,   BSEED, "Rz+3_tz+0.1"   });

    /* ---- Condition C: pose fixed at TOP, seed variation ---- */
    const unsigned C_SEEDS[5] = { 42u, 123u, 456u, 789u, 2024u };
    for (int i = 0; i < 5; i++) {
        std::string lab = "seed=" + std::to_string(C_SEEDS[i]);
        trials.push_back({ 'C', i,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,
                           C_SEEDS[i], lab });
    }
}

inline void ProtocolState::begin(const std::string& imageName_,
                                 const std::string& outDir_)
{
    if (active) {
        std::cerr << "[Protocol] Already active — ignoring begin()" << std::endl;
        return;
    }

    imageName = imageName_;
    outDir    = outDir_;

    buildTrialTable();

    trial_i = 0;
    phase   = PRE_TRIAL;
    call_i  = 0;

    best_H_verts.clear();   best_H_normals.clear();   best_H_rmse = 0.0f;   best_H_call_idx = -1;
    best_C_verts.clear();   best_C_normals.clear();   best_C_rmse = 0.0f;   best_C_call_idx = -1;
    before_R_rmse = 0.0f;

    g_protocolLog.begin(imageName, outDir);
    g_suppressPoseLibSave = true;

    active = true;
    progress.active = true;
    progress.trial_total = (int)trials.size();

    std::cout << "[Protocol] begin() — image='" << imageName
              << "'  trials=" << trials.size() << std::endl;
    std::cout << "[Protocol] (Step 4b-1 skeleton: advance() is a no-op until 4b-3)" << std::endl;
}

inline void ProtocolState::abort() {
    if (!active) return;
    std::cerr << "[Protocol] abort() — flushing log and stopping" << std::endl;
    g_protocolLog.save();
    g_suppressPoseLibSave = false;
    active = false;
    progress.active = false;
}

/* Step 4b-1: no-op stub. Real state machine arrives in 4b-3. */
inline void ProtocolState::advance() {
    if (!active) return;
    /* Intentionally empty. When 4b-3 lands this becomes the
     * switch(phase) { PRE_TRIAL / PHASE_H / PHASE_C / PHASE_R
     * / POST_TRIAL / FINISHED } block. */
}


#endif /* PROTOCOL_RUNNER_H */
