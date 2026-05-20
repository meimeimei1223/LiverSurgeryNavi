#pragma once
/*
 * ProtocolLog.h
 * --------------
 * Lightweight, write-only logger for the Shift+P MICCAI-WS protocol runs.
 *
 * Completely independent of PoseLibrary (which logs interactive user
 * sessions). This class captures one run of the automated protocol:
 *
 *   Condition B (pose perturbation, fixed seed)  x  5 trials
 *   Condition C (seed variation, TOP pose)       x  5 trials
 *
 *   Each trial:  H (HemiAuto x2)  +  C (CMA-ES x10)  +  R (Refine x1)
 *               = 13 entries per trial
 *               = 130 entries per image
 *
 * Outputs (per image, per Shift+P invocation):
 *   (1)  protocol_YYYYMMDD_HHMMSS.csv   — every entry in full detail
 *   (2)  protocol_YYYYMMDD_HHMMSS.txt   — one line per trial, summary
 *
 * Design notes:
 *   - Write-only. No import / undo / apply. Keep it simple.
 *   - Mesh state is managed externally via g_initOrganVertices and the
 *     existing PoseLibrary::applyTransformToMeshes helpers; we only
 *     record the resulting 4x4 transform here.
 *   - save() flushes the in-memory buffer to disk; call once at the
 *     end of the protocol run. CSV and TXT are written atomically in
 *     this single call.
 *   - Determinism: trial_seed drives FGR/BIPOP/CMA via offsets
 *     (trial_seed, trial_seed+1000, trial_seed+2000). These are
 *     stored verbatim in the CSV for reproducibility.
 */

#ifndef PROTOCOL_LOG_H
#define PROTOCOL_LOG_H

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <iostream>

#include <glm/glm.hpp>


/* ============================================================
 * One row in the protocol CSV.
 * ============================================================ */
struct ProtocolEntry {
    /* --- identity --- */
    int         entry_id      = 0;      /* 1-based running counter within this run  */
    std::string timestamp;              /* "YYYY-MM-DD HH:MM:SS"                     */
    std::string image_name;             /* basename stem, e.g. "image1"              */

    /* --- trial locator --- */
    char        condition     = '?';    /* 'B' (pose perturb) or 'C' (seed variation) */
    int         trial_idx     = -1;     /* 0..4 within the condition                 */
    std::string perturbation;           /* "TOP" / "Rx+3_tx+0.1" / "seed=42"         */

    /* --- seeds (trial_seed derives the other two via +1000 / +2000) --- */
    unsigned    trial_seed       = 0;   /* master seed for this trial                */
    unsigned    fgr_seed         = 0;   /* = trial_seed                              */
    unsigned    bipop_outer_seed = 0;   /* = trial_seed + 1000                       */
    unsigned    cma_inner_seed   = 0;   /* = trial_seed + 2000 (+ call_idx offset?)  */

    /* --- phase locator --- */
    char        phase         = '?';    /* 'H' (HemiAuto) / 'C' (CMA-ES) / 'R' (Refine) */
    int         call_idx      = 0;      /* 0-based, within this phase of this trial  */

    /* --- metrics (from computeUnifiedMetrics after this step) --- */
    float       comp_rmse      = 0.0f;
    float       comp_avg_error = 0.0f;
    float       comp_max_error = 0.0f;
    int         comp_count     = 0;

    /* --- HemiAuto-specific (valid only when phase=='H') --- */
    float       base_fitness    = 0.0f;
    float       base_icp_rmse   = 0.0f;
    float       base_scale      = 1.0f;

    /* --- Refine-specific (valid only when phase=='R') ---
     * refine_applied   : 1 if Refine was invoked at all (always 1 for R entries)
     * refine_improved  : 1 if Refine moved the pose (i.e. compRMSE decreased)
     * refine_delta_rmse: comp_rmse_after - comp_rmse_before_refine
     *                    (negative = improvement, 0.0 = reverted / no change)
     */
    int         refine_applied    = 0;
    int         refine_improved   = 0;
    float       refine_delta_rmse = 0.0f;

    /* --- timing --- */
    float       elapsed_sec   = 0.0f;   /* wall-clock for this entry's computation  */

    /* --- final 4x4 transform for this entry (world-space, cumulative from TOP) --- */
    glm::mat4   transform     = glm::mat4(1.0f);
};


/* ============================================================
 * The protocol log itself.
 * ============================================================ */
struct ProtocolLog {
    /* --- buffer --- */
    std::vector<ProtocolEntry> entries;

    /* --- metadata about the current run --- */
    std::string imageName;
    std::string outDir;                 /* directory for output files                */
    std::string fileTimestamp;          /* "YYYYMMDD_HHMMSS" for filenames           */
    int         nextEntryId = 1;

    /* --- helpers ----------------------------------------------------- */

    static std::string nowTimestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
        return std::string(buf);
    }

    static std::string nowFileStamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
        return std::string(buf);
    }

    /* --- public API -------------------------------------------------- */

    /* Call once at the start of a protocol run (e.g. from Shift+P handler).
     * `imageName` should be the basename stem (e.g. "image1").
     * `outDir` is where the CSV/TXT will be written; typically the same
     * directory used by PoseLibrary::exportToCsv, or "./" for cwd.
     */
    void begin(const std::string& imageName_,
               const std::string& outDir_ = "./")
    {
        entries.clear();
        nextEntryId   = 1;
        imageName     = imageName_;
        outDir        = outDir_;
        fileTimestamp = nowFileStamp();

        std::cout << "[ProtocolLog] Session start: image='" << imageName
                  << "' stamp=" << fileTimestamp << std::endl;
    }

    /* Append a single entry.  The caller sets all fields except entry_id
     * and timestamp, which we fill in here to keep sequencing consistent. */
    void add(ProtocolEntry e) {
        e.entry_id  = nextEntryId++;
        e.timestamp = nowTimestamp();
        entries.push_back(std::move(e));
    }

    /* Total count of entries currently buffered. */
    size_t size() const { return entries.size(); }

    /* --- output: full CSV ------------------------------------------- */

    bool saveCsv(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "[ProtocolLog] Cannot open CSV for write: "
                      << filepath << std::endl;
            return false;
        }

        /* header */
        ofs << "entry_id,timestamp,image_name,"
            << "condition,trial_idx,perturbation,"
            << "trial_seed,fgr_seed,bipop_outer_seed,cma_inner_seed,"
            << "phase,call_idx,"
            << "comp_rmse,comp_avg_error,comp_max_error,comp_count,"
            << "base_fitness,base_icp_rmse,base_scale,"
            << "refine_applied,refine_improved,refine_delta_rmse,"
            << "elapsed_sec,"
            << "m00,m01,m02,m03,"
            << "m10,m11,m12,m13,"
            << "m20,m21,m22,m23,"
            << "m30,m31,m32,m33"
            << "\n";

        ofs << std::fixed << std::setprecision(6);

        for (const auto& e : entries) {
            ofs << e.entry_id          << ","
                << e.timestamp         << ","
                << e.image_name        << ","
                << e.condition         << ","
                << e.trial_idx         << ","
                << e.perturbation      << ","
                << e.trial_seed        << ","
                << e.fgr_seed          << ","
                << e.bipop_outer_seed  << ","
                << e.cma_inner_seed    << ","
                << e.phase             << ","
                << e.call_idx          << ","
                << e.comp_rmse         << ","
                << e.comp_avg_error    << ","
                << e.comp_max_error    << ","
                << e.comp_count        << ","
                << e.base_fitness      << ","
                << e.base_icp_rmse     << ","
                << e.base_scale        << ","
                << e.refine_applied    << ","
                << e.refine_improved   << ","
                << e.refine_delta_rmse << ","
                << e.elapsed_sec;

            /* 4x4 transform, column-major like PoseLibrary */
            for (int col = 0; col < 4; col++)
                for (int row = 0; row < 4; row++)
                    ofs << "," << e.transform[col][row];
            ofs << "\n";
        }

        ofs.close();
        std::cout << "[ProtocolLog] CSV written: " << filepath
                  << " (" << entries.size() << " entries)" << std::endl;
        return true;
    }

    /* --- output: human-readable trial summary ----------------------- */

    /* For each unique (condition, trial_idx) pair, compute a one-line
     * summary by scanning the entries. Layout:
     *
     *   image1 B0 (TOP,        seed=42   ) H=0.26461 C_min=0.20468 (run 7)  R=0.20468 improved=0  time=4.81s
     *   image1 B1 (Rx+3,tx+0.1 seed=43   ) H=0.27102 C_min=0.19844 (run 4)  R=0.19844 improved=0  time=4.56s
     *   ...
     *
     * Plus a final "SUMMARY:" line giving the best trial overall.
     */
    bool saveTxt(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "[ProtocolLog] Cannot open TXT for write: "
                      << filepath << std::endl;
            return false;
        }

        ofs << "# Protocol summary for image '" << imageName << "'\n"
            << "# Generated: " << nowTimestamp() << "\n"
            << "# Columns: <image> <cond><trial> (<perturbation>) "
               "H=<hemi_best> C_min=<cma_best> (run <idx>) R=<after_refine> "
               "improved=<0|1> time=<trial_seconds>\n"
            << "# Notes:\n"
            << "#   H      = min compRMSE across HemiAuto calls (phase='H')\n"
            << "#   C_min  = min compRMSE across CMA-ES calls (phase='C')\n"
            << "#   run    = 1-based call_idx that produced C_min\n"
            << "#   R      = compRMSE after the Refine attempt (phase='R')\n"
            << "#   improved=1 iff Refine reduced compRMSE below C_min\n"
            << "#   time   = sum of elapsed_sec across all entries in the trial\n"
            << "# -----------------------------------------------------------\n";

        ofs << std::fixed << std::setprecision(5);

        /* Collect unique trials in first-seen order to preserve
         * protocol execution ordering in the TXT. */
        struct TrialKey {
            char cond;
            int  idx;
            bool operator==(const TrialKey& o) const {
                return cond == o.cond && idx == o.idx;
            }
        };

        std::vector<TrialKey> order;
        for (const auto& e : entries) {
            TrialKey k{ e.condition, e.trial_idx };
            bool found = false;
            for (const auto& existing : order)
                if (existing == k) { found = true; break; }
            if (!found) order.push_back(k);
        }

        /* For the final SUMMARY line we track the global best. */
        float best_overall = std::numeric_limits<float>::infinity();
        char  best_cond    = '?';
        int   best_trial   = -1;

        for (const auto& k : order) {
            /* Scan entries belonging to this trial. */
            float h_best = std::numeric_limits<float>::infinity();
            float c_best = std::numeric_limits<float>::infinity();
            int   c_best_call = -1;
            float r_after = std::numeric_limits<float>::quiet_NaN();
            int   r_improved = 0;
            float trial_time = 0.0f;
            std::string perturb_str;

            for (const auto& e : entries) {
                if (e.condition != k.cond || e.trial_idx != k.idx) continue;
                if (perturb_str.empty()) perturb_str = e.perturbation;
                trial_time += e.elapsed_sec;

                if (e.phase == 'H') {
                    if (e.comp_rmse < h_best) h_best = e.comp_rmse;
                } else if (e.phase == 'C') {
                    if (e.comp_rmse < c_best) {
                        c_best      = e.comp_rmse;
                        c_best_call = e.call_idx;
                    }
                } else if (e.phase == 'R') {
                    r_after    = e.comp_rmse;
                    r_improved = e.refine_improved;
                }
            }

            /* Track global best (using the "final" value: prefer R if
             * present and improved, else C_min). */
            float trial_final = c_best;
            if (!std::isnan(r_after) && r_improved == 1) trial_final = r_after;
            if (trial_final < best_overall) {
                best_overall = trial_final;
                best_cond    = k.cond;
                best_trial   = k.idx;
            }

            /* Emit the line.  std::setw for gentle column alignment. */
            ofs << imageName << " "
                << k.cond << k.idx << " "
                << "(" << std::left << std::setw(22) << perturb_str << std::right << ") "
                << "H=" << std::setw(8) << h_best << " "
                << "C_min=" << std::setw(8) << c_best << " "
                << "(run " << std::setw(2) << (c_best_call + 1) << ") ";

            if (std::isnan(r_after)) {
                ofs << "R=   N/A   improved=-  ";
            } else {
                ofs << "R=" << std::setw(8) << r_after << " "
                    << "improved=" << r_improved << "  ";
            }
            ofs << "time=" << std::fixed << std::setprecision(2)
                << std::setw(6) << trial_time << "s"
                << std::setprecision(5) << "\n";
        }

        /* Final SUMMARY */
        ofs << "# -----------------------------------------------------------\n";
        if (best_trial >= 0) {
            ofs << "# SUMMARY: best trial = " << best_cond << best_trial
                << "  compRMSE = " << std::setprecision(6) << best_overall
                << "  (total entries: " << entries.size() << ")\n";
        } else {
            ofs << "# SUMMARY: no valid trials recorded.\n";
        }

        ofs.close();
        std::cout << "[ProtocolLog] TXT written: " << filepath << std::endl;
        return true;
    }

    /* Save both artefacts using the standard naming convention. Returns
     * true only if both succeeded. */
    bool save() {
        if (entries.empty()) {
            std::cerr << "[ProtocolLog] No entries to save." << std::endl;
            return false;
        }
        std::string base = outDir;
        if (!base.empty() && base.back() != '/' && base.back() != '\\')
            base += "/";
        base += "protocol_" + fileTimestamp;

        bool ok_csv = saveCsv(base + ".csv");
        bool ok_txt = saveTxt(base + ".txt");
        return ok_csv && ok_txt;
    }
};

#endif /* PROTOCOL_LOG_H */
