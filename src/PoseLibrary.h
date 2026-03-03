#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <iostream>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "mCutMesh.h"

struct PoseEntry {
    int id = -1;

    enum Method { FULL_AUTO = 0, HEMI_AUTO = 1, UMEYAMA = 2 };
    Method baseMethod = FULL_AUTO;
    int refineCount = 0;

    std::string timestamp;

    // --- Base registration metrics ---
    float baseFitness    = 0.0f;
    float baseIcpRmse    = 0.0f;
    float baseAvgError   = 0.0f;
    float baseRmse       = 0.0f;
    float baseMaxError   = 0.0f;
    float baseScale      = 1.0f;

    // --- Refine metrics (valid only when refineCount > 0) ---
    float refineInitialRMSE  = 0.0f;
    float refineBestRMSE     = 0.0f;
    int   refineBestIteration = 0;

    // --- Unified comparison metrics (all-organ vs depth, all correspondences) ---
    float compRmse     = 0.0f;
    float compAvgError = 0.0f;
    float compMaxError = 0.0f;
    int   compCount    = 0;
    std::vector<glm::vec3> corrSource;  // correspondence pairs for export
    std::vector<glm::vec3> corrTarget;

    // --- Vertex snapshots for all organ meshes ---
    std::vector<std::vector<GLfloat>> organVertices;
    std::vector<std::vector<GLfloat>> organNormals;

    float finalRmse() const {
        // Unified comparison RMSE (same method for all entries)
        return compRmse;
    }

    const char* methodStr() const {
        switch (baseMethod) {
        case FULL_AUTO: return "FullAuto";
        case HEMI_AUTO: return "HemiAuto";
        case UMEYAMA:   return "Umeyama";
        }
        return "Unknown";
    }

    std::string label() const {
        std::string s = methodStr();
        if (refineCount > 0) {
            s += "+Refine";
            if (refineCount > 1) s += "x" + std::to_string(refineCount);
        }
        return s;
    }
};

class PoseLibrary {
public:
    std::vector<PoseEntry> entries;
    int maxEntries = 50;
    int nextId     = 1;

    // --- Last registration auto-saved (for undo) ---
    PoseEntry lastRegistration;
    bool hasLastRegistration = false;

    // --- Currently applied entry (for highlighting) ---
    int activeEntryId = -1;

    // --- UI state ---
    bool showWindow = false;

    // -------------------------------------------------------
    // Snapshot helpers
    // -------------------------------------------------------
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

    static void snapshotMeshes(PoseEntry& entry,
                               const std::vector<mCutMesh*>& organs) {
        entry.organVertices.resize(organs.size());
        entry.organNormals.resize(organs.size());
        for (size_t i = 0; i < organs.size(); i++) {
            entry.organVertices[i] = organs[i]->mVertices;
            entry.organNormals[i]  = organs[i]->mNormals;
        }
    }

    static void restoreMeshes(const PoseEntry& entry,
                              std::vector<mCutMesh*>& organs) {
        for (size_t i = 0; i < organs.size() && i < entry.organVertices.size(); i++) {
            organs[i]->mVertices = entry.organVertices[i];
            organs[i]->mNormals  = entry.organNormals[i];
            setUp(*organs[i]);
        }
    }

    // -------------------------------------------------------
    //  Auto-save on every registration completion
    // -------------------------------------------------------
    void autoSaveLastRegistration(const PoseEntry& entry) {
        lastRegistration = entry;
        hasLastRegistration = true;
        std::cout << "[PoseLibrary] Undo snapshot saved" << std::endl;
    }

    // -------------------------------------------------------
    //  Undo = restore lastRegistration
    // -------------------------------------------------------
    bool undoToLast(std::vector<mCutMesh*>& organs) {
        if (!hasLastRegistration) {
            std::cout << "[PoseLibrary] Nothing to undo" << std::endl;
            return false;
        }
        restoreMeshes(lastRegistration, organs);
        activeEntryId = -1;
        std::cout << "[PoseLibrary] Undo: restored to last saved state ("
                  << lastRegistration.label() << ")" << std::endl;
        return true;
    }

    // -------------------------------------------------------
    //  Manual save to library (user presses key)
    // -------------------------------------------------------
    PoseEntry buildEntryFromCurrent(
        PoseEntry::Method method,
        int refineCount,
        float baseFitness, float baseIcpRmse,
        float baseAvgError, float baseRmse, float baseMaxError,
        float baseScale,
        float refineInitRMSE, float refineBestRMSE, int refineBestIter,
        float compRmse, float compAvgError, float compMaxError, int compCount,
        const std::vector<glm::vec3>& compSrc,
        const std::vector<glm::vec3>& compTgt,
        const std::vector<mCutMesh*>& organs)
    {
        PoseEntry e;
        e.id              = nextId++;
        e.baseMethod      = method;
        e.refineCount     = refineCount;
        e.timestamp       = nowTimestamp();
        e.baseFitness     = baseFitness;
        e.baseIcpRmse     = baseIcpRmse;
        e.baseAvgError    = baseAvgError;
        e.baseRmse        = baseRmse;
        e.baseMaxError    = baseMaxError;
        e.baseScale       = baseScale;
        e.refineInitialRMSE   = refineInitRMSE;
        e.refineBestRMSE      = refineBestRMSE;
        e.refineBestIteration = refineBestIter;
        e.compRmse     = compRmse;
        e.compAvgError = compAvgError;
        e.compMaxError = compMaxError;
        e.compCount    = compCount;
        e.corrSource   = compSrc;
        e.corrTarget   = compTgt;
        snapshotMeshes(e, organs);
        return e;
    }

    void addEntry(const PoseEntry& entry) {
        entries.push_back(entry);

        // Trim to max (remove oldest)
        while ((int)entries.size() > maxEntries) {
            entries.erase(entries.begin());
        }

        std::cout << "[PoseLibrary] Added entry #" << entry.id
                  << " (" << entry.label()
                  << ", CompRMSE=" << entry.compRmse
                  << "). Library size: " << entries.size() << std::endl;
    }

    void saveCurrentToLibrary(
        PoseEntry::Method method,
        int refineCount,
        float baseFitness, float baseIcpRmse,
        float baseAvgError, float baseRmse, float baseMaxError,
        float baseScale,
        float refineInitRMSE, float refineBestRMSE, int refineBestIter,
        float compRmse, float compAvgError, float compMaxError, int compCount,
        const std::vector<glm::vec3>& compSrc,
        const std::vector<glm::vec3>& compTgt,
        const std::vector<mCutMesh*>& organs)
    {
        PoseEntry e = buildEntryFromCurrent(
            method, refineCount,
            baseFitness, baseIcpRmse, baseAvgError, baseRmse, baseMaxError, baseScale,
            refineInitRMSE, refineBestRMSE, refineBestIter,
            compRmse, compAvgError, compMaxError, compCount,
            compSrc, compTgt,
            organs);
        addEntry(e);
    }

    // -------------------------------------------------------
    //  Apply a library entry
    // -------------------------------------------------------
    bool applyEntry(int entryId, std::vector<mCutMesh*>& organs) {
        for (auto& e : entries) {
            if (e.id == entryId) {
                // Save current state as lastRegistration before overwriting
                PoseEntry backup;
                backup.timestamp = nowTimestamp();
                snapshotMeshes(backup, organs);
                lastRegistration = backup;
                hasLastRegistration = true;

                restoreMeshes(e, organs);
                activeEntryId = entryId;
                std::cout << "[PoseLibrary] Applied entry #" << entryId
                          << " (" << e.label() << ")" << std::endl;
                return true;
            }
        }
        std::cout << "[PoseLibrary] Entry #" << entryId << " not found" << std::endl;
        return false;
    }

    // -------------------------------------------------------
    //  Delete an entry
    // -------------------------------------------------------
    void deleteEntry(int entryId) {
        entries.erase(
            std::remove_if(entries.begin(), entries.end(),
                           [entryId](const PoseEntry& e) { return e.id == entryId; }),
            entries.end());
        if (activeEntryId == entryId) activeEntryId = -1;
    }

    // -------------------------------------------------------
    //  Export to CSV (Python-friendly)
    // -------------------------------------------------------
    bool exportToCsv(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "[PoseLibrary] Cannot open " << filepath << std::endl;
            return false;
        }

        // Header
        ofs << "id,method,refine_count,timestamp,"
            << "base_fitness,base_icp_rmse,base_corr_avg_error,base_corr_rmse,base_corr_max_error,base_scale,"
            << "refine_initial_rmse,refine_best_rmse,refine_best_iteration,"
            << "comp_rmse,comp_avg_error,comp_max_error,comp_count" << std::endl;

        ofs << std::fixed << std::setprecision(8);

        for (const auto& e : entries) {
            ofs << e.id << ","
                << e.label() << ","
                << e.refineCount << ","
                << e.timestamp << ","
                << e.baseFitness << ","
                << e.baseIcpRmse << ","
                << e.baseAvgError << ","
                << e.baseRmse << ","
                << e.baseMaxError << ","
                << e.baseScale << ","
                << e.refineInitialRMSE << ","
                << e.refineBestRMSE << ","
                << e.refineBestIteration << ","
                << e.compRmse << ","
                << e.compAvgError << ","
                << e.compMaxError << ","
                << e.compCount << std::endl;
        }

        std::cout << "[PoseLibrary] Exported " << entries.size()
                  << " entries to " << filepath << std::endl;
        return true;
    }

    // -------------------------------------------------------
    //  Export correspondence pairs for a single entry (Python analysis)
    // -------------------------------------------------------
    bool exportCorrespondences(int entryId, const std::string& filepath) const {
        for (const auto& e : entries) {
            if (e.id == entryId) {
                std::ofstream ofs(filepath);
                if (!ofs.is_open()) return false;

                ofs << "source_x,source_y,source_z,target_x,target_y,target_z,distance"
                    << std::endl;
                ofs << std::fixed << std::setprecision(8);

                for (size_t i = 0; i < e.corrSource.size() && i < e.corrTarget.size(); i++) {
                    float d = glm::distance(e.corrSource[i], e.corrTarget[i]);
                    ofs << e.corrSource[i].x << ","
                        << e.corrSource[i].y << ","
                        << e.corrSource[i].z << ","
                        << e.corrTarget[i].x << ","
                        << e.corrTarget[i].y << ","
                        << e.corrTarget[i].z << ","
                        << d << std::endl;
                }

                std::cout << "[PoseLibrary] Exported " << e.corrSource.size()
                          << " correspondences for entry #" << entryId
                          << " to " << filepath << std::endl;
                return true;
            }
        }
        return false;
    }

};
