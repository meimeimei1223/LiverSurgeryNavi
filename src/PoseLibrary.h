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

    // --- Transform from initial pose (applied to all organs equally) ---
    glm::mat4 transform = glm::mat4(1.0f);

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

    static glm::mat4 computeTransformFromLiver(
        const std::vector<GLfloat>& initVerts,
        const std::vector<GLfloat>& curVerts)
    {
        size_t n = initVerts.size() / 3;
        if (n == 0) return glm::mat4(1.0f);

        glm::vec3 srcCenter(0), dstCenter(0);
        for (size_t i = 0; i < n; i++) {
            srcCenter += glm::vec3(initVerts[i*3], initVerts[i*3+1], initVerts[i*3+2]);
            dstCenter += glm::vec3(curVerts[i*3],  curVerts[i*3+1],  curVerts[i*3+2]);
        }
        srcCenter /= (float)n;
        dstCenter /= (float)n;

        glm::mat3 H(0.0f);
        for (size_t i = 0; i < n; i++) {
            glm::vec3 s = glm::vec3(initVerts[i*3], initVerts[i*3+1], initVerts[i*3+2]) - srcCenter;
            glm::vec3 d = glm::vec3(curVerts[i*3],  curVerts[i*3+1],  curVerts[i*3+2])  - dstCenter;
            H += glm::outerProduct(d, s);
        }

        // SVD via Jacobi (simple 3x3)
        // Use glm decomposition via QR-ish approach — approximate for rigid body
        // For exact SVD we use the cross-covariance directly with glm::mat3
        // R = H * inverse(H^T * H)^0.5  — use polar decomposition approximation
        // Since meshes are rigid, we use the standard approach:
        // build normal equations and extract rotation
        glm::mat3 Ht = glm::transpose(H);
        glm::mat3 HtH = Ht * H;

        // Approximate sqrt via eigendecomposition is complex; use iterative polar decomp
        glm::mat3 R = H;
        for (int iter = 0; iter < 100; iter++) {
            glm::mat3 Rinv = glm::inverse(R);
            R = 0.5f * (R + glm::transpose(Rinv));
        }

        glm::vec3 t = dstCenter - R * srcCenter;

        glm::mat4 T(1.0f);
        T[0] = glm::vec4(R[0], 0.0f);
        T[1] = glm::vec4(R[1], 0.0f);
        T[2] = glm::vec4(R[2], 0.0f);
        T[3] = glm::vec4(t, 1.0f);
        return T;
    }

    static void applyTransformToMeshes(
        const glm::mat4& T,
        const std::vector<std::vector<GLfloat>>& initVerts,
        const std::vector<std::vector<GLfloat>>& initNormals,
        std::vector<mCutMesh*>& organs)
    {
        glm::mat3 R = glm::mat3(T);
        for (size_t m = 0; m < organs.size() && m < initVerts.size(); m++) {
            const auto& iv = initVerts[m];
            const auto& in_ = initNormals[m];
            auto* mesh = organs[m];
            size_t nv = iv.size() / 3;
            mesh->mVertices.resize(iv.size());
            mesh->mNormals.resize(in_.size());
            for (size_t i = 0; i < nv; i++) {
                glm::vec4 v(iv[i*3], iv[i*3+1], iv[i*3+2], 1.0f);
                v = T * v;
                mesh->mVertices[i*3]   = v.x;
                mesh->mVertices[i*3+1] = v.y;
                mesh->mVertices[i*3+2] = v.z;
            }
            size_t nn = in_.size() / 3;
            for (size_t i = 0; i < nn; i++) {
                glm::vec3 n(in_[i*3], in_[i*3+1], in_[i*3+2]);
                n = glm::normalize(R * n);
                mesh->mNormals[i*3]   = n.x;
                mesh->mNormals[i*3+1] = n.y;
                mesh->mNormals[i*3+2] = n.z;
            }
            setUp(*mesh);
        }
    }

    // -------------------------------------------------------
    //  Auto-save on every registration completion
    // -------------------------------------------------------
    void autoSaveLastRegistration(const glm::mat4& transform) {
        lastRegistration = PoseEntry();
        lastRegistration.transform = transform;
        lastRegistration.timestamp = nowTimestamp();
        hasLastRegistration = true;
        std::cout << "[PoseLibrary] Undo snapshot saved" << std::endl;
    }

    // -------------------------------------------------------
    //  Undo = restore lastRegistration
    // -------------------------------------------------------
    bool undoToLast(
        const std::vector<std::vector<GLfloat>>& initVerts,
        const std::vector<std::vector<GLfloat>>& initNormals,
        std::vector<mCutMesh*>& organs)
    {
        if (!hasLastRegistration) {
            std::cout << "[PoseLibrary] Nothing to undo" << std::endl;
            return false;
        }
        applyTransformToMeshes(lastRegistration.transform, initVerts, initNormals, organs);
        activeEntryId = -1;
        std::cout << "[PoseLibrary] Undo: restored" << std::endl;
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
        const glm::mat4& transform)
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
        e.transform    = transform;
        return e;
    }

    void addEntry(const PoseEntry& entry) {
        entries.push_back(entry);
        while ((int)entries.size() > maxEntries)
            entries.erase(entries.begin());
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
        const glm::mat4& transform)
    {
        PoseEntry e = buildEntryFromCurrent(
            method, refineCount,
            baseFitness, baseIcpRmse, baseAvgError, baseRmse, baseMaxError, baseScale,
            refineInitRMSE, refineBestRMSE, refineBestIter,
            compRmse, compAvgError, compMaxError, compCount,
            compSrc, compTgt, transform);
        addEntry(e);
    }

    // -------------------------------------------------------
    //  Apply a library entry
    // -------------------------------------------------------
    bool applyEntry(
        int entryId,
        const std::vector<std::vector<GLfloat>>& initVerts,
        const std::vector<std::vector<GLfloat>>& initNormals,
        std::vector<mCutMesh*>& organs)
    {
        for (auto& e : entries) {
            if (e.id == entryId) {
                lastRegistration = PoseEntry();
                lastRegistration.transform = e.transform;
                lastRegistration.timestamp = nowTimestamp();
                hasLastRegistration = true;

                applyTransformToMeshes(e.transform, initVerts, initNormals, organs);
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

        ofs << "id,method,refine_count,timestamp,"
            << "base_fitness,base_icp_rmse,base_corr_avg_error,base_corr_rmse,base_corr_max_error,base_scale,"
            << "refine_initial_rmse,refine_best_rmse,refine_best_iteration,"
            << "comp_rmse,comp_avg_error,comp_max_error,comp_count,"
            << "m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
            << std::endl;

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
                << e.compCount;
            for (int col = 0; col < 4; col++)
                for (int row = 0; row < 4; row++)
                    ofs << "," << e.transform[col][row];
            ofs << std::endl;
        }

        std::cout << "[PoseLibrary] Exported " << entries.size()
                  << " entries to " << filepath << std::endl;
        return true;
    }

    bool importFromCsv(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            std::cerr << "[PoseLibrary] Cannot open " << filepath << std::endl;
            return false;
        }

        std::string line;
        std::getline(ifs, line); // skip header

        int count = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<std::string> tok;
            std::string field;
            while (std::getline(ss, field, ',')) tok.push_back(field);
            if (tok.size() < 33) continue;

            PoseEntry e;
            e.id           = nextId++;
            std::string lbl = tok[1];
            if      (lbl.find("FullAuto") != std::string::npos) e.baseMethod = PoseEntry::FULL_AUTO;
            else if (lbl.find("HemiAuto") != std::string::npos) e.baseMethod = PoseEntry::HEMI_AUTO;
            else                                                 e.baseMethod = PoseEntry::UMEYAMA;
            e.refineCount         = std::stoi(tok[2]);
            e.timestamp           = tok[3];
            e.baseFitness         = std::stof(tok[4]);
            e.baseIcpRmse         = std::stof(tok[5]);
            e.baseAvgError        = std::stof(tok[6]);
            e.baseRmse            = std::stof(tok[7]);
            e.baseMaxError        = std::stof(tok[8]);
            e.baseScale           = std::stof(tok[9]);
            e.refineInitialRMSE   = std::stof(tok[10]);
            e.refineBestRMSE      = std::stof(tok[11]);
            e.refineBestIteration = std::stoi(tok[12]);
            e.compRmse            = std::stof(tok[13]);
            e.compAvgError        = std::stof(tok[14]);
            e.compMaxError        = std::stof(tok[15]);
            e.compCount           = std::stoi(tok[16]);
            int ti = 17;
            for (int col = 0; col < 4; col++)
                for (int row = 0; row < 4; row++)
                    e.transform[col][row] = std::stof(tok[ti++]);

            entries.push_back(e);
            count++;
        }

        std::cout << "[PoseLibrary] Imported " << count
                  << " entries from " << filepath << std::endl;
        return count > 0;
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
