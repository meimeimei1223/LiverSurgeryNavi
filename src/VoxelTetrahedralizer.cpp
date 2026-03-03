#include "VoxelTetrahedralizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cmath>
#include <omp.h>

VoxelTetrahedralizer::VoxelTetrahedralizer(int gridSize, const std::string& targetPath, const std::string& outputTetPath)
    : gridSize_(gridSize), targetPath_(targetPath), outputTetPath_(outputTetPath) {

    voxels_.resize(gridSize_);
    for (int i = 0; i < gridSize_; i++) {
        voxels_[i].resize(gridSize_);
        for (int j = 0; j < gridSize_; j++) {
            voxels_[i][j].resize(gridSize_);
        }
    }

    initializeCube();
}

const VoxelTetrahedralizer::MeshData& VoxelTetrahedralizer::getVoxTetData() const {
    return voxTetData;
}

void VoxelTetrahedralizer::setSmoothingEnabled(bool enable, int iterations, float factor, bool adjustSize, int scalingMethod) {
    enableSmoothing_ = enable;
    smoothingIterations_ = iterations;
    smoothingFactor_ = factor;
    enableSizeAdjustment_ = adjustSize;
    scalingMethod_ = scalingMethod;
}

void VoxelTetrahedralizer::setInflationSettings(const InflationSettings& settings) {
    inflationSettings_ = settings;
}

void VoxelTetrahedralizer::setInflationEnabled(bool enabled) {
    inflationSettings_.enabled = enabled;
}

void VoxelTetrahedralizer::setTargetCoverage(float coverage) {
    inflationSettings_.targetCoverage = coverage;
}

void VoxelTetrahedralizer::setInflationParameters(float baseAmount, float increaseRate, float decreaseRate) {
    inflationSettings_.baseInflateAmount = baseAmount;
    inflationSettings_.inflateIncreaseRate = increaseRate;
    inflationSettings_.inflateDecreaseRate = decreaseRate;
}

void VoxelTetrahedralizer::setVerbose(bool v) {
    verbose_ = v;
}

// Main execute
MeshDataTypes::SimpleMeshData VoxelTetrahedralizer::execute() {
    std::cout << "\n=== Voxel Tetrahedralization ===" << std::endl;
    std::cout << "Grid: " << gridSize_ << "x" << gridSize_ << "x" << gridSize_;
    if (enableSmoothing_) std::cout << " | Smooth: ON";
    if (inflationSettings_.enabled) std::cout << " | Inflate: ON (target " << inflationSettings_.targetCoverage << "%)";
    std::cout << std::endl;

    totalStartTime_ = std::chrono::high_resolution_clock::now();

    // Step 1
    if (verbose_) std::cout << "\n[Step 1/4] Preparing cube mesh..." << std::endl;
    auto step1Start = std::chrono::high_resolution_clock::now();
    prepareCube();
    initializeVoxelGrid();
    step1Duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - step1Start);

    // Step 2
    if (verbose_) std::cout << "\n[Step 2/4] Checking voxel centers..." << std::endl;
    auto step2Start = std::chrono::high_resolution_clock::now();
    checkVoxelCenters();
    step2Duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - step2Start);

    // Step 3
    if (verbose_) std::cout << "\n[Step 3/4] Carving external voxels..." << std::endl;
    auto step3Start = std::chrono::high_resolution_clock::now();
    carveExternalVoxels();
    step3Duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - step3Start);

    // Step 4
    if (verbose_) std::cout << "\n[Step 4/4] Generating tetrahedra..." << std::endl;
    auto step4Start = std::chrono::high_resolution_clock::now();
    generateTetrahedra();
    step4Duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - step4Start);

    // Auto-inflation
    if (inflationSettings_.enabled) {
        performAutomaticInflation();
    }

    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - totalStartTime_);

    std::cout << "=== Tetrahedralization Complete (" << std::fixed << std::setprecision(2)
              << totalDuration.count() / 1000.0 << "s) ===" << std::endl;

    return generateDisplayMeshData();
}

// Debug coverage check
VoxelTetrahedralizer::DebugInfo VoxelTetrahedralizer::debugCheckTargetVertexCoverage() {
    DebugInfo info;

    size_t numTargetVerts = targetVertices_.size() / 3;
    info.totalTargetVertices = numTargetVerts;

    if (voxTetData.tetIds.empty()) {
        std::cerr << "Error: No tetrahedra generated yet." << std::endl;
        return info;
    }

    if (verbose_) {
        std::cout << "\n=== Debug: Target Vertex Coverage Check ===" << std::endl;
        std::cout << "Checking " << numTargetVerts << " Target vertices against "
                  << voxTetData.tetIds.size() / 4 << " tetrahedra..." << std::endl;
    }

    for (size_t i = 0; i < numTargetVerts; i++) {
        glm::vec3 vertex(targetVertices_[i * 3],
                         targetVertices_[i * 3 + 1],
                         targetVertices_[i * 3 + 2]);

        bool foundInTet = false;
        std::vector<int> containingTets;

        size_t numTets = voxTetData.tetIds.size() / 4;
        for (size_t tetIdx = 0; tetIdx < numTets; tetIdx++) {
            if (isPointInTetrahedron(vertex, tetIdx)) {
                foundInTet = true;
                containingTets.push_back(tetIdx);
            }
        }

        if (foundInTet) {
            info.verticesInsideTets++;
            info.vertexToTetMap[i] = containingTets;
        } else {
            info.verticesOutsideTets++;
            info.outsideVertexIndices.push_back(i);
        }

        if (verbose_ && ((i + 1) % 100 == 0 || i == numTargetVerts - 1)) {
            std::cout << "\rProgress: " << (i + 1) << "/" << numTargetVerts
                      << " (" << (100.0f * (i + 1) / numTargetVerts) << "%)" << std::flush;
        }
    }
    if (verbose_) std::cout << std::endl;

    info.coveragePercentage = (info.totalTargetVertices > 0) ?
                                  (100.0f * info.verticesInsideTets / info.totalTargetVertices) : 0.0f;

    if (verbose_) {
        std::cout << "\n=== Coverage Results ===" << std::endl;
        std::cout << "Total target vertices: " << info.totalTargetVertices << std::endl;
        std::cout << "Vertices inside tetrahedra: " << info.verticesInsideTets
                  << " (" << info.coveragePercentage << "%)" << std::endl;
        std::cout << "Vertices outside tetrahedra: " << info.verticesOutsideTets
                  << " (" << (100.0f - info.coveragePercentage) << "%)" << std::endl;

        if (info.verticesOutsideTets > 0) {
            std::cout << "\nWARNING: " << info.verticesOutsideTets
                      << " target vertices are not covered!" << std::endl;

            std::cout << "First " << std::min(10, info.verticesOutsideTets)
                      << " uncovered vertex indices: ";
            for (int i = 0; i < std::min(10, (int)info.outsideVertexIndices.size()); i++) {
                std::cout << info.outsideVertexIndices[i] << " ";
            }
            std::cout << std::endl;

            if (!info.outsideVertexIndices.empty()) {
                glm::vec3 minPos(FLT_MAX), maxPos(-FLT_MAX);
                for (int idx : info.outsideVertexIndices) {
                    glm::vec3 v(targetVertices_[idx * 3],
                                targetVertices_[idx * 3 + 1],
                                targetVertices_[idx * 3 + 2]);
                    minPos = glm::min(minPos, v);
                    maxPos = glm::max(maxPos, v);
                }
                std::cout << "Uncovered vertices bounding box:" << std::endl;
                std::cout << "  Min: (" << minPos.x << ", " << minPos.y << ", " << minPos.z << ")" << std::endl;
                std::cout << "  Max: (" << maxPos.x << ", " << maxPos.y << ", " << maxPos.z << ")" << std::endl;
            }
        } else {
            std::cout << "\nSUCCESS: All target vertices are covered!" << std::endl;
        }

        if (!info.vertexToTetMap.empty()) {
            int maxTetsPerVertex = 0;
            float avgTetsPerVertex = 0;
            for (const auto& pair : info.vertexToTetMap) {
                int numTets = pair.second.size();
                maxTetsPerVertex = std::max(maxTetsPerVertex, numTets);
                avgTetsPerVertex += numTets;
            }
            avgTetsPerVertex /= info.vertexToTetMap.size();

            std::cout << "\n=== Statistics for covered vertices ===" << std::endl;
            std::cout << "Average tetrahedra per vertex: " << avgTetsPerVertex << std::endl;
            std::cout << "Maximum tetrahedra per vertex: " << maxTetsPerVertex << std::endl;
        }
    }

    return info;
}

// Inflate mesh
void VoxelTetrahedralizer::inflateMesh(float inflateAmount) {
    if (voxTetData.verts.empty() || voxTetData.tetSurfaceTriIds.empty()) {
        std::cerr << "Error: No mesh data available for inflation" << std::endl;
        return;
    }

    if (verbose_) {
        std::cout << "\n=== Inflating Mesh ===" << std::endl;
        std::cout << "Inflate amount: " << inflateAmount << std::endl;
    }

    std::vector<float>* targetVerts = nullptr;
    std::vector<int>* targetSurfaceTriIds = nullptr;

    if (voxTetData.isSmoothed && !voxTetData.smoothedVerts.empty()) {
        targetVerts = &voxTetData.smoothedVerts;
        targetSurfaceTriIds = &voxTetData.smoothedSurfaceTriIds;
        if (verbose_) std::cout << "Inflating smoothed mesh" << std::endl;
    } else {
        targetVerts = &voxTetData.verts;
        targetSurfaceTriIds = &voxTetData.tetSurfaceTriIds;
        if (verbose_) std::cout << "Inflating original mesh" << std::endl;
    }

    std::map<int, glm::vec3> vertexNormals;
    std::set<int> surfaceVertices;

    for (size_t i = 0; i < targetSurfaceTriIds->size(); i += 3) {
        int v0 = (*targetSurfaceTriIds)[i];
        int v1 = (*targetSurfaceTriIds)[i + 1];
        int v2 = (*targetSurfaceTriIds)[i + 2];

        glm::vec3 p0((*targetVerts)[v0 * 3], (*targetVerts)[v0 * 3 + 1], (*targetVerts)[v0 * 3 + 2]);
        glm::vec3 p1((*targetVerts)[v1 * 3], (*targetVerts)[v1 * 3 + 1], (*targetVerts)[v1 * 3 + 2]);
        glm::vec3 p2((*targetVerts)[v2 * 3], (*targetVerts)[v2 * 3 + 1], (*targetVerts)[v2 * 3 + 2]);

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

        vertexNormals[v0] += faceNormal;
        vertexNormals[v1] += faceNormal;
        vertexNormals[v2] += faceNormal;

        surfaceVertices.insert(v0);
        surfaceVertices.insert(v1);
        surfaceVertices.insert(v2);
    }

    int movedVertices = 0;
    for (const auto& pair : vertexNormals) {
        int vid = pair.first;
        glm::vec3 normal = glm::normalize(pair.second);

        (*targetVerts)[vid * 3] += normal.x * inflateAmount;
        (*targetVerts)[vid * 3 + 1] += normal.y * inflateAmount;
        (*targetVerts)[vid * 3 + 2] += normal.z * inflateAmount;
        movedVertices++;
    }

    if (verbose_) std::cout << "Moved " << movedVertices << " surface vertices" << std::endl;

    if (inflateAmount > 0) {
        adjustInternalVertices(surfaceVertices, inflateAmount * 0.5f);
    }

    if (verbose_) std::cout << "Mesh inflation complete" << std::endl;
}

// OBJ export (smooth surface)
bool VoxelTetrahedralizer::exportSmoothSurfaceToOBJ(const std::string& filename) const {
    if (!voxTetData.isSmoothed) {
        std::cerr << "No smoothed data available" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    file << "# Smoothed Surface Mesh\n";
    file << "# Vertices: " << voxTetData.smoothedVerts.size() / 3 << "\n";
    file << "# Faces: " << voxTetData.smoothedSurfaceTriIds.size() / 3 << "\n\n";

    for (size_t i = 0; i < voxTetData.smoothedVerts.size(); i += 3) {
        file << "v " << voxTetData.smoothedVerts[i] << " "
             << voxTetData.smoothedVerts[i + 1] << " "
             << voxTetData.smoothedVerts[i + 2] << "\n";
    }

    for (size_t i = 0; i < voxTetData.smoothedSurfaceTriIds.size(); i += 3) {
        file << "f " << (voxTetData.smoothedSurfaceTriIds[i] + 1) << " "
             << (voxTetData.smoothedSurfaceTriIds[i + 1] + 1) << " "
             << (voxTetData.smoothedSurfaceTriIds[i + 2] + 1) << "\n";
    }

    file.close();
    if (verbose_) std::cout << "Exported smooth surface to: " << filename << std::endl;
    return true;
}

// Export current mesh to file
bool VoxelTetrahedralizer::exportCurrentMeshToFile(const std::string& filename, bool exportSmoothedVersion) {
    if (verbose_) std::cout << "\n=== Exporting Current Mesh State ===" << std::endl;

    if (voxTetData.tetIds.empty()) {
        std::cerr << "Error: No tetrahedral mesh data available" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create output file: " << filename << std::endl;
        return false;
    }

    const std::vector<float>* vertsToExport = nullptr;
    const std::vector<int>* surfaceToExport = nullptr;

    if (exportSmoothedVersion && voxTetData.isSmoothed && !voxTetData.smoothedVerts.empty()) {
        vertsToExport = &voxTetData.smoothedVerts;
        surfaceToExport = &voxTetData.smoothedSurfaceTriIds;
        if (verbose_) std::cout << "Exporting smoothed and inflated mesh" << std::endl;
    } else {
        vertsToExport = &voxTetData.verts;
        surfaceToExport = &voxTetData.tetSurfaceTriIds;
        if (verbose_) std::cout << "Exporting original mesh structure" << std::endl;
    }

    size_t numVertices = vertsToExport->size() / 3;
    size_t numTets = voxTetData.tetIds.size() / 4;
    size_t numSurfaceTris = surfaceToExport->size() / 3;

    file << "# Tetrahedral mesh (possibly inflated)" << std::endl;
    file << "# Generated by VoxelTetrahedralizer" << std::endl;
    file << "# Grid size: " << gridSize_ << "x" << gridSize_ << "x" << gridSize_ << std::endl;
    file << "# Vertices: " << numVertices << std::endl;
    file << "# Tetrahedra: " << numTets << std::endl;
    file << "# Surface triangles: " << numSurfaceTris << std::endl;
    if (voxTetData.isSmoothed) {
        file << "# Smoothing: Applied (iterations=" << smoothingIterations_
             << ", factor=" << smoothingFactor_ << ")" << std::endl;
    }
    file << std::endl;

    file << "VERTICES" << std::endl;
    for (size_t i = 0; i < numVertices; i++) {
        file << std::fixed << std::setprecision(6)
             << (*vertsToExport)[i * 3] << " "
             << (*vertsToExport)[i * 3 + 1] << " "
             << (*vertsToExport)[i * 3 + 2] << std::endl;
    }

    file << "\nTETRAHEDRA" << std::endl;
    for (size_t i = 0; i < numTets; i++) {
        file << voxTetData.tetIds[i * 4] << " "
             << voxTetData.tetIds[i * 4 + 1] << " "
             << voxTetData.tetIds[i * 4 + 2] << " "
             << voxTetData.tetIds[i * 4 + 3] << std::endl;
    }

    file << "\nEDGES" << std::endl;
    if (!voxTetData.tetEdgeIds.empty()) {
        for (size_t i = 0; i < voxTetData.tetEdgeIds.size() / 2; i++) {
            file << voxTetData.tetEdgeIds[i * 2] << " "
                 << voxTetData.tetEdgeIds[i * 2 + 1] << std::endl;
        }
    } else {
        std::set<std::pair<int, int>> edges;
        for (size_t i = 0; i < numTets; i++) {
            int v0 = voxTetData.tetIds[i * 4];
            int v1 = voxTetData.tetIds[i * 4 + 1];
            int v2 = voxTetData.tetIds[i * 4 + 2];
            int v3 = voxTetData.tetIds[i * 4 + 3];
            edges.insert({std::min(v0, v1), std::max(v0, v1)});
            edges.insert({std::min(v0, v2), std::max(v0, v2)});
            edges.insert({std::min(v0, v3), std::max(v0, v3)});
            edges.insert({std::min(v1, v2), std::max(v1, v2)});
            edges.insert({std::min(v1, v3), std::max(v1, v3)});
            edges.insert({std::min(v2, v3), std::max(v2, v3)});
        }
        for (const auto& edge : edges) {
            file << edge.first << " " << edge.second << std::endl;
        }
    }

    file << "\nSURFACE_TRIANGLES" << std::endl;
    for (size_t i = 0; i < numSurfaceTris; i++) {
        file << (*surfaceToExport)[i * 3] << " "
             << (*surfaceToExport)[i * 3 + 1] << " "
             << (*surfaceToExport)[i * 3 + 2] << std::endl;
    }

    file.close();

    std::cout << "Exported mesh: " << filename
              << " (V:" << numVertices << " T:" << numTets
              << " E:" << (voxTetData.tetEdgeIds.empty() ? "auto" : std::to_string(voxTetData.tetEdgeIds.size() / 2))
              << " S:" << numSurfaceTris << ")" << std::endl;

    return true;
}

// Export surface to OBJ
bool VoxelTetrahedralizer::exportCurrentSurfaceToOBJ(const std::string& filename) {
    if (verbose_) std::cout << "\n=== Exporting Current Surface to OBJ ===" << std::endl;

    if (voxTetData.verts.empty()) {
        std::cerr << "Error: No mesh data available" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create output file: " << filename << std::endl;
        return false;
    }

    const std::vector<float>* verts = (voxTetData.isSmoothed && !voxTetData.smoothedVerts.empty())
                                          ? &voxTetData.smoothedVerts : &voxTetData.verts;
    const std::vector<int>* triIds = (voxTetData.isSmoothed && !voxTetData.smoothedSurfaceTriIds.empty())
                                         ? &voxTetData.smoothedSurfaceTriIds : &voxTetData.tetSurfaceTriIds;

    std::set<int> usedVertices;
    for (int idx : *triIds) {
        usedVertices.insert(idx);
    }

    std::map<int, int> vertexRemap;
    int newIndex = 1;

    file << "# Surface mesh (inflated)" << std::endl;
    file << "# Vertices: " << usedVertices.size() << std::endl;
    file << "# Faces: " << triIds->size() / 3 << std::endl;
    file << std::endl;

    for (int vid : usedVertices) {
        file << "v " << std::fixed << std::setprecision(6)
             << (*verts)[vid * 3] << " "
             << (*verts)[vid * 3 + 1] << " "
             << (*verts)[vid * 3 + 2] << std::endl;
        vertexRemap[vid] = newIndex++;
    }

    file << std::endl;

    for (size_t i = 0; i < triIds->size(); i += 3) {
        file << "f " << vertexRemap[(*triIds)[i]] << " "
             << vertexRemap[(*triIds)[i + 1]] << " "
             << vertexRemap[(*triIds)[i + 2]] << std::endl;
    }

    file.close();

    std::cout << "Exported surface: " << filename
              << " (V:" << usedVertices.size() << " F:" << triIds->size() / 3 << ")" << std::endl;

    return true;
}

MeshDataTypes::SimpleMeshData VoxelTetrahedralizer::generateDisplayMeshData() {
    MeshDataTypes::SimpleMeshData meshData;

    if (enableSmoothing_ && !voxTetData.smoothedVerts.empty()) {
        meshData.vertices = voxTetData.smoothedVerts;
        meshData.indices.reserve(voxTetData.smoothedSurfaceTriIds.size());
        for (int idx : voxTetData.smoothedSurfaceTriIds) {
            meshData.indices.push_back(static_cast<unsigned int>(idx));
        }
        if (verbose_) {
            std::cout << "Generated display mesh from smoothed data: "
                      << voxTetData.smoothedVerts.size() / 3 << " vertices, "
                      << voxTetData.smoothedSurfaceTriIds.size() / 3 << " triangles" << std::endl;
        }
    } else {
        meshData.vertices = voxTetData.verts;
        meshData.indices.reserve(voxTetData.tetSurfaceTriIds.size());
        for (int idx : voxTetData.tetSurfaceTriIds) {
            meshData.indices.push_back(static_cast<unsigned int>(idx));
        }
        if (verbose_) {
            std::cout << "Generated display mesh: "
                      << voxTetData.verts.size() / 3 << " vertices, "
                      << voxTetData.tetSurfaceTriIds.size() / 3 << " triangles" << std::endl;
        }
    }

    return meshData;
}

// === Automatic Inflation ===
void VoxelTetrahedralizer::performAutomaticInflation() {
    DebugInfo debugInfo = debugCheckTargetVertexCoverage();

    if (debugInfo.coveragePercentage >= inflationSettings_.targetCoverage) {
        std::cout << "Coverage OK (" << std::fixed << std::setprecision(1)
                  << debugInfo.coveragePercentage << "%), no inflation needed" << std::endl;
        return;
    }

    std::cout << "Inflating: " << std::fixed << std::setprecision(1)
              << debugInfo.coveragePercentage << "% -> target " << inflationSettings_.targetCoverage << "%" << std::endl;

    float baseInflateAmount = inflationSettings_.baseInflateAmount;
    if (debugInfo.coveragePercentage < 60.0f) {
        baseInflateAmount = inflationSettings_.lowCoverageInflate;
    } else if (debugInfo.coveragePercentage < 50.0f) {
        baseInflateAmount = inflationSettings_.veryLowCoverageInflate;
    }

    float currentInflateAmount = baseInflateAmount;
    int inflationAttempts = 0;
    float previousCoverage = debugInfo.coveragePercentage;

    while (debugInfo.coveragePercentage < inflationSettings_.targetCoverage &&
           inflationAttempts < inflationSettings_.maxAttempts) {

        if (verbose_) {
            std::cout << "  Attempt " << (inflationAttempts + 1)
                      << " coverage: " << debugInfo.coveragePercentage
                      << "% inflate: " << currentInflateAmount << std::endl;
        }

        inflateMesh(currentInflateAmount);
        debugInfo = debugCheckTargetVertexCoverage();

        float coverageImprovement = debugInfo.coveragePercentage - previousCoverage;

        if (coverageImprovement < inflationSettings_.slowProgressThreshold && inflationAttempts > 0) {
            currentInflateAmount *= inflationSettings_.inflateIncreaseRate;
            if (verbose_) std::cout << "  Slow progress, inflate -> " << currentInflateAmount << std::endl;
        } else if (coverageImprovement > inflationSettings_.fastProgressThreshold) {
            currentInflateAmount *= inflationSettings_.inflateDecreaseRate;
        }

        if (debugInfo.coveragePercentage > inflationSettings_.fineAdjustThreshold) {
            currentInflateAmount = std::min(currentInflateAmount, inflationSettings_.fineAdjustAmount);
        }

        previousCoverage = debugInfo.coveragePercentage;
        inflationAttempts++;

        if (debugInfo.coveragePercentage >= inflationSettings_.successThreshold) {
            std::cout << "Inflation OK: " << std::fixed << std::setprecision(1)
                      << debugInfo.coveragePercentage << "% after " << inflationAttempts << " iterations" << std::endl;
            break;
        }
    }

    if (inflationAttempts >= inflationSettings_.maxAttempts &&
        debugInfo.coveragePercentage < inflationSettings_.successThreshold) {
        std::cout << "WARNING: Max inflation attempts (" << inflationSettings_.maxAttempts
                  << ") reached. Coverage: " << debugInfo.coveragePercentage << "%" << std::endl;
    }

    if (exportCurrentMeshToFile(outputTetPath_)) {
        if (verbose_) std::cout << "Inflated mesh saved to: " << outputTetPath_ << std::endl;
    }
}

bool VoxelTetrahedralizer::isPointInTetrahedron(const glm::vec3& point, size_t tetIdx) {
    int v0 = voxTetData.tetIds[tetIdx * 4];
    int v1 = voxTetData.tetIds[tetIdx * 4 + 1];
    int v2 = voxTetData.tetIds[tetIdx * 4 + 2];
    int v3 = voxTetData.tetIds[tetIdx * 4 + 3];

    const std::vector<float>& verts = (voxTetData.isSmoothed && !voxTetData.smoothedVerts.empty())
                                          ? voxTetData.smoothedVerts : voxTetData.verts;

    glm::vec3 p0(verts[v0 * 3], verts[v0 * 3 + 1], verts[v0 * 3 + 2]);
    glm::vec3 p1(verts[v1 * 3], verts[v1 * 3 + 1], verts[v1 * 3 + 2]);
    glm::vec3 p2(verts[v2 * 3], verts[v2 * 3 + 1], verts[v2 * 3 + 2]);
    glm::vec3 p3(verts[v3 * 3], verts[v3 * 3 + 1], verts[v3 * 3 + 2]);

    return isPointInTetrahedronBarycentric(point, p0, p1, p2, p3);
}

bool VoxelTetrahedralizer::isPointInTetrahedronBarycentric(const glm::vec3& p,
                                                           const glm::vec3& a,
                                                           const glm::vec3& b,
                                                           const glm::vec3& c,
                                                           const glm::vec3& d) {
    auto volume6 = [](const glm::vec3& a, const glm::vec3& b,
                      const glm::vec3& c, const glm::vec3& d) -> float {
        return glm::dot(a - d, glm::cross(b - d, c - d));
    };

    float v0 = volume6(a, b, c, d);

    if (std::abs(v0) < 1e-10f) {
        return false;
    }

    float v1 = volume6(p, b, c, d);
    float v2 = volume6(a, p, c, d);
    float v3 = volume6(a, b, p, d);
    float v4 = volume6(a, b, c, p);

    if (v0 > 0) {
        return v1 >= -1e-6f && v2 >= -1e-6f && v3 >= -1e-6f && v4 >= -1e-6f;
    } else {
        return v1 <= 1e-6f && v2 <= 1e-6f && v3 <= 1e-6f && v4 <= 1e-6f;
    }
}

void VoxelTetrahedralizer::adjustInternalVertices(const std::set<int>& surfaceVertices, float adjustAmount) {
    std::vector<float>* targetVerts = (voxTetData.isSmoothed && !voxTetData.smoothedVerts.empty())
                                          ? &voxTetData.smoothedVerts : &voxTetData.verts;

    size_t numVertices = targetVerts->size() / 3;

    glm::vec3 meshCenter(0.0f);
    for (size_t i = 0; i < numVertices; i++) {
        meshCenter.x += (*targetVerts)[i * 3];
        meshCenter.y += (*targetVerts)[i * 3 + 1];
        meshCenter.z += (*targetVerts)[i * 3 + 2];
    }
    meshCenter /= static_cast<float>(numVertices);

    int adjustedVertices = 0;
    for (size_t i = 0; i < numVertices; i++) {
        if (surfaceVertices.find(i) != surfaceVertices.end()) {
            continue;
        }

        glm::vec3 vertex((*targetVerts)[i * 3],
                         (*targetVerts)[i * 3 + 1],
                         (*targetVerts)[i * 3 + 2]);

        glm::vec3 direction = vertex - meshCenter;
        if (glm::length(direction) > 0.0001f) {
            direction = glm::normalize(direction);
            (*targetVerts)[i * 3] += direction.x * adjustAmount;
            (*targetVerts)[i * 3 + 1] += direction.y * adjustAmount;
            (*targetVerts)[i * 3 + 2] += direction.z * adjustAmount;
            adjustedVertices++;
        }
    }

    if (verbose_ && adjustedVertices > 0) {
        std::cout << "Adjusted " << adjustedVertices << " internal vertices" << std::endl;
    }
}

void VoxelTetrahedralizer::initializeCube() {
    cubeVertices_ = {
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f, -1.0f,  1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f, -1.0f, -1.0f
    };

    cubeIndices_ = {
        4, 2, 0,  4, 6, 2,
        2, 7, 3,  2, 6, 7,
        6, 5, 7,  6, 4, 5,
        1, 7, 5,  1, 3, 7,
        0, 3, 1,  0, 2, 3,
        4, 1, 5,  4, 0, 1
    };
}

VoxelTetrahedralizer::BoundingBox VoxelTetrahedralizer::getBoundingBoxFromVertices(const std::vector<GLfloat>& vertices) {
    BoundingBox bbox;
    if (vertices.empty()) {
        bbox.min = bbox.max = bbox.size = bbox.center = glm::vec3(0.0f);
        return bbox;
    }

    bbox.min = glm::vec3(vertices[0], vertices[1], vertices[2]);
    bbox.max = bbox.min;

    for (size_t i = 0; i < vertices.size(); i += 3) {
        glm::vec3 vertex(vertices[i], vertices[i + 1], vertices[i + 2]);
        bbox.min = glm::min(bbox.min, vertex);
        bbox.max = glm::max(bbox.max, vertex);
    }

    bbox.size = bbox.max - bbox.min;
    bbox.center = (bbox.min + bbox.max) * 0.5f;
    return bbox;
}

VoxelTetrahedralizer::BoundingBox VoxelTetrahedralizer::calculateBoundingBox(const std::vector<float>& vertices,
                                                                             const std::set<int>& surfaceVertices) {
    BoundingBox bbox;
    bbox.min = glm::vec3(FLT_MAX);
    bbox.max = glm::vec3(-FLT_MAX);

    for (int vid : surfaceVertices) {
        glm::vec3 v(vertices[vid * 3], vertices[vid * 3 + 1], vertices[vid * 3 + 2]);
        bbox.min = glm::min(bbox.min, v);
        bbox.max = glm::max(bbox.max, v);
    }

    bbox.center = (bbox.min + bbox.max) * 0.5f;
    bbox.size = bbox.max - bbox.min;
    return bbox;
}

bool VoxelTetrahedralizer::loadTargetMesh() {
    MeshDataTypes::SimpleMeshData targetData = MeshDataTypes::loadOBJFile(targetPath_.c_str());

    if (targetData.vertices.empty()) {
        return false;
    }

    targetVertices_.clear();
    for (float v : targetData.vertices) {
        targetVertices_.push_back(v);
    }

    targetIndices_.clear();
    for (unsigned int idx : targetData.indices) {
        targetIndices_.push_back(idx);
    }

    return true;
}

void VoxelTetrahedralizer::prepareCube() {
    glm::vec3 center(0.0f);
    size_t vertexCount = cubeVertices_.size() / 3;

    for (size_t i = 0; i < cubeVertices_.size(); i += 3) {
        center.x += cubeVertices_[i];
        center.y += cubeVertices_[i + 1];
        center.z += cubeVertices_[i + 2];
    }
    center /= static_cast<float>(vertexCount);

    for (size_t i = 0; i < cubeVertices_.size(); i += 3) {
        cubeVertices_[i] -= center.x;
        cubeVertices_[i + 1] -= center.y;
        cubeVertices_[i + 2] -= center.z;
    }

    BoundingBox cubeBBox = getBoundingBoxFromVertices(cubeVertices_);
    float maxDim = std::max({cubeBBox.size.x, cubeBBox.size.y, cubeBBox.size.z});

    if (maxDim > 0.0001f) {
        float scale = 2.0f / maxDim;
        for (auto& v : cubeVertices_) {
            v *= scale;
        }
    }

    if (!loadTargetMesh()) {
        std::cerr << "Failed to load target mesh" << std::endl;
        return;
    }

    if (verbose_) {
        std::cout << "Target mesh loaded: " << targetVertices_.size() / 3 << " vertices, "
                  << targetIndices_.size() / 3 << " triangles" << std::endl;
    }

    BoundingBox targetBBox = getBoundingBoxFromVertices(targetVertices_);
    cubeBBox = getBoundingBoxFromVertices(cubeVertices_);

    float targetMaxDim = std::max({targetBBox.size.x, targetBBox.size.y, targetBBox.size.z});
    float cubeMaxDim = std::max({cubeBBox.size.x, cubeBBox.size.y, cubeBBox.size.z});

    float targetSize = targetMaxDim * 1.1f;
    float scaleFactor = targetSize / cubeMaxDim;

    for (auto& v : cubeVertices_) {
        v *= scaleFactor;
    }

    cubeBBox = getBoundingBoxFromVertices(cubeVertices_);
    glm::vec3 offset = targetBBox.center - cubeBBox.center;

    for (size_t i = 0; i < cubeVertices_.size(); i += 3) {
        cubeVertices_[i] += offset.x;
        cubeVertices_[i + 1] += offset.y;
        cubeVertices_[i + 2] += offset.z;
    }
}

void VoxelTetrahedralizer::initializeVoxelGrid() {
    BoundingBox bbox = getBoundingBoxFromVertices(cubeVertices_);
    gridMin_ = bbox.min;
    gridMax_ = bbox.max;
    voxelSize_ = (gridMax_ - gridMin_) / static_cast<float>(gridSize_);

    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                Voxel& v = voxels_[x][y][z];
                v.min.x = gridMin_.x + x * voxelSize_.x;
                v.min.y = gridMin_.y + y * voxelSize_.y;
                v.min.z = gridMin_.z + z * voxelSize_.z;
                v.max = v.min + voxelSize_;
                v.center = (v.min + v.max) * 0.5f;
                v.targetTriangleCount = 0;
                v.isRemoved = false;
            }
        }
    }

    if (verbose_) {
        std::cout << "Voxel grid initialized: " << gridSize_ << "x" << gridSize_ << "x" << gridSize_ << std::endl;
    }
}

bool VoxelTetrahedralizer::isPointInsideTarget(const glm::vec3& point) {
    int intersectionCount = 0;
    glm::vec3 rayDir(1.0f, 0.0001f, 0.0f);

    for (size_t i = 0; i < targetIndices_.size(); i += 3) {
        glm::vec3 v0(targetVertices_[targetIndices_[i] * 3],
                     targetVertices_[targetIndices_[i] * 3 + 1],
                     targetVertices_[targetIndices_[i] * 3 + 2]);
        glm::vec3 v1(targetVertices_[targetIndices_[i + 1] * 3],
                     targetVertices_[targetIndices_[i + 1] * 3 + 1],
                     targetVertices_[targetIndices_[i + 1] * 3 + 2]);
        glm::vec3 v2(targetVertices_[targetIndices_[i + 2] * 3],
                     targetVertices_[targetIndices_[i + 2] * 3 + 1],
                     targetVertices_[targetIndices_[i + 2] * 3 + 2]);

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(rayDir, edge2);
        float a = glm::dot(edge1, h);

        if (std::abs(a) < 0.000001f) continue;

        float f = 1.0f / a;
        glm::vec3 s = point - v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) continue;

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(rayDir, q);

        if (v < 0.0f || u + v > 1.0f) continue;

        float t = f * glm::dot(edge2, q);
        if (t > 0.000001f) {
            intersectionCount++;
        }
    }

    return (intersectionCount % 2) == 1;
}

void VoxelTetrahedralizer::checkVoxelCenters() {
    int totalVoxels = gridSize_ * gridSize_ * gridSize_;
    std::vector<bool> results(totalVoxels, false);

#pragma omp parallel for schedule(dynamic, 256)
    for (int idx = 0; idx < totalVoxels; idx++) {
        int z = idx / (gridSize_ * gridSize_);
        int y = (idx % (gridSize_ * gridSize_)) / gridSize_;
        int x = idx % gridSize_;

        results[idx] = isPointInsideTarget(voxels_[x][y][z].center);
    }

    int insideCount = 0;
    for (int idx = 0; idx < totalVoxels; idx++) {
        int z = idx / (gridSize_ * gridSize_);
        int y = (idx % (gridSize_ * gridSize_)) / gridSize_;
        int x = idx % gridSize_;

        if (results[idx]) {
            voxels_[x][y][z].targetTriangleCount = 1;
            insideCount++;
        }
    }

    if (verbose_) {
        std::cout << "Voxels inside mesh: " << insideCount << " / " << totalVoxels
                  << " (" << (100.0f * insideCount / totalVoxels) << "%)" << std::endl;
    }
}

void VoxelTetrahedralizer::carveExternalVoxels() {
    bool changed = true;
    int iterations = 0;
    int removedCount = 0;

    while (changed) {
        changed = false;
        iterations++;

        for (int x = 0; x < gridSize_; x++) {
            for (int y = 0; y < gridSize_; y++) {
                for (int z = 0; z < gridSize_; z++) {
                    if (voxels_[x][y][z].isRemoved) continue;
                    if (voxels_[x][y][z].targetTriangleCount > 0) continue;

                    bool exposed = (x == 0 || x == gridSize_-1 ||
                                    y == 0 || y == gridSize_-1 ||
                                    z == 0 || z == gridSize_-1);

                    if (!exposed) {
                        exposed = (x > 0 && voxels_[x-1][y][z].isRemoved) ||
                                  (x < gridSize_-1 && voxels_[x+1][y][z].isRemoved) ||
                                  (y > 0 && voxels_[x][y-1][z].isRemoved) ||
                                  (y < gridSize_-1 && voxels_[x][y+1][z].isRemoved) ||
                                  (z > 0 && voxels_[x][y][z-1].isRemoved) ||
                                  (z < gridSize_-1 && voxels_[x][y][z+1].isRemoved);
                    }

                    if (exposed) {
                        voxels_[x][y][z].isRemoved = true;
                        changed = true;
                        removedCount++;
                    }
                }
            }
        }
    }

    if (verbose_) {
        std::cout << "Removed " << removedCount << " external voxels in " << iterations << " iterations" << std::endl;
    }
}

void VoxelTetrahedralizer::generateTetrahedra() {
    std::vector<std::tuple<int, int, int>> voxelList;
    voxelList.reserve(gridSize_ * gridSize_ * gridSize_ / 8);

    for (int z = 0; z < gridSize_; z++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int x = 0; x < gridSize_; x++) {
                if (!voxels_[x][y][z].isRemoved) {
                    voxelList.push_back({x, y, z});
                }
            }
        }
    }

    int cubeCount = voxelList.size();
    if (verbose_) std::cout << "Processing " << cubeCount << " cubes..." << std::endl;

    std::unordered_map<size_t, int> gridVertexMap;
    std::vector<glm::vec3> vertices;
    vertices.reserve(cubeCount * 8);

    auto gridHash = [this](int x, int y, int z) -> size_t {
        return ((size_t)z * (gridSize_+1) + y) * (gridSize_+1) + x;
    };

    for (const auto& [vx, vy, vz] : voxelList) {
        for (int dz = 0; dz <= 1; dz++) {
            for (int dy = 0; dy <= 1; dy++) {
                for (int dx = 0; dx <= 1; dx++) {
                    int gx = vx + dx;
                    int gy = vy + dy;
                    int gz = vz + dz;

                    size_t hash = gridHash(gx, gy, gz);

                    if (gridVertexMap.find(hash) == gridVertexMap.end()) {
                        glm::vec3 pos = gridMin_ +
                                        glm::vec3(gx * voxelSize_.x,
                                                  gy * voxelSize_.y,
                                                  gz * voxelSize_.z);
                        gridVertexMap[hash] = vertices.size();
                        vertices.push_back(pos);
                    }
                }
            }
        }
    }

    if (verbose_) std::cout << "Generated " << vertices.size() << " vertices" << std::endl;

    std::vector<std::vector<int>> tetrahedra;
    tetrahedra.reserve(cubeCount * 5);

    for (const auto& [x, y, z] : voxelList) {
        int v[8];
        v[0] = gridVertexMap[gridHash(x, y, z)];
        v[1] = gridVertexMap[gridHash(x+1, y, z)];
        v[2] = gridVertexMap[gridHash(x+1, y+1, z)];
        v[3] = gridVertexMap[gridHash(x, y+1, z)];
        v[4] = gridVertexMap[gridHash(x, y, z+1)];
        v[5] = gridVertexMap[gridHash(x+1, y, z+1)];
        v[6] = gridVertexMap[gridHash(x+1, y+1, z+1)];
        v[7] = gridVertexMap[gridHash(x, y+1, z+1)];

        bool parity = ((x + y + z) % 2) == 0;

        if (parity) {
            tetrahedra.push_back({v[0], v[1], v[3], v[4]});
            tetrahedra.push_back({v[1], v[2], v[3], v[6]});
            tetrahedra.push_back({v[1], v[3], v[4], v[6]});
            tetrahedra.push_back({v[1], v[4], v[5], v[6]});
            tetrahedra.push_back({v[3], v[4], v[6], v[7]});
        } else {
            tetrahedra.push_back({v[0], v[1], v[2], v[5]});
            tetrahedra.push_back({v[0], v[2], v[3], v[7]});
            tetrahedra.push_back({v[0], v[4], v[5], v[7]});
            tetrahedra.push_back({v[2], v[5], v[6], v[7]});
            tetrahedra.push_back({v[0], v[2], v[5], v[7]});
        }
    }

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(tetrahedra.size()); i++) {
        auto& tet = tetrahedra[i];

        glm::vec3 v0 = vertices[tet[0]];
        glm::vec3 v1 = vertices[tet[1]];
        glm::vec3 v2 = vertices[tet[2]];
        glm::vec3 v3 = vertices[tet[3]];

        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;
        glm::vec3 e3 = v3 - v0;

        float det = glm::dot(e1, glm::cross(e2, e3));

        if (det < 0) {
            std::swap(tet[1], tet[2]);
        }
    }

    std::unordered_map<std::string, int> faceCount;
    std::unordered_map<std::string, int> faceToTetMap;

    auto faceKey = [](int a, int b, int c) -> std::string {
        std::vector<int> v = {a, b, c};
        std::sort(v.begin(), v.end());
        return std::to_string(v[0]) + "_" + std::to_string(v[1]) + "_" + std::to_string(v[2]);
    };

    for (size_t tetIdx = 0; tetIdx < tetrahedra.size(); tetIdx++) {
        const auto& tet = tetrahedra[tetIdx];

        std::vector<std::string> faces = {
            faceKey(tet[0], tet[1], tet[2]),
            faceKey(tet[0], tet[1], tet[3]),
            faceKey(tet[0], tet[2], tet[3]),
            faceKey(tet[1], tet[2], tet[3])
        };

        for (const auto& face : faces) {
            faceCount[face]++;
            if (faceToTetMap.find(face) == faceToTetMap.end()) {
                faceToTetMap[face] = tetIdx;
            }
        }
    }

    std::vector<std::vector<int>> surfaceTriangles;

    for (const auto& [faceStr, count] : faceCount) {
        if (count == 1) {
            std::vector<int> verts;
            std::stringstream ss(faceStr);
            std::string token;
            while (std::getline(ss, token, '_')) {
                verts.push_back(std::stoi(token));
            }

            int tetIdx = faceToTetMap[faceStr];

            glm::vec3 tetCenter(0.0f);
            for (int vidx : tetrahedra[tetIdx]) {
                tetCenter += vertices[vidx];
            }
            tetCenter /= 4.0f;

            glm::vec3 faceCenter = (vertices[verts[0]] +
                                    vertices[verts[1]] +
                                    vertices[verts[2]]) / 3.0f;

            glm::vec3 edge1 = vertices[verts[1]] - vertices[verts[0]];
            glm::vec3 edge2 = vertices[verts[2]] - vertices[verts[0]];
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

            glm::vec3 outward = faceCenter - tetCenter;

            if (glm::dot(normal, outward) < 0.0f) {
                std::swap(verts[1], verts[2]);
            }

            surfaceTriangles.push_back(verts);
        }
    }

    // Store to MeshData
    voxTetData.verts.clear();
    voxTetData.verts.reserve(vertices.size() * 3);
    for (const auto& v : vertices) {
        voxTetData.verts.push_back(v.x);
        voxTetData.verts.push_back(v.y);
        voxTetData.verts.push_back(v.z);
    }

    voxTetData.tetIds.clear();
    voxTetData.tetIds.reserve(tetrahedra.size() * 4);
    for (const auto& tet : tetrahedra) {
        for (int i = 0; i < 4; i++) {
            voxTetData.tetIds.push_back(tet[i]);
        }
    }

    voxTetData.tetEdgeIds.clear();
    if (gridSize_ <= 200) {
        std::set<std::pair<int, int>> edges;
        for (const auto& tet : tetrahedra) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    edges.insert({std::min(tet[i], tet[j]), std::max(tet[i], tet[j])});
                }
            }
        }

        voxTetData.tetEdgeIds.reserve(edges.size() * 2);
        for (const auto& edge : edges) {
            voxTetData.tetEdgeIds.push_back(edge.first);
            voxTetData.tetEdgeIds.push_back(edge.second);
        }
    }

    voxTetData.tetSurfaceTriIds.clear();
    voxTetData.tetSurfaceTriIds.reserve(surfaceTriangles.size() * 3);
    for (const auto& tri : surfaceTriangles) {
        voxTetData.tetSurfaceTriIds.push_back(tri[0]);
        voxTetData.tetSurfaceTriIds.push_back(tri[1]);
        voxTetData.tetSurfaceTriIds.push_back(tri[2]);
    }

    if (verbose_) {
        std::cout << "Stored " << vertices.size() << " vertices" << std::endl;
        std::cout << "Stored " << tetrahedra.size() << " tetrahedra" << std::endl;
        std::cout << "Stored " << surfaceTriangles.size() << " surface triangles" << std::endl;
    }

    if (enableSmoothing_) {
        applySmoothingToMeshData();
    }

    if (!inflationSettings_.enabled) {
        writeOutputFile(vertices, tetrahedra, surfaceTriangles);
    }
}

// === Smoothing ===

void VoxelTetrahedralizer::applySmoothingToMeshData() {
    if (verbose_) std::cout << "\n=== Applying Surface Smoothing ===" << std::endl;

    generateSmoothSurface();
    applyLaplacianSmoothing();

    voxTetData.isSmoothed = true;

    if (verbose_) {
        std::cout << "Smoothing complete: " << smoothingIterations_
                  << " iterations, factor: " << smoothingFactor_ << std::endl;
    }
}

void VoxelTetrahedralizer::generateSmoothSurface() {
    voxTetData.smoothedSurfaceTriIds.clear();

    struct Face {
        int v0, v1, v2;
        int origV0, origV1, origV2;
        int tetIndex;

        Face(int a, int b, int c, int tet) : tetIndex(tet) {
            origV0 = a; origV1 = b; origV2 = c;
            int vertices[3] = {a, b, c};
            if (vertices[0] > vertices[1]) std::swap(vertices[0], vertices[1]);
            if (vertices[1] > vertices[2]) std::swap(vertices[1], vertices[2]);
            if (vertices[0] > vertices[1]) std::swap(vertices[0], vertices[1]);
            v0 = vertices[0]; v1 = vertices[1]; v2 = vertices[2];
        }

        bool operator<(const Face& other) const {
            if (v0 != other.v0) return v0 < other.v0;
            if (v1 != other.v1) return v1 < other.v1;
            return v2 < other.v2;
        }

        bool operator==(const Face& other) const {
            return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
        }
    };

    size_t numTets = voxTetData.tetIds.size() / 4;
    std::vector<Face> allFaces;
    allFaces.reserve(numTets * 4);

    for (size_t i = 0; i < numTets; i++) {
        int v0 = voxTetData.tetIds[i * 4];
        int v1 = voxTetData.tetIds[i * 4 + 1];
        int v2 = voxTetData.tetIds[i * 4 + 2];
        int v3 = voxTetData.tetIds[i * 4 + 3];

        allFaces.emplace_back(v0, v1, v2, i);
        allFaces.emplace_back(v0, v1, v3, i);
        allFaces.emplace_back(v0, v2, v3, i);
        allFaces.emplace_back(v1, v2, v3, i);
    }

    std::sort(allFaces.begin(), allFaces.end());

    voxTetData.smoothedSurfaceTriIds.reserve(allFaces.size() / 10);

    for (size_t i = 0; i < allFaces.size(); ) {
        if (i + 1 < allFaces.size() && allFaces[i] == allFaces[i + 1]) {
            i += 2;
        } else {
            const Face& face = allFaces[i];

            glm::vec3 tetCenter(0.0f);
            for (int j = 0; j < 4; j++) {
                int vid = voxTetData.tetIds[face.tetIndex * 4 + j];
                tetCenter.x += voxTetData.verts[vid * 3];
                tetCenter.y += voxTetData.verts[vid * 3 + 1];
                tetCenter.z += voxTetData.verts[vid * 3 + 2];
            }
            tetCenter /= 4.0f;

            glm::vec3 fv0(voxTetData.verts[face.origV0 * 3], voxTetData.verts[face.origV0 * 3 + 1], voxTetData.verts[face.origV0 * 3 + 2]);
            glm::vec3 fv1(voxTetData.verts[face.origV1 * 3], voxTetData.verts[face.origV1 * 3 + 1], voxTetData.verts[face.origV1 * 3 + 2]);
            glm::vec3 fv2(voxTetData.verts[face.origV2 * 3], voxTetData.verts[face.origV2 * 3 + 1], voxTetData.verts[face.origV2 * 3 + 2]);

            glm::vec3 faceCenter = (fv0 + fv1 + fv2) / 3.0f;
            glm::vec3 edge1 = fv1 - fv0;
            glm::vec3 edge2 = fv2 - fv0;
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
            glm::vec3 outward = faceCenter - tetCenter;

            if (glm::dot(normal, outward) > 0) {
                voxTetData.smoothedSurfaceTriIds.push_back(face.origV0);
                voxTetData.smoothedSurfaceTriIds.push_back(face.origV1);
                voxTetData.smoothedSurfaceTriIds.push_back(face.origV2);
            } else {
                voxTetData.smoothedSurfaceTriIds.push_back(face.origV0);
                voxTetData.smoothedSurfaceTriIds.push_back(face.origV2);
                voxTetData.smoothedSurfaceTriIds.push_back(face.origV1);
            }

            i++;
        }
    }

    if (voxTetData.smoothedSurfaceTriIds.empty()) {
        voxTetData.smoothedSurfaceTriIds = voxTetData.tetSurfaceTriIds;
    }

    if (verbose_) {
        std::cout << "Generated smooth surface with "
                  << voxTetData.smoothedSurfaceTriIds.size() / 3 << " triangles" << std::endl;
    }
}

void VoxelTetrahedralizer::applyLaplacianSmoothing() {
    voxTetData.smoothedVerts = voxTetData.verts;

    size_t numParticles = voxTetData.verts.size() / 3;

    std::vector<std::set<int>> neighbors(numParticles);
    std::set<int> surfaceVertices;

    for (size_t i = 0; i < voxTetData.smoothedSurfaceTriIds.size(); i += 3) {
        int v0 = voxTetData.smoothedSurfaceTriIds[i];
        int v1 = voxTetData.smoothedSurfaceTriIds[i + 1];
        int v2 = voxTetData.smoothedSurfaceTriIds[i + 2];

        neighbors[v0].insert(v1); neighbors[v0].insert(v2);
        neighbors[v1].insert(v0); neighbors[v1].insert(v2);
        neighbors[v2].insert(v0); neighbors[v2].insert(v1);

        surfaceVertices.insert(v0);
        surfaceVertices.insert(v1);
        surfaceVertices.insert(v2);
    }

    if (verbose_) std::cout << "Surface vertices to smooth: " << surfaceVertices.size() << std::endl;

    for (int iter = 0; iter < smoothingIterations_; iter++) {
        std::vector<float> newVertices = voxTetData.smoothedVerts;

        for (int vid : surfaceVertices) {
            if (neighbors[vid].empty()) continue;

            glm::vec3 oldPos(voxTetData.smoothedVerts[vid * 3],
                             voxTetData.smoothedVerts[vid * 3 + 1],
                             voxTetData.smoothedVerts[vid * 3 + 2]);

            glm::vec3 avgPos(0.0f);
            for (int n : neighbors[vid]) {
                avgPos.x += voxTetData.smoothedVerts[n * 3];
                avgPos.y += voxTetData.smoothedVerts[n * 3 + 1];
                avgPos.z += voxTetData.smoothedVerts[n * 3 + 2];
            }
            avgPos /= float(neighbors[vid].size());

            glm::vec3 newPos = (1.0f - smoothingFactor_) * oldPos + smoothingFactor_ * avgPos;

            newVertices[vid * 3] = newPos.x;
            newVertices[vid * 3 + 1] = newPos.y;
            newVertices[vid * 3 + 2] = newPos.z;
        }

        voxTetData.smoothedVerts = newVertices;
    }

    if (enableSizeAdjustment_) {
        adjustMeshSize(surfaceVertices);
    }
}

void VoxelTetrahedralizer::adjustMeshSize(const std::set<int>& surfaceVertices) {
    BoundingBox originalBBox = calculateBoundingBox(voxTetData.verts, surfaceVertices);
    BoundingBox smoothedBBox = calculateBoundingBox(voxTetData.smoothedVerts, surfaceVertices);

    glm::vec3 scaleFactors;
    scaleFactors.x = (smoothedBBox.size.x > 0.001f) ? originalBBox.size.x / smoothedBBox.size.x : 1.0f;
    scaleFactors.y = (smoothedBBox.size.y > 0.001f) ? originalBBox.size.y / smoothedBBox.size.y : 1.0f;
    scaleFactors.z = (smoothedBBox.size.z > 0.001f) ? originalBBox.size.z / smoothedBBox.size.z : 1.0f;

    if (scalingMethod_ == 1) {
        float uniformScale = (scaleFactors.x + scaleFactors.y + scaleFactors.z) / 3.0f;
        for (int vid : surfaceVertices) {
            glm::vec3 vertex(voxTetData.smoothedVerts[vid * 3], voxTetData.smoothedVerts[vid * 3 + 1], voxTetData.smoothedVerts[vid * 3 + 2]);
            vertex = smoothedBBox.center + (vertex - smoothedBBox.center) * uniformScale;
            vertex = vertex - smoothedBBox.center + originalBBox.center;
            voxTetData.smoothedVerts[vid * 3] = vertex.x;
            voxTetData.smoothedVerts[vid * 3 + 1] = vertex.y;
            voxTetData.smoothedVerts[vid * 3 + 2] = vertex.z;
        }
    }
    else if (scalingMethod_ == 2) {
        for (int vid : surfaceVertices) {
            glm::vec3 vertex(voxTetData.smoothedVerts[vid * 3], voxTetData.smoothedVerts[vid * 3 + 1], voxTetData.smoothedVerts[vid * 3 + 2]);
            glm::vec3 relPos = vertex - smoothedBBox.center;
            relPos.x *= scaleFactors.x;
            relPos.y *= scaleFactors.y;
            relPos.z *= scaleFactors.z;
            vertex = originalBBox.center + relPos;
            voxTetData.smoothedVerts[vid * 3] = vertex.x;
            voxTetData.smoothedVerts[vid * 3 + 1] = vertex.y;
            voxTetData.smoothedVerts[vid * 3 + 2] = vertex.z;
        }
    }
    else {
        float maxScale = std::max({scaleFactors.x, scaleFactors.y, scaleFactors.z});
        float minScale = std::min({scaleFactors.x, scaleFactors.y, scaleFactors.z});
        float scaleDiff = maxScale - minScale;
        float uniformWeight = (scaleDiff < 0.05f) ? 1.0f : std::exp(-scaleDiff * 10.0f);
        float uniformScale = (scaleFactors.x + scaleFactors.y + scaleFactors.z) / 3.0f;

        for (int vid : surfaceVertices) {
            glm::vec3 vertex(voxTetData.smoothedVerts[vid * 3], voxTetData.smoothedVerts[vid * 3 + 1], voxTetData.smoothedVerts[vid * 3 + 2]);
            glm::vec3 relPos = vertex - smoothedBBox.center;
            glm::vec3 uniformScaled = relPos * uniformScale;
            glm::vec3 nonUniformScaled = glm::vec3(relPos.x * scaleFactors.x, relPos.y * scaleFactors.y, relPos.z * scaleFactors.z);
            relPos = uniformScaled * uniformWeight + nonUniformScaled * (1.0f - uniformWeight);
            vertex = originalBBox.center + relPos;
            voxTetData.smoothedVerts[vid * 3] = vertex.x;
            voxTetData.smoothedVerts[vid * 3 + 1] = vertex.y;
            voxTetData.smoothedVerts[vid * 3 + 2] = vertex.z;
        }
    }
}

void VoxelTetrahedralizer::writeOutputFile(const std::vector<glm::vec3>& vertices,
                                           const std::vector<std::vector<int>>& tetrahedra,
                                           const std::vector<std::vector<int>>& surfaceTriangles) {

    if (verbose_) std::cout << "Writing to file: " << outputTetPath_ << std::endl;
    std::ofstream out_file(outputTetPath_);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not create output file" << std::endl;
        return;
    }

    out_file << "# Voxel-based tetrahedral mesh\n";
    out_file << "# Grid: " << gridSize_ << "x" << gridSize_ << "x" << gridSize_ << "\n";
    out_file << "# Vertices: " << vertices.size() << "\n";
    out_file << "# Tetrahedra: " << tetrahedra.size() << "\n";
    if (enableSmoothing_) {
        out_file << "# Smoothing: Enabled (iterations=" << smoothingIterations_
                 << ", factor=" << smoothingFactor_ << ")\n";
    }
    out_file << "\n";

    out_file << "VERTICES\n";
    if (enableSmoothing_ && !voxTetData.smoothedVerts.empty()) {
        for (size_t i = 0; i < voxTetData.smoothedVerts.size() / 3; i++) {
            out_file << std::fixed << std::setprecision(6)
                     << voxTetData.smoothedVerts[i * 3] << " "
                     << voxTetData.smoothedVerts[i * 3 + 1] << " "
                     << voxTetData.smoothedVerts[i * 3 + 2] << "\n";
        }
    } else {
        for (const auto& v : vertices) {
            out_file << std::fixed << std::setprecision(6)
                     << v.x << " " << v.y << " " << v.z << "\n";
        }
    }

    out_file << "\nTETRAHEDRA\n";
    for (const auto& tet : tetrahedra) {
        out_file << tet[0] << " " << tet[1] << " " << tet[2] << " " << tet[3] << "\n";
    }

    if (gridSize_ <= 200) {
        std::set<std::pair<int, int>> edges;
        for (const auto& tet : tetrahedra) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    edges.insert({std::min(tet[i], tet[j]), std::max(tet[i], tet[j])});
                }
            }
        }
        out_file << "\nEDGES\n";
        for (const auto& edge : edges) {
            out_file << edge.first << " " << edge.second << "\n";
        }
    }

    out_file << "\nSURFACE_TRIANGLES\n";
    if (enableSmoothing_ && !voxTetData.smoothedSurfaceTriIds.empty()) {
        for (size_t i = 0; i < voxTetData.smoothedSurfaceTriIds.size() / 3; i++) {
            out_file << voxTetData.smoothedSurfaceTriIds[i * 3] << " "
                     << voxTetData.smoothedSurfaceTriIds[i * 3 + 1] << " "
                     << voxTetData.smoothedSurfaceTriIds[i * 3 + 2] << "\n";
        }
    } else {
        for (const auto& tri : surfaceTriangles) {
            out_file << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
        }
    }

    out_file.close();

    if (verbose_) {
        std::cout << "  Vertices: " << vertices.size() << std::endl;
        std::cout << "  Tetrahedra: " << tetrahedra.size() << std::endl;
        std::cout << "  Surface triangles: "
                  << (enableSmoothing_ ? voxTetData.smoothedSurfaceTriIds.size() / 3 : surfaceTriangles.size())
                  << std::endl;
    }
}
