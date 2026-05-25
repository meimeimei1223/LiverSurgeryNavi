// =============================================================================
//  MeshCleanup.h
// -----------------------------------------------------------------------------
//  Four-stage cleanup for depth-derived OBJ meshes.
//
//    1. removeDegenerateTriangles
//         Zero-area triangles (collinear / duplicated vertices). Their face
//         normals are undefined, which poisons downstream normal estimation
//         and lighting. About 0.3% of triangles in a typical K4A frame.
//
//    2. removeLongEdgeTriangles
//         "Spikes" -- thin triangles that straddle a depth discontinuity at
//         the segmentation mask boundary. Threshold is auto-computed from
//         the edge-length median, so it adapts to different capture scales.
//
//    3. keepLargestComponent
//         Isolated islands from floating depth samples that are not part of
//         the main object. Pure bookkeeping: we BFS the vertex-adjacency
//         graph and discard anything not in the largest component.
//
//    4. compactVertices (automatic, runs at the end of cleanupOBJMesh)
//         After the three steps above, some vertices may no longer be
//         referenced by any triangle. Drop them and renumber indices so
//         the cached cloud in NoOpen3DRegistration sees only live points.
//
//  Order matters: degenerate -> spike -> component -> compact. Reordering
//  changes the edge-length median used for spike detection.
//
//  This header is intentionally header-only and depends on nothing beyond
//  mCutMesh + GLM. Safe to include from main.cpp or any registration code.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>

#include <glm/glm.hpp>

#include "mCutMesh.h"

namespace Reg3DCustom {

// -----------------------------------------------------------------------------
//  Cleanup report: what was removed and why.
// -----------------------------------------------------------------------------
struct CleanupStats {
    size_t trianglesBefore          = 0;
    size_t trianglesAfter           = 0;
    size_t verticesBefore           = 0;
    size_t verticesAfter            = 0;
    int    degenerateRemoved        = 0;
    int    spikeRemoved             = 0;             // long-edge filter
    int    silhouetteRemoved        = 0;             // normal-cos filter
    int    componentsFound          = 0;
    int    componentTrianglesRemoved = 0;
    float  edgeMedian               = 0.0f;
    float  spikeThreshold           = 0.0f;
    float  silhouetteCosThreshold   = 0.0f;
};

// -----------------------------------------------------------------------------
//  Internal helpers
// -----------------------------------------------------------------------------
namespace cleanup_detail {

inline glm::vec3 vertAt(const mCutMesh& mesh, GLuint idx) {
    return glm::vec3(mesh.mVertices[idx * 3 + 0],
                     mesh.mVertices[idx * 3 + 1],
                     mesh.mVertices[idx * 3 + 2]);
}

} // namespace cleanup_detail

// -----------------------------------------------------------------------------
//  1. Remove zero-area triangles (collinear or duplicate-vertex faces).
//     Threshold is on |cross|^2 (which equals (2*area)^2) so we avoid a
//     sqrt per face. The default 1e-16 corresponds to a triangle area of
//     ~5e-9 -- safely below any real-world capture (sub-micrometre on
//     metric meshes, sub-1e-3 unit on dimensionless ones).
//
//     Earlier versions used 1e-10, which removed ~98% of a typical metric
//     OBJ because (cross)^2 for a 1mm triangle is ~5e-13. Don't tighten
//     this threshold without re-examining your scene's natural scale --
//     printEdgeLengthStats() lets you see it.
// -----------------------------------------------------------------------------
inline int removeDegenerateTriangles(mCutMesh& mesh, float areaEps = 1e-16f) {
    std::vector<GLuint> keep;
    keep.reserve(mesh.mIndices.size());
    int removed = 0;
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        GLuint a = mesh.mIndices[i    ];
        GLuint b = mesh.mIndices[i + 1];
        GLuint c = mesh.mIndices[i + 2];
        glm::vec3 va = cleanup_detail::vertAt(mesh, a);
        glm::vec3 vb = cleanup_detail::vertAt(mesh, b);
        glm::vec3 vc = cleanup_detail::vertAt(mesh, c);
        glm::vec3 n  = glm::cross(vb - va, vc - va);
        if (glm::dot(n, n) < areaEps) {
            ++removed;
            continue;
        }
        keep.push_back(a);
        keep.push_back(b);
        keep.push_back(c);
    }
    mesh.mIndices = std::move(keep);
    return removed;
}

// -----------------------------------------------------------------------------
//  2a. Compute the edge-length median (for adaptive spike threshold).
//      Uses std::nth_element (O(N)) rather than a full sort.
// -----------------------------------------------------------------------------
inline float computeEdgeMedian(const mCutMesh& mesh) {
    std::vector<float> lens;
    lens.reserve(mesh.mIndices.size());
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        glm::vec3 va = cleanup_detail::vertAt(mesh, mesh.mIndices[i    ]);
        glm::vec3 vb = cleanup_detail::vertAt(mesh, mesh.mIndices[i + 1]);
        glm::vec3 vc = cleanup_detail::vertAt(mesh, mesh.mIndices[i + 2]);
        lens.push_back(glm::length(vb - va));
        lens.push_back(glm::length(vc - vb));
        lens.push_back(glm::length(va - vc));
    }
    if (lens.empty()) return 0.0f;
    auto mid = lens.begin() + static_cast<ptrdiff_t>(lens.size() / 2);
    std::nth_element(lens.begin(), mid, lens.end());
    return *mid;
}

// -----------------------------------------------------------------------------
//  2b. Remove triangles whose longest edge exceeds `maxEdgeLen`.
//      Square the threshold once so each test is pure multiplication.
//
//      NOTE: long-edge filtering correlates with depth (far parts of the
//      surface naturally have longer edges in 3D), so it can over-remove
//      legitimate but distant geometry. For depth-derived OBJs we
//      generally prefer removeSilhouetteFaces() instead -- it targets
//      the silhouette spikes directly via face orientation, independent
//      of distance.
// -----------------------------------------------------------------------------
inline int removeLongEdgeTriangles(mCutMesh& mesh, float maxEdgeLen) {
    if (maxEdgeLen <= 0.0f) return 0;
    const float maxSq = maxEdgeLen * maxEdgeLen;
    std::vector<GLuint> keep;
    keep.reserve(mesh.mIndices.size());
    int removed = 0;
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        GLuint a = mesh.mIndices[i    ];
        GLuint b = mesh.mIndices[i + 1];
        GLuint c = mesh.mIndices[i + 2];
        glm::vec3 va = cleanup_detail::vertAt(mesh, a);
        glm::vec3 vb = cleanup_detail::vertAt(mesh, b);
        glm::vec3 vc = cleanup_detail::vertAt(mesh, c);
        glm::vec3 e0 = vb - va;
        glm::vec3 e1 = vc - vb;
        glm::vec3 e2 = va - vc;
        float m = std::max({glm::dot(e0, e0),
                            glm::dot(e1, e1),
                            glm::dot(e2, e2)});
        if (m > maxSq) {
            ++removed;
            continue;
        }
        keep.push_back(a);
        keep.push_back(b);
        keep.push_back(c);
    }
    mesh.mIndices = std::move(keep);
    return removed;
}

// -----------------------------------------------------------------------------
//  2c. Remove silhouette / grazing-incidence faces.
//      For each triangle, compute |dot(face_normal, view_dir)| where
//      view_dir = (centroid - cameraOrigin). A small absolute cosine
//      means the face is nearly parallel to the line of sight, which
//      is the geometric definition of a silhouette spike from a depth-
//      sensor reconstruction.
//
//      Assumes cameraOrigin = (0,0,0) (camera-space OBJ). Pass a
//      different origin if your OBJ is in a different frame.
//
//      Recommended threshold: 0.15..0.25. Lower = more aggressive.
// -----------------------------------------------------------------------------
inline int removeSilhouetteFaces(mCutMesh& mesh,
                                 float cosThreshold = 0.20f,
                                 const glm::vec3& cameraOrigin =
                                     glm::vec3(0.0f, 0.0f, 0.0f))
{
    std::vector<GLuint> keep;
    keep.reserve(mesh.mIndices.size());
    int removed = 0;
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        GLuint a = mesh.mIndices[i    ];
        GLuint b = mesh.mIndices[i + 1];
        GLuint c = mesh.mIndices[i + 2];
        glm::vec3 va = cleanup_detail::vertAt(mesh, a);
        glm::vec3 vb = cleanup_detail::vertAt(mesh, b);
        glm::vec3 vc = cleanup_detail::vertAt(mesh, c);
        glm::vec3 cross = glm::cross(vb - va, vc - va);
        float clen = glm::length(cross);
        if (clen < 1e-12f) {
            // Degenerate face -- normal undefined, drop it.
            ++removed;
            continue;
        }
        glm::vec3 normal   = cross / clen;
        glm::vec3 centroid = (va + vb + vc) / 3.0f;
        glm::vec3 viewDir  = centroid - cameraOrigin;
        float vlen = glm::length(viewDir);
        if (vlen < 1e-6f) {
            keep.push_back(a); keep.push_back(b); keep.push_back(c);
            continue;
        }
        viewDir /= vlen;
        float cosAng = std::abs(glm::dot(normal, viewDir));
        if (cosAng < cosThreshold) {
            ++removed;
            continue;
        }
        keep.push_back(a); keep.push_back(b); keep.push_back(c);
    }
    mesh.mIndices = std::move(keep);
    return removed;
}

// -----------------------------------------------------------------------------
//  3. Keep only the largest connected component.
//     Graph: vertices are nodes, triangle edges are undirected links.
//     Uses iterative DFS on a pre-allocated stack (no recursion, no queue
//     allocations inside the loop) for cache friendliness on ~100k vertices.
// -----------------------------------------------------------------------------
// inline int keepLargestComponent(mCutMesh& mesh,
//                                  int& numComponentsOut) {
//     const size_t nV = mesh.mVertices.size() / 3;
//     numComponentsOut = 0;
//     if (nV == 0 || mesh.mIndices.empty()) return 0;

//     // Build adjacency: for each vertex, the set of neighbor vertex indices.
//     // Duplicates are harmless for BFS/DFS correctness.
//     std::vector<std::vector<GLuint>> adj(nV);
//     for (auto& v : adj) v.reserve(6);   // typical mesh valence
//     for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
//         GLuint a = mesh.mIndices[i    ];
//         GLuint b = mesh.mIndices[i + 1];
//         GLuint c = mesh.mIndices[i + 2];
//         if (a >= nV || b >= nV || c >= nV) continue;
//         adj[a].push_back(b); adj[a].push_back(c);
//         adj[b].push_back(a); adj[b].push_back(c);
//         adj[c].push_back(a); adj[c].push_back(b);
//     }

//     std::vector<int> compId(nV, -1);
//     std::vector<int> compSize;
//     std::vector<GLuint> stack;
//     stack.reserve(nV);
//     int curComp = 0;

//     for (size_t start = 0; start < nV; ++start) {
//         if (compId[start] != -1) continue;
//         if (adj[start].empty()) continue;       // orphan, leave unlabeled

//         stack.clear();
//         stack.push_back(static_cast<GLuint>(start));
//         compId[start] = curComp;
//         int count = 0;

//         while (!stack.empty()) {
//             GLuint v = stack.back();
//             stack.pop_back();
//             ++count;
//             for (GLuint nb : adj[v]) {
//                 if (compId[nb] == -1) {
//                     compId[nb] = curComp;
//                     stack.push_back(nb);
//                 }
//             }
//         }
//         compSize.push_back(count);
//         ++curComp;
//     }

//     numComponentsOut = curComp;
//     if (curComp <= 1) return 0;

//     // Find the largest component's ID.
//     int largestId   = 0;
//     int largestSize = compSize[0];
//     for (int i = 1; i < curComp; ++i) {
//         if (compSize[i] > largestSize) {
//             largestSize = compSize[i];
//             largestId   = i;
//         }
//     }

//     // Drop triangles any of whose vertices are outside the largest component.
//     std::vector<GLuint> keep;
//     keep.reserve(mesh.mIndices.size());
//     int removed = 0;
//     for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
//         GLuint a = mesh.mIndices[i    ];
//         GLuint b = mesh.mIndices[i + 1];
//         GLuint c = mesh.mIndices[i + 2];
//         if (compId[a] == largestId &&
//             compId[b] == largestId &&
//             compId[c] == largestId) {
//             keep.push_back(a);
//             keep.push_back(b);
//             keep.push_back(c);
//         } else {
//             ++removed;
//         }
//     }
//     mesh.mIndices = std::move(keep);
//     return removed;
// }


// -----------------------------------------------------------------------------
//  3'. Keep components whose size is >= minRatio * (largest component size).
//      Replacement for keepLargestComponent that preserves multiple
//      legitimate parts of the scene (e.g. detached liver lobes, separate
//      organs visible through the surgical window) while still discarding
//      stray speckles.
//
//      minRatio guidance:
//          0.001 - 0.01  : keep almost everything except 1-2 vert specks
//          0.01  - 0.05  : balanced, drops obvious noise islands
//          0.10+         : aggressive, close to keepLargestComponent
// -----------------------------------------------------------------------------
inline int keepLargeComponents(mCutMesh& mesh,
                               int& numComponentsOut,
                               int& numComponentsKeptOut,
                               float minRatio = 0.005f)
{
    const size_t nV = mesh.mVertices.size() / 3;
    numComponentsOut    = 0;
    numComponentsKeptOut = 0;
    if (nV == 0 || mesh.mIndices.empty()) return 0;

    std::vector<std::vector<GLuint>> adj(nV);
    for (auto& v : adj) v.reserve(6);
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        GLuint a = mesh.mIndices[i    ];
        GLuint b = mesh.mIndices[i + 1];
        GLuint c = mesh.mIndices[i + 2];
        if (a >= nV || b >= nV || c >= nV) continue;
        adj[a].push_back(b); adj[a].push_back(c);
        adj[b].push_back(a); adj[b].push_back(c);
        adj[c].push_back(a); adj[c].push_back(b);
    }

    std::vector<int> compId(nV, -1);
    std::vector<int> compSize;
    std::vector<GLuint> stack;
    stack.reserve(nV);
    int curComp = 0;

    for (size_t start = 0; start < nV; ++start) {
        if (compId[start] != -1) continue;
        if (adj[start].empty()) continue;
        stack.clear();
        stack.push_back(static_cast<GLuint>(start));
        compId[start] = curComp;
        int count = 0;
        while (!stack.empty()) {
            GLuint v = stack.back();
            stack.pop_back();
            ++count;
            for (GLuint nb : adj[v]) {
                if (compId[nb] == -1) {
                    compId[nb] = curComp;
                    stack.push_back(nb);
                }
            }
        }
        compSize.push_back(count);
        ++curComp;
    }

    numComponentsOut = curComp;
    if (curComp == 0) return 0;

    // Find largest size, then build keep-set by ratio.
    int largestSize = 0;
    for (int s : compSize) if (s > largestSize) largestSize = s;
    const int sizeThreshold =
        std::max(1, (int)std::ceil((double)largestSize * (double)minRatio));

    std::vector<uint8_t> keepComp(curComp, 0);
    int kept = 0;
    for (int i = 0; i < curComp; ++i) {
        if (compSize[i] >= sizeThreshold) {
            keepComp[i] = 1;
            ++kept;
        }
    }
    numComponentsKeptOut = kept;

    // Log the size distribution so the user can tune minRatio.
    std::vector<int> sortedSizes = compSize;
    std::sort(sortedSizes.begin(), sortedSizes.end(), std::greater<int>());
    std::cout << "[MeshCleanup] component sizes (top 8 of " << curComp << "): ";
    for (int i = 0; i < (int)std::min<size_t>(8, sortedSizes.size()); ++i) {
        std::cout << sortedSizes[i] << " ";
    }
    std::cout << " threshold=" << sizeThreshold
              << " (=" << minRatio << " x largest " << largestSize << ")"
              << std::endl;

    if (kept == curComp) return 0;   // nothing to drop

    std::vector<GLuint> keep;
    keep.reserve(mesh.mIndices.size());
    int removed = 0;
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        GLuint a = mesh.mIndices[i    ];
        GLuint b = mesh.mIndices[i + 1];
        GLuint c = mesh.mIndices[i + 2];
        if (compId[a] >= 0 && keepComp[compId[a]] &&
            compId[b] >= 0 && keepComp[compId[b]] &&
            compId[c] >= 0 && keepComp[compId[c]]) {
            keep.push_back(a);
            keep.push_back(b);
            keep.push_back(c);
        } else {
            ++removed;
        }
    }
    mesh.mIndices = std::move(keep);
    return removed;
}

// -----------------------------------------------------------------------------
//  4. Drop vertices no longer referenced by any triangle; renumber indices.
//     Also compacts mNormals and mTexCoords if they are per-vertex.
// -----------------------------------------------------------------------------
inline void compactVertices(mCutMesh& mesh) {
    const size_t nV = mesh.mVertices.size() / 3;
    if (nV == 0) return;

    std::vector<uint8_t> used(nV, 0);
    for (GLuint idx : mesh.mIndices) {
        if (idx < nV) used[idx] = 1;
    }

    std::vector<GLuint> oldToNew(nV, UINT32_MAX);
    GLuint newIdx = 0;
    for (size_t v = 0; v < nV; ++v) {
        if (used[v]) oldToNew[v] = newIdx++;
    }

    if (newIdx == nV) return;   // nothing to compact, all live

    // Compact vertices.
    std::vector<GLfloat> newV(newIdx * 3);
    for (size_t v = 0; v < nV; ++v) {
        if (!used[v]) continue;
        GLuint ni = oldToNew[v];
        newV[ni * 3 + 0] = mesh.mVertices[v * 3 + 0];
        newV[ni * 3 + 1] = mesh.mVertices[v * 3 + 1];
        newV[ni * 3 + 2] = mesh.mVertices[v * 3 + 2];
    }
    mesh.mVertices = std::move(newV);

    // Compact per-vertex normals if they match.
    if (mesh.mNormals.size() / 3 == nV) {
        std::vector<GLfloat> newN(newIdx * 3);
        for (size_t v = 0; v < nV; ++v) {
            if (!used[v]) continue;
            GLuint ni = oldToNew[v];
            newN[ni * 3 + 0] = mesh.mNormals[v * 3 + 0];
            newN[ni * 3 + 1] = mesh.mNormals[v * 3 + 1];
            newN[ni * 3 + 2] = mesh.mNormals[v * 3 + 2];
        }
        mesh.mNormals = std::move(newN);
    }

    // Compact per-vertex tex coords if they match.
    if (mesh.mTexCoords.size() / 2 == nV) {
        std::vector<GLfloat> newT(newIdx * 2);
        for (size_t v = 0; v < nV; ++v) {
            if (!used[v]) continue;
            GLuint ni = oldToNew[v];
            newT[ni * 2 + 0] = mesh.mTexCoords[v * 2 + 0];
            newT[ni * 2 + 1] = mesh.mTexCoords[v * 2 + 1];
        }
        mesh.mTexCoords = std::move(newT);
    }

    // Renumber indices.
    for (auto& idx : mesh.mIndices) idx = oldToNew[idx];
}

// -----------------------------------------------------------------------------
//  One-call entry point: degenerate -> silhouette -> component -> compact.
//  Logs a summary. Returns stats for programmatic inspection.
//
//  silhouetteCosThreshold guidance:
//    0.10 - very conservative, only the most extreme grazing faces
//    0.15 - moderate
//    0.20 - default, around 4-5% of triangles on a typical depth OBJ
//    0.30 - aggressive, can clip legitimate side-facing geometry
//
//  longEdgeMultiplier:
//    <= 0  - disable long-edge filter (default; silhouette filter is
//            preferred since it doesn't conflate distance with badness)
//    >  0  - additionally remove triangles whose longest edge exceeds
//            (median edge) * longEdgeMultiplier. Use only if specific
//            spikes survive the silhouette filter.
// -----------------------------------------------------------------------------
inline CleanupStats cleanupOBJMesh(mCutMesh& mesh,
                                   float silhouetteCosThreshold = 0.20f,
                                   float longEdgeMultiplier      = 0.0f,
                                   float degenerateAreaEps       = 1e-16f)
{
    CleanupStats s;
    s.trianglesBefore = mesh.mIndices.size() / 3;
    s.verticesBefore  = mesh.mVertices.size() / 3;

    s.degenerateRemoved = removeDegenerateTriangles(mesh, degenerateAreaEps);

    // --- Primary spike removal: silhouette / grazing-incidence faces.
    s.silhouetteCosThreshold = silhouetteCosThreshold;
    s.silhouetteRemoved      = removeSilhouetteFaces(mesh,
                                                     silhouetteCosThreshold);

    // --- Optional secondary: long-edge filter (off by default).
    if (longEdgeMultiplier > 0.0f) {
        s.edgeMedian     = computeEdgeMedian(mesh);
        s.spikeThreshold = s.edgeMedian * longEdgeMultiplier;
        s.spikeRemoved   = removeLongEdgeTriangles(mesh, s.spikeThreshold);
    }

    int ncomp = 0, nkept = 0;
    //s.componentTrianglesRemoved = keepLargestComponent(mesh, ncomp);
    s.componentTrianglesRemoved = keepLargeComponents(mesh, ncomp, nkept,
                                                      /*minRatio=*/0.3f);
    s.componentsFound           = ncomp;

    compactVertices(mesh);

    s.trianglesAfter = mesh.mIndices.size() / 3;
    s.verticesAfter  = mesh.mVertices.size() / 3;

    // -------- Log ---------------------------------------------------------
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[MeshCleanup] "
              << "triangles " << s.trianglesBefore << " -> " << s.trianglesAfter
              << "  (" << (100.0 * (double)s.trianglesAfter
                           / (double)std::max<size_t>(1, s.trianglesBefore))
              << "% kept),  "
              << "vertices " << s.verticesBefore  << " -> " << s.verticesAfter
              << std::endl;
    std::cout << "[MeshCleanup]   degenerate    : " << s.degenerateRemoved
              << "  (area<" << degenerateAreaEps << ")" << std::endl;
    std::cout << "[MeshCleanup]   silhouette    : " << s.silhouetteRemoved
              << "  (|cos(n,view)| < " << s.silhouetteCosThreshold << ")"
              << std::endl;
    if (longEdgeMultiplier > 0.0f) {
        std::cout << "[MeshCleanup]   long-edge     : " << s.spikeRemoved
                  << "  (edge>" << s.spikeThreshold
                  << " = median " << s.edgeMedian
                  << " x" << longEdgeMultiplier << ")" << std::endl;
    }
    std::cout << "[MeshCleanup]   isolated comp : " << s.componentTrianglesRemoved
              << "  (" << s.componentsFound << " component"
              << (s.componentsFound == 1 ? "" : "s") << " found)" << std::endl;

    if (s.trianglesAfter < s.trianglesBefore * 4 / 5) {
        std::cerr << "[MeshCleanup] WARNING: "
                  << (100 - 100 * s.trianglesAfter
                      / std::max<size_t>(1, s.trianglesBefore))
                  << "% of triangles removed. "
                  << "Consider relaxing thresholds." << std::endl;
    }
    return s;
}

// -----------------------------------------------------------------------------
//  Diagnostic: edge-length distribution (prints percentiles).
//  Call this BEFORE cleanup to choose a sensible threshold multiplier.
// -----------------------------------------------------------------------------
inline void printEdgeLengthStats(const mCutMesh& mesh,
                                 const std::string& label = "edges") {
    std::vector<float> lens;
    lens.reserve(mesh.mIndices.size());
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        glm::vec3 va = cleanup_detail::vertAt(mesh, mesh.mIndices[i    ]);
        glm::vec3 vb = cleanup_detail::vertAt(mesh, mesh.mIndices[i + 1]);
        glm::vec3 vc = cleanup_detail::vertAt(mesh, mesh.mIndices[i + 2]);
        lens.push_back(glm::length(vb - va));
        lens.push_back(glm::length(vc - vb));
        lens.push_back(glm::length(va - vc));
    }
    if (lens.empty()) {
        std::cout << "[EdgeStats/" << label << "] empty" << std::endl;
        return;
    }
    std::sort(lens.begin(), lens.end());
    const size_t n = lens.size();
    auto pct = [&](size_t num, size_t den) -> float {
        return lens[std::min(n - 1, n * num / den)];
    };
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[EdgeStats/" << label << "] " << n << " edges:"
              << "  min="    << lens.front()
              << "  p5="     << pct( 5, 100)
              << "  p50="    << pct(50, 100)
              << "  p95="    << pct(95, 100)
              << "  p99="    << pct(99, 100)
              << "  max="    << lens.back()
              << std::endl;
}

} // namespace Reg3DCustom
