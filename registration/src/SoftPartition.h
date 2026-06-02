#pragma once
// =============================================================================
// SoftPartition.h  —  Phase U-1: Umeyama-Seeded Soft Partition
// -----------------------------------------------------------------------------
// Header-only library. Given >=3 (typically 5) Umeyama anchor correspondences,
// build a per-vertex probability distribution over the anchor groups for both
// the target mesh (screenMesh) and the source mesh (liver), via geodesic
// flood-fill (per-anchor Dijkstra on the mesh edge graph).
//
//   - Target side: softmax over -dist^2/(2 sigma^2)  -> partition of unity
//                  (every visible vertex belongs somewhere; no "none").
//   - Source side: non-normalised Gaussian + baseline "none" mass
//                  (distant vertices fall into "none" and are excluded).
//
// Coordinate frame note (confirmed against the existing pipeline):
//   computeSourcePartition / computeTargetPartition are run AFTER the Umeyama
//   registration has already scaled the liver mesh into the same world frame
//   as the target (applyRegistrationToMesh writes world coords in-place).
//   So source and target geodesic distances are in the SAME units, and the
//   per-group radii do NOT need the s_ume correction.  The s_ume argument is
//   kept in the API for diagnostics / future local-frame variants: callers in
//   U-1 pass s_ume = 1.0f, which makes r_src_i == r_tgt_i.
//
// Depends only on mCutMesh + glm + std.  No global state lives here.
// =============================================================================

#include <vector>
#include <array>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "mCutMesh.h"

namespace SoftPartition {

// ---- compile-time constants -------------------------------------------------
constexpr int   MAX_GROUPS = 8;
constexpr int   MIN_GROUPS = 3;
constexpr int   NONE_IDX   = MAX_GROUPS;            // index 8 in the probs array
constexpr float kInf       = std::numeric_limits<float>::max();
constexpr float kUnreachableLogit = -1e9f;          // softmax -> ~0 for these

// ============================================================================
//  mCutMesh access helpers
//  mCutMesh stores a FLAT GLfloat array (3 floats == 1 vertex). Never read
//  m.mVertices[i] as a glm::vec3 directly; always go through vertexAt().
// ============================================================================
inline int vertexCount(const mCutMesh& m) {
    return static_cast<int>(m.mVertices.size() / 3);
}

inline glm::vec3 vertexAt(const mCutMesh& m, int vi) {
    const size_t b = static_cast<size_t>(vi) * 3;
    return glm::vec3(m.mVertices[b], m.mVertices[b + 1], m.mVertices[b + 2]);
}

inline int triangleCount(const mCutMesh& m) {
    return static_cast<int>(m.mIndices.size() / 3);
}

// ============================================================================
//  Data structures
// ============================================================================

// One set of anchor correspondences captured from a completed Umeyama pass.
//   srcVertIdx[i]  : vertex index into the source (liver) mesh  (idx-stable)
//   srcLocal[i]    : source point in mesh-local coords (pre-registration)
//   tgtWorld[i]    : target point in world coords (fixed)
struct AnchorSet {
    std::vector<int>       srcVertIdx;
    std::vector<glm::vec3> srcLocal;
    std::vector<glm::vec3> tgtWorld;

    int count() const { return static_cast<int>(tgtWorld.size()); }

    bool valid() const {
        const int n = count();
        return n >= MIN_GROUPS && n <= MAX_GROUPS &&
               static_cast<int>(srcVertIdx.size()) == n &&
               static_cast<int>(srcLocal.size())   == n;
    }

    void clear() {
        srcVertIdx.clear();
        srcLocal.clear();
        tgtWorld.clear();
    }
};

// Per-vertex probability field. probs[v][g] for g in [0, numGroups), and
// probs[v][NONE_IDX] for the "none" mass (always 0 on the target side).
struct PartitionField {
    std::vector<std::array<float, MAX_GROUPS + 1>> probs;
    int  numGroups = 0;
    bool valid     = false;

    void clear() {
        probs.clear();
        numGroups = 0;
        valid     = false;
    }
};

struct PartitionParams {
    bool  autoSigma             = true;   // sigma_tgt from anchor spacing
    float sigmaTgtAutoFactor    = 0.5f;   // sigma = minPairwiseDist * factor
    float sigmaTgtManual        = 0.05f;  // used when autoSigma == false
    float baselineNone          = 0.368f; // source "none" mass (exp(-1))
    float groupRadiusPercentile = 0.80f;  // r_tgt_i percentile of in-group dist
};

// Phase U-2: soft-weighted ICP parameters.
struct SoftICPParams {
    int   maxIters       = 30;      // ICP iterations
    float maxCorrDistFac = 0.05f;   // reject NN farther than this * sceneDiag
    float noneSkip       = 0.70f;   // skip source verts with p_none > this
    float minWeight      = 0.02f;   // skip correspondences with soft weight < this
    float tikhonov       = 1e-4f;   // Levenberg-style damping on JTJ diagonal
    float convergeEps    = 1e-6f;   // stop when |delta_x| < this
    // Debug / large-scale: voxel-downsample the target cloud before NN.
    // 1920x1080 dense depth clouds can exceed 300k points; downsampling keeps
    // the KD-tree and per-iter NN cheap. voxel size = voxelFac * sceneDiag.
    bool  downsampleTarget = false;
    float voxelFac         = 0.01f;
    // Ablation control: when true, every accepted correspondence gets weight 1
    // (plain point-to-plane ICP) instead of the soft probability weight. Lets
    // you A/B the soft partition's contribution from the same Umeyama init.
    bool  uniformWeight    = false;
};

// Undirected adjacency list over mesh vertices, built from triangle indices.
struct MeshAdj {
    std::vector<std::vector<int>> nbr;
    void clear()       { nbr.clear(); }
    bool valid() const { return !nbr.empty(); }
    int  size() const  { return static_cast<int>(nbr.size()); }
};

// ============================================================================
//  Graph build + geodesic Dijkstra
// ============================================================================

// Build an undirected, de-duplicated adjacency list from the mesh triangles.
inline void buildMeshAdj(const mCutMesh& mesh, MeshAdj& out) {
    out.clear();
    const int nV = vertexCount(mesh);
    if (nV <= 0 || mesh.mIndices.empty()) return;

    out.nbr.assign(nV, {});
    const auto& idx = mesh.mIndices;
    auto addEdge = [&](int a, int b) {
        if (a < 0 || b < 0 || a >= nV || b >= nV || a == b) return;
        out.nbr[a].push_back(b);
        out.nbr[b].push_back(a);
    };
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        const int a = static_cast<int>(idx[t]);
        const int b = static_cast<int>(idx[t + 1]);
        const int c = static_cast<int>(idx[t + 2]);
        addEdge(a, b);
        addEdge(b, c);
        addEdge(c, a);
    }
    // de-duplicate neighbours per vertex
    for (auto& v : out.nbr) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
}

// Brute-force nearest vertex (O(N)); fine for liver ~10k and acceptable as a
// one-shot for the target mesh during a manual Recompute.
inline int nearestVertexIdx(const mCutMesh& mesh, const glm::vec3& worldPos) {
    const int nV = vertexCount(mesh);
    int   best  = -1;
    float bestD = kInf;
    for (int v = 0; v < nV; ++v) {
        const glm::vec3 p = vertexAt(mesh, v);
        const float d = glm::dot(p - worldPos, p - worldPos);
        if (d < bestD) { bestD = d; best = v; }
    }
    return best;
}

// One Dijkstra per anchor source vertex. distOut[i][v] = geodesic distance
// from anchor i to vertex v (edge weights = Euclidean vertex distances).
// Unreachable vertices stay at kInf.
inline void perAnchorDijkstra(const mCutMesh& mesh, const MeshAdj& adj,
                              const std::vector<int>& anchorVertIdx,
                              std::vector<std::vector<float>>& distOut) {
    const int nV = vertexCount(mesh);
    const int nA = static_cast<int>(anchorVertIdx.size());
    distOut.assign(nA, std::vector<float>(nV, kInf));
    if (!adj.valid() || adj.size() != nV) return;

    using QN = std::pair<float, int>;             // (dist, vertex)
    for (int i = 0; i < nA; ++i) {
        const int src = anchorVertIdx[i];
        if (src < 0 || src >= nV) continue;
        std::vector<float>& dist = distOut[i];
        std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;
        dist[src] = 0.0f;
        pq.push({0.0f, src});
        while (!pq.empty()) {
            const auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;             // stale entry
            const glm::vec3 pu = vertexAt(mesh, u);
            for (int w : adj.nbr[u]) {
                const glm::vec3 pw = vertexAt(mesh, w);
                const float nd = d + glm::length(pw - pu);
                if (nd < dist[w]) {
                    dist[w] = nd;
                    pq.push({nd, w});
                }
            }
        }
    }
}

// ============================================================================
//  Sigma / scale helpers
// ============================================================================

// sigma_tgt = (min pairwise distance among target anchors) * factor
inline float computeSigmaTgtAuto(const std::vector<glm::vec3>& tgtAnchors,
                                 float factor) {
    const int n = static_cast<int>(tgtAnchors.size());
    float minD = kInf;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            minD = std::min(minD, glm::length(tgtAnchors[i] - tgtAnchors[j]));
    if (minD == kInf || minD <= 0.0f) minD = 1.0f;  // degenerate guard
    return minD * factor;
}

// Umeyama mat4 is T = s*R + t baked in; the scale is the length of the upper
// 3 components of its first column.
inline float extractScaleFromUmeyama(const glm::mat4& T) {
    return glm::length(glm::vec3(T[0]));
}

// ============================================================================
//  Target partition  (softmax, partition of unity)
// ============================================================================
inline bool computeTargetPartition(const mCutMesh& screenMesh,
                                    const AnchorSet& anchors,
                                    const PartitionParams& params,
                                    PartitionField& fieldOut,
                                    std::vector<float>& groupRadiiOut) {
    fieldOut.clear();
    groupRadiiOut.clear();
    if (!anchors.valid()) return false;

    const int nV = vertexCount(screenMesh);
    if (nV <= 0 || screenMesh.mIndices.empty()) return false;

    const int G = anchors.count();

    // 1) adjacency
    MeshAdj adj;
    buildMeshAdj(screenMesh, adj);
    if (!adj.valid()) return false;

    // 2) snap each target anchor to its nearest screenMesh vertex
    std::vector<int> anchorVert(G, -1);
    for (int i = 0; i < G; ++i)
        anchorVert[i] = nearestVertexIdx(screenMesh, anchors.tgtWorld[i]);

    // 3) geodesic distance fields
    std::vector<std::vector<float>> dist;          // [group][vertex]
    perAnchorDijkstra(screenMesh, adj, anchorVert, dist);

    // 4) sigma
    const float sigma = params.autoSigma
        ? computeSigmaTgtAuto(anchors.tgtWorld, params.sigmaTgtAutoFactor)
        : params.sigmaTgtManual;
    const float twoSig2 = 2.0f * sigma * sigma + 1e-12f;

    // 5) softmax per vertex
    fieldOut.probs.assign(nV, {});
    fieldOut.numGroups = G;
    for (int v = 0; v < nV; ++v) {
        std::array<float, MAX_GROUPS + 1>& p = fieldOut.probs[v];
        p.fill(0.0f);

        float logit[MAX_GROUPS];
        float mx = -kInf;
        for (int i = 0; i < G; ++i) {
            const float d = dist[i][v];
            logit[i] = (d >= kInf) ? kUnreachableLogit
                                   : -(d * d) / twoSig2;
            mx = std::max(mx, logit[i]);
        }
        float Z = 0.0f;
        for (int i = 0; i < G; ++i) {
            logit[i] = std::exp(logit[i] - mx);     // stable softmax
            Z += logit[i];
        }
        if (Z <= 0.0f) Z = 1.0f;
        for (int i = 0; i < G; ++i) p[i] = logit[i] / Z;
        p[NONE_IDX] = 0.0f;                          // target has no "none"
    }

    // 6) per-group radius = percentile of in-group geodesic distance
    groupRadiiOut.assign(G, 0.0f);
    const float pct = glm::clamp(params.groupRadiusPercentile, 0.05f, 0.99f);
    for (int i = 0; i < G; ++i) {
        std::vector<float> ds;
        ds.reserve(nV / 4);
        for (int v = 0; v < nV; ++v) {
            if (fieldOut.probs[v][i] > 0.1f && dist[i][v] < kInf)
                ds.push_back(dist[i][v]);
        }
        if (ds.empty()) { groupRadiiOut[i] = sigma; continue; }
        std::sort(ds.begin(), ds.end());
        int k = static_cast<int>(pct * (ds.size() - 1));
        k = glm::clamp(k, 0, static_cast<int>(ds.size()) - 1);
        groupRadiiOut[i] = ds[k];
    }

    fieldOut.valid = true;
    return true;
}

// ============================================================================
//  Source partition  (non-normalised Gaussian + baseline "none")
// ============================================================================
inline bool computeSourcePartition(const mCutMesh& liverMesh,
                                   const AnchorSet& anchors,
                                   const std::vector<float>& groupRadiiTgt,
                                   float s_ume,
                                   const PartitionParams& params,
                                   PartitionField& fieldOut) {
    fieldOut.clear();
    if (!anchors.valid()) return false;

    const int nV = vertexCount(liverMesh);
    if (nV <= 0 || liverMesh.mIndices.empty()) return false;

    const int G = anchors.count();
    if (static_cast<int>(groupRadiiTgt.size()) != G) return false;

    if (s_ume <= 1e-6f) s_ume = 1.0f;               // degenerate guard

    // 1) adjacency
    MeshAdj adj;
    buildMeshAdj(liverMesh, adj);
    if (!adj.valid()) return false;

    // 2) anchor source vertices are stored directly (raycast already snapped
    //    to the nearest liver vertex when the anchor was picked).
    std::vector<int> anchorVert = anchors.srcVertIdx;

    // 3) geodesic distance fields on the liver mesh
    std::vector<std::vector<float>> dist;
    perAnchorDijkstra(liverMesh, adj, anchorVert, dist);

    // 4) per-group source radius. Liver is already in world units after
    //    registration, so r_src == r_tgt for s_ume == 1 (the U-1 default).
    std::vector<float> rSrc(G, 0.0f);
    for (int i = 0; i < G; ++i) {
        float r = groupRadiiTgt[i] / s_ume;
        if (r <= 1e-6f) r = 1e-3f;
        rSrc[i] = r;
    }

    // 5) Gaussian + baseline none per vertex
    const float baseline = std::max(params.baselineNone, 1e-6f);
    fieldOut.probs.assign(nV, {});
    fieldOut.numGroups = G;
    for (int v = 0; v < nV; ++v) {
        std::array<float, MAX_GROUPS + 1>& p = fieldOut.probs[v];
        p.fill(0.0f);

        float raw[MAX_GROUPS];
        float Z = baseline;
        for (int i = 0; i < G; ++i) {
            const float d = dist[i][v];
            if (d >= kInf) { raw[i] = 0.0f; continue; }
            const float twoR2 = 2.0f * rSrc[i] * rSrc[i] + 1e-12f;
            raw[i] = std::exp(-(d * d) / twoR2);
            Z += raw[i];
        }
        if (Z <= 0.0f) Z = 1.0f;
        for (int i = 0; i < G; ++i) p[i] = raw[i] / Z;
        p[NONE_IDX] = baseline / Z;
    }

    fieldOut.valid = true;
    return true;
}

// ============================================================================
//  Visualisation helpers
// ============================================================================

// 8 visually distinct group colours (matches the Umeyama marker palette order
// closely enough for eyeballing which anchor a region belongs to).
inline glm::vec3 paletteColor(int idx) {
    static const glm::vec3 pal[MAX_GROUPS] = {
        {1.00f, 0.20f, 0.20f},  // red
        {0.20f, 1.00f, 0.30f},  // green
        {0.30f, 0.50f, 1.00f},  // blue
        {1.00f, 0.95f, 0.20f},  // yellow
        {1.00f, 0.30f, 1.00f},  // magenta
        {0.20f, 0.95f, 1.00f},  // cyan
        {1.00f, 0.60f, 0.10f},  // orange
        {0.65f, 0.40f, 1.00f},  // violet
    };
    if (idx < 0 || idx >= MAX_GROUPS) return glm::vec3(0.5f);
    return pal[idx];
}

// Blend palette colours by group probability. Alpha encodes "how strongly does
// this vertex belong anywhere" = (1 - none). The "none"-dominated grey shows
// vertices excluded from every group.
inline glm::vec4 blendVertexColor(const std::array<float, MAX_GROUPS + 1>& p,
                                  int numGroups, float minAlpha = 0.15f) {
    glm::vec3 col(0.0f);
    float wsum = 0.0f;
    for (int i = 0; i < numGroups && i < MAX_GROUPS; ++i) {
        col  += paletteColor(i) * p[i];
        wsum += p[i];
    }
    const float none = p[NONE_IDX];
    if (wsum <= 1e-5f) {
        // fully "none" -> neutral grey, faint
        return glm::vec4(0.35f, 0.35f, 0.38f, minAlpha);
    }
    col /= wsum;                                    // normalise hue by group mass
    const float alpha = glm::clamp(1.0f - none, minAlpha, 1.0f);
    return glm::vec4(col, alpha);
}

} // namespace SoftPartition
