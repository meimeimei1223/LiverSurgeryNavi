#pragma once
// RimPairSampling.h
// =====================================================================
//  [Phase C] Helpers for selecting K representative pairs out of the
//  19,000+ rim-rim NN pairs captured at Ctrl+G / Ctrl+Shift+G Phase F.5.
//
//  Inputs come straight from the publish/restore globals
//    g_lastRimPairSrcVertIdx  (size == g_lastRimMatched, full-mesh idx)
//    g_lastRimPairTgtPos      (size == g_lastRimMatched, world coords)
//  or, after applyEntry, from the PoseEntry copies of the same.
//
//  Output is a vector of indices INTO those pair vectors. The caller
//  (main.cpp drawing block) then looks up the actual src vertex
//  position from liverMesh3D->mVertices (so the marker follows
//  subsequent ICP/Apply) and the tgt position from g_lastRimPairTgtPos.
//
//  All functions are pure — no globals read or written, deterministic
//  given the same (seed, mode, inputs). Cheap enough to call every
//  frame (sort O(N log N) for ~20k entries is ~1 ms; ArcUniform O(N log N)
//  similarly).
// =====================================================================

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <GL/glew.h>     // GLfloat
#include <glm/glm.hpp>

namespace RimPairSampling {

// =====================================================================
// Mode selector for sampleRimPairIndices.
//
//   ArcUniform : sort all tgt points by their azimuth angle around the
//                tgt centroid (xy-plane, atan2(y - cy, x - cx)), then
//                pick K evenly-spaced indices. Visually stable — the
//                same "clock positions" are highlighted across runs,
//                which helps comparing rim drift between Pose Library
//                entries.
//                ← DEFAULT.
//
//   WorstK     : sort all pairs by current src-tgt distance descending,
//                take the top K. Diagnostic mode — surfaces the worst-
//                aligned pairs, useful for understanding what's
//                pulling the CMA-ES away from the rim band.
//
//   BestK      : sort ascending, take top K. Sanity-check mode —
//                "these are the pairs the optimiser is happy with."
//                Less informative than WorstK in practice.
//
//   Random     : uniform sample without replacement, seeded by the
//                seed argument. Useful for exploratory eyeballing
//                ("are the well-aligned pairs evenly distributed?").
//                Reshuffle button increments the seed.
// =====================================================================
enum class Mode {
    ArcUniform = 0,
    WorstK     = 1,
    BestK      = 2,
    Random     = 3
};

// =====================================================================
// Pick K indices into (pairSrcVertIdx, pairTgtPos) according to `mode`.
//
//   pairSrcVertIdx : g_lastRimPairSrcVertIdx (or PoseEntry.rimPairSrcVertIdx)
//   pairTgtPos     : g_lastRimPairTgtPos (or PoseEntry.rimPairTgtPos)
//   liverVerts     : liverMesh3D->mVertices, used only by WorstK/BestK
//                    to compute current src-tgt distances. Pass empty
//                    if mode is ArcUniform/Random — they ignore it.
//   K              : how many pairs to return. Clamped to [1, N] where
//                    N = pairSrcVertIdx.size().
//   seed           : only used by Random; ignored by other modes.
//
// Returns: indices in ascending order (deterministic given inputs).
//          Empty if pairSrcVertIdx is empty.
//
// Cost: O(N log N) for ArcUniform/WorstK/BestK (one sort).
//       O(N + K) for Random (Fisher-Yates partial shuffle of an index
//       array). All cheap at N≈20k.
// =====================================================================
inline std::vector<int> sampleRimPairIndices(
    const std::vector<int>&        pairSrcVertIdx,
    const std::vector<glm::vec3>&  pairTgtPos,
    const std::vector<GLfloat>&    liverVerts,   // may be empty
    Mode                           mode,
    int                            K,
    uint32_t                       seed)
{
    std::vector<int> out;
    const int N = (int)std::min(pairSrcVertIdx.size(), pairTgtPos.size());
    if (N <= 0 || K <= 0) return out;
    if (K > N) K = N;

    // ----- ArcUniform -------------------------------------------------
    if (mode == Mode::ArcUniform) {
        // 1. Compute tgt centroid in xy-plane (depth-axis is +Z in this
        //    project, so azimuth on xy is the natural "around the rim"
        //    coordinate from the AR camera's viewpoint).
        double cx = 0.0, cy = 0.0;
        for (const auto& p : pairTgtPos) { cx += p.x; cy += p.y; }
        cx /= (double)N;
        cy /= (double)N;

        // 2. Compute azimuth angle for each pair.
        struct AnglePair { float angle; int idx; };
        std::vector<AnglePair> angs(N);
        for (int i = 0; i < N; i++) {
            const glm::vec3& p = pairTgtPos[i];
            angs[i].angle = std::atan2((float)((double)p.y - cy),
                                       (float)((double)p.x - cx));
            angs[i].idx   = i;
        }

        // 3. Sort by angle (full O(N log N); cheap at N≈20k).
        std::sort(angs.begin(), angs.end(),
                  [](const AnglePair& a, const AnglePair& b) {
                      return a.angle < b.angle;
                  });

        // 4. Pick K evenly-spaced positions and grab their original idx.
        //    Use floor((k + 0.5) * N / K) for centred sampling so the
        //    selection is symmetric around the angular wheel.
        out.reserve(K);
        for (int k = 0; k < K; k++) {
            int pos = (int)(((double)k + 0.5) * (double)N / (double)K);
            if (pos < 0) pos = 0;
            if (pos >= N) pos = N - 1;
            out.push_back(angs[pos].idx);
        }
        // Return in ascending original-index order so successive frames
        // present the K pairs in a stable colour assignment (otherwise
        // the angular ordering would re-shuffle colours every redraw if
        // pairs were added/removed across sessions).
        std::sort(out.begin(), out.end());
        return out;
    }

    // ----- WorstK / BestK ---------------------------------------------
    // Need current src position from liverVerts. If liverVerts is too
    // small or doesn't match the indices, fall back to Random so the
    // viewer still shows SOMETHING rather than blanking out.
    if (mode == Mode::WorstK || mode == Mode::BestK) {
        const size_t nV = liverVerts.size() / 3;
        bool can_measure = (nV > 0);
        // Quick sanity scan: spot-check the first valid idx fits in nV.
        if (can_measure) {
            for (int i = 0; i < std::min(N, 16); i++) {
                int v = pairSrcVertIdx[i];
                if (v < 0 || (size_t)v >= nV) { can_measure = false; break; }
            }
        }
        if (!can_measure) {
            // Degrade silently to Random (caller already sees the same
            // pair count, just not sorted by distance).
            mode = Mode::Random;
        } else {
            struct DistPair { float dist_sq; int idx; };
            std::vector<DistPair> dists;
            dists.reserve(N);
            for (int i = 0; i < N; i++) {
                int v = pairSrcVertIdx[i];
                if (v < 0 || (size_t)v >= nV) continue;
                glm::vec3 src(liverVerts[v*3],
                              liverVerts[v*3+1],
                              liverVerts[v*3+2]);
                glm::vec3 d = src - pairTgtPos[i];
                dists.push_back({ glm::dot(d, d), i });
            }
            const bool worst = (mode == Mode::WorstK);
            std::sort(dists.begin(), dists.end(),
                      [worst](const DistPair& a, const DistPair& b) {
                          return worst ? (a.dist_sq > b.dist_sq)
                                       : (a.dist_sq < b.dist_sq);
                      });
            const int actualK = std::min<int>(K, (int)dists.size());
            out.reserve(actualK);
            for (int k = 0; k < actualK; k++) out.push_back(dists[k].idx);
            // Stable order regardless of sort criterion → consistent
            // colour assignment per redraw (same rationale as ArcUniform).
            std::sort(out.begin(), out.end());
            return out;
        }
    }

    // ----- Random (also: WorstK/BestK fallback) -----------------------
    // Fisher–Yates partial shuffle: select K out of N without replacement,
    // O(N + K), deterministic for a given seed.
    {
        std::vector<int> all(N);
        for (int i = 0; i < N; i++) all[i] = i;
        std::mt19937 rng(seed ? seed : 1u);   // 0-seed → 1 (mt19937 quirk)
        for (int k = 0; k < K; k++) {
            std::uniform_int_distribution<int> dist(k, N - 1);
            int j = dist(rng);
            std::swap(all[k], all[j]);
        }
        out.assign(all.begin(), all.begin() + K);
        std::sort(out.begin(), out.end());   // stable colour assignment
        return out;
    }
}

}  // namespace RimPairSampling
