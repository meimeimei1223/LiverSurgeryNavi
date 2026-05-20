// =============================================================================
//  OBJDistributionDiag.h
// -----------------------------------------------------------------------------
//  Diagnostic visualizations for a depth-derived OBJ mesh. Writes three
//  PNG files into DEPTH_OUTPUT_PATH and prints summary statistics:
//
//    debug_z_heatmap.png     Z value at each vertex's reprojected pixel,
//                            colored cool->warm (near->far).  This is the
//                            "where are the spikes?" map.
//
//    debug_z_histogram.png   1D histogram of Z values across all vertices,
//                            with annotated percentile lines (p50, p95,
//                            p99). Lets you eyeball whether the body and
//                            the silhouette spikes form distinct modes.
//
//    debug_normal_cos.png    abs(dot(normal, -view_dir)) at each pixel.
//                            Black = silhouette face (likely spike), white
//                            = camera-facing (likely body). This is the
//                            preview for the normal-based cleanup filter.
//
//  Call once after loading the OBJ -- before any cleanup so you see the
//  raw distribution. Triggered manually from main.cpp:
//
//      Reg3DCustom::diagnoseOBJDistribution(*screenMesh, K,
//                                           DEPTH_OUTPUT_PATH);
//
//  Requires: stb_image_write.h available in include path.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "stb_image_write.h"

#include "mCutMesh.h"
#include "OBJTargetExtraction.h"   // CameraIntrinsics, OBJ_Y_SIGN_*

namespace Reg3DCustom {

// -----------------------------------------------------------------------------
//  Helpers
// -----------------------------------------------------------------------------
namespace diag_detail {

//  Cool (blue) -> warm (red) colormap. t in [0, 1].
inline void coolWarm(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    t = std::max(0.0f, std::min(1.0f, t));
    if (t < 0.5f) {
        const float u = t * 2.0f;             // 0..1 across cool half
        r = static_cast<uint8_t>(  0  + 100 * u);
        g = static_cast<uint8_t>( 80 + 175 * u);
        b = static_cast<uint8_t>(255 -  50 * u);
    } else {
        const float u = (t - 0.5f) * 2.0f;    // 0..1 across warm half
        r = static_cast<uint8_t>(100 + 155 * u);
        g = static_cast<uint8_t>(255 - 175 * u);
        b = static_cast<uint8_t>(205 - 205 * u);
    }
}

inline float quantile(std::vector<float>& v, float q) {
    if (v.empty()) return 0.0f;
    const size_t k = static_cast<size_t>(q * (v.size() - 1));
    auto it = v.begin() + k;
    std::nth_element(v.begin(), it, v.end());
    return *it;
}

} // namespace diag_detail

// -----------------------------------------------------------------------------
//  Render the Z value heatmap by projecting every vertex into the image
//  with the supplied intrinsics. Each pixel gets the *minimum* Z that
//  landed on it (i.e. the closest surface), so silhouette spikes (large Z)
//  show up only where they don't overlap the body. y_sign matches the
//  convention used by extractTargetFromOBJ.
// -----------------------------------------------------------------------------
inline void renderZHeatmap(const mCutMesh& mesh,
                           const CameraIntrinsics& K,
                           const std::string& outPath,
                           float y_sign,
                           float zMin, float zMax)
{
    const int W = K.width, H = K.height;
    std::vector<float> zbuf(static_cast<size_t>(W) * H,
                            std::numeric_limits<float>::infinity());

    const size_t nV = mesh.mVertices.size() / 3;
    for (size_t v = 0; v < nV; ++v) {
        const float X = mesh.mVertices[v * 3 + 0];
        const float Y = mesh.mVertices[v * 3 + 1];
        const float Z = mesh.mVertices[v * 3 + 2];
        if (Z <= 0.01f) continue;
        const int u = static_cast<int>(K.fx * X / Z + K.cx);
        const int p = static_cast<int>(K.fy * (y_sign * Y) / Z + K.cy);
        if (u < 0 || u >= W || p < 0 || p >= H) continue;
        const size_t idx = static_cast<size_t>(p) * W + u;
        if (Z < zbuf[idx]) zbuf[idx] = Z;
    }

    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 30);
    const float zRange = std::max(1e-6f, zMax - zMin);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = static_cast<size_t>(y) * W + x;
            const float z = zbuf[idx];
            if (!std::isfinite(z)) continue;
            const float t = (z - zMin) / zRange;
            uint8_t r, g, b;
            diag_detail::coolWarm(t, r, g, b);
            img[idx * 3 + 0] = r;
            img[idx * 3 + 1] = g;
            img[idx * 3 + 2] = b;
        }
    }
    stbi_write_png(outPath.c_str(), W, H, 3, img.data(), W * 3);
    std::cout << "[Diag] wrote " << outPath
              << "  (Z range " << zMin << ".." << zMax << " m)" << std::endl;
}

// -----------------------------------------------------------------------------
//  1D histogram of Z values rendered as a PNG. Useful for spotting modes
//  (body vs silhouette spikes) at a glance.
// -----------------------------------------------------------------------------
inline void renderZHistogram(const mCutMesh& mesh,
                             const std::string& outPath,
                             float zMin, float zMax,
                             int nBins = 200,
                             int imgH  = 360)
{
    const size_t nV = mesh.mVertices.size() / 3;
    std::vector<int> bins(nBins, 0);
    const float zRange = std::max(1e-6f, zMax - zMin);
    int outOfRange = 0;
    for (size_t v = 0; v < nV; ++v) {
        const float Z = mesh.mVertices[v * 3 + 2];
        const int b = static_cast<int>((Z - zMin) / zRange * nBins);
        if (b < 0 || b >= nBins) { ++outOfRange; continue; }
        ++bins[b];
    }
    const int peak = std::max(1, *std::max_element(bins.begin(), bins.end()));

    // Image canvas: width = nBins*4 px (chunky bars), height = imgH
    const int barW = 4;
    const int W    = nBins * barW + 80;       // +80 for left margin
    const int H    = imgH;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 245);  // light bg

    auto setPx = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        const size_t i = (static_cast<size_t>(y) * W + x) * 3;
        img[i + 0] = r; img[i + 1] = g; img[i + 2] = b;
    };

    // Bars
    for (int i = 0; i < nBins; ++i) {
        const int barH = static_cast<int>(static_cast<float>(bins[i]) / peak
                                          * (imgH - 40));
        const int x0 = 60 + i * barW;
        for (int y = imgH - 20 - barH; y < imgH - 20; ++y) {
            for (int dx = 0; dx < barW - 1; ++dx) {
                setPx(x0 + dx, y, 60, 110, 200);
            }
        }
    }

    // Percentile markers (vertical lines)
    std::vector<float> zs;
    zs.reserve(nV);
    for (size_t v = 0; v < nV; ++v) zs.push_back(mesh.mVertices[v * 3 + 2]);
    auto drawLine = [&](float zVal, uint8_t r, uint8_t g, uint8_t b) {
        const int bin = static_cast<int>((zVal - zMin) / zRange * nBins);
        if (bin < 0 || bin >= nBins) return;
        const int x = 60 + bin * barW + barW / 2;
        for (int y = 0; y < imgH - 20; ++y) setPx(x, y, r, g, b);
    };
    drawLine(diag_detail::quantile(zs, 0.50f), 200,  60,  60);   // red   p50
    drawLine(diag_detail::quantile(zs, 0.95f), 200, 140,  40);   // amber p95
    drawLine(diag_detail::quantile(zs, 0.99f), 160,  40, 160);   // purple p99

    // Baseline
    for (int x = 0; x < W; ++x) setPx(x, imgH - 20, 0, 0, 0);

    stbi_write_png(outPath.c_str(), W, H, 3, img.data(), W * 3);
    std::cout << "[Diag] wrote " << outPath
              << "  (bars=blue, p50=red, p95=amber, p99=purple"
              << ", outOfRange=" << outOfRange << ")" << std::endl;
}

// -----------------------------------------------------------------------------
//  Render |dot(face_normal, -view_dir)| as grayscale per-pixel. Computes
//  face normals from triangle geometry on the fly, so this works whether
//  or not per-vertex normals have been built yet.
//  Black = silhouette (likely spike), white = camera-facing (likely body).
// -----------------------------------------------------------------------------
inline void renderNormalCosMap(const mCutMesh& mesh,
                               const CameraIntrinsics& K,
                               const std::string& outPath,
                               float y_sign)
{
    const int W = K.width, H = K.height;
    std::vector<float> zbuf(static_cast<size_t>(W) * H,
                            std::numeric_limits<float>::infinity());
    std::vector<float> cosbuf(static_cast<size_t>(W) * H, 0.0f);

    auto vertAt = [&](GLuint i) {
        return glm::vec3(mesh.mVertices[i * 3 + 0],
                         mesh.mVertices[i * 3 + 1],
                         mesh.mVertices[i * 3 + 2]);
    };

    // Walk triangles, project the centroid, write per-pixel cos angle.
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        const glm::vec3 a = vertAt(mesh.mIndices[i    ]);
        const glm::vec3 b = vertAt(mesh.mIndices[i + 1]);
        const glm::vec3 c = vertAt(mesh.mIndices[i + 2]);
        const glm::vec3 cross = glm::cross(b - a, c - a);
        const float cl = glm::length(cross);
        if (cl < 1e-12f) continue;
        const glm::vec3 N        = cross / cl;
        const glm::vec3 centroid = (a + b + c) / 3.0f;
        if (centroid.z <= 0.01f) continue;
        const glm::vec3 viewDir  = glm::normalize(centroid);  // origin -> P
        const float cosAng = std::abs(glm::dot(N, viewDir));

        const int u = static_cast<int>(K.fx * centroid.x / centroid.z + K.cx);
        const int p = static_cast<int>(K.fy * (y_sign * centroid.y) / centroid.z + K.cy);
        if (u < 0 || u >= W || p < 0 || p >= H) continue;
        const size_t idx = static_cast<size_t>(p) * W + u;
        if (centroid.z < zbuf[idx]) {
            zbuf[idx]   = centroid.z;
            cosbuf[idx] = cosAng;
        }
    }

    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 30);
    int spikePixels = 0, totalPixels = 0;
    for (size_t i = 0; i < cosbuf.size(); ++i) {
        if (!std::isfinite(zbuf[i])) continue;
        ++totalPixels;
        const uint8_t v = static_cast<uint8_t>(cosbuf[i] * 255.0f);
        img[i * 3 + 0] = v;
        img[i * 3 + 1] = v;
        img[i * 3 + 2] = v;
        if (cosbuf[i] < 0.2f) ++spikePixels;
    }
    stbi_write_png(outPath.c_str(), W, H, 3, img.data(), W * 3);
    std::cout << "[Diag] wrote " << outPath
              << "  silhouette-like pixels (cos<0.2): "
              << spikePixels << "/" << totalPixels
              << "  (" << (totalPixels ? 100.0 * spikePixels / totalPixels : 0.0)
              << "%)" << std::endl;
}

// -----------------------------------------------------------------------------
//  One-call driver. Computes Z range from data, runs all three renders,
//  and prints percentile summary so the user can pick a strategy.
// -----------------------------------------------------------------------------
inline void diagnoseOBJDistribution(const mCutMesh& mesh,
                                    const CameraIntrinsics& K,
                                    const std::string& outDir,
                                    float y_sign = OBJ_Y_SIGN_OPENGL)
{
    const size_t nV = mesh.mVertices.size() / 3;
    if (nV == 0) {
        std::cerr << "[Diag] empty mesh, skipping" << std::endl;
        return;
    }

    // Z range and percentiles
    std::vector<float> zs;
    zs.reserve(nV);
    for (size_t v = 0; v < nV; ++v) zs.push_back(mesh.mVertices[v * 3 + 2]);
    auto p = [&](float q) { return diag_detail::quantile(zs, q); };

    const float zMin   = p(0.00f);
    const float zMax   = p(1.00f);
    const float zP50   = p(0.50f);
    const float zP95   = p(0.95f);
    const float zP99   = p(0.99f);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[Diag] Z stats over " << nV << " vertices:"
              << "  min="  << zMin
              << "  p5="   << p(0.05f)
              << "  p50="  << zP50
              << "  p95="  << zP95
              << "  p99="  << zP99
              << "  max="  << zMax
              << std::endl;
    std::cout << "[Diag] body width estimate (p95-p5) = "
              << (zP95 - p(0.05f))
              << " m, tail (max-p99) = " << (zMax - zP99) << " m"
              << std::endl;

    std::string base = outDir;
    if (!base.empty() && base.back() != '/' && base.back() != '\\') base += '/';

    renderZHeatmap   (mesh, K, base + "debug_z_heatmap.png",   y_sign,
                      zMin, zMax);
    renderZHistogram (mesh,    base + "debug_z_histogram.png",
                      zMin, zMax);
    renderNormalCosMap(mesh, K, base + "debug_normal_cos.png", y_sign);
}

} // namespace Reg3DCustom
