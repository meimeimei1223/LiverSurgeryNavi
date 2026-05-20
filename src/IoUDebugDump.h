// =============================================================================
//  IoUDebugDump.h  —  visual debug for the silhouette IoU pipeline
//
//  Dumps the EXACT bitmaps that computeSilhouette2DObjectiveFast() compares
//  when computing registrationHandle.compIoU2D, so a human can check
//  whether the right mask is being used.
//
//  Outputs (one set per call) under DEPTH_OUTPUT_PATH:
//      <prefix>_hitmap.png       1ch  gw×gh   liver silhouette (rasterized
//                                              from current liverMesh3D pose)
//      <prefix>_target_mask.png  1ch  gw×gh   what we score against (either
//                                              g_projectedLiverMask or, fallback,
//                                              g_boundaryDistMap stretched)
//      <prefix>_composite.png    3ch  gw×gh   green=intersection
//                                              red  =target only  (we miss)
//                                              blue =source only  (we overshoot)
//      <prefix>_boundary_map.png 3ch  mw×mh   current g_boundaryDistMap as
//                                              colour ramp (compare against
//                                              segmentation_mask.png on disk)
//
//  Intended usage (bind to a debug key, e.g. Shift+I):
//      glm::mat4 V = buildSilhouetteView();
//      glm::mat4 P = buildSilhouetteProj();
//      int W = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
//      int H = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;
//      IoUDebug::dump(DEPTH_OUTPUT_PATH, "iou_debug",
//                     liverMesh3D, V, P, W, H, /*step=*/8);
//
//  Note: the function takes imgW/imgH explicitly so the caller does NOT need
//  to play the gWindowWidth/Height override game that production code does.
// =============================================================================

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "stb_image.h"
#include "stb_image_write.h"

#include "DepthUtils.h"     // g_boundaryDistMap, g_projectedLiverMask, saveBoundaryMapVisualization
#include "mCutMesh.h"
#include "PathConfig.h"

namespace IoUDebug {

inline void dump(const std::string& outDir,
                 const std::string& prefix,
                 const mCutMesh*    liver,
                 const glm::mat4&   viewMat,
                 const glm::mat4&   projMat,
                 int                imgW,
                 int                imgH,
                 int                step = 8)
{
    auto t0 = std::chrono::steady_clock::now();

    if (!liver) {
        std::cerr << "[IoUDebug] liver mesh is null -- skipping" << std::endl;
        return;
    }
    if (!g_boundaryDistMap.valid) {
        std::cerr << "[IoUDebug] g_boundaryDistMap is INVALID --"
                     " run depth pipeline first; skipping" << std::endl;
        return;
    }
    if (imgW <= 0 || imgH <= 0) {
        std::cerr << "[IoUDebug] bad imgW/imgH (" << imgW << "x" << imgH
                  << ") -- skipping" << std::endl;
        return;
    }

    const int mw = g_boundaryDistMap.width;
    const int mh = g_boundaryDistMap.height;
    const int gw = (imgW + step - 1) / step;
    const int gh = (imgH + step - 1) / step;

    // ------------------------------------------------------------
    //  Step A  rasterize liver silhouette into hitmap[gw*gh]
    //          (faithful copy of computeSilhouette2DObjectiveFast)
    // ------------------------------------------------------------
    const auto& V = liver->mVertices;
    const auto& I = liver->mIndices;
    const int   nVerts = (int)(V.size() / 3);
    const int   nTris  = (int)(I.size() / 3);
    if (nVerts == 0 || nTris == 0) {
        std::cerr << "[IoUDebug] liver mesh empty (verts=" << nVerts
                  << ", tris=" << nTris << ") -- skipping" << std::endl;
        return;
    }

    glm::mat4 MVP = projMat * viewMat;
    std::vector<glm::vec3> screen(nVerts);
    std::vector<uint8_t>   inside(nVerts, 0);
    const float halfW   = imgW * 0.5f;
    const float halfH   = imgH * 0.5f;
    const float invStep = 1.0f / (float)step;

    for (int i = 0; i < nVerts; i++) {
        glm::vec4 v(V[i*3], V[i*3+1], V[i*3+2], 1.0f);
        glm::vec4 c = MVP * v;
        if (std::abs(c.w) < 1e-8f) {
            screen[i] = glm::vec3(0.f, 0.f, 2.f);
            continue;
        }
        float ndcX = c.x / c.w;
        float ndcY = c.y / c.w;
        float ndcZ = c.z / c.w;
        float px = (ndcX + 1.f) * halfW;
        float py = (1.f - ndcY) * halfH;     // Y flip to image coords
        screen[i] = glm::vec3(px * invStep, py * invStep, ndcZ);
        inside[i] = (ndcX > -1.2f && ndcX < 1.2f &&
                     ndcY > -1.2f && ndcY < 1.2f &&
                     ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }

    std::vector<uint8_t> hitmap(gw * gh, 0);
    for (int ti = 0; ti < nTris; ti++) {
        int i0 = I[ti*3+0], i1 = I[ti*3+1], i2 = I[ti*3+2];
        if (!inside[i0] && !inside[i1] && !inside[i2]) continue;
        const glm::vec3& s0 = screen[i0];
        const glm::vec3& s1 = screen[i1];
        const glm::vec3& s2 = screen[i2];
        float area2 = (s1.x - s0.x)*(s2.y - s0.y)
                    - (s2.x - s0.x)*(s1.y - s0.y);
        if (area2 <= 0.0f) continue;        // backface
        float minX = std::min({s0.x, s1.x, s2.x});
        float maxX = std::max({s0.x, s1.x, s2.x});
        float minY = std::min({s0.y, s1.y, s2.y});
        float maxY = std::max({s0.y, s1.y, s2.y});
        int x0 = std::max(0,    (int)std::floor(minX));
        int x1 = std::min(gw-1, (int)std::ceil (maxX));
        int y0 = std::max(0,    (int)std::floor(minY));
        int y1 = std::min(gh-1, (int)std::ceil (maxY));
        if (x0 > x1 || y0 > y1) continue;
        float invArea = 1.0f / area2;
        for (int y = y0; y <= y1; y++) {
            float py = (float)y + 0.5f;
            for (int x = x0; x <= x1; x++) {
                float px = (float)x + 0.5f;
                float w0 = ((s1.x - px)*(s2.y - py)
                          - (s2.x - px)*(s1.y - py)) * invArea;
                float w1 = ((s2.x - px)*(s0.y - py)
                          - (s0.x - px)*(s2.y - py)) * invArea;
                float w2 = 1.0f - w0 - w1;
                if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    hitmap[y*gw + x] = 1;
            }
        }
    }

    // ------------------------------------------------------------
    //  Step B  build target mask (same priority as production)
    // ------------------------------------------------------------
    std::vector<uint8_t> targetMask(gw * gh, 0);
    const bool useProjected = (g_projectedLiverMask.valid &&
                               g_projectedLiverMask.width  == imgW &&
                               g_projectedLiverMask.height == imgH);
    if (useProjected) {
        const auto& pm = g_projectedLiverMask.data;
        for (int gy = 0; gy < gh; gy++)
            for (int gx = 0; gx < gw; gx++) {
                int ipx = gx * step + step/2;
                int ipy = gy * step + step/2;
                ipx = std::clamp(ipx, 0, imgW - 1);
                ipy = std::clamp(ipy, 0, imgH - 1);
                targetMask[gy*gw + gx] = pm[ipy * imgW + ipx] ? 1 : 0;
            }
    } else {
        for (int gy = 0; gy < gh; gy++)
            for (int gx = 0; gx < gw; gx++) {
                int ipx = gx * step + step/2;
                int ipy = gy * step + step/2;
                int mx = std::clamp(ipx * mw / imgW, 0, mw - 1);
                int my = std::clamp(ipy * mh / imgH, 0, mh - 1);
                targetMask[gy*gw + gx] =
                    (g_boundaryDistMap.data[my*mw + mx] < 9000.f) ? 1 : 0;
            }
    }

    // ------------------------------------------------------------
    //  Step C  IoU stats
    // ------------------------------------------------------------
    int inter = 0, uni = 0, srcOnly = 0, tgtOnly = 0;
    int srcPix = 0, tgtPix = 0;
    for (int i = 0; i < gw*gh; i++) {
        bool s = hitmap[i] != 0;
        bool t = targetMask[i] != 0;
        if (s) srcPix++;
        if (t) tgtPix++;
        if (s && t)      { inter++;  uni++; }
        else if (s)      { srcOnly++; uni++; }
        else if (t)      { tgtOnly++; uni++; }
    }
    float iou = uni > 0 ? (float)inter / (float)uni : 0.0f;

    // ------------------------------------------------------------
    //  Step D  save PNGs
    // ------------------------------------------------------------
    auto write1ch = [&](const std::string& path,
                        const std::vector<uint8_t>& bin,
                        int w, int h) -> bool {
        std::vector<uint8_t> tmp(w*h);
        for (int i = 0; i < w*h; i++) tmp[i] = bin[i] ? 255 : 0;
        return stbi_write_png(path.c_str(), w, h, 1, tmp.data(), w) != 0;
    };

    const std::string pHit  = outDir + prefix + "_hitmap.png";
    const std::string pTgt  = outDir + prefix + "_target_mask.png";
    const std::string pComp = outDir + prefix + "_composite.png";
    const std::string pBdm  = outDir + prefix + "_boundary_map.png";

    bool okH = write1ch(pHit, hitmap,     gw, gh);
    bool okT = write1ch(pTgt, targetMask, gw, gh);

    // composite: green = inter, red = tgt only (missed), blue = src only (overshot)
    std::vector<uint8_t> comp(gw*gh*3, 0);
    for (int i = 0; i < gw*gh; i++) {
        bool s = hitmap[i] != 0;
        bool t = targetMask[i] != 0;
        uint8_t r = 30, g = 30, b = 30;     // background (neither)
        if (s && t)       { r = 0;   g = 200; b = 80;  }
        else if (s)       { r = 80;  g = 80;  b = 230; }   // overshoot
        else if (t)       { r = 230; g = 80;  b = 80;  }   // miss
        comp[i*3+0] = r; comp[i*3+1] = g; comp[i*3+2] = b;
    }
    bool okC = stbi_write_png(pComp.c_str(), gw, gh, 3, comp.data(), gw*3) != 0;

    // current g_boundaryDistMap as a heat-style PNG (full mask resolution)
    saveBoundaryMapVisualization(pBdm);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ------------------------------------------------------------
    //  Step E  human-readable summary
    // ------------------------------------------------------------
    std::cout << "\n[IoUDebug] ====== " << prefix << " ======\n"
              << "  imgW x imgH       : " << imgW << " x " << imgH
              << "   step=" << step
              << "   grid=" << gw << " x " << gh << "\n"
              << "  boundary map size : " << mw << " x " << mh
              << "  (valid=" << (g_boundaryDistMap.valid ? "YES" : "NO") << ")\n"
              << "  projectedLiverMask: valid=" << (g_projectedLiverMask.valid ? "YES":"NO");
    if (g_projectedLiverMask.valid) {
        std::cout << "   size=" << g_projectedLiverMask.width
                  << "x" << g_projectedLiverMask.height
                  << "   (matches imgW/imgH? "
                  << (useProjected ? "YES" : "NO")
                  << ")";
    }
    std::cout << "\n"
              << "  source path used  : "
              << (useProjected ? "g_projectedLiverMask"
                               : "g_boundaryDistMap (fallback)") << "\n"
              << "  liver mesh        : verts=" << nVerts
              << "   tris=" << nTris << "\n"
              << "  pixel counts      : src(hitmap)=" << srcPix
              << "   tgt(mask)=" << tgtPix << "\n"
              << "  inter / union     : " << inter << " / " << uni << "\n"
              << "  src_only / tgt_only: " << srcOnly << " / " << tgtOnly
              << "    (over / miss)\n"
              << "  IoU               : " << iou
              << "    Hausdorff target diag est=" << (gw + gh)/2 << " px\n"
              << "  saved (" << ms << " ms):\n"
              << "    " << pHit  << "   " << (okH ? "OK" : "FAIL") << "\n"
              << "    " << pTgt  << "   " << (okT ? "OK" : "FAIL") << "\n"
              << "    " << pComp << "   " << (okC ? "OK" : "FAIL") << "\n"
              << "    " << pBdm  << "   (compare with segmentation_mask.png)\n"
              << "[IoUDebug] ===================="
              << std::endl;
}

} // namespace IoUDebug
