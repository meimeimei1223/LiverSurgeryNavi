#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "MeshDrawing.h"   // setUp()
#include "PinholeProjection.h"

// stb_image / stb_image_write の関数を直接 #include せず、forward 宣言だけで使う。
// stb_image.h を別の TU が先に include すると extern "C" / C++ リンケージの
// 競合が発生するため、mCutMesh.h と同じガード方式 (STBI_INCLUDE_STB_IMAGE_H が
// 既に定義されていれば forward 宣言をスキップ) で衝突を回避する。
// 実体定義は main.cpp の STB_IMAGE_IMPLEMENTATION ブロックで提供される。
#ifndef STBI_INCLUDE_STB_IMAGE_H
extern unsigned char *stbi_load(char const *filename, int *x, int *y,
                                int *channels_in_file, int desired_channels);
extern void stbi_image_free(void *retval_from_stbi_load);
#endif
extern int stbi_write_png(const char* filename, int w, int h, int comp,
                          const void* data, int stride_in_bytes);

#include "PathConfig.h"

// =========================================================================
//  Helper function to load mask PNG
// =========================================================================
inline bool loadMaskPNG(const std::string& path,
                        std::vector<unsigned char>& out,
                        int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (!data) { std::cerr << "[Mask] Cannot load " << path << std::endl; return false; }
    out.assign(data, data + (size_t)width * height);
    stbi_image_free(data);
    return true;
}

// =========================================================================
//  extern globals (defined in main.cpp)
// =========================================================================
extern int gGridWidth;

// =========================================================================
//  convert3DToImagePixel
//    3D hit position -> 2D image pixel coords (from screenMesh bounding box)
// =========================================================================
inline bool convert3DToImagePixel(const glm::vec3& hitPos3D,
                                  const mCutMesh* mesh,
                                  int& outPixelX, int& outPixelY) {
    if (!mesh || mesh->mVertices.empty()) return false;

    int imgW = mesh->loadedImageWidth;
    int imgH = mesh->loadedImageHeight;
    if (imgW <= 0 || imgH <= 0) return false;

    float minX =  1e30f, maxX = -1e30f;
    float minY =  1e30f, maxY = -1e30f;
    for (size_t i = 0; i < mesh->mVertices.size(); i += 3) {
        float x = mesh->mVertices[i];
        float y = mesh->mVertices[i + 1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    float rangeX = maxX - minX;
    float rangeY = maxY - minY;
    if (rangeX < 1e-6f || rangeY < 1e-6f) return false;

    float u = (hitPos3D.x - minX) / rangeX;
    float v = (maxY - hitPos3D.y) / rangeY;

    float imageAspect = (float)imgW / (float)imgH;
    float meshAspect  = rangeX / rangeY;

    if (meshAspect > imageAspect + 1e-4f) {
        float scale = meshAspect / imageAspect;
        float center = 0.5f;
        u = center + (u - center) / scale;
    } else if (imageAspect > meshAspect + 1e-4f) {
        float scale = imageAspect / meshAspect;
        float center = 0.5f;
        v = center + (v - center) / scale;
    }

    u = std::max(0.0f, std::min(1.0f, u));
    v = std::max(0.0f, std::min(1.0f, v));

    outPixelX = static_cast<int>(u * imgW);
    outPixelY = static_cast<int>(v * imgH);

    outPixelX = std::max(0, std::min(outPixelX, imgW - 1));
    outPixelY = std::max(0, std::min(outPixelY, imgH - 1));

    std::cout << "[SegPoint] 3D hit: (" << hitPos3D.x << ", " << hitPos3D.y << ", " << hitPos3D.z << ")" << std::endl;
    std::cout << "[SegPoint] BB: X[" << minX << ", " << maxX << "] Y[" << minY << ", " << maxY << "]" << std::endl;
    std::cout << "[SegPoint] u=" << u << " v=" << v << " -> 2D: (" << outPixelX << ", " << outPixelY << ")" << std::endl;

    return true;
}

// =========================================================================
//  regenerateDepthMesh
//    Rebuild screenMesh geometry with updated depth scale
// =========================================================================
inline void regenerateDepthMesh(mCutMesh* mesh, float depthScale, float meshScale,
                                int pinMode = 0,
                                const Pinhole::Intrinsics& K = Pinhole::Intrinsics{},
                                float baseScale = 0.0f) {
    if (!mesh || mesh->depthImageData.empty()) {
        std::cout << "[DepthScale] No depth data loaded" << std::endl;
        return;
    }
    int gw = gGridWidth;
    int gh = gw * mesh->loadedImageHeight / mesh->loadedImageWidth;
    int effectiveMode = (pinMode != 0 && K.valid) ? pinMode : 0;

    std::vector<float> depths;
    if (effectiveMode == 2) {
        // Pure Pinhole: metric depth + mask PNG
        std::string maskPath = std::string(DEPTH_OUTPUT_PATH) + "segmentation_mask.png";
        std::vector<unsigned char> maskData;
        int mW = 0, mH = 0;
        if (loadMaskPNG(maskPath, maskData, mW, mH)) {
            depths = mesh->calculateBlockDepthMetric(gw, gh, maskData, mW, mH, 0.9f);
        } else {
            std::cerr << "[DepthUtils] Mask load failed, fallback to /255 raw block" << std::endl;
            depths = mesh->calculateBlockDepthRaw(gw, gh, 0.9f);
        }
    } else {
        depths = mesh->calculateNormalizedDepth(gw, gh, 0.99f, 0.9f);
    }

    mesh->generateGridPlaneWithDepth(gw, gh, depths, 0.05f, depthScale,
                                     effectiveMode,
                                     effectiveMode ? K.fx : 0.0f,
                                     effectiveMode ? K.fy : 0.0f,
                                     effectiveMode ? K.cx : 0.0f,
                                     effectiveMode ? K.cy : 0.0f,
                                     effectiveMode ? K.width  : 0,
                                     effectiveMode ? K.height : 0,
                                     effectiveMode == 1 ? baseScale : 0.0f);
    for (size_t i = 0; i < mesh->mVertices.size(); i++)
        mesh->mVertices[i] *= meshScale;
    setUp(*mesh);
    std::cout << "[DepthScale] depthScale=" << depthScale;
    if (effectiveMode == 1) std::cout << " [DIFF PINHOLE base=" << baseScale << "]";
    else if (effectiveMode == 2) std::cout << " [PURE PINHOLE raw-depth]";
    std::cout << std::endl;
}

// =========================================================================
//  Boundary Distance Map
// =========================================================================
#include <queue>
#include <string>
#include "PathConfig.h"

struct BoundaryDistMap {
    std::vector<float> data;
    int width  = 0;
    int height = 0;
    bool valid = false;

    // Drop the cached map. Call this whenever segmentation_mask.png is
    // overwritten on disk (e.g. after a fresh Run Depth pass) so the next
    // ensureBoundaryMap() reloads from the new file instead of returning
    // the stale in-memory map.
    void invalidate() {
        valid = false;
        data.clear();
        width  = 0;
        height = 0;
    }
};

inline BoundaryDistMap g_boundaryDistMap;

// =========================================================================
//  Instrument Mask -- distance to instrument-segmented region (in pixels).
// =========================================================================
//  Built once per scene from instrument_segmentation_mask.png (optional).
//  Used by extractTargetFromOBJ to flag vertices near instruments so the
//  boundary classifier can exclude them. When the mask file is absent or
//  size-mismatched against the liver mask, this stays invalid and the
//  pipeline behaves as before (no instrument exclusion).
//
//  Storage convention: data[y*w+x] = BFS distance (in pixels) from the
//  nearest instrument pixel; 0 inside the instrument region, growing
//  outward; clamped to 9999 when no instrument seeds exist.
// =========================================================================
inline BoundaryDistMap g_instrumentDistMap;

// =========================================================================
//  Projected Liver Mask — camera-aware target for Shift+E
// =========================================================================
//  Shift+E optimizes silhouette IoU against g_boundaryDistMap. The legacy
//  shortcut in computeSilhouette2DObjectiveFast samples that mask by
//  stretching the window linearly onto the original SAM2 image — correct
//  only when the virtual camera stays at the initial pose where screenMesh
//  fills the screen. After the user scrolls/pans the camera, the shortcut
//  still stretches to the WHOLE window, so the target drifts off from
//  where the SAM2 mask actually lies on screenMesh — Shift+E then
//  optimizes against a wrong target and destroys the pose.
//
//  ProjectedLiverMask is the safety-net: before running Shift+E we render
//  screenMesh with the SAM2 mask as texture from the CURRENT camera, read
//  the color buffer back, and stash it as a window-sized binary image.
//  computeSilhouette2DObjectiveFast checks g_projectedLiverMask.valid and
//  prefers it when available; otherwise it falls back to the legacy
//  shortcut for backward compatibility.
//
//  Built on demand in main.cpp (needs GL + shaders so it can't live here);
//  invalidated immediately after Shift+E finishes so a subsequent camera
//  move doesn't silently use a stale mask.
// =========================================================================
struct ProjectedLiverMask {
    std::vector<uint8_t> data;   // width * height, 0 or 255, top-left origin
    int  width  = 0;
    int  height = 0;
    bool valid  = false;

    void invalidate() { valid = false; }
};

inline ProjectedLiverMask g_projectedLiverMask;

// =========================================================================
//  Mask cleanup helpers used before computing the boundary distance map.
// -------------------------------------------------------------------------
//  The raw SAM2 segmentation mask occasionally has:
//    (a) tiny black holes inside the liver region (SAM2 saw a specular
//        highlight or a small noise patch as background)
//    (b) tiny isolated white blobs outside the main liver (single pixel or
//        small clusters of false positives)
//
//  Both produce SPURIOUS internal "boundary" pixels in the BFS distance
//  map -- which then attract green points all over the interior.
//
//  Fix:
//    1. Morphological closing with a small structuring element (3x3, k iters)
//       fills holes whose width is < 2k pixels. Probe-shaped holes (tens of
//       pixels wide) are preserved.
//    2. Keep only the largest connected component to remove far-flung
//       speckles outside the liver.
//
//  These are O(w*h*k) with no allocations beyond temporary buffers and
//  add ~10ms at 1920x1080.
// =========================================================================
inline void maskDilate3x3(std::vector<bool>& m, int w, int h, std::vector<bool>& tmp) {
    tmp.assign((size_t)w * h, false);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool any = false;
            for (int dy = -1; dy <= 1 && !any; dy++)
                for (int dx = -1; dx <= 1 && !any; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    if (m[ny * w + nx]) any = true;
                }
            tmp[y * w + x] = any;
        }
    }
    m.swap(tmp);
}
inline void maskErode3x3(std::vector<bool>& m, int w, int h, std::vector<bool>& tmp) {
    tmp.assign((size_t)w * h, false);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool all = true;
            for (int dy = -1; dy <= 1 && all; dy++)
                for (int dx = -1; dx <= 1 && all; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) { all = false; break; }
                    if (!m[ny * w + nx]) all = false;
                }
            tmp[y * w + x] = all;
        }
    }
    m.swap(tmp);
}
inline void maskClose(std::vector<bool>& m, int w, int h, int iters) {
    std::vector<bool> tmp((size_t)w * h);
    for (int it = 0; it < iters; it++) maskDilate3x3(m, w, h, tmp);
    for (int it = 0; it < iters; it++) maskErode3x3 (m, w, h, tmp);
}
inline void maskKeepLargestComponent(std::vector<bool>& m, int w, int h) {
    std::vector<int> label((size_t)w * h, 0);
    int nextLabel = 0;
    int bestLabel = 0;
    int bestSize  = 0;
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!m[y * w + x] || label[y * w + x] != 0) continue;
            ++nextLabel;
            int compSize = 0;
            std::queue<std::pair<int,int>> q;
            q.push({x, y});
            label[y * w + x] = nextLabel;
            while (!q.empty()) {
                auto [cx, cy] = q.front(); q.pop();
                ++compSize;
                for (int d = 0; d < 4; d++) {
                    int nx = cx + dx[d], ny = cy + dy[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    int ni = ny * w + nx;
                    if (!m[ni] || label[ni] != 0) continue;
                    label[ni] = nextLabel;
                    q.push({nx, ny});
                }
            }
            if (compSize > bestSize) { bestSize = compSize; bestLabel = nextLabel; }
        }
    }
    if (bestLabel == 0) return;
    int kept = 0, dropped = 0;
    for (int i = 0; i < w * h; i++) {
        if (m[i] && label[i] != bestLabel) { m[i] = false; dropped++; }
        else if (m[i]) kept++;
    }
    std::cout << "[MaskCleanup] components=" << nextLabel
              << "  kept=" << kept << "px  dropped=" << dropped << "px" << std::endl;
}


// Hole filling: flood-fill the BACKGROUND from image borders. Any non-mask
// pixel NOT reached from a border is an internal hole (closed off by liver
// tissue on all sides) -- typically a specular highlight that the SAM2
// segmentation misclassified. We unconditionally flip those to mask=true.
//
// Why this works on our scene:
//   * specular highlights are isolated dark spots fully surrounded by liver
//     -> not reachable from border -> filled
//   * the surgical instrument enters from the edge of the field of view and
//     exits on the other side, so the dark "valley" under it is connected to
//     the image border -> reached from border -> NOT filled (correctly)
//
// Cost: one BFS over w*h pixels, ~30 ms at 1920x1080.
inline void fillInternalHoles(std::vector<bool>& m, int w, int h) {
    std::vector<bool> reachedFromBorder((size_t)w * h, false);
    std::queue<std::pair<int,int>> q;
    auto seed = [&](int x, int y) {
        int idx = y * w + x;
        if (!m[idx] && !reachedFromBorder[idx]) {
            reachedFromBorder[idx] = true;
            q.push({x, y});
        }
    };
    for (int x = 0; x < w; x++) { seed(x, 0); seed(x, h - 1); }
    for (int y = 0; y < h; y++) { seed(0, y); seed(w - 1, y); }

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx[d], ny = cy + dy[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            int ni = ny * w + nx;
            if (m[ni] || reachedFromBorder[ni]) continue;
            reachedFromBorder[ni] = true;
            q.push({nx, ny});
        }
    }

    int filled = 0;
    for (int i = 0; i < w * h; i++) {
        if (!m[i] && !reachedFromBorder[i]) {
            m[i] = true;
            ++filled;
        }
    }
    std::cout << "[MaskCleanup] hole fill: " << filled
              << " interior pixels (specular gaps) filled" << std::endl;
}

inline bool loadMaskAndComputeBoundaryMap(const std::string& maskPath) {
    int w, h, ch;
    unsigned char* img = stbi_load(maskPath.c_str(), &w, &h, &ch, 1);
    if (!img) {
        std::cerr << "[Boundary] Failed to load mask: " << maskPath << std::endl;
        return false;
    }

    std::vector<bool> mask(w * h);
    for (int i = 0; i < w * h; i++)
        mask[i] = (img[i] > 127);
    stbi_image_free(img);

    // ---- Mask cleanup pipeline ----
    // 1. Closing fills small internal holes (up to ~8px diameter at iters=4)
    //    -- removes pin-prick specular highlights cheaply.
    // 2. Hole filling (flood fill from border) eliminates ALL remaining
    //    closed internal holes regardless of size. This is the workhorse
    //    that kills the big specular reflection blobs.
    // 3. Largest connected component drops far-flung white speckles outside
    //    the liver. Done last so hole-fill operates on the most complete mask.
    maskClose(mask, w, h, /*iters=*/4);
    fillInternalHoles(mask, w, h);
    maskKeepLargestComponent(mask, w, h);

    std::vector<bool> isBoundary(w * h, false);
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!mask[y * w + x]) continue;
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h || !mask[ny * w + nx]) {
                    isBoundary[y * w + x] = true;
                    break;
                }
            }
        }
    }

    // ---- Stage 2 safety net: occupancy filter ----
    // After hole filling, any remaining "boundary" pixel surrounded by liver
    // is suspicious. We measure mask occupancy in a disk of radius R around
    // each boundary candidate; if it exceeds threshold, the candidate is on
    // the inside edge of something and we drop the boundary flag.
    // With hole-fill in place this rarely fires -- it's a backstop for
    // edge-touching specular gaps that hole-fill can't catch.
    constexpr int   kOccRadius = 25;     // disk radius in pixels (slightly bigger)
    constexpr float kOccThresh = 0.85f;  // drop boundary if interior ratio above this
    {
        // Build SAT of mask values (0 or 1). SAT is (h+1) x (w+1).
        std::vector<int> sat((size_t)(w + 1) * (h + 1), 0);
        for (int y = 1; y <= h; y++) {
            int rowSum = 0;
            for (int x = 1; x <= w; x++) {
                rowSum += mask[(y - 1) * w + (x - 1)] ? 1 : 0;
                sat[(size_t)y * (w + 1) + x] = sat[(size_t)(y - 1) * (w + 1) + x] + rowSum;
            }
        }
        auto rectSum = [&](int x0, int y0, int x1, int y1) -> int {
            // inclusive [x0, x1] x [y0, y1] ; clamp
            if (x0 < 0) x0 = 0;  if (y0 < 0) y0 = 0;
            if (x1 >= w) x1 = w - 1;  if (y1 >= h) y1 = h - 1;
            if (x0 > x1 || y0 > y1) return 0;
            return  sat[(size_t)(y1 + 1) * (w + 1) + (x1 + 1)]
                   - sat[(size_t)(y0    ) * (w + 1) + (x1 + 1)]
                   - sat[(size_t)(y1 + 1) * (w + 1) + (x0    )]
                   + sat[(size_t)(y0    ) * (w + 1) + (x0    )];
        };
        int dropped = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (!isBoundary[y * w + x]) continue;
                int x0 = x - kOccRadius, x1 = x + kOccRadius;
                int y0 = y - kOccRadius, y1 = y + kOccRadius;
                int areaPx = (std::min(x1, w - 1) - std::max(x0, 0) + 1)
                             * (std::min(y1, h - 1) - std::max(y0, 0) + 1);
                int occPx  = rectSum(x0, y0, x1, y1);
                float ratio = (areaPx > 0) ? (float)occPx / (float)areaPx : 0.0f;
                if (ratio > kOccThresh) {
                    isBoundary[y * w + x] = false;
                    dropped++;
                }
            }
        }
        std::cout << "[MaskCleanup] occupancy filter (R=" << kOccRadius
                  << "px, th=" << kOccThresh << "): dropped "
                  << dropped << " interior pseudo-boundary pixels" << std::endl;
    }

    g_boundaryDistMap.data.assign(w * h, -1.0f);
    g_boundaryDistMap.width  = w;
    g_boundaryDistMap.height = h;

    std::queue<std::pair<int,int>> q;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (isBoundary[y * w + x]) {
                g_boundaryDistMap.data[y * w + x] = 0.0f;
                q.push({x, y});
            }

    while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        float cd = g_boundaryDistMap.data[cy * w + cx];
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx[d], ny = cy + dy[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            if (!mask[ny * w + nx]) continue;
            int ni = ny * w + nx;
            if (g_boundaryDistMap.data[ni] < 0.0f) {
                g_boundaryDistMap.data[ni] = cd + 1.0f;
                q.push({nx, ny});
            }
        }
    }

    for (int i = 0; i < w * h; i++)
        if (g_boundaryDistMap.data[i] < 0.0f)
            g_boundaryDistMap.data[i] = 9999.0f;

    g_boundaryDistMap.valid = true;
    g_projectedLiverMask.invalidate();   /* SAM2 mask changed → old projection stale */

    int bdCount = 0;
    for (int i = 0; i < w * h; i++)
        if (isBoundary[i]) bdCount++;
    std::cout << "[Boundary] Map computed: " << w << "x" << h
              << " boundary_pixels=" << bdCount << "  (after cleanup)" << std::endl;
    return true;
}

inline bool ensureBoundaryMap() {
    if (g_boundaryDistMap.valid) return true;
    std::string maskPath = DEPTH_OUTPUT_PATH + "segmentation_mask.png";
    return loadMaskAndComputeBoundaryMap(maskPath);
}

// =========================================================================
//  Instrument mask loader (mirrors loadMaskAndComputeBoundaryMap).
// -------------------------------------------------------------------------
//  Difference from the liver-mask loader:
//      - Seeds the BFS from instrument pixels themselves (not from the
//        boundary), so distance == 0 INSIDE the instrument and grows
//        outward. This lets the registration pipeline reject vertices
//        whose reprojection is "close to or inside" any instrument.
//      - The mask is OPTIONAL: missing file is logged and returns false
//        without setting valid=true. extractTargetFromOBJ then runs as
//        before (no instrument exclusion).
// =========================================================================
inline bool loadInstrumentMaskAndComputeDistMap(const std::string& maskPath) {
    int w, h, ch;
    unsigned char* img = stbi_load(maskPath.c_str(), &w, &h, &ch, 1);
    if (!img) {
        std::cout << "[InstrumentMask] not found: " << maskPath
                  << "  (proceeding without instrument exclusion)" << std::endl;
        return false;
    }

    g_instrumentDistMap.data.assign((size_t)w * h, -1.0f);
    g_instrumentDistMap.width  = w;
    g_instrumentDistMap.height = h;

    std::queue<std::pair<int,int>> q;
    int seedCount = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (img[y * w + x] > 127) {
                g_instrumentDistMap.data[y * w + x] = 0.0f;
                q.push({x, y});
                seedCount++;
            }
        }
    }
    stbi_image_free(img);

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        float cd = g_instrumentDistMap.data[cy * w + cx];
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx[d], ny = cy + dy[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            int ni = ny * w + nx;
            if (g_instrumentDistMap.data[ni] < 0.0f) {
                g_instrumentDistMap.data[ni] = cd + 1.0f;
                q.push({nx, ny});
            }
        }
    }
    for (int i = 0; i < w * h; i++)
        if (g_instrumentDistMap.data[i] < 0.0f)
            g_instrumentDistMap.data[i] = 9999.0f;

    g_instrumentDistMap.valid = (seedCount > 0);
    std::cout << "[InstrumentMask] Map computed: " << w << "x" << h
              << "  instrument_pixels=" << seedCount
              << "  valid=" << (g_instrumentDistMap.valid ? "YES" : "NO (empty mask)")
              << std::endl;
    return g_instrumentDistMap.valid;
}

inline bool ensureInstrumentDistMap() {
    if (g_instrumentDistMap.valid) return true;
    std::string maskPath = DEPTH_OUTPUT_PATH + "instrument_segmentation_mask.png";
    return loadInstrumentMaskAndComputeDistMap(maskPath);
}

inline float getBoundaryDistForGridVertex(int vertIdx, int gridW, int gridH) {
    if (!g_boundaryDistMap.valid) return 9999.0f;
    int gx = vertIdx % (gridW + 1);
    int gy = vertIdx / (gridW + 1);
    float u = (float)gx / (float)gridW;
    float v = (float)gy / (float)gridH;
    int px = std::clamp((int)(u * g_boundaryDistMap.width),  0, g_boundaryDistMap.width  - 1);
    int py = std::clamp((int)(v * g_boundaryDistMap.height), 0, g_boundaryDistMap.height - 1);
    return g_boundaryDistMap.data[py * g_boundaryDistMap.width + px];
}

inline float computeAlphaZ(float bdist, float boundaryWidth,
                           float zWeightBoundary, float zWeightInterior) {
    float t = std::clamp(bdist / boundaryWidth, 0.0f, 1.0f);
    return zWeightBoundary + t * (zWeightInterior - zWeightBoundary);
}

inline void saveBoundaryMapVisualization(const std::string& outPath) {
    if (!g_boundaryDistMap.valid) return;
    int w = g_boundaryDistMap.width, h = g_boundaryDistMap.height;
    std::vector<unsigned char> rgb(w * h * 3);
    float maxDist = 0;
    for (auto d : g_boundaryDistMap.data)
        if (d < 9000.0f && d > maxDist) maxDist = d;
    if (maxDist < 1.0f) maxDist = 1.0f;
    for (int i = 0; i < w * h; i++) {
        float d = g_boundaryDistMap.data[i];
        if (d >= 9000.0f) {
            rgb[i*3] = rgb[i*3+1] = rgb[i*3+2] = 0;
        } else {
            float t = std::clamp(d / maxDist, 0.0f, 1.0f);
            rgb[i*3]   = (unsigned char)((1.0f - t) * 255);
            rgb[i*3+1] = 0;
            rgb[i*3+2] = (unsigned char)(t * 255);
        }
    }
    stbi_write_png(outPath.c_str(), w, h, 3, rgb.data(), w * 3);
    std::cout << "[Boundary] Saved: " << outPath << std::endl;
}

inline void saveBoundaryOverlayVisualization(const std::string& maskPath,
                                             const std::string& outPath) {
    int w, h, ch;
    unsigned char* img = stbi_load(maskPath.c_str(), &w, &h, &ch, 3);
    if (!img || !g_boundaryDistMap.valid) { if(img) stbi_image_free(img); return; }
    std::vector<unsigned char> out(w * h * 3);
    memcpy(out.data(), img, w * h * 3);
    stbi_image_free(img);
    for (int i = 0; i < w * h; i++) {
        if (g_boundaryDistMap.data[i] < 1.5f && g_boundaryDistMap.data[i] >= 0.0f) {
            out[i*3]   = 255;
            out[i*3+1] = 0;
            out[i*3+2] = 0;
        }
    }
    stbi_write_png(outPath.c_str(), w, h, 3, out.data(), w * 3);
    std::cout << "[Boundary] Overlay saved: " << outPath << std::endl;
}
