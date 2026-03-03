#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "MeshDrawing.h"   // setUp()

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

    // get BB from all vertices
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

    // 3D -> normalized (0~1)
    float u = (hitPos3D.x - minX) / rangeX;   // L->R
    float v = (maxY - hitPos3D.y) / rangeY;   // T->B (Y flip)

    // -> pixel
    int imgW = mesh->loadedImageWidth;
    int imgH = mesh->loadedImageHeight;
    outPixelX = static_cast<int>(u * imgW);
    outPixelY = static_cast<int>(v * imgH);

    // clamp
    outPixelX = std::max(0, std::min(outPixelX, imgW - 1));
    outPixelY = std::max(0, std::min(outPixelY, imgH - 1));

    // debug
    std::cout << "[SegPoint] 3D hit: (" << hitPos3D.x << ", " << hitPos3D.y << ", " << hitPos3D.z << ")" << std::endl;
    std::cout << "[SegPoint] BB: X[" << minX << ", " << maxX << "] Y[" << minY << ", " << maxY << "]" << std::endl;
    std::cout << "[SegPoint] u=" << u << " v=" << v << " -> 2D: (" << outPixelX << ", " << outPixelY << ")" << std::endl;

    return true;
}

// =========================================================================
//  regenerateDepthMesh
//    Rebuild screenMesh geometry with updated depth scale
// =========================================================================
inline void regenerateDepthMesh(mCutMesh* mesh, float depthScale, float meshScale) {
    if (!mesh || mesh->depthImageData.empty()) {
        std::cout << "[DepthScale] No depth data loaded" << std::endl;
        return;
    }
    int gw = gGridWidth;
    int gh = gw * mesh->loadedImageHeight / mesh->loadedImageWidth;
    auto depths = mesh->calculateNormalizedDepth(gw, gh, 0.99f, 0.9f);
    mesh->generateGridPlaneWithDepth(gw, gh, depths, 0.05f, depthScale);
    for (size_t i = 0; i < mesh->mVertices.size(); i++)
        mesh->mVertices[i] *= meshScale;
    setUp(*mesh);
    std::cout << "[DepthScale] depthScale=" << depthScale << std::endl;
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
};

inline BoundaryDistMap g_boundaryDistMap;

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

    int bdCount = 0;
    for (int i = 0; i < w * h; i++)
        if (isBoundary[i]) bdCount++;
    std::cout << "[Boundary] Map computed: " << w << "x" << h
              << " boundary_pixels=" << bdCount << std::endl;
    return true;
}

inline bool ensureBoundaryMap() {
    if (g_boundaryDistMap.valid) return true;
    std::string maskPath = DEPTH_OUTPUT_PATH + "segmentation_mask.png";
    return loadMaskAndComputeBoundaryMap(maskPath);
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
