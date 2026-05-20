// =============================================================================
//  Undistort.h
//  ---------------------------------------------------------------------------
//  Brown-Conrady (radial + tangential) lens-distortion correction for RGB
//  images, with helpers to load/save through stb_image.
//
//  Math (OpenCV convention, normalized image coords x' = (u-cx)/fx, ...):
//      r^2 = x'^2 + y'^2
//      radial = 1 + k1*r^2 + k2*r^4 + k3*r^6 + k4*r^8
//      x_d = x' * radial + 2*p1*x'*y'   + p2*(r^2 + 2*x'^2)
//      y_d = y' * radial + p1*(r^2 + 2*y'^2) + 2*p2*x'*y'
//      u_d = fx * x_d + cx,    v_d = fy * y_d + cy
//
//  Algorithm (per output rectified pixel):
//      1. (u_und, v_und) -> (x', y')
//      2. apply distortion -> (x_d, y_d)
//      3. (x_d, y_d) -> (u_d, v_d) in source
//      4. bilinear sample source[u_d, v_d]
//
//  Note: this maps DESTINATION (rectified) -> SOURCE (distorted), which is
//  the standard "inverse warp" formulation used by cv::undistort. Pixels that
//  fall outside the source image are written as black (could optionally be
//  marked transparent or filled).
// =============================================================================

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>

#include "stb_image.h"           // for stbi_load / stbi_image_free
#include "stb_image_write.h"     // for stbi_write_jpg
#include "OBJTargetExtraction.h" // Reg3DCustom::CameraIntrinsics

namespace Undistort {

// Bilinear sample of a 3-channel uint8 image; returns 0 (black) if out of range.
inline void sampleBilinearRGB(const uint8_t* src, int W, int H,
                              float u, float v, uint8_t out[3])
{
    if (u < 0.0f || v < 0.0f || u > (float)(W - 1) || v > (float)(H - 1)) {
        out[0] = out[1] = out[2] = 0;
        return;
    }
    int   x0 = (int)std::floor(u);
    int   y0 = (int)std::floor(v);
    int   x1 = std::min(x0 + 1, W - 1);
    int   y1 = std::min(y0 + 1, H - 1);
    float fx = u - (float)x0;
    float fy = v - (float)y0;

    const uint8_t* p00 = src + (y0 * W + x0) * 3;
    const uint8_t* p10 = src + (y0 * W + x1) * 3;
    const uint8_t* p01 = src + (y1 * W + x0) * 3;
    const uint8_t* p11 = src + (y1 * W + x1) * 3;

    for (int c = 0; c < 3; c++) {
        float a = p00[c] * (1 - fx) + p10[c] * fx;
        float b = p01[c] * (1 - fx) + p11[c] * fx;
        float v01 = a * (1 - fy) + b * fy;
        if (v01 < 0)     v01 = 0;
        if (v01 > 255.f) v01 = 255.f;
        out[c] = (uint8_t)(v01 + 0.5f);
    }
}

// Apply Brown-Conrady undistortion to an interleaved RGB buffer.
// Output buffer is written in-place into 'dst' (must be pre-sized W*H*3).
// Source and destination have the same dimensions (W = K.width, H = K.height).
inline bool undistortRGB(const uint8_t* src, int srcW, int srcH,
                         const Reg3DCustom::CameraIntrinsics& K,
                         std::vector<uint8_t>& dst)
{
    if (!K.valid()) {
        std::cerr << "[Undistort] invalid intrinsics" << std::endl;
        return false;
    }
    if (srcW != K.width || srcH != K.height) {
        std::cerr << "[Undistort] image size " << srcW << "x" << srcH
                  << " != intrinsics " << K.width << "x" << K.height
                  << " -- aborting" << std::endl;
        return false;
    }

    const float fx = K.fx, fy = K.fy;
    const float cx = K.cx, cy = K.cy;
    const float k1 = K.k1, k2 = K.k2, k3 = K.k3, k4 = K.k4;
    const float p1 = K.p1, p2 = K.p2;

    dst.assign((size_t)srcW * srcH * 3, 0);

    for (int v = 0; v < srcH; v++) {
        for (int u = 0; u < srcW; u++) {
            // 1. Normalized rectified coords
            float xp = ((float)u - cx) / fx;
            float yp = ((float)v - cy) / fy;
            float r2 = xp * xp + yp * yp;
            float r4 = r2 * r2;
            float r6 = r4 * r2;
            float r8 = r4 * r4;

            // 2. Apply distortion (rectified -> distorted normalized)
            float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
            float xd = xp * radial + 2.0f * p1 * xp * yp + p2 * (r2 + 2.0f * xp * xp);
            float yd = yp * radial + p1 * (r2 + 2.0f * yp * yp) + 2.0f * p2 * xp * yp;

            // 3. Back to pixel coords in source
            float us = fx * xd + cx;
            float vs = fy * yd + cy;

            // 4. Sample
            uint8_t* outPix = dst.data() + ((size_t)v * srcW + u) * 3;
            sampleBilinearRGB(src, srcW, srcH, us, vs, outPix);
        }
    }
    return true;
}

// Write an RGB buffer as JPEG (quality 95). Returns true on success.
inline bool saveJPEG(const std::string& path,
                     const uint8_t* data, int W, int H, int quality = 95)
{
    int rc = stbi_write_jpg(path.c_str(), W, H, 3, data, quality);
    if (!rc) {
        std::cerr << "[Undistort] saveJPEG failed: " << path << std::endl;
        return false;
    }
    std::cout << "[Undistort] wrote " << path
              << " (" << W << "x" << H << ")" << std::endl;
    return true;
}

// Convenience: undistort src image (file path) and write rectified to dst path.
// Falls back to a plain copy when K has no distortion (k1..k4, p1, p2 all ~ 0).
// Returns true on success. Sets outW/outH to the output image size (= input size).
inline bool undistortFile(const std::string& srcPath,
                          const std::string& dstPath,
                          const Reg3DCustom::CameraIntrinsics& K,
                          int& outW, int& outH)
{
    int w = 0, h = 0, ch = 0;
    uint8_t* src = ::stbi_load(srcPath.c_str(), &w, &h, &ch, 3);
    if (!src) {
        std::cerr << "[Undistort] cannot load: " << srcPath << std::endl;
        return false;
    }
    outW = w; outH = h;

    if (!K.hasDistortion()) {
        std::cout << "[Undistort] K has no distortion -- copying as-is"
                  << std::endl;
        bool ok = saveJPEG(dstPath, src, w, h, 95);
        ::stbi_image_free(src);
        return ok;
    }

    if (w != K.width || h != K.height) {
        std::cerr << "[Undistort] image " << w << "x" << h
                  << " != K " << K.width << "x" << K.height << std::endl;
        ::stbi_image_free(src);
        return false;
    }

    std::vector<uint8_t> dst;
    bool ok = undistortRGB(src, w, h, K, dst);
    ::stbi_image_free(src);
    if (!ok) return false;

    return saveJPEG(dstPath, dst.data(), w, h, 95);
}

} // namespace Undistort
