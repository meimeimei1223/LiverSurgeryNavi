#pragma once

#include <string>
#include <iostream>
#include <filesystem>

#include "AppContext.h"
#include "stb_image.h"
#include "Undistort.h"
#include "OBJTargetExtraction.h"
#include "PathConfig.h"

namespace ImageSession {

inline bool readDimensions(const std::string& path, int& w, int& h) {
    int channels = 0;
    if (!stbi_info(path.c_str(), &w, &h, &channels)) {
        std::cerr << "[ImageSession] stbi_info failed: " << path
                  << "  reason: " << stbi_failure_reason() << std::endl;
        return false;
    }
    return true;
}

// Resolve which file path the AR background and depth pipeline should use.
// If 'K' carries non-zero distortion, undistort 'srcPath' once and write the
// rectified result to 'depth_output/original_rectified.jpg'; the rectified
// path becomes the "live" image for everything downstream. If K has no
// distortion (or is invalid), the original path is returned unchanged.
//
// Returns the path to use for arBg.loadTexture() and the depth runner.
inline std::string maybeRectify(const std::string& srcPath,
                                const Reg3DCustom::CameraIntrinsics& K)
{
    if (!K.valid() || !K.hasDistortion()) return srcPath;

    const std::string outPath = DEPTH_OUTPUT_PATH + "original_rectified.jpg";

    // Make sure the output directory exists -- the depth runner does this
    // too, but image-load can run before any depth invocation.
    std::error_code ec;
    std::filesystem::create_directories(DEPTH_OUTPUT_PATH, ec);

    int outW = 0, outH = 0;
    if (!Undistort::undistortFile(srcPath, outPath, K, outW, outH)) {
        std::cerr << "[ImageSession] rectify failed; using original"
                  << std::endl;
        return srcPath;
    }
    std::cout << "[ImageSession] rectified -> " << outPath
              << "  (" << outW << "x" << outH << ")" << std::endl;
    return outPath;
}

// Full path-only load: backwards compatible -- callers that don't pass K get
// the previous behaviour (no rectification).
inline bool load(AppContext& ctx, const std::string& path) {
    int w = 0, h = 0;
    if (!readDimensions(path, w, h)) return false;
    if (!ctx.arBg.loadTexture(path)) return false;

    ctx.image.path   = path;
    ctx.image.width  = w;
    ctx.image.height = h;
    ctx.image.loaded = true;

    std::cout << "[ImageSession] Current mode: " << (int)ctx.mode << std::endl;

    AppMode oldMode = ctx.mode;
    ctx.mode = AppMode::kMaskSelection;
    ctx.maskPoints.clear();
    // Also drop any leftover instrument-mask points and reset the active
    // mask kind to Liver. Without this, dragging a new image onto the
    // window keeps the old prompts (and the active=Instrument flag),
    // which is both visually confusing and produces a stale instrument
    // mask if the user runs Run Depth straight after.
    ctx.instrumentMaskPoints.clear();
    ctx.activeMaskKind = MaskKind::Liver;
    std::cout << "[ImageSession] Changed to MaskSelection mode (from "
              << (int)oldMode << ")" << std::endl;

    std::cout << "[ImageSession] loaded " << path
              << "  (" << w << "x" << h << ")"
              << "  mode=" << (ctx.mode == AppMode::kMaskSelection ? "MaskSelection" :
                              ctx.mode == AppMode::kImageOnly ? "ImageOnly" : "Registration")
              << std::endl;
    return true;
}

// Load + rectify if K has distortion. The caller (e.g. main.cpp's drop
// handler or onLoadLocalImage callback) passes the active intrinsics. The
// raw input file stays on disk untouched; rectified pixels are written to
// 'depth_output/original_rectified.jpg' and that's what the AR background
// + depth runner see from this point on.
inline bool loadWithIntrinsics(AppContext& ctx,
                               const std::string& srcPath,
                               const Reg3DCustom::CameraIntrinsics& K)
{
    // If image dimensions disagree with K, skip rectification (the user is
    // either cross-loading an image from a different camera or hasn't set
    // intrinsics correctly). We still load the original so they can iterate.
    int w = 0, h = 0;
    if (!readDimensions(srcPath, w, h)) return false;

    std::string usePath = srcPath;
    if (K.valid() && K.hasDistortion()) {
        if (w == K.width && h == K.height) {
            usePath = maybeRectify(srcPath, K);
        } else {
            std::cerr << "[ImageSession] image " << w << "x" << h
                      << " mismatches K " << K.width << "x" << K.height
                      << " -- skipping rectification" << std::endl;
        }
    }

    if (!ctx.arBg.loadTexture(usePath)) return false;

    // Re-read dimensions in case rectification produced a differently sized
    // file (currently it doesn't, but stay defensive).
    int rw = 0, rh = 0;
    if (!readDimensions(usePath, rw, rh)) { rw = w; rh = h; }

    ctx.image.path   = usePath;       // <-- downstream sees rectified path
    ctx.image.width  = rw;
    ctx.image.height = rh;
    ctx.image.loaded = true;

    AppMode oldMode = ctx.mode;
    ctx.mode = AppMode::kMaskSelection;
    ctx.maskPoints.clear();
    // Also drop any leftover instrument-mask points and reset the active
    // mask kind to Liver. See note above (load()) for rationale.
    ctx.instrumentMaskPoints.clear();
    ctx.activeMaskKind = MaskKind::Liver;
    std::cout << "[ImageSession] Changed to MaskSelection mode (from "
              << (int)oldMode << ")" << std::endl;

    std::cout << "[ImageSession] loaded " << usePath
              << "  (" << rw << "x" << rh << ")"
              << (usePath == srcPath ? "" : "  [RECTIFIED]")
              << std::endl;
    return true;
}

inline void clear(AppContext& ctx) {
    ctx.image = ImageSessionState{};
    if (ctx.mode == AppMode::kImageOnly) {
        ctx.mode = AppMode::kEmpty;
    }
}

inline bool isSupportedExtension(const std::string& path) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    return (ext == ".png"  || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".bmp"  || ext == ".ppm");
}

}
