#pragma once

#include "image_utils.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace ply {

struct CameraIntrinsics {
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;
    bool valid() const { return fx > 0.0f && fy > 0.0f; }
};

enum class Projection { PlaneRelief, Pinhole, PinholeMetric };
enum class Normalize  { None, PercentileMasked };
enum class MaskMode   { IgnoreMask, SkipOutside, KeepOutsideFlat };

struct ExportOptions {
    Projection projection  = Projection::PlaneRelief;
    Normalize  normalize   = Normalize::PercentileMasked;
    MaskMode   maskMode    = MaskMode::KeepOutsideFlat;
    CameraIntrinsics intrinsics;
    float fovDeg      = 60.0f;
    float depthScale  = 1.0f;
    float thickness   = 0.05f;
    bool  invertDepth = false;
    bool  binary      = true;
    bool  flipY       = true;

    const std::vector<float>* confidence = nullptr;
    float confidenceMin = 0.0f;
};

bool saveTexturedPly(
    const std::string& outPath,
    const img::Image& rgbImage,
    const std::vector<float>& depthRaw,
    const std::vector<uint8_t>& mask,
    const ExportOptions& opt = {}
);

}
