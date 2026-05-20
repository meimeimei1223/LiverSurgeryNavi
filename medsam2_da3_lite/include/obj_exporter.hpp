#pragma once

#include "image_utils.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace objexp {

struct CameraIntrinsics {
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;
    bool valid() const { return fx > 0.0f && fy > 0.0f; }
};

enum class ZMode {
    Metric,
    Negated,
    HillInverted
};

enum class Projection {
    PinholeMetric,
    PlaneRelief
};

struct ObjExportOptions {
    Projection projection = Projection::PinholeMetric;
    CameraIntrinsics intrinsics;
    float depthScale     = 1.0f;
    float thickness      = 0.05f;
    bool  flipY          = true;
    bool  invertDepth    = false;
    ZMode zMode          = ZMode::Metric;
    float skirtThreshold = 0.0f;
    bool  writeTexture   = true;
    std::string textureFilename = "texture.png";
    std::string materialName    = "screenMat";

    int   stride          = 1;   // 1=full res, 10=1/10 (saveFullMeshObj only)

    const std::vector<float>* confidence = nullptr;
    float confidenceMin = 0.0f;
};

bool saveFullMeshObj(
    const std::string& outPath,
    const img::Image& rgbImage,
    const std::vector<float>& depthMetric,
    const ObjExportOptions& opt = {}
);

bool saveMaskedMeshObj(
    const std::string& outPath,
    const img::Image& rgbImage,
    const std::vector<float>& depthMetric,
    const std::vector<uint8_t>& mask,
    const ObjExportOptions& opt = {}
);

}
