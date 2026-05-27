#pragma once
// =============================================================================
//  DepthToObjExport.h  (obj-migration Phase 3)
//  ---------------------------------------------------------------------------
//  REG-side wrapper that takes a metric depth map + RGB + mask + REG's own K and
//  writes the OBJ / intrinsics.txt / texture artifacts that sam2_da3_lite used to
//  produce. K is owned by REG; sam2 no longer round-trips it.
//
//  Canonical (tag-less) names are what DEFORM and other consumers read. Tagged
//  copies (`_<tag>`) are debug / cross-K comparison outputs (compared against the
//  parallel sam2 output during Phase 3).
//
//  External-OBJ-injection guard (FINAL §2.7): the *decision* to call
//  exportDepthArtifacts() lives in the caller (REG Run-Depth hook). This module
//  only writes when asked; loadDepthMetricBin() lets the caller probe whether a
//  metric-depth handoff exists.
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>
#include "obj_exporter.hpp"
#include "OBJTargetExtraction.h"   // Reg3DCustom::CameraIntrinsics

namespace depthexport {

struct Request {
    // --- input data ---
    const uint8_t*               rgbPixels = nullptr;  // W*H*3, RGB row-major
    int                          width = 0;
    int                          height = 0;
    const std::vector<float>*    depthMetric = nullptr; // size W*H, meters
    const std::vector<uint8_t>*  mask = nullptr;         // size W*H, 0/255
    Reg3DCustom::CameraIntrinsics K;                      // REG's K (== depth res)
    std::string                  tag;                     // intrinsicsSourceToTag()

    // --- output dir (DEPTH_OUTPUT_PATH; trailing slash tolerated) ---
    std::string                  outDir;

    // --- options (geometry defaults mirror sam2_da3_lite's: skirt 0.05,
    //     dilate 2, stride 10, metric scale 1.0) ---
    float skirtThreshold       = 0.05f;
    int   fullMeshStride       = 10;
    int   maskDilate           = 2;
    // Canonical-only policy: REG writes a single tag-less OBJ set. "K is owned by
    // REG; one correct K at a time" -> tagged copies are not produced (provenance
    // lives in intrinsics.txt's name field). These remain as opt-in debug knobs
    // (default off); the implementation is retained but not invoked by REG.
    bool  writeTaggedCopies    = false;
    bool  writeNoSkirtVariants = false;
    bool  writeTextureImage    = true;
};

struct Result {
    bool ok = false;
    std::string canonicalMaskedObj;   // pc_metric_pinhole_masked.obj
    std::string canonicalFullObj;     // pc_metric_pinhole_full_light.obj
    std::string canonicalIntrinsics;  // intrinsics.txt
    std::string canonicalTexture;     // texture.png
    std::vector<std::string> debugCopies;
};

// Write canonical + tagged OBJ/intrinsics/texture from depth+mask+rgb+K.
Result exportDepthArtifacts(const Request& req);

// Write a single intrinsics txt (fx/fy/cx/cy/width/height/name + distortion if
// nonzero). Usable standalone for the bin-absent / external-OBJ path.
bool saveIntrinsicsFile(const std::string& path,
                        const Reg3DCustom::CameraIntrinsics& K,
                        int width, int height);

// depth_metric.bin reader. Validates the 16-byte "DEPT" header.
struct DepthBin {
    int width = 0, height = 0;
    std::vector<float> depth;
    bool valid() const { return width > 0 && height > 0 && !depth.empty(); }
};
bool loadDepthMetricBin(const std::string& path, DepthBin& out);

} // namespace depthexport
