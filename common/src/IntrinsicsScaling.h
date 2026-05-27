#pragma once
// =============================================================================
//  IntrinsicsScaling.h
//  ---------------------------------------------------------------------------
//  Scale a pinhole CameraIntrinsics to a target image resolution. Used when the
//  K's calibration resolution differs from the resolution of the depth/image it
//  is being applied to (e.g. a 4K Custom K used to unproject a 1080p depth map
//  that sam2 downscaled). fx/fy/cx/cy scale linearly with the resolution ratio;
//  distortion coefficients (k1..p2) are defined in NORMALIZED image coordinates
//  and are therefore resolution-invariant -- they are NOT scaled.
//
//  (FEATURE_PLAN_external_depth_drop.md Task 1. Also applied to the Run Depth
//   path so the "K-res != depth-res" limitation noted in the obj-migration
//   STATUS is resolved.)
// =============================================================================
#include "OBJTargetExtraction.h"   // Reg3DCustom::CameraIntrinsics

namespace Reg3DCustom {

inline CameraIntrinsics scaleIntrinsics(const CameraIntrinsics& K,
                                        int targetWidth,
                                        int targetHeight) {
    if (K.width <= 0 || K.height <= 0) return K;            // K invalid -> no-op
    if (targetWidth <= 0 || targetHeight <= 0) return K;    // bad target -> no-op
    if (K.width == targetWidth && K.height == targetHeight) return K;  // already matches

    const float sx = (float)targetWidth  / (float)K.width;
    const float sy = (float)targetHeight / (float)K.height;

    CameraIntrinsics K2 = K;
    K2.fx     = K.fx * sx;
    K2.fy     = K.fy * sy;
    K2.cx     = K.cx * sx;
    K2.cy     = K.cy * sy;
    K2.width  = targetWidth;
    K2.height = targetHeight;
    // k1..k4, p1, p2 unchanged (normalized-coordinate coefficients).
    return K2;
}

} // namespace Reg3DCustom
