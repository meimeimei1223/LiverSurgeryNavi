#pragma once
// =============================================================================
//  IntrinsicsPresets.h
//  ---------------------------------------------------------------------------
//  Built-in camera-intrinsics presets, looked up by string key. Factory
//  presets reuse the CameraIntrinsics::*() factories in OBJTargetExtraction.h;
//  the rest are inline literals. One "dynamic" preset (da3_last) starts invalid
//  and is filled in at runtime via updateDynamicPreset() after a DA3 run.
//
//  Used by the Source dropdown (UI), autoSelect, and Run Depth tag dispatch.
// =============================================================================
#include "OBJTargetExtraction.h"   // Reg3DCustom::CameraIntrinsics + factories
#include <string>
#include <vector>

namespace Reg3DCustom {

struct IntrinsicsPreset {
    const char*      key;            // internal id (alnum + underscore)
    const char*      displayName;    // UI label
    CameraIntrinsics K;
    bool             isDynamic = false;  // true => rewritten at runtime (da3_last)
};

// Mutable singleton registry (mutable so updateDynamicPreset can rewrite the
// dynamic entries' K in place). Declaration order = dropdown order = priority.
inline std::vector<IntrinsicsPreset>& presetRegistry() {
    static std::vector<IntrinsicsPreset> reg = {
        // ---- Factory presets (reuse OBJTargetExtraction.h factories) ----
        { "azure_kinect_720p",   "Azure Kinect 720p",   CameraIntrinsics::k4a_color_720p()  },
        { "azure_kinect_1080p",  "Azure Kinect 1080p",  CameraIntrinsics::k4a_color_1080p() },
        { "realsense_d435_720p", "RealSense D435 720p", CameraIntrinsics::d435_color_720p() },
        { "realsense_d435_480p", "RealSense D435 480p", CameraIntrinsics::d435_color_480p() },
        { "realsense_d455_720p", "RealSense D455 720p", CameraIntrinsics::d455_color_720p() },
        // ---- Inline-literal presets (fx, fy, cx, cy, width, height, name) ----
        { "logitech_c920_1080p", "Logitech C920 1080p",
          { 1394.6f, 1394.6f, 956.0f, 540.0f, 1920, 1080, "logitech_c920_1080p" } },
        { "iphone_12_wide",      "iPhone 12 (wide)",
          { 1530.0f, 1530.0f, 1512.0f, 2016.0f, 3024, 4032, "iphone_12_wide" } },
        { "generic_webcam_720p", "Generic Webcam 720p",
          { 918.0f, 918.0f, 640.0f, 360.0f, 1280, 720, "generic_webcam_720p" } },
        // ---- Dynamic (filled in at runtime) ----
        { "da3_last",            "DA3 estimated (last)", {}, true },
    };
    return reg;
}

// Look up a preset by key. Returns false if the key is unknown OR the matched
// preset's K is not yet valid (e.g. da3_last before its first run).
inline bool lookupPreset(const std::string& key, CameraIntrinsics& out) {
    for (auto& p : presetRegistry()) {
        if (key == p.key) { out = p.K; return p.K.valid(); }
    }
    return false;
}

// Rewrite a dynamic preset's K (no-op for non-dynamic or unknown keys).
inline void updateDynamicPreset(const std::string& key, const CameraIntrinsics& K) {
    for (auto& p : presetRegistry()) {
        if (p.isDynamic && key == p.key) { p.K = K; return; }
    }
}

} // namespace Reg3DCustom
