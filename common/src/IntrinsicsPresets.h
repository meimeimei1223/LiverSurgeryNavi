#pragma once
// =============================================================================
//  IntrinsicsPresets.h
//  ---------------------------------------------------------------------------
//  Built-in camera-intrinsics presets, looked up by string key. Factory
//  presets reuse the CameraIntrinsics::*() factories in OBJTargetExtraction.h.
//
//  The registry is intentionally minimal (the UI is matched to what is actually
//  in use). Add cameras as needed; commented-out examples below show the format
//  and are kept for easy revival. The DA3 estimate is NOT a preset anymore -- it
//  is a file-driven source (intrinsics_da3_last.txt), symmetric with Custom and
//  Calib (see IntrinsicsSource.h / autoSelectIntrinsicsSource).
// =============================================================================
#include "OBJTargetExtraction.h"   // Reg3DCustom::CameraIntrinsics + factories
#include <string>
#include <vector>

namespace Reg3DCustom {

struct IntrinsicsPreset {
    const char*      key;            // internal id (alnum + underscore)
    const char*      displayName;    // UI label
    CameraIntrinsics K;
};

// Singleton registry. Declaration order = dropdown order.
inline std::vector<IntrinsicsPreset>& presetRegistry() {
    static std::vector<IntrinsicsPreset> reg = {
        // === Factory presets ===
        // 必要に応じてここに追加。各プリセットは:
        //   { "internal_key", "Display Name", CameraIntrinsics value }
        // 例 (コメントアウトで残す、将来復活用):
        //   { "azure_kinect_720p",   "Azure Kinect 720p",
        //     CameraIntrinsics::k4a_color_720p()  },
        //   { "realsense_d435_720p", "RealSense D435 720p",
        //     CameraIntrinsics::d435_color_720p() },
        //   { "logitech_c920_1080p", "Logitech C920 1080p",
        //     { 1394.6f, 1394.6f, 956.0f, 540.0f, 1920, 1080, "logitech_c920" } },
        //   { "iphone_12_wide",      "iPhone 12 (wide)",
        //     { 1530.0f, 1530.0f, 1512.0f, 2016.0f, 3024, 4032, "iphone_12" } },

        { "azure_kinect_1080p", "Azure Kinect 1080p",
          CameraIntrinsics::k4a_color_1080p() },
    };
    return reg;
}

// Look up a preset by key. Returns false if the key is unknown OR the matched
// preset's K is not valid.
inline bool lookupPreset(const std::string& key, CameraIntrinsics& out) {
    for (auto& p : presetRegistry()) {
        if (key == p.key) { out = p.K; return p.K.valid(); }
    }
    return false;
}

} // namespace Reg3DCustom
