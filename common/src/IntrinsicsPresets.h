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
        //   azure_kinect_720p: ユーザー要請により無効化。本機の運用は 4K + custom
        //   (Patient3 のカメラは 4K factory 校正 intrinsics_custom.txt を使用)。
        //   { "azure_kinect_720p",   "Azure Kinect 720p",
        //     CameraIntrinsics::k4a_color_720p()  },
        //   { "realsense_d435_720p", "RealSense D435 720p",
        //     CameraIntrinsics::d435_color_720p() },
        //   { "logitech_c920_1080p", "Logitech C920 1080p",
        //     { 1394.6f, 1394.6f, 956.0f, 540.0f, 1920, 1080, "logitech_c920" } },
        //   { "iphone_12_wide",      "iPhone 12 (wide)",
        //     { 1530.0f, 1530.0f, 1512.0f, 2016.0f, 3024, 4032, "iphone_12" } },

        // 注意: k4a_color_1080p の値 (fx=1377.35, cx=960.228, ...) は Azure Kinect
        // の「一般的な」公称値であり、個体ごとの per-device 校正値ではない。
        // 例えば Patient3 のカメラの実測値は 4K で fx=2606.48, cx=1925.19
        // (= 1080p 換算 1303.24, 962.60) で、この一般プリセットとは一致しない。
        // このプリセットは「校正ファイル(custom/calib)が不明なときの暫定値」用途。
        // 正確な投影が必要な個体は intrinsics_custom.txt (Custom source) を使うこと。
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
