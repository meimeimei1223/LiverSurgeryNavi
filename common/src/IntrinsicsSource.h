#pragma once
// =============================================================================
//  IntrinsicsSource.h
//  ---------------------------------------------------------------------------
//  Camera-intrinsics source selector, shared between main.cpp (which owns the
//  g_intrinsicsSource global + autoSelect logic) and RegistrationImGuiManager.h
//  (which renders the source dropdown). Kept in a tiny standalone header so the
//  UI layer can name the type without pulling in main.cpp.
//
//  Priority order (declaration order = highest priority first):
//      Custom  : intrinsics_custom.txt          (user-explicit, highest)
//      Calib   : Zhang calibration result        (intrinsics_calib_last.txt)
//      Preset  : built-in camera preset          (g_currentPresetKey)
//      DA3     : DA3 推定結果 (intrinsics_da3_last.txt) を使用  (last resort)
//
//  All four sources are file-driven and symmetric: DA3, like Custom and Calib,
//  loads its K from a txt file. intrinsics_da3_last.txt is written after a
//  Run Depth (the DA3 estimate that SAM2 always produces); if it is absent the
//  DA3 source is unavailable (run depth once to populate it).
//
//  Legacy int mapping (old 4-button UI / RegUIState::intrinsicsSource, and any
//  serialized state) used 0=DA3, 1=Kinect, 2=Custom, 3=Calib. The two helpers
//  below bridge that numbering to/from the enum so the old UI keeps working and
//  ImGui (which speaks int) can pass values across the boundary. (DA3 maps to
//  legacy 0, matching the old "DA3" button.)
// =============================================================================

enum class IntrinsicsSource {
    Custom = 0,   // intrinsics_custom.txt         (priority 1, highest)
    Calib  = 1,   // Zhang calibration result       (priority 2)
    Preset = 2,   // built-in preset                (priority 3)
    DA3    = 3    // intrinsics_da3_last.txt         (priority 4, lowest)
};

// Old UI int (0=DA3, 1=Kinect, 2=Custom, 3=Calib) -> enum.
//   0 (DA3)    -> DA3
//   1 (Kinect) -> Preset   (caller sets the preset key)
//   2 (Custom) -> Custom
//   3 (Calib)  -> Calib
inline IntrinsicsSource intrinsicsSourceFromLegacyInt(int v) {
    switch (v) {
        case 0:  return IntrinsicsSource::DA3;
        case 1:  return IntrinsicsSource::Preset;
        case 2:  return IntrinsicsSource::Custom;
        case 3:  return IntrinsicsSource::Calib;
        default: return IntrinsicsSource::DA3;
    }
}

// enum -> old UI int. Inverse of intrinsicsSourceFromLegacyInt.
inline int intrinsicsSourceToLegacyInt(IntrinsicsSource s) {
    switch (s) {
        case IntrinsicsSource::DA3:    return 0;
        case IntrinsicsSource::Preset: return 1;
        case IntrinsicsSource::Custom: return 2;
        case IntrinsicsSource::Calib:  return 3;
    }
    return 0;
}

// Filename tag for OBJ / intrinsics outputs (obj-migration Phase 3).
//   Custom -> "custom", Calib -> "calib", Preset -> the preset key (e.g. "k4a"),
//   DA3 -> "da3". Stored in CameraIntrinsics::name and used as the <tag> in
//   pc_metric_pinhole_*_<tag>*.obj / intrinsics_<tag>.txt debug copies.
#include <string>
inline std::string intrinsicsSourceToTag(IntrinsicsSource s,
                                         const std::string& presetKey = "k4a") {
    switch (s) {
        case IntrinsicsSource::Custom: return "custom";
        case IntrinsicsSource::Calib:  return "calib";
        case IntrinsicsSource::Preset: return presetKey;
        case IntrinsicsSource::DA3:    return "da3";
    }
    return "unknown";
}
