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
//      Auto    : delegate to SAM2/DA3            (last resort)
//
//  Legacy int mapping (old 4-button UI / RegUIState::intrinsicsSource, and any
//  serialized state) used 0=DA3, 1=Kinect, 2=Custom, 3=Calib. The two helpers
//  below bridge that numbering to/from the enum so the old UI keeps working and
//  ImGui (which speaks int) can pass values across the boundary.
// =============================================================================

enum class IntrinsicsSource {
    Custom = 0,   // intrinsics_custom.txt        (priority 1, highest)
    Calib  = 1,   // Zhang calibration result      (priority 2)
    Preset = 2,   // built-in preset               (priority 3)
    Auto   = 3    // SAM2/DA3 delegate             (priority 4, lowest)
};

// Old UI int (0=DA3, 1=Kinect, 2=Custom, 3=Calib) -> enum.
//   0 (DA3)    -> Auto
//   1 (Kinect) -> Preset   (the azure_kinect_720p preset; caller sets the key)
//   2 (Custom) -> Custom
//   3 (Calib)  -> Calib
inline IntrinsicsSource intrinsicsSourceFromLegacyInt(int v) {
    switch (v) {
        case 0:  return IntrinsicsSource::Auto;
        case 1:  return IntrinsicsSource::Preset;
        case 2:  return IntrinsicsSource::Custom;
        case 3:  return IntrinsicsSource::Calib;
        default: return IntrinsicsSource::Auto;
    }
}

// enum -> old UI int. Inverse of intrinsicsSourceFromLegacyInt.
inline int intrinsicsSourceToLegacyInt(IntrinsicsSource s) {
    switch (s) {
        case IntrinsicsSource::Auto:   return 0;
        case IntrinsicsSource::Preset: return 1;
        case IntrinsicsSource::Custom: return 2;
        case IntrinsicsSource::Calib:  return 3;
    }
    return 0;
}
