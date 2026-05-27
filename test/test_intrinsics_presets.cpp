// test_intrinsics_presets.cpp — IntrinsicsPresets.h unit check.
// Built/run by test/test_intrinsics_presets.sh. No GUI; pure CPU.
#include "IntrinsicsPresets.h"
#include <cstdio>
#include <cmath>
using namespace Reg3DCustom;

int main() {
    int fails = 0;

    // A. lookupPreset("azure_kinect_1080p", K) -> K.fx == 1377.35f
    CameraIntrinsics K;
    bool ok = lookupPreset("azure_kinect_1080p", K);
    printf("[A] lookup(azure_kinect_1080p)=%d fx=%.3f res=%dx%d\n",
           ok, K.fx, K.width, K.height);
    if (!ok || std::fabs(K.fx - 1377.35f) > 1e-2f) { printf("  FAIL A\n"); fails++; }

    // B. lookupPreset("nonexistent", K) -> false
    CameraIntrinsics K2;
    bool ok2 = lookupPreset("nonexistent", K2);
    printf("[B] lookup(nonexistent)=%d\n", ok2);
    if (ok2) { printf("  FAIL B\n"); fails++; }

    // C. da3_last is no longer a preset (now file-driven). Must NOT resolve.
    CameraIntrinsics K3;
    bool ok3 = lookupPreset("da3_last", K3);
    printf("[C] lookup(da3_last)=%d (expect 0: DA3 is file-driven, not a preset)\n", ok3);
    if (ok3) { printf("  FAIL C\n"); fails++; }

    // D. Registry is minimal: exactly 1 factory preset.
    size_t n = presetRegistry().size();
    printf("[D] preset count=%zu (expect 1)\n", n);
    if (n != 1) { printf("  FAIL D\n"); fails++; }

    printf(fails == 0 ? "All preset tests passed\n" : "Preset tests FAILED\n");
    return fails == 0 ? 0 : 1;
}
