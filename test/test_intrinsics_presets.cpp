// test_intrinsics_presets.cpp — Step 5 (IntrinsicsPresets.h) unit check.
// Built/run by test/test_intrinsics_presets.sh. No GUI; pure CPU.
#include "IntrinsicsPresets.h"
#include <cstdio>
#include <cmath>
using namespace Reg3DCustom;

int main() {
    int fails = 0;

    // A. lookupPreset("azure_kinect_720p", K) -> K.fx == 918.234f
    CameraIntrinsics K;
    bool ok = lookupPreset("azure_kinect_720p", K);
    printf("[A] lookup(azure_kinect_720p)=%d fx=%.3f\n", ok, K.fx);
    if (!ok || std::fabs(K.fx - 918.234f) > 1e-3f) { printf("  FAIL A\n"); fails++; }

    // B. lookupPreset("nonexistent", K) -> false
    CameraIntrinsics K2;
    bool ok2 = lookupPreset("nonexistent", K2);
    printf("[B] lookup(nonexistent)=%d\n", ok2);
    if (ok2) { printf("  FAIL B\n"); fails++; }

    // C. da3_last is invalid before update, valid (and rewritten) after.
    CameraIntrinsics K3;
    bool before = lookupPreset("da3_last", K3);
    updateDynamicPreset("da3_last", CameraIntrinsics::k4a_color_1080p());
    bool after = lookupPreset("da3_last", K3);
    printf("[C] da3_last before=%d after=%d fx=%.2f\n", before, after, K3.fx);
    if (before || !after || std::fabs(K3.fx - 1377.35f) > 1e-2f) { printf("  FAIL C\n"); fails++; }

    printf("[info] preset count=%zu\n", presetRegistry().size());
    printf(fails == 0 ? "All preset tests passed\n" : "Preset tests FAILED\n");
    return fails == 0 ? 0 : 1;
}
