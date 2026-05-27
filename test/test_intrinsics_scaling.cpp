// test_intrinsics_scaling.cpp — IntrinsicsScaling.h unit check (FEATURE Task 1).
// Built/run by test/test_intrinsics_scaling.sh. No GUI; pure CPU.
#include "IntrinsicsScaling.h"
#include <cstdio>
#include <cmath>
using namespace Reg3DCustom;

int main() {
    int fails = 0;

    // Patient3 4K factory K.
    CameraIntrinsics K4k;
    K4k.fx = 2606.4793f; K4k.fy = 2606.4793f;
    K4k.cx = 1925.1948f; K4k.cy = 1038.7789f;
    K4k.width = 3840; K4k.height = 2160;
    K4k.k1 = -0.0947043f; K4k.k2 = -0.0690935f; K4k.k3 = -0.0107949f;

    // A. 4K -> 1080p (factor 0.5): fx/cx halve, distortion unchanged.
    CameraIntrinsics K1080 = scaleIntrinsics(K4k, 1920, 1080);
    printf("[A] 4K->1080p fx=%.4f cx=%.4f cy=%.4f res=%dx%d k1=%.7f\n",
           K1080.fx, K1080.cx, K1080.cy, K1080.width, K1080.height, K1080.k1);
    if (std::fabs(K1080.fx - 1303.2396f) > 1e-2f ||
        std::fabs(K1080.cx - 962.5974f)  > 1e-2f ||
        std::fabs(K1080.cy - 519.3894f)  > 1e-2f ||
        K1080.width != 1920 || K1080.height != 1080 ||
        std::fabs(K1080.k1 - K4k.k1) > 1e-9f) { printf("  FAIL A\n"); fails++; }

    // B. roundtrip 1080p -> 4K -> 1080p: error < 1e-3 px (integer factor -> exact).
    CameraIntrinsics back = scaleIntrinsics(scaleIntrinsics(K1080, 3840, 2160), 1920, 1080);
    printf("[B] roundtrip fx=%.6f cx=%.6f (orig fx=%.6f cx=%.6f)\n",
           back.fx, back.cx, K1080.fx, K1080.cx);
    if (std::fabs(back.fx - K1080.fx) > 1e-3f || std::fabs(back.cx - K1080.cx) > 1e-3f ||
        std::fabs(back.cy - K1080.cy) > 1e-3f) { printf("  FAIL B\n"); fails++; }

    // C. same resolution -> identity (no-op).
    CameraIntrinsics same = scaleIntrinsics(K4k, 3840, 2160);
    if (same.fx != K4k.fx || same.cx != K4k.cx || same.width != 3840) { printf("  FAIL C\n"); fails++; }
    printf("[C] same-res identity ok=%d\n", (same.fx == K4k.fx && same.width == 3840));

    // D. invalid K -> no-op.
    CameraIntrinsics inv;  // width=height=0
    CameraIntrinsics invOut = scaleIntrinsics(inv, 1920, 1080);
    if (invOut.width != 0) { printf("  FAIL D\n"); fails++; }
    printf("[D] invalid-K no-op ok=%d\n", (invOut.width == 0));

    printf(fails == 0 ? "All scaling tests passed\n" : "Scaling tests FAILED\n");
    return fails == 0 ? 0 : 1;
}
