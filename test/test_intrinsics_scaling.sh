#!/usr/bin/env bash
# =============================================================================
#  test/test_intrinsics_scaling.sh
#  ---------------------------------------------------------------------------
#  common/src/IntrinsicsScaling.h の scaleIntrinsics() 単体検証 (FEATURE Task 1)。
#  GUI なし。test_intrinsics_scaling.cpp をビルドして実行。
#    A. 4K->1080p で fx/cx/cy が 0.5 倍、distortion 不変
#    B. 往復スケール (1080p->4K->1080p) 誤差 < 1e-3 px
#    C. 同解像度は no-op (identity)
#    D. 無効 K は no-op
#  GL リンクは IntrinsicsScaling.h -> OBJTargetExtraction.h -> mCutMesh.h 由来
#  (関数自体は GL 不使用)。-lGLEW -lGL でリンク。
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
BIN="$(mktemp /tmp/test_scaling.XXXXXX)"
trap 'rm -f "$BIN"' EXIT

echo "=== building test_intrinsics_scaling.cpp ==="
if ! g++ -std=c++17 \
        -I "$ROOT/common/src" -I "$ROOT/third_party/glm" -I "$ROOT/third_party/eigen" \
        -I "$ROOT/third_party/tinyobjloader" -I "$ROOT/third_party/nanoflann" \
        "$SCRIPT_DIR/test_intrinsics_scaling.cpp" -o "$BIN" -lGLEW -lGL \
        2> /tmp/test_scaling_build.log; then
    echo "[FAIL] build failed (see /tmp/test_scaling_build.log)"
    grep -iE 'error' /tmp/test_scaling_build.log | head
    exit 1
fi
echo "=== running ==="
"$BIN"; rc=$?
[[ $rc -eq 0 ]] && { echo "================================================================="; echo " All tests passed"; echo "================================================================="; }
exit $rc
