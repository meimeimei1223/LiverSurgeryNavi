#!/usr/bin/env bash
# =============================================================================
#  test/test_intrinsics_presets.sh
#  ---------------------------------------------------------------------------
#  INTRINSICS Step 5 (common/src/IntrinsicsPresets.h) の単体検証。
#  GUI なし。test_intrinsics_presets.cpp をビルドして実行する。
#
#  検証内容:
#    A. lookupPreset("azure_kinect_720p", K) -> K.fx == 918.234f
#    B. lookupPreset("nonexistent", K)       -> false
#    C. updateDynamicPreset("da3_last", ...)  で da3_last が書き換わる
#
#  GL リンクについて: IntrinsicsPresets.h -> OBJTargetExtraction.h -> mCutMesh.h
#  が GL シンボルを引き込むため -lGLEW -lGL でリンクする (プリセットコード自体は
#  GL を一切呼ばないので実行時に GL は不要)。
#
#  使い方:  bash test/test_intrinsics_presets.sh
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BIN="$(mktemp /tmp/test_presets.XXXXXX)"
trap 'rm -f "$BIN"' EXIT

echo "=== building test_intrinsics_presets.cpp ==="
if ! g++ -std=c++17 \
        -I "$ROOT/common/src" \
        -I "$ROOT/third_party/glm" \
        -I "$ROOT/third_party/eigen" \
        -I "$ROOT/third_party/tinyobjloader" \
        -I "$ROOT/third_party/nanoflann" \
        "$SCRIPT_DIR/test_intrinsics_presets.cpp" -o "$BIN" \
        -lGLEW -lGL 2> /tmp/test_presets_build.log; then
    echo "[FAIL] build failed (see /tmp/test_presets_build.log)"
    grep -iE 'error' /tmp/test_presets_build.log | head
    exit 1
fi

echo "=== running ==="
"$BIN"
rc=$?
if [[ $rc -eq 0 ]]; then
    echo "================================================================="
    echo " All tests passed"
    echo "================================================================="
fi
exit $rc
