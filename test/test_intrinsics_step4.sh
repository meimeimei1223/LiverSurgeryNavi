#!/usr/bin/env bash
# =============================================================================
#  test/test_intrinsics_step4.sh
#  ---------------------------------------------------------------------------
#  INTRINSICS Step 4 / step7-cleanup (autoSelectIntrinsicsSource) の CLI 自動検証。
#  GUI なし。lsn_registration --check-intrinsics で実関数を起動し、優先順位
#    Custom(file) > Calib(file) > DA3(file) > Preset(default fallback)
#  を検証する。 (legacy int: Custom=2, Calib=3, Preset=1, DA3=0)
#
#  Tests:
#    A. Custom のみ存在               -> source=2 (Custom)
#    B. Calib_last のみ存在           -> source=3 (Calib)
#    C. Custom/Calib/DA3 すべて無し   -> source=1 (Preset = azure_kinect_1080p)
#    D. Custom と Calib 両方           -> Custom が勝つ (source=2)
#    E. da3_last のみ存在             -> source=0 (DA3)
#    F. Calib と DA3 両方             -> Calib が勝つ (source=3)
#
#  depth_output/intrinsics_{custom,calib_last,calib,da3_last}.txt をバックアップし、
#  終了時(成否問わず)に必ず復元する。
#
#  使い方:  bash test/test_intrinsics_step4.sh
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
DO="$ROOT/depth_output"
BIN="$ROOT/build/bin/lsn_registration"

PASS=0; FAIL=0; FAILED=()
ok(){ echo "  [PASS] $1"; PASS=$((PASS+1)); }
ng(){ echo "  [FAIL] $1 :: $2"; FAIL=$((FAIL+1)); FAILED+=("$1"); }

if [[ ! -x "$BIN" ]]; then
    echo "[FAIL] $BIN not built. Build first: cmake --build build --target lsn_registration"
    exit 1
fi

# --- backup / restore the source files we touch ---
FILES=(intrinsics_custom.txt intrinsics_calib_last.txt intrinsics_calib.txt intrinsics_da3_last.txt)
BK="$(mktemp -d)"
for f in "${FILES[@]}"; do [[ -f "$DO/$f" ]] && cp "$DO/$f" "$BK/$f"; done
restore() {
    for f in "${FILES[@]}"; do
        rm -f "$DO/$f"
        [[ -f "$BK/$f" ]] && cp "$BK/$f" "$DO/$f"
    done
    rm -rf "$BK"
    echo "[cleanup] intrinsics_{custom,calib_last,calib,da3_last}.txt を復元しました"
}
trap restore EXIT

clean()  { for f in "${FILES[@]}"; do rm -f "$DO/$f"; done; }
wcustom(){ printf 'fx 1377.35\nfy 1377.17\ncx 960.228\ncy 549.671\nwidth 1920\nheight 1080\n' > "$DO/intrinsics_custom.txt"; }
wcalib() { printf 'fx 800\nfy 800\ncx 320\ncy 240\nwidth 640\nheight 480\n' > "$DO/intrinsics_calib_last.txt"; }
wda3()   { printf 'fx 700\nfy 700\ncx 360\ncy 360\nwidth 720\nheight 720\n' > "$DO/intrinsics_da3_last.txt"; }
run()    { ( cd "$ROOT/build/bin" && ./lsn_registration --check-intrinsics 2>&1 | grep -E 'CheckIntrinsics' ); }
srcof()  { echo "$1" | grep -oE 'source=[0-9]' | head -n1 | cut -d= -f2; }

echo "================================================================="
echo " INTRINSICS Step 4 autoSelect verification (--check-intrinsics)"
echo "================================================================="

# A. Custom only -> 2
clean; wcustom
L="$(run)"; echo "$L" | sed 's/^/  /'
[[ "$(srcof "$L")" == "2" ]] && ok "A (Custom only -> Custom)" || ng "A" "expected source=2, got $(srcof "$L")"

# B. Calib only -> 3
clean; wcalib
L="$(run)"; echo "$L" | sed 's/^/  /'
[[ "$(srcof "$L")" == "3" ]] && ok "B (Calib only -> Calib)" || ng "B" "expected source=3, got $(srcof "$L")"

# C. none -> 1 (Preset = Azure Kinect 1080p)
clean
L="$(run)"; echo "$L" | sed 's/^/  /'
[[ "$(srcof "$L")" == "1" ]] && ok "C (none -> Preset/AK1080p)" || ng "C" "expected source=1, got $(srcof "$L")"

# D. Custom + Calib -> Custom wins (2)
clean; wcustom; wcalib
L="$(run)"; echo "$L" | sed 's/^/  /'
[[ "$(srcof "$L")" == "2" ]] && ok "D (Custom+Calib -> Custom wins)" || ng "D" "expected source=2, got $(srcof "$L")"

# E. da3_last only -> 0 (DA3)
clean; wda3
L="$(run)"; echo "$L" | sed 's/^/  /'
[[ "$(srcof "$L")" == "0" ]] && ok "E (da3_last only -> DA3)" || ng "E" "expected source=0, got $(srcof "$L")"

# F. Calib + DA3 -> Calib wins (3)
clean; wcalib; wda3
L="$(run)"; echo "$L" | sed 's/^/  /'
[[ "$(srcof "$L")" == "3" ]] && ok "F (Calib+DA3 -> Calib wins)" || ng "F" "expected source=3, got $(srcof "$L")"

echo "================================================================="
if [[ "$FAIL" -eq 0 ]]; then
    echo " All tests passed  (PASS=$PASS FAIL=0)"; echo "================================================================="; exit 0
else
    echo " FAILED: ${FAILED[*]}  (PASS=$PASS FAIL=$FAIL)"; echo "================================================================="; exit 1
fi
