#!/usr/bin/env bash
# =============================================================================
#  test/test_intrinsics_step1.sh
#  ---------------------------------------------------------------------------
#  INTRINSICS Step 1 (DeformPipeline.h: K4A 720p ハードコード除去) の CLI 自動検証。
#
#  検証対象: deform/src/DeformPipeline.h が canonical depth_output/intrinsics.txt
#  (Phase 4 で primary、legacy intrinsics_k4a.txt は fallback) から K を読み、
#  board UV を K から計算していること。GUI/OpenGL なし lsn_deform --dry-run で確認。
#
#  Tests:
#    A. ハードコード残渣 grep (918.234 等が DeformPipeline.h に無いこと)
#    B. lsn_deform のビルドが成功すること (exit 0)
#    C. 720p の K を intrinsics.txt に書き、"deformK fx=918" がログに出ること
#    D. 1080p の K を intrinsics.txt に書き、"deformK fx=1377" がログに出ること
#    E. K を変えると board UV 範囲が変わること (= K が UV 計算に効いている)
#    F. intrinsics.txt も intrinsics_k4a.txt も無い -> K4A 720p hardcode fallback
#    G. intrinsics.txt 無し + intrinsics_k4a.txt 有り -> legacy fallback (fx=918 + warn)
#
#  使い方:  bash test/test_intrinsics_step1.sh
#  depth_output/intrinsics.txt と intrinsics_k4a.txt を冒頭でバックアップし、終了時
#  (成否問わず) に必ず元の状態へ復元する (trap)。テストで壊さない。
# =============================================================================
set -u

# --- パス解決 (このスクリプトの位置からリポジトリルートを求める) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DEFORM_H="$ROOT/deform/src/DeformPipeline.h"
# obj-migration Phase 4: DEFORM now loads the canonical intrinsics.txt first
# (legacy intrinsics_k4a.txt is only a fallback). Tests write the canonical file.
INTR="$ROOT/depth_output/intrinsics.txt"
INTR_K4A="$ROOT/depth_output/intrinsics_k4a.txt"
BIN="$ROOT/build/bin/lsn_deform"
BUILD_DIR="$ROOT/build"

PASS=0
FAIL=0
FAILED_TESTS=()

ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
ng()   { echo "  [FAIL] $1 :: $2"; FAIL=$((FAIL+1)); FAILED_TESTS+=("$1"); }
hr()   { echo "-----------------------------------------------------------------"; }

# --- intrinsics.txt + intrinsics_k4a.txt のバックアップ / 復元 --------------------
BK_DIR="$(mktemp -d)"
declare -A HAD
for f in "$INTR" "$INTR_K4A"; do
    if [[ -f "$f" ]]; then HAD["$f"]=1; cp "$f" "$BK_DIR/$(basename "$f")"; else HAD["$f"]=0; fi
done
restore_intrinsics() {
    for f in "$INTR" "$INTR_K4A"; do
        if [[ "${HAD[$f]}" -eq 1 ]]; then cp "$BK_DIR/$(basename "$f")" "$f";
        else rm -f "$f"; fi
    done
    rm -rf "$BK_DIR"
    echo "[cleanup] intrinsics.txt / intrinsics_k4a.txt を元の状態に復元しました"
}
trap restore_intrinsics EXIT

# 720p / 1080p の K を書くヘルパ
write_720p() {
    cat > "$INTR" <<EOF
fx 918.234
fy 918.112
cx 640.152
cy 366.447
width 1280
height 720
EOF
}
write_1080p() {
    cat > "$INTR" <<EOF
fx 1377.35
fy 1377.17
cx 960.228
cy 549.671
width 1920
height 1080
EOF
}
# FOV が異なる(比例スケールでない) K。Test E 用。
write_widefov() {
    cat > "$INTR" <<EOF
fx 600
fy 600
cx 300
cy 300
width 1280
height 720
EOF
}

# --dry-run を実行してログを標準出力に返す
run_dry_run() {
    ( cd "$ROOT/build/bin" && ./lsn_deform --dry-run 2>&1 )
}
# ログから board UV 行を抜き出す ("UV u=[...] v=[...]")
extract_uv() {
    echo "$1" | grep -oE 'UV u=\[[^]]*\] v=\[[^]]*\]' | head -n1
}

echo "================================================================="
echo " INTRINSICS Step 1 verification  (lsn_deform --dry-run)"
echo " root: $ROOT"
echo "================================================================="

# === Test A: ハードコード残渣 grep ==============================================
hr; echo "Test A: ハードコード残渣 grep (DeformPipeline.h)"
if grep -nE '918\.234|918\.112|640\.152|366\.447' "$DEFORM_H" > /tmp/_stepA.txt 2>&1; then
    ng "A" "ハードコードされた K4A 720p 定数が残っている: $(cat /tmp/_stepA.txt)"
else
    ok "A (DeformPipeline.h に 918.234 等のハードコード無し)"
fi

# === Test B: ビルド =============================================================
hr; echo "Test B: lsn_deform ビルド"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "  build/ 未構成 -> cmake configure"
    cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_REG=ON -DBUILD_DEFORM=ON > /tmp/_stepB_cfg.log 2>&1
fi
if cmake --build "$BUILD_DIR" --target lsn_deform -j"$(nproc)" > /tmp/_stepB.log 2>&1; then
    if [[ -x "$BIN" ]]; then
        ok "B (ビルド成功, $BIN)"
    else
        ng "B" "ビルドは exit 0 だが $BIN が見つからない"
    fi
else
    ng "B" "ビルド失敗 (詳細: /tmp/_stepB.log)"
    echo "  ----- ビルドログ末尾 -----"
    tail -n 15 /tmp/_stepB.log | sed 's/^/  /'
fi

# 以降のテストはバイナリが無ければスキップ不能なので、無ければ即終了
if [[ ! -x "$BIN" ]]; then
    hr
    echo "lsn_deform が無いため C-F をスキップします。"
    echo "================================================================="
    echo " RESULT: PASS=$PASS FAIL=$FAIL"
    echo "================================================================="
    exit 1
fi

# === Test C: 720p ==============================================================
hr; echo "Test C: 720p の K -> deformK fx=918"
write_720p
LOG_C="$(run_dry_run)"
DEFORMK_C="$(echo "$LOG_C" | grep -E '\[DryRun\] deformK' | head -n1)"
echo "  log: $DEFORMK_C"
if echo "$LOG_C" | grep -qE 'deformK fx=918'; then
    ok "C"
else
    ng "C" "'deformK fx=918' がログに無い"
fi
UV_C="$(extract_uv "$LOG_C")"
echo "  board $UV_C"

# === Test D: 1080p =============================================================
hr; echo "Test D: 1080p の K -> deformK fx=1377"
write_1080p
LOG_D="$(run_dry_run)"
DEFORMK_D="$(echo "$LOG_D" | grep -E '\[DryRun\] deformK' | head -n1)"
echo "  log: $DEFORMK_D"
if echo "$LOG_D" | grep -qE 'deformK fx=1377'; then
    ok "D"
else
    ng "D" "'deformK fx=1377' がログに無い"
fi
UV_D="$(extract_uv "$LOG_D")"
echo "  board $UV_D"

# === Test E: K の変更が UV に効くこと ==========================================
hr; echo "Test E: K を変えると board UV が変わること"
# 注意: 720p(C) と 1080p(D) は同一カメラの比例スケールなので、解像度正規化された
#       UV は ほぼ同一になる (これ自体は正しい挙動)。よって C vs D の差では検証せず、
#       FOV が実際に異なる K (fx=600,cx=300) を使って UV が変化することを確認する。
write_widefov
LOG_E="$(run_dry_run)"
UV_E="$(extract_uv "$LOG_E")"
echo "  720p   (C): $UV_C"
echo "  widefov(E): $UV_E"
# 参考: C と D が(正しく)一致することも表示
if [[ "$UV_C" == "$UV_D" ]]; then
    echo "  (参考) 720p と 1080p の UV は一致 = 比例スケールなので正規化UVは不変(正しい)"
fi
if [[ -n "$UV_C" && -n "$UV_E" && "$UV_C" != "$UV_E" ]]; then
    ok "E (FOV の異なる K で UV 範囲が変化 = K が UV 計算に効いている)"
else
    ng "E" "K を変えても UV が変化しない (C='$UV_C' E='$UV_E')"
fi

# === Test F: fallback ==========================================================
hr; echo "Test F: canonical & legacy 両方 欠損 -> K4A 720p hardcode fallback"
# canonical-first なので、両方とも無い状態でハードコード fallback を確認する。
rm -f "$INTR" "$INTR_K4A"
LOG_F="$(run_dry_run)"
FALLBACK_LINE="$(echo "$LOG_F" | grep -iE 'fallback|default|k4a_color_720p|missing|invalid' | head -n1)"
echo "  log: $FALLBACK_LINE"
if echo "$LOG_F" | grep -qiE 'fallback|default|k4a_color_720p'; then
    ok "F"
else
    ng "F" "fallback/default を示すログが無い"
fi

# === Test G: legacy _k4a fallback (canonical 無し, _k4a 有り) ===================
hr; echo "Test G: intrinsics.txt 無し + intrinsics_k4a.txt(720p) -> legacy fallback で fx=918"
rm -f "$INTR"
printf 'fx 918.234\nfy 918.112\ncx 640.152\ncy 366.447\nwidth 1280\nheight 720\n' > "$INTR_K4A"
LOG_G="$(run_dry_run)"
echo "  $(echo "$LOG_G" | grep -E '\[DryRun\] deformK' | head -n1)"
if echo "$LOG_G" | grep -qE 'deformK fx=918' && echo "$LOG_G" | grep -qiE 'legacy|intrinsics_k4a'; then
    ok "G (legacy _k4a fallback works + warns)"
else
    ng "G" "legacy _k4a fallback で fx=918 + warn が出ていない"
fi

# === サマリ =====================================================================
hr
echo "================================================================="
if [[ "$FAIL" -eq 0 ]]; then
    echo " All tests passed  (PASS=$PASS FAIL=0)"
    echo "================================================================="
    exit 0
else
    echo " FAILED: ${FAILED_TESTS[*]}   (PASS=$PASS FAIL=$FAIL)"
    echo "================================================================="
    exit 1
fi
