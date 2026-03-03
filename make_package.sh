#!/bin/bash
# =============================================================================
# AAA_Reg3D_ONNX Linux配布パッケージ作成スクリプト
# =============================================================================
# 使い方: cd AAA_Reg3D_ONNX && bash make_package.sh
# 出力:   AAA_Reg3D_ONNX_linux.tar.gz
# =============================================================================

set -e

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
ONNXRUNTIME_ROOT="/home/meidaikasai/onnxruntime-linux-x64-1.15.1"
PACKAGE_DIR="${PROJ_ROOT}/AAA_Reg3D_ONNX_linux"

echo "============================================"
echo "  AAA_Reg3D_ONNX Package Builder (Linux)"
echo "============================================"
echo "Project root: ${PROJ_ROOT}"
echo ""

# ---------------------------------------------------------
# Step 1: Release ビルド
# ---------------------------------------------------------
echo "=== Step 1: Building Release ==="

BUILD_DIR="${PROJ_ROOT}/build_release"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJ_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}"

make -j$(nproc)

# 実行ファイル検索
AAA_EXE=$(find "${BUILD_DIR}" -name "AAA_Reg3D_ONNX" -type f -executable | head -1)
DEPTH_EXE=$(find "${BUILD_DIR}" -name "medsam2_da3_lite" -type f -executable | head -1)

if [ -z "${AAA_EXE}" ]; then
    echo "ERROR: AAA_Reg3D_ONNX not found"
    exit 1
fi
echo "Found: ${AAA_EXE}"
echo "Found: ${DEPTH_EXE:-'(not built)'}"
echo ""

# ---------------------------------------------------------
# Step 2: パッケージ構成作成
# ---------------------------------------------------------
echo "=== Step 2: Creating package ==="

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/bin"
mkdir -p "${PACKAGE_DIR}/lib"
mkdir -p "${PACKAGE_DIR}/models"
mkdir -p "${PACKAGE_DIR}/model"
mkdir -p "${PACKAGE_DIR}/registration_model"
mkdir -p "${PACKAGE_DIR}/shaders"
mkdir -p "${PACKAGE_DIR}/input_image"
mkdir -p "${PACKAGE_DIR}/depth_output"

# ---------------------------------------------------------
# Step 3: 実行ファイル
# ---------------------------------------------------------
echo "Copying executables..."
cp "${AAA_EXE}" "${PACKAGE_DIR}/bin/"
if [ -n "${DEPTH_EXE}" ]; then
    cp "${DEPTH_EXE}" "${PACKAGE_DIR}/bin/"
fi
strip "${PACKAGE_DIR}/bin/"* 2>/dev/null || true

# ---------------------------------------------------------
# Step 4: ONNX Runtime
# ---------------------------------------------------------
echo "Copying ONNX Runtime..."
if [ -d "${ONNXRUNTIME_ROOT}/lib" ]; then
    cp -P "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"* "${PACKAGE_DIR}/lib/"
    echo "  OK"
else
    echo "  WARNING: not found"
fi

# ---------------------------------------------------------
# Step 5: ONNXモデル
# ---------------------------------------------------------
echo "Copying ONNX models..."
MODELS_SRC="${PROJ_ROOT}/medsam2_da3_lite/models"
if [ -d "${MODELS_SRC}" ]; then
    cp "${MODELS_SRC}"/*.onnx "${PACKAGE_DIR}/models/" 2>/dev/null || true
    echo "  OK ($(ls "${PACKAGE_DIR}/models/" | wc -l) files)"
else
    echo "  WARNING: Place .onnx files in models/ manually"
fi

# ---------------------------------------------------------
# Step 6: アセット
# ---------------------------------------------------------
echo "Copying assets..."

copy_assets() {
    local name="$1" src="$2" dst="$3"
    shift 3
    if [ -d "${src}" ]; then
        for pat in "$@"; do
            cp ${src}/${pat} "${dst}/" 2>/dev/null || true
        done
        local count=$(ls "${dst}/" 2>/dev/null | wc -l)
        echo "  ${name}: ${count} files"
    else
        echo "  ${name}: NOT FOUND (${src})"
    fi
}

copy_assets "shaders" "${PROJ_ROOT}/shaders" "${PACKAGE_DIR}/shaders" "*.vert" "*.frag"
copy_assets "model" "${PROJ_ROOT}/model" "${PACKAGE_DIR}/model" "*.obj" "*.txt"
copy_assets "reg_model" "${PROJ_ROOT}/registration_model" "${PACKAGE_DIR}/registration_model" "*.obj"
copy_assets "input_image" "${PROJ_ROOT}/input_image" "${PACKAGE_DIR}/input_image" "*"

# ---------------------------------------------------------
# Step 7: 起動スクリプト
# ---------------------------------------------------------
echo "Creating launch script..."

cat > "${PACKAGE_DIR}/run.sh" << 'EOF'
#!/bin/bash
# ===========================================
# AAA_Reg3D_ONNX 起動スクリプト
# ===========================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ONNX Runtime ライブラリパス設定
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"

# bin/ に移動して実行（相対パスが ../model/ 等を指す）
cd "${SCRIPT_DIR}/bin"
./AAA_Reg3D_ONNX "$@"
EOF
chmod +x "${PACKAGE_DIR}/run.sh"

# ---------------------------------------------------------
# Step 8: README
# ---------------------------------------------------------
cat > "${PACKAGE_DIR}/README.txt" << 'EOF'
======================================
  AAA_Reg3D_ONNX - Linux版
======================================

■ 起動方法:
  cd AAA_Reg3D_ONNX_linux
  ./run.sh

■ 必要パッケージ:
  sudo apt install libglfw3 libglew2.2 libeigen3-dev

■ フォルダ構成:
  bin/                実行ファイル
  lib/                ONNX Runtime (.so)
  models/             ONNXモデル（深度推定用、別途配布）
  model/              3Dモデル（liver.obj 等）
  registration_model/ レジストレーション用モデル
  shaders/            シェーダーファイル
  input_image/        入力画像
  depth_output/       深度推定出力（自動生成）

■ ONNXモデル（models/に配置）:
  - depth_anything_v3_small.onnx
  - sam2_hiera_tiny.encoder.onnx
  - sam2_hiera_tiny.decoder.onnx

■ キー操作:
  U     カメラプレビュー開始/停止
  I     キャプチャ＆深度推定
  G     Umeyamaレジストレーション
  T     レジストレーション実行
  Y     自動レジストレーション
  O     カメラビューレジストレーション
  L     Raycastレジストレーション
  TAB   表示切替
  1-7   メッシュ表示切替
  F2    カメラリセット
  ESC   終了
EOF

# ---------------------------------------------------------
# Step 9: 確認 & tar.gz
# ---------------------------------------------------------
echo ""
echo "============================================"
echo "  Package contents:"
echo "============================================"
find "${PACKAGE_DIR}" -type f | sort | while read f; do
    SIZE=$(du -h "$f" | cut -f1)
    REL=${f#${PACKAGE_DIR}/}
    echo "  ${SIZE}  ${REL}"
done

TOTAL=$(du -sh "${PACKAGE_DIR}" | cut -f1)
echo ""
echo "Total: ${TOTAL}"

# tar.gz
cd "${PROJ_ROOT}"
TAR_NAME="AAA_Reg3D_ONNX_linux.tar.gz"
tar czf "${TAR_NAME}" "AAA_Reg3D_ONNX_linux/"
TAR_SIZE=$(du -h "${TAR_NAME}" | cut -f1)

echo ""
echo "============================================"
echo "  DONE!"
echo "  Archive: ${TAR_NAME} (${TAR_SIZE})"
echo "============================================"
echo ""
echo "配布: ${TAR_NAME} を共有"
echo "実行: tar xzf ${TAR_NAME} && cd AAA_Reg3D_ONNX_linux && ./run.sh"
echo ""
