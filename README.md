# LiverSurgeryNavi

肝臓手術ナビゲーションシステム — 3D Registration + Depth Estimation + Segmentation

## 依存関係

### 共通（third_party/ に同梱済み）
- ImGui, GLM, Eigen3, stb, nanoflann, tinyfiledialogs

### Ubuntu
```
sudo apt install build-essential cmake libglew-dev libglfw3-dev
```

### Windows
- Visual Studio 2022
- win_deps/ に GLEW/GLFW 同梱済み

### ONNX Runtime（medsam2_da3_lite 使用時のみ）
https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1
- Windows: `onnxruntime-win-x64-1.15.1.zip` → `medsam2_da3_lite/onnxruntime-win-x64-1.15.1/`
- Linux: `onnxruntime-linux-x64-1.15.1.tgz` → `medsam2_da3_lite/onnxruntime-linux-x64-1.15.1/`

## ビルド

### Ubuntu
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows (Visual Studio 2022)
```
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## 実行
```
cd build/bin
./LiverSurgeryNavi
```
