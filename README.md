# LiverSurgeryNavi
肝臓手術ナビゲーションシステム — 3D Registration + Depth Estimation + Segmentation

## クイックスタート（ビルド済みパッケージ）

[GitHub Actions](https://github.com/meimeimei1223/LiverSurgeryNavi/actions) からビルド済みパッケージをダウンロードできます。

```bash
# Linux
cd LiverSurgeryNavi-Linux
chmod +x LiverSurgeryNavi medsam2_da3_lite
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./LiverSurgeryNavi
```

パッケージには Depth Anything V3 Small + SAM2 モデルが同梱されており、そのまま起動できます。

## 深度推定モデル

UI の「Depth Model」コンボボックスから3種類のモデルを切り替えられます。

| モデル | サイズ | 推論時間(CPU) | 同梱 |
|--------|--------|---------------|------|
| **Small** (デフォルト) | 101MB | ~0.8秒 | ✅ |
| **Base** | 394MB | ~2.1秒 | 手動DL |
| **Large** | 1.3GB | ~6.5秒 | 手動DL |

Small は同梱済みです。Base / Large を使いたい場合は以下の手順でダウンロードしてください。

### Base モデルのダウンロード (~394MB)

```bash
cd LiverSurgeryNavi/models
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('base', exist_ok=True)
hf_hub_download(repo_id='onnx-community/depth-anything-v3-base', filename='onnx/model.onnx', local_dir='base/')
hf_hub_download(repo_id='onnx-community/depth-anything-v3-base', filename='onnx/model.onnx_data', local_dir='base/')
for f in ['base/onnx/model.onnx', 'base/onnx/model.onnx_data']:
    os.rename(f, 'base/' + os.path.basename(f))
os.rmdir('base/onnx')
print('Done!')
"
```

### Large モデルのダウンロード (~1.3GB)

```bash
cd LiverSurgeryNavi/models
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('large', exist_ok=True)
hf_hub_download(repo_id='onnx-community/depth-anything-v3-large', filename='onnx/model.onnx', local_dir='large/')
hf_hub_download(repo_id='onnx-community/depth-anything-v3-large', filename='onnx/model.onnx_data', local_dir='large/')
for f in ['large/onnx/model.onnx', 'large/onnx/model.onnx_data']:
    os.rename(f, 'large/' + os.path.basename(f))
os.rmdir('large/onnx')
print('Done!')
"
```

### Git LFS を使う場合

```bash
sudo apt install git-lfs && git lfs install

# Base
cd LiverSurgeryNavi/models
git clone https://huggingface.co/onnx-community/depth-anything-v3-base
mkdir -p base && mv depth-anything-v3-base/onnx/model.onnx* base/
rm -rf depth-anything-v3-base

# Large
git clone https://huggingface.co/onnx-community/depth-anything-v3-large
mkdir -p large && mv depth-anything-v3-large/onnx/model.onnx* large/
rm -rf depth-anything-v3-large
```

ダウンロード後、UI の「Depth Model」コンボボックスで Base / Large が選択可能になります。

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

## モデルのソース

| モデル | ライセンス | ソース |
|--------|-----------|--------|
| Depth Anything V3 | Apache-2.0 | [onnx-community/depth-anything-v3-*](https://huggingface.co/onnx-community/depth-anything-v3-small) |
| SAM2 Hiera Tiny | Apache-2.0 | [vietanhdev/segment-anything-2-onnx-models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models) |
| ONNX Runtime | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
