# LiverSurgeryNavi
Liver Surgery Navigation System — 3D Registration + Depth Estimation + Segmentation

## Quick Start (Pre-built Package)

Download pre-built packages from [GitHub Actions](https://github.com/meimeimei1223/LiverSurgeryNavi/actions).

```bash
# Linux
cd LiverSurgeryNavi-Linux
chmod +x LiverSurgeryNavi medsam2_da3_lite
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./LiverSurgeryNavi
```

The package includes Depth Anything V3 Small + SAM2 models and runs out of the box.

## Depth Estimation Models

You can switch between 3 models using the "Depth Model" combo box in the UI.

| Model | Size | Inference (CPU) | Included |
|-------|------|-----------------|----------|
| **Small** (default) | 101MB | ~0.8s | Yes |
| **Base** | 394MB | ~2.1s | Manual download |
| **Large** | 1.3GB | ~6.5s | Manual download |

Small is included in the package. To use Base or Large, follow the download instructions below.

### Download Base Model (~394MB)

```bash
cd LiverSurgeryNavi/onnx_models
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

### Download Large Model (~1.3GB)

```bash
cd LiverSurgeryNavi/onnx_models
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

After downloading, the models will appear in the "Depth Model" combo box in the UI.

## Features

### Registration Methods

| Method | Shortcut | Description |
|--------|----------|-------------|
| HemiAuto | UI Button | Hemisphere-based automatic registration |
| BIPOP-CMA-ES | Shift+V / UI Button | Multi-start CMA-ES optimization (Hansen 2009) |
| Refine | UI Button | Normal-compatible ICP refinement |
| Umeyama Manual | UI Button | Manual point-correspondence registration |

### Pose Library
- Session management (Front#1, Back#1, etc.)
- Method tracking with BIPOP/Refine cumulative counts
- Elapsed time measurement and display
- CSV export with session, bipop_count, refine_count, elapsed_sec columns

### Deformation
- Sphere Radius slider for deformation control

## Dependencies

### Bundled (in third_party/)
- ImGui, GLM, Eigen3, stb, nanoflann, tinyfiledialogs
- c-cmaes (Apache-2.0) - CMA-ES optimization library

### Ubuntu
```
sudo apt install build-essential cmake libglew-dev libglfw3-dev
```

### Windows
- Visual Studio 2022
- GLEW/GLFW bundled in win_deps/

### ONNX Runtime (only when building from source)
Pre-built packages already include ONNX Runtime. If building from source, download manually:

https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1
- Windows: `onnxruntime-win-x64-1.15.1.zip`
- Linux: `onnxruntime-linux-x64-1.15.1.tgz`

```bash
cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.15.1
```

## Build

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

## Run
```
cd build/bin
./LiverSurgeryNavi
```

## Model Sources

| Model/Library | License | Source |
|---------------|---------|--------|
| Depth Anything V3 | Apache-2.0 | [onnx-community/depth-anything-v3-*](https://huggingface.co/onnx-community/depth-anything-v3-small) |
| SAM2 Hiera Tiny | Apache-2.0 | [vietanhdev/segment-anything-2-onnx-models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models) |
| ONNX Runtime | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| c-cmaes | Apache-2.0 | [CMA-ES/c-cmaes](https://github.com/CMA-ES/c-cmaes) |


## Download 3D Models Only

To download only the 3D mesh models without cloning the entire repository:

| File | Direct Download |
|------|----------------|
| liver.obj | [Download](https://github.com/meimeimei1223/LiverSurgeryNavi/raw/main/model/liver.obj) |
| tumor.obj | [Download](https://github.com/meimeimei1223/LiverSurgeryNavi/raw/main/model/tumor.obj) |
| portal.obj | [Download](https://github.com/meimeimei1223/LiverSurgeryNavi/raw/main/model/portal.obj) |
| vein.obj | [Download](https://github.com/meimeimei1223/LiverSurgeryNavi/raw/main/model/vein.obj) |
| gb.obj | [Download](https://github.com/meimeimei1223/LiverSurgeryNavi/raw/main/model/gb.obj) |
| res.obj | [Download](https://github.com/meimeimei1223/LiverSurgeryNavi/raw/main/model/res.obj) |
