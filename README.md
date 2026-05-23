# LiverSurgeryNavi
Liver Surgery Navigation System — 3D Registration + Depth Estimation + Segmentation with **GPU Acceleration**

## 🚀 GPU-Accelerated Performance

This system now includes **CUDA GPU support** for dramatically faster inference:
- **10-50x faster** depth estimation and segmentation
- GPU-accelerated ONNX Runtime with CUDA provider
- Automatic GPU detection and fallback to CPU if unavailable

## Quick Start (Pre-built Package)

Download pre-built packages from [GitHub Actions](https://github.com/meimeimei1223/LiverSurgeryNavi/actions).

```bash
# Linux (GPU-enabled)
cd LiverSurgeryNavi-Linux
chmod +x LiverSurgeryNavi medsam2_da3_lite
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./LiverSurgeryNavi
```

The package includes:
- GPU-accelerated ONNX Runtime (CUDA)
- Depth Anything V3 Small + SAM2 models
- Automatic GPU/CPU selection

## Depth Estimation Models

You can switch between 3 models using the "Depth Model" combo box in the UI.

| Model | Size | Inference (GPU) | Inference (CPU) | Included |
|-------|------|-----------------|-----------------|----------|
| **Small** (default) | 101MB | ~0.05s | ~0.8s | Yes |
| **Base** | 394MB | ~0.12s | ~2.1s | Manual download |
| **Large** | 1.3GB | ~0.35s | ~6.5s | Manual download |

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

## Requirements

### GPU Requirements (Optional but Recommended)
- NVIDIA GPU with CUDA Compute Capability ≥ 5.0
- CUDA 11.8 or later
- NVIDIA Driver ≥ 450.80.02
- **Note:** System automatically falls back to CPU if GPU is unavailable

### Dependencies

#### Bundled (in third_party/)
- ImGui, GLM, Eigen3, stb, nanoflann, tinyfiledialogs
- c-cmaes (Apache-2.0) - CMA-ES optimization library

#### Ubuntu
```
sudo apt install build-essential cmake libglew-dev libglfw3-dev
# For GPU support (optional)
# Install CUDA toolkit from NVIDIA
```

#### Windows
- Visual Studio 2022
- GLEW/GLFW bundled in win_deps/
- CUDA Toolkit (optional for GPU support)

#### ONNX Runtime (only when building from source)
Pre-built packages already include **GPU-enabled** ONNX Runtime. If building from source:

https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1
- Windows: `onnxruntime-win-x64-gpu-1.15.1.zip` (GPU version)
- Linux: `onnxruntime-linux-x64-gpu-1.15.1.tgz` (GPU version)

```bash
cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.15.1 -DENABLE_CUDA=ON
```

## Build

### Ubuntu (with GPU support)
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)
```

### Windows (Visual Studio 2022, with GPU support)
```
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON
cmake --build . --config Release
```

**Note:** Omit `-DENABLE_CUDA=ON` to build CPU-only version.

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
