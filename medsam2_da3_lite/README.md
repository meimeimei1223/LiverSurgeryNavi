# medsam2_da3_lite

Lightweight Depth Anything V3 + SAM2 C++ ONNX inference engine.  
No OpenCV dependency — uses stb_image only.

## Dependencies

- **ONNX Runtime** (required): https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1
- **stb_image** (auto-downloaded by CMake)

## Build

```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
make -j$(nproc)
```

## Usage

```bash
# Basic usage
./medsam2_da3_lite image.jpg --point 640,360

# Multiple points
./medsam2_da3_lite image.jpg --point 640,360 --point 800,400 --bg-point 200,200

# Full options
./medsam2_da3_lite image.jpg \
    --depth-model models/depth_anything_v3_small.onnx \
    --sam-encoder models/sam2_hiera_tiny.encoder.onnx \
    --sam-decoder models/sam2_hiera_tiny.decoder.onnx \
    --output results \
    --point 640,360
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--depth-model` | Depth Anything V3 ONNX path | models/depth_anything_v3_small.onnx |
| `--sam-encoder` | SAM2 encoder ONNX path | models/sam2_hiera_tiny.encoder.onnx |
| `--sam-decoder` | SAM2 decoder ONNX path | models/sam2_hiera_tiny.decoder.onnx |
| `--output` | Output directory | output |
| `--point` | Foreground point (x,y) | Image center |
| `--bg-point` | Background point (x,y) | None |
| `--cuda` | Use CUDA | OFF |

## ONNX Models

Download from Hugging Face:

| Model | Size | Source |
|-------|------|--------|
| Depth Anything V3 Small | 101MB | [onnx-community/depth-anything-v3-small](https://huggingface.co/onnx-community/depth-anything-v3-small) |
| SAM2 Hiera Tiny Encoder | 129MB | [vietanhdev/segment-anything-2-onnx-models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models) |
| SAM2 Hiera Tiny Decoder | 20MB | [vietanhdev/segment-anything-2-onnx-models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models) |

## Output Files

```
output/
├── original.jpg                    # Original image
├── segmentation_mask.png           # Segmentation mask
├── segmentation_overlay.jpg        # Overlay visualization
├── depth_full.png                  # Full depth map (grayscale)
├── depth_full_colored.png          # Full depth map (colored)
├── depth_masked.png                # Masked depth (grayscale)
├── depth_masked_colored.png        # Masked depth (colored)
├── depth_masked_renorm.png         # Masked depth renormalized
└── depth_masked_renorm_colored.png # Masked depth renormalized (colored)
```

## License

- stb_image: Public Domain / MIT
- ONNX Runtime: MIT
- Depth Anything V3: Apache-2.0
- SAM2: Apache-2.0
