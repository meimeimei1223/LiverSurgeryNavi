# Depth Anything V3 + SAM2 C++ ONNX 軽量版

OpenCVを使用しない軽量版。画像処理はstb_imageのみ使用。

## 依存関係

- **ONNX Runtime** (必須)
- **stb_image** (CMakeで自動ダウンロード)

OpenCVは不要！

## ビルド

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 使用方法

```bash
# 基本的な使用方法
./medsam2_da3_lite image.jpg --point 640,360

# 複数点指定
./medsam2_da3_lite image.jpg --point 640,360 --point 800,400 --bg-point 200,200

# オプション指定
./medsam2_da3_lite image.jpg \
    --depth-model models/depth_anything_v3_small.onnx \
    --sam-encoder models/sam2_hiera_tiny.encoder.onnx \
    --sam-decoder models/sam2_hiera_tiny.decoder.onnx \
    --output results \
    --point 640,360
```

## オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--depth-model` | Depth Anything V3 ONNXパス | models/depth_anything_v3_small.onnx |
| `--sam-encoder` | SAM2エンコーダーONNXパス | models/sam2_hiera_tiny.encoder.onnx |
| `--sam-decoder` | SAM2デコーダーONNXパス | models/sam2_hiera_tiny.decoder.onnx |
| `--output` | 出力ディレクトリ | output |
| `--point` | 前景点 (x,y) | 画像中心 |
| `--bg-point` | 背景点 (x,y) | なし |
| `--cuda` | CUDA使用 | OFF |

## 出力ファイル

```
output/
├── original.jpg              # 元画像
├── segmentation_mask.png     # セグメンテーションマスク
├── segmentation_overlay.jpg  # オーバーレイ画像
├── depth_full.png            # 深度マップ（グレースケール）
├── depth_full_colored.png    # 深度マップ（カラー）
├── depth_masked.png          # マスク領域の深度（グレースケール）
└── depth_masked_colored.png  # マスク領域の深度（カラー）
```

## ファイル構成

```
cpp_onnx_lite/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── image_utils.hpp       # 画像処理ユーティリティ
│   ├── depth_anything_v3.hpp # Depth推論
│   ├── sam2_segmentor.hpp    # SAM2推論
│   ├── stb_image.h           # (自動ダウンロード)
│   ├── stb_image_write.h     # (自動ダウンロード)
│   └── stb_image_resize2.h   # (自動ダウンロード)
├── src/
│   ├── image_utils.cpp
│   ├── depth_anything_v3.cpp
│   ├── sam2_segmentor.cpp
│   └── main.cpp
└── models/                   # モデル配置場所
```

## 配布

OpenCVがないため、配布が簡単：

```
配布パッケージ/
├── medsam2_da3_lite          # 実行ファイル
├── libonnxruntime.so.1.15.1  # ONNX Runtime
├── models/
│   ├── depth_anything_v3_small.onnx
│   ├── sam2_hiera_tiny.encoder.onnx
│   └── sam2_hiera_tiny.decoder.onnx
└── run.sh                    # 起動スクリプト
```

## ライセンス

- stb_image: Public Domain / MIT
- ONNX Runtime: MIT
- Depth Anything V3: Apache 2.0
- SAM2: Apache 2.0
