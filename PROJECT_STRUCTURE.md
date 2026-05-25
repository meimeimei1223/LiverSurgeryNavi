# AAA_LiverSurgeryNaviComb プロジェクト構造

> 最終更新: 2026-05-25
> 統合版から **common / registration / deform の3プロジェクト構成** へ再編成済み。

---

## 1. 全体ディレクトリ構成

```
AAA_LiverSurgeryNaviComb/
├── CMakeLists.txt              ← ルート。lsn_registration / lsn_deform の2アプリをビルド
├── CMakeLists_single_old.txt   ← 旧・単一プロジェクト用CMakeのバックアップ(未使用)
├── README.md
├── PROJECT_STRUCTURE.md        ← このファイル
│
├── common/
│   └── src/                    ← 両アプリ共有 35ファイル (.cpp 5 + .h/.hpp 30)
│
├── registration/
│   ├── main.cpp                ← REGアプリのmain
│   └── src/                    ← REG専用 26ファイル
│
├── deform/
│   ├── main.cpp                ← DEFORMアプリのmain
│   └── src/                    ← DEFORM専用 3ファイル
│
├── sam2_da3_lite/              ← 外部ヘルパー(深度推定+SAM2セグメンテーション)
│   ├── CMakeLists.txt          ← project(sam2_da3_lite) 別ターゲット
│   ├── src/
│   ├── include/
│   ├── onnx_models/            ← depth_anything_v3_small.onnx, sam2_hiera_tiny.{encoder,decoder}.onnx, base/, large/
│   ├── onnxruntime-linux-x64-1.15.1/      ← CPU版ONNX Runtime
│   └── onnxruntime-linux-x64-gpu-1.15.1/  ← GPU(CUDA/TensorRT)版
│
├── calibration_tool/           ← 外部ヘルパー(チェスボード カメラキャリブレーション)
│
├── third_party/                ← imgui, glm, eigen, tinyobjloader, stb, nanoflann, c-cmaes, tinyfiledialogs
├── win_deps/                   ← Windowsビルド用依存(Linuxでは未使用)
│
├── shaders/  data/  model/  input_image/  chessboard/   ← 共有リソース(ビルド時 bin/ へコピー)
├── registration_model/         ← REGの出力 = DEFORMの入力 (reg_*.obj)
├── depth_output/               ← 深度/セグメンテーションの出力先 (camera_frame_temp.jpg もここ)
├── include/
│
└── build/                      ← Qt Creator のビルドディレクトリ (Desktop-Release/bin/ に成果物)
```

---

## 2. ビルドターゲット

ルート `CMakeLists.txt` が以下を生成（出力先は `build/.../bin/`）:

| ターゲット | 内容 | ソース |
|---|---|---|
| `lsn_registration` | REGISTRATIONアプリ | registration/main.cpp + registration/src + common/src |
| `lsn_deform` | DEFORMアプリ | deform/main.cpp + deform/src + common/src |
| `sam2_da3_lite` | 深度/セグメンテーション外部ツール | sam2_da3_lite/ (サブプロジェクト) |
| `calibration_tool` | カメラキャリブレーション外部ツール | calibration_tool/ |

- 既定で **両アプリ + 両ヘルパー** がビルドされる（`option(BUILD_REG ON)`, `option(BUILD_DEFORM ON)`）。
- `lsn_registration` / `lsn_deform` は `sam2_da3_lite` と `calibration_tool` に **依存** している
  (`add_dependencies`)。IDEでアプリだけビルドしても、ヘルパーが自動で `bin/` に揃う。
- 片方だけ: `cmake .. -DBUILD_DEFORM=OFF` / `--target lsn_deform` など。

### include path の方針
各アプリは `common/src` + 自分の `src` のみを見る。両アプリは互いに干渉しない。

---

## 3. ソースの分類

### common/src (35) — 両アプリ共有
- **完全同一だった26ファイル**: AutoHandleController, CentVoxTetrahedralizerHybrid(.cpp/.h),
  FullSphereCameraWithTarget, HandleControllerBase, Hash, MeshDataTypes(.cpp/.h), MeshDrawing,
  PathConfig, PinholeProjection, PlatformCompat, RayCast, SemiAutoHandleController,
  SemiAutoPickState, SequentialDeformController, ShaderProgram(.cpp/.h), SimpleCamera,
  SoftBody(.cpp/.h), Sphere(.cpp/.h), TetoMeshData, VectorMath, simple_multi_obj_processor
- **差分を統合した9ファイル**: AR.h, OBJTargetExtraction.h, DepthRunner.h, DepthUtils.h,
  mCutMesh.h, MeshCleanup.h, NoOpen3DRegistration.h, RegistrationCore.h（以上REG版採用）,
  **AutoDeform.h（DEF版採用）**

### registration/src (26) — REG専用
AppContext, CameraPreview, CmaesRefineV2/V3/V3R/V3RS, CmaesUtils, FileDropHandler, ImageSession,
InteractionHelpers, IoUDebugDump, LiverCranioCaudalLabel, LiverLeftRightLabel, LiverRegionLabel,
MaskPicker, NormalCompatibleRefine, OBJDistributionDiag, PoseLibrary, RegistrationActions,
RegistrationImGuiManager, RegistrationUI, RimPairSampling, RimShapeMatch, SilOverlayDebug,
UmeyamaController, Undistort

### deform/src (3) — DEFORM専用
DeformGlobals, DeformPipeline, Grabber

---

## 4. 実行時のヘルパー連携

`lsn_registration` / `lsn_deform` は外部実行ファイルをサブプロセスとして起動する
（パス解決は `common/src/PathConfig.h`）:

- **sam2_da3_lite**: 深度推定(Depth Anything V3) + セグメンテーション(SAM2)。
  入力画像はディスク上のファイルを `fopen` で読む。カメラ撮影時は
  `depth_output/camera_frame_temp.jpg` に保存してから渡す。
- **calibration_tool**: チェスボードによるカメラ内部パラメータ算出。

出力は `depth_output/` に書き出される。

---

## 5. Qt Creator 利用時の注意

- ディレクトリ/ターゲットを改名したら **`build/` を削除してから Run CMake**
  （`CACHE PATH` 変数の旧値が残ってビルド失敗するため）。
- CMakeLists を変更したら **Build → Run CMake**。`.cpp`/`.h` だけの変更は通常ビルドでOK。
- `CMakeLists.txt.user` は Qt Creator のper-user設定。改名時はここのターゲット名も要更新。
