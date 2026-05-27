# AAA_LiverSurgeryNaviComb プロジェクト構造

> 最終更新: 2026-05-26
> 3プロジェクト構成（common / registration / deform）＋ Ctrl+D Debug Panel ＋ DEFORM 分離済み。

---

## 1. 全体ディレクトリ構成

```
AAA_LiverSurgeryNaviComb/
├── CMakeLists.txt              ← 単一ルート。lsn_registration / lsn_deform を生成
├── CMakeLists_single_old.txt   ← 旧・単一プロジェクト用CMakeのバックアップ(未使用)
├── README.md / PROJECT_STRUCTURE.md(このファイル)
│
├── common/
│   └── src/                    ← 両アプリ共有 25ファイル (.cpp 3 + .h/.hpp 22)
│
├── registration/
│   ├── main.cpp                ← REGアプリのmain
│   └── src/                    ← REG専用 28ファイル
│
├── deform/
│   ├── main.cpp                ← DEFORMアプリのmain
│   └── src/                    ← DEFORM専用 15ファイル
│
├── sam2_da3_lite/              ← 外部ヘルパー(深度推定 Depth Anything V3 + SAM2)
│   ├── CMakeLists.txt, src/, include/, onnx_models/
│   ├── onnxruntime-linux-x64-1.15.1/      (CPU)
│   └── onnxruntime-linux-x64-gpu-1.15.1/  (CUDA/TensorRT)
│
├── calibration_tool/           ← 外部ヘルパー(チェスボード カメラキャリブレーション)
├── third_party/                ← imgui, glm, eigen, tinyobjloader, stb, nanoflann, c-cmaes, tinyfiledialogs
├── win_deps/                   ← Windowsビルド用依存(Linuxでは未使用)
│
├── shaders/ data/ model/ input_image/ chessboard/   ← 共有リソース(ビルド時 bin/ へコピー)
├── registration_model/         ← REGの出力 = DEFORMの入力 (reg_*.obj)
├── depth_output/               ← 深度/セグメンテーション出力 (camera_frame_temp.jpg もここ)
├── docs/                       ← 設計/リファクタ文書 (KEY_REFERENCE.md, ui-refactor/ 等)
│
└── build/                      ← Qt Creator のビルド (Desktop-Release/bin/ に成果物)
```

---

## 2. ビルドターゲット（単一ルート CMakeLists.txt）

| ターゲット | 内容 | ソース |
|---|---|---|
| `lsn_registration` | REGISTRATIONアプリ | registration/main.cpp + registration/src + **common/src** |
| `lsn_deform` | DEFORMアプリ | deform/main.cpp + deform/src + **common/src** |
| `sam2_da3_lite` | 深度/セグメンテーション外部ツール | sam2_da3_lite/ (サブプロジェクト) |
| `calibration_tool` | カメラキャリブレーション外部ツール | calibration_tool/ |

- 既定で全部ビルド（`option(BUILD_REG ON)`, `option(BUILD_DEFORM ON)`）。
- アプリは `sam2_da3_lite` / `calibration_tool` に **add_dependencies** 済み
  → IDE でアプリだけビルドしてもヘルパーが bin/ に揃う。
- include path: 各アプリは `common/src` + 自分の `src` のみ。**registration の include path に deform/src は無い**（相互非依存）。

---

## 3. ソース分類

### common/src (25) — 両アプリ共有・**SoftBody 非依存**
```
AR.h  DepthRunner.h  DepthUtils.h  FullSphereCameraWithTarget.h  Hash.h
MeshCleanup.h  MeshDataTypes(.cpp/.h)  MeshDrawing.h  NoOpen3DRegistration.h
OBJTargetExtraction.h  PathConfig.h  PinholeProjection.h  PlatformCompat.h
RayCast.h  RegistrationCore.h  ScreenMeshPoints.h  ShaderProgram(.cpp/.h)
SimpleCamera.hpp  Sphere(.cpp/.h)  VectorMath.h  mCutMesh.h  simple_multi_obj_processor.h
```
- `RayCast.h`：SoftBody 非依存の汎用 ray-mesh 交差（deform-separation で汎用化）。
- `MeshDrawing.h`：mCutMesh 描画のみ（SoftBody 版は deform へ分離）。
- `ScreenMeshPoints.h`：重い screen/target メッシュを GL_POINTS で軽量描画（REG/DEFORM 共有）。

### registration/src (28) — REG専用
AppContext, CameraPreview, CmaesRefineV2/V3/V3R/V3RS, CmaesUtils, **DebugPanel.h**,
FileDropHandler, ImageSession, InteractionHelpers, IoUDebugDump, LiverCranioCaudalLabel,
LiverLeftRightLabel, LiverRegionLabel, MaskPicker, NormalCompatibleRefine,
OBJDistributionDiag, PoseLibrary, RegistrationActions, RegistrationImGuiManager,
RegistrationUI, RimPairSampling, RimShapeMatch, SilOverlayDebug, **StlExport.h**,
UmeyamaController, Undistort
- `DebugPanel.h`：Ctrl+D で開く統合デバッグパネル（6タブ G/O/N/W/U/Viz）。
- `StlExport.h`：旧 M/Shift+M キーの STL/OBJ export（宣言。定義は main.cpp）。

### deform/src (15) — DEFORM専用（SoftBody 系を含む）
```
DeformGlobals.h  DeformPipeline.h  Grabber.h                  (元からの3つ)
SoftBody(.cpp/.h)  CentVoxTetrahedralizerHybrid(.cpp/.h)  TetoMeshData.h
HandleControllerBase.h  AutoHandleController.h  SemiAutoHandleController.h
SemiAutoPickState.h  SequentialDeformController.h  AutoDeform.h  (common から移動)
MeshDrawingSoftBody.h                                          (MeshDrawing.h から分離)
```

---

## 4. UI / キーボード（registration）

- **Ctrl+D Debug Panel**：3つの旧・常時 floating panel（Ctrl+G Quadrant Selector /
  Normal-Compatible Refine / ScreenMesh Display）と、可視化/デバッグのキー操作を
  6タブに集約。詳細は `docs/ui-refactor/IMPLEMENTATION_REPORT.md`。
- **キーボードは action/display/tuning のみ16キー**。viz/debug トグルは Ctrl+D へ、
  BIPOP V1/V2 は Alt+G/Alt+Shift+G、Silhouette Align は Alt+P、camera/depth/export は
  サイドバーボタン。早見表は `docs/KEY_REFERENCE.md`。

---

## 5. 実行時のヘルパー連携

`lsn_registration` / `lsn_deform` は外部実行ファイルをサブプロセス起動（パス解決は
`common/src/PathConfig.h`）：
- **sam2_da3_lite**：深度推定 + SAM2 セグメンテーション + vignette 検出。入力はディスク上
  ファイルを `fopen`（カメラ撮影時は `depth_output/camera_frame_temp.jpg` に保存してから渡す）。
  出力は depth/mask 系（`depth_metric.bin`=float32+DEPTH ヘッダ, `segmentation_mask.png`,
  `depth_*.png`）と DA3 推定 K（`intrinsics_da3.txt`）のみ。**K は受け取らず、OBJ/texture も書かない**
  （obj-migration リファクタ。詳細 `docs/refactor/`）。
- **calibration_tool**：チェスボードによるカメラ内部パラメータ算出。

**OBJ / intrinsics の生成は REG 側**（`common/src/DepthToObjExport.h`）が担当：REG が自前の K で
`depth_metric.bin` をアンプロジェクトし、カノニカル名 `pc_metric_pinhole_masked.obj` /
`pc_metric_pinhole_full_light.obj` / `intrinsics.txt`（name フィールドに source）/ `texture.png` を
`depth_output/` に書く。DEFORM はこのカノニカル名を読む（旧 `_k4a` 名はレガシー fallback）。
`depth_metric.bin` が無く既存 OBJ がある場合は REG はそれをそのまま使う（外部 depth ソース投入経路）。
REG→DEFORM 連携は `registration_model/reg_liver.obj` 等。

**Reconstruct from BIN（経路 D）**：過去/外部由来の `depth_metric.bin` + `segmentation_mask.png`
（+ optional `original.jpg`）を REG の UI にドロップすると、DA3/SAM2 推論なしで現在の K（解像度が
違えば自動スケール）で OBJ を再生成できる（`registration/src/ReconstructFromBin.{h,cpp}` +
`common/src/IntrinsicsScaling.h`）。K だけ変えた焼き直し・過去症例の再処理・DA3 以外の depth ソース
（Kinect/RealSense/Stereo を bin 形式に変換）に使える。実行前に depth_output を
`depth_output_backup_<ts>/` へ退避し、失敗時はロールバック。詳細 `FEATURE_PLAN_external_depth_drop.md`。

---

## 6. Qt Creator 利用時の注意

- ディレクトリ/ターゲット改名や CMakeLists 変更後は **`build/` 削除 → Run CMake**
  （`CACHE PATH` 変数の旧値が残ってビルド失敗するため）。
- `.cpp`/`.h` の変更のみは通常ビルドでOK。`CMakeLists.txt.user` は改名時にターゲット名要更新。
- ビルドが `No space left on device` で落ちたらディスク容量を確認。
