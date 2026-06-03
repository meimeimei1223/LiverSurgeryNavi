# AAA_LiverSurgeryNaviComb プロジェクト構造

> 最終更新: 2026-06-03
> 3プロジェクト構成（common / registration / deform）＋ Ctrl+D Debug Panel ＋ DEFORM 分離済み。
> **DEFORM UI 統合 Phase 1 完了**（共有 `RegistrationImGuiManager` を `common/src/` に集約、DEFORM 側もサイドバー表示）。
> **REG/DEFORM は独立プロセスのまま** — 互いの遷移ボタンは export + log のみで、ユーザが手動で他方を起動する運用。
> **CMA-ES を本家 Hansen c-cmaes に置換**（`feature/cmaes-upstream`、未マージ）— 旧自作実装を削除し、アダプタ shim 経由で全 BIPOP エンジンが本家を使用。Run 並列が決定的（直列==並列ビット一致）。詳細は §8。

---

## 1. 全体ディレクトリ構成

```
AAA_LiverSurgeryNaviComb/
├── CMakeLists.txt              ← 単一ルート。lsn_registration / lsn_deform を生成
├── CMakeLists_single_old.txt   ← 旧・単一プロジェクト用CMakeのバックアップ(未使用)
├── README.md / PROJECT_STRUCTURE.md(このファイル)
│
├── common/
│   └── src/                    ← 両アプリ共有 35ファイル (.cpp 5 + .h/.hpp 30)
│
├── registration/
│   ├── main.cpp                ← REGアプリのmain
│   └── src/                    ← REG専用 29ファイル
│
├── deform/
│   ├── main.cpp                ← DEFORMアプリのmain (Phase 1 で ImGui サイドバー導入 +190行)
│   └── src/                    ← DEFORM専用 15ファイル
│
├── sam2_da3_lite/              ← 外部ヘルパー(深度推定 Depth Anything V3 + SAM2)
│   ├── CMakeLists.txt, src/, include/, onnx_models/
│   ├── onnxruntime-linux-x64-1.15.1/      (CPU)
│   └── onnxruntime-linux-x64-gpu-1.15.1/  (CUDA/TensorRT)
│
├── calibration_tool/           ← 外部ヘルパー(チェスボード カメラキャリブレーション)
├── third_party/                ← imgui, glm, eigen, tinyobjloader, stb, nanoflann, tinyfiledialogs
│   └── c-cmaes/                ← upstream/(本家 Hansen 無改変)＋ wrapper/(我々のラッパ)(§8)
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

### common/src (35) — 両アプリ共有・**SoftBody 非依存**
```
AR.h  AppImGuiBoot.h  DepthRunner.h  DepthToObjExport(.cpp/.h)  DepthUtils.h
FullSphereCameraWithTarget.h  Hash.h  IntrinsicsPresets.h  IntrinsicsScaling.h
IntrinsicsSource.h  MeshCleanup.h  MeshDataTypes(.cpp/.h)  MeshDrawing.h
NoOpen3DRegistration.h  OBJTargetExtraction.h  PathConfig.h  PinholeProjection.h
PlatformCompat.h  RayCast.h  RegistrationCore.h  RegistrationImGuiManager.h
ScreenMeshPoints.h  ShaderProgram(.cpp/.h)  SimpleCamera.hpp  Sphere(.cpp/.h)
VectorMath.h  image_utils.hpp  mCutMesh.h  obj_exporter(.cpp/.hpp)
simple_multi_obj_processor.h
```
- `RayCast.h`：SoftBody 非依存の汎用 ray-mesh 交差（deform-separation で汎用化）。
- `MeshDrawing.h`：mCutMesh 描画のみ（SoftBody 版は deform へ分離）。
- `ScreenMeshPoints.h`：重い screen/target メッシュを GL_POINTS で軽量描画（REG/DEFORM 共有）。
- **`RegistrationImGuiManager.h`** [Phase 1 移動]：REG/DEFORM 共有 UI 本体。クラス名は据え置き、`state.mainMode=0/1` で REG/DEFORM 表示を切替（include path は両アプリで `common/src` を含むので `#include "RegistrationImGuiManager.h"` のまま）。
- **`AppImGuiBoot.h`** [Phase 1 新規]：ImGui ライフサイクル集約（`init` / `shutdown` / `beginFrame` / `endFrame` / `loadFont`）。Phase 1 では DEFORM 側のみ採用、REG 側は既存 inline コードを据え置き（Phase 3 で移行予定）。
- **`IntrinsicsSource.h` / `IntrinsicsPresets.h` / `IntrinsicsScaling.h`** [intrinsics-refactor]：K の出所管理（DA3/Calib/Custom）、プリセット、解像度スケーリング。
- **`DepthToObjExport.h/cpp`** [obj-migration]：`depth_metric.bin` から OBJ + intrinsics.txt + texture.png を生成。
- **`obj_exporter.hpp/cpp`**：汎用 OBJ exporter。
- `PlatformCompat.h`：cross-platform マクロ + `platform_launch_detached`（detach spawn ヘルパ。Phase 1 では未使用、`nohup ... </dev/null >/dev/null 2>&1 &` で SIGPIPE 対策済み）。
- `PathConfig.h`：`DEPTH_EXE_PATH` / `REG_EXE_PATH` / `DEFORM_EXE_PATH` を `initPaths()` で解決。

### registration/src (29) — REG専用
AppContext, CameraPreview, CmaesRefineV2/V3/V3R/V3RS, CmaesUtils, **DebugPanel.h**,
FileDropHandler, ImageSession, InteractionHelpers, IoUDebugDump, LiverCranioCaudalLabel,
LiverLeftRightLabel, LiverRegionLabel, MaskPicker, NormalCompatibleRefine,
OBJDistributionDiag, PoseLibrary, **ReconstructFromBin(.cpp/.h)**, RegistrationActions,
RegistrationUI, RimPairSampling, RimShapeMatch, SilOverlayDebug, **StlExport.h**,
UmeyamaController, Undistort
- `DebugPanel.h`：Ctrl+D で開く統合デバッグパネル（6タブ G/O/N/W/U/Viz）。
- `CmaesRefineV2/V3/V3R/V3RS.h` + `CmaesUtils.h`：BIPOP-CMA-ES エンジン群。本家 c-cmaes をアダプタ経由で使用（旧自作からの差分は per-run シード 1 行 `srand`→`cmaes_set_seed` のみ）。V3R/V3RS は Run レベル並列機構（`g_v3rParallelRuns`/`g_v3rsParallelRuns`、既定 OFF）を持つ。詳細 §8。
- `StlExport.h`：旧 M/Shift+M キーの STL/OBJ export（宣言。定義は main.cpp）。Phase 1 で `exportRegisteredObjs()` が「Deform >>」ボタンからも呼ばれる。
- **`ReconstructFromBin.h/cpp`** [reconstruct-bin]：過去/外部由来の `depth_metric.bin` から OBJ 再生成（経路 D。Section 5 参照）。
- ※ `RegistrationImGuiManager.h` は Phase 1 で `common/src/` へ移動済み（rename なし、include 文も無修正）。

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
- **Live Calibration（Take Picture タブ）**：Intrinsics セクションのタブ内で、カメラからチェスボード画像をライブ撮影 → `calibration_tool` サブプロセスへそのまま投げる UX。Start Live Calibration 1 ボタンでセッション作成＋カメラ起動、Stop でセッション終了＋カメラ停止、Run で既存 `onRunCalibration` に委譲（`intrinsics_calib.txt` 出力・Source→Calib 切替は既存処理）。

## 4b. UI / キーボード（deform） [Phase 1 で追加]

- **サイドバー**：`common/src/RegistrationImGuiManager.h` を共有し、`state.mainMode=1` 固定で DEFORM 専用表示（DEPTH compact / REG "Registration: Done" / DEFORM + VISIBILITY が active）。Rigid / Handle / Deform ボタン、Sphere Radius スライダ、Reset All、Visibility（organs 一括 + Board + Target）、Start From Depth（log only）を提供。
- **キーボード**：R/H/D サブモード、V/T/B alpha cycle、1-7/0/P/N/Bksp/-/= は従来通り `DeformPipeline::onKey` が処理。A は AR 背景オーバーレイ切替。
- **マウス**：RIGID_MODE 中の左ドラッグ=回転 / 右ドラッグ=画面平面移動 / 左右同時=カメラ奥行移動（旧 `元の長いmain.cpp` から復活）。DEFORM_MODE 中の左ドラッグは grab、非ドラッグ時はカメラ orbit/pan。ImGui がマウス・キーボードを capture しているときは scene 側を short-return（`WantCaptureMouse/Keyboard` ガード）。
- **alpha は単一値 3 個**（`gOrganAlpha`/`gBoardAlpha`/`gTargetAlpha`）：REG の `g_meshAlpha[8]` per-organ 配列と異なり、Visibility ボタン i=0..5 は全 organ 一括で `gOrganAlpha` を cycle。Phase 3 統合時に統一予定。

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

**REG ↔ DEFORM の遷移ボタン**（Phase 1）：両アプリは独立プロセスのまま運用。
- REG「Deform >>」：`StlExport::exportRegisteredObjs()` で reg_*.obj を出力＋ログのみ。REG ウィンドウは閉じない。ユーザは Qt Creator から手動で `lsn_deform` を起動。
- DEFORM「Start From Depth」：ログのみ。ユーザが Qt Creator から手動で `lsn_registration` を再起動。
- `platform_launch_detached`（`PlatformCompat.h`）と `REG_EXE_PATH`/`DEFORM_EXE_PATH`（`PathConfig.h`）は Phase 3 統合時の足場として残置（現在未使用）。auto-spawn を試した際に Qt Creator の親パイプ閉鎖で子が SIGPIPE silent crash する罠を踏んだ経緯あり（対策は `nohup ... </dev/null >/dev/null 2>&1 &`、実装済みだが未使用）。

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
- **Qt Creator から起動した親プロセスが `std::system("./child &")` で子を spawn すると、子は親の stdout/stderr fd を継承し、親終了時に Qt Creator がそのパイプを閉じる → 子は次の `std::cout` で SIGPIPE silent crash する**。detach spawn する関数の Linux 版は `nohup <exe> </dev/null >/dev/null 2>&1 &` のように stdio を全部リダイレクトすること（`platform_launch_detached` は対策済み）。症状: 親 exit code 0・子は単独起動なら動く・Qt Creator から直接子起動でも動く・REG → DEFORM 系の auto-spawn でだけ落ちる。

---

## 7. ブランチ・リファクタ履歴メモ

- **`main`**：直近のマージ済み実装（live-calibration 含む）。
- **`feature/deform-ui-integration`**（未マージ）：Phase 1 (DEFORM UI 統合)。設計書 `IMPLEMENTATION_PLAN_DEFORM_UI_v2.md`、進捗 `memory/deform-ui-progress.md`。末尾に `3f12eb6 [wip]`（Run 並列機構＋本家 c-cmaes の vendoring を退避保存）。
- **`feature/cmaes-upstream`**（未マージ、`3f12eb6` 基点）：CMA-ES を本家 Hansen c-cmaes に置換（§8）。`ef527ab` 置換＋旧削除＋決定性2修正 → `a43c773` 本家不要ファイル整理 → `bef1791` 本家を `upstream/` に隔離＋README。設計書 `HANDOFF_upstream_cmaes_migration_v2.md`、進捗 `memory/cmaes-upstream-migration.md`。残：GUI §7 検証後に main マージ判断。
- **過去ブランチ**：feature/reconstruct-from-bin（Reconstruct from BIN、merge commit e84a63e）/ feature/live-calibration（Take Picture タブ、merge commit 895d230）/ intrinsics-refactor 系。

**今後 (Phase 3)**：単一バイナリ `lsn_unified` への統合。`onSwitchToDeformMode`/`onStartFromDepth` を in-process `currentMainMode` switch に置換、AR モードグローバル統一、alpha state 統一（REG `g_meshAlpha[8]` ↔ DEFORM 単一値 3 個）、`MainMode` enum を `common/` へ、REG 側も `AppImGuiBoot` に移行、`RegistrationImGuiManager` を `AppImGuiManager` 等にリネーム。詳細は `IMPLEMENTATION_PLAN_DEFORM_UI_v2.md` Section 10 のチェックリスト。

---

## 8. CMA-ES（third_party/c-cmaes）[feature/cmaes-upstream]

BIPOP-CMA-ES エンジン（registration/src の `CmaesRefineV2/V3/V3R/V3RS` + `CmaesUtils`）は、
旧・自作 CMA-ES 実装を廃し、**本家 Hansen c-cmaes** をアダプタ shim 経由で使用する。

**upstream/（本家無改変）と wrapper/（我々）にトップで二分**。GitHub 公開しやすい構成。

```
third_party/c-cmaes/
├── README.md                 ← 出自・ライセンス・構成・決定性の説明
├── upstream/                 ← 本家 Hansen c-cmaes を verbatim で vendoring（無改変・original names）
│   ├── src/{cmaes.c, cmaes.h, cmaes_interface.h, boundary_transformation.c/h}
│   └── CMakeLists.txt, LICENSE, README.md, compile, doc.txt, docfunctions.txt
└── wrapper/                  ← 我々のラッパ（書いたものは全部ここ）
    ├── cmaes.h               ← エンジンが #include する公開 API（旧自作と同一署名／gen,lambda,sigma + 不透明 impl）
    ├── cmaes_adapter.cpp     ← アダプタ本体（公開 API を本家へ橋渡し）
    ├── hansen_renames.h      ← 自動生成: cmaes_* → HANSEN_*（cmaes_boundary_* は除外）
    ├── hansen_unrenames.h    ← 自動生成: 上記を打ち消す
    └── hansen_cmaes_renamed.c ← C シム: renames + #include "../upstream/src/cmaes.c"
```

- **設計**：エンジンは 6 引数 `cmaes_init(N,xstart,sigma0,lambda,lb,ub)` のまま。本家のシンボルは
  *コンパイル時のマクロ*で `HANSEN_*` に改名し公開 `cmaes_*` と共存（**本家ソース・関数名は無改変**）。
  箱拘束は本家の smooth boundary transformation。CMake は `wrapper/hansen_cmaes_renamed.c` /
  `upstream/src/boundary_transformation.c` / `wrapper/cmaes_adapter.cpp` のみコンパイル。
  エンジンは `third_party/c-cmaes/wrapper/cmaes.h` を明示パスで include。`upstream/src/` は
  include path に入らないので公開 `cmaes.h` と本家 `cmaes.h` は衝突しない。
- **決定性（直列==並列ビット一致）**：①本家の per-instance RNG を `cmaes_set_seed`
  （`cmaes_random_Start`）で再シード、②`cmaes_TestForTermination` の共有 static 戻りバッファを
  `omp critical` で per-instance バッファへ退避、③**固有系更新の経過 CPU 時間ゲートを無効化**
  （`updateCmode.maxtime=1.0`）— これを外さないと並列負荷で更新スケジュールがズレ軌道が発散する。
  併せて `actparcmaes.par` の CWD 書き出しを抑止。
- **検証**：スタンドアロンで収束・直列決定性・並列8スレッド全ビット一致・別シード差異・.par 非書き出しを確認、
  `lsn_registration` のビルド/リンクも確認済み。**GUI 上の §（HANDOFF §7）検証は未実施**
  （Ctrl+G=V3R で実ポーズの直列==並列、その後 V3RS/Ctrl+I/V3/V2/V1）。
- 旧自作実装（旧 `third_party/自作のc-cmaes/`）と本家の不要ファイル（example_*.c, plotcmaesdat.*, *.par）は削除済み。
- 進捗・経緯：`memory/cmaes-upstream-migration.md`、`HANDOFF_upstream_cmaes_migration_v2.md`。
