# STATUS / 確定版作業計画: sam2_da3_lite → common OBJ export migration

> 親計画(背景資料): `REFACTOR_PLAN_sam2da_obj_migration.md`
> ブランチ: `refactor/sam2da-obj-migration`（`intrinsics-step-9-10` から分岐）
> 開始: 2026-05-27 / 本ファイルが最新の確定計画（元計画書 §4.1.2 等を以下で上書き）

## 目的（一行）
OBJ/intrinsics/texture の書き出しを sam2_da3_lite から `common/` へ移し、REG が
自前の K でプロセス内生成する。カノニカル（タグなし）名を DEFORM が読む正規ファイルにする。

## 確定した設計判断
- **Phase 1+2 を先に完遂 → 停止 → ユーザーが GPU で Run Depth 検証 → Phase 3 合意**で進める。
  Phase 3 以降の検証は実機 GPU が必須でエージェント環境では実行不可。
- Phase 2 の img:: 依存解決 = **`image_utils.hpp` ごと `common/src/` へ移す**（自己完結・std のみ依存・182行）。
  `RgbImageView` 化と texture 書き出し責務の REG 移管は **Phase 3** で実施。
- Phase 5 の sam2 リサイズ方針 = **1920×1080 cap 維持**（Step 9 現行挙動）。
- enum は `IntrinsicsSource::DA3`（旧 `Auto` から改名済み）。`intrinsicsSourceToTag` は
  `Custom→"custom" / Calib→"calib" / Preset→presetKey / DA3→"da3"`。

## ★ 採用した追加方針 4 点（2026-05-27、ユーザー指示）

### (1) DA3 推定 K の出力は残す（sam2→REG 方向の推論出力）
「sam2_da は K を一切扱わない」は **REG→sam2_da 方向**の話。**sam2_da→REG 方向**で DA3 が推定した
K（`DepthResult.intrinsics` の fx/fy/cx/cy）を出力する経路は**残す**＝推論結果の出力であり round-trip ではない。
- sam2 出力ファイル名は **`intrinsics_da3.txt`**（現状の `intrinsics.txt` から改名。REG カノニカル
  `intrinsics.txt` との衝突回避）。
- **DA3 は distortion を推定しない** → `intrinsics_da3.txt` は **fx/fy/cx/cy/width/height/name の 7 行のみ**、
  distortion セクションは省略。
- REG は `IntrinsicsSource::DA3` のソースとしてこれを読む（既存 promote 入力もこれに合わせる）。
- 反映先: Phase 3（DepthToObjExport 設計 / sam2 改名）、Phase 5（sam2 削減時も DA3 出力は KEEP）。

### (2) float depth 経路 = 既存 `depth_metric.bin` を流用
新規 `depth_raw.bin` は不要。sam2 main.cpp 678–697 の `depth_metric.bin`（現在 `#if 0`）が既に該当:
```
ヘッダ16B: magic=0x44455054("DEPT") / W(uint32) / H(uint32) / reserved(uint32)
本体: float32 × W*H  (depthRaw = メートル単位の生 depth)
```
- **Phase 3 で再有効化して流用**。読み込み側は magic で検証可能。
- 注意: 既存ヘッダは 16B（magic 込み）で、当初案の 8B(W,H) とは異なる。**既存16B形式を採用**。
- 現状ディスクに出ている depth は grayscale PNG（正規化済み＝metric 復元不可）なので bin が必須。

### (3) デバッグ CLI `depth_to_obj_tool`（低優先・Phase 2 完了後の余力で）
common 化した obj_exporter を呼ぶ薄い CLI。引数 = `depth_raw.bin(=depth_metric.bin) + mask.png +
texture.jpg + K`、出力 = `OBJ + .mtl`。用途 = A/B 比較（sam2 が直接書いた OBJ vs bin 経由再生成 OBJ の
diff）でリファクタ無損失性を実証。**タスクとして記録のみ**、実装は後回し可。

### (4) main.cpp 867–903（DA3-intrinsics OBJ `#if 0`）は Phase 1 で温存
元計画 §4.1.2 では削除対象だったが**変更**: `DepthResult.intrinsics` を読む部分が Phase 3 の
`intrinsics_da3.txt` 書き出しの前駆実装として再利用できる。
- **Phase 1: `#if 0` のまま残す**（削除しない）。
- **Phase 3: intrinsics 読み出し部分だけ切り出して使う**。

## 直近作業との関係（重要）
Step 9/10 + `intrinsics_custom_used.txt` + rectify-clear + sidecar の K 受け渡し機構は、
Phase 3/5 で大半が削除・置換される（REG-owns-K が上位互換）。→ K 受け渡し経路へのこれ以上の投資は避ける。

## Phase 1 死コード削除スコープ（確定）
削除する:
- `sam2_da3_lite/src/ply_exporter.cpp` + `include/ply_exporter.hpp`、`#include "ply_exporter.hpp"`、
  CMakeLists の ply_exporter 行
- main.cpp Relief `#if 0`（721–825）
- main.cpp 死コメント(PLY/Confidence) + HQ confidence `#if 0`（828–859）
- `Options` の `confPercentile/saveHq/saveRelief/reliefThickness` と CLI `--conf-percentile/--no-hq/
  --no-relief/--thickness`

**温存する（削除しない）:**
- `depth_metric.bin` `#if 0`（678–697）← (2) で Phase 3 再利用
- **DA3-intrinsics OBJ `#if 0`（861–900 / ユーザー表記 867–903）← (4) で Phase 3 再利用**
- live Kinect intrinsics OBJ ブロック（902+）← Phase 5 で削除予定

## Phase 2 の実装メモ（Option R、計画からの逸脱）
FINAL doc §2.2/§6.1 は「image_utils は std のみ依存の自己完結ヘッダ」と仮定していたが**誤り**:
`image_utils.cpp` は STB 実装 TU（STB_IMAGE/WRITE/RESIZE_IMPLEMENTATION）で、sam2 コア
（depth_anything/sam2_segmentor）が常時依存。reg/deform main.cpp も各々 STB 実装 TU。
→ image_utils.cpp を common に移すと reg/deform で **STB シンボル重複リンクエラー**。
**採用した Option R:**
- `obj_exporter.{cpp,hpp}` + `image_utils.hpp`（ヘッダのみ）を common/src へ移動。
- **`image_utils.cpp` は sam2 に残す**（sam2 コア／STB 実装）。
- obj_exporter.cpp の唯一の `img::saveImage` を local `saveImageStb()`（`stbi_write_png` 直呼び・
  元と bit-identical: PNG stride=W*ch, JPG q=95）に置換。実装は各実行ファイルが提供
  （reg/deform=main.cpp, sam2=image_utils.cpp）。→ obj_exporter は img::Image 型のみ依存。
- sam2 CMakeLists: obj_exporter.cpp を common パス参照に変更＋ include path に common/src 追加
  （Phase 5 まで sam2 が common の obj_exporter を使う橋渡し）。

## TODO（本タスク完了後の別 PR）
- **R2: STB 実装の1本化** — reg/deform main.cpp の STB_IMAGE_IMPLEMENTATION 群を撤去し、
  単一 TU（例 common の image_utils.cpp 等）に集約。STB 初期化順（SimpleCamera.hpp 等）に
  触れる中リスク作業なので本移管とは別 PR で。

## Phase 3 実装メモ
- sam2: `depth_metric.bin` 再有効化（16B "DEPT" ヘッダ）、`intrinsics.txt`→`intrinsics_da3.txt`
  に改名（writeIntrinsicsTxt が name=da3 + setprecision(9) の7行）。Kinect OBJ ブロック(902+)は並行存置。
- 新規 `common/src/DepthToObjExport.{h,cpp}`: exportDepthArtifacts（canonical+tagged+noskirt）/
  saveIntrinsicsFile / loadDepthMetricBin。`img::dilateMask` は local 複製（image_utils.cpp 非依存）。
  texture は obj_exporter の saveImageStb 任せ。
- `IntrinsicsSource.h`: `intrinsicsSourceToTag()` 追加。
- REG `runDepthAndUpdateScene`: 経路 A（bin→exportDepthArtifacts, RGB=original.jpg・depth解像度,
  mask=segmentation_mask.png, K=g_intrinsics）/ B（bin無→既存OBJ温存+intrinsics.txt）/ C（警告のみ）。
  objPath に canonical フォールバック追加。DA3 promote を `intrinsics_da3.txt`→`intrinsics_da3_last.txt` に更新。
- ODR fix: `mCutMesh.h::setUp()` を inline 化（common の新 .cpp と多重定義回避）。
- **既知の制限 → 解消済み (2026-05-27)**: K 解像度 ≠ depth 解像度（4K入力を sam2 が縮小した
  ケース）の K スケーリングは当初未実装(警告のみ)だったが、後続 feature (FEATURE_PLAN_external_depth_drop.md
  Task 1/2 の `common/src/IntrinsicsScaling.h::scaleIntrinsics`) を Run Depth 経路 A に適用して解消。
  Patient3 4K Custom で頂点数が 804k(半減)→939k に復活、検証済み。

## 進捗
- [x] Phase 1: 死コード掃除（867–903 温存・depth_metric.bin 温存）→ `[obj-migration p1]`
- [x] Phase 2: obj_exporter + image_utils.hpp を common/src へ移動（Option R）→ `[obj-migration p2]`
- [x] Phase 3: depth_metric.bin/intrinsics_da3.txt + DepthToObjExport + REG フック A/B/C → `[obj-migration p3]`
- [ ] 🛑 停止 → GPU で Run Depth 検証（ユーザー）= 現在地
- [ ] (task) depth_to_obj_tool 追加（低優先・Phase 2 後）
- [x] Phase 4: DEFORM カノニカル名 + _k4a フォールバック（4箇所 + test 更新）→ `[obj-migration p4]`
- [x] Phase 5: sam2 から OBJ/round-trip K/texture 削除（intrinsics_da3.txt KEEP）→ `[obj-migration p5a/p5b]`
      - p5a: sam2 main.cpp 910→583行（OBJ/texture/K Options+CLI 削除、depth_metric.bin/intrinsics_da3.txt KEEP）
      - p5b: DepthRunner config K フィールド+buildCmd K append 削除、REG applyIntrinsicsToRunnerConfig 撤去、objPath を canonical 化
      - ガードレール確認済: sam2 K round-trip=0 / OBJ-read 経路温存 / 外部OBJ A/B/C 温存
      - ★Shift+M snapshot(main.cpp 1465-1530)は tagged 名のまま=Phase 6 で canonical 化(現状 missing をコピーしないだけ、compile OK)
- [x] Phase 6 (コード/ドキュメント分): 起動時OBJ選択 canonical化 / Shift+M snapshot canonical化 /
      PROJECT_STRUCTURE §5 + OBJTargetExtraction.h コメント更新 → `[obj-migration p6]`
  - [ ] クリーン再生成テスト(§10.4) = ユーザーが GPU で実施（検証1+4 もここで）
  - [ ] (Optional/別PR) depth_to_obj_tool

## 検証コマンド
```
cmake --build build --target sam2_da3_lite lsn_registration lsn_deform -j$(nproc)
bash test/test_intrinsics_step1.sh && bash test/test_intrinsics_presets.sh && bash test/test_intrinsics_step4.sh
./build/bin/lsn_deform --dry-run
```
