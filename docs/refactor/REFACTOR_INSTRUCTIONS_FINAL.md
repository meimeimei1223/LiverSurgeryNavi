# 実装指示書 (統合確定版): sam2_da3_lite → common OBJ Export 移管

> **このドキュメントが最終版**。`REFACTOR_PLAN_sam2da_obj_migration.md` と
> `REFACTOR_sam2da_obj_STATUS.md` の内容を統合し、その後判明した追加方針
> (特に外部 OBJ 投入経路の温存) と既存資産 (depth_metric.bin 16B ヘッダ) を
> 反映した確定指示。
>
> プロジェクト: AAA_LiverSurgeryNaviComb
> ブランチ: `refactor/sam2da-obj-migration` (`intrinsics-step-9-10` から分岐)
> 対象読者: Claude Code (CLI agent)
> 最終更新: 2026-05-27

---

## 0. このドキュメントの読み方

旧 2 文書 (`REFACTOR_PLAN_sam2da_obj_migration.md` と `REFACTOR_sam2da_obj_STATUS.md`)
の関係:

- 旧計画書 = 全体方針と各 Phase の詳細
- 旧 STATUS = 着手前の追加方針 4 点 (DA3 推定 K, depth_metric.bin 流用, デバッグ CLI, 867-903 温存)
- **本書 = 上記 2 つに加えて「外部 OBJ 投入経路の温存」(追加方針 5 点目) と
  enum 改名 (`IntrinsicsSource::DA3`) を反映した確定版**

矛盾があった場合は本書を最優先。

---

## 1. 目的 (1 行)

OBJ / intrinsics / texture の書き出しを `sam2_da3_lite` から `common/` へ移し、
REG が自前の K でプロセス内生成する。カノニカル (タグなし) 名を DEFORM が読む正規ファイルにする。
**ただし将来 DA3 以外の depth ソースに対応するため、外部から OBJ を直投入できる経路は温存する**。

---

## 2. 確定した設計判断 (一覧)

### 2.1 進め方
- **Phase 1+2 を先に完遂 → 停止 → ユーザーが GPU で Run Depth 検証 → Phase 3 合意** で進める
- Phase 3 以降の検証は実機 GPU が必須でエージェント環境では実行不可
- 各 Phase で git commit、Phase 内では小ステップごとにビルド確認

### 2.2 Phase 2 の依存解決
- `obj_exporter.{cpp,hpp}` だけでなく **`image_utils.hpp` も同時に `common/src/` へ移す**
  (`image_utils.hpp` は std のみ依存・182 行の自己完結ヘッダなので安全)
- `RgbImageView` 化と texture 書き出し責務の REG 移管は **Phase 3** で実施

### 2.3 sam2 リサイズ方針
- **1920×1080 cap 維持** (Step 9 現行挙動を変更しない)
- 「本来のリファクタの主目的は責務整理であり、推論挙動を変えることではない」

### 2.4 enum / タグ命名
- `IntrinsicsSource::DA3` (旧 `Auto` から改名済み)
- `intrinsicsSourceToTag()`:
  - `Custom` → `"custom"`
  - `Calib` → `"calib"`
  - `Preset` → presetKey (例 `"k4a"`)
  - `DA3` → `"da3"`

### 2.5 K の流れの原則 (改訂版)
**「REG が K を所有する」原則** は維持。ただし方向を区別する:
- **REG → sam2_da 方向**: K を渡さない (`--kinect-intrinsics`, `--intrinsics-source` 等の CLI 削除)
- **sam2_da → REG 方向**: DA3 が推定した K (`DepthResult.intrinsics`) を `intrinsics_da3.txt` として出力
  - これは「推論結果の出力」であって K の round-trip ではない (sam2_da が REG の K を書き戻す処理ではない)
  - `IntrinsicsSource::DA3` のソースとして REG が読む
- **DA3 は distortion を推定しない** ので `intrinsics_da3.txt` は 7 行のみ
  (fx/fy/cx/cy/width/height/name、distortion セクション省略)

### 2.6 float depth 経路
- 新規ファイル形式は作らない。**既存 `depth_metric.bin` (sam2 main.cpp 678-697 行、現在 `#if 0`) を流用**
- ヘッダ形式は既存仕様を採用:
  ```
  ヘッダ 16 バイト:
    magic    = 0x44455054 ("DEPT" little-endian)  uint32
    width    = W                                   uint32
    height   = H                                   uint32
    reserved = 0                                   uint32
  本体: float32 × W × H  (メートル単位の生 depth、row-major)
  ```
- Phase 3 で再有効化して流用。読み込み側は magic で検証可能
- 当初の私の 8B ヘッダ案より優れているので **16B 既存仕様を採用**

### 2.7 外部 OBJ 直投入経路の温存 (★追加方針 5)
**将来 DA3 以外の depth ソース (Kinect SDK 生 depth, RealSense, Stereo, 別モデル) を使う想定があるので、
REG は「depth_metric.bin がなくても、OBJ ファイルさえ `depth_output/pc_metric_pinhole_masked.obj` に
置けば動く」状態を保つ**。

具体的に守ること:
1. **Phase 5 完了後も、REG の OBJ 読み込み経路 (`mCutMesh::loadMeshFromFile` +
   `OBJTargetExtraction.h:extractTargetFromOBJ`) は絶対に削除しない**
2. **Run Depth ハンドラの設計** — `depth_metric.bin` が存在すれば自動で
   `depthexport::exportDepthArtifacts()` を呼んで OBJ 生成、**存在しなければスキップして既存 OBJ を
   そのまま使う**
3. **`intrinsics.txt` は OBJ とは独立に書く** — bin → OBJ 経路がスキップされても、REG は
   現在の K (`IntrinsicsSource` 由来) を `intrinsics.txt` カノニカルに dump する
4. **エラー処理** — bin も OBJ もない場合は明確なエラーで「depth ソースが何もない」ことを
   ユーザーに知らせる。サイレントに進めない

将来的な depth ソース抽象 (`IDepthSource` interface 等) は別タスク。
本リファクタでは上記 1〜4 を守れば、抽象化なしでも「DA3 ロックインしない」状態は維持できる。

---

## 3. アーキテクチャ Before / After

### 3.1 Before (現状)

```
[REG] K を所有
   ├─ DepthRunner.run() ─CLI args (--kinect-intrinsics 等)─→ [sam2_da3_lite]
   │                                  ├─ DA3 推論 → depth_*.png
   │                                  ├─ SAM2 推論 → segmentation_mask.png
   │                                  ├─ Vignette 検出
   │                                  ├─ K で OBJ Export
   │                                  └─ K を intrinsics_<tag>.txt に round-trip
   │                                          │
   │            depth_output/{intrinsics_<tag>.txt, pc_metric_pinhole_*_<tag>*.obj}
   │                                          │
   └─ extractTargetFromOBJ ←─ mCutMesh::loadMeshFromFile ←─┘

[DEFORM] _k4a ハードコードで読む
```

### 3.2 After (目標)

```
[sam2_da3_lite] K は受け取らない
   ├─ DA3 推論 → depth_metric.bin (float32, 16B ヘッダ) + depth_*.png (可視化)
   ├─ DA3 推定 K → intrinsics_da3.txt (推論結果として、7 行)
   ├─ SAM2 推論 → segmentation_mask.png, instrument_segmentation_mask.png
   └─ Vignette 検出
                          │
[REG] K を所有 (Custom / Calib / Preset / DA3 のいずれか)
   ├─ Run Depth 完了後フック:
   │    if (depth_metric.bin exists)
   │        depth + mask + 自前 K で common::exportDepthArtifacts() 呼ぶ
   │             → カノニカル OBJ + タグ付きデバッグ OBJ + intrinsics.txt
   │    else if (既存 OBJ exists)
   │        スキップ (外部投入された OBJ をそのまま使う)
   │    else
   │        エラー
   │
   └─ extractTargetFromOBJ ←─ mCutMesh::loadMeshFromFile (経路温存)

[DEFORM] カノニカル名読み + _k4a レガシー fallback
```

### 3.3 K の方向別整理

| 方向 | リファクタ前 | リファクタ後 |
|---|---|---|
| REG → sam2_da | `--kinect-intrinsics`, `--intrinsics-source`, `--kinect-distortion` で渡す | **渡さない** (CLI 削除) |
| sam2_da → REG | `intrinsics_<tag>.txt` (REG の K を round-trip) | `intrinsics_da3.txt` (**DA3 推定値のみ**、7 行) |
| REG 内部 | `IntrinsicsSource` enum で切替 | 同上 (`DA3` ソースは `intrinsics_da3.txt` を読む) |
| REG → ディスク (DEFORM 用) | sam2 経由で `intrinsics_<tag>.txt` | REG 直書きで `intrinsics.txt` (カノニカル) + `intrinsics_<tag>.txt` (デバッグ) |

---

## 4. 出力ファイル命名規則 (確定版)

### 4.1 sam2_da3_lite の出力 (リファクタ後)

正規データ (REG が機械処理する):
| ファイル | 形式 | 用途 |
|---|---|---|
| `segmentation_mask.png` | 8-bit grayscale | 肝臓マスク |
| `instrument_segmentation_mask.png` | 8-bit grayscale | 器具マスク (SAM2 + Vignette merge) |
| `depth_metric.bin` | float32 + 16B ヘッダ | metric depth (REG が読む正規データ) ★再有効化 |
| `intrinsics_da3.txt` | テキスト 7 行 | DA3 推定 K (Auto/DA3 モード用) |
| `original.jpg` | JPEG | 入力画像のコピー |

デバッグ可視化:
- `depth_full.png`, `depth_masked.png`, `depth_masked_renorm.png` (+ colored 版)
- `segmentation_overlay.jpg`, `instrument_segmentation_overlay.jpg`

**sam2_da から削除されるもの:**
- `pc_metric_pinhole_*_<tag>*.obj` (全種)
- `texture.png`
- `intrinsics_<tag>.txt` (round-trip ファイル、`intrinsics_da3.txt` だけは残す)
- 関連する .mtl

### 4.2 REG が書く出力 (リファクタ後)

| ファイル | 用途 | DEFORM が読む |
|---|---|---|
| `pc_metric_pinhole_masked.obj` | カノニカル target mesh | ✅ |
| `pc_metric_pinhole_full_light.obj` | カノニカル board mesh (stride=10) | ✅ |
| `intrinsics.txt` | カノニカル K + distortion | ✅ |
| `texture.png` | カノニカル | ✅ |
| `pc_metric_pinhole_masked_<tag>.obj` | デバッグ (cross-K 比較) | ❌ |
| `pc_metric_pinhole_full_<tag>_light.obj` | デバッグ | ❌ |
| `pc_metric_pinhole_masked_<tag>_noskirt.obj` | デバッグ (skirt 影響検証) | ❌ |
| `pc_metric_pinhole_full_<tag>_light_noskirt.obj` | デバッグ | ❌ |
| `intrinsics_<tag>.txt` | デバッグ | ❌ |

`<tag>` の値は `intrinsicsSourceToTag()` の戻り値 (custom / calib / k4a 等 / da3)。

---

## 5. Phase 1: 死コード掃除 (sam2_da3_lite のみ、挙動ゼロ変化)

**ブランチ:** `refactor/sam2da-obj-migration`
**コミットメッセージ:** `[obj-migration p1] death-code cleanup`
**影響範囲:** `sam2_da3_lite/` のみ

### 5.1 削除する

#### 5.1.1 ply_exporter ファイルごと
- `sam2_da3_lite/src/ply_exporter.cpp`
- `sam2_da3_lite/include/ply_exporter.hpp` (実所在を確認してから)
- `main.cpp` の `#include "ply_exporter.hpp"` 行
- `sam2_da3_lite/CMakeLists.txt` の sources から `ply_exporter.cpp` 行

#### 5.1.2 main.cpp 内の `#if 0` ブロック (削除する分)

| 範囲 | 内容 |
|---|---|
| 721-825 | Relief OBJ/PLY 出力 |
| 828-859 | 死コメント (PLY/HQ confidence) + HQ confidence `#if 0` |

各ブロックを `#if 0` 〜 `#endif` ごと削除。先頭の `[COMMENTED OUT]` コメント行も併せて削除。

#### 5.1.3 Options フィールドと CLI フラグ

`Options` 構造体から:
- `saveRelief` (デフォルト true)
- `saveHq`
- `confPercentile`
- `reliefThickness`

CLI (`printUsage` と `parseArgs` 両方) から:
- `--no-relief`
- `--no-hq`
- `--conf-percentile`
- `--thickness`

**先に grep で他参照ゼロを確認すること:**
```bash
grep -rn "saveRelief\|saveHq\|confPercentile\|reliefThickness" sam2_da3_lite/
```

### 5.2 温存する (削除しない、★重要)

以下の `#if 0` ブロックは **Phase 3 で再利用する** ので削除しない:

#### 5.2.1 `depth_metric.bin` 書き出しブロック (main.cpp 678-697 行付近)
理由: §2.6 で確定したように、Phase 3 で再有効化して float metric depth の正規受け渡し経路に使う。
16B ヘッダ (magic="DEPT" + W + H + reserved) + float32 本体の既存形式をそのまま流用する。

#### 5.2.2 DA3-intrinsics OBJ ブロック (main.cpp 861-900 / 旧表記 867-903 行)
理由: `DepthResult.intrinsics` を読む部分が Phase 3 の `intrinsics_da3.txt` 書き出しの前駆実装。
Phase 3 で intrinsics 読み出し部分だけ切り出して使う。

#### 5.2.3 live Kinect intrinsics OBJ ブロック (main.cpp 902 行以降)
理由: Phase 5 で削除予定。Phase 1 ではまだ生かす (DEFORM が現状のままなら動く必要がある)。

### 5.3 検証

```bash
# 1. ビルド
cmake --build build --target sam2_da3_lite -j$(nproc)

# 2. main.cpp の行数確認 (1054 → 推定 850-900 行台)
wc -l sam2_da3_lite/src/main.cpp

# 3. 既存 Run Depth テスト (ユーザーが GPU マシンで実行)
#    出力ファイルが Phase 1 前と byte-identical であることを期待
```

**Phase 1 完了条件:**
- sam2_da3_lite ビルド成功
- main.cpp が ~850-900 行台
- 削除した `Options` フィールド・CLI フラグへの参照がゼロ
- 温存対象 3 ブロックが残っている

---

## 6. Phase 2: obj_exporter + image_utils を common/src/ へ移動

**コミットメッセージ:** `[obj-migration p2] move obj_exporter + image_utils to common/`
**影響範囲:** `sam2_da3_lite/`, `common/src/`, ルート CMakeLists.txt
**目標:** API 無改変で挙動ゼロ変化

### 6.1 ファイル移動

```bash
git mv sam2_da3_lite/src/obj_exporter.cpp common/src/obj_exporter.cpp
git mv sam2_da3_lite/include/obj_exporter.hpp common/src/obj_exporter.hpp
git mv sam2_da3_lite/include/image_utils.hpp common/src/image_utils.hpp
```

(実パスは `sam2_da3_lite/` 内の構造に合わせて調整)

**`image_utils.cpp` も同時移動するか確認:**
```bash
ls sam2_da3_lite/src/image_utils.cpp 2>/dev/null
```
存在すれば `common/src/` へ移動。

### 6.2 CMakeLists 修正

#### 6.2.1 ルート `CMakeLists.txt`
- `common/src/*.cpp` の glob は既に存在する (line 169 付近で確認)
- 移動した `obj_exporter.cpp` (と必要なら `image_utils.cpp`) が自動でリンクされる
- 必要な include path も `common/src` 経由で解決される

#### 6.2.2 `sam2_da3_lite/CMakeLists.txt`
- sources から `obj_exporter.cpp` 行を削除
- (image_utils.cpp が移動なら) sources から `image_utils.cpp` 行を削除
- `target_include_directories` に `${CMAKE_SOURCE_DIR}/common/src` を追加
  (Phase 5 まで sam2 が common の obj_exporter / image_utils を使う橋渡し)

例:
```cmake
target_include_directories(sam2_da3_lite PRIVATE
    src
    include
    ${CMAKE_SOURCE_DIR}/common/src    # ← 追加
    # 既存の onnxruntime include 等
)
```

### 6.3 include 文の調整

sam2_da3_lite/main.cpp 等で `#include "obj_exporter.hpp"` `#include "image_utils.hpp"` が
そのまま解決されることを確認。include path で見えていれば書き換え不要。

`obj_exporter.cpp` の内部 include (`#include "image_utils.hpp"` 等) も path 経由で
解決できるか確認。

### 6.4 API は無改変 (重要)

Phase 2 では `obj_exporter` の関数シグネチャは一切変更しない。`img::Image` 依存も
そのまま (`image_utils.hpp` を一緒に移したので壊れない)。
`RgbImageView` 化と texture 書き出し責務の REG 移管は **Phase 3** で実施する。

### 6.5 検証

```bash
# 1. 3 ターゲット全ビルド
cmake --build build --target sam2_da3_lite lsn_registration lsn_deform -j$(nproc)

# 2. lsn_deform --dry-run
./build/bin/lsn_deform --dry-run

# 3. 既存 intrinsics テスト
bash test/test_intrinsics_step1.sh
bash test/test_intrinsics_presets.sh
bash test/test_intrinsics_step4.sh

# 4. sam2_da3_lite の OBJ 出力が Phase 2 前と byte-identical (ユーザーが GPU で確認)
```

**Phase 2 完了条件:**
- 全ターゲットビルド成功
- 全 intrinsics テスト pass
- ファイルが `common/src/` に存在し、`sam2_da3_lite/src/` から消えている
- sam2_da3_lite の挙動が Phase 2 前と完全同一

### 6.6 (Optional) デバッグツール `depth_to_obj_tool` のタスク記録

Phase 2 完了後の余力で実装する低優先タスク:
- 引数: `depth_metric.bin` (16B ヘッダ込み) + mask.png + texture.jpg + K (CLI または intrinsics ファイル)
- 出力: `OBJ + .mtl`
- 用途: A/B 比較 (sam2 が直接書いた OBJ と bin 経由再生成 OBJ の diff) でリファクタ無損失性を実証
- 場所: `tools/depth_to_obj/main.cpp` 等の新ディレクトリ
- 優先度: 低 (Phase 5 完了後でも可)

Phase 2 のコミット内またはセパレートコミットで TODO として STATUS に記録のみ。実装は後回し可。

---

## 🛑 ここで一旦停止

Phase 1+2 完了後、以下をユーザーが GPU マシンで確認する:

1. **既存挙動の維持**:
   ```bash
   # Phase 2 完了前にバックアップしておいた depth_output と比較
   diff -r depth_output/ depth_output_baseline_pre_p1p2/
   ```
   完全一致を期待 (Phase 1+2 はゼロ変化リファクタ)

2. **REG/DEFORM の動作**:
   - `lsn_registration` で Run Depth 実行 → OBJ が生成される (sam2 経由)
   - `lsn_deform` で Target/Board が表示される

3. **問題なければ Phase 3 着手承認をユーザーから受ける**

---

## 7. Phase 3: float depth 経路再有効化 + REG 側 OBJ 生成

**コミットメッセージ:** `[obj-migration p3] enable depth_metric.bin + REG-side OBJ export`
**影響範囲:** sam2_da3_lite/main.cpp, common/src/ (新規ヘッダ), registration/src/, IntrinsicsSource.h
**目標:** REG が自前の K で OBJ + intrinsics.txt + texture.png を書く。sam2 側 OBJ 書き出しは並行存置 (フォールバック)

### 7.1 sam2_da3_lite 側の変更

#### 7.1.1 `depth_metric.bin` 書き出しを再有効化

main.cpp 678-697 行の `#if 0` を外す。既存ヘッダ形式 (16B) を維持:

```cpp
// depth_metric.bin: float32 metric depth, row-major, with 16-byte header
{
    std::string binPath = opts.outputDir + "/depth_metric.bin";
    std::ofstream ofs(binPath, std::ios::binary);
    if (ofs.is_open()) {
        uint32_t magic    = 0x44455054;  // "DEPT"
        uint32_t W        = (uint32_t)image.width;
        uint32_t H        = (uint32_t)image.height;
        uint32_t reserved = 0;
        ofs.write((const char*)&magic, 4);
        ofs.write((const char*)&W, 4);
        ofs.write((const char*)&H, 4);
        ofs.write((const char*)&reserved, 4);
        ofs.write((const char*)depthRaw.data(),
                  depthRaw.size() * sizeof(float));
        std::cout << "[depth_metric.bin] saved: " << binPath
                  << " (" << W << "x" << H << ", float32)" << std::endl;
    }
}
```

#### 7.1.2 `intrinsics.txt` を `intrinsics_da3.txt` に改名

main.cpp 861-900 行の `#if 0` (DA3-intrinsics OBJ ブロック) から、**`intrinsics.txt` 書き出し部分だけ
切り出して活コードに変える**。OBJ 書き出し部分は捨てる。

新しい書き出しコード (7 行のみ、distortion なし):

```cpp
// DA3 が推定した K を出力 (推論結果として)
if (depthResult.hasIntrinsics && depthResult.intrinsics.valid()) {
    std::string intrPath = opts.outputDir + "/intrinsics_da3.txt";
    std::ofstream ofs(intrPath);
    if (ofs.is_open()) {
        ofs << std::setprecision(9);
        ofs << "fx "     << depthResult.intrinsics.fx << "\n";
        ofs << "fy "     << depthResult.intrinsics.fy << "\n";
        ofs << "cx "     << depthResult.intrinsics.cx << "\n";
        ofs << "cy "     << depthResult.intrinsics.cy << "\n";
        ofs << "width "  << image.width  << "\n";
        ofs << "height " << image.height << "\n";
        ofs << "name   " << "da3" << "\n";
        // distortion セクションは省略 (DA3 は推定しない)
        std::cout << "[intrinsics_da3.txt] Saved (DA3 estimated K)" << std::endl;
    }
}
```

**重要:** 旧 `intrinsics.txt` (DA3-estimated K を書いていたファイル名) は **REG のカノニカル
`intrinsics.txt` と衝突するので使わない**。必ず `intrinsics_da3.txt` に改名する。

#### 7.1.3 既存の Kinect intrinsics OBJ ブロック (main.cpp 902 行以降) は **Phase 3 では残す**
Phase 5 で削除する。Phase 3 では並行存置することで、REG 側の新 OBJ と sam2 側の旧 OBJ を
diff で比較できる状態を作る。

### 7.2 新規: `common/src/DepthToObjExport.h`

```cpp
#pragma once
// =============================================================================
//  DepthToObjExport.h
//  ---------------------------------------------------------------------------
//  REG 側で Run Depth 後に呼ぶラッパー。sam2_da3_lite が以前担っていた
//  OBJ + intrinsics.txt + texture.png の書き出し責務をこちらに移す。
//
//  外部 OBJ 投入経路の温存:
//      depth_metric.bin がディスクに無いときは exportDepthArtifacts() は
//      呼ばれず、ユーザーが手で置いた OBJ ファイルがそのまま使われる
//      (この判断は呼び出し側で行う)。
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>
#include "obj_exporter.hpp"
#include "OBJTargetExtraction.h"  // CameraIntrinsics

namespace depthexport {

struct Request {
    // --- 入力データ ---
    const uint8_t* rgbPixels;             // H*W*3 (RGB, row-major)
    int width;
    int height;
    const std::vector<float>* depthMetric;  // size = W*H, メートル単位
    const std::vector<uint8_t>* mask;       // size = W*H, 0/255
    Reg3DCustom::CameraIntrinsics K;        // REG が持っている K
    std::string tag;                        // intrinsicsSourceToTag() の戻り値

    // --- 出力先 ---
    std::string outDir;                     // = DEPTH_OUTPUT_PATH

    // --- オプション ---
    float skirtThreshold    = 0.05f;
    int   fullMeshStride    = 10;
    bool  writeTaggedCopies = true;
    bool  writeNoSkirtVariants = true;
    bool  writeTextureImage = true;
    int   maskDilate        = 0;
};

struct Result {
    bool ok = false;
    std::string canonicalMaskedObj;
    std::string canonicalFullObj;
    std::string canonicalIntrinsics;
    std::string canonicalTexture;
    std::vector<std::string> debugCopies;
};

Result exportDepthArtifacts(const Request& req);

// intrinsics.txt 単体書き出し (.name フィールドから distortion 込み)
// OBJ 経路スキップ時 (外部 OBJ 投入時) にも単独で呼べる
bool saveIntrinsicsFile(const std::string& path,
                        const Reg3DCustom::CameraIntrinsics& K,
                        int width, int height);

// depth_metric.bin 読み込みヘルパー
// ヘッダ検証 (magic="DEPT") を含む
struct DepthBin {
    int width = 0;
    int height = 0;
    std::vector<float> depth;
    bool valid() const { return width > 0 && height > 0 && !depth.empty(); }
};
bool loadDepthMetricBin(const std::string& path, DepthBin& out);

} // namespace depthexport
```

実装 (`common/src/DepthToObjExport.cpp`) で:
1. **`writeTextureImage` true** のとき: stb_image_write で `texture.png` を書く
2. **カノニカル**: `objexp::saveFullMeshObj("pc_metric_pinhole_full_light.obj", stride=10)` と
   `objexp::saveMaskedMeshObj("pc_metric_pinhole_masked.obj")`
3. **タグ付きコピー** (`writeTaggedCopies` true): `_<tag>` 入りで同じ 2 種類
4. **`_noskirt` 変種** (`writeNoSkirtVariants` true): タグ付きで 2 種類追加
5. **`intrinsics.txt` (カノニカル)** + **`intrinsics_<tag>.txt` (デバッグ)** を書く
6. **`saveIntrinsicsFile()`** は §7.1.2 と同じ形式 (ただし `distortion` セクションは
   K に値があれば書く、全部ゼロなら省略)

### 7.3 `IntrinsicsSource.h` に追加するヘルパー

```cpp
// IntrinsicsSource.h 末尾に追加

#include <string>

inline std::string intrinsicsSourceToTag(IntrinsicsSource s,
                                         const std::string& presetKey = "k4a") {
    switch (s) {
        case IntrinsicsSource::Custom: return "custom";
        case IntrinsicsSource::Calib:  return "calib";
        case IntrinsicsSource::Preset: return presetKey;
        case IntrinsicsSource::DA3:    return "da3";
    }
    return "unknown";
}
```

### 7.4 REG main.cpp / RegistrationActions.h での Run Depth 後フック

呼び出し位置を grep で特定:
```bash
grep -rn "DepthRunner\|depthRunner.run\|RunDepth" registration/
```

新しい呼び出しパターン (**外部 OBJ 投入経路の温存を含む**):

```cpp
// Run Depth の完了ハンドラ内

DepthRunnerResult depthRes = depthRunner.run(imagePath, points);
if (!depthRes.success) {
    // エラー処理
    return;
}

// --- 新規: REG 側で OBJ + intrinsics + texture を書く ---
std::string binPath = DEPTH_OUTPUT_PATH + "depth_metric.bin";
std::string existingObjPath = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj";

if (std::filesystem::exists(binPath)) {
    // ★ 経路 A: sam2 が depth_metric.bin を書いた → REG が OBJ を生成する
    depthexport::DepthBin db;
    if (depthexport::loadDepthMetricBin(binPath, db)) {
        // RGB と mask をロード
        int W, H;
        std::vector<uint8_t> rgb = loadRgb(depthRes.originalPath, W, H);
        std::vector<uint8_t> mask = loadMask(depthRes.segmentationMaskPath);

        // 現在の K を取得 (g_intrinsicsSource 由来)
        Reg3DCustom::CameraIntrinsics K = currentIntrinsics();
        std::string tag = intrinsicsSourceToTag(g_intrinsicsSource, g_currentPresetKey);
        K.name = tag;

        // エクスポート呼び出し
        depthexport::Request req;
        req.rgbPixels       = rgb.data();
        req.width           = db.width;
        req.height          = db.height;
        req.depthMetric     = &db.depth;
        req.mask            = &mask;
        req.K               = K;
        req.tag             = tag;
        req.outDir          = DEPTH_OUTPUT_PATH;
        req.writeTextureImage = true;
        // maskDilate, skirtThreshold 等は既存の DepthRunnerConfig 値から引き継ぐ

        auto exportRes = depthexport::exportDepthArtifacts(req);
        if (!exportRes.ok) {
            std::cerr << "[REG] exportDepthArtifacts failed" << std::endl;
        }
    }
} else if (std::filesystem::exists(existingObjPath)) {
    // ★ 経路 B (外部 OBJ 投入): ユーザーが OBJ を手で置いた、bin がない
    //    → OBJ 生成はスキップ、intrinsics.txt だけは書く
    std::cout << "[REG] depth_metric.bin not found, using existing OBJ as-is: "
              << existingObjPath << std::endl;

    Reg3DCustom::CameraIntrinsics K = currentIntrinsics();
    int W = K.width, H = K.height;  // K に解像度情報がある前提
    depthexport::saveIntrinsicsFile(DEPTH_OUTPUT_PATH + "intrinsics.txt", K, W, H);
} else {
    // ★ 経路 C: bin も OBJ もない → エラー
    std::cerr << "[REG] ERROR: no depth source (depth_metric.bin nor existing OBJ found)"
              << std::endl;
    // UI にエラー表示
    return;
}

// 以降の処理 (mCutMesh::loadMeshFromFile → extractTargetFromOBJ) は変更なし
```

### 7.5 並行存置による diff 検証

Phase 3 の時点では sam2 の Kinect intrinsics OBJ ブロック (main.cpp 902+) **は残す**。
これにより:
- sam2 が `pc_metric_pinhole_masked_<tag>.obj` を書く (旧経路)
- REG が `pc_metric_pinhole_masked.obj` (カノニカル) と `pc_metric_pinhole_masked_<tag>.obj` を書く (新経路、上書き)
- 上書きの結果、ディスク上に残るのは REG 経由のもの
- diff で「sam2 が書いたタグ付き OBJ をバックアップして、REG が書いたタグ付き OBJ と比較」できる

ユーザー検証手順 (GPU マシンで):
```bash
# 1. sam2 だけが書く状態でバックアップ
# (Phase 3 着手前にユーザーが取っておく)
cp depth_output/pc_metric_pinhole_masked_custom.obj /tmp/from_sam2da.obj

# 2. Phase 3 適用後に Run Depth
# 3. 比較
diff /tmp/from_sam2da.obj depth_output/pc_metric_pinhole_masked_custom.obj
# 期待: 浮動小数点誤差レベル (printf "%.6f" 桁内) の差のみ
```

### 7.6 検証

```bash
# 1. ビルド
cmake --build build -j$(nproc)

# 2. 既存テスト
bash test/test_intrinsics_step1.sh
bash test/test_intrinsics_presets.sh
bash test/test_intrinsics_step4.sh

# 3. Run Depth → カノニカル名のファイルが揃う (ユーザー検証)
ls depth_output/pc_metric_pinhole_masked.obj
ls depth_output/pc_metric_pinhole_full_light.obj
ls depth_output/intrinsics.txt
ls depth_output/texture.png
ls depth_output/depth_metric.bin
ls depth_output/intrinsics_da3.txt

# 4. タグ付きデバッグコピーも存在
ls depth_output/pc_metric_pinhole_masked_custom.obj
ls depth_output/intrinsics_custom.txt

# 5. 外部 OBJ 投入経路 (経路 B) の動作確認
rm depth_output/depth_metric.bin
# (Run Depth 実行をスキップして、既存 OBJ + intrinsics.txt が再書き込みされるか確認)

# 6. エラー経路 (経路 C) の確認
rm depth_output/depth_metric.bin depth_output/pc_metric_pinhole_masked*.obj
# Run Depth でエラー表示されること
```

**Phase 3 完了条件:**
- カノニカル名の OBJ + intrinsics.txt + texture.png が REG 経由で書かれる
- タグ付き OBJ が sam2 出力と等価 (浮動小数点誤差レベル)
- `depth_metric.bin` がない場合に既存 OBJ がそのまま使われる (経路 B)
- bin も OBJ もない場合に明確なエラーが出る (経路 C)
- `intrinsics_da3.txt` が 7 行で書かれる
- DEFORM はまだ `_k4a` を読んでいるが動作不変

---

## 8. Phase 4: DEFORM のファイル名カノニカル化

**コミットメッセージ:** `[obj-migration p4] DEFORM read canonical names with _k4a fallback`
**影響範囲:** `deform/src/DeformPipeline.h` 4 箇所

### 8.1 `loadDeformIntrinsics()` (line 170-178 付近)

```cpp
inline Reg3DCustom::CameraIntrinsics loadDeformIntrinsics() {
    Reg3DCustom::CameraIntrinsics K;

    // カノニカルパスを最優先 (REG が Run Depth 完了時に書く)
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics.txt", K)
        && K.valid()) {
        return K;
    }

    // 移行期 fallback: 旧 _k4a タグ付きファイル
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt", K)
        && K.valid()) {
        std::cout << "[Deform] (warn) intrinsics.txt not found, falling back to "
                  << "intrinsics_k4a.txt. Re-run Run Depth in REG to update."
                  << std::endl;
        return K;
    }

    std::cout << "[Deform] intrinsics.txt and intrinsics_k4a.txt both missing/invalid"
              << " -> K4A 720p hardcode fallback" << std::endl;
    // K = K4A 720p fallback (既存処理を残す);
    return K;
}
```

### 8.2 board / target mesh ロード (line 219, 274, 307)

3 箇所とも同じパターン。例 (line 274):

```cpp
// 変更前
const std::string p = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked_k4a.obj";

// 変更後
std::string p = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked.obj";
if (!std::filesystem::exists(p)) {
    std::string legacy = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_masked_k4a.obj";
    if (std::filesystem::exists(legacy)) {
        std::cout << "[Deform] (warn) using legacy " << legacy << std::endl;
        p = legacy;
    }
}
```

line 219 と 307 の `pc_metric_pinhole_full_k4a_light.obj` も同様に
`pc_metric_pinhole_full_light.obj` を最優先、`_k4a_light` をフォールバック。

### 8.3 検証

```bash
# 1. ビルド
cmake --build build --target lsn_deform -j$(nproc)

# 2. 新カノニカル OBJ がある状態で起動
ls depth_output/pc_metric_pinhole_masked.obj depth_output/intrinsics.txt
./build/bin/lsn_deform --dry-run
# -> 正しい K がロードされ、verts > 0

# 3. レガシー fallback 確認
mv depth_output/pc_metric_pinhole_masked.obj /tmp/
mv depth_output/intrinsics.txt /tmp/
./build/bin/lsn_deform --dry-run
# -> "(warn) using legacy ..." のログが出て、動作すること
mv /tmp/pc_metric_pinhole_masked.obj depth_output/
mv /tmp/intrinsics.txt depth_output/

# 4. フル DEFORM ワークフロー (REG で reg_liver.obj 作成後)
./build/bin/lsn_deform
```

**Phase 4 完了条件:**
- カノニカル名優先、レガシー `_k4a` 名に fallback
- `--dry-run` および本番起動が両方動作
- 警告ログが適切に出る

---

## 9. Phase 5: sam2_da3_lite から OBJ 関連を削除

**コミットメッセージ:** `[obj-migration p5] remove OBJ/round-trip K/texture from sam2_da3_lite`
**影響範囲:** `sam2_da3_lite/main.cpp`, `common/src/DepthRunner.h`
**★ 外部 OBJ 投入経路温存ガードレール適用フェーズ**

### 9.1 sam2_da3_lite/main.cpp の削除対象

#### 9.1.1 `Options` 構造体から削除

```cpp
bool  saveObjFull;
bool  saveObjMasked;
bool  hasKinectIntrinsics;
float kinectFx, kinectFy, kinectCx, kinectCy;
float kinectK1, kinectK2, kinectK3, kinectK4;
float kinectP1, kinectP2;
std::string intrinsicsSourceName;
int   maskDilate;
objexp::ZMode zMode;
float skirtThreshold;
```

#### 9.1.2 CLI フラグから削除 (`printUsage` と `parseArgs` 両方)

```
--kinect-intrinsics
--kinect-distortion
--intrinsics-source
--no-kinect
--no-obj-full
--no-obj-masked
--skirt
--zmode
--dilate
```

#### 9.1.3 main 関数本体から削除するブロック

| 範囲 (Phase 1 後の行番号で再確認) | 内容 |
|---|---|
| 414-453 付近 (リサイズロジック) | `intrinsicsSourceName != "k4a"` の skipResize 分岐削除。1920×1080 cap 維持 |
| 704-722 付近 | `depthForOutput` 計算 (zMode 依存) → metric (素の depth) のみで OK |
| 902 行以降 (Kinect intrinsics OBJ ブロック) | 全削除。テクスチャ書き出しも含む |

**ただし以下は KEEP:**
- `depth_metric.bin` 書き出し (Phase 3 で活コード化済み)
- `intrinsics_da3.txt` 書き出し (Phase 3 で活コード化済み、DA3 推論結果の出力経路として継続)

#### 9.1.4 include 削除

```cpp
#include "obj_exporter.hpp"   // 削除
```

これにより `common/src` 側の `obj_exporter` への依存も切れる
(sam2_da3_lite の include path から `common/src` 参照を CMakeLists で外せるか確認)。

#### 9.1.5 リサイズ挙動

§2.3 で確定したように、**1920×1080 cap 維持** (Step 9 現行挙動)。
`intrinsicsSourceName` を見ていた条件分岐は単純化:
- 入力 ≤ 1920×1080 → そのまま処理
- 入力 > 1920×1080 → アスペクト比保持で 1920×1080 までダウンスケール

### 9.2 `common/src/DepthRunner.h` の削除対象

#### 9.2.1 `DepthRunnerConfig` から削除

```cpp
bool  useCustomIntrinsics;
float fx, fy, cx, cy;
float k1, k2, k3, k4, p1, p2;
std::string intrinsicsSourceName;
```

#### 9.2.2 `buildCmd()` から削除

- `--kinect-intrinsics` / `--kinect-distortion` の append
- `--intrinsics-source` の append

(該当行: line 325-358, 362-364 付近)

### 9.3 ★ 外部 OBJ 投入経路温存ガードレール

**以下は絶対に削除/変更しない**:

#### 9.3.1 REG 側の OBJ 読み込み経路
- `mCutMesh::loadMeshFromFile` の呼び出し箇所
- `OBJTargetExtraction.h::extractTargetFromOBJ()`
- `OBJTargetExtraction.h::loadCameraIntrinsics()`
- `OBJTargetExtraction.h::loadCameraIntrinsicsAny()`

#### 9.3.2 Phase 3 で実装した経路 B (bin なし → 既存 OBJ 使用)
- §7.4 で実装した `if (std::filesystem::exists(binPath)) ... else if ... existingObjPath` の分岐
- 経路 B のコードパス
- 経路 C のエラー処理

#### 9.3.3 Phase 4 で実装したカノニカル名 + レガシー fallback
- DEFORM 側の 4 箇所、特にレガシー `_k4a` fallback は **将来も残す** (運用が安定するまで)

### 9.4 削除漏れ確認 grep

Phase 5 完了前に以下を実行:

```bash
# K の round-trip 経路が完全に消えているか
grep -rn "kinectFx\|kinectFy\|kinectK1\|intrinsicsSourceName" sam2_da3_lite/
# -> 結果ゼロを期待

# REG が sam2 に K を渡していないか
grep -rn "useCustomIntrinsics\|--kinect-intrinsics\|--intrinsics-source" common/src/DepthRunner.h
# -> 結果ゼロを期待

# OBJ 読み込み経路は残っているか
grep -rn "extractTargetFromOBJ\|loadMeshFromFile" registration/ common/
# -> 複数ヒット (経路温存の確認)

# 外部 OBJ 投入の経路 B コードが残っているか
grep -rn "depth_metric.bin\|existingObjPath" registration/
# -> 経路 A/B 分岐のコードがヒット
```

### 9.5 検証

```bash
# 1. ビルド
cmake --build build -j$(nproc)

# 2. sam2_da3_lite 単体実行 (K 関連 CLI なし)
./build/bin/sam2_da3_lite test.jpg --output /tmp/sam2out \
    --depth-model models/depth_anything_v3_small.onnx \
    --sam-encoder models/sam2_hiera_tiny.encoder.onnx \
    --sam-decoder models/sam2_hiera_tiny.decoder.onnx \
    --point 640,360

# 出力されるべき:
ls /tmp/sam2out/
# - original.jpg
# - segmentation_mask.png, segmentation_overlay.jpg
# - instrument_segmentation_mask.png, instrument_segmentation_overlay.jpg
# - depth_full.png, depth_masked.png, depth_masked_renorm.png (+colored)
# - depth_metric.bin
# - intrinsics_da3.txt

# 出力されないべき (Phase 5 で削除):
ls /tmp/sam2out/pc_metric_pinhole*.obj 2>/dev/null     # -> なし
ls /tmp/sam2out/intrinsics_k4a.txt 2>/dev/null         # -> なし
ls /tmp/sam2out/intrinsics_custom.txt 2>/dev/null      # -> なし
ls /tmp/sam2out/texture.png 2>/dev/null                # -> なし

# 3. REG 経由のフルワークフロー
./build/bin/lsn_registration
# Run Depth → 全カノニカルファイルが揃う

# 4. 外部 OBJ 投入経路 (経路 B) のフル検証
# - depth_metric.bin を削除
# - pc_metric_pinhole_masked.obj だけは置いておく
# - Run Depth (sam2 経由) でも経路 B が動くこと
# - DEFORM が問題なく動くこと

# 5. main.cpp の最終行数 (1054 → 推定 500-600 行台)
wc -l sam2_da3_lite/src/main.cpp
```

**Phase 5 完了条件:**
- sam2_da3_lite が OBJ も texture も round-trip K も書かない
- DA3 推定 K (`intrinsics_da3.txt`) と `depth_metric.bin` は出力する
- REG 経由のフルワークフローで全カノニカルファイルが揃う
- 経路 B (bin なし → 既存 OBJ 使用) が動作する
- main.cpp が ~500-600 行台

---

## 10. Phase 6: 後始末

**コミットメッセージ:** `[obj-migration p6] docs + final cleanup`

### 10.1 ドキュメント更新

- `PROJECT_STRUCTURE.md` §5 「実行時のヘルパー連携」を更新:
  - sam2_da3_lite の出力リストを修正
  - REG 側で `common/src/DepthToObjExport.h` 経由で書く旨を追記
  - 「外部 OBJ 投入経路」の存在を明記
- `docs/KEY_REFERENCE.md` 該当箇所更新
- `OBJTargetExtraction.h:26` のコメント `pc_metric_pinhole_masked_k4a.obj` を
  `pc_metric_pinhole_masked.obj` に修正
- `REFACTOR_sam2da_obj_STATUS.md` を完了状態に更新

### 10.2 不要なログメッセージ整理

- sam2_da の Done メッセージ (main.cpp 末尾) で OBJ/intrinsics/texture に言及する行を削除
- 「外部 OBJ 投入経路」を REG 側ログで明示

### 10.3 (Optional) `depth_to_obj_tool` 実装

§6.6 で記録しておいたタスク。余力で実装。

### 10.4 クリーンな depth_output/ からの全ワークフロー検証

```bash
# バックアップ
cp -r depth_output depth_output.bak

# クリーン
rm -rf depth_output/*

# REG 起動して Run Depth
./build/bin/lsn_registration

# 期待される出力:
ls depth_output/
# original.jpg
# segmentation_mask.png, segmentation_overlay.jpg
# instrument_segmentation_mask.png, instrument_segmentation_overlay.jpg
# depth_full.png, depth_full_colored.png
# depth_masked.png, depth_masked_colored.png
# depth_masked_renorm.png, depth_masked_renorm_colored.png
# depth_metric.bin
# intrinsics_da3.txt
# texture.png
# pc_metric_pinhole_masked.obj  + .mtl
# pc_metric_pinhole_full_light.obj  + .mtl
# intrinsics.txt
# pc_metric_pinhole_masked_<tag>.obj  + .mtl
# pc_metric_pinhole_full_<tag>_light.obj  + .mtl
# pc_metric_pinhole_masked_<tag>_noskirt.obj  + .mtl
# pc_metric_pinhole_full_<tag>_light_noskirt.obj  + .mtl
# intrinsics_<tag>.txt

# DEFORM 起動
./build/bin/lsn_deform
```

### 10.5 外部 OBJ 投入経路の最終検証

```bash
# クリーン
rm -rf depth_output/*

# 別の depth ソースで作った OBJ を手動配置 (将来のユースケースをシミュレート)
cp /path/to/external_kinect_obj/pc_metric_pinhole_masked.obj depth_output/
cp /path/to/external_kinect_obj/pc_metric_pinhole_full_light.obj depth_output/
cp /path/to/external_kinect_obj/texture.png depth_output/

# REG 起動 (Run Depth は実行せず、既存 OBJ をロードするモード)
./build/bin/lsn_registration
# intrinsics.txt が REG の現在の K から書き出されること

# DEFORM が動作すること
./build/bin/lsn_deform
```

---

## 11. 検証戦略の総括

### 11.1 各 Phase の独立性

各 Phase は **単独で revert 可能**:
- Phase 1: 死コード削除のみ。挙動変化なし
- Phase 2: ファイル移動のみ。挙動変化なし
- Phase 3: 追加のみ (sam2 の OBJ 書き出しは並行存置)
- Phase 4: DEFORM の読み先変更 (fallback あり)
- Phase 5: sam2 の OBJ 削除 (外部 OBJ 経路は温存)
- Phase 6: ドキュメント整備

### 11.2 連続テストポイント

各 Phase 完了時:
1. **ビルド:** 3 ターゲット全部
2. **sam2_da 単体:** Phase ごとに出力ファイル一覧を確認
3. **REG Run Depth:** UI から実行できる
4. **DEFORM:** `--dry-run` と本番起動が両方通る
5. **OBJ 内容:** Phase 1, 2 では完全一致、Phase 3 以降は浮動小数点誤差レベルの一致
6. **★ 外部 OBJ 投入:** Phase 5 完了後に必ず経路 B を検証

### 11.3 回帰テスト用データ

Phase 0 (着手前) に:
```bash
cp -r depth_output golden_baseline_pre_refactor/
```

Phase 1, 2 完了時に同じ画像で Run Depth → `golden_baseline_pre_refactor/` と比較。

---

## 12. Out of Scope (このリファクタで触らない)

- `registration_model/reg_*.obj` 系 (REG → DEFORM 出力、対象外)
- AutoDeform の `gLiverStaticMesh` ロード
- DA3 / SAM2 推論ロジック (`depth_anything_v3.cpp`, `sam2_segmentor.cpp`)
- Vignette 検出
- `mCutMesh::loadMeshFromFile` の実装 (経路として温存、実装は触らない)
- ImGui UI レイアウト
- CMA-ES 関連
- `calibration_tool/` (別ヘルパー)
- **`IDepthSource` 抽象インタフェース** (将来の別タスク)

---

## 13. リスクと対処

### 13.1 リスク: depth_metric.bin の既存実装と現状の depth 形式の整合

**症状:** Phase 3 で `depth_metric.bin` を再有効化したとき、`depthRaw` 変数が main.cpp 内で
本当に float32 メートル単位で保持されているか確認が必要。

**対処:** Phase 3 着手前に main.cpp の depth pipeline を読み、`depthRaw` の単位と
スケールを確認。`opts.metricScale` がかかっている可能性に注意。

### 13.2 リスク: テクスチャ PNG の bit-identical

**症状:** sam2 が書いていた `texture.png` を REG が書くと、PNG エンコーダの違いで
bit-identical にならない。

**対処:** DEFORM が表示にしか使っていないなら問題なし。decoded RGB レベルで一致すれば OK。
他の読み手があるか grep で確認:
```bash
grep -rn "texture.png" registration/ deform/ common/
```

### 13.3 リスク: 1920×1080 cap の挙動変化

**症状:** sam2 のリサイズロジック簡略化で意図せず挙動変化。

**対処:** Phase 5 着手前に現状の Step 9 リサイズ仕様を main.cpp から正確に読み取り、
同じ挙動になるコードに書き換える。テスト画像で diff 確認。

### 13.4 リスク: `_k4a` レガシー名の他出現箇所

**症状:** Phase 4 で grep し損ねた `_k4a` 残存。

**対処:**
```bash
grep -rn "_k4a\|intrinsics_k4a\|pc_metric_pinhole.*k4a" deform/ common/ registration/
```
Phase 4 完了時に必ず実行。

### 13.5 リスク: 外部 OBJ 投入経路の動作確認漏れ

**症状:** Phase 5 で sam2 削減後、bin なしの場合の動作が壊れている。

**対処:** §10.5 の外部 OBJ 投入検証を Phase 5 完了条件に含める。
省略しない。

---

## 14. 進捗チェックリスト

- [ ] Phase 1: 死コード掃除 (867-903 温存・depth_metric.bin 温存)
- [ ] Phase 2: obj_exporter + image_utils を common/src へ移動
- [ ] (task) depth_to_obj_tool 追加 (低優先・Phase 2 後)
- [ ] 🛑 停止 → GPU で Run Depth 検証 (ユーザー)
- [ ] Phase 3:
  - [ ] depth_metric.bin 再有効化 (16B ヘッダ)
  - [ ] intrinsics_da3.txt 出力 (7 行、distortion なし)
  - [ ] DepthToObjExport.h 実装
  - [ ] REG Run Depth 後フック (経路 A/B/C)
  - [ ] intrinsicsSourceToTag() ヘルパー追加
  - [ ] 867-903 から intrinsics 読み出し部分切り出し
- [ ] Phase 4: DEFORM カノニカル名 + _k4a fallback (4 箇所)
- [ ] Phase 5:
  - [ ] sam2 から OBJ/round-trip K/texture 削除
  - [ ] 1920×1080 cap 簡略化
  - [ ] DA3 推定出力 (intrinsics_da3.txt) は KEEP
  - [ ] ★ 外部 OBJ 投入経路温存ガードレール確認
- [ ] Phase 6:
  - [ ] PROJECT_STRUCTURE.md 等更新
  - [ ] クリーン再生成テスト
  - [ ] 外部 OBJ 投入経路最終検証

---

## 15. 完了条件 (Definition of Done)

1. ✅ `sam2_da3_lite/src/main.cpp` が 500〜600 行台
2. ✅ `sam2_da3_lite/src/obj_exporter.*` と `ply_exporter.*` が存在しない
3. ✅ `sam2_da3_lite/include/image_utils.hpp` が `common/src/` に存在する
4. ✅ `common/src/obj_exporter.*`, `common/src/image_utils.hpp`, `common/src/DepthToObjExport.{h,cpp}` が存在する
5. ✅ sam2 CLI に K 関連フラグがない (`--kinect-*`, `--intrinsics-source`, `--zmode` 等)
6. ✅ sam2 が `depth_metric.bin` (16B ヘッダ) と `intrinsics_da3.txt` (7 行) を出力する
7. ✅ REG の Run Depth 後にカノニカル `pc_metric_pinhole_masked.obj`, `pc_metric_pinhole_full_light.obj`,
     `intrinsics.txt`, `texture.png` が生成される
8. ✅ タグ付きデバッグコピー (`_<tag>` 入り) も生成される
9. ✅ DEFORM がカノニカル名を最優先で読み、`_k4a` 名にもフォールバックする
10. ✅ ★ `depth_metric.bin` がなくても、既存 OBJ ファイルさえあれば REG/DEFORM が動く
11. ✅ ★ bin も OBJ もないとき、明確なエラーが出る
12. ✅ ★ Phase 5 完了後、`mCutMesh::loadMeshFromFile` + `extractTargetFromOBJ` 経路が残っている
13. ✅ クリーンな `depth_output/` から REG Run Depth → DEFORM 起動のフルワークフローが通る
14. ✅ `PROJECT_STRUCTURE.md` が新状態を反映している
15. ✅ 全 3 ビルドターゲットがビルド警告なしで通る
16. ✅ 全 intrinsics テスト (`test_intrinsics_step1.sh`, `test_intrinsics_presets.sh`, `test_intrinsics_step4.sh`) pass

---

## 16. 重要原則のサマリー (1 ページで)

| 原則 | 反例 (やってはいけない) |
|---|---|
| **REG が K を所有** | sam2 が K を round-trip する |
| **sam2 → REG の DA3 推定 K 出力は OK** | sam2 が intrinsics_custom.txt を書く |
| **カノニカル名は DEFORM の正規入力** | DEFORM が `_k4a` 固定で読む |
| **タグ付き名はデバッグ専用** | DEFORM がタグ付き名を読む |
| **`depth_metric.bin` は float metric 経路の正規データ** | PNG から float を復元する |
| **★ 外部 OBJ 投入経路は永続的に温存** | `mCutMesh::loadMeshFromFile` 経路を削除する |
| **★ bin なし → 既存 OBJ 使用 → エラー の 3 段階分岐** | bin がないと無条件にエラー |
| **挙動変化は Phase 3 から、Phase 1+2 はゼロ変化** | Phase 1 で挙動が変わる |
| **Phase 5 後も `_k4a` fallback は当面残す** | 即座に削除して移行期データを壊す |

---

## 17. 用語集

- **カノニカル名**: タグサフィックスのないファイル名 (例: `pc_metric_pinhole_masked.obj`)。
  DEFORM や他コンポーネントが読む正規ファイル。
- **タグ付き名**: `_<tag>` サフィックス付き (例: `pc_metric_pinhole_masked_custom.obj`)。
  デバッグ・cross-K 比較用。
- **`<tag>`**: `intrinsicsSourceToTag()` の戻り値。`custom / calib / k4a (preset key) / da3` のいずれか。
- **K**: カメラ内部パラメータ (fx, fy, cx, cy + 場合により distortion)。
- **round-trip**: REG が値を渡し、sam2 が同じ値を書き戻す経路。今回のリファクタで撲滅する対象。
- **DA3 推定 K**: DA3 ONNX モデルが画像から推定したピンホール K (distortion なし)。
  これは推論結果なので sam2 → REG 方向に流れることが「round-trip ではない」。
- **外部 OBJ 投入経路**: `depth_metric.bin` を使わず、ユーザーが手で `pc_metric_pinhole_masked.obj` を
  `depth_output/` に置く運用。将来 DA3 以外の depth ソース対応のため温存する。
- **経路 A/B/C**: REG の Run Depth 後フックの 3 分岐
  - A: bin あり → REG が OBJ 生成
  - B: bin なし、OBJ あり → 既存 OBJ 使用、intrinsics.txt だけ REG が書く
  - C: bin も OBJ もない → エラー

---

## 付録 A: 主要ファイル参照表

### Phase 1〜2 で触る
- `sam2_da3_lite/src/main.cpp` (1054 行 → 800 行台 → ~600 行台 @ Phase 5)
- `sam2_da3_lite/src/ply_exporter.cpp` (削除)
- `sam2_da3_lite/include/ply_exporter.hpp` (削除)
- `sam2_da3_lite/src/obj_exporter.cpp` → `common/src/obj_exporter.cpp` (Phase 2 で移動)
- `sam2_da3_lite/include/obj_exporter.hpp` → `common/src/obj_exporter.hpp`
- `sam2_da3_lite/include/image_utils.hpp` → `common/src/image_utils.hpp`
- `sam2_da3_lite/CMakeLists.txt`
- ルート `CMakeLists.txt` (確認のみ、変更不要のはず)

### Phase 3 で触る
- `sam2_da3_lite/src/main.cpp` (depth_metric.bin と intrinsics_da3.txt の再有効化)
- `common/src/DepthToObjExport.h` (新規)
- `common/src/DepthToObjExport.cpp` (新規)
- `common/src/IntrinsicsSource.h` (intrinsicsSourceToTag 追加)
- `registration/main.cpp` または `registration/src/RegistrationActions.h` (Run Depth フック)

### Phase 4 で触る
- `deform/src/DeformPipeline.h` (4 箇所)

### Phase 5 で触る
- `sam2_da3_lite/src/main.cpp` (大幅削減)
- `common/src/DepthRunner.h` (CLI 構築から OBJ 関連削除)

### Phase 6 で触る
- `PROJECT_STRUCTURE.md`
- `docs/KEY_REFERENCE.md`
- `registration/src/OBJTargetExtraction.h` (コメント修正)
- `REFACTOR_sam2da_obj_STATUS.md` (完了状態へ)

---

## 付録 B: コマンド早見表

```bash
# 全ビルド
cmake --build build -j$(nproc)

# 単体ターゲット
cmake --build build --target sam2_da3_lite -j$(nproc)
cmake --build build --target lsn_registration -j$(nproc)
cmake --build build --target lsn_deform -j$(nproc)

# Intrinsics テスト 3 種
bash test/test_intrinsics_step1.sh
bash test/test_intrinsics_presets.sh
bash test/test_intrinsics_step4.sh

# sam2_da3_lite 単体実行
./build/bin/sam2_da3_lite test.jpg \
    --depth-model models/depth_anything_v3_small.onnx \
    --sam-encoder models/sam2_hiera_tiny.encoder.onnx \
    --sam-decoder models/sam2_hiera_tiny.decoder.onnx \
    --output /tmp/sam2out \
    --point 640,360

# DEFORM dry-run
./build/bin/lsn_deform --dry-run

# 出力一覧確認
ls -la depth_output/

# OBJ 頂点数確認
grep -c "^v " depth_output/pc_metric_pinhole_masked.obj

# レガシー名 grep (Phase 4 完了時)
grep -rn "_k4a\|intrinsics_k4a\|pc_metric_pinhole.*k4a" deform/ common/ registration/

# K round-trip 残存 grep (Phase 5 完了時)
grep -rn "kinectFx\|kinectFy\|kinectK1\|intrinsicsSourceName\|--kinect-intrinsics" \
    sam2_da3_lite/ common/src/DepthRunner.h

# 外部 OBJ 投入経路確認 (Phase 5 完了時)
grep -rn "extractTargetFromOBJ\|loadMeshFromFile" registration/ common/
# -> 複数ヒットすること

# depth_metric.bin ヘッダ確認 (Phase 3 完了時)
head -c 16 depth_output/depth_metric.bin | xxd
# magic 4 バイトが "DEPT" であること
```

---

## 付録 C: depth_metric.bin ヘッダ仕様

```
オフセット 0-3   uint32   magic     = 0x44455054 ("DEPT" little-endian)
オフセット 4-7   uint32   width     = W
オフセット 8-11  uint32   height    = H
オフセット 12-15 uint32   reserved  = 0
オフセット 16+   float32 × W × H   depth (row-major, メートル単位)
```

合計サイズ: 16 + W * H * 4 バイト
例: 1920×1080 の場合、16 + 1920 * 1080 * 4 = 8,294,416 バイト (約 7.9 MB)

書き込みコード (sam2 側):
```cpp
uint32_t magic = 0x44455054, w = W, h = H, reserved = 0;
ofs.write((const char*)&magic, 4);
ofs.write((const char*)&w, 4);
ofs.write((const char*)&h, 4);
ofs.write((const char*)&reserved, 4);
ofs.write((const char*)depthRaw.data(), depthRaw.size() * sizeof(float));
```

読み込みコード (REG 側):
```cpp
uint32_t magic, w, h, reserved;
ifs.read((char*)&magic, 4);
ifs.read((char*)&w, 4);
ifs.read((char*)&h, 4);
ifs.read((char*)&reserved, 4);
if (magic != 0x44455054) { /* エラー */ }
std::vector<float> depth(w * h);
ifs.read((char*)depth.data(), w * h * sizeof(float));
```
