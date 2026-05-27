# 実装計画書: sam2_da3_lite からの OBJ Export 移管とファイル名カノニカル化

> 対象: Claude Code (CLI agent)
> プロジェクト: AAA_LiverSurgeryNaviComb
> 最終更新: 2026-05-27

---

## 0. このドキュメントの読み方

このドキュメントは、`sam2_da3_lite` (外部ヘルパー) に肥大化した OBJ エクスポート責務を `common/` の共有ライブラリへ移し、それを REG (`lsn_registration`) がプロセス内で呼ぶ構成へ移行するための段階的実装計画です。

**作業の進め方:**
- Phase 1 → 6 を順番に実施する
- 各 Phase の末尾に「検証」セクションがあり、ここを通過してから次の Phase に進む
- Phase 単位で git commit する。Phase 内では小ステップごとにビルドが通ることを確認
- 不明点が出たら、`PROJECT_STRUCTURE.md` と `docs/KEY_REFERENCE.md` を最初に参照

**作業対象 (確定済):**
- 削減: `sam2_da3_lite/main.cpp` 1054 行 → 推定 500〜600 行
- 移動: `sam2_da3_lite/src/obj_exporter.{hpp,cpp}` → `common/src/`
- 削除: `sam2_da3_lite/src/ply_exporter.{hpp,cpp}` (全 save が `#if 0`)
- 新規: `common/src/DepthToObjExport.h` (REG から objexp を呼ぶラッパー)
- 改修: `registration/main.cpp` または `RegistrationActions.h` の Run Depth 完了後フック
- 改修: `deform/src/DeformPipeline.h` (4 箇所)
- 改修: `common/src/DepthRunner.h` (CLI 構築から OBJ 関連削除)

---

## 1. 背景

### 1.1 sam2_da3_lite が太りすぎている

`sam2_da3_lite` は本来「DA3 で深度推定」「SAM2 でセグメンテーション」だけを担う外部ヘルパー (`PROJECT_STRUCTURE.md` §5) だが、現状の `main.cpp` 1054 行のうち相当部分が OBJ/PLY エクスポート関連で占められている:

- `Options` 構造体に OBJ 専用フィールド 14 個 (zMode, skirtThreshold, kinectFx/Fy/Cx/Cy, kinectK1..P2, intrinsicsSourceName など)
- 対応する CLI フラグ 13 個 (`--kinect-intrinsics`, `--zmode`, `--skirt`, `--dilate` 等)
- `obj_exporter.cpp` 303 行 (実コード)
- `ply_exporter.cpp` 181 行 (**全 save が `#if 0` で完全死蔵**)
- main.cpp 内に `#if 0` で囲われた Relief / DA3-intrinsics OBJ / HQ confidence ブロック合計 ~200 行
- 905-1031 行: Kinect intrinsics OBJ ブロック (活コード、移管対象)
- 913-955 行: `intrinsics_<tag>.txt` 書き出し (REG が渡した値を round-trip するだけの無計算ファイル)

### 1.2 `_k4a` タグが負債

旧構成では sam2_da が単一 K (Azure Kinect 720p hardcode) で動いていたため、出力に `_k4a` サフィックスが付いていた。`--intrinsics-source <name>` 導入後はサフィックスが可変化したが、**DEFORM 側は `_k4a` ハードコードのまま** (`DeformPipeline.h:172, 219, 274, 307`)。

現状のワークフローでは Custom (`intrinsics_custom.txt`) が主流で K4A はほぼ使われない。**「ある時点で正解の K は 1 つ」** という事実に対し、ファイル名でタグを区別する設計が機能していない。

### 1.3 設計判断: Option C (共有ライブラリ化)

代替案として検討した 3 つ:
- **A.** REG 内で in-memory unproject (OBJ 中間ファイルなし)
- **B.** sam2_da とは別の中間ツール `depth_to_obj` を挟む
- **C.** `obj_exporter` を `common/` に移し、REG がプロセス内で呼ぶ ← **採用**

**C を採用した理由:**
- デバッグ用に OBJ をディスクに残す要件がある (skirt あり/なし変種等の目視検証)
- DEFORM は OBJ ファイル経由のままで動作する (REG → DEFORM の実行順なので、書き手が REG に変わってもタイミング問題なし)
- 他の depth source (Kinect SDK 生 depth, RealSense 等) からも将来同じライブラリを呼べる
- subprocess hop が増えない (sam2_da は元から 1 プロセスなので)

---

## 2. アーキテクチャ Before / After

### 2.1 Before (現状)

```
[REG] DepthRunner.run() ─CLI args─→ [sam2_da3_lite]
                                       ├─ DA3 推論 → depth_*.png
                                       ├─ SAM2 推論 → segmentation_mask.png
                                       ├─ Vignette 検出
                                       └─ OBJ Export ───┐
                                                        │ K は CLI から round-trip
                                          intrinsics_<tag>.txt
                                          pc_metric_pinhole_full_<tag>_light.obj
                                          pc_metric_pinhole_masked_<tag>.obj
                                          pc_metric_pinhole_*_<tag>_noskirt.obj
                                          texture.png
                                                  │
[REG] mCutMesh::loadMeshFromFile ←────┘
       extractTargetFromOBJ
[DEFORM] mCutMesh::loadMeshFromFile ←─ (_k4a ハードコード)
```

問題点:
- K を CLI で渡して、何の計算もせずファイルに書いて、また読む round-trip
- sam2_da の関心が 4 つに膨張 (推論 + マスク + Vignette + ジオメトリ生成)
- DEFORM が読むファイル名が `_k4a` 固定で実態と乖離

### 2.2 After (目標)

```
[REG] DepthRunner.run() ─CLI args─→ [sam2_da3_lite]
                                       ├─ DA3 推論 → depth_*.png
                                       ├─ SAM2 推論 → segmentation_mask.png
                                       └─ Vignette 検出 → instrument_segmentation_mask.png
                                                  │
[REG] depth_*.png + mask.png をロード ←─┘
       ├─ 自分が持っている K で
       └─ common::exportDepthArtifacts() を呼ぶ
                ├─ カノニカル: pc_metric_pinhole_masked.obj
                │              pc_metric_pinhole_full_light.obj
                │              intrinsics.txt
                │              texture.png
                └─ デバッグ:   pc_metric_pinhole_masked_<tag>.obj
                               pc_metric_pinhole_*_<tag>_noskirt.obj
                               intrinsics_<tag>.txt
[DEFORM] mCutMesh::loadMeshFromFile ←─ カノニカル名のみ
```

新ルール:
- **K は REG 側だけが知っている**。sam2_da は K を一切受け取らない
- **カノニカル名 = タグなし** が DEFORM/他コンポーネントの読む正規ファイル
- **タグ付きコピー = `<tag>` 入り** はデバッグ・cross-K 比較用 (REG が同時に書く)
- どの K で焼いたかのメタデータは `intrinsics.txt` 内の `name` フィールドに保持される (情報損失なし)

---

## 3. 新しい出力ファイル命名規則

| ファイル | 用途 | DEFORM が読む |
|---|---|---|
| `pc_metric_pinhole_masked.obj` | カノニカル target mesh | ✅ |
| `pc_metric_pinhole_full_light.obj` | カノニカル board mesh (stride=10) | ✅ |
| `intrinsics.txt` | カノニカル K + distortion | ✅ |
| `texture.png` | カノニカル (既にカノニカル) | ✅ |
| `pc_metric_pinhole_masked_<tag>.obj` | デバッグ (cross-K 比較) | ❌ |
| `pc_metric_pinhole_full_<tag>_light.obj` | デバッグ | ❌ |
| `pc_metric_pinhole_masked_<tag>_noskirt.obj` | デバッグ (skirt 影響検証) | ❌ |
| `pc_metric_pinhole_full_<tag>_light_noskirt.obj` | デバッグ | ❌ |
| `intrinsics_<tag>.txt` | デバッグ (log と一緒に残す) | ❌ |

**`<tag>` の決め方:**
- REG の `IntrinsicsSource` enum (`IntrinsicsSource.h`) から派生:
  - `Custom` → `"custom"`
  - `Calib`  → `"calib"`
  - `Preset` → そのプリセットキー (例 `"k4a"`)
  - `Auto`   → `"auto"`
- `CameraIntrinsics::name` フィールドにそのまま格納
- ファイル名の `<tag>` プレースホルダにも同じ文字列を使う

---

## 4. Phase 1: 死コード掃除 (sam2_da3_lite)

**目的:** 後続 Phase の差分を読みやすくするため、明らかに死んでいる `#if 0` コードを先に消す。
**影響範囲:** `sam2_da3_lite/` のみ。REG / DEFORM への影響ゼロ。
**所要: 短時間**

### 4.1 削除対象

#### 4.1.1 `sam2_da3_lite/src/ply_exporter.{hpp,cpp}` ファイルごと

理由: `main.cpp` 内の `ply::` 呼び出しは全て `#if 0` で囲われており実コードゼロ。
具体的箇所:
- `main.cpp:738-822` (Relief PLY ブロック内)
- `main.cpp:830-836` (DA3-intrinsics PLY コメント)
- `main.cpp:957-960` (k4a PLY コメント)

作業:
1. `sam2_da3_lite/src/ply_exporter.cpp` 削除
2. `sam2_da3_lite/src/ply_exporter.hpp` (または `.h`) 削除
3. `main.cpp:4` の `#include "ply_exporter.hpp"` 削除
4. `sam2_da3_lite/CMakeLists.txt` から `ply_exporter.cpp` の sources 行削除

#### 4.1.2 `main.cpp` 内の `#if 0` ブロック削除

| 範囲 (現行行番号) | 内容 |
|---|---|
| 727-827 | Relief OBJ/PLY 出力 (全て `#if 0`) |
| 841-862 | HQ confidence filter 計算 (`#if 0`) |
| 867-903 | DA3-intrinsics OBJ 出力 (`#if 0`) |
| 830-836 内のコメント | DA3-intrinsics PLY (実コードなし) |
| 957-960 内のコメント | k4a PLY (実コードなし) |
| 894-902 内 | HQ OBJ 保存 |

各ブロックを `#if 0` 〜 `#endif` ごと削除。**先頭の `[COMMENTED OUT]` コメント行も併せて削除**。

#### 4.1.3 関連する `Options` フィールドと CLI フラグ削除

`Options` 構造体から:
- `saveRelief` (デフォルト true、Relief ブロックで使用)
- `saveHq` (HQ ブロックで使用)
- `confPercentile` (HQ ブロックで使用)
- `reliefThickness` (Relief ブロックで使用)

CLI から (`printUsage` と `parseArgs` の両方):
- `--no-relief`
- `--no-hq`
- `--conf-percentile`
- `--thickness`

### 4.2 検証

```bash
# 1. sam2_da3_lite が単体でビルドできる
cd build
cmake --build . --target sam2_da3_lite

# 2. 既存の Run Depth が動作する (REG 起動して Depth ボタン押下)
#    出力ファイルが Phase 1 前と byte-identical であることを確認
diff <(ls depth_output/) <(cat known_good_filelist.txt)

# 3. OBJ ファイルの内容比較 (Phase 1 では出力変化ゼロのはず)
diff depth_output/pc_metric_pinhole_masked_k4a.obj backup/pc_metric_pinhole_masked_k4a.obj
```

**Phase 1 完了条件:** sam2_da3_lite の挙動が Phase 1 前と完全同一 (出力ファイル名・内容ともに変化なし)。main.cpp が ~800 行台に縮む。

---

## 5. Phase 2: `obj_exporter` を `common/src/` に移動

**目的:** REG/DEFORM 両方から呼べる位置にライブラリを移す。
**影響範囲:** `sam2_da3_lite` の include path、`common/` のソースリスト、CMakeLists 3 つ。
**所要: 中程度**

### 5.1 作業

#### 5.1.1 ファイル移動

```bash
git mv sam2_da3_lite/src/obj_exporter.hpp common/src/obj_exporter.hpp
git mv sam2_da3_lite/src/obj_exporter.cpp common/src/obj_exporter.cpp
```

または手動コピー後 `git rm` でも可。

#### 5.1.2 依存関係の整理

`obj_exporter.cpp` は `img::Image` 型 (sam2_da3_lite の `image_utils.hpp`) に依存している。これを common 側で完結させる必要がある。

**方針 (推奨):** `img::Image` 相当の最小型を common 側に作るか、もしくは `obj_exporter` が受ける入力を一段抽象化する。

```cpp
// common/src/obj_exporter.hpp (移動後)
namespace objexp {

// 既存の img::Image 依存を除去するため、入力型を構造体で受ける
struct RgbImageView {
    const uint8_t* pixels;   // RGB, row-major, 3 channels interleaved
    int width;
    int height;
    bool empty() const { return pixels == nullptr || width <= 0 || height <= 0; }
    int channels() const { return 3; }
};

bool saveFullMeshObj(
    const std::string& outPath,
    const RgbImageView& rgbImage,
    const std::vector<float>& depthMetric,
    const ObjExportOptions& opt);

bool saveMaskedMeshObj(
    const std::string& outPath,
    const RgbImageView& rgbImage,
    const std::vector<float>& depthMetric,
    const std::vector<uint8_t>& mask,
    const ObjExportOptions& opt);

} // namespace objexp
```

`writeTextureImage()` 内で行っている PNG 書き出しは、`img::saveImage` 依存を切るため:
- 呼び出し側 (REG) で stb_image_write を使って PNG を別途書く責務にする
- もしくは `ObjExportOptions` に `std::function<bool(const std::string&, const RgbImageView&)> textureWriter` を持たせて DI する

**シンプルさを優先するなら、テクスチャ書き出しはエクスポータの責務から外す** のが clean:

```cpp
struct ObjExportOptions {
    // 既存フィールドはそのまま
    // ただし writeTexture, textureFilename, materialName 関連の動作は
    // 「.mtl にテクスチャファイル名を書くだけ。実 PNG は呼び出し側が書く」に変更
};
```

呼び出し側 (Phase 3 で実装) は:
```cpp
// REG 側
stb_image_write_png("texture.png", rgb.data, W, H, 3);
objexp::saveMaskedMeshObj(outPath, rgbView, depth, mask, opt);
```

#### 5.1.3 CMakeLists 修正

1. **`sam2_da3_lite/CMakeLists.txt`**: sources から `src/obj_exporter.cpp` を削除
2. **ルート `CMakeLists.txt`**: `lsn_registration` と `lsn_deform` の sources に `common/src/obj_exporter.cpp` が含まれることを確認 (common/src/*.cpp で glob してるなら自動)
3. include path: `common/src` は既に両ターゲットの include path に入っている (`PROJECT_STRUCTURE.md` §2)

#### 5.1.4 sam2_da3_lite 側の include 削除

`main.cpp:5` の `#include "obj_exporter.hpp"` は Phase 5 で削除するが、**Phase 2 の時点で main.cpp はまだ obj_exporter を使っている**。
そのため:
- sam2_da3_lite の include path に `../common/src` を追加 (CMakeLists 内 `target_include_directories`)
- `#include "obj_exporter.hpp"` を `#include "../../common/src/obj_exporter.hpp"` に書き換える、または include path 経由で解決

これは Phase 5 で sam2_da3_lite が obj_exporter 依存を完全に手放した時点で消える「橋渡し」依存。

### 5.2 検証

```bash
# 1. 全ターゲットがビルドできる
cmake --build build

# 2. sam2_da3_lite の挙動が Phase 2 前と完全同一
#    (obj_exporter の実装は変わっていないので OBJ も byte-identical のはず)
diff depth_output/pc_metric_pinhole_masked_k4a.obj backup/pc_metric_pinhole_masked_k4a.obj

# 3. REG/DEFORM が起動できる (まだ objexp は呼んでいないが、共有ライブラリとしてリンクされている)
./build/bin/lsn_registration
./build/bin/lsn_deform --dry-run
```

**Phase 2 完了条件:** ビルド全通過、sam2_da3_lite の OBJ 出力が変化ゼロ。

---

## 6. Phase 3: REG 側にエクスポート呼び出しを追加

**目的:** REG の Run Depth 完了後に `objexp` を呼んで OBJ + `intrinsics.txt` を書く。これにより **sam2_da が OBJ を書かなくても DEFORM が動く状態を先に作る**。
**影響範囲:** `registration/main.cpp` または `RegistrationActions.h`、新規 `common/src/DepthToObjExport.h`。
**所要: 大**

### 6.1 新規: `common/src/DepthToObjExport.h`

```cpp
#pragma once
// =============================================================================
//  DepthToObjExport.h
//  ---------------------------------------------------------------------------
//  REG 側で Run Depth 後に呼ぶ薄いラッパー。sam2_da3_lite が以前担っていた
//  OBJ + intrinsics.txt の書き出し責務をこちらに移す。
//
//  入力: depth (float, meters, H*W) + RGB texture + mask + 自身が持つ K
//  出力: カノニカル名 OBJ x2 + タグ付きデバッグ OBJ x4 + intrinsics.txt x2
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>
#include "obj_exporter.hpp"
#include "OBJTargetExtraction.h"  // CameraIntrinsics

namespace depthexport {

struct Request {
    // --- 入力データ ---
    const uint8_t* rgbPixels;          // H*W*3 (RGB, row-major)
    int width;
    int height;
    const std::vector<float>* depthMetric;  // size = W*H, meters
    const std::vector<uint8_t>* mask;       // size = W*H, 0/255
    Reg3DCustom::CameraIntrinsics K;        // K.name が <tag> として使われる

    // --- 出力先 ---
    std::string outDir;                // = DEPTH_OUTPUT_PATH

    // --- オプション ---
    float skirtThreshold = 0.05f;
    int   fullMeshStride = 10;         // light variant 用
    bool  writeTaggedCopies = true;    // タグ付きデバッグコピーも書くか
    bool  writeNoSkirtVariants = true; // _noskirt 変種を書くか
    bool  writeTextureImage = true;    // texture.png 書き出し
    int   maskDilate = 0;              // マスク膨張 px (0 = off)
};

struct Result {
    bool ok = false;
    std::string canonicalMaskedObj;    // pc_metric_pinhole_masked.obj のパス
    std::string canonicalFullObj;      // pc_metric_pinhole_full_light.obj のパス
    std::string canonicalIntrinsics;   // intrinsics.txt のパス
    std::vector<std::string> debugCopies;  // タグ付きコピーのパス一覧
};

Result exportDepthArtifacts(const Request& req);

// intrinsics.txt 単体書き出し (.name フィールドから distortion 込み)
bool saveIntrinsicsFile(const std::string& path,
                        const Reg3DCustom::CameraIntrinsics& K,
                        int width, int height);

} // namespace depthexport
```

実装の中で:
1. `texture.png` を書く (`writeTextureImage` true のとき) - stb_image_write 使用
2. `objexp::saveFullMeshObj` をカノニカル名で 1 回 (stride=10、skirt あり)
3. `objexp::saveMaskedMeshObj` をカノニカル名で 1 回 (skirt あり)
4. タグ付きコピー (`writeTaggedCopies` true のとき):
   - `pc_metric_pinhole_full_<tag>_light.obj` (stride=10, skirt あり)
   - `pc_metric_pinhole_masked_<tag>.obj` (skirt あり)
   - `_noskirt` 変種は `writeNoSkirtVariants` true のとき追加
5. `intrinsics.txt` 書き出し (カノニカル)
6. `writeTaggedCopies` true のとき `intrinsics_<tag>.txt` も書く

**`intrinsics.txt` の書式** (`sam2_da3_lite/main.cpp:914-955` の実装を移植):

```
fx <float>
fy <float>
cx <float>
cy <float>
width <int>
height <int>
name <string>            # IntrinsicsSource 由来のタグ
# Brown-Conrady distortion (OpenCV convention)   <- 全てゼロなら省略
k1 <float>
k2 <float>
k3 <float>
k4 <float>
p1 <float>
p2 <float>
```

- `std::setprecision(9)` を使う (single-precision float の round-trip 保証)
- distortion は eps=1e-6 を超える値が 1 つでもあれば全 6 行書く、全部ゼロなら 6 行ごと省略 (既存挙動を維持)

### 6.2 REG main.cpp / RegistrationActions.h での呼び出し

Run Depth の完了ハンドラを特定する:

```bash
grep -n "DepthRunner\|depthRunner.run\|runDepth\|RunDepth" registration/main.cpp registration/src/*.h
```

呼び出しパターンの典型例:
```cpp
// 既存
DepthRunnerResult depthRes = depthRunner.run(imagePath, points);

// 追加 (sam2_da が OBJ を書かなくなった後の補填)
if (depthRes.success) {
    // 1. depth_full.png をロードして float depth に変換
    //    (sam2_da は depth_full.png を 16-bit grayscale で書いている想定)
    std::vector<float> depthMetric = loadDepthAsMetric(depthRes.depthFullPath);

    // 2. mask をロード
    std::vector<uint8_t> mask = loadMask(depthRes.segmentationMaskPath);

    // 3. RGB をロード (original.jpg は sam2_da が書いている)
    int W, H;
    std::vector<uint8_t> rgb = loadRgb(depthRes.originalPath, W, H);

    // 4. REG が持っている K を準備 (g_intrinsicsSource 由来の K)
    Reg3DCustom::CameraIntrinsics K = currentIntrinsics();
    K.name = tagFromIntrinsicsSource(g_intrinsicsSource);  // "custom"/"calib"/...

    // 5. エクスポート呼び出し
    depthexport::Request req;
    req.rgbPixels        = rgb.data();
    req.width            = W;
    req.height           = H;
    req.depthMetric      = &depthMetric;
    req.mask             = &mask;
    req.K                = K;
    req.outDir           = DEPTH_OUTPUT_PATH;
    req.maskDilate       = /* depthRunner.config.maskDilate or 0 */;
    auto exportRes = depthexport::exportDepthArtifacts(req);

    if (!exportRes.ok) {
        std::cerr << "[REG] exportDepthArtifacts failed" << std::endl;
    }
}
```

**注意点:**
- `loadDepthAsMetric()` は sam2_da が depth をどう書いているかに依存する。現状の `depth_full.png` (8bit colored) や `depth_full_colored.png` ではなく、**生の float depth を別途出してもらう必要があるかもしれない**
- もし sam2_da の出力で float depth がディスクに残っていない場合、Phase 3 の前準備として **sam2_da に raw depth 出力 (`depth_raw.exr` または `depth_full_16bit.png`) を追加する小タスク** を入れる必要がある。これは Phase 3 のサブ作業として実施
- `depth_full.png` がもし 16-bit raw depth (mm or 1/4000 m スケール) を保持しているなら、そこから float metric depth に逆変換できる

**`tagFromIntrinsicsSource` ヘルパー** (`IntrinsicsSource.h` に追加):

```cpp
inline std::string intrinsicsSourceToTag(IntrinsicsSource s,
                                         const std::string& presetKey = "k4a") {
    switch (s) {
        case IntrinsicsSource::Custom: return "custom";
        case IntrinsicsSource::Calib:  return "calib";
        case IntrinsicsSource::Preset: return presetKey;  // 例 "k4a"
        case IntrinsicsSource::Auto:   return "auto";
    }
    return "unknown";
}
```

### 6.3 sam2_da 側はまだ OBJ を書き続けている (このフェーズでは)

**重要:** Phase 3 の時点では sam2_da の OBJ 書き出しを止めない。**両方が書く状態にしておく** ことで:
- REG 側の新コードが書く OBJ と、sam2_da が書く OBJ を直接 diff で比較できる
- カノニカル名のファイルは REG が新規に書く、タグ付き名は両方が書く (last-writer-wins だが、REG が後に書くので REG 側が残る)
- 万一 REG 側の実装にバグがあっても sam2_da の出力がフォールバックとして動く

### 6.4 検証

```bash
# 1. Run Depth 実行後、新カノニカルファイルが存在する
ls depth_output/pc_metric_pinhole_masked.obj
ls depth_output/pc_metric_pinhole_full_light.obj
ls depth_output/intrinsics.txt

# 2. タグ付きコピーも存在する
ls depth_output/pc_metric_pinhole_masked_custom.obj   # custom 主流の前提
ls depth_output/intrinsics_custom.txt

# 3. REG が書いたタグ付き OBJ と sam2_da が書いたタグ付き OBJ が一致する
#    (sam2_da は同じファイル名で書き、REG が後で上書きするので、両方の実装が等価か確認)
#    具体的には sam2_da を先に走らせて backup を取り、REG の export を呼んだ結果と比較:
cp depth_output/pc_metric_pinhole_masked_custom.obj /tmp/from_sam2da.obj
# (REG の export を実行)
diff /tmp/from_sam2da.obj depth_output/pc_metric_pinhole_masked_custom.obj
# -> 浮動小数点誤差レベル (printf "%.6f" の桁内) の差のみであることを確認

# 4. intrinsics.txt の中身が期待通り
cat depth_output/intrinsics.txt
# name フィールドが現在の IntrinsicsSource に対応していることを確認
```

**Phase 3 完了条件:** カノニカル名の OBJ + intrinsics.txt が REG 経由で書かれる。タグ付き OBJ が sam2_da 出力と等価。DEFORM はまだ `_k4a` を読んでいるので動作不変。

---

## 7. Phase 4: DEFORM のファイル名カノニカル化

**目的:** DEFORM が読むファイル名を `_k4a` ハードコードからカノニカル名に切り替える。
**影響範囲:** `deform/src/DeformPipeline.h` 4 箇所のみ。
**所要: 小〜中**

### 7.1 作業

`deform/src/DeformPipeline.h` を以下のように修正:

#### 7.1.1 line 170-178 `loadDeformIntrinsics()`

```cpp
// 変更前
inline Reg3DCustom::CameraIntrinsics loadDeformIntrinsics() {
    Reg3DCustom::CameraIntrinsics K;
    if (!Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt", K)
        || !K.valid()) {
        std::cout << "[Deform] intrinsics_k4a.txt missing/invalid -> K4A 720p fallback"
                  << std::endl;
        K = /* K4A 720p fallback */;
    }
    return K;
}

// 変更後
inline Reg3DCustom::CameraIntrinsics loadDeformIntrinsics() {
    Reg3DCustom::CameraIntrinsics K;

    // カノニカルパスを最優先 (REG が Run Depth 完了時に書く)
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics.txt", K)
        && K.valid()) {
        return K;
    }

    // 移行期 fallback: 旧 _k4a タグ付きファイルも見る
    // (REG で Run Depth されていない旧データ向け)
    if (Reg3DCustom::loadCameraIntrinsics(DEPTH_OUTPUT_PATH + "intrinsics_k4a.txt", K)
        && K.valid()) {
        std::cout << "[Deform] (warn) intrinsics.txt not found, falling back to "
                  << "intrinsics_k4a.txt. Re-run Run Depth in REG to update."
                  << std::endl;
        return K;
    }

    std::cout << "[Deform] intrinsics.txt and intrinsics_k4a.txt both missing/invalid"
              << " -> K4A 720p hardcode fallback" << std::endl;
    // K = K4A 720p fallback (既存と同じ);
    return K;
}
```

#### 7.1.2 line 219 (dryRunStep1 内)

```cpp
// 変更前
const std::string p = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_k4a_light.obj";

// 変更後
const std::string p = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_light.obj";
```

同様にカノニカル名がなければ `_k4a_light` にフォールバックする処理を追加してもいい。`std::filesystem::exists` で先にチェックする箇所なので追加は容易:

```cpp
std::string p = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_light.obj";
if (!std::filesystem::exists(p)) {
    std::string legacy = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_k4a_light.obj";
    if (std::filesystem::exists(legacy)) {
        std::cout << "[DryRun] (warn) using legacy " << legacy << std::endl;
        p = legacy;
    }
}
```

#### 7.1.3 line 274 (target mesh)

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

#### 7.1.4 line 307 (board mesh)

```cpp
// 変更前
const std::string objPath = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_k4a_light.obj";

// 変更後 (line 219 と同じパターン)
std::string objPath = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_light.obj";
if (!std::filesystem::exists(objPath)) {
    std::string legacy = DEPTH_OUTPUT_PATH + "pc_metric_pinhole_full_k4a_light.obj";
    if (std::filesystem::exists(legacy)) {
        std::cout << "[Deform] (warn) using legacy " << legacy << std::endl;
        objPath = legacy;
    }
}
```

### 7.2 検証

```bash
# 1. 新カノニカル OBJ が存在する状態で DEFORM 起動
ls depth_output/pc_metric_pinhole_masked.obj   # 存在
ls depth_output/intrinsics.txt                  # 存在
./build/bin/lsn_deform --dry-run
# -> "[DryRun] deformK fx=... " が正しい K を出すこと
# -> board OBJ がロードされ verts > 0 であること

# 2. カノニカル OBJ を一時的にリネームしてフォールバックが動くことを確認
mv depth_output/pc_metric_pinhole_masked.obj /tmp/
mv depth_output/intrinsics.txt /tmp/
./build/bin/lsn_deform --dry-run
# -> "(warn) using legacy intrinsics_k4a.txt" のログが出ること
# -> それでも動作すること
mv /tmp/pc_metric_pinhole_masked.obj depth_output/
mv /tmp/intrinsics.txt depth_output/

# 3. フル DEFORM ワークフロー (REG で reg_liver.obj を作って DEFORM に渡す)
#    が問題なく動くこと
```

**Phase 4 完了条件:** DEFORM がカノニカル名を優先して読み、レガシー名にもフォールバックする。`--dry-run` および本番ワークフローが両方動作。

---

## 8. Phase 5: sam2_da3_lite から OBJ 関連を削除

**目的:** sam2_da3_lite を本来の単一責任 (DA3 推論 + SAM2 推論 + Vignette 検出 + マスク出力) に戻す。
**影響範囲:** `sam2_da3_lite/main.cpp`、`common/src/DepthRunner.h`。
**所要: 中**

### 8.1 sam2_da3_lite/main.cpp の削除対象

#### 8.1.1 `Options` 構造体から削除するフィールド (現行 line 番号)

```cpp
// 全部削除
bool  saveObjFull;          // line 39
bool  saveObjMasked;        // line 40
bool  hasKinectIntrinsics;  // line 46
float kinectFx, kinectFy, kinectCx, kinectCy;     // line 47-50
float kinectK1, kinectK2, kinectK3, kinectK4;     // line 62
float kinectP1, kinectP2;                          // line 63
std::string intrinsicsSourceName;                  // line 75
int   maskDilate;                                  // line 77
objexp::ZMode zMode;                               // line 45
float skirtThreshold;                              // line 43
```

#### 8.1.2 CLI フラグから削除 (`printUsage` と `parseArgs` 両方)

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

#### 8.1.3 main 関数本体から削除するブロック

| 範囲 | 内容 |
|---|---|
| 414-453 (リサイズロジック内 `skipResize` 分岐) | `intrinsicsSourceName != "k4a"` のスキップ条件を削除し、常に 1280x720 リサイズする (または DA3 ONNX の要求に従う固定挙動) |
| 704-722 | `depthForOutput` 計算 (zMode 依存) → metric (素の depth) のみで OK、変換不要 |
| 830-1034 | `depthResult.hasIntrinsics` ブロック全体 (中の `#if 0` は Phase 1 で消えているので残るのは Kinect intrinsics ブロックのみ。これを全削除) |
| 905-1031 | Kinect intrinsics OBJ ブロック (上記範囲の一部だが念のため明示) |
| 913-955 | `intrinsics_<tag>.txt` 書き出し |

**注意:** Phase 1 で `#if 0` ブロックを消した後の行番号は変わっているので、grep で確認しながら作業。

#### 8.1.4 include 削除

```cpp
#include "obj_exporter.hpp"   // 削除 (line 5)
```

ここまでで `main.cpp` は推定 500-600 行に縮む。

#### 8.1.5 1280x720 リサイズ挙動の決定

旧コードのリサイズ条件:
```cpp
const bool skipResize = (opts.intrinsicsSourceName != "k4a") && opts.hasKinectIntrinsics;
```

これは OBJ 書き出し時の K と画像解像度を一致させるための分岐。OBJ 書き出しを REG 側に移すと、sam2_da 側でこの分岐は不要になる。

**選択肢:**
- **A.** 常に 1280x720 にリサイズする (DA3 ONNX 推論コストを抑える方向)
- **B.** リサイズせず入力画像の native 解像度で DA3 を走らせる (REG の負担増、画質向上の可能性)

**推奨: A** (現状の k4a パスと挙動同一)。REG 側で受け取った depth は 1280x720 だが、REG が持っている K がもし native 解像度向けなら、REG が K をスケールして depth と合わせる責務を取る。または **そもそも sam2_da は入力解像度のまま処理** に倒してもよい。

この判断は **Phase 5 着手前にユーザーに確認** すること。

### 8.2 `common/src/DepthRunner.h` の削除対象

#### 8.2.1 `DepthRunnerConfig` 構造体から削除

```cpp
bool  useCustomIntrinsics;
float fx, fy, cx, cy;
float k1, k2, k3, k4, p1, p2;
std::string intrinsicsSourceName;
```

これらは sam2_da の CLI に渡すための情報だったが、sam2_da がもう K を受け取らないので不要。

**ただし**、REG 側コードがこれらのフィールド経由で `DepthRunner` に K を入れている箇所があれば、その経路は `depthexport::exportDepthArtifacts` の `Request` への直接代入に切り替える必要がある。

#### 8.2.2 `buildCmd()` から削除する行

```cpp
// 325-358: --kinect-intrinsics と --kinect-distortion の append
// 362-364: --intrinsics-source の append
```

### 8.3 検証

```bash
# 1. sam2_da3_lite を単体で実行 (CLI を最小に)
./build/bin/sam2_da3_lite test.jpg --output /tmp/sam2out \
    --depth-model models/depth_anything_v3_small.onnx \
    --sam-encoder models/sam2_hiera_tiny.encoder.onnx \
    --sam-decoder models/sam2_hiera_tiny.decoder.onnx \
    --point 640,360

# 出力されるべき:
ls /tmp/sam2out/
# - original.jpg
# - segmentation_mask.png, segmentation_overlay.jpg
# - instrument_segmentation_mask.png, instrument_segmentation_overlay.jpg (vignette検出のみ)
# - depth_full.png, depth_masked.png, depth_masked_renorm.png (+colored 版)

# 出力されないべき (Phase 5 で削除):
ls /tmp/sam2out/pc_metric_pinhole*.obj 2>/dev/null
# -> "No such file" であること
ls /tmp/sam2out/intrinsics_*.txt 2>/dev/null
# -> "No such file" であること
ls /tmp/sam2out/texture.png 2>/dev/null
# -> "No such file" であること (texture は REG が書く)

# 2. REG 経由の Run Depth フルワークフロー
./build/bin/lsn_registration
# UI で Run Depth → カノニカル OBJ + intrinsics.txt + texture.png が
# depth_output/ に揃うこと

# 3. DEFORM が新フロー後の depth_output/ を読めること
./build/bin/lsn_deform
# Target / Board mesh が正しくロード、AR background が表示されること
```

**Phase 5 完了条件:** sam2_da3_lite が OBJ も intrinsics も texture も書かない。REG 経由のフルワークフローで全カノニカルファイルが揃い、DEFORM が動作する。

---

## 9. Phase 6: 後始末

### 9.1 ドキュメント更新

- `PROJECT_STRUCTURE.md` §5 「実行時のヘルパー連携」を更新:
  - sam2_da3_lite の出力リストから OBJ / intrinsics / texture を削除
  - REG 側で `common/src/DepthToObjExport.h` 経由で書く旨を追記
- `docs/KEY_REFERENCE.md` で OBJ ファイル名に言及している箇所があれば更新
- `OBJTargetExtraction.h:26` のコメント `pc_metric_pinhole_masked_k4a.obj` を `pc_metric_pinhole_masked.obj` に修正

### 9.2 不要なログメッセージの整理

- sam2_da の Done メッセージ (main.cpp:1037-1051) で OBJ / intrinsics / texture に言及している行を削除
- 出力ファイルリストを実態に合わせる

### 9.3 移行期 fallback の扱い

Phase 4 で `intrinsics_k4a.txt` 等への fallback を入れたが、これは「一度 REG で Run Depth すれば不要になる」種類のフォールバック。

**選択肢:**
- 当面残す (運用が安定するまで)
- 警告ログを定期的に出して気づきやすくする
- ある時点で削除し、`intrinsics.txt` 必須にする

Phase 6 の時点では **当面残す** を推奨。将来のクリーンアップ TODO として `docs/TODO.md` 等に記録するに留める。

### 9.4 既存の `depth_output/` を一度クリーンアップして再生成テスト

```bash
# バックアップ
cp -r depth_output depth_output.bak

# クリーン
rm -rf depth_output/*

# REG を起動して Run Depth を実行
./build/bin/lsn_registration
# (UI で 1 枚のテスト画像で Run Depth)

# 期待される出力:
ls depth_output/
# original.jpg
# segmentation_mask.png, segmentation_overlay.jpg
# instrument_segmentation_mask.png, instrument_segmentation_overlay.jpg
# depth_full.png, depth_full_colored.png
# depth_masked.png, depth_masked_colored.png
# depth_masked_renorm.png, depth_masked_renorm_colored.png
# texture.png
# pc_metric_pinhole_masked.obj  + .mtl
# pc_metric_pinhole_full_light.obj  + .mtl
# intrinsics.txt
# pc_metric_pinhole_masked_custom.obj  + .mtl   (タグ付きデバッグコピー)
# pc_metric_pinhole_full_custom_light.obj  + .mtl
# pc_metric_pinhole_masked_custom_noskirt.obj  + .mtl
# pc_metric_pinhole_full_custom_light_noskirt.obj  + .mtl
# intrinsics_custom.txt

# DEFORM を起動 (REG の reg_*.obj 出力後)
./build/bin/lsn_deform
# Target/Board mesh が正常ロードされること
```

---

## 10. 検証戦略の総括

### 10.1 各 Phase の独立性

各 Phase は **単独で revert 可能** に設計されている:
- Phase 1: 死コード削除のみ。挙動変化なし
- Phase 2: ファイル移動のみ。挙動変化なし
- Phase 3: 追加のみ (sam2_da の OBJ 書き出しはまだ残っている)
- Phase 4: DEFORM の読み先変更 (フォールバックあり)
- Phase 5: sam2_da の OBJ 書き出し削除
- Phase 6: ドキュメント整備

万一どこかで重大な問題が出たら、その Phase だけ revert すれば前 Phase の状態に戻る。

### 10.2 連続テストポイント

各 Phase 完了時に以下を確認:

1. **ビルド:** `cmake --build build` が 3 ターゲット (`lsn_registration`, `lsn_deform`, `sam2_da3_lite`) 全部通る
2. **sam2_da 単体:** 上記サンプルコマンドで期待ファイルが出る
3. **REG Run Depth:** UI から実行できる
4. **DEFORM:** `--dry-run` と本番起動が両方通る
5. **OBJ 内容:** Phase 1, 2 では完全一致、Phase 3 以降は浮動小数点誤差レベルの一致

### 10.3 回帰テスト用データの準備

Phase 0 (作業着手前) に:
```bash
# テスト用画像を 1〜2 枚決めて、現状の sam2_da を走らせた結果を保存
cp -r depth_output golden_baseline_pre_refactor/
```

各 Phase 完了時に同じ画像で Run Depth を実行し、`golden_baseline_pre_refactor/` と比較する。

---

## 11. Out of Scope (このリファクタで触らない)

以下は **意図的に範囲外**:

- `registration_model/reg_*.obj` 系 (REG が DEFORM に渡すための出力。今回の対象は `depth_output/` 系のみ)
- AutoDeform の `gLiverStaticMesh` ロード (`DeformPipeline.h:252`)
- DA3 / SAM2 推論ロジックそのもの (`depth_anything_v3.cpp`, `sam2_segmentor.cpp`)
- Vignette 検出 (`VignetteDetection.h`)
- `mCutMesh::loadMeshFromFile` の実装
- ImGui UI レイアウト (`RegistrationImGuiManager.h`)
- CMA-ES 関連 (`CmaesRefineV3R.h` 等)
- `calibration_tool/` (別の外部ヘルパー)

---

## 12. リスクと対処

### 12.1 リスク: float depth のディスク表現が決まっていない

**症状:** sam2_da が float depth を別ファイルで残していない場合、Phase 3 の REG 側で `depth_full.png` (8bit colored) からでは metric depth が復元できない。

**対処:**
- 現状の `depth_full.png` の中身を確認 (8bit colored か 16bit raw か)
- 16bit raw 保存形式なら、その単位 (mm or 4000-unit) を確認し、float 復元式を REG 側 loader に実装
- 8bit colored しか残らないなら、**Phase 3 のサブタスク** として sam2_da に `depth_raw.bin` (float32 H*W binary) または `depth_raw_16bit.png` の出力を追加

### 12.2 リスク: テクスチャ PNG の書き出し責務移動による差分

**症状:** sam2_da が書いていた `texture.png` (= original RGB の PNG コピー) を REG 側が書くと、PNG 圧縮設定の違いで bit-identical にはならない。

**対処:**
- DEFORM が `texture.png` を表示にしか使っていないなら問題なし (見た目同一なら OK)
- 実際に差分テストする場合は decoded RGB 配列で比較する

### 12.3 リスク: 1280x720 リサイズ挙動の変更

**症状:** sam2_da が画像を 1280x720 にリサイズしていた箇所を変更すると、DA3 推論結果の depth が変わる可能性がある。

**対処:**
- Phase 5 の 8.1.5 で詳述。**着手前にユーザーに確認**
- 推奨は「常に 1280x720」(現状の k4a 同一挙動)
- 検証は同じ入力画像で Phase 5 前後の `depth_full.png` を visual diff

### 12.4 リスク: DEFORM の `_k4a` ハードコードを見落とす

**症状:** `_k4a` 文字列が DEFORM の他の場所にも残っていて Phase 4 で見落とすと、深いコードパスで失敗する。

**対処:**
```bash
# 移行期 fallback を入れる前後で必ず実行
grep -rn "_k4a\|intrinsics_k4a\|pc_metric_pinhole.*k4a" deform/ common/
```

### 12.5 リスク: `image_utils` 依存を切るコスト

**症状:** Phase 2 で `obj_exporter` を common に移すとき、`img::Image` 型と `img::saveImage` を取り除く改修が広範になる。

**対処:**
- 最小の改修は「`obj_exporter.cpp` を common に置きつつ、内部で使う `img::Image` 相当の最小 view 構造体を common にも作る」(同等の 2 型が存在することになるが、責務境界としては clean)
- 大きな改修になりそうなら、Phase 2 の中で `obj_exporter` の API を `RgbImageView` ベースに変更する小サブタスクを切る

---

## 13. 進め方の推奨

**Day 1-2: Phase 1, 2**
- 死コード削除は機械的なので diff レビューしやすい
- common 化はビルドシステムの修正が中心

**Day 3-5: Phase 3**
- 最も重い Phase。`DepthToObjExport.h` の API 設計と実装
- float depth の読み込み形式を含む

**Day 6: Phase 4**
- DEFORM 修正 4 箇所 + fallback ロジック

**Day 7: Phase 5**
- sam2_da を削るだけだが、削除漏れと CLI 整合を慎重に

**Day 8: Phase 6 + 総合検証**
- ドキュメント整備
- クリーン状態からの全ワークフロー再生成テスト

---

## 14. 完了条件 (Definition of Done)

1. ✅ `sam2_da3_lite/main.cpp` が 500〜600 行台
2. ✅ `sam2_da3_lite/src/obj_exporter.*` と `ply_exporter.*` が存在しない
3. ✅ `common/src/obj_exporter.*` と `common/src/DepthToObjExport.h` が存在する
4. ✅ `sam2_da3_lite` の CLI に `--kinect-*`, `--intrinsics-source`, `--zmode`, `--skirt`, `--dilate`, `--no-obj-*` フラグがない
5. ✅ REG の Run Depth 後に `depth_output/pc_metric_pinhole_masked.obj`, `pc_metric_pinhole_full_light.obj`, `intrinsics.txt`, `texture.png` が生成される
6. ✅ REG の Run Depth 後にタグ付きデバッグコピー (`_<tag>` 入り) も生成される
7. ✅ DEFORM がカノニカル名を最優先で読み、レガシー `_k4a` 名にもフォールバックする
8. ✅ クリーンな `depth_output/` から REG Run Depth → DEFORM 起動のフルワークフローが通る
9. ✅ `PROJECT_STRUCTURE.md` がリファクタ後の状態を反映している
10. ✅ 全 3 ビルドターゲットがビルド警告なしで通る

---

## 付録 A: 主要ファイルの行番号リファレンス (Phase 0 時点)

### sam2_da3_lite/main.cpp (1054 行)
- L25: `enum class Stage`
- L27-108: `struct Options`
- L110-154: `printUsage`
- L156-310: `parseArgs`
- L328-391: `writeOccluderMaskMerged` (削除しない)
- L393-1054: `int main`
  - L406-454: 画像ロードと 1280x720 リサイズ
  - L462-660: SAM2 + DA3 推論 (削除しない)
  - L704-722: `depthForOutput` 計算 (zMode 依存)
  - L724-828: Relief ブロック (`#if 0`)
  - L830-1034: depthResult.hasIntrinsics ブロック (中の `#if 0` と Kinect 活コード)
  - L905-1031: Kinect intrinsics OBJ ブロック (活、削除対象)

### common/src/DepthRunner.h (414 行)
- L26-104: `struct DepthRunnerConfig`
- L106-110: `struct DepthRunnerPoint`
- L112-146: `struct DepthRunnerResult`
- L148-413: `class DepthRunner`
  - L178-281: `run(...)`
  - L294-407: `buildCmd(...)`
    - L329-358: `--kinect-intrinsics` と `--kinect-distortion` の append (削除対象)
    - L362-364: `--intrinsics-source` の append (削除対象)

### deform/src/DeformPipeline.h (1357 行)
- L170-178: `loadDeformIntrinsics()` (Phase 4)
- L212-230: `dryRunStep1()` (Phase 4, L219)
- L233-265: `loadReferenceMeshes()` 前半 (reg_liver.obj、対象外)
- L266-303: Target mesh ロード (Phase 4, L274)
- L305-323: Board mesh ロード (Phase 4, L307, L308)

### common/src/IntrinsicsSource.h (54 行)
- L22-27: `enum class IntrinsicsSource`
- 推奨: `intrinsicsSourceToTag(IntrinsicsSource, presetKey)` を追加 (Phase 3)

### registration/src/OBJTargetExtraction.h
- L26: コメント (`pc_metric_pinhole_masked_k4a.obj` 言及、Phase 6 で更新)
- L74-89: `struct CameraIntrinsics`
- L172: `loadCameraIntrinsics()` (intrinsics.txt 読み込みヘルパー、既存)
- L409: `extractTargetFromOBJ()` (既存、変更不要)

---

## 付録 B: コマンド早見表

```bash
# 全ビルド
cmake --build build

# 単体ターゲットビルド
cmake --build build --target sam2_da3_lite
cmake --build build --target lsn_registration
cmake --build build --target lsn_deform

# sam2_da3_lite 単体実行
./build/bin/sam2_da3_lite test.jpg \
    --depth-model models/depth_anything_v3_small.onnx \
    --sam-encoder models/sam2_hiera_tiny.encoder.onnx \
    --sam-decoder models/sam2_hiera_tiny.decoder.onnx \
    --output /tmp/sam2out \
    --point 640,360

# DEFORM dry-run
./build/bin/lsn_deform --dry-run

# 出力ファイル一覧
ls -la depth_output/

# 出力 OBJ の頂点数確認
grep -c "^v " depth_output/pc_metric_pinhole_masked.obj

# レガシー名 grep
grep -rn "_k4a\|intrinsics_k4a\|pc_metric_pinhole.*k4a" deform/ common/ registration/
```
