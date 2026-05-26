# test/ — 自動検証スクリプト

| スクリプト | 対象 | 概要 |
|---|---|---|
| `test_intrinsics_step1.sh` | Step 1 | `lsn_deform --dry-run` で DeformPipeline の K 読込/UV を検証 |
| `test_intrinsics_presets.sh` | Step 5 | `IntrinsicsPresets.h` の lookupPreset / 最小化後の registry |
| `test_intrinsics_step4.sh` | Step 4 / step7-cleanup | `lsn_registration --check-intrinsics` で autoSelect 優先順位を検証 |

実行はいずれも `bash test/<script>` 。全テスト pass で exit 0、失敗で exit 1。
intrinsics 系ファイル（`intrinsics_k4a.txt` / `intrinsics_custom.txt` /
`intrinsics_calib_last.txt` / `intrinsics_calib.txt`）は各スクリプトが
バックアップ→復元するので破壊しない。

## 検証用 CLI フラグ

- `lsn_deform --dry-run` … GUI なしで Step 1 のロジックのみ実行しログ出力（下記）。
- `lsn_registration --check-intrinsics` … GUI なしで Step 4 の
  `autoSelectIntrinsicsSource()` + `loadIntrinsicsFromCurrentSource()` を実行し、
  選択された source（legacy int 0=DA3/1=Preset/2=Custom/3=Calib）と
  ロードされた K を出力して即 exit 0。
  優先順位: Custom > Calib > DA3(`intrinsics_da3_last.txt`) > Preset(default fallback)。

## test_intrinsics_step1.sh

INTRINSICS **Step 1**（`deform/src/DeformPipeline.h` から K4A 720p ハードコードを除去し、
カメラ内部パラメータ K を `depth_output/intrinsics_k4a.txt` から読むようにした変更）が
正しく機能しているかを **GUI / OpenGL なし**で検証する。

### 実行方法

```bash
bash test/test_intrinsics_step1.sh
```

- 全テスト成功で `All tests passed` を表示し **exit 0**。
- いずれか失敗で失敗テスト名と理由を表示し **exit 1**。
- `depth_output/intrinsics_k4a.txt` は冒頭でバックアップし、終了時（成否問わず）に
  必ず復元する（`trap`）。**既存ファイルを壊さない。**

### `lsn_deform --dry-run` について

検証の中核。`deform/main.cpp` が `--dry-run` 引数を受けると、
**OpenGL ウィンドウを開かず**に Step 1 のロジック（`DeformPipeline::dryRunStep1()`）だけを
CPU 上で実行し、以下をログ出力して **即 exit 0** する：

```
[DryRun] === Step 1 intrinsics check (no GL) ===
[Deform] intrinsics: 1280x720  fx=918.234 fy=918.112 cx=640.152 cy=366.447   (または fallback ログ)
[DryRun] deformK fx=918.234 fy=918.112 cx=640.152 cy=366.447 width=1280 height=720
[DryRun] board verts=9216 UV u=[...] v=[...]
```

- `dryRunStep1()` は `loadReferenceMeshes()` 全体ではなく、その中で使う共有ヘルパ
  `loadDeformIntrinsics()`（K の読込 + fallback）と `computeBoardUV()`（K → 正規化UV）を
  直接呼ぶ。これにより `registration_model/reg_*.obj`（DEFORM の入力）が無い環境でも、
  また GL/ディスプレイが無い CI 環境でも、Step 1 の実コードパスを検証できる。
- `--dry-run` は `lsn_deform` 専用。`lsn_registration` には一切影響しない。

### 各テストが検証する内容

| Test | 内容 | 合格条件 |
|---|---|---|
| **A** | `DeformPipeline.h` のハードコード残渣 grep | `918.234` 等の 720p 定数が**無い** |
| **B** | `lsn_deform` のビルド | `cmake --build` が exit 0、バイナリ生成 |
| **C** | 720p の K を書いて `--dry-run` | ログに `deformK fx=918` |
| **D** | 1080p の K を書いて `--dry-run` | ログに `deformK fx=1377` |
| **E** | K を変えると board UV が変わること | FOV の異なる K で UV 範囲が変化 |
| **F** | K が無効/欠損のとき | `fallback` / `default` ログが出る |

#### Test E の注意（重要）

board UV は `u = (fx·x/z + cx)/W`, `v = (fy·y/z + cy)/H` と**解像度で正規化**される。
K4A の 720p と 1080p は**同一カメラの単純な 1.5 倍スケール**（同一 FOV）なので、
`fx/W` と `cx/W` が不変 → **正規化 UV は両者でほぼ完全に一致する**（＝正しい挙動）。

そのため Test E は「Test C(720p) と Test D(1080p) の UV を比較」では検証できない
（一致してしまう）。代わりに **FOV が実際に異なる K**（`fx=600, cx=300, W=1280`）を使い、
UV 範囲が 720p と明確に変わることで「K が UV 計算に効いている」ことを確認する。
スクリプトは参考として「720p と 1080p の UV が一致＝比例スケールなので正しい」旨も表示する。
