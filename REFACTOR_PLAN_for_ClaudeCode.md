# AAA_LiverSurgeryNaviComb プロジェクト再編成 指示計画書

> **対象実行者**: Claude Code (CLI)
> **対象ディレクトリ**: `/home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviComb`
> **作業方針**: 案A（モノレポ + INTERFACE library 方式）で 3 本立て構成へ移行
> **重要原則**: 各 Phase 終了時に **必ずビルド & スモークテストを通す**。通らなければ即ロールバック。

---

## 0. 用語定義（このドキュメント全体で使用）

| 略号 | 意味 |
|---|---|
| `${ROOT}` | `/home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviComb` |
| `${REG_SRC}` | `${ROOT}/src` （現行 REGISTRATION プロジェクトのソース） |
| `${DEF_SRC}` | `${ROOT}/AAA_LiverSurgeryNaviForDefotm/src` （現行 DEFORM プロジェクトのソース） |
| **A群** | REG と DEFORM で完全同一（md5 一致）の 24 ファイル |
| **B群** | REG と DEFORM で差分のある同名 9 ファイル |
| **C群** | REG 専用 26 ファイル |
| **D群** | DEFORM 専用 3 ファイル |
| **lsn** | LiverSurgeryNavi の略（CMake target prefix） |

---

## 1. 最終ディレクトリ構成（ゴール）

```
AAA_LiverSurgeryNaviComb/
├── common/
│   ├── src/                      ← A群 24 ファイル
│   └── CMakeLists.txt            ← INTERFACE library "lsn_common"
├── registration/
│   ├── src/
│   │   ├── reg_specific/         ← C群 26 ファイル
│   │   └── overrides/            ← B群 9 ファイル（REG 版）
│   ├── main.cpp                  ← 現行 ${REG_SRC}/main.cpp
│   ├── data/                     ← REG 用データ（シンボリックリンクでも可）
│   ├── model/                    ← 〃
│   ├── registration_model/       ← REG の出力先
│   ├── input_image/              ← REG 用入力画像
│   └── CMakeLists.txt            ← target "lsn_registration"
├── deform/
│   ├── src/
│   │   ├── deform_specific/      ← D群 3 ファイル
│   │   └── overrides/            ← B群 9 ファイル（DEFORM 版）
│   ├── main.cpp                  ← 現行 ${DEF_SRC}/main.cpp
│   ├── registration_model/       ← DEFORM の入力（REG の出力 reg_*.obj）
│   ├── data/, model/             ← 必要に応じて
│   └── CMakeLists.txt            ← target "lsn_deform"
├── combined/                     ← 将来用、Phase 9 以降で着手
│   └── README.md                 ← 「Phase 10 以降で着手」と記載のみ
│
├── third_party/                  ← 全プロジェクト共有
├── shaders/                      ← 全プロジェクト共有
├── chessboard/                   ← 全プロジェクト共有
├── calibration_tool/             ← 全プロジェクト共有
├── sam2_da3_lite/             ← 全プロジェクト共有
├── win_deps/                     ← 全プロジェクト共有
├── onnxruntime-linux-x64-1.15.1/ ← 全プロジェクト共有
├── depth_output/                 ← 全プロジェクト共有（出力先）
│
├── reference/                    ← 過去資料（読み取り専用扱い）
│   ├── 元の長いRegistrationImGuiManager.h
│   ├── 元の長いmain.cpp
│   └── README.md                 ← 「過去の統合版スナップショット」と説明
│
├── CMakeLists.txt                ← ルート、option(BUILD_REG/BUILD_DEFORM/BUILD_COMBINED ON)
├── README.md                     ← 新構成の説明を追記
└── REFACTOR_PLAN.md              ← このファイルのコピーを置いておく
```

---

## 2. ファイル分類リスト（完全列挙、移動先付き）

### A群（完全同一・24ファイル）→ `common/src/`

```
AutoHandleController.h
CentVoxTetrahedralizerHybrid.cpp
CentVoxTetrahedralizerHybrid.h
FullSphereCameraWithTarget.h
HandleControllerBase.h
Hash.h
MeshDataTypes.cpp
MeshDataTypes.h
MeshDrawing.h
PathConfig.h
PinholeProjection.h
PlatformCompat.h
RayCast.h
SemiAutoHandleController.h
SemiAutoPickState.h
SequentialDeformController.h
ShaderProgram.cpp
ShaderProgram.h
SimpleCamera.hpp
SoftBody.cpp
SoftBody.h
Sphere.cpp
Sphere.h
TetoMeshData.h
VectorMath.h
simple_multi_obj_processor.h
```
> ⚠️ 上は実際には 26 行ある（cpp/h を別カウントで列挙）。`md5sum` で SAME と出た 24 項目をそのまま使う。

### B群（差分あり・9ファイル）→ `*/overrides/` で並走させ Phase 5 で片寄せ

| ファイル | REG サイズ | DEF サイズ | 暫定方針 | 最終配置 |
|---|---|---|---|---|
| `AR.h` | 18865B | 18780B | REG 版採用（差 8 行のみ） | `common/src/`（Phase 5 で確定） |
| `MeshCleanup.h` | 25318B | 20737B | REG 版採用 | 同上 |
| `NoOpen3DRegistration.h` | 204777B | 194392B | REG 版採用 | 同上 |
| `OBJTargetExtraction.h` | 39332B | 35981B | REG 版採用 | 同上 |
| `RegistrationCore.h` | 39045B | 26949B | REG 版採用 | 同上 |
| `mCutMesh.h` | 74704B | 70353B | REG 版採用 | 同上 |
| `DepthRunner.h` | 18516B | 8774B | REG 版採用 | 同上 |
| `DepthUtils.h` | 27624B | 13150B | REG 版採用 | 同上 |
| **`AutoDeform.h`** | 70327B | **76212B** | **DEFORM 版採用** | 同上 |

> **方針**: ファイルサイズが大きい方を「機能リードしている」と仮定して採用候補にする。
> ただし Phase 5 で 1 件ずつ `diff` を取り、「小さい方にしかない関数 / マクロ / 構造体メンバ」を
> マージしてから片寄せる。失われる機能ゼロを目標にする。

### C群（REG 専用・26ファイル）→ `registration/src/reg_specific/`

```
AppContext.h
CameraPreview.h
CmaesRefineV2.h
CmaesRefineV3.h
CmaesRefineV3R.h
CmaesRefineV3RS.h
CmaesUtils.h
FileDropHandler.h
ImageSession.h
InteractionHelpers.h
IoUDebugDump.h
LiverCranioCaudalLabel.h
LiverLeftRightLabel.h
LiverRegionLabel.h
MaskPicker.h
NormalCompatibleRefine.h
OBJDistributionDiag.h
PoseLibrary.h
RegistrationActions.h
RegistrationImGuiManager.h
RegistrationUI.h
RimPairSampling.h
RimShapeMatch.h
SilOverlayDebug.h
UmeyamaController.h
Undistort.h
```

### D群（DEFORM 専用・3ファイル）→ `deform/src/deform_specific/`

```
DeformGlobals.h
DeformPipeline.h
Grabber.h
```

### main.cpp の扱い

- `${REG_SRC}/main.cpp` → `registration/main.cpp`
- `${DEF_SRC}/main.cpp` → `deform/main.cpp`
- `${ROOT}/元の長いmain.cpp` → `reference/元の長いmain.cpp`（読み取り専用、Phase 10 で参照）

---

## 3. Phase 0 — 準備（必須・絶対にスキップしない）

### 0.1 状態確認

```bash
cd ${ROOT}
git status
git log -1 --oneline
ls -la
```

**ガード**:
- `git status` で uncommitted な変更があれば **作業中止し、ユーザに通知**してコミットを促す。
- `.git` ディレクトリが無い場合は `git init && git add -A && git commit -m "snapshot before refactor"` を実行。

### 0.2 リファクタ用ブランチを切る

```bash
cd ${ROOT}
git checkout -b refactor/3project-split-$(date +%Y%m%d)
```

### 0.3 完全バックアップ（git とは別に物理コピー）

```bash
cd $(dirname ${ROOT})
cp -a AAA_LiverSurgeryNaviComb AAA_LiverSurgeryNaviComb_BACKUP_$(date +%Y%m%d_%H%M%S)
```

**検証**: バックアップディレクトリのファイル数とサイズが元と一致するか `du -sh` で確認。

### 0.4 現状のビルド成功を記録（リファレンス）

```bash
cd ${ROOT}
mkdir -p build_reg_before && cd build_reg_before
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake.log
cmake --build . -j$(nproc) 2>&1 | tee build.log
echo "=== REG BUILD EXIT CODE: $? ===" | tee -a build.log
ls -la bin/

cd ${ROOT}/AAA_LiverSurgeryNaviForDefotm
mkdir -p build_def_before && cd build_def_before
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake.log
cmake --build . -j$(nproc) 2>&1 | tee build.log
echo "=== DEF BUILD EXIT CODE: $? ===" | tee -a build.log
ls -la bin/
```

**ガード**:
- どちらかが ビルド失敗 した場合、**リファクタ前から壊れている**ことを意味する。
  → ユーザに通知し、原因を先に潰してもらう。リファクタは開始しない。

### 0.5 計画書を `${ROOT}/REFACTOR_PLAN.md` にコピーして git add

```bash
cp <この計画書> ${ROOT}/REFACTOR_PLAN.md
cd ${ROOT}
git add REFACTOR_PLAN.md
git commit -m "Add refactor plan"
```

---

## 4. Phase 1 — 骨格ディレクトリ作成

### 4.1 ディレクトリ生成（中身は空）

```bash
cd ${ROOT}
mkdir -p common/src
mkdir -p registration/src/reg_specific
mkdir -p registration/src/overrides
mkdir -p deform/src/deform_specific
mkdir -p deform/src/overrides
mkdir -p combined
mkdir -p reference

# .gitkeep を入れて git に追跡させる
touch common/src/.gitkeep
touch registration/src/reg_specific/.gitkeep
touch registration/src/overrides/.gitkeep
touch deform/src/deform_specific/.gitkeep
touch deform/src/overrides/.gitkeep
```

### 4.2 reference/ に過去スナップショットを移動

```bash
mv ${ROOT}/元の長いRegistrationImGuiManager.h ${ROOT}/reference/
mv ${ROOT}/元の長いmain.cpp ${ROOT}/reference/
chmod a-w ${ROOT}/reference/*  # 書き込み禁止にして誤改変防止

cat > ${ROOT}/reference/README.md <<'EOF'
# reference/

統合版（DEFORM と REGISTRATION が同じ UI で切り替わっていた頃）の
**過去スナップショット**。読み取り専用。

- `元の長いRegistrationImGuiManager.h`: 統合 UI の ImGui マネージャ全体
- `元の長いmain.cpp`: 統合版 main 全体（AppMode::kRegistration / kDeform 切替）

将来 `combined/` プロジェクトを作る際の **設計参考資料**。
このディレクトリのファイルは **絶対に編集しない**。
EOF
```

### 4.3 combined/ には README だけ置く

```bash
cat > ${ROOT}/combined/README.md <<'EOF'
# combined/

将来の統合プロジェクト用ディレクトリ。
**現時点では空**。Phase 10 以降で着手する。

設計方針:
- common/ の INTERFACE library を link
- registration/src/reg_specific/ と deform/src/deform_specific/ の両方を組み込む
- B群の同名差分ファイルは Phase 5 で片寄せ済みなので common/ のものを使用
- main.cpp は reference/元の長いmain.cpp をベースに、AppMode 切替で REG/DEFORM を両搭載
EOF
```

### 4.4 コミット

```bash
cd ${ROOT}
git add -A
git commit -m "Phase 1: scaffold common/registration/deform/combined directories"
```

### 4.5 検証

- `ls -la ${ROOT}` に新ディレクトリが存在
- 既存のビルド（`${ROOT}/build/` と `${ROOT}/AAA_LiverSurgeryNaviForDefotm/build/`）は **触らない**
- 既存の `${REG_SRC}/`, `${DEF_SRC}/` の中身は **一切変えていない**

---

## 5. Phase 2 — A群 24 ファイルを `common/src/` へ移動

### 5.1 移動コマンド（A群は REG 側から取る。md5 一致確認済みのため安全）

```bash
cd ${REG_SRC}
A_FILES=(
  AutoHandleController.h
  CentVoxTetrahedralizerHybrid.cpp
  CentVoxTetrahedralizerHybrid.h
  FullSphereCameraWithTarget.h
  HandleControllerBase.h
  Hash.h
  MeshDataTypes.cpp
  MeshDataTypes.h
  MeshDrawing.h
  PathConfig.h
  PinholeProjection.h
  PlatformCompat.h
  RayCast.h
  SemiAutoHandleController.h
  SemiAutoPickState.h
  SequentialDeformController.h
  ShaderProgram.cpp
  ShaderProgram.h
  SimpleCamera.hpp
  SoftBody.cpp
  SoftBody.h
  Sphere.cpp
  Sphere.h
  TetoMeshData.h
  VectorMath.h
  simple_multi_obj_processor.h
)

for f in "${A_FILES[@]}"; do
  if [ -f "${REG_SRC}/$f" ] && [ -f "${DEF_SRC}/$f" ]; then
    # 念のため md5 再確認
    h1=$(md5sum "${REG_SRC}/$f" | cut -d' ' -f1)
    h2=$(md5sum "${DEF_SRC}/$f" | cut -d' ' -f1)
    if [ "$h1" != "$h2" ]; then
      echo "ABORT: $f md5 mismatch ($h1 vs $h2)"
      exit 1
    fi
    git mv "${REG_SRC}/$f" "${ROOT}/common/src/$f"
    rm "${DEF_SRC}/$f"
    echo "MOVED: $f"
  else
    echo "WARN: $f missing in REG or DEF"
  fi
done
```

> **重要**:
> - `git mv` を使うこと（履歴を保つため）
> - DEF 側のファイルは単に `rm` で消す（git mv は使えない、別リポジトリ扱いの可能性）
> - md5 再確認は必須（Phase 0 と Phase 2 の間に手動編集があった場合の事故防止）

### 5.2 検証

```bash
# common/src に 24 ファイル + .gitkeep があるか
ls ${ROOT}/common/src | wc -l   # → 25 (24 + .gitkeep)

# REG src と DEF src からは消えているか
for f in "${A_FILES[@]}"; do
  [ -f "${REG_SRC}/$f" ] && echo "ERROR: still exists in REG: $f"
  [ -f "${DEF_SRC}/$f" ] && echo "ERROR: still exists in DEF: $f"
done
```

### 5.3 この時点ではまだビルドしない

- CMakeLists.txt は **まだ更新しない**ので、ビルドは壊れている状態。
- Phase 7 でまとめて CMake を直すまで「動かない状態」を許容する。
- **ただし git のステージングは確認**:
  ```bash
  cd ${ROOT}
  git status  # → renamed: で 24 件出るはず
  git diff --stat
  ```

### 5.4 コミット

```bash
cd ${ROOT}
git add -A
git commit -m "Phase 2: move 24 identical files to common/src (REG/DEF both updated)"
```

---

## 6. Phase 3 — C群 REG 専用 26 ファイルを `registration/src/reg_specific/` へ

### 6.1 移動

```bash
C_FILES=(
  AppContext.h
  CameraPreview.h
  CmaesRefineV2.h
  CmaesRefineV3.h
  CmaesRefineV3R.h
  CmaesRefineV3RS.h
  CmaesUtils.h
  FileDropHandler.h
  ImageSession.h
  InteractionHelpers.h
  IoUDebugDump.h
  LiverCranioCaudalLabel.h
  LiverLeftRightLabel.h
  LiverRegionLabel.h
  MaskPicker.h
  NormalCompatibleRefine.h
  OBJDistributionDiag.h
  PoseLibrary.h
  RegistrationActions.h
  RegistrationImGuiManager.h
  RegistrationUI.h
  RimPairSampling.h
  RimShapeMatch.h
  SilOverlayDebug.h
  UmeyamaController.h
  Undistort.h
)

cd ${ROOT}
for f in "${C_FILES[@]}"; do
  if [ -f "${REG_SRC}/$f" ]; then
    git mv "${REG_SRC}/$f" "${ROOT}/registration/src/reg_specific/$f"
    echo "MOVED REG-only: $f"
  else
    echo "WARN: $f missing in REG"
  fi
  # DEF 側に同名ファイルが**ない**ことを確認（あれば C群分類ミス）
  if [ -f "${DEF_SRC}/$f" ]; then
    echo "ERROR: $f exists in DEF too, classification wrong!"
    exit 1
  fi
done
```

### 6.2 main.cpp も移動

```bash
git mv "${REG_SRC}/main.cpp" "${ROOT}/registration/main.cpp"
```

### 6.3 検証

```bash
ls ${ROOT}/registration/src/reg_specific | wc -l  # → 27 (26 + .gitkeep)
ls ${ROOT}/registration/main.cpp                  # exists
ls ${REG_SRC}/  # → 残っているのは B群 9 ファイルのみのはず
```

### 6.4 コミット

```bash
git add -A
git commit -m "Phase 3: move 26 REG-only files to registration/src/reg_specific"
```

---

## 7. Phase 4 — D群 DEFORM 専用 3 ファイルを `deform/src/deform_specific/` へ

### 7.1 移動

```bash
D_FILES=(
  DeformGlobals.h
  DeformPipeline.h
  Grabber.h
)

cd ${ROOT}
for f in "${D_FILES[@]}"; do
  if [ -f "${DEF_SRC}/$f" ]; then
    # git mv が効くかは ${DEF_SRC} が同じ git リポジトリかによる
    if git ls-files --error-unmatch "${DEF_SRC}/$f" 2>/dev/null; then
      git mv "${DEF_SRC}/$f" "${ROOT}/deform/src/deform_specific/$f"
    else
      mv "${DEF_SRC}/$f" "${ROOT}/deform/src/deform_specific/$f"
    fi
    echo "MOVED DEF-only: $f"
  fi
  if [ -f "${REG_SRC}/$f" ]; then
    echo "ERROR: $f exists in REG too, classification wrong!"
    exit 1
  fi
done

# DEFORM 側の main.cpp も移動
if git ls-files --error-unmatch "${DEF_SRC}/main.cpp" 2>/dev/null; then
  git mv "${DEF_SRC}/main.cpp" "${ROOT}/deform/main.cpp"
else
  mv "${DEF_SRC}/main.cpp" "${ROOT}/deform/main.cpp"
fi
```

### 7.2 検証

```bash
ls ${ROOT}/deform/src/deform_specific | wc -l  # → 4 (3 + .gitkeep)
ls ${ROOT}/deform/main.cpp                     # exists
ls ${DEF_SRC}/  # → 残っているのは B群 9 ファイルのみ（DEF版）
```

### 7.3 コミット

```bash
git add -A
git commit -m "Phase 4: move 3 DEFORM-only files to deform/src/deform_specific"
```

---

## 8. Phase 5 — B群 9 ファイルの差分マージ & 片寄せ（最重要・最も慎重に）

### 8.1 まず両者を `*/overrides/` に並走させる

```bash
B_FILES=(
  AR.h
  MeshCleanup.h
  NoOpen3DRegistration.h
  OBJTargetExtraction.h
  RegistrationCore.h
  mCutMesh.h
  DepthRunner.h
  DepthUtils.h
  AutoDeform.h
)

cd ${ROOT}
for f in "${B_FILES[@]}"; do
  if [ -f "${REG_SRC}/$f" ]; then
    git mv "${REG_SRC}/$f" "${ROOT}/registration/src/overrides/$f"
  fi
  if [ -f "${DEF_SRC}/$f" ]; then
    if git ls-files --error-unmatch "${DEF_SRC}/$f" 2>/dev/null; then
      git mv "${DEF_SRC}/$f" "${ROOT}/deform/src/overrides/$f"
    else
      mv "${DEF_SRC}/$f" "${ROOT}/deform/src/overrides/$f"
    fi
  fi
done

git commit -m "Phase 5.1: place B-group files in respective overrides/ dirs"
```

> この時点で `${REG_SRC}/`, `${DEF_SRC}/` は **空** になっているはず。

### 8.2 各 B 群ファイルについて diff レポート生成

```bash
mkdir -p ${ROOT}/reference/diffs
for f in "${B_FILES[@]}"; do
  R="${ROOT}/registration/src/overrides/$f"
  D="${ROOT}/deform/src/overrides/$f"
  if [ -f "$R" ] && [ -f "$D" ]; then
    diff -u "$D" "$R" > "${ROOT}/reference/diffs/${f}.diff" || true
    echo "=== ${f} ===" >> ${ROOT}/reference/diffs/_summary.txt
    diff --brief "$R" "$D" >> ${ROOT}/reference/diffs/_summary.txt
    diff "$R" "$D" | wc -l >> ${ROOT}/reference/diffs/_summary.txt
  fi
done
```

### 8.3 ファイル別マージ方針（1 ファイルずつ慎重に）

各 B 群ファイルについて **以下の決定フローを実行**:

```
1. diff -u DEFORM版 REGISTRATION版 を眺める
2. 「DEFORM版にあって REG版にない」シンボル（関数/構造体/マクロ/include）を列挙
3. その中で**実際に使用されているもの**を grep で確認
   - DEFORM側で使われている: → REG版にマージしてから採用候補（基本これ）
   - DEFORM側でも未使用（dead code）: → 採用候補をそのまま使う
4. マージ後、最終版を common/src/ に移動
5. registration/src/overrides/ と deform/src/overrides/ から削除
```

#### 個別注意事項

| ファイル | 注意点 |
|---|---|
| **AR.h** | 差 8 行のみ。多分コメント差・改行差程度。**先頭の `#include` リストと namespace の宣言行が一致しているか**だけ確認すれば OK |
| **MeshCleanup.h** | 184 行差。REG 版が機能追加されている可能性。DEFORM 版独自の clean 処理が無いか要確認 |
| **NoOpen3DRegistration.h** | 252 行差・204 KB の大物。FGR/ICP 周りの最新化が REG 側で進んでいる。**DEFORM 版独自関数の有無を `grep "^inline\|^void\|^float" DEFORM版`** で列挙 |
| **OBJTargetExtraction.h** | 80 行差。`extractTargetFromOBJ` の引数シグネチャ差がないか要確認（あれば呼び出し側 main.cpp の修正必要） |
| **RegistrationCore.h** | 304 行差。`RegistrationData` 構造体のメンバ差がないか必ず確認。**メンバ差は両 main.cpp で初期化漏れを起こす最大の地雷** |
| **mCutMesh.h** | 104 行差。Mesh クラスのメンバ追加・削除はビルド全壊リスクあり |
| **DepthRunner.h** | DEF 側が大幅に小さい(8KB)。REG 側(18KB)が機能リッチ。DEF 版を捨てて REG 版採用で良い見込み |
| **DepthUtils.h** | DEF 側が大幅に小さい(13KB)。REG 側(27KB)採用で良い見込み |
| **AutoDeform.h** | ★★ **唯一 DEFORM 側がリード** ★★。DEFORM版 76KB vs REG版 70KB。**REG 版にしかない機能** がないか特に注意してマージ。マージ後 DEFORM 版ベースで採用 |

#### 1ファイル分のマージ作業テンプレ

```bash
# 例: AR.h
F=AR.h
R="${ROOT}/registration/src/overrides/$F"
D="${ROOT}/deform/src/overrides/$F"

# 1. diff を確認
diff -u "$D" "$R" | less

# 2. DEFORM 側にしかない可能性のあるシンボルを抽出
diff "$D" "$R" | grep "^< " | grep -E "^\< (inline|static|void|float|int|class|struct|namespace)" || echo "no DEFORM-unique symbols"

# 3. DEFORM 側のそのシンボルが実際使われているか確認
#    (DEFORM 側 main.cpp / DeformPipeline.h などから grep)
grep -r "シンボル名" ${ROOT}/deform/

# 4a. マージ不要（REG版で十分）の場合:
git rm "$D"
git mv "$R" "${ROOT}/common/src/$F"

# 4b. マージ必要（DEFORM版にしかない機能あり）の場合:
#    REG版を編集して DEFORM 版の機能を追加 → その後上記 4a を実行

# 5. コミット
git commit -m "Phase 5.X: unify ${F} (taking REG version, ...notes...)"
```

#### `AutoDeform.h` の特殊処理（リード方向が逆）

```bash
F=AutoDeform.h
R="${ROOT}/registration/src/overrides/$F"
D="${ROOT}/deform/src/overrides/$F"

# DEFORM 版がリード → 基本 DEFORM 版を採用
diff -u "$R" "$D" | less   # REG 版にしかないものを確認

# REG 側にしかない機能があれば DEFORM 版に追記してから採用
git rm "$R"
git mv "$D" "${ROOT}/common/src/$F"
git commit -m "Phase 5.X: unify AutoDeform.h (taking DEFORM version - DEFORM leads)"
```

### 8.4 Phase 5 完了時の検証

```bash
# overrides/ が空になっているか
ls ${ROOT}/registration/src/overrides/  # → .gitkeep のみ
ls ${ROOT}/deform/src/overrides/        # → .gitkeep のみ

# common/src に B群 9 ファイルが追加されているか
for f in "${B_FILES[@]}"; do
  [ -f "${ROOT}/common/src/$f" ] || echo "MISSING in common: $f"
done

# common/src の総ファイル数: 24(A) + 9(B) + .gitkeep = 34
ls ${ROOT}/common/src | wc -l
```

### 8.5 もし「片寄せできない（互換不能な差）」が判明したら

- 該当ファイルだけ **`*/overrides/` に残し続ける** ことを許容する
- CMake の include 順で `overrides/` を `common/src/` より先に通す
- TODO リストとして `${ROOT}/reference/diffs/UNRESOLVED.md` に記録
- ユーザに「このファイルは将来 combined/ 着手時に再検討」と報告

---

## 9. Phase 6 — 共有リソースの整理

### 9.1 重複ディレクトリの統合方針

`${ROOT}/AAA_LiverSurgeryNaviForDefotm/` 配下にある以下のリソースは
**親（`${ROOT}/`）と重複している可能性が高い**。`diff -r` で比較してから統合する。

```bash
# 重複候補
for dir in third_party shaders chessboard calibration_tool sam2_da3_lite win_deps onnxruntime-linux-x64-1.15.1 model; do
  P="${ROOT}/$dir"
  S="${ROOT}/AAA_LiverSurgeryNaviForDefotm/$dir"
  if [ -d "$P" ] && [ -d "$S" ]; then
    echo "=== $dir ==="
    diff -rq "$P" "$S" | head -20
    echo ""
  fi
done
```

### 9.2 統合ルール

| ディレクトリ | 中身が同じ場合 | 中身が違う場合 |
|---|---|---|
| `third_party/` | DEF 側を削除 | 「DEF 側にしかないライブラリ」を REG 側にマージしてから削除 |
| `shaders/` | DEF 側を削除 | DEF 専用シェーダがあればファイル名を変えてマージ |
| `chessboard/` | DEF 側を削除 | 画像なら REG 側に統合 |
| `calibration_tool/` | DEF 側を削除 | 同上 |
| `sam2_da3_lite/` | DEF 側を削除 | 同上 |
| `win_deps/` | DEF 側を削除 | バージョン差注意。新しい方を採用 |
| `onnxruntime-...` | DEF 側を削除 | バージョン差確認 |
| `model/` | **両方残す方向で検討**（REG 用と DEF 用で別物の可能性） | `registration/model/` と `deform/model/` で分離 |
| `data/` | 同上 | `registration/data/`, `deform/data/` |
| `input_image/` | REG 専用 | `registration/input_image/` |
| `registration_model/` | REG の出力 = DEF の入力 | `registration/registration_model/` に置き、DEF からはシンボリックリンク |
| `depth_output/` | 両方の出力先 | ルート直下に残す |
| `resized_1280x720/` | 用途確認 | 大体 REG 専用 |

### 9.3 統合作業

```bash
cd ${ROOT}

# 例: third_party が完全一致なら DEF 側削除
if diff -rq third_party AAA_LiverSurgeryNaviForDefotm/third_party | grep -q .; then
  echo "third_party differs - manual review needed"
else
  rm -rf AAA_LiverSurgeryNaviForDefotm/third_party
  echo "removed DEF third_party (identical)"
fi

# DEF プロジェクト固有のディレクトリ (deform/registration_model/ など) は移動
if [ -d AAA_LiverSurgeryNaviForDefotm/registration_model ]; then
  mv AAA_LiverSurgeryNaviForDefotm/registration_model deform/registration_model
fi
```

### 9.4 `AAA_LiverSurgeryNaviForDefotm/` ディレクトリの最終的な扱い

Phase 6 終了時点で `AAA_LiverSurgeryNaviForDefotm/` の中身は ほぼ空 になっているはず。
最終的に **このディレクトリ自体を削除** する:

```bash
# 残っているものを確認
find ${ROOT}/AAA_LiverSurgeryNaviForDefotm -type f

# 何も無ければ削除
rmdir ${ROOT}/AAA_LiverSurgeryNaviForDefotm/{build,build_def_before,src,delete_src,'src (コピー)','model\high','modelいの'} 2>/dev/null
rm -rf ${ROOT}/AAA_LiverSurgeryNaviForDefotm  # 全部消す
```

> **重要**: `AAA_LiverSurgeryNaviForDefotm/build*/` などビルド成果物は削除して OK。
> ただし `delete_src/` `src (コピー)/` 等の「明らかにバックアップ的な名前のディレクトリ」は
> **削除前に中身を確認**し、もし unique なファイルがあれば `reference/` に退避する。

### 9.5 コミット

```bash
git add -A
git commit -m "Phase 6: consolidate shared resources, remove duplicate dirs in DEF subdir"
```

---

## 10. Phase 7 — CMakeLists.txt の書き直し

### 10.1 `${ROOT}/common/CMakeLists.txt`（新規作成）

```cmake
# common/CMakeLists.txt
# ===========================================================================
# Header-only / source-included common code for both REGISTRATION and DEFORM.
# Exposes:
#   - INTERFACE include path: ${CMAKE_CURRENT_SOURCE_DIR}/src
#   - OBJECT sources for .cpp files (linked into each executable)
# ===========================================================================

# .cpp ファイルだけ OBJECT library にまとめる
file(GLOB LSN_COMMON_CPP CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(lsn_common_obj OBJECT ${LSN_COMMON_CPP})
target_include_directories(lsn_common_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# INTERFACE ライブラリで include path を expose
add_library(lsn_common INTERFACE)
target_include_directories(lsn_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(lsn_common INTERFACE lsn_common_obj)

message(STATUS "lsn_common: ${CMAKE_CURRENT_SOURCE_DIR}/src")
```

### 10.2 `${ROOT}/registration/CMakeLists.txt`（新規作成）

```cmake
# registration/CMakeLists.txt
# ===========================================================================
# REGISTRATION application
# - sources: main.cpp + reg_specific/*.cpp + overrides/*.cpp (if any)
# - links:   lsn_common + 共通 third_party (imgui, tinyobj, cmaes, ...)
# ===========================================================================

file(GLOB LSN_REG_SPECIFIC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reg_specific/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/overrides/*.cpp
)

add_executable(lsn_registration
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${LSN_REG_SPECIFIC}
    ${IMGUI_SOURCES}
    ${TINYOBJ_SOURCES}
    ${LSN_THIRD_PARTY_SOURCES}  # tinyfiledialogs.c, cmaes_tls_wrapper.c などをルートで定義しておく
)

# include path:
#   1. overrides/  (B群 REG版を優先)
#   2. reg_specific/
#   3. common/src/ (lsn_common 経由)
target_include_directories(lsn_registration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/overrides
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reg_specific
    # third_party はルート CMake で LSN_THIRD_PARTY_INCLUDES として定義
    ${LSN_THIRD_PARTY_INCLUDES}
)

target_link_libraries(lsn_registration PRIVATE
    lsn_common
    ${LSN_PLATFORM_LIBS}
)

target_compile_definitions(lsn_registration PRIVATE
    GLEW_STATIC
    _USE_MATH_DEFINES
    NOMINMAX
    LSN_MODE_REGISTRATION
)

set_target_properties(lsn_registration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# データディレクトリのコピー（既存 CMake と同じ挙動）
foreach(d data model shaders registration_model input_image)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${d})
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${d}
             DESTINATION ${CMAKE_BINARY_DIR}/bin)
    endif()
endforeach()
```

### 10.3 `${ROOT}/deform/CMakeLists.txt`（新規作成）

```cmake
# deform/CMakeLists.txt
file(GLOB LSN_DEF_SPECIFIC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_specific/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/overrides/*.cpp
)

add_executable(lsn_deform
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${LSN_DEF_SPECIFIC}
    ${IMGUI_SOURCES}
    ${TINYOBJ_SOURCES}
    ${LSN_THIRD_PARTY_SOURCES}
)

target_include_directories(lsn_deform PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/overrides
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_specific
    ${LSN_THIRD_PARTY_INCLUDES}
)

target_link_libraries(lsn_deform PRIVATE
    lsn_common
    ${LSN_PLATFORM_LIBS}
)

target_compile_definitions(lsn_deform PRIVATE
    GLEW_STATIC
    _USE_MATH_DEFINES
    NOMINMAX
    LSN_MODE_DEFORM
)

set_target_properties(lsn_deform PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

foreach(d data model shaders registration_model)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${d})
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${d}
             DESTINATION ${CMAKE_BINARY_DIR}/bin)
    endif()
endforeach()
```

### 10.4 ルート `${ROOT}/CMakeLists.txt`（既存を全面書き換え）

> 既存 CMakeLists.txt の **third_party / ImGui / Eigen / OpenMP / OpenGL 検出ロジックは流用** する。
> その後で `add_subdirectory(common)`, `(registration)`, `(deform)` を追加し、
> 共通設定を変数 `LSN_THIRD_PARTY_INCLUDES`, `LSN_THIRD_PARTY_SOURCES`, `LSN_PLATFORM_LIBS` にまとめる。

```cmake
cmake_minimum_required(VERSION 3.16)
project(LiverSurgeryNaviComb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------- ビルドターゲット選択 --------
option(BUILD_REG       "Build REGISTRATION app" ON)
option(BUILD_DEFORM    "Build DEFORM app"       ON)
option(BUILD_COMBINED  "Build COMBINED app (future)" OFF)
option(ENABLE_AVX2     "Enable AVX2" ON)
option(ENABLE_SSE2     "Enable SSE2" ON)

# -------- コンパイラフラグ (既存と同じ) --------
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/wd4819 /wd4005 /wd4267 /wd4244 /wd4996 /wd4458 /wd4312 /wd4305 /MP /utf-8)
    if(ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wno-volatile -Wno-sign-compare -Wno-unused-parameter -Wno-conversion -Wno-deprecated-declarations)
    if(ENABLE_AVX2)
        add_compile_options(-mavx2)
    endif()
endif()

# -------- 既存の third_party 検出ロジック (流用) --------
# WIN_DEPS_DIR, INCLUDES_DIR, LIBS_DIR
# GLM_DIR
# STB_IMAGE_DIR
# IMGUI_DIR + IMGUI_SOURCES
# TINYOBJ_DIR + TINYOBJ_SOURCES
# CMAES_DIR
# (… 既存 CMakeLists.txt の line 24〜138 をそのままコピー …)

# -------- サブプロジェクトが共有する変数を作る --------
set(LSN_THIRD_PARTY_INCLUDES
    ${STB_IMAGE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs
    ${CMAKE_SOURCE_DIR}/third_party/nanoflann
    ${TINYOBJ_DIR}
    ${GLM_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAES_DIR}
    CACHE INTERNAL "Third-party include dirs shared by all apps"
)

set(LSN_THIRD_PARTY_SOURCES "" CACHE INTERNAL "Third-party sources to compile into each app")
if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs/tinyfiledialogs.c)
    list(APPEND LSN_THIRD_PARTY_SOURCES ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs/tinyfiledialogs.c)
    add_compile_definitions(HAS_TINYFILEDIALOGS)
endif()
if(EXISTS "${CMAES_DIR}/cmaes_tls_wrapper.c" AND EXISTS "${CMAES_DIR}/cmaes.c")
    list(APPEND LSN_THIRD_PARTY_SOURCES "${CMAES_DIR}/cmaes_tls_wrapper.c")
elseif(EXISTS "${CMAES_DIR}/cmaes.c")
    list(APPEND LSN_THIRD_PARTY_SOURCES "${CMAES_DIR}/cmaes.c")
endif()

# -------- プラットフォーム別ライブラリ --------
set(LSN_PLATFORM_LIBS "" CACHE INTERNAL "Platform-specific libs")
if(WIN32)
    list(APPEND LSN_PLATFORM_LIBS opengl32 glew32s glfw3_mt)
    link_directories(${LIBS_DIR})
else()
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glfw3 REQUIRED)
    list(APPEND LSN_PLATFORM_LIBS ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw)
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND LSN_PLATFORM_LIBS OpenMP::OpenMP_CXX)
endif()
if(OpenMP_C_FOUND)
    list(APPEND LSN_PLATFORM_LIBS OpenMP::OpenMP_C)
endif()

# Eigen
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen/Eigen/Dense")
    list(APPEND LSN_THIRD_PARTY_INCLUDES ${CMAKE_SOURCE_DIR}/third_party/eigen)
elseif(EXISTS "${INCLUDES_DIR}/eigen3/Eigen/Dense")
    list(APPEND LSN_THIRD_PARTY_INCLUDES ${INCLUDES_DIR}/eigen3)
elseif(EXISTS "${INCLUDES_DIR}/Eigen/Dense")
    list(APPEND LSN_THIRD_PARTY_INCLUDES ${INCLUDES_DIR})
endif()

# -------- 子プロジェクト --------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(common)

if(BUILD_REG)
    add_subdirectory(registration)
endif()
if(BUILD_DEFORM)
    add_subdirectory(deform)
endif()
if(BUILD_COMBINED)
    if(EXISTS ${CMAKE_SOURCE_DIR}/combined/CMakeLists.txt)
        add_subdirectory(combined)
    else()
        message(STATUS "combined/ has no CMakeLists.txt yet, skipping")
    endif()
endif()

# 既存のサブプロジェクト (sam2_da3_lite, calibration_tool) は流用
if(EXISTS ${CMAKE_SOURCE_DIR}/sam2_da3_lite/CMakeLists.txt)
    add_subdirectory(sam2_da3_lite ${PROJECT_BINARY_DIR}/sam2_da3_lite)
    set_target_properties(sam2_da3_lite PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/calibration_tool/CMakeLists.txt)
    add_subdirectory(calibration_tool ${PROJECT_BINARY_DIR}/calibration_tool)
    set_target_properties(calibration_tool PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif()

# 共有 shaders/data はルートでもコピー（バックアップ用）
if(EXISTS ${CMAKE_SOURCE_DIR}/shaders)
    file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()

# サマリ
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Project:    ${PROJECT_NAME}")
message(STATUS "BUILD_REG:        ${BUILD_REG}")
message(STATUS "BUILD_DEFORM:     ${BUILD_DEFORM}")
message(STATUS "BUILD_COMBINED:   ${BUILD_COMBINED}")
message(STATUS "Output:     ${PROJECT_BINARY_DIR}/bin/")
message(STATUS "========================================")
```

### 10.5 旧 CMakeLists.txt を退避

```bash
cd ${ROOT}
mv CMakeLists.txt reference/CMakeLists.txt.old_combined
# 上記 10.4 の新ルート CMakeLists.txt を ${ROOT}/CMakeLists.txt として作成
```

### 10.6 コミット

```bash
git add -A
git commit -m "Phase 7: rewrite CMakeLists.txt for 3-project layout"
```

---

## 11. Phase 8 — ビルド検証

### 11.1 ビルド試行

```bash
cd ${ROOT}
rm -rf build  # クリーンビルド
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_REG=ON \
    -DBUILD_DEFORM=ON \
    -DBUILD_COMBINED=OFF \
    2>&1 | tee cmake.log

cmake --build . -j$(nproc) 2>&1 | tee build.log
EXIT_CODE=$?
echo "=== BUILD EXIT CODE: ${EXIT_CODE} ==="
```

### 11.2 想定エラーと対処

| エラーパターン | 原因 | 対処 |
|---|---|---|
| `fatal error: 'XXX.h' file not found` | include path 不足 | 該当 .h が common か reg か def かを確認、CMake の `target_include_directories` を修正 |
| `undefined reference to 'YYY'` | 必要な .cpp が SOURCES に入っていない | `file(GLOB ...)` のパターンを修正、または explicit に追加 |
| `multiple definition of 'ZZZ'` | header-only と思っていた関数が inline 抜けで cpp 化されて両方リンク | 該当関数に `inline` を付ける、または 1 か所だけで定義 |
| `redefinition of 'stbi__err'` 等 | stb_image の `STB_IMAGE_IMPLEMENTATION` 多重定義 | main.cpp 内の `#define STB_IMAGE_IMPLEMENTATION` 直後の `#undef` が抜けていないか確認 |
| `RegistrationData::xxxx` 未定義 | B群片寄せでメンバが消えた | Phase 5 の `RegistrationCore.h` マージをやり直す |
| `gApp.mode` 未定義 (DEF 側で) | DEF 側で AppContext.h を参照していた | DEF 側 main.cpp で `LSN_MODE_DEFORM` を見て条件コンパイル化 |

### 11.3 両方の実行ファイルが生成されたか

```bash
ls -la ${ROOT}/build/bin/
# 期待:
#   lsn_registration  (実行ファイル)
#   lsn_deform        (実行ファイル)
#   sam2_da3_lite  (もし build 対象なら)
#   calibration_tool  (もし build 対象なら)
#   shaders/          (コピー済み)
```

### 11.4 ビルドが通ったらコミット

```bash
git add -A  # build/ は .gitignore 対象なので何も追加されないはず
git commit --allow-empty -m "Phase 8: both lsn_registration and lsn_deform build successfully"
```

### 11.5 ビルドが通らない場合のロールバック

```bash
# Phase 7 のコミットまで戻す
git reset --hard HEAD~1  # Phase 7 だけ戻す
# または完全に Phase 0 まで戻す
git checkout main
git branch -D refactor/3project-split-XXXXX
# バックアップから復元
cp -a ../AAA_LiverSurgeryNaviComb_BACKUP_XXXX/* ${ROOT}/
```

---

## 12. Phase 9 — スモークテスト

### 12.1 lsn_registration の起動確認

```bash
cd ${ROOT}/build/bin
./lsn_registration &
PID=$!
sleep 10
if kill -0 $PID 2>/dev/null; then
    echo "REG: launched OK"
    kill $PID
else
    echo "REG: crashed at startup"
    exit 1
fi
```

> ※ GUI アプリなので CI 環境では headless ビルド確認まで。実機での GUI 動作はユーザ確認。

### 12.2 lsn_deform の起動確認

```bash
cd ${ROOT}/build/bin
./lsn_deform &
PID=$!
sleep 10
if kill -0 $PID 2>/dev/null; then
    echo "DEFORM: launched OK"
    kill $PID
else
    echo "DEFORM: crashed at startup"
    exit 1
fi
```

### 12.3 ユーザに動作確認を依頼

Claude Code は以下の **手動確認チェックリスト** をユーザに提示する:

```
□ lsn_registration を起動し、画像をドラッグ&ドロップして REG 動作を一通り試す
   - チェスボード認識
   - OBJ ロード
   - ICP 実行 (ボタン)
   - CMAES Refine 実行
   - AR モード (Key A)
   - AR Save (Key D)
□ lsn_deform を起動し、reg_*.obj を読み込んで DEFORM 動作を一通り試す
   - Key R / H / D / C のモード切替
   - 左クリックでハンドル配置
   - 左ドラッグで変形
   - AR モード (Key A)
□ depth_output/ への書き出しが両方で動く
□ shaders/ が両方でロードされる
```

### 12.4 コミット & マージ提案

```bash
git commit --allow-empty -m "Phase 9: smoke tests passed for both apps"
git log --oneline | head -20  # 全 Phase コミットが見える

# main へのマージ準備
echo "次のステップ: ユーザに git checkout main && git merge refactor/... の承認を求める"
```

---

## 13. Phase 10 (将来) — combined プロジェクトの復活

**Phase 9 までが今回のスコープ**。combined は後日着手。

設計指針（メモのみ）:
- `combined/main.cpp` は `reference/元の長いmain.cpp` をベースに、
  以下を共通ヘッダ参照に書き換える:
  - `#include "AppContext.h"` ← `registration/src/reg_specific/` から
  - `#include "DeformGlobals.h"` ← `deform/src/deform_specific/` から
- `gApp.mode == AppMode::kRegistration` / `kDeform` で分岐するロジックを残す
- `combined/CMakeLists.txt` は `lsn_common` + reg_specific の全 cpp + deform_specific の全 cpp を全部リンクする
- B 群ファイルが片寄せ済みなので衝突しない（Phase 5 完了が前提）

---

## 14. 全 Phase 共通ルール

### 14.1 コミットメッセージ規約

```
Phase N[.X]: <一行サマリ>

<必要なら詳細>
```

例:
```
Phase 5.3: unify RegistrationCore.h (taking REG version)

RegistrationData struct has 2 extra members in REG (refineInitialRMSE,
compHausdorff) that DEFORM never used. Adopted REG version as-is.
```

### 14.2 各 Phase 終了時のチェックリスト

```
□ git status で意図しない変更が無い
□ git log で Phase コミットが期待通り
□ ${REG_SRC}/, ${DEF_SRC}/ の中身が期待通り（Phase 5 終了時点で空）
□ common/src の中身が期待数（最終 33 ファイル: 24 A群 + 9 B群）
□ Phase 7 以降はビルド成功
```

### 14.3 「迷ったら止まる」原則

以下の場合 **作業を一時中断してユーザに判断を仰ぐ**:

- B 群ファイルのマージで「DEFORM 版にしかない非自明な機能」を発見
- `model/`, `data/` ディレクトリの中身が REG/DEF で違う
- 「これは A 群と判定したが diff が出ている」など分類矛盾
- ビルドエラーが想定外（上記 11.2 の表に無いパターン）
- `AAA_LiverSurgeryNaviForDefotm/` 配下の `delete_src/`, `src (コピー)/`, `modelいの/` などの謎ディレクトリの中身判断

---

## 15. 既知のハザード（注意リスト）

| # | ハザード | 対策 |
|---|---|---|
| H1 | DEFORM プロジェクトが別 git リポジトリの可能性 | Phase 0 で `cd ${DEF_SRC}/.. && git status` を確認。別リポジトリなら `git mv` 不可、`mv` + `git add` で対応 |
| H2 | `元の長いmain.cpp` を編集してしまう | Phase 1 で `chmod a-w` を施す |
| H3 | `model\high` `modelいの` `src (コピー)` などの全角・スペース込みディレクトリ | shell でクォート必須。`"..."` で囲む |
| H4 | Linux/Windows パス区切り差 (model\high はバックスラッシュ含む) | `find -name 'model\\high'` のようにエスケープ |
| H5 | `STB_IMAGE_IMPLEMENTATION` の多重定義 | 各 main.cpp の `#undef` を必ず維持 (DEFORM 用のmain.cpp に既存コメントあり) |
| H6 | `AutoDeform.h` だけ DEFORM 版採用なのを忘れる | Phase 5.X コミットメッセージに明記 |
| H7 | `RegistrationData` 構造体のメンバが REG/DEF で違う | Phase 5 で `RegistrationCore.h` マージ時に最も注意 |
| H8 | `cmaes_tls_wrapper.c` の OpenMP 連携が外れる | Phase 7 ルート CMake で `LSN_THIRD_PARTY_SOURCES` に確実に追加 |
| H9 | `sam2_da3_lite/` の subdir ビルドが両プロジェクトから参照されると重複ビルド | ルート CMake で 1 回だけ `add_subdirectory` する（上記 10.4 の通り） |
| H10 | `registration_model/` を REG が書き出し DEF が読み込む依存関係 | Phase 6 でシンボリックリンクを deform/registration_model -> ../registration/registration_model に張る運用を推奨 |
| H11 | `shaders/` を 2 か所にコピーすると古い方を編集して反映されないバグ | ルート shaders/ のみを正とし、ビルド時に各 bin/ にコピー（Phase 7 で実装済み） |

---

## 16. 完了条件（Definition of Done）

以下を **全て満たした時点で完了**:

1. `${ROOT}/common/src/` に 33 ファイル（24 A群 + 9 B群片寄せ後）
2. `${ROOT}/registration/src/reg_specific/` に 26 ファイル
3. `${ROOT}/deform/src/deform_specific/` に 3 ファイル
4. `${REG_SRC}/`, `${DEF_SRC}/`, `${ROOT}/AAA_LiverSurgeryNaviForDefotm/` が削除されている
5. `${ROOT}/reference/` に `元の長い*.h/.cpp` が読み取り専用で保存
6. `cmake -DBUILD_REG=ON -DBUILD_DEFORM=ON ..` が成功
7. `cmake --build .` が成功し `bin/lsn_registration` と `bin/lsn_deform` が生成
8. 両実行ファイルがクラッシュせず起動できる（10 秒生存）
9. ユーザによる手動 GUI 動作確認をパス
10. `git log` に Phase 0〜9 のコミットが残っている

---

## 17. Claude Code への最終指示

```
あなたはこの計画書に従って Phase 0 から順に実行してください。

各 Phase 終了時に必ず:
  1. このドキュメント 14.2 のチェックリストを実施
  2. 結果を私（ユーザ）に簡潔に報告
  3. 次の Phase に進む承認を求める

特に Phase 5 (B 群片寄せ) は 1 ファイルずつ:
  - diff を表示
  - マージ方針を提案
  - 私の承認を得てから実行

Phase 7 (CMakeLists.txt 書き直し) は:
  - 旧 CMakeLists.txt の third_party 検出ロジックを完全に保持
  - 新規追加分は最小限に
  - 書き直し後の差分を私に見せてから commit

「迷ったら止まる」を厳守してください。
```

---

**END OF PLAN**
