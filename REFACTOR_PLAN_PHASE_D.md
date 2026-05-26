# common/ DEFORM 寄りファイル切り離し 指示計画書

> **対象実行者**: Claude Code (CLI)
> **対象ディレクトリ**: `/home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviComb`
> **前提**: 前回の `REFACTOR_PLAN_for_ClaudeCode.md` による 3 本立て化（common / registration / deform / combined）が完了し、`lsn_registration` と `lsn_deform` が両方ビルド・起動できる状態。

---

## 1. 前回のリファクタリングの軽いまとめ

前回の作業で以下が完了している:

- ルート直下に 3 ディレクトリを作った: `common/`, `registration/`, `deform/`, （将来用に `combined/`）
- A 群 24 ファイル（REG と DEFORM で完全同一）と B 群 9 ファイル（差分マージ後）の計 33 ファイルを `common/src/` に集約
- REG 専用 26 ファイル → `registration/src/reg_specific/`
- DEFORM 専用 3 ファイル（DeformGlobals, DeformPipeline, Grabber）→ `deform/src/deform_specific/`
- ルート CMakeLists.txt を `option(BUILD_REG/BUILD_DEFORM/BUILD_COMBINED)` 切替式に書き換え

**結果として動く状態にはなったが**、`common/src/` には「両方が使える形になっているが概念的には DEFORM のもの」が大量に紛れ込んでいる。本計画書はそれを **deform 側へ寄せる二次リファクタリング**。

---

## 2. ディレクトリ命名の変更

前回 `deform/src/deform_specific/` という冗長な名前を使っていたが、これを廃止して **`deform/src/` 直下にフラットに置く**。サブディレクトリは作らず、DEFORM 専用ファイルは全部 `deform/src/` の直下に並べる。

`registration/src/reg_specific/` も本来は同様に `registration/src/` 直下にフラット化したいが、本計画書では **deform 側だけ** 整理する（REG 側は次回以降で検討）。

---

## 3. 切り離す対象ファイル（概念的に DEFORM 専用）

### 3.1 まるごと common/src/ → deform/src/ に移動するもの（11 ファイル）

| # | ファイル | 移動の根拠 |
|---|---|---|
| 1 | `SoftBody.h` | XPBD 物理シミュレータ本体。REG では参照されない |
| 2 | `SoftBody.cpp` | 同上 |
| 3 | `CentVoxTetrahedralizerHybrid.h` | 四面体化。`DeformPipeline.h` でのみ参照 |
| 4 | `CentVoxTetrahedralizerHybrid.cpp` | 同上 |
| 5 | `TetoMeshData.h` | `SoftBody::MeshData` を扱う OBJ ローダ |
| 6 | `HandleControllerBase.h` | 1 行目で `#include "SoftBody.h"`、API が `SoftBody*` |
| 7 | `AutoHandleController.h` | HandleControllerBase 継承 |
| 8 | `SemiAutoHandleController.h` | 同上 |
| 9 | `SemiAutoPickState.h` | DEFORM のピック state（fix/move pair） |
| 10 | `SequentialDeformController.h` | 順次変形コントローラ |
| 11 | `AutoDeform.h` | AutoDeform パイプライン本体 |

### 3.2 RayCast.h の汎用化（新規ファイル作らず単純化）

**現状の問題点**:

`RayCast.h` は SoftBody 用とメッシュ汎用用で **二系統の API** を持っている:

```cpp
// SoftBody 専用
struct RayHit { bool hit; float distance; glm::vec3 position; SoftBody* hitObject; };
static RayHit intersectMesh(const Ray& ray, SoftBody& mesh);

// メッシュ汎用 (mCutMesh から使用)
struct RayHitTri { bool hit; float distance; glm::vec3 position; };
static RayHitTri intersectMesh(const Ray& ray,
                               std::vector<GLfloat> vertices,
                               std::vector<GLuint>  indices);
```

しかし、SoftBody 版の関数の中身を読むと、本質的にやっているのは:

```cpp
const auto& positions     = mesh.getPositions();              // → vector<float>
const auto& surfaceTriIds = mesh.getMeshData().tetSurfaceTriIds; // → vector<int>
// あとは ray-triangle intersect ループ
```

**つまり `SoftBody&` を引数に取っているのは、頂点配列とインデックス配列を取り出すための回りくどい手段でしかない**。本質的な計算は完全に generic。

さらに **Grabber は戻り値の `RayHit::hitObject` (SoftBody*) を一度も使っていない**。Grabber は自分の `physicsObject` メンバを持っているので、その情報が戻ってくる必要がない。

つまり SoftBody 版 API 全体が「不要な抽象化」になっており、削除して **既存の汎用版に統一できる**。

**統一後の API**:

```cpp
class RayCast {
public:
    struct Ray { glm::vec3 origin, direction; };
    
    struct RayHit {                  // RayHit / RayHitTri を統合
        bool      hit;
        float     distance;
        glm::vec3 position;
    };
    
    static Ray screenToRay(...);
    
    // template 化して int / GLuint 両対応
    template<typename IndexT>
    static RayHit intersectMesh(
        const Ray& ray,
        const std::vector<float>& vertices,    // xyz triples
        const std::vector<IndexT>& indices);   // 3つで1三角形
    
    static bool rayTriangleIntersect(...);  // public のまま
};
```

**変更量**:
- `RayCast.h`: SoftBody 関連 30 行削除 + template 化 1 行 + `RayHitTri` → `RayHit` 統合
- `Grabber.h`: 4 箇所の `intersectMesh(localRay, *physicsObject)` を `intersectMesh(localRay, verts, tris)` に書き換え
- 新規ファイル **ゼロ**

### 3.3 MeshDrawing.h の分割

現状の `MeshDrawing.h` には:

- mCutMesh ベースの描画関数群 → 共通
- `drawCombinedDepthSorted(SoftBody* softBody, ...)` → DEFORM 用（クラス外の自由関数）

**分離方針**:
- `common/src/MeshDrawing.h` から `drawCombinedDepthSorted` の SoftBody 版を削除し、`#include "SoftBody.h"` も削除
- 新規 `deform/src/MeshDrawingSoftBody.h` を作成し、SoftBody 版だけそこに移動
- DeformPipeline.h など呼び出し側で `#include "MeshDrawingSoftBody.h"` を追加

> 注意: RayCast と違って、ここは「SoftBody* を受け取って `visVAOs` `vis_positions_array` などの SoftBody 固有メンバをアクセスする」描画関数なので、RayCast のように generic 化はできない。新規ヘッダで分離する方法を取る。

### 3.4 判断保留 — `Sphere.h` / `Sphere.cpp` の扱い

両プロジェクトとも「マーカー描画」用途で同じクラスを使用しており、中身は単純な「単位球メッシュ生成 + 描画」で SoftBody 依存ゼロ。

**結論**: 今回は **触らない**。`common/src/Sphere.h` の先頭に将来の継承化方針を記す **TODO コメント** を追加するのみ（Phase D-5）。

---

## 4. 移動後の最終構成（期待値）

```
common/src/                       現状 33 → 移動後 約 22 ファイル
├── （A 群残留・SoftBody 非依存）
│   AR.h, FullSphereCameraWithTarget.h, Hash.h, MeshDataTypes(.h/.cpp),
│   MeshDrawing.h, PathConfig.h, PinholeProjection.h, PlatformCompat.h,
│   RayCast.h, ShaderProgram(.h/.cpp), SimpleCamera.hpp, Sphere(.h/.cpp),
│   VectorMath.h, simple_multi_obj_processor.h
└── （B 群残留）
    mCutMesh.h, MeshCleanup.h, NoOpen3DRegistration.h, OBJTargetExtraction.h,
    RegistrationCore.h, DepthRunner.h, DepthUtils.h

deform/src/                       現状 3 (deform_specific 内) → 移動後 約 15 ファイル
├── main.cpp                     ※既存
├── （既存 3、deform_specific から昇格）
│   DeformGlobals.h, DeformPipeline.h, Grabber.h
├── （common から移動 11）
│   SoftBody.h, SoftBody.cpp, CentVoxTetrahedralizerHybrid.h,
│   CentVoxTetrahedralizerHybrid.cpp, TetoMeshData.h,
│   HandleControllerBase.h, AutoHandleController.h,
│   SemiAutoHandleController.h, SemiAutoPickState.h,
│   SequentialDeformController.h, AutoDeform.h
└── （新規分離 1）
    MeshDrawingSoftBody.h    (MeshDrawing.h から SoftBody 関連を分離)

registration/src/reg_specific/    変更なし 26 ファイル
```

期待効果:
- `lsn_registration` のコンパイル時に SoftBody.h / CentVoxTetrahedralizerHybrid 等が一切引きずられない
- DEFORM 関連のコードが全部 `deform/src/` 直下に集まる → 一覧性向上、サブディレクトリ階段なし
- **RayCast がメッシュ非依存の真の汎用ライブラリになる**（副次効果として、将来別のメッシュ表現が出てきても同じ API で使える）

---

## 5. Phase 構成

前回計画書の Phase 9 まで完了済みを前提に、本計画書では Phase D-0 〜 D-7 として進める。
（D = "DEFORM separation"）

| Phase | 内容 |
|---|---|
| D-0 | 前提確認（前回計画の完了確認、ブランチ作成、バックアップ） |
| D-1 | `deform/src/deform_specific/` の中身を `deform/src/` 直下にフラット化 |
| D-2 | 11 ファイルを common → deform/src/ へ移動 |
| D-3 | **RayCast.h を汎用化（SoftBody 依存を除去、template 化）** |
| D-4 | MeshDrawing.h 分割（MeshDrawingSoftBody.h を新規作成） |
| D-5 | Sphere.h に TODO コメント追記のみ |
| D-6 | CMakeLists.txt の最終確認 |
| D-7 | ビルド・スモークテスト |

---

## 6. Phase D-0 — 前提確認

### 6.1 状態確認

```bash
cd ${ROOT}
git status
git log --oneline | head -20
ls common/src/ | wc -l    # 期待: 33 + .gitkeep = 34
ls deform/src/deform_specific/ 2>/dev/null | wc -l  # 期待: 3 + .gitkeep = 4
ls registration/src/reg_specific/ | wc -l  # 期待: 26 + .gitkeep = 27
```

**ガード**:
- 前回計画の Phase 9 までのコミット（"Phase 9: smoke tests passed for both apps"）が `git log` に出るか確認
- 出ない場合 → 前回計画を先に完了させる。本計画書は実行しない

### 6.2 ブランチ作成

```bash
git checkout -b refactor/deform-separation-$(date +%Y%m%d)
```

### 6.3 バックアップ

```bash
cd $(dirname ${ROOT})
cp -a AAA_LiverSurgeryNaviComb AAA_LiverSurgeryNaviComb_BACKUP_D_$(date +%Y%m%d_%H%M%S)
```

### 6.4 現状のビルド成功を再確認

```bash
cd ${ROOT}
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_REG=ON -DBUILD_DEFORM=ON 2>&1 | tee cmake.log
cmake --build . -j$(nproc) 2>&1 | tee build.log
ls bin/  # lsn_registration と lsn_deform があるか
```

**ガード**: ビルド失敗 → 本計画書を実行しない

### 6.5 計画書を `${ROOT}/REFACTOR_PLAN_PHASE_D.md` にコピーして git add

```bash
cp <この計画書> ${ROOT}/REFACTOR_PLAN_PHASE_D.md
git add REFACTOR_PLAN_PHASE_D.md
git commit -m "Phase D-0: add deform-separation plan"
```

---

## 7. Phase D-1 — `deform/src/deform_specific/` を `deform/src/` にフラット化

### 7.1 移動

```bash
cd ${ROOT}

shopt -s dotglob nullglob
for f in deform/src/deform_specific/*; do
    fname=$(basename "$f")
    if [ "$fname" = ".gitkeep" ]; then
        git rm "$f" 2>/dev/null || rm "$f"
    else
        git mv "$f" "deform/src/$fname"
    fi
done
shopt -u dotglob nullglob

rmdir deform/src/deform_specific

# overrides も同様（B 群片寄せ後で空のはず）
if [ -d deform/src/overrides ]; then
    contents=$(ls -A deform/src/overrides/ | grep -v '^\.gitkeep$')
    if [ -z "$contents" ]; then
        git rm deform/src/overrides/.gitkeep 2>/dev/null || rm -f deform/src/overrides/.gitkeep
        rmdir deform/src/overrides
    else
        echo "WARN: deform/src/overrides not empty:"
        echo "$contents"
    fi
fi
```

### 7.2 検証

```bash
ls deform/src/
ls deform/src/deform_specific 2>/dev/null  # → 存在しないはず
ls deform/src/overrides 2>/dev/null        # → 存在しない
grep -rn "deform_specific\|src/overrides" deform/CMakeLists.txt
```

### 7.3 CMakeLists.txt 修正

`deform/CMakeLists.txt` を以下に変更:

```cmake
# === 旧 ===
# file(GLOB LSN_DEF_SPECIFIC CONFIGURE_DEPENDS
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_specific/*.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/overrides/*.cpp
# )
# target_include_directories(lsn_deform PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/overrides
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_specific
#     ${LSN_THIRD_PARTY_INCLUDES}
# )

# === 新 ===
file(GLOB LSN_DEF_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
# main.cpp が src/ 直下にある場合は重複避けのため除外:
# list(REMOVE_ITEM LSN_DEF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_include_directories(lsn_deform PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LSN_THIRD_PARTY_INCLUDES}
)
```

> main.cpp の場所を `ls deform/main.cpp deform/src/main.cpp 2>&1` で確認してから `list(REMOVE_ITEM ...)` の要否を決めること。

確認:

```bash
grep -rn "deform_specific\|src/overrides" deform/CMakeLists.txt
```

### 7.4 ビルド確認

```bash
cd ${ROOT}/build
cmake .. 2>&1 | tail -20
cmake --build . -j$(nproc) 2>&1 | tail -20
```

### 7.5 コミット

```bash
git add -A
git commit -m "Phase D-1: flatten deform/src layout (remove deform_specific/ and overrides/)"
```

---

## 8. Phase D-2 — 11 ファイルを common → deform/src/ へ移動

### 8.1 移動コマンド

```bash
cd ${ROOT}

DEFORM_FILES=(
    SoftBody.h
    SoftBody.cpp
    CentVoxTetrahedralizerHybrid.h
    CentVoxTetrahedralizerHybrid.cpp
    TetoMeshData.h
    HandleControllerBase.h
    AutoHandleController.h
    SemiAutoHandleController.h
    SemiAutoPickState.h
    SequentialDeformController.h
    AutoDeform.h
)

for f in "${DEFORM_FILES[@]}"; do
    if [ -f "common/src/$f" ]; then
        git mv "common/src/$f" "deform/src/$f"
        echo "MOVED: $f"
    else
        echo "ERROR: common/src/$f not found"
    fi
done
```

### 8.2 検証

```bash
ls common/src/ | grep -v '^\.gitkeep$' | wc -l    # 33 - 11 = 22
ls deform/src/ | grep -v '^\.gitkeep$' | wc -l    # 既存 + 11

for f in "${DEFORM_FILES[@]}"; do
    [ -f "common/src/$f" ] && echo "WARN: still in common: $f"
done
```

### 8.3 REG 側で「うっかり include していたか」確認

```bash
echo "=== REG 側で削除ファイルを include しているか ==="
for f in "${DEFORM_FILES[@]}"; do
    fname=$(echo "$f" | sed 's/\.[ch]pp\?$//')
    matches=$(grep -rln "include[[:space:]]*[\"<]${fname}\\.h" registration/ 2>/dev/null)
    if [ -n "$matches" ]; then
        echo "FOUND in REG: $f"
        echo "$matches" | sed 's/^/    /'
    fi
done
```

**結果の解釈**:
- 何も出てこなければ完璧
- 出てきた場合 → そのファイルは REG でも include されている
  - 多くの場合は transitive include なので直接 include を消す
  - もしくはそのヘッダは「真の意味で両方使う」と再分類して common に戻す

### 8.4 ビルド試行

```bash
cd ${ROOT}/build
cmake --build . -j$(nproc) 2>&1 | tee build_after_d2.log
```

### 8.5 コミット

```bash
git add -A
git commit -m "Phase D-2: move 11 SoftBody-dependent files from common/src to deform/src"
```

---

## 9. Phase D-3 — RayCast.h の汎用化（SoftBody 依存除去）

### 9.1 設計方針

`RayCast.h` の API を以下のように一本化する:

- `struct RayHit { bool hit; float distance; glm::vec3 position; }`（SoftBody* フィールドなし）
- `template<typename IndexT> static RayHit intersectMesh(const Ray&, const vector<float>&, const vector<IndexT>&)` 一本だけ
- `#include "SoftBody.h"` を削除
- 不要になった `RayHitTri` 構造体は削除（`RayHit` と同じ内容に統一）

`Grabber.h` 側は、SoftBody から頂点配列とインデックス配列を取り出して RayCast に渡すように呼び出しを書き換える。

### 9.2 `common/src/RayCast.h` の編集手順

#### (a) include 削除

```cpp
// 削除
#include "SoftBody.h"
```

#### (b) RayHit / RayHitTri を統合

**旧**:
```cpp
struct RayHit {
    bool hit;
    float distance;
    glm::vec3 position;
    SoftBody* hitObject;   // ← Grabber は使っていない不要フィールド
};

struct RayHitTri {
    bool hit;
    float distance;
    glm::vec3 position;
};
```

**新**:
```cpp
struct RayHit {
    bool      hit;
    float     distance;
    glm::vec3 position;
};
// RayHitTri は削除（RayHit に統一）
```

#### (c) SoftBody 版 intersectMesh を削除

**削除する関数全体**:

```cpp
static RayHit intersectMesh(const Ray& ray, SoftBody& mesh) {
    // ... 30 行ほどの実装 ...
}
```

#### (d) 汎用 intersectMesh を template 化 + 値渡し→const&

**旧**:
```cpp
static RayHitTri intersectMesh(const Ray& ray,
                               std::vector<GLfloat> vertices,
                               std::vector<GLuint>  indices) {
    RayHitTri result = { false, std::numeric_limits<float>::max(), glm::vec3(0) };
    // ...
    for (size_t i = 0; i < surfaceTriIds.size(); i += 3) {
        // ...
        if (rayTriangleIntersect(...)) {
            if (t < result.distance) {
                result.hit = true;
                // ...
                hit_index = i/3;   // ← 副作用（global 書き込み、しかも直後に上書きされる死にコード）
            }
        }
    }
    return result;
}
```

**新**:
```cpp
template<typename IndexT>
static RayHit intersectMesh(const Ray& ray,
                            const std::vector<float>& vertices,
                            const std::vector<IndexT>& indices) {
    RayHit result = { false, std::numeric_limits<float>::max(), glm::vec3(0) };

    float t, u, v;
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        int idx1 = static_cast<int>(indices[i]);
        int idx2 = static_cast<int>(indices[i + 1]);
        int idx3 = static_cast<int>(indices[i + 2]);

        glm::vec3 v1(vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]);
        glm::vec3 v2(vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]);
        glm::vec3 v3(vertices[idx3 * 3], vertices[idx3 * 3 + 1], vertices[idx3 * 3 + 2]);

        if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
            if (t < result.distance) {
                result.hit      = true;
                result.distance = t;
                result.position = ray.origin + ray.direction * t;
                // ← hit_index への書き込みは削除（呼び出し元で行う）
            }
        }
    }
    return result;
}
```

**変更点まとめ**:
- 関数自体を `template<typename IndexT>` 化
- 引数を **値渡し → const&** に変更（コピー削減）
- 戻り値型を `RayHitTri` → `RayHit` に変更（型統一）
- 関数内の `hit_index = i/3` を削除（global 副作用を捨てる）
- vertices 型を `vector<GLfloat>` → `vector<float>` に変更（GLfloat は実質 float、`#include <GL/glew.h>` 依存を減らせる）

#### (e) rayTriangleIntersect を public に昇格

> 旧計画 D-3 では「外部から呼ぶ必要があるから public に」だったが、新計画では template 化したので **public 昇格は不要**。private のままで OK。

ただし、後述の `FindHit*` ヘルパーから呼ぶ場合は public でも構わない。今回は **そのまま現状維持**（private）。

#### (f) FindHit / FindHitWithCamera / FindHitWithCameraMultipleMeshes の戻り値処理修正

これらの関数は旧 `RayHitTri` を受け取り、内部で `hit_index` を書き換えていた。`RayHitTri` → `RayHit` の rename と、`hit_index` 書き込みのタイミング変更（intersectMesh 内部 → FindHit 内部）に対応する。

**FindHit() の修正**:

```cpp
// 旧
RayCast::RayHitTri hit = RayCast::intersectMesh(worldRay, vertices, indices);
// （hit_index は intersectMesh 内部で書かれていた）

// 新
RayCast::RayHit hit = RayCast::intersectMesh(worldRay, vertices, indices);
// hit_index は FindHit が自分で書く必要はなし（後段の nearest-vertex ループが書く）
// もしくは hit.hit==true のときに「最近傍頂点」を計算して hit_index に書く（既存ロジック）
```

> 元の `FindHit()` は intersectMesh の戻り値だけ見て isDragging を更新し、 `hit_index = -1` を no-hit 時に書く程度。intersectMesh 内部の `hit_index = i/3` は直後の nearest-vertex ループでどうせ上書きされるため、消しても外部挙動は変わらない（**重要**: ここを確認してから消すこと）。

**該当箇所の編集**: ファイル全体を grep して `RayHitTri` を `RayHit` に置換。

```bash
# vertices 型の置換も含めて確認
sed -i 's/RayHitTri/RayHit/g' common/src/RayCast.h
grep -n "RayHitTri" common/src/RayCast.h   # 何も出ないはず
grep -n "GLfloat\|GLuint" common/src/RayCast.h   # 残っている箇所をチェック（intersectMesh 引数だけは残らない）
```

#### (g) `RayCast.h` から `#include <GL/glew.h>` が必要か再評価

`intersectMesh` の引数から `GLfloat`/`GLuint` を消したことで、もし他に GL 型を使っていなければ `<GL/glew.h>` の include も削れる。
ただし `FindHit*` ヘルパーが `viewport` で GL 文脈を持っている可能性があるので慎重に。**今回は include 削除は見送り**（リスクに見合わない）。

### 9.3 `deform/src/Grabber.h` の編集手順

`Grabber.h` で `RayCast::intersectMesh(localRay, *physicsObject)` を呼んでいる箇所が 4 つある:

```bash
grep -n "intersectMesh\|RayCast::RayHit\b" deform/src/Grabber.h
```

期待される結果（行番号は環境により異なる）:

```
 53:        RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);
 84:        return RayCast::intersectMesh(localRay, *physicsObject).hit;
102:        RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);
```

#### (a) ヘルパー lambda を追加（DRY 化）

`Grabber.h` の各メソッド先頭に同じ呼び出しが 3 回登場するので、private なヘルパーラムダ or メンバ関数を 1 つ作って共通化する。

`class Grabber` の private セクションに以下を追加（既存 `private:` ブロック内）:

```cpp
private:
    // RayCast に渡すための SoftBody mesh データ取得ヘルパー
    RayCast::RayHit raycastAgainstMesh(const RayCast::Ray& localRay) const {
        return RayCast::intersectMesh(
            localRay,
            physicsObject->getPositions(),
            physicsObject->getMeshData().tetSurfaceTriIds);
    }
```

> SoftBody.h は Grabber.h で既に include 済みなので追加 include 不要。

#### (b) 呼び出し側の置換

**3 箇所を以下のように書き換え**:

```cpp
// 旧 (4 箇所)
RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);

// 新
RayCast::RayHit hit = raycastAgainstMesh(localRay);
```

```cpp
// 旧 (hitTest の return 文)
return RayCast::intersectMesh(localRay, *physicsObject).hit;

// 新
return raycastAgainstMesh(localRay).hit;
```

sed コマンド例:

```bash
# Grabber.h で 4 箇所を機械置換
sed -i 's|RayCast::intersectMesh(localRay, \*physicsObject)|raycastAgainstMesh(localRay)|g' deform/src/Grabber.h

# 結果確認
grep -n "RayCast::intersectMesh\|raycastAgainstMesh" deform/src/Grabber.h
```

#### (c) ヘルパー関数の挿入

class 宣言の末尾、`private:` ブロックの先頭に挿入:

```bash
# 既存の private: 行の直後に挿入
sed -i '/^private:$/a\
\
    // RayCast に渡すための SoftBody mesh データ取得ヘルパー\
    RayCast::RayHit raycastAgainstMesh(const RayCast::Ray\& localRay) const {\
        return RayCast::intersectMesh(\
            localRay,\
            physicsObject->getPositions(),\
            physicsObject->getMeshData().tetSurfaceTriIds);\
    }\
' deform/src/Grabber.h
```

> sed の改行とエスケープが面倒なので、実際は **手動で編集** することを推奨する。手動の場合は class Grabber の private 末尾（`SoftBody* physicsObject;` 等の前）に上記コードブロックを挿入。

#### (d) 確認

```bash
grep -n "RayCast\|SoftBody\|raycastAgainstMesh" deform/src/Grabber.h
```

期待:
- `#include "RayCast.h"`, `#include "SoftBody.h"` はそのまま存在
- `RayCast::intersectMesh(*physicsObject)` 系は **存在しない**
- `raycastAgainstMesh` が 1 個の定義と 4 個の呼び出し（合計 5 行）

### 9.4 ビルド試行

```bash
cd ${ROOT}/build
cmake --build . -j$(nproc) 2>&1 | tee build_after_d3.log
```

DEFORM 側で template 関数のインスタンス化エラーや、`vector<float>` への暗黙変換失敗が起きる可能性があるので注意。

想定されるトラブル:

| エラー | 原因 | 対処 |
|---|---|---|
| `no matching function for call to 'intersectMesh'` | template 推論失敗 | `physicsObject->getPositions()` の戻り値型を確認、`std::vector<float>` でない場合は変換が必要 |
| `tetSurfaceTriIds` の型が `vector<int>` でない | SoftBody::MeshData の定義確認 | 実際の型に合わせて IndexT を明示するか cast |
| REG 側で `RayHitTri` が見つからない | 旧名で参照しているコード | `RayHit` に書き換え |
| `hit_index` が不定になる | intersectMesh から global 書き込みを消したため | FindHit 内部で書いていることを確認、もしバグになるなら復元検討 |

### 9.5 コミット

```bash
git add -A
git commit -m "Phase D-3: generalize RayCast - remove SoftBody dependency via template intersectMesh

- Delete SoftBody-specific RayHit::hitObject and intersectMesh(Ray, SoftBody&)
- Unify RayHitTri into RayHit
- Template-ize intersectMesh<IndexT>(Ray, vec<float>, vec<IndexT>)
- Change value-by-copy to const& for vertices/indices arguments
- Drop side-effect 'hit_index = i/3' from intersectMesh (was dead code)
- Grabber: add raycastAgainstMesh() helper that extracts SoftBody mesh data
- No new file created (cleaner than the previous RayCastSoftBody.h proposal)"
```

---

## 10. Phase D-4 — MeshDrawing.h 分割

### 10.1 該当関数の特定

```bash
grep -n "SoftBody\|softBody" common/src/MeshDrawing.h
```

主に `drawCombinedDepthSorted(SoftBody* softBody, ...)` を含む 1〜2 関数。

### 10.2 新規ファイル作成: `deform/src/MeshDrawingSoftBody.h`

```cpp
#pragma once

// =============================================================================
//  MeshDrawingSoftBody.h
//  ---------------------------------------------------------------------------
//  MeshDrawing.h から SoftBody 関連の描画関数を分離。Phase D-4 で作成。
//
//  共通の mCutMesh 描画関数は common/src/MeshDrawing.h のまま。
//  ここには SoftBody + mCutMesh をミックスして深度ソート描画する関数等を置く。
// =============================================================================

#include <vector>
#include <glm/glm.hpp>

#include "MeshDrawing.h"      // mCutMesh 描画関数（共通）
#include "SoftBody.h"
#include "mCutMesh.h"
#include "ShaderProgram.h"

// （元 MeshDrawing.h から SoftBody* を取る関数群をここに移植）

inline void drawCombinedDepthSorted(
    SoftBody* softBody,
    /* ...残りの引数を元コードからコピー... */) {
    // 元 MeshDrawing.h の line 214〜 をそのままコピーペースト
    // ...
}
```

> 実際の関数本体・引数リストは Phase D-4 実行時に `common/src/MeshDrawing.h` から切り出す。
> bit-identical コピーを心がける（`diff` で確認）。

### 10.3 `common/src/MeshDrawing.h` の編集

該当関数を削除し、`#include "SoftBody.h"` と前方宣言 `class SoftBody;` を削除:

```cpp
// 削除
// class SoftBody;
// #include "SoftBody.h"
```

### 10.4 呼び出し元の修正

```bash
grep -rln "drawCombinedDepthSorted" ${ROOT}/deform/ ${ROOT}/registration/ ${ROOT}/common/
```

DEFORM 側の `DeformPipeline.h` 等に `#include "MeshDrawingSoftBody.h"` を追加。
REG 側で呼ばれていれば、それは間違いなので削除。

### 10.5 コミット

```bash
git add -A
git commit -m "Phase D-4: split MeshDrawing.h - SoftBody parts move to deform/src/MeshDrawingSoftBody.h"
```

---

## 11. Phase D-5 — Sphere.h に TODO コメント追記

### 11.1 編集

`common/src/Sphere.h` の先頭（既存 `#pragma once` の直後）に追記:

```cpp
#pragma once

// =============================================================================
//  Sphere.h — shared by REG and DEFORM
// =============================================================================
//  TODO(future): If REG and DEFORM start needing different sphere behaviour
//  (e.g. labelled correspondence markers for Umeyama vs handle-progress
//  visualisation for AutoDeform), split this into:
//      common/src/Sphere.h                       -- base mesh + draw
//      registration/src/reg_specific/SphereREG.h -- REG-specific extension
//      deform/src/SphereDEF.h                    -- DEFORM-specific extension
//  using inheritance (SphereBase + SphereREG : SphereBase, etc.).
//  Until that need arises, both projects share this class as-is.
// =============================================================================
```

### 11.2 コミット

```bash
git add common/src/Sphere.h
git commit -m "Phase D-5: add inheritance roadmap TODO to Sphere.h"
```

---

## 12. Phase D-6 — CMakeLists.txt の最終確認

### 12.1 `deform/CMakeLists.txt`

Phase D-1 で既にフラット化済み。最終形:

```cmake
file(GLOB LSN_DEF_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
# main.cpp が src/ 直下にあるなら除外:
# list(REMOVE_ITEM LSN_DEF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_executable(lsn_deform
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${LSN_DEF_SOURCES}
    ${IMGUI_SOURCES}
    ${TINYOBJ_SOURCES}
    ${LSN_THIRD_PARTY_SOURCES}
)

target_include_directories(lsn_deform PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LSN_THIRD_PARTY_INCLUDES}
)
```

確認:

```bash
grep -rn "deform_specific\|src/overrides" deform/CMakeLists.txt
```

### 12.2 `registration/CMakeLists.txt`

変更不要。REG 側の include path に `deform/src` が入っていないことを確認:

```bash
grep -n "deform/src" registration/CMakeLists.txt
```

### 12.3 `common/CMakeLists.txt`

```bash
ls common/src/SoftBody.cpp 2>/dev/null && echo "ERROR: still in common"
ls common/src/CentVoxTetrahedralizerHybrid.cpp 2>/dev/null && echo "ERROR: still in common"
```

両方ともファイル無しのはず。

### 12.4 コミット

```bash
git add -A
git commit -m "Phase D-6: final CMakeLists check for flat deform/src layout"
```

---

## 13. Phase D-7 — ビルド・スモークテスト

### 13.1 クリーンビルド

```bash
cd ${ROOT}
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_REG=ON -DBUILD_DEFORM=ON 2>&1 | tee cmake.log
cmake --build . -j$(nproc) 2>&1 | tee build.log
EXIT=$?
echo "BUILD EXIT: $EXIT"
ls -la bin/
```

### 13.2 想定エラーと対処

| エラー | 原因 | 対処 |
|---|---|---|
| `'SoftBody.h' file not found` (DEFORM build) | include path に `deform/src` が無い | Phase D-1.3 / D-6.1 を再確認 |
| `'SoftBody.h' file not found` (REG build) | REG 側のどこかが残留 include | エラー出力の該当ファイルから `#include "SoftBody.h"` を消す |
| `'RayHitTri' was not declared` | 旧名参照が残っている | `grep -rn RayHitTri` で全箇所を `RayHit` に置換 |
| `no matching function for call to 'intersectMesh'` (DEFORM) | template 推論失敗 | Grabber の raycastAgainstMesh ヘルパー定義を確認 |
| `no matching function` (REG, FindHit 系) | template 引数の型不一致 | `vector<GLuint>` を渡している箇所を確認、必要なら明示インスタンス化 |
| `hit_index` の挙動異常 | Phase D-3 で intersectMesh 内部の書き込みを消したため | FindHit / FindHitWithCamera が「最近傍頂点」を計算する箇所で hit_index に書いているはず。そちらが活きているか確認 |
| `multiple definition of drawCombinedDepthSorted` | Phase D-4 で両方に残った | common 側から消したか確認 |

### 13.3 動作確認

```bash
cd ${ROOT}/build/bin
./lsn_registration &
PID=$!
sleep 10
kill -0 $PID 2>/dev/null && echo "REG: OK" || echo "REG: crashed"
kill $PID 2>/dev/null

./lsn_deform &
PID=$!
sleep 10
kill -0 $PID 2>/dev/null && echo "DEFORM: OK" || echo "DEFORM: crashed"
kill $PID 2>/dev/null
```

### 13.4 ユーザ手動確認

```
□ lsn_registration で REG の主要操作（depth、ICP、CMAES、AR モード、AR Save）
□ lsn_registration の右クリック「臓器ヒット判定」が動く（← Phase D-3 で触った FindHit 系）
□ lsn_registration の Umeyama 2 画面モードで対応点ピックが動く
□ lsn_deform で DEFORM の主要操作（R/H/D/C 切替）
□ lsn_deform の **左クリックでハンドル配置**（← Phase D-3 の Grabber::placeSphere）
□ lsn_deform の **左ドラッグで変形**（← Phase D-3 の Grabber::startGrab/moveGrab）
□ depth_output/ 書き出し
```

> 特に Phase D-3 で触った RayCast 系は Grabber と FindHit の **両方で実機動作確認** が必須。
> ビルドが通っても挙動が変わっている可能性がある。

### 13.5 コミット

```bash
cd ${ROOT}
git add -A
git commit --allow-empty -m "Phase D-7: build & smoke test passed after DEFORM separation"
```

---

## 14. 完了条件（Definition of Done）

1. `common/src/` に 22 ファイル前後（A 群残留 14 + B 群残留 7 + α）
2. `deform/src/` 直下に 15 ファイル前後（既存 3 + 移動 11 + 分離新規 1）。**サブディレクトリなし**
3. `registration/src/reg_specific/` は 26 ファイルで変更なし
4. `cmake -DBUILD_REG=ON -DBUILD_DEFORM=ON ..` 成功
5. 両実行ファイル起動成功（10 秒生存）
6. **REG 側のビルドログに `SoftBody.h` `CentVoxTetrahedralizerHybrid.h` などが一切現れない**
7. **`common/src/RayCast.h` から `#include "SoftBody.h"` が消えている**
8. ユーザによる GUI 動作確認パス（特に RayCast 系 = Grabber と FindHit の両方）
9. `Sphere.h` に将来の継承化 TODO コメント追加済み

---

## 15. 全 Phase 共通ルール

### 15.1 コミットメッセージ規約

```
Phase D-N: <一行サマリ>

<必要なら詳細>
```

### 15.2 「迷ったら止まる」原則

- ファイル移動中に「これも DEFORM 側じゃない？」と気づいた場合、**勝手に追加しない**でユーザに確認
- 特に Phase D-3 の RayCast 汎用化で、Grabber 以外で `RayCast::RayHit::hitObject` を参照しているコードを発見したら **即報告**（想定外）
- `physicsObject->getPositions()` や `physicsObject->getMeshData()` の戻り値型が想定と違ったら止まる

### 15.3 ロールバック

各 Phase で問題が出たら、その Phase のコミットだけ `git reset --hard HEAD~1` で戻して原因を直す。
特に Phase D-3 は副作用範囲が広いので、ビルド失敗時は **躊躇なくロールバック** すること。

---

## 16. Claude Code への最終指示

```
あなたは本計画書 (REFACTOR_PLAN_PHASE_D.md) に従って Phase D-0 から順に実行してください。

前提:
  前回の REFACTOR_PLAN_for_ClaudeCode.md による作業が
  Phase 9 まで完了し、両方の実行ファイルがビルド・起動できる状態
  であること。git log で確認できなければ実行しないでください。

ディレクトリ命名:
  本計画では deform/src/d/ などのサブディレクトリは作らず、
  DEFORM 関連ファイルは全部 deform/src/ 直下にフラットに置きます。

Phase D-3 (RayCast 汎用化) について特に注意:
  - 新規ファイル RayCastSoftBody.h は作らない（旧計画から方針変更）
  - 既存 RayCast.h を template 化して SoftBody 依存を取り除く
  - Grabber.h 側で SoftBody から頂点/インデックス配列を取り出すヘルパーを追加
  - hit_index への副作用書き込みを intersectMesh 内部から削除する（死にコードだが念のため挙動確認）
  - RayHitTri を RayHit に統一する rename を伴う（FindHit 系の修正必須）

各 Phase 終了時に必ず:
  1. このドキュメント 15.2 のチェックリスト相当を実施
  2. 結果を私（ユーザ）に簡潔に報告
  3. 次の Phase に進む承認を求める

特に Phase D-3 終了時は:
  - REG の右クリック臓器ピック動作
  - DEFORM の左クリックハンドル配置
  - DEFORM の左ドラッグ変形
  の 3 つを必ずユーザに動作確認してもらってから D-4 に進む

「迷ったら止まる」を厳守してください。
```

---

**END OF PLAN (Phase D)**
