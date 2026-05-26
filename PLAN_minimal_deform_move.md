# common/src/ から DEFORM 系を deform/src/ へ移動する作業指示書

> **対象実行者**: Claude Code (CLI)
> **対象ディレクトリ**: `/home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviComb`
> **構造前提**: 現状のフラット構成（`common/src/`, `registration/src/`, `deform/src/` が各々直下にファイル）と **単一ルート CMakeLists.txt** をそのまま維持
> **目標**: SoftBody/四面体化/ハンドル系の 11 ファイルを `common/src/` から `deform/src/` に移し、`lsn_registration` のビルド単位から DEFORM 系を完全に切り離す

---

## 0. このドキュメントが想定する構造

```
AAA_LiverSurgeryNaviComb/
├── CMakeLists.txt              ← 単一ルート、ここで全部 build
├── common/src/                 ← 共通ファイル（フラット、サブdir なし）
├── registration/src/           ← REG 専用（フラット）
├── deform/src/                 ← DEFORM 専用（フラット、現状 3 ファイル）
│   ├── DeformGlobals.h
│   ├── DeformPipeline.h
│   └── Grabber.h
└── ...
```

**前回計画書 `REFACTOR_PLAN_PHASE_D.md` で言及された `deform/src/deform_specific/` 等のサブディレクトリは存在しない。CMake もサブプロジェクト分割されておらず単一ルート構成。** これに合わせて作業する。

---

## 1. やることの全体像

3 ステップだけ:

| Step | 内容 | 触るファイル数 |
|---|---|---|
| 1 | `common/src/RayCast.h` から SoftBody 依存を除去 + `Grabber.h` の呼び出し書き換え | 2 |
| 2 | `common/src/MeshDrawing.h` から SoftBody 関連の描画関数を `deform/src/MeshDrawingSoftBody.h` に切り出し | 2〜3 |
| 3 | 11 ファイルを `common/src/` から `deform/src/` に `git mv` | 11 |

**順序が重要**: Step 1 → Step 2 → Step 3 の順で必ず実行する。逆順だと REG 側ビルドが壊れる中間状態が発生する。

各 Step 終了時に **必ずビルド成功を確認** してから次へ進む。

---

## 2. 事前準備

### 2.1 状態確認

```bash
cd /home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviComb

git status
git log --oneline -5
```

**ガード**:
- uncommit な変更があれば、まず内容を確認してコミット or stash すること
- 現在 main ブランチで作業する想定だが、安全のため新ブランチを切る:

```bash
git checkout -b refactor/deform-separation
```

### 2.2 バックアップ

```bash
cd ..
cp -a AAA_LiverSurgeryNaviComb AAA_LiverSurgeryNaviComb_BACKUP_$(date +%Y%m%d_%H%M%S)
cd AAA_LiverSurgeryNaviComb
```

### 2.3 現状ビルドが通ることを確認

```bash
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
cmake --build . -j$(nproc) 2>&1 | tail -20
ls bin/
cd ..
```

**ガード**: `bin/lsn_registration` と `bin/lsn_deform` 両方が生成されていなければ作業中止し、状況を報告。

---

## 3. Step 1 — RayCast.h の SoftBody 依存除去

### 3.1 背景

`common/src/RayCast.h` には現状 2 系統の `intersectMesh` 関数がある:

1. `intersectMesh(Ray, SoftBody&)` → SoftBody 専用、`Grabber.h` のみが使用
2. `intersectMesh(Ray, vector<GLfloat>, vector<GLuint>)` → 汎用、`FindHit*` ヘルパーが使用

(1) を消して (2) に統一すれば SoftBody.h への依存が消える。`Grabber.h` 側は SoftBody から `getPositions()` と `getMeshData().tetSurfaceTriIds` を取り出して (2) に渡すように変更する（4 行追加されるだけ）。

加えて、`RayHit` 構造体に SoftBody* メンバがあるが、これは `Grabber.h` で **一度も使われていない死にコード**。`RayHitTri` と統合して `SoftBody*` メンバを削除する。

### 3.2 編集後の `common/src/RayCast.h` の該当箇所

#### (a) include を削除

```diff
- #include "SoftBody.h"
  #include "mCutMesh.h"
  #include "FullSphereCameraWithTarget.h"
```

#### (b) RayHit 構造体から SoftBody* メンバを削除し、RayHitTri を削除

```diff
      struct RayHit {
          bool hit;
          float distance;
          glm::vec3 position;
-         SoftBody* hitObject;
      };
- 
-     struct RayHitTri {
-         bool hit;
-         float distance;
-         glm::vec3 position;
-     };
```

#### (c) SoftBody 版 intersectMesh を完全削除

```diff
-     static RayHit intersectMesh(const Ray& ray, SoftBody& mesh) {
-         RayHit result = { false, std::numeric_limits<float>::max(), glm::vec3(0), nullptr };
- 
-         const auto& positions = mesh.getPositions();
-         const auto& surfaceTriIds = mesh.getMeshData().tetSurfaceTriIds;
- 
-         std::cout << "Ray origin: " << ray.origin.x << ", " << ray.origin.y << ", " << ray.origin.z << std::endl;
-         std::cout << "Ray direction: " << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << std::endl;
- 
-         float t, u, v;
-         for (size_t i = 0; i < surfaceTriIds.size(); i += 3) {
-             int idx1 = surfaceTriIds[i];
-             int idx2 = surfaceTriIds[i + 1];
-             int idx3 = surfaceTriIds[i + 2];
- 
-             glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
-             glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
-             glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);
- 
-             if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
-                 if (t < result.distance) {
-                     result.hit = true;
-                     result.distance = t;
-                     result.position = ray.origin + ray.direction * t;
-                     result.hitObject = &mesh;
-                     std::cout << "Hit at triangle " << i/3 << std::endl;
-                     std::cout << "Hit position: " << result.position.x << ", "
-                               << result.position.y << ", " << result.position.z << std::endl;
-                 }
-             }
-         }
- 
-         return result;
-     }
```

#### (d) 汎用 intersectMesh の戻り値型を `RayHitTri` → `RayHit` に変更、引数を `const&` 化

```diff
-     static RayHitTri intersectMesh(const Ray& ray, std::vector<GLfloat> vertices, std::vector<GLuint> indices) {
-         RayHitTri result = { false, std::numeric_limits<float>::max(), glm::vec3(0)};
+     static RayHit intersectMesh(const Ray& ray, const std::vector<GLfloat>& vertices, const std::vector<GLuint>& indices) {
+         RayHit result = { false, std::numeric_limits<float>::max(), glm::vec3(0) };
```

> 注意: 関数内の `hit_index = i/3;` の行はそのまま残す。これは `FindHit*` のヘルパーが期待している副作用なので、消すと REG 側の挙動が変わる可能性がある（後続の最近傍頂点ループで上書きされる気もするが、念のため触らない）。

#### (e) `FindHit` / `FindHitWithCamera` / `FindHitWithCameraMultipleMeshes` 内の `RayHitTri` を `RayHit` に置換

```diff
-     RayCast::RayHitTri hit = RayCast::intersectMesh(worldRay, vertices, indices);
+     RayCast::RayHit hit = RayCast::intersectMesh(worldRay, vertices, indices);
```

該当箇所は 3 か所 (line 222, 263, 323 付近)。`sed` で一括置換可能:

```bash
sed -i 's/RayCast::RayHitTri/RayCast::RayHit/g' common/src/RayCast.h
sed -i 's/\bRayHitTri\b/RayHit/g' common/src/RayCast.h

# 確認: RayHitTri が完全に消えているか
grep -n "RayHitTri" common/src/RayCast.h    # 何も出ないのが正解
```

### 3.3 編集後の `deform/src/Grabber.h` の該当箇所

`Grabber.h` で `RayCast::intersectMesh(localRay, *physicsObject)` を呼んでいる 3 か所 (line 53, 84, 102 付近) を書き換える。SoftBody は `Grabber.h` で既に include 済みなので追加 include 不要。

#### (a) `Grabber` クラスの private セクションに、ヘルパー関数を 1 つ追加

`class Grabber` の `private:` ブロック内に以下を挿入（既存の `SoftBody* physicsObject;` 等のメンバ宣言の **前** に置くと自然）:

```cpp
private:
    // SoftBody から頂点・三角形インデックスを取り出して RayCast に渡すヘルパー
    RayCast::RayHit raycastAgainstMesh(const RayCast::Ray& localRay) const {
        // SoftBody::getPositions() は const std::vector<float>& を返す
        // SoftBody::getMeshData().tetSurfaceTriIds は const std::vector<int>&
        // RayCast::intersectMesh は vector<GLfloat>, vector<GLuint> を期待するので変換が必要
        const auto& posF  = physicsObject->getPositions();           // vector<float>
        const auto& triI  = physicsObject->getMeshData().tetSurfaceTriIds; // vector<int>

        // GLfloat == float なのでそのまま渡せる（暗黙変換可）
        // GLuint vs int は型不一致なので vector を作り直す必要がある
        std::vector<GLuint> triU(triI.begin(), triI.end());
        return RayCast::intersectMesh(localRay, posF, triU);
    }

    SoftBody* physicsObject;  // ← 既存メンバ（位置はそのまま）
    // ...
```

> **重要**: もし `SoftBody::getPositions()` の戻り値型が `vector<float>` ではなく `vector<GLfloat>` だったり、`tetSurfaceTriIds` の要素型が `int` ではなく `GLuint` だったりしたら、上記の変換は不要 or 別の変換が必要。**実物を `grep -n "getPositions\|tetSurfaceTriIds" deform/src/SoftBody.h` で確認してから書く** こと。型不一致でビルドエラーが出たら、ヘルパー内部の変換を実態に合わせて調整する。

#### (b) 3 か所の呼び出しを書き換え

```diff
      void placeSphere(float screenX, float screenY, float groupRadius = 1.0f) {
          // ...localRay 計算...
-         RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);
+         RayCast::RayHit hit = raycastAgainstMesh(localRay);
          if (hit.hit) {
              // hit.position だけ使う。hit.hitObject は使っていない。
```

```diff
      bool hitTest(float screenX, float screenY) {
          // ...localRay 計算...
-         return RayCast::intersectMesh(localRay, *physicsObject).hit;
+         return raycastAgainstMesh(localRay).hit;
      }
```

```diff
      void startGrab(float screenX, float screenY) {
          // ...localRay 計算...
-         RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);
+         RayCast::RayHit hit = raycastAgainstMesh(localRay);
          if (hit.hit) {
```

### 3.4 Step 1 のビルド確認

```bash
cd build
cmake --build . -j$(nproc) 2>&1 | tee build_step1.log
echo "EXIT: $?"
ls -la bin/
cd ..
```

**期待**:
- `lsn_registration` と `lsn_deform` の両方がビルド成功
- 警告は OK、エラーゼロ

**想定されるエラーと対処**:

| エラー | 原因 | 対処 |
|---|---|---|
| `'SoftBody' was not declared in this scope` (RayCast.h 内) | (c) の削除漏れ | RayCast.h を再確認、SoftBody を参照している行が残っていないか grep |
| `RayHitTri` 未定義 (どこか別ファイル) | REG 側で `RayHitTri` を直接使っている箇所がある | `grep -rn "RayHitTri" .` で発見し、`RayHit` に置換 |
| `no matching function for call to 'intersectMesh'` (Grabber.h) | 型変換ミス | 3.3 (a) のヘルパー内部の変換型を実態に合わせて調整 |
| `cannot convert std::vector<int> to std::vector<GLuint>` | (a) の `std::vector<GLuint> triU(triI.begin(), triI.end())` のコピー失敗 | これは普通に通るはず。出るならコンパイラ警告レベルを確認 |

### 3.5 Step 1 の GUI 動作確認（重要）

ビルドが通ったら **必ず実機で動作確認**。RayCast を触ったので影響範囲が広い:

1. `./bin/lsn_registration` を起動
   - **右クリックで臓器ヒット判定** が動くか（liver/portal/vein/tumor/segment/gb のどれかを右クリックして hit するか）
   - Umeyama 2 画面モードで対応点ピックができるか
2. `./bin/lsn_deform` を起動
   - 左クリックで **ハンドル配置** ができるか
   - 左ドラッグで **変形** できるか

> **このセクションだけは Claude Code では完結しない**。ユーザに実機確認を依頼し、報告を待つこと。

### 3.6 Step 1 のコミット

```bash
git add -A
git commit -m "Step 1: decouple RayCast.h from SoftBody

- Remove #include \"SoftBody.h\" from common/src/RayCast.h
- Delete intersectMesh(Ray, SoftBody&) overload (only Grabber used it)
- Drop unused SoftBody* hitObject field from RayHit struct
- Unify RayHitTri into RayHit (identical content, just different name)
- Change intersectMesh argument from value-copy to const&
- Grabber.h: add raycastAgainstMesh() helper that extracts mesh data
  from SoftBody and forwards to the generic intersectMesh"
```

---

## 4. Step 2 — MeshDrawing.h からの SoftBody 関連描画関数の分離

### 4.1 背景

`common/src/MeshDrawing.h` には `draw_AllVisMeshesWithExtraMesh(SoftBody*, ...)` という関数があり、これが SoftBody の内部メンバ (`visVAOs`, `vis_positions_array`, `visSurfaceTriIds_array`, `getModelMatrix()`) を直接さわっている。

これは generic 化が困難（SoftBody 固有のメンバアクセスが多い）なので、**新規ファイル `deform/src/MeshDrawingSoftBody.h` を作って、そこに関数だけ移動** する方針。

### 4.2 新規ファイル作成: `deform/src/MeshDrawingSoftBody.h`

`common/src/MeshDrawing.h` の line 214〜364 (関数 `draw_AllVisMeshesWithExtraMesh` の全体) を、bit-identical で新規ファイルにコピー:

```cpp
#ifndef MESH_DRAWING_SOFTBODY_H
#define MESH_DRAWING_SOFTBODY_H

// =============================================================================
//  MeshDrawingSoftBody.h
//  ---------------------------------------------------------------------------
//  MeshDrawing.h から SoftBody 関連の描画関数を分離したもの。
//  SoftBody の内部メンバ (visVAOs, vis_positions_array, visSurfaceTriIds_array,
//  getModelMatrix() など) を直接アクセスする関数なので、common には置けない。
//
//  共通の mCutMesh 描画関数は common/src/MeshDrawing.h のまま。
// =============================================================================

#include <vector>
#include <algorithm>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include "ShaderProgram.h"
#include "mCutMesh.h"
#include "SoftBody.h"
#include "MeshDrawing.h"   // 共通の描画関数を再エクスポート（呼び出し側の include を 1 行で済ます）

// SoftBody + extra mCutMesh combined drawing with depth sorting
inline void draw_AllVisMeshesWithExtraMesh(
    SoftBody* softBody,
    ShaderProgram& shaderBasic,
    ShaderProgram& shaderTexture,
    mCutMesh* extraMesh,
    glm::vec3 camPos,
    std::vector<glm::vec4>& meshColors,
    glm::vec4 extraMeshColor,
    const glm::mat4& model,
    const glm::mat4& view,
    const glm::mat4& projection) {

    // ↓ ここに common/src/MeshDrawing.h の line 226〜364 をコピペ ↓
    // （関数本体まるごと。中身は一切変更しない）
    
    struct GlobalTriangleInfo {
        bool isExtraMesh;
        size_t meshIndex;
        size_t triangleIndex;
        float distance;
        GLuint vao;
    };

    std::vector<GlobalTriangleInfo> allTriangles;

    for (size_t meshIdx = 0; meshIdx < softBody->visVAOs.size(); meshIdx++) {
        const auto& positions = softBody->vis_positions_array[meshIdx];
        const auto& indices = softBody->visSurfaceTriIds_array[meshIdx];

        for (size_t j = 0; j < indices.size(); j += 3) {
            int idx1 = indices[j];
            int idx2 = indices[j + 1];
            int idx3 = indices[j + 2];

            glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
            glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
            glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);

            glm::vec3 center = (v1 + v2 + v3) / 3.0f;
            glm::vec4 worldCenter = softBody->getModelMatrix() * glm::vec4(center, 1.0f);
            float distance = glm::length(camPos - glm::vec3(worldCenter));

            allTriangles.push_back({
                false,
                meshIdx,
                j / 3,
                distance,
                softBody->visVAOs[meshIdx]
            });
        }
    }

    if (extraMesh != nullptr) {
        const auto& positions = extraMesh->mVertices;
        const auto& indices = extraMesh->mIndices;

        for (size_t j = 0; j < indices.size(); j += 3) {
            GLuint idx1 = indices[j];
            GLuint idx2 = indices[j + 1];
            GLuint idx3 = indices[j + 2];

            glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
            glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
            glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);

            glm::vec3 center = (v1 + v2 + v3) / 3.0f;
            glm::vec4 worldCenter = model * glm::vec4(center, 1.0f);
            float distance = glm::length(camPos - glm::vec3(worldCenter));

            allTriangles.push_back({
                true,
                0,
                j / 3,
                distance,
                extraMesh->VAO
            });
        }
    }

    std::sort(allTriangles.begin(), allTriangles.end(),
              [](const GlobalTriangleInfo& a, const GlobalTriangleInfo& b) {
                  return a.distance > b.distance;
              });

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    GLuint lastVAO = 0;
    glm::vec4 lastColor(-1.0f);
    int lastShaderType = -1;

    for (const auto& tri : allTriangles) {
        int currentShaderType = tri.isExtraMesh ? 1 : 0;

        if (lastShaderType != currentShaderType) {
            if (currentShaderType == 1) {
                shaderTexture.use();
                shaderTexture.setUniform("model", model);
                shaderTexture.setUniform("view", view);
                shaderTexture.setUniform("projection", projection);
                shaderTexture.setUniform("lightPos", camPos);
                shaderTexture.setUniform("viewPos", camPos);
                shaderTexture.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
                shaderTexture.setUniform("useTexture", true);

                if (extraMesh && extraMesh->hasTexture) {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, extraMesh->textureID);
                    shaderTexture.setUniform("texture1", 0);
                }
            } else {
                shaderBasic.use();
                shaderBasic.setUniform("model", softBody->getModelMatrix());
                shaderBasic.setUniform("view", view);
                shaderBasic.setUniform("projection", projection);
                shaderBasic.setUniform("lightPos", camPos);
                shaderBasic.setUniform("viewPos", camPos);
                shaderBasic.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
                glBindTexture(GL_TEXTURE_2D, 0);
            }

            lastShaderType = currentShaderType;
            lastColor = glm::vec4(-1.0f);
        }

        if (lastVAO != tri.vao) {
            glBindVertexArray(tri.vao);
            lastVAO = tri.vao;
        }

        glm::vec4 currentColor = tri.isExtraMesh
                                     ? extraMeshColor
                                     : meshColors[tri.meshIndex % meshColors.size()];

        if (lastColor != currentColor) {
            if (currentShaderType == 1) {
                shaderTexture.setUniform("vertColor", currentColor);
            } else {
                shaderBasic.setUniform("vertColor", currentColor);
            }
            lastColor = currentColor;
        }

        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT,
                       (void*)(tri.triangleIndex * 3 * sizeof(GLuint)));
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

#endif // MESH_DRAWING_SOFTBODY_H
```

### 4.3 `common/src/MeshDrawing.h` から該当部分を削除

```diff
  #include "ShaderProgram.h"
  #include "mCutMesh.h"
- #include "SoftBody.h"
- 
- // Forward declarations
- class SoftBody;
```

そして line 214〜364 の `draw_AllVisMeshesWithExtraMesh(SoftBody*, ...)` 関数まるごと削除。

### 4.4 呼び出し元の修正

`draw_AllVisMeshesWithExtraMesh` を呼んでいる箇所を探す:

```bash
grep -rln "draw_AllVisMeshesWithExtraMesh" .
```

結果として出てくるのは多分 DEFORM 側のファイル（`deform/src/DeformPipeline.h`, `deform/src/main.cpp`, あるいは `deform/src/AutoDeform.h` 等）。それぞれで:

```diff
  #include "MeshDrawing.h"
+ #include "MeshDrawingSoftBody.h"
```

を追加する。

> **注意**: もし REG 側で呼ばれていたら、それは設計上のバグ。報告すること（おそらく無いはず）。

### 4.5 Step 2 のビルド確認

```bash
cd build
cmake --build . -j$(nproc) 2>&1 | tee build_step2.log
echo "EXIT: $?"
cd ..
```

**期待**:
- 両ターゲットビルド成功

**想定エラー**:

| エラー | 原因 | 対処 |
|---|---|---|
| `'SoftBody' was not declared` in MeshDrawing.h | 削除漏れ | MeshDrawing.h を再確認、SoftBody 参照が完全に消えているか grep |
| `draw_AllVisMeshesWithExtraMesh not declared` in 呼び出し元 | include 追加忘れ | 4.4 を再確認 |
| 二重定義 | 同じ関数が両ヘッダにある | common/src/MeshDrawing.h から削除したか確認 |

### 4.6 Step 2 のコミット

```bash
git add -A
git commit -m "Step 2: split MeshDrawing.h - move SoftBody-dependent function to deform/src/MeshDrawingSoftBody.h

- Move draw_AllVisMeshesWithExtraMesh(SoftBody*, ...) out of common/src/MeshDrawing.h
- New file: deform/src/MeshDrawingSoftBody.h (includes MeshDrawing.h for shared funcs)
- common/src/MeshDrawing.h no longer #includes SoftBody.h
- DEFORM callers now include MeshDrawingSoftBody.h"
```

---

## 5. Step 3 — 11 ファイルを common → deform に移動

### 5.1 移動コマンド

```bash
cd /home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviComb

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

### 5.2 検証

```bash
# common/src から 11 ファイルが消えているか
for f in "${DEFORM_FILES[@]}"; do
    [ -f "common/src/$f" ] && echo "STILL IN COMMON: $f"
done
# 何も出ないのが正解

# deform/src に 11 ファイル + 既存 3 + Step 2 で作った MeshDrawingSoftBody.h = 15 ファイル
ls deform/src/

# REG 側で削除ファイルを include していないか念のため確認
for f in "${DEFORM_FILES[@]}"; do
    base=$(echo "$f" | sed 's/\.[ch]pp\?$//')
    matches=$(grep -rln "include[[:space:]]*[\"<]${base}\.h" registration/ 2>/dev/null)
    if [ -n "$matches" ]; then
        echo "WARN: REG includes $f via:"
        echo "$matches" | sed 's/^/    /'
    fi
done
# 何も出ないのが正解
```

### 5.3 CMakeLists.txt の確認

ルート CMakeLists.txt が `file(GLOB ...)` を使っているなら、ファイル移動だけで CMake 側は自動的に追随する。ただし `CONFIGURE_DEPENDS` 指定がない場合は **cmake 再実行が必要**:

```bash
cd build
cmake .. 2>&1 | tail -5    # CMake を再走らせて新しいファイル構成を認識させる
cd ..
```

ルート CMakeLists.txt の deform 関連 glob パターンを確認:

```bash
grep -n "deform/src\|common/src" CMakeLists.txt
```

期待される記述例 (実態に合わせて読み替え):

```cmake
file(GLOB COMMON_SOURCES common/src/*.cpp common/src/*.h)
file(GLOB DEFORM_SOURCES deform/src/*.cpp deform/src/*.h)
file(GLOB REG_SOURCES   registration/src/*.cpp registration/src/*.h)
```

特別な修正は不要のはずだが、もし個別ファイル名を列挙している場合 (e.g. `set(COMMON_SOURCES common/src/SoftBody.cpp ...)`) は手動で更新が必要。

### 5.4 ビルド確認

```bash
cd build
rm -rf CMakeCache.txt CMakeFiles    # CONFIGURE_DEPENDS が無い場合に備えて完全再構成
cmake .. 2>&1 | tail -10
cmake --build . -j$(nproc) 2>&1 | tee build_step3.log
echo "EXIT: $?"
ls -la bin/
cd ..
```

**期待**: `lsn_registration` と `lsn_deform` の両方が無事ビルドできる。

**想定エラー**:

| エラー | 原因 | 対処 |
|---|---|---|
| `'SoftBody.h' file not found` (DEFORM build) | deform/src/ が include path に入っていない | CMake の `target_include_directories` か `include_directories` で `deform/src` を `lsn_deform` の include path に追加 |
| `'SoftBody.h' file not found` (REG build) | REG 側で SoftBody.h を残留 include しているファイルがある | エラーメッセージのファイル名を確認、`#include "SoftBody.h"` を消す。Step 1〜2 で処理しきれなかったケース |
| `undefined reference to SoftBody::...` (REG リンク時) | REG ターゲットが SoftBody.cpp をまだリンク対象にしている | CMake で `lsn_registration` の sources から SoftBody.cpp を除外する必要あり。`file(GLOB)` を使っているなら common/src/*.cpp の glob が SoftBody.cpp を含んでしまうケースは無いはずだが、REG ターゲットに `deform/src/*.cpp` を加えていないか確認 |

### 5.5 Step 3 のコミット

```bash
git add -A
git commit -m "Step 3: move 11 SoftBody-dependent files from common/src to deform/src

Moved files:
  SoftBody.h, SoftBody.cpp
  CentVoxTetrahedralizerHybrid.h, CentVoxTetrahedralizerHybrid.cpp
  TetoMeshData.h
  HandleControllerBase.h
  AutoHandleController.h
  SemiAutoHandleController.h
  SemiAutoPickState.h
  SequentialDeformController.h
  AutoDeform.h

After this commit:
- lsn_registration build no longer pulls in any DEFORM-specific headers
- All DEFORM-specific code lives under deform/src/"
```

---

## 6. 最終検証（Step 3 完了後）

### 6.1 ビルドログから SoftBody 系が REG 側に現れないことを確認

```bash
cd build
rm -rf CMakeFiles bin
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
cmake --build . --target lsn_registration -j$(nproc) 2>&1 | grep -i "softbody\|tetrahedraliz\|HandleController\|AutoDeform\|SemiAutoPick"
# 何も出力されないのが正解（REG はもう DEFORM 系ヘッダを触っていない）
cd ..
```

### 6.2 動作確認チェックリスト（ユーザ依頼）

```
□ lsn_registration:
  □ 起動できる
  □ 画像ドラッグ&ドロップで読み込める
  □ デプス推論が動く
  □ ICP が動く
  □ CMAES Refine が動く
  □ Umeyama 2 画面モードで対応点クリックが動く（← Step 1 で触った）
  □ 右クリック臓器ピックが動く（← Step 1 で触った）
  □ AR モード切替 (Key A) が動く
  □ AR Save (Key D) が動く

□ lsn_deform:
  □ 起動できる
  □ reg_*.obj 読み込みができる
  □ Key R / H / D / C モード切替
  □ 左クリックでハンドル配置（← Step 1 で触った）
  □ 左ドラッグで変形（← Step 1 で触った）
  □ AR モード切替
  □ メッシュ描画が正しく見える（← Step 2 で触った）
```

### 6.3 完了コミット

```bash
git commit --allow-empty -m "Complete: DEFORM separation finished - REG no longer depends on SoftBody/Tetrahedralizer/HandleController"
git log --oneline | head -10
```

---

## 7. 完了条件（Definition of Done）

1. `common/src/` から 11 ファイルが消えている
2. `deform/src/` に 11 ファイル + 既存 3 + MeshDrawingSoftBody.h = 15 ファイル
3. `common/src/RayCast.h` に `SoftBody.h` 参照が無い
4. `common/src/MeshDrawing.h` に `SoftBody.h` 参照が無い
5. `cmake --build .` で `lsn_registration` と `lsn_deform` の両方が成功
6. REG 側のビルドログで SoftBody/Tetrahedralizer/HandleController 等が一切現れない
7. 両アプリの主要機能が動く（GUI 確認）

---

## 8. 失敗時のロールバック

各 Step 後にビルド失敗した場合:

```bash
# その Step のコミットだけ取り消す
git reset --hard HEAD~1

# それでも復旧しない場合、ブランチごと捨ててバックアップから復元
git checkout main
git branch -D refactor/deform-separation
cd ..
rm -rf AAA_LiverSurgeryNaviComb
cp -a AAA_LiverSurgeryNaviComb_BACKUP_XXXXXXXX_XXXXXX AAA_LiverSurgeryNaviComb
```

---

## 9. Claude Code への最終指示

```
この指示書に従って Step 1 → Step 2 → Step 3 の順に実行してください。

順序厳守:
  Step 1 (RayCast の SoftBody 依存除去) を最初にやらないと、
  Step 3 でファイル移動した瞬間に REG ビルドが壊れます。

各 Step 終了時に必ず:
  1. ビルド成功を build_stepN.log で確認
  2. 結果を私（ユーザ）に簡潔に報告
  3. 次の Step に進む承認を求める

Step 1 完了時は特に:
  GUI 動作確認（REG の右クリック臓器ピック、DEFORM の左クリック
  ハンドル配置・左ドラッグ変形）を私にお願いしてから Step 2 へ進む。

迷ったら止まる:
  - SoftBody のメンバ型（getPositions の戻り値型、tetSurfaceTriIds
    の要素型）が想定と違う場合、勝手に推測せず確認してから書く
  - REG 側で SoftBody.h を include しているファイルが見つかった場合、
    その include を消すべきか、それとも「実は両方使う」のかを確認する
  - その他、Plan の想定と現物が食い違う場合は手を止めて報告
```

---

**END OF PLAN**
