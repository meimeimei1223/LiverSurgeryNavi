# キー整理 実装計画書 (Claude Code 向け)

**プロジェクト**: AAA_LiverSurgeryNaviComb (`lsn_registration` ターゲット)
**ブランチ案**: `key-reorg` (`ui-refactor` の続きとして派生推奨)
**対象ファイル**: 主に `registration/main.cpp` + `common/src/DebugPanel.h`
**作成日**: 2026-05-25

---

## 1. ゴール

1. **散在している BIPOP-CMA-ES 系のキーを G ファミリーに統合**
   (Shift+V/Shift+F → Alt+G/Alt+Shift+G に移動、5 メソッドすべて G 文字に集約)
2. **viz トグル系のキー (10 個) を全廃して Ctrl+D デバッグパネルのチェックボックスへ移行**
3. **ラベル再計算系のキー (3 個) を Ctrl+D 内のボタンに移行**
4. **既存の dispatch ロジック (label auto-trigger / poseAutoSave / Live セッション停止) は一切変えない** — 中身はキープし、入口だけ整理する

## 2. 設計方針

- **既存の `ui-refactor` ブランチで導入された "hook パターン" を踏襲する**:
  `g_debugPanel.drawVizExtra = [&]() { ... }` のように main.cpp のフレームループ内
  でラムダを登録して、main.cpp のローカル/グローバルにアクセスする
- **byte-identical 動作の保証**: キー dispatch から呼ばれるアクション関数
  (`runBipopCmaes`, `runShiftE` 等) は中身ゼロ変更。switch case 内のロジック
  (`g_stepStartTime`, `g_sessionBipopN`, `poseAutoSaveBeforeRegistration`,
  `poseSaveToLibrary` の順序) もコピーで保つ
- **commit 単位 = 1 フェーズ**: 各フェーズで `lsn_registration` がコンパイル通る
  ことを保証する
- **Phase 0 で必ず "現状アンカー" のテキスト dump を取る** (Phase 8 の比較用)

---

## 3. 現状のキー全マップ (Phase 0 のリファレンス)

`registration/main.cpp` `glfw_onKey` (line 1365–) の **完全な現状**:

| キー | mod | 行 | 動作 | 種類 |
|---|---|---|---|---|
| `ESC` | - | 1408 | window close | action |
| `O` | - | 1411 | `runHemiAuto` (OrbitCam視点) | **action ★keep** |
| `O` | Shift | 1412 | `runQuadAuto` (AR固定視点) | **action ★keep** |
| `V` | - | 1455 | `g_showClusterVisualization` toggle | **viz toggle → checkbox** |
| `V` | Shift | 1446 | `runBipopCmaes` (V1) | **action → Alt+G に移動** |
| `F` | Shift | 1468 | `runBipopCmaesV2` (V2) | **action → Alt+Shift+G に移動** |
| `G` | Shift | 1587 | `runBipopCmaesV3` (V3) | **action ★keep** |
| `G` | Ctrl | 1536 | `runBipopCmaesV3R` (V3-R) | **action ★keep** |
| `G` | Ctrl+Shift | 1494 | `runBipopCmaesV3RS` (V3-RS) | **action ★keep** |
| `B` | - | 1613 | `g_showBoundaryCandidates` toggle | **viz toggle → checkbox** |
| `B` | Shift | 1603 | `g_showCyclicCorrespondence` toggle | **viz toggle → checkbox** |
| `N` | - | 1705 | `g_showSourceVisualization` toggle | **viz toggle → checkbox** |
| `N` | Shift | 1663 | `startNormalCompatRefineLive(NORMAL_COMPAT)` | **action ★keep** |
| `N` | Ctrl+Shift | 1623 | `startNormalCompatRefineLive(SRT_VARIANCE)` | **action ★keep** |
| `W` | - | 2263 | `g_showDebugSourceRimChain` toggle (+ auto label recompute) | **viz toggle → checkbox** |
| `W` | Shift | 2238 | `g_showDebugTargetBoundary` toggle | **viz toggle → checkbox** |
| `W` | Ctrl | 2145 | Shape Match coarse → apply → save | **action ★keep** |
| `W` | Ctrl+Shift | 1856 | Shape Match + axis sweep + Live ICP bridge | **action ★keep** |
| `W` | Alt | 1768 | Shape Match Coarse2D + GN refine | **action ★keep** |
| `W` | Ctrl+Alt | 1727 | startContourSweep / startSilhouetteSweep | **action ★keep** |
| `P` | - | 2407 | `runSilhouetteHemi` | **action ★keep** |
| `P` | Shift | 2392 | `runCyclicBoundaryReg` | **action ★keep** |
| `P` | Ctrl | 2368 | `runQuadCyclic` | **action ★keep** |
| `P` | Ctrl+Shift | 2340 | `runQuadCyclicRansac` | **action ★keep** |
| `P` | Ctrl+Alt | 2312 | `runAutoQuadCyclicRansac` (AutoQCR) | **action ★keep** |
| `E` | Shift | 2419 | `runShiftE` (Silhouette Align) | **action ★keep** |
| `I` | Shift | 2435 | `IoUDebug::dump` | **debug → button** |
| `Q` | - | 2447 | `g_poseLibrary.showWindow` toggle | window toggle ★keep |
| `X` | - | 2453 | `poseUndo` | **action ★keep** |
| `F2` | - | 2459 | Camera reset | **action ★keep** |
| `A` | - | 2468 | `gApp.arMode` toggle | toggle ★keep |
| `D` | - | 2473 | `ARSave::capture` | **action ★keep** |
| `D` | Ctrl | 1401 | `g_debugPanel.showWindow` toggle | window toggle ★keep |
| `F9` | - | 2490 | `SilOverlay::g_silOverlay.showWindow` toggle | window toggle ★keep |
| `F10` | - | 2502 | `diagnoseVertexSquashV3RS` | **debug → button (キーも残す)** |
| `,` | - / Shift | 2516 | `g_silhouetteCosThreshold -= step` | param adjust ★keep |
| `.` | - / Shift | 2523 | `g_silhouetteCosThreshold += step` | param adjust ★keep |
| `R` | - | 2547 | `runDepthAndUpdateScene` (image mode) | **action ★keep** |
| `R` | Shift | 2531 | `g_showLiverRegion` toggle (+ auto recompute) | **viz toggle → checkbox** |
| `T` | Shift | 2560 | `recomputeLiverRegion` | **action → button** |
| `Y` | - | 2577 | `g_showLiverLR` toggle (+ auto recompute) | **viz toggle → checkbox** |
| `Y` | Shift | 2570 | `recomputeLiverLR` | **action → button** |
| `H` | - | 2621 | `g_showLiverQuad` toggle | **viz toggle → checkbox** |
| `H` | Shift | 2595 | `g_showLiverCC` toggle | **viz toggle → checkbox** |
| `U` | - | 2636 | `MaskPicker::undo` (画像モードのみ) | action ★keep |
| `C` | - | 2639 | `MaskPicker::clear` (画像モードのみ) | action ★keep |
| `UP` | - | 2642 | `g_voxelSize += 0.05` | param adjust ★keep |
| `DOWN` | - | 2648 | `g_voxelSize -= 0.05` | param adjust ★keep |
| `K` | - | 2654 | `runCameraDepthEstimation` | **action ★keep** |
| `J` | - | 2677 | `gCamera.saveForDepthEstimation` | **action ★keep** |
| `S` | - | 2692 | `gCamera.captureCurrentFrame` | **action ★keep** |
| `L` | - | 2701 | `gCamera.releaseCapture` | **action ★keep** |
| `M` | - | 2710 | export STL (long block) | **action ★keep** |
| `M` | Shift | 2717 | export STL with X+Z flip | **action ★keep** |

**集計**: action ★keep = 24, viz/debug → checkbox/button 移行 = 14, 移動 (V/F → G) = 2

---

## 4. 移行後の最終キー表

```
■ Action 系 (キーボードに残す)
─────────────────────────────────────────────────────────────
  Registration 系:
    O                 HemiAuto                (OrbitCam視点)
    Shift+O           QuadAuto                (AR固定視点)

  BIPOP-CMA-ES 系 (G ファミリーに統合):
    Alt+G             V1 BIPOP-CMA-ES         [旧 Shift+V]
    Alt+Shift+G       V2 BIPOP-CMA-ES Fast    [旧 Shift+F]
    Shift+G           V3 BIPOP-CMA-ES         (Good performance)
    Ctrl+G            V3-R Region-aware       (主動線)
    Ctrl+Shift+G      V3-RS Silhouette anchor

  Silhouette / Cyclic 系 (P ファミリー):
    P                 SilhouetteHemi
    Shift+P           Cyclic Boundary
    Ctrl+P            QuadCyclic
    Ctrl+Shift+P      QuadCyclic-RANSAC
    Ctrl+Alt+P        AutoQCR                 (推奨初期化)

  Shift+E           Silhouette Align (2D BIPOP)

  Shape Match 系 (W ファミリー):
    Ctrl+W            Shape Match → apply → save
    Ctrl+Shift+W      Shape Match → axis sweep → Live ICP bridge
    Alt+W             Shape Match + GN refine
    Ctrl+Alt+W        Contour/Silhouette Sweep

  Normal-Compat Refine (N ファミリー):
    Shift+N           NORMAL_COMPAT refine
    Ctrl+Shift+N      SRT_VARIANCE refine

  Camera / IO:
    R                 Run depth (image mode)
    K                 Camera depth estimation
    J                 Save camera frame
    S                 Snapshot
    L                 Live view
    M                 Export STL
    Shift+M           Export STL with X+Z flip

  Pose Library / Undo:
    Q                 Pose Library window
    X                 Pose Undo

  Display:
    D                 AR save snapshot
    Ctrl+D            Debug Panel window
    F2                Camera reset
    A                 AR background overlay
    F9                Silhouette IoU overlay window

  Voxel / threshold tuning:
    Up / Down         g_voxelSize ±0.05
    , / .             silhouette threshold ∓0.01 (Shift で ∓0.05)

  Mask picker (image-only mode):
    U                 undo
    C                 clear

  Other:
    Esc               Close

■ 廃止 (Ctrl+D のチェックボックス / ボタンに移行)
─────────────────────────────────────────────────────────────
  V (cluster viz), B (boundary cand viz), Shift+B (cyclic corr viz),
  N (source vis), W (rim chain viz), Shift+W (target boundary viz),
  Shift+R (liver region viz), Y (liver LR viz),
  H (liver quad viz), Shift+H (liver CC viz),
  Shift+T (region recompute), Shift+Y (LR recompute),
  Shift+I (IoU debug dump)
  F10 (vertex squash diag) — キーは互換のため残す + Ctrl+D にも追加
```

---

## 5. 実装フェーズ (Claude Code 推奨手順)

各 phase は **独立した commit**。各 phase 終了時に `lsn_registration` が必ずビルド
通ること、可能なら起動して動作確認すること。

### Phase 0: アンカー取得 (10 分)

**目的**: 後の phase で「壊れていない」と判断するための基準を残す。

```bash
git checkout -b key-reorg ui-refactor   # ui-refactor を親ブランチに
cd registration
# 現状のキー dispatch をテキスト dump (人間が後で diff 用に読める形)
sed -n '1365,2820p' main.cpp > /tmp/keymap_before.cpp
wc -l /tmp/keymap_before.cpp   # 期待: ~1456 行
# ビルド確認 (アンカー)
cd ../build && cmake --build . --target lsn_registration
```

**コミット**: なし (作業メモのみ)。

### Phase 1: Viz Panel 拡張 (Ctrl+D > Viz タブに 10 チェックボックス + 3 ボタン追加)

**ファイル**: `registration/main.cpp` 6459 行目の
`g_debugPanel.drawVizExtra = [&]() { ... };` ラムダを **拡張する** (新規ラムダ
ではない)。既存内容 (Boundary candidates / Source vis / Screen mesh / debug
AABB) は **そのまま残す**。

#### 1-A. 追加するチェックボックス (現状の `g_show*` グローバルへの bool 参照)

ラムダの末尾 (line 6505 直前) に挿入:

```cpp
            // ---- [Phase 1] viz toggles migrated from keyboard ---------------
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                               "Visualization toggles (formerly keyboard):");

            // V key family
            ImGui::Checkbox("Cluster visualization (was V)##viz_cluster_full",
                            &g_showClusterVisualization);

            // B key family
            ImGui::Checkbox("Cyclic Correspondence — Shift+P pairs (was Shift+B)##viz_cyclic",
                            &g_showCyclicCorrespondence);

            // W key family
            ImGui::Checkbox("Debug Source Rim Chain — green dots (was W)##viz_rim_src",
                            &g_showDebugSourceRimChain);
            ImGui::Checkbox("Debug Target Boundary — purple dots (was Shift+W)##viz_rim_tgt",
                            &g_showDebugTargetBoundary);

            // Liver label viz
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Liver labels:");
            // Region (Shift+R)
            if (ImGui::Checkbox("Liver Region (anterior/rim/posterior) — was Shift+R##viz_region",
                                &g_showLiverRegion)) {
                if (g_showLiverRegion && !g_liverRegion.valid()) {
                    recomputeLiverRegion();
                }
            }
            // LR (Y)
            if (ImGui::Checkbox("Liver Left/Right — was Y##viz_lr",
                                &g_showLiverLR)) {
                if (g_showLiverLR && !g_liverLR.valid()) {
                    recomputeLiverLR();
                }
            }
            // CC (Shift+H)
            if (ImGui::Checkbox("Liver Cranio/Caudal — was Shift+H##viz_cc",
                                &g_showLiverCC)) {
                if (g_showLiverCC && !g_liverCC.valid()) {
                    recomputeLiverCC();
                }
            }
            // Quad (H)
            if (ImGui::Checkbox("Liver 4-Quadrant overlay — was H##viz_quad",
                                &g_showLiverQuad)) {
                if (g_showLiverQuad &&
                    g_quadVizIdxAR.empty() && g_quadVizIdxAL.empty() &&
                    g_quadVizIdxPR.empty() && g_quadVizIdxPL.empty()) {
                    recomputeLiverQuad();
                }
            }

            // Recompute buttons (formerly Shift+T / Shift+Y)
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Recompute labels:");
            if (ImGui::Button("Recompute Region  (was Shift+T)##btn_recompute_region")) {
                std::cout << "[Region] recomputing with target_rim_mm = "
                          << g_rimTargetMm << std::endl;
                recomputeLiverRegion();
            }
            ImGui::SameLine();
            if (ImGui::Button("Recompute LR  (was Shift+Y)##btn_recompute_lr")) {
                std::cout << "[LR] recomputing  right_pure_fraction = "
                          << g_lrPureFrac << "  right_full_fraction = "
                          << g_lrFullFrac << std::endl;
                recomputeLiverLR();
            }

            // Debug dump (formerly Shift+I)
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f), "Debug dumps:");
            if (ImGui::Button("Dump IoU debug PNG  (was Shift+I)##btn_iou_dump")) {
                if (gApp.mode == AppMode::kRegistration) {
                    glm::mat4 silView = buildSilhouetteView();
                    glm::mat4 silProj = buildSilhouetteProj();
                    int silW = (OrbitCam.calibWidth  > 0) ? OrbitCam.calibWidth  : 1280;
                    int silH = (OrbitCam.calibHeight > 0) ? OrbitCam.calibHeight : 720;
                    IoUDebug::dump(DEPTH_OUTPUT_PATH, "iou_debug",
                                   liverMesh3D, silView, silProj, silW, silH, 8);
                } else {
                    std::cout << "[IoU dump] requires Registration mode" << std::endl;
                }
            }
            if (ImGui::Button("Vertex-Squash diagnose  (was F10)##btn_vsq_diag")) {
                diagnoseVertexSquashV3RS(g_activeQuadrantMask);
            }
            // ---- end Phase 1 migration -------------------------------------
```

#### 1-B. 既存の重複を整理

`drawVizExtra` 冒頭の **既存の**「Other markers:」セクション (line 6463–6467) は、
`g_showBoundaryCandidates` と `g_showSourceVisualization` のチェックボックスを
すでに持っている。**今回新規追加するもの** と重ならないように、既存ラベル文を
下記に更新:

```cpp
// 旧: "Boundary candidates (B key)##viz_b"  →  新:
ImGui::Checkbox("Boundary candidates (was B)##viz_b", &g_showBoundaryCandidates);
// 旧: "Source visualization (N key)##viz_n"  →  新:
ImGui::Checkbox("Source visualization (was N)##viz_n", &g_showSourceVisualization);
```

**変更箇所**: line 6464, 6466 の文字列のみ (2 行)。

#### 1-C. ビルド確認

```bash
cd build && cmake --build . --target lsn_registration -j8
```

**コミット**: `[key-reorg] Phase 1: add migrated viz/recompute UI to Debug Panel Viz tab`

#### 1-D. 動作確認 (Qt Creator で起動して)

- Ctrl+D で Debug Panel を開く → Viz タブを開く
- 新規追加した 10 チェックボックス + 3 ボタンが見えること
- 既存のキーは **まだ全部動く** (この phase ではキー dispatch は触っていない)
- チェックボックスでトグル → キー (V / B / N / W / Shift+W / Shift+R / Y / H /
  Shift+H) でトグル → 両方で同じ flag をいじっているので二箇所から触れる

---

### Phase 2: V キーを Alt+G に移動 (BIPOP-CMA-ES V1)

**ファイル**: `registration/main.cpp`

#### 2-A. `isAltG` フラグを追加

line 1373 の `isCtrlG` 定義の **直下** に追加:

```cpp
    const bool isAltG       = (key == GLFW_KEY_G) && (mods & GLFW_MOD_ALT)
                                                  && !(mods & GLFW_MOD_CONTROL)
                                                  && !(mods & GLFW_MOD_SHIFT);  // V1 BIPOP-CMA-ES (旧 Shift+V)
    const bool isAltShiftG  = (key == GLFW_KEY_G) && (mods & GLFW_MOD_ALT)
                                                  && (mods & GLFW_MOD_SHIFT)
                                                  && !(mods & GLFW_MOD_CONTROL); // V2 BIPOP-CMA-ES Fast (旧 Shift+F)
```

#### 2-B. `needsScene` に追加

line 1379–1391 の `needsScene` 判定に `isAltG` と `isAltShiftG` を追加:

```cpp
    const bool needsScene = (key == GLFW_KEY_O      ||
                             key == GLFW_KEY_P      ||
                             isShiftV               ||  // ※削除予定 (Phase 3)、現状は残す
                             isShiftE               ||
                             isShiftF               ||  // ※削除予定 (Phase 3)、現状は残す
                             isShiftG               ||
                             isCtrlG                ||
                             isCtrlShiftG           ||
                             isAltG                 ||  // ★新規
                             isAltShiftG            ||  // ★新規
                             isShiftN               ||
                             isCtrlShiftN           ||
                             key == GLFW_KEY_D      ||
                             key == GLFW_KEY_COMMA  ||
                             key == GLFW_KEY_PERIOD);
```

#### 2-C. `GLFW_KEY_G` の switch case に Alt+G / Alt+Shift+G 分岐を追加

line 1494 の `if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT))` の **直前**
に挿入 (Alt 系を先に判定):

```cpp
        // ----- Alt+G : V1 BIPOP-CMA-ES (旧 Shift+V) -----
        //   Shift+V から Alt+G に移動。動作は完全に同一。
        //   元コード line 1446-1453 (旧 GLFW_KEY_V Shift 分岐) と byte-identical。
        if ((mods & GLFW_MOD_ALT) && !(mods & GLFW_MOD_CONTROL)
                                  && !(mods & GLFW_MOD_SHIFT)) {
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            runBipopCmaes();
            poseSaveToLibrary(SaveCriterion::RMSE);
            break;
        }
        // ----- Alt+Shift+G : V2 BIPOP-CMA-ES Fast (旧 Shift+F) -----
        //   Shift+F から Alt+Shift+G に移動。動作は完全に同一。
        //   元コード line 1468-1474 (旧 GLFW_KEY_F Shift 分岐) と byte-identical。
        if ((mods & GLFW_MOD_ALT) && (mods & GLFW_MOD_SHIFT)
                                  && !(mods & GLFW_MOD_CONTROL)) {
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            runBipopCmaesV2();
            poseSaveToLibrary(SaveCriterion::RMSE);
            break;
        }
        // ---- (既存 Ctrl+Shift+G, Ctrl+G, Shift+G はそのまま) ----
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT)) {
            // ... 既存の Ctrl+Shift+G コード ...
```

#### 2-D. **互換性のため、旧 Shift+V / Shift+F は当面残す**

Phase 2 の段階では旧キー (Shift+V / Shift+F) は **削除しない**。
両方 (新 + 旧) で同じ関数を呼び、後の Phase で旧側を削除する。
これにより、ユーザの muscle memory が完全に追従するまで両刀使えるようにする。

理由: Claude Code がデバッグ中に Shift+V を癖で押す可能性 + Phase 別 commit で
他チームメンバ (もしいれば) が中間 revision を使う可能性。

#### 2-E. ログメッセージで「移行」を出す (Phase 7 で削除)

旧 Shift+V / Shift+F の分岐内に **一時的に** 警告ログを足す:

```cpp
    case GLFW_KEY_V:
        if (mods & GLFW_MOD_SHIFT) {
            std::cout << "[DEPRECATION] Shift+V is moving to Alt+G next release"
                      << std::endl;
            g_stepStartTime = std::chrono::steady_clock::now();
            // ...
```

```cpp
    case GLFW_KEY_F:
        if (mods & GLFW_MOD_SHIFT) {
            std::cout << "[DEPRECATION] Shift+F is moving to Alt+Shift+G next release"
                      << std::endl;
            // ...
```

#### 2-F. ビルド + 動作確認

- `lsn_registration` ビルドが通る
- HemiAuto (O) → Alt+G で V1 が走る (旧 Shift+V と同等)
- HemiAuto (O) → Alt+Shift+G で V2 が走る (旧 Shift+F と同等)
- 旧 Shift+V / Shift+F も動く (deprecation ログが出る)

**コミット**: `[key-reorg] Phase 2: add Alt+G (V1) and Alt+Shift+G (V2) BIPOP shortcuts`

---

### Phase 3: viz トグルキーを無効化 (10 個)

**ファイル**: `registration/main.cpp` glfw_onKey switch case

各 viz トグル分岐を **削除せず**、`std::cout` に「Ctrl+D の Viz タブに移動」と
案内を出して **早期 return** する。実装は維持する (チェックボックスから既に呼ばれる)
のでバグるリスクなし。

#### 対応表

| キー | 旧コード行 | 修正後の挙動 |
|---|---|---|
| `V` (plain) | 1454-1458 | DEPRECATION ログ + `g_showClusterVisualization` トグル維持 |
| `B` (plain) | 1611-1620 | DEPRECATION ログ + トグル維持 |
| `Shift+B` | 1597-1610 | DEPRECATION ログ + トグル維持 |
| `N` (plain) | 1702-1711 | DEPRECATION ログ + トグル維持 |
| `W` (plain) | 2263-2308 | DEPRECATION ログ + トグル維持 |
| `Shift+W` | 2238-2261 | DEPRECATION ログ + トグル維持 |
| `Shift+R` | 2531-2546 | DEPRECATION ログ + トグル維持 |
| `Y` (plain) | 2577-2591 | DEPRECATION ログ + トグル維持 |
| `H` (plain) | 2616-2634 | DEPRECATION ログ + トグル維持 |
| `Shift+H` | 2595-2615 | DEPRECATION ログ + トグル維持 |
| `Shift+T` | 2560-2566 | DEPRECATION ログ + recompute 実行維持 |
| `Shift+Y` | 2569-2576 | DEPRECATION ログ + recompute 実行維持 |
| `Shift+I` | 2435-2445 | DEPRECATION ログ + dump 実行維持 |

#### パターン例 (V キーの場合)

```cpp
    case GLFW_KEY_V:
        if (mods & GLFW_MOD_SHIFT) {
            std::cout << "[DEPRECATION] Shift+V is moving to Alt+G next release"
                      << std::endl;
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUI.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            runBipopCmaes();
            poseSaveToLibrary(SaveCriterion::RMSE);
        } else {
            // [DEPRECATION] Plain V (cluster viz toggle) → use Ctrl+D > Viz tab
            std::cout << "[DEPRECATION] Plain V cluster viz toggle moved to "
                         "Ctrl+D > Viz tab. Toggling here too for back-compat."
                      << std::endl;
            g_showClusterVisualization = !g_showClusterVisualization;
            std::cout << "Cluster visualization: "
                      << (g_showClusterVisualization ? "ON" : "OFF") << std::endl;
        }
        break;
```

**全 13 個に同じパターン**を適用する。ログ文言は

`[DEPRECATION] <Old Key> <function> moved to Ctrl+D > Viz tab. ...`

統一する。

**コミット**: `[key-reorg] Phase 3: emit deprecation warnings on viz-toggle keys`

---

### Phase 4: viz トグルキーの完全削除

**前提**: Phase 3 を merge してから **最低 1 週間** 運用、または別ユーザに
動作確認してもらう。問題がなければ実行。

#### 削除対象の switch case (case 全部、または該当 mod 分岐のみ)

**完全削除する case** (他の mod 分岐がない場合):
- `case GLFW_KEY_H:` (line 2594-2635) — Plain H / Shift+H 両方 viz、case 自体削除

**条件付き削除** (action もある case):
- `GLFW_KEY_V`: `else` 分岐 (plain V) 削除、Shift+V 分岐は **Phase 5 で削除予定**
- `GLFW_KEY_F`: 何もしない (Shift+F のみ、Phase 5 で削除予定)
- `GLFW_KEY_B`: case 全部削除 (両方 viz)
- `GLFW_KEY_N`: `else` 分岐削除 (Shift+N / Ctrl+Shift+N は keep)
- `GLFW_KEY_W`: `mods == 0` 分岐削除 + `Shift && !Ctrl` 分岐削除 (Ctrl+W, Ctrl+Shift+W, Alt+W, Ctrl+Alt+W は keep)
- `GLFW_KEY_R`: `if (mods & GLFW_MOD_SHIFT) { ... break; }` 削除 (Plain R は keep)
- `GLFW_KEY_T`: case 全部削除 (Shift+T のみ)
- `GLFW_KEY_Y`: case 全部削除 (両方 viz/recompute)
- `GLFW_KEY_I`: case 全部削除

#### needsScene からも削除

line 1379–1391 から `key == GLFW_KEY_D` の前の関連キーを残しつつ、無効化された
キーは削除。具体的には `isShiftR_viz`, `isPlainY` のような明示判定は導入しない
(needsScene には影響しない、トグルなのでチェックボックスから常に呼べる)。

**コミット**: `[key-reorg] Phase 4: remove migrated viz-toggle keyboard cases`

---

### Phase 5: 旧 Shift+V / Shift+F の削除

**前提**: Phase 2 を merge してから 1 週間以上、Alt+G / Alt+Shift+G が問題なく
動いていること。

- `case GLFW_KEY_V:` を完全削除 (Phase 4 で plain V は既に削除されているはず)
- `case GLFW_KEY_F:` を完全削除
- `needsScene` から `isShiftV` と `isShiftF` を削除
- `isShiftV` / `isShiftF` の定数定義 (line 1368, 1370) を削除

**コミット**: `[key-reorg] Phase 5: remove deprecated Shift+V and Shift+F`

---

### Phase 6: Debug Panel の改善 (オプション、ベスト)

#### 6-A. Viz タブのチェックボックスをカテゴリ別にグループ化

`drawVizExtra` 内を CollapsingHeader で整理:

```cpp
if (ImGui::CollapsingHeader("Registration cluster viz", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Cluster markers (formerly J/V key)##...", &g_showClusterVisualization);
    ImGui::Checkbox("Correspondence points##...", &g_showCorrespondencePoints);
    ImGui::Checkbox("Cyclic correspondence (Shift+P pairs)##...", &g_showCyclicCorrespondence);
}
if (ImGui::CollapsingHeader("Boundary / Rim debug")) {
    ImGui::Checkbox("Boundary candidates##...", &g_showBoundaryCandidates);
    ImGui::Checkbox("Source visualization##...", &g_showSourceVisualization);
    ImGui::Checkbox("Source rim chain##...", &g_showDebugSourceRimChain);
    ImGui::Checkbox("Target boundary##...", &g_showDebugTargetBoundary);
}
if (ImGui::CollapsingHeader("Liver labels")) {
    // Region / LR / CC / Quad checkboxes + recompute buttons
}
if (ImGui::CollapsingHeader("Screen mesh rendering")) {
    // Existing screen mesh content
}
if (ImGui::CollapsingHeader("Debug dumps")) {
    // IoU dump button, Vertex-squash diag button
}
```

#### 6-B. F9 と Ctrl+D の整理

F9 (SilOverlay window) はキー保持する必要があるか UI で再評価:
- 既存運用で F9 を多用していれば残す
- 使ってなければ Ctrl+D 内 "Debug dumps" セクションに `Show SilOverlay window` チェックボックス追加

**コミット**: `[key-reorg] Phase 6: regroup Debug Panel Viz tab with collapsing headers`

---

### Phase 7: コメント / ドキュメント整理

#### 7-A. main.cpp 内のキー言及コメントを更新

```bash
cd registration
# 旧キー言及を全部 grep
grep -n "Shift+V\|Shift+F\|key V\|key B\|key N\|Shift+R\|Shift+T\|Shift+Y\|Shift+I\|Shift+H\|key H\|key Y" main.cpp | wc -l
# (期待値: 数十件)
```

これらコメント内のキー参照を新しいキーに置換、または「(now in Ctrl+D Viz tab)」
を付記する。中身を変えるのは Phase 5/6 で済んだコードのコメント部分のみ。

#### 7-B. PROJECT_STRUCTURE.md / README にキー早見表追加

Phase 4 完成版の表 (本書 §4) を `docs/KEY_REFERENCE.md` として独立配置。

#### 7-C. DebugPanel.h のヘッダコメント更新

`DebugPanel.h` line 8–18 の「Tabs organize controls by keyboard-shortcut family」
表を更新。Viz タブの説明に「formerly V/B/N/W/Y/H/R/T/I keys」を追記。

**コミット**: `[key-reorg] Phase 7: documentation update`

---

### Phase 8: 受け入れテスト (verification)

#### 8-A. 自動 smoke test (バッチ)

```
1. アプリ起動 (lsn_registration)
2. ファイルドロップ → R で depth 走らせる
3. Ctrl+D で Debug Panel 表示確認、Viz タブで全チェックボックス見える
4. O → HemiAuto 実行
5. Ctrl+D > Viz タブ "Recompute Region/LR" ボタン押下 → 結果ラベル表示
6. Alt+G → V1 BIPOP 実行 (旧 Shift+V と同じ結果)
7. Alt+Shift+G → V2 BIPOP 実行 (旧 Shift+F と同じ結果)
8. Ctrl+G → V3-R 実行 (回帰なし)
9. Shift+N → Normal-Compat refine 実行
10. Pose Library (Q) で全 entry が見える
11. 旧キー Shift+V/F/T/Y/I/H/Plain V/B/N/W/Shift+W/Y/Shift+R を押す
    → DEPRECATION ログ (Phase 3 段階) または何も起こらない (Phase 4 後)
```

#### 8-B. 退行確認

Phase 2 のコミット時点で `git stash` した状態の `runBipopCmaes` の最終 CompRMSE
と、Alt+G から呼ばれた `runBipopCmaes` の最終 CompRMSE が **byte-identical** で
あること。`g_trialSeed` と `g_callIdx` をリセットして同じ seed/index から走らせ
れば再現可能。

---

## 6. ロールバック手順

各 phase は単独 commit なので:

```bash
git revert <phase-N-commit>   # 単一 phase だけ戻す
git reset --hard <pre-phase-N> # 複数 phase をまとめて戻す (本格巻き戻し)
```

特に Phase 4 (キー削除) を revert すると、Ctrl+D のチェックボックスと
キーが両方使える状態 (Phase 3 末) に戻る。これが「両刀状態」として安全圏。

---

## 7. 注意点 (Claude Code が忘れがちなポイント)

1. **`needsScene` の更新を忘れない**: 新キー (Alt+G, Alt+Shift+G) を追加するときは
   必ず `needsScene` 判定にも追加。さもないと Image-only モードで押した時に
   早期 return せず、未初期化 mesh にアクセスして crash する可能性。

2. **mod 判定順は Alt+Ctrl > Shift+Ctrl > Ctrl > Shift > 単独**: GLFW_KEY_G の
   switch 内で `if-else if` の順序を間違えると、例えば Alt+Shift+G が
   Shift+G (V3) に喰われる。Alt+G 分岐は **必ず Ctrl+Shift+G の前に置く**。
   GLFW_KEY_P の既存実装 (line 2312 から) を雛形として真似る。

3. **`g_sessionBipopN++` のタイミング**: 新キー (Alt+G/Alt+Shift+G) でも
   `g_sessionBipopN++` を BIPOP 系では必ず実行。HemiAuto/QuadAuto は逆に
   `g_sessionBipopN = 0` でリセット (新規試行扱い)。旧 V/F 分岐の挙動と完全一致
   させる。

4. **`runHemiAuto` の事前要求**: BIPOP 系は全部 `runHemiAuto` (= O) を事前に
   走らせていることが前提 (`registrationHandle.compRmse > 0` を内部チェック)。
   Alt+G/Alt+Shift+G 押下時のエラーメッセージは旧 Shift+V/Shift+F の同等の文言を
   流用 (RegistrationActions.h 内の `runBipopCmaes` / `runBipopCmaesV2` 冒頭の
   `std::cerr << "[Shift+V] No registration yet"` 等)。**この cerr のプレフィクス
   は変えない** (ログ grep の互換性のため)。

5. **`g_showLiverRegion` 系 checkbox の auto-recompute**: 旧 Shift+R の動作は
   「toggle ON 時に未計算なら自動 recompute」。Phase 1 の checkbox にも同じ
   if 分岐を入れた。ここを忘れると初回 ON 時に何も表示されなくなる。

6. **コメント内のキー参照**: `RegistrationActions.h` / `PoseLibrary.h` 等の
   ヘッダ内コメントには「(Shift+V)」「(Ctrl+G)」等の参照が大量にある。Phase 7 で
   一括 grep して、削除されたキーへの参照は新キーまたは「Ctrl+D Viz tab」に
   置換。**ヘッダ内の関数名は変えない** (`runBipopCmaes` のまま、`runV1AltG` 等に
   リネームしない)。

7. **Phase 1 と Phase 6 の差**: Phase 1 は「とにかく動くチェックボックスを足す」、
   Phase 6 は「CollapsingHeader で整理して見やすくする」。Phase 1 完了時点で
   Viz タブはチェックボックスが縦に大量に並ぶが、機能的には完成。Phase 6 は
   見た目だけの改善なので skip しても可。

8. **Qt Creator でビルドする場合**: `build/` が cmake cache を保持しているので、
   `DebugPanel.h` を変更すると `main.cpp` の再ビルドが必要 (include 関係)。
   `cmake --build build --target lsn_registration` でフルビルド推奨。

---

## 8. チェックリスト (Claude Code 用)

Phase 1 完了:
- [ ] `g_debugPanel.drawVizExtra` のラムダに 10 チェックボックス + 3 ボタン追加
- [ ] 既存「Other markers」の文字列ラベルを `(was B)` / `(was N)` に更新
- [ ] `lsn_registration` ビルド成功
- [ ] Ctrl+D 開いて Viz タブで全チェックボックス可視

Phase 2 完了:
- [ ] `isAltG` / `isAltShiftG` 定数を line 1373 直下に追加
- [ ] `needsScene` に `isAltG` / `isAltShiftG` 追加
- [ ] `GLFW_KEY_G` の switch case 内、Ctrl+Shift+G 分岐の直前に Alt+G と
      Alt+Shift+G 分岐を追加 (mod 判定順注意)
- [ ] 旧 Shift+V / Shift+F に DEPRECATION ログ追加
- [ ] Alt+G で V1 BIPOP 動作確認

Phase 3 完了:
- [ ] 13 個の viz/recompute/debug キーに DEPRECATION ログ追加、機能は維持

Phase 4 完了:
- [ ] 13 個のキー dispatch を削除 (チェックボックスのみで動く状態)
- [ ] 各 case 削除に伴う `needsScene` の整理

Phase 5 完了:
- [ ] `case GLFW_KEY_V:` / `case GLFW_KEY_F:` 削除
- [ ] `isShiftV` / `isShiftF` 定数定義削除
- [ ] `needsScene` から `isShiftV` / `isShiftF` 削除

Phase 6 (optional) 完了:
- [ ] Viz タブを CollapsingHeader で 5 セクション化

Phase 7 完了:
- [ ] main.cpp 内コメントのキー参照更新
- [ ] `DebugPanel.h` ヘッダコメント表更新
- [ ] `docs/KEY_REFERENCE.md` 作成

Phase 8 完了:
- [ ] 8-A の smoke test 全 11 ステップ pass
- [ ] 8-B byte-identical 確認 (Alt+G ↔ 旧 Shift+V)

---

## 9. 推定工数

| Phase | 内容 | 推定時間 |
|---|---|---|
| 0 | アンカー取得 | 10 分 |
| 1 | Viz Panel 拡張 | 1 時間 |
| 2 | Alt+G / Alt+Shift+G 追加 | 30 分 |
| 3 | DEPRECATION ログ追加 | 30 分 |
| 4 | キー削除 | 1 時間 |
| 5 | V/F case 削除 | 15 分 |
| 6 | CollapsingHeader 整理 (任意) | 30 分 |
| 7 | コメント / docs 更新 | 1 時間 |
| 8 | 受け入れテスト | 1-2 時間 (実機) |
| **合計** | | **5-7 時間** |

---

## Appendix A: 修正ファイルサマリ

```
registration/main.cpp                  ~50 修正 / ~200 削除 / ~120 追加 (実質 net +70 行)
common/src/DebugPanel.h                ヘッダコメント 5 行更新
docs/KEY_REFERENCE.md                  新規作成 (Phase 7)
```

**他のヘッダ** (`RegistrationActions.h`, `PoseLibrary.h`, `CmaesRefineV3*.h` 等):
- Phase 7 のコメント内キー参照置換のみ
- **関数名・引数・ロジックは一切変更しない**

これにより、Phase 5 完了時点で「キー dispatch のロジック書き換えは main.cpp の中だけ」
で完結する。
