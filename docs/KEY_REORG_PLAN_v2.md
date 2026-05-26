# キー整理 Phase 9–13 実装計画書 (v2)

**プロジェクト**: AAA_LiverSurgeryNaviComb (`lsn_registration`)
**前提**: ブランチ `key-reorg` で Phase 1–7 が完了済み (cluster checkbox 重複は Phase 5.5 で解消推奨)
**作成日**: 2026-05-25
**目的**: Shift+E を P ファミリーに統合 + UI 重複キー (R/K/S/L/J/F2/F9/F10/M) を整理

---

## 0. 整理方針の根拠

`RegistrationImGuiManager.h` を精査した結果、以下のキーには **既に UI ボタンが存在** していて、キーボード側は重複していました:

| キー | 現動作 | UI ボタン (行番号) |
|---|---|---|
| `R` (image-only) | `runDepthAndUpdateScene` | "Run Depth" (line 2469) |
| `K` (camera) | `runCameraDepthEstimation` | "Run Depth" (両 mode で同じボタン) |
| `S` | live → snapshot | Camera toggle (state 1→2, line 781) |
| `L` | snapshot → live 戻り | Camera toggle / "Re-Capture" (line 2476) |
| `F2` | OrbitCam reset | "Cam Init" (line 2051) |
| `F10` | vertex-squash diag | Phase 1 で Debug Panel にボタン追加済み |

UI ボタンがない or 部分的なもの:

| キー | 状態 | 計画 |
|---|---|---|
| `J` | raw camera frame 保存 (UI ボタンなし) | **削除** (副産物として Run Depth 時に書かれる) |
| `M` / `Shift+M` | STL export (UI ボタンなし) | **UI 追加 → キー削除** |
| `F9` | SilOverlay window 開閉 (UI なし) | **Debug Panel にチェックボックス追加 → キー削除** |
| `Shift+E` | Silhouette Align | **`Alt+P` に移動** |

---

## 1. 移行後のキー全体図

```
■ 削除されるキー (UI で代替)
─────────────────────────────────────────
  R, K, S, L     ← Camera state machine UI ボタン
  F2             ← "Cam Init" ボタン
  F9, F10        ← Debug Panel
  J              ← 副産物として自動保存 / 不要
  M, Shift+M     ← 新規 "Export STL" ボタン
  Shift+E        ← Alt+P に移動

■ 最終キー表 (Phase 13 で docs 更新)
─────────────────────────────────────────
  Registration:        O, Shift+O
  BIPOP-CMA-ES (G):    Alt+G, Alt+Shift+G, Shift+G, Ctrl+G, Ctrl+Shift+G
  Silhouette (P):      P, Shift+P, Ctrl+P, Ctrl+Shift+P, Ctrl+Alt+P, Alt+P (★new=Shift+E)
  Shape Match (W):     Ctrl+W, Ctrl+Shift+W, Alt+W, Ctrl+Alt+W
  Refine (N):          Shift+N, Ctrl+Shift+N
  Pose:                Q, X
  Display:             D, Ctrl+D
  Tuning:              Up/Down, , / .
  Mask picker:         U, C (image-only)
  AR overlay:          A
  Close:               Esc
```

**削除されるキー総数**: 11 個 (R, K, S, L, F2, F9, F10, J, M, Shift+M, Shift+E)
**追加**: 1 個 (Alt+P)

---

## 2. Phase 9: Shift+E → Alt+P 移動

**ファイル**: `registration/main.cpp`

### 9-A. mod 判定フラグ追加

旧 Shift+E は `isShiftE` (line 1369) 定義済みだが、`Alt+P` 用フラグを追加:

```cpp
    // Phase 9: Shift+E (Silhouette Align) を Alt+P へ移動
    // mod 判定順は Alt+Ctrl > Shift+Ctrl > Ctrl > Shift > Alt > 単独
    // ⇒ Alt+P は Ctrl+Alt+P より後、Ctrl+Shift+P/Ctrl+P/Shift+P より後、Plain P より前
    const bool isAltP = (key == GLFW_KEY_P) && (mods & GLFW_MOD_ALT)
                                            && !(mods & GLFW_MOD_CONTROL)
                                            && !(mods & GLFW_MOD_SHIFT);
```

`needsScene` 判定に `isAltP` も追加。

### 9-B. `GLFW_KEY_P` switch case 内に Alt+P 分岐追加

既存の `GLFW_KEY_P` case (~line 2311) の **`else if (mods & GLFW_MOD_SHIFT)` の前**、
**`else (Plain P)` の前** に Alt+P 分岐を挿入。位置注意:

```cpp
    case GLFW_KEY_P:
        if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_ALT)) {
            // Ctrl+Alt+P : AutoQCR (既存)
            runAutoQuadCyclicRansac(gUI.state.autoQcrLockScale);
        } else if ((mods & GLFW_MOD_CONTROL) && (mods & GLFW_MOD_SHIFT)) {
            // Ctrl+Shift+P : QuadCyclic-RANSAC (既存)
            ...
        } else if (mods & GLFW_MOD_CONTROL) {
            // Ctrl+P : QuadCyclic (既存)
            ...
        } else if ((mods & GLFW_MOD_ALT) && !(mods & GLFW_MOD_SHIFT)) {
            // ★ Phase 9: Alt+P (Silhouette Align, 旧 Shift+E)
            // 旧コード: case GLFW_KEY_E if (mods & GLFW_MOD_SHIFT)
            // 動作は byte-identical: poseAutoSave → runShiftE → poseSaveToLibrary(IOU)
            g_stepStartTime = std::chrono::steady_clock::now();
            poseAutoSaveBeforeRegistration();
            runShiftE();
            g_sessionSilhouetteN++;
            gUI.state.regMethod = 5;
            poseSaveToLibrary(SaveCriterion::IOU);
        } else if (mods & GLFW_MOD_SHIFT) {
            // Shift+P : Cyclic Boundary (既存)
            ...
        } else {
            // Plain P : SilhouetteHemi (既存)
            ...
        }
        break;
```

### 9-C. 旧 `GLFW_KEY_E` case を完全削除

main.cpp の `case GLFW_KEY_E:` (~line 2418) ブロック全体を削除。
`isShiftE` 変数定義 (line 1369) も削除、`needsScene` 内の `isShiftE` 参照も削除。

### 9-D. `runShiftE()` 内部のログプレフィクス保持

`RegistrationActions.h` の `runShiftE` 関数内の `std::cout << "[Shift+E]"` などの
ログ文字列は **そのまま** にする。grep 互換性のため。
新しい入口で `std::cout << "[Alt+P] dispatching Silhouette Align..." << std::endl`
だけ追加。

### 9-E. ビルド + 確認

- `lsn_registration` ビルド通る
- Alt+P で `runShiftE` が走る (regMethod=5, IoU criterion で PoseLibrary 保存)
- 旧 Shift+E は何も起こらない (ログも出ない)

**コミット**: `[key-reorg] Phase 9: move Silhouette Align Shift+E -> Alt+P`

---

## 3. Phase 10: R / K / S / L キー削除

**ファイル**: `registration/main.cpp`

### 10-A. 削除対象 case

```cpp
case GLFW_KEY_R:    // ~line 2530 — case 全体削除
case GLFW_KEY_K:    // ~line 2654 — case 全体削除
case GLFW_KEY_S:    // ~line 2692 — case 全体削除
case GLFW_KEY_L:    // ~line 2701 — case 全体削除
```

### 10-B. `needsScene` 整理

`needsScene` 判定から `key == GLFW_KEY_O || key == GLFW_KEY_P ...` の列を見直して
不要キー参照を削除。`R/K/S/L` は元々 `needsScene` に列挙されていなかったので影響なし
だが、念のため grep で確認:

```bash
grep -n "GLFW_KEY_[RKSL]\b" registration/main.cpp
```

(`R/K/S/L` への参照は他にないはず — switch case 内のみ。)

### 10-C. 「キーで操作してた人向け」の代替案内

`docs/KEY_REFERENCE.md` の "Removed keys" セクションに追加:

| 旧キー | 代替 UI 操作 |
|---|---|
| `R` | サイドバー "Run Depth" ボタン |
| `K` | サイドバー "Run Depth" ボタン (camera mode で同じボタン) |
| `S` | サイドバー Camera toggle (Start → Capture 状態遷移) |
| `L` | サイドバー Camera toggle / "Re-Capture" ボタン |

### 10-D. ビルド + 動作確認

- ビルド通る
- 画像ドロップ → "Run Depth" ボタン押下 → depth pipeline 走る (旧 R 相当)
- カメラ起動 → "Capture" → "Run Depth" 押下 → depth 走る (旧 S→K 相当)
- "Re-Capture" → live 戻り (旧 L 相当)

**コミット**: `[key-reorg] Phase 10: remove R/K/S/L keys (UI buttons cover all)`

---

## 4. Phase 11: F2 / F9 / F10 キー削除

### 11-A. F10 削除 (Debug Panel ボタン既存)

Phase 1 で `drawVizExtra` 内に「Vertex-Squash diagnose (was F10)」ボタンを追加済み。
main.cpp の `case GLFW_KEY_F10:` (~line 2502) を **case 全体削除**。

### 11-B. F2 削除 (UI "Cam Init" ボタン既存)

`case GLFW_KEY_F2:` (~line 2459) を **case 全体削除**。
UI ボタン "Cam Init" (RegistrationImGuiManager.h line 2051) が既に同じ
`OrbitCam.resetToInitialState()` + Registration mode の特殊処理を実行している
ことを **必ず確認**。

```bash
# UI 側の onResetCamera lambda が main.cpp で何を呼んでいるか確認
grep -A8 "gUI.actions.onResetCamera = " registration/main.cpp
```

もし UI lambda が `OrbitCam.resetToInitialState()` のみで、Registration mode 時の
追加処理 (rotation, currentTarget = TARGET_TEXTURE) を含んでいなかった場合は、
UI lambda 側に同じロジックを移植してから F2 を削除。

### 11-C. F9 → Debug Panel チェックボックス化

`g_debugPanel.drawVizExtra` (Phase 1 で拡張済み) の末尾「Debug dumps」セクションに
追加:

```cpp
            // F9 (SilOverlay window) -> checkbox
            ImGui::Spacing();
            ImGui::Checkbox("Show Silhouette Overlay window  (was F9)##viz_sil_overlay",
                            &SilOverlay::g_silOverlay.showWindow);
```

`case GLFW_KEY_F9:` (~line 2490) を **case 全体削除**。

### 11-D. ビルド + 確認

- ビルド通る
- "Cam Init" ボタンで F2 と同じ動作
- Debug Panel の "Show Silhouette Overlay window" チェックで F9 と同じ window 開閉
- Debug Panel の "Vertex-Squash diagnose" ボタンで F10 と同じ実行 (Phase 1 で完了済み確認)

**コミット**: `[key-reorg] Phase 11: remove F2/F9/F10 keys (UI covers all)`

---

## 5. Phase 12: J 削除 + M を UI ボタン化

### 12-A. J 削除

`case GLFW_KEY_J:` (~line 2677) を **case 全体削除**。

判断理由 (docs/KEY_REFERENCE.md に注記):
> J キーは raw camera frame を `depth_output/camera_frame_temp.jpg` に保存していたが、
> 同じファイルは "Run Depth" ボタン (camera mode) を押した時に自動で書かれる。
> 単独保存が必要な場合は手動でカメラ live view の状態でファイルシステムから直接
> アクセス、または将来 Debug Panel に "Save Raw Frame" ボタンを追加して対応。

### 12-B. M / Shift+M を UI ボタンに移行

#### 12-B-1. RegistrationImGuiManager.h に新規 callback

`RegUIActions` (line 21–) に追加:

```cpp
    // Phase 12: Export STL (旧 M / Shift+M)
    std::function<void()> onExportStl;        // M
    std::function<void()> onExportStlFlipped; // Shift+M (X+Z flip variant)
```

#### 12-B-2. UI ボタンの配置場所

`drawSaveAR()` (line 2045) の隣 (AR セクションの下) に新セクションを追加するか、
"Switch to Deform Mode" ボタン (line 1838 周辺) の前に Export section を追加。

推奨: **新規 CollapsingHeader "Export"** をサイドバーの下部 (Switch to Deform の上)
に作って、2 ボタンを並べる。

```cpp
    void drawExport() {
        if (!ImGui::CollapsingHeader("Export")) return;
        ImGui::Indent(16); ImGui::Spacing();
        float halfW = (ImGui::GetContentRegionAvail().x - 6) / 2.0f;
        if (colorButton("Export STL", colYellow(), false, false, halfW)) {
            if (actions.onExportStl) actions.onExportStl();
        }
        ImGui::SameLine();
        if (colorButton("Export STL (X+Z flip)", colOrange(), false, false, halfW)) {
            if (actions.onExportStlFlipped) actions.onExportStlFlipped();
        }
        ImGui::Spacing(); ImGui::Unindent(16); ImGui::Separator();
    }
```

`drawRegistrationSection()` 内のレイアウト関数呼出列に `drawExport();` を追加。

#### 12-B-3. main.cpp で callback 設定

`syncUIActions` または同等の lambda 設定箇所で:

```cpp
    gUI.actions.onExportStl = [&]() {
        // 旧 case GLFW_KEY_M (mods == 0) のロジックをまるごとコピー
        // 中身は ~line 2710 以降の STL export コード
        if (gApp.mode != AppMode::kRegistration) {
            std::cout << "[ExportSTL] only valid in Registration mode" << std::endl;
            return;
        }
        std::filesystem::create_directories(REG_MODEL_PATH);
        // ... 既存のロジック (cam-mm STL 書き出し) ...
    };
    gUI.actions.onExportStlFlipped = [&]() {
        // 旧 case GLFW_KEY_M (mods & GLFW_MOD_SHIFT) のロジックをまるごとコピー
        // 中身は ~line 2717 以降の Shift+M (X+Z flip) STL export コード
        if (gApp.mode != AppMode::kRegistration) {
            std::cout << "[ExportSTL] only valid in Registration mode" << std::endl;
            return;
        }
        std::filesystem::create_directories(REG_MODEL_PATH);
        // ... 既存のロジック (X flip + Z flip + scale restore STL 書き出し) ...
    };
```

注意点:
- ロジック本体は 2 つとも長い (~200 行)。lambda 内に直接書くと main.cpp が肥大化
  する。**ヘルパー inline 関数** を `registration/src/RegistrationActions.h` か
  新規ヘッダ `StlExport.h` に切り出して、lambda からはそれを呼ぶだけにする方が
  きれい。
- 既存の switch case 内の `std::cout << "[Shift+M]"` 等のログ文字列は **そのまま**
  維持 (grep 互換性)。lambda 入口で `std::cout << "[ExportSTL/UI]"` を追加してもよい。

#### 12-B-4. main.cpp の `case GLFW_KEY_M:` を完全削除

`case GLFW_KEY_M:` (~line 2710–) **case 全体を削除**。
ロジックは 12-B-3 の lambda またはヘルパーに移植済み。

### 12-C. ビルド + 動作確認

- ビルド通る
- サイドバーに "Export" CollapsingHeader が出現
- 開いて "Export STL" 押す → 旧 M キーと同じファイルが `registration_model/` に出る
- "Export STL (X+Z flip)" 押す → 旧 Shift+M と同じファイル

**コミット**: `[key-reorg] Phase 12: delete J key + move M/Shift+M to UI buttons`

---

## 6. Phase 13: ドキュメント整理 + 最終クリーンアップ

### 13-A. docs/KEY_REFERENCE.md 更新

```markdown
# Keyboard reference — lsn_registration (post key-reorg Phase 13)

## Action keys (kept)

### Registration
- O : HemiAuto
- Shift+O : QuadAuto

### BIPOP-CMA-ES (G family)
- Alt+G : V1
- Alt+Shift+G : V2 Fast
- Shift+G : V3
- Ctrl+G : V3-R region-aware
- Ctrl+Shift+G : V3-RS silhouette anchor

### Silhouette / Cyclic (P family)
- P : SilhouetteHemi
- Shift+P : Cyclic Boundary
- Ctrl+P : QuadCyclic
- Ctrl+Shift+P : QuadCyclic-RANSAC
- Ctrl+Alt+P : AutoQCR
- Alt+P : Silhouette Align (was Shift+E)

### Shape Match (W family) + refine
- Ctrl+W, Ctrl+Shift+W, Alt+W, Ctrl+Alt+W
- Shift+N, Ctrl+Shift+N

### Pose / Display / Tuning
- Q : Pose Library window
- X : Pose Undo
- D : AR save snapshot
- Ctrl+D : Debug Panel
- A : AR background overlay
- , / . : silhouette threshold tuning
- Up / Down : voxel size tuning

### Mask picker (image-only mode)
- U : undo
- C : clear

- Esc : Close

## Removed keys

These features moved to UI buttons:

| Old key | Now in |
|---|---|
| R, K | "Run Depth" sidebar button |
| S | Camera toggle button (state 1→2) |
| L | Camera toggle / "Re-Capture" button |
| F2 | "Cam Init" sidebar button |
| F9 | "Show Silhouette Overlay window" checkbox in Debug Panel > Viz |
| F10 | "Vertex-Squash diagnose" button in Debug Panel > Viz |
| J | (deleted — `camera_frame_temp.jpg` written automatically by "Run Depth") |
| M | "Export STL" button in new Export section |
| Shift+M | "Export STL (X+Z flip)" button |
| Shift+E | Moved to Alt+P |

These viz toggles moved to Debug Panel > Viz tab (Phase 1):

| Old key | Now in |
|---|---|
| V (plain) | "Cluster visualization" checkbox |
| B (plain) | "Boundary candidates" checkbox |
| Shift+B | "Cyclic correspondence" checkbox |
| N (plain) | "Source visualization" checkbox |
| W (plain) | "Debug source rim chain" checkbox |
| Shift+W | "Debug target boundary" checkbox |
| Shift+R | "Liver Region viz" checkbox |
| Y (plain) | "Liver Left/Right viz" checkbox |
| H (plain) | "Liver 4-Quadrant viz" checkbox |
| Shift+H | "Liver Cranio/Caudal viz" checkbox |
| Shift+T | "Recompute Region" button |
| Shift+Y | "Recompute LR" button |
| Shift+I | "Dump IoU debug PNG" button |
```

### 13-B. main.cpp のキー dispatch 全体の最終形を確認

```bash
grep -n "case GLFW_KEY_" registration/main.cpp
```

期待する残存 case 一覧 (約 19 個):

```
ESC, O, G, N, W, P, A, D, F, Q, X, COMMA, PERIOD, H?(消えてる), U, C, UP, DOWN
```

(`F` は file picker。`H` は Phase 4 で削除済みのはず。)

実際の残存 case が想定と一致するか確認、ズレがあれば調査。

### 13-C. DebugPanel.h ヘッダコメント更新

```cpp
//  Tabs organize controls by keyboard-shortcut family:
//    G    — Ctrl+G / Ctrl+Shift+G  (V3-R / V3-RS BIPOP-CMA-ES)
//    O    — Shift+O                (HemiAuto / QuadAuto, voxel, instrument)
//    N    — Shift+N / Ctrl+Shift+N (Normal-Compatible / SRT-Variance refine)
//    W    — (Phase 5 reserved)     RIM popups (silsw_*_popup)
//    U    — Umeyama Manual         (point-pair list, error stats)
//    Viz  — General visualization  + migrated keyboard toggles
//             (formerly V/B/N/W/Y/H/R/T/I/F9/F10 keys; see KEY_REFERENCE.md)
//
//  Toggle key: Ctrl+D
```

### 13-D. ヘッダ内コメントのキー参照整理

```bash
# 削除されたキーへの言及を grep
cd registration/src common/src
grep -rn "Shift+V\|Shift+F\|Shift+E\|Shift+I\|Shift+T\|Shift+Y\|Shift+R\|Shift+H\|key V\|key B\|key N\b\|key Y\|key H\|key J\|key K\|key S\|key L\|key M\|key R\|key F2\|key F9\|key F10" *.h *.cpp 2>/dev/null
```

ヒットしたコメントを以下のルールで置換:
- `Shift+V` → `Alt+G`
- `Shift+F` → `Alt+Shift+G`
- `Shift+E` → `Alt+P`
- `Key K/S/L/R` → "(deprecated, use UI button)"
- `F2` → "(deprecated, use Cam Init UI button)"
- 他の viz 系 → "(in Ctrl+D Debug Panel > Viz tab)"

**コミット**: `[key-reorg] Phase 13: final docs cleanup + comment references`

---

## 7. 動作確認チェックリスト (Phase 9–13 完了時)

### 必須確認 (実機)

- [ ] `Alt+P` で Silhouette Align (旧 Shift+E) が走る、PoseLibrary に IoU criterion で entry 追加
- [ ] 旧 `Shift+E` 押下時は何も起こらない (ログも出ない)
- [ ] 旧 `R`, `K`, `S`, `L`, `F2`, `F9`, `F10`, `J`, `M`, `Shift+M` 押下時に何も起こらない
- [ ] サイドバー "Run Depth" ボタンで image-only mode の depth 走る
- [ ] サイドバー Camera toggle で Start → Capture → Re-Capture の状態遷移 (live ↔ snapshot)
- [ ] サイドバー "Run Depth" ボタンで camera mode の depth 走る (旧 K 相当)
- [ ] サイドバー "Cam Init" ボタンで OrbitCam reset (旧 F2 相当)
- [ ] Debug Panel "Show Silhouette Overlay window" チェックで window 開閉 (旧 F9 相当)
- [ ] Debug Panel "Vertex-Squash diagnose" ボタンで実行 (旧 F10 相当)
- [ ] 新 "Export STL" ボタン → `registration_model/reg_*.stl` 生成 (旧 M 相当)
- [ ] 新 "Export STL (X+Z flip)" ボタン → X+Z flip 版 STL 生成 (旧 Shift+M 相当)

### byte-identical 確認 (退行テスト)

- [ ] 同じ trial_seed + callIdx で Alt+P 実行 → 旧 Shift+E と同じ最終 IoU / RMSE
- [ ] 同じシーンで "Run Depth" UI ボタン → 旧 R (image mode) と同じ screenMesh 生成

---

## 8. ロールバック

各 phase は独立 commit:

```bash
git revert <phase-N-commit>     # 単一 phase 戻す
git reset --hard <pre-phase-9>  # Phase 9-13 まとめて戻す
```

特に Phase 12 (M を UI 化) は STL export ロジック移植を含むので、retest 重要。
不安なら **Phase 12 だけスキップ** して M/Shift+M をキー残し可。

---

## 9. 注意点 (Claude Code 用)

### 9-A. mod 判定順 (Phase 9 用)

`GLFW_KEY_P` の switch 内で `else if` の順序は **必ず以下を守る**:

```
Ctrl+Alt+P → Ctrl+Shift+P → Ctrl+P → Alt+P (★新規) → Shift+P → Plain P
```

`Alt+P` 分岐を `Ctrl+P` の前に置くと、`Ctrl+Alt+P` 押下時に Alt+P に喰われる。
逆に `Shift+P` の後に置くと OK だが、可読性のため上記順序を推奨。

### 9-B. F2 削除前の onResetCamera 確認 (Phase 11)

main.cpp の lambda 設定箇所:

```bash
grep -A12 "gUI.actions.onResetCamera = " registration/main.cpp
```

この lambda が **以下と等価か** 確認:

```cpp
// 旧 F2 のロジック (main.cpp ~line 2459)
OrbitCam.resetToInitialState();
if (gApp.mode == AppMode::kRegistration) {
    OrbitCam.rotation = glm::angleAxis(glm::radians(180.0f),
                                       glm::vec3(0.0f, 1.0f, 0.0f));
    OrbitCam.currentTarget = TARGET_TEXTURE;
}
std::cout << "Camera reset" << std::endl;
```

UI lambda が **単純な `OrbitCam.resetToInitialState()` のみ** だった場合は、
Registration mode の特殊処理 (180° rotation, currentTarget = TARGET_TEXTURE) を
UI lambda 側に移植してから F2 を削除。**この移植を忘れると挙動が変わる**。

### 9-C. M を lambda に移植する際の長さ問題 (Phase 12)

旧 `case GLFW_KEY_M:` は ~200 行と長く、`SCALE_RESTORE` 計算、STL 出力、ファイル名
生成等が含まれる。これを lambda 内に直接書くと:
- main.cpp が肥大化
- syncUIActions のスコープが分かりにくくなる

**推奨**: 新規ヘッダ `registration/src/StlExport.h` を作成して `exportStl()` /
`exportStlFlipped()` の inline 関数として切り出す。lambda からは1行呼ぶだけ。
`RegistrationActions.h` の既存 pattern (runShiftE 等) と一致。

ただし、これは Phase 12 単独で完結できる範囲。STL ロジックを動かすのに必要な
extern global 参照 (liverMesh3D, OrbitCam, REG_MODEL_PATH, gWindowWidth など) を
ヘッダ冒頭で extern 宣言する必要あり。

### 9-D. Phase 1 の cluster checkbox 重複 (再掲)

Phase 1 完了報告で「cluster viz の checkbox が `drawTabViz` と `drawVizExtra` の
両方に存在する」と判明。Phase 9 開始前に **Phase 5.5** として下記 commit を
入れておくと綺麗:

```cpp
// DebugPanel.h drawTabViz() の冒頭 "Registration cluster (key J):" セクション
// (clusterVis / corresVis のチェックボックス) を削除
// →  drawVizExtra (main.cpp) の Phase 1 追加チェックボックスのみ残す
```

または逆 (Phase 1 で追加した方を削除して `drawTabViz` を残す) のいずれか統一。
**`drawTabViz` 側を残す方が、main.cpp lambda の依存が減って好み**。

---

## 10. 推定工数

| Phase | 内容 | 推定時間 |
|---|---|---|
| 5.5 | cluster checkbox 重複解消 | 5 分 |
| 9 | Shift+E → Alt+P | 15 分 |
| 10 | R/K/S/L 削除 | 10 分 |
| 11 | F2/F9/F10 削除 + F9 を Debug Panel checkbox | 20 分 |
| 12 | J 削除 + M を UI 化 (StlExport.h 切り出し) | 1 時間 |
| 13 | docs + コメント整理 | 30 分 |
| 動作確認 (実機) | 全 Phase 通し | 1 時間 |
| **合計** | | **3 時間** |

---

## Appendix: コミット粒度サマリ

```
[key-reorg] Phase 5.5: dedup cluster viz checkbox  ← optional
[key-reorg] Phase 9:   move Silhouette Align Shift+E -> Alt+P
[key-reorg] Phase 10:  remove R/K/S/L keys (UI buttons cover all)
[key-reorg] Phase 11:  remove F2/F9/F10 keys (UI covers all)
[key-reorg] Phase 12:  delete J key + move M/Shift+M to UI buttons
[key-reorg] Phase 13:  final docs cleanup + comment references
```
