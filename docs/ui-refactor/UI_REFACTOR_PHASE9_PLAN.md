# Phase 9 — Cleaner sidebar pass

**Branch:** `ui-refactor` (continues from Phase 0-8)
**Files touched:** `RegistrationImGuiManager.h` only
**Effort:** ~50 lines change, 2 sub-commits
**Risk:** Low — both are localized refactors with no callback wiring change

---

## What & why

After Phase 8, the sidebar still has 9 visible rows including 2 separate debug collapsing headers (`▸ QCR Tuning` and `▸ Advanced`) that each waste a full row on what is essentially a thin decoration. Two small improvements:

### Change 9a — POSITION quadrant grid: 2x2 → 1x4

The 2x2 grid inside `INITIAL ORIENTATION > POSITION` (4 checkboxes for `ant_R / ant_L / pos_R / pos_L`) currently uses ~80px vertical space. Reshaping to 1 row of 4 columns saves ~40px and reads naturally left-to-right.

The directional hint "Top=anterior, Left=patient's right" becomes irrelevant in a horizontal layout (no top/bottom mapping) and should be removed.

### Change 9b — Merge `▸ QCR Tuning` + `▸ Advanced` into one `▸ Tuning & Advanced`

Both contain debug/expert controls. Keeping them as 2 separate headers wastes a row. Merge into 1 collapsing header placed near the bottom (the existing Advanced location). Total sidebar visible rows: 9 → 7.

---

## Change 9a — POSITION grid 1x4

**File:** `RegistrationImGuiManager.h`
**Function:** `drawInitOrientationPanel()`
**Anchor:** search for `"##initorient_quad_grid"` (currently at line ~1272)

### Find the block:

```cpp
        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 0.85f), "  POSITION");
        ImGui::TextColored(ImVec4(0.60f, 0.65f, 0.75f, 0.85f),
                           "  Top=anterior, Left=patient's right");

        // ---- 2x2 grid (Ctrl+G と同じレイアウト) ----
        {
            const ImGuiTableFlags tableFlags =
                ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame;
            if (ImGui::BeginTable("##initorient_quad_grid", 2, tableFlags)) {
                auto checkbox_cell = [&](const char* shortName,
                                         int nv,
                                         uint8_t bit)
                {
                    // ... lambda body unchanged ...
                };

                // Row 1: anterior
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("ant_R", state.quadNAR, kMaskAR);
                ImGui::TableNextColumn();
                checkbox_cell("ant_L", state.quadNAL, kMaskAL);

                // Row 2: posterior
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("pos_R", state.quadNPR, kMaskPR);
                ImGui::TableNextColumn();
                checkbox_cell("pos_L", state.quadNPL, kMaskPL);

                ImGui::EndTable();
            }
        }
```

### Replace with:

```cpp
        ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 0.85f), "  POSITION");
        // [Phase 9a] Removed "Top=anterior, Left=patient's right" hint.
        // The hint described the 2x2 spatial mapping; with 1x4 horizontal
        // layout the mapping is just reading order (ant_R, ant_L, pos_R, pos_L).

        // ---- 1x4 grid (horizontal, Phase 9a) ----
        // [Phase 9a] Changed from 2x2 to 1x4 to reduce vertical footprint
        // (~40px saved). Reading order left-to-right: ant_R, ant_L, pos_R, pos_L.
        {
            const ImGuiTableFlags tableFlags =
                ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame;
            if (ImGui::BeginTable("##initorient_quad_grid", 4, tableFlags)) {
                auto checkbox_cell = [&](const char* shortName,
                                         int nv,
                                         uint8_t bit)
                {
                    bool on = (state.activeQuadrantMask & bit) != 0;
                    char label[64];
                    if (state.quadLabelsReady && nv >= 0) {
                        std::snprintf(label, sizeof(label),
                                      "%s\n(%d v)", shortName, nv);
                    } else {
                        std::snprintf(label, sizeof(label),
                                      "%s\n(--)", shortName);
                    }
                    ImGui::PushID(bit + 400);
                    if (ImGui::Checkbox(label, &on)) {
                        uint8_t newMask = state.activeQuadrantMask;
                        if (on)  newMask |= bit;
                        else     newMask = static_cast<uint8_t>(newMask & ~bit);
                        applyMaskChange(newMask);
                    }
                    ImGui::PopID();
                };

                // [Phase 9a] Single row, 4 columns. Same callbacks as 2x2.
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                checkbox_cell("ant_R", state.quadNAR, kMaskAR);
                ImGui::TableNextColumn();
                checkbox_cell("ant_L", state.quadNAL, kMaskAL);
                ImGui::TableNextColumn();
                checkbox_cell("pos_R", state.quadNPR, kMaskPR);
                ImGui::TableNextColumn();
                checkbox_cell("pos_L", state.quadNPL, kMaskPL);

                ImGui::EndTable();
            }
        }
```

**Note:** the existing `checkbox_cell` lambda body was identical, just re-pasted here for clarity. If the lambda is defined outside the visible find-block, keep its definition where it is and only change the table/column structure.

**Note on `childH`:** the function pre-computes a fixed `childH` for `BeginChild("##initOrient", ...)` around line 1144-1147. The `+ 64.0f // 2x2 grid` term should be reduced. Find:

```cpp
                  + 64.0f                                // 2x2 grid
```

Change to:

```cpp
                  + 32.0f                                // 1x4 grid (Phase 9a, was 64 for 2x2)
```

### Verification (9a)

1. Build clean.
2. Open INITIAL ORIENTATION CollapsingHeader. POSITION section shows 4 checkboxes in 1 row instead of 2x2.
3. Tick `ant_L` — Mask updates to `0x02 (AL)` and Orient preset auto-switches to `Left`.
4. The "Top=anterior" hint is gone.
5. Reading order left-to-right: ant_R | ant_L | pos_R | pos_L.

**Commit:** `[ui-refactor] Phase 9a: POSITION quadrant grid 2x2 → 1x4`

---

## Change 9b — Merge QCR Tuning + Advanced

**File:** `RegistrationImGuiManager.h`
**Function:** `drawRegistrationSection()`

### Step 9b.1 — Delete the standalone QCR Tuning block

**Anchor:** search for `"QCR Tuning  (Shift+Ctrl+P / AutoQCR)"` (currently at line ~1643).

Delete this entire block (currently lines ~1632-1690):

```cpp
        // ----------------------------------------------------------------
        //  QCR Tuning  (Shift+Ctrl+P / AutoQCR 共通、折りたたみ既定 OFF)
        // ----------------------------------------------------------------
        //  ... 8-line comment ...
        if (ImGui::CollapsingHeader("QCR Tuning  (Shift+Ctrl+P / AutoQCR)")) {
            ImGui::Indent(8);
            // ... entire body (AutoQCR 6-DoF lock + K/MaxTrials/AxisRot/TotalRot sliders) ...
            ImGui::Unindent(8);
        }
        ImGui::Spacing();
```

Replace with a single-line marker:

```cpp
        // [Phase 9b] QCR Tuning block merged into ▸ Tuning & Advanced below.
```

### Step 9b.2 — Expand the Advanced block to include QCR Tuning content

**Anchor:** search for `if (ImGui::CollapsingHeader("Advanced"))` (currently at line ~1821).

Replace the existing Advanced block:

```cpp
        if (ImGui::CollapsingHeader("Advanced")) {
            ImGui::Indent(8);

            // [Phase 8] Ctrl+G 6-DoF lock (relocated from the main button rim).
            {
                bool lockScale = state.ctrlgLockScale;
                if (ImGui::Checkbox("Ctrl+G 6-DoF lock (scale=1)##ctrlg_lock_adv", &lockScale)) {
                    const_cast<RegUIState&>(state).ctrlgLockScale = lockScale;
                    if (actions.onCtrlgLockScaleChanged)
                        actions.onCtrlgLockScaleChanged(lockScale);
                }
            }
            ImGui::Spacing();

            // ---- Voxel slider ----
            drawVoxelInfo();
            ImGui::Spacing();

            // ---- Ctrl+G / Ctrl+Shift+G RIM/raycast 関連コントロール ----
            if (actions.onDrawAdvancedCtrlG) {
                ImGui::PushID("##adv_ctrlg");
                actions.onDrawAdvancedCtrlG();
                ImGui::PopID();
            }

            ImGui::Unindent(8);
        }
```

With the merged version:

```cpp
        // ====================================================================
        //  Tuning & Advanced  (CollapsingHeader, 既定 OFF, Phase 9b)
        // --------------------------------------------------------------------
        //  Merges the old "QCR Tuning" + "Advanced" headers (Phase 8) into a
        //  single end-of-section header. Both contain debug/expert controls
        //  that should not occupy 2 separate rows in the main sidebar.
        //
        //  Layout (top to bottom):
        //    1. AutoQCR 6-DoF lock
        //    2. AutoQCR subset K + Max trials + rotation limits
        //    3. Ctrl+G 6-DoF lock
        //    4. Voxel slider (drawVoxelInfo)
        //    5. Ctrl+G / Ctrl+Shift+G RIM-raycast controls (callback to main.cpp)
        // ====================================================================
        if (ImGui::CollapsingHeader("Tuning & Advanced")) {
            ImGui::Indent(8);

            // ===== AutoQCR section =====
            ImGui::TextColored(colMuted(), "AutoQCR (Alt+Ctrl+P / Shift+Ctrl+P):");
            {
                bool lockScale = state.autoQcrLockScale;
                if (ImGui::Checkbox("AutoQCR 6-DoF lock (scale=1)##qcr_lock", &lockScale)) {
                    const_cast<RegUIState&>(state).autoQcrLockScale = lockScale;
                }
            }
            ImGui::Spacing();
            // K subset size: 3 (Fischler-Bolles min), 4-5 (more stable, over-determined)
            ImGui::TextColored(colMuted(), "Subset size K:");
            ImGui::SliderInt("##qcrK", &g_qcrSubsetK, 3, 5, "K = %d");
            ImGui::TextColored(ImVec4(0.45f,0.45f,0.5f,1),
                               "  K=3: exact fit  K=4: balanced  K=5: stable");
            ImGui::Spacing();
            ImGui::TextColored(colMuted(), "Max trials (Stage 1 cap):");
            ImGui::SliderInt("##qcrCap", &g_qcrMaxTrials, 10000, 500000, "%d");
            ImGui::Spacing();
            ImGui::TextColored(colMuted(), "Max axis rotation (per-axis):");
            ImGui::SliderFloat("##qcrMaxAxis", &g_qcrMaxAxisRotDeg,
                               5.0f, 90.0f, "%.1f deg");
            ImGui::TextColored(colMuted(), "Max total rotation (axis-angle):");
            ImGui::SliderFloat("##qcrMaxTotal", &g_qcrMaxTotalRotDeg,
                               5.0f, 90.0f, "%.1f deg");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // ===== Ctrl+G section =====
            ImGui::TextColored(colMuted(), "Ctrl+G / Ctrl+Shift+G (V3-R):");
            {
                bool lockScale = state.ctrlgLockScale;
                if (ImGui::Checkbox("Ctrl+G 6-DoF lock (scale=1)##ctrlg_lock_adv", &lockScale)) {
                    const_cast<RegUIState&>(state).ctrlgLockScale = lockScale;
                    if (actions.onCtrlgLockScaleChanged)
                        actions.onCtrlgLockScaleChanged(lockScale);
                }
            }
            ImGui::Spacing();
            drawVoxelInfo();
            ImGui::Spacing();
            if (actions.onDrawAdvancedCtrlG) {
                ImGui::PushID("##adv_ctrlg");
                actions.onDrawAdvancedCtrlG();
                ImGui::PopID();
            }

            ImGui::Unindent(8);
        }
```

### Verification (9b)

1. Build clean.
2. Sidebar has **one** collapsing header named `▸ Tuning & Advanced` near the bottom (above the 3-column footer).
3. `▸ QCR Tuning` is **gone** (no longer between AutoQCR and Ctrl+G).
4. Expand `▸ Tuning & Advanced`. See AutoQCR section (6-DoF lock + K + Max trials + axis limits), separator, then Ctrl+G section (6-DoF lock + Voxel slider + RIM/raycast controls).
5. Tick `AutoQCR 6-DoF lock`. Run AutoQCR. Confirm runs in 6-DoF mode.
6. Tick `Ctrl+G 6-DoF lock`. Run Ctrl+G. Confirm runs in 6-DoF mode.
7. Drag voxel slider. Debug Panel O tab voxel reflects (same global).
8. Sidebar total visible rows count: 7 (was 9 after Phase 8).

**Commit:** `[ui-refactor] Phase 9b: Merge QCR Tuning + Advanced into one collapsing header`

---

---

## Change 9c — Auto-collapse INITIAL ORIENTATION after Apply Init Pose

**File:** `RegistrationImGuiManager.h`
**Functions:** class member declaration + `drawInitOrientationPanel()` + `drawRegistrationSection()`

### Rationale

Currently the INITIAL ORIENTATION CollapsingHeader auto-opens on first frame (via `ImGuiCond_Once`) but never auto-closes. The user workflow is:

1. Set anatomical axes
2. Pick POSITION quadrants
3. Pick ORIENTATION preset
4. Press **Apply Init Pose** ← INITIAL ORIENTATION is no longer needed
5. Proceed to Hemi Quad → AutoQCR → Ctrl+G

Steps 1-3 happen inside INITIAL ORIENTATION; once step 4 is done, the panel just takes up screen space. Auto-fold after Apply.

### Step 9c.1 — Add member flag

**Anchor:** existing member flags at line ~302-304 (`bool regPhaseActive_ = false;`).

Add immediately after:

```cpp
    bool regPhaseActive_ = false;
    bool initOrientShouldCollapse_ = false;   // [Phase 9c] One-shot flag set by
                                              // Apply Init Pose button, consumed
                                              // by the next drawRegistrationSection
                                              // frame to force-fold INITIAL ORIENT.
```

### Step 9c.2 — Set the flag on Apply Init Pose click

**Anchor:** `drawInitOrientationPanel()`, line ~1494.

Find:

```cpp
            if (ImGui::Button("Apply Init Pose", ImVec2(-1, 24.0f))) {
                if (actions.onApplyInitPose) actions.onApplyInitPose();
            }
```

Replace with:

```cpp
            if (ImGui::Button("Apply Init Pose", ImVec2(-1, 24.0f))) {
                if (actions.onApplyInitPose) actions.onApplyInitPose();
                // [Phase 9c] Auto-fold INITIAL ORIENTATION on next frame.
                initOrientShouldCollapse_ = true;
            }
```

### Step 9c.3 — Consume the flag in drawRegistrationSection

**Anchor:** existing auto-open block in `drawRegistrationSection()`, line ~1571-1573:

```cpp
            // Auto-open before any registration / labels exist; fold up after.
            if (!state.useRegistration && !state.quadLabelsReady) {
                ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            }
```

Replace with:

```cpp
            // [Phase 9c] One-shot fold trigger from Apply Init Pose. Cleared on
            // consume so the user can manually re-open the header any time after.
            if (initOrientShouldCollapse_) {
                ImGui::SetNextItemOpen(false, ImGuiCond_Always);
                initOrientShouldCollapse_ = false;
            }
            // Auto-open before any registration / labels exist; fold up after.
            else if (!state.useRegistration && !state.quadLabelsReady) {
                ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            }
```

### Verification (9c)

1. Build clean.
2. Open REGISTRATION fresh. INITIAL ORIENTATION is auto-open (existing behavior).
3. Tick quadrants, pick orient, click **Apply Init Pose**.
4. INITIAL ORIENTATION folds to its header on the next frame.
5. Manually click the header to expand. It opens. (One-shot, not sticky.)
6. Click Apply Init Pose again (with different preset). Folds again.

**Commit:** `[ui-refactor] Phase 9c: Auto-collapse INITIAL ORIENTATION after Apply Init Pose`

---

## Change 9d — Move Apply Init Pose OUT of CollapsingHeader

**File:** `RegistrationImGuiManager.h`
**Functions:** `drawInitOrientationPanel()` + `drawRegistrationSection()`

### Rationale

The original Plan v2 specified Apply Init Pose as a separate full-width button OUTSIDE the INITIAL ORIENTATION CollapsingHeader. The Phase 8 implementation embedded it inside `drawInitOrientationPanel()`, so the button is only visible when the header is expanded.

Apply Init Pose IS the main action of this section — burying it inside a collapsible header is wrong:

- Workflow tells you to press it.
- After Phase 9c auto-fold fires, the button disappears with the rest of the header.
- If the user wants to re-Apply (different orient preset), they have to re-expand first.

Hoist Apply Init Pose to the sidebar level: configuration controls live inside the CollapsingHeader; the Apply button sits **below** the header, always visible.

### Step 9d.1 — Remove Apply Init Pose from `drawInitOrientationPanel()`

**Anchor:** `drawInitOrientationPanel()`, the `[§D] Apply Init Pose` block at line ~1475-1500.

Find:

```cpp
        // ============================================================
        //  [§D] Apply Init Pose (主動線、常時表示、最下部)
        // ============================================================
        //  POSITION / ORIENTATION を確定したあとにこのボタンで applyInitRotation
        //  を実行する。ORIENTATION が折りたたみのとき、現在値は ORIENTATION ヘッダ
        //  の "[Base]" 等で確認できる。
        ImGui::Spacing();
        {
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.10f, 0.30f, 0.55f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.15f, 0.40f, 0.70f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  ImVec4(0.20f, 0.50f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(0.95f, 0.97f, 1.00f, 1.0f));
            bool empty = (state.activeQuadrantMask == kMaskNone);
            if (empty) ImGui::BeginDisabled();
            if (ImGui::Button("Apply Init Pose", ImVec2(-1, 24.0f))) {
                if (actions.onApplyInitPose) actions.onApplyInitPose();
            }
            if (empty) ImGui::EndDisabled();
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar();
        }

        ImGui::Spacing();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
```

Replace with:

```cpp
        // ============================================================
        //  [§D] Apply Init Pose は drawRegistrationSection 側に移動
        //       (Phase 9d, CollapsingHeader の外で常時表示)
        // ============================================================

        ImGui::Spacing();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
```

### Step 9d.2 — Adjust `childH` calculation

The `childH` pre-computation around line 1155-1157 includes Apply button space; remove those lines.

Find:

```cpp
        childH += 6.0f                                 // spacing before Apply
                  + (24.0f + 6.0f)                       // apply button
                  + 8.0f;                                // tail padding
```

Replace with:

```cpp
        childH += 8.0f;                                // tail padding only
                                                       // (Phase 9d: Apply moved out)
```

### Step 9d.3 — Add Apply Init Pose to `drawRegistrationSection()` (outside the CollapsingHeader)

**Anchor:** the existing INITIAL ORIENTATION block in `drawRegistrationSection()` that wraps `drawInitOrientationPanel()` (line ~1555-1585).

Find the closing of that block:

```cpp
            if (ioOpen) {
                drawInitOrientationPanel();  // unchanged
            }
        }
```

Replace with:

```cpp
            if (ioOpen) {
                drawInitOrientationPanel();  // contents only (Apply moved out, Phase 9d)
            }
        }

        // [Phase 9d] Apply Init Pose — main-line button, ALWAYS visible (outside
        // the CollapsingHeader). Disabled if no quadrant is selected. Pressing
        // it also triggers the Phase 9c one-shot auto-fold on the next frame.
        {
            constexpr uint8_t kMaskNone_apply = 0x00;
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.10f, 0.30f, 0.55f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.15f, 0.40f, 0.70f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  ImVec4(0.20f, 0.50f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(0.95f, 0.97f, 1.00f, 1.0f));
            bool empty = (state.activeQuadrantMask == kMaskNone_apply);
            if (empty) ImGui::BeginDisabled();
            // Slightly taller than before (28 vs 24) so it visually anchors as
            // the main action button.
            if (ImGui::Button("Apply Init Pose", ImVec2(-1, 28.0f))) {
                if (actions.onApplyInitPose) actions.onApplyInitPose();
                initOrientShouldCollapse_ = true;   // [Phase 9c]
            }
            if (empty) ImGui::EndDisabled();
            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar();
        }
        ImGui::Spacing();
```

### Verification (9d)

1. Build clean.
2. Open REGISTRATION fresh. INITIAL ORIENTATION CollapsingHeader auto-opens. **Apply Init Pose button is visible immediately below the header content.**
3. Collapse the header manually. **Apply Init Pose stays visible.**
4. Tick a quadrant, click Apply. (a) Header folds (Phase 9c), (b) Apply button still visible below header.
5. Change orient preset via the folded `▸ INITIAL ORIENTATION [Q:AL | Right]` label (re-expand), pick different preset, fold again manually, click Apply again. Works.
6. With no quadrant selected (e.g. Quick preset "None"), Apply button is disabled (greyed out) — same behavior as before.

**Commit:** `[ui-refactor] Phase 9d: Hoist Apply Init Pose outside INITIAL ORIENTATION header`

### Note on relationship to Phase 9c

Phase 9c set `initOrientShouldCollapse_` inside the Apply Init Pose button click handler **while the button still lived in `drawInitOrientationPanel()`**. After Phase 9d the button moves to `drawRegistrationSection()` — the same flag-set line moves with it (see Step 9d.3). The collapse-consume logic in `drawRegistrationSection()` (Step 9c.3) is unchanged.

If implementing Phase 9c and 9d together in a single commit, you can skip Step 9c.2's edit in `drawInitOrientationPanel()` (since the button is being removed from there anyway) and just put the `initOrientShouldCollapse_ = true;` line directly in the new Apply button location in 9d.3.

---

## Combined verification

After 9a + 9b + 9c + 9d:

- [ ] Build clean.
- [ ] POSITION grid is 1 row of 4 checkboxes (was 2x2).
- [ ] No "Top=anterior, Left=patient's right" hint inside POSITION.
- [ ] Apply Init Pose button is OUTSIDE the CollapsingHeader (always visible).
- [ ] Apply Init Pose auto-folds INITIAL ORIENTATION on next frame.
- [ ] After auto-fold, Apply Init Pose button is STILL visible (key Phase 9d benefit).
- [ ] User can manually re-expand the header at any time.
- [ ] Sidebar REGISTRATION section is 8 rows (visible always):
   1. ▸ INITIAL ORIENTATION [Q:... | ...]
   2. [Apply Init Pose] (always visible, full-width, taller)
   3. [Hemi Quad (Shift+O)] [Probe]
   4. [AutoQCR]
   5. [Ctrl+G]
   6. [Umeyama Manual]
   7. [Pose Library (N)] [Pose Undo]
   8. ▸ Tuning & Advanced
   9. [Deform >>][<< Depth][Reset]
- [ ] All keyboard shortcuts still work.
- [ ] End-to-end registration produces same RMSE as before Phase 9.

---

## Optional: callback diff check

Since this phase only changes ImGui rendering (no callback wiring), the callback list should NOT change. Verify:

```bash
awk '/void drawRegistrationSection/,/^    void drawDeformSection/' \
    RegistrationImGuiManager.h \
    | grep -oE "actions\.\w+" | sort -u
```

Should match the Phase 8 baseline exactly. If anything is added or removed, investigate before commit.

**End of plan.**
