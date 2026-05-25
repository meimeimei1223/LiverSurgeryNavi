# Key Reorganization — Implementation Report

**Date:** 2026-05-25
**Branch:** `key-reorg` (off `main`, after the ui-refactor merge)
**Plan:** `KEY_REORG_PLAN.md`
**Files touched:** `registration/main.cpp`, `registration/src/DebugPanel.h`, `docs/`

---

## Goal

Tidy the `lsn_registration` keyboard: keep every **action** key, move all
**visualization / debug toggle** keys into the `Ctrl+D` Debug Panel, and
consolidate the scattered BIPOP-CMA-ES shortcuts into the G family. Algorithm
dispatch logic is unchanged — only the entry points were reorganized.

---

## What was implemented

| Phase | Title | Commit | Build |
|---|---|---|---|
| 1 | Viz/recompute/debug controls added to Ctrl+D > Viz tab | `3c8cc07` | OK |
| 2 | `Alt+G` (V1) and `Alt+Shift+G` (V2) BIPOP shortcuts | `1528486` | OK |
| 4+5 | Remove migrated viz/debug keys + legacy `Shift+V`/`Shift+F` | `e766b09` | OK |
| 7 | `docs/KEY_REFERENCE.md` + DebugPanel header comment | (latest) | OK |

**Phase 3 (deprecation warnings) was skipped** by user decision — went straight
to removal instead of the soak-gated deprecate-then-remove rollout.

### Phase 1 — Debug Panel Viz tab
Extended the existing `g_debugPanel.drawVizExtra` hook lambda (main.cpp) with the
toggles/buttons that used to be keyboard-only:
- Checkboxes: cluster viz, cyclic correspondence, source rim chain, target
  boundary, Liver Region / Left-Right / Cranio-Caudal / 4-Quadrant (with the
  same auto-recompute-on-first-enable behavior as the old keys).
- Buttons: Recompute Region, Recompute LR, Dump IoU debug PNG, Vertex-Squash
  diagnose.
- Relabeled the existing "Boundary candidates / Source visualization" to
  "(was B/N)".

### Phase 2 — BIPOP keys to the G family
Added `isAltG` / `isAltShiftG` (and to `needsScene`). In the `GLFW_KEY_G`
switch, `Alt+G` runs V1 (`runBipopCmaes`) and `Alt+Shift+G` runs V2
(`runBipopCmaesV2`) — byte-identical to the old `Shift+V` / `Shift+F` branches
(same `g_stepStartTime` / `g_sessionBipopN++` / `regMethod=3` / `poseAutoSave` /
run / `poseSaveToLibrary` sequence). Placed before the `Ctrl+Shift+G` branch for
correct modifier precedence.

### Phase 4+5 — remove migrated keys
- Full case removal: `V`, `F`, `B`, `T`, `Y`, `H`, `I`.
- Partial removal (action branches kept): `N` (plain source-vis only),
  `W` (plain rim-chain + `Shift+W` target-boundary only), `R` (`Shift+R`
  region-viz only — plain `R` run-depth kept).
- Removed `isShiftV` / `isShiftF` flags and their `needsScene` entries.
- The viz globals and the rendering that reads them are unchanged; they are now
  toggled only from the Debug Panel.

---

## Final key layout

Action keys are unchanged (O/Shift+O, G-family BIPOP incl. new Alt+G/Alt+Shift+G,
P-family, W-family actions, Shift+N/Ctrl+Shift+N, Shift+E, camera/IO, Q/X/D/
Ctrl+D/F2/A/F9/F10/arrows/`,`/`.`/U/C). Removed viz/debug toggles now live in
**Ctrl+D > Viz tab**. Full table: `docs/KEY_REFERENCE.md`.

---

## Verification

- Every phase builds clean (`lsn_registration`, 0 errors).
- Post-removal: 0 remaining `case GLFW_KEY_{V,F,B,T,Y,H,I}`.
- BIPOP V1/V2 logic copied byte-identical into the Alt+G branches.
- GUI behavior NOT yet verified — needs the app (Phase 8). Confirm in Qt Creator:
  Ctrl+D Viz tab toggles each viz on/off; Alt+G/Alt+Shift+G reproduce old
  Shift+V/Shift+F results; all action keys still work.

Note: `build/` is the CLI verification dir; Qt Creator uses `build/Desktop-*`.
If Qt's cache was cleared, Run CMake to reconfigure.

---

## Deferred / next time

- **Phase 6 (optional, cosmetic):** group the now-long Viz tab with
  CollapsingHeaders; also de-duplicate the cluster checkbox (it currently appears
  twice — once via `drawTabViz`'s onToggleClusterVis callback and once via the
  Phase-1 direct `g_showClusterVisualization` checkbox).
- **Camera/IO key grouping** (`R/K/J/S/L/M`, `Q/X/D/F2/A/U/C`) — user-flagged for
  later tidy.
- **`Shift+E`** (Silhouette Align) is a standalone key; consider folding it into
  the G or P family.
- **Phase 8:** full GUI acceptance test.

Branch `key-reorg` is local only (not merged to `main`, not pushed).
