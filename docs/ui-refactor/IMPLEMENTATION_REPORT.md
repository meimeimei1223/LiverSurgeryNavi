# Registration UI Refactor — Implementation Report

**Date:** 2026-05-25
**Branch:** `ui-refactor`
**Plan:** `docs/ui-refactor/UI_REFACTOR_PLAN.md`

---

## Summary

Implemented the core of the UI refactor: the **three always-on floating debug
panels are now consolidated into a single tabbed `Debug Panel` (Ctrl+D)**, and
the misrouted `Clear CorresPoints` button is removed. Each phase compiles clean
and is a separate commit. GUI behavior verification is pending on real hardware
(the plan's per-phase checks require launching the app, loading an image,
running DEPTH, clicking — done by the user in Qt Creator).

---

## Completed phases

| Phase | Title | Commit | Build |
|---|---|---|---|
| baseline | 3-project split / sam2 rename / fixes / UI defaults snapshot | `f4cffc1` | OK |
| 0 | Remove misrouted `Clear CorresPoints` button | `82169be` | OK |
| 1 | `DebugPanel.h` skeleton + Ctrl+D toggle (6 tabs) | `6e79d1a` | OK |
| 2 | Viz tab — migrate Cluster / CorresPoints toggles | `c70dfbf` | OK |
| 3 | G tab — move **full** Ctrl+G Quadrant Selector (~1680 lines) | `f40827b` | OK |
| 4 | N tab — move Normal-Compatible Refine | `6824f6a` | OK |
| 7 | Viz tab — move ScreenMesh Display + surface B/N viz | `2ce247e` | OK |

**Goal achieved:** all three legacy always-on floating panels
(`Ctrl+G Quadrant Selector`, `Normal-Compatible Refine`, `ScreenMesh Display`)
are removed from the always-on render path and live under Ctrl+D.

---

## Key design decision — the "hook" pattern (deviation from the plan)

The plan proposed lifting each floating panel's body **verbatim into
`DebugPanel.h`**. That is unworkable here: the panel bodies reference hundreds
of `main.cpp`-local statics, globals, and helpers (e.g. the Ctrl+G panel alone
is ~1680 lines, far larger than the plan's "~200 line" estimate) that are not
visible from a separate header, and several are `static` (internal linkage).

Instead each migrated panel body is **kept in place in `main.cpp`** and
registered as a `std::function<void()>` hook on `DebugPanel::State`:

```cpp
// DebugPanel::State
std::function<void()> drawGBody;     // Phase 3
std::function<void()> drawNBody;     // Phase 4
std::function<void()> drawVizExtra;  // Phase 7

// main.cpp frame loop (where all symbols are in scope):
g_debugPanel.drawGBody = [&]() { /* original panel body, unchanged */ };

// DebugPanel::draw() G tab:
if (st.drawGBody) st.drawGBody(); else drawTabG(...);
```

The `[&]` lambda captures frame-loop locals by reference, so the body compiles
unchanged. This **preserves 100% of the existing functionality** (per the user's
explicit instruction: "keep all functionality, debug from the sub-panel; the
sub-panel can be as large as it needs"). It is also more efficient than before:
the panel body now renders only when its tab is open, not every frame.

---

## Plan-vs-reality divergences found

1. **Ctrl+G Quadrant Selector size** — plan said ~200 lines; actual is
   ~1680 lines (main.cpp 4470–6154) containing search-mode radio, RIM/raycast
   knobs, overlay-probe locks, reshuffle, tables, etc. Handled by moving the
   whole body (hook), not deleting/reducing.

2. **Phase 5 (W tab) premise is wrong** — the plan's six flags
   `g_debugShow2DProjPopup_RawRim` … do **not exist**. The real W-key family is
   *RIM Shape Match* (W = source overlay, Shift+W = target overlay, Ctrl+W =
   shape match, Ctrl+Shift+W = sweep, Alt+W = GN refine, Ctrl+Alt+W = contour
   sweep). The actual RIM debug popups are three windows gated by
   silhouette-sweep state: `silsw_rawrim_popup`, `silsw_src_popup`,
   `silsw_tgt_popup` (main.cpp ~7388 / 8758 / 8940). Phase 5 was **not**
   implemented as specified.

---

## Phases 5/6/8 — completed (per UI_REFACTOR_PLAN_v2.md)

All remaining phases are now implemented (plan v2's reconnaissance was accurate
— every flag/field/callback verified to exist before writing):

| Phase | Title | Commit | Notes / deviations |
|---|---|---|---|
| **5** | W tab — silhouette-sweep RIM popup toggles + F9 | `b345bad` | Globals reached via `extern` (inline at global scope in RegistrationActions.h, included after DebugPanel.h). **Deviation:** `g_silOverlay` is namespace-scoped (`SilOverlay::`), so used qualified via the SilOverlayDebug.h include — NOT plan v2's (wrong) global extern. F9 button also added to the G-tab hook (`##g_f9`). |
| **6** | O tab (HemiAuto knobs) + U tab (Umeyama stats) | `afc9823` | Pure `RegUIState`/`RegUIActions`; no externs/hooks. Also retired the unreachable drawTabG/drawTabN "will populate" fallback text. |
| **8** | Main sidebar final layout cleanup | `b69c0ea` | Callback pre/post diff matched exactly (`-onHemiAuto -onInstrumentPxThreshChanged -onToggleClusterVis +onQuadAuto`). **Deviation:** AutoQCR/Ctrl+G are full-width buttons with the 6-DoF lock inside the existing collapsing headers, instead of plan v2's fragile side-by-side button+CollapsingHeader layout (ImGui aligns a tall button poorly beside a header). Same functionality, robust layout. |

All 6 Debug Panel tabs (G/O/N/W/U/Viz) now show real content; no "will populate"
placeholders remain. The refactor (Phases 0–8) is functionally complete pending
GUI verification.

---

## Verification status

- **Compile:** every phase builds clean (`lsn_registration`, 0 errors) in `build/`.
- **GUI behavior:** NOT yet verified (needs the app + camera/sample image). Per
  the plan's verification protocol, please confirm in Qt Creator:
  - Ctrl+D opens the Debug Panel; G/N/Viz tabs show the migrated controls.
  - The three legacy floating panels no longer appear always-on.
  - Apply Init Pose → Shift+O → AutoQCR → Ctrl+G → Shift+N still completes with
    the same final RMSE as before.
  - Umeyama 2-screen mode hides the Debug Panel.
  - `Clear CorresPoints` button is gone; `Reset Reg` is full-width.

---

## How to continue

```
git checkout ui-refactor
# implement Phase 5 (re-scoped), then 6, then 8, one commit each:
#   [ui-refactor] Phase N: <summary>
```

Use the hook pattern for any tab whose content touches `main.cpp` locals/globals;
use direct `RegUIState`/`RegUIActions` (passed into `drawTab*`) for content that
only needs already-synced UI state.
