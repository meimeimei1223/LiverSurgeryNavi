# Keyboard reference — lsn_registration

> Updated after the `key-reorg` pass (2026-05-25).
> Visualization / debug toggles moved to **Ctrl+D Debug Panel > Viz tab**.
> All registration **action** keys are unchanged.

## Action keys (kept on keyboard)

### Registration
| Key | Action |
|---|---|
| `O` | HemiAuto (OrbitCam view) |
| `Shift+O` | QuadAuto (AR-fixed view) |

### BIPOP-CMA-ES (G family)
| Key | Method |
|---|---|
| `Alt+G` | V1 BIPOP-CMA-ES (was Shift+V) |
| `Alt+Shift+G` | V2 BIPOP-CMA-ES Fast (was Shift+F) |
| `Shift+G` | V3 BIPOP-CMA-ES (good performance) |
| `Ctrl+G` | V3-R region-aware (main path) |
| `Ctrl+Shift+G` | V3-RS silhouette anchor |

### Silhouette / Cyclic (P family)
| Key | Action |
|---|---|
| `P` | SilhouetteHemi |
| `Shift+P` | Cyclic Boundary |
| `Ctrl+P` | QuadCyclic |
| `Ctrl+Shift+P` | QuadCyclic-RANSAC |
| `Ctrl+Alt+P` | AutoQCR (recommended init) |

### Shape Match (W family) + refine + align
| Key | Action |
|---|---|
| `Ctrl+W` | Shape Match -> apply -> save |
| `Ctrl+Shift+W` | Shape Match -> axis sweep -> Live ICP bridge |
| `Alt+W` | Shape Match Coarse2D + GN refine |
| `Ctrl+Alt+W` | Contour / Silhouette sweep |
| `Shift+N` | Normal-Compatible refine |
| `Ctrl+Shift+N` | SRT-variance refine |
| `Shift+E` | Silhouette Align (2D BIPOP) |

### Camera / IO
| Key | Action |
|---|---|
| `R` | Run depth (image mode) |
| `K` | Camera depth estimation |
| `J` | Save camera frame |
| `S` | Snapshot |
| `L` | Live view |
| `M` / `Shift+M` | Export STL / Export STL with X+Z flip |

### Pose / display / tuning
| Key | Action |
|---|---|
| `Q` | Pose Library window |
| `X` | Pose Undo |
| `D` | AR save snapshot |
| `Ctrl+D` | Debug Panel window |
| `F2` | Camera reset |
| `A` | AR background overlay |
| `F9` | Silhouette IoU overlay window |
| `F10` | Vertex-squash diagnose (also in Debug Panel) |
| `Up` / `Down` | g_voxelSize +/- 0.05 |
| `,` / `.` | silhouette threshold -/+ 0.01 (Shift: 0.05) |
| `U` / `C` | Mask picker undo / clear (image-only mode) |
| `Esc` | Close |

## Removed keys -> Ctrl+D > Viz tab

These keyboard toggles were removed; use the checkboxes/buttons in the Debug
Panel Viz tab instead:

| Old key | Now in Debug Panel Viz tab |
|---|---|
| `V` | Cluster visualization |
| `B` | Boundary candidates |
| `Shift+B` | Cyclic correspondence |
| `N` (plain) | Source visualization |
| `W` (plain) | Debug source rim chain |
| `Shift+W` | Debug target boundary |
| `Shift+R` | Liver Region viz |
| `Y` | Liver Left/Right viz |
| `H` | Liver 4-Quadrant viz |
| `Shift+H` | Liver Cranio/Caudal viz |
| `Shift+T` | "Recompute Region" button |
| `Shift+Y` | "Recompute LR" button |
| `Shift+I` | "Dump IoU debug PNG" button |

## Deferred (future cleanup)

- Camera/IO keys (`R/K/J/S/L/M`) and `Q/X/D/F2/A/U/C` grouping — to be tidied later.
- `Shift+E` (Silhouette Align) is currently a standalone key; consider folding it
  into the G or P family.
