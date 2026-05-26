# Keyboard reference — lsn_registration

> Updated after the `key-reorg` pass, Phases 1–13 (2026-05-26).
> Only **action / display / tuning** keys remain on the keyboard.
> Visualization & debug toggles live in **Ctrl+D Debug Panel > Viz tab**;
> camera / depth / export functions live in **sidebar buttons**.

## Action keys (kept)

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
| `Shift+G` | V3 BIPOP-CMA-ES |
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
| `Alt+P` | Silhouette Align (was Shift+E) |

### Shape Match (W family) + refine
| Key | Action |
|---|---|
| `Ctrl+W` | Shape Match -> apply -> save |
| `Ctrl+Shift+W` | Shape Match -> axis sweep -> Live ICP bridge |
| `Alt+W` | Shape Match Coarse2D + GN refine |
| `Ctrl+Alt+W` | Contour / Silhouette sweep |
| `Shift+N` | Normal-Compatible refine |
| `Ctrl+Shift+N` | SRT-variance refine |

### Display / pose / tuning
| Key | Action |
|---|---|
| `Q` | Pose Library window |
| `X` | Pose Undo |
| `D` | AR save snapshot |
| `Ctrl+D` | Debug Panel window |
| `A` | AR background overlay |
| `Up` / `Down` | g_voxelSize +/- 0.05 |
| `,` / `.` | silhouette threshold -/+ 0.01 (Shift: 0.05) |
| `U` / `C` | Mask picker undo / clear (image-only mode) |
| `Esc` | Close |

## Removed keys -> UI

Camera / depth / export / diagnostics moved to UI:

| Old key | Now in |
|---|---|
| `R`, `K` | sidebar "Run Depth" button (same button in image / camera mode) |
| `S` | sidebar camera toggle ("Capture" state) |
| `L` | sidebar camera toggle / "Re-Capture" |
| `F2` | sidebar "Cam Init" button |
| `F9` | Debug Panel > Viz tab "Show Silhouette Overlay window" checkbox |
| `F10` | Debug Panel > Viz tab "Vertex-Squash diagnose" button |
| `J` | (deleted — `camera_frame_temp.jpg` written by "Run Depth") |
| `M` | sidebar Export > "Export Reg OBJs" button |
| `Shift+M` | sidebar Export > "Export cam-mm STL" button |
| `Shift+E` | moved to `Alt+P` |

Visualization toggles moved to **Debug Panel > Viz tab**:

| Old key | Now |
|---|---|
| `V` | "Cluster visualization" (via "Cluster markers") |
| `B` | "Boundary candidates" checkbox |
| `Shift+B` | "Cyclic correspondence" checkbox |
| `N` (plain) | "Source visualization" checkbox |
| `W` (plain) | "Debug source rim chain" checkbox |
| `Shift+W` | "Debug target boundary" checkbox |
| `Shift+R` | "Liver Region viz" checkbox |
| `Y` | "Liver Left/Right viz" checkbox |
| `H` | "Liver 4-Quadrant viz" checkbox |
| `Shift+H` | "Liver Cranio/Caudal viz" checkbox |
| `Shift+T` | "Recompute Region" button |
| `Shift+Y` | "Recompute LR" button |
| `Shift+I` | "Dump IoU debug PNG" button |

## Final keyboard cases (16)

`A, C, D, G, N, O, P, Q, U, W, X, Up, Down, `,`, `.`, Esc`
(plus modifier combinations on G / O / P / W / N as listed above).
