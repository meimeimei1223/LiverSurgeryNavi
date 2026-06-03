# c-cmaes integration

CMA-ES for the registration BIPOP engines (V1/V2/V3/V3R/V3RS), built on the
**upstream Hansen c-cmaes** behind a thin engine-facing adapter.

## Layout

```
c-cmaes/
├─ cmaes.h                 ← PUBLIC engine-facing API (the only header engines #include)
│                            6-arg cmaes_init + cmaes_set_seed; struct exposes gen/lambda/sigma
├─ cmaes_adapter.cpp       ← adapter: implements cmaes.h on top of upstream
├─ hansen_renames.h        ← AUTO-GENERATED: cmaes_* → HANSEN_*  (cmaes_boundary_* excluded)
├─ hansen_unrenames.h      ← AUTO-GENERATED: cancels the above
├─ hansen_cmaes_renamed.c  ← C shim: renames + #include "upstream/src/cmaes.c"
└─ upstream/               ← UPSTREAM Hansen c-cmaes, vendored verbatim — DO NOT EDIT
   ├─ src/{cmaes.c,cmaes.h,cmaes_interface.h,boundary_transformation.c,boundary_transformation.h}
   └─ CMakeLists.txt, LICENSE, README.md, compile, doc.txt, docfunctions.txt  (upstream's own)
```

Source: https://github.com/CMA-ES/c-cmaes (Apache-2.0 / LGPL-2.1). The files in
`upstream/` are byte-identical to upstream; integration is done entirely in the
root files. The top-level CMakeLists compiles only `hansen_cmaes_renamed.c`,
`upstream/src/boundary_transformation.c`, and `cmaes_adapter.cpp`. The engine
include path is this directory (root) only, so engines resolve `cmaes.h` to the
public adapter header — never `upstream/src/cmaes.h`.

## Why an adapter

* **API compatibility** — the engines were written against a 6-arg
  `cmaes_init(N, xstart, sigma0, lambda, lb, ub)`; upstream's is different. The
  adapter keeps the old signature so engines change only their per-run seeding
  line (`srand` → `cmaes_set_seed`).
* **No symbol clash** — upstream `cmaes_*` symbols are renamed to `HANSEN_*` via
  preprocessor macros (the C shim), so the public `cmaes_*` API can coexist.
  Renaming is macro-only; upstream sources are untouched.
* **Box constraints** — via upstream's smooth `boundary_transformation` (the
  adapter evaluates on bounded x; upstream updates from its internal unbounded
  population).

## Determinism (serial == parallel)

Run-level OpenMP parallelism (`g_v3rParallelRuns` / `g_v3rsParallelRuns`) is
bit-for-bit reproducible because:

1. Each run uses upstream's **per-instance RNG**, reseeded via `cmaes_set_seed`
   (`cmaes_random_Start`, no `rgrand` realloc).
2. `cmaes_TestForTermination`'s shared static return buffer is copied out under
   an `omp critical` into a per-instance buffer.
3. The eigensystem update is forced off its **elapsed-CPU-time** gate
   (`updateCmode.maxtime = 1.0`) so it follows only the deterministic
   generation-based schedule — otherwise parallel CPU contention changes the
   update cadence and the search trajectory diverges.

`cmaes_init` also points `sp.filename` at `"non"` so upstream writes no
`actparcmaes.par` to the working directory.

## Regenerating the rename headers

`hansen_renames.h` / `hansen_unrenames.h` are generated from every `cmaes_*`
token in `upstream/src/{cmaes.c,cmaes.h,cmaes_interface.h}`, excluding
`cmaes_boundary_*`. If upstream is updated, regenerate them (see the generation
command in the project's CMA-ES migration notes).
