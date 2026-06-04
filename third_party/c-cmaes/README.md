# c-cmaes

CMA-ES for the registration BIPOP engines, structured as **vendored upstream +
a thin wrapper** so the two never get confused and the upstream stays trivially
updatable.

```
c-cmaes/
├── upstream/   ← Hansen c-cmaes, vendored VERBATIM — do not edit (original names)
│   ├── src/{cmaes.c, cmaes.h, cmaes_interface.h, boundary_transformation.c, boundary_transformation.h}
│   └── CMakeLists.txt, LICENSE, README.md, compile, doc.txt, docfunctions.txt  (upstream's own)
└── wrapper/    ← our integration layer (everything we wrote lives here)
    ├── cmaes.h               PUBLIC engine-facing API (the only header engines #include)
    ├── cmaes_adapter.cpp     implements cmaes.h on top of upstream
    ├── hansen_renames.h      AUTO-GENERATED: cmaes_* → HANSEN_*  (cmaes_boundary_* excluded)
    ├── hansen_unrenames.h    AUTO-GENERATED: cancels the above
    └── hansen_cmaes_renamed.c  C shim: renames + #include "../upstream/src/cmaes.c"
```

## upstream/ — Hansen c-cmaes (unmodified)

Source: <https://github.com/CMA-ES/c-cmaes> (Apache-2.0 / LGPL-2.1; see
`upstream/LICENSE`). Vendored **byte-for-byte** — file names *and* the in-source
`cmaes_*` symbol names are the original upstream ones. Nothing in `upstream/` is
edited, so it can be diffed against or re-synced with upstream at any time.

## wrapper/ — our adapter

Everything we wrote is here; nothing of ours leaks into `upstream/`.

* **API compatibility** — the engines call a 6-arg
  `cmaes_init(N, xstart, sigma0, lambda, lb, ub)`; upstream's signature differs.
  `wrapper/cmaes.h` keeps the old signature, so the engines change only their
  per-run seeding line (`srand` → `cmaes_set_seed`).
* **No symbol clash, upstream untouched** — upstream's `cmaes_*` symbols are
  renamed to `HANSEN_*` *at compile time* via preprocessor macros
  (`hansen_renames.h`, applied by the C shim that `#include`s the upstream
  `.c`). The upstream source files keep their original `cmaes_*` names; the
  renaming never touches them.
* **Box constraints** — via upstream's smooth `boundary_transformation`.

The build compiles only `wrapper/hansen_cmaes_renamed.c`,
`wrapper/cmaes_adapter.cpp`, and `upstream/src/boundary_transformation.c`.
Engines include `third_party/c-cmaes/wrapper/cmaes.h` explicitly; `upstream/src/`
is never placed on the include path, so there is no header collision between the
public `cmaes.h` and upstream's `cmaes.h`.

## Determinism (serial == parallel, bit-exact)

Run-level OpenMP parallelism (`g_v3rParallelRuns` / `g_v3rsParallelRuns`) is
reproducible because the wrapper:

1. reseeds upstream's **per-instance RNG** via `cmaes_set_seed`
   (`cmaes_random_Start`, no `rgrand` realloc);
2. copies `cmaes_TestForTermination`'s shared static return buffer out under an
   `omp critical` into a per-instance buffer;
3. disables the eigensystem update's **elapsed-CPU-time** gate
   (`updateCmode.maxtime = 1.0`) so it follows only the deterministic
   generation-based schedule — otherwise parallel CPU contention changes the
   update cadence and the search trajectory diverges.

`cmaes_init` also points `sp.filename` at `"non"` so upstream writes no
`actparcmaes.par` to the working directory.

## Updating upstream / regenerating renames

Replace the contents of `upstream/` with a fresh checkout, then regenerate
`wrapper/hansen_renames.h` and `hansen_unrenames.h` from every `cmaes_*` token
in `upstream/src/{cmaes.c,cmaes.h,cmaes_interface.h}`, excluding `cmaes_boundary_*`
(see the project's CMA-ES migration notes for the exact command).
