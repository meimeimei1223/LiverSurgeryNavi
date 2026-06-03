/* hansen_cmaes_renamed.c — C shim (HANDOFF v2 FIX3)
 *
 * Compiles the upstream Hansen c-cmaes (src/cmaes.c) with every external
 * cmaes_* symbol prefixed to HANSEN_ via hansen_renames.h, so it links
 * alongside the thin engine-facing API defined in cmaes_adapter.cpp / cmaes.h
 * without symbol clashes.
 *
 * The upstream source under src/ is byte-identical to upstream; the renaming
 * is done purely through preprocessor macros here. This translation unit is
 * compiled as C (never C++), keeping the ANSI-C body away from C++ implicit
 * void* conversion issues.
 *
 * Do not add anything else here.
 */
#include "hansen_renames.h"
#include "src/cmaes.c"
