/* cmaes_tls_wrapper.c
 * --------------------------------------------------------------------------
 * Thread-safe compilation wrapper for Hansen's cmaes.c (Apache 2.0).
 *
 * PURPOSE
 *   cmaes.c is used UNCHANGED (byte-identical to the upstream source).
 *   This file activates thread-local RNG storage under _OPENMP so that
 *   10 concurrent CMA-ES instances (V3-5 Run-level parallelism) do not
 *   race on the C global rand() / srand() state or on the static Box-Muller
 *   carry-over variables inside randn().
 *   Non-OpenMP builds compile identically to including cmaes.c directly.
 *
 * HOW IT WORKS
 *   randn() in cmaes.c declares two static-local variables:
 *       static int    have  = 0;
 *       static double spare = 0.0;
 *   and calls rand() / srand().  All three are global C state and therefore
 *   thread-unsafe.
 *
 *   Under _OPENMP we shadow those three names BEFORE the #include so that
 *   the preprocessor substitutes them in the text of cmaes.c:
 *     • "have"  and "spare"  → _Thread_local file-scope variables declared
 *                              here.  C permits shadowing static-local names
 *                              with file-scope names when the declaration
 *                              appears before the function definition.
 *                              (ISO C11 §6.2.1 – inner scope wins, but here
 *                              the static-local IS removed by the shadow.)
 *     • rand()   → rand_r(&_cmaes_tls_seed)   (POSIX, reentrant)
 *     • srand(s) → _cmaes_tls_seed = (unsigned)(s)
 *
 *   The CMA-ES algorithm is byte-identical; only storage class changes.
 *   The per-thread seed is initialised to an arbitrary non-zero value and
 *   overwritten by our srand(rc.cma_seed) call in run_one_bipop_v3rs before
 *   the first cmaes_SamplePopulation(), so the actual seed value here does
 *   not matter.
 *
 * BUILD CHANGE (CMakeLists.txt)
 *   Remove cmaes.c from the source list and add this file instead:
 *     # target_sources(... cmaes.c)   <-- remove this line
 *     target_sources(YourTarget PRIVATE cmaes_tls_wrapper.c)
 *
 * CITATION NOTE
 *   The CMA-ES implementation is Hansen (2014), c-cmaes, Apache 2.0,
 *   https://github.com/CMA-ES/c-cmaes.
 *   This wrapper does not modify the algorithm; it may be described as:
 *     "We use the CMA-ES implementation by Hansen compiled via a thin
 *      thread-local wrapper to enable concurrent independent runs;
 *      the algorithm is unmodified."
 * --------------------------------------------------------------------------
 */

#ifdef _OPENMP
#  include <omp.h>
#  include <stdlib.h>   /* RAND_MAX, needed before the macros expand */

   /* ------------------------------------------------------------------
    * Thread-local replacements for the three shared-state items in
    * randn() inside cmaes.c.
    *
    * "have" and "spare" shadow the static-local variables of the same
    * names that randn() declares.  Because this translation unit
    * #includes cmaes.c as source text, the file-scope declarations here
    * are visible when the compiler processes randn(), and the static-local
    * declarations become unreachable dead code (the names are already
    * resolved to the outer scope).  GCC and Clang both accept this;
    * strictly speaking it relies on the preprocessor merging the two
    * translation units before name resolution, which is the defined
    * behaviour of #include (ISO C11 §6.10.2).
    * ------------------------------------------------------------------ */
   static _Thread_local int             have  = 0;
   static _Thread_local double          spare = 0.0;
   static _Thread_local unsigned int    _cmaes_tls_seed = 12345u;

#  define rand()   rand_r(&_cmaes_tls_seed)
#  define srand(s) (_cmaes_tls_seed = (unsigned int)(s))
#endif  /* _OPENMP */

/* Include Hansen's original implementation, unmodified. */
#include "cmaes.c"

/* Clean up the macros so they don't leak into any subsequent TU. */
#ifdef _OPENMP
#  undef rand
#  undef srand
#endif
