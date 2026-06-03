/* cmaes.h — engine-facing CMA-ES API (thin shim over upstream Hansen c-cmaes)
 *
 * This header preserves the exact 6-argument API the registration engines
 * (V1/V2/V3/V3R/V3RS) were written against, so they compile unchanged except
 * for the per-run seeding call (srand -> cmaes_set_seed).
 *
 * The implementation lives in cmaes_adapter.cpp and forwards to the upstream
 * Hansen library compiled in hansen_cmaes_renamed.c (upstream/src/). The upstream API
 * (cmaes_init with inseed/filename, boundary_transformation.*, per-instance
 * RNG, ...) is hidden behind `impl`.
 *
 * Engines read only gen/lambda/sigma off this struct (verified across all five
 * engines); the adapter mirrors those after every Sample/Update.
 */
#pragma once
#ifndef CMAES_H
#define CMAES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Engine-facing CMA-ES object. Only gen/lambda/sigma are read by callers;
 * `impl` points at the hidden AdapterImpl (upstream cmaes_t + boundary xform). */
typedef struct cmaes_s {
    int    gen;     /* generation counter  (mirror of upstream gen)  */
    int    lambda;  /* effective population size (set at init)       */
    double sigma;   /* step size           (mirror of upstream sigma) */
    void  *impl;    /* AdapterImpl* — opaque                          */
} cmaes_t;

/* Allocate and initialize one CMA-ES run.
 *   xstart : initial mean (length N), may be NULL -> zeros
 *   sigma0 : initial step size (applied to every coordinate)
 *   lambda : population size (0 -> upstream default 4+floor(3*ln N))
 *   lbounds/ubounds : box constraints (length N), may both be NULL -> unbounded.
 *     Bounds are enforced by upstream's smooth boundary transformation
 *     (not hard clamping); the values are copied internally.
 */
cmaes_t* cmaes_init(int N, const double *xstart, double sigma0,
                    int lambda,
                    const double *lbounds, const double *ubounds);

/* Install a deterministic per-run seed (replaces the old srand() override).
 * Uses the upstream per-instance RNG (cmaes_random_Start) so runs are
 * reproducible and parallel == serial. seed 0 is mapped to 1. Call once,
 * after cmaes_init() and before the first cmaes_SamplePopulation(). */
void cmaes_set_seed(cmaes_t *evo, unsigned int seed);

/* Sample lambda candidate solutions. Returns an internal lambda x N buffer of
 * box-constrained (bounded) candidates; do not free. Valid until the next
 * call or cmaes_exit(). */
double** cmaes_SamplePopulation(cmaes_t *evo);

/* Update the distribution from fitness values fval[0..lambda-1] (evaluated on
 * the bounded candidates returned by cmaes_SamplePopulation). */
void cmaes_UpdateDistribution(cmaes_t *evo, const double *fval);

/* Upstream getter passthrough, e.g. "xbestever" (UNBOUNDED). Provided for API
 * completeness; the engines extract best from the population+fitness instead. */
const double* cmaes_GetPtr(const cmaes_t *evo, const char *s);

/* Free all memory owned by this run. */
void cmaes_exit(cmaes_t *evo);

/* Stop-condition check. maxgen/tolfun/tolx are forwarded to the upstream
 * stop parameters (stopMaxIter/stopTolFun/stopTolX); upstream's other default
 * criteria also apply, so the returned reason string and generation count may
 * differ from the previous implementation (expected — see HANDOFF v2 §7).
 * Returns a description string when stopping, NULL otherwise. */
const char* cmaes_TestForTermination(const cmaes_t *evo,
                                     int maxgen, double tolfun, double tolx);

#ifdef __cplusplus
}
#endif

#endif /* CMAES_H */
