/* cmaes.h
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
 * Author and copyright: Nikolaus Hansen, 2014
 * Licensed under Apache License 2.0
 * https://github.com/CMA-ES/c-cmaes
 *
 * Faithfully reproduced for use in AAA_LiverSurgeryNavi_ver2
 * under Apache License 2.0.
 */
#pragma once
#ifndef CMAES_H
#define CMAES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct {
    int    N;           /* problem dimension */
    double sigma;       /* step size */
    double *xmean;      /* current mean */
    double *xbest;      /* best solution found */
    double fbest;       /* best function value */

    /* internal state */
    double *pc;         /* evolution path for C */
    double *ps;         /* evolution path for sigma */
    double **B;         /* eigenvectors of C */
    double *D;          /* sqrt of eigenvalues */
    double **C;         /* covariance matrix */
    double *invsqrtC;   /* diagonal of C^{-1/2} in principal axes */
    double *artmp;      /* temporary */
    double **arx;       /* sampled population */
    double **arfitness; /* not used directly; fitness stored in arfval */
    double *arfval;     /* fitness values of population */
    double *weights;    /* recombination weights */
    double *xold;       /* previous mean */

    int    lambda;      /* population size */
    int    mu;          /* number of parents */
    double mueff;       /* variance-effectiveness */
    double cc;          /* time constant for pc */
    double cs;          /* time constant for ps */
    double c1;          /* learning rate for rank-one update */
    double cmu;         /* learning rate for rank-mu update */
    double damps;       /* damping for sigma */
    double chiN;        /* expectation of ||N(0,I)|| */

    int    eigeneval;   /* track when to update B,D */
    int    counteval;   /* function evaluation counter */
    int    gen;         /* generation counter */

    double *lbounds;    /* lower bounds (optional, may be NULL) */
    double *ubounds;    /* upper bounds (optional, may be NULL) */
} cmaes_t;

/* Allocate and initialize.
 * xstart: initial mean (length N), may be NULL → zeros
 * sigma0: initial step size
 * lambda: population size (0 → default 4+floor(3*ln(N)))
 * lbounds/ubounds: box constraints, may be NULL
 */
cmaes_t* cmaes_init(int N, const double *xstart, double sigma0,
                    int lambda,
                    const double *lbounds, const double *ubounds);

/* Sample lambda candidate solutions into pop (N x lambda).
 * Returns pointer to internal arx (do not free). */
double** cmaes_SamplePopulation(cmaes_t *evo);

/* Update distribution from fitness values fval[0..lambda-1]. */
void cmaes_UpdateDistribution(cmaes_t *evo, const double *fval);

/* Returns pointer to best-so-far solution (length N). */
const double* cmaes_GetPtr(const cmaes_t *evo, const char *s);

/* Free all memory. */
void cmaes_exit(cmaes_t *evo);

/* Convenience: check stop conditions.
 * Returns non-zero string description when stop, NULL otherwise. */
const char* cmaes_TestForTermination(const cmaes_t *evo,
                                     int maxgen, double tolfun, double tolx);

#ifdef __cplusplus
}
#endif

#endif /* CMAES_H */
