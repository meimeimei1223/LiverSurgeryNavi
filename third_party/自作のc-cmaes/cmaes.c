/* cmaes.c
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
 * Author and copyright: Nikolaus Hansen, 2014
 * Licensed under Apache License 2.0
 * https://github.com/CMA-ES/c-cmaes
 *
 * This implementation follows the tutorial:
 * N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772
 */

#include "cmaes.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

static double* alloc1d(int n) {
    double *p = (double*)calloc(n, sizeof(double));
    if (!p) { fprintf(stderr, "[cmaes] alloc failed\n"); exit(1); }
    return p;
}

static double** alloc2d(int rows, int cols) {
    double **p = (double**)malloc(rows * sizeof(double*));
    if (!p) { fprintf(stderr, "[cmaes] alloc2d failed\n"); exit(1); }
    for (int i = 0; i < rows; i++) {
        p[i] = (double*)calloc(cols, sizeof(double));
        if (!p[i]) { fprintf(stderr, "[cmaes] alloc2d row failed\n"); exit(1); }
    }
    return p;
}

static void free2d(double **p, int rows) {
    if (!p) return;
    for (int i = 0; i < rows; i++) free(p[i]);
    free(p);
}

/* Box-Muller normal sample */
static double randn(void) {
    static int    have = 0;
    static double spare = 0.0;
    if (have) { have = 0; return spare; }
    double u, v, s;
    do {
        u = 2.0 * rand() / (double)RAND_MAX - 1.0;
        v = 2.0 * rand() / (double)RAND_MAX - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);
    double mul = sqrt(-2.0 * log(s) / s);
    spare = v * mul; have = 1;
    return u * mul;
}

/* ------------------------------------------------------------------ */
/* Symmetric eigendecomposition (Jacobi method)                        */
/* C = B * diag(D^2) * B^T,  D[i] = sqrt of eigenvalue                */
/* ------------------------------------------------------------------ */
static void eigen_sym(int n, double **C, double *d, double **V) {
    /* Copy C into V */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V[i][j] = C[i][j];

    /* initialise d to diagonal */
    for (int i = 0; i < n; i++) d[i] = V[i][i];

    /* Jacobi sweeps */
    double *e = alloc1d(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            V[i][j] = (i == j) ? 1.0 : 0.0;
    }
    /* rebuild d from diagonal of C copy */
    double **A = alloc2d(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = C[i][j];
    for (int i = 0; i < n; i++) d[i] = A[i][i];

    int nrot = 0;
    for (int sweep = 0; sweep < 50*n; sweep++) {
        double offdiag = 0.0;
        for (int i = 0; i < n-1; i++)
            for (int j = i+1; j < n; j++)
                offdiag += fabs(A[i][j]);
        if (offdiag < 1e-15) break;

        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                double apq = A[p][q];
                if (fabs(apq) < 1e-20) continue;
                double app = A[p][p], aqq = A[q][q];
                double tau = (aqq - app) / (2.0 * apq);
                double t   = (tau >= 0.0)
                             ?  1.0 / ( tau + sqrt(1.0 + tau*tau))
                             : -1.0 / (-tau + sqrt(1.0 + tau*tau));
                double c   = 1.0 / sqrt(1.0 + t*t);
                double s2  = t * c;

                A[p][p] = app - t*apq;
                A[q][q] = aqq + t*apq;
                A[p][q] = A[q][p] = 0.0;
                for (int r = 0; r < n; r++) {
                    if (r != p && r != q) {
                        double arp = A[r][p], arq = A[r][q];
                        A[r][p] = A[p][r] = c*arp - s2*arq;
                        A[r][q] = A[q][r] = s2*arp + c*arq;
                    }
                }
                for (int r = 0; r < n; r++) {
                    double vrp = V[r][p], vrq = V[r][q];
                    V[r][p] = c*vrp - s2*vrq;
                    V[r][q] = s2*vrp + c*vrq;
                }
                nrot++;
            }
        }
    }
    for (int i = 0; i < n; i++) d[i] = A[i][i];

    /* sort eigenvalues ascending */
    for (int i = 0; i < n-1; i++) {
        int k = i;
        for (int j = i+1; j < n; j++)
            if (d[j] < d[k]) k = j;
        if (k != i) {
            double tmp = d[i]; d[i] = d[k]; d[k] = tmp;
            for (int r = 0; r < n; r++) {
                double tv = V[r][i]; V[r][i] = V[r][k]; V[r][k] = tv;
            }
        }
    }
    free(e);
    free2d(A, n);
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

cmaes_t* cmaes_init(int N, const double *xstart, double sigma0,
                    int lambda,
                    const double *lbounds, const double *ubounds)
{
    srand((unsigned)time(NULL));

    cmaes_t *evo = (cmaes_t*)calloc(1, sizeof(cmaes_t));
    evo->N = N;
    evo->sigma = sigma0;
    evo->counteval = 0;
    evo->gen = 0;
    evo->fbest = DBL_MAX;

    /* Population size */
    if (lambda <= 0)
        lambda = 4 + (int)floor(3.0 * log((double)N));
    evo->lambda = lambda;
    evo->mu     = lambda / 2;

    /* Weights */
    evo->weights = alloc1d(evo->mu);
    double wsum = 0.0, wsum2 = 0.0;
    for (int i = 0; i < evo->mu; i++) {
        evo->weights[i] = log((double)(evo->mu + 1)) - log((double)(i + 1));
        wsum  += evo->weights[i];
        wsum2 += evo->weights[i] * evo->weights[i];
    }
    for (int i = 0; i < evo->mu; i++) evo->weights[i] /= wsum;
    evo->mueff = wsum * wsum / wsum2;

    /* Strategy parameters */
    evo->cc    = (4.0 + evo->mueff / N) / (N + 4.0 + 2.0*evo->mueff/N);
    evo->cs    = (evo->mueff + 2.0) / (N + evo->mueff + 5.0);
    evo->c1    = 2.0 / ((N + 1.3)*(N + 1.3) + evo->mueff);
    evo->cmu   = fmin(1.0 - evo->c1,
                      2.0*(evo->mueff - 2.0 + 1.0/evo->mueff)
                      / ((N+2.0)*(N+2.0) + evo->mueff));
    evo->damps = 1.0 + 2.0*fmax(0.0, sqrt((evo->mueff-1.0)/(N+1.0))-1.0) + evo->cs;
    evo->chiN  = sqrt((double)N) * (1.0 - 1.0/(4.0*N) + 1.0/(21.0*N*N));

    /* Allocate vectors */
    evo->xmean    = alloc1d(N);
    evo->xbest    = alloc1d(N);
    evo->xold     = alloc1d(N);
    evo->pc       = alloc1d(N);
    evo->ps       = alloc1d(N);
    evo->D        = alloc1d(N);
    evo->artmp    = alloc1d(N);
    evo->arfval   = alloc1d(lambda);
    evo->invsqrtC = alloc1d(N);

    /* Allocate matrices */
    evo->B = alloc2d(N, N);
    evo->C = alloc2d(N, N);
    evo->arx = alloc2d(lambda, N);

    /* Initialize */
    for (int i = 0; i < N; i++) {
        evo->xmean[i] = xstart ? xstart[i] : 0.0;
        evo->D[i]     = 1.0;
        evo->B[i][i]  = 1.0;
        evo->C[i][i]  = 1.0;
        evo->invsqrtC[i] = 1.0;
    }
    memcpy(evo->xbest, evo->xmean, N * sizeof(double));
    evo->eigeneval = 0;

    /* Box constraints */
    if (lbounds) {
        evo->lbounds = alloc1d(N);
        memcpy(evo->lbounds, lbounds, N * sizeof(double));
    }
    if (ubounds) {
        evo->ubounds = alloc1d(N);
        memcpy(evo->ubounds, ubounds, N * sizeof(double));
    }

    return evo;
}

double** cmaes_SamplePopulation(cmaes_t *evo)
{
    int N = evo->N;
    for (int k = 0; k < evo->lambda; k++) {
        /* z ~ N(0,I) */
        double *z = evo->artmp;
        for (int i = 0; i < N; i++) z[i] = evo->D[i] * randn();
        /* x = mean + sigma * B * z */
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) sum += evo->B[i][j] * z[j];
            evo->arx[k][i] = evo->xmean[i] + evo->sigma * sum;
        }
        /* Apply box constraints (clamp) */
        if (evo->lbounds || evo->ubounds) {
            for (int i = 0; i < N; i++) {
                if (evo->lbounds && evo->arx[k][i] < evo->lbounds[i])
                    evo->arx[k][i] = evo->lbounds[i];
                if (evo->ubounds && evo->arx[k][i] > evo->ubounds[i])
                    evo->arx[k][i] = evo->ubounds[i];
            }
        }
    }
    return evo->arx;
}

/* Comparison function for argsort */
typedef struct { double val; int idx; } FitIdx;
static int fitcmp(const void *a, const void *b) {
    double da = ((const FitIdx*)a)->val;
    double db = ((const FitIdx*)b)->val;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

void cmaes_UpdateDistribution(cmaes_t *evo, const double *fval)
{
    int N      = evo->N;
    int lambda = evo->lambda;
    int mu     = evo->mu;

    /* Copy fitness */
    FitIdx *fi = (FitIdx*)malloc(lambda * sizeof(FitIdx));
    for (int i = 0; i < lambda; i++) { fi[i].val = fval[i]; fi[i].idx = i; }
    memcpy(evo->arfval, fval, lambda * sizeof(double));

    /* Sort ascending */
    qsort(fi, lambda, sizeof(FitIdx), fitcmp);

    /* Track best */
    if (fi[0].val < evo->fbest) {
        evo->fbest = fi[0].val;
        memcpy(evo->xbest, evo->arx[fi[0].idx], N * sizeof(double));
    }

    /* Save old mean */
    memcpy(evo->xold, evo->xmean, N * sizeof(double));

    /* New mean = weighted sum of top-mu */
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < mu; j++)
            sum += evo->weights[j] * evo->arx[fi[j].idx][i];
        evo->xmean[i] = sum;
    }

    /* Clamp mean to bounds */
    if (evo->lbounds || evo->ubounds) {
        for (int i = 0; i < N; i++) {
            if (evo->lbounds && evo->xmean[i] < evo->lbounds[i])
                evo->xmean[i] = evo->lbounds[i];
            if (evo->ubounds && evo->xmean[i] > evo->ubounds[i])
                evo->xmean[i] = evo->ubounds[i];
        }
    }

    /* Cumulation: ps */
    /* invsqrtC * (xmean - xold) / sigma */
    /* With diagonal invsqrtC in principal axes: B * diag(1/D) * B^T */
    {
        /* tmp = B^T * (xmean - xold) / sigma */
        double *tmp = evo->artmp;
        for (int i = 0; i < N; i++) {
            double s = 0.0;
            for (int j = 0; j < N; j++)
                s += evo->B[j][i] * (evo->xmean[j] - evo->xold[j]);
            tmp[i] = s / evo->sigma;
        }
        /* ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * B * (1/D) * tmp */
        double fac = sqrt(evo->cs * (2.0 - evo->cs) * evo->mueff);
        for (int i = 0; i < N; i++) {
            double s = 0.0;
            for (int j = 0; j < N; j++)
                s += evo->B[i][j] * (tmp[j] / evo->D[j]);
            evo->ps[i] = (1.0 - evo->cs) * evo->ps[i] + fac * s;
        }
    }

    /* h_sigma indicator */
    double ps_norm = 0.0;
    for (int i = 0; i < N; i++) ps_norm += evo->ps[i] * evo->ps[i];
    ps_norm = sqrt(ps_norm);
    double hsig = ps_norm
        / sqrt(1.0 - pow(1.0 - evo->cs, 2.0*(evo->counteval/(double)lambda + 1.0)))
        / evo->chiN
        < 1.4 + 2.0/(N+1.0) ? 1.0 : 0.0;

    /* Cumulation: pc */
    {
        double fac = sqrt(evo->cc * (2.0 - evo->cc) * evo->mueff);
        for (int i = 0; i < N; i++) {
            evo->pc[i] = (1.0 - evo->cc) * evo->pc[i]
                       + hsig * fac * (evo->xmean[i] - evo->xold[i]) / evo->sigma;
        }
    }

    /* Covariance matrix update */
    {
        double c1  = evo->c1;
        double cmu = evo->cmu;
        double cc  = evo->cc;

        /* C = (1 - c1 - cmu) * C */
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                evo->C[i][j] *= (1.0 - c1 - cmu);

        /* rank-one: c1 * pc * pc^T  (+ correction if hsig==0) */
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                evo->C[i][j] += c1 * (evo->pc[i]*evo->pc[j]
                    + (1.0 - hsig) * cc*(2.0-cc) * evo->C[i][j]);

        /* rank-mu: cmu * sum w_j * (arx_j - xold)(arx_j - xold)^T / sigma^2 */
        for (int k = 0; k < mu; k++) {
            int idx = fi[k].idx;
            for (int i = 0; i < N; i++) {
                double di = (evo->arx[idx][i] - evo->xold[i]) / evo->sigma;
                for (int j = 0; j < N; j++) {
                    double dj = (evo->arx[idx][j] - evo->xold[j]) / evo->sigma;
                    evo->C[i][j] += cmu * evo->weights[k] * di * dj;
                }
            }
        }

        /* Enforce symmetry */
        for (int i = 0; i < N; i++)
            for (int j = i+1; j < N; j++)
                evo->C[i][j] = evo->C[j][i] = 0.5*(evo->C[i][j]+evo->C[j][i]);
    }

    /* Sigma update */
    {
        double ps_norm2 = 0.0;
        for (int i = 0; i < N; i++) ps_norm2 += evo->ps[i]*evo->ps[i];
        evo->sigma *= exp((evo->cs/evo->damps)
                         * (sqrt(ps_norm2)/evo->chiN - 1.0));
    }

    evo->counteval += lambda;
    evo->gen++;

    /* Eigendecomposition: update every ~(lambda/(c1+cmu))/N evals */
    if (evo->counteval - evo->eigeneval
            > lambda / (evo->c1 + evo->cmu) / N / 10.0) {
        evo->eigeneval = evo->counteval;

        /* Enforce symmetry before decomposition */
        for (int i = 0; i < N; i++)
            for (int j = i+1; j < N; j++)
                evo->C[i][j] = evo->C[j][i];

        /* Eigendecomposition: C = B * diag(d) * B^T (d = eigenvalues) */
        double *d = alloc1d(N);
        eigen_sym(N, evo->C, d, evo->B);

        /* D[i] = sqrt(eigenvalue[i]), clamp to avoid NaN */
        for (int i = 0; i < N; i++) {
            if (d[i] < 0.0) d[i] = 0.0;
            evo->D[i] = sqrt(d[i]);
            if (evo->D[i] < 1e-20) evo->D[i] = 1e-20;
        }
        free(d);
    }

    free(fi);
}

const double* cmaes_GetPtr(const cmaes_t *evo, const char *s)
{
    if (!s) return evo->xmean;
    if (s[0] == 'x' && s[1] == 'b') return evo->xbest;
    if (s[0] == 'x' && s[1] == 'm') return evo->xmean;
    return evo->xmean;
}

const char* cmaes_TestForTermination(const cmaes_t *evo,
                                      int maxgen, double tolfun, double tolx)
{
    if (maxgen > 0 && evo->gen >= maxgen)
        return "MaxGenerations";

    /* TolFun: range of fitness in last generation */
    if (tolfun > 0.0 && evo->gen > 10) {
        double fmin = evo->arfval[0], fmax = evo->arfval[0];
        for (int i = 1; i < evo->lambda; i++) {
            if (evo->arfval[i] < fmin) fmin = evo->arfval[i];
            if (evo->arfval[i] > fmax) fmax = evo->arfval[i];
        }
        if (fmax - fmin < tolfun) return "TolFun";
    }

    /* TolX: step size too small */
    if (tolx > 0.0 && evo->sigma * evo->D[evo->N-1] < tolx)
        return "TolX";

    return NULL;
}

void cmaes_exit(cmaes_t *evo)
{
    if (!evo) return;
    int N = evo->N, lam = evo->lambda;
    free(evo->xmean); free(evo->xbest); free(evo->xold);
    free(evo->pc);    free(evo->ps);    free(evo->D);
    free(evo->artmp); free(evo->arfval);free(evo->weights);
    free(evo->invsqrtC);
    free2d(evo->B,   N);
    free2d(evo->C,   N);
    free2d(evo->arx, lam);
    if (evo->lbounds) free(evo->lbounds);
    if (evo->ubounds) free(evo->ubounds);
    free(evo);
}
