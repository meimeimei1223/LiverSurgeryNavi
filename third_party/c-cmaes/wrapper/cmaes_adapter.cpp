/* cmaes_adapter.cpp — engine-facing 6-arg CMA-ES API over upstream Hansen.
 *
 * Implements the API declared in cmaes.h by forwarding to the upstream Hansen
 * c-cmaes (compiled with HANSEN_-prefixed symbols in hansen_cmaes_renamed.c).
 *
 * Design (HANDOFF v2 §4):
 *   - one upstream cmaes_t per run, by value, inside AdapterImpl (per-instance
 *     RNG => parallel runs are deterministic and race-free for the numerics);
 *   - box constraints via upstream's smooth boundary transformation (evaluate
 *     on bounded x; upstream updates from its internal unbounded population);
 *   - gen/lambda/sigma mirrored onto the public struct after Sample/Update;
 *   - FIX1: cmaes_TestForTermination's shared static return buffer is copied
 *     out under an omp critical into a per-instance buffer;
 *   - FIX2: per-run reseeding via cmaes_random_Start (no rgrand realloc/leak);
 *   - FIX3: upstream is included via renames in a C shim; here we take it in
 *     under extern "C" then #undef the renames before the public header.
 */
#include <cstdlib>
#include <cstring>
#include <vector>
#include <new>

extern "C" {
  #include "hansen_renames.h"             /* cmaes_* -> HANSEN_cmaes_* */
  #include "../upstream/src/cmaes_interface.h"         /* HANSEN_cmaes_* decls + HANSEN_cmaes_t */
  #include "../upstream/src/boundary_transformation.h" /* cmaes_boundary_* (NOT renamed) */
  /* cmaes_random_Start is declared inside cmaes.c, not in any header. Declare
   * the renamed symbol directly so cmaes_set_seed can reach it. */
  long HANSEN_cmaes_random_Start(HANSEN_cmaes_random_t *, long unsigned inseed);
  #include "hansen_unrenames.h"           /* cancel renames before public header */
}

#include "cmaes.h"                         /* public cmaes_t + 6-arg API */

/* Hidden per-run state. */
struct AdapterImpl {
    HANSEN_cmaes_t                  h;             /* upstream object, by value   */
    cmaes_boundary_transformation_t b;             /* box-constraint transform    */
    bool                            has_bounds = false;
    std::vector<double>             lb, ub;        /* owned copies (b references)  */
    double**                        pop_bounded = nullptr; /* lambda x N (owned)  */
    int                             N = 0;
    int                             lambda = 0;
    char                            stop_buf[4096]; /* per-instance; >= 3024       */
};

static inline AdapterImpl* impl_of(const cmaes_t* evo) {
    return static_cast<AdapterImpl*>(evo->impl);
}

extern "C" {

cmaes_t* cmaes_init(int N, const double* xstart, double sigma0, int lambda,
                    const double* lbounds, const double* ubounds)
{
    cmaes_t*     evo = static_cast<cmaes_t*>(std::calloc(1, sizeof(cmaes_t)));
    AdapterImpl* im  = new (std::nothrow) AdapterImpl();
    if (!evo || !im) { std::free(evo); delete im; return nullptr; }

    im->N = N;

    /* Per-coordinate stddev all equal to sigma0 => upstream sigma == sigma0
     * (upstream uses sigma = sqrt(sum(stddev^2)/N)). */
    std::vector<double> stddev(N, sigma0);
    std::vector<double> xs(N);
    for (int i = 0; i < N; ++i) xs[i] = xstart ? xstart[i] : 0.0;

    /* Placeholder non-zero seed (deterministic); the real per-run seed is
     * installed later via cmaes_set_seed. filename "non" disables all file IO. */
    HANSEN_cmaes_init_para(&im->h, N, xs.data(), stddev.data(),
                           /*inseed*/ 1, lambda, "non");

    /* Suppress the "actparcmaes.par" dump that cmaes_init_final writes to the
     * CWD: that write is gated on sp.filename (not on the "non" passed above,
     * which readpara_init resets to NULL), so we point sp.filename at a "non"
     * string to disable it. malloc'd because upstream free()s it in
     * readpara_exit. Avoids littering the CWD and file IO (with shared statics)
     * on every per-run init. */
    {
        char* nf = static_cast<char*>(std::malloc(4));
        std::memcpy(nf, "non", 4);
        im->h.sp.filename = nf;
    }

    HANSEN_cmaes_init_final(&im->h);   /* allocates rgrand once, finalizes params */

    /* Determinism (critical for parallel == serial): upstream gates the
     * eigensystem update on elapsed CPU time (updateCmode.maxtime; see
     * cmaes_UpdateEigensystem). Under parallel CPU contention that timing
     * differs run-to-run, so the update schedule — and hence the entire search
     * trajectory — diverges. Setting maxtime >= 1.0 disables the time gate, so
     * updates follow only the deterministic generation-based (modulo) schedule.
     * The per-instance RNG handles the rest; numerics are then reproducible. */
    im->h.sp.updateCmode.maxtime = 1.0;

    /* Effective population size (handles lambda==0 default). */
    im->lambda = im->h.sp.lambda;

    /* Box constraints (copied so the transform doesn't reference caller stack). */
    if (lbounds && ubounds) {
        im->lb.assign(lbounds, lbounds + N);
        im->ub.assign(ubounds, ubounds + N);
        cmaes_boundary_transformation_init(&im->b, im->lb.data(), im->ub.data(),
                                           static_cast<unsigned long>(N));
        im->has_bounds = true;
    }

    im->pop_bounded = static_cast<double**>(std::malloc(sizeof(double*) * im->lambda));
    for (int k = 0; k < im->lambda; ++k)
        im->pop_bounded[k] = static_cast<double*>(std::malloc(sizeof(double) * N));

    evo->impl   = im;
    evo->lambda = im->lambda;
    evo->gen    = static_cast<int>(im->h.gen);
    evo->sigma  = im->h.sigma;
    return evo;
}

void cmaes_set_seed(cmaes_t* evo, unsigned int seed)
{
    if (!evo || !evo->impl) return;
    AdapterImpl* im = impl_of(evo);
    /* Reseed the per-instance RNG without reallocating rgrand (FIX2).
     * seed 0 -> 1 (upstream treats 0 as "use clock"). */
    HANSEN_cmaes_random_Start(&im->h.rand,
                              static_cast<long unsigned>(seed ? seed : 1u));
}

double** cmaes_SamplePopulation(cmaes_t* evo)
{
    AdapterImpl* im = impl_of(evo);
    double* const* pu = HANSEN_cmaes_SamplePopulation(&im->h);  /* unbounded */
    for (int k = 0; k < im->lambda; ++k) {
        if (im->has_bounds)
            cmaes_boundary_transformation(&im->b, pu[k], im->pop_bounded[k],
                                          static_cast<unsigned long>(im->N));
        else
            std::memcpy(im->pop_bounded[k], pu[k], sizeof(double) * im->N);
    }
    evo->gen   = static_cast<int>(im->h.gen);
    evo->sigma = im->h.sigma;
    return im->pop_bounded;  /* engines evaluate the bounded candidates */
}

void cmaes_UpdateDistribution(cmaes_t* evo, const double* fval)
{
    AdapterImpl* im = impl_of(evo);
    /* fval was evaluated on bounded x; upstream updates from its internal
     * unbounded population ranked by fval (example_boundary.c pattern). */
    HANSEN_cmaes_UpdateDistribution(&im->h, fval);
    evo->gen   = static_cast<int>(im->h.gen);
    evo->sigma = im->h.sigma;
}

const double* cmaes_GetPtr(const cmaes_t* evo, const char* s)
{
    AdapterImpl* im = impl_of(evo);
    return HANSEN_cmaes_GetPtr(&im->h, s);  /* e.g. "xbestever" (UNBOUNDED) */
}

const char* cmaes_TestForTermination(const cmaes_t* evo,
                                     int maxgen, double tolfun, double tolx)
{
    AdapterImpl* im = impl_of(evo);
    im->h.sp.stopMaxIter = static_cast<double>(maxgen);
    im->h.sp.stopTolFun  = tolfun;
    im->h.sp.stopTolX    = tolx;

    /* Keep the public mirror current even when the engine reads gen/sigma
     * around the stop check. */
    cmaes_t* m = const_cast<cmaes_t*>(evo);
    m->gen   = static_cast<int>(im->h.gen);
    m->sigma = im->h.sigma;

    const char* s;
    /* FIX1: upstream returns a pointer into a function-local static buffer
     * (shared across instances) -> copy it out under a critical section into
     * this run's own buffer so parallel runs don't race / corrupt it. */
    #pragma omp critical (cmaes_test)
    {
        s = HANSEN_cmaes_TestForTermination(&im->h);
        if (s) {
            std::strncpy(im->stop_buf, s, sizeof(im->stop_buf) - 1);
            im->stop_buf[sizeof(im->stop_buf) - 1] = '\0';
        } else {
            im->stop_buf[0] = '\0';
        }
    }
    return im->stop_buf[0] ? im->stop_buf : nullptr;
}

void cmaes_exit(cmaes_t* evo)
{
    if (!evo) return;
    AdapterImpl* im = impl_of(evo);
    if (im) {
        if (im->pop_bounded) {
            for (int k = 0; k < im->lambda; ++k) std::free(im->pop_bounded[k]);
            std::free(im->pop_bounded);
        }
        if (im->has_bounds) cmaes_boundary_transformation_exit(&im->b);
        HANSEN_cmaes_exit(&im->h);   /* frees rgrand and the rest, once */
        delete im;
    }
    std::free(evo);
}

} /* extern "C" */
