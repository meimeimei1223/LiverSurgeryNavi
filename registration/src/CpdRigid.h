#pragma once
// =====================================================================
//  CpdRigid.h  — Pure Rigid Coherent Point Drift (anchor-independent)
// ---------------------------------------------------------------------
//  Phase U-CPD.  CPD = Myronenko & Song 2010, rigid (+ optional scale).
//  Source Y = GMM centroids (e.g. liver vertices, world coords),
//  Target X = observations (e.g. depth cloud).  Solves T(y)=s*R*y+t to
//  move SOURCE onto TARGET (same direction as U-2 soft-ICP).
//
//  This header is intentionally self-contained: it depends ONLY on
//  <glm> and <Eigen> (no mCutMesh, no SoftPartition) so that the
//  anchor-independence of the CPD path is guaranteed structurally.
//
//  Numerics:
//   - double precision throughout (sigma^2 underflows in float).
//   - E-step is full N*M per iteration (downsample upstream!).
//   - M-step reuses the Umeyama SVD->det-correction->R->scale->t
//     structure (RegistrationCore.h:273-347), just responsibility-weighted.
//   - sufficient statistics are accumulated in ONE pass over the target
//     points using the centering identity, so no M*N matrix is stored.
//   - optional OpenMP parallelisation of the E-step (per-thread partials,
//     merged after); falls back to serial if _OPENMP is not defined.
// =====================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <glm/glm.hpp>
#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace CpdRigid {

struct Params {
    int   maxIters   = 50;
    float w_outlier  = 0.1f;    // outlier ratio [0,1)
    float tol        = 1e-5f;   // relative convergence threshold on sigma^2
    bool  solveScale = false;   // false: s fixed at 1 (Umeyama already scaled)
    // Acceleration (E-step is O(N*M) — keep clouds small):
    bool  downsample  = true;   // (used by the driver, not here)
    float tgtVoxelFac = 0.02f;  // target voxel = fac * sceneDiag
    float srcVoxelFac = 0.0f;   // 0 => do not thin the source
    // Robustness — prevents the sigma^2 -> 0 collapse that happens when the
    // initial pose is already very good (many source points land exactly on
    // target points, sigma^2 shrinks, the GMM overfits a small subset).
    // The floor is set from the TARGET point spacing (the cloud's resolution /
    // noise scale), which is independent of how misaligned the init is:
    float sigmaFloorFac  = 1.0f;  // sigma^2 floored at (sigmaFloorFac * target_NN_spacing)^2 (0 => abs 1e-12)
    float npCollapseFrac = 0.30f; // if final N_P < this * N_target, flag result collapsed/overfit (0 => off)
};

struct Result {
    glm::dmat4 T          = glm::dmat4(1.0);  // s*R + t  (initial source -> final)
    int        iters      = 0;
    double     sigma2     = 0.0;
    double     finalScale = 1.0;
    double     N_P        = 0.0;
    double     sigma2Floor = 0.0;   // the sigma^2 floor actually used
    bool       collapsed   = false; // N_P collapse (overfit) guard tripped
    bool       ok         = false;
};

// ---------------------------------------------------------------------
//  sigma^2 initialisation in O(N+M) (closed form of the O(N*M) double sum)
//      sigma2_init = (1/(3 N M)) * sum_m sum_n || x_n - y_m ||^2
//                  = (1/3)( mean||X||^2 + mean||Y||^2 - 2 meanX . meanY )
// ---------------------------------------------------------------------
inline double initialSigma2(const std::vector<glm::dvec3>& src,
                            const std::vector<glm::dvec3>& tgt) {
    const int M = (int)src.size();
    const int N = (int)tgt.size();
    if (M == 0 || N == 0) return 1e-12;

    glm::dvec3 sumX(0.0), sumY(0.0);
    double sumX2 = 0.0, sumY2 = 0.0;
    for (int n = 0; n < N; ++n) { sumX += tgt[n]; sumX2 += glm::dot(tgt[n], tgt[n]); }
    for (int m = 0; m < M; ++m) { sumY += src[m]; sumY2 += glm::dot(src[m], src[m]); }

    glm::dvec3 meanX = sumX / (double)N;
    glm::dvec3 meanY = sumY / (double)M;

    double val = (sumX2 / (double)N + sumY2 / (double)M
                  - 2.0 * glm::dot(meanX, meanY)) / 3.0;
    return std::max(val, 1e-12);
}

// ---------------------------------------------------------------------
//  Median nearest-neighbour spacing of a point cloud (the cloud's
//  resolution / noise scale). Capped brute force: every point is queried
//  when N is small; for large N a strided subset of queries is used (each
//  query still searches the full cloud, so the estimate stays unbiased).
//  This is a one-time cost; CPD's per-iteration E-step dominates anyway.
// ---------------------------------------------------------------------
inline double medianNNSpacing(const std::vector<glm::dvec3>& pts) {
    const int N = (int)pts.size();
    if (N < 2) return 0.0;
    const int Q      = std::min(N, 2000);          // number of query points
    const int stride = std::max(1, N / Q);
    std::vector<double> nn;
    nn.reserve(Q);
    for (int qi = 0; qi < N && (int)nn.size() < Q; qi += stride) {
        const glm::dvec3 p = pts[qi];
        double best = std::numeric_limits<double>::max();
        for (int j = 0; j < N; ++j) {
            if (j == qi) continue;
            const glm::dvec3 d = p - pts[j];
            const double d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (d2 < best) best = d2;
        }
        if (best < std::numeric_limits<double>::max()) nn.push_back(std::sqrt(best));
    }
    if (nn.empty()) return 0.0;
    std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
    return nn[nn.size() / 2];
}

// ---------------------------------------------------------------------
//  Rigid (+ optional scale) CPD.
//  src = source / GMM centroids (Y, M points).  tgt = observations (X, N).
//  Result.T maps the *given* source onto the target.
//  outTrajectory (optional): if non-null, the cumulative source->estimate
//    transform after every EM iteration is appended. Used by the LIVE
//    visualisation to replay the convergence frame-by-frame. Recording adds a
//    cheap 4x4 copy per iteration and does not change the result.
// ---------------------------------------------------------------------
inline Result runRigidCPD(const std::vector<glm::dvec3>& src,
                          const std::vector<glm::dvec3>& tgt,
                          const Params& p,
                          std::vector<glm::dmat4>* outTrajectory = nullptr) {
    Result res;
    const int M = (int)src.size();   // GMM centroids (Y)
    const int N = (int)tgt.size();   // observations (X)
    if (M < 3 || N < 3) return res;  // ok stays false

    const double w = std::min(std::max((double)p.w_outlier, 0.0), 0.99);
    const double PI = 3.14159265358979323846;

    // Current FULL transform (CPD re-solves the full transform each
    // iteration relative to the ORIGINAL source — it is not accumulated).
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    double     s = 1.0;
    glm::dvec3 t(0.0);
    double sigma2 = initialSigma2(src, tgt);
    const double sigma2_init = sigma2;
    // Floor from the target spacing (resolution/noise scale), independent of
    // the init misalignment. Capped below sigma^2_init so the EM never starts
    // below its own floor. sigma^2 is the noise-variance estimate, so this
    // stops the runaway shrink-to-zero on already-good inits.
    double sigma2_floor = 1e-12;
    if (p.sigmaFloorFac > 0.0f) {
        const double spacing = medianNNSpacing(tgt);
        const double f = (double)p.sigmaFloorFac * spacing;
        sigma2_floor = std::max(1e-12, std::min(f * f, 0.5 * sigma2_init));
    }
    double lastN_P = 0.0;

    // Builds the current cumulative source->estimate transform from R,t,s
    // (glm column-major; identical layout to RegistrationCore.h). Used for the
    // optional per-iteration trajectory and for the final Result.T.
    auto buildT = [&]() -> glm::dmat4 {
        glm::dmat4 T(1.0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T[j][i] = s * R(i, j);
        T[3][0] = t.x; T[3][1] = t.y; T[3][2] = t.z;
        return T;
    };

    int nThreads = 1;
#ifdef _OPENMP
    nThreads = std::max(1, omp_get_max_threads());
#endif

    struct Accum {
        double     N_P = 0.0;
        glm::dvec3 sum_Px{0.0};
        double     sum_Px_xnorm2 = 0.0;
        double     A_raw[3][3] = {{0,0,0},{0,0,0},{0,0,0}}; // (i,j) = sum P x_i y_j
        std::vector<double> Py;                              // row sums per source m
    };

    int iter = 0;
    for (; iter < p.maxIters; ++iter) {
        // ----- transformed source Ty[m] = s*R*y_m + t -----
        std::vector<glm::dvec3> Ty(M);
        for (int m = 0; m < M; ++m) {
            Eigen::Vector3d y(src[m].x, src[m].y, src[m].z);
            Eigen::Vector3d ry = R * y;
            Ty[m] = glm::dvec3(s * ry(0) + t.x, s * ry(1) + t.y, s * ry(2) + t.z);
        }

        const double two_sigma2 = 2.0 * sigma2;
        const double c = (w > 0.0)
            ? (w / (1.0 - w)) * ((double)M / (double)N)
              * std::pow(2.0 * PI * sigma2, 1.5)
            : 0.0;

        std::vector<Accum> acc(nThreads);
        for (int th = 0; th < nThreads; ++th) acc[th].Py.assign(M, 0.0);

        // ----- E-step + sufficient-statistic accumulation -----
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            Accum& a = acc[tid];
            std::vector<double> ebuf(M);
            double* Py = a.Py.data();

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int n = 0; n < N; ++n) {
                const glm::dvec3 xn = tgt[n];
                double denom = c;
                for (int m = 0; m < M; ++m) {
                    const glm::dvec3 d = xn - Ty[m];
                    const double d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                    const double e = std::exp(-d2 / two_sigma2);
                    ebuf[m] = e;
                    denom += e;
                }
                if (denom <= 0.0) continue;
                const double inv  = 1.0 / denom;
                const double Pt_n = (denom - c) * inv;   // sum_m P(m|n)

                glm::dvec3 q(0.0);                        // q_n = sum_m P(m|n) y_m
                for (int m = 0; m < M; ++m) {
                    const double pmn = ebuf[m] * inv;
                    Py[m] += pmn;
                    q += pmn * src[m];                    // ORIGINAL source y_m
                }

                a.N_P           += Pt_n;
                a.sum_Px        += Pt_n * xn;
                a.sum_Px_xnorm2 += Pt_n * (xn.x * xn.x + xn.y * xn.y + xn.z * xn.z);
                // A_raw += outer(xn, q) : element (i,j) += xn[i]*q[j]
                a.A_raw[0][0] += xn.x * q.x; a.A_raw[0][1] += xn.x * q.y; a.A_raw[0][2] += xn.x * q.z;
                a.A_raw[1][0] += xn.y * q.x; a.A_raw[1][1] += xn.y * q.y; a.A_raw[1][2] += xn.y * q.z;
                a.A_raw[2][0] += xn.z * q.x; a.A_raw[2][1] += xn.z * q.y; a.A_raw[2][2] += xn.z * q.z;
            }
        }

        // ----- merge per-thread partials -----
        double     N_P = 0.0, sum_Px_xnorm2 = 0.0;
        glm::dvec3 sum_Px(0.0);
        double     A_raw[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        std::vector<double> Py(M, 0.0);
        for (int th = 0; th < nThreads; ++th) {
            N_P           += acc[th].N_P;
            sum_Px        += acc[th].sum_Px;
            sum_Px_xnorm2 += acc[th].sum_Px_xnorm2;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    A_raw[i][j] += acc[th].A_raw[i][j];
            for (int m = 0; m < M; ++m) Py[m] += acc[th].Py[m];
        }
        lastN_P = N_P;

        if (N_P < 1e-9) {            // all observations classified as outliers
            res.iters  = iter;
            res.sigma2 = sigma2;
            res.ok     = false;
            break;
        }

        // ----- weighted centroids & source-side sums (from Py[]) -----
        glm::dvec3 sum_Py(0.0);
        double     sum_Py_ynorm2 = 0.0;
        for (int m = 0; m < M; ++m) {
            sum_Py        += Py[m] * src[m];
            sum_Py_ynorm2 += Py[m] * (src[m].x * src[m].x + src[m].y * src[m].y + src[m].z * src[m].z);
        }
        const glm::dvec3 mu_x = sum_Px / N_P;
        const glm::dvec3 mu_y = sum_Py / N_P;
        const double mx[3] = { mu_x.x, mu_x.y, mu_x.z };
        const double my[3] = { mu_y.x, mu_y.y, mu_y.z };

        // ----- centered cross-covariance A = A_raw - N_P mu_x mu_y^T -----
        Eigen::Matrix3d A;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A(i, j) = A_raw[i][j] - N_P * mx[i] * my[j];

        // ----- SVD + det correction (identical to UmeyamaRegistration) -----
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix3d U  = svd.matrixU();
        const Eigen::Vector3d Sv = svd.singularValues();
        const Eigen::Matrix3d V  = svd.matrixV();
        Eigen::Matrix3d Dm = Eigen::Matrix3d::Identity();
        if (U.determinant() * V.determinant() < 0.0) Dm(2, 2) = -1.0;
        R = U * Dm * V.transpose();

        const double trSC = Sv(0) * Dm(0, 0) + Sv(1) * Dm(1, 1) + Sv(2) * Dm(2, 2);

        // ----- scale (solve or fix at 1) -----
        const double denom_s = sum_Py_ynorm2 - N_P * (my[0]*my[0] + my[1]*my[1] + my[2]*my[2]);
        if (p.solveScale) s = (denom_s > 1e-12) ? (trSC / denom_s) : 1.0;
        else              s = 1.0;

        // ----- translation t = mu_x - s R mu_y -----
        const Eigen::Vector3d muyE(my[0], my[1], my[2]);
        const Eigen::Vector3d Rmuy = R * muyE;
        t = glm::dvec3(mx[0] - s * Rmuy(0), mx[1] - s * Rmuy(1), mx[2] - s * Rmuy(2));

        // Record this iteration's cumulative transform for LIVE replay. Done
        // here (after R,t,s are final for this iter) so trajectory.back()
        // equals Result.T even when the loop exits via the convergence break.
        if (outTrajectory) outTrajectory->push_back(buildT());

        // ----- sigma^2 update -----
        const double trXX = sum_Px_xnorm2 - N_P * (mx[0]*mx[0] + mx[1]*mx[1] + mx[2]*mx[2]);
        double sigma2_new = (trXX - s * trSC) / (3.0 * N_P);
        sigma2_new = std::max(sigma2_new, sigma2_floor);

        const double dchg = std::abs(sigma2_new - sigma2);
        sigma2 = sigma2_new;
        res.ok = true;

        if (dchg < (double)p.tol * std::max(sigma2, sigma2_floor)) { ++iter; break; }
        // Once sigma^2 reaches the floor it stops moving, so the dchg test above
        // converges naturally — no separate "collapsed to ~0" break needed.
    }

    // ----- build final T (glm column-major; matches RegistrationCore.h) -----
    glm::dmat4 T = buildT();

    // N_P collapse / overfit guard: if too few observations are explained, the
    // GMM has shrunk onto a small subset of the target (sigma^2 too small for
    // this init). Refuse the result so the caller keeps the current pose.
    if (p.npCollapseFrac > 0.0f && N > 0 &&
        lastN_P < (double)p.npCollapseFrac * (double)N) {
        res.collapsed = true;
        res.ok        = false;
    }

    res.T          = T;
    res.iters      = iter;
    res.sigma2     = sigma2;
    res.finalScale = s;
    res.N_P        = lastN_P;
    res.sigma2Floor = sigma2_floor;
    return res;
}

} // namespace CpdRigid
