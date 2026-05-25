#pragma once
#ifndef LIVER_REGION_LABEL_H
#define LIVER_REGION_LABEL_H

/*
 * LiverRegionLabel.h
 * ----------------------------------------------------------------------
 * 肝臓メッシュの頂点を anterior(前面) / rim(ヘリ) / posterior(後面) の
 * 3 領域にラベル付けする。Python 側 liver_raycast_4patients.py と
 * liver_anterior_rim_4patients.py の手法を C++ で再実装したもの。
 *
 * 手法:
 *   1. PCA: 全頂点の最小固有値方向 = 「薄い軸」(antero-posterior)
 *   2. Raycast: ±方向の 2 カメラから可視判定 (既存 Reg3D::BVHTree 利用)
 *   3. 解剖学的判定: 平均符号付き曲率の高い側 = anterior (横隔膜面)
 *   4. BFS dilation: anterior から ~target_mm の物理距離分だけ
 *      posterior 側に rim 帯を取る
 *
 * 出力ラベル:
 *   ANTERIOR_CORE = 0   赤に塗る前面コア (raycast で +d 側可視)
 *   POSTERIOR     = 1   青に塗る後面 (どちらの拡張にも入らなかった)
 *   RIM           = 2   橙に塗るヘリ帯 (BFS で anterior から target_mm 拡張)
 *
 * 使い方:
 *   LiverRegionLabel::Result r =
 *       LiverRegionLabel::labelVertices(*liverMesh3D, 8.0f);
 *   if (r.valid()) {
 *       // r.labels[i] is one of: ANTERIOR_CORE / POSTERIOR / RIM
 *   }
 *
 * 計算は頂点 index に紐づくので、registration で transform しても
 * ラベルは不変 (再計算不要)。メッシュ自体を再ロードするときだけ
 * 再計算する。
 */

#include <vector>
#include <set>
#include <utility>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Eigen/Dense>

#include "mCutMesh.h"
#include "RegistrationCore.h"  // Reg3D::BVHTree, BVHNode, Triangle

namespace LiverRegionLabel {

enum Label : uint8_t {
    ANTERIOR_CORE = 0,
    POSTERIOR     = 1,
    RIM           = 2,
};

struct Result {
    std::vector<uint8_t> labels;       // per-vertex (size = nV)
    glm::vec3            view_axis;    // PCA 最小固有ベクトル (anterior 向き、swap 反映済)
    glm::vec3            bbox_center;
    float                bbox_diag = 0.0f;
    float                mean_edge_length = 0.0f;
    int                  n_rings = 0;
    int                  n_anterior = 0;
    int                  n_rim = 0;
    int                  n_posterior = 0;
    bool                 swapped = false;   // 解剖判定で +d/-d を入れ替えたか

    // 検証用 (vis 不要なら空のままでも良い)
    std::vector<float>   signed_H_smoothed;

    bool valid() const { return !labels.empty(); }
};


// ===================================================================
//  内部ヘルパ
// ===================================================================

inline glm::vec3 fetchVertex(const mCutMesh& m, size_t i) {
    return glm::vec3(m.mVertices[i*3], m.mVertices[i*3+1], m.mVertices[i*3+2]);
}

// 一意エッジ (a < b ペア) を構築。
// 132k 頂点なら ~400k 三角形 -> ~600k 一意エッジで std::set でも数十 ms。
inline std::vector<std::pair<int,int>> buildUniqueEdges(const mCutMesh& mesh) {
    std::set<std::pair<int,int>> es;
    const size_t nF = mesh.mIndices.size() / 3;
    for (size_t f = 0; f < nF; f++) {
        int v[3] = { (int)mesh.mIndices[f*3],
                    (int)mesh.mIndices[f*3+1],
                    (int)mesh.mIndices[f*3+2] };
        for (int e = 0; e < 3; e++) {
            int a = v[e], b = v[(e+1)%3];
            if (a > b) std::swap(a, b);
            es.insert({a, b});
        }
    }
    return std::vector<std::pair<int,int>>(es.begin(), es.end());
}

// PCA: 最小固有値ベクトルを返す (Eigen::SelfAdjointEigenSolver、3x3、安定)
inline glm::vec3 computeViewAxisPCA(const mCutMesh& mesh,
                                    glm::vec3* out_mean = nullptr,
                                    Eigen::Vector3d* out_eigvals = nullptr)
{
    const size_t n = mesh.mVertices.size() / 3;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < n; i++) {
        mean += Eigen::Vector3d(mesh.mVertices[i*3],
                                mesh.mVertices[i*3+1],
                                mesh.mVertices[i*3+2]);
    }
    mean /= double(n);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < n; i++) {
        Eigen::Vector3d v(mesh.mVertices[i*3],
                          mesh.mVertices[i*3+1],
                          mesh.mVertices[i*3+2]);
        Eigen::Vector3d d = v - mean;
        cov += d * d.transpose();
    }
    cov /= double(std::max<size_t>(n - 1, 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    Eigen::Vector3d evals = eig.eigenvalues();           // 昇順
    Eigen::Vector3d v0    = eig.eigenvectors().col(0);   // 最小固有値
    v0.normalize();

    if (out_mean)    *out_mean = glm::vec3(mean.x(), mean.y(), mean.z());
    if (out_eigvals) *out_eigvals = evals;
    return glm::vec3(v0.x(), v0.y(), v0.z());
}

// BVH レイキャスト可視判定。
//   各頂点 i について cam_pos -> v[i] のレイを飛ばし、
//   一番手前のヒット距離 >= |v - cam| - tol なら可視 (= 1 番手前にいる)
//
// extractVisibleVerticesCustom と異なり backface culling はしない。
// 「カメラから見える」だけが目的で、表裏の向きは問わない。
//
// OpenMP で頂点ループを並列化 (BVH は read-only なので thread-safe)
inline void raycastVisibilityBVH(
    const mCutMesh& mesh,
    const Reg3D::BVHTree& bvh,
    const glm::vec3& cam_pos,
    float diag,
    std::vector<uint8_t>& visible)
{
    const int n = (int)(mesh.mVertices.size() / 3);
    visible.assign(n, 0);
    const float tol = 1e-3f * diag;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> bvhStack;
        bvhStack.reserve(64);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n; i++) {
            glm::vec3 v(mesh.mVertices[i*3],
                        mesh.mVertices[i*3+1],
                        mesh.mVertices[i*3+2]);
            glm::vec3 dvec = v - cam_pos;
            float    dist  = glm::length(dvec);
            if (dist < 1e-9f) { visible[i] = 1; continue; }
            glm::vec3 dir = dvec / dist;

            float closestHit = FLT_MAX;
            bvhStack.clear();
            bvhStack.push_back(0);
            while (!bvhStack.empty()) {
                int nodeIdx = bvhStack.back();
                bvhStack.pop_back();
                const Reg3D::BVHNode& node = bvh.nodes[nodeIdx];
                if (!node.bbox.intersectRay(cam_pos, dir)) continue;
                if (node.isLeaf()) {
                    for (int ti = 0; ti < node.triangleCount; ti++) {
                        const Reg3D::Triangle& tri =
                            bvh.triangles[node.triangleStart + ti];
                        glm::vec3 e1 = tri.v1 - tri.v0;
                        glm::vec3 e2 = tri.v2 - tri.v0;
                        glm::vec3 h  = glm::cross(dir, e2);
                        float a = glm::dot(e1, h);
                        if (std::abs(a) < 1e-8f) continue;
                        float fInv = 1.0f / a;
                        glm::vec3 s = cam_pos - tri.v0;
                        float u = fInv * glm::dot(s, h);
                        if (u < 0.0f || u > 1.0f) continue;
                        glm::vec3 q = glm::cross(s, e1);
                        float vv = fInv * glm::dot(dir, q);
                        if (vv < 0.0f || u + vv > 1.0f) continue;
                        float t = fInv * glm::dot(e2, q);
                        if (t > 1e-4f && t < closestHit) closestHit = t;
                    }
                } else {
                    if (node.leftChild  >= 0) bvhStack.push_back(node.leftChild);
                    if (node.rightChild >= 0) bvhStack.push_back(node.rightChild);
                }
            }

            // Python: visible iff t_hit >= dist - tol
            visible[i] = (closestHit >= dist - tol) ? 1 : 0;
        }
    }
}

// Cotan-Laplacian で頂点ごとの符号付き平均曲率 H[i] を計算する。
//   定義: H[i] * n[i] = (1/(4 A_i)) * Σ_j (cot α + cot β) (p_j - p_i)
//   ここでは barycentric 1/3 area を A_i に使う (Voronoi 近似でよく使われる)
//   sign は (Δp と頂点法線) の内積で決定: 凸なら正、凹なら負。
//
// 軽量実装: face ループ 1 回で grad/mass/normal を集計。
inline void cotanMeanCurvatureSigned(const mCutMesh& mesh,
                                     std::vector<float>& H_out)
{
    const int nV = (int)(mesh.mVertices.size() / 3);
    const int nF = (int)(mesh.mIndices.size() / 3);
    H_out.assign(nV, 0.0f);

    auto vget = [&](int i) {
        return glm::dvec3(mesh.mVertices[i*3],
                          mesh.mVertices[i*3+1],
                          mesh.mVertices[i*3+2]);
    };

    std::vector<glm::dvec3> grad(nV, glm::dvec3(0.0));
    std::vector<double>     mass(nV, 0.0);   // 1/3 sum tri area
    std::vector<glm::dvec3> nrm (nV, glm::dvec3(0.0));

    for (int f = 0; f < nF; f++) {
        int i0 = (int)mesh.mIndices[f*3];
        int i1 = (int)mesh.mIndices[f*3+1];
        int i2 = (int)mesh.mIndices[f*3+2];
        glm::dvec3 v0 = vget(i0), v1 = vget(i1), v2 = vget(i2);

        // 各エッジベクトル (v_{e+1} - v_{e})
        glm::dvec3 e0 = v1 - v0;   // edge 0->1
        glm::dvec3 e1 = v2 - v1;   // edge 1->2
        glm::dvec3 e2 = v0 - v2;   // edge 2->0

        // 三角形面積と面法線
        glm::dvec3 cr0 = glm::cross(-e2, e0);   // 2*A * face_normal
        double area = 0.5 * glm::length(cr0);
        if (area < 1e-20) continue;
        double inv2A = 1.0 / (2.0 * area);

        // 各頂点における内角の cotangent
        //   cot(angle at v0) = dot(-e2, e0) / |cross(-e2, e0)|
        double cot0 = glm::dot(-e2, e0) * inv2A;
        double cot1 = glm::dot(-e0, e1) * inv2A;
        double cot2 = glm::dot(-e1, e2) * inv2A;

        // Laplace-Beltrami への寄与:
        //   各エッジ (i,j) について (cot α + cot β) (p_j - p_i)
        //   ここでは face 1 つ分の寄与だけ。共有 face で蓄積される。
        //   edge (v0,v1): opposite vertex は v2 -> cot2
        //   edge (v1,v2): opposite vertex は v0 -> cot0
        //   edge (v2,v0): opposite vertex は v1 -> cot1
        glm::dvec3 c01 = cot2 * (v1 - v0);
        glm::dvec3 c12 = cot0 * (v2 - v1);
        glm::dvec3 c20 = cot1 * (v0 - v2);

        grad[i0] += c01 - c20;
        grad[i1] += c12 - c01;
        grad[i2] += c20 - c12;

        mass[i0] += area / 3.0;
        mass[i1] += area / 3.0;
        mass[i2] += area / 3.0;

        // 頂点法線 (face normal sum、単純平均よりこれが定石)
        nrm[i0] += cr0;
        nrm[i1] += cr0;
        nrm[i2] += cr0;
    }

    for (int i = 0; i < nV; i++) {
        if (mass[i] < 1e-20) { H_out[i] = 0.0f; continue; }
        glm::dvec3 lap = grad[i] / (2.0 * mass[i]);
        double mag = glm::length(lap);
        glm::dvec3 nv = nrm[i];
        double nl = glm::length(nv);
        double sign = 1.0;
        if (nl > 1e-20) {
            sign = (glm::dot(lap, nv / nl) >= 0.0) ? 1.0 : -1.0;
        }
        H_out[i] = float(0.5 * sign * mag);
    }
}

// 0.5%, 99.5% percentile clip (異常値抑制)
inline void clipPercentile(std::vector<float>& v, float lo_pct, float hi_pct) {
    if (v.empty()) return;
    std::vector<float> tmp = v;
    std::sort(tmp.begin(), tmp.end());
    auto qf = [&](float p) {
        size_t idx = (size_t)std::round(p * (tmp.size() - 1) / 100.0f);
        idx = std::min(idx, tmp.size() - 1);
        return tmp[idx];
    };
    float lo = qf(lo_pct), hi = qf(hi_pct);
    for (auto& x : v) x = std::max(lo, std::min(hi, x));
}

// グラフ Laplacian 平滑化:
//   v_new[i] = 0.5 * v[i] + 0.5 * mean(v[neighbors of i])
//   iters 回繰り返す。Python 側と同じ式。
inline void laplacianSmooth(std::vector<float>& v,
                            const std::vector<std::pair<int,int>>& edges,
                            int iters)
{
    const int n = (int)v.size();
    std::vector<int>   deg(n, 0);
    for (const auto& e : edges) {
        deg[e.first]++;
        deg[e.second]++;
    }
    std::vector<float> sum(n, 0.0f);
    std::vector<float> next(n, 0.0f);
    for (int it = 0; it < iters; it++) {
        std::fill(sum.begin(), sum.end(), 0.0f);
        for (const auto& e : edges) {
            sum[e.first]  += v[e.second];
            sum[e.second] += v[e.first];
        }
        for (int i = 0; i < n; i++) {
            int d = std::max(deg[i], 1);
            next[i] = 0.5f * v[i] + 0.5f * (sum[i] / float(d));
        }
        std::swap(v, next);
    }
}

inline float meanEdgeLength(const mCutMesh& mesh,
                            const std::vector<std::pair<int,int>>& edges)
{
    if (edges.empty()) return 1.0f;
    double sum = 0.0;
    for (const auto& e : edges) {
        glm::vec3 a = fetchVertex(mesh, e.first);
        glm::vec3 b = fetchVertex(mesh, e.second);
        sum += glm::length(a - b);
    }
    return float(sum / edges.size());
}

// BFS リング拡張: 各 ring で「現在の anterior/rim フロンティア」の
// 隣接 posterior を rim にラベル変更。Python 側のロジックと一致。
inline void bfsDilateRim(std::vector<uint8_t>& labels,
                         const std::vector<std::pair<int,int>>& edges,
                         int n_rings)
{
    const int n = (int)labels.size();

    // CSR 形式 neighbor list 構築
    std::vector<int> deg(n, 0);
    for (const auto& e : edges) { deg[e.first]++; deg[e.second]++; }
    std::vector<int> off(n+1, 0);
    for (int i = 0; i < n; i++) off[i+1] = off[i] + deg[i];
    std::vector<int> ne(off[n], 0);
    std::vector<int> cur(n, 0);
    for (const auto& e : edges) {
        int a = e.first, b = e.second;
        ne[off[a] + cur[a]++] = b;
        ne[off[b] + cur[b]++] = a;
    }

    // フロンティアは初回 anterior_core
    std::vector<uint8_t> frontier(n, 0);
    for (int i = 0; i < n; i++) {
        frontier[i] = (labels[i] == ANTERIOR_CORE) ? 1 : 0;
    }

    for (int r = 0; r < n_rings; r++) {
        std::vector<int> newly;
        newly.reserve(1024);
        for (int i = 0; i < n; i++) {
            if (!frontier[i]) continue;
            for (int k = off[i]; k < off[i+1]; k++) {
                int j = ne[k];
                if (labels[j] == POSTERIOR) {
                    labels[j] = RIM;
                    newly.push_back(j);
                }
            }
        }
        if (newly.empty()) break;
        std::fill(frontier.begin(), frontier.end(), 0);
        for (int j : newly) frontier[j] = 1;
    }
}


// CSR adjacency builder (extracted so cleanupRimCC can re-use it).
// Returns (offsets, neighbours).
inline void buildCSRAdjacency(int n,
                              const std::vector<std::pair<int,int>>& edges,
                              std::vector<int>& off,
                              std::vector<int>& ne)
{
    std::vector<int> deg(n, 0);
    for (const auto& e : edges) { deg[e.first]++; deg[e.second]++; }
    off.assign(n + 1, 0);
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    ne.assign(off[n], 0);
    std::vector<int> cur(n, 0);
    for (const auto& e : edges) {
        int a = e.first, b = e.second;
        ne[off[a] + cur[a]++] = b;
        ne[off[b] + cur[b]++] = a;
    }
}


// Connected-component cleanup of the rim.
//
// The "true" silhouette rim is one continuous band wrapping once around
// the anterior outline. BFS dilation also tags the rims of internal
// concavities (gallbladder fossa floor, IVC groove, etc.). Those are
// small isolated CCs that should be re-classified by NEIGHBOUR-LABEL
// MAJORITY, not blindly dropped:
//   - rim islands embedded in anterior-rich neighbourhoods are dimples
//     -> re-merge into anterior;
//   - rim islands in posterior-rich neighbourhoods are concavity floors
//     -> back to posterior.
// Only the largest CC is kept as rim by default.
inline void cleanupRimCC(std::vector<uint8_t>& labels,
                         const std::vector<int>& off,
                         const std::vector<int>& ne,
                         int& n_to_anterior_out,
                         int& n_to_posterior_out,
                         int& n_cc_out,
                         int& largest_cc_size_out)
{
    const int n = (int)labels.size();
    n_to_anterior_out  = 0;
    n_to_posterior_out = 0;
    n_cc_out           = 0;
    largest_cc_size_out = 0;

    // 1. find rim CCs by BFS restricted to rim-rim edges
    std::vector<int> cc(n, -1);   // CC id per vertex (-1 = not rim)
    std::vector<int> cc_size;
    std::vector<int> stack; stack.reserve(1024);
    int next_cc = 0;
    for (int s = 0; s < n; s++) {
        if (labels[s] != RIM || cc[s] != -1) continue;
        cc[s] = next_cc;
        stack.clear();
        stack.push_back(s);
        int sz = 0;
        while (!stack.empty()) {
            int u = stack.back(); stack.pop_back();
            sz++;
            for (int k = off[u]; k < off[u + 1]; k++) {
                int v = ne[k];
                if (labels[v] == RIM && cc[v] == -1) {
                    cc[v] = next_cc;
                    stack.push_back(v);
                }
            }
        }
        cc_size.push_back(sz);
        next_cc++;
    }
    n_cc_out = next_cc;
    if (next_cc == 0) return;

    // 2. find largest CC
    int largest_cc = 0;
    for (int c = 1; c < next_cc; c++) {
        if (cc_size[c] > cc_size[largest_cc]) largest_cc = c;
    }
    largest_cc_size_out = cc_size[largest_cc];

    // 3. for each non-largest CC, neighbour-majority vote
    // group rim vertices by CC for cheap iteration
    std::vector<std::vector<int>> cc_members(next_cc);
    for (int c = 0; c < next_cc; c++) cc_members[c].reserve(cc_size[c]);
    for (int i = 0; i < n; i++) {
        if (cc[i] >= 0) cc_members[cc[i]].push_back(i);
    }

    for (int c = 0; c < next_cc; c++) {
        if (c == largest_cc) continue;
        // Count anterior vs posterior labels among graph-neighbours
        // (excluding the CC itself, which is all-rim).
        int n_ant = 0, n_post = 0;
        for (int u : cc_members[c]) {
            for (int k = off[u]; k < off[u + 1]; k++) {
                int v = ne[k];
                if (cc[v] == c) continue;        // same CC -> skip
                if (labels[v] == ANTERIOR_CORE) n_ant++;
                else if (labels[v] == POSTERIOR) n_post++;
                // RIM neighbours from a different CC are ignored to
                // avoid coupling decisions across CCs.
            }
        }
        // Tie-break: empty boundary or tie -> posterior (safer for target)
        uint8_t new_label = (n_ant > n_post) ? ANTERIOR_CORE : POSTERIOR;
        for (int u : cc_members[c]) labels[u] = new_label;
        if (new_label == ANTERIOR_CORE) n_to_anterior_out  += (int)cc_members[c].size();
        else                            n_to_posterior_out += (int)cc_members[c].size();
    }
}


// ===================================================================
//  公開関数
// ===================================================================

// Main entry point.
//
//   target_rim_mm     : desired rim band thickness in PHYSICAL MILLIMETRES.
//   smooth_iters      : (kept for backward compatibility, no longer used now
//                       that the curvature path is removed).
//   original_diag_mm  : the original (CT-space) bbox diagonal of the mesh
//                       in mm.  Required if the live mesh has been rescaled
//                       (e.g. via prealignSourceToTarget): the function will
//                       compute  mesh_units_per_mm = current_diag / orig_mm
//                       internally and convert target_rim_mm -> mesh units.
//                       If 0 (default), the function assumes the mesh is
//                       already in mm and skips conversion -- this is the
//                       Python-script case and the unit-test case.
inline Result labelVertices(const mCutMesh& mesh,
                            float target_rim_mm    = 8.0f,
                            int   smooth_iters     = 40,
                            float original_diag_mm = 0.0f)
{
    auto t0 = std::chrono::steady_clock::now();
    Result R;
    (void)smooth_iters;   // no longer used (kept for ABI stability)

    const int nV = (int)(mesh.mVertices.size() / 3);
    if (nV < 3 || mesh.mIndices.size() < 3) {
        std::cerr << "[Region] empty or invalid mesh (V=" << nV
                  << ", I=" << mesh.mIndices.size() << ")\n";
        return R;
    }

    // ----- bbox -----
    glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
    for (int i = 0; i < nV; i++) {
        glm::vec3 v = fetchVertex(mesh, i);
        mn = glm::min(mn, v); mx = glm::max(mx, v);
    }
    R.bbox_center = 0.5f * (mn + mx);
    R.bbox_diag   = glm::length(mx - mn);

    // ----- PCA: view axis -----
    glm::vec3 mean_xyz;
    Eigen::Vector3d evals;
    R.view_axis = computeViewAxisPCA(mesh, &mean_xyz, &evals);
    std::cout << "[Region] V=" << nV
              << "  diag=" << R.bbox_diag
              << "  PCA eigvals=(" << evals(0)
              << ", " << evals(1)
              << ", " << evals(2) << ")\n"
              << "         view_axis=("
              << R.view_axis.x << ", "
              << R.view_axis.y << ", "
              << R.view_axis.z << ")\n";

    // ----- BVH 構築 -----
    auto t_bvh0 = std::chrono::steady_clock::now();
    Reg3D::BVHTree bvh;
    bvh.build(mesh.mVertices, mesh.mIndices);
    auto t_bvh1 = std::chrono::steady_clock::now();
    double ms_bvh =
        std::chrono::duration<double, std::milli>(t_bvh1 - t_bvh0).count();

    // ----- 2 cameras + raycast -----
    glm::vec3 cam_a = R.bbox_center - R.view_axis * (R.bbox_diag * 5.0f);
    glm::vec3 cam_b = R.bbox_center + R.view_axis * (R.bbox_diag * 5.0f);

    auto t_ray0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> vis_a, vis_b;
    raycastVisibilityBVH(mesh, bvh, cam_a, R.bbox_diag, vis_a);
    raycastVisibilityBVH(mesh, bvh, cam_b, R.bbox_diag, vis_b);
    auto t_ray1 = std::chrono::steady_clock::now();
    double ms_ray =
        std::chrono::duration<double, std::milli>(t_ray1 - t_ray0).count();

    int n_a = 0, n_b = 0;
    for (int i = 0; i < nV; i++) { if (vis_a[i]) n_a++; if (vis_b[i]) n_b++; }
    std::cout << "[Region] visible from +d: " << n_a
              << "  from -d: " << n_b
              << "  (BVH build " << ms_bvh
              << " ms, raycast x2 " << ms_ray << " ms)\n";

    // ----- Edges + mean edge length (needed for n_rings) -----
    auto t_e0 = std::chrono::steady_clock::now();
    auto edges = buildUniqueEdges(mesh);
    R.mean_edge_length = meanEdgeLength(mesh, edges);
    auto t_e1 = std::chrono::steady_clock::now();
    double ms_e =
        std::chrono::duration<double, std::milli>(t_e1 - t_e0).count();

    // ----- Anatomical disambiguation by "neither-visible" centroid -----
    //
    // nei = vertices invisible from BOTH +d and -d cameras.
    // These are the floors of deep concavities (gallbladder fossa,
    // IVC groove, porta hepatis) which exist almost exclusively on
    // the POSTERIOR (visceral) surface. So the centroid of nei lies
    // on the posterior side -- the side OPPOSITE the anterior.
    //
    // proj = (nei_centroid - bbox_center) . d
    //   proj > 0  -> nei sits on +d side  -> +d is posterior -> swap
    //   proj < 0  -> nei sits on -d side  -> -d is posterior -> keep
    //
    // Equivalent to the Python implementation in liver_anterior_rim_4patients.py.
    glm::vec3 nei_sum(0.0f);
    int       n_nei = 0;
    for (int i = 0; i < nV; i++) {
        if (!vis_a[i] && !vis_b[i]) {
            nei_sum += fetchVertex(mesh, i);
            n_nei++;
        }
    }
    bool do_swap = false;
    if (n_nei >= 5) {
        glm::vec3 nei_centroid = nei_sum / float(n_nei);
        float proj = glm::dot(nei_centroid - R.bbox_center, R.view_axis);
        std::cout << "[Region] nei concavity-floor: " << n_nei
                  << " vertices, centroid_proj_along_d = "
                  << (proj >= 0.0f ? "+" : "") << proj << "\n";
        do_swap = (proj > 0.0f);
        if (do_swap) {
            std::cout << "[Region] swapped: anterior = -d side  "
                         "(nei centroid on +d side => +d is posterior)\n";
        } else {
            std::cout << "[Region] kept:    anterior = +d side  "
                         "(nei centroid on -d side => -d is posterior)\n";
        }
    } else {
        // Fallback: use vertex-normal alignment to PCA axis.
        // anterior is the side with HIGHER mean(|n . d|).
        const auto& nrm = mesh.mNormals;
        bool have_normals = (nrm.size() == mesh.mVertices.size()
                             && nrm.size() >= 3);
        double sumA = 0.0, sumB = 0.0;
        int    cA = 0,    cB = 0;
        if (have_normals) {
            for (int i = 0; i < nV; i++) {
                glm::vec3 n(nrm[i*3], nrm[i*3+1], nrm[i*3+2]);
                float dotabs = std::fabs(glm::dot(n, R.view_axis));
                if (vis_a[i]) { sumA += dotabs; cA++; }
                if (vis_b[i]) { sumB += dotabs; cB++; }
            }
        }
        float sa = (cA > 0) ? float(sumA / cA) : 0.0f;
        float sb = (cB > 0) ? float(sumB / cB) : 0.0f;
        std::cout << "[Region] nei too small (" << n_nei
                  << "); fallback |n.d|: +d=" << sa
                  << " -d=" << sb << "\n";
        do_swap = (sa < sb);
        std::cout << "[Region] " << (do_swap ? "swapped" : "kept   ")
                  << ": anterior = " << (do_swap ? "-d" : "+d")
                  << " side  (fallback)\n";
    }
    if (do_swap) {
        std::swap(vis_a, vis_b);
        R.view_axis = -R.view_axis;
        R.swapped = true;
    }

    // ----- 初期ラベル -----
    R.labels.assign(nV, POSTERIOR);
    for (int i = 0; i < nV; i++) {
        if (vis_a[i]) R.labels[i] = ANTERIOR_CORE;
    }

    // ----- BFS dilation で rim 帯 -----
    // target_rim_mm は物理 mm。mean_edge_length は現在のメッシュ座標単位。
    // original_diag_mm が与えられていれば、それを使って mm -> mesh-units
    // 変換係数を計算する:
    //   mesh_units_per_mm = bbox_diag / original_diag_mm
    //   target_in_mesh    = target_rim_mm * mesh_units_per_mm
    // original_diag_mm <= 0 なら「メッシュは既に mm 単位」とみなしてスキップ
    // (Python スクリプトと同じ挙動)。
    float scale_used;
    float target_in_mesh;
    if (original_diag_mm > 1e-6f && R.bbox_diag > 1e-9f) {
        scale_used     = R.bbox_diag / original_diag_mm;   // mesh-units per mm
        target_in_mesh = target_rim_mm * scale_used;
        std::cout << "[Region] scale: bboxDiag=" << R.bbox_diag
                  << "  origCTDiag(mm)=" << original_diag_mm
                  << "  -> units/mm=" << scale_used << "\n";
    } else {
        scale_used     = 1.0f;
        target_in_mesh = target_rim_mm;   // assume mesh already in mm
        std::cout << "[Region] no original-diag info -- assuming mesh in mm "
                  << "(units/mm=1.0)\n";
    }
    R.n_rings = std::max(1,
                         (int)std::round(target_in_mesh /
                                          std::max(R.mean_edge_length, 1e-6f)));
    std::cout << "[Region] mean_edge=" << R.mean_edge_length
              << " (mesh-units)  target_in_mesh=" << target_in_mesh
              << "  n_rings=" << R.n_rings
              << "  (target " << target_rim_mm << " mm)\n";

    auto t_bfs0 = std::chrono::steady_clock::now();
    bfsDilateRim(R.labels, edges, R.n_rings);
    auto t_bfs1 = std::chrono::steady_clock::now();
    double ms_bfs =
        std::chrono::duration<double, std::milli>(t_bfs1 - t_bfs0).count();

    // ----- Connected-component cleanup of rim islands -----
    auto t_cc0 = std::chrono::steady_clock::now();
    std::vector<int> off, ne;
    buildCSRAdjacency(nV, edges, off, ne);
    int n_to_ant = 0, n_to_post = 0, n_cc = 0, largest_cc_size = 0;
    cleanupRimCC(R.labels, off, ne,
                 n_to_ant, n_to_post, n_cc, largest_cc_size);
    auto t_cc1 = std::chrono::steady_clock::now();
    double ms_cc =
        std::chrono::duration<double, std::milli>(t_cc1 - t_cc0).count();
    if (n_cc > 0) {
        std::cout << "[Region] rim CCs: " << n_cc
                  << "  largest=" << largest_cc_size << "\n";
        if (n_to_ant || n_to_post) {
            std::cout << "[Region] rim cleanup: kept largest CC ("
                      << largest_cc_size << " v); merged "
                      << n_to_ant << " into anterior, "
                      << n_to_post << " back to posterior\n";
        }
    }

    // ----- カウント -----
    R.n_anterior = R.n_rim = R.n_posterior = 0;
    for (int i = 0; i < nV; i++) {
        if      (R.labels[i] == ANTERIOR_CORE) R.n_anterior++;
        else if (R.labels[i] == RIM)           R.n_rim++;
        else                                    R.n_posterior++;
    }

    R.signed_H_smoothed.clear();   // no longer computed (curvature path removed)

    auto t1 = std::chrono::steady_clock::now();
    double ms_total =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[Region] core=" << R.n_anterior
              << "  rim=" << R.n_rim
              << "  post=" << R.n_posterior
              << "  (rim/core="
              << (100.0f * R.n_rim / std::max(R.n_anterior, 1)) << "%)\n";
    std::cout << "[Region] timing: BVH=" << ms_bvh
              << " raycast=" << ms_ray
              << " edges=" << ms_e
              << " bfs=" << ms_bfs
              << " cc=" << ms_cc
              << " TOTAL=" << ms_total << " ms\n";

    return R;
}


// ===================================================================
//  可視化用ヘルパ
//
//  132k 頂点を全部球マーカーで描画は重すぎるので、ラベルごとに
//  最大 N 点を決定論的に subsample する。返るのは「mCutMesh の頂点
//  index」配列なので、登録後のメッシュ移動にも mVertices[idx*3] で
//  追従可能 (Shift+B と同じパターン)。
// ===================================================================

inline std::vector<int> sampleVertexIndices(
    const std::vector<uint8_t>& labels,
    uint8_t target_label,
    int max_points,
    uint32_t seed = 0xA5A5u)
{
    std::vector<int> all;
    all.reserve(labels.size() / 3 + 1);
    for (int i = 0; i < (int)labels.size(); i++) {
        if (labels[i] == target_label) all.push_back(i);
    }
    if ((int)all.size() <= max_points) return all;

    // 決定論的 stride sample
    std::vector<int> out;
    out.reserve(max_points);
    double step = double(all.size()) / double(max_points);
    for (int k = 0; k < max_points; k++) {
        int idx = (int)std::floor((k + 0.5) * step);
        if (idx >= (int)all.size()) idx = (int)all.size() - 1;
        out.push_back(all[idx]);
    }
    (void)seed;  // 確率性が要らないので未使用
    return out;
}

}  // namespace LiverRegionLabel

#endif  // LIVER_REGION_LABEL_H
