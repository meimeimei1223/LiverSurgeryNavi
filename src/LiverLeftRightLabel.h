#pragma once
#ifndef LIVER_LEFT_RIGHT_LABEL_H
#define LIVER_LEFT_RIGHT_LABEL_H

/*
 * LiverLeftRightLabel.h
 * ----------------------------------------------------------------------
 * 肝臓メッシュの頂点を pure-right (右葉) / boundary (鎌状間膜あたり) /
 * pure-left (左葉) の 3 領域にラベル付けする。Python 側
 * liver_leftright_4patients.py を C++ で再実装したもの。
 *
 * 既存の LiverRegionLabel.h (anterior/rim/posterior) と完全パラレル
 * の構造で、main.cpp 側に Y/Shift+Y キーで統合する。最終的には
 * (LR_label, AP_label) の積で 4 象限分類に使う予定。
 *
 * 手法 (5 フェーズ):
 *   1. PCA 3 軸抽出: 共分散の固有分解 (昇順)
 *   2. d_LR 軸選択: 中間/最大固有ベクトルのうち射影 extent が大きい方
 *   3. サイン決定: ±d_LR 両方向で raycast → 可視面積比較
 *      a_vis(+d) < a_vis(-d) なら +d が右 (大葉が前で隠してる)
 *      |a_diff| / a_avg ≥ 2% で decisive、弱ければ area-centroid lean
 *   4. 二閾値パーセンタイル切断: 頂点 mass を p[i]=V[i]·d_LR で降順
 *      累積し、60% / 70% 達成点を p_thr_pure, p_thr_full とする
 *   5. 3 バンドラベル付け
 *
 * 出力ラベル:
 *   PURE_RIGHT = 0   緑に塗る純右 (上位 60% mass)
 *   PURE_LEFT  = 1   紫に塗る純左 (下位 30% mass)
 *   BOUNDARY   = 2   黄に塗る境界帯 (60-70% mass、両属)
 *
 * 使い方:
 *   LiverLeftRightLabel::Result r =
 *       LiverLeftRightLabel::labelVertices(*liverMesh3D);
 *   if (r.valid()) {
 *       // r.labels[i] is one of: PURE_RIGHT / PURE_LEFT / BOUNDARY
 *   }
 *
 * 既存 LiverRegionLabel との関係:
 *   - raycastVisibilityBVH, fetchVertex はそのまま流用
 *   - PCA は 3 軸全部欲しいので別 helper (computeAllPCAAxes)
 *   - スケール変換は不要 (閾値が無次元 mass 比なのでスケール不変)
 *
 * 計算は頂点 index に紐づくので、registration で transform しても
 * ラベルは不変 (再計算不要)。メッシュ自体を再ロードするときだけ
 * 再計算する。
 */

#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cfloat>
#include <cstdio>     // std::snprintf  (quadrantMaskString 用)
#include <string>     // std::string    (quadrantMaskString 戻り値用)

#include <glm/glm.hpp>
#include <Eigen/Dense>

#include "mCutMesh.h"
#include "RegistrationCore.h"  // Reg3D::BVHTree
#include "LiverRegionLabel.h"  // raycastVisibilityBVH, fetchVertex を流用

namespace LiverLeftRightLabel {

// ===================================================================
//  ラベル定義
// ===================================================================
enum Label : uint8_t {
    PURE_RIGHT = 0,   // 純右 (上位 right_pure_fraction の mass)
    PURE_LEFT  = 1,   // 純左 (下位 (1 - right_full_fraction) の mass)
    BOUNDARY   = 2,   // 境界帯 (right_pure ~ right_full の mass)
};

// ===================================================================
//  診断情報 (eclipse signal の中身を全部公開しておく)
// ===================================================================
struct EclipseInfo {
    int   n_vis_pos = 0,   n_vis_neg = 0;     // 可視頂点数 (診断用; 面積が PRIMARY)
    float a_vis_pos = 0.0f, a_vis_neg = 0.0f; // 可視面の面積合計 (PRIMARY)
    float lean_area = 0.0f;                    // area-centroid lean (fallback signal)
    bool  sign_eclipse_n = false;              // 診断: vert count rule
    bool  sign_eclipse_a = false;              // PRIMARY: area rule (true なら +d_LR が右)
    bool  sign_area      = false;              // fallback: lean rule
    bool  decisive       = false;              // |a_diff| >= 2% * a_avg
    bool  flipped_manual = false;              // 手動 override 適用済か
};

// ===================================================================
//  Result struct (LiverRegionLabel::Result と同形パターン)
// ===================================================================
struct Result {
    std::vector<uint8_t> labels;       // per-vertex (size = nV)
    glm::vec3            d_lr;         // sign-normalised: +d_lr が RIGHT
    glm::vec3            bbox_center;
    float                bbox_diag = 0.0f;
    float                p_thr_pure = 0.0f; // 上位閾値 (p[i]>=これ → PURE_RIGHT)
    float                p_thr_full = 0.0f; // 下位閾値 (p[i]<これ → PURE_LEFT)
    EclipseInfo          eclipse;
    int                  n_pure_right = 0;
    int                  n_boundary   = 0;
    int                  n_pure_left  = 0;

    // PCA 詳細 (デバッグ用)
    int                  lr_axis_idx = -1;   // 1=mid, 2=max eigvec が選ばれた
    float                ext_mid     = 0.0f;
    float                ext_max     = 0.0f;

    bool valid() const { return !labels.empty(); }
};


// ===================================================================
//  内部ヘルパ
// ===================================================================

// PCA 全 3 軸 + 固有値 (昇順)
inline void computeAllPCAAxes(const mCutMesh& mesh,
                              glm::vec3 axes[3],
                              Eigen::Vector3d& evals,
                              glm::vec3& mean_out)
{
    const size_t n = mesh.mVertices.size() / 3;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < n; i++) {
        mean += Eigen::Vector3d(mesh.mVertices[i*3],
                                mesh.mVertices[i*3+1],
                                mesh.mVertices[i*3+2]);
    }
    mean /= double(std::max<size_t>(n, 1));

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
    evals = eig.eigenvalues();   // 昇順
    for (int k = 0; k < 3; k++) {
        Eigen::Vector3d v = eig.eigenvectors().col(k).normalized();
        axes[k] = glm::vec3(float(v.x()), float(v.y()), float(v.z()));
    }
    mean_out = glm::vec3(float(mean.x()), float(mean.y()), float(mean.z()));
}

// d_LR 軸選択: axes[1] (中間) と axes[2] (最大) のうち、
//   頂点を射影した extent (max - min) が大きい方を採用。
//   多くの患者では axes[2] (最大固有値) になるが、頭尾方向が大きく
//   出る個体に対する保険として extent を比較する。
inline glm::vec3 pickLRAxis(const mCutMesh& mesh,
                            const glm::vec3 axes[3],
                            float& out_ext_mid,
                            float& out_ext_max,
                            int&   out_chosen_idx /* 1 or 2 */)
{
    const size_t n = mesh.mVertices.size() / 3;
    float pmin1 =  FLT_MAX, pmax1 = -FLT_MAX;
    float pmin2 =  FLT_MAX, pmax2 = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        glm::vec3 v(mesh.mVertices[i*3],
                    mesh.mVertices[i*3+1],
                    mesh.mVertices[i*3+2]);
        float p1 = glm::dot(v, axes[1]);
        float p2 = glm::dot(v, axes[2]);
        if (p1 < pmin1) pmin1 = p1;
        if (p1 > pmax1) pmax1 = p1;
        if (p2 < pmin2) pmin2 = p2;
        if (p2 > pmax2) pmax2 = p2;
    }
    out_ext_mid = pmax1 - pmin1;
    out_ext_max = pmax2 - pmin2;
    if (out_ext_max >= out_ext_mid) {
        out_chosen_idx = 2;
        return axes[2];
    } else {
        out_chosen_idx = 1;
        return axes[1];
    }
}

// 全 3 頂点が mask 内である面の面積合計
//   Python: extract_submesh + measure_area と同じ動作
//   境界面 (頂点が一部だけ mask 内) は捨てられる。
inline float faceAreaSum(const mCutMesh& mesh,
                         const std::vector<uint8_t>& vmask)
{
    float total = 0.0f;
    const size_t nT = mesh.mIndices.size() / 3;
    for (size_t f = 0; f < nT; f++) {
        GLuint i0 = mesh.mIndices[f*3];
        GLuint i1 = mesh.mIndices[f*3+1];
        GLuint i2 = mesh.mIndices[f*3+2];
        if (!vmask[i0] || !vmask[i1] || !vmask[i2]) continue;
        glm::vec3 v0(mesh.mVertices[i0*3], mesh.mVertices[i0*3+1], mesh.mVertices[i0*3+2]);
        glm::vec3 v1(mesh.mVertices[i1*3], mesh.mVertices[i1*3+1], mesh.mVertices[i1*3+2]);
        glm::vec3 v2(mesh.mVertices[i2*3], mesh.mVertices[i2*3+1], mesh.mVertices[i2*3+2]);
        total += 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
    }
    return total;
}

// 全面の面積合計 (mask 不問)
inline float totalFaceArea(const mCutMesh& mesh) {
    float total = 0.0f;
    const size_t nT = mesh.mIndices.size() / 3;
    for (size_t f = 0; f < nT; f++) {
        GLuint i0 = mesh.mIndices[f*3];
        GLuint i1 = mesh.mIndices[f*3+1];
        GLuint i2 = mesh.mIndices[f*3+2];
        glm::vec3 v0(mesh.mVertices[i0*3], mesh.mVertices[i0*3+1], mesh.mVertices[i0*3+2]);
        glm::vec3 v1(mesh.mVertices[i1*3], mesh.mVertices[i1*3+1], mesh.mVertices[i1*3+2]);
        glm::vec3 v2(mesh.mVertices[i2*3], mesh.mVertices[i2*3+1], mesh.mVertices[i2*3+2]);
        total += 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
    }
    return total;
}

// 1 方向 d について「bbox_center + d * 5*diag」のカメラから raycast し、
//   可視頂点数 n_visible と全 3 頂点可視な面の面積合計 a_visible を返す。
inline void eclipseSignal(const mCutMesh& mesh,
                          const Reg3D::BVHTree& bvh,
                          const glm::vec3& d,
                          const glm::vec3& bbox_center,
                          float diag,
                          int& n_visible,
                          float& a_visible)
{
    glm::vec3 cam = bbox_center + d * (diag * 5.0f);
    std::vector<uint8_t> visible;
    LiverRegionLabel::raycastVisibilityBVH(mesh, bvh, cam, diag, visible);
    n_visible = 0;
    for (uint8_t b : visible) if (b) n_visible++;
    a_visible = faceAreaSum(mesh, visible);
}

// サイン決定 (PRIMARY: eclipse area asymmetry, fallback: area-centroid lean)
//   戻り値: true なら +d_lr_in が RIGHT を指す。
//   info_out にすべての中間信号を埋める。
inline bool decideLRSign(const mCutMesh& mesh,
                         const Reg3D::BVHTree& bvh,
                         const glm::vec3& d_lr_in,
                         const glm::vec3& bbox_center,
                         float diag,
                         EclipseInfo& info_out)
{
    int   n_pos = 0, n_neg = 0;
    float a_pos = 0.0f, a_neg = 0.0f;
    eclipseSignal(mesh, bvh,  d_lr_in, bbox_center, diag, n_pos, a_pos);
    eclipseSignal(mesh, bvh, -d_lr_in, bbox_center, diag, n_neg, a_neg);

    info_out.n_vis_pos = n_pos;
    info_out.n_vis_neg = n_neg;
    info_out.a_vis_pos = a_pos;
    info_out.a_vis_neg = a_neg;
    info_out.sign_eclipse_n = (n_pos < n_neg);          // 診断のみ
    info_out.sign_eclipse_a = (a_pos < a_neg);          // PRIMARY

    float a_avg = 0.5f * (a_pos + a_neg);
    info_out.decisive = std::fabs(a_pos - a_neg) >= 0.02f * std::max(a_avg, 1e-9f);

    // Fallback: area-weighted face centroid lean along d_lr
    Eigen::Vector3d a_centroid = Eigen::Vector3d::Zero();
    double a_total_d = 0.0;
    const size_t nT = mesh.mIndices.size() / 3;
    for (size_t f = 0; f < nT; f++) {
        GLuint i0 = mesh.mIndices[f*3];
        GLuint i1 = mesh.mIndices[f*3+1];
        GLuint i2 = mesh.mIndices[f*3+2];
        glm::vec3 v0(mesh.mVertices[i0*3], mesh.mVertices[i0*3+1], mesh.mVertices[i0*3+2]);
        glm::vec3 v1(mesh.mVertices[i1*3], mesh.mVertices[i1*3+1], mesh.mVertices[i1*3+2]);
        glm::vec3 v2(mesh.mVertices[i2*3], mesh.mVertices[i2*3+1], mesh.mVertices[i2*3+2]);
        glm::vec3 fc = (v0 + v1 + v2) / 3.0f;
        float fa = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
        a_centroid.x() += double(fc.x) * double(fa);
        a_centroid.y() += double(fc.y) * double(fa);
        a_centroid.z() += double(fc.z) * double(fa);
        a_total_d += double(fa);
    }
    a_centroid /= std::max(a_total_d, 1e-12);
    glm::vec3 a_centroid_glm(float(a_centroid.x()),
                             float(a_centroid.y()),
                             float(a_centroid.z()));
    info_out.lean_area = glm::dot(a_centroid_glm - bbox_center, d_lr_in);
    info_out.sign_area = info_out.lean_area > 0.0f;

    return info_out.decisive ? info_out.sign_eclipse_a : info_out.sign_area;
}


// ===================================================================
//  メイン関数
// ===================================================================
//   right_pure_fraction = 0.60 デフォルト (Python と同一)
//   right_full_fraction = 0.70 デフォルト
//   flip_manual = true でサイン手動反転 (Python の FLIP_OVERRIDE 相当)
//
//   ログ出力フォーマットは Python (liver_leftright_4patients.py) と
//   同じ並びにしてあるので、ターミナルで diff 比較できる。
inline Result labelVertices(const mCutMesh& mesh,
                            float right_pure_fraction = 0.60f,
                            float right_full_fraction = 0.70f,
                            bool  flip_manual = false)
{
    Result R;
    const int nV = (int)(mesh.mVertices.size() / 3);
    if (nV < 3 || mesh.mIndices.size() < 3) {
        std::cerr << "[LR] mesh empty (nV=" << nV
                  << ", nT=" << mesh.mIndices.size()/3 << ")\n";
        return R;
    }
    if (right_pure_fraction > right_full_fraction) {
        std::cerr << "[LR] right_pure_fraction (" << right_pure_fraction
                  << ") must be <= right_full_fraction ("
                  << right_full_fraction << ")\n";
        return R;
    }
    if (right_pure_fraction < 0.05f || right_pure_fraction > 0.95f ||
        right_full_fraction < 0.05f || right_full_fraction > 0.95f) {
        std::cerr << "[LR] fractions out of [0.05, 0.95] range\n";
        return R;
    }

    std::cout << "[LR] V=" << nV
              << "  F=" << (mesh.mIndices.size()/3) << std::endl;

    auto t0 = std::chrono::steady_clock::now();

    // ---- bbox ----
    glm::vec3 mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < nV; i++) {
        glm::vec3 v = LiverRegionLabel::fetchVertex(mesh, i);
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    R.bbox_center = 0.5f * (mn + mx);
    R.bbox_diag   = glm::length(mx - mn);

    std::cout << "[LR] diag=" << R.bbox_diag << std::endl;

    // ---- PCA 全 3 軸 ----
    glm::vec3 axes[3];
    Eigen::Vector3d evals;
    glm::vec3 mean_xyz;
    computeAllPCAAxes(mesh, axes, evals, mean_xyz);

    // d_AP は最小固有 (Python と同じ; 本実装では使わないが log のみ)
    const glm::vec3& d_ap = axes[0];
    std::cout << "[LR] d_AP (smallest) = ["
              << d_ap.x << "  " << d_ap.y << "  " << d_ap.z << "]" << std::endl;

    // d_LR 選択
    int chosen = -1;
    glm::vec3 d_lr = pickLRAxis(mesh, axes, R.ext_mid, R.ext_max, chosen);
    R.lr_axis_idx = chosen;
    std::cout << "[LR] mid-extent = " << R.ext_mid
              << ", large-extent = " << R.ext_max
              << "  ->  d_LR = "
              << (chosen == 2 ? "largest-eigvec" : "mid-eigvec") << std::endl;
    std::cout << "[LR] d_LR (before sign) = ["
              << d_lr.x << "  " << d_lr.y << "  " << d_lr.z << "]" << std::endl;

    // ---- BVH ビルド (1 回だけ、両方向の raycast で再利用) ----
    auto t_bvh0 = std::chrono::steady_clock::now();
    Reg3D::BVHTree bvh;
    bvh.build(mesh.mVertices, mesh.mIndices);
    auto t_bvh1 = std::chrono::steady_clock::now();
    double ms_bvh =
        std::chrono::duration<double, std::milli>(t_bvh1 - t_bvh0).count();
    std::cout << "[LR] BVH build: " << ms_bvh << " ms" << std::endl;

    // ---- サイン決定 ----
    auto t_ec0 = std::chrono::steady_clock::now();
    bool right_is_pos = decideLRSign(mesh, bvh, d_lr,
                                     R.bbox_center, R.bbox_diag, R.eclipse);
    auto t_ec1 = std::chrono::steady_clock::now();
    double ms_ec =
        std::chrono::duration<double, std::milli>(t_ec1 - t_ec0).count();

    if (flip_manual) {
        right_is_pos = !right_is_pos;
        R.eclipse.flipped_manual = true;
    }

    // ログ (Python フォーマットを踏襲)
    const auto& info = R.eclipse;
    int   n_diff = info.n_vis_pos - info.n_vis_neg;
    float a_diff = info.a_vis_pos - info.a_vis_neg;
    float n_avg  = 0.5f * float(info.n_vis_pos + info.n_vis_neg);
    float a_avg  = 0.5f * (info.a_vis_pos + info.a_vis_neg);
    std::cout << "[LR] sign signals along d_LR (positive = +d_LR is right):" << std::endl;
    std::cout << "[LR]   ECLIPSE (raycast):" << std::endl;
    std::cout << "[LR]     +d_LR cam: " << info.n_vis_pos
              << " visible verts, " << info.a_vis_pos << " visible area" << std::endl;
    std::cout << "[LR]     -d_LR cam: " << info.n_vis_neg
              << " visible verts, " << info.a_vis_neg << " visible area" << std::endl;
    std::cout << "[LR]     diff vert = " << (n_diff >= 0 ? "+" : "") << n_diff
              << " (" << (n_diff >= 0 ? "+" : "")
              << (100.0f * n_diff / std::max(n_avg, 1.0f)) << "%)"
              << "  diff area = " << (a_diff >= 0 ? "+" : "") << a_diff
              << " (" << (a_diff >= 0 ? "+" : "")
              << (100.0f * a_diff / std::max(a_avg, 1e-9f)) << "%)"
              << std::endl;
    const char* sgn_a = info.sign_eclipse_a ? "+d_LR" : "-d_LR";
    const char* sgn_n = info.sign_eclipse_n ? "+d_LR" : "-d_LR";
    std::cout << "[LR]     PRIMARY (area): " << sgn_a << " is RIGHT"
              << "  |  diag (verts): " << sgn_n << " is RIGHT"
              << "  |  decisive=" << (info.decisive ? "True" : "False")
              << std::endl;
    if (info.sign_eclipse_n != info.sign_eclipse_a) {
        std::cout << "[LR]     note: vert-count and area-rule DISAGREE -- "
                     "trusting AREA (vertex count is sampling-biased)."
                  << std::endl;
    }
    std::cout << "[LR]   area-centroid lean (informational): "
              << (info.lean_area >= 0 ? "+" : "") << info.lean_area
              << "  (" << (info.sign_area ? "+d_LR" : "-d_LR")
              << " is right by lean)" << std::endl;
    const char* primary = info.decisive
        ? "eclipse asymmetry (raycast, area-weighted)"
        : "area-centroid lean (fallback; eclipse signal too weak)";
    std::cout << "[LR]   -> primary = " << primary << std::endl;
    std::cout << "[LR]   -> +d_LR is "
              << (right_is_pos ? "RIGHT" : "LEFT") << std::endl;
    if (flip_manual) {
        std::cout << "[LR]   (flip_manual = TRUE, sign was inverted manually)"
                  << std::endl;
    }
    std::cout << "[LR] eclipse decision: " << ms_ec << " ms" << std::endl;

    // d_LR をサイン正規化 (+d_LR 方向が右になるように)
    if (!right_is_pos) d_lr = -d_lr;
    R.d_lr = d_lr;

    // ---- 頂点 mass: m[i] = (1/3) * Σ_{f∋i} face_area ----
    std::vector<float> vmass(nV, 0.0f);
    const size_t nT = mesh.mIndices.size() / 3;
    for (size_t f = 0; f < nT; f++) {
        GLuint i0 = mesh.mIndices[f*3];
        GLuint i1 = mesh.mIndices[f*3+1];
        GLuint i2 = mesh.mIndices[f*3+2];
        glm::vec3 v0(mesh.mVertices[i0*3], mesh.mVertices[i0*3+1], mesh.mVertices[i0*3+2]);
        glm::vec3 v1(mesh.mVertices[i1*3], mesh.mVertices[i1*3+1], mesh.mVertices[i1*3+2]);
        glm::vec3 v2(mesh.mVertices[i2*3], mesh.mVertices[i2*3+1], mesh.mVertices[i2*3+2]);
        float fa = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
        vmass[i0] += fa / 3.0f;
        vmass[i1] += fa / 3.0f;
        vmass[i2] += fa / 3.0f;
    }
    double total_mass_d = 0.0;
    for (float m : vmass) total_mass_d += double(m);
    float total_mass = float(total_mass_d);

    // ---- 投影 + 降順ソート + 二閾値 ----
    std::vector<float> p(nV);
    std::vector<int>   order(nV);
    for (int i = 0; i < nV; i++) {
        glm::vec3 v(mesh.mVertices[i*3],
                    mesh.mVertices[i*3+1],
                    mesh.mVertices[i*3+2]);
        p[i] = glm::dot(v, d_lr);
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return p[a] > p[b]; });   // descending

    // searchsorted (left): cum >= target を最初に満たす index
    float target_pure = right_pure_fraction * total_mass;
    float target_full = right_full_fraction * total_mass;
    int   cut_pure = nV - 1;
    int   cut_full = nV - 1;
    bool  found_pure = false, found_full = false;
    double cum_d = 0.0;
    for (int k = 0; k < nV; k++) {
        cum_d += double(vmass[order[k]]);
        if (!found_pure && float(cum_d) >= target_pure) {
            cut_pure = k;
            found_pure = true;
        }
        if (!found_full && float(cum_d) >= target_full) {
            cut_full = k;
            found_full = true;
            break;   // pure < full なので full を見つけた段階で終了
        }
    }
    cut_pure = std::min(std::max(cut_pure, 0), nV - 1);
    cut_full = std::min(std::max(cut_full, 0), nV - 1);
    R.p_thr_pure = p[order[cut_pure]];
    R.p_thr_full = p[order[cut_full]];

    std::cout << "[LR] right_pure_fraction = " << right_pure_fraction
              << "  right_full_fraction = " << right_full_fraction << std::endl;
    std::cout << "[LR] thresholds along d_LR : "
              << "p_thr_pure = " << R.p_thr_pure
              << "  p_thr_full = " << R.p_thr_full
              << "  (boundary band = " << (R.p_thr_pure - R.p_thr_full) << ")"
              << std::endl;

    // ---- 3 バンドラベル付け (Python と同じ順序で上書き) ----
    R.labels.assign(nV, PURE_LEFT);
    for (int i = 0; i < nV; i++) {
        if (p[i] >= R.p_thr_full) R.labels[i] = BOUNDARY;       // 右 + 境界
        if (p[i] >= R.p_thr_pure) R.labels[i] = PURE_RIGHT;     // 純右に上書き
    }

    // ---- 集計 ----
    R.n_pure_right = 0;
    R.n_boundary   = 0;
    R.n_pure_left  = 0;
    double mass_pure_R = 0.0, mass_bnd = 0.0, mass_pure_L = 0.0;
    for (int i = 0; i < nV; i++) {
        switch (R.labels[i]) {
            case PURE_RIGHT: R.n_pure_right++; mass_pure_R += vmass[i]; break;
            case BOUNDARY:   R.n_boundary++;   mass_bnd    += vmass[i]; break;
            default:         R.n_pure_left++;  mass_pure_L += vmass[i]; break;
        }
    }
    float a_pure_R = 0.0f, a_bnd = 0.0f, a_pure_L = 0.0f;
    {
        std::vector<uint8_t> m_r(nV, 0), m_b(nV, 0), m_l(nV, 0);
        for (int i = 0; i < nV; i++) {
            if (R.labels[i] == PURE_RIGHT) m_r[i] = 1;
            else if (R.labels[i] == BOUNDARY) m_b[i] = 1;
            else m_l[i] = 1;
        }
        a_pure_R = faceAreaSum(mesh, m_r);
        a_bnd    = faceAreaSum(mesh, m_b);
        a_pure_L = faceAreaSum(mesh, m_l);
    }
    float a_total = totalFaceArea(mesh);
    float a_extracted = a_pure_R + a_bnd + a_pure_L;
    float a_dropped = a_total - a_extracted;

    std::cout << "[LR] pure-R : boundary : pure-L  by vertex mass = "
              << (100.0 * mass_pure_R / std::max<double>(total_mass_d, 1e-12)) << " : "
              << (100.0 * mass_bnd    / std::max<double>(total_mass_d, 1e-12)) << " : "
              << (100.0 * mass_pure_L / std::max<double>(total_mass_d, 1e-12))
              << std::endl;
    std::cout << "[LR] pure-R : boundary : pure-L  by extracted face area = "
              << (100.0f * a_pure_R / std::max(a_extracted, 1e-9f)) << " : "
              << (100.0f * a_bnd    / std::max(a_extracted, 1e-9f)) << " : "
              << (100.0f * a_pure_L / std::max(a_extracted, 1e-9f))
              << std::endl;
    std::cout << "[LR] vertex counts: PURE_R=" << R.n_pure_right
              << "  BOUNDARY=" << R.n_boundary
              << "  PURE_L=" << R.n_pure_left
              << "  (total=" << nV << ")" << std::endl;
    std::cout << "[LR] boundary faces dropped: "
              << a_dropped << " ("
              << (100.0f * a_dropped / std::max(a_total, 1e-9f))
              << "% of total area)" << std::endl;

    auto t1 = std::chrono::steady_clock::now();
    double ms_total =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[LR] labelVertices done in " << ms_total << " ms"
              << std::endl;

    return R;
}


// ===================================================================
//  描画用 subsample (LiverRegionLabel::sampleVertexIndices と同形)
// ===================================================================
inline std::vector<int> sampleVertexIndices(
    const std::vector<uint8_t>& labels,
    uint8_t target_label,
    int max_points)
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
    return out;
}


// ===================================================================
//  Quadrant (4 象限) 関連 — Ctrl+G (V3-R) で使う
//  -------------------------------------------------------------------
//  AP軸 (anterior/rim/posterior) と LR軸 (pure_R/boundary/pure_L) の
//  組合せから 4 象限を合成する。rim と boundary は両属するので、
//  例えば「anterior_core ∩ boundary」の頂点は ant_R と ant_L の両方に
//  所属する (案D・重複所属方式)。
//
//  Ctrl+G (V3-R) は QuadrantMask で複数象限を OR 選択し、対応する
//  頂点 index 集合を makeQuadrantSubsetIdx() で得て、CMA-ES 内部の
//  KDTree に渡す subset として使う。重複所属頂点は OR 集合で 1 回だけ
//  含まれる (deduplicated).
//
//  H キーの可視化 (main.cpp::recomputeLiverQuad) はこれとは独立した
//  ロジックで実装されているので、本節の関数を変更しても H キー表示は
//  影響を受けない。
// ===================================================================

// 4-quadrant 選択を表すビットマスク。
// QUAD_LEGACY_FULL (0xFF) は PoseLibrary 既存エントリ (Ctrl+G 登場前)
// との後方互換のための値で、新規エントリでは使わない。
enum QuadrantMask : uint8_t {
    QUAD_NONE        = 0,
    QUAD_AR          = 1 << 0,    // 0x01  ant_right (緑)
    QUAD_AL          = 1 << 1,    // 0x02  ant_left  (紫)
    QUAD_PR          = 1 << 2,    // 0x04  pos_right (青)
    QUAD_PL          = 1 << 3,    // 0x08  pos_left  (橙)
    QUAD_ALL         = 0x0F,      // 全選択 (= V3 の subset と同じ全頂点)
    QUAD_LEGACY_FULL = 0xFF,      // 後方互換用 (PoseLibrary の旧エントリ)
};

// 選択された象限に属する頂点 index の集合を返す (重複排除済み、natural order)。
//
// 引数:
//   region_labels : LiverRegionLabel::Result::labels (ANTERIOR_CORE/RIM/POSTERIOR)
//   lr_labels     : LiverLeftRightLabel::Result::labels (PURE_RIGHT/BOUNDARY/PURE_LEFT)
//   quadrant_mask : QUAD_AR / QUAD_AL / QUAD_PR / QUAD_PL の OR
//
// 戻り値:
//   選択象限のいずれかに属する頂点 index 配列 (昇順、重複なし)。
//   QUAD_ALL を渡した場合: 全頂点が含まれる (= V3 と同じ集合) ことが byte-identical
//                          検証の前提条件。
//   QUAD_NONE / 不正値: 空配列を返す。
//
// 重複所属の判定 (案D・overlap-allowed):
//   is_ant = (R == ANTERIOR_CORE) || (R == RIM)         // rim は前後両属
//   is_pos = (R == POSTERIOR)     || (R == RIM)
//   is_R   = (L == PURE_RIGHT)    || (L == BOUNDARY)    // boundary は左右両属
//   is_L   = (L == PURE_LEFT)     || (L == BOUNDARY)
//   各象限への所属判定:
//     ant_R: is_ant && is_R
//     ant_L: is_ant && is_L
//     pos_R: is_pos && is_R
//     pos_L: is_pos && is_L
inline std::vector<int> makeQuadrantSubsetIdx(
    const std::vector<uint8_t>& region_labels,
    const std::vector<uint8_t>& lr_labels,
    uint8_t                      quadrant_mask)
{
    if (region_labels.empty() || lr_labels.empty() ||
        region_labels.size() != lr_labels.size() ||
        (quadrant_mask & QUAD_ALL) == 0)
    {
        return {};
    }

    std::vector<int> out;
    out.reserve(region_labels.size());
    for (size_t i = 0; i < region_labels.size(); i++) {
        const uint8_t R = region_labels[i];
        const uint8_t L = lr_labels[i];

        // 重複所属判定 (案D)
        const bool is_ant = (R == LiverRegionLabel::ANTERIOR_CORE) ||
                            (R == LiverRegionLabel::RIM);
        const bool is_pos = (R == LiverRegionLabel::POSTERIOR) ||
                            (R == LiverRegionLabel::RIM);
        const bool is_R   = (L == PURE_RIGHT) || (L == BOUNDARY);
        const bool is_L   = (L == PURE_LEFT)  || (L == BOUNDARY);

        bool hit = false;
        if ((quadrant_mask & QUAD_AR) && is_ant && is_R) hit = true;
        if ((quadrant_mask & QUAD_AL) && is_ant && is_L) hit = true;
        if ((quadrant_mask & QUAD_PR) && is_pos && is_R) hit = true;
        if ((quadrant_mask & QUAD_PL) && is_pos && is_L) hit = true;

        if (hit) out.push_back(static_cast<int>(i));
    }
    return out;
}

// 各象限ごとの頂点数を数える (UIで「ant_R: 3500 v」のように表示する用)。
// rim/boundary 両属頂点は該当する全象限でカウントされるので、
// (n_AR + n_AL + n_PR + n_PL) は通常 region_labels.size() を超える。
inline void countByQuadrant(
    const std::vector<uint8_t>& region_labels,
    const std::vector<uint8_t>& lr_labels,
    int& n_AR, int& n_AL, int& n_PR, int& n_PL)
{
    n_AR = n_AL = n_PR = n_PL = 0;
    if (region_labels.empty() || lr_labels.empty() ||
        region_labels.size() != lr_labels.size())
    {
        return;
    }
    for (size_t i = 0; i < region_labels.size(); i++) {
        const uint8_t R = region_labels[i];
        const uint8_t L = lr_labels[i];
        const bool is_ant = (R == LiverRegionLabel::ANTERIOR_CORE) ||
                            (R == LiverRegionLabel::RIM);
        const bool is_pos = (R == LiverRegionLabel::POSTERIOR) ||
                            (R == LiverRegionLabel::RIM);
        const bool is_R   = (L == PURE_RIGHT) || (L == BOUNDARY);
        const bool is_L   = (L == PURE_LEFT)  || (L == BOUNDARY);
        if (is_ant && is_R) n_AR++;
        if (is_ant && is_L) n_AL++;
        if (is_pos && is_R) n_PR++;
        if (is_pos && is_L) n_PL++;
    }
}

// QuadrantMask を表示用文字列に変換。
// 例:
//   0x00       -> "Q:NONE"
//   0x01       -> "Q:AR"
//   0x03       -> "Q:AR+AL"
//   0x0F       -> "Q:ALL"
//   0xFF       -> "FULL"      (legacy, PoseLibrary 旧エントリ)
//   その他    -> "Q:?(0xNN)"
inline std::string quadrantMaskString(uint8_t mask)
{
    if (mask == QUAD_LEGACY_FULL) return "FULL";
    if (mask == QUAD_NONE)        return "Q:NONE";
    if (mask == QUAD_ALL)         return "Q:ALL";

    // 4 ビット以外が立っていたら不正値表示
    if (mask & ~QUAD_ALL) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "Q:?(0x%02X)", (unsigned)mask);
        return std::string(buf);
    }

    std::string s = "Q:";
    bool first = true;
    auto add = [&](const char* name) {
        if (!first) s += "+";
        s += name;
        first = false;
    };
    if (mask & QUAD_AR) add("AR");
    if (mask & QUAD_AL) add("AL");
    if (mask & QUAD_PR) add("PR");
    if (mask & QUAD_PL) add("PL");
    return s;
}

}  // namespace LiverLeftRightLabel

#endif  // LIVER_LEFT_RIGHT_LABEL_H
