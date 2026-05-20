#pragma once
#ifndef LIVER_CRANIO_CAUDAL_LABEL_H
#define LIVER_CRANIO_CAUDAL_LABEL_H

/*
 * LiverCranioCaudalLabel.h
 * ----------------------------------------------------------------------
 * 肝臓メッシュの頂点を cranial(頭側) / caudal(足側) の 2 領域に
 * ラベル付けする。Python 側 liver_craniocaudal_v7_4patients.py を
 * C++ で再実装したもの (4 患者全例正解の確定アルゴリズム)。
 *
 * 既存の LiverRegionLabel.h (anterior/rim/posterior) および
 * LiverLeftRightLabel.h (pure_R/boundary/pure_L) と完全パラレルの
 * 構造で、main.cpp 側に Shift+H キーで統合する。最終的には
 * Initial Orientation の幾何ベース化 (Phase 2) で使う予定。
 *
 * 手法 v7 (5 ステップ):
 *   1. RIM 抽出: LiverRegionLabel::Result::labels で RIM ラベルを取得
 *   2. PCA 3 軸を再計算 (LiverLeftRightLabel::computeAllPCAAxes を流用)。
 *      LR で選ばれなかった軸 (1 または 2) を d_CC_raw とする
 *      (引き継ぎ書 §1: 「extent 大 = d_LR、残り = d_CC_raw」決定通り)
 *   3. per-face dihedral angle 平均 → per-vertex roughness 集約
 *   4. RIM 頂点を ±d_CC_raw 半球に分割し、area-weighted mean roughness
 *      を半球ごとに計算
 *   5. roughness 高い側を caudal とする (下縁が鋭い解剖学的事実)
 *
 * 出力ラベル:
 *   CRANIAL = 0   黄に塗る頭側
 *   CAUDAL  = 1   青に塗る足側
 *
 * 使い方:
 *   LiverCranioCaudalLabel::Result r =
 *       LiverCranioCaudalLabel::labelVertices(*liverMesh3D,
 *                                             g_liverRegion,
 *                                             g_liverLR);
 *   if (r.valid()) {
 *       // r.labels[i] is one of: CRANIAL / CAUDAL
 *       // r.d_cc は signed (+d_cc が CRANIAL 方向)
 *   }
 *
 * 既存 LiverRegionLabel / LiverLeftRightLabel との関係:
 *   - 依存: 両方の Result.valid() が必要 (RIM mask + LR axis idx)
 *   - 副作用なし: 入力の Result は変更しない
 *   - computeAllPCAAxes を再呼び出し (deterministic、結果は LR と一致)
 *
 * Python 検証実績 (confidence):
 *   Pt1=6.1%  Pt2=11.7%  Pt3=8.3%  Pt4=4.7%
 * confidence < 5% で "WEAK" 警告を std::cout に出力する (UI flip 推奨)。
 *
 * 計算は頂点 index に紐づくので、registration で transform しても
 * ラベルは不変 (再計算不要)。メッシュ自体を再ロードするときだけ
 * 再計算する。
 */

#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cfloat>
#include <utility>

#include <glm/glm.hpp>
#include <Eigen/Dense>

#include "mCutMesh.h"
#include "LiverRegionLabel.h"     // RIM ラベル + fetchVertex を流用
#include "LiverLeftRightLabel.h"  // computeAllPCAAxes + Result を流用

namespace LiverCranioCaudalLabel {

// ===================================================================
//  ラベル定義
// ===================================================================
enum Label : uint8_t {
    CRANIAL = 0,   // 頭側 (黄)
    CAUDAL  = 1,   // 足側 (青)
};

// ===================================================================
//  診断情報 (v7 の中身を全部公開しておく)
// ===================================================================
struct CCInfo {
    float mean_plus  = 0.0f;   // +d_cc_raw 側 RIM の area-weighted mean roughness
    float mean_minus = 0.0f;   // -d_cc_raw 側 RIM の area-weighted mean roughness
    float area_plus  = 0.0f;   // 上記 + 側の総面積 (sanity check 用)
    float area_minus = 0.0f;   // 上記 - 側の総面積
    int   n_rim_plus  = 0;     // + 側 RIM 頂点数
    int   n_rim_minus = 0;     // - 側 RIM 頂点数
    float confidence  = 0.0f;  // |Δ| / sum, 0..1
    bool  weak        = false; // confidence < 0.05
    bool  flipped_manual = false;  // ユーザー override で反転したか
};

// ===================================================================
//  Result struct (LiverLeftRightLabel::Result と同形パターン)
// ===================================================================
struct Result {
    std::vector<uint8_t> labels;       // per-vertex (size = nV)
    glm::vec3            d_cc;         // sign-normalised: +d_cc が CRANIAL
    glm::vec3            bbox_center;
    float                bbox_diag = 0.0f;
    CCInfo               cc;
    int                  n_cranial = 0;
    int                  n_caudal  = 0;

    // PCA 詳細 (デバッグ用)
    int                  cc_axis_idx = -1;   // 1=mid eigvec, 2=max eigvec
    glm::vec3            d_cc_raw    = glm::vec3(0.0f);

    bool valid() const { return !labels.empty(); }
};


// ===================================================================
//  内部ヘルパ
// ===================================================================

// face → vertex フェッチ簡略 (LiverRegionLabel と同じ規約)
inline glm::vec3 vertAt(const mCutMesh& mesh, uint32_t v) {
    return glm::vec3(mesh.mVertices[v*3],
                     mesh.mVertices[v*3 + 1],
                     mesh.mVertices[v*3 + 2]);
}

// -------------------------------------------------------------------
//  perFaceRoughness:
//    各 face f に対して、edge を共有する隣接 face との間の
//    dihedral angle を平均する (単位 rad、[0, pi])。
//    境界 face (隣接無し edge を含む) はその edge を寄与から除外。
//    実装: edge → (face_a, face_b) のハッシュマップを構築し、
//          内部 edge ごとに 1 回二面角を計算して両 face に加算。
//          最後に face ごとの加算回数で割る。
//    計算量: O(F) 程度 (5万 face で数百 ms)。
// -------------------------------------------------------------------
inline std::vector<float> perFaceRoughness(const mCutMesh& mesh) {
    const int nT = (int)(mesh.mIndices.size() / 3);
    std::vector<float> face_rough(nT, 0.0f);
    if (nT == 0) return face_rough;

    // 1. Face normals
    std::vector<glm::vec3> fnormals(nT);
    for (int t = 0; t < nT; t++) {
        uint32_t i0 = mesh.mIndices[t*3];
        uint32_t i1 = mesh.mIndices[t*3 + 1];
        uint32_t i2 = mesh.mIndices[t*3 + 2];
        glm::vec3 v0 = vertAt(mesh, i0);
        glm::vec3 v1 = vertAt(mesh, i1);
        glm::vec3 v2 = vertAt(mesh, i2);
        glm::vec3 n  = glm::cross(v1 - v0, v2 - v0);
        float len = glm::length(n);
        fnormals[t] = (len > 1e-20f) ? (n / len) : glm::vec3(0.0f);
    }

    // 2. Edge → faces map
    auto makeKey = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a > b) std::swap(a, b);
        return ((uint64_t)a << 32) | (uint64_t)b;
    };
    std::unordered_map<uint64_t, std::pair<int,int>> e2f;
    e2f.reserve((size_t)nT * 2);
    for (int t = 0; t < nT; t++) {
        uint32_t i0 = mesh.mIndices[t*3];
        uint32_t i1 = mesh.mIndices[t*3 + 1];
        uint32_t i2 = mesh.mIndices[t*3 + 2];
        uint64_t keys[3] = { makeKey(i0, i1), makeKey(i1, i2), makeKey(i2, i0) };
        for (uint64_t k : keys) {
            auto it = e2f.find(k);
            if (it == e2f.end()) {
                e2f.emplace(k, std::make_pair(t, -1));
            } else if (it->second.second < 0) {
                it->second.second = t;
            }
            // else: non-manifold edge (3+ faces); ignore extras
        }
    }

    // 3. Dihedral angle per internal edge → accumulate on both faces
    std::vector<int> face_count(nT, 0);
    for (const auto& kv : e2f) {
        int t0 = kv.second.first;
        int t1 = kv.second.second;
        if (t1 < 0) continue;  // boundary edge
        float c = glm::dot(fnormals[t0], fnormals[t1]);
        if (c >  1.0f) c =  1.0f;
        if (c < -1.0f) c = -1.0f;
        float ang = std::acos(c);   // [0, pi]
        face_rough[t0] += ang;
        face_rough[t1] += ang;
        face_count[t0]++;
        face_count[t1]++;
    }

    // 4. Mean
    for (int t = 0; t < nT; t++) {
        if (face_count[t] > 0) face_rough[t] /= (float)face_count[t];
    }
    return face_rough;
}

// -------------------------------------------------------------------
//  perVertexRoughness:
//    vertex_roughness[v] = mean over incident faces of face_roughness.
//    Python 版 (trimesh) の挙動と同等。
// -------------------------------------------------------------------
inline std::vector<float> perVertexRoughness(
    const mCutMesh& mesh,
    const std::vector<float>& face_rough)
{
    const int nV = (int)(mesh.mVertices.size() / 3);
    const int nT = (int)(mesh.mIndices.size() / 3);
    std::vector<float> vert_rough(nV, 0.0f);
    std::vector<int>   vert_count(nV, 0);
    for (int t = 0; t < nT; t++) {
        float r = face_rough[t];
        for (int k = 0; k < 3; k++) {
            int v = (int)mesh.mIndices[t*3 + k];
            if (v < 0 || v >= nV) continue;
            vert_rough[v] += r;
            vert_count[v]++;
        }
    }
    for (int v = 0; v < nV; v++) {
        if (vert_count[v] > 0) vert_rough[v] /= (float)vert_count[v];
    }
    return vert_rough;
}

// -------------------------------------------------------------------
//  computeVertexAreas:
//    Standard "barycentric area": A_v = (sum of incident face areas) / 3.
//    重み付き平均で半球比較するときの分母として使う。
// -------------------------------------------------------------------
inline std::vector<float> computeVertexAreas(const mCutMesh& mesh) {
    const int nV = (int)(mesh.mVertices.size() / 3);
    const int nT = (int)(mesh.mIndices.size() / 3);
    std::vector<float> va(nV, 0.0f);
    for (int t = 0; t < nT; t++) {
        uint32_t i0 = mesh.mIndices[t*3];
        uint32_t i1 = mesh.mIndices[t*3 + 1];
        uint32_t i2 = mesh.mIndices[t*3 + 2];
        glm::vec3 v0 = vertAt(mesh, i0);
        glm::vec3 v1 = vertAt(mesh, i1);
        glm::vec3 v2 = vertAt(mesh, i2);
        float A = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
        float third = A / 3.0f;
        if ((int)i0 < nV) va[i0] += third;
        if ((int)i1 < nV) va[i1] += third;
        if ((int)i2 < nV) va[i2] += third;
    }
    return va;
}

// -------------------------------------------------------------------
//  decideCCSignV7:
//    RIM 上の area-weighted mean roughness を ±d_cc_raw 半球で比較し、
//    roughness 高い側を caudal とする (引き継ぎ書 §2.2 Step 4-5)。
//    bbox_center を原点とした射影で半球分け。
//    結果:
//      info に診断情報を格納
//      d_cc_signed に「CRANIAL 方向に正の単位ベクトル」を格納
// -------------------------------------------------------------------
inline void decideCCSignV7(
    const mCutMesh& mesh,
    const LiverRegionLabel::Result& region,
    const std::vector<float>& vert_rough,
    const std::vector<float>& vert_areas,
    const glm::vec3& d_cc_raw,
    const glm::vec3& bbox_center,
    CCInfo& info_out,
    glm::vec3& d_cc_signed_out)
{
    double R_plus  = 0.0, A_plus  = 0.0;
    double R_minus = 0.0, A_minus = 0.0;
    int    n_plus = 0, n_minus = 0;

    const int nV = (int)(mesh.mVertices.size() / 3);
    for (int v = 0; v < nV; v++) {
        if (v >= (int)region.labels.size()) break;
        if (region.labels[v] != LiverRegionLabel::RIM) continue;

        glm::vec3 p = vertAt(mesh, v);
        float proj = glm::dot(p - bbox_center, d_cc_raw);
        float a = (v < (int)vert_areas.size()) ? vert_areas[v] : 0.0f;
        float r = (v < (int)vert_rough.size()) ? vert_rough[v] : 0.0f;

        if (proj > 0.0f) {
            R_plus += (double)r * (double)a;
            A_plus += (double)a;
            n_plus++;
        } else {
            R_minus += (double)r * (double)a;
            A_minus += (double)a;
            n_minus++;
        }
    }

    info_out.area_plus  = (float)A_plus;
    info_out.area_minus = (float)A_minus;
    info_out.n_rim_plus  = n_plus;
    info_out.n_rim_minus = n_minus;
    info_out.mean_plus  = (A_plus  > 1e-20) ? (float)(R_plus  / A_plus ) : 0.0f;
    info_out.mean_minus = (A_minus > 1e-20) ? (float)(R_minus / A_minus) : 0.0f;

    float sum = info_out.mean_plus + info_out.mean_minus;
    info_out.confidence = (sum > 1e-20f)
        ? std::fabs(info_out.mean_plus - info_out.mean_minus) / sum
        : 0.0f;
    info_out.weak = (info_out.confidence < 0.05f);

    // Sign decision:
    //   roughness が大きい側 = 下縁が乗っている = caudal
    //   よって d_cc_signed = (-d_cc_raw if mean_plus > mean_minus else +d_cc_raw)
    //   とすれば +d_cc_signed は CRANIAL 方向になる。
    float sign = (info_out.mean_plus > info_out.mean_minus) ? -1.0f : 1.0f;
    d_cc_signed_out = sign * d_cc_raw;
}


// ===================================================================
//  メイン関数
// ===================================================================
//   region       : LiverRegionLabel::Result (RIM mask の参照元)
//   lr           : LiverLeftRightLabel::Result (lr_axis_idx から
//                  CC 候補軸を選ぶ; d_lr 自体は本関数では使わない)
//   flip_manual  : true で d_cc を反転 (Python の FLIP_OVERRIDE 相当)
//
//   ログ出力フォーマットは Python (liver_craniocaudal_v7_4patients.py)
//   と同じ並びにしてあるので、ターミナルで diff 比較できる。
inline Result labelVertices(const mCutMesh& mesh,
                            const LiverRegionLabel::Result& region,
                            const LiverLeftRightLabel::Result& lr,
                            bool flip_manual = false)
{
    Result R;
    const int nV = (int)(mesh.mVertices.size() / 3);
    if (nV < 3 || mesh.mIndices.size() < 3) {
        std::cerr << "[CC] mesh empty (nV=" << nV
                  << ", nT=" << mesh.mIndices.size()/3 << ")\n";
        return R;
    }
    if (!region.valid()) {
        std::cerr << "[CC] LiverRegion (Shift+R) result is not valid\n";
        return R;
    }
    if (!lr.valid()) {
        std::cerr << "[CC] LiverLeftRight (Y) result is not valid\n";
        return R;
    }
    if ((int)region.labels.size() != nV) {
        std::cerr << "[CC] region.labels size mismatch ("
                  << region.labels.size() << " vs nV=" << nV
                  << ") -- mesh may have been reloaded; please recompute Shift+R\n";
        return R;
    }
    if ((int)lr.labels.size() != nV) {
        std::cerr << "[CC] lr.labels size mismatch ("
                  << lr.labels.size() << " vs nV=" << nV
                  << ") -- mesh may have been reloaded; please recompute Y\n";
        return R;
    }
    if (lr.lr_axis_idx != 1 && lr.lr_axis_idx != 2) {
        std::cerr << "[CC] lr.lr_axis_idx invalid: " << lr.lr_axis_idx
                  << " (expected 1 or 2)\n";
        return R;
    }

    std::cout << "[CC] V=" << nV
              << "  F=" << (mesh.mIndices.size()/3)
              << "  flip_manual=" << (flip_manual ? "true" : "false")
              << std::endl;

    auto t0 = std::chrono::steady_clock::now();

    // ---- bbox ----
    glm::vec3 mn( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < nV; i++) {
        glm::vec3 v = vertAt(mesh, i);
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    R.bbox_center = 0.5f * (mn + mx);
    R.bbox_diag   = glm::length(mx - mn);

    std::cout << "[CC] diag=" << R.bbox_diag << std::endl;

    // ---- PCA 全 3 軸 (LiverLeftRightLabel から流用) ----
    //   LR 計算時と同じ結果になる (deterministic) が、Result に
    //   3 軸が保存されていないので再計算する。コスト < 10ms。
    glm::vec3 axes[3];
    Eigen::Vector3d evals;
    glm::vec3 mean_xyz;
    LiverLeftRightLabel::computeAllPCAAxes(mesh, axes, evals, mean_xyz);

    // ---- CC raw 軸選択: LR で使われた軸の反対側 ----
    //   引き継ぎ書 §1: LR は (axes[1], axes[2]) のうち extent 大が選ばれる。
    //   残った方が CC raw 軸 (符号未確定)。
    R.cc_axis_idx = (lr.lr_axis_idx == 1) ? 2 : 1;
    R.d_cc_raw = axes[R.cc_axis_idx];
    std::cout << "[CC] LR axis idx=" << lr.lr_axis_idx
              << "  -> CC axis idx=" << R.cc_axis_idx
              << "  d_cc_raw=[" << R.d_cc_raw.x
              << "  " << R.d_cc_raw.y
              << "  " << R.d_cc_raw.z << "]"
              << std::endl;

    auto t1 = std::chrono::steady_clock::now();

    // ---- per-face dihedral roughness ----
    std::vector<float> face_rough = perFaceRoughness(mesh);

    auto t2 = std::chrono::steady_clock::now();

    // ---- per-vertex roughness + areas ----
    std::vector<float> vert_rough = perVertexRoughness(mesh, face_rough);
    std::vector<float> vert_areas = computeVertexAreas(mesh);

    auto t3 = std::chrono::steady_clock::now();

    // ---- v7 sign decision on RIM ----
    decideCCSignV7(mesh, region, vert_rough, vert_areas,
                   R.d_cc_raw, R.bbox_center, R.cc, R.d_cc);

    // ---- Manual flip (Python FLIP_OVERRIDE 相当) ----
    if (flip_manual) {
        R.d_cc = -R.d_cc;
        R.cc.flipped_manual = true;
    }

    auto t4 = std::chrono::steady_clock::now();

    // ---- diagnostics log (Python v7 と同じフォーマット) ----
    std::cout << "[CC] mean_plus="  << R.cc.mean_plus
              << "  mean_minus=" << R.cc.mean_minus
              << "  area_plus="  << R.cc.area_plus
              << "  area_minus=" << R.cc.area_minus
              << "  n_rim+=" << R.cc.n_rim_plus
              << "  n_rim-=" << R.cc.n_rim_minus
              << std::endl;
    std::cout << "[CC] confidence=" << (R.cc.confidence * 100.0f) << "%"
              << (R.cc.weak ? "  [WEAK]" : "  [OK]")
              << "  flipped_manual=" << (R.cc.flipped_manual ? "true" : "false")
              << "  d_cc=[" << R.d_cc.x
              << "  " << R.d_cc.y
              << "  " << R.d_cc.z << "]  (+d_cc -> CRANIAL)"
              << std::endl;
    if (R.cc.weak) {
        std::cout << "[CC] WARNING: confidence < 5%. "
                     "CC sign may be unreliable. "
                     "Inspect the visualization; "
                     "set flip_manual=true to override if needed."
                  << std::endl;
    }

    // ---- per-vertex label assignment ----
    //   proj > 0 → CRANIAL, else CAUDAL (d_cc は +CRANIAL に正規化済み)
    R.labels.assign(nV, (uint8_t)CAUDAL);
    R.n_cranial = 0;
    R.n_caudal  = 0;
    for (int v = 0; v < nV; v++) {
        glm::vec3 p = vertAt(mesh, v);
        float proj = glm::dot(p - R.bbox_center, R.d_cc);
        if (proj > 0.0f) {
            R.labels[v] = (uint8_t)CRANIAL;
            R.n_cranial++;
        } else {
            R.labels[v] = (uint8_t)CAUDAL;
            R.n_caudal++;
        }
    }

    auto t5 = std::chrono::steady_clock::now();
    using ms = std::chrono::duration<double, std::milli>;
    double ms_face   = ms(t2 - t1).count();
    double ms_vert   = ms(t3 - t2).count();
    double ms_decide = ms(t4 - t3).count();
    double ms_label  = ms(t5 - t4).count();
    double ms_total  = ms(t5 - t0).count();

    std::cout << "[CC] cranial=" << R.n_cranial
              << "  caudal=" << R.n_caudal
              << "  (cranial/total="
              << (100.0f * R.n_cranial / std::max(nV, 1)) << "%)\n";
    std::cout << "[CC] timing: face_rough=" << ms_face
              << " vert_rough+areas=" << ms_vert
              << " decide_sign=" << ms_decide
              << " label=" << ms_label
              << " TOTAL=" << ms_total << " ms\n";

    return R;
}


// ===================================================================
//  可視化用ヘルパ (LiverRegionLabel / LiverLeftRightLabel と同形)
//
//  ラベルごとに最大 max_points 個を決定論的に subsample する。返り値は
//  「mCutMesh の頂点 index」配列なので、登録後のメッシュ移動にも
//  mVertices[idx*3] で追従可能 (Shift+B と同じパターン)。
// ===================================================================
inline std::vector<int> sampleVertexIndices(
    const std::vector<uint8_t>& labels,
    uint8_t target_label,
    int max_points)
{
    std::vector<int> all;
    all.reserve(labels.size() / 2 + 1);
    for (int i = 0; i < (int)labels.size(); i++) {
        if (labels[i] == target_label) all.push_back(i);
    }
    if ((int)all.size() <= max_points) return all;

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

}  // namespace LiverCranioCaudalLabel

#endif  // LIVER_CRANIO_CAUDAL_LABEL_H
