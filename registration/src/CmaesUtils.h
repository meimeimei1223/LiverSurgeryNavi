#pragma once
/*
 * CmaesUtils.h
 * CMA-ES post-registration refinement for HemiAuto.
 * Wraps c-cmaes (Apache 2.0, Nikolaus Hansen 2014).
 *
 * Optimizes 7 DOF: [tx, ty, tz, rx, ry, rz, scale]
 * Objective: compRmse (Target->Source unified metric)
 *
 * Usage (in main.cpp onHemiAuto / GLFW_KEY_C):
 *   CmaesRefine::Params p;
 *   CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
 *                               registrationHandle, p,
 *                               gGridWidth, gGridHeight(), gDepthScale);
 *   if (r.improved) { ... apply already done inside run() }
 *   else            { ... vertices already restored inside run() }
 */

#ifndef CMAES_UTILS_H
#define CMAES_UTILS_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <functional>
#include <climits>
#include <numeric>
#include <map>
#include <queue>
#include <chrono>
#include <unordered_map>
#include <cstdlib>     /* srand() — used for deterministic CMA-ES seeding */
#ifdef _OPENMP
#include <omp.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mCutMesh.h"
/* stbi_load / stbi_image_free は mCutMesh.h の前方宣言経由で使用可能
   stbi_write_jpg は stb_image_write.h が main.cpp で定義済み */
#ifndef STBI_WRITE_NO_STDIO
extern int stbi_write_jpg(const char*, int, int, int, const void*, int);
extern int stbi_write_png(const char*, int, int, int, const void*, int);
#endif
#include "NoOpen3DRegistration.h"
#include "DepthUtils.h"
#include "PathConfig.h"

extern "C" {
#include "third_party/c-cmaes/cmaes.h"
}

/* RegistrationData forward declaration - defined in RegistrationUI.h */
struct RegistrationData;
extern RegistrationData registrationHandle;

/* boundary points computed in computeUnifiedMetrics() */
extern std::vector<glm::vec3> g_targetPoints;

/* camera / window globals from main.cpp */
extern int         gWindowWidth, gWindowHeight;
extern glm::mat4   view, projection;

/* forward declaration of computeUnifiedMetrics() defined in main.cpp */
static void computeUnifiedMetrics();

/* quiet flag: suppresses verbose logging during CMA-ES loop */
extern bool g_quietMetrics;

/* Scene scale (target AABB diagonal) — used for size-invariant
 * registration parameters. Defined in main.cpp. */
extern float g_sceneDiag;

namespace CmaesRefine {

/* ------------------------------------------------------------------ */
/* Silhouette debug: project source contour onto image, save JPG       */
/* ------------------------------------------------------------------ */

/* 3D点をスクリーン座標(px,py)に投影。画面外はfalseを返す */
inline bool projectToScreen(const glm::vec3& pt,
                            const glm::mat4& viewMat,
                            const glm::mat4& projMat,
                            int imgW, int imgH,
                            int& px, int& py)
{
    glm::vec4 clip = projMat * viewMat * glm::vec4(pt, 1.0f);
    if (clip.w <= 0.0f) return false;
    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.x < -1.0f || ndc.x > 1.0f ||
        ndc.y < -1.0f || ndc.y > 1.0f) return false;
    px = (int)(( ndc.x + 1.0f) * 0.5f * imgW);
    py = (int)((-ndc.y + 1.0f) * 0.5f * imgH);
    px = std::clamp(px, 0, imgW - 1);
    py = std::clamp(py, 0, imgH - 1);
    return true;
}

/* ================================================================
   BVH for silhouette computation
   毎回再ビルド方式: 頂点変形に完全追従
   ================================================================ */
struct SilBVHNode {
    glm::vec3 bmin, bmax;
    int left=-1, right=-1;
    int triBegin=-1, triEnd=-1;
    bool isLeaf() const { return triBegin>=0; }
};

struct SilBVH {
    std::vector<SilBVHNode> nodes;
    std::vector<int>        order;
    const std::vector<GLfloat>* V = nullptr;
    const std::vector<GLuint>*  I = nullptr;

    glm::vec3 vp(int ti, int i) const {
        int idx=(*I)[ti*3+i];
        return {(*V)[idx*3],(*V)[idx*3+1],(*V)[idx*3+2]};
    }
    glm::vec3 tMin(int ti) const { return glm::min(vp(ti,0),glm::min(vp(ti,1),vp(ti,2))); }
    glm::vec3 tMax(int ti) const { return glm::max(vp(ti,0),glm::max(vp(ti,1),vp(ti,2))); }

    static bool rayAABB(const glm::vec3& ro, const glm::vec3& inv,
                        const glm::vec3& mn, const glm::vec3& mx) {
        glm::vec3 t0=(mn-ro)*inv, t1=(mx-ro)*inv;
        glm::vec3 tlo=glm::min(t0,t1), thi=glm::max(t0,t1);
        float tmin=std::max({tlo.x,tlo.y,tlo.z});
        float tmax=std::min({thi.x,thi.y,thi.z});
        return tmax>=tmin && tmax>0.f;
    }
    bool rayTri(int ti, const glm::vec3& ro, const glm::vec3& rd,
                float maxD) const {
        glm::vec3 e1=vp(ti,1)-vp(ti,0), e2=vp(ti,2)-vp(ti,0);
        glm::vec3 h=glm::cross(rd,e2);
        float a=glm::dot(e1,h);
        if(std::abs(a)<1e-8f) return false;
        float f=1.f/a;
        glm::vec3 s=ro-vp(ti,0);
        float u=f*glm::dot(s,h);
        if(u<0.f||u>1.f) return false;
        glm::vec3 q=glm::cross(s,e1);
        float v=f*glm::dot(rd,q);
        if(v<0.f||u+v>1.f) return false;
        float t=f*glm::dot(e2,q);
        return t>1e-4f && t<maxD;
    }
    bool rayTriFront(int ti, const glm::vec3& ro, const glm::vec3& rd,
                     float maxD) const {
        glm::vec3 e1=vp(ti,1)-vp(ti,0), e2=vp(ti,2)-vp(ti,0);
        glm::vec3 h=glm::cross(rd,e2);
        float a=glm::dot(e1,h);
        if(a < 1e-8f) return false;
        float f=1.f/a;
        glm::vec3 s=ro-vp(ti,0);
        float u=f*glm::dot(s,h);
        if(u<0.f||u>1.f) return false;
        glm::vec3 q=glm::cross(s,e1);
        float v=f*glm::dot(rd,q);
        if(v<0.f||u+v>1.f) return false;
        float t=f*glm::dot(e2,q);
        return t>1e-4f && t<maxD;
    }
    int build(int begin, int end_) {
        SilBVHNode n;
        n.bmin=glm::vec3(1e30f); n.bmax=glm::vec3(-1e30f);
        for(int i=begin;i<end_;i++){
            n.bmin=glm::min(n.bmin,tMin(order[i]));
            n.bmax=glm::max(n.bmax,tMax(order[i]));
        }
        if(end_-begin<=4){n.triBegin=begin;n.triEnd=end_;nodes.push_back(n);return(int)nodes.size()-1;}
        glm::vec3 ext=n.bmax-n.bmin;
        int ax=(ext.x>ext.y&&ext.x>ext.z)?0:(ext.y>ext.z)?1:2;
        float mid=(n.bmin[ax]+n.bmax[ax])*0.5f;
        int m=(int)(std::partition(order.begin()+begin,order.begin()+end_,
                                      [&](int ti){return (tMin(ti)[ax]+tMax(ti)[ax])*0.5f<mid;})-order.begin());
        if(m==begin||m==end_) m=(begin+end_)/2;
        nodes.push_back(n); int idx=(int)nodes.size()-1;
        int l=build(begin,m); int r=build(m,end_);
        nodes[idx].left=l; nodes[idx].right=r;
        return idx;
    }
    void init(const mCutMesh* mesh) {
        V=&mesh->mVertices; I=&mesh->mIndices;
        int nTri=(int)(I->size()/3);
        order.resize(nTri); std::iota(order.begin(),order.end(),0);
        nodes.clear(); nodes.reserve(nTri*2);
        build(0,nTri);
    }
    bool intersect(int ni, const glm::vec3& ro,
                   const glm::vec3& rd, const glm::vec3& inv,
                   float maxD) const {
        const SilBVHNode& n=nodes[ni];
        if(!rayAABB(ro,inv,n.bmin,n.bmax)) return false;
        if(n.isLeaf()){
            for(int i=n.triBegin;i<n.triEnd;i++)
                if(rayTri(order[i],ro,rd,maxD)) return true;
            return false;
        }
        return intersect(n.left,ro,rd,inv,maxD)||
               intersect(n.right,ro,rd,inv,maxD);
    }
    bool intersectFront(int ni, const glm::vec3& ro,
                        const glm::vec3& rd, const glm::vec3& inv,
                        float maxD) const {
        const SilBVHNode& n=nodes[ni];
        if(!rayAABB(ro,inv,n.bmin,n.bmax)) return false;
        if(n.isLeaf()){
            for(int i=n.triBegin;i<n.triEnd;i++)
                if(rayTriFront(order[i],ro,rd,maxD)) return true;
            return false;
        }
        return intersectFront(n.left,ro,rd,inv,maxD)||
               intersectFront(n.right,ro,rd,inv,maxD);
    }
    bool hit(const glm::vec3& ro, const glm::vec3& rd, float maxD=1e30f) const {
        if(nodes.empty()) return false;
        glm::vec3 inv(1.f/(rd.x+1e-30f),1.f/(rd.y+1e-30f),1.f/(rd.z+1e-30f));
        return intersect(0,ro,rd,inv,maxD);
    }
    bool hitFront(const glm::vec3& ro, const glm::vec3& rd, float maxD=1e30f) const {
        if(nodes.empty()) return false;
        glm::vec3 inv(1.f/(rd.x+1e-30f),1.f/(rd.y+1e-30f),1.f/(rd.z+1e-30f));
        return intersectFront(0,ro,rd,inv,maxD);
    }
    /* レイとmeshの最近傍交点（3D座標）を返す */
    bool hitPoint(const glm::vec3& ro, const glm::vec3& rd,
                  glm::vec3& outPt, float maxD=1e30f) const {
        if(nodes.empty()) return false;
        float bestT = maxD;
        bool found = false;
        std::vector<int> stack; stack.reserve(64); stack.push_back(0);
        glm::vec3 inv(1.f/(rd.x+1e-30f),1.f/(rd.y+1e-30f),1.f/(rd.z+1e-30f));
        while (!stack.empty()) {
            int ni = stack.back(); stack.pop_back();
            const SilBVHNode& n = nodes[ni];
            if (!rayAABB(ro, inv, n.bmin, n.bmax)) continue;
            if (n.isLeaf()) {
                for (int i = n.triBegin; i < n.triEnd; i++) {
                    int ti = order[i];
                    glm::vec3 e1=vp(ti,1)-vp(ti,0), e2=vp(ti,2)-vp(ti,0);
                    glm::vec3 h=glm::cross(rd,e2);
                    float a=glm::dot(e1,h);
                    if(std::abs(a)<1e-8f) continue;
                    float f=1.f/a;
                    glm::vec3 s=ro-vp(ti,0);
                    float u=f*glm::dot(s,h);
                    if(u<0.f||u>1.f) continue;
                    glm::vec3 q=glm::cross(s,e1);
                    float v=f*glm::dot(rd,q);
                    if(v<0.f||u+v>1.f) continue;
                    float t=f*glm::dot(e2,q);
                    if(t>1e-4f && t<bestT){ bestT=t; found=true; }
                }
            } else {
                stack.push_back(n.left);
                stack.push_back(n.right);
            }
        }
        if(found) outPt = ro + bestT * rd;
        return found;
    }
};

/* 面法線（変形後頂点から毎回計算） */
inline glm::vec3 faceNormal(const std::vector<GLfloat>& v,
                            const std::vector<GLuint>&  idx, int t)
{
    glm::vec3 a(v[idx[t*3+0]*3+0],v[idx[t*3+0]*3+1],v[idx[t*3+0]*3+2]);
    glm::vec3 b(v[idx[t*3+1]*3+0],v[idx[t*3+1]*3+1],v[idx[t*3+1]*3+2]);
    glm::vec3 c(v[idx[t*3+2]*3+0],v[idx[t*3+2]*3+1],v[idx[t*3+2]*3+2]);
    return glm::normalize(glm::cross(b-a,c-a));
}

/* view行列からカメラ位置をワールド座標で取得 */
inline glm::vec3 camPosFromView(const glm::mat4& viewMat)
{
    glm::mat3 R(viewMat);
    glm::vec3 t(viewMat[3]);
    return -(glm::transpose(R)*t);
}

/* IoUベースシルエット目的関数
   fval = 1 - IoU(Source hitmap, Target mask)
   IoU = |Source∩Target| / |Source∪Target|
   → 偽の最適解なし、スケール誤差にも敏感 */
inline float computeSilhouette2DObjective(
    const mCutMesh*  liver,
    const SilBVH&    /*unused*/,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    int step = 4)
{
    if (!liver || !g_boundaryDistMap.valid) return 9.9f;

    SilBVH bvh;
    bvh.init(liver);

    int imgW = gWindowWidth, imgH = gWindowHeight;
    int mw   = g_boundaryDistMap.width;
    int mh   = g_boundaryDistMap.height;
    int gw   = (imgW+step-1)/step;
    int gh   = (imgH+step-1)/step;

    glm::mat4 invVP = glm::inverse(projMat * viewMat);
    std::vector<uint8_t> hitmap(gw*gh, 0);

/* hitmap構築 */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,4)
#endif
    for (int gy=0; gy<gh; gy++) {
        int py=gy*step+step/2;
        for (int gx=0; gx<gw; gx++) {
            int px=gx*step+step/2;
            float ndcX= 2.f*px/imgW-1.f;
            float ndcY=-2.f*py/imgH+1.f;
            glm::vec4 nH=invVP*glm::vec4(ndcX,ndcY,-1.f,1.f);
            glm::vec4 fH=invVP*glm::vec4(ndcX,ndcY, 1.f,1.f);
            glm::vec3 ro=glm::vec3(nH)/nH.w;
            glm::vec3 rd=glm::normalize(glm::vec3(fH)/fH.w-ro);
            hitmap[gy*gw+gx]=bvh.hit(ro,rd)?1:0;
        }
    }

    /* Target mask を hitmap と同スケールで構築。Fast 版と同じロジック:
       g_projectedLiverMask が valid ならそれを優先 (camera-aware)、
       無ければ g_boundaryDistMap を画面全体に線形引き伸ばし (legacy shortcut)。
       g_boundaryDistMap.data[i] < 9000 = マスク内側 */
    std::vector<uint8_t> targetMask(gw*gh, 0);
    const bool useProjected = (g_projectedLiverMask.valid &&
                               g_projectedLiverMask.width  == imgW &&
                               g_projectedLiverMask.height == imgH);
    if (useProjected) {
        const auto& pm = g_projectedLiverMask.data;
        for (int gy = 0; gy < gh; gy++)
            for (int gx = 0; gx < gw; gx++) {
                int ipx = gx*step + step/2, ipy = gy*step + step/2;
                ipx = std::clamp(ipx, 0, imgW - 1);
                ipy = std::clamp(ipy, 0, imgH - 1);
                targetMask[gy*gw + gx] = pm[ipy * imgW + ipx] ? 1 : 0;
            }
    } else {
        for (int gy=0;gy<gh;gy++)
            for (int gx=0;gx<gw;gx++) {
                int ipx=gx*step+step/2, ipy=gy*step+step/2;
                int mx=std::clamp(ipx*mw/imgW,0,mw-1);
                int my=std::clamp(ipy*mh/imgH,0,mh-1);
                targetMask[gy*gw+gx] = (g_boundaryDistMap.data[my*mw+mx]<9000.f)?1:0;
            }
    }

    /* IoU計算 */
    int inter=0, uni=0;
    for (int i=0;i<gw*gh;i++){
        bool s=hitmap[i], t=targetMask[i];
        if(s||t) uni++;
        if(s&&t) inter++;
    }

    if (!g_quietMetrics)
        std::cout << "[Sil2D] IoU=" << (uni>0?(float)inter/uni:0.f)
                  << "  inter=" << inter << "  uni=" << uni << std::endl;

    return (uni==0) ? 9.9f : 1.0f - (float)inter/uni;
}

/* ==========================================================================
   Fast 2D Silhouette Objective (Rasterization-based, no BVH)
   ==========================================================================
   computeSilhouette2DObjective のレイキャスト版と同等の結果を狙う高速版。
   各 eval で:
     1) 全頂点をスクリーン座標に変換 (行列乗算のみ、BVHなし)
     2) 前面向き三角形 (スクリーン空間 CCW) だけラスタライズ
     3) hitmap を構築し target mask と IoU
   BVH 構築 + O(pixels × BVH深さ) のレイキャストを排除することで、
   レイキャスト版より概ね 1〜2桁高速。
   互換性: 既存 computeSilhouette2DObjective は無改変、同じシグネチャの
   Fast 版を別関数として提供。比較実験用。
   ========================================================================== */
inline float computeSilhouette2DObjectiveFast(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    int step = 4,
    int* outInter = nullptr,
    int* outUnion = nullptr,
    double* outMs = nullptr,
    float* outHausdorff2D = nullptr)
{
    if (!liver || !g_boundaryDistMap.valid) return 9.9f;

    auto t0 = std::chrono::steady_clock::now();

    const int imgW = gWindowWidth, imgH = gWindowHeight;
    const int mw   = g_boundaryDistMap.width;
    const int mh   = g_boundaryDistMap.height;
    const int gw   = (imgW+step-1)/step;
    const int gh   = (imgH+step-1)/step;

    const auto& V = liver->mVertices;
    const auto& I = liver->mIndices;
    const int nVerts = (int)(V.size()/3);
    const int nTris  = (int)(I.size()/3);
    if (nVerts == 0 || nTris == 0) return 9.9f;

    /* Step 1: 全頂点をスクリーン空間へ変換 (x,y は hitmap grid 座標、z は深度参考) */
    glm::mat4 MVP = projMat * viewMat;
    std::vector<glm::vec3> screen(nVerts);  /* (gx_float, gy_float, z_ndc) */
    std::vector<uint8_t>   inside(nVerts, 0);

    const float halfW = imgW * 0.5f;
    const float halfH = imgH * 0.5f;
    const float invStep = 1.0f / (float)step;

    for (int i = 0; i < nVerts; i++) {
        glm::vec4 v(V[i*3], V[i*3+1], V[i*3+2], 1.0f);
        glm::vec4 c = MVP * v;
        if (std::abs(c.w) < 1e-8f) {
            screen[i] = glm::vec3(0.f, 0.f, 2.f);  /* 無効マーカー */
            continue;
        }
        float ndcX = c.x / c.w;
        float ndcY = c.y / c.w;
        float ndcZ = c.z / c.w;
        /* NDC → pixel → grid coord */
        float px = (ndcX + 1.f) * halfW;
        float py = (1.f - ndcY) * halfH;  /* Y flip to match computeSilhouette2DObjective */
        screen[i] = glm::vec3(px * invStep, py * invStep, ndcZ);
        /* 視野内フラグ: NDC z も前面(-1..1)にあること */
        inside[i] = (ndcX > -1.2f && ndcX < 1.2f &&
                     ndcY > -1.2f && ndcY < 1.2f &&
                     ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }

    /* Step 2: 三角形ラスタライズで hitmap を構築
       前面判定: スクリーン空間で 3頂点が CCW (signed area > 0) のものだけ描画。
       これは back-face culling と同等(view 依存は MVP 変換内で処理済み)。 */
    std::vector<uint8_t> hitmap(gw*gh, 0);

#ifdef _OPENMP
    /* 三角形ごとの並列化: 書き込み先が共通のため、各スレッドがローカル hitmap
       を持ち、最後にマージする。hitmap は 0/1 のみなので OR でマージ。 */
    int nThreads = omp_get_max_threads();
    std::vector<std::vector<uint8_t>> local(nThreads, std::vector<uint8_t>(gw*gh, 0));
#pragma omp parallel for schedule(dynamic, 64)
    for (int ti = 0; ti < nTris; ti++) {
        int tid = omp_get_thread_num();
        auto& hm = local[tid];
#else
    {
        auto& hm = hitmap;
        for (int ti = 0; ti < nTris; ti++) {
#endif
        int i0 = I[ti*3+0], i1 = I[ti*3+1], i2 = I[ti*3+2];
        if (!inside[i0] && !inside[i1] && !inside[i2]) continue;
        const glm::vec3& s0 = screen[i0];
        const glm::vec3& s1 = screen[i1];
        const glm::vec3& s2 = screen[i2];
        /* スクリーン空間 signed area (backface culling) */
        float area2 = (s1.x - s0.x) * (s2.y - s0.y) -
                      (s2.x - s0.x) * (s1.y - s0.y);
        if (area2 <= 0.0f) continue;  /* 裏向き or 縮退 */

        /* バウンディング矩形 (grid 座標、整数に丸め) */
        float minX = std::min({s0.x, s1.x, s2.x});
        float maxX = std::max({s0.x, s1.x, s2.x});
        float minY = std::min({s0.y, s1.y, s2.y});
        float maxY = std::max({s0.y, s1.y, s2.y});
        int x0 = std::max(0,  (int)std::floor(minX));
        int x1 = std::min(gw-1, (int)std::ceil(maxX));
        int y0 = std::max(0,  (int)std::floor(minY));
        int y1 = std::min(gh-1, (int)std::ceil(maxY));
        if (x0 > x1 || y0 > y1) continue;

        /* 重心座標でポイント-イン-トライアングル */
        float invArea = 1.0f / area2;
        for (int y = y0; y <= y1; y++) {
            float py = (float)y + 0.5f;
            for (int x = x0; x <= x1; x++) {
                float px = (float)x + 0.5f;
                float w0 = ((s1.x - px) * (s2.y - py) - (s2.x - px) * (s1.y - py)) * invArea;
                float w1 = ((s2.x - px) * (s0.y - py) - (s0.x - px) * (s2.y - py)) * invArea;
                float w2 = 1.0f - w0 - w1;
                if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    hm[y*gw + x] = 1;
            }
        }
#ifdef _OPENMP
    }
    /* マージ: OR 結合 */
    for (int t = 0; t < nThreads; t++) {
        const auto& src = local[t];
        for (int i = 0; i < gw*gh; i++) hitmap[i] |= src[i];
    }
#else
            }
    }
#endif

    /* Step 3: Target mask を構築。
       優先: g_projectedLiverMask が valid (= Shift+E 冒頭で buildProjectedLiverMask
             が呼ばれ、現カメラから screenMesh 越しに SAM2 mask を描画した window
             サイズの binary mask) なら、それを直接サンプル。source hitmap と
             同じ MVP で生成されたので、画面のどこに mask が見えているかが
             正しく反映される。カメラをスクロールした状態で Shift+E を押しても
             破綻しない。
       fallback: 旧 shortcut — g_boundaryDistMap を画面全体に線形引き伸ばし。
                 初期カメラで screenMesh が画面フィルする前提でのみ valid。
                 Shift+E 以外 (直接 metric 計算等) ではこちらが使われる。 */
    std::vector<uint8_t> targetMask(gw*gh, 0);
    const bool useProjected = (g_projectedLiverMask.valid &&
                               g_projectedLiverMask.width  == imgW &&
                               g_projectedLiverMask.height == imgH);
    if (useProjected) {
        const auto& pm = g_projectedLiverMask.data;   /* top-left origin, imgW × imgH */
        for (int gy = 0; gy < gh; gy++)
            for (int gx = 0; gx < gw; gx++) {
                int ipx = gx*step + step/2, ipy = gy*step + step/2;
                ipx = std::clamp(ipx, 0, imgW - 1);
                ipy = std::clamp(ipy, 0, imgH - 1);
                targetMask[gy*gw + gx] = pm[ipy * imgW + ipx] ? 1 : 0;
            }
    } else {
        for (int gy=0;gy<gh;gy++)
            for (int gx=0;gx<gw;gx++) {
                int ipx=gx*step+step/2, ipy=gy*step+step/2;
                int mx=std::clamp(ipx*mw/imgW,0,mw-1);
                int my=std::clamp(ipy*mh/imgH,0,mh-1);
                targetMask[gy*gw+gx] = (g_boundaryDistMap.data[my*mw+mx]<9000.f)?1:0;
            }
    }

    /* Step 4: IoU */
    int inter=0, uni=0;
    for (int i=0;i<gw*gh;i++){
        bool s=hitmap[i], t=targetMask[i];
        if(s||t) uni++;
        if(s&&t) inter++;
    }

    /* Step 5: 2D Hausdorff (シルエット境界の双方向最大距離、グリッドピクセル単位を画像px換算)
       - source 境界 ∂S: hitmap[i]=1 で 4近傍に hitmap[j]=0 がある画素
       - target 境界 ∂T: g_boundaryDistMap で bd<1.5 (外側1px)、ただし grid 解像度に合わせる
       - H_S2T = max_{p∈∂S} dist(p, ∂T)
       - H_T2S = max_{p∈∂T} dist(p, ∂S)
       - dist(., ∂T) は g_boundaryDistMap を grid stride で引くだけで済む
       - dist(., ∂S) は grid 解像度で BFS DT を構築 (gw*gh は通常 ~2万、軽量)
       片側のみ実装すると "source が target に内包" / "target が source に内包" のどちらかの
       外れ点を見落とすため、両側 max を取る。 */
    if (outHausdorff2D) {
        /* 5a. source 境界抽出 (grid 座標) */
        std::vector<std::pair<int,int>> srcBoundary;
        srcBoundary.reserve(gw + gh);
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                if (!hitmap[y*gw + x]) continue;
                bool isB = false;
                if (x == 0 || x == gw-1 || y == 0 || y == gh-1) {
                    isB = true;
                } else {
                    if (!hitmap[(y-1)*gw + x] || !hitmap[(y+1)*gw + x] ||
                        !hitmap[y*gw + (x-1)] || !hitmap[y*gw + (x+1)])
                        isB = true;
                }
                if (isB) srcBoundary.push_back({x, y});
            }
        }

        /* 5b. target 境界抽出 (grid 座標、targetMask の境界画素) */
        std::vector<std::pair<int,int>> tgtBoundary;
        tgtBoundary.reserve(gw + gh);
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                if (!targetMask[y*gw + x]) continue;
                bool isB = false;
                if (x == 0 || x == gw-1 || y == 0 || y == gh-1) {
                    isB = true;
                } else {
                    if (!targetMask[(y-1)*gw + x] || !targetMask[(y+1)*gw + x] ||
                        !targetMask[y*gw + (x-1)] || !targetMask[y*gw + (x+1)])
                        isB = true;
                }
                if (isB) tgtBoundary.push_back({x, y});
            }
        }

        /* 5c. ∂S の DT (BFS、L1 ではなく L2 近似のため Chamfer 3-4 weights を使う)
           簡易のため uint16 で Chamfer 距離 (3=直交,4=斜め) を計算、最後に /3 で px 距離。
           gw*gh が小さいのでこれで十分高速。 */
        const int INF16 = 65535;
        std::vector<int> dtSrc(gw*gh, INF16);
        {
            std::queue<std::pair<int,int>> q;
            for (auto& p : srcBoundary) {
                dtSrc[p.second*gw + p.first] = 0;
                q.push(p);
            }
            const int dx8[] = {1,-1, 0, 0, 1, 1,-1,-1};
            const int dy8[] = {0, 0, 1,-1, 1,-1, 1,-1};
            const int dw8[] = {3, 3, 3, 3, 4, 4, 4, 4};
            while (!q.empty()) {
                auto [cx, cy] = q.front(); q.pop();
                int cd = dtSrc[cy*gw + cx];
                for (int d = 0; d < 8; d++) {
                    int nx = cx + dx8[d], ny = cy + dy8[d];
                    if (nx < 0 || nx >= gw || ny < 0 || ny >= gh) continue;
                    int nd = cd + dw8[d];
                    if (nd < dtSrc[ny*gw + nx]) {
                        dtSrc[ny*gw + nx] = nd;
                        q.push({nx, ny});
                    }
                }
            }
        }

        /* 5d. ∂T の DT は g_boundaryDistMap (画像 px 単位) を grid 座標で引く。
           g_boundaryDistMap は target mask の各内側画素について boundary までの BFS 距離 (1画素=1unit)。
           grid → 画像 px 変換は Step 3 と同じロジック。 */
        auto distToTgtBoundaryPx = [&](int gx, int gy) -> float {
            int ipx = gx*step + step/2, ipy = gy*step + step/2;
            int mx = std::clamp(ipx*g_boundaryDistMap.width  / imgW, 0, g_boundaryDistMap.width  - 1);
            int my = std::clamp(ipy*g_boundaryDistMap.height / imgH, 0, g_boundaryDistMap.height - 1);
            float bd = g_boundaryDistMap.data[my*g_boundaryDistMap.width + mx];
            /* bd は target mask 内部での boundary までの距離。
               target 外部の画素は bd=9999 でマークされている → そのまま返してしまうと
               外側の source 画素について「target 境界までの距離」を直接得られない。
               外部画素については近傍の target 境界画素までユークリッド距離を BFS する
               必要があるが、Chamfer DT を target 境界で別途構築するのが綺麗。 */
            return bd;
        };

        /* g_boundaryDistMap は target 内側のみ DT が入っているため、
           source 境界が target 外側にある場合を扱えない。grid 解像度で
           target 境界からの DT を Chamfer で構築する。 */
        std::vector<int> dtTgt(gw*gh, INF16);
        {
            std::queue<std::pair<int,int>> q;
            for (auto& p : tgtBoundary) {
                dtTgt[p.second*gw + p.first] = 0;
                q.push(p);
            }
            const int dx8[] = {1,-1, 0, 0, 1, 1,-1,-1};
            const int dy8[] = {0, 0, 1,-1, 1,-1, 1,-1};
            const int dw8[] = {3, 3, 3, 3, 4, 4, 4, 4};
            while (!q.empty()) {
                auto [cx, cy] = q.front(); q.pop();
                int cd = dtTgt[cy*gw + cx];
                for (int d = 0; d < 8; d++) {
                    int nx = cx + dx8[d], ny = cy + dy8[d];
                    if (nx < 0 || nx >= gw || ny < 0 || ny >= gh) continue;
                    int nd = cd + dw8[d];
                    if (nd < dtTgt[ny*gw + nx]) {
                        dtTgt[ny*gw + nx] = nd;
                        q.push({nx, ny});
                    }
                }
            }
        }

        /* 5e. H_S2T と H_T2S の最大値 (Chamfer 単位 → grid 単位 → 画像 px 単位) */
        int maxS2T_chamfer = 0, maxT2S_chamfer = 0;
        for (auto& p : srcBoundary) {
            int v = dtTgt[p.second*gw + p.first];
            if (v < INF16 && v > maxS2T_chamfer) maxS2T_chamfer = v;
        }
        for (auto& p : tgtBoundary) {
            int v = dtSrc[p.second*gw + p.first];
            if (v < INF16 && v > maxT2S_chamfer) maxT2S_chamfer = v;
        }
        /* Chamfer 3-4 → 距離換算: grid_dist = chamfer / 3、画像 px = grid_dist * step */
        float h_s2t_px = (float)maxS2T_chamfer / 3.0f * (float)step;
        float h_t2s_px = (float)maxT2S_chamfer / 3.0f * (float)step;
        *outHausdorff2D = std::max(h_s2t_px, h_t2s_px);

        /* 任意ビューワー (将来 hover hint 等で使う) */
        (void)distToTgtBoundaryPx;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (outInter) *outInter = inter;
    if (outUnion) *outUnion = uni;
    if (outMs)    *outMs    = ms;

    if (!g_quietMetrics) {
        std::cout << "[Sil2D-Fast] IoU=" << (uni>0?(float)inter/uni:0.f)
        << "  inter=" << inter << "  uni=" << uni;
        if (outHausdorff2D)
            std::cout << "  H2D=" << std::fixed << std::setprecision(1)
                      << *outHausdorff2D << "px";
        std::cout << "  time=" << std::fixed << std::setprecision(2) << ms << " ms"
                  << std::endl;
    }

    return (uni==0) ? 9.9f : 1.0f - (float)inter/uni;
}

/* レイキャスト版と高速版を両方実行し、IoU の差分と速度比を出力する比較関数。
   Ctrl+Shift+E の直前や、別のデバッグキーから呼ぶ用。 */
inline void compareSilhouette2DImplementations(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    int step = 4)
{
    if (!liver || !g_boundaryDistMap.valid) {
        std::cerr << "[Sil2D-Compare] Invalid liver or boundary map" << std::endl;
        return;
    }
    /* レイキャスト版 */
    auto t0 = std::chrono::steady_clock::now();
    SilBVH dummy;
    float fval_ray = computeSilhouette2DObjective(liver, dummy, viewMat, projMat, step);
    auto t1 = std::chrono::steady_clock::now();
    double ms_ray = std::chrono::duration<double, std::milli>(t1 - t0).count();
    float iou_ray = 1.0f - fval_ray;

    /* 高速版 */
    int inter_f, union_f;
    double ms_fast;
    float fval_fast = computeSilhouette2DObjectiveFast(liver, viewMat, projMat, step,
                                                       &inter_f, &union_f, &ms_fast);
    float iou_fast = 1.0f - fval_fast;

    double speedup = (ms_fast > 0.0) ? (ms_ray / ms_fast) : 0.0;
    float  iouDiff = std::abs(iou_ray - iou_fast);

    std::cout << "[Sil2D-Compare] ======================================" << std::endl;
    std::cout << "[Sil2D-Compare] Raycast : IoU=" << std::fixed << std::setprecision(6)
              << iou_ray << "  time=" << std::setprecision(2) << ms_ray << " ms" << std::endl;
    std::cout << "[Sil2D-Compare] Fast    : IoU=" << std::fixed << std::setprecision(6)
              << iou_fast << "  time=" << std::setprecision(2) << ms_fast << " ms" << std::endl;
    std::cout << "[Sil2D-Compare] Speedup : " << std::setprecision(1) << speedup << "x"
              << "    |IoU_diff|=" << std::setprecision(6) << iouDiff << std::endl;
    std::cout << "[Sil2D-Compare] ======================================" << std::endl;
}

/* 輪郭ペアリングRMSE
   depth境界点（g_targetPoints）から3D meshへレイキャストして交点を求め
   depth境界点との距離RMSEを返す */
inline float computeContourPairRMSE(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat)
{
    if (!liver || g_targetPoints.empty()) return 9.9f;

    SilBVH bvh;
    bvh.init(liver);

    int imgW = gWindowWidth, imgH = gWindowHeight;
    glm::mat4 invVP = glm::inverse(projMat * viewMat);

    double sumSq = 0.0;
    int    count = 0;

    for (const auto& tgt : g_targetPoints) {
        /* depth境界点をスクリーン座標に投影 */
        glm::vec4 clip = projMat * viewMat * glm::vec4(tgt, 1.0f);
        if (std::abs(clip.w) < 1e-6f) continue;
        float ndcX =  clip.x / clip.w;
        float ndcY =  clip.y / clip.w;
        if (ndcX < -1.f || ndcX > 1.f || ndcY < -1.f || ndcY > 1.f) continue;

        /* NDC → レイ */
        glm::vec4 nH = invVP * glm::vec4(ndcX, ndcY, -1.f, 1.f);
        glm::vec4 fH = invVP * glm::vec4(ndcX, ndcY,  1.f, 1.f);
        glm::vec3 ro = glm::vec3(nH) / nH.w;
        glm::vec3 rd = glm::normalize(glm::vec3(fH) / fH.w - ro);

        /* 3D meshとの交点 */
        glm::vec3 hitPt;
        if (bvh.hitPoint(ro, rd, hitPt)) {
            float d = glm::length(hitPt - tgt);
            sumSq += (double)(d * d);
            count++;
        }
    }

    if (count == 0) return 9.9f;
    return std::sqrt((float)(sumSq / count));
}

/* デバッグJPG用: hitmap境界ピクセルを2Dリストで返す（BVH再ビルド） */
inline std::vector<std::pair<int,int>> extractSourceContour(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    int imgW, int imgH,
    float /*unused*/ = 0.3f,
    int   step       = 4)
{
    std::vector<std::pair<int,int>> pts;
    if (!liver || liver->mVertices.empty() || liver->mIndices.empty())
        return pts;

    SilBVH bvh;
    bvh.init(liver);

    int gw=(imgW+step-1)/step, gh=(imgH+step-1)/step;
    glm::mat4 invVP=glm::inverse(projMat*viewMat);
    std::vector<uint8_t> hitmap(gw*gh,0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,4)
#endif
    for (int gy=0;gy<gh;gy++){
        int py=gy*step+step/2;
        for (int gx=0;gx<gw;gx++){
            int px=gx*step+step/2;
            float ndcX= 2.f*px/imgW-1.f;
            float ndcY=-2.f*py/imgH+1.f;
            glm::vec4 nH=invVP*glm::vec4(ndcX,ndcY,-1.f,1.f);
            glm::vec4 fH=invVP*glm::vec4(ndcX,ndcY, 1.f,1.f);
            glm::vec3 ro=glm::vec3(nH)/nH.w;
            glm::vec3 rd=glm::normalize(glm::vec3(fH)/fH.w-ro);
            hitmap[gy*gw+gx]=bvh.hit(ro,rd)?1:0;
        }
    }

    const int dx4[]={1,-1,0,0}, dy4[]={0,0,1,-1};
    for (int gy=0;gy<gh;gy++)
        for (int gx=0;gx<gw;gx++){
            if(!hitmap[gy*gw+gx]) continue;
            bool border=false;
            for(int d=0;d<4&&!border;d++){
                int nx=gx+dx4[d],ny=gy+dy4[d];
                if(nx<0||nx>=gw||ny<0||ny>=gh){border=true;break;}
                if(!hitmap[ny*gw+nx]) border=true;
            }
            if(border) pts.emplace_back(gx*step+step/2, gy*step+step/2);
        }
    return pts;
}

/* ---- 円を描画（JPGオーバーレイ用） ---- */
inline void drawDot(std::vector<unsigned char>& img, int w, int h,
                    int cx, int cy, int r,
                    unsigned char R, unsigned char G, unsigned char B)
{
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++) {
            if (dx*dx + dy*dy > r*r) continue;
            int x = cx+dx, y = cy+dy;
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            int idx = (y*w + x)*3;
            img[idx]=R; img[idx+1]=G; img[idx+2]=B;
        }
}

inline std::vector<unsigned char> loadBaseImage(int& imgW, int& imgH)
{
    std::string basePath = std::string(DEPTH_OUTPUT_PATH) + "segmentation_overlay.jpg";
    int bw, bh, bch;
    unsigned char* base = stbi_load(basePath.c_str(), &bw, &bh, &bch, 3);
    imgW = bw > 0 ? bw : gWindowWidth;
    imgH = bh > 0 ? bh : gWindowHeight;
    std::vector<unsigned char> img(imgW * imgH * 3, 50);
    if (base) { memcpy(img.data(), base, imgW * imgH * 3); stbi_image_free(base); }
    return img;
}

inline void drawTargetBoundaryDirect(std::vector<unsigned char>& img,
                                     int imgW, int imgH,
                                     unsigned char R, unsigned char G, unsigned char B)
{
    if (!g_boundaryDistMap.valid) return;
    int mw = g_boundaryDistMap.width, mh = g_boundaryDistMap.height;
    for (int y = 0; y < mh; y++)
        for (int x = 0; x < mw; x++) {
            if (g_boundaryDistMap.data[y * mw + x] > 1.5f) continue;
            int px = x * imgW / mw;
            int py = y * imgH / mh;
            drawDot(img, imgW, imgH, px, py, 1, R, G, B);
        }
}

/* 3枚セット（target/source/composite）を outPrefix_*.jpg で出力 */
inline void saveSilhouetteDebugJPG(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    const std::string& outPrefix,
    float /*unused*/ = 0.3f)
{
    int imgW, imgH;
    auto srcPts = extractSourceContour(liver, viewMat, projMat,
                                       gWindowWidth, gWindowHeight);

    /* 1枚目: Target境界のみ（赤） */
    {
        auto img = loadBaseImage(imgW, imgH);
        drawTargetBoundaryDirect(img, imgW, imgH, 255, 0, 0);
        std::string path = outPrefix + "_target.jpg";
        stbi_write_jpg(path.c_str(), imgW, imgH, 3, img.data(), 90);
        std::cout << "[SilhouetteDebug] Target    : " << path << std::endl;
    }
    /* 2枚目: Source輪郭のみ（青） */
    {
        auto img = loadBaseImage(imgW, imgH);
        for (const auto& [px, py] : srcPts) {
            int sx = px * imgW / gWindowWidth;
            int sy = py * imgH / gWindowHeight;
            drawDot(img, imgW, imgH, sx, sy, 2, 0, 150, 255);
        }
        std::string path = outPrefix + "_source.jpg";
        stbi_write_jpg(path.c_str(), imgW, imgH, 3, img.data(), 90);
        std::cout << "[SilhouetteDebug] Source    : " << path
                  << "  pts=" << srcPts.size() << std::endl;
    }
    /* 3枚目: 合成（赤=target, 青=source） */
    {
        auto img = loadBaseImage(imgW, imgH);
        drawTargetBoundaryDirect(img, imgW, imgH, 255, 0, 0);
        for (const auto& [px, py] : srcPts) {
            int sx = px * imgW / gWindowWidth;
            int sy = py * imgH / gWindowHeight;
            drawDot(img, imgW, imgH, sx, sy, 2, 0, 150, 255);
        }
        std::string path = outPrefix + "_composite.jpg";
        stbi_write_jpg(path.c_str(), imgW, imgH, 3, img.data(), 90);
        std::cout << "[SilhouetteDebug] Composite : " << path << std::endl;
    }
}

/* ==========================================================================
   ラスタライズ版 hitmap の輪郭抽出 (Fast版 debug 用)
   computeSilhouette2DObjectiveFast と同じ手順で hitmap を作り、
   境界ピクセル (hitmap の 4近傍のどれかが 0 のピクセル) を画像座標で返す。
   ========================================================================== */
inline std::vector<std::pair<int,int>> extractSourceContourFast(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    int imgW, int imgH,
    int step = 4)
{
    std::vector<std::pair<int,int>> out;
    if (!liver) return out;

    const int gw = (imgW+step-1)/step;
    const int gh = (imgH+step-1)/step;
    const auto& V = liver->mVertices;
    const auto& I = liver->mIndices;
    const int nVerts = (int)(V.size()/3);
    const int nTris  = (int)(I.size()/3);
    if (nVerts == 0 || nTris == 0) return out;

    /* Step 1: 頂点→スクリーン */
    glm::mat4 MVP = projMat * viewMat;
    std::vector<glm::vec3> screen(nVerts);
    std::vector<uint8_t>   vinside(nVerts, 0);
    const float halfW = imgW * 0.5f, halfH = imgH * 0.5f;
    const float invStep = 1.0f / (float)step;
    for (int i = 0; i < nVerts; i++) {
        glm::vec4 c = MVP * glm::vec4(V[i*3], V[i*3+1], V[i*3+2], 1.0f);
        if (std::abs(c.w) < 1e-8f) { screen[i] = glm::vec3(0,0,2); continue; }
        float ndcX = c.x / c.w, ndcY = c.y / c.w, ndcZ = c.z / c.w;
        float px = (ndcX + 1.f) * halfW;
        float py = (1.f - ndcY) * halfH;
        screen[i] = glm::vec3(px * invStep, py * invStep, ndcZ);
        vinside[i] = (ndcX > -1.2f && ndcX < 1.2f &&
                      ndcY > -1.2f && ndcY < 1.2f &&
                      ndcZ > -1.0f && ndcZ <  1.0f) ? 1 : 0;
    }

    /* Step 2: 三角形ラスタライズ (単スレッド版、debug 用なので簡略) */
    std::vector<uint8_t> hitmap(gw*gh, 0);
    for (int ti = 0; ti < nTris; ti++) {
        int i0 = I[ti*3+0], i1 = I[ti*3+1], i2 = I[ti*3+2];
        if (!vinside[i0] && !vinside[i1] && !vinside[i2]) continue;
        const glm::vec3& s0 = screen[i0];
        const glm::vec3& s1 = screen[i1];
        const glm::vec3& s2 = screen[i2];
        float area2 = (s1.x - s0.x) * (s2.y - s0.y) -
                      (s2.x - s0.x) * (s1.y - s0.y);
        if (area2 <= 0.0f) continue;
        float minX = std::min({s0.x, s1.x, s2.x});
        float maxX = std::max({s0.x, s1.x, s2.x});
        float minY = std::min({s0.y, s1.y, s2.y});
        float maxY = std::max({s0.y, s1.y, s2.y});
        int x0 = std::max(0,  (int)std::floor(minX));
        int x1 = std::min(gw-1, (int)std::ceil(maxX));
        int y0 = std::max(0,  (int)std::floor(minY));
        int y1 = std::min(gh-1, (int)std::ceil(maxY));
        if (x0 > x1 || y0 > y1) continue;
        float invArea = 1.0f / area2;
        for (int y = y0; y <= y1; y++) {
            float py = (float)y + 0.5f;
            for (int x = x0; x <= x1; x++) {
                float px = (float)x + 0.5f;
                float w0 = ((s1.x - px) * (s2.y - py) - (s2.x - px) * (s1.y - py)) * invArea;
                float w1 = ((s2.x - px) * (s0.y - py) - (s0.x - px) * (s2.y - py)) * invArea;
                float w2 = 1.0f - w0 - w1;
                if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    hitmap[y*gw + x] = 1;
            }
        }
    }

    /* Step 3: 境界ピクセル抽出 (4近傍のどれかが 0) */
    for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
            if (!hitmap[gy*gw + gx]) continue;
            bool edge = false;
            if (gx > 0    && !hitmap[gy*gw + (gx-1)]) edge = true;
            if (gx < gw-1 && !hitmap[gy*gw + (gx+1)]) edge = true;
            if (gy > 0    && !hitmap[(gy-1)*gw + gx]) edge = true;
            if (gy < gh-1 && !hitmap[(gy+1)*gw + gx]) edge = true;
            if (edge) {
                int px = gx*step + step/2;
                int py = gy*step + step/2;
                out.emplace_back(px, py);
            }
        }
    }
    return out;
}

/* ==========================================================================
   Fast 版デバッグ JPG 出力。saveSilhouetteDebugJPG と同形式だが
   source 輪郭はラスタライズから取る。出力名は outPrefix_{target,source,composite}.jpg
   Raycast 版と Fast 版を別 prefix で呼べば、両者の画像差分を目視確認できる。
   ========================================================================== */
inline void saveSilhouetteFastDebugJPG(
    const mCutMesh*  liver,
    const glm::mat4& viewMat,
    const glm::mat4& projMat,
    const std::string& outPrefix,
    int step = 4)
{
    int imgW, imgH;
    auto srcPts = extractSourceContourFast(liver, viewMat, projMat,
                                           gWindowWidth, gWindowHeight, step);

    /* 1枚目: Target境界のみ（赤） */
    {
        auto img = loadBaseImage(imgW, imgH);
        drawTargetBoundaryDirect(img, imgW, imgH, 255, 0, 0);
        std::string path = outPrefix + "_target.jpg";
        stbi_write_jpg(path.c_str(), imgW, imgH, 3, img.data(), 90);
        std::cout << "[SilFastDebug] Target    : " << path << std::endl;
    }
    /* 2枚目: Source輪郭のみ（緑: Fast版は緑でRaycast版の青と区別） */
    {
        auto img = loadBaseImage(imgW, imgH);
        for (const auto& [px, py] : srcPts) {
            int sx = px * imgW / gWindowWidth;
            int sy = py * imgH / gWindowHeight;
            drawDot(img, imgW, imgH, sx, sy, 2, 0, 255, 80);
        }
        std::string path = outPrefix + "_source.jpg";
        stbi_write_jpg(path.c_str(), imgW, imgH, 3, img.data(), 90);
        std::cout << "[SilFastDebug] Source    : " << path
                  << "  pts=" << srcPts.size() << std::endl;
    }
    /* 3枚目: 合成（赤=target, 緑=source-fast） */
    {
        auto img = loadBaseImage(imgW, imgH);
        drawTargetBoundaryDirect(img, imgW, imgH, 255, 0, 0);
        for (const auto& [px, py] : srcPts) {
            int sx = px * imgW / gWindowWidth;
            int sy = py * imgH / gWindowHeight;
            drawDot(img, imgW, imgH, sx, sy, 2, 0, 255, 80);
        }
        std::string path = outPrefix + "_composite.jpg";
        stbi_write_jpg(path.c_str(), imgW, imgH, 3, img.data(), 90);
        std::cout << "[SilFastDebug] Composite : " << path << std::endl;
    }
}

/* ------------------------------------------------------------------ */
/* Internal: apply incremental SRT to all organs                       */
/* ------------------------------------------------------------------ */
inline void applyIncrementalSRT(
    const std::vector<mCutMesh*>& organs,
    float tx, float ty, float tz,
    float rx_deg, float ry_deg, float rz_deg,
    float scale)
{
    /* Build transform around centroid of liver (organs[0]) */
    glm::vec3 centroid(0.0f);
    int cnt = 0;
    if (!organs.empty() && organs[0]) {
        const auto& verts = organs[0]->mVertices;
        for (size_t i = 0; i + 2 < verts.size(); i += 3) {
            centroid += glm::vec3(verts[i], verts[i+1], verts[i+2]);
            cnt++;
        }
        if (cnt > 0) centroid /= (float)cnt;
    }

    const float deg2rad = (float)(M_PI / 180.0);
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(tx, ty, tz));
    glm::mat4 Rx = glm::rotate(glm::mat4(1.0f), rx_deg * deg2rad, glm::vec3(1,0,0));
    glm::mat4 Ry = glm::rotate(glm::mat4(1.0f), ry_deg * deg2rad, glm::vec3(0,1,0));
    glm::mat4 Rz = glm::rotate(glm::mat4(1.0f), rz_deg * deg2rad, glm::vec3(0,0,1));
    glm::mat4 S  = glm::scale(glm::mat4(1.0f), glm::vec3(scale));

    /* Compose: scale around centroid, then rotate, then translate */
    glm::mat4 toCentroid    = glm::translate(glm::mat4(1.0f), -centroid);
    glm::mat4 fromCentroid  = glm::translate(glm::mat4(1.0f),  centroid);
    glm::mat4 M = T * fromCentroid * Rz * Ry * Rx * S * toCentroid;

    for (auto* mesh : organs) {
        if (!mesh) continue;
        auto& v = mesh->mVertices;
        auto& n = mesh->mNormals;
        glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(M)));
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            glm::vec4 p(v[i], v[i+1], v[i+2], 1.0f);
            glm::vec4 tp = M * p;
            v[i]   = tp.x; v[i+1] = tp.y; v[i+2] = tp.z;
        }
        for (size_t i = 0; i + 2 < n.size(); i += 3) {
            glm::vec3 nm(n[i], n[i+1], n[i+2]);
            glm::vec3 tn = normalMat * nm;
            float len = glm::length(tn);
            if (len > 1e-8f) tn /= len;
            n[i]   = tn.x; n[i+1] = tn.y; n[i+2] = tn.z;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Params / Result                                                     */
/* ------------------------------------------------------------------ */
struct Params {
    float tx_range        = 0.5f;
    float ty_range        = 0.5f;
    float tz_range        = 0.5f;
    float rx_range        = 10.0f;
    float ry_range        = 10.0f;
    float rz_range        = 10.0f;
    float scale_lo        = 0.90f;
    float scale_hi        = 1.10f;
    float min_match_ratio = 0.30f;
    float penalty_value   = 9.9f;
    bool  use_boundary_weight = false;
    float boundary_width      = 12.0f;
    float boundary_boost      = 3.0f;
    bool  use_silhouette_2d   = false;
    bool  use_silhouette_2d_fast = false;  /* true: ラスタライズ版(高速,BVHなし) */
    float alpha_silhouette    = 1.0f;
    float alpha_3d            = 0.3f;
    int   silhouette_step     = 4;
    float silhouette_thresh   = 0.3f;
    int   maxgen          = 150;
    int   lambda          = 0;
    double sigma0         = 0.3;
    double tolfun         = 1e-6;
    double tolx           = 1e-8;
    bool  verbose         = true;
    int   log_every       = 30;
    bool  save_debug_jpg  = true;

    /* ------------------------------------------------------------
     * CMA-ES determinism control (reproducibility protocol)
     * ------------------------------------------------------------
     * When rng_seed != 0, srand() is called with this value
     * immediately after cmaes_init(), overriding c-cmaes's
     * internal srand(time(NULL)) for reproducible results.
     * When rng_seed == 0 (default), c-cmaes retains its original
     * non-deterministic time-based seeding.
     *
     * Note: cmaes.c / cmaes.h are NOT modified. Hansen's original
     * implementation is preserved bit-for-bit; seed control is
     * applied externally via this parameter.
     * ------------------------------------------------------------ */
    unsigned rng_seed     = 0;
};

struct Result {
    bool   improved       = false;
    float  rmse_before    = 0.0f;
    float  rmse_after     = 0.0f;
    float  initial_bw_rmse = 0.0f;
    float  delta_tx       = 0.0f;
    float  delta_ty       = 0.0f;
    float  delta_tz       = 0.0f;
    float  delta_rx_deg   = 0.0f;
    float  delta_ry_deg   = 0.0f;
    float  delta_rz_deg   = 0.0f;
    float  scale_applied  = 1.0f;
    int    generations    = 0;
    std::string stop_reason;
};

/* ------------------------------------------------------------------ */
/* Main entry point                                                     */
/* ------------------------------------------------------------------ */
inline Result run(
    const std::vector<mCutMesh*>& organs,
    mCutMesh*                     screenMesh,
    int gridWidth, int gridHeight, float depthScale,
    const Params& params = Params())
{
    Result result;

    /* 1. Snapshot current vertices (for rollback) */
    std::vector<std::vector<float>> snap_v(organs.size());
    std::vector<std::vector<float>> snap_n(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            snap_v[i] = organs[i]->mVertices;
            snap_n[i] = organs[i]->mNormals;
        }
    }

    /* 2. Measure initial compRmse */
    computeUnifiedMetrics();
    result.rmse_before = registrationHandle.compRmse;
    int   init_matched = registrationHandle.compCount;

    /* 2b. Silhouette debug: before */
    if (params.verbose && params.save_debug_jpg && !organs.empty() && organs[0]) {
        std::string tag;
        if (params.use_silhouette_2d)
            tag = params.use_silhouette_2d_fast ? "shiftE_fast" : "shiftE_raycast";
        else if (params.use_boundary_weight)
            tag = "keyE";
        else
            tag = "keyC";
        std::string prefix = std::string(DEPTH_OUTPUT_PATH) + "silhouette_before_" + tag;
        saveSilhouetteDebugJPG(organs[0], view, projection,
                               prefix, params.silhouette_thresh);
        /* Fast 版なら、同じ姿勢で ラスタライズ輪郭(緑) も別名で保存 */
        if (params.use_silhouette_2d && params.use_silhouette_2d_fast) {
            saveSilhouetteFastDebugJPG(organs[0], view, projection,
                                       prefix + "_rast", params.silhouette_step);
        }
    }

    /* 2b. BVHを事前ビルド（silhouette_2dモード用）
       best_rmse初期化より前に必要 */
    SilBVH g_sed; /* unused: BVH is rebuilt inside computeSilhouette2DObjective */
    if (params.use_silhouette_2d && !organs.empty() && organs[0]) {
        /* g_sed unused - BVH rebuilt per call */
    }

    /* best_rmse の初期値: モードに応じてfvalと同スケールで計算 */
    float best_rmse;
    if (params.use_silhouette_2d) {
        float sil0 = params.use_silhouette_2d_fast
                         ? computeSilhouette2DObjectiveFast(
                               organs.empty() ? nullptr : organs[0],
                               view, projection, params.silhouette_step)
                         : computeSilhouette2DObjective(
                               organs.empty() ? nullptr : organs[0],
                               g_sed, view, projection, params.silhouette_step);
        best_rmse = sil0;
        if (params.verbose)
            std::cout << "\n[CMA-ES] === Starting 2D Silhouette Refinement (IoU"
                      << (params.use_silhouette_2d_fast ? ", Fast" : ", Raycast")
                      << ") ===" << std::endl
                      << "[CMA-ES] Initial compRMSE=" << result.rmse_before
                      << "  Initial (1-IoU)=" << sil0
                      << "  Initial fval=" << best_rmse << std::endl;
    } else if (params.use_boundary_weight) {
        best_rmse = computeContourPairRMSE(
            organs.empty() ? nullptr : organs[0], view, projection);
        result.initial_bw_rmse = best_rmse;
        if (params.verbose)
            std::cout << "\n[CMA-ES] === Starting Contour-Pair Refinement (Key E) ===" << std::endl
                      << "[CMA-ES] Initial compRMSE=" << result.rmse_before
                      << "  Initial contourRMSE=" << best_rmse << std::endl;
    } else {
        best_rmse = result.rmse_before;
        if (params.verbose)
            std::cout << "\n[CMA-ES] === Starting Post-HemiAuto Refinement ===" << std::endl
                      << "[CMA-ES] Initial compRMSE: " << result.rmse_before << std::endl;
    }

    /* 3. Normalise parameter space to [-1, 1] for CMA-ES */
    /* x[0..2]: translation  (normalised by range)
       x[3..5]: rotation deg (normalised by range)
       x[6]:    scale        (normalised: 0 = scale_lo, 1 = scale_hi) */
    const int  DIM = 7;
    double lb[DIM], ub[DIM];
    /* translation */
    lb[0] = -1.0; ub[0] = 1.0;
    lb[1] = -1.0; ub[1] = 1.0;
    lb[2] = -1.0; ub[2] = 1.0;
    /* rotation */
    lb[3] = -1.0; ub[3] = 1.0;
    lb[4] = -1.0; ub[4] = 1.0;
    lb[5] = -1.0; ub[5] = 1.0;
    /* scale: map [scale_lo, scale_hi] → [-1,1] */
    lb[6] = -1.0; ub[6] = 1.0;

    double xstart[DIM] = {0,0,0,0,0,0,0}; /* start at current pose */

    cmaes_t* evo = cmaes_init(DIM, xstart, params.sigma0,
                              params.lambda, lb, ub);

    /* ------------------------------------------------------------
     * CMA-ES determinism hook (reproducibility protocol)
     * ------------------------------------------------------------
     * cmaes_init() internally calls srand(time(NULL)). If the
     * caller has specified a non-zero rng_seed, override that
     * seed here so that subsequent rand() / randn() calls within
     * this CMA-ES run are reproducible.
     * cmaes.c / cmaes.h are NOT modified; this is the minimal
     * external intervention required for determinism.
     * ------------------------------------------------------------ */
    if (params.rng_seed != 0) {
        srand(params.rng_seed);
        if (params.verbose) {
            std::cout << "[CMA-ES] Deterministic seed set: "
                      << params.rng_seed << std::endl;
        }
    }

    /* Best solution tracking */
    double best_x[DIM] = {0,0,0,0,0,0,0};

    /* Snapshot for best-so-far restore */
    std::vector<std::vector<float>> best_v = snap_v;
    std::vector<std::vector<float>> best_n = snap_n;

    /* CMA-ESループ中は詳細ログを抑制 */
    g_quietMetrics = true;

    /* targetCloud（depth mesh側）はループ中不変なのでキャッシュ */
    Reg3DCustom::NoOpen3DRegistration reg_cache;
    // depthScale param now passes a scene-scaled zThresh directly from
    // RegRatios::zThresh() (see RegistrationActions.h). Use it as-is,
    // floored at a small positive value.
    float zThresh_cache = std::max(0.001f, depthScale);
    auto targetCloud_cache = reg_cache.extractFrontFacePoints(
        *screenMesh, gridWidth, gridHeight, zThresh_cache);
    // Scale-invariant: was hardcoded max_dist=1.0 at sceneDiag≈7.36.
    // Reference ratio = 1.0 / 7.36 ≈ 0.1359
    constexpr float kRefSceneDiag_cache = 7.36f;
    const float max_dist_for_cache = g_sceneDiag * (1.0f / kRefSceneDiag_cache);
    const float max_dist_sq_cache  = max_dist_for_cache * max_dist_for_cache;

    const size_t tgt_size = targetCloud_cache->size();
    const std::vector<glm::vec3>& tgt_points_cache = targetCloud_cache->points;

    /* 対応点キャッシュ方式:
       初回 + 定期的にKDTreeで正確な対応点を計算してキャッシュ
       通常evalは前回の対応インデックスで距離だけ再計算 → O(tgt_size)
       更新頻度: UPDATE_INTERVAL世代ごとにKDTree再計算 */
    const int UPDATE_INTERVAL = 10;

    /* tgt[i] に対応する src頂点インデックス（-1=対応なし） */
    std::vector<int> corr_idx(tgt_size, -1);

    /* KDTreeで対応点を（再）計算 */
    auto updateCorrespondences = [&]() {
        const auto& verts = liverMesh3D->mVertices;
        std::vector<glm::vec3> srcPts;
        srcPts.reserve(verts.size() / 3);
        for (size_t i = 0; i + 2 < verts.size(); i += 3)
            srcPts.emplace_back(verts[i], verts[i+1], verts[i+2]);
        Reg3DCustom::NanoflannAdaptor adaptor(srcPts);
        auto tree = Reg3DCustom::buildKDTree(adaptor);
        for (size_t i = 0; i < tgt_size; i++) {
            size_t nnIdx; float dist_sq;
            if (Reg3DCustom::searchKNN1(*tree, tgt_points_cache[i], nnIdx, dist_sq)
                && dist_sq < max_dist_sq_cache)
                corr_idx[i] = (int)nnIdx;
            else
                corr_idx[i] = -1;
        }
    };

    /* 初回対応点計算 */
    updateCorrespondences();

    /* 高速RMSE: キャッシュ済み対応インデックスで距離だけ再計算 O(tgt_size) */
    auto fastComputeRMSE = [&]() -> float {
        const auto& verts = liverMesh3D->mVertices;
        float sumSq = 0.0f;
        int   count = 0;
        for (size_t i = 0; i < tgt_size; i++) {
            int j = corr_idx[i];
            if (j < 0) continue;
            size_t vi = (size_t)j * 3;
            if (vi + 2 >= verts.size()) continue;
            glm::vec3 srcPt(verts[vi], verts[vi+1], verts[vi+2]);
            glm::vec3 d = srcPt - tgt_points_cache[i];
            float sq = glm::dot(d, d);
            if (sq < max_dist_sq_cache) { sumSq += sq; count++; }
        }
        if (count == 0) return 9.9f;
        registrationHandle.compCount = count;
        return std::sqrt(sumSq / count);
    };

    /* 4. CMA-ES loop */
    const char* stop = nullptr;

    /* 時間計測 */
    double t_snapshot = 0, t_srt = 0, t_metrics = 0, t_fval = 0, t_best = 0;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };

    for (int gen = 0; gen < params.maxgen && !stop; gen++) {

        double** pop = cmaes_SamplePopulation(evo);
        std::vector<double> fval(evo->lambda);

        for (int k = 0; k < evo->lambda; k++) {
            float tx    = (float)(pop[k][0] * params.tx_range);
            float ty    = (float)(pop[k][1] * params.ty_range);
            float tz    = (float)(pop[k][2] * params.tz_range);
            float rx    = (float)(pop[k][3] * params.rx_range);
            float ry    = (float)(pop[k][4] * params.ry_range);
            float rz    = (float)(pop[k][5] * params.rz_range);
            float sc    = params.scale_lo
                       + (float)((pop[k][6]+1.0)*0.5)
                             * (params.scale_hi - params.scale_lo);

            auto t0 = now();
            /* liver（organs[0]）のみsnapshotに戻す
               他臓器はベスト確定時のみ更新するためここでは不要 */
            if (organs[0]) {
                organs[0]->mVertices = snap_v[0];
                organs[0]->mNormals  = snap_n[0];
            }
            auto t1 = now();
            /* liver（organs[0]）のみ変換（RMSEはliverのみで計算） */
            if (organs[0]) {
                std::vector<mCutMesh*> liver_only = { organs[0] };
                applyIncrementalSRT(liver_only, tx, ty, tz, rx, ry, rz, sc);
            }
            auto t2 = now();
            float rmse    = fastComputeRMSE();
            int   matched = registrationHandle.compCount;
            auto t3 = now();
            {
                int   min_ok  = (int)(init_matched * params.min_match_ratio);
                if (min_ok < 10) min_ok = 10;
                bool bad = (matched < min_ok || rmse == 0.0f);

                if (bad) {
                    fval[k] = (double)params.penalty_value;
                } else if (params.use_silhouette_2d) {
                    fval[k] = params.use_silhouette_2d_fast
                                  ? (double)computeSilhouette2DObjectiveFast(
                                        organs[0], view, projection,
                                        params.silhouette_step)
                                  : (double)computeSilhouette2DObjective(
                                        organs[0], g_sed, view, projection,
                                        params.silhouette_step);
                } else if (params.use_boundary_weight) {
                    fval[k] = (double)computeContourPairRMSE(
                        organs.empty() ? nullptr : organs[0],
                        view, projection);
                } else {
                    fval[k] = (double)rmse;
                }
                auto t4 = now();

                /* kループ内ではパラメータだけ保存（全臓器変換はループ後に1回） */
                if (fval[k] < best_rmse) {
                    best_rmse = (float)fval[k];
                    for (int d = 0; d < DIM; d++) best_x[d] = pop[k][d];
                }
                auto t5 = now();

                using ms = std::chrono::duration<double, std::milli>;
                t_snapshot += ms(t1-t0).count();
                t_srt      += ms(t2-t1).count();
                t_metrics  += ms(t3-t2).count();
                t_fval     += ms(t4-t3).count();
                t_best     += ms(t5-t4).count();
            }
        }

        cmaes_UpdateDistribution(evo, fval.data());

        /* UPDATE_INTERVAL世代ごとにbest_xからbest_vを再構築して対応点更新 */
        if (gen > 0 && gen % UPDATE_INTERVAL == 0) {
            float tx_b = (float)(best_x[0] * params.tx_range);
            float ty_b = (float)(best_x[1] * params.ty_range);
            float tz_b = (float)(best_x[2] * params.tz_range);
            float rx_b = (float)(best_x[3] * params.rx_range);
            float ry_b = (float)(best_x[4] * params.ry_range);
            float rz_b = (float)(best_x[5] * params.rz_range);
            float sc_b = params.scale_lo
                         + (float)((best_x[6]+1.0)*0.5)
                               * (params.scale_hi - params.scale_lo);
            for (size_t m = 0; m < organs.size(); m++)
                if (organs[m]) organs[m]->mVertices = snap_v[m];
            applyIncrementalSRT(organs, tx_b, ty_b, tz_b, rx_b, ry_b, rz_b, sc_b);
            updateCorrespondences();
            for (size_t m = 0; m < organs.size(); m++)
                if (organs[m]) organs[m]->mVertices = snap_v[m];
        }

        if (params.verbose && (gen % params.log_every == 0))
            std::cout << "[CMA-ES] Gen " << std::setw(4) << gen
                      << "  best=" << std::fixed << std::setprecision(5) << best_rmse
                      << "  sigma=" << std::setprecision(4) << evo->sigma
                      << std::endl;

        stop = cmaes_TestForTermination(evo, params.maxgen,
                                        params.tolfun, params.tolx);
    }

    result.generations  = evo->gen;
    result.stop_reason  = stop ? stop : "MaxGen";
    cmaes_exit(evo);

    /* ループ終了後にbest_xから全臓器のbest_vを1回だけ構築 */
    {
        float tx_b = (float)(best_x[0] * params.tx_range);
        float ty_b = (float)(best_x[1] * params.ty_range);
        float tz_b = (float)(best_x[2] * params.tz_range);
        float rx_b = (float)(best_x[3] * params.rx_range);
        float ry_b = (float)(best_x[4] * params.ry_range);
        float rz_b = (float)(best_x[5] * params.rz_range);
        float sc_b = params.scale_lo
                     + (float)((best_x[6]+1.0)*0.5)
                           * (params.scale_hi - params.scale_lo);
        for (size_t m = 0; m < organs.size(); m++) {
            if (organs[m]) {
                organs[m]->mVertices = snap_v[m];
                organs[m]->mNormals  = snap_n[m];
            }
        }
        applyIncrementalSRT(organs, tx_b, ty_b, tz_b, rx_b, ry_b, rz_b, sc_b);
        for (size_t m = 0; m < organs.size(); m++) {
            if (organs[m]) {
                best_v[m] = organs[m]->mVertices;
                best_n[m] = organs[m]->mNormals;
            }
        }
    }

    if (params.verbose) {
        int total_evals = result.generations * 10; /* lambda=10 */
        double t_total = t_snapshot + t_srt + t_metrics + t_fval + t_best;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "[CMA-ES] === Time Breakdown (total " << (int)t_total << " ms, "
                  << total_evals << " evals) ===" << std::endl;
        std::cout << "[CMA-ES]   snapshot restore  : " << (int)t_snapshot
                  << " ms (" << (int)(100*t_snapshot/t_total) << "%)" << std::endl;
        std::cout << "[CMA-ES]   applyIncrementalSRT: " << (int)t_srt
                  << " ms (" << (int)(100*t_srt/t_total) << "%)" << std::endl;
        std::cout << "[CMA-ES]   fastComputeRMSE   : " << (int)t_metrics
                  << " ms (" << (int)(100*t_metrics/t_total) << "%)" << std::endl;
        std::cout << "[CMA-ES]   fval(compRMSE/IoU): " << (int)t_fval
                  << " ms (" << (int)(100*t_fval/t_total) << "%)" << std::endl;
        std::cout << "[CMA-ES]   best snapshot      : " << (int)t_best
                  << " ms (" << (int)(100*t_best/t_total) << "%)" << std::endl;
        std::cout << "[CMA-ES]   per eval avg     : "
                  << std::setprecision(3) << (t_total/total_evals) << " ms" << std::endl;
    }

    for (size_t m = 0; m < organs.size(); m++) {
        if (organs[m]) {
            organs[m]->mVertices = snap_v[m];
            organs[m]->mNormals  = snap_n[m];
        }
    }
    computeUnifiedMetrics();
    g_quietMetrics = false;

    float initial_fval;
    if (params.use_silhouette_2d) {
        initial_fval = params.use_silhouette_2d_fast
                           ? computeSilhouette2DObjectiveFast(
                                 organs.empty() ? nullptr : organs[0],
                                 view, projection, params.silhouette_step)
                           : computeSilhouette2DObjective(
                                 organs.empty() ? nullptr : organs[0],
                                 g_sed, view, projection, params.silhouette_step);
    } else if (params.use_boundary_weight) {
        initial_fval = computeContourPairRMSE(
            organs.empty() ? nullptr : organs[0], view, projection);
    } else {
        initial_fval = result.rmse_before * 1.001f;
    }

    if (best_rmse < initial_fval) {
        for (size_t m = 0; m < organs.size(); m++) {
            if (organs[m]) {
                organs[m]->mVertices = best_v[m];
                organs[m]->mNormals  = best_n[m];
            }
        }
        computeUnifiedMetrics();
        float rmse_after_uniform = registrationHandle.compRmse;

        /* シルエットモード: fvalが改善していれば無条件採用
           (fval = 1-IoU のみ。alpha_3dは最適化には使わない)
           compRMSEが大幅悪化(50%超)の場合のみ安全ガードとしてリバート */
        bool accepted;
        if (params.use_silhouette_2d) {
            float iou_before = 1.0f - initial_fval;
            float iou_after  = 1.0f - (params.use_silhouette_2d_fast
                                          ? computeSilhouette2DObjectiveFast(
                                                organs.empty() ? nullptr : organs[0],
                                                view, projection, params.silhouette_step)
                                          : computeSilhouette2DObjective(
                                                organs.empty() ? nullptr : organs[0],
                                                g_sed, view, projection, params.silhouette_step));
            bool iou_ok  = (iou_after > iou_before * 1.05f);
            bool rmse_ok = (rmse_after_uniform < result.rmse_before * 1.2f);
            accepted = iou_ok && rmse_ok;
            if (params.verbose)
                std::cout << "[CMA-ES] best candidate:"
                          << "  compRMSE=" << rmse_after_uniform
                          << "  IoU=" << iou_before << "->" << iou_after
                          << "  accepted=" << (accepted ? "YES" : "NO") << std::endl;
        } else {
            if (params.use_boundary_weight) {
                float contour_after = computeContourPairRMSE(
                    organs.empty() ? nullptr : organs[0], view, projection);
                bool contour_ok = (contour_after < initial_fval);
                bool rmse_ok    = (rmse_after_uniform < result.rmse_before * 1.2f);
                accepted = contour_ok && rmse_ok;
                if (params.verbose)
                    std::cout << "[CMA-ES] best candidate:"
                              << "  compRMSE=" << rmse_after_uniform
                              << "  contourRMSE=" << initial_fval << "->" << contour_after
                              << "  accepted=" << (accepted ? "YES" : "NO") << std::endl;
            } else {
                accepted = (rmse_after_uniform < result.rmse_before);
            }
        }

        if (accepted) {
            result.improved  = true;
            result.rmse_after = rmse_after_uniform;

            result.delta_tx     = (float)(best_x[0] * params.tx_range);
            result.delta_ty     = (float)(best_x[1] * params.ty_range);
            result.delta_tz     = (float)(best_x[2] * params.tz_range);
            result.delta_rx_deg = (float)(best_x[3] * params.rx_range);
            result.delta_ry_deg = (float)(best_x[4] * params.ry_range);
            result.delta_rz_deg = (float)(best_x[5] * params.rz_range);
            result.scale_applied = params.scale_lo
                                   + (float)((best_x[6]+1.0)*0.5)
                                         * (params.scale_hi - params.scale_lo);

            if (params.verbose) {
                float pct = 100.0f*(result.rmse_before - result.rmse_after)/result.rmse_before;
                std::cout << "[CMA-ES] *** IMPROVED ***" << std::endl
                          << "[CMA-ES] compRMSE: " << result.rmse_before
                          << " -> " << result.rmse_after
                          << "  (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl
                          << "[CMA-ES] dT=(" << result.delta_tx << ", "
                          << result.delta_ty << ", " << result.delta_tz << ")" << std::endl
                          << "[CMA-ES] dR=(" << result.delta_rx_deg << ", "
                          << result.delta_ry_deg << ", " << result.delta_rz_deg << ") deg" << std::endl
                          << "[CMA-ES] scale=" << result.scale_applied << std::endl
                          << "[CMA-ES] Stop: " << result.stop_reason
                          << "  gens=" << result.generations << std::endl;
            }
        } else {
            result.improved   = false;
            result.rmse_after = result.rmse_before;
            for (size_t m = 0; m < organs.size(); m++) {
                if (organs[m]) {
                    organs[m]->mVertices = snap_v[m];
                    organs[m]->mNormals  = snap_n[m];
                }
            }
            computeUnifiedMetrics();
            if (params.verbose)
                std::cout << "[CMA-ES] No improvement (compRMSE "
                          << result.rmse_before << " -> " << rmse_after_uniform
                          << "). Reverted." << std::endl;
        }
    } else {
        result.improved   = false;
        result.rmse_after = result.rmse_before;
        for (size_t m = 0; m < organs.size(); m++) {
            if (organs[m]) {
                organs[m]->mVertices = snap_v[m];
                organs[m]->mNormals  = snap_n[m];
            }
        }
        computeUnifiedMetrics();
        if (params.verbose)
            std::cout << "[CMA-ES] No improvement ("
                      << result.rmse_before << " -> best_tried=" << best_rmse
                      << "). Reverted." << std::endl
                      << "[CMA-ES] Stop: " << result.stop_reason << std::endl;
    }

    /* 5b. Silhouette debug: after (現在のmesh状態で出力) */
    if (params.verbose && params.save_debug_jpg && !organs.empty() && organs[0]) {
        std::string tag;
        if (params.use_silhouette_2d)
            tag = params.use_silhouette_2d_fast ? "shiftE_fast" : "shiftE_raycast";
        else if (params.use_boundary_weight)
            tag = "keyE";
        else
            tag = "keyC";
        std::string prefix = std::string(DEPTH_OUTPUT_PATH) + "silhouette_after_" + tag;
        saveSilhouetteDebugJPG(organs[0], view, projection,
                               prefix, params.silhouette_thresh);
        if (params.use_silhouette_2d && params.use_silhouette_2d_fast) {
            saveSilhouetteFastDebugJPG(organs[0], view, projection,
                                       prefix + "_rast", params.silhouette_step);
        }
    }

    return result;
}

} /* namespace CmaesRefine */

#endif /* CMAES_UTILS_H */
