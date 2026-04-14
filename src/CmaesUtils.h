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

    /* Target maskをhitmapと同スケールで構築
       g_boundaryDistMap.data[i] < 9000 = マスク内側 */
    std::vector<uint8_t> targetMask(gw*gh, 0);
    for (int gy=0;gy<gh;gy++)
        for (int gx=0;gx<gw;gx++) {
            int ipx=gx*step+step/2, ipy=gy*step+step/2;
            int mx=std::clamp(ipx*mw/imgW,0,mw-1);
            int my=std::clamp(ipy*mh/imgH,0,mh-1);
            targetMask[gy*gw+gx] = (g_boundaryDistMap.data[my*mw+mx]<9000.f)?1:0;
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
        std::string tag = params.use_silhouette_2d ? "shiftE" : params.use_boundary_weight ? "keyE" : "keyC";
        saveSilhouetteDebugJPG(organs[0], view, projection,
                               std::string(DEPTH_OUTPUT_PATH) + "silhouette_before_" + tag,
                               params.silhouette_thresh);
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
        float sil0 = computeSilhouette2DObjective(
            organs.empty() ? nullptr : organs[0],
            g_sed, view, projection, params.silhouette_step);
        best_rmse = sil0;
        if (params.verbose)
            std::cout << "\n[CMA-ES] === Starting 2D Silhouette Refinement (IoU) ===" << std::endl
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

    /* Best solution tracking */
    double best_x[DIM] = {0,0,0,0,0,0,0};

    /* Snapshot for best-so-far restore */
    std::vector<std::vector<float>> best_v = snap_v;
    std::vector<std::vector<float>> best_n = snap_n;

    /* CMA-ESループ中は詳細ログを抑制 */
    g_quietMetrics = true;

    /* targetCloud（depth mesh側）はループ中不変なのでキャッシュ */
    Reg3DCustom::NoOpen3DRegistration reg_cache;
    float zThresh_cache = std::max(0.01f, depthScale * 0.05f);
    auto targetCloud_cache = reg_cache.extractFrontFacePoints(
        *screenMesh, gridWidth, gridHeight, zThresh_cache);
    const float max_dist_sq_cache = 1.0f;

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
                    fval[k] = (double)computeSilhouette2DObjective(
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
        initial_fval = computeSilhouette2DObjective(
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
            float iou_after  = 1.0f - computeSilhouette2DObjective(
                                  organs.empty() ? nullptr : organs[0],
                                  g_sed, view, projection, params.silhouette_step);
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
        std::string tag = params.use_silhouette_2d ? "shiftE" : params.use_boundary_weight ? "keyE" : "keyC";
        saveSilhouetteDebugJPG(organs[0], view, projection,
                               std::string(DEPTH_OUTPUT_PATH) + "silhouette_after_" + tag,
                               params.silhouette_thresh);
    }

    return result;
}

} /* namespace CmaesRefine */

#endif /* CMAES_UTILS_H */
