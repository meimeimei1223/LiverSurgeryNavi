#pragma once
// ============================================================================
// ChessboardCalibration.h — Zhang's camera calibration (no OpenCV)
// Dependencies: Eigen3, stb_image (linked elsewhere)
//
// Pipeline:
//   1. Harris corner detection + saddle filtering (multi-stage trials)
//   2. Sub-pixel refinement
//   3. Geiger-style grid growth (clean-room from ICRA 2012 concepts)
//   4. Zhang's calibration: homography per image → V → B → K
//   5. Radial distortion (k1, k2) least-squares fit
// ----------------------------------------------------------------------------
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// ============================================================================

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <climits>
#include <map>
#include <utility>
#include <Eigen/Dense>

extern "C" {
unsigned char* stbi_load(const char*, int*, int*, int*, int);
void stbi_image_free(void*);
}

namespace calib {

struct Pt2 { double x, y; };
struct Board { int cols=9, rows=6; float sqMM=22.0f; };

struct Result {
    double fx=0, fy=0, cx=0, cy=0, k1=0, k2=0;
    int width=0, height=0;
    double rmsError=1e9;
    int numImages=0, numPoints=0;
    bool valid=false;
    std::string message;
};

inline Result calibrateFromFolder(const std::string& folder, Board board = {});
inline bool   saveResult(const std::string& path, const Result& r);

namespace detail {

// ---- Image processing ------------------------------------------------------

inline std::vector<float> toGrayFloat(const uint8_t* d, int W, int H, int ch) {
    std::vector<float> g(W*H);
    for (int i = 0; i < W*H; i++) {
        if (ch==1) g[i]=d[i]; else if (ch==3) g[i]=0.299f*d[i*3]+0.587f*d[i*3+1]+0.114f*d[i*3+2];
        else if (ch==4) g[i]=0.299f*d[i*4]+0.587f*d[i*4+1]+0.114f*d[i*4+2];
    }
    return g;
}

inline std::vector<float> gaussBlur(const std::vector<float>& in, int W, int H) {
    static const float K[25]={1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};
    std::vector<float> out(W*H);
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            float s=0;
            for (int ky=-2;ky<=2;ky++) for (int kx=-2;kx<=2;kx++) {
                    s += in[std::clamp(y+ky,0,H-1)*W+std::clamp(x+kx,0,W-1)] * K[(ky+2)*5+(kx+2)];
                }
            out[y*W+x]=s/256.0f;
        }
    return out;
}

inline std::vector<float> multiBlur(const std::vector<float>& in, int W, int H, int n) {
    auto b=in; for(int i=0;i<n;i++) b=gaussBlur(b,W,H); return b;
}

inline void sobelGrad(const std::vector<float>& img, std::vector<float>& Ix, std::vector<float>& Iy, int W, int H) {
    Ix.assign(W*H,0); Iy.assign(W*H,0);
    for (int y=1;y<H-1;y++) for (int x=1;x<W-1;x++) {
            int i=y*W+x;
            Ix[i]=-img[i-W-1]+img[i-W+1]-2*img[i-1]+2*img[i+1]-img[i+W-1]+img[i+W+1];
            Iy[i]=-img[i-W-1]-2*img[i-W]-img[i-W+1]+img[i+W-1]+2*img[i+W]+img[i+W+1];
        }
}

// ---- Corner detection (multi-stage) ----------------------------------------

struct Corner { double x,y; float response; };

inline std::vector<float> harrisResponse(const std::vector<float>& Ix, const std::vector<float>& Iy,
                                         int W, int H, int winH=4, float kH=0.04f) {
    std::vector<float> R(W*H,0);
    for (int y=winH+1;y<H-winH-1;y++) for (int x=winH+1;x<W-winH-1;x++) {
            float sxx=0,sxy=0,syy=0;
            for (int dy=-winH;dy<=winH;dy++) for (int dx=-winH;dx<=winH;dx++) {
                    int j=(y+dy)*W+(x+dx); sxx+=Ix[j]*Ix[j]; sxy+=Ix[j]*Iy[j]; syy+=Iy[j]*Iy[j];
                }
            R[y*W+x]=sxx*syy-sxy*sxy-kH*(sxx+syy)*(sxx+syy);
        }
    return R;
}

inline std::vector<Corner> extractCorners(const std::vector<float>& R, int W, int H, float threshFrac, int nmsR) {
    float maxR=*std::max_element(R.begin(),R.end());
    if (maxR<=0) return {};
    float thresh=maxR*threshFrac;
    std::vector<Corner> out;
    for (int y=nmsR;y<H-nmsR;y++) for (int x=nmsR;x<W-nmsR;x++) {
            float r=R[y*W+x]; if (r<thresh) continue;
            bool mx=true;
            for (int dy=-nmsR;dy<=nmsR&&mx;dy++) for (int dx=-nmsR;dx<=nmsR&&mx;dx++) {
                    if(dx==0&&dy==0) continue; if(R[(y+dy)*W+(x+dx)]>=r) mx=false;
                }
            if (mx) out.push_back({(double)x,(double)y,r});
        }
    return out;
}

inline std::vector<Corner> filterSaddle(const std::vector<Corner>& c, const std::vector<float>& blur, int W, int H) {
    std::vector<Corner> out;
    for (auto& p:c) {
        int x=(int)std::round(p.x),y=(int)std::round(p.y);
        if(x<2||x>=W-2||y<2||y>=H-2) continue;
        int i=y*W+x;
        float Ixx=blur[i-1]-2*blur[i]+blur[i+1];
        float Iyy=blur[i-W]-2*blur[i]+blur[i+W];
        float Ixy=0.25f*(blur[i-W-1]-blur[i-W+1]-blur[i+W-1]+blur[i+W+1]);
        if (Ixx*Iyy-Ixy*Ixy<0) out.push_back(p);
    }
    return out;
}

inline void refineSubpixel(std::vector<Corner>& pts, const std::vector<float>& Ix, const std::vector<float>& Iy, int W, int H, int hw=4) {
    for (auto& c:pts) {
        int cx=(int)std::round(c.x),cy=(int)std::round(c.y);
        if(cx<hw||cx>=W-hw||cy<hw||cy>=H-hw) continue;
        double a11=0,a12=0,a22=0,b1=0,b2=0;
        for (int dy=-hw;dy<=hw;dy++) for (int dx=-hw;dx<=hw;dx++) {
                int j=(cy+dy)*W+(cx+dx); double gx=Ix[j],gy=Iy[j];
                a11+=gx*gx; a12+=gx*gy; a22+=gy*gy;
                b1+=gx*gx*(cx+dx)+gx*gy*(cy+dy); b2+=gx*gy*(cx+dx)+gy*gy*(cy+dy);
            }
        double det=a11*a22-a12*a12; if(std::abs(det)<1e-6) continue;
        double nx=(a22*b1-a12*b2)/det, ny=(a11*b2-a12*b1)/det;
        if(std::abs(nx-cx)<hw&&std::abs(ny-cy)<hw) { c.x=nx; c.y=ny; }
    }
}

// ---- Grid growth (Geiger 2012-style, clean-room implementation) ------------
//
// Starts from a seed triple (origin, +x neighbor, +y neighbor) where the
// neighbors are at roughly the chessboard square spacing and orthogonal in
// the image. Then iteratively grows the grid in 4 directions, predicting each
// new cell's position via a local symmetric basis (here - opposite-side
// neighbor) — which automatically tracks perspective curvature — falling back
// to the global average basis at the boundary. A new corner is accepted iff
// it lies within spacing*0.45 of the prediction. Conflicts (same corner
// claimed by multiple slots) are resolved by minimum prediction distance.
// Finally, only grids whose bounding box matches cols×rows or rows×cols
// (transposed) and which are completely filled are accepted.

inline std::vector<Pt2> selectAndOrderGrid(std::vector<Corner>& corners, int cols, int rows, double spacingHint) {
    using Vec2 = Eigen::Vector2d;
    using Cell = std::pair<int,int>;

    int N = (int)corners.size();
    int expected = cols * rows;
    if (N < expected) return {};

    // Sort by Harris response (highest first) — used for seed candidate priority.
    std::sort(corners.begin(), corners.end(),
              [](const Corner& a, const Corner& b){ return a.response > b.response; });

    std::vector<Vec2> P(N);
    for (int i = 0; i < N; i++) P[i] = Vec2(corners[i].x, corners[i].y);

    // --- Estimate grid spacing ---------------------------------------------
    // We need the real grid spacing despite many spurious corners that may
    // contaminate the nearest-neighbor distribution. Median is fragile when
    // spurious-spurious or real-spurious pairs cluster below the real spacing.
    // Instead we histogram all NN distances in [hint*0.5, hint*1.6] and pick
    // the densest bin (mode), which corresponds to the dominant grid spacing.
    double spacing = spacingHint;
    {
        std::vector<double> nn;
        nn.reserve(N);
        for (int i = 0; i < N; i++) {
            double minD = 1e18;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double d = (P[i] - P[j]).norm();
                if (d < minD) minD = d;
            }
            if (minD > spacingHint*0.5 && minD < spacingHint*1.6) nn.push_back(minD);
        }
        if (!nn.empty()) {
            // Histogram with bin width hint/8. Pick the densest bin's mean.
            const double binW = spacingHint / 8.0;
            const double lo = spacingHint * 0.5;
            const int nBins = (int)std::ceil((spacingHint*1.6 - lo) / binW) + 1;
            std::vector<int> counts(nBins, 0);
            std::vector<double> sums(nBins, 0.0);
            for (double d : nn) {
                int b = (int)((d - lo) / binW);
                if (b < 0) b = 0;
                if (b >= nBins) b = nBins - 1;
                counts[b]++;
                sums[b] += d;
            }
            int bestBin = 0;
            for (int b = 1; b < nBins; b++)
                if (counts[b] > counts[bestBin]) bestBin = b;
            if (counts[bestBin] > 0) {
                // Use the bin's mean for sub-bin precision; smear over neighbors
                // (bestBin±1) for stability if they exist.
                int totalC = counts[bestBin];
                double totalS = sums[bestBin];
                if (bestBin > 0)        { totalC += counts[bestBin-1]; totalS += sums[bestBin-1]; }
                if (bestBin < nBins-1)  { totalC += counts[bestBin+1]; totalS += sums[bestBin+1]; }
                spacing = totalS / totalC;
            } else {
                std::sort(nn.begin(), nn.end());
                spacing = nn[nn.size()/2];
            }
        }
    }
    if (spacing < 5.0) return {};

    // --- Grow a grid from a seed -------------------------------------------
    // findNearest scores candidates by predDist minus a response-strength
    // bonus, so that high-response (real chessboard) corners are preferred
    // over spurious low-response corners that happen to fall slightly closer
    // to the predicted location. The hard distance gate is still maxDist.
    auto findNearest = [&](const Vec2& tgt, double maxDist, const std::vector<char>& used) -> int {
        int best = -1;
        double bestScore = 1e18;
        for (int i = 0; i < N; i++) {
            if (used[i]) continue;
            double d = (P[i] - tgt).norm();
            if (d > maxDist) continue;
            double score = d - 0.15 * spacing * (double)corners[i].response;
            if (score < bestScore) { bestScore = score; best = i; }
        }
        return best;
    };

    auto tryGrow = [&](int idxO, int idxX, int idxY) -> std::map<Cell,int> {
        std::map<Cell,int> grid;
        std::vector<char> used(N, 0);
        grid[{0,0}] = idxO; used[idxO] = 1;
        grid[{1,0}] = idxX; used[idxX] = 1;
        grid[{0,1}] = idxY; used[idxY] = 1;

        const int maxIters = (cols + rows) * 4;
        for (int iter = 0; iter < maxIters; iter++) {
            // Global average X/Y basis from currently filled adjacent pairs
            Vec2 avgBX = Vec2::Zero(), avgBY = Vec2::Zero();
            int cntX = 0, cntY = 0;
            for (auto& kv : grid) {
                int gx = kv.first.first, gy = kv.first.second;
                auto itx = grid.find({gx+1, gy});
                if (itx != grid.end()) { avgBX += P[itx->second] - P[kv.second]; cntX++; }
                auto ity = grid.find({gx, gy+1});
                if (ity != grid.end()) { avgBY += P[ity->second] - P[kv.second]; cntY++; }
            }
            if (cntX > 0) avgBX /= (double)cntX;
            if (cntY > 0) avgBY /= (double)cntY;

            // Collect candidate additions
            struct Cand { int gx, gy, ci; double dist; };
            std::vector<Cand> cands;
            const int dxs[] = {1, -1, 0, 0};
            const int dys[] = {0, 0, 1, -1};

            for (auto& kv : grid) {
                int gx = kv.first.first, gy = kv.first.second;
                Vec2 here = P[kv.second];
                for (int d = 0; d < 4; d++) {
                    int nx = gx + dxs[d], ny = gy + dys[d];
                    if (grid.count({nx,ny})) continue;

                    Vec2 basis;
                    auto opp = grid.find({gx - dxs[d], gy - dys[d]});
                    if (opp != grid.end()) {
                        // Local symmetric basis — perspective-aware
                        basis = here - P[opp->second];
                    } else {
                        // Fall back to global average basis
                        if      (dxs[d] ==  1) basis =  avgBX;
                        else if (dxs[d] == -1) basis = -avgBX;
                        else if (dys[d] ==  1) basis =  avgBY;
                        else                    basis = -avgBY;
                    }
                    if (basis.norm() < spacing*0.3) continue;  // degenerate

                    Vec2 pred = here + basis;
                    int found = findNearest(pred, spacing*0.45, used);
                    if (found < 0) continue;
                    double dist = (P[found] - pred).norm();
                    cands.push_back({nx, ny, found, dist});
                }
            }

            if (cands.empty()) break;

            // Apply best (smallest prediction error) non-conflicting additions
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b){ return a.dist < b.dist; });
            int added = 0;
            for (auto& c : cands) {
                if (grid.count({c.gx, c.gy})) continue;
                if (used[c.ci]) continue;
                grid[{c.gx, c.gy}] = c.ci;
                used[c.ci] = 1;
                added++;
            }
            if (added == 0) break;
        }
        return grid;
    };

    // Convert grid to row-major Pt2 list. The grown grid may be larger than
    // the chessboard (spurious corners may have extended it) or smaller in one
    // dimension. We slide a cols×rows window across the grid bounding box and
    // accept the first window that is completely filled. We also try
    // rows×cols (transposed) to handle seed orientations that swap axes.
    auto gridToPts = [&](const std::map<Cell,int>& grid) -> std::vector<Pt2> {
        if ((int)grid.size() < expected) return {};
        int minX = INT_MAX, maxX = INT_MIN, minY = INT_MAX, maxY = INT_MIN;
        for (auto& kv : grid) {
            minX = std::min(minX, kv.first.first);
            maxX = std::max(maxX, kv.first.first);
            minY = std::min(minY, kv.first.second);
            maxY = std::max(maxY, kv.first.second);
        }

        auto tryDims = [&](int cw, int ch, bool transpose) -> std::vector<Pt2> {
            if (maxX - minX + 1 < cw) return {};
            if (maxY - minY + 1 < ch) return {};
            for (int y0 = minY; y0 + ch - 1 <= maxY; y0++) {
                for (int x0 = minX; x0 + cw - 1 <= maxX; x0++) {
                    bool full = true;
                    for (int dy = 0; dy < ch && full; dy++)
                        for (int dx = 0; dx < cw && full; dx++)
                            if (!grid.count({x0 + dx, y0 + dy})) full = false;
                    if (!full) continue;

                    std::vector<Pt2> out;
                    out.reserve(cw * ch);
                    if (!transpose) {
                        for (int dy = 0; dy < ch; dy++)
                            for (int dx = 0; dx < cw; dx++) {
                                int idx = grid.at({x0 + dx, y0 + dy});
                                out.push_back({P[idx].x(), P[idx].y()});
                            }
                    } else {
                        for (int dx = 0; dx < cw; dx++)
                            for (int dy = 0; dy < ch; dy++) {
                                int idx = grid.at({x0 + dx, y0 + dy});
                                out.push_back({P[idx].x(), P[idx].y()});
                            }
                    }
                    return out;
                }
            }
            return {};
        };

        auto pts = tryDims(cols, rows, false);
        if (!pts.empty()) return pts;
        return tryDims(rows, cols, true);
    };

    // --- Seed search: top-response corners as origin candidates -----------
    // For each origin, try the top-K pairs ranked by orthogonality × magnitude
    // similarity. This bounds work while ensuring every origin gets a fair
    // chance even when earlier origins have many candidate neighbors.
    const int seedLimit  = std::min(N, 80);
    const int pairsPerOrigin = 3;

    for (int i = 0; i < seedLimit; i++) {
        std::vector<int> nbrs;
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            double d = (P[j] - P[i]).norm();
            if (d > spacing*0.65 && d < spacing*1.45) nbrs.push_back(j);
        }
        if ((int)nbrs.size() < 2) continue;

        // Score every valid pair, keep top-K
        struct Pair { int a, b; double q; };
        std::vector<Pair> pairs;
        for (int a = 0; a < (int)nbrs.size(); a++) {
            Vec2 v1 = P[nbrs[a]] - P[i];
            double n1 = v1.norm();
            for (int b = a+1; b < (int)nbrs.size(); b++) {
                Vec2 v2 = P[nbrs[b]] - P[i];
                double n2 = v2.norm();
                double cosA = std::abs(v1.dot(v2) / (n1*n2));
                if (cosA > 0.25) continue;                            // ~75-105°
                double magR = std::min(n1,n2) / std::max(n1,n2);
                if (magR < 0.65) continue;
                pairs.push_back({nbrs[a], nbrs[b], (1.0 - cosA) * magR});
            }
        }
        if (pairs.empty()) continue;

        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b){ return a.q > b.q; });
        int trials = std::min((int)pairs.size(), pairsPerOrigin);

        for (int t = 0; t < trials; t++) {
            Vec2 v1 = P[pairs[t].a] - P[i];
            Vec2 v2 = P[pairs[t].b] - P[i];
            // Right-handed (image y-down): cross > 0 means v2 is the +y direction
            double cross = v1.x()*v2.y() - v1.y()*v2.x();
            int idxX, idxY;
            if (cross > 0) { idxX = pairs[t].a; idxY = pairs[t].b; }
            else            { idxX = pairs[t].b; idxY = pairs[t].a; }

            auto grid = tryGrow(i, idxX, idxY);
            auto pts  = gridToPts(grid);
            if (!pts.empty()) return pts;
        }
    }
    return {};
}

// ---- Detection pipeline ----------------------------------------------------

inline std::vector<Pt2> detectCorners(const uint8_t* data, int W, int H, int ch, Board board) {
    int expected=board.cols*board.rows;
    auto gray=toGrayFloat(data,W,H,ch);
    auto blur=multiBlur(gray,W,H,2);
    std::vector<float> Ix,Iy; sobelGrad(blur,Ix,Iy,W,H);
    auto R=harrisResponse(Ix,Iy,W,H,4,0.04f);
    int sp=W/(board.cols+4);  // estimated grid spacing (px)

    struct Trial { int nmsR; float thresh; };
    std::vector<Trial> trials={
                                 {sp/3,0.002f},{sp/4,0.002f},{sp/5,0.002f},
                                 {sp/3,0.001f},{sp/4,0.001f},{sp/5,0.001f},{sp/6,0.001f},
                                 {sp/4,0.0005f},{sp/5,0.0005f},{sp/6,0.0005f},{sp/8,0.0005f},
                                 {std::max(6,sp/10),0.0003f},{std::max(4,sp/12),0.0002f},
                                 {std::max(3,sp/15),0.0001f},
                                 };

    for (auto& t:trials) {
        int nmsR=std::max(3,t.nmsR);
        auto corners=extractCorners(R,W,H,t.thresh,nmsR);
        if ((int)corners.size()<expected) continue;

        // If way too many, try saddle filter
        if ((int)corners.size()>expected*4) {
            auto f=filterSaddle(corners,blur,W,H);
            if ((int)f.size()>=expected) corners=f;
        }

        std::cout << "[Calib]   nmsR="<<nmsR<<" thresh="<<t.thresh<<" -> "<<corners.size()<<" corners";

        refineSubpixel(corners,Ix,Iy,W,H);
        auto ordered=selectAndOrderGrid(corners,board.cols,board.rows,(double)sp);
        if (!ordered.empty()) { std::cout<<" -> GRID OK"<<std::endl; return ordered; }
        std::cout<<" -> grid fail"<<std::endl;
    }
    std::cerr<<"[Calib] Detection failed (need "<<expected<<")"<<std::endl;
    return {};
}

// ---- Homography (normalized DLT) ------------------------------------------

inline Eigen::Matrix3d computeHomography(const std::vector<Pt2>& src, const std::vector<Pt2>& dst) {
    int n=(int)src.size();
    auto normT=[](const std::vector<Pt2>& pts){
        double mx=0,my=0; for(auto&p:pts){mx+=p.x;my+=p.y;} mx/=pts.size();my/=pts.size();
        double s=0; for(auto&p:pts) s+=std::sqrt((p.x-mx)*(p.x-mx)+(p.y-my)*(p.y-my));
        s=std::sqrt(2.0)*pts.size()/std::max(s,1e-12);
        Eigen::Matrix3d T; T<<s,0,-s*mx,0,s,-s*my,0,0,1;
        std::vector<Pt2> out(pts.size());
        for(int i=0;i<(int)pts.size();i++) out[i]={s*(pts[i].x-mx),s*(pts[i].y-my)};
        return std::make_pair(T,out);
    };
    auto[T1,sn]=normT(src); auto[T2,dn]=normT(dst);
    Eigen::MatrixXd A(2*n,9);
    for(int i=0;i<n;i++){
        double X=sn[i].x,Y=sn[i].y,u=dn[i].x,v=dn[i].y;
        A.row(2*i)<<X,Y,1,0,0,0,-u*X,-u*Y,-u;
        A.row(2*i+1)<<0,0,0,X,Y,1,-v*X,-v*Y,-v;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullV);
    Eigen::VectorXd h=svd.matrixV().col(8);
    Eigen::Matrix3d H; H<<h(0),h(1),h(2),h(3),h(4),h(5),h(6),h(7),h(8);
    H=T2.inverse()*H*T1; H/=H(2,2); return H;
}

// ---- Zhang's method --------------------------------------------------------

inline Eigen::VectorXd vij(const Eigen::Matrix3d& H,int i,int j){
    Eigen::VectorXd v(6);
    v<<H(0,i)*H(0,j),H(0,i)*H(1,j)+H(1,i)*H(0,j),H(1,i)*H(1,j),
        H(2,i)*H(0,j)+H(0,i)*H(2,j),H(2,i)*H(1,j)+H(1,i)*H(2,j),H(2,i)*H(2,j);
    return v;
}

inline Result zhangCalibrate(const std::vector<Eigen::Matrix3d>& Hs,
                             const std::vector<std::vector<Pt2>>& allImg, const std::vector<Pt2>& objPts, int W, int H)
{
    Result res; res.width=W; res.height=H;
    int nImg=(int)Hs.size();
    if(nImg<3){res.message="Need >= 3 images";return res;}

    Eigen::MatrixXd V(2*nImg,6);
    for(int k=0;k<nImg;k++){
        V.row(2*k)=vij(Hs[k],0,1).transpose();
        V.row(2*k+1)=(vij(Hs[k],0,0)-vij(Hs[k],1,1)).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svdV(V,Eigen::ComputeFullV);
    Eigen::VectorXd b=svdV.matrixV().col(5);
    double B11=b(0),B12=b(1),B22=b(2),B13=b(3),B23=b(4),B33=b(5);
    double denom=B11*B22-B12*B12;

    auto tryExtract=[&](double B11,double B12,double B22,double B13,double B23,double B33,double denom)->bool{
        if(std::abs(denom)<1e-15||std::abs(B11)<1e-15) return false;
        double v0=(B12*B13-B11*B23)/denom;
        double lam=B33-(B13*B13+v0*(B12*B13-B11*B23))/B11;
        if(lam/B11<0||lam*B11/denom<0) return false;
        res.fx=std::sqrt(std::abs(lam/B11)); res.fy=std::sqrt(std::abs(lam*B11/denom));
        double gamma=-B12*res.fx*res.fx*res.fy/lam;
        res.cx=gamma*v0/res.fy-B13*res.fx*res.fx/lam; res.cy=v0;
        return(res.fx>10&&res.fy>10&&res.cx>0&&res.cx<W&&res.cy>0&&res.cy<H);
    };

    if(!tryExtract(B11,B12,B22,B13,B23,B33,denom)){
        b=-b; B11=b(0);B12=b(1);B22=b(2);B13=b(3);B23=b(4);B33=b(5);
        denom=B11*B22-B12*B12;
        if(!tryExtract(B11,B12,B22,B13,B23,B33,denom)){res.message="Cannot extract intrinsics";return res;}
    }

    Eigen::Matrix3d K; K<<res.fx,0,res.cx,0,res.fy,res.cy,0,0,1;
    Eigen::Matrix3d Kinv=K.inverse();
    // Per-image extrinsics + per-point residuals. For the radial distortion
    // fit we model:
    //   u_obs - u_pinhole = (u_pinhole - cx) * (k1*r^2 + k2*r^4)
    //   v_obs - v_pinhole = (v_pinhole - cy) * (k1*r^2 + k2*r^4)
    // and solve for (k1, k2) by least squares.
    std::vector<Eigen::Matrix3d> Rs(nImg);
    std::vector<Eigen::Vector3d> ts(nImg);
    std::vector<double> Dr2,Dr4,Du,Dv;

    for(int k=0;k<nImg;k++){
        Eigen::Vector3d h1=Hs[k].col(0),h2=Hs[k].col(1),h3=Hs[k].col(2);
        double lam=1.0/(Kinv*h1).norm();
        Eigen::Matrix3d R; R.col(0)=lam*Kinv*h1; R.col(1)=lam*Kinv*h2; R.col(2)=R.col(0).cross(R.col(1));
        Eigen::Vector3d t=lam*Kinv*h3;
        Eigen::JacobiSVD<Eigen::Matrix3d> svdR(R,Eigen::ComputeFullU|Eigen::ComputeFullV);
        R=svdR.matrixU()*svdR.matrixV().transpose();
        if(R.determinant()<0) R.col(2)=-R.col(2);
        Rs[k]=R; ts[k]=t;

        for(int i=0;i<(int)objPts.size();i++){
            Eigen::Vector3d M(objPts[i].x,objPts[i].y,0), Mc=R*M+t;
            if(std::abs(Mc(2))<1e-8) continue;
            double xn=Mc(0)/Mc(2), yn=Mc(1)/Mc(2);
            double up=res.fx*xn+res.cx, vp=res.fy*yn+res.cy;
            double uo=allImg[k][i].x, vo=allImg[k][i].y;
            double r2=xn*xn+yn*yn;
            Dr2.push_back((up-res.cx)*r2);    Dr4.push_back((up-res.cx)*r2*r2); Du.push_back(uo-up);
            Dr2.push_back((vp-res.cy)*r2);    Dr4.push_back((vp-res.cy)*r2*r2); Dv.push_back(vo-vp);
        }
    }

    if(Du.size()==Dv.size()&&Du.size()>4){
        int m=(int)Du.size();
        Eigen::MatrixXd A(2*m,2); Eigen::VectorXd bb(2*m);
        for(int i=0;i<m;i++){
            A(2*i,0)=Dr2[2*i];   A(2*i,1)=Dr4[2*i];   bb(2*i)=Du[i];
            A(2*i+1,0)=Dr2[2*i+1];A(2*i+1,1)=Dr4[2*i+1];bb(2*i+1)=Dv[i];
        }
        Eigen::Vector2d kk=(A.transpose()*A).ldlt().solve(A.transpose()*bb);
        res.k1=kk(0); res.k2=kk(1);
    }

    // Per-point weight (1=inlier, 0=outlier). Populated by IRLS pass inside LM.
    Eigen::VectorXd ptW;
    // ------------------------------------------------------------------
    // Nonlinear refinement (Levenberg-Marquardt bundle adjustment).
    // Jointly optimizes fx, fy, cx, cy, k1, k2 and per-image (R_k, t_k)
    // (R encoded as Rodrigues vector). Uses numerical Jacobian — for
    // ~120 params this completes in well under a second.
    //
    // Without this step the linear Zhang RMS is typically 3-5 px on
    // 1080p images; with it the RMS drops to <1 px and the intrinsics
    // converge to the global minimum.
    // ------------------------------------------------------------------
    {
        const int M = (int)objPts.size();
        const int nParams = 6 + 6*nImg;
        const int nRes = 2 * nImg * M;

        auto rodrigues = [](const Eigen::Vector3d& r) -> Eigen::Matrix3d {
            Eigen::Matrix3d R;
            double th = r.norm();
            if (th < 1e-12) { R.setIdentity(); return R; }
            Eigen::Vector3d kv = r / th;
            Eigen::Matrix3d Kx;
            Kx << 0, -kv(2), kv(1),
                kv(2), 0, -kv(0),
                -kv(1), kv(0), 0;
            R = Eigen::Matrix3d::Identity() + std::sin(th)*Kx + (1-std::cos(th))*Kx*Kx;
            return R;
        };
        auto invRodrigues = [](const Eigen::Matrix3d& R) -> Eigen::Vector3d {
            double tr = R.trace();
            double c = std::max(-1.0, std::min(1.0, (tr - 1.0)*0.5));
            double th = std::acos(c);
            if (th < 1e-12) return Eigen::Vector3d::Zero();
            double s = std::sin(th);
            if (std::abs(s) < 1e-12) {
                // Near 180°: extract axis from diagonal
                Eigen::Vector3d k;
                k(0) = std::sqrt(std::max(0.0, (R(0,0)+1.0)*0.5));
                k(1) = std::sqrt(std::max(0.0, (R(1,1)+1.0)*0.5));
                k(2) = std::sqrt(std::max(0.0, (R(2,2)+1.0)*0.5));
                if (R(0,1) < 0) k(1) = -k(1);
                if (R(0,2) < 0) k(2) = -k(2);
                Eigen::Vector3d out = th * k;
                return out;
            }
            Eigen::Vector3d r(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
            Eigen::Vector3d out = r * (th / (2.0*s));
            return out;
        };

        // Pack initial parameters
        Eigen::VectorXd p(nParams);
        p(0)=res.fx; p(1)=res.fy; p(2)=res.cx; p(3)=res.cy; p(4)=res.k1; p(5)=res.k2;
        for (int k=0; k<nImg; k++) {
            Eigen::Vector3d rod = invRodrigues(Rs[k]);
            p(6+6*k+0)=rod(0); p(6+6*k+1)=rod(1); p(6+6*k+2)=rod(2);
            p(6+6*k+3)=ts[k](0); p(6+6*k+4)=ts[k](1); p(6+6*k+5)=ts[k](2);
        }

        auto residuals = [&](const Eigen::VectorXd& pp, Eigen::VectorXd& r) {
            double fx=pp(0), fy=pp(1), cx=pp(2), cy=pp(3), k1=pp(4), k2=pp(5);
            int idx = 0;
            for (int k=0; k<nImg; k++) {
                Eigen::Vector3d rod(pp(6+6*k+0), pp(6+6*k+1), pp(6+6*k+2));
                Eigen::Vector3d t  (pp(6+6*k+3), pp(6+6*k+4), pp(6+6*k+5));
                Eigen::Matrix3d R = rodrigues(rod);
                for (int i=0; i<M; i++) {
                    Eigen::Vector3d Mw(objPts[i].x, objPts[i].y, 0.0);
                    Eigen::Vector3d Mc = R*Mw + t;
                    double xn = Mc(0)/Mc(2), yn = Mc(1)/Mc(2);
                    double r2 = xn*xn + yn*yn;
                    double d = 1.0 + k1*r2 + k2*r2*r2;
                    double u = fx*xn*d + cx, v = fy*yn*d + cy;
                    r(idx++) = allImg[k][i].x - u;
                    r(idx++) = allImg[k][i].y - v;
                }
            }
        };

        Eigen::VectorXd r(nRes); residuals(p, r);
        double err = r.squaredNorm();
        double lambda = 1e-3;
        const int maxIter = 50;

        // Per-point weight (one weight per (image, point) pair, applied
        // identically to its u and v residuals). All ones initially; populated
        // by the IRLS pass below to suppress gross outliers.
        ptW = Eigen::VectorXd::Ones(nImg * M);
        auto applyWeights = [&](Eigen::VectorXd& rr) {
            for (int k=0; k<nImg; k++) for (int i=0; i<M; i++) {
                    double w = std::sqrt(ptW(k*M + i));
                    rr(2*(k*M+i))     *= w;
                    rr(2*(k*M+i) + 1) *= w;
                }
        };
        applyWeights(r); err = r.squaredNorm();

        auto runLM = [&]() {
            err = r.squaredNorm();
            lambda = 1e-3;
            for (int iter = 0; iter < maxIter; iter++) {
                Eigen::MatrixXd J(nRes, nParams);
                Eigen::VectorXd rPert(nRes);
                for (int j = 0; j < nParams; j++) {
                    double eps = 1e-7 * std::max(1.0, std::abs(p(j)));
                    Eigen::VectorXd pp = p; pp(j) += eps;
                    residuals(pp, rPert);
                    applyWeights(rPert);
                    J.col(j) = (rPert - r) / eps;
                }
                Eigen::MatrixXd JtJ = J.transpose()*J;
                Eigen::VectorXd Jtr = J.transpose()*r;
                Eigen::MatrixXd Hm = JtJ;
                for (int d = 0; d < nParams; d++) Hm(d,d) += lambda * JtJ(d,d);
                Eigen::VectorXd delta = Hm.ldlt().solve(-Jtr);
                Eigen::VectorXd pNew = p + delta;
                Eigen::VectorXd rNew(nRes); residuals(pNew, rNew); applyWeights(rNew);
                double errNew = rNew.squaredNorm();
                if (errNew < err) {
                    p = pNew; r = rNew;
                    double rel = (err - errNew) / std::max(err, 1e-12);
                    err = errNew;
                    lambda = std::max(lambda*0.5, 1e-9);
                    if (rel < 1e-7 || delta.norm() < 1e-8) break;
                } else {
                    lambda *= 4.0;
                    if (lambda > 1e10) break;
                }
            }
        };

        // Pass 1: LM on all points
        runLM();

        // IRLS pass: compute per-point unweighted residual magnitudes,
        // estimate sigma from the median, and zero-weight any point with
        // residual > 3*sigma_robust. Then re-run LM.
        {
            Eigen::VectorXd rRaw(nRes); residuals(p, rRaw);
            std::vector<double> mags(nImg * M);
            for (int k=0; k<nImg; k++) for (int i=0; i<M; i++) {
                    double du = rRaw(2*(k*M+i)), dv = rRaw(2*(k*M+i)+1);
                    mags[k*M+i] = std::sqrt(du*du + dv*dv);
                }
            std::vector<double> sorted = mags;
            std::sort(sorted.begin(), sorted.end());
            double med = sorted[sorted.size()/2];
            // Robust sigma: 1.4826 * median (MAD-equivalent for 1D).
            // Threshold = max(3*sigma, 2px) so we don't over-trim when the
            // calibration is already excellent.
            double sigmaR = 1.4826 * med;
            double thresh = std::max(3.0 * sigmaR, 2.0);

            int nOut = 0;
            for (int k=0; k<nImg; k++) for (int i=0; i<M; i++) {
                    if (mags[k*M+i] > thresh) { ptW(k*M+i) = 0.0; nOut++; }
                }
            if (nOut > 0 && nOut < nImg*M/4) {
                residuals(p, r); applyWeights(r);
                runLM();
            }
        }

        // Unpack
        res.fx=p(0); res.fy=p(1); res.cx=p(2); res.cy=p(3);
        res.k1=p(4); res.k2=p(5);
        for (int k=0; k<nImg; k++) {
            Eigen::Vector3d rod(p(6+6*k+0), p(6+6*k+1), p(6+6*k+2));
            Rs[k] = rodrigues(rod);
            ts[k] = Eigen::Vector3d(p(6+6*k+3), p(6+6*k+4), p(6+6*k+5));
        }
    }

    // Final RMS (inlier-only). After the IRLS pass, points with residual >
    // robust 3-sigma threshold are flagged as outliers via ptW=0; we report
    // RMS over inliers only so a few bad detections don't inflate the metric.
    double totalErrIn=0; int totalPtsIn=0;
    double totalErrAll=0; int totalPtsAll=0;
    int nOut = 0, M = (int)objPts.size();
    for(int k=0;k<nImg;k++){
        for(int i=0;i<M;i++){
            Eigen::Vector3d Mw(objPts[i].x,objPts[i].y,0), Mc=Rs[k]*Mw+ts[k];
            if(std::abs(Mc(2))<1e-8) continue;
            double xn=Mc(0)/Mc(2), yn=Mc(1)/Mc(2);
            double r2=xn*xn+yn*yn, distort=1.0+res.k1*r2+res.k2*r2*r2;
            double up=res.fx*xn*distort+res.cx, vp=res.fy*yn*distort+res.cy;
            double uo=allImg[k][i].x, vo=allImg[k][i].y;
            double sq = (uo-up)*(uo-up)+(vo-vp)*(vo-vp);
            totalErrAll += sq; totalPtsAll++;
            bool inlier = (ptW.size() == 0) || (ptW(k*M+i) > 0.5);
            if (inlier) { totalErrIn += sq; totalPtsIn++; }
            else nOut++;
        }
    }

    res.rmsError = std::sqrt(totalErrIn / std::max(totalPtsIn, 1));
    res.numImages=nImg; res.numPoints=totalPtsIn; res.valid=true;
    res.message = (nOut > 0)
                      ? ("OK (" + std::to_string(nOut) + " outliers excluded, all-pt RMS=" +
                         std::to_string(std::sqrt(totalErrAll/std::max(totalPtsAll,1))) + ")")
                      : "OK";
    return res;
}

} // namespace detail

// ============================================================================
//  Public API
// ============================================================================

inline Result calibrateFromFolder(const std::string& folder, Board board) {
    Result res;
    if(!std::filesystem::exists(folder)){res.message="Folder not found: "+folder;return res;}
    std::vector<std::string> paths;
    for(auto&e:std::filesystem::directory_iterator(folder)){
        auto ext=e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if(ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp") paths.push_back(e.path().string());
    }
    std::sort(paths.begin(),paths.end());
    std::cout<<"[Calib] "<<paths.size()<<" images in "<<folder<<std::endl;
    if(paths.size()<3){res.message="Need >= 3 images (found "+std::to_string(paths.size())+")";return res;}

    std::vector<Pt2> objPts;
    for(int r=0;r<board.rows;r++) for(int c=0;c<board.cols;c++)
            objPts.push_back({c*(double)board.sqMM,r*(double)board.sqMM});

    std::vector<std::vector<Pt2>> allImgPts; std::vector<Eigen::Matrix3d> homos;
    int imgW=0,imgH=0;

    for(auto&path:paths){
        int w,h,ch; uint8_t*data=stbi_load(path.c_str(),&w,&h,&ch,0);
        if(!data){std::cerr<<"[Calib] Load fail: "<<path<<std::endl;continue;}
        if(imgW==0){imgW=w;imgH=h;}
        std::cout<<"[Calib] "<<std::filesystem::path(path).filename().string()<<" ("<<w<<"x"<<h<<")"<<std::endl;
        auto corners=detail::detectCorners(data,w,h,ch,board);
        stbi_image_free(data);
        if(corners.empty()){std::cerr<<"[Calib]   SKIP"<<std::endl;continue;}
        std::cout<<"[Calib]   OK ("<<corners.size()<<" corners)"<<std::endl;
        homos.push_back(detail::computeHomography(objPts,corners));
        allImgPts.push_back(corners);
    }

    std::cout<<"[Calib] "<<homos.size()<<"/"<<paths.size()<<" images OK"<<std::endl;
    if((int)homos.size()<3){res.message="Only "+std::to_string(homos.size())+" images (need 3+)";return res;}

    res=detail::zhangCalibrate(homos,allImgPts,objPts,imgW,imgH);
    if(res.valid){
        std::cout<<"\n[Calib] === Result ===\n"
                  <<"  fx="<<res.fx<<"  fy="<<res.fy<<"\n"
                  <<"  cx="<<res.cx<<"  cy="<<res.cy<<"\n"
                  <<"  k1="<<res.k1<<"  k2="<<res.k2<<"\n"
                  <<"  "<<res.width<<"x"<<res.height<<"\n"
                  <<"  RMS="<<res.rmsError<<" px ("<<res.numImages<<" images, "<<res.numPoints<<" points)\n"
                  <<"  ================\n"<<std::endl;
    }
    return res;
}

inline bool saveResult(const std::string& path, const Result& r) {
    std::ofstream ofs(path); if(!ofs.is_open()) return false;
    ofs<<"fx "<<r.fx<<"\nfy "<<r.fy<<"\ncx "<<r.cx<<"\ncy "<<r.cy
        <<"\nwidth "<<r.width<<"\nheight "<<r.height
        <<"\nk1 "<<r.k1<<"\nk2 "<<r.k2<<"\nrms "<<r.rmsError<<"\n";
    std::cout<<"[Calib] Saved: "<<path<<std::endl; return true;
}

} // namespace calib
