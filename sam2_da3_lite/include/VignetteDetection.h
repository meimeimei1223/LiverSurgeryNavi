// =============================================================================
//  VignetteDetection.h    (v3b -- multi-CC union + edge band)
//  ---------------------------------------------------------------------------
//  Detect the black FOV-cutoff (vignette) at the periphery of a laparoscopic
//  image and return a binary mask suitable for OR-merging into the occluder
//  mask (instrument_segmentation_mask.png).
//
//  ALGORITHM (geometric, robust to dark scenes & instrument-split FOVs):
//    0. (Optional) Stamp the top N and bottom N image rows as vignette
//       unconditionally. Many laparoscopic clips have JPEG noise / colour
//       fringing at the very top/bottom rows even when the FOV touches the
//       image edge; the band keeps those rows out of registration too.
//       Controlled by Params::edgeBand (default 3 px).
//    1. Sample 3x3 patches at the 4 image corners (with a 2-px inset).
//       Take cornerMaxLum = max luminance among them. If cornerMaxLum
//       exceeds brightCornerCutoff, the image likely has no vignette
//       body, so we return with edge-band-only and valid = (edgeBand > 0).
//    2. Compute dynamic dark threshold T = cornerMaxLum + cornerBoost.
//       Adapts to image gain: a clip whose black level is boosted to
//       RGB(39,39,39) still gets a useful threshold above the corner
//       darkness.
//    3. Build an "interior-candidate" mask:
//          interior = (lum >= T) OR (sat >= satThresh)
//       where sat = max(R,G,B) - min(R,G,B). The OR is intentional:
//       a pixel is FOV interior if it's either bright OR colourful.
//       Only pixels that are simultaneously dark AND desaturated remain
//       vignette candidates.
//    4. Find ALL connected components of interior candidates whose size
//       is >= minCCFraction * total_pixels (default 0.5%). Keep their
//       union. This handles the case where a horizontal instrument
//       splits the FOV interior into upper and lower halves -- both
//       qualify and the hull then captures the full FOV. Tiny CCs (JPEG
//       artefacts < ~100 px) are excluded as noise.
//    5. Compute the 2D convex hull of the kept-CC union and rasterize it.
//       Geometric constraint that fixes earlier failure modes:
//         - instrument shafts inside the FOV stay inside the hull (NOT
//           inside the vignette -- user marks those via SAM2 separately)
//         - dark-shadow notches inside the FOV stay inside the hull
//         - the hull matches the convex shape the lens projects: circle,
//           ellipse, or circle clipped by image edges -- all convex by
//           construction
//    6. vignette_body = NOT hull, then dilated by dilateRadius for
//       JPEG-halo safety margin.
//    7. Sanity check: vignette_body pixel count must fall in
//       [minFraction, maxFraction] of total image area. Outside that
//       range the BODY is discarded as a false positive. The edge band
//       remains regardless (it does not need the body to be valid).
//    8. final mask = vignette_body OR edge_band.
//
//  COMPLEXITY: O(w*h) for the gate + CC labelling, O(h) row-extreme
//  extraction, O(h log h) convex hull, O(w*h) rasterization + dilation.
//  Total ~25-50 ms at 1920x1080.
//
//  USAGE:
//    auto vig = VignetteDetection::detect(image.data.data(),
//                                         image.width, image.height);
//    if (vig.valid) {
//        // OR vig.mask into the occluder mask, then save.
//    }
//
//  THREAD SAFETY: stateless; pure function over (rgb, w, h, params).
// =============================================================================

#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <utility>

namespace VignetteDetection {

// ---------------------------------------------------------------------------
//  Tunables.
// ---------------------------------------------------------------------------
struct Params {
    // Force the top N and bottom N rows of the image to be vignette
    // unconditionally. Default 3 catches the typical JPEG/codec artefact
    // band at the very image edges. Set to 0 to disable; clipped to h/2
    // internally if larger.
    int edgeBand = 3;

    // T_dark = cornerMaxLum + cornerBoost. Larger -> more lenient
    // classifier for "dark-and-desaturated == vignette candidate".
    int cornerBoost = 20;

    // max(R,G,B) - min(R,G,B) gate. Below this -> "no hue". Larger lets
    // more colour into the vignette mask (typically undesired).
    int satThresh = 12;

    // Bail-out if the brightest corner luminance exceeds this. 60 covers
    // gain-boosted clinical footage whose black level rests around 35-45.
    int brightCornerCutoff = 60;

    // Minimum CC size (as a fraction of total pixels) for a connected
    // component of "interior candidates" to be included in the hull
    // computation. Tiny CCs (typically <100 px JPEG artefacts) are
    // filtered as noise; substantial regions (>0.5% of image) are kept
    // even if they are not the LARGEST CC. This handles the case where
    // an instrument horizontally splits the FOV interior into two halves.
    float minCCFraction = 0.005f;   // 0.5%

    // Vignette safety margin after the convex hull is computed (px).
    // The vignette is dilated outward (= the hull is eroded inward) by
    // this many pixels so JPEG halos at the FOV edge are excluded.
    int dilateRadius = 2;

    // Reject pathologically small (false-positive) or large (whole-image
    // dark) BODY detections. Fractions of the total pixel count.
    // The edge band is independent of these checks.
    float minFraction = 0.005f;   // 0.5%
    float maxFraction = 0.70f;    // 70%

    // Console one-line summary of the decision. Set false to silence.
    bool verbose = true;
};

struct Result {
    std::vector<uint8_t> mask;   // w*h, 0 or 255.
    // True iff the mask has any non-zero pixels (body found OR edgeBand>0).
    bool valid = false;
    int pixelCount = 0;
    int cornerLumMax = 0;
    int threshold = 0;            // T_dark = cornerLumMax + cornerBoost
    bool bodyDetected = false;    // True iff the convex-hull body passed sanity.
};

// ===========================================================================
//  Implementation helpers
// ===========================================================================
namespace detail {

// 3x3 patch luminance at (cx,cy), skipping out-of-bounds neighbours.
inline int patchLum(const uint8_t* rgb, int w, int h, int cx, int cy) {
    int sum = 0, n = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int y = cy + dy;
        if (y < 0 || y >= h) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int x = cx + dx;
            if (x < 0 || x >= w) continue;
            const int i = (y * w + x) * 3;
            sum += (rgb[i] + rgb[i + 1] + rgb[i + 2]) / 3;
            ++n;
        }
    }
    return n > 0 ? sum / n : 0;
}

// 4-connectivity BFS connected-component labelling. Stores the label of
// each foreground pixel into labels[]; returns nLabels (1-indexed count)
// and per-label sizes via compSizes[label].
inline void labelComponents(const std::vector<bool>& mask, int w, int h,
                            std::vector<int>& labels,
                            std::vector<int>& compSizes,
                            int& nLabels)
{
    labels.assign((size_t)w * h, 0);
    compSizes.clear();
    compSizes.push_back(0);   // label 0 = background sentinel
    nLabels = 0;

    static const int dx4[] = {1, -1, 0, 0};
    static const int dy4[] = {0, 0, 1, -1};

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i0 = (size_t)y * w + x;
            if (!mask[i0] || labels[i0] != 0) continue;
            ++nLabels;
            int compSize = 0;
            std::queue<std::pair<int, int>> q;
            q.push({x, y});
            labels[i0] = nLabels;
            while (!q.empty()) {
                const auto [cx, cy] = q.front();
                q.pop();
                ++compSize;
                for (int d = 0; d < 4; ++d) {
                    const int nx = cx + dx4[d];
                    const int ny = cy + dy4[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const size_t ni = (size_t)ny * w + nx;
                    if (!mask[ni] || labels[ni] != 0) continue;
                    labels[ni] = nLabels;
                    q.push({nx, ny});
                }
            }
            compSizes.push_back(compSize);
        }
    }
}

// Build a mask containing only the pixels whose CC label has size >=
// minSize. Returns the total kept-pixel count.
inline int keepLargeCCs(const std::vector<int>& labels,
                        const std::vector<int>& compSizes,
                        int w, int h, int minSize,
                        std::vector<bool>& outMask)
{
    outMask.assign((size_t)w * h, false);
    int kept = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            const int lab = labels[i];
            if (lab > 0 && compSizes[(size_t)lab] >= minSize) {
                outMask[i] = true;
                ++kept;
            }
        }
    }
    return kept;
}

// Extract row-extreme boundary points of a binary mask. For each row that
// has any set pixel, we add the leftmost and rightmost x as a boundary
// point. Sufficient for convex-hull computation because the hull depends
// only on extreme points along each row.
inline std::vector<std::pair<int, int>>
extractRowExtremes(const std::vector<bool>& mask, int w, int h)
{
    std::vector<std::pair<int, int>> pts;
    pts.reserve((size_t)h * 2);
    for (int y = 0; y < h; ++y) {
        int xMin = -1, xMax = -1;
        for (int x = 0; x < w; ++x) {
            if (mask[(size_t)y * w + x]) {
                xMax = x;
                if (xMin < 0) xMin = x;
            }
        }
        if (xMin >= 0) {
            pts.push_back({xMin, y});
            if (xMax != xMin) pts.push_back({xMax, y});
        }
    }
    return pts;
}

// 2D convex hull via Andrew's monotone chain. O(n log n). Counter-clockwise
// order starting from the lowest-leftmost point; last point is NOT a
// duplicate of the first. 64-bit cross product is used to avoid overflow
// on full-HD coords (~2e6 squared = 4e12, fits in int64 with room).
inline std::vector<std::pair<int, int>>
convexHull2D(std::vector<std::pair<int, int>> P)
{
    const int n = (int)P.size();
    if (n < 3) return P;
    std::sort(P.begin(), P.end());

    auto cross = [](const std::pair<int, int>& O,
                    const std::pair<int, int>& A,
                    const std::pair<int, int>& B) -> long long {
        return (long long)(A.first  - O.first ) * (long long)(B.second - O.second)
             - (long long)(A.second - O.second) * (long long)(B.first  - O.first );
    };

    std::vector<std::pair<int, int>> H(2 * n);
    int k = 0;
    // Lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0) --k;
        H[k++] = P[i];
    }
    // Upper hull
    for (int i = n - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && cross(H[k - 2], H[k - 1], P[i]) <= 0) --k;
        H[k++] = P[i];
    }
    H.resize(k - 1);
    return H;
}

// Rasterize a convex polygon. For each scanline y, we trace every polygon
// edge and update {leftX[y], rightX[y]} with the intersection x. Because
// the polygon is convex, exactly two edges cross each interior row, and
// {leftX,rightX} fully describes the inside region.
inline void rasterizeConvexPolygon(
    const std::vector<std::pair<int, int>>& poly,
    int w, int h, std::vector<bool>& outMask)
{
    outMask.assign((size_t)w * h, false);
    if (poly.size() < 3) return;

    std::vector<int> leftX(h, w + 1);
    std::vector<int> rightX(h, -1);

    const int n = (int)poly.size();
    for (int i = 0; i < n; ++i) {
        int x1 = poly[i].first,             y1 = poly[i].second;
        int x2 = poly[(i + 1) % n].first,   y2 = poly[(i + 1) % n].second;
        if (y1 == y2) {
            if (y1 < 0 || y1 >= h) continue;
            const int xLo = std::min(x1, x2);
            const int xHi = std::max(x1, x2);
            if (xLo < leftX[y1])  leftX[y1]  = xLo;
            if (xHi > rightX[y1]) rightX[y1] = xHi;
            continue;
        }
        if (y1 > y2) { std::swap(x1, x2); std::swap(y1, y2); }
        const int yLo = std::max(0, y1);
        const int yHi = std::min(h - 1, y2);
        for (int y = yLo; y <= yHi; ++y) {
            const int x = x1 + (x2 - x1) * (y - y1) / (y2 - y1);
            if (x < leftX[y])  leftX[y]  = x;
            if (x > rightX[y]) rightX[y] = x;
        }
    }

    for (int y = 0; y < h; ++y) {
        if (rightX[y] < 0 || leftX[y] > rightX[y]) continue;
        const int x0 = std::max(0, leftX[y]);
        const int x1 = std::min(w - 1, rightX[y]);
        for (int x = x0; x <= x1; ++x)
            outMask[(size_t)y * w + x] = true;
    }
}

// Square-shaped binary dilation. radius==1 yields a 3x3 element, radius==2
// a 5x5, etc. In-place via a scratch copy.
inline void dilate(std::vector<uint8_t>& mask, int w, int h, int radius) {
    if (radius <= 0) return;
    std::vector<uint8_t> tmp(mask);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (mask[(size_t)y * w + x]) continue;
            bool hit = false;
            for (int dy = -radius; dy <= radius && !hit; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= h) continue;
                for (int dx = -radius; dx <= radius && !hit; ++dx) {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= w) continue;
                    if (tmp[(size_t)ny * w + nx]) hit = true;
                }
            }
            if (hit) mask[(size_t)y * w + x] = 255;
        }
    }
}

// Apply the top/bottom edge band unconditionally. `band` is clipped to h/2
// internally to avoid over-stamping.
inline void applyEdgeBand(std::vector<uint8_t>& mask, int w, int h, int band) {
    if (band <= 0) return;
    const int eb = std::min(band, h / 2);
    for (int y = 0; y < eb; ++y)
        for (int x = 0; x < w; ++x)
            mask[(size_t)y * w + x] = 255;
    for (int y = h - eb; y < h; ++y)
        for (int x = 0; x < w; ++x)
            mask[(size_t)y * w + x] = 255;
}

} // namespace detail

// ===========================================================================
//  Public entry point.
// ===========================================================================
inline Result detect(const uint8_t* rgb, int width, int height,
                     const Params& params = Params{}) {
    Result r;
    r.mask.assign((size_t)width * height, 0);

    if (!rgb || width < 4 || height < 4) {
        if (params.verbose)
            std::cerr << "[VignetteDetection] invalid input "
                      << "(rgb=" << (rgb ? "ok" : "null")
                      << " w=" << width << " h=" << height << ")\n";
        return r;
    }

    // ---- Step 0: edge band (unconditional, kept even if body fails) -------
    detail::applyEdgeBand(r.mask, width, height, params.edgeBand);
    const int edgeBandPx = std::min(params.edgeBand, height / 2);

    // ---- Step 1: corner sampling ------------------------------------------
    const int corners[4] = {
        detail::patchLum(rgb, width, height, 2,         2),
        detail::patchLum(rgb, width, height, width - 3, 2),
        detail::patchLum(rgb, width, height, 2,         height - 3),
        detail::patchLum(rgb, width, height, width - 3, height - 3),
    };
    const int cMax = *std::max_element(corners, corners + 4);
    r.cornerLumMax = cMax;

    if (cMax > params.brightCornerCutoff) {
        if (params.verbose)
            std::cout << "[VignetteDetection] no vignette body: corner_max_lum="
                      << cMax << " > cutoff=" << params.brightCornerCutoff
                      << " (edge_band only)\n";
        r.bodyDetected = false;
        r.pixelCount = 0;
        for (auto v : r.mask) if (v) ++r.pixelCount;
        r.valid = (r.pixelCount > 0);
        return r;
    }

    // ---- Step 2: dynamic threshold ----------------------------------------
    const int T = cMax + params.cornerBoost;
    r.threshold = T;

    // ---- Step 3: interior-candidate mask ----------------------------------
    const size_t N = (size_t)width * height;
    std::vector<bool> interior(N, false);
    int interiorCount = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = ((size_t)y * width + x) * 3;
            const int rr = rgb[i], gg = rgb[i + 1], bb = rgb[i + 2];
            const int lum = (rr + gg + bb) / 3;
            const int hi  = std::max({rr, gg, bb});
            const int lo  = std::min({rr, gg, bb});
            if (lum >= T || (hi - lo) >= params.satThresh) {
                interior[(size_t)y * width + x] = true;
                ++interiorCount;
            }
        }
    }

    // ---- Step 4: keep all CCs >= minCCFraction of image -------------------
    std::vector<int> labels, compSizes;
    int nLabels = 0;
    detail::labelComponents(interior, width, height, labels, compSizes, nLabels);

    if (nLabels == 0) {
        if (params.verbose)
            std::cout << "[VignetteDetection] no CCs found (edge_band only)\n";
        r.bodyDetected = false;
        r.pixelCount = 0;
        for (auto v : r.mask) if (v) ++r.pixelCount;
        r.valid = (r.pixelCount > 0);
        return r;
    }

    const int total = width * height;
    const int minCCSize = std::max(100,
                                   (int)((double)params.minCCFraction * total));
    std::vector<bool> keptMask;
    const int keptPx = detail::keepLargeCCs(labels, compSizes,
                                            width, height, minCCSize, keptMask);

    if (keptPx == 0) {
        if (params.verbose)
            std::cout << "[VignetteDetection] no CC >= " << minCCSize
                      << "px (edge_band only)\n";
        r.bodyDetected = false;
        r.pixelCount = 0;
        for (auto v : r.mask) if (v) ++r.pixelCount;
        r.valid = (r.pixelCount > 0);
        return r;
    }

    // ---- Step 5: convex hull of kept CCs ----------------------------------
    std::vector<std::pair<int, int>> pts =
        detail::extractRowExtremes(keptMask, width, height);

    std::vector<std::pair<int, int>> hull = detail::convexHull2D(std::move(pts));
    if (hull.size() < 3) {
        if (params.verbose)
            std::cout << "[VignetteDetection] degenerate hull (edge_band only)\n";
        r.bodyDetected = false;
        r.pixelCount = 0;
        for (auto v : r.mask) if (v) ++r.pixelCount;
        r.valid = (r.pixelCount > 0);
        return r;
    }

    std::vector<bool> hullMask;
    detail::rasterizeConvexPolygon(hull, width, height, hullMask);

    // ---- Step 6: vignette_body = NOT hull, then dilate --------------------
    std::vector<uint8_t> bodyMask((size_t)width * height, 0);
    for (size_t i = 0; i < N; ++i) {
        if (!hullMask[i]) bodyMask[i] = 255;
    }
    detail::dilate(bodyMask, width, height, params.dilateRadius);

    // ---- Step 7: body sanity check ----------------------------------------
    int bodyPx = 0;
    for (auto v : bodyMask) if (v) ++bodyPx;
    const int minPx = (int)((double)params.minFraction * (double)total);
    const int maxPx = (int)((double)params.maxFraction * (double)total);

    bool bodyOk = (bodyPx >= minPx) && (bodyPx <= maxPx);
    if (!bodyOk) {
        if (params.verbose)
            std::cout << "[VignetteDetection] body discarded: "
                      << bodyPx << " px (" << (100.0 * bodyPx / total)
                      << "%) outside [" << minPx << "," << maxPx
                      << "] (edge_band only)\n";
        r.bodyDetected = false;
    } else {
        // ---- Step 8: merge body into result mask --------------------------
        for (size_t i = 0; i < N; ++i) {
            if (bodyMask[i]) r.mask[i] = 255;
        }
        r.bodyDetected = true;
    }

    r.pixelCount = 0;
    for (auto v : r.mask) if (v) ++r.pixelCount;
    r.valid = (r.pixelCount > 0);

    if (params.verbose) {
        const double pct = 100.0 * (double)r.pixelCount / (double)total;
        std::cout << "[VignetteDetection] "
                  << (r.bodyDetected ? "OK" : "edge_band only")
                  << ": corner_max_lum=" << cMax
                  << " T=" << T
                  << " interior=" << interiorCount
                  << " (" << (100.0 * interiorCount / total) << "%)"
                  << " ccs=" << nLabels
                  << " kept_ccs_px=" << keptPx
                  << " (" << (100.0 * keptPx / total) << "%)"
                  << " hull_pts=" << hull.size()
                  << " body=" << bodyPx
                  << " edge_band=" << edgeBandPx << "px"
                  << " total=" << r.pixelCount
                  << " (" << pct << "%)\n";
    }
    return r;
}

} // namespace VignetteDetection
