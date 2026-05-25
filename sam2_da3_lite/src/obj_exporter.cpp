#include "obj_exporter.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <cstdio>

namespace objexp {

namespace fs = std::filesystem;

static std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    return (fs::path(dir) / name).string();
}

static inline float computeZ(float d, float scale, ZMode mode, float maxD) {
    switch (mode) {
    case ZMode::Negated:      return -d * scale;
    case ZMode::HillInverted: return (maxD - d) * scale;
    default:                  return  d * scale;
    }
}

static inline void projectVertex(
    int x, int y, int W, int H,
    float d, float maxD,
    const ObjExportOptions& opt,
    float& X, float& Y, float& Z)
{
    if (opt.projection == Projection::PlaneRelief) {
        float u = (W > 1) ? (float)x / (float)(W - 1) : 0.5f;
        float v = (H > 1) ? (float)y / (float)(H - 1) : 0.5f;
        float aspect = (float)W / (float)H;
        float dd = opt.invertDepth ? -d : d;
        X = (u - 0.5f) * aspect;
        Y = opt.flipY ? (0.5f - v) : (v - 0.5f);
        Z = opt.thickness * 0.5f + dd * opt.depthScale;
    } else {
        float fx = opt.intrinsics.fx, fy = opt.intrinsics.fy;
        float cx = opt.intrinsics.cx, cy = opt.intrinsics.cy;
        Z = computeZ(d, opt.depthScale, opt.zMode, maxD);
        X = ((float)x - cx) * Z / fx;
        Y = ((float)y - cy) * Z / fy;
        if (opt.flipY) Y = -Y;
    }
}

static inline float meshZ(float d, float maxD, const ObjExportOptions& opt) {
    if (opt.projection == Projection::PlaneRelief) {
        float dd = opt.invertDepth ? -d : d;
        return opt.thickness * 0.5f + dd * opt.depthScale;
    }
    return computeZ(d, opt.depthScale, opt.zMode, maxD);
}

static bool writeMtl(const std::string& mtlPath, const ObjExportOptions& opt) {
    std::ofstream ofs(mtlPath);
    if (!ofs.is_open()) return false;
    ofs << "newmtl " << opt.materialName << "\n";
    ofs << "Ka 1.000 1.000 1.000\n";
    ofs << "Kd 1.000 1.000 1.000\n";
    ofs << "Ks 0.000 0.000 0.000\n";
    ofs << "d 1.0\nillum 1\n";
    if (!opt.textureFilename.empty()) ofs << "map_Kd " << opt.textureFilename << "\n";
    return true;
}

static bool writeTextureImage(const std::string& dir,
                              const ObjExportOptions& opt,
                              const img::Image& rgb)
{
    if (!opt.writeTexture || opt.textureFilename.empty()) return true;
    return img::saveImage(joinPath(dir, opt.textureFilename), rgb);
}

static void streamVertex(std::ostringstream& buf, float X, float Y, float Z) {
    char tmp[96];
    std::snprintf(tmp, sizeof(tmp), "v %.6f %.6f %.6f\n", X, Y, Z);
    buf << tmp;
}

static void streamUV(std::ostringstream& buf, float u, float v) {
    char tmp[64];
    std::snprintf(tmp, sizeof(tmp), "vt %.6f %.6f\n", u, v);
    buf << tmp;
}

static void streamFace(std::ostringstream& buf, int a, int b, int c) {
    char tmp[96];
    std::snprintf(tmp, sizeof(tmp), "f %d/%d %d/%d %d/%d\n", a, a, b, b, c, c);
    buf << tmp;
}

bool saveFullMeshObj(
    const std::string& outPath,
    const img::Image& rgbImage,
    const std::vector<float>& depthMetric,
    const ObjExportOptions& opt)
{
    if (rgbImage.empty() || rgbImage.channels != 3) {
        std::cerr << "[obj] Invalid RGB image" << std::endl; return false;
    }
    if (opt.projection == Projection::PinholeMetric && !opt.intrinsics.valid()) {
        std::cerr << "[obj] PinholeMetric requires valid intrinsics" << std::endl; return false;
    }
    const int W = rgbImage.width, H = rgbImage.height, N = W * H;
    if ((int)depthMetric.size() != N) {
        std::cerr << "[obj] Depth size mismatch" << std::endl; return false;
    }

    std::ofstream ofs(outPath);
    if (!ofs.is_open()) {
        std::cerr << "[obj] Failed to open: " << outPath << std::endl; return false;
    }

    std::string dir    = fs::path(outPath).parent_path().string();
    std::string stem   = fs::path(outPath).stem().string();
    std::string mtlRel = stem + ".mtl";
    std::string mtlAbs = joinPath(dir, mtlRel);

    writeMtl(mtlAbs, opt);
    writeTextureImage(dir, opt, rgbImage);

    ofs << "mtllib " << mtlRel << "\n";
    ofs << "o FullMesh\n";

    float maxD = 0.0f;
    for (int i = 0; i < N; ++i) if (depthMetric[i] > maxD) maxD = depthMetric[i];
    if (maxD <= 0.0f) maxD = 1.0f;

    const int S = std::max(1, opt.stride);
    const int cols = (W + S - 1) / S;  // stride grid columns

    std::ostringstream vbuf, tbuf, fbuf;
    int vertCount = 0;

    for (int y = 0; y < H; y += S) {
        for (int x = 0; x < W; x += S) {
            int i = y * W + x;
            float X, Y, Z;
            projectVertex(x, y, W, H, depthMetric[i], maxD, opt, X, Y, Z);
            streamVertex(vbuf, X, Y, Z);

            float u = (W > 1) ? (float)x / (float)(W - 1) : 0.5f;
            float v = (H > 1) ? 1.0f - (float)y / (float)(H - 1) : 0.5f;
            streamUV(tbuf, u, v);
            ++vertCount;
        }
    }

    fbuf << "usemtl " << opt.materialName << "\n";

    // OBJ indices are 1-based; map stride-grid (gx,gy) to vertex index
    auto idx = [cols](int gx, int gy) { return gy * cols + gx + 1; };
    auto zAt = [&](int x, int y) {
        return meshZ(depthMetric[y * W + x], maxD, opt);
    };
    auto validTri = [&](int x0, int y0, int x1, int y1, int x2, int y2) {
        if (opt.skirtThreshold <= 0.0f) return true;
        float z0 = zAt(x0, y0), z1 = zAt(x1, y1), z2 = zAt(x2, y2);
        float m = std::max({ std::abs(z0 - z1), std::abs(z1 - z2), std::abs(z2 - z0) });
        return m <= opt.skirtThreshold;
    };

    const int rows = (H + S - 1) / S;
    size_t triCount = 0;
    for (int gy = 0; gy < rows - 1; ++gy) {
        for (int gx = 0; gx < cols - 1; ++gx) {
            int x0 = gx * S, y0 = gy * S;
            int x1 = (gx + 1) * S, y1 = (gy + 1) * S;
            // clamp to image bounds (last stride column/row may overshoot)
            if (x1 >= W) x1 = W - 1;
            if (y1 >= H) y1 = H - 1;

            int a = idx(gx,     gy);
            int b = idx(gx + 1, gy);
            int c = idx(gx,     gy + 1);
            int d = idx(gx + 1, gy + 1);
            if (validTri(x0, y0, x0, y1, x1, y0)) {
                streamFace(fbuf, a, b, c); ++triCount;
            }
            if (validTri(x1, y0, x0, y1, x1, y1)) {
                streamFace(fbuf, b, d, c); ++triCount;
            }
        }
    }

    ofs << vbuf.str() << tbuf.str() << fbuf.str();
    std::cout << "[obj] Full mesh: " << vertCount << " verts, " << triCount
              << " tris (stride=" << S << ") -> " << outPath << std::endl;
    return true;
}

bool saveMaskedMeshObj(
    const std::string& outPath,
    const img::Image& rgbImage,
    const std::vector<float>& depthMetric,
    const std::vector<uint8_t>& mask,
    const ObjExportOptions& opt)
{
    if (rgbImage.empty() || rgbImage.channels != 3) {
        std::cerr << "[obj] Invalid RGB image" << std::endl; return false;
    }
    if (opt.projection == Projection::PinholeMetric && !opt.intrinsics.valid()) {
        std::cerr << "[obj] PinholeMetric requires valid intrinsics" << std::endl; return false;
    }
    const int W = rgbImage.width, H = rgbImage.height, N = W * H;
    if ((int)depthMetric.size() != N) {
        std::cerr << "[obj] Depth size mismatch" << std::endl; return false;
    }
    if ((int)mask.size() != N) {
        std::cerr << "[obj] Mask required for masked mesh" << std::endl; return false;
    }

    std::ofstream ofs(outPath);
    if (!ofs.is_open()) {
        std::cerr << "[obj] Failed to open: " << outPath << std::endl; return false;
    }

    std::string dir    = fs::path(outPath).parent_path().string();
    std::string stem   = fs::path(outPath).stem().string();
    std::string mtlRel = stem + ".mtl";
    std::string mtlAbs = joinPath(dir, mtlRel);

    writeMtl(mtlAbs, opt);
    writeTextureImage(dir, opt, rgbImage);

    ofs << "mtllib " << mtlRel << "\n";
    ofs << "o MaskedMesh\n";

    const float maxD = [&]() {
        float m = 0.0f;
        for (int i = 0; i < N; ++i) if (depthMetric[i] > m) m = depthMetric[i];
        return m > 0.0f ? m : 1.0f;
    }();

    std::vector<int> vertIdx(N, 0);
    std::ostringstream vbuf, tbuf, fbuf;
    int counter = 0;

    const bool useConf = (opt.confidence != nullptr &&
                          (int)opt.confidence->size() == N &&
                          opt.confidenceMin > 0.0f);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            if (mask[i] == 0) continue;
            if (useConf && (*opt.confidence)[i] < opt.confidenceMin) continue;
            float X, Y, Z;
            projectVertex(x, y, W, H, depthMetric[i], maxD, opt, X, Y, Z);
            streamVertex(vbuf, X, Y, Z);

            float u = (W > 1) ? (float)x / (float)(W - 1) : 0.5f;
            float v = (H > 1) ? 1.0f - (float)y / (float)(H - 1) : 0.5f;
            streamUV(tbuf, u, v);

            ++counter;
            vertIdx[i] = counter;
        }
    }

    fbuf << "usemtl " << opt.materialName << "\n";

    auto zAtIdx = [&](int i) {
        return meshZ(depthMetric[i], maxD, opt);
    };
    auto validTri = [&](int i0, int i1, int i2) {
        if (vertIdx[i0] == 0 || vertIdx[i1] == 0 || vertIdx[i2] == 0) return false;
        if (opt.skirtThreshold <= 0.0f) return true;
        float z0 = zAtIdx(i0), z1 = zAtIdx(i1), z2 = zAtIdx(i2);
        float m = std::max({ std::abs(z0 - z1), std::abs(z1 - z2), std::abs(z2 - z0) });
        return m <= opt.skirtThreshold;
    };

    size_t triCount = 0;
    for (int y = 0; y < H - 1; ++y) {
        for (int x = 0; x < W - 1; ++x) {
            int i00 = y * W + x;
            int i10 = y * W + (x + 1);
            int i01 = (y + 1) * W + x;
            int i11 = (y + 1) * W + (x + 1);
            if (validTri(i00, i01, i10)) {
                streamFace(fbuf, vertIdx[i00], vertIdx[i10], vertIdx[i01]);
                ++triCount;
            }
            if (validTri(i10, i01, i11)) {
                streamFace(fbuf, vertIdx[i10], vertIdx[i11], vertIdx[i01]);
                ++triCount;
            }
        }
    }

    ofs << vbuf.str() << tbuf.str() << fbuf.str();
    std::cout << "[obj] Masked mesh: " << counter << " verts, " << triCount
              << " tris -> " << outPath << std::endl;
    return true;
}

}
