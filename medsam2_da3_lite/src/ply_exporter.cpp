#include "ply_exporter.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ply {

static std::vector<float> normalizeDepthFloat(
    const std::vector<float>& depth,
    const std::vector<uint8_t>& mask,
    int width, int height,
    bool invert)
{
    int n = width * height;
    std::vector<float> out(n, 0.0f);
    bool hasMask = (static_cast<int>(mask.size()) == n);

    std::vector<float> values;
    values.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!hasMask || mask[i] > 0) values.push_back(depth[i]);
    }
    if (values.empty()) return out;

    std::sort(values.begin(), values.end());
    size_t sz = values.size();
    float lo = values[static_cast<size_t>(sz * 0.02)];
    float hi = values[static_cast<size_t>(sz * 0.98)];
    float range = hi - lo;
    if (range < 1e-6f) range = 1.0f;

    for (int i = 0; i < n; ++i) {
        if (hasMask && mask[i] == 0) { out[i] = 0.0f; continue; }
        float v = std::clamp(depth[i], lo, hi);
        float t = (v - lo) / range;
        if (invert) t = 1.0f - t;
        out[i] = t;
    }
    return out;
}

bool saveTexturedPly(
    const std::string& outPath,
    const img::Image& rgbImage,
    const std::vector<float>& depthRaw,
    const std::vector<uint8_t>& mask,
    const ExportOptions& opt)
{
    if (rgbImage.empty() || rgbImage.channels != 3) {
        std::cerr << "[ply] Invalid RGB image" << std::endl;
        return false;
    }
    const int W = rgbImage.width;
    const int H = rgbImage.height;
    const int N = W * H;
    if (static_cast<int>(depthRaw.size()) != N) {
        std::cerr << "[ply] Depth size mismatch: expected "
                  << N << ", got " << depthRaw.size() << std::endl;
        return false;
    }

    bool hasMask = (static_cast<int>(mask.size()) == N);
    bool hasConf = (opt.confidence != nullptr &&
                    static_cast<int>(opt.confidence->size()) == N &&
                    opt.confidenceMin > 0.0f);

    if (opt.projection == Projection::PinholeMetric && !opt.intrinsics.valid()) {
        std::cerr << "[ply] PinholeMetric requires valid intrinsics" << std::endl;
        return false;
    }

    std::vector<float> dUsed;
    if (opt.normalize == Normalize::None) {
        dUsed.assign(depthRaw.begin(), depthRaw.end());
        if (opt.invertDepth) {
            for (auto& v : dUsed) v = -v;
        }
        if (opt.maskMode == MaskMode::KeepOutsideFlat && hasMask) {
            for (int i = 0; i < N; ++i) if (mask[i] == 0) dUsed[i] = 0.0f;
        }
    } else {
        dUsed = normalizeDepthFloat(depthRaw, mask, W, H, opt.invertDepth);
    }

    const float aspect  = static_cast<float>(W) / static_cast<float>(H);
    const float halfThk = opt.thickness * 0.5f;
    const float k = 2.0f * std::tan(opt.fovDeg * 0.5f * static_cast<float>(M_PI) / 180.0f);

    std::vector<float>   pts;
    std::vector<uint8_t> cols;
    pts.reserve(static_cast<size_t>(N) * 3);
    cols.reserve(static_cast<size_t>(N) * 3);

    size_t skippedMask = 0, skippedConf = 0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;

            if (opt.maskMode == MaskMode::SkipOutside && hasMask && mask[i] == 0) {
                skippedMask++;
                continue;
            }
            if (hasConf && (*opt.confidence)[i] < opt.confidenceMin) {
                skippedConf++;
                continue;
            }

            float u = (W > 1) ? static_cast<float>(x) / static_cast<float>(W - 1) : 0.5f;
            float v = (H > 1) ? static_cast<float>(y) / static_cast<float>(H - 1) : 0.5f;
            float d = dUsed[i];

            float X, Y, Z;
            if (opt.projection == Projection::PinholeMetric) {
                Z = d * opt.depthScale;
                X = (static_cast<float>(x) - opt.intrinsics.cx) * Z / opt.intrinsics.fx;
                Y = (static_cast<float>(y) - opt.intrinsics.cy) * Z / opt.intrinsics.fy;
                if (opt.flipY) Y = -Y;
            } else if (opt.projection == Projection::Pinhole) {
                float z = halfThk + d * opt.depthScale;
                X = (u - 0.5f) * aspect * k * z;
                Y = (0.5f - v) * k * z;
                Z = z;
            } else {
                X = (u - 0.5f) * aspect;
                Y = (0.5f - v);
                Z = halfThk + d * opt.depthScale;
            }

            pts.push_back(X);
            pts.push_back(Y);
            pts.push_back(Z);
            cols.push_back(rgbImage.data[i * 3 + 0]);
            cols.push_back(rgbImage.data[i * 3 + 1]);
            cols.push_back(rgbImage.data[i * 3 + 2]);
        }
    }

    size_t count = pts.size() / 3;

    std::ofstream ofs(outPath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[ply] Failed to open: " << outPath << std::endl;
        return false;
    }

    ofs << "ply\n";
    ofs << (opt.binary ? "format binary_little_endian 1.0\n" : "format ascii 1.0\n");
    ofs << "element vertex " << count << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";

    if (opt.binary) {
        for (size_t i = 0; i < count; ++i) {
            ofs.write(reinterpret_cast<const char*>(&pts[i * 3]), sizeof(float) * 3);
            ofs.write(reinterpret_cast<const char*>(&cols[i * 3]), 3);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            ofs << pts[i * 3 + 0] << " " << pts[i * 3 + 1] << " " << pts[i * 3 + 2]
                << " " << static_cast<int>(cols[i * 3 + 0])
                << " " << static_cast<int>(cols[i * 3 + 1])
                << " " << static_cast<int>(cols[i * 3 + 2]) << "\n";
        }
    }

    std::cout << "[ply] Saved " << count << " points to " << outPath;
    if (skippedMask > 0) std::cout << " (mask-skipped " << skippedMask << ")";
    if (skippedConf > 0) std::cout << " (conf-skipped " << skippedConf << ")";
    std::cout << std::endl;
    return true;
}

}
