#include "DepthToObjExport.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace depthexport {

// outDir may or may not end with '/'.
static std::string join(const std::string& dir, const std::string& f) {
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) return dir + f;
    return dir + "/" + f;
}

// Separable box dilation, bit-identical to img::dilateMask (image_utils.cpp).
// Replicated locally so this TU does not depend on sam2's image_utils.cpp symbol
// (which is its STB-implementation TU and is not linked into reg/deform).
static std::vector<uint8_t> dilateMaskLocal(const std::vector<uint8_t>& mask,
                                            int width, int height, int radius) {
    if (radius <= 0) return mask;
    std::vector<uint8_t> tmp(mask.size(), 0), out(mask.size(), 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            uint8_t v = 0;
            int x0 = std::max(0, x - radius), x1 = std::min(width - 1, x + radius);
            for (int xx = x0; xx <= x1; ++xx)
                if (mask[y * width + xx]) { v = 255; break; }
            tmp[y * width + x] = v;
        }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            uint8_t v = 0;
            int y0 = std::max(0, y - radius), y1 = std::min(height - 1, y + radius);
            for (int yy = y0; yy <= y1; ++yy)
                if (tmp[yy * width + x]) { v = 255; break; }
            out[y * width + x] = v;
        }
    return out;
}

bool saveIntrinsicsFile(const std::string& path,
                        const Reg3DCustom::CameraIntrinsics& K,
                        int width, int height) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[DepthExport] cannot write " << path << std::endl;
        return false;
    }
    ofs << std::setprecision(9);
    ofs << "fx "     << K.fx << "\n";
    ofs << "fy "     << K.fy << "\n";
    ofs << "cx "     << K.cx << "\n";
    ofs << "cy "     << K.cy << "\n";
    ofs << "width "  << width  << "\n";
    ofs << "height " << height << "\n";
    ofs << "name   " << (K.name.empty() ? "custom" : K.name) << "\n";
    // Brown-Conrady distortion: only when at least one coefficient is nonzero
    // (keeps distortion-free files byte-identical to the legacy format).
    if (K.hasDistortion()) {
        ofs << "# Brown-Conrady distortion (OpenCV convention)\n";
        ofs << "k1     " << K.k1 << "\n";
        ofs << "k2     " << K.k2 << "\n";
        ofs << "k3     " << K.k3 << "\n";
        ofs << "k4     " << K.k4 << "\n";
        ofs << "p1     " << K.p1 << "\n";
        ofs << "p2     " << K.p2 << "\n";
    }
    return true;
}

bool loadDepthMetricBin(const std::string& path, DepthBin& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    uint32_t magic = 0, w = 0, h = 0, reserved = 0;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&w), 4);
    ifs.read(reinterpret_cast<char*>(&h), 4);
    ifs.read(reinterpret_cast<char*>(&reserved), 4);
    if (!ifs || magic != 0x44455054u) {           // "DEPT"
        std::cerr << "[DepthExport] bad depth_metric.bin header: " << path << std::endl;
        return false;
    }
    if (w == 0 || h == 0 || w > 100000u || h > 100000u) return false;
    out.width  = (int)w;
    out.height = (int)h;
    out.depth.resize((size_t)w * h);
    ifs.read(reinterpret_cast<char*>(out.depth.data()),
             (std::streamsize)(out.depth.size() * sizeof(float)));
    return (bool)ifs && out.valid();
}

Result exportDepthArtifacts(const Request& req) {
    Result res;
    const int W = req.width, H = req.height;
    if (!req.rgbPixels || !req.depthMetric || !req.mask || W <= 0 || H <= 0) {
        std::cerr << "[DepthExport] invalid request (null/empty input)" << std::endl;
        return res;
    }
    if ((int)req.depthMetric->size() != W * H || (int)req.mask->size() != W * H) {
        std::cerr << "[DepthExport] size mismatch: depth=" << req.depthMetric->size()
                  << " mask=" << req.mask->size() << " expected=" << (W * H) << std::endl;
        return res;
    }
    if (req.K.width > 0 && req.K.width != W) {
        // K resolution != depth resolution -> unprojection scale would be wrong.
        // (For downscaled inputs the caller must scale K to the depth resolution;
        //  not yet handled here. Warn loudly rather than silently misproject.)
        std::cerr << "[DepthExport] WARNING: K res " << req.K.width << "x" << req.K.height
                  << " != depth res " << W << "x" << H
                  << " (OBJ projection may be scaled-mismatched)" << std::endl;
    }

    img::Image rgb;
    rgb.width = W; rgb.height = H; rgb.channels = 3;
    rgb.data.assign(req.rgbPixels, req.rgbPixels + (size_t)W * H * 3);

    objexp::ObjExportOptions base;
    base.projection      = objexp::Projection::PinholeMetric;
    base.intrinsics.fx   = req.K.fx; base.intrinsics.fy = req.K.fy;
    base.intrinsics.cx   = req.K.cx; base.intrinsics.cy = req.K.cy;
    base.depthScale      = 1.0f;
    base.flipY           = true;
    base.zMode           = objexp::ZMode::Metric;
    base.skirtThreshold  = req.skirtThreshold;
    base.textureFilename = "texture.png";
    base.materialName    = "screenMat";

    const std::vector<float>& depth = *req.depthMetric;
    std::vector<uint8_t> maskDil = dilateMaskLocal(*req.mask, W, H, req.maskDilate);

    bool textureDone = false;
    // 1. Canonical full-light (stride) — writes texture.png on the first OBJ.
    {
        objexp::ObjExportOptions o = base;
        o.stride = req.fullMeshStride;
        o.writeTexture = req.writeTextureImage && !textureDone;
        if (objexp::saveFullMeshObj(join(req.outDir, "pc_metric_pinhole_full_light.obj"),
                                    rgb, depth, o)) {
            res.canonicalFullObj = join(req.outDir, "pc_metric_pinhole_full_light.obj");
            textureDone = textureDone || o.writeTexture;
        }
    }
    // 2. Canonical masked (full res, dilated mask).
    {
        objexp::ObjExportOptions o = base;
        o.writeTexture = req.writeTextureImage && !textureDone;
        if (objexp::saveMaskedMeshObj(join(req.outDir, "pc_metric_pinhole_masked.obj"),
                                      rgb, depth, maskDil, o)) {
            res.canonicalMaskedObj = join(req.outDir, "pc_metric_pinhole_masked.obj");
            textureDone = textureDone || o.writeTexture;
        }
    }
    // 3. Tagged debug copies (for cross-K / sam2 parallel-diff).
    if (req.writeTaggedCopies && !req.tag.empty()) {
        const std::string& t = req.tag;
        objexp::ObjExportOptions of = base; of.stride = req.fullMeshStride; of.writeTexture = false;
        const std::string fp = join(req.outDir, "pc_metric_pinhole_full_" + t + "_light.obj");
        if (objexp::saveFullMeshObj(fp, rgb, depth, of)) res.debugCopies.push_back(fp);

        objexp::ObjExportOptions om = base; om.writeTexture = false;
        const std::string mp = join(req.outDir, "pc_metric_pinhole_masked_" + t + ".obj");
        if (objexp::saveMaskedMeshObj(mp, rgb, depth, maskDil, om)) res.debugCopies.push_back(mp);

        if (req.writeNoSkirtVariants) {
            objexp::ObjExportOptions ofn = of; ofn.skirtThreshold = 0.0f;
            const std::string fpn = join(req.outDir, "pc_metric_pinhole_full_" + t + "_light_noskirt.obj");
            if (objexp::saveFullMeshObj(fpn, rgb, depth, ofn)) res.debugCopies.push_back(fpn);

            objexp::ObjExportOptions omn = om; omn.skirtThreshold = 0.0f;
            const std::string mpn = join(req.outDir, "pc_metric_pinhole_masked_" + t + "_noskirt.obj");
            if (objexp::saveMaskedMeshObj(mpn, rgb, depth, maskDil, omn)) res.debugCopies.push_back(mpn);
        }
    }
    // 4. intrinsics.txt (canonical) + intrinsics_<tag>.txt (debug).
    if (saveIntrinsicsFile(join(req.outDir, "intrinsics.txt"), req.K, W, H))
        res.canonicalIntrinsics = join(req.outDir, "intrinsics.txt");
    if (req.writeTaggedCopies && !req.tag.empty())
        saveIntrinsicsFile(join(req.outDir, "intrinsics_" + req.tag + ".txt"), req.K, W, H);

    res.canonicalTexture = join(req.outDir, "texture.png");
    res.ok = !res.canonicalMaskedObj.empty();
    std::cout << "[DepthExport] wrote canonical OBJ + intrinsics.txt (tag=" << req.tag
              << ", " << W << "x" << H << ")" << std::endl;
    return res;
}

} // namespace depthexport
