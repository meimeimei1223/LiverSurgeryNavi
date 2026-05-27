#include "ReconstructFromBin.h"
#include "IntrinsicsScaling.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <vector>
// No stb here: PNG/JPG decoding is delegated to the injected ImageDecoder (the
// stb_image implementation lives in registration/main.cpp's TU only).

namespace ReconstructFromBin {

namespace fs = std::filesystem;

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}
static std::string ext_of(const std::string& path) {
    return lower(fs::path(path).extension().string());
}
static std::string name_of(const std::string& path) {
    return fs::path(path).filename().string();
}

bool isReconstructFile(const std::string& path) {
    const std::string e = ext_of(path);
    const std::string n = name_of(path);
    if (e == ".bin") return true;                       // depth_metric.bin (any .bin)
    if (n == "segmentation_mask.png") return true;       // canonical mask name
    if (n == "original.jpg" || n == "original.jpeg") return true;  // canonical rgb name
    return false;   // generic images keep going to the normal image-load path
}

// ---- slot loaders ----------------------------------------------------------
static void loadBin(State& s, const std::string& path) {
    s.bin = Slot{}; s.bin.filePath = path;
    if (depthexport::loadDepthMetricBin(path, s.depthData) && s.depthData.valid()) {
        s.bin.status = SlotStatus::Loaded;
        s.bin.width  = s.depthData.width;
        s.bin.height = s.depthData.height;
        std::cout << "[Reconstruct] bin loaded: " << s.bin.width << "x" << s.bin.height
                  << "  (" << path << ")" << std::endl;
    } else {
        s.bin.status = SlotStatus::Error;
        s.bin.errorMessage = "Invalid DEPT header / unreadable bin";
        s.depthData = depthexport::DepthBin{};
        std::cerr << "[Reconstruct] bin ERROR: " << s.bin.errorMessage
                  << "  (" << path << ")" << std::endl;
    }
}

static void loadMask(State& s, const std::string& path, const ImageDecoder& decode) {
    s.mask = Slot{}; s.mask.filePath = path;
    int w = 0, h = 0; std::vector<uint8_t> px;
    if (decode && decode(path, 1, px, w, h) && w > 0 && h > 0
        && px.size() >= (size_t)w * h) {
        s.maskData.assign((size_t)w * h, 0);
        for (size_t i = 0; i < s.maskData.size(); ++i) s.maskData[i] = (px[i] > 127) ? 255 : 0;
        s.mask.status = SlotStatus::Loaded;
        s.mask.width = w; s.mask.height = h;
        std::cout << "[Reconstruct] mask loaded: " << w << "x" << h
                  << "  (" << path << ")" << std::endl;
    } else {
        s.mask.status = SlotStatus::Error;
        s.mask.errorMessage = "Failed to decode mask png";
        s.maskData.clear();
        std::cerr << "[Reconstruct] mask ERROR: " << s.mask.errorMessage
                  << "  (" << path << ")" << std::endl;
    }
}

static void loadRgb(State& s, const std::string& path, const ImageDecoder& decode) {
    s.rgb = Slot{}; s.rgb.filePath = path;
    int w = 0, h = 0; std::vector<uint8_t> px;
    if (decode && decode(path, 3, px, w, h) && w > 0 && h > 0
        && px.size() >= (size_t)w * h * 3) {
        s.rgbImage = img::Image(w, h, 3);
        std::copy(px.begin(), px.begin() + (size_t)w * h * 3, s.rgbImage.data.begin());
        s.rgb.status = SlotStatus::Loaded;
        s.rgb.width = w; s.rgb.height = h;
        std::cout << "[Reconstruct] rgb loaded: " << w << "x" << h
                  << "  (" << path << ")" << std::endl;
    } else {
        s.rgb.status = SlotStatus::Error;
        s.rgb.errorMessage = "Failed to decode rgb image";
        s.rgbImage = img::Image{};
        std::cerr << "[Reconstruct] rgb ERROR: " << s.rgb.errorMessage
                  << "  (" << path << ")" << std::endl;
    }
}

void onFileDropped(State& state, const std::string& path, const ImageDecoder& decode) {
    const std::string e = ext_of(path);
    const std::string n = name_of(path);
    if (e == ".bin") {
        loadBin(state, path);
    } else if (n == "segmentation_mask.png") {
        loadMask(state, path, decode);
    } else if (n == "original.jpg" || n == "original.jpeg") {
        loadRgb(state, path, decode);
    } else {
        std::cerr << "[Reconstruct] not a recognized reconstruct file: " << path << std::endl;
    }
    std::cout << "[Reconstruct] ready=" << (state.isReconstructReady() ? 1 : 0)
              << " (bin=" << (int)state.bin.status
              << " mask=" << (int)state.mask.status
              << " rgb=" << (int)state.rgb.status << ")" << std::endl;
}

void onFolderDropped(State& state, const std::string& folderPath, const ImageDecoder& decode) {
    std::error_code ec;
    if (!fs::is_directory(folderPath, ec)) return;
    const std::string bin  = folderPath + "/depth_metric.bin";
    const std::string mask = folderPath + "/segmentation_mask.png";
    const std::string rgb  = folderPath + "/original.jpg";
    if (fs::exists(bin,  ec)) loadBin(state, bin);
    if (fs::exists(mask, ec)) loadMask(state, mask, decode);
    if (fs::exists(rgb,  ec)) loadRgb(state, rgb, decode);
    std::cout << "[Reconstruct] folder scanned: " << folderPath
              << "  ready=" << (state.isReconstructReady() ? 1 : 0) << std::endl;
}

static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

static std::string joinp(const std::string& dir, const std::string& f) {
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) return dir + f;
    return dir + "/" + f;
}

// Files in depth_output/ that Reconstruct overwrites; backed up before + the
// rollback set on failure.
static const char* kManagedFiles[] = {
    "depth_metric.bin", "segmentation_mask.png", "original.jpg", "texture.png",
    "intrinsics.txt", "intrinsics_da3.txt",
    "pc_metric_pinhole_masked.obj",  "pc_metric_pinhole_masked.mtl",
    "pc_metric_pinhole_full_light.obj", "pc_metric_pinhole_full_light.mtl",
};

ReconstructResult execute(State& state,
                          const std::string& depthOutputPath,
                          const Reg3DCustom::CameraIntrinsics& currentK) {
    ReconstructResult r;
    if (!state.isReconstructReady()) {
        r.errorMessage = "Not ready (need bin + mask of matching resolution)";
        std::cerr << "[Reconstruct] " << r.errorMessage << std::endl;
        return r;
    }
    if (!currentK.valid()) {
        r.errorMessage = "No valid camera intrinsics. Set a K source first.";
        std::cerr << "[Reconstruct] " << r.errorMessage << std::endl;
        return r;
    }

    namespace fs = std::filesystem;
    std::error_code ec;

    // 1. Back up the managed files before overwriting.
    std::string base = depthOutputPath;
    while (!base.empty() && (base.back() == '/' || base.back() == '\\')) base.pop_back();
    const std::string backupDir = base + "_backup_" + timestamp();
    fs::create_directories(backupDir, ec);
    std::vector<std::string> backedUp;
    for (const char* f : kManagedFiles) {
        const std::string src = joinp(depthOutputPath, f);
        if (fs::exists(src, ec)) {
            fs::copy_file(src, joinp(backupDir, f),
                          fs::copy_options::overwrite_existing, ec);
            if (!ec) backedUp.push_back(f);
        }
    }
    r.backupPath = backupDir;
    std::cout << "[Reconstruct] backed up " << backedUp.size() << " files -> "
              << backupDir << std::endl;

    auto rollback = [&](const std::string& why) {
        std::cerr << "[Reconstruct] FAILED: " << why << " -- rolling back from "
                  << backupDir << std::endl;
        for (const auto& f : backedUp)
            fs::copy_file(joinp(backupDir, f), joinp(depthOutputPath, f),
                          fs::copy_options::overwrite_existing, ec);
    };

    // 2. Copy the dropped inputs into depth_output/ (skip self-copies).
    auto copyInto = [&](const std::string& srcPath, const std::string& dstName) {
        if (srcPath.empty()) return;
        const std::string dst = joinp(depthOutputPath, dstName);
        fs::path s = fs::weakly_canonical(srcPath, ec), d = fs::weakly_canonical(dst, ec);
        if (!s.empty() && s == d) return;  // already in place
        fs::create_directories(depthOutputPath, ec);
        fs::copy_file(srcPath, dst, fs::copy_options::overwrite_existing, ec);
    };
    copyInto(state.bin.filePath,  "depth_metric.bin");
    copyInto(state.mask.filePath, "segmentation_mask.png");
    if (state.rgb.status == SlotStatus::Loaded) copyInto(state.rgb.filePath, "original.jpg");

    // 3. Scale K to the data (bin) resolution (Phase 3 limitation fix, reused).
    Reg3DCustom::CameraIntrinsics K =
        Reg3DCustom::scaleIntrinsics(currentK, state.bin.width, state.bin.height);
    if (currentK.width != state.bin.width || currentK.height != state.bin.height)
        std::cout << "[Reconstruct] K auto-scaled from " << currentK.width << "x"
                  << currentK.height << " to " << state.bin.width << "x"
                  << state.bin.height << " (fx " << currentK.fx << " -> " << K.fx << ")"
                  << std::endl;

    // 4. RGB for texture: use the dropped rgb, else a white placeholder
    //    (writeTextureImage off so no texture.png is produced).
    const int W = state.bin.width, H = state.bin.height;
    const bool haveRgb = (state.rgb.status == SlotStatus::Loaded
                          && state.rgbImage.width == W && state.rgbImage.height == H);
    std::vector<uint8_t> whiteRgb;
    const uint8_t* rgbPx = nullptr;
    if (haveRgb) {
        rgbPx = state.rgbImage.data.data();
    } else {
        whiteRgb.assign((size_t)W * H * 3, 255);
        rgbPx = whiteRgb.data();
        std::cout << "[Reconstruct] no matching rgb; using white texture placeholder"
                  << std::endl;
    }

    // 5. Generate canonical OBJ + intrinsics.txt via the shared exporter.
    depthexport::Request req;
    req.rgbPixels        = rgbPx;
    req.width            = W;
    req.height           = H;
    req.depthMetric      = &state.depthData.depth;
    req.mask             = &state.maskData;
    req.K                = K;
    req.tag              = K.name.empty() ? "custom" : K.name;
    req.outDir           = depthOutputPath;
    req.writeTextureImage = haveRgb;
    auto er = depthexport::exportDepthArtifacts(req);
    if (!er.ok) {
        rollback("exportDepthArtifacts failed");
        r.errorMessage = "OBJ export failed";
        return r;
    }

    r.ok = true;
    std::cout << "[Reconstruct] OK: canonical OBJ + intrinsics.txt written ("
              << W << "x" << H << ", tag=" << req.tag << ")" << std::endl;
    return r;
}

} // namespace ReconstructFromBin
