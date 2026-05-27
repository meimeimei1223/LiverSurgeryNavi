#include "ReconstructFromBin.h"
#include "IntrinsicsScaling.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cctype>
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

// execute(): implemented in Task 6. Stub keeps the build green until then.
ReconstructResult execute(State& /*state*/,
                          const std::string& /*depthOutputPath*/,
                          const Reg3DCustom::CameraIntrinsics& /*currentK*/) {
    ReconstructResult r;
    r.ok = false;
    r.errorMessage = "Reconstruct execute() not yet implemented (Task 6)";
    std::cerr << "[Reconstruct] " << r.errorMessage << std::endl;
    return r;
}

} // namespace ReconstructFromBin
