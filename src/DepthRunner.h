#pragma once
/**
 * @file DepthRunner.h
 * @brief medsam2_da3_lite 外部実行ラッパー
 *
 * 実行ファイル: build/Desktop-Debug/bin/AAA_Reg3D_ONNX
 * ../../../ で AAA_Reg3D_ONNX プロジェクトルートに到達
 */

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <chrono>
#include <filesystem>
#include <functional>
#include "PlatformCompat.h"
// =============================================================================
// 設定
// =============================================================================
struct DepthRunnerConfig {
    // build/Desktop-Debug/bin/ から ../../../ で AAA_Reg3D_ONNX ルート
    std::string exePath    = "../../../medsam2_da3_lite/build/medsam2_da3_lite";
    std::string depthModel = "../../../medsam2_da3_lite/models/depth_anything_v3_small.onnx";
    std::string samEncoder = "../../../medsam2_da3_lite/models/sam2_hiera_tiny.encoder.onnx";
    std::string samDecoder = "../../../medsam2_da3_lite/models/sam2_hiera_tiny.decoder.onnx";
    std::string outputDir  = "../../../depth_output";
    bool useCuda  = false;
    bool verbose  = true;
};

struct DepthRunnerPoint {
    float x, y;
    bool isForeground;
    DepthRunnerPoint(float x_, float y_, bool fg = true) : x(x_), y(y_), isForeground(fg) {}
};

struct DepthRunnerResult {
    bool success = false;
    int exitCode = -1;
    double elapsedMs = 0.0;

    std::string originalPath;
    std::string segmentationMaskPath;
    std::string segmentationOverlayPath;
    std::string depthFullPath;
    std::string depthFullColoredPath;
    std::string depthMaskedPath;
    std::string depthMaskedColoredPath;
    std::string depthMaskedRenormPath;
    std::string depthMaskedRenormColoredPath;

    bool hasOriginal() const         { return fileExists(originalPath); }
    bool hasDepthFull() const        { return fileExists(depthFullPath); }
    bool hasDepthMasked() const      { return fileExists(depthMaskedPath); }
    bool hasDepthMaskedRenorm() const { return fileExists(depthMaskedRenormPath); }
    bool hasSegmentationMask() const { return fileExists(segmentationMaskPath); }

private:
    static bool fileExists(const std::string& p) {
        if (p.empty()) return false;
        struct stat buf;
        return (stat(p.c_str(), &buf) == 0);
    }
};

// =============================================================================
// DepthRunner
// =============================================================================
class DepthRunner {
public:
    DepthRunnerConfig config;

    DepthRunner() = default;
    explicit DepthRunner(const DepthRunnerConfig& cfg) : config(cfg) {}

    bool isAvailable() const {
        struct stat buf;
        return (stat(config.exePath.c_str(), &buf) == 0);
    }

    bool areModelsAvailable() const {
        struct stat buf;
        bool ok = true;
        if (stat(config.depthModel.c_str(), &buf) != 0) {
            std::cerr << "[DepthRunner] Not found: " << config.depthModel << std::endl;
            ok = false;
        }
        if (stat(config.samEncoder.c_str(), &buf) != 0) {
            std::cerr << "[DepthRunner] Not found: " << config.samEncoder << std::endl;
            ok = false;
        }
        if (stat(config.samDecoder.c_str(), &buf) != 0) {
            std::cerr << "[DepthRunner] Not found: " << config.samDecoder << std::endl;
            ok = false;
        }
        return ok;
    }

    DepthRunnerResult run(
        const std::string& imagePath,
        const std::vector<DepthRunnerPoint>& points = {},
        std::function<void(float, const char*)> progressCb = nullptr
        ) {
        DepthRunnerResult result;

        if (!isAvailable()) {
            std::cerr << "[DepthRunner] Executable not found: " << config.exePath << std::endl;
            std::cerr << "[DepthRunner] Build: cd medsam2_da3_lite && mkdir build && cd build && cmake .. && make" << std::endl;
            return result;
        }

        ensureDir(config.outputDir);
        std::string cmd = buildCmd(imagePath, points);

        if (config.verbose)
            std::cout << "\n[DepthRunner] " << cmd << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();

        std::string pipeCmd = cmd + " 2>&1";
        FILE* pipe = PLATFORM_POPEN(pipeCmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "[DepthRunner] popen failed" << std::endl;
            return result;
        }

        char lineBuf[512];
        while (fgets(lineBuf, sizeof(lineBuf), pipe)) {
            std::string line(lineBuf);
            if (config.verbose) std::cout << line;

            if (progressCb) {
                if (line.find("Loading image") != std::string::npos)
                    progressCb(0.10f, "Loading image...");
                else if (line.find("Step 1") != std::string::npos)
                    progressCb(0.15f, "SAM2 Segmentation...");
                else if (line.find("Encoding image") != std::string::npos)
                    progressCb(0.20f, "SAM2: Encoding...");
                else if (line.find("Running segmentation") != std::string::npos)
                    progressCb(0.35f, "SAM2: Decoding...");
                else if (line.find("Total segmentation") != std::string::npos)
                    progressCb(0.40f, "Segmentation done");
                else if (line.find("Step 2") != std::string::npos)
                    progressCb(0.45f, "Depth Anything V3...");
                else if (line.find("Preprocessing") != std::string::npos)
                    progressCb(0.48f, "Depth: Preprocessing...");
                else if (line.find("Inference") != std::string::npos)
                    progressCb(0.50f, "Depth: Inference...");
                else if (line.find("Depth estimation total") != std::string::npos)
                    progressCb(0.80f, "Depth inference done");
                else if (line.find("Step 3") != std::string::npos)
                    progressCb(0.85f, "Extracting masked depth...");
                else if (line.find("Step 4") != std::string::npos)
                    progressCb(0.90f, "Saving results...");
                else if (line.find("Done!") != std::string::npos)
                    progressCb(0.95f, "External process done");
            }
        }

        int ret = PLATFORM_PCLOSE(pipe);
        auto t1 = std::chrono::high_resolution_clock::now();

#ifdef _WIN32
        result.exitCode = ret;
#else
        result.exitCode = WEXITSTATUS(ret);
#endif
        result.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.success = (result.exitCode == 0);

        if (result.success) {
            std::string d = config.outputDir;
            result.originalPath              = d + "/original.jpg";
            result.segmentationMaskPath      = d + "/segmentation_mask.png";
            result.segmentationOverlayPath   = d + "/segmentation_overlay.jpg";
            result.depthFullPath             = d + "/depth_full.png";
            result.depthFullColoredPath      = d + "/depth_full_colored.png";
            result.depthMaskedPath           = d + "/depth_masked.png";
            result.depthMaskedColoredPath    = d + "/depth_masked_colored.png";
            result.depthMaskedRenormPath     = d + "/depth_masked_renorm.png";
            result.depthMaskedRenormColoredPath = d + "/depth_masked_renorm_colored.png";

            if (config.verbose)
                std::cout << "[DepthRunner] Done (" << result.elapsedMs << " ms)" << std::endl;
        } else {
            std::cerr << "[DepthRunner] Failed (exit " << result.exitCode << ")" << std::endl;
        }
        return result;
    }

    DepthRunnerResult runDepthOnly(const std::string& imagePath) {
        return run(imagePath, {});
    }

    DepthRunnerResult runWithPoint(const std::string& imagePath, float px, float py) {
        return run(imagePath, { DepthRunnerPoint(px, py, true) });
    }

    void printDiagnostics() const {
        std::cout << "\n=== DepthRunner ===" << std::endl;
        std::cout << "  Exe:    " << config.exePath    << (isAvailable() ? " [OK]" : " [NOT FOUND]") << std::endl;
        std::cout << "  Depth:  " << config.depthModel << std::endl;
        std::cout << "  SAMenc: " << config.samEncoder << std::endl;
        std::cout << "  SAMdec: " << config.samDecoder << std::endl;
        std::cout << "  Output: " << config.outputDir  << std::endl;
        std::cout << "==================\n" << std::endl;
    }

private:
    std::string buildCmd(const std::string& img, const std::vector<DepthRunnerPoint>& pts) const {
        // Windows の cmd.exe は "./" を理解できないので絶対パスに変換
        std::string exe = config.exePath;
        try {
            exe = std::filesystem::absolute(config.exePath).string();
        } catch (...) {
            // 変換失敗時はそのまま使用
        }

        // モデルパスも絶対パスに変換（Windowsで相対パスが壊れるため）
        auto absPath = [](const std::string& p) -> std::string {
            std::string r;
            try { r = std::filesystem::absolute(p).string(); }
            catch (...) { r = p; }
            while (!r.empty() && (r.back() == '/' || r.back() == '\\'))
                r.pop_back();
            return r;
        };

        std::ostringstream s;
        s << "\"" << exe << "\" \"" << img << "\""
          << " --depth-model \"" << absPath(config.depthModel) << "\""
          << " --sam-encoder \"" << absPath(config.samEncoder) << "\""
          << " --sam-decoder \"" << absPath(config.samDecoder) << "\""
          << " --output \""      << absPath(config.outputDir)  << "\"";
        if (config.useCuda) s << " --cuda";
        for (auto& p : pts) {
            s << (p.isForeground ? " --point " : " --bg-point ")
            << (int)p.x << "," << (int)p.y;
        }

#ifdef _WIN32
        // Windows cmd.exe: 内部にクォートがある場合、全体をクォートで囲む必要がある
        return "\"" + s.str() + "\"";
#else
        return s.str();
#endif
    }

    static void ensureDir(const std::string& p) {
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
    }
};

// =============================================================================
// 統合ヘルパー
// =============================================================================
namespace DepthRunnerIntegration {

template<typename MeshType, typename SetupFunc>
bool updateScreenMeshDepth(
    DepthRunner& runner,
    const std::string& imagePath,
    MeshType* screenMesh,
    int gridWidth, float scale, float depthScale,
    const std::vector<DepthRunnerPoint>& points,
    SetupFunc setupFunc,
    std::function<void(float, const char*)> progressCb = nullptr
    ) {
    if (!screenMesh) return false;

    auto result = runner.run(imagePath, points, progressCb);
    if (!result.success || !result.hasOriginal()) return false;

    if (progressCb) progressCb(0.96f, "Loading texture...");
    screenMesh->loadTextureFromFile(result.originalPath.c_str());

    std::string depthPath;
    if      (result.hasDepthMaskedRenorm()) depthPath = result.depthMaskedRenormPath;
    else if (result.hasDepthMasked())       depthPath = result.depthMaskedPath;
    else if (result.hasDepthFull())         depthPath = result.depthFullPath;
    else return false;

    std::cout << "[DepthRunner] Using: " << depthPath << std::endl;

    if (progressCb) progressCb(0.97f, "Generating depth mesh...");
    screenMesh->loadDepthImage(depthPath, screenMesh->loadedImageWidth, screenMesh->loadedImageHeight);

    int gh = gridWidth * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
    auto depths = screenMesh->calculateNormalizedDepth(gridWidth, gh, 0.99f, 1.0f);
    screenMesh->generateGridPlaneWithDepth(gridWidth, gh, depths, 0.05f, depthScale);

    for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
        screenMesh->mVertices[i] *= scale;

    setupFunc(*screenMesh);
    return true;
}

} // namespace DepthRunnerIntegration

// =============================================================================
// Depth-only (uses depth_full.png, ignores segmentation mask)
// =============================================================================
namespace DepthRunnerIntegration {

template<typename MeshType, typename SetupFunc>
bool updateScreenMeshDepthFullOnly(
    DepthRunner& runner,
    const std::string& imagePath,
    MeshType* screenMesh,
    int gridWidth, float scale, float depthScale,
    const std::vector<DepthRunnerPoint>& dummyPoints,
    SetupFunc setupFunc,
    std::function<void(float, const char*)> progressCb = nullptr
    ) {
    if (!screenMesh) return false;

    auto result = runner.run(imagePath, dummyPoints, progressCb);
    if (!result.success || !result.hasOriginal()) return false;

    if (progressCb) progressCb(0.96f, "Loading texture...");
    screenMesh->loadTextureFromFile(result.originalPath.c_str());

    if (!result.hasDepthFull()) {
        std::cerr << "[DepthRunner] depth_full.png not found" << std::endl;
        return false;
    }
    std::string depthPath = result.depthFullPath;
    std::cout << "[DepthRunner] Using (NO SEG): " << depthPath << std::endl;

    if (progressCb) progressCb(0.97f, "Generating depth mesh...");
    screenMesh->loadDepthImage(depthPath, screenMesh->loadedImageWidth, screenMesh->loadedImageHeight);

    int gh = gridWidth * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
    auto depths = screenMesh->calculateNormalizedDepth(gridWidth, gh, 0.99f, 0.9f);
    screenMesh->generateGridPlaneWithDepth(gridWidth, gh, depths, 0.05f, depthScale);

    for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
        screenMesh->mVertices[i] *= scale;

    setupFunc(*screenMesh);
    return true;
}

} // namespace DepthRunnerIntegration
