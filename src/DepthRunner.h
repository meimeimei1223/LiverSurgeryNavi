#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>          // std::setprecision for distortion round-trip
#include <cstdlib>
#include <cstdio>
#include <cmath>            // std::fabs for distortion eps-check
#include <sys/stat.h>
#include <chrono>
#include <filesystem>
#include <functional>
#include "PlatformCompat.h"

// Pipeline stage selector for the external medsam2_da3_lite executable.
//   All     : SAM2 + DA3 in one shot (default, legacy behaviour).
//   Segment : SAM2 only -- writes segmentation_mask.png and exits.
//             Lets the UI show the mask to the user before paying the
//             depth-inference cost.
//   Depth   : Skip SAM2; reuse the existing segmentation_mask.png from
//             the output dir, then run DA3 + OBJ export.
enum class DepthStage { All, Segment, Depth };

struct DepthRunnerConfig {
    std::string exePath;
    std::string depthModel;
    std::string samEncoder;
    std::string samDecoder;
    std::string outputDir;
    // CUDA / GPU acceleration toggle. Appended as --cuda to the CLI.
    // Requires medsam2_da3_lite built with USE_CUDA=ON (CMakeLists picks
    // the GPU ORT variant in that case). When OFF, the pipeline runs on
    // CPU. When ON but the binary is CPU-only, the in-pipeline fallback
    // logs "CUDA not available, using CPU" and continues without crashing.
    //
    // Speedup observed on RTX-class hardware (1920x1080 input):
    //   SAM2 encoder      1820 ms -> ~100 ms   (~18x)
    //   DA3 inference     6850 ms -> ~500 ms   (~14x)
    //   Total Run Depth   ~17 s   -> ~3-5 s    (~4-5x)
    bool useCuda  = false;
    bool verbose  = true;

    // Optional intrinsics override. If set, --kinect-intrinsics fx,fy,cx,cy
    // will be appended so the external pipeline back-projects depth using
    // the user's K instead of its hard-coded Azure Kinect 720p default.
    // The OBJ outputs (pc_metric_pinhole_*_<tag>*.obj) and intrinsics_<tag>.txt
    // will then carry these values, where <tag> = intrinsicsSourceName.
    bool  useCustomIntrinsics = false;
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;

    // ---- Brown-Conrady distortion (OpenCV convention) -----------------
    // Used only when useCustomIntrinsics is true AND at least one of these
    // coefficients is non-zero (eps=1e-6). The external pipeline does NOT
    // apply these to its own processing; it only writes them back into
    // intrinsics_<tag>.txt so the registration app can read them and call
    // Undistort.h on the input image. Default 0 = no distortion (legacy
    // behaviour, same command line as before this field existed).
    //
    // Closes the bug where Run Depth used to silently truncate k1..p2 from
    // a user-edited intrinsics_<tag>.txt on every invocation.
    float k1 = 0.0f, k2 = 0.0f, k3 = 0.0f, k4 = 0.0f;
    float p1 = 0.0f, p2 = 0.0f;

    // Tag used as a suffix on output filenames produced by the depth pipeline.
    // Pass "k4a" (default), "custom", "calib", or any short ASCII label.
    // The corresponding outputs are intrinsics_<name>.txt and
    // pc_metric_pinhole_*_<name>*.obj. Caller-side OBJ loading must use the
    // same name to find the right file.
    std::string intrinsicsSourceName = "k4a";

    // Which pipeline stage to run. Default = All preserves legacy behaviour.
    // Caller flow for split mode:
    //   1. cfg.stage = DepthStage::Segment;  runner.run(...)
    //      -> displays segmentation_mask.png to the user.
    //   2. (user inspects / accepts the mask)
    //   3. cfg.stage = DepthStage::Depth;    runner.run(...)
    //      -> reuses the mask, produces depth + OBJ.
    DepthStage stage = DepthStage::All;

    // ---- Mask output filename selector (passed as --mask-output-name) ----
    // ""             -> liver outputs (segmentation_mask.png + overlay).
    //                   Default; matches legacy behaviour.
    // "instrument"   -> Stage=Segment shortcut: treat the points argument
    //                   to run() as instrument prompts and write only
    //                   instrument_segmentation_mask.png + matching overlay.
    //                   Used by the UI's Instrument preview button so the
    //                   liver mask on disk is not overwritten.
    std::string maskOutputName;

    // ---- Vignette auto-detection toggle (passed as --no-vignette-detect) ----
    // true  (default) -> The external pipeline auto-detects the black FOV
    //                    vignette and OR-merges it into the occluder mask
    //                    (instrument_segmentation_mask.png). Matches the
    //                    standard production behaviour.
    // false           -> Pass --no-vignette-detect to disable. The saved
    //                    occluder mask contains ONLY the SAM2 instrument
    //                    output (or no file at all if no instrument
    //                    prompts were given). Useful for A/B comparison
    //                    against pre-vignette-feature output.
    bool detectVignette = true;
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
    // Instrument mask outputs (only populated when --instrument-* points
    // were passed to run() OR when maskOutputName == "instrument").
    // Caller uses hasInstrumentMask() to test presence on disk.
    std::string instrumentSegmentationMaskPath;
    std::string instrumentSegmentationOverlayPath;
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
    bool hasInstrumentMask() const   { return fileExists(instrumentSegmentationMaskPath); }
    bool hasInstrumentOverlay() const{ return fileExists(instrumentSegmentationOverlayPath); }

private:
    static bool fileExists(const std::string& p) {
        if (p.empty()) return false;
        struct stat buf;
        return (stat(p.c_str(), &buf) == 0);
    }
};

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
        std::function<void(float, const char*)> progressCb = nullptr,
        // Optional instrument prompts. When non-empty AND maskOutputName
        // is not "instrument", the external pipeline runs a 2nd SAM2 pass
        // and writes instrument_segmentation_mask.png alongside the liver
        // mask. Empty vector preserves the original single-mask behaviour.
        const std::vector<DepthRunnerPoint>& instrumentPoints = {}
        ) {
        DepthRunnerResult result;

        if (!isAvailable()) {
            std::cerr << "[DepthRunner] Executable not found: " << config.exePath << std::endl;
            std::cerr << "[DepthRunner] Build: cd medsam2_da3_lite && mkdir build && cd build && cmake .. && make" << std::endl;
            return result;
        }

        ensureDir(config.outputDir);
        std::string cmd = buildCmd(imagePath, points, instrumentPoints);

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
            // Instrument outputs (filled by the external pipeline only when
            // instrument prompts are supplied or maskOutputName=="instrument";
            // hasInstrumentMask() / hasInstrumentOverlay() check disk so
            // setting the paths unconditionally here is safe).
            result.instrumentSegmentationMaskPath    =
                d + "/instrument_segmentation_mask.png";
            result.instrumentSegmentationOverlayPath =
                d + "/instrument_segmentation_overlay.jpg";
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
    std::string buildCmd(
        const std::string& img,
        const std::vector<DepthRunnerPoint>& pts,
        const std::vector<DepthRunnerPoint>& instPts = {}
        ) const
    {

        std::string exe = config.exePath;
        try {
            exe = std::filesystem::absolute(config.exePath).string();
        } catch (...) {

        }

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

        // Pass user-provided intrinsics to the external pipeline. This makes
        // the *_<tag>* OBJs and intrinsics_<tag>.txt use the same K the C++
        // side is rendering with -- otherwise the mesh is back-projected with
        // the wrong focal length and AR alignment fails.
        if (config.useCustomIntrinsics &&
            config.fx > 0.0f && config.fy > 0.0f) {
            s << " --kinect-intrinsics "
              << config.fx << "," << config.fy << ","
              << config.cx << "," << config.cy;

            // Also pass distortion coefficients when at least one is
            // non-zero, so the external pipeline can round-trip them into
            // intrinsics_<tag>.txt. Empty (all-zero) case is suppressed
            // to keep the command line byte-identical to legacy invocations
            // for cameras without a distortion model.
            const float kEps = 1e-6f;
            const bool hasDist =
                std::fabs(config.k1) > kEps || std::fabs(config.k2) > kEps ||
                std::fabs(config.k3) > kEps || std::fabs(config.k4) > kEps ||
                std::fabs(config.p1) > kEps || std::fabs(config.p2) > kEps;
            if (hasDist) {
                // 9-digit precision: enough to round-trip an IEEE-754 single
                // through text (numeric_limits<float>::max_digits10). The
                // default 6 digits would lose the last sig-fig of small
                // coefficients like p1 ~ 1e-3. Restored to 6 immediately
                // below since this stream may still emit more args (points).
                s << " --kinect-distortion "
                  << std::setprecision(9)
                  << config.k1 << "," << config.k2 << ","
                  << config.k3 << "," << config.k4 << ","
                  << config.p1 << "," << config.p2
                  << std::setprecision(6);
            }
        }
        // Tag the output files so the source is traceable in the filename.
        // Default "k4a" preserves backwards-compat output names when the
        // caller hasn't explicitly set this.
        if (!config.intrinsicsSourceName.empty()) {
            s << " --intrinsics-source " << config.intrinsicsSourceName;
        }

        // Pipeline stage flag. Omitted when stage=All so legacy invocations
        // remain bit-identical (no --stage in the command line).
        switch (config.stage) {
        case DepthStage::Segment: s << " --stage segment"; break;
        case DepthStage::Depth:   s << " --stage depth";   break;
        case DepthStage::All:     break;
        }

        // Mask output filename selector. Omitted (= legacy "liver" output
        // names) unless the caller is doing an Instrument-only preview.
        if (!config.maskOutputName.empty()) {
            s << " --mask-output-name " << config.maskOutputName;
        }

        // Vignette auto-detection toggle. Omitted when on (= default
        // production behaviour); passed as a negative flag when off so
        // the external pipeline knows to skip merging the auto-detected
        // FOV vignette into the occluder mask.
        if (!config.detectVignette) {
            s << " --no-vignette-detect";
        }

        for (auto& p : pts) {
            s << (p.isForeground ? " --point " : " --bg-point ")
            << (int)p.x << "," << (int)p.y;
        }
        // Instrument prompts (separate flags so the external pipeline can
        // run a second SAM2 pass without the prompts colliding with the
        // liver ones).
        for (auto& p : instPts) {
            s << (p.isForeground ? " --instrument-point "
                                 : " --instrument-bg-point ")
              << (int)p.x << "," << (int)p.y;
        }

#ifdef _WIN32

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
