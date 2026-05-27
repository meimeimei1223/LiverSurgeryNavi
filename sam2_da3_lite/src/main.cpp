#include "depth_anything_v3.hpp"
#include "sam2_segmentor.hpp"
#include "image_utils.hpp"
#include "VignetteDetection.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>          // std::setprecision for intrinsics_<tag>.txt
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>            // std::fabs for distortion eps-check
#include <cstdint>

// Pipeline stage selector.
//   All     : run SAM2 + DA3 in one shot (legacy behaviour, default)
//   Segment : run SAM2 only, write segmentation_mask.png, then exit.
//             Lets the caller inspect / accept the mask before paying
//             the depth-inference cost.
//   Depth   : skip SAM2, load existing segmentation_mask.png from the
//             output dir, then run DA3 + downstream OBJ export.
enum class Stage { All, Segment, Depth };

struct Options {
    std::string imagePath;
    std::string depthModelPath = "models/depth_anything_v3_small.onnx";
    std::string samEncoderPath = "models/sam2_hiera_tiny.encoder.onnx";
    std::string samDecoderPath = "models/sam2_hiera_tiny.decoder.onnx";
    std::string outputDir      = "output";
    std::vector<sam2::PointPrompt> points;
    bool  useCuda        = false;
    bool  showHelp       = false;
    float metricScale    = 1.0f;


    // ---- Vignette auto-detection ----
    // When true (default), every code path that writes the occluder mask
    // (instrument_segmentation_mask.png) merges in the auto-detected black
    // FOV vignette via VignetteDetection::detect(). Set false via
    // --no-vignette-detect to bypass for debugging or to preserve legacy
    // byte-for-byte output.
    bool detectVignette = true;

    // Which pipeline stage to run. Default = All preserves legacy behaviour.
    Stage stage = Stage::All;

    // ---- Instrument mask (secondary SAM2 pass) ----
    // When non-empty, the SAM2 stage runs a SECOND prediction with these
    // prompts and writes the result to instrument_segmentation_mask.png
    // (and the matching overlay). Used downstream for boundary rejection
    // around tools / instruments occluding the liver.
    std::vector<sam2::PointPrompt> instrumentPoints;

    // ---- Mask output filename selector ----
    // ""             -> produce both segmentation_mask.png (liver, from
    //                   `points`) and, if `instrumentPoints` is non-empty,
    //                   instrument_segmentation_mask.png. Default behaviour.
    // "instrument"   -> Stage=Segment shortcut: treat `points` as the
    //                   instrument prompts and write only
    //                   instrument_segmentation_mask.png + overlay. Used
    //                   by the UI when the user clicks the Instrument
    //                   preview button so we don't overwrite the liver
    //                   mask just to look at the tool segmentation.
    std::string maskOutputName;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <image_path> [options]\n\n"
              << "Options:\n"
              << "  --depth-model <path>       Depth Anything V3 ONNX model path\n"
              << "  --sam-encoder <path>       SAM2 encoder ONNX model path\n"
              << "  --sam-decoder <path>       SAM2 decoder ONNX model path\n"
              << "  --output <dir>             Output directory\n"
              << "  --point <x,y>              Foreground point (can specify multiple)\n"
              << "  --bg-point <x,y>           Background point (can specify multiple)\n"
              << "  --instrument-point <x,y>     Foreground prompt for the secondary instrument mask\n"
              << "  --instrument-bg-point <x,y>  Background prompt for the secondary instrument mask\n"
              << "                               (when any --instrument-* flag is given, a 2nd SAM2\n"
              << "                                pass writes instrument_segmentation_mask.png)\n"
              << "  --mask-output-name <name>    'instrument' = treat --point/--bg-point as instrument\n"
              << "                                prompts and write only instrument_segmentation_mask.png\n"
              << "                                (used by the UI's Instrument preview button so the\n"
              << "                                liver mask is not overwritten).\n"
              << "  --no-vignette-detect       Disable auto-detection of the black FOV vignette\n"
              << "                             (otherwise it is OR-merged into the occluder mask).\n"
              << "  --cuda                     Use CUDA for inference\n"
              << "  --scale <f>                Depth scale factor (default 1.0)\n"
              << "  --stage <all|segment|depth>  Pipeline stage to run (default: all).\n"
              << "                               segment: SAM2 only, writes segmentation_mask.png and exits.\n"
              << "                               depth  : skip SAM2, load existing segmentation_mask.png\n"
              << "                                        from --output dir, then run DA3 + OBJ export.\n"
              << "  --help                     Show this help\n";
}

Options parseArgs(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.showHelp = true;
        } else if (arg == "--depth-model" && i + 1 < argc) {
            opts.depthModelPath = argv[++i];
        } else if (arg == "--sam-encoder" && i + 1 < argc) {
            opts.samEncoderPath = argv[++i];
        } else if (arg == "--sam-decoder" && i + 1 < argc) {
            opts.samDecoderPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts.outputDir = argv[++i];
        } else if (arg == "--point" && i + 1 < argc) {
            std::string coords = argv[++i];
            size_t comma = coords.find(',');
            if (comma != std::string::npos) {
                float x = std::stof(coords.substr(0, comma));
                float y = std::stof(coords.substr(comma + 1));
                opts.points.emplace_back(x, y, sam2::PointLabel::Foreground);
            }
        } else if (arg == "--bg-point" && i + 1 < argc) {
            std::string coords = argv[++i];
            size_t comma = coords.find(',');
            if (comma != std::string::npos) {
                float x = std::stof(coords.substr(0, comma));
                float y = std::stof(coords.substr(comma + 1));
                opts.points.emplace_back(x, y, sam2::PointLabel::Background);
            }
        } else if (arg == "--instrument-point" && i + 1 < argc) {
            std::string coords = argv[++i];
            size_t comma = coords.find(',');
            if (comma != std::string::npos) {
                float x = std::stof(coords.substr(0, comma));
                float y = std::stof(coords.substr(comma + 1));
                opts.instrumentPoints.emplace_back(
                    x, y, sam2::PointLabel::Foreground);
            }
        } else if (arg == "--instrument-bg-point" && i + 1 < argc) {
            std::string coords = argv[++i];
            size_t comma = coords.find(',');
            if (comma != std::string::npos) {
                float x = std::stof(coords.substr(0, comma));
                float y = std::stof(coords.substr(comma + 1));
                opts.instrumentPoints.emplace_back(
                    x, y, sam2::PointLabel::Background);
            }
        } else if (arg == "--mask-output-name" && i + 1 < argc) {
            opts.maskOutputName = argv[++i];
        } else if (arg == "--cuda") {
            opts.useCuda = true;
        } else if (arg == "--scale" && i + 1 < argc) {
            opts.metricScale = std::stof(argv[++i]);
        } else if (arg == "--no-vignette-detect") {
            opts.detectVignette = false;
        } else if (arg == "--stage" && i + 1 < argc) {
            std::string s = argv[++i];
            if      (s == "all")     opts.stage = Stage::All;
            else if (s == "segment") opts.stage = Stage::Segment;
            else if (s == "depth")   opts.stage = Stage::Depth;
            else {
                std::cerr << "[warn] --stage expects all|segment|depth, got '"
                          << s << "'; defaulting to all\n";
                opts.stage = Stage::All;
            }
        } else if (arg[0] != '-' && opts.imagePath.empty()) {
            opts.imagePath = arg;
        }
    }
    return opts;
}

static bool writeIntrinsicsTxt(const std::string& path,
                               const depth::DepthResult& r)
{
    if (!r.hasIntrinsics) return false;
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[intrinsics] Failed to open: " << path << std::endl;
        return false;
    }
    // DA3-estimated K (Phase 3): 7 lines, no distortion (DA3 does not estimate it).
    // name="da3" so REG reads this as the IntrinsicsSource::DA3 source.
    ofs << std::setprecision(9);
    ofs << "fx "     << r.intrinsics.fx << "\n";
    ofs << "fy "     << r.intrinsics.fy << "\n";
    ofs << "cx "     << r.intrinsics.cx << "\n";
    ofs << "cy "     << r.intrinsics.cy << "\n";
    ofs << "width "  << r.width  << "\n";
    ofs << "height " << r.height << "\n";
    ofs << "name   da3\n";
    std::cout << "[intrinsics_da3] Saved (DA3 estimated K): " << path << std::endl;
    return true;
}

// =============================================================================
//  Occluder mask writer (instrument SAM2 mask + auto-detected vignette)
// -----------------------------------------------------------------------------
//  Centralises the writing of instrument_segmentation_mask.png so every path
//  applies the same merge logic. The on-disk filename is preserved for
//  binary compat with the registration pipeline -- g_instrumentDistMap in
//  DepthUtils.h reads this exact path so no downstream change is needed.
//
//  When BOTH the SAM2 instrument mask and the auto-detected vignette have
//  content, the function additionally drops debug-only single-source files
//  (instrument_only_mask.png, vignette_only_mask.png) so an operator can
//  diff and see WHERE the merged mask came from.
//
//  Returns true iff any mask file was written.
// =============================================================================
static bool writeOccluderMaskMerged(const img::Image& image,
                                    const std::vector<uint8_t>& instMask,
                                    const std::string& outDir,
                                    bool detectVignette)
{
    const size_t N = (size_t)image.width * image.height;
    const bool hasInst = (instMask.size() == N);

    // Run vignette detection (when enabled). Pure function over the RGB
    // buffer; cheap (~25-50 ms at 1920x1080).
    VignetteDetection::Result vig;
    if (detectVignette) {
        vig = VignetteDetection::detect(image.data.data(),
                                        image.width, image.height);
    }

    // No SAM2 instrument prompts AND no vignette -> nothing to write.
    // Preserves the pre-feature behaviour ("no instrument prompts -> no
    // instrument file") when vignette detection is disabled or fails.
    if (!hasInst && !vig.valid) {
        std::cout << "[OccluderMask] skipped: no instrument prompts and"
                     " no vignette detected\n";
        return false;
    }

    // OR the two sources into a single occluder mask.
    std::vector<uint8_t> merged(N, 0);
    if (hasInst) {
        for (size_t i = 0; i < N; ++i)
            if (instMask[i] > 127) merged[i] = 255;
    }
    if (vig.valid) {
        for (size_t i = 0; i < N; ++i)
            if (vig.mask[i] > 127) merged[i] = 255;
    }

    int mergedCount = 0;
    for (auto v : merged) if (v) ++mergedCount;

    // Save the merged mask under the legacy filename. Overlay uses cyan
    // (50,220,255) to match the Instrument UI colour.
    img::saveGrayscale(outDir + "/instrument_segmentation_mask.png",
                       merged, image.width, image.height);
    img::Image overlay = img::overlayMask(image, merged,
                                          50, 220, 255, 0.5f);
    img::saveImage(outDir + "/instrument_segmentation_overlay.jpg", overlay);
    std::cout << "[OccluderMask] merged=" << mergedCount << " px"
              << "  (inst=" << (hasInst ? "yes" : "no")
              << ", vignette=" << (vig.valid ? "yes" : "no") << ")\n"
              << "[OccluderMask] saved: " << outDir
              << "/instrument_segmentation_{mask.png,overlay.jpg}\n";

    // When both sources contributed, also drop per-source debug files so
    // operators can see WHERE the merged mask came from.
    if (hasInst && vig.valid) {
        img::saveGrayscale(outDir + "/instrument_only_mask.png",
                           instMask, image.width, image.height);
        img::saveGrayscale(outDir + "/vignette_only_mask.png",
                           vig.mask, image.width, image.height);
        std::cout << "[OccluderMask] debug: " << outDir
                  << "/{instrument_only_mask.png,vignette_only_mask.png}\n";
    }
    return true;
}

int main(int argc, char* argv[]) {

    std::cout << "======================================================================\n"
              << "Depth Anything V3 + SAM2 C++ ONNX Lite (No OpenCV)\n"
              << "======================================================================\n\n";

    Options opts = parseArgs(argc, argv);

    if (opts.showHelp || opts.imagePath.empty()) {
        printUsage(argv[0]);
        return opts.showHelp ? 0 : 1;
    }

    std::cout << "Loading image: " << opts.imagePath << "\n";
    img::Image image = img::loadImage(opts.imagePath);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image: " << opts.imagePath << "\n";
        return 1;
    }
    std::cout << "Image size: " << image.width << "x" << image.height << "\n";

    // --- Resize policy: cap at 1920x1080, preserve aspect ---
    //
    // Images within the 1920x1080 cap are processed at native resolution; larger
    // images (e.g. phone photos) are downscaled with aspect preserved, and mask
    // point coordinates are scaled likewise. sam2 no longer receives or scales K
    // (obj-migration Phase 5): REG owns K and unprojects depth_metric.bin itself,
    // scaling its own K to the depth resolution as needed.
    const int MAX_W = 1920, MAX_H = 1080;
    if (image.width > MAX_W || image.height > MAX_H) {
        float scale = std::min((float)MAX_W / image.width,
                               (float)MAX_H / image.height);
        int newW = (int)std::round(image.width  * scale);
        int newH = (int)std::round(image.height * scale);
        std::cout << "[Resize] " << image.width << "x" << image.height
                  << " -> " << newW << "x" << newH
                  << " (scale " << scale << ")" << std::endl;
        image = img::resize(image, newW, newH);
        // mask 座標もスケール
        for (auto& p : opts.points) { p.x *= scale; p.y *= scale; }
    } else {
        std::cout << "[Resize] skipped (within " << MAX_W << "x" << MAX_H
                  << ", processing at native "
                  << image.width << "x" << image.height << ")" << std::endl;
    }
    std::cout << "Processing size: " << image.width << "x" << image.height << "\n\n";

    if (opts.points.empty()) {
        opts.points.emplace_back(
            image.width / 2.0f, image.height / 2.0f,
            sam2::PointLabel::Foreground);
    }

    std::vector<uint8_t> mask;
    std::vector<float>   depthRaw;
    std::vector<uint8_t> depthMap;
    std::vector<uint8_t> maskedDepth;
    std::vector<uint8_t> maskedDepthRenorm;
    depth::DepthResult   depthResult;
    float segScore = 0.0f;

    img::createDirectory(opts.outputDir);

    bool hasSam2 = img::fileExists(opts.samEncoderPath) &&
                   img::fileExists(opts.samDecoderPath);

    if (opts.stage == Stage::Depth) {
        // Stage=depth: skip SAM2, reuse mask from previous segment stage.
        std::cout << "===== Step 1: Load existing mask (stage=depth) =====\n";
        std::string maskPath = opts.outputDir + "/segmentation_mask.png";
        if (!img::fileExists(maskPath)) {
            std::cerr << "Error: --stage=depth requires existing mask: "
                      << maskPath << "\n";
            std::cerr << "       Run with --stage=segment first.\n";
            return 1;
        }
        img::Image maskImage = img::loadImage(maskPath);
        if (maskImage.empty()) {
            std::cerr << "Error: failed to load mask image: " << maskPath << "\n";
            return 1;
        }
        if (maskImage.width != image.width || maskImage.height != image.height) {
            std::cerr << "Error: mask size mismatch (mask="
                      << maskImage.width << "x" << maskImage.height
                      << ", image=" << image.width << "x" << image.height << ")\n";
            std::cerr << "       Re-run --stage=segment with the same image"
                         " and --intrinsics-source.\n";
            return 1;
        }
        // loadImage forces 3-channel RGB; for a grayscale mask R==G==B,
        // so the R byte is the mask value.
        mask.resize(static_cast<size_t>(image.width) * image.height);
        for (int i = 0; i < image.width * image.height; ++i) {
            mask[i] = maskImage.data[i * 3];
        }
        int maskCount = 0;
        for (auto v : mask) if (v > 0) ++maskCount;
        std::cout << "Loaded mask: " << maskCount << " foreground pixels\n\n";
    } else if (hasSam2) {
        std::cout << "===== Step 1: SAM2 =====\n";
        try {
            sam2::SAM2Segmentor segmentor(
                opts.samEncoderPath, opts.samDecoderPath, opts.useCuda);
            segmentor.printModelInfo();
            img::Timer timer;
            segmentor.setImage(image.data, image.width, image.height);

            // Determine prompts for the PRIMARY mask:
            //   maskOutputName == "instrument" -> treat --point/--bg-point as
            //                                     instrument prompts. The
            //                                     "primary" mask we put into
            //                                     `mask` for the rest of the
            //                                     pipeline is the instrument
            //                                     mask in this case.
            //   else                          -> standard liver path.
            const bool primaryIsInstrument = (opts.maskOutputName == "instrument");
            auto result = segmentor.predict(opts.points);
            timer.printElapsed("Total segmentation");
            mask = result.mask;
            segScore = result.score;
            int maskCount = 0;
            for (auto v : mask) if (v > 0) maskCount++;
            std::cout << (primaryIsInstrument ? "Instrument" : "Liver")
                      << " segmentation score: " << segScore
                      << " pixels=" << maskCount << "\n";

            // Optional secondary pass for the instrument mask. Runs when
            // the caller actually provided instrument prompts; the SAM2
            // image embedding is reused, so the cost is just a decoder
            // pass (~50 ms in the legacy log).
            std::vector<uint8_t> instMask;
            if (!primaryIsInstrument && !opts.instrumentPoints.empty()) {
                std::cout << "----- SAM2: instrument pass -----\n";
                img::Timer t2;
                auto instResult = segmentor.predict(opts.instrumentPoints);
                t2.printElapsed("Instrument segmentation");
                int instCount = 0;
                for (auto v : instResult.mask) if (v > 0) instCount++;
                std::cout << "Instrument segmentation score: "
                          << instResult.score
                          << " pixels=" << instCount << "\n";
                instMask = std::move(instResult.mask);
            }

            // Write the occluder mask = (SAM2 instrument) ∪ (auto-detected
            // vignette). instMask may be empty -- in that case the helper
            // writes the vignette alone as the occluder mask, OR skips the
            // write if neither source is valid. The legacy filename
            // (instrument_segmentation_mask.png) is preserved so the
            // registration pipeline reads it unchanged.
            if (!primaryIsInstrument) {
                writeOccluderMaskMerged(image, instMask,
                                        opts.outputDir,
                                        opts.detectVignette);
            }
            std::cout << "\n";
        } catch (const std::exception& e) {
            std::cerr << "SAM2 error: " << e.what() << "\n";
            mask.assign(image.width * image.height, 255);
        }
    } else {
        std::cout << "SAM2 models not found, using full image mask.\n\n";
        mask.assign(image.width * image.height, 255);
    }

    // Stage=segment: write the mask (and a quick overlay for visual check)
    // and exit before paying the cost of depth inference. The downstream
    // depth+OBJ writes happen later; we replicate just the mask outputs here.
    if (opts.stage == Stage::Segment) {
        img::saveImage(opts.outputDir + "/original.jpg", image);
        // Pick output behaviour based on which mask the caller asked for.
        // The Instrument preview button passes maskOutputName="instrument";
        // the Liver preview button leaves it empty.
        const bool instrumentMode = (opts.maskOutputName == "instrument");

        if (instrumentMode) {
            // Instrument preview: feed the SAM2 result (mask) through the
            // merging helper so the saved file already reflects the
            // vignette union. The user inspects the FINAL occluder mask,
            // not just SAM2's contribution. The helper writes the legacy
            // filename so re-running --stage=depth picks it up.
            writeOccluderMaskMerged(image, mask, opts.outputDir,
                                    opts.detectVignette);
            std::cout << "===== Stage=segment done (instrument) =====\n"
                      << "Mask saved to: " << opts.outputDir
                      << "/instrument_segmentation_mask.png\n"
                      << "Overlay saved to: " << opts.outputDir
                      << "/instrument_segmentation_overlay.jpg\n";
        } else {
            // Liver preview: save as-is. Vignette is not relevant here --
            // the liver mask must remain tight around the organ and SAM2
            // already excluded the FOV cutoff via the prompt geometry.
            img::saveGrayscale(opts.outputDir + "/segmentation_mask.png",
                               mask, image.width, image.height);
            img::Image overlay = img::overlayMask(image, mask, 0, 255, 0, 0.5f);
            img::saveImage(opts.outputDir + "/segmentation_overlay.jpg",
                           overlay);
            std::cout << "===== Stage=segment done (liver) =====\n"
                      << "Mask saved to: " << opts.outputDir
                      << "/segmentation_mask.png\n"
                      << "Overlay saved to: " << opts.outputDir
                      << "/segmentation_overlay.jpg\n";
        }
        std::cout << "(Skipped depth inference; rerun with --stage=depth"
                     " when the mask looks good.)\n";
        return 0;
    }

    if (img::fileExists(opts.depthModelPath)) {
        std::cout << "===== Step 2: Depth Anything V3 =====\n";
        try {
            depth::DepthAnythingV3 depthEstimator(opts.depthModelPath, opts.useCuda);
            depthEstimator.printModelInfo();
            img::Timer timer;
            depthResult = depthEstimator.predictFull(image.data, image.width, image.height);
            depthRaw = depthResult.depth;
            depthMap = img::normalizeDepth(depthRaw, image.width, image.height, true);
            timer.printElapsed("Depth estimation total");
        } catch (const std::exception& e) {
            std::cerr << "Depth estimation error: " << e.what() << "\n";
            return 1;
        }
    } else {
        std::cerr << "Depth model not found: " << opts.depthModelPath << "\n";
        return 1;
    }

    std::cout << "===== Step 3: Masked depth =====\n";
    maskedDepth = depthMap;
    for (int i = 0; i < image.width * image.height; ++i) {
        if (mask[i] == 0) maskedDepth[i] = 0;
    }
    maskedDepthRenorm = img::normalizeDepthMasked(
        depthRaw, mask, image.width, image.height, true);

    auto stats = img::computeDepthStats(depthMap, mask, image.width, image.height);
    std::cout << "Global norm: Min=" << stats.min << " Max=" << stats.max
              << " Mean=" << stats.mean << " Median=" << stats.median << "\n";
    auto statsRenorm = img::computeDepthStats(
        maskedDepthRenorm, mask, image.width, image.height);
    std::cout << "Re-norm:    Min=" << statsRenorm.min << " Max=" << statsRenorm.max
              << " Mean=" << statsRenorm.mean << " Median=" << statsRenorm.median << "\n\n";

    std::cout << "===== Step 4: Saving results =====\n";

    img::saveImage(opts.outputDir + "/original.jpg", image);
    img::saveGrayscale(opts.outputDir + "/segmentation_mask.png",
                       mask, image.width, image.height);

    img::Image overlay = img::overlayMask(image, mask, 0, 255, 0, 0.5f);
    img::saveImage(opts.outputDir + "/segmentation_overlay.jpg", overlay);

    img::saveGrayscale(opts.outputDir + "/depth_full.png",
                       depthMap, image.width, image.height);
    img::saveGrayscale(opts.outputDir + "/depth_masked.png",
                       maskedDepth, image.width, image.height);
    img::saveGrayscale(opts.outputDir + "/depth_masked_renorm.png",
                       maskedDepthRenorm, image.width, image.height);

    img::Image depthColored = img::applyViridisColormap(
        depthMap, image.width, image.height);
    img::saveImage(opts.outputDir + "/depth_full_colored.png", depthColored);
    img::Image maskedDepthColored = img::applyViridisColormap(
        maskedDepth, image.width, image.height);
    img::saveImage(opts.outputDir + "/depth_masked_colored.png", maskedDepthColored);
    img::Image maskedDepthRenormColored = img::applyViridisColormap(
        maskedDepthRenorm, image.width, image.height);
    img::saveImage(opts.outputDir + "/depth_masked_renorm_colored.png",
                   maskedDepthRenormColored);

    // Phase 3: DA3-estimated K is an inference OUTPUT (not a K round-trip), so it
    // goes to intrinsics_da3.txt -- NOT intrinsics.txt (which REG owns as canonical).
    writeIntrinsicsTxt(opts.outputDir + "/intrinsics_da3.txt", depthResult);

    // Phase 3: depth_metric.bin RE-ENABLED. float32 metric depth + 16B header
    // ("DEPT" magic, W, H, reserved). This is the canonical metric-depth handoff
    // that REG reads (DepthToObjExport::loadDepthMetricBin) to build the OBJ.
    {
        std::string binPath = opts.outputDir + "/depth_metric.bin";
        std::ofstream ofs(binPath, std::ios::binary);
        if (ofs.is_open()) {
            uint32_t magic    = 0x44455054;
            uint32_t W        = static_cast<uint32_t>(image.width);
            uint32_t H        = static_cast<uint32_t>(image.height);
            uint32_t reserved = 0;
            ofs.write(reinterpret_cast<const char*>(&magic), 4);
            ofs.write(reinterpret_cast<const char*>(&W), 4);
            ofs.write(reinterpret_cast<const char*>(&H), 4);
            ofs.write(reinterpret_cast<const char*>(&reserved), 4);
            ofs.write(reinterpret_cast<const char*>(depthRaw.data()),
                      sizeof(float) * depthRaw.size());
            ofs.close();
            std::cout << "[depth_metric.bin] saved: " << binPath
                      << " (" << W << "x" << H << ", float32)" << std::endl;
        }
    }


    std::cout << "Results saved to: " << opts.outputDir << "\n";
    {
        // obj-migration Phase 5: sam2 no longer writes OBJ / texture / round-trip K.
        // It outputs depth + masks + the DA3-estimated K; REG builds the OBJ.
        std::cout << "\n===== Done! =====\n"
                  << "Output files:\n"
                  << "  - original.jpg / segmentation_mask.png / segmentation_overlay.jpg\n"
                  << "  - depth_full.png / depth_masked.png / depth_masked_renorm.png (+colored)\n"
                  << "  - depth_metric.bin (float32 metric depth, 16B DEPT header)\n"
                  << "  - intrinsics_da3.txt (DA3-estimated K, 7 lines)\n";
    }

    return 0;
}
