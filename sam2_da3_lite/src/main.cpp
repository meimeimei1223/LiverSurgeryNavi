#include "depth_anything_v3.hpp"
#include "sam2_segmentor.hpp"
#include "image_utils.hpp"
#include "ply_exporter.hpp"
#include "obj_exporter.hpp"
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
    float confPercentile = 0.15f;

    bool  saveObjFull    = true;
    bool  saveObjMasked  = true;
    bool  saveHq         = true;
    bool  saveRelief     = true;
    float skirtThreshold = 0.05f;
    float reliefThickness = 0.05f;
    objexp::ZMode zMode  = objexp::ZMode::Metric;
    bool  hasKinectIntrinsics = true;
    float kinectFx = 918.234f;
    float kinectFy = 918.112f;
    float kinectCx = 640.152f;
    float kinectCy = 366.447f;

    // Brown-Conrady distortion coefficients (OpenCV convention). When the
    // caller passes --kinect-distortion k1,k2,k3,k4,p1,p2, these values are
    // written back into intrinsics_<tag>.txt alongside fx/fy/cx/cy. The
    // depth pipeline itself does NOT apply undistortion to the input image
    // -- that happens C++ side via Undistort.h. The values here are a
    // pass-through so user-edited distortion coefficients survive a Run
    // Depth (which would otherwise silently truncate them).
    //
    // Default 0 = no distortion lines written; output file is byte-identical
    // to the legacy format for cameras without a distortion model.
    float kinectK1 = 0.0f, kinectK2 = 0.0f, kinectK3 = 0.0f, kinectK4 = 0.0f;
    float kinectP1 = 0.0f, kinectP2 = 0.0f;

    // Tag used as a suffix on intrinsics-related output files. Default keeps
    // backwards compat ("k4a"); the caller passes --intrinsics-source <name>
    // to label outputs accurately when overriding K with custom or calibrated
    // values. Examples:
    //   --intrinsics-source custom -> intrinsics_custom.txt,
    //                                 pc_metric_pinhole_*_custom*.obj
    //   --intrinsics-source calib  -> intrinsics_calib.txt,
    //                                 pc_metric_pinhole_*_calib*.obj
    // Only ASCII letters/digits/_ are expected; not validated here, but the
    // string is interpolated into filenames so avoid spaces/slashes upstream.
    std::string intrinsicsSourceName = "k4a";

    int   maskDilate = 2;

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
              << "  --conf-percentile <f>      Drop lowest X fraction of conf for _hq (default 0.15)\n"
              << "  --skirt <f>                OBJ skirt threshold in meters (default 0.05)\n"
              << "  --thickness <f>            Relief thickness in units (default 0.05)\n"
              << "  --zmode <metric|neg|hill>  Z convention (default: metric, camera at origin, Z=d)\n"
              << "  --no-obj-full              Skip full OBJ\n"
              << "  --no-obj-masked            Skip masked OBJ\n"
              << "  --no-hq                    Skip confidence-filtered _hq outputs\n"
              << "  --no-relief                Skip PlaneRelief outputs (raw-style)\n"
              << "  --kinect-intrinsics <fx,fy,cx,cy>     Override intrinsics (default: Azure Kinect 720p)\n"
              << "  --kinect-distortion <k1,k2,k3,k4,p1,p2>  Brown-Conrady distortion (OpenCV convention).\n"
              << "                             Pipeline does NOT apply these; values are written back into\n"
              << "                             intrinsics_<name>.txt so the registration app can apply\n"
              << "                             undistortion. Default: all zero (no distortion).\n"
              << "  --intrinsics-source <name> Tag for output filenames (default: k4a). Examples: custom, calib.\n"
              << "                             intrinsics_<name>.txt and pc_metric_pinhole_*_<name>*.obj\n"
              << "  --no-kinect                Skip _<name> outputs\n"
              << "  --dilate <px>              Dilate mask by N pixels for masked OBJ (default 0 = off;\n"
              << "                             optional ablation parameter for boundary mixed-pixel handling)\n"
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
        } else if (arg == "--conf-percentile" && i + 1 < argc) {
            opts.confPercentile = std::stof(argv[++i]);
            if (opts.confPercentile < 0.0f) opts.confPercentile = 0.0f;
            if (opts.confPercentile > 0.99f) opts.confPercentile = 0.99f;
        } else if (arg == "--skirt" && i + 1 < argc) {
            opts.skirtThreshold = std::stof(argv[++i]);
        } else if (arg == "--thickness" && i + 1 < argc) {
            opts.reliefThickness = std::stof(argv[++i]);
        } else if (arg == "--zmode" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "metric") opts.zMode = objexp::ZMode::Metric;
            else if (m == "neg")    opts.zMode = objexp::ZMode::Negated;
            else if (m == "hill")   opts.zMode = objexp::ZMode::HillInverted;
        } else if (arg == "--no-obj-full") {
            opts.saveObjFull = false;
        } else if (arg == "--no-obj-masked") {
            opts.saveObjMasked = false;
        } else if (arg == "--no-hq") {
            opts.saveHq = false;
        } else if (arg == "--no-relief") {
            opts.saveRelief = false;
        } else if (arg == "--no-kinect") {
            opts.hasKinectIntrinsics = false;
        } else if (arg == "--no-vignette-detect") {
            opts.detectVignette = false;
        } else if (arg == "--dilate" && i + 1 < argc) {
            opts.maskDilate = std::stoi(argv[++i]);
            if (opts.maskDilate < 0) opts.maskDilate = 0;
        } else if (arg == "--kinect-intrinsics" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            std::vector<float> vs;
            while (std::getline(ss, tok, ',')) {
                try { vs.push_back(std::stof(tok)); } catch (...) {}
            }
            if (vs.size() >= 4) {
                opts.hasKinectIntrinsics = true;
                opts.kinectFx = vs[0]; opts.kinectFy = vs[1];
                opts.kinectCx = vs[2]; opts.kinectCy = vs[3];
            } else {
                std::cerr << "[warn] --kinect-intrinsics expects fx,fy,cx,cy\n";
            }
        } else if (arg == "--kinect-distortion" && i + 1 < argc) {
            // Brown-Conrady distortion coefficients (OpenCV convention).
            // Expects exactly 6 comma-separated floats: k1,k2,k3,k4,p1,p2.
            // Used downstream only as a pass-through into intrinsics_<tag>.txt;
            // see Options::kinectK1..kinectP2 for rationale.
            std::stringstream ss(argv[++i]);
            std::string tok;
            std::vector<float> vs;
            while (std::getline(ss, tok, ',')) {
                try { vs.push_back(std::stof(tok)); } catch (...) {}
            }
            if (vs.size() >= 6) {
                opts.kinectK1 = vs[0]; opts.kinectK2 = vs[1];
                opts.kinectK3 = vs[2]; opts.kinectK4 = vs[3];
                opts.kinectP1 = vs[4]; opts.kinectP2 = vs[5];
            } else {
                std::cerr << "[warn] --kinect-distortion expects "
                             "k1,k2,k3,k4,p1,p2 (6 values)\n";
            }
        } else if (arg == "--intrinsics-source" && i + 1 < argc) {
            opts.intrinsicsSourceName = argv[++i];
            if (opts.intrinsicsSourceName.empty()) {
                std::cerr << "[warn] --intrinsics-source empty; reverting to k4a\n";
                opts.intrinsicsSourceName = "k4a";
            }
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
    ofs << "fx "     << r.intrinsics.fx << "\n";
    ofs << "fy "     << r.intrinsics.fy << "\n";
    ofs << "cx "     << r.intrinsics.cx << "\n";
    ofs << "cy "     << r.intrinsics.cy << "\n";
    ofs << "width "  << r.width  << "\n";
    ofs << "height " << r.height << "\n";
    std::cout << "[intrinsics] Saved: " << path << std::endl;
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

    // --- Resize policy: cap at 1920x1080, preserve aspect, scale K to match ---
    //
    // No special-casing of intrinsicsSourceName. Images within the 1920x1080
    // cap are processed at native resolution (so a 1080p Kinect / preset flows
    // through unchanged); larger images (e.g. phone photos) are downscaled with
    // aspect preserved. When the caller passed its own K (--kinect-intrinsics),
    // that K MUST be scaled by the same factor so the K<->image correspondence
    // (and hence the back-projected mesh) stays correct. Mask point coordinates
    // are scaled likewise.
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
        // K も同じスケールで変換 (重要)
        if (opts.hasKinectIntrinsics) {
            opts.kinectFx *= scale;
            opts.kinectFy *= scale;
            opts.kinectCx *= scale;
            opts.kinectCy *= scale;
            std::cout << "[Resize] K scaled: fx=" << opts.kinectFx
                      << " fy=" << opts.kinectFy
                      << " cx=" << opts.kinectCx
                      << " cy=" << opts.kinectCy << std::endl;
        }
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

    writeIntrinsicsTxt(opts.outputDir + "/intrinsics.txt", depthResult);

// [COMMENTED OUT] depth_metric.bin — not used by registration app
#if 0
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
        }
    }
#endif

    bool textureWritten = false;

    std::vector<float> depthForOutput(depthRaw.size());
    {
        float maxD = 0.0f;
        for (float d : depthRaw) if (d > maxD) maxD = d;
        if (maxD <= 0.0f) maxD = 1.0f;
        for (size_t i = 0; i < depthRaw.size(); ++i) {
            float d = depthRaw[i];
            switch (opts.zMode) {
            case objexp::ZMode::HillInverted: depthForOutput[i] = maxD - d; break;
            case objexp::ZMode::Negated:      depthForOutput[i] = -d; break;
            default:                          depthForOutput[i] = d; break;
            }
        }
        std::cout << "[zMode] "
                  << (opts.zMode == objexp::ZMode::HillInverted ? "hill (Z=maxD-d, far=0, near=+Z)" :
                          opts.zMode == objexp::ZMode::Negated ? "neg (Z=-d)" :
                          "metric (Z=d, near=0, far=+Z)")
                  << " maxD=" << maxD << std::endl;
    }

// [COMMENTED OUT] Relief outputs — not used by registration app
// To re-enable: uncomment and set opts.saveRelief = true
#if 0
    if (opts.saveRelief) {
        ply::ExportOptions pr;
        pr.projection   = ply::Projection::PlaneRelief;
        pr.normalize    = ply::Normalize::None;
        pr.invertDepth  = false;
        pr.flipY        = true;
        pr.thickness    = 0.0f;
        pr.depthScale   = opts.metricScale;
        pr.binary       = true;

        pr.maskMode = ply::MaskMode::IgnoreMask;
        ply::saveTexturedPly(opts.outputDir + "/pc_relief_full.ply",
                             image, depthForOutput, mask, pr);

        pr.maskMode = ply::MaskMode::SkipOutside;
        ply::saveTexturedPly(opts.outputDir + "/pc_relief_masked.ply",
                             image, depthForOutput, mask, pr);

        objexp::ObjExportOptions orel;
        orel.projection      = objexp::Projection::PlaneRelief;
        orel.depthScale      = opts.metricScale;
        orel.thickness       = 0.0f;
        orel.flipY           = true;
        orel.invertDepth     = false;
        orel.skirtThreshold  = opts.skirtThreshold;
        orel.writeTexture    = !textureWritten;
        orel.textureFilename = "texture.png";
        orel.materialName    = "screenMat";

        objexp::saveFullMeshObj(
            opts.outputDir + "/pc_relief_full.obj",
            image, depthForOutput, orel);
        textureWritten = true;

        orel.writeTexture = false;
        objexp::saveMaskedMeshObj(
            opts.outputDir + "/pc_relief_masked.obj",
            image, depthForOutput, mask, orel);

        std::vector<float> maskVals;
        maskVals.reserve(depthRaw.size());
        for (size_t i = 0; i < depthRaw.size(); ++i) {
            if (mask[i] > 0) maskVals.push_back(depthRaw[i]);
        }
        if (maskVals.size() >= 10) {
            std::sort(maskVals.begin(), maskVals.end());
            float lo = maskVals[static_cast<size_t>(maskVals.size() * 0.02)];
            float hi = maskVals[static_cast<size_t>(maskVals.size() * 0.98)];
            float range = hi - lo;
            if (range < 1e-6f) range = 1.0f;

            std::vector<float> depthNorm(depthRaw.size(), 0.0f);
            for (size_t i = 0; i < depthRaw.size(); ++i) {
                if (mask[i] == 0) continue;
                float v = std::clamp(depthRaw[i], lo, hi);
                float d = (v - lo) / range;
                switch (opts.zMode) {
                case objexp::ZMode::HillInverted: depthNorm[i] = 1.0f - d; break;
                case objexp::ZMode::Negated:      depthNorm[i] = -d; break;
                default:                          depthNorm[i] = d; break;
                }
            }

            ply::ExportOptions prn;
            prn.projection   = ply::Projection::PlaneRelief;
            prn.normalize    = ply::Normalize::None;
            prn.invertDepth  = false;
            prn.flipY        = true;
            prn.thickness    = 0.0f;
            prn.depthScale   = 1.0f;
            prn.binary       = true;
            prn.maskMode     = ply::MaskMode::SkipOutside;
            ply::saveTexturedPly(opts.outputDir + "/pc_relief_masked_norm.ply",
                                 image, depthNorm, mask, prn);

            objexp::ObjExportOptions orn;
            orn.projection      = objexp::Projection::PlaneRelief;
            orn.depthScale      = 1.0f;
            orn.thickness       = 0.0f;
            orn.flipY           = true;
            orn.invertDepth     = false;
            orn.skirtThreshold  = opts.skirtThreshold;
            orn.writeTexture    = false;
            orn.textureFilename = "texture.png";
            orn.materialName    = "screenMat";
            orn.zMode           = objexp::ZMode::Metric;
            objexp::saveMaskedMeshObj(
                opts.outputDir + "/pc_relief_masked_norm.obj",
                image, depthNorm, mask, orn);

            std::vector<float> depthNormFlat(depthNorm.size());
            for (size_t i = 0; i < depthNorm.size(); ++i) {
                depthNormFlat[i] = depthNorm[i] * 0.3f;
            }
            ply::saveTexturedPly(opts.outputDir + "/pc_relief_masked_norm_flat.ply",
                                 image, depthNormFlat, mask, prn);
            objexp::saveMaskedMeshObj(
                opts.outputDir + "/pc_relief_masked_norm_flat.obj",
                image, depthNormFlat, mask, orn);
        }
    }
#endif

    if (depthResult.hasIntrinsics) {
        // [COMMENTED OUT] DA3-intrinsics PLY options — all PLY saves disabled
        // ply::ExportOptions po; ...

// [COMMENTED OUT] DA3-intrinsics PLY — not used by registration app
// ply::saveTexturedPly(opts.outputDir + "/pc_metric_pinhole_full.ply", ...);
// ply::saveTexturedPly(opts.outputDir + "/pc_metric_pinhole_masked.ply", ...);

// [COMMENTED OUT] Confidence threshold — not used (HQ saves disabled)
// To re-enable HQ outputs, uncomment this and the HQ save blocks below.
#if 0
        float confThreshold = 0.0f;
        bool  hasConfFilter = false;
        if (opts.saveHq && depthResult.hasConfidence &&
            !depthResult.confidence.empty())
        {
            std::vector<float> cs;
            cs.reserve(static_cast<size_t>(image.width) * image.height);
            for (int i = 0; i < image.width * image.height; ++i) {
                if (mask[i] > 0) cs.push_back(depthResult.confidence[i]);
            }
            if (!cs.empty()) {
                std::sort(cs.begin(), cs.end());
                size_t idx = static_cast<size_t>(cs.size() * opts.confPercentile);
                if (idx >= cs.size()) idx = cs.size() - 1;
                confThreshold = cs[idx];
                hasConfFilter = true;
                std::cout << "[hq] conf filter drop="
                          << (opts.confPercentile * 100.0f)
                          << "% threshold=" << confThreshold << std::endl;
            }
        }
#endif

// [COMMENTED OUT] DA3-intrinsics OBJ — not used by registration app
// (Kinect intrinsics version is used instead)
#if 0
        objexp::ObjExportOptions oo;
        oo.intrinsics.fx    = depthResult.intrinsics.fx;
        oo.intrinsics.fy    = depthResult.intrinsics.fy;
        oo.intrinsics.cx    = depthResult.intrinsics.cx;
        oo.intrinsics.cy    = depthResult.intrinsics.cy;
        oo.depthScale       = opts.metricScale;
        oo.flipY            = true;
        oo.zMode            = objexp::ZMode::Metric;
        oo.skirtThreshold   = opts.skirtThreshold;
        oo.writeTexture     = !textureWritten;
        oo.textureFilename  = "texture.png";
        oo.materialName     = "screenMat";

        if (opts.saveObjFull) {
            objexp::saveFullMeshObj(
                opts.outputDir + "/pc_metric_pinhole_full.obj",
                image, depthForOutput, oo);
            textureWritten = true;
        }
        if (opts.saveObjMasked) {
            objexp::ObjExportOptions ooMasked = oo;
            ooMasked.writeTexture = !textureWritten;
            objexp::saveMaskedMeshObj(
                opts.outputDir + "/pc_metric_pinhole_masked.obj",
                image, depthForOutput, mask, ooMasked);
            textureWritten = true;
        }
        if (opts.saveHq && hasConfFilter) {
            objexp::ObjExportOptions ooHq = oo;
            ooHq.writeTexture   = !textureWritten;
            ooHq.confidence     = &depthResult.confidence;
            ooHq.confidenceMin  = confThreshold;
            objexp::saveMaskedMeshObj(
                opts.outputDir + "/pc_metric_pinhole_masked_hq.obj",
                image, depthForOutput, mask, ooHq);
        }
#endif

        if (opts.hasKinectIntrinsics) {
            const std::string& tag = opts.intrinsicsSourceName;  // "k4a"|"custom"|"calib"|...
            std::cout << "[intrinsics:" << tag
                      << "] Using intrinsics fx=" << opts.kinectFx
                      << " fy=" << opts.kinectFy
                      << " cx=" << opts.kinectCx
                      << " cy=" << opts.kinectCy << std::endl;

            {
                // This file records the K we ACTUALLY USED (post-resize). For
                // tag="custom" the user-provided intrinsics_custom.txt is the
                // authoritative source K and must NOT be overwritten with the
                // resized values, so we write to intrinsics_custom_used.txt
                // instead. Other tags keep intrinsics_<tag>.txt.
                std::string saveName = (tag == "custom")
                                     ? std::string("intrinsics_custom_used.txt")
                                     : std::string("intrinsics_") + tag + ".txt";
                std::string intrPath = opts.outputDir + "/" + saveName;
                std::ofstream ofs(intrPath);
                if (ofs.is_open()) {
                    // Bump precision: default 6 sig-figs would drop the
                    // trailing digit of small coefficients (e.g. p1 ~ 1e-3),
                    // and we want round-trips with the C++ side's float K
                    // to be lossless. 9 digits is enough for IEEE-754
                    // single-precision (≤ 7-8 significant digits).
                    ofs << std::setprecision(9);
                    ofs << "fx "     << opts.kinectFx << "\n";
                    ofs << "fy "     << opts.kinectFy << "\n";
                    ofs << "cx "     << opts.kinectCx << "\n";
                    ofs << "cy "     << opts.kinectCy << "\n";
                    ofs << "width "  << image.width  << "\n";
                    ofs << "height " << image.height << "\n";
                    ofs << "name   " << tag << "\n";

                    // Brown-Conrady distortion (round-tripped from caller via
                    // --kinect-distortion). Written only when at least one
                    // coefficient is non-zero, so files for distortion-free
                    // cameras stay byte-identical to the legacy format and
                    // are unchanged when --kinect-distortion is not passed.
                    const float kEps = 1e-6f;
                    const bool hasDist =
                        std::fabs(opts.kinectK1) > kEps || std::fabs(opts.kinectK2) > kEps ||
                        std::fabs(opts.kinectK3) > kEps || std::fabs(opts.kinectK4) > kEps ||
                        std::fabs(opts.kinectP1) > kEps || std::fabs(opts.kinectP2) > kEps;
                    if (hasDist) {
                        ofs << "# Brown-Conrady distortion (OpenCV convention)\n";
                        ofs << "k1     " << opts.kinectK1 << "\n";
                        ofs << "k2     " << opts.kinectK2 << "\n";
                        ofs << "k3     " << opts.kinectK3 << "\n";
                        ofs << "k4     " << opts.kinectK4 << "\n";
                        ofs << "p1     " << opts.kinectP1 << "\n";
                        ofs << "p2     " << opts.kinectP2 << "\n";
                    }
                    std::cout << "[intrinsics_" << tag << "] Saved: "
                              << intrPath
                              << (hasDist ? "  (with distortion)" : "")
                              << std::endl;
                }
            }

            // [COMMENTED OUT] k4a PLY — not used by registration app
            // ply::saveTexturedPly(".../pc_metric_pinhole_full_k4a.ply", ...);
            // ply::saveTexturedPly(".../pc_metric_pinhole_masked_k4a.ply", ...);
            // ply::saveTexturedPly(".../pc_metric_pinhole_masked_hq_k4a.ply", ...);

            objexp::ObjExportOptions ok;
            ok.intrinsics.fx    = opts.kinectFx;
            ok.intrinsics.fy    = opts.kinectFy;
            ok.intrinsics.cx    = opts.kinectCx;
            ok.intrinsics.cy    = opts.kinectCy;
            ok.depthScale       = opts.metricScale;
            ok.flipY            = true;
            ok.zMode            = objexp::ZMode::Metric;
            ok.skirtThreshold   = opts.skirtThreshold;
            ok.writeTexture     = !textureWritten;
            ok.textureFilename  = "texture.png";
            ok.materialName     = "screenMat";
            ok.confidence       = nullptr;
            ok.confidenceMin    = 0.0f;

            // Full mesh — stride=10 for lightweight display mesh
            if (opts.saveObjFull) {
                objexp::ObjExportOptions okLight = ok;
                okLight.stride = 10;
                objexp::saveFullMeshObj(
                    opts.outputDir + "/pc_metric_pinhole_full_" + tag + "_light.obj",
                    image, depthForOutput, okLight);
                textureWritten = true;

                // Companion full mesh with skirt disabled, for inspection.
                // All depth-jump triangles preserved (rubber sheets visible).
                objexp::ObjExportOptions okLightNoSkirt = okLight;
                okLightNoSkirt.skirtThreshold = 0.0f;
                okLightNoSkirt.writeTexture   = false;
                objexp::saveFullMeshObj(
                    opts.outputDir + "/pc_metric_pinhole_full_" + tag + "_light_noskirt.obj",
                    image, depthForOutput, okLightNoSkirt);
            }
            // Masked mesh — full resolution for registration
            if (opts.saveObjMasked) {
                std::vector<uint8_t> maskForObj =
                    (opts.maskDilate > 0)
                        ? img::dilateMask(mask, image.width, image.height,
                                          opts.maskDilate)
                        : mask;
                if (opts.maskDilate > 0) {
                    int before = 0, after = 0;
                    for (auto v : mask)       if (v) ++before;
                    for (auto v : maskForObj) if (v) ++after;
                    std::cout << "[mask-dilate] " << opts.maskDilate
                              << " px : " << before << " -> " << after
                              << " pixels (+" << (after - before) << ")"
                              << std::endl;
                }

                objexp::ObjExportOptions okMasked = ok;
                okMasked.writeTexture = !textureWritten;
                objexp::saveMaskedMeshObj(
                    opts.outputDir + "/pc_metric_pinhole_masked_" + tag + ".obj",
                    image, depthForOutput, maskForObj, okMasked);
                textureWritten = true;

                // Companion masked mesh with skirt disabled, for inspection.
                // Mask already excludes background, so internal depth jumps
                // (crease edges) stay connected.
                objexp::ObjExportOptions okMaskedNoSkirt = okMasked;
                okMaskedNoSkirt.skirtThreshold = 0.0f;
                okMaskedNoSkirt.writeTexture   = false;
                objexp::saveMaskedMeshObj(
                    opts.outputDir + "/pc_metric_pinhole_masked_" + tag + "_noskirt.obj",
                    image, depthForOutput, maskForObj, okMaskedNoSkirt);
            }
            // [COMMENTED OUT] HQ confidence-filtered OBJ — not used
            // if (opts.saveHq && hasConfFilter) { ... }
        }
    } else {
        std::cout << "[ply/obj] No intrinsics from model, skipping metric exports\n";
    }

    std::cout << "Results saved to: " << opts.outputDir << "\n";
    {
        const std::string& tag = opts.intrinsicsSourceName;
        std::cout << "\n===== Done! =====\n"
                  << "Output files:\n"
                  << "  - original.jpg / segmentation_mask.png / segmentation_overlay.jpg\n"
                  << "  - depth_full.png / depth_masked.png / depth_masked_renorm.png (+colored)\n"
                  << "  - intrinsics.txt (DA3-estimated fx/fy/cx/cy/width/height)\n";
        if (opts.hasKinectIntrinsics) {
            std::cout
                << "  - intrinsics_" << tag << ".txt (intrinsics actually used)\n"
                << "  - pc_metric_pinhole_full_" << tag << "_light.obj (stride=10, display mesh)\n"
                << "  - pc_metric_pinhole_masked_" << tag << ".obj (full-res, registration)\n";
        }
        std::cout << "  - texture.png + .mtl per obj\n";
    }

    return 0;
}
