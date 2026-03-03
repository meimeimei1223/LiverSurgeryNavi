/**
 * @file main.cpp
 * @brief Depth Anything V3 + SAM2 C++ ONNX Lite (No OpenCV)
 *
 * Usage:
 *   ./medsam2_da3_lite <image_path> [options]
 *
 * Options:
 *   --depth-model <path>       Depth Anything V3 ONNX model path
 *   --sam-encoder <path>       SAM2 encoder ONNX path
 *   --sam-decoder <path>       SAM2 decoder ONNX path
 *   --output <dir>             Output directory
 *   --point <x,y>              Segmentation point (multiple allowed)
 *   --bg-point <x,y>           Background point (multiple allowed)
 *   --cuda                     Use CUDA
 *   --help                     Show help
 */

#include "depth_anything_v3.hpp"
#include "sam2_segmentor.hpp"
#include "image_utils.hpp"

#include <iostream>
#include <string>
#include <vector>

// =============================================================================
// Command line parsing
// =============================================================================
struct Options {
    std::string imagePath;
    std::string depthModelPath = "models/depth_anything_v3_small.onnx";
    std::string samEncoderPath = "models/sam2_hiera_tiny.encoder.onnx";
    std::string samDecoderPath = "models/sam2_hiera_tiny.decoder.onnx";
    std::string outputDir = "output";
    std::vector<sam2::PointPrompt> points;
    bool useCuda = false;
    bool showHelp = false;
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
              << "  --cuda                     Use CUDA for inference\n"
              << "  --help                     Show this help\n\n"
              << "Example:\n"
              << "  " << programName << " image.jpg --point 640,360 --point 800,400\n";
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
        } else if (arg == "--cuda") {
            opts.useCuda = true;
        } else if (arg[0] != '-' && opts.imagePath.empty()) {
            opts.imagePath = arg;
        }
    }

    return opts;
}

// =============================================================================
// Main
// =============================================================================
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
    std::cout << "Image size: " << image.width << "x" << image.height << "\n\n";

    if (opts.points.empty()) {
        opts.points.emplace_back(
            image.width / 2.0f,
            image.height / 2.0f,
            sam2::PointLabel::Foreground
            );
        std::cout << "No points specified, using image center as default\n";
    }

    std::cout << "Segmentation points:\n";
    for (const auto& p : opts.points) {
        std::cout << "  [" << p.x << ", " << p.y << "] - "
                  << (p.label == sam2::PointLabel::Foreground ? "Foreground" : "Background")
                  << "\n";
    }
    std::cout << "\n";

    std::vector<uint8_t> mask;
    std::vector<float> depthRaw;
    std::vector<uint8_t> depthMap;
    std::vector<uint8_t> maskedDepth;
    std::vector<uint8_t> maskedDepthRenorm;
    float segScore = 0.0f;

    img::createDirectory(opts.outputDir);

    // ==========================================================================
    // SAM2 Segmentation
    // ==========================================================================
    bool hasSam2 = img::fileExists(opts.samEncoderPath) &&
                   img::fileExists(opts.samDecoderPath);

    if (hasSam2) {
        std::cout << "======================================================================\n"
                  << "Step 1: Running SAM2 Segmentation\n"
                  << "======================================================================\n";

        try {
            sam2::SAM2Segmentor segmentor(
                opts.samEncoderPath,
                opts.samDecoderPath,
                opts.useCuda
                );
            segmentor.printModelInfo();

            img::Timer timer;

            std::cout << "\nEncoding image..." << std::endl;
            segmentor.setImage(image.data, image.width, image.height);

            std::cout << "Running segmentation..." << std::endl;
            auto result = segmentor.predict(opts.points);
            timer.printElapsed("Total segmentation");

            mask = result.mask;
            segScore = result.score;

            int maskCount = 0;
            for (auto v : mask) if (v > 0) maskCount++;

            std::cout << "\nSegmentation score: " << segScore << "\n";
            std::cout << "Mask region: " << maskCount << " pixels\n\n";

        } catch (const std::exception& e) {
            std::cerr << "SAM2 error: " << e.what() << "\n";
            std::cerr << "Continuing without segmentation...\n\n";
            mask.assign(image.width * image.height, 255);
        }
    } else {
        std::cout << "SAM2 models not found:\n"
                  << "  Encoder: " << opts.samEncoderPath << "\n"
                  << "  Decoder: " << opts.samDecoderPath << "\n"
                  << "Skipping segmentation, using full image mask.\n\n";
        mask.assign(image.width * image.height, 255);
    }

    // ==========================================================================
    // Depth Anything V3
    // ==========================================================================
    if (img::fileExists(opts.depthModelPath)) {
        std::cout << "======================================================================\n"
                  << "Step 2: Running Depth Anything V3\n"
                  << "======================================================================\n";

        try {
            depth::DepthAnythingV3 depthEstimator(opts.depthModelPath, opts.useCuda);
            depthEstimator.printModelInfo();

            img::Timer timer;

            depthRaw = depthEstimator.predict(image.data, image.width, image.height);
            depthMap = img::normalizeDepth(depthRaw, image.width, image.height, true);

            timer.printElapsed("Depth estimation total");

            std::cout << "Depth map size: " << image.width << "x" << image.height << "\n\n";

        } catch (const std::exception& e) {
            std::cerr << "Depth estimation error: " << e.what() << "\n";
            return 1;
        }
    } else {
        std::cerr << "Depth model not found: " << opts.depthModelPath << "\n";
        return 1;
    }

    // ==========================================================================
    // Extract masked depth
    // ==========================================================================
    std::cout << "======================================================================\n"
              << "Step 3: Extracting masked depth\n"
              << "======================================================================\n";

    maskedDepth = depthMap;
    for (int i = 0; i < image.width * image.height; ++i) {
        if (mask[i] == 0) {
            maskedDepth[i] = 0;
        }
    }

    maskedDepthRenorm = img::normalizeDepthMasked(depthRaw, mask, image.width, image.height, true);

    auto stats = img::computeDepthStats(depthMap, mask, image.width, image.height);
    std::cout << "Depth statistics in masked region (global normalized):\n"
              << "  Min: " << stats.min << "\n"
              << "  Max: " << stats.max << "\n"
              << "  Mean: " << stats.mean << "\n"
              << "  Median: " << stats.median << "\n"
              << "  Std: " << stats.stddev << "\n";

    auto statsRenorm = img::computeDepthStats(maskedDepthRenorm, mask, image.width, image.height);
    std::cout << "Depth statistics in masked region (re-normalized):\n"
              << "  Min: " << statsRenorm.min << "\n"
              << "  Max: " << statsRenorm.max << "\n"
              << "  Mean: " << statsRenorm.mean << "\n"
              << "  Median: " << statsRenorm.median << "\n"
              << "  Std: " << statsRenorm.stddev << "\n\n";

    // ==========================================================================
    // Save results
    // ==========================================================================
    std::cout << "======================================================================\n"
              << "Step 4: Saving results\n"
              << "======================================================================\n";

    img::saveImage(opts.outputDir + "/original.jpg", image);

    img::saveGrayscale(opts.outputDir + "/segmentation_mask.png", mask, image.width, image.height);

    img::Image overlay = img::overlayMask(image, mask, 0, 255, 0, 0.5f);
    img::saveImage(opts.outputDir + "/segmentation_overlay.jpg", overlay);

    img::saveGrayscale(opts.outputDir + "/depth_full.png", depthMap, image.width, image.height);
    img::saveGrayscale(opts.outputDir + "/depth_masked.png", maskedDepth, image.width, image.height);
    img::saveGrayscale(opts.outputDir + "/depth_masked_renorm.png", maskedDepthRenorm, image.width, image.height);

    img::Image depthColored = img::applyViridisColormap(depthMap, image.width, image.height);
    img::saveImage(opts.outputDir + "/depth_full_colored.png", depthColored);

    img::Image maskedDepthColored = img::applyViridisColormap(maskedDepth, image.width, image.height);
    img::saveImage(opts.outputDir + "/depth_masked_colored.png", maskedDepthColored);

    img::Image maskedDepthRenormColored = img::applyViridisColormap(maskedDepthRenorm, image.width, image.height);
    img::saveImage(opts.outputDir + "/depth_masked_renorm_colored.png", maskedDepthRenormColored);

    std::cout << "Results saved to: " << opts.outputDir << "\n";

    std::cout << "\n======================================================================\n"
              << "Done!\n"
              << "======================================================================\n"
              << "Output files:\n"
              << "  - original.jpg\n"
              << "  - segmentation_mask.png\n"
              << "  - segmentation_overlay.jpg\n"
              << "  - depth_full.png              (global normalized)\n"
              << "  - depth_full_colored.png      (global normalized, color)\n"
              << "  - depth_masked.png            (global normalized, masked)\n"
              << "  - depth_masked_colored.png    (global normalized, masked, color)\n"
              << "  - depth_masked_renorm.png     (re-normalized in mask)\n"
              << "  - depth_masked_renorm_colored.png (re-normalized in mask, color)\n";

    return 0;
}
