#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include "DepthRunner.h"

// --- パス変数（initPaths()で解決される） ---
inline std::string MODEL_PATH       = "../../../model/";
inline std::string SHADERS_PATH     = "../../../shaders/";
inline std::string REG_MODEL_PATH   = "../../../registration_model/";
inline std::string INPUT_IMAGE_PATH = "../../../input_image/";
inline std::string DEPTH_OUTPUT_PATH = "../../../depth_output/";
inline std::string ONNX_MODELS_PATH = "../../../medsam2_da3_lite/onnx_models/";

#ifdef _WIN32
inline std::string DEPTH_EXE_PATH   = "./medsam2_da3_lite.exe";
#else
inline std::string DEPTH_EXE_PATH   = "./medsam2_da3_lite";
#endif

// --- Depth Model Selector ---
enum DepthModelSize {
    DEPTH_MODEL_SMALL = 0,
    DEPTH_MODEL_BASE = 1,
    DEPTH_MODEL_LARGE = 2,
    DEPTH_MODEL_COUNT
};

inline int gCurrentDepthModel = DEPTH_MODEL_SMALL;

inline const char* depthModelName(int idx) {
    switch (idx) {
    case DEPTH_MODEL_SMALL: return "Small (fast, ~100MB)";
    case DEPTH_MODEL_BASE:  return "Base (~400MB)";
    case DEPTH_MODEL_LARGE: return "Large (accurate, ~1.3GB)";
    default: return "Unknown";
    }
}

inline std::string depthModelPath(int idx) {
    switch (idx) {
    case DEPTH_MODEL_SMALL: return ONNX_MODELS_PATH + "depth_anything_v3_small.onnx";
    case DEPTH_MODEL_BASE:  return ONNX_MODELS_PATH + "base/model.onnx";
    case DEPTH_MODEL_LARGE: return ONNX_MODELS_PATH + "large/model.onnx";
    default: return ONNX_MODELS_PATH + "depth_anything_v3_small.onnx";
    }
}

inline bool isDepthModelAvailable(int idx) {
    return std::filesystem::exists(depthModelPath(idx));
}

inline void switchDepthModel(DepthRunner& depthRunner, int modelIdx) {
    gCurrentDepthModel = modelIdx;
    depthRunner.config.depthModel = depthModelPath(modelIdx);
    std::cout << "[DepthModel] Switched to: " << depthModelName(modelIdx)
              << " (" << depthModelPath(modelIdx) << ")" << std::endl;
}

// --- ファイルパス変数 ---
inline std::string TARGET_FILE_PATH;
inline std::string OUTPUT_TET_FILE;
inline std::string SMOOTH_OBJ_PATH;
inline int SMOOTH_ITERATION = 5;
inline float SMOOTH_FACTOR = 0.5;
inline std::string PORTAL_FILE_PATH;
inline std::string VEIN_FILE_PATH;
inline std::string TUMOR_FILE_PATH;
inline std::string SEGMENT_FILE_PATH;
inline std::string GB_FILE_PATH;

inline std::string Reg_TARGET_FILE_PATH;
inline std::string Reg_PORTAL_FILE_PATH;
inline std::string Reg_VEIN_FILE_PATH;
inline std::string Reg_TUMOR_FILE_PATH;
inline std::string Reg_SEGMENT_FILE_PATH;
inline std::string Reg_GB_FILE_PATH;

inline std::string gDepthInputImage;

// --- 自動検出ヘルパー ---
inline std::string findPath(const std::string& name,
                            const std::string& testFile,
                            const std::vector<std::string>& candidates) {
    std::string envName = "AAA_" + name;
    const char* envVal = std::getenv(envName.c_str());
    if (envVal) {
        std::string p = envVal;
        if (!p.empty() && p.back() != '/') p += '/';
        std::cout << "[Path] " << name << " (env " << envName << "): " << p << std::endl;
        return p;
    }

    for (const auto& path : candidates) {
        if (testFile.empty()) {
            if (std::filesystem::exists(path)) {
                std::cout << "[Path] " << name << " (auto): " << path << std::endl;
                return path;
            }
        } else {
            if (std::filesystem::exists(path + testFile)) {
                std::cout << "[Path] " << name << " (auto): " << path << std::endl;
                return path;
            }
        }
    }

    std::cerr << "[Path] " << name << ": NOT FOUND, using default: " << candidates.front() << std::endl;
    return candidates.front();
}

inline std::string findExe(const std::string& name,
                           const std::vector<std::string>& candidates) {
    const char* envVal = std::getenv(("AAA_" + name).c_str());
    if (envVal) {
        std::cout << "[Path] " << name << " (env): " << envVal << std::endl;
        return envVal;
    }

    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            std::cout << "[Path] " << name << " (auto): " << path << std::endl;
            return path;
        }
    }

    std::cerr << "[Path] " << name << ": NOT FOUND" << std::endl;
    return candidates.front();
}

// --- メイン初期化関数 ---
inline void initPaths() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "========================================" << std::endl;

    MODEL_PATH = findPath("MODEL_PATH", "liver.obj", {
                                                         "model/",
                                                         "../model/",
                                                         "../../model/",
                                                         "../../../model/",
                                                         "../../../../model/"
                                                     });

    SHADERS_PATH = findPath("SHADERS_PATH", "basic.vert", {
                                                              "shaders/",
                                                              "../shaders/",
                                                              "../../shaders/",
                                                              "../../../shaders/",
                                                              "../../../../shaders/"
                                                          });

    REG_MODEL_PATH = findPath("REG_MODEL_PATH", "", {
                                                        "registration_model/",
                                                        "../registration_model/",
                                                        "../../registration_model/",
                                                        "../../../registration_model/",
                                                        "../../../../registration_model/"
                                                    });
    if (!std::filesystem::exists(REG_MODEL_PATH)) {
        std::filesystem::create_directories(REG_MODEL_PATH);
        std::cout << "[Path] Created: " << REG_MODEL_PATH << std::endl;
    }

    INPUT_IMAGE_PATH = findPath("INPUT_IMAGE_PATH", "target.jpg", {
                                                                      "input_image/",
                                                                      "../input_image/",
                                                                      "../../input_image/",
                                                                      "../../../input_image/",
                                                                      "../../../../input_image/"
                                                                  });

    DEPTH_OUTPUT_PATH = findPath("DEPTH_OUTPUT_PATH", "", {
                                                              "depth_output/",
                                                              "../depth_output/",
                                                              "../../depth_output/",
                                                              "../../../depth_output/",
                                                              "../../../../depth_output/"
                                                          });
    if (!std::filesystem::exists(DEPTH_OUTPUT_PATH)) {
        std::filesystem::create_directories(DEPTH_OUTPUT_PATH);
        std::cout << "[Path] Created: " << DEPTH_OUTPUT_PATH << std::endl;
    }

    ONNX_MODELS_PATH = findPath("ONNX_MODELS_PATH", "depth_anything_v3_small.onnx", {
                                                                                        "onnx_models/",
                                                                                        "../onnx_models/",
                                                                                        "medsam2_da3_lite/onnx_models/",
                                                                                        "../medsam2_da3_lite/onnx_models/",
                                                                                        "../../medsam2_da3_lite/onnx_models/",
                                                                                        "../../../medsam2_da3_lite/onnx_models/",
                                                                                        "../../../../medsam2_da3_lite/onnx_models/"
                                                                                    });

    DEPTH_EXE_PATH = findExe("DEPTH_EXE_PATH", {
#ifdef _WIN32
                                                   "./medsam2_da3_lite.exe",
                                                   "medsam2_da3_lite.exe",
                                                   "../bin/medsam2_da3_lite.exe",
                                                   "../../bin/medsam2_da3_lite.exe",
                                                   "../../../bin/medsam2_da3_lite.exe",
#else
                                                   "./medsam2_da3_lite",
                                                   "../../../medsam2_da3_lite/build/medsam2_da3_lite",
                                                   "../../medsam2_da3_lite/build/medsam2_da3_lite",
                                                   "../medsam2_da3_lite/build/medsam2_da3_lite",
                                                   "../bin/medsam2_da3_lite",
                                                   "medsam2_da3_lite",
#endif
                                               });

    std::cout << "========================================" << std::endl;
    std::cout << "Final paths:" << std::endl;
    std::cout << "  MODEL_PATH:       " << MODEL_PATH << std::endl;
    std::cout << "  SHADERS_PATH:     " << SHADERS_PATH << std::endl;
    std::cout << "  REG_MODEL_PATH:   " << REG_MODEL_PATH << std::endl;
    std::cout << "  INPUT_IMAGE_PATH: " << INPUT_IMAGE_PATH << std::endl;
    std::cout << "  DEPTH_OUTPUT_PATH:" << DEPTH_OUTPUT_PATH << std::endl;
    std::cout << "  ONNX_MODELS_PATH: " << ONNX_MODELS_PATH << std::endl;
    std::cout << "  DEPTH_EXE_PATH:   " << DEPTH_EXE_PATH << std::endl;
    std::cout << "========================================\n" << std::endl;
}

inline void initFilePaths() {
    TARGET_FILE_PATH  = MODEL_PATH + "liver.obj";
    OUTPUT_TET_FILE   = MODEL_PATH + "smoothliverG30_tetrahedral_mesh.txt";
    SMOOTH_OBJ_PATH   = MODEL_PATH + "smoothliverG30.obj";
    PORTAL_FILE_PATH  = MODEL_PATH + "portal.obj";
    VEIN_FILE_PATH    = MODEL_PATH + "vein.obj";
    TUMOR_FILE_PATH   = MODEL_PATH + "tumor.obj";
    SEGMENT_FILE_PATH = MODEL_PATH + "res.obj";
    GB_FILE_PATH      = MODEL_PATH + "gb.obj";

    Reg_TARGET_FILE_PATH  = REG_MODEL_PATH + "reg_liver.obj";
    Reg_PORTAL_FILE_PATH  = REG_MODEL_PATH + "reg_portal.obj";
    Reg_VEIN_FILE_PATH    = REG_MODEL_PATH + "reg_vein.obj";
    Reg_TUMOR_FILE_PATH   = REG_MODEL_PATH + "reg_tumor.obj";
    Reg_SEGMENT_FILE_PATH = REG_MODEL_PATH + "reg_res.obj";
    Reg_GB_FILE_PATH      = REG_MODEL_PATH + "reg_gb.obj";

    gDepthInputImage = INPUT_IMAGE_PATH + "target.jpg";
}

inline void initDepthRunnerConfig(DepthRunner& depthRunner) {
    depthRunner.config.exePath    = DEPTH_EXE_PATH;
    depthRunner.config.depthModel = depthModelPath(gCurrentDepthModel);
    depthRunner.config.samEncoder = ONNX_MODELS_PATH + "sam2_hiera_tiny.encoder.onnx";
    depthRunner.config.samDecoder = ONNX_MODELS_PATH + "sam2_hiera_tiny.decoder.onnx";
    depthRunner.config.outputDir  = DEPTH_OUTPUT_PATH;
}
