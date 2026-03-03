#pragma once
/**
 * @file FileDropHandler.h
 * @brief ローカル画像のドラッグ＆ドロップ / ファイルピッカー対応
 *
 * ドロップ/選択された画像は CameraPreview::loadLocalImageAsFrozen() で
 * frozen 状態にロードされ、既存の FG/BG クリック → Key I/K フローに合流する。
 *
 * 使い方:
 *   initOpenGL() 内で: glfwSetDropCallback(gWindow, glfw_onFileDrop);
 *   メインループ内で:  if (gFileDropped) { ... }
 */

#include <string>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <GLFW/glfw3.h>

// tinyfiledialogs（オプション）
// src/ に tinyfiledialogs.h / .c を配置し CMakeLists.txt で HAS_TINYFILEDIALOGS 定義時に有効
#ifdef HAS_TINYFILEDIALOGS
#include "tinyfiledialogs.h"
#endif

// =========================================================================
//  グローバル変数（main.cpp で定義）
// =========================================================================
extern std::string gDroppedFilePath;
extern bool        gFileDropped;

// =========================================================================
//  ヘルパー: 拡張子チェック
// =========================================================================
inline std::string getFileExtensionLower(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return ext;
}

inline bool isSupportedImageExtension(const std::string& ext) {
    return (ext == ".png"  || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".ppm"  || ext == ".bmp");
}

// =========================================================================
//  GLFW ドロップコールバック（常時有効）
// =========================================================================
inline void glfw_onFileDrop(GLFWwindow* window, int count, const char** paths) {
    (void)window;
    if (count <= 0) return;

    std::string filePath = paths[0];
    std::string ext = getFileExtensionLower(filePath);

    if (!isSupportedImageExtension(ext)) {
        std::cerr << "[FileDrop] Unsupported format: " << ext << std::endl;
        std::cerr << "[FileDrop] Supported: .png .jpg .jpeg .ppm .bmp" << std::endl;
        return;
    }

    gDroppedFilePath = filePath;
    gFileDropped     = true;

    std::cout << "[FileDrop] Queued: " << filePath << std::endl;
}

// =========================================================================
//  ファイルピッカー（Key F 用）
// =========================================================================
inline bool openImageFilePicker() {
#ifdef HAS_TINYFILEDIALOGS
    const char* filters[] = {"*.png", "*.jpg", "*.jpeg", "*.ppm", "*.bmp"};
    const char* selected = tinyfd_openFileDialog(
        "Load Image for Depth",
        "",
        5, filters,
        "Image Files (png/jpg/ppm/bmp)",
        0
        );
    if (selected) {
        gDroppedFilePath = selected;
        gFileDropped     = true;
        std::cout << "[FilePicker] Selected: " << selected << std::endl;
        return true;
    }
    std::cout << "[FilePicker] Cancelled" << std::endl;
    return false;
#else
    std::cerr << "[FilePicker] tinyfiledialogs not available." << std::endl;
    std::cerr << "[FilePicker] Use drag & drop, or build with -DHAS_TINYFILEDIALOGS" << std::endl;
    return false;
#endif
}
