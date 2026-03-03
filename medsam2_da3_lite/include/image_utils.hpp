#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace img {

// =============================================================================
// 画像構造体
// =============================================================================

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;
    
    Image() = default;
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c, 0) {}
    
    bool empty() const { return data.empty(); }
    
    uint8_t at(int x, int y, int c) const {
        return data[(y * width + x) * channels + c];
    }
    
    uint8_t& at(int x, int y, int c) {
        return data[(y * width + x) * channels + c];
    }
};

struct ImageF {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;
    
    ImageF() = default;
    ImageF(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c, 0.0f) {}
    
    bool empty() const { return data.empty(); }
    
    float at(int x, int y, int c) const {
        return data[(y * width + x) * channels + c];
    }
    
    float& at(int x, int y, int c) {
        return data[(y * width + x) * channels + c];
    }
};

// =============================================================================
// 深度統計情報
// =============================================================================

struct DepthStats {
    float min = 0;
    float max = 0;
    float mean = 0;
    float median = 0;
    float stddev = 0;
};

// =============================================================================
// タイマー
// =============================================================================

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsedMs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    void printElapsed(const std::string& label) const {
        std::cout << label << ": " << elapsedMs() << " ms" << std::endl;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// 画像I/O
// =============================================================================

Image loadImage(const std::string& path);
bool saveImage(const std::string& path, const Image& image);
bool saveGrayscale(const std::string& path, const std::vector<uint8_t>& data, int width, int height);

// =============================================================================
// 画像処理
// =============================================================================

Image resize(const Image& src, int newWidth, int newHeight);
ImageF resizeF(const ImageF& src, int newWidth, int newHeight);

void rgbToBgr(Image& image);
void bgrToRgb(Image& image);

std::vector<float> preprocessImageNet(const Image& image, int targetWidth, int targetHeight);
std::vector<float> preprocessForDA3(const Image& image, int targetSize);
std::vector<float> preprocessForSAM2(const Image& image, int targetSize);

// =============================================================================
// マスク処理
// =============================================================================

std::vector<uint8_t> postprocessMask(
    const std::vector<float>& maskData,
    int maskWidth,
    int maskHeight,
    int targetWidth,
    int targetHeight,
    float threshold = 0.0f
);

Image overlayMask(
    const Image& image,
    const std::vector<uint8_t>& mask,
    uint8_t r = 0, uint8_t g = 255, uint8_t b = 0,
    float alpha = 0.5f
);

// =============================================================================
// 深度マップ処理
// =============================================================================

std::vector<uint8_t> normalizeDepth(
    const std::vector<float>& depth,
    int width,
    int height,
    bool invert = false
);

std::vector<uint8_t> normalizeDepthMasked(
    const std::vector<float>& depth,
    const std::vector<uint8_t>& mask,
    int width,
    int height,
    bool invert = false
);

Image applyViridisColormap(const std::vector<uint8_t>& grayscale, int width, int height);

// =============================================================================
// 統計情報
// =============================================================================

DepthStats computeDepthStats(
    const std::vector<uint8_t>& depth,
    const std::vector<uint8_t>& mask,
    int width,
    int height
);

// =============================================================================
// ユーティリティ
// =============================================================================

bool fileExists(const std::string& path);
bool createDirectory(const std::string& path);

} // namespace img
