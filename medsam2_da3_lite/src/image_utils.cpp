// stb_imageã®å®Ÿè£…ã‚’æœ‰åŠ¹åŒ–ï¼ˆã“ã®ãƒ•ã‚¡ã‚¤ãƒ«ã®ã¿ï¼‰
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"

#include "image_utils.hpp"
#include <filesystem>
#include <cstring>
#include <cmath>
#include <limits>

namespace img {

// =============================================================================
// ç”»åƒI/O
// =============================================================================

Image loadImage(const std::string& path) {
    Image img;
    
    int w, h, c;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 3);  // å¼·åˆ¶RGB
    
    if (!data) {
        std::cerr << "Failed to load image: " << path << " - " << stbi_failure_reason() << std::endl;
        return img;
    }
    
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(data, data + w * h * 3);
    
    stbi_image_free(data);
    return img;
}

bool saveImage(const std::string& path, const Image& image) {
    if (image.empty()) return false;
    
    // æ‹¡å¼µå­ã§å½¢å¼åˆ¤å®š
    std::string ext = path.substr(path.find_last_of('.'));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".png") {
        return stbi_write_png(path.c_str(), image.width, image.height,
                              image.channels, image.data.data(),
                              image.width * image.channels) != 0;
    } else if (ext == ".jpg" || ext == ".jpeg") {
        return stbi_write_jpg(path.c_str(), image.width, image.height,
                              image.channels, image.data.data(), 95) != 0;
    } else if (ext == ".bmp") {
        return stbi_write_bmp(path.c_str(), image.width, image.height,
                              image.channels, image.data.data()) != 0;
    }
    
    // ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã¯PNG
    return stbi_write_png(path.c_str(), image.width, image.height,
                          image.channels, image.data.data(),
                          image.width * image.channels) != 0;
}

bool saveGrayscale(const std::string& path, const std::vector<uint8_t>& data, int width, int height) {
    if (data.empty()) return false;
    return stbi_write_png(path.c_str(), width, height, 1, data.data(), width) != 0;
}

// =============================================================================
// ç”»åƒå‡¦ç†
// =============================================================================

Image resize(const Image& src, int newWidth, int newHeight) {
    Image dst(newWidth, newHeight, src.channels);
    
    if (src.empty()) return dst;
    
    stbir_resize_uint8_linear(
        src.data.data(), src.width, src.height, src.width * src.channels,
        dst.data.data(), newWidth, newHeight, newWidth * src.channels,
        (stbir_pixel_layout)src.channels
    );
    
    return dst;
}

ImageF resizeF(const ImageF& src, int newWidth, int newHeight) {
    ImageF dst(newWidth, newHeight, src.channels);
    
    if (src.empty()) return dst;
    
    // ãƒã‚¤ãƒªãƒ‹ã‚¢è£œé–“
    float scaleX = static_cast<float>(src.width) / newWidth;
    float scaleY = static_cast<float>(src.height) / newHeight;
    
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            
            int x0 = static_cast<int>(srcX);
            int y0 = static_cast<int>(srcY);
            int x1 = std::min(x0 + 1, src.width - 1);
            int y1 = std::min(y0 + 1, src.height - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            for (int c = 0; c < src.channels; ++c) {
                float v00 = src.at(x0, y0, c);
                float v10 = src.at(x1, y0, c);
                float v01 = src.at(x0, y1, c);
                float v11 = src.at(x1, y1, c);
                
                float value = (1 - fx) * (1 - fy) * v00 +
                              fx * (1 - fy) * v10 +
                              (1 - fx) * fy * v01 +
                              fx * fy * v11;
                
                dst.at(x, y, c) = value;
            }
        }
    }
    
    return dst;
}

void rgbToBgr(Image& image) {
    for (int i = 0; i < image.width * image.height; ++i) {
        std::swap(image.data[i * 3 + 0], image.data[i * 3 + 2]);
    }
}

void bgrToRgb(Image& image) {
    rgbToBgr(image);  // åŒã˜æ“ä½œ
}

std::vector<float> preprocessImageNet(const Image& image, int targetWidth, int targetHeight) {
    // ãƒªã‚µã‚¤ã‚º
    Image resized = resize(image, targetWidth, targetHeight);
    
    // ImageNetæ­£è¦åŒ–ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
    // CHWå½¢å¼ã«å¤‰æ›ã—ãªãŒã‚‰æ­£è¦åŒ–
    std::vector<float> output(3 * targetHeight * targetWidth);
    
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < targetHeight; ++y) {
            for (int x = 0; x < targetWidth; ++x) {
                float value = resized.at(x, y, c) / 255.0f;
                value = (value - mean[c]) / std[c];
                output[c * targetHeight * targetWidth + y * targetWidth + x] = value;
            }
        }
    }
    
    return output;
}

std::vector<float> preprocessForDA3(const Image& image, int targetSize) {
    return preprocessImageNet(image, targetSize, targetSize);
}

std::vector<float> preprocessForSAM2(const Image& image, int targetSize) {
    return preprocessImageNet(image, targetSize, targetSize);
}

// =============================================================================
// ãƒžã‚¹ã‚¯å‡¦ç†
// =============================================================================

std::vector<uint8_t> postprocessMask(
    const std::vector<float>& maskData,
    int maskWidth,
    int maskHeight,
    int targetWidth,
    int targetHeight,
    float threshold
) {
    // äºŒå€¤åŒ–
    std::vector<uint8_t> binaryMask(maskWidth * maskHeight);
    for (int i = 0; i < maskWidth * maskHeight; ++i) {
        binaryMask[i] = (maskData[i] > threshold) ? 255 : 0;
    }
    
    // ãƒªã‚µã‚¤ã‚ºï¼ˆãƒ‹ã‚¢ãƒ¬ã‚¹ãƒˆãƒã‚¤ãƒãƒ¼ï¼‰
    std::vector<uint8_t> result(targetWidth * targetHeight);
    float scaleX = static_cast<float>(maskWidth) / targetWidth;
    float scaleY = static_cast<float>(maskHeight) / targetHeight;
    
    for (int y = 0; y < targetHeight; ++y) {
        for (int x = 0; x < targetWidth; ++x) {
            int srcX = static_cast<int>(x * scaleX);
            int srcY = static_cast<int>(y * scaleY);
            srcX = std::min(srcX, maskWidth - 1);
            srcY = std::min(srcY, maskHeight - 1);
            result[y * targetWidth + x] = binaryMask[srcY * maskWidth + srcX];
        }
    }
    
    return result;
}

Image overlayMask(
    const Image& image,
    const std::vector<uint8_t>& mask,
    uint8_t r, uint8_t g, uint8_t b,
    float alpha
) {
    Image result = image;  // ã‚³ãƒ”ãƒ¼
    
    for (int i = 0; i < image.width * image.height; ++i) {
        if (mask[i] > 0) {
            result.data[i * 3 + 0] = static_cast<uint8_t>(result.data[i * 3 + 0] * (1 - alpha) + r * alpha);
            result.data[i * 3 + 1] = static_cast<uint8_t>(result.data[i * 3 + 1] * (1 - alpha) + g * alpha);
            result.data[i * 3 + 2] = static_cast<uint8_t>(result.data[i * 3 + 2] * (1 - alpha) + b * alpha);
        }
    }
    
    return result;
}

// =============================================================================
// æ·±åº¦ãƒžãƒƒãƒ—å‡¦ç†
// =============================================================================

std::vector<uint8_t> normalizeDepth(
    const std::vector<float>& depth,
    int width,
    int height,
    bool invert
) {
    std::vector<uint8_t> result(width * height);
    
    // æœ€å°ãƒ»æœ€å¤§å€¤
    float minVal = *std::min_element(depth.begin(), depth.end());
    float maxVal = *std::max_element(depth.begin(), depth.end());
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;
    
    for (int i = 0; i < width * height; ++i) {
        float normalized = (depth[i] - minVal) / range;
        if (invert) {
            normalized = 1.0f - normalized;
        }
        result[i] = static_cast<uint8_t>(normalized * 255.0f);
    }
    
    return result;
}

std::vector<uint8_t> normalizeDepthMasked(
    const std::vector<float>& depth,
    const std::vector<uint8_t>& mask,
    int width,
    int height,
    bool invert
) {
    // マスク内の深度値を収集
    std::vector<float> values;
    values.reserve(width * height / 4);
    for (int i = 0; i < width * height; ++i) {
        if (mask[i] > 0) {
            values.push_back(depth[i]);
        }
    }
    
    if (values.empty()) {
        return std::vector<uint8_t>(width * height, 0);
    }
    
    // ソートしてパーセンタイルで外れ値を除外（2%-98%）
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    float minVal = values[static_cast<size_t>(n * 0.02)];
    float maxVal = values[static_cast<size_t>(n * 0.98)];
    
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;
    
    std::vector<uint8_t> result(width * height, 0);  // 背景は0（黒）
    
    // マスク内をクリップ＆正規化
    for (int i = 0; i < width * height; ++i) {
        if (mask[i] > 0) {
            float val = depth[i];
            // クリップ
            val = std::max(val, minVal);
            val = std::min(val, maxVal);
            
            float normalized = (val - minVal) / range;
            if (invert) {
                normalized = 1.0f - normalized;
            }
            uint8_t out = static_cast<uint8_t>(normalized * 255.0f);
            result[i] = std::max(out, static_cast<uint8_t>(1));
        }
    }
    
    return result;
}

// Viridisã‚«ãƒ©ãƒ¼ãƒžãƒƒãƒ—ï¼ˆ256è‰²ãƒ†ãƒ¼ãƒ–ãƒ«ï¼‰
static const uint8_t VIRIDIS_R[] = {68,68,68,69,69,69,70,70,70,70,71,71,71,71,71,71,71,72,72,72,72,72,72,72,72,72,71,71,71,71,71,71,71,70,70,70,70,69,69,69,69,68,68,67,67,67,66,66,65,65,64,64,63,63,62,61,61,60,60,59,58,58,57,56,56,55,54,54,53,52,51,51,50,49,48,48,47,46,45,44,44,43,42,41,40,40,39,38,37,36,35,35,34,33,32,31,31,30,29,28,27,27,26,25,25,24,23,23,22,21,21,20,20,19,19,18,18,18,17,17,17,17,16,16,16,16,16,16,16,17,17,17,17,18,18,19,19,20,20,21,22,22,23,24,25,26,27,28,29,30,31,32,34,35,36,38,39,40,42,43,45,46,48,50,51,53,55,57,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,91,93,95,97,99,102,104,106,108,110,113,115,117,119,122,124,126,128,130,133,135,137,139,141,143,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,179,181,183,185,186,188,190,191,193,194,196,197,199,200,202,203,204,206,207,208,209,210,212,213,214,215,216,217,218,219};
static const uint8_t VIRIDIS_G[] = {1,2,3,5,6,8,9,11,12,14,15,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,66,67,68,69,70,71,72,73,73,74,75,76,77,78,78,79,80,81,82,82,83,84,85,85,86,87,88,89,89,90,91,92,92,93,94,95,95,96,97,98,98,99,100,100,101,102,103,103,104,105,106,106,107,108,108,109,110,110,111,112,113,113,114,115,115,116,117,117,118,119,120,120,121,122,122,123,124,125,125,126,127,128,128,129,130,131,131,132,133,134,135,135,136,137,138,139,140,141,141,142,143,144,145,146,147,148,149,150,151,152,153,155,156,157,158,159,160,162,163,164,165,167,168,169,171,172,173,175,176,177,179,180,182,183,185,186,187,189,190,192,193,195,196,198,199,201,202,204,205,207,208,210,211,213,214,215,217,218,220,221,222,224,225,226,227,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,244,245,246,246,247,248,248,249,249,250,250,251,251,252,252,252,253,253,253,253};
static const uint8_t VIRIDIS_B[] = {84,85,87,88,90,91,93,94,96,97,99,100,101,103,104,105,107,108,109,110,111,112,113,114,115,116,117,117,118,119,119,120,120,121,121,122,122,122,123,123,123,123,124,124,124,124,124,124,124,124,124,124,124,124,124,123,123,123,123,123,122,122,122,121,121,121,120,120,119,119,118,118,117,117,116,116,115,114,114,113,112,112,111,110,109,109,108,107,106,105,104,103,102,101,100,99,98,97,96,95,94,93,91,90,89,88,87,85,84,83,81,80,79,77,76,74,73,71,70,68,67,65,63,62,60,58,57,55,53,51,50,48,46,44,42,41,39,37,35,33,31,29,27,25,24,22,20,18,16,15,13,11,10,8,7,5,4,3,2,2,1,1,1,0,0,0,0,0,0,1,1,2,2,3,4,5,6,7,9,10,12,13,15,17,18,20,22,24,26,28,30,32,34,36,38,41,43,45,48,50,52,55,57,60,62,65,67,70,73,75,78,81,83,86,89,91,94,97,100,103,106,108,111,114,117,120,123,126,129,132,135,138,141,144,148,151,154,157,160,164,167,170,174,177,181,184,188,191,195,198,202,206,210,214,217,221};

Image applyViridisColormap(const std::vector<uint8_t>& grayscale, int width, int height) {
    Image result(width, height, 3);
    
    for (int i = 0; i < width * height; ++i) {
        uint8_t val = grayscale[i];
        result.data[i * 3 + 0] = VIRIDIS_R[val];
        result.data[i * 3 + 1] = VIRIDIS_G[val];
        result.data[i * 3 + 2] = VIRIDIS_B[val];
    }
    
    return result;
}

// =============================================================================
// çµ±è¨ˆæƒ…å ±
// =============================================================================

DepthStats computeDepthStats(
    const std::vector<uint8_t>& depth,
    const std::vector<uint8_t>& mask,
    int width,
    int height
) {
    DepthStats stats;
    std::vector<float> values;
    
    for (int i = 0; i < width * height; ++i) {
        if (mask[i] > 0) {
            values.push_back(static_cast<float>(depth[i]));
        }
    }
    
    if (values.empty()) {
        return stats;
    }
    
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    stats.mean = sum / values.size();
    
    // ä¸­å¤®å€¤
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    stats.median = (n % 2 == 0) ? (values[n/2 - 1] + values[n/2]) / 2.0f : values[n/2];
    
    // æ¨™æº–åå·®
    float sqSum = 0;
    for (float v : values) {
        sqSum += (v - stats.mean) * (v - stats.mean);
    }
    stats.stddev = std::sqrt(sqSum / values.size());
    
    return stats;
}

// =============================================================================
// ãƒ¦ãƒ¼ãƒ†ã‚£ãƒªãƒ†ã‚£
// =============================================================================

bool fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}

bool createDirectory(const std::string& path) {
    return std::filesystem::create_directories(path);
}

} // namespace img
