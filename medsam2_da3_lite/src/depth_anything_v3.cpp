#include "depth_anything_v3.hpp"
#include "image_utils.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "OrtPathHelper.h"   // ← ファイル先頭に追加

namespace depth {

DepthAnythingV3::DepthAnythingV3(const std::string& modelPath, bool useCuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DepthAnythingV3"),
      inputSize_(518)
{
    if (!img::fileExists(modelPath)) {
        throw std::runtime_error("Model not found: " + modelPath);
    }
    
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (useCuda) {
#ifdef USE_CUDA
        OrtCUDAProviderOptions cudaOptions;
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
        std::cout << "Using CUDA" << std::endl;
#else
        std::cerr << "Warning: CUDA not available, using CPU" << std::endl;
#endif
    }
    
    //session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);

    session_ = std::make_unique<Ort::Session>(env_, ORT_MODEL_PATH(modelPath), sessionOptions_);
    std::cout << "Depth Anything V3 loaded successfully" << std::endl;
}

void DepthAnythingV3::printModelInfo() const {
    std::cout << "=== Depth Anything V3 Model Info ===" << std::endl;
    std::cout << "Input size: " << inputSize_ << "x" << inputSize_ << std::endl;
    
    std::cout << "Inputs:" << std::endl;
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        auto typeInfo = session_->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        auto shape = tensorInfo.GetShape();
        
        std::cout << "  " << name.get() << ": [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

std::vector<float> DepthAnythingV3::preprocess(
    const std::vector<uint8_t>& imageData,
    int srcWidth,
    int srcHeight
) {
    // ImageNet正規化パラメータ
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
    // バイリニア補間でリサイズしながらCHW形式に変換
    std::vector<float> output(3 * inputSize_ * inputSize_);
    
    float scaleX = static_cast<float>(srcWidth) / inputSize_;
    float scaleY = static_cast<float>(srcHeight) / inputSize_;
    
    for (int y = 0; y < inputSize_; ++y) {
        for (int x = 0; x < inputSize_; ++x) {
            float srcXf = x * scaleX;
            float srcYf = y * scaleY;
            
            int x0 = static_cast<int>(srcXf);
            int y0 = static_cast<int>(srcYf);
            int x1 = std::min(x0 + 1, srcWidth - 1);
            int y1 = std::min(y0 + 1, srcHeight - 1);
            
            float fx = srcXf - x0;
            float fy = srcYf - y0;
            
            for (int c = 0; c < 3; ++c) {
                float v00 = imageData[(y0 * srcWidth + x0) * 3 + c];
                float v10 = imageData[(y0 * srcWidth + x1) * 3 + c];
                float v01 = imageData[(y1 * srcWidth + x0) * 3 + c];
                float v11 = imageData[(y1 * srcWidth + x1) * 3 + c];
                
                float value = (1 - fx) * (1 - fy) * v00 +
                              fx * (1 - fy) * v10 +
                              (1 - fx) * fy * v01 +
                              fx * fy * v11;
                
                // 0-255 -> 0-1 -> 正規化
                value = value / 255.0f;
                value = (value - mean[c]) / std[c];
                
                output[c * inputSize_ * inputSize_ + y * inputSize_ + x] = value;
            }
        }
    }
    
    return output;
}

std::vector<float> DepthAnythingV3::predict(
    const std::vector<uint8_t>& imageData,
    int width,
    int height
) {
    img::Timer timer;
    
    // 前処理
    auto inputData = preprocess(imageData, width, height);
    std::cout << "Preprocessing: " << timer.elapsedMs() << " ms" << std::endl;
    timer.reset();
    
    // 入力テンソル作成（DA3は5次元: [1, 1, 3, H, W]）
    std::vector<int64_t> inputShape = {1, 1, 3, inputSize_, inputSize_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    // 推論実行
    const char* inputNames[] = {"pixel_values"};
    const char* outputNames[] = {"predicted_depth"};
    
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );
    
    std::cout << "Inference: " << timer.elapsedMs() << " ms" << std::endl;
    timer.reset();
    
    // 出力取得
    auto& outputTensor = outputTensors[0];
    auto* outputData = outputTensor.GetTensorMutableData<float>();
    auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();
    
    // 出力サイズ
    int outH = static_cast<int>(outputShape[outputShape.size() - 2]);
    int outW = static_cast<int>(outputShape[outputShape.size() - 1]);
    
    // 元のサイズにリサイズ
    std::vector<float> depthSmall(outputData, outputData + outH * outW);
    std::vector<float> depth(width * height);
    
    float scaleX = static_cast<float>(outW) / width;
    float scaleY = static_cast<float>(outH) / height;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float srcXf = x * scaleX;
            float srcYf = y * scaleY;
            
            int x0 = static_cast<int>(srcXf);
            int y0 = static_cast<int>(srcYf);
            int x1 = std::min(x0 + 1, outW - 1);
            int y1 = std::min(y0 + 1, outH - 1);
            
            float fx = srcXf - x0;
            float fy = srcYf - y0;
            
            float v00 = depthSmall[y0 * outW + x0];
            float v10 = depthSmall[y0 * outW + x1];
            float v01 = depthSmall[y1 * outW + x0];
            float v11 = depthSmall[y1 * outW + x1];
            
            depth[y * width + x] = (1 - fx) * (1 - fy) * v00 +
                                   fx * (1 - fy) * v10 +
                                   (1 - fx) * fy * v01 +
                                   fx * fy * v11;
        }
    }
    
    std::cout << "Postprocessing: " << timer.elapsedMs() << " ms" << std::endl;
    
    return depth;
}

std::vector<uint8_t> DepthAnythingV3::predictNormalized(
    const std::vector<uint8_t>& imageData,
    int width,
    int height,
    bool invert
) {
    auto depth = predict(imageData, width, height);
    return img::normalizeDepth(depth, width, height, invert);
}

} // namespace depth
