#include "sam2_segmentor.hpp"
#include "image_utils.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "OrtPathHelper.h"

namespace sam2 {

SAM2Segmentor::SAM2Segmentor(
    const std::string& encoderPath,
    const std::string& decoderPath,
    bool useCuda
) : env_(ORT_LOGGING_LEVEL_WARNING, "SAM2Segmentor"),
    inputSize_(1024)
{
    if (!img::fileExists(encoderPath)) {
        throw std::runtime_error("Encoder not found: " + encoderPath);
    }
    if (!img::fileExists(decoderPath)) {
        throw std::runtime_error("Decoder not found: " + decoderPath);
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
    
    //encoderSession_ = std::make_unique<Ort::Session>(env_, encoderPath.c_str(), sessionOptions_);
    //decoderSession_ = std::make_unique<Ort::Session>(env_, decoderPath.c_str(), sessionOptions_);

    encoderSession_ = std::make_unique<Ort::Session>(env_, ORT_MODEL_PATH(encoderPath), sessionOptions_);
    decoderSession_ = std::make_unique<Ort::Session>(env_, ORT_MODEL_PATH(decoderPath), sessionOptions_);
    std::cout << "SAM2 Segmentor loaded successfully" << std::endl;
}

void SAM2Segmentor::printModelInfo() const {
    std::cout << "=== SAM2 Segmentor Model Info ===" << std::endl;
    std::cout << "Input size: " << inputSize_ << "x" << inputSize_ << std::endl;
    
    std::cout << "\nEncoder inputs:" << std::endl;
    for (size_t i = 0; i < encoderSession_->GetInputCount(); ++i) {
        auto name = encoderSession_->GetInputNameAllocated(i, allocator_);
        std::cout << "  " << name.get() << std::endl;
    }
    
    std::cout << "\nEncoder outputs:" << std::endl;
    for (size_t i = 0; i < encoderSession_->GetOutputCount(); ++i) {
        auto name = encoderSession_->GetOutputNameAllocated(i, allocator_);
        std::cout << "  " << name.get() << std::endl;
    }
    
    std::cout << "\nDecoder inputs:" << std::endl;
    for (size_t i = 0; i < decoderSession_->GetInputCount(); ++i) {
        auto name = decoderSession_->GetInputNameAllocated(i, allocator_);
        std::cout << "  " << name.get() << std::endl;
    }
    
    std::cout << "\nDecoder outputs:" << std::endl;
    for (size_t i = 0; i < decoderSession_->GetOutputCount(); ++i) {
        auto name = decoderSession_->GetOutputNameAllocated(i, allocator_);
        std::cout << "  " << name.get() << std::endl;
    }
}

std::vector<float> SAM2Segmentor::preprocess(
    const std::vector<uint8_t>& imageData,
    int srcWidth,
    int srcHeight
) {
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
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
                
                value = value / 255.0f;
                value = (value - mean[c]) / std[c];
                
                output[c * inputSize_ * inputSize_ + y * inputSize_ + x] = value;
            }
        }
    }
    
    return output;
}

void SAM2Segmentor::setImage(
    const std::vector<uint8_t>& imageData,
    int width,
    int height
) {
    imageWidth_ = width;
    imageHeight_ = height;
    hasEncodedImage_ = false;
    
    img::Timer timer;
    
    // 前処理
    auto inputData = preprocess(imageData, width, height);
    
    // エンコード実行
    encodeImage(inputData);
    
    timer.printElapsed("Encoder");
    hasEncodedImage_ = true;
}

void SAM2Segmentor::encodeImage(const std::vector<float>& inputData) {
    // 入力テンソル作成
    std::vector<int64_t> inputShape = {1, 3, inputSize_, inputSize_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        const_cast<float*>(inputData.data()),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    const char* inputNames[] = {"image"};
    const char* outputNames[] = {"high_res_feats_0", "high_res_feats_1", "image_embed"};
    
    auto outputTensors = encoderSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        3
    );
    
    // 出力を保存
    // high_res_feats_0: [1, 32, 256, 256]
    {
        auto& tensor = outputTensors[0];
        auto* data = tensor.GetTensorMutableData<float>();
        auto info = tensor.GetTensorTypeAndShapeInfo();
        size_t size = 1;
        for (auto dim : info.GetShape()) size *= dim;
        highResFeats0_.assign(data, data + size);
    }
    
    // high_res_feats_1: [1, 64, 128, 128]
    {
        auto& tensor = outputTensors[1];
        auto* data = tensor.GetTensorMutableData<float>();
        auto info = tensor.GetTensorTypeAndShapeInfo();
        size_t size = 1;
        for (auto dim : info.GetShape()) size *= dim;
        highResFeats1_.assign(data, data + size);
    }
    
    // image_embed: [1, 256, 64, 64]
    {
        auto& tensor = outputTensors[2];
        auto* data = tensor.GetTensorMutableData<float>();
        auto info = tensor.GetTensorTypeAndShapeInfo();
        size_t size = 1;
        for (auto dim : info.GetShape()) size *= dim;
        imageEmbed_.assign(data, data + size);
    }
    
    std::cout << "Image encoded: embed size = " << imageEmbed_.size() << std::endl;
}

SegmentationResult SAM2Segmentor::predict(const std::vector<PointPrompt>& points) {
    if (!hasEncodedImage_) {
        throw std::runtime_error("No image encoded. Call setImage() first.");
    }
    if (points.empty()) {
        throw std::runtime_error("No points provided");
    }
    
    return decode(points);
}

SegmentationResult SAM2Segmentor::decode(const std::vector<PointPrompt>& points) {
    SegmentationResult result;
    result.width = imageWidth_;
    result.height = imageHeight_;
    
    img::Timer timer;
    
    // 座標をスケーリング
    float scaleX = static_cast<float>(inputSize_) / imageWidth_;
    float scaleY = static_cast<float>(inputSize_) / imageHeight_;
    
    std::vector<float> coords;
    std::vector<float> labels;
    for (const auto& p : points) {
        coords.push_back(p.x * scaleX);
        coords.push_back(p.y * scaleY);
        labels.push_back(static_cast<float>(p.label));
    }
    
    // メモリ情報
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // 入力テンソル作成
    std::vector<Ort::Value> inputTensors;
    
    // image_embed: [1, 256, 64, 64]
    std::vector<int64_t> embedShape = {1, 256, 64, 64};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, imageEmbed_.data(), imageEmbed_.size(),
        embedShape.data(), embedShape.size()
    ));
    
    // high_res_feats_0: [1, 32, 256, 256]
    std::vector<int64_t> feats0Shape = {1, 32, 256, 256};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, highResFeats0_.data(), highResFeats0_.size(),
        feats0Shape.data(), feats0Shape.size()
    ));
    
    // high_res_feats_1: [1, 64, 128, 128]
    std::vector<int64_t> feats1Shape = {1, 64, 128, 128};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, highResFeats1_.data(), highResFeats1_.size(),
        feats1Shape.data(), feats1Shape.size()
    ));
    
    // point_coords: [1, num_points, 2]
    std::vector<int64_t> coordsShape = {1, static_cast<int64_t>(points.size()), 2};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, coords.data(), coords.size(),
        coordsShape.data(), coordsShape.size()
    ));
    
    // point_labels: [1, num_points]
    std::vector<int64_t> labelsShape = {1, static_cast<int64_t>(points.size())};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, labels.data(), labels.size(),
        labelsShape.data(), labelsShape.size()
    ));
    
    // mask_input: [1, 1, 256, 256]
    std::vector<float> maskInput(1 * 1 * 256 * 256, 0.0f);
    std::vector<int64_t> maskInputShape = {1, 1, 256, 256};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, maskInput.data(), maskInput.size(),
        maskInputShape.data(), maskInputShape.size()
    ));
    
    // has_mask_input: [1]
    std::vector<float> hasMaskInput = {0.0f};
    std::vector<int64_t> hasMaskShape = {1};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, hasMaskInput.data(), hasMaskInput.size(),
        hasMaskShape.data(), hasMaskShape.size()
    ));
    
    const char* inputNames[] = {
        "image_embed", "high_res_feats_0", "high_res_feats_1",
        "point_coords", "point_labels", "mask_input", "has_mask_input"
    };
    const char* outputNames[] = {"masks", "iou_predictions"};
    
    // デコーダー実行
    auto outputTensors = decoderSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        inputTensors.data(),
        inputTensors.size(),
        outputNames,
        2
    );
    
    timer.printElapsed("Decoder");
    
    // 出力処理
    auto& maskTensor = outputTensors[0];
    auto* maskData = maskTensor.GetTensorMutableData<float>();
    auto maskInfo = maskTensor.GetTensorTypeAndShapeInfo();
    auto maskShape = maskInfo.GetShape();
    
    auto& iouTensor = outputTensors[1];
    auto* iouData = iouTensor.GetTensorMutableData<float>();
    
    // 最高スコアのマスクを選択
    int numMasks = static_cast<int>(maskShape[1]);
    int bestIdx = 0;
    float bestScore = iouData[0];
    for (int i = 1; i < numMasks; ++i) {
        if (iouData[i] > bestScore) {
            bestScore = iouData[i];
            bestIdx = i;
        }
    }
    
    result.score = bestScore;
    std::cout << "Best mask index: " << bestIdx << ", score: " << bestScore << std::endl;
    
    // マスクを二値化してリサイズ
    int maskH = 256, maskW = 256;
    std::vector<float> bestMask(maskData + bestIdx * maskH * maskW,
                                 maskData + (bestIdx + 1) * maskH * maskW);
    
    result.mask = img::postprocessMask(bestMask, maskW, maskH, imageWidth_, imageHeight_, 0.0f);
    
    return result;
}

} // namespace sam2
