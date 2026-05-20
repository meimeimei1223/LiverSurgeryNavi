#include "depth_anything_v3.hpp"
#include "image_utils.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "OrtPathHelper.h"

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

    // ------------------------------------------------------------------
    //  CUDA execution provider with automatic CPU fallback.
    //
    //  Three control flow paths:
    //   1) USE_CUDA build flag is OFF (compile-time guard via #ifdef):
    //      No CUDA code is compiled. If useCuda=true, we just warn and
    //      proceed to CPU. sessionOptions_ stays clean.
    //   2) USE_CUDA build flag ON, useCuda=true, runtime CUDA OK:
    //      AppendExecutionProvider_CUDA + Session ctor succeed -> GPU path.
    //   3) USE_CUDA build flag ON, useCuda=true, runtime CUDA FAILS:
    //      Either AppendExecutionProvider_CUDA throws (CUDA driver/
    //      libraries missing) OR Session ctor throws (no compatible GPU,
    //      cuDNN version mismatch, etc.). We catch, log the reason,
    //      reset session_, and fall through to the CPU retry below.
    //
    //  For the CPU retry we construct a FRESH SessionOptions. The original
    //  sessionOptions_ may have had AppendExecutionProvider_CUDA called on
    //  it before the exception was thrown, and ORT exposes no way to
    //  un-append an EP. Using the original would re-trigger the same
    //  failure on retry.
    // ------------------------------------------------------------------
    bool cudaUsed      = false;
    bool cudaAttempted = false;

#ifdef USE_CUDA
    if (useCuda) {
        cudaAttempted = true;
        try {
            OrtCUDAProviderOptions cudaOptions;
            sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
            session_ = std::make_unique<Ort::Session>(
                env_, ORT_MODEL_PATH(modelPath), sessionOptions_);
            cudaUsed = true;
            std::cout << "Using CUDA" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "[DA3] CUDA initialization failed: " << e.what()
                      << "  -- falling back to CPU" << std::endl;
            session_.reset();
        }
    }
#else
    if (useCuda) {
        std::cerr << "Warning: CUDA not available (USE_CUDA=OFF build),"
                  << " using CPU" << std::endl;
    }
#endif

    if (!session_) {
        if (cudaAttempted) {
            // CUDA was attempted and failed. Use a FRESH SessionOptions
            // for CPU retry (see rationale in the long comment above).
            Ort::SessionOptions cpuOpts;
            cpuOpts.SetIntraOpNumThreads(4);
            cpuOpts.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = std::make_unique<Ort::Session>(
                env_, ORT_MODEL_PATH(modelPath), cpuOpts);
        } else {
            // CUDA was never attempted (useCuda=false, or USE_CUDA=OFF
            // build). The member sessionOptions_ is clean -- reuse it so
            // the binary-level behaviour exactly matches the legacy path.
            session_ = std::make_unique<Ort::Session>(
                env_, ORT_MODEL_PATH(modelPath), sessionOptions_);
        }
    }

    std::cout << "Depth Anything V3 loaded successfully"
              << (cudaUsed ? " (CUDA)" : " (CPU)") << std::endl;
}

void DepthAnythingV3::printModelInfo() const {
    std::cout << "=== Depth Anything V3 Model Info ===" << std::endl;
    std::cout << "Input size: " << inputSize_ << "x" << inputSize_ << std::endl;

    auto printTensor = [](size_t idx,
                          const char* name,
                          const std::vector<int64_t>& shape,
                          ONNXTensorElementDataType dtype) {
        std::cout << "  [" << idx << "] " << name << ": [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]  dtype=" << static_cast<int>(dtype) << std::endl;
    };

    size_t numInputs = session_->GetInputCount();
    std::cout << "\nInputs (" << numInputs << "):" << std::endl;
    for (size_t i = 0; i < numInputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        Ort::TypeInfo typeInfo = session_->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();
        ONNXTensorElementDataType dtype = tensorInfo.GetElementType();
        printTensor(i, name.get(), shape, dtype);
    }

    size_t numOutputs = session_->GetOutputCount();
    std::cout << "\nOutputs (" << numOutputs << "):" << std::endl;
    for (size_t i = 0; i < numOutputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        Ort::TypeInfo typeInfo = session_->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();
        ONNXTensorElementDataType dtype = tensorInfo.GetElementType();
        printTensor(i, name.get(), shape, dtype);
    }
    std::cout << std::endl;
}

void DepthAnythingV3::dumpAllOutputs(
    const std::vector<uint8_t>& imageData,
    int width, int height)
{
    auto inputData = preprocess(imageData, width, height);

    std::vector<int64_t> inputShape = {1, 1, 3, inputSize_, inputSize_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size());

    std::vector<Ort::AllocatedStringPtr> outNamesAlloc;
    std::vector<const char*> outNamesRaw;
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        outNamesAlloc.push_back(session_->GetOutputNameAllocated(i, allocator_));
        outNamesRaw.push_back(outNamesAlloc.back().get());
    }

    const char* inputNames[] = {"pixel_values"};
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outNamesRaw.data(), outNamesRaw.size());

    std::cout << "\n=== Full Output Dump ===" << std::endl;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        size_t elemCount = info.GetElementCount();

        std::cout << "\n[" << i << "] " << outNamesRaw[i] << " shape=[";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j] << (j + 1 < shape.size() ? "," : "");
        }
        std::cout << "] elements=" << elemCount << std::endl;

        float* data = outputs[i].GetTensorMutableData<float>();
        std::string name = outNamesRaw[i];

        if (name == "intrinsics" || name == "extrinsics") {
            size_t n = std::min<size_t>(elemCount, 32);
            std::cout << "  raw values: ";
            for (size_t k = 0; k < n; ++k) std::cout << data[k] << " ";
            if (n < elemCount) std::cout << "... (+" << (elemCount - n) << ")";
            std::cout << std::endl;
        } else {
            if (elemCount == 0) continue;
            float minV = data[0], maxV = data[0];
            double sumV = 0;
            for (size_t k = 0; k < elemCount; ++k) {
                if (data[k] < minV) minV = data[k];
                if (data[k] > maxV) maxV = data[k];
                sumV += data[k];
            }
            std::cout << "  min=" << minV << " max=" << maxV
                      << " mean=" << (sumV / elemCount) << std::endl;
        }
    }
    std::cout << "=== End Dump ===\n" << std::endl;
}

std::vector<float> DepthAnythingV3::preprocess(
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

std::vector<float> DepthAnythingV3::predict(
    const std::vector<uint8_t>& imageData,
    int width,
    int height
) {
    img::Timer timer;

    auto inputData = preprocess(imageData, width, height);
    std::cout << "Preprocessing: " << timer.elapsedMs() << " ms" << std::endl;
    timer.reset();

    std::vector<int64_t> inputShape = {1, 1, 3, inputSize_, inputSize_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );

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

    auto& outputTensor = outputTensors[0];
    auto* outputData = outputTensor.GetTensorMutableData<float>();
    auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();

    int outH = static_cast<int>(outputShape[outputShape.size() - 2]);
    int outW = static_cast<int>(outputShape[outputShape.size() - 1]);

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

DepthResult DepthAnythingV3::predictFull(
    const std::vector<uint8_t>& imageData,
    int width,
    int height
) {
    DepthResult result;
    result.width = width;
    result.height = height;

    img::Timer timer;

    auto inputData = preprocess(imageData, width, height);
    std::cout << "Preprocessing: " << timer.elapsedMs() << " ms" << std::endl;
    timer.reset();

    std::vector<int64_t> inputShape = {1, 1, 3, inputSize_, inputSize_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size());

    std::vector<Ort::AllocatedStringPtr> outNamesAlloc;
    std::vector<const char*> outNamesRaw;
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        outNamesAlloc.push_back(session_->GetOutputNameAllocated(i, allocator_));
        outNamesRaw.push_back(outNamesAlloc.back().get());
    }

    const char* inputNames[] = {"pixel_values"};
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outNamesRaw.data(), outNamesRaw.size());

    std::cout << "Inference: " << timer.elapsedMs() << " ms" << std::endl;
    timer.reset();

    auto resizeBilinear = [](const float* src, int sw, int sh,
                             std::vector<float>& dst, int dw, int dh) {
        dst.resize(static_cast<size_t>(dw) * dh);
        float scaleX = static_cast<float>(sw) / dw;
        float scaleY = static_cast<float>(sh) / dh;
        for (int y = 0; y < dh; ++y) {
            for (int x = 0; x < dw; ++x) {
                float srcXf = x * scaleX;
                float srcYf = y * scaleY;
                int x0 = static_cast<int>(srcXf);
                int y0 = static_cast<int>(srcYf);
                int x1 = std::min(x0 + 1, sw - 1);
                int y1 = std::min(y0 + 1, sh - 1);
                float fx = srcXf - x0;
                float fy = srcYf - y0;
                float v00 = src[y0 * sw + x0];
                float v10 = src[y0 * sw + x1];
                float v01 = src[y1 * sw + x0];
                float v11 = src[y1 * sw + x1];
                dst[y * dw + x] = (1 - fx) * (1 - fy) * v00 +
                                  fx * (1 - fy) * v10 +
                                  (1 - fx) * fy * v01 +
                                  fx * fy * v11;
            }
        }
    };

    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string name = outNamesRaw[i];
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        size_t elemCount = info.GetElementCount();
        float* data = outputs[i].GetTensorMutableData<float>();

        if (name == "predicted_depth") {
            int outH = static_cast<int>(shape[shape.size() - 2]);
            int outW = static_cast<int>(shape[shape.size() - 1]);
            resizeBilinear(data, outW, outH, result.depth, width, height);
        } else if (name == "confidence") {
            int outH = static_cast<int>(shape[shape.size() - 2]);
            int outW = static_cast<int>(shape[shape.size() - 1]);
            resizeBilinear(data, outW, outH, result.confidence, width, height);
            result.hasConfidence = true;
        } else if (name == "intrinsics" && elemCount >= 9) {
            float fx = data[0];
            float cx = data[2];
            float fy = data[4];
            float cy = data[5];
            float sx = static_cast<float>(width)  / static_cast<float>(inputSize_);
            float sy = static_cast<float>(height) / static_cast<float>(inputSize_);
            result.intrinsics.fx = fx * sx;
            result.intrinsics.fy = fy * sy;
            result.intrinsics.cx = cx * sx;
            result.intrinsics.cy = cy * sy;
            result.hasIntrinsics = true;
        }
    }

    std::cout << "Postprocessing: " << timer.elapsedMs() << " ms" << std::endl;

    if (result.hasIntrinsics) {
        std::cout << "[DA3] Intrinsics (image res " << width << "x" << height << "):"
                  << " fx=" << result.intrinsics.fx
                  << " fy=" << result.intrinsics.fy
                  << " cx=" << result.intrinsics.cx
                  << " cy=" << result.intrinsics.cy << std::endl;
    }
    if (!result.depth.empty()) {
        auto mm = std::minmax_element(result.depth.begin(), result.depth.end());
        std::cout << "[DA3] Depth range (metric): " << *mm.first << " .. " << *mm.second << std::endl;
    }
    if (result.hasConfidence && !result.confidence.empty()) {
        auto mm = std::minmax_element(result.confidence.begin(), result.confidence.end());
        std::cout << "[DA3] Confidence range: " << *mm.first << " .. " << *mm.second << std::endl;
    }

    return result;
}

}
