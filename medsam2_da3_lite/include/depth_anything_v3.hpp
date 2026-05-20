#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace depth {

struct CameraK {
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;
    bool valid() const { return fx > 0.0f && fy > 0.0f; }
};

struct DepthResult {
    std::vector<float> depth;
    std::vector<float> confidence;
    CameraK intrinsics;
    int width = 0;
    int height = 0;
    bool hasConfidence = false;
    bool hasIntrinsics = false;
};

class DepthAnythingV3 {
public:
    DepthAnythingV3(const std::string& modelPath, bool useCuda = false);
    ~DepthAnythingV3() = default;

    std::vector<float> predict(
        const std::vector<uint8_t>& imageData,
        int width,
        int height
    );

    std::vector<uint8_t> predictNormalized(
        const std::vector<uint8_t>& imageData,
        int width,
        int height,
        bool invert = true
    );

    DepthResult predictFull(
        const std::vector<uint8_t>& imageData,
        int width,
        int height
    );

    int getInputSize() const { return inputSize_; }

    void printModelInfo() const;
    void dumpAllOutputs(const std::vector<uint8_t>& imageData, int width, int height);

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    int inputSize_ = 518;

    std::vector<float> preprocess(
        const std::vector<uint8_t>& imageData,
        int srcWidth,
        int srcHeight
    );
};

}
