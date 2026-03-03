#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace depth {

/**
 * @brief Depth Anything V3 ONNX推論クラス（OpenCVなし）
 */
class DepthAnythingV3 {
public:
    /**
     * @brief コンストラクタ
     * @param modelPath ONNXモデルパス
     * @param useCuda CUDAを使用するか
     */
    DepthAnythingV3(const std::string& modelPath, bool useCuda = false);
    ~DepthAnythingV3() = default;
    
    /**
     * @brief 深度推定（生データ）
     * @param imageData RGB画像データ
     * @param width 画像幅
     * @param height 画像高さ
     * @return 深度マップ（float）
     */
    std::vector<float> predict(
        const std::vector<uint8_t>& imageData,
        int width,
        int height
    );
    
    /**
     * @brief 深度推定（正規化済み、0-255）
     * @param imageData RGB画像データ
     * @param width 画像幅
     * @param height 画像高さ
     * @param invert true=近いほど明るい
     * @return 深度マップ（uint8_t、0-255）
     */
    std::vector<uint8_t> predictNormalized(
        const std::vector<uint8_t>& imageData,
        int width,
        int height,
        bool invert = true
    );
    
    /**
     * @brief 入力サイズ取得
     */
    int getInputSize() const { return inputSize_; }
    
    /**
     * @brief モデル情報を表示
     */
    void printModelInfo() const;

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

} // namespace depth
