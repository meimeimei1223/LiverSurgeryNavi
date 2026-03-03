#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace sam2 {

/**
 * @brief 点プロンプトの種類
 */
enum class PointLabel {
    Background = 0,
    Foreground = 1
};

/**
 * @brief 点プロンプト
 */
struct PointPrompt {
    float x;
    float y;
    PointLabel label;
    
    PointPrompt(float x_, float y_, PointLabel label_ = PointLabel::Foreground)
        : x(x_), y(y_), label(label_) {}
};

/**
 * @brief セグメンテーション結果
 */
struct SegmentationResult {
    std::vector<uint8_t> mask;  // 二値マスク（0/255）
    int width = 0;
    int height = 0;
    float score = 0.0f;
};

/**
 * @brief SAM2 ONNX推論クラス（エンコーダー/デコーダー分離、OpenCVなし）
 */
class SAM2Segmentor {
public:
    /**
     * @brief コンストラクタ
     * @param encoderPath エンコーダーONNXモデルのパス
     * @param decoderPath デコーダーONNXモデルのパス
     * @param useCuda CUDAを使用するか
     */
    SAM2Segmentor(
        const std::string& encoderPath,
        const std::string& decoderPath,
        bool useCuda = false
    );
    
    ~SAM2Segmentor() = default;
    
    /**
     * @brief 画像を設定してエンコード
     * @param imageData RGB画像データ
     * @param width 画像幅
     * @param height 画像高さ
     */
    void setImage(
        const std::vector<uint8_t>& imageData,
        int width,
        int height
    );
    
    /**
     * @brief セグメンテーションを実行
     * @param points 点プロンプトのリスト
     * @return セグメンテーション結果
     */
    SegmentationResult predict(const std::vector<PointPrompt>& points);
    
    /**
     * @brief 入力サイズを取得
     */
    int getInputSize() const { return inputSize_; }
    
    /**
     * @brief モデル情報を表示
     */
    void printModelInfo() const;

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> encoderSession_;
    std::unique_ptr<Ort::Session> decoderSession_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    int inputSize_ = 1024;
    int imageWidth_ = 0;
    int imageHeight_ = 0;
    
    // キャッシュされた特徴量
    std::vector<float> imageEmbed_;
    std::vector<float> highResFeats0_;
    std::vector<float> highResFeats1_;
    bool hasEncodedImage_ = false;
    
    // 前処理
    std::vector<float> preprocess(
        const std::vector<uint8_t>& imageData,
        int srcWidth,
        int srcHeight
    );
    
    // エンコード実行
    void encodeImage(const std::vector<float>& inputData);
    
    // デコード実行
    SegmentationResult decode(const std::vector<PointPrompt>& points);
};

} // namespace sam2
