#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include "PlatformCompat.h"
#include <GL/glew.h>

#include "mCutMesh.h"
#include "MeshDrawing.h"
#include "DepthRunner.h"
#include "PathConfig.h"

// =========================================================================
// デフォルトセグメンテーションポイント
// =========================================================================
inline std::vector<DepthRunnerPoint> createDefaultSegPoints(int imgWidth, int imgHeight) {
    std::vector<DepthRunnerPoint> points;
    float cx = imgWidth / 2.0f;
    float cy = imgHeight / 2.0f;
    float marginX = imgWidth * 0.1f;
    float marginY = imgHeight * 0.1f;

    points.emplace_back(cx, cy, true);                                    // 前景:中心
    points.emplace_back(marginX, marginY, false);                         // 背景:左上
    points.emplace_back(imgWidth - marginX, marginY, false);              // 背景:右上
    points.emplace_back(marginX, imgHeight - marginY, false);             // 背景:左下
    points.emplace_back(imgWidth - marginX, imgHeight - marginY, false);  // 背景:右下
    return points;
}

inline void fallbackLoadExistingOutput(mCutMesh* screenMesh,
                                       int gw, float scale, float depthScale) {
    std::string basePath = DEPTH_OUTPUT_PATH;
    std::string origPath = basePath + "/original.jpg";

    if (!std::filesystem::exists(origPath)) {
        std::cerr << "[Fallback] No existing output: " << origPath << std::endl;
        std::cerr << "[Fallback] Using empty flat grid as placeholder." << std::endl;
        int gh = gw * 9 / 16;
        screenMesh->loadedImageWidth  = 1280;
        screenMesh->loadedImageHeight = 720;
        screenMesh->generateGridPlaneWithSides(gw, gh, 0.05f);
        for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
            screenMesh->mVertices[i] *= scale;
        setUp(*screenMesh);
        return;
    }

    screenMesh->loadTextureFromFile(origPath.c_str());
    if (screenMesh->loadedImageWidth <= 0 || screenMesh->loadedImageHeight <= 0) {
        std::cerr << "[Fallback] Failed to load texture, using flat grid." << std::endl;
        int gh = gw * 9 / 16;
        screenMesh->loadedImageWidth  = 1280;
        screenMesh->loadedImageHeight = 720;
        screenMesh->generateGridPlaneWithSides(gw, gh, 0.05f);
        for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
            screenMesh->mVertices[i] *= scale;
        setUp(*screenMesh);
        return;
    }

    int gh = gw * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
    std::string depthPath = basePath + "/depth_masked_renorm.png";
    if (std::filesystem::exists(depthPath)) {
        screenMesh->loadDepthImage(depthPath, screenMesh->loadedImageWidth, screenMesh->loadedImageHeight);
        std::vector<float> depths = screenMesh->calculateNormalizedDepth(gw, gh, 0.99f, 0.99f);
        screenMesh->generateGridPlaneWithDepth(gw, gh, depths, 0.05f, depthScale);
    } else {
        std::cerr << "[Fallback] No depth image, using flat grid with texture." << std::endl;
        screenMesh->generateGridPlaneWithSides(gw, gh, 0.05f);
    }

    for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
        screenMesh->mVertices[i] *= scale;
    setUp(*screenMesh);
}

// =========================================================================
// screenMesh 初期化（DepthRunner 使用）
// =========================================================================
inline void initScreenMeshWithDepthRunner(DepthRunner& depthRunner,
                                          mCutMesh*& screenMesh,
                                          bool cameraUse,
                                          std::function<void(float, const char*)> progressCb = nullptr) {
    depthRunner.printDiagnostics();

    screenMesh = new mCutMesh();
    int gw = 128;
    float scale = 10.0f;
    float depthScale = 0.3f;

    if (cameraUse) {
        screenMesh->initCamera(0, 1280, 720);
        int gh = gw * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
        screenMesh->generateGridPlaneWithSides(gw, gh, 0.05f);
        for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
            screenMesh->mVertices[i] *= scale;
        setUp(*screenMesh);

    } else {
        if (depthRunner.isAvailable() && depthRunner.areModelsAvailable()) {
            std::cout << "[Init] Running depth estimation..." << std::endl;
            screenMesh->loadTextureFromFile(gDepthInputImage.c_str());
            auto segPoints = createDefaultSegPoints(
                screenMesh->loadedImageWidth, screenMesh->loadedImageHeight);
            bool ok = DepthRunnerIntegration::updateScreenMeshDepth(
                depthRunner, gDepthInputImage, screenMesh,
                gw, scale, depthScale, segPoints,
                [](mCutMesh& mesh) { setUp(mesh); },
                progressCb
                );
            if (!ok) {
                std::cerr << "[Init] DepthRunner failed, trying fallback..." << std::endl;
                fallbackLoadExistingOutput(screenMesh, gw, scale, depthScale);
            }
        } else {
            std::cout << "[Init] DepthRunner not available, loading existing output..." << std::endl;
            fallbackLoadExistingOutput(screenMesh, gw, scale, depthScale);
        }
    }
}

// =========================================================================
// CameraPreview クラス
// =========================================================================
class CameraPreview {
public:
    bool active = false;
    bool frozen = false;
    std::string frozenCapturePath;
    SimpleCamera camera;
    std::vector<unsigned char> frame;
    int width = 0;
    int height = 0;

    // --- 開始 ---
    bool start(mCutMesh* screenMesh,
               int cameraIndex = 0, int reqWidth = 1280, int reqHeight = 720) {

        if (active) {
            std::cout << "[Preview] Already active" << std::endl;
            return true;
        }

        std::cout << "\n=== Starting Camera Preview ===" << std::endl;

        if (!camera.open(cameraIndex, reqWidth, reqHeight)) {
            std::cerr << "[Preview] Failed to open camera" << std::endl;
            return false;
        }

        width  = camera.getWidth();
        height = camera.getHeight();
        frame.resize(width * height * 3);

        // 数フレーム捨てる（露出安定のため）
        for (int i = 0; i < 5; i++) {
            camera.captureFrame(frame);
            usleep(30000);
        }

        // screenMeshのテクスチャを作成
        if (screenMesh->textureID == 0) {
            glGenTextures(1, &screenMesh->textureID);
        }
        glBindTexture(GL_TEXTURE_2D, screenMesh->textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        screenMesh->hasTexture = true;
        screenMesh->loadedImageWidth  = width;
        screenMesh->loadedImageHeight = height;

        // フラットグリッドに再生成（深度なし・プレビュー用）
        int gw = 128;
        float scale = 10.0f;
        int gh = gw * height / width;
        screenMesh->generateGridPlaneWithSides(gw, gh, 0.05f);
        for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
            screenMesh->mVertices[i] *= scale;
        setUp(*screenMesh);

        active = true;

        std::cout << "[Preview] Active: " << width << "x" << height << std::endl;
        std::cout << "[Preview] Press I to capture and run depth estimation" << std::endl;

        return true;
    }

    // --- 停止 ---
    void stop() {
        if (!active) return;
        camera.close();
        active = false;
        std::cout << "[Preview] Stopped" << std::endl;
    }

    // --- clearFrozen ---
    void clearFrozen() {
        frozen = false;
        frozenCapturePath.clear();
    }

    // =========================================================================
    // loadLocalImageAsFrozen
    //   ローカル画像ファイルを frozen 状態としてロードする。
    //   カメラの captureAndFreeze() 後と同じ状態になり、
    //   既存の FG/BG クリック → Key I/K フローがそのまま使える。
    //
    //   呼び出し元: ドラッグ＆ドロップ / ファイルピッカー
    // =========================================================================
    bool loadLocalImageAsFrozen(mCutMesh* screenMesh, const std::string& imagePath) {
        // カメラが動いていたら停止
        if (active) {
            stop();
        }

        // 既存の frozen をクリア
        if (frozen) {
            clearFrozen();
        }

        std::cout << "\n=== Loading Local Image as Frozen ===" << std::endl;
        std::cout << "[FileDrop] Path: " << imagePath << std::endl;

        // stb_image でテクスチャ読み込み
        screenMesh->loadTextureFromFile(imagePath.c_str());

        int imgW = screenMesh->loadedImageWidth;
        int imgH = screenMesh->loadedImageHeight;

        if (imgW <= 0 || imgH <= 0) {
            std::cerr << "[FileDrop] Failed to load image: " << imagePath << std::endl;
            return false;
        }

        // フラットメッシュ生成（カメラ freeze と同じ処理）
        int gw = 128;
        float scale = 10.0f;
        int gh = gw * imgH / imgW;
        screenMesh->generateGridPlaneWithSides(gw, gh, 0.05f);
        for (size_t i = 0; i < screenMesh->mVertices.size(); i++)
            screenMesh->mVertices[i] *= scale;
        setUp(*screenMesh);

        // frozen 状態にする
        // frozenCapturePath に画像パスをセット → runDepthFromFrozen() がこれを使う
        frozen = true;
        frozenCapturePath = imagePath;

        // width/height を更新（runDepthFullFromFrozen が使う）
        width  = imgW;
        height = imgH;

        std::cout << "[FileDrop] Loaded: " << imgW << "x" << imgH << std::endl;
        std::cout << "[FileDrop] === Image Frozen ===" << std::endl;
        std::cout << "  Left-click  = FG (object)" << std::endl;
        std::cout << "  Right-click = BG (background)" << std::endl;
        std::cout << "  Z           = Undo last point" << std::endl;
        std::cout << "  Key I       = Run depth WITH segmentation" << std::endl;
        std::cout << "  Key K       = Run depth WITHOUT segmentation" << std::endl;

        return true;
    }

    // --- captureAndFreeze ---
    bool captureAndFreeze(mCutMesh* screenMesh) {
        if (!active) {
            std::cerr << "[Freeze] Camera not active" << std::endl;
            return false;
        }

        std::cout << "\n=== Capturing Frame (Freeze) ===" << std::endl;

        int imgW = width;
        int imgH = height;

        // capture frame
        std::vector<unsigned char> rawFrame(imgW * imgH * 3);
        camera.captureFrame(rawFrame);

        // stop camera
        stop();

        // mirror flip for PPM
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW / 2; x++) {
                int leftIdx  = (y * imgW + x) * 3;
                int rightIdx = (y * imgW + (imgW - 1 - x)) * 3;
                std::swap(rawFrame[leftIdx],     rawFrame[rightIdx]);
                std::swap(rawFrame[leftIdx + 1], rawFrame[rightIdx + 1]);
                std::swap(rawFrame[leftIdx + 2], rawFrame[rightIdx + 2]);
            }
        }

        // save PPM
        frozenCapturePath = DEPTH_OUTPUT_PATH + "camera_capture.ppm";
        if (!savePPM(frozenCapturePath, rawFrame.data(), imgW, imgH)) {
            std::cerr << "[Freeze] Failed to save PPM" << std::endl;
            return false;
        }
        std::cout << "[Freeze] Saved: " << frozenCapturePath << std::endl;

        // update texture (flip for OpenGL)
        int rowBytes = imgW * 3;
        std::vector<unsigned char> rowTemp(rowBytes);
        for (int y = 0; y < imgH / 2; y++) {
            int topIdx = y * rowBytes;
            int botIdx = (imgH - 1 - y) * rowBytes;
            memcpy(rowTemp.data(),     &rawFrame[topIdx], rowBytes);
            memcpy(&rawFrame[topIdx],  &rawFrame[botIdx], rowBytes);
            memcpy(&rawFrame[botIdx],  rowTemp.data(),    rowBytes);
        }

        glBindTexture(GL_TEXTURE_2D, screenMesh->textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        imgW, imgH,
                        GL_RGB, GL_UNSIGNED_BYTE, rawFrame.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        frozen = true;

        std::cout << "[Freeze] Image frozen. Click to place seg points, then press I." << std::endl;
        return true;
    }

    // --- runDepthFromFrozen ---
    bool runDepthFromFrozen(DepthRunner& depthRunner,
                            mCutMesh* screenMesh,
                            const std::vector<DepthRunnerPoint>& segPoints,
                            std::function<void(float, const char*)> progressCb = nullptr) {
        if (!frozen || frozenCapturePath.empty()) {
            std::cerr << "[Depth] No frozen capture available" << std::endl;
            return false;
        }

        std::cout << "\n=== Running Depth from Frozen Capture ===" << std::endl;

        std::cout << "[Depth] Seg points: " << segPoints.size() << std::endl;
        for (size_t i = 0; i < segPoints.size(); i++) {
            std::cout << "  [" << i << "] (" << segPoints[i].x << ", " << segPoints[i].y
                      << ") " << (segPoints[i].isForeground ? "FG" : "BG") << std::endl;
        }

        bool ok = DepthRunnerIntegration::updateScreenMeshDepth(
            depthRunner, frozenCapturePath, screenMesh,
            128, 10.0f, 0.3f, segPoints,
            [](mCutMesh& mesh) { setUp(mesh); },
            progressCb
            );

        std::cout << (ok ? "[Depth] Complete!" : "[Depth] Failed") << std::endl;

        clearFrozen();
        return ok;
    }

    // --- runDepthFullFromFrozen (no seg) ---
    bool runDepthFullFromFrozen(DepthRunner& depthRunner,
                                mCutMesh* screenMesh,
                                std::function<void(float, const char*)> progressCb = nullptr) {
        if (!frozen || frozenCapturePath.empty()) {
            std::cerr << "[Depth] No frozen capture available" << std::endl;
            return false;
        }

        std::cout << "\n=== Running Depth (NO SEGMENTATION) ===" << std::endl;

        auto dummyPts = createDefaultSegPoints(width, height);
        bool ok = DepthRunnerIntegration::updateScreenMeshDepthFullOnly(
            depthRunner, frozenCapturePath, screenMesh,
            128, 10.0f, 0.3f, dummyPts,
            [](mCutMesh& mesh) { setUp(mesh); },
            progressCb
            );

        std::cout << (ok ? "[Depth] Complete (no seg)!" : "[Depth] Failed") << std::endl;

        clearFrozen();
        return ok;
    }

    // --- プレビューフレーム更新（メインループから毎フレーム呼ぶ） ---
    void update(mCutMesh* screenMesh) {
        if (!active) return;
        if (!camera.isOpened()) return;

        if (camera.captureFrame(frame)) {
            // 上下反転
            int rowBytes = width * 3;
            std::vector<unsigned char> rowTemp(rowBytes);
            for (int y = 0; y < height / 2; y++) {
                int topIdx = y * rowBytes;
                int botIdx = (height - 1 - y) * rowBytes;
                memcpy(rowTemp.data(),    &frame[topIdx], rowBytes);
                memcpy(&frame[topIdx],    &frame[botIdx], rowBytes);
                memcpy(&frame[botIdx],    rowTemp.data(), rowBytes);
            }

            // 左右反転（鏡像）
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width / 2; x++) {
                    int leftIdx  = (y * width + x) * 3;
                    int rightIdx = (y * width + (width - 1 - x)) * 3;
                    std::swap(frame[leftIdx],     frame[rightIdx]);
                    std::swap(frame[leftIdx + 1], frame[rightIdx + 1]);
                    std::swap(frame[leftIdx + 2], frame[rightIdx + 2]);
                }
            }

            // テクスチャ更新
            glBindTexture(GL_TEXTURE_2D, screenMesh->textureID);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            width, height,
                            GL_RGB, GL_UNSIGNED_BYTE, frame.data());
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    // --- キャプチャして深度推定を実行 ---
    bool captureAndRunDepth(DepthRunner& depthRunner, mCutMesh* screenMesh) {
        if (!active) {
            std::cerr << "[Capture] Preview not active. Press U first." << std::endl;
            return false;
        }

        std::cout << "\n=== Capturing Frame + Running Depth ===" << std::endl;

        int imgW = width;
        int imgH = height;

        // 最新フレームを取得（生データ）
        std::vector<unsigned char> rawFrame(imgW * imgH * 3);
        camera.captureFrame(rawFrame);

        // プレビュー停止
        stop();

        // 左右反転（プレビューと同じ鏡像にする）
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW / 2; x++) {
                int leftIdx  = (y * imgW + x) * 3;
                int rightIdx = (y * imgW + (imgW - 1 - x)) * 3;
                std::swap(rawFrame[leftIdx],     rawFrame[rightIdx]);
                std::swap(rawFrame[leftIdx + 1], rawFrame[rightIdx + 1]);
                std::swap(rawFrame[leftIdx + 2], rawFrame[rightIdx + 2]);
            }
        }

        // PPM保存（鏡像済みデータ）
        std::string capturePath = DEPTH_OUTPUT_PATH + "camera_capture.ppm";
        if (!savePPM(capturePath, rawFrame.data(), imgW, imgH)) {
            std::cerr << "[Capture] Failed to save" << std::endl;
            return false;
        }
        std::cout << "[Capture] Saved: " << capturePath << std::endl;

        // 深度推定実行
        auto segPoints = createDefaultSegPoints(imgW, imgH);
        bool ok = DepthRunnerIntegration::updateScreenMeshDepth(
            depthRunner, capturePath, screenMesh,
            128, 10.0f, 0.3f, segPoints,
            [](mCutMesh& mesh) { setUp(mesh); }
            );

        std::cout << (ok ? "[Capture] Depth estimation complete!"
                         : "[Capture] Depth estimation failed") << std::endl;
        return ok;
    }

    // --- captureAndRunDepthWithPoints ---
    bool captureAndRunDepthWithPoints(DepthRunner& depthRunner,
                                      mCutMesh* screenMesh,
                                      const std::vector<DepthRunnerPoint>& segPoints,
                                      std::function<void(float, const char*)> progressCb = nullptr) {
        if (!active) {
            std::cerr << "[Capture] Preview not active. Press U first." << std::endl;
            return false;
        }

        std::cout << "\n=== Capturing Frame + Running Depth (with user points) ===" << std::endl;

        int imgW = width;
        int imgH = height;

        std::vector<unsigned char> rawFrame(imgW * imgH * 3);
        camera.captureFrame(rawFrame);

        stop();

        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW / 2; x++) {
                int leftIdx  = (y * imgW + x) * 3;
                int rightIdx = (y * imgW + (imgW - 1 - x)) * 3;
                std::swap(rawFrame[leftIdx],     rawFrame[rightIdx]);
                std::swap(rawFrame[leftIdx + 1], rawFrame[rightIdx + 1]);
                std::swap(rawFrame[leftIdx + 2], rawFrame[rightIdx + 2]);
            }
        }

        std::string capturePath = DEPTH_OUTPUT_PATH + "camera_capture.ppm";
        if (!savePPM(capturePath, rawFrame.data(), imgW, imgH)) {
            std::cerr << "[Capture] Failed to save" << std::endl;
            return false;
        }
        std::cout << "[Capture] Saved: " << capturePath << std::endl;

        std::cout << "[Capture] Seg points: " << segPoints.size() << std::endl;
        for (size_t i = 0; i < segPoints.size(); i++) {
            std::cout << "  [" << i << "] (" << segPoints[i].x << ", " << segPoints[i].y
                      << ") " << (segPoints[i].isForeground ? "FG" : "BG") << std::endl;
        }

        bool ok = DepthRunnerIntegration::updateScreenMeshDepth(
            depthRunner, capturePath, screenMesh,
            128, 10.0f, 0.3f, segPoints,
            [](mCutMesh& mesh) { setUp(mesh); },
            progressCb
            );

        std::cout << (ok ? "[Capture] Depth estimation complete!"
                         : "[Capture] Depth estimation failed") << std::endl;
        return ok;
    }

private:
    // --- RGBフレームをPPM形式で保存（外部ライブラリ不要） ---
    static bool savePPM(const std::string& path,
                        const unsigned char* data, int w, int h) {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) return false;
        fprintf(fp, "P6\n%d %d\n255\n", w, h);
        fwrite(data, 1, w * h * 3, fp);
        fclose(fp);
        return true;
    }
};
