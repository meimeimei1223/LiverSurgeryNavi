#pragma once
// CameraPreview.h — USBカメラプレビュー＋キャプチャ機能
// SimpleCamera.hpp のラッパー。ライブビュー、静止画キャプチャ、JPEG保存。

#include <vector>
#include <string>
#include <iostream>

#include <GL/glew.h>
#include "SimpleCamera.hpp"
#include "AR.h"
#include "stb_image_write.h"
#include "PathConfig.h"

class SimpleCameraPreview {
public:
    bool active = false;
    bool captured = false;
    SimpleCamera camera;
    std::vector<unsigned char> frame;
    std::vector<unsigned char> originalFrame;
    std::vector<unsigned char> capturedFrame;
    std::vector<unsigned char> capturedOriginal;
    int width = 0;
    int height = 0;
    GLuint textureID = 0;

    bool start() {
        if (active) return true;
        std::cout << "[Camera] Trying to open camera device 0..." << std::endl;
        auto devices = SimpleCamera::listDevices();
        std::cout << "[Camera] Available devices:" << std::endl;
        for (const auto& dev : devices)
            std::cout << "  - " << dev << std::endl;

        if (!camera.open(0, 1280, 720)) {
            std::cerr << "[Camera] Failed to open camera device 0" << std::endl;
            for (int i = 1; i <= 3; i++) {
                std::cout << "[Camera] Trying device " << i << "..." << std::endl;
                if (camera.open(i, 1280, 720)) {
                    std::cout << "[Camera] Success with device " << i << std::endl;
                    break;
                }
            }
            if (!camera.isOpened()) {
                std::cerr << "[Camera] Could not open any camera device" << std::endl;
                return false;
            }
        }

        width = camera.getWidth();
        height = camera.getHeight();
        frame.resize(width * height * 3);
        originalFrame.resize(width * height * 3);
        for (size_t i = 0; i < frame.size(); i++) {
            frame[i] = 128;
            originalFrame[i] = 128;
        }

        active = true;
        captured = false;
        std::cout << "[Camera] Started (" << width << "x" << height << ")" << std::endl;
        return true;
    }

    void stop() {
        if (!active) return;
        camera.close();
        active = false;
        captured = false;
        std::cout << "[Camera] Stopped" << std::endl;
    }

    void captureCurrentFrame() {
        if (!active) return;
        capturedFrame = frame;
        capturedOriginal = originalFrame;
        captured = true;
        std::cout << "[Camera] Frame captured" << std::endl;
    }

    void releaseCapture() {
        captured = false;
        std::cout << "[Camera] Returned to live view" << std::endl;
    }

    bool capture(AR::Background& arBg) {
        if (!active) return false;
        if (captured) {
            arBg.updateTextureData(capturedFrame.data(), width, height, 3);
            return true;
        }

        bool result = camera.captureFrame(originalFrame);
        if (!result) {
            static int errorCount = 0;
            if (errorCount++ < 5)
                std::cerr << "[Camera] captureFrame failed" << std::endl;
            return false;
        }

        // ミラー表示用に左右反転＋上下反転
        frame.resize(originalFrame.size());
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int srcX = width - 1 - x;
                int srcY = height - 1 - y;
                int dstIdx = (y * width + x) * 3;
                int srcIdx = (srcY * width + srcX) * 3;
                frame[dstIdx + 0] = originalFrame[srcIdx + 0];
                frame[dstIdx + 1] = originalFrame[srcIdx + 1];
                frame[dstIdx + 2] = originalFrame[srcIdx + 2];
            }
        }
        arBg.updateTextureData(frame.data(), width, height, 3);
        return true;
    }

    std::string saveForDepthEstimation() {
        if (!active) return "";
        const auto& targetFrame = captured ? capturedOriginal : originalFrame;
        if (targetFrame.empty()) return "";
        std::string fullPath = DEPTH_OUTPUT_PATH + "camera_frame_temp.jpg";
        if (stbi_write_jpg(fullPath.c_str(), width, height, 3, targetFrame.data(), 90)) {
            std::cout << "[Camera] Saved depth input: " << fullPath << std::endl;
            return fullPath;
        }
        return "";
    }

    bool saveFrameAsJPEG(const std::string& path) {
        if (!active) return false;
        const auto& targetFrame = captured ? capturedOriginal : originalFrame;
        if (targetFrame.empty()) return false;
        return stbi_write_jpg(path.c_str(), width, height, 3, targetFrame.data(), 90) != 0;
    }

    void updateARBackground(AR::Background& arBg) {
        if (!active || frame.empty()) return;
        arBg.updateTextureData(frame.data(), width, height, 3);
    }
};
