// =============================================================================
//  AR.h
//  ---------------------------------------------------------------------------
//  AR (augmented-reality overlay) functionality, in two namespaces:
//
//    namespace AR     — Background renderer (fullscreen-quad camera image
//                       behind 3D scene, with embedded GLSL shaders).
//    namespace ARSave — Capture-to-FBO + ImGui preview window for saving
//                       the AR composite at the input image's native
//                       resolution.
//
//  Previously split as ARBackground.h + ARSave.h. Merged because they share
//  the same domain and ARSave depends on AR::Background.
//
//  Background design notes (preserved from ARBackground.h):
//    - The fullscreen quad lives at far plane (NDC z = 0.999) so depth-tested
//      meshes always draw on top. Depth TEST stays on; depth WRITE is turned
//      off while the background renders, so meshes compare against a clean
//      depth buffer from glClear rather than the constant 0.999 of the
//      background.
//    - No model/view/projection used — the quad is authored directly in NDC.
//    - UV is flipped vertically because stb_image loads top-down while OpenGL
//      textures address bottom-up.
//
//  Usage in main.cpp:
//    #include "AR.h"
//    static AR::Background g_arBg;
//    static ARSave::State  g_arSave;
//    static bool           g_arMode = false;
//
//    // after GL init:
//    g_arBg.initGL();
//    g_arBg.loadTexture(DEPTH_OUTPUT_PATH + "original.jpg");
//
//    // render loop, after glClear:
//    if (g_arMode) g_arBg.draw();
//    // then draw meshes as usual
//
//    // Save / preview:
//    ARSave::capture(g_arSave, ...);
//    ARSave::drawPreviewWindow(g_arSave, vpW, vpH);
// =============================================================================

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <filesystem>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "stb_image.h"
#include "stb_image_write.h"
#include "imgui.h"
#include "ShaderProgram.h"
#include "MeshDrawing.h"
#include "FullSphereCameraWithTarget.h"
#include "mCutMesh.h"


// =============================================================================
//  Part 1: namespace AR  —  Background renderer
// =============================================================================
namespace AR {

// -----------------------------------------------------------------------------
//  Embedded GLSL shaders
// -----------------------------------------------------------------------------
static const char* BG_VERT_SRC = R"GLSL(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    // z = 0.999 pushes the quad to the far end of the depth range;
    // any real-geometry fragment will pass the depth test and draw on top.
    gl_Position = vec4(aPos, 0.999, 1.0);
}
)GLSL";

static const char* BG_FRAG_SRC = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vUV);
}
)GLSL";

// -----------------------------------------------------------------------------
//  Background renderer
// -----------------------------------------------------------------------------
class Background {
public:
    GLuint programID = 0;
    GLuint textureID = 0;
    GLuint VAO = 0, VBO = 0;
    int    texW = 0, texH = 0;
    bool   ready = false;

    // -------------------------------------------------------------------------
    //  GL setup: compile shaders, create fullscreen quad
    // -------------------------------------------------------------------------
    bool initGL() {
        if (!compileProgram()) return false;
        if (!createQuad())      return false;
        std::cout << "[ARBackground] GL initialized (program="
                  << programID << ", VAO=" << VAO << ")" << std::endl;
        return true;
    }

    // -------------------------------------------------------------------------
    //  Load background image (RGB or RGBA)
    // -------------------------------------------------------------------------
    bool loadTexture(const std::string& path) {
        stbi_set_flip_vertically_on_load(1);    // OpenGL texture origin fix
        int ch = 0;
        unsigned char* data = stbi_load(path.c_str(), &texW, &texH, &ch, 3);
        stbi_set_flip_vertically_on_load(0);
        if (!data) {
            std::cerr << "[ARBackground] cannot load: " << path << std::endl;
            return false;
        }

        if (textureID) glDeleteTextures(1, &textureID);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     texW, texH, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        stbi_image_free(data);

        std::cout << "[ARBackground] loaded " << path
                  << "  (" << texW << "x" << texH
                  << ", " << ch << " channel)" << std::endl;
        ready = (textureID != 0 && programID != 0 && VAO != 0);
        return ready;
    }

    // -------------------------------------------------------------------------
    //  Update texture data (for live camera feed)
    // -------------------------------------------------------------------------
    void updateTextureData(unsigned char* data, int w, int h, int channels = 3) {
        if (!textureID) {
            glGenTextures(1, &textureID);
            std::cout << "[ARBackground] Generated new texture ID: " << textureID << std::endl;
        }

        texW = w;
        texH = h;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;

        // デバッグ：最初の数ピクセルを確認
        static int updateCount = 0;
        if (updateCount++ < 5 || updateCount % 100 == 0) {
            std::cout << "[ARBackground] Updating texture " << textureID
                      << " (" << w << "x" << h << ") First pixels: "
                      << (int)data[0] << "," << (int)data[1] << "," << (int)data[2] << std::endl;
        }

        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);

        ready = (textureID != 0 && programID != 0 && VAO != 0);
    }

    // -------------------------------------------------------------------------
    //  Draw the background (call after glClear, before mesh drawing)
    // -------------------------------------------------------------------------
    void draw() {
        if (!ready) return;

        // Preserve caller state we touch
        GLboolean prevDepthMask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
        GLboolean prevBlend = glIsEnabled(GL_BLEND);

        glDepthMask(GL_FALSE);   // don't touch depth buffer
        if (prevBlend) glDisable(GL_BLEND);

        glUseProgram(programID);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        GLint loc = glGetUniformLocation(programID, "uTex");
        glUniform1i(loc, 0);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        // Restore
        glDepthMask(prevDepthMask);
        if (prevBlend) glEnable(GL_BLEND);
    }

    ~Background() {
        if (programID) glDeleteProgram(programID);
        if (textureID) glDeleteTextures(1, &textureID);
        if (VAO)       glDeleteVertexArrays(1, &VAO);
        if (VBO)       glDeleteBuffers(1, &VBO);
    }

private:
    static GLuint compileOne(GLenum type, const char* src) {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024] = {0};
            glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            std::cerr << "[ARBackground] shader compile error:\n" << log << std::endl;
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    bool compileProgram() {
        GLuint vs = compileOne(GL_VERTEX_SHADER,   BG_VERT_SRC);
        if (!vs) return false;
        GLuint fs = compileOne(GL_FRAGMENT_SHADER, BG_FRAG_SRC);
        if (!fs) { glDeleteShader(vs); return false; }

        programID = glCreateProgram();
        glAttachShader(programID, vs);
        glAttachShader(programID, fs);
        glLinkProgram(programID);

        GLint linked = 0;
        glGetProgramiv(programID, GL_LINK_STATUS, &linked);
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (!linked) {
            char log[1024] = {0};
            glGetProgramInfoLog(programID, sizeof(log), nullptr, log);
            std::cerr << "[ARBackground] link error:\n" << log << std::endl;
            glDeleteProgram(programID);
            programID = 0;
            return false;
        }
        return true;
    }

    bool createQuad() {
        // Fullscreen NDC quad with UVs (0..1, origin bottom-left)
        const float verts[] = {
            // pos        // uv
            -1.f, -1.f,   0.f, 0.f,
             1.f, -1.f,   1.f, 0.f,
             1.f,  1.f,   1.f, 1.f,
            -1.f, -1.f,   0.f, 0.f,
             1.f,  1.f,   1.f, 1.f,
            -1.f,  1.f,   0.f, 1.f,
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        return (VAO != 0 && VBO != 0);
    }
};

} // namespace AR


// =============================================================================
//  Part 2: namespace ARSave  —  AR mode capture + ImGui preview
// =============================================================================
//  ARモード(Key A)表示状態を入力画像サイズでFBOキャプチャ+プレビュー
namespace ARSave {

struct State {
    GLuint previewTex = 0;
    bool   showPreview = false;
    int    previewW = 0;
    int    previewH = 0;
};

// ARモード表示をFBOキャプチャ（入力画像と同じサイズ）
inline bool capture(
    State& st,
    FullSphereCamera& orbitCam,
    ShaderProgram& shader,
    ShaderProgram& shaderCube,
    AR::Background& arBg,
    std::vector<mCutMesh*> organMeshes,
    const float* meshAlpha,
    const glm::vec3& objPos,
    int imgW, int imgH,
    const std::string& outputDir,
    int windowW, int windowH)
{
    if (imgW <= 0 || imgH <= 0) {
        std::cerr << "[ARSave] Invalid image size" << std::endl;
        return false;
    }

    GLuint fbo, fboColorTex, depthRbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &fboColorTex);
    glBindTexture(GL_TEXTURE_2D, fboColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgW, imgH, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, fboColorTex, 0);

    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, imgW, imgH);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, depthRbo);

    bool success = false;

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        glViewport(0, 0, imgW, imgH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- ARモードと同じ描画を再現 ---

        // 1) AR背景（元画像テクスチャ）
        arBg.draw();

        // 2) ARモード固定view: 原点から+Z方向
        glm::mat4 arView = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));

        // 3) intrinsicsベースprojection（入力画像サイズで構築）
        glm::mat4 arProj(0.0f);
        arProj[0][0] =  2.0f * orbitCam.fx / imgW;
        arProj[1][1] =  2.0f * orbitCam.fy / imgH;
        arProj[2][0] =  1.0f - 2.0f * orbitCam.cx / imgW;
        arProj[2][1] =  2.0f * orbitCam.cy / imgH - 1.0f;
        arProj[2][2] = -(orbitCam.farPlane + orbitCam.nearPlane)
                       / (orbitCam.farPlane - orbitCam.nearPlane);
        arProj[2][3] = -1.0f;
        arProj[3][2] = -2.0f * orbitCam.farPlane * orbitCam.nearPlane
                       / (orbitCam.farPlane - orbitCam.nearPlane);

        glm::mat4 arModel = glm::translate(glm::mat4(1.0f), objPos);

        // 4) 臓器メッシュのみ描画（screenMeshは含めない）
        std::vector<mCutMesh*> meshes;
        std::vector<glm::vec4> colors;
        const glm::vec4 defCol[] = {
            {0.8f,0.2f,0.2f,0.8f}, {0.2f,0.2f,0.8f,0.8f},
            {0.2f,0.5f,0.5f,0.8f}, {0.8f,0.5f,0.5f,0.8f},
            {0.2f,0.8f,0.5f,0.6f}, {0.2f,0.8f,0.2f,0.8f},
        };
        for (size_t i = 0; i < organMeshes.size() && i < 6; i++) {
            float a = meshAlpha ? meshAlpha[i] : 0.8f;
            if (organMeshes[i] && a > 0.01f) {
                meshes.push_back(organMeshes[i]);
                glm::vec4 c = defCol[i];
                c.a = a;
                colors.push_back(c);
            }
        }
        glm::vec3 camPos(0.0f);
        draw_AllmCutMeshes(meshes, shader, shaderCube,
                           camPos, colors,
                           arModel, arView, arProj, -1);

        // --- 読み取り＋保存 ---
        std::vector<unsigned char> pixels(imgW * imgH * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, imgW, imgH, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        int stride = imgW * 3;
        std::vector<unsigned char> flipped(pixels.size());
        for (int y = 0; y < imgH; y++)
            std::memcpy(&flipped[y * stride],
                        &pixels[(imgH - 1 - y) * stride], stride);

        // タイムスタンプ
        auto now = std::chrono::system_clock::now();
        auto tt  = std::chrono::system_clock::to_time_t(now);
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch()) % 1000;
        struct tm lt;
#ifdef _WIN32
        localtime_s(&lt, &tt);  // Windows
#else
        localtime_r(&tt, &lt);  // Unix/Linux
#endif
        char stamp[64];
        std::snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d_%03d",
                      lt.tm_year+1900, lt.tm_mon+1, lt.tm_mday,
                      lt.tm_hour, lt.tm_min, lt.tm_sec, (int)ms.count());

        std::string saveDir = outputDir;
        if (!saveDir.empty() && saveDir.back() != '/') saveDir += "/";
        saveDir += "screenshots/";
        std::filesystem::create_directories(saveDir);

        std::string path = saveDir + "ar_" + stamp + ".png";
        stbi_write_png(path.c_str(), imgW, imgH, 3, flipped.data(), stride);
        std::cout << "[ARSave] " << imgW << "x" << imgH << " -> "
                  << std::filesystem::absolute(path).string() << std::endl;

        // プレビューテクスチャ
        if (st.previewTex == 0) glGenTextures(1, &st.previewTex);
        glBindTexture(GL_TEXTURE_2D, st.previewTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgW, imgH, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, flipped.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        st.previewW = imgW;
        st.previewH = imgH;
        st.showPreview = true;
        success = true;
    } else {
        std::cerr << "[ARSave] FBO creation failed!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteTextures(1, &fboColorTex);
    glDeleteRenderbuffers(1, &depthRbo);
    glDeleteFramebuffers(1, &fbo);
    glViewport(0, 0, windowW, windowH);
    return success;
}

// ImGuiプレビューウィンドウ（×で閉じ可能）
inline void drawPreviewWindow(State& st, float viewportW, float windowH) {
    if (!st.showPreview || st.previewTex == 0) return;

    float prevW = viewportW * 0.5f;
    float prevH = prevW * (float)st.previewH / (float)st.previewW;
    float maxH = windowH * 0.6f;
    if (prevH > maxH) {
        prevH = maxH;
        prevW = prevH * (float)st.previewW / (float)st.previewH;
    }

    ImGui::SetNextWindowSize(ImVec2(prevW + 16, prevH + 50), ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(
        ImVec2(viewportW * 0.5f, windowH * 0.5f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.04f,0.04f,0.06f,0.95f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));

    if (ImGui::Begin("AR Screenshot", &st.showPreview,
                     ImGuiWindowFlags_NoCollapse)) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float iw = avail.x;
        float ih = iw * (float)st.previewH / (float)st.previewW;
        if (ih > avail.y) { ih = avail.y; iw = ih * (float)st.previewW / (float)st.previewH; }
        float off = (avail.x - iw) * 0.5f;
        if (off > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
        ImGui::Image((ImTextureID)(intptr_t)st.previewTex, ImVec2(iw, ih));
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

} // namespace ARSave
