#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <GL/glew.h>

#include "AppContext.h"

namespace MaskPicker {

// Resolve which point vector to operate on given the currently-active mask
// kind. Returning a reference keeps the call sites short and makes the
// semantics ("we always push to ONE list at a time") explicit.
inline std::vector<MaskPoint>& activePoints(AppContext& ctx) {
    return (ctx.activeMaskKind == MaskKind::Instrument)
                ? ctx.instrumentMaskPoints
                : ctx.maskPoints;
}
inline const std::vector<MaskPoint>& activePoints(const AppContext& ctx) {
    return (ctx.activeMaskKind == MaskKind::Instrument)
                ? ctx.instrumentMaskPoints
                : ctx.maskPoints;
}

inline const char* activeKindName(const AppContext& ctx) {
    return (ctx.activeMaskKind == MaskKind::Instrument) ? "Instrument" : "Liver";
}

inline void addFromScreen(AppContext& ctx, float sx, float sy, bool fg) {
    if (ctx.image.width <= 0 || ctx.image.height <= 0) return;
    
    // Image is displayed at full window size, so screen coords map directly to image pixels
    // But we need to consider that the image might have different aspect ratio
    // For now, assume image fills the window (as in AR background mode)
    float u = sx * (float)ctx.image.width  / (float)ctx.windowW;
    float v = sy * (float)ctx.image.height / (float)ctx.windowH;
    
    // Clamp to image bounds
    u = std::max(0.0f, std::min(u, (float)ctx.image.width - 1));
    v = std::max(0.0f, std::min(v, (float)ctx.image.height - 1));
    
    auto& dst = activePoints(ctx);
    dst.push_back({u, v, fg});

    int nFg = 0, nBg = 0;
    for (const auto& p : dst) (p.fg ? nFg : nBg)++;
    std::cout << "[MaskPicker] " << activeKindName(ctx) << " "
              << (fg ? "FG" : "BG")
              << " (" << (int)u << "," << (int)v << ")"
              << "  fg=" << nFg << " bg=" << nBg << std::endl;
}

inline void undo(AppContext& ctx) {
    auto& dst = activePoints(ctx);
    if (dst.empty()) return;
    bool wasFg = dst.back().fg;
    dst.pop_back();
    std::cout << "[MaskPicker] " << activeKindName(ctx)
              << " undo " << (wasFg ? "FG" : "BG")
              << ", remaining=" << dst.size() << std::endl;
}

inline void clear(AppContext& ctx) {
    // Clear both lists: a global "start over" should not leave stray
    // instrument points behind when the user is editing Liver.
    bool any = !ctx.maskPoints.empty() || !ctx.instrumentMaskPoints.empty();
    if (!any) return;
    ctx.maskPoints.clear();
    ctx.instrumentMaskPoints.clear();
    std::cout << "[MaskPicker] cleared (both Liver and Instrument)" << std::endl;
}

class Renderer {
public:
    bool initGL() {
        const char* vsSrc =
            "#version 330 core\n"
            "layout(location=0) in vec2 aPos;\n"
            "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); gl_PointSize = 16.0; }\n";
        const char* fsSrc =
            "#version 330 core\n"
            "uniform vec3 uColor;\n"
            "out vec4 FragColor;\n"
            "void main(){\n"
            "  vec2 d = gl_PointCoord - vec2(0.5);\n"
            "  float r2 = dot(d, d);\n"
            "  if (r2 > 0.25) discard;\n"
            "  float ring = smoothstep(0.18, 0.25, r2);\n"
            "  vec3 col = mix(uColor, vec3(0.0), ring*0.7);\n"
            "  FragColor = vec4(col, 1.0);\n"
            "}\n";

        GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
        if (!vs) return false;
        GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
        if (!fs) { glDeleteShader(vs); return false; }

        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);
        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint linked = 0;
        glGetProgramiv(program_, GL_LINK_STATUS, &linked);
        if (!linked) {
            char log[1024] = {0};
            glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
            std::cerr << "[MaskPicker] link error:\n" << log << std::endl;
            glDeleteProgram(program_);
            program_ = 0;
            return false;
        }
        locColor_ = glGetUniformLocation(program_, "uColor");

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);

        ready_ = (program_ != 0 && vao_ != 0 && vbo_ != 0);
        std::cout << "[MaskPicker] GL initialized (program=" << program_
                  << ", VAO=" << vao_ << ")" << std::endl;
        return ready_;
    }

    void draw(const AppContext& ctx) {
        if (!ready_) return;
        if (ctx.maskPoints.empty() && ctx.instrumentMaskPoints.empty()) return;
        if (ctx.image.width <= 0 || ctx.image.height <= 0) return;

        // Four buckets so we can issue 4 colored draws (FG/BG x Liver/Instr).
        // Slightly more memory but keeps the GL setup simple (no per-vertex
        // attributes); the lists are rarely > a dozen points each.
        std::vector<float> liverFg, liverBg, instFg, instBg;
        liverFg.reserve(ctx.maskPoints.size() * 2);
        liverBg.reserve(ctx.maskPoints.size() * 2);
        instFg.reserve(ctx.instrumentMaskPoints.size() * 2);
        instBg.reserve(ctx.instrumentMaskPoints.size() * 2);

        // Compute the image -> NDC mapping once. In MaskSelection mode the
        // image is letterboxed inside the window; in other modes it fills
        // the framebuffer 1:1.
        int viewW = ctx.windowW, viewH = ctx.windowH;
        int viewX = 0, viewY = 0;
        if (ctx.mode == AppMode::kMaskSelection) {
            float imgAspect = (float)ctx.image.width / (float)ctx.image.height;
            float winAspect = (float)ctx.windowW   / (float)ctx.windowH;
            if (imgAspect > winAspect) {
                viewH = ctx.windowW / imgAspect;
                viewY = (ctx.windowH - viewH) / 2;
            } else {
                viewW = ctx.windowH * imgAspect;
                viewX = (ctx.windowW - viewW) / 2;
            }
        }

        auto pushPoint = [&](float u, float v, bool fg, MaskKind k) {
            float ndcX, ndcY;
            if (ctx.mode == AppMode::kMaskSelection) {
                float screenX = viewX + (u / ctx.image.width)  * viewW;
                float screenY = viewY + (v / ctx.image.height) * viewH;
                ndcX = 2.0f * screenX / ctx.windowW - 1.0f;
                ndcY = 1.0f - 2.0f * screenY / ctx.windowH;
            } else {
                ndcX = 2.0f * u / (float)ctx.image.width  - 1.0f;
                ndcY = 1.0f - 2.0f * v / (float)ctx.image.height;
            }
            std::vector<float>* dst = nullptr;
            if (k == MaskKind::Liver)   dst = fg ? &liverFg : &liverBg;
            else                        dst = fg ? &instFg  : &instBg;
            dst->push_back(ndcX);
            dst->push_back(ndcY);
        };
        for (const auto& p : ctx.maskPoints)
            pushPoint(p.u, p.v, p.fg, MaskKind::Liver);
        for (const auto& p : ctx.instrumentMaskPoints)
            pushPoint(p.u, p.v, p.fg, MaskKind::Instrument);

        GLboolean prevDepth;
        glGetBooleanv(GL_DEPTH_TEST, &prevDepth);
        if (prevDepth) glDisable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glUseProgram(program_);
        glBindVertexArray(vao_);

        auto upload = [&](const std::vector<float>& v, float r, float g, float b) {
            if (v.empty()) return;
            glBindBuffer(GL_ARRAY_BUFFER, vbo_);
            glBufferData(GL_ARRAY_BUFFER,
                         (GLsizeiptr)(v.size() * sizeof(float)),
                         v.data(), GL_STREAM_DRAW);
            glUniform3f(locColor_, r, g, b);
            glDrawArrays(GL_POINTS, 0, (GLsizei)(v.size() / 2));
        };
        // Color choices (kept in sync with the legend / button hints in the
        // ImGui overlay):
        //   Liver       FG = bright green   BG = bright red
        //   Instrument  FG = cyan           BG = orange
        // These four hues are well separated in HSV and remain readable
        // over both the dark periphery and the dark-red liver tissue
        // typical of laparoscopic frames.
        upload(liverFg, 0.20f, 1.00f, 0.30f);
        upload(liverBg, 1.00f, 0.25f, 0.25f);
        upload(instFg,  0.20f, 0.85f, 1.00f);
        upload(instBg,  1.00f, 0.70f, 0.20f);

        glBindVertexArray(0);
        glUseProgram(0);
        if (prevDepth) glEnable(GL_DEPTH_TEST);
    }

    ~Renderer() {
        if (program_) glDeleteProgram(program_);
        if (vbo_)     glDeleteBuffers(1, &vbo_);
        if (vao_)     glDeleteVertexArrays(1, &vao_);
    }

private:
    static GLuint compile(GLenum type, const char* src) {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024] = {0};
            glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            std::cerr << "[MaskPicker] shader compile error:\n" << log << std::endl;
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    GLuint program_  = 0;
    GLuint vao_      = 0;
    GLuint vbo_      = 0;
    GLint  locColor_ = -1;
    bool   ready_    = false;
};

}
