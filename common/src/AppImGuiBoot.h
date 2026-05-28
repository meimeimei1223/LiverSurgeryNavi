// AppImGuiBoot.h
//
// Centralizes the ImGui context lifecycle so the same init/shutdown/frame
// pump can be used from both REG and DEFORM main.cpp. The body mirrors the
// inline initialization currently sitting in registration/main.cpp's
// initOpenGL(): create context + nav flag, dark style with project-specific
// tweaks, font fallback chain, GLFW + OpenGL3 backend install.
//
// Phase 1 only DEFORM main.cpp adopts this; REG's existing inline code stays
// put. Phase 3 (the eventual single-binary merge) is when REG migrates over.
#pragma once

#include <cstdio>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

namespace AppImGuiBoot {

inline void loadFont() {
    ImGuiIO& io = ImGui::GetIO();
    const float fontSize = 18.0f;
    bool fontLoaded = false;
    const char* fontPaths[] = {
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        nullptr
    };
    for (int i = 0; fontPaths[i]; i++) {
        FILE* f = fopen(fontPaths[i], "rb");
        if (f) {
            fclose(f);
            io.Fonts->AddFontFromFileTTF(fontPaths[i], fontSize);
            fontLoaded = true;
            printf("[ImGui] Font loaded: %s (%.0fpx)\n", fontPaths[i], fontSize);
            break;
        }
    }
    if (!fontLoaded) {
        ImFontConfig cfg;
        cfg.SizePixels = fontSize;
        io.Fonts->AddFontDefault(&cfg);
        printf("[ImGui] Using default font (%.0fpx)\n", fontSize);
    }
}

inline void init(GLFWwindow* win) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 0.0f;
    style.FrameRounding     = 4.0f;
    style.GrabRounding      = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.075f, 0.094f, 1.0f);

    loadFont();

    ImGui_ImplGlfw_InitForOpenGL(win, true);  // true = install chained callbacks
    ImGui_ImplOpenGL3_Init("#version 330");
}

inline void shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

inline void beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

inline void endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace AppImGuiBoot
